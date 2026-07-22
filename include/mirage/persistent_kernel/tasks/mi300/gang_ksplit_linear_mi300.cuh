/* Cross-XCD K-Split Linear (SKXCCM-style)
 *
 * Splits K dimension ACROSS 8 XCDs. Each XCD computes partial GEMM
 * for K/8 of the reduction over ALL output tiles, then atomicAdd
 * partial results to a float32 workspace. A separate finalize task
 * (dispatched after the event fires) adds residual and converts to bf16.
 *
 * Key advantage: ALL 30 workers per XCD are active (processing all N-tiles)
 * vs N-split which has only 8 tiles/XCD for O_proj.
 *
 * Low merge contention: 8 writers from 8 different XCDs per output element.
 * Each writer's atomicAdd goes through HBM (GPU-scope), well-pipelined.
 */
#pragma once
#include "linear_ck_mi300.cuh"

namespace kernel {

using bfloat16 = type::bfloat16_t;

// Phase 1: Partial GEMM + cross-XCD atomic merge
template <typename T,
          int BATCH_SIZE,
          int REDUCTION_SIZE,
          int K_SPLITS>
__device__ __forceinline__ void gang_ksplit_gemm_kernel(
    void const *input_ptr,       // [batch, REDUCTION_SIZE] full
    void const *weight_ptr,      // [N_total, REDUCTION_SIZE] full
    void *workspace_ptr,         // [batch, N_total] float32 atomic target
    int num_active_tokens,
    int tile_n,
    int ws_stride,               // workspace stride (= N_total typically)
    int n_tiles,                 // total N-tiles
    int k_split_idx,             // 0..K_SPLITS-1, from bid.x
    int tile_idx)                // N-tile index
{
  using namespace ck_tile;

  if (tile_idx >= n_tiles) return;

  constexpr int K_per_split = REDUCTION_SIZE / K_SPLITS;
  int k_offset = k_split_idx * K_per_split;

  constexpr index_t MPerBlock = 16;
  constexpr index_t NPerBlock = 64;
  constexpr index_t KPerBlock = 256;
  constexpr index_t NumLoopK = K_per_split / KPerBlock;
  constexpr index_t LoopM = (BATCH_SIZE + MPerBlock - 1) / MPerBlock;

  using BlockTile = sequence<MPerBlock, NPerBlock, KPerBlock>;
  using BlockWarps = sequence<1, 4>;
  using WarpTile = sequence<16, 16, KPerBlock>;
  using GemmShape = TileGemmShape<BlockTile, BlockWarps, WarpTile>;
  using GemmTraits = TileGemmUniversalTraits<
      true, false, true, false,
      tensor_layout::gemm::RowMajor,
      tensor_layout::gemm::ColumnMajor,
      tensor_layout::gemm::RowMajor>;
  using Problem = GemmPipelineProblem<bf16, bf16, float, GemmShape, GemmTraits>;
  using PipelinePolicy = GemmPipelineSmallTilePolicy<MPerBlock, NPerBlock, KPerBlock>;
  using Pipeline = GemmPipelineAGmemBGmemCRegV2<Problem, PipelinePolicy>;

  const bf16* d_input = reinterpret_cast<const bf16*>(
      __uniform_addr(static_cast<T const*>(input_ptr)));
  const bf16* d_weight = reinterpret_cast<const bf16*>(
      __uniform_addr(static_cast<T const*>(weight_ptr) +
                     static_cast<size_t>(tile_idx) * tile_n * REDUCTION_SIZE));
  float* d_ws = reinterpret_cast<float*>(__uniform_addr(workspace_ptr));
  int ws_stride_u = __builtin_amdgcn_readfirstlane(ws_stride);

  extern __shared__ char smem[];

  for (index_t mm = 0; mm < LoopM; mm++) {
    index_t m_offset = mm * MPerBlock;
    index_t m_size = (mm == LoopM - 1) ? (BATCH_SIZE - m_offset) : MPerBlock;
    if (m_size <= 0) continue;

    auto a_view = make_naive_tensor_view<address_space_enum::global>(
        d_input + m_offset * REDUCTION_SIZE + k_offset,
        make_tuple(m_size, index_t(K_per_split)),
        make_tuple(index_t(REDUCTION_SIZE), index_t(1)),
        number<8>{}, number<1>{});

#ifdef MPK_NT_WEIGHT_LOADS
    auto b_view = make_naive_tensor_view<
        address_space_enum::global,
        memory_operation_enum::set,
        static_cast<amd_buffer_coherence_enum>(18)>(
#else
    auto b_view = make_naive_tensor_view<address_space_enum::global>(
#endif
        d_weight + k_offset,
        make_tuple(index_t(tile_n), index_t(K_per_split)),
        make_tuple(index_t(REDUCTION_SIZE), index_t(1)),
        number<8>{}, number<1>{});

    auto a_win = make_tile_window(a_view,
        make_tuple(number<MPerBlock>{}, number<KPerBlock>{}), {0, 0});
    auto b_win = make_tile_window(b_view,
        make_tuple(number<NPerBlock>{}, number<KPerBlock>{}), {0, 0});

    Pipeline pipeline;
    auto c_tile = pipeline(a_win, b_win, NumLoopK, smem);
    block_sync_lds();

    // GPU-scope atomicAdd to workspace
    index_t warp_id = threadIdx.x >> 6;
    index_t lane_id = threadIdx.x & 63;
    index_t tile_row = lane_id & 15;
    index_t tile_col_base = warp_id * 16 + ((lane_id >> 4) << 2);
    auto& c_buf = c_tile.get_thread_buffer();

    index_t global_m = m_offset + tile_row;
    index_t global_n = tile_idx * tile_n + tile_col_base;

    if (global_m < BATCH_SIZE && tile_col_base + 3 < NPerBlock) {
      index_t base = global_m * ws_stride_u + global_n;
      atomicAdd(&d_ws[base],     c_buf[0]);
      atomicAdd(&d_ws[base + 1], c_buf[1]);
      atomicAdd(&d_ws[base + 2], c_buf[2]);
      atomicAdd(&d_ws[base + 3], c_buf[3]);
    } else if (global_m < BATCH_SIZE) {
      #pragma unroll
      for (index_t i = 0; i < 4; i++) {
        if (tile_col_base + i < NPerBlock)
          atomicAdd(&d_ws[global_m * ws_stride_u + global_n + i], c_buf[i]);
      }
    }
  }
}

// Phase 2: Finalize — add residual, convert bf16, zero workspace.
// Dispatched as a separate gang task AFTER the K-split GEMM event fires.
// Each XCD handles its N-partition of the output (standard N-split).
template <typename T, int BATCH_SIZE>
__device__ __forceinline__ void gang_ksplit_finalize_kernel(
    void *workspace_ptr,         // [batch, ws_stride] float32 — full workspace
    void const *residual_ptr,    // [batch, o_stride] — XCD's N-partition
    void *output_ptr,            // [batch, o_stride] — XCD's N-partition
    int ws_stride,               // workspace stride (= full N)
    int o_stride,                // output stride (= full N)
    int n_cols,                  // columns this XCD finalizes
    int n_col_offset,            // starting column for this XCD
    int tile_idx)                // distributes work across workers
{
  float* d_ws = reinterpret_cast<float*>(__uniform_addr(workspace_ptr));
  const bfloat16* res = reinterpret_cast<const bfloat16*>(
      __uniform_addr(static_cast<T const*>(residual_ptr)));
  bfloat16* out = reinterpret_cast<bfloat16*>(
      __uniform_addr(static_cast<T*>(output_ptr)));
  int ws_stride_u = __builtin_amdgcn_readfirstlane(ws_stride);
  int o_stride_u = __builtin_amdgcn_readfirstlane(o_stride);
  int n_col_offset_u = __builtin_amdgcn_readfirstlane(n_col_offset);

  // Each tile_idx handles a chunk of output elements
  constexpr int ELEMS_PER_TILE = 512;  // ~2KB per tile
  int start = tile_idx * ELEMS_PER_TILE;
  int total = BATCH_SIZE * n_cols;

  for (int idx = start + (int)threadIdx.x; idx < min(start + ELEMS_PER_TILE, total);
       idx += (int)blockDim.x) {
    int row = idx / n_cols;
    int col = idx % n_cols;
    int ws_idx = row * ws_stride_u + n_col_offset_u + col;
    int out_idx = row * o_stride_u + col;

    float val = d_ws[ws_idx];
    val += ck_tile::type_convert<float>(res[out_idx]);
    out[out_idx] = ck_tile::type_convert<bfloat16>(val);
    d_ws[ws_idx] = 0.0f;
  }
}

} // namespace kernel
