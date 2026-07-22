/* Gang Split-K Linear with Residual: splits the K dimension within a gang task
 * to utilize more workers per XCD.
 *
 * Without splitk: O_proj has 8 N-tiles → 8 of 30 workers active (27% util)
 * With splitk K=4: 8 N-tiles × 4 K-splits = 32 tiles → 30 workers active (94%)
 *
 * Key advantage over regular splitk: all workers are on the same XCD,
 * so the atomic merge uses XCD-local L2 atomics (~100 cycles) instead of
 * GPU-scope atomics (~400+ cycles with buffer_wbl2 fence).
 */
#pragma once
#include "linear_ck_mi300.cuh"

namespace kernel {

using bfloat16 = type::bfloat16_t;

// Gang split-K linear with residual.
// tile_idx maps to (n_tile, k_split) within the XCD's partition.
// Each worker computes a partial GEMM over K/k_splits of the reduction dim,
// atomically merges into float32 workspace. Last k_split adds residual.
template <typename T,
          int BATCH_SIZE,        // = m_per_tile
          int REDUCTION_SIZE,    // full K dimension
          int K_SPLITS>          // number of K splits
__device__ __forceinline__ void gang_splitk_linear_res_kernel(
    void const *input_ptr,       // [full_batch, REDUCTION_SIZE]
    void const *weight_ptr,      // [chunk_N, REDUCTION_SIZE] - XCD's weight chunk
    void const *residual_ptr,    // [full_batch, o_stride] - XCD's residual columns
    void *workspace_ptr,         // [full_batch, chunk_N] float32 workspace
    void *output_ptr,            // [full_batch, o_stride] - XCD's output columns
    int *done_counter_ptr,       // [n_tiles] per-tile done counters
    int num_active_tokens,
    int tile_n,                  // N columns per tile (64)
    int o_stride,                // Full output stride
    int n_tiles,                 // N-tiles per XCD
    int tile_idx)                // encodes (n_tile, k_split)
{
  using namespace ck_tile;

  // Decode tile_idx → (n_tile, k_split)
  int n_tile = tile_idx / K_SPLITS;
  int k_split = tile_idx % K_SPLITS;

  if (n_tile >= n_tiles) return;

  constexpr int K_per_split = REDUCTION_SIZE / K_SPLITS;
  int k_offset = k_split * K_per_split;

  // CK GEMM setup — same tile config as regular splitk
  constexpr index_t MPerBlock = 16;
  constexpr index_t NPerBlock = 64;  // = tile_n
  constexpr index_t KPerBlock = 256;
  constexpr index_t NumLoopK = K_per_split / KPerBlock;
  constexpr index_t LoopM = (BATCH_SIZE + MPerBlock - 1) / MPerBlock;

  using BlockTile = sequence<MPerBlock, NPerBlock, KPerBlock>;
  using BlockWarps = sequence<1, 4>;  // MWarp=1, NWarp=4
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

  const bf16* d_input  = reinterpret_cast<const bf16*>(
      __uniform_addr(static_cast<T const*>(input_ptr)));
  const bf16* d_weight = reinterpret_cast<const bf16*>(
      __uniform_addr(static_cast<T const*>(weight_ptr) +
                     static_cast<size_t>(n_tile) * tile_n * REDUCTION_SIZE));
  float* d_ws = reinterpret_cast<float*>(__uniform_addr(workspace_ptr)) +
      static_cast<size_t>(n_tile) * tile_n;  // workspace partitioned by n_tile
  int ws_stride = __builtin_amdgcn_readfirstlane(n_tiles * tile_n);
  int o_stride_u = __builtin_amdgcn_readfirstlane(o_stride);

  extern __shared__ char smem[];

  // Phase 1: Partial GEMM over [K_per_split] + atomicAdd to workspace
  for (index_t mm = 0; mm < LoopM; mm++) {
    index_t m_offset = mm * MPerBlock;
    index_t m_size = (mm == LoopM - 1) ? (BATCH_SIZE - m_offset) : MPerBlock;
    if (m_size <= 0) continue;

    // Input: offset to m_tile row AND k_split column
    auto a_view = make_naive_tensor_view<address_space_enum::global>(
        d_input + m_offset * REDUCTION_SIZE + k_offset,
        make_tuple(m_size, index_t(K_per_split)),
        make_tuple(index_t(REDUCTION_SIZE), index_t(1)),
        number<8>{}, number<1>{});

    // Weight: offset to k_split column (n_tile already in d_weight base)
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

    // Atomic epilogue: accumulate partial results to workspace
    // XCD-local atomics — all workers on same XCD, L2 coherent
    index_t warp_id = threadIdx.x >> 6;
    index_t lane_id = threadIdx.x & 63;
    index_t tile_row = lane_id & 15;
    index_t tile_col_base = warp_id * 16 + ((lane_id >> 4) << 2);
    auto& c_buf = c_tile.get_thread_buffer();

    index_t global_m = m_offset + tile_row;
    if (global_m < BATCH_SIZE && tile_col_base + 3 < NPerBlock) {
      index_t base = global_m * ws_stride + tile_col_base;
      atomicAdd(&d_ws[base],     c_buf[0]);
      atomicAdd(&d_ws[base + 1], c_buf[1]);
      atomicAdd(&d_ws[base + 2], c_buf[2]);
      atomicAdd(&d_ws[base + 3], c_buf[3]);
    } else if (global_m < BATCH_SIZE) {
      #pragma unroll
      for (index_t i = 0; i < 4; i++) {
        if (tile_col_base + i < NPerBlock) {
          atomicAdd(&d_ws[global_m * ws_stride + tile_col_base + i], c_buf[i]);
        }
      }
    }
  }

  // Phase 2: XCD-local done counter (no GPU-scope fence needed!)
  // All workers on same XCD share L2 — atomics are naturally coherent.
  int* d_done = done_counter_ptr + n_tile;
  __shared__ int is_last;
  if (threadIdx.x == 0) {
    // XCD-local fence: ensure workspace writes visible to other workers on XCD
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    int old = atomicAdd(d_done, 1);
    is_last = (old == K_SPLITS - 1) ? 1 : 0;
  }
  __syncthreads();

  // Phase 3: Last split — add residual, convert to bf16, zero workspace
  if (is_last) {
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");

    const bf16* res = residual_ptr
        ? reinterpret_cast<const bf16*>(__uniform_addr(
              static_cast<T const*>(residual_ptr) +
              static_cast<size_t>(n_tile) * tile_n))
        : nullptr;
    bf16* out = reinterpret_cast<bf16*>(__uniform_addr(
        static_cast<T*>(output_ptr) +
        static_cast<size_t>(n_tile) * tile_n));

    constexpr int TOTAL = BATCH_SIZE * NPerBlock;
    for (int idx = threadIdx.x; idx < TOTAL; idx += 256) {
      int row = idx / NPerBlock;
      int col = idx % NPerBlock;
      index_t ws_idx = row * ws_stride + col;
      index_t out_idx = row * o_stride_u + col;

      float val = d_ws[ws_idx];
      if (res) val += type_convert<float>(res[out_idx]);
      out[out_idx] = type_convert<bf16>(val);
      d_ws[ws_idx] = 0.0f;
    }
    if (threadIdx.x == 0) {
      *d_done = 0;
    }
  }
}

} // namespace kernel
