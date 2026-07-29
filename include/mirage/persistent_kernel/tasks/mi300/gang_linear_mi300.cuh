/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "linear_ck_mi300.cuh"

namespace kernel {

using bfloat16 = type::bfloat16_t;

// Gang linear with HipKittens Algorithm 1 windowed traversal.
//
// Workers are mapped to 2D tiles: (m_tile, n_tile) from a 1D tile_idx.
// BATCH_SIZE template param = m_per_tile (rows per M-tile), not full batch.
//
// HipKittens Algorithm 1 (Step 2 — windowed traversal):
//   Window height W controls L2 cache reuse. Within a window of W M-rows,
//   consecutive tile indices sweep M first (fast index), then advance N (slow
//   index). This keeps the same weight tile in L2 across W activations.
//
//   When W >= m_tiles, this reduces to simple M-major ordering.
//   When W < m_tiles, tiles are grouped into horizontal bands of W rows,
//   and each band processes all N-columns before moving to the next band.
//
// XCD grouping (Algorithm 1, Step 1) is implicit: the scheduler dispatches
// each gang task to a single XCD, so all workers on that XCD share L2.
//
// Pointer offsets:
//   input:  m_tile * BATCH_SIZE * REDUCTION_SIZE  (M-tile of activation)
//   weight: n_tile * tile_n * REDUCTION_SIZE      (N-tile of weight)
//   output: m_tile * BATCH_SIZE * o_stride + n_tile * tile_n
template <typename T,
          int BATCH_SIZE, // = m_per_tile (rows per worker)
          int REDUCTION_SIZE>
__device__ __forceinline__ void gang_linear_kernel(
    void const *input_ptr,  // [full_batch, REDUCTION_SIZE]
    void const *weight_ptr, // [chunk_N, REDUCTION_SIZE] - XCD's weight chunk
    void *output_ptr,       // [full_batch, o_stride] - XCD's output columns
    int num_active_tokens,
    int tile_n,   // N columns per worker
    int o_stride, // Full output stride
    int m_tiles,  // Total M-tiles
    int n_tiles,  // Total N-tiles per XCD
    int wgm, // Window height W (Algorithm 1): 0 or >= m_tiles = full M-major
    int tile_idx, // 1D tile index: encodes both M and N position
    void const *bias_ptr = nullptr) // [1, full_N] bf16, optional
{
  assert(tile_idx >= 0);

  // HipKittens Algorithm 1, Step 2: windowed traversal
  // When wgm <= 0 or wgm >= m_tiles, use full M-major (W = m_tiles)
  int W = (wgm > 0 && wgm < m_tiles) ? wgm : m_tiles;

  int tid_per_group = W * n_tiles;         // tiles in one window
  int group_id = tile_idx / tid_per_group; // which window of M-rows
  int first_row = group_id * W;
  int win_h = m_tiles - first_row; // remaining rows
  if (win_h > W) {
    win_h = W; // clamp to window height
  }
  int local = tile_idx % tid_per_group; // position within window
  // Handle tail group (fewer rows than W)
  if (local >= win_h * n_tiles) {
    return; // out-of-bounds tile in tail group — should not happen with correct
            // tile count
  }
  int m_tile = first_row + (local % win_h); // fast index: sweep M within column
  int n_tile = local / win_h; // slow index: advance N after win_h rows

  // Input: offset to M-tile (skip m_tile * BATCH_SIZE rows)
  T const *tile_input =
      static_cast<T const *>(input_ptr) +
      static_cast<size_t>(m_tile) * BATCH_SIZE * REDUCTION_SIZE;

  // Weight: offset to N-tile (skip n_tile * tile_n rows of weight)
  T const *tile_weight = static_cast<T const *>(weight_ptr) +
                         static_cast<size_t>(n_tile) * tile_n * REDUCTION_SIZE;

  // Output: offset to (m_tile, n_tile) corner
  T *tile_output = static_cast<T *>(output_ptr) +
                   static_cast<size_t>(m_tile) * BATCH_SIZE * o_stride +
                   static_cast<size_t>(n_tile) * tile_n;

  // Bias: offset to N-tile's column range (bias is 1D, broadcast across batch)
  void const *tile_bias = bias_ptr ? static_cast<T const *>(bias_ptr) +
                                         static_cast<size_t>(n_tile) * tile_n
                                   : nullptr;

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  unsigned long long _t0 = __builtin_amdgcn_s_memrealtime();
#endif
  linear_kernel_ck<T, BATCH_SIZE, REDUCTION_SIZE>(tile_input,
                                                  tile_weight,
                                                  nullptr,
                                                  tile_output,
                                                  num_active_tokens,
                                                  false,
                                                  tile_n,
                                                  o_stride,
                                                  tile_bias);
#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  __syncthreads();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    unsigned long long _dur = (__builtin_amdgcn_s_memrealtime() - _t0) * 10;
    printf("[GANG_LINEAR] ostride=%d tile=%d dur_us=%.1f\n",
           o_stride,
           tile_idx,
           (double)_dur / 1000.0);
  }
#endif
}

// Gang linear with residual + HipKittens Algorithm 1 windowed traversal.
template <typename T,
          int BATCH_SIZE, // = m_per_tile
          int REDUCTION_SIZE>
__device__ __forceinline__ void gang_linear_residual_kernel(
    void const *input_ptr,
    void const *weight_ptr,
    void const *residual_ptr,
    void *output_ptr,
    int num_active_tokens,
    int tile_n,
    int o_stride,
    int m_tiles,
    int n_tiles, // Total N-tiles per XCD
    int wgm,     // Window height W (Algorithm 1)
    int tile_idx,
    void const *bias_ptr = nullptr) // [1, full_N] bf16, optional
{
  assert(tile_idx >= 0);

  // HipKittens Algorithm 1, Step 2: windowed traversal
  int W = (wgm > 0 && wgm < m_tiles) ? wgm : m_tiles;

  int tid_per_group = W * n_tiles;
  int group_id = tile_idx / tid_per_group;
  int first_row = group_id * W;
  int win_h = m_tiles - first_row;
  if (win_h > W) {
    win_h = W;
  }
  int local = tile_idx % tid_per_group;
  if (local >= win_h * n_tiles) {
    return;
  }
  int m_tile = first_row + (local % win_h);
  int n_tile = local / win_h;

  T const *tile_input =
      static_cast<T const *>(input_ptr) +
      static_cast<size_t>(m_tile) * BATCH_SIZE * REDUCTION_SIZE;

  T const *tile_weight = static_cast<T const *>(weight_ptr) +
                         static_cast<size_t>(n_tile) * tile_n * REDUCTION_SIZE;

  T const *tile_residual = static_cast<T const *>(residual_ptr) +
                           static_cast<size_t>(m_tile) * BATCH_SIZE * o_stride +
                           static_cast<size_t>(n_tile) * tile_n;

  T *tile_output = static_cast<T *>(output_ptr) +
                   static_cast<size_t>(m_tile) * BATCH_SIZE * o_stride +
                   static_cast<size_t>(n_tile) * tile_n;

  // Bias: offset to N-tile's column range
  void const *tile_bias = bias_ptr ? static_cast<T const *>(bias_ptr) +
                                         static_cast<size_t>(n_tile) * tile_n
                                   : nullptr;

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  unsigned long long _t0 = __builtin_amdgcn_s_memrealtime();
#endif
  linear_kernel_ck<T, BATCH_SIZE, REDUCTION_SIZE>(tile_input,
                                                  tile_weight,
                                                  tile_residual,
                                                  tile_output,
                                                  num_active_tokens,
                                                  true,
                                                  tile_n,
                                                  o_stride,
                                                  tile_bias);
#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  __syncthreads();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    unsigned long long _dur = (__builtin_amdgcn_s_memrealtime() - _t0) * 10;
    printf("[GANG_LINEAR_RES] ostride=%d tile=%d dur_us=%.1f\n",
           o_stride,
           tile_idx,
           (double)_dur / 1000.0);
  }
#endif
}

// Gang linear with SiLU fusion: gate_up GEMM + SiLU+mul → [bs, inter_size]
//
// Weight layout (from shuffle_tensors with num_groups=G):
//   Per XCD chunk: [gate_0(128 rows), up_0(128 rows), gate_1(128 rows),
//   up_1(128 rows), ...] With tile_n=64, each 128-row block = 2 tiles. For
//   output column pair out_n:
//     gate_weight_tile = (out_n/2)*4 + (out_n%2)
//     up_weight_tile   = (out_n/2)*4 + 2 + (out_n%2)
//
// For each tile_idx, this kernel:
//   1. CK GEMM gate tile → SiLU(result) → LDS scratch
//   2. CK GEMM up tile → multiply by SiLU from LDS → write bf16 output
//
// Output is [bs, inter_size] (half of gate_up_size), eliminating mlp_mid
// buffer.
//
// Tile selection:
//   BATCH_SIZE >= 32: 64×64×128 tile (32×32×16 MFMA, MWarp=2 NWarp=2)
//                     Processes full batch in one pass, 2.4x faster at bs=64
//   BATCH_SIZE < 32:  16×64×256 tile (16×16×16 MFMA, MWarp=1 NWarp=4)
//                     M-loop for BATCH_SIZE > 16
template <typename T,
          int BATCH_SIZE, // = m_per_tile
          int REDUCTION_SIZE>
__device__ __noinline__ void gang_linear_silu_kernel(
    void const *input_ptr,  // [full_batch, REDUCTION_SIZE]
    void const *weight_ptr, // [chunk_N_gateup, REDUCTION_SIZE] - XCD's
                            // interleaved gate+up weight
    void *output_ptr,       // [full_batch, o_stride] - XCD's output columns
                            // (inter_size/8 wide)
    int num_active_tokens,
    int tile_n,   // N columns per tile (64)
    int o_stride, // Output stride (inter_size = gate_up_size/2)
    int m_tiles,  // Total M-tiles
    int n_tiles,  // N-tiles per XCD in output space (chunk_n_out / tile_n)
    int wgm,      // Window height W (Algorithm 1)
    int tile_idx) {
  using namespace ck_tile;
  assert(tile_idx >= 0);

  // Windowed traversal (same as gang_linear_kernel)
  int W = (wgm > 0 && wgm < m_tiles) ? wgm : m_tiles;
  int tid_per_group = W * n_tiles;
  int group_id = tile_idx / tid_per_group;
  int first_row = group_id * W;
  int win_h = m_tiles - first_row;
  if (win_h > W) {
    win_h = W;
  }
  int local = tile_idx % tid_per_group;
  if (local >= win_h * n_tiles) {
    return;
  }
  int m_tile = first_row + (local % win_h);
  int out_n_tile = local / win_h;

  // Map output n_tile to gate/up weight tile indices in interleaved layout
  int grp = out_n_tile / 2;
  int sub = out_n_tile % 2;
  int gate_weight_tile = grp * 4 + sub;
  int up_weight_tile = grp * 4 + 2 + sub;

  // Tile selection: large batch → 64×64×128 (32×32 MFMA, 2×2 warps)
  //                 small batch → 16×64×256 (16×16 MFMA, 1×4 warps)
  constexpr bool use_large_tile = (BATCH_SIZE >= 32);
  constexpr index_t MPerBlock = use_large_tile ? 64 : 16;
  constexpr index_t NPerBlock = 64;
  constexpr index_t KPerBlock = use_large_tile ? 128 : 256;
  constexpr index_t NumLoopK = REDUCTION_SIZE / KPerBlock;
  constexpr index_t LoopM = (BATCH_SIZE + MPerBlock - 1) / MPerBlock;

  using BlockTile = sequence<MPerBlock, NPerBlock, KPerBlock>;
  using BlockWarps =
      std::conditional_t<use_large_tile, sequence<2, 2>, sequence<1, 4>>;
  constexpr index_t WarpMN = use_large_tile ? 32 : 16;
  using WarpTile = sequence<WarpMN, WarpMN, 16>;
  using GemmShape = TileGemmShape<BlockTile, BlockWarps, WarpTile>;
  using GemmTraits = TileGemmUniversalTraits<true,
                                             false,
                                             true,
                                             false,
                                             tensor_layout::gemm::RowMajor,
                                             tensor_layout::gemm::ColumnMajor,
                                             tensor_layout::gemm::RowMajor,
                                             false>;
  using Problem = GemmPipelineProblem<bf16, bf16, float, GemmShape, GemmTraits>;
  using PipelinePolicy =
      GemmPipelineSmallTilePolicy<MPerBlock, NPerBlock, KPerBlock>;
  using Pipeline = GemmPipelineAGmemBGmemCRegV2<Problem, PipelinePolicy>;

  bf16 const *d_input_base = reinterpret_cast<bf16 const *>(__uniform_addr(
      static_cast<T const *>(input_ptr) +
      static_cast<size_t>(m_tile) * BATCH_SIZE * REDUCTION_SIZE));
  bf16 const *d_gate_weight = reinterpret_cast<bf16 const *>(__uniform_addr(
      static_cast<T const *>(weight_ptr) +
      static_cast<size_t>(gate_weight_tile) * tile_n * REDUCTION_SIZE));
  bf16 const *d_up_weight = reinterpret_cast<bf16 const *>(__uniform_addr(
      static_cast<T const *>(weight_ptr) +
      static_cast<size_t>(up_weight_tile) * tile_n * REDUCTION_SIZE));
  bf16 *d_output_base = reinterpret_cast<bf16 *>(
      __uniform_addr(static_cast<T *>(output_ptr) +
                     static_cast<size_t>(m_tile) * BATCH_SIZE * o_stride +
                     static_cast<size_t>(out_n_tile) * tile_n));
  int o_stride_u = __builtin_amdgcn_readfirstlane(o_stride);

  extern __shared__ char smem[];
  // LDS scratch for SiLU results: 256 threads × N_REGS floats
  // Placed after CK pipeline's smem (which it reuses between gate and up GEMMs)
  constexpr int CK_SMEM_SIZE = PipelinePolicy::template GetSmemSize<Problem>();
  // For 64×64: 16 regs/thread. For 16×64: 4 regs/thread.
  constexpr int N_REGS = use_large_tile ? 16 : 4;
  float *silu_scratch = reinterpret_cast<float *>(smem + CK_SMEM_SIZE);

  index_t warp_id = threadIdx.x >> 6;
  index_t lane_id = threadIdx.x & 63;
  index_t silu_base = threadIdx.x * N_REGS;

  // Weight views are constant across M-iterations (weight reuse in L2)
#ifdef MPK_NT_WEIGHT_LOADS
  auto b_gate_view =
      make_naive_tensor_view<address_space_enum::global,
                             memory_operation_enum::set,
                             static_cast<amd_buffer_coherence_enum>(18)>(
#else
  auto b_gate_view = make_naive_tensor_view<address_space_enum::global>(
#endif
          d_gate_weight,
          make_tuple(number<NPerBlock>{}, index_t(REDUCTION_SIZE)),
          make_tuple(index_t(REDUCTION_SIZE), index_t(1)),
          number<8>{},
          number<1>{});
  auto b_gate_win =
      make_tile_window(b_gate_view,
                       make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                       {0, 0});

#ifdef MPK_NT_WEIGHT_LOADS
  auto b_up_view =
      make_naive_tensor_view<address_space_enum::global,
                             memory_operation_enum::set,
                             static_cast<amd_buffer_coherence_enum>(18)>(
#else
  auto b_up_view = make_naive_tensor_view<address_space_enum::global>(
#endif
          d_up_weight,
          make_tuple(number<NPerBlock>{}, index_t(REDUCTION_SIZE)),
          make_tuple(index_t(REDUCTION_SIZE), index_t(1)),
          number<8>{},
          number<1>{});
  auto b_up_win = make_tile_window(
      b_up_view, make_tuple(number<NPerBlock>{}, number<KPerBlock>{}), {0, 0});

  // M-loop: iterate over M-tiles, reusing weight data in L2
  for (index_t mm = 0; mm < LoopM; mm++) {
    index_t m_offset = mm * MPerBlock;
    bf16 const *d_input = d_input_base + m_offset * REDUCTION_SIZE;
    bf16 *d_output = d_output_base + m_offset * o_stride_u;

    // ── Step 1: Gate GEMM ──
    {
      auto a_view = make_naive_tensor_view<address_space_enum::global>(
          d_input,
          make_tuple(number<MPerBlock>{}, index_t(REDUCTION_SIZE)),
          make_tuple(index_t(REDUCTION_SIZE), index_t(1)),
          number<8>{},
          number<1>{});
      auto a_win = make_tile_window(
          a_view, make_tuple(number<MPerBlock>{}, number<KPerBlock>{}), {0, 0});

      Pipeline pipeline;
      auto c_gate = pipeline(a_win, b_gate_win, NumLoopK, smem);
      block_sync_lds();

      // Apply SiLU to gate results and store to LDS scratch
      auto &gate_buf = c_gate.get_thread_buffer();
#pragma unroll
      for (index_t i = 0; i < N_REGS; i++) {
        float g = gate_buf[i];
        silu_scratch[silu_base + i] = g / (1.0f + __expf(-g)); // SiLU
      }
    }
    __syncthreads();

    // ── Step 2: Up GEMM + multiply by SiLU ──
    {
      auto a_view = make_naive_tensor_view<address_space_enum::global>(
          d_input,
          make_tuple(number<MPerBlock>{}, index_t(REDUCTION_SIZE)),
          make_tuple(index_t(REDUCTION_SIZE), index_t(1)),
          number<8>{},
          number<1>{});
      auto a_win = make_tile_window(
          a_view, make_tuple(number<MPerBlock>{}, number<KPerBlock>{}), {0, 0});

      Pipeline pipeline;
      auto c_up = pipeline(a_win, b_up_win, NumLoopK, smem);
      block_sync_lds();

      // Multiply up result by SiLU(gate) from LDS and write to output
      auto &up_buf = c_up.get_thread_buffer();

      if constexpr (use_large_tile) {
        // 64×64 tile: 32×32 MFMA, MWarp=2 NWarp=2, 16 regs/thread
        // Row = warp_m * 32 + (lane_id % 32)
        // Col groups of 4: warp_n * 32 + (g*8 + (lane_id/32)*4)
        index_t warp_m = warp_id >> 1;
        index_t warp_n = warp_id & 1;
        index_t tile_row = lane_id & 31;
        index_t m_lane = lane_id >> 5;
        index_t global_m = warp_m * 32 + tile_row + m_offset;

        if (global_m < BATCH_SIZE) {
#pragma unroll
          for (index_t g = 0; g < 4; g++) {
            index_t col_in_warp = g * 8 + m_lane * 4;
            index_t global_n = warp_n * 32 + col_in_warp;
            index_t r_base = g * 4;

            if (global_n + 3 < NPerBlock) {
              index_t out_idx =
                  (warp_m * 32 + tile_row) * o_stride_u + global_n;
              uint64_t out_packed;
              bf16 *out = reinterpret_cast<bf16 *>(&out_packed);
#pragma unroll
              for (index_t i = 0; i < 4; i++) {
                out[i] = type_convert<bf16>(
                    silu_scratch[silu_base + r_base + i] * up_buf[r_base + i]);
              }
              nt_store_u64(&d_output[out_idx], out_packed);
            }
          }
        }
      } else {
        // 16×64 tile: 16×16 MFMA, MWarp=1 NWarp=4, 4 regs/thread
        index_t tile_row = lane_id & 15;
        index_t tile_col_base = warp_id * 16 + ((lane_id >> 4) << 2);

        if (tile_row + m_offset < BATCH_SIZE && tile_col_base + 3 < NPerBlock) {
          index_t out_idx = tile_row * o_stride_u + tile_col_base;
          uint64_t out_packed;
          bf16 *out = reinterpret_cast<bf16 *>(&out_packed);
#pragma unroll
          for (index_t i = 0; i < 4; i++) {
            out[i] =
                type_convert<bf16>(silu_scratch[silu_base + i] * up_buf[i]);
          }
          nt_store_u64(&d_output[out_idx], out_packed);
        } else if (tile_row + m_offset < BATCH_SIZE) {
#pragma unroll
          for (index_t i = 0; i < 4; i++) {
            index_t col = tile_col_base + i;
            if (col < NPerBlock) {
              float val = silu_scratch[silu_base + i] * up_buf[i];
              nt_store_bf16(&d_output[tile_row * o_stride_u + col],
                            type_convert<bf16>(val));
            }
          }
        }
      }
    }
    __syncthreads(); // Barrier before next M-iteration reuses silu_scratch
  }
}

} // namespace kernel
