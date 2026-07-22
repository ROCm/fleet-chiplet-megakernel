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

// Gang MoE linear kernel for MI300/MI350.
//
// Processes one expert at a time per XCD. All ~30 workers on an XCD cooperate
// to tile one expert's GEMM, then move to the next expert. This eliminates
// L2 thrashing caused by concurrent expert weight loads on the same XCD.
//
// Dispatch: 8 gang tasks (1 per XCD), each with flattened tile_idx encoding:
//   tile_idx = expert_local_idx * TILES_PER_EXPERT + tile_within_expert
//   ae_idx = xcd_id + expert_local_idx * 8  (round-robin XCD assignment)
//   tile_within_expert → (m_tile, n_tile)
//
// Workers stride through [0, n_tile_count) via the gang dispatch loop,
// naturally staying in sync on the same expert since tiles_per_expert > workers_per_xcd.

#pragma once

namespace kernel {

// Local XCD ID helper (defined here to avoid include-order dependency
// on persistent_kernel.cuh where the global get_current_xcd_id lives).
__device__ __forceinline__ int _gang_moe_get_xcd_id() {
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
  int xcd_id;
  asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcd_id));
  return xcd_id;
#else
  return 0;
#endif
}

// Gang MoE W13 linear: gate+up fused GEMM with expert routing.
// Input: [batch, K=hidden_size]
// Weight: [num_experts, N=2*intermediate, K=hidden_size]
// Output: [batch, topk, N=2*intermediate]
template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int OUTPUT_STRIDE,
          int REDUCTION_SIZE,
          int NUM_EXPERTS,
          int NUM_TOPK,
          int TILES_PER_EXPERT,
          int N_TILES>
__device__ __noinline__ void
    gang_moe_w13_linear_kernel(void const *input_ptr,
                                void const *weight_ptr,
                                void const *routing_ptr,
                                void const *mask_ptr,
                                void *output_ptr,
                                int tile_idx) {
  using namespace ck_tile;

  // Tile sizes: 16x64x256 (small tile for bs <= 16)
  constexpr index_t MPerBlock = 16;
  constexpr index_t NPerBlock = 64;
  constexpr index_t KPerBlock = 256;
  constexpr index_t NumLoopK = REDUCTION_SIZE / KPerBlock;
  static_assert(REDUCTION_SIZE % KPerBlock == 0,
                "W13 REDUCTION_SIZE must be divisible by KPerBlock=256");

  constexpr index_t MWarp = 1;
  constexpr index_t NWarp = 4;

  using BlockTile = sequence<MPerBlock, NPerBlock, KPerBlock>;
  using BlockWarps = sequence<MWarp, NWarp>;
  using WarpTile = sequence<MPerBlock, NPerBlock / NWarp, KPerBlock>;
  using GemmShape = TileGemmShape<BlockTile, BlockWarps, WarpTile>;

  using GemmTraits = TileGemmUniversalTraits<
      true, false, true, false,
      tensor_layout::gemm::RowMajor,
      tensor_layout::gemm::ColumnMajor,
      tensor_layout::gemm::RowMajor>;

  using Problem = GemmPipelineProblem<bf16, bf16, float, GemmShape, GemmTraits>;
  using PipelinePolicy = GemmPipelineSmallTilePolicy<MPerBlock, NPerBlock, KPerBlock>;
  using Pipeline = GemmPipelineAGmemBGmemCRegV2<Problem, PipelinePolicy>;

  // Get XCD ID from hardware register
  int xcd_id = _gang_moe_get_xcd_id();

  int const *__restrict__ d_mask = static_cast<int const *>(mask_ptr);
  int const num_activated_experts = d_mask[NUM_EXPERTS];

  // Decode flat tile_idx into expert and tile position
  int expert_local_idx = tile_idx / TILES_PER_EXPERT;
  int tile_within_expert = tile_idx % TILES_PER_EXPERT;

  // Round-robin XCD assignment: XCD i gets activated experts i, i+8, i+16, ...
  int ae_idx = xcd_id + expert_local_idx * 8;
  if (ae_idx >= num_activated_experts) return;

  int expert_id = d_mask[ae_idx];

  // Decode tile position
  constexpr index_t M_TILES = (BATCH_SIZE + MPerBlock - 1) / MPerBlock;
  int m_tile = tile_within_expert / N_TILES;
  int n_tile = tile_within_expert % N_TILES;
  if (m_tile >= M_TILES) return;  // out of bounds for M dimension

  bf16 const *__restrict__ d_input = static_cast<bf16 const *>(input_ptr);
  bf16 const *__restrict__ d_weight = static_cast<bf16 const *>(weight_ptr);
  bf16 *__restrict__ d_output = static_cast<bf16 *>(output_ptr);
  int const *__restrict__ d_routing = static_cast<int const *>(routing_ptr);

  // Input A: offset to M-tile
  index_t m_offset = m_tile * MPerBlock;
  bf16 const *a_base = d_input + static_cast<size_t>(m_offset) * REDUCTION_SIZE;

  // Weight B: expert's weight slice, offset to N-tile
  bf16 const *expert_weight = d_weight +
      static_cast<int64_t>(expert_id) * OUTPUT_STRIDE * REDUCTION_SIZE;
  bf16 const *b_base = expert_weight +
      static_cast<size_t>(n_tile) * NPerBlock * REDUCTION_SIZE;

  // Routing indices for this expert
  int const *expert_routing = d_routing + expert_id * BATCH_SIZE;

  index_t m_size = (m_offset + MPerBlock <= BATCH_SIZE)
                       ? MPerBlock
                       : (BATCH_SIZE - m_offset);
  index_t n_offset = n_tile * NPerBlock;
  index_t n_size = (n_offset + NPerBlock <= OUTPUT_SIZE)
                       ? NPerBlock
                       : (OUTPUT_SIZE - n_offset);

  extern __shared__ char smem[];

  // Create CK Tile tensor views
  auto a_tensor_view = make_naive_tensor_view<address_space_enum::global>(
      a_base,
      make_tuple(m_size, index_t(REDUCTION_SIZE)),
      make_tuple(index_t(REDUCTION_SIZE), index_t(1)),
      number<8>{},
      number<1>{});

#ifdef MPK_NT_WEIGHT_LOADS
  auto b_tensor_view = make_naive_tensor_view<
      address_space_enum::global,
      memory_operation_enum::set,
      static_cast<amd_buffer_coherence_enum>(18)>(
#else
  auto b_tensor_view = make_naive_tensor_view<address_space_enum::global>(
#endif
      b_base,
      make_tuple(n_size, index_t(REDUCTION_SIZE)),
      make_tuple(index_t(REDUCTION_SIZE), index_t(1)),
      number<8>{},
      number<1>{});

  auto a_tile_window = make_tile_window(
      a_tensor_view,
      make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
      {0, 0});

  auto b_tile_window = make_tile_window(
      b_tensor_view,
      make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
      {0, 0});

  // Run CK pipeline GEMM
  Pipeline pipeline;
  auto c_block_tile = pipeline(a_tile_window, b_tile_window,
                               NumLoopK, smem);
  block_sync_lds();

  // ---- Epilogue: scatter-write results using routing indices ----
  // (same as moe_linear_mi300.cuh 16x64 tile epilogue)
  auto &c_buf = c_block_tile.get_thread_buffer();

  index_t warp_id = threadIdx.x >> 6;
  index_t lane_id = threadIdx.x & 63;
  index_t tile_row = lane_id & 15;
  index_t tile_col_base = warp_id * 16 + ((lane_id >> 4) << 2);
  index_t global_m = m_offset + tile_row;
  index_t global_n_base = n_offset + tile_col_base;

  if (global_m < BATCH_SIZE) {
    int const route_val = expert_routing[global_m];
    if (route_val != 0) {
      int const topk_slot = route_val - 1;

      if (global_n_base + 3 < OUTPUT_SIZE) {
        bf16 *out_addr = d_output +
            global_m * (NUM_TOPK * OUTPUT_STRIDE) +
            topk_slot * OUTPUT_STRIDE +
            global_n_base;

        uint64_t out_packed;
        bf16 *out = reinterpret_cast<bf16 *>(&out_packed);
        out[0] = type_convert<bf16>(c_buf[0]);
        out[1] = type_convert<bf16>(c_buf[1]);
        out[2] = type_convert<bf16>(c_buf[2]);
        out[3] = type_convert<bf16>(c_buf[3]);
        *reinterpret_cast<uint64_t *>(out_addr) = out_packed;
      } else if (global_m < BATCH_SIZE) {
        #pragma unroll
        for (index_t i = 0; i < 4; i++) {
          index_t global_n = global_n_base + i;
          if (global_n < OUTPUT_SIZE) {
            d_output[global_m * (NUM_TOPK * OUTPUT_STRIDE) +
                     topk_slot * OUTPUT_STRIDE +
                     global_n] =
                type_convert<bf16>(c_buf[i]);
          }
        }
      }
    }
  }
}

// Gang MoE W2 linear: down projection with expert routing.
// Input: [batch, topk, K=intermediate]
// Weight: [num_experts, N=hidden_size, K=intermediate]
// Output: [batch, topk, N=hidden_size]
//
// W2 processes tokens one at a time because each token has a different
// topk_slot, so input row pointers differ per token.
// KPerBlock=128 since K=1408 is not divisible by 256.
template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int OUTPUT_STRIDE,
          int REDUCTION_SIZE,
          int NUM_EXPERTS,
          int NUM_TOPK,
          int TILES_PER_EXPERT,
          int N_TILES>
__device__ __noinline__ void
    gang_moe_w2_linear_kernel(void const *input_ptr,
                               void const *weight_ptr,
                               void const *routing_ptr,
                               void const *mask_ptr,
                               void *output_ptr,
                               int tile_idx) {
  using namespace ck_tile;

  // W2 uses KPerBlock=128 since K (e.g. 1408) may not divide 256
  constexpr index_t MPerBlock = 16;
  constexpr index_t NPerBlock = 64;
  constexpr index_t KPerBlock = (REDUCTION_SIZE % 256 == 0) ? 256 : 128;
  constexpr index_t NumLoopK = REDUCTION_SIZE / KPerBlock;
  static_assert(REDUCTION_SIZE % KPerBlock == 0,
                "W2 REDUCTION_SIZE must be divisible by KPerBlock");

  constexpr index_t MWarp = 1;
  constexpr index_t NWarp = 4;

  using BlockTile = sequence<MPerBlock, NPerBlock, KPerBlock>;
  using BlockWarps = sequence<MWarp, NWarp>;
  using WarpTile = sequence<MPerBlock, NPerBlock / NWarp, KPerBlock>;
  using GemmShape = TileGemmShape<BlockTile, BlockWarps, WarpTile>;

  using GemmTraits = TileGemmUniversalTraits<
      true, false, true, false,
      tensor_layout::gemm::RowMajor,
      tensor_layout::gemm::ColumnMajor,
      tensor_layout::gemm::RowMajor>;

  using Problem = GemmPipelineProblem<bf16, bf16, float, GemmShape, GemmTraits>;
  using PipelinePolicy = GemmPipelineSmallTilePolicy<MPerBlock, NPerBlock, KPerBlock>;
  using Pipeline = GemmPipelineAGmemBGmemCRegV2<Problem, PipelinePolicy>;

  int xcd_id = _gang_moe_get_xcd_id();

  int const *__restrict__ d_mask = static_cast<int const *>(mask_ptr);
  int const num_activated_experts = d_mask[NUM_EXPERTS];

  // Decode flat tile_idx
  // For W2, tile space = N_TILES * BATCH_SIZE (one N-tile per token)
  // tile_within_expert encodes (token_idx, n_tile)
  int expert_local_idx = tile_idx / TILES_PER_EXPERT;
  int tile_within_expert = tile_idx % TILES_PER_EXPERT;

  int ae_idx = xcd_id + expert_local_idx * 8;
  if (ae_idx >= num_activated_experts) return;

  int expert_id = d_mask[ae_idx];

  // For W2: tiles_per_expert = N_TILES * BATCH_SIZE
  // Decode: token_idx and n_tile
  int w2_tok = tile_within_expert / N_TILES;
  int n_tile = tile_within_expert % N_TILES;
  if (w2_tok >= BATCH_SIZE) return;

  bf16 const *__restrict__ d_input = static_cast<bf16 const *>(input_ptr);
  bf16 const *__restrict__ d_weight = static_cast<bf16 const *>(weight_ptr);
  bf16 *__restrict__ d_output = static_cast<bf16 *>(output_ptr);
  int const *__restrict__ d_routing = static_cast<int const *>(routing_ptr);

  // Routing indices for this expert
  int const *expert_routing = d_routing + expert_id * BATCH_SIZE;

  // Check routing — skip if this token doesn't use this expert
  int const route_val = expert_routing[w2_tok];
  if (route_val == 0) return;
  int const topk_slot = route_val - 1;

  // Input A: input[w2_tok, topk_slot, :] — one token at a time
  bf16 const *a_base = d_input +
      static_cast<size_t>(w2_tok) * (NUM_TOPK * REDUCTION_SIZE) +
      static_cast<size_t>(topk_slot) * REDUCTION_SIZE;

  // Weight B: expert's weight slice, offset to N-tile
  bf16 const *expert_weight = d_weight +
      static_cast<int64_t>(expert_id) * OUTPUT_STRIDE * REDUCTION_SIZE;
  index_t n_offset = n_tile * NPerBlock;
  bf16 const *b_base = expert_weight +
      static_cast<size_t>(n_offset) * REDUCTION_SIZE;

  index_t n_size = (n_offset + NPerBlock <= OUTPUT_SIZE)
                       ? NPerBlock
                       : (OUTPUT_SIZE - n_offset);

  extern __shared__ char smem[];

  // M=1 GEMM for this token
  auto a_tensor_view = make_naive_tensor_view<address_space_enum::global>(
      a_base,
      make_tuple(index_t(1), index_t(REDUCTION_SIZE)),
      make_tuple(index_t(REDUCTION_SIZE), index_t(1)),
      number<8>{},
      number<1>{});

#ifdef MPK_NT_WEIGHT_LOADS
  auto b_tensor_view = make_naive_tensor_view<
      address_space_enum::global,
      memory_operation_enum::set,
      static_cast<amd_buffer_coherence_enum>(18)>(
#else
  auto b_tensor_view = make_naive_tensor_view<address_space_enum::global>(
#endif
      b_base,
      make_tuple(n_size, index_t(REDUCTION_SIZE)),
      make_tuple(index_t(REDUCTION_SIZE), index_t(1)),
      number<8>{},
      number<1>{});

  auto a_tile_window = make_tile_window(
      a_tensor_view,
      make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
      {0, 0});

  auto b_tile_window = make_tile_window(
      b_tensor_view,
      make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
      {0, 0});

  Pipeline pipeline;
  auto c_block_tile = pipeline(a_tile_window, b_tile_window,
                               NumLoopK, smem);
  block_sync_lds();

  // ---- Epilogue: write result for this token ----
  auto &c_buf = c_block_tile.get_thread_buffer();

  index_t warp_id = threadIdx.x >> 6;
  index_t lane_id = threadIdx.x & 63;
  index_t tile_row = lane_id & 15;
  index_t tile_col_base = warp_id * 16 + ((lane_id >> 4) << 2);
  index_t global_n_base = n_offset + tile_col_base;

  // Only row 0 has valid data (M=1 GEMM)
  if (tile_row == 0) {
    if (global_n_base + 3 < OUTPUT_SIZE) {
      bf16 *out_addr = d_output +
          static_cast<size_t>(w2_tok) * (NUM_TOPK * OUTPUT_STRIDE) +
          static_cast<size_t>(topk_slot) * OUTPUT_STRIDE +
          global_n_base;

      uint64_t out_packed;
      bf16 *out = reinterpret_cast<bf16 *>(&out_packed);
      out[0] = type_convert<bf16>(c_buf[0]);
      out[1] = type_convert<bf16>(c_buf[1]);
      out[2] = type_convert<bf16>(c_buf[2]);
      out[3] = type_convert<bf16>(c_buf[3]);
      *reinterpret_cast<uint64_t *>(out_addr) = out_packed;
    } else {
      #pragma unroll
      for (index_t i = 0; i < 4; i++) {
        index_t global_n = global_n_base + i;
        if (global_n < OUTPUT_SIZE) {
          d_output[static_cast<size_t>(w2_tok) * (NUM_TOPK * OUTPUT_STRIDE) +
                   static_cast<size_t>(topk_slot) * OUTPUT_STRIDE +
                   global_n] =
              type_convert<bf16>(c_buf[i]);
        }
      }
    }
  }
}

} // namespace kernel
