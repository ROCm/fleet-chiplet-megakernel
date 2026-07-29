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

// MoE linear kernel for MI300/MI350 using CK Tile GEMM infrastructure.
// Reuses GemmPipelineSmallTilePolicy from linear_ck_mi300.cuh with
// expert routing via moe_mask and moe_routing_indices.

#pragma once

// Note: linear_ck_mi300.cuh is included before this file in task_header.cuh,
// providing GemmPipelineSmallTilePolicy, BlockGemmSmallM16Policy, etc.

namespace kernel {

// MoE linear kernel using CK Tile GEMM.
// Iterates over activated experts and performs GEMM with expert routing.
//
// For W13 (gate+up fused): input [batch, reduction] -> output [batch, topk,
// 2*intermediate] For W2 (down project):   input [batch, topk, intermediate] ->
// output [batch, topk, hidden]
//
// Template parameters:
//   T              - data type (bf16)
//   BATCH_SIZE     - number of tokens in the batch
//   OUTPUT_SIZE    - output dimension per expert (may be padded)
//   OUTPUT_STRIDE  - stride between consecutive output rows
//   REDUCTION_SIZE - inner dimension (K)
//   NUM_EXPERTS    - total number of experts
//   NUM_TOPK       - number of experts selected per token
//   EXPERT_STRIDE  - number of thread blocks sharing the expert workload
//   W13_LINEAR     - true for W13 (2D input), false for W2 (3D input)
template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int OUTPUT_STRIDE,
          int REDUCTION_SIZE,
          int NUM_EXPERTS,
          int NUM_TOPK,
          int EXPERT_STRIDE,
          bool W13_LINEAR>
__device__ __forceinline__ void moe_linear_kernel_mi300(void const *input_ptr,
                                                        void const *weight_ptr,
                                                        void const *routing_ptr,
                                                        void const *mask_ptr,
                                                        void const *bias_ptr,
                                                        void *output_ptr,
                                                        int expert_offset) {
  using namespace ck_tile;

  // Select tile sizes based on batch size (same heuristic as linear_ck_mi300)
  constexpr bool use_large_tile = (BATCH_SIZE > 16);
  constexpr index_t MPerBlock = use_large_tile ? 32 : 16;
  constexpr index_t NPerBlock = use_large_tile ? 128 : 64;
  constexpr index_t KPerBlock = use_large_tile ? 128 : 256;

  constexpr index_t LoopM = (BATCH_SIZE + MPerBlock - 1) / MPerBlock;
  constexpr index_t LoopN = (OUTPUT_SIZE + NPerBlock - 1) / NPerBlock;
  constexpr index_t NumLoopK = REDUCTION_SIZE / KPerBlock;

  constexpr index_t MWarp = 1;
  constexpr index_t NWarp = 4;

  using BlockTile = sequence<MPerBlock, NPerBlock, KPerBlock>;
  using BlockWarps = sequence<MWarp, NWarp>;
  using WarpTile = sequence<MPerBlock, NPerBlock / NWarp, KPerBlock>;
  using GemmShape = TileGemmShape<BlockTile, BlockWarps, WarpTile>;

  using GemmTraits = TileGemmUniversalTraits<true,
                                             false,
                                             true,
                                             false,
                                             tensor_layout::gemm::RowMajor,
                                             tensor_layout::gemm::ColumnMajor,
                                             tensor_layout::gemm::RowMajor>;

  using Problem = GemmPipelineProblem<bf16, bf16, float, GemmShape, GemmTraits>;
  using PipelinePolicy =
      GemmPipelineSmallTilePolicy<MPerBlock, NPerBlock, KPerBlock>;
  using Pipeline = GemmPipelineAGmemBGmemCRegV2<Problem, PipelinePolicy>;

  bf16 const *__restrict__ d_input = static_cast<bf16 const *>(input_ptr);
  bf16 const *__restrict__ d_weight = static_cast<bf16 const *>(weight_ptr);
  bf16 *__restrict__ d_output = static_cast<bf16 *>(output_ptr);
  int const *__restrict__ d_routing = static_cast<int const *>(routing_ptr);
  int const *__restrict__ d_mask = static_cast<int const *>(mask_ptr);
  bf16 const *__restrict__ d_bias = static_cast<bf16 const *>(bias_ptr);

  extern __shared__ char smem[];

  // Last element of mask stores the total number of activated experts
  int const num_activated_experts = d_mask[NUM_EXPERTS];

  // Unpack expert_offset and n_tile_idx from packed metadata:
  //   lower 16 bits = expert_offset, upper 16 bits = n_tile_idx
  int const actual_expert_offset = expert_offset & 0xFFFF;
  int const n_tile_idx = (expert_offset >> 16) & 0xFFFF;

  // Iterate over activated experts with stride
#pragma unroll 1
  for (int ae_idx = actual_expert_offset; ae_idx < num_activated_experts;
       ae_idx += EXPERT_STRIDE) {
    // Get the actual expert ID from the mask (dense list of active expert IDs)
    int32_t expert_id = d_mask[ae_idx];

    // Expert's weight slice: weight[expert_id, :, :]
    bf16 const *expert_weight = d_weight + static_cast<int64_t>(expert_id) *
                                               OUTPUT_STRIDE * REDUCTION_SIZE;

    // Routing indices for this expert: routing[expert_id, :]
    int const *expert_routing = d_routing + expert_id * BATCH_SIZE;

    // For W2 with BATCH_SIZE>1, each token may have a different topk_slot.
    // We can't represent per-row offsets in a single CK Tile tensor view,
    // so W2 processes one token at a time.
    constexpr index_t W2_LoopM_Outer = W13_LINEAR ? 1 : BATCH_SIZE;
    constexpr index_t W2_BatchPerIter = W13_LINEAR ? BATCH_SIZE : 1;
    constexpr index_t EffectiveLoopM =
        W13_LINEAR ? LoopM : ((W2_BatchPerIter + MPerBlock - 1) / MPerBlock);

    // Loop over tokens individually for W2, or all at once for W13
    for (index_t w2_tok = 0; w2_tok < W2_LoopM_Outer; ++w2_tok) {

      // Loop over M-tiles (batch dimension)
      for (index_t m_iter = 0; m_iter < EffectiveLoopM; ++m_iter) {
        index_t const m_local = m_iter * MPerBlock;
        index_t const m_offset = W13_LINEAR ? m_local : w2_tok;
        index_t const m_size = W13_LINEAR ? ((m_local + MPerBlock <= BATCH_SIZE)
                                                 ? MPerBlock
                                                 : (BATCH_SIZE - m_local))
                                          : 1;

        // Each task handles one N-tile (indexed by n_tile_idx from bid.y)
        // instead of looping over all N-tiles (eliminates redundant compute).
        if (n_tile_idx < LoopN) {
          index_t n_iter = n_tile_idx;
          index_t const n_offset = n_iter * NPerBlock;
          index_t const n_size = (n_offset + NPerBlock <= OUTPUT_SIZE)
                                     ? NPerBlock
                                     : (OUTPUT_SIZE - n_offset);

          // Input A pointer for this M-tile
          bf16 const *a_base;
          index_t a_row_stride;
          if constexpr (W13_LINEAR) {
            // W13: input is [batch, reduction], row-major
            a_base = d_input + m_offset * REDUCTION_SIZE;
            a_row_stride = REDUCTION_SIZE;
          } else {
            // W2: input is [batch, topk, reduction], one token at a time
            int w2_topk_slot = 0;
            int route_val = expert_routing[w2_tok];
            if (route_val > 0) {
              w2_topk_slot = route_val - 1;
            }
            a_base = d_input + w2_tok * (NUM_TOPK * REDUCTION_SIZE) +
                     w2_topk_slot * REDUCTION_SIZE;
            a_row_stride = REDUCTION_SIZE;
          }

          // Weight B pointer for this N-tile
          bf16 const *b_base = expert_weight + n_offset * REDUCTION_SIZE;

          // Create CK Tile tensor views
          auto a_tensor_view =
              make_naive_tensor_view<address_space_enum::global>(
                  a_base,
                  make_tuple(m_size, index_t(REDUCTION_SIZE)),
                  make_tuple(a_row_stride, index_t(1)),
                  number<8>{},
                  number<1>{});

          auto b_tensor_view =
              make_naive_tensor_view<address_space_enum::global>(
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
          auto c_block_tile =
              pipeline(a_tile_window, b_tile_window, NumLoopK, smem);
          block_sync_lds();

          // ---- Epilogue: scatter-write results using routing indices ----
          auto &c_buf = c_block_tile.get_thread_buffer();

          index_t warp_id = threadIdx.x >> 6;
          index_t lane_id = threadIdx.x & 63;

          if constexpr (use_large_tile) {
            // 32x128 tiles: 32x32 MFMA TransposedCDistribution
            index_t tile_row = lane_id & 31;
            index_t m_lane = lane_id >> 5;
            index_t global_m = m_offset + tile_row;

            if (global_m < BATCH_SIZE) {
              int const route_val = expert_routing[global_m];
              if (route_val != 0) {
                int const topk_slot = route_val - 1;
#pragma unroll
                for (index_t g = 0; g < 4; g++) {
                  index_t col_in_warp = g * 8 + m_lane * 4;
                  index_t global_n_base = n_offset + warp_id * 32 + col_in_warp;
                  index_t r_base = g * 4;

                  if (global_n_base + 3 < OUTPUT_SIZE) {
                    bf16 *out_addr = d_output +
                                     global_m * (NUM_TOPK * OUTPUT_STRIDE) +
                                     topk_slot * OUTPUT_STRIDE + global_n_base;
                    bf16 const *bias_addr =
                        d_bias + expert_id * OUTPUT_STRIDE + global_n_base;

                    uint64_t bias_packed =
                        *reinterpret_cast<uint64_t const *>(bias_addr);
                    bf16 const *bv =
                        reinterpret_cast<bf16 const *>(&bias_packed);
                    uint64_t out_packed;
                    bf16 *out = reinterpret_cast<bf16 *>(&out_packed);
                    out[0] = type_convert<bf16>(c_buf[r_base] +
                                                type_convert<float>(bv[0]));
                    out[1] = type_convert<bf16>(c_buf[r_base + 1] +
                                                type_convert<float>(bv[1]));
                    out[2] = type_convert<bf16>(c_buf[r_base + 2] +
                                                type_convert<float>(bv[2]));
                    out[3] = type_convert<bf16>(c_buf[r_base + 3] +
                                                type_convert<float>(bv[3]));
                    *reinterpret_cast<uint64_t *>(out_addr) = out_packed;
                  } else {
                    for (index_t i = 0; i < 4; i++) {
                      index_t global_n = global_n_base + i;
                      if (global_n < OUTPUT_SIZE) {
                        float bval = type_convert<float>(
                            d_bias[expert_id * OUTPUT_STRIDE + global_n]);
                        d_output[global_m * (NUM_TOPK * OUTPUT_STRIDE) +
                                 topk_slot * OUTPUT_STRIDE + global_n] =
                            type_convert<bf16>(c_buf[r_base + i] + bval);
                      }
                    }
                  }
                }
              }
            }
          } else {
            // 16x64 tiles: 16x16 MFMA TransposedCDistribution
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
                                   topk_slot * OUTPUT_STRIDE + global_n_base;
                  bf16 const *bias_addr =
                      d_bias + expert_id * OUTPUT_STRIDE + global_n_base;

                  uint64_t bias_packed =
                      *reinterpret_cast<uint64_t const *>(bias_addr);
                  bf16 const *bv = reinterpret_cast<bf16 const *>(&bias_packed);
                  uint64_t out_packed;
                  bf16 *out = reinterpret_cast<bf16 *>(&out_packed);
                  out[0] =
                      type_convert<bf16>(c_buf[0] + type_convert<float>(bv[0]));
                  out[1] =
                      type_convert<bf16>(c_buf[1] + type_convert<float>(bv[1]));
                  out[2] =
                      type_convert<bf16>(c_buf[2] + type_convert<float>(bv[2]));
                  out[3] =
                      type_convert<bf16>(c_buf[3] + type_convert<float>(bv[3]));
                  *reinterpret_cast<uint64_t *>(out_addr) = out_packed;
                } else {
#pragma unroll
                  for (index_t i = 0; i < 4; i++) {
                    index_t global_n = global_n_base + i;
                    if (global_n < OUTPUT_SIZE) {
                      float bval = type_convert<float>(
                          d_bias[expert_id * OUTPUT_STRIDE + global_n]);
                      d_output[global_m * (NUM_TOPK * OUTPUT_STRIDE) +
                               topk_slot * OUTPUT_STRIDE + global_n] =
                          type_convert<bf16>(c_buf[i] + bval);
                    }
                  }
                }
              }
            }
          }
          __syncthreads();
        } // n_iter
      }   // m_iter
    }     // w2_tok (W2 per-token loop, trivial for W13)
  }       // activated expert loop
}

} // namespace kernel
