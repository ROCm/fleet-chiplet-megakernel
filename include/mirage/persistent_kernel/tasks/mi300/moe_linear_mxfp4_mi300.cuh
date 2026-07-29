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

// MoE linear kernel for MI300/MI350 with native MXFP4 weights.
// Uses scalar GEMV with LUT-based FP4 dequantization and wave reduction.
// Based on croc's mxfp4_dequant_gemv_kernel approach.
//
// Weight format: MXFP4 (FP4 E2M1 + E8M0 block scales, 32 elements per block)
//   Per workgroup: [data: OPW * K/2 bytes][scales: OPW * K/32 bytes]
//   where OPW = OUTPUT_PER_WG (typically 16)

#pragma once
#include "tasks/common/common_header.cuh"
#include <hip/hip_bf16.h>

namespace kernel {

// FP4 E2M1 dequantization lookup table
// Nibble values 0-15 map to: {0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5,
// -2, -3, -4, -6}
__device__ __constant__ float c_mxfp4_lut[16] = {0.0f,
                                                 0.5f,
                                                 1.0f,
                                                 1.5f,
                                                 2.0f,
                                                 3.0f,
                                                 4.0f,
                                                 6.0f,
                                                 -0.0f,
                                                 -0.5f,
                                                 -1.0f,
                                                 -1.5f,
                                                 -2.0f,
                                                 -3.0f,
                                                 -4.0f,
                                                 -6.0f};

// MXFP4 MoE linear GEMV kernel.
// For decode (BS=1 or small), performs per-expert GEMV with MXFP4 weights.
//
// Template parameters:
//   BATCH_SIZE     - number of tokens
//   OUTPUT_SIZE    - output dimension per expert
//   OUTPUT_STRIDE  - stride between topk slots in output
//   REDUCTION_SIZE - inner dimension (K, must be multiple of 32)
//   NUM_EXPERTS    - total number of experts (128)
//   NUM_TOPK       - experts per token (4)
//   EXPERT_STRIDE  - stride for expert iteration (grid_dim.x)
//   OUTPUT_PER_WG  - output rows per MXFP4 workgroup (16)
//   W13_LINEAR     - true for W13 (2D input), false for W2 (3D input)
template <int BATCH_SIZE,
          int OUTPUT_SIZE,
          int OUTPUT_STRIDE,
          int REDUCTION_SIZE,
          int NUM_EXPERTS,
          int NUM_TOPK,
          int EXPERT_STRIDE,
          int OUTPUT_PER_WG,
          bool W13_LINEAR>
__device__ __forceinline__ void
    moe_linear_mxfp4_kernel_mi300(void const *input_ptr,
                                  void const *weight_ptr,
                                  void const *routing_ptr,
                                  void const *mask_ptr,
                                  void const *bias_ptr,
                                  void *output_ptr,
                                  int expert_offset) {
  using bf16 = __hip_bfloat16;
  constexpr int WAVE_SIZE = 64;
  constexpr int ROWS_PER_WG = 4; // 4 waves per workgroup, 1 row per wave

  // MXFP4 layout constants
  constexpr int NUM_BLOCKS = REDUCTION_SIZE / 32; // blocks of 32 elements
  constexpr int WG_DATA_BYTES = OUTPUT_PER_WG * (REDUCTION_SIZE / 2);
  constexpr int WG_SCALE_BYTES = OUTPUT_PER_WG * NUM_BLOCKS;
  constexpr int WG_BYTES = WG_DATA_BYTES + WG_SCALE_BYTES;
  // OUTPUT_SIZE is per-tile (= OUTPUT_PER_WG). OUTPUT_STRIDE is the full output
  // dim.
  constexpr int EXPERT_WGS = OUTPUT_STRIDE / OUTPUT_PER_WG;
  constexpr int EXPERT_BYTES = EXPERT_WGS * WG_BYTES;

  bf16 const *__restrict__ d_input = static_cast<bf16 const *>(input_ptr);
  uint8_t const *__restrict__ d_weight =
      static_cast<uint8_t const *>(weight_ptr);
  int const *__restrict__ d_routing = static_cast<int const *>(routing_ptr);
  int const *__restrict__ d_mask = static_cast<int const *>(mask_ptr);
  bf16 const *__restrict__ d_bias = static_cast<bf16 const *>(bias_ptr);
  bf16 *__restrict__ d_output = static_cast<bf16 *>(output_ptr);

  // Shared memory for input activation (convert bf16 → float32)
  // Use dynamic shared memory to avoid bloating static LDS allocation
  extern __shared__ char _mxfp4_smem[];
  float *s_input = reinterpret_cast<float *>(_mxfp4_smem);

  int const tid = threadIdx.x;
  int const wave_id = tid / WAVE_SIZE;
  int const lane = tid % WAVE_SIZE;

  // Last element of mask stores the total number of activated experts
  int const num_activated_experts = d_mask[NUM_EXPERTS];

  // Unpack actual_expert_offset from packed metadata (lower 16 bits)
  int const actual_expert_offset = expert_offset & 0xFFFF;

  // Each tile covers OUTPUT_PER_WG output rows (one workgroup).
  // Within a tile, 4 waves handle 4 rows each, iterating to cover all OPW rows.
  constexpr int ROWS_PER_ITER =
      ROWS_PER_WG; // 4 rows per iteration (1 per wave)
  constexpr int ITERS_PER_TILE = OUTPUT_PER_WG / ROWS_PER_ITER; // 16/4 = 4

// Iterate over activated experts with stride
#pragma unroll 1
  for (int ae_idx = actual_expert_offset; ae_idx < num_activated_experts;
       ae_idx += EXPERT_STRIDE) {

    int32_t expert_id = d_mask[ae_idx];
    int const *expert_routing = d_routing + expert_id * BATCH_SIZE;

    // Expert weight base pointer
    uint8_t const *expert_weight =
        d_weight + static_cast<int64_t>(expert_id) * EXPERT_BYTES;

    // Process each token
    for (int tok = 0; tok < BATCH_SIZE; tok++) {
      int route_val = expert_routing[tok];
      if (route_val == 0) {
        continue;
      }
      int topk_slot = route_val - 1;

      // Load input to shared memory
      bf16 const *input_base;
      if constexpr (W13_LINEAR) {
        input_base = d_input + tok * REDUCTION_SIZE;
      } else {
        input_base = d_input + tok * (NUM_TOPK * REDUCTION_SIZE) +
                     topk_slot * REDUCTION_SIZE;
      }

      for (int i = tid; i < REDUCTION_SIZE; i += blockDim.x) {
        s_input[i] = __bfloat162float(input_base[i]);
      }
      __syncthreads();

      // Process one workgroup (OUTPUT_PER_WG rows).
      // The framework already offsets d_weight and d_output by the tile
      // position (n_tile_idx), so we access data at offset 0 within
      // expert_weight and write output at local row index (not global).
      {
        uint8_t const *wg_data = expert_weight;
        uint8_t const *wg_scales = expert_weight + WG_DATA_BYTES;
        // Each iteration: 4 waves handle 4 rows; loop to cover all OPW rows
        for (int iter = 0; iter < ITERS_PER_TILE; iter++) {
          int lr = iter * ROWS_PER_ITER + wave_id; // local row within WG

          if (lr < OUTPUT_PER_WG) {
            float sum = 0.0f;

            // Iterate over blocks (32 elements each)
            for (int blk = lane; blk < NUM_BLOCKS; blk += WAVE_SIZE) {
              int data_off = lr * (REDUCTION_SIZE / 2) + blk * 16;
              float scale =
                  exp2f((float)wg_scales[lr * NUM_BLOCKS + blk] - 127.0f);

              // Each block: 16 bytes = 32 FP4 values
              for (int j = 0; j < 16; j++) {
                uint8_t packed = wg_data[data_off + j];
                sum += c_mxfp4_lut[packed & 0xF] * scale *
                           s_input[blk * 32 + j * 2] +
                       c_mxfp4_lut[packed >> 4] * scale *
                           s_input[blk * 32 + j * 2 + 1];
              }
            }

            // Wave-level reduction
            for (int offset = WAVE_SIZE / 2; offset > 0; offset >>= 1) {
              sum += __shfl_xor(sum, offset, WAVE_SIZE);
            }

            // Add per-expert bias (d_bias is offset to this tile's position)
            // Bias tensor: [E, expert_wgs, OPW] tiled (-1, 1, -1)
            // Each tile gets [E, 1, OPW] with stride[0] = expert_wgs * OPW =
            // OUTPUT_STRIDE So expert_id indexes at stride OUTPUT_STRIDE, lr
            // indexes within OPW
            if (lane == 0) {
              float bias_val =
                  __bfloat162float(d_bias[expert_id * OUTPUT_STRIDE + lr]);
              int out_idx = tok * (NUM_TOPK * OUTPUT_STRIDE) +
                            topk_slot * OUTPUT_STRIDE + lr;
              d_output[out_idx] = __float2bfloat16(sum + bias_val);
#ifdef EMBED_DEBUG
              if (lr == 0 && !W13_LINEAR) {
                printf("[MXFP4_W2] expert=%d slot=%d lr=%d sum=%.4f bias=%.4f "
                       "out=%.4f out_idx=%d\n",
                       expert_id,
                       topk_slot,
                       lr,
                       sum,
                       bias_val,
                       sum + bias_val,
                       out_idx);
                // Print first few input values and weight data for debugging
                printf("[MXFP4_W2_DBG] input[0..3]=%.4f %.4f %.4f %.4f\n",
                       s_input[0],
                       s_input[1],
                       s_input[2],
                       s_input[3]);
                printf("[MXFP4_W2_DBG] wg_data[0..3]=%d %d %d %d scale0=%d "
                       "EXPERT_BYTES=%d WG_BYTES=%d\n",
                       (int)wg_data[0],
                       (int)wg_data[1],
                       (int)wg_data[2],
                       (int)wg_data[3],
                       (int)wg_scales[0],
                       (int)EXPERT_BYTES,
                       (int)WG_BYTES);
              }
              if (lr == 0 && W13_LINEAR) {
                printf("[MXFP4_W13] expert=%d slot=%d lr=%d sum=%.4f bias=%.4f "
                       "out=%.4f out_idx=%d\n",
                       expert_id,
                       topk_slot,
                       lr,
                       sum,
                       bias_val,
                       sum + bias_val,
                       out_idx);
              }
#endif
            }
          }
        }
      }

      __syncthreads(); // Ensure shared memory is free for next token
    }
  }
}

} // namespace kernel
