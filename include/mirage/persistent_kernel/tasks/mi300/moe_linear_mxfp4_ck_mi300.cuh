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

// MFMA-based MoE linear kernel for MI300/MI350 with native MXFP4 weights.
// Uses __builtin_amdgcn_mfma_f32_16x16x16bf16_1k with 4-wave N-parallel.
// Each task handles OUTPUT_PER_WG output rows (multiple of 16).
//
// Architecture: 256 threads = 4 waves. Each wave independently computes
// 16 output rows across the full K dimension (no cross-wave reduction).
// For OPW=64: 4 waves x 16 rows = 64 rows per task.
//
// Optimizations:
// - Scale hoisting: exp2f computed once per 32-element block (every 2 MFMA
// iters)
// - Integer bf16 construction: FP4 dequant via bit manipulation instead of
// float LUT
// - K-loop unrolled in pairs to amortize scale computation
//
// Weight format: MXFP4 (FP4 E2M1 + E8M0 block scales, 32 elements per block)
//   Per workgroup: [data: OPW * K/2 bytes][scales: OPW * K/32 bytes]

#pragma once
#include "tasks/common/common_header.cuh"
#include <hip/hip_bf16.h>

namespace kernel {

typedef short mxfp4_v4s __attribute__((ext_vector_type(4)));
typedef float mxfp4_v4f __attribute__((ext_vector_type(4)));

// FP4 E2M1 -> bf16 conversion (dequant = lut[nibble] * 2^(scale - 127))
__device__ __forceinline__ unsigned short mxfp4_to_bf16(int nibble,
                                                        float scale) {
  static constexpr float lut[16] = {0.0f,
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
  float val = lut[nibble] * scale;
  unsigned t;
  __builtin_memcpy(&t, &val, 4);
  return (unsigned short)(t >> 16);
}

// MFMA-based MXFP4 MoE linear kernel (N-parallel, optimized compute).
// Each task handles OUTPUT_PER_WG output rows for one N-tile.
// 4 waves each compute 16 output rows independently across full K.
// No cross-wave LDS reduction needed.
//
// MFMA lane mapping (v_mfma_f32_16x16x16_bf16):
//   Lane l (0-63): A[l%16, (l/16)*4..+3], B[l%16, (l/16)*4..+3]
//   Result: C[m,n] = sum_k A[m,k] * B[n,k]
//   Output: acc[i] = C[(l/16)*4+i, l%16] -- 4 M rows at N column l%16
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
    moe_linear_mxfp4_ck_kernel_mi300(void const *input_ptr,
                                     void const *weight_ptr,
                                     void const *routing_ptr,
                                     void const *mask_ptr,
                                     void const *bias_ptr,
                                     void *output_ptr,
                                     int expert_offset) {
  static_assert(OUTPUT_PER_WG % 16 == 0,
                "OUTPUT_PER_WG must be multiple of 16");
  // K must be multiple of 32 (scale block size) for scale hoisting
  static_assert(REDUCTION_SIZE % 32 == 0,
                "REDUCTION_SIZE must be multiple of 32");

  // MXFP4 layout constants
  constexpr int NUM_BLOCKS = REDUCTION_SIZE / 32;
  constexpr int WG_DATA_BYTES = OUTPUT_PER_WG * (REDUCTION_SIZE / 2);
  constexpr int WG_SCALE_BYTES = OUTPUT_PER_WG * NUM_BLOCKS;
  constexpr int WG_BYTES = WG_DATA_BYTES + WG_SCALE_BYTES;
  constexpr int EXPERT_WGS = OUTPUT_STRIDE / OUTPUT_PER_WG;
  constexpr int64_t EXPERT_BYTES = static_cast<int64_t>(EXPERT_WGS) * WG_BYTES;

  // N-parallel: 4 waves, each handles 16 output rows independently
  constexpr int NUM_WAVES = 4;
  constexpr int KTILE = 16; // MFMA K dimension
  // Process K in blocks of 32 (one scale block = 2 MFMA iterations)
  constexpr int K_BLOCKS = NUM_BLOCKS; // = REDUCTION_SIZE / 32
  // Number of 16-row N-tiles per task
  constexpr int N_TILES = OUTPUT_PER_WG / 16;
  // Tiles per wave (for OPW > 64, each wave iterates over multiple tiles)
  constexpr int TILES_PER_WAVE = N_TILES / NUM_WAVES;

  unsigned short const *A = (unsigned short const *)input_ptr;
  uint8_t const *W = (uint8_t const *)weight_ptr;
  int const *d_routing = (int const *)routing_ptr;
  int const *d_mask = (int const *)mask_ptr;
  unsigned short const *d_bias = (unsigned short const *)bias_ptr;
  unsigned short *d_output = (unsigned short *)output_ptr;

  // Shared memory: only input activation buffer (no reduction buffer needed)
  extern __shared__ char _mxfp4_ck_smem[];
  unsigned short *s_input = (unsigned short *)_mxfp4_ck_smem;

  int const tid = threadIdx.x;
  int const warp_id = tid >> 6;           // 0-3 (wave ID)
  int const lane_id = tid & 63;           // 0-63
  int const m_idx = lane_id & 15;         // MFMA lane row index
  int const k_group = lane_id >> 4;       // K group: 0-3
  int const k_lane_offset = k_group << 2; // K offset within MFMA: 0, 4, 8, 12

  int const num_activated_experts = d_mask[NUM_EXPERTS];
  int const actual_expert_offset = expert_offset & 0xFFFF;

#pragma unroll 1
  for (int ae_idx = actual_expert_offset; ae_idx < num_activated_experts;
       ae_idx += EXPERT_STRIDE) {

    int32_t expert_id = d_mask[ae_idx];
    int const *expert_routing = d_routing + expert_id * BATCH_SIZE;

    // Expert weight base pointer (framework already offsets by N-tile)
    uint8_t const *expert_weight =
        W + static_cast<int64_t>(expert_id) * EXPERT_BYTES;
    uint8_t const *wg_data = expert_weight;
    uint8_t const *wg_scales = expert_weight + WG_DATA_BYTES;

    for (int tok = 0; tok < BATCH_SIZE; tok++) {
      int route_val = expert_routing[tok];
      if (route_val == 0) {
        continue;
      }
      int topk_slot = route_val - 1;

      // Load input activation to shared memory (bf16)
      unsigned short const *input_base;
      if constexpr (W13_LINEAR) {
        input_base = A + tok * REDUCTION_SIZE;
      } else {
        input_base =
            A + tok * (NUM_TOPK * REDUCTION_SIZE) + topk_slot * REDUCTION_SIZE;
      }

      for (int i = tid; i < REDUCTION_SIZE; i += blockDim.x) {
        s_input[i] = input_base[i];
      }
      __syncthreads();

      // Each wave processes TILES_PER_WAVE N-tiles of 16 rows each
      for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; tile_iter++) {
        int wave_tile = warp_id + tile_iter * NUM_WAVES;

        // Output row for this lane within the workgroup
        int b_row = wave_tile * 16 + m_idx; // [0, OUTPUT_PER_WG)

        // Weight row pointers for this output row
        uint8_t const *row_data = wg_data + b_row * (REDUCTION_SIZE / 2);
        uint8_t const *row_scales = wg_scales + b_row * NUM_BLOCKS;

        // Initialize accumulator
        mxfp4_v4f acc = {0.0f, 0.0f, 0.0f, 0.0f};

// K-loop over scale blocks (32 elements = 2 MFMA iterations each)
// Scale is computed once per block, amortized over 2 MFMA iters
#pragma unroll 1
        for (int blk = 0; blk < K_BLOCKS; blk++) {
          int k_block_start = blk * 32;

          // Compute scale ONCE for this 32-element block
          float scale = exp2f((float)row_scales[blk] - 127.0f);

// Two MFMA iterations per scale block (16 + 16 = 32 elements)
#pragma unroll
          for (int sub = 0; sub < 2; sub++) {
            int k_offset = k_block_start + sub * KTILE + k_lane_offset;

            // Load A fragment: 4 bf16 from shared memory
            mxfp4_v4s a_reg;
            if (m_idx < BATCH_SIZE) {
              unsigned const *a_src = (unsigned const *)(s_input + k_offset);
              unsigned a_lo = a_src[0];
              unsigned a_hi = a_src[1];
              a_reg[0] = (short)(a_lo & 0xFFFF);
              a_reg[1] = (short)(a_lo >> 16);
              a_reg[2] = (short)(a_hi & 0xFFFF);
              a_reg[3] = (short)(a_hi >> 16);
            } else {
              a_reg[0] = a_reg[1] = a_reg[2] = a_reg[3] = 0;
            }

            // Load B fragment: dequant 4 MXFP4 values -> 4 bf16
            // All 4 elements guaranteed to be in the same scale block
            mxfp4_v4s b_reg;
            {
              int byte_pos = k_offset / 2;
              uint8_t packed0 = row_data[byte_pos];
              uint8_t packed1 = row_data[byte_pos + 1];

              b_reg[0] = mxfp4_to_bf16(packed0 & 0xF, scale);
              b_reg[1] = mxfp4_to_bf16(packed0 >> 4, scale);
              b_reg[2] = mxfp4_to_bf16(packed1 & 0xF, scale);
              b_reg[3] = mxfp4_to_bf16(packed1 >> 4, scale);
            }

            acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                a_reg, b_reg, acc, 0, 0, 0);
          }
        } // blk loop

        // MFMA output: acc[i] = C[k_group*4+i, m_idx]
        // For BS=1: only C[0, m_idx] matters -> k_group=0, acc[0]
        // Each wave has fully reduced results (no cross-wave reduction needed)
        if (k_group == 0) {
          int out_n = b_row; // output column within workgroup
          float sum = acc[0];

          // Add bias
          float bias_val;
          unsigned bt = (unsigned)d_bias[expert_id * OUTPUT_STRIDE + out_n]
                        << 16;
          __builtin_memcpy(&bias_val, &bt, 4);

          float val = sum + bias_val;

          // Convert to bf16 and store
          unsigned t;
          __builtin_memcpy(&t, &val, 4);
          int out_idx = tok * (NUM_TOPK * OUTPUT_STRIDE) +
                        topk_slot * OUTPUT_STRIDE + out_n;
          d_output[out_idx] = (unsigned short)(t >> 16);
        }
      } // tile_iter loop

      __syncthreads(); // Ensure shared memory is free for next token
    }                  // token loop
  }                    // expert loop
}

} // namespace kernel
