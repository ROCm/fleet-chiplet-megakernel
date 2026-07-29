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

// Fused SwiGLU + W2 (down) MoE MXFP4 kernel for MI350 (gfx950).
//
// Fuses the SwiGLU activation into W2's input quantization step.
// Instead of reading one BF16 input (silu_mul_out), reads two BF16 inputs
// (gate_out, up_out) and applies SwiGLU(gate, up) = silu(gate) * (up + 1)
// during FP8 quantization. Result feeds directly into W2 MFMA.
//
// Eliminates:
//   - Separate SwiGLU task dispatch
//   - HBM write of SwiGLU intermediate (3072 × bs × topk × 2 bytes)
//   - HBM read of SwiGLU intermediate by W2
//
// Input layout: W13 output = interleaved [gate, up] per row
//   mlp_mid[tok][slot][0::2] = gate values
//   mlp_mid[tok][slot][1::2] = up values
//   Total dim: [bs, topk, 2 * intermediate]

#pragma once
#include "tasks/mi300/gang_moe_linear_mxfp4_mi300.cuh"
#include "tasks/mi300/swigluoai_mi300.cuh"

namespace kernel {

template <int BATCH_SIZE,
          int INTERMEDIATE_SIZE,
          int HIDDEN_SIZE,
          int NUM_EXPERTS,
          int NUM_TOPK,
          int TILES_PER_EXPERT,
          int OUTPUT_PER_WG>
__device__ __noinline__ void gang_moe_swiglu_w2_mxfp4_kernel_mi300(
    void const *
        w13_output_ptr, // [bs, topk, 2*intermediate] BF16 (interleaved gate/up)
    void const *weight_ptr,  // [E, hidden/OPW, wg_bytes] MXFP4
    void const *routing_ptr, // [E, bs] int
    void const *mask_ptr,    // [E+1] int
    void const *bias_ptr,    // [E, hidden] BF16
    void *output_ptr,        // [bs, topk, hidden] BF16
    int tile_idx) {

  static_assert(OUTPUT_PER_WG % 16 == 0,
                "OUTPUT_PER_WG must be multiple of 16");
  static_assert(INTERMEDIATE_SIZE % 128 == 0,
                "INTERMEDIATE_SIZE must be multiple of 128");

  // Weight layout constants (W2: reduction over INTERMEDIATE_SIZE, output
  // HIDDEN_SIZE)
  constexpr int REDUCTION_SIZE = INTERMEDIATE_SIZE;
  constexpr int OUTPUT_SIZE = HIDDEN_SIZE;
  constexpr int NUM_BLOCKS_32 = REDUCTION_SIZE / 32;
  constexpr int WG_DATA_BYTES = OUTPUT_PER_WG * (REDUCTION_SIZE / 2);
  constexpr int WG_SCALE_BYTES = OUTPUT_PER_WG * NUM_BLOCKS_32;
  constexpr int WG_BYTES = WG_DATA_BYTES + WG_SCALE_BYTES;
  constexpr int EXPERT_WGS = HIDDEN_SIZE / OUTPUT_PER_WG;
  constexpr int64_t EXPERT_BYTES = static_cast<int64_t>(EXPERT_WGS) * WG_BYTES;

  // MFMA constants
  constexpr int K_PER_MFMA = 128;
  constexpr int MFMA_ITERS = REDUCTION_SIZE / K_PER_MFMA;
  constexpr int NUM_WAVES = 4;
  constexpr int N_TILES = OUTPUT_PER_WG / 16;
  constexpr int TILES_PER_WAVE = N_TILES / NUM_WAVES;

  // Pointers
  unsigned short const *A = (unsigned short const *)w13_output_ptr;
  uint8_t const *W = (uint8_t const *)weight_ptr;
  int const *d_routing = (int const *)routing_ptr;
  int const *d_mask = (int const *)mask_ptr;
  unsigned short const *d_bias = (unsigned short const *)bias_ptr;
  unsigned short *d_output = (unsigned short *)output_ptr;

  // Shared memory: FP8 quantized tokens + scales
  extern __shared__ char _swiglu_w2_smem[];
  uint8_t *s_tok_fp8 = (uint8_t *)_swiglu_w2_smem;
  uint8_t *s_tok_scales = s_tok_fp8 + REDUCTION_SIZE;

  int const tid = threadIdx.x;
  int const warp_id = tid >> 6;
  int const lane_id = tid & 63;
  int const col = lane_id & 15;
  int const g = lane_id >> 4;

  // ── Tile decode (flat global distribution across 8 XCDs) ─────────────────
  int xcd_id = _gang_moe_get_xcd_id();
  int const num_activated_experts = d_mask[NUM_EXPERTS];

  int global_tile = tile_idx * 8 + xcd_id;
  int total_tiles = num_activated_experts * TILES_PER_EXPERT;
  if (global_tile >= total_tiles) {
    return;
  }
  int expert_idx = global_tile / TILES_PER_EXPERT;
  int tile_within_expert = global_tile % TILES_PER_EXPERT;
  int expert_id = d_mask[expert_idx];
  int const *expert_routing = d_routing + expert_id * BATCH_SIZE;

  uint8_t const *expert_weight =
      W + static_cast<int64_t>(expert_id) * EXPERT_BYTES;

  int tok_idx = tile_within_expert / EXPERT_WGS;
  int wg_idx = tile_within_expert % EXPERT_WGS;

  if (tok_idx >= BATCH_SIZE) {
    return;
  }

  int route_val = expert_routing[tok_idx];
  if (route_val == 0) {
    return;
  }
  int topk_slot = route_val - 1;

  uint8_t const *wg_data =
      expert_weight + static_cast<int64_t>(wg_idx) * WG_BYTES;
  uint8_t const *wg_scales = wg_data + WG_DATA_BYTES;

  // ── Phase 1: Read interleaved gate/up, apply SwiGLU, quantize to FP8 ────
  // W13 output layout: [bs, topk, 2*intermediate] with gate at [::2], up at
  // [1::2]
  constexpr int W13_OUT_DIM = 2 * INTERMEDIATE_SIZE;
  unsigned short const *w13_base =
      A + tok_idx * (NUM_TOPK * W13_OUT_DIM) + topk_slot * W13_OUT_DIM;

#pragma unroll 1
  for (int blk = tid; blk < MFMA_ITERS; blk += blockDim.x) {
    int base_k = blk * K_PER_MFMA;
    unsigned short const *blk_base = w13_base + 2 * base_k;

    // Pass 1: compute SwiGLU and find amax (no intermediate storage)
    float amax = 0.0f;
    for (int j = 0; j < K_PER_MFMA; j++) {
      float gate_val = _gang_bf16_to_float(blk_base[2 * j]);
      float up_val = _gang_bf16_to_float(blk_base[2 * j + 1]);
      float v = fast_swigluoai(gate_val, up_val);
      float av = v < 0.0f ? -v : v;
      amax = amax > av ? amax : av;
    }

    // Compute block scale
    uint8_t se = _gang_compute_e8m0_fp8(amax);
    s_tok_scales[blk] = se;
    float scale_f = (se == 0) ? 1.0f : exp2f((float)se - 127.0f);

    // Pass 2: re-read, re-apply SwiGLU, quantize to FP8
    for (int j = 0; j < K_PER_MFMA; j += 4) {
      float v0 = fast_swigluoai(_gang_bf16_to_float(blk_base[2 * (j + 0)]),
                                _gang_bf16_to_float(blk_base[2 * (j + 0) + 1]));
      float v1 = fast_swigluoai(_gang_bf16_to_float(blk_base[2 * (j + 1)]),
                                _gang_bf16_to_float(blk_base[2 * (j + 1) + 1]));
      float v2 = fast_swigluoai(_gang_bf16_to_float(blk_base[2 * (j + 2)]),
                                _gang_bf16_to_float(blk_base[2 * (j + 2) + 1]));
      float v3 = fast_swigluoai(_gang_bf16_to_float(blk_base[2 * (j + 3)]),
                                _gang_bf16_to_float(blk_base[2 * (j + 3) + 1]));
      fp8x4_t pk = _gang_quant_4xfp8(v0, v1, v2, v3, scale_f);
      *(fp8x4_t *)(s_tok_fp8 + base_k + j) = pk;
    }
  }
  __syncthreads();

  // ── Phase 2: MFMA FP4(weights) × FP8(swiglu_tokens) ─────────────────────
  for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; tile_iter++) {
    int wave_tile = warp_id + tile_iter * NUM_WAVES;

    f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll 1
    for (int kt = 0; kt < REDUCTION_SIZE; kt += K_PER_MFMA) {
      int w_row = wave_tile * 16 + col;
      int a_off = w_row * (REDUCTION_SIZE / 2) + kt / 2 + g * 16;
      i32x8_t a_reg = *(i32x8_t const *)(wg_data + a_off);
      i32x8_t b_reg = _gang_load_fp8_mfma_b(s_tok_fp8, kt, g);
      int sa = (int)wg_scales[w_row * NUM_BLOCKS_32 + kt / 32 + g];
      int sb = (int)s_tok_scales[kt / K_PER_MFMA];
      acc = _gang_mfma_f4xf8(a_reg, b_reg, acc, sa, sb);
    }

    // ── Epilogue: add bias, write BF16 ──────────────────────────────────
    if (col == 0) {
      for (int i = 0; i < 4; i++) {
        int out_n = wg_idx * OUTPUT_PER_WG + wave_tile * 16 + g * 4 + i;
        if (out_n < OUTPUT_SIZE) {
          float sum = acc[i];

          unsigned bt = (unsigned)d_bias[expert_id * HIDDEN_SIZE + out_n] << 16;
          float bv;
          __builtin_memcpy(&bv, &bt, 4);

          int out_idx = tok_idx * (NUM_TOPK * HIDDEN_SIZE) +
                        topk_slot * HIDDEN_SIZE + out_n;
          d_output[out_idx] = _gang_float_to_bf16(sum + bv);
        }
      }
    }
  }

  __syncthreads();
}

} // namespace kernel
