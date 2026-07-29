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

// Software-pipelined gang MoE MXFP4 linear kernel for MI350 (gfx950).
//
// Key optimization over gang_moe_linear_mxfp4_mi300.cuh:
//
//  Depth-4 software pipelining: pre-load weight data for 4 future K-tiles
//  while computing MFMA on the current tile. Each MFMA takes ~32 cycles;
//  with 4 in flight, the load-to-use distance is ~128 cycles, covering a
//  significant fraction of the ~400-cycle HBM latency per global load.
//
//  IMPORTANT: The main loop MUST use #pragma unroll 1. Without it, the
//  ROCm compiler unrolls and miscompiles the pipeline control flow,
//  producing incorrect results.
//
// Same interface as gang_moe_linear_mxfp4_kernel_mi300: drop-in replacement.
// Same weight layout, same dispatch, same epilogue (bias / fused SwiGLU).
//
// Measured: 7.25ms -> 7.05ms/token (2.8% speedup) on GPT-OSS 120B decode.

#pragma once
#include "tasks/mi300/gang_moe_linear_mxfp4_mi300.cuh" // type defs + helpers
#include "tasks/mi300/swigluoai_mi300.cuh"

namespace kernel {

// ── Pipelined MoE MXFP4 kernel ───────────────────────────────────────────
template <int BATCH_SIZE,
          int OUTPUT_SIZE,
          int OUTPUT_STRIDE,
          int REDUCTION_SIZE,
          int NUM_EXPERTS,
          int NUM_TOPK,
          int TILES_PER_EXPERT,
          int OUTPUT_PER_WG,
          bool W13_LINEAR,
          bool FUSE_SWIGLU = false>
__device__ __noinline__ void
    gang_moe_pipelined_mxfp4_kernel_mi300(void const *input_ptr,
                                          void const *weight_ptr,
                                          void const *routing_ptr,
                                          void const *mask_ptr,
                                          void const *bias_ptr,
                                          void *output_ptr,
                                          int tile_idx) {
  static_assert(OUTPUT_PER_WG % 16 == 0,
                "OUTPUT_PER_WG must be multiple of 16");
  static_assert(REDUCTION_SIZE % 128 == 0,
                "REDUCTION_SIZE must be multiple of 128 for FP4 MFMA");

  // ── Weight layout constants ─────────────────────────────────────────────
  constexpr int NUM_BLOCKS_32 = REDUCTION_SIZE / 32;
  constexpr int WG_DATA_BYTES = OUTPUT_PER_WG * (REDUCTION_SIZE / 2);
  constexpr int WG_SCALE_BYTES = OUTPUT_PER_WG * NUM_BLOCKS_32;
  constexpr int WG_BYTES = WG_DATA_BYTES + WG_SCALE_BYTES;
  constexpr int EXPERT_WGS = OUTPUT_STRIDE / OUTPUT_PER_WG;
  constexpr int64_t EXPERT_BYTES = static_cast<int64_t>(EXPERT_WGS) * WG_BYTES;

  // ── MFMA constants ─────────────────────────────────────────────────────
  constexpr int K_PER_MFMA = 128;
  constexpr int MFMA_ITERS = REDUCTION_SIZE / K_PER_MFMA;

  // ── Wave tiling ─────────────────────────────────────────────────────────
  constexpr int NUM_WAVES = 4;
  constexpr int TILES_PER_WAVE = OUTPUT_PER_WG / 16 / NUM_WAVES;
  constexpr int N_TILES_PER_WG = EXPERT_WGS;

  // ── FP8 token in shared memory ──────────────────────────────────────────
  constexpr int FP8_TOK_DATA = REDUCTION_SIZE;

  unsigned short const *A = (unsigned short const *)input_ptr;
  uint8_t const *W = (uint8_t const *)weight_ptr;
  int const *d_routing = (int const *)routing_ptr;
  int const *d_mask = (int const *)mask_ptr;
  unsigned short const *d_bias = (unsigned short const *)bias_ptr;
  unsigned short *d_output = (unsigned short *)output_ptr;

  extern __shared__ char _pipe_smem[];
  uint8_t *s_tok_fp8 = (uint8_t *)_pipe_smem;
  uint8_t *s_tok_scales = s_tok_fp8 + FP8_TOK_DATA;

  int const tid = threadIdx.x;
  int const warp_id = tid >> 6;
  int const lane_id = tid & 63;
  int const col = lane_id & 15;
  int const g = lane_id >> 4;

  // ── Tile dispatch (same as gang_moe_linear_mxfp4_kernel_mi300) ────────
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

  int tok_idx = tile_within_expert / N_TILES_PER_WG;
  int wg_idx = tile_within_expert % N_TILES_PER_WG;
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

  // ── Phase 1: Quantize BF16 input → FP8 in LDS ──────────────────────────
  unsigned short const *input_base;
  if constexpr (W13_LINEAR) {
    input_base = A + tok_idx * REDUCTION_SIZE;
  } else {
    input_base =
        A + tok_idx * (NUM_TOPK * REDUCTION_SIZE) + topk_slot * REDUCTION_SIZE;
  }

  _gang_wave_parallel_fp8_quant<REDUCTION_SIZE>(
      input_base, s_tok_fp8, s_tok_scales);

  // ── Phase 2: Depth-4 pipelined MFMA FP4(weights) × FP8(tokens) ────────
  //
  // Pre-fill 4 pipeline slots with weight tiles from HBM.
  // Main loop: compute with slot N, then prefetch for slot N (ki+4).
  // Between a slot's prefetch and its next use, 4 MFMAs execute,
  // giving ~128 cycles for the global load to complete.

  for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; tile_iter++) {
    int wave_tile = warp_id + tile_iter * NUM_WAVES;
    int w_row = wave_tile * 16 + col;

    // Row base offsets (constant across K iterations)
    int const row_data_base = w_row * (REDUCTION_SIZE / 2);
    int const row_scale_base = w_row * NUM_BLOCKS_32;

    f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

    // Pre-fill: load k-tiles 0..3 into pipeline slots
    i32x8_t a0 = *(i32x8_t const *)(wg_data + row_data_base + 0 * 64 + g * 16);
    int sa0 = (int)wg_scales[row_scale_base + 0 * 4 + g];
    i32x8_t a1 = *(i32x8_t const *)(wg_data + row_data_base + 1 * 64 + g * 16);
    int sa1 = (int)wg_scales[row_scale_base + 1 * 4 + g];
    i32x8_t a2 = *(i32x8_t const *)(wg_data + row_data_base + 2 * 64 + g * 16);
    int sa2 = (int)wg_scales[row_scale_base + 2 * 4 + g];
    i32x8_t a3 = *(i32x8_t const *)(wg_data + row_data_base + 3 * 64 + g * 16);
    int sa3 = (int)wg_scales[row_scale_base + 3 * 4 + g];

// IMPORTANT: #pragma unroll 1 is required. Without it, the ROCm
// compiler unrolls and miscompiles this loop, producing wrong results.
#pragma unroll 1
    for (int ki = 0; ki < MFMA_ITERS; ki += 4) {
      // Slot 0: compute k-tile ki, prefetch ki+4
      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, ki * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki];
        acc = _gang_mfma_f4xf8(a0, b, acc, sa0, sb);
      }
      if (ki + 4 < MFMA_ITERS) {
        int kt4 = (ki + 4) * K_PER_MFMA;
        a0 = *(i32x8_t const *)(wg_data + row_data_base + kt4 / 2 + g * 16);
        sa0 = (int)wg_scales[row_scale_base + kt4 / 32 + g];
      }

      // Slot 1: compute k-tile ki+1, prefetch ki+5
      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 1) * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki + 1];
        acc = _gang_mfma_f4xf8(a1, b, acc, sa1, sb);
      }
      if (ki + 5 < MFMA_ITERS) {
        int kt5 = (ki + 5) * K_PER_MFMA;
        a1 = *(i32x8_t const *)(wg_data + row_data_base + kt5 / 2 + g * 16);
        sa1 = (int)wg_scales[row_scale_base + kt5 / 32 + g];
      }

      // Slot 2: compute k-tile ki+2, prefetch ki+6
      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 2) * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki + 2];
        acc = _gang_mfma_f4xf8(a2, b, acc, sa2, sb);
      }
      if (ki + 6 < MFMA_ITERS) {
        int kt6 = (ki + 6) * K_PER_MFMA;
        a2 = *(i32x8_t const *)(wg_data + row_data_base + kt6 / 2 + g * 16);
        sa2 = (int)wg_scales[row_scale_base + kt6 / 32 + g];
      }

      // Slot 3: compute k-tile ki+3, prefetch ki+7
      if (ki + 3 < MFMA_ITERS) {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 3) * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki + 3];
        acc = _gang_mfma_f4xf8(a3, b, acc, sa3, sb);
      }
      if (ki + 7 < MFMA_ITERS) {
        int kt7 = (ki + 7) * K_PER_MFMA;
        a3 = *(i32x8_t const *)(wg_data + row_data_base + kt7 / 2 + g * 16);
        sa3 = (int)wg_scales[row_scale_base + kt7 / 32 + g];
      }
    }

    // ── Epilogue: bias + optional SwiGLU activation ─────────────────────
    if (col == 0) {
      if constexpr (FUSE_SWIGLU) {
        constexpr int ACT_STRIDE = OUTPUT_STRIDE / 2;
        for (int i = 0; i < 4; i += 2) {
          int out_n = wg_idx * OUTPUT_PER_WG + wave_tile * 16 + g * 4 + i;
          if (out_n + 1 < OUTPUT_SIZE) {
            unsigned bt_g = (unsigned)d_bias[expert_id * OUTPUT_STRIDE + out_n]
                            << 16;
            unsigned bt_u =
                (unsigned)d_bias[expert_id * OUTPUT_STRIDE + out_n + 1] << 16;
            float bias_g;
            __builtin_memcpy(&bias_g, &bt_g, 4);
            float bias_u;
            __builtin_memcpy(&bias_u, &bt_u, 4);

            float activated =
                fast_swigluoai(acc[i] + bias_g, acc[i + 1] + bias_u);

            int act_n = out_n / 2;
            int out_idx = tok_idx * (NUM_TOPK * ACT_STRIDE) +
                          topk_slot * ACT_STRIDE + act_n;
            d_output[out_idx] = _gang_float_to_bf16(activated);
          }
        }
      } else {
        for (int i = 0; i < 4; i++) {
          int out_n = wg_idx * OUTPUT_PER_WG + wave_tile * 16 + g * 4 + i;
          if (out_n < OUTPUT_SIZE) {
            float sum = acc[i];

            unsigned bt = (unsigned)d_bias[expert_id * OUTPUT_STRIDE + out_n]
                          << 16;
            float bv;
            __builtin_memcpy(&bv, &bt, 4);

            int out_idx = tok_idx * (NUM_TOPK * OUTPUT_STRIDE) +
                          topk_slot * OUTPUT_STRIDE + out_n;
            d_output[out_idx] = _gang_float_to_bf16(sum + bv);
          }
        }
      }
    }
  }

  __syncthreads();
}

} // namespace kernel
