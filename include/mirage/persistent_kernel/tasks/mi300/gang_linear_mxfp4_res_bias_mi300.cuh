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

// MXFP4 Gang Linear with Residual + Bias for MI300/MI350.
//
// Replaces gang_splitk_linear_res_bias for O-proj when weights are MXFP4.
// No split-K needed: MXFP4 weight is 3.7x smaller than BF16, so each tile
// completes in ~1us — fast enough that 6 tiles per XCD don't need split-K
// to keep workers busy.
//
// Weight format: MXFP4 packed per workgroup (same layout as MoE kernels):
//   [n_wgs_per_xcd, wg_bytes] where wg_bytes = OPW*(K/2) + OPW*(K/32)
//
// Dispatch: 8 gang tasks (1 per XCD), tiles assigned by tile_idx.
//   tok_idx = tile_idx / n_wgs_per_xcd
//   wg_idx  = tile_idx % n_wgs_per_xcd

#pragma once
#include "tasks/mi300/gang_moe_linear_mxfp4_mi300.cuh" // FP4xFP8 helpers

namespace kernel {

template <int BATCH_SIZE,
          int OUTPUT_PER_WG,
          int REDUCTION_SIZE>
__device__ __noinline__ void gang_linear_mxfp4_res_bias_kernel(
    void const *input_ptr,    // [batch, REDUCTION_SIZE] bf16
    void const *weight_ptr,   // [n_wgs_per_xcd, wg_bytes] packed MXFP4
    void const *residual_ptr, // [batch, output_stride] bf16 (partitioned)
    void const *bias_ptr,     // [1, output_size_per_xcd] bf16 (partitioned)
    void *output_ptr,         // [batch, output_stride] bf16 (partitioned)
    int num_active_tokens,
    int n_wgs_per_xcd,
    int output_stride,
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

  // ── MFMA constants ─────────────────────────────────────────────────────
  constexpr int K_PER_MFMA = 128;
  constexpr int MFMA_ITERS = REDUCTION_SIZE / K_PER_MFMA;
  static_assert(MFMA_ITERS >= 4,
                "Depth-4 pipeline requires REDUCTION_SIZE >= 512");

  // ── Wave tiling ─────────────────────────────────────────────────────────
  constexpr int NUM_WAVES = 4;
  constexpr int TILES_PER_WAVE = OUTPUT_PER_WG / 16 / NUM_WAVES;

  // ── Token activation in shared memory ────────────────────────────────────
  constexpr int FP8_TOK_DATA = REDUCTION_SIZE;

  unsigned short const *A = (unsigned short const *)input_ptr;
  uint8_t const *W = (uint8_t const *)weight_ptr;
  unsigned short const *d_residual = (unsigned short const *)residual_ptr;
  unsigned short const *d_bias = (unsigned short const *)bias_ptr;
  unsigned short *d_output = (unsigned short *)output_ptr;

  extern __shared__ char _lm_smem[];
  uint8_t *s_tok_fp8 = (uint8_t *)_lm_smem;
  uint8_t *s_tok_scales = s_tok_fp8 + FP8_TOK_DATA;

  int const tid = threadIdx.x;
  int const warp_id = tid >> 6;
  int const lane_id = tid & 63;
  int const col = lane_id & 15;
  int const g = lane_id >> 4;

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  unsigned long long _sp_t0 = 0, _sp_t1 = 0, _sp_t2 = 0;
  bool _sp_rec = (tile_idx == 0 && tid == 0 && g_subphase_active);
  if (_sp_rec) {
    _sp_t0 = __builtin_amdgcn_s_memrealtime();
  }
#endif

  // ── Tile dispatch ───────────────────────────────────────────────────────
  int tok_idx = tile_idx / n_wgs_per_xcd;
  int wg_idx = tile_idx % n_wgs_per_xcd;

  int batch_count =
      (num_active_tokens < BATCH_SIZE) ? num_active_tokens : BATCH_SIZE;
  if (tok_idx >= batch_count) {
    return;
  }

  // Workgroup weight pointers
  uint8_t const *wg_data = W + static_cast<int64_t>(wg_idx) * WG_BYTES;
  uint8_t const *wg_scales = wg_data + WG_DATA_BYTES;

  // ── Step 1: Quantize BF16 input -> FP8 in LDS ─────────────────
  unsigned short const *input_row = A + tok_idx * REDUCTION_SIZE;

  _gang_wave_parallel_fp8_quant<REDUCTION_SIZE>(
      input_row, s_tok_fp8, s_tok_scales);

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  if (_sp_rec) {
    _sp_t1 = __builtin_amdgcn_s_memrealtime();
  }
#endif

  // ── Step 2: MFMA FP4(weights) x BF16/FP4/FP8(tokens) ──────────────────
  if constexpr (OUTPUT_PER_WG >= 64) {
    // N-parallel: 4 waves handle different output rows (depth-4 pipeline)
    for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; tile_iter++) {
      int wave_tile = warp_id + tile_iter * NUM_WAVES;
      int w_row = wave_tile * 16 + col;

      int const row_data_base = w_row * (REDUCTION_SIZE / 2);
      int const row_scale_base = w_row * NUM_BLOCKS_32;

      f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

      // Pre-fill: load k-tiles 0..3 into pipeline slots
      i32x8_t a0 =
          *(i32x8_t const *)(wg_data + row_data_base + 0 * 64 + g * 16);
      int sa0 = (int)wg_scales[row_scale_base + 0 * 4 + g];
      i32x8_t a1 =
          *(i32x8_t const *)(wg_data + row_data_base + 1 * 64 + g * 16);
      int sa1 = (int)wg_scales[row_scale_base + 1 * 4 + g];
      i32x8_t a2 =
          *(i32x8_t const *)(wg_data + row_data_base + 2 * 64 + g * 16);
      int sa2 = (int)wg_scales[row_scale_base + 2 * 4 + g];
      i32x8_t a3 =
          *(i32x8_t const *)(wg_data + row_data_base + 3 * 64 + g * 16);
      int sa3 = (int)wg_scales[row_scale_base + 3 * 4 + g];

#pragma unroll 1
      for (int ki = 0; ki < MFMA_ITERS; ki += 4) {
        // Slot 0
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

        // Slot 1
        {
          i32x8_t b =
              _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 1) * K_PER_MFMA, g);
          int sb = (int)s_tok_scales[ki + 1];
          acc = _gang_mfma_f4xf8(a1, b, acc, sa1, sb);
        }
        if (ki + 5 < MFMA_ITERS) {
          int kt5 = (ki + 5) * K_PER_MFMA;
          a1 = *(i32x8_t const *)(wg_data + row_data_base + kt5 / 2 + g * 16);
          sa1 = (int)wg_scales[row_scale_base + kt5 / 32 + g];
        }

        // Slot 2
        {
          i32x8_t b =
              _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 2) * K_PER_MFMA, g);
          int sb = (int)s_tok_scales[ki + 2];
          acc = _gang_mfma_f4xf8(a2, b, acc, sa2, sb);
        }
        if (ki + 6 < MFMA_ITERS) {
          int kt6 = (ki + 6) * K_PER_MFMA;
          a2 = *(i32x8_t const *)(wg_data + row_data_base + kt6 / 2 + g * 16);
          sa2 = (int)wg_scales[row_scale_base + kt6 / 32 + g];
        }

        // Slot 3
        if (ki + 3 < MFMA_ITERS) {
          i32x8_t b =
              _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 3) * K_PER_MFMA, g);
          int sb = (int)s_tok_scales[ki + 3];
          acc = _gang_mfma_f4xf8(a3, b, acc, sa3, sb);
        }
        if (ki + 7 < MFMA_ITERS) {
          int kt7 = (ki + 7) * K_PER_MFMA;
          a3 = *(i32x8_t const *)(wg_data + row_data_base + kt7 / 2 + g * 16);
          sa3 = (int)wg_scales[row_scale_base + kt7 / 32 + g];
        }
      }

      // ── Step 3: Epilogue: result + bias + residual -> bf16 output ─────────
      // Vectorized: load 4 bf16 bias + 4 bf16 residual as uint2, store 4 bf16
      // output as uint2.
      if (col == 0) {
        int out_n_base = wg_idx * OUTPUT_PER_WG + wave_tile * 16 + g * 4;
        int out_idx_base = tok_idx * output_stride + out_n_base;

        // Single vectorized load of 4 consecutive bf16 bias values
        // (flat_load_dwordx2)
        uint2 bias_packed;
        __builtin_memcpy(&bias_packed, &d_bias[out_n_base], 8);
        // Single vectorized load of 4 consecutive bf16 residual values
        uint2 res_packed;
        __builtin_memcpy(&res_packed, &d_residual[out_idx_base], 8);

        // Extract 4 bf16 bias → f32
        unsigned bt0 = (bias_packed.x & 0xFFFFu) << 16;
        unsigned bt1 = bias_packed.x & 0xFFFF0000u;
        unsigned bt2 = (bias_packed.y & 0xFFFFu) << 16;
        unsigned bt3 = bias_packed.y & 0xFFFF0000u;
        float bv0, bv1, bv2, bv3;
        __builtin_memcpy(&bv0, &bt0, 4);
        __builtin_memcpy(&bv1, &bt1, 4);
        __builtin_memcpy(&bv2, &bt2, 4);
        __builtin_memcpy(&bv3, &bt3, 4);

        // Extract 4 bf16 residual → f32
        unsigned rt0 = (res_packed.x & 0xFFFFu) << 16;
        unsigned rt1 = res_packed.x & 0xFFFF0000u;
        unsigned rt2 = (res_packed.y & 0xFFFFu) << 16;
        unsigned rt3 = res_packed.y & 0xFFFF0000u;
        float rv0, rv1, rv2, rv3;
        __builtin_memcpy(&rv0, &rt0, 4);
        __builtin_memcpy(&rv1, &rt1, 4);
        __builtin_memcpy(&rv2, &rt2, 4);
        __builtin_memcpy(&rv3, &rt3, 4);

        // Add accumulator + bias + residual, pack 4 bf16 output, store as uint2
        unsigned short o0 = _gang_float_to_bf16(acc[0] + bv0 + rv0);
        unsigned short o1 = _gang_float_to_bf16(acc[1] + bv1 + rv1);
        unsigned short o2 = _gang_float_to_bf16(acc[2] + bv2 + rv2);
        unsigned short o3 = _gang_float_to_bf16(acc[3] + bv3 + rv3);
        uint2 out_packed;
        out_packed.x = (unsigned)o0 | ((unsigned)o1 << 16);
        out_packed.y = (unsigned)o2 | ((unsigned)o3 << 16);
        __builtin_memcpy(&d_output[out_idx_base], &out_packed, 8);
      }
    }
  } else {
    // K-parallel: 4 waves all process same 16 rows, split K across waves.
    // Depth-4 pipelined: 4 weight K-tiles pre-loaded, compute overlaps with
    // prefetch of next 4. Each MFMA ~32 cycles; 4 in flight = ~128 cycles
    // of latency hiding.
    constexpr int TOTAL_K_ITERS = MFMA_ITERS;
    constexpr int KP_BASE = TOTAL_K_ITERS / NUM_WAVES;
    constexpr int KP_EXTRA = TOTAL_K_ITERS % NUM_WAVES;
    static_assert(KP_BASE >= 4,
                  "Depth-4 K-parallel requires >= 4 iters per wave");

    int const ki_start =
        (warp_id < KP_EXTRA)
            ? warp_id * (KP_BASE + 1)
            : KP_EXTRA * (KP_BASE + 1) + (warp_id - KP_EXTRA) * KP_BASE;
    int const ki_end = ki_start + KP_BASE + (warp_id < KP_EXTRA ? 1 : 0);

    int w_row = col;
    int const row_data_base = w_row * (REDUCTION_SIZE / 2);
    int const row_scale_base = w_row * NUM_BLOCKS_32;

    f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

    // Pre-fill: load k-tiles 0..3 into pipeline slots
    i32x8_t a0 =
        *(i32x8_t const *)(wg_data + row_data_base + ki_start * 64 + g * 16);
    int sa0 = (int)wg_scales[row_scale_base + ki_start * 4 + g];
    i32x8_t a1 = *(i32x8_t const *)(wg_data + row_data_base +
                                    (ki_start + 1) * 64 + g * 16);
    int sa1 = (int)wg_scales[row_scale_base + (ki_start + 1) * 4 + g];
    i32x8_t a2 = *(i32x8_t const *)(wg_data + row_data_base +
                                    (ki_start + 2) * 64 + g * 16);
    int sa2 = (int)wg_scales[row_scale_base + (ki_start + 2) * 4 + g];
    i32x8_t a3 = *(i32x8_t const *)(wg_data + row_data_base +
                                    (ki_start + 3) * 64 + g * 16);
    int sa3 = (int)wg_scales[row_scale_base + (ki_start + 3) * 4 + g];

// IMPORTANT: #pragma unroll 1 prevents ROCm miscompilation.
#pragma unroll 1
    for (int ki = ki_start; ki < ki_end; ki += 4) {
      // Slot 0: compute k-tile ki, prefetch ki+4
      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, ki * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki];
        acc = _gang_mfma_f4xf8(a0, b, acc, sa0, sb);
      }
      if (ki + 4 < ki_end) {
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
      if (ki + 5 < ki_end) {
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
      if (ki + 6 < ki_end) {
        int kt6 = (ki + 6) * K_PER_MFMA;
        a2 = *(i32x8_t const *)(wg_data + row_data_base + kt6 / 2 + g * 16);
        sa2 = (int)wg_scales[row_scale_base + kt6 / 32 + g];
      }

      // Slot 3: compute k-tile ki+3, prefetch ki+7
      if (ki + 3 < ki_end) {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 3) * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki + 3];
        acc = _gang_mfma_f4xf8(a3, b, acc, sa3, sb);
      }
      if (ki + 7 < ki_end) {
        int kt7 = (ki + 7) * K_PER_MFMA;
        a3 = *(i32x8_t const *)(wg_data + row_data_base + kt7 / 2 + g * 16);
        sa3 = (int)wg_scales[row_scale_base + kt7 / 32 + g];
      }
    }

    float *lds_reduce = (float *)_lm_smem;
    if (col == 0) {
      for (int i = 0; i < 4; i++) {
        lds_reduce[warp_id * OUTPUT_PER_WG + g * 4 + i] = acc[i];
      }
    }
    __syncthreads();

    if (warp_id == 0 && col == 0) {
      // Reduce across waves
      float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
      for (int w = 0; w < NUM_WAVES; w++) {
        v0 += lds_reduce[w * OUTPUT_PER_WG + g * 4 + 0];
        v1 += lds_reduce[w * OUTPUT_PER_WG + g * 4 + 1];
        v2 += lds_reduce[w * OUTPUT_PER_WG + g * 4 + 2];
        v3 += lds_reduce[w * OUTPUT_PER_WG + g * 4 + 3];
      }

      int out_n_base = wg_idx * OUTPUT_PER_WG + g * 4;
      int out_idx_base = tok_idx * output_stride + out_n_base;

      // Vectorized load of 4 bf16 bias + 4 bf16 residual (flat_load_dwordx2
      // each)
      uint2 bias_packed;
      __builtin_memcpy(&bias_packed, &d_bias[out_n_base], 8);
      uint2 res_packed;
      __builtin_memcpy(&res_packed, &d_residual[out_idx_base], 8);

      // Extract bias bf16 → f32
      unsigned bt0 = (bias_packed.x & 0xFFFFu) << 16;
      unsigned bt1 = bias_packed.x & 0xFFFF0000u;
      unsigned bt2 = (bias_packed.y & 0xFFFFu) << 16;
      unsigned bt3 = bias_packed.y & 0xFFFF0000u;
      float bv0, bv1, bv2, bv3;
      __builtin_memcpy(&bv0, &bt0, 4);
      __builtin_memcpy(&bv1, &bt1, 4);
      __builtin_memcpy(&bv2, &bt2, 4);
      __builtin_memcpy(&bv3, &bt3, 4);

      // Extract residual bf16 → f32
      unsigned rt0 = (res_packed.x & 0xFFFFu) << 16;
      unsigned rt1 = res_packed.x & 0xFFFF0000u;
      unsigned rt2 = (res_packed.y & 0xFFFFu) << 16;
      unsigned rt3 = res_packed.y & 0xFFFF0000u;
      float rv0, rv1, rv2, rv3;
      __builtin_memcpy(&rv0, &rt0, 4);
      __builtin_memcpy(&rv1, &rt1, 4);
      __builtin_memcpy(&rv2, &rt2, 4);
      __builtin_memcpy(&rv3, &rt3, 4);

      // Pack 4 bf16 output, store as uint2 (flat_store_dwordx2)
      unsigned short o0 = _gang_float_to_bf16(v0 + bv0 + rv0);
      unsigned short o1 = _gang_float_to_bf16(v1 + bv1 + rv1);
      unsigned short o2 = _gang_float_to_bf16(v2 + bv2 + rv2);
      unsigned short o3 = _gang_float_to_bf16(v3 + bv3 + rv3);
      uint2 out_packed;
      out_packed.x = (unsigned)o0 | ((unsigned)o1 << 16);
      out_packed.y = (unsigned)o2 | ((unsigned)o3 << 16);
      __builtin_memcpy(&d_output[out_idx_base], &out_packed, 8);
    }
  }

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  if (_sp_rec) {
    _sp_t2 = __builtin_amdgcn_s_memrealtime();
    // Slot 2: OPROJ. [0]=FP8Quant [1]=MFMA+ResidBiasEpilogue
    atomicAdd(&g_subphase_ns[2][0], (_sp_t1 - _sp_t0) * 10);
    atomicAdd(&g_subphase_ns[2][1], (_sp_t2 - _sp_t1) * 10);
    atomicAdd(&g_subphase_cnt[2], 1ULL);
  }
#endif

  __syncthreads();
}

} // namespace kernel
