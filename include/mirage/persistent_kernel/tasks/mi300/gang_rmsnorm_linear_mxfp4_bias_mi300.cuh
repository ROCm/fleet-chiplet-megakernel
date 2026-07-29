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

// Fused RMSNorm + MXFP4 Gang Linear + Bias for MI300/MI350.
//
// Same structure as gang_rmsnorm_linear_bias_mi300.cuh but uses MXFP4 weights
// with hardware FP4xFP8 MFMA instead of BF16 CK GEMM pipeline.
//
// Weight format: MXFP4 packed per workgroup (same layout as MoE kernels):
//   [n_wgs_per_xcd, wg_bytes] where wg_bytes = OPW*(K/2) + OPW*(K/32)
//
// Dispatch: 8 gang tasks (1 per XCD), tiles assigned by tile_idx.
//   tok_idx = tile_idx / n_wgs_per_xcd
//   wg_idx  = tile_idx % n_wgs_per_xcd
//
// Step 1: Every worker computes RMSNorm redundantly (same as existing kernel).
// Step 2: Quantize normalized BF16 input to FP8 in LDS.
// Step 3: Depth-4 pipelined MFMA FP4(weight) x FP8(input).
// Step 4: Bias epilogue, write BF16 output.
//
// Expected speedup: QKV weight 31.5 MB BF16 -> 8.4 MB MXFP4 (3.7x reduction).
// QKV compute: ~26us -> ~8us per layer.

#pragma once
#include "tasks/mi300/gang_moe_linear_mxfp4_mi300.cuh" // FP4xFP8 type defs + helpers
#include "tasks/mi300/gang_rmsnorm_linear_bias_mi300.cuh" // RMSNorm prologue

namespace kernel {

// Fused RMSNorm + MXFP4 Gang Linear + Bias.
//
// Template params:
//   BATCH_SIZE       - max batch size (usually 1 for decode)
//   OUTPUT_PER_WG    - output rows per workgroup (e.g. 64)
//   REDUCTION_SIZE   - input/reduction dimension (e.g. 3072)
//   ACTUAL_HIDDEN_DIM - unpadded hidden size for RMSNorm denominator (e.g.
//   2880)
//
// Runtime params:
//   tile_idx          - gang task tile index (0..total_tiles_per_xcd-1)
//   n_wgs_per_xcd     - number of workgroups per XCD
//   output_stride      - full output stride (for row indexing)
//   num_active_tokens  - actual number of active tokens
template <int BATCH_SIZE,
          int OUTPUT_PER_WG,
          int REDUCTION_SIZE,
          int ACTUAL_HIDDEN_DIM = REDUCTION_SIZE>
__device__ __noinline__ void gang_rmsnorm_linear_mxfp4_bias_kernel(
    void const *norm_input_ptr,  // [batch, REDUCTION_SIZE] bf16
    void const *norm_weight_ptr, // [REDUCTION_SIZE] bf16
    void *norm_output_ptr,       // [batch, REDUCTION_SIZE] bf16 scratch
    void const *weight_ptr,      // [n_wgs_per_xcd, wg_bytes] packed MXFP4
    void const *bias_ptr,        // [1, output_size_per_xcd] bf16 (partitioned)
    void *output_ptr,            // [batch, output_stride] bf16 (partitioned)
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
  constexpr int BF16_MFMA_ITERS = REDUCTION_SIZE / 32;

  // ── Wave tiling ─────────────────────────────────────────────────────────
  constexpr int NUM_WAVES = 4;
  constexpr int TILES_PER_WAVE = OUTPUT_PER_WG / 16 / NUM_WAVES;

  // ── Token activation in shared memory ────────────────────────────────────
  constexpr int FP8_TOK_DATA = REDUCTION_SIZE;

  uint8_t const *W = (uint8_t const *)weight_ptr;
  unsigned short const *d_bias = (unsigned short const *)bias_ptr;
  unsigned short *d_output = (unsigned short *)output_ptr;

  extern __shared__ char _rnlm_smem[];
  uint8_t *s_tok_fp8 = (uint8_t *)_rnlm_smem;
  uint8_t *s_tok_scales = s_tok_fp8 + FP8_TOK_DATA;

  int const tid = threadIdx.x;
  int const warp_id = tid >> 6;
  int const lane_id = tid & 63;
  int const col = lane_id & 15;
  int const g = lane_id >> 4;

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  unsigned long long _sp_t0 = 0, _sp_t1 = 0, _sp_t2 = 0, _sp_t3 = 0;
  bool _sp_rec = (tile_idx == 0 && tid == 0 && g_subphase_active);
  if (_sp_rec) {
    _sp_t0 = __builtin_amdgcn_s_memrealtime();
  }
#endif
  // ── Step 1: Redundant RMSNorm ───────────────────────────────────────────
  // All workers compute the same RMSNorm and write to norm_output_ptr.
  // For bs>1, loop over active tokens.
  int batch_count =
      (num_active_tokens < BATCH_SIZE) ? num_active_tokens : BATCH_SIZE;
  for (int b = 0; b < batch_count; b++) {
    unsigned short const *row_in =
        (unsigned short const *)norm_input_ptr + b * REDUCTION_SIZE;
    unsigned short *row_out =
        (unsigned short *)norm_output_ptr + b * REDUCTION_SIZE;
    gang_rmsnorm_detail::rmsnorm_inline_amd<REDUCTION_SIZE, ACTUAL_HIDDEN_DIM>(
        row_in, norm_weight_ptr, row_out);
  }

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  if (_sp_rec) {
    _sp_t1 = __builtin_amdgcn_s_memrealtime();
  }
#endif
  // ── Tile dispatch ───────────────────────────────────────────────────────
  int tok_idx = tile_idx / n_wgs_per_xcd;
  int wg_idx = tile_idx % n_wgs_per_xcd;

  if (tok_idx >= batch_count) {
    return;
  }

  // Workgroup weight pointers
  uint8_t const *wg_data = W + static_cast<int64_t>(wg_idx) * WG_BYTES;
  uint8_t const *wg_scales = wg_data + WG_DATA_BYTES;

  // ── Step 2: Quantize BF16 normalized input -> FP4/FP8 in LDS ────────────
  unsigned short const *input_row =
      (unsigned short const *)norm_output_ptr + tok_idx * REDUCTION_SIZE;

  _gang_wave_parallel_fp8_quant<REDUCTION_SIZE>(
      input_row, s_tok_fp8, s_tok_scales);

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  if (_sp_rec) {
    _sp_t2 = __builtin_amdgcn_s_memrealtime();
  }
#endif
  // ── Step 3: MFMA FP4(weights) x FP4/FP8/BF16(tokens) ──────────────────
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

// IMPORTANT: #pragma unroll 1 prevents ROCm miscompilation.
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

        // Slot 2: compute k-tile ki+2, prefetch ki+6
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

        // Slot 3: compute k-tile ki+3, prefetch ki+7
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

      // ── Step 4: Bias epilogue, write BF16 output ─────────────────────────
      if (col == 0) {
        for (int i = 0; i < 4; i++) {
          int out_n = wg_idx * OUTPUT_PER_WG + wave_tile * 16 + g * 4 + i;
          float sum = acc[i];

          // Add bias (partitioned per XCD)
          unsigned bt = (unsigned)d_bias[out_n] << 16;
          float bv;
          __builtin_memcpy(&bv, &bt, 4);

          int out_idx = tok_idx * output_stride + out_n;
          d_output[out_idx] = _gang_float_to_bf16(sum + bv);
        }
      }
    }
  } else {
    // K-parallel: 4 waves all process same 16 rows, split K across waves.
    // Depth-4 pipelined: 4 weight K-tiles pre-loaded, compute overlaps with
    // prefetch of next 4. Each MFMA ~32 cycles; 4 in flight = ~128 cycles
    // of latency hiding.
    constexpr int TOTAL_K_ITERS = MFMA_ITERS;
    constexpr int ITERS_PER_WAVE = TOTAL_K_ITERS / NUM_WAVES;
    static_assert(TOTAL_K_ITERS % NUM_WAVES == 0,
                  "MFMA_ITERS must be divisible by NUM_WAVES for K-parallel");
    static_assert(ITERS_PER_WAVE >= 4,
                  "Depth-4 K-parallel requires ITERS_PER_WAVE >= 4");

    int const ki_start = warp_id * ITERS_PER_WAVE;
    int const ki_end = ki_start + ITERS_PER_WAVE;

    int w_row = col; // All 4 waves process same 16 output rows
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

    // Cross-wave LDS reduction (reuse token scratch area, dead after MFMA)
    float *lds_reduce = (float *)_rnlm_smem;
    if (col == 0) {
      for (int i = 0; i < 4; i++) {
        lds_reduce[warp_id * OUTPUT_PER_WG + g * 4 + i] = acc[i];
      }
    }
    __syncthreads();

    // Wave 0 reduces across waves and writes output with bias
    if (warp_id == 0 && col == 0) {
      for (int i = 0; i < 4; i++) {
        float v = 0.0f;
        for (int w = 0; w < NUM_WAVES; w++) {
          v += lds_reduce[w * OUTPUT_PER_WG + g * 4 + i];
        }

        int out_n = wg_idx * OUTPUT_PER_WG + g * 4 + i;

        unsigned bt = (unsigned)d_bias[out_n] << 16;
        float bv;
        __builtin_memcpy(&bv, &bt, 4);

        int out_idx = tok_idx * output_stride + out_n;
        d_output[out_idx] = _gang_float_to_bf16(v + bv);
      }
    }
  }

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  if (_sp_rec) {
    _sp_t3 = __builtin_amdgcn_s_memrealtime();
    // Slot 0: QKV. [1]=RMSNorm [2]=FP8Quant [3]=MFMA+Epi
    atomicAdd(&g_subphase_ns[0][1], (_sp_t1 - _sp_t0) * 10);
    atomicAdd(&g_subphase_ns[0][2], (_sp_t2 - _sp_t1) * 10);
    atomicAdd(&g_subphase_ns[0][3], (_sp_t3 - _sp_t2) * 10);
    atomicAdd(&g_subphase_cnt[0], 1ULL);
  }
#endif
  __syncthreads();
}

// Fused MulSumAdd + RMSNorm + MXFP4 Gang Linear + Bias.
//
// Merges the standalone MulSumAdd task into the QKV kernel's prologue:
//   Step 0: x_output = residual + sum_k(mlp_out[k] * routing_weight[k])
//   Step 1: RMSNorm(x_output) → norm_scratch
//   Steps 2-4: FP8 quantize → MFMA → bias epilogue (unchanged)
//
// Eliminates one task dispatch + two scheduling gaps (~10 µs/layer).
// Used for layers 1+ where a preceding MoE block provides mlp_out.
template <int BATCH_SIZE,
          int OUTPUT_PER_WG,
          int REDUCTION_SIZE,
          int ACTUAL_HIDDEN_DIM = REDUCTION_SIZE,
          int NUM_TOPK = 4,
          int INPUT_STRIDE = REDUCTION_SIZE>
__device__ __noinline__ void gang_mulsumradd_rmsnorm_linear_mxfp4_bias_kernel(
    void const *mlp_out_ptr,        // [batch, topk, INPUT_STRIDE] bf16
    void const *routing_weight_ptr, // [batch, topk] f32
    void const *residual_ptr,       // [batch, REDUCTION_SIZE] bf16
    void const *norm_weight_ptr,    // [REDUCTION_SIZE] bf16
    void *norm_scratch_ptr, // [batch, REDUCTION_SIZE] bf16 (RMSNorm scratch)
    void const *weight_ptr, // [n_wgs_per_xcd, wg_bytes] packed MXFP4
    void const *bias_ptr,   // [1, output_size_per_xcd] bf16 (partitioned)
    void *x_output_ptr,     // [batch, REDUCTION_SIZE] bf16 (MulSumAdd result)
    void *qkv_output_ptr,   // [batch, output_stride] bf16 (partitioned)
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
  constexpr int BF16_MFMA_ITERS = REDUCTION_SIZE / 32;

  // ── Wave tiling ─────────────────────────────────────────────────────────
  constexpr int NUM_WAVES = 4;
  constexpr int TILES_PER_WAVE = OUTPUT_PER_WG / 16 / NUM_WAVES;

  // ── Token activation in shared memory ────────────────────────────────────
  constexpr int FP8_TOK_DATA = REDUCTION_SIZE;

  uint8_t const *W = (uint8_t const *)weight_ptr;
  unsigned short const *d_bias = (unsigned short const *)bias_ptr;
  unsigned short *d_qkv_output = (unsigned short *)qkv_output_ptr;

  extern __shared__ char _rnlm_smem[];
  uint8_t *s_tok_fp8 = (uint8_t *)_rnlm_smem;
  uint8_t *s_tok_scales = s_tok_fp8 + FP8_TOK_DATA;

  int const tid = threadIdx.x;
  int const warp_id = tid >> 6;
  int const lane_id = tid & 63;
  int const col = lane_id & 15;
  int const g = lane_id >> 4;

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  unsigned long long _sp_t0 = 0, _sp_t1 = 0, _sp_t2 = 0, _sp_t3 = 0,
                     _sp_t3b = 0, _sp_t4 = 0;
  bool _sp_rec = (tile_idx == 0 && tid == 0 && g_subphase_active);
  if (_sp_rec) {
    _sp_t0 = __builtin_amdgcn_s_memrealtime();
  }
#endif
  int batch_count =
      (num_active_tokens < BATCH_SIZE) ? num_active_tokens : BATCH_SIZE;

  // ── Step 0+1 FUSED: MulSumAdd + RMSNorm ─────────────────────────────
  // Pass 1: weighted-sum + residual-add + SSQ accumulation (one loop).
  //         Uses inline-asm v_fmac to keep SSQ in the inner loop body
  //         (without this the compiler hoists SSQ into a separate pass).
  // Pass 2: apply norm weight, vectorized 4×bf16 loads/stores.
  {
    unsigned short const *d_mlp_out = (unsigned short const *)mlp_out_ptr;
    float const *d_rw = (float const *)routing_weight_ptr;
    unsigned short const *d_residual = (unsigned short const *)residual_ptr;
    unsigned short *d_x_out = (unsigned short *)x_output_ptr;
    unsigned short const *d_norm_w = (unsigned short const *)norm_weight_ptr;
    unsigned short *d_norm_out = (unsigned short *)norm_scratch_ptr;

    for (int b = 0; b < batch_count; b++) {
      // Load routing weights for this token
      float rw[NUM_TOPK];
#pragma unroll
      for (int k = 0; k < NUM_TOPK; k++) {
        rw[k] = d_rw[b * NUM_TOPK + k];
      }

      // Pass 1: Fused weighted-sum + residual-add + SSQ accumulation
      float ssq = 0.0f;
      for (int i = tid; i < REDUCTION_SIZE; i += blockDim.x) {
        unsigned r_bits = (unsigned)d_residual[b * REDUCTION_SIZE + i] << 16;
        float sum;
        __builtin_memcpy(&sum, &r_bits, 4);

#pragma unroll
        for (int k = 0; k < NUM_TOPK; k++) {
          unsigned v_bits =
              (unsigned)
                  d_mlp_out[b * INPUT_STRIDE * NUM_TOPK + k * INPUT_STRIDE + i]
              << 16;
          float val;
          __builtin_memcpy(&val, &v_bits, 4);
          sum += val * rw[k];
        }

        // Write residual-added result for next layer
        d_x_out[b * REDUCTION_SIZE + i] = _gang_float_to_bf16(sum);
        // Accumulate SSQ in f32 (no bf16 round-trip)
        ssq += sum * sum;
      }

// Wavefront reduction (AMD wavefront = 64 lanes)
#pragma unroll
      for (int offset = 32; offset > 0; offset >>= 1) {
        ssq += __shfl_xor(ssq, offset);
      }

      // Cross-wave reduction via shared memory (reuse FP8 area, dead here)
      float *s_red = (float *)_rnlm_smem;
      int _wave_id = tid >> 6;
      int _lane_id = tid & 63;
      int _num_waves = blockDim.x >> 6;
      if (_lane_id == 0) {
        s_red[_wave_id] = ssq;
      }
      __syncthreads();

      float rms_rcp;
      if (_wave_id == 0) {
        ssq = (_lane_id < _num_waves) ? s_red[_lane_id] : 0.0f;
        for (int offset = _num_waves >> 1; offset > 0; offset >>= 1) {
          ssq += __shfl_xor(ssq, offset);
        }
        if (_lane_id == 0) {
          s_red[0] = rsqrtf(ssq / (float)ACTUAL_HIDDEN_DIM + 1e-5f);
        }
      }
      __syncthreads();
      rms_rcp = s_red[0];

      // Pass 2: Apply norm weight + write to norm scratch (vectorized 4×bf16)
      {
        using bf16 = __hip_bfloat16;
        constexpr int VEC = 4; // 4 bf16 per 64-bit load
        bf16 const *x_in = (bf16 const *)d_x_out + b * REDUCTION_SIZE;
        bf16 const *w_in = (bf16 const *)d_norm_w;
        bf16 *out = (bf16 *)d_norm_out + b * REDUCTION_SIZE;
        constexpr int VEC_ITERS = REDUCTION_SIZE / (VEC);
        for (int vi = tid; vi < VEC_ITERS; vi += blockDim.x) {
          int off = vi * VEC;
          uint64_t xv, wv;
          __builtin_memcpy(&xv, &x_in[off], 8);
          __builtin_memcpy(&wv, &w_in[off], 8);
          bf16 const *xa = (bf16 const *)&xv;
          bf16 const *wa = (bf16 const *)&wv;
          bf16 ov[VEC];
#pragma unroll
          for (int j = 0; j < VEC; j++) {
            ov[j] = __float2bfloat16(__bfloat162float(xa[j]) * rms_rcp *
                                     __bfloat162float(wa[j]));
          }
          __builtin_memcpy(&out[off], ov, 8);
        }
      }
    }
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  __syncthreads();

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  // _sp_t1 now captures fused MulSumAdd+RMSNorm (was two separate steps)
  if (_sp_rec) {
    _sp_t1 = __builtin_amdgcn_s_memrealtime();
    _sp_t2 = _sp_t1;
  }
#endif
  // ── Tile dispatch ───────────────────────────────────────────────────────
  int tok_idx = tile_idx / n_wgs_per_xcd;
  int wg_idx = tile_idx % n_wgs_per_xcd;

  if (tok_idx >= batch_count) {
    return;
  }

  // Workgroup weight pointers
  uint8_t const *wg_data = W + static_cast<int64_t>(wg_idx) * WG_BYTES;
  uint8_t const *wg_scales = wg_data + WG_DATA_BYTES;

  // ── Step 2: Quantize/copy BF16 normalized input to LDS ──────────────────
  unsigned short const *input_row =
      (unsigned short const *)norm_scratch_ptr + tok_idx * REDUCTION_SIZE;
  _gang_wave_parallel_fp8_quant<REDUCTION_SIZE>(
      input_row, s_tok_fp8, s_tok_scales);

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  if (_sp_rec) {
    _sp_t3 = __builtin_amdgcn_s_memrealtime();
  }
#endif
  // ── Step 3: MFMA FP4(weights) x FP4/FP8/BF16(tokens) ──────────────────
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

      // ── Step 4: Bias epilogue, write BF16 output ─────────────────────────
      if (col == 0) {
        for (int i = 0; i < 4; i++) {
          int out_n = wg_idx * OUTPUT_PER_WG + wave_tile * 16 + g * 4 + i;
          float sum = acc[i];

          unsigned bt = (unsigned)d_bias[out_n] << 16;
          float bv;
          __builtin_memcpy(&bv, &bt, 4);

          int out_idx = tok_idx * output_stride + out_n;
          d_qkv_output[out_idx] = _gang_float_to_bf16(sum + bv);
        }
      }
    }
  } else {
    // K-parallel: 4 waves all process same 16 rows, split K across waves.
    // Depth-4 pipelined: 4 weight K-tiles pre-loaded, compute overlaps with
    // prefetch of next 4.
    constexpr int TOTAL_K_ITERS = MFMA_ITERS;
    constexpr int ITERS_PER_WAVE = TOTAL_K_ITERS / NUM_WAVES;
    static_assert(TOTAL_K_ITERS % NUM_WAVES == 0,
                  "MFMA_ITERS must be divisible by NUM_WAVES for K-parallel");
    static_assert(ITERS_PER_WAVE >= 4,
                  "Depth-4 K-parallel requires ITERS_PER_WAVE >= 4");

    int const ki_start = warp_id * ITERS_PER_WAVE;
    int const ki_end = ki_start + ITERS_PER_WAVE;

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

#pragma unroll 1
    for (int ki = ki_start; ki < ki_end; ki += 4) {
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

    float *lds_reduce = (float *)_rnlm_smem;
    if (col == 0) {
      for (int i = 0; i < 4; i++) {
        lds_reduce[warp_id * OUTPUT_PER_WG + g * 4 + i] = acc[i];
      }
    }
    __syncthreads();

    if (warp_id == 0 && col == 0) {
      for (int i = 0; i < 4; i++) {
        float v = 0.0f;
        for (int w = 0; w < NUM_WAVES; w++) {
          v += lds_reduce[w * OUTPUT_PER_WG + g * 4 + i];
        }

        int out_n = wg_idx * OUTPUT_PER_WG + g * 4 + i;

        unsigned bt = (unsigned)d_bias[out_n] << 16;
        float bv;
        __builtin_memcpy(&bv, &bt, 4);

        int out_idx = tok_idx * output_stride + out_n;
        d_qkv_output[out_idx] = _gang_float_to_bf16(v + bv);
      }
    }
  }

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  if (_sp_rec) {
    _sp_t4 = __builtin_amdgcn_s_memrealtime();
    // Slot 0: QKV. [0]=Fused MulSumAdd+RMSNorm [1]=0 (fused) [2]=FP8Quant
    // [3]=MFMA+Epi
    atomicAdd(&g_subphase_ns[0][0], (_sp_t1 - _sp_t0) * 10);
    atomicAdd(&g_subphase_ns[0][1], (_sp_t2 - _sp_t1) * 10);
    atomicAdd(&g_subphase_ns[0][2], (_sp_t3 - _sp_t2) * 10);
    atomicAdd(&g_subphase_ns[0][3], (_sp_t4 - _sp_t3) * 10);
    atomicAdd(&g_subphase_cnt[0], 1ULL);
  }
#endif
  __syncthreads();
}

// =========================================================================
// Fused RMSNorm + MXFP4 Gang Linear + KV Cache Update (RoPE + paged write).
//
// Eliminates the separate kv_cache_update task by fusing it into the QKV
// MFMA epilogue.  Each XCD handles exactly one KV head group:
//   wg_idx 0 .. NUM_Q_PER_KV-1  →  Q heads  →  RoPE → q_workspace
//   wg_idx NUM_Q_PER_KV          →  K head   →  RoPE → paged K cache
//   wg_idx NUM_Q_PER_KV+1        →  V head   →  paged V cache (no RoPE)
//
// RoPE cross-wave communication uses LDS (HEAD_DIM bf16 values reusing FP8
// token area that is dead after MFMA).
// =========================================================================

// Local XCD ID helper (avoids include-order dependency on
// persistent_kernel.cuh)
__device__ __forceinline__ int _kvupd_get_xcd_id() {
#if defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300)
  int xcd_id;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcd_id));
  return xcd_id;
#else
  return 0;
#endif
}

// ── Inline RoPE epilogue helper ─────────────────────────────────────────
// Writes MFMA accumulator to LDS, applies RoPE via cross-wave communication,
// then writes the result to `dst`.
// Call from within the tile_iter loop with all threads participating.
template <int HEAD_DIM>
__device__ __forceinline__ void _kvupd_rope_epilogue(
    float const *acc,               // [4] accumulator values for this lane
    unsigned short const *bias,     // bias array (partitioned per XCD)
    int wg_idx,                     // workgroup index within XCD
    int wave_tile,                  // wave_tile index
    int g,                          // lane group (0..3)
    int col,                        // lane column (0..15)
    int tid,                        // threadIdx.x
    unsigned short const *cos_data, // cos[HEAD_DIM] bf16
    unsigned short const *sin_data, // sin[HEAD_DIM] bf16
    unsigned short *dst,            // destination array
    int dst_stride,                 // stride between tokens in dst
    int tok_idx,                    // token index
    int head_offset,                // head offset within dst row
    int output_per_wg,              // OUTPUT_PER_WG
    unsigned short *s_rope)         // LDS buffer [HEAD_DIM] bf16
{
  // Step 1: col==0 lanes write biased values to LDS
  if (col == 0) {
    int d0 = wave_tile * 16 + g * 4;
    // Batch-load 4 contiguous bf16 bias values as one 64-bit load
    uint64_t bias4;
    __builtin_memcpy(&bias4, &bias[wg_idx * output_per_wg + d0], 8);
    unsigned short const *b4 = (unsigned short const *)&bias4;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      unsigned bt = (unsigned)b4[i] << 16;
      float bv;
      __builtin_memcpy(&bv, &bt, 4);
      s_rope[d0 + i] = _gang_float_to_bf16(acc[i] + bv);
    }
  }
  __syncthreads();

  // Step 2: 32 threads apply RoPE (element d paired with d + HEAD_DIM/2)
  constexpr int HALF = HEAD_DIM / 2;
  if (tid < HALF) {
    int d = tid;
    unsigned v0_bits = (unsigned)s_rope[d] << 16;
    unsigned v1_bits = (unsigned)s_rope[d + HALF] << 16;
    float v0, v1;
    __builtin_memcpy(&v0, &v0_bits, 4);
    __builtin_memcpy(&v1, &v1_bits, 4);

    unsigned c_bits = (unsigned)cos_data[d] << 16;
    unsigned s_bits = (unsigned)sin_data[d] << 16;
    float c, s;
    __builtin_memcpy(&c, &c_bits, 4);
    __builtin_memcpy(&s, &s_bits, 4);

    s_rope[d] = _gang_float_to_bf16(v0 * c - v1 * s);
    s_rope[d + HALF] = _gang_float_to_bf16(v0 * s + v1 * c);
  }
  __syncthreads();

  // Step 3: Write HEAD_DIM bf16 values from LDS to destination
  for (int d = tid; d < HEAD_DIM; d += 256 /*blockDim.x*/) {
    dst[tok_idx * dst_stride + head_offset + d] = s_rope[d];
  }
}

// ── Layer 0 variant (no MulSumAdd) ─────────────────────────────────────
template <int BATCH_SIZE,
          int OUTPUT_PER_WG,
          int REDUCTION_SIZE,
          int ACTUAL_HIDDEN_DIM,
          int HEAD_DIM,
          int NUM_Q_PER_KV,
          int PAGE_SIZE>
__device__ __noinline__ void gang_rmsnorm_linear_mxfp4_bias_kvupd_kernel(
    void const *norm_input_ptr,  // [batch, REDUCTION_SIZE] bf16
    void const *norm_weight_ptr, // [REDUCTION_SIZE] bf16
    void *norm_output_ptr,       // [batch, REDUCTION_SIZE] bf16 scratch
    void const *weight_ptr,      // [n_wgs_per_xcd, wg_bytes] packed MXFP4
    void const *bias_ptr,        // [1, output_size_per_xcd] bf16 (partitioned)
    void *k_cache_ptr,           // paged K cache (un-partitioned base)
    void *v_cache_ptr,           // paged V cache (un-partitioned base)
    void *q_workspace_ptr,       // [batch, q_ws_stride] bf16
    void const *cos_ptr,         // [max_seq_len, HEAD_DIM] bf16
    void const *sin_ptr,         // [max_seq_len, HEAD_DIM] bf16
    int const *qo_indptr,
    int const *kv_indptr,
    int const *kv_indices,
    int const *kv_last_page_len,
    int num_active_tokens,
    int n_wgs_per_xcd,
    int kv_stride,   // HEAD_DIM * num_kv_heads
    int q_ws_stride, // num_q_heads * HEAD_DIM
    int tile_idx) {

  static_assert(OUTPUT_PER_WG % 16 == 0);
  static_assert(REDUCTION_SIZE % 128 == 0);
  static_assert(OUTPUT_PER_WG == HEAD_DIM,
                "KV_UPD fusion requires OUTPUT_PER_WG == HEAD_DIM");

  constexpr int NUM_BLOCKS_32 = REDUCTION_SIZE / 32;
  constexpr int WG_DATA_BYTES = OUTPUT_PER_WG * (REDUCTION_SIZE / 2);
  constexpr int WG_SCALE_BYTES = OUTPUT_PER_WG * NUM_BLOCKS_32;
  constexpr int WG_BYTES = WG_DATA_BYTES + WG_SCALE_BYTES;
  constexpr int K_PER_MFMA = 128;
  constexpr int MFMA_ITERS = REDUCTION_SIZE / K_PER_MFMA;
  constexpr int NUM_WAVES = 4;
  constexpr int TILES_PER_WAVE = OUTPUT_PER_WG / 16 / NUM_WAVES;
  constexpr int FP8_TOK_DATA = REDUCTION_SIZE;

  uint8_t const *W = (uint8_t const *)weight_ptr;
  unsigned short const *d_bias = (unsigned short const *)bias_ptr;

  extern __shared__ char _rnlm_smem[];
  uint8_t *s_tok_fp8 = (uint8_t *)_rnlm_smem;
  uint8_t *s_tok_scales = s_tok_fp8 + FP8_TOK_DATA;

  int const tid = threadIdx.x;
  int const warp_id = tid >> 6;
  int const lane_id = tid & 63;
  int const col = lane_id & 15;
  int const g = lane_id >> 4;

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  unsigned long long _sp_t0 = 0, _sp_t1 = 0, _sp_t2 = 0, _sp_t3 = 0;
  bool _sp_rec = (tile_idx == 0 && tid == 0 && g_subphase_active);
  if (_sp_rec) {
    _sp_t0 = __builtin_amdgcn_s_memrealtime();
  }
#endif
  // ── Step 1: Redundant RMSNorm ─────────────────────────────────────────
  int batch_count =
      (num_active_tokens < BATCH_SIZE) ? num_active_tokens : BATCH_SIZE;
  for (int b = 0; b < batch_count; b++) {
    unsigned short const *row_in =
        (unsigned short const *)norm_input_ptr + b * REDUCTION_SIZE;
    unsigned short *row_out =
        (unsigned short *)norm_output_ptr + b * REDUCTION_SIZE;
    gang_rmsnorm_detail::rmsnorm_inline_amd<REDUCTION_SIZE, ACTUAL_HIDDEN_DIM>(
        row_in, norm_weight_ptr, row_out);
  }

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  if (_sp_rec) {
    _sp_t1 = __builtin_amdgcn_s_memrealtime();
  }
#endif
  // ── Tile dispatch ─────────────────────────────────────────────────────
  int tok_idx = tile_idx / n_wgs_per_xcd;
  int wg_idx = tile_idx % n_wgs_per_xcd;
  if (tok_idx >= batch_count) {
    return;
  }

  uint8_t const *wg_data = W + static_cast<int64_t>(wg_idx) * WG_BYTES;
  uint8_t const *wg_scales = wg_data + WG_DATA_BYTES;

  // ── Step 2: Quantize/copy normalized input to LDS ────────────────────
  {
    unsigned short const *input_row =
        (unsigned short const *)norm_output_ptr + tok_idx * REDUCTION_SIZE;
    _gang_wave_parallel_fp8_quant<REDUCTION_SIZE>(
        input_row, s_tok_fp8, s_tok_scales);
  }

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  if (_sp_rec) {
    _sp_t2 = __builtin_amdgcn_s_memrealtime();
  }
#endif
  // ── Step 3: MFMA ───────────────────────────────────────────────────
  for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; tile_iter++) {
    int wave_tile = warp_id + tile_iter * NUM_WAVES;
    int w_row = wave_tile * 16 + col;
    int const row_data_base = w_row * (REDUCTION_SIZE / 2);
    int const row_scale_base = w_row * NUM_BLOCKS_32;

    f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

    i32x8_t a0 = *(i32x8_t const *)(wg_data + row_data_base + 0 * 64 + g * 16);
    int sa0 = (int)wg_scales[row_scale_base + 0 * 4 + g];
    i32x8_t a1 = *(i32x8_t const *)(wg_data + row_data_base + 1 * 64 + g * 16);
    int sa1 = (int)wg_scales[row_scale_base + 1 * 4 + g];
    i32x8_t a2 = *(i32x8_t const *)(wg_data + row_data_base + 2 * 64 + g * 16);
    int sa2 = (int)wg_scales[row_scale_base + 2 * 4 + g];
    i32x8_t a3 = *(i32x8_t const *)(wg_data + row_data_base + 3 * 64 + g * 16);
    int sa3 = (int)wg_scales[row_scale_base + 3 * 4 + g];

#pragma unroll 1
    for (int ki = 0; ki < MFMA_ITERS; ki += 4) {
      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, ki * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki];
        acc = _gang_mfma_f4xf8(a0, b, acc, sa0, sb);
      }
      if (ki + 4 < MFMA_ITERS) {
        int kt = (ki + 4) * K_PER_MFMA;
        a0 = *(i32x8_t const *)(wg_data + row_data_base + kt / 2 + g * 16);
        sa0 = (int)wg_scales[row_scale_base + kt / 32 + g];
      }
      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 1) * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki + 1];
        acc = _gang_mfma_f4xf8(a1, b, acc, sa1, sb);
      }
      if (ki + 5 < MFMA_ITERS) {
        int kt = (ki + 5) * K_PER_MFMA;
        a1 = *(i32x8_t const *)(wg_data + row_data_base + kt / 2 + g * 16);
        sa1 = (int)wg_scales[row_scale_base + kt / 32 + g];
      }
      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 2) * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki + 2];
        acc = _gang_mfma_f4xf8(a2, b, acc, sa2, sb);
      }
      if (ki + 6 < MFMA_ITERS) {
        int kt = (ki + 6) * K_PER_MFMA;
        a2 = *(i32x8_t const *)(wg_data + row_data_base + kt / 2 + g * 16);
        sa2 = (int)wg_scales[row_scale_base + kt / 32 + g];
      }
      if (ki + 3 < MFMA_ITERS) {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 3) * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki + 3];
        acc = _gang_mfma_f4xf8(a3, b, acc, sa3, sb);
      }
      if (ki + 7 < MFMA_ITERS) {
        int kt = (ki + 7) * K_PER_MFMA;
        a3 = *(i32x8_t const *)(wg_data + row_data_base + kt / 2 + g * 16);
        sa3 = (int)wg_scales[row_scale_base + kt / 32 + g];
      }
    }
    // ── Step 4: Fused KV_UPD epilogue ──────────────────────────────────
    int kv_head = _kvupd_get_xcd_id();
    int request_id = 0;
    while (qo_indptr[request_id + 1] <= tok_idx) {
      request_id++;
    }
    int token_in_request = tok_idx - qo_indptr[request_id];
    int first_page = kv_indptr[request_id];
    int num_pages = kv_indptr[request_id + 1] - first_page;
    int last_page_len_val = kv_last_page_len[request_id];
    int global_seq_len = (num_pages - 1) * PAGE_SIZE + last_page_len_val;
    int num_new_tokens = qo_indptr[request_id + 1] - qo_indptr[request_id];
    int global_pos = global_seq_len - num_new_tokens + token_in_request;
    unsigned short const *cos_row =
        (unsigned short const *)cos_ptr + global_pos * HEAD_DIM;
    unsigned short const *sin_row =
        (unsigned short const *)sin_ptr + global_pos * HEAD_DIM;
    unsigned short *s_rope = (unsigned short *)_rnlm_smem;
    if (wg_idx < NUM_Q_PER_KV) {
      int q_head_global = kv_head * NUM_Q_PER_KV + wg_idx;
      _kvupd_rope_epilogue<HEAD_DIM>((float const *)&acc,
                                     d_bias,
                                     wg_idx,
                                     wave_tile,
                                     g,
                                     col,
                                     tid,
                                     cos_row,
                                     sin_row,
                                     (unsigned short *)q_workspace_ptr,
                                     q_ws_stride,
                                     tok_idx,
                                     q_head_global * HEAD_DIM,
                                     OUTPUT_PER_WG,
                                     s_rope);
    } else if (wg_idx == NUM_Q_PER_KV) {
      int page_num = global_pos / PAGE_SIZE;
      int page_offset = global_pos % PAGE_SIZE;
      int page_idx = kv_indices[first_page + page_num];
      int dst_idx = page_idx * PAGE_SIZE + page_offset;
      if (col == 0) {
        int d0 = wave_tile * 16 + g * 4;
        uint64_t bias4;
        __builtin_memcpy(&bias4, &d_bias[wg_idx * OUTPUT_PER_WG + d0], 8);
        unsigned short const *b4 = (unsigned short const *)&bias4;
#pragma unroll
        for (int i = 0; i < 4; i++) {
          unsigned bt = (unsigned)b4[i] << 16;
          float bv;
          __builtin_memcpy(&bv, &bt, 4);
          s_rope[d0 + i] = _gang_float_to_bf16(acc[i] + bv);
        }
      }
      __syncthreads();
      constexpr int HALF = HEAD_DIM / 2;
      if (tid < HALF) {
        unsigned v0b = (unsigned)s_rope[tid] << 16;
        unsigned v1b = (unsigned)s_rope[tid + HALF] << 16;
        float v0, v1;
        __builtin_memcpy(&v0, &v0b, 4);
        __builtin_memcpy(&v1, &v1b, 4);
        unsigned cb = (unsigned)cos_row[tid] << 16;
        unsigned sb = (unsigned)sin_row[tid] << 16;
        float c, s;
        __builtin_memcpy(&c, &cb, 4);
        __builtin_memcpy(&s, &sb, 4);
        s_rope[tid] = _gang_float_to_bf16(v0 * c - v1 * s);
        s_rope[tid + HALF] = _gang_float_to_bf16(v0 * s + v1 * c);
      }
      __syncthreads();
      unsigned short *d_k = (unsigned short *)k_cache_ptr;
      for (int d = tid; d < HEAD_DIM; d += 256) {
        d_k[dst_idx * kv_stride + kv_head * HEAD_DIM + d] = s_rope[d];
      }
    } else {
      int page_num = global_pos / PAGE_SIZE;
      int page_offset = global_pos % PAGE_SIZE;
      int page_idx = kv_indices[first_page + page_num];
      int dst_idx = page_idx * PAGE_SIZE + page_offset;
      unsigned short *d_v = (unsigned short *)v_cache_ptr;
      if (col == 0) {
        int d0 = wave_tile * 16 + g * 4;
        uint64_t bias4;
        __builtin_memcpy(&bias4, &d_bias[wg_idx * OUTPUT_PER_WG + d0], 8);
        unsigned short const *b4 = (unsigned short const *)&bias4;
#pragma unroll
        for (int i = 0; i < 4; i++) {
          unsigned bt = (unsigned)b4[i] << 16;
          float bv;
          __builtin_memcpy(&bv, &bt, 4);
          d_v[dst_idx * kv_stride + kv_head * HEAD_DIM + d0 + i] =
              _gang_float_to_bf16(acc[i] + bv);
        }
      }
    }
  }

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  if (_sp_rec) {
    _sp_t3 = __builtin_amdgcn_s_memrealtime();
    // Slot 1: QKV_KVUPD. [1]=RMSNorm [2]=FP8Quant [3]=MFMA
    atomicAdd(&g_subphase_ns[1][1], (_sp_t1 - _sp_t0) * 10);
    atomicAdd(&g_subphase_ns[1][2], (_sp_t2 - _sp_t1) * 10);
    atomicAdd(&g_subphase_ns[1][3], (_sp_t3 - _sp_t2) * 10);
    atomicAdd(&g_subphase_cnt[1], 1ULL);
  }
#endif
  __syncthreads();
}

// ── Layers 1+ variant (MulSumAdd + RMSNorm + MXFP4 + KV_UPD) ──────────
template <int BATCH_SIZE,
          int OUTPUT_PER_WG,
          int REDUCTION_SIZE,
          int ACTUAL_HIDDEN_DIM,
          int HEAD_DIM,
          int NUM_Q_PER_KV,
          int PAGE_SIZE,
          int NUM_TOPK = 4,
          int INPUT_STRIDE = REDUCTION_SIZE>
__device__ __noinline__ void
    gang_mulsumradd_rmsnorm_linear_mxfp4_bias_kvupd_kernel(
        void const *mlp_out_ptr,
        void const *routing_weight_ptr,
        void const *residual_ptr,
        void const *norm_weight_ptr,
        void *norm_scratch_ptr,
        void const *weight_ptr,
        void const *bias_ptr,
        void *x_output_ptr,
        void *k_cache_ptr,
        void *v_cache_ptr,
        void *q_workspace_ptr,
        void const *cos_ptr,
        void const *sin_ptr,
        int const *qo_indptr,
        int const *kv_indptr,
        int const *kv_indices,
        int const *kv_last_page_len,
        int num_active_tokens,
        int n_wgs_per_xcd,
        int kv_stride,
        int q_ws_stride,
        int tile_idx) {

  static_assert(OUTPUT_PER_WG % 16 == 0);
  static_assert(REDUCTION_SIZE % 128 == 0);
  static_assert(OUTPUT_PER_WG == HEAD_DIM,
                "KV_UPD fusion requires OUTPUT_PER_WG == HEAD_DIM");

  constexpr int NUM_BLOCKS_32 = REDUCTION_SIZE / 32;
  constexpr int WG_DATA_BYTES = OUTPUT_PER_WG * (REDUCTION_SIZE / 2);
  constexpr int WG_SCALE_BYTES = OUTPUT_PER_WG * NUM_BLOCKS_32;
  constexpr int WG_BYTES = WG_DATA_BYTES + WG_SCALE_BYTES;
  constexpr int K_PER_MFMA = 128;
  constexpr int MFMA_ITERS = REDUCTION_SIZE / K_PER_MFMA;
  constexpr int NUM_WAVES = 4;
  constexpr int TILES_PER_WAVE = OUTPUT_PER_WG / 16 / NUM_WAVES;
  constexpr int FP8_TOK_DATA = REDUCTION_SIZE;

  uint8_t const *W = (uint8_t const *)weight_ptr;
  unsigned short const *d_bias = (unsigned short const *)bias_ptr;

  extern __shared__ char _rnlm_smem[];
  uint8_t *s_tok_fp8 = (uint8_t *)_rnlm_smem;
  uint8_t *s_tok_scales = s_tok_fp8 + FP8_TOK_DATA;

  int const tid = threadIdx.x;
  int const warp_id = tid >> 6;
  int const lane_id = tid & 63;
  int const col = lane_id & 15;
  int const g = lane_id >> 4;

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  unsigned long long _sp_t0 = 0, _sp_t1 = 0, _sp_t2 = 0, _sp_t3 = 0,
                     _sp_t3b = 0, _sp_t4 = 0;
  bool _sp_rec = (tile_idx == 0 && tid == 0 && g_subphase_active);
  if (_sp_rec) {
    _sp_t0 = __builtin_amdgcn_s_memrealtime();
  }
#endif
  int batch_count =
      (num_active_tokens < BATCH_SIZE) ? num_active_tokens : BATCH_SIZE;

  // ── Step 0+1 FUSED: MulSumAdd + RMSNorm in single pass ───────────────
  // Computes x = residual + Σ(mlp_out[k] * w[k]) and accumulates SSQ for
  // RMSNorm in the same loop. Eliminates the intermediate bf16 global
  // memory round-trip (was ~4.5 µs MulSumAdd + ~3.2 µs RMSNorm = ~7.7 µs).
  {
    unsigned short const *d_mlp_out = (unsigned short const *)mlp_out_ptr;
    float const *d_rw = (float const *)routing_weight_ptr;
    unsigned short const *d_residual = (unsigned short const *)residual_ptr;
    unsigned short *d_x_out = (unsigned short *)x_output_ptr;
    unsigned short const *d_norm_w = (unsigned short const *)norm_weight_ptr;
    unsigned short *d_norm_out = (unsigned short *)norm_scratch_ptr;

    for (int b = 0; b < batch_count; b++) {
      float rw[NUM_TOPK];
#pragma unroll
      for (int k = 0; k < NUM_TOPK; k++) {
        rw[k] = d_rw[b * NUM_TOPK + k];
      }

      // Pass 1: Fused weighted-sum + residual-add + SSQ accumulation
      float ssq = 0.0f;
      for (int i = tid; i < REDUCTION_SIZE; i += blockDim.x) {
        unsigned r_bits = (unsigned)d_residual[b * REDUCTION_SIZE + i] << 16;
        float sum;
        __builtin_memcpy(&sum, &r_bits, 4);
#pragma unroll
        for (int k = 0; k < NUM_TOPK; k++) {
          unsigned v_bits =
              (unsigned)
                  d_mlp_out[b * INPUT_STRIDE * NUM_TOPK + k * INPUT_STRIDE + i]
              << 16;
          float val;
          __builtin_memcpy(&val, &v_bits, 4);
          sum += val * rw[k];
        }
        d_x_out[b * REDUCTION_SIZE + i] = _gang_float_to_bf16(sum);
        // Accumulate SSQ — inline asm prevents compiler from splitting
        // this into a separate re-read pass
        asm volatile("v_fmac_f32 %0, %1, %1" : "+v"(ssq) : "v"(sum));
      }

// Wavefront reduction (AMD wavefront = 64 lanes)
#pragma unroll
      for (int offset = 32; offset > 0; offset >>= 1) {
        ssq += __shfl_xor(ssq, offset);
      }

      // Cross-wave reduction via shared memory (reuse FP8 area, dead here)
      float *s_red = (float *)_rnlm_smem;
      int _wave_id = tid >> 6;
      int _lane_id = tid & 63;
      int _num_waves = blockDim.x >> 6;
      if (_lane_id == 0) {
        s_red[_wave_id] = ssq;
      }
      __syncthreads();

      float rms_rcp;
      if (_wave_id == 0) {
        ssq = (_lane_id < _num_waves) ? s_red[_lane_id] : 0.0f;
        for (int offset = _num_waves >> 1; offset > 0; offset >>= 1) {
          ssq += __shfl_xor(ssq, offset);
        }
        if (_lane_id == 0) {
          s_red[0] = rsqrtf(ssq / (float)ACTUAL_HIDDEN_DIM + 1e-5f);
        }
      }
      __syncthreads();
      rms_rcp = s_red[0];

      // Pass 2: Apply norm weight + write to norm scratch (or stash for fused
      // BF16 copy)
      {
        using bf16 = __hip_bfloat16;
        constexpr int VEC = 4; // 4 bf16 per 64-bit load
        bf16 const *x_in = (bf16 const *)d_x_out + b * REDUCTION_SIZE;
        bf16 const *w_in = (bf16 const *)d_norm_w;
        bf16 *out = (bf16 *)d_norm_out + b * REDUCTION_SIZE;
        constexpr int VEC_ITERS = REDUCTION_SIZE / (VEC);
        for (int vi = tid; vi < VEC_ITERS; vi += blockDim.x) {
          int off = vi * VEC;
          uint64_t xv, wv;
          __builtin_memcpy(&xv, &x_in[off], 8);
          __builtin_memcpy(&wv, &w_in[off], 8);
          bf16 const *xa = (bf16 const *)&xv;
          bf16 const *wa = (bf16 const *)&wv;
          bf16 ov[VEC];
#pragma unroll
          for (int j = 0; j < VEC; j++) {
            ov[j] = __float2bfloat16(__bfloat162float(xa[j]) * rms_rcp *
                                     __bfloat162float(wa[j]));
          }
          __builtin_memcpy(&out[off], ov, 8);
        }
      }
    }
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  __syncthreads();

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  if (_sp_rec) {
    _sp_t1 = __builtin_amdgcn_s_memrealtime();
    _sp_t2 = _sp_t1;
  }
#endif
  // ── Tile dispatch ─────────────────────────────────────────────────────
  int tok_idx = tile_idx / n_wgs_per_xcd;
  int wg_idx = tile_idx % n_wgs_per_xcd;
  if (tok_idx >= batch_count) {
    return;
  }

  uint8_t const *wg_data = W + static_cast<int64_t>(wg_idx) * WG_BYTES;
  uint8_t const *wg_scales = wg_data + WG_DATA_BYTES;

  // ── Step 2: Prepare activation in LDS ──────────────────────────────────
  {
    unsigned short const *input_row =
        (unsigned short const *)norm_scratch_ptr + tok_idx * REDUCTION_SIZE;
    _gang_wave_parallel_fp8_quant<REDUCTION_SIZE>(
        input_row, s_tok_fp8, s_tok_scales);
  }

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  if (_sp_rec) {
    _sp_t3 = __builtin_amdgcn_s_memrealtime();
  }
#endif
  // ── Step 3: MFMA ───────────────────────────────────────────────────
  for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; tile_iter++) {
    int wave_tile = warp_id + tile_iter * NUM_WAVES;
    int w_row = wave_tile * 16 + col;
    int const row_data_base = w_row * (REDUCTION_SIZE / 2);
    int const row_scale_base = w_row * NUM_BLOCKS_32;

    f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

    i32x8_t a0 = *(i32x8_t const *)(wg_data + row_data_base + 0 * 64 + g * 16);
    int sa0 = (int)wg_scales[row_scale_base + 0 * 4 + g];
    i32x8_t a1 = *(i32x8_t const *)(wg_data + row_data_base + 1 * 64 + g * 16);
    int sa1 = (int)wg_scales[row_scale_base + 1 * 4 + g];
    i32x8_t a2 = *(i32x8_t const *)(wg_data + row_data_base + 2 * 64 + g * 16);
    int sa2 = (int)wg_scales[row_scale_base + 2 * 4 + g];
    i32x8_t a3 = *(i32x8_t const *)(wg_data + row_data_base + 3 * 64 + g * 16);
    int sa3 = (int)wg_scales[row_scale_base + 3 * 4 + g];

#pragma unroll 1
    for (int ki = 0; ki < MFMA_ITERS; ki += 4) {
      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, ki * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki];
        acc = _gang_mfma_f4xf8(a0, b, acc, sa0, sb);
      }
      if (ki + 4 < MFMA_ITERS) {
        int kt = (ki + 4) * K_PER_MFMA;
        a0 = *(i32x8_t const *)(wg_data + row_data_base + kt / 2 + g * 16);
        sa0 = (int)wg_scales[row_scale_base + kt / 32 + g];
      }
      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 1) * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki + 1];
        acc = _gang_mfma_f4xf8(a1, b, acc, sa1, sb);
      }
      if (ki + 5 < MFMA_ITERS) {
        int kt = (ki + 5) * K_PER_MFMA;
        a1 = *(i32x8_t const *)(wg_data + row_data_base + kt / 2 + g * 16);
        sa1 = (int)wg_scales[row_scale_base + kt / 32 + g];
      }
      {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 2) * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki + 2];
        acc = _gang_mfma_f4xf8(a2, b, acc, sa2, sb);
      }
      if (ki + 6 < MFMA_ITERS) {
        int kt = (ki + 6) * K_PER_MFMA;
        a2 = *(i32x8_t const *)(wg_data + row_data_base + kt / 2 + g * 16);
        sa2 = (int)wg_scales[row_scale_base + kt / 32 + g];
      }
      if (ki + 3 < MFMA_ITERS) {
        i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 3) * K_PER_MFMA, g);
        int sb = (int)s_tok_scales[ki + 3];
        acc = _gang_mfma_f4xf8(a3, b, acc, sa3, sb);
      }
      if (ki + 7 < MFMA_ITERS) {
        int kt = (ki + 7) * K_PER_MFMA;
        a3 = *(i32x8_t const *)(wg_data + row_data_base + kt / 2 + g * 16);
        sa3 = (int)wg_scales[row_scale_base + kt / 32 + g];
      }
    }
    // ── Step 4: Fused KV_UPD epilogue ──────────────────────────────────
    int kv_head = _kvupd_get_xcd_id();
    int request_id = 0;
    while (qo_indptr[request_id + 1] <= tok_idx) {
      request_id++;
    }
    int token_in_request = tok_idx - qo_indptr[request_id];
    int first_page = kv_indptr[request_id];
    int num_pages = kv_indptr[request_id + 1] - first_page;
    int last_page_len_val = kv_last_page_len[request_id];
    int global_seq_len = (num_pages - 1) * PAGE_SIZE + last_page_len_val;
    int num_new_tokens = qo_indptr[request_id + 1] - qo_indptr[request_id];
    int global_pos = global_seq_len - num_new_tokens + token_in_request;
    unsigned short const *cos_row =
        (unsigned short const *)cos_ptr + global_pos * HEAD_DIM;
    unsigned short const *sin_row =
        (unsigned short const *)sin_ptr + global_pos * HEAD_DIM;
    unsigned short *s_rope = (unsigned short *)_rnlm_smem;
    if (wg_idx < NUM_Q_PER_KV) {
      int q_head_global = kv_head * NUM_Q_PER_KV + wg_idx;
      _kvupd_rope_epilogue<HEAD_DIM>((float const *)&acc,
                                     d_bias,
                                     wg_idx,
                                     wave_tile,
                                     g,
                                     col,
                                     tid,
                                     cos_row,
                                     sin_row,
                                     (unsigned short *)q_workspace_ptr,
                                     q_ws_stride,
                                     tok_idx,
                                     q_head_global * HEAD_DIM,
                                     OUTPUT_PER_WG,
                                     s_rope);
    } else if (wg_idx == NUM_Q_PER_KV) {
      int page_num = global_pos / PAGE_SIZE;
      int page_offset = global_pos % PAGE_SIZE;
      int page_idx = kv_indices[first_page + page_num];
      int dst_idx = page_idx * PAGE_SIZE + page_offset;
      if (col == 0) {
        int d0 = wave_tile * 16 + g * 4;
        uint64_t bias4;
        __builtin_memcpy(&bias4, &d_bias[wg_idx * OUTPUT_PER_WG + d0], 8);
        unsigned short const *b4 = (unsigned short const *)&bias4;
#pragma unroll
        for (int i = 0; i < 4; i++) {
          unsigned bt = (unsigned)b4[i] << 16;
          float bv;
          __builtin_memcpy(&bv, &bt, 4);
          s_rope[d0 + i] = _gang_float_to_bf16(acc[i] + bv);
        }
      }
      __syncthreads();
      constexpr int HALF = HEAD_DIM / 2;
      if (tid < HALF) {
        unsigned v0b = (unsigned)s_rope[tid] << 16;
        unsigned v1b = (unsigned)s_rope[tid + HALF] << 16;
        float v0, v1;
        __builtin_memcpy(&v0, &v0b, 4);
        __builtin_memcpy(&v1, &v1b, 4);
        unsigned cb = (unsigned)cos_row[tid] << 16;
        unsigned sb = (unsigned)sin_row[tid] << 16;
        float c, s;
        __builtin_memcpy(&c, &cb, 4);
        __builtin_memcpy(&s, &sb, 4);
        s_rope[tid] = _gang_float_to_bf16(v0 * c - v1 * s);
        s_rope[tid + HALF] = _gang_float_to_bf16(v0 * s + v1 * c);
      }
      __syncthreads();
      unsigned short *d_k = (unsigned short *)k_cache_ptr;
      for (int d = tid; d < HEAD_DIM; d += 256) {
        d_k[dst_idx * kv_stride + kv_head * HEAD_DIM + d] = s_rope[d];
      }
    } else {
      int page_num = global_pos / PAGE_SIZE;
      int page_offset = global_pos % PAGE_SIZE;
      int page_idx = kv_indices[first_page + page_num];
      int dst_idx = page_idx * PAGE_SIZE + page_offset;
      unsigned short *d_v = (unsigned short *)v_cache_ptr;
      if (col == 0) {
        int d0 = wave_tile * 16 + g * 4;
        uint64_t bias4;
        __builtin_memcpy(&bias4, &d_bias[wg_idx * OUTPUT_PER_WG + d0], 8);
        unsigned short const *b4 = (unsigned short const *)&bias4;
#pragma unroll
        for (int i = 0; i < 4; i++) {
          unsigned bt = (unsigned)b4[i] << 16;
          float bv;
          __builtin_memcpy(&bv, &bt, 4);
          d_v[dst_idx * kv_stride + kv_head * HEAD_DIM + d0 + i] =
              _gang_float_to_bf16(acc[i] + bv);
        }
      }
    }
  }

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  if (_sp_rec) {
    _sp_t4 = __builtin_amdgcn_s_memrealtime();
    // Slot 1: QKV_KVUPD. [0]=MulSumAdd [1]=RMSNorm [2]=FP8Quant [3]=MFMA
    // [4]=KVUpdEpi
    atomicAdd(&g_subphase_ns[1][0], (_sp_t1 - _sp_t0) * 10);
    atomicAdd(&g_subphase_ns[1][1], (_sp_t2 - _sp_t1) * 10);
    atomicAdd(&g_subphase_ns[1][2], (_sp_t3 - _sp_t2) * 10);
    atomicAdd(&g_subphase_ns[1][3], (_sp_t4 - _sp_t3) * 10);
    atomicAdd(&g_subphase_cnt[1], 1ULL);
  }
#endif
  __syncthreads();
}

// ════════════════════════════════════════════════════════════════════════════
// ResAddF32 variants: read from f32 workspace (written by W2 atomicAdd)
// instead of doing MulSumAdd from 4 bf16 expert slots.
// ════════════════════════════════════════════════════════════════════════════

// ── Non-KVUpd variant: ResAddF32 + RMSNorm + MXFP4 + Bias ──────────────
// Replaces gang_mulsumradd_rmsnorm_linear_mxfp4_bias_kernel for layers 1+
// when W2 epilogue already did routing_weight * result → atomicAdd to f32
// workspace.
//
// Prologue: x[h] = workspace_f32[h] + residual_bf16[h]; workspace_f32[h] = 0;
// Then RMSNorm + FP8 quant + MFMA + bias epilogue (identical to mulsumradd).
//
// 5 inputs (workspace_f32, residual, norm_weight, mxfp4_weight, bias)
// + 1 inout (norm_scratch) → workspace_f32 is also zeroed (read+write).
// We pass workspace_f32 as both input[0] and output[0] (x_output replaced).
//
// Input layout matches mulsumradd for easy swapping:
//   input[0] = workspace_f32 [batch, REDUCTION_SIZE] f32
//   input[1] = residual [batch, REDUCTION_SIZE] bf16
//   input[2] = norm_weight [REDUCTION_SIZE] bf16
//   input[3] = norm_scratch [batch, REDUCTION_SIZE] bf16
//   input[4] = mxfp4_weight [n_wgs_per_xcd, wg_bytes]
//   input[5] = bias [1, output_size_per_xcd] bf16
//   output[0] = x_output [batch, REDUCTION_SIZE] bf16 (residual stream)
//   output[1] = qkv_output [batch, output_stride] bf16
template <int BATCH_SIZE,
          int OUTPUT_PER_WG,
          int REDUCTION_SIZE,
          int ACTUAL_HIDDEN_DIM = REDUCTION_SIZE>
__device__ __noinline__ void gang_resaddf32_rmsnorm_linear_mxfp4_bias_kernel(
    void *workspace_f32_ptr,     // [batch, REDUCTION_SIZE] f32 (read + zero)
    void const *residual_ptr,    // [batch, REDUCTION_SIZE] bf16
    void const *norm_weight_ptr, // [REDUCTION_SIZE] bf16
    void *norm_scratch_ptr,      // [batch, REDUCTION_SIZE] bf16
    void const *weight_ptr,      // [n_wgs_per_xcd, wg_bytes] packed MXFP4
    void const *bias_ptr,        // [1, output_size_per_xcd] bf16
    void *x_output_ptr,          // [batch, REDUCTION_SIZE] bf16
    void *qkv_output_ptr,        // [batch, output_stride] bf16
    int num_active_tokens,
    int n_wgs_per_xcd,
    int output_stride,
    int tile_idx) {

  static_assert(OUTPUT_PER_WG % 16 == 0,
                "OUTPUT_PER_WG must be multiple of 16");
  static_assert(REDUCTION_SIZE % 128 == 0,
                "REDUCTION_SIZE must be multiple of 128 for FP4 MFMA");

  constexpr int NUM_BLOCKS_32 = REDUCTION_SIZE / 32;
  constexpr int WG_DATA_BYTES = OUTPUT_PER_WG * (REDUCTION_SIZE / 2);
  constexpr int WG_SCALE_BYTES = OUTPUT_PER_WG * NUM_BLOCKS_32;
  constexpr int WG_BYTES = WG_DATA_BYTES + WG_SCALE_BYTES;
  constexpr int K_PER_MFMA = 128;
  constexpr int MFMA_ITERS = REDUCTION_SIZE / K_PER_MFMA;
  static_assert(MFMA_ITERS >= 4,
                "Depth-4 pipeline requires REDUCTION_SIZE >= 512");
  constexpr int BF16_MFMA_ITERS = REDUCTION_SIZE / 32;
  constexpr int NUM_WAVES = 4;
  constexpr int TILES_PER_WAVE = OUTPUT_PER_WG / 16 / NUM_WAVES;

  // ── Token activation in shared memory ────────────────────────────────────
  constexpr int FP8_TOK_DATA = REDUCTION_SIZE;

  uint8_t const *W = (uint8_t const *)weight_ptr;
  unsigned short const *d_bias = (unsigned short const *)bias_ptr;
  unsigned short *d_qkv_output = (unsigned short *)qkv_output_ptr;

  extern __shared__ char _rnlm_smem[];
  uint8_t *s_tok_fp8 = (uint8_t *)_rnlm_smem;
  uint8_t *s_tok_scales = s_tok_fp8 + FP8_TOK_DATA;

  int const tid = threadIdx.x;
  int const warp_id = tid >> 6;
  int const lane_id = tid & 63;
  int const col = lane_id & 15;
  int const g = lane_id >> 4;

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  unsigned long long _sp_t0 = 0, _sp_t1 = 0, _sp_t2 = 0, _sp_t3 = 0,
                     _sp_t3b = 0, _sp_t4 = 0;
  bool _sp_rec = (tile_idx == 0 && tid == 0 && g_subphase_active);
  if (_sp_rec) {
    _sp_t0 = __builtin_amdgcn_s_memrealtime();
  }
#endif
  int batch_count =
      (num_active_tokens < BATCH_SIZE) ? num_active_tokens : BATCH_SIZE;

  // ── Step 0+1 FUSED: ResAddF32 + RMSNorm ───────────────────────────────
  // Reads f32 workspace (pre-accumulated by W2 atomicAdd), adds bf16 residual,
  // zeros workspace, computes SSQ for RMSNorm — all in a single pass.
  {
    float *d_ws = (float *)workspace_f32_ptr;
    unsigned short const *d_residual = (unsigned short const *)residual_ptr;
    unsigned short *d_x_out = (unsigned short *)x_output_ptr;
    unsigned short const *d_norm_w = (unsigned short const *)norm_weight_ptr;
    unsigned short *d_norm_out = (unsigned short *)norm_scratch_ptr;

    for (int b = 0; b < batch_count; b++) {
      // Pass 1: workspace_f32 + residual → x_out, accumulate SSQ
      // Vectorized: float4 for f32 workspace, uint2 for bf16 residual/output
      // NOTE: Do NOT zero workspace here — multiple gang workers race on this.
      float ssq = 0.0f;
      // Register cache: store resadd sums (f32) to avoid re-reading x_out in
      // Pass 2
      constexpr int _VEC = 4;
      constexpr int _BLOCK_VEC = 256 * _VEC;
      constexpr int _MAX_ITERS = (REDUCTION_SIZE + _BLOCK_VEC - 1) / _BLOCK_VEC;
      float s_cache[_MAX_ITERS * _VEC];
      int n_cached = 0;
      {
        constexpr int VEC = 4;
        // REDUCTION_SIZE / (blockDim.x * VEC) iterations per thread
        // e.g. 3072 / (256 * 4) = 3 iterations — fully unrollable
        constexpr int BLOCK_VEC = 256 * VEC; // elements per block per iteration
        float *ws_base = d_ws + b * REDUCTION_SIZE;
        unsigned short const *res_base = d_residual + b * REDUCTION_SIZE;
        unsigned short *xout_base = d_x_out + b * REDUCTION_SIZE;

#pragma unroll
        for (int off = tid * VEC; off < REDUCTION_SIZE; off += BLOCK_VEC) {
          // Vectorized load: 4 f32 from workspace (flat_load_dwordx4)
          float4 ws4;
          __builtin_memcpy(&ws4, ws_base + off, 16);

          // Vectorized load: 4 bf16 from residual as uint2 (flat_load_dwordx2)
          uint2 res_packed;
          __builtin_memcpy(&res_packed, res_base + off, 8);

          // Extract 4 bf16 → f32, add, accumulate SSQ
          unsigned r0 = (res_packed.x & 0xFFFFu) << 16;
          unsigned r1 = res_packed.x & 0xFFFF0000u;
          unsigned r2 = (res_packed.y & 0xFFFFu) << 16;
          unsigned r3 = res_packed.y & 0xFFFF0000u;
          float rv0, rv1, rv2, rv3;
          __builtin_memcpy(&rv0, &r0, 4);
          __builtin_memcpy(&rv1, &r1, 4);
          __builtin_memcpy(&rv2, &r2, 4);
          __builtin_memcpy(&rv3, &r3, 4);

          float s0 = ws4.x + rv0;
          float s1 = ws4.y + rv1;
          float s2 = ws4.z + rv2;
          float s3 = ws4.w + rv3;

          asm volatile("v_fmac_f32 %0, %1, %1" : "+v"(ssq) : "v"(s0));
          asm volatile("v_fmac_f32 %0, %1, %1" : "+v"(ssq) : "v"(s1));
          asm volatile("v_fmac_f32 %0, %1, %1" : "+v"(ssq) : "v"(s2));
          asm volatile("v_fmac_f32 %0, %1, %1" : "+v"(ssq) : "v"(s3));

          // Cache resadd sums in registers for Pass 2 (avoid re-reading x_out)
          s_cache[n_cached + 0] = s0;
          s_cache[n_cached + 1] = s1;
          s_cache[n_cached + 2] = s2;
          s_cache[n_cached + 3] = s3;
          n_cached += _VEC;

          // Pack 4 bf16 output and store as uint2 (flat_store_dwordx2)
          unsigned short o0 = _gang_float_to_bf16(s0);
          unsigned short o1 = _gang_float_to_bf16(s1);
          unsigned short o2 = _gang_float_to_bf16(s2);
          unsigned short o3 = _gang_float_to_bf16(s3);
          uint2 out_packed;
          out_packed.x = (unsigned)o0 | ((unsigned)o1 << 16);
          out_packed.y = (unsigned)o2 | ((unsigned)o3 << 16);
          __builtin_memcpy(xout_base + off, &out_packed, 8);
        }
      }

// Wavefront reduction
#pragma unroll
      for (int offset = 32; offset > 0; offset >>= 1) {
        ssq += __shfl_xor(ssq, offset);
      }

      // Cross-wave reduction via shared memory
      float *s_red = (float *)_rnlm_smem;
      int _wave_id = tid >> 6;
      int _lane_id = tid & 63;
      int _num_waves = blockDim.x >> 6;
      if (_lane_id == 0) {
        s_red[_wave_id] = ssq;
      }
      __syncthreads();

      float rms_rcp;
      if (_wave_id == 0) {
        ssq = (_lane_id < _num_waves) ? s_red[_lane_id] : 0.0f;
        for (int offset = _num_waves >> 1; offset > 0; offset >>= 1) {
          ssq += __shfl_xor(ssq, offset);
        }
        if (_lane_id == 0) {
          s_red[0] = rsqrtf(ssq / (float)ACTUAL_HIDDEN_DIM + 1e-5f);
        }
      }
      __syncthreads();
      rms_rcp = s_red[0];

      // Pass 2: Apply norm weight using CACHED sums (no re-read of x_out)
      {
        using bf16 = __hip_bfloat16;
        constexpr int VEC = 4;
        bf16 const *w_in = (bf16 const *)d_norm_w;
        bf16 *out = (bf16 *)d_norm_out + b * REDUCTION_SIZE;
        int ci = 0;
#pragma unroll
        for (int off = tid * VEC; off < REDUCTION_SIZE; off += _BLOCK_VEC) {
          uint64_t wv;
          __builtin_memcpy(&wv, &w_in[off], 8);
          bf16 const *wa = (bf16 const *)&wv;
          bf16 ov[VEC];
#pragma unroll
          for (int j = 0; j < VEC; j++) {
            ov[j] = __float2bfloat16(s_cache[ci + j] * rms_rcp *
                                     __bfloat162float(wa[j]));
          }
          __builtin_memcpy(&out[off], ov, 8);
          ci += VEC;
        }
      }
    }
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  __syncthreads();

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  if (_sp_rec) {
    _sp_t1 = __builtin_amdgcn_s_memrealtime();
    _sp_t2 = _sp_t1;
  }
#endif
  // ── Tile dispatch ───────────────────────────────────────────────────────
  int tok_idx = tile_idx / n_wgs_per_xcd;
  int wg_idx = tile_idx % n_wgs_per_xcd;

  if (tok_idx >= batch_count) {
    return;
  }

  uint8_t const *wg_data = W + static_cast<int64_t>(wg_idx) * WG_BYTES;
  uint8_t const *wg_scales = wg_data + WG_DATA_BYTES;

  // ── Step 2: Quantize/copy BF16 input to LDS ────────────────────────────
  unsigned short const *input_row =
      (unsigned short const *)norm_scratch_ptr + tok_idx * REDUCTION_SIZE;
  _gang_wave_parallel_fp8_quant<REDUCTION_SIZE>(
      input_row, s_tok_fp8, s_tok_scales);

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  if (_sp_rec) {
    _sp_t3 = __builtin_amdgcn_s_memrealtime();
  }
#endif
  // ── Step 3+4: MFMA + bias epilogue ─────────────────────────────────────
  if constexpr (OUTPUT_PER_WG >= 64) {
    for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; tile_iter++) {
      int wave_tile = warp_id + tile_iter * NUM_WAVES;
      int w_row = wave_tile * 16 + col;

      int const row_data_base = w_row * (REDUCTION_SIZE / 2);
      int const row_scale_base = w_row * NUM_BLOCKS_32;

      f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

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

      if (col == 0) {
        for (int i = 0; i < 4; i++) {
          int out_n = wg_idx * OUTPUT_PER_WG + wave_tile * 16 + g * 4 + i;
          float sum = acc[i];

          unsigned bt = (unsigned)d_bias[out_n] << 16;
          float bv;
          __builtin_memcpy(&bv, &bt, 4);

          int out_idx = tok_idx * output_stride + out_n;
          d_qkv_output[out_idx] = _gang_float_to_bf16(sum + bv);
        }
      }
    }
  } else {
    // K-parallel path: depth-4 pipelined
    constexpr int TOTAL_K_ITERS = MFMA_ITERS;
    constexpr int ITERS_PER_WAVE = TOTAL_K_ITERS / NUM_WAVES;
    static_assert(TOTAL_K_ITERS % NUM_WAVES == 0,
                  "MFMA_ITERS must be divisible by NUM_WAVES for K-parallel");
    static_assert(ITERS_PER_WAVE >= 4,
                  "Depth-4 K-parallel requires ITERS_PER_WAVE >= 4");

    int const ki_start = warp_id * ITERS_PER_WAVE;
    int const ki_end = ki_start + ITERS_PER_WAVE;

    int w_row = col;
    int const row_data_base = w_row * (REDUCTION_SIZE / 2);
    int const row_scale_base = w_row * NUM_BLOCKS_32;

    f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

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

#pragma unroll 1
    for (int ki = ki_start; ki < ki_end; ki += 4) {
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

    float *lds_reduce = (float *)_rnlm_smem;
    if (col == 0) {
      for (int i = 0; i < 4; i++) {
        lds_reduce[warp_id * OUTPUT_PER_WG + g * 4 + i] = acc[i];
      }
    }
    __syncthreads();

    if (warp_id == 0 && col == 0) {
      for (int i = 0; i < 4; i++) {
        float v = 0.0f;
        for (int w = 0; w < NUM_WAVES; w++) {
          v += lds_reduce[w * OUTPUT_PER_WG + g * 4 + i];
        }

        int out_n = wg_idx * OUTPUT_PER_WG + g * 4 + i;

        unsigned bt = (unsigned)d_bias[out_n] << 16;
        float bv;
        __builtin_memcpy(&bv, &bt, 4);

        int out_idx = tok_idx * output_stride + out_n;
        d_qkv_output[out_idx] = _gang_float_to_bf16(v + bv);
      }
    }
  }

  // ── Zero workspace_f32 for next iteration ──────────────────────────────
  // Done AFTER all MFMA computation so all gang workers have finished reading.
  // Only one worker (tile_idx % n_wgs_per_xcd == 0) does this to avoid races.
  // Vectorized: float4 zero stores (flat_store_dwordx4).
  {
    int wg_idx = tile_idx % n_wgs_per_xcd;
    if (wg_idx == 0) {
      float *d_ws = (float *)workspace_f32_ptr;
      int batch_count_z =
          (num_active_tokens < BATCH_SIZE) ? num_active_tokens : BATCH_SIZE;
      constexpr int VEC = 4;
      constexpr int BLOCK_VEC = 256 * VEC;
      float4 zero4 = {0.0f, 0.0f, 0.0f, 0.0f};
      for (int b = 0; b < batch_count_z; b++) {
        float *ws_row = d_ws + b * REDUCTION_SIZE;
#pragma unroll
        for (int off = tid * VEC; off < REDUCTION_SIZE; off += BLOCK_VEC) {
          __builtin_memcpy(ws_row + off, &zero4, 16);
        }
      }
    }
  }

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  if (_sp_rec) {
    _sp_t4 = __builtin_amdgcn_s_memrealtime();
    // Slot 0: QKV. [0]=Fused ResAddF32+RMSNorm [1]=0 (fused) [2]=FP8Quant
    // [3]=MFMA+Epi
    atomicAdd(&g_subphase_ns[0][0], (_sp_t1 - _sp_t0) * 10);
    atomicAdd(&g_subphase_ns[0][1], (_sp_t2 - _sp_t1) * 10);
    atomicAdd(&g_subphase_ns[0][2], (_sp_t3 - _sp_t2) * 10);
    atomicAdd(&g_subphase_ns[0][3], (_sp_t4 - _sp_t3) * 10);
    atomicAdd(&g_subphase_cnt[0], 1ULL);
  }
#endif
  __syncthreads();
}

// ── KVUpd variant: ResAddF32 + RMSNorm + MXFP4 + KV Cache Update ───────
// Replaces gang_mulsumradd_rmsnorm_linear_mxfp4_bias_kvupd_kernel for layers
// 1+.
template <int BATCH_SIZE,
          int OUTPUT_PER_WG,
          int REDUCTION_SIZE,
          int ACTUAL_HIDDEN_DIM,
          int HEAD_DIM,
          int NUM_Q_PER_KV,
          int PAGE_SIZE>
__device__ __noinline__ void
    gang_resaddf32_rmsnorm_linear_mxfp4_bias_kvupd_kernel(
        void *workspace_f32_ptr,  // [batch, REDUCTION_SIZE] f32 (read + zero)
        void const *residual_ptr, // [batch, REDUCTION_SIZE] bf16
        void const *norm_weight_ptr, // [REDUCTION_SIZE] bf16
        void *norm_scratch_ptr,      // [batch, REDUCTION_SIZE] bf16
        void const *weight_ptr,      // [n_wgs_per_xcd, wg_bytes] packed MXFP4
        void const *bias_ptr,        // [1, output_size_per_xcd] bf16
        void *x_output_ptr,          // [batch, REDUCTION_SIZE] bf16
        void *k_cache_ptr,
        void *v_cache_ptr,
        void *q_workspace_ptr,
        void const *cos_ptr,
        void const *sin_ptr,
        int const *qo_indptr,
        int const *kv_indptr,
        int const *kv_indices,
        int const *kv_last_page_len,
        int num_active_tokens,
        int n_wgs_per_xcd,
        int kv_stride,
        int q_ws_stride,
        int tile_idx) {

  static_assert(OUTPUT_PER_WG % 16 == 0);
  static_assert(REDUCTION_SIZE % 128 == 0);
  static_assert(OUTPUT_PER_WG == HEAD_DIM,
                "KV_UPD fusion requires OUTPUT_PER_WG == HEAD_DIM");

  constexpr int NUM_BLOCKS_32 = REDUCTION_SIZE / 32;
  constexpr int WG_DATA_BYTES = OUTPUT_PER_WG * (REDUCTION_SIZE / 2);
  constexpr int WG_SCALE_BYTES = OUTPUT_PER_WG * NUM_BLOCKS_32;
  constexpr int WG_BYTES = WG_DATA_BYTES + WG_SCALE_BYTES;
  constexpr int K_PER_MFMA = 128;
  constexpr int MFMA_ITERS = REDUCTION_SIZE / K_PER_MFMA;
  constexpr int NUM_WAVES = 4;
  constexpr int TILES_PER_WAVE = OUTPUT_PER_WG / 16 / NUM_WAVES;

  // ── Token activation in shared memory ────────────────────────────────────
  constexpr int FP8_TOK_DATA = REDUCTION_SIZE;

  uint8_t const *W = (uint8_t const *)weight_ptr;
  unsigned short const *d_bias = (unsigned short const *)bias_ptr;

  extern __shared__ char _rnlm_smem[];
  uint8_t *s_tok_fp8 = (uint8_t *)_rnlm_smem;
  uint8_t *s_tok_scales = s_tok_fp8 + FP8_TOK_DATA;

  int const tid = threadIdx.x;
  int const warp_id = tid >> 6;
  int const lane_id = tid & 63;
  int const col = lane_id & 15;
  int const g = lane_id >> 4;

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  unsigned long long _sp_t0 = 0, _sp_t1 = 0, _sp_t2 = 0, _sp_t3 = 0,
                     _sp_t3b = 0, _sp_t4 = 0;
  bool _sp_rec = (tile_idx == 0 && tid == 0 && g_subphase_active);
  if (_sp_rec) {
    _sp_t0 = __builtin_amdgcn_s_memrealtime();
  }
#endif
  int batch_count =
      (num_active_tokens < BATCH_SIZE) ? num_active_tokens : BATCH_SIZE;

  int tok_idx = tile_idx / n_wgs_per_xcd;
  int wg_idx = tile_idx % n_wgs_per_xcd;

  uint8_t const *wg_data = W + static_cast<int64_t>(wg_idx) * WG_BYTES;
  uint8_t const *wg_scales = wg_data + WG_DATA_BYTES;

  // ── Phase A: Issue HBM→LDS weight prefetch BEFORE RMSNorm ─────────────
  constexpr int QKV_TILE_ROWS = 16;
  constexpr int QKV_TILE_DATA = QKV_TILE_ROWS * (REDUCTION_SIZE / 2);
  constexpr int QKV_TILE_SCALE = QKV_TILE_ROWS * NUM_BLOCKS_32;
  constexpr int qkv_n16_data = QKV_TILE_DATA / 16;
  constexpr int QKV_LPT = (qkv_n16_data + 255) / 256;
  constexpr int QKV_TILE_DATA_PADDED = QKV_LPT * 256 * 16;
  constexpr int QKV_TILE_BYTES = QKV_TILE_DATA_PADDED + QKV_TILE_SCALE;

  uint32_t qkv_buf_range = static_cast<uint32_t>(n_wgs_per_xcd) * WG_BYTES;
  constexpr int QKV_LDS_OFF_A = ((FP8_TOK_DATA + MFMA_ITERS + 15) / 16) * 16;
  static_assert(QKV_LDS_OFF_A + QKV_TILE_BYTES * NUM_WAVES <= 155 * 1024,
                "QKV LDS weights exceed MI350X LDS budget");
  uint8_t *qkv_lds_w = (uint8_t *)_rnlm_smem + QKV_LDS_OFF_A;

  {
    i32x4_t qkv_rsrc = make_w_buffer_rsrc(W, qkv_buf_range);
    uint32_t qkv_wg_voff = static_cast<uint32_t>(wg_idx) * WG_BYTES;
    auto *qkv_lds_warp_base = (__attribute__((address_space(3)))
                               uint32_t *)(qkv_lds_w + warp_id * 1024);
#pragma unroll
    for (int t = 0; t < NUM_WAVES; t++) {
#pragma unroll
      for (int j = 0; j < QKV_LPT; j++) {
        int idx = tid + j * 256;
        int clamped = idx < qkv_n16_data ? idx : qkv_n16_data - 1;
        uint32_t voff =
            qkv_wg_voff +
            static_cast<uint32_t>(t * QKV_TILE_ROWS * (REDUCTION_SIZE / 2)) +
            static_cast<uint32_t>(clamped * 16);
        auto *lds_dst = (__attribute__((address_space(3)))
                         uint32_t *)((uint8_t __attribute__((address_space(
                                         3))) *)qkv_lds_warp_base +
                                     t * QKV_TILE_BYTES + j * 4096);
        __llvm_amdgcn_raw_buffer_load_lds(
            qkv_rsrc, lds_dst, 16, static_cast<int>(voff), 0, 0, 3);
      }
    }
  }

  // ── Step 0+1 FUSED: ResAddF32 + RMSNorm ───────────────────────────────
  {
    float *d_ws = (float *)workspace_f32_ptr;
    unsigned short const *d_residual = (unsigned short const *)residual_ptr;
    unsigned short *d_x_out = (unsigned short *)x_output_ptr;
    unsigned short const *d_norm_w = (unsigned short const *)norm_weight_ptr;
    unsigned short *d_norm_out = (unsigned short *)norm_scratch_ptr;

    for (int b = 0; b < batch_count; b++) {
      float ssq = 0.0f;
      // Register cache: store resadd sums (f32) to avoid re-reading x_out in
      // Pass 2
      constexpr int _VEC = 4;
      constexpr int _BLOCK_VEC = 256 * _VEC;
      constexpr int _MAX_ITERS = (REDUCTION_SIZE + _BLOCK_VEC - 1) / _BLOCK_VEC;
      float s_cache[_MAX_ITERS * _VEC];
      int n_cached = 0;
      {
        constexpr int VEC = 4;
        constexpr int BLOCK_VEC = 256 * VEC;
        float *ws_base = d_ws + b * REDUCTION_SIZE;
        unsigned short const *res_base = d_residual + b * REDUCTION_SIZE;
        unsigned short *xout_base = d_x_out + b * REDUCTION_SIZE;

#pragma unroll
        for (int off = tid * VEC; off < REDUCTION_SIZE; off += BLOCK_VEC) {
          float4 ws4;
          __builtin_memcpy(&ws4, ws_base + off, 16);
          uint2 res_packed;
          __builtin_memcpy(&res_packed, res_base + off, 8);

          unsigned r0 = (res_packed.x & 0xFFFFu) << 16;
          unsigned r1 = res_packed.x & 0xFFFF0000u;
          unsigned r2 = (res_packed.y & 0xFFFFu) << 16;
          unsigned r3 = res_packed.y & 0xFFFF0000u;
          float rv0, rv1, rv2, rv3;
          __builtin_memcpy(&rv0, &r0, 4);
          __builtin_memcpy(&rv1, &r1, 4);
          __builtin_memcpy(&rv2, &r2, 4);
          __builtin_memcpy(&rv3, &r3, 4);

          float s0 = ws4.x + rv0;
          float s1 = ws4.y + rv1;
          float s2 = ws4.z + rv2;
          float s3 = ws4.w + rv3;

          asm volatile("v_fmac_f32 %0, %1, %1" : "+v"(ssq) : "v"(s0));
          asm volatile("v_fmac_f32 %0, %1, %1" : "+v"(ssq) : "v"(s1));
          asm volatile("v_fmac_f32 %0, %1, %1" : "+v"(ssq) : "v"(s2));
          asm volatile("v_fmac_f32 %0, %1, %1" : "+v"(ssq) : "v"(s3));

          // Cache resadd sums in registers for Pass 2 (avoid re-reading x_out)
          s_cache[n_cached + 0] = s0;
          s_cache[n_cached + 1] = s1;
          s_cache[n_cached + 2] = s2;
          s_cache[n_cached + 3] = s3;
          n_cached += _VEC;

          unsigned short o0 = _gang_float_to_bf16(s0);
          unsigned short o1 = _gang_float_to_bf16(s1);
          unsigned short o2 = _gang_float_to_bf16(s2);
          unsigned short o3 = _gang_float_to_bf16(s3);
          uint2 out_packed;
          out_packed.x = (unsigned)o0 | ((unsigned)o1 << 16);
          out_packed.y = (unsigned)o2 | ((unsigned)o3 << 16);
          __builtin_memcpy(xout_base + off, &out_packed, 8);
        }
      }

#pragma unroll
      for (int offset = 32; offset > 0; offset >>= 1) {
        ssq += __shfl_xor(ssq, offset);
      }

      float *s_red = (float *)_rnlm_smem;
      int _wave_id = tid >> 6;
      int _lane_id = tid & 63;
      int _num_waves = blockDim.x >> 6;
      if (_lane_id == 0) {
        s_red[_wave_id] = ssq;
      }
      __syncthreads();

      float rms_rcp;
      if (_wave_id == 0) {
        ssq = (_lane_id < _num_waves) ? s_red[_lane_id] : 0.0f;
        for (int offset = _num_waves >> 1; offset > 0; offset >>= 1) {
          ssq += __shfl_xor(ssq, offset);
        }
        if (_lane_id == 0) {
          s_red[0] = rsqrtf(ssq / (float)ACTUAL_HIDDEN_DIM + 1e-5f);
        }
      }
      __syncthreads();
      rms_rcp = s_red[0];

      // Pass 2: Apply norm weight using CACHED sums (no re-read of x_out)
      {
        using bf16 = __hip_bfloat16;
        constexpr int VEC = 4;
        bf16 const *w_in = (bf16 const *)d_norm_w;
        bf16 *out = (bf16 *)d_norm_out + b * REDUCTION_SIZE;
        int ci = 0;
#pragma unroll
        for (int off = tid * VEC; off < REDUCTION_SIZE; off += _BLOCK_VEC) {
          uint64_t wv;
          __builtin_memcpy(&wv, &w_in[off], 8);
          bf16 const *wa = (bf16 const *)&wv;
          bf16 ov[VEC];
#pragma unroll
          for (int j = 0; j < VEC; j++) {
            ov[j] = __float2bfloat16(s_cache[ci + j] * rms_rcp *
                                     __bfloat162float(wa[j]));
          }
          __builtin_memcpy(&out[off], ov, 8);
          ci += VEC;
        }
      }
    }
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  __syncthreads();

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  if (_sp_rec) {
    _sp_t1 = __builtin_amdgcn_s_memrealtime();
    _sp_t2 = _sp_t1;
  }
#endif

  // ── Step 2: Prepare activation in LDS ──────────────────────────────────

  unsigned short const *input_row =
      (unsigned short const *)norm_scratch_ptr + tok_idx * REDUCTION_SIZE;
  _gang_wave_parallel_fp8_quant<REDUCTION_SIZE>(
      input_row, s_tok_fp8, s_tok_scales);

  // ── Phase B: Drain buffer_load_lds, scatter scales to LDS ──
  {
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");

    constexpr int QKV_SC_DW4_PER_TILE = QKV_TILE_SCALE / 16;
    constexpr int QKV_TOTAL_SC_DW4 = (QKV_TILE_SCALE * NUM_WAVES) / 16;
    constexpr int QKV_SC_LPT = (QKV_TOTAL_SC_DW4 + 255) / 256;
    i32x4_t qkv_sc_buf[QKV_SC_LPT];
    {
      i32x4_t const *sc_src = (i32x4_t const *)wg_scales;
#pragma unroll
      for (int j = 0; j < QKV_SC_LPT; j++) {
        int idx = tid + j * 256;
        if (idx < QKV_TOTAL_SC_DW4) {
          qkv_sc_buf[j] = sc_src[idx];
        }
      }
    }

    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    {
#pragma unroll
      for (int j = 0; j < QKV_SC_LPT; j++) {
        int idx = tid + j * 256;
        if (idx < QKV_TOTAL_SC_DW4) {
          int tile = idx / QKV_SC_DW4_PER_TILE;
          int off = idx % QKV_SC_DW4_PER_TILE;
          i32x4_t *dst_sc = (i32x4_t *)(qkv_lds_w + tile * QKV_TILE_BYTES +
                                        QKV_TILE_DATA_PADDED);
          dst_sc[off] = qkv_sc_buf[j];
        }
      }
    }

    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    __syncthreads();
  }

  if (tok_idx >= batch_count) {
    return;
  }

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  if (_sp_rec) {
    _sp_t3 = __builtin_amdgcn_s_memrealtime();
  }
#endif

  // ── MFMA + KV Update epilogue ─────────────────────────────────────────
  // ── QKV asm MFMA loop (weights already in LDS from Phase A/B) ──
  {
    constexpr int QKV_LDS_OFF_M = ((FP8_TOK_DATA + MFMA_ITERS + 15) / 16) * 16;
    uint8_t *lds_qkv_base = (uint8_t *)_rnlm_smem + QKV_LDS_OFF_M;

    constexpr int ROW_DATA = REDUCTION_SIZE / 2;   // 1536
    constexpr int ROW_SCALE = REDUCTION_SIZE / 32; // 96

    for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; tile_iter++) {
      int wave_tile = warp_id + tile_iter * NUM_WAVES;

      uint8_t *lds_wd = lds_qkv_base + warp_id * QKV_TILE_BYTES;
      uint8_t *lds_ws = lds_wd + QKV_TILE_DATA_PADDED;

      // ── Pipelined asm MFMA loop — sequential addressing (W13 pattern) ──
      // Weight data layout: per row = K/2 = 1536 bytes, stride +64 per iter
      // Weight scale layout: per row = K/32 = 96 bytes, stride +4 per iter
      // Token layout: stride +0x80 per iter (128 bytes)
      // Token scale: stride +1 per iter (1 byte)
      float qa0, qa1, qa2, qa3;
      {
        unsigned w_addr =
            (unsigned)(uintptr_t)(lds_wd + col * ROW_DATA + g * 16);
        unsigned ws_addr = (unsigned)(uintptr_t)(lds_ws + col * ROW_SCALE + g);
        unsigned t_addr = (unsigned)(uintptr_t)(s_tok_fp8 + g * 16);
        unsigned ts_addr = (unsigned)(uintptr_t)(s_tok_scales);

        asm volatile(
            // Zero accumulator
            "v_accvgpr_write_b32 a0, 0\n"
            "v_accvgpr_write_b32 a1, 0\n"
            "v_accvgpr_write_b32 a2, 0\n"
            "v_accvgpr_write_b32 a3, 0\n"

            // Pre-issue 5 reads for iteration 0
            "ds_read_b128 v[22:25], %[wa]\n"           // weight A (16B FP4)
            "ds_read_u8   v7, %[wsa]\n"                // weight A scale
            "ds_read_b128 v[8:11], %[ta]\n"            // token B lo (16B FP8)
            "ds_read_b128 v[12:15], %[ta] offset:64\n" // token B hi (16B FP8)
            "ds_read_u8   v16, %[tsa]\n"               // token B scale
            "s_mov_b32 s13, 0\n"                       // loop counter

            // ── Iterations 0..22: prefetch next, MFMA current ──
            "PIPELINED_QKV_%=:\n"
            "s_waitcnt lgkmcnt(0)\n" // iter N reads complete

            // Advance addresses for iter N+1
            "v_add_u32_e32 %[wa], 64, %[wa]\n"
            "v_add_u32_e32 %[wsa], 4, %[wsa]\n"
            "v_add_u32_e32 %[ta], 0x80, %[ta]\n"
            "s_add_i32 s13, s13, 1\n"

            // Token B scale → v17 FIRST (oldest in queue, completes first)
            "v_add_u32_e32 v17, s13, %[tsa]\n"
            "ds_read_u8   v17, v17\n" // [lgkmcnt +1] oldest
            // Prefetch iter N+1 data into SAME regs (MFMA reads old values)
            "ds_read_b128 v[22:25], %[wa]\n"           // [lgkmcnt +2]
            "ds_read_u8   v7, %[wsa]\n"                // [lgkmcnt +3]
            "ds_read_b128 v[8:11], %[ta]\n"            // [lgkmcnt +4]
            "ds_read_b128 v[12:15], %[ta] offset:64\n" // [lgkmcnt +5]

            // MFMA from iter N data (32 cycles, reads v[22:25] v[8:15] v7 v16)
            "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[8:15], "
            "a[0:3], v7, v16 op_sel_hi:[0,0,0] cbsz:4\n"

            // Copy next token scale during MFMA execution.
            // lgkmcnt(4): wait for token scale (oldest, issued first), leave 4
            // data reads flying
            "s_waitcnt lgkmcnt(4)\n"
            "v_mov_b32_e32 v16, v17\n"

            "s_cmpk_lt_i32 s13, %[iters_m1]\n"
            "s_cbranch_scc1 PIPELINED_QKV_%=\n"

            // ── Final iteration: no more prefetch needed ──
            "s_waitcnt lgkmcnt(0)\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[8:15], "
            "a[0:3], v7, v16 op_sel_hi:[0,0,0] cbsz:4\n"

            // Read accumulator into output
            "s_nop 7\n"
            "s_nop 0\n"
            "v_accvgpr_read_b32 %[acc0], a0\n"
            "v_accvgpr_read_b32 %[acc1], a1\n"
            "v_accvgpr_read_b32 %[acc2], a2\n"
            "v_accvgpr_read_b32 %[acc3], a3\n"
            : [acc0] "=v"(qa0),
              [acc1] "=v"(qa1),
              [acc2] "=v"(qa2),
              [acc3] "=v"(qa3),
              [wa] "+v"(w_addr),
              [wsa] "+v"(ws_addr),
              [ta] "+v"(t_addr)
            : [tsa] "v"(ts_addr), [iters_m1] "n"(MFMA_ITERS - 1)
            : "memory",
              "s13",
              "v7",
              "v8",
              "v9",
              "v10",
              "v11",
              "v12",
              "v13",
              "v14",
              "v15",
              "v16",
              "v17",
              "v22",
              "v23",
              "v24",
              "v25",
              "a0",
              "a1",
              "a2",
              "a3");
      }
      f32x4_t acc;
      acc[0] = qa0;
      acc[1] = qa1;
      acc[2] = qa2;
      acc[3] = qa3;

      // ── Fused KV_UPD epilogue ──────────────────────────────────────────
      int kv_head = _kvupd_get_xcd_id();
      int request_id = 0;
      while (qo_indptr[request_id + 1] <= tok_idx) {
        request_id++;
      }
      int token_in_request = tok_idx - qo_indptr[request_id];
      int first_page = kv_indptr[request_id];
      int num_pages = kv_indptr[request_id + 1] - first_page;
      int last_page_len_val = kv_last_page_len[request_id];
      int global_seq_len = (num_pages - 1) * PAGE_SIZE + last_page_len_val;
      int num_new_tokens = qo_indptr[request_id + 1] - qo_indptr[request_id];
      int global_pos = global_seq_len - num_new_tokens + token_in_request;
      unsigned short const *cos_row =
          (unsigned short const *)cos_ptr + global_pos * HEAD_DIM;
      unsigned short const *sin_row =
          (unsigned short const *)sin_ptr + global_pos * HEAD_DIM;
      unsigned short *s_rope = (unsigned short *)_rnlm_smem;
      if (wg_idx < NUM_Q_PER_KV) {
        int q_head_global = kv_head * NUM_Q_PER_KV + wg_idx;
        _kvupd_rope_epilogue<HEAD_DIM>((float const *)&acc,
                                       d_bias,
                                       wg_idx,
                                       wave_tile,
                                       g,
                                       col,
                                       tid,
                                       cos_row,
                                       sin_row,
                                       (unsigned short *)q_workspace_ptr,
                                       q_ws_stride,
                                       tok_idx,
                                       q_head_global * HEAD_DIM,
                                       OUTPUT_PER_WG,
                                       s_rope);
      } else if (wg_idx == NUM_Q_PER_KV) {
        int page_num = global_pos / PAGE_SIZE;
        int page_offset = global_pos % PAGE_SIZE;
        int page_idx = kv_indices[first_page + page_num];
        int dst_idx = page_idx * PAGE_SIZE + page_offset;
        if (col == 0) {
          int d0 = wave_tile * 16 + g * 4;
          uint64_t bias4;
          __builtin_memcpy(&bias4, &d_bias[wg_idx * OUTPUT_PER_WG + d0], 8);
          unsigned short const *b4 = (unsigned short const *)&bias4;
#pragma unroll
          for (int i = 0; i < 4; i++) {
            unsigned bt = (unsigned)b4[i] << 16;
            float bv;
            __builtin_memcpy(&bv, &bt, 4);
            s_rope[d0 + i] = _gang_float_to_bf16(acc[i] + bv);
          }
        }
        __syncthreads();
        constexpr int HALF = HEAD_DIM / 2;
        if (tid < HALF) {
          unsigned v0b = (unsigned)s_rope[tid] << 16;
          unsigned v1b = (unsigned)s_rope[tid + HALF] << 16;
          float v0, v1;
          __builtin_memcpy(&v0, &v0b, 4);
          __builtin_memcpy(&v1, &v1b, 4);
          unsigned cb = (unsigned)cos_row[tid] << 16;
          unsigned sb_r = (unsigned)sin_row[tid] << 16;
          float c, s;
          __builtin_memcpy(&c, &cb, 4);
          __builtin_memcpy(&s, &sb_r, 4);
          s_rope[tid] = _gang_float_to_bf16(v0 * c - v1 * s);
          s_rope[tid + HALF] = _gang_float_to_bf16(v0 * s + v1 * c);
        }
        __syncthreads();
        unsigned short *d_k = (unsigned short *)k_cache_ptr;
        for (int d = tid; d < HEAD_DIM; d += 256) {
          d_k[dst_idx * kv_stride + kv_head * HEAD_DIM + d] = s_rope[d];
        }
      } else {
        int page_num = global_pos / PAGE_SIZE;
        int page_offset = global_pos % PAGE_SIZE;
        int page_idx = kv_indices[first_page + page_num];
        int dst_idx = page_idx * PAGE_SIZE + page_offset;
        unsigned short *d_v = (unsigned short *)v_cache_ptr;
        if (col == 0) {
          int d0 = wave_tile * 16 + g * 4;
          uint64_t bias4;
          __builtin_memcpy(&bias4, &d_bias[wg_idx * OUTPUT_PER_WG + d0], 8);
          unsigned short const *b4 = (unsigned short const *)&bias4;
#pragma unroll
          for (int i = 0; i < 4; i++) {
            unsigned bt = (unsigned)b4[i] << 16;
            float bv;
            __builtin_memcpy(&bv, &bt, 4);
            d_v[dst_idx * kv_stride + kv_head * HEAD_DIM + d0 + i] =
                _gang_float_to_bf16(acc[i] + bv);
          }
        }
      }
    }
  }

  // ── Zero workspace_f32 for next iteration ──────────────────────────────
  // Done AFTER all MFMA computation so all gang workers have finished reading.
  // Only one worker (tile_idx % n_wgs_per_xcd == 0) does this to avoid races.
  // Uses write-through stores (sc0 sc1) so zeros are visible to MoE's
  // cross-XCD coherent atomicAdd in fused kernels (task 216).
  {
    int wg_idx_z = tile_idx % n_wgs_per_xcd;
    if (wg_idx_z == 0) {
      float *d_ws = (float *)workspace_f32_ptr;
      int batch_count_z =
          (num_active_tokens < BATCH_SIZE) ? num_active_tokens : BATCH_SIZE;
      for (int b = 0; b < batch_count_z; b++) {
        float *ws_row = d_ws + b * REDUCTION_SIZE;
#pragma unroll
        for (int off = tid * 4; off < REDUCTION_SIZE; off += 256 * 4) {
          st_wt_zero128((void *)&ws_row[off]);
        }
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }
  }

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  if (_sp_rec) {
    _sp_t4 = __builtin_amdgcn_s_memrealtime();
    // Slot 1: QKV_KVUPD. [0]=ResAddF32+RMSNorm [1]=0(fused) [2]=FP8Quant
    // [3]=MFMA [4]=KVUpdEpi
    atomicAdd(&g_subphase_ns[1][0], (_sp_t1 - _sp_t0) * 10);
    atomicAdd(&g_subphase_ns[1][1], (_sp_t2 - _sp_t1) * 10);
    atomicAdd(&g_subphase_ns[1][2], (_sp_t3 - _sp_t2) * 10);
    atomicAdd(&g_subphase_ns[1][3], (_sp_t3b - _sp_t3) * 10);
    atomicAdd(&g_subphase_ns[1][4], (_sp_t4 - _sp_t3b) * 10);
    atomicAdd(&g_subphase_cnt[1], 1ULL);
  }
#endif
  __syncthreads();
}

} // namespace kernel
