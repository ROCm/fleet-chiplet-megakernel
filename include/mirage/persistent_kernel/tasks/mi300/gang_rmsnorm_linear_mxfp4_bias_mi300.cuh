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
#include "tasks/mi300/moe_ws_layout.cuh" // MOE_WS_SLOTS, moe_ws_offset()

namespace kernel {

// Native vector types for address-space-qualified loads. HIP's float4/uint2 are
// class templates (HIP_vector_type), so they cannot be copy-constructed through
// an AS(1) pointer, and __builtin_memcpy is declared __host__ and rejects one
// outright. These ext_vector_types can, which is what lets the ResAdd prologue
// issue GLOBAL rather than generic FLAT loads -- see the comment at its use.
typedef float _gl_f32x4 __attribute__((ext_vector_type(4)));
typedef unsigned int _gl_u32x2 __attribute__((ext_vector_type(2)));

// ── Shared QKV weight-tile LDS geometry ──────────────────────────────────
//
// The DMA below writes where these say and the MFMA in the kvupd kernel reads
// where these say. They were previously spelled out once, inline; they are now
// shared with the fused layer's Phase 9 prefetch, which stages the NEXT
// layer's weights into the same region during the layer-barrier spin. A
// mismatch between the two callers is silent wrong numerics, so both derive
// every offset from these functions and neither open-codes any of it.
template <int BATCH_SIZE, int OUTPUT_PER_WG, int REDUCTION_SIZE>
struct QkvWeightLds {
  static constexpr int NUM_WAVES = 4;
  static constexpr int NUM_BLOCKS_32 = REDUCTION_SIZE / 32;
  static constexpr int WG_DATA_BYTES = OUTPUT_PER_WG * (REDUCTION_SIZE / 2);
  static constexpr int WG_BYTES = WG_DATA_BYTES + OUTPUT_PER_WG * NUM_BLOCKS_32;

  static constexpr int MFMA_N = 16;
  static constexpr int TOK_ROWS = BATCH_SIZE < MFMA_N ? BATCH_SIZE : MFMA_N;
  static constexpr int MFMA_ITERS = REDUCTION_SIZE / 128;
  static constexpr int TOK_ROW_STRIDE = REDUCTION_SIZE + 16;
  static constexpr int SC_STRIDE = ((MFMA_ITERS + 3) / 4) * 4;
  static constexpr int TOK_REGION = TOK_ROWS * TOK_ROW_STRIDE;
  static constexpr int SC_REGION = TOK_ROWS * SC_STRIDE;

  static constexpr int TILE_ROWS = 16;
  static constexpr int TILE_DATA = TILE_ROWS * (REDUCTION_SIZE / 2);
  static constexpr int TILE_SCALE = TILE_ROWS * NUM_BLOCKS_32;
  static constexpr int N16_DATA = TILE_DATA / 16;
  static constexpr int LPT = (N16_DATA + 255) / 256;
  static constexpr int TILE_DATA_PADDED = LPT * 256 * 16;
#ifdef MPK_QKV_PREFETCH_SCALES
  // buffer_load_lds advances the LDS pointer in 1 KiB wave chunks, so a scale
  // region that is DMA'd rather than scattered has to be a whole number of
  // them. Costs (pad - TILE_SCALE) bytes of LDS per tile and buys the removal
  // of the VGPR round trip in Phase B.
  static constexpr int TILE_SCALE_PADDED = ((TILE_SCALE + 1023) / 1024) * 1024;
  static constexpr int TILE_BYTES = TILE_DATA_PADDED + TILE_SCALE_PADDED;
#else
  static constexpr int TILE_BYTES = TILE_DATA_PADDED + TILE_SCALE;
#endif

  // Weight region base within the task's dynamic LDS.
  static constexpr int W_OFF = ((TOK_REGION + SC_REGION + 15) / 16) * 16;
  // Total LDS the weight staging occupies, from the base of _rnlm_smem.
  static constexpr int W_END = W_OFF + TILE_BYTES * NUM_WAVES;
};

// Issue the HBM->LDS weight DMA for one QKV tile. Does NOT wait: the caller
// drains with `s_waitcnt vmcnt(0)` once it has something else to do first.
// `tile_idx` selects the weight group exactly as the kvupd kernel does, so a
// worker prefetching for the next layer must pass the tile index it will
// itself execute there (xcd_rank is stable across layers).
template <int BATCH_SIZE, int OUTPUT_PER_WG, int REDUCTION_SIZE>
__device__ __forceinline__ void qkv_prefetch_weights_lds(void const *weight_ptr,
                                                         int n_wgs_per_xcd,
                                                         int tile_idx) {
  using G = QkvWeightLds<BATCH_SIZE, OUTPUT_PER_WG, REDUCTION_SIZE>;
  extern __shared__ char _rnlm_smem[];

  int const tid = threadIdx.x;
  int const warp_id = tid >> 6;
  int const wg_idx = tile_idx % n_wgs_per_xcd;

  uint8_t const *W = (uint8_t const *)weight_ptr;
  i32x4_t rsrc = make_w_buffer_rsrc(
      W, static_cast<uint32_t>(n_wgs_per_xcd) * G::WG_BYTES);
  uint32_t wg_voff = static_cast<uint32_t>(wg_idx) * G::WG_BYTES;

  auto *lds_warp_base =
      (__attribute__((address_space(3))) uint32_t *)((uint8_t *)_rnlm_smem +
                                                     G::W_OFF +
                                                     warp_id * 1024);
#pragma unroll
  for (int t = 0; t < G::NUM_WAVES; t++) {
#pragma unroll
    for (int j = 0; j < G::LPT; j++) {
      int idx = tid + j * 256;
      int clamped = idx < G::N16_DATA ? idx : G::N16_DATA - 1;
      uint32_t voff =
          wg_voff +
          static_cast<uint32_t>(t * G::TILE_ROWS * (REDUCTION_SIZE / 2)) +
          static_cast<uint32_t>(clamped * 16);
      auto *lds_dst =
          (__attribute__((address_space(3)))
           uint32_t *)((uint8_t __attribute__((address_space(3))) *)
                           lds_warp_base +
                       t * G::TILE_BYTES + j * 4096);
      __llvm_amdgcn_raw_buffer_load_lds(
          rsrc, lds_dst, 16, static_cast<int>(voff), 0, 0, 3);
    }
#ifdef MPK_QKV_PREFETCH_SCALES
    // Stage this tile's block scales straight to LDS as well, so Phase B has
    // nothing left to do but drain. The default path instead reads the scales
    // into VGPRs after the data drain and scatters them, which costs a second
    // HBM round trip serialized behind the first. Only waves 0-1 participate:
    // TILE_SCALE is 16*NUM_BLOCKS_32 bytes, well under the 2 KiB two waves
    // move, and every lane past that would just re-fetch the clamped tail.
    if (warp_id < 2) {
      constexpr int SCALE_VECTORS = (G::TILE_SCALE + 15) / 16;
      int const vector = warp_id * 64 + (tid & 63);
      int const clamped = vector < SCALE_VECTORS ? vector : SCALE_VECTORS - 1;
      uint32_t voff = wg_voff + static_cast<uint32_t>(G::WG_DATA_BYTES) +
                      static_cast<uint32_t>(t * G::TILE_SCALE) +
                      static_cast<uint32_t>(clamped * 16);
      auto *lds_dst =
          (__attribute__((address_space(3)))
           uint32_t *)((uint8_t __attribute__((address_space(3))) *)
                           lds_warp_base +
                       t * G::TILE_BYTES + G::TILE_DATA_PADDED);
      __llvm_amdgcn_raw_buffer_load_lds(
          rsrc, lds_dst, 16, static_cast<int>(voff), 0, 0, 3);
    }
#endif
  }
}

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

// ── N-axis-packed RoPE epilogue ─────────────────────────────────────────
// Same three steps as above, with a token dimension. The MFMA produced
// D[m][n] for all 16 n columns; lane (g, col) holds m = wave_tile*16 + g*4 + i
// for token n = col, so step 1 writes s_rope[col][d0+i] instead of
// s_rope[d0+i]. Steps 2 and 3 flatten (token, element) into one strided loop
// so all 256 threads stay busy -- HALF = 32, so the single-token version left
// 224 of them idle.
//
// Positions are per token, so cos/sin rows come from s_global_pos[t] rather
// than one precomputed row pointer.
template <int HEAD_DIM, int TOK_ROWS>
__device__ __forceinline__ void _kvupd_rope_epilogue_packed(
    float const *acc,             // [4] accumulator values for this lane
    unsigned short const *bias,   // bias array (partitioned per XCD)
    int wg_idx,                   // workgroup index within XCD
    int wave_tile,                // wave_tile index
    int g,                        // lane group (0..3)
    int col,                      // lane column (0..15) == this lane's token
    int tid,                      // threadIdx.x
    bool tok_active,              // col names a real token
    int n_valid,                  // number of real tokens in this block
    unsigned short const *cos_base, // cos[max_pos][HEAD_DIM] bf16
    unsigned short const *sin_base, // sin[max_pos][HEAD_DIM] bf16
    int const *s_global_pos,        // LDS [TOK_ROWS] absolute positions
    unsigned short *dst,            // destination array
    int dst_stride,                 // stride between tokens in dst
    int tok_row_base,               // first token row of this column block
    int head_offset,                // head offset within dst row
    int output_per_wg,              // OUTPUT_PER_WG
    unsigned short *s_rope)         // LDS buffer [TOK_ROWS][HEAD_DIM] bf16
{
  // Step 1: each lane writes its own token's slice.
  //
  // This guard sits strictly AFTER the caller's MFMA asm block has retired.
  // Masking inactive lanes any earlier would let the compiler sink the exec
  // mask up over the ds_read_b128 that feeds the MFMA, and then 60 of 64 lanes
  // read stale B operands -- which corrupts the *active* lanes too, because a
  // 16x16x128 MFMA is a whole-wave operation.
  if (tok_active) {
    int d0 = wave_tile * 16 + g * 4;
    uint64_t bias4;
    __builtin_memcpy(&bias4, &bias[wg_idx * output_per_wg + d0], 8);
    unsigned short const *b4 = (unsigned short const *)&bias4;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      unsigned bt = (unsigned)b4[i] << 16;
      float bv;
      __builtin_memcpy(&bv, &bt, 4);
      s_rope[col * HEAD_DIM + d0 + i] = _gang_float_to_bf16(acc[i] + bv);
    }
  }
  __syncthreads();

  // Step 2: rotate in place; element d of token t pairs with d + HEAD_DIM/2.
  constexpr int HALF = HEAD_DIM / 2;
  for (int idx = tid; idx < TOK_ROWS * HALF; idx += 256) {
    int const t = idx / HALF;
    if (t >= n_valid) {
      continue;
    }
    int const d = idx - t * HALF;
    unsigned short *row = s_rope + t * HEAD_DIM;
    int const pos = s_global_pos[t];
    unsigned short const *cos_data = cos_base + (long long)pos * HEAD_DIM;
    unsigned short const *sin_data = sin_base + (long long)pos * HEAD_DIM;

    unsigned v0_bits = (unsigned)row[d] << 16;
    unsigned v1_bits = (unsigned)row[d + HALF] << 16;
    float v0, v1;
    __builtin_memcpy(&v0, &v0_bits, 4);
    __builtin_memcpy(&v1, &v1_bits, 4);

    unsigned c_bits = (unsigned)cos_data[d] << 16;
    unsigned s_bits = (unsigned)sin_data[d] << 16;
    float c, s;
    __builtin_memcpy(&c, &c_bits, 4);
    __builtin_memcpy(&s, &s_bits, 4);

    row[d] = _gang_float_to_bf16(v0 * c - v1 * s);
    row[d + HALF] = _gang_float_to_bf16(v0 * s + v1 * c);
  }
  __syncthreads();

  // Step 3: write TOK_ROWS * HEAD_DIM bf16 values out.
  for (int idx = tid; idx < TOK_ROWS * HEAD_DIM; idx += 256) {
    int const t = idx / HEAD_DIM;
    if (t >= n_valid) {
      continue;
    }
    int const d = idx - t * HEAD_DIM;
    dst[(long long)(tok_row_base + t) * dst_stride + head_offset + d] =
        s_rope[idx];
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
        float const *ws_base = d_ws + moe_ws_offset(b, 0, REDUCTION_SIZE);
        unsigned short const *res_base = d_residual + b * REDUCTION_SIZE;
        unsigned short *xout_base = d_x_out + b * REDUCTION_SIZE;

#pragma unroll
        for (int off = tid * VEC; off < REDUCTION_SIZE; off += BLOCK_VEC) {
          // Vectorized load: 4 f32 from workspace (flat_load_dwordx4), then
          // sum the remaining expert slots in fixed slot order. See
          // moe_ws_layout.cuh for why this is a per-slot reduction here rather
          // than an atomicAdd in the W2 epilogue.
          float4 ws4;
          __builtin_memcpy(&ws4, ws_base + off, 16);
          // MPK_ABLATE_WS_FOLD: timing-only. Reads slot 0 and drops the other
          // MOE_WS_SLOTS-1 slabs, so the numerics are wrong by three quarters
          // of the MoE output -- never read tokens from a run with this set.
          // It exists because this fold is the largest structural cost in the
          // QKV prologue: the alternative shape is a W2 epilogue that
          // atomicAdds into one slab so the consumer reads one, while the
          // row-symmetry fix here (see moe_ws_layout.cuh) moved that sum into
          // this loop, on every QKV worker, in QKV's serial window. The layout
          // comment asks for exactly this measurement.
#ifndef MPK_ABLATE_WS_FOLD
#pragma unroll
          for (int s = 1; s < MOE_WS_SLOTS; s++) {
            float4 slot4;
            __builtin_memcpy(&slot4, ws_base + s * REDUCTION_SIZE + off, 16);
            ws4.x += slot4.x;
            ws4.y += slot4.y;
            ws4.z += slot4.z;
            ws4.w += slot4.w;
          }
#endif

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

  // ── No workspace zeroing ───────────────────────────────────────────────
  // Every (token, slot, hidden) element is assigned by exactly one W2 tile per
  // layer, so nothing stale survives and there is nothing to clear. The old
  // per-token layout needed this pass because it accumulated with atomicAdd.
  // See moe_ws_layout.cuh.

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
        int tile_idx,
        // True when the fused layer already staged this exact tile's weights
        // into this exact LDS region during the previous layer's Phase 9
        // barrier spin. Defaulted so the standalone generated variant and the
        // layer-0 / world_size>1 caller need no change.
        bool weights_preloaded = false) {

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

  // ── Token activation in shared memory (N-axis packed) ────────────────────
  // The MFMA is 16x16x128: 16 weight rows x 16 token columns. Staging token
  // `col` at LDS row `col` gives lane (g, col) its own token's B operand, so
  // one tile now covers up to 16 batch rows for the same MFMA count and the
  // same weight traffic. See _gang_multirow_fp8_quant for the layout contract.
  constexpr int MFMA_N = 16;
  constexpr int TOK_ROWS = BATCH_SIZE < MFMA_N ? BATCH_SIZE : MFMA_N;
  constexpr int NUM_BBLK = (BATCH_SIZE + TOK_ROWS - 1) / TOK_ROWS;

  // The +16 pad is load-bearing, not alignment slack. At ds_read_b128
  // granularity lane `col` lands in bank group (col * TOK_ROW_STRIDE/16) % 8;
  // with 2960 that is (col*185) % 8 and 185 % 8 == 1, so the 16 lanes spread
  // over all 8 groups. Unpadded 2944 collapses them onto one group.
  constexpr int TOK_ROW_STRIDE = REDUCTION_SIZE + 16;
  constexpr int SC_STRIDE = ((MFMA_ITERS + 3) / 4) * 4;
  constexpr int TOK_REGION = TOK_ROWS * TOK_ROW_STRIDE;
  constexpr int SC_REGION = TOK_ROWS * SC_STRIDE;

  uint8_t const *W = (uint8_t const *)weight_ptr;
  unsigned short const *d_bias = (unsigned short const *)bias_ptr;

  extern __shared__ char _rnlm_smem[];
  uint8_t *s_tok_fp8 = (uint8_t *)_rnlm_smem;
  uint8_t *s_tok_scales = s_tok_fp8 + TOK_REGION;

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

  // Tile space is now (column block, weight group) instead of (token, weight
  // group): the token axis moved into the MFMA's N dimension, so the host
  // emits n_bblk * n_wgs_per_xcd tiles rather than batch_size * n_wgs_per_xcd.
  int bblk = tile_idx / n_wgs_per_xcd;
  int wg_idx = tile_idx % n_wgs_per_xcd;
  int tok_row_base = bblk * MFMA_N;
  int tok_row = tok_row_base + col;

  uint8_t const *wg_data = W + static_cast<int64_t>(wg_idx) * WG_BYTES;
  uint8_t const *wg_scales = wg_data + WG_DATA_BYTES;

  // ── Phase A: Issue HBM→LDS weight prefetch BEFORE RMSNorm ─────────────
  //
  // The geometry lives in QkvWeightLds so the fused layer's Phase 9 prefetch
  // (which stages the NEXT layer's weights into this same region during the
  // barrier spin) cannot drift from what the MFMA below reads.
  using G = QkvWeightLds<BATCH_SIZE, OUTPUT_PER_WG, REDUCTION_SIZE>;
  constexpr int QKV_TILE_SCALE = G::TILE_SCALE;
  constexpr int QKV_TILE_DATA_PADDED = G::TILE_DATA_PADDED;
  constexpr int QKV_TILE_BYTES = G::TILE_BYTES;
  constexpr int QKV_LDS_OFF = G::W_OFF;

  // The real budget is 155 KB minus AMD's 3 KB reserve minus the layer-index
  // word the fused wrapper parks at the top; the old 155*1024 bound was
  // permissive by 3 KB.
  static_assert(G::W_END <= mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE -
                                mirage::runtime::LAYER_IDX_SMEM_OFFSET_FROM_END,
                "QKV LDS weights exceed MI350X LDS budget");
  // The shared geometry must agree with this kernel's own token-region math,
  // or the DMA writes where one says and the MFMA reads where the other says.
  static_assert(G::TOK_REGION == TOK_REGION && G::SC_REGION == SC_REGION,
                "QkvWeightLds token region disagrees with kvupd kernel");
  // RoPE reuses the token region, which is dead once the MFMA has retired.
  static_assert(TOK_ROWS * HEAD_DIM * 2 <= TOK_REGION,
                "packed RoPE scratch must fit in the dead token region");
  uint8_t *qkv_lds_w = (uint8_t *)_rnlm_smem + QKV_LDS_OFF;

  // MPK_ABLATE_QKV_DMA: timing-only upper bound on what any prefetch hoist
  // could recover. Skipping the load leaves garbage weights in LDS, so the
  // numerics are meaningless -- but it is safe to run (the garbage stays in
  // the data path; every address comes from host-built index tables) and the
  // wall clock is the honest ceiling for hiding this DMA perfectly.
  // Measured 2.422 vs 2.486/2.491 ms at B=1 seq512: 64 us, 2.6%.
#ifndef MPK_ABLATE_QKV_DMA
  if (!weights_preloaded) {
    qkv_prefetch_weights_lds<BATCH_SIZE, OUTPUT_PER_WG, REDUCTION_SIZE>(
        weight_ptr, n_wgs_per_xcd, tile_idx);
  }
#endif

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
      // Pass 2's norm weights, hoisted ahead of the cross-wave reduction. See
      // the issue loop below the pass-1 body for why.
      uint64_t prefetched_norm_weights[_MAX_ITERS];
      int n_cached = 0;
      {
        constexpr int VEC = 4;
        constexpr int BLOCK_VEC = 256 * VEC;
        // address_space(1) -- GLOBAL, not generic FLAT. Both buffers are
        // device-global tensors on every path here (the workspace is
        // host-allocated, the residual is a model activation), so the cast is
        // sound; what it buys is that these loads increment vmcnt only. As
        // generic FLAT they bumped lgkmcnt as well, and the compiler then has
        // no way to express "wait for the first two slabs but not the rest" --
        // it must emit a full `s_waitcnt vmcnt(0) lgkmcnt(0)`. The pass-1 body
        // had four of those, one per slab, so the MOE_WS_SLOTS fold ran fully
        // serialized: issue, drain, add, issue, drain, add. With GLOBAL the
        // waits become counted vmcnt(N) and the four slabs overlap.
        //
        // Same reasoning applies to the norm-weight prefetch below, with
        // more force here, because this loop reads MOE_WS_SLOTS slabs where
        // the single-slab shape reads one.
        auto const *ws_base = (__attribute__((address_space(1))) float const *)(
            d_ws + moe_ws_offset(b, 0, REDUCTION_SIZE));
        auto const *res_base =
            (__attribute__((address_space(1))) unsigned short const *)(
                d_residual + b * REDUCTION_SIZE);
        unsigned short *xout_base = d_x_out + b * REDUCTION_SIZE;

#pragma unroll
        for (int off = tid * VEC; off < REDUCTION_SIZE; off += BLOCK_VEC) {
          // Sum the MoE expert contributions here, in fixed slot order, rather
          // than letting the W2 epilogue atomicAdd them into one slab. The
          // order is a compile-time constant, so this row's result does not
          // depend on the order experts happened to retire, nor on what else
          // is in the batch. That is what makes identical prompts in different
          // slots produce identical output. See moe_ws_layout.cuh.
          // Typed AS(1) vector loads rather than __builtin_memcpy: the builtin
          // is declared __host__ and rejects an address-space qualified
          // pointer, and dereferencing the AS(1) type is what makes the
          // backend pick global_load_dwordx4 over flat_load_dwordx4.
          _gl_f32x4 const ws_v =
              *(__attribute__((address_space(1))) _gl_f32x4 const *)(ws_base +
                                                                     off);
          float4 ws4;
          ws4.x = ws_v[0];
          ws4.y = ws_v[1];
          ws4.z = ws_v[2];
          ws4.w = ws_v[3];
          // MPK_ABLATE_WS_FOLD: timing-only. Reads slot 0 and drops the other
          // MOE_WS_SLOTS-1 slabs, so the numerics are wrong by three quarters
          // of the MoE output -- never read tokens from a run with this set.
          // It exists because this fold is the largest structural cost in the
          // QKV prologue: the alternative shape is a W2 epilogue that
          // atomicAdds into one slab so the consumer reads one, while the
          // row-symmetry fix here (see moe_ws_layout.cuh) moved that sum into
          // this loop, on every QKV worker, in QKV's serial window. The layout
          // comment asks for exactly this measurement.
#ifndef MPK_ABLATE_WS_FOLD
#pragma unroll
          for (int s = 1; s < MOE_WS_SLOTS; s++) {
            _gl_f32x4 const sv =
                *(__attribute__((address_space(1))) _gl_f32x4 const *)(
                    ws_base + s * REDUCTION_SIZE + off);
            float4 slot4;
            slot4.x = sv[0];
            slot4.y = sv[1];
            slot4.z = sv[2];
            slot4.w = sv[3];
            ws4.x += slot4.x;
            ws4.y += slot4.y;
            ws4.z += slot4.z;
            ws4.w += slot4.w;
          }
#endif
          _gl_u32x2 const rv_v =
              *(__attribute__((address_space(1))) _gl_u32x2 const *)(res_base +
                                                                     off);
          uint2 res_packed;
          res_packed.x = rv_v[0];
          res_packed.y = rv_v[1];

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

      // ── Pass 2's norm weights, issued here rather than at their use ──────
      //
      // The RMSNorm weights are immutable and pass 2 cannot consume them until
      // after the cross-wave reduction and the two __syncthreads below. Issued
      // here, that whole tree covers the HBM latency; left at the use site
      // (where this used to be) every iteration of pass 2 pays an exposed
      // round trip, and this kernel sits in QKV's serial window on the
      // critical path.
      //
      // `global_load_dwordx2` in asm rather than a plain dereference is
      // load-bearing, not a micro-optimisation. norm_weight_ptr is a
      // device-global tensor on every path, but a generic-pointer FLAT load
      // increments lgkmcnt as well as vmcnt -- so the very first
      // `s_waitcnt lgkmcnt(0)` in the reduction (there is one, for the LDS
      // s_red write) would drain these too and collapse the overlap this
      // exists to create. GLOBAL touches vmcnt only.
#pragma unroll
      for (int iter = 0; iter < _MAX_ITERS; ++iter) {
        int const off = tid * _VEC + iter * _BLOCK_VEC;
        if (off < REDUCTION_SIZE) {
          uint64_t value;
          asm volatile("global_load_dwordx2 %0, %1, off"
                       : "=v"(value)
                       : "v"(d_norm_w + off)
                       : "memory");
          prefetched_norm_weights[iter] = value;
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
        bf16 *out = (bf16 *)d_norm_out + b * REDUCTION_SIZE;
        int ci = 0;
        int wi = 0;
#pragma unroll
        for (int off = tid * VEC; off < REDUCTION_SIZE; off += _BLOCK_VEC) {
          // The inline-asm load that produced this is opaque to the compiler,
          // so nothing here makes it insert a vmcnt wait. What does is the
          // second __syncthreads between the issue site and this use: its
          // fence emits `s_waitcnt vmcnt(0)` ahead of the barrier. That is a
          // property of the generated code rather than of the source, so it is
          // verified in the disassembly -- look for a vmcnt(0) between the
          // global_load_dwordx2 block and this loop before trusting a number
          // from this path.
          uint64_t wv = prefetched_norm_weights[wi++];
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

  _gang_multirow_fp8_quant<REDUCTION_SIZE, TOK_ROWS, BATCH_SIZE, TOK_ROW_STRIDE,
                           SC_STRIDE>(
      (unsigned short const *)norm_scratch_ptr, REDUCTION_SIZE, tok_row_base,
      batch_count - tok_row_base, s_tok_fp8, s_tok_scales);

  // ── Phase B: Drain buffer_load_lds, scatter scales to LDS ──
  {
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");

#ifndef MPK_QKV_PREFETCH_SCALES
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
#endif // MPK_QKV_PREFETCH_SCALES

    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    __syncthreads();
  }

  // Whole-block early-out only: a column block with no real tokens at all.
  // Individual inactive lanes within a live block must NOT return here -- they
  // still have to issue their ds_reads and their share of the MFMA, and they
  // participate in the __syncthreads inside the RoPE epilogue.
  int const n_valid_tok = batch_count - tok_row_base;
  if (n_valid_tok <= 0) {
    return;
  }
  bool const tok_active = col < n_valid_tok;

  // ── Per-token position metadata ────────────────────────────────────────
  // The single-token path derived request_id / global_pos / dst_idx from one
  // tok_idx; every thread computed the same values in registers. With 16
  // tokens in flight each column needs its own set, so 16 threads compute them
  // into LDS and everyone reads them back.
  //
  // At TOK_ROWS == 1 that broadcast is pure overhead -- one barrier and 128 B
  // of static LDS to publish a value every thread could recompute. So that
  // case keeps the register form and the tables become thread-private arrays
  // of length 1, which the consumers index identically.
  int _priv_global_pos[TOK_ROWS];
  int _priv_dst_idx[TOK_ROWS];
  __shared__ int s_pos_tbl[TOK_ROWS > 1 ? 2 * MFMA_N : 1];
  int const *s_global_pos;
  int const *s_dst_idx;

  auto compute_pos = [&](int t, int &gp_out, int &dst_out) {
    int my_tok = tok_row_base + (t < n_valid_tok ? t : 0);
    int request_id = 0;
    while (qo_indptr[request_id + 1] <= my_tok) {
      request_id++;
    }
    int token_in_request = my_tok - qo_indptr[request_id];
    int first_page = kv_indptr[request_id];
    int num_pages = kv_indptr[request_id + 1] - first_page;
    int last_page_len_val = kv_last_page_len[request_id];
    int global_seq_len = (num_pages - 1) * PAGE_SIZE + last_page_len_val;
    int num_new_tokens = qo_indptr[request_id + 1] - qo_indptr[request_id];
    gp_out = global_seq_len - num_new_tokens + token_in_request;
    dst_out = kv_indices[first_page + gp_out / PAGE_SIZE] * PAGE_SIZE +
              gp_out % PAGE_SIZE;
  };

  if constexpr (TOK_ROWS == 1) {
    compute_pos(0, _priv_global_pos[0], _priv_dst_idx[0]);
    s_global_pos = _priv_global_pos;
    s_dst_idx = _priv_dst_idx;
  } else {
    if (tid < TOK_ROWS) {
      int gp, dst;
      compute_pos(tid, gp, dst);
      s_pos_tbl[tid] = gp;
      s_pos_tbl[MFMA_N + tid] = dst;
    }
    __syncthreads();
    s_global_pos = s_pos_tbl;
    s_dst_idx = s_pos_tbl + MFMA_N;
  }

#ifdef MPK_ENABLE_SUBPHASE_TIMING
  if (_sp_rec) {
    _sp_t3 = __builtin_amdgcn_s_memrealtime();
  }
#endif

  // ── MFMA + KV Update epilogue ─────────────────────────────────────────
  // ── QKV asm MFMA loop (weights already in LDS from Phase A/B) ──
  {
    uint8_t *lds_qkv_base = (uint8_t *)_rnlm_smem + QKV_LDS_OFF;

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
        // B operand: lane (g, col) reads token row `col`. Inactive lanes clamp
        // to row 0 rather than skipping -- see the exec-mask note in
        // _kvupd_rope_epilogue_packed.
        int const b_row = tok_active ? col : 0;
        unsigned t_addr =
            (unsigned)(uintptr_t)(s_tok_fp8 + b_row * TOK_ROW_STRIDE + g * 16);
        unsigned ts_addr =
            (unsigned)(uintptr_t)(s_tok_scales + b_row * SC_STRIDE);

        asm volatile(
            // Zero accumulator
            "v_accvgpr_write_b32 a0, 0\n"
            "v_accvgpr_write_b32 a1, 0\n"
            "v_accvgpr_write_b32 a2, 0\n"
            "v_accvgpr_write_b32 a3, 0\n"

            // ── Two disjoint operand banks (see below) ──
            //   Bank 0: A v[22:25], A scale v7,  B v[8:15],  B scale v16
            //   Bank 1: A v[26:29], A scale v18, B v[32:39], B scale v19
            //   Address scratch v17, accumulator a[0:3].
            //
            // The loop MUST NOT prefetch into the registers the current MFMA
            // reads as sources. lgkmcnt tracks when LDS data lands in the
            // VGPR; it says nothing about when the MFMA has finished sampling
            // its operands, and a 16x16x128 MFMA streams them over the op's
            // duration rather than latching at issue. Overwriting in place is
            // therefore a WAR race: when LDS returns fast the write-back lands
            // mid-MFMA and the op sees mixed-iteration operands. Measured at
            // ~17-22% of launches wrong before banking.
            //
            // Ping-pong rule: while the MFMA consumes bank X, the prefetch
            // writes bank 1-X. No register is ever both a live MFMA source and
            // an in-flight LDS destination, so the race cannot occur.
            //
            // Verified by tests/standalone/test_mfma_pipeline_hazards.hip.

            // Pre-issue 5 reads for iteration 0 into bank 0
            "ds_read_b128 v[22:25], %[wa]\n"           // weight A (16B FP4)
            "ds_read_u8   v7, %[wsa]\n"                // weight A scale
            "ds_read_b128 v[8:11], %[ta]\n"            // token B lo (16B FP8)
            "ds_read_b128 v[12:15], %[ta] offset:64\n" // token B hi (16B FP8)
            "ds_read_u8   v16, %[tsa]\n"               // token B scale
            "s_mov_b32 s13, 0\n"                       // loop counter

            // ── MFMA_ITERS-1 in-loop MFMAs, alternating banks ──
            "PIPELINED_QKV_%=:\n"

            // ---- consume bank 0, prefetch into bank 1 ----
            "s_waitcnt lgkmcnt(0)\n" // bank 0 operands resident

            // Advance addresses for the next iteration
            "v_add_u32_e32 %[wa], 64, %[wa]\n"
            "v_add_u32_e32 %[wsa], 4, %[wsa]\n"
            "v_add_u32_e32 %[ta], 0x80, %[ta]\n"
            "s_add_i32 s13, s13, 1\n"

            "v_add_u32_e32 v17, s13, %[tsa]\n"
            "ds_read_u8   v19, v17\n"                  // B scale  -> bank 1
            "ds_read_b128 v[26:29], %[wa]\n"           // A data   -> bank 1
            "ds_read_u8   v18, %[wsa]\n"               // A scale  -> bank 1
            "ds_read_b128 v[32:35], %[ta]\n"           // B lo     -> bank 1
            "ds_read_b128 v[36:39], %[ta] offset:64\n" // B hi     -> bank 1

            "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[8:15], "
            "a[0:3], v7, v16 op_sel_hi:[0,0,0] cbsz:4\n"

            "s_cmpk_lt_i32 s13, %[iters_m1]\n"
            "s_cbranch_scc0 QKV_TAIL_B1_%=\n"

            // ---- consume bank 1, prefetch into bank 0 ----
            "s_waitcnt lgkmcnt(0)\n" // bank 1 operands resident

            "v_add_u32_e32 %[wa], 64, %[wa]\n"
            "v_add_u32_e32 %[wsa], 4, %[wsa]\n"
            "v_add_u32_e32 %[ta], 0x80, %[ta]\n"
            "s_add_i32 s13, s13, 1\n"

            "v_add_u32_e32 v17, s13, %[tsa]\n"
            "ds_read_u8   v16, v17\n"                  // B scale  -> bank 0
            "ds_read_b128 v[22:25], %[wa]\n"           // A data   -> bank 0
            "ds_read_u8   v7, %[wsa]\n"                // A scale  -> bank 0
            "ds_read_b128 v[8:11], %[ta]\n"            // B lo     -> bank 0
            "ds_read_b128 v[12:15], %[ta] offset:64\n" // B hi     -> bank 0

            "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[32:39], "
            "a[0:3], v18, v19 op_sel_hi:[0,0,0] cbsz:4\n"

            "s_cmpk_lt_i32 s13, %[iters_m1]\n"
            "s_cbranch_scc1 PIPELINED_QKV_%=\n"

            // ── Final MFMA ──
            // Exit parity matters, so BOTH tails are emitted. In the live
            // instantiation REDUCTION_SIZE is 2944, so MFMA_ITERS is 23 (odd)
            // and the loop falls out of the bank 1 half with the final
            // operands in BANK 0 -- this path. An even MFMA_ITERS exits via
            // QKV_TAIL_B1 with the final operands in bank 1. Emitting only one
            // tail would silently use the wrong bank for one of the two
            // parities and reintroduce mixed-iteration operands.
            "s_waitcnt lgkmcnt(0)\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[8:15], "
            "a[0:3], v7, v16 op_sel_hi:[0,0,0] cbsz:4\n"
            "s_branch QKV_ACC_%=\n"

            "QKV_TAIL_B1_%=:\n"
            "s_waitcnt lgkmcnt(0)\n"
            "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[32:39], "
            "a[0:3], v18, v19 op_sel_hi:[0,0,0] cbsz:4\n"

            "QKV_ACC_%=:\n"
            // 32 clocks before reading the accumulator. The scaled MFMA is a
            // 32-cycle op on CDNA4; the previous "s_nop 7; s_nop 0" was 9
            // clocks (the correct wait for a 4-pass MFMA) and returned a
            // partially-retired accumulator on every launch.
            "s_nop 15\n"
            "s_nop 15\n"
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
              "v18",
              "v19",
              "v22",
              "v23",
              "v24",
              "v25",
              "v26",
              "v27",
              "v28",
              "v29",
              "v32",
              "v33",
              "v34",
              "v35",
              "v36",
              "v37",
              "v38",
              "v39",
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

      // ── Fused KV_UPD epilogue (N-axis packed) ──────────────────────────
      // Position metadata is per token and already in s_global_pos /
      // s_dst_idx. RoPE scratch aliases the token region, which is dead now
      // that the MFMA has retired.
      //
      // "Retired" is a per-WAVE property, and this is a 4-wave block. The asm
      // above ends with s_waitcnt lgkmcnt(0), so when *this* wave falls out of
      // it, *its* ds_reads have landed -- but nothing has ordered it against
      // the other three waves, which may still be issuing the ds_read_b128
      // pair that feeds their own MFMA. s_rope is _rnlm_smem, i.e. byte 0 of
      // s_tok_fp8, so wave 0's step-1 stores land exactly on token row 0's
      // activations: s_rope halfword (t*HEAD_DIM + ...) is byte 128*t + ...,
      // and token row 0's MFMA iteration i reads bytes [128*i, 128*i+128).
      // Wave 0 writing rope token t therefore destroys iteration t of token
      // row 0 for whichever wave has not read it yet.
      //
      // That makes the corruption window exactly TOK_ROWS of the 23 MFMA
      // iterations -- it scales with the batch width, is confined to token row
      // 0 (col == 0), and vanishes at TOK_ROWS == 1 only because a single
      // iteration is a narrow enough target to almost always lose the race.
      // It is why row 0's Q/K diverged from rows 1..B-1 with identical
      // prompts, and why B=8 was the reliable repro.
      //
      // One block-wide rendezvous closes it: every wave's ds_reads are
      // complete when it arrives here, so the aliased region really is dead
      // before the first store. The epilogue's own __syncthreads calls are all
      // *after* its step-1 stores and cannot substitute.
      __syncthreads();
      int kv_head = _kvupd_get_xcd_id();
      unsigned short const *cos_base = (unsigned short const *)cos_ptr;
      unsigned short const *sin_base = (unsigned short const *)sin_ptr;
      unsigned short *s_rope = (unsigned short *)_rnlm_smem;
      if (wg_idx < NUM_Q_PER_KV) {
        int q_head_global = kv_head * NUM_Q_PER_KV + wg_idx;
        _kvupd_rope_epilogue_packed<HEAD_DIM, TOK_ROWS>(
            (float const *)&acc,
            d_bias,
            wg_idx,
            wave_tile,
            g,
            col,
            tid,
            tok_active,
            n_valid_tok < TOK_ROWS ? n_valid_tok : TOK_ROWS,
            cos_base,
            sin_base,
            s_global_pos,
            (unsigned short *)q_workspace_ptr,
            q_ws_stride,
            tok_row_base,
            q_head_global * HEAD_DIM,
            OUTPUT_PER_WG,
            s_rope);
      } else if (wg_idx == NUM_Q_PER_KV) {
        // K: bias + RoPE, then scatter to the paged cache. Same three steps as
        // the Q path but the destination row is a page slot, so it cannot go
        // through the shared helper's contiguous store.
        int const n_tok = n_valid_tok < TOK_ROWS ? n_valid_tok : TOK_ROWS;
        if (tok_active) {
          int d0 = wave_tile * 16 + g * 4;
          uint64_t bias4;
          __builtin_memcpy(&bias4, &d_bias[wg_idx * OUTPUT_PER_WG + d0], 8);
          unsigned short const *b4 = (unsigned short const *)&bias4;
#pragma unroll
          for (int i = 0; i < 4; i++) {
            unsigned bt = (unsigned)b4[i] << 16;
            float bv;
            __builtin_memcpy(&bv, &bt, 4);
            s_rope[col * HEAD_DIM + d0 + i] = _gang_float_to_bf16(acc[i] + bv);
          }
        }
        __syncthreads();
        constexpr int HALF = HEAD_DIM / 2;
        for (int idx = tid; idx < TOK_ROWS * HALF; idx += 256) {
          int t = idx / HALF;
          if (t >= n_tok) {
            continue;
          }
          int d = idx - t * HALF;
          unsigned short *row = s_rope + t * HEAD_DIM;
          int pos = s_global_pos[t];
          unsigned short const *cos_row = cos_base + (long long)pos * HEAD_DIM;
          unsigned short const *sin_row = sin_base + (long long)pos * HEAD_DIM;
          unsigned v0b = (unsigned)row[d] << 16;
          unsigned v1b = (unsigned)row[d + HALF] << 16;
          float v0, v1;
          __builtin_memcpy(&v0, &v0b, 4);
          __builtin_memcpy(&v1, &v1b, 4);
          unsigned cb = (unsigned)cos_row[d] << 16;
          unsigned sb_r = (unsigned)sin_row[d] << 16;
          float c, s;
          __builtin_memcpy(&c, &cb, 4);
          __builtin_memcpy(&s, &sb_r, 4);
          row[d] = _gang_float_to_bf16(v0 * c - v1 * s);
          row[d + HALF] = _gang_float_to_bf16(v0 * s + v1 * c);
        }
        __syncthreads();
        unsigned short *d_k = (unsigned short *)k_cache_ptr;
        for (int idx = tid; idx < TOK_ROWS * HEAD_DIM; idx += 256) {
          int t = idx / HEAD_DIM;
          if (t >= n_tok) {
            continue;
          }
          int d = idx - t * HEAD_DIM;
          d_k[(long long)s_dst_idx[t] * kv_stride + kv_head * HEAD_DIM + d] =
              s_rope[idx];
        }
      } else {
        // V: no RoPE, so lane (g, col) stores its own 4 values directly.
        unsigned short *d_v = (unsigned short *)v_cache_ptr;
        if (tok_active) {
          int d0 = wave_tile * 16 + g * 4;
          uint64_t bias4;
          __builtin_memcpy(&bias4, &d_bias[wg_idx * OUTPUT_PER_WG + d0], 8);
          unsigned short const *b4 = (unsigned short const *)&bias4;
          long long row_off =
              (long long)s_dst_idx[col] * kv_stride + kv_head * HEAD_DIM;
#pragma unroll
          for (int i = 0; i < 4; i++) {
            unsigned bt = (unsigned)b4[i] << 16;
            float bv;
            __builtin_memcpy(&bv, &bt, 4);
            d_v[row_off + d0 + i] = _gang_float_to_bf16(acc[i] + bv);
          }
        }
      }
    }
  }

  // ── No workspace zeroing ───────────────────────────────────────────────
  // The old layout accumulated with atomicAdd, so it had to be cleared before
  // the next layer could add into it. The per-slot layout is *assigned*, not
  // accumulated: every (token, slot, hidden) element is written exactly once
  // per layer by its one owning W2 tile, so there is nothing stale to clear.
  // Dropping this also removes a full write-through pass over the workspace
  // (st_wt bypasses L2 to HBM) from every layer. See moe_ws_layout.cuh.
  //
  // This holds only while coverage is total: a token routes to exactly
  // MOE_WS_SLOTS experts and wg_idx partitions the hidden axis. Padding expert
  // slots have no active token, write nothing, and own no (token, slot) pair,
  // so they cannot leave a hole.

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
