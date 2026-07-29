/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

// Fused RMSNorm + MXFP4 Gang Linear + Bias + Argmax for MI300/MI350.
//
// CROC-style: norm once, FP8 quant once, internal tile loop over all WGs
// assigned to this worker. Argmax accumulated in registers across all tiles.
// Each worker writes one (bf16 max_val, int64 abs_idx) pair.
//
// total_tiles_per_xcd = workers_per_xcd (30 on MI300X).
// tile_idx = worker rank within XCD (0..29).
// Internal loop: wg = tile_idx, tile_idx + workers_per_xcd, ...
//
// Eliminates:
//   - 393 redundant RMSNorm + FP8 quant (down to 30 — one per worker)
//   - ~786 KB HBM logits write + read
//   - argmax_partial task dispatch

#pragma once
#include "tasks/mi300/gang_moe_linear_mxfp4_mi300.cuh"
#include "tasks/mi300/gang_rmsnorm_linear_bias_mi300.cuh"

namespace kernel {

template <int BATCH_SIZE,
          int OUTPUT_PER_WG,
          int REDUCTION_SIZE,
          int ACTUAL_HIDDEN_DIM = REDUCTION_SIZE>
__device__ __noinline__ void gang_rmsnorm_linear_mxfp4_bias_argmax_kernel(
    void const *norm_input_ptr,
    void const *norm_weight_ptr,
    void *norm_output_ptr,
    void const *weight_ptr,
    void const *bias_ptr,
    void *argmax_val_ptr, // [num_workers] bf16: per-worker max value
    void *argmax_idx_ptr, // [num_workers] int64: per-worker absolute index
    int num_active_tokens,
    int n_wgs_per_xcd,
    int workers_per_xcd,
    int output_stride,
    int tile_idx) { // tile_idx = partition_index * workers_per_xcd +
                    // worker_rank

  static_assert(OUTPUT_PER_WG >= 64,
                "Argmax fusion only supports N-parallel path (OPW>=64)");
  static_assert(OUTPUT_PER_WG % 16 == 0);
  static_assert(REDUCTION_SIZE % 128 == 0);

  constexpr int NUM_BLOCKS_32 = REDUCTION_SIZE / 32;
  constexpr int WG_DATA_BYTES = OUTPUT_PER_WG * (REDUCTION_SIZE / 2);
  constexpr int WG_SCALE_BYTES = OUTPUT_PER_WG * NUM_BLOCKS_32;
  constexpr int WG_BYTES = WG_DATA_BYTES + WG_SCALE_BYTES;
  constexpr int K_PER_MFMA = 128;
  constexpr int MFMA_ITERS = REDUCTION_SIZE / K_PER_MFMA;
  constexpr int NUM_WAVES = 4;
  constexpr int TILES_PER_WAVE = OUTPUT_PER_WG / 16 / NUM_WAVES;

  // LDS weight tile layout (padded for buffer_load_lds alignment)
  constexpr int QKV_TILE_ROWS = 16;
  constexpr int QKV_TILE_DATA = QKV_TILE_ROWS * (REDUCTION_SIZE / 2);
  constexpr int QKV_TILE_SCALE = QKV_TILE_ROWS * NUM_BLOCKS_32;
  constexpr int qkv_n16_data = QKV_TILE_DATA / 16;
  constexpr int QKV_LPT = (qkv_n16_data + 255) / 256;
  constexpr int QKV_TILE_DATA_PADDED = QKV_LPT * 256 * 16;
  constexpr int QKV_TILE_BYTES = QKV_TILE_DATA_PADDED + QKV_TILE_SCALE;

  // FP8 token data sits at start of LDS
  constexpr int FP8_TOK_DATA = REDUCTION_SIZE;
  constexpr int QKV_LDS_OFF = ((FP8_TOK_DATA + MFMA_ITERS + 15) / 16) * 16;
  static_assert(QKV_LDS_OFF + QKV_TILE_BYTES * NUM_WAVES <= 155 * 1024,
                "QKV LDS weights exceed MI350X LDS budget");

  uint8_t const *W = (uint8_t const *)weight_ptr;
  unsigned short const *d_bias = (unsigned short const *)bias_ptr;

  extern __shared__ char _rnlm_smem[];
  uint8_t *s_tok_fp8 = (uint8_t *)_rnlm_smem;
  uint8_t *s_tok_scales = s_tok_fp8 + REDUCTION_SIZE;
  uint8_t *qkv_lds_w = (uint8_t *)_rnlm_smem + QKV_LDS_OFF;

  int const tid = threadIdx.x;
  int const warp_id = tid >> 6;
  int const lane_id = tid & 63;
  int const col = lane_id & 15;
  int const g = lane_id >> 4;

  unsigned short *argmax_vals =
      reinterpret_cast<unsigned short *>(argmax_val_ptr);
  long long *argmax_idxs = reinterpret_cast<long long *>(argmax_idx_ptr);

  int batch_count =
      (num_active_tokens < BATCH_SIZE) ? num_active_tokens : BATCH_SIZE;
  int partition_index = tile_idx / workers_per_xcd; // logical XCD (0..7)
  int worker_rank = tile_idx % workers_per_xcd;     // 0..workers_per_xcd-1
  int partition_start = partition_index * n_wgs_per_xcd * OUTPUT_PER_WG;

  // ── Step 1: RMSNorm (ONCE) ──────────────────────────────────────────
  for (int b = 0; b < batch_count; b++) {
    unsigned short const *row_in =
        (unsigned short const *)norm_input_ptr + b * REDUCTION_SIZE;
    unsigned short *row_out =
        (unsigned short *)norm_output_ptr + b * REDUCTION_SIZE;
    gang_rmsnorm_detail::rmsnorm_inline_amd<REDUCTION_SIZE, ACTUAL_HIDDEN_DIM>(
        row_in, norm_weight_ptr, row_out);
  }

  // ── Step 2: FP8 quant (ONCE) ────────────────────────────────────────
  unsigned short const *input_row = (unsigned short const *)norm_output_ptr;
  _gang_wave_parallel_fp8_quant<REDUCTION_SIZE>(
      input_row, s_tok_fp8, s_tok_scales);

  // ── Step 3: Internal tile loop + argmax ──────────────────────────────
  float thread_max = -1e30f;
  long long thread_max_abs_idx = -1;

  // SRD for weight buffer (hoisted outside loop)
  uint32_t qkv_buf_range = static_cast<uint32_t>(n_wgs_per_xcd) * WG_BYTES;
  i32x4_t qkv_rsrc = make_w_buffer_rsrc(W, qkv_buf_range);

  for (int wg_idx = worker_rank; wg_idx < n_wgs_per_xcd;
       wg_idx += workers_per_xcd) {
    uint32_t qkv_wg_voff = static_cast<uint32_t>(wg_idx) * WG_BYTES;

    // ── Phase A: Prefetch weight data into LDS via buffer_load_dwordx4 lds:1
    // ──
    {
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

    // ── Phase B: Drain data loads, load scales via VGPR, scatter to LDS ──
    constexpr int QKV_SC_DW4_PER_TILE = QKV_TILE_SCALE / 16;
    constexpr int QKV_TOTAL_SC_DW4 = QKV_SC_DW4_PER_TILE * NUM_WAVES;
    constexpr int QKV_SC_LPT = (QKV_TOTAL_SC_DW4 + 255) / 256;

    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");

    // Issue scale loads from HBM via VGPR
    uint8_t const *wg_scales_hbm =
        W + static_cast<int64_t>(wg_idx) * WG_BYTES + WG_DATA_BYTES;
    i32x4_t qkv_sc_buf[QKV_SC_LPT];
    {
      i32x4_t const *sc_src = (i32x4_t const *)wg_scales_hbm;
#pragma unroll
      for (int j = 0; j < QKV_SC_LPT; j++) {
        int idx = tid + j * 256;
        if (idx < QKV_TOTAL_SC_DW4) {
          qkv_sc_buf[j] = sc_src[idx];
        }
      }
    }

    // Drain scale loads, scatter to per-tile LDS slots
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

    // ── Phase C: MFMA loop reading from LDS ──
    for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; tile_iter++) {
      int wave_tile = warp_id + tile_iter * NUM_WAVES;
      uint8_t const *tile_data_lds =
          (uint8_t const *)(qkv_lds_w + wave_tile * QKV_TILE_BYTES);
      uint8_t const *tile_scale_lds = tile_data_lds + QKV_TILE_DATA_PADDED;

      int w_row_in_tile = col;
      int const row_data_base = w_row_in_tile * (REDUCTION_SIZE / 2);
      int const row_scale_base = w_row_in_tile * NUM_BLOCKS_32;

      f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

      i32x8_t a0 =
          *(i32x8_t const *)(tile_data_lds + row_data_base + 0 * 64 + g * 16);
      int sa0 = (int)tile_scale_lds[row_scale_base + 0 * 4 + g];
      i32x8_t a1 =
          *(i32x8_t const *)(tile_data_lds + row_data_base + 1 * 64 + g * 16);
      int sa1 = (int)tile_scale_lds[row_scale_base + 1 * 4 + g];
      i32x8_t a2 =
          *(i32x8_t const *)(tile_data_lds + row_data_base + 2 * 64 + g * 16);
      int sa2 = (int)tile_scale_lds[row_scale_base + 2 * 4 + g];
      i32x8_t a3 =
          *(i32x8_t const *)(tile_data_lds + row_data_base + 3 * 64 + g * 16);
      int sa3 = (int)tile_scale_lds[row_scale_base + 3 * 4 + g];

#pragma unroll 1
      for (int ki = 0; ki < MFMA_ITERS; ki += 4) {
        {
          i32x8_t b = _gang_load_fp8_mfma_b(s_tok_fp8, ki * K_PER_MFMA, g);
          int sb = (int)s_tok_scales[ki];
          acc = _gang_mfma_f4xf8(a0, b, acc, sa0, sb);
        }
        if (ki + 4 < MFMA_ITERS) {
          int kt4 = (ki + 4) * K_PER_MFMA;
          a0 = *(i32x8_t const *)(tile_data_lds + row_data_base + kt4 / 2 +
                                  g * 16);
          sa0 = (int)tile_scale_lds[row_scale_base + kt4 / 32 + g];
        }
        {
          i32x8_t b =
              _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 1) * K_PER_MFMA, g);
          int sb = (int)s_tok_scales[ki + 1];
          acc = _gang_mfma_f4xf8(a1, b, acc, sa1, sb);
        }
        if (ki + 5 < MFMA_ITERS) {
          int kt5 = (ki + 5) * K_PER_MFMA;
          a1 = *(i32x8_t const *)(tile_data_lds + row_data_base + kt5 / 2 +
                                  g * 16);
          sa1 = (int)tile_scale_lds[row_scale_base + kt5 / 32 + g];
        }
        {
          i32x8_t b =
              _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 2) * K_PER_MFMA, g);
          int sb = (int)s_tok_scales[ki + 2];
          acc = _gang_mfma_f4xf8(a2, b, acc, sa2, sb);
        }
        if (ki + 6 < MFMA_ITERS) {
          int kt6 = (ki + 6) * K_PER_MFMA;
          a2 = *(i32x8_t const *)(tile_data_lds + row_data_base + kt6 / 2 +
                                  g * 16);
          sa2 = (int)tile_scale_lds[row_scale_base + kt6 / 32 + g];
        }
        if (ki + 3 < MFMA_ITERS) {
          i32x8_t b =
              _gang_load_fp8_mfma_b(s_tok_fp8, (ki + 3) * K_PER_MFMA, g);
          int sb = (int)s_tok_scales[ki + 3];
          acc = _gang_mfma_f4xf8(a3, b, acc, sa3, sb);
        }
        if (ki + 7 < MFMA_ITERS) {
          int kt7 = (ki + 7) * K_PER_MFMA;
          a3 = *(i32x8_t const *)(tile_data_lds + row_data_base + kt7 / 2 +
                                  g * 16);
          sa3 = (int)tile_scale_lds[row_scale_base + kt7 / 32 + g];
        }
      }

      // Argmax epilogue: accumulate across ALL tiles in registers
      if (col == 0) {
        for (int i = 0; i < 4; i++) {
          int rel_idx = wave_tile * 16 + g * 4 + i;
          int abs_idx = partition_start + wg_idx * OUTPUT_PER_WG + rel_idx;
          float sum = acc[i];
          unsigned bt = (unsigned)d_bias[wg_idx * OUTPUT_PER_WG + rel_idx]
                        << 16;
          float bv;
          __builtin_memcpy(&bv, &bt, 4);
          float val = sum + bv;
          if (val > thread_max) {
            thread_max = val;
            thread_max_abs_idx = (long long)abs_idx;
          }
        }
      }
    }
  } // end tile loop

// ── Warp reduce ─────────────────────────────────────────────────────
#pragma unroll
  for (int offset = 32; offset > 0; offset >>= 1) {
    float other_val = __shfl_xor(thread_max, offset, 64);
    unsigned int idx_lo =
        static_cast<unsigned int>(thread_max_abs_idx & 0xFFFFFFFF);
    unsigned int idx_hi =
        static_cast<unsigned int>((thread_max_abs_idx >> 32) & 0xFFFFFFFF);
    unsigned int other_lo = __shfl_xor(idx_lo, offset, 64);
    unsigned int other_hi = __shfl_xor(idx_hi, offset, 64);
    long long other_idx = (static_cast<long long>(other_hi) << 32) | other_lo;
    if (other_val > thread_max) {
      thread_max = other_val;
      thread_max_abs_idx = other_idx;
    }
  }

  // ── Cross-warp reduce via LDS ─────────────────────────────────────────
  __shared__ float s_max_vals[4];
  __shared__ long long s_max_idxs[4];
  if (lane_id == 0) {
    s_max_vals[warp_id] = thread_max;
    s_max_idxs[warp_id] = thread_max_abs_idx;
  }
  __syncthreads();

  // ── Write per-worker result ───────────────────────────────────────────
  if (tid == 0) {
    float best_val = -1e30f;
    long long best_idx = -1;
    for (int w = 0; w < 4; w++) {
      if (s_max_vals[w] > best_val) {
        best_val = s_max_vals[w];
        best_idx = s_max_idxs[w];
      }
    }
    unsigned int vbits;
    __builtin_memcpy(&vbits, &best_val, 4);
    unsigned short bf16_val = (unsigned short)(vbits >> 16);
    argmax_vals[worker_rank] = bf16_val;
    argmax_idxs[worker_rank] = best_idx;
  }
}

} // namespace kernel
