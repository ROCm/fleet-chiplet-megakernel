/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

// Full-layer + LM-head + argmax fused gang task for MI300/MI350.
// Extends gang_full_layer_fused_mi300 (type 216) with tail phases:
//
//   Phases 1-8: same as type 216 (QKV, Attn, O-proj, TopK, MoE)
//   Phase 8b: MoE global barrier (all 240 workers)
//   Phase 9:  Residual add (bf16 = f32_workspace + residual, zero workspace)
//   Phase 9b: Resadd barrier
//   Phase 10: LM head RMSNorm + MXFP4 GEMM with fused argmax
//             (argmax accumulated in registers during GEMM epilogue;
//              logits never written to HBM)
//   Phase 10b: LM head global barrier
//   Phase 11: Cross-worker/XCD argmax reduce
//
// Counter buffer slot map (extends type 216's layout):
//   Slots 0..43*16:  type 216's counters (attn_xcd_release[7] ends at 44*16-1)
//   Slot 44*16:      moe_done_global (cross-XCD MoE barrier)
//   Slot 45*16:      resadd_done (resadd completion flag)
//   Slot 46*16:      lmhead_done_global (cross-XCD LM head barrier)
//   Slot 47*16:      argmax_packed[0..7] (per-XCD packed float val + int32 idx)
//
// Total counter buffer: 48*16 = 768 ints (fits in 896 allocation)

#pragma once
#include "tasks/mi300/gang_full_layer_fused_mi300.cuh"
#include "tasks/mi300/gang_rmsnorm_linear_mxfp4_bias_mi300.cuh"
#include "tasks/mi300/moe_residual_add_f32_mi300.cuh"

namespace kernel {

// Type 216 uses slots 0..43*16 (attn_xcd_release[7] ends at 44*16-1 = 703).
// Tail counters start at 44*16 = 704.
static constexpr int FUSED_TAIL_MOE_DONE_SLOT = 44 * 16;
static constexpr int FUSED_TAIL_RESADD_DONE_SLOT = 45 * 16;
static constexpr int FUSED_TAIL_LMHEAD_DONE_SLOT = 46 * 16;
// Packed (float val, int32 idx) per XCD — 8 slots × 8 bytes = 64 bytes = 16
// ints
static constexpr int FUSED_TAIL_ARGMAX_PACKED_SLOT = 47 * 16;

template <int QKV_BATCH_SIZE,
          int QKV_OUTPUT_PER_WG,
          int QKV_REDUCTION_SIZE,
          int ACTUAL_HIDDEN_DIM,
          int HEAD_DIM,
          int NUM_Q_PER_KV,
          int PAGE_SIZE,
          int MAX_SEQ_LEN,
          int NUM_KV_CHUNKS,
          int Q_WORKSPACE_STRIDE,
          int KV_CACHE_STRIDE,
          int NUM_KV_HEADS,
          int SLIDING_WINDOW,
          int HAS_SINKS,
          int OPROJ_OUTPUT_PER_WG,
          int OPROJ_REDUCTION_SIZE,
          int NUM_EXPERTS,
          int TOPK_K,
          int MOE_INTERMEDIATE_SIZE,
          int MOE_HIDDEN_SIZE,
          int MOE_W13_OUTPUT_PER_WG,
          int MOE_W2_OUTPUT_PER_WG,
          int LM_OUTPUT_PER_WG,
          int LM_REDUCTION_SIZE>
__device__ __noinline__ void
    gang_full_layer_with_lmhead_fused_kernel_mi300(void *const *input_ptrs,
                                                   void *const *output_ptrs,
                                                   void const *cos_ptr,
                                                   void const *sin_ptr,
                                                   int const *qo_indptr,
                                                   int const *kv_indptr,
                                                   int const *kv_indices,
                                                   int const *kv_last_page_len,
                                                   int num_active_tokens,
                                                   int qkv_n_wgs_per_xcd,
                                                   int kv_stride,
                                                   int q_ws_stride,
                                                   float attn_scale,
                                                   int total_qkv_tiles_per_xcd,
                                                   int oproj_n_wgs_per_xcd,
                                                   int oproj_output_stride,
                                                   int router_tile_n,
                                                   int total_oproj_tiles,
                                                   int total_topk_tiles,
                                                   int oproj_tiles_per_xcd,
                                                   int moe_total_tiles_per_xcd,
                                                   int workers_per_xcd,
                                                   int lm_n_wgs_per_xcd,
                                                   int lm_output_stride,
                                                   int lm_actual_hidden_dim,
                                                   int tile_idx) {
  return; // TEMPORARILY skip entire fused task to measure fixed overhead
  // ══════════════════════════════════════════════════════════════════
  // Phases 1-8: run type 216 (QKV, Attn, O-proj, TopK, MoE)
  // ══════════════════════════════════════════════════════════════════
  gang_full_layer_fused_kernel_mi300<QKV_BATCH_SIZE,
                                     QKV_OUTPUT_PER_WG,
                                     QKV_REDUCTION_SIZE,
                                     ACTUAL_HIDDEN_DIM,
                                     HEAD_DIM,
                                     NUM_Q_PER_KV,
                                     PAGE_SIZE,
                                     MAX_SEQ_LEN,
                                     NUM_KV_CHUNKS,
                                     Q_WORKSPACE_STRIDE,
                                     KV_CACHE_STRIDE,
                                     NUM_KV_HEADS,
                                     SLIDING_WINDOW,
                                     HAS_SINKS,
                                     OPROJ_OUTPUT_PER_WG,
                                     OPROJ_REDUCTION_SIZE,
                                     NUM_EXPERTS,
                                     TOPK_K,
                                     MOE_INTERMEDIATE_SIZE,
                                     MOE_HIDDEN_SIZE,
                                     MOE_W13_OUTPUT_PER_WG,
                                     MOE_W2_OUTPUT_PER_WG>(
      input_ptrs,
      output_ptrs,
      cos_ptr,
      sin_ptr,
      qo_indptr,
      kv_indptr,
      kv_indices,
      kv_last_page_len,
      num_active_tokens,
      qkv_n_wgs_per_xcd,
      kv_stride,
      q_ws_stride,
      attn_scale,
      total_qkv_tiles_per_xcd,
      oproj_n_wgs_per_xcd,
      oproj_output_stride,
      router_tile_n,
      total_oproj_tiles,
      total_topk_tiles,
      oproj_tiles_per_xcd,
      moe_total_tiles_per_xcd,
      workers_per_xcd,
      tile_idx);

  // input_ptrs layout (28 inputs):
  //  [0..23] same as type 216
  //  [24] lm_norm_weight   [25] lm_norm_scratch   [26] lm_mxfp4_weight   [27]
  //  lm_bias
  //
  // output_ptrs layout (13 outputs):
  //  [0..10] same as type 216
  //  [11] lm_logits         [12] argmax_output

  int xcd_id;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcd_id));
  int xcd_rank = tile_idx % workers_per_xcd;
  int tid = threadIdx.x;
  int total_workers = workers_per_xcd * 8;

  int *oproj_counters_base = static_cast<int *>(input_ptrs[16]);
  int *moe_done_global = oproj_counters_base + FUSED_TAIL_MOE_DONE_SLOT;
  int *resadd_done = oproj_counters_base + FUSED_TAIL_RESADD_DONE_SLOT;
  int *lmhead_done_global = oproj_counters_base + FUSED_TAIL_LMHEAD_DONE_SLOT;

  // Division-based epoch targets: ((cur / stride) + 1) * stride
  __shared__ int s_moe_expected;
  __shared__ int s_resadd_expected;
  __shared__ int s_lmhead_expected;

  if (tid == 0) {
    int cur_moe = __atomic_load_n(moe_done_global, __ATOMIC_RELAXED);
    s_moe_expected = ((cur_moe / total_workers) + 1) * total_workers;
    int cur_resadd = __atomic_load_n(resadd_done, __ATOMIC_RELAXED);
    s_resadd_expected = cur_resadd + 1;
    int cur_lmhead = __atomic_load_n(lmhead_done_global, __ATOMIC_RELAXED);
    s_lmhead_expected = ((cur_lmhead / total_workers) + 1) * total_workers;
  }
  __syncthreads();
  int moe_expected = s_moe_expected;
  int resadd_expected = s_resadd_expected;
  int lmhead_expected = s_lmhead_expected;

#ifdef MPK_FUSED_TAIL_TIMING
  uint64_t t_moe_barrier_start, t_moe_barrier_end;
  if (tid == 0) {
    asm volatile("s_memrealtime %0" : "=s"(t_moe_barrier_start));
  }
#endif
  // ══════════════════════════════════════════════════════════════════
  // Phase 8b: MoE global barrier
  // All 240 workers must finish Phase 8 MoE before residual add.
  // Each worker increments, then polls until counter reaches target.
  // Also serves as a publish barrier for the argmax slot init above.
  // ══════════════════════════════════════════════════════════════════
  __syncthreads();
  if (tid == 0) {
    atom_add_release_gpu_s32(moe_done_global, 1);
  }
  if (tid == 0) {
    while (__atomic_load_n(moe_done_global, __ATOMIC_RELAXED) < moe_expected) {
      __builtin_amdgcn_s_sleep(1);
    }
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  }
  __syncthreads();
  asm volatile("buffer_inv" ::: "memory");
#ifdef MPK_FUSED_TAIL_TIMING
  if (tid == 0) {
    asm volatile("s_memrealtime %0" : "=s"(t_moe_barrier_end));
  }
#endif

#ifdef MPK_FUSED_TAIL_TIMING
  uint64_t t_resadd_start;
  if (tid == 0) {
    asm volatile("s_memrealtime %0" : "=s"(t_resadd_start));
  }
#endif
  // ══════════════════════════════════════════════════════════════════
  // Phase 9: Residual add
  // output[0] = bf16(output[10] + output[5])  (workspace_f32 + residual)
  // Also zeros workspace_f32 for next iteration.
  // Only worker 0 on XCD 0 does this (single WG, 256 threads).
  // ══════════════════════════════════════════════════════════════════
  if (xcd_id == 0 && xcd_rank == 0) {
    moe_residual_add_f32_mi300_impl<QKV_BATCH_SIZE,
                                    LM_REDUCTION_SIZE,
                                    LM_REDUCTION_SIZE>(
        output_ptrs[10], // moe_workspace_f32
        output_ptrs[5],  // attn_proj_out (residual)
        output_ptrs[0]); // x_output
    __syncthreads();
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    if (tid == 0) {
      atom_add_release_gpu_s32(resadd_done, 1);
    }
  }

  // ══════════════════════════════════════════════════════════════════
  // Phase 9b: Resadd barrier
  // All workers wait for resadd to complete.
  // ══════════════════════════════════════════════════════════════════
  if (tid == 0) {
    while (__atomic_load_n(resadd_done, __ATOMIC_RELAXED) < resadd_expected) {
      __builtin_amdgcn_s_sleep(1);
    }
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  }
  __syncthreads();
  asm volatile("buffer_inv" ::: "memory");
#ifdef MPK_FUSED_TAIL_TIMING
  uint64_t t_resadd_end;
  if (tid == 0) {
    asm volatile("s_memrealtime %0" : "=s"(t_resadd_end));
  }
#endif

  // ══════════════════════════════════════════════════════════════════
  // Phase 10+11 fused: LM head GEMM with inline argmax
  // RMSNorm + FP8 quant done ONCE, then loop only MFMA tiles.
  // Each worker accumulates running max in registers across all tiles.
  // No logits written to HBM.
  // ══════════════════════════════════════════════════════════════════
#ifdef MPK_FUSED_TAIL_TIMING
  uint64_t t_lmhead_start, t_lmhead_end;
  if (tid == 0) {
    asm volatile("s_memrealtime %0" : "=s"(t_lmhead_start));
  }
#endif

  float *argmax_packed_base = reinterpret_cast<float *>(
      oproj_counters_base + FUSED_TAIL_ARGMAX_PACKED_SLOT);

  // Initialize per-XCD argmax packed slots to (-inf, -1).
  // Worker 0 per XCD writes; the lmhead barrier ensures visibility.
  if (xcd_rank == 0 && tid == 0) {
    float neg_inf = -1e30f;
    int neg_inf_bits;
    __builtin_memcpy(&neg_inf_bits, &neg_inf, 4);
    int neg_one = -1;
    unsigned long long init_packed =
        (static_cast<unsigned long long>(static_cast<unsigned int>(neg_one))
         << 32) |
        static_cast<unsigned int>(neg_inf_bits);
    reinterpret_cast<unsigned long long *>(&argmax_packed_base[xcd_id * 4])[0] =
        init_packed;
  }

  // ── Step 1: RMSNorm (once for all tiles) ──────────────────────────
  {
    int batch_count = (num_active_tokens < QKV_BATCH_SIZE) ? num_active_tokens
                                                           : QKV_BATCH_SIZE;
    for (int b = 0; b < batch_count; b++) {
      unsigned short const *row_in =
          (unsigned short const *)output_ptrs[0] + b * LM_REDUCTION_SIZE;
      unsigned short *row_out =
          (unsigned short *)input_ptrs[25] + b * LM_REDUCTION_SIZE;
      gang_rmsnorm_detail::rmsnorm_inline_amd<LM_REDUCTION_SIZE,
                                              ACTUAL_HIDDEN_DIM>(
          row_in, input_ptrs[24], row_out);
    }
  }

  // ── Step 2: FP8 quant (once for all tiles) ────────────────────────
  {
    extern __shared__ char _rnlm_smem[];
    uint8_t *s_tok_fp8 = (uint8_t *)_rnlm_smem;
    uint8_t *s_tok_scales = s_tok_fp8 + LM_REDUCTION_SIZE;

    unsigned short const *input_row = (unsigned short const *)input_ptrs[25];
    _gang_wave_parallel_fp8_quant<LM_REDUCTION_SIZE>(
        input_row, s_tok_fp8, s_tok_scales);
  }

  // ── Step 3: MFMA tile loop with argmax ────────────────────────────
  {
    constexpr int LM_NUM_BLOCKS_32 = LM_REDUCTION_SIZE / 32;
    constexpr int LM_WG_DATA_BYTES = LM_OUTPUT_PER_WG * (LM_REDUCTION_SIZE / 2);
    constexpr int LM_WG_SCALE_BYTES = LM_OUTPUT_PER_WG * LM_NUM_BLOCKS_32;
    constexpr int LM_WG_BYTES = LM_WG_DATA_BYTES + LM_WG_SCALE_BYTES;
    constexpr int K_PER_MFMA = 128;
    constexpr int MFMA_ITERS = LM_REDUCTION_SIZE / K_PER_MFMA;
    constexpr int NUM_WAVES = 4;
    constexpr int TILES_PER_WAVE = LM_OUTPUT_PER_WG / 16 / NUM_WAVES;

    extern __shared__ char _rnlm_smem[];
    uint8_t *s_tok_fp8 = (uint8_t *)_rnlm_smem;
    uint8_t *s_tok_scales = s_tok_fp8 + LM_REDUCTION_SIZE;

    uint8_t const *lm_W = (uint8_t const *)input_ptrs[26];
    unsigned short const *lm_bias = (unsigned short const *)input_ptrs[27];

    int const warp_id = tid >> 6;
    int const lane_id = tid & 63;
    int const col = lane_id & 15;
    int const g = lane_id >> 4;

    float thread_max = -1e30f;
    long long thread_max_idx = -1;
    int partition_start = xcd_id * lm_n_wgs_per_xcd * LM_OUTPUT_PER_WG;

    int lm_total_tiles_per_xcd = lm_n_wgs_per_xcd * num_active_tokens;
    for (int lm_t = xcd_rank; lm_t < lm_total_tiles_per_xcd;
         lm_t += workers_per_xcd) {
      int wg_idx = lm_t % lm_n_wgs_per_xcd;
      uint8_t const *wg_data =
          lm_W + static_cast<int64_t>(wg_idx) * LM_WG_BYTES;
      uint8_t const *wg_scales = wg_data + LM_WG_DATA_BYTES;

      for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; tile_iter++) {
        int wave_tile = warp_id + tile_iter * NUM_WAVES;
        int w_row = wave_tile * 16 + col;
        int const row_data_base = w_row * (LM_REDUCTION_SIZE / 2);
        int const row_scale_base = w_row * LM_NUM_BLOCKS_32;

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

        // Argmax epilogue: accumulate in registers, no HBM write
        if (col == 0) {
          for (int i = 0; i < 4; i++) {
            int out_n = wg_idx * LM_OUTPUT_PER_WG + wave_tile * 16 + g * 4 + i;
            float sum = acc[i];
            unsigned bt = (unsigned)lm_bias[out_n] << 16;
            float bv;
            __builtin_memcpy(&bv, &bt, 4);
            float val = sum + bv;
            long long abs_idx = (long long)(partition_start + out_n);
            if (val > thread_max) {
              thread_max = val;
              thread_max_idx = abs_idx;
            }
          }
        }
      }
    }

// ── Per-worker warp reduce ──────────────────────────────────────
#pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
      float other_val = __shfl_xor(thread_max, offset, 64);
      unsigned int idx_lo =
          static_cast<unsigned int>(thread_max_idx & 0xFFFFFFFF);
      unsigned int idx_hi =
          static_cast<unsigned int>((thread_max_idx >> 32) & 0xFFFFFFFF);
      unsigned int other_lo = __shfl_xor(idx_lo, offset, 64);
      unsigned int other_hi = __shfl_xor(idx_hi, offset, 64);
      long long other_idx = (static_cast<long long>(other_hi) << 32) | other_lo;
      if (other_val > thread_max) {
        thread_max = other_val;
        thread_max_idx = other_idx;
      }
    }

    // Cross-warp reduce via shared memory
    __shared__ float s_max_vals[4];
    __shared__ long long s_max_idxs[4];
    if (lane_id == 0) {
      s_max_vals[warp_id] = thread_max;
      s_max_idxs[warp_id] = thread_max_idx;
    }
    __syncthreads();

    if (tid == 0) {
      float best_val = -1e30f;
      long long best_idx = -1;
      for (int w = 0; w < 4; w++) {
        if (s_max_vals[w] > best_val) {
          best_val = s_max_vals[w];
          best_idx = s_max_idxs[w];
        }
      }
      // 64-bit CAS to atomically update per-XCD max
      unsigned long long *argmax_packed =
          reinterpret_cast<unsigned long long *>(
              &argmax_packed_base[xcd_id * 4]);
      unsigned long long old_packed =
          __atomic_load_n(argmax_packed, __ATOMIC_RELAXED);
      while (true) {
        int old_val_bits = static_cast<int>(old_packed & 0xFFFFFFFF);
        float old_val;
        __builtin_memcpy(&old_val, &old_val_bits, 4);
        if (best_val <= old_val) {
          break;
        }
        int new_val_bits;
        __builtin_memcpy(&new_val_bits, &best_val, 4);
        int new_idx_bits = static_cast<int>(best_idx);
        unsigned long long new_packed =
            (static_cast<unsigned long long>(
                 static_cast<unsigned int>(new_idx_bits))
             << 32) |
            static_cast<unsigned int>(new_val_bits);
        if (__atomic_compare_exchange_n(argmax_packed,
                                        &old_packed,
                                        new_packed,
                                        true,
                                        __ATOMIC_RELAXED,
                                        __ATOMIC_RELAXED)) {
          break;
        }
      }
    }
  }

#ifdef MPK_FUSED_TAIL_TIMING
  if (tid == 0) {
    asm volatile("s_memrealtime %0" : "=s"(t_lmhead_end));
  }
#endif

  // ══════════════════════════════════════════════════════════════════
  // Phase 10b: LM head + argmax global barrier
  // ══════════════════════════════════════════════════════════════════
  __syncthreads();
  if (tid == 0) {
    atom_add_release_gpu_s32(lmhead_done_global, 1);
  }
  if (tid == 0) {
    while (__atomic_load_n(lmhead_done_global, __ATOMIC_RELAXED) <
           lmhead_expected) {
      __builtin_amdgcn_s_sleep(1);
    }
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  }
  __syncthreads();
  asm volatile("buffer_inv" ::: "memory");

  // ══════════════════════════════════════════════════════════════════
  // Phase 11: Cross-XCD argmax reduce
  // Worker 0 on XCD 0 reads 8 per-XCD packed (val, idx) and writes final.
  // ══════════════════════════════════════════════════════════════════
  if (xcd_id == 0 && xcd_rank == 0 && tid == 0) {
    float best_val = -1e30f;
    long long best_idx = -1;
    for (int x = 0; x < 8; x++) {
      unsigned long long packed =
          reinterpret_cast<unsigned long long *>(&argmax_packed_base[x * 4])[0];
      int val_bits = static_cast<int>(packed & 0xFFFFFFFF);
      float v;
      __builtin_memcpy(&v, &val_bits, 4);
      int idx_bits = static_cast<int>(packed >> 32);
      if (v > best_val) {
        best_val = v;
        best_idx = static_cast<long long>(idx_bits);
      }
    }
    static_cast<long long *>(output_ptrs[12])[0] = best_idx;
  }

#ifdef MPK_FUSED_TAIL_TIMING
  if (xcd_id == 0 && xcd_rank == 0 && tid == 0) {
    double ns_per_tick = 10.0;
    double moe_barrier_us =
        (t_moe_barrier_end - t_moe_barrier_start) * ns_per_tick / 1000.0;
    double resadd_us = (t_resadd_end - t_resadd_start) * ns_per_tick / 1000.0;
    double lmhead_us = (t_lmhead_end - t_lmhead_start) * ns_per_tick / 1000.0;
    printf(
        "[FUSED_TAIL] moe_barrier=%.1fus resadd=%.1fus lmhead_argmax=%.1fus\n",
        moe_barrier_us,
        resadd_us,
        lmhead_us);
  }
#endif

  // MOE_SUBPHASE printing moved to gang_full_layer_fused (which is actually
  // dispatched)
}

} // namespace kernel
