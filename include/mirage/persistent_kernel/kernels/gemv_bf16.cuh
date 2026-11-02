/* Copyright 2026 Nikhil (github.com/badnikhil)
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

// out[r] = sum_k W[r,k] * x[k] for the ROWS rows of one tile, fp32 accumulate. K is compile time (512/1408/2048/
// 2736/2816 in the graph), a multiple of 8. Pointers are PRE-OFFSET to the tile by the caller: w = first row of the
// tile, out/res = the tile's first output (so scattered outputs such as the kv_b_proj rows into the K/V cache are
// just a different `out`). Split-K = this kernel with OutT=float and (x, w) pointing at the K-slice;
// splitk_reduce.cuh sums the partials.
// Vectorised 2026-09-14 (design D10d): each wave keeps RPW = 4 rows in flight and issues all
// ceil(K/512) 16-byte loads of those rows before the fma chains (16 independent loads per lane at K = 2048 instead
// of 1). Per row the per-lane fp32 accumulation order (k ascending, then wave_sum) is exactly the previous
// one-row-per-wave kernel's, so outputs are bit-identical (demo/deepseek_v2_lite/tests/test_kernels_cpu.py vs gemv_bf16_v1.cc there).
// RPW_SW (switch DSK_GEMV_RPW, A2): RPW_SW rows per wave in flight for any ROWS, the last trip of a wave may
// be ragged (rows >= ROWS are neither loaded nor stored; their zero accumulator still joins the per-wave wave_sum so all
// lanes stay in lockstep). Row r is handled by the unique (trip, wave, i) with r = 4*RPW*trip + RPW*wave + i, with the
// same per-row fma chain and wave_sum -> bit-identical. 88/176-row W13 tiles: 11 trips per wave -> 6 at RPW 4/8.
// Epilogues mirror HF's bf16 op order: NONE -> bf16(acc); RES -> bf16(res + bf16(acc));
// SILU -> tile rows [0,ROWS/2) are gate rows, [ROWS/2,ROWS) their matching up rows (design.md 4.1),
//         h[j] = bf16(bf16(silu(bf16(g))) * bf16(u)) for j < ROWS/2.
// LDS: x staged as bf16 (2K bytes) + ROWS fp32 partials (+ 4 floats for the NORM prologue's block reduce).
// Prologues (decision 11b): a fused task recomputes the previous single-task op straight into the LDS staging area
// instead of copying x: gemv_bf16_norm_task_impl = rmsnorm.cuh's exact loops (bit-identical to the RMS_NORM task);
// attn_merge.cuh::o_proj_merge_task_impl = the LSE merge. STAGED = the caller already wrote xs.
#pragma once
#include "common.cuh"
#include "rmsnorm.cuh"
#include "switches.h"

namespace dsk {
enum GemvEpi { GEMV_NONE = 0, GEMV_RES = 1, GEMV_SILU = 2 };

// Stage x in LDS and reduce the ROWS rows of the tile into acc[ROWS] (fp32, in LDS). Ends with a __syncthreads.
template <int ROWS, int K, bool STAGED = false, int RPW_SW = DSK_GEMV_RPW>
__device__ __forceinline__ void gemv_rows(const bf16 *x, const bf16 *w, bf16 *xs, float *acc) {
  constexpr int NWAVE = WG_THREADS / WAVE_SIZE;
  constexpr int RPW = RPW_SW ? RPW_SW : ROWS % (NWAVE * 4) == 0 ? 4 : ROWS % (NWAVE * 2) == 0 ? 2 : 1;  // in flight per wave (2: 88-row tiles 11a; 1: 4-row gate tiles L5)
  constexpr int BLK = WAVE_SIZE * 8, NB = (K + BLK - 1) / BLK;   // 512-element k blocks per row (last may be ragged)
  static_assert(K % 8 == 0 && (RPW_SW > 0 || ROWS % (NWAVE * RPW) == 0), "rows split over 4 waves x RPW");
  if constexpr (!STAGED)
    for (int i = threadIdx.x * 8; i < K; i += WG_THREADS * 8) store8(xs + i, load8(x + i));
  __syncthreads();
  const int lane = threadIdx.x & (WAVE_SIZE - 1), wave = threadIdx.x / WAVE_SIZE;
  if constexpr (RPW_SW == 0) {
    for (int r0 = wave * RPW; r0 < ROWS; r0 += NWAVE * RPW) {
      bf16x8 a[RPW][NB];
#pragma unroll
      for (int i = 0; i < RPW; i++)
#pragma unroll
        for (int b = 0; b < NB; b++)
          if (lane * 8 + b * BLK < K) a[i][b] = load8(w + (size_t)(r0 + i) * K + lane * 8 + b * BLK);
      float s[RPW] = {};
#pragma unroll
      for (int b = 0; b < NB; b++) {
        const int k = lane * 8 + b * BLK;
        if (k < K) {
          const bf16x8 xb = load8(xs + k);
#pragma unroll
          for (int i = 0; i < RPW; i++)
#pragma unroll
            for (int j = 0; j < 8; j++) s[i] = fmaf(bf2f(a[i][b].v[j]), bf2f(xb.v[j]), s[i]);
        }
      }
#pragma unroll
      for (int i = 0; i < RPW; i++) {
        const float t = wave_sum(s[i]);
        if (lane == 0) acc[r0 + i] = t;
      }
    }
  } else {
    for (int r0 = wave * RPW; r0 < ROWS; r0 += NWAVE * RPW) {
      bf16x8 a[RPW][NB];
#pragma unroll
      for (int i = 0; i < RPW; i++)
#pragma unroll
        for (int b = 0; b < NB; b++)
          if (r0 + i < ROWS && lane * 8 + b * BLK < K) a[i][b] = load8(w + (size_t)(r0 + i) * K + lane * 8 + b * BLK);
      float s[RPW] = {};
#pragma unroll
      for (int b = 0; b < NB; b++) {
        const int k = lane * 8 + b * BLK;
        if (k < K) {
          const bf16x8 xb = load8(xs + k);
#pragma unroll
          for (int i = 0; i < RPW; i++)
            if (r0 + i < ROWS)
#pragma unroll
              for (int j = 0; j < 8; j++) s[i] = fmaf(bf2f(a[i][b].v[j]), bf2f(xb.v[j]), s[i]);
        }
      }
#pragma unroll
      for (int i = 0; i < RPW; i++) {
        const float t = wave_sum(s[i]);
        if (lane == 0 && r0 + i < ROWS) acc[r0 + i] = t;
      }
    }
  }
  __syncthreads();
}

template <int ROWS, int EPI, typename OutT, int K, bool STAGED = false>
__device__ __forceinline__ void gemv_bf16_task_impl(const void *x_ptr, const void *w_ptr, void *out_ptr,
                                                    const void *res_ptr) {
  static_assert(EPI != GEMV_SILU || ROWS % 8 == 0, "silu pairs gate/up rows");
  extern __shared__ char smem[];
  const bf16 *res = static_cast<const bf16 *>(res_ptr);
  OutT *out = static_cast<OutT *>(out_ptr);
  float *acc = reinterpret_cast<float *>(smem + ((K * 2 + 15) & ~15));
  gemv_rows<ROWS, K, STAGED>(static_cast<const bf16 *>(x_ptr), static_cast<const bf16 *>(w_ptr),
                             reinterpret_cast<bf16 *>(smem), acc);

  if constexpr (EPI == GEMV_SILU) {
    for (int j = threadIdx.x; j < ROWS / 2; j += WG_THREADS) {
      float g = bfround(acc[j]), u = bfround(acc[j + ROWS / 2]);
      out[j] = to_out<OutT>(bfround(silu(g)) * u);
    }
  } else {
    for (int r = threadIdx.x; r < ROWS; r += WG_THREADS) {
      float v = acc[r];
      if constexpr (EPI == GEMV_RES) v = bf2f(res[r]) + bfround(v);
      out[r] = to_out<OutT>(v);
    }
  }
}

// NORM prologue: xs = rmsnorm(x, nw) (K elements, the same loops as the RMS_NORM task), red = 4 floats after acc.
template <int ROWS, int K>
__device__ __forceinline__ void gemv_norm_stage(const void *x_ptr, const void *nw_ptr, float eps) {
  extern __shared__ char smem[];
  rmsnorm_rows(static_cast<const bf16 *>(x_ptr), static_cast<const bf16 *>(nw_ptr), reinterpret_cast<bf16 *>(smem), K,
               eps, reinterpret_cast<float *>(smem + ((K * 2 + 15) & ~15) + ROWS * 4));
}

// RMS_NORM + GEMV in one task (QKV_PROJ, GATE_UP_SILU, LM_HEAD; the KV_B_PROJ tiles via mla_prep.cuh).
template <int ROWS, int EPI, typename OutT, int K>
__device__ __forceinline__ void gemv_bf16_norm_task_impl(const void *x_ptr, const void *nw_ptr, const void *w_ptr,
                                                         void *out_ptr, const void *res_ptr, float eps) {
  gemv_norm_stage<ROWS, K>(x_ptr, nw_ptr, eps);
  gemv_bf16_task_impl<ROWS, EPI, OutT, K, true>(nullptr, w_ptr, out_ptr, res_ptr);
}
}  // namespace dsk
