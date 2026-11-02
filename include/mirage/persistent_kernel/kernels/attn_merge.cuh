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

// LSE merge of NCHUNK partials for one head -> out[head*DV + d] bf16. Same math and partial layout as
// Fleet's ampere/merge_splitkv.cuh (m/lse in log2 units, o pre-normalised, d accumulates exp2 weights);
// Fleet's kernel is a drop-in if the task also gets qo_indptr=[0,1], paged_kv_indptr=[0,1],
// last_page_len=[T] (it derives num_chunks from those).
// Empty chunks carry lse = -inf and contribute exp2(-inf) = 0.
// Decision 12e L6: o_proj_merge_task_impl below runs the merge as O_PROJ_RESIDUAL's prologue (ported from perf/fold-full
// 9379248); the per-head task stays for the chain graph and as the CPU oracle.
#pragma once
#include "gemv_bf16.cuh"
#include "switches.h"

namespace dsk {
template <int H, int NCHUNK, int DV>
__device__ __forceinline__ void attn_merge_task_impl(const void *o_part_ptr, const void *lse_ptr, void *out_ptr, int head) {
  const float *o_part = static_cast<const float *>(o_part_ptr);
  const float *lse = static_cast<const float *>(lse_ptr);
  bf16 *out = static_cast<bf16 *>(out_ptr) + head * DV;
  float m = -__builtin_inff();
#pragma unroll
  for (int c = 0; c < NCHUNK; c++) m = fmaxf(m, lse[c * H + head]);
  for (int d = threadIdx.x; d < DV; d += WG_THREADS) {
    float o = 0.f, den = 0.f;
#pragma unroll
    for (int c = 0; c < NCHUNK; c++) {
      float wgt = exp2f(lse[c * H + head] - m);
      den += wgt;
      o = fmaf(o_part[(c * H + head) * DV + d], wgt, o);
    }
    out[d] = f2bf(o / den);
  }
}

struct alignas(16) f32x4 { float v[4]; };  // one 16-byte global load of fp32
__device__ __forceinline__ f32x4 load4f(const float *p) { return *reinterpret_cast<const f32x4 *>(p); }

// O_PROJ_RESIDUAL tile with the ATTN_MERGE prologue: thread t stages xs[8t, 8t+8) (head t*8/DV) with attn_merge's exact
// math (m, then per chunk wgt / den in chunk order, o = fma(o_part, wgt, o)), then the GEMV + residual epilogue.
// FIXED (switch DSK_MERGE_FIXED, F2):
//   0: perf/fold-full verbatim: o over c < nc = ceil(T/CHUNK), runtime-bounded scalar loads (72 per lane at T > 1024).
//      Chunks at or past T are not read for o: their lse is -inf, so attn_merge adds wgt = 0 to den and o_part * 0 to an o
//      that is never -0 (starts +0; x + y is -0 only if both are) -> identical bits.
//   1: o over c < NCHUNK, unrolled (attn_merge's loop): empty chunks have wgt = 0 and o_part = 0 (attn_chunk writes it), and
//      fma(0, 0, o) = o exactly -> identical bits.
//   2: as 1, with the thread's 8 o_part values of each chunk read as two 16-byte loads (2 x NCHUNK loads issued together).
template <int H, int NCHUNK, int CHUNK, int DV, int ROWS, int FIXED = DSK_MERGE_FIXED>
__device__ __forceinline__ void o_proj_merge_task_impl(const void *o_part_ptr, const void *lse_ptr, const void *w_ptr,
                                                       void *out_ptr, const void *res_ptr, int T) {
  constexpr int K = H * DV;
  static_assert(K == WG_THREADS * 8 && DV % 8 == 0, "one staged block of 8 per thread");
  extern __shared__ char smem[];
  const float *o_part = static_cast<const float *>(o_part_ptr), *lse = static_cast<const float *>(lse_ptr);
  const int i0 = threadIdx.x * 8, head = i0 / DV;
  float m = -__builtin_inff(), wgt[NCHUNK], den = 0.f;
#pragma unroll
  for (int c = 0; c < NCHUNK; c++) m = fmaxf(m, lse[c * H + head]);
#pragma unroll
  for (int c = 0; c < NCHUNK; c++) { wgt[c] = exp2f(lse[c * H + head] - m); den += wgt[c]; }
  bf16x8 v;
  if constexpr (FIXED == 0) {
    const int nc = (T + CHUNK - 1) / CHUNK;
#pragma unroll
    for (int j = 0; j < 8; j++) {
      float o = 0.f;
      // Scalar fp32 loads, nc per value (1 at T <= 128); 16 B loads if the op shows up in the probe.
      for (int c = 0; c < nc; c++) o = fmaf(o_part[(size_t)c * K + i0 + j], wgt[c], o);
      v.v[j] = f2bf(o / den);
    }
  } else if constexpr (FIXED == 1) {
#pragma unroll
    for (int j = 0; j < 8; j++) {
      float o = 0.f;
#pragma unroll
      for (int c = 0; c < NCHUNK; c++) o = fmaf(o_part[(size_t)c * K + i0 + j], wgt[c], o);
      v.v[j] = f2bf(o / den);
    }
  } else {
    f32x4 a[NCHUNK][2];
#pragma unroll
    for (int c = 0; c < NCHUNK; c++)
#pragma unroll
      for (int b = 0; b < 2; b++) a[c][b] = load4f(o_part + (size_t)c * K + i0 + b * 4);
#pragma unroll
    for (int j = 0; j < 8; j++) {
      float o = 0.f;
#pragma unroll
      for (int c = 0; c < NCHUNK; c++) o = fmaf(a[c][j / 4].v[j % 4], wgt[c], o);
      v.v[j] = f2bf(o / den);
    }
  }
  store8(reinterpret_cast<bf16 *>(smem) + i0, v);
  gemv_bf16_task_impl<ROWS, GEMV_RES, bf16, K, true>(nullptr, w_ptr, out_ptr, res_ptr);
}
}  // namespace dsk
