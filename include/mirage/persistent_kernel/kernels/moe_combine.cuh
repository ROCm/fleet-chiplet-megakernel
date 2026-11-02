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

// x2[n] = bf16(res[n] + bf16(bf16(sum_{k<TOPK} w[k] * fp32(e_out[idx[k]][n])) + e_out[E][n]))
// HF's exact order (decision 8d, reference.py::moe): fp32 routed sum -> bf16, + shared in bf16, + residual.
// BLOCK (switch DSK_COMBINE_BLOCK, F1): thread t owns the contiguous elements [8t, 8t+8) instead of the
// striped t, t+256, ...; all 8 of its 16-byte loads (res, shared, TOPK routed rows) are issued together, then the same
// per-element expression, one store. Elements are independent, so the partition change is bit-identical.
#pragma once
#include "common.cuh"
#include "switches.h"

namespace dsk {
template <int E, int TOPK, int N, bool BLOCK = DSK_COMBINE_BLOCK>
__device__ __forceinline__ void moe_combine_task_impl(const void *e_out_ptr, const void *w_ptr, const void *idx_ptr,
                                                      const void *res_ptr, void *out_ptr) {
  const bf16 *e_out = static_cast<const bf16 *>(e_out_ptr);
  const float *w = static_cast<const float *>(w_ptr);
  const int *idx = static_cast<const int *>(idx_ptr);
  const bf16 *res = static_cast<const bf16 *>(res_ptr);
  bf16 *out = static_cast<bf16 *>(out_ptr);
  if constexpr (BLOCK) {
    static_assert(N == WG_THREADS * 8, "one 16-byte block per thread");
    const size_t i0 = threadIdx.x * 8;
    const bf16x8 r = load8(res + i0), sh = load8(e_out + (size_t)E * N + i0);
    bf16x8 eo[TOPK];
#pragma unroll
    for (int k = 0; k < TOPK; k++) eo[k] = load8(e_out + (size_t)idx[k] * N + i0);
    bf16x8 o;
#pragma unroll
    for (int j = 0; j < 8; j++) {
      float s = 0.f;
#pragma unroll
      for (int k = 0; k < TOPK; k++) s += w[k] * bf2f(eo[k].v[j]);
      float y = bfround(bfround(s) + bf2f(sh.v[j]));
      o.v[j] = f2bf(bf2f(r.v[j]) + y);
    }
    store8(out + i0, o);
  } else {
    for (int n = threadIdx.x; n < N; n += WG_THREADS) {
      float s = 0.f;
#pragma unroll
      for (int k = 0; k < TOPK; k++) s += w[k] * bf2f(e_out[(size_t)idx[k] * N + n]);
      float y = bfround(bfround(s) + bf2f(e_out[(size_t)E * N + n]));
      out[n] = f2bf(bf2f(res[n]) + y);
    }
  }
}
}  // namespace dsk
