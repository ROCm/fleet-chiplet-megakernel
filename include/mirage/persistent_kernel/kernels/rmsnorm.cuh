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

// HF DeepseekV2RMSNorm, mirrored op for op (fleet_ds/reference.py::rmsnorm):
//   xf = fp32(x); xf *= rsqrt(mean(xf^2) + eps); out = bf16(w * bf16(xf))
// N = 2048 (input/post_attention/model norm) or 512 (kv_a_layernorm). LDS: 4 floats.
// FIXED (switch DSK_NORM_FIXED, F3): N = 2048 / 512 run with a compile-time trip count (8 / 2 per
// thread), both loops unrolled. Same striping (thread t owns t, t+256, ...) and the same fma order, so the block_sum
// partition is unchanged: bit-identical. Any other N falls back to the runtime-bounded loops.
#pragma once
#include "common.cuh"
#include "switches.h"

namespace dsk {
template <int N>
__device__ __forceinline__ void rmsnorm_rows_fixed(const bf16 *x, const bf16 *w, bf16 *out, float eps, float *red) {
  constexpr int TRIPS = N / WG_THREADS;
  static_assert(N % WG_THREADS == 0, "fixed trips per thread");
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < TRIPS; j++) { float v = bf2f(x[threadIdx.x + j * WG_THREADS]); ss = fmaf(v, v, ss); }
  ss = block_sum(ss, red);
  float rcp = rsqrtf(ss / (float)N + eps);
#pragma unroll
  for (int j = 0; j < TRIPS; j++) {
    const int i = threadIdx.x + j * WG_THREADS;
    out[i] = f2bf(bf2f(w[i]) * bfround(bf2f(x[i]) * rcp));
  }
}

template <bool FIXED = DSK_NORM_FIXED>
__device__ __forceinline__ void rmsnorm_rows(const bf16 *x, const bf16 *w, bf16 *out, int N, float eps, float *red) {
  if constexpr (FIXED) {
    if (N == 2048) return rmsnorm_rows_fixed<2048>(x, w, out, eps, red);
    if (N == 512) return rmsnorm_rows_fixed<512>(x, w, out, eps, red);
  }
  float ss = 0.f;
  // Scalar loads (N/256 = 8 or 2 per thread); 16B vector loads if it matters.
  for (int i = threadIdx.x; i < N; i += WG_THREADS) { float v = bf2f(x[i]); ss = fmaf(v, v, ss); }
  ss = block_sum(ss, red);
  float rcp = rsqrtf(ss / (float)N + eps);
  for (int i = threadIdx.x; i < N; i += WG_THREADS) out[i] = f2bf(bf2f(w[i]) * bfround(bf2f(x[i]) * rcp));
}

__device__ __forceinline__ void rmsnorm_task_impl(const void *x_ptr, const void *w_ptr, void *out_ptr, int N, float eps) {
  extern __shared__ char smem[];
  rmsnorm_rows(static_cast<const bf16 *>(x_ptr), static_cast<const bf16 *>(w_ptr), static_cast<bf16 *>(out_ptr),
               N, eps, reinterpret_cast<float *>(smem));
}
}  // namespace dsk
