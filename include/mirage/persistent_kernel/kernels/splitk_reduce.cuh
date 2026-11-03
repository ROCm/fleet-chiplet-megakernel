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

// out[n] = bf16(res[n] + bf16(sum_s ws[s*N + n])) over S fp32 split-K partials (layer-0 down_proj, S=4).
#pragma once
#include "common.cuh"

namespace dsk {
template <int S>
__device__ __forceinline__ void splitk_reduce_task_impl(const void *ws_ptr, const void *res_ptr, void *out_ptr, int N) {
  const float *ws = static_cast<const float *>(ws_ptr);
  const bf16 *res = static_cast<const bf16 *>(res_ptr);
  bf16 *out = static_cast<bf16 *>(out_ptr);
  // Scalar loop; N=2048 is 8 per thread, vectorise if it ever shows in a profile.
  for (int n = threadIdx.x; n < N; n += WG_THREADS) {
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < S; k++) s += ws[k * N + n];
    out[n] = f2bf(bf2f(res[n]) + bfround(s));
  }
}
}  // namespace dsk
