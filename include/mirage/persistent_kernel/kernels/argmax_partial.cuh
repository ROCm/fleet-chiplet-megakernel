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

// argmax over one CHUNK of fp32 logits -> val[0], idx[0] (int64, chunk-relative). Lowest index wins ties
// (torch.argmax). Fleet's argmax_mi300.cuh::argmax_partial_kernel<float,1,CHUNK,NUM> has the same
// signature and could be used instead, but its shuffle reduction keeps the lower LANE on ties, which is not
// the lower index. LDS + thread-0 serial fold over 256 candidates; fine at 1024 logits/task.
#pragma once
#include "common.cuh"

namespace dsk {
template <int CHUNK>
__device__ __forceinline__ void argmax_partial_task_impl(const void *logits_ptr, void *val_ptr, void *idx_ptr) {
  extern __shared__ char smem[];
  const float *x = static_cast<const float *>(logits_ptr);
  float *bv = reinterpret_cast<float *>(smem);
  int *bi = reinterpret_cast<int *>(bv + WG_THREADS);
  float v = -__builtin_inff(); int i = -1;
  for (int k = threadIdx.x; k < CHUNK; k += WG_THREADS)
    if (x[k] > v) { v = x[k]; i = k; }
  bv[threadIdx.x] = v; bi[threadIdx.x] = i;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int t = 1; t < WG_THREADS; t++)
      if (bv[t] > v || (bv[t] == v && bi[t] >= 0 && (i < 0 || bi[t] < i))) { v = bv[t]; i = bi[t]; }
    *static_cast<float *>(val_ptr) = v;
    *static_cast<long long *>(idx_ptr) = i;
  }
}
}  // namespace dsk
