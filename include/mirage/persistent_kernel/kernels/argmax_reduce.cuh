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

// out_token[0] = argmax over NUM partials (val[c], idx[c]) -> c*CHUNK + idx[c]; lowest index on ties.
// Written as the next token id (int64) for Fleet's prepare_next_batch. Serial on thread 0 (100 values).
#pragma once
#include "common.cuh"

namespace dsk {
template <int CHUNK, int NUM>
__device__ __forceinline__ void argmax_reduce_task_impl(const void *val_ptr, const void *idx_ptr, void *token_ptr) {
  if (threadIdx.x != 0) return;
  const float *v = static_cast<const float *>(val_ptr);
  const long long *ix = static_cast<const long long *>(idx_ptr);
  int best = 0;
  for (int c = 1; c < NUM; c++)
    if (v[c] > v[best]) best = c;  // strict: earlier chunk (lower index) wins ties
  *static_cast<long long *>(token_ptr) = (long long)best * CHUNK + ix[best];
}
}  // namespace dsk
