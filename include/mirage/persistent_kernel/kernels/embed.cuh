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

// out[D] = table[tokens[step]][:], Fleet's single_embedding_kernel signature (tokens/step from RuntimeConfig).
#pragma once
#include "common.cuh"

namespace dsk {
template <int D>
__device__ __forceinline__ void embed_task_impl(const void *table_ptr, const long long *tokens, int step, void *out_ptr) {
  const bf16 *row = static_cast<const bf16 *>(table_ptr) + (size_t)tokens[step] * D;
  bf16 *out = static_cast<bf16 *>(out_ptr);
  for (int i = threadIdx.x * 8; i < D; i += WG_THREADS * 8) store8(out + i, load8(row + i));
}
}  // namespace dsk
