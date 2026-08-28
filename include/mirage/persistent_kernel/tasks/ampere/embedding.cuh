/* Copyright 2025 CMU
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
#pragma once
#include "tasks/common/common_header.cuh"

namespace kernel {

template <typename T, int OUT_DIM>
__device__ __forceinline__ void
    single_embedding_kernel(void const *__restrict__ input_ptr,
                            void const *__restrict__ embedding_ptr,
                            void *__restrict__ output_ptr,
                            int step,
                            long long *tokens) {
  // int64_t const *__restrict__ input_ids =
  //     static_cast<int64_t const *>(input_ptr);
  T const *__restrict__ embedding = static_cast<T const *>(embedding_ptr);
  T *__restrict__ output = static_cast<T *>(output_ptr);
  constexpr int BATCH_SIZE = 1;

  for (int i = threadIdx.x; i < BATCH_SIZE * OUT_DIM; i += blockDim.x) {
    // int idx = i / OUT_DIM;
    int off = i % OUT_DIM;
    // int64_t wordIdx = input_ids[idx];
    int64_t wordIdx = tokens[step];
    output[i] = embedding[wordIdx * OUT_DIM + off];
  }
}

template <typename T, int BATCH_SIZE, int CHUNK_SIZE, int OUTPUT_DIM_SIZE>
__device__ __forceinline__ void
    embedding_kernel(void const *__restrict__ input_ptr,
                     void const *__restrict__ embedding_ptr,
                     void *__restrict__ output_ptr) {
  // barrier for first 4 worker warps
  // TODO:(Jianan Ji) In vllm, type should be int32_t instead of int64_t.
  int64_t const *__restrict__ input_ids =
      static_cast<int64_t const *>(input_ptr);
  T const *__restrict__ embedding = static_cast<T const *>(embedding_ptr);
  T *__restrict__ output = static_cast<T *>(output_ptr);

#pragma unroll
  for (int batch_idx = 0; batch_idx < BATCH_SIZE; batch_idx++) {
    int64_t wordIdx = input_ids[batch_idx];
#ifdef EMBED_DEBUG
    if (threadIdx.x == 0) {
      printf("[EMBED] wordIdx=%lld output_ptr=%p\n",
             (long long)wordIdx,
             output_ptr);
    }
#endif
#ifdef MPK_EMBED_WIDE
    // Copy 8 bytes per lane instead of one element. The scalar arm below moves
    // sizeof(T) bytes per lane per trip, so a 2944-wide bf16 row takes 23 trips
    // at blockDim.x == 128, and blockDim.x is a runtime value so the trip count
    // is not known at compile time and the loop keeps its load->store waitcnt.
    // Four elements per lane cuts that to 6 trips over the same bytes.
    //
    // Guarded rather than unconditional because this kernel is shared by every
    // model in the tree: VALS_PER_LANE has to divide both the per-task chunk
    // and the row stride, and both base pointers have to be 8-byte aligned.
    // The alignment test is a runtime scalar branch (the pointers are uniform),
    // so an odd offset falls back to the scalar arm instead of faulting.
    constexpr int VALS_PER_LANE = 8 / sizeof(T);
    constexpr bool WIDE_SHAPE_OK = sizeof(T) * VALS_PER_LANE == 8 &&
                                   CHUNK_SIZE % VALS_PER_LANE == 0 &&
                                   OUTPUT_DIM_SIZE % VALS_PER_LANE == 0;
    bool const wide_aligned = ((reinterpret_cast<uintptr_t>(embedding) |
                                reinterpret_cast<uintptr_t>(output)) &
                               7u) == 0;
    if (WIDE_SHAPE_OK && wide_aligned) {
      struct __align__(8) wide_t {
        uint32_t lo, hi;
      };
      wide_t const *__restrict__ src =
          reinterpret_cast<wide_t const *>(embedding) +
          (wordIdx >= 0 ? wordIdx * (OUTPUT_DIM_SIZE / VALS_PER_LANE) : 0);
      wide_t *__restrict__ dst = reinterpret_cast<wide_t *>(output) +
                                 batch_idx * (OUTPUT_DIM_SIZE / VALS_PER_LANE);
      wide_t const zero{0u, 0u};
      for (int i = threadIdx.x; i < CHUNK_SIZE / VALS_PER_LANE;
           i += blockDim.x) {
        dst[i] = wordIdx >= 0 ? src[i] : zero;
      }
    } else
#endif
        if (wordIdx >= 0) {
#pragma unroll
      for (int i = threadIdx.x; i < CHUNK_SIZE; i += blockDim.x) {
        output[batch_idx * OUTPUT_DIM_SIZE + i] =
            embedding[wordIdx * OUTPUT_DIM_SIZE + i];
      }
    } else {
      // TODO: This might not be necessary
      for (int i = threadIdx.x; i < CHUNK_SIZE;
           i += blockDim.x) { // writing 0 to output
        output[batch_idx * OUTPUT_DIM_SIZE + i] = T(0.0f);
      }
    }
#ifdef EMBED_DEBUG
    __syncthreads();
    if (threadIdx.x == 0) {
      float v0 = static_cast<float>(output[0]);
      float v1 = static_cast<float>(output[1]);
      float v2 = static_cast<float>(output[2]);
      printf("[EMBED] output[0..2]: %f %f %f\n", v0, v1, v2);
    }
#endif
  }
}

} // namespace kernel
