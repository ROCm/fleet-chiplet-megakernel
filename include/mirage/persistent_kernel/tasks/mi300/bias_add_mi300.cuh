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

// Simple element-wise bias-add kernel for MI300/MI350.
// Adds a 1D bias vector (broadcast across batch dim) to a 2D tensor in-place.
// Used for attention biases (q_proj, k_proj, v_proj, o_proj) and router bias.

#pragma once
#include "tasks/common/common_header.cuh"
#include <hip/hip_bf16.h>

namespace kernel {

// bias_add: output[b][i] = input[b][i] + bias[i]
// Template parameters:
//   BATCH_SIZE  - number of tokens (rows)
//   SIZE        - vector size (columns), must be multiple of 8
//   I_STRIDE    - input stride (elements per row)
//   O_STRIDE    - output stride (elements per row)
template <int BATCH_SIZE, int SIZE, int I_STRIDE, int O_STRIDE>
__device__ __forceinline__ void bias_add_task_impl(void const *input_ptr,
                                                   void const *bias_ptr,
                                                   void *output_ptr) {
  using bf16 = __hip_bfloat16;

  bf16 const *__restrict__ d_input = static_cast<bf16 const *>(input_ptr);
  bf16 const *__restrict__ d_bias = static_cast<bf16 const *>(bias_ptr);
  bf16 *__restrict__ d_output = static_cast<bf16 *>(output_ptr);

  if constexpr (BATCH_SIZE == 1) {
    // Fast path for decode (BS=1)
    constexpr int VEC_SIZE = 8;
    constexpr int VEC_ITERS = SIZE / VEC_SIZE;

    for (int vec_idx = threadIdx.x; vec_idx < VEC_ITERS;
         vec_idx += blockDim.x) {
      int offset = vec_idx * VEC_SIZE;

      uint64_t in_lo = *reinterpret_cast<uint64_t const *>(&d_input[offset]);
      uint64_t in_hi =
          *reinterpret_cast<uint64_t const *>(&d_input[offset + 4]);
      uint64_t b_lo = *reinterpret_cast<uint64_t const *>(&d_bias[offset]);
      uint64_t b_hi = *reinterpret_cast<uint64_t const *>(&d_bias[offset + 4]);

      bf16 out_arr[VEC_SIZE];

      bf16 const *in_arr = reinterpret_cast<bf16 const *>(&in_lo);
      bf16 const *bias_arr = reinterpret_cast<bf16 const *>(&b_lo);
#pragma unroll
      for (int v = 0; v < 4; v++) {
        out_arr[v] = __float2bfloat16(__bfloat162float(in_arr[v]) +
                                      __bfloat162float(bias_arr[v]));
      }

      in_arr = reinterpret_cast<bf16 const *>(&in_hi);
      bias_arr = reinterpret_cast<bf16 const *>(&b_hi);
#pragma unroll
      for (int v = 0; v < 4; v++) {
        out_arr[4 + v] = __float2bfloat16(__bfloat162float(in_arr[v]) +
                                          __bfloat162float(bias_arr[v]));
      }

      *reinterpret_cast<uint64_t *>(&d_output[offset]) =
          *reinterpret_cast<uint64_t *>(&out_arr[0]);
      *reinterpret_cast<uint64_t *>(&d_output[offset + 4]) =
          *reinterpret_cast<uint64_t *>(&out_arr[4]);
    }

    // Remainder
    constexpr int REMAINDER_START = VEC_ITERS * VEC_SIZE;
    if constexpr (SIZE % VEC_SIZE != 0) {
      for (int i = REMAINDER_START + threadIdx.x; i < SIZE; i += blockDim.x) {
        d_output[i] = __float2bfloat16(__bfloat162float(d_input[i]) +
                                       __bfloat162float(d_bias[i]));
      }
    }
  } else {
    // General path for batch_size > 1
    constexpr int VEC_SIZE = 8;
    constexpr int TOTAL_ELEMS = BATCH_SIZE * SIZE;
    constexpr int VEC_ITERS = TOTAL_ELEMS / VEC_SIZE;

    for (int vec_idx = threadIdx.x; vec_idx < VEC_ITERS;
         vec_idx += blockDim.x) {
      int elem_start = vec_idx * VEC_SIZE;
      int batch_idx = elem_start / SIZE;
      int offset = elem_start % SIZE;

      if (offset + VEC_SIZE > SIZE) {
#pragma unroll
        for (int v = 0; v < VEC_SIZE && offset + v < SIZE; v++) {
          d_output[batch_idx * O_STRIDE + offset + v] = __float2bfloat16(
              __bfloat162float(d_input[batch_idx * I_STRIDE + offset + v]) +
              __bfloat162float(d_bias[offset + v]));
        }
        continue;
      }

      int base_in = batch_idx * I_STRIDE + offset;
      int base_out = batch_idx * O_STRIDE + offset;

      uint64_t in_lo = *reinterpret_cast<uint64_t const *>(&d_input[base_in]);
      uint64_t in_hi =
          *reinterpret_cast<uint64_t const *>(&d_input[base_in + 4]);
      uint64_t b_lo = *reinterpret_cast<uint64_t const *>(&d_bias[offset]);
      uint64_t b_hi = *reinterpret_cast<uint64_t const *>(&d_bias[offset + 4]);

      bf16 out_arr[VEC_SIZE];

      bf16 const *in_arr = reinterpret_cast<bf16 const *>(&in_lo);
      bf16 const *bias_arr = reinterpret_cast<bf16 const *>(&b_lo);
#pragma unroll
      for (int v = 0; v < 4; v++) {
        out_arr[v] = __float2bfloat16(__bfloat162float(in_arr[v]) +
                                      __bfloat162float(bias_arr[v]));
      }

      in_arr = reinterpret_cast<bf16 const *>(&in_hi);
      bias_arr = reinterpret_cast<bf16 const *>(&b_hi);
#pragma unroll
      for (int v = 0; v < 4; v++) {
        out_arr[4 + v] = __float2bfloat16(__bfloat162float(in_arr[v]) +
                                          __bfloat162float(bias_arr[v]));
      }

      *reinterpret_cast<uint64_t *>(&d_output[base_out]) =
          *reinterpret_cast<uint64_t *>(&out_arr[0]);
      *reinterpret_cast<uint64_t *>(&d_output[base_out + 4]) =
          *reinterpret_cast<uint64_t *>(&out_arr[4]);
    }

    constexpr int REMAINDER = TOTAL_ELEMS % VEC_SIZE;
    if constexpr (REMAINDER > 0) {
      int start_idx = VEC_ITERS * VEC_SIZE;
      for (int i = start_idx + threadIdx.x; i < TOTAL_ELEMS; i += blockDim.x) {
        int batch_idx = i / SIZE;
        int offset = i % SIZE;
        d_output[batch_idx * O_STRIDE + offset] = __float2bfloat16(
            __bfloat162float(d_input[batch_idx * I_STRIDE + offset]) +
            __bfloat162float(d_bias[offset]));
      }
    }
  }
}

} // namespace kernel
