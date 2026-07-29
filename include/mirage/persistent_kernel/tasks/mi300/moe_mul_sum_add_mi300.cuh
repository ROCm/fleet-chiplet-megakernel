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

// MoE weighted sum + residual addition kernel for MI300/MI350.
// output[b, h] = residual[b, h] + sum_k(input[b, k, h] * weight[b, k])

#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

namespace kernel {

template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int NUM_TOPK,
          int OUTPUT_STRIDE,
          int INPUT_STRIDE = OUTPUT_STRIDE>
__device__ __forceinline__ void
    mul_sum_add_mi300_task_impl(void const *input_ptr,
                                void const *weight_ptr,
                                void const *residual_ptr,
                                void *output_ptr) {
  T const *__restrict__ d_input = static_cast<T const *>(input_ptr);
  T const *__restrict__ d_residual = static_cast<T const *>(residual_ptr);
  float const *__restrict__ d_weight = static_cast<float const *>(weight_ptr);
  T *__restrict__ d_output = static_cast<T *>(output_ptr);

#ifdef MPK_DEBUG_MOE_MUL_SUM
  // Debug: print residual and input for first tile (position 0)
  if (threadIdx.x == 0) {
    float r0 = static_cast<float>(d_residual[0]);
    float r1 = static_cast<float>(d_residual[1]);
    float in0_0 = static_cast<float>(d_input[0]);
    float in1_0 = static_cast<float>(d_input[INPUT_STRIDE]);
    float w0 = d_weight[0];
    float w1 = d_weight[1];
    printf("[MUL_SUM_ADD] res_ptr=%p out_ptr=%p res[0,1]=%f,%f in[0][0]=%f "
           "in[1][0]=%f w[0,1]=%f,%f\n",
           residual_ptr,
           output_ptr,
           r0,
           r1,
           in0_0,
           in1_0,
           w0,
           w1);
  }
#endif

  for (int row_idx = 0; row_idx < BATCH_SIZE; ++row_idx) {
    for (int i = threadIdx.x; i < OUTPUT_SIZE; i += blockDim.x) {
      T res_val = d_residual[row_idx * OUTPUT_STRIDE + i];
      float sum_val = static_cast<float>(res_val);
#pragma unroll
      for (int topk_idx = 0; topk_idx < NUM_TOPK; ++topk_idx) {
        T val = d_input[row_idx * INPUT_STRIDE * NUM_TOPK +
                        topk_idx * INPUT_STRIDE + i];
        float weight = d_weight[row_idx * NUM_TOPK + topk_idx];
        sum_val += static_cast<float>(val) * weight;
      }
      d_output[row_idx * OUTPUT_STRIDE + i] = static_cast<T>(sum_val);
    }
  }
}

} // namespace kernel
