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
#include "gang_ksplit_linear_mi300.cuh"
#include "gang_linear_mi300.cuh"
#include "gang_splitk_linear_mi300.cuh"
#include "gemm_handtuned_mi300.cuh"
#include "linear_ck_mi300.cuh"
#include "tasks/common/common_header.cuh"

#define DEBUG 0

#if DEBUG
#define DCHECK(condition)                                                      \
  if ((condition) == 0) {                                                      \
    printf("Dcheck failed at %s:%d\n", __FILE__, __LINE__);                    \
  }
#else
#define DCHECK(condition)
#endif // DEBUG

namespace kernel {

using bfloat16 = type::bfloat16_t;

// AMD MFMA-based GEMM using CK Tile library for optimized memory access
template <typename T,
          int BATCH_SIZE,
          int REDUCTION_SIZE,
          bool FORCE_SMALL_TILE = false>
__device__ __forceinline__ void linear_kernel(void const *input_ptr,
                                              void const *weight_ptr,
                                              void const *residual_ptr,
                                              void *output_ptr,
                                              int num_active_tokens,
                                              bool residual_add,
                                              int output_size,
                                              int o_stride) {
#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  unsigned long long _t0 = __builtin_amdgcn_s_memrealtime();
#endif
  linear_kernel_ck<T, BATCH_SIZE, REDUCTION_SIZE, FORCE_SMALL_TILE>(
      input_ptr,
      weight_ptr,
      residual_ptr,
      output_ptr,
      num_active_tokens,
      residual_add,
      output_size,
      o_stride);
#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  __syncthreads();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    unsigned long long _dur = (__builtin_amdgcn_s_memrealtime() - _t0) * 10;
    printf("[NOGANG_LINEAR] osize=%d ostride=%d dur_us=%.1f\n",
           output_size,
           o_stride,
           (double)_dur / 1000.0);
  }
#endif
}

// Split-K linear: writes partial GEMM float32 results to workspace
template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int O_STRIDE = OUTPUT_SIZE,
          int PIPE_MAX = 3>
__device__ __forceinline__ void splitk_linear_kernel(void const *input_ptr,
                                                     void const *weight_ptr,
                                                     void *workspace_ptr,
                                                     int num_active_tokens) {
  linear_kernel_ck_splitk<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, O_STRIDE>(
      input_ptr, weight_ptr, workspace_ptr, num_active_tokens);
}

// Split-K reduce: sum K float32 workspace slices + residual → bf16 output
template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int K_SPLITS,
          int WS_STRIDE,
          int O_STRIDE = OUTPUT_SIZE>
__device__ __forceinline__ void splitk_reduce(void const *workspace_ptr,
                                              void const *residual_ptr,
                                              void *output_ptr,
                                              int num_active_tokens) {
  splitk_reduce_kernel<T,
                       BATCH_SIZE,
                       OUTPUT_SIZE,
                       K_SPLITS,
                       WS_STRIDE,
                       O_STRIDE>(
      workspace_ptr, residual_ptr, output_ptr, num_active_tokens);
}

// Split-K with float32 atomicAdd: single task, no separate reduce
template <typename T,
          int BATCH_SIZE,
          int NPerBlock,
          int REDUCTION_SIZE,
          int K_SPLITS>
__device__ __forceinline__ void
    splitk_linear_res_atomic(void const *input_ptr,
                             void const *weight_ptr,
                             void const *residual_ptr,
                             void *workspace_ptr,
                             void *output_ptr,
                             int *done_counter_ptr,
                             int num_active_tokens,
                             int ws_stride,
                             int o_stride) {
#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  unsigned long long _t0 = __builtin_amdgcn_s_memrealtime();
#endif
  splitk_linear_res_atomic_kernel<T,
                                  BATCH_SIZE,
                                  NPerBlock,
                                  REDUCTION_SIZE,
                                  K_SPLITS>(input_ptr,
                                            weight_ptr,
                                            residual_ptr,
                                            workspace_ptr,
                                            output_ptr,
                                            done_counter_ptr,
                                            num_active_tokens,
                                            ws_stride,
                                            o_stride);
#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  __syncthreads();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    unsigned long long _dur = (__builtin_amdgcn_s_memrealtime() - _t0) * 10;
    printf("[SPLITK_RES] K=%d ostride=%d dur_us=%.1f\n",
           REDUCTION_SIZE,
           o_stride,
           (double)_dur / 1000.0);
  }
#endif
}

} // namespace kernel
