/* Copyright 2025 Mirage Team
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
#include <hip/hip_bf16.h>

namespace kernel {

// SwigluOAI activation for GPT-OSS:
//   gate = clamp(gate, max=7.0)
//   up = clamp(up, min=-7.0, max=7.0)
//   glu = gate * sigmoid(1.702 * gate)
//   output = (up + 1) * glu
//
// Input layout is INTERLEAVED: gate=input[::2], up=input[1::2]
// So for each pair at positions 2i and 2i+1:
//   gate = input[2i], up = input[2i+1]

constexpr float SWIGLUOAI_ALPHA = 1.702f;
constexpr float SWIGLUOAI_LIMIT = 7.0f;

__device__ __forceinline__ float fast_swigluoai(float gate, float up) {
  gate = fminf(gate, SWIGLUOAI_LIMIT);
  up = fmaxf(fminf(up, SWIGLUOAI_LIMIT), -SWIGLUOAI_LIMIT);
  float glu =
      gate * __builtin_amdgcn_rcpf(1.0f + __expf(-SWIGLUOAI_ALPHA * gate));
  return (up + 1.0f) * glu;
}

// Non-temporal store helpers (reuse from silu_mul if already defined)
#ifndef NT_STORE_HIP_SWIGLUOAI_DEFINED
#define NT_STORE_HIP_SWIGLUOAI_DEFINED
__device__ __forceinline__ void nt_store_u64_swigluoai(void *addr,
                                                       uint64_t val) {
  *reinterpret_cast<uint64_t *>(addr) = val;
}
#endif

template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int I_STRIDE,
          int O_STRIDE>
__device__ __forceinline__ void swigluoai_task_impl(void const *input_ptr,
                                                    void *output_ptr,
                                                    int num_active_tokens) {
#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  unsigned long long _t0 = __builtin_amdgcn_s_memrealtime();
#endif
  using bf16 = __hip_bfloat16;

  // Input is interleaved: [gate0, up0, gate1, up1, ...] with size 2*OUTPUT_SIZE
  bf16 const *__restrict__ d_input = static_cast<bf16 const *>(input_ptr);
  bf16 *__restrict__ d_output = static_cast<bf16 *>(output_ptr);

  if constexpr (BATCH_SIZE == 1) {
    // Vectorized path for single-token decode
    constexpr int VEC_SIZE =
        4; // Process 4 output elements at a time (reads 8 input elements)

    for (int vec_idx = threadIdx.x; vec_idx < OUTPUT_SIZE / VEC_SIZE;
         vec_idx += blockDim.x) {
      int out_offset = vec_idx * VEC_SIZE;
      int in_offset = out_offset * 2; // interleaved layout

      bf16 out_arr[VEC_SIZE];

#pragma unroll
      for (int v = 0; v < VEC_SIZE; v++) {
        float gate = __bfloat162float(d_input[in_offset + 2 * v]);
        float up = __bfloat162float(d_input[in_offset + 2 * v + 1]);
        out_arr[v] = __float2bfloat16(fast_swigluoai(gate, up));
      }

      // Store 4 bf16 values (8 bytes) using NT store
      nt_store_u64_swigluoai(&d_output[out_offset],
                             *reinterpret_cast<uint64_t *>(&out_arr[0]));
    }

    // Handle remainder
    constexpr int VEC_ITERS = OUTPUT_SIZE / VEC_SIZE;
    constexpr int REMAINDER_START = VEC_ITERS * VEC_SIZE;
    if constexpr (OUTPUT_SIZE % VEC_SIZE != 0) {
      for (int i = REMAINDER_START + threadIdx.x; i < OUTPUT_SIZE;
           i += blockDim.x) {
        float gate = __bfloat162float(d_input[2 * i]);
        float up = __bfloat162float(d_input[2 * i + 1]);
        d_output[i] = __float2bfloat16(fast_swigluoai(gate, up));
      }
    }
  } else {
    // General path for batch_size > 1 (MoE: batch_size = max_tokens *
    // num_experts_per_tok)
    constexpr int VEC_SIZE = 4;
    constexpr int TOTAL_ELEMS = BATCH_SIZE * OUTPUT_SIZE;

    for (int elem_idx = threadIdx.x * VEC_SIZE; elem_idx < TOTAL_ELEMS;
         elem_idx += blockDim.x * VEC_SIZE) {
      int batch_idx = elem_idx / OUTPUT_SIZE;
      int offset = elem_idx % OUTPUT_SIZE;

      if (batch_idx >= num_active_tokens) {
        continue;
      }

      // Check we don't cross batch boundary
      if (offset + VEC_SIZE > OUTPUT_SIZE) {
        for (int v = 0; v < VEC_SIZE && offset + v < OUTPUT_SIZE; v++) {
          int in_base = batch_idx * I_STRIDE;
          float gate = __bfloat162float(d_input[in_base + 2 * (offset + v)]);
          float up = __bfloat162float(d_input[in_base + 2 * (offset + v) + 1]);
          d_output[batch_idx * O_STRIDE + offset + v] =
              __float2bfloat16(fast_swigluoai(gate, up));
        }
        continue;
      }

      int in_base = batch_idx * I_STRIDE;
      int out_base = batch_idx * O_STRIDE + offset;

      bf16 out_arr[VEC_SIZE];
#pragma unroll
      for (int v = 0; v < VEC_SIZE; v++) {
        float gate = __bfloat162float(d_input[in_base + 2 * (offset + v)]);
        float up = __bfloat162float(d_input[in_base + 2 * (offset + v) + 1]);
        out_arr[v] = __float2bfloat16(fast_swigluoai(gate, up));
      }

      nt_store_u64_swigluoai(&d_output[out_base],
                             *reinterpret_cast<uint64_t *>(&out_arr[0]));
    }

    // Handle remainder
    constexpr int REMAINDER = TOTAL_ELEMS % VEC_SIZE;
    if constexpr (REMAINDER > 0) {
      int start_idx = (TOTAL_ELEMS / VEC_SIZE) * VEC_SIZE;
      for (int i = start_idx + threadIdx.x; i < TOTAL_ELEMS; i += blockDim.x) {
        int batch_idx = i / OUTPUT_SIZE;
        int offset = i % OUTPUT_SIZE;
        if (batch_idx < num_active_tokens) {
          int in_base = batch_idx * I_STRIDE;
          float gate = __bfloat162float(d_input[in_base + 2 * offset]);
          float up = __bfloat162float(d_input[in_base + 2 * offset + 1]);
          d_output[batch_idx * O_STRIDE + offset] =
              __float2bfloat16(fast_swigluoai(gate, up));
        }
      }
    }
  }
#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  __syncthreads();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    unsigned long long _dur = (__builtin_amdgcn_s_memrealtime() - _t0) * 10;
    printf("[SWIGLUOAI] dur_us=%.1f\n", (double)_dur / 1000.0);
  }
#endif
}

} // namespace kernel
