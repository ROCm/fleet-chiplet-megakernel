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

// Non-temporal store helpers for MALL cache optimization (NT=1 → MALL: no-allocate)
#ifndef NT_STORE_HIP_HELPERS_DEFINED
#define NT_STORE_HIP_HELPERS_DEFINED
__device__ __forceinline__ void nt_store_u64_hip(void* addr, uint64_t val) {
    __builtin_nontemporal_store(val, reinterpret_cast<uint64_t*>(addr));
}
__device__ __forceinline__ void nt_store_bf16_hip(__hip_bfloat16* addr, __hip_bfloat16 val) {
    *addr = val;  // scalar bf16 stores are rare (remainder path), use default
}
#endif

// CK-optimized SiLU*MUL for AMD MI300X
// Uses __builtin_amdgcn_rcpf (hardware reciprocal, V_RCP_F32, ~4 cycles)
// instead of fdiv (~16 cycles) for the sigmoid computation.
// Pattern from composable_kernel/ops/elementwise Silu fp32x2 specialization.

// Hardware-accelerated SiLU: x * rcp(1 + exp(-x))
__device__ __forceinline__ float fast_silu(float x) {
    return x * __builtin_amdgcn_rcpf(1.0f + __expf(-x));
}

template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int I_STRIDE,
          int O_STRIDE>
__device__ __forceinline__ void silu_mul_task_impl(void const *input_ptr,
                                                   void *output_ptr,
                                                   int num_active_tokens) {
#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  unsigned long long _t0 = __builtin_amdgcn_s_memrealtime();
#endif
  using bf16 = __hip_bfloat16;

  const bf16* __restrict__ d_input = static_cast<const bf16*>(input_ptr);
  const bf16* __restrict__ d_mul = static_cast<const bf16*>(input_ptr) + OUTPUT_SIZE;
  bf16* __restrict__ d_output = static_cast<bf16*>(output_ptr);

  // Specialized path for batch_size=1 (common case in decode)
  if constexpr (BATCH_SIZE == 1) {
    constexpr int VEC_SIZE = 8;
    constexpr int VEC_ITERS = OUTPUT_SIZE / VEC_SIZE;

    for (int vec_idx = threadIdx.x; vec_idx < VEC_ITERS; vec_idx += blockDim.x) {
      int offset = vec_idx * VEC_SIZE;

      uint64_t in_lo = *reinterpret_cast<const uint64_t*>(&d_input[offset]);
      uint64_t in_hi = *reinterpret_cast<const uint64_t*>(&d_input[offset + 4]);
      uint64_t mul_lo = *reinterpret_cast<const uint64_t*>(&d_mul[offset]);
      uint64_t mul_hi = *reinterpret_cast<const uint64_t*>(&d_mul[offset + 4]);

      bf16 out_arr[VEC_SIZE];

      const bf16* in_arr = reinterpret_cast<const bf16*>(&in_lo);
      const bf16* mul_arr = reinterpret_cast<const bf16*>(&mul_lo);
      #pragma unroll
      for (int v = 0; v < 4; v++) {
        float x = __bfloat162float(in_arr[v]);
        float m = __bfloat162float(mul_arr[v]);
        out_arr[v] = __float2bfloat16(fast_silu(x) * m);
      }

      in_arr = reinterpret_cast<const bf16*>(&in_hi);
      mul_arr = reinterpret_cast<const bf16*>(&mul_hi);
      #pragma unroll
      for (int v = 0; v < 4; v++) {
        float x = __bfloat162float(in_arr[v]);
        float m = __bfloat162float(mul_arr[v]);
        out_arr[4 + v] = __float2bfloat16(fast_silu(x) * m);
      }

      nt_store_u64_hip(&d_output[offset],
          *reinterpret_cast<uint64_t*>(&out_arr[0]));
      nt_store_u64_hip(&d_output[offset + 4],
          *reinterpret_cast<uint64_t*>(&out_arr[4]));
    }

    constexpr int REMAINDER_START = VEC_ITERS * VEC_SIZE;
    if constexpr (OUTPUT_SIZE % VEC_SIZE != 0) {
      for (int i = REMAINDER_START + threadIdx.x; i < OUTPUT_SIZE; i += blockDim.x) {
        float x = __bfloat162float(d_input[i]);
        float m = __bfloat162float(d_mul[i]);
        nt_store_bf16_hip(&d_output[i], __float2bfloat16(fast_silu(x) * m));
      }
    }
  } else {
    // General path for batch_size > 1
    constexpr int VEC_SIZE = 8;
    constexpr int TOTAL_ELEMS = BATCH_SIZE * OUTPUT_SIZE;
    constexpr int VEC_ITERS = TOTAL_ELEMS / VEC_SIZE;

    for (int vec_idx = threadIdx.x; vec_idx < VEC_ITERS; vec_idx += blockDim.x) {
      int elem_start = vec_idx * VEC_SIZE;
      int batch_idx = elem_start / OUTPUT_SIZE;
      int offset = elem_start % OUTPUT_SIZE;

      if (batch_idx >= num_active_tokens) continue;

      if (offset + VEC_SIZE > OUTPUT_SIZE) {
        #pragma unroll
        for (int v = 0; v < VEC_SIZE && offset + v < OUTPUT_SIZE; v++) {
          float x = __bfloat162float(d_input[batch_idx * I_STRIDE + offset + v]);
          float m = __bfloat162float(d_mul[batch_idx * I_STRIDE + offset + v]);
          nt_store_bf16_hip(&d_output[batch_idx * O_STRIDE + offset + v], __float2bfloat16(fast_silu(x) * m));
        }
        continue;
      }

      int base_in = batch_idx * I_STRIDE + offset;
      int base_out = batch_idx * O_STRIDE + offset;

      uint64_t in_lo = *reinterpret_cast<const uint64_t*>(&d_input[base_in]);
      uint64_t in_hi = *reinterpret_cast<const uint64_t*>(&d_input[base_in + 4]);
      uint64_t mul_lo = *reinterpret_cast<const uint64_t*>(&d_mul[base_in]);
      uint64_t mul_hi = *reinterpret_cast<const uint64_t*>(&d_mul[base_in + 4]);

      bf16 out_arr[VEC_SIZE];

      const bf16* in_arr = reinterpret_cast<const bf16*>(&in_lo);
      const bf16* mul_arr = reinterpret_cast<const bf16*>(&mul_lo);
      #pragma unroll
      for (int v = 0; v < 4; v++) {
        float x = __bfloat162float(in_arr[v]);
        float m = __bfloat162float(mul_arr[v]);
        out_arr[v] = __float2bfloat16(fast_silu(x) * m);
      }

      in_arr = reinterpret_cast<const bf16*>(&in_hi);
      mul_arr = reinterpret_cast<const bf16*>(&mul_hi);
      #pragma unroll
      for (int v = 0; v < 4; v++) {
        float x = __bfloat162float(in_arr[v]);
        float m = __bfloat162float(mul_arr[v]);
        out_arr[4 + v] = __float2bfloat16(fast_silu(x) * m);
      }

      nt_store_u64_hip(&d_output[base_out],
          *reinterpret_cast<uint64_t*>(&out_arr[0]));
      nt_store_u64_hip(&d_output[base_out + 4],
          *reinterpret_cast<uint64_t*>(&out_arr[4]));
    }

    constexpr int REMAINDER = TOTAL_ELEMS % VEC_SIZE;
    if constexpr (REMAINDER > 0) {
      int start_idx = VEC_ITERS * VEC_SIZE;
      for (int i = start_idx + threadIdx.x; i < TOTAL_ELEMS; i += blockDim.x) {
        int batch_idx = i / OUTPUT_SIZE;
        int offset = i % OUTPUT_SIZE;
        if (batch_idx < num_active_tokens) {
          float x = __bfloat162float(d_input[batch_idx * I_STRIDE + offset]);
          float m = __bfloat162float(d_mul[batch_idx * I_STRIDE + offset]);
          nt_store_bf16_hip(&d_output[batch_idx * O_STRIDE + offset], __float2bfloat16(fast_silu(x) * m));
        }
      }
    }
  }
#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  __syncthreads();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    unsigned long long _dur = (__builtin_amdgcn_s_memrealtime() - _t0) * 10;
    printf("[SILU_MUL] dur_us=%.1f\n", (double)_dur / 1000.0);
  }
#endif
}

} // namespace kernel
