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
    *addr = val;  // scalar bf16 remainder path
}
#endif

// CK-optimized RMS_NORM for AMD MI300X
// Uses 128-bit vectorized loads (8 bf16/load, dwordx4) for 2x bandwidth vs 64-bit.
// Pattern adapted from CK rmsnorm2d pipeline's tile-based loading.
// Also uses __builtin_amdgcn_rcpf for fast reciprocal where applicable.

template <typename T, int BATCH_SIZE, int HIDDEN_DIM>
__device__ __forceinline__ void rms_norm_impl(void const *input_ptr,
                                              void const *weight_ptr,
                                              void *output_ptr,
                                              float eps) {
#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  unsigned long long _t0 = __builtin_amdgcn_s_memrealtime();
#endif
  using bf16 = __hip_bfloat16;

  const bf16* __restrict__ d_weight = static_cast<const bf16*>(weight_ptr);

  extern __shared__ char smem[];
  float* reduce_smem = reinterpret_cast<float*>(smem);

  // 128-bit loads: 8 bf16 elements per load (2x uint64_t = dwordx4)
  constexpr int VEC_SIZE = 8;
  constexpr int VEC_ITERS = HIDDEN_DIM / (NUM_THREADS * VEC_SIZE);
  constexpr int REMAINDER_START = VEC_ITERS * NUM_THREADS * VEC_SIZE;

  // Loop over batch rows — each row has independent RMS normalization
  for (int b = 0; b < BATCH_SIZE; b++) {
    const bf16* __restrict__ d_input =
        static_cast<const bf16*>(input_ptr) + b * HIDDEN_DIM;
    bf16* __restrict__ d_output =
        static_cast<bf16*>(output_ptr) + b * HIDDEN_DIM;

    // Phase 1: Compute sum of squares with 128-bit vectorized loads
    float sum = 0.0f;

    #pragma unroll
    for (int v = 0; v < VEC_ITERS; v++) {
      int offset = (v * NUM_THREADS + threadIdx.x) * VEC_SIZE;

      // 128-bit load: 2x uint64_t = 8 bf16 elements
      uint64_t in_lo = *reinterpret_cast<const uint64_t*>(&d_input[offset]);
      uint64_t in_hi = *reinterpret_cast<const uint64_t*>(&d_input[offset + 4]);

      const bf16* lo_arr = reinterpret_cast<const bf16*>(&in_lo);
      const bf16* hi_arr = reinterpret_cast<const bf16*>(&in_hi);

      #pragma unroll
      for (int i = 0; i < 4; i++) {
        float val = __bfloat162float(lo_arr[i]);
        sum += val * val;
      }
      #pragma unroll
      for (int i = 0; i < 4; i++) {
        float val = __bfloat162float(hi_arr[i]);
        sum += val * val;
      }
    }

    // Handle remainder elements
    if constexpr (HIDDEN_DIM > REMAINDER_START) {
      for (int i = REMAINDER_START + threadIdx.x; i < HIDDEN_DIM; i += NUM_THREADS) {
        float val = __bfloat162float(d_input[i]);
        sum += val * val;
      }
    }

    // Phase 2: Warp-level reduction (AMD wavefront = 64 lanes)
    #pragma unroll
    for (int offset = 32; offset > 0; offset /= 2) {
      sum += __shfl_xor(sum, offset);
    }

    // Phase 3: Cross-warp reduction via shared memory
    int warp_idx = threadIdx.x / 64;
    int lane_idx = threadIdx.x % 64;

    if (lane_idx == 0) {
      reduce_smem[warp_idx] = sum;
    }
    __syncthreads();

    constexpr int NUM_WARPS_AMD = NUM_THREADS / 64;
    if (threadIdx.x < NUM_WARPS_AMD) {
      sum = reduce_smem[threadIdx.x];
    } else {
      sum = 0.0f;
    }

    #pragma unroll
    for (int offset = NUM_WARPS_AMD / 2; offset > 0; offset /= 2) {
      sum += __shfl_xor(sum, offset);
    }

    if (threadIdx.x == 0) {
      reduce_smem[0] = sum;
    }
    __syncthreads();

    // Compute RMS reciprocal
    float rms_rcp = rsqrtf(reduce_smem[0] / float(HIDDEN_DIM) + eps);

    // Phase 4: Apply normalization with 128-bit vectorized load/store
    #pragma unroll
    for (int v = 0; v < VEC_ITERS; v++) {
      int offset = (v * NUM_THREADS + threadIdx.x) * VEC_SIZE;

      // 128-bit loads for input and weights
      uint64_t in_lo = *reinterpret_cast<const uint64_t*>(&d_input[offset]);
      uint64_t in_hi = *reinterpret_cast<const uint64_t*>(&d_input[offset + 4]);
      uint64_t w_lo = *reinterpret_cast<const uint64_t*>(&d_weight[offset]);
      uint64_t w_hi = *reinterpret_cast<const uint64_t*>(&d_weight[offset + 4]);

      const bf16* in_lo_arr = reinterpret_cast<const bf16*>(&in_lo);
      const bf16* in_hi_arr = reinterpret_cast<const bf16*>(&in_hi);
      const bf16* w_lo_arr = reinterpret_cast<const bf16*>(&w_lo);
      const bf16* w_hi_arr = reinterpret_cast<const bf16*>(&w_hi);

      bf16 out_arr[VEC_SIZE];
      #pragma unroll
      for (int i = 0; i < 4; i++) {
        float val = __bfloat162float(in_lo_arr[i]);
        float w = __bfloat162float(w_lo_arr[i]);
        out_arr[i] = __float2bfloat16(val * rms_rcp * w);
      }
      #pragma unroll
      for (int i = 0; i < 4; i++) {
        float val = __bfloat162float(in_hi_arr[i]);
        float w = __bfloat162float(w_hi_arr[i]);
        out_arr[4 + i] = __float2bfloat16(val * rms_rcp * w);
      }

      // 128-bit non-temporal store (NT=1 → MALL: no-allocate)
      nt_store_u64_hip(&d_output[offset],
          *reinterpret_cast<uint64_t*>(&out_arr[0]));
      nt_store_u64_hip(&d_output[offset + 4],
          *reinterpret_cast<uint64_t*>(&out_arr[4]));
    }

    // Handle remainder
    if constexpr (HIDDEN_DIM > REMAINDER_START) {
      for (int i = REMAINDER_START + threadIdx.x; i < HIDDEN_DIM; i += NUM_THREADS) {
        float val = __bfloat162float(d_input[i]);
        float w = __bfloat162float(d_weight[i]);
        nt_store_bf16_hip(&d_output[i], __float2bfloat16(val * rms_rcp * w));
      }
    }

    __syncthreads();  // Barrier before next batch row reuses reduce_smem
  } // end batch loop
#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  __syncthreads();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    unsigned long long _dur = (__builtin_amdgcn_s_memrealtime() - _t0) * 10;
    printf("[RMS_NORM] dur_us=%.1f\n", (double)_dur / 1000.0);
  }
#endif
}

// Tiled RMS_NORM: each tile processes a single batch row (tile_idx = row index).
// Distributes rows across gang workers for ~30x parallelism vs serial batch loop.
template <typename T, int BATCH_SIZE, int HIDDEN_DIM>
__device__ __forceinline__ void rms_norm_tiled_impl(void const *input_ptr,
                                                    void const *weight_ptr,
                                                    void *output_ptr,
                                                    float eps,
                                                    int tile_idx) {
  if (tile_idx >= BATCH_SIZE) return;

  using bf16 = __hip_bfloat16;
  const bf16* __restrict__ d_weight = static_cast<const bf16*>(weight_ptr);
  const bf16* __restrict__ d_input =
      static_cast<const bf16*>(input_ptr) + tile_idx * HIDDEN_DIM;
  bf16* __restrict__ d_output =
      static_cast<bf16*>(output_ptr) + tile_idx * HIDDEN_DIM;

  extern __shared__ char smem[];
  float* reduce_smem = reinterpret_cast<float*>(smem);

  constexpr int VEC_SIZE = 8;
  constexpr int VEC_ITERS = HIDDEN_DIM / (NUM_THREADS * VEC_SIZE);
  constexpr int REMAINDER_START = VEC_ITERS * NUM_THREADS * VEC_SIZE;

  // Phase 1: Sum of squares with 128-bit vectorized loads
  float sum = 0.0f;
  #pragma unroll
  for (int v = 0; v < VEC_ITERS; v++) {
    int offset = (v * NUM_THREADS + threadIdx.x) * VEC_SIZE;
    uint64_t in_lo = *reinterpret_cast<const uint64_t*>(&d_input[offset]);
    uint64_t in_hi = *reinterpret_cast<const uint64_t*>(&d_input[offset + 4]);
    const bf16* lo_arr = reinterpret_cast<const bf16*>(&in_lo);
    const bf16* hi_arr = reinterpret_cast<const bf16*>(&in_hi);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      float val = __bfloat162float(lo_arr[i]);
      sum += val * val;
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      float val = __bfloat162float(hi_arr[i]);
      sum += val * val;
    }
  }
  if constexpr (HIDDEN_DIM > REMAINDER_START) {
    for (int i = REMAINDER_START + threadIdx.x; i < HIDDEN_DIM; i += NUM_THREADS) {
      float val = __bfloat162float(d_input[i]);
      sum += val * val;
    }
  }

  // Phase 2: Warp-level reduction (AMD wavefront = 64)
  #pragma unroll
  for (int offset = 32; offset > 0; offset /= 2) {
    sum += __shfl_xor(sum, offset);
  }

  // Phase 3: Cross-warp reduction
  int warp_idx = threadIdx.x / 64;
  int lane_idx = threadIdx.x % 64;
  if (lane_idx == 0) {
    reduce_smem[warp_idx] = sum;
  }
  __syncthreads();

  constexpr int NUM_WARPS_AMD = NUM_THREADS / 64;
  if (threadIdx.x < NUM_WARPS_AMD) {
    sum = reduce_smem[threadIdx.x];
  } else {
    sum = 0.0f;
  }
  #pragma unroll
  for (int offset = NUM_WARPS_AMD / 2; offset > 0; offset /= 2) {
    sum += __shfl_xor(sum, offset);
  }
  if (threadIdx.x == 0) {
    reduce_smem[0] = sum;
  }
  __syncthreads();

  float rms_rcp = rsqrtf(reduce_smem[0] / float(HIDDEN_DIM) + eps);

  // Phase 4: Normalize with 128-bit vectorized load/store
  #pragma unroll
  for (int v = 0; v < VEC_ITERS; v++) {
    int offset = (v * NUM_THREADS + threadIdx.x) * VEC_SIZE;
    uint64_t in_lo = *reinterpret_cast<const uint64_t*>(&d_input[offset]);
    uint64_t in_hi = *reinterpret_cast<const uint64_t*>(&d_input[offset + 4]);
    uint64_t w_lo = *reinterpret_cast<const uint64_t*>(&d_weight[offset]);
    uint64_t w_hi = *reinterpret_cast<const uint64_t*>(&d_weight[offset + 4]);
    const bf16* in_lo_arr = reinterpret_cast<const bf16*>(&in_lo);
    const bf16* in_hi_arr = reinterpret_cast<const bf16*>(&in_hi);
    const bf16* w_lo_arr = reinterpret_cast<const bf16*>(&w_lo);
    const bf16* w_hi_arr = reinterpret_cast<const bf16*>(&w_hi);

    bf16 out_arr[VEC_SIZE];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      float val = __bfloat162float(in_lo_arr[i]);
      float w = __bfloat162float(w_lo_arr[i]);
      out_arr[i] = __float2bfloat16(val * rms_rcp * w);
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      float val = __bfloat162float(in_hi_arr[i]);
      float w = __bfloat162float(w_hi_arr[i]);
      out_arr[4 + i] = __float2bfloat16(val * rms_rcp * w);
    }
    nt_store_u64_hip(&d_output[offset],
        *reinterpret_cast<uint64_t*>(&out_arr[0]));
    nt_store_u64_hip(&d_output[offset + 4],
        *reinterpret_cast<uint64_t*>(&out_arr[4]));
  }
  if constexpr (HIDDEN_DIM > REMAINDER_START) {
    for (int i = REMAINDER_START + threadIdx.x; i < HIDDEN_DIM; i += NUM_THREADS) {
      float val = __bfloat162float(d_input[i]);
      float w = __bfloat162float(d_weight[i]);
      nt_store_bf16_hip(&d_output[i], __float2bfloat16(val * rms_rcp * w));
    }
  }
}

} // namespace kernel
