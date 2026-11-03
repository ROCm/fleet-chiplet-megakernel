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

// Shared helpers for our task kernels. gfx942 target: bf16 is plain uint16 storage converted with integer
// ops (no gfx950-only cvt instructions); all math is fp32 fma. Compiles under HIP or the CPU shim.
#pragma once
#ifdef FLEET_CPU_SHIM
#include "cpu_shim.h"
#else
#include <hip/hip_runtime.h>
#endif
#include <cstdint>
#include <type_traits>

namespace dsk {
constexpr int WG_THREADS = 256;  // Fleet worker: 256 threads = 4 wave64
constexpr int WAVE_SIZE = 64;
constexpr int SMEM_BYTES = 57 * 1024;  // Fleet's MAX_DYNAMIC_SHARED_MEMORY_SIZE on AMD
constexpr float LOG2E = 1.4426950408889634f;
#ifdef FLEET_CPU_SHIM
alignas(16) inline char smem[SMEM_BYTES];  // the shim's stand-in for `extern __shared__ char smem[]`
#endif

using bf16 = uint16_t;
__device__ __forceinline__ float bf2f(bf16 u) { return __builtin_bit_cast(float, (uint32_t)u << 16); }
__device__ __forceinline__ bf16 f2bf(float f) {  // round-to-nearest-even, same as torch / __float2bfloat16
  uint32_t u = __builtin_bit_cast(uint32_t, f);
  if ((u & 0x7f800000u) == 0x7f800000u) return (bf16)(u >> 16);  // inf/nan: truncate
  u += 0x7fffu + ((u >> 16) & 1u);
  return (bf16)(u >> 16);
}
__device__ __forceinline__ float bfround(float f) { return bf2f(f2bf(f)); }

struct alignas(16) bf16x8 { bf16 v[8]; };  // one 16-byte global load (global_load_dwordx4)
__device__ __forceinline__ bf16x8 load8(const bf16 *p) { return *reinterpret_cast<const bf16x8 *>(p); }
__device__ __forceinline__ void store8(bf16 *p, bf16x8 v) { *reinterpret_cast<bf16x8 *>(p) = v; }

template <typename OutT> __device__ __forceinline__ OutT to_out(float f) {
  if constexpr (std::is_same<OutT, float>::value) return f; else return f2bf(f);
}

#ifdef FLEET_CPU_SHIM
__device__ __forceinline__ float wave_sum(float v) { return shim::wave_reduce(v, [](float a, float b) { return a + b; }); }
__device__ __forceinline__ float wave_max(float v) { return shim::wave_reduce(v, [](float a, float b) { return a > b ? a : b; }); }
__device__ __forceinline__ int wave_min(int v) { return shim::wave_reduce(v, [](int a, int b) { return a < b ? a : b; }); }
#else
__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
  for (int o = WAVE_SIZE / 2; o > 0; o >>= 1) v += __shfl_xor(v, o, WAVE_SIZE);
  return v;
}
__device__ __forceinline__ float wave_max(float v) {
#pragma unroll
  for (int o = WAVE_SIZE / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, WAVE_SIZE));
  return v;
}
__device__ __forceinline__ int wave_min(int v) {
#pragma unroll
  for (int o = WAVE_SIZE / 2; o > 0; o >>= 1) { int u = __shfl_xor(v, o, WAVE_SIZE); v = u < v ? u : v; }
  return v;
}
#endif

// Block-wide reductions; `red` = 4 floats of LDS. Every thread gets the result. Two barriers.
__device__ __forceinline__ float block_sum(float v, float *red) {
  v = wave_sum(v);
  if ((threadIdx.x & (WAVE_SIZE - 1)) == 0) red[threadIdx.x / WAVE_SIZE] = v;
  __syncthreads();
  v = (red[0] + red[1]) + (red[2] + red[3]);
  __syncthreads();
  return v;
}
__device__ __forceinline__ float block_max(float v, float *red) {
  v = wave_max(v);
  if ((threadIdx.x & (WAVE_SIZE - 1)) == 0) red[threadIdx.x / WAVE_SIZE] = v;
  __syncthreads();
  v = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
  __syncthreads();
  return v;
}
__device__ __forceinline__ float silu(float g) { return g / (1.0f + expf(-g)); }
}  // namespace dsk
