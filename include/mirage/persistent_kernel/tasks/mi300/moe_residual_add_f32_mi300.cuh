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

// MoE residual addition from float32 workspace for MI300/MI350.
//
// Replaces moe_mul_sum_add when W2 already did routing weight multiplication
// and float32 atomicAdd into a workspace buffer.
//
// output[b, h] = bf16(workspace_f32[b, h] + residual_bf16[b, h])
// workspace_f32[b, h] = 0.0f  (zero for next iteration)

#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

namespace kernel {

template <int BATCH_SIZE, int OUTPUT_SIZE, int OUTPUT_STRIDE>
__device__ __forceinline__ void moe_residual_add_f32_mi300_impl(
    void *workspace_f32_ptr, void const *residual_ptr, void *output_ptr) {
  float *__restrict__ d_ws = static_cast<float *>(workspace_f32_ptr);
  unsigned short const *__restrict__ d_res =
      static_cast<unsigned short const *>(residual_ptr);
  unsigned short *__restrict__ d_out =
      static_cast<unsigned short *>(output_ptr);

  // Vectorized: float4 for f32 workspace load+zero, uint2 for bf16
  // residual/output
  constexpr int VEC = 4;
  constexpr int BLOCK_VEC = 256 * VEC; // assumes blockDim.x == 256
  float4 zero4 = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int row = 0; row < BATCH_SIZE; ++row) {
    float *ws_row = d_ws + row * OUTPUT_STRIDE;
    unsigned short const *res_row = d_res + row * OUTPUT_STRIDE;
    unsigned short *out_row = d_out + row * OUTPUT_STRIDE;

#pragma unroll
    for (int off = threadIdx.x * VEC; off < OUTPUT_SIZE; off += BLOCK_VEC) {
      // Vectorized load: 4 f32 from workspace (flat_load_dwordx4)
      float4 ws4;
      __builtin_memcpy(&ws4, ws_row + off, 16);

      // Vectorized load: 4 bf16 from residual as uint2 (flat_load_dwordx2)
      uint2 res_packed;
      __builtin_memcpy(&res_packed, res_row + off, 8);

      // Extract 4 bf16 → f32
      unsigned r0 = (res_packed.x & 0xFFFFu) << 16;
      unsigned r1 = res_packed.x & 0xFFFF0000u;
      unsigned r2 = (res_packed.y & 0xFFFFu) << 16;
      unsigned r3 = res_packed.y & 0xFFFF0000u;
      float rv0, rv1, rv2, rv3;
      __builtin_memcpy(&rv0, &r0, 4);
      __builtin_memcpy(&rv1, &r1, 4);
      __builtin_memcpy(&rv2, &r2, 4);
      __builtin_memcpy(&rv3, &r3, 4);

      // Add and convert to bf16, pack into uint2 for vectorized store
      float s0 = ws4.x + rv0, s1 = ws4.y + rv1;
      float s2 = ws4.z + rv2, s3 = ws4.w + rv3;

      // bf16 conversion with rounding
      auto f2bf16 = [](float f) -> unsigned short {
        union {
          float f;
          unsigned u;
        } v;
        v.f = f;
        unsigned rounding_bias = ((v.u >> 16) & 1) + 0x7FFF;
        return (unsigned short)((v.u + rounding_bias) >> 16);
      };

      uint2 out_packed;
      out_packed.x = (unsigned)f2bf16(s0) | ((unsigned)f2bf16(s1) << 16);
      out_packed.y = (unsigned)f2bf16(s2) | ((unsigned)f2bf16(s3) << 16);
      __builtin_memcpy(out_row + off, &out_packed, 8);

      // Zero workspace (flat_store_dwordx4)
      __builtin_memcpy(ws_row + off, &zero4, 16);
    }
  }
}

} // namespace kernel
