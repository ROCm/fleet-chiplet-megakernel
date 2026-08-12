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
// and wrote each expert's contribution to its own slot of the f32 workspace.
//
// output[b, h] = bf16(sum_s workspace_f32[b, s, h] + residual_bf16[b, h])
//
// The slot sum runs in fixed compile-time order and touches only row `b`, so
// the result cannot depend on expert arrival order or on batch composition.
// See moe_ws_layout.cuh.
//
// This one consumer still zeroes, even though the producer now assigns rather
// than accumulates. It is the LAST layer's consumer, and the next thing to read
// the workspace is layer 0's QKV prologue on the following iteration -- which
// adds it to the embedding unconditionally, before any MoE has run. Leave it
// populated and every token double-counts layer 35's MoE output. The
// per-layer prologues need no zero pass; this boundary does.

#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include "mpk_atoms.cuh" // st_wt_zero128
#include "moe_ws_layout.cuh"

namespace kernel {

template <int BATCH_SIZE, int OUTPUT_SIZE, int OUTPUT_STRIDE>
__device__ __forceinline__ void moe_residual_add_f32_mi300_impl(
    void *workspace_f32_ptr, void const *residual_ptr, void *output_ptr) {
  float *__restrict__ d_ws = static_cast<float *>(workspace_f32_ptr);
  unsigned short const *__restrict__ d_res =
      static_cast<unsigned short const *>(residual_ptr);
  unsigned short *__restrict__ d_out =
      static_cast<unsigned short *>(output_ptr);

  // Vectorized: float4 for f32 workspace loads, uint2 for bf16 residual/output
  constexpr int VEC = 4;
  constexpr int BLOCK_VEC = 256 * VEC; // assumes blockDim.x == 256

  // ACQUIRE against the W2 producers, and it must be `sc1` (vL1 + L2).
  //
  // This task is the ONLY consumer of moe_workspace_f32 that is not the fused
  // layer's QKV prologue, and it is the only one that was missing this. The
  // prologue's acquire lives at the top of gang_full_layer_fused_mi300.cuh and
  // its comment states the contract: W2 writes the workspace with
  // `st_wt_f32x4` (sc0 sc1), which bypasses L2 and lands in HBM, and the
  // producing W2 tile runs on a different XCD than this consumer. MI300/MI350
  // L2 is not coherent across XCDs, so without this invalidate a line this XCD
  // already holds -- from *last* iteration's read of the same address, since
  // this task reads the identical buffer every iteration -- is served straight
  // out of L2 and returns the previous iteration's value.
  //
  // A plain `buffer_inv` would not do: it drops vL1 only, and vL1 is not where
  // the stale line lives. Which lines survive depends on inter-XCD L2 eviction
  // timing, so the corruption is nondeterministic run to run and looks like a
  // plausible float, never garbage -- the same signature the prologue's comment
  // describes.
  asm volatile("buffer_inv sc1" ::: "memory");

  for (int row = 0; row < BATCH_SIZE; ++row) {
    float *ws_row = d_ws + moe_ws_offset(row, 0, OUTPUT_STRIDE);
    unsigned short const *res_row = d_res + row * OUTPUT_STRIDE;
    unsigned short *out_row = d_out + row * OUTPUT_STRIDE;

#pragma unroll
    for (int off = threadIdx.x * VEC; off < OUTPUT_SIZE; off += BLOCK_VEC) {
      // Vectorized load: 4 f32 per topk slot (flat_load_dwordx4), summed in
      // fixed slot order.
      //
      // The zero stores are WRITE-THROUGH for the same reason W2's value
      // stores are (see moe_ws_layout.cuh): their consumer is the *next*
      // iteration's layer-0 QKV prologue, which runs on all 8 XCDs while this
      // task runs on one. A plain store stops in this XCD's L2 and the other
      // seven never see the zero -- they read layer 35's leftover MoE output
      // and add it to the embedding, which is exactly the double-count this
      // zero pass exists to prevent.
      float4 ws4;
      __builtin_memcpy(&ws4, ws_row + off, 16);
      st_wt_zero128(ws_row + off);
#pragma unroll
      for (int s = 1; s < MOE_WS_SLOTS; s++) {
        float4 slot4;
        __builtin_memcpy(&slot4, ws_row + s * OUTPUT_STRIDE + off, 16);
        st_wt_zero128(ws_row + s * OUTPUT_STRIDE + off);
        ws4.x += slot4.x;
        ws4.y += slot4.y;
        ws4.z += slot4.z;
        ws4.w += slot4.w;
      }

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
    }
  }
}

} // namespace kernel
