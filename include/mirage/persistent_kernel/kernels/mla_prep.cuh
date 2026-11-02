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

// From qkva[H*192 + 512 + 64] (= cat(q_proj, kv_a_proj) output):
//   c_kvn[512]        = kv_a_layernorm(c_kv)                       (rmsnorm.cuh, eps)
//   q_rope[H][192]    = [q_nope | RoPE(q_pe)]  HF permuted+rotate-half layout, per reference.py::apply_rope
//   K[h][slot][128:192] = RoPE(k_pe) for every head h (HF cache layout, row stride 192, head stride TMAX*192)
// slot = cache row, pos = RoPE position (equal in real decode; separate like reference.decode_step for tests).
// cos/sin tables [pos][64] bf16 (cat(freqs,freqs)), precomputed on the host with HF's fp32 op order.
// RoPE mirrors HF's bf16 op order: out = bf16(bf16(x*cos) + bf16(rot*sin)).
// Decision 11b: the graph no longer runs MLA_PREP as a task. kv_b_proj_task_impl below folds kv_a_layernorm into every
// KV_B_PROJ tile's prologue and writes K[h][slot][128:192] from tile n = 4h (one writer per cache row); the q RoPE
// is ATTN_CHUNK's prologue (attn_chunk.cuh ROPE). This kernel stays as the CPU oracle for those folds.
#pragma once
#include "gemv_bf16.cuh"

namespace dsk {
// RoPE of one DR-vector x (original layout) at output index j: permuted p_j = x[2j] (j<DR/2) / x[2(j-DR/2)+1];
// rotate_half(p)_j = -p_{j+DR/2} = -x[2j+1] (j<DR/2) / p_{j-DR/2} = x[2(j-DR/2)]. cs/sn = the position's table row.
template <int DR>
__device__ __forceinline__ bf16 rope_at(const bf16 *x, const bf16 *cs, const bf16 *sn, int j) {
  float p = j < DR / 2 ? bf2f(x[2 * j]) : bf2f(x[2 * (j - DR / 2) + 1]);
  float r = j < DR / 2 ? -bf2f(x[2 * j + 1]) : bf2f(x[2 * (j - DR / 2)]);
  return f2bf(bfround(p * bf2f(cs[j])) + bfround(r * bf2f(sn[j])));
}
template <int H, int DN, int DR, int DC, int TMAX>
__device__ __forceinline__ void mla_prep_task_impl(const void *qkva_ptr, const void *kv_norm_w_ptr, const void *cos_ptr,
                                                   const void *sin_ptr, void *ckvn_ptr, void *q_rope_ptr,
                                                   void *kcache_ptr, int slot, int pos, float eps) {
  constexpr int DQ = DN + DR;
  extern __shared__ char smem[];
  const bf16 *qkva = static_cast<const bf16 *>(qkva_ptr);
  const bf16 *cs = static_cast<const bf16 *>(cos_ptr) + pos * DR;
  const bf16 *sn = static_cast<const bf16 *>(sin_ptr) + pos * DR;
  bf16 *q_rope = static_cast<bf16 *>(q_rope_ptr);
  bf16 *kc = static_cast<bf16 *>(kcache_ptr);

  rmsnorm_rows(qkva + H * DQ, static_cast<const bf16 *>(kv_norm_w_ptr), static_cast<bf16 *>(ckvn_ptr), DC, eps,
               reinterpret_cast<float *>(smem));

  auto rope = [&](const bf16 *x, int j) { return rope_at<DR>(x, cs, sn, j); };
  for (int i = threadIdx.x; i < H * DQ; i += WG_THREADS) {  // q_nope copy + q_pe rope
    int h = i / DQ, d = i % DQ;
    q_rope[i] = d < DN ? qkva[i] : rope(qkva + h * DQ + DN, d - DN);
  }
  for (int i = threadIdx.x; i < H * DR; i += WG_THREADS) {  // k_pe -> every head's cache row
    int h = i / DR, j = i % DR;
    kc[((size_t)h * TMAX + slot) * DQ + DN + j] = rope(qkva + H * DQ + DC, j);
  }
}

// KV_B_PROJ tile n with the MLA_PREP folds (decision 11b): xs = kv_a_layernorm(qkva[H*DQ : H*DQ+DC]) as the NORM prologue,
// then the ROWS-row GEMV into `out` (pre-offset by the caller into K[h][slot][0:DN] or V[h][slot]); a head's 2*DN kv_b rows
// (k_nope DN + v DN, DV == DN here) are TPH = 2*DN/ROWS tiles, and tile n % TPH == 0 also writes K[h][slot][DN:DQ] =
// RoPE(k_pe) for its head h = n / TPH. ROWS = 64 (4 tiles per head) is the measured graph; 32 / 16 are L4 (decision 12e).
template <int H, int DN, int DR, int DC, int TMAX, int ROWS = 64>
__device__ __forceinline__ void kv_b_proj_task_impl(const void *qkva_ptr, const void *kv_norm_w_ptr, const void *w_ptr,
                                                    const void *cos_ptr, const void *sin_ptr, void *out_ptr,
                                                    void *kcache_ptr, int n, int slot, int pos, float eps) {
  constexpr int DQ = DN + DR;
  const bf16 *qkva = static_cast<const bf16 *>(qkva_ptr);
  constexpr int TPH = 2 * DN / ROWS;
  gemv_bf16_norm_task_impl<ROWS, GEMV_NONE, bf16, DC>(qkva + H * DQ, kv_norm_w_ptr, w_ptr, out_ptr, nullptr, eps);
  if (n % TPH) return;  // uniform per task
  const bf16 *cs = static_cast<const bf16 *>(cos_ptr) + pos * DR, *sn = static_cast<const bf16 *>(sin_ptr) + pos * DR;
  bf16 *kc = static_cast<bf16 *>(kcache_ptr) + ((size_t)(n / TPH) * TMAX + slot) * DQ + DN;
  for (int j = threadIdx.x; j < DR; j += WG_THREADS) kc[j] = rope_at<DR>(qkva + H * DQ + DC, cs, sn, j);
}
}  // namespace dsk
