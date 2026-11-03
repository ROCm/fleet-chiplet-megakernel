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

// One head, one CHUNK-token slice of the HF-layout cache K[H][TMAX][192] / V[H][TMAX][128] (bf16):
//   s[t] = bf16(bf16(q.K[t]) * scale) * log2e   (HF: bf16 matmul, then bf16 scale, then fp32 softmax)
//   m = max s, p = exp2(s - m), l = sum p,  o_part[d] = (sum_t p[t] V[t][d]) / l,  lse = m + log2(l)
// Partial layout = Fleet's merge_splitkv: o_part[(chunk*H + head)*128 + d] fp32, lse[chunk*H + head] fp32
// (log2 domain, o already normalised by the chunk's l). Chunk start >= T: lse = -inf, o = 0.
// PV rewritten 2026-09-14 (design D10d): wave w owns d in [32w, 32w+32), lane l owns tokens
// l and l+64 and issues the 4 x 16 B V loads of a token together (8 independent loads per lane instead of a chain
// of 2-byte loads); 32 fp32 accumulators per lane; one LDS cross-lane reduce at the end. The fp32 summation order
// over t changes (per-lane partials, then a 64-way sum) — inside the 2 % gate, the bf16 rounding of P that HF
// does and we cannot (D9f) dominates. Scores loop unchanged.
// LDS: q fp32[192] + s/p fp32[CHUNK] + 4 floats, then red fp32[4 waves][32][64] = 32 KB.
// ROPE (decision 11b, MLA_PREP folded): q_ptr is qkva (its first H*DQK values are q, same layout as q_rope) and the task
// stages qs = [q_nope | RoPE(q_pe)] itself with mla_prep.cuh's rope_at (bit-identical to the MLA_PREP output).
// HOIST (switch DSK_ATTN_SCORE_HOIST, A1): the score loop reads the token's 24 K blocks (16-byte loads) first,
// unrolled, then runs the same 192-step fma chain (k ascending) -> bit-identical; the loads no longer interleave the chain.
#pragma once
#include "mla_prep.cuh"
#include "switches.h"

namespace dsk {
template <int H, int TMAX, int CHUNK, int DQK, int DV, bool ROPE = false, bool HOIST = DSK_ATTN_SCORE_HOIST>
__device__ __forceinline__ void attn_chunk_task_impl(const void *q_ptr, const void *k_ptr, const void *v_ptr,
                                                     void *o_part_ptr, void *lse_ptr, int head, int chunk, int T,
                                                     float scale, const void *cos_ptr = nullptr,
                                                     const void *sin_ptr = nullptr, int pos = 0) {
  constexpr int NWAVE = WG_THREADS / WAVE_SIZE, DW = DV / NWAVE, TPL = CHUNK / WAVE_SIZE;  // d per wave, tokens per lane
  static_assert(CHUNK <= WG_THREADS && CHUNK % WAVE_SIZE == 0 && DQK % 8 == 0 && DW % 8 == 0 && DV == NWAVE * DW,
                "one thread per token for the scores; PV: 4 waves over d, lanes over tokens");
  extern __shared__ char smem[];
  float *qs = reinterpret_cast<float *>(smem);
  float *p = qs + DQK;
  float *red = p + CHUNK;
  float *xr = red + 4;  // [NWAVE][DW][WAVE_SIZE] per-lane PV partials
  float *o_part = static_cast<float *>(o_part_ptr) + (chunk * H + head) * DV;
  float *lse = static_cast<float *>(lse_ptr) + chunk * H + head;
  const int t0 = chunk * CHUNK, n = T - t0 < CHUNK ? T - t0 : CHUNK;

  if (n <= 0) {  // empty chunk (uniform exit)
    for (int d = threadIdx.x; d < DV; d += WG_THREADS) o_part[d] = 0.f;
    if (threadIdx.x == 0) *lse = -__builtin_inff();
    return;
  }
  const bf16 *q = static_cast<const bf16 *>(q_ptr) + head * DQK;
  const bf16 *K = static_cast<const bf16 *>(k_ptr) + ((size_t)head * TMAX + t0) * DQK;
  const bf16 *V = static_cast<const bf16 *>(v_ptr) + ((size_t)head * TMAX + t0) * DV;
  if constexpr (ROPE) {
    constexpr int DR = 64, DN = DQK - DR;
    const bf16 *cs = static_cast<const bf16 *>(cos_ptr) + pos * DR, *sn = static_cast<const bf16 *>(sin_ptr) + pos * DR;
    for (int i = threadIdx.x; i < DQK; i += WG_THREADS) qs[i] = bf2f(i < DN ? q[i] : rope_at<DR>(q + DN, cs, sn, i - DN));
  } else {
    for (int i = threadIdx.x; i < DQK; i += WG_THREADS) qs[i] = bf2f(q[i]);
  }
  __syncthreads();

  // Thread t owns token t (uncoalesced 16B row loads, half the block idle); a wave-per-token
  // or MFMA layout is the GPU optimisation once the merge is validated.
  float s = -__builtin_inff();
  if (threadIdx.x < CHUNK && (int)threadIdx.x < n) {
    const bf16 *kr = K + (size_t)threadIdx.x * DQK;
    float dot = 0.f;
    if constexpr (HOIST) {
      bf16x8 a[DQK / 8];
#pragma unroll
      for (int b = 0; b < DQK / 8; b++) a[b] = load8(kr + b * 8);
#pragma unroll
      for (int b = 0; b < DQK / 8; b++)
#pragma unroll
        for (int j = 0; j < 8; j++) dot = fmaf(bf2f(a[b].v[j]), qs[b * 8 + j], dot);
    } else {
      for (int k = 0; k < DQK; k += 8) {
        bf16x8 a = load8(kr + k);
#pragma unroll
        for (int j = 0; j < 8; j++) dot = fmaf(bf2f(a.v[j]), qs[k + j], dot);
      }
    }
    s = bfround(bfround(dot) * scale) * LOG2E;
  }
  float m = block_max(s, red);
  float e = (threadIdx.x < CHUNK && (int)threadIdx.x < n) ? exp2f(s - m) : 0.f;
  if (threadIdx.x < CHUNK) p[threadIdx.x] = e;
  float l = block_sum(e, red);  // also publishes p[] (block_sum barriers)

  const int lane = threadIdx.x & (WAVE_SIZE - 1), wave = threadIdx.x / WAVE_SIZE;
  float acc[DW] = {};
#pragma unroll
  for (int i = 0; i < TPL; i++) {
    const int t = lane + i * WAVE_SIZE;
    if (t < n) {
      const bf16 *vr = V + (size_t)t * DV + wave * DW;
      bf16x8 a[DW / 8];
#pragma unroll
      for (int b = 0; b < DW / 8; b++) a[b] = load8(vr + b * 8);
      const float pt = p[t];
#pragma unroll
      for (int b = 0; b < DW / 8; b++)
#pragma unroll
        for (int j = 0; j < 8; j++) acc[b * 8 + j] = fmaf(pt, bf2f(a[b].v[j]), acc[b * 8 + j]);
    }
  }
#pragma unroll
  for (int j = 0; j < DW; j++) xr[(wave * DW + j) * WAVE_SIZE + lane] = acc[j];
  __syncthreads();
  for (int d = threadIdx.x; d < DV; d += WG_THREADS) {  // d -> (wave d/DW, j d%DW): sum the 64 lane partials
    const float *row = xr + d * WAVE_SIZE;
    float o = 0.f;
#pragma unroll
    for (int k = 0; k < WAVE_SIZE; k++) o += row[k];
    o_part[d] = o / l;
  }
  if (threadIdx.x == 0) *lse = m + log2f(l);
}
}  // namespace dsk
