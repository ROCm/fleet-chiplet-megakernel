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

// MoE routing in ONE task (decision 10c, 2026-09-14): gate GEMV [E,K] fp32 out -> softmax (fp32) -> top-TOPK
// (greedy, no renorm; routed_scaling_factor = 1.0). Outputs: logits[E] fp32 (the gate output, kept for the per-op
// gate), weights[TOPK] fp32 (descending), idx[TOPK] int32. The expert tasks read idx[slot] at run time (decision 10a),
// so there is no active[] array any more.
// Selection is wave-parallel: lane e of wave 0 keeps p[e] in a register; each round is one
// wave_max (best remaining score) + one wave_min (lowest lane holding it: same tie rule as the old serial loop and
// torch.topk on the reference). No scratch array, no global loads inside the loop.
#pragma once
#include "gemv_bf16.cuh"

namespace dsk {
// One wave: lane e passes logit e (E <= WAVE_SIZE; lanes >= E pass anything). Every lane of the wave must call.
template <int E, int TOPK>
__device__ __forceinline__ void moe_topk_select(float logit, void *w_ptr, void *idx_ptr) {
  static_assert(E <= WAVE_SIZE && TOPK <= E, "one lane per expert");
  float *w = static_cast<float *>(w_ptr);
  int *idx = static_cast<int *>(idx_ptr);
  const int lane = threadIdx.x & (WAVE_SIZE - 1);
  const bool live = lane < E;
  const float ninf = -__builtin_inff();
  const float m = wave_max(live ? logit : ninf);
  float p = live ? expf(logit - m) : 0.f;
  p /= wave_sum(p);
  bool sel = false;
  for (int k = 0; k < TOPK; k++) {
    const float best = wave_max(live && !sel ? p : ninf);
    const int e = wave_min(live && !sel && p == best ? lane : WAVE_SIZE);
    if (lane == e) { w[k] = p; idx[k] = e; sel = true; }
  }
}

// [x bf16 [K], gate bf16 [E,K]] -> [logits fp32 [E], w fp32 [TOPK], idx int32 [TOPK]]. LDS: x (2K B) + E floats.
template <int E, int TOPK, int K>
__device__ __forceinline__ void moe_gate_topk_task_impl(const void *x_ptr, const void *gate_ptr, void *logits_ptr,
                                                        void *w_ptr, void *idx_ptr) {
  extern __shared__ char smem[];
  float *acc = reinterpret_cast<float *>(smem + ((K * 2 + 15) & ~15));
  gemv_rows<E, K>(static_cast<const bf16 *>(x_ptr), static_cast<const bf16 *>(gate_ptr),
                  reinterpret_cast<bf16 *>(smem), acc);  // same accumulation as the former MOE_GATE task
  float *logits = static_cast<float *>(logits_ptr);
  for (int e = threadIdx.x; e < E; e += WG_THREADS) logits[e] = acc[e];
  if (threadIdx.x >= WAVE_SIZE) return;  // wave 0 selects (uniform per wave)
  moe_topk_select<E, TOPK>(threadIdx.x < E ? acc[threadIdx.x] : 0.f, w_ptr, idx_ptr);
}
// L5 (decision 12e): top-k as its own single task over the fp32 logits of the gate op (the gate GEMV tiled as its own op).
// Lane e of wave 0 passes logits[e], the value moe_gate_topk_task_impl passes as acc[e], so the selection is bit-identical.
template <int E, int TOPK>
__device__ __forceinline__ void moe_topk_task_impl(const void *logits_ptr, void *w_ptr, void *idx_ptr) {
  const float *logits = static_cast<const float *>(logits_ptr);
  if (threadIdx.x >= WAVE_SIZE) return;  // wave 0 selects (uniform per wave)
  moe_topk_select<E, TOPK>(threadIdx.x < E ? logits[threadIdx.x] : 0.f, w_ptr, idx_ptr);
}

// Decision 12e L6 (ported from perf/fold-full 9379248): [x_attn bf16 [K], post_attention_layernorm [K], gate bf16 [E,K]] ->
// [logits fp32 [E], w fp32 [TOPK], idx int32 [TOPK], xn2 bf16 [K]]. norm2 is this task's NORM prologue (rmsnorm_rows, the
// dsk norm; DSK_NORM_FIXED applies) and the task publishes xn2 for the expert tasks. LDS: xs (2K B) + E floats + 4.
template <int E, int TOPK, int K>
__device__ __forceinline__ void moe_gate_topk_task_impl(const void *x_ptr, const void *nw_ptr, const void *gate_ptr,
                                                        void *logits_ptr, void *w_ptr, void *idx_ptr, void *xn_ptr,
                                                        float eps) {
  extern __shared__ char smem[];
  bf16 *xs = reinterpret_cast<bf16 *>(smem);
  float *acc = reinterpret_cast<float *>(smem + ((K * 2 + 15) & ~15));
  gemv_norm_stage<E, K>(x_ptr, nw_ptr, eps);
  gemv_rows<E, K, true>(nullptr, static_cast<const bf16 *>(gate_ptr), xs, acc);  // same accumulation as the former MOE_GATE task
  float *logits = static_cast<float *>(logits_ptr);
  for (int e = threadIdx.x; e < E; e += WG_THREADS) logits[e] = acc[e];
  bf16 *xn = static_cast<bf16 *>(xn_ptr);
  for (int i = threadIdx.x * 8; i < K; i += WG_THREADS * 8) store8(xn + i, load8(xs + i));
  if (threadIdx.x >= WAVE_SIZE) return;  // wave 0 selects (uniform per wave)
  moe_topk_select<E, TOPK>(threadIdx.x < E ? acc[threadIdx.x] : 0.f, w_ptr, idx_ptr);
}
}  // namespace dsk
