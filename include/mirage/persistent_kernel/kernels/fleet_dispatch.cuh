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

// Fleet glue: one `ds_*` entry per task type (kernels/task_types.h), taking Fleet's (TaskDesc*, RuntimeConfig&)
// and doing ALL pointer / metadata arithmetic here, so the strings emitted by
// src/kernel/ds_task_register.cc are one-liners (`dsk::ds_attn_chunk<1152, 128>(task_desc, runtime_config);`)
// and this file, not a C++ string literal, is what kernels/dispatch_cpu.cc compiles on the laptop.
// Expert tasks are slot-indexed since 2026-09-14 (decision 10a): idx[slot] is read at run time here.
// TD/RC are template parameters (mirage::runtime::TaskDesc / RuntimeConfig on the GPU box, the mocks in
// dispatch_cpu.cc under the shim); the fields used: input_ptrs[], output_ptrs[], task_metadata.expert_offset,
// step[0]. Contract (bid = expert_offset: x = low 16 bits, y = high 16; p = step[0], T = p + 1).
#pragma once
#include "attn_chunk.cuh"
#include "attn_merge.cuh"
#include "gemv_bf16.cuh"
#include "moe_combine.cuh"
#include "moe_topk.cuh"

namespace dsk {
constexpr int DS_H = 16, DS_DN = 128, DS_DR = 64, DS_DC = 512, DS_DQK = DS_DN + DS_DR, DS_DV = 128;
// 192^-0.5 * (0.1 * mscale_all_dim * ln(factor) + 1)^2, DeepSeek-V2-Lite yarn (reference.softmax_scale)
constexpr float DS_SOFTMAX_SCALE = 0.114721388f;

__device__ __forceinline__ bf16 *at(void *p, size_t elems) { return static_cast<bf16 *>(p) + elems; }
__device__ __forceinline__ int bid_x(int eo) { return eo & 0xFFFF; }
__device__ __forceinline__ int bid_y(int eo) { return eo >> 16; }

// gemv_bf16 / gemv_bf16_res / gemv_bf16_silu / gemv_splitk: x whole, w tile pre-offset, out pre-offset; RES: in2.
template <int ROWS, int EPI, typename OutT, int K, class TD, class RC>
__device__ __forceinline__ void ds_gemv(TD const *t, RC const &) {
  gemv_bf16_task_impl<ROWS, EPI, OutT, K>(t->input_ptrs[0], t->input_ptrs[1], t->output_ptrs[0],
                                          EPI == GEMV_RES ? t->input_ptrs[2] : nullptr);
}

// Decision 11b (MLA_PREP fold, the measured-faster subset): kv_a_layernorm + k_pe RoPE in KV_B_PROJ, q RoPE in ATTN_CHUNK.
constexpr float DS_EPS = 1e-6f;

// KV_B_PROJ: [qkva, kv_a_layernorm, kv_b_proj tile, cos, sin] -> (K, V); a head's 256 kv_b rows are TPH = 256/ROWS tiles:
// tile n = bid.x, h = n/TPH, sub = n%TPH < TPH/2 -> K[h][p][sub*ROWS : +ROWS], else V[h][p][(sub-TPH/2)*ROWS : +ROWS];
// prologue = kv_a_layernorm; tile sub == 0 also writes K[h][p][128:192] = RoPE(k_pe). ROWS 64 = the measured graph (TPH 4),
// 32 / 16 = L4 (12e). Outputs (K, V) are declared as Fleet outputs so the chain assert sees them.
template <int TMAX, int ROWS, class TD, class RC>
__device__ __forceinline__ void ds_kv_b_proj(TD const *t, RC const &rc) {
  const int tph = (DS_DN + DS_DV) / ROWS, n = bid_x(t->task_metadata.expert_offset), h = n / tph, sub = n % tph;
  const int p = rc.step[0];
  void *out = sub >= tph / 2 ? at(t->output_ptrs[1], ((size_t)h * TMAX + p) * DS_DV + (sub - tph / 2) * ROWS)
                             : at(t->output_ptrs[0], ((size_t)h * TMAX + p) * DS_DQK + sub * ROWS);
  kv_b_proj_task_impl<DS_H, DS_DN, DS_DR, DS_DC, TMAX, ROWS>(t->input_ptrs[0], t->input_ptrs[1], t->input_ptrs[2],
                                                             t->input_ptrs[3], t->input_ptrs[4], out, t->output_ptrs[0], n,
                                                             p, p, DS_EPS);
}

// L6 (12e, perf/fold-full structure): [x whole, norm weight, w tile] -> out tile; xs = rmsnorm(x) is the NORM prologue.
template <int ROWS, int EPI, typename OutT, int K, class TD, class RC>
__device__ __forceinline__ void ds_gemv_norm(TD const *t, RC const &) {
  gemv_bf16_norm_task_impl<ROWS, EPI, OutT, K>(t->input_ptrs[0], t->input_ptrs[1], t->input_ptrs[2], t->output_ptrs[0],
                                               nullptr, DS_EPS);
}

// attn_chunk_192_128: [qkva, K, V, cos, sin] -> [part_o, lse]; head = bid.x, chunk = bid.y, T = p + 1; q RoPE prologue.
template <int TMAX, int CHUNK, class TD, class RC>
__device__ __forceinline__ void ds_attn_chunk(TD const *t, RC const &rc) {
  const int eo = t->task_metadata.expert_offset, p = rc.step[0];
  attn_chunk_task_impl<DS_H, TMAX, CHUNK, DS_DQK, DS_DV, true>(t->input_ptrs[0], t->input_ptrs[1], t->input_ptrs[2],
                                                               t->output_ptrs[0], t->output_ptrs[1], bid_x(eo),
                                                               bid_y(eo), p + 1, DS_SOFTMAX_SCALE, t->input_ptrs[3],
                                                               t->input_ptrs[4], p);
}

// attn_merge: [part_o, lse] -> attn_o; head = bid.x.
template <int NCHUNK, class TD, class RC>
__device__ __forceinline__ void ds_attn_merge(TD const *t, RC const &) {
  attn_merge_task_impl<DS_H, NCHUNK, DS_DV>(t->input_ptrs[0], t->input_ptrs[1], t->output_ptrs[0],
                                            bid_x(t->task_metadata.expert_offset));
}

// moe_gate_topk: [xn2, gate [E,K]] -> [logits fp32, w, idx] (gate GEMV + top-k in one task, decision 10c).
template <int E, int TOPK, int K, class TD, class RC>
__device__ __forceinline__ void ds_moe_gate_topk(TD const *t, RC const &) {
  moe_gate_topk_task_impl<E, TOPK, K>(t->input_ptrs[0], t->input_ptrs[1], t->output_ptrs[0], t->output_ptrs[1],
                                      t->output_ptrs[2]);
}

// L6 (12e): O_PROJ_RESIDUAL with the LSE merge prologue: [part_o, lse, o_proj tile, x tile] -> x_attn tile (perf/kernels F2).
template <int NCHUNK, int CHUNK, int ROWS, class TD, class RC>
__device__ __forceinline__ void ds_o_proj_merge(TD const *t, RC const &rc) {
  o_proj_merge_task_impl<DS_H, NCHUNK, CHUNK, DS_DV, ROWS>(t->input_ptrs[0], t->input_ptrs[1], t->input_ptrs[2],
                                                           t->output_ptrs[0], t->input_ptrs[3], rc.step[0] + 1);
}

// L6: [x_attn, post_attention_layernorm, gate [E, K], xn2 (written; declared an input, Fleet allows 3 outputs)] ->
// [logits fp32, w, idx]; norm2 is the prologue (perf/kernels' 8-argument moe_gate_topk_task_impl).
template <int E, int TOPK, int K, class TD, class RC>
__device__ __forceinline__ void ds_moe_gate_topk_norm(TD const *t, RC const &) {
  moe_gate_topk_task_impl<E, TOPK, K>(t->input_ptrs[0], t->input_ptrs[1], t->input_ptrs[2], t->output_ptrs[0],
                                      t->output_ptrs[1], t->output_ptrs[2], t->input_ptrs[3], DS_EPS);
}

// moe_topk (L5, 12e): [logits fp32 [1, E]] -> [w fp32 [TOPK], idx int32 [TOPK]]; the gate GEMV is its own op of tiles.
template <int E, int TOPK, class TD, class RC>
__device__ __forceinline__ void ds_moe_topk(TD const *t, RC const &) {
  moe_topk_task_impl<E, TOPK>(t->input_ptrs[0], t->output_ptrs[0], t->output_ptrs[1]);
}

// Slot-indexed expert tasks (decision 10a): slot = bid.x, tile = bid.y. slot < SLOTS: routed expert e = idx[slot],
// read at run time, weight pointer = w13/w2 + e * expert stride; slot == SLOTS: the shared expert (e = E, its own
// tensors w13_s/w2_s, 128-row w13 tiles). h and e_out stay expert-indexed, so moe_combine is unchanged.
// moe_expert_w13_silu: [idx, x, w13 tile (tile pre-offset), w13_s tile] -> h (whole). N_R = 2816 rows per routed
// expert (-> 1408 outputs), K = 2048.
template <int E, int N_R, int K, int ROWS_R, int ROWS_S, int SLOTS, class TD, class RC>
__device__ __forceinline__ void ds_expert_w13(TD const *t, RC const &) {
  const int eo = t->task_metadata.expert_offset, slot = bid_x(eo), tile = bid_y(eo);
  if (slot < SLOTS) {
    const int e = static_cast<const int *>(t->input_ptrs[0])[slot];
    gemv_bf16_task_impl<ROWS_R, GEMV_SILU, bf16, K>(t->input_ptrs[1], at(t->input_ptrs[2], (size_t)e * N_R * K),
                                                    at(t->output_ptrs[0], (size_t)e * (N_R / 2) + tile * (ROWS_R / 2)),
                                                    nullptr);
  } else {
    gemv_bf16_task_impl<ROWS_S, GEMV_SILU, bf16, K>(t->input_ptrs[1], t->input_ptrs[3],
                                                    at(t->output_ptrs[0], (size_t)E * (N_R / 2) + tile * (ROWS_S / 2)),
                                                    nullptr);
  }
}

// moe_expert_w2: [idx, h (whole), w2 tile, w2_s tile] -> e_out [E+1, N] (column tile pre-offset by the map; row e here).
template <int E, int N, int K_R, int K_S, int ROWS, int SLOTS, class TD, class RC>
__device__ __forceinline__ void ds_expert_w2(TD const *t, RC const &) {
  const int slot = bid_x(t->task_metadata.expert_offset);
  if (slot < SLOTS) {
    const int e = static_cast<const int *>(t->input_ptrs[0])[slot];
    gemv_bf16_task_impl<ROWS, GEMV_NONE, bf16, K_R>(at(t->input_ptrs[1], (size_t)e * K_R),
                                                    at(t->input_ptrs[2], (size_t)e * N * K_R),
                                                    at(t->output_ptrs[0], (size_t)e * N), nullptr);
  } else {
    gemv_bf16_task_impl<ROWS, GEMV_NONE, bf16, K_S>(at(t->input_ptrs[1], (size_t)E * K_R), t->input_ptrs[3],
                                                    at(t->output_ptrs[0], (size_t)E * N), nullptr);
  }
}

// moe_combine: [e_out, w, idx, residual] -> x.
template <int E, int TOPK, int N, class TD, class RC>
__device__ __forceinline__ void ds_moe_combine(TD const *t, RC const &) {
  moe_combine_task_impl<E, TOPK, N>(t->input_ptrs[0], t->input_ptrs[1], t->input_ptrs[2], t->input_ptrs[3],
                                    t->output_ptrs[0]);
}
}  // namespace dsk
