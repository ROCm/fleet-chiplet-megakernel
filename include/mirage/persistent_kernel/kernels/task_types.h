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

// Task-type ids for our DeepSeek-V2-Lite decode graph.
// Values start at 300 so they never collide with Fleet's TaskType (runtime_header.h: 1xx-2xx, SM100 end
// placeholder 298); names are global-scope so they cannot clash with mirage::runtime::TASK_EMBEDDING etc.
// "graph" = emitted by fleet_ds/mirage_graph.py; "fallback" = the graph emits Fleet's kernel for that op
// (embedding, rmsnorm, splitk_reduce_mi300, argmax_partial/reduce); ours exists for the CPU harness and
// as the drop-in if Fleet's does not build on gfx942 (design.md M1 fallback).
#pragma once
enum DsTaskType {
  TASK_GEMV_BF16 = 300,        // graph: gemv_bf16 (QKV, KV_B_PROJ, MOE_GATE, LM_HEAD)  gemv_bf16.cuh <ROWS, GEMV_NONE, bf16|float>
  TASK_GEMV_BF16_SILU = 301,   // graph: gemv_bf16_silu   gemv_bf16.cuh <64, GEMV_SILU, bf16> (gate/up rows interleaved, design.md 4.1)
  TASK_GEMV_BF16_RES = 302,    // graph: gemv_bf16_res    gemv_bf16.cuh <64, GEMV_RES, bf16>
  TASK_GEMV_SPLITK = 303,      // graph: gemv_splitk      gemv_bf16.cuh <64, GEMV_NONE, float> on a contiguous K-slice
  TASK_SPLITK_REDUCE = 304,    // fallback: splitk_reduce.cuh (graph emits Fleet's splitk_reduce_mi300, fp32 ws)
  TASK_MLA_PREP = 305,         // graph: mla_prep         mla_prep.cuh
  TASK_ATTN_CHUNK_192_128 = 306, // graph: attn_chunk_192_128  attn_chunk.cuh
  TASK_ATTN_MERGE = 307,       // graph: attn_merge       attn_merge.cuh (Fleet's merge_splitkv has the same partial layout)
  TASK_MOE_TOPK = 308,         // graph: moe_gate_topk    moe_topk.cuh (gate GEMV + top-k: logits[64] fp32, w[6], idx[6]; 10c)
  TASK_MOE_EXPERT_W13_SILU = 309, // graph: moe_expert_w13_silu  fleet_dispatch.cuh::ds_expert_w13 (slot-indexed, 10a) -> gemv_bf16.cuh
  TASK_MOE_EXPERT_W2 = 310,    // graph: moe_expert_w2    fleet_dispatch.cuh::ds_expert_w2 (slot-indexed, 10a) -> gemv_bf16.cuh
  TASK_MOE_COMBINE = 311,      // graph: moe_combine      moe_combine.cuh
  TASK_ARGMAX_PARTIAL = 312,   // fallback: argmax_partial.cuh (graph emits Fleet's argmax_partial; open question)
  TASK_ARGMAX_REDUCE = 313,    // fallback: argmax_reduce.cuh
  TASK_EMBED = 314,            // fallback: embed.cuh (graph emits Fleet's embedding)
  TASK_RMSNORM = 315,          // fallback: rmsnorm.cuh (graph emits Fleet's rmsnorm; also used inside mla_prep)
};
