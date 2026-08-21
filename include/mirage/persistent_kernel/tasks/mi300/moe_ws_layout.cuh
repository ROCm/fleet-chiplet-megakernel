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

// Layout of the MoE f32 output workspace, shared by its producer (the W2
// epilogue in gang_moe_fused_mxfp4_mi300.cuh) and its consumers (the ResAddF32
// prologues in gang_rmsnorm_linear_mxfp4_bias_mi300.cuh and
// moe_residual_add_f32_mi300.cuh).
//
// ── Why the workspace is per-(token, topk slot) rather than per-token ───────
//
// The W2 epilogue used to accumulate every routed expert's contribution into a
// single [token, hidden] f32 slab with `atomicAdd`. That is what broke row
// symmetry at batch > 1: identical prompts in different slots produced
// different continuations (5/6 runs at B=8), while B=1 was stable across ~25
// runs.
//
// The mechanism is float non-associativity, not a race. Experts E1 and E2 hit
// *different addresses* for token 0 than for token 1, so the memory system may
// serve E1-then-E2 for one row and E2-then-E1 for the other. Both sums are
// legal; they differ in the last ulp, and 36 layers of that is enough to cross
// an argmax boundary between two near-tied tokens. A single-row batch has no
// second row to disagree with, which is exactly why B=1 never showed it.
//
// vLLM does not have this property. In aiter every reduction stays *within* one
// row -- `max_num_partitions` derives from `max_seq_len` alone, never from
// `num_seqs`, and `tmp_output` is per-sequence -- so batch composition cannot
// perturb a row. Row symmetry there holds by construction. (Batch invariance,
// batched == solo, is broken in vLLM too; that is what VLLM_BATCH_INVARIANT
// exists for and it is off by default. Only row symmetry is our defect.)
//
// The fix restores the same "no shared accumulator" property. A token routes to
// exactly MOE_WS_SLOTS experts, `topk_slot` is already known per tile, and
// `wg_idx` partitions the hidden axis, so (token, topk_slot, hidden range) has
// exactly ONE writer. The atomics become plain stores and the consumer sums the
// slots in fixed slot order -- a row-private reduction with a fixed shape, so
// the result cannot depend on arrival order or on what else is in the batch.
//
// Two consequences worth stating because they are easy to get wrong:
//
//   * No per-layer zeroing, but the last layer still needs one. Coverage is
//     total and unique -- every (token, slot, hidden) is written exactly once
//     per layer -- so no *per-layer* consumer has anything stale to clear; the
//     old layout needed a zero pass precisely because it accumulated. Padding
//     expert slots (`is_padding_slot`) have no active token, write nothing, and
//     correspond to no (token, slot) pair, so they cannot leave a hole.
//     The exception is the iteration boundary: moe_residual_add_f32 consumes
//     layer 35, and the next reader is layer 0's QKV prologue on the *following*
//     iteration, which adds the workspace to the embedding before any MoE has
//     run. That consumer therefore zeroes as it reads -- see the comment there.
//
//   * The stores must be WRITE-THROUGH (`st_wt_f32x4`, sc0 sc1), not plain
//     stores. MI300/MI350 L2 is not coherent across XCDs -- see the comment on
//     threadfence_gpu in mpk_atoms.cuh -- and the consumer always runs on a
//     different XCD than the W2 tile that produced a given element. The
//     atomicAdd this replaced reached the coherent point implicitly, so the
//     requirement was invisible until it was removed. Getting this wrong does
//     not look like a numerical drift; it produces immediate token garbage at
//     every batch size, because the consumer reads whatever its own L2 holds.
//     The read side needs no change: the layer-boundary buffer_inv already
//     invalidates before the prologue runs, which is why the old write-through
//     zero pass was readable by ordinary loads.
//
//   * The atomics do not come back in another form. This is strictly less work
//     than before: N atomic read-modify-writes become N write-through stores
//     (one dwordx4 where there were four dwords), and the separate zero pass
//     disappears. The cost is that the consumer reads MOE_WS_SLOTS slabs
//     instead of one. That buffer is small (8 * 4 * 2944 * 4B = 376 KB) but the
//     consumer runs on every QKV worker, so the read is amplified -- measure,
//     do not assume.
//
// MOE_WS_SLOTS must equal the model's experts-per-token (NUM_TOPK). It is a
// constant rather than a template parameter so that both sides stay in a .cuh
// and are JIT-compiled at runtime -- changing it needs no rebuild of
// task_register.cc. The producer static_asserts it against its own NUM_TOPK.

#pragma once

namespace kernel {

// Experts per token. GPT-OSS 120B: num_experts_per_tok == 4.
constexpr int MOE_WS_SLOTS = 4;

// Element offset of (token `b`, slot `s`) within the workspace, where
// `hidden_stride` is the padded hidden size the buffer was allocated with.
// The host allocates [batch, MOE_WS_SLOTS * hidden_stride] f32, which keeps the
// tensor 2-D so the existing `num_dims == 2` asserts in persistent_kernel.py
// still hold.
__device__ __forceinline__ int
    moe_ws_offset(int b, int slot, int hidden_stride) {
  return (b * MOE_WS_SLOTS + slot) * hidden_stride;
}

} // namespace kernel
