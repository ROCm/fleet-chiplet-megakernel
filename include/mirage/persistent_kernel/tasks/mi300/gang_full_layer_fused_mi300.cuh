/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

// Full-layer fused gang task for MI300/MI350.
//
// Combines task 214 (QKV+Attn) and task 215 (O-proj+TopK+MoE) into ONE
// gang task per transformer decoder layer, eliminating one inter-task
// event barrier and one scheduler dispatch per layer.
//
// 8-phase pipeline (single dispatch, all 240 workers stay alive):
//   Phase 1: QKV GEMM          — workers 0..(qkv_tiles_per_xcd-1) per XCD
//   Phase 2: QKV barrier       — epoch-based, all QKV workers wait
//   Phase 3: Parallel attention — workers 0..(NUM_KV_CHUNKS-1) each run one
//   chunk Phase 4: Chunk barrier      — last chunk worker runs merge inline
//   Phase 5: Merge + flush     — merge_splitkv_ck_fmha → bf16, write-through,
//   signal Phase 6: Cross-XCD barrier — all 30 workers poll attn_global Phase
//   7: O-proj + RMSNorm + Router + TopK Phase 8: MoE (W13+SwiGLU+W2)
//
// Counter buffer slot map (extends type 215's layout):
//   oproj_counters_ptr + 0..18*16-1  : type 215's counters
//   oproj_counters_ptr + 19*16       : attn_global_counter (cross-XCD sync)
//   oproj_counters_ptr + 20*16       : qkv_epoch[0..7] per-XCD epoch flags
//   oproj_counters_ptr + 28*16       : chunk_barrier[0..7] per-XCD chunk
//   arrival
//
// Stack-frame optimization: input/output pointers are NOT unpacked into
// local variables. Instead, input_ptrs[N] and output_ptrs[N] are accessed
// directly, saving ~272 bytes of stack frame per thread.

#pragma once
#include "tasks/mi300/gang_linear_mxfp4_res_bias_rmsnorm_topk_mi300.cuh"
#include "tasks/mi300/gang_moe_fused_mxfp4_mi300.cuh"
#include "tasks/mi300/gang_rmsnorm_linear_mxfp4_bias_mi300.cuh"
#include "tasks/mi300/paged_attention_ck_fmha_split_kv_mi300.cuh"

namespace kernel {

static constexpr int FULL_LAYER_ATTN_GLOBAL_COUNTER_SLOT = 19 * 16;
static constexpr int FULL_LAYER_QKV_EPOCH_SLOT = 20 * 16;
static constexpr int FULL_LAYER_ATTN_XCD_RELEASE_SLOT = 36 * 16;

// The chunk barrier is the one counter whose size depends on NUM_REQS: it needs
// one line per (XCD, request) so that the last chunk to arrive *for a given
// request* is the worker that merges *that* request. It therefore spans
// 8 * NUM_REQS lines = 128 * NUM_REQS ints.
//
// It used to live at 28 * 16, which is fine at NUM_REQS == 1 (8 lines, ending
// just below attn_xcd_release at 36 * 16) but grows straight through the
// fused-tail counters that gang_full_layer_with_lmhead_fused_mi300.cuh pins at
// the absolute literals 44*16 .. 47*16. At NUM_REQS >= 3 chunk arrivals would
// be written on top of the MoE-done / resadd-done / lmhead-done flags. That is
// latent today only because type 217 is dead behind an early `return`, and it
// would come back as silent corruption the moment FUSE_TAIL is re-enabled.
//
// Moving the barrier above every fixed slot instead of moving attn_xcd_release
// keeps every other counter at the address it has today, so nothing else in
// either file has to change. Callers must size the counter buffer to
// FULL_LAYER_CHUNK_BARRIER_SLOT + 128 * NUM_REQS (see demo.py counter_size).
static constexpr int FULL_LAYER_CHUNK_BARRIER_SLOT = 48 * 16;

// Layer-boundary global barrier: one line per XCD, immediately above the chunk
// barrier's 8 * NUM_REQS lines. See the barrier itself at the end of this file
// for why a *global* (all-XCD) rendezvous is required and a threadfence is not
// enough. Callers must size the counter buffer to
// FULL_LAYER_LAYER_BARRIER_SLOT + 272: 8 per-XCD arrival lines, one global
// arrival line, and 8 per-XCD release lines (see demo.py counter_size).
static constexpr int FULL_LAYER_LAYER_BARRIER_SLOT(int num_reqs) {
  return FULL_LAYER_CHUNK_BARRIER_SLOT + 128 * num_reqs;
}

// Per-XCD O-proj slice-ready flags for MPK_ROUTER_XCD_FOLD. 8 lines, 16 ints
// each, immediately above the layer-barrier region. demo.py counter_size
// adds 128 ints for this. Local-last of 23 O-proj tiles publishes its XCD
// here so the router can FMA that 368-col slice before the other XCDs land.
static constexpr int FULL_LAYER_OPROJ_XCD_READY_SLOT(int num_reqs) {
  return FULL_LAYER_LAYER_BARRIER_SLOT(num_reqs) + 272;
}

// MPK_XCD_LOCAL_BARRIER: drop `sc1` from the arrival atomics of the barriers
// whose counters never leave one XCD.
//
// Fleet routes every barrier arrival in this file through
// atom_add_release_gpu_s32, which is `flat_atomic_add ... sc0 sc1`. On MI350
// `sc1` is what forces the operation out of the XCD's private L2 and through
// the device-wide coherency point. For a counter that another XCD reads --
// attn_global, layer_global -- that is exactly right and must stay. For a
// counter indexed by the arriving worker's own `xcd_id` on both the write and
// the read side, it is a cross-die round trip taken to publish a value no
// other die will ever load.
//
// Three of this file's barriers are in the second category (each verified by
// grepping every access to the counter):
//
//   Phase 2  input_ptrs[7][xcd_id] and qkv_epoch[xcd_id * 16]
//   Phase 4  chunk_barrier[(xcd_id * NUM_REQS + attn_req) * 16]
//   Phase 9  layer_local[xcd_id * 16]   (layer_global stays sc0 sc1)
//
// The releases are untouched: those really are cross-XCD and keep their st_wt
// fan-out. This changes only how the arrival is counted, not who waits for
// what, so the ordering argument at each site is unaffected.
#ifdef MPK_XCD_LOCAL_BARRIER
#define MPK_XCD_LOCAL_ATOM_ADD(addr, val) atom_add_xcd_local_s32((addr), (val))
#else
#define MPK_XCD_LOCAL_ATOM_ADD(addr, val)                                      \
  atom_add_release_gpu_s32((addr), (val))
#endif

template <int QKV_BATCH_SIZE,
          int QKV_OUTPUT_PER_WG,
          int QKV_REDUCTION_SIZE,
          int ACTUAL_HIDDEN_DIM,
          int HEAD_DIM,
          int NUM_Q_PER_KV,
          int PAGE_SIZE,
          int MAX_SEQ_LEN,
          int NUM_KV_CHUNKS,
          int Q_WORKSPACE_STRIDE,
          int KV_CACHE_STRIDE,
          int NUM_KV_HEADS,
          int SLIDING_WINDOW,
          int HAS_SINKS,
          int OPROJ_OUTPUT_PER_WG,
          int OPROJ_REDUCTION_SIZE,
          int NUM_EXPERTS,
          int TOPK_K,
          int MOE_INTERMEDIATE_SIZE,
          int MOE_HIDDEN_SIZE,
          int MOE_W13_OUTPUT_PER_WG,
          int MOE_W2_OUTPUT_PER_WG,
          bool DECODE_ONLY = false,
          int NUM_REQS = 1>
__device__ __noinline__ void
    gang_full_layer_fused_kernel_mi300(void *const *input_ptrs,
                                       void *const *output_ptrs,
                                       void const *cos_ptr,
                                       void const *sin_ptr,
                                       int const *qo_indptr,
                                       int const *kv_indptr,
                                       int const *kv_indices,
                                       int const *kv_last_page_len,
                                       int num_active_tokens,
                                       int qkv_n_wgs_per_xcd,
                                       int kv_stride,
                                       int q_ws_stride,
                                       float attn_scale,
                                       int total_qkv_tiles_per_xcd,
                                       int oproj_n_wgs_per_xcd,
                                       int oproj_output_stride,
                                       int router_tile_n,
                                       int total_oproj_tiles,
                                       int total_topk_tiles,
                                       int oproj_tiles_per_xcd,
                                       int moe_total_tiles_per_xcd,
                                       int workers_per_xcd,
                                       int tile_idx,
                                       int task_layer_idx) {
  // (All phases enabled — buffer_inv before Phase 6 poll fixes L2 stale read)
  // input_ptrs layout:
  //  [0] workspace_f32    [1] residual          [2] norm_weight_pre
  //  [3] norm_scratch_pre [4] qkv_weight        [5] qkv_bias
  //  [6] attn_sinks       [7] qkv_barrier       [8] lse_acc
  //  [9] oproj_weight     [10] oproj_bias        [11] norm_weight_post
  //  [12] norm_scratch_post [13] router_weight   [14] router_bias
  //  [15] logits_scratch   [16] oproj_counters   [17] moe_gate_up_weight
  //  [18] moe_down_weight  [19] moe_w13_bias     [20] moe_w2_bias
  //  [21] moe_barrier      [22] moe_swiglu_out   [23] o_acc_f32
  //
  // output_ptrs layout:
  //  [0] x_output         [1] k_cache           [2] v_cache
  //  [3] q_workspace      [4] o_acc             [5] attn_proj_out
  //  [6] topk_weight      [7] routing_indices   [8] active_expert_ids
  //  [9] moe_routing_weight [10] moe_workspace_f32

#ifdef MPK_INTERLAYER_SPLIT
  unsigned long long _il_entry = __builtin_amdgcn_s_memrealtime();
#endif

  int xcd_id;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcd_id));

  int xcd_rank = tile_idx % workers_per_xcd;
  int tid = threadIdx.x;

  // ── Optional: rotate QKV/attention ownership off the O-proj ranks ─────────
  //
  // Phase 7 (O-proj + router + TopK) is owned by ranks
  // [0, oproj_topk_tiles_per_xcd); the ranks above it have no Phase 7 work and
  // therefore reach the MoE phase -- and finish its W2 tail -- ahead of
  // everyone else. Those are precisely the workers that have slack to spend on
  // the *next* layer's QKV weight prefetch and its GEMM, and today they are
  // the ones that never get either: Phase 1 and Phase 3 are both keyed on
  // `xcd_rank < K` with small K, so QKV and attention land on ranks 0..K-1 --
  // the same ranks that are still busy in Phase 7.
  //
  // Rotating by exactly the size of the no-Phase-7 block hands QKV and
  // attention to that block. It is a rotation, not a reassignment: `(r + d) %
  // workers_per_xcd` is a bijection on [0, workers_per_xcd), so every guard of
  // the form `rank < K` still selects exactly K workers and every tile index
  // derived from it is still covered exactly once. Only *which* physical
  // worker runs a given QKV tile or attention chunk changes.
  //
  // O-proj, TopK, MoE and every barrier arrival keep the physical rank. That
  // asymmetry is the point -- rotating those too would just relabel the same
  // overlap. It also keeps the Phase 9 barrier accounting untouched, which
  // counts arrivals per XCD and does not care who did what.
  //
  // The rotation distance is computed from the topology rather than hardcoded
  // for a 32-worker/23-tile layout, so it tracks fleet's 31 workers.
  //
  // ── DEFAULT OFF: it measures slower here, and the reason is structural ────
  //
  // 1.892 ms/token against 1.877 for the same build with the flag off (three
  // runs each, per-run medians, alternating, generated text bit-identical to
  // the reference on all six). The first numbers this produced were faster --
  // 1.854 to 1.871 -- but those runs were wrong: the rotation moved the layer
  // gate's waiter set off its arriver set, and the eight workers per XCD that
  // stopped waiting are the whole apparent gain. See the union computation at
  // MPK_LAYER_GATE_ARRIVES.
  //
  // Once the gate is honest the rotation cannot pay, because fleet already
  // spends the same slack twice over. In that topology ranks 23..31 are idle in the
  // W2 tail, so handing them QKV is free; fleet's are not idle, they are
  // *excused* -- MPK_W2_ONLY_ARRIVE keeps them out of the layer-gate arrival
  // and MPK_W2_CONSUMER_GATE keeps them out of its wait, precisely because
  // they have no second MoE tile. Rotating QKV onto them makes them waiters
  // again, which puts the arrival count back from 22 to 30 of 31 and costs
  // more than the prefetch overlap returns. The two optimizations are
  // competing for one resource; fleet's existing pair wins.
  //
  // Kept, behind the flag, because it is correct and because that trade could
  // invert at a geometry where the no-Phase-7 block and the no-second-tile
  // block stop coinciding.
  int const oproj_topk_ranks =
      oproj_tiles_per_xcd > router_tile_n ? oproj_tiles_per_xcd : router_tile_n;
#ifdef MPK_ROTATE_QKV_ATTN_RANKS
#ifdef MPK_QKV_ROT_OVERRIDE
  // Diagnostic only: pick the rotation distance by hand. Used to separate
  // "the rotation itself is unsound" from "the destination block of ranks is
  // special" -- a distance that lands QKV/attention on ranks that already run
  // Phase 7 exercises the same relabelling without moving the work off the
  // O-proj workers.
  int const qkv_rot = MPK_QKV_ROT_OVERRIDE % workers_per_xcd;
#else
  int const qkv_rot = workers_per_xcd > oproj_topk_ranks
                          ? workers_per_xcd - oproj_topk_ranks
                          : 0;
#endif
  int const qkv_attn_rank = (xcd_rank + qkv_rot) % workers_per_xcd;
#else
  int const qkv_attn_rank = xcd_rank;
#endif
#if defined(MPK_OPROJ_SWAP_ATTN) && defined(MPK_ROTATE_QKV_ATTN_RANKS)
#error "MPK_OPROJ_SWAP_ATTN keeps QKV/attn on the prefix; it is incompatible with rotating those ranks onto the idle block"
#endif
#if defined(MPK_ATTN_SPLIT_CHUNK) && defined(MPK_OPROJ_SWAP_ATTN)
#error "MPK_ATTN_SPLIT_CHUNK uses idle ranks 23-30 for the high half of each attention chunk; incompatible with SWAP_ATTN"
#endif
#if defined(MPK_ATTN_SPLIT_CHUNK) && defined(MPK_ROTATE_QKV_ATTN_RANKS)
#error "MPK_ATTN_SPLIT_CHUNK uses idle ranks 23-30; incompatible with ROTATE_QKV_ATTN_RANKS"
#endif

  // Attention work is (request, kv_head, kv_chunk). kv_head is bound to the
  // XCD (kv_head_idx = xcd_id below), so the request and chunk axes both have
  // to fit in xcd_rank -- the 30 workers on this XCD. The host keeps
  // NUM_KV_CHUNKS = workers_per_xcd / NUM_REQS so this never exceeds 30.
  //
  // NUM_REQS is the *compile-time* MPK_MAX_NUM_BATCHED_REQUESTS, never the live
  // request count. prepare_next_batch pads unused slots so
  // qo_indptr[i] == qo_indptr[i+1], and every attention/merge callee
  // early-returns on that *inside itself* -- so a padded slot still arrives at
  // the chunk barrier below. Keeping the participant set fixed is what makes
  // every modulus here independent of how many requests happen to be live.
  constexpr int ATTN_PARTICIPANTS = NUM_REQS * NUM_KV_CHUNKS;
  int const attn_req = qkv_attn_rank / NUM_KV_CHUNKS;
  int const attn_chunk = qkv_attn_rank % NUM_KV_CHUNKS;
#ifdef MPK_ATTN_SPLIT_CHUNK
  static_assert(NUM_REQS == 1,
                "MPK_ATTN_SPLIT_CHUNK maps one idle rank per chunk");
  int const split_idle_n = workers_per_xcd > oproj_topk_ranks
                               ? workers_per_xcd - oproj_topk_ranks
                               : 0;
  int const split_hchunk = xcd_rank - oproj_topk_ranks;
  bool const is_split_helper = (split_idle_n == ATTN_PARTICIPANTS) &&
                               (split_hchunk >= 0) &&
                               (split_hchunk < NUM_KV_CHUNKS);
#endif

  // Optional: swap O-proj tiles [0, ATTN) onto the idle ranks
  // [oproj_topk_ranks, W) so attention workers skip Phase 7.
  //
  // Dual of MPK_ROTATE_QKV_ATTN_RANKS. That flag moves QKV/attn onto the
  // idle ranks and lost because those ranks are the W2_ONLY_ARRIVE
  // excusees -- putting QKV on them made them layer-gate waiters again.
  // This keeps QKV/attn on the prefix (waiter set unchanged) and moves the
  // eight attention-owned O-proj tiles onto ranks that skip QKV+attn, so
  // they can issue the Phase 6 weight DMA during that work and GEMM after
  // attn_release with hot LDS. Hier-barrier arrival stays exactly
  // oproj_topk_ranks: the workers that enter Phase 7. Ranks 0..ATTN-1 wait
  // at the skip-gate the way 23..30 do today.
  //
  // Shipped geometry: W=31, oproj=23, ATTN=8. Ranks 8-22 keep tiles 8-22;
  // ranks 23-30 take tiles 0-7; ranks 0-7 skip. If idle_n != ATTN the swap
  // is a no-op rather than a mismatched arrival count.
#ifdef MPK_OPROJ_SWAP_ATTN
  int const oproj_idle_n = workers_per_xcd > oproj_topk_ranks
                                ? workers_per_xcd - oproj_topk_ranks
                                : 0;
  bool const oproj_do_swap =
      (oproj_idle_n == ATTN_PARTICIPANTS) && (oproj_idle_n > 0);
  bool const does_oproj =
      oproj_do_swap ? (xcd_rank >= ATTN_PARTICIPANTS)
                    : (xcd_rank < oproj_topk_ranks);
  int const oproj_local =
      !does_oproj ? 0
      : (oproj_do_swap && xcd_rank >= oproj_topk_ranks)
            ? (xcd_rank - oproj_topk_ranks)
            : xcd_rank;
#else
  bool const does_oproj = xcd_rank < oproj_topk_ranks;
  int const oproj_local = xcd_rank;
#endif

#ifdef MPK_QKV_KSPLIT
  // 2-way K-split. Ranks [0, T) do K-lo; ranks [W-T, W) do K-hi of the same
  // tile. Prefix mapping (K-hi on 10-19) measured +141 / +92 / +103 µs:
  // those ranks are the ones that overlap O-proj DMA with QKV+attn. High
  // ranks skip O-proj (23-30) or sit at its tail (21-22), so they can take
  // the extra GEMM without stealing that overlap. T=10, W=31 → K-hi on
  // 21-30. QKV epoch is no longer a prefix of 20; it is the two ranges.
  int const qkv_k1_base = workers_per_xcd - total_qkv_tiles_per_xcd;
  bool const qkv_does_k0 = qkv_attn_rank < total_qkv_tiles_per_xcd;
  bool const qkv_does_k1 = qkv_attn_rank >= qkv_k1_base;
  bool const qkv_does_qkv = qkv_does_k0 || qkv_does_k1;
  int const qkv_work_slots = total_qkv_tiles_per_xcd * 2;
  constexpr int QKV_MFMA_ITERS = QKV_REDUCTION_SIZE / 128;
  constexpr int QKV_K_LO = (QKV_MFMA_ITERS + 1) / 2;
#else
  bool const qkv_does_qkv = qkv_attn_rank < total_qkv_tiles_per_xcd;
  int const qkv_work_slots = total_qkv_tiles_per_xcd;
#endif

  // Layer-boundary ACQUIRE for moe_workspace_f32. Plain `buffer_inv` (vL1
  // only) -- NOT `buffer_inv sc1`.
  //
  // ── Why vL1-only is sufficient here, and how that was established ────────
  //
  // This site and four others (see the back-references below) were `sc1`
  // (vL1 + L2). The reasoning was: W2 writes the workspace with `st_wt_f32x4`
  // (sc0 sc1), bypassing L2 into HBM; the producing tile is on a different
  // XCD; MI300/MI350 L2 is not coherent across XCDs; so a line this XCD still
  // holds from last layer's read of the same address would be served stale
  // out of L2.
  //
  // Every step of that is true about the *hardware*. What it omits is the
  // Phase 9 layer barrier at the end of this function, which did not exist
  // when these were written. Phase 9 is a global all-XCD rendezvous whose
  // release path drains stores (`s_waitcnt vmcnt(0)`) and fans out
  // write-through flags. No worker can reach this line for layer N+1 until
  // every worker has passed Phase 9 for layer N, by which point the producers'
  // write-through stores have retired to HBM. The stale-L2 window the `sc1`
  // was closing is a window Phase 9 already closes.
  //
  // Measured, not argued from the memory model. Ablated at HEAD across five
  // configurations, every run compared bitwise against an unablated reference:
  //
  //   B=8 x 5 runs, B=1 x 3 runs, B=16 x 3 runs, 36 layers, 200 positions
  //     -> all KV caches and all output tokens bitwise identical, and
  //        bitwise identical to the `sc1` build. Same arithmetic, not merely
  //        a different-but-stable answer.
  //   `--max-layers 2` x 4 runs at B=8 -> also identical. This one is the
  //        load-bearing check: a 36-layer run streams ~1.7 GB of MoE weights
  //        through a ~4 MB L2 per layer, so a stale line might simply be
  //        evicted by capacity pressure rather than by any guarantee, which
  //        would make the result accidental. At 2 layers that pressure is
  //        largely gone and a stale line can survive. It still matched.
  //
  // Cost recovered: 3.455 -> 2.527 ms/iter at B=1, seq 512 (-27%). The five
  // sites are not individually free (0.03-0.27 ms each) but they overlap; the
  // combined figure is the one that matters.
  //
  // ── What would invalidate this ───────────────────────────────────────────
  //
  // The dependency is on Phase 9. If the layer barrier is weakened, moved
  // above a workspace consumer, or removed, these five sites must go back to
  // `sc1` -- or the ablation must be re-run. Phase 9 is not optional for its
  // own sake either: removing it fails this same gate immediately and loudly
  // (all 72 KV tensors differ, output tokens differ, every run).
  //
  // Note this is an ablation result, not a proof. It says removal is harmless
  // across every gate above; it does not derive safety from the memory model.
  //
  // MPK_W2_CONSUMER_GATE is exactly the "weakened" case this warning names, so
  // under that flag the acquire moves to the consumer gate in Phase 1 and goes
  // back to `sc1`. See the gate site for the full argument.
  //
  // ── The bs>1 arm: one system-scope acquire for the whole layer ────────────
  //
  // Everything above is a bs=1 argument. It leans on Phase 9 having drained
  // every producer's write-through stores before any consumer reaches this
  // line, which is true for the workspace but says nothing about data
  // published *within* a layer by one XCD and read by another -- and at bs>1
  // there is such data: rmsnorm_out_moe, written by one router worker per XCD
  // with a plain global_store into that XCD's non-coherent L2, then read in
  // Phase 8 by workers that never entered Phase 7 (xcd_rank >=
  // oproj_topk_tiles_per_xcd) and so never touched the line on their own die.
  //
  // That was the bs>1 MoE W13 corruption. It was first fixed with a scoped
  // `sc1` at the Phase 7b acquire, patching the one site that reproduced.
  // What ships instead is one system-scope acquire for the whole layer, and
  // the difference in shape is the point.
  //
  // Why the broad form is preferable to the narrow patch: the per-site fix is
  // only known-correct at the site that happened to reproduce. This file has
  // seven bare `buffer_inv` sites, all argued safe from bs=1 ablations, and
  // the bs=4 evidence shows how weak that kind of argument is here -- with the
  // scoped fix REMOVED, bs=4 still reports 0/720 bad weight groups across
  // three reps, not because it is safe but because twelve active experts
  // stream enough MXFP4 weight to evict the stale line by L2 capacity before
  // it is read. The same capacity accident the `--max-layers 2` ablation above
  // was designed to defeat. Correctness that depends on cache pressure is not
  // correctness, so gate a coherence fix at the LOWEST batch that reproduces
  // (bs=2 here), never the highest: bad-group counts fall 30 / 27 / 0 as the
  // batch grows.
  //
  // Cost: `sc0 sc1` here drops L2 once per layer, at a point where the layer is
  // about to stream its weights in anyway, rather than mid-Phase-7 after the
  // MXFP4 expert weights are already resident -- which is what made the
  // unconditional form at that site cost +0.259 ms/token. QKV_BATCH_SIZE is a
  // template parameter, so the bs=1 build compiles to exactly the `#else` and
  // pays nothing.
#ifdef MPK_INTERLAYER_SPLIT
  unsigned long long _il_binv0 = __builtin_amdgcn_s_memrealtime();
#endif
  if (QKV_BATCH_SIZE > 1) {
    asm volatile("buffer_inv sc0 sc1" ::: "memory");
  } else {
#ifndef MPK_W2_CONSUMER_GATE
    asm volatile("buffer_inv" ::: "memory");
#endif
  }
#ifdef MPK_INTERLAYER_SPLIT
  // buffer_inv has no completion counter to wait on, so this brackets issue
  // cost plus whatever the invalidate stalls behind it, not drain latency.
  unsigned long long _il_binv1 = __builtin_amdgcn_s_memrealtime();
#endif

#ifdef MPK_ATTN_KV_PREFETCH
  // Idle ranks (no QKV, no O-proj DMA) issue this XCD's past-KV into L2
  // while ranks 0-9 run QKV. Attention is on a different CU of the same
  // XCD, so L2 hits are what we want; vL1 on these CUs is irrelevant.
  // Mapping is physical idle rank -> chunk, not qkv_attn_rank % CHUNKS
  // (rank 23 would decode as request 2).
  static_assert(NUM_REQS == 1,
                "MPK_ATTN_KV_PREFETCH maps one idle rank per chunk at "
                "NUM_REQS == 1");
  {
    int const idle_n = workers_per_xcd > oproj_topk_ranks
                           ? workers_per_xcd - oproj_topk_ranks
                           : 0;
    int const pf_chunk = xcd_rank - oproj_topk_ranks;
    if (idle_n == ATTN_PARTICIPANTS && pf_chunk >= 0 &&
        pf_chunk < NUM_KV_CHUNKS) {
      using bf16_t = __hip_bfloat16;
      char const *k_base = reinterpret_cast<char const *>(
          reinterpret_cast<bf16_t const *>(output_ptrs[1]) +
          static_cast<size_t>(xcd_id) * HEAD_DIM);
      char const *v_base = reinterpret_cast<char const *>(
          reinterpret_cast<bf16_t const *>(output_ptrs[2]) +
          static_cast<size_t>(xcd_id) * HEAD_DIM);
      mpk_prefetch_kv_chunk_l2<PAGE_SIZE, HEAD_DIM, NUM_KV_CHUNKS,
                               KV_CACHE_STRIDE>(k_base,
                                                v_base,
                                                kv_indptr,
                                                kv_indices,
                                                kv_last_page_len,
                                                /*request_id=*/0,
                                                pf_chunk,
                                                SLIDING_WINDOW);
    }
  }
#endif

  // NOTE: the layer counter that the MoE W13->W2 barrier derives its release
  // value from is published further down, once qkv_epoch_expected is known.
  // See the LAYER_IDX_SMEM_OFF store just before Phase 1.

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  unsigned long long _fused_t0 = __builtin_amdgcn_s_memrealtime();
  unsigned long long _fused_t0a = 0, _fused_t0b = 0, _fused_t0c = 0,
                     _fused_t0d = 0, _merge_done = 0;
#endif

  // Phase slot index. xcd_rank is fixed for the life of the task and xcd_id
  // is the die, so (xcd, rank) is a stable per-worker identity and the trace's
  // per-rank bands line up across runs.
  int const _pslot_w = xcd_id * workers_per_xcd + xcd_rank;
  MPK_PHASE_MARK(_pslot_w, 0);

#ifdef MPK_INTERLAYER_SPLIT
  // Gated on the same arm as the phase slots, so these three segments are
  // directly subtractable from the PSLOTW slot-0 column. Ungated they would
  // average 70 context-growing prefill iterations into the answer.
  if (tid == 0 && g_phase_arm) {
    unsigned long long _il_slot0 = __builtin_amdgcn_s_memrealtime();
    atomicAdd(&g_il_fnpre_sum, (_il_binv0 - _il_entry) * 10);
    atomicAdd(&g_il_binv_sum, (_il_binv1 - _il_binv0) * 10);
    atomicAdd(&g_il_post_sum, (_il_slot0 - _il_binv1) * 10);
    atomicAdd(&g_il_n, 1ULL);
  }
#endif

#ifdef MPK_ENABLE_MOE_SUBPHASE
  // Activate MoE subphase timing on first decode iteration (nat <= 1).
  if (tid == 0 && num_active_tokens <= 1) {
    int old = atomicCAS(&g_subphase_active, 0, 1);
    if (old == 0) {
      for (int s = 4; s <= 5; s++) {
        for (int p = 0; p < SUBPHASE_MAX_PHASES; p++) {
          g_subphase_ns[s][p] = 0;
        }
        g_subphase_cnt[s] = 0;
      }
      __threadfence();
    }
  }
#endif

  MPK_TW_SUB(1, tile_idx);

  int *oproj_counters_base = static_cast<int *>(input_ptrs[16]);
  int *attn_global = oproj_counters_base + FULL_LAYER_ATTN_GLOBAL_COUNTER_SLOT;
  int *qkv_epoch = oproj_counters_base + FULL_LAYER_QKV_EPOCH_SLOT;
  int *chunk_barrier = oproj_counters_base + FULL_LAYER_CHUNK_BARRIER_SLOT;
  int *routing_ready = oproj_counters_base + 10 * 16;
  int *attn_release = oproj_counters_base + FULL_LAYER_ATTN_XCD_RELEASE_SLOT;

  // Barrier release values, derived from the layer counter rather than read.
  //
  // These used to be snapshots of "current value + 1". That is only correct if
  // this worker reads *before* this layer's producer bumps the counter, and
  // nothing guaranteed it: all 36 layers run inside a single task (see the ml
  // loop in persistent_kernel.cuh) with only a per-block __syncthreads between
  // them, so workers skew freely across layer boundaries. A worker that fell a
  // full layer behind could read an already-bumped counter and then wait for a
  // bump this layer never produces -- an intermittent deadlock, because it
  // needs a full-layer skew to happen.
  //
  // f1fa720 fixed that by forcing *every* worker on the XCD to arrive at the
  // Phase 2 barrier, which ordered each read before every producer. It worked,
  // and it also cost 2.19 -> 2.46 ms/iter. 2c1071c recovered most of that by
  // splitting arrival from waiting (all 30 arrive, only ranks < NUM_KV_CHUNKS
  // block), reaching 2.39. The remaining ~0.2ms was *not* recovered by this
  // change and is still unattributed -- device timing under MPK_DEVICE_TIMING=1
  // cannot localize it, because the ~147k printfs inflate iterations to ~56ms
  // and the skew lands in Phase 6's xcd_barrier, whose median then moves
  // opposite to real latency. This change is a correctness fix; treat its
  // latency effect as neutral.
  //
  // The layer counter removes the race at its source. All three counters start
  // at 0, bump exactly once per layer, and are never reset, so the value this
  // layer drives them to is a pure function of the layer index -- no shared
  // read, nothing to order, and no arrival requirement. See the
  // _linear_reserved store in the ml loop for how it is published.
  //
  //   qkv_epoch[x]      bumped once by the last worker to arrive on XCD x
  //   attn_release[x]   written by the last XCD to reach the attn_global
  //                     barrier, using its *own* expected value -- so producer
  //                     and consumer now agree by construction
  //   routing_ready[*]  read-modify-written once by the single TopK completer
  //
  // Because they are all the same per-layer count, all three expected values
  // are the same number.
  int const layer_counter = task_layer_idx;
  int const routing_expected = layer_counter + 1;
  int const attn_release_expected = layer_counter + 1;
  int const qkv_epoch_expected = layer_counter + 1;
#ifdef MPK_EARLY_ROUTING
  // Packed into MoE tile_idx[15:8] as expert_id+1. 0 means "no early expert".
  int routed_expert0 = 0;
#endif

  // Only the workers that read this layer's QKV output take part in the epoch
  // barrier. The expected values above no longer depend on arrival ordering,
  // so the participant set is free to be the set that actually needs the
  // barrier. This is the pre-f1fa720 participant set, now safe to use because
  // nothing reads a shared counter -- it measured neutral, not faster.
  //
  // The max() is what keeps the guard and the modulus in agreement: attention
  // workers *wait* on this epoch, so they must also be counted as arriving.
  // With multiple requests the attention set is ATTN_PARTICIPANTS, not
  // NUM_KV_CHUNKS, and this has to track it or the modulus never fires.
  int const qkv_epoch_participants = qkv_work_slots > ATTN_PARTICIPANTS
                                         ? qkv_work_slots
                                         : ATTN_PARTICIPANTS;

  // Publish the layer counter the MoE W13->W2 barrier keys off.
  //
  // That barrier derives its release value as layer_idx + 1
  // (gang_moe_fused_mxfp4_mi300.cuh), and its d_barrier is monotonic -- never
  // reset. So layer_idx must be monotonic across the whole run, not per-layer.
  // This slot used to be stored as 0 on every entry, which made release_val
  // permanently 1: after the very first layer wrote 1, the barrier was already
  // satisfied for every later layer, so W2 workers stopped waiting for their
  // own layer's W13 and could read swiglu_out before it was written. The tile
  // ordering (all W13 tiles precede all W2 tiles, padded so every worker
  // starts on W13) usually hid it, but nothing enforced it.
  //
  // qkv_epoch_expected is exactly the counter needed: it counts
  // (iterations * num_layers + layer), monotonic and never reset. It is now a
  // pure function of the layer index rather than a snapshot, so every worker
  // -- on this XCD or any other -- computes the same value with no barrier
  // required to make it agree. That is strictly stronger than what this slot
  // relied on before.
  {
    constexpr int LAYER_IDX_SMEM_OFF =
        mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE -
        mirage::runtime::LAYER_IDX_SMEM_OFFSET_FROM_END;
    extern __shared__ char _layer_smem_init[];
    if (tid == 0) {
      *reinterpret_cast<int *>(&_layer_smem_init[LAYER_IDX_SMEM_OFF]) =
          qkv_epoch_expected;
    }
  }
  __syncthreads();

  // ══════════════════════════════════════════════════════════════════
  // Phase 1: QKV GEMM
  // ══════════════════════════════════════════════════════════════════
  MPK_TW_SUB(10, xcd_rank);
  if (qkv_does_qkv) {
#ifdef MPK_QKV_KSPLIT
    int const qkv_k_part = qkv_does_k1 ? 1 : 0;
    int const qkv_tile =
        qkv_does_k1 ? qkv_attn_rank - qkv_k1_base : qkv_attn_rank;
#else
    int const qkv_tile = qkv_attn_rank;
#endif
    gang_resaddf32_rmsnorm_linear_mxfp4_bias_kvupd_kernel<QKV_BATCH_SIZE,
                                                          QKV_OUTPUT_PER_WG,
                                                          QKV_REDUCTION_SIZE,
                                                          ACTUAL_HIDDEN_DIM,
                                                          HEAD_DIM,
                                                          NUM_Q_PER_KV,
                                                          PAGE_SIZE>(
        input_ptrs[0],
        input_ptrs[1],
        input_ptrs[2],
        input_ptrs[3],
        input_ptrs[4],
        input_ptrs[5],
        output_ptrs[0],
        output_ptrs[1],
        output_ptrs[2],
        output_ptrs[3],
        cos_ptr,
        sin_ptr,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_active_tokens,
        qkv_n_wgs_per_xcd,
        kv_stride,
        q_ws_stride,
        qkv_tile,
#ifdef MPK_PREFETCH_NEXT_QKV
        // Skip the DMA: the previous layer's Phase 9 already staged these
        // exact bytes into this exact LDS region during its barrier spin.
        // input_ptrs[25] is this layer's own weight pointer when the previous
        // layer prefetched and null otherwise (it is null on layer 0, which
        // has no previous layer), so this reduces to "was there a producer".
        // The equality check is not redundant: it is the one place the two
        // ends of the hand-off can disagree, and a mismatch here would be
        // silent wrong numerics rather than a fault.
        /*weights_preloaded=*/input_ptrs[25] == input_ptrs[4]
#else
        /*weights_preloaded=*/false
#endif
#ifdef MPK_QKV_KSPLIT
        ,
        qkv_k_part
#endif
    );

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
    _fused_t0a = __builtin_amdgcn_s_memrealtime();
#endif
  } // end Phase 1: QKV GEMM
  MPK_PHASE_MARK(_pslot_w, 1);

  // ══════════════════════════════════════════════════════════════════
  // Phase 2: QKV barrier — epoch-based
  //
  // Joined only by the workers that produce or consume this layer's QKV
  // output: the QKV GEMM workers and the attention-chunk workers. The MoE-only
  // workers beyond both skip it entirely and go straight to Phase 6, where
  // their O-proj weight DMA can start overlapping immediately.
  //
  // This barrier no longer carries any ordering for the other two counters.
  // It used to: every worker had to arrive so that the attn_release /
  // routing_ready snapshots were ordered against their producers. Those are
  // now derived from the layer counter and read nothing shared, so the
  // participant set is just the set that needs the barrier for its own sake --
  // workers that read K/V in Phase 3 must see the GEMM land first.
  //
  // The arrival count is qkv_epoch_participants, and the modular test only
  // fires when it matches the number of workers that actually arrive -- so
  // this constant and the guard below must stay in agreement.
  // ══════════════════════════════════════════════════════════════════
  MPK_TW_SUB(20, qkv_epoch_expected);
  MPK_WS_PHASE(20, qkv_epoch_expected, xcd_id);
#ifdef MPK_QKV_KSPLIT
  if (qkv_does_qkv)
#else
  if (qkv_attn_rank < qkv_epoch_participants)
#endif
  {
    __shared__ int s_prev;
#ifdef MPK_QKV_EPOCH_PRODUCER_DRAIN
    // Publish this workgroup's Q/K/V stores into this XCD's L2 before the
    // arrival atomic. Same intra-XCD pattern as the Phase 4 chunk barrier:
    // s_waitcnt is per-wave, so drain first, then rendezvous, then arrive.
    // The last arriver's epoch bump then means every producer has retired
    // its payload, and the consumer only needs the vL1 `buffer_inv` below
    // -- not the agent fence that `buffer_inv sc1`s this XCD's whole L2.
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __syncthreads();
#endif
    if (tid == 0) {
      // Both counters on this barrier are XCD-private: the arrival line is
      // `[xcd_id]` and the epoch line is `[xcd_id * 16]`, and the only reader
      // of either -- the poll below -- indexes by its own `xcd_id` too. No
      // other XCD ever touches these lines, so the `sc1` that pushed each
      // atomic out to the device coherency point was buying visibility to
      // nobody. `sc0` keeps them in this XCD's L2 where every participant
      // already is.
      s_prev = MPK_XCD_LOCAL_ATOM_ADD(
          &static_cast<int *>(input_ptrs[7])[xcd_id], 1);
    }
    __syncthreads();

    if ((s_prev % qkv_epoch_participants) == qkv_epoch_participants - 1) {
      // Last worker to arrive: bump epoch (no reset needed — modular check)
      if (tid == 0) {
        MPK_XCD_LOCAL_ATOM_ADD(&qkv_epoch[xcd_id * 16], 1);
      }
    }

    // Every participant polls. The participant set is now exactly the workers
    // that need this barrier, so there is no longer a subset that arrives only
    // to carry ordering for someone else.
    MPK_WS_WAIT_BEGIN(20, qkv_epoch_expected);
    if (tid == 0) {
      int _obs;
      int _spins = 0;
      while ((_obs = __atomic_load_n(&qkv_epoch[xcd_id * 16],
                                     __ATOMIC_RELAXED)) < qkv_epoch_expected) {
        MPK_WS_WAIT_TICK(_obs, _spins);
        _spins++;
        __builtin_amdgcn_s_sleep(1);
      }
#if !defined(MPK_QKV_GATE_NO_AGENT_FENCE) &&                                    \
    !defined(MPK_QKV_EPOCH_PRODUCER_DRAIN)
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
#endif
    }
    __syncthreads();
    // The agent-scope fence above is LOAD-BEARING unless the producer already
    // drained Q/K/V into this XCD's L2 before arriving (MPK_QKV_EPOCH_PRODUCER_DRAIN).
    // Do not drop it alone (MPK_QKV_GATE_NO_AGENT_FENCE): that is a ctx-4096
    // race. The drain arm is the Phase 4 publish protocol applied here.
    //
    // The argument for dropping it (MPK_QKV_GATE_NO_AGENT_FENCE, now default
    // off) was: both counters here are XCD-private and written with sc0
    // (MPK_XCD_LOCAL_ATOM_ADD), so the fence's `buffer_inv sc1` throws away
    // this XCD's whole L2 -- including the weights the next phase reads -- to
    // acquire a line that never left it, and the vL1 `buffer_inv` below is the
    // acquire the *counter* needs.
    //
    // Every clause of that is true. The conclusion does not follow. An acquire
    // fence does not only order the load of the flag; it orders every load
    // AFTER it against every write the releaser made BEFORE its release. What
    // this gate publishes is not the counter, it is the QKV output the
    // attention phase is about to read. The vL1-only `buffer_inv` drops this
    // CU's stale lines but carries no such ordering, so the reader can observe
    // the bumped epoch and still miss the writes it was standing for.
    //
    // Measured, not argued: at ctx 4096 three byte-identical runs produced
    // three different token sequences with the fence dropped, and 5/5 runs are
    // identical with it restored (matching the all-flags-off reference). At
    // ctx 512 both forms give the same text hash -- the same seq-len
    // dependence the Phase 4 comment below describes, and the reason an A/B at
    // 512 scored this as free. It costs 1.945 -> 1.981 ms/tok at bs=1.
    asm volatile("buffer_inv" ::: "memory");
  }

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  _fused_t0b = __builtin_amdgcn_s_memrealtime();
#endif
  MPK_PHASE_MARK(_pslot_w, 2);

  // ══════════════════════════════════════════════════════════════════
  // Phase 3: Parallel attention chunks
  // Workers 0..(NUM_KV_CHUNKS-1) each run one chunk.
  //
  // NOTE: this block is deliberately NOT nested inside the Phase 1 QKV guard.
  // It used to be, which capped the usable chunk count at
  // total_qkv_tiles_per_xcd (10 for GPT-OSS 120B): with NUM_KV_CHUNKS > 10 only
  // 10 workers ever reached the chunk barrier, the
  // `(s_chunk_prev % NUM_KV_CHUNKS) == NUM_KV_CHUNKS-1` merge condition never
  // fired, and the megakernel deadlocked. Attention chunks are now served by
  // any of the workers_per_xcd (30) workers on this XCD.
  MPK_TW_SUB(30, xcd_rank);
  MPK_WS_PHASE(30, qkv_epoch_expected, xcd_id);
  {
    if (qkv_attn_rank < ATTN_PARTICIPANTS) {
      int kv_chunk_idx = attn_chunk;
      using bf16_t = __hip_bfloat16;
      void const *offset_k = reinterpret_cast<bf16_t const *>(output_ptrs[1]) +
                             static_cast<size_t>(xcd_id) * HEAD_DIM;
      void const *offset_v = reinterpret_cast<bf16_t const *>(output_ptrs[2]) +
                             static_cast<size_t>(xcd_id) * HEAD_DIM;

      // Write float32 partials to o_acc_f32 (input_ptrs[23])
      // Write LSE to lse_acc (input_ptrs[8])
      // NO sinks for per-chunk — sinks applied in merge step
#ifdef MPK_ATTN_SETPRIO
      asm volatile("s_setprio 1");
#endif
      paged_attention_ck_fmha_split_kv_impl<bfloat16,
                                            NUM_Q_PER_KV,
                                            HEAD_DIM,
                                            PAGE_SIZE,
                                            MAX_SEQ_LEN,
                                            NUM_KV_CHUNKS,
                                            Q_WORKSPACE_STRIDE,
                                            KV_CACHE_STRIDE,
                                            NUM_KV_HEADS,
                                            DECODE_ONLY>(
          output_ptrs[3], // q_workspace
          const_cast<void *>(offset_k),
          const_cast<void *>(offset_v),
          input_ptrs[23], // o_acc_f32 (float32 partials)
          input_ptrs[8],  // lse_acc
          qo_indptr,
          kv_indptr,
          kv_indices,
          kv_last_page_len,
          /*request_id=*/attn_req,
          /*kv_head_idx=*/xcd_id,
          kv_chunk_idx,
          attn_scale,
          SLIDING_WINDOW,
          nullptr); // no sinks per-chunk
#ifdef MPK_ATTN_SETPRIO
      asm volatile("s_setprio 0");
#endif

#ifdef MPK_ATTN_SPLIT_CHUNK
      // Drain this chunk's low-half stores before the fold reads them.
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      __syncthreads();
      // Helper published this chunk's high-half (O, LSE). Fold into the
      // primary slot before the 8-way merge so NUM_KV_CHUNKS stays 8.
      {
        int *split_flag =
            &chunk_barrier[(xcd_id * NUM_REQS + attn_req) * 16 + 1 +
                           kv_chunk_idx];
        if (tid == 0) {
          while (MPK_LD_GATE(split_flag) < qkv_epoch_expected) {
            __builtin_amdgcn_s_sleep(1);
          }
        }
        __syncthreads();
        asm volatile("buffer_inv" ::: "memory");
        mpk_fold_split_chunk_partials<NUM_Q_PER_KV,
                                      HEAD_DIM,
                                      NUM_KV_HEADS,
                                      NUM_KV_CHUNKS>(
            reinterpret_cast<float *>(input_ptrs[8]),
            reinterpret_cast<float *>(input_ptrs[23]),
            qo_indptr[attn_req],
            xcd_id,
            kv_chunk_idx);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        __syncthreads();
      }
#endif

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
      _fused_t0c = __builtin_amdgcn_s_memrealtime();
#endif

      // ══════════════════════════════════════════════════════════════════
      // Phase 4: Chunk barrier — last-chunk-worker runs merge
      // ══════════════════════════════════════════════════════════════════
      MPK_TW_SUB(40, kv_chunk_idx);
      MPK_WS_PHASE(40, qkv_epoch_expected, xcd_id);
      // Flush this chunk's o_acc/lse_acc partials before arriving.
      //
      // The chunk kernel writes its partials with ordinary stores, which may
      // still be sitting in this CU's write buffer / vL1 when the atomic
      // below retires. atom_add_release_gpu_s32 orders *this thread's* prior
      // writes, but tid 0 is not the thread that wrote most of this chunk's
      // partials -- the other 255 threads did, and __syncthreads is a
      // block-execution barrier, not a memory-visibility one to other CUs.
      // Without this, the merging worker (a *different* block, possibly a
      // different CU) can read a partially-written o_acc/lse_acc slot.
      //
      // The result is silent numerical corruption, not a crash: merge weights
      // each chunk by exp2(lse - m_global), so a stale/torn lse reads as
      // garbage magnitude. A torn lse that lands large makes m_global huge and
      // drives every other chunk's weight to zero; one that lands as raw
      // uninitialized bits can be NaN, which propagates through the whole
      // attention output. That is the seq-len-dependent part: at 512 tokens
      // ntiles=32 so 14 of 30 chunks exit early via the empty-chunk path and
      // never race, while at 32k ntiles=2048 gives every one of the 30 chunks
      // real work to write, so the window is open on all of them every layer.
      //
      // Scope: this barrier is per-XCD (chunk_barrier[xcd_id * 16]), so the
      // chunk workers and the merging worker are always on the same XCD and
      // share one 32MB L2. Making the partials visible therefore only requires
      // getting them *into* L2 -- not flushing L2 to HBM.
      //
      // db48239 used threadfence_gpu() here, which is agent scope and lowers to
      // `buffer_wbl2 sc1; s_waitcnt vmcnt(0)` (verified in gfx950 ISA). The
      // buffer_wbl2 is an L2->HBM writeback of the whole cache, paid by every
      // chunk worker on every layer, and it buys nothing a same-XCD consumer
      // can observe: see the intra-XCD section of mpk_atoms.cuh, "all CUs
      // within an XCD share the same 32MB L2 ... No buffer_wbl2 required".
      // Dropping it is worth ~0.16 ms/iter at seq 512.
      //
      // What remains is the part that is actually load-bearing: s_waitcnt
      // vmcnt(0) retires all 256 threads' outstanding stores so they have
      // reached L2 before tid 0's release atomic. Workgroup-scope fences emit
      // no instruction at all on gfx950, so they cannot substitute -- the
      // consumer is a different block.
      //
      // One barrier line per (XCD, request). The modulus stays NUM_KV_CHUNKS,
      // so the releaser is the last chunk to arrive *for this request* and it
      // merges *this* request -- exactly the invariant the single-request code
      // had, replicated NUM_REQS times.
      //
      // A single shared line with modulus ATTN_PARTICIPANTS would also be a
      // correct barrier, but the releaser's identity would then carry no
      // information about which request finished, so it would have to merge all
      // NUM_REQS requests serially. That puts (NUM_REQS-1) merges on the
      // critical path of every one of the 36 layers, while these per-request
      // merges run on workers that would otherwise just spin in Phase 6.
      // Drain, THEN rendezvous, THEN arrive. The order matters and was wrong
      // here: with __syncthreads() first, wave 0 can clear its own vmcnt and
      // execute tid 0's arrival while waves 1-3 have not yet retired the
      // partials they wrote -- s_waitcnt is a per-wave counter, and s_barrier
      // carries no vmcnt guarantee for the *other* waves. Draining before the
      // barrier makes the rendezvous the point at which all four waves are
      // known to have retired, which is what the arrival needs to publish.
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      __syncthreads();
      __shared__ int s_chunk_prev;
      if (tid == 0) {
        // XCD-private counter: the line is indexed by this worker's own
        // `xcd_id`, and its only reader is the `% NUM_KV_CHUNKS` test right
        // below on the same XCD. See MPK_XCD_LOCAL_BARRIER at the top.
        s_chunk_prev = MPK_XCD_LOCAL_ATOM_ADD(
            &chunk_barrier[(xcd_id * NUM_REQS + attn_req) * 16], 1);
      }
      __syncthreads();

      if ((s_chunk_prev % NUM_KV_CHUNKS) == NUM_KV_CHUNKS - 1) {
        // Last chunk worker: run merge (no reset needed — modular check)
        //
        // Acquire the partials the other chunks released above. buffer_inv
        // drops this CU's stale vL1 lines so the merge reads what they
        // actually wrote rather than a cached copy from a previous layer --
        // vL1 is per-CU, so this is required even though the producers share
        // our L2.
        //
        // Plain `buffer_inv` (no sc1) invalidates vL1 only. The acquire fence
        // that used to precede it was agent scope, which emits `buffer_inv
        // sc1` and additionally invalidates L2 -- discarding lines this XCD's
        // own chunk workers had just written, and forcing them to be re-read
        // from HBM. The producers are all on this XCD (the barrier is
        // per-XCD), so their stores are already in our L2 and invalidating it
        // is both unnecessary and actively harmful.
        asm volatile("buffer_inv" ::: "memory");

        // ══════════════════════════════════════════════════════════════════
        // Phase 5: Merge (write-through fused) + signal
        // ══════════════════════════════════════════════════════════════════
        MPK_TW_SUB(50, s_chunk_prev);
        MPK_WS_PHASE(50, qkv_epoch_expected, xcd_id);
        // WRITE_THROUGH=true: merge writes bf16 output directly via st_wt,
        // eliminating the separate __syncthreads + readback + flush pass.
        merge_splitkv_ck_fmha<__hip_bfloat16,
                              NUM_Q_PER_KV,
                              NUM_KV_HEADS,
                              HEAD_DIM,
                              NUM_KV_CHUNKS,
                              128,
                              4096,
                              /*WRITE_THROUGH=*/true>(
            reinterpret_cast<float const *>(input_ptrs[8]),  // lse_acc
            reinterpret_cast<float const *>(input_ptrs[23]), // o_acc_f32
            qo_indptr,
            kv_indptr,
            kv_last_page_len,
            /*request_id=*/attn_req,
            reinterpret_cast<__hip_bfloat16 *>(
                output_ptrs[4]), // attn_out (bf16)
            /*kv_head_idx=*/xcd_id,
            HAS_SINKS ? input_ptrs[6] : nullptr); // sinks applied here

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
        _merge_done = __builtin_amdgcn_s_memrealtime();
#endif
        // Wait for all write-through stores from merge to complete.
        //
        // The __syncthreads() is load-bearing and was missing. merge's st_wt
        // stores are issued by all 256 threads (its work loop is strided by
        // `group_id = threadIdx.x / THREADS_PER_TOKEN`, covering every wave),
        // but only tid 0 publishes the release below. s_waitcnt vmcnt(0) is a
        // *per-wave* counter, so wave 0 draining its own stores says nothing
        // about waves 1-3 -- their attn_out writes can still be in flight when
        // attn_global is bumped and the release flags fan out.
        //
        // The consumer is Phase 7's O-proj, which reads attn_out from a
        // different block. The corruption is silent and batch-size-dependent:
        // it shows up as nondeterministic attn_proj_out confined to the
        // workers that clear the release poll earliest -- those with no Phase 1
        // QKV work, i.e. wg_idx >= total_qkv_tiles_per_xcd.
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        __syncthreads();
        // TESTED AND NOT ADOPTED: the lane-parallel release fan-out that pays
        // off at the Phase 9 and O-proj barriers (readfirstlane the epoch out
        // of tid 0, then `if (tid < 8) st_wt_u32`) measures 2.047 here against
        // 2.038 for this serial loop. The transform's premise -- that 239
        // workgroups are already spinning on flags this one lane has not
        // written -- does not hold at this barrier: only the workers with a
        // Phase 1 or attention role join Phase 6, they arrive spread out rather
        // than pre-parked, and this releaser is not on anyone's critical path
        // for the seven store-issues it saves. Left serial.
#ifdef MPK_ATTN_SLICE_RELEASE
        // ── Per-XCD slice release ────────────────────────────────────────
        //
        // This XCD's merge just finished, and what it finished is exactly the
        // 512-element K slice [xcd_id*512, xcd_id*512+512) of attn_out: the
        // merge is called with kv_head_idx == xcd_id, so XCD x owns q heads
        // [8x, 8x+8) and NUM_Q_PER_KV * HEAD_DIM == 8 * 64 == 512 of the
        // O-proj's 4096-wide reduction. Eight XCDs tile it exactly.
        //
        // So there is nothing for this XCD to learn from the other seven. The
        // default path below still routes every arrival through attn_global
        // and has the *last* XCD fan out all eight flags, which makes the
        // slowest merge on the machine the release time for all eight slices.
        // Here each XCD publishes its own slice the moment it is ready and the
        // consumer waits per slice (see the ATTENTION_SLICE_RELEASE arm in
        // gang_linear_mxfp4_res_bias_rmsnorm_topk_mi300.cuh, where wave w
        // polls only XCDs 2w and 2w+1 -- the two slices its K interval reads).
        //
        // The `s_waitcnt vmcnt(0)` + `__syncthreads()` above is what makes
        // this sound: all four merge waves have drained their st_wt stores
        // before any of them reaches here, so tid 0's flag cannot outrun a
        // sibling wave's attn_out writes.
        //
        // NUM_REQS == 1 only. With more requests attn_global is bumped once
        // per (XCD, request) and an XCD's slice is not complete until all of
        // its requests have merged, so a single unconditioned store here would
        // release after the first. The static_assert is at the top of the
        // MPK_ATTN_SLICE_RELEASE block in Phase 6.
        if (tid == 0) {
          st_wt_u32((void *)&attn_release[xcd_id * 16],
                    (unsigned)attn_release_expected);
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        }
#else
        if (tid == 0) {
          // attn_global is bumped exactly ATTN_ARRIVALS times per layer: the
          // modulus above fires once per (XCD, request), and every empty-work
          // early return lives *inside* a callee, so no arrival is ever
          // skipped. The releaser is therefore the arrival whose *own* return
          // value is the last of a group of ATTN_ARRIVALS -- a pure function
          // of this atomic's result, reading nothing shared.
          //
          // This used to snapshot the counter and round up:
          //
          //   v = relaxed_load(attn_global);
          //   expected = (v / ATTN_ARRIVALS + 1) * ATTN_ARRIVALS;
          //   if (atom_add(attn_global, 1) == expected - 1) release;
          //
          // which is only correct if the snapshot lands in this layer's group.
          // Nothing enforced that. All 36 layers run inside one task with only
          // a per-block __syncthreads between them, so a worker that reaches
          // here before the previous layer's last arrival reads a `v` from the
          // *previous* group, computes `expected` one full ATTN_ARRIVALS too
          // low, and then satisfies `prev == expected - 1` with its own bump --
          // firing this layer's release when as few as one of the merges has
          // run. The consumer's Phase 6 poll clears immediately and O-proj
          // reads attn_out rows that no merge has written yet.
          //
          // That is exactly the observed defect: silent, run-to-run varying,
          // and concentrated in the high-numbered requests (whose merges finish
          // last) and in the workers that reach Phase 6 earliest (xcd_rank >=
          // total_qkv_tiles_per_xcd, i.e. the wg >= 10 block). It also explains
          // why a delay *anywhere* in Phase 6 -- before the poll or after it --
          // cleaned it up completely while no fence on either side did
          // anything: the data was never unflushed, it was simply not computed
          // yet, and the delay just gave the real merges time to land.
          //
          // The modulus removes the race at its source: no shared read, nothing
          // to order, and no dependence on the value being sampled inside the
          // right group. It is also self-healing across kernel relaunches (the
          // counter is a host allocation that is never reset) and correct at
          // every NUM_REQS -- unlike `(v | (ATTN_ARRIVALS-1)) + 1`, which only
          // rounds to a power of two and released early at NUM_REQS in
          // {3,5,6,7}.
          constexpr int ATTN_ARRIVALS = 8 * NUM_REQS;
          int prev = atom_add_release_gpu_s32(attn_global, 1);
          if ((prev % ATTN_ARRIVALS) == ATTN_ARRIVALS - 1) {
            // LAST XCD to arrive: fan out per-XCD release flags via st_wt
            // (write-through to HBM, bypasses L2 — guarantees ld_nt sees it).
            // This eliminates the cross-XCD poll of attn_global, which hangs
            // because ld_nt/relaxed loads read stale L2 cached values.
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            for (int x = 0; x < 8; x++) {
              st_wt_u32((void *)&attn_release[x * 16],
                        (unsigned)attn_release_expected);
            }
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          }
        }
#endif
#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
        _fused_t0d = __builtin_amdgcn_s_memrealtime();
#endif
      }
#ifdef MPK_ATTN_SPLIT_CHUNK
    } else if (is_split_helper) {
      // Idle rank 23+c: high half of chunk c. Consumer of Q — wait for the
      // epoch without arriving (arrival count stays QKV+attn prefix).
      if (tid == 0) {
        while (MPK_LD_GATE(&qkv_epoch[xcd_id * 16]) < qkv_epoch_expected) {
          __builtin_amdgcn_s_sleep(1);
        }
      }
      __syncthreads();
      asm volatile("buffer_inv" ::: "memory");
      {
        using bf16_t = __hip_bfloat16;
        void const *offset_k =
            reinterpret_cast<bf16_t const *>(output_ptrs[1]) +
            static_cast<size_t>(xcd_id) * HEAD_DIM;
        void const *offset_v =
            reinterpret_cast<bf16_t const *>(output_ptrs[2]) +
            static_cast<size_t>(xcd_id) * HEAD_DIM;
        paged_attention_ck_fmha_split_kv_impl<bfloat16,
                                              NUM_Q_PER_KV,
                                              HEAD_DIM,
                                              PAGE_SIZE,
                                              MAX_SEQ_LEN,
                                              NUM_KV_CHUNKS,
                                              Q_WORKSPACE_STRIDE,
                                              KV_CACHE_STRIDE,
                                              NUM_KV_HEADS,
                                              DECODE_ONLY>(
            output_ptrs[3],
            const_cast<void *>(offset_k),
            const_cast<void *>(offset_v),
            input_ptrs[23],
            input_ptrs[8],
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            /*request_id=*/0,
            /*kv_head_idx=*/xcd_id,
            split_hchunk,
            attn_scale,
            SLIDING_WINDOW,
            nullptr,
            /*split_part=*/1);
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      __syncthreads();
      if (tid == 0) {
        st_wt_u32((void *)&chunk_barrier[(xcd_id * NUM_REQS) * 16 + 1 +
                                         split_hchunk],
                  (unsigned)qkv_epoch_expected);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      }
#endif
    }
  }

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  unsigned long long _fused_t1 = __builtin_amdgcn_s_memrealtime();
#endif
  // Slot 3 = end of attention chunk compute + merge; slot 4 = the attn_release
  // wait begins. They are adjacent here because Fleet's O-proj weight DMA is
  // issued inside the Phase 6 block below, so the DMA overlaps the gate's
  // spin-wait rather than serializing ahead of it.
  MPK_PHASE_MARK(_pslot_w, 3);
  MPK_PHASE_MARK(_pslot_w, 4);

  // ══════════════════════════════════════════════════════════════════
  // Phase 6: Cross-XCD attention barrier + O-proj weight DMA
  // Issue buffer_load_lds for O-proj weights BEFORE the barrier poll
  // so DMA runs in the background during the spin-wait (~5-60us).
  // ══════════════════════════════════════════════════════════════════
  MPK_TW_SUB(60, attn_release_expected);
  MPK_WS_PHASE(60, qkv_epoch_expected, xcd_id);
  // Same quantity the QKV/attention rotation is sized from; computed once at
  // the top of the kernel so the two cannot drift.
  int oproj_topk_tiles_per_xcd = oproj_topk_ranks;

  {
    constexpr int OPROJ_WG_DATA =
        OPROJ_OUTPUT_PER_WG * (OPROJ_REDUCTION_SIZE / 2);
    constexpr int OPROJ_NUM_B32 = OPROJ_REDUCTION_SIZE / 32;
    constexpr int OPROJ_WG_SCALE = OPROJ_OUTPUT_PER_WG * OPROJ_NUM_B32;
    constexpr int OPROJ_WG_BYTES = OPROJ_WG_DATA + OPROJ_WG_SCALE;
    constexpr int OPROJ_N16_DATA = OPROJ_WG_DATA / 16;
    constexpr int OPROJ_LPT = (OPROJ_N16_DATA + 255) / 256;
    constexpr int OPROJ_DATA_PAD = OPROJ_LPT * 256 * 16;
    constexpr int OPROJ_N16_SCALE = (OPROJ_WG_SCALE + 15) / 16;
    constexpr int OPROJ_SLPT = (OPROJ_N16_SCALE + 255) / 256;
    constexpr int OPROJ_SCALE_PAD = OPROJ_SLPT * 256 * 16;

    // Weight region base. This must agree byte-for-byte with the O-proj
    // kernel's own OPROJ_LDS_OFF -- the DMA below writes where this says and
    // the MFMA reads where that says, and a mismatch is silent wrong
    // numerics. Both derive it from the same constexpr function.
    constexpr int OPROJ_LDS_W_OFF =
        oproj_lds_w_off(QKV_BATCH_SIZE, OPROJ_REDUCTION_SIZE);

    extern __shared__ char _oproj_pf_smem[];

    int oproj_tile_idx_pf = xcd_id * oproj_topk_tiles_per_xcd + oproj_local;
    // Tile space is (column block, weight group) now, matching the O-proj
    // kernel's decode. The weight group is what selects the DMA source; the
    // column block only decides whether this tile has any live token at all.
    int oproj_bblk_pf = oproj_local / oproj_n_wgs_per_xcd;
    int oproj_wg_pf = oproj_local % oproj_n_wgs_per_xcd;

    if (does_oproj && oproj_bblk_pf * 16 < num_active_tokens) {
      uint8_t const *oproj_W = (uint8_t const *)input_ptrs[9];
      uint32_t oproj_buf_range =
          static_cast<uint32_t>(oproj_n_wgs_per_xcd) * OPROJ_WG_BYTES;
      i32x4_t oproj_rsrc = make_w_buffer_rsrc(oproj_W, oproj_buf_range);
      uint32_t oproj_wg_voff =
          static_cast<uint32_t>(oproj_wg_pf) * OPROJ_WG_BYTES;

      auto *oproj_lds_base = (__attribute__((address_space(3)))
                              uint32_t *)(_oproj_pf_smem + OPROJ_LDS_W_OFF);
      // buffer_load_lds writes to M0 + lane_id*16.  Each wave fills 1024 bytes.
      // Must offset by warp_id*1024 so 4 waves write to distinct slices.
      int const pf_warp_id = tid >> 6;

#pragma unroll
      for (int j = 0; j < OPROJ_LPT; j++) {
        int idx = tid + j * 256;
        int clamped = idx < OPROJ_N16_DATA ? idx : OPROJ_N16_DATA - 1;
        uint32_t voff = oproj_wg_voff + static_cast<uint32_t>(clamped) * 16;
        auto *lds_dst =
            (__attribute__((address_space(3)))
             uint32_t *)((uint8_t
                          __attribute__((address_space(3))) *)oproj_lds_base +
                         j * 4096 + pf_warp_id * 1024);
        __llvm_amdgcn_raw_buffer_load_lds(
            oproj_rsrc, lds_dst, 16, static_cast<int>(voff), 0, 0, 3);
      }
#pragma unroll
      for (int j = 0; j < OPROJ_SLPT; j++) {
        int idx = tid + j * 256;
        int clamped = idx < OPROJ_N16_SCALE ? idx : OPROJ_N16_SCALE - 1;
        uint32_t voff =
            oproj_wg_voff + OPROJ_WG_DATA + static_cast<uint32_t>(clamped) * 16;
        auto *lds_dst =
            (__attribute__((address_space(3)))
             uint32_t *)((uint8_t
                          __attribute__((address_space(3))) *)oproj_lds_base +
                         OPROJ_DATA_PAD + j * 4096 + pf_warp_id * 1024);
        __llvm_amdgcn_raw_buffer_load_lds(
            oproj_rsrc, lds_dst, 16, static_cast<int>(voff), 0, 0, 3);
      }
    }
  }

  MPK_WS_WAIT_BEGIN(60, attn_release_expected);
#ifdef MPK_ATTN_SLICE_RELEASE
  static_assert(NUM_REQS == 1,
                "MPK_ATTN_SLICE_RELEASE publishes one flag per XCD from the "
                "merge, but with NUM_REQS > 1 an XCD's attn_out slice is only "
                "complete after all of its requests have merged. Use the "
                "attn_global rendezvous at more than one request.");
  static_assert(OPROJ_REDUCTION_SIZE == 8 * NUM_Q_PER_KV * HEAD_DIM,
                "the eight per-XCD merge slices must tile the O-proj "
                "reduction exactly -- the consumer maps wave w to XCDs 2w and "
                "2w+1 on that assumption");
  // The O-proj/TopK workers do not wait here at all. They are the only
  // consumers of attn_out, and each of their waves waits for just the two
  // slices its own K interval reads, inside the Phase 7 kernel. Waiting for
  // this XCD's flag first would be strictly worse than useless: it is one
  // more slice than wave 0 needs and one fewer than wave 3 needs, so it
  // would delay the early waves without protecting the late ones.
  //
  // Everyone else still waits, unchanged. They read nothing from attn_out --
  // this is the ordering that keeps them from running ahead into Phase 8 --
  // and their own XCD's flag is the cheapest sufficient one to wait on.
  if (!does_oproj)
#endif
  {
    int _obs;
    int _spins = 0;
#ifdef MPK_NARROW_GATE_POLL
    // One thread polls for the block instead of all 256.
    //
    // Every thread reading the same release line turns one arrival into 240
    // (30 workers x 4 waves x 2 sites) coherent reads of a single cacheline
    // per layer, and at `sc0 sc1` each of those is a real trip to the
    // coherency point rather than a vL1 hit. The line is contended precisely
    // when it is about to be written, which is the worst case for it.
    //
    // Free here: the block already rendezvouses immediately below (the
    // `__syncthreads()` that publishes the Phase 6 weight DMA), so the other
    // 255 threads gain nothing from having observed the flag themselves --
    // they cannot pass that barrier until tid 0 has. The `buffer_inv` is
    // still executed by every wave, which is what the acquire needs, since
    // buffer_inv is per-wave.
    if (tid == 0)
#endif
      while ((_obs = MPK_LD_GATE(&attn_release[xcd_id * 16])) <
             attn_release_expected) {
        MPK_WS_WAIT_TICK(_obs, _spins);
        _spins++;
        __builtin_amdgcn_s_sleep(1);
      }
#ifdef MPK_NARROW_GATE_POLL
    // Order the other waves behind tid 0's observation *before* they run the
    // acquire below: a wave that invalidated early would just refill vL1 with
    // pre-release lines and the acquire would be void.
    __syncthreads();
#endif
  }
  // Cross-XCD ACQUIRE for attn_out. Plain `buffer_inv` (vL1 only), not `sc1`.
  //
  // Under MPK_ATTN_SLICE_RELEASE this one no longer acquires attn_out for the
  // O-proj workers -- they reach it without having observed any release flag,
  // so an attn_out line left in vL1 by the *previous* layer would survive the
  // invalidate and then be hit by this layer's read. Their acquire moved to
  // where their observation is, per wave, right after the slice poll in the
  // Phase 7 kernel. This site is kept for the workers that still wait above,
  // and because it is the acquire for the whole Phase 6 hand-off, not just
  // for attn_out.
  //
  // The producer is the Phase 5 merge, which writes attn_out with st_wt
  // (WRITE_THROUGH=true) -- sc0 sc1, bypassing L2 and landing in HBM. The
  // consumer is Phase 7's O-proj on a *different* XCD (the merge that produced
  // a given kv_head runs on the XCD that owns it), and MI300/MI350 L2 is not
  // coherent across XCDs. This was `sc1` on that reasoning; see the
  // layer-boundary acquire at the top of this function for why vL1-only is
  // sufficient given the Phase 9 barrier, and for the ablation that
  // established it.
  //
  // Note the ordering below is unchanged and still required: the Phase 6
  // weight DMA is drained BEFORE the invalidate, not after.
  //
  // This is distinct from the Phase 5 barrier's `buffer_inv`, which is correct
  // for a different reason: that one is intra-XCD (the chunk barrier is
  // per-XCD), so the producers share this L2 and an `sc1` there would discard
  // their partials -- it must never become `sc1`, Phase 9 or no Phase 9.
  //
  // buffer_load_lds retires on vmcnt, and it was issued above so it overlaps
  // the release poll. Draining first keeps `buffer_inv sc1` -- which drops L2,
  // what those `sc0 nt` global->LDS reads are sourcing -- from landing while
  // they are still in flight. The wait has to happen before the MFMA reads LDS
  // regardless, so ordering it ahead of the invalidate costs nothing.
  //
  // This ordering is defensive, not the fix for any observed bug. The wg >= 10
  // corruption that was originally attributed here was the fused layer's
  // barrier release value being -1 (see the layer_counter comment above and
  // persistent_kernel.cuh's per-layer dispatch loop): every barrier was a
  // no-op, so O-proj ran against attn_out that the merges had not written yet.
  // That also explains the evidence that never fit a DMA race -- a delay
  // *before* the release poll cleaned it up just as completely as one after,
  // and neither an extra invalidate nor a system-scope fence on either side
  // changed anything.
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  asm volatile("buffer_inv" ::: "memory");
  // Publish the Phase 6 O-proj weight DMA to the whole block.
  //
  // buffer_load_lds retires on vmcnt, and the drain above is per-wave -- it
  // covers only the loads *this* wave issued. Each wave DMAs its own
  // 1024-byte slices (j*4096 + pf_warp_id*1024), and those slices tile the
  // weight region contiguously, so the tile every wave then reads for its
  // K-range is interleaved across all four waves' slices. Wave 0's MFMA reads
  // bytes wave 3 loaded; without a rendezvous it can read them before they
  // land, and the B operand is whatever LDS held from the previous layer.
  //
  // Silent wrong numerics, no fault: the stale bytes are still a valid MXFP4
  // encoding, so the tile computes cleanly against garbage weights. It shows
  // up in attn_proj_out as columns carrying ~2000-magnitude values where the
  // correct output tops out near 33.
  __syncthreads();

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  unsigned long long _fused_t2 = __builtin_amdgcn_s_memrealtime();
#endif
  MPK_PHASE_MARK(_pslot_w, 5);

  // ══════════════════════════════════════════════════════════════════
  // Phase 7: O-proj + RMSNorm + Router + TopK
  // ══════════════════════════════════════════════════════════════════
  MPK_TW_SUB(70, oproj_topk_tiles_per_xcd);
  MPK_WS_PHASE(70, qkv_epoch_expected, xcd_id);
  {
    if (does_oproj) {
      int oproj_tile_idx = xcd_id * oproj_topk_tiles_per_xcd + oproj_local;
      void const *oproj_residual_ptr =
          reinterpret_cast<__hip_bfloat16 const *>(output_ptrs[0]) +
          static_cast<size_t>(xcd_id) * oproj_n_wgs_per_xcd *
              OPROJ_OUTPUT_PER_WG;
      gang_linear_mxfp4_res_bias_rmsnorm_topk_kernel<QKV_BATCH_SIZE,
                                                     OPROJ_OUTPUT_PER_WG,
                                                     OPROJ_REDUCTION_SIZE,
                                                     ACTUAL_HIDDEN_DIM,
                                                     NUM_EXPERTS,
                                                     TOPK_K>(
          output_ptrs[4],
          input_ptrs[9],
          oproj_residual_ptr,
          input_ptrs[10],
          input_ptrs[11],
          input_ptrs[12],
          input_ptrs[13],
          input_ptrs[14],
          input_ptrs[15],
          input_ptrs[16],
          output_ptrs[5],
          output_ptrs[6],
          output_ptrs[7],
          output_ptrs[8],
          num_active_tokens,
          oproj_n_wgs_per_xcd,
          oproj_output_stride,
          router_tile_n,
          total_oproj_tiles,
          total_topk_tiles,
          oproj_topk_tiles_per_xcd,
          oproj_tile_idx,
          routing_ready,
          // layer_epoch: the O-proj hierarchical barrier's release target.
          // Same layer-derived value as the other three counters in this file
          // -- passing it explicitly is what lets that barrier stop
          // snapshotting "current value + 1", which races with the releaser.
          qkv_epoch_expected,
          // ts_base: the O-proj/TopK kernel's optional per-sub-op timestamp
          // sink. Nothing here reads those slots -- the [FUSED_PHASE] printf
          // below derives every number from _fused_t0.._fused_t4, which are
          // taken in this function. This used to name a _ts_base that was
          // never declared, so MPK_DEVICE_TIMING=1 did not compile.
          nullptr,
#ifdef MPK_ATTN_SLICE_RELEASE
          // The kernel does its own per-wave slice wait; the Phase 6 gate
          // above skipped this worker entirely.
          attn_release
#else
          nullptr
#endif
      );
    }
  }
  MPK_PHASE_MARK(_pslot_w, 6);

  // ══════════════════════════════════════════════════════════════════
  // Phase 7a': O-proj barrier for the workers that skipped Phase 7
  // ══════════════════════════════════════════════════════════════════
  // `does_oproj` above is 23 workers at this model (184 O-proj weight groups /
  // 8 XCDs), while the dispatch is 31 workers/XCD. The other 8 never enter
  // the Phase 7 kernel at all -- so they never reach its O-proj hierarchical
  // barrier -- and then draw MoE tiles in Phase 8 that read rmsnorm_out_moe.
  // Default mapping is xcd_rank < 23; MPK_OPROJ_SWAP_ATTN moves the skippers
  // to ranks 0-7.
  //
  // Phase 7b below does not cover them. It gates on routing_ready, which
  // publishes active_expert_ids / routing_indices; nothing in that release
  // orders the *normed row* those workers are about to consume, and the
  // `buffer_inv` after it invalidates vL1 only, not the per-XCD L2 the plain
  // global_store under MPK_ONE_NORM_WRITER lands in.
  //
  // Measured, 2 layers at bs=2, against an offline MXFP4+SwiGLU oracle with
  // the error normalized by the per-slot scale: the corrupt W13 weight groups
  // are exactly the tiles owned by xcd_rank >= 23, in three separate
  // geometries --
  //
  //   workers/XCD 31 -> skippers 23..30 -> corrupt ranks 26..30 -> 33/360 bad
  //   workers/XCD 26 -> skippers 23..25 -> corrupt ranks 23..25 -> 14/360 bad
  //   workers/XCD 23 -> skippers none   -> corrupt ranks none   ->  0/360 bad
  //
  // Ranks 16..22 also skip the RMSNorm (`local_tile >= router_tile_n`, 16) and
  // are clean in every config, so the discriminator is entering this barrier,
  // not running the norm.
  //
  // bs=1 is clean because the Phase 7 kernel's per-token loop is a
  // compile-time 1 there and its closing __syncthreads compiles out, so there
  // is no partially written multi-token row to catch. The corrupt reads are
  // uncorrelated noise that scores *worse* than all-zeros, which is what a
  // half-filled row looks like -- not a coherent wrong input.
  //
  // Arrival is unchanged: these workers have no O-proj columns to publish, so
  // they must not bump the counter (`tiles_per_xcd` arrivals per XCD is what
  // the release fires on, and an extra arrival would desynchronize the
  // modular release). They only wait.
  //
  // ── This is a PARTIAL fix; the rest of the race is still open ─────────
  //
  // Matched control in one build, gate vs MPK_NO_OPROJ_SKIP_GATE=1: 32 bad
  // weight groups against 52, with XCDs 4 and 5 going 12+8 -> 0. Real, and
  // reproducible. But XCDs 1, 6 and 7 are untouched, and the residual sits in
  // expert slots 4 and 5 -- the last two *real* slots of the 8-slot
  // MAX_ACTIVATED space at bs=2 (6 real, 2 padding).
  //
  // The reason this cannot be the whole fix: the barrier polled here releases
  // *before* the RMSNorm that writes rmsnorm_out_moe, which runs afterwards in
  // the Phase 7 kernel's own Phase 3. So it orders the O-proj columns, not the
  // normed row. What makes the residual genuinely puzzling is that Phase 7b
  // below already polls routing_ready, which the router workers publish only
  // after draining their norm stores -- if that release were sufficient there
  // would be no residual at all, so something about it does not order the norm
  // for these consumers.
  //
  // rmsnorm_out_moe is *correct* in the final dump (max err 0.0039 against a
  // 0.35 scale, zero bad elements) while swiglu_out is wrong, so the MoE reads
  // it transiently before it settles. The buffer is fine; the ordering is not.
  //
  // Counts move run to run (33/22/32/32 across reps at the shipped geometry),
  // so do not size a change here off a single run -- always rebuild with the
  // ablation flag and compare inside one build.
#ifndef MPK_NO_OPROJ_SKIP_GATE
  if (!does_oproj) {
    int *oproj_hier = static_cast<int *>(input_ptrs[16]);
    while (MPK_LD_GATE2(&oproj_hier[xcd_id * 16]) < qkv_epoch_expected) {
#ifndef MPK_OPROJ_SKIP_GATE_BUSY_POLL
      __builtin_amdgcn_s_sleep(1);
#endif
    }
  }
#endif

  // ══════════════════════════════════════════════════════════════════
  // Phase 7b: wait for TopK (XCD-local release flag)
  // ══════════════════════════════════════════════════════════════════
  // Cross-workgroup sync: the last OProj workgroup runs TopK and
  // signals routing_ready. Other workgroups poll here until TopK
  // results are globally visible.
  // All threads poll independently — eliminates __syncthreads overhead.
  MPK_TW_SUB(75, routing_expected);
  MPK_WS_PHASE(75, qkv_epoch_expected, xcd_id);
  {
#ifdef MPK_EARLY_ROUTING
    MPK_WS_WAIT_BEGIN(75, routing_expected);
    int _obs;
    int _spins = 0;
    unsigned long long _rec = 0;
#ifdef MPK_NARROW_GATE_POLL
    if (tid == 0)
#endif
      do {
        _rec = ld_sys_u64(&routing_ready[(1 + xcd_id) * 16 + 2]);
        _obs = (int)_rec;
        MPK_WS_WAIT_TICK(_obs, _spins);
        _spins++;
      } while (_obs < routing_expected);
#ifdef MPK_NARROW_GATE_POLL
    __syncthreads();
    _rec = ld_sys_u64(&routing_ready[(1 + xcd_id) * 16 + 2]);
#endif
    routed_expert0 = (int)(_rec >> 32);
#else
    int *my_release = &routing_ready[(1 + xcd_id) * 16];
    MPK_WS_WAIT_BEGIN(75, routing_expected);
    int _obs;
    int _spins = 0;
#ifdef MPK_NARROW_GATE_POLL
    // Same one-poller-per-block argument as the Phase 6 gate above; see there.
    // This site has no rendezvous of its own -- the comment above the phase
    // says so -- so the narrowing has to supply one.
    if (tid == 0)
#endif
      while ((_obs = MPK_LD_GATE(my_release)) < routing_expected) {
        MPK_WS_WAIT_TICK(_obs, _spins);
        _spins++;
        __builtin_amdgcn_s_sleep(1);
      }
#ifdef MPK_NARROW_GATE_POLL
    __syncthreads();
#endif
#endif
  }
  // Cross-XCD ACQUIRE for the routing data. Bare `buffer_inv`, deliberately.
  //
  // Read the gfx950 rule first, because it is the opposite of what the
  // mnemonic suggests: an *unscoped* buffer_inv invalidates nothing. It is an
  // architectural NOP. `sc1` invalidates L2; `sc0 sc1` is system scope.
  // The disassembly confirms this is the understood rule elsewhere in the
  // kernel -- 376 of the 396 buffer_inv sites in the built binary already
  // carry `sc0 sc1`; this one was among the few that did not.
  //
  // So this instruction does nothing, and that is fine *here*: everything the
  // poll gates -- active_expert_ids / routing_indices / topk_weight -- is
  // written st_wt (sc0 sc1) straight to HBM, bypassing L2, and needs no
  // invalidate to be seen.
  //
  // What is NOT fine, and was the bs>1 MoE W13 corruption, is that Phase 8
  // also reads rmsnorm_out_moe, which under MPK_ONE_NORM_WRITER is published
  // by one router worker *per XCD* with a plain global_store into that XCD's
  // non-coherent L2. That line needs a real acquire, and this NOP was standing
  // in for one. What that line needs, and what the branch below gives it, is
  // an `sc1` invalidate. Applying it to every worker here is correct -- it
  // takes the bs=2 two-layer dump from 32 bad weight groups to 0/360 -- but
  // costs +0.259 ms/token (2.196 vs 1.937 at ctx 512), because dropping all of
  // L2 also throws away the streamed MXFP4 expert weights Phase 8 is about to
  // read. So it is applied only to the workers that need it.
  //
  // Which workers those are is not a guess. The corrupt tiles were exactly the
  // ranks at or past oproj_topk_tiles_per_xcd -- {23..28} at the shipped
  // geometry -- the same set the Phase 7a' gate above singles out, because
  // those are the workers that never entered Phase 7 and so never touched the
  // norm row on their own die. Every other rank read it while producing it.
  //
  // A cheaper form was tried and does not work: passing NT_LOAD to the W13
  // activation gather. `__builtin_nontemporal_load` emits the `nt` cache-
  // replacement hint, not a coherence scope, and left 30 of 32 groups bad.
  //
  // How the defect was localized, since the shape is worth recognizing again:
  // it hit 100% of the affected weight groups on XCDs 1, 6 and 7 and 0% on the
  // other five, was bit-identical under MPK_MOE_ENTRY_DELAY=2000 (2000 x
  // s_sleep(127) before Phase 8), and was unchanged by making the producer's
  // store write-through. A defect that survives an unbounded consumer delay
  // and a write-through producer, while staying deterministic per XCD, is a
  // consumer-side stale line -- not a race, and not a producer visibility gap.
  //
  // The acquire rmsnorm_out_moe needs is NO LONGER HERE. It moved to the top
  // of the layer, as a single `buffer_inv sc0 sc1` under QKV_BATCH_SIZE > 1.
  // See the long note at that site for why the layer-top form is preferred to
  // scoping this one.
  //
  // The scoped form that used to be here was:
  //
  //   if (QKV_BATCH_SIZE > 1 && xcd_rank >= oproj_topk_tiles_per_xcd)
  //     buffer_inv sc1;   else   buffer_inv;
  //
  // measuring 1.940 ms/token against a 1.937 pre-fix baseline, with the three
  // arms at ctx 512 being: unconditional flush at this site 2.196, selective
  // by rank 1.997, selective + batch-gated 1.940. It was correct (0/360 at
  // bs=2) but only at the site that happened to reproduce.
  asm volatile("buffer_inv" ::: "memory");

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  unsigned long long _fused_t3 = __builtin_amdgcn_s_memrealtime();
#endif
  MPK_PHASE_MARK(_pslot_w, 7);

  // ══════════════════════════════════════════════════════════════════
  // Phase 8: MoE (W13+SwiGLU+W2)
  // ══════════════════════════════════════════════════════════════════
#ifdef MPK_MOE_ENTRY_DELAY
  // Diagnostic only: stall every MoE worker before it reads rmsnorm_out_moe.
  // Separates "reads too early" from "reads the wrong thing". If the bs>1 W13
  // corruption is a race, a delay long enough to let every producer finish
  // makes it go away; if the corrupt tiles survive an unbounded wait, no
  // amount of ordering will fix them and the defect is in what is read, not
  // when. Costs a full stall on the hot path -- never ship this on.
  for (int _d = 0; _d < MPK_MOE_ENTRY_DELAY; _d++) {
    __builtin_amdgcn_s_sleep(127);
  }
#endif
#if defined(MPK_MOE_XCD_PAIR) && !defined(MPK_EARLY_ROUTING)
#error "MPK_MOE_XCD_PAIR requires MPK_EARLY_ROUTING (Phase 7b u64 wait)"
#endif
#ifdef MPK_MOE_XCD_PAIR
  // Two packed tiles per rank 0..22: W13 (bit7=0) then W2 (bit7=1). Ranks
  // 23..30 have no pair-map work; they still join Phase 9.
  constexpr int kMoePairRanks = 23;
  int const moe_begin = 0;
  int const moe_end = (xcd_rank < kMoePairRanks) ? 2 : 0;
  int const moe_step = 1;
#else
  int const moe_begin = xcd_rank;
  int const moe_end = moe_total_tiles_per_xcd;
  int const moe_step = workers_per_xcd;
#endif
  for (int moe_i = moe_begin; moe_i < moe_end; moe_i += moe_step) {
#ifdef MPK_MOE_XCD_PAIR
    int const moe_t =
        xcd_rank | (moe_i << 7) | ((routed_expert0 + 1) << 8);
#else
    int const moe_t = moe_i;
#endif
    MPK_TW_SUB(80, moe_t);
    MPK_WS_PHASE(80, qkv_epoch_expected, xcd_id);
    gang_moe_fused_mxfp4_kernel_mi300<QKV_BATCH_SIZE,
                                      MOE_INTERMEDIATE_SIZE,
                                      MOE_HIDDEN_SIZE,
                                      NUM_EXPERTS,
                                      TOPK_K,
                                      MOE_W13_OUTPUT_PER_WG,
                                      MOE_W2_OUTPUT_PER_WG>(input_ptrs[12],
                                                            input_ptrs[17],
                                                            input_ptrs[18],
                                                            output_ptrs[7],
                                                            output_ptrs[8],
                                                            input_ptrs[19],
                                                            input_ptrs[20],
                                                            output_ptrs[9],
                                                            input_ptrs[22],
                                                            output_ptrs[10],
                                                            input_ptrs[21],
#if defined(MPK_EARLY_ROUTING) && !defined(MPK_MOE_XCD_PAIR)
                                                            moe_t |
                                                                ((routed_expert0 +
                                                                  1)
                                                                 << 8)
#else
                                                            moe_t
#endif
#ifdef MPK_MOE_XCD_STRIPE_LAYER
                                                            ,
                                                            layer_counter
#endif
#ifdef MPK_EARLY_ROUTING
                                                            ,
                                                            routing_ready,
                                                            routing_expected
#endif
    );
  }
  MPK_PHASE_MARK(_pslot_w, 8);

  // ══════════════════════════════════════════════════════════════════
  // Phase 9: Layer-boundary GLOBAL barrier
  //
  // The next layer's QKV prologue (ResAddF32) reads moe_workspace_f32 -- the
  // buffer this layer's W2 epilogue just wrote. moe_ws_layout.cuh guarantees
  // every (token, slot, hidden) element has exactly one writer, so there is no
  // accumulator race *within* a layer. What was missing is the guarantee that
  // the writer has run *at all* before the next layer reads.
  //
  // Nothing provided it. All 36 layers run inside one persistent dispatch (the
  // `ml` loop in persistent_kernel.cuh), and the only thing between them is:
  //
  //     if (threadIdx.x == 0 && block_xcd_local_rank == 0) threadfence_gpu();
  //     __syncthreads();
  //
  // Both are per-workgroup. `__syncthreads` synchronizes the 256 threads of
  // *this* block and says nothing about the other 239 workers, and
  // `threadfence_gpu` is a memory *ordering* fence, not a rendezvous -- it
  // makes this worker's prior writes visible if someone reads them later, but
  // it never makes anyone wait. So a workgroup that finished its MoE tiles
  // could roll straight into layer N+1 and read workspace slots belonging to
  // tokens whose W2 tiles were still executing on another XCD.
  //
  // The result is not a hang and not garbage: the reader gets the *previous*
  // layer's value for those elements (the buffer is deliberately never zeroed
  // -- coverage is total, so nothing clears it), which is a plausible-looking
  // float. Which elements lose the race depends on inter-XCD timing, so it
  // varies run to run. That is precisely the observed signature: layer 0's K/V
  // is bit-identical across runs because on the first layer the workspace has
  // just been zeroed and there is no prior value to read, while layer 1 -- the
  // first layer with a real predecessor -- diverges at decode step 0.
  //
  // The barrier is a two-level rendezvous over the same counter block every
  // other phase here uses: each XCD elects one worker to publish its arrival,
  // then everyone waits for all 8 XCDs to reach this layer.
  //
  // The release target is a snapshot-free function of task_layer_idx, which
  // persistent_kernel.cuh computes as (pc_iter - 1) * num_layers + ml --
  // monotonic across the whole run and identical on every worker, so there is
  // nothing to race and a relaunch cannot wait on a value already in the past.
  //
  // Cost is one cross-XCD rendezvous per layer. That is the price of the
  // dependency actually being there; the previous code was fast because it
  // simply did not wait.
#ifndef MPK_NO_LAYER_BARRIER
  {
    // Who joins the release wait. Default: everyone, a full gang rendezvous.
    // Under MPK_W2_CONSUMER_GATE: only the workers that read
    // moe_workspace_f32 in the next layer's Phase 1, which is the same
    // `qkv_attn_rank < total_qkv_tiles_per_xcd` set that guards the reader
    // itself (qkv_attn_rank is fixed for the life of the task, so this worker
    // is the one that will do that read). Everyone still *arrives* either way
    // -- the arrival tree is what publishes the release, so narrowing it would
    // deadlock; this narrows only the wait.
#ifdef MPK_W2_CONSUMER_GATE
    bool const MPK_LAYER_GATE_JOINS = qkv_does_qkv;
#else
    bool const MPK_LAYER_GATE_JOINS = true;
#endif

    // Two-level rendezvous. Slot layout, one 16-int line each:
    //   layer_local[x]   : arrivals from XCD x's own workers_per_xcd workers
    //   layer_global[0]  : one arrival per XCD, bumped by that XCD's last
    //   layer_release[x] : release flag for XCD x, broadcast by the last XCD
    //
    // The tree is the whole point. A flat barrier -- 240 workgroups each
    // bumping one shared counter and then polling eight remote lines -- is
    // correct but measures ~1.0 ms/iter over 36 layers on this machine,
    // because every one of those 240 atomics is a cross-die read-modify-write
    // serialising on a single cache line, and the poll adds 240 waves x 8
    // cache-bypassing loads of lines homed on other dies. Phase 6's barrier
    // is cheap precisely because it only ever puts 8 * NUM_REQS atomics on
    // its shared line. This gets to the same 8 by aggregating per-XCD first,
    // and borrows Phase 6's arrive-then-broadcast release so no worker polls
    // a line owned by another die.
    int *layer_local =
        oproj_counters_base + FULL_LAYER_LAYER_BARRIER_SLOT(NUM_REQS);
    int *layer_global = layer_local + 8 * 16;
    int *layer_release = layer_global + 16;

    // ── Who *arrives* ────────────────────────────────────────────────────
    //
    // Default: all workers_per_xcd. The barrier exists to order this layer's
    // W2 stores against the next layer's read of moe_workspace_f32, and only
    // the workers that ran a W2 tile have any such store -- so the rest are
    // reporting the retirement of nothing, while the release cannot fire
    // until they do. At the shipped geometry that is 8 of 31 workers per XCD
    // whose arrival is pure latency: moe_total_tiles_per_xcd is 53, the loop
    // strides by workers_per_xcd, so ranks 0..21 get a second tile (global
    // tile >= TOTAL_W13, i.e. a W2 tile) and ranks 22..30 get only their
    // first, a W13 tile.
    //
    // MPK_W2_ONLY_ARRIVE narrows the arrival to exactly those producers,
    // guarding on `xcd_rank < moe_total_tiles_per_xcd - workers_per_xcd`.
    //
    // The non-producers stay ordered for the same reason the non-consumers
    // do under MPK_W2_CONSUMER_GATE: they cannot outrun the next layer's
    // Phase 6, whose release requires every XCD's attention chunks, which
    // requires QKV, which requires this release. What they lose is only the
    // right to *delay* it.
    //
    // The count must be computed exactly as Phase 8's loop bound is, or the
    // local counter's modulus stops being exact and the release never fires.
    //
    // `moe_total_tiles_per_xcd - workers_per_xcd` counts the ranks that get a
    // SECOND tile, which is a producer count only while the loop deals at most
    // two tiles per rank. Past that it exceeds the worker count and becomes a
    // modulus no counter can reach: at bs=4 the tile space is 212 per XCD
    // against 31 workers, so the unclamped form asks for 181 arrivals from a
    // counter that advances 31 per layer, and every layer gate deadlocks. The
    // clamp is also the semantically right answer -- once there are at least
    // as many W2 tiles as workers, every worker is a producer and there is
    // nobody left to exclude, so this degenerates to the default arrival.
#ifdef MPK_MOE_XCD_PAIR
    // Pair map: ranks 0..22 each produce one W2 tile (23 groups × 2 XCDs).
    int const n_w2_workers_per_xcd = 23;
#else
    int const n_w2_workers_per_xcd =
        moe_total_tiles_per_xcd > workers_per_xcd
            ? moe_total_tiles_per_xcd - workers_per_xcd
            : moe_total_tiles_per_xcd;
#endif
#ifdef MPK_W2_ONLY_ARRIVE
    // ── Every waiter must also arrive ────────────────────────────────────
    //
    // `s_layer_rel_prev` is written only on the arrival path, so a worker
    // that *waits* below without *arriving* still holds the -1 seed. -1 is
    // beaten by the release already sitting in the line -- last layer's --
    // so its gate clears immediately and it runs the next layer's Phase 1
    // against a moe_workspace_f32 this layer's W2 has not published yet.
    // Silent, run-to-run varying wrong numerics, not a hang.
    //
    // The seed comment above names one configuration that trips this
    // (MPK_W2_CONSUMER_GATE=0, where JOINS is unconditionally true while
    // only the W2 producers arrive) and calls it a hang; the -1 seed
    // converted it into this race. MPK_ROTATE_QKV_ATTN_RANKS trips it too,
    // and worse: it moves the waiter set to a cyclic block of ranks chosen
    // *because* they have no Phase 7 work, and those are very nearly the
    // same ranks that get no second MoE tile and therefore do not arrive.
    // Measured at the shipped geometry, rotation on: three runs, three
    // different completions, against a control that is bit-identical over
    // eight.
    //
    // So arrive on the union of "W2 producer" and "gate waiter" rather than
    // on the producers alone, and move the modulus with it. The waiter set
    // is a cyclic block, so the union is a prefix plus at most two pieces
    // and its size is closed-form -- it has to be exact, or the local
    // counter's modulus stops firing and the release never comes.
    int const n_w2_arrivers = n_w2_workers_per_xcd < workers_per_xcd
                                  ? n_w2_workers_per_xcd
                                  : workers_per_xcd;
#if !defined(MPK_W2_CONSUMER_GATE)
    // JOINS is `true` here, so the union is everybody.
    int const arrivers_per_xcd = workers_per_xcd;
    bool const MPK_LAYER_GATE_ARRIVES = true;
#elif defined(MPK_ROTATE_QKV_ATTN_RANKS)
    // qkv_attn_rank = (r + qkv_rot) % W, so the waiters
    // `qkv_attn_rank < total_qkv_tiles_per_xcd` are exactly
    // r in [W - qkv_rot, W - qkv_rot + T) taken mod W.
    int const wait_b = (workers_per_xcd - qkv_rot) % workers_per_xcd;
    int const wait_n = qkv_work_slots < workers_per_xcd
                           ? qkv_work_slots
                           : workers_per_xcd;
    int const wait_hi = wait_b + wait_n; // may run past workers_per_xcd
    // Part of the block at or above the producer prefix, before the wrap...
    int const seg_lo = wait_b > n_w2_arrivers ? wait_b : n_w2_arrivers;
    int const seg_hi = wait_hi < workers_per_xcd ? wait_hi : workers_per_xcd;
    int const seg_head = seg_hi > seg_lo ? seg_hi - seg_lo : 0;
    // ...and the wrapped tail [0, wrap), likewise only where it is not
    // already counted by the prefix.
    int const wrap = wait_hi > workers_per_xcd ? wait_hi - workers_per_xcd : 0;
    int const seg_tail = wrap > n_w2_arrivers ? wrap - n_w2_arrivers : 0;
    int const arrivers_per_xcd = n_w2_arrivers + seg_head + seg_tail;
    bool const MPK_LAYER_GATE_ARRIVES =
        (xcd_rank < n_w2_arrivers) || MPK_LAYER_GATE_JOINS;
#else
#ifdef MPK_QKV_KSPLIT
    // Waiters are [0, T) U [W-T, W). Union with W2 producers [0, n_w2).
    // At T=10, W=31, n_w2=22 this is 31: every rank arrives. The last
    // arriver is still a W2 producer; the extra ranks arrive early.
    int const extra_hi = qkv_k1_base >= n_w2_arrivers
                             ? (workers_per_xcd - qkv_k1_base)
                             : (workers_per_xcd - n_w2_arrivers);
    int const arrivers_per_xcd = n_w2_arrivers + extra_hi;
    bool const MPK_LAYER_GATE_ARRIVES =
        (xcd_rank < n_w2_arrivers) || MPK_LAYER_GATE_JOINS;
#else
    // Unrotated: the waiters are the prefix [0, T), which the producer
    // prefix already covers at every shipped geometry.
    int const n_waiters_unrot = qkv_work_slots < workers_per_xcd
                                    ? qkv_work_slots
                                    : workers_per_xcd;
    int const arrivers_per_xcd =
        n_w2_arrivers > n_waiters_unrot ? n_w2_arrivers : n_waiters_unrot;
    bool const MPK_LAYER_GATE_ARRIVES = (xcd_rank < arrivers_per_xcd);
#endif
#endif
#else
    int const arrivers_per_xcd = workers_per_xcd;
    bool const MPK_LAYER_GATE_ARRIVES = true;
#endif
    (void)n_w2_workers_per_xcd;

    // Every worker must retire its own MoE stores before its arrival is
    // published: vmcnt is per-wave, so neither the release on the atomic nor
    // s_barrier covers the other waves. Drain, then barrier, then arrive --
    // the same ordering the other release sites in this file use.
    //
    // MPK_DRAIN_OVERLAP moves that drain to *after* the arrival, so the store
    // retirement overlaps the barrier spin instead of preceding it. It is a
    // measurement arm, not a shipping mode: see the correctness note at the
    // post-arrival site for why it may not be enabled by default.
    //
    // TESTED AND REJECTED: dropping this pair on the grounds that Phase 8
    // already ends in the same two instructions is *wrong*, and wrong in the
    // silent way. Under it one run in eight produced different generated text
    // at identical flags. The argument fails on the W13 arm: its arrival
    // block runs after Phase 8's `__syncthreads()`, and the last arrival's
    // eight-slot `st_wt_u32` release fan-out is a store issued past it -- so
    // the block is neither drained nor reconverged on that path. The
    // `goto w13_arrive` empty-slot path skips Phase 8's drain outright. The
    // pair stays unconditional.
#ifdef MPK_DRAIN_STATS
    unsigned long long _dr0 = __builtin_amdgcn_s_memrealtime();
#endif
#ifndef MPK_DRAIN_OVERLAP
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
#ifdef MPK_DRAIN_STATS
    unsigned long long _dr1 = __builtin_amdgcn_s_memrealtime();
#endif
    __syncthreads();
#ifdef MPK_DRAIN_STATS
    unsigned long long _dr2 = __builtin_amdgcn_s_memrealtime();
#endif

    // Carries "I am the releaser, publish epoch N" from lane 0 out to lanes
    // 1..7 of the same wave; see the fan-out below. Zero means not the
    // releaser, which is unambiguous since a real epoch is >= 8.
    int lean_rel_epoch = 0;

    // Thread-local, not __shared__, and that is the whole point. Every write
    // (the seed below, the snapshot, the MPK_LEAN_ARRIVE derivation) and the
    // only read (the gate poll) is guarded by `tid == 0`, so no thread but
    // lane 0 of wave 0 ever touches it -- LDS was buying nothing.
    //
    // What it cost is visible in the asm. As a __shared__ int the compiler
    // cannot hoist it out of the spin, so the poll loop was
    //
    //     global_load_dword v3, v[0:1], off sc0 sc1
    //     s_waitcnt vmcnt(0)
    //     ds_read_b32 v4, v2 offset:336      <-- reloaded every iteration
    //     s_waitcnt lgkmcnt(0)               <-- and serialized behind it
    //     v_cmp_le_i32 vcc, v3, v4
    //
    // an LDS round trip appended to each poll, after the vmcnt wait, on the
    // one thread whose latency to observe the release is the layer boundary
    // for its entire workgroup. Holding the previous release in a register
    // instead removes the round trip: the compare is then just the load and
    // the cmp.
    //
    // The `__syncthreads()` after the seed is kept: it is a barrier the block
    // needs on this path regardless, and dropping it is a separate change.
    //
    // MPK_GATE_PREV_LDS=1 restores the __shared__ form for A/B.
#ifdef MPK_GATE_PREV_LDS
    __shared__ int s_layer_rel_prev;
    if (tid == 0) {
      s_layer_rel_prev = -1;
    }
#else
    int s_layer_rel_prev = -1;
#endif
    // Seeded for the workers that do NOT arrive. `s_layer_rel_prev` is written
    // only under `tid == 0 && MPK_LAYER_GATE_ARRIVES`, but the wait below is
    // joined by every worker for which MPK_LAYER_GATE_JOINS holds. Those two
    // sets are equal in the two shipped configurations (both flags on, or both
    // off), which is why this never bit -- but MPK_W2_ONLY_ARRIVE=1 with
    // MPK_W2_CONSUMER_GATE=0 makes JOINS a strict superset of ARRIVES, and
    // ranks 22..30 then spin against uninitialized LDS. Whatever garbage the
    // slot holds is compared `<=` against a release value that only ever
    // increases by 8 per layer, so a large positive leaves the gate closed
    // forever: all 248 workers hang with tasks_done=0 and no error.
    //
    // -1 is below every real release value (they are 8*L, L >= 1), so under
    // MPK_GATE_PREV_LDS a non-arriving worker reads a defined value instead of
    // whatever the slot held.
    //
    // It is NOT, as this comment used to claim, "the correct semantics for a
    // worker that published nothing". A non-arriving *waiter* is beaten by the
    // release already in the line -- last layer's -- so its gate clears
    // without waiting for this layer's release at all, and it reads
    // moe_workspace_f32 in the next layer's Phase 1 before this layer's W2
    // published it. The seed turned a hang into a silent race. What makes
    // the seed harmless is the arriver set above now covering the waiter set
    // by construction, so no waiter reaches this line unseeded; see the
    // union computation at MPK_LAYER_GATE_ARRIVES for the two configurations
    // that used to break it. The seed stays as the belt to that suspenders.
    //
    // Cost is one LDS store per workgroup per layer on a path that already
    // does several.
    __syncthreads();
#ifdef MPK_DRAIN_STATS
    __shared__ unsigned long long s_dr3;
    __shared__ bool s_was_last_local;
    __shared__ bool s_was_last_global;
    if (tid == 0) {
      s_was_last_global = false;
    }
    __syncthreads();
#endif

    if (tid == 0 && MPK_LAYER_GATE_ARRIVES) {
      // Snapshot this XCD's release line *before* arriving. Layer L's release
      // is written only after all 240 workers arrive, this one included, so
      // at this instant the line still holds layer L-1's value -- and it
      // holds at least that, because this worker cleared layer L-1's release
      // to get here. So `rel_prev` is exactly layer L-1's value on every
      // worker, and "layer L has released" is exactly "the line moved".
      //
      // This is why the wait target is a snapshot rather than a function of
      // task_layer_idx: this counter block is a host allocation that is never
      // reset, while task_layer_idx restarts at 0 on every kernel launch
      // (pc_iter is __shared__). A layer-derived target would make every
      // relaunch after a warmup wait on a value already in the past and hang
      // all 240 workers. Same reasoning as the attn_global bump above.
      //
      // MPK_LEAN_ARRIVE removes this load, and the `local_expected` load
      // below, from the critical path. Both are inferable from the value the
      // arrival atomic already returns, so the arrival becomes a single
      // read-modify-write with nothing in front of it -- see the derivation
      // at the post-atomic site.
#ifndef MPK_LEAN_ARRIVE
      s_layer_rel_prev = ld_nt_s32(&layer_release[xcd_id * 16]);
#endif

      // The two arrival targets are division snapshots for the same reason,
      // and are race-free because the barrier each belongs to bounds it.
      // Division, never `(v | mask) + 1`: workers_per_xcd is 30, not a power
      // of two, so the bitwise form would release early.
      //
      // For the local counter: every worker on this XCD has cleared layer
      // L-1's release (so v >= workers_per_xcd * L) and none can reach layer
      // L+1's arrival without layer L's release, which needs this very
      // arrival (so v < workers_per_xcd * (L+1)). For the global counter the
      // same argument runs one level up, over the 8 XCDs.
#ifndef MPK_LEAN_ARRIVE
      int local_expected =
          (__atomic_load_n(&layer_local[xcd_id * 16], __ATOMIC_RELAXED) /
               arrivers_per_xcd +
           1) *
          arrivers_per_xcd;
#endif
      // XCD-private: written and read only at `[xcd_id * 16]`. This is the
      // hottest arrival in the layer -- 30 workers per XCD, 240 per GPU, every
      // layer -- and under the default `sc1` every one of them is a cross-die
      // round trip to a coherency point no other XCD reads through.
      // `layer_global` below is the level that *is* cross-XCD and keeps its
      // `sc0 sc1`, so the two-level tree still publishes exactly as before.
      //
      // The drain that makes this arrival a release is the explicit
      // `s_waitcnt vmcnt(0)` above, not the atomic's scope, so weakening the
      // atomic does not weaken the release: the XCD leader elected here still
      // executes `atom_add_release_gpu_s32(layer_global)` after every one of
      // its XCD's workers has both drained and arrived.
      int local_prev =
          MPK_XCD_LOCAL_ATOM_ADD(&layer_local[xcd_id * 16], 1);

#ifdef MPK_LEAN_ARRIVE
      // ── Everything the two removed loads carried, recovered from local_prev
      //
      // Both counters are monotonic and never reset, and the layer barrier
      // bounds them: no worker reaches layer L+1's arrival without layer L's
      // release, which needs every layer-L arrival. So when this worker
      // arrives at layer L the local counter holds
      // `(L-1) * workers_per_xcd + p` with `0 <= p < workers_per_xcd`, and
      // one integer division of the value the atomic already returned splits
      // it into both halves that were previously loaded separately:
      //
      //   p == workers_per_xcd - 1        <=>  this is the XCD's last arrival
      //   local_prev / workers_per_xcd    ==   L - 1
      //
      // The release value written for layer L is `8 * L` (the global counter
      // takes exactly one arrival per XCD per layer), so the previous
      // release -- what the wait below must beat -- is `8 * (L - 1)`. That is
      // still derived from the counter block rather than from
      // task_layer_idx, so it keeps the relaunch-safety the snapshot had:
      // task_layer_idx restarts at 0 on every dispatch, this does not.
      //
      // Net effect: the arrival is one read-modify-write with no dependent
      // load in front of it.
      int const lean_layers_done = local_prev / arrivers_per_xcd;
      s_layer_rel_prev = 8 * lean_layers_done;
      bool const lean_is_last_local =
          (local_prev - lean_layers_done * arrivers_per_xcd) ==
          arrivers_per_xcd - 1;
#ifdef MPK_DRAIN_STATS
      s_was_last_local = lean_is_last_local;
#endif
      if (lean_is_last_local) {
        int global_prev = atom_add_release_gpu_s32(layer_global, 1);
        // Same identity one level up: the counter takes 8 arrivals per layer,
        // so the last XCD is the one whose pre-increment value is 7 mod 8 and
        // the release it publishes is its post-increment value.
        int global_expected = global_prev + 1;
        bool const is_last_global = (global_prev & 7) == 7;
#else
#ifdef MPK_DRAIN_STATS
      s_was_last_local = (local_prev == local_expected - 1);
#endif
      if (local_prev == local_expected - 1) {
        // Last worker on this XCD: this XCD's MoE stores are all retired, so
        // publish one arrival on behalf of the whole die.
        int global_expected =
            (__atomic_load_n(layer_global, __ATOMIC_RELAXED) / 8 + 1) * 8;
        int global_prev = atom_add_release_gpu_s32(layer_global, 1);
        bool const is_last_global = (global_prev == global_expected - 1);
#endif
#ifdef MPK_DRAIN_STATS
        if (is_last_global) {
          atomicAdd(&g_last_xcd[xcd_id], 1ULL);
          s_was_last_global = true;
        }
#endif
        if (is_last_global) {
          // Last XCD overall: fan the release out with st_wt (write-through,
          // bypasses L2) so every worker polls a line it already owns.
          //
          // The eight stores are handed to lanes 0..7 below rather than
          // issued here, so they become one instruction instead of eight
          // serial ones from the single most critical lane on the GPU -- 239
          // workgroups are spinning on flags it has not written yet, and the
          // eighth XCD would otherwise be released seven store-issues after
          // the first. The drain that has to precede the flags still runs
          // here, on the lane that owns the stores being advertised.
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          lean_rel_epoch = global_expected;
        }
      }
#ifdef MPK_DRAIN_STATS
      s_dr3 = __builtin_amdgcn_s_memrealtime();
#endif
    }

    // Lanes 0..7 of wave 0 publish the eight release flags in one wave
    // instruction. readfirstlane rather than LDS: the arrival above runs with
    // only lane 0 active, so once the `if` closes every lane reads lane 0's
    // value from the first active lane.
    lean_rel_epoch = __builtin_amdgcn_readfirstlane(lean_rel_epoch);
    if (lean_rel_epoch != 0) {
      if (tid < 8) {
        st_wt_u32((void *)&layer_release[tid * 16], (unsigned)lean_rel_epoch);
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }

#ifdef MPK_DRAIN_OVERLAP
    // ── Drain overlapped with the barrier spin (measurement arm) ──────────
    //
    // The arrival above has already been published; this retires the stores
    // it was meant to advertise. Every thread drains, because vmcnt is
    // per-wave and the arrival speaks for all four waves of this block.
    //
    // CORRECTNESS: this is deliberately weaker than the default. The arrival
    // *is* the statement "my W2 stores are visible", so publishing it before
    // the drain lets another XCD observe the arrival, clear the barrier, and
    // read a moe_workspace_f32 slot whose store has not landed -- the exact
    // race Phase 9 exists to close. It survives in practice only because the
    // release still has to cross all 8 XCDs, which takes far longer than the
    // drain; that is a timing accident, not a guarantee. Kept as a flag so
    // the ceiling of the idea is a measured number rather than an argument.
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif

#ifdef MPK_PREFETCH_NEXT_QKV
    // ── Next layer's QKV weight DMA, issued into the barrier spin ────────
    //
    // Measured, this barrier costs mean_wait=7683 ns per worker per layer
    // (n=4.39M): 0.277 ms of the 2.49 ms iteration spent waiting on the
    // slowest XCD. The wait is a scheduling cost, not a bandwidth one, so
    // weight traffic issued here is close to free -- the memory system is
    // otherwise idle for those microseconds.
    //
    // Placement is load-bearing on both sides:
    //
    //   * AFTER the arrival block above, not before it. The last-arriving
    //     XCD's leader executes `s_waitcnt vmcnt(0)` in its release fan-out,
    //     which drains *all* of this wave's outstanding vector memory --
    //     including a prefetch issued earlier. That worker is by definition
    //     the critical path, so it would stall on its own DMA and delay the
    //     release for all 240. Issuing after the fan-out keeps it off.
    //
    //   * AFTER the `__syncthreads()` that opens this block. buffer_load_lds
    //     retires under vmcnt while MoE's ds_writes retire under lgkmcnt, so
    //     the two are not ordered against each other; a Phase 8 LDS write
    //     still in flight could land on top of the staged weights. The
    //     s_barrier inside __syncthreads drains lgkmcnt first.
    //
    // Only Phase-1 workers stage anything: the rest have no QKV tile next
    // layer and their LDS would just be dirtied for nothing. tile_idx there
    // is qkv_attn_rank, and qkv_attn_rank is fixed for the life of the task,
    // so this worker prefetches exactly the tile it will itself execute --
    // which is why the rotation has to be applied here too, not just at the
    // Phase 1 guard.
    //
    // input_ptrs[24] is null on the last layer of the iteration -- see the
    // publish site in persistent_kernel.cuh for why nothing may be staged
    // across the iteration boundary.
    if (input_ptrs[24] != nullptr &&
        qkv_does_qkv
#ifdef MPK_QKV_PF_WAVE_SPLIT
        // ── Keep the poller's wave out of the pre-gate DMA ─────────────────
        //
        // tid 0 is the only thread that spins on `layer_release` below, and it
        // is the thread whose latency the whole workgroup then waits on. Its
        // wave issues 1/4 of this prefetch: NUM_WAVES buffer_load_lds per lane,
        // all still outstanding when it enters the spin. The poll is itself a
        // vector load, so it queues behind them in the same wave's memory
        // pipeline -- the poller pays its own prefetch's latency on every
        // iteration of the loop, and the release it is looking for is seen late
        // by exactly the amount.
        //
        // Waves 1..3 have nothing to do during the spin, so they keep their
        // three quarters here; wave 0's quarter is re-issued after the gate,
        // where it costs the poller nothing. The split is clean because
        // qkv_prefetch_weights_lds partitions LDS by warp_id (warp_id * 1024)
        // and each wave's buffer_load_lds writes only its own 1 KiB block, so
        // deferring one wave defers exactly one quarter of the staged tile and
        // touches nobody else's bytes.
        && tid >= 64
#endif
    ) {
#ifdef MPK_QKV_KSPLIT
      int const qkv_k_part = qkv_does_k1 ? 1 : 0;
      int const qkv_tile =
          qkv_does_k1 ? qkv_attn_rank - qkv_k1_base : qkv_attn_rank;
      int const qkv_k_iter0 = qkv_k_part ? QKV_K_LO : 0;
      int const qkv_k_niters =
          qkv_k_part ? (QKV_MFMA_ITERS - QKV_K_LO) : QKV_K_LO;
      qkv_prefetch_weights_lds<QKV_BATCH_SIZE,
                               QKV_OUTPUT_PER_WG,
                               QKV_REDUCTION_SIZE>(
          input_ptrs[24],
          qkv_n_wgs_per_xcd,
          qkv_tile,
          qkv_k_iter0,
          qkv_k_niters);
#else
      qkv_prefetch_weights_lds<QKV_BATCH_SIZE,
                               QKV_OUTPUT_PER_WG,
                               QKV_REDUCTION_SIZE>(
          input_ptrs[24], qkv_n_wgs_per_xcd, qkv_attn_rank);
#endif
    }
#endif

    // ── MPK_W2_CONSUMER_GATE: publish-and-go ─────────────────────────────
    //
    // Under this flag the arrival tree above still runs -- every worker still
    // reports in, the last XCD still fans out `layer_release` -- but nobody
    // waits here. The wait moves to the one place that needs it: the Phase 1
    // workspace consumer of the *next* layer.
    //
    // Why the rendezvous was never the requirement. The hazard Phase 9 exists
    // to close is exactly one edge: the next layer's QKV prologue
    // (gang_resaddf32_rmsnorm_linear_mxfp4_bias_kvupd_kernel) reads
    // moe_workspace_f32, which this layer's W2 epilogue writes. That reader is
    // already guarded by `qkv_attn_rank < total_qkv_tiles_per_xcd`, so only
    // total_qkv_tiles_per_xcd of the workers_per_xcd workers on each XCD ever
    // touch the buffer. Ordering the other ~20 against a buffer they never
    // load is synchronization the dependency does not ask for, and the
    // measured rank spin cliff is that surplus made visible: ranks 0-22 spin
    // ~5.1 us, ranks 23-29 ~25.2 us.
    //
    // What keeps the non-consumers in line is Phase 6. Its attn_release poll
    // is joined by all workers_per_xcd and its release cannot be published
    // until every XCD's attention chunks have arrived, which requires QKV,
    // which requires this release. So the non-consumers are still ordered --
    // transitively, at a gate they already had to pay -- rather than at a
    // second gate of their own.
    //
    // This is the "weakened" case the note at the top of this file warns
    // about, so the layer-boundary acquire moves with the wait and returns to
    // `sc1`: the five sc1 drops in 981e124 were justified by "no worker
    // reaches layer N+1 until every worker passed Phase 9 for layer N", and
    // that premise is exactly what this flag retires. Only the consumers pay
    // the sc1 now, which is what makes it affordable.
    //
    // The wait stays here rather than moving to the top of the next layer:
    // xcd_rank is fixed for the life of the task (same property the QKV
    // prefetch relies on), so the worker that waits at the end of layer N is
    // exactly the worker that reads the workspace in layer N+1. Keeping it
    // here also keeps `s_layer_rel_prev` valid -- it is a snapshot taken in
    // this invocation, and there is no cross-invocation channel to carry a
    // release target through.
#ifdef MPK_DRAIN_STATS
    // Splits the "spin" segment. Everything between s_dr3 and here is the
    // release fan-out plus the next-layer QKV prefetch *issue*, which the
    // original window silently folded into the wait -- which is why the
    // worker that publishes the release, and therefore waits on nobody, still
    // showed ~2 us of "spin".
    unsigned long long _dr35 = __builtin_amdgcn_s_memrealtime();
#endif
    // Same split point MPK_DRAIN_STATS uses: everything before this is the
    // arrival tree plus the release fan-out and the next-layer QKV prefetch
    // *issue*; everything after is the poll proper. Only slot 9->10 is a
    // like-for-like wait span: the pre-gate half is release-publishing work,
    // not waiting, so folding it in overstates the gate.
    MPK_PHASE_MARK(_pslot_w, 9);
    if (tid == 0 && MPK_LAYER_GATE_JOINS) {
      while (MPK_LD_GATE(&layer_release[xcd_id * 16]) <= s_layer_rel_prev) {
#ifndef MPK_LAYER_GATE_BUSY_POLL
        __builtin_amdgcn_s_sleep(1);
#endif
      }
#ifndef MPK_W2_CONSUMER_GATE
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
#endif
#ifdef MPK_W2_CONSUMER_GATE
      // The agent-scope acquire fence is gone from this arm on purpose: it
      // lowers to `buffer_inv sc1`, the exact instruction below, so the two
      // together emitted a full L2 invalidate twice back to back (visible in
      // ATT as the adjacent pair at the gate's exit). The asm carries a
      // "memory" clobber, so the compiler-side ordering the fence provided is
      // unchanged; the bare asm with no fence is what this gate wants.
      //
      // Layer-boundary acquire, relocated from the top of the layer and
      // promoted back to `sc1` (vL1 + L2). W2 writes the workspace
      // write-through from another XCD and MI350 L2 is not coherent across
      // XCDs, so a line this XCD still holds from last layer's read of the
      // same address would otherwise be served stale. The default build gets
      // away with a vL1-only `buffer_inv` because the full rendezvous
      // guarantees every producer retired before any consumer proceeds; this
      // arm retires that guarantee, so the L2 invalidate comes back. Only the
      // gate-joining workers execute it.
      asm volatile("buffer_inv sc1" ::: "memory");
#endif
#ifdef MPK_DRAIN_STATS
      unsigned long long _dr4 = __builtin_amdgcn_s_memrealtime();
      // s_memrealtime ticks at 100 MHz -> 10 ns per tick.
      unsigned long long drain = (_dr1 - _dr0) * 10;
      unsigned long long sync = (_dr2 - _dr1) * 10;
      unsigned long long arrive = (s_dr3 - _dr2) * 10;
      unsigned long long spin = (_dr4 - s_dr3) * 10;
      unsigned long long fanpf = (_dr35 - s_dr3) * 10;
      unsigned long long poll = (_dr4 - _dr35) * 10;
      atomicAdd(&g_fanpf_sum, fanpf);
      atomicAdd(&g_poll_sum, poll);
      if (s_was_last_global) {
        atomicAdd(&g_poll_lastglobal, poll);
      }
      unsigned long long n = atomicAdd(&g_drain_n, 1ULL);
      atomicAdd(&g_drain_sum, drain);
      atomicAdd(&g_sync_sum, sync);
      atomicAdd(&g_arrive_sum, arrive);
      atomicAdd(&g_spin_sum, spin);
      atomicAdd(&g_spin_xcd[xcd_id], spin);
      atomicAdd(&g_n_xcd[xcd_id], 1ULL);
      if (xcd_rank < 64) {
        atomicAdd(&g_spin_rank[xcd_rank], spin);
        atomicAdd(&g_n_rank[xcd_rank], 1ULL);
      }
      if (s_was_last_local) {
        atomicAdd(&g_spin_lastlocal, spin);
        atomicAdd(&g_n_lastlocal, 1ULL);
      }
      if (s_was_last_global) {
        atomicAdd(&g_spin_lastglobal, spin);
        atomicAdd(&g_n_lastglobal, 1ULL);
      }
      if (n % 100000 == 0 && n > 0) {
        printf("[DRAIN] n=%llu drain=%llu sync=%llu arrive=%llu spin=%llu "
               "tot=%llu\n",
               n,
               g_drain_sum / n,
               g_sync_sum / n,
               g_arrive_sum / n,
               g_spin_sum / n,
               (g_drain_sum + g_sync_sum + g_arrive_sum + g_spin_sum) / n);
        printf("[XCDSPIN] %llu %llu %llu %llu %llu %llu %llu %llu | last "
               "%llu %llu %llu %llu %llu %llu %llu %llu\n",
               g_spin_xcd[0] / (g_n_xcd[0] ? g_n_xcd[0] : 1),
               g_spin_xcd[1] / (g_n_xcd[1] ? g_n_xcd[1] : 1),
               g_spin_xcd[2] / (g_n_xcd[2] ? g_n_xcd[2] : 1),
               g_spin_xcd[3] / (g_n_xcd[3] ? g_n_xcd[3] : 1),
               g_spin_xcd[4] / (g_n_xcd[4] ? g_n_xcd[4] : 1),
               g_spin_xcd[5] / (g_n_xcd[5] ? g_n_xcd[5] : 1),
               g_spin_xcd[6] / (g_n_xcd[6] ? g_n_xcd[6] : 1),
               g_spin_xcd[7] / (g_n_xcd[7] ? g_n_xcd[7] : 1),
               g_last_xcd[0],
               g_last_xcd[1],
               g_last_xcd[2],
               g_last_xcd[3],
               g_last_xcd[4],
               g_last_xcd[5],
               g_last_xcd[6],
               g_last_xcd[7]);
        printf("[INTER] lastlocal_spin=%llu n=%llu | lastglobal_spin=%llu "
               "n=%llu\n",
               g_spin_lastlocal / (g_n_lastlocal ? g_n_lastlocal : 1),
               g_n_lastlocal,
               g_spin_lastglobal / (g_n_lastglobal ? g_n_lastglobal : 1),
               g_n_lastglobal);
        printf("[SPLIT] fanout+prefetch=%llu poll=%llu | lastglobal_poll=%llu\n",
               g_fanpf_sum / n,
               g_poll_sum / n,
               g_poll_lastglobal / (g_n_lastglobal ? g_n_lastglobal : 1));
        printf("[MLPRO] prologue=%llu copy=%llu n=%llu\n",
               g_mlprologue_sum / (g_mlprologue_n ? g_mlprologue_n : 1),
               g_mlcopy_sum / (g_mlcopy_n ? g_mlcopy_n : 1),
               g_mlprologue_n);
        for (int r0 = 0; r0 < 30; r0 += 10) {
          printf("[RANK%02d] %llu %llu %llu %llu %llu %llu %llu %llu %llu "
                 "%llu\n",
                 r0,
                 g_spin_rank[r0 + 0] / (g_n_rank[r0 + 0] ? g_n_rank[r0 + 0] : 1),
                 g_spin_rank[r0 + 1] / (g_n_rank[r0 + 1] ? g_n_rank[r0 + 1] : 1),
                 g_spin_rank[r0 + 2] / (g_n_rank[r0 + 2] ? g_n_rank[r0 + 2] : 1),
                 g_spin_rank[r0 + 3] / (g_n_rank[r0 + 3] ? g_n_rank[r0 + 3] : 1),
                 g_spin_rank[r0 + 4] / (g_n_rank[r0 + 4] ? g_n_rank[r0 + 4] : 1),
                 g_spin_rank[r0 + 5] / (g_n_rank[r0 + 5] ? g_n_rank[r0 + 5] : 1),
                 g_spin_rank[r0 + 6] / (g_n_rank[r0 + 6] ? g_n_rank[r0 + 6] : 1),
                 g_spin_rank[r0 + 7] / (g_n_rank[r0 + 7] ? g_n_rank[r0 + 7] : 1),
                 g_spin_rank[r0 + 8] / (g_n_rank[r0 + 8] ? g_n_rank[r0 + 8] : 1),
                 g_spin_rank[r0 + 9] / (g_n_rank[r0 + 9] ? g_n_rank[r0 + 9] : 1));
        }
      }
#endif
    }
    // Broadcasts the gate to the other 255 threads: only tid 0 waited (and,
    // under MPK_W2_CONSUMER_GATE, only on gate-joining workers), so this is
    // what makes the release visible to the whole block.
    __syncthreads();
    MPK_PHASE_MARK(_pslot_w, 10);

#if defined(MPK_PREFETCH_NEXT_QKV) && defined(MPK_QKV_PF_WAVE_SPLIT)
    // Wave 0's deferred quarter of the staged tile. It has to be issued here,
    // past the gate, because that is the whole point of deferring it: before
    // the gate it would sit in tid 0's memory pipeline ahead of the poll.
    //
    // Issuing it *after* the __syncthreads rather than inside the `tid == 0`
    // block is required -- buffer_load_lds is per lane, so all 64 lanes of
    // wave 0 must run it to fill the 1 KiB block, and only lane 0 was inside.
    //
    // It is still a prefetch: the consumer in the next layer drains with
    // `s_waitcnt vmcnt(0)` before its first ds_read, so the DMA has the whole
    // rest of this layer's epilogue plus the next layer's prologue to land.
    if (input_ptrs[24] != nullptr && qkv_does_qkv &&
        tid < 64) {
#ifdef MPK_QKV_KSPLIT
      int const qkv_k_part = qkv_does_k1 ? 1 : 0;
      int const qkv_tile =
          qkv_does_k1 ? qkv_attn_rank - qkv_k1_base : qkv_attn_rank;
      int const qkv_k_iter0 = qkv_k_part ? QKV_K_LO : 0;
      int const qkv_k_niters =
          qkv_k_part ? (QKV_MFMA_ITERS - QKV_K_LO) : QKV_K_LO;
      qkv_prefetch_weights_lds<QKV_BATCH_SIZE,
                               QKV_OUTPUT_PER_WG,
                               QKV_REDUCTION_SIZE>(
          input_ptrs[24],
          qkv_n_wgs_per_xcd,
          qkv_tile,
          qkv_k_iter0,
          qkv_k_niters);
#else
      qkv_prefetch_weights_lds<QKV_BATCH_SIZE,
                               QKV_OUTPUT_PER_WG,
                               QKV_REDUCTION_SIZE>(
          input_ptrs[24], qkv_n_wgs_per_xcd, qkv_attn_rank);
#endif
    }
#endif
    // No invalidate here: in the default build the task body re-entered for
    // the next layer opens with the layer-boundary acquire near the top of
    // this function, which is the same instruction against the same data.
    // Issuing it twice per layer costs a full L2 invalidate for nothing.
    // Under MPK_W2_CONSUMER_GATE that top-of-layer site is compiled out and
    // the acquire is the `buffer_inv sc1` above, inside the gate.
  }
#endif

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  __syncthreads();
  if (tid == 0 && xcd_rank == 0) {
    unsigned long long _fused_t4 = __builtin_amdgcn_s_memrealtime();
    double p1_5 = (double)(_fused_t1 - _fused_t0) * 10.0 / 1000.0;
    double p6 = (double)(_fused_t2 - _fused_t1) * 10.0 / 1000.0;
    double p7 = (double)(_fused_t3 - _fused_t2) * 10.0 / 1000.0;
    double p8 = (double)(_fused_t4 - _fused_t3) * 10.0 / 1000.0;
    double total = (double)(_fused_t4 - _fused_t0) * 10.0 / 1000.0;
    // Fine-grained sub-ops (xcd_rank==0 is a QKV+attn worker)
    double qkv_gemm = (_fused_t0a > 0)
                          ? (double)(_fused_t0a - _fused_t0) * 10.0 / 1000.0
                          : -1;
    double qkv_barrier = (_fused_t0b > 0)
                             ? (double)(_fused_t0b - _fused_t0a) * 10.0 / 1000.0
                             : -1;
    double attn = (_fused_t0c > 0)
                      ? (double)(_fused_t0c - _fused_t0b) * 10.0 / 1000.0
                      : -1;
    double merge_only = (_merge_done > 0)
                            ? (double)(_merge_done - _fused_t0c) * 10.0 / 1000.0
                            : -1;
    double flush_sig = (_fused_t0d > 0 && _merge_done > 0)
                           ? (double)(_fused_t0d - _merge_done) * 10.0 / 1000.0
                           : -1;
    double merge_flush = (_fused_t0d > 0)
                             ? (double)(_fused_t0d - _fused_t0c) * 10.0 / 1000.0
                             : -1;
    double wait_others =
        (_fused_t0d > 0)   ? (double)(_fused_t1 - _fused_t0d) * 10.0 / 1000.0
        : (_fused_t0c > 0) ? (double)(_fused_t1 - _fused_t0c) * 10.0 / 1000.0
                           : -1;
    printf("[FUSED_PHASE] xcd=%d qkv_attn=%.1f xcd_barrier=%.1f "
           "oproj_topk=%.1f moe=%.1f total=%.1f"
           " | qkv_gemm=%.1f qkv_bar=%.1f attn=%.1f merge=%.1f(m=%.1f+f=%.1f) "
           "wait=%.1f\n",
           xcd_id,
           p1_5,
           p6,
           p7,
           p8,
           total,
           qkv_gemm,
           qkv_barrier,
           attn,
           merge_flush,
           merge_only,
           flush_sig,
           wait_others);
  }
#endif
  MPK_PHASE_MARK(_pslot_w, 11);
  MPK_TW_SUB(90, tile_idx);
  MPK_WS_PHASE(90, qkv_epoch_expected, xcd_id);
}

} // namespace kernel
