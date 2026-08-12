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

  int xcd_id;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcd_id));

  int xcd_rank = tile_idx % workers_per_xcd;
  int tid = threadIdx.x;

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
  int const attn_req = xcd_rank / NUM_KV_CHUNKS;
  int const attn_chunk = xcd_rank % NUM_KV_CHUNKS;

  // Layer-boundary ACQUIRE. This must be `sc1` (vL1 + L2), not a plain
  // `buffer_inv` (vL1 only).
  //
  // The comment this replaces described the world before moe_ws_layout.cuh:
  // the MoE W2 epilogue accumulated into the workspace with `atomicAdd`, which
  // reaches the device-coherent point implicitly, so a consumer only had to
  // drop its own vL1 to see it. Those atomics are now plain write-through
  // stores (`st_wt_f32x4`, sc0 sc1), which bypass L2 and land in HBM. The
  // acquire side was never upgraded to match.
  //
  // The consequence is a stale read, and it is confined to exactly the shape
  // observed: the W2 tile that wrote a given (token, slot, hidden) element runs
  // on a different XCD than the QKV prologue that reads it, and MI300/MI350 L2
  // is not coherent across XCDs. If this reader's L2 still holds the line from
  // *last* layer's read of the same address, `buffer_inv` leaves it there and
  // the load is served from L2 -- returning the previous layer's value. Whether
  // a given line survives depends on inter-XCD L2 eviction timing, so it varies
  // run to run.
  //
  // That is why layer 0 is bit-identical across runs while layers 1..35 all
  // differ: on layer 0 the workspace was just zeroed and there is no prior
  // value for a stale line to hold. Layer 1 is the first layer with a real
  // predecessor, and it is the first to diverge.
  //
  // Cost is one L2 invalidate per worker per layer, which is what an acquire
  // against a write-through producer actually costs. The Phase 9 barrier
  // already documents this instruction as being here; it just was not.
  asm volatile("buffer_inv sc1" ::: "memory");

  // NOTE: the layer counter that the MoE W13->W2 barrier derives its release
  // value from is published further down, once qkv_epoch_expected is known.
  // See the LAYER_IDX_SMEM_OFF store just before Phase 1.

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  unsigned long long _fused_t0 = __builtin_amdgcn_s_memrealtime();
  unsigned long long _fused_t0a = 0, _fused_t0b = 0, _fused_t0c = 0,
                     _fused_t0d = 0, _merge_done = 0;
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
  int const qkv_epoch_participants = total_qkv_tiles_per_xcd > ATTN_PARTICIPANTS
                                         ? total_qkv_tiles_per_xcd
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
  if (xcd_rank < total_qkv_tiles_per_xcd) {
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
        xcd_rank);

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
    _fused_t0a = __builtin_amdgcn_s_memrealtime();
#endif
  } // end Phase 1: QKV GEMM

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
  if (xcd_rank < qkv_epoch_participants) {
    __shared__ int s_prev;
    if (tid == 0) {
      s_prev = atom_add_release_gpu_s32(
          &static_cast<int *>(input_ptrs[7])[xcd_id], 1);
    }
    __syncthreads();

    if ((s_prev % qkv_epoch_participants) == qkv_epoch_participants - 1) {
      // Last worker to arrive: bump epoch (no reset needed — modular check)
      if (tid == 0) {
        atom_add_release_gpu_s32(&qkv_epoch[xcd_id * 16], 1);
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
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
    __syncthreads();
    asm volatile("buffer_inv" ::: "memory");
  }

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  _fused_t0b = __builtin_amdgcn_s_memrealtime();
#endif

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
    if (xcd_rank < ATTN_PARTICIPANTS) {
      int kv_chunk_idx = attn_chunk;
      using bf16_t = __hip_bfloat16;
      void const *offset_k = reinterpret_cast<bf16_t const *>(output_ptrs[1]) +
                             static_cast<size_t>(xcd_id) * HEAD_DIM;
      void const *offset_v = reinterpret_cast<bf16_t const *>(output_ptrs[2]) +
                             static_cast<size_t>(xcd_id) * HEAD_DIM;

      // Write float32 partials to o_acc_f32 (input_ptrs[23])
      // Write LSE to lse_acc (input_ptrs[8])
      // NO sinks for per-chunk — sinks applied in merge step
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
        s_chunk_prev = atom_add_release_gpu_s32(
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
#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
        _fused_t0d = __builtin_amdgcn_s_memrealtime();
#endif
      }
    }
  }

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  unsigned long long _fused_t1 = __builtin_amdgcn_s_memrealtime();
#endif

  // ══════════════════════════════════════════════════════════════════
  // Phase 6: Cross-XCD attention barrier + O-proj weight DMA
  // Issue buffer_load_lds for O-proj weights BEFORE the barrier poll
  // so DMA runs in the background during the spin-wait (~5-60us).
  // ══════════════════════════════════════════════════════════════════
  MPK_TW_SUB(60, attn_release_expected);
  MPK_WS_PHASE(60, qkv_epoch_expected, xcd_id);
  int oproj_topk_tiles_per_xcd =
      oproj_tiles_per_xcd > router_tile_n ? oproj_tiles_per_xcd : router_tile_n;

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

    int oproj_tile_idx_pf = xcd_id * oproj_topk_tiles_per_xcd + xcd_rank;
    // Tile space is (column block, weight group) now, matching the O-proj
    // kernel's decode. The weight group is what selects the DMA source; the
    // column block only decides whether this tile has any live token at all.
    int oproj_bblk_pf =
        (xcd_rank % oproj_topk_tiles_per_xcd) / oproj_n_wgs_per_xcd;
    int oproj_wg_pf =
        (xcd_rank % oproj_topk_tiles_per_xcd) % oproj_n_wgs_per_xcd;

    if (xcd_rank < oproj_topk_tiles_per_xcd &&
        oproj_bblk_pf * 16 < num_active_tokens) {
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
  {
    int _obs;
    int _spins = 0;
    while ((_obs = ld_nt_s32(&attn_release[xcd_id * 16])) <
           attn_release_expected) {
      MPK_WS_WAIT_TICK(_obs, _spins);
      _spins++;
      __builtin_amdgcn_s_sleep(1);
    }
  }
  // Cross-XCD ACQUIRE for attn_out. Must be `sc1` (vL1 + L2).
  //
  // The producer is the Phase 5 merge, which writes attn_out with st_wt
  // (WRITE_THROUGH=true) -- sc0 sc1, bypassing L2 and landing in HBM. The
  // consumer is Phase 7's O-proj on a *different* XCD (the merge that produced
  // a given kv_head runs on the XCD that owns it), and MI300/MI350 L2 is not
  // coherent across XCDs.
  //
  // Plain `buffer_inv` drops vL1 only. This same worker read the same attn_out
  // addresses one layer ago, so its L2 still holds those lines -- and the load
  // is served from L2, returning the *previous* layer's attention output.
  // Which lines survive depends on inter-XCD L2 eviction timing, hence run to
  // run variation.
  //
  // This is distinct from the Phase 5 barrier's `buffer_inv` (no sc1), which
  // is correct as written: that one is intra-XCD (the chunk barrier is
  // per-XCD), so the producers share this L2 and invalidating it would discard
  // their partials. Here the producer is remote and the invalidate is required.
  // Retire the Phase 6 weight DMA BEFORE invalidating, not after.
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
  asm volatile("buffer_inv sc1" ::: "memory");
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

  // ══════════════════════════════════════════════════════════════════
  // Phase 7: O-proj + RMSNorm + Router + TopK
  // ══════════════════════════════════════════════════════════════════
  MPK_TW_SUB(70, oproj_topk_tiles_per_xcd);
  MPK_WS_PHASE(70, qkv_epoch_expected, xcd_id);
  {
    if (xcd_rank < oproj_topk_tiles_per_xcd) {
      int oproj_tile_idx = xcd_id * oproj_topk_tiles_per_xcd + xcd_rank;
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
          nullptr);
    }
  }

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
    int *my_release = &routing_ready[(1 + xcd_id) * 16];
    MPK_WS_WAIT_BEGIN(75, routing_expected);
    int _obs;
    int _spins = 0;
    while ((_obs = ld_nt_s32(my_release)) < routing_expected) {
      MPK_WS_WAIT_TICK(_obs, _spins);
      _spins++;
      __builtin_amdgcn_s_sleep(1);
    }
  }
  // Cross-XCD ACQUIRE for the routing data. Must be `sc1` (vL1 + L2).
  //
  // What this poll gates is active_expert_ids / routing_indices / topk_weight,
  // all written by the single TopK worker with st_wt (sc0 sc1) -- straight to
  // HBM, bypassing L2. Every MoE worker on the other seven XCDs reads them
  // here, and L2 is not coherent across XCDs, so dropping vL1 alone leaves
  // this XCD's L2 free to serve the previous layer's routing.
  //
  // A stale read here is not a crash: last layer's expert ids and route values
  // are all in range, so the token is quietly routed through the wrong
  // experts. That is the same failure the release-side fence in
  // gang_linear_mxfp4_res_bias_rmsnorm_topk_mi300.cuh describes -- this is its
  // acquire-side counterpart, which was never written.
  asm volatile("buffer_inv sc1" ::: "memory");

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  unsigned long long _fused_t3 = __builtin_amdgcn_s_memrealtime();
#endif

  // ══════════════════════════════════════════════════════════════════
  // Phase 8: MoE (W13+SwiGLU+W2)
  // ══════════════════════════════════════════════════════════════════
  for (int moe_t = xcd_rank; moe_t < moe_total_tiles_per_xcd;
       moe_t += workers_per_xcd) {
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
                                                            moe_t);
  }

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

    // Every worker must retire its own MoE stores before its arrival is
    // published: vmcnt is per-wave, so neither the release on the atomic nor
    // s_barrier covers the other waves. Drain, then barrier, then arrive --
    // the same ordering the other release sites in this file use.
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __syncthreads();

    if (tid == 0) {
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
      int const rel_prev = ld_nt_s32(&layer_release[xcd_id * 16]);

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
      int local_expected =
          (__atomic_load_n(&layer_local[xcd_id * 16], __ATOMIC_RELAXED) /
               workers_per_xcd +
           1) *
          workers_per_xcd;
      int local_prev =
          atom_add_release_gpu_s32(&layer_local[xcd_id * 16], 1);

      if (local_prev == local_expected - 1) {
        // Last worker on this XCD: this XCD's MoE stores are all retired, so
        // publish one arrival on behalf of the whole die.
        int global_expected =
            (__atomic_load_n(layer_global, __ATOMIC_RELAXED) / 8 + 1) * 8;
        int global_prev = atom_add_release_gpu_s32(layer_global, 1);
        if (global_prev == global_expected - 1) {
          // Last XCD overall: fan the release out with st_wt (write-through,
          // bypasses L2) so every worker polls a line it already owns.
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          for (int x = 0; x < 8; x++) {
            st_wt_u32((void *)&layer_release[x * 16],
                      (unsigned)global_expected);
          }
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        }
      }

      while (ld_nt_s32(&layer_release[xcd_id * 16]) <= rel_prev) {
        __builtin_amdgcn_s_sleep(1);
      }
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
    __syncthreads();
    // No invalidate here: the task body re-entered for the next layer opens
    // with `buffer_inv sc1` (the layer-boundary acquire near the top of this
    // function), which is the same instruction against the same data. Issuing
    // it twice per layer costs a full L2 invalidate for nothing.
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
  MPK_TW_SUB(90, tile_idx);
  MPK_WS_PHASE(90, qkv_epoch_expected, xcd_id);
}

} // namespace kernel
