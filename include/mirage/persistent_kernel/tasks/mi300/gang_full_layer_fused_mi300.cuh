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
static constexpr int FULL_LAYER_CHUNK_BARRIER_SLOT = 28 * 16;
static constexpr int FULL_LAYER_ATTN_XCD_RELEASE_SLOT = 36 * 16;

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
          bool DECODE_ONLY = false>
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
                                       int tile_idx) {
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

  // Invalidate vL1 to ensure we read fresh MoE atomicAdd results from L2.
  // Without this, stale zeros from QKV's workspace zeroing (flat_store in
  // previous iteration) can persist in vL1 across gang task boundaries.
  asm volatile("buffer_inv" ::: "memory");

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

  // All threads independently compute expected values (ld_nt is uniform).
  // Eliminates shared variables and __syncthreads broadcast.
  //
  // These snapshots are "current value + 1". That is only correct if this
  // worker is guaranteed to read *before* this layer's producer bumps the
  // counter. All 36 layers run inside a single task (see the ml loop in
  // persistent_kernel.cuh) with only a per-block __syncthreads between them,
  // so workers skew freely -- bounded only by barriers they mutually arrive
  // at. The ordering therefore comes from the Phase 2 barrier below, which
  // every worker on this XCD arrives at before either producer can run:
  //
  //   attn_release[x] is written by the last XCD to reach the attn_global
  //   barrier, and routing_ready[1+x] by the worker that completes TopK
  //   (gated on total_topk_tiles across all 8 XCDs). Both are downstream of
  //   every XCD's Phase 2 barrier, so both happen after this worker arrives
  //   there, hence after these loads.
  //
  // This is why Phase 2 must be joined by *all* workers_per_xcd and not just
  // the QKV/attention workers. Narrowing it lets a worker that arrives
  // nowhere before Phase 7b fall a full layer behind, read an already-bumped
  // counter, and then wait for a bump this layer never produces -- while the
  // layer that would produce it waits on that worker's MoE tiles. That is a
  // real deadlock, and it was an intermittent one because it needs a
  // full-layer skew to occur.
  int routing_expected = ld_nt_s32(routing_ready) + 1;
  int attn_release_expected = ld_nt_s32(&attn_release[xcd_id * 16]) + 1;

  // Every worker on this XCD takes part in the QKV epoch barrier. The
  // dispatcher gives this task exactly workers_per_xcd tiles per XCD (see
  // gang_task_tiles_per_xcd = params[28] in graph.cc), one per worker, so
  // ranks 0..workers_per_xcd-1 each arrive exactly once. Workers with no QKV
  // tile of their own arrive immediately and add no latency -- the barrier
  // still clears when the slowest QKV worker arrives.
  int const qkv_epoch_participants = workers_per_xcd;

  // Snapshot the QKV epoch before arriving at the barrier.
  //
  // This read is safe because the epoch bump is gated on *every* participant
  // having arrived (see the modular test below), and every participant
  // snapshots before it arrives. So no bump for this layer can land between
  // this load and this worker's arrival, and E+1 is always the epoch this
  // layer will reach.
  //
  // Do not narrow the participant set: a worker that waits on the epoch but
  // never arrives has no such ordering, can read an already-bumped value, and
  // will then wait for an epoch that this layer never reaches. Narrowing it
  // also breaks the ordering the routing_ready/attn_release snapshots above
  // depend on.
  int qkv_epoch_expected =
      __atomic_load_n(&qkv_epoch[xcd_id * 16], __ATOMIC_RELAXED) + 1;

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
  // qkv_epoch_expected is exactly the counter needed: the QKV epoch is bumped
  // once per layer, never reset, and counts (iterations * num_layers + layer).
  // Every worker on this XCD computes the same value -- that is guaranteed by
  // the Phase 2 barrier below, which is also what makes the snapshot valid.
  // Producer and consumer of the MoE barrier can sit on different XCDs, but
  // all XCDs stay on the same layer: the attn_global and TopK barriers are
  // cross-XCD, so no XCD can start layer L+1 until every XCD finished layer L,
  // and per-XCD epochs therefore advance in lockstep.
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
  // Arrived at by every worker on this XCD: the QKV GEMM workers, the
  // attention-chunk workers, and the MoE-only workers beyond both. Workers
  // with no QKV tile arrive immediately and then wait here for the GEMM to
  // land before reading K/V in Phase 3.
  //
  // This is the single point in the layer that every worker on the XCD passes
  // through, which is what makes it the layer's skew bound. Two things depend
  // on that:
  //
  //   1. Every waiter must also arrive, or its epoch snapshot is unordered
  //      against the bump and it can wait forever (see qkv_epoch_expected).
  //   2. It pins every worker to this layer before any of them can reach the
  //      attn_release / routing_ready producers, which is what makes those
  //      two "+1" snapshots well-defined (see the note above them).
  //
  // The arrival count is qkv_epoch_participants (= workers_per_xcd), not
  // total_qkv_tiles_per_xcd -- the modular test only fires when the arrival
  // count matches the number of workers that actually arrive.
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

    // All participants poll epoch (L2 coherent, per-XCD only)
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
    if (xcd_rank < NUM_KV_CHUNKS) {
      int kv_chunk_idx = xcd_rank;
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
          /*request_id=*/0,
          /*kv_head_idx=*/xcd_id,
          kv_chunk_idx,
          attn_scale,
          SLIDING_WINDOW,
          nullptr); // no sinks per-chunk

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
      _fused_t0c = __builtin_amdgcn_s_memrealtime();
#endif

      // ══════════════════════════════════════════════════════════════════
      // Phase 4: Chunk barrier — CROC last-chunk-worker runs merge
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
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      __syncthreads();
      threadfence_gpu();
      __shared__ int s_chunk_prev;
      if (tid == 0) {
        s_chunk_prev = atom_add_release_gpu_s32(&chunk_barrier[xcd_id * 16], 1);
      }
      __syncthreads();

      if ((s_chunk_prev % NUM_KV_CHUNKS) == NUM_KV_CHUNKS - 1) {
        // Last chunk worker: run merge (no reset needed — modular check)
        // Acquire the partials the other chunks released above. buffer_inv
        // drops stale vL1 lines so the merge reads what they actually wrote
        // rather than this CU's cached copy from a previous layer.
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
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
            /*request_id=*/0,
            reinterpret_cast<__hip_bfloat16 *>(
                output_ptrs[4]), // attn_out (bf16)
            /*kv_head_idx=*/xcd_id,
            HAS_SINKS ? input_ptrs[6] : nullptr); // sinks applied here

#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
        _merge_done = __builtin_amdgcn_s_memrealtime();
#endif
        // Wait for all write-through stores from merge to complete
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        if (tid == 0) {
          int attn_expected =
              (__atomic_load_n(attn_global, __ATOMIC_RELAXED) | 7) + 1;
          int prev = atom_add_release_gpu_s32(attn_global, 1);
          if (prev == attn_expected - 1) {
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

    // Activations are FP8 E4M3: one byte per element, one E8M0 scale byte per
    // 32-element block. See _gang_wave_parallel_fp8_quant.
    constexpr int OPROJ_FP_TOK = OPROJ_REDUCTION_SIZE;
    constexpr int OPROJ_FP_SCL = OPROJ_NUM_B32;
    constexpr int OPROJ_LDS_W_OFF =
        ((OPROJ_FP_TOK + OPROJ_FP_SCL + 15) / 16) * 16;

    extern __shared__ char _oproj_pf_smem[];

    int oproj_tile_idx_pf = xcd_id * oproj_topk_tiles_per_xcd + xcd_rank;
    int oproj_tok_pf =
        (xcd_rank % oproj_topk_tiles_per_xcd) / oproj_n_wgs_per_xcd;
    int oproj_wg_pf =
        (xcd_rank % oproj_topk_tiles_per_xcd) % oproj_n_wgs_per_xcd;

    if (xcd_rank < oproj_topk_tiles_per_xcd &&
        oproj_tok_pf < num_active_tokens) {
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
  asm volatile("buffer_inv" ::: "memory");
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

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
#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
          _ts_base
#else
          nullptr
#endif
      );
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
  asm volatile("buffer_inv" ::: "memory");

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
