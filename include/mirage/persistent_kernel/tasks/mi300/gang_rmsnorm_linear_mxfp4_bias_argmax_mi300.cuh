/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

// Fused RMSNorm + MXFP4 Gang Linear + Bias + Argmax for MI300/MI350.
//
// Norm once, FP8 quant once, internal tile loop over all WGs
// assigned to this worker. Argmax accumulated in registers across all tiles.
// Each worker writes one (bf16 max_val, int64 abs_idx) pair per batch row.
//
// total_tiles_per_xcd = workers_per_xcd (30 on MI300X).
// tile_idx = worker rank within XCD (0..29).
// Internal loop: wg = tile_idx, tile_idx + workers_per_xcd, ...
//
// ── Batching (bs > 1) rides the MFMA's idle N dimension ──────────────────
// The instruction is mfma_scale_f32_16x16x128_f8f6f4: M=16 weight rows,
// N=16 token columns, K=128. Lane l supplies A[m = l&15][k-blk = l>>4] and
// B[n = l&15][k-blk = l>>4], and receives D[m = (l>>4)*4 + i][n = l&15].
// The single-token version broadcast the same token into all 16 N columns
// and read back only n == 0, so 15/16 of every MFMA was thrown away. Here
// token `col` occupies N column `col`, so up to TOK_ROWS = 16 batch rows
// cost exactly what 1 row used to: same MFMA count, same weight traffic.
//
// Above 16 rows the batch is swept in column blocks (NUM_BBLK passes). That
// pass loop has to sit OUTSIDE the weight-tile loop, because 64 fp8 rows of
// K=2944 are 188 KB and the LDS budget is 152 KB — the token rows for the
// whole batch cannot be resident at once no matter how the weights are
// staged. So bs>16 re-reads the LM-head weights once per 16-row block; the
// alternative (re-quantizing inside the tile loop) would pay the quant's
// fixed ~15 us launch latency ~50x per worker and is far worse.
//
// Eliminates:
//   - 393 redundant RMSNorm + FP8 quant (down to 30 — one per worker)
//   - ~786 KB HBM logits write + read
//   - argmax_partial task dispatch

#pragma once
#include "tasks/mi300/gang_moe_linear_mxfp4_mi300.cuh"
#include "tasks/mi300/gang_rmsnorm_linear_bias_mi300.cuh"

namespace kernel {

#if defined(MPK_LM_HEAD_GROUP_PIPELINE) && !defined(MPK_LM_HEAD_KMAJOR)
#error "MPK_LM_HEAD_GROUP_PIPELINE requires MPK_LM_HEAD_KMAJOR"
#endif
#if defined(MPK_LM_HEAD_GROUP_PIPELINE) && !defined(MPK_LM_HEAD_WAVE_TILE_DMA)
#error "MPK_LM_HEAD_GROUP_PIPELINE requires MPK_LM_HEAD_WAVE_TILE_DMA"
#endif

#ifdef MPK_LM_HEAD_GROUP_PIPELINE
// gfx950 does not interlock a SALU write to M0 against the following
// load-to-LDS MUBUF. Keep the required independent wait state in the same asm
// block as the destination update and request.
__device__ __forceinline__ void lm_head_load_lds_dwordx4_hazard_safe(
    i32x4_t resource,
    __attribute__((address_space(3))) uint32_t *destination,
    uint32_t vector_offset,
    bool non_temporal = true) {
  unsigned const lds_offset = __builtin_amdgcn_readfirstlane(
      static_cast<unsigned>(reinterpret_cast<uintptr_t>(destination)));
  if (non_temporal) {
    asm volatile(
        "s_mov_b32 m0, %[lds_offset]\n"
        "s_nop 0\n"
        "buffer_load_dwordx4 %[vector_offset], %[resource], 0 offen sc0 nt "
        "lds\n"
        :
        : [vector_offset] "v"(vector_offset), [resource] "s"(resource),
          [lds_offset] "s"(lds_offset)
        : "memory", "m0");
  } else {
    asm volatile(
        "s_mov_b32 m0, %[lds_offset]\n"
        "s_nop 0\n"
        "buffer_load_dwordx4 %[vector_offset], %[resource], 0 offen lds\n"
        :
        : [vector_offset] "v"(vector_offset), [resource] "s"(resource),
          [lds_offset] "s"(lds_offset)
        : "memory", "m0");
  }
}
#endif

#ifdef MPK_ARGMAX_DUAL_REDUCE
// ── The LM head's g-group argmax, as two VALU swaps instead of four LDS ─────
//
// The reduction this replaces is only two steps wide -- xor-16 then xor-32,
// walking the four `g` groups and leaving `col` alone -- but each step costs
// two `__shfl_xor`, one for the value and one for the index, and `__shfl_xor`
// lowers to ds_bpermute: an LDS round trip and an lgkmcnt wait per shuffle for
// data that never left the register file. Four round trips, two waits.
//
// Both steps have an exact VALU-native equivalent on gfx950 and they are the
// only two the whole reduction needs: `v_permlane16_swap_b32` is xor-16 and
// `v_permlane32_swap_b32` is xor-32, so unlike the router's six-step chain
// this one needs no DPP row_shl stages at all.
//
// The swaps exchange one operand's low half with the other's high half, so
// seeding each peer with a copy of its own value makes swap-then-select a
// lane-uniform butterfly: in the low lanes the peer holds lane i^k's value
// and in the high lanes it holds the same, which is exactly what the shuffle
// delivered.
//
// The value and the index are two independent partials over the same lane
// crossing, so they interleave: the swap's result-hazard slot is filled by the
// other partial's swap rather than by an `s_nop`. That is why this is worth
// doing at two steps -- the pair costs barely more than one would.
//
// ── Contract: the result is valid in g == 0 (lanes 0..15) ──────────────────
//
// The value is lane-uniform and matches the butterfly everywhere. The *index*
// does not, and the reason is the same asymmetry that makes the sum version
// free: the swap leaves the lower member of each pair holding its own value in
// `val` and its partner's in `peer`, and the upper member holding them the
// other way round. A commutative reduction cannot see that. A strict `>`
// tie-break can -- in the upper member the incumbent is `peer`, so keeping
// `val` on a tie keeps the wrong side.
//
// Lanes 0..15 are the lower member at both steps (bit 4 clear for the xor-16
// swap, bit 5 clear for the xor-32 swap), so their tie-break is unchanged.
// That is all the caller reads: `if (g == 0) s_red_val[...] = thread_max`.
// Hence the name -- do not lift this to a caller that consumes g1..g3.
//
// Verified over 4096 cases including an all-equal block where every
// comparison is a tie: g0 value and index bit-identical to the butterfly,
// value bit-identical in all 64 lanes.
// See tests/standalone/test_argmax_dual_reduce.hip.
__device__ __forceinline__ void argmax_dual_wave_reduce_to_g0(float &val,
                                                              int &idx) {
  float val_peer = val;
  int idx_peer = idx;
  asm volatile(
      // ── xor-16: walk g groups 0<->1 and 2<->3 ──
      "s_nop 1\n"
      "v_permlane16_swap_b32 %[val], %[val_peer]\n"
      "v_permlane16_swap_b32 %[idx], %[idx_peer]\n"
      "v_cmp_gt_f32 vcc, %[val_peer], %[val]\n"
      "v_cndmask_b32 %[idx], %[idx], %[idx_peer], vcc\n"
      "v_cndmask_b32 %[val], %[val], %[val_peer], vcc\n"
      // Re-seed: the swaps are destructive in both operands, so the peer has
      // to be a fresh copy of the survivor rather than carried across steps.
      "v_mov_b32_e32 %[val_peer], %[val]\n"
      "v_mov_b32_e32 %[idx_peer], %[idx]\n"
      // ── xor-32: walk the two halves ──
      "s_nop 1\n"
      "v_permlane32_swap_b32 %[val], %[val_peer]\n"
      "v_permlane32_swap_b32 %[idx], %[idx_peer]\n"
      "v_cmp_gt_f32 vcc, %[val_peer], %[val]\n"
      "v_cndmask_b32 %[idx], %[idx], %[idx_peer], vcc\n"
      "v_cndmask_b32 %[val], %[val], %[val_peer], vcc"
      : [val] "+v"(val),
        [idx] "+v"(idx),
        [val_peer] "+v"(val_peer),
        [idx_peer] "+v"(idx_peer)
      :
      : "vcc");
}
#endif

template <int BATCH_SIZE,
          int OUTPUT_PER_WG,
          int REDUCTION_SIZE,
          int ACTUAL_HIDDEN_DIM = REDUCTION_SIZE>
__device__ __noinline__ void gang_rmsnorm_linear_mxfp4_bias_argmax_kernel(
    void const *norm_input_ptr,
    void const *norm_weight_ptr,
    void *norm_output_ptr,
    void const *weight_ptr,
    void const *bias_ptr,
    void *argmax_val_ptr, // [num_workers] bf16: per-worker max value
    void *argmax_idx_ptr, // [num_workers] int64: per-worker absolute index
    void *logits_out_ptr, // [max_seq_length, output_stride] f32, or nullptr.
                          // Perplexity mode only: when non-null the full logit
                          // row is also written to HBM at row `step`. Null on
                          // the serving path, which keeps logits in registers
                          // and pays no HBM traffic.
    int step,             // row of logits_out_ptr to write (ignored if null)
    int num_active_tokens,
    int n_wgs_per_xcd,
    int workers_per_xcd,
    int output_stride,
    int tile_idx) { // tile_idx = partition_index * workers_per_xcd +
                    // worker_rank

  static_assert(OUTPUT_PER_WG >= 64,
                "Argmax fusion only supports N-parallel path (OPW>=64)");
  static_assert(OUTPUT_PER_WG % 16 == 0);
  static_assert(REDUCTION_SIZE % 128 == 0);

  constexpr int NUM_BLOCKS_32 = REDUCTION_SIZE / 32;
  constexpr int WG_DATA_BYTES = OUTPUT_PER_WG * (REDUCTION_SIZE / 2);
  constexpr int WG_SCALE_BYTES = OUTPUT_PER_WG * NUM_BLOCKS_32;
  constexpr int WG_BYTES = WG_DATA_BYTES + WG_SCALE_BYTES;
  constexpr int K_PER_MFMA = 128;
  constexpr int MFMA_ITERS = REDUCTION_SIZE / K_PER_MFMA;
  constexpr int NUM_WAVES = 4;
  constexpr int TILES_PER_WAVE = OUTPUT_PER_WG / 16 / NUM_WAVES;

  // Batch tiling over the MFMA N dimension.
  constexpr int MFMA_N = 16;
  constexpr int TOK_ROWS = BATCH_SIZE < MFMA_N ? BATCH_SIZE : MFMA_N;
  constexpr int NUM_BBLK = (BATCH_SIZE + TOK_ROWS - 1) / TOK_ROWS;

  // LDS weight tile layout (padded for buffer_load_lds alignment)
  constexpr int QKV_TILE_ROWS = 16;
  constexpr int QKV_TILE_DATA = QKV_TILE_ROWS * (REDUCTION_SIZE / 2);
  constexpr int QKV_TILE_SCALE = QKV_TILE_ROWS * NUM_BLOCKS_32;
  constexpr int qkv_n16_data = QKV_TILE_DATA / 16;
  constexpr int QKV_LPT = (qkv_n16_data + 255) / 256;
  constexpr int QKV_TILE_DATA_PADDED = QKV_LPT * 256 * 16;
  constexpr int QKV_TILE_BYTES = QKV_TILE_DATA_PADDED + QKV_TILE_SCALE;
#ifdef MPK_LM_HEAD_GROUP_PIPELINE
  constexpr int QKV_SCALE_TILE_PADDED =
      ((QKV_TILE_SCALE + 1023) / 1024) * 1024;
  constexpr int QKV_SCALE_PIPELINE_BYTES = QKV_SCALE_TILE_PADDED * NUM_WAVES;
#else
  constexpr int QKV_SCALE_PIPELINE_BYTES = 0;
#endif

  // FP8 token rows sit at the start of LDS, one row per MFMA N column.
  // The +16 pad keeps consecutive rows off one bank group: the 16 lanes of a
  // g-group each read 16 B at TOK_ROW_STRIDE spacing, and an unpadded 2944 B
  // stride (736 dwords, 736 % 32 == 0) would serialize all 16 into one bank.
  constexpr int TOK_ROW_STRIDE = REDUCTION_SIZE + 16;
  constexpr int SC_STRIDE = ((MFMA_ITERS + 3) / 4) * 4;
  constexpr int TOK_REGION = TOK_ROWS * TOK_ROW_STRIDE;
  constexpr int SC_REGION = TOK_ROWS * SC_STRIDE;
  // Cross-wave argmax scratch: NUM_WAVES x MFMA_N of (float val, int idx).
  constexpr int RED_OFF = ((TOK_REGION + SC_REGION + 15) / 16) * 16;
  constexpr int RED_REGION = NUM_WAVES * MFMA_N * 8;
  constexpr int QKV_LDS_OFF = ((RED_OFF + RED_REGION + 15) / 16) * 16;
  static_assert(QKV_LDS_OFF + QKV_TILE_BYTES * NUM_WAVES +
                            2 * QKV_SCALE_PIPELINE_BYTES <=
                    mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE -
                        mirage::runtime::LAYER_IDX_SMEM_OFFSET_FROM_END,
                "LM-head LDS exceeds the MI350X dynamic LDS budget");
  static_assert(TOK_ROW_STRIDE % 16 == 0,
                "token row stride must keep the i32x4 B-operand loads aligned");
#ifdef MPK_LM_HEAD_GROUP_PIPELINE
  static_assert(BATCH_SIZE == 1,
                "MPK_LM_HEAD_GROUP_PIPELINE is batch-1 only");
  static_assert(REDUCTION_SIZE == 2944,
                "MPK_LM_HEAD_GROUP_PIPELINE requires K=2944");
  static_assert(OUTPUT_PER_WG == 64,
                "MPK_LM_HEAD_GROUP_PIPELINE requires OPW=64");
  static_assert(NUM_WAVES == 4 && TILES_PER_WAVE == 1,
                "MPK_LM_HEAD_GROUP_PIPELINE requires four waves, one tile each");
  static_assert(QKV_TILE_DATA == MFMA_ITERS * 1024,
                "each K128 fragment must be one 64-lane 16-byte request");
#endif

  uint8_t const *W = (uint8_t const *)weight_ptr;
  unsigned short const *d_bias = (unsigned short const *)bias_ptr;

  extern __shared__ char _rnlm_smem[];
  uint8_t *s_tok_fp8 = (uint8_t *)_rnlm_smem;
  uint8_t *s_tok_scales = s_tok_fp8 + TOK_REGION;
  float *s_red_val = (float *)((uint8_t *)_rnlm_smem + RED_OFF);
  int *s_red_idx = (int *)(s_red_val + NUM_WAVES * MFMA_N);
  uint8_t *qkv_lds_w = (uint8_t *)_rnlm_smem + QKV_LDS_OFF;
#ifdef MPK_LM_HEAD_GROUP_PIPELINE
  uint8_t *scale_pipeline_lds = qkv_lds_w + QKV_TILE_BYTES * NUM_WAVES;
#endif

  int const tid = threadIdx.x;
  int const warp_id = tid >> 6;
  int const lane_id = tid & 63;
  int const col = lane_id & 15; // MFMA N index == token within block
  int const g = lane_id >> 4;   // MFMA k-block / output-row group

  unsigned short *argmax_vals =
      reinterpret_cast<unsigned short *>(argmax_val_ptr);
  long long *argmax_idxs = reinterpret_cast<long long *>(argmax_idx_ptr);

  int batch_count =
      (num_active_tokens < BATCH_SIZE) ? num_active_tokens : BATCH_SIZE;
  int partition_index = tile_idx / workers_per_xcd; // logical XCD (0..7)
  int worker_rank = tile_idx % workers_per_xcd;     // 0..workers_per_xcd-1
  int partition_start = partition_index * n_wgs_per_xcd * OUTPUT_PER_WG;
  // argmax_part_* are [batch, num_workers], XCD-partitioned along dim 1, so
  // the pointer already points at this XCD's worker slice while the row
  // stride is still the full worker count.
  int const argmax_row_stride = workers_per_xcd * 8;

  // ── Step 1: RMSNorm ─────────────────────────────────────────────────────
  // DO NOT add a __syncthreads() to this loop body. Measured at bs=2: without
  // it the run completes and both requests decode correct prose; with it the
  // kernel deadlocks before a single task retires (tasks_done=0), and that
  // holds for BOTH a runtime bound (batch_count) and a compile-time one
  // (BATCH_SIZE), so it is the barrier and not the trip count.
  // rmsnorm_inline_amd already ends with its own cross-wave barrier pair; an
  // extra one here is what breaks it.
  for (int b = 0; b < batch_count; b++) {
    unsigned short const *row_in =
        (unsigned short const *)norm_input_ptr + (long long)b * REDUCTION_SIZE;
    unsigned short *row_out =
        (unsigned short *)norm_output_ptr + (long long)b * REDUCTION_SIZE;
    gang_rmsnorm_detail::rmsnorm_inline_amd<REDUCTION_SIZE, ACTUAL_HIDDEN_DIM>(
        row_in, norm_weight_ptr, row_out);
  }

  // SRD for weight buffer (hoisted outside every loop)
  uint32_t qkv_buf_range = static_cast<uint32_t>(n_wgs_per_xcd) * WG_BYTES;
  i32x4_t qkv_rsrc = make_w_buffer_rsrc(W, qkv_buf_range);

#ifdef MPK_LM_HEAD_WAVE_TILE_DMA
  // Scalar LDS destination for the Phase A weight train, hoisted out of every
  // loop. buffer_load_dwordx4 ... lds takes its LDS base from M0, which is
  // scalar; leaving the destination as a pointer derived from threadIdx makes
  // the compiler re-prove uniformity and emit a v_readfirstlane + s_mov m0 per
  // request. Reading it once here leaves the train nothing to do but
  // s_addk_i32 m0 between loads. Casting through the address_space(3) pointer
  // (32-bit) rather than truncating the generic 64-bit address keeps this off
  // any assumption about where the LDS aperture sits.
  unsigned const lm_lds_wave_base = __builtin_amdgcn_readfirstlane(
      static_cast<unsigned>(reinterpret_cast<uintptr_t>(
          (__attribute__((address_space(3)))
           uint8_t *)(qkv_lds_w + warp_id * QKV_TILE_BYTES))));
#endif

  // ── Batch column-block sweep ────────────────────────────────────────────
  // NUM_BBLK == 1 for bs <= 16, so this collapses away entirely and the
  // bs<=16 path costs exactly what the old single-token path did.
#pragma unroll 1
  for (int bblk = 0; bblk < NUM_BBLK; bblk++) {
    int const tok_row = bblk * MFMA_N + col;
    bool const tok_active = tok_row < batch_count;

    // ── Step 2: FP8 quant of this block's token rows ──────────────────────
    _gang_multirow_fp8_quant<REDUCTION_SIZE, TOK_ROWS, BATCH_SIZE,
                             TOK_ROW_STRIDE, SC_STRIDE>(
        (unsigned short const *)norm_output_ptr, REDUCTION_SIZE, bblk * MFMA_N,
        TOK_ROWS, s_tok_fp8, s_tok_scales);

    // Each lane owns exactly one token (N column `col`), so the running
    // argmax stays a scalar per lane.
    float thread_max = -1e30f;
    int thread_max_idx = -1;

    // B operand base for this lane's token. Inactive lanes clamp to row 0 to
    // keep their LDS reads in range; their results are discarded below.
    uint8_t const *b_data = s_tok_fp8 + (tok_active ? col : 0) * TOK_ROW_STRIDE;
    uint8_t const *b_scale =
        s_tok_scales + (tok_active ? col : 0) * SC_STRIDE;

    // ── Step 3: Internal tile loop + argmax ────────────────────────────────
    int group_iteration = 0;
    for (int wg_idx = worker_rank; wg_idx < n_wgs_per_xcd;
         wg_idx += workers_per_xcd, ++group_iteration) {
      uint32_t qkv_wg_voff = static_cast<uint32_t>(wg_idx) * WG_BYTES;

    // ── Phase A: Prefetch weight data into LDS via buffer_load_dwordx4 lds:1
    // ──
#ifdef MPK_LM_HEAD_GROUP_PIPELINE
      // Group zero remains synchronous. Every later group was streamed into
      // this recycled tile by the preceding group's MFMA loop. Fleet's queue
      // stages TaskDesc only immediately before its dependency acquire; unlike
      // Redline's immutable fixed-rank plan, it has no publication-safe earlier
      // descriptor from which to prefetch group zero. Adding that hook requires
      // a separate scheduler protocol change, so this patch deliberately does
      // not move task-descriptor reads across the dependency loop.
      if (group_iteration == 0) {
#endif
#ifdef MPK_LM_HEAD_WAVE_TILE_DMA
      // ── One wave, one tile, one instruction train ─────────────────────────
      //
      // The striped form below has all four waves interleave 16-byte slots
      // across all four tiles: 24 intrinsic calls per thread, each recomputing
      // a clamped index and an LDS destination the compiler has to re-scalarize
      // for M0. But a wave's 64 lanes already cover exactly 1 KiB, the tile is
      // an exact multiple of 1 KiB, and Phase C has wave `w` read tile `w`
      // (TILES_PER_WAVE == 1) -- so the natural unit is the whole tile, issued
      // by its own consumer as one chain of loads that advances the LDS base in
      // M0 and the global offset in a VGPR together. The clamp disappears
      // (nothing is partial), the duplicate loads it produced disappear, and
      // the per-request address math collapses to one s_addk_i32 + one
      // v_add_u32.
      //
      // The gfx950 M0-to-MUBUF wait state is why the s_nop pair guards the
      // s_mov: a buffer_load ... lds issued too close to the M0 write reads the
      // stale base.
      {
        static_assert(TILES_PER_WAVE == 1,
                      "the wave-tile train assumes wave w owns exactly tile w; "
                      "with more tiles per wave the train has to be repeated "
                      "per tile and the LDS base recomputed between them");
        static_assert(QKV_TILE_DATA % 1024 == 0,
                      "a tile must be a whole number of 64-lane x 16 B loads");
        static_assert(QKV_TILE_DATA_PADDED % 1024 == 0);
#ifdef MPK_LM_HEAD_KMAJOR
        // K-major, Phase C reads exactly [0, QKV_TILE_DATA): 16 B per lane,
        // MFMA_ITERS blocks of 1 KiB. Nothing past the data is ever touched.
        constexpr int LM_DMA_CHUNKS = QKV_TILE_DATA / 1024;
#else
        // Row-major, Phase C's 32-byte reads run 16 B past the last row's last
        // fragment, so the train covers the padded region too. That last chunk
        // reads from the record's scale suffix rather than out of the buffer --
        // in range, and it lands only in the don't-care upper half of an FP4
        // A-operand.
        constexpr int LM_DMA_CHUNKS = QKV_TILE_DATA_PADDED / 1024;
        static_assert(NUM_WAVES * QKV_TILE_DATA +
                              (LM_DMA_CHUNKS * 1024 - QKV_TILE_DATA) <=
                          WG_BYTES,
                      "the padded tail must stay inside the workgroup record");
#endif
        uint32_t voff = qkv_wg_voff +
                        static_cast<uint32_t>(warp_id * QKV_TILE_DATA) +
                        static_cast<uint32_t>(lane_id * 16);
        asm volatile(
            "s_nop 4\n"
            "s_mov_b32 m0, %[lds_base]\n"
            "s_nop 0\n"
            "buffer_load_dwordx4 %[voff], %[rsrc], 0 offen sc0 nt lds\n"
            ".rept %c[reps]\n"
            "s_addk_i32 m0, 0x400\n"
            "v_add_u32_e32 %[voff], 0x400, %[voff]\n"
            "buffer_load_dwordx4 %[voff], %[rsrc], 0 offen sc0 nt lds\n"
            ".endr\n"
            : [voff] "+v"(voff)
            : [rsrc] "s"(qkv_rsrc),
              [lds_base] "s"(lm_lds_wave_base),
              [reps] "n"(LM_DMA_CHUNKS - 1)
            : "memory", "m0");
      }
#else
      {
        auto *qkv_lds_warp_base = (__attribute__((address_space(3)))
                                   uint32_t *)(qkv_lds_w + warp_id * 1024);
#pragma unroll
      for (int t = 0; t < NUM_WAVES; t++) {
#pragma unroll
        for (int j = 0; j < QKV_LPT; j++) {
          int idx = tid + j * 256;
          int clamped = idx < qkv_n16_data ? idx : qkv_n16_data - 1;
          uint32_t voff =
              qkv_wg_voff +
              static_cast<uint32_t>(t * QKV_TILE_ROWS * (REDUCTION_SIZE / 2)) +
              static_cast<uint32_t>(clamped * 16);
          auto *lds_dst = (__attribute__((address_space(3)))
                           uint32_t *)((uint8_t __attribute__((address_space(
                                           3))) *)qkv_lds_warp_base +
                                       t * QKV_TILE_BYTES + j * 4096);
          __llvm_amdgcn_raw_buffer_load_lds(
              qkv_rsrc, lds_dst, 16, static_cast<int>(voff), 0, 0, 3);
        }
      }
    }
#endif
#ifdef MPK_LM_HEAD_GROUP_PIPELINE
      }
#endif

      // ── Phase B: Drain data loads, load scales via VGPR, scatter to LDS ──
      constexpr int QKV_SC_DW4_PER_TILE = QKV_TILE_SCALE / 16;
      constexpr int QKV_TOTAL_SC_DW4 = QKV_SC_DW4_PER_TILE * NUM_WAVES;
      constexpr int QKV_SC_LPT = (QKV_TOTAL_SC_DW4 + 255) / 256;

      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");

      // Issue group-zero scales from HBM via VGPR. Later groups use the
      // ping-pong direct-to-LDS scale buffers filled below.
#ifdef MPK_LM_HEAD_GROUP_PIPELINE
      if (group_iteration == 0) {
#endif
      uint8_t const *wg_scales_hbm =
          W + static_cast<int64_t>(wg_idx) * WG_BYTES + WG_DATA_BYTES;
      i32x4_t qkv_sc_buf[QKV_SC_LPT];
      {
        i32x4_t const *sc_src = (i32x4_t const *)wg_scales_hbm;
#pragma unroll
      for (int j = 0; j < QKV_SC_LPT; j++) {
        int idx = tid + j * 256;
        if (idx < QKV_TOTAL_SC_DW4) {
          qkv_sc_buf[j] = sc_src[idx];
        }
      }
    }

    // Drain scale loads, scatter to per-tile LDS slots
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    {
#pragma unroll
      for (int j = 0; j < QKV_SC_LPT; j++) {
        int idx = tid + j * 256;
        if (idx < QKV_TOTAL_SC_DW4) {
          int tile = idx / QKV_SC_DW4_PER_TILE;
          int off = idx % QKV_SC_DW4_PER_TILE;
          i32x4_t *dst_sc = (i32x4_t *)(qkv_lds_w + tile * QKV_TILE_BYTES +
                                        QKV_TILE_DATA_PADDED);
          dst_sc[off] = qkv_sc_buf[j];
        }
      }
    }
#ifdef MPK_LM_HEAD_GROUP_PIPELINE
      }
#endif

    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    __syncthreads();

#ifdef MPK_LM_HEAD_GROUP_PIPELINE
    // Stage the following group's 1,472-byte scale tile per wave into one of
    // two disjoint buffers. One wave request covers 1 KiB and a second,
    // partially-executing request covers the exact 448-byte tail.
    int const pipelined_next_wg_idx = wg_idx + workers_per_xcd;
    if (pipelined_next_wg_idx < n_wgs_per_xcd) {
      constexpr int SCALE_TAIL_VECTORS = (QKV_TILE_SCALE - 1024) / 16;
      static_assert(QKV_TILE_SCALE > 1024);
      static_assert(QKV_TILE_SCALE - 1024 == SCALE_TAIL_VECTORS * 16);
      int const next_scale_buffer = group_iteration & 1;
      auto *next_scale_wave_lds = (__attribute__((address_space(3)))
                                   uint32_t *)(scale_pipeline_lds +
                                               next_scale_buffer *
                                                   QKV_SCALE_PIPELINE_BYTES +
                                               warp_id *
                                                   QKV_SCALE_TILE_PADDED);
      uint32_t next_scale_offset =
          static_cast<uint32_t>(pipelined_next_wg_idx) * WG_BYTES +
          WG_DATA_BYTES + warp_id * QKV_TILE_SCALE + lane_id * 16;
      lm_head_load_lds_dwordx4_hazard_safe(qkv_rsrc, next_scale_wave_lds,
                                           next_scale_offset, false);
      if (lane_id < SCALE_TAIL_VECTORS) {
        lm_head_load_lds_dwordx4_hazard_safe(
            qkv_rsrc, next_scale_wave_lds + 1024 / sizeof(uint32_t),
            next_scale_offset + 1024, false);
      }
    }
#endif

    // ── Phase C: MFMA loop reading from LDS ──
    for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; tile_iter++) {
      int wave_tile = warp_id + tile_iter * NUM_WAVES;
      uint8_t const *tile_data_lds =
          (uint8_t const *)(qkv_lds_w + wave_tile * QKV_TILE_BYTES);
      uint8_t const *tile_scale_lds;
#ifdef MPK_LM_HEAD_GROUP_PIPELINE
      if (group_iteration == 0) {
        tile_scale_lds = tile_data_lds + QKV_TILE_DATA_PADDED;
      } else {
        int const current_scale_buffer = (group_iteration - 1) & 1;
        tile_scale_lds =
            scale_pipeline_lds +
            current_scale_buffer * QKV_SCALE_PIPELINE_BYTES +
            warp_id * QKV_SCALE_TILE_PADDED;
      }
#else
      tile_scale_lds = tile_data_lds + QKV_TILE_DATA_PADDED;
#endif

      int w_row_in_tile = col;
      int const row_scale_base = w_row_in_tile * NUM_BLOCKS_32;

#ifdef MPK_LM_HEAD_GROUP_PIPELINE
      // This readonly request is older than all 23 next-group data requests.
      // vmcnt(23) below therefore retires it without draining that train.
      using BiasDwords = unsigned int __attribute__((ext_vector_type(2)));
      BiasDwords prefetched_bias;
      int const rel_idx_base = wave_tile * 16 + g * 4;
      if (tok_active) {
        auto const *bias_address =
            d_bias + wg_idx * OUTPUT_PER_WG + rel_idx_base;
        asm volatile("global_load_dwordx2 %0, %1, off"
                     : "=v"(prefetched_bias)
                     : "v"(bias_address)
                     : "memory");
      }
#endif

      // ── Weight fragment addressing ──────────────────────────────────────
      //
      // Row-major (default): lane (g, col) reads
      //   col * (REDUCTION_SIZE / 2) + KI * 64 + g * 16.
      // At REDUCTION_SIZE 2944 the row stride is 1472 B = 368 dwords and
      // 368 % 32 == 16, so the sixteen values of `col` land on just two bank
      // groups; with g * 16 adding four more the whole wave starts on eight
      // distinct banks, eight lanes deep on each -- an 8-way conflict on
      // every weight ds_read in the K loop. It also reads 32 B per lane when
      // an FP4 A-operand needs 16.
      //
      // K-major (MPK_LM_HEAD_KMAJOR): the record is repacked offline from
      // [tile][row][k128][quarter][16B] to [tile][k128][quarter][row][16B],
      // so the lane's fragment sits at lane_id * 16 within its K block --
      // lane_id == g * 16 + col is exactly quarter-major-then-row. The wave
      // then reads 64 consecutive 16-byte fragments, which is the
      // conflict-free pattern ds_read_b128 is built for, and the HBM side of
      // Phase A is a byte-for-byte image of the tile, so the permutation
      // carries into LDS with no change to the DMA.
      //
      // Only the address changes; the same bytes reach the same lane, so this
      // is bit-exact. The scale suffix stays row-major in both layouts (one
      // byte per 32-element block, read as a scalar), hence the shared
      // row_scale_base. Host side: shuffle_lm_head_record_kmajor() in
      // demo/gpt_oss/demo.py, gated on the same MPK_LM_HEAD_KMAJOR.
#ifdef MPK_LM_HEAD_KMAJOR
      static_assert(QKV_TILE_ROWS == 16,
                    "MPK_LM_HEAD_KMAJOR assumes a 16-row MFMA tile: lane_id "
                    "spans exactly 16 rows x 4 K-quarters, which is what "
                    "makes lane_id * 16 the fragment address.");
      // One K128 block holds all 16 rows x 4 quarters x 16 B.
      constexpr int LM_LDS_K_STRIDE = QKV_TILE_ROWS * (K_PER_MFMA / 2);
      static_assert(((MFMA_ITERS - 1) * LM_LDS_K_STRIDE) + 64 * 16 <=
                        QKV_TILE_DATA,
                    "K-major fragment sweep must stay inside the tile");
      int const lds_data_lane_offset = lane_id * 16;
#define MPK_LM_W_FRAG(KI)                                                      \
  _gang_lm_load_fp4_a_kmajor(tile_data_lds + lds_data_lane_offset +            \
                             (KI)*LM_LDS_K_STRIDE)
#else
      int const row_data_base = w_row_in_tile * (REDUCTION_SIZE / 2);
#define MPK_LM_W_FRAG(KI)                                                      \
  (*(i32x8_t const *)(tile_data_lds + row_data_base +                          \
                      (KI) * (K_PER_MFMA / 2) + g * 16))
#endif

      f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

      i32x8_t a0 = MPK_LM_W_FRAG(0);
      int sa0 = (int)tile_scale_lds[row_scale_base + 0 * 4 + g];
      i32x8_t a1 = MPK_LM_W_FRAG(1);
      int sa1 = (int)tile_scale_lds[row_scale_base + 1 * 4 + g];
      i32x8_t a2 = MPK_LM_W_FRAG(2);
      int sa2 = (int)tile_scale_lds[row_scale_base + 2 * 4 + g];
      i32x8_t a3 = MPK_LM_W_FRAG(3);
      int sa3 = (int)tile_scale_lds[row_scale_base + 3 * 4 + g];

#ifdef MPK_LM_HEAD_GROUP_PIPELINE
#define MPK_LM_PIPE_NEXT(KI)                                                   \
  do {                                                                         \
    if (pipelined_next_wg_idx < n_wgs_per_xcd) {                               \
      uint32_t const next_vector_offset =                                      \
          static_cast<uint32_t>(pipelined_next_wg_idx) * WG_BYTES +            \
          static_cast<uint32_t>(warp_id * QKV_TILE_DATA) +                     \
          static_cast<uint32_t>((KI) * 1024 + lane_id * 16);                   \
      auto *next_destination = (__attribute__((address_space(3)))              \
                                 uint32_t *)(qkv_lds_w +                        \
                                             warp_id * QKV_TILE_BYTES +        \
                                             (KI) * 1024);                     \
      lm_head_load_lds_dwordx4_hazard_safe(                                    \
          qkv_rsrc, next_destination, next_vector_offset);                     \
    }                                                                          \
  } while (0)
#endif

#pragma unroll 1
      for (int ki = 0; ki < MFMA_ITERS; ki += 4) {
        {
          i32x8_t b = _gang_load_fp8_mfma_b(b_data, ki * K_PER_MFMA, g);
          int sb = (int)b_scale[ki];
          acc = _gang_mfma_f4xf8(a0, b, acc, sa0, sb);
        }
#ifdef MPK_LM_HEAD_GROUP_PIPELINE
        MPK_LM_PIPE_NEXT(ki);
#endif
        if (ki + 4 < MFMA_ITERS) {
          a0 = MPK_LM_W_FRAG(ki + 4);
          sa0 = (int)tile_scale_lds[row_scale_base + (ki + 4) * 4 + g];
        }
        {
          i32x8_t b = _gang_load_fp8_mfma_b(b_data, (ki + 1) * K_PER_MFMA, g);
          int sb = (int)b_scale[ki + 1];
          acc = _gang_mfma_f4xf8(a1, b, acc, sa1, sb);
        }
#ifdef MPK_LM_HEAD_GROUP_PIPELINE
        MPK_LM_PIPE_NEXT(ki + 1);
#endif
        if (ki + 5 < MFMA_ITERS) {
          a1 = MPK_LM_W_FRAG(ki + 5);
          sa1 = (int)tile_scale_lds[row_scale_base + (ki + 5) * 4 + g];
        }
        {
          i32x8_t b = _gang_load_fp8_mfma_b(b_data, (ki + 2) * K_PER_MFMA, g);
          int sb = (int)b_scale[ki + 2];
          acc = _gang_mfma_f4xf8(a2, b, acc, sa2, sb);
        }
#ifdef MPK_LM_HEAD_GROUP_PIPELINE
        MPK_LM_PIPE_NEXT(ki + 2);
#endif
        if (ki + 6 < MFMA_ITERS) {
          a2 = MPK_LM_W_FRAG(ki + 6);
          sa2 = (int)tile_scale_lds[row_scale_base + (ki + 6) * 4 + g];
        }
        if (ki + 3 < MFMA_ITERS) {
          i32x8_t b = _gang_load_fp8_mfma_b(b_data, (ki + 3) * K_PER_MFMA, g);
          int sb = (int)b_scale[ki + 3];
          acc = _gang_mfma_f4xf8(a3, b, acc, sa3, sb);
#ifdef MPK_LM_HEAD_GROUP_PIPELINE
          MPK_LM_PIPE_NEXT(ki + 3);
#endif
        }
        if (ki + 7 < MFMA_ITERS) {
          a3 = MPK_LM_W_FRAG(ki + 7);
          sa3 = (int)tile_scale_lds[row_scale_base + (ki + 7) * 4 + g];
        }
      }
#undef MPK_LM_W_FRAG
#ifdef MPK_LM_HEAD_GROUP_PIPELINE
#undef MPK_LM_PIPE_NEXT
      // Final emitted order per wave is: two next-scale requests, one current
      // bias request, then exactly 23 next-data requests. Keep the 23 younger
      // data requests in flight across argmax while retiring every current
      // consumer. The next iteration's vmcnt(0)/lgkmcnt(0) drains all of them
      // before the barrier makes the recycled LDS tile visible.
      if (pipelined_next_wg_idx < n_wgs_per_xcd) {
        asm volatile("s_waitcnt vmcnt(23)" ::: "memory");
      } else {
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      }
#endif

      // Argmax epilogue: accumulate across ALL tiles in registers.
      // The guard is on `tok_active`, not `col == 0`: lane (g, col) now holds
      // the 4 outputs for token `col`, so all 16 N columns are live. It sits
      // strictly AFTER the MFMA -- narrowing exec around the ds_read_b128 /
      // MFMA pair lets the compiler hoist the mask and feed stale B operands
      // to the lanes that ARE active.
      if (tok_active) {
        for (int i = 0; i < 4; i++) {
          int rel_idx = wave_tile * 16 + g * 4 + i;
          int abs_idx = partition_start + wg_idx * OUTPUT_PER_WG + rel_idx;
          float sum = acc[i];
#ifdef MPK_LM_HEAD_GROUP_PIPELINE
          unsigned const packed_bias = prefetched_bias[i >> 1];
          unsigned const bt = (i & 1) != 0 ? packed_bias & 0xffff0000U
                                            : packed_bias << 16;
#else
          unsigned bt = (unsigned)d_bias[wg_idx * OUTPUT_PER_WG + rel_idx]
                        << 16;
#endif
          float bv;
          __builtin_memcpy(&bv, &bt, 4);
          float val = sum + bv;
          if (val > thread_max) {
            thread_max = val;
            thread_max_idx = abs_idx;
          }
          // Perplexity mode: also spill the logit to HBM. abs_idx is the
          // vocab column this lane owns, so the writes across all workers
          // tile the row exactly once -- no atomics, no races.
          //
          // Stored as f32, not bf16: bf16 carries ~0.4% relative precision,
          // which is the same order as the GEMM error this buffer exists to
          // measure. Truncating here would fold a comparable one-sided error
          // into the very quantity under test.
          //
          // `step` is the position scored by token row 0 of this iteration;
          // row `tok_row` scored the position `tok_row` later, so each token
          // gets its own logits row and the rows still tile exactly once.
          if (logits_out_ptr != nullptr) {
            reinterpret_cast<float *>(
                logits_out_ptr)[(long long)(step + tok_row) * output_stride +
                                abs_idx] = val;
          }
        }
      }
    }
    } // end tile loop

    // ── Reduce within the wave, across g only ───────────────────────────
    // NOT the full 64-lane butterfly: lanes differing in `col` hold different
    // tokens, so mixing them would return one token's argmax for all 16.
    // XOR on bits 4 and 5 walks the 4 g-groups and leaves `col` alone.
#ifdef MPK_ARGMAX_DUAL_REDUCE
    // Same two steps, same order, no LDS. Valid in g == 0 only, which is the
    // only group the `if (g == 0)` write below reads -- see the contract note
    // at argmax_dual_wave_reduce_to_g0.
    argmax_dual_wave_reduce_to_g0(thread_max, thread_max_idx);
#else
#pragma unroll
    for (int offset = 16; offset <= 32; offset <<= 1) {
      float other_val = __shfl_xor(thread_max, offset, 64);
      int other_idx = __shfl_xor(thread_max_idx, offset, 64);
      if (other_val > thread_max) {
        thread_max = other_val;
        thread_max_idx = other_idx;
      }
    }
#endif

    // ── Cross-wave reduce via LDS, one slot per (wave, token) ───────────
    // No barrier before the write: s_red_* is disjoint from the token region
    // the MFMA loop just read, and the previous block's readers were drained
    // by the barrier at the bottom of this sweep.
    if (g == 0) {
      s_red_val[warp_id * MFMA_N + col] = thread_max;
      s_red_idx[warp_id * MFMA_N + col] = thread_max_idx;
    }
    __syncthreads();

    // ── Write per-(token, worker) result ────────────────────────────────
    if (tid < MFMA_N) {
      int out_row = bblk * MFMA_N + tid;
      if (out_row < batch_count) {
        float best_val = -1e30f;
        int best_idx = -1;
        for (int w = 0; w < NUM_WAVES; w++) {
          float v = s_red_val[w * MFMA_N + tid];
          if (v > best_val) {
            best_val = v;
            best_idx = s_red_idx[w * MFMA_N + tid];
          }
        }
        unsigned int vbits;
        __builtin_memcpy(&vbits, &best_val, 4);
        unsigned short bf16_val = (unsigned short)(vbits >> 16);
        long long out_off = (long long)out_row * argmax_row_stride + worker_rank;
        argmax_vals[out_off] = bf16_val;
        argmax_idxs[out_off] = (long long)best_idx;
      }
    }
    // The next block's quantizer overwrites s_tok_fp8, and s_red_* is reused.
    // Compiled out at bs <= 16, where there is no next block.
    if constexpr (NUM_BBLK > 1) {
      __syncthreads();
    }
  } // end batch column-block sweep
}

} // namespace kernel
