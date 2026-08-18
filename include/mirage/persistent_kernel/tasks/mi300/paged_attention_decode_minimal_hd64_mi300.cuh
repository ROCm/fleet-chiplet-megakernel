#pragma once
// MFMA decode attention for HEAD_DIM=64 (GPT-OSS 120B).
// Adapted from n-tiling-on-mpk's paged_attention_decode_minimal_mi300.cuh
// (HD=128).
//
// One request per kernel call (matches MPK split-KV scheme).
// 256 threads, 4 warps × 64 lanes:
//   - 2 QK MFMAs cover HD=64 reduction (NUM_K32 = 2)
//   - 1 PV MFMA per warp covers 16 dims (4 warps × 16 = 64)
//   - LDS: 2 KB K + 2 KB V (plain row-major, no XOR needed at HD=64)
//
// Online softmax: scores in log2 base (scale_s already includes log2(e)),
// LSE written as m_val + logf(d_val) to match existing CK FMHA convention.

#include <hip/hip_bf16.h>

// Default ON. See prefetch_tile in __attn_wave_local_scan_hd64 for what this
// gates and why it is bit-exact; MPK_ATTN_SCALAR_PAGE=0 ablates.
#ifndef MPK_ATTN_SCALAR_PAGE
#define MPK_ATTN_SCALAR_PAGE 1
#endif

// Minimum tiles per chunk for the wave-local scan, i.e. >= 2 tiles per wave.
//
// This was 16 (>= 4 tiles per wave), and 16 was correct when it was measured:
// against the old per-lane page lookup, a shorter scan lost at every short
// context. The scalar page-id hoist and the AS(1) global loads above made a
// tile substantially cheaper to fetch, which moved the crossover -- the fixed
// merge cost is now amortized by a much shorter scan. Neither change is worth
// anything alone; the hoist alone A/Bs to a wash. See the
// MPK_ATTN_SCALAR_PAGE comment in prefetch_tile.
//
// 8 and not 4, even though 4 is faster and 4 is the natural bound
// (physical_tile_count >= kWaveCount): a threshold change reassociates the
// online softmax over a different token partition, so it is PPL-testable, not
// hash-testable, and 4 fails that gate. Over 4 windows, paired per-position
// vs T=16: T=4 is +0.0322 nats (se 0.0060, t=+5.35, same sign in all four
// windows) against an FP-reordering yardstick of +0.0093 +/- 0.0073. T=8 is
// -0.0017 (se 0.0028, t=-0.61), i.e. inside the yardstick and if anything on
// the good side. T=8 keeps most of the speed: ctx 4096 2.148 vs 2.137 for T=4
// and 2.206 for T=16.
#ifndef MPK_ATTN_WAVE_LOCAL_MIN_TILES
#define MPK_ATTN_WAVE_LOCAL_MIN_TILES 8
#endif

namespace kernel {

using __mfma_hd64_fp16x8 = __attribute__((ext_vector_type(8))) _Float16;
using __mfma_hd64_fp16x4 = __attribute__((ext_vector_type(4))) _Float16;
using __mfma_hd64_fp32x4 = __attribute__((ext_vector_type(4))) float;

__device__ __forceinline__ __mfma_hd64_fp32x4
    __mfma_qk_hd64(__mfma_hd64_fp32x4 c, _Float16 const *a, _Float16 const *b) {
  __mfma_hd64_fp16x8 av, bv;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    av[i] = a[i];
    bv[i] = b[i];
  }
  return __builtin_amdgcn_mfma_f32_16x16x32_f16(av, bv, c, 0, 0, 0);
}

__device__ __forceinline__ __mfma_hd64_fp32x4
    __mfma_pv_hd64(__mfma_hd64_fp32x4 c, _Float16 const *a, _Float16 const *b) {
  __mfma_hd64_fp16x4 av, bv;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    av[i] = a[i];
    bv[i] = b[i];
  }
  return __builtin_amdgcn_mfma_f32_16x16x16f16(av, bv, c, 0, 0, 0);
}

__device__ __forceinline__ float __fast_exp2_hd64(float x) {
  float r;
  asm("v_exp_f32 %0, %1" : "=v"(r) : "v"(x));
  return r;
}

// Vectorized bf16 load + convert to fp16 (uint4 = 8 bf16 → 8 fp16).
__device__ __forceinline__ void
    __load_bf16x4_to_fp16(_Float16 *__restrict__ dst,
                          void const *__restrict__ src) {
  uint2 raw = *reinterpret_cast<uint2 const *>(src);
  unsigned words[2] = {raw.x, raw.y};
#pragma unroll
  for (int i = 0; i < 2; i++) {
    float lo_f, hi_f;
    asm("v_cvt_f32_bf16 %0, %1" : "=v"(lo_f) : "v"(words[i]));
    asm("v_cvt_f32_bf16 %0, %1" : "=v"(hi_f) : "v"(words[i] >> 16));
    dst[i * 2] = (_Float16)lo_f;
    dst[i * 2 + 1] = (_Float16)hi_f;
  }
}

// Load 4 bf16 as raw dword (for prefetch — conversion deferred).
__device__ __forceinline__ void
    __load_bf16x4_raw(uint2 *__restrict__ dst, void const *__restrict__ src) {
  *dst = *reinterpret_cast<uint2 const *>(src);
}

// Convert 2 raw dwords (4 bf16) → 4 fp16.
__device__ __forceinline__ void __cvt_bf16x4_to_fp16(_Float16 *__restrict__ dst,
                                                     uint2 const &raw) {
  unsigned words[2] = {raw.x, raw.y};
#pragma unroll
  for (int i = 0; i < 2; i++) {
    float lo_f, hi_f;
    asm("v_cvt_f32_bf16 %0, %1" : "=v"(lo_f) : "v"(words[i]));
    asm("v_cvt_f32_bf16 %0, %1" : "=v"(hi_f) : "v"(words[i] >> 16));
    dst[i * 2] = (_Float16)lo_f;
    dst[i * 2 + 1] = (_Float16)hi_f;
  }
}

// ============================================================================
// Wave-local scan: the long-context specialization.
//
// The baseline loop below splits the *output dimensions* across the four
// waves: every wave redundantly computes the same 2 QK MFMAs and the same
// online-softmax update for a tile, and each keeps only the 16 dims of the
// single PV MFMA it owns. K and V live in one workgroup-shared 4 KB buffer, so
// each tile costs two __syncthreads() -- one to publish tile t, one to close
// the write-after-read hazard against the t+1 overwrite.
//
// That is a good trade when a chunk is a handful of tiles. It is the wrong
// trade at long context. Fleet's chunk count saturates at 31 (MAX_KV_CHUNKS =
// workers_per_xcd / max_num_batched_requests), so past ~4k tokens the tile
// count per chunk grows linearly: 9 tiles at ctx 4096, 67 at 32768. Every one
// of those 67 iterations pays both barriers and 4x-redundant QK.
//
// Here each wave instead takes a disjoint quarter of the chunk's *tiles* and
// scans it with private K/V staging. Per 16-token tile of wave-serial time:
//
//   baseline     3 MFMA issues (2 QK + 1 PV) + 2 barriers + 1 softmax update
//   wave-local   1.5          (0.5 QK + 1 PV) + 0          + 0.25
//
// The QK redundancy is gone (each tile is scanned once, not four times), the
// barriers are gone (a wave is lockstep, so LDS write-after-read inside it is
// ordered by waitcnt alone), and the loop is 4x shorter. PV work is unchanged
// in total -- a wave now issues 4 PV MFMAs to cover all 64 dims instead of 1
// covering 16 -- which is why this is a ~2x cut in MFMA issue rather than 4x.
//
// The cost is one cross-wave merge per chunk instead of per tile, and 4x the
// LDS (16 KB staging + 9 KB merge scratch, against a 155 KB budget).
//
// Sizing: at 32k the context-dependent cost was 2.80x the KV-bandwidth
// floor, so the scan loop -- not HBM -- is what is being paid for; a
// bandwidth-bound implementation of the same shape lands near 1.4x.
//
// Eligibility: every wave must own at least one tile, and
// SLIDING_WINDOW must be off. Sliding layers cap at 128 tokens (8 tiles)
// regardless of context, so they are not where the scaling gap lives, and
// excluding them keeps the straddling-tile mask out of this path entirely.
// ============================================================================
template <int NUM_QO_PER_KV,
          int HEAD_DIM,
          int PAGE_SIZE,
          int NUM_KV_CHUNKS,
          int Q_WORKSPACE_STRIDE,
          int KV_CACHE_STRIDE,
          int NUM_KV_HEADS>
__device__ __noinline__ void
    __attn_wave_local_scan_hd64(void const *q_workspace_ptr,
                                char const *k_base,
                                char const *v_base,
                                void *output_ptr,
                                void *lse_ptr,
                                int const *kv_indices,
                                int first_page,
                                int query_start,
                                int kv_head_idx,
                                int kv_chunk_idx,
                                float scale_s,
                                int kv_start,
                                int effective_len,
                                int ntiles) {
  constexpr int KV_TILE = 16;
  constexpr int NUM_K32 = HEAD_DIM / 32; // 2
  constexpr int WAVES = 4;
  constexpr int DBLK = HEAD_DIM / 16; // 4 PV MFMAs to cover all dims

  int const tid = threadIdx.x;
  int const wave_id = tid >> 6;
  int const lane = tid & 63;
  int const midx = lane & 15; // MFMA column: q head
  int const kgrp = lane >> 4; // MFMA row group: token / dim quad

  // Wave-private staging at 4 KB stride, then merge scratch. Disjoint by
  // construction: no wave ever addresses another's K/V.
  extern __shared__ char smem_minimal[];
  _Float16 *wk = reinterpret_cast<_Float16 *>(smem_minimal + wave_id * 4096);
  _Float16 *wv =
      reinterpret_cast<_Float16 *>(smem_minimal + wave_id * 4096 + 2048);
  float *m_lds = reinterpret_cast<float *>(smem_minimal + 16384);
  float *l_lds = reinterpret_cast<float *>(smem_minimal + 16384 + 256);
  float *o_lds = reinterpret_cast<float *>(smem_minimal + 16384 + 512);

  // Q is wave-invariant; every wave loads the same 8 heads.
  _Float16 qr[NUM_K32][8];
  {
    int q_midx = (midx < NUM_QO_PER_KV) ? midx : 0;
    char const *q_ptr =
        reinterpret_cast<char const *>(q_workspace_ptr) +
        (static_cast<long>(query_start) * Q_WORKSPACE_STRIDE +
         static_cast<long>(kv_head_idx * NUM_QO_PER_KV + q_midx) * HEAD_DIM) *
            2;
#pragma unroll
    for (int kc = 0; kc < NUM_K32; kc++) {
      int dim_off = kc * 32 + kgrp * 8;
      uint4 raw = *reinterpret_cast<uint4 const *>(q_ptr + dim_off * 2);
      unsigned words[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float lo_f, hi_f;
        asm("v_cvt_f32_bf16 %0, %1" : "=v"(lo_f) : "v"(words[i]));
        asm("v_cvt_f32_bf16 %0, %1" : "=v"(hi_f) : "v"(words[i] >> 16));
        qr[kc][i * 2] = (_Float16)lo_f;
        qr[kc][i * 2 + 1] = (_Float16)hi_f;
      }
    }
  }

  // Tile range for this wave, relative to the chunk frame the caller rebased.
  int const tpw = (ntiles + WAVES - 1) / WAVES;
  int const w_first = wave_id * tpw;
  int w_last = w_first + tpw;
  if (w_last > ntiles) {
    w_last = ntiles;
  }
  // A wave can still come up empty when ntiles is not a multiple of 4 (5 tiles
  // -> tpw 2 -> wave 3 starts at 6). It contributes m=-inf, l=0 to the merge.
  int const w_ntiles = (w_first < ntiles) ? (w_last - w_first) : 0;
  int const w_kv_start = kv_start + w_first * KV_TILE;
  int w_len = w_ntiles * KV_TILE;
  {
    int const remaining = effective_len - w_first * KV_TILE;
    if (w_len > remaining) {
      w_len = remaining;
    }
    if (w_len < 0) {
      w_len = 0;
    }
  }

  // 64 lanes stage a 16x64 tile as 4 rounds of 4 tokens. Lanes 0..15 cover one
  // token's 64 dims in 8-byte strides, so each round is a 128 B contiguous
  // burst per token -- the same coalescing the 256-thread baseline load gets.
  int const ld_dim = (lane & 15) * 4;
  int const ld_tok = lane >> 4;

  auto kv_off = [&](int global_tok) -> long {
    int pid = kv_indices[first_page + global_tok / PAGE_SIZE];
    return (static_cast<long>(pid) * PAGE_SIZE * KV_CACHE_STRIDE +
            static_cast<long>(global_tok % PAGE_SIZE) * KV_CACHE_STRIDE +
            ld_dim) *
           2;
  };

  uint2 k_pre[4], v_pre[4];
  bool has_pre = false;

  // A tile's page id is uniform, and that is a provable property here, not an
  // assumption: PAGE_SIZE (4096) is a multiple of KV_TILE (16) and every tile
  // start is KV_TILE-aligned (kv_start is aligned down, and w_kv_start and
  // base are both KV_TILE multiples), so a tile can never straddle a page.
  //
  // The generic `kv_off` above did not know that. It recomputed
  // `kv_indices[...]` per lane per round, and because the result feeds the
  // *address* of the K/V load, each of the four rounds became
  //
  //     flat_load_dword  v30, v[30:31]        <-- page id, vector load
  //     s_waitcnt vmcnt(0) lgkmcnt(0)         <-- full drain
  //     flat_load_dwordx2 v[30:31], v[36:37]  <-- K
  //     flat_load_dwordx2 v[34:35], v[38:39]  <-- V
  //
  // Four *serialized* HBM round trips per tile: the drain before round i+1's
  // page id also waits on round i's K/V, so the eight payload loads could
  // never be in flight together. That is a per-tile cost, so it scales
  // linearly with context -- which is where the long-context gap lives.
  //
  // Two independent fixes below:
  //
  //  1. Hoist the page id to ONE scalar load per tile. `readfirstlane` tells
  //     the compiler the index is wave-uniform so it can use s_load and keep
  //     the base offset in SGPRs. Same value every lane, same value as before.
  //
  //  2. Load K/V through address_space(1) (GLOBAL, not generic FLAT). A
  //     flat_load bumps lgkmcnt as well as vmcnt, so the compiler cannot emit
  //     a counted partial wait and falls back to full drains -- exactly the
  //     defect docs/mpk/long-context notes hit in the split-KV merge. As
  //     global_load the eight loads issue as one queue with vmcnt(N) waits.
  //
  // Both are bit-exact: identical addresses, identical bytes, no reassociation.
  // MPK_ATTN_SCALAR_PAGE=0 ablates back to the per-lane form.
  using _kv_u32x2 = uint32_t __attribute__((ext_vector_type(2)));
  auto prefetch_tile = [&](int t) {
    int const base = t * KV_TILE;
    int tlen = w_len - base;
    if (tlen > KV_TILE) {
      tlen = KV_TILE;
    }
#if MPK_ATTN_SCALAR_PAGE
    int const tile_tok0 = w_kv_start + base;
    int const pid = __builtin_amdgcn_readfirstlane(
        kv_indices[first_page + tile_tok0 / PAGE_SIZE]);
    long const tile_off =
        (static_cast<long>(pid) * PAGE_SIZE * KV_CACHE_STRIDE +
         static_cast<long>(tile_tok0 % PAGE_SIZE) * KV_CACHE_STRIDE) *
        2;
    long const lane_off = tile_off + static_cast<long>(ld_dim) * 2;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int tok = ld_tok + i * 4;
      if (tok < tlen) {
        long const off =
            lane_off + static_cast<long>(tok) * KV_CACHE_STRIDE * 2;
        // Assign the vector type and extract components rather than casting
        // through a uint2 reference: dereferencing the AS(1)-qualified vector
        // type is precisely what makes the backend pick global_load_dwordx2
        // over flat_load_dwordx2, and a reinterpret through a generic
        // reference throws that away. Same note as
        // gang_rmsnorm_linear_mxfp4_bias_mi300.cuh:2458.
        _kv_u32x2 const kv =
            *(__attribute__((address_space(1))) _kv_u32x2 const *)(k_base +
                                                                   off);
        _kv_u32x2 const vv =
            *(__attribute__((address_space(1))) _kv_u32x2 const *)(v_base +
                                                                   off);
        k_pre[i] = make_uint2(kv[0], kv[1]);
        v_pre[i] = make_uint2(vv[0], vv[1]);
      } else {
        k_pre[i] = make_uint2(0u, 0u);
        v_pre[i] = make_uint2(0u, 0u);
      }
    }
#else
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int tok = ld_tok + i * 4;
      if (tok < tlen) {
        long off = kv_off(w_kv_start + base + tok);
        __load_bf16x4_raw(&k_pre[i], k_base + off);
        __load_bf16x4_raw(&v_pre[i], v_base + off);
      } else {
        k_pre[i] = make_uint2(0u, 0u);
        v_pre[i] = make_uint2(0u, 0u);
      }
    }
#endif
    has_pre = true;
  };

  // Padding rows must be zeroed, not left stale: a partial tail tile otherwise
  // reuses the previous tile's K/V for the masked lanes. The score mask covers
  // QK, but PV consumes V for every row.
  auto commit_prefetch = [&]() {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int tok = ld_tok + i * 4;
      _Float16 kf[4], vf[4];
      __cvt_bf16x4_to_fp16(kf, k_pre[i]);
      __cvt_bf16x4_to_fp16(vf, v_pre[i]);
      *(uint64_t *)&wk[tok * HEAD_DIM + ld_dim] = *(uint64_t *)kf;
      *(uint64_t *)&wv[tok * HEAD_DIM + ld_dim] = *(uint64_t *)vf;
    }
  };

  if (w_ntiles > 0) {
    prefetch_tile(0);
    commit_prefetch();
    has_pre = false;
    if (w_ntiles > 1) {
      prefetch_tile(1);
    }
  }

  __mfma_hd64_fp32x4 o_acc[DBLK];
#pragma unroll
  for (int d = 0; d < DBLK; d++) {
    o_acc[d] = {0, 0, 0, 0};
  }
  float m_running = -INFINITY;
  float l_head[4] = {0, 0, 0, 0};

  for (int t = 0; t < w_ntiles; t++) {
    int const tile_start = t * KV_TILE;
    int tile_len = w_len - tile_start;
    if (tile_len > KV_TILE) {
      tile_len = KV_TILE;
    }

    // No s_barrier here. wk/wv are private to this wave, and a wavefront is
    // lockstep, so the compiler's lgkmcnt wait is the whole ordering
    // requirement. wave_barrier() emits no instruction; it only stops the
    // scheduler from hoisting the tile t+1 stores above these loads.
    __builtin_amdgcn_wave_barrier();

    _Float16 kr[NUM_K32][8];
#pragma unroll
    for (int kc = 0; kc < NUM_K32; kc++) {
      _Float16 const *k_ptr = &wk[midx * HEAD_DIM + kc * 32 + kgrp * 8];
#pragma unroll
      for (int i = 0; i < 8; i++) {
        kr[kc][i] = k_ptr[i];
      }
    }

    __mfma_hd64_fp16x4 va[DBLK];
#pragma unroll
    for (int d = 0; d < DBLK; d++) {
      _Float16 const *v_ptr = &wv[(kgrp * 4) * HEAD_DIM + d * 16 + midx];
      va[d][0] = v_ptr[0 * HEAD_DIM];
      va[d][1] = v_ptr[1 * HEAD_DIM];
      va[d][2] = v_ptr[2 * HEAD_DIM];
      va[d][3] = v_ptr[3 * HEAD_DIM];
    }

    __builtin_amdgcn_wave_barrier();

    __mfma_hd64_fp32x4 scores = {0, 0, 0, 0};
#pragma unroll
    for (int kc = 0; kc < NUM_K32; kc++) {
      scores = __mfma_qk_hd64(scores, kr[kc], qr[kc]);
    }

    scores[0] *= scale_s;
    scores[1] *= scale_s;
    scores[2] *= scale_s;
    scores[3] *= scale_s;

#pragma unroll
    for (int h = 0; h < 4; h++) {
      if (kgrp * 4 + h >= tile_len) {
        scores[h] = -INFINITY;
      }
    }

    // Online softmax. Identical arithmetic and identical reduction order to
    // the baseline loop, so a wave's partial is bit-comparable to the same
    // token range scanned there.
    float tile_max =
        fmaxf(fmaxf(scores[0], scores[1]), fmaxf(scores[2], scores[3]));
    {
      float a = tile_max, b = tile_max;
      asm volatile("s_nop 1\n\tv_permlane32_swap_b32_e32 %0, %1"
                   : "+v"(a), "+v"(b));
      tile_max = fmaxf(a, b);
      a = tile_max;
      b = tile_max;
      asm volatile("s_nop 1\n\tv_permlane16_swap_b32_e32 %0, %1"
                   : "+v"(a), "+v"(b));
      tile_max = fmaxf(a, b);
    }

    float new_max = fmaxf(m_running, tile_max);
    float rescale =
        (m_running == -INFINITY) ? 0.0f : __fast_exp2_hd64(m_running - new_max);

#pragma unroll
    for (int d = 0; d < DBLK; d++) {
      o_acc[d][0] *= rescale;
      o_acc[d][1] *= rescale;
      o_acc[d][2] *= rescale;
      o_acc[d][3] *= rescale;
    }

    float w0 = __fast_exp2_hd64(scores[0] - new_max);
    float w1 = __fast_exp2_hd64(scores[1] - new_max);
    float w2 = __fast_exp2_hd64(scores[2] - new_max);
    float w3 = __fast_exp2_hd64(scores[3] - new_max);

    l_head[0] = l_head[0] * rescale + w0;
    l_head[1] = l_head[1] * rescale + w1;
    l_head[2] = l_head[2] * rescale + w2;
    l_head[3] = l_head[3] * rescale + w3;
    m_running = new_max;

    // Tile t is fully register-resident now, so staging t+1 is free to run
    // ahead of the PV MFMAs below.
    if (has_pre) {
      commit_prefetch();
      has_pre = false;
    }

    __mfma_hd64_fp16x4 pb;
    pb[0] = (_Float16)w0;
    pb[1] = (_Float16)w1;
    pb[2] = (_Float16)w2;
    pb[3] = (_Float16)w3;
#pragma unroll
    for (int d = 0; d < DBLK; d++) {
      o_acc[d] = __mfma_pv_hd64(
          o_acc[d], (_Float16 const *)&va[d], (_Float16 const *)&pb);
    }

    if (t + 2 < w_ntiles) {
      prefetch_tile(t + 2);
    }
  }

  // ===== Cross-wave merge: once per chunk, not once per tile =====
  float l_sum = l_head[0] + l_head[1] + l_head[2] + l_head[3];
  l_sum += __shfl_xor(l_sum, 16);
  l_sum += __shfl_xor(l_sum, 32);

  if (kgrp == 0 && midx < NUM_QO_PER_KV) {
    m_lds[wave_id * 16 + midx] = m_running;
    l_lds[wave_id * 16 + midx] = l_sum;
  }
  if (midx < NUM_QO_PER_KV) {
    float *dst = o_lds + (wave_id * NUM_QO_PER_KV + midx) * HEAD_DIM;
#pragma unroll
    for (int d = 0; d < DBLK; d++) {
#pragma unroll
      for (int h = 0; h < 4; h++) {
        dst[d * 16 + kgrp * 4 + h] = o_acc[d][h];
      }
    }
  }
  __syncthreads();

  constexpr int LSE_S = NUM_KV_HEADS * NUM_KV_CHUNKS * NUM_QO_PER_KV;
  constexpr int O_S = LSE_S * HEAD_DIM;
  float *o_out = reinterpret_cast<float *>(output_ptr) +
                 static_cast<long>(query_start) * O_S +
                 static_cast<long>(kv_head_idx) * NUM_KV_CHUNKS *
                     NUM_QO_PER_KV * HEAD_DIM +
                 static_cast<long>(kv_chunk_idx) * NUM_QO_PER_KV * HEAD_DIM;

  // 512 (q head, dim) outputs over 256 threads. The denominator is rebuilt per
  // element rather than staged behind a second barrier -- four FMAs against an
  // s_barrier is not a close call.
  constexpr int NOUT = NUM_QO_PER_KV * HEAD_DIM;
#pragma unroll 1
  for (int idx = tid; idx < NOUT; idx += 256) {
    int const q = idx / HEAD_DIM;
    int const d = idx % HEAD_DIM;
    float mg = -INFINITY;
#pragma unroll
    for (int w = 0; w < WAVES; w++) {
      mg = fmaxf(mg, m_lds[w * 16 + q]);
    }
    float num = 0.0f, den = 0.0f;
#pragma unroll
    for (int w = 0; w < WAVES; w++) {
      float const mw = m_lds[w * 16 + q];
      // Empty waves carry m=-inf; -inf minus -inf is NaN, so gate rather than
      // relying on exp2 to flush it.
      float const s = (mw == -INFINITY) ? 0.0f : __fast_exp2_hd64(mw - mg);
      num += o_lds[(w * NUM_QO_PER_KV + q) * HEAD_DIM + d] * s;
      den += l_lds[w * 16 + q] * s;
    }
    o_out[q * HEAD_DIM + d] = (den > 0.0f) ? (num / den) : 0.0f;
  }

  if (tid < NUM_QO_PER_KV) {
    int const q = tid;
    float mg = -INFINITY;
#pragma unroll
    for (int w = 0; w < WAVES; w++) {
      mg = fmaxf(mg, m_lds[w * 16 + q]);
    }
    float den = 0.0f;
#pragma unroll
    for (int w = 0; w < WAVES; w++) {
      float const mw = m_lds[w * 16 + q];
      float const s = (mw == -INFINITY) ? 0.0f : __fast_exp2_hd64(mw - mg);
      den += l_lds[w * 16 + q] * s;
    }
    // Natural-log LSE, matching the baseline epilogue's units exactly. See the
    // long comment there for why this conversion is not optional.
    constexpr float INV_LOG2E = 0.693147180559945309417f;
    float *lse_out = reinterpret_cast<float *>(lse_ptr) +
                     static_cast<long>(query_start) * LSE_S +
                     kv_head_idx * NUM_KV_CHUNKS * NUM_QO_PER_KV +
                     kv_chunk_idx * NUM_QO_PER_KV + q;
    *lse_out =
        (den > 0.0f) ? ((mg + __log2f(den)) * INV_LOG2E) : -1e30f;
  }
}

template <typename T,
          int NUM_QO_PER_KV,
          int HEAD_DIM,
          int PAGE_SIZE,
          int MAX_SEQ_LEN,
          int NUM_KV_CHUNKS,
          int Q_WORKSPACE_STRIDE,
          int KV_CACHE_STRIDE,
          int NUM_KV_HEADS>
__device__ __noinline__ void
    paged_attention_minimal_decode_hd64(void const *q_workspace_ptr,
                                        void *paged_k_cache_ptr,
                                        void *paged_v_cache_ptr,
                                        void *output_ptr,
                                        void *lse_ptr,
                                        int const *qo_indptr,
                                        int const *kv_indptr,
                                        int const *kv_indices,
                                        int const *kv_last_page_len,
                                        int16_t request_id,
                                        int kv_head_idx,
                                        int kv_chunk_idx,
                                        float scale_s,
                                        int sliding_window = 0,
                                        void const *sinks_ptr = nullptr) {
  using bf16 = __hip_bfloat16;
  static_assert(HEAD_DIM == 64, "This kernel is HD=64 only");

  int const req = request_id;
  int const query_start = qo_indptr[req];
  if (query_start == qo_indptr[req + 1]) {
    return;
  }

  int const first_page = kv_indptr[req];
  int const num_pages = kv_indptr[req + 1] - first_page;
  int const seqlen_k = (num_pages - 1) * PAGE_SIZE + kv_last_page_len[req];

  int const tid = threadIdx.x;
  int const warp_id = tid / 64;
  int const lane = tid & 63;
  int const midx = lane & 15;
  int const kgrp = lane >> 4;

  constexpr int KV_TILE = 16;
  constexpr int NUM_K32 = HEAD_DIM / 32; // 2

  char const *k_base = reinterpret_cast<char const *>(paged_k_cache_ptr);
  char const *v_base = reinterpret_cast<char const *>(paged_v_cache_ptr);

  // LDS: K[16][64] + V[16][64] = 4 KB total
  extern __shared__ char smem_minimal[];
  _Float16 *lds_k = reinterpret_cast<_Float16 *>(smem_minimal);
  _Float16 *lds_v = reinterpret_cast<_Float16 *>(smem_minimal + 2048);

  // 256 threads × 4 bf16 = 16 tok × 64 dim = 1024 elements
  int my_tok = tid / 16;
  int my_dim = (tid % 16) * 4;

  // Load Q (lanes >= NUM_QO_PER_KV map to head 0; output is guarded)
  _Float16 qr[NUM_K32][8];
  {
    int q_midx = (midx < NUM_QO_PER_KV) ? midx : 0;
    char const *q_ptr =
        reinterpret_cast<char const *>(q_workspace_ptr) +
        (static_cast<long>(query_start) * Q_WORKSPACE_STRIDE +
         static_cast<long>(kv_head_idx * NUM_QO_PER_KV + q_midx) * HEAD_DIM) *
            2;
#pragma unroll
    for (int kc = 0; kc < NUM_K32; kc++) {
      int dim_off = kc * 32 + kgrp * 8;
      // 8 bf16 = uint4 load
      uint4 raw = *reinterpret_cast<uint4 const *>(q_ptr + dim_off * 2);
      unsigned words[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float lo_f, hi_f;
        asm("v_cvt_f32_bf16 %0, %1" : "=v"(lo_f) : "v"(words[i]));
        asm("v_cvt_f32_bf16 %0, %1" : "=v"(hi_f) : "v"(words[i] >> 16));
        qr[kc][i * 2] = (_Float16)lo_f;
        qr[kc][i * 2 + 1] = (_Float16)hi_f;
      }
    }
  }

  // Sliding window: skip KV positions before the window.
  // kv_start is aligned down to KV_TILE for clean tiling.
  int kv_start = 0;
  if (sliding_window > 0 && seqlen_k > sliding_window) {
    kv_start = ((seqlen_k - sliding_window) / KV_TILE) * KV_TILE;
  }
  int effective_len = seqlen_k - kv_start;

  int ntiles = (effective_len + KV_TILE - 1) / KV_TILE;

  // Split-KV partitioning: each chunk_idx processes a contiguous slice of
  // tiles. For NUM_KV_CHUNKS==1 this is a no-op (chunk_first=0,
  // chunk_last=ntiles). For >1 chunks, an empty chunk writes LSE=-inf and exits
  // so the merge step gives it zero weight (and the prior-iteration garbage in
  // the LSE slot does not contaminate the global softmax).
  int chunk_first_tile = 0;
  int chunk_last_tile = ntiles;
  if constexpr (NUM_KV_CHUNKS > 1) {
    int tiles_per_chunk = (ntiles + NUM_KV_CHUNKS - 1) / NUM_KV_CHUNKS;
    chunk_first_tile = kv_chunk_idx * tiles_per_chunk;
    chunk_last_tile = chunk_first_tile + tiles_per_chunk;
    if (chunk_last_tile > ntiles) {
      chunk_last_tile = ntiles;
    }
    if (chunk_first_tile >= ntiles) {
      // Empty chunk: stamp LSE=-inf for all q heads in this chunk slot.
      // (Output buffer values are weighted by exp(LSE - m_global) = 0, so they
      // don't matter, but LSE must be -inf or the merge picks up junk.)
      if (warp_id == 0 && kgrp == 0 && midx < NUM_QO_PER_KV) {
        constexpr int LSE_STRIDE = NUM_KV_HEADS * NUM_KV_CHUNKS * NUM_QO_PER_KV;
        float *lse_out = reinterpret_cast<float *>(lse_ptr) +
                         static_cast<long>(query_start) * LSE_STRIDE +
                         kv_head_idx * NUM_KV_CHUNKS * NUM_QO_PER_KV +
                         kv_chunk_idx * NUM_QO_PER_KV + midx;
        *lse_out = -1e30f;
      }
      return;
    }
    kv_start += chunk_first_tile * KV_TILE;
    effective_len = (chunk_last_tile - chunk_first_tile) * KV_TILE;
    // Clamp to remaining KV in the global window.
    int remaining = (seqlen_k - kv_start);
    if (effective_len > remaining) {
      effective_len = remaining;
    }
    ntiles = chunk_last_tile - chunk_first_tile;
  }
  if (ntiles == 0) {
    return;
  }

  // Long-context path. Eligibility is deliberately conservative:
  //
  //   ntiles >= MPK_ATTN_WAVE_LOCAL_MIN_TILES (8)
  //                      every wave owns >= 2 tiles.
  //
  //                      This threshold was 16 (>= 4 tiles per wave) and the
  //                      measurement behind that is worth keeping, because it
  //                      is a clean example of a tuned constant expiring. On
  //                      the per-lane page-lookup cost model, 4 was a wash or
  //                      a loss at short context -- ctx 512 (1 tile/wave)
  //                      +0.003 ms, 1024 (2) +0.034, 2048 (2) -0.006, 4096
  //                      (3) +0.014 -- and only won once the scan amortized
  //                      the one-shot merge: 8192 -0.048, 32768 -0.305. The
  //                      merge costs a __syncthreads() plus a 512-element
  //                      weighted reduction whether the scan is 1 tile or 17.
  //
  //                      The scalar page-id hoist and AS(1) global K/V loads
  //                      cut the per-tile fetch cost, which moved the
  //                      crossover: re-measured ctx 4096 T=4/8/12 ->
  //                      2.137/2.153/2.194 ms. The merge is unchanged and the
  //                      scan got cheaper, which is the direction that makes a
  //                      shorter scan worth entering. 8 rather than the faster
  //                      4 because 4 fails the perplexity gate -- see the
  //                      MPK_ATTN_WAVE_LOCAL_MIN_TILES definition.
  //   sliding_window==0  sliding layers see at most 128 tokens (8 tiles) no
  //                      matter the context, so they contribute nothing to the
  //                      scaling gap, and skipping them keeps the
  //                      straddling-first-tile mask out of this path.
  //   sinks_ptr==nullptr the per-chunk call never passes sinks (they are
  //                      applied in the merge); this is an assertion, not a
  //                      restriction.
  //
  // Set MPK_ATTN_NO_WAVE_LOCAL to ablate back to the shared-LDS loop.
#ifndef MPK_ATTN_NO_WAVE_LOCAL
  if constexpr (NUM_KV_CHUNKS > 1) {
    if (ntiles >= MPK_ATTN_WAVE_LOCAL_MIN_TILES && sliding_window == 0 &&
        sinks_ptr == nullptr) {
      __attn_wave_local_scan_hd64<NUM_QO_PER_KV,
                                  HEAD_DIM,
                                  PAGE_SIZE,
                                  NUM_KV_CHUNKS,
                                  Q_WORKSPACE_STRIDE,
                                  KV_CACHE_STRIDE,
                                  NUM_KV_HEADS>(q_workspace_ptr,
                                                k_base,
                                                v_base,
                                                output_ptr,
                                                lse_ptr,
                                                kv_indices,
                                                first_page,
                                                query_start,
                                                kv_head_idx,
                                                kv_chunk_idx,
                                                scale_s,
                                                kv_start,
                                                effective_len,
                                                ntiles);
      return;
    }
  }
#endif

  // KV byte offset (bf16 = 2 bytes per element). KV cache stride per token =
  // KV_CACHE_STRIDE. Note: the KV pointers passed in are already offset by
  // kv_head_idx * HEAD_DIM, matching the existing CK FMHA decode convention
  // (see gang_attention_mi300.cuh).
  auto get_kv_off = [&](int global_tok) -> long {
    int pid = kv_indices[first_page + global_tok / PAGE_SIZE];
    return (static_cast<long>(pid) * PAGE_SIZE * KV_CACHE_STRIDE +
            static_cast<long>(global_tok % PAGE_SIZE) * KV_CACHE_STRIDE +
            my_dim) *
           2;
  };

  // Tile-uniform variant of get_kv_off. Same provable invariant as in the
  // wave-local scan: PAGE_SIZE (4096) is a multiple of KV_TILE (16) and every
  // tile start is KV_TILE-aligned, so all 16 tokens of a tile share one page
  // and the lookup is wave-uniform. `readfirstlane` is what tells the backend
  // that, so the page id becomes an s_load into an SGPR instead of a
  // per-lane vector load whose result feeds the K/V address -- which is what
  // forced `s_waitcnt vmcnt(0) lgkmcnt(0)` between the page load and the
  // payload load, serializing them.
  //
  // This is the loop that runs at ctx 4096: 256 tiles / 31 chunks = 9 tiles
  // per chunk, below the wave-local path's ntiles>=16 threshold, so this is
  // the path the hoist pays off on -- not the wave-local variant.
  //
  // Bit-exact: identical page id, identical address, identical bytes.
  auto get_kv_tile_off = [&](int tile_tok0) -> long {
    int const pid = __builtin_amdgcn_readfirstlane(
        kv_indices[first_page + tile_tok0 / PAGE_SIZE]);
    return (static_cast<long>(pid) * PAGE_SIZE * KV_CACHE_STRIDE +
            static_cast<long>(tile_tok0 % PAGE_SIZE) * KV_CACHE_STRIDE) *
           2;
  };
  // Per-lane byte offset within the tile, loop-invariant.
  long const lane_kv_off =
      static_cast<long>(my_tok) * KV_CACHE_STRIDE * 2 +
      static_cast<long>(my_dim) * 2;
  using _kv_u32x2 = uint32_t __attribute__((ext_vector_type(2)));
  // AS(1) == GLOBAL, not generic FLAT. A flat_load bumps lgkmcnt as well as
  // vmcnt, so the compiler cannot emit a counted partial wait and falls back
  // to full drains; as global_load the K and V loads issue as one queue.
  // Assign the vector type and extract -- a reinterpret through a generic
  // uint2 reference throws the address space away and reverts to flat_load.
  auto load_kv_pair = [&](uint2 *kd, uint2 *vd, long off) {
    _kv_u32x2 const kv =
        *(__attribute__((address_space(1))) _kv_u32x2 const *)(k_base + off);
    _kv_u32x2 const vv =
        *(__attribute__((address_space(1))) _kv_u32x2 const *)(v_base + off);
    *kd = make_uint2(kv[0], kv[1]);
    *vd = make_uint2(vv[0], vv[1]);
  };

  // ===== PROLOGUE: K[0]+V[0] -> LDS =====
  int tile0_len = (KV_TILE < effective_len) ? KV_TILE : effective_len;
  if (my_tok < tile0_len) {
    long kv0 = get_kv_off(kv_start + my_tok);
    _Float16 k_fp[4], v_fp[4];
    __load_bf16x4_to_fp16(k_fp, k_base + kv0);
    __load_bf16x4_to_fp16(v_fp, v_base + kv0);
    *(uint64_t *)&lds_k[my_tok * HEAD_DIM + my_dim] = *(uint64_t *)k_fp;
    *(uint64_t *)&lds_v[my_tok * HEAD_DIM + my_dim] = *(uint64_t *)v_fp;
  } else {
    _Float16 zero[4] = {0, 0, 0, 0};
    *(uint64_t *)&lds_k[my_tok * HEAD_DIM + my_dim] = *(uint64_t *)zero;
    *(uint64_t *)&lds_v[my_tok * HEAD_DIM + my_dim] = *(uint64_t *)zero;
  }

  // Prefetch K[1]+V[1] -> registers as raw dwords
  uint2 k_pre, v_pre;
  bool has_pre = false;
  if (ntiles > 1) {
    int tile1_len = ((effective_len - KV_TILE) < KV_TILE)
                        ? (effective_len - KV_TILE)
                        : KV_TILE;
    if (my_tok < tile1_len) {
#if MPK_ATTN_SCALAR_PAGE
      long kv1 = get_kv_tile_off(kv_start + KV_TILE) + lane_kv_off;
      load_kv_pair(&k_pre, &v_pre, kv1);
#else
      long kv1 = get_kv_off(kv_start + KV_TILE + my_tok);
      __load_bf16x4_raw(&k_pre, k_base + kv1);
      __load_bf16x4_raw(&v_pre, v_base + kv1);
#endif
      has_pre = true;
    }
  }

  // ===== MAIN LOOP =====
  __mfma_hd64_fp32x4 o_acc = {0, 0, 0, 0};
  float m_running = -INFINITY;
  float l_head[4] = {0, 0, 0, 0};

  for (int t = 0; t < ntiles; t++) {
    int tile_start = t * KV_TILE;
    int tile_len = ((effective_len - tile_start) < KV_TILE)
                       ? (effective_len - tile_start)
                       : KV_TILE;

    __syncthreads();

    // Read K[t] from LDS — lane (midx, kgrp) needs
    // K[tok=midx][dim=kc*32+kgrp*8..+7]
    _Float16 kr[NUM_K32][8];
#pragma unroll
    for (int kc = 0; kc < NUM_K32; kc++) {
      _Float16 const *k_ptr = &lds_k[midx * HEAD_DIM + kc * 32 + kgrp * 8];
#pragma unroll
      for (int i = 0; i < 8; i++) {
        kr[kc][i] = k_ptr[i];
      }
    }

    // Read V[t] from LDS — lane (midx, kgrp) needs
    // V[dim=warp*16+midx][tok=kgrp*4..kgrp*4+3] (transposed for PV MFMA where
    // A=V).
    __mfma_hd64_fp16x4 va;
    {
      _Float16 const *v_ptr =
          &lds_v[(kgrp * 4) * HEAD_DIM + warp_id * 16 + midx];
      va[0] = v_ptr[0 * HEAD_DIM];
      va[1] = v_ptr[1 * HEAD_DIM];
      va[2] = v_ptr[2 * HEAD_DIM];
      va[3] = v_ptr[3 * HEAD_DIM];
    }

    // Every warp has now copied tile t out of LDS into registers (kr, va).
    // Later in this same iteration each thread overwrites lds_k/lds_v with
    // tile t+1 from the prefetch registers. The __syncthreads() at the top of
    // the loop only orders those writes against the *next* iteration's reads;
    // nothing ordered them against *this* iteration's reads, so a warp that
    // ran ahead to the write could clobber a tile a slower warp had not
    // finished reading. That is a genuine cross-warp write-after-read race on
    // LDS, and it silently corrupts K/V for the lagging warp.
    //
    // This barrier closes the read half of the double-buffer-free pipeline.
    // It only fires when the prefetch is live (ntiles > 1); with one tile per
    // chunk the writes are dead and this costs a single s_barrier.
    __syncthreads();

    // 2x QK MFMA
    __mfma_hd64_fp32x4 scores = {0, 0, 0, 0};
#pragma unroll
    for (int kc = 0; kc < NUM_K32; kc++) {
      scores = __mfma_qk_hd64(scores, kr[kc], qr[kc]);
    }

    scores[0] *= scale_s;
    scores[1] *= scale_s;
    scores[2] *= scale_s;
    scores[3] *= scale_s;

#pragma unroll
    for (int h = 0; h < 4; h++) {
      if (kgrp * 4 + h >= tile_len) {
        scores[h] = -INFINITY;
      }
    }

    // Sliding-window head mask.
    //
    // kv_start is aligned DOWN to a KV_TILE boundary so the tiling stays
    // clean, which means the first tile can begin up to KV_TILE-1 tokens
    // BEFORE the window opens. Without this mask the kernel attends over a
    // window of sliding_window..sliding_window+15 tokens instead of exactly
    // sliding_window -- e.g. at seqlen 143 with a 128 window it attends to all
    // 143 tokens. The reference masks strictly at (row - col) >= window
    // (models/modeling_gpt_oss.py), so those extra tokens are pure divergence,
    // and GPT-OSS applies sliding attention on 18 of its 36 layers.
    //
    // Only the first tile of the window can straddle the boundary, so this is
    // a predicated no-op everywhere else.
#ifndef MPK_NO_SW_MASK
    if (sliding_window > 0) {
      int const win_first = seqlen_k - sliding_window; // first allowed token
      if (win_first > 0) {
#pragma unroll
        for (int h = 0; h < 4; h++) {
          if (kv_start + tile_start + kgrp * 4 + h < win_first) {
            scores[h] = -INFINITY;
          }
        }
      }
    }
#endif

    // Online softmax
    float tile_max =
        fmaxf(fmaxf(scores[0], scores[1]), fmaxf(scores[2], scores[3]));
    {
      float a = tile_max, b = tile_max;
      asm volatile("s_nop 1\n\tv_permlane32_swap_b32_e32 %0, %1"
                   : "+v"(a), "+v"(b));
      tile_max = fmaxf(a, b);
      a = tile_max;
      b = tile_max;
      asm volatile("s_nop 1\n\tv_permlane16_swap_b32_e32 %0, %1"
                   : "+v"(a), "+v"(b));
      tile_max = fmaxf(a, b);
    }

    float new_max = fmaxf(m_running, tile_max);
    float rescale =
        (m_running == -INFINITY) ? 0.0f : __fast_exp2_hd64(m_running - new_max);

    o_acc[0] *= rescale;
    o_acc[1] *= rescale;
    o_acc[2] *= rescale;
    o_acc[3] *= rescale;

    float w0 = __fast_exp2_hd64(scores[0] - new_max);
    float w1 = __fast_exp2_hd64(scores[1] - new_max);
    float w2 = __fast_exp2_hd64(scores[2] - new_max);
    float w3 = __fast_exp2_hd64(scores[3] - new_max);

    l_head[0] = l_head[0] * rescale + w0;
    l_head[1] = l_head[1] * rescale + w1;
    l_head[2] = l_head[2] * rescale + w2;
    l_head[3] = l_head[3] * rescale + w3;
    m_running = new_max;

    // Write V[t+1] from v_pre
    if (has_pre) {
      _Float16 v_fp[4];
      __cvt_bf16x4_to_fp16(v_fp, v_pre);
      *(uint64_t *)&lds_v[my_tok * HEAD_DIM + my_dim] = *(uint64_t *)v_fp;
    }

    // PV MFMA (1x for HD=64; warp covers 16 dims)
    __mfma_hd64_fp16x4 pb;
    pb[0] = (_Float16)w0;
    pb[1] = (_Float16)w1;
    pb[2] = (_Float16)w2;
    pb[3] = (_Float16)w3;
    o_acc = __mfma_pv_hd64(o_acc, (_Float16 const *)&va, (_Float16 const *)&pb);

    // Write K[t+1] from k_pre
    if (has_pre) {
      _Float16 k_fp[4];
      __cvt_bf16x4_to_fp16(k_fp, k_pre);
      *(uint64_t *)&lds_k[my_tok * HEAD_DIM + my_dim] = *(uint64_t *)k_fp;
    }

    // Prefetch K[t+2]+V[t+2]
    has_pre = false;
    if (t + 2 < ntiles) {
      int t2_start = (t + 2) * KV_TILE;
      int t2_len = ((effective_len - t2_start) < KV_TILE)
                       ? (effective_len - t2_start)
                       : KV_TILE;
      if (my_tok < t2_len) {
#if MPK_ATTN_SCALAR_PAGE
        long kv2 = get_kv_tile_off(kv_start + t2_start) + lane_kv_off;
        load_kv_pair(&k_pre, &v_pre, kv2);
#else
        long kv2 = get_kv_off(kv_start + t2_start + my_tok);
        __load_bf16x4_raw(&k_pre, k_base + kv2);
        __load_bf16x4_raw(&v_pre, v_base + kv2);
#endif
        has_pre = true;
      }
    }
  }

  // ===== Output =====
  // Reduce l_head across lanes within MFMA-M slot (kgrp lanes 16..63 hold
  // partials).
  float l_sum = l_head[0] + l_head[1] + l_head[2] + l_head[3];
  l_sum += __shfl_xor(l_sum, 16);
  l_sum += __shfl_xor(l_sum, 32);

  if (midx < NUM_QO_PER_KV) {
    int q_head_local = midx;
    {
      // Split-KV partial output: float + LSE for later merge.
      //
      // This layout is used for EVERY NUM_KV_CHUNKS, including 1. There used
      // to be a `if constexpr (NUM_KV_CHUNKS == 1)` fast path here that wrote
      // *bf16* directly at Q_WORKSPACE_STRIDE and skipped the merge. It was
      // silently wrong: the caller allocates this buffer (ck_fmha_o_acc) as
      // f32 with stride O_S regardless of chunk count, and the caller's merge
      // gate is `(chunk % NUM_KV_CHUNKS) == NUM_KV_CHUNKS - 1`, which at
      // NUM_KV_CHUNKS==1 is `% 1 == 0` -- always true. So the merge ran anyway
      // and reinterpreted the bf16 bytes as f32. That is the whole explanation
      // for CK_FMHA_NUM_KV_CHUNKS=1 scoring PPL 128522 while 8 scores 282.
      //
      // At NUM_KV_CHUNKS==1 the merge is a mathematical identity (one chunk to
      // combine), so taking the split path unconditionally is correct, just
      // marginally slower than the deleted fast path would have been had it
      // worked. Shipping config is 8 chunks, so this costs nothing in practice.
      constexpr int LSE_S = NUM_KV_HEADS * NUM_KV_CHUNKS * NUM_QO_PER_KV;
      constexpr int O_S = LSE_S * HEAD_DIM;
      float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
      float *o = reinterpret_cast<float *>(output_ptr) +
                 static_cast<long>(query_start) * O_S +
                 static_cast<long>(kv_head_idx) * NUM_KV_CHUNKS *
                     NUM_QO_PER_KV * HEAD_DIM +
                 static_cast<long>(kv_chunk_idx) * NUM_QO_PER_KV * HEAD_DIM +
                 static_cast<long>(q_head_local) * HEAD_DIM;
#pragma unroll
      for (int h = 0; h < 4; h++) {
        int dim_offset = warp_id * 16 + kgrp * 4 + h;
        o[dim_offset] = o_acc[h] * inv_l;
      }
    }

    // Write LSE (for sink correction or split-KV merge).
    // Only one lane per (q_head_local) writes — pick lane warp_id==0, kgrp==0.
    if (warp_id == 0 && kgrp == 0) {
      constexpr int LSE_STRIDE = NUM_KV_HEADS * NUM_KV_CHUNKS * NUM_QO_PER_KV;
      float *lse_out = reinterpret_cast<float *>(lse_ptr) +
                       static_cast<long>(query_start) * LSE_STRIDE +
                       kv_head_idx * NUM_KV_CHUNKS * NUM_QO_PER_KV +
                       kv_chunk_idx * NUM_QO_PER_KV + q_head_local;
      // LSE in NATURAL log, which is what the merge expects (it multiplies by
      // log2(e) on load).
      //
      // The units here are easy to get wrong. `scale_s` is log2(e)/sqrt(HD)
      // (0.180337 for HD=64, not 0.125), so the scores, `m_running` and the
      // exp2-based `l_sum` are all in LOG2 space. The natural-log LSE is
      // therefore m_running/log2(e) + ln(l_sum), i.e. (m_running +
      // log2(l_sum)) / log2(e).
      //
      // This used to write `m_running + logf(l_sum)` -- a log2 exponent added
      // to a natural-log mantissa. The merge scales that by log2(e) and gets
      // log2(e)*m + log2(l) where it needs m + log2(l), so every LSE was too
      // large by 0.4427*m_running. Two consequences, both silent:
      //   1. Chunk merge weights are exp2 of *differences* of these values, so
      //      the inter-chunk spread is stretched by log2(e) and chunks with a
      //      higher local max are over-weighted.
      //   2. The sink correction is 1/(1 + exp2(sink_log2 - lse_log2)); an
      //      inflated lse drives that toward 1, under-applying the sink and
      //      inflating the attention output (measured: ~5.6% norm inflation,
      //      8.2% error against the reference at layer 0).
      constexpr float INV_LOG2E = 0.693147180559945309417f; // ln(2)
#ifdef MPK_LSE_LOG_BUG
      // Deliberately reintroduce the pre-49f446b unit bug. This exists so the
      // layer-comparison test can be shown to FAIL on a known-bad kernel --
      // a correctness gate nobody has seen go red is not known to work.
      float lse_val = (l_sum > 0.0f) ? (m_running + logf(l_sum)) : -1e30f;
#else
      float lse_val =
          (l_sum > 0.0f) ? ((m_running + __log2f(l_sum)) * INV_LOG2E) : -1e30f;
#endif
      *lse_out = lse_val;
    }
  }
}

} // namespace kernel
