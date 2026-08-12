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
      long kv1 = get_kv_off(kv_start + KV_TILE + my_tok);
      __load_bf16x4_raw(&k_pre, k_base + kv1);
      __load_bf16x4_raw(&v_pre, v_base + kv1);
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
        long kv2 = get_kv_off(kv_start + t2_start + my_tok);
        __load_bf16x4_raw(&k_pre, k_base + kv2);
        __load_bf16x4_raw(&v_pre, v_base + kv2);
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
    if constexpr (NUM_KV_CHUNKS == 1) {
      // Direct bf16 output
      float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
      // Fuse per-head attention sink correction:
      //   out_h *= sigmoid(LSE_h - sink_h) = 1 / (1 + exp(sink_h - LSE_h))
      // This eliminates the standalone attention_sink_layer task for decode.
      if (sinks_ptr != nullptr) {
        float lse_val = (l_sum > 0.0f) ? (m_running + logf(l_sum)) : -1e30f;
        bf16 const *d_sinks = reinterpret_cast<bf16 const *>(sinks_ptr);
        float sink_val = static_cast<float>(
            d_sinks[kv_head_idx * NUM_QO_PER_KV + q_head_local]);
        float correction = 1.0f / (1.0f + expf(sink_val - lse_val));
        inv_l *= correction;
      }
      bf16 *o = reinterpret_cast<bf16 *>(output_ptr) +
                static_cast<long>(query_start) * Q_WORKSPACE_STRIDE +
                static_cast<long>(kv_head_idx * NUM_QO_PER_KV + q_head_local) *
                    HEAD_DIM;
#pragma unroll
      for (int h = 0; h < 4; h++) {
        int dim_offset = warp_id * 16 + kgrp * 4 + h;
        o[dim_offset] = static_cast<bf16>(o_acc[h] * inv_l);
      }
    } else {
      // Split-KV partial output: float + LSE for later merge
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
      // Match existing CK FMHA convention: m_val + logf(d_val) (kept as-is, not
      // log2).
      float lse_val = (l_sum > 0.0f) ? (m_running + logf(l_sum)) : -1e30f;
      *lse_out = lse_val;
    }
  }
}

} // namespace kernel
