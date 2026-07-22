#pragma once
// MFMA decode attention — same pipeline as bench_hip_decode_attn.cpp.
// Single-barrier software pipeline: overlap global K/V prefetch with compute.
//
// Pipeline per tile:
//   sync                -- K[t]+V[t] in LDS
//   ds_read K[t]        -- 4x ds_read_b128
//   ds_read_tr V[t]     -- 2x ds_read_b64_tr_b16
//   4x QK MFMA
//   softmax (fast_exp2)
//   ds_write V[t+1]     -- from v_pre (loaded prev iter, no stall)
//   2x PV MFMA
//   ds_write K[t+1]     -- from k_pre (loaded prev iter, no stall)
//   prefetch K[t+2]+V[t+2] into registers

namespace kernel {

using __mfma_fp16x8 = __attribute__((ext_vector_type(8))) _Float16;
using __mfma_fp16x4 = __attribute__((ext_vector_type(4))) _Float16;
using __mfma_fp32x4 = __attribute__((ext_vector_type(4))) float;

__device__ __forceinline__ __mfma_fp32x4 __mfma_qk(
    __mfma_fp32x4 c, const _Float16 *a, const _Float16 *b) {
  __mfma_fp16x8 av, bv;
  #pragma unroll
  for (int i = 0; i < 8; i++) { av[i] = a[i]; bv[i] = b[i]; }
  return __builtin_amdgcn_mfma_f32_16x16x32_f16(av, bv, c, 0, 0, 0);
}

__device__ __forceinline__ __mfma_fp32x4 __mfma_pv(
    __mfma_fp32x4 c, const _Float16 *a, const _Float16 *b) {
  __mfma_fp16x4 av, bv;
  #pragma unroll
  for (int i = 0; i < 4; i++) { av[i] = a[i]; bv[i] = b[i]; }
  return __builtin_amdgcn_mfma_f32_16x16x16f16(av, bv, c, 0, 0, 0);
}

// ds_read_b64_tr_b16 via builtin
#define __DECODE_LDS __attribute__((address_space(3)))
typedef __attribute__((__vector_size__(4 * sizeof(__fp16)))) __fp16 __decode_llvm_fp16x4_t;

__device__ __forceinline__ __mfma_fp16x4 ds_read_tr_b16_decode(const void *lds_ptr) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
  auto p = (__DECODE_LDS __decode_llvm_fp16x4_t *)(lds_ptr);
#pragma clang diagnostic pop
  __decode_llvm_fp16x4_t r = __builtin_amdgcn_ds_read_tr16_b64_v4f16(p);
  __mfma_fp16x4 result;
  __builtin_memcpy(&result, &r, sizeof(result));
  return result;
}

__device__ __forceinline__ float __fast_exp2(float x) {
  float r;
  asm("v_exp_f32 %0, %1" : "=v"(r) : "v"(x));
  return r;
}

// Vectorized bf16 load + convert to fp16.
// Load 8 bf16 as uint4 (compiler emits global_load_dwordx4), then
// v_cvt_f32_bf16 each half-word to float, truncate to fp16.
// This avoids scalar flat_load_ushort that __bfloat162float() generates.
__device__ __forceinline__ void __load_bf16x8_to_fp16(
    _Float16 *__restrict__ dst, const void *__restrict__ src)
{
  uint4 raw = *reinterpret_cast<const uint4 *>(src);
  unsigned words[4] = {raw.x, raw.y, raw.z, raw.w};
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    float lo_f, hi_f;
    asm("v_cvt_f32_bf16 %0, %1" : "=v"(lo_f) : "v"(words[i]));
    asm("v_cvt_f32_bf16 %0, %1" : "=v"(hi_f) : "v"(words[i] >> 16));
    dst[i*2]   = (_Float16)lo_f;
    dst[i*2+1] = (_Float16)hi_f;
  }
}

// Load 8 bf16 as raw dwords (for prefetch — conversion deferred to LDS write).
__device__ __forceinline__ void __load_bf16x8_raw(
    uint4 *__restrict__ dst, const void *__restrict__ src)
{
  *dst = *reinterpret_cast<const uint4 *>(src);
}

// Convert 4 raw dwords (8 bf16) to 8 fp16.
__device__ __forceinline__ void __cvt_bf16x8_to_fp16(
    _Float16 *__restrict__ dst, const uint4 &raw)
{
  unsigned words[4] = {raw.x, raw.y, raw.z, raw.w};
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    float lo_f, hi_f;
    asm("v_cvt_f32_bf16 %0, %1" : "=v"(lo_f) : "v"(words[i]));
    asm("v_cvt_f32_bf16 %0, %1" : "=v"(hi_f) : "v"(words[i] >> 16));
    dst[i*2]   = (_Float16)lo_f;
    dst[i*2+1] = (_Float16)hi_f;
  }
}

template <typename T, int NUM_QO_PER_KV, int HEAD_DIM, int PAGE_SIZE,
          int MAX_SEQ_LEN, int NUM_KV_CHUNKS, int Q_WORKSPACE_STRIDE,
          int KV_CACHE_STRIDE, int NUM_KV_HEADS>
__device__ __noinline__ void paged_attention_minimal_decode(
    void const *q_workspace_ptr, void *paged_k_cache_ptr,
    void *paged_v_cache_ptr, void *output_ptr, void *lse_ptr,
    int const *qo_indptr, int const *kv_indptr, int const *kv_indices,
    int const *kv_last_page_len, int16_t request_id, int kv_head_idx,
    int kv_chunk_idx, float scale_s)
{
  using bf16 = ck_tile::bf16_t;
  const int req = request_id;
  const int query_start = qo_indptr[req];
  if (query_start == qo_indptr[req + 1]) return;

  const int first_page = kv_indptr[req];
  const int num_pages = kv_indptr[req + 1] - first_page;
  const int seqlen_k = (num_pages - 1) * PAGE_SIZE + kv_last_page_len[req];

  const int tid = threadIdx.x;
  const int warp_id = tid / 64;
  const int lane = tid & 63;
  const int midx = lane & 15;
  const int kgrp = lane >> 4;

  static_assert(HEAD_DIM == 128);
  constexpr int KV_TILE = 16;
  constexpr int NUM_K32 = HEAD_DIM / 32;  // 4

  // Byte pointers for global loads — all vectorized via __load_bf16x8_to_fp16
  const char *k_base = reinterpret_cast<const char *>(paged_k_cache_ptr);
  const char *v_base = reinterpret_cast<const char *>(paged_v_cache_ptr);

  // LDS: K[16][128] + V[16][128](XOR-swizzled) = 8KB
  extern __shared__ char smem[];
  _Float16 *lds_k = reinterpret_cast<_Float16 *>(smem);
  char *lds_v = smem + 4096;

  int my_tok = tid / 16;
  int my_dim = (tid % 16) * 8;

  // V write addresses (XOR swizzle)
  unsigned v_wr0 = ((unsigned)tid << 4) ^ (((unsigned)tid & 0xf0) >> 1);
  unsigned v_wr1 = v_wr0 ^ 8;

  // V read address (XOR swizzle)
  unsigned v21_orig = tid & 0x3C;
  unsigned v22 = (tid << 3) & 24;
  unsigned v23 = tid >> 1;
  unsigned v20 = (v21_orig << 1) ^ (v23 & 0x60);
  unsigned v_rd = v20 ^ ((v21_orig << 6) | v22);

  // Load Q unconditionally (lanes >= NUM_QO_PER_KV load from head 0, output is guarded)
  // Uses uint4 load -> v_cvt_f32_bf16 to get global_load_dwordx4 in ISA
  _Float16 qr[NUM_K32][8];
  {
    int q_midx = (midx < NUM_QO_PER_KV) ? midx : 0;
    const char *q_ptr = reinterpret_cast<const char *>(q_workspace_ptr) +
        (static_cast<long>(query_start) * Q_WORKSPACE_STRIDE +
         static_cast<long>(kv_head_idx * NUM_QO_PER_KV + q_midx) * HEAD_DIM) * 2;
    #pragma unroll
    for (int kc = 0; kc < NUM_K32; kc++) {
      int dim_off = kc * 32 + kgrp * 8;
      __load_bf16x8_to_fp16(qr[kc], q_ptr + dim_off * 2);
    }
  }

  int ntiles = (seqlen_k + KV_TILE - 1) / KV_TILE;
  if (ntiles == 0) return;

  // KV byte offset helper (bf16 = 2 bytes per element)
  auto get_kv_off = [&](int global_tok) -> long {
    int pid = kv_indices[first_page + global_tok / PAGE_SIZE];
    return (static_cast<long>(pid) * PAGE_SIZE * KV_CACHE_STRIDE
         + static_cast<long>(global_tok % PAGE_SIZE) * KV_CACHE_STRIDE + my_dim) * 2;
  };

  // ===== PROLOGUE: K[0]+V[0] -> LDS =====
  int tile0_len = (KV_TILE < seqlen_k) ? KV_TILE : seqlen_k;
  if (my_tok < tile0_len) {
    long kv0 = get_kv_off(my_tok);
    _Float16 k_fp[8], v_fp[8];
    __load_bf16x8_to_fp16(k_fp, k_base + kv0);
    __load_bf16x8_to_fp16(v_fp, v_base + kv0);
    #pragma unroll
    for (int i = 0; i < 8; i++)
      lds_k[my_tok * HEAD_DIM + my_dim + i] = k_fp[i];
    *(uint64_t *)(lds_v + v_wr0) = *(uint64_t *)&v_fp[0];
    *(uint64_t *)(lds_v + v_wr1) = *(uint64_t *)&v_fp[4];
  } else {
    #pragma unroll
    for (int i = 0; i < 8; i++)
      lds_k[my_tok * HEAD_DIM + my_dim + i] = (_Float16)0.0f;
    _Float16 zero[8] = {0,0,0,0,0,0,0,0};
    *(uint64_t *)(lds_v + v_wr0) = *(uint64_t *)&zero[0];
    *(uint64_t *)(lds_v + v_wr1) = *(uint64_t *)&zero[4];
  }

  // Prefetch K[1]+V[1] -> registers as raw dwords (converted on write to LDS)
  uint4 k_pre, v_pre;
  bool has_pre = false;
  if (ntiles > 1) {
    int tile1_len = ((seqlen_k - KV_TILE) < KV_TILE) ? (seqlen_k - KV_TILE) : KV_TILE;
    if (my_tok < tile1_len) {
      long kv1 = get_kv_off(KV_TILE + my_tok);
      __load_bf16x8_raw(&k_pre, k_base + kv1);
      __load_bf16x8_raw(&v_pre, v_base + kv1);
      has_pre = true;
    }
  }

  // ===== MAIN LOOP =====
  __mfma_fp32x4 o_acc[2] = {{0,0,0,0}, {0,0,0,0}};
  float m_running = -INFINITY;
  float l_head[4] = {0, 0, 0, 0};

  for (int t = 0; t < ntiles; t++) {
    int tile_start = t * KV_TILE;
    int tile_len = ((seqlen_k - tile_start) < KV_TILE) ? (seqlen_k - tile_start) : KV_TILE;

    __syncthreads();

    // Read K[t] from LDS + ds_read_tr V[t].
    // Issue all reads, then pipeline MFMAs with inline asm waitcnts.
    _Float16 kr[NUM_K32][8];
    #pragma unroll
    for (int kc = 0; kc < NUM_K32; kc++) {
      const _Float16 *k_ptr = &lds_k[midx * HEAD_DIM + kc * 32 + kgrp * 8];
      #pragma unroll
      for (int i = 0; i < 8; i++)
        kr[kc][i] = k_ptr[i];
    }

    __mfma_fp16x4 va0 = ds_read_tr_b16_decode((_Float16 *)(lds_v + v_rd));
    __mfma_fp16x4 va1 = ds_read_tr_b16_decode((_Float16 *)(lds_v + v_rd + 128));

    // 4x QK MFMA — compiler will insert its own waitcnts
    __mfma_fp32x4 scores = {0, 0, 0, 0};
    #pragma unroll
    for (int kc = 0; kc < NUM_K32; kc++)
      scores = __mfma_qk(scores, kr[kc], qr[kc]);

    scores[0] *= scale_s; scores[1] *= scale_s;
    scores[2] *= scale_s; scores[3] *= scale_s;

    #pragma unroll
    for (int h = 0; h < 4; h++)
      if (kgrp * 4 + h >= tile_len) scores[h] = -INFINITY;

    // Online softmax
    float tile_max = fmaxf(fmaxf(scores[0], scores[1]), fmaxf(scores[2], scores[3]));
    {
      float a = tile_max, b = tile_max;
      asm volatile("s_nop 1\n\tv_permlane32_swap_b32_e32 %0, %1" : "+v"(a), "+v"(b));
      tile_max = fmaxf(a, b);
      a = tile_max; b = tile_max;
      asm volatile("s_nop 1\n\tv_permlane16_swap_b32_e32 %0, %1" : "+v"(a), "+v"(b));
      tile_max = fmaxf(a, b);
    }

    float new_max = fmaxf(m_running, tile_max);
    float rescale = (m_running == -INFINITY) ? 0.0f : __fast_exp2(m_running - new_max);

    o_acc[0][0] *= rescale; o_acc[0][1] *= rescale;
    o_acc[0][2] *= rescale; o_acc[0][3] *= rescale;
    o_acc[1][0] *= rescale; o_acc[1][1] *= rescale;
    o_acc[1][2] *= rescale; o_acc[1][3] *= rescale;

    float w0 = __fast_exp2(scores[0] - new_max);
    float w1 = __fast_exp2(scores[1] - new_max);
    float w2 = __fast_exp2(scores[2] - new_max);
    float w3 = __fast_exp2(scores[3] - new_max);

    l_head[0] = l_head[0] * rescale + w0;
    l_head[1] = l_head[1] * rescale + w1;
    l_head[2] = l_head[2] * rescale + w2;
    l_head[3] = l_head[3] * rescale + w3;
    m_running = new_max;

    // Write V[t+1] from v_pre (raw dwords -> fp16 -> LDS)
    if (has_pre) {
      _Float16 v_fp[8];
      __cvt_bf16x8_to_fp16(v_fp, v_pre);
      *(uint64_t *)(lds_v + v_wr0) = *(uint64_t *)&v_fp[0];
      *(uint64_t *)(lds_v + v_wr1) = *(uint64_t *)&v_fp[4];
    }

    // PV MFMA
    __mfma_fp16x4 pb;
    pb[0] = (_Float16)w0; pb[1] = (_Float16)w1;
    pb[2] = (_Float16)w2; pb[3] = (_Float16)w3;
    o_acc[0] = __mfma_pv(o_acc[0], (const _Float16 *)&va0, (const _Float16 *)&pb);
    o_acc[1] = __mfma_pv(o_acc[1], (const _Float16 *)&va1, (const _Float16 *)&pb);

    // Write K[t+1] from k_pre (raw dwords -> fp16 -> LDS)
    if (has_pre) {
      _Float16 k_fp[8];
      __cvt_bf16x8_to_fp16(k_fp, k_pre);
      #pragma unroll
      for (int i = 0; i < 8; i++)
        lds_k[my_tok * HEAD_DIM + my_dim + i] = k_fp[i];
    }

    // Prefetch K[t+2]+V[t+2] -> raw dwords (overlap with next iter)
    has_pre = false;
    if (t + 2 < ntiles) {
      int t2_start = (t + 2) * KV_TILE;
      int t2_len = ((seqlen_k - t2_start) < KV_TILE) ? (seqlen_k - t2_start) : KV_TILE;
      if (my_tok < t2_len) {
        long kv2 = get_kv_off(t2_start + my_tok);
        __load_bf16x8_raw(&k_pre, k_base + kv2);
        __load_bf16x8_raw(&v_pre, v_base + kv2);
        has_pre = true;
      }
    }
  }

  // Output
  {
    float l_sum = l_head[0] + l_head[1] + l_head[2] + l_head[3];
    l_sum += __shfl_xor(l_sum, 16);
    l_sum += __shfl_xor(l_sum, 32);

    if (midx >= NUM_QO_PER_KV) return;
    if (l_sum <= 0.0f) return;

    float inv_l = 1.0f / l_sum;
    int head = kv_head_idx * NUM_QO_PER_KV + midx;

    if constexpr (NUM_KV_CHUNKS == 1) {
      bf16 *o = reinterpret_cast<bf16 *>(output_ptr) +
          static_cast<long>(query_start) * Q_WORKSPACE_STRIDE +
          static_cast<long>(head) * HEAD_DIM;
      #pragma unroll
      for (int h = 0; h < 4; h++) {
        int dim_offset = warp_id * 16 + kgrp * 4 + h;
        o[dim_offset] = static_cast<bf16>(o_acc[0][h] * inv_l);
      }
      #pragma unroll
      for (int h = 0; h < 4; h++) {
        int dim_offset = 64 + warp_id * 16 + kgrp * 4 + h;
        o[dim_offset] = static_cast<bf16>(o_acc[1][h] * inv_l);
      }
    } else {
      constexpr int LSE_S = NUM_KV_HEADS * NUM_KV_CHUNKS * NUM_QO_PER_KV;
      constexpr int O_S = LSE_S * HEAD_DIM;
      float *o = reinterpret_cast<float *>(output_ptr) +
          static_cast<long>(query_start) * O_S +
          static_cast<long>(kv_head_idx) * NUM_KV_CHUNKS * NUM_QO_PER_KV * HEAD_DIM +
          static_cast<long>(kv_chunk_idx) * NUM_QO_PER_KV * HEAD_DIM +
          static_cast<long>(midx) * HEAD_DIM;
      #pragma unroll
      for (int h = 0; h < 4; h++) {
        int dim_offset = warp_id * 16 + kgrp * 4 + h;
        o[dim_offset] = o_acc[0][h] * inv_l;
      }
      #pragma unroll
      for (int h = 0; h < 4; h++) {
        int dim_offset = 64 + warp_id * 16 + kgrp * 4 + h;
        o[dim_offset] = o_acc[1][h] * inv_l;
      }
      if (lane == 0) {
        float lse_val = m_running + log2f(l_sum);
        float *lse = reinterpret_cast<float *>(lse_ptr) +
            static_cast<long>(query_start) * LSE_S +
            kv_head_idx * NUM_KV_CHUNKS * NUM_QO_PER_KV +
            kv_chunk_idx * NUM_QO_PER_KV + midx;
        *lse = lse_val;
      }
    }
  }
}

} // namespace kernel
