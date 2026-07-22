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

#pragma once
// CK FMHA PagedKV and SplitKV exec functions are compiled in separate TUs
// (ck_fmha_rdc_wrapper.hip, ck_fmha_splitkv_rdc_wrapper.hip) and linked via
// -fgpu-rdc. This avoids AMD hipcc codegen bugs where CK FMHA templates
// get miscompiled in the large persistent kernel TU.
//
// CK FMHA includes (ck_tile/ops/fmha.hpp) are NOT included here to avoid
// template instantiation in the main TU. Only core CK types (bf16, type_convert)
// are available via ck_tile/core.hpp included by the GEMM code.

namespace kernel {

// RDC wrapper declarations (definitions in separate .hip files)
// Stub: when not linking with RDC .hip TUs, provide a no-op definition
// to satisfy the linker. The function is only called if
// TASK_PAGED_ATTENTION_CK_FMHA tasks are actually dispatched.
__device__ __noinline__ void ck_fmha_pagedkv_exec(
    const void* q_ptr, const void* k_ptr, const void* v_ptr,
    void* o_ptr,
    const int* page_table_ptr, int num_pages, int page_size,
    int seqlen_q, int seqlen_k, int head_dim,
    int num_heads_q, int nhead_ratio_qk,
    float scale_s,
    int stride_q, int kv_cache_stride, int stride_o,
    int nhead_stride_q, int nhead_stride_o,
    int batch_stride_q, int batch_stride_kv, int batch_stride_o,
    int i_nhead, char* smem_ptr) {
  // Stub — should never be reached unless RDC .hip TUs are linked
}

// ============================================================================
// CK PagedKV attention for decode (seqlen_q=1) via RDC wrapper
// ============================================================================
template <typename T,
          int NUM_QO_HEADS,
          int NUM_KV_HEADS,
          int KV_CACHE_STRIDE,
          int QKV_STRIDE,
          int O_STRIDE,
          int HEAD_DIM,
          int MAX_SEQ_LEN,
          int PAGE_SIZE,
          int MAX_TOKENS = 8>
__device__ __forceinline__ void multitoken_paged_attention_ck_pagedkv(
    void const *qkv_ptr,
    void *paged_k_cache_ptr,
    void *paged_v_cache_ptr,
    void *output_ptr,
    int const *qo_indptr_buffer_ptr,
    int const *paged_kv_indptr_buffer_ptr,
    int const *paged_kv_indices_buffer_ptr,
    int const *paged_kv_last_page_len_buffer_ptr,
    int16_t request_id,
    bool qk_norm,
    bool rope,
    void const *q_norm_weight_ptr,
    void const *k_norm_weight_ptr,
    void const *cos_ptr,
    void const *sin_ptr,
    float q_eps,
    float k_eps) {

  constexpr int NUM_QO_PER_KV = NUM_QO_HEADS / NUM_KV_HEADS;
  constexpr int NUM_THREADS = 256;
  constexpr int MAX_PAGES_PER_REQUEST = (MAX_SEQ_LEN + PAGE_SIZE - 1) / PAGE_SIZE;
  constexpr int VEC_SIZE = 8;
  constexpr int HEAD_DIM_VECS = HEAD_DIM / VEC_SIZE;
  constexpr int Q_ROWS = MAX_TOKENS * NUM_QO_PER_KV;
  constexpr int NUM_WARPS = NUM_THREADS / 64;

  int const first_token_pos = qo_indptr_buffer_ptr[request_id];
  int const last_token_pos = qo_indptr_buffer_ptr[request_id + 1];
  if (first_token_pos == last_token_pos) {
    return;
  }
  int const num_tokens = last_token_pos - first_token_pos;

  int const first_page_pos = paged_kv_indptr_buffer_ptr[request_id];
  int const last_page_pos = paged_kv_indptr_buffer_ptr[request_id + 1];
  int const num_pages = last_page_pos - first_page_pos;
  int const global_seq_len = (num_pages - 1) * PAGE_SIZE +
                              paged_kv_last_page_len_buffer_ptr[request_id];

  bf16 const *__restrict__ d_q =
      reinterpret_cast<bf16 const *>(qkv_ptr) + first_token_pos * QKV_STRIDE;
  bf16 const *__restrict__ d_k = d_q + NUM_QO_PER_KV * HEAD_DIM;
  bf16 const *__restrict__ d_v = d_k + HEAD_DIM;
  bf16 *__restrict__ d_paged_k_cache = reinterpret_cast<bf16 *>(paged_k_cache_ptr);
  bf16 *__restrict__ d_paged_v_cache = reinterpret_cast<bf16 *>(paged_v_cache_ptr);
  bf16 *__restrict__ d_output =
      reinterpret_cast<bf16 *>(output_ptr) + first_token_pos * O_STRIDE;

  extern __shared__ char smem[];

  // Shared memory for Q/K processing (reused by CK later)
  constexpr size_t S_Q_SIZE = sizeof(bf16) * Q_ROWS * HEAD_DIM;
  constexpr size_t S_REDUCE_OFFSET = ((S_Q_SIZE + 15) & ~15);

  bf16 *s_q = reinterpret_cast<bf16 *>(smem);
  float *s_reduce = reinterpret_cast<float *>(smem + S_REDUCE_OFFSET);

  int warp_idx = threadIdx.x / 64;

  // =========================================================================
  // Phase 1: Load Q, apply QK norm + RoPE, write to output (staging for CK)
  // =========================================================================

  int total_q_vecs = num_tokens * NUM_QO_PER_KV * HEAD_DIM_VECS;
  for (int vec_idx = threadIdx.x; vec_idx < total_q_vecs; vec_idx += NUM_THREADS) {
    int row = vec_idx / HEAD_DIM_VECS;
    int vec_col = vec_idx % HEAD_DIM_VECS;
    int token = row / NUM_QO_PER_KV;
    int head = row % NUM_QO_PER_KV;
    vec_load_8(&s_q[row * HEAD_DIM + vec_col * VEC_SIZE],
               &d_q[token * QKV_STRIDE + head * HEAD_DIM + vec_col * VEC_SIZE]);
  }
  __syncthreads();

  if (qk_norm) {
    bf16 const *q_weight = reinterpret_cast<bf16 const *>(q_norm_weight_ptr);
    for (int token = 0; token < num_tokens; token++) {
      int pos = global_seq_len - num_tokens + token;
      bf16 const *cos_data = rope ? reinterpret_cast<bf16 const *>(cos_ptr) + pos * HEAD_DIM : nullptr;
      bf16 const *sin_data = rope ? reinterpret_cast<bf16 const *>(sin_ptr) + pos * HEAD_DIM : nullptr;
      for (int head = 0; head < NUM_QO_PER_KV; head++) {
        int row = token * NUM_QO_PER_KV + head;
        bf16 *q_head = s_q + row * HEAD_DIM;

        float sum_sq = 0.0f;
        for (int i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
          float val = ck_tile::type_convert<float>(q_head[i]);
          sum_sq += val * val;
        }
        for (int offset = 32; offset > 0; offset /= 2) {
          sum_sq += shfl_xor_sync(sum_sq, offset);
        }
        if (threadIdx.x % 64 == 0) { s_reduce[warp_idx] = sum_sq; }
        __syncthreads();
        sum_sq = threadIdx.x < NUM_WARPS ? s_reduce[threadIdx.x] : 0.0f;
        for (int offset = 32; offset > 0; offset /= 2) {
          sum_sq += shfl_xor_sync(sum_sq, offset);
        }
        if (threadIdx.x == 0) { s_reduce[0] = sum_sq; }
        __syncthreads();
        float rms_rcp = rsqrt(s_reduce[0] / float(HEAD_DIM) + q_eps);

        for (int i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
          float val = ck_tile::type_convert<float>(q_head[i]) * rms_rcp * ck_tile::type_convert<float>(q_weight[i]);
          q_head[i] = ck_tile::type_convert<bf16>(val);
        }
        __syncthreads();

        if (rope) {
          for (int i = threadIdx.x; i < HEAD_DIM / 2; i += NUM_THREADS) {
            float v0 = ck_tile::type_convert<float>(q_head[i]);
            float v1 = ck_tile::type_convert<float>(q_head[i + HEAD_DIM / 2]);
            float c = ck_tile::type_convert<float>(cos_data[i]);
            float s_val = ck_tile::type_convert<float>(sin_data[i]);
            q_head[i] = ck_tile::type_convert<bf16>(v0 * c - v1 * s_val);
            q_head[i + HEAD_DIM / 2] = ck_tile::type_convert<bf16>(v0 * s_val + v1 * c);
          }
          __syncthreads();
        }
      }
    }
  } else if (rope) {
    for (int token = 0; token < num_tokens; token++) {
      int pos = global_seq_len - num_tokens + token;
      bf16 const *cos_data = reinterpret_cast<bf16 const *>(cos_ptr) + pos * HEAD_DIM;
      bf16 const *sin_data = reinterpret_cast<bf16 const *>(sin_ptr) + pos * HEAD_DIM;
      for (int idx = threadIdx.x; idx < NUM_QO_PER_KV * HEAD_DIM / 2; idx += NUM_THREADS) {
        int head = idx / (HEAD_DIM / 2);
        int d = idx % (HEAD_DIM / 2);
        int base = (token * NUM_QO_PER_KV + head) * HEAD_DIM;
        float q0 = ck_tile::type_convert<float>(s_q[base + d]);
        float q1 = ck_tile::type_convert<float>(s_q[base + d + HEAD_DIM / 2]);
        float c = ck_tile::type_convert<float>(cos_data[d]);
        float sv = ck_tile::type_convert<float>(sin_data[d]);
        s_q[base + d] = ck_tile::type_convert<bf16>(q0 * c - q1 * sv);
        s_q[base + d + HEAD_DIM / 2] = ck_tile::type_convert<bf16>(q0 * sv + q1 * c);
      }
    }
    __syncthreads();
  }

  // Write processed Q to output buffer (staging area for CK to read)
  for (int vec_idx = threadIdx.x; vec_idx < total_q_vecs; vec_idx += NUM_THREADS) {
    int row = vec_idx / HEAD_DIM_VECS;
    int vec_col = vec_idx % HEAD_DIM_VECS;
    int token = row / NUM_QO_PER_KV;
    int head = row % NUM_QO_PER_KV;
    vec_store_8(&d_output[token * O_STRIDE + head * HEAD_DIM + vec_col * VEC_SIZE],
                &s_q[row * HEAD_DIM + vec_col * VEC_SIZE]);
  }
  __syncthreads();

  // =========================================================================
  // Phase 2: Process new K/V tokens (QK norm + RoPE + cache write)
  // =========================================================================

  bf16 const *k_weight = reinterpret_cast<bf16 const *>(k_norm_weight_ptr);
  bf16 *s_k = s_q;  // reuse buffer

  for (int tok = 0; tok < num_tokens; tok++) {
    int pos = global_seq_len - num_tokens + tok;

    for (int i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
      s_k[i] = d_k[tok * QKV_STRIDE + i];
    }
    __syncthreads();

    if (qk_norm) {
      bf16 const *cos_data = rope ? reinterpret_cast<bf16 const *>(cos_ptr) + pos * HEAD_DIM : nullptr;
      bf16 const *sin_data = rope ? reinterpret_cast<bf16 const *>(sin_ptr) + pos * HEAD_DIM : nullptr;

      float sum_sq = 0.0f;
      for (int i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
        float val = ck_tile::type_convert<float>(s_k[i]);
        sum_sq += val * val;
      }
      for (int offset = 32; offset > 0; offset /= 2) {
        sum_sq += shfl_xor_sync(sum_sq, offset);
      }
      if (threadIdx.x % 64 == 0) { s_reduce[warp_idx] = sum_sq; }
      __syncthreads();
      sum_sq = threadIdx.x < NUM_WARPS ? s_reduce[threadIdx.x] : 0.0f;
      for (int offset = 32; offset > 0; offset /= 2) {
        sum_sq += shfl_xor_sync(sum_sq, offset);
      }
      if (threadIdx.x == 0) { s_reduce[0] = sum_sq; }
      __syncthreads();
      float rms_rcp = rsqrt(s_reduce[0] / float(HEAD_DIM) + k_eps);

      for (int i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
        float val = ck_tile::type_convert<float>(s_k[i]) * rms_rcp * ck_tile::type_convert<float>(k_weight[i]);
        s_k[i] = ck_tile::type_convert<bf16>(val);
      }
      __syncthreads();

      if (rope) {
        for (int i = threadIdx.x; i < HEAD_DIM / 2; i += NUM_THREADS) {
          float v0 = ck_tile::type_convert<float>(s_k[i]);
          float v1 = ck_tile::type_convert<float>(s_k[i + HEAD_DIM / 2]);
          float c = ck_tile::type_convert<float>(cos_data[i]);
          float s_val = ck_tile::type_convert<float>(sin_data[i]);
          s_k[i] = ck_tile::type_convert<bf16>(v0 * c - v1 * s_val);
          s_k[i + HEAD_DIM / 2] = ck_tile::type_convert<bf16>(v0 * s_val + v1 * c);
        }
        __syncthreads();
      }
    } else if (rope) {
      bf16 const *cos_data = reinterpret_cast<bf16 const *>(cos_ptr) + pos * HEAD_DIM;
      bf16 const *sin_data = reinterpret_cast<bf16 const *>(sin_ptr) + pos * HEAD_DIM;
      for (int i = threadIdx.x; i < HEAD_DIM / 2; i += NUM_THREADS) {
        float k0 = ck_tile::type_convert<float>(s_k[i]);
        float k1 = ck_tile::type_convert<float>(s_k[i + HEAD_DIM / 2]);
        float c = ck_tile::type_convert<float>(cos_data[i]);
        float s = ck_tile::type_convert<float>(sin_data[i]);
        s_k[i] = ck_tile::type_convert<bf16>(k0 * c - k1 * s);
        s_k[i + HEAD_DIM / 2] = ck_tile::type_convert<bf16>(k0 * s + k1 * c);
      }
      __syncthreads();
    }

    // Write K to paged cache
    int page_num = pos / PAGE_SIZE;
    int page_offset = pos % PAGE_SIZE;
    int page_idx = paged_kv_indices_buffer_ptr[first_page_pos + page_num];
    int dst_idx = page_idx * PAGE_SIZE + page_offset;
    for (int vec_idx = threadIdx.x; vec_idx < HEAD_DIM_VECS; vec_idx += NUM_THREADS) {
      vec_store_8(&d_paged_k_cache[dst_idx * KV_CACHE_STRIDE + vec_idx * VEC_SIZE],
                  &s_k[vec_idx * VEC_SIZE]);
    }

    // Write V to paged cache (no norm/rope)
    for (int vec_idx = threadIdx.x; vec_idx < HEAD_DIM_VECS; vec_idx += NUM_THREADS) {
      bf16 v_buf[VEC_SIZE];
      vec_load_8(v_buf, &d_v[tok * QKV_STRIDE + vec_idx * VEC_SIZE]);
      vec_store_8(&d_paged_v_cache[dst_idx * KV_CACHE_STRIDE + vec_idx * VEC_SIZE],
                  v_buf);
    }
  }
  __threadfence();  // Ensure K/V cache writes are globally visible
  __syncthreads();

  // =========================================================================
  // Phase 3: CK PagedKV FMHA (via RDC wrapper)
  // =========================================================================

  const float scale_s = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));

  for (int qh = 0; qh < NUM_QO_PER_KV; qh++) {
    ck_fmha_pagedkv_exec(
        static_cast<const void*>(d_output),                // q_ptr (staged Q)
        static_cast<const void*>(d_paged_k_cache),         // k_ptr
        static_cast<const void*>(d_paged_v_cache),         // v_ptr
        static_cast<void*>(d_output),                      // o_ptr
        paged_kv_indices_buffer_ptr + first_page_pos,      // page_table_ptr
        num_pages,                                         // num_pages
        PAGE_SIZE,                                         // page_size
        1,                                                 // seqlen_q (decode)
        global_seq_len,                                    // seqlen_k
        HEAD_DIM,                                          // head_dim
        NUM_QO_PER_KV,                                     // num_heads_q
        NUM_QO_PER_KV,                                     // nhead_ratio_qk
        scale_s,                                           // scale_s
        O_STRIDE,                                          // stride_q
        KV_CACHE_STRIDE,                                   // kv_cache_stride
        O_STRIDE,                                          // stride_o
        HEAD_DIM,                                          // nhead_stride_q
        HEAD_DIM,                                          // nhead_stride_o
        O_STRIDE,                                          // batch_stride_q
        PAGE_SIZE * KV_CACHE_STRIDE,                       // batch_stride_kv
        O_STRIDE,                                          // batch_stride_o
        qh,                                                // i_nhead
        smem);                                             // smem_ptr
    __syncthreads();
  }
}

// Wrapper: CK for decode, hand-written for prefill
template <typename T,
          int NUM_QO_HEADS,
          int NUM_KV_HEADS,
          int KV_CACHE_STRIDE,
          int QKV_STRIDE,
          int O_STRIDE,
          int HEAD_DIM,
          int MAX_SEQ_LEN,
          int PAGE_SIZE,
          int MAX_TOKENS = 8>
__device__ __forceinline__ void multitoken_paged_attention_ck_pagedkv_task_impl(
    void const *qkv_ptr,
    void *paged_k_cache_ptr,
    void *paged_v_cache_ptr,
    void *output_ptr,
    int const *qo_indptr_buffer_ptr,
    int const *paged_kv_indptr_buffer_ptr,
    int const *paged_kv_indices_buffer_ptr,
    int const *paged_kv_last_page_len_buffer_ptr,
    int16_t request_id,
    bool qk_norm,
    bool rope,
    void const *q_norm_weight_ptr,
    void const *k_norm_weight_ptr,
    void const *cos_ptr,
    void const *sin_ptr,
    float q_eps,
    float k_eps) {

  int const first_tp = qo_indptr_buffer_ptr[request_id];
  int const last_tp = qo_indptr_buffer_ptr[request_id + 1];
  int const nt = last_tp - first_tp;

  // Always use the MFMA-based implementation. The CK PagedKV path
  // (multitoken_paged_attention_ck_pagedkv) depends on RDC-linked CK FMHA
  // TUs that are not yet available, so its ck_fmha_pagedkv_exec is a stub.
  multitoken_paged_attention_ck<T, NUM_QO_HEADS, NUM_KV_HEADS,
      KV_CACHE_STRIDE, QKV_STRIDE, O_STRIDE, HEAD_DIM, MAX_SEQ_LEN,
      PAGE_SIZE, MAX_TOKENS>(
      qkv_ptr, paged_k_cache_ptr, paged_v_cache_ptr, output_ptr,
      qo_indptr_buffer_ptr, paged_kv_indptr_buffer_ptr,
      paged_kv_indices_buffer_ptr, paged_kv_last_page_len_buffer_ptr,
      request_id, qk_norm, rope, q_norm_weight_ptr, k_norm_weight_ptr,
      cos_ptr, sin_ptr, q_eps, k_eps);
}

} // namespace kernel
