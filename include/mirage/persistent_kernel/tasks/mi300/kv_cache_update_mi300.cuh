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

namespace kernel {

// KV Cache Update task for AMD MI300.
// Handles Phase A of the 3-phase attention:
//   Phase A: QK norm + RoPE on Q and new K tokens, write K/V to paged cache,
//            write processed Q to workspace
//   Phase B: CK FMHA split-KV attention (batch-independent)
//   Phase C: Merge partial results
//
// This task runs per (request_id, kv_head) with grid_dim=(max_requests,
// num_kv_heads, 1). It is a lightweight operation (~3.8K cycles at decode)
// since it only does preprocessing and cache writes, no attention compute.
template <typename T,
          int NUM_QO_HEADS,
          int NUM_KV_HEADS,
          int NUM_QO_GROUPS,
          int KV_CACHE_STRIDE,
          int QKV_STRIDE,
          int HEAD_DIM,
          int MAX_SEQ_LEN,
          int PAGE_SIZE,
          int MAX_TOKENS,
          int Q_WORKSPACE_STRIDE>
__device__ __forceinline__ void
    kv_cache_update_impl(void const *qkv_ptr,
                         void *paged_k_cache_ptr,
                         void *paged_v_cache_ptr,
                         void *q_workspace_ptr,
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
  constexpr int NUM_WARPS = NUM_THREADS / 64;
  constexpr int NUM_THREADS_PER_WARP = 64;
  constexpr int MAX_PAGES_PER_REQUEST =
      (MAX_SEQ_LEN + PAGE_SIZE - 1) / PAGE_SIZE;
  constexpr int VEC_SIZE = 8;
  constexpr int HEAD_DIM_VECS = HEAD_DIM / VEC_SIZE;

  using bf16 = ck_tile::bf16_t;

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

  int warp_idx = threadIdx.x / 64;

  // Load page indices to shared memory
  __shared__ __align__(16) int page_indices[MAX_PAGES_PER_REQUEST];
  for (int i = threadIdx.x; i < num_pages; i += NUM_THREADS) {
    page_indices[i] = paged_kv_indices_buffer_ptr[first_page_pos + i];
  }

  extern __shared__ char smem[];
  // Shared memory layout for Q/K preprocessing
  // s_q: NUM_QO_PER_KV * num_tokens * HEAD_DIM (for Q preprocessing)
  // s_k: num_tokens * HEAD_DIM (for K preprocessing)
  // s_reduce: NUM_WARPS floats (for RMS norm reduction)
  constexpr int Q_ROWS = MAX_TOKENS * NUM_QO_PER_KV;
  constexpr size_t S_Q_SIZE = sizeof(bf16) * Q_ROWS * HEAD_DIM;
  constexpr size_t S_K_SIZE = sizeof(bf16) * MAX_TOKENS * HEAD_DIM;
  constexpr size_t S_REDUCE_OFFSET = ((S_Q_SIZE + S_K_SIZE + 15) & ~15);
  constexpr size_t S_REDUCE_SIZE = sizeof(float) * NUM_WARPS;

  bf16 *s_q = reinterpret_cast<bf16 *>(smem);
  bf16 *s_k = reinterpret_cast<bf16 *>(smem + S_Q_SIZE);
  float *s_reduce = reinterpret_cast<float *>(smem + S_REDUCE_OFFSET);

  bf16 const *__restrict__ d_q =
      reinterpret_cast<bf16 const *>(qkv_ptr) + first_token_pos * QKV_STRIDE;
  bf16 const *__restrict__ d_k = d_q + NUM_QO_PER_KV * HEAD_DIM;
  bf16 const *__restrict__ d_v = d_k + HEAD_DIM;
  bf16 *__restrict__ d_paged_k_cache =
      reinterpret_cast<bf16 *>(paged_k_cache_ptr);
  bf16 *__restrict__ d_paged_v_cache =
      reinterpret_cast<bf16 *>(paged_v_cache_ptr);

  __syncthreads();

  // =========================================================================
  // Step 1: Load Q from QKV buffer, apply QK norm + RoPE, write to workspace
  // =========================================================================
  int total_q_vecs = num_tokens * NUM_QO_PER_KV * HEAD_DIM_VECS;
  for (int vec_idx = threadIdx.x; vec_idx < total_q_vecs;
       vec_idx += NUM_THREADS) {
    int row = vec_idx / HEAD_DIM_VECS;
    int vec_col = vec_idx % HEAD_DIM_VECS;
    int token = row / NUM_QO_PER_KV;
    int head = row % NUM_QO_PER_KV;
    vec_load_8(&s_q[row * HEAD_DIM + vec_col * VEC_SIZE],
               &d_q[token * QKV_STRIDE + head * HEAD_DIM + vec_col * VEC_SIZE]);
  }
  __syncthreads();

  // Apply QK norm and/or RoPE to Q
  if (qk_norm) {
    bf16 const *q_weight = reinterpret_cast<bf16 const *>(q_norm_weight_ptr);
    for (int token = 0; token < num_tokens; token++) {
      int pos = global_seq_len - num_tokens + token;
      bf16 const *cos_data =
          rope ? reinterpret_cast<bf16 const *>(cos_ptr) + pos * HEAD_DIM
               : nullptr;
      bf16 const *sin_data =
          rope ? reinterpret_cast<bf16 const *>(sin_ptr) + pos * HEAD_DIM
               : nullptr;
      for (int head = 0; head < NUM_QO_PER_KV; head++) {
        int row = token * NUM_QO_PER_KV + head;
        bf16 *q_head = s_q + row * HEAD_DIM;

        // RMS norm reduction
        float sum_sq = 0.0f;
        for (int i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
          float val = ck_tile::type_convert<float>(q_head[i]);
          sum_sq += val * val;
        }
        for (int offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
          sum_sq += shfl_xor_sync(sum_sq, offset);
        }
        if (threadIdx.x % 64 == 0) {
          s_reduce[warp_idx] = sum_sq;
        }
        __syncthreads();
        sum_sq = threadIdx.x < NUM_WARPS ? s_reduce[threadIdx.x] : 0.0f;
        for (int offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
          sum_sq += shfl_xor_sync(sum_sq, offset);
        }
        if (threadIdx.x == 0) {
          s_reduce[0] = sum_sq;
        }
        __syncthreads();
        float rms_rcp = rsqrt(s_reduce[0] / float(HEAD_DIM) + q_eps);

        // Apply normalization
        for (int i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
          float val = ck_tile::type_convert<float>(q_head[i]) * rms_rcp *
                      ck_tile::type_convert<float>(q_weight[i]);
          q_head[i] = ck_tile::type_convert<bf16>(val);
        }
        __syncthreads();

        // Apply RoPE
        if (rope) {
          for (int i = threadIdx.x; i < HEAD_DIM / 2; i += NUM_THREADS) {
            float v0 = ck_tile::type_convert<float>(q_head[i]);
            float v1 = ck_tile::type_convert<float>(q_head[i + HEAD_DIM / 2]);
            float c = ck_tile::type_convert<float>(cos_data[i]);
            float s_val = ck_tile::type_convert<float>(sin_data[i]);
            q_head[i] = ck_tile::type_convert<bf16>(v0 * c - v1 * s_val);
            q_head[i + HEAD_DIM / 2] =
                ck_tile::type_convert<bf16>(v0 * s_val + v1 * c);
          }
          __syncthreads();
        }
      }
    }
  } else if (rope) {
    for (int token = 0; token < num_tokens; token++) {
      int pos = global_seq_len - num_tokens + token;
      bf16 const *cos_data =
          reinterpret_cast<bf16 const *>(cos_ptr) + pos * HEAD_DIM;
      bf16 const *sin_data =
          reinterpret_cast<bf16 const *>(sin_ptr) + pos * HEAD_DIM;
      for (int idx = threadIdx.x; idx < NUM_QO_PER_KV * HEAD_DIM / 2;
           idx += NUM_THREADS) {
        int head = idx / (HEAD_DIM / 2);
        int d = idx % (HEAD_DIM / 2);
        int base = (token * NUM_QO_PER_KV + head) * HEAD_DIM;
        float q0 = ck_tile::type_convert<float>(s_q[base + d]);
        float q1 = ck_tile::type_convert<float>(s_q[base + d + HEAD_DIM / 2]);
        float c = ck_tile::type_convert<float>(cos_data[d]);
        float sv = ck_tile::type_convert<float>(sin_data[d]);
        s_q[base + d] = ck_tile::type_convert<bf16>(q0 * c - q1 * sv);
        s_q[base + d + HEAD_DIM / 2] =
            ck_tile::type_convert<bf16>(q0 * sv + q1 * c);
      }
    }
    __syncthreads();
  }

  // Write processed Q to workspace
  // Layout: q_workspace[token_pos * Q_WORKSPACE_STRIDE + head * HEAD_DIM + d]
  // where Q_WORKSPACE_STRIDE = NUM_QO_PER_KV * HEAD_DIM
  bf16 *d_q_workspace = reinterpret_cast<bf16 *>(q_workspace_ptr) +
                        first_token_pos * Q_WORKSPACE_STRIDE;
  for (int vec_idx = threadIdx.x; vec_idx < total_q_vecs;
       vec_idx += NUM_THREADS) {
    int row = vec_idx / HEAD_DIM_VECS;
    int vec_col = vec_idx % HEAD_DIM_VECS;
    int token = row / NUM_QO_PER_KV;
    int head = row % NUM_QO_PER_KV;
    vec_store_8(&d_q_workspace[token * Q_WORKSPACE_STRIDE + head * HEAD_DIM +
                               vec_col * VEC_SIZE],
                &s_q[row * HEAD_DIM + vec_col * VEC_SIZE]);
  }

  __syncthreads();

  // =========================================================================
  // Step 2: Load new K tokens, apply QK norm + RoPE, write to KV cache
  // =========================================================================
  // New K tokens are the last `num_tokens` in the sequence
  int total_k_vecs = num_tokens * HEAD_DIM_VECS;
  for (int vec_idx = threadIdx.x; vec_idx < total_k_vecs;
       vec_idx += NUM_THREADS) {
    int token = vec_idx / HEAD_DIM_VECS;
    int vec_col = vec_idx % HEAD_DIM_VECS;
    vec_load_8(&s_k[token * HEAD_DIM + vec_col * VEC_SIZE],
               &d_k[token * QKV_STRIDE + vec_col * VEC_SIZE]);
  }
  __syncthreads();

  bf16 const *k_weight = reinterpret_cast<bf16 const *>(k_norm_weight_ptr);

  if (qk_norm) {
    for (int k_tok = 0; k_tok < num_tokens; k_tok++) {
      int pos = global_seq_len - num_tokens + k_tok;
      bf16 const *cos_data =
          rope ? reinterpret_cast<bf16 const *>(cos_ptr) + pos * HEAD_DIM
               : nullptr;
      bf16 const *sin_data =
          rope ? reinterpret_cast<bf16 const *>(sin_ptr) + pos * HEAD_DIM
               : nullptr;
      bf16 *k_head = s_k + k_tok * HEAD_DIM;

      // RMS norm reduction
      float sum_sq = 0.0f;
      for (int i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
        float val = ck_tile::type_convert<float>(k_head[i]);
        sum_sq += val * val;
      }
      for (int offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
        sum_sq += shfl_xor_sync(sum_sq, offset);
      }
      if (threadIdx.x % 64 == 0) {
        s_reduce[warp_idx] = sum_sq;
      }
      __syncthreads();
      sum_sq = threadIdx.x < NUM_WARPS ? s_reduce[threadIdx.x] : 0.0f;
      for (int offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
        sum_sq += shfl_xor_sync(sum_sq, offset);
      }
      if (threadIdx.x == 0) {
        s_reduce[0] = sum_sq;
      }
      __syncthreads();
      float rms_rcp = rsqrt(s_reduce[0] / float(HEAD_DIM) + k_eps);

      for (int i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
        float val = ck_tile::type_convert<float>(k_head[i]) * rms_rcp *
                    ck_tile::type_convert<float>(k_weight[i]);
        k_head[i] = ck_tile::type_convert<bf16>(val);
      }
      __syncthreads();

      if (rope) {
        for (int i = threadIdx.x; i < HEAD_DIM / 2; i += NUM_THREADS) {
          float v0 = ck_tile::type_convert<float>(k_head[i]);
          float v1 = ck_tile::type_convert<float>(k_head[i + HEAD_DIM / 2]);
          float c = ck_tile::type_convert<float>(cos_data[i]);
          float s_val = ck_tile::type_convert<float>(sin_data[i]);
          k_head[i] = ck_tile::type_convert<bf16>(v0 * c - v1 * s_val);
          k_head[i + HEAD_DIM / 2] =
              ck_tile::type_convert<bf16>(v0 * s_val + v1 * c);
        }
        __syncthreads();
      }
    }
  } else if (rope) {
    for (int k_tok = 0; k_tok < num_tokens; k_tok++) {
      int pos = global_seq_len - num_tokens + k_tok;
      bf16 const *cos_data =
          reinterpret_cast<bf16 const *>(cos_ptr) + pos * HEAD_DIM;
      bf16 const *sin_data =
          reinterpret_cast<bf16 const *>(sin_ptr) + pos * HEAD_DIM;
      for (int d = threadIdx.x; d < HEAD_DIM / 2; d += NUM_THREADS) {
        float k0 = ck_tile::type_convert<float>(s_k[k_tok * HEAD_DIM + d]);
        float k1 = ck_tile::type_convert<float>(
            s_k[k_tok * HEAD_DIM + d + HEAD_DIM / 2]);
        float c = ck_tile::type_convert<float>(cos_data[d]);
        float s = ck_tile::type_convert<float>(sin_data[d]);
        s_k[k_tok * HEAD_DIM + d] =
            ck_tile::type_convert<bf16>(k0 * c - k1 * s);
        s_k[k_tok * HEAD_DIM + d + HEAD_DIM / 2] =
            ck_tile::type_convert<bf16>(k0 * s + k1 * c);
      }
    }
    __syncthreads();
  }

  // Write processed K to paged cache
  for (int k_tok = 0; k_tok < num_tokens; k_tok++) {
    int global_pos = global_seq_len - num_tokens + k_tok;
    int page_num = global_pos / PAGE_SIZE;
    int page_offset = global_pos % PAGE_SIZE;
    int page_idx = page_indices[page_num];
    int dst_idx = page_idx * PAGE_SIZE + page_offset;
    for (int vec_col = threadIdx.x; vec_col < HEAD_DIM_VECS;
         vec_col += NUM_THREADS) {
      vec_store_8(
          &d_paged_k_cache[dst_idx * KV_CACHE_STRIDE + vec_col * VEC_SIZE],
          &s_k[k_tok * HEAD_DIM + vec_col * VEC_SIZE]);
    }
  }

  // =========================================================================
  // Step 3: Write new V tokens to paged cache (V doesn't need norm/RoPE)
  // =========================================================================
  for (int v_tok = 0; v_tok < num_tokens; v_tok++) {
    int global_pos = global_seq_len - num_tokens + v_tok;
    int page_num = global_pos / PAGE_SIZE;
    int page_offset = global_pos % PAGE_SIZE;
    int page_idx = page_indices[page_num];
    int dst_idx = page_idx * PAGE_SIZE + page_offset;
    for (int vec_col = threadIdx.x; vec_col < HEAD_DIM_VECS;
         vec_col += NUM_THREADS) {
      vec_store_8(
          &d_paged_v_cache[dst_idx * KV_CACHE_STRIDE + vec_col * VEC_SIZE],
          &d_v[v_tok * QKV_STRIDE + vec_col * VEC_SIZE]);
    }
  }
}

} // namespace kernel
