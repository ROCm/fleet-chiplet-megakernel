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
#include "element_binary.cuh"
#include "element_unary.cuh"
#include "norm.cuh"
#include "reduction.cuh"
#include "rotary_embedding.cuh"
#include "smem_layout.cuh"
#include "tasks/common/common_header.cuh"

namespace kernel {

// Helper function for inline RMS normalization on scalar arrays
template <typename T, int HEAD_DIM>
__device__ __forceinline__ void scalar_rms_norm_and_rope(
    T* data,  // [HEAD_DIM] array to normalize in-place
    T const* weight_ptr,
    float eps,
    bool apply_rope,
    T const* cos_ptr,
    T const* sin_ptr,
    float* reduce_smem,
    int warp_idx) {

  // Compute sum of squares
  float sum_sq = 0.0f;
  for (int i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
    float val = static_cast<float>(data[i]);
    sum_sq += val * val;
  }

  // Warp reduction
  for (int offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
    sum_sq += shfl_xor_sync(sum_sq, offset);
  }

  if (threadIdx.x % 32 == 0) {
    reduce_smem[warp_idx] = sum_sq;
  }
  __syncthreads();

  sum_sq = threadIdx.x < NUM_WARPS ? reduce_smem[threadIdx.x] : 0.0f;
  for (int offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
    sum_sq += shfl_xor_sync(sum_sq, offset);
  }

  if (threadIdx.x == 0) {
    reduce_smem[0] = sum_sq;
  }
  __syncthreads();

  float rms_rcp = rsqrt(reduce_smem[0] / float(HEAD_DIM) + eps);

  // Apply normalization and RoPE
  for (int i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
    float val = static_cast<float>(data[i]) * rms_rcp * static_cast<float>(weight_ptr[i]);
    data[i] = static_cast<T>(val);
  }
  __syncthreads();

  // Apply RoPE if needed
  if (apply_rope) {
    for (int i = threadIdx.x; i < HEAD_DIM / 2; i += NUM_THREADS) {
      float v0 = static_cast<float>(data[i]);
      float v1 = static_cast<float>(data[i + HEAD_DIM / 2]);
      float c = static_cast<float>(cos_ptr[i]);
      float s = static_cast<float>(sin_ptr[i]);
      data[i] = static_cast<T>(v0 * c - v1 * s);
      data[i + HEAD_DIM / 2] = static_cast<T>(v0 * s + v1 * c);
    }
    __syncthreads();
  }
}

// Scalar attention for AMD - memory-efficient single-buffer version
// Fits within MI300's 64KB LDS limit
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
__device__ __forceinline__ void multitoken_paged_attention_scalar(
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
  constexpr int KV_TILE_SIZE = 32;  // Reduced from 64 to save shared memory
  constexpr int MAX_PAGES_PER_REQUEST = (MAX_SEQ_LEN + PAGE_SIZE - 1) / PAGE_SIZE;
  constexpr float log2e = 1.4426950408889634f;
  constexpr int Q_ROWS = MAX_TOKENS * NUM_QO_PER_KV;

  float const sm_scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM)) * log2e;

  int const first_token_pos = qo_indptr_buffer_ptr[request_id];
  int const last_token_pos = qo_indptr_buffer_ptr[request_id + 1];
  if (first_token_pos == last_token_pos) {
    return;
  }
  int const num_tokens = last_token_pos - first_token_pos;

  int const first_page_pos = paged_kv_indptr_buffer_ptr[request_id];
  int const last_page_pos = paged_kv_indptr_buffer_ptr[request_id + 1];
  int const num_pages = last_page_pos - first_page_pos;
  int const seq_len = (num_pages - 1) * PAGE_SIZE +
                      paged_kv_last_page_len_buffer_ptr[request_id];

  __shared__ __align__(16) int page_indices[MAX_PAGES_PER_REQUEST];
  for (int i = threadIdx.x; i < num_pages; i += NUM_THREADS) {
    page_indices[i] = paged_kv_indices_buffer_ptr[first_page_pos + i];
  }
  __syncthreads();

  T const *__restrict__ d_q =
      reinterpret_cast<T const *>(qkv_ptr) + first_token_pos * QKV_STRIDE;
  T const *__restrict__ d_k = d_q + NUM_QO_PER_KV * HEAD_DIM;
  T const *__restrict__ d_v = d_k + HEAD_DIM;
  T *__restrict__ d_paged_k_cache = reinterpret_cast<T *>(paged_k_cache_ptr);
  T *__restrict__ d_paged_v_cache = reinterpret_cast<T *>(paged_v_cache_ptr);
  T *__restrict__ d_output =
      reinterpret_cast<T *>(output_ptr) + first_token_pos * O_STRIDE;

  extern __shared__ char smem[];

  // Memory-efficient layout - single buffered K/V
  // Total ~48KB for Q_ROWS=32, HEAD_DIM=128, KV_TILE_SIZE=32
  constexpr size_t S_Q_OFFSET = 0;
  constexpr size_t S_Q_SIZE = sizeof(T) * Q_ROWS * HEAD_DIM;  // 32*128*2 = 8KB

  constexpr size_t S_K_OFFSET = S_Q_OFFSET + S_Q_SIZE;
  constexpr size_t S_K_SIZE = sizeof(T) * KV_TILE_SIZE * HEAD_DIM;  // 32*128*2 = 8KB

  constexpr size_t S_V_OFFSET = S_K_OFFSET + S_K_SIZE;
  constexpr size_t S_V_SIZE = sizeof(T) * KV_TILE_SIZE * HEAD_DIM;  // 8KB

  // Reduction buffer for RMS norm
  constexpr size_t S_REDUCE_OFFSET = S_V_OFFSET + S_V_SIZE;
  constexpr size_t S_REDUCE_SIZE = sizeof(float) * NUM_WARPS;  // 32 bytes

  // Online softmax state - m and d per query row
  constexpr size_t S_M_OFFSET = ((S_REDUCE_OFFSET + S_REDUCE_SIZE + 15) & ~15);
  constexpr size_t S_M_SIZE = sizeof(float) * Q_ROWS;  // 128 bytes
  constexpr size_t S_D_OFFSET = S_M_OFFSET + S_M_SIZE;
  constexpr size_t S_D_SIZE = sizeof(float) * Q_ROWS;  // 128 bytes

  // Output accumulator in fp32 - reuse Q space after Q processing is done
  constexpr size_t S_O_OFFSET = S_D_OFFSET + S_D_SIZE;
  constexpr size_t S_O_SIZE = sizeof(float) * Q_ROWS * HEAD_DIM;  // 32*128*4 = 16KB

  // Total: 8 + 8 + 8 + 0.032 + 0.128 + 0.128 + 16 = ~40KB

  T *s_q = reinterpret_cast<T *>(smem + S_Q_OFFSET);
  T *s_k = reinterpret_cast<T *>(smem + S_K_OFFSET);
  T *s_v = reinterpret_cast<T *>(smem + S_V_OFFSET);
  float *s_reduce = reinterpret_cast<float *>(smem + S_REDUCE_OFFSET);
  float *s_m = reinterpret_cast<float *>(smem + S_M_OFFSET);
  float *s_d = reinterpret_cast<float *>(smem + S_D_OFFSET);
  float *s_o = reinterpret_cast<float *>(smem + S_O_OFFSET);

  int warp_idx = warp_id();
  int q_rows = num_tokens * NUM_QO_PER_KV;

  // Initialize m, d, o
  for (int idx = threadIdx.x; idx < q_rows; idx += NUM_THREADS) {
    s_m[idx] = -INFINITY;
    s_d[idx] = 1.0f;
  }
  for (int idx = threadIdx.x; idx < q_rows * HEAD_DIM; idx += NUM_THREADS) {
    s_o[idx] = 0.0f;
  }

  // Load Q
  for (int idx = threadIdx.x; idx < num_tokens * NUM_QO_PER_KV * HEAD_DIM; idx += NUM_THREADS) {
    int row = idx / HEAD_DIM;
    int col = idx % HEAD_DIM;
    int token = row / NUM_QO_PER_KV;
    int head = row % NUM_QO_PER_KV;
    s_q[row * HEAD_DIM + col] = d_q[token * QKV_STRIDE + head * HEAD_DIM + col];
  }
  __syncthreads();

  // Apply QK norm and/or RoPE to Q
  if (qk_norm) {
    T const *q_weight = reinterpret_cast<T const *>(q_norm_weight_ptr);
    for (int token = 0; token < num_tokens; token++) {
      int pos = seq_len - num_tokens + token;
      T const *cos_data = rope ? reinterpret_cast<T const *>(cos_ptr) + pos * HEAD_DIM : nullptr;
      T const *sin_data = rope ? reinterpret_cast<T const *>(sin_ptr) + pos * HEAD_DIM : nullptr;
      for (int head = 0; head < NUM_QO_PER_KV; head++) {
        int row = token * NUM_QO_PER_KV + head;
        T *q_head = s_q + row * HEAD_DIM;

        // Compute sum of squares
        float sum_sq = 0.0f;
        for (int i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
          float val = static_cast<float>(q_head[i]);
          sum_sq += val * val;
        }

        // Warp reduction
        for (int offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
          sum_sq += shfl_xor_sync(sum_sq, offset);
        }
        if (threadIdx.x % 32 == 0) {
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
          float val = static_cast<float>(q_head[i]) * rms_rcp * static_cast<float>(q_weight[i]);
          q_head[i] = static_cast<T>(val);
        }
        __syncthreads();

        // Apply RoPE if needed
        if (rope) {
          for (int i = threadIdx.x; i < HEAD_DIM / 2; i += NUM_THREADS) {
            float v0 = static_cast<float>(q_head[i]);
            float v1 = static_cast<float>(q_head[i + HEAD_DIM / 2]);
            float c = static_cast<float>(cos_data[i]);
            float s_val = static_cast<float>(sin_data[i]);
            q_head[i] = static_cast<T>(v0 * c - v1 * s_val);
            q_head[i + HEAD_DIM / 2] = static_cast<T>(v0 * s_val + v1 * c);
          }
          __syncthreads();
        }
      }
    }
  } else if (rope) {
    for (int token = 0; token < num_tokens; token++) {
      int pos = seq_len - num_tokens + token;
      T const *cos_data = reinterpret_cast<T const *>(cos_ptr) + pos * HEAD_DIM;
      T const *sin_data = reinterpret_cast<T const *>(sin_ptr) + pos * HEAD_DIM;
      for (int idx = threadIdx.x; idx < NUM_QO_PER_KV * HEAD_DIM / 2; idx += NUM_THREADS) {
        int head = idx / (HEAD_DIM / 2);
        int d = idx % (HEAD_DIM / 2);
        int base = (token * NUM_QO_PER_KV + head) * HEAD_DIM;
        float q0 = static_cast<float>(s_q[base + d]);
        float q1 = static_cast<float>(s_q[base + d + HEAD_DIM / 2]);
        float c = static_cast<float>(cos_data[d]);
        float sv = static_cast<float>(sin_data[d]);
        s_q[base + d] = static_cast<T>(q0 * c - q1 * sv);
        s_q[base + d + HEAD_DIM / 2] = static_cast<T>(q0 * sv + q1 * c);
      }
    }
    __syncthreads();
  }

  T const *k_weight = reinterpret_cast<T const *>(k_norm_weight_ptr);
  int const num_iters = (seq_len + KV_TILE_SIZE - 1) / KV_TILE_SIZE;

  for (int iter = 0; iter < num_iters; iter++) {
    int tile_start = iter * KV_TILE_SIZE;
    int curr_iter_len = min(seq_len - tile_start, KV_TILE_SIZE);

    // Load KV tile
    for (int idx = threadIdx.x; idx < curr_iter_len * HEAD_DIM; idx += NUM_THREADS) {
      int row = idx / HEAD_DIM;
      int col = idx % HEAD_DIM;
      int global_pos = tile_start + row;
      if (global_pos < seq_len - num_tokens) {
        // From KV cache
        int page_num = global_pos / PAGE_SIZE;
        int page_offset = global_pos % PAGE_SIZE;
        int page_idx = page_indices[page_num];
        int src_idx = page_idx * PAGE_SIZE + page_offset;
        s_k[row * HEAD_DIM + col] = d_paged_k_cache[src_idx * KV_CACHE_STRIDE + col];
        s_v[row * HEAD_DIM + col] = d_paged_v_cache[src_idx * KV_CACHE_STRIDE + col];
      } else {
        // From QKV input
        int qkv_row = global_pos - (seq_len - num_tokens);
        s_k[row * HEAD_DIM + col] = d_k[qkv_row * QKV_STRIDE + col];
        s_v[row * HEAD_DIM + col] = d_v[qkv_row * QKV_STRIDE + col];
      }
    }
    __syncthreads();

    // Apply QK norm and/or RoPE to new K tokens from QKV input
    int first_new_token = max(tile_start, seq_len - num_tokens);
    int last_new_token = min(tile_start + curr_iter_len, seq_len);
    int num_new_tokens = max(0, last_new_token - first_new_token);

    if (num_new_tokens > 0) {
      int local_start = first_new_token - tile_start;

      if (qk_norm) {
        for (int kv_idx = 0; kv_idx < num_new_tokens; kv_idx++) {
          int pos = first_new_token + kv_idx;
          int local_row = local_start + kv_idx;
          T *k_head = s_k + local_row * HEAD_DIM;
          T const *cos_data = rope ? reinterpret_cast<T const *>(cos_ptr) + pos * HEAD_DIM : nullptr;
          T const *sin_data = rope ? reinterpret_cast<T const *>(sin_ptr) + pos * HEAD_DIM : nullptr;

          // Compute sum of squares
          float sum_sq = 0.0f;
          for (int i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
            float val = static_cast<float>(k_head[i]);
            sum_sq += val * val;
          }

          // Warp reduction
          for (int offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
            sum_sq += shfl_xor_sync(sum_sq, offset);
          }
          if (threadIdx.x % 32 == 0) {
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

          // Apply normalization
          for (int i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
            float val = static_cast<float>(k_head[i]) * rms_rcp * static_cast<float>(k_weight[i]);
            k_head[i] = static_cast<T>(val);
          }
          __syncthreads();

          // Apply RoPE if needed
          if (rope) {
            for (int i = threadIdx.x; i < HEAD_DIM / 2; i += NUM_THREADS) {
              float v0 = static_cast<float>(k_head[i]);
              float v1 = static_cast<float>(k_head[i + HEAD_DIM / 2]);
              float c = static_cast<float>(cos_data[i]);
              float s_val = static_cast<float>(sin_data[i]);
              k_head[i] = static_cast<T>(v0 * c - v1 * s_val);
              k_head[i + HEAD_DIM / 2] = static_cast<T>(v0 * s_val + v1 * c);
            }
            __syncthreads();
          }
        }
      } else if (rope) {
        for (int kv_idx = 0; kv_idx < num_new_tokens; kv_idx++) {
          int pos = first_new_token + kv_idx;
          int local_row = local_start + kv_idx;
          T const *cos_data = reinterpret_cast<T const *>(cos_ptr) + pos * HEAD_DIM;
          T const *sin_data = reinterpret_cast<T const *>(sin_ptr) + pos * HEAD_DIM;
          for (int d = threadIdx.x; d < HEAD_DIM / 2; d += NUM_THREADS) {
            float k0 = static_cast<float>(s_k[local_row * HEAD_DIM + d]);
            float k1 = static_cast<float>(s_k[local_row * HEAD_DIM + d + HEAD_DIM / 2]);
            float c = static_cast<float>(cos_data[d]);
            float s = static_cast<float>(sin_data[d]);
            s_k[local_row * HEAD_DIM + d] = static_cast<T>(k0 * c - k1 * s);
            s_k[local_row * HEAD_DIM + d + HEAD_DIM / 2] = static_cast<T>(k0 * s + k1 * c);
          }
        }
        __syncthreads();
      }

      // Write processed K/V to cache
      for (int idx = threadIdx.x; idx < num_new_tokens * HEAD_DIM; idx += NUM_THREADS) {
        int token = idx / HEAD_DIM;
        int col = idx % HEAD_DIM;
        int global_pos = first_new_token + token;
        int page_num = global_pos / PAGE_SIZE;
        int page_offset = global_pos % PAGE_SIZE;
        int page_idx = page_indices[page_num];
        int dst_idx = page_idx * PAGE_SIZE + page_offset;
        int local_row = local_start + token;
        d_paged_k_cache[dst_idx * KV_CACHE_STRIDE + col] = s_k[local_row * HEAD_DIM + col];
        d_paged_v_cache[dst_idx * KV_CACHE_STRIDE + col] = s_v[local_row * HEAD_DIM + col];
      }
      __syncthreads();
    }

    // Compute attention scores and accumulate output
    // Each thread handles one or more query rows
    for (int qr = threadIdx.x; qr < q_rows; qr += NUM_THREADS) {
      int token = qr / NUM_QO_PER_KV;
      int q_pos = seq_len - num_tokens + token;

      // Compute scores for this tile
      float m_local = -INFINITY;
      float scores[KV_TILE_SIZE];

      for (int kc = 0; kc < curr_iter_len; kc++) {
        int kv_pos = tile_start + kc;
        float score = 0.0f;

        // Dot product Q[qr] . K[kc]
        for (int d = 0; d < HEAD_DIM; d++) {
          score += static_cast<float>(s_q[qr * HEAD_DIM + d]) *
                   static_cast<float>(s_k[kc * HEAD_DIM + d]);
        }
        score *= sm_scale;

        // Causal mask
        if (kv_pos > q_pos) {
          score = -INFINITY;
        }

        scores[kc] = score;
        if (score > m_local) m_local = score;
      }

      // Online softmax update
      float m_prev = s_m[qr];
      float d_prev = s_d[qr];
      float m_new = fmaxf(m_prev, m_local);
      float rescale = ptx_exp2(m_prev - m_new);

      // Rescale previous O
      for (int d = 0; d < HEAD_DIM; d++) {
        s_o[qr * HEAD_DIM + d] *= rescale;
      }

      // Compute sum and accumulate weighted V
      float d_local = 0.0f;
      for (int kc = 0; kc < curr_iter_len; kc++) {
        float attn = ptx_exp2(scores[kc] - m_new);
        d_local += attn;
        for (int d = 0; d < HEAD_DIM; d++) {
          s_o[qr * HEAD_DIM + d] += attn * static_cast<float>(s_v[kc * HEAD_DIM + d]);
        }
      }

      s_m[qr] = m_new;
      s_d[qr] = d_prev * rescale + d_local;
    }
    __syncthreads();
  }

  // Final normalization and output
  for (int idx = threadIdx.x; idx < num_tokens * NUM_QO_PER_KV * HEAD_DIM; idx += NUM_THREADS) {
    int row = idx / HEAD_DIM;
    int col = idx % HEAD_DIM;
    int token = row / NUM_QO_PER_KV;
    int head = row % NUM_QO_PER_KV;
    float val = s_o[row * HEAD_DIM + col] / s_d[row];
    int dst_col = head * HEAD_DIM + col;
    d_output[token * O_STRIDE + dst_col] = static_cast<T>(val);
  }
}

} // namespace kernel
