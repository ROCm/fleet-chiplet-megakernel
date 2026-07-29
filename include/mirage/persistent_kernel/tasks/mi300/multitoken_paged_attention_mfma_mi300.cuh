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
#include "tasks/common/common_header.cuh"

namespace kernel {

// MFMA vector types for AMD
typedef short __attribute__((ext_vector_type(4))) mfma_attn_short4;
typedef float __attribute__((ext_vector_type(4))) mfma_attn_float4;

union mfma_attn_bf16x4 {
  unsigned short u16[4];
  mfma_attn_short4 vec;
};

// MFMA-optimized paged attention implementation for AMD MI300
// Uses MFMA 16x16x16 bf16 instructions for Q*K^T and attn*V computations
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
__device__ __forceinline__ void multitoken_paged_attention_mfma(
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
  constexpr int KV_TILE_SIZE = 64;
  constexpr int MAX_PAGES_PER_REQUEST =
      (MAX_SEQ_LEN + PAGE_SIZE - 1) / PAGE_SIZE;
  constexpr int CP_CHUNK_SIZE = 16 / sizeof(T);

  // MFMA tile dimensions
  constexpr int MFMA_M = 16;
  constexpr int MFMA_N = 16;
  constexpr int MFMA_K = 16;

  float const sm_scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));
  float const log2e = 1.4426950408889634f;

  int tid = threadIdx.x;
  int lane = tid % 64; // Lane within 64-thread AMD wavefront
  int wavefront_id = tid / 64;

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

  // Load page indices into shared memory
  __shared__ __align__(16) int page_indices[MAX_PAGES_PER_REQUEST];
  for (int i = tid; i < num_pages; i += NUM_THREADS) {
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

  // Shared memory layout for MFMA
  constexpr size_t ALIGN_PAD = 64;
  constexpr size_t S_Q_OFFSET = ALIGN_PAD;
  constexpr size_t S_Q_SIZE = sizeof(T) * MAX_TOKENS * NUM_QO_PER_KV * HEAD_DIM;
  constexpr size_t S_K_OFFSET = S_Q_OFFSET + S_Q_SIZE;
  constexpr size_t S_K_SIZE = sizeof(T) * KV_TILE_SIZE * HEAD_DIM;
  constexpr size_t S_V_OFFSET = S_K_OFFSET + S_K_SIZE;
  constexpr size_t S_V_SIZE = S_K_SIZE;
  constexpr size_t S_QK_OFFSET = S_V_OFFSET + S_V_SIZE;
  constexpr size_t S_QK_SIZE =
      sizeof(float) * MAX_TOKENS * NUM_QO_PER_KV * KV_TILE_SIZE;
  constexpr size_t S_O_OFFSET = S_QK_OFFSET + S_QK_SIZE;
  constexpr size_t S_O_SIZE =
      sizeof(float) * MAX_TOKENS * NUM_QO_PER_KV * HEAD_DIM;
  constexpr size_t S_M_OFFSET = S_O_OFFSET + S_O_SIZE;
  constexpr size_t S_M_SIZE = sizeof(float) * MAX_TOKENS * NUM_QO_PER_KV;
  constexpr size_t S_L_OFFSET = S_M_OFFSET + S_M_SIZE;
  constexpr size_t S_L_SIZE = sizeof(float) * MAX_TOKENS * NUM_QO_PER_KV;

  T *s_q = reinterpret_cast<T *>(smem + S_Q_OFFSET);
  T *s_k = reinterpret_cast<T *>(smem + S_K_OFFSET);
  T *s_v = reinterpret_cast<T *>(smem + S_V_OFFSET);
  float *s_qk = reinterpret_cast<float *>(smem + S_QK_OFFSET);
  float *s_o = reinterpret_cast<float *>(smem + S_O_OFFSET);
  float *s_m = reinterpret_cast<float *>(smem + S_M_OFFSET);
  float *s_l = reinterpret_cast<float *>(smem + S_L_OFFSET);

  // Load Q into shared memory with proper layout for MFMA
  for (int idx = tid; idx < num_tokens * NUM_QO_PER_KV * HEAD_DIM;
       idx += NUM_THREADS) {
    int token = idx / (NUM_QO_PER_KV * HEAD_DIM);
    int head_elem = idx % (NUM_QO_PER_KV * HEAD_DIM);
    int head = head_elem / HEAD_DIM;
    int elem = head_elem % HEAD_DIM;
    s_q[(token * NUM_QO_PER_KV + head) * HEAD_DIM + elem] =
        d_q[token * QKV_STRIDE + head * HEAD_DIM + elem];
  }

  __syncthreads();

  // Initialize output accumulator and softmax stats
  for (int idx = tid; idx < num_tokens * NUM_QO_PER_KV * HEAD_DIM;
       idx += NUM_THREADS) {
    s_o[idx] = 0.0f;
  }
  for (int idx = tid; idx < num_tokens * NUM_QO_PER_KV; idx += NUM_THREADS) {
    s_m[idx] = -INFINITY;
    s_l[idx] = 0.0f;
  }
  __syncthreads();

  // Apply RoPE to Q if needed
  if (rope && !qk_norm) {
    for (int token = 0; token < num_tokens; token++) {
      int pos = seq_len - num_tokens + token;
      T const *cos_data = reinterpret_cast<T const *>(cos_ptr) + pos * HEAD_DIM;
      T const *sin_data = reinterpret_cast<T const *>(sin_ptr) + pos * HEAD_DIM;
      for (int idx = tid; idx < NUM_QO_PER_KV * HEAD_DIM / 2;
           idx += NUM_THREADS) {
        int head = idx / (HEAD_DIM / 2);
        int d = idx % (HEAD_DIM / 2);
        int base = (token * NUM_QO_PER_KV + head) * HEAD_DIM;
        float q0 = static_cast<float>(s_q[base + d]);
        float q1 = static_cast<float>(s_q[base + d + HEAD_DIM / 2]);
        float c = static_cast<float>(cos_data[d]);
        float s_val = static_cast<float>(sin_data[d]);
        s_q[base + d] = static_cast<T>(q0 * c - q1 * s_val);
        s_q[base + d + HEAD_DIM / 2] = static_cast<T>(q0 * s_val + q1 * c);
      }
    }
    __syncthreads();
  }

  int const num_iters = (seq_len + KV_TILE_SIZE - 1) / KV_TILE_SIZE;

  for (int iter = 0; iter < num_iters; iter++) {
    int kv_start = iter * KV_TILE_SIZE;
    int kv_len = min(KV_TILE_SIZE, seq_len - kv_start);

    // Load K tile into shared memory
    for (int idx = tid; idx < kv_len * HEAD_DIM; idx += NUM_THREADS) {
      int kv_pos = idx / HEAD_DIM;
      int elem = idx % HEAD_DIM;
      int global_pos = kv_start + kv_pos;

      T val;
      if (global_pos < seq_len - num_tokens) {
        int page_idx = page_indices[global_pos / PAGE_SIZE];
        int page_offset = global_pos % PAGE_SIZE;
        val = d_paged_k_cache[page_idx * PAGE_SIZE * KV_CACHE_STRIDE +
                              page_offset * KV_CACHE_STRIDE + elem];
      } else {
        int token_idx = global_pos - (seq_len - num_tokens);
        val = d_k[token_idx * QKV_STRIDE + elem];
      }
      s_k[kv_pos * HEAD_DIM + elem] = val;
    }

    // Load V tile into shared memory
    for (int idx = tid; idx < kv_len * HEAD_DIM; idx += NUM_THREADS) {
      int kv_pos = idx / HEAD_DIM;
      int elem = idx % HEAD_DIM;
      int global_pos = kv_start + kv_pos;

      T val;
      if (global_pos < seq_len - num_tokens) {
        int page_idx = page_indices[global_pos / PAGE_SIZE];
        int page_offset = global_pos % PAGE_SIZE;
        val = d_paged_v_cache[page_idx * PAGE_SIZE * KV_CACHE_STRIDE +
                              page_offset * KV_CACHE_STRIDE + elem];
      } else {
        int token_idx = global_pos - (seq_len - num_tokens);
        val = d_v[token_idx * QKV_STRIDE + elem];
      }
      s_v[kv_pos * HEAD_DIM + elem] = val;
    }
    __syncthreads();

    // Compute Q * K^T using MFMA
    // Q: [num_tokens * NUM_QO_PER_KV, HEAD_DIM]
    // K^T: [HEAD_DIM, kv_len]
    // Result: [num_tokens * NUM_QO_PER_KV, kv_len]

    int q_rows = num_tokens * NUM_QO_PER_KV;
    constexpr int TILES_M = (MAX_TOKENS * NUM_QO_PER_KV + MFMA_M - 1) / MFMA_M;
    constexpr int TILES_N = (KV_TILE_SIZE + MFMA_N - 1) / MFMA_N;
    constexpr int TILES_K = (HEAD_DIM + MFMA_K - 1) / MFMA_K;

    // Initialize QK results to zero
    for (int idx = tid; idx < q_rows * kv_len; idx += NUM_THREADS) {
      s_qk[idx] = 0.0f;
    }
    __syncthreads();

    // MFMA-based matrix multiply for Q * K^T
    if (wavefront_id < 2) {
      // Two wavefronts participate in compute
      constexpr int MAX_TILES_PER_WAVE = (TILES_N + 1) / 2;

      for (int tile_m = 0; tile_m < TILES_M; tile_m++) {
        mfma_attn_float4 accum[MAX_TILES_PER_WAVE > 0 ? MAX_TILES_PER_WAVE : 1];

#pragma unroll
        for (int t = 0; t < MAX_TILES_PER_WAVE; t++) {
          accum[t] = {0.0f, 0.0f, 0.0f, 0.0f};
        }

        // K-dimension loop
        for (int tile_k = 0; tile_k < TILES_K; tile_k++) {
          int q_row = tile_m * MFMA_M + (lane % 16);
          int q_col = tile_k * MFMA_K + (lane / 16) * 4;

          // Load Q fragment
          mfma_attn_bf16x4 reg_q;
          if (q_row < q_rows && q_col + 3 < HEAD_DIM) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
              reg_q.u16[i] = reinterpret_cast<unsigned short *>(
                  &s_q[q_row * HEAD_DIM + q_col])[i];
            }
          } else {
            reg_q.u16[0] = reg_q.u16[1] = reg_q.u16[2] = reg_q.u16[3] = 0;
          }

// Process N tiles
#pragma unroll
          for (int local_tile = 0; local_tile < MAX_TILES_PER_WAVE;
               local_tile++) {
            int tile_n = local_tile * 2 + wavefront_id;
            if (tile_n < TILES_N) {
              int k_col = tile_n * MFMA_N + (lane % 16);     // KV position
              int k_row = tile_k * MFMA_K + (lane / 16) * 4; // HEAD_DIM

              // Load K^T fragment (K is stored as [kv_len, HEAD_DIM], we need
              // transpose)
              mfma_attn_bf16x4 reg_k;
              if (k_col < kv_len && k_row + 3 < HEAD_DIM) {
#pragma unroll
                for (int i = 0; i < 4; i++) {
                  reg_k.u16[i] = reinterpret_cast<unsigned short *>(
                      &s_k[k_col * HEAD_DIM + k_row])[i];
                }
              } else {
                reg_k.u16[0] = reg_k.u16[1] = reg_k.u16[2] = reg_k.u16[3] = 0;
              }

              // Execute MFMA
              accum[local_tile] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                  reg_q.vec, reg_k.vec, accum[local_tile], 0, 0, 0);
            }
          }
        }

// Store QK results
#pragma unroll
        for (int local_tile = 0; local_tile < MAX_TILES_PER_WAVE;
             local_tile++) {
          int tile_n = local_tile * 2 + wavefront_id;
          if (tile_n >= TILES_N) {
            continue;
          }

          int out_row_base = tile_m * MFMA_M + (lane / 16) * 4;
          int out_col = tile_n * MFMA_N + (lane % 16);

          if (out_col < kv_len) {
#pragma unroll
            for (int r = 0; r < 4; r++) {
              int out_row = out_row_base + r;
              if (out_row < q_rows) {
                s_qk[out_row * KV_TILE_SIZE + out_col] =
                    accum[local_tile][r] * sm_scale;
              }
            }
          }
        }
      }
    }
    __syncthreads();

    // Apply causal mask and compute softmax update
    for (int q_idx = tid; q_idx < q_rows; q_idx += NUM_THREADS) {
      int token = q_idx / NUM_QO_PER_KV;
      int q_pos = seq_len - num_tokens + token;
      float m_prev = s_m[q_idx];
      float l_prev = s_l[q_idx];

      // Find max with causal mask
      float m_new = m_prev;
      for (int k = 0; k < kv_len; k++) {
        int k_pos = kv_start + k;
        float score =
            (k_pos <= q_pos) ? s_qk[q_idx * KV_TILE_SIZE + k] : -INFINITY;
        s_qk[q_idx * KV_TILE_SIZE + k] = score; // Store masked score
        m_new = fmaxf(m_new, score);
      }

      // Compute new sum
      float l_new = l_prev * expf(m_prev - m_new);
      for (int k = 0; k < kv_len; k++) {
        float score = s_qk[q_idx * KV_TILE_SIZE + k];
        if (score > -INFINITY) {
          l_new += expf(score - m_new);
        }
      }

      // Rescale previous output and update
      float scale = expf(m_prev - m_new);
      for (int d = 0; d < HEAD_DIM; d++) {
        s_o[q_idx * HEAD_DIM + d] *= scale;
      }

      // Add weighted V
      for (int k = 0; k < kv_len; k++) {
        float score = s_qk[q_idx * KV_TILE_SIZE + k];
        if (score > -INFINITY) {
          float attn_weight = expf(score - m_new);
          for (int d = 0; d < HEAD_DIM; d++) {
            s_o[q_idx * HEAD_DIM + d] +=
                attn_weight * static_cast<float>(s_v[k * HEAD_DIM + d]);
          }
        }
      }

      s_m[q_idx] = m_new;
      s_l[q_idx] = l_new;
    }
    __syncthreads();

    // Update KV cache for new tokens
    int kv_tokens_to_process =
        min(kv_len, max(kv_start + kv_len - (seq_len - num_tokens), 0));
    if (kv_tokens_to_process > 0) {
      int first_kv_token = kv_start + kv_len - kv_tokens_to_process;
      for (int idx = tid; idx < kv_tokens_to_process * HEAD_DIM;
           idx += NUM_THREADS) {
        int kv_offset = idx / HEAD_DIM;
        int elem = idx % HEAD_DIM;
        int global_pos = first_kv_token + kv_offset;
        int page_idx = page_indices[global_pos / PAGE_SIZE];
        int page_offset = global_pos % PAGE_SIZE;
        int local_kv_pos = (first_kv_token + kv_offset) % KV_TILE_SIZE;

        d_paged_k_cache[page_idx * PAGE_SIZE * KV_CACHE_STRIDE +
                        page_offset * KV_CACHE_STRIDE + elem] =
            s_k[local_kv_pos * HEAD_DIM + elem];
        d_paged_v_cache[page_idx * PAGE_SIZE * KV_CACHE_STRIDE +
                        page_offset * KV_CACHE_STRIDE + elem] =
            s_v[local_kv_pos * HEAD_DIM + elem];
      }
    }
    __syncthreads();
  }

  // Normalize output and write to global memory
  for (int idx = tid; idx < num_tokens * NUM_QO_PER_KV * HEAD_DIM;
       idx += NUM_THREADS) {
    int q_idx = idx / HEAD_DIM;
    int d = idx % HEAD_DIM;
    int token = q_idx / NUM_QO_PER_KV;
    int head = q_idx % NUM_QO_PER_KV;

    float o_val = s_o[q_idx * HEAD_DIM + d] / s_l[q_idx];
    d_output[token * O_STRIDE + head * HEAD_DIM + d] = static_cast<T>(o_val);
  }
}

} // namespace kernel
