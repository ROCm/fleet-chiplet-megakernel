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
#include "ck_tile/core.hpp"
#include "tasks/common/common_header.cuh"

namespace kernel {

// Types for MFMA operations
using bf16 = ck_tile::bf16_t;
using mfma_bf16x4 = short __attribute__((ext_vector_type(4)));
using mfma_float4 = float __attribute__((ext_vector_type(4)));

// Vectorized load/store helper (float4 = 16 bytes = 8 bf16)
struct float4_t {
  float x, y, z, w;
};

template <typename T>
__device__ __forceinline__ void vec_load_8(T *dst, T const *src) {
  *reinterpret_cast<float4_t *>(dst) = *reinterpret_cast<float4_t const *>(src);
}

template <typename T>
__device__ __forceinline__ void vec_store_8(T *dst, T const *src) {
  *reinterpret_cast<float4_t *>(dst) = *reinterpret_cast<float4_t const *>(src);
}

// MFMA-based paged attention implementation for AMD MI300
// Uses MFMA 16x16x16 bf16 instructions with vectorized memory access
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
__device__ __forceinline__ void
    multitoken_paged_attention_ck(void const *qkv_ptr,
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
  constexpr int KV_TILE_SIZE = 32;
  constexpr int MAX_PAGES_PER_REQUEST =
      (MAX_SEQ_LEN + PAGE_SIZE - 1) / PAGE_SIZE;
  constexpr float log2e = 1.4426950408889634f;
  constexpr int Q_ROWS = MAX_TOKENS * NUM_QO_PER_KV;

  // MFMA tile dimensions
  constexpr int MFMA_M = 16;
  constexpr int MFMA_N = 16;
  constexpr int MFMA_K = 16;

  // Number of MFMA tiles
  constexpr int Q_TILES_M = (Q_ROWS + MFMA_M - 1) / MFMA_M;
  constexpr int KV_TILES_N = (KV_TILE_SIZE + MFMA_N - 1) / MFMA_N;
  constexpr int HEAD_TILES_K = HEAD_DIM / MFMA_K;
  constexpr int OUT_TILES_N = HEAD_DIM / MFMA_N;
  constexpr int PV_TILES_K = (KV_TILE_SIZE + MFMA_K - 1) / MFMA_K;

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

  // Page indices in static shared memory
  __shared__ __align__(16) int page_indices[MAX_PAGES_PER_REQUEST];
  for (int i = threadIdx.x; i < num_pages; i += NUM_THREADS) {
    page_indices[i] = paged_kv_indices_buffer_ptr[first_page_pos + i];
  }
  __syncthreads();

  bf16 const *__restrict__ d_q =
      reinterpret_cast<bf16 const *>(qkv_ptr) + first_token_pos * QKV_STRIDE;
  bf16 const *__restrict__ d_k = d_q + NUM_QO_PER_KV * HEAD_DIM;
  bf16 const *__restrict__ d_v = d_k + HEAD_DIM;
  bf16 *__restrict__ d_paged_k_cache =
      reinterpret_cast<bf16 *>(paged_k_cache_ptr);
  bf16 *__restrict__ d_paged_v_cache =
      reinterpret_cast<bf16 *>(paged_v_cache_ptr);
  bf16 *__restrict__ d_output =
      reinterpret_cast<bf16 *>(output_ptr) + first_token_pos * O_STRIDE;

  extern __shared__ char smem[];

  // Shared memory layout
  constexpr size_t S_Q_OFFSET = 0;
  constexpr size_t S_Q_SIZE = sizeof(bf16) * Q_ROWS * HEAD_DIM;

  constexpr size_t S_K_OFFSET = S_Q_OFFSET + S_Q_SIZE;
  constexpr size_t S_K_SIZE = sizeof(bf16) * KV_TILE_SIZE * HEAD_DIM;

  constexpr size_t S_V_OFFSET = S_K_OFFSET + S_K_SIZE;
  constexpr size_t S_V_SIZE = sizeof(bf16) * KV_TILE_SIZE * HEAD_DIM;

  constexpr size_t S_SCORES_OFFSET = S_V_OFFSET + S_V_SIZE;
  constexpr size_t S_SCORES_SIZE = sizeof(float) * Q_ROWS * KV_TILE_SIZE;

  constexpr size_t S_REDUCE_OFFSET =
      ((S_SCORES_OFFSET + S_SCORES_SIZE + 15) & ~15);
  constexpr size_t S_REDUCE_SIZE = sizeof(float) * NUM_WARPS;

  constexpr size_t S_M_OFFSET = ((S_REDUCE_OFFSET + S_REDUCE_SIZE + 15) & ~15);
  constexpr size_t S_M_SIZE = sizeof(float) * Q_ROWS;

  constexpr size_t S_D_OFFSET = S_M_OFFSET + S_M_SIZE;
  constexpr size_t S_D_SIZE = sizeof(float) * Q_ROWS;

  constexpr size_t S_O_OFFSET = S_D_OFFSET + S_D_SIZE;
  constexpr size_t S_O_SIZE = sizeof(float) * Q_ROWS * HEAD_DIM;

  bf16 *s_q = reinterpret_cast<bf16 *>(smem + S_Q_OFFSET);
  bf16 *s_k = reinterpret_cast<bf16 *>(smem + S_K_OFFSET);
  bf16 *s_v = reinterpret_cast<bf16 *>(smem + S_V_OFFSET);
  float *s_scores = reinterpret_cast<float *>(smem + S_SCORES_OFFSET);
  float *s_reduce = reinterpret_cast<float *>(smem + S_REDUCE_OFFSET);
  float *s_m = reinterpret_cast<float *>(smem + S_M_OFFSET);
  float *s_d = reinterpret_cast<float *>(smem + S_D_OFFSET);
  float *s_o = reinterpret_cast<float *>(smem + S_O_OFFSET);

  int warp_idx = warp_id();
  int lane = threadIdx.x % 64;
  int q_rows = num_tokens * NUM_QO_PER_KV;

  // Initialize m, d, o
  for (int idx = threadIdx.x; idx < q_rows; idx += NUM_THREADS) {
    s_m[idx] = -INFINITY;
    s_d[idx] = 1.0f;
  }
  for (int idx = threadIdx.x; idx < q_rows * HEAD_DIM; idx += NUM_THREADS) {
    s_o[idx] = 0.0f;
  }

  // Load Q with vectorized access (8 bf16 per load = 16 bytes)
  constexpr int VEC_SIZE = 8;
  constexpr int HEAD_DIM_VECS = HEAD_DIM / VEC_SIZE;
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
      int pos = seq_len - num_tokens + token;
      bf16 const *cos_data =
          rope ? reinterpret_cast<bf16 const *>(cos_ptr) + pos * HEAD_DIM
               : nullptr;
      bf16 const *sin_data =
          rope ? reinterpret_cast<bf16 const *>(sin_ptr) + pos * HEAD_DIM
               : nullptr;
      for (int head = 0; head < NUM_QO_PER_KV; head++) {
        int row = token * NUM_QO_PER_KV + head;
        bf16 *q_head = s_q + row * HEAD_DIM;

        // Compute sum of squares
        float sum_sq = 0.0f;
        for (int i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
          float val = ck_tile::type_convert<float>(q_head[i]);
          sum_sq += val * val;
        }

        // Warp reduction
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

        // Apply RoPE if needed
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
      int pos = seq_len - num_tokens + token;
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

  bf16 const *k_weight = reinterpret_cast<bf16 const *>(k_norm_weight_ptr);
  int const num_iters = (seq_len + KV_TILE_SIZE - 1) / KV_TILE_SIZE;

  for (int iter = 0; iter < num_iters; iter++) {
    int tile_start = iter * KV_TILE_SIZE;
    int curr_iter_len = min(seq_len - tile_start, KV_TILE_SIZE);

    // Load KV tile with vectorized access
    int total_kv_vecs = curr_iter_len * HEAD_DIM_VECS;
    for (int vec_idx = threadIdx.x; vec_idx < total_kv_vecs;
         vec_idx += NUM_THREADS) {
      int row = vec_idx / HEAD_DIM_VECS;
      int vec_col = vec_idx % HEAD_DIM_VECS;
      int global_pos = tile_start + row;

      if (global_pos < seq_len - num_tokens) {
        // From KV cache
        int page_num = global_pos / PAGE_SIZE;
        int page_offset = global_pos % PAGE_SIZE;
        int page_idx = page_indices[page_num];
        int src_idx = page_idx * PAGE_SIZE + page_offset;
        vec_load_8(
            &s_k[row * HEAD_DIM + vec_col * VEC_SIZE],
            &d_paged_k_cache[src_idx * KV_CACHE_STRIDE + vec_col * VEC_SIZE]);
        vec_load_8(
            &s_v[row * HEAD_DIM + vec_col * VEC_SIZE],
            &d_paged_v_cache[src_idx * KV_CACHE_STRIDE + vec_col * VEC_SIZE]);
      } else {
        // From QKV input
        int qkv_row = global_pos - (seq_len - num_tokens);
        vec_load_8(&s_k[row * HEAD_DIM + vec_col * VEC_SIZE],
                   &d_k[qkv_row * QKV_STRIDE + vec_col * VEC_SIZE]);
        vec_load_8(&s_v[row * HEAD_DIM + vec_col * VEC_SIZE],
                   &d_v[qkv_row * QKV_STRIDE + vec_col * VEC_SIZE]);
      }
    }

    // Zero padding for remaining rows (use scalar for simplicity since it's a
    // small amount)
    for (int idx = threadIdx.x + curr_iter_len * HEAD_DIM;
         idx < KV_TILE_SIZE * HEAD_DIM;
         idx += NUM_THREADS) {
      s_k[idx] = bf16(0.0f);
      s_v[idx] = bf16(0.0f);
    }
    __syncthreads();

    // Apply QK norm and/or RoPE to new K tokens
    int first_new_token = max(tile_start, seq_len - num_tokens);
    int last_new_token = min(tile_start + curr_iter_len, seq_len);
    int num_new_tokens = max(0, last_new_token - first_new_token);

    if (num_new_tokens > 0) {
      int local_start = first_new_token - tile_start;

      if (qk_norm) {
        for (int kv_idx = 0; kv_idx < num_new_tokens; kv_idx++) {
          int pos = first_new_token + kv_idx;
          int local_row = local_start + kv_idx;
          bf16 *k_head = s_k + local_row * HEAD_DIM;
          bf16 const *cos_data =
              rope ? reinterpret_cast<bf16 const *>(cos_ptr) + pos * HEAD_DIM
                   : nullptr;
          bf16 const *sin_data =
              rope ? reinterpret_cast<bf16 const *>(sin_ptr) + pos * HEAD_DIM
                   : nullptr;

          // Compute sum of squares
          float sum_sq = 0.0f;
          for (int i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
            float val = ck_tile::type_convert<float>(k_head[i]);
            sum_sq += val * val;
          }

          // Warp reduction
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

          // Apply normalization
          for (int i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
            float val = ck_tile::type_convert<float>(k_head[i]) * rms_rcp *
                        ck_tile::type_convert<float>(k_weight[i]);
            k_head[i] = ck_tile::type_convert<bf16>(val);
          }
          __syncthreads();

          // Apply RoPE if needed
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
        for (int kv_idx = 0; kv_idx < num_new_tokens; kv_idx++) {
          int pos = first_new_token + kv_idx;
          int local_row = local_start + kv_idx;
          bf16 const *cos_data =
              reinterpret_cast<bf16 const *>(cos_ptr) + pos * HEAD_DIM;
          bf16 const *sin_data =
              reinterpret_cast<bf16 const *>(sin_ptr) + pos * HEAD_DIM;
          for (int d = threadIdx.x; d < HEAD_DIM / 2; d += NUM_THREADS) {
            float k0 =
                ck_tile::type_convert<float>(s_k[local_row * HEAD_DIM + d]);
            float k1 = ck_tile::type_convert<float>(
                s_k[local_row * HEAD_DIM + d + HEAD_DIM / 2]);
            float c = ck_tile::type_convert<float>(cos_data[d]);
            float s = ck_tile::type_convert<float>(sin_data[d]);
            s_k[local_row * HEAD_DIM + d] =
                ck_tile::type_convert<bf16>(k0 * c - k1 * s);
            s_k[local_row * HEAD_DIM + d + HEAD_DIM / 2] =
                ck_tile::type_convert<bf16>(k0 * s + k1 * c);
          }
        }
        __syncthreads();
      }

      // Write processed K/V to cache with vectorized access
      int total_new_vecs = num_new_tokens * HEAD_DIM_VECS;
      for (int vec_idx = threadIdx.x; vec_idx < total_new_vecs;
           vec_idx += NUM_THREADS) {
        int token = vec_idx / HEAD_DIM_VECS;
        int vec_col = vec_idx % HEAD_DIM_VECS;
        int global_pos = first_new_token + token;
        int page_num = global_pos / PAGE_SIZE;
        int page_offset = global_pos % PAGE_SIZE;
        int page_idx = page_indices[page_num];
        int dst_idx = page_idx * PAGE_SIZE + page_offset;
        int local_row = local_start + token;
        vec_store_8(
            &d_paged_k_cache[dst_idx * KV_CACHE_STRIDE + vec_col * VEC_SIZE],
            &s_k[local_row * HEAD_DIM + vec_col * VEC_SIZE]);
        vec_store_8(
            &d_paged_v_cache[dst_idx * KV_CACHE_STRIDE + vec_col * VEC_SIZE],
            &s_v[local_row * HEAD_DIM + vec_col * VEC_SIZE]);
      }
      __syncthreads();
    }

    // Initialize scores
    for (int idx = threadIdx.x; idx < Q_ROWS * KV_TILE_SIZE;
         idx += NUM_THREADS) {
      s_scores[idx] = 0.0f;
    }
    __syncthreads();

    // =========================================================================
    // MFMA Q * K^T computation
    // =========================================================================
    if (warp_idx < KV_TILES_N) {
      int tile_n = warp_idx;

      for (int tile_m = 0; tile_m < Q_TILES_M; tile_m++) {
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
        for (int tile_k = 0; tile_k < HEAD_TILES_K; tile_k++) {
          int k_base = tile_k * MFMA_K;

          // Load Q fragment
          int q_row = tile_m * MFMA_M + (lane % 16);
          int q_col = k_base + (lane / 16) * 4;

          bf16 q_frag[4];
#pragma unroll
          for (int i = 0; i < 4; i++) {
            q_frag[i] = (q_row < q_rows) ? s_q[q_row * HEAD_DIM + q_col + i]
                                         : bf16(0.0f);
          }

          // Load K fragment
          int k_row = tile_n * MFMA_N + (lane % 16);
          int k_col = k_base + (lane / 16) * 4;

          bf16 k_frag[4];
#pragma unroll
          for (int i = 0; i < 4; i++) {
            k_frag[i] = (k_row < curr_iter_len)
                            ? s_k[k_row * HEAD_DIM + k_col + i]
                            : bf16(0.0f);
          }

          // Execute MFMA
          mfma_bf16x4 a_vec, b_vec;
#pragma unroll
          for (int i = 0; i < 4; i++) {
            a_vec[i] = ck_tile::bit_cast<short>(q_frag[i]);
            b_vec[i] = ck_tile::bit_cast<short>(k_frag[i]);
          }

          mfma_float4 c_vec = {acc[0], acc[1], acc[2], acc[3]};
          c_vec = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
              a_vec, b_vec, c_vec, 0, 0, 0);
          acc[0] = c_vec[0];
          acc[1] = c_vec[1];
          acc[2] = c_vec[2];
          acc[3] = c_vec[3];
        }

        // Store scores with causal masking
        int out_row_base = tile_m * MFMA_M + (lane / 16) * 4;
        int out_col = tile_n * MFMA_N + (lane % 16);

        if (out_col < curr_iter_len) {
#pragma unroll
          for (int r = 0; r < 4; r++) {
            int out_row = out_row_base + r;
            if (out_row < q_rows) {
              // Causal mask
              int token_idx = out_row / NUM_QO_PER_KV;
              int q_pos = seq_len - num_tokens + token_idx;
              int kv_pos = tile_start + out_col;
              float score = acc[r] * sm_scale;
              if (kv_pos > q_pos) {
                score = -INFINITY;
              }
              s_scores[out_row * KV_TILE_SIZE + out_col] = score;
            }
          }
        }
      }
    }
    __syncthreads();

    // Online softmax update (scalar per thread)
    for (int qr = threadIdx.x; qr < q_rows; qr += NUM_THREADS) {
      float m_local = -INFINITY;
      for (int kc = 0; kc < curr_iter_len; kc++) {
        m_local = fmaxf(m_local, s_scores[qr * KV_TILE_SIZE + kc]);
      }

      float m_prev = s_m[qr];
      float m_new = fmaxf(m_prev, m_local);
      float rescale = ptx_exp2(m_prev - m_new);

      // Rescale previous O
      for (int d = 0; d < HEAD_DIM; d++) {
        s_o[qr * HEAD_DIM + d] *= rescale;
      }

      // Compute normalized attention weights
      float d_local = 0.0f;
      for (int kc = 0; kc < curr_iter_len; kc++) {
        float attn = ptx_exp2(s_scores[qr * KV_TILE_SIZE + kc] - m_new);
        s_scores[qr * KV_TILE_SIZE + kc] = attn; // Store for P*V
        d_local += attn;
      }

      s_m[qr] = m_new;
      s_d[qr] = s_d[qr] * rescale + d_local;
    }
    __syncthreads();

    // =========================================================================
    // MFMA P * V computation
    // =========================================================================
    for (int out_tile = warp_idx; out_tile < OUT_TILES_N;
         out_tile += NUM_WARPS) {
      for (int tile_m = 0; tile_m < Q_TILES_M; tile_m++) {
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int tile_k = 0; tile_k < PV_TILES_K; tile_k++) {
          int k_base = tile_k * MFMA_K;
          if (k_base >= curr_iter_len) {
            break;
          }

          // Load P fragment (attention weights)
          int p_row = tile_m * MFMA_M + (lane % 16);
          int p_col = k_base + (lane / 16) * 4;

          bf16 p_frag[4];
#pragma unroll
          for (int i = 0; i < 4; i++) {
            float val = (p_row < q_rows && p_col + i < curr_iter_len)
                            ? s_scores[p_row * KV_TILE_SIZE + p_col + i]
                            : 0.0f;
            p_frag[i] = ck_tile::type_convert<bf16>(val);
          }

          // Load V fragment (transposed access)
          int v_row = k_base + (lane / 16) * 4;
          int v_col = out_tile * MFMA_N + (lane % 16);

          bf16 v_frag[4];
#pragma unroll
          for (int i = 0; i < 4; i++) {
            int vr = v_row + i;
            v_frag[i] =
                (vr < curr_iter_len) ? s_v[vr * HEAD_DIM + v_col] : bf16(0.0f);
          }

          // Execute MFMA
          mfma_bf16x4 a_vec, b_vec;
#pragma unroll
          for (int i = 0; i < 4; i++) {
            a_vec[i] = ck_tile::bit_cast<short>(p_frag[i]);
            b_vec[i] = ck_tile::bit_cast<short>(v_frag[i]);
          }

          mfma_float4 c_vec = {acc[0], acc[1], acc[2], acc[3]};
          c_vec = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
              a_vec, b_vec, c_vec, 0, 0, 0);
          acc[0] = c_vec[0];
          acc[1] = c_vec[1];
          acc[2] = c_vec[2];
          acc[3] = c_vec[3];
        }

        // Accumulate to output
        int out_row_base = tile_m * MFMA_M + (lane / 16) * 4;
        int out_col = out_tile * MFMA_N + (lane % 16);

#pragma unroll
        for (int r = 0; r < 4; r++) {
          int out_row = out_row_base + r;
          if (out_row < q_rows && out_col < HEAD_DIM) {
            s_o[out_row * HEAD_DIM + out_col] += acc[r];
          }
        }
      }
    }
    __syncthreads();
  }

  // Final normalization and output with vectorized writes
  int total_out_vecs = num_tokens * NUM_QO_PER_KV * HEAD_DIM_VECS;
  for (int vec_idx = threadIdx.x; vec_idx < total_out_vecs;
       vec_idx += NUM_THREADS) {
    int row = vec_idx / HEAD_DIM_VECS;
    int vec_col = vec_idx % HEAD_DIM_VECS;
    int token = row / NUM_QO_PER_KV;
    int head = row % NUM_QO_PER_KV;

    // Load 8 float values, normalize, convert to bf16, and store
    bf16 out_buf[8];
    float inv_d = 1.0f / s_d[row];
#pragma unroll
    for (int i = 0; i < 8; i++) {
      float val = s_o[row * HEAD_DIM + vec_col * VEC_SIZE + i] * inv_d;
      out_buf[i] = ck_tile::type_convert<bf16>(val);
    }

    int dst_col = head * HEAD_DIM + vec_col * VEC_SIZE;
    vec_store_8(&d_output[token * O_STRIDE + dst_col], out_buf);
  }
}

} // namespace kernel
