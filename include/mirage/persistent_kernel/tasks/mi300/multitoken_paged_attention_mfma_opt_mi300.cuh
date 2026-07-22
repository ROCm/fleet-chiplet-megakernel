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

// MFMA vector types for AMD attention (optimized version)
typedef short __attribute__((ext_vector_type(4))) mfma_attn_opt_short4;
typedef float __attribute__((ext_vector_type(4))) mfma_attn_opt_float4;

union mfma_attn_opt_bf16x4 {
  unsigned short u16[4];
  uint64_t u64;
  mfma_attn_opt_short4 vec;
};

union mfma_attn_opt_bf16x8 {
  unsigned short u16[8];
  __uint128_t u128;
};

// Portable float to bfloat16 conversion for HIP
__device__ __forceinline__ unsigned short float_to_bf16_bits(float val) {
  union { float f; unsigned int u; } bits;
  bits.f = val;
  // Round to nearest even
  unsigned int rounding_bias = ((bits.u >> 16) & 1) + 0x7FFF;
  return static_cast<unsigned short>((bits.u + rounding_bias) >> 16);
}

// Optimized MFMA-based paged attention implementation for AMD MI300
// Uses MFMA 16x16x16 bf16 instructions with vectorized tile loads
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
__device__ __forceinline__ void multitoken_paged_attention_mfma_opt(
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
  constexpr int KV_TILE_SIZE = 32;  // KV tile for attention computation
  constexpr int MAX_PAGES_PER_REQUEST = (MAX_SEQ_LEN + PAGE_SIZE - 1) / PAGE_SIZE;
  constexpr float log2e = 1.4426950408889634f;
  constexpr int Q_ROWS = MAX_TOKENS * NUM_QO_PER_KV;

  // MFMA tile dimensions
  constexpr int MFMA_M = 16;
  constexpr int MFMA_N = 16;
  constexpr int MFMA_K = 16;

  // Number of MFMA tiles
  constexpr int Q_TILES_M = (Q_ROWS + MFMA_M - 1) / MFMA_M;  // Tiles for Q rows
  constexpr int KV_TILES_N = (KV_TILE_SIZE + MFMA_N - 1) / MFMA_N;  // Tiles for KV positions
  constexpr int HEAD_TILES_K = HEAD_DIM / MFMA_K;  // Tiles for head dimension

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

  int tid = threadIdx.x;
  int lane = tid % 64;
  int wavefront_id = tid / 64;

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

  // Shared memory layout - optimized for 64KB LDS
  constexpr size_t ALIGN = 128;
  constexpr size_t S_Q_SIZE = sizeof(T) * Q_ROWS * HEAD_DIM;
  constexpr size_t S_K_SIZE = sizeof(T) * KV_TILE_SIZE * HEAD_DIM;
  constexpr size_t S_V_SIZE = sizeof(T) * KV_TILE_SIZE * HEAD_DIM;
  constexpr size_t S_SCORES_SIZE = sizeof(float) * Q_ROWS * KV_TILE_SIZE;
  constexpr size_t S_O_SIZE = sizeof(float) * Q_ROWS * HEAD_DIM;
  constexpr size_t S_M_SIZE = sizeof(float) * Q_ROWS;
  constexpr size_t S_D_SIZE = sizeof(float) * Q_ROWS;
  constexpr size_t S_REDUCE_SIZE = sizeof(float) * NUM_WARPS;

  constexpr size_t S_Q_OFFSET = ALIGN;
  constexpr size_t S_K_OFFSET = S_Q_OFFSET + S_Q_SIZE;
  constexpr size_t S_V_OFFSET = S_K_OFFSET + S_K_SIZE;
  constexpr size_t S_SCORES_OFFSET = S_V_OFFSET + S_V_SIZE;
  constexpr size_t S_O_OFFSET = S_SCORES_OFFSET + S_SCORES_SIZE;
  constexpr size_t S_M_OFFSET = S_O_OFFSET + S_O_SIZE;
  constexpr size_t S_D_OFFSET = S_M_OFFSET + S_M_SIZE;
  constexpr size_t S_REDUCE_OFFSET = S_D_OFFSET + S_D_SIZE;

  T *s_q = reinterpret_cast<T *>(smem + S_Q_OFFSET);
  T *s_k = reinterpret_cast<T *>(smem + S_K_OFFSET);
  T *s_v = reinterpret_cast<T *>(smem + S_V_OFFSET);
  float *s_scores = reinterpret_cast<float *>(smem + S_SCORES_OFFSET);
  float *s_o = reinterpret_cast<float *>(smem + S_O_OFFSET);
  float *s_m = reinterpret_cast<float *>(smem + S_M_OFFSET);
  float *s_d = reinterpret_cast<float *>(smem + S_D_OFFSET);
  float *s_reduce = reinterpret_cast<float *>(smem + S_REDUCE_OFFSET);

  int warp_idx = warp_id();
  int q_rows = num_tokens * NUM_QO_PER_KV;

  // Initialize m, d, o
  for (int idx = tid; idx < q_rows; idx += NUM_THREADS) {
    s_m[idx] = -INFINITY;
    s_d[idx] = 1.0f;
  }
  for (int idx = tid; idx < q_rows * HEAD_DIM; idx += NUM_THREADS) {
    s_o[idx] = 0.0f;
  }

  // Load Q with vectorized loads (8 bf16 = 128 bits)
  constexpr int Q_VEC_LOADS = (Q_ROWS * HEAD_DIM) / 8;
  for (int idx = tid; idx < Q_VEC_LOADS; idx += NUM_THREADS) {
    int elem_idx = idx * 8;
    int row = elem_idx / HEAD_DIM;
    int col = elem_idx % HEAD_DIM;
    if (row < num_tokens * NUM_QO_PER_KV && col + 7 < HEAD_DIM) {
      int token = row / NUM_QO_PER_KV;
      int head = row % NUM_QO_PER_KV;
      mfma_attn_opt_bf16x8 vec;
      #pragma unroll
      for (int i = 0; i < 8; i++) {
        vec.u16[i] = reinterpret_cast<unsigned short const*>(
            &d_q[token * QKV_STRIDE + head * HEAD_DIM + col + i])[0];
      }
      *reinterpret_cast<__uint128_t*>(&s_q[row * HEAD_DIM + col]) = vec.u128;
    }
  }
  __syncthreads();

  // Apply RoPE to Q if needed (simplified - skip QK norm for now)
  if (rope && !qk_norm) {
    for (int token = 0; token < num_tokens; token++) {
      int pos = seq_len - num_tokens + token;
      T const *cos_data = reinterpret_cast<T const *>(cos_ptr) + pos * HEAD_DIM;
      T const *sin_data = reinterpret_cast<T const *>(sin_ptr) + pos * HEAD_DIM;
      for (int idx = tid; idx < NUM_QO_PER_KV * HEAD_DIM / 2; idx += NUM_THREADS) {
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

  int const num_iters = (seq_len + KV_TILE_SIZE - 1) / KV_TILE_SIZE;

  for (int iter = 0; iter < num_iters; iter++) {
    int tile_start = iter * KV_TILE_SIZE;
    int curr_iter_len = min(seq_len - tile_start, KV_TILE_SIZE);

    // Load K and V tiles with vectorized loads
    constexpr int KV_VEC_LOADS = (KV_TILE_SIZE * HEAD_DIM) / 8;
    for (int idx = tid; idx < KV_VEC_LOADS; idx += NUM_THREADS) {
      int elem_idx = idx * 8;
      int row = elem_idx / HEAD_DIM;
      int col = elem_idx % HEAD_DIM;
      int global_pos = tile_start + row;

      mfma_attn_opt_bf16x8 vec_k, vec_v;
      if (row < curr_iter_len && col + 7 < HEAD_DIM) {
        if (global_pos < seq_len - num_tokens) {
          // From KV cache
          int page_num = global_pos / PAGE_SIZE;
          int page_offset = global_pos % PAGE_SIZE;
          int page_idx = page_indices[page_num];
          int src_idx = page_idx * PAGE_SIZE + page_offset;
          #pragma unroll
          for (int i = 0; i < 8; i++) {
            vec_k.u16[i] = reinterpret_cast<unsigned short const*>(
                &d_paged_k_cache[src_idx * KV_CACHE_STRIDE + col + i])[0];
            vec_v.u16[i] = reinterpret_cast<unsigned short const*>(
                &d_paged_v_cache[src_idx * KV_CACHE_STRIDE + col + i])[0];
          }
        } else {
          // From QKV input
          int qkv_row = global_pos - (seq_len - num_tokens);
          #pragma unroll
          for (int i = 0; i < 8; i++) {
            vec_k.u16[i] = reinterpret_cast<unsigned short const*>(
                &d_k[qkv_row * QKV_STRIDE + col + i])[0];
            vec_v.u16[i] = reinterpret_cast<unsigned short const*>(
                &d_v[qkv_row * QKV_STRIDE + col + i])[0];
          }
        }
      } else {
        vec_k.u128 = 0;
        vec_v.u128 = 0;
      }
      *reinterpret_cast<__uint128_t*>(&s_k[row * HEAD_DIM + col]) = vec_k.u128;
      *reinterpret_cast<__uint128_t*>(&s_v[row * HEAD_DIM + col]) = vec_v.u128;
    }
    __syncthreads();

    // Apply RoPE to new K tokens and write to cache
    int first_new_token = max(tile_start, seq_len - num_tokens);
    int last_new_token = min(tile_start + curr_iter_len, seq_len);
    int num_new_tokens = max(0, last_new_token - first_new_token);

    if (num_new_tokens > 0 && rope && !qk_norm) {
      int local_start = first_new_token - tile_start;
      for (int kv_idx = 0; kv_idx < num_new_tokens; kv_idx++) {
        int pos = first_new_token + kv_idx;
        int local_row = local_start + kv_idx;
        T const *cos_data = reinterpret_cast<T const *>(cos_ptr) + pos * HEAD_DIM;
        T const *sin_data = reinterpret_cast<T const *>(sin_ptr) + pos * HEAD_DIM;
        for (int d = tid; d < HEAD_DIM / 2; d += NUM_THREADS) {
          float k0 = static_cast<float>(s_k[local_row * HEAD_DIM + d]);
          float k1 = static_cast<float>(s_k[local_row * HEAD_DIM + d + HEAD_DIM / 2]);
          float c = static_cast<float>(cos_data[d]);
          float s = static_cast<float>(sin_data[d]);
          s_k[local_row * HEAD_DIM + d] = static_cast<T>(k0 * c - k1 * s);
          s_k[local_row * HEAD_DIM + d + HEAD_DIM / 2] = static_cast<T>(k0 * s + k1 * c);
        }
      }
      __syncthreads();

      // Write processed K/V to cache
      for (int idx = tid; idx < num_new_tokens * HEAD_DIM; idx += NUM_THREADS) {
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

    // Initialize scores to zero
    for (int idx = tid; idx < q_rows * KV_TILE_SIZE; idx += NUM_THREADS) {
      s_scores[idx] = 0.0f;
    }
    __syncthreads();

    // =========================================================================
    // MFMA: Compute Q * K^T -> scores [Q_ROWS x KV_TILE_SIZE]
    // Q: [Q_ROWS, HEAD_DIM], K^T: [HEAD_DIM, KV_TILE_SIZE]
    // =========================================================================
    if (wavefront_id < KV_TILES_N) {
      int tile_n = wavefront_id;  // Each wavefront handles one KV tile column

      for (int tile_m = 0; tile_m < Q_TILES_M; tile_m++) {
        mfma_attn_opt_float4 accum = {0.0f, 0.0f, 0.0f, 0.0f};

        // Iterate over HEAD_DIM in MFMA_K chunks
        #pragma unroll
        for (int tile_k = 0; tile_k < HEAD_TILES_K; tile_k++) {
          int k_base = tile_k * MFMA_K;

          // Load Q fragment: Q[tile_m*16 + (lane%16), k_base + (lane/16)*4 : +4]
          int q_row = tile_m * MFMA_M + (lane % 16);
          int q_col = k_base + (lane / 16) * 4;

          mfma_attn_opt_bf16x4 reg_q;
          if (q_row < q_rows) {
            reg_q.u64 = *reinterpret_cast<uint64_t const*>(
                &s_q[q_row * HEAD_DIM + q_col]);
          } else {
            reg_q.u64 = 0;
          }

          // Load K^T fragment: K[tile_n*16 + (lane%16), k_base + (lane/16)*4 : +4]
          // K is stored as [KV_TILE_SIZE, HEAD_DIM], we need K^T
          int k_row = tile_n * MFMA_N + (lane % 16);  // KV position
          int k_col = k_base + (lane / 16) * 4;  // HEAD_DIM

          mfma_attn_opt_bf16x4 reg_k;
          if (k_row < curr_iter_len) {
            reg_k.u64 = *reinterpret_cast<uint64_t const*>(
                &s_k[k_row * HEAD_DIM + k_col]);
          } else {
            reg_k.u64 = 0;
          }

          accum = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
              reg_q.vec, reg_k.vec, accum, 0, 0, 0);
        }

        // Store scores and apply scale + causal mask
        int out_row_base = tile_m * MFMA_M + (lane / 16) * 4;
        int out_col = tile_n * MFMA_N + (lane % 16);

        if (out_col < curr_iter_len) {
          int kv_pos = tile_start + out_col;
          #pragma unroll
          for (int r = 0; r < 4; r++) {
            int out_row = out_row_base + r;
            if (out_row < q_rows) {
              int token = out_row / NUM_QO_PER_KV;
              int q_pos = seq_len - num_tokens + token;
              float score = accum[r] * sm_scale;
              // Causal mask
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

    // =========================================================================
    // Online softmax: compute max and rescale previous output
    // =========================================================================
    for (int qr = tid; qr < q_rows; qr += NUM_THREADS) {
      // Find local max
      float m_local = -INFINITY;
      for (int kc = 0; kc < curr_iter_len; kc++) {
        float score = s_scores[qr * KV_TILE_SIZE + kc];
        if (score > m_local) m_local = score;
      }

      // Online softmax update
      float m_prev = s_m[qr];
      float m_new = fmaxf(m_prev, m_local);
      float rescale = ptx_exp2(m_prev - m_new);

      // Rescale previous O
      for (int d = 0; d < HEAD_DIM; d++) {
        s_o[qr * HEAD_DIM + d] *= rescale;
      }

      // Compute softmax denominator and convert scores to attention probs
      float d_local = 0.0f;
      for (int kc = 0; kc < curr_iter_len; kc++) {
        float attn = ptx_exp2(s_scores[qr * KV_TILE_SIZE + kc] - m_new);
        s_scores[qr * KV_TILE_SIZE + kc] = attn;  // Store attention probs
        d_local += attn;
      }

      s_m[qr] = m_new;
      s_d[qr] = s_d[qr] * rescale + d_local;
    }
    __syncthreads();

    // =========================================================================
    // MFMA: Compute P * V -> output [Q_ROWS x HEAD_DIM]
    // P: [Q_ROWS, KV_TILE_SIZE], V: [KV_TILE_SIZE, HEAD_DIM]
    // =========================================================================
    constexpr int HEAD_TILES_N = HEAD_DIM / MFMA_N;  // Output tiles
    constexpr int PV_TILES_K = (KV_TILE_SIZE + MFMA_K - 1) / MFMA_K;

    // Each wavefront handles different output tiles
    int tiles_per_warp = (HEAD_TILES_N + NUM_WARPS - 1) / NUM_WARPS;

    for (int tile_idx = 0; tile_idx < tiles_per_warp; tile_idx++) {
      int out_tile = wavefront_id * tiles_per_warp + tile_idx;
      if (out_tile >= HEAD_TILES_N) break;

      for (int tile_m = 0; tile_m < Q_TILES_M; tile_m++) {
        mfma_attn_opt_float4 accum = {0.0f, 0.0f, 0.0f, 0.0f};

        // Iterate over KV dimension in MFMA_K chunks
        for (int tile_k = 0; tile_k < PV_TILES_K; tile_k++) {
          int k_base = tile_k * MFMA_K;
          if (k_base >= curr_iter_len) break;

          // Load P fragment (fp32 attention probs -> bf16)
          int p_row = tile_m * MFMA_M + (lane % 16);
          int p_col = k_base + (lane / 16) * 4;

          mfma_attn_opt_bf16x4 reg_p;
          #pragma unroll
          for (int i = 0; i < 4; i++) {
            float val = 0.0f;
            if (p_row < q_rows && p_col + i < curr_iter_len) {
              val = s_scores[p_row * KV_TILE_SIZE + p_col + i];
            }
            reg_p.u16[i] = float_to_bf16_bits(val);
          }

          // Load V fragment
          int v_row = k_base + (lane / 16) * 4;
          int v_col = out_tile * MFMA_N + (lane % 16);

          mfma_attn_opt_bf16x4 reg_v;
          #pragma unroll
          for (int i = 0; i < 4; i++) {
            int vr = v_row + i;
            if (vr < curr_iter_len) {
              reg_v.u16[i] = reinterpret_cast<unsigned short const*>(
                  &s_v[vr * HEAD_DIM + v_col])[0];
            } else {
              reg_v.u16[i] = 0;
            }
          }

          accum = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
              reg_p.vec, reg_v.vec, accum, 0, 0, 0);
        }

        // Accumulate to output
        int out_row_base = tile_m * MFMA_M + (lane / 16) * 4;
        int out_col = out_tile * MFMA_N + (lane % 16);

        #pragma unroll
        for (int r = 0; r < 4; r++) {
          int out_row = out_row_base + r;
          if (out_row < q_rows && out_col < HEAD_DIM) {
            atomicAdd(&s_o[out_row * HEAD_DIM + out_col], accum[r]);
          }
        }
      }
    }
    __syncthreads();
  }

  // Final normalization and output with vectorized stores
  for (int idx = tid; idx < num_tokens * NUM_QO_PER_KV * HEAD_DIM; idx += NUM_THREADS) {
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
