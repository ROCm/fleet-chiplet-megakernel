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

// this kernel merges the result of one KV head chunk back to the full KV
// cache，
// it taks the output of multitoken_paged_attention_task_impl_32_64_split_kv and
// the log exp sum as input,
namespace kernel {

template <typename T,
          int NUM_QO_HEADS_PER_KV,
          int NUM_KV_HEADS,
          int NUM_QO_GROUPS,
          int HEAD_DIM,
          int MAX_TOKENS = 8,
          bool PARTITION_KV = true,
          int NUM_KV_CHUNKS = 1,
          int KV_CHUNK_SIZE = 256,
          int PAGE_SIZE = 4096,
          typename OaccT = T>
__device__ __forceinline__ void
    merge_splitkv(void const *lse,
                  void const *o,
                  int const *qo_indptr_buffer_ptr,
                  int const *paged_kv_indptr_buffer_ptr,
                  int const *paged_kv_last_page_len_buffer_ptr,
                  int16_t request_id,
                  void *output,
                  int merge_task_offset) {
#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  unsigned long long _t0 = __builtin_amdgcn_s_memrealtime();
#endif
  OaccT const *o_ptr = reinterpret_cast<OaccT const *>(o);
  T *output_ptr = reinterpret_cast<T *>(output);
  float const *lse_ptr = reinterpret_cast<float const *>(lse);
  // constexpr int GLOBAL_ITERS_M = (NUM_QO_HEADS_PER_KV + 64 - 1) / 64;

  int const first_page_pos = paged_kv_indptr_buffer_ptr[request_id];
  int const last_page_pos = paged_kv_indptr_buffer_ptr[request_id + 1];
  int const num_pages = last_page_pos - first_page_pos;
  int seq_len = (num_pages - 1) * PAGE_SIZE +
                paged_kv_last_page_len_buffer_ptr[request_id];

  int num_chunks = (seq_len + KV_CHUNK_SIZE - 1) / KV_CHUNK_SIZE;

  // size of o and output is NUM_QO_HEADS_PER_KV * MAX_TOKENS * 128
  // let each thread process one
  //  constexpr int NUM_QO_PER_KV = NUM_QO_HEADS_PER_KV / NUM_KV_HEADS;
  //  constexpr int NUM_Q = MAX_TOKENS * NUM_QO_PER_KV;
  //  constexpr int GLOBAL_ITERS_M = (NUM_Q + 64 - 1) / 64;

  int const first_token_pos = qo_indptr_buffer_ptr[request_id];
  int const last_token_pos = qo_indptr_buffer_ptr[request_id + 1];
  // Exit the current task is number of query tokens is zero
  if (first_token_pos == last_token_pos) {
    return;
  }
  int const num_tokens = last_token_pos - first_token_pos;

  constexpr int THREADS_PER_TOKEN = 16; // let 16 threads process one head
  constexpr int VAL_PER_THREAD = HEAD_DIM / THREADS_PER_TOKEN;
  constexpr int num_groups = NUM_THREADS / THREADS_PER_TOKEN;

  int thread_in_group = threadIdx.x % THREADS_PER_TOKEN;
  int group_id = threadIdx.x / THREADS_PER_TOKEN;
  int head_partition = thread_in_group;

  // let 16 threads to process one head_dim
#pragma unroll 1
  for (int tok = group_id; tok < num_tokens * NUM_QO_HEADS_PER_KV;
       tok += num_groups) {

    int token_idx = tok / NUM_QO_HEADS_PER_KV;
    int head_idx = tok % NUM_QO_HEADS_PER_KV;

#pragma unroll 1
    for (int i = 0; i < VAL_PER_THREAD; ++i) {
      float m_global = -inf;
      float d_global = 1.f;
      float o_global = 0.f;
#pragma unroll
      for (int kv_idx = 0; kv_idx < num_chunks; ++kv_idx) {
        // process 8 tokens
        float m_prev = m_global,
              d_prev = d_global; // save previous values
        // int lse_offset = kv_idx * (MAX_TOKENS * NUM_QO_HEADS_PER_KV) + tok;
        // int o_offset = (kv_idx * (MAX_TOKENS * NUM_QO_HEADS_PER_KV) + tok) *
        // HEAD_DIM +  head_partition * VAL_PER_THREAD + i;

        int lse_offset = head_idx + kv_idx * NUM_QO_HEADS_PER_KV +
                         (first_token_pos + token_idx) * NUM_QO_GROUPS *
                             NUM_KV_CHUNKS * NUM_QO_HEADS_PER_KV;
        // int lse_offset = merge_task_offset * NUM_QO_HEADS_PER_KV + head_idx +
        // kv_idx * NUM_QO_HEADS_PER_KV + token_idx * NUM_QO_GROUPS *
        // NUM_KV_CHUNKS * NUM_QO_HEADS_PER_KV;
        int o_offset =
            lse_offset * HEAD_DIM + head_partition * VAL_PER_THREAD + i;

        float other_m = lse_ptr[lse_offset], other_d = 1;
        m_global = max(m_prev, other_m);
        d_global = d_prev * ptx_exp2(m_prev - m_global) +
                   other_d * ptx_exp2(other_m - m_global);
        // accumulate o
        float other_o = (float)o_ptr[o_offset];

        o_global = o_global * ptx_exp2(m_prev - m_global) +
                   other_o * ptx_exp2(other_m - m_global);
      }
      T out_val = (T)__fdividef(o_global, d_global);
      output_ptr[(first_token_pos + token_idx) * NUM_QO_GROUPS *
                     NUM_QO_HEADS_PER_KV * HEAD_DIM +
                 head_idx * HEAD_DIM + head_partition * VAL_PER_THREAD + i] =
          out_val;
    }
  }
#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  __syncthreads();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    unsigned long long _dur = (__builtin_amdgcn_s_memrealtime() - _t0) * 10;
    printf("[ATTN_MERGE] dur_us=%.1f\n", (double)_dur / 1000.0);
  }
#endif
}

// CK FMHA merge variant: reads float o_acc/lse_acc with interleaved kv_head
// layout lse layout:  lse[token * (NUM_QO_GROUPS * NUM_KV_CHUNKS * QO_PER_KV)
//                  + kv_head * NUM_KV_CHUNKS * QO_PER_KV
//                  + chunk * QO_PER_KV + head]
// o layout:    same indexing * HEAD_DIM + d
// output layout: output[token * NUM_QO_GROUPS * QO_PER_KV * HEAD_DIM
//                       + kv_head * QO_PER_KV * HEAD_DIM + head * HEAD_DIM + d]
template <typename T,
          int NUM_QO_HEADS_PER_KV,
          int NUM_QO_GROUPS,
          int HEAD_DIM,
          int NUM_KV_CHUNKS,
          int KV_CHUNK_SIZE = 128,
          int PAGE_SIZE = 4096,
          bool WRITE_THROUGH = false>
__device__ __forceinline__ void
    merge_splitkv_ck_fmha(float const *lse_ptr,
                          float const *o_ptr,
                          int const *qo_indptr_buffer_ptr,
                          int const *paged_kv_indptr_buffer_ptr,
                          int const *paged_kv_last_page_len_buffer_ptr,
                          int16_t request_id,
                          T *output_ptr,
                          int kv_head_idx,
                          void const *sinks_ptr = nullptr) {

  int const first_token_pos = qo_indptr_buffer_ptr[request_id];
  int const last_token_pos = qo_indptr_buffer_ptr[request_id + 1];
  if (first_token_pos == last_token_pos) {
    return;
  }
  int const num_tokens = last_token_pos - first_token_pos;

  // Use NUM_KV_CHUNKS (template param) as the chunk count — the CK FMHA
  // pipeline processes exactly NUM_KV_CHUNKS chunks worth of data into
  // o_acc/lse_acc
  constexpr int num_chunks = NUM_KV_CHUNKS;

  // Full token stride matches CK FMHA's LSE_STRIDE = NUM_KV_HEADS *
  // NUM_KV_CHUNKS * QO_PER_KV
  constexpr int LSE_TOKEN_STRIDE =
      NUM_QO_GROUPS * NUM_KV_CHUNKS * NUM_QO_HEADS_PER_KV;
  // Offset to this kv_head's slice within one token
  int const lse_kv_offset = kv_head_idx * NUM_KV_CHUNKS * NUM_QO_HEADS_PER_KV;
  // Output stride (full width across all kv_heads)
  constexpr int OUT_TOKEN_STRIDE =
      NUM_QO_GROUPS * NUM_QO_HEADS_PER_KV * HEAD_DIM;

  // Use 32 threads per head to match work items (1 token * 8 heads = 8 groups =
  // 256/32)
  constexpr int THREADS_PER_TOKEN = (NUM_QO_HEADS_PER_KV <= 8) ? 32 : 16;
  constexpr int VAL_PER_THREAD = HEAD_DIM / THREADS_PER_TOKEN;
  constexpr int num_groups = NUM_THREADS / THREADS_PER_TOKEN;

  int thread_in_group = threadIdx.x % THREADS_PER_TOKEN;
  int group_id = threadIdx.x / THREADS_PER_TOKEN;

  // Optional sink correction (GPT-OSS): out *= 1 / (1 + exp(sink -
  // LSE_natural)) Layout: sinks[num_q_heads] in bf16, indexed by
  // kv_head_idx*NUM_QO_HEADS_PER_KV + head.
  using __sink_bf16 = __hip_bfloat16;
  __sink_bf16 const *d_sinks = reinterpret_cast<__sink_bf16 const *>(sinks_ptr);

#pragma unroll
  for (int tok = group_id; tok < num_tokens * NUM_QO_HEADS_PER_KV;
       tok += num_groups) {
    int token_idx = tok / NUM_QO_HEADS_PER_KV;
    int head_idx = tok % NUM_QO_HEADS_PER_KV;

    // Load this head's sink once (independent of dim).
    float sink_val_log2 = 0.0f;
    if (sinks_ptr != nullptr) {
      sink_val_log2 =
          static_cast<float>(
              d_sinks[kv_head_idx * NUM_QO_HEADS_PER_KV + head_idx]) *
          1.44269504088896340736f; // convert sink from ln to log2
    }

    // Base output offset for this token+head (dims start here)
    int out_offset_base = (first_token_pos + token_idx) * OUT_TOKEN_STRIDE +
                          kv_head_idx * NUM_QO_HEADS_PER_KV * HEAD_DIM +
                          head_idx * HEAD_DIM +
                          thread_in_group * VAL_PER_THREAD;

    // Compute all VAL_PER_THREAD dimensions
    float out_vals[VAL_PER_THREAD];
#pragma unroll
    for (int i = 0; i < VAL_PER_THREAD; ++i) {
      float m_global = -inf;
      float d_global = 1.f;
      float o_global = 0.f;
#pragma unroll
      for (int kv_idx = 0; kv_idx < num_chunks; ++kv_idx) {
        float m_prev = m_global, d_prev = d_global;

        int lse_linear = head_idx + kv_idx * NUM_QO_HEADS_PER_KV +
                         (first_token_pos + token_idx) * LSE_TOKEN_STRIDE +
                         lse_kv_offset;
        int lse_offset = lse_linear;
        int o_offset =
            lse_linear * HEAD_DIM + thread_in_group * VAL_PER_THREAD + i;

        // CK FMHA stores LSE in natural log scale; convert to log2 for ptx_exp2
        float other_m = lse_ptr[lse_offset] * 1.44269504088896340736f,
              other_d = 1;
        m_global = max(m_prev, other_m);
        d_global = d_prev * ptx_exp2(m_prev - m_global) +
                   other_d * ptx_exp2(other_m - m_global);
        float other_o = o_ptr[o_offset];
        o_global = o_global * ptx_exp2(m_prev - m_global) +
                   other_o * ptx_exp2(other_m - m_global);
      }
      float out_f = __fdividef(o_global, d_global);
      if (sinks_ptr != nullptr) {
        float lse_log2 = m_global + ptx_log2(d_global);
        float diff = sink_val_log2 - lse_log2;
        float correction = __fdividef(1.0f, 1.0f + ptx_exp2(diff));
        out_f *= correction;
      }
      out_vals[i] = out_f;
    }

    // Write output: either write-through (st_wt) or regular global store
    if constexpr (WRITE_THROUGH) {
      // Pack pairs of bf16 into uint32 and write-through to HBM
      static_assert(VAL_PER_THREAD % 2 == 0 || VAL_PER_THREAD == 1,
                    "WRITE_THROUGH requires even VAL_PER_THREAD or 1");
#pragma unroll
      for (int i = 0; i < VAL_PER_THREAD; i += 2) {
        __hip_bfloat16 v0 = (__hip_bfloat16)out_vals[i];
        __hip_bfloat16 v1 = (__hip_bfloat16)out_vals[i + 1];
        uint32_t packed;
        uint16_t lo, hi;
        memcpy(&lo, &v0, 2);
        memcpy(&hi, &v1, 2);
        packed = lo | ((uint32_t)hi << 16);
        st_wt_u32((void *)&output_ptr[out_offset_base + i], packed);
      }
    } else {
#pragma unroll
      for (int i = 0; i < VAL_PER_THREAD; ++i) {
        output_ptr[out_offset_base + i] = (T)out_vals[i];
      }
    }
  }
}

} // namespace kernel