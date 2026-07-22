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
#include "multitoken_paged_attention_ck_split_kv_mi300.cuh"

namespace kernel {
template <typename T,
          int NUM_QO_HEADS,
          int NUM_KV_HEADS,
          int NUM_QO_GROUPS,
          int KV_CACHE_STRIDE,
          int QKV_STRIDE,
          int O_STRIDE,
          int HEAD_DIM,
          int SEQ_LEN_PER_BLOCK,
          int MAX_SEQ_LEN,
          int PAGE_SIZE,
          int MAX_TOKENS,
          int NUM_KV_CHUNKS>
__device__ __forceinline__ void multitoken_paged_attention_split_kv_task_impl(
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
    float k_eps,
    void *lse_ptr,
    int kv_chunk_idx) {
#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  unsigned long long _t0 = __builtin_amdgcn_s_memrealtime();
#endif
  // AMD MI300: Use CK Tile MFMA-based split-KV attention
  multitoken_paged_attention_ck_split_kv<T,
                                         NUM_QO_HEADS,
                                         NUM_KV_HEADS,
                                         NUM_QO_GROUPS,
                                         KV_CACHE_STRIDE,
                                         QKV_STRIDE,
                                         O_STRIDE,
                                         HEAD_DIM,
                                         SEQ_LEN_PER_BLOCK,
                                         MAX_SEQ_LEN,
                                         PAGE_SIZE,
                                         MAX_TOKENS,
                                         NUM_KV_CHUNKS>(
      qkv_ptr,
      paged_k_cache_ptr,
      paged_v_cache_ptr,
      output_ptr,
      qo_indptr_buffer_ptr,
      paged_kv_indptr_buffer_ptr,
      paged_kv_indices_buffer_ptr,
      paged_kv_last_page_len_buffer_ptr,
      request_id,
      qk_norm,
      rope,
      q_norm_weight_ptr,
      k_norm_weight_ptr,
      cos_ptr,
      sin_ptr,
      q_eps,
      k_eps,
      lse_ptr,
      kv_chunk_idx);
#ifdef MPK_ENABLE_DEVICE_TASK_TIMING
  __syncthreads();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    unsigned long long _dur = (__builtin_amdgcn_s_memrealtime() - _t0) * 10;
    printf("[ATTN_SPLIT] dur_us=%.1f\n", (double)_dur / 1000.0);
  }
#endif
}
} // namespace kernel
