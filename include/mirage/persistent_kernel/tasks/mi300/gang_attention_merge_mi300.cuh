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
// Requires merge_splitkv.cuh to be included before this file

namespace kernel {

// Gang merge for split-KV CK FMHA attention.
// 8 tasks (1 per XCD), tile_idx -> (request_id, kv_head).
// Calls merge_splitkv_ck_fmha to reduce float32 partial results across
// NUM_KV_CHUNKS into final bf16 output.
template <typename T,
          int NUM_QO_HEADS_PER_KV,
          int NUM_KV_HEADS,
          int HEAD_DIM,
          int MAX_TOKENS,
          int NUM_KV_CHUNKS,
          int KV_CHUNK_SIZE,
          int PAGE_SIZE,
          int MERGE_OUTPUT_HEAD_OFFSET>
__device__ __noinline__ void gang_attention_merge_kernel(
    void const *lse_ptr,
    void const *o_acc_ptr,
    int const *qo_indptr,
    int const *kv_indptr,
    int const *kv_last_page_len,
    void *output_ptr,
    int num_kv_heads,
    int total_work_items,
    int tile_idx)
{
  if (tile_idx >= total_work_items) return;

  int kv_head = tile_idx % num_kv_heads;
  int16_t request_id = static_cast<int16_t>(tile_idx / num_kv_heads);

  merge_splitkv_ck_fmha<T,
      NUM_QO_HEADS_PER_KV,
      NUM_KV_HEADS,
      HEAD_DIM,
      NUM_KV_CHUNKS,
      KV_CHUNK_SIZE,
      PAGE_SIZE>(
      reinterpret_cast<float const *>(lse_ptr),
      reinterpret_cast<float const *>(o_acc_ptr),
      qo_indptr,
      kv_indptr,
      kv_last_page_len,
      request_id,
      reinterpret_cast<T *>(output_ptr),
      kv_head);
}

} // namespace kernel
