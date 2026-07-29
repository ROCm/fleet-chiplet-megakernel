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
// Requires kv_cache_update_mi300.cuh and
// paged_attention_ck_fmha_split_kv_mi300.cuh to be included before this file
// (via task_header.cuh)

namespace kernel {

// Gang CK FMHA attention: 8 tasks (1 per XCD), broadcast to workers.
// Each worker uses tile_idx to determine (request_id, kv_head).
// Fuses KV cache update (Phase A) + CK FMHA attention (Phase B) into one gang
// task. With NUM_KV_CHUNKS=1, CK FMHA writes bf16 directly — no merge needed.
//
// tile_idx decomposition:
//   kv_head    = tile_idx % num_kv_heads
//   request_id = tile_idx / num_kv_heads
//
// The base pointers are un-partitioned (no bid.y offset).
// This wrapper applies per-kv-head pointer offsets before calling
// kv_cache_update_impl and paged_attention_ck_fmha_split_kv_impl.
template <typename T,
          int NUM_QO_HEADS_PER_KV,
          int NUM_KV_HEADS_ACTUAL,
          int KV_CACHE_STRIDE,
          int QKV_STRIDE,
          int HEAD_DIM,
          int MAX_SEQ_LEN,
          int PAGE_SIZE,
          int MAX_TOKENS,
          int NUM_KV_CHUNKS,
          int QKV_HEAD_OFFSET,
          int KV_CACHE_HEAD_OFFSET,
          int Q_WORKSPACE_STRIDE>
__device__ __noinline__ void
    gang_attention_split_kv_kernel(void const *qkv_ptr,
                                   void *paged_k_cache_ptr,
                                   void *paged_v_cache_ptr,
                                   void *output_ptr,
                                   int const *qo_indptr,
                                   int const *kv_indptr,
                                   int const *kv_indices,
                                   int const *kv_last_page_len,
                                   bool qk_norm,
                                   bool rope,
                                   void const *q_norm_ptr,
                                   void const *k_norm_ptr,
                                   void const *cos_ptr,
                                   void const *sin_ptr,
                                   float q_eps,
                                   float k_eps,
                                   void *lse_ptr,
                                   void *q_workspace_ptr,
                                   int num_kv_heads,
                                   int total_work_items,
                                   int tile_idx,
                                   float scale_s) {
  if (tile_idx >= total_work_items) {
    return;
  }

  int kv_head = tile_idx % num_kv_heads;
  int16_t request_id = static_cast<int16_t>(tile_idx / num_kv_heads);

  using bf16 = ck_tile::bf16_t;

  // Apply per-kv-head pointer offsets
  T const *offset_qkv = static_cast<T const *>(qkv_ptr) +
                        static_cast<size_t>(kv_head) * QKV_HEAD_OFFSET;
  T *offset_k_cache = static_cast<T *>(paged_k_cache_ptr) +
                      static_cast<size_t>(kv_head) * KV_CACHE_HEAD_OFFSET;
  T *offset_v_cache = static_cast<T *>(paged_v_cache_ptr) +
                      static_cast<size_t>(kv_head) * KV_CACHE_HEAD_OFFSET;
  // q_workspace: offset by kv_head to match non-gang bid.y partitioning
  // Non-gang kv_cache_update partitions q_workspace by (-1, 1, -1) with bid.y =
  // kv_head Partition offset = kv_head * (Q_WORKSPACE_STRIDE /
  // NUM_KV_HEADS_ACTUAL)
  //                  = kv_head * NUM_QO_HEADS_PER_KV * HEAD_DIM
  T *offset_q_workspace =
      static_cast<T *>(q_workspace_ptr) +
      static_cast<size_t>(kv_head) * NUM_QO_HEADS_PER_KV * HEAD_DIM;

  // Phase A: KV cache update + Q preprocessing
  // kv_cache_update_impl operates on 1 KV head at a time (NUM_KV_HEADS=1)
  kv_cache_update_impl<T,
                       NUM_QO_HEADS_PER_KV,
                       1,                   // NUM_KV_HEADS (process 1 head)
                       NUM_KV_HEADS_ACTUAL, // NUM_QO_GROUPS
                       KV_CACHE_STRIDE,
                       QKV_STRIDE,
                       HEAD_DIM,
                       MAX_SEQ_LEN,
                       PAGE_SIZE,
                       MAX_TOKENS,
                       Q_WORKSPACE_STRIDE>(
      offset_qkv,
      offset_k_cache,
      offset_v_cache,
      offset_q_workspace, // q_workspace offset by kv_head
      qo_indptr,
      kv_indptr,
      kv_indices,
      kv_last_page_len,
      request_id,
      qk_norm,
      rope,
      q_norm_ptr,
      k_norm_ptr,
      cos_ptr,
      sin_ptr,
      q_eps,
      k_eps);

  __syncthreads();

  // Phase B: CK FMHA attention (reads Q from workspace, K/V from cache)
  // With NUM_KV_CHUNKS=1, writes bf16 directly to output (no merge needed)
  paged_attention_ck_fmha_split_kv_impl<T,
                                        NUM_QO_HEADS_PER_KV,
                                        HEAD_DIM,
                                        PAGE_SIZE,
                                        MAX_SEQ_LEN,
                                        NUM_KV_CHUNKS,
                                        Q_WORKSPACE_STRIDE,
                                        KV_CACHE_STRIDE,
                                        NUM_KV_HEADS_ACTUAL>(
      q_workspace_ptr, // q_workspace (full)
      offset_k_cache,  // k_cache offset by kv_head
      offset_v_cache,  // v_cache offset by kv_head
      output_ptr,      // output (full)
      lse_ptr,         // lse (full)
      qo_indptr,
      kv_indptr,
      kv_indices,
      kv_last_page_len,
      request_id,
      kv_head, // kv_head_idx
      0,       // kv_chunk_idx = 0 (single chunk)
      scale_s);
}

} // namespace kernel
