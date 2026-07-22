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

// MoE TopK Softmax routing kernel for MI300/MI350.
// Fused softmax + top-k selection + routing index generation.
// Adapted from topk_softmax_sm100.cuh for HIP/AMD.

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

namespace kernel {

static constexpr int MI300_WARP_SIZE = 64; // AMD wavefront size

// Fused TopK softmax kernel for AMD MI300/MI350.
// Uses wave-level reductions with __shfl_xor for max/sum across experts.
//
// Block size: 256 threads (4 wavefronts of 64).
// Each wavefront processes multiple rows in parallel.
//
// Template parameters:
//   T           - data type (bf16)
//   VPT         - values per thread (power of 2)
//   NUM_EXPERTS - total number of experts (power of 2)
//   WARPS_PER_CTA - number of wavefronts per CTA (typically 4)
//   BYTES_PER_LDG - bytes per vectorized load (8 or 16)
template <typename T,
          int VPT,
          int NUM_EXPERTS,
          int WARPS_PER_CTA,
          int BYTES_PER_LDG>
__device__ __forceinline__ void topk_softmax_mi300_task_impl(
    void *__restrict__ input_ptr,              // [num_rows, NUM_EXPERTS]
    void *__restrict__ output_ptr,             // [num_rows, k] (float weights)
    int const num_rows,
    int const k,
    void *__restrict__ routing_indices_ptr,    // [NUM_EXPERTS, num_rows] int32
    void *__restrict__ active_expert_ids_ptr,  // [NUM_EXPERTS + 1] int32
    int const start_expert,
    int const end_expert,
    bool const renormalize) {
  T *input = static_cast<T *>(input_ptr);
  float *output = static_cast<float *>(output_ptr);
  int *routing_indices = static_cast<int *>(routing_indices_ptr);
  int *active_expert_ids = static_cast<int *>(active_expert_ids_ptr);

  // Initialize routing indices to 0 and active expert marks to -1
  for (int expert = start_expert + threadIdx.x; expert < end_expert;
       expert += blockDim.x) {
    if (routing_indices != nullptr) {
      for (int row = 0; row < num_rows; ++row) {
        routing_indices[expert * num_rows + row] = 0;
      }
    }
    if (active_expert_ids != nullptr) {
      active_expert_ids[expert - start_expert] = -1;
    }
  }
  if (threadIdx.x == NUM_EXPERTS && active_expert_ids != nullptr) {
    active_expert_ids[NUM_EXPERTS] = 0;
  }
  __syncthreads();

  // Compile-time constants
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
  static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
  static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
  static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

  static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
  static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS),
                "NUM_EXPERTS must be power of 2");
  static_assert(MI300_WARP_SIZE % THREADS_PER_ROW == 0,
                "THREADS_PER_ROW must divide warp size");

  static constexpr int ELTS_PER_WARP = MI300_WARP_SIZE * VPT;
  static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;

  int const warp_idx = threadIdx.x / MI300_WARP_SIZE;
  int const lane_idx = threadIdx.x % MI300_WARP_SIZE;
  int const warp_base_row = warp_idx * ROWS_PER_WARP;
  int const thread_row_in_warp = lane_idx / THREADS_PER_ROW;
  int const thread_row = warp_base_row + thread_row_in_warp;
  int const thread_group_idx = lane_idx % THREADS_PER_ROW;

  if (thread_row < num_rows) {
    // Load row data
    T *thread_row_ptr = input + thread_row * ELTS_PER_ROW;
    int const first_elt = thread_group_idx * ELTS_PER_LDG;
    T *thread_read_ptr = thread_row_ptr + first_elt;

    float row_chunk[VPT];
    // Vectorized load
    for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) {
      int base = ldg * ELTS_PER_LDG;
      int src_offset = ldg * THREADS_PER_ROW * ELTS_PER_LDG;
      for (int e = 0; e < ELTS_PER_LDG; ++e) {
        row_chunk[base + e] = static_cast<float>(thread_read_ptr[src_offset + e]);
      }
    }

    // Reset input buffer to 0 (for split-k gate linear compatibility)
    for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) {
      int src_offset = ldg * THREADS_PER_ROW * ELTS_PER_LDG;
      for (int e = 0; e < ELTS_PER_LDG; ++e) {
        thread_read_ptr[src_offset + e] = static_cast<T>(0);
      }
    }

    // Max reduction within subgroup
    float thread_max = row_chunk[0];
    for (int ii = 1; ii < VPT; ++ii) {
      thread_max = fmaxf(thread_max, row_chunk[ii]);
    }
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      float other = __shfl_xor(thread_max, mask, THREADS_PER_ROW);
      thread_max = fmaxf(thread_max, other);
    }

    // Softmax
    float row_sum = 0.f;
    for (int ii = 0; ii < VPT; ++ii) {
      row_chunk[ii] = expf(row_chunk[ii] - thread_max);
      row_sum += row_chunk[ii];
    }
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      row_sum += __shfl_xor(row_sum, mask, THREADS_PER_ROW);
    }
    float const inv_sum = 1.f / row_sum;
    for (int ii = 0; ii < VPT; ++ii) {
      row_chunk[ii] *= inv_sum;
    }

    // Fused Top-K selection
    int const start_col = first_elt;
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;
    float row_sum_for_renorm = 0.f;

    for (int k_idx = 0; k_idx < k; ++k_idx) {
      // Find local max
      float max_val = row_chunk[0];
      int expert = start_col;
      for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD;
           ++ldg, col += COLS_PER_GROUP_LDG) {
        for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
          float val = row_chunk[ldg * ELTS_PER_LDG + ii];
          if (val > max_val) {
            max_val = val;
            expert = col + ii;
          }
        }
      }

      // Argmax reduce across subgroup
      for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        float other_max = __shfl_xor(max_val, mask, THREADS_PER_ROW);
        int other_expert = __shfl_xor(expert, mask, THREADS_PER_ROW);
        if (other_max > max_val ||
            (other_max == max_val && other_expert < expert)) {
          max_val = other_max;
          expert = other_expert;
        }
      }

      // Write top-k result
      if (thread_group_idx == 0) {
        bool const node_uses = (expert >= start_expert && expert < end_expert);
        int const out_idx = k * thread_row + k_idx;
        output[out_idx] = max_val;
        row_sum_for_renorm += max_val;

        if (node_uses && routing_indices != nullptr) {
          int const local_expert = expert - start_expert;
          routing_indices[local_expert * num_rows + thread_row] = k_idx + 1;
          if (active_expert_ids != nullptr) {
            active_expert_ids[local_expert] = local_expert;
          }
        }
      }

      // Blank out winner for next iteration
      if (k_idx + 1 < k) {
        int const ldg_group = expert / COLS_PER_GROUP_LDG;
        int const thread_to_clear = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;
        if (thread_group_idx == thread_to_clear) {
          int const offset = expert % ELTS_PER_LDG;
          row_chunk[ldg_group * ELTS_PER_LDG + offset] = -10000.f;
        }
      }
    }

    // Optional renormalization
    if (renormalize && thread_group_idx == 0) {
      float inv = 1.f / row_sum_for_renorm;
      for (int k_idx = 0; k_idx < k; ++k_idx) {
        int const out_idx = k * thread_row + k_idx;
        output[out_idx] *= inv;
      }
    }
  }
  __syncthreads();

  __syncthreads();

  // Compact active expert marks into dense list
  if (active_expert_ids != nullptr) {
    for (int expert = start_expert + threadIdx.x; expert < end_expert;
         expert += blockDim.x) {
      int const local_expert = expert - start_expert;
      int const mark = active_expert_ids[local_expert];
      if (mark >= 0) {
        int const pos = atomicAdd(active_expert_ids + NUM_EXPERTS, 1);
        active_expert_ids[pos] = expert;
      }
    }
  }
}

} // namespace kernel
