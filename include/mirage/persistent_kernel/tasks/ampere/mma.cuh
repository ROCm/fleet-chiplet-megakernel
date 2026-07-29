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

// Helper to convert bf16 (stored in uint16_t) to float
__device__ __forceinline__ float bf16_to_float(uint16_t v) {
  // BF16 is upper 16 bits of float32
  uint32_t tmp = static_cast<uint32_t>(v) << 16;
  return *reinterpret_cast<float *>(&tmp);
}

__device__ static __forceinline__ void
    mma_m16n16k16_bf16bf16bf32(float *C, uint32_t *A, uint32_t *B, float *D) {
#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    (defined(__HIP_PLATFORM_AMD__) || defined(MIRAGE_AMD_MI300))
  // AMD scalar implementation of CUDA m16n8k16 MMA
  // This function computes two m16n8k16 matrix multiplies:
  //   D[0:4] += A * B[0:2] (first n8 block)
  //   D[4:8] += A * B[2:4] (second n8 block)
  //
  // CUDA MMA layout (m16n8k16 with bf16):
  // - A: Each thread t has 4 registers, containing elements for row (t%16)
  //      A[i] contains k = (t/16)*8 + i*2 + {0,1}
  // - B: Each thread t has 2 registers per n8 block, containing elements for
  // col (t%8)
  //      B[i] contains k = (t/8 % 2)*8 + i*2 + {0,1}
  // - D: Each thread t has 4 registers per n8 block
  //      D[i] = output at row (t/4)*2 + (i/2), col (t%4)*2 + (i%2) + (t/16)*8

  int lane = threadIdx.x & 31;

// Initialize D with C
#pragma unroll
  for (int i = 0; i < 8; i++) {
    D[i] = C[i];
  }

  // Decode this thread's output positions
  // For D[0-3] (first m16n8 block):
  //   row = (lane / 4) * 2 + {0,0,1,1} for D[0,1,2,3]
  //   col = (lane % 4) * 2 + {0,1,0,1} for D[0,1,2,3], adjusted by (lane/16)*8
  //   -> not quite right
  //
  // Actually, CUDA's m16n8k16 output layout is:
  //   Thread t gets D[0-3] at:
  //     D[0]: row = (t%4)*2,   col = t/4
  //     D[1]: row = (t%4)*2+1, col = t/4
  //     D[2]: row = (t%4)*2+8, col = t/4
  //     D[3]: row = (t%4)*2+9, col = t/4
  //   Where t/4 wraps at 8 (so cols 0-7)

  int out_col = lane / 4; // 0-7 for first m16n8
  int row0 = (lane % 4) * 2;
  int row1 = row0 + 1;
  int row2 = row0 + 8;
  int row3 = row1 + 8;

  // For each output element, compute the dot product by gathering A and B
  // values A[row, k] from appropriate thread, B[k, col] from appropriate thread

  // Helper: get A element at (row, k)
  // Thread that has A[row, k]: lane = (k/8)*16 + (row%16)
  // Register index: (k%8)/2
  // Element in register: k%2

  // Helper: get B element at (k, col)
  // Thread that has B[k, col]: lane = (k/8)*8 + col (for B[0:2]) or lane =
  // (k/8)*8 + col (for B[2:4]) Wait, B only has cols 0-7 in first 2 registers

  // Actually simpler: since ldmatrix loaded data, A and B already contain the
  // right values for this thread. We just need to do the reduction across
  // threads.

  // CUDA MMA computes: for each output (row, col), sum over k: A[row,k] *
  // B[k,col] The challenge is that different threads hold different (row, k)
  // pairs for A and different (k, col) pairs for B.

  // Approach: Use shared memory to collect all A and B values, then compute
  // scalar matmul But shared memory is limited and would require
  // synchronization.

  // Alternative: Pure shuffle-based approach
  // For output D[0] at (row0, out_col), we need sum_k A[row0,k] * B[k,out_col]

  // First m16n8 block: D[0-3]
  float sums[4] = {0, 0, 0, 0};
  int rows[4] = {row0, row1, row2, row3};

#pragma unroll
  for (int d_idx = 0; d_idx < 4; d_idx++) {
    int row = rows[d_idx];
    float sum = 0.0f;

#pragma unroll
    for (int k = 0; k < 16; k++) {
      // Get A[row, k]
      int a_lane = (k / 8) * 16 + (row % 16);
      int a_reg = (k % 8) / 2;
      int a_elem = k % 2;
      uint32_t a_packed = __shfl(A[a_reg], a_lane, 32);
      uint16_t a_bf16 = (a_elem == 0) ? (a_packed & 0xFFFF) : (a_packed >> 16);

      // Get B[k, out_col] from first m16n8 block (B[0:2])
      int b_lane = (k / 8) * 8 + out_col;
      int b_reg = (k % 8) / 2;
      int b_elem = k % 2;
      uint32_t b_packed = __shfl(B[b_reg], b_lane, 32);
      uint16_t b_bf16 = (b_elem == 0) ? (b_packed & 0xFFFF) : (b_packed >> 16);

      sum += bf16_to_float(a_bf16) * bf16_to_float(b_bf16);
    }
    D[d_idx] += sum;
  }

// Second m16n8 block: D[4-7] (cols 8-15)
// B[2:4] contains data for cols 8-15, but thread layout is same (col = lane/4)
// So B[2+reg] for the second block
#pragma unroll
  for (int d_idx = 0; d_idx < 4; d_idx++) {
    int row = rows[d_idx];
    float sum = 0.0f;

#pragma unroll
    for (int k = 0; k < 16; k++) {
      // Get A[row, k] - same as before
      int a_lane = (k / 8) * 16 + (row % 16);
      int a_reg = (k % 8) / 2;
      int a_elem = k % 2;
      uint32_t a_packed = __shfl(A[a_reg], a_lane, 32);
      uint16_t a_bf16 = (a_elem == 0) ? (a_packed & 0xFFFF) : (a_packed >> 16);

      // Get B[k, out_col+8] from second m16n8 block (B[2:4])
      int b_lane = (k / 8) * 8 + out_col;
      int b_reg = 2 + (k % 8) / 2; // Offset by 2 for second block
      int b_elem = k % 2;
      uint32_t b_packed = __shfl(B[b_reg], b_lane, 32);
      uint16_t b_bf16 = (b_elem == 0) ? (b_packed & 0xFFFF) : (b_packed >> 16);

      sum += bf16_to_float(a_bf16) * bf16_to_float(b_bf16);
    }
    D[4 + d_idx] += sum;
  }
#else
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
               : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
               : "r"(A[0]),
                 "r"(A[1]),
                 "r"(A[2]),
                 "r"(A[3]),
                 "r"(B[0]),
                 "r"(B[1]),
                 "f"(C[0]),
                 "f"(C[1]),
                 "f"(C[2]),
                 "f"(C[3]));

  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
               : "=f"(D[4]), "=f"(D[5]), "=f"(D[6]), "=f"(D[7])
               : "r"(A[0]),
                 "r"(A[1]),
                 "r"(A[2]),
                 "r"(A[3]),
                 "r"(B[2]),
                 "r"(B[3]),
                 "f"(C[4]),
                 "f"(C[5]),
                 "f"(C[6]),
                 "f"(C[7]));
#endif
}
} // namespace kernel
