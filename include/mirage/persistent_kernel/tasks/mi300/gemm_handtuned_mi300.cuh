/* Hand-tuned MFMA GEMM for MI300X (gfx942)
 * Minimal code footprint to fit in 32KB L1 I-cache.
 * Same tile config as CK version: MPerBlock=16, NPerBlock=64, 4 wavefronts.
 * Uses __builtin_amdgcn_mfma_f32_16x16x16bf16_1k directly.
 *
 * NO LDS version: each thread loads its MFMA inputs directly from global
 * memory. Eliminates barriers and prevents compiler from mis-optimizing
 * the K loop.
 */
#pragma once

namespace kernel {

typedef short v4s __attribute__((ext_vector_type(4)));
typedef float v4f __attribute__((ext_vector_type(4)));

// ============================================================================
// Hand-tuned GEMM: direct global→register→MFMA loop, no LDS staging.
// M=BATCH_SIZE (1-16), N=output_size (runtime, ≤64), K=REDUCTION_SIZE.
// 256 threads = 4 wavefronts. Each wavefront computes a 16×16 MFMA tile.
//
// MFMA lane mapping (v_mfma_f32_16x16x16_bf16):
//   Lane l (0-63):
//     A input: A[l%16, (l/16)*4 .. (l/16)*4+3]  (4 consecutive K values)
//     B input: B[l%16, (l/16)*4 .. (l/16)*4+3]  (4 consecutive K values)
//     C output: C[l%16, (l/16)*4 .. (l/16)*4+3] (4 consecutive N values)
//   Result: C[m,n] = sum_k A[m,k] * B[n,k]
//
// Per wavefront w: B rows are B[w*16 + 0..15, :] (16 output columns per warp)
// All wavefronts share the same A data (duplicated loads).
// ============================================================================

template <int BATCH_SIZE, int REDUCTION_SIZE>
__device__ __forceinline__ void
    gemm_handtuned(void const *__restrict__ input_ptr,
                   void const *__restrict__ weight_ptr,
                   void const *__restrict__ residual_ptr,
                   void *__restrict__ output_ptr,
                   int num_active_tokens,
                   bool residual_add,
                   int output_size,
                   int o_stride) {
  unsigned short const *A =
      (unsigned short const *)input_ptr; // [M, K] bf16 row-major
  unsigned short const *B =
      (unsigned short const *)weight_ptr; // [N, K] bf16 row-major
  unsigned short const *R = (unsigned short const *)residual_ptr;
  unsigned short *C = (unsigned short *)output_ptr;

  int warp_id = threadIdx.x >> 6; // 0-3
  int lane_id = threadIdx.x & 63; // 0-63

  // MFMA lane decomposition
  int m_idx = lane_id & 15;         // row in 16×16 tile: 0-15
  int k_group = lane_id >> 4;       // K group: 0-3, each provides 4 K values
  int k_lane_offset = k_group << 2; // K offset within tile: 0, 4, 8, 12

  // B row for this wavefront: warp*16 + lane%16
  int b_row = (warp_id << 4) + m_idx;

  // Precompute base pointers for A and B row access
  // A[m_idx, :] row start (only valid if m_idx < BATCH_SIZE)
  unsigned short const *A_row = A + m_idx * REDUCTION_SIZE;
  // B[b_row, :] row start
  unsigned short const *B_row = B + b_row * REDUCTION_SIZE;

  int LoopN = (output_size + 63) >> 6; // ceil(output_size / 64)
  constexpr int KTILE = 16;
  constexpr int NumLoopK = REDUCTION_SIZE / KTILE;

  for (int nn = 0; nn < LoopN; nn++) {
    int n_offset = nn * 64;
    int global_b_row = n_offset + b_row;

    // Recompute B_row for this N-tile
    unsigned short const *B_row_n = B + global_b_row * REDUCTION_SIZE;
    bool b_valid = (global_b_row < output_size);

    // Initialize accumulator
    v4f acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kk = 0; kk < NumLoopK; kk++) {
      int k_offset = kk * KTILE + k_lane_offset;

      // Load A: 4 consecutive bf16 from A[m_idx, k_offset .. k_offset+3]
      v4s a_reg;
      if (m_idx < BATCH_SIZE) {
        // Load 8 bytes (4 bf16) as two 32-bit words
        unsigned const *a_src = (unsigned const *)(A_row + k_offset);
        unsigned a_lo = a_src[0];
        unsigned a_hi = a_src[1];
        a_reg[0] = (short)(a_lo & 0xFFFF);
        a_reg[1] = (short)(a_lo >> 16);
        a_reg[2] = (short)(a_hi & 0xFFFF);
        a_reg[3] = (short)(a_hi >> 16);
      } else {
        a_reg[0] = a_reg[1] = a_reg[2] = a_reg[3] = 0;
      }

      // Load B: 4 consecutive bf16 from B[global_b_row, k_offset .. k_offset+3]
      v4s b_reg;
      if (b_valid) {
        unsigned const *b_src = (unsigned const *)(B_row_n + k_offset);
        unsigned b_lo = b_src[0];
        unsigned b_hi = b_src[1];
        b_reg[0] = (short)(b_lo & 0xFFFF);
        b_reg[1] = (short)(b_lo >> 16);
        b_reg[2] = (short)(b_hi & 0xFFFF);
        b_reg[3] = (short)(b_hi >> 16);
      } else {
        b_reg[0] = b_reg[1] = b_reg[2] = b_reg[3] = 0;
      }

      // MFMA: acc += A_tile × B_tile^T
      acc =
          __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_reg, b_reg, acc, 0, 0, 0);
    }

    // === Store results ===
    // Lane l output: C[l%16, warp*16 + (l/16)*4 .. +3] = acc[0..3]
    {
      int out_row = m_idx;
      int out_col_base = n_offset + (warp_id << 4) + k_lane_offset;

      if (out_row < BATCH_SIZE && out_col_base + 3 < output_size) {
        int base_idx = out_row * o_stride + out_col_base;
        unsigned out01, out23;
        if (residual_add && R != nullptr) {
          unsigned res01 = *(unsigned const *)(R + base_idx);
          unsigned res23 = *(unsigned const *)(R + base_idx + 2);
          float r0, r1, r2, r3;
          unsigned t;
          t = (unsigned)(res01 & 0xFFFF) << 16;
          __builtin_memcpy(&r0, &t, 4);
          t = (unsigned)(res01 >> 16) << 16;
          __builtin_memcpy(&r1, &t, 4);
          t = (unsigned)(res23 & 0xFFFF) << 16;
          __builtin_memcpy(&r2, &t, 4);
          t = (unsigned)(res23 >> 16) << 16;
          __builtin_memcpy(&r3, &t, 4);
          float v0 = acc[0] + r0, v1 = acc[1] + r1, v2 = acc[2] + r2,
                v3 = acc[3] + r3;
          unsigned short b0, b1, b2, b3;
          t = 0;
          __builtin_memcpy(&t, &v0, 4);
          b0 = t >> 16;
          t = 0;
          __builtin_memcpy(&t, &v1, 4);
          b1 = t >> 16;
          t = 0;
          __builtin_memcpy(&t, &v2, 4);
          b2 = t >> 16;
          t = 0;
          __builtin_memcpy(&t, &v3, 4);
          b3 = t >> 16;
          out01 = (unsigned)b0 | ((unsigned)b1 << 16);
          out23 = (unsigned)b2 | ((unsigned)b3 << 16);
        } else {
          unsigned short b0, b1, b2, b3;
          unsigned t;
          float tmp;
          tmp = acc[0];
          __builtin_memcpy(&t, &tmp, 4);
          b0 = t >> 16;
          tmp = acc[1];
          __builtin_memcpy(&t, &tmp, 4);
          b1 = t >> 16;
          tmp = acc[2];
          __builtin_memcpy(&t, &tmp, 4);
          b2 = t >> 16;
          tmp = acc[3];
          __builtin_memcpy(&t, &tmp, 4);
          b3 = t >> 16;
          out01 = (unsigned)b0 | ((unsigned)b1 << 16);
          out23 = (unsigned)b2 | ((unsigned)b3 << 16);
        }
        *(unsigned *)(C + base_idx) = out01;
        *(unsigned *)(C + base_idx + 2) = out23;
      } else if (out_row < BATCH_SIZE) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
          int col = out_col_base + i;
          if (col < output_size) {
            float val = acc[i];
            if (residual_add && R != nullptr) {
              unsigned t = (unsigned)R[out_row * o_stride + col] << 16;
              float rv;
              __builtin_memcpy(&rv, &t, 4);
              val += rv;
            }
            unsigned t;
            __builtin_memcpy(&t, &val, 4);
            C[out_row * o_stride + col] = (unsigned short)(t >> 16);
          }
        }
      }
    }
  }
}

} // namespace kernel
