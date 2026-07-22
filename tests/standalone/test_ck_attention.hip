/* Copyright 2025 CMU
 *
 * Standalone test for CK Tile-based attention on AMD MI300
 * Tests Q*K^T and P*V using CK Tile's warp GEMM primitives
 */

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

// CK Tile includes
#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t err = call;                                                     \
    if (err != hipSuccess) {                                                   \
      std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << ": "      \
                << hipGetErrorString(err) << std::endl;                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// ============================================================================
// Configuration
// ============================================================================
constexpr int Q_ROWS = 8;           // Number of query rows (batch * heads)
constexpr int KV_LEN = 32;          // KV sequence length tile
constexpr int HEAD_DIM = 128;       // Head dimension

constexpr int MFMA_M = 16;
constexpr int MFMA_N = 16;
constexpr int MFMA_K = 16;

constexpr int NUM_THREADS = 256;
constexpr int NUM_WARPS = NUM_THREADS / 64;

constexpr int NUM_WARMUP = 10;
constexpr int NUM_ITERS = 100;

using bf16 = ck_tile::bf16_t;

// ============================================================================
// CK Tile Warp GEMM types
// ============================================================================
using WarpGemmQK = ck_tile::WarpGemmMfmaBf16Bf16F32M16N16K16;
using WarpGemmPV = ck_tile::WarpGemmMfmaBf16Bf16F32M16N16K16;

// ============================================================================
// Reference implementation
// ============================================================================
void attention_reference(
    const bf16* Q,      // [Q_ROWS, HEAD_DIM]
    const bf16* K,      // [KV_LEN, HEAD_DIM]
    const bf16* V,      // [KV_LEN, HEAD_DIM]
    float* O,           // [Q_ROWS, HEAD_DIM]
    float scale) {

  // Compute scores = Q @ K^T
  std::vector<float> scores(Q_ROWS * KV_LEN);
  for (int q = 0; q < Q_ROWS; q++) {
    for (int k = 0; k < KV_LEN; k++) {
      float sum = 0.0f;
      for (int d = 0; d < HEAD_DIM; d++) {
        float qv = ck_tile::type_convert<float>(Q[q * HEAD_DIM + d]);
        float kv = ck_tile::type_convert<float>(K[k * HEAD_DIM + d]);
        sum += qv * kv;
      }
      scores[q * KV_LEN + k] = sum * scale;
    }
  }

  // Softmax per row
  std::vector<float> probs(Q_ROWS * KV_LEN);
  for (int q = 0; q < Q_ROWS; q++) {
    float max_val = -INFINITY;
    for (int k = 0; k < KV_LEN; k++) {
      max_val = std::max(max_val, scores[q * KV_LEN + k]);
    }
    float sum_exp = 0.0f;
    for (int k = 0; k < KV_LEN; k++) {
      probs[q * KV_LEN + k] = std::exp(scores[q * KV_LEN + k] - max_val);
      sum_exp += probs[q * KV_LEN + k];
    }
    for (int k = 0; k < KV_LEN; k++) {
      probs[q * KV_LEN + k] /= sum_exp;
    }
  }

  // Output = probs @ V
  for (int q = 0; q < Q_ROWS; q++) {
    for (int d = 0; d < HEAD_DIM; d++) {
      float sum = 0.0f;
      for (int k = 0; k < KV_LEN; k++) {
        float vv = ck_tile::type_convert<float>(V[k * HEAD_DIM + d]);
        sum += probs[q * KV_LEN + k] * vv;
      }
      O[q * HEAD_DIM + d] = sum;
    }
  }
}

// ============================================================================
// CK Tile-based attention kernel
// ============================================================================
__global__ void attention_ck_kernel(
    const bf16* __restrict__ Q,    // [Q_ROWS, HEAD_DIM]
    const bf16* __restrict__ K,    // [KV_LEN, HEAD_DIM]
    const bf16* __restrict__ V,    // [KV_LEN, HEAD_DIM]
    float* __restrict__ O,         // [Q_ROWS, HEAD_DIM]
    float scale) {

  // Shared memory
  __shared__ bf16 s_Q[Q_ROWS * HEAD_DIM];
  __shared__ bf16 s_K[KV_LEN * HEAD_DIM];
  __shared__ bf16 s_V[KV_LEN * HEAD_DIM];
  __shared__ float s_scores[Q_ROWS * KV_LEN];
  __shared__ float s_O[Q_ROWS * HEAD_DIM];
  __shared__ float s_max[Q_ROWS];
  __shared__ float s_sum[Q_ROWS];

  int tid = threadIdx.x;
  int lane = tid % 64;
  int warp_id = tid / 64;

  // Load Q, K, V to shared memory
  for (int i = tid; i < Q_ROWS * HEAD_DIM; i += NUM_THREADS) {
    s_Q[i] = Q[i];
  }
  for (int i = tid; i < KV_LEN * HEAD_DIM; i += NUM_THREADS) {
    s_K[i] = K[i];
    s_V[i] = V[i];
  }
  __syncthreads();

  // Initialize scores and output
  for (int i = tid; i < Q_ROWS * KV_LEN; i += NUM_THREADS) {
    s_scores[i] = 0.0f;
  }
  for (int i = tid; i < Q_ROWS * HEAD_DIM; i += NUM_THREADS) {
    s_O[i] = 0.0f;
  }
  __syncthreads();

  // =========================================================================
  // Step 1: Compute Q @ K^T using CK Tile warp GEMM
  // Q: [Q_ROWS, HEAD_DIM], K: [KV_LEN, HEAD_DIM]
  // scores: [Q_ROWS, KV_LEN]
  // =========================================================================

  // Use CK Tile's warp GEMM types
  constexpr int HEAD_TILES = HEAD_DIM / MFMA_K;  // 8 tiles for 128-dim
  constexpr int Q_TILES = (Q_ROWS + MFMA_M - 1) / MFMA_M;
  constexpr int KV_TILES = (KV_LEN + MFMA_N - 1) / MFMA_N;

  // Each warp handles one KV tile (16 columns of output)
  if (warp_id < KV_TILES) {
    int kv_tile = warp_id;

    for (int q_tile = 0; q_tile < Q_TILES; q_tile++) {
      // Accumulator for this tile
      using CDataType = float;
      float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

      // Iterate over HEAD_DIM
      for (int k_tile = 0; k_tile < HEAD_TILES; k_tile++) {
        int k_offset = k_tile * MFMA_K;

        // Load Q fragment using CK Tile's expected layout
        // MFMA 16x16x16: lane % 16 = M index, lane / 16 = K group (4 elements each)
        int q_row = q_tile * MFMA_M + (lane % 16);
        int q_col = k_offset + (lane / 16) * 4;

        ck_tile::bf16_t q_frag[4];
        if (q_row < Q_ROWS) {
          for (int i = 0; i < 4; i++) {
            q_frag[i] = s_Q[q_row * HEAD_DIM + q_col + i];
          }
        } else {
          for (int i = 0; i < 4; i++) {
            q_frag[i] = ck_tile::bf16_t(0.0f);
          }
        }

        // Load K fragment (K stored as [KV_LEN, HEAD_DIM])
        int k_row = kv_tile * MFMA_N + (lane % 16);
        int k_col = k_offset + (lane / 16) * 4;

        ck_tile::bf16_t k_frag[4];
        if (k_row < KV_LEN) {
          for (int i = 0; i < 4; i++) {
            k_frag[i] = s_K[k_row * HEAD_DIM + k_col + i];
          }
        } else {
          for (int i = 0; i < 4; i++) {
            k_frag[i] = ck_tile::bf16_t(0.0f);
          }
        }

        // Convert to MFMA vector types
        using mfma_bf16x4 = short __attribute__((ext_vector_type(4)));
        using mfma_float4 = float __attribute__((ext_vector_type(4)));

        mfma_bf16x4 a_vec, b_vec;
        for (int i = 0; i < 4; i++) {
          a_vec[i] = ck_tile::bit_cast<short>(q_frag[i]);
          b_vec[i] = ck_tile::bit_cast<short>(k_frag[i]);
        }

        mfma_float4 c_vec = {acc[0], acc[1], acc[2], acc[3]};

        // MFMA instruction
        c_vec = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_vec, b_vec, c_vec, 0, 0, 0);

        acc[0] = c_vec[0];
        acc[1] = c_vec[1];
        acc[2] = c_vec[2];
        acc[3] = c_vec[3];
      }

      // Store scores with scale
      // MFMA output: lane / 16 gives row group (0-3), lane % 16 gives column
      int out_row_base = q_tile * MFMA_M + (lane / 16) * 4;
      int out_col = kv_tile * MFMA_N + (lane % 16);

      if (out_col < KV_LEN) {
        for (int r = 0; r < 4; r++) {
          int out_row = out_row_base + r;
          if (out_row < Q_ROWS) {
            s_scores[out_row * KV_LEN + out_col] = acc[r] * scale;
          }
        }
      }
    }
  }
  __syncthreads();

  // =========================================================================
  // Step 2: Softmax
  // =========================================================================
  // Find max per row
  if (tid < Q_ROWS) {
    float max_val = -INFINITY;
    for (int k = 0; k < KV_LEN; k++) {
      max_val = fmaxf(max_val, s_scores[tid * KV_LEN + k]);
    }
    s_max[tid] = max_val;
  }
  __syncthreads();

  // Compute exp and sum
  if (tid < Q_ROWS) {
    float sum_exp = 0.0f;
    float max_val = s_max[tid];
    for (int k = 0; k < KV_LEN; k++) {
      float exp_val = expf(s_scores[tid * KV_LEN + k] - max_val);
      s_scores[tid * KV_LEN + k] = exp_val;
      sum_exp += exp_val;
    }
    s_sum[tid] = sum_exp;
  }
  __syncthreads();

  // Normalize
  for (int i = tid; i < Q_ROWS * KV_LEN; i += NUM_THREADS) {
    int row = i / KV_LEN;
    s_scores[i] /= s_sum[row];
  }
  __syncthreads();

  // =========================================================================
  // Step 3: Compute P @ V using MFMA
  // P: [Q_ROWS, KV_LEN], V: [KV_LEN, HEAD_DIM]
  // O: [Q_ROWS, HEAD_DIM]
  // =========================================================================
  constexpr int OUT_TILES = HEAD_DIM / MFMA_N;  // 8 tiles for 128-dim
  constexpr int KV_K_TILES = (KV_LEN + MFMA_K - 1) / MFMA_K;  // 2 tiles for 32

  // Each warp handles one output HEAD_DIM tile
  if (warp_id < OUT_TILES) {
    int out_tile = warp_id;

    for (int q_tile = 0; q_tile < Q_TILES; q_tile++) {
      float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

      // Iterate over KV dimension
      for (int kv_tile = 0; kv_tile < KV_K_TILES; kv_tile++) {
        int kv_offset = kv_tile * MFMA_K;

        // Load P fragment (attention probabilities as bf16)
        int p_row = q_tile * MFMA_M + (lane % 16);
        int p_col = kv_offset + (lane / 16) * 4;

        ck_tile::bf16_t p_frag[4];
        if (p_row < Q_ROWS && p_col + 3 < KV_LEN) {
          for (int i = 0; i < 4; i++) {
            p_frag[i] = ck_tile::type_convert<ck_tile::bf16_t>(s_scores[p_row * KV_LEN + p_col + i]);
          }
        } else {
          for (int i = 0; i < 4; i++) {
            float val = (p_row < Q_ROWS && p_col + i < KV_LEN) ?
                        s_scores[p_row * KV_LEN + p_col + i] : 0.0f;
            p_frag[i] = ck_tile::type_convert<ck_tile::bf16_t>(val);
          }
        }

        // Load V fragment
        int v_row = kv_offset + (lane / 16) * 4;
        int v_col = out_tile * MFMA_N + (lane % 16);

        ck_tile::bf16_t v_frag[4];
        for (int i = 0; i < 4; i++) {
          int vr = v_row + i;
          if (vr < KV_LEN) {
            v_frag[i] = s_V[vr * HEAD_DIM + v_col];
          } else {
            v_frag[i] = ck_tile::bf16_t(0.0f);
          }
        }

        // Convert to MFMA types
        using mfma_bf16x4 = short __attribute__((ext_vector_type(4)));
        using mfma_float4 = float __attribute__((ext_vector_type(4)));

        mfma_bf16x4 a_vec, b_vec;
        for (int i = 0; i < 4; i++) {
          a_vec[i] = ck_tile::bit_cast<short>(p_frag[i]);
          b_vec[i] = ck_tile::bit_cast<short>(v_frag[i]);
        }

        mfma_float4 c_vec = {acc[0], acc[1], acc[2], acc[3]};
        c_vec = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_vec, b_vec, c_vec, 0, 0, 0);

        acc[0] = c_vec[0];
        acc[1] = c_vec[1];
        acc[2] = c_vec[2];
        acc[3] = c_vec[3];
      }

      // Store output
      int out_row_base = q_tile * MFMA_M + (lane / 16) * 4;
      int out_col = out_tile * MFMA_N + (lane % 16);

      for (int r = 0; r < 4; r++) {
        int out_row = out_row_base + r;
        if (out_row < Q_ROWS && out_col < HEAD_DIM) {
          s_O[out_row * HEAD_DIM + out_col] = acc[r];
        }
      }
    }
  }
  __syncthreads();

  // Write output to global memory
  for (int i = tid; i < Q_ROWS * HEAD_DIM; i += NUM_THREADS) {
    O[i] = s_O[i];
  }
}

// ============================================================================
// Main
// ============================================================================
int main() {
  std::cout << "=== CK Tile Attention Test ===" << std::endl;
  std::cout << "Q_ROWS: " << Q_ROWS << ", KV_LEN: " << KV_LEN << ", HEAD_DIM: " << HEAD_DIM << std::endl;

  // Device info
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  std::cout << "Device: " << props.name << std::endl << std::endl;

  float scale = 1.0f / sqrtf(float(HEAD_DIM));

  // Allocate host memory
  std::vector<bf16> h_Q(Q_ROWS * HEAD_DIM);
  std::vector<bf16> h_K(KV_LEN * HEAD_DIM);
  std::vector<bf16> h_V(KV_LEN * HEAD_DIM);
  std::vector<float> h_O(Q_ROWS * HEAD_DIM);
  std::vector<float> h_O_ref(Q_ROWS * HEAD_DIM);

  // Initialize with random data
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-0.5f, 0.5f);

  for (int i = 0; i < Q_ROWS * HEAD_DIM; i++) {
    h_Q[i] = ck_tile::type_convert<bf16>(dis(gen));
  }
  for (int i = 0; i < KV_LEN * HEAD_DIM; i++) {
    h_K[i] = ck_tile::type_convert<bf16>(dis(gen));
    h_V[i] = ck_tile::type_convert<bf16>(dis(gen));
  }

  // Compute reference
  std::cout << "Computing reference..." << std::endl;
  attention_reference(h_Q.data(), h_K.data(), h_V.data(), h_O_ref.data(), scale);

  // Allocate device memory
  bf16 *d_Q, *d_K, *d_V;
  float *d_O;
  HIP_CHECK(hipMalloc(&d_Q, Q_ROWS * HEAD_DIM * sizeof(bf16)));
  HIP_CHECK(hipMalloc(&d_K, KV_LEN * HEAD_DIM * sizeof(bf16)));
  HIP_CHECK(hipMalloc(&d_V, KV_LEN * HEAD_DIM * sizeof(bf16)));
  HIP_CHECK(hipMalloc(&d_O, Q_ROWS * HEAD_DIM * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_Q, h_Q.data(), Q_ROWS * HEAD_DIM * sizeof(bf16), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_K, h_K.data(), KV_LEN * HEAD_DIM * sizeof(bf16), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_V, h_V.data(), KV_LEN * HEAD_DIM * sizeof(bf16), hipMemcpyHostToDevice));

  // Launch kernel
  dim3 grid(1);
  dim3 block(NUM_THREADS);

  std::cout << "Running CK Tile attention kernel..." << std::endl;

  // Warmup
  for (int i = 0; i < NUM_WARMUP; i++) {
    attention_ck_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, scale);
  }
  HIP_CHECK(hipDeviceSynchronize());

  // Benchmark
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  HIP_CHECK(hipEventRecord(start));
  for (int i = 0; i < NUM_ITERS; i++) {
    attention_ck_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, scale);
  }
  HIP_CHECK(hipEventRecord(stop));
  HIP_CHECK(hipEventSynchronize(stop));

  float ms;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
  float avg_us = ms * 1000.0f / NUM_ITERS;

  // Copy back and verify
  HIP_CHECK(hipMemcpy(h_O.data(), d_O, Q_ROWS * HEAD_DIM * sizeof(float), hipMemcpyDeviceToHost));

  // Check correctness
  float max_error = 0.0f;
  float avg_error = 0.0f;
  for (int i = 0; i < Q_ROWS * HEAD_DIM; i++) {
    float error = std::abs(h_O[i] - h_O_ref[i]);
    max_error = std::max(max_error, error);
    avg_error += error;
  }
  avg_error /= (Q_ROWS * HEAD_DIM);

  std::cout << std::endl << "=== Results ===" << std::endl;
  std::cout << "Average time: " << avg_us << " us" << std::endl;
  std::cout << "Max error: " << max_error << std::endl;
  std::cout << "Avg error: " << avg_error << std::endl;

  // Print some output values for debugging
  std::cout << std::endl << "Sample outputs (first 8 values of row 0):" << std::endl;
  std::cout << "GPU: ";
  for (int i = 0; i < 8; i++) {
    std::cout << h_O[i] << " ";
  }
  std::cout << std::endl << "Ref: ";
  for (int i = 0; i < 8; i++) {
    std::cout << h_O_ref[i] << " ";
  }
  std::cout << std::endl;

  bool passed = max_error < 0.2f && avg_error < 0.05f;  // Allow tolerance for bf16
  std::cout << std::endl << "Test " << (passed ? "PASSED" : "FAILED") << std::endl;

  // Cleanup
  HIP_CHECK(hipFree(d_Q));
  HIP_CHECK(hipFree(d_K));
  HIP_CHECK(hipFree(d_V));
  HIP_CHECK(hipFree(d_O));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  return passed ? 0 : 1;
}
