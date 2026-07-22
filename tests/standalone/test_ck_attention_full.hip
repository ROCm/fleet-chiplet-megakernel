/* Copyright 2025 CMU
 *
 * Full attention test with realistic model parameters
 * Simulates Qwen3-8B: 8 QO heads per KV head, 128 head dim
 */

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#include "ck_tile/core.hpp"

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t err = call;                                                     \
    if (err != hipSuccess) {                                                   \
      std::cerr << "HIP error: " << hipGetErrorString(err) << std::endl;       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// ============================================================================
// Realistic configuration (Qwen3-8B like)
// ============================================================================
constexpr int NUM_TOKENS = 8;         // Batch of tokens
constexpr int NUM_QO_PER_KV = 8;      // GQA ratio
constexpr int Q_ROWS = NUM_TOKENS * NUM_QO_PER_KV;  // 64 query rows
constexpr int HEAD_DIM = 128;
constexpr int KV_TILE_SIZE = 32;      // Process 32 KV positions at a time

constexpr int MFMA_M = 16;
constexpr int MFMA_N = 16;
constexpr int MFMA_K = 16;

constexpr int NUM_THREADS = 256;

using bf16 = ck_tile::bf16_t;

// MFMA vector types
typedef short __attribute__((ext_vector_type(4))) mfma_bf16x4;
typedef float __attribute__((ext_vector_type(4))) mfma_float4;

// ============================================================================
// Optimized attention kernel with MFMA for both Q*K and P*V
// ============================================================================
__global__ void attention_mfma_full(
    const bf16* __restrict__ Q,    // [Q_ROWS, HEAD_DIM]
    const bf16* __restrict__ K,    // [KV_TILE_SIZE, HEAD_DIM]
    const bf16* __restrict__ V,    // [KV_TILE_SIZE, HEAD_DIM]
    float* __restrict__ O,         // [Q_ROWS, HEAD_DIM]
    int kv_len,                    // Actual KV length <= KV_TILE_SIZE
    float scale) {

  constexpr int HEAD_TILES = HEAD_DIM / MFMA_K;
  constexpr int Q_TILES = (Q_ROWS + MFMA_M - 1) / MFMA_M;  // 4 tiles for 64 rows
  constexpr int KV_TILES = (KV_TILE_SIZE + MFMA_N - 1) / MFMA_N;  // 2 tiles for 32
  constexpr int OUT_TILES = HEAD_DIM / MFMA_N;  // 8 tiles for 128

  // LDS usage: 16384 + 8192 + 8192 + 8192 + 512 = 41,472 bytes < 65536 limit
  __shared__ bf16 s_Q[Q_ROWS * HEAD_DIM];           // 16384 bytes
  __shared__ bf16 s_K[KV_TILE_SIZE * HEAD_DIM];     // 8192 bytes
  __shared__ bf16 s_V[KV_TILE_SIZE * HEAD_DIM];     // 8192 bytes
  __shared__ float s_scores[Q_ROWS * KV_TILE_SIZE]; // 8192 bytes
  // s_O removed - write directly to global memory to save 32KB LDS
  __shared__ float s_m[Q_ROWS];  // Max for softmax (256 bytes)
  __shared__ float s_l[Q_ROWS];  // Sum for softmax (256 bytes)

  int tid = threadIdx.x;
  int lane = tid % 64;
  int warp_id = tid / 64;

  // Load Q, K, V
  for (int i = tid; i < Q_ROWS * HEAD_DIM; i += NUM_THREADS) {
    s_Q[i] = Q[i];
  }
  for (int i = tid; i < KV_TILE_SIZE * HEAD_DIM; i += NUM_THREADS) {
    s_K[i] = K[i];
    s_V[i] = V[i];
  }

  // Initialize scores and global output
  for (int i = tid; i < Q_ROWS * KV_TILE_SIZE; i += NUM_THREADS) {
    s_scores[i] = 0.0f;
  }
  for (int i = tid; i < Q_ROWS * HEAD_DIM; i += NUM_THREADS) {
    O[i] = 0.0f;
  }
  __syncthreads();

  // =========================================================================
  // Q * K^T using MFMA (all 4 warps work on different output tiles)
  // =========================================================================
  {
    // Warp assignment: each warp handles different Q tile rows
    // With 4 warps and 4 Q_TILES, each warp handles one Q_TILE
    int q_tile = warp_id;
    if (q_tile < Q_TILES) {
      for (int kv_tile = 0; kv_tile < KV_TILES; kv_tile++) {
        mfma_float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

        #pragma unroll
        for (int k_tile = 0; k_tile < HEAD_TILES; k_tile++) {
          int k_off = k_tile * MFMA_K;

          // Load Q fragment
          int q_row = q_tile * MFMA_M + (lane % 16);
          int q_col = k_off + (lane / 16) * 4;

          mfma_bf16x4 q_frag;
          if (q_row < Q_ROWS) {
            q_frag[0] = ck_tile::bit_cast<short>(s_Q[q_row * HEAD_DIM + q_col + 0]);
            q_frag[1] = ck_tile::bit_cast<short>(s_Q[q_row * HEAD_DIM + q_col + 1]);
            q_frag[2] = ck_tile::bit_cast<short>(s_Q[q_row * HEAD_DIM + q_col + 2]);
            q_frag[3] = ck_tile::bit_cast<short>(s_Q[q_row * HEAD_DIM + q_col + 3]);
          } else {
            q_frag = {0, 0, 0, 0};
          }

          // Load K fragment
          int k_row = kv_tile * MFMA_N + (lane % 16);
          int k_col = k_off + (lane / 16) * 4;

          mfma_bf16x4 k_frag;
          if (k_row < kv_len) {
            k_frag[0] = ck_tile::bit_cast<short>(s_K[k_row * HEAD_DIM + k_col + 0]);
            k_frag[1] = ck_tile::bit_cast<short>(s_K[k_row * HEAD_DIM + k_col + 1]);
            k_frag[2] = ck_tile::bit_cast<short>(s_K[k_row * HEAD_DIM + k_col + 2]);
            k_frag[3] = ck_tile::bit_cast<short>(s_K[k_row * HEAD_DIM + k_col + 3]);
          } else {
            k_frag = {0, 0, 0, 0};
          }

          acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(q_frag, k_frag, acc, 0, 0, 0);
        }

        // Store scores
        int out_row_base = q_tile * MFMA_M + (lane / 16) * 4;
        int out_col = kv_tile * MFMA_N + (lane % 16);

        if (out_col < kv_len) {
          #pragma unroll
          for (int r = 0; r < 4; r++) {
            int out_row = out_row_base + r;
            if (out_row < Q_ROWS) {
              s_scores[out_row * KV_TILE_SIZE + out_col] = acc[r] * scale;
            }
          }
        }
      }
    }
  }
  __syncthreads();

  // =========================================================================
  // Softmax (parallel per row)
  // =========================================================================
  for (int row = tid; row < Q_ROWS; row += NUM_THREADS) {
    float max_val = -INFINITY;
    for (int k = 0; k < kv_len; k++) {
      max_val = fmaxf(max_val, s_scores[row * KV_TILE_SIZE + k]);
    }
    s_m[row] = max_val;

    float sum_exp = 0.0f;
    for (int k = 0; k < kv_len; k++) {
      float exp_val = expf(s_scores[row * KV_TILE_SIZE + k] - max_val);
      s_scores[row * KV_TILE_SIZE + k] = exp_val;
      sum_exp += exp_val;
    }
    s_l[row] = sum_exp;

    // Normalize in place
    for (int k = 0; k < kv_len; k++) {
      s_scores[row * KV_TILE_SIZE + k] /= sum_exp;
    }
  }
  __syncthreads();

  // =========================================================================
  // P * V using MFMA
  // =========================================================================
  {
    // Each warp handles 2 output tiles (8 tiles / 4 warps = 2 each)
    int tiles_per_warp = (OUT_TILES + 3) / 4;

    for (int tile_idx = 0; tile_idx < tiles_per_warp; tile_idx++) {
      int out_tile = warp_id * tiles_per_warp + tile_idx;
      if (out_tile >= OUT_TILES) break;

      for (int q_tile = 0; q_tile < Q_TILES; q_tile++) {
        mfma_float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

        // Iterate over KV dimension
        for (int kv_tile = 0; kv_tile < KV_TILES; kv_tile++) {
          int kv_off = kv_tile * MFMA_K;

          // Load P fragment (convert fp32 scores to bf16)
          int p_row = q_tile * MFMA_M + (lane % 16);
          int p_col = kv_off + (lane / 16) * 4;

          mfma_bf16x4 p_frag;
          #pragma unroll
          for (int i = 0; i < 4; i++) {
            float val = 0.0f;
            if (p_row < Q_ROWS && p_col + i < kv_len) {
              val = s_scores[p_row * KV_TILE_SIZE + p_col + i];
            }
            p_frag[i] = ck_tile::bit_cast<short>(ck_tile::type_convert<bf16>(val));
          }

          // Load V fragment
          int v_row = kv_off + (lane / 16) * 4;
          int v_col = out_tile * MFMA_N + (lane % 16);

          mfma_bf16x4 v_frag;
          #pragma unroll
          for (int i = 0; i < 4; i++) {
            int vr = v_row + i;
            if (vr < kv_len) {
              v_frag[i] = ck_tile::bit_cast<short>(s_V[vr * HEAD_DIM + v_col]);
            } else {
              v_frag[i] = 0;
            }
          }

          acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(p_frag, v_frag, acc, 0, 0, 0);
        }

        // Store output directly to global memory (saves 32KB LDS)
        int out_row_base = q_tile * MFMA_M + (lane / 16) * 4;
        int out_col = out_tile * MFMA_N + (lane % 16);

        #pragma unroll
        for (int r = 0; r < 4; r++) {
          int out_row = out_row_base + r;
          if (out_row < Q_ROWS && out_col < HEAD_DIM) {
            O[out_row * HEAD_DIM + out_col] = acc[r];
          }
        }
      }
    }
  }
}

// Reference
void attention_ref(const bf16* Q, const bf16* K, const bf16* V, float* O, int kv_len, float scale) {
  std::vector<float> scores(Q_ROWS * kv_len);

  // Q @ K^T
  for (int q = 0; q < Q_ROWS; q++) {
    for (int k = 0; k < kv_len; k++) {
      float sum = 0.0f;
      for (int d = 0; d < HEAD_DIM; d++) {
        sum += ck_tile::type_convert<float>(Q[q * HEAD_DIM + d]) *
               ck_tile::type_convert<float>(K[k * HEAD_DIM + d]);
      }
      scores[q * kv_len + k] = sum * scale;
    }
  }

  // Softmax
  for (int q = 0; q < Q_ROWS; q++) {
    float max_val = -INFINITY;
    for (int k = 0; k < kv_len; k++) max_val = std::max(max_val, scores[q * kv_len + k]);
    float sum = 0.0f;
    for (int k = 0; k < kv_len; k++) {
      scores[q * kv_len + k] = std::exp(scores[q * kv_len + k] - max_val);
      sum += scores[q * kv_len + k];
    }
    for (int k = 0; k < kv_len; k++) scores[q * kv_len + k] /= sum;
  }

  // P @ V
  for (int q = 0; q < Q_ROWS; q++) {
    for (int d = 0; d < HEAD_DIM; d++) {
      float sum = 0.0f;
      for (int k = 0; k < kv_len; k++) {
        sum += scores[q * kv_len + k] * ck_tile::type_convert<float>(V[k * HEAD_DIM + d]);
      }
      O[q * HEAD_DIM + d] = sum;
    }
  }
}

int main() {
  std::cout << "=== Full Attention MFMA Test ===" << std::endl;
  std::cout << "Q_ROWS: " << Q_ROWS << " (tokens=" << NUM_TOKENS << " x heads=" << NUM_QO_PER_KV << ")" << std::endl;
  std::cout << "KV_TILE: " << KV_TILE_SIZE << ", HEAD_DIM: " << HEAD_DIM << std::endl;

  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  std::cout << "Device: " << props.name << std::endl << std::endl;

  float scale = 1.0f / sqrtf(float(HEAD_DIM));
  int kv_len = KV_TILE_SIZE;

  // Allocate
  std::vector<bf16> h_Q(Q_ROWS * HEAD_DIM);
  std::vector<bf16> h_K(KV_TILE_SIZE * HEAD_DIM);
  std::vector<bf16> h_V(KV_TILE_SIZE * HEAD_DIM);
  std::vector<float> h_O(Q_ROWS * HEAD_DIM);
  std::vector<float> h_O_ref(Q_ROWS * HEAD_DIM);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
  for (auto& v : h_Q) v = ck_tile::type_convert<bf16>(dis(gen));
  for (auto& v : h_K) v = ck_tile::type_convert<bf16>(dis(gen));
  for (auto& v : h_V) v = ck_tile::type_convert<bf16>(dis(gen));

  // Reference
  attention_ref(h_Q.data(), h_K.data(), h_V.data(), h_O_ref.data(), kv_len, scale);

  // GPU
  bf16 *d_Q, *d_K, *d_V;
  float *d_O;
  HIP_CHECK(hipMalloc(&d_Q, Q_ROWS * HEAD_DIM * sizeof(bf16)));
  HIP_CHECK(hipMalloc(&d_K, KV_TILE_SIZE * HEAD_DIM * sizeof(bf16)));
  HIP_CHECK(hipMalloc(&d_V, KV_TILE_SIZE * HEAD_DIM * sizeof(bf16)));
  HIP_CHECK(hipMalloc(&d_O, Q_ROWS * HEAD_DIM * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_Q, h_Q.data(), Q_ROWS * HEAD_DIM * sizeof(bf16), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_K, h_K.data(), KV_TILE_SIZE * HEAD_DIM * sizeof(bf16), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_V, h_V.data(), KV_TILE_SIZE * HEAD_DIM * sizeof(bf16), hipMemcpyHostToDevice));

  // Warmup
  for (int i = 0; i < 10; i++) {
    attention_mfma_full<<<1, NUM_THREADS>>>(d_Q, d_K, d_V, d_O, kv_len, scale);
  }
  HIP_CHECK(hipDeviceSynchronize());

  // Benchmark
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  HIP_CHECK(hipEventRecord(start));
  for (int i = 0; i < 100; i++) {
    attention_mfma_full<<<1, NUM_THREADS>>>(d_Q, d_K, d_V, d_O, kv_len, scale);
  }
  HIP_CHECK(hipEventRecord(stop));
  HIP_CHECK(hipEventSynchronize(stop));

  float ms;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
  float avg_us = ms * 1000.0f / 100;

  // Verify
  HIP_CHECK(hipMemcpy(h_O.data(), d_O, Q_ROWS * HEAD_DIM * sizeof(float), hipMemcpyDeviceToHost));

  float max_err = 0.0f, avg_err = 0.0f;
  for (int i = 0; i < Q_ROWS * HEAD_DIM; i++) {
    float e = std::abs(h_O[i] - h_O_ref[i]);
    max_err = std::max(max_err, e);
    avg_err += e;
  }
  avg_err /= (Q_ROWS * HEAD_DIM);

  std::cout << "=== Results ===" << std::endl;
  std::cout << "Time per attention tile: " << avg_us << " us" << std::endl;
  std::cout << "Max error: " << max_err << ", Avg error: " << avg_err << std::endl;

  bool passed = max_err < 0.2f && avg_err < 0.05f;
  std::cout << "Test " << (passed ? "PASSED" : "FAILED") << std::endl;

  HIP_CHECK(hipFree(d_Q));
  HIP_CHECK(hipFree(d_K));
  HIP_CHECK(hipFree(d_V));
  HIP_CHECK(hipFree(d_O));

  return passed ? 0 : 1;
}
