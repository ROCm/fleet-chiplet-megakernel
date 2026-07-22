/* Copyright 2025 CMU
 *
 * Test CK Tile attention with online softmax (like the persistent kernel)
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

// Configuration - same as Qwen3 with 8 tokens
constexpr int NUM_TOKENS = 8;
constexpr int NUM_QO_PER_KV = 4;
constexpr int Q_ROWS = NUM_TOKENS * NUM_QO_PER_KV;  // 32
constexpr int HEAD_DIM = 128;
constexpr int KV_TILE_SIZE = 32;
constexpr int SEQ_LEN = 64;  // Total sequence length (will process in 2 tiles)

constexpr int MFMA_M = 16;
constexpr int MFMA_N = 16;
constexpr int MFMA_K = 16;

constexpr int NUM_THREADS = 256;
constexpr int NUM_WARPS = 4;

constexpr float log2e = 1.4426950408889634f;

using bf16 = ck_tile::bf16_t;
using mfma_bf16x4 = short __attribute__((ext_vector_type(4)));
using mfma_float4 = float __attribute__((ext_vector_type(4)));

// Reference with online softmax
void attention_ref_online(const bf16* Q, const bf16* K, const bf16* V, float* O, float scale) {
  std::vector<float> m(Q_ROWS, -INFINITY);
  std::vector<float> d(Q_ROWS, 1.0f);
  std::vector<float> o(Q_ROWS * HEAD_DIM, 0.0f);

  int num_iters = (SEQ_LEN + KV_TILE_SIZE - 1) / KV_TILE_SIZE;

  for (int iter = 0; iter < num_iters; iter++) {
    int tile_start = iter * KV_TILE_SIZE;
    int curr_iter_len = std::min(SEQ_LEN - tile_start, KV_TILE_SIZE);

    // Q*K^T for this tile
    std::vector<float> scores(Q_ROWS * KV_TILE_SIZE);
    for (int q = 0; q < Q_ROWS; q++) {
      for (int k = 0; k < curr_iter_len; k++) {
        float sum = 0.0f;
        for (int dd = 0; dd < HEAD_DIM; dd++) {
          sum += ck_tile::type_convert<float>(Q[q * HEAD_DIM + dd]) *
                 ck_tile::type_convert<float>(K[(tile_start + k) * HEAD_DIM + dd]);
        }
        scores[q * KV_TILE_SIZE + k] = sum * scale * log2e;
      }
    }

    // Online softmax
    for (int q = 0; q < Q_ROWS; q++) {
      float m_local = -INFINITY;
      for (int k = 0; k < curr_iter_len; k++) {
        m_local = std::max(m_local, scores[q * KV_TILE_SIZE + k]);
      }

      float m_prev = m[q];
      float m_new = std::max(m_prev, m_local);
      float rescale = std::exp2(m_prev - m_new);

      // Rescale previous output
      for (int dd = 0; dd < HEAD_DIM; dd++) {
        o[q * HEAD_DIM + dd] *= rescale;
      }

      // Compute attention and accumulate
      float d_local = 0.0f;
      for (int k = 0; k < curr_iter_len; k++) {
        float attn = std::exp2(scores[q * KV_TILE_SIZE + k] - m_new);
        d_local += attn;
        for (int dd = 0; dd < HEAD_DIM; dd++) {
          o[q * HEAD_DIM + dd] += attn * ck_tile::type_convert<float>(V[(tile_start + k) * HEAD_DIM + dd]);
        }
      }

      m[q] = m_new;
      d[q] = d[q] * rescale + d_local;
    }
  }

  // Final normalization
  for (int q = 0; q < Q_ROWS; q++) {
    for (int dd = 0; dd < HEAD_DIM; dd++) {
      O[q * HEAD_DIM + dd] = o[q * HEAD_DIM + dd] / d[q];
    }
  }
}

__device__ float inline ptx_exp2(float x) {
  return exp2f(x);
}

// CK Tile attention with online softmax
__global__ void attention_ck_online(
    const bf16* __restrict__ Q,
    const bf16* __restrict__ K,
    const bf16* __restrict__ V,
    float* __restrict__ O,
    float scale) {

  constexpr int Q_TILES_M = (Q_ROWS + MFMA_M - 1) / MFMA_M;  // 2
  constexpr int KV_TILES_N = (KV_TILE_SIZE + MFMA_N - 1) / MFMA_N;  // 2
  constexpr int HEAD_TILES_K = HEAD_DIM / MFMA_K;  // 8
  constexpr int OUT_TILES_N = HEAD_DIM / MFMA_N;  // 8
  constexpr int PV_TILES_K = (KV_TILE_SIZE + MFMA_K - 1) / MFMA_K;  // 2

  __shared__ bf16 s_q[Q_ROWS * HEAD_DIM];
  __shared__ bf16 s_k[KV_TILE_SIZE * HEAD_DIM];
  __shared__ bf16 s_v[KV_TILE_SIZE * HEAD_DIM];
  __shared__ float s_scores[Q_ROWS * KV_TILE_SIZE];
  __shared__ float s_o[Q_ROWS * HEAD_DIM];
  __shared__ float s_m[Q_ROWS];
  __shared__ float s_d[Q_ROWS];

  int tid = threadIdx.x;
  int lane = tid % 64;
  int warp_id = tid / 64;

  // Initialize
  for (int i = tid; i < Q_ROWS; i += NUM_THREADS) {
    s_m[i] = -INFINITY;
    s_d[i] = 1.0f;
  }
  for (int i = tid; i < Q_ROWS * HEAD_DIM; i += NUM_THREADS) {
    s_o[i] = 0.0f;
  }

  // Load Q
  for (int i = tid; i < Q_ROWS * HEAD_DIM; i += NUM_THREADS) {
    s_q[i] = Q[i];
  }
  __syncthreads();

  int num_iters = (SEQ_LEN + KV_TILE_SIZE - 1) / KV_TILE_SIZE;

  for (int iter = 0; iter < num_iters; iter++) {
    int tile_start = iter * KV_TILE_SIZE;
    int curr_iter_len = min(SEQ_LEN - tile_start, KV_TILE_SIZE);

    // Load K and V for this tile
    for (int i = tid; i < KV_TILE_SIZE * HEAD_DIM; i += NUM_THREADS) {
      int row = i / HEAD_DIM;
      if (row < curr_iter_len) {
        s_k[i] = K[(tile_start + row) * HEAD_DIM + (i % HEAD_DIM)];
        s_v[i] = V[(tile_start + row) * HEAD_DIM + (i % HEAD_DIM)];
      } else {
        s_k[i] = bf16(0.0f);
        s_v[i] = bf16(0.0f);
      }
    }

    // Init scores
    for (int i = tid; i < Q_ROWS * KV_TILE_SIZE; i += NUM_THREADS) {
      s_scores[i] = 0.0f;
    }
    __syncthreads();

    // MFMA Q*K^T
    if (warp_id < KV_TILES_N) {
      int tile_n = warp_id;

      for (int tile_m = 0; tile_m < Q_TILES_M; tile_m++) {
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        #pragma unroll
        for (int tile_k = 0; tile_k < HEAD_TILES_K; tile_k++) {
          int k_base = tile_k * MFMA_K;

          int q_row = tile_m * MFMA_M + (lane % 16);
          int q_col = k_base + (lane / 16) * 4;

          bf16 q_frag[4];
          for (int i = 0; i < 4; i++) {
            q_frag[i] = (q_row < Q_ROWS) ? s_q[q_row * HEAD_DIM + q_col + i] : bf16(0.0f);
          }

          int k_row = tile_n * MFMA_N + (lane % 16);
          int k_col = k_base + (lane / 16) * 4;

          bf16 k_frag[4];
          for (int i = 0; i < 4; i++) {
            k_frag[i] = (k_row < curr_iter_len) ? s_k[k_row * HEAD_DIM + k_col + i] : bf16(0.0f);
          }

          mfma_bf16x4 a_vec, b_vec;
          for (int i = 0; i < 4; i++) {
            a_vec[i] = ck_tile::bit_cast<short>(q_frag[i]);
            b_vec[i] = ck_tile::bit_cast<short>(k_frag[i]);
          }

          mfma_float4 c_vec = {acc[0], acc[1], acc[2], acc[3]};
          c_vec = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_vec, b_vec, c_vec, 0, 0, 0);
          acc[0] = c_vec[0]; acc[1] = c_vec[1]; acc[2] = c_vec[2]; acc[3] = c_vec[3];
        }

        int out_row_base = tile_m * MFMA_M + (lane / 16) * 4;
        int out_col = tile_n * MFMA_N + (lane % 16);

        if (out_col < curr_iter_len) {
          for (int r = 0; r < 4; r++) {
            int out_row = out_row_base + r;
            if (out_row < Q_ROWS) {
              s_scores[out_row * KV_TILE_SIZE + out_col] = acc[r] * scale * log2e;
            }
          }
        }
      }
    }
    __syncthreads();

    // Online softmax (scalar)
    for (int qr = tid; qr < Q_ROWS; qr += NUM_THREADS) {
      float m_local = -INFINITY;
      for (int kc = 0; kc < curr_iter_len; kc++) {
        m_local = fmaxf(m_local, s_scores[qr * KV_TILE_SIZE + kc]);
      }

      float m_prev = s_m[qr];
      float m_new = fmaxf(m_prev, m_local);
      float rescale = ptx_exp2(m_prev - m_new);

      for (int d = 0; d < HEAD_DIM; d++) {
        s_o[qr * HEAD_DIM + d] *= rescale;
      }

      float d_local = 0.0f;
      for (int kc = 0; kc < curr_iter_len; kc++) {
        float attn = ptx_exp2(s_scores[qr * KV_TILE_SIZE + kc] - m_new);
        s_scores[qr * KV_TILE_SIZE + kc] = attn;
        d_local += attn;
      }

      s_m[qr] = m_new;
      s_d[qr] = s_d[qr] * rescale + d_local;
    }
    __syncthreads();

    // MFMA P*V
    for (int out_tile = warp_id; out_tile < OUT_TILES_N; out_tile += NUM_WARPS) {
      for (int tile_m = 0; tile_m < Q_TILES_M; tile_m++) {
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int tile_k = 0; tile_k < PV_TILES_K; tile_k++) {
          int k_base = tile_k * MFMA_K;
          if (k_base >= curr_iter_len) break;

          int p_row = tile_m * MFMA_M + (lane % 16);
          int p_col = k_base + (lane / 16) * 4;

          bf16 p_frag[4];
          for (int i = 0; i < 4; i++) {
            float val = (p_row < Q_ROWS && p_col + i < curr_iter_len) ?
                        s_scores[p_row * KV_TILE_SIZE + p_col + i] : 0.0f;
            p_frag[i] = ck_tile::type_convert<bf16>(val);
          }

          int v_row = k_base + (lane / 16) * 4;
          int v_col = out_tile * MFMA_N + (lane % 16);

          bf16 v_frag[4];
          for (int i = 0; i < 4; i++) {
            int vr = v_row + i;
            v_frag[i] = (vr < curr_iter_len) ? s_v[vr * HEAD_DIM + v_col] : bf16(0.0f);
          }

          mfma_bf16x4 a_vec, b_vec;
          for (int i = 0; i < 4; i++) {
            a_vec[i] = ck_tile::bit_cast<short>(p_frag[i]);
            b_vec[i] = ck_tile::bit_cast<short>(v_frag[i]);
          }

          mfma_float4 c_vec = {acc[0], acc[1], acc[2], acc[3]};
          c_vec = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_vec, b_vec, c_vec, 0, 0, 0);
          acc[0] = c_vec[0]; acc[1] = c_vec[1]; acc[2] = c_vec[2]; acc[3] = c_vec[3];
        }

        int out_row_base = tile_m * MFMA_M + (lane / 16) * 4;
        int out_col = out_tile * MFMA_N + (lane % 16);

        for (int r = 0; r < 4; r++) {
          int out_row = out_row_base + r;
          if (out_row < Q_ROWS && out_col < HEAD_DIM) {
            s_o[out_row * HEAD_DIM + out_col] += acc[r];
          }
        }
      }
    }
    __syncthreads();
  }

  // Final normalization and output
  for (int i = tid; i < Q_ROWS * HEAD_DIM; i += NUM_THREADS) {
    int row = i / HEAD_DIM;
    O[i] = s_o[i] / s_d[row];
  }
}

int main() {
  std::cout << "=== CK Tile Attention with Online Softmax ===" << std::endl;
  std::cout << "Q_ROWS: " << Q_ROWS << " (tokens=" << NUM_TOKENS << " x heads_per_kv=" << NUM_QO_PER_KV << ")" << std::endl;
  std::cout << "SEQ_LEN: " << SEQ_LEN << ", KV_TILE: " << KV_TILE_SIZE << ", HEAD_DIM: " << HEAD_DIM << std::endl;

  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  std::cout << "Device: " << props.name << std::endl << std::endl;

  float scale = 1.0f / sqrtf(float(HEAD_DIM));

  std::vector<bf16> h_Q(Q_ROWS * HEAD_DIM);
  std::vector<bf16> h_K(SEQ_LEN * HEAD_DIM);
  std::vector<bf16> h_V(SEQ_LEN * HEAD_DIM);
  std::vector<float> h_O(Q_ROWS * HEAD_DIM);
  std::vector<float> h_O_ref(Q_ROWS * HEAD_DIM);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
  for (auto& v : h_Q) v = ck_tile::type_convert<bf16>(dis(gen));
  for (auto& v : h_K) v = ck_tile::type_convert<bf16>(dis(gen));
  for (auto& v : h_V) v = ck_tile::type_convert<bf16>(dis(gen));

  std::cout << "Computing reference with online softmax..." << std::endl;
  attention_ref_online(h_Q.data(), h_K.data(), h_V.data(), h_O_ref.data(), scale);

  bf16 *d_Q, *d_K, *d_V;
  float *d_O;
  HIP_CHECK(hipMalloc(&d_Q, Q_ROWS * HEAD_DIM * sizeof(bf16)));
  HIP_CHECK(hipMalloc(&d_K, SEQ_LEN * HEAD_DIM * sizeof(bf16)));
  HIP_CHECK(hipMalloc(&d_V, SEQ_LEN * HEAD_DIM * sizeof(bf16)));
  HIP_CHECK(hipMalloc(&d_O, Q_ROWS * HEAD_DIM * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_Q, h_Q.data(), Q_ROWS * HEAD_DIM * sizeof(bf16), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_K, h_K.data(), SEQ_LEN * HEAD_DIM * sizeof(bf16), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_V, h_V.data(), SEQ_LEN * HEAD_DIM * sizeof(bf16), hipMemcpyHostToDevice));

  // Warmup
  for (int i = 0; i < 10; i++) {
    attention_ck_online<<<1, NUM_THREADS>>>(d_Q, d_K, d_V, d_O, scale);
  }
  HIP_CHECK(hipDeviceSynchronize());

  // Benchmark
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  HIP_CHECK(hipEventRecord(start));
  for (int i = 0; i < 100; i++) {
    attention_ck_online<<<1, NUM_THREADS>>>(d_Q, d_K, d_V, d_O, scale);
  }
  HIP_CHECK(hipEventRecord(stop));
  HIP_CHECK(hipEventSynchronize(stop));

  float ms;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
  float avg_us = ms * 1000.0f / 100;

  HIP_CHECK(hipMemcpy(h_O.data(), d_O, Q_ROWS * HEAD_DIM * sizeof(float), hipMemcpyDeviceToHost));

  float max_err = 0.0f, avg_err = 0.0f;
  for (int i = 0; i < Q_ROWS * HEAD_DIM; i++) {
    float e = std::abs(h_O[i] - h_O_ref[i]);
    max_err = std::max(max_err, e);
    avg_err += e;
  }
  avg_err /= (Q_ROWS * HEAD_DIM);

  std::cout << "=== Results ===" << std::endl;
  std::cout << "Time: " << avg_us << " us" << std::endl;
  std::cout << "Max error: " << max_err << ", Avg error: " << avg_err << std::endl;

  std::cout << "\nSample outputs (first 8 values of row 0):" << std::endl;
  std::cout << "GPU: ";
  for (int i = 0; i < 8; i++) std::cout << h_O[i] << " ";
  std::cout << std::endl << "Ref: ";
  for (int i = 0; i < 8; i++) std::cout << h_O_ref[i] << " ";
  std::cout << std::endl;

  bool passed = max_err < 0.3f && avg_err < 0.1f;
  std::cout << "\nTest " << (passed ? "PASSED" : "FAILED") << std::endl;

  HIP_CHECK(hipFree(d_Q));
  HIP_CHECK(hipFree(d_K));
  HIP_CHECK(hipFree(d_V));
  HIP_CHECK(hipFree(d_O));

  return passed ? 0 : 1;
}
