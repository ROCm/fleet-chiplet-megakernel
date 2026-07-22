/* Copyright 2025 CMU
 *
 * Simple MFMA performance test for AMD MI300
 * Tests raw MFMA 16x16x16 bf16 throughput
 */

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

// MFMA intrinsics - use custom names to avoid conflict with HIP types
typedef short __attribute__((ext_vector_type(4))) mfma_bf16x4;
typedef float __attribute__((ext_vector_type(4))) mfma_float4;

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
constexpr int BATCH_SIZE = 8;
constexpr int HIDDEN_SIZE = 4096;
constexpr int OUTPUT_SIZE = 4096;

constexpr int MFMA_M = 16;
constexpr int MFMA_N = 16;
constexpr int MFMA_K = 16;

constexpr int TILE_K = 64;  // K-tile for double buffering
constexpr int NUM_THREADS = 256;
constexpr int NUM_WARPS = NUM_THREADS / 64;

constexpr int NUM_WARMUP = 10;
constexpr int NUM_ITERS = 100;

// ============================================================================
// MFMA-based GEMM kernel
// ============================================================================
__global__ void gemm_mfma_kernel(
    const __hip_bfloat16* __restrict__ A,   // [M, K]
    const __hip_bfloat16* __restrict__ B,   // [N, K] (column-major weight)
    float* __restrict__ C,                   // [M, N]
    int M, int N, int K) {

  int tid = threadIdx.x;
  int lane = tid % 64;
  int warp_id = tid / 64;

  // Each block handles BATCH_SIZE rows x 64 columns
  int block_m = blockIdx.x * BATCH_SIZE;
  int block_n = blockIdx.y * 64;

  // Each warp handles 16 columns
  int warp_n = block_n + warp_id * MFMA_N;

  // Shared memory for tiles
  __shared__ __hip_bfloat16 s_A[BATCH_SIZE * TILE_K];
  __shared__ __hip_bfloat16 s_B[TILE_K * 64];

  // Accumulators - each thread holds 4 fp32 values
  mfma_float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

  // Loop over K dimension
  for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
    // Load A tile [BATCH_SIZE x TILE_K]
    for (int idx = tid; idx < BATCH_SIZE * TILE_K; idx += NUM_THREADS) {
      int row = idx / TILE_K;
      int col = idx % TILE_K;
      int global_row = block_m + row;
      int global_col = k_tile + col;
      if (global_row < M && global_col < K) {
        s_A[row * TILE_K + col] = A[global_row * K + global_col];
      } else {
        s_A[row * TILE_K + col] = __float2bfloat16(0.0f);
      }
    }

    // Load B tile [TILE_K x 64] - B is stored as [N, K]
    for (int idx = tid; idx < TILE_K * 64; idx += NUM_THREADS) {
      int k = idx / 64;
      int n = idx % 64;
      int global_k = k_tile + k;
      int global_n = block_n + n;
      if (global_k < K && global_n < N) {
        s_B[k * 64 + n] = B[global_n * K + global_k];
      } else {
        s_B[k * 64 + n] = __float2bfloat16(0.0f);
      }
    }
    __syncthreads();

    // MFMA computation - iterate over K in MFMA_K chunks
    #pragma unroll
    for (int k = 0; k < TILE_K; k += MFMA_K) {
      // Load A fragment for this warp
      // MFMA 16x16x16: each lane loads 4 bf16 from A
      int a_row = (lane % 16);  // M dimension
      int a_col = k + (lane / 16) * 4;  // K dimension, 4 consecutive elements

      mfma_bf16x4 a_frag;
      if (a_row < BATCH_SIZE) {
        a_frag[0] = reinterpret_cast<const short*>(&s_A[a_row * TILE_K + a_col])[0];
        a_frag[1] = reinterpret_cast<const short*>(&s_A[a_row * TILE_K + a_col + 1])[0];
        a_frag[2] = reinterpret_cast<const short*>(&s_A[a_row * TILE_K + a_col + 2])[0];
        a_frag[3] = reinterpret_cast<const short*>(&s_A[a_row * TILE_K + a_col + 3])[0];
      } else {
        a_frag = {0, 0, 0, 0};
      }

      // Load B fragment
      // For B[K, N], each lane loads 4 bf16 from different K positions
      int b_col = warp_id * MFMA_N + (lane % 16);  // N dimension
      int b_row = k + (lane / 16) * 4;  // K dimension

      mfma_bf16x4 b_frag;
      b_frag[0] = reinterpret_cast<const short*>(&s_B[(b_row + 0) * 64 + (b_col - block_n)])[0];
      b_frag[1] = reinterpret_cast<const short*>(&s_B[(b_row + 1) * 64 + (b_col - block_n)])[0];
      b_frag[2] = reinterpret_cast<const short*>(&s_B[(b_row + 2) * 64 + (b_col - block_n)])[0];
      b_frag[3] = reinterpret_cast<const short*>(&s_B[(b_row + 3) * 64 + (b_col - block_n)])[0];

      // MFMA instruction
      acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b_frag, acc, 0, 0, 0);
    }
    __syncthreads();
  }

  // Store results
  // MFMA output layout: each lane holds 4 values at positions:
  // row = (lane / 16) * 4 + r, col = lane % 16
  int out_row_base = block_m + (lane / 16) * 4;
  int out_col = warp_n + (lane % 16);

  if (out_col < N) {
    #pragma unroll
    for (int r = 0; r < 4; r++) {
      int out_row = out_row_base + r;
      if (out_row < M) {
        C[out_row * N + out_col] = acc[r];
      }
    }
  }
}

// ============================================================================
// Reference GEMM
// ============================================================================
void gemm_reference(
    const __hip_bfloat16* A,
    const __hip_bfloat16* B,
    float* C,
    int M, int N, int K) {

  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        float a = __bfloat162float(A[m * K + k]);
        float b = __bfloat162float(B[n * K + k]);
        sum += a * b;
      }
      C[m * N + n] = sum;
    }
  }
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
  std::cout << "=== AMD MI300 MFMA GEMM Performance Test ===" << std::endl;
  std::cout << "Configuration:" << std::endl;
  std::cout << "  M (batch): " << BATCH_SIZE << std::endl;
  std::cout << "  N (output): " << OUTPUT_SIZE << std::endl;
  std::cout << "  K (hidden): " << HIDDEN_SIZE << std::endl;
  std::cout << std::endl;

  // Device info
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  std::cout << "Device: " << props.name << std::endl;
  std::cout << "  CUs: " << props.multiProcessorCount << std::endl;
  std::cout << "  Clock: " << props.clockRate / 1000 << " MHz" << std::endl;
  std::cout << std::endl;

  // Allocate memory
  size_t A_size = BATCH_SIZE * HIDDEN_SIZE;
  size_t B_size = OUTPUT_SIZE * HIDDEN_SIZE;
  size_t C_size = BATCH_SIZE * OUTPUT_SIZE;

  std::vector<__hip_bfloat16> h_A(A_size);
  std::vector<__hip_bfloat16> h_B(B_size);
  std::vector<float> h_C(C_size);
  std::vector<float> h_C_ref(C_size);

  // Initialize with random data
  for (size_t i = 0; i < A_size; i++) {
    h_A[i] = __float2bfloat16((float)(rand() % 100) / 100.0f - 0.5f);
  }
  for (size_t i = 0; i < B_size; i++) {
    h_B[i] = __float2bfloat16((float)(rand() % 100) / 100.0f - 0.5f);
  }

  // Device memory
  __hip_bfloat16 *d_A, *d_B;
  float *d_C;
  HIP_CHECK(hipMalloc(&d_A, A_size * sizeof(__hip_bfloat16)));
  HIP_CHECK(hipMalloc(&d_B, B_size * sizeof(__hip_bfloat16)));
  HIP_CHECK(hipMalloc(&d_C, C_size * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_A, h_A.data(), A_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_B, h_B.data(), B_size * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));

  // Compute reference
  std::cout << "Computing reference..." << std::endl;
  gemm_reference(h_A.data(), h_B.data(), h_C_ref.data(), BATCH_SIZE, OUTPUT_SIZE, HIDDEN_SIZE);

  // Launch kernel
  dim3 grid((BATCH_SIZE + BATCH_SIZE - 1) / BATCH_SIZE, (OUTPUT_SIZE + 63) / 64);
  dim3 block(NUM_THREADS);

  std::cout << "Grid: " << grid.x << " x " << grid.y << std::endl;
  std::cout << "Block: " << block.x << std::endl;
  std::cout << std::endl;

  // Warmup
  std::cout << "Warming up..." << std::endl;
  for (int i = 0; i < NUM_WARMUP; i++) {
    gemm_mfma_kernel<<<grid, block>>>(d_A, d_B, d_C, BATCH_SIZE, OUTPUT_SIZE, HIDDEN_SIZE);
  }
  HIP_CHECK(hipDeviceSynchronize());

  // Benchmark
  std::cout << "Benchmarking..." << std::endl;
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  HIP_CHECK(hipEventRecord(start));
  for (int i = 0; i < NUM_ITERS; i++) {
    gemm_mfma_kernel<<<grid, block>>>(d_A, d_B, d_C, BATCH_SIZE, OUTPUT_SIZE, HIDDEN_SIZE);
  }
  HIP_CHECK(hipEventRecord(stop));
  HIP_CHECK(hipEventSynchronize(stop));

  float ms;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
  float avg_ms = ms / NUM_ITERS;

  // Calculate TFLOPS
  double flops = 2.0 * BATCH_SIZE * OUTPUT_SIZE * HIDDEN_SIZE;
  double tflops = flops / (avg_ms * 1e9);

  std::cout << std::endl;
  std::cout << "=== Results ===" << std::endl;
  std::cout << "Average time: " << avg_ms * 1000 << " us" << std::endl;
  std::cout << "Throughput: " << tflops << " TFLOPS" << std::endl;

  // Verify correctness
  HIP_CHECK(hipMemcpy(h_C.data(), d_C, C_size * sizeof(float), hipMemcpyDeviceToHost));

  float max_error = 0.0f;
  for (size_t i = 0; i < C_size; i++) {
    float error = std::abs(h_C[i] - h_C_ref[i]);
    if (error > max_error) max_error = error;
  }
  std::cout << "Max error vs reference: " << max_error << std::endl;

  // Cleanup
  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_B));
  HIP_CHECK(hipFree(d_C));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  std::cout << std::endl << "=== Test Complete ===" << std::endl;
  return 0;
}
