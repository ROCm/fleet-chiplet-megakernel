/* Copyright 2025 CMU
 *
 * Standalone performance test for AMD MI300 kernels
 * Tests both linear (GEMM) and attention kernels
 */

#include <chrono>
#include <cmath>
#include <hip/hip_runtime.h>
#include <iostream>
#include <random>
#include <vector>

// CK Tile includes for AMD
#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm.hpp"

using bf16 = ck_tile::bf16_t;

// ============================================================================
// Configuration
// ============================================================================
constexpr int BATCH_SIZE = 8;            // Number of tokens
constexpr int HEAD_DIM = 128;            // Head dimension
constexpr int NUM_QO_HEADS = 8;          // Number of QO heads
constexpr int NUM_KV_HEADS = 1;          // Number of KV heads (GQA)
constexpr int HIDDEN_SIZE = 4096;        // Hidden size
constexpr int INTERMEDIATE_SIZE = 14336; // MLP intermediate size
constexpr int SEQ_LEN = 2048;            // Sequence length for attention
constexpr int PAGE_SIZE = 16;            // Page size for paged attention

constexpr int NUM_WARMUP = 5;
constexpr int NUM_ITERS = 100;

// ============================================================================
// Helper functions
// ============================================================================
#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t err = call;                                                     \
    if (err != hipSuccess) {                                                   \
      std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << ": "      \
                << hipGetErrorString(err) << std::endl;                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

void fill_random_bf16(bf16 *data, size_t n) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
  for (size_t i = 0; i < n; i++) {
    data[i] = ck_tile::type_convert<bf16>(dis(gen));
  }
}

// ============================================================================
// Warp GEMM types for MI300 (BF16 input, FP32 accumulator)
// ============================================================================
using WarpGemmBf16_16x16x16 = ck_tile::WarpGemmMfmaBf16Bf16F32M16N16K16;
using WarpGemmBf16_32x32x8 = ck_tile::WarpGemmMfmaBf16Bf16F32M32N32K8;

// ============================================================================
// Simple Linear kernel using CK Tile warp GEMM
// ============================================================================
__global__ void
    linear_kernel_ck(bf16 const *__restrict__ input,  // [BATCH, HIDDEN]
                     bf16 const *__restrict__ weight, // [OUT, HIDDEN]
                     bf16 *__restrict__ output,       // [BATCH, OUT]
                     int batch_size,
                     int hidden_size,
                     int output_size) {

  // Use CK Tile's warp GEMM
  using WarpGemm = WarpGemmBf16_16x16x16;

  constexpr int MFMA_M = WarpGemm::kM; // 16
  constexpr int MFMA_N = WarpGemm::kN; // 16
  constexpr int MFMA_K = WarpGemm::kK; // 16

  int warp_id = threadIdx.x / 64;
  int lane_id = threadIdx.x % 64;

  // Each warp handles a 16x16 output tile
  int tile_m = blockIdx.x;
  int tile_n = warp_id;

  if (tile_m * MFMA_M >= batch_size || tile_n * MFMA_N >= output_size) {
    return;
  }

  // Initialize accumulator
  using CWarpTensor = typename WarpGemm::CWarpTensor;
  CWarpTensor c_warp_tensor;
  ck_tile::clear_tile(c_warp_tensor);

  // K-dimension loop
  for (int k = 0; k < hidden_size; k += MFMA_K) {
    // Load A tile (input)
    using AWarpTensor = typename WarpGemm::AWarpTensor;
    AWarpTensor a_warp_tensor;

    // Load B tile (weight)
    using BWarpTensor = typename WarpGemm::BWarpTensor;
    BWarpTensor b_warp_tensor;

    // TODO: Load from global memory using CK Tile's load_tile

    // Warp GEMM
    WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);
  }

  // Store output
  // TODO: Store using CK Tile's store_tile
}

// ============================================================================
// Reference implementations for validation
// ============================================================================
void linear_reference(bf16 const *input,
                      bf16 const *weight,
                      bf16 *output,
                      int batch_size,
                      int hidden_size,
                      int output_size) {

  for (int b = 0; b < batch_size; b++) {
    for (int o = 0; o < output_size; o++) {
      float sum = 0.0f;
      for (int h = 0; h < hidden_size; h++) {
        float a = ck_tile::type_convert<float>(input[b * hidden_size + h]);
        float w = ck_tile::type_convert<float>(weight[o * hidden_size + h]);
        sum += a * w;
      }
      output[b * output_size + o] = ck_tile::type_convert<bf16>(sum);
    }
  }
}

// ============================================================================
// Main test function
// ============================================================================
int main(int argc, char **argv) {
  std::cout << "=== AMD MI300 Kernel Performance Test ===" << std::endl;
  std::cout << "Configuration:" << std::endl;
  std::cout << "  BATCH_SIZE: " << BATCH_SIZE << std::endl;
  std::cout << "  HEAD_DIM: " << HEAD_DIM << std::endl;
  std::cout << "  NUM_QO_HEADS: " << NUM_QO_HEADS << std::endl;
  std::cout << "  NUM_KV_HEADS: " << NUM_KV_HEADS << std::endl;
  std::cout << "  HIDDEN_SIZE: " << HIDDEN_SIZE << std::endl;
  std::cout << "  INTERMEDIATE_SIZE: " << INTERMEDIATE_SIZE << std::endl;
  std::cout << "  SEQ_LEN: " << SEQ_LEN << std::endl;
  std::cout << std::endl;

  // Initialize HIP
  int device_count;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  std::cout << "Found " << device_count << " HIP device(s)" << std::endl;

  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  std::cout << "Using device: " << props.name << std::endl;
  std::cout << "  Compute capability: " << props.major << "." << props.minor
            << std::endl;
  std::cout << "  Total memory: " << props.totalGlobalMem / (1024 * 1024)
            << " MB" << std::endl;
  std::cout << "  Shared memory per block: " << props.sharedMemPerBlock
            << " bytes" << std::endl;
  std::cout << std::endl;

  // =========================================================================
  // Test 1: Linear kernel
  // =========================================================================
  std::cout << "=== Test 1: Linear Kernel ===" << std::endl;
  {
    size_t input_size = BATCH_SIZE * HIDDEN_SIZE;
    size_t weight_size = INTERMEDIATE_SIZE * HIDDEN_SIZE;
    size_t output_size = BATCH_SIZE * INTERMEDIATE_SIZE;

    // Allocate host memory
    std::vector<bf16> h_input(input_size);
    std::vector<bf16> h_weight(weight_size);
    std::vector<bf16> h_output(output_size);
    std::vector<bf16> h_ref_output(output_size);

    // Initialize with random data
    fill_random_bf16(h_input.data(), input_size);
    fill_random_bf16(h_weight.data(), weight_size);

    // Allocate device memory
    bf16 *d_input, *d_weight, *d_output;
    HIP_CHECK(hipMalloc(&d_input, input_size * sizeof(bf16)));
    HIP_CHECK(hipMalloc(&d_weight, weight_size * sizeof(bf16)));
    HIP_CHECK(hipMalloc(&d_output, output_size * sizeof(bf16)));

    // Copy to device
    HIP_CHECK(hipMemcpy(d_input,
                        h_input.data(),
                        input_size * sizeof(bf16),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weight,
                        h_weight.data(),
                        weight_size * sizeof(bf16),
                        hipMemcpyHostToDevice));

    // Compute reference
    linear_reference(h_input.data(),
                     h_weight.data(),
                     h_ref_output.data(),
                     BATCH_SIZE,
                     HIDDEN_SIZE,
                     INTERMEDIATE_SIZE);

    // Launch kernel
    dim3 grid((BATCH_SIZE + 15) / 16, 1, 1);
    dim3 block(256, 1, 1);

    // Warmup
    for (int i = 0; i < NUM_WARMUP; i++) {
      linear_kernel_ck<<<grid, block>>>(d_input,
                                        d_weight,
                                        d_output,
                                        BATCH_SIZE,
                                        HIDDEN_SIZE,
                                        INTERMEDIATE_SIZE);
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Benchmark
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < NUM_ITERS; i++) {
      linear_kernel_ck<<<grid, block>>>(d_input,
                                        d_weight,
                                        d_output,
                                        BATCH_SIZE,
                                        HIDDEN_SIZE,
                                        INTERMEDIATE_SIZE);
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / NUM_ITERS;

    // Calculate TFLOPS
    double flops = 2.0 * BATCH_SIZE * HIDDEN_SIZE * INTERMEDIATE_SIZE;
    double tflops = flops / (avg_ms * 1e9);

    std::cout << "  Average time: " << avg_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << tflops << " TFLOPS" << std::endl;

    // Cleanup
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_weight));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
  }

  // =========================================================================
  // Test 2: Attention kernel (placeholder)
  // =========================================================================
  std::cout << std::endl << "=== Test 2: Attention Kernel ===" << std::endl;
  {
    std::cout << "  TODO: Implement CK Tile attention test" << std::endl;

    // Configuration for attention
    size_t q_size = BATCH_SIZE * NUM_QO_HEADS * HEAD_DIM;
    size_t kv_size = SEQ_LEN * NUM_KV_HEADS * HEAD_DIM;
    size_t o_size = BATCH_SIZE * NUM_QO_HEADS * HEAD_DIM;

    std::cout << "  Q size: " << q_size << " elements" << std::endl;
    std::cout << "  KV size: " << kv_size << " elements" << std::endl;
    std::cout << "  O size: " << o_size << " elements" << std::endl;
  }

  std::cout << std::endl << "=== Tests Complete ===" << std::endl;
  return 0;
}
