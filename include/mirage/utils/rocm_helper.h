/* Copyright 2023-2024 CMU
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

#include "mirage/config.h"
#include "mirage/utils/device_macros.h"

#ifdef MIRAGE_BACKEND_USE_ROCM
// Ensure HIP platform is defined before including HIP headers
#include "mirage/hip_platform.h"
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
// rocblas_helper provides cutlass::Array, cutlass::half_t, cutlass::fast_exp_op, etc.
#include "mirage/utils/rocblas_helper.h"
#endif

#include "mirage/type.h"
#include <iostream>
#include <sstream>
#include <string>

namespace mirage {

#define FatalError(s)                                                          \
  do {                                                                         \
    std::stringstream _where, _message;                                        \
    _where << __FILE__ << ':' << __LINE__;                                     \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;          \
    std::cerr << _message.str() << "\nAborting...\n";                          \
    assert(false);                                                             \
    exit(1);                                                                   \
  } while (0)

// HIP doesn't have CURAND, so checkCURAND is a no-op for ROCm
#define checkCURAND(status)                                                    \
  do {                                                                         \
    /* No-op for ROCm */                                                       \
  } while (0)

#define checkCUDA(status)                                                      \
  do {                                                                         \
    std::stringstream _error;                                                  \
    if (status != 0) {                                                         \
      _error << "HIP failure: " << status;                                    \
      FatalError(_error.str());                                                \
    }                                                                          \
  } while (0)

// HIP shuffle functions don't use _sync suffix
template <typename T>
CUTLASS_DEVICE T warp_uniform(T value) {
  struct {
    union {
      T value;
      uint32_t asInt;
    };
  } p;
  p.value = value;
  // HIP: __shfl doesn't need mask parameter
  p.asInt = __shfl((unsigned)p.asInt, 0);
  return p.value;
}

template <typename T>
CUTLASS_DEVICE T *warp_uniform(T *ptr) {
  struct {
    union {
      T *ptr;
      uint32_t asInt[2];
    };
  } p;
  p.ptr = ptr;
  p.asInt[0] = warp_uniform(p.asInt[0]);
  p.asInt[1] = warp_uniform(p.asInt[1]);
  return p.ptr;
}

template <int WARPS_PER_BLOCK = 4, int WARP_SIZE = 32>
CUTLASS_DEVICE float block_sum_fp32(float sum) {

  __shared__ float red_smem[WARPS_PER_BLOCK];
  // Decompose the thread index into warp / lane.
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

// Compute the sum per warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    // HIP: __shfl_xor doesn't need mask parameter
    sum += __shfl_xor(sum, mask);
  }

  // Warp leaders store the data to shared memory.
  if (lane == 0) {
    red_smem[warp] = sum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The warps compute the final sums.
  if (lane < WARPS_PER_BLOCK) {
    sum = red_smem[lane];
  }

// Parallel reduction inside the warp.
#pragma unroll
  for (int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2) {
    // HIP: __shfl_xor doesn't need mask parameter
    sum += __shfl_xor(sum, mask);
  }

  // Broadcast to other threads.
  return __shfl(sum, 0);
}

using namespace mirage::type;

template <ActivationType act_type, int N>
struct act_function {
  CUTLASS_DEVICE cutlass::Array<cutlass::half_t, N>
      operator()(cutlass::Array<cutlass::half_t, N> const &rhs) const {
    // default do nothing
    return rhs;
  }
};

template <int N>
struct act_function<ActivationType::ACT_EXP, N> {
  using func = cutlass::fast_exp_op<cutlass::Array<cutlass::half_t, N>>;
  func exp;

  CUTLASS_DEVICE cutlass::Array<cutlass::half_t, N>
      operator()(cutlass::Array<cutlass::half_t, N> const &rhs) const {
    return exp(rhs);
  }
};

namespace utils {
using namespace mirage::type;

struct FpPointerList {
  mirage::type::FPType *ptrs[mirage::config::MAX_NUM_DEVICES];
};

// For ROCm, use rocblas_datatype instead of cudaDataType_t
using cudaDataType_t = rocblas_datatype;
rocblas_datatype to_cuda_datatype(DataType type);

size_t get_max_shared_mem();

CUTLASS_DEVICE
ActivationType get_matmul_activation_type(TBOperatorType const *operator_types,
                                          int op,
                                          int num_operators) {
  assert(op <= num_operators);
  assert(operator_types[op] == TB_MATMUL_OP);
  if (op == num_operators) {
    return ACT_NONE;
  }

  if (operator_types[op + 1] == TB_EXP_OP) {
    return ACT_EXP;
  }

  return ACT_NONE;
}

CUTLASS_HOST_DEVICE inline
int get_reduction_dim(TBOperatorType type) {
  if (type >= TB_REDUCTION_0_TO_DIMX_OP && type <= TB_REDUCTION_2_TO_DIMX_OP) {
    return type - TB_REDUCTION_0_TO_DIMX_OP;
  } else if (type >= TB_REDUCTION_0_OP && type <= TB_REDUCTION_2_OP) {
    return type - TB_REDUCTION_0_OP;
  } else {
    assert(false);
    return -1;
  }
}

} // namespace utils
} // namespace mirage
