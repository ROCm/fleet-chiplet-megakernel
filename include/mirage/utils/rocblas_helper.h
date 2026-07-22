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
// Device macros (CUTLASS_HOST_DEVICE, etc.) - defined centrally
#include "mirage/utils/device_macros.h"

#ifdef MIRAGE_BACKEND_USE_ROCM

// Ensure HIP platform is defined before including HIP headers
#include "mirage/hip_platform.h"

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>
#include <hipblas/hipblas.h>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <cmath>
#include "mirage/type.h"

// Define device macros for regular C++ compilation (when not using HIP compiler)
// These are needed when compiling Cython-generated code with regular g++
#ifndef __HIP__
  #ifndef __device__
    #define __device__
  #endif
  #ifndef __host__
    #define __host__
  #endif
  // Note: __host__ __device__ is two separate macros, not one
  // HIP headers should define these, but if not, we define them as empty
#endif

namespace mirage {
namespace rocblas_wrapper {

// rocBLAS handle management
class RocblasHandle {
public:
  static rocblas_handle get() {
    static rocblas_handle handle = nullptr;
    if (handle == nullptr) {
      rocblas_create_handle(&handle);
    }
    return handle;
  }
  
  static void destroy() {
    static rocblas_handle handle = get();
    if (handle != nullptr) {
      rocblas_destroy_handle(handle);
      handle = nullptr;
    }
  }
};

// CUTLASS type replacements for ROCm
// Note: __fp16 and __bf16 are only available with HIP/ROCm compilers (hipcc or clang with HIP)
// For regular g++ compilation (like Cython), we use uint16_t as storage
// These will be properly converted when interfacing with rocBLAS APIs

// Check if we're compiling with HIP compiler (hipcc or clang with HIP support)
#if defined(__HIP__) || (defined(__clang__) && defined(__HIP_PLATFORM_AMD__))
  // HIP compiler: use native half types
  using half_t = __fp16;
  using bfloat16_t = __bf16;
#else
  // Regular g++ or other compilers: use uint16_t as storage type
  // Note: This is fine for host code compilation - device code will use proper types when compiled with HIP
  // The uint16_t is just a placeholder for type system compatibility
  using half_t = uint16_t;
  using bfloat16_t = uint16_t;
#endif

// Array type replacement (simple fixed-size array)
template <typename T, int N>
struct Array {
  T data[N];
  
  Array() {
    #if defined(__HIP__) || (defined(__clang__) && defined(__HIP_PLATFORM_AMD__))
    #pragma unroll
    #endif
    for (int i = 0; i < N; i++) {
      data[i] = T(0);
    }
  }
  
  Array(const T& val) {
    #if defined(__HIP__) || (defined(__clang__) && defined(__HIP_PLATFORM_AMD__))
    #pragma unroll
    #endif
    for (int i = 0; i < N; i++) {
      data[i] = val;
    }
  }
  
  T& operator[](int i) { return data[i]; }
  const T& operator[](int i) const { return data[i]; }
  
  Array& operator=(const Array& other) {
    #if defined(__HIP__) || (defined(__clang__) && defined(__HIP_PLATFORM_AMD__))
    #pragma unroll
    #endif
    for (int i = 0; i < N; i++) {
      data[i] = other.data[i];
    }
    return *this;
  }
  
  bool operator==(const Array& other) const {
    #if defined(__HIP__) || (defined(__clang__) && defined(__HIP_PLATFORM_AMD__))
    #pragma unroll
    #endif
    for (int i = 0; i < N; i++) {
      if (data[i] != other.data[i]) return false;
    }
    return true;
  }
};

// Layout types (simplified)
namespace layout {
  struct RowMajor {};
  struct ColumnMajor {};
}

// rocBLAS data type conversion using template specialization
// Note: __fp16 and __bf16 cannot be used as function parameters or template parameters
// So we use a traits-based approach with wrapper types
template <typename T>
struct rocblas_type_traits {
  static constexpr rocblas_datatype value = rocblas_datatype_f32_r; // default
};

template <>
struct rocblas_type_traits<float> {
  static constexpr rocblas_datatype value = rocblas_datatype_f32_r;
};

template <>
struct rocblas_type_traits<double> {
  static constexpr rocblas_datatype value = rocblas_datatype_f64_r;
};

template <>
struct rocblas_type_traits<int8_t> {
  static constexpr rocblas_datatype value = rocblas_datatype_i8_r;
};

template <>
struct rocblas_type_traits<int32_t> {
  static constexpr rocblas_datatype value = rocblas_datatype_i32_r;
};

// Helper function to get rocblas_datatype from a type
// Note: __fp16 and __bf16 cannot be used as template parameters, so we use a constexpr function
// that checks type properties at compile time
template<typename T>
constexpr rocblas_datatype get_rocblas_type_impl() {
  if constexpr (std::is_same_v<T, float>) {
    return rocblas_datatype_f32_r;
  } else if constexpr (std::is_same_v<T, double>) {
    return rocblas_datatype_f64_r;
  } else if constexpr (std::is_same_v<T, int8_t>) {
    return rocblas_datatype_i8_r;
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return rocblas_datatype_i32_r;
  } else if constexpr (sizeof(T) == 2) {
    // For 16-bit types (could be __fp16, __bf16, or uint16_t fallback)
    // Check if it's actually a floating point type or our uint16_t fallback
    if constexpr (std::is_same_v<T, uint16_t>) {
      // This is our fallback type for half_t/bfloat16_t when not using HIP compiler
      // Treat it as half precision
      return rocblas_datatype_f16_r;
    } else if constexpr (std::is_floating_point_v<T>) {
      // Native __fp16 or __bf16
      return rocblas_datatype_f16_r;
    } else {
      return rocblas_datatype_f32_r; // default
    }
  } else {
    return rocblas_datatype_f32_r; // default
  }
}

// rocBLAS GEMM wrapper (replaces CUTLASS GEMM)
template <typename ElementA, typename ElementB, typename ElementC, typename ElementD>
rocblas_status gemm(
    rocblas_handle handle,
    rocblas_operation transA,
    rocblas_operation transB,
    int m, int n, int k,
    const void* alpha,
    const ElementA* A, int lda,
    const ElementB* B, int ldb,
    const void* beta,
    const ElementC* C, int ldc,
    ElementD* D, int ldd,
    hipStream_t stream = nullptr) {
  
  if (stream != nullptr) {
    rocblas_set_stream(handle, stream);
  }
  
  rocblas_datatype type_a = get_rocblas_type_impl<ElementA>();
  rocblas_datatype type_b = get_rocblas_type_impl<ElementB>();
  rocblas_datatype type_c = get_rocblas_type_impl<ElementC>();
  rocblas_datatype type_d = get_rocblas_type_impl<ElementD>();
  
  // Use rocblas_gemm_ex for mixed precision or rocblas_gemm for same precision
  // For simplicity, always use rocblas_gemm_ex which supports all types
  return rocblas_gemm_ex(handle, transA, transB, m, n, k,
                        alpha, A, type_a, lda,
                        B, type_b, ldb,
                        beta, C, type_c, ldc,
                        D, type_d, ldd,
                        rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
                        0, 0);
}

// Fast exp operation replacement (simple implementation)
template <typename T>
struct fast_exp_op {
  T operator()(const T& x) const {
    // Simple fast exp approximation
    // For production, use a better approximation or library function
    if constexpr (std::is_same_v<T, float>) {
      return expf(x);
    } else if constexpr (std::is_same_v<T, double>) {
      return exp(x);
    } else {
      return static_cast<T>(expf(static_cast<float>(x)));
    }
  }
};

template <typename T, int N>
struct fast_exp_op<Array<T, N>> {
  Array<T, N> operator()(const Array<T, N>& x) const {
    Array<T, N> result;
    fast_exp_op<T> exp_op;
    #if defined(__HIP__) || (defined(__clang__) && defined(__HIP_PLATFORM_AMD__))
    #pragma unroll
    #endif
    for (int i = 0; i < N; i++) {
      result[i] = exp_op(x[i]);
    }
    return result;
  }
};

} // namespace rocblas_wrapper
} // namespace mirage

// Define CUTLASS namespace aliases for compatibility
namespace cutlass {
  using half_t = mirage::rocblas_wrapper::half_t;
  using bfloat16_t = mirage::rocblas_wrapper::bfloat16_t;
  
  template <typename T, int N>
  using Array = mirage::rocblas_wrapper::Array<T, N>;
  
  namespace layout {
    using RowMajor = mirage::rocblas_wrapper::layout::RowMajor;
    using ColumnMajor = mirage::rocblas_wrapper::layout::ColumnMajor;
  }
  
  template <typename T>
  using fast_exp_op = mirage::rocblas_wrapper::fast_exp_op<T>;
}

// Activation functions for ROCm (similar to CUDA version)
namespace mirage {
namespace utils {

template <mirage::type::ActivationType act_type, int N>
struct act_function {
  cutlass::Array<cutlass::half_t, N>
      operator()(cutlass::Array<cutlass::half_t, N> const &rhs) const {
    // default do nothing
    return rhs;
  }
};

template <int N>
struct act_function<mirage::type::ActivationType::ACT_EXP, N> {
  using func = cutlass::fast_exp_op<cutlass::Array<cutlass::half_t, N>>;
  func exp;

  cutlass::Array<cutlass::half_t, N>
      operator()(cutlass::Array<cutlass::half_t, N> const &rhs) const {
    return exp(rhs);
  }
};

} // namespace utils
} // namespace mirage

#endif // MIRAGE_BACKEND_USE_ROCM
