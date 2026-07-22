/*
 * ROCm/HIP compatibility layer for CUTLASS.
 * When building with MIRAGE_BACKEND_USE_ROCM, this header is used instead of
 * NVIDIA CUTLASS. It provides minimal cutlass:: types and macros so that
 * mirage code (e.g. threadblock/cuda/, kernel/cuda/) can compile with HIP.
 *
 * CUTLASS_HOST_DEVICE, CUTLASS_DEVICE, etc. are defined in
 * mirage/utils/device_macros.h when using ROCm.
 */

#pragma once

#include "mirage/config.h"
#include "mirage/utils/device_macros.h"

#include <cstddef>
#include <cstdint>
#include <cassert>

// CUTLASS_UNUSED: suppress unused variable warnings
template <typename T>
CUTLASS_HOST_DEVICE void __rocblas_cutlass_unused(T const&) {}
#if defined(__GNUC__) || defined(__clang__)
#define CUTLASS_UNUSED(expr) __rocblas_cutlass_unused(expr)
#else
#define CUTLASS_UNUSED(expr) ((void)(expr))
#endif

// CUTLASS_NOT_IMPLEMENTED: stub for unimplemented paths (e.g. in functional.h)
#define CUTLASS_NOT_IMPLEMENTED() assert(0 && __PRETTY_FUNCTION__)

// Define CUTLASS_CONSTEXPR_IF_CXX17 macro (needed by CUTLASS headers)
// This should match the definition in cutlass/detail/helper_macros.hpp
#if (201703L <= __cplusplus)
#define CUTLASS_CONSTEXPR_IF_CXX17 constexpr
#define CUTLASS_CXX17_OR_LATER 1
#else
#define CUTLASS_CONSTEXPR_IF_CXX17
#define CUTLASS_CXX17_OR_LATER 0
#endif

// CUTLASS_CMATH_NAMESPACE: host uses std::, device uses global ::
#ifndef CUTLASS_CMATH_NAMESPACE
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#define CUTLASS_CMATH_NAMESPACE
#else
#define CUTLASS_CMATH_NAMESPACE std
#endif
#endif

// CUTLASS_PRAGMA_UNROLL: helper_macros.hpp only handles CUDA, so define for HIP here
#ifndef CUTLASS_PRAGMA_UNROLL
#if defined(__HIP__) || (defined(__clang__) && defined(__HIP_PLATFORM_AMD__))
#define CUTLASS_PRAGMA_UNROLL _Pragma("unroll")
#define CUTLASS_PRAGMA_NO_UNROLL _Pragma("unroll 1")
#else
// Will be defined by helper_macros.hpp for CUDA
#endif
#endif

// GEMM loop macro (stub for ROCm)
#ifndef CUTLASS_GEMM_LOOP
#define CUTLASS_GEMM_LOOP
#endif

namespace cutlass {

/// Status code (minimal subset for compat)
enum class Status {
  kSuccess = 0,
  kErrorMisalignedOperand,
  kErrorInvalidDataType,
  kErrorInvalidLayout,
  kErrorInvalidProblem,
  kErrorNotSupported,
  kErrorWorkspaceNull,
  kErrorInternal,
  kErrorArchMismatch,
  kErrorInsufficientDriver,
  kErrorMemoryAllocation,
  kInvalid
};

CUTLASS_HOST_DEVICE
inline char const* cutlassGetStatusString(Status status) {
  switch (status) {
    case Status::kSuccess: return "Success";
    case Status::kErrorMisalignedOperand: return "Error Misaligned Operand";
    case Status::kErrorInvalidDataType: return "Error Invalid Data Type";
    case Status::kErrorInvalidLayout: return "Error Invalid Layout";
    case Status::kErrorInvalidProblem: return "Error Invalid Problem";
    case Status::kErrorNotSupported: return "Error Not Supported";
    case Status::kErrorWorkspaceNull: return "Error Workspace Null";
    case Status::kErrorInternal: return "Error Internal";
    case Status::kErrorArchMismatch: return "Error Arch Mismatch";
    case Status::kErrorInsufficientDriver: return "Error Insufficient Driver";
    case Status::kErrorMemoryAllocation: return "Error Memory Allocation";
    default: return "Invalid";
  }
}

/// FastDivmod - HIP-compatible stub for fast division/modulo operations
/// This is a minimal implementation for compilation compatibility
struct FastDivmod {
  using value_div_type = int;
  using value_mod_type = int64_t;
  int32_t divisor = 1;
  uint32_t multiplier = 0u;
  uint32_t shift_right = 0u;

  CUTLASS_HOST_DEVICE
  FastDivmod() = default;
  
  CUTLASS_HOST_DEVICE
  FastDivmod(int32_t d) : divisor(d) {
    // Simplified initialization - full implementation would compute multiplier/shift
    // For now, just store the divisor
  }

  // Find quotient and remainder - HIP-compatible implementation
  // Supports both fast_divmod method and operator() for cute compatibility
  CUTLASS_HOST_DEVICE
  void fast_divmod(int& quotient, int& remainder, int dividend) const {
    if (divisor != 1) {
      // Standard division for HIP (no __umulhi equivalent needed for basic case)
      quotient = dividend / divisor;
    } else {
      quotient = dividend;
    }
    remainder = dividend - (quotient * divisor);
  }

  // Operator() for cute compatibility (takes int64_t remainder)
  CUTLASS_HOST_DEVICE
  void operator()(int& quotient, int64_t& remainder, int dividend) const {
    int rem_int = 0;
    fast_divmod(quotient, rem_int, dividend);
    remainder = static_cast<int64_t>(rem_int);
  }
};

} // namespace cutlass
