/*
 * ROCm/HIP compatibility layer for CUTLASS fast_math.
 * Minimal stub when building with MIRAGE_BACKEND_USE_ROCM.
 * Mirage fingerprint code uses cutlass mainly for CUTLASS_DEVICE and
 * cutlass namespace; fast_math is included but most usage is via
 * mirage fingerprint helpers.
 */

#pragma once

#include "cutlass/cutlass.h"

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace cutlass {

template <typename T>
CUTLASS_HOST_DEVICE void swap(T& a, T& b) {
  T t = a;
  a = b;
  b = t;
}

template <int N>
struct is_pow2 {
  static bool const value = ((N & (N - 1)) == 0);
};

template <int N, int CurrentVal = N, int Count = 0>
struct log2_down {
  enum { value = log2_down<N, (CurrentVal >> 1), Count + 1>::value };
};
template <int N, int Count>
struct log2_down<N, 1, Count> {
  enum { value = Count };
};

template <int N, int CurrentVal = N, int Count = 0>
struct log2_up {
  enum { value = log2_up<N, (CurrentVal >> 1), Count + 1>::value };
};
template <int N, int Count>
struct log2_up<N, 1, Count> {
  enum { value = Count };
};

} // namespace cutlass
