/*
 * ROCm cutlass-compat: LinearCombination stub.
 */

#pragma once

#include "cutlass/cutlass.h"

namespace cutlass {
namespace epilogue {
namespace thread {

enum class ScaleType {
  Nothing,
  OnlyAlphaScaling,
  Full
};

template <
    typename ElementOutput_,
    int Count,
    typename ElementAccumulator_,
    typename ElementCompute_,
    ScaleType Scale = ScaleType::Nothing>
struct LinearCombination {
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
};

} // namespace thread
} // namespace epilogue
} // namespace cutlass
