/*
 * ROCm cutlass-compat: MmaTensorOp stub.
 * Minimal stub for ROCm builds.
 */

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"

namespace cutlass {
namespace arch {
  // Minimal arch op enum stub
  struct OpMultiplyAdd {};
}

namespace gemm {
namespace warp {

// Minimal stub - actual implementation would require ROCm-specific MMA ops
template <
    typename WarpShape_,
    typename InstructionShape_,
    typename ElementA_,
    typename LayoutA_,
    typename ElementB_,
    typename LayoutB_,
    typename ElementC_,
    typename LayoutC_,
    typename Operator_ = arch::OpMultiplyAdd,
    int PartitionsK = 1,
    bool AccumulatorsInRowMajor = false>
struct MmaTensorOp {
  using FragmentA = void*; // Stub
  using FragmentB = void*; // Stub
  using FragmentC = void*; // Stub
  
  struct IteratorA {
    using TensorRef = void*; // Stub
  };
  
  struct IteratorB {
    using TensorRef = void*; // Stub
  };
  
  // Stub operator
  void operator()(FragmentC&, FragmentA&, FragmentB&, FragmentC&) {}
};

} // namespace warp
} // namespace gemm
} // namespace cutlass
