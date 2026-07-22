/*
 * ROCm cutlass-compat: DefaultMmaTensorOp stub.
 * This is a minimal stub for ROCm builds. The actual warp-level GEMM
 * operations would need to be implemented using ROCm-specific primitives
 * or rocBLAS. For now, this provides the type structure needed for compilation.
 */

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/warp/mma_tensor_op.h"

namespace cutlass {
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
struct DefaultMmaTensorOp {
  using Type = MmaTensorOp<
      WarpShape_,
      InstructionShape_,
      ElementA_,
      LayoutA_,
      ElementB_,
      LayoutB_,
      ElementC_,
      LayoutC_,
      Operator_,
      PartitionsK,
      AccumulatorsInRowMajor>;
};

} // namespace warp
} // namespace gemm
} // namespace cutlass
