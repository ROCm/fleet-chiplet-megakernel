/*
 * ROCm cutlass-compat: FragmentIteratorTensorOp stub.
 */

#pragma once

#include "cutlass/cutlass.h"

namespace cutlass {
namespace epilogue {
namespace warp {

template <
    typename WarpShape_,
    typename InstructionShape_,
    typename Element_,
    typename Fragment_,
    typename Layout_>
struct FragmentIteratorTensorOp {
  using Fragment = Fragment_;
};

} // namespace warp
} // namespace epilogue
} // namespace cutlass
