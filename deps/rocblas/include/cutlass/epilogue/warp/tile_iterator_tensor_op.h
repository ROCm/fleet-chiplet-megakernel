/*
 * ROCm cutlass-compat: TileIteratorTensorOp stub.
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
    typename Layout_>
struct TileIteratorTensorOp {
  using Element = Element_;
  using TensorRef = void*; // Stub
};

template <
    typename WarpShape_,
    typename InstructionShape_,
    typename Element_,
    typename Layout_>
struct TileIteratorTensorOpCanonical {
  using Element = Element_;
  using TensorRef = void*; // Stub
};

} // namespace warp
} // namespace epilogue
} // namespace cutlass
