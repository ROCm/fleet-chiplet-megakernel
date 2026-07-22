/*
 * ROCm cutlass-compat: EpilogueSmemAccumulator stub.
 */

#pragma once

#include "cutlass/cutlass.h"

namespace cutlass {
namespace epilogue {
namespace threadblock {

template <
    typename SmemIteratorD_,
    typename FragmentIteratorAccumulator_,
    typename SmemIteratorC_,
    typename OutputOp_>
struct EpilogueSmemAccumulator {
  using SmemIteratorD = SmemIteratorD_;
  using FragmentIteratorAccumulator = FragmentIteratorAccumulator_;
  using SmemIteratorC = SmemIteratorC_;
  using OutputOp = OutputOp_;
};

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass
