/*
 * ROCm cutlass-compat: Coord stub.
 * Provides coordinate types for tensor/matrix indexing.
 */

#pragma once

#include "cutlass/cutlass.h"
#include <cstdint>

namespace cutlass {

template <
  int Rank_,
  typename Index_ = int,
  typename LongIndex_ = int64_t>
struct Coord {
  static int const kRank = Rank_;
  using Index = Index_;
  using LongIndex = LongIndex_;

private:
  Index idx[kRank];

public:
  CUTLASS_HOST_DEVICE
  explicit Coord(Index value = Index(0)) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] = value;
    }
  }

  CUTLASS_HOST_DEVICE
  Coord(Index const (&_idx)[kRank]) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] = _idx[i];
    }
  }

  template <int R, typename I, typename L>
  CUTLASS_HOST_DEVICE
  Coord(Coord<R, I, L> other) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] = other[i];
    }
  }

  CUTLASS_HOST_DEVICE
  Index& at(int i) { return idx[i]; }

  CUTLASS_HOST_DEVICE
  Index const& at(int i) const { return idx[i]; }

  CUTLASS_HOST_DEVICE
  Index& operator[](int i) { return idx[i]; }

  CUTLASS_HOST_DEVICE
  Index const& operator[](int i) const { return idx[i]; }
};

// Helper function to create Coord
template <int Rank, typename Index = int>
CUTLASS_HOST_DEVICE
Coord<Rank, Index> make_Coord(Index const (&values)[Rank]) {
  return Coord<Rank, Index>(values);
}

// Helper for 2-arg case (used by MatrixCoord)
template <typename Index>
CUTLASS_HOST_DEVICE
Coord<2, Index> make_Coord(Index a, Index b) {
  Index values[] = {a, b};
  return Coord<2, Index>(values);
}

// Helper for 3-arg case
template <typename Index>
CUTLASS_HOST_DEVICE
Coord<3, Index> make_Coord(Index a, Index b, Index c) {
  Index values[] = {a, b, c};
  return Coord<3, Index>(values);
}

// Helper for 4-arg case
template <typename Index>
CUTLASS_HOST_DEVICE
Coord<4, Index> make_Coord(Index a, Index b, Index c, Index d) {
  Index values[] = {a, b, c, d};
  return Coord<4, Index>(values);
}

} // namespace cutlass
