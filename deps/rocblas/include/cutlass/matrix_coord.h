/*
 * ROCm cutlass-compat: MatrixCoord stub.
 * Provides matrix coordinate (row, column) for rank-2 matrices.
 */

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/coord.h"

namespace cutlass {

struct MatrixCoord : public Coord<2, int> {
public:
  using Index = int;
  using Base = Coord<2, Index>;
  using LongIndex = typename Base::LongIndex;

private:
  static int const kRow = 0;
  static int const kColumn = 1;

public:
  CUTLASS_HOST_DEVICE
  MatrixCoord() { }

  CUTLASS_HOST_DEVICE
  MatrixCoord(Coord<2, Index> const &coord): Base(coord) { }

  CUTLASS_HOST_DEVICE
  MatrixCoord(Index row, Index column): Base(make_Coord(row, column)) { }

  CUTLASS_HOST_DEVICE
  MatrixCoord(LongIndex row, LongIndex column): Base(make_Coord(Index(row), Index(column))) { }

  CUTLASS_HOST_DEVICE
  Index const & row() const { return this->at(kRow); }

  CUTLASS_HOST_DEVICE
  Index & row() { return this->at(kRow); }

  CUTLASS_HOST_DEVICE
  Index const & column() const { return this->at(kColumn); }

  CUTLASS_HOST_DEVICE
  Index & column() { return this->at(kColumn); }
};

} // namespace cutlass
