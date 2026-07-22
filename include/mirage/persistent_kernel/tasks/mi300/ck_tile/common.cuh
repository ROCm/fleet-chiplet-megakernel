/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// CK Tile core includes
#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm.hpp"

namespace mirage_ck {

using namespace ck_tile;

// =============================================================================
// Type Aliases
// =============================================================================
using bf16 = ck_tile::bf16_t;
using fp16 = ck_tile::fp16_t;
using fp32 = float;

// =============================================================================
// Warp GEMM Types for MI300 (BF16 input, FP32 accumulator)
// =============================================================================

// 16x16x16 MFMA tile - matches our current implementation
using WarpGemmBf16_16x16x16 = ck_tile::WarpGemmMfmaBf16Bf16F32M16N16K16;

// 32x32x8 MFMA tile - larger tile for better throughput
using WarpGemmBf16_32x32x8 = ck_tile::WarpGemmMfmaBf16Bf16F32M32N32K8;

// 32x32x16 MFMA tile - larger tile with K iteration
using WarpGemmBf16_32x32x16 = ck_tile::WarpGemmMfmaBf16Bf16F32M32N32K16;

// Transposed C distribution variants (useful for attention)
using WarpGemmBf16_16x16x16_CTransposed =
    ck_tile::WarpGemmMfmaBf16Bf16F32M16N16K16TransposedCDistribution;
using WarpGemmBf16_32x32x16_CTransposed =
    ck_tile::WarpGemmMfmaBf16Bf16F32M32N32K16TransposedCDistribution;

} // namespace mirage_ck
