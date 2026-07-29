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

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1.hpp"
#include "ck_tile/ops/gemm/block/block_universal_gemm_as_bs_cr.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v2.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_traits.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"
#include "tasks/common/common_header.cuh"

namespace kernel {

// ============================================================================
// Non-temporal store helpers for MALL cache optimization (NT=1 → MALL:
// no-allocate) Transient output data should not pollute MALL; weights should
// stay resident. On GFX942/GFX950: NT bit controls MALL allocation policy:
//   NT=0 (default): TCP/TCC=LRU, MALL=allocate
//   NT=1 (non-temporal): TCP/TCC=evict/stream, MALL=no-allocate
// ============================================================================
__device__ __forceinline__ void nt_store_u64(void *addr, uint64_t val) {
  *reinterpret_cast<uint64_t *>(addr) = val;
}

__device__ __forceinline__ void nt_store_bf16(ck_tile::bf16_t *addr,
                                              ck_tile::bf16_t val) {
  *addr = val;
}

using bf16 = ck_tile::bf16_t;

// Promote a device pointer to wave-uniform (SGPR) representation.
// On AMDGPU, buffer_load requires the base address in an SGPR buffer resource
// descriptor. When the compiler cannot prove a pointer is uniform across the
// wave, it emits a v_readfirstlane "waterfall" loop for every buffer_load —
// adding ~40% overhead to the GEMM K-loop. In the persistent kernel all threads
// in a workgroup execute the same task, so pointers are identical across lanes;
// this intrinsic makes that explicit.
__device__ __forceinline__ uintptr_t __uniform_addr(void const *ptr) {
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  uint32_t lo = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(addr));
  uint32_t hi =
      __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(addr >> 32));
  return static_cast<uintptr_t>(hi) << 32 | static_cast<uintptr_t>(lo);
}

// ============================================================================
// Custom block gemm policy for small batch sizes using 16x16x16 MFMA
// Uses MWarp=1, NWarp=4 configuration to allow MPerBlock=16, NPerBlock=64
// This minimizes padding overhead for small batches (batch_size=8 -> 2x
// padding)
// ============================================================================
struct BlockGemmSmallM16Policy {
  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp() {
    // Use 16x16x16 MFMA with transposed C distribution
    // MWarp=1 allows MPerBlock=16 (minimal padding for batch=8)
    // NWarp=4 to maintain parallelism across 4 warps (128 threads)
    return ck_tile::make_tuple(
        ck_tile::WarpGemmMfmaBf16Bf16F32M16N16K16TransposedCDistribution{},
        1, // MWarp = 1 -> MPerBlock >= 16
        4  // NWarp = 4 -> NPerBlock >= 64
    );
  }
};

// Default block gemm policy using 32x32x16 MFMA (same as CK default)
// MWarp=4, NWarp=1 -> MPerBlock=128, NPerBlock=32
struct BlockGemmDefaultPolicy {
  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp() {
    return ck_tile::make_tuple(
        ck_tile::WarpGemmMfmaBf16Bf16F32M32N32K16TransposedCDistribution<>{},
        4, // MWarp
        1  // NWarp
    );
  }
};

// 2x2 warp grid policy using 32x32x16 MFMA
// MWarp=2, NWarp=2 -> MPerBlock=64, NPerBlock=64 (or 64x128 with NIter=2)
struct BlockGemm2x2Policy {
  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp() {
    return ck_tile::make_tuple(
        ck_tile::WarpGemmMfmaBf16Bf16F32M32N32K16TransposedCDistribution<>{},
        2, // MWarp = 2 -> M = 2 * 32 * MIter
        2  // NWarp = 2 -> N = 2 * 32 * NIter
    );
  }
};

// Large tile policy for bigger batch sizes using 32x32x16 MFMA
// MWarp=1, NWarp=4 -> MPerBlock=32, NPerBlock=128
struct BlockGemmLargeM32Policy {
  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp() {
    return ck_tile::make_tuple(
        ck_tile::WarpGemmMfmaBf16Bf16F32M32N32K16TransposedCDistribution<>{},
        1, // MWarp = 1 -> MPerBlock = 32
        4  // NWarp = 4 -> NPerBlock = 128
    );
  }
};

// ============================================================================
// Custom LDS descriptor for small M tiles (MPerBlock=16)
// Adapted from CK's default policy with smaller tile support
// ============================================================================
template <ck_tile::index_t MPerBlock_,
          ck_tile::index_t NPerBlock_,
          ck_tile::index_t KPerBlock_>
struct GemmPipelineSmallTilePolicy {
  static constexpr ck_tile::index_t MPerBlock = MPerBlock_;
  static constexpr ck_tile::index_t NPerBlock = NPerBlock_;
  static constexpr ck_tile::index_t KPerBlock = KPerBlock_;

  // LDS descriptor for A matrix [M, K] with padding to avoid bank conflicts
  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor() {
    using namespace ck_tile;
    // 3D layout with padding: [K/8, M, 8] with stride (M+1)*8 on K/8 dimension
    constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
        make_tuple(number<KPerBlock / 8>{}, number<MPerBlock>{}, number<8>{}),
        make_tuple(number<(MPerBlock + 1) * 8>{}, number<8>{}, number<1>{}),
        number<8>{},
        number<1>{});

    constexpr auto a_lds_block_desc = transform_tensor_descriptor(
        a_lds_block_desc_0,
        make_tuple(make_pass_through_transform(MPerBlock),
                   make_merge_transform(make_tuple(KPerBlock / 8, 8))),
        make_tuple(sequence<1>{}, sequence<0, 2>{}),
        make_tuple(sequence<0>{}, sequence<1>{}));

    return a_lds_block_desc;
  }

  // LDS descriptor for B matrix [N, K]
  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor() {
    using namespace ck_tile;
    constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
        make_tuple(number<KPerBlock / 8>{}, number<NPerBlock>{}, number<8>{}),
        make_tuple(number<(NPerBlock + 1) * 8>{}, number<8>{}, number<1>{}),
        number<8>{},
        number<1>{});

    constexpr auto b_lds_block_desc = transform_tensor_descriptor(
        b_lds_block_desc_0,
        make_tuple(make_pass_through_transform(NPerBlock),
                   make_merge_transform(make_tuple(KPerBlock / 8, 8))),
        make_tuple(sequence<1>{}, sequence<0, 2>{}),
        make_tuple(sequence<0>{}, sequence<1>{}));

    return b_lds_block_desc;
  }

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeA() {
    return sizeof(typename Problem::ADataType) *
           MakeALdsBlockDescriptor<Problem>().get_element_space_size();
  }

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeB() {
    return sizeof(typename Problem::BDataType) *
           MakeBLdsBlockDescriptor<Problem>().get_element_space_size();
  }

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize() {
    return GetSmemSizeA<Problem>() + GetSmemSizeB<Problem>();
  }

  // DRAM tile distribution for A [M, K] - distributes loads across threads
  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto MakeADramTileDistribution() {
    using namespace ck_tile;
    using ADataType = remove_cvref_t<typename Problem::ADataType>;

    constexpr index_t kBlockSize = Problem::kBlockSize;
    constexpr index_t K1 = 16 / sizeof(ADataType); // Vector load size (8 bf16)
    constexpr index_t K0 = KPerBlock / K1;
    constexpr index_t M2 = get_warp_size() / K0;
    constexpr index_t M1 = kBlockSize / get_warp_size();
    constexpr index_t M0 = MPerBlock / (M2 * M1);

    // Handle case where MPerBlock is smaller than thread distribution
    if constexpr (M0 == 0) {
      // For very small M, all threads cooperate on same M rows
      return make_static_tile_distribution(
          tile_distribution_encoding<
              sequence<1>,
              tuple<sequence<1, M1, M2>, sequence<K0, K1>>,
              tuple<sequence<1>, sequence<1, 2>>,
              tuple<sequence<1>, sequence<2, 0>>,
              sequence<1, 2>,
              sequence<0, 1>>{});
    } else {
      return make_static_tile_distribution(
          tile_distribution_encoding<
              sequence<1>,
              tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
              tuple<sequence<1>, sequence<1, 2>>,
              tuple<sequence<1>, sequence<2, 0>>,
              sequence<1, 2>,
              sequence<0, 1>>{});
    }
  }

  // DRAM tile distribution for B [N, K]
  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto MakeBDramTileDistribution() {
    using namespace ck_tile;
    using BDataType = remove_cvref_t<typename Problem::BDataType>;

    constexpr index_t kBlockSize = Problem::kBlockSize;
    constexpr index_t K1 = 16 / sizeof(BDataType);
    constexpr index_t K0 = KPerBlock / K1;
    constexpr index_t N2 = get_warp_size() / K0;
    constexpr index_t N1 = kBlockSize / get_warp_size();
    constexpr index_t N0 = NPerBlock / (N2 * N1);

    return make_static_tile_distribution(
        tile_distribution_encoding<
            sequence<1>,
            tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
            tuple<sequence<1>, sequence<1, 2>>,
            tuple<sequence<1>, sequence<2, 0>>,
            sequence<1, 2>,
            sequence<0, 1>>{});
  }

  // Use small-batch optimized block gemm policy
  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm() {
    if constexpr (MPerBlock == 16) {
      // Use 16x16 MFMA for small M tiles
      return ck_tile::BlockUniversalGemmAsBsCr<Problem,
                                               BlockGemmSmallM16Policy>{};
    } else if constexpr (MPerBlock == 64) {
      // Use 32x32 MFMA with 2x2 warp grid for full-batch M=64 tiles
      return ck_tile::BlockUniversalGemmAsBsCr<Problem, BlockGemm2x2Policy>{};
    } else if constexpr (MPerBlock == 32 && NPerBlock == 128) {
      // Use 32x32 MFMA for larger batch sizes
      return ck_tile::BlockUniversalGemmAsBsCr<Problem,
                                               BlockGemmLargeM32Policy>{};
    } else {
      // Use default 32x32 MFMA for other configurations
      return ck_tile::BlockUniversalGemmAsBsCr<Problem,
                                               BlockGemmDefaultPolicy>{};
    }
  }
};

// ============================================================================
// CK Pipeline-based Linear Kernel
// Tile size adapts to batch size:
//   BATCH_SIZE <= 16:  16x64x256 tiles with 16x16x16 MFMA (minimal padding)
//   16 < BATCH_SIZE <= 64: 32x128x128 tiles with 32x32x16 MFMA
//   BATCH_SIZE > 64:  128x128x64 tiles with 32x32x16 MFMA, MWarp=2 NWarp=2
//                     (MIterPerWarp=2, NIterPerWarp=2 — matches hipBLASLt
//                     strategy)
// ============================================================================
template <typename T,
          int BATCH_SIZE,
          int REDUCTION_SIZE,
          bool FORCE_SMALL_TILE = false>
__device__ __forceinline__ void
    linear_kernel_ck(void const *input_ptr,
                     void const *weight_ptr,
                     void const *residual_ptr,
                     void *output_ptr,
                     int num_active_tokens,
                     bool residual_add,
                     int output_size,
                     int o_stride,
                     void const *bias_ptr = nullptr) {
#ifdef MPK_DISABLE_LINEAR
  return;
#endif
  using namespace ck_tile;

  // Four-tier tile selection:
  //   Tier 0 (small):  16x64x256,  16x16 MFMA, MWarp=1 NWarp=4 (bs<=16)
  //   Tier 1 (medium): 64x64x128,  32x32 MFMA, MWarp=2 NWarp=2 (17<=bs<=64)
  //   Tier 2 (large):  128x128x64, 32x32 MFMA, MWarp=2 NWarp=2 (bs>64)
  constexpr bool use_xlarge_tile = !FORCE_SMALL_TILE && (BATCH_SIZE > 64);
  constexpr bool use_medium_tile =
      !FORCE_SMALL_TILE && (BATCH_SIZE > 16) && !use_xlarge_tile;

  constexpr index_t MPerBlock =
      use_xlarge_tile ? 128 : (use_medium_tile ? 64 : 16);
  constexpr index_t NPerBlock =
      use_xlarge_tile ? 128 : (use_medium_tile ? 64 : 64);
  constexpr index_t KPerBlock =
      use_xlarge_tile ? 64 : (use_medium_tile ? 128 : 256);

  constexpr index_t LoopM = (BATCH_SIZE + MPerBlock - 1) / MPerBlock;
  index_t LoopN = (output_size + NPerBlock - 1) / NPerBlock;
  constexpr index_t NumLoopK = REDUCTION_SIZE / KPerBlock;

  // Warp grid: 2x2 for 64x64 and 128x128, 1x4 for 16x64
  constexpr index_t MWarp = (use_xlarge_tile || use_medium_tile) ? 2 : 1;
  constexpr index_t NWarp = (use_xlarge_tile || use_medium_tile) ? 2 : 4;

  using BlockTile = sequence<MPerBlock, NPerBlock, KPerBlock>;
  using BlockWarps = sequence<MWarp, NWarp>;
  // WarpTile = MFMA dimensions (32x32x16 or 16x16x16), NOT the iterated tile.
  // MIterPerWarp/NIterPerWarp are computed automatically by
  // BlockUniversalGemmAsBsCr.
  constexpr index_t WarpM = (use_xlarge_tile || use_medium_tile) ? 32 : 16;
  constexpr index_t WarpN = (use_xlarge_tile || use_medium_tile) ? 32 : 16;
  constexpr index_t WarpK = 16;
  using WarpTile = sequence<WarpM, WarpN, WarpK>;

  using GemmShape = TileGemmShape<BlockTile, BlockWarps, WarpTile>;

  // Define traits — 128x128 uses TransposeC=true for CK default policy
  // compatibility
  using GemmTraits =
      TileGemmUniversalTraits<true,  // kPadM
                              false, // kPadN
                              true,  // kPadK
                              false, // DoubleSmemBuffer
                              tensor_layout::gemm::RowMajor,
                              tensor_layout::gemm::ColumnMajor,
                              tensor_layout::gemm::RowMajor,
                              use_xlarge_tile // TransposeC: true for 128x128
                                              // (default policy), false for
                                              // smaller
                              >;

  using Problem = GemmPipelineProblem<bf16, bf16, float, GemmShape, GemmTraits>;

  // For 128x128 tiles, use CK default policy (handles MIterPerWarp=2,
  // NIterPerWarp=2). For smaller tiles, use custom policy.
  using PipelinePolicy = std::conditional_t<
      use_xlarge_tile,
      GemmPipelineAGmemBGmemCRegV1DefaultPolicy,
      GemmPipelineSmallTilePolicy<MPerBlock, NPerBlock, KPerBlock>>;
  using Pipeline = GemmPipelineAGmemBGmemCRegV2<Problem, PipelinePolicy>;

  // Promote pointers and runtime dims to wave-uniform (SGPR) to eliminate
  // v_readfirstlane waterfall sequences in CK's buffer_load instructions.
  bf16 const *d_input =
      reinterpret_cast<bf16 const *>(__uniform_addr(input_ptr));
  bf16 const *d_weight =
      reinterpret_cast<bf16 const *>(__uniform_addr(weight_ptr));
  bf16 const *d_residual =
      reinterpret_cast<bf16 const *>(__uniform_addr(residual_ptr));
  bf16 const *d_bias =
      bias_ptr ? reinterpret_cast<bf16 const *>(__uniform_addr(bias_ptr))
               : nullptr;
  bf16 *d_output = reinterpret_cast<bf16 *>(__uniform_addr(output_ptr));
  output_size = __builtin_amdgcn_readfirstlane(output_size);
  o_stride = __builtin_amdgcn_readfirstlane(o_stride);

  extern __shared__ char smem[];

  // Iterate over M tiles
  for (index_t mm = 0; mm < LoopM; mm++) {
    index_t m_offset = mm * MPerBlock;
    index_t m_size = (mm == LoopM - 1) ? (BATCH_SIZE - m_offset) : MPerBlock;
    if (m_size <= 0) {
      continue;
    }

    // Iterate over N tiles
    for (index_t nn = 0; nn < LoopN; nn++) {
      index_t n_offset = nn * NPerBlock;
      index_t n_size = (nn == LoopN - 1) ? (output_size - n_offset) : NPerBlock;
      if (n_size <= 0) {
        continue;
      }

      // Create tensor views for A (input) - offset by m_offset
      auto a_tensor_view = make_naive_tensor_view<address_space_enum::global>(
          d_input + m_offset * REDUCTION_SIZE,
          make_tuple(m_size, index_t(REDUCTION_SIZE)),
          make_tuple(index_t(REDUCTION_SIZE), index_t(1)),
          number<8>{},
          number<1>{});

      // B tensor view - weight matrix offset by n_offset
      // Use non-temporal loads for weights: they're streamed once per iteration
      // and never reused at BS=1. NT loads bypass L2, keeping it free for
      // KV cache and activations that DO benefit from L2 reuse.
#ifdef MPK_NT_WEIGHT_LOADS
      // Cache-stream (.cs) for weight loads on gfx950:
      // DEVICE_NT1 = sc1=1, nt=1 = value 18
      // L1: MISS_EVICT, L2: Cache_Stream (allocate, use, discard)
      // Weights get short-lived L2 residency for M-tile sharing,
      // then are immediately evictable to protect KV cache.
      auto b_tensor_view =
          make_naive_tensor_view<address_space_enum::global,
                                 memory_operation_enum::set,
                                 static_cast<amd_buffer_coherence_enum>(18)>(
#else
      auto b_tensor_view = make_naive_tensor_view<address_space_enum::global>(
#endif
              d_weight + n_offset * REDUCTION_SIZE,
              make_tuple(n_size, index_t(REDUCTION_SIZE)),
              make_tuple(index_t(REDUCTION_SIZE), index_t(1)),
              number<8>{},
              number<1>{});

      // Create tile windows
      auto a_tile_window =
          make_tile_window(a_tensor_view,
                           make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                           {0, 0});

      auto b_tile_window =
          make_tile_window(b_tensor_view,
                           make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                           {0, 0});

      // Run CK pipeline GEMM
      Pipeline pipeline;

      auto c_block_tile =
          pipeline(a_tile_window, b_tile_window, NumLoopK, smem);

      block_sync_lds();

      // Epilogue: write GEMM results to global memory
      // Use CK store_tile for 128x128 (complex C distribution with
      // MIter/NIter), hand-written epilogues for smaller tiles (known register
      // layout).
      if constexpr (use_xlarge_tile) {
        // 128x128 tiles: 32x32 MFMA TransposedCDistribution
        // MWarp=2, NWarp=2 with MIterPerWarp=2, NIterPerWarp=2
        // Each thread has 64 float32 registers (4 MFMA tiles × 16 regs each).
        //
        // Register layout per MFMA tile (32x32 TransposedC):
        //   Row within tile: lane_id % 32
        //   Col within tile: (r/4)*8 + (lane_id/32)*4 + (r%4), r=0..15
        //
        // Global position:
        //   warp_m = warp_id / 2, warp_n = warp_id % 2
        //   For (mIter, nIter) in {0,1}×{0,1}:
        //     M_base = warp_m * 64 + mIter * 32
        //     N_base = warp_n * 64 + nIter * 32
        //     reg_offset = (mIter * 2 + nIter) * 16
        index_t warp_id = threadIdx.x >> 6;
        index_t lane_id = threadIdx.x & 63;
        index_t tile_row_in_mfma = lane_id & 31;
        index_t m_lane = lane_id >> 5;
        index_t warp_m = warp_id >> 1;
        index_t warp_n = warp_id & 1;

        auto &c_buf = c_block_tile.get_thread_buffer();

#pragma unroll
        for (index_t mIter = 0; mIter < 2; mIter++) {
          index_t base_m =
              m_offset + warp_m * 64 + mIter * 32 + tile_row_in_mfma;
          if (base_m >= BATCH_SIZE) {
            continue;
          }

#pragma unroll
          for (index_t nIter = 0; nIter < 2; nIter++) {
            index_t r_offset = (mIter * 2 + nIter) * 16;
            index_t base_n = n_offset + warp_n * 64 + nIter * 32;

#pragma unroll
            for (index_t g = 0; g < 4; g++) {
              index_t col_in_mfma = g * 8 + m_lane * 4;
              index_t global_n_base = base_n + col_in_mfma;
              index_t r_base = r_offset + g * 4;

              if (global_n_base + 3 < output_size) {
                index_t out_base = base_m * o_stride + global_n_base;
                float v0 = c_buf[r_base], v1 = c_buf[r_base + 1],
                      v2 = c_buf[r_base + 2], v3 = c_buf[r_base + 3];
                if (residual_add && d_residual != nullptr) {
                  uint64_t res_packed = *reinterpret_cast<uint64_t const *>(
                      &d_residual[out_base]);
                  bf16 const *res = reinterpret_cast<bf16 const *>(&res_packed);
                  v0 += type_convert<float>(res[0]);
                  v1 += type_convert<float>(res[1]);
                  v2 += type_convert<float>(res[2]);
                  v3 += type_convert<float>(res[3]);
                }
                if (d_bias) {
                  uint64_t bias_packed = *reinterpret_cast<uint64_t const *>(
                      &d_bias[global_n_base]);
                  bf16 const *bv = reinterpret_cast<bf16 const *>(&bias_packed);
                  v0 += type_convert<float>(bv[0]);
                  v1 += type_convert<float>(bv[1]);
                  v2 += type_convert<float>(bv[2]);
                  v3 += type_convert<float>(bv[3]);
                }
                uint64_t out_packed;
                bf16 *out = reinterpret_cast<bf16 *>(&out_packed);
                out[0] = type_convert<bf16>(v0);
                out[1] = type_convert<bf16>(v1);
                out[2] = type_convert<bf16>(v2);
                out[3] = type_convert<bf16>(v3);
                nt_store_u64(&d_output[out_base], out_packed);
              } else {
                for (index_t i = 0; i < 4; i++) {
                  index_t global_n = global_n_base + i;
                  if (global_n < output_size) {
                    float val = c_buf[r_base + i];
                    if (residual_add && d_residual != nullptr) {
                      val += type_convert<float>(
                          d_residual[base_m * o_stride + global_n]);
                    }
                    if (d_bias) {
                      val += type_convert<float>(d_bias[global_n]);
                    }
                    nt_store_bf16(&d_output[base_m * o_stride + global_n],
                                  type_convert<bf16>(val));
                  }
                }
              }
            }
          }
        }
      } else if constexpr (use_medium_tile) {
        // 64x64 tiles: 32x32 MFMA TransposedCDistribution
        // MWarp=2, NWarp=2, MIterPerWarp=1, NIterPerWarp=1
        // Each thread has 16 float32 registers.
        // warp_m = warp_id / 2, warp_n = warp_id % 2
        // Row = warp_m * 32 + (lane_id % 32)
        // Col = warp_n * 32 + (g*8 + (lane_id/32)*4 + r%4)
        index_t warp_id = threadIdx.x >> 6;
        index_t lane_id = threadIdx.x & 63;
        index_t warp_m = warp_id >> 1;
        index_t warp_n = warp_id & 1;
        index_t tile_row = lane_id & 31;
        index_t m_lane = lane_id >> 5;

        auto &c_buf = c_block_tile.get_thread_buffer();
        index_t global_m = m_offset + warp_m * 32 + tile_row;

        if (global_m < BATCH_SIZE) {
#pragma unroll
          for (index_t g = 0; g < 4; g++) {
            index_t col_in_warp = g * 8 + m_lane * 4;
            index_t global_n_base = n_offset + warp_n * 32 + col_in_warp;
            index_t r_base = g * 4;

            if (global_n_base + 3 < output_size) {
              index_t out_base = global_m * o_stride + global_n_base;
              float v0 = c_buf[r_base], v1 = c_buf[r_base + 1],
                    v2 = c_buf[r_base + 2], v3 = c_buf[r_base + 3];
              if (residual_add && d_residual != nullptr) {
                uint64_t res_packed =
                    *reinterpret_cast<uint64_t const *>(&d_residual[out_base]);
                bf16 const *res = reinterpret_cast<bf16 const *>(&res_packed);
                v0 += type_convert<float>(res[0]);
                v1 += type_convert<float>(res[1]);
                v2 += type_convert<float>(res[2]);
                v3 += type_convert<float>(res[3]);
              }
              if (d_bias) {
                uint64_t bias_packed =
                    *reinterpret_cast<uint64_t const *>(&d_bias[global_n_base]);
                bf16 const *bv = reinterpret_cast<bf16 const *>(&bias_packed);
                v0 += type_convert<float>(bv[0]);
                v1 += type_convert<float>(bv[1]);
                v2 += type_convert<float>(bv[2]);
                v3 += type_convert<float>(bv[3]);
              }
              uint64_t out_packed;
              bf16 *out = reinterpret_cast<bf16 *>(&out_packed);
              out[0] = type_convert<bf16>(v0);
              out[1] = type_convert<bf16>(v1);
              out[2] = type_convert<bf16>(v2);
              out[3] = type_convert<bf16>(v3);
              nt_store_u64(&d_output[out_base], out_packed);
            } else {
              for (index_t i = 0; i < 4; i++) {
                index_t global_n = global_n_base + i;
                if (global_n < output_size) {
                  float val = c_buf[r_base + i];
                  if (residual_add && d_residual != nullptr) {
                    val += type_convert<float>(
                        d_residual[global_m * o_stride + global_n]);
                  }
                  if (d_bias) {
                    val += type_convert<float>(d_bias[global_n]);
                  }
                  nt_store_bf16(&d_output[global_m * o_stride + global_n],
                                type_convert<bf16>(val));
                }
              }
            }
          }
        }
      } else {
        // 16x64 tiles: hand-written epilogue for 16x16 MFMA
        // TransposedCDistribution
        index_t warp_id = threadIdx.x >> 6;
        index_t lane_id = threadIdx.x & 63;
        index_t tile_row = lane_id & 15;
        index_t tile_col_base = warp_id * 16 + ((lane_id >> 4) << 2);

        auto &c_buf = c_block_tile.get_thread_buffer();

        index_t global_m = m_offset + tile_row;
        index_t global_n_base = n_offset + tile_col_base;

        if (global_m < BATCH_SIZE && global_n_base + 3 < output_size) {
          index_t out_base = global_m * o_stride + global_n_base;
          float v0 = c_buf[0], v1 = c_buf[1], v2 = c_buf[2], v3 = c_buf[3];
          if (residual_add && d_residual != nullptr) {
            uint64_t res_packed =
                *reinterpret_cast<uint64_t const *>(&d_residual[out_base]);
            bf16 const *res = reinterpret_cast<bf16 const *>(&res_packed);
            v0 += type_convert<float>(res[0]);
            v1 += type_convert<float>(res[1]);
            v2 += type_convert<float>(res[2]);
            v3 += type_convert<float>(res[3]);
          }
          if (d_bias) {
            uint64_t bias_packed =
                *reinterpret_cast<uint64_t const *>(&d_bias[global_n_base]);
            bf16 const *bv = reinterpret_cast<bf16 const *>(&bias_packed);
            v0 += type_convert<float>(bv[0]);
            v1 += type_convert<float>(bv[1]);
            v2 += type_convert<float>(bv[2]);
            v3 += type_convert<float>(bv[3]);
          }
          uint64_t out_packed;
          bf16 *out = reinterpret_cast<bf16 *>(&out_packed);
          out[0] = type_convert<bf16>(v0);
          out[1] = type_convert<bf16>(v1);
          out[2] = type_convert<bf16>(v2);
          out[3] = type_convert<bf16>(v3);
          nt_store_u64(&d_output[out_base], out_packed);
        } else if (global_m < BATCH_SIZE) {
#pragma unroll
          for (index_t i = 0; i < 4; i++) {
            index_t global_n = global_n_base + i;
            if (global_n < output_size) {
              float val = c_buf[i];
              if (residual_add && d_residual != nullptr) {
                val += type_convert<float>(
                    d_residual[global_m * o_stride + global_n]);
              }
              if (d_bias) {
                val += type_convert<float>(d_bias[global_n]);
              }
              nt_store_bf16(&d_output[global_m * o_stride + global_n],
                            type_convert<bf16>(val));
            }
          }
        }
      }
    }
  }
}

// ============================================================================
// Split-K variant: writes partial GEMM results to float32 workspace.
// Each K-split writes to its own slice (no atomics needed).
// A separate reduce kernel sums all K-split partials + residual → bf16 output.
// ============================================================================

template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int O_STRIDE = OUTPUT_SIZE>
__device__ __forceinline__ void linear_kernel_ck_splitk(void const *input_ptr,
                                                        void const *weight_ptr,
                                                        void *workspace_ptr,
                                                        int num_active_tokens) {
#ifdef MPK_DISABLE_LINEAR
  return;
#endif
  using namespace ck_tile;

  constexpr bool use_large_tile = (BATCH_SIZE > 16);
  constexpr index_t MPerBlock = use_large_tile ? 32 : 16;
  constexpr index_t NPerBlock = use_large_tile ? 128 : 64;
  constexpr index_t KPerBlock = use_large_tile ? 128 : 256;

  constexpr index_t LoopM = (BATCH_SIZE + MPerBlock - 1) / MPerBlock;
  constexpr index_t LoopN = (OUTPUT_SIZE + NPerBlock - 1) / NPerBlock;
  constexpr index_t NumLoopK = REDUCTION_SIZE / KPerBlock;

  constexpr index_t MWarp = 1;
  constexpr index_t NWarp = 4;

  using BlockTile = sequence<MPerBlock, NPerBlock, KPerBlock>;
  using BlockWarps = sequence<MWarp, NWarp>;
  using WarpTile = sequence<MPerBlock, NPerBlock / NWarp, KPerBlock>;

  using GemmShape = TileGemmShape<BlockTile, BlockWarps, WarpTile>;
  using GemmTraits = TileGemmUniversalTraits<true,
                                             false,
                                             true,
                                             false,
                                             tensor_layout::gemm::RowMajor,
                                             tensor_layout::gemm::ColumnMajor,
                                             tensor_layout::gemm::RowMajor>;

  using Problem = GemmPipelineProblem<bf16, bf16, float, GemmShape, GemmTraits>;
  using PipelinePolicy =
      GemmPipelineSmallTilePolicy<MPerBlock, NPerBlock, KPerBlock>;
  using Pipeline = GemmPipelineAGmemBGmemCRegV2<Problem, PipelinePolicy>;

  // Promote pointers to wave-uniform to eliminate waterfall sequences.
  bf16 const *d_input =
      reinterpret_cast<bf16 const *>(__uniform_addr(input_ptr));
  bf16 const *d_weight =
      reinterpret_cast<bf16 const *>(__uniform_addr(weight_ptr));
  float *d_workspace = reinterpret_cast<float *>(__uniform_addr(workspace_ptr));

  extern __shared__ char smem[];

  for (index_t mm = 0; mm < LoopM; mm++) {
    index_t m_offset = mm * MPerBlock;
    index_t m_size = (mm == LoopM - 1) ? (BATCH_SIZE - m_offset) : MPerBlock;
    if (m_size <= 0) {
      continue;
    }

    for (index_t nn = 0; nn < LoopN; nn++) {
      index_t n_offset = nn * NPerBlock;
      index_t n_size = (nn == LoopN - 1) ? (OUTPUT_SIZE - n_offset) : NPerBlock;
      if (n_size <= 0) {
        continue;
      }

      auto a_tensor_view = make_naive_tensor_view<address_space_enum::global>(
          d_input + m_offset * REDUCTION_SIZE,
          make_tuple(m_size, index_t(REDUCTION_SIZE)),
          make_tuple(index_t(REDUCTION_SIZE), index_t(1)),
          number<8>{},
          number<1>{});

      auto b_tensor_view = make_naive_tensor_view<address_space_enum::global>(
          d_weight + n_offset * REDUCTION_SIZE,
          make_tuple(n_size, index_t(REDUCTION_SIZE)),
          make_tuple(index_t(REDUCTION_SIZE), index_t(1)),
          number<8>{},
          number<1>{});

      auto a_tile_window =
          make_tile_window(a_tensor_view,
                           make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                           {0, 0});

      auto b_tile_window =
          make_tile_window(b_tensor_view,
                           make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                           {0, 0});

      Pipeline pipeline;
      auto c_block_tile =
          pipeline(a_tile_window, b_tile_window, NumLoopK, smem);

      block_sync_lds();

      // Store f32 results to workspace using CK store_tile
      {
        auto ws_tensor_view =
            make_naive_tensor_view<address_space_enum::global>(
                d_workspace + m_offset * O_STRIDE + n_offset,
                make_tuple(number<MPerBlock>{}, number<NPerBlock>{}),
                make_tuple(index_t(O_STRIDE), index_t(1)),
                number<1>{},
                number<1>{});
        auto ws_tile_window = make_tile_window(
            ws_tensor_view,
            make_tuple(number<MPerBlock>{}, number<NPerBlock>{}),
            {0, 0});
        store_tile(ws_tile_window, c_block_tile);
      }
    }
  }
}

// ============================================================================
// Split-K reduce: sum K_SPLITS float32 workspace slices + bf16 residual → bf16
// ============================================================================
template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int K_SPLITS,
          int WS_STRIDE,
          int O_STRIDE = OUTPUT_SIZE>
__device__ __forceinline__ void splitk_reduce_kernel(void const *workspace_ptr,
                                                     void const *residual_ptr,
                                                     void *output_ptr,
                                                     int num_active_tokens) {
  float const *ws =
      reinterpret_cast<float const *>(__uniform_addr(workspace_ptr));
  bf16 const *res =
      reinterpret_cast<bf16 const *>(__uniform_addr(residual_ptr));
  bf16 *out = reinterpret_cast<bf16 *>(__uniform_addr(output_ptr));

  constexpr int TOTAL = BATCH_SIZE * OUTPUT_SIZE;
  for (int idx = threadIdx.x; idx < TOTAL; idx += 256) {
    int row = idx / OUTPUT_SIZE;
    int col = idx % OUTPUT_SIZE;

    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < K_SPLITS; k++) {
      sum += ws[k * BATCH_SIZE * WS_STRIDE + row * WS_STRIDE + col];
    }
    sum += ck_tile::type_convert<float>(res[row * O_STRIDE + col]);
    out[row * O_STRIDE + col] = ck_tile::type_convert<bf16>(sum);
  }
}

// ============================================================================
// Split-K with float32 atomicAdd — SINGLE TASK (no separate reduce)
// ============================================================================
template <typename T,
          int BATCH_SIZE,
          int NPerBlock,
          int REDUCTION_SIZE,
          int K_SPLITS>
__device__ __forceinline__ void
    splitk_linear_res_atomic_kernel(void const *input_ptr,
                                    void const *weight_ptr,
                                    void const *residual_ptr,
                                    void *workspace_ptr,
                                    void *output_ptr,
                                    int *done_counter_ptr,
                                    int num_active_tokens,
                                    int ws_stride,
                                    int o_stride) {
#ifdef MPK_DISABLE_LINEAR
  return;
#endif
  using namespace ck_tile;

  // Split-K uses fixed small tiles (atomicAdd epilogue needs manual register
  // mapping)
  constexpr index_t MPerBlock_CK = 16;
  constexpr index_t NPerBlock_CK = 64;
  constexpr index_t KPerBlock = 256;

  constexpr index_t K_per_split = REDUCTION_SIZE / K_SPLITS;
  constexpr index_t LoopM = (BATCH_SIZE + MPerBlock_CK - 1) / MPerBlock_CK;
  constexpr index_t LoopN = (NPerBlock + NPerBlock_CK - 1) / NPerBlock_CK;
  constexpr index_t NumLoopK = K_per_split / KPerBlock;

  constexpr index_t MWarp = 1;
  constexpr index_t NWarp = 4;

  using BlockTile_CK = sequence<MPerBlock_CK, NPerBlock_CK, KPerBlock>;
  using BlockWarps = sequence<MWarp, NWarp>;
  using WarpTile = sequence<16, 16, KPerBlock>;

  using GemmShape = TileGemmShape<BlockTile_CK, BlockWarps, WarpTile>;
  using GemmTraits = TileGemmUniversalTraits<true,
                                             false,
                                             true,
                                             false,
                                             tensor_layout::gemm::RowMajor,
                                             tensor_layout::gemm::ColumnMajor,
                                             tensor_layout::gemm::RowMajor>;

  using Problem = GemmPipelineProblem<bf16, bf16, float, GemmShape, GemmTraits>;
  using PipelinePolicy =
      GemmPipelineSmallTilePolicy<MPerBlock_CK, NPerBlock_CK, KPerBlock>;
  using Pipeline = GemmPipelineAGmemBGmemCRegV2<Problem, PipelinePolicy>;

  // Promote pointers and runtime dims to wave-uniform to eliminate waterfall.
  bf16 const *d_input =
      reinterpret_cast<bf16 const *>(__uniform_addr(input_ptr));
  bf16 const *d_weight =
      reinterpret_cast<bf16 const *>(__uniform_addr(weight_ptr));
  float *d_ws = reinterpret_cast<float *>(__uniform_addr(workspace_ptr));
  ws_stride = __builtin_amdgcn_readfirstlane(ws_stride);
  o_stride = __builtin_amdgcn_readfirstlane(o_stride);

  extern __shared__ char smem[];

  // Phase 1: GEMM + atomicAdd to workspace
  for (index_t mm = 0; mm < LoopM; mm++) {
    index_t m_offset = mm * MPerBlock_CK;
    index_t m_size = (mm == LoopM - 1) ? (BATCH_SIZE - m_offset) : MPerBlock_CK;
    if (m_size <= 0) {
      continue;
    }

    for (index_t nn = 0; nn < LoopN; nn++) {
      index_t n_offset = nn * NPerBlock_CK;
      index_t n_size =
          (nn == LoopN - 1) ? (NPerBlock - n_offset) : NPerBlock_CK;
      if (n_size <= 0) {
        continue;
      }

      auto a_view = make_naive_tensor_view<address_space_enum::global>(
          d_input + m_offset * REDUCTION_SIZE,
          make_tuple(m_size, index_t(K_per_split)),
          make_tuple(index_t(REDUCTION_SIZE), index_t(1)),
          number<8>{},
          number<1>{});

      auto b_view = make_naive_tensor_view<address_space_enum::global>(
          d_weight + n_offset * REDUCTION_SIZE,
          make_tuple(n_size, index_t(K_per_split)),
          make_tuple(index_t(REDUCTION_SIZE), index_t(1)),
          number<8>{},
          number<1>{});

      auto a_win = make_tile_window(
          a_view,
          make_tuple(number<MPerBlock_CK>{}, number<KPerBlock>{}),
          {0, 0});
      auto b_win = make_tile_window(
          b_view,
          make_tuple(number<NPerBlock_CK>{}, number<KPerBlock>{}),
          {0, 0});

      Pipeline pipeline;
      auto c_tile = pipeline(a_win, b_win, NumLoopK, smem);
      block_sync_lds();

      // float32 atomicAdd epilogue
      index_t warp_id = threadIdx.x >> 6;
      index_t lane_id = threadIdx.x & 63;
      index_t tile_row = lane_id & 15;
      index_t tile_col_base = warp_id * 16 + ((lane_id >> 4) << 2);
      auto &c_buf = c_tile.get_thread_buffer();

      index_t global_m = m_offset + tile_row;
      index_t global_n_base = n_offset + tile_col_base;

      if (global_m < BATCH_SIZE && global_n_base + 3 < NPerBlock) {
        index_t base = global_m * ws_stride + global_n_base;
        atomicAdd(&d_ws[base], c_buf[0]);
        atomicAdd(&d_ws[base + 1], c_buf[1]);
        atomicAdd(&d_ws[base + 2], c_buf[2]);
        atomicAdd(&d_ws[base + 3], c_buf[3]);
      } else if (global_m < BATCH_SIZE) {
#pragma unroll
        for (index_t i = 0; i < 4; i++) {
          if (global_n_base + i < NPerBlock) {
            atomicAdd(&d_ws[global_m * ws_stride + global_n_base + i],
                      c_buf[i]);
          }
        }
      }
    }
  }

  // Phase 2: Track completion with atomic counter
  // Agent-scope release fence — L2 NOT coherent across XCDs, need writeback
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  int *d_done = reinterpret_cast<int *>(__uniform_addr(done_counter_ptr));
  __shared__ int is_last;
  if (threadIdx.x == 0) {
    int old = atomicAdd(d_done, 1);
    is_last = (old == K_SPLITS - 1) ? 1 : 0;
  }
  __syncthreads();

  // Phase 3: Last block — add residual, convert to bf16, zero workspace, reset
  // counter
  if (is_last) {
    bf16 const *res =
        residual_ptr
            ? reinterpret_cast<bf16 const *>(__uniform_addr(residual_ptr))
            : nullptr;
    bf16 *out = reinterpret_cast<bf16 *>(__uniform_addr(output_ptr));

    constexpr int TOTAL = BATCH_SIZE * NPerBlock;
    for (int idx = threadIdx.x; idx < TOTAL; idx += 256) {
      int row = idx / NPerBlock;
      int col = idx % NPerBlock;
      index_t ws_idx = row * ws_stride + col;
      index_t out_idx = row * o_stride + col;

      float val = d_ws[ws_idx];
      if (res) {
        val += type_convert<float>(res[out_idx]);
      }
      out[out_idx] = type_convert<bf16>(val);
      d_ws[ws_idx] = 0.0f;
    }
    if (threadIdx.x == 0) {
      *d_done = 0;
    }
  }
}

} // namespace kernel
