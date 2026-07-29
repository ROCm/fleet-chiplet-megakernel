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

// Reuse CK infrastructure: policies, bf16, BlockGemm, LDS descriptors.
#include "linear_ck_mi300.cuh"

namespace kernel {

// Fused SiLU-Mul-Linear kernel for AMD MI300 (V5: double-buffered LDS +
// register GEMM). Computes: output = silu(gate) * up @ weight^T [+ residual]
//
// Uses BlockGemmARegBRegCRegV1: GEMM reads from registers, NOT LDS.
// This enables true overlap of MFMA with LDS writes and VMEM loads
// since MFMA and DS/VMEM use separate hardware units.
//
// Pipeline (steady state, PING step):
//   sync_lds()                        // wait for prev buf1 writes
//   a_reg1 = DS_read(LDS buf1)       // load from LDS → registers
//   b_reg1 = DS_read(LDS buf1)
//   silu_mul(gate, up) → DS_write(buf0) // process + store to OTHER buffer
//   VMEM_load(next gate, up, b)       // prefetch next tile
//   GEMM(c, a_reg0, b_reg0)          // MFMA from registers (overlaps above!)
//
// Input layout: [BATCH_SIZE, REDUCTION_SIZE * 2]
//   gate = input[:, :REDUCTION_SIZE]
//   up   = input[:, REDUCTION_SIZE:]
template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int O_STRIDE = OUTPUT_SIZE>
__device__ __forceinline__ void
    silu_mul_linear_task_impl(void const *input_ptr,
                              void const *weight_ptr,
                              void const *residual_ptr,
                              void *output_ptr,
                              bool residual_add = true) {
  using namespace ck_tile;

  constexpr index_t MPerBlock = 16;
  constexpr index_t NPerBlock = 64;
  constexpr index_t KPerBlock = 128;

  constexpr index_t LoopM = (BATCH_SIZE + MPerBlock - 1) / MPerBlock;
  constexpr index_t LoopN = (OUTPUT_SIZE + NPerBlock - 1) / NPerBlock;
  constexpr index_t NumLoopK = REDUCTION_SIZE / KPerBlock;

  constexpr index_t MWarp = 1;
  constexpr index_t NWarp = 4;

  using BlockTile = sequence<MPerBlock, NPerBlock, KPerBlock>;
  using BlockWarps = sequence<MWarp, NWarp>;
  using WarpTile = sequence<16, 16, KPerBlock>;

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

  // Register-based block GEMM (reads A, B from registers, not LDS)
  using RegBlockGemm =
      BlockGemmARegBRegCRegV1<Problem, BlockGemmSmallM16Policy>;

  // Warp GEMM type for distribution encoding
  using WG =
      remove_cvref_t<decltype(BlockGemmSmallM16Policy::
                                  template GetWarpGemmMWarpNWarp<Problem>()
                                      .template at<0>())>;
  constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
  constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
  constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

  // A register distribution for GEMM (must match BlockGemmARegBRegCRegV1's
  // expectation)
  constexpr auto a_outer_enc = tile_distribution_encoding<
      sequence<NWarp>,
      tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
      tuple<sequence<1, 0>>,
      tuple<sequence<1, 0>>,
      sequence<1, 2>,
      sequence<0, 0>>{};
  constexpr auto a_reg_dstr_encode =
      detail::make_embed_tile_distribution_encoding(
          a_outer_enc, typename WG::AWarpDstrEncoding{});
  constexpr auto a_reg_dstr = make_static_tile_distribution(a_reg_dstr_encode);

  // B register distribution for GEMM
  constexpr auto b_outer_enc = tile_distribution_encoding<
      sequence<MWarp>,
      tuple<sequence<NIterPerWarp, NWarp>, sequence<KIterPerWarp>>,
      tuple<sequence<0, 1>>,
      tuple<sequence<0, 1>>,
      sequence<1, 2>,
      sequence<0, 0>>{};
  constexpr auto b_reg_dstr_encode =
      detail::make_embed_tile_distribution_encoding(
          b_outer_enc, typename WG::BWarpDstrEncoding{});
  constexpr auto b_reg_dstr = make_static_tile_distribution(b_reg_dstr_encode);

  constexpr index_t INPUT_STRIDE = REDUCTION_SIZE * 2;
  bf16 const *d_gate = static_cast<bf16 const *>(input_ptr);
  bf16 const *d_up = d_gate + REDUCTION_SIZE;
  bf16 const *d_weight = static_cast<bf16 const *>(weight_ptr);
  bf16 const *d_residual = static_cast<bf16 const *>(residual_ptr);
  bf16 *d_output = static_cast<bf16 *>(output_ptr);

  extern __shared__ char smem[];

  // === Double-buffered LDS setup ===
  constexpr auto a_lds_block_desc =
      PipelinePolicy::template MakeALdsBlockDescriptor<Problem>();
  constexpr auto b_lds_block_desc =
      PipelinePolicy::template MakeBLdsBlockDescriptor<Problem>();

  constexpr index_t a_lds_bytes_aligned =
      integer_divide_ceil(
          sizeof(bf16) * a_lds_block_desc.get_element_space_size(), 16) *
      16;
  constexpr index_t b_lds_bytes =
      sizeof(bf16) * b_lds_block_desc.get_element_space_size();
  constexpr index_t single_buf_bytes = a_lds_bytes_aligned + b_lds_bytes;
  constexpr index_t single_buf_aligned =
      integer_divide_ceil(single_buf_bytes, 16) * 16;

  // Buffer 0
  bf16 *p_a_lds0 = static_cast<bf16 *>(static_cast<void *>(smem));
  bf16 *p_b_lds0 =
      static_cast<bf16 *>(static_cast<void *>(smem + a_lds_bytes_aligned));
  // Buffer 1
  bf16 *p_a_lds1 =
      static_cast<bf16 *>(static_cast<void *>(smem + single_buf_aligned));
  bf16 *p_b_lds1 = static_cast<bf16 *>(
      static_cast<void *>(smem + single_buf_aligned + a_lds_bytes_aligned));

  // LDS tensor views
  auto a_lds_view0 =
      make_tensor_view<address_space_enum::lds>(p_a_lds0, a_lds_block_desc);
  auto b_lds_view0 =
      make_tensor_view<address_space_enum::lds>(p_b_lds0, b_lds_block_desc);
  auto a_lds_view1 =
      make_tensor_view<address_space_enum::lds>(p_a_lds1, a_lds_block_desc);
  auto b_lds_view1 =
      make_tensor_view<address_space_enum::lds>(p_b_lds1, b_lds_block_desc);

  // DRAM copy distribution (for global load → LDS store)
  constexpr auto a_dram_dstr =
      PipelinePolicy::template MakeADramTileDistribution<Problem>();
  constexpr auto b_dram_dstr =
      PipelinePolicy::template MakeBDramTileDistribution<Problem>();

  // LDS copy windows (for store_tile from registers → LDS)
  auto a_copy_lds0 =
      make_tile_window(a_lds_view0,
                       make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                       {0, 0},
                       a_dram_dstr);
  auto b_copy_lds0 =
      make_tile_window(b_lds_view0,
                       make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                       {0, 0},
                       b_dram_dstr);
  auto a_copy_lds1 =
      make_tile_window(a_lds_view1,
                       make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                       {0, 0},
                       a_dram_dstr);
  auto b_copy_lds1 =
      make_tile_window(b_lds_view1,
                       make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                       {0, 0},
                       b_dram_dstr);

  // LDS GEMM windows (for load_tile from LDS → registers with GEMM
  // distribution)
  auto a_lds_gemm0 =
      make_tile_window(a_lds_view0,
                       make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                       {0, 0},
                       a_reg_dstr);
  auto b_lds_gemm0 =
      make_tile_window(b_lds_view0,
                       make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                       {0, 0},
                       b_reg_dstr);
  auto a_lds_gemm1 =
      make_tile_window(a_lds_view1,
                       make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                       {0, 0},
                       a_reg_dstr);
  auto b_lds_gemm1 =
      make_tile_window(b_lds_view1,
                       make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                       {0, 0},
                       b_reg_dstr);

  constexpr auto block_gemm = RegBlockGemm{};

  // SiLU-Mul element function
  auto silu_mul_func = [](bf16 const &gate_val, bf16 const &up_val) {
    float x = type_convert<float>(gate_val);
    float m = type_convert<float>(up_val);
    float silu_x = x / (1.0f + __expf(-x));
    return type_convert<bf16>(silu_x * m);
  };

  // === Main computation ===
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

      // Gate DRAM window
      auto gate_view = make_naive_tensor_view<address_space_enum::global>(
          d_gate + m_offset * INPUT_STRIDE,
          make_tuple(m_size, index_t(REDUCTION_SIZE)),
          make_tuple(index_t(INPUT_STRIDE), index_t(1)),
          number<8>{},
          number<1>{});
      auto gate_window =
          make_tile_window(gate_view,
                           make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                           {0, 0},
                           a_dram_dstr);

      // Up DRAM window
      auto up_view = make_naive_tensor_view<address_space_enum::global>(
          d_up + m_offset * INPUT_STRIDE,
          make_tuple(m_size, index_t(REDUCTION_SIZE)),
          make_tuple(index_t(INPUT_STRIDE), index_t(1)),
          number<8>{},
          number<1>{});
      auto up_window =
          make_tile_window(up_view,
                           make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                           {0, 0},
                           a_dram_dstr);

      // B (weight) DRAM window
      auto b_view = make_naive_tensor_view<address_space_enum::global>(
          d_weight + n_offset * REDUCTION_SIZE,
          make_tuple(n_size, index_t(REDUCTION_SIZE)),
          make_tuple(index_t(REDUCTION_SIZE), index_t(1)),
          number<8>{},
          number<1>{});
      auto b_window =
          make_tile_window(b_view,
                           make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                           {0, 0},
                           b_dram_dstr);

      // Initialize accumulator
      auto c_block_tile = block_gemm.MakeCBlockTile();
      tile_elementwise_inout([](auto &c) { c = 0; }, c_block_tile);

      // ============================================================
      // V5 Pipeline: Double-buffered LDS + Register GEMM
      // Modeled on CK V4 (CompV4) pipeline
      // ============================================================

      // --- Global prefetch tile 0 → LDS buf0 ---
      auto gate_tile = load_tile(gate_window);
      auto up_tile = load_tile(up_window);
      auto b_tile = load_tile(b_window);
      move_tile_window(gate_window, {0, KPerBlock});
      move_tile_window(up_window, {0, KPerBlock});
      move_tile_window(b_window, {0, KPerBlock});

      auto fused_a = tile_elementwise_in(silu_mul_func, gate_tile, up_tile);
      store_tile(a_copy_lds0, fused_a);
      store_tile(b_copy_lds0, b_tile);

      // --- Global prefetch tile 1 ---
      gate_tile = load_tile(gate_window);
      up_tile = load_tile(up_window);
      b_tile = load_tile(b_window);
      move_tile_window(gate_window, {0, KPerBlock});
      move_tile_window(up_window, {0, KPerBlock});
      move_tile_window(b_window, {0, KPerBlock});

      fused_a = tile_elementwise_in(silu_mul_func, gate_tile, up_tile);

      // Register tile types for GEMM (declared here to persist across loop)
      using ARegTile = decltype(load_tile(a_lds_gemm0));
      using BRegTile = decltype(load_tile(b_lds_gemm0));
      ARegTile a_reg0, a_reg1;
      BRegTile b_reg0, b_reg1;

      // Sync: wait for buf0 writes, then load buf0 → registers
      block_sync_lds();
      a_reg0 = load_tile(a_lds_gemm0);
      b_reg0 = load_tile(b_lds_gemm0);

      // Store tile 1 → LDS buf1
      store_tile(a_copy_lds1, fused_a);
      store_tile(b_copy_lds1, b_tile);

      // Prefetch tile 2 from DRAM
      gate_tile = load_tile(gate_window);
      up_tile = load_tile(up_window);
      b_tile = load_tile(b_window);
      move_tile_window(gate_window, {0, KPerBlock});
      move_tile_window(up_window, {0, KPerBlock});
      move_tile_window(b_window, {0, KPerBlock});

      // === Ping-pong main loop ===
      index_t iCounter = NumLoopK - 2;
      while (iCounter > 1) {
        // PING: GEMM from a_reg0/b_reg0 (buf0 data)
        //       Fill buf0 with new data, prefetch from buf1
        {
          block_sync_lds();                // wait for buf1 writes
          a_reg1 = load_tile(a_lds_gemm1); // DS read buf1 → reg
          b_reg1 = load_tile(b_lds_gemm1);

          // Process prefetched tile → store to buf0
          fused_a = tile_elementwise_in(silu_mul_func, gate_tile, up_tile);
          store_tile(a_copy_lds0, fused_a);
          store_tile(b_copy_lds0, b_tile);

          // Prefetch next from DRAM (overlaps with GEMM below!)
          gate_tile = load_tile(gate_window);
          up_tile = load_tile(up_window);
          b_tile = load_tile(b_window);
          move_tile_window(gate_window, {0, KPerBlock});
          move_tile_window(up_window, {0, KPerBlock});
          move_tile_window(b_window, {0, KPerBlock});

          // GEMM from registers (MFMA, no LDS access!)
          block_gemm(c_block_tile, a_reg0, b_reg0);
        }

        // PONG: GEMM from a_reg1/b_reg1 (buf1 data)
        //       Fill buf1 with new data, prefetch from buf0
        {
          block_sync_lds();                // wait for buf0 writes
          a_reg0 = load_tile(a_lds_gemm0); // DS read buf0 → reg
          b_reg0 = load_tile(b_lds_gemm0);

          // Process prefetched tile → store to buf1
          fused_a = tile_elementwise_in(silu_mul_func, gate_tile, up_tile);
          store_tile(a_copy_lds1, fused_a);
          store_tile(b_copy_lds1, b_tile);

          // Prefetch next from DRAM
          gate_tile = load_tile(gate_window);
          up_tile = load_tile(up_window);
          b_tile = load_tile(b_window);
          move_tile_window(gate_window, {0, KPerBlock});
          move_tile_window(up_window, {0, KPerBlock});
          move_tile_window(b_window, {0, KPerBlock});

          // GEMM from registers
          block_gemm(c_block_tile, a_reg1, b_reg1);
        }
        iCounter -= 2;
      }

      // === Tail ===
      // After the loop: a_reg0 has buf0 data, registers have last prefetched
      // tile Remaining: GEMM a_reg0 (buf0), process last tile → buf0, GEMM
      // buf1, GEMM buf0
      if (iCounter == 1) {
        // 3 remaining: GEMM(a_reg0), then buf1, then last tile in buf0
        block_sync_lds();
        auto a_reg1 = load_tile(a_lds_gemm1);
        auto b_reg1 = load_tile(b_lds_gemm1);
        fused_a = tile_elementwise_in(silu_mul_func, gate_tile, up_tile);
        store_tile(a_copy_lds0, fused_a);
        store_tile(b_copy_lds0, b_tile);
        block_gemm(c_block_tile, a_reg0, b_reg0);

        block_sync_lds();
        a_reg0 = load_tile(a_lds_gemm0);
        b_reg0 = load_tile(b_lds_gemm0);
        block_gemm(c_block_tile, a_reg1, b_reg1);

        block_gemm(c_block_tile, a_reg0, b_reg0);
      } else {
        // iCounter == 0: 2 remaining: GEMM(a_reg0) buf0, then buf1
        block_sync_lds();
        auto a_reg1 = load_tile(a_lds_gemm1);
        auto b_reg1 = load_tile(b_lds_gemm1);
        block_gemm(c_block_tile, a_reg0, b_reg0);

        block_gemm(c_block_tile, a_reg1, b_reg1);
      }

      // === Epilogue (identical to linear_kernel_ck) ===
      {
        index_t warp_id = threadIdx.x >> 6;
        index_t lane_id = threadIdx.x & 63;
        index_t tile_row = lane_id & 15;
        index_t tile_col_base = warp_id * 16 + ((lane_id >> 4) << 2);

        auto &c_buf = c_block_tile.get_thread_buffer();

        index_t global_m = m_offset + tile_row;
        index_t global_n_base = n_offset + tile_col_base;

        if (global_m < BATCH_SIZE && global_n_base + 3 < OUTPUT_SIZE) {
          index_t out_base = global_m * O_STRIDE + global_n_base;
          if (residual_add && d_residual != nullptr) {
            uint64_t res_packed =
                *reinterpret_cast<uint64_t const *>(&d_residual[out_base]);
            bf16 const *res = reinterpret_cast<bf16 const *>(&res_packed);
            uint64_t out_packed;
            bf16 *out = reinterpret_cast<bf16 *>(&out_packed);
            out[0] = type_convert<bf16>(c_buf[0] + type_convert<float>(res[0]));
            out[1] = type_convert<bf16>(c_buf[1] + type_convert<float>(res[1]));
            out[2] = type_convert<bf16>(c_buf[2] + type_convert<float>(res[2]));
            out[3] = type_convert<bf16>(c_buf[3] + type_convert<float>(res[3]));
            nt_store_u64(&d_output[out_base], out_packed);
          } else {
            uint64_t out_packed;
            bf16 *out = reinterpret_cast<bf16 *>(&out_packed);
            out[0] = type_convert<bf16>(c_buf[0]);
            out[1] = type_convert<bf16>(c_buf[1]);
            out[2] = type_convert<bf16>(c_buf[2]);
            out[3] = type_convert<bf16>(c_buf[3]);
            nt_store_u64(&d_output[out_base], out_packed);
          }
        } else if (global_m < BATCH_SIZE) {
#pragma unroll
          for (index_t i = 0; i < 4; i++) {
            index_t global_n = global_n_base + i;
            if (global_n < OUTPUT_SIZE) {
              float val = c_buf[i];
              if (residual_add && d_residual != nullptr) {
                val += type_convert<float>(
                    d_residual[global_m * O_STRIDE + global_n]);
              }
              nt_store_bf16(&d_output[global_m * O_STRIDE + global_n],
                            type_convert<bf16>(val));
            }
          }
        }
      }
    }
  }
}

} // namespace kernel
