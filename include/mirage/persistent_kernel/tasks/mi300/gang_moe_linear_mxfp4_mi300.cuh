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

// Gang MoE MXFP4 linear kernel for MI350 (gfx950).
//
// Uses hardware-accelerated FP4×FP8 MFMA:
// __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4
// - Weights: FP4 E2M1 with E8M0 block scales (pre-packed, read from HBM)
// - Tokens:  BF16 input, quantized on-the-fly to FP8 E4M3 in shared memory
// - K=128 elements per MFMA instruction (vs 16 in the old bf16 path)
// - Hardware handles dequant + scale application + multiply-accumulate
//
// Weight format: MXFP4 packed per workgroup (same layout as before):
//   [E, expert_wgs, wg_bytes] where wg_bytes = OPW*(K/2) + OPW*(K/32)
//
// Dispatch: 8 gang tasks (1 per XCD), tiles spread across all XCDs:
//   global_tile = tile_idx * 8 + xcd_id  (interleaved round-robin)
//   expert_idx = global_tile / TILES_PER_EXPERT
//   tile_within_expert = global_tile % TILES_PER_EXPERT

#pragma once
#include "tasks/common/common_header.cuh"
#include "tasks/mi300/swigluoai_mi300.cuh"
#include <hip/hip_bf16.h>

namespace kernel {

// ── FP8 E4M3 quantization types and helpers (gfx950) ──────────────────────
typedef int __attribute__((ext_vector_type(4))) i32x4_t;
typedef int __attribute__((ext_vector_type(8))) i32x8_t;
typedef float __attribute__((ext_vector_type(4))) f32x4_t;
typedef short __attribute__((ext_vector_type(2))) fp8x4_t;

// Direct HBM→LDS via MUBUF buffer_load_dwordx4 lds:1
extern "C" __device__ void __llvm_amdgcn_raw_buffer_load_lds(
    i32x4_t rsrc,
    __attribute__((address_space(3))) uint32_t *lds_ptr,
    int32_t size,
    int32_t voffset,
    int32_t soffset,
    int32_t imm_offset,
    int32_t aux) __asm("llvm.amdgcn.raw.buffer.load.lds");

// VGPR-target raw buffer load (16 bytes / dwordx4)
extern "C" __device__ i32x4_t __llvm_amdgcn_raw_buffer_load_v4i32(
    i32x4_t rsrc,
    int32_t voffset,
    int32_t soffset,
    int32_t aux) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

// Wave-uniform buffer resource descriptor (V#)
__device__ __forceinline__ i32x4_t make_w_buffer_rsrc(void const *base,
                                                      uint32_t range_bytes) {
  i32x4_t r;
  uint64_t addr = reinterpret_cast<uint64_t>(base);
  r[0] = static_cast<int>(__builtin_amdgcn_readfirstlane(
      static_cast<uint32_t>(addr & 0xFFFFFFFFu)));
  r[1] = static_cast<int>(
      __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(addr >> 32)));
  r[2] = static_cast<int>(__builtin_amdgcn_readfirstlane(range_bytes));
  r[3] = static_cast<int>(0x00020000u);
  return r;
}

// Convert bf16 (as unsigned short) to float
__device__ __forceinline__ float _gang_bf16_to_float(unsigned short b) {
  union {
    float f;
    unsigned u;
  } v;
  v.u = ((unsigned)(unsigned short)b) << 16;
  return v.f;
}

// Convert float to bf16 (as unsigned short) with rounding
__device__ __forceinline__ unsigned short _gang_float_to_bf16(float f) {
  union {
    float f;
    unsigned u;
  } v;
  v.f = f;
  unsigned rounding_bias = ((v.u >> 16) & 1) + 0x7FFF;
  return (unsigned short)((v.u + rounding_bias) >> 16);
}

// Compute E8M0 block scale for FP8 E4M3 (max value 448)
__device__ __forceinline__ uint8_t _gang_compute_e8m0_fp8(float amax) {
  if (amax == 0.0f) {
    return 0;
  }
  float target = amax * (1.0f / 448.0f);
  union {
    float f;
    uint32_t u;
  } v;
  v.f = target;
  int raw_exp = (int)((v.u >> 23) & 0xFF);
  if (v.u & 0x7FFFFF) {
    raw_exp++; // round up if mantissa non-zero
  }
  return (uint8_t)max(0, min(255, raw_exp));
}

// Quantize 4 floats to packed FP8 E4M3 using hardware cvt instruction
__device__ __forceinline__ fp8x4_t
    _gang_quant_4xfp8(float v0, float v1, float v2, float v3, float scale_f) {
  fp8x4_t pk = {};
  pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(pk, v0, v1, scale_f, false);
  pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(pk, v2, v3, scale_f, true);
  return pk;
}

// Quantize a block of BF16 values to FP8 E4M3 with E8M0 block scale
// n must be 128 (K_PER_MFMA)
__device__ __forceinline__ void
    _gang_quant_bf16_block_fp8(unsigned short const *__restrict__ src_bf16,
                               uint8_t *__restrict__ data_out,
                               uint8_t *__restrict__ scale_out,
                               int n) {
  float amax = 0.0f;
  for (int j = 0; j < n; j++) {
    float v = _gang_bf16_to_float(src_bf16[j]);
    float av = v < 0.0f ? -v : v;
    amax = amax > av ? amax : av;
  }

  uint8_t se = _gang_compute_e8m0_fp8(amax);
  *scale_out = se;
  float scale_f;
  if (se == 0) {
    scale_f = 1.0f;
  } else {
    union {
      float f;
      uint32_t u;
    } sv;
    sv.u = (uint32_t)se << 23;
    scale_f = sv.f;
  }

  for (int j = 0; j < n; j += 4) {
    float v0 = _gang_bf16_to_float(src_bf16[j]);
    float v1 = _gang_bf16_to_float(src_bf16[j + 1]);
    float v2 = _gang_bf16_to_float(src_bf16[j + 2]);
    float v3 = _gang_bf16_to_float(src_bf16[j + 3]);
    fp8x4_t pk = _gang_quant_4xfp8(v0, v1, v2, v3, scale_f);
    *(fp8x4_t *)(data_out + j) = pk;
  }
}

// Per-thread FP8 quantization mirroring FP4 structure.
// Splits each 128-element MFMA block into 4 × 32-element sub-blocks.
// Each thread handles one sub-block: load 32 bf16, find local amax,
// shuffle with 3 neighbors to get 128-element block amax, compute scale,
// pack 32 values to FP8. Single iteration for K=3072 (96 sub-blocks < 256
// threads).
template <int REDUCTION_SIZE>
__device__ __forceinline__ void
    _gang_wave_parallel_fp8_quant(unsigned short const *__restrict__ src_bf16,
                                  uint8_t *__restrict__ s_tok_fp8,
                                  uint8_t *__restrict__ s_tok_scales) {

  constexpr int SUB_BLOCK = 32;
  constexpr int NSUBBLOCKS = REDUCTION_SIZE / SUB_BLOCK; // 96 for K=3072
  int const tid = threadIdx.x;
  int const lane_id = tid & 63;

  for (int sb = tid; sb < NSUBBLOCKS; sb += blockDim.x) {
    int const base = sb * SUB_BLOCK;
    int const super_blk = sb / 4; // which 128-element block
    int const sub_idx = sb & 3;   // which sub-block within super-block

    // Load 32 bf16 values and find local amax (identical to FP4 quant)
    float vals[32];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 32; j++) {
      vals[j] = _gang_bf16_to_float(src_bf16[base + j]);
      amax = fmaxf(amax, fabsf(vals[j]));
    }

    // Combine amaxes from 4 sub-blocks sharing the same 128-element
    // super-block. Threads sb, sb+1, sb+2, sb+3 are consecutive lanes in the
    // same wave. Use __shfl to read each neighbor's amax (4 reads, 3 fmaxf).
    //
    // The partner index is clamped for the same reason as in the NT variant
    // below: NSUBBLOCKS = REDUCTION_SIZE/32 need not be a multiple of 4, so
    // the tail super-block would otherwise reduce against lanes whose loop
    // condition failed and whose `amax` register was never written.
    int base_lane = lane_id & ~3; // round down to group of 4
    int const sb_first = sb - sub_idx;
    int const n_valid = min(4, (NSUBBLOCKS - 1) - sb_first + 1);
    float a0 = __shfl(amax, base_lane);
    float a1 = __shfl(amax, base_lane + min(1, n_valid - 1));
    float a2 = __shfl(amax, base_lane + min(2, n_valid - 1));
    float a3 = __shfl(amax, base_lane + min(3, n_valid - 1));
    float block_amax = fmaxf(fmaxf(a0, a1), fmaxf(a2, a3));

    // Compute E8M0 scale
    uint8_t se = _gang_compute_e8m0_fp8(block_amax);
    float scale_f;
    if (se == 0) {
      scale_f = 1.0f;
    } else {
      union {
        float f;
        uint32_t u;
      } sv;
      sv.u = (uint32_t)se << 23;
      scale_f = sv.f;
    }

// Pack 32 values to FP8 (4 values → 4 bytes per cvt pair)
#pragma unroll
    for (int j = 0; j < 32; j += 4) {
      fp8x4_t pk = {};
      pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(
          pk, vals[j], vals[j + 1], scale_f, false);
      pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(
          pk, vals[j + 2], vals[j + 3], scale_f, true);
      *(int *)(s_tok_fp8 + base + j) = *(int const *)&pk;
    }

    // First sub-block writes scale byte for the super-block
    if (sub_idx == 0) {
      s_tok_scales[super_blk] = se;
    }
  }
  __syncthreads();
}

// ── Multi-row FP8 quant: stages up to 16 token rows for N-axis MFMA packing ──
//
// The MFMA is mfma_scale_f32_16x16x128_f8f6f4: M=16 weight rows, N=16 token
// columns, K=128. Lane l supplies A[m = l&15][k-blk = l>>4] and
// B[n = l&15][k-blk = l>>4], and receives D[m = (l>>4)*4 + i][n = l&15]. The
// single-token path broadcasts one token into all 16 N columns and reads back
// only n == 0, throwing away 15/16 of every MFMA. Staging token `col` at LDS
// row `col` lets up to 16 batch rows cost exactly what 1 row used to: same
// MFMA count, same weight traffic.
//
// Calling the single-row helper once per row would leave 164 of 256 threads
// idle on every call (REDUCTION_SIZE/32 = 92 sub-blocks) and cost one
// __syncthreads per row. Flattening (row, sub_block) into one strided loop
// keeps all 256 threads busy and needs a single barrier at the end.
//
// The __shfl amax combine requires the 4 sub-blocks of a 128-element super
// block to land in 4 consecutive lanes. Flattening preserves that: NSUB % 4
// == 0 keeps a super-block from straddling a row boundary, and idx = tid +
// n*256 preserves idx % 4 == tid % 4.
//
// `row_off(r)` returns the element offset of row r within src_bf16. It is a
// callable, not a table, so the contiguous callers (QKV / O-proj / LM head)
// can compute it in registers while MoE reads a gathered LDS table. Rows
// >= n_rows clamp so their loads stay in bounds; their N columns are masked
// off at the epilogue.
template <int REDUCTION_SIZE, int ROWS, int TOK_ROW_STRIDE, int SC_STRIDE,
          bool NT_LOAD, typename RowOff>
__device__ __forceinline__ void _gang_multirow_fp8_quant_impl(
    unsigned short const *__restrict__ src_bf16,
    RowOff row_off,
    int n_rows,
    uint8_t *__restrict__ s_tok_fp8,
    uint8_t *__restrict__ s_tok_scales) {

  constexpr int SUB_BLOCK = 32;
  constexpr int NSUB = REDUCTION_SIZE / SUB_BLOCK;
  constexpr int TOTAL = ROWS * NSUB;
  static_assert(NSUB % 4 == 0,
                "128-element super-blocks must not straddle a token row");
  static_assert(TOK_ROW_STRIDE % 16 == 0,
                "keeps the i32x4 B-operand loads 16B-aligned");

  int const tid = threadIdx.x;
  int const lane_id = tid & 63;

  for (int idx = tid; idx < TOTAL; idx += 256) {
    int const r = ROWS == 1 ? 0 : idx / NSUB;
    int const sb = idx - r * NSUB;
    int const base = sb * SUB_BLOCK;
    int const super_blk = sb >> 2;
    int const sub_idx = sb & 3;

    unsigned short const *row = src_bf16 + row_off(r < n_rows ? r : 0) + base;

    float vals[SUB_BLOCK];
    float amax = 0.0f;
    if constexpr (NT_LOAD) {
      // Non-temporal loads for cross-XCD reads (W2): the source was just
      // written by another XCD and will not be reused, so bypass L2.
#pragma unroll
      for (int j = 0; j < SUB_BLOCK; j += 8) {
        i32x4_t v4 = __builtin_nontemporal_load((i32x4_t const *)(row + j));
        unsigned short const *vs = (unsigned short const *)&v4;
#pragma unroll
        for (int q = 0; q < 8; q++) {
          vals[j + q] = _gang_bf16_to_float(vs[q]);
          amax = fmaxf(amax, fabsf(vals[j + q]));
        }
      }
    } else {
#pragma unroll
      for (int j = 0; j < SUB_BLOCK; j++) {
        vals[j] = _gang_bf16_to_float(row[j]);
        amax = fmaxf(amax, fabsf(vals[j]));
      }
    }

    int base_lane = lane_id & ~3;
    float a0 = __shfl(amax, base_lane);
    float a1 = __shfl(amax, base_lane + 1);
    float a2 = __shfl(amax, base_lane + 2);
    float a3 = __shfl(amax, base_lane + 3);
    float block_amax = fmaxf(fmaxf(a0, a1), fmaxf(a2, a3));

    uint8_t se = _gang_compute_e8m0_fp8(block_amax);
    float scale_f;
    if (se == 0) {
      scale_f = 1.0f;
    } else {
      union {
        float f;
        uint32_t u;
      } sv;
      sv.u = (uint32_t)se << 23;
      scale_f = sv.f;
    }

    uint8_t *dst = s_tok_fp8 + (ROWS == 1 ? 0 : r * TOK_ROW_STRIDE) + base;
#pragma unroll
    for (int j = 0; j < SUB_BLOCK; j += 4) {
      fp8x4_t pk = {};
      pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(
          pk, vals[j], vals[j + 1], scale_f, false);
      pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(
          pk, vals[j + 2], vals[j + 3], scale_f, true);
      *(int *)(dst + j) = *(int const *)&pk;
    }

    if (sub_idx == 0) {
      s_tok_scales[(ROWS == 1 ? 0 : r * SC_STRIDE) + super_blk] = se;
    }
  }
  __syncthreads();
}

// Gather form: `row_elem_off` is an LDS table of ROWS element offsets. This is
// what lets MoE quantize only the tokens routed to one expert, and lets W2
// address tok*(NUM_TOPK*INTERMEDIATE) + slot*INTERMEDIATE. The caller must
// have published the table (its own __syncthreads) before calling.
template <int REDUCTION_SIZE, int ROWS, int TOK_ROW_STRIDE, int SC_STRIDE,
          bool NT_LOAD = false>
__device__ __forceinline__ void _gang_multirow_fp8_quant_gather(
    unsigned short const *__restrict__ src_bf16,
    int const *__restrict__ row_elem_off,
    int n_rows,
    uint8_t *__restrict__ s_tok_fp8,
    uint8_t *__restrict__ s_tok_scales) {
  _gang_multirow_fp8_quant_impl<REDUCTION_SIZE, ROWS, TOK_ROW_STRIDE, SC_STRIDE,
                                NT_LOAD>(
      src_bf16, [&](int r) { return row_elem_off[r]; }, n_rows, s_tok_fp8,
      s_tok_scales);
}

// Contiguous form: rows [row_base, row_base + ROWS) of a plain
// [MAX_ROW, src_row_stride] bf16 matrix. Used by QKV / O-proj / LM head, which
// read straight from a norm buffer and need no gather. The offset is a
// register computation, so unlike the gather form this costs no LDS and no
// extra barrier -- at ROWS == 1 it reduces to the same address the old
// single-row quantizer formed.
template <int REDUCTION_SIZE, int ROWS, int MAX_ROW, int TOK_ROW_STRIDE,
          int SC_STRIDE, bool NT_LOAD = false>
__device__ __forceinline__ void _gang_multirow_fp8_quant(
    unsigned short const *__restrict__ src_bf16,
    int src_row_stride,
    int row_base,
    int n_rows,
    uint8_t *__restrict__ s_tok_fp8,
    uint8_t *__restrict__ s_tok_scales) {
  // Clamp: BATCH_SIZE need not be a multiple of 16, so the last column block
  // can address rows past the end of the source. Those N columns are masked
  // off by tok_active at the epilogue; here the read only has to stay in
  // bounds.
  _gang_multirow_fp8_quant_impl<REDUCTION_SIZE, ROWS, TOK_ROW_STRIDE, SC_STRIDE,
                                NT_LOAD>(
      src_bf16,
      [&](int r) {
        int src_row = row_base + r;
        src_row = src_row < MAX_ROW ? src_row : MAX_ROW - 1;
        return src_row * src_row_stride;
      },
      n_rows, s_tok_fp8, s_tok_scales);
}

// NT-load variant of per-thread FP8 quantization for W2 cross-XCD reads.
// Same sub-block structure as non-NT variant but uses dwordx4 NT loads.
template <int REDUCTION_SIZE>
__device__ __forceinline__ void _gang_wave_parallel_fp8_quant_nt(
    unsigned short const *__restrict__ src_bf16,
    uint8_t *__restrict__ s_tok_fp8,
    uint8_t *__restrict__ s_tok_scales) {

  constexpr int SUB_BLOCK = 32;
  constexpr int NSUBBLOCKS = REDUCTION_SIZE / SUB_BLOCK; // 96 for K=3072
  int const tid = threadIdx.x;
  int const lane_id = tid & 63;
  uint32_t const *src32 = (uint32_t const *)src_bf16;

  for (int sb = tid; sb < NSUBBLOCKS; sb += blockDim.x) {
    int const base = sb * SUB_BLOCK;
    int const super_blk = sb / 4;
    int const sub_idx = sb & 3;
    uint32_t const *base_ptr = src32 + base / 2;

    // 4 wide NT loads (64 bytes = 32 bf16)
    //
    // The outputs MUST be early-clobber ("=&v"). This is one asm block with
    // four separate instructions, so the compiler is free to allocate an
    // output register on top of an input it believes is dead after the
    // block -- and it does: without the '&' it emits
    //     global_load_dwordx4 v[4:7],   v[4:5],  ...
    //     global_load_dwordx4 v[8:11],  v[6:7],  ...
    //     global_load_dwordx4 v[12:15], v[8:9],  ...
    //     global_load_dwordx4 v[16:19], v[10:11],...
    // where the first load's destination overwrites the address operands of
    // the next three before they issue. Those loads then use whatever the
    // returned data happened to be as an address. A wild address that never
    // completes leaves the wave parked on the s_waitcnt vmcnt(0) below
    // forever, which hangs the __syncthreads at the end of this function and
    // through it the whole block -- the captured deadlock is exactly that:
    // wave 0 missing from the quant sync mask (0xe) while every wave had
    // already cleared the W13->W2 barrier (0xf).
    uint32_t dw[16];
    asm volatile("global_load_dwordx4 %0, %4, off sc0 sc1 nt\n"
                 "global_load_dwordx4 %1, %5, off sc0 sc1 nt\n"
                 "global_load_dwordx4 %2, %6, off sc0 sc1 nt\n"
                 "global_load_dwordx4 %3, %7, off sc0 sc1 nt"
                 : "=&v"(*(i32x4_t *)&dw[0]),
                   "=&v"(*(i32x4_t *)&dw[4]),
                   "=&v"(*(i32x4_t *)&dw[8]),
                   "=&v"(*(i32x4_t *)&dw[12])
                 : "v"(base_ptr),
                   "v"(base_ptr + 4),
                   "v"(base_ptr + 8),
                   "v"(base_ptr + 12)
                 : "memory");
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

    // Convert to float and find amax
    float vals[32];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; j++) {
      float lo = _gang_bf16_to_float((unsigned short)(dw[j] & 0xFFFF));
      float hi = _gang_bf16_to_float((unsigned short)(dw[j] >> 16));
      vals[j * 2] = lo;
      vals[j * 2 + 1] = hi;
      amax = fmaxf(amax, fmaxf(fabsf(lo), fabsf(hi)));
    }

    // Combine amaxes across the 4 sub-blocks of this 128-element super-block.
    //
    // NSUBBLOCKS is REDUCTION_SIZE/32 and is NOT guaranteed to be a multiple
    // of 4: for the W2 path REDUCTION_SIZE = INTERMEDIATE_SIZE = 2880, giving
    // NSUBBLOCKS = 90. The last super-block therefore has only 2 real
    // sub-blocks (sb 88, 89), but the shuffles below still read lanes for
    // sb 90 and 91 -- threads whose loop condition failed, so their `amax`
    // is an uninitialized register. Clamping the partner index to the last
    // valid sub-block makes the reduction read only lanes that ran.
    int base_lane = lane_id & ~3;
    int const sb_first = sb - sub_idx;  // first sb of this super-block
    int const sb_last = NSUBBLOCKS - 1; // last sb that actually runs
    int const n_valid = min(4, sb_last - sb_first + 1);
    float a0 = __shfl(amax, base_lane);
    float a1 = __shfl(amax, base_lane + min(1, n_valid - 1));
    float a2 = __shfl(amax, base_lane + min(2, n_valid - 1));
    float a3 = __shfl(amax, base_lane + min(3, n_valid - 1));
    float block_amax = fmaxf(fmaxf(a0, a1), fmaxf(a2, a3));

    // Compute E8M0 scale
    uint8_t se = _gang_compute_e8m0_fp8(block_amax);
    float scale_f;
    if (se == 0) {
      scale_f = 1.0f;
    } else {
      union {
        float f;
        uint32_t u;
      } sv;
      sv.u = (uint32_t)se << 23;
      scale_f = sv.f;
    }

// Pack 32 values to FP8
#pragma unroll
    for (int j = 0; j < 32; j += 4) {
      fp8x4_t pk = {};
      pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(
          pk, vals[j], vals[j + 1], scale_f, false);
      pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(
          pk, vals[j + 2], vals[j + 3], scale_f, true);
      *(int *)(s_tok_fp8 + base + j) = *(int const *)&pk;
    }

    // First sub-block writes scale byte
    if (sub_idx == 0) {
      s_tok_scales[super_blk] = se;
    }
  }
  // Record that this wave finished its strided share and is entering the
  // block-wide sync. If a stall shows a mask below 0xf here, the missing
  // wave never got out of the loop above -- almost certainly stuck on the
  // vmcnt(0) that drains its NT loads of d_swiglu_out.
  MPK_WS_WAVE_SYNC(tid >> 6);
  __syncthreads();
}

// ── Wide FP8 quant: one lane per 16 values instead of one per 32 ─────────────
//
// The two functions above give each active thread a 32-value serial chain and
// derive the 128-value E8M0 domain from four consecutive lanes via __shfl. For
// K = 2880/3072 that activates only K/32 = 90..96 of the 256 threads: the other
// ~160 sit out the whole quant, and the ones that do work run 32 bf16->f32
// converts, 32 fabs/fmax, and 16 cvt_scalef32 back to back.
//
// The wide mapping halves the per-lane chain (16 values) and doubles the active
// lanes (K/16 = 180..192), so the same K is quantized in half the serial steps.
// The scale domain is unchanged at 128 values -- it is now spread over eight
// lanes instead of four, so the reduction needs one more step. Three DPP row
// ops do it without LDS and without __shfl's lane-index arithmetic:
// quad_perm[1,0,3,2] and quad_perm[2,3,0,1] reduce within each group of four,
// then row_half_mirror exchanges the two already-broadcast halves. Every lane
// in the group of eight ends up holding the full amax, so no broadcast pass is
// needed afterwards.
//
// Unlike the __shfl variants there is no tail-clamping logic here: the
// static_assert below requires K to be a whole number of 128-value domains, so
// no group of eight ever straddles the end of the row and every lane a DPP op
// reads is a lane that ran.
//
// Numerically identical to the narrow variants: same E8M0 exponent rule over
// the same 128 values, same cvt instruction, same packing order.
template <int REDUCTION_SIZE>
__device__ __forceinline__ void
    _gang_wave_parallel_fp8_quant_wide(unsigned short const *__restrict__ src_bf16,
                                       uint8_t *__restrict__ s_tok_fp8,
                                       uint8_t *__restrict__ s_tok_scales) {
  constexpr int VALUES_PER_LANE = 16;
  constexpr int LANES_PER_SCALE = 8;
  constexpr int VALUES_PER_SCALE = VALUES_PER_LANE * LANES_PER_SCALE; // 128
  static_assert(REDUCTION_SIZE % VALUES_PER_SCALE == 0,
                "wide FP8 quant needs K to be a multiple of 128");

  int const tid = threadIdx.x;
  constexpr int ACTIVE_LANES = REDUCTION_SIZE / VALUES_PER_LANE;
  if (tid < ACTIVE_LANES) {
    int const base = tid * VALUES_PER_LANE;
    float vals[VALUES_PER_LANE];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < VALUES_PER_LANE; i++) {
      vals[i] = _gang_bf16_to_float(src_bf16[base + i]);
      amax = fmaxf(amax, fabsf(vals[i]));
    }

    // `peer` must be early-clobber: it is written before the last read of
    // `amax` in each pair, so sharing a register with the input corrupts it.
    float peer;
    asm volatile("s_nop 1\n"
                 "v_mov_b32_dpp %1, %0 quad_perm:[1,0,3,2] row_mask:0xf "
                 "bank_mask:0xf\n"
                 "v_max_f32 %0, %0, %1\n"
                 "s_nop 1\n"
                 "v_mov_b32_dpp %1, %0 quad_perm:[2,3,0,1] row_mask:0xf "
                 "bank_mask:0xf\n"
                 "v_max_f32 %0, %0, %1\n"
                 "s_nop 1\n"
                 "v_mov_b32_dpp %1, %0 row_half_mirror row_mask:0xf "
                 "bank_mask:0xf\n"
                 "v_max_f32 %0, %0, %1"
                 : "+v"(amax), "=&v"(peer));

    uint8_t const se = _gang_compute_e8m0_fp8(amax);
    float scale_f = 1.0f;
    if (se != 0) {
      uint32_t const bits = (uint32_t)se << 23;
      __builtin_memcpy(&scale_f, &bits, sizeof(scale_f));
    }

#pragma unroll
    for (int i = 0; i < VALUES_PER_LANE; i += 4) {
      fp8x4_t pk = {};
      pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(
          pk, vals[i], vals[i + 1], scale_f, false);
      pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(
          pk, vals[i + 2], vals[i + 3], scale_f, true);
      __builtin_memcpy(s_tok_fp8 + base + i, &pk, sizeof(pk));
    }

    if ((tid & (LANES_PER_SCALE - 1)) == 0) {
      s_tok_scales[tid / LANES_PER_SCALE] = se;
    }
  }
  __syncthreads();
}

// NT-load counterpart of the wide mapping, for the W2 input. The SwiGLU output
// was written by another XCD and is never reused, so the loads keep sc0/sc1/nt
// exactly as the narrow NT variant does. Two dwordx4 cover a lane's 16 bf16.
//
// The "=&v" early-clobber is load-bearing for the same reason spelled out at
// length in _gang_wave_parallel_fp8_quant_nt above: without it the allocator
// puts the first load's destination on top of the second's address operand.
template <int REDUCTION_SIZE>
__device__ __forceinline__ void _gang_wave_parallel_fp8_quant_nt_wide(
    unsigned short const *__restrict__ src_bf16,
    uint8_t *__restrict__ s_tok_fp8,
    uint8_t *__restrict__ s_tok_scales) {
  constexpr int VALUES_PER_LANE = 16;
  constexpr int LANES_PER_SCALE = 8;
  constexpr int VALUES_PER_SCALE = VALUES_PER_LANE * LANES_PER_SCALE; // 128
  static_assert(REDUCTION_SIZE % VALUES_PER_SCALE == 0,
                "wide FP8 quant needs K to be a multiple of 128");

  int const tid = threadIdx.x;
  constexpr int ACTIVE_LANES = REDUCTION_SIZE / VALUES_PER_LANE;
  if (tid < ACTIVE_LANES) {
    int const base = tid * VALUES_PER_LANE;
    uint32_t const *base_ptr = (uint32_t const *)(src_bf16 + base);
    uint32_t words[8];
    asm volatile("global_load_dwordx4 %0, %2, off sc0 sc1 nt\n"
                 "global_load_dwordx4 %1, %3, off sc0 sc1 nt"
                 : "=&v"(*(i32x4_t *)&words[0]), "=&v"(*(i32x4_t *)&words[4])
                 : "v"(base_ptr), "v"(base_ptr + 4)
                 : "memory");
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

    float vals[VALUES_PER_LANE];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < VALUES_PER_LANE / 2; i++) {
      float const lo = _gang_bf16_to_float((unsigned short)(words[i] & 0xFFFF));
      float const hi = _gang_bf16_to_float((unsigned short)(words[i] >> 16));
      vals[i * 2] = lo;
      vals[i * 2 + 1] = hi;
      amax = fmaxf(amax, fmaxf(fabsf(lo), fabsf(hi)));
    }

    float peer;
    asm volatile("s_nop 1\n"
                 "v_mov_b32_dpp %1, %0 quad_perm:[1,0,3,2] row_mask:0xf "
                 "bank_mask:0xf\n"
                 "v_max_f32 %0, %0, %1\n"
                 "s_nop 1\n"
                 "v_mov_b32_dpp %1, %0 quad_perm:[2,3,0,1] row_mask:0xf "
                 "bank_mask:0xf\n"
                 "v_max_f32 %0, %0, %1\n"
                 "s_nop 1\n"
                 "v_mov_b32_dpp %1, %0 row_half_mirror row_mask:0xf "
                 "bank_mask:0xf\n"
                 "v_max_f32 %0, %0, %1"
                 : "+v"(amax), "=&v"(peer));

    uint8_t const se = _gang_compute_e8m0_fp8(amax);
    float scale_f = 1.0f;
    if (se != 0) {
      uint32_t const bits = (uint32_t)se << 23;
      __builtin_memcpy(&scale_f, &bits, sizeof(scale_f));
    }

#pragma unroll
    for (int i = 0; i < VALUES_PER_LANE; i += 4) {
      fp8x4_t pk = {};
      pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(
          pk, vals[i], vals[i + 1], scale_f, false);
      pk = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(
          pk, vals[i + 2], vals[i + 3], scale_f, true);
      __builtin_memcpy(s_tok_fp8 + base + i, &pk, sizeof(pk));
    }

    if ((tid & (LANES_PER_SCALE - 1)) == 0) {
      s_tok_scales[tid / LANES_PER_SCALE] = se;
    }
  }
  MPK_WS_WAVE_SYNC(tid >> 6);
  __syncthreads();
}

// FP4×FP8 scaled MFMA: 16x16x128, hardware dequant + multiply
// A = weights (FP4 E2M1), 16 bytes/lane in lower 128 bits of i32x8
// B = tokens  (FP8 E4M3), 32 bytes/lane split across i32x8
__device__ __forceinline__ f32x4_t _gang_mfma_f4xf8(
    i32x8_t a, i32x8_t b, f32x4_t c, int scale_a, int scale_b) {
  return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
      a,
      b,
      c,
      4, // cbsz: FP4 E2M1 for src0 (weights)
      0, // blgp: FP8 E4M3 for src1 (tokens)
      0,
      scale_a,
      0,
      scale_b);
}

// Load FP8 B-operand for 16x16x128 MFMA.
// FP8 layout requires two 16-byte chunks at K offsets [g*16..g*16+15]
// and [g*16+64..g*16+79] within the 128-element tile.
__device__ __forceinline__ i32x8_t _gang_load_fp8_mfma_b(uint8_t const *data,
                                                         int kt,
                                                         int g) {
  i32x4_t lo = *(i32x4_t const *)(data + kt + g * 16);
  i32x4_t hi = *(i32x4_t const *)(data + kt + g * 16 + 64);
  i32x8_t r;
  r[0] = lo[0];
  r[1] = lo[1];
  r[2] = lo[2];
  r[3] = lo[3];
  r[4] = hi[0];
  r[5] = hi[1];
  r[6] = hi[2];
  r[7] = hi[3];
  return r;
}

// ── FP4×FP4 helpers (gfx950) ─────────────────────────────────────────────

// Compute E8M0 block scale for FP4 (max representable value = 6.0)
// Uses multiply by reciprocal (1 v_mul_f32) instead of division
// (Newton-Raphson).
__device__ __forceinline__ uint8_t _gang_compute_e8m0_fp4(float amax) {
  if (amax == 0.0f) {
    return 0;
  }
  union {
    float f;
    uint32_t u;
  } v;
  v.f = amax * (1.0f / 6.0f); // v_mul_f32, not v_div (Newton-Raphson)
  int raw_exp = (int)((v.u >> 23) & 0xFF);
  if (v.u & 0x7FFFFF) {
    raw_exp++; // round up if mantissa non-zero
  }
  return (uint8_t)max(0, min(255, raw_exp));
}

// Per-thread FP4 quantization of bf16 input into LDS.
// 256 threads, K/32 blocks (96 for K=3072). Each thread handles one 32-element
// block per iteration. Writes nibble-packed FP4 data + E8M0 scales.
// LDS layout: [K/2 bytes data | K/32 bytes scales]
template <int REDUCTION_SIZE>
__device__ __forceinline__ void
    _gang_fp4_quant(unsigned short const *__restrict__ src_bf16,
                    uint8_t *__restrict__ s_tok_fp4,
                    uint8_t *__restrict__ s_tok_scales) {

  constexpr int BLOCK_SIZE = 32;
  constexpr int NBLOCKS = REDUCTION_SIZE / BLOCK_SIZE;
  int const tid = threadIdx.x;

  for (int blk = tid; blk < NBLOCKS; blk += blockDim.x) {
    int base = blk * BLOCK_SIZE;

    // Load 32 bf16 values and find amax using fmaxf/fabsf (v_max_f32, no VCC
    // stalls)
    float vals[32];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 32; j++) {
      vals[j] = _gang_bf16_to_float(src_bf16[base + j]);
      amax = fmaxf(amax, fabsf(vals[j]));
    }

    // Compute E8M0 scale via multiply + bit-shift (no division or exp2f)
    uint8_t se = _gang_compute_e8m0_fp4(amax);
    s_tok_scales[blk] = se;
    float scale_f;
    if (se == 0) {
      scale_f = 1.0f;
    } else {
      union {
        float f;
        uint32_t u;
      } sv;
      sv.u = (uint32_t)se << 23; // 2^(se-127) via bit construction
      scale_f = sv.f;
    }

// Pack to FP4 using hardware cvt instruction (4 values → 2 bytes)
#pragma unroll
    for (int j = 0; j < 32; j += 4) {
      unsigned pk = 0;
      pk = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
          pk, vals[j], vals[j + 1], scale_f, false);
      pk = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
          pk, vals[j + 2], vals[j + 3], scale_f, true);
      *(uint16_t *)(s_tok_fp4 + base / 2 + j / 2) = (uint16_t)pk;
    }
  }
  __syncthreads();
}

// NT-load variant of FP4 quant for W2 cross-XCD reads.
// Per-thread: each thread handles one 32-element block per iteration.
// Optimized: 4×dwordx4 NT loads, fmaxf/fabsf amax, multiply+bit-shift E8M0.
template <int REDUCTION_SIZE>
__device__ __forceinline__ void
    _gang_fp4_quant_nt(unsigned short const *__restrict__ src_bf16,
                       uint8_t *__restrict__ s_tok_fp4,
                       uint8_t *__restrict__ s_tok_scales) {

  constexpr int BLOCK_SIZE = 32;
  constexpr int NBLOCKS = REDUCTION_SIZE / BLOCK_SIZE;
  int const tid = threadIdx.x;
  uint32_t const *src32 = (uint32_t const *)src_bf16;

  for (int blk = tid; blk < NBLOCKS; blk += blockDim.x) {
    int base = blk * BLOCK_SIZE;
    uint32_t const *base_ptr = src32 + base / 2;

    // 4 wide NT loads (64 bytes = 32 bf16) instead of 16 individual dword loads
    // Early-clobber outputs are required here for the same reason as in
    // _gang_wave_parallel_fp8_quant_nt: without '&' the allocator puts the
    // first load's destination on top of the later loads' address registers.
    uint32_t dw[16];
    asm volatile("global_load_dwordx4 %0, %4, off sc0 sc1 nt\n"
                 "global_load_dwordx4 %1, %5, off sc0 sc1 nt\n"
                 "global_load_dwordx4 %2, %6, off sc0 sc1 nt\n"
                 "global_load_dwordx4 %3, %7, off sc0 sc1 nt"
                 : "=&v"(*(i32x4_t *)&dw[0]),
                   "=&v"(*(i32x4_t *)&dw[4]),
                   "=&v"(*(i32x4_t *)&dw[8]),
                   "=&v"(*(i32x4_t *)&dw[12])
                 : "v"(base_ptr),
                   "v"(base_ptr + 4),
                   "v"(base_ptr + 8),
                   "v"(base_ptr + 12)
                 : "memory");
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

    // Convert to float and find amax using fmaxf/fabsf (v_max_f32, no VCC
    // stalls)
    float vals[32];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; j++) {
      float lo = _gang_bf16_to_float((unsigned short)(dw[j] & 0xFFFF));
      float hi = _gang_bf16_to_float((unsigned short)(dw[j] >> 16));
      vals[j * 2] = lo;
      vals[j * 2 + 1] = hi;
      amax = fmaxf(amax, fmaxf(fabsf(lo), fabsf(hi)));
    }

    // Compute E8M0 scale via multiply + bit-shift (no division or exp2f)
    uint8_t se = _gang_compute_e8m0_fp4(amax);
    s_tok_scales[blk] = se;
    float scale_f;
    if (se == 0) {
      scale_f = 1.0f;
    } else {
      union {
        float f;
        uint32_t u;
      } sv;
      sv.u = (uint32_t)se << 23; // 2^(se-127) via bit construction
      scale_f = sv.f;
    }

// Pack to FP4
#pragma unroll
    for (int j = 0; j < 32; j += 4) {
      unsigned pk = 0;
      pk = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
          pk, vals[j], vals[j + 1], scale_f, false);
      pk = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
          pk, vals[j + 2], vals[j + 3], scale_f, true);
      *(uint16_t *)(s_tok_fp4 + base / 2 + j / 2) = (uint16_t)pk;
    }
  }
  __syncthreads();
}

// Load FP4 B-operand for 16x16x128 MFMA.
// FP4 uses only the lower 128 bits (16 bytes) of i32x8 — contiguous load.
// Data layout: [K/2 bytes], offset = kt/2 + g*16
__device__ __forceinline__ i32x8_t _gang_load_fp4_mfma_b(uint8_t const *data,
                                                         int kt,
                                                         int g) {
  i32x4_t lo = *(i32x4_t const *)(data + kt / 2 + g * 16);
  i32x8_t r = {};
  r[0] = lo[0];
  r[1] = lo[1];
  r[2] = lo[2];
  r[3] = lo[3];
  return r;
}

// FP4×FP4 scaled MFMA: 16x16x128, 16 cycles (half of FP4×FP8)
// Both A and B are FP4 in lower 128 bits of i32x8
__device__ __forceinline__ f32x4_t _gang_mfma_f4xf4(
    i32x8_t a, i32x8_t b, f32x4_t c, int scale_a, int scale_b) {
  return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
      a,
      b,
      c,
      4, // cbsz: FP4 for src0 (weights)
      4, // blgp: FP4 for src1 (tokens)
      0,
      scale_a,
      0,
      scale_b);
}

// ── BF16 MFMA helpers (W4A16 path: FP4 weights dequanted to BF16) ─────────

// BF16×BF16 MFMA: v_mfma_f32_16x16x32_bf16
// A = weights (BF16, dequanted from FP4), 8 BF16 values per lane
// B = tokens  (BF16, native), 8 BF16 values per lane
// K=32 per instruction (vs K=128 for FP4/FP8)
typedef __bf16 __attribute__((ext_vector_type(2))) bf16x2_t;
typedef __bf16 __attribute__((ext_vector_type(8))) bf16x8_t;

__device__ __forceinline__ f32x4_t _gang_mfma_bf16(bf16x8_t a,
                                                   bf16x8_t b,
                                                   f32x4_t c) {
  return __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c, 0, 0, 0);
}

// Hardware FP4→BF16 dequant using v_cvt_scalef32_pk_bf16_fp4 (gfx950).
// 4 instructions produce bf16x8_t (vs ~40 ALU instructions with software LUT).
// Each instruction converts 1 byte (2 FP4 nibbles) → 2 BF16 values, scaled by
// E8M0.
//
// fp4_data: 4 bytes (8 nibbles) for this lane's K-group
// scale_e8m0: E8M0 block scale byte (1 per 32 K-elements)
// Returns: bf16x8_t with 8 BF16 values
__device__ __forceinline__ bf16x8_t _gang_dequant_fp4_to_bf16_8(
    uint8_t const *__restrict__ fp4_data, uint8_t scale_e8m0) {

  uint32_t raw_fp4;
  __builtin_memcpy(&raw_fp4, fp4_data, 4);
  // Scale: reinterpret E8M0 byte as float exponent (2^(e8m0-127))
  float scale;
  uint32_t su = (uint32_t)scale_e8m0 << 23;
  __builtin_memcpy(&scale, &su, 4);

  // v_cvt_scalef32_pk_bf16_fp4: converts 2 FP4 nibbles (1 byte) → 2 BF16 values
  // word_sel 0..3 selects byte 0..3 of the 32-bit source
  union {
    bf16x2_t h[4];
    bf16x8_t v;
  } r;
  r.h[0] = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(raw_fp4, scale, 0);
  r.h[1] = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(raw_fp4, scale, 1);
  r.h[2] = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(raw_fp4, scale, 2);
  r.h[3] = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(raw_fp4, scale, 3);
  return r.v;
}

// Hardware FP4→BF16 dequant from raw pre-loaded FP4 data (no memory access).
// raw_fp4: 4 bytes (8 nibbles = 8 E2M1 values), already loaded
// scale_e8m0: E8M0 block scale byte (already loaded)
__device__ __forceinline__ bf16x8_t
    _gang_dequant_raw_fp4_to_bf16_8(uint32_t raw_fp4, uint8_t scale_e8m0) {

  float scale;
  uint32_t su = (uint32_t)scale_e8m0 << 23;
  __builtin_memcpy(&scale, &su, 4);

  union {
    bf16x2_t h[4];
    bf16x8_t v;
  } r;
  r.h[0] = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(raw_fp4, scale, 0);
  r.h[1] = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(raw_fp4, scale, 1);
  r.h[2] = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(raw_fp4, scale, 2);
  r.h[3] = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(raw_fp4, scale, 3);
  return r.v;
}

// Load BF16 token B-operand for v_mfma_f32_16x16x32_bf16.
// Each lane loads 8 consecutive BF16 values for its K-group.
// Data layout: [K * 2 bytes] (BF16 native, 2 bytes per element)
// g = lane_id >> 4 (0..3), each group covers K positions [g*8..g*8+7]
__device__ __forceinline__ bf16x8_t
    _gang_load_bf16_mfma_b(unsigned short const *data, int kt, int g) {
  bf16x8_t r;
  __builtin_memcpy(&r, data + kt + g * 8, sizeof(bf16x8_t));
  return r;
}

// ── Gang MoE MXFP4 kernel (gfx950 hardware FP4×FP8 MFMA path) ───────────
// Handles both W13 and W2 projections.
// Each gang task processes tiles assigned to its XCD.
//
// FUSE_SWIGLU: When true (only valid with W13_LINEAR=true), fuses SwiGLU
// activation into the epilogue. The output has interleaved gate/up pairs
// (acc[0]=gate, acc[1]=up, acc[2]=gate, acc[3]=up). The epilogue applies
// fast_swigluoai(gate+bias, up+bias) and writes half the output (2 values
// per lane instead of 4). Output tensor is [bs, topk, OUTPUT_SIZE/2].
template <int BATCH_SIZE,
          int OUTPUT_SIZE,
          int OUTPUT_STRIDE,
          int REDUCTION_SIZE,
          int NUM_EXPERTS,
          int NUM_TOPK,
          int TILES_PER_EXPERT,
          int OUTPUT_PER_WG,
          bool W13_LINEAR,
          bool FUSE_SWIGLU = false>
__device__ __noinline__ void
    gang_moe_linear_mxfp4_kernel_mi300(void const *input_ptr,
                                       void const *weight_ptr,
                                       void const *routing_ptr,
                                       void const *mask_ptr,
                                       void const *bias_ptr,
                                       void *output_ptr,
                                       int tile_idx) {
  static_assert(OUTPUT_PER_WG % 16 == 0,
                "OUTPUT_PER_WG must be multiple of 16");
  static_assert(REDUCTION_SIZE % 128 == 0,
                "REDUCTION_SIZE must be multiple of 128 for FP4 MFMA");

  // MXFP4 weight layout constants (unchanged from original)
  constexpr int NUM_BLOCKS_32 =
      REDUCTION_SIZE / 32; // number of 32-element scale blocks
  constexpr int WG_DATA_BYTES =
      OUTPUT_PER_WG * (REDUCTION_SIZE / 2); // FP4 nibble-packed
  constexpr int WG_SCALE_BYTES = OUTPUT_PER_WG * NUM_BLOCKS_32; // E8M0 scales
  constexpr int WG_BYTES = WG_DATA_BYTES + WG_SCALE_BYTES;
  constexpr int EXPERT_WGS = OUTPUT_STRIDE / OUTPUT_PER_WG;
  constexpr int64_t EXPERT_BYTES = static_cast<int64_t>(EXPERT_WGS) * WG_BYTES;

  // Hardware MFMA constants
  constexpr int K_PER_MFMA =
      128; // FP4 MFMA processes 128 K-elements per instruction
  constexpr int MFMA_ITERS = REDUCTION_SIZE / K_PER_MFMA; // e.g. 3072/128 = 24

  // N-parallel: 4 waves, each handles 16 output rows
  constexpr int NUM_WAVES = 4;
  constexpr int N_TILES = OUTPUT_PER_WG / 16;
  constexpr int TILES_PER_WAVE = N_TILES / NUM_WAVES;
  constexpr int N_TILES_PER_WG = EXPERT_WGS;

  // FP4 token storage in shared memory: data + scales
  constexpr int FP4_TOK_DATA =
      REDUCTION_SIZE / 2; // 0.5 byte per element (nibble-packed)
  constexpr int FP4_TOK_SCALES =
      REDUCTION_SIZE / 32; // 1 E8M0 scale per 32-element block
  constexpr int FP4_TOK_BYTES = FP4_TOK_DATA + FP4_TOK_SCALES;

  unsigned short const *A = (unsigned short const *)input_ptr;
  uint8_t const *W = (uint8_t const *)weight_ptr;
  int const *d_routing = (int const *)routing_ptr;
  int const *d_mask = (int const *)mask_ptr;
  unsigned short const *d_bias = (unsigned short const *)bias_ptr;
  unsigned short *d_output = (unsigned short *)output_ptr;

  extern __shared__ char _gang_mxfp4_smem[];
  // Shared memory layout: FP4 quantized tokens + scales
  uint8_t *s_tok_fp4 = (uint8_t *)_gang_mxfp4_smem;
  uint8_t *s_tok_scales = s_tok_fp4 + FP4_TOK_DATA;

  int const tid = threadIdx.x;
  int const warp_id = tid >> 6; // wave ID (0..3)
  int const lane_id = tid & 63;
  int const col = lane_id & 15; // output row within 16x16 MFMA tile (N-dim)
  int const g = lane_id >> 4;   // K-group (0..3), each handles 32 FP4 nibbles

  // Get XCD ID for flat global tile distribution
  int xcd_id = _gang_moe_get_xcd_id();

  int const num_activated_experts = d_mask[NUM_EXPERTS];

  // Flat global tile distribution across all 8 XCDs.
  // Interleave tiles round-robin so all XCDs stay active even when
  // num_activated_experts < 8 (e.g. top-4 at bs=1).
  int global_tile = tile_idx * 8 + xcd_id;
  int total_tiles = num_activated_experts * TILES_PER_EXPERT;
  if (global_tile >= total_tiles) {
    return;
  }
  int expert_idx = global_tile / TILES_PER_EXPERT;
  int tile_within_expert = global_tile % TILES_PER_EXPERT;
  int expert_id = d_mask[expert_idx];
  int const *expert_routing = d_routing + expert_id * BATCH_SIZE;

  // Expert weight base
  uint8_t const *expert_weight =
      W + static_cast<int64_t>(expert_id) * EXPERT_BYTES;

  // Decode tile position: (token_idx, wg_idx)
  int tok_idx = tile_within_expert / N_TILES_PER_WG;
  int wg_idx = tile_within_expert % N_TILES_PER_WG;

  if (tok_idx >= BATCH_SIZE) {
    return;
  }

  // Check routing
  int route_val = expert_routing[tok_idx];
  if (route_val == 0) {
    return;
  }
  int topk_slot = route_val - 1;

  // Workgroup weight pointers (FP4 data followed by E8M0 scales)
  uint8_t const *wg_data =
      expert_weight + static_cast<int64_t>(wg_idx) * WG_BYTES;
  uint8_t const *wg_scales = wg_data + WG_DATA_BYTES;

  // ── Phase 1: Load BF16 input and quantize to FP4 in shared memory ──────
  unsigned short const *input_base;
  if constexpr (W13_LINEAR) {
    input_base = A + tok_idx * REDUCTION_SIZE;
  } else {
    input_base =
        A + tok_idx * (NUM_TOPK * REDUCTION_SIZE) + topk_slot * REDUCTION_SIZE;
  }

  _gang_fp4_quant<REDUCTION_SIZE>(input_base, s_tok_fp4, s_tok_scales);

  // ── Phase 2: MFMA FP4(weights) × FP4(tokens) ─────────────────────────
  // Distribute MFMA K-iterations across waves evenly
  constexpr int base_iters = MFMA_ITERS / NUM_WAVES;
  constexpr int extra_iters = MFMA_ITERS % NUM_WAVES;

  for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; tile_iter++) {
    int wave_tile = warp_id + tile_iter * NUM_WAVES;
    // MFMA 16x16: col (lane%16) = batch dim, g (lane/16) = output group (4 rows
    // each)

    f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

// K-reduction loop: each MFMA processes 128 K-elements
#pragma unroll 1
    for (int kt = 0; kt < REDUCTION_SIZE; kt += K_PER_MFMA) {
      // Load FP4 weight A-operand: 16 bytes per lane (32 FP4 nibbles)
      int w_row = wave_tile * 16 + col;
      int a_off = w_row * (REDUCTION_SIZE / 2) + kt / 2 + g * 16;
      i32x8_t a_reg = *(i32x8_t const *)(wg_data + a_off);

      // Load FP4 token B-operand: contiguous 16B per lane
      i32x8_t b_reg = _gang_load_fp4_mfma_b(s_tok_fp4, kt, g);

      // Weight scale: E8M0 per 32-element block
      int sa = (int)wg_scales[w_row * NUM_BLOCKS_32 + kt / 32 + g];

      // Token scale: E8M0 per 32-element block (4 per 128 K-elements)
      int sb = (int)s_tok_scales[kt / 32 + g];

      // Hardware MFMA: FP4×FP4, 16 cycles
      acc = _gang_mfma_f4xf4(a_reg, b_reg, acc, sa, sb);
    }

    // ── Epilogue: write result with bias ────────────────────────────────
    // MFMA C layout: col (lane_id & 15) = batch dimension, g (lane_id >> 4) =
    // output group acc[i] = C[g*4+i][col], so with BS=1 only col==0 is valid.
    if (col == 0) {
      if constexpr (FUSE_SWIGLU) {
        // Fused SwiGLU epilogue: acc[0]=gate, acc[1]=up, acc[2]=gate, acc[3]=up
        // Apply SwiGLU(gate+bias, up+bias) and write 2 activated values.
        constexpr int ACT_STRIDE = OUTPUT_STRIDE / 2;
        for (int i = 0; i < 4; i += 2) {
          int out_n = wg_idx * OUTPUT_PER_WG + wave_tile * 16 + g * 4 + i;
          if (out_n + 1 < OUTPUT_SIZE) {
            // Add biases (interleaved in [E, OUTPUT_STRIDE])
            unsigned bt_g = (unsigned)d_bias[expert_id * OUTPUT_STRIDE + out_n]
                            << 16;
            unsigned bt_u =
                (unsigned)d_bias[expert_id * OUTPUT_STRIDE + out_n + 1] << 16;
            float bias_g;
            __builtin_memcpy(&bias_g, &bt_g, 4);
            float bias_u;
            __builtin_memcpy(&bias_u, &bt_u, 4);

            float activated =
                fast_swigluoai(acc[i] + bias_g, acc[i + 1] + bias_u);

            int act_n = out_n / 2;
            int out_idx = tok_idx * (NUM_TOPK * ACT_STRIDE) +
                          topk_slot * ACT_STRIDE + act_n;
            d_output[out_idx] = _gang_float_to_bf16(activated);
          }
        }
      } else {
        // Standard epilogue: write 4 output elements per lane.
        for (int i = 0; i < 4; i++) {
          int out_n = wg_idx * OUTPUT_PER_WG + wave_tile * 16 + g * 4 + i;
          if (out_n < OUTPUT_SIZE) {
            float sum = acc[i];

            // Add bias (2D: [E, OUTPUT_STRIDE])
            float bias_val;
            unsigned bt = (unsigned)d_bias[expert_id * OUTPUT_STRIDE + out_n]
                          << 16;
            __builtin_memcpy(&bias_val, &bt, 4);

            float val = sum + bias_val;
            int out_idx = tok_idx * (NUM_TOPK * OUTPUT_STRIDE) +
                          topk_slot * OUTPUT_STRIDE + out_n;
            d_output[out_idx] = _gang_float_to_bf16(val);
          }
        }
      }
    }
  }

  __syncthreads();
}

} // namespace kernel
