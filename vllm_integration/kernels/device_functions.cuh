/* Titan: Unified Device Function Library
 *
 * Composable building blocks for persistent kernel compilation.
 * Two-level fusion:
 *   Level 1 (CODA-style): Epilogue templates fuse element-wise ops into GEMM
 *   Level 2 (Persistent):  Device functions compose via LDS within one kernel
 *
 * All GEMMs share the SAME MXFP4 depth-4 MFMA mainloop. The only difference
 * between QKV, O-proj, GateUp, Down, LM head, MoE W13, MoE W2 is the
 * epilogue template parameter.
 *
 * Dependencies: common.cuh (NT loads, atomics, XCD ID)
 *               tasks/mi300/gang_moe_linear_mxfp4_mi300.cuh (MFMA helpers)
 */
#pragma once

#include "common.cuh"

// Import MFMA helper types and functions from mirage
// Provides: kernel::f32x4_t, kernel::i32x8_t, kernel::_gang_mfma_f4xf8,
//           kernel::_gang_load_fp8_mfma_b, kernel::_gang_wave_parallel_fp8_quant,
//           kernel::_gang_float_to_bf16
#include "tasks/mi300/gang_moe_linear_mxfp4_mi300.cuh"

// RMSNorm (provides kernel::gang_rmsnorm_detail::rmsnorm_inline_amd)
#include "tasks/mi300/gang_rmsnorm_linear_mxfp4_bias_mi300.cuh"

// NOTE: Heavy includes (CK FMHA attention, residual add, merge, etc.)
// are NOT included here. The model-specific kernel file includes those
// directly, keeping this library lightweight and fast to compile.

namespace titan {

// ============================================================================
// Section A: Composable Epilogue Templates
// ============================================================================
// Each epilogue is called after the MFMA mainloop, while the f32x4 accumulator
// is still in VGPR registers. The epilogue decides what to do with the result
// (store to HBM, store to LDS, fuse with another operation, etc.)

// Column index within a 16x16 MFMA tile
// col = lane_id & 15, g = lane_id >> 4 (K-group 0..3)
// Each lane with col==0 owns 4 output values: acc[0..3]
// Corresponding output indices: wave_tile*16 + g*4 + {0,1,2,3}

// ── EpilogueStore: convert f32 acc to bf16, write to HBM ──
struct EpilogueStore {
    unsigned short *output;
    int output_stride;
    int output_size;  // actual output dim (bounds check)

    __device__ __forceinline__ void operator()(
        kernel::f32x4_t acc, int wave_tile, int col, int g,
        int n_tile, int m_tile, int OUTPUT_PER_WG,
        const unsigned short *bias, int batch_size) {
        if (col != 0) return;
        for (int i = 0; i < 4; i++) {
            int out_n = n_tile * OUTPUT_PER_WG + wave_tile * 16 + g * 4 + i;
            if (out_n < output_size) {
                float sum = acc[i];
                if (bias) {
                    unsigned bt = (unsigned)bias[out_n] << 16;
                    float bv;
                    __builtin_memcpy(&bv, &bt, 4);
                    sum += bv;
                }
                // Write to global column index (out_n), not local tile offset
                int out_idx = m_tile * batch_size * output_stride + out_n;
                output[out_idx] = kernel::_gang_float_to_bf16(sum);
            }
        }
    }
};

// ── EpilogueResAdd: acc + residual -> bf16 to HBM ──
// Used by O-proj and Down-proj to fuse the residual connection.
struct EpilogueResAdd {
    unsigned short *output;
    const unsigned short *residual;  // [bs, hidden] bf16
    int output_stride;
    int output_size;

    __device__ __forceinline__ void operator()(
        kernel::f32x4_t acc, int wave_tile, int col, int g,
        int n_tile, int m_tile, int OUTPUT_PER_WG,
        const unsigned short *bias, int batch_size) {
        if (col != 0) return;
        for (int i = 0; i < 4; i++) {
            int out_n = n_tile * OUTPUT_PER_WG + wave_tile * 16 + g * 4 + i;
            if (out_n < output_size) {
                float sum = acc[i];
                // Add bias
                if (bias) {
                    unsigned bt = (unsigned)bias[out_n] << 16;
                    float bv;
                    __builtin_memcpy(&bv, &bt, 4);
                    sum += bv;
                }
                // Add residual
                unsigned rt = (unsigned)residual[out_n] << 16;
                float rv;
                __builtin_memcpy(&rv, &rt, 4);
                sum += rv;

                // Write to global column index (out_n), not local tile offset
                int out_idx = m_tile * batch_size * output_stride + out_n;
                output[out_idx] = kernel::_gang_float_to_bf16(sum);
            }
        }
    }
};

// ── EpilogueSwiGLU: SiLU(gate) * up -> bf16 to HBM ──
// Fuses SwiGLU activation into GateUp GEMM epilogue.
// Requires OPW=128: columns [0:64] = gate, columns [64:128] = up.
// TILES_PER_WAVE=2: tile_iter=0 computes gate half, tile_iter=1 computes up half.
//
// Each wave processes its gate tile first (tile_iter=0, wave_tile=warp_id),
// saves gate+bias values in registers (gate_save[4]), then processes its
// matching up tile (tile_iter=1, wave_tile=warp_id+4).
// No LDS or cross-wave synchronization needed — each wave pairs its own
// gate and up tiles in registers.
struct EpilogueSwiGLU {
    unsigned short *output;     // [bs, intermediate_size] bf16
    int output_stride;          // intermediate_size (output row stride)
    int output_size;            // per-XCD output columns

    // Per-wave register storage for gate values between tile_iter=0 and tile_iter=1.
    // Set by user before calling gemm_mxfp4.
    float gate_save[4];

    __device__ __forceinline__ void operator()(
        kernel::f32x4_t acc, int wave_tile, int col, int g,
        int n_tile, int m_tile, int OUTPUT_PER_WG,
        const unsigned short *bias, int batch_size) {
        if (col != 0) return;

        bool is_gate = (wave_tile < 4);  // tile_iter=0 → gate, tile_iter=1 → up

        if (is_gate) {
            // Gate half: add bias, save to registers
            for (int i = 0; i < 4; i++) {
                float sum = acc[i];
                if (bias) {
                    int abs_n = n_tile * OUTPUT_PER_WG + wave_tile * 16 + g * 4 + i;
                    unsigned bt = (unsigned)bias[abs_n] << 16;
                    float bv;
                    __builtin_memcpy(&bv, &bt, 4);
                    sum += bv;
                }
                gate_save[i] = sum;
            }
        } else {
            // Up half: add bias, fuse with saved gate, write output
            for (int i = 0; i < 4; i++) {
                float up_val = acc[i];
                if (bias) {
                    int abs_n = n_tile * OUTPUT_PER_WG + wave_tile * 16 + g * 4 + i;
                    unsigned bt = (unsigned)bias[abs_n] << 16;
                    float bv;
                    __builtin_memcpy(&bv, &bt, 4);
                    up_val += bv;
                }
                // SiLU(gate) * up — gate_save from tile_iter=0
                float gate_val = gate_save[i];
                float silu_gate = gate_val / (1.0f + __expf(-gate_val));
                float result = silu_gate * up_val;

                // Output index: each WG produces OUTPUT_PER_WG/2 = 64 elements
                // local column within this WG's output: (wave_tile-4)*16 + g*4 + i
                int local_n = (wave_tile - 4) * 16 + g * 4 + i;
                int out_n = n_tile * (OUTPUT_PER_WG / 2) + local_n;
                if (out_n < output_size) {
                    int out_idx = m_tile * batch_size * output_stride + out_n;
                    output[out_idx] = kernel::_gang_float_to_bf16(result);
                }
            }
        }
    }
};

// ── EpilogueArgmax: update running (max_val, max_idx) in registers ──
// Used by LM head GEMM. No HBM write — just tracks the running maximum.
struct EpilogueArgmax {
    float *thread_max;
    long long *thread_max_idx;
    int partition_start;  // xcd_id * n_wgs_per_xcd * output_per_wg

    __device__ __forceinline__ void operator()(
        kernel::f32x4_t acc, int wave_tile, int col, int g,
        int n_tile, int m_tile, int OUTPUT_PER_WG,
        const unsigned short *bias, int batch_size) {
        if (col != 0) return;
        for (int i = 0; i < 4; i++) {
            int out_n = n_tile * OUTPUT_PER_WG + wave_tile * 16 + g * 4 + i;
            float sum = acc[i];
            if (bias) {
                unsigned bt = (unsigned)bias[out_n] << 16;
                float bv;
                __builtin_memcpy(&bv, &bt, 4);
                sum += bv;
            }
            long long abs_idx = (long long)(partition_start + out_n);
            if (sum > *thread_max) {
                *thread_max = sum;
                *thread_max_idx = abs_idx;
            }
        }
    }
};


// ============================================================================
// Section B: MXFP4 GEMM Mainloop with Pluggable Epilogue
// ============================================================================
// Dead code guard: these standalone GEMM/RoPE/TopK functions are NOT used
// when the kernel uses mirage's fused sub-kernels. Guard them to prevent
// __noinline__ from forcing compilation of unused template instantiations.
#ifdef TITAN_ENABLE_LEGACY

// ============================================================================
// This is the CORE building block. The depth-4 MFMA pipeline is IDENTICAL
// for all GEMMs (QKV, O-proj, GateUp, Down, LM head, MoE W13, MoE W2).
// The epilogue template parameter is the ONLY customization point.
//
// Hardware: FP4 weights x FP8 activations via
//   __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4
//
// Weight format: MXFP4 packed per workgroup
//   [FP4 data: OPW * K/2 bytes][E8M0 scales: OPW * K/32 bytes]
//
// Input: bf16, quantized to FP8 E4M3 on-the-fly in shared memory.

template <typename Epilogue,
          int BATCH_SIZE,         // M-tiles (rows per tile, typically 1 for decode)
          int REDUCTION_SIZE,     // K dimension (hidden_size or intermediate_size)
          int OUTPUT_PER_WG = 64> // N per workgroup (4 waves x 16 rows)
__device__ __noinline__ void
gemm_mxfp4(
    void const *input_ptr,       // [batch, REDUCTION_SIZE] bf16
    void const *weight_ptr,      // [n_wgs, wg_bytes] packed MXFP4
    void *output_ptr,            // (passed to epilogue)
    int num_active_tokens,
    int n_tiles,                 // N-tiles per XCD
    int wg_idx,                  // which workgroup this worker processes
    void const *bias_ptr,        // optional bias [1, output_size] bf16
    int output_stride,           // output row stride
    int output_size,             // actual output dim
    Epilogue &epilogue)
{
    static_assert(REDUCTION_SIZE % 128 == 0, "K must be multiple of 128");

    // MXFP4 weight layout constants
    constexpr int NUM_BLOCKS_32 = REDUCTION_SIZE / 32;
    constexpr int WG_DATA_BYTES = OUTPUT_PER_WG * (REDUCTION_SIZE / 2);
    constexpr int WG_SCALE_BYTES = OUTPUT_PER_WG * NUM_BLOCKS_32;
    constexpr int WG_BYTES = WG_DATA_BYTES + WG_SCALE_BYTES;

    // MFMA constants
    constexpr int K_PER_MFMA = 128;
    constexpr int MFMA_ITERS = REDUCTION_SIZE / K_PER_MFMA;
    constexpr int NUM_WAVES = 4;
    constexpr int TILES_PER_WAVE = OUTPUT_PER_WG / 16 / NUM_WAVES;  // = 1

    const unsigned short *A = (const unsigned short *)input_ptr;
    const uint8_t *W_data = (const uint8_t *)weight_ptr;
    const unsigned short *d_bias = (const unsigned short *)bias_ptr;

    extern __shared__ char _gemm_smem[];
    uint8_t *s_tok_fp8 = (uint8_t *)_gemm_smem;
    uint8_t *s_tok_scales = s_tok_fp8 + REDUCTION_SIZE;

    int const tid = threadIdx.x;
    int const warp_id = tid >> 6;
    int const lane_id = tid & 63;
    int const col = lane_id & 15;
    int const g = lane_id >> 4;

    // For decode (BATCH_SIZE=1), m_tile=0 always
    int m_tile = 0;

    // Quantize bf16 input to FP8 E4M3 in shared memory
    kernel::_gang_wave_parallel_fp8_quant<REDUCTION_SIZE>(
        A + static_cast<size_t>(m_tile) * BATCH_SIZE * REDUCTION_SIZE,
        s_tok_fp8, s_tok_scales);

    // Weight pointer for this workgroup
    uint8_t const *wg_data = W_data + static_cast<int64_t>(wg_idx) * WG_BYTES;
    uint8_t const *wg_scales = wg_data + WG_DATA_BYTES;

    // Depth-4 pipelined MFMA loop
    for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; tile_iter++) {
        int wave_tile = warp_id + tile_iter * NUM_WAVES;
        int w_row = wave_tile * 16 + col;
        int const row_data_base = w_row * (REDUCTION_SIZE / 2);
        int const row_scale_base = w_row * NUM_BLOCKS_32;

        kernel::f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

        // Pre-fill pipeline slots 0..3
        kernel::i32x8_t a0 = *(const kernel::i32x8_t*)(wg_data + row_data_base + 0 * 64 + g * 16);
        int sa0 = (int)wg_scales[row_scale_base + 0 * 4 + g];
        kernel::i32x8_t a1 = *(const kernel::i32x8_t*)(wg_data + row_data_base + 1 * 64 + g * 16);
        int sa1 = (int)wg_scales[row_scale_base + 1 * 4 + g];
        kernel::i32x8_t a2 = *(const kernel::i32x8_t*)(wg_data + row_data_base + 2 * 64 + g * 16);
        int sa2 = (int)wg_scales[row_scale_base + 2 * 4 + g];
        kernel::i32x8_t a3 = *(const kernel::i32x8_t*)(wg_data + row_data_base + 3 * 64 + g * 16);
        int sa3 = (int)wg_scales[row_scale_base + 3 * 4 + g];

        #pragma unroll 1
        for (int ki = 0; ki < MFMA_ITERS; ki += 4) {
            // Slot 0
            {
                kernel::i32x8_t b = kernel::_gang_load_fp8_mfma_b(s_tok_fp8, ki * K_PER_MFMA, g);
                int sb = (int)s_tok_scales[ki];
                acc = kernel::_gang_mfma_f4xf8(a0, b, acc, sa0, sb);
            }
            if (ki + 4 < MFMA_ITERS) {
                int kt = (ki + 4) * K_PER_MFMA;
                a0 = *(const kernel::i32x8_t*)(wg_data + row_data_base + kt / 2 + g * 16);
                sa0 = (int)wg_scales[row_scale_base + kt / 32 + g];
            }
            // Slot 1
            {
                kernel::i32x8_t b = kernel::_gang_load_fp8_mfma_b(s_tok_fp8, (ki + 1) * K_PER_MFMA, g);
                int sb = (int)s_tok_scales[ki + 1];
                acc = kernel::_gang_mfma_f4xf8(a1, b, acc, sa1, sb);
            }
            if (ki + 5 < MFMA_ITERS) {
                int kt = (ki + 5) * K_PER_MFMA;
                a1 = *(const kernel::i32x8_t*)(wg_data + row_data_base + kt / 2 + g * 16);
                sa1 = (int)wg_scales[row_scale_base + kt / 32 + g];
            }
            // Slot 2
            {
                kernel::i32x8_t b = kernel::_gang_load_fp8_mfma_b(s_tok_fp8, (ki + 2) * K_PER_MFMA, g);
                int sb = (int)s_tok_scales[ki + 2];
                acc = kernel::_gang_mfma_f4xf8(a2, b, acc, sa2, sb);
            }
            if (ki + 6 < MFMA_ITERS) {
                int kt = (ki + 6) * K_PER_MFMA;
                a2 = *(const kernel::i32x8_t*)(wg_data + row_data_base + kt / 2 + g * 16);
                sa2 = (int)wg_scales[row_scale_base + kt / 32 + g];
            }
            // Slot 3
            if (ki + 3 < MFMA_ITERS) {
                kernel::i32x8_t b = kernel::_gang_load_fp8_mfma_b(s_tok_fp8, (ki + 3) * K_PER_MFMA, g);
                int sb = (int)s_tok_scales[ki + 3];
                acc = kernel::_gang_mfma_f4xf8(a3, b, acc, sa3, sb);
            }
            if (ki + 7 < MFMA_ITERS) {
                int kt = (ki + 7) * K_PER_MFMA;
                a3 = *(const kernel::i32x8_t*)(wg_data + row_data_base + kt / 2 + g * 16);
                sa3 = (int)wg_scales[row_scale_base + kt / 32 + g];
            }
        }

        // ── EPILOGUE: applied while acc is still in registers ──
        epilogue(acc, wave_tile, col, g,
                 wg_idx, m_tile, OUTPUT_PER_WG,
                 d_bias, BATCH_SIZE);
    }

    __syncthreads();
}


// ============================================================================
// Section B2: MXFP4 GEMM WITHOUT FP8 re-quantization
// ============================================================================
// Same as gemm_mxfp4 but SKIPS the FP8 quantization step. The caller must
// ensure shared memory already contains the quantized input (via fp8_quant).
// Use this when the same input is processed by multiple tiles (e.g., MoE
// expert loop where all W13 tiles share the same norm output).

template <typename Epilogue,
          int BATCH_SIZE,
          int REDUCTION_SIZE,
          int OUTPUT_PER_WG = 64>
__device__ __noinline__ void
gemm_mxfp4_noquant(
    void const *weight_ptr,
    void *output_ptr,
    int num_active_tokens,
    int n_tiles,
    int wg_idx,
    void const *bias_ptr,
    int output_stride,
    int output_size,
    Epilogue &epilogue)
{
    static_assert(REDUCTION_SIZE % 128 == 0, "K must be multiple of 128");

    constexpr int NUM_BLOCKS_32 = REDUCTION_SIZE / 32;
    constexpr int WG_DATA_BYTES = OUTPUT_PER_WG * (REDUCTION_SIZE / 2);
    constexpr int WG_SCALE_BYTES = OUTPUT_PER_WG * NUM_BLOCKS_32;
    constexpr int WG_BYTES = WG_DATA_BYTES + WG_SCALE_BYTES;

    constexpr int K_PER_MFMA = 128;
    constexpr int MFMA_ITERS = REDUCTION_SIZE / K_PER_MFMA;
    constexpr int NUM_WAVES = 4;
    constexpr int TILES_PER_WAVE = OUTPUT_PER_WG / 16 / NUM_WAVES;

    const uint8_t *W_data = (const uint8_t *)weight_ptr;
    const unsigned short *d_bias = (const unsigned short *)bias_ptr;

    extern __shared__ char _gemm_smem[];
    uint8_t *s_tok_fp8 = (uint8_t *)_gemm_smem;
    uint8_t *s_tok_scales = s_tok_fp8 + REDUCTION_SIZE;

    int const tid = threadIdx.x;
    int const warp_id = tid >> 6;
    int const lane_id = tid & 63;
    int const col = lane_id & 15;
    int const g = lane_id >> 4;

    int m_tile = 0;

    // NO FP8 quantization — shared memory already has quantized input

    uint8_t const *wg_data = W_data + static_cast<int64_t>(wg_idx) * WG_BYTES;
    uint8_t const *wg_scales = wg_data + WG_DATA_BYTES;

    for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; tile_iter++) {
        int wave_tile = warp_id + tile_iter * NUM_WAVES;
        int w_row = wave_tile * 16 + col;
        int const row_data_base = w_row * (REDUCTION_SIZE / 2);
        int const row_scale_base = w_row * NUM_BLOCKS_32;

        kernel::f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

        kernel::i32x8_t a0 = *(const kernel::i32x8_t*)(wg_data + row_data_base + 0 * 64 + g * 16);
        int sa0 = (int)wg_scales[row_scale_base + 0 * 4 + g];
        kernel::i32x8_t a1 = *(const kernel::i32x8_t*)(wg_data + row_data_base + 1 * 64 + g * 16);
        int sa1 = (int)wg_scales[row_scale_base + 1 * 4 + g];
        kernel::i32x8_t a2 = *(const kernel::i32x8_t*)(wg_data + row_data_base + 2 * 64 + g * 16);
        int sa2 = (int)wg_scales[row_scale_base + 2 * 4 + g];
        kernel::i32x8_t a3 = *(const kernel::i32x8_t*)(wg_data + row_data_base + 3 * 64 + g * 16);
        int sa3 = (int)wg_scales[row_scale_base + 3 * 4 + g];

        #pragma unroll 1
        for (int ki = 0; ki < MFMA_ITERS; ki += 4) {
            {
                kernel::i32x8_t b = kernel::_gang_load_fp8_mfma_b(s_tok_fp8, ki * K_PER_MFMA, g);
                int sb = (int)s_tok_scales[ki];
                acc = kernel::_gang_mfma_f4xf8(a0, b, acc, sa0, sb);
            }
            if (ki + 4 < MFMA_ITERS) {
                int kt = (ki + 4) * K_PER_MFMA;
                a0 = *(const kernel::i32x8_t*)(wg_data + row_data_base + kt / 2 + g * 16);
                sa0 = (int)wg_scales[row_scale_base + kt / 32 + g];
            }
            {
                kernel::i32x8_t b = kernel::_gang_load_fp8_mfma_b(s_tok_fp8, (ki + 1) * K_PER_MFMA, g);
                int sb = (int)s_tok_scales[ki + 1];
                acc = kernel::_gang_mfma_f4xf8(a1, b, acc, sa1, sb);
            }
            if (ki + 5 < MFMA_ITERS) {
                int kt = (ki + 5) * K_PER_MFMA;
                a1 = *(const kernel::i32x8_t*)(wg_data + row_data_base + kt / 2 + g * 16);
                sa1 = (int)wg_scales[row_scale_base + kt / 32 + g];
            }
            {
                kernel::i32x8_t b = kernel::_gang_load_fp8_mfma_b(s_tok_fp8, (ki + 2) * K_PER_MFMA, g);
                int sb = (int)s_tok_scales[ki + 2];
                acc = kernel::_gang_mfma_f4xf8(a2, b, acc, sa2, sb);
            }
            if (ki + 6 < MFMA_ITERS) {
                int kt = (ki + 6) * K_PER_MFMA;
                a2 = *(const kernel::i32x8_t*)(wg_data + row_data_base + kt / 2 + g * 16);
                sa2 = (int)wg_scales[row_scale_base + kt / 32 + g];
            }
            if (ki + 3 < MFMA_ITERS) {
                kernel::i32x8_t b = kernel::_gang_load_fp8_mfma_b(s_tok_fp8, (ki + 3) * K_PER_MFMA, g);
                int sb = (int)s_tok_scales[ki + 3];
                acc = kernel::_gang_mfma_f4xf8(a3, b, acc, sa3, sb);
            }
            if (ki + 7 < MFMA_ITERS) {
                int kt = (ki + 7) * K_PER_MFMA;
                a3 = *(const kernel::i32x8_t*)(wg_data + row_data_base + kt / 2 + g * 16);
                sa3 = (int)wg_scales[row_scale_base + kt / 32 + g];
            }
        }

        epilogue(acc, wave_tile, col, g,
                 wg_idx, m_tile, OUTPUT_PER_WG,
                 d_bias, BATCH_SIZE);
    }

    // No __syncthreads — caller manages synchronization
}

// ============================================================================
// Section B3: MXFP4 GEMM with LDS-resident weights
// ============================================================================
// Same MFMA pipeline as gemm_mxfp4 but reads weight data from LDS instead of
// HBM. The caller must DMA weights into LDS (via buffer_load_lds) before calling.
// Uses __builtin_memcpy for LDS reads which compiles to ds_read_b128.
//
// lds_weight_offset: byte offset within extern __shared__ where weights start.
//   Layout in LDS starting at lds_weight_offset:
//     [0 .. WG_DATA_BYTES)                        = FP4 weight data
//     [WG_DATA_BYTES .. WG_DATA_BYTES+WG_SCALE)   = E8M0 block scales
//   The caller's buffer_load_lds must produce this exact layout.

template <typename Epilogue,
          int BATCH_SIZE,
          int REDUCTION_SIZE,
          int OUTPUT_PER_WG = 64>
__device__ __noinline__ void
gemm_mxfp4_lds_weight(
    void const *input_ptr,       // [batch, REDUCTION_SIZE] bf16 (HBM)
    int lds_weight_offset,       // byte offset in LDS where weight tile starts
    void *output_ptr,
    int num_active_tokens,
    int n_tiles,
    int wg_idx,                  // used ONLY for epilogue (output column offset)
    void const *bias_ptr,
    int output_stride,
    int output_size,
    Epilogue &epilogue)
{
    static_assert(REDUCTION_SIZE % 128 == 0, "K must be multiple of 128");

    constexpr int NUM_BLOCKS_32 = REDUCTION_SIZE / 32;
    constexpr int WG_DATA_BYTES = OUTPUT_PER_WG * (REDUCTION_SIZE / 2);
    constexpr int WG_SCALE_BYTES = OUTPUT_PER_WG * NUM_BLOCKS_32;

    constexpr int K_PER_MFMA = 128;
    constexpr int MFMA_ITERS = REDUCTION_SIZE / K_PER_MFMA;
    constexpr int NUM_WAVES = 4;
    constexpr int TILES_PER_WAVE = OUTPUT_PER_WG / 16 / NUM_WAVES;

    const unsigned short *A = (const unsigned short *)input_ptr;
    const unsigned short *d_bias = (const unsigned short *)bias_ptr;

    extern __shared__ char _gemm_smem[];
    uint8_t *s_tok_fp8 = (uint8_t *)_gemm_smem;
    uint8_t *s_tok_scales = s_tok_fp8 + REDUCTION_SIZE;

    // LDS weight pointers
    const uint8_t *lds_w_data = (const uint8_t *)_gemm_smem + lds_weight_offset;
    const uint8_t *lds_w_scales = lds_w_data + WG_DATA_BYTES;

    int const tid = threadIdx.x;
    int const warp_id = tid >> 6;
    int const lane_id = tid & 63;
    int const col = lane_id & 15;
    int const g = lane_id >> 4;

    int m_tile = 0;

    // Quantize bf16 input to FP8 E4M3 in shared memory
    kernel::_gang_wave_parallel_fp8_quant<REDUCTION_SIZE>(
        A + static_cast<size_t>(m_tile) * BATCH_SIZE * REDUCTION_SIZE,
        s_tok_fp8, s_tok_scales);

    // MFMA loop with LDS weight reads
    for (int tile_iter = 0; tile_iter < TILES_PER_WAVE; tile_iter++) {
        int wave_tile = warp_id + tile_iter * NUM_WAVES;
        int w_row = wave_tile * 16 + col;
        int const row_data_base = w_row * (REDUCTION_SIZE / 2);
        int const row_scale_base = w_row * NUM_BLOCKS_32;

        kernel::f32x4_t acc = {0.0f, 0.0f, 0.0f, 0.0f};

        // Pre-fill pipeline: load 4 weight slots from LDS via __builtin_memcpy
        kernel::i32x4_t wt0, wt1, wt2, wt3;
        __builtin_memcpy(&wt0, lds_w_data + row_data_base + 0 * 64 + g * 16, 16);
        int sa0 = (int)lds_w_scales[row_scale_base + 0 * 4 + g];
        __builtin_memcpy(&wt1, lds_w_data + row_data_base + 1 * 64 + g * 16, 16);
        int sa1 = (int)lds_w_scales[row_scale_base + 1 * 4 + g];
        __builtin_memcpy(&wt2, lds_w_data + row_data_base + 2 * 64 + g * 16, 16);
        int sa2 = (int)lds_w_scales[row_scale_base + 2 * 4 + g];
        __builtin_memcpy(&wt3, lds_w_data + row_data_base + 3 * 64 + g * 16, 16);
        int sa3 = (int)lds_w_scales[row_scale_base + 3 * 4 + g];

        // Construct i32x8_t with upper half zeroed (FP4 weight in lower 128 bits)
        kernel::i32x8_t a0 = {wt0[0], wt0[1], wt0[2], wt0[3], 0, 0, 0, 0};
        kernel::i32x8_t a1 = {wt1[0], wt1[1], wt1[2], wt1[3], 0, 0, 0, 0};
        kernel::i32x8_t a2 = {wt2[0], wt2[1], wt2[2], wt2[3], 0, 0, 0, 0};
        kernel::i32x8_t a3 = {wt3[0], wt3[1], wt3[2], wt3[3], 0, 0, 0, 0};

        #pragma unroll 1
        for (int ki = 0; ki < MFMA_ITERS; ki += 4) {
            // Slot 0
            {
                kernel::i32x8_t b = kernel::_gang_load_fp8_mfma_b(s_tok_fp8, ki * K_PER_MFMA, g);
                int sb = (int)s_tok_scales[ki];
                acc = kernel::_gang_mfma_f4xf8(a0, b, acc, sa0, sb);
            }
            if (ki + 4 < MFMA_ITERS) {
                int kt = (ki + 4) * K_PER_MFMA;
                __builtin_memcpy(&wt0, lds_w_data + row_data_base + kt / 2 + g * 16, 16);
                sa0 = (int)lds_w_scales[row_scale_base + kt / 32 + g];
                a0 = {wt0[0], wt0[1], wt0[2], wt0[3], 0, 0, 0, 0};
            }
            // Slot 1
            {
                kernel::i32x8_t b = kernel::_gang_load_fp8_mfma_b(s_tok_fp8, (ki + 1) * K_PER_MFMA, g);
                int sb = (int)s_tok_scales[ki + 1];
                acc = kernel::_gang_mfma_f4xf8(a1, b, acc, sa1, sb);
            }
            if (ki + 5 < MFMA_ITERS) {
                int kt = (ki + 5) * K_PER_MFMA;
                __builtin_memcpy(&wt1, lds_w_data + row_data_base + kt / 2 + g * 16, 16);
                sa1 = (int)lds_w_scales[row_scale_base + kt / 32 + g];
                a1 = {wt1[0], wt1[1], wt1[2], wt1[3], 0, 0, 0, 0};
            }
            // Slot 2
            {
                kernel::i32x8_t b = kernel::_gang_load_fp8_mfma_b(s_tok_fp8, (ki + 2) * K_PER_MFMA, g);
                int sb = (int)s_tok_scales[ki + 2];
                acc = kernel::_gang_mfma_f4xf8(a2, b, acc, sa2, sb);
            }
            if (ki + 6 < MFMA_ITERS) {
                int kt = (ki + 6) * K_PER_MFMA;
                __builtin_memcpy(&wt2, lds_w_data + row_data_base + kt / 2 + g * 16, 16);
                sa2 = (int)lds_w_scales[row_scale_base + kt / 32 + g];
                a2 = {wt2[0], wt2[1], wt2[2], wt2[3], 0, 0, 0, 0};
            }
            // Slot 3
            if (ki + 3 < MFMA_ITERS) {
                kernel::i32x8_t b = kernel::_gang_load_fp8_mfma_b(s_tok_fp8, (ki + 3) * K_PER_MFMA, g);
                int sb = (int)s_tok_scales[ki + 3];
                acc = kernel::_gang_mfma_f4xf8(a3, b, acc, sa3, sb);
            }
            if (ki + 7 < MFMA_ITERS) {
                int kt = (ki + 7) * K_PER_MFMA;
                __builtin_memcpy(&wt3, lds_w_data + row_data_base + kt / 2 + g * 16, 16);
                sa3 = (int)lds_w_scales[row_scale_base + kt / 32 + g];
                a3 = {wt3[0], wt3[1], wt3[2], wt3[3], 0, 0, 0, 0};
            }
        }

        epilogue(acc, wave_tile, col, g,
                 wg_idx, m_tile, OUTPUT_PER_WG,
                 d_bias, BATCH_SIZE);
    }

    __syncthreads();
}

#endif // TITAN_ENABLE_LEGACY (gemm_mxfp4 + variants)

// ============================================================================
// Section C: RMSNorm Device Function
// ============================================================================
// Standalone RMSNorm for prologue fusion. All workers redundantly compute
// the same norm on the input row. Output written to scratch buffer.

template<int HIDDEN_SIZE, int ACTUAL_HIDDEN_DIM>
__device__ __noinline__ void
rmsnorm(void const *input, void const *weight, void *output) {
    const unsigned short *row_in = (const unsigned short *)input;
    unsigned short *row_out = (unsigned short *)output;
    kernel::gang_rmsnorm_detail::rmsnorm_inline_amd<HIDDEN_SIZE, ACTUAL_HIDDEN_DIM>(
        row_in, weight, row_out);
    __syncthreads();
}

#ifdef TITAN_ENABLE_LEGACY
// Fused ResAddF32 + RMSNorm: reads f32 workspace + bf16 residual, computes
// resadd, writes bf16 residual output + norm output. All workers redundantly
// compute the same result. Workspace zeroing done by single worker (is_zero_worker).
// Adapted from mirage's gang_resaddf32_rmsnorm_linear_mxfp4_bias_kernel.
template<int HIDDEN_SIZE, int ACTUAL_HIDDEN_DIM>
__device__ __noinline__ void
rmsnorm_resadd(float *workspace_f32,          // [HIDDEN_SIZE] f32 (MoE output)
               const unsigned short *residual, // [HIDDEN_SIZE] bf16 (OProj output)
               const unsigned short *norm_weight, // [HIDDEN_SIZE] bf16
               unsigned short *output_residual,   // [HIDDEN_SIZE] bf16 (resadd result)
               unsigned short *norm_output,       // [HIDDEN_SIZE] bf16 (norm result)
               bool is_zero_worker)               // only this worker zeros workspace
{
    using bf16 = __hip_bfloat16;
    int tid = threadIdx.x;

    // Pass 1: workspace_f32 + residual → output_residual, accumulate SSQ
    // Vectorized: float4 for f32 workspace, uint2 for bf16 residual/output
    float ssq = 0.0f;
    constexpr int VEC = 4;
    constexpr int BLOCK_VEC = 256 * VEC;  // 1024 elements per pass
    constexpr int MAX_ITERS = (HIDDEN_SIZE + BLOCK_VEC - 1) / BLOCK_VEC;
    float s_cache[MAX_ITERS * VEC];
    int n_cached = 0;

    #pragma unroll
    for (int off = tid * VEC; off < HIDDEN_SIZE; off += BLOCK_VEC) {
        // Vectorized load: 4 f32 from workspace
        float4 ws4;
        __builtin_memcpy(&ws4, workspace_f32 + off, 16);

        // Vectorized load: 4 bf16 from residual as uint2
        uint2 res_packed;
        __builtin_memcpy(&res_packed, residual + off, 8);

        // Extract 4 bf16 → f32, add
        unsigned r0 = (res_packed.x & 0xFFFFu) << 16;
        unsigned r1 = res_packed.x & 0xFFFF0000u;
        unsigned r2 = (res_packed.y & 0xFFFFu) << 16;
        unsigned r3 = res_packed.y & 0xFFFF0000u;
        float rv0, rv1, rv2, rv3;
        __builtin_memcpy(&rv0, &r0, 4);
        __builtin_memcpy(&rv1, &r1, 4);
        __builtin_memcpy(&rv2, &r2, 4);
        __builtin_memcpy(&rv3, &r3, 4);

        float s0 = ws4.x + rv0;
        float s1 = ws4.y + rv1;
        float s2 = ws4.z + rv2;
        float s3 = ws4.w + rv3;

        // Accumulate SSQ for RMSNorm
        asm volatile("v_fmac_f32 %0, %1, %1" : "+v"(ssq) : "v"(s0));
        asm volatile("v_fmac_f32 %0, %1, %1" : "+v"(ssq) : "v"(s1));
        asm volatile("v_fmac_f32 %0, %1, %1" : "+v"(ssq) : "v"(s2));
        asm volatile("v_fmac_f32 %0, %1, %1" : "+v"(ssq) : "v"(s3));

        // Cache resadd sums in registers for Pass 2
        s_cache[n_cached + 0] = s0;
        s_cache[n_cached + 1] = s1;
        s_cache[n_cached + 2] = s2;
        s_cache[n_cached + 3] = s3;
        n_cached += VEC;

        // Pack 4 bf16 output and store as uint2
        unsigned short o0 = kernel::_gang_float_to_bf16(s0);
        unsigned short o1 = kernel::_gang_float_to_bf16(s1);
        unsigned short o2 = kernel::_gang_float_to_bf16(s2);
        unsigned short o3 = kernel::_gang_float_to_bf16(s3);
        uint2 out_packed;
        out_packed.x = (unsigned)o0 | ((unsigned)o1 << 16);
        out_packed.y = (unsigned)o2 | ((unsigned)o3 << 16);
        __builtin_memcpy(output_residual + off, &out_packed, 8);
    }

    // Wavefront reduction (AMD wave64)
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        ssq += __shfl_xor(ssq, offset);
    }

    // Cross-wave reduction via shared memory
    __shared__ float red[16];
    int wave_id = tid >> 6;
    int lane_id = tid & 63;
    int num_waves = 256 >> 6;  // = 4

    if (lane_id == 0) red[wave_id] = ssq;
    __syncthreads();

    if (wave_id == 0) {
        ssq = (lane_id < num_waves) ? red[lane_id] : 0.0f;
        for (int offset = num_waves >> 1; offset > 0; offset >>= 1) {
            ssq += __shfl_xor(ssq, offset);
        }
        if (lane_id == 0) red[0] = rsqrtf(ssq / (float)ACTUAL_HIDDEN_DIM + 1e-5f);
    }
    __syncthreads();
    float rms_rcp = red[0];

    // Pass 2: Apply norm weight using CACHED sums (no re-read of output_residual)
    {
        const bf16 *w_in = (const bf16 *)norm_weight;
        bf16 *out = (bf16 *)norm_output;
        int ci = 0;
        #pragma unroll
        for (int off = tid * VEC; off < HIDDEN_SIZE; off += BLOCK_VEC) {
            uint64_t wv;
            __builtin_memcpy(&wv, &w_in[off], 8);
            const bf16 *wa = (const bf16 *)&wv;
            bf16 ov[VEC];
            #pragma unroll
            for (int j = 0; j < VEC; j++) {
                ov[j] = __float2bfloat16(s_cache[ci + j] * rms_rcp
                                         * __bfloat162float(wa[j]));
            }
            __builtin_memcpy(&out[off], ov, 8);
            ci += VEC;
        }
    }

    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __syncthreads();

    // Zero workspace for next layer (single worker only)
    if (is_zero_worker) {
        float4 zero4 = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int off = tid * VEC; off < HIDDEN_SIZE; off += BLOCK_VEC) {
            __builtin_memcpy(workspace_f32 + off, &zero4, 16);
        }
    }
    __syncthreads();
}

// FP8 quantization: bf16 input -> FP8 E4M3 in shared memory
template<int REDUCTION_SIZE>
__device__ __noinline__ void
fp8_quant(void const *input_bf16, uint8_t *smem_fp8, uint8_t *smem_scales) {
    kernel::_gang_wave_parallel_fp8_quant<REDUCTION_SIZE>(
        (const unsigned short *)input_bf16, smem_fp8, smem_scales);
}


// ============================================================================
// Section C2: Gang-Parallel Router GEMV (bf16)
// ============================================================================
// All workers redundantly compute RMSNorm (same as rmsnorm), then each worker
// computes ONE expert's logit via bf16 dot product. Workers with expert_id >=
// NUM_EXPERTS are idle for the dot product but still participate in RMSNorm.
// Output is written as bf16 logits to the routing buffer by each worker's tid==0.

template<int HIDDEN_SIZE, int ACTUAL_HIDDEN_DIM, int NUM_EXPERTS>
__device__ __noinline__ void
router_gemv_bf16(
    const void *norm_output,           // [HIDDEN_SIZE] bf16 (already norm'd)
    const unsigned short *router_w_bf16, // [NUM_EXPERTS, HIDDEN_SIZE] bf16 row-major
    const unsigned short *router_bias,   // [NUM_EXPERTS] bf16
    unsigned short *logit_out,           // [NUM_EXPERTS] bf16 output
    int expert_id,                       // which expert this worker computes (-1 = skip)
    float *smem_reduce)                  // [4] shared memory for cross-warp reduce
{
    int const tid = threadIdx.x;

    if (expert_id < 0 || expert_id >= NUM_EXPERTS) {
        __syncthreads();  // match __syncthreads below
        __syncthreads();  // match final __syncthreads
        return;
    }

    const unsigned short *input = (const unsigned short *)norm_output;
    const unsigned short *w_row = router_w_bf16 + expert_id * HIDDEN_SIZE;

    // Each thread accumulates a partial dot product
    float sum = 0.0f;
    constexpr int ELEMS_PER_THREAD = (ACTUAL_HIDDEN_DIM + 255) / 256;

    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int idx = tid + i * 256;
        if (idx < ACTUAL_HIDDEN_DIM) {
            float a = __bfloat162float(input[idx]);
            float w = __bfloat162float(w_row[idx]);
            sum += a * w;
        }
    }

    // Wave64 reduction via shfl_xor
    #pragma unroll
    for (int offset = 32; offset >= 1; offset >>= 1) {
        sum += __shfl_xor(sum, offset);
    }

    // Cross-warp reduction: 4 warps, lane 0 of each warp writes to LDS
    int warp_id = tid >> 6;
    int lane_id = tid & 63;

    if (lane_id == 0) {
        smem_reduce[warp_id] = sum;
    }
    __syncthreads();

    if (tid == 0) {
        float total = smem_reduce[0] + smem_reduce[1] + smem_reduce[2] + smem_reduce[3];
        // Add bias
        total += __bfloat162float(router_bias[expert_id]);
        // Store as bf16
        logit_out[expert_id] = __float2bfloat16(total);
    }
    __syncthreads();
}
#endif // TITAN_ENABLE_LEGACY (rmsnorm_resadd + fp8_quant + router_gemv_bf16)

// ============================================================================
// Section D: Barrier Primitives
// ============================================================================
// Wrappers around the low-level atomics from common.cuh.

// Hierarchical barrier: all workers across all XCDs synchronize.
// Two-phase arrival: first per-XCD (local_arrival), then last per XCD does
// buffer_wbl2 and increments global counter. Only 1 worker/XCD pays the
// buffer_wbl2 cost (vs all 12 workers in the flat barrier).
// local_arrival: per-XCD arrival counter at local_arrival[xcd_id * 16].
__device__ __forceinline__ void
barrier_global(int *counter, int expected, int total_workers,
               int *local_arrival,
               int xcd_id, int workers_per_xcd) {
    int tid = threadIdx.x;
    __syncthreads();

    // Phase 1: Per-XCD arrival — drain this workgroup's stores into L2
    if (tid == 0) {
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
        int local_prev = atom_add_release_gpu_s32(
            &local_arrival[xcd_id * 16], 1);

        // Last worker on this XCD: flush XCD's L2 → HBM, then global arrive
        if (local_prev % workers_per_xcd == workers_per_xcd - 1) {
            asm volatile("buffer_wbl2" ::: "memory");
            asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
            atom_add_release_gpu_s32(counter, 1);
        }
    }

    // All workers poll global counter
    if (tid == 0) {
        while (__atomic_load_n(counter, __ATOMIC_RELAXED) < expected) {
            __builtin_amdgcn_s_sleep(1);
        }
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
    __syncthreads();
    asm volatile("buffer_inv" ::: "memory");
}

// Hierarchical barrier with agent-scope fence instead of buffer_wbl2.
__device__ __forceinline__ void
barrier_global_fence(int *counter, int expected, int total_workers,
                     int *local_arrival,
                     int xcd_id, int workers_per_xcd) {
    int tid = threadIdx.x;
    __syncthreads();

    if (tid == 0) {
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
        int local_prev = atom_add_release_gpu_s32(
            &local_arrival[xcd_id * 16], 1);

        // Last worker on this XCD: agent fence + global arrive
        if (local_prev % workers_per_xcd == workers_per_xcd - 1) {
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
            atom_add_release_gpu_s32(counter, 1);
        }
    }

    if (tid == 0) {
        while (__atomic_load_n(counter, __ATOMIC_RELAXED) < expected) {
            __builtin_amdgcn_s_sleep(1);
        }
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
    __syncthreads();
    asm volatile("buffer_inv" ::: "memory");
}

// Write-through release barrier without buffer_wbl2.
// Arrival: per-XCD local → last per-XCD → global counter → last globally → st_wt release.
// Poll: per-XCD ld_nt on release flag.
// Cheaper than barrier_global because:
//   1) No buffer_wbl2 (cross-XCD writes are via atomicAdd/st_wt, already L2-coherent)
//   2) Workers poll per-XCD flag (no cross-XCD counter polling traffic)
__device__ __forceinline__ void
barrier_wt_release_no_wbl2(int *counter, int *local_arrival,
                           int *release_base,
                           int xcd_id, int workers_per_xcd) {
    int tid = threadIdx.x;
    __syncthreads();

    // Snapshot expected release value BEFORE any atomic
    __shared__ int s_release_expected;
    if (tid == 0) {
        s_release_expected = ld_nt_s32(&release_base[xcd_id * 16]) + 1;
    }
    __syncthreads();

    // Phase 1: per-XCD local arrive
    if (tid == 0) {
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
        int local_prev = atom_add_release_gpu_s32(
            &local_arrival[xcd_id * 16], 1);

        // Last worker on this XCD: agent fence + global arrive
        if (local_prev % workers_per_xcd == workers_per_xcd - 1) {
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
            int global_prev = atom_add_release_gpu_s32(counter, 1);

            // Last XCD globally: fan out release flags
            if (global_prev % 8 == 7) {
                for (int x = 0; x < 8; x++) {
                    st_wt_u32((void *)&release_base[x * 16],
                              (unsigned)s_release_expected);
                }
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            }
        }
    }

    // Phase 2: all workers poll per-XCD release flag
    if (tid == 0) {
        while (ld_nt_s32(&release_base[xcd_id * 16]) < s_release_expected) {
            __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();
    // NOTE: buffer_inv intentionally omitted — caller (gang_full_layer_fused)
    // issues buffer_inv at the start of each layer.
}

// Minimal barrier: threadfence_gpu + flat global counter.
// No per-XCD hierarchy, no buffer_wbl2. Relies on atomicAdd being L2-coherent.
// MEASURED 8.69µs/barrier as the gpt-oss-120b inter-layer barrier, vs 3.56µs
// for barrier_wt_release_no_wbl2 — i.e. 2.4x WORSE, not better. The flat
// counter puts all 240 workgroups on one cache line across 8 XCDs; the
// hierarchy is what makes the WT-release version cheap. Do not swap this in
// on the strength of it looking simpler.
__device__ __forceinline__ void
barrier_fence_minimal(int *counter, int xcd_id, int workers_per_xcd) {
    int tid = threadIdx.x;
    __syncthreads();
    if (tid == 0) {
        // Agent-scope fence ensures prior stores (including MoE atomicAdd)
        // are visible to all XCDs via L2 coherence protocol.
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        // Global arrive — no per-XCD hierarchy, just flat counter.
        int prev = __atomic_fetch_add(counter, 1, __ATOMIC_RELAXED);
        int total = 8 * workers_per_xcd;
        int expected = ((prev / total) + 1) * total;
        // Poll until all workers arrived
        while (__atomic_load_n(counter, __ATOMIC_RELAXED) < expected) {
            __builtin_amdgcn_s_sleep(1);
        }
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
    __syncthreads();
    asm volatile("buffer_inv" ::: "memory");
}

// Selective buffer_wbl2 barrier: only workers where is_writer=true do wbl2.
// Use when only some workers wrote data that needs cross-XCD visibility.
__device__ __forceinline__ void
barrier_global_selective_wbl2(int *counter, int expected, int total_workers,
                               bool is_writer) {
    int tid = threadIdx.x;
    __syncthreads();
    if (tid == 0) {
        if (is_writer) {
            asm volatile("buffer_wbl2" ::: "memory");
        }
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
        atom_add_release_gpu_s32(counter, 1);
    }
    if (tid == 0) {
        while (__atomic_load_n(counter, __ATOMIC_RELAXED) < expected) {
            __builtin_amdgcn_s_sleep(1);
        }
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
    __syncthreads();
    asm volatile("buffer_inv" ::: "memory");
}

// Legacy barrier_global without release flags (uses buffer_wbl2 fallback)
__device__ __forceinline__ void
barrier_global(int *counter, int expected, int total_workers) {
    int tid = threadIdx.x;
    __syncthreads();
    if (tid == 0) {
        asm volatile("buffer_wbl2" ::: "memory");
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
        atom_add_release_gpu_s32(counter, 1);
    }
    if (tid == 0) {
        while (__atomic_load_n(counter, __ATOMIC_RELAXED) < expected) {
            __builtin_amdgcn_s_sleep(1);
        }
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
    __syncthreads();
    asm volatile("buffer_inv" ::: "memory");
}

// Local barrier: all workers synchronize but NO L2 writeback.
// Use when preceding stores are either redundant (all workers wrote same data)
// or will only be read by the same XCD that wrote them.
__device__ __forceinline__ void
barrier_global_local(int *counter, int expected, int total_workers) {
    int tid = threadIdx.x;
    __syncthreads();
    if (tid == 0) {
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
        atom_add_release_gpu_s32(counter, 1);
    }
    if (tid == 0) {
        while (__atomic_load_n(counter, __ATOMIC_RELAXED) < expected) {
            __builtin_amdgcn_s_sleep(1);
        }
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
    __syncthreads();
    asm volatile("buffer_inv" ::: "memory");
}

// Single-producer barrier with wbl2: one worker writes, all wait.
// is_producer=true on the single worker that produces cross-XCD-visible data.
// Only the producer pays buffer_wbl2. All workers poll a single counter.
// expected = ((cur / 1) + 1) * 1 = cur + 1 (since only 1 writer increments).
__device__ __forceinline__ void
barrier_single_producer(int *counter, int expected, bool is_producer) {
    int tid = threadIdx.x;
    if (is_producer) {
        __syncthreads();
        if (tid == 0) {
            asm volatile("buffer_wbl2" ::: "memory");
            asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
            atom_add_release_gpu_s32(counter, 1);
        }
    }
    // All workers wait for producer
    if (tid == 0) {
        while (__atomic_load_n(counter, __ATOMIC_RELAXED) < expected) {
            __builtin_amdgcn_s_sleep(1);
        }
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
    __syncthreads();
    asm volatile("buffer_inv" ::: "memory");
}

// Per-XCD epoch barrier: only synchronizes workers within one XCD.
// Uses epoch counter (monotonic, no reset needed) at epoch_base[xcd_id * 16].
// Much cheaper than barrier_global: no cross-XCD sync, no buffer_wbl2/inv.
// workers_per_xcd = number of workers that participate on this XCD.
__device__ __forceinline__ void
barrier_xcd_epoch(int *epoch_base, int xcd_id, int workers_per_xcd) {
    int tid = threadIdx.x;
    __syncthreads();
    if (tid == 0) {
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
    }
    // All workers atomicAdd to the per-XCD arrival counter.
    // When count reaches workers_per_xcd, last worker bumps epoch.
    // Others poll epoch.
    int *arrival = epoch_base + xcd_id * 16;
    int *epoch = epoch_base + xcd_id * 16 + 1;  // epoch at offset +1

    __shared__ int s_prev_arrival;
    __shared__ int s_epoch_expected;
    if (tid == 0) {
        s_epoch_expected = __atomic_load_n(epoch, __ATOMIC_RELAXED) + 1;
        s_prev_arrival = atomicAdd(arrival, 1);
    }
    __syncthreads();

    if (s_prev_arrival % workers_per_xcd == workers_per_xcd - 1) {
        // Last worker to arrive: bump epoch
        if (tid == 0) {
            atom_add_release_gpu_s32(epoch, 1);
        }
    }

    // All workers poll epoch
    if (tid == 0) {
        while (__atomic_load_n(epoch, __ATOMIC_RELAXED) < s_epoch_expected) {
            __builtin_amdgcn_s_sleep(1);
        }
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
    __syncthreads();
    // No buffer_inv needed — same-XCD L2 is coherent
}

// Write-through cross-XCD barrier: all workers on all XCDs synchronize.
// Uses per-XCD release flags written via st_wt (write-through to HBM).
// Other XCDs poll via ld_nt (non-temporal, bypasses L2).
// Includes buffer_wbl2 to flush GEMM output stores for cross-XCD visibility,
// but uses ld_nt polling instead of buffer_inv on the consumer side.
// counter: single global arrival counter
// release_base: 8 per-XCD release flags at release_base[xcd_id * 16]
// xcd_id: this worker's XCD index
// total_workers: total across all XCDs (e.g., 240)
__device__ __forceinline__ void
barrier_wt_release(int *counter, int *release_base, int xcd_id, int total_workers) {
    int tid = threadIdx.x;
    __syncthreads();
    if (tid == 0) {
        // Flush GEMM output from local L2 to HBM for cross-XCD visibility
        asm volatile("buffer_wbl2" ::: "memory");
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
    }

    // Snapshot expected release value before any atomic
    __shared__ int s_release_expected;
    if (tid == 0) {
        s_release_expected = ld_nt_s32(&release_base[xcd_id * 16]) + 1;
    }
    __syncthreads();

    // Global arrive
    if (tid == 0) {
        int prev = atom_add_release_gpu_s32(counter, 1);
        int expected = ((prev / total_workers) + 1) * total_workers;
        if (prev == expected - 1) {
            // Last worker globally: fan out release flags via write-through
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            for (int x = 0; x < 8; x++) {
                st_wt_u32((void *)&release_base[x * 16],
                          (unsigned)s_release_expected);
            }
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        }
    }

    // All workers poll their own XCD's release flag via ld_nt
    if (tid == 0) {
        while (ld_nt_s32(&release_base[xcd_id * 16]) < s_release_expected) {
            __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();
    // After release, invalidate L2 so subsequent reads see cross-XCD data
    asm volatile("buffer_inv" ::: "memory");
}

// Inter-layer barrier: agent-scope fence on leader, block sync for all.
__device__ __forceinline__ void
inter_layer_sync() {
    int tid = threadIdx.x;
    if (tid == 0) {
        threadfence_gpu();
    }
    __syncthreads();
}

// L2 invalidate + fence for cache coherence after barriers.
__device__ __forceinline__ void
invalidate_l2() {
    asm volatile("buffer_inv" ::: "memory");
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
}

// Final writeback: flush all pending stores to HBM.
__device__ __forceinline__ void
final_writeback() {
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    asm volatile("buffer_wbl2" ::: "memory");
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
}


#ifdef TITAN_ENABLE_LEGACY
// ============================================================================
// Section E: RoPE + KV Cache Update
// ============================================================================
// Standalone RoPE application + paged KV cache write.
// Called AFTER QKV GEMM output is available in HBM.
// This replaces the fused KV-update epilogue from mirage's QKV kernel.
//
// QKV output layout (interleaved by KV groups, matching weight packing):
//   For each KV head g in [0, NUM_KV_HEADS):
//     Q: columns [g*GS .. g*GS + NUM_Q_PER_KV*HD)
//     K: columns [g*GS + NUM_Q_PER_KV*HD .. g*GS + (NUM_Q_PER_KV+1)*HD)
//     V: columns [g*GS + (NUM_Q_PER_KV+1)*HD .. g*GS + (NUM_Q_PER_KV+2)*HD)
//   where GS = (NUM_Q_PER_KV+2)*HD = group stride
//
// For Qwen3-8B: NUM_Q_PER_KV=4, HD=128, NUM_KV_HEADS=8
//   Group stride = (4+2)*128 = 768
//   QKV_OUTPUT_SIZE = 8 * 768 = 6144

template<int HEAD_DIM, int NUM_Q_PER_KV, int NUM_KV_HEADS, int PAGE_SIZE,
         bool HAS_QK_NORM = true>
__device__ __noinline__ void
rope_kv_update(
    const unsigned short *qkv_output,  // [bs, QKV_OUTPUT_SIZE] bf16
    const unsigned short *cos_ptr,     // [max_seq, HEAD_DIM] bf16
    const unsigned short *sin_ptr,     // [max_seq, HEAD_DIM] bf16
    unsigned short *q_workspace,       // [bs, NUM_Q_HEADS * HEAD_DIM] bf16
    unsigned short *k_cache,           // paged [num_pages*page_size, kv_stride]
    unsigned short *v_cache,           // paged
    const int *qo_indptr,
    const int *kv_indptr,
    const int *kv_indices,
    const int *kv_last_page_len,
    int kv_head_idx,                   // which KV head (= xcd_id)
    int kv_cache_stride,               // NUM_KV_HEADS * HEAD_DIM
    int q_ws_stride,                   // NUM_Q_PER_KV * HEAD_DIM
    const unsigned short *q_norm_weight = nullptr,  // [HEAD_DIM] bf16 (per-head QK norm)
    const unsigned short *k_norm_weight = nullptr)  // [HEAD_DIM] bf16 (per-head QK norm)
{
    int tid = threadIdx.x;
    constexpr int HALF = HEAD_DIM / 2;
    constexpr int GROUP_STRIDE = (NUM_Q_PER_KV + 2) * HEAD_DIM;

    // Compute position for RoPE
    int request_id = 0;
    while (qo_indptr[request_id + 1] <= 0) request_id++;
    int first_page = kv_indptr[request_id];
    int num_pages = kv_indptr[request_id + 1] - first_page;
    int last_page_len_val = kv_last_page_len[request_id];
    int global_seq_len = (num_pages - 1) * PAGE_SIZE + last_page_len_val;
    int num_new_tokens = qo_indptr[request_id + 1] - qo_indptr[request_id];
    int global_pos = global_seq_len - num_new_tokens;

    const unsigned short *cos_row = cos_ptr + global_pos * HEAD_DIM;
    const unsigned short *sin_row = sin_ptr + global_pos * HEAD_DIM;

    // Base of this KV head group in QKV output
    int group_base = kv_head_idx * GROUP_STRIDE;

    // LDS for RoPE staging + reduction scratch
    extern __shared__ char _rope_smem[];
    unsigned short *s_rope = (unsigned short *)_rope_smem;
    // Use space after s_rope for reduction (HEAD_DIM bf16 = 256 bytes, well within LDS)
    float *s_reduce = reinterpret_cast<float *>(s_rope + HEAD_DIM);

    // ── Process Q heads (QK norm + RoPE, write to q_workspace) ──
    for (int q = 0; q < NUM_Q_PER_KV; q++) {
        int q_head_global = kv_head_idx * NUM_Q_PER_KV + q;
        int src_offset = group_base + q * HEAD_DIM;

        // Load Q values to LDS
        for (int d = tid; d < HEAD_DIM; d += 256) {
            s_rope[d] = qkv_output[src_offset + d];
        }
        __syncthreads();

        // QK normalization: RMSNorm per head (HEAD_DIM=128 elements)
        if constexpr (HAS_QK_NORM) {
            // Step 1: compute sum of squares
            float my_sq = 0.0f;
            for (int d = tid; d < HEAD_DIM; d += 256) {
                unsigned vb = (unsigned)s_rope[d] << 16;
                float v;
                __builtin_memcpy(&v, &vb, 4);
                my_sq += v * v;
            }
            // Warp-level reduction
            #pragma unroll
            for (int offset = 32; offset > 0; offset >>= 1) {
                my_sq += __shfl_xor(my_sq, offset, 64);
            }
            // Thread 0 of each warp writes to shared
            int warp_id = tid >> 6;
            int lane_id = tid & 63;
            if (lane_id == 0) s_reduce[warp_id] = my_sq;
            __syncthreads();
            // Thread 0 does final reduction
            if (tid == 0) {
                float total = 0.0f;
                for (int w = 0; w < 4; w++) total += s_reduce[w];
                float rms = __builtin_amdgcn_sqrtf(total / (float)HEAD_DIM + 1e-6f);
                s_reduce[0] = 1.0f / rms;
            }
            __syncthreads();
            float inv_rms = s_reduce[0];
            // Apply: x_normed[d] = (x[d] / rms) * weight[d]
            for (int d = tid; d < HEAD_DIM; d += 256) {
                unsigned vb = (unsigned)s_rope[d] << 16;
                float v;
                __builtin_memcpy(&v, &vb, 4);
                unsigned wb = (unsigned)q_norm_weight[d] << 16;
                float w;
                __builtin_memcpy(&w, &wb, 4);
                s_rope[d] = kernel::_gang_float_to_bf16(v * inv_rms * w);
            }
            __syncthreads();
        }

        // Apply RoPE: first HALF threads
        if (tid < HALF) {
            unsigned v0b = (unsigned)s_rope[tid] << 16;
            unsigned v1b = (unsigned)s_rope[tid + HALF] << 16;
            float v0, v1;
            __builtin_memcpy(&v0, &v0b, 4);
            __builtin_memcpy(&v1, &v1b, 4);

            unsigned cb = (unsigned)cos_row[tid] << 16;
            unsigned sb = (unsigned)sin_row[tid] << 16;
            float c, s;
            __builtin_memcpy(&c, &cb, 4);
            __builtin_memcpy(&s, &sb, 4);

            s_rope[tid]        = kernel::_gang_float_to_bf16(v0 * c - v1 * s);
            s_rope[tid + HALF] = kernel::_gang_float_to_bf16(v0 * s + v1 * c);
        }
        __syncthreads();

        // Write RoPE'd Q to workspace
        for (int d = tid; d < HEAD_DIM; d += 256) {
            q_workspace[q_head_global * HEAD_DIM + d] = s_rope[d];
        }
        __syncthreads();
    }

    // ── Process K head (QK norm + RoPE, write to paged K cache) ──
    {
        int src_offset = group_base + NUM_Q_PER_KV * HEAD_DIM;

        for (int d = tid; d < HEAD_DIM; d += 256) {
            s_rope[d] = qkv_output[src_offset + d];
        }
        __syncthreads();

        // QK normalization for K
        if constexpr (HAS_QK_NORM) {
            float my_sq = 0.0f;
            for (int d = tid; d < HEAD_DIM; d += 256) {
                unsigned vb = (unsigned)s_rope[d] << 16;
                float v;
                __builtin_memcpy(&v, &vb, 4);
                my_sq += v * v;
            }
            #pragma unroll
            for (int offset = 32; offset > 0; offset >>= 1) {
                my_sq += __shfl_xor(my_sq, offset, 64);
            }
            int warp_id = tid >> 6;
            int lane_id = tid & 63;
            if (lane_id == 0) s_reduce[warp_id] = my_sq;
            __syncthreads();
            if (tid == 0) {
                float total = 0.0f;
                for (int w = 0; w < 4; w++) total += s_reduce[w];
                float rms = __builtin_amdgcn_sqrtf(total / (float)HEAD_DIM + 1e-6f);
                s_reduce[0] = 1.0f / rms;
            }
            __syncthreads();
            float inv_rms = s_reduce[0];
            for (int d = tid; d < HEAD_DIM; d += 256) {
                unsigned vb = (unsigned)s_rope[d] << 16;
                float v;
                __builtin_memcpy(&v, &vb, 4);
                unsigned wb = (unsigned)k_norm_weight[d] << 16;
                float w;
                __builtin_memcpy(&w, &wb, 4);
                s_rope[d] = kernel::_gang_float_to_bf16(v * inv_rms * w);
            }
            __syncthreads();
        }

        if (tid < HALF) {
            unsigned v0b = (unsigned)s_rope[tid] << 16;
            unsigned v1b = (unsigned)s_rope[tid + HALF] << 16;
            float v0, v1;
            __builtin_memcpy(&v0, &v0b, 4);
            __builtin_memcpy(&v1, &v1b, 4);

            unsigned cb = (unsigned)cos_row[tid] << 16;
            unsigned sb = (unsigned)sin_row[tid] << 16;
            float c, s;
            __builtin_memcpy(&c, &cb, 4);
            __builtin_memcpy(&s, &sb, 4);

            s_rope[tid]        = kernel::_gang_float_to_bf16(v0 * c - v1 * s);
            s_rope[tid + HALF] = kernel::_gang_float_to_bf16(v0 * s + v1 * c);
        }
        __syncthreads();

        // Paged KV cache write
        int page_num = global_pos / PAGE_SIZE;
        int page_offset = global_pos % PAGE_SIZE;
        int page_idx = kv_indices[first_page + page_num];
        int dst_idx = page_idx * PAGE_SIZE + page_offset;

        for (int d = tid; d < HEAD_DIM; d += 256) {
            k_cache[dst_idx * kv_cache_stride + kv_head_idx * HEAD_DIM + d] = s_rope[d];
        }
        __syncthreads();
    }

    // ── Process V head (NO RoPE, direct write to paged V cache) ──
    {
        int src_offset = group_base + (NUM_Q_PER_KV + 1) * HEAD_DIM;

        int page_num = global_pos / PAGE_SIZE;
        int page_offset = global_pos % PAGE_SIZE;
        int page_idx = kv_indices[first_page + page_num];
        int dst_idx = page_idx * PAGE_SIZE + page_offset;

        for (int d = tid; d < HEAD_DIM; d += 256) {
            v_cache[dst_idx * kv_cache_stride + kv_head_idx * HEAD_DIM + d] =
                qkv_output[src_offset + d];
        }
        __syncthreads();
    }
}


// rope_kv_update_single_q: Process a single Q head only (for parallel RoPE)
// Called by xcd_rank < NUM_Q_PER_KV, each processing q_head = xcd_rank.
template<int HEAD_DIM, int NUM_Q_PER_KV, int NUM_KV_HEADS, int PAGE_SIZE,
         bool HAS_QK_NORM = true>
__device__ __noinline__ void
rope_single_q(
    const unsigned short *qkv_output,
    const unsigned short *cos_ptr,
    const unsigned short *sin_ptr,
    unsigned short *q_workspace,
    const int *qo_indptr,
    const int *kv_indptr,
    const int *kv_indices,
    const int *kv_last_page_len,
    int kv_head_idx,
    int q_local_idx,    // which Q head within this KV group (0..NUM_Q_PER_KV-1)
    const unsigned short *q_norm_weight = nullptr)
{
    int tid = threadIdx.x;
    constexpr int HALF = HEAD_DIM / 2;
    constexpr int GROUP_STRIDE = (NUM_Q_PER_KV + 2) * HEAD_DIM;

    int request_id = 0;
    while (qo_indptr[request_id + 1] <= 0) request_id++;
    int first_page = kv_indptr[request_id];
    int num_pages = kv_indptr[request_id + 1] - first_page;
    int last_page_len_val = kv_last_page_len[request_id];
    int global_seq_len = (num_pages - 1) * PAGE_SIZE + last_page_len_val;
    int num_new_tokens = qo_indptr[request_id + 1] - qo_indptr[request_id];
    int global_pos = global_seq_len - num_new_tokens;

    const unsigned short *cos_row = cos_ptr + global_pos * HEAD_DIM;
    const unsigned short *sin_row = sin_ptr + global_pos * HEAD_DIM;
    int group_base = kv_head_idx * GROUP_STRIDE;

    extern __shared__ char _gemm_smem[];
    unsigned short *s_rope = (unsigned short *)_gemm_smem;
    float *s_reduce = reinterpret_cast<float *>(s_rope + HEAD_DIM);

    int q_head_global = kv_head_idx * NUM_Q_PER_KV + q_local_idx;
    int src_offset = group_base + q_local_idx * HEAD_DIM;

    for (int d = tid; d < HEAD_DIM; d += 256) {
        s_rope[d] = qkv_output[src_offset + d];
    }
    __syncthreads();

    if constexpr (HAS_QK_NORM) {
        float my_sq = 0.0f;
        for (int d = tid; d < HEAD_DIM; d += 256) {
            unsigned vb = (unsigned)s_rope[d] << 16;
            float v;
            __builtin_memcpy(&v, &vb, 4);
            my_sq += v * v;
        }
        #pragma unroll
        for (int offset = 32; offset > 0; offset >>= 1) {
            my_sq += __shfl_xor(my_sq, offset, 64);
        }
        int warp_id = tid >> 6;
        int lane_id = tid & 63;
        if (lane_id == 0) s_reduce[warp_id] = my_sq;
        __syncthreads();
        if (tid == 0) {
            float total = 0.0f;
            for (int w = 0; w < 4; w++) total += s_reduce[w];
            float rms = __builtin_amdgcn_sqrtf(total / (float)HEAD_DIM + 1e-6f);
            s_reduce[0] = 1.0f / rms;
        }
        __syncthreads();
        float inv_rms = s_reduce[0];
        for (int d = tid; d < HEAD_DIM; d += 256) {
            unsigned vb = (unsigned)s_rope[d] << 16;
            float v;
            __builtin_memcpy(&v, &vb, 4);
            unsigned wb = (unsigned)q_norm_weight[d] << 16;
            float w;
            __builtin_memcpy(&w, &wb, 4);
            s_rope[d] = kernel::_gang_float_to_bf16(v * inv_rms * w);
        }
        __syncthreads();
    }

    if (tid < HALF) {
        unsigned v0b = (unsigned)s_rope[tid] << 16;
        unsigned v1b = (unsigned)s_rope[tid + HALF] << 16;
        float v0, v1;
        __builtin_memcpy(&v0, &v0b, 4);
        __builtin_memcpy(&v1, &v1b, 4);
        unsigned cb = (unsigned)cos_row[tid] << 16;
        unsigned sb = (unsigned)sin_row[tid] << 16;
        float c, s;
        __builtin_memcpy(&c, &cb, 4);
        __builtin_memcpy(&s, &sb, 4);
        s_rope[tid]        = kernel::_gang_float_to_bf16(v0 * c - v1 * s);
        s_rope[tid + HALF] = kernel::_gang_float_to_bf16(v0 * s + v1 * c);
    }
    __syncthreads();

    for (int d = tid; d < HEAD_DIM; d += 256) {
        q_workspace[q_head_global * HEAD_DIM + d] = s_rope[d];
    }
    __syncthreads();
}

// rope_kv_only: Process K head (with RoPE+QK norm) and V head (copy only)
template<int HEAD_DIM, int NUM_Q_PER_KV, int NUM_KV_HEADS, int PAGE_SIZE,
         bool HAS_QK_NORM = true>
__device__ __noinline__ void
rope_kv_only(
    const unsigned short *qkv_output,
    const unsigned short *cos_ptr,
    const unsigned short *sin_ptr,
    unsigned short *k_cache,
    unsigned short *v_cache,
    const int *qo_indptr,
    const int *kv_indptr,
    const int *kv_indices,
    const int *kv_last_page_len,
    int kv_head_idx,
    int kv_cache_stride,
    const unsigned short *k_norm_weight = nullptr)
{
    int tid = threadIdx.x;
    constexpr int HALF = HEAD_DIM / 2;
    constexpr int GROUP_STRIDE = (NUM_Q_PER_KV + 2) * HEAD_DIM;

    int request_id = 0;
    while (qo_indptr[request_id + 1] <= 0) request_id++;
    int first_page = kv_indptr[request_id];
    int num_pages = kv_indptr[request_id + 1] - first_page;
    int last_page_len_val = kv_last_page_len[request_id];
    int global_seq_len = (num_pages - 1) * PAGE_SIZE + last_page_len_val;
    int num_new_tokens = qo_indptr[request_id + 1] - qo_indptr[request_id];
    int global_pos = global_seq_len - num_new_tokens;

    const unsigned short *cos_row = cos_ptr + global_pos * HEAD_DIM;
    const unsigned short *sin_row = sin_ptr + global_pos * HEAD_DIM;
    int group_base = kv_head_idx * GROUP_STRIDE;

    extern __shared__ char _gemm_smem[];
    unsigned short *s_rope = (unsigned short *)_gemm_smem;
    float *s_reduce = reinterpret_cast<float *>(s_rope + HEAD_DIM);

    // K head
    {
        int src_offset = group_base + NUM_Q_PER_KV * HEAD_DIM;
        for (int d = tid; d < HEAD_DIM; d += 256) {
            s_rope[d] = qkv_output[src_offset + d];
        }
        __syncthreads();

        if constexpr (HAS_QK_NORM) {
            float my_sq = 0.0f;
            for (int d = tid; d < HEAD_DIM; d += 256) {
                unsigned vb = (unsigned)s_rope[d] << 16;
                float v;
                __builtin_memcpy(&v, &vb, 4);
                my_sq += v * v;
            }
            #pragma unroll
            for (int offset = 32; offset > 0; offset >>= 1) {
                my_sq += __shfl_xor(my_sq, offset, 64);
            }
            int warp_id = tid >> 6;
            int lane_id = tid & 63;
            if (lane_id == 0) s_reduce[warp_id] = my_sq;
            __syncthreads();
            if (tid == 0) {
                float total = 0.0f;
                for (int w = 0; w < 4; w++) total += s_reduce[w];
                float rms = __builtin_amdgcn_sqrtf(total / (float)HEAD_DIM + 1e-6f);
                s_reduce[0] = 1.0f / rms;
            }
            __syncthreads();
            float inv_rms = s_reduce[0];
            for (int d = tid; d < HEAD_DIM; d += 256) {
                unsigned vb = (unsigned)s_rope[d] << 16;
                float v;
                __builtin_memcpy(&v, &vb, 4);
                unsigned wb = (unsigned)k_norm_weight[d] << 16;
                float w;
                __builtin_memcpy(&w, &wb, 4);
                s_rope[d] = kernel::_gang_float_to_bf16(v * inv_rms * w);
            }
            __syncthreads();
        }

        if (tid < HALF) {
            unsigned v0b = (unsigned)s_rope[tid] << 16;
            unsigned v1b = (unsigned)s_rope[tid + HALF] << 16;
            float v0, v1;
            __builtin_memcpy(&v0, &v0b, 4);
            __builtin_memcpy(&v1, &v1b, 4);
            unsigned cb = (unsigned)cos_row[tid] << 16;
            unsigned sb = (unsigned)sin_row[tid] << 16;
            float c, s;
            __builtin_memcpy(&c, &cb, 4);
            __builtin_memcpy(&s, &sb, 4);
            s_rope[tid]        = kernel::_gang_float_to_bf16(v0 * c - v1 * s);
            s_rope[tid + HALF] = kernel::_gang_float_to_bf16(v0 * s + v1 * c);
        }
        __syncthreads();

        int page_num = global_pos / PAGE_SIZE;
        int page_offset = global_pos % PAGE_SIZE;
        int page_idx = kv_indices[first_page + page_num];
        int dst_idx = page_idx * PAGE_SIZE + page_offset;
        for (int d = tid; d < HEAD_DIM; d += 256) {
            k_cache[dst_idx * kv_cache_stride + kv_head_idx * HEAD_DIM + d] = s_rope[d];
        }
        __syncthreads();
    }

    // V head (no RoPE, direct copy)
    {
        int src_offset = group_base + (NUM_Q_PER_KV + 1) * HEAD_DIM;
        int page_num = global_pos / PAGE_SIZE;
        int page_offset = global_pos % PAGE_SIZE;
        int page_idx = kv_indices[first_page + page_num];
        int dst_idx = page_idx * PAGE_SIZE + page_offset;
        for (int d = tid; d < HEAD_DIM; d += 256) {
            v_cache[dst_idx * kv_cache_stride + kv_head_idx * HEAD_DIM + d] =
                qkv_output[src_offset + d];
        }
        __syncthreads();
    }
}

// ============================================================================
// Section F: MoE Epilogue Templates
// ============================================================================
// Epilogue templates for MoE expert GEMMs (W13+SwiGLU, W2+atomicAdd).
// These work with the SAME gemm_mxfp4 mainloop as dense epilogues.

// Import fast_swigluoai for W13 epilogue
#include "tasks/mi300/swigluoai_mi300.cuh"

// ── EpilogueMoeSwiGLU: W13 gate/up interleaved GEMM → SwiGLU activation ──
// W13 weights are interleaved: consecutive column pairs (2i, 2i+1) map to
// (gate_i, up_i). The MFMA accumulator for each tile has 4 values:
// acc[0]=gate_0, acc[1]=up_0, acc[2]=gate_1, acc[3]=up_1.
// Each pair produces one SwiGLU output element.
//
// This epilogue is called for EVERY tile_iter (not split gate/up across iters).
// Each call processes acc[0:1] and acc[2:3] as two gate/up pairs.
//
// GPT-OSS uses fast_swigluoai: gate clamped to [-inf, 7], up to [-7, 7],
// glu = gate * sigmoid(1.702 * gate), output = (up + 1) * glu
struct EpilogueMoeSwiGLU {
    unsigned short *swiglu_out;   // [bs, topk, intermediate_size] bf16
    int act_stride;               // intermediate_size (per-topk output stride)
    int w13_output_size;          // 2 * intermediate_size (W13 total output dim)
    int topk_slot;                // which top-k slot this expert fills
    int tok_idx;                  // which batch token
    int num_topk;                 // k (for output address calculation)
    int expert_id;                // current expert index (for bias lookup)

    __device__ __forceinline__ void operator()(
        kernel::f32x4_t acc, int wave_tile, int col, int g,
        int n_tile, int m_tile, int OUTPUT_PER_WG,
        const unsigned short *bias, int batch_size) {
        if (col != 0) return;

        // Process pairs: (acc[0], acc[1]) and (acc[2], acc[3])
        for (int i = 0; i < 4; i += 2) {
            int out_n = n_tile * OUTPUT_PER_WG + wave_tile * 16 + g * 4 + i;
            if (out_n + 1 < w13_output_size) {
                float gate_val = acc[i];
                float up_val = acc[i + 1];

                // Add per-expert bias (caller already offsets bias ptr by expert_id)
                if (bias) {
                    unsigned bt_g = (unsigned)bias[out_n] << 16;
                    unsigned bt_u = (unsigned)bias[out_n + 1] << 16;
                    float bv_g, bv_u;
                    __builtin_memcpy(&bv_g, &bt_g, 4);
                    __builtin_memcpy(&bv_u, &bt_u, 4);
                    gate_val += bv_g;
                    up_val += bv_u;
                }

                float activated = kernel::fast_swigluoai(gate_val, up_val);

                // Output index: pair (2i, 2i+1) → output element i
                int act_n = out_n / 2;
                int out_idx = tok_idx * (num_topk * act_stride) +
                              topk_slot * act_stride + act_n;
                if (act_n < act_stride) {
                    st_wt_u16(&swiglu_out[out_idx],
                              kernel::_gang_float_to_bf16(activated));
                }
            }
        }
    }
};

// ── EpilogueMoeAtomicAdd: W2 down-projection → weighted atomicAdd to f32 workspace ──
// Each expert's W2 output is multiplied by the routing weight and accumulated
// into a shared f32 workspace via atomicAdd. After all experts complete,
// moe_residual_add converts f32 workspace + residual → bf16 output.
struct EpilogueMoeAtomicAdd {
    float *workspace_f32;          // [bs, hidden_size] f32 (atomicAdd target)
    const float *routing_weight;   // [bs, topk] f32
    int output_size;               // hidden_size (actual, for bounds check)
    int tok_idx;                   // which batch token
    int topk_slot;                 // which top-k slot
    int num_topk;                  // k
    int expert_id;                 // current expert (for bias lookup)

    __device__ __forceinline__ void operator()(
        kernel::f32x4_t acc, int wave_tile, int col, int g,
        int n_tile, int m_tile, int OUTPUT_PER_WG,
        const unsigned short *bias, int batch_size) {
        if (col != 0) return;

        int out_n_base = n_tile * OUTPUT_PER_WG + wave_tile * 16 + g * 4;
        if (out_n_base >= output_size) return;

        // Prefetched routing weight
        float rw = routing_weight[tok_idx * num_topk + topk_slot];

        for (int i = 0; i < 4; i++) {
            int out_n = out_n_base + i;
            if (out_n < output_size) {
                float val = acc[i];
                // Add bias (caller already offsets bias ptr by expert_id)
                if (bias) {
                    unsigned bt = (unsigned)bias[out_n] << 16;
                    float bv;
                    __builtin_memcpy(&bv, &bt, 4);
                    val += bv;
                }
                // Weighted accumulate into workspace
                atomicAdd(&workspace_f32[tok_idx * output_size + out_n], val * rw);
            }
        }
    }
};

// ── EpilogueMoeFused: Unified epilogue for W13 and W2 ──
// When is_w2 == false: performs SwiGLU (same as EpilogueMoeSwiGLU)
// When is_w2 == true:  performs weighted atomicAdd (same as EpilogueMoeAtomicAdd)
// This avoids a second template instantiation of gemm_mxfp4 which triggers
// a compiler issue on gfx950.
struct EpilogueMoeFused {
    bool is_w2;                    // false = W13+SwiGLU, true = W2+atomicAdd
    // W13 fields
    unsigned short *swiglu_out;    // [bs, topk, intermediate_size] bf16
    int act_stride;                // intermediate_size
    int w13_output_size;           // 2 * intermediate_size
    // W2 fields
    float *workspace_f32;          // [bs, hidden_size] f32
    const float *routing_weight;   // [bs, topk] f32
    int w2_output_size;            // hidden_size
    // Common
    int topk_slot;
    int tok_idx;
    int num_topk;
    int expert_id;

    __device__ __forceinline__ void operator()(
        kernel::f32x4_t acc, int wave_tile, int col, int g,
        int n_tile, int m_tile, int OUTPUT_PER_WG,
        const unsigned short *bias, int batch_size) {
        if (col != 0) return;

        if (!is_w2) {
            // ── W13 + SwiGLU path ──
            for (int i = 0; i < 4; i += 2) {
                int out_n = n_tile * OUTPUT_PER_WG + wave_tile * 16 + g * 4 + i;
                if (out_n + 1 < w13_output_size) {
                    float gate_val = acc[i];
                    float up_val = acc[i + 1];
                    if (bias) {
                        unsigned bt_g = (unsigned)bias[out_n] << 16;
                        unsigned bt_u = (unsigned)bias[out_n + 1] << 16;
                        float bv_g, bv_u;
                        __builtin_memcpy(&bv_g, &bt_g, 4);
                        __builtin_memcpy(&bv_u, &bt_u, 4);
                        gate_val += bv_g;
                        up_val += bv_u;
                    }
                    float activated = kernel::fast_swigluoai(gate_val, up_val);
                    int act_n = out_n / 2;
                    int out_idx = tok_idx * (num_topk * act_stride) +
                                  topk_slot * act_stride + act_n;
                    if (act_n < act_stride) {
                        st_wt_u16(&swiglu_out[out_idx],
                                  kernel::_gang_float_to_bf16(activated));
                    }
                }
            }
        } else {
            // ── W2 + atomicAdd path ──
            int out_n_base = n_tile * OUTPUT_PER_WG + wave_tile * 16 + g * 4;
            if (out_n_base >= w2_output_size) return;
            float rw = routing_weight[tok_idx * num_topk + topk_slot];
            for (int i = 0; i < 4; i++) {
                int out_n = out_n_base + i;
                if (out_n < w2_output_size) {
                    float val = acc[i];
                    if (bias) {
                        unsigned bt = (unsigned)bias[out_n] << 16;
                        float bv;
                        __builtin_memcpy(&bv, &bt, 4);
                        val += bv;
                    }
                    atomicAdd(&workspace_f32[tok_idx * w2_output_size + out_n], val * rw);
                }
            }
        }
    }
};


// ============================================================================
// Section G: MoE TopK + Softmax
// ============================================================================
// Branchless top-k selection from router logits.
// Produces routing_indices [E, batch], topk_weight [batch, topk], mask [E+1].
//
// Template parameters:
//   NUM_EXPERTS: total experts (e.g., 128)
//   NUM_TOPK: experts per token (e.g., 4)
// All 256 threads collaborate on one token's router output.
//
// Adapted from mirage's moe_topk_softmax_mi300.cuh.

template<int NUM_EXPERTS, int NUM_TOPK>
__device__ __noinline__ void
moe_topk_softmax(
    const unsigned short *router_output,  // [1, NUM_EXPERTS] bf16 (biased router logits)
    int *routing_indices,                 // [NUM_EXPERTS, bs] int32 (output: topk_slot+1 or 0)
    float *topk_weight,                   // [bs, NUM_TOPK] f32 (output: softmax weights)
    int *mask,                            // [NUM_EXPERTS+1] int32 (output: active expert IDs + count)
    int tok_idx,                          // which batch token
    int batch_size)                       // total batch size
{
    int tid = threadIdx.x;

    // Load router logits into registers (NUM_EXPERTS / 256 values per thread)
    constexpr int VPT = (NUM_EXPERTS + 255) / 256;
    float vals[VPT];
    #pragma unroll
    for (int i = 0; i < VPT; i++) {
        int idx = tid + i * 256;
        if (idx < NUM_EXPERTS) {
            unsigned bt = (unsigned)router_output[idx] << 16;
            __builtin_memcpy(&vals[i], &bt, 4);
        } else {
            vals[i] = -1e30f;
        }
    }

    // Softmax: find max for numerical stability
    float thread_max = -1e30f;
    #pragma unroll
    for (int i = 0; i < VPT; i++) {
        thread_max = fmaxf(thread_max, vals[i]);
    }
    // Warp-level max reduction
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_xor(thread_max, offset, 64));
    }
    // Cross-warp reduction via LDS
    extern __shared__ char _topk_smem[];
    float *s_reduce = (float *)_topk_smem;
    int warp_id = tid >> 6;
    int lane_id = tid & 63;
    if (lane_id == 0) s_reduce[warp_id] = thread_max;
    __syncthreads();
    if (tid == 0) {
        float m = s_reduce[0];
        for (int w = 1; w < 4; w++) m = fmaxf(m, s_reduce[w]);
        s_reduce[0] = m;
    }
    __syncthreads();
    float global_max = s_reduce[0];

    // Compute exp(x - max) and sum
    float thread_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < VPT; i++) {
        vals[i] = __expf(vals[i] - global_max);
        int idx = tid + i * 256;
        if (idx < NUM_EXPERTS) thread_sum += vals[i];
    }
    // Warp-level sum reduction
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        thread_sum += __shfl_xor(thread_sum, offset, 64);
    }
    if (lane_id == 0) s_reduce[warp_id] = thread_sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (int w = 0; w < 4; w++) total += s_reduce[w];
        s_reduce[0] = 1.0f / total;
    }
    __syncthreads();
    float inv_sum = s_reduce[0];

    // Normalize
    #pragma unroll
    for (int i = 0; i < VPT; i++) {
        vals[i] *= inv_sum;
    }

    // Zero routing_indices for this token
    for (int e = tid; e < NUM_EXPERTS; e += 256) {
        routing_indices[e * batch_size + tok_idx] = 0;
    }
    __syncthreads();

    // Iterative branchless top-k selection
    int *s_winner_idx = (int *)&s_reduce[4];   // scratch for winner index
    float *s_winner_val = (float *)&s_reduce[8]; // scratch for winner value
    int mask_count = 0;

    for (int k = 0; k < NUM_TOPK; k++) {
        // Find local max across VPT elements
        float local_max = -1.0f;
        int local_idx = -1;
        #pragma unroll
        for (int i = 0; i < VPT; i++) {
            int idx = tid + i * 256;
            if (idx < NUM_EXPERTS && vals[i] > local_max) {
                local_max = vals[i];
                local_idx = idx;
            }
        }

        // Warp-level argmax reduction
        #pragma unroll
        for (int offset = 32; offset > 0; offset >>= 1) {
            float other_val = __shfl_xor(local_max, offset, 64);
            int other_idx = __shfl_xor(local_idx, offset, 64);
            if (other_val > local_max || (other_val == local_max && other_idx < local_idx)) {
                local_max = other_val;
                local_idx = other_idx;
            }
        }

        // Cross-warp argmax via LDS
        if (lane_id == 0) {
            s_reduce[warp_id] = local_max;
            s_winner_idx[warp_id] = local_idx;
        }
        __syncthreads();

        if (tid == 0) {
            float best_val = s_reduce[0];
            int best_idx = s_winner_idx[0];
            for (int w = 1; w < 4; w++) {
                if (s_reduce[w] > best_val ||
                    (s_reduce[w] == best_val && s_winner_idx[w] < best_idx)) {
                    best_val = s_reduce[w];
                    best_idx = s_winner_idx[w];
                }
            }
            s_winner_val[0] = best_val;
            s_winner_idx[0] = best_idx;
        }
        __syncthreads();

        int winner = s_winner_idx[0];
        float winner_val = s_winner_val[0];

        // Record winner
        if (tid == 0) {
            topk_weight[tok_idx * NUM_TOPK + k] = winner_val;
            routing_indices[winner * batch_size + tok_idx] = k + 1;  // 1-indexed
            mask[mask_count] = winner;
            mask_count++;
        }

        // Blank winner for next iteration
        #pragma unroll
        for (int i = 0; i < VPT; i++) {
            if (tid + i * 256 == winner) vals[i] = -1.0f;
        }
        __syncthreads();
    }

    // Renormalize top-k weights to sum to 1.0
    if (tid == 0) {
        float sum = 0.0f;
        for (int k = 0; k < NUM_TOPK; k++) {
            sum += topk_weight[tok_idx * NUM_TOPK + k];
        }
        float inv_sum = 1.0f / sum;
        for (int k = 0; k < NUM_TOPK; k++) {
            topk_weight[tok_idx * NUM_TOPK + k] *= inv_sum;
        }
    }

    // Write count to mask[NUM_EXPERTS]
    if (tid == 0) {
        mask[NUM_EXPERTS] = mask_count;
    }
    __syncthreads();
}


// ============================================================================
// Section H: MoE Residual Add (f32 workspace → bf16 output)
// ============================================================================
// After all W2 experts complete their atomicAdd into workspace_f32,
// this function adds the residual (bf16) and converts back to bf16 output,
// then zeros the workspace for the next layer.
//
// Adapted from mirage's moe_residual_add_f32_mi300.cuh.

template<int HIDDEN_SIZE, int ACTUAL_HIDDEN_DIM>
__device__ __noinline__ void
moe_residual_add(
    float *workspace_f32,              // [bs, HIDDEN_SIZE] f32 (input, zeroed after use)
    const unsigned short *residual,    // [bs, HIDDEN_SIZE] bf16
    unsigned short *output,            // [bs, HIDDEN_SIZE] bf16
    int tok_idx)
{
    int tid = threadIdx.x;

    // Process 4 elements per thread for vectorization
    constexpr int VEC = 4;
    constexpr int BLOCK_VEC = 256 * VEC;  // 1024 elements per pass

    int ws_base = tok_idx * HIDDEN_SIZE;

    for (int base = tid * VEC; base < ACTUAL_HIDDEN_DIM; base += BLOCK_VEC) {
        if (base + VEC <= ACTUAL_HIDDEN_DIM) {
            // Vectorized path: load 4 f32 from workspace
            float w0 = workspace_f32[ws_base + base + 0];
            float w1 = workspace_f32[ws_base + base + 1];
            float w2 = workspace_f32[ws_base + base + 2];
            float w3 = workspace_f32[ws_base + base + 3];

            // Load 4 bf16 residual values, convert to f32
            unsigned r01_packed;
            unsigned r23_packed;
            __builtin_memcpy(&r01_packed, &residual[base], 4);
            __builtin_memcpy(&r23_packed, &residual[base + 2], 4);

            unsigned r0t = (r01_packed & 0xFFFFu) << 16;
            unsigned r1t = r01_packed & 0xFFFF0000u;
            unsigned r2t = (r23_packed & 0xFFFFu) << 16;
            unsigned r3t = r23_packed & 0xFFFF0000u;
            float r0, r1, r2, r3;
            __builtin_memcpy(&r0, &r0t, 4);
            __builtin_memcpy(&r1, &r1t, 4);
            __builtin_memcpy(&r2, &r2t, 4);
            __builtin_memcpy(&r3, &r3t, 4);

            // Add and convert back to bf16 with banker's rounding
            float s0 = w0 + r0, s1 = w1 + r1, s2 = w2 + r2, s3 = w3 + r3;
            output[base + 0] = kernel::_gang_float_to_bf16(s0);
            output[base + 1] = kernel::_gang_float_to_bf16(s1);
            output[base + 2] = kernel::_gang_float_to_bf16(s2);
            output[base + 3] = kernel::_gang_float_to_bf16(s3);

            // Zero workspace for next layer
            workspace_f32[ws_base + base + 0] = 0.0f;
            workspace_f32[ws_base + base + 1] = 0.0f;
            workspace_f32[ws_base + base + 2] = 0.0f;
            workspace_f32[ws_base + base + 3] = 0.0f;
        } else {
            // Scalar tail
            for (int i = 0; i < VEC && base + i < ACTUAL_HIDDEN_DIM; i++) {
                float w = workspace_f32[ws_base + base + i];
                unsigned rt = (unsigned)residual[base + i] << 16;
                float r;
                __builtin_memcpy(&r, &rt, 4);
                output[base + i] = kernel::_gang_float_to_bf16(w + r);
                workspace_f32[ws_base + base + i] = 0.0f;
            }
        }
    }
    __syncthreads();
}


#endif // TITAN_ENABLE_LEGACY (RoPE + MoE epilogues + TopK + moe_residual_add)

// ── moe_residual_add_no_zero: same as moe_residual_add but does NOT zero workspace.
// Used when workspace zeroing is handled separately (after a barrier) to avoid
// a race condition where one worker zeros ws[i] before another reads it.
template<int HIDDEN_SIZE, int ACTUAL_HIDDEN_DIM>
__device__ __noinline__ void
moe_residual_add_no_zero(
    const float *workspace_f32,        // [bs, HIDDEN_SIZE] f32 (read-only)
    const unsigned short *residual,    // [bs, HIDDEN_SIZE] bf16
    unsigned short *output,            // [bs, HIDDEN_SIZE] bf16
    int tok_idx)
{
    int tid = threadIdx.x;
    constexpr int VEC = 4;
    constexpr int BLOCK_VEC = 256 * VEC;
    int ws_base = tok_idx * HIDDEN_SIZE;

    for (int base = tid * VEC; base < ACTUAL_HIDDEN_DIM; base += BLOCK_VEC) {
        if (base + VEC <= ACTUAL_HIDDEN_DIM) {
            float w0 = workspace_f32[ws_base + base + 0];
            float w1 = workspace_f32[ws_base + base + 1];
            float w2 = workspace_f32[ws_base + base + 2];
            float w3 = workspace_f32[ws_base + base + 3];

            unsigned r01_packed, r23_packed;
            __builtin_memcpy(&r01_packed, &residual[base], 4);
            __builtin_memcpy(&r23_packed, &residual[base + 2], 4);

            unsigned r0t = (r01_packed & 0xFFFFu) << 16;
            unsigned r1t = r01_packed & 0xFFFF0000u;
            unsigned r2t = (r23_packed & 0xFFFFu) << 16;
            unsigned r3t = r23_packed & 0xFFFF0000u;
            float r0, r1, r2, r3;
            __builtin_memcpy(&r0, &r0t, 4);
            __builtin_memcpy(&r1, &r1t, 4);
            __builtin_memcpy(&r2, &r2t, 4);
            __builtin_memcpy(&r3, &r3t, 4);

            float s0 = w0 + r0, s1 = w1 + r1, s2 = w2 + r2, s3 = w3 + r3;
            output[base + 0] = kernel::_gang_float_to_bf16(s0);
            output[base + 1] = kernel::_gang_float_to_bf16(s1);
            output[base + 2] = kernel::_gang_float_to_bf16(s2);
            output[base + 3] = kernel::_gang_float_to_bf16(s3);
        } else {
            for (int i = 0; i < VEC && base + i < ACTUAL_HIDDEN_DIM; i++) {
                float w = workspace_f32[ws_base + base + i];
                unsigned rt = (unsigned)residual[base + i] << 16;
                float r;
                __builtin_memcpy(&r, &rt, 4);
                output[base + i] = kernel::_gang_float_to_bf16(w + r);
            }
        }
    }
    __syncthreads();
}


// ============================================================================
// Section I: MoE Per-Expert Barrier (W13 → W2 handoff)
// ============================================================================
// Hierarchical per-expert barrier using monotonic counters.
// Barrier layout: d_barrier[expert_id * 16 + 0..7] = per-XCD release flags
//                 d_barrier[expert_id * 16 + 8]     = global arrival counter
//
// W13 signal: each W13 tile atomicAdds global_arrive. When count reaches
//   w13_tiles, last arrival writes per-XCD release = expected_val.
// W2 wait: polls its own XCD's release flag via ld_nt until >= expected_val.
//
// expected_val is monotonically increasing (layer_idx + 1), so no reset needed.

// W13 signal: call after W13 tile completes (tid==0 only)
__device__ __forceinline__ void
moe_barrier_w13_signal(int *d_barrier, int expert_id, int w13_tiles,
                        int expected_val) {
    constexpr int HIER_STRIDE = 16;
    int base = expert_id * HIER_STRIDE;

    // Global arrival (all W13 tiles for this expert increment one counter)
    int prev_global = atom_add_release_gpu_s32(&d_barrier[base + 8], 1);
    if ((prev_global % w13_tiles) == w13_tiles - 1) {
        // Last W13 arrival: write per-XCD release flags
        for (int x = 0; x < 8; x++)
            st_wt_u32((void *)&d_barrier[base + x], (unsigned)expected_val);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }
}

// W2 wait: poll until this expert's W13 phase is complete on this XCD
__device__ __forceinline__ void
moe_barrier_w2_wait(int *d_barrier, int expert_id, int expected_val, int xcd_id) {
    constexpr int HIER_STRIDE = 16;
    int base = expert_id * HIER_STRIDE;
    while (__atomic_load_n(&d_barrier[base + xcd_id], __ATOMIC_RELAXED) < expected_val) {
        __builtin_amdgcn_s_sleep(1);
    }
}


// ============================================================================
// Section J: Counter buffer layout constants
// ============================================================================
// Each layer gets COUNTERS_PER_LAYER int32s in the counter buffer.
// Slot assignments for dense model (no MoE):

static constexpr int COUNTERS_PER_LAYER = 103 * 16;  // 84 base + 10 fused oproj + 9 routing ready

// Dense layer barrier slots (global counters)
static constexpr int SLOT_QKV_DONE        = 0 * 16;
static constexpr int SLOT_ATTN_DONE       = 1 * 16;
static constexpr int SLOT_OPROJ_DONE      = 2 * 16;
static constexpr int SLOT_FFN_MID_DONE    = 3 * 16;
static constexpr int SLOT_DOWN_DONE       = 4 * 16;
static constexpr int SLOT_LAYER_DONE      = 5 * 16;   // end-of-layer barrier

// Tail phase slots (in last layer's counter block)
static constexpr int SLOT_TAIL_LMHEAD     = 6 * 16;
static constexpr int SLOT_TAIL_ARGMAX     = 7 * 16;

// Per-XCD local arrival counters for hierarchical barriers (8 XCDs, stride 16)
// Used by barrier_global(6-arg): all workers on XCD arrive locally,
// last one does buffer_wbl2 + global arrive.
static constexpr int SLOT_OPROJ_LOCAL     = 8 * 16;   // local arrive for OPROJ barrier
static constexpr int SLOT_FFN_LOCAL       = 16 * 16;  // local arrive for FFN barrier
static constexpr int SLOT_DOWN_LOCAL      = 24 * 16;  // local arrive for DOWN barrier
static constexpr int SLOT_LAYER_LOCAL     = 32 * 16;  // local arrive for LAYER barrier

// MoE layer barrier slots — MUST NOT alias dense slots used in same layer.
// Attention block reuses QKV/OPROJ/FFN_MID slots (shared with dense).
// MoE FFN needs its OWN slots to avoid collisions with FFN_MID_DONE and DOWN_DONE.
static constexpr int SLOT_MOE_ROUTER_DONE = 40 * 16;  // after router GEMM (flat barrier)
static constexpr int SLOT_MOE_TOPK_DONE   = 41 * 16;  // after TopK (flat barrier)
static constexpr int SLOT_MOE_W13_DONE    = 42 * 16;  // after W13+SwiGLU (flat barrier)
static constexpr int SLOT_MOE_W2_DONE     = 43 * 16;  // after W2+atomicAdd (flat barrier)

// MoE local arrival slots for hierarchical barriers (8 XCDs, stride 16)
static constexpr int SLOT_MOE_W13_LOCAL    = 44 * 16;  // local arrive for W13→W2 barrier
static constexpr int SLOT_MOE_W2_LOCAL     = 52 * 16;  // local arrive for W2→end barrier

// Per-XCD attention chunk barrier (8 slots, one per XCD)
static constexpr int SLOT_ATTN_CHUNK       = 60 * 16;  // [60..67] per-XCD chunk counters

// Router GEMV local arrival slots for hierarchical barrier
static constexpr int SLOT_MOE_ROUTER_LOCAL = 68 * 16;  // [68..75] local arrive for Router GEMV barrier

// Attention local arrival slots for hierarchical barrier
static constexpr int SLOT_ATTN_LOCAL       = 76 * 16;  // [76..83] local arrive for attention barrier

// Fused OProj+RMSNorm+Router+TopK (Mechanism C) counter slots
// oproj_counters: 10 cache lines — [0..7] per-XCD release, [8] global arrive, [9] topk
static constexpr int SLOT_FUSED_OPROJ      = 84 * 16;  // [84..93] oproj Mechanism C counters
// routing_ready: 9 cache lines — [0] global epoch, [1..8] per-XCD release flags
static constexpr int SLOT_ROUTING_READY    = 94 * 16;  // [94..102] routing ready flags

#ifdef TITAN_ENABLE_LEGACY
// ============================================================================
// Section K: Fused RMSNorm + Router GEMV + TopK (single-worker)
// ============================================================================
// Single worker (xcd_id==0, xcd_rank==0) does:
//   1. RMSNorm on OProj output (writes norm_scratch2)
//   2. Router bf16 GEMV: 256 threads compute all 128 expert logits
//      from the register-cached norm output (no HBM re-read)
//   3. TopK selection + softmax
//
// All other workers do just RMSNorm (same as before), then skip to TopK barrier.
//
// This eliminates:
//   - FFN mid barrier (2.9µs) — RMSNorm runs right after OProj barrier
//   - Separate Router MXFP4 GEMM (27.9µs → ~5µs via bf16 GEMV from registers)
//   - Saves ~25µs/layer × 36 = ~0.9ms total

template <int HIDDEN_SIZE, int ACTUAL_HIDDEN_DIM, int NUM_EXPERTS, int NUM_TOPK>
__device__ __noinline__ void
rmsnorm_router_topk(
    void const *oproj_output,           // [HIDDEN_SIZE] bf16 (OProj+ResAdd output)
    void const *norm_weight,            // [HIDDEN_SIZE] bf16
    void *norm_output,                  // [HIDDEN_SIZE] bf16 (scratch)
    const unsigned short *router_w_bf16, // [NUM_EXPERTS, HIDDEN_SIZE] bf16 row-major
    const unsigned short *router_bias,   // [NUM_EXPERTS] bf16
    int *routing_indices,               // [NUM_EXPERTS, batch] int32
    float *topk_weight,                 // [batch, NUM_TOPK] f32
    int *active_expert_ids,             // [NUM_EXPERTS + 1] int32
    int tok_idx,
    int num_active_tokens)
{
    using bf16_t = __hip_bfloat16;
    int tid = threadIdx.x;

    const unsigned short *row_in = (const unsigned short *)oproj_output;
    const unsigned short *w_norm = (const unsigned short *)norm_weight;
    unsigned short *row_out = (unsigned short *)norm_output;

    // -- Pass 1: Compute SSQ for RMSNorm, cache hidden values in registers --
    constexpr int VEC = 4;
    constexpr int BLOCK_VEC = 256 * VEC;
    constexpr int MAX_ITERS = (HIDDEN_SIZE + BLOCK_VEC - 1) / BLOCK_VEC;
    float h_cache[MAX_ITERS * VEC];
    int n_cached = 0;
    float ssq = 0.0f;

    #pragma unroll
    for (int off = tid * VEC; off < HIDDEN_SIZE; off += BLOCK_VEC) {
        uint2 packed;
        __builtin_memcpy(&packed, &row_in[off], 8);

        unsigned r0 = (packed.x & 0xFFFFu) << 16;
        unsigned r1 = packed.x & 0xFFFF0000u;
        unsigned r2 = (packed.y & 0xFFFFu) << 16;
        unsigned r3 = packed.y & 0xFFFF0000u;
        float v0, v1, v2, v3;
        __builtin_memcpy(&v0, &r0, 4);
        __builtin_memcpy(&v1, &r1, 4);
        __builtin_memcpy(&v2, &r2, 4);
        __builtin_memcpy(&v3, &r3, 4);

        h_cache[n_cached + 0] = v0;
        h_cache[n_cached + 1] = v1;
        h_cache[n_cached + 2] = v2;
        h_cache[n_cached + 3] = v3;
        n_cached += VEC;

        asm volatile("v_fmac_f32 %0, %1, %1" : "+v"(ssq) : "v"(v0));
        asm volatile("v_fmac_f32 %0, %1, %1" : "+v"(ssq) : "v"(v1));
        asm volatile("v_fmac_f32 %0, %1, %1" : "+v"(ssq) : "v"(v2));
        asm volatile("v_fmac_f32 %0, %1, %1" : "+v"(ssq) : "v"(v3));
    }

    // Wave64 reduction
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        ssq += __shfl_xor(ssq, offset);
    }

    __shared__ float red[16];
    int wave_id = tid >> 6;
    int lane_id = tid & 63;
    int num_waves = 4;

    if (lane_id == 0) red[wave_id] = ssq;
    __syncthreads();

    if (wave_id == 0) {
        ssq = (lane_id < num_waves) ? red[lane_id] : 0.0f;
        for (int offset = num_waves >> 1; offset > 0; offset >>= 1) {
            ssq += __shfl_xor(ssq, offset);
        }
        if (lane_id == 0) red[0] = rsqrtf(ssq / (float)ACTUAL_HIDDEN_DIM + 1e-5f);
    }
    __syncthreads();
    float rms_rcp = red[0];

    // -- Pass 2: Apply norm + write output, cache norm'd values for Router GEMV --
    // Each thread caches its ELEMS_PER_THREAD norm'd values in registers
    constexpr int ELEMS_PER_THREAD = (HIDDEN_SIZE + 255) / 256;
    float norm_cache[ELEMS_PER_THREAD];

    {
        int ci = 0;
        int ni = 0;
        #pragma unroll
        for (int off = tid * VEC; off < HIDDEN_SIZE; off += BLOCK_VEC) {
            uint64_t wv;
            __builtin_memcpy(&wv, &w_norm[off], 8);
            const bf16_t *wa = (const bf16_t *)&wv;

            unsigned short ov[VEC];
            #pragma unroll
            for (int j = 0; j < VEC; j++) {
                float nv = h_cache[ci + j] * rms_rcp * __bfloat162float(wa[j]);
                ov[j] = kernel::_gang_float_to_bf16(nv);
                norm_cache[ni + j] = nv;
            }
            __builtin_memcpy(&row_out[off], ov, 8);
            ci += VEC;
            ni += VEC;
        }
    }

    __syncthreads();

    // -- Router GEMV: 256 threads compute all NUM_EXPERTS logits --
    // Each thread has ELEMS_PER_THREAD cached norm'd values.
    // Layout: thread t owns elements t*ELEMS_PER_THREAD .. (t+1)*ELEMS_PER_THREAD-1
    // Wait — the above layout is WRONG. The current norm_cache layout follows the
    // strided pattern: thread t owns elements at offsets tid*VEC, tid*VEC + BLOCK_VEC, ...
    // We need a contiguous mapping for the router weight reads.
    //
    // Actually, let's use the SAME approach as router_gemv_bf16_topk:
    // reload norm output from register cache with contiguous mapping.
    // Thread t loads elements [t * ELEMS_PER_THREAD .. (t+1) * ELEMS_PER_THREAD)

    // Rebuild norm_cache with contiguous layout for router GEMV
    float input_reg[ELEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int d = tid * ELEMS_PER_THREAD + i;
        if (d < HIDDEN_SIZE) {
            // Read from the norm output we just wrote (it's in L1 cache)
            unsigned bt = (unsigned)row_out[d] << 16;
            float v;
            __builtin_memcpy(&v, &bt, 4);
            input_reg[i] = v;
        } else {
            input_reg[i] = 0.0f;
        }
    }

    // Compute all expert logits
    const bf16_t *weight = (const bf16_t *)router_w_bf16;
    const bf16_t *bias = (const bf16_t *)router_bias;

    __shared__ float s_logits[NUM_EXPERTS];
    __shared__ float s_wave_partial[4 * NUM_EXPERTS];

    constexpr int BATCH = 8;
    static_assert(NUM_EXPERTS % BATCH == 0, "NUM_EXPERTS must be multiple of BATCH");

    for (int eb = 0; eb < NUM_EXPERTS; eb += BATCH) {
        float dot[BATCH] = {0.0f};

        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; i++) {
            int d = tid * ELEMS_PER_THREAD + i;
            if (d < HIDDEN_SIZE) {
                float in_val = input_reg[i];
                #pragma unroll
                for (int b = 0; b < BATCH; b++) {
                    dot[b] += in_val * static_cast<float>(
                        weight[static_cast<int64_t>(eb + b) * HIDDEN_SIZE + d]);
                }
            }
        }

        // Intra-wave reduction
        #pragma unroll
        for (int b = 0; b < BATCH; b++) {
            float val = dot[b];
            #pragma unroll
            for (int offset = 32; offset > 0; offset >>= 1) {
                val += __shfl_xor(val, offset, 64);
            }
            if (lane_id == 0) {
                s_wave_partial[wave_id * NUM_EXPERTS + eb + b] = val;
            }
        }
    }

    __syncthreads();

    // Cross-wave reduction + bias
    if (tid < NUM_EXPERTS) {
        int e = tid;
        float total = s_wave_partial[0 * NUM_EXPERTS + e]
                    + s_wave_partial[1 * NUM_EXPERTS + e]
                    + s_wave_partial[2 * NUM_EXPERTS + e]
                    + s_wave_partial[3 * NUM_EXPERTS + e];
        total += static_cast<float>(bias[e]);
        s_logits[e] = total;
    }

    __syncthreads();

    // -- TopK from shared memory logits --
    // Use existing moe_topk_softmax but read from s_logits
    {
        constexpr int VPT = (NUM_EXPERTS + 255) / 256;
        float vals[VPT];
        #pragma unroll
        for (int i = 0; i < VPT; i++) {
            int idx = tid + i * 256;
            vals[i] = (idx < NUM_EXPERTS) ? s_logits[idx] : -1e30f;
        }

        // Softmax: find max
        float thread_max = -1e30f;
        #pragma unroll
        for (int i = 0; i < VPT; i++) {
            thread_max = fmaxf(thread_max, vals[i]);
        }
        #pragma unroll
        for (int offset = 32; offset > 0; offset >>= 1) {
            thread_max = fmaxf(thread_max, __shfl_xor(thread_max, offset, 64));
        }

        extern __shared__ char _topk_smem[];
        float *s_reduce = (float *)_topk_smem;
        if (lane_id == 0) s_reduce[wave_id] = thread_max;
        __syncthreads();
        if (tid == 0) {
            float m = s_reduce[0];
            for (int w = 1; w < 4; w++) m = fmaxf(m, s_reduce[w]);
            s_reduce[0] = m;
        }
        __syncthreads();
        float global_max = s_reduce[0];

        // exp(x - max) and sum
        float thread_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < VPT; i++) {
            vals[i] = __expf(vals[i] - global_max);
            int idx = tid + i * 256;
            if (idx < NUM_EXPERTS) thread_sum += vals[i];
        }
        #pragma unroll
        for (int offset = 32; offset > 0; offset >>= 1) {
            thread_sum += __shfl_xor(thread_sum, offset, 64);
        }
        if (lane_id == 0) s_reduce[wave_id] = thread_sum;
        __syncthreads();
        if (tid == 0) {
            float total = 0.0f;
            for (int w = 0; w < 4; w++) total += s_reduce[w];
            s_reduce[0] = 1.0f / total;
        }
        __syncthreads();
        float inv_sum = s_reduce[0];

        #pragma unroll
        for (int i = 0; i < VPT; i++) {
            vals[i] *= inv_sum;
        }

        // Zero routing_indices
        for (int e = tid; e < NUM_EXPERTS; e += 256) {
            routing_indices[e * num_active_tokens + tok_idx] = 0;
        }
        __syncthreads();

        // Iterative top-k
        int *s_winner_idx = (int *)&s_reduce[4];
        float *s_winner_val = (float *)&s_reduce[8];
        int mask_count = 0;

        for (int k = 0; k < NUM_TOPK; k++) {
            float local_max = -1.0f;
            int local_idx = -1;
            #pragma unroll
            for (int i = 0; i < VPT; i++) {
                int idx = tid + i * 256;
                if (idx < NUM_EXPERTS && vals[i] > local_max) {
                    local_max = vals[i];
                    local_idx = idx;
                }
            }

            #pragma unroll
            for (int offset = 32; offset > 0; offset >>= 1) {
                float other_val = __shfl_xor(local_max, offset, 64);
                int other_idx = __shfl_xor(local_idx, offset, 64);
                if (other_val > local_max || (other_val == local_max && other_idx < local_idx)) {
                    local_max = other_val;
                    local_idx = other_idx;
                }
            }

            if (lane_id == 0) {
                s_reduce[wave_id] = local_max;
                s_winner_idx[wave_id] = local_idx;
            }
            __syncthreads();

            if (tid == 0) {
                float best_val = s_reduce[0];
                int best_idx = s_winner_idx[0];
                for (int w = 1; w < 4; w++) {
                    if (s_reduce[w] > best_val ||
                        (s_reduce[w] == best_val && s_winner_idx[w] < best_idx)) {
                        best_val = s_reduce[w];
                        best_idx = s_winner_idx[w];
                    }
                }
                s_winner_val[0] = best_val;
                s_winner_idx[0] = best_idx;
            }
            __syncthreads();

            int winner = s_winner_idx[0];

            if (tid == 0) {
                topk_weight[tok_idx * NUM_TOPK + k] = s_winner_val[0];
                routing_indices[winner * num_active_tokens + tok_idx] = k + 1;
                active_expert_ids[mask_count] = winner;
                mask_count++;
            }

            #pragma unroll
            for (int i = 0; i < VPT; i++) {
                if (tid + i * 256 == winner) vals[i] = -1.0f;
            }
            __syncthreads();
        }

        // Renormalize
        if (tid == 0) {
            float sum = 0.0f;
            for (int k = 0; k < NUM_TOPK; k++) {
                sum += topk_weight[tok_idx * NUM_TOPK + k];
            }
            float inv = 1.0f / sum;
            for (int k = 0; k < NUM_TOPK; k++) {
                topk_weight[tok_idx * NUM_TOPK + k] *= inv;
            }
            active_expert_ids[NUM_EXPERTS] = mask_count;
        }
        __syncthreads();
    }
}


// (LEGACY) Redundant bf16 Router GEMV + TopK — kept for reference
template <int NUM_EXPERTS, int HIDDEN_SIZE, int NUM_TOPK>
__device__ __forceinline__ void router_gemv_bf16_topk(
    const void *input_ptr,          // [HIDDEN_SIZE] bf16
    const void *weight_ptr,         // [NUM_EXPERTS, HIDDEN_SIZE] bf16
    const void *bias_ptr,           // [NUM_EXPERTS] bf16
    int *routing_indices,           // [NUM_EXPERTS * num_tokens] int32
    float *topk_weight,             // [num_tokens * NUM_TOPK] f32
    int *active_expert_ids,         // [NUM_EXPERTS + 1] int32
    int tok_idx,
    int num_active_tokens)
{
    const int tid = threadIdx.x;
    const __hip_bfloat16 *input = reinterpret_cast<const __hip_bfloat16 *>(input_ptr);
    const __hip_bfloat16 *weight = reinterpret_cast<const __hip_bfloat16 *>(weight_ptr);
    const __hip_bfloat16 *bias = reinterpret_cast<const __hip_bfloat16 *>(bias_ptr);

    // Each thread loads its portion of the input once
    constexpr int ELEMS_PER_THREAD = (HIDDEN_SIZE + 255) / 256;  // ceil(2944/256) = 12
    float input_reg[ELEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int d = tid * ELEMS_PER_THREAD + i;
        input_reg[i] = (d < HIDDEN_SIZE) ? static_cast<float>(input[d]) : 0.0f;
    }

    // Each thread computes partial dot products for batches of experts,
    // reduces within wave (shfl_xor), stores to shared memory.
    // Only 1 __syncthreads() needed for cross-wave reduction.
    __shared__ float s_logits[NUM_EXPERTS];
    __shared__ float s_wave_partial[4 * NUM_EXPERTS];  // 4 waves × 128 experts

    const int wave_id = tid / 64;
    const int lane = tid % 64;

    // Process experts in batches of BATCH to limit register pressure
    constexpr int BATCH = 8;
    static_assert(NUM_EXPERTS % BATCH == 0, "NUM_EXPERTS must be multiple of BATCH");

    for (int eb = 0; eb < NUM_EXPERTS; eb += BATCH) {
        float dot[BATCH] = {0.0f};

        // Accumulate dot products for this batch
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; i++) {
            int d = tid * ELEMS_PER_THREAD + i;
            if (d < HIDDEN_SIZE) {
                float in_val = input_reg[i];
                #pragma unroll
                for (int b = 0; b < BATCH; b++) {
                    dot[b] += in_val * static_cast<float>(
                        weight[static_cast<int64_t>(eb + b) * HIDDEN_SIZE + d]);
                }
            }
        }

        // Intra-wave reduction for this batch (no sync needed)
        #pragma unroll
        for (int b = 0; b < BATCH; b++) {
            float val = dot[b];
            #pragma unroll
            for (int offset = 32; offset > 0; offset >>= 1) {
                val += __shfl_xor(val, offset, 64);
            }
            if (lane == 0) {
                s_wave_partial[wave_id * NUM_EXPERTS + eb + b] = val;
            }
        }
    }

    __syncthreads();

    // Cross-wave reduction + bias (threads 0..127 each handle one expert)
    if (tid < NUM_EXPERTS) {
        int e = tid;
        float total = s_wave_partial[0 * NUM_EXPERTS + e]
                    + s_wave_partial[1 * NUM_EXPERTS + e]
                    + s_wave_partial[2 * NUM_EXPERTS + e]
                    + s_wave_partial[3 * NUM_EXPERTS + e];
        total += static_cast<float>(bias[e]);
        s_logits[e] = total;
    }

    __syncthreads();

    // TopK selection (tid==0 does it serially — only 128 experts, fast)
    if (tid == 0) {
        // Initialize routing indices to 0
        for (int e = 0; e < NUM_EXPERTS; e++) {
            routing_indices[e * num_active_tokens + tok_idx] = 0;
        }

        // Find max for softmax stability
        float max_logit = -1e30f;
        for (int e = 0; e < NUM_EXPERTS; e++) {
            if (s_logits[e] > max_logit) max_logit = s_logits[e];
        }

        // Find top-K by repeated argmax
        int top_experts[NUM_TOPK];
        float top_scores[NUM_TOPK];
        for (int k = 0; k < NUM_TOPK; k++) {
            float best_val = -1e30f;
            int best_idx = 0;
            for (int e = 0; e < NUM_EXPERTS; e++) {
                if (s_logits[e] > best_val) {
                    best_val = s_logits[e];
                    best_idx = e;
                }
            }
            top_experts[k] = best_idx;
            top_scores[k] = best_val;
            s_logits[best_idx] = -1e30f;  // mask out
        }

        // Compute softmax weights for top-K
        float sum_exp = 0.0f;
        for (int k = 0; k < NUM_TOPK; k++) {
            top_scores[k] = expf(top_scores[k] - max_logit);
            sum_exp += top_scores[k];
        }
        float inv_sum = 1.0f / sum_exp;

        // Write routing results
        for (int k = 0; k < NUM_TOPK; k++) {
            routing_indices[top_experts[k] * num_active_tokens + tok_idx] = k + 1;
            topk_weight[tok_idx * NUM_TOPK + k] = top_scores[k] * inv_sum;
        }

        // Write active_expert_ids
        int count = 0;
        for (int e = 0; e < NUM_EXPERTS; e++) {
            if (routing_indices[e * num_active_tokens + tok_idx] != 0) {
                active_expert_ids[count++] = e;
            }
        }
        active_expert_ids[NUM_EXPERTS] = count;
    }
    __syncthreads();
}

// ============================================================================
// Section L: Distributed Router GEMV (1 expert per worker)
// ============================================================================
// Each worker computes exactly 1 expert's logit via bf16 dot product.
// 128 workers each handle 1 expert → all 128 logits computed in parallel.
// The norm output (input) is read from HBM (same address, L2 cached).
// The weight row for each expert is read from HBM (unique per worker).
// Output: 1 bf16 logit written per worker.
//
// Uses vectorized uint2 loads (4 bf16 at a time) with STRIDED thread layout
// to maximize cache line utilization across threads.

template <int HIDDEN_SIZE, int ACTUAL_HIDDEN_DIM>
__device__ __noinline__ void
router_gemv_one_expert(
    const unsigned short *norm_output,      // [HIDDEN_SIZE] bf16
    const unsigned short *router_weight,    // [NUM_EXPERTS, HIDDEN_SIZE] bf16 row-major
    const unsigned short *router_bias,      // [NUM_EXPERTS] bf16
    unsigned short *logit_output,           // [NUM_EXPERTS] bf16 (each worker writes 1)
    int expert_id)
{
    int tid = threadIdx.x;
    int lane_id = tid & 63;
    int wave_id = tid >> 6;

    // Weight row for this expert
    const unsigned short *w_row = router_weight
        + static_cast<int64_t>(expert_id) * HIDDEN_SIZE;

    // Strided vectorized dot product: thread t processes elements at
    // offsets t*4, t*4 + 1024, t*4 + 2048, ... (stride = 256*4 = 1024)
    // This ensures adjacent threads read adjacent cache lines.
    constexpr int VEC = 4;
    constexpr int STRIDE = 256 * VEC;  // 1024 elements per pass
    float dot = 0.0f;

    #pragma unroll
    for (int off = tid * VEC; off < ACTUAL_HIDDEN_DIM; off += STRIDE) {
        // Vectorized load: 4 bf16 as uint2 from input
        uint2 in_packed;
        __builtin_memcpy(&in_packed, &norm_output[off], 8);

        // Vectorized load: 4 bf16 as uint2 from weight
        uint2 w_packed;
        __builtin_memcpy(&w_packed, &w_row[off], 8);

        // Unpack and multiply-add
        unsigned in0 = (in_packed.x & 0xFFFFu) << 16;
        unsigned in1 = in_packed.x & 0xFFFF0000u;
        unsigned in2 = (in_packed.y & 0xFFFFu) << 16;
        unsigned in3 = in_packed.y & 0xFFFF0000u;

        unsigned w0 = (w_packed.x & 0xFFFFu) << 16;
        unsigned w1 = w_packed.x & 0xFFFF0000u;
        unsigned w2 = (w_packed.y & 0xFFFFu) << 16;
        unsigned w3 = w_packed.y & 0xFFFF0000u;

        float iv0, iv1, iv2, iv3, wv0, wv1, wv2, wv3;
        __builtin_memcpy(&iv0, &in0, 4);
        __builtin_memcpy(&iv1, &in1, 4);
        __builtin_memcpy(&iv2, &in2, 4);
        __builtin_memcpy(&iv3, &in3, 4);
        __builtin_memcpy(&wv0, &w0, 4);
        __builtin_memcpy(&wv1, &w1, 4);
        __builtin_memcpy(&wv2, &w2, 4);
        __builtin_memcpy(&wv3, &w3, 4);

        asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(dot) : "v"(iv0), "v"(wv0));
        asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(dot) : "v"(iv1), "v"(wv1));
        asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(dot) : "v"(iv2), "v"(wv2));
        asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(dot) : "v"(iv3), "v"(wv3));
    }

    // Wave64 reduction
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        dot += __shfl_xor(dot, offset, 64);
    }

    // Cross-wave reduction via shared memory
    __shared__ float s_reduce[4];
    if (lane_id == 0) s_reduce[wave_id] = dot;
    __syncthreads();

    if (tid == 0) {
        float total = s_reduce[0] + s_reduce[1] + s_reduce[2] + s_reduce[3];
        // Add bias
        unsigned b_bits = (unsigned)router_bias[expert_id] << 16;
        float bias_val;
        __builtin_memcpy(&bias_val, &b_bits, 4);
        total += bias_val;
        // Write as bf16
        logit_output[expert_id] = kernel::_gang_float_to_bf16(total);
    }
    __syncthreads();
}

#endif // TITAN_ENABLE_LEGACY (rmsnorm_router_topk + router_gemv_one_expert)

} // namespace titan
