# AMD CDNA4 (gfx950 / MI350 Series) Architecture Reference

*Source: CDNA4 Instruction Set Architecture Reference Guide (5-August-2025), and AMD product documentation (CU counts, HBM, clocks).*

For cross-architecture comparison tables (CDNA3 vs CDNA4 vs RDNA3 vs RDNA4) see [`comparison.md`](../comparison.md).

---

## Registers

| Register Type | Count | Width | Notes |
|---|---|---|---|
| VGPR (V0–V255) | 256 | 32-bit | Allocated in groups of 8 Dwords |
| AGPR (AV0–AV255) | 256 | 32-bit | Exclusive to matrix core unit |
| VGPR + AGPR combined | 512 max | 32-bit | Flexible split |
| SGPR (S0–S103) | 104 | 32-bit | Allocated 16–102 per wavefront, in units of 16 |
| VCC | 2 SGPRs | 64-bit | Physically SGPR 106–107 |
| TTMP0–TTMP15 | 16 | 32-bit | Trap temporary SGPRs (with trap handler) |

## Memory

| Resource | Size | Organization |
|---|---|---|
| LDS per CU | **160 KB** | 64 banks × 640 entries (4 bytes each) |
| LDS atomic units | 32 integer | Built-in fast unordered atomics |
| LDS allocation granularity | 1280-byte blocks | 1280-byte aligned |

## Execution

| Parameter | Value |
|---|---|
| Wavefront size | 64 work-items |
| EXEC mask | 64-bit |
| Max wavefronts per workgroup | 16 |
| Max work-items per workgroup | 1024 |

## CU Count (product-level, not in ISA manual)

<!-- AUTOGEN:gpu-table id=cdna4_cu_count -->
| Product | CUs | HBM |
|---|---:|---|
| AMD Instinct MI350X | 256 | 288 GB HBM3e |
| AMD Instinct MI355X | 256 | 288 GB HBM3e |
<!-- /AUTOGEN -->

*MI350X / MI355X both use 8 XCDs with 32 active CUs/XCD (physical 36, 4/XCD disabled for yield) = 256 active CUs total. Per AMD product pages and the MI350X / MI355X GPU brochures. MI355X differs only in clock (2.4 GHz vs 2.2 GHz) and TBP (1400W vs 1000W); compute peaks scale ~ +8% accordingly.*

## In-flight Instruction Counters (S_WAITCNT)

| Counter | Bits | Max in-flight | Tracks |
|---|---|---|---|
| VMCNT | 6 | 63 | Vector memory (global/buffer) loads and stores |
| LGKMCNT | 4 | 15 | LDS, scalar memory (SMEM), messages (no GDS in CDNA4) |
| EXPCNT | 3 | 7 | Unused on CDNA4 |

`S_WAITCNT` stalls the wavefront until all tracked counters reach **≤** the specified value.

## Cache Line Size

Scalar L0 data cache uses **64-byte cache lines** (per ISA manual). Vector/global memory cache line assumed 64 bytes.

## MFMA + VALU Concurrency

MFMA instructions are long-latency but the pipeline allows issuing VALU instructions while an MFMA is in-flight. Key rules:

- **XDLOPs** (CDNA4 term for matrix math on I8/F16/BF16) support back-to-back SrcC (accumulator) forwarding with **0 wait states** when the same opcode is chained.
- **Non-XDLOP VALU writing a VGPR** read by subsequent MFMA SrcA/B requires **2 wait cycles**.
- Reading an MFMA output VGPR with a VMEM/LDS instruction overlapped with the first MFMA output Dword requires **4 wait cycles**.

## MFMA Instructions

**Dense MFMA:**

| Instruction | Dimensions | Blocks | Cycles | FLOPs/wave-inst | Input Type |
|---|---|---|---|---:|---|
| V_MFMA_F32_32x32x1_2B | 32×32×1 | 2 | 64 | 4096 | F32 |
| V_MFMA_F32_16x16x1_4B | 16×16×1 | 4 | 32 | 2048 | F32 |
| V_MFMA_F32_4x4x1_16B | 4×4×1 | 16 | 8 | 512 | F32 |
| V_MFMA_F32_32x32x2 | 32×32×2 | 1 | 64 | 4096 | F32 |
| V_MFMA_F32_16x16x4 | 16×16×4 | 1 | 32 | 2048 | F32 |
| V_MFMA_F32_32x32x4_2B | 32×32×4 | 2 | 64 | 16384 | F16 |
| V_MFMA_F32_16x16x4_4B | 16×16×4 | 4 | 32 | 8192 | F16 |
| V_MFMA_F32_4x4x4_16B | 4×4×4 | 16 | 8 | 2048 | F16 |
| V_MFMA_F32_32x32x8 | 32×32×8 | 1 | 32 | 16384 | F16 |
| V_MFMA_F32_16x16x16 | 16×16×16 | 1 | 16 | 8192 | F16 |
| V_MFMA_F32_32x32x4_2B | 32×32×4 | 2 | 64 | 16384 | BF16 |
| V_MFMA_F32_16x16x4_4B | 16×16×4 | 4 | 32 | 8192 | BF16 |
| V_MFMA_F32_4x4x4_16B | 4×4×4 | 16 | 8 | 2048 | BF16 |
| V_MFMA_F32_32x32x8 | 32×32×8 | 1 | 32 | 16384 | BF16 |
| V_MFMA_F32_16x16x16 | 16×16×16 | 1 | 16 | 8192 | BF16 |
| V_MFMA_I32_32x32x4_2B | 32×32×4 | 2 | 64 | 16384 | I8 |
| V_MFMA_I32_16x16x4_4B | 16×16×4 | 4 | 32 | 8192 | I8 |
| V_MFMA_I32_4x4x4_16B | 4×4×4 | 16 | 8 | 2048 | I8 |
| V_MFMA_I32_32x32x16 | 32×32×16 | 1 | 32 | 32768 | I8 |
| V_MFMA_I32_16x16x32 | 16×16×32 | 1 | 16 | 16384 | I8 |
| V_MFMA_F64_16x16x4 | 16×16×4 | 1 | **64** | 2048 | F64 |
| V_MFMA_F64_4x4x4_4B | 4×4×4 | 4 | **32** | 512 | F64 |
| V_MFMA_F32_16x16x32 | 16×16×32 | 1 | 16 | 16384 | BF8/FP8 |
| V_MFMA_F32_32x32x16 | 32×32×16 | 1 | 32 | 32768 | BF8/FP8 |
| V_MFMA_F32_16x16x32_BF16 | 16×16×32 | 1 | 16 | 16384 | BF16 *(CDNA4-new)* |
| V_MFMA_F32_32x32x16_F16 | 32×32×16 | 1 | 32 | 32768 | F16 *(CDNA4-new)* |
| V_MFMA_I32_16x16x64_I8 | 16×16×64 | 1 | 16 | 32768 | I8 *(CDNA4-new)* |
| V_MFMA_I32_32x32x32_I8 | 32×32×32 | 1 | 32 | 65536 | I8 *(CDNA4-new)* |
| V_MFMA_F32_16x16x128_F8F6F4 | 16×16×128 | 1 | 16 (FP4/FP6) / 32 (FP8) | 65536 | FP4/FP6/FP8 mixed |
| V_MFMA_F32_32x32x64_F8F6F4 | 32×32×64 | 1 | 32 (FP4/FP6) / 64 (FP8) | 131072 | FP4/FP6/FP8 mixed |
| V_MFMA_SCALE_F32_16x16x128 | 16×16×128 | 1 | 16 (FP4/FP6) / 32 (FP8) | 65536 | F4/F6/F8 with block scaling |
| V_MFMA_SCALE_F32_32x32x64 | 32×32×64 | 1 | 32 (FP4/FP6) / 64 (FP8) | 131072 | F4/F6/F8 with block scaling |

Note: F8F6F4 instructions take the longer cycle count when any operand is FP8; the shorter count otherwise.

**Block scaling (V_MFMA_SCALE_*) details:**
- Scale factor encoding: 8-bit exponent, bias 127; valid range −127 to +127 (0xFF = NaN)
- Block size: **32 elements** per scale factor
- Instructions are encoded as 4-DWORD (128-bit) instructions; scale factors are loaded and applied inline with the MFMA

**Sparse MFMA (V_SMFMAC, 2:4 structured sparsity):**

*Inherited from CDNA3:*

| Instruction | Dimensions | Blocks | Cycles | FLOPs/wave-inst | Input Type |
|---|---|---|---|---:|---|
| V_SMFMAC_F32_16x16x32 | 16×16×32 | 1 | 16 | 16384 | F16 |
| V_SMFMAC_F32_32x32x16 | 32×32×16 | 1 | 32 | 32768 | F16 |
| V_SMFMAC_F32_16x16x32 | 16×16×32 | 1 | 16 | 16384 | BF16 |
| V_SMFMAC_F32_32x32x16 | 32×32×16 | 1 | 32 | 32768 | BF16 |
| V_SMFMAC_I32_16x16x64 | 16×16×64 | 1 | 16 | 32768 | I8 |
| V_SMFMAC_I32_32x32x32 | 32×32×32 | 1 | 32 | 65536 | I8 |
| V_SMFMAC_F32_16x16x64 | 16×16×64 | 1 | 16 | 32768 | BF8/FP8 |
| V_SMFMAC_F32_32x32x32 | 32×32×32 | 1 | 32 | 65536 | BF8/FP8 |

*New in CDNA4 (doubled K-dimensions):*

| Instruction | Dimensions | Blocks | Cycles | FLOPs/wave-inst | Input Type |
|---|---|---|---|---:|---|
| V_SMFMAC_F32_16x16x64 | 16×16×64 | 1 | 16 | 32768 | F16 |
| V_SMFMAC_F32_32x32x32 | 32×32×32 | 1 | 32 | 65536 | F16 |
| V_SMFMAC_F32_16x16x64 | 16×16×64 | 1 | 16 | 32768 | BF16 |
| V_SMFMAC_F32_32x32x32 | 32×32×32 | 1 | 32 | 65536 | BF16 |
| V_SMFMAC_I32_16x16x128 | 16×16×128 | 1 | 16 | 65536 | I8 |
| V_SMFMAC_I32_32x32x64 | 32×32×64 | 1 | 32 | 131072 | I8 |
| V_SMFMAC_F32_16x16x128 | 16×16×128 | 1 | 16 | 65536 | BF8/FP8 (4 combinations) |
| V_SMFMAC_F32_32x32x64 | 32×32×64 | 1 | 32 | 131072 | BF8/FP8 (4 combinations) |

**MFMA dependency hazards (minimum waits required):**

| Producing instruction | Consuming instruction | Min waits |
|---|---|---|
| Non-XDLOP VALU write VGPR | V_MFMA* read SrcA/B | 2 |
| Same XDLOP opcode chain (SrcC) | Next XDLOP read SrcC | **0** (forwarded; **2** for 2-pass 4x4x4 XDL) |
| XDLOP write SrcA/B | Next XDLOP read SrcA/B | 3 |
| **XDL waits by size:** | | |
| V_MFMA_*_4x4x4 (2-pass XDL) write | XDL/MFMA read SrcA/B | 5 |
| V_MFMA_*_16x16x16 (4-pass XDL) write | XDL/MFMA read SrcA/B | 8 |
| V_MFMA_*_32x32x8 (8-pass XDL) write | Any VGPR read (RAW+WAW) | 12 |
| V_MFMA_F32_32x32x4_2B (16-pass XDL) write | Memory/VALU read VGPR | 20 |
| **SGEMM (F32-input) waits by size:** | | |
| V_MFMA_F32_4x4x1_16B (2-pass SGEMM) write | SGEMM/XDL read SrcA/B | 4 |
| V_MFMA_F32_16x16x4 (4-pass SGEMM) write | SGEMM/XDL read SrcA/B | 6 |
| V_MFMA_F32_32x32x2 (8-pass SGEMM) write | Any VGPR read (RAW+WAW) | 10 |
| V_MFMA_F32_32x32x1_2B (16-pass SGEMM) write | Memory/VALU read VGPR | 18 |
| **DGEMM (F64) waits:** | | |
| V_MFMA_F64_16x16x4 write | MFMA read SrcA/B | 19 |
| V_MFMA_F64_16x16x4 write | VM/LDS/FLAT/Export read overlapped | 20 |
| V_CMPX write EXEC | V_MFMA | 4 |

**MFMA properties:**
- Matrix core primitive: 4×1 outer product yielding 16 outputs
- Ignores MODE denorm, rounding, and EXEC mask (all lanes always active) — **except** `V_MFMA_F32_*_F32` (F32 inputs), which honor MODE denorm-handling flags
- Clamp supported via FP16_OVFL bit
- Block scaling supported (V_MFMA_SCALE_*) for dynamic range extension
- Inputs A/B from VGPR only; accumulator can be inline constant

---

## Parameters not in the ISA manual (CDNA4 only)

The values below are critical for kernel optimization but are defined in product/microarchitecture specs, not the ISA manual. Cross-architecture comparison tables for these parameters are in [`comparison.md`](../comparison.md).

### Occupancy Limits (per CU, MI350X)

| Parameter | Value |
|---|---|
| Max wavefronts per CU | 32 |
| Max workgroups per CU | 8 |
| Max VGPRs allocated per CU | 512 × 32 = 16384 |

**VGPR count → max wavefronts per CU** (32-wave hardware cap, identical to CDNA3):

| VGPRs per wave | Max wavefronts/CU |
|---|---|
| 0–16 | 32 |
| 17–32 | 32 |
| 33–64 | 16 |
| 65–96 | 12 |
| 97–128 | 8 |
| 129–192 | 6 |
| 193–256 | 4 |
| 257–512 (with AGPRs) | 2 |

**LDS bytes per workgroup → max workgroups per CU** (CDNA4, **160 KB** LDS):

| LDS per workgroup | Max WGs/CU |
|---|---|
| 0 | 8 |
| ≤8 KB | 8 |
| ≤16 KB | 8 |
| ≤32 KB | 4 |
| ≤64 KB | 2 |
| ≤80 KB | 2 |
| ≤160 KB | 1 |

*Occupancy is the minimum of the wavefront limit (from VGPRs), workgroup limit (from LDS), and the hardware max. Use `rocminfo` or `hipOccupancyMaxActiveBlocksPerMultiprocessor` to compute this at runtime.*

### Cache Sizes

| Cache | MI350X / MI355X |
|---|---|
| L1 vector cache per CU | 32 KB |
| L2 cache per XCD | 4 MB |
| Total L2 (8 XCDs) | ~32 MB |
| Scalar L0 cache per CU | 16 KB |
| Infinity Cache (LLC, memory-side) | **256 MB** |
| HBM memory | 288 GB HBM3e |
| HBM bandwidth | ~8.0 TB/s |

*MI350X / MI355X carry the same 8-XCD topology as the MI300 series. Cross-XCD L2 access goes through Infinity Fabric.*

### Global Memory Latency (approximate)

| Level | Latency (cycles) |
|---|---|
| L1 vector cache hit | ~20–40 |
| L2 cache hit | ~100–200 |
| HBM (L2 miss) | ~400–700 |
| Cross-XCD | higher than local L2 |

*Covering HBM latency requires ~10–22 in-flight wavefronts per CU at typical memory-bound occupancy.*

### VALU Throughput (per CU per clock, CDNA4)

| Data type | Ops/cycle/CU |
|---|---|
| FP32 (V_ADD_F32) | 64 |
| FP32 FMA (V_FMA_F32) | 128 |
| FP16 / BF16 packed (V_PK_FMA_F16) | 256 |
| FP64 (V_ADD_F64) | 64 |
| INT32 | 64 |
| INT8 packed (V_DOT4_I32_I8) | 512 |

*Ops/cycle counts each multiply and each add as a separate operation. Vector throughput is unchanged from CDNA3; the doubling on CDNA4 is at the matrix engine (MFMA), not the vector engine.*

### Reference SKU peak summary

<!-- AUTOGEN:gpu-table id=cdna4_peak_summary -->
| Metric | MI350X | MI355X |
|---|---:|---:|
| FP16 matrix peak | 2300 TF | 2510 TF |
| BF16 matrix peak | 2300 TF | 2510 TF |
| FP8 matrix peak | 4600 TF | 5020 TF |
| INT8 matrix peak | 4600 TOPS | 5020 TOPS |
| MXFP4 matrix peak | 9200 TF | 10040 TF |
| Memory bandwidth | ~8.0 TB/s | ~8.0 TB/s |
| FP16 ridge (peak / BW) | ~288 FLOP/B | ~314 FLOP/B |
| FP8 ridge | ~575 FLOP/B | ~628 FLOP/B |
| MXFP4 ridge | ~1150 FLOP/B | ~1255 FLOP/B |
<!-- /AUTOGEN -->
