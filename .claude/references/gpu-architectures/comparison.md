# Cross-Architecture Comparison (CDNA3 / CDNA4 / RDNA3 / RDNA4)

This file collects every cross-architecture comparison table. Per-architecture deep-dives live in [`cdna3/hardware-overview.md`](cdna3/hardware-overview.md), [`cdna4/hardware-overview.md`](cdna4/hardware-overview.md), [`rdna3/hardware-overview.md`](rdna3/hardware-overview.md), [`rdna4/hardware-overview.md`](rdna4/hardware-overview.md). Use this file when:

- Evaluating "what changes if we move from X to Y" (e.g. CDNA3 -> CDNA4, RDNA3 -> RDNA4, CDNA -> RDNA)
- Picking a target arch for a new kernel
- Computing ridge points across SKUs
- Sizing tiles / occupancy / LDS budgets that have to work on multiple targets

---

## CDNA3 vs CDNA4

| Feature | CDNA3 (MI300) | CDNA4 (MI350) |
|---|---|---|
| Wavefront size | 64 | 64 |
| VGPR per wavefront | 256 | 256 |
| AGPR per wavefront | 256 | 256 |
| SGPR per wavefront | up to 102 | up to 102 |
| LDS per CU | 64 KB | **160 KB** |
| LDS banks | 32 | 64 |
| LDS allocation granularity | 512 bytes | 1280 bytes |
| LDS atomic units | 32 | 32 |
| Cache line size (L1/L2/scalar) | 64 bytes | 64 bytes |
| VMCNT width (max in-flight VMEM) | 6 bits (63) | 6 bits (63) |
| LGKMCNT width (max in-flight LDS/SMEM) | 4 bits (15) | 4 bits (15) |
| MFMA+VALU overlap | Yes (DLop forwarding) | Yes (XDLop forwarding) |
| SrcC 0-wait forwarding | Same DLop opcode | Same XDLop opcode |
| Lowest-precision MFMA | FP8 (8-bit) | **FP4 (4-bit)** |
| Block scaling MFMA | No | **Yes** (block size 32) |
| Max MFMA K-dim (F16/BF16) | 16 | **32** (BF16: 16x16x32) / **16** (F16: 32x32x16) |
| Max MFMA K-dim (I8) | 32 | **64** (16x16x64) |
| Max MFMA K-dim (F8) | 32 | **128** |
| F64 MFMA cycles (16x16x4) | 32 | **64** |
| MFMA sparse support | Yes | Yes |

Key differences:
- **LDS 2.5× larger** in CDNA4 (160 KB vs 64 KB) with twice the bank count
- **CDNA4 adds FP4 and FP6** data types for aggressive model compression
- **Block scaling (V_MFMA_SCALE_*)** is new in CDNA4 for extended dynamic range
- **K-dimension expanded across low-precision types** in CDNA4: BF16 K=32 (vs 16), F16 gains 32×32×16 shape (K still 16, but new single-block variant), I8 K=64 (vs 32), FP8/FP6/FP4 K=128 (vs 32)
- **F64 MFMA latency doubled** in CDNA4 (64 vs 32 cycles for 16x16x4)
- LDS allocation granularity differs: 512-byte blocks (CDNA3) vs 1280-byte blocks (CDNA4)

---

## RDNA3 vs RDNA4

| Feature | RDNA3 (RX 7900 XTX) | RDNA4 (RX 9070 XT / AI PRO R9700) |
|---|---|---|
| Default wave size | 32 | 32 |
| Wave64 support | Yes (via `-mwavefrontsize64`) | Yes for both VALU and WMMA (`IU4` wave64 uses lanes 0–31 only) |
| VGPR per wave | 256 | 256 |
| VGPR pool per SIMD32 | 1024 (gfx1102/1103) / **1536** (gfx1100/1101) | **1536** (all SKUs) |
| Allocation granularity (wave32) | 16 / 24 | 24 |
| Hardware thread slots per SIMD32 | 16 | 16 |
| AGPR file | — (none) | — (none) |
| LDS per WGP (physical) | 128 KB (2 × 64 KB halves, 32 banks each) | 128 KB (2 × 64 KB halves, 32 banks each) |
| Max LDS per workgroup | 64 KB | 64 KB |
| Banks visible to one wave32 | 32 | 32 |
| Cache line size | 128 bytes | 128 bytes |
| Wait counters | **Combined**: `vmcnt`, `vscnt`, `lgkmcnt` (6 b), `expcnt` | **Split**: `loadcnt`, `storecnt`, `dscnt`, `kmcnt`, `samplecnt`, `bvhcnt`, `expcnt` |
| Workgroup barrier | `s_barrier` (single opcode) | `s_barrier_signal -1; s_barrier_wait -1;` |
| Implicit VMEM->VGPR/LDS hazard interlock | Present | **Removed** -- explicit `s_wait_loadcnt` required (compiler-managed in HIP) |
| WMMA tile shapes | 16×16×16 only | 16×16×16, **16×16×32 (IU4)**, plus 16×16×32/64 SWMMAC |
| WMMA dtypes | FP16, BF16, IU8, IU4 | + **FP8 / BF8** (4 combinations) |
| WMMA cycles per wave-inst (SIMD32) | 32 | **16** (half of RDNA3, doubled per-CU throughput) |
| Sparse matrix (SWMMAC) | No | **Yes** (2:4 structured sparsity on A) |
| A/B half-wave replication required | **Yes** (lanes 0–15 == 16–31) | **No** |
| WMMA intrinsic naming | `__builtin_amdgcn_wmma_*_w32` (`fp16x16_t`) | `__builtin_amdgcn_wmma_*_w32_gfx12` (`fp16x8_t`) |
| Dynamic VGPR allocation (`S_ALLOC_VGPR`) | No | **Yes** |
| Buffer Resource (V#) word3 | gfx11 layout | **gfx12 layout** (e.g. `0x30004FAC` with `OOB_SELECT=3`, `FORMAT=4`) |
| Memory scope encoding | SC0/SC1 bits | **SCOPE[1:0]** (CU/SE/DEV/SYS) + `TH[2:0]` temporal hint |

Key differences:
- **GFX11 -> GFX12 wait counters split** -- `vmcnt` becomes `loadcnt`/`storecnt`, `lgkmcnt` becomes `dscnt`/`kmcnt`. Inline-asm pipelines must use the new names.
- **Implicit hazard interlocks removed in GFX12** -- the compiler inserts `s_wait_loadcnt 0` / `s_wait_dscnt 0` automatically in regular HIP code, but hand-written inline assembly must spell them out.
- **Split barrier** -- the old fused `s_barrier` is gone; use the signal/wait pair on RDNA4.
- **RDNA4 adds FP8 / BF8 WMMA** plus SWMMAC sparse variants (RDNA3 has neither).
- **RDNA3 A/B half-wave replication** doubles A/B register pressure on RDNA3; RDNA4 removes the requirement.
- **Buffer Resource word3 changed** -- the gfx9 / gfx10 / gfx11 value (`0x00020000`) silently disables stores under the gfx12 V# layout; RDNA4 needs `0x30004FAC` (or equivalent with `OOB_SELECT=3`).

---

## Unified four-way comparison

### Execution Model

| Feature | CDNA3 | CDNA4 | RDNA3 | RDNA4 |
|---|---|---|---|---|
| GFX generation | gfx942 | gfx950 | gfx1100–1103 (GFX11) | gfx1200–1201 (GFX12) |
| Wavefront size | 64 | 64 | **32** (default) / 64 | **32** (default) / 64 |
| EXEC mask width | 64-bit | 64-bit | 32 / 64-bit | 32 / 64-bit |
| Scheduling unit | CU | CU | **WGP** (2 CUs, 4 SIMD32s) | **WGP** (2 CUs, 4 SIMD32s) |
| SIMDs per scheduling unit | 4 (SIMD16) | 4 (SIMD16) | 4 (SIMD32) | 4 (SIMD32) |
| Max wavefronts / workgroup | 16 | 16 | 32 (w32) / 16 (w64) | 32 (w32) / 16 (w64) |
| Max work-items / workgroup | 1024 | 1024 | 1024 | 1024 |

### Registers

| Feature | CDNA3 | CDNA4 | RDNA3 | RDNA4 |
|---|---|---|---|---|
| VGPRs per wave | 256 | 256 | 256 | 256 |
| AGPR per wave | **256** | **256** | none (no AGPR file) | none (no AGPR file) |
| VGPR + AGPR combined | **512** | **512** | 256 (no AGPR) | 256 (no AGPR) |
| SGPRs per wave | up to 102 | up to 102 | up to 106 | up to 106 |
| VCC width | 64-bit (2 SGPRs) | 64-bit (2 SGPRs) | 64/32-bit | 64/32-bit |
| VGPR alloc block (wave32/64) | 8 Dwords | 8 Dwords | 16 or 24 (w32) / 8 or 12 (w64) | 24 (w32) / 12 (w64) |
| VGPR pool per SIMD | n/a (CU-level alloc: 16384 per CU) | n/a (CU-level alloc: 16384 per CU) | 1024 or **1536** per SIMD32 | **1536** per SIMD32 (all SKUs) |
| Dynamic VGPR alloc | No | No | No | **Yes** (`S_ALLOC_VGPR`) |
| Accumulator register | **AGPR** (separate file) | **AGPR** (separate file) | VGPR (unified) | VGPR (unified) |
| Inline-asm acc constraint | `"+a"` | `"+a"` | `"+v"` | `"+v"` |

### LDS (Local Data Share)

| Feature | CDNA3 | CDNA4 | RDNA3 | RDNA4 |
|---|---|---|---|---|
| LDS per scheduling unit | 64 KB / CU | **160 KB / CU** | 128 KB / WGP | 128 KB / WGP |
| Max LDS per workgroup | 64 KB | **160 KB** (no per-WG cap in ISA; 64 KB is HIP software limit) | 64 KB | 64 KB |
| Physical banks | 32 | **64** | 64 (2 × 32 halves) | 64 (2 × 32 halves) |
| Banks visible per wave | 32 | 64 | 32 | 32 |
| Bank width | 4 bytes | 4 bytes | 4 bytes | 4 bytes |
| Conflict-free stride | `N % 32 != 0` | `N % 64 != 0` | `N % 32 != 0` | `N % 32 != 0` |
| Alloc granularity | 512 bytes | 1280 bytes | 1024 bytes | 1024 bytes |
| LDS allocation modes | n/a (CU is the only unit) | n/a (CU is the only unit) | WGP / CU mode | WGP (default) / CU mode |

### Cache Hierarchy

| Feature | CDNA3 (MI300X) | CDNA4 (MI350X) | RDNA3 (RX 7900 XTX) | RDNA4 (RX 9070 XT) |
|---|---|---|---|---|
| Cache line size | **64 bytes** | **64 bytes** | **128 bytes** | **128 bytes** |
| L0/L1 vector per CU | 32 KB | 32 KB | 32 KB | 32 KB |
| Scalar L0 per CU | 16 KB | 16 KB | 16 KB | 16 KB |
| GL1 (mid-level) | none (L1 -> L2 -> LLC) | none (L1 -> L2 -> LLC) | 256 KB / SA | 256 KB / SA (write-combining) |
| L2 cache (total) | ~32 MB (8 × 4 MB) | ~32 MB (8 × 4 MB) | 6 MB | **8 MB** |
| LLC (Infinity Cache) | **256 MB** | **256 MB** | 96 MB | **64 MB** |
| Memory type | HBM3 | HBM3e | GDDR6 384-bit | GDDR6 256-bit |
| Memory capacity | 192 GB (MI300X) / 256 GB (MI325X) | 288 GB | 24 GB | 16 GB (32 GB ECC on R9700) |
| Memory bandwidth | ~5.3 TB/s (MI300X) / ~6.0 TB/s (MI325X) | ~8.0 TB/s | ~960 GB/s | ~640 GB/s |

### Wait Counters & Synchronization

| Feature | CDNA3 | CDNA4 | RDNA3 (GFX11) | RDNA4 (GFX12) |
|---|---|---|---|---|
| Counter model | Combined | Combined | Combined | **Split** |
| VMEM load counter | `vmcnt` (6 b, max 63) | `vmcnt` (6 b, max 63) | `vmcnt` (6 b, max 63) -- loads, samples, atomics-w/return | **`loadcnt`** (6 b) |
| VMEM store counter | (included in `vmcnt`) | (included in `vmcnt`) | `vscnt` (6 b, max 63) | **`storecnt`** (6 b) |
| LDS / scalar counter | `lgkmcnt` (4 b, max 15) | `lgkmcnt` (4 b, max 15) | `lgkmcnt` (**6 b, max 63**) | **`dscnt`** (6 b) / **`kmcnt`** (5 b) |
| Export counter | `expcnt` (3 b) | `expcnt` (unused) | `expcnt` (3 b) | `expcnt` (3 b) -- exports + param loads |
| Sample counter | n/a | n/a | n/a | **`samplecnt`** (6 b) |
| BVH counter | n/a | n/a | n/a | **`bvhcnt`** (3 b) |
| Workgroup barrier | `s_barrier` | `s_barrier` | `s_barrier` | **`s_barrier_signal -1; s_barrier_wait -1;`** |
| Implicit VMEM->VGPR stall | Yes | Yes | Yes | **No** (explicit waits required) |

### Matrix Instruction Support

| Feature | CDNA3 | CDNA4 | RDNA3 | RDNA4 |
|---|---|---|---|---|
| Instruction family | **MFMA** | **MFMA** | **WMMA** | **WMMA** |
| Wave size for matrix | 64 | 64 | 32 (or 64) | 32 (or 64; IU4 wave64 uses lanes 0–31 only) |
| Tile shapes (M×N) | 4×4, 16×16, 32×32 | 4×4, 16×16, 32×32 | 16×16 only | 16×16 only |
| FP16 / BF16 input | Yes | Yes | Yes | Yes |
| FP8 / BF8 input | Yes | Yes | No | **Yes** |
| FP4 / FP6 input | No | **Yes** | No | No |
| INT8 input | Yes | Yes | Yes | Yes |
| INT4 input | No | No | Yes | Yes |
| F32 input (MFMA only) | Yes | Yes | no (WMMA is sub-32-bit only) | no (WMMA is sub-32-bit only) |
| F64 input (MFMA only) | Yes | Yes (2× latency) | no (no RDNA matrix F64) | no (no RDNA matrix F64) |
| XF32 (TF32) input | Yes | **No** (removed in CDNA4) | no (MFMA-only dtype) | no (MFMA-only dtype) |
| Block scaling | No | **Yes** (V_MFMA_SCALE) | No | No |
| 2:4 structured sparsity | Yes (SMFMAC) | Yes (SMFMAC) | No | **Yes** (SWMMAC) |
| D-matrix layout (wave32) | n/a (MFMA) | n/a (MFMA) | **Interleaved rows** (even rows lanes 0–15, odd rows 16–31) | **Contiguous row blocks** (rows 0–7 lanes 0–15, rows 8–15 lanes 16–31) |
| Accumulator location | AGPR | AGPR | VGPR | VGPR |
| EXEC mask ignored | Yes | Yes | Yes | Yes |

### Matrix Throughput (ops/cycle/CU)

All values count each multiply and each add as a separate operation.

| Dtype | CDNA3 | CDNA4 | RDNA3 | RDNA4 |
|---|---:|---:|---:|---:|
| FP16 / BF16 matrix | 2048 | 4096 | 512 | 1024 |
| FP8 / BF8 matrix | 4096 | 8192 | unsupported | 2048 |
| INT8 matrix | 4096 | 8192 | 512 | 2048 |
| INT4 matrix | no MFMA INT4 | no MFMA INT4 | 1024 | 4096 |
| FP4 matrix | unsupported | 8192 | unsupported | unsupported |

*CDNA3 FP16 per-CU throughput derived from MI300X published specs: 1307 TF / 304 CUs / ~2100 MHz ≈ 2048 FLOPs/cycle/CU. FP8 and INT8 have 2× K-dimension at the same cycle count, giving 4096 ops/cycle/CU (matching AMD's published ~2615 TOPS INT8/FP8 for MI300X). CDNA4 doubles all rates via further K-dimension expansion. RDNA values from the per-architecture tables above.*

### VALU Throughput (ops/cycle/CU)

| Dtype | CDNA3 | CDNA4 | RDNA3 | RDNA4 |
|---|---:|---:|---:|---:|
| FP32 ADD | 64 | 64 | 64 | 64 |
| FP32 FMA | 128 | 128 | 128 | 128 |
| FP16/BF16 packed FMA | 256 | 256 | 256 | 256 |
| FP64 ADD | 64 (full rate) | 64 (full rate) | 2 (1/32 rate) | 2 (1/32 rate) |
| INT32 | 64 | 64 | 64 | 64 |
| INT8 packed (dot4) | 512 | 512 | 512 | 512 |
| FP32 dual-issue | not supported | not supported | up to 256 | up to 256 |

*Each multiply + add counted as 2 ops. CDNA lanes: 64 per CU (4 SIMDs × 16). RDNA lanes: 64 per CU (2 SIMD32s × 32). RDNA dual-issue pairs two compatible VOP1/VOP2 instructions per slot. RDNA FP64 is 1/32 rate -- present for correctness, not for performance.*

### Occupancy

| Feature | CDNA3 | CDNA4 | RDNA3 (1536-pool) | RDNA4 |
|---|---|---|---|---|
| Max wavefronts / sched. unit | 32 / CU | 32 / CU | 64 / WGP (16/SIMD32) | 64 / WGP (16/SIMD32) |
| Max workgroups / sched. unit | 8 / CU | 8 / CU | 32 / WGP | 32 / WGP |
| VGPR budget for full occupancy | ≤ 16 VGPRs | ≤ 16 VGPRs | ≤ 96 VGPRs/wave | ≤ 96 VGPRs/wave |
| LDS budget for max WGs | ≤ 8 KB / WG | ≤ 20 KB / WG | ≤ 4 KB / WG | ≤ 4 KB / WG |

### Reference SKU Summary

<!-- AUTOGEN:gpu-table id=comparison_reference_skus -->
|  | CDNA3 | CDNA4 | RDNA3 | RDNA4 |
|---|---:|---:|---:|---:|
| Reference SKU | MI300X | MI350X | RX 7900 XTX | RX 9070 XT |
| CUs | 304 | 256 | 96 (48 WGPs) | 64 (32 WGPs) |
| Boost clock | 2100 MHz | 2200 MHz | 2500 MHz | 2970 MHz |
| FP16 matrix peak | 1307.4 TF | 2300 TF | 123 TF | 195 TF |
| INT8 matrix peak | 2614.9 TOPS | 4600 TOPS | 123 TOPS | 389 TOPS |
| FP8 matrix peak | 2614.9 TF | 4600 TF | unsupported | 389 TF |
| Memory bandwidth | ~5.3 TB/s HBM3 | ~8.0 TB/s HBM3e | ~960 GB/s GDDR6 | ~640 GB/s GDDR6 |
| FP16 ridge (mat peak / BW) | ~247 FLOP/B | ~288 FLOP/B | ~128 FLOP/B | ~305 FLOP/B |
<!-- /AUTOGEN -->

*Ridge = matrix peak / memory bandwidth -- higher means more compute-bound, requiring more data reuse to saturate. Numbers above are regenerated from `references/gpu_specs.yaml`.*

### Vector vs Matrix Compute Roof (FP16 path)

The matrix engine peaks at a multiple of the V_DOT2_F32_F16 / packed-FP16 vec roof at full M=16 packing. At M < 16 the effective matrix throughput drops by `M/16`; at M=1 (decode-time GEMV) the matrix engine is structurally below the vec roof.

<!-- AUTOGEN:gpu-table id=comparison_vec_vs_matrix -->
| SKU | Family | Vec FP16->FP32 peak | Matrix FP16 peak | Matrix / Vec |
|---|---|---:|---:|---:|
| MI300X | CDNA3 | ~163 TFLOPS | ~1307.4 TFLOPS | 8.0x |
| MI350X | CDNA4 | ~144 TFLOPS | ~2300 TFLOPS | 16.0x |
| RX 7900 XTX | RDNA3 | ~61 TFLOPS | ~123 TFLOPS | 2.0x |
| RX 7800 XT | RDNA3 | ~37 TFLOPS | ~74.6 TFLOPS | 2.0x |
| RX 7600 XT | RDNA3 | ~23 TFLOPS | ~45.1 TFLOPS | 2.0x |
| RX 9070 XT | RDNA4 | ~49 TFLOPS | ~195 TFLOPS | 4.0x |
| AI PRO R9700 | RDNA4 | ~48 TFLOPS | ~191 TFLOPS | 4.0x |
| RX 9060 XT | RDNA4 | ~26 TFLOPS | ~103 TFLOPS | 4.0x |
<!-- /AUTOGEN -->

See [`wmma_opcodes.md`](./wmma_opcodes.md) for the partial-M effective throughput derivation and the WMMA-vs-vec crossover in `tile_height`.
