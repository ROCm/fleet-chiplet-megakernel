# AMD GPU Memory and synchronization reference (CDNA4)

## Description

This reference is **scoped to memory ordering, cache coherency, and synchronization** — cache flags, cache management, waits, fences, atomics, buffer descriptors, cross-lane ops, and timing — with worked HIP+asm examples. It deliberately does *not* cover compute instructions (scalar/vector ALU, transcendentals, conversions, matrix/MFMA) or exhaustive per-opcode encodings. For any of those, or for the authoritative spec definition of an instruction, use the full ISA spec at [`isa-manual/`](isa-manual/README.md)..

Assembly in this document was generated with Godbolt using `--offload-arch=gfx950 -S -O3` (MI355 / gfx950) unless noted otherwise.

---

## Key Cache Flags

AMD GCN/CDNA/RDNA instructions carry modifier bits that control cache behavior. These bits control the coherency scope (which cache levels are involved) and eviction policy (whether data is retained in L1 or bypasses directly to L2/memory).

| Flag | ISA name | Scope / Meaning |
|------|----------|-----------------|
| `sc0` | GLC (Global Level Coherency) | Group scope. Causes the CU/L1 cache to miss (data is not retained in L1 after use). L2 still caches the data with normal LRU policy. Used for workgroup-level coherency across CUs. |
| `sc1` | SLC (System Level Coherency) | Device / System scope. Combined with `sc0`: causes L2 to perform a coherent bypass (data goes directly to/from memory on multi-L2 chips). Alone (`sc1` only, no `sc0`): no L2 effect; CU cache is missed only in TG-Split workgroups. |
| `nt` | Non-Temporal (streaming hint) | Streaming hint. Changes CU cache from Hit→Miss (data not retained in L1 after use), L2 from LRU→Stream eviction policy, and LLC from LRU→Evict. Can be combined with any scope flags. |
| `dlc` | Data Last-Level Cache (RDNA 2+) | Bypasses the GL1 (shared shader-array-level L1 cache), going directly to GL2. Not available on CDNA (MI series). |

### Scope summary (CDNA)

| Scope | `sc1` | `sc0` | CU / L1 Cache | L2 Cache |
|-------|-------|-------|----------------|----------|
| Wave | 0 | 0 | Normal (Hit LRU) | Normal (Hit LRU) |
| Group | 0 | 1 | Miss / bypass L1 | Normal (Hit LRU) |
| Device | 1 | 0 | Miss / bypass L1 | Coherent bypass on multi-L2; else Hit LRU |
| System | 1 | 1 | Miss / bypass L1 | Coherent bypass (always) |

The precise per-instruction behavior (including the effect of `nt` and multi-L2 conditions) is given in the load/store control tables below.

---

## Cache Management Instructions

These instructions explicitly flush or invalidate portions of the cache hierarchy without performing a data load or store.

### `buffer_wbl2` — Write Back L2

Writes back dirty lines from the GL2 (L2 / Infinity Cache / MALL) to memory, making them visible to the CPU and XGMI-connected GPUs. `sc1` is the primary trigger — without it the instruction is a NOP regardless of `sc0`. `sc0` determines whether the writeback applies on single-L2 chips as well.

| `sc1` | `sc0` | L2 Cache Behavior |
|-------|-------|-------------------|
| 0 | any | NOP — no writeback occurs |
| 1 | 0 | NOP if only one L2 cache; write-back dirty data if multiple L2 caches |
| 1 | 1 | Write-back dirty data (always; required for single-L2 configs) |

```asm
buffer_wbl2              ; NOP (sc1=0, no writeback)
buffer_wbl2 sc1          ; Write-back dirty L2 data (effective only on multi-L2 systems) - (agent fence uses this)
buffer_wbl2 sc0 sc1      ; Write-back dirty L2 data (always, regardless of L2 count) - (system fence uses this)
```

### `buffer_inv` — Invalidate Caches

Invalidates cache lines so subsequent loads fetch fresh data rather than stale cached values. Necessary after another agent (CPU, remote GPU) has written to a buffer this GPU has previously cached. `sc1` is the primary trigger for both CU/L1 and L2 invalidation.

| `sc1` | `sc0` | CU / L1 Cache Behavior | L2 Cache Behavior |
|-------|-------|------------------------|-------------------|
| 0 | 0 | NOP | NOP |
| 0 | 1 | Invalidate if TG-Split (workgroup spans multiple CUs); else NOP | NOP |
| 1 | 0 | Invalidate | NOP if one L2 cache; invalidate non-coherently cached lines if multiple L2 caches |
| 1 | 1 | Invalidate | Invalidate non-coherently cached lines (always) |

```asm
buffer_inv               ; NOP (sc0=0, sc1=0)
buffer_inv sc0           ; CU/L1 invalidate only in TG-Split workgroups; L2 unaffected
buffer_inv sc1           ; CU/L1 invalidate + L2 invalidate on multi-L2 systems
buffer_inv sc0 sc1       ; CU/L1 invalidate + L2 invalidate (always; the portable form)
```

TG-Split is when a workgroup is dispatched across multiple CUs. In that case `sc0` alone ensures cross-CU coherency within the workgroup without touching L2.

### Coherency Sequence

The canonical cross-agent coherency sequence is:

```asm
buffer_wbl2 sc0 sc1   ; write-back dirty L2 lines to the system coherency point
s_waitcnt vmcnt(0)    ; wait for all outstanding vector memory ops to retire
buffer_inv  sc0 sc1   ; invalidate CU/L1 and L2 so the next load fetches the written value
```

This is exactly what `__threadfence_system()` emits. Both instructions use `sc0 sc1` to guarantee their effect regardless of L2 count or TG-Split configuration.

Note that `buffer_wbl2` is a write-back (data was buffered in L2, now being evicted), while `sc1` on an individual `global_store` is a write-bypass (data never enters L2 at all). These are mechanically different but share the same consequence: the L2 never has an opportunity to coalesce consecutive stores into wider DRAM transactions, which can substantially reduce write bandwidth (unless the threads' access is coalesced to the memory fabric).

---

## Wait Instructions

Wait instructions stall the wavefront until a counter drops to or below a threshold, ensuring in-flight memory operations have completed before proceeding.

| Instruction | Counter | What it waits for |
|-------------|---------|-------------------|
| `s_waitcnt vmcnt(0)` | VMCNT | All vector memory (global/flat loads and stores) complete |
| `s_waitcnt lgkmcnt(0)` | LGKMCNT | All LDS/GDS/scalar memory/message ops complete |
| `s_waitcnt expcnt(0)` | EXPCNT | Export count (VGPR exports); unused on CDNA4 |
| `s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)` | all | Full drain of all in-flight ops |
| `s_waitcnt_vscnt null, 0` | VSCNT (GFX10+) | Vector store count (separated from load count on RDNA) |

`vmcnt(N)` waits until the count is ≤ N, so `vmcnt(0)` means "wait for all".

---

## Fence Instructions

Fences combine a `buffer_wbl2` and/or `buffer_inv` with a `s_waitcnt` to enforce memory ordering. The scope argument controls which cache levels are included.

| # | C++ | Assembly |
|---|-----|----------|
| 1 | `__builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "")` | `buffer_wbl2 sc0 sc1` / `s_waitcnt vmcnt(0)` / `buffer_inv sc0 sc1` |
| 2 | `__builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "agent")` | `buffer_wbl2 sc1` / `s_waitcnt vmcnt(0)` / `buffer_inv sc1` |
| 3 | `__builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "workgroup")` | `s_waitcnt vmcnt(0)` |
| 4 | `__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "")` | `s_waitcnt vmcnt(0)` / `buffer_inv sc0 sc1` |
| 5 | `__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent")` | `s_waitcnt vmcnt(0)` / `buffer_inv sc1` |
| 6 | `__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup")` | `s_waitcnt vmcnt(0)` |
| 7 | `__builtin_amdgcn_fence(__ATOMIC_RELEASE, "")` | `buffer_wbl2 sc0 sc1` / `s_waitcnt vmcnt(0)` |
| 8 | `__builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent")` | `buffer_wbl2 sc1` / `s_waitcnt vmcnt(0)` |
| 9 | `__builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup")` | `s_waitcnt vmcnt(0)` |
| 10 | `__builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "")` | `buffer_wbl2 sc0 sc1` / `s_waitcnt vmcnt(0)` / `buffer_inv sc0 sc1` |
| 11 | `__builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "agent")` | `buffer_wbl2 sc1` / `s_waitcnt vmcnt(0)` / `buffer_inv sc1` |
| 12 | `__threadfence_system()` | `buffer_wbl2 sc0 sc1` / `s_waitcnt vmcnt(0)` / `buffer_inv sc0 sc1` |
| 13 | `__threadfence()` | `buffer_wbl2 sc1` / `s_waitcnt vmcnt(0)` / `buffer_inv sc1` |
| 14 | `__threadfence_block()` | `s_waitcnt vmcnt(0) lgkmcnt(0)` |

### Scope mapping

| Scope string | Meaning |
|--------------|---------|
| `""` (empty) | System — CPU + all GPUs (default) |
| `"agent"` | Current GPU device only |
| `"workgroup"` | Current CU / workgroup only |
| `"wavefront"` | Current wavefront (64 lanes) only |

---

## Memory Load Instructions

### Global loads (`global_load_*`)

Used for flat/global address space pointers (e.g., raw `void*` or `__global` pointers).

| Instruction | Width | C++ equivalent |
|-------------|-------|----------------|
| `global_load_ubyte` | 8-bit unsigned | `*(uint8_t*)ptr` |
| `global_load_sbyte` | 8-bit signed | `*(int8_t*)ptr` |
| `global_load_ushort` | 16-bit unsigned | `*(uint16_t*)ptr` |
| `global_load_dword` | 32-bit | `*(uint32_t*)ptr` |
| `global_load_dwordx2` | 64-bit | `*(uint64_t*)ptr` |
| `global_load_dwordx3` | 96-bit | 3 × `uint32_t` |
| `global_load_dwordx4` | 128-bit | `*(uint4*)ptr` |

### Load cache control (from ISA)

The combination of `sc1`, `sc0`, and `nt` selects the coherency scope and cache eviction policy for every load instruction. "Hit LRU" means the line is fetched into cache with normal LRU priority; "Miss Evict" means the CU cache is bypassed (data comes from L2 or memory, line not retained in L1); "Coherent Cache Bypass" means the L2 is bypassed and data is fetched directly from memory.

| Scope | `sc1` | `sc0` | `nt` | CU / L1 Cache | L2 Cache | Last-Level Cache |
|-------|-------|-------|------|----------------|----------|------------------|
| Wave | 0 | 0 | 0 | Hit LRU | Hit LRU | Hit LRU |
| Wave | 0 | 0 | 1 | Miss Evict | Hit Stream | Hit Evict |
| Group | 0 | 1 | 0 | Hit LRU | Hit LRU | Hit Evict |
| Group | 0 | 1 | 1 | Miss Evict | Hit Stream | Hit Evict |
| Device | 1 | 0 | 0 | Miss Evict | (1 L2): Hit LRU; (>1 L2): Coherent Cache Bypass | Hit LRU |
| Device | 1 | 0 | 1 | Miss Evict | (1 L2): Hit Stream; (>1 L2): Coherent Cache Bypass | Hit Evict |
| System | 1 | 1 | 0 | Miss Evict | Coherent Cache Bypass | Hit LRU |
| System | 1 | 1 | 1 | Miss Evict | Coherent Cache Bypass | Hit Evict |

Key observations:
- `sc1` alone (Device scope) bypasses L2 only on multi-L2 chips; on single-L2 chips L2 is still used.
- `sc1 sc0` (System scope) bypasses L2 unconditionally.
- `nt` independently controls eviction priority (LRU → Stream/Evict) without changing the coherency scope.
- `nt` alone (Wave + NT) bypasses the CU/L1 cache (Miss Evict) despite having no explicit scope flag — useful for streaming reads that should not pollute L1.

### Buffer loads (`buffer_load_*`)

Use a buffer descriptor (4 SGPR "resource constant", also called V#) for bounds-checked, strided, or structured access. Preferred by the compiler for kernel arguments and structured data.

```asm
buffer_load_dword   v0, v1, s[0:3], s4  offen
buffer_load_dwordx4 v[0:3], v4, s[0:3], s4  offen
```

---

## Buffer Descriptor (V#) Layout

A V# is a 128-bit (4 × DWORD) value held in four consecutive SGPRs, e.g. `s[0:3]`. Its field layout for GFX9/CDNA (MI series):

| Bits | Size | Name | Description |
|------|------|------|-------------|
| 47:0 | 48 | Base address | Byte address |
| 61:48 | 14 | Stride | Bytes 0 to 16383 |
| 62 | 1 | Cache swizzle | Optionally swizzle texture cache TC L1 cache banks |
| 63 | 1 | Swizzle enable | Swizzle AOS according to stride, index_stride, and element_size; else linear (stride × index + offset) |
| 95:64 | 32 | Num_records | In units of stride or bytes |
| 98:96 | 3 | Dst_sel_x | Destination channel select: 0=0, 1=1, 4=R, 5=G, 6=B, 7=A |
| 101:99 | 3 | Dst_sel_y | |
| 104:102 | 3 | Dst_sel_z | |
| 107:105 | 3 | Dst_sel_w | |
| 110:108 | 3 | Num format | Numeric data type (float, int, ...) |
| 114:111 | 4 | Data format | Number of fields and size of each field. For MUBUF with ADD_TID_EN=1, this holds Stride[17:14] |
| 115 | 1 | User VM Enable | Resource is mapped via tiled pool / heap |
| 116 | 1 | User VM mode | Unmapped behavior: 0=null (return 0 / drop write); 1=invalid (error) |
| 118:117 | 2 | Index stride | 8, 16, 32, or 64. Used for swizzled buffer addressing |
| 119 | 1 | Add tid enable | Add thread ID to the index to calculate the address |
| 122:120 | 3 | RSVD | Reserved (must be zero) |
| 123 | 1 | NV | Non-volatile (0=volatile) |
| 125:124 | 2 | RSVD | Reserved (must be zero) |
| 127:126 | 2 | Type | Value == 0 for buffer |

Key rules:
- When `STRIDE = 0`: `NUM_RECORDS` is the buffer size in bytes; `offen` VGPR is a raw byte offset.
- When `STRIDE > 0`: `NUM_RECORDS` is the number of elements; `idxen` VGPR is an element index; effective address = BASE + index × STRIDE + offen.
- Out-of-bounds accesses return 0 for loads and are silently dropped for stores — hardware bounds checking is free.

### Building a V# in HIP/C++

```cpp
// Raw byte buffer - the pattern used by rocSHMEM and most custom kernels
uint32_t desc[4];
uintptr_t base = reinterpret_cast<uintptr_t>(ptr);
desc[0] = static_cast<uint32_t>(base);
desc[1] = static_cast<uint32_t>(base >> 32);  // upper 16 bits only; stride=0
desc[2] = static_cast<uint32_t>(size_bytes);   // NUM_RECORDS
desc[3] = 0x00027000;  // DST_SEL=xyzw passthrough, FORMAT=32_FLOAT, type=buffer
// Pass desc as __builtin parameter or via inline asm "s" constraint
```

### Instruction modifiers

| Modifier | Meaning |
|----------|---------|
| `offen` | Add VGPR as byte offset to base |
| `idxen` | Use VGPR as element index (multiplied by STRIDE) |
| `offen idxen` | VGPR pair: v[idx, off] — index + byte offset |
| `offset:N` | Add constant immediate byte offset |
| `glc` / `sc0` | Group scope: CU/L1 cache miss (data comes from / goes to L2) |
| `slc` / `sc1` | Device/System scope: L2 coherent bypass on multi-L2 chips; see load/store control tables |
| `lds` | Write load result directly to LDS instead of VGPRs |

---

## Memory Store Instructions

### Global stores (`global_store_*`)

| Instruction | Width |
|-------------|-------|
| `global_store_byte` | 8-bit |
| `global_store_short` | 16-bit |
| `global_store_dword` | 32-bit |
| `global_store_dwordx2` | 64-bit |
| `global_store_dwordx3` | 96-bit |
| `global_store_dwordx4` | 128-bit (highest bandwidth per instruction) |

### Store cache control (from ISA)

The combination of `sc1`, `sc0`, and `nt` selects the coherency scope and eviction policy. "Miss LRU" for the CU column means the store goes to L2 without allocating in L1; "Miss Evict" means the line is written to L2 and marked for early eviction; "Coherent Cache Bypass" means the write goes directly to memory, bypassing L2.

| Scope | `sc1` | `sc0` | `nt` | CU / L1 Cache | L2 Cache | Last-Level Cache |
|-------|-------|-------|------|----------------|----------|------------------|
| Wave | 0 | 0 | 0 | Miss LRU | Hit LRU | Hit LRU |
| Wave | 0 | 0 | 1 | Miss Evict | Hit Stream | Hit Evict |
| Group | 0 | 1 | 0 | Miss LRU | Hit LRU | Hit LRU |
| Group | 0 | 1 | 1 | Miss Evict | Hit Stream | Hit Evict |
| Device | 1 | 0 | 0 | Miss Evict | (1 L2): Hit LRU; (>1 L2): Coherent Cache Bypass | Hit LRU |
| Device | 1 | 0 | 1 | Miss Evict | (1 L2): Hit Stream; (>1 L2): Coherent Cache Bypass | Hit Evict |
| System | 1 | 1 | 0 | Miss Evict | Coherent Cache Bypass | Hit LRU |
| System | 1 | 1 | 1 | Miss Evict | Coherent Cache Bypass | Hit Evict |

```asm
global_store_dwordx4 v[0:3], v4, s[0:1]              ; Wave scope: write allocates in L2 (best BW)
global_store_dwordx4 v[0:3], v4, s[0:1]  sc0         ; Group scope: bypasses CU/L1, L2 still buffers
global_store_dwordx4 v[0:3], v4, s[0:1]  sc1         ; Device scope: L2 bypass on multi-L2 chips
global_store_dwordx4 v[0:3], v4, s[0:1]  sc0 sc1     ; System scope: L2 coherent bypass (kills write BW)
global_store_dwordx4 v[0:3], v4, s[0:1]  nt          ; Wave + streaming: evict-first in L2
```

Using `sc0 sc1` (System scope) on every store is a write-bypass: each write skips L2 entirely and goes directly to memory. This prevents L2 from write-coalescing consecutive stores into wider DRAM transactions — and unlike `buffer_wbl2` (which flushes data that was cached), write-bypass gives the L2 no opportunity to buffer at all, so effective write bandwidth drops substantially. The correct pattern for high-BW stores is to use no flags (or at most `sc0`), then issue a single `buffer_wbl2 sc0 sc1` at the end to make the data coherent. However, this idea backfires when data size is large enough to fill either of the cache levels, resulting in thrashing.

### Buffer stores (`buffer_store_*`)

Analogous to buffer loads; use a descriptor for bounds checking and structured access. Same modifiers (`offen`, `idxen`, `glc`/`sc0`, `slc`/`sc1`) apply.

```asm
buffer_store_dword   v0,    v1, s[0:3], s4  offen          ; 32-bit store
buffer_store_dwordx4 v[0:3], v4, s[0:3], s4  offen          ; 128-bit store (best BW)
buffer_store_dwordx4 v[0:3], v4, s[0:3], s4  offen sc0      ; write-through L1
buffer_store_dwordx4 v[0:3], v4, s[0:3], s4  offen sc0 sc1  ; write-bypass L1+L2
```

`buffer_store` vs `global_store`: `buffer_store` uses the V# descriptor for hardware bounds checking and structured (strided) addressing. `global_store` uses a flat 64-bit virtual address directly. For raw pointer arithmetic, `global_store` is simpler; for structured data or when bounds safety matters, `buffer_store` is preferred (and typically what the compiler emits for kernel arguments).

---

## Atomic Operations

Atomic operations read-modify-write a memory location in one indivisible step. The `sc0` / `sc1` flags control the coherency scope of the atomic.

### Global atomics

When `glc`/`sc0` is not set, no value is returned and the destination register is omitted entirely from the assembler syntax:

```asm
; No-return forms - destination register is absent
global_atomic_add     v0, v1, s[0:1]           ; 32-bit add  (v0=addr, v1=data)
global_atomic_add_x2  v[0:1], v[2:3], s[0:1]  ; 64-bit add  (v[0:1]=addr, v[2:3]=data)
global_atomic_cmpswap v0, v[1:2], s[0:1]       ; 32-bit CAS  (v0=addr, v[1:2]={new,cmp})
global_atomic_swap    v0, v1, s[0:1]            ; 32-bit exchange
```

Append `sc0` (or the legacy alias `glc`) to get the old value back; only then is a destination register present:

```asm
; Return-value forms - destination register is first
global_atomic_add     v2, v0, v1, s[0:1] sc0   ; v2 ← old value; v0=addr, v1=data
global_atomic_add_x2  v[4:5], v[0:1], v[2:3], s[0:1] sc0
global_atomic_cmpswap v3, v0, v[1:2], s[0:1] sc0
```

### LDS atomics

```asm
ds_add_u32  v0, v1                                    ; 32-bit LDS add
ds_cmpst_b32 v0, v1, v2                               ; 32-bit LDS CAS
```

---

## Timing / Hardware Counters

### `wall_clock64` — GPU hardware wall clock

Returns the 64-bit GPU wall clock in cycles. The clock frequency is fixed at device initialization and readable via `hipDeviceAttributeWallClockRate` (in kHz).

```cpp
// C++ / HIP
uint64_t t = wall_clock64();

// Conversion to microseconds
double us = static_cast<double>(cycles) * 1000.0 / static_cast<double>(freq_khz);
```

Assembly emitted (`s_memrealtime`):

```asm
s_memrealtime s[0:1]    ; read 64-bit real-time counter into two SGPRs
s_waitcnt lgkmcnt(0)    ; wait for the counter to be valid
```

`s_memrealtime` reads a free-running counter that does not stop during sleep or power-gating, making it suitable for cross-block latency measurement (unlike `s_memtime` which may stall).

### `s_memtime` — Shader memory clock

Similar to `s_memrealtime` but counts only active clock cycles (pauses during wavefront sleep). On GFX9+, `s_memtime` writes its result directly to the destination SGPRs without going through the scalar memory messaging unit, so no `s_waitcnt lgkmcnt(0)` is required afterward. Less useful for wall-time measurements since it does not tick when the wavefront is sleeping:

```asm
s_memtime s[0:1]    ; result available immediately; no lgkmcnt wait needed on GFX9+
```

### Performance counters (`__builtin_amdgcn_s_getreg`)

Read hardware performance registers. The correct macros are defined in `<amdgcnintrin.h>`:

```cpp
#include <amdgcnintrin.h>
uint32_t hw_id  = __builtin_amdgcn_s_getreg(__AMDGCN_HWREG_HW_ID);
uint32_t ib_sts = __builtin_amdgcn_s_getreg(__AMDGCN_HWREG_IB_STS);
// → s_getreg_b32 s0, hwreg(HW_ID)
```

---

## Workgroup / Wave Synchronization

| C++ | Assembly | Description |
|-----|----------|-------------|
| `__syncthreads()` | `s_barrier` | Barrier for all threads in a workgroup; also implies LDS ordering |
| `__builtin_amdgcn_wave_barrier()` | (no instruction emitted) | Compile-time scheduling barrier only. Wavefronts execute in lockstep (SIMD), so all 64 lanes are always at the same PC — a runtime intra-wave barrier is physically meaningless. This intrinsic tells the LLVM instruction scheduler not to reorder memory or ALU ops across it, but generates zero machine instructions. |
| `__threadfence_block()` | `s_waitcnt vmcnt(0) lgkmcnt(0)` | Drain both global (vmcnt) and LDS/scalar (lgkmcnt) memory ops; makes all writes visible within the workgroup |
| `__threadfence()` | `buffer_wbl2 sc1` / `s_waitcnt vmcnt(0)` / `buffer_inv sc1` | Ensure global writes visible within the GPU |
| `__threadfence_system()` | `buffer_wbl2 sc0 sc1` / `s_waitcnt vmcnt(0)` / `buffer_inv sc0 sc1` | Ensure global writes visible to all agents (CPU + GPUs) |

### LDS (Local Data Share) operations

```asm
ds_write_b32  v0, v1                ; 32-bit store to LDS
ds_write_b128 v0, v[1:4]           ; 128-bit store to LDS
ds_read_b32   v0, v1               ; 32-bit load from LDS
ds_read_b128  v[0:3], v4           ; 128-bit load from LDS
ds_swizzle_b32 v0, v1  offset:...  ; swizzle lanes within a wavefront via LDS
```

---

## Cross-Lane Operations

These operate across the 64 lanes of a wavefront without going through LDS.

| Instruction | C++ intrinsic | Description |
|-------------|---------------|-------------|
| `ds_permute_b32` | `__builtin_amdgcn_ds_permute` | Permute lane data using another lane's index |
| `ds_bpermute_b32` | `__builtin_amdgcn_ds_bpermute` | Backward permute (dst lane selects src lane) |
| `v_readlane_b32` | `__builtin_amdgcn_readlane` | Broadcast one lane's value to all lanes (scalar result) |
| `v_writelane_b32` | `__builtin_amdgcn_writelane` | Write a scalar value into one specific lane |
| `v_readfirstlane_b32` | `__builtin_amdgcn_readfirstlane` | Move the value of the lowest active lane to an SGPR |

### Wavefront reductions

```asm
; Sum all 64 lanes - compiler lowers to a tree of ds_swizzle + ds_permute
v_add_f32 v0, v0, v1    ; after lane folding, result is in lane 0
```

```cpp
// Warp-level reduction intrinsics (ROCm / HIP)
float sum = __builtin_amdgcn_ds_fmaxf(val, 0);   // max reduction within wave
```

### Active lane mask

```asm
s_mov_b64 s[0:1], exec          ; copy current exec mask to SGPRs
v_cmp_gt_u32 vcc, v0, v1        ; sets bits in VCC for lanes where v0 > v1
s_and_b64 exec, exec, vcc       ; mask out inactive lanes
```

---

## Scalar Memory

Scalar loads fetch data that is uniform across all lanes (e.g., kernel arguments, constants). They use SGPRs and go through the scalar data cache (K-cache / L1K).

| Instruction | Width | Use |
|-------------|-------|-----|
| `s_load_dword` | 32-bit | Single uniform value |
| `s_load_dwordx2` | 64-bit | Pointer (two 32-bit SGPRs) |
| `s_load_dwordx4` | 128-bit | Buffer descriptor |
| `s_load_dwordx8` | 256-bit | Pair of buffer descriptors |
| `s_load_dwordx16` | 512-bit | Four buffer descriptors |

```asm
s_load_dwordx2  s[0:1], s[2:3], 0x0    ; load 64-bit pointer from kernel args
s_waitcnt lgkmcnt(0)                    ; wait for scalar memory to complete
```

Scalar memory is cached in the K-cache (separate from the vector L1). It is read-only from the kernel's perspective — use global or LDS instructions for writable data.

---

## LLVM Builtin Intrinsics

Clang exposes AMD GPU operations at two levels below C++/HIP:
- `__builtin_amdgcn_*` : Clang compiler builtins. Callable directly from C++/HIP device code; the compiler lowers them to the matching machine instruction.
- `llvm.amdgcn.*` : LLVM IR intrinsics. Not directly callable from C++, but reachable via the `__asm__` symbol-alias trick (see below).

### Fence and Barrier Builtins

```cpp
// Core fence primitive — all __threadfence_* functions reduce to this
void __builtin_amdgcn_fence(int memory_order, const char* scope);
//   memory_order: __ATOMIC_RELAXED, __ATOMIC_ACQUIRE, __ATOMIC_RELEASE,
//                 __ATOMIC_ACQ_REL, __ATOMIC_SEQ_CST
//   scope:        ""            → system (CPU + all GPUs)
//                 "agent"       → current GPU device
//                 "workgroup"   → current CU / workgroup
//                 "wavefront"   → current wavefront only

// Workgroup barrier — emits s_barrier; implies memory ordering for both
// global and LDS memory within the workgroup
void __builtin_amdgcn_s_barrier(void);

// Wave scheduling barrier — emits NO machine instructions.
// Prevents the LLVM instruction scheduler from reordering across this point.
// Useful when you need to separate two groups of instructions for analysis
// but do not want actual synchronization overhead.
void __builtin_amdgcn_wave_barrier(void);
```

How HIP's high-level functions are built from these primitives (from `amd_device_functions.h`):

```cpp
__device__ void __threadfence() {
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "agent");
}

__device__ void __threadfence_block() {
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "workgroup");
}

__device__ void __threadfence_system() {
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "");
}

// Full workgroup barrier with fences on both sides of s_barrier.
// The fence before s_barrier is RELEASE (make my writes visible);
// the fence after is ACQUIRE (pick up everyone else's writes).
__device__ void __syncthreads() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}

// Fine-grained variant: fence only over global memory, not LDS
__device__ void __work_group_barrier_global() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "global");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "global");
}
```

### Calling LLVM Intrinsics via Asm Alias

For intrinsic variants not exposed as Clang builtins (e.g., i128 / 16-byte integer loads), use GCC's `__asm__` symbol-renaming extension to declare a C++ function whose link name is the LLVM intrinsic:

```cpp
// Declare a device function whose symbol name IS the LLVM intrinsic name.
// The compiler resolves the call directly to the intrinsic — no wrapper overhead.
extern "C" __device__
int4 __raw_buffer_load_i128(int4 rsrc, int voffset, int soffset, int cache_policy)
    __asm("llvm.amdgcn.raw.buffer.load.i128");

extern "C" __device__
void __raw_buffer_store_i128(int4 val, int4 rsrc, int voffset, int soffset, int cache_policy)
    __asm("llvm.amdgcn.raw.buffer.store.i128");

// Usage — identical to the typed builtins
int4 vec = __raw_buffer_load_i128(desc, byte_offset, 0, 0);
// → buffer_load_dwordx4
__raw_buffer_store_i128(vec, desc, byte_offset, 0, 3);
// → buffer_store_dwordx4  sc0 sc1
```

The same trick works for any LLVM intrinsic:

```cpp
// s_memrealtime via intrinsic name (equivalent to __builtin_amdgcn_s_memrealtime)
extern "C" __device__
unsigned long long __wall_clock()
    __asm("llvm.amdgcn.s.memrealtime");

// Non-temporal buffer load not exposed as a builtin
extern "C" __device__
float4 __buffer_load_nt(int4 rsrc, int voffset, int soffset)
    __asm("llvm.amdgcn.raw.buffer.load.v4f32");
// call with cache_policy=4 (nt flag) to get the non-temporal behavior
```

### Hardware Register Builtins

```cpp
// Read a hardware register into a 32-bit value.
// Register constants are defined in <amdgcnintrin.h> as __AMDGCN_HWREG_*.
unsigned int __builtin_amdgcn_s_getreg(unsigned int reg);
// → s_getreg_b32 s_dst, hwreg(REG)

// Common register IDs:
//   __AMDGCN_HWREG_HW_ID      - identifies current CU, SIMD, and wave slot
//   __AMDGCN_HWREG_IB_STS     - instruction buffer status
//   __AMDGCN_HWREG_STATUS     - shader status (SCC, EXECZ, VCCZ, ...)
//   __AMDGCN_HWREG_WAVE_ID    - wave ID within the SIMD unit
```

Example — identifying which CU a wavefront is running on:

```cpp
unsigned int hw_id = __builtin_amdgcn_s_getreg(__AMDGCN_HWREG_HW_ID);
unsigned int wave_id = (hw_id >>  0) & 0xF;   // bits [3:0]
unsigned int simd_id = (hw_id >>  4) & 0x3;   // bits [5:4]
unsigned int cu_id   = (hw_id >>  8) & 0xF;   // bits [11:8]
```

---

## References

- **AMD Instinct CDNA4 Instruction Set Architecture** — MI300 / MI355 (gfx940, gfx942, gfx950). Primary source for `buffer_wbl2`, `buffer_inv`, and the load/store cache control tables used in this document.
- **AMD Instinct MI300 CDNA3 Instruction Set Architecture** — MI300 series (gfx940, gfx942). Compare with CDNA4 to see how flag semantics evolved between generations.
- **AMD RDNA4 Instruction Set Architecture** — RDNA4 consumer/workstation GPUs (gfx12xx). Reference for `dlc` and RDNA-specific cache hierarchy (L0 / GL1 / GL2).
- **LLVM AMDGPU Backend Documentation** — Overview of the LLVM AMDGPU target: address spaces, calling conventions, memory model, intrinsics, and resource types.
- **LLVM AMDGPU GFX950 Assembly Reference** — Full assembler reference for gfx950 (MI355 / CDNA4), including every instruction encoding, modifier, and operand constraint.
- **AMD GPU Architecture and Programming Documentation** — GPUOpen portal collecting architecture whitepapers, optimization guides, and blog posts for both RDNA and CDNA architectures.
