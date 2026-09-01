---
name: dpp-row-operations
description: Replace LDS-based intra-wave shuffles (ds_bpermute, __shfl, __shfl_up, __shfl_xor) with DPP cross-lane VALU ops for wavefront reductions, prefix sums/scans, and broadcasts on AMD CDNA (gfx942/gfx950). Use when optimizing intra-wavefront communication or removing LDS traffic and s_waitcnt lgkmcnt from shuffle-based reductions.
---

# DPP Row Operations for Intra-Wave Communication

## When to use

A kernel uses LDS-based shuffles (`ds_bpermute`, `__shfl`, `__shfl_up`, `__shfl_xor`) for reductions, prefix sums, or broadcasts within a wavefront. DPP replaces these with direct cross-lane VALU ops — no LDS traffic, no `s_waitcnt lgkmcnt`.

## Background

On AMD CDNA (gfx942/gfx950), a wavefront is 64 lanes organized as **4 rows of 16 lanes**. DPP operates on this 4x16 grid:

```
Row 0: lanes  0-15
Row 1: lanes 16-31
Row 2: lanes 32-47
Row 3: lanes 48-63
```

DPP modifiers move data between lanes **within each row** independently. Cross-row communication requires `__shfl` or explicit fixup.

## Key DPP modifiers

| Modifier | What it does | Identity element |
|----------|-------------|-----------------|
| `row_shr:N` | Shift right by N within each 16-lane row. Lane gets value from lane+N. OOB lanes get 0 (with `bound_ctrl:1`) | 0 (for add) |
| `row_shl:N` | Shift left by N within each 16-lane row. Lane gets value from lane-N. | 0 |
| `row_bcast:15` | Broadcast lane 15 of the **previous** row to all lanes in the current row. Row 0 gets its own lane 15. | N/A |

`bound_ctrl:1` is critical — it makes OOB source lanes return 0, which is the identity for addition. Without it, OOB lanes return the destination's original value.

## Pattern 1: Inclusive prefix sum (16-lane row)

Replaces 4 `__shfl_up` calls with 4 DPP steps. Each row computes an independent prefix sum.

```cpp
#define DPP_ADD_STEP(SHIFT, val, tmp) \
    asm volatile( \
        "s_nop 1\n" \
        "v_mov_b32_dpp %1, %0 row_shr:" #SHIFT \
        " row_mask:0xf bank_mask:0xf bound_ctrl:1\n" \
        "v_add_u32 %0, %0, %1\n" \
        : "+v"(val), "=&v"(tmp))

int x = input_value;
int tmp;
DPP_ADD_STEP(1, x, tmp);  // stride 1
DPP_ADD_STEP(2, x, tmp);  // stride 2
DPP_ADD_STEP(4, x, tmp);  // stride 4
DPP_ADD_STEP(8, x, tmp);  // stride 8
// x now holds inclusive prefix sum within 16-lane row
```

**Cross-row fixup** (to extend to full 64-lane wavefront):

After DPP, each row has an independent prefix sum. To combine them:

```cpp
int lane = threadIdx.x & 63;
int row_total_0 = __shfl(x, 15);   // total of row 0
int row_total_1 = __shfl(x, 31);   // total of row 1
int row_total_2 = __shfl(x, 47);   // total of row 2

if (lane >= 16) x += row_total_0;
if (lane >= 32) x += row_total_1;
if (lane >= 48) x += row_total_2;
```

**Do NOT use `__shfl_up` for the cross-row step.** `__shfl_up(x, 16)` gives lane 16 the value from lane 0, which is only `input[0]` — not `sum(input[0..15])`. You need `__shfl` to explicitly read the row-end lanes.

## Pattern 2: Max reduction (16-lane row)

Replaces shuffle-based tree reduction. Uses `row_shl` to propagate values left, then `row_shr` to propagate back right. After 8 steps every lane holds the row maximum.

```cpp
__device__ uint32_t dpp_row_max_broadcast(uint32_t key) {
    uint32_t tmp;
    asm volatile(
        "s_nop 1\n"
        "v_mov_b32_dpp %1, %0 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
        "v_max_u32 %0, %0, %1\n"
        "s_nop 1\n"
        "v_mov_b32_dpp %1, %0 row_shl:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
        "v_max_u32 %0, %0, %1\n"
        "s_nop 1\n"
        "v_mov_b32_dpp %1, %0 row_shl:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
        "v_max_u32 %0, %0, %1\n"
        "s_nop 1\n"
        "v_mov_b32_dpp %1, %0 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
        "v_max_u32 %0, %0, %1\n"
        "s_nop 1\n"
        "v_mov_b32_dpp %1, %0 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
        "v_max_u32 %0, %0, %1\n"
        "s_nop 1\n"
        "v_mov_b32_dpp %1, %0 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
        "v_max_u32 %0, %0, %1\n"
        "s_nop 1\n"
        "v_mov_b32_dpp %1, %0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
        "v_max_u32 %0, %0, %1\n"
        "s_nop 1\n"
        "v_mov_b32_dpp %1, %0 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n"
        "v_max_u32 %0, %0, %1\n"
        : "+v"(key), "=&v"(tmp)
    );
    return key;
}
```

## VALU-to-DPP hazard rules

There is a 2-wait-state hazard when a VALU instruction writes a VGPR and the **next** VALU+DPP instruction reads that **same VGPR as its DPP source**.

- **`s_nop 1` before each `v_mov_b32_dpp`** covers this hazard (1 NOP = 2 wait states on CDNA).
- **No NOP needed after `v_mov_b32_dpp`** — the subsequent `v_add/v_max` reads the DPP result via regular VGPR, not DPP.
- **Use a single `asm volatile` block** for multi-step DPP sequences. Separate blocks let the compiler insert spurious `s_nop 0` between them.

## When DPP is NOT the right tool

- **Cross-row operations** (stride >= 16): DPP only works within 16-lane rows. Use `__shfl` for cross-row.
- **Irregular permutations**: DPP supports fixed shift/rotate patterns, not arbitrary lane-to-lane.
- **Wavefront-wide broadcast**: `__shfl(x, lane)` is simpler and just as fast for broadcasting one lane's value to all 64 lanes.
- **`row_bcast:15`**: On gfx950, this broadcasts lane 15 of the *previous* row (not the same row). Rarely useful for reductions.
