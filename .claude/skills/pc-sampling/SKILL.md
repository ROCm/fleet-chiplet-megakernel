---
name: pc-sampling
description: >-
  Profile AMD GPU (CDNA3/CDNA4: MI300, MI350) kernels with rocprofv3 PC
  sampling to find hotspots, stall reasons (waitcnt, I-fetch, ALU dependency,
  barrier, LDS bank conflict), per-pipe utilization, occupancy, and register
  spills — per source/assembly line, without instrumenting source. Use when
  you don't know what to optimize first, why a kernel or phase stalls, or need
  to confirm memory latency, LDS bank conflicts, low occupancy, register
  spills, or pipeline oversubscription with data.
---

# Kernel Profiling with PC Sampling

## When to use

You need to identify **where** a kernel spends its time and **why** it stalls — without instrumenting the source code. PC sampling profiles the entire workload in a single pass and reports hotspots, stall reasons, and pipeline utilization per source/assembly line. Use it when:

- You have a new kernel (or workload) and don't know what to optimize first
- Phase profiling shows a phase is slow but you don't know *why*
- You suspect instruction-fetch stalls, LDS bank conflicts, memory latency, or pipeline oversubscription but need data to confirm

PC sampling is complementary to ATT (instruction traces on a few CUs) and phase profiling (cycle-level timing of kernel sections). It trades detail for breadth — it covers all CUs and all dispatches in one run.

## Prerequisites

- amdgpu driver from ROCm >= 7.0
- rocprofiler-sdk >= 6.4 (ships with ROCm 7.0+)
- CDNA3/CDNA4 GPU (MI300A, MI300X, MI308, MI350, MI355)

## Collecting PC samples

### Basic PC sample collection

```bash
rocprofv3 \
  --kernel-trace --stats \
  --pc-sampling-beta-enabled true \
  --pc-sampling-unit time \
  --pc-sampling-method host_trap \
  --pc-sampling-interval 5000 \
  -f csv json -d <out_dir> -- <cmd>
```

Key options:
- `--pc-sampling-unit`: `time` or `cycles`. `time` gives wall-clock sampling; `cycles` gives cycle-count sampling.
- `--pc-sampling-method`: `host_trap` or `stochastic`. Both work on CDNA3/4. `stochastic` with a large interval (e.g. `$((1024 * 1024))`) is recommended for low-overhead whole-workload profiling.
- `--pc-sampling-interval`: Lower = more samples = more overhead. `5000` for `time` mode, `$((1024 * 1024))` for `stochastic` mode are good defaults.
- `--kernel-trace`: Required — ties samples to kernel dispatches.
- `--stats`: Adds summary statistics.

Key Gotchas:
- do not try and read/write from /tmp when using rocprofv3 it will cause errors.
- rocprofv3 will sometimes hang/crash when trying to sample multiple PCs at once.

### Collecting spill/resource info

In `<out_dir>/<host>/<pid>_kernel_trace.csv`, check these columns to understand register pressure and resource usage:

| Field | What it tells you |
|-------|-------------------|
| `Scratch_Size` | Bytes of scratch (spill) memory per work-item. Non-zero = register spills. |
| `VGPR_Count` | Vector general-purpose registers used. Max 256 on CDNA3/4. Higher = lower occupancy. |
| `Accum_VGPR_Count` | Accumulation VGPRs (AGPRs) used. Used by MFMA instructions. |
| `SGPR_Count` | Scalar GPRs used. |
| `LDS_Block_Size` | LDS bytes allocated per workgroup. |

### Collecting LDS bank conflict counters

```bash
rocprofv3 \
  --pmc LDSBankConflict LdsBankConflict LdsLatency LdsUtil \
        SQ_INSTS_LDS SQ_INSTS_VALU SQ_INSTS_VALU_MFMA_F16 SQ_INSTS_MFMA \
  -f csv json -d <out_dir> -- <cmd>
```

Results are in `<out_dir>/<host>/<pid>_counter_collection.csv`.

| Counter | What it tells you |
|---------|-------------------|
| `LDSBankConflict` | Percent GPU time stalled by LDS bank conflicts |
| `LdsBankConflict` | Conflicts per access |
| `LdsLatency` | Average LDS latency (cycles) |
| `LdsUtil` | LDS utilization (fraction) |
| `SQ_INSTS_LDS` | Total LDS instructions issued |
| `SQ_INSTS_VALU` | Total VALU instructions issued |
| `SQ_INSTS_VALU_MFMA_F16` | MFMA FP16 instructions issued |
| `SQ_INSTS_MFMA` | Total MFMA instructions issued |

## Understanding PC sampling output

### Key concepts

**Hotspots**: Where the program spent time, regardless of whether that time was productive. Useful for identifying oversubscribed pipelines.

**Holes**: Samples where a hardware pipeline was idle and could have done work — these are in the critical path and directly cause slower performance. Holes are the primary signal for optimization.

**Active samples**: Which instructions executed most frequently.

Relationship: Hotspots = Holes + Active samples.

### CDNA3/4 CU execution pipelines

Each CU has these independent issue pipes (one instruction per pipe per quad-cycle):

| Pipeline | Work |
|----------|------|
| SCALAR | Uniform data: kernel args, addresses, control-flow |
| VALU | Per-workitem ALU: FP, integer, conversions |
| MATRIX | MFMA operations (low-precision <= 16b) |
| LDS | Shared memory, shuffles |
| FLAT | Global / scratch / flat memory |
| VMEM_TEX | Buffer memory |
| MISC | Branching, barriers |

### Stall reasons

When the sampler catches a wave not making progress, it records **why**:

| Stall reason | Meaning |
|--------------|---------|
| NO INSTRUCTION AVAILABLE | Instruction not in the I-buffer (branch target, I$ miss). Common with heavy branching at low occupancy. |
| ALU DEPENDENCY | Instruction blocked on a hardware-internal data dependency. |
| WAITCNT | Wave waiting at a memory fence — data not yet returned from memory. |
| INTERNAL INSTRUCTION | Wave executing an internal instruction (NOP). |
| BARRIER WAIT | Wave waiting at a barrier for other waves in the workgroup. |
| ARBITER NOT WIN | Wave was ready but another wave won arbitration — indicates pipeline oversubscription (not a problem per se). |
| ARBITER WIN EX STALL | Wave won arbitration but the execution pipe stalled (backpressure or resource stall). |

### Pipeline status fields

| Field | Meaning |
|-------|---------|
| ISSUE_PIPE | A wave on the sampled SIMD attempted to issue to this pipe |
| STALL_PIPE | The issued instruction stalled (STALL_PIPE <= ISSUE_PIPE) |

### Summary metrics

| Metric | Meaning |
|--------|---------|
| IPC | Instructions issued per quad-cycle across all pipes |
| ACTIVE THREADS | Average work-items predicated on (max 64 on CDNA3/4) |
| OCCUPANCY | Average waves per CU (max 32 on CDNA3/4). Low occupancy = less latency hiding. |
| UTILIZATION | Fraction of time a pipe was attempting to issue |
| IMBALANCE | Excess/deficit of samples from a given CU or chiplet vs. even distribution |

## Interpreting results: what to look for

### 1. Low occupancy + NO INSTRUCTION AVAILABLE

The kernel runs few waves per SIMD (e.g. occupancy ~1), and a large fraction of time is `NO INSTRUCTION AVAILABLE`. This typically means heavy branching with no waves to hide the I-buffer refill latency. Fix: inline hot call targets, reduce branching, or increase occupancy.

**Example**: A hipBLASLt GEMM ran at 1 wave/SIMD with ~11% of time stalled on I-buffer misses. A single `s_setpc_64` (indirect call to an activation function) burned ~5% of kernel runtime. Inlining the activation functions gave 10-20% kernel speedup.

### 2. FLAT stalls + high WAITCNT

Memory-bound kernel. The FLAT pipe is heavily utilized but mostly stalled, and waves spend significant time at `waitcnt` instructions. Look at the holes table to find which specific load/store lines are bottlenecked. Fix: reduce memory traffic, improve access patterns, prefetch.

### 3. High ARBITER NOT WIN (oversubscription)

A pipe is oversubscribed — many waves compete to issue to the same pipe. This isn't necessarily bad (it means the pipe is fully utilized), but if combined with holes it indicates a structural imbalance. Look at hotspots to find which instructions dominate that pipe.

### 4. LDS bank conflicts

If `LDSBankConflict` is high, LDS accesses are serializing. Check `LdsBankConflict` for conflicts/access. Fix: pad LDS arrays or rearrange access patterns to avoid same-bank access across threads.

### 5. Scratch_Size > 0

Register spills. The kernel uses more VGPRs than available, so the compiler spills to scratch (global memory). This dramatically increases memory traffic. Fix: reduce live variable count, split the kernel, or reduce occupancy target.

## Workflow summary

1. **Collect** — Run your workload once with `rocprofv3` and PC sampling enabled.
2. **Identify hot kernels** — Sort kernels by total GPU wall-clock time from kernel trace.
3. **Read top-level tables** — Check occupancy, pipe utilization, and CU state for each hot kernel.
4. **Find holes** — The holes table shows exactly which source/assembly lines are in the critical path and why. This is the main actionable output.
5. **Drill into hotspots** — For oversubscribed pipes, hotspots show which instructions dominate. Open the source code to understand the pattern.
6. **Collect PMC counters** — If LDS bank conflicts or specific instruction counts are needed, do a separate PMC counter run.
7. **Fix and re-measure** — Make the change, re-profile, verify the holes shrunk.

## Quick reference

```bash
# PC sample collection (stochastic, low overhead)
rocprofv3 --kernel-trace --stats \
  --pc-sampling-beta-enabled true \
  --pc-sampling-unit cycles \
  --pc-sampling-method stochastic \
  --pc-sampling-interval $((1024 * 1024)) \
  -f csv json -d prof_pc -- <cmd>

# PC sample collection (host_trap, higher resolution)
rocprofv3 --kernel-trace --stats \
  --pc-sampling-beta-enabled true \
  --pc-sampling-unit time \
  --pc-sampling-method host_trap \
  --pc-sampling-interval 5000 \
  -f csv json -d prof_pc -- <cmd>

# Resource/spill info
# Check: prof_pc/<host>/<pid>_kernel_trace.csv
# Columns: Scratch_Size, VGPR_Count, Accum_VGPR_Count, SGPR_Count, LDS_Block_Size

# LDS bank conflict + instruction counters
rocprofv3 \
  --pmc LDSBankConflict LdsBankConflict LdsLatency LdsUtil \
        SQ_INSTS_LDS SQ_INSTS_VALU SQ_INSTS_VALU_MFMA_F16 SQ_INSTS_MFMA \
  -f csv json -d prof_pmc -- <cmd>
# Check: prof_pmc/<host>/<pid>_counter_collection.csv
```
