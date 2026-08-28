# Two-Level Fusion: Composable Device Functions for Persistent Kernel Compilation

## Overview

Modern LLM inference on multi-die GPUs (AMD MI300/MI350) is bottlenecked by HBM
traffic from intermediate activations written between operators. We propose a
**two-level fusion framework** that eliminates these round-trips by composing
reusable device functions with pluggable epilogues inside a persistent kernel.

**Level 1 (CODA-style register fusion):** Element-wise operations (RoPE, SwiGLU,
residual add, RMSNorm partial stats, argmax) are fused as epilogues of the
preceding GEMM. The GEMM accumulator remains in registers; the epilogue operates
on it before any HBM write. This eliminates the intermediate tensor entirely.

**Level 2 (Persistent kernel LDS fusion):** Multiple fused-GEMMs (each with
their own epilogue) share LDS within a single persistent kernel. Data flows
between phases through LDS, never touching HBM. The persistent kernel also
eliminates kernel launch overhead and enables cross-layer pipelining.

The compiler's job is to select which operations become epilogues (Level 1) vs
which become LDS-connected phases (Level 2), subject to register pressure and
LDS capacity constraints.

## Device Function Library Architecture

All building blocks live in a single file: `kernels/device_functions.cuh`.

### A. GEMM Mainloop

One templatized MXFP4 GEMM with a pluggable epilogue parameter:

```cpp
template<typename Epilogue, int BATCH_SIZE, int REDUCTION_SIZE, int OUTPUT_PER_WG>
__device__ __noinline__ void gemm_mxfp4(
    void const *input,       // [bs, REDUCTION_SIZE] bf16 or fp8
    void const *weight,      // [n_wgs, wg_bytes] packed MXFP4
    int wg_idx,
    int num_active_tokens,
    Epilogue &epilogue);
```

The mainloop is the SAME depth-4 MFMA pipeline for ALL GEMMs (QKV, O-proj,
gate_up, down, LM head, MoE W13, MoE W2). The only thing that changes is the
epilogue template parameter.

Hardware: `__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4` (FP4 x FP8, K=128)

### B. Composable Epilogues

Each epilogue is a struct with an `operator()` called on the MFMA accumulator
while it is still in VGPR registers:

| Epilogue | What it does | Used by |
|----------|-------------|---------|
| `EpilogueStore` | Convert f32 acc to bf16, write to HBM | Default (unfused) |
| `EpilogueStoreLDS` | Convert f32 acc to bf16, write to LDS | QKV output for attention |
| `EpilogueRoPE` | Rotate pairs of accumulator elements, write to LDS | QKV GEMM |
| `EpilogueSwiGLU` | SiLU(gate_acc) * up_acc, write to LDS | GateUp GEMM |
| `EpilogueResAdd` | acc + load(residual), write to HBM | O-proj, Down-proj |
| `EpilogueResAddRMSNormPartial` | ResAdd + compute partial sum-of-squares | Future: CODA RMSNorm trick |
| `EpilogueArgmax` | Update running (max_val, max_idx) per thread | LM head GEMM |
| `EpilogueBiasAdd` | acc + bias, write to HBM | Biased GEMMs |

### C. Non-GEMM Device Functions

| Function | Description |
|----------|-------------|
| `attention_decode<HD, Q_PER_KV, PAGE_SIZE, MAX_SEQ, CHUNKS>` | Paged attention (CK FMHA, HD=64 or 128) |
| `rmsnorm<HIDDEN, ACTUAL_DIM>` | RMSNorm (all workers redundantly, for prologue) |
| `fp8_quant<SIZE>` | BF16 to FP8 quantization (for GEMM input) |
| `topk_router<NUM_EXPERTS, TOPK>` | MoE routing (softmax + top-K selection) |

### D. Barrier Primitives

| Barrier | Scope | Use |
|---------|-------|-----|
| `barrier_global` | All 240 workers across 8 XCDs | Layer boundaries, all-reduce points |
| `barrier_per_xcd` | 30 workers within one XCD | Intra-phase sync (e.g., QKV done on this XCD) |
| `barrier_single_producer` | 1 writer, N readers | ResAdd (1 worker writes, all read) |
| `barrier_epoch` | Monotonic counter, modular arithmetic | Reusable across iterations without zeroing |

## Per-Model Fusion Tables

### GPT-OSS 120B (MoE, 36 layers, hidden=2880, 128 experts, top-4)

| Operator | Level 1 (Epilogue) | Level 2 (Phase) |
|----------|-------------------|-----------------|
| RMSNorm_1 | Prologue (FP8 quant in LDS) | Phase 1 |
| QKV GEMM | + RoPE epilogue | Phase 1 |
| Attention | (standalone, reads from LDS) | Phase 1 |
| Merge split-KV | (standalone) | Phase 1 |
| O-proj GEMM | + ResAdd epilogue | Phase 2 |
| RMSNorm_2 | Prologue | Phase 2 |
| TopK Router | (standalone GEMV) | Phase 2 |
| MoE W13 GEMM | + SwiGLU epilogue | Phase 3 |
| MoE W2 GEMM | + ResAdd epilogue (atomicAdd to f32 workspace) | Phase 3 |
| LM head GEMM | + Argmax epilogue | Tail |

### Qwen3-8B (Dense, 32 layers, hidden=4096, intermediate=22016)

| Operator | Level 1 (Epilogue) | Level 2 (Phase) |
|----------|-------------------|-----------------|
| RMSNorm_1 | Prologue (FP8 quant in LDS) | Phase 1 |
| QKV GEMM | + RoPE epilogue | Phase 1 |
| Attention | (standalone, CK FMHA HD=128) | Phase 1 |
| O-proj GEMM | + ResAdd epilogue | Phase 2 |
| RMSNorm_2 | Prologue | Phase 2 |
| GateUp GEMM | + SwiGLU epilogue | Phase 3 |
| Down GEMM | + ResAdd epilogue | Phase 3 |
| LM head GEMM | + Argmax epilogue | Tail |

### Key Observation

The SAME device function library serves both models. The compiler selects:
1. Which epilogue to attach to each GEMM
2. How to group fused-GEMMs into persistent kernel phases
3. What barriers to insert between phases
4. How to partition work across XCDs

## HBM Traffic Comparison

Per layer, decode bs=1, hidden_dim H:

| Approach | Intermediate HBM writes | Intermediate HBM reads | Total round-trips |
|----------|------------------------|----------------------|-------------------|
| Kernel-per-operator | ~10 x H bytes | ~10 x H bytes | 10 |
| CODA only (epilogue fusion) | ~4 x H bytes | ~4 x H bytes | 4 |
| Persistent only (LDS fusion) | ~4 x H bytes | ~4 x H bytes | 4 |
| **Two-level (both)** | **0** | **0** | **0** |

With two-level fusion, the only HBM traffic per layer is:
- Weight reads (unavoidable, MXFP4 compressed)
- KV cache reads/writes (unavoidable)
- Residual stream (2 writes per layer: after O-proj and after Down-proj)

## Comparison with Prior Work

| | CODA (Dao et al.) | MPK (Mirage) | This Work |
|---|---|---|---|
| Fusion level | Register (GEMM epilogue) | LDS (persistent kernel) | Both |
| Kernel launches | ~6-8 per layer | 1 for entire model | 1 for entire model |
| Intermediate storage | Registers only | LDS only | Registers + LDS |
| Multi-die awareness | No (NVIDIA single-die) | Partial (runtime dispatch) | Full (XCD-aware barriers, weight partitioning) |
| Model support | Dense only | Dense + MoE | Dense + MoE (from same library) |
| Automation | Manual epilogue authoring | Compiler + superoptimizer | Compiler selects epilogues + phases |
| Hardware | NVIDIA Hopper | NVIDIA + AMD | AMD MI300/MI350 (multi-XCD) |

## ASPLOS Paper Outline

**Title:** Two-Level Fusion: Compiling Persistent Kernels from Composable Device
Functions on Multi-Die GPUs

**Contributions:**
1. A device function library with composable GEMM epilogues that cover all
   non-attention transformer operations
2. A two-level fusion framework: register-level epilogue fusion (CODA-style) +
   LDS-level persistent kernel fusion, with a compiler that selects both
3. XCD-aware compilation: automatic barrier selection, weight partitioning, and
   work distribution for multi-die AMD GPUs
4. Evaluation on GPT-OSS 120B (MoE) and Qwen3-8B (dense) showing the same
   library achieves competitive TPOT on both architectures

**Evaluation:**
- GPT-OSS 120B: 2.22ms TPOT (matches hand-tuned mirage at 2.18ms)
- Qwen3-8B: target competitive with mirage/vLLM baselines
- Ablation: Level 1 only vs Level 2 only vs both
- Compilation time: seconds (vs hours for mirage superoptimizer)
- Lines of model-specific code: ~0 (config YAML only)
