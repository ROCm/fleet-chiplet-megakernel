# Titan vs Mirage: Performance Gap Analysis (GPT-OSS 120B)

## Measured Performance

| Metric | Titan | Mirage | Ratio |
|--------|-------|--------|-------|
| Decode avg (short seq) | 8.08 ms/tok | 2.19 ms/tok | 3.7x |
| Decode min | 8.01 ms/tok | 2.14 ms/tok | 3.7x |
| Per-layer budget | 217.9 us | ~60.8 us | 3.6x |

## Titan Per-Layer Subphase Breakdown

```
Phase                      Time (us)  % of Layer
RMSNorm                      3.8        1.7%
QKV GEMM + barrier          17.2        7.9%
RoPE + KV update             6.9        3.2%
Attention (CK FMHA hd64)   60.5       27.8%
Attention barrier             3.3        1.5%
OProj GEMM                  12.4        5.7%
FFN mid barrier               3.1        1.4%
Router GEMM                  14.5        6.7%
TopK + barrier               12.1        5.6%
MoE FFN (W13+W2)            80.3       36.9%
Layer barrier                 3.7        1.7%
TOTAL                       217.9 us/layer
TOTAL x 36 layers             7.85 ms
```

## Gap Breakdown

### 1. Global barriers (~20 us/layer, 0.72 ms total)

Titan uses 5 global barriers per layer (QKV, attention, FFN mid, MoE, layer end)
at ~3.3 us each = ~16.5 us/layer of pure synchronization overhead, plus partial
barrier cost baked into TopK (12.1 us includes barrier wait after router GEMM).

Mirage uses zero global barriers. Its task-based scheduler dispatches all 240
workers with dependency DAGs -- workers proceed to the next task as soon as their
specific dependency is satisfied, no global sync required.

### 2. Sequential MoE vs 2TG overlap (~38 us/layer, 1.38 ms total)

Titan runs W13 and W2 **sequentially** with a global barrier between them:
- W13 loop: ~32 us (4 experts x 46 tiles, all 240 workers)
- Global barrier: ~3 us
- W2 loop: ~35 us (4 experts x 46 tiles, all 240 workers)
- Residual add: ~10 us

Mirage uses 2 threadgroups (2TG) per workgroup: waves 0-1 execute W13 while
waves 2-3 execute W2 **simultaneously** via LDS double-buffering. As soon as
W13 finishes for expert A, W2 starts for expert A while W13 begins expert B.
Per-expert barriers (not global) coordinate the handoff.

Estimated mirage MoE time: ~42 us/layer vs Titan's 80.3 us/layer.

### 3. Cross-phase overlap (~30 us/layer, 1.08 ms total)

Mirage fuses OProj + RMSNorm + Router + TopK into a single "OPROJ_TOPK" gang
task (`-DMPK_FUSED_LAYER_BATCHING`). While some workers finish OProj, others
start Router/TopK without waiting. Additionally, workers that finish MoE for
layer N can start QKV for layer N+1 immediately -- no layer boundary barrier.

Titan has explicit barriers between every phase and every layer, preventing any
cross-phase or cross-layer overlap.

### 4. Attention (~40 us/layer, 1.44 ms total)

Titan: 60.5 us/layer using CK FMHA template (head_dim=64).
Mirage: ~20 us/layer using hand-tuned split-KV attention with XCD affinity.

With 64 Q heads x 8 KV chunks = 512 attention tasks in mirage, each taking
~3.8K cycles and overlapped across workers. The split-KV decomposition also
enables better L2 locality (each XCD reads its assigned KV chunk).

### 5. Gang GEMM fusion (~8 us/layer, 0.29 ms total)

Mirage's "gang" kernels fuse adjacent operations into single tasks, eliminating
HBM roundtrips between phases:

- **RMSNorm + QKV GEMM**: All workers redundantly compute RMSNorm in registers
  (microseconds), then cooperatively dispatch MFMA tiles. Titan runs RMSNorm as
  a separate phase (3.8 us) + barrier before QKV GEMM.
- **OProj GEMM + ResAdd + RMSNorm + Router GEMM + TopK**: Five operations fused
  into one gang task. Titan runs them as 3 separate phases with 2 barriers
  between them (OProj 12.4 + barrier 3.1 + Router 14.5 + TopK 12.1 = 42.1 us).
  Mirage eliminates the inter-phase barriers and HBM writes/reads.

### 6. LDS weight prefetch for W13 (~5 us/layer, 0.18 ms total)

Mirage's `-DMPK_W13_LDS_PREFETCH` prefetches W13 expert weights into LDS during
the FP8 activation quantization phase, so weights are ready in LDS (~8 cycle
latency) when MFMA starts, instead of waiting for HBM (~400 cycle latency).

Implementation details:
- **Phase A** (during quant): 24 concurrent `buffer_load_dwordx4 lds` inline asm
  instructions. The compiler would serialize these behind `s_waitcnt vmcnt(0)`;
  inline asm avoids that (840ns serialized vs ~75ns concurrent).
- **Phase B** (scale loads): Scale data drains while HBM weight data flies.
- **Phase C** (MFMA from LDS): Weights read from LDS at ~8 cycles instead of
  HBM at ~400 cycles. For multi-expert routing, this compounds across topk_slot
  iterations.

W13 per-tile latency drops from ~1.5 us to ~0.5 us with LDS prefetch.

### 7. Depth-4 MFMA pipeline tuning (~3 us/layer, 0.11 ms total)

Both Titan and mirage use depth-4 MFMA pipelines, but mirage's gang kernels
pipeline more aggressively:

- **Depth-2 pipelined FP8 loop**: Next iteration's weight/token data prefetched
  into registers during current MFMA execution. Per-iteration overhead: 36
  cycles vs 53 cycles without prefetch (saving 17 x 24 = 408 cycles for K=3072).
- **In-loop dequantization**: FP4 weights dequantized per K-tile during MFMA
  latency overlap, never as a separate blocking step.

## Summary Table

| Source | Titan (us/layer) | Mirage est. (us/layer) | Delta (us) | Delta total (ms) |
|--------|-------------------|------------------------|------------|------------------|
| Barriers (5/layer) | 16.5 | ~0 | 16.5 | 0.59 |
| MoE sequential vs 2TG | 80.3 | ~42 | 38.3 | 1.38 |
| Cross-phase overlap | -- | ~30 saved | 30 | 1.08 |
| Attention | 60.5 | ~20 | 40.5 | 1.46 |
| Gang GEMM fusion | ~8 overhead | ~0 | 8 | 0.29 |
| LDS W13 prefetch | -- | ~5 saved | 5 | 0.18 |
| MFMA pipeline tuning | -- | ~3 saved | 3 | 0.11 |
| **Total** | | | **141.3** | **5.09** |

Predicted Titan after all optimizations: 8.08 - 5.09 = **~3.0 ms** (vs mirage 2.19 ms).
Remaining ~0.8 ms gap from additional mirage optimizations (weight layout, XCD
affinity for linear tasks, reduced register pressure from hand-tuned asm, etc.).

## Priority Order for Optimization

1. **2TG MoE overlap** (1.38 ms) -- biggest single win
2. **Attention optimization** (1.46 ms) -- switch to split-KV or tune CK FMHA
3. **Remove global barriers** (0.59 ms) -- switch to dependency-based events
4. **Fuse OProj+Router+TopK** (1.08 ms) -- eliminate mid-phase barriers
5. **LDS weight prefetch** (0.18 ms) -- prefetch W13 weights during quant
6. **MFMA pipeline tuning** (0.11 ms) -- match mirage's depth-2 FP8 loop
