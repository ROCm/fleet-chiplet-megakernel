# B>1 row symmetry: identical prompts, different continuations

Status: **open defect**, root cause narrowed but not proven. No code change yet.

Applies to the fused megakernel decode path at
`--max-num-batched-requests > 1` (commit `5680625`).

## Symptom

Feed the *same* prompt into every batch slot with greedy argmax and the slots
do not all produce the same continuation:

```
[B=8 rep1] rows=8 distinct=2
  row5: 'analysisWe need to answer: "What is'
  rest: 'analysisThe user asks: "What is the'
```

Measured on MI350, gpt-oss-120b, seq 512, 20 new tokens, GPU 2:

| config | result |
|---|---|
| B=8, 6 reps | **diverged 5/6** |
| B=1, ~25 runs (1/3/8 chunks, 1/16 batched tokens, 8 rotated requests) | **0 divergences** |

The odd row **shuffles** between reps -- row5, then rows2+7, then row1, then
rows7+8. A fixed per-request addressing bug cannot shuffle, so this is
numerical, not an indexing defect.

Every diverged row still answers *its own* prompt correctly. Prompt-to-answer
pairing holds under permutation and under rotation. The defect is
reproducibility, not correctness of content.

## Why this is a defect and not baseline noise

vLLM on the same box and model, 8 identical prompts in one batch, prefix
caching off: **0/28** pairs differ at token 0. MPK: **7/28**.

Two properties that must not be conflated:

| | vLLM | MPK |
|---|---|---|
| identical rows in one batch -> identical output (**row symmetry**) | holds 8/8 | **broken** |
| batched output == solo output (**batch invariance**) | broken 3/8 | broken |

Batch invariance is broken in vLLM too -- that is what
`VLLM_BATCH_INVARIANT` exists for, and it is off by default. Row symmetry is
not broken there. Only the row-symmetry gap is ours.

Both confounds were ruled out before drawing that conclusion:
prefix caching was off (`enable_prefix_caching = False`), and the requests were
genuinely co-batched (with *distinct* prompts, rows 1/3/6 differ from their own
solo runs, which is impossible if the scheduler had serialized them).

## What vLLM does differently

Every reduction in the vLLM/aiter decode path stays *within* one row.
`aiter/paged_attn.py`:

```python
max_num_partitions = (max_seq_len + _PARTITION_SIZE_ROCM - 1) // _PARTITION_SIZE_ROCM
tmp_output = torch.empty((num_seqs, num_heads, max_num_partitions, head_size), ...)
```

The partition count derives from `max_seq_len` alone, never from `num_seqs`,
and each sequence reduces over its own private partitions. Row symmetry holds
by construction. (`num_seqs` appears only in the V1-vs-V2 occupancy heuristic,
not in the partition count.)

Where vLLM *does* hit split-reduction batch dependence, its own remedy is to
turn the split off -- `vllm/v1/attention/backends/flash_attn.py:311`:

```python
if vllm_is_batch_invariant():
    max_num_splits = 1
```

MPK's MoE instead has all activated experts `atomicAdd` into a *shared* f32
workspace, so one row's result depends on the arrival order of work belonging
to an accumulator it shares with other rows.

## Ruled out (do not re-run these)

- **Slot indexing.** The odd row shuffles between reps.
- **The batch-dependent attention split.** `NUM_KV_CHUNKS = 30/B` clamped by
  `max(8, kv_tiles//2)` yields **8 chunks at both B=1 and B=2** at seq 512
  (only B=4->7 and B=8->3 differ). B=2 diverges, B=1 does not, with a
  bit-identical reduction tree. The aiter contrast above is the right analogy
  but the wrong diagnosis.
- **`ATTN_PARTICIPANTS` boundary arithmetic.** Reproduces at B=2 (= 2*8),
  nowhere near a boundary.
- **The FP8 activation quantizer.** `_gang_multirow_fp8_quant_impl` takes amax
  per row, and `static_assert(NSUB % 4 == 0)` keeps 128-element super-blocks
  from straddling a row boundary, so row `t`'s scale cannot depend on its
  tile-mates.
- **gfx950 scaled-MFMA hazards.** Both the operand-overwrite hazard and the
  9-clock accumulator read were already repaired in `b0adda9`: all four
  pipelined loops (QKV, W13 T0, W13 T1, W2 T0) use the two-bank ping-pong
  (`v[22:25]/v7/v[8:15]/v16` vs `v[26:29]/v18/v[32:39]/v19`, `v17` scratch) and
  `s_nop 15` x2 = 32 clocks before `v_accvgpr_read_b32`. No live `s_nop 7`
  remains; the four matches are comments recording the old form. Mixed-iteration
  operands would have been an excellent explanation for row asymmetry, so this
  mattered to check. Note the read-only `~/mirage` reference still carries the
  *unfixed* code.

## The MoE atomicAdd was fixed and did NOT fix this (2026-08-19)

**Read this before re-reading the suspect section below.** The fix described
there was implemented and shipped in `826039e` ("MoE: row-private f32
workspace (per-topk-slot), no shared accumulator"), which is an ancestor of
HEAD (`e0d8000`). The shared accumulator is gone: the W2 epilogue now does
plain write-through stores into per-`(token, topk_slot)` slabs
(`gang_moe_fused_mxfp4_mi300.cuh:3107`), and the consumer folds the 4 slots in
fixed slot order (`gang_rmsnorm_linear_mxfp4_bias_mi300.cuh:1911`). See
`moe_ws_layout.cuh` for the full rationale.

**Row symmetry is still broken at HEAD.** Measured 2026-08-19, ctx 512,
`CK_FMHA_1TOK=1` (a flag since removed -- fair-share prefill admission gives
the same concurrent-decode schedule by default, and faster), 3 requests all
given "The capital of France is", 32 new tokens, HIP device 1:

```
rep1 row0: '...The user asks: ... Paris. So respond: "Paris".'
rep1 row1: '...The user says: ...'          <- diverges at token 3
rep1 row2: '...The user asks: ... Paris. So respond with "Paris".'
rep2 row0: '...Paris. The user wrote a statement, maybe they want'
rep2 row1: '...Paris. The user gave an incomplete statement. Probably they'
rep2 row2: '...Paris. They might be expecting a short answer. The'
bs=1 ref : '...Paris. Provide concise answer.'
```

Three failures: rows disagree within a batch, the two reps disagree, and no
row reproduces the bs=1 reference. Divergence is at token 3, not in the tail.

**So the prime suspect below is CLOSED and the cause is elsewhere.** What has
been eliminated since:

- **Attention FP reordering.** `num_kv_chunks` is **8 at both bs=1 and bs=3**
  (demo.py:1371 clamps so "B<=3 (budget >= 10) bit-identical to the
  single-request path"), confirmed in the run logs. Identical reduction order,
  so it cannot be the source.
- **The split-KV merge worker's identity.** The merge is run by the last chunk
  worker to arrive, whose identity does vary -- but `merge_splitkv.cuh:302-316`
  loops `kv_idx = 0..num_chunks` in fixed order, so *who* runs it does not
  change the summation order. Not the cause.
- **Float atomics on data.** A repo-wide sweep of `tasks/mi300/*.cuh` finds no
  float `atomicAdd` left on the fused decode path; the remaining ones are
  integer barrier/profiling counters, plus `gang_ksplit_linear_mi300.cuh` and
  the `linear_ck_mi300.cuh` split-K epilogue, neither of which is on it.

## Localized to a depth-dependent race (2026-08-19, at 02b81d0)

Use the in-tree sweep, not hand-diffed continuations. `demo.py --verify` with
identical prompts prints `max|row_i - row_0|` for every captured tensor; the
first nonzero one in dataflow order is the defect. Two bugs made it unusable
until now and are fixed:

- it sat **inside** the `not MPK_FP_ONLY` guard, so it only ran together with
  the Torch reference pass -- which OOMs at full `--max-layers`, exactly the
  depth the defect needs. It compares rows to each other, never to Torch, so
  it now runs independently.
- it swept **every** tensor including `moe_mask` (129 rows), `moe_routing_
  indices` (128), the KV caches (pages) and the barrier arrays. Those are not
  batch-indexed, so "row 0 vs row 1" compared expert 0 to expert 1 and always
  reported DIFFERS. Now filtered to `shape[0] == bs`.

Run it prefill-only so there is no legitimate divergence at all --
`--max-new-tokens 1` with fair-share prefill (see 02b81d0) makes every
iteration feed all rows the same token, and `embed_out` comes out exactly 0,
which proves the precondition rather than assuming it. This is what the
"end-of-run buffer comparison is meaningless" trap below was about; that trap
no longer applies to this configuration.

Result, ctx 512, 3 identical prompts, HIP device 1, batch-row tensors only:

| `--max-layers` | tensors with row diff > 0 |
|---|---|
| 1 | **0** |
| 2 | **0** |
| 4 | 11 |
| 8 | 12 |
| 16 | 12 |
| 36 | 12 |

**Layers 1 and 2 are perfectly row-symmetric over 71 iterations** -- bit-zero
on all 19 batch-row tensors. So the per-iteration ops (QKV, attention, merge,
router, MoE, o_proj) are all row-symmetric in isolation at bs=3. The defect
needs depth.

**It is a race, not deterministic batch-composition dependence.** Two
independent depth-36 runs of the same build:

| | `attn_out` worst row | max diff | row norms |
|---|---|---|---|
| run A | row 2 | 11.03 | 185.9 / 191.7 / 191.6 |
| run B | row 1 | 10.94 | 197.9 / 217.2 / 211.5 |

Different row, different magnitude, and even row 0's norm moves. Anything
deterministic in batch composition would reproduce exactly. **This eliminates
the two leading suspects from the previous section** -- the activated-expert
set from `max_activated = min(num_topk * batch_size, num_experts)`, and the
W13/W2 GEMM M-packing. Both are pure functions of the routing, so both would
be bit-reproducible run to run.

At depth 4 the divergence is ~1 ULP (`moe_topk_weight` 8.9e-08 in fp32;
`rmsnorm_out` 1.0 in bf16 at norm 574), growing with depth. A tiny FP
perturbation amplified by the residual stream, not a wrong value.

Prime suspect, and it is documented in the code as accepted: **the MoE mask
has one buffer for all 36 layers and no layer-boundary barrier**, so "a worker
still finishing layer L can read layer L+1's mask"
(`gang_moe_fused_mxfp4_mi300.cuh`, the "Nothing read from the mask may steer
control flow" note). The code deliberately keeps that safe for *partitioning*
-- the tile space is compile-time so no worker disagrees about structure --
but still reads `expert_id` from it, and calls the consequence "the benign
numeric drift the mask always had (a token gets a neighbouring layer's
expert)". That drift is benign for deadlock and for a single row. It is
exactly a row-asymmetry source, it is a race, and it needs >= 2 layers to
straddle -- which matches every measurement above. Not yet proven; the test is
to order the mask at the layer boundary (or double-buffer it by layer parity)
and re-run the depth table.

Also still open: the FP8 activation quantizer under `USE_FP8_ACT=1`
(previously ruled out on the per-row-amax argument).

## The same defect shows up with DISTINCT prompts (2026-08-19, at b97ac60)

Row symmetry needs identical prompts to observe, which made it look like a
narrow curiosity. It is not. Run the same build twice with *distinct* prompts
and the continuations differ run to run:

Prompts: "The capital of France is" / "2 + 2 =" / "The author of Hamlet is" /
"The chemical symbol for gold is". ctx 512, 96 new tokens, HIP device 1,
`--prompts` (one per request), two reps of each arm.

| bs | rep1 vs rep2 |
|---|---|
| 1 | **identical** |
| 2 | differs |
| 3 | differs |
| 4 | differs |

bs=1 is the control and is bit-deterministic, as always. This is a strictly
stronger statement of the same underlying race: it needs no identical-prompt
setup, so it is reproducible in any ordinary multi-request run.

**It is not caused by fair-share prefill admission (02b81d0).** The greedy
ablation `MPK_NO_FAIR_PREFILL=1`, which restores the pre-02b81d0 admission
rule, is equally nondeterministic at bs=2. Admission changes *which* schedule
runs; the race is in the schedule-independent path.

Semantics are unaffected in every run measured -- each row answers its own
prompt (Paris / 4 / William Shakespeare / Au) and only the phrasing of the
sampled continuation moves. That matches the "reproducibility, not
correctness" framing at the top of this doc. It does mean **a text hash is
not a usable correctness gate at bs>1**; check per-row semantics instead. See
also [[bs-gt-1-accuracy-gating]].

## Prime suspect (SUPERSEDED -- fixed in 826039e, did not resolve the defect)

The MoE W2 epilogue, `gang_moe_fused_mxfp4_mi300.cuh:2085`:

```c
int ws_base = my_tok * HIDDEN_SIZE + out_n_base;
atomicAdd(&d_workspace_f32[ws_base + 0], (acc[0] + bv0) * pf_rw);
```

Experts E1 and E2 hit *different addresses* for row 0 vs row 1, so the hardware
may serve E1-then-E2 for one row and E2-then-E1 for the other. Float add is not
associative, so the rows drift independently. B=1 has one row and structurally
cannot show it. It is the only float `atomicAdd` on data in the fused decode
path (the split-K variants are not on it).

The fix is a deterministic reduction: a private output slab per expert slot,
summed by a designated worker in fixed slot order. That costs extra workspace
traffic plus a reduction pass. Implementing it is simultaneously the proof and
the fix -- if row symmetry does not return, the suspect was wrong.

## Traps that cost hours

- **`MPK_MOE_SINGLE_EXPERT=1` is void as a discriminator.** It was meant to make
  accumulation order irrelevant (one add per element), but it produces 8
  distinct outputs at **B=1 sequential** too, so the regime is chaotic
  per-request and its B=8 result carries no information.
- **`demo.py` only prints `[cont  ]` when `total_num_requests > 1`**, so a naive
  B=1 arm emits zero lines and looks like a launch failure. Use
  `--num-requests 8 --max-num-batched-requests 1` for a parseable sequential
  control.
- **End-of-run buffer comparison is meaningless.** Requests hit EOS at different
  steps (`step=[118,144,128,...]`), so `argmax_part_*` rows hold partials for
  different tokens. `--max-new-tokens` does *not* cap this: the megakernel is
  persistent and each request runs to its own EOS.
- **`CK_FMHA_NUM_KV_CHUNKS=1` is garbage at B=1 too** -- a pre-existing breakage
  of the non-split path, useless as a control.
- **Divergence is stochastic; 3 reps is not enough.** An early 3/3-clean control
  was luck against a 5/6 rate, and an early "divergence tracks chunk count"
  correlation dissolved at 3 reps per config.

## Current gate

Until this is fixed:

- Serving (independent requests, sampling anyway): B>1 is fine.
- Reproducibility-sensitive work -- eval harnesses, regression baselines,
  bitwise A/B: use B=1.
- PPL mode already asserts `B == 1`, which is required for a different reason:
  `task_register.cc:1242` emits `runtime_config.step[0] + 1` as the logits sink
  row, so at B>1 every request would write the same row.
