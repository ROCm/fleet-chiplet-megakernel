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

## Prime suspect (unproven)

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
