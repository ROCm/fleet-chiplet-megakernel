---
name: optimize-megakernel
description: >-
  Run an evidence-driven C++/HIP optimization loop for fleet's fused GPU
  megakernels, especially the GPT-OSS gfx950 persistent decoder. Use when
  profiling a megakernel, reading final AMDGPU assembly, changing worker
  partitioning, barriers, cache protocols, load scheduling, prefetching,
  attention, or MoE pipelines, validating native numerical and lifecycle
  correctness, measuring TPOT, or deciding whether to keep or revert a kernel
  optimization.
---

# Optimize a Megakernel

Treat optimization as a controlled experiment. Locate a measured bottleneck,
predict a concrete ISA change, modify one variable, and keep the change only
when final assembly, native correctness, and repeatable performance all agree.

For the GPT-OSS gfx950 persistent decoder, read
[references/gpt-oss-gfx950.md](references/gpt-oss-gfx950.md) before building or
running. It contains the current source map, native gates, and production smoke
tests.

## Preserve the experimental boundary

- Work in an isolated container and a clean worktree as required by the
  repository `AGENTS.md`.
- Pin the effort to one explicit GPU. Check that it is idle before measuring.
- Keep the inner loop in C++/HIP. Enter Python/framework code only to verify
  production integration after a native win, then always run the mandatory
  production smoke and A/B gate below before calling the candidate a winner,
  unless the task explicitly targets framework overhead.
- Change one optimization variable per checkpoint. Do not combine a worker-map
  change, cache-policy change, and wait change in one experiment.
- Preserve unrelated user changes. Revert only the experiment being rejected.
- Maintain an experiment log outside production source paths. Record failed and
  neutral experiments as carefully as successful ones.

## Use architecture evidence

Use the matching architecture skill for ISA and hazard claims. For gfx950, use
`cdna4-expert`. Use `phase-profiling` to localize a fused stage and
`pc-sampling` when the limiting stall is unknown. Use `dpp-row-operations` when
LDS or shuffle traffic is the specific target.

Never infer hardware behavior from a source intrinsic alone. The final linked
code object is the source of truth.

## Run the iteration loop

### 1. Freeze the baseline

Record all inputs needed to reproduce the result:

- Git revision and diff state.
- ROCm image, compiler version, GPU architecture, and visible GPU ID.
- Exact model/checkpoint and dtype.
- Exact input/output lengths, context position, concurrency, cache profile,
  warm-up count, and measured sample count.
- Kernel/code-object identity and resource use: SGPRs, VGPRs, AGPRs, LDS,
  scratch, and occupancy tier.
- Median and dispersion for both the target phase and total native TPOT.

Run the native correctness gates before editing. A broken or drifting baseline
invalidates the experiment.

### 2. Localize the cost

Start from measurements, not the visual size of a source block.

1. Use device timestamps to split the fused kernel into phases.
2. Add narrower timestamps only around the dominant phase.
3. Use PC sampling or counters when phase timing identifies *where* but not
   *why*.
4. Separate work, exposed latency, synchronization, and load imbalance. A long
   phase is not necessarily doing the most arithmetic.
5. Check multiple context lengths when attention or cache addressing is in
   scope.

Keep diagnostic instrumentation out of the production binary or behind the
existing diagnostic build options. Recheck resource allocation because timing
code can perturb register pressure.

### 3. Read the final assembly

Disassemble the exact library exercised by the oracle or benchmark. Map source
stages to assembly blocks using symbols, branches, instruction signatures, and
temporary diagnostic markers when necessary.

Inspect at least:

- VMEM/SMEM issue-to-consumer distance and load width.
- `s_waitcnt` values, full drains, barriers, fences, invalidations, and sleeps.
- `sc0`, `sc1`, and `nt` cache flags.
- MFMA issue cadence, input-register reuse spacing, and accumulator reads.
- Direct-to-LDS sequences, M0 updates, LDS operations, and cross-lane traffic.
- Divergent branches, repeated address generation, scalarizable uniform work,
  duplicated stores, and serialized release fanouts.
- SGPR/VGPR/AGPR counts, scratch, LDS, and occupancy changes.

Do not optimize instruction count in isolation. Identify the dependency chain
or hardware resource that places those instructions on the critical path.

### 4. Write one falsifiable hypothesis

State the experiment before editing:

```text
Bottleneck:
Source and ISA evidence:
One proposed change:
Expected assembly delta:
Expected phase and TPOT delta:
Correctness or hazard risks:
Rejection conditions:
```

Prefer hypotheses that predict an observable assembly change. “Make attention
faster” is not falsifiable; “reuse one uniform page-table entry for both
16-token halves of a 32-token page and remove one scalar load/wait per page” is.

### 5. Make the smallest C++/HIP change

Keep layout, arithmetic, synchronization, and cache policy unchanged unless one
of those is the explicit experimental variable. Preserve compile-time switches
when they make A/B comparison reliable, but remove dead experimental branches
before production handoff.

Prefer this order of attack:

1. Remove redundant work or traffic.
2. Scalarize uniform address or metadata work.
3. Improve worker/tile balance without changing protocols.
4. Overlap already-required work across stages.
5. Move independent loads earlier and waits toward the first true consumer.
6. Insert controlled prefetches into opaque assembly regions.
7. Change cache scope, release/acquire behavior, or counted waits only with an
   explicit visibility and hazard proof.

### 6. Apply the assembly gate before GPU testing

Rebuild the exact target and diff final assembly against the baseline. Reject
or revise immediately if:

- The intended instruction transformation did not occur.
- A cache flag, fence, invalidation, wait, or barrier changed unintentionally.
- A load moved outside its valid guard.
- MFMA, accumulator, M0, or direct-to-LDS hazard spacing changed without proof.
- The compiler duplicated or narrowed a request unexpectedly.
- Scratch appeared or register/LDS growth crossed an occupancy tier.
- Code growth introduces a new hot-path branch or instruction-fetch risk.

Compiler ordering and GPU ordering are different. A `"memory"` clobber limits
LLVM motion; it is not a hardware cache flush. Conversely, removing a compiler
barrier is safe only when SSA, aliasing, control-flow, and hardware ordering all
remain correct.

### 7. Apply the native correctness gate

Run correctness before performance:

1. Run the smallest oracle that exercises the changed operation.
2. Run the persistent-decoder numerical oracle.
3. Run repeated lifecycle tests without process reinitialization.
4. Exercise boundary positions and context lengths affected by addressing or
   scheduling changes.
5. Stress cache-sensitive changes across many repeated launches.

Validate payload visibility separately from readiness visibility. A counter or
flag becoming visible does not prove the payload is visible. Preserve producer
store drains before publication and consumer acquire/invalidation before
payload reads.

Treat intermittent failure, first-token divergence, or unexplained nondeterminism
as a correctness failure. A successful launch or `backend_failure: 0` is not a
numerical oracle.

### 8. Measure the target phase and total TPOT

- Use the same binary, inputs, GPU, and measurement method as the baseline.
- Warm up before collecting samples.
- Report median plus dispersion or multiple independent medians; do not keep a
  change on one favorable sample.
- Measure the targeted phase first, then total native TPOT.
- Sweep relevant context lengths for attention changes.
- Check neighboring phases for displaced work or a new synchronization tail.
- Distinguish native kernel time from framework TPOT. Investigate framework
  overhead only after the native comparison is stable.

Keep a change only when the targeted transformation occurred, correctness is
clean, and the improvement is repeatable beyond noise without a material
regression elsewhere.

Native C++/HIP results make a candidate eligible for production testing; they
do not by themselves establish a user-visible TPOT win.

### 9. Run the end-to-end A/B gate

Fleet has no separate framework wheel: the demo runner *is* the production
path, and the megakernel is JIT-compiled, so a `.cuh` edit is live on the next
run with no rebuild. That removes the stale-library failure mode and puts the
whole burden on measurement discipline.

1. Fix the comparison matrix before running either arm: model path, GPU,
   output length, warmups, and sample count identical on both sides. For
   attention, KV-cache, paging, or position-dependent changes, include short,
   target, and long output lengths.
2. Run control and variant **alternating**, at least three pairs. Machine
   drift across a batch is larger than most individual wins here, so a block
   of three controls followed by three variants is not a valid comparison.
3. Take a per-run median of the in-run samples, then report the pair table --
   not the best number, and not a single median.
4. Hash the generated text on every run, including the controls. A change
   that is not supposed to move a bit and does has failed, whatever it timed.
5. Do not edit any `.cuh` or `.py` while a batch is in flight. The JIT picks
   the edit up mid-batch and poisons every remaining run.
6. Keep the change only when it wins every pair, or wins most pairs and you
   can explain why the loss is structural. Otherwise record the number and
   leave the code behind an opt-in flag.

### 10. Log the result and close the checkpoint

For every attempt, append:

```text
Experiment ID and revision:
Hypothesis:
Source change:
Final-assembly/resource delta:
Correctness commands and results:
Baseline and candidate phase timing:
Baseline and candidate TPOT distribution:
Control/variant pair table and output hashes:
Codegen-gate and schedule-unit-test results:
Decision: success | neutral | failure
Reason and reusable lesson:
```

For failures or neutral results, revert the code change but keep the log entry.
For successes, commit a focused checkpoint before beginning the next idea.

## Respect non-negotiable hazards

Do not weaken these contracts without an architecture-backed proof and a
dedicated experiment:

- M0 update spacing before direct-to-LDS MUBUF operations.
- MFMA input reuse and MFMA-to-AccVGPR-read spacing.
- Workgroup barriers around global-to-LDS production and LDS consumption.
- Producer payload drain, release publication, consumer acquire/invalidation,
  and payload read ordering.
- System- or device-scope operations used for cross-XCD handoffs.
- Exact cache policy on persistent handoff payloads.

Counted waits are tied to final emitted request order, not source-level load
count. Re-audit them after any compiler, ROCm, or surrounding-source change.

## Final qualification

After the loop produces a provisional winner:

1. Re-run the codegen gate and the schedule unit tests with diagnostic code
   and dead experiment switches removed.
2. Re-run the full A/B pair table on the final source, and the correctness
   suite.
3. Run the perplexity gate if the change reassociates floating-point
   arithmetic; token-exactness cannot judge a change that legitimately moves
   a few ULP.
4. Decide the default. A win ships via `_opt(...)`; a neutral or negative
   result ships behind `os.environ.get(name, "0")` with the measured numbers
   recorded in the comment beside it, and gets added to `MPK_FORWARD_VARS` in
   `demo/gpt_oss/env_common.sh` or it will be silently dropped at run time.
5. Run clang-format over the changed line ranges before committing.

Do not claim an improvement without the pair table and the matching output
hashes.
