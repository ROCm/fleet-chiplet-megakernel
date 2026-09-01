# GPT-OSS gfx950 persistent-megakernel reference (fleet)

Use this reference for the GPT-OSS 120B batch-1 decode megakernel on CDNA4
(MI355X, gfx950). Treat commands as repository-current templates; check
`demo/gpt_oss/README.md` and `demo/gpt_oss/env_common.sh` before running.

## The one thing that changes the loop

**The megakernel is JIT-compiled from a generated translation unit at run
time.** Editing a `.cuh` under `include/mirage/persistent_kernel/tasks/` does
not require rebuilding `core.so` -- the next run picks it up. That makes the
inner loop fast, and it also means:

- Nothing in the tree type-checks a `.cuh` edit until a run that needs the
  weights. Use the codegen gate below instead of discovering a malformed
  inline-asm operand list twenty minutes in.
- Editing any `.cuh` or `.py` while a background A/B run is in flight poisons
  every remaining run in that batch. Finish the batch, then edit.

## Source map

- Graph construction, task types, and the demo runner: `demo/gpt_oss/demo.py`
- Compile flags, every `MPK_*` switch, and the measured note next to each:
  `python/mirage/mpk/persistent_kernel.py` (`get_compile_command()`)
- Flag defaulting helper and the batch-1-only list:
  `python/mirage/utils.py` (`mpk_opt`, `_BS1_ONLY_OPTS`)
- Environment-variable forwarding allowlist (a flag not listed here is
  silently dropped): `demo/gpt_oss/env_common.sh` (`MPK_FORWARD_VARS`)
- Persistent scheduler, worker loop, task dispatch:
  `include/mirage/persistent_kernel/persistent_kernel.cuh`
- Fused full-layer scheduling:
  `tasks/mi300/gang_full_layer_fused_mi300.cuh`
- Fused layer with LM head:
  `tasks/mi300/gang_full_layer_with_lmhead_fused_mi300.cuh`
- QKV projection + RMSNorm:
  `tasks/mi300/gang_rmsnorm_linear_mxfp4_bias_mi300.cuh`
- Attention and split-KV merge: `tasks/mi300/gang_attention_mi300.cuh`,
  `tasks/mi300/multitoken_paged_attention_*_mi300.cuh`
- O-proj + residual + RMSNorm + router/TopK:
  `tasks/mi300/gang_linear_mxfp4_res_bias_rmsnorm_topk_mi300.cuh`
- TopK softmax: `tasks/mi300/moe_topk_softmax_mi300.cuh`
- W13 / SwiGLU / W2 MoE pipeline (the largest single file, and where most of
  the ISA-level work lives): `tasks/mi300/gang_moe_fused_mxfp4_mi300.cuh`
- LM head + argmax:
  `tasks/mi300/gang_rmsnorm_linear_mxfp4_bias_argmax_mi300.cuh`
- Embedding (shared with the ampere path):
  `tasks/ampere/embedding.cuh`, included via `tasks/mi300/task_header.cuh`

## Live topology (assume these, verify before relying on them)

```text
workers_per_xcd     31          XCDs                8
NUM_KV_CHUNKS       8           NUM_REQS            1 (batch-1 decode)
hidden 2880 -> PADDED_HIDDEN_SIZE 2944 = 23 * 128
W13_MFMA_ITERS == W2_MFMA_ITERS == 23   (odd: ping-pong precondition;
                                         23 % 4 == 3: quad-chain precondition)
padded vocab        201216
```

Several schedules `static_assert` on `MFMA_ITERS` being odd or `% 4 == 3`. A
shape change that breaks those fails the build rather than silently falling
back -- that is intended, do not soften it.

## Flag conventions

Two distinct idioms in `get_compile_command()`, and the difference is load
bearing:

```python
if _opt("MPK_FOO"):                                    # default ON
    ...                                                # MPK_FOO=0 disables
if int(os.environ.get("MPK_FOO", "0")) == 1:           # default OFF, opt-in
```

`_opt(name)` is `mpk_opt(name, mpk.max_num_batched_tokens)`. Flags in
`_BS1_ONLY_OPTS` have their *default* narrowed to batch 1; setting one
explicitly at bs>1 still reaches the `static_assert`, which is the intended
loud failure rather than a knob that quietly does nothing.

Every flag carries a comment recording what it measured and whether it is on.
Keep that discipline: a flag with no number next to it is an unfalsifiable
claim.

## Run the model

```bash
MODEL_PATH=/root/schowdha/models/gpt-oss-120b \
HIP_VISIBLE_DEVICES=<idle-gpu> \
MAX_NEW_TOKENS=64 \
  bash demo/gpt_oss/run_1gpu.sh
```

Pin one explicitly-chosen idle GPU. Confirm it is idle before measuring.

## The codegen gate -- run this before every GPU test

Drives `hipcc` over a one-instantiation TU all the way to gfx950 object code,
so the C++ front end *and* the assembler see the edit. Its define set comes
from `get_compile_command()`, so it cannot drift from the JIT build. Seconds,
no weights, no GPU:

```bash
USE_FP8_ACT=1 bash tests/standalone/compile_check_moe_fused.sh
USE_FP8_ACT=1 bash tests/standalone/compile_check_moe_fused.sh \
  -DMPK_MOE_DUAL_ACCUMULATOR
```

`USE_FP8_ACT=1` is required -- exactly one of `USE_FP4_ACT` / `USE_FP8_ACT` /
`USE_FP16_ACT` must be set or the define generation raises, and a `$(...)`
failure will escape `set -eu` and hand you an empty `$DEFINES` whose
disassembly means nothing. If a build reports zero MFMA, suspect this first.

## Read the final assembly

The JIT output is a HIP fat binary, so a plain `-c` object gives you host code
and `llvm-objdump` will report zero MFMA. Go through `--genco` and unbundle:

```bash
hipcc -x hip --genco --offload-arch=gfx950 -O2 $DEFINES $INCLUDES \
  tests/standalone/compile_check_moe_fused.hip -o /tmp/mk.co
/opt/rocm/llvm/bin/clang-offload-bundler --type=o \
  --targets=hipv4-amdgcn-amd-amdhsa--gfx950 --unbundle \
  --input=/tmp/mk.co --output=/tmp/mk.gfx950.o
/opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx950 /tmp/mk.gfx950.o > /tmp/mk.s
```

Useful property checks on the result -- these are what turn a plausible story
into a measured one:

```bash
grep -c 'v_mfma_scale_f32_16x16x128_f8f6f4' /tmp/mk.s          # count unchanged?
grep -o 'v_mfma[^ ]* a\[[0-9]*:' /tmp/mk.s | sort | uniq -c    # which chains?
grep -c 'v_accvgpr_write' /tmp/mk.s                            # init count
awk '/v_mfma/{if(p)print n; p=1; n=0} /s_nop/{if(p)n++}' /tmp/mk.s | sort | uniq -c
```

That last one -- the `s_nop` histogram between consecutive MFMAs -- is the
single most useful check when evaluating an accumulator-chaining or scheduling
change. If the gaps are already zero, there is no padding for the transform to
reclaim and it will cost you.

Do not optimize instruction count in isolation. Identify the dependency chain
or hardware resource putting those instructions on the critical path.

## Localize the cost

Device-side phase timers, opt-in:

```text
MPK_OPROJ_INNER_TIMING=1     O-proj sub-phases
MPK_MOE_INNER_TIMING=1       MoE sub-phases (tile 0 by default; see the
                             widening note in get_compile_command)
```

Timing code perturbs register pressure -- recheck resource allocation and
compare against the uninstrumented build before trusting a delta. Use
`phase-profiling` to localize a stage and `pc-sampling` when phase timing
identifies *where* but not *why*.

## Correctness gates

**Token-exactness (run every time).** The decode is deterministic, so any
change that is not supposed to move a bit must reproduce the text hash:

```bash
awk '/^systemYou are ChatGPT/,/^\[WALL\]/' <log> | grep -v '^\[WALL\]' | md5sum
```

Record the reference hash from a clean baseline run at the start of a session
and compare every run against it. Bit-identical text is necessary, not
sufficient -- 64 tokens is a short window.

**Perplexity (required when the change reassociates FP arithmetic).** Any
accumulator chaining, reduction reordering, or fused dot product moves results
by a few ULP, so the text hash will legitimately differ and cannot be the
gate:

```bash
PPL_MODE=1 PPL_MXFP4_MATCH=1 MODEL_PATH=... bash demo/gpt_oss/run_1gpu.sh
```

`PPL_MXFP4_MATCH=1` scores the Torch reference at MPK's weight precision, so
the comparison isolates your change instead of the quantization.

**Schedule unit tests.**

```bash
hipcc -O2 --offload-arch=gfx950 tests/standalone/test_mfma_pipeline_hazards.hip \
  -o /tmp/t_haz && /tmp/t_haz
hipcc -O2 --offload-arch=gfx950 tests/standalone/test_topk_local_max3.hip \
  -o /tmp/t_max3 && /tmp/t_max3
```

`test_mfma_pipeline_hazards.hip` validates the MFMA schedules bit-exact against
order-matched references, including the multi-chain arms and the SrcC
forwarding spacing they rely on. **If you add a new schedule arm, extend this
test -- it does not automatically cover arms it does not know about.**
`test_topk_local_max3.hip` proves the max3 argmax tie-break is bit-exact
against the serial scan.

**Correctness suite.** `bash demo/gpt_oss/run_correctness_suite.sh`

**Long output.** A 64-token smoke does not exercise KV growth, page remaps, or
chunk-boundary addressing. For anything touching attention addressing, KV
paging, or split-KV merge, raise `MAX_NEW_TOKENS` well past the page and chunk
boundaries and sweep context lengths.

## Measure

Alternate control and variant, at least three pairs, per-run median of the
`time_ms=` samples, hash every run:

```text
c1 1.8430   v1 1.8465
c2 1.8425   v2 1.8440
c3 1.8420   v3 1.8455
```

Keep the change only when it wins every pair, or when it wins most pairs and
you can say why the loss is structural. Machine drift between the first and
last run of a batch is larger than most individual wins here -- a single
favorable sample is not evidence. Report the pair table, not just the best
number.

For a flag that measures neutral or negative, do not delete the work: leave it
behind an opt-in `os.environ.get(name, "0")` gate with the measured numbers in
the comment. A recorded negative result is worth more than a deleted branch.

## Protocol audit for risky changes

Before changing a readiness handoff, write down:

1. Payload producer and every cache level that may retain the write.
2. Producer drain/fence before readiness publication.
3. Scope and cache policy of the readiness operation.
4. Consumer polling/acquire operation.
5. Consumer invalidation and first payload read.
6. Whether producer and consumer are wave-, workgroup-, XCD-, or GPU-local.

Before changing load scheduling or waits, write down:

1. Exact final-assembly request issue order.
2. Which requests the next consumer needs complete.
3. Which younger requests may remain outstanding.
4. What prevents LLVM from changing that order.
5. Cache flags and guard validity for every moved request.
6. New live ranges and occupancy impact.

`vmcnt` is a **per-wave** counter tied to final emitted request order, not to
the source-level load count. A tid-major work split that gives different waves
different request counts makes a single counted wait wrong for some of them --
re-cut the split wave-uniform instead of emitting several different counts.
Re-audit every counted wait after any compiler, ROCm, or surrounding-source
change.

Reject a change when this proof is ambiguous, even if a short smoke passes.

## Non-negotiable hazards

- MFMA input reuse and MFMA-to-AccVGPR-read spacing (fleet pads to 32 states
  where the measured gfx950 retirement threshold is 11).
- M0 update spacing before direct-to-LDS MUBUF operations.
- Workgroup barriers around global-to-LDS production and LDS consumption.
- Producer payload drain, release publication, consumer acquire/invalidation,
  and payload read ordering.
- System- or device-scope operations for cross-XCD handoffs.
- Exact cache policy on persistent handoff payloads.

A `"memory"` clobber limits LLVM motion; it is not a hardware cache flush.
Removing a compiler barrier is safe only when SSA, aliasing, control flow, and
hardware ordering all remain correct.

## Environment note

This tree builds with AMD clang 20 on ROCm 7.0.0. The QKV LDS-norm path is
known to miscompile on Clang 23 (intermittent invalid decode tokens when a
batched quantum drains to concurrency one). Re-validate the inline-asm-heavy
flags before trusting them on a newer compiler.
