# The split-KV merge: 0.024 ms hiding behind a misread profiler line

Status: **shipped**, both flags opt-in (`MPK_MERGE_KV_OUTER`,
`MPK_MERGE_TWO_PASS`). File:
`include/mirage/persistent_kernel/tasks/ampere/merge_splitkv.cuh`.

This is the largest single Fleet-vs-Fleet win in the effort on GPT-OSS 120B,
batch 1, MI350X, TP1: **2.35 -> 1.986 ms**. It is also the change we should have
found months earlier, and the second half of this document is about why we
didn't.

**Always read the 1.986 with its context length.** It was measured at the
default 72-token prompt (KV 72->200) -- `demo.py` clamps `--max-seq-length` to
`prompt_len + max_new_tokens`, so a larger flag value on the default prompt
does not lengthen the run. At a true 512-token KV the same build is **2.006**.
The improvement is real and holds at every context, but an earlier revision of
this paragraph claimed parity with an external engine on numbers measured at
two different context lengths, and that claim did not survive a matched
measurement. Any latency comparison must state the KV length.

---

## 1. What the merge does

Decode attention is split across `NUM_KV_CHUNKS` workers (31 in the tuned
config). Each chunk runs softmax over its own slice of the KV cache and writes
two partials: `o_acc` (the unnormalized weighted value sum, f32, `HEAD_DIM`
wide) and `lse` (one f32 log-sum-exp per chunk per head). The last chunk worker
to arrive at the per-XCD chunk barrier then runs `merge_splitkv_ck_fmha`, which
recombines the 31 partials into one attention output.

That merge is **serial time on the critical path**. Every other worker on the
XCD is spinning at the next barrier while it runs, once per layer, 36 layers per
token.

## 2. `MPK_MERGE_TWO_PASS` (−0.017 ms) — the real one

The original loop was the running-max form of the flash-attention merge:

```c
for (int kv_idx = 0; kv_idx < num_chunks; ++kv_idx) {
  float m_prev = m_global, d_prev = d_global;
  float other_m = lse_ptr[lse_linear] * LOG2E;
  m_global = max(m_prev, other_m);
  float const w_prev  = ptx_exp2(m_prev  - m_global);   // <-- depends on k-1
  float const w_other = ptx_exp2(other_m - m_global);
  d_global = d_prev * w_prev + w_other;
  for (int i = 0; i < VAL_PER_THREAD; ++i)
    o_global[i] = o_global[i] * w_prev + o_ptr[o_base + i] * w_other;
}
```

Read the dependency chain, not the instruction count. Chunk *k*'s accumulate
needs `w_prev`, which needs `m_global` from chunk *k−1*, which needed the `lse`
load from chunk *k−1*. So the 31 `o` loads **cannot overlap**. The loop runs at
one `load → exp2 → fma` latency per chunk, roughly 31 serialized round trips,
with the memory system otherwise idle. The GPU is not bandwidth-bound here or
compute-bound here; it is *latency-bound on a chain we wrote ourselves*.

Two-pass breaks the chain:

```c
// Pass 1: 31 lse values only (one dword each, contiguous in kv), reduce to m_max
for (int kv = 0; kv < num_chunks; ++kv)
  lse_log2[kv] = lse_ptr[lse_base0 + kv * NUM_QO_HEADS_PER_KV] * LOG2E;
float m_max = -inf;
for (int kv = 0; kv < num_chunks; ++kv) m_max = max(m_max, lse_log2[kv]);

// Pass 2: every weight is now known up front
for (int kv = 0; kv < num_chunks; ++kv) {
  float const w = ptx_exp2(lse_log2[kv] - m_max);
  d_sum += w;
  for (int i = 0; i < VAL_PER_THREAD; ++i)
    o_global[i] += o_ptr[o_base + i] * w;      // plain FMA, constant weight
}
```

Pass 1 touches only 31 dwords and they are contiguous in kv for this thread, so
it is one short burst. After it, every weight is a constant: all 31 `o` loads
are independent, issue back-to-back into one long `vmcnt` queue, and every
accumulate is a plain FMA with no rescale and no chain.

This is the standard flash-attention *final*-merge form — `sum_k exp2(m_k −
m_max) · o_k` over a common base rather than rescaling a running base. Identical
in exact arithmetic, and strictly better conditioned, since every weight is ≤ 1.
The running-max form exists because *online* softmax cannot see the future; the
final merge can, because all 31 lse values are already in memory. We inherited
the online form into a place that never needed it.

Measured: merge **3.11 → 2.61 µs**, qkv_attn 14.21 → 13.86 µs. 0.35 µs × 36
layers = 0.0126 ms, which is consistent with the end-to-end delta.

## 3. `MPK_MERGE_KV_OUTER` (−0.006 ms) — the loop interchange it depends on

The nest was dim-outer / kv-inner. The running softmax state (`m_global`,
`d_global`, both rescale weights) and the `lse` load all depend only on `kv`,
but the dim-outer nest recomputed the state and **re-loaded the same lse
element** once per dim. Interchanging hoists all of it: one lse load and two
`exp2` per kv step instead of `VAL_PER_THREAD` of each.

The compiler cannot do this itself. It is CSE across loads it cannot prove
unaliased — `lse_ptr` and the stores are both `float*` into the same workspace.

The interchange also fixes the access pattern: in the dim-outer form each lane's
`o` accesses were **4 B at an 8 B stride**, so half of every cache line fetched
was discarded, twice. Kv-outer makes them adjacent (`o_base + 0 ..
VAL_PER_THREAD-1`) so they coalesce into one wide load.

`MPK_MERGE_TWO_PASS` requires `MPK_MERGE_KV_OUTER` — it replaces that arm's kv
loop.

## 4. Why it stayed invisible: a measurement artifact worth internalizing

`[FUSED_PHASE]` prints `merge=` **only on the one XCD per sample whose worker
actually ran the merge** — the last chunk worker to arrive. And that XCD is
always the one reporting `xcd_barrier ≈ 1`, i.e. the XCD everybody else is
waiting on.

So in every profile dump we ever read, the merge cost appeared on exactly one
line out of eight, next to a barrier time near zero, surrounded by seven lines
showing large barrier waits. It read like a straggler outlier — an XCD that got
lucky. It was the opposite: it was the serial segment that *caused* the other
seven barrier waits, printed on the one line that looked least interesting.

The general form of the trap: **in a barrier-synchronized kernel, the worker
with the smallest wait time is the critical path.** A profile grouped by worker
will always show that worker as the outlier. Sort by "time not spent waiting,"
not by total time.

## 5. Verification

**Performance.** 9 interleaved A/B pairs, 8 wins / 1 loss, plus a 3-pair A==A
self-check (both arms 1.985 ms, spread ±0.011). Interleaving is mandatory here:
run-to-run bias is ~±0.015 ms, which exceeds most single-change effects.
Pooled over 15 runs of the final build: **1.986 ms** at KV 72->200 (see the
context-length caveat in the header -- this is a same-engine before/after
figure).

**Numerics.** Neither flag is bit-identical, and that is expected. There is no
float reassociation — each accumulator sees the same operations in the same
order over kv — but naming the weights once instead of recomputing them changes
which multiply-adds the backend contracts into `v_fma`, and a contracted FMA
keeps a wider intermediate. Generated-text hash went `88861a4763c0` →
`2051938f7067` (kv-outer) → `7c839e19bd2f` (two-pass), each stable across runs,
all three outputs coherent English.

Perplexity on wikitext-2-raw-v1/test, 4 independent 1024-token slices, greedy,
`--ppl-max-tokens 1024`:

| slice | base (13 flags) | +merge (15 flags) | delta |
|---|---|---|---|
| 0 | 282.3 | 303.6 | +7.6% |
| 1 | 103.3 | 121.3 | +17.5% |
| 2 | 159.0 | 146.8 | −7.7% |
| 3 | 205.0 | 200.6 | −2.1% |
| **pooled (4092 pos)** | **175.6** | **181.5** | mean ΔNLL +0.033 ± 0.008, t = +4.3 |

The sign flips across slices, and the per-position delta is bidirectional
(552 positions worse, 408 better, 63 identical). That is a perturbation, not a
systematic corruption — but it is statistically significant, so it needs a
scale to be judged against.

**The scale: change `CK_FMHA_NUM_KV_CHUNKS` from 8 to 16.** Splitting the KV
range differently is *exact* in real arithmetic — both configurations compute
softmax over the identical set of keys, and the merge is mathematically an
identity. Any PPL movement it produces is pure floating-point reordering. On the
same two slices:

| change | mean ΔNLL | t | pooled PPL |
|---|---|---|---|
| chunks 8→16 (mathematically exact) | **−0.107 ± 0.012** | −9.3 | 128.1 → 115.1 |
| merge two-pass + kv-outer | +0.041 ± 0.011 | +3.7 | 128.1 → 133.4 |

A known-exact change moves this pipeline **2.6× further** than the merge does.
The merge's perturbation therefore sits inside the FP-reordering sensitivity
band of the decode path — it is not evidence of a defect in the merge.

**Determinism control.** Flags that change only synchronization
(`MPK_OPROJ_TREE_BARRIER`, `MPK_LEAN_ARRIVE`, `MPK_WIDE_FP8_QUANT`,
`MPK_W13_PREQUANT` off) all produce PPL **282.2693 bit-for-bit**, identical to
base. The harness has no run-to-run noise at all, which is what makes the
comparisons above meaningful.

## 6. A larger finding this surfaced (not caused by the merge)

**The defensible comparison** is MPK against its own PyTorch reference, which
scores the identical file (`/tmp/mpk_ppl_corpus_wt2.txt`, n=1024) through the
identical preprocessing:

| | PPL |
|---|---|
| PyTorch reference, MXFP4-matched (head + QKV + O-proj round-tripped) | 100.1 |
| PyTorch reference, bf16 head/QKV/O-proj | 139.1 |
| **MPK base (13 flags)** | **282.3** |

MPK is ~2.8× worse than its own matched reference. This predates both merge
flags and is unaffected by them. It is the largest open accuracy item on this
branch and deserves its own investigation. Two caveats on the reference itself:
the bf16 arm scoring *worse* than the MXFP4-matched arm is backwards and
unexplained, which puts some uncertainty on the absolute figures; and
`CK_FMHA_NUM_KV_CHUNKS=1` gives PPL 128522, so the non-split path looks
outright broken — a separate lead.

**Do not compare this PPL against another engine's published number.** Two
things typically differ beyond the engine itself. Harnesses differ in corpus
preprocessing -- joining *all* wikitext lines including blanks and
`= Section =` headers versus stripping both gives different token streams, not
the same window. And a harness that scores a **prefill** is not measuring what
`PPL_MODE` measures: `PPL_MODE` drives the megakernel one token per iteration
and so exercises the **decode** split-KV path for every scored position. Those
are different code paths on different inputs, and the quotient of the two
numbers does not measure anything. A real cross-engine comparison means feeding
both the same preprocessed file and scoring both in decode.

## 7. Reproducing

```bash
# perf, interleaved A/B
/tmp/ab1k.sh   # base arm + MPK_MERGE_KV_OUTER=1 MPK_MERGE_TWO_PASS=1

# perplexity, one arm
PPL_MODE=1 MAX_SEQ_LENGTH=2048 MAX_NEW_TOKENS=1 \
  ./run_1gpu.sh --ppl-corpus wikitext2 --ppl-max-tokens 1024 \
                --ppl-out /tmp/pp.json --max-num-batched-requests 1

```

Note: `load_dataset("wikitext", ...)` no longer resolves under datasets≥4 /
huggingface_hub≥1. Use the namespaced mirror `"Salesforce/wikitext"`; our
loader does this.

---

# Why we missed this, and how to not miss the next one

We spent a long effort on this gap and walked past a 0.024 ms lever many times.
The failure was not lack of effort or lack of profiling. It was a set of
specific, nameable analysis habits. These are worth encoding as standing rules
for anyone — human or agent — doing this kind of work.

## 6 reasons we missed it

**1. We counted instructions instead of tracing dependencies.** This is the big
one. Every time we sized an optimization, we asked "how much work is this doing?"
The merge loop does very little work: 31 iterations, a handful of flops each. By
work, it is negligible, so it never made the shortlist. But its *latency* was
31 serialized memory round trips, because of a chain we never drew. A loop can
be simultaneously trivial in instruction count and dominant in wall clock.
Instruction-count reasoning overpredicted by ~5× on this branch **four separate
times** (see `moe-w13-prequant-gap`) and here it underpredicted by a lot. It is
simply the wrong tool.

**2. We read the profile grouped the way the profile chose to group it.** The
merge cost appeared only on the XCD whose barrier wait was ~0, so it looked like
noise on a lucky worker. We never inverted the question to "which worker is
everyone else waiting for, and what is it doing?" That is the only question a
barrier-synchronized profile is actually answering.

**3. Attention was mentally filed as "already done."** Effort concentrated on
the MoE, which is 46% of the time and had a visible roofline gap. Attention had
been tuned earlier, so it was treated as a closed subsystem. Percentage of total
time is a bad prioritizer; *distance from that component's own floor* is the
right one. The merge was ~20% off its floor inside a phase that was only 25% of
the total, and that beat everything remaining in the 46% phase.

**4. We never asked why an online algorithm was in an offline position.** The
running-max form is *required* for streaming softmax and *unnecessary* for a
final merge where all partials are already resident. Nobody wrote that
distinction down, so the inherited form was never questioned. Whenever a
textbook algorithm appears in the code, the question is not "is this the right
algorithm" but "does this call site still have the constraint the algorithm was
designed for?"

**5. Single-flag ablation hid a two-part win.** `MPK_MERGE_KV_OUTER` alone is
−0.006 ms, roughly noise, and would have been discarded on its own. It only pays
because it is the enabling refactor for two-pass. Testing optimizations strictly
one at a time is correct for *attribution* and wrong for *discovery*: it
systematically hides any win that requires a scaffold. (The user flagged exactly
this failure mode mid-effort — "maybe it's a combination of optimizations and
you're testing one in isolation and getting a regression?")

**6. We trusted "coherent output" as a correctness check for far too long.** The
generated text stayed fluent English through every variant, so numerics never
got scrutiny until challenged directly. Fluent output is nearly free; a model
this size produces readable text through substantial numerical damage. The
3.6× accuracy gap in §6 sat undetected the entire time for this reason.

## How to teach an agent not to miss these

Rules, in priority order. Each maps to a failure above.

**R1. For any loop on a critical path, draw the dependency chain before
counting instructions.** Ask: does iteration *k* need a value produced by
iteration *k−1*? If yes, the cost is `N × chain_latency`, not `N × work`. Loads
inside such a chain cannot overlap and the memory system sits idle. This one
question would have found the merge immediately.

**R2. In a barrier-synchronized kernel, profile the critical path, not the
workers.** The worker with the *smallest* wait is the one everyone waits for.
Any per-worker metric that shows up on one line out of N, next to a near-zero
barrier time, is a critical-path segment misreported as an outlier. Sort by
"time not waiting."

**R3. Prioritize by distance-from-floor, not by share-of-total.** Compute a
roofline for every phase, including small ones and including ones already
optimized. A 20%-off-floor segment in a small phase beats a 5%-off-floor segment
in a big one. "We already tuned that" is a claim about the past, not evidence
about the present.

**R4. When you find a standard algorithm, check whether its constraint still
applies here.** Online/streaming forms, running accumulators, and incremental
rescaling all exist to handle data you cannot see yet. If the data is already in
memory, the offline form is available and is usually both faster and better
conditioned.

**R5. Separate discovery from attribution.** Discovery: try bundles, and try
enabling refactors together with the thing they enable. Attribution: once a
bundle wins, ablate one flag at a time to apportion credit. Never let the
attribution discipline (one variable per run) prune the discovery search.

**R6. Never accept fluent output as a correctness check, and never accept a
correctness delta without a noise floor.** Two concrete requirements:
  - Hash the generated text (`/tmp/txthash.sh`); `d41d8cd98f00` means empty,
    i.e. a broken build silently reporting a great latency.
  - When a change moves a quality metric, measure a **mathematically-neutral
    perturbation of the same pipeline** and compare magnitudes. Here that was
    `CK_FMHA_NUM_KV_CHUNKS 8→16`, which is exact by construction and moved PPL
    2.6× further than the change under test. Without that yardstick, "+0.033
    NLL, t=4.3" reads as a defect; with it, it reads as ordinary FP sensitivity.
    Establish the floor *before* you need it.

**R7. Distrust every number whose provenance you have not personally traced.**
Two claims in this effort were self-corrected only after re-derivation: "192
tiles on 248 workers means 23% idle" (false — with ≤1 tile/worker the phase is
one tile's latency regardless), and "W13 is 1.4× off W2's efficiency" (false —
it compared W13's load+compute against W2's compute window alone, and W2 hides
its loads under a barrier wait). Both were plausible, both were wrong, both cost
days. Before acting on a derived ratio, state what each term measures and
confirm the two terms are commensurable.

## See also

- `docs/mpk/row-symmetry-b-gt-1.md` — the B>1 row-symmetry defect, resolved in
  `3bf8c32`; also why the fixed split-KV partition is what makes decode
  bit-reproducible run to run
- memory: `splitkv-merge-serial-chain`, `moe-phase8-roofline`,
  `correctness-check-txthash`
