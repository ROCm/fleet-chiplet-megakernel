# Correctness infrastructure for GPT-OSS on MPK

Three gates, cheapest first. Run the one that matches what you changed; run all
three before claiming a numerical change is safe.

| what | what it catches | cost | gated in CI |
|---|---|---|---|
| generated-text hash | build is broken, kernel emits garbage | seconds | no |
| per-stage layer compare | *which op* diverged | ~4 min | not yet |
| perplexity vs matched reference | distribution-level quality | ~10 min/arm | yes |

---

## 0. The rule that comes before all of them

**Never quote a latency number without checking the generated text.** A
megakernel that faults, or one built from a stale `permanent_output_dir`, still
prints a plausible `avg_ms`. Hash the output:

```bash
/tmp/txthash.sh <logfile>     # d41d8cd98f00 == empty == the run produced nothing
```

**Never run two MPK arms concurrently.** Two `--use-mirage` processes on
different GPUs with different build directories and genuinely different `.so`
files (distinct md5s, real opcode diffs) have been observed emitting
*bit-identical* logits. `persistent_kernel.py` hardcodes
`tempdir = "./permanent_output_dir/"` in both branches and a separate cwd is
not sufficient isolation. Serialize A/B arms, and prove determinism by
repeating one arm and diffing the dumps before trusting any delta.

---

## 1. Per-stage layer comparison (the sharp instrument)

```bash
MODEL_PATH=/path/to/gpt-oss-120b \
  tests/ci-tests/run_ci_tests_gpt_oss_layer_compare.sh
```

Dumps MPK's live scratch buffers and the reference's forward-hook
intermediates at depth 1 and depth 2, then gates cosine similarity and
relative RMSE per stage. Depth 1 pins the comparison to layer 0
(`sliding_attention`), depth 2 to layer 1 (`full_attention`) — MPK's stage
buffers are single-token scratch holding the *last* layer's values, so
`--max-layers` is what selects the layer under test.

Files: `tests/ci-tests/test_gpt_oss_layer_compare.py`,
`tests/ci-tests/run_ci_tests_gpt_oss_layer_compare.sh`.

### The reference must match quantization on both sides

`PPL_MXFP4_MATCH=1` round-trips the LM head, QKV and O-proj weights through
MXFP4. `PPL_FP8_ACT=1` additionally quantizes every GEMM *input* the way
`_gang_wave_parallel_fp8_quant` does. Both are set by the runner script.

Without the second one the reference feeds bf16 activations into every GEMM
while MPK feeds FP8 — about 2.3% mean relative error on the input of every
expert, QKV, O-proj and LM-head GEMM in all 36 layers, all of it charged to
"kernel error." That forces thresholds so loose they gate nothing -- a layer
test that compares an FP8 fused MoE against an fp16 eager reference ends up
having to set its `mlp` fail threshold near cos 0.60, which no longer
distinguishes a working kernel from a broken one.

### Reading the output

```
[layer-compare] L0 (sliding_attention) OK   attn       cos=0.999848 rel_rmse=1.7501e-02
[layer-compare] L0 (sliding_attention) OK   ln2        cos=0.999823 rel_rmse=1.8853e-02
[layer-compare] L0 (sliding_attention) OK   layer_out  cos=0.999958 rel_rmse=9.4680e-03
[layer-compare] L1 (full_attention)    OK   ln2        cos=0.999403 rel_rmse=3.4802e-02
[layer-compare] L1 (full_attention)    OK   layer_out  cos=0.999871 rel_rmse=1.6161e-02
```

Gates: `cos >= 0.995` **and** `rel_rmse <= 0.055`. Both, because cosine alone
is a weak discriminator once a residual and the MoE dilute an upstream error —
see the table below.

### Prove the gate works before you trust it

`MPK_LSE_LOG_BUG=1` rebuilds the decode kernel with the pre-`49f446b` split-KV
LSE unit bug deliberately restored. It is fault injection, not a tuning knob.

```bash
MPK_LSE_LOG_BUG=1 MODEL_PATH=... \
  tests/ci-tests/run_ci_tests_gpt_oss_layer_compare.sh   # must FAIL
```

| depth | stage | fixed cos | bug cos | fixed rel | bug rel |
|---|---|---|---|---|---|
| L=1 | attn | 0.999848 | 0.998377 | 1.75e-02 | 8.20e-02 |
| L=1 | ln2 | 0.999823 | 0.997701 | 1.89e-02 | 7.29e-02 |
| L=1 | layer_out | 0.999958 | 0.999159 | 9.47e-03 | 4.79e-02 |
| L=2 | ln2 | 0.999403 | 0.992377 | 3.48e-02 | 1.27e-01 |
| L=2 | layer_out | 0.999871 | 0.995638 | 1.62e-02 | 1.13e-01 |

Note L=1 `layer_out`: the bug moves cosine only 0.99996 → 0.99916, because an
error that is 8.2% at the attention output is diluted by the residual and the
MoE. Relative RMSE separates 4–7× on every stage. If you ever widen a
threshold, widen it to a *measured* number and re-run this injection.

### Which stages are gated, and why not more

The pairings were found by cross-correlating every MPK buffer against every
reference stage, not by matching names — and the names mislead:

| MPK buffer | reference stage | cos | note |
|---|---|---|---|
| `attn_proj_out` − `embed_out` | `L0.attn` | 0.9998 | **residual already added** |
| `rmsnorm_out_moe` | `L<i>.ln2` | 0.9994 | next best 0.5606 |
| `mlp_weighted_sum_out` | `L<i>.out` | 0.9999 | next best 0.9155 |

Not gated, deliberately:

- `attn_in`, `moe_gate_out`, `mlp_final` are **exactly zero** — dead buffers in
  the fused full-layer path. A threshold on a zero tensor is coverage theater.
  `test_known_dead_buffers_still_dead` pins this so we notice if one comes
  alive.
- `attn_out` is pre-o_proj and head-major (4096 wide); the reference's
  `self_attn` hook fires post-o_proj (2880). Different tensors — comparing them
  gives cos ≈ 0.03, which looks like catastrophe and means nothing.
- `rmsnorm_out` has no exact counterpart among the reference's hook points; its
  best match (0.956) is a *neighbouring* stage, so a threshold there would
  measure hook placement.
- The attention pair is depth-1 only: at depth 2 the layer input is layer 0's
  output, which the single-token scratch no longer holds, so subtracting
  `embed_out` is wrong (it lands at 0.637).

Extending coverage means adding reference hook points that line up with MPK's
actual buffers — not loosening a threshold until a mismatched pair passes.

---

## 2. Perplexity (the distribution-level gate)

```bash
MODEL_PATH=/path/to/gpt-oss-120b \
  tests/ci-tests/run_ci_tests_gpt_oss_perplexity.sh
```

Scores a WikiText-2 window with the Torch reference and with MPK, then gates on
an absolute ceiling and an MPK/Torch ratio
(`tests/ci-tests/test_gpt_oss_perplexity.py`).

### Never size an accuracy change on one window

This is the single most expensive lesson in this file. Headline perplexity on
the 1024-token windows of this corpus ranges over **103..282** on the same
reference, and the MPK-minus-reference delta *sign-flips* between them:

| window | MPK − ref, mean NLL | t |
|---|---|---|
| 0 | +0.108 | +5.9 |
| 1 | +0.009 | +0.8 |
| 2 | +0.025 | +1.9 |
| 3 | **−0.054** | −4.4 |
| **pooled (n=4092)** | **+0.0222 ± 0.0070** | **+3.2** |

Window 0 alone reads as a hard defect. It is not; MPK *beats* the reference on
window 3. Use `PPL_SLICE=k` to score window *k*, run at least four, and pool
per-position NLL with a paired t-test:

```bash
for s in 0 1 2 3; do
  PPL_SLICE=$s PPL_MODE=1 python3 demo/gpt_oss/demo.py --model-path "$MODEL_PATH" \
    --use-mirage --max-num-batched-tokens 1 --max-seq-length 1088 \
    --ppl-max-tokens 1024 --ppl-out /tmp/mpk_s$s.json
done
```

`--ppl-out` JSON carries `per_position_nll`, which is what the t-test consumes.
Headline perplexity on one slice proves nothing.

### The noise floor

MPK is bit-deterministic — sync-only flags reproduce perplexity bit-for-bit —
so the floor is not run-to-run variance but the pipeline's sensitivity to
floating-point reordering, and it needs a deliberate probe.

**The probe is `CK_FMHA_NUM_KV_CHUNKS=8` vs `16`.** Splitting the KV range
differently is exact in real arithmetic (same keys; the split-KV merge is a
mathematical identity), so anything it moves is pure FP reordering. Pooled over
the same four slices it moves mean NLL by **+0.0093 ± 0.0073**. A change that
moves less than that is inside the band.

Current standing: MPK is **+0.0222 ± 0.0070** nats from its matched reference,
about 2.4× the yardstick.

### Corpus gotcha

`load_dataset("wikitext", ...)` fails under `datasets>=4` / `huggingface_hub>=1`.
Use `"Salesforce/wikitext"` — already fixed in `load_ppl_corpus`.

---

## 3. Op-by-op localization (when a gate fires)

The gates say *what* broke. To find *where*, dump raw stages and diff them
yourself:

```bash
# reference arm
PPL_MODE=1 PPL_MXFP4_MATCH=1 PPL_FP8_ACT=1 \
  PPL_STAGE_DUMP=/tmp/ref.pt PPL_STAGE_ROW=-1 \
  python3 demo/gpt_oss/demo.py --model-path "$MODEL_PATH" --max-layers 1 \
    --ppl-max-tokens 64 --ppl-out /tmp/ref.json

# MPK arm -- separately, never concurrently
PPL_MODE=1 PPL_STAGE_DUMP=/tmp/mpk.pt PPL_STAGE_ROW=-1 \
  python3 demo/gpt_oss/demo.py --model-path "$MODEL_PATH" --use-mirage \
    --max-layers 1 --ppl-max-tokens 64 --ppl-out /tmp/mpk.json
```

`PPL_STAGE_ROW=-1` is the last scored position — the one still live in MPK's
scratch when the run ends.

When a buffer's identity is unclear, **cross-correlate rather than assume**:

```python
import torch
r, m = torch.load('/tmp/ref.pt'), torch.load('/tmp/mpk.pt')
cos = lambda a, b: torch.nn.functional.cosine_similarity(
    a.float().reshape(-1)[:2880].unsqueeze(0),
    b.float().reshape(-1)[:2880].unsqueeze(0)).item()
for mk in m:
    best = sorted(((cos(r[rk], m[mk]), rk) for rk in r), reverse=True)[:3]
    print(mk, ['%s=%+.4f' % (k, c) for c, k in best])
```

This is how the residual inside `attn_proj_out` and the three dead buffers were
found. Both would otherwise have produced confidently wrong conclusions — a
name-based pairing of `attn_out` to `L0.attn` gives cos 0.03.

### Related knobs

| flag | effect |
|---|---|
| `PPL_SLICE=k` | score corpus window *k* |
| `PPL_STAGE_DUMP` / `PPL_STAGE_ROW` | per-stage intermediates |
| `PPL_DUMP_LOGITS` / `PPL_DUMP_ROWS` | raw logits at chosen rows |
| `PPL_MXFP4_MATCH=1` | reference uses MXFP4 weights |
| `PPL_FP8_ACT=1` | reference uses FP8 GEMM inputs |
| `PPL_FP8_ROUTER=1` | also quantize the MoE router input |
| `MPK_NO_SW_MASK=1` | ablate the sliding-window head mask |
| `MPK_LSE_LOG_BUG=1` | fault injection: restore the LSE unit bug |

---

## 4. What is not covered

Honest gaps, roughly in priority order:

- **No CI coverage of gpt-oss at all.** Neither the perplexity nor the layer
  comparison runs in `.github/workflows`; both are manual. The layer comparison
  is cheap enough to run on every push.
- **No GSM8K or any task-level eval.** Perplexity can miss a regression that
  task accuracy catches; lm-eval 3-shot `exact_match,flexible-extract` is the
  obvious thing to add.
- **No batch-invariance test.** Nothing gates that decode at high concurrency
  agrees with concurrency 1, and the fused MoE path is exactly the kind of code
  where batch shape changes reduction order.
- **Only 2 of 36 layers, one token, at depth ≤ 2.** The stage buffers are
  single-token scratch, so deeper coverage needs either more scratch retention
  or a different capture mechanism.
- **`--max-layers` has latent bugs** beyond the depths used here: `4` faults,
  `0` raises IndexError, and truncated-depth perplexity is meaningless
  (5.2e7 at depth 1) because the LM head sees an unfinished residual stream.
  Only stage comparison is valid under truncation, and only because both arms
  are truncated identically.

## See also

- `docs/mpk/splitkv-merge.md` — the merge these gates most often catch
- `tests/ci-tests/test_gpt_oss_perplexity.py` — perplexity thresholds
- `tests/ci-tests/test_gpt_oss_layer_compare.py` — per-stage thresholds
