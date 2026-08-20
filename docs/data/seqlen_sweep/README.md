# Decode-latency sweep data

Raw measurements behind `docs/img/seqlen_sweep.svg` and the table in the
top-level README. Committed so the plot can be regenerated, and so the
numbers can be audited without re-running ~40 minutes of benchmarks.

Regenerate the plot:

```bash
python3 tests/ci-tests/plot_seqlen_sweep.py \
  --fleet docs/data/seqlen_sweep/fleet_summary.json \
  --vllm  docs/data/seqlen_sweep \
  --out   docs/img/seqlen_sweep.svg
```

## What is here

`fleet_summary.json` — written by `tests/ci-tests/summarize_seqlen_sweep.py`.
One row per context length. `tpot_ms` is the **median** of the decode-phase
`[FWD_PASS]` device-side ring samples at the 400-token point, not a mean and
not `[FWD_PASS_TOTAL] avg_ms` (which blends prefill in). Each row also keeps
`txthash`, the hash of the generated text: a faulted kernel still prints a
plausible latency, so a distinct non-empty hash per point is what makes the
number trustworthy. The two `points` entries (4 and 400 generated tokens)
exist so prefill can be separated from decode.

`vllm_<ctx>_o<outlen>.json` — `vllm bench latency --output-json`. Two output
lengths per context, 1 and 129. `bench latency` reports end-to-end latency
including prefill, so TPOT is the difference `(latency@129 - latency@1) / 128`;
both points pay the same prefill, so it cancels.

The two arms use different estimators on purpose: Fleet exposes per-iteration
decode samples directly, so differencing two totals there would subtract two
noisy prefills (~1% of a 10.7 s prefill is ~107 ms, against ~1000 ms of
signal) for no benefit.

## Provenance

- Hardware: MI355X (gfx950), 1 GPU, batch 1
- ROCm 7.0.0 / HIP 7.0.51831
- vLLM 0.11.1rc2 (`38f225c2a`), ROCm build

**The vLLM arm is not native FP4.** That build's `Mxfp4Backend` selector
returns Triton on every ROCm device, so the MXFP4 MoE weights are
dequantized and run as BF16 GEMMs. `VLLM_ROCM_USE_AITER=1` does not change
it. A build with native FP4 MoE kernels is substantially faster than this
column and is the fair comparison to make. Check the `Using <X> backend`
line in the run log before quoting any of these numbers; "Triton" means
dequantized.

Arms were run serially on the same GPU — two concurrent MPK builds have been
observed contaminating each other (see the top-level `CLAUDE.md`).
