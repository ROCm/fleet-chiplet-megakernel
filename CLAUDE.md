# Working in this repo

Fleet: GPT-OSS 120B as a Mirage Persistent Kernel (megakernel) on AMD MI350X
(gfx950), batch 1. All 36 layers fuse into one persistent task.

## Build and run

```bash
export MIRAGE_HOME=$(pwd)
export MODEL_PATH=/path/to/gpt-oss-120b
export GPU=0

rm -rf demo/gpt_oss/permanent_output_dir     # header edits do NOT trigger a rebuild
HIP_VISIBLE_DEVICES=$GPU python3 demo/gpt_oss/demo.py --use-mirage \
  --model-path "$MODEL_PATH" --prompt "Tell me the history of america" \
  --max-seq-length 512
```

Read `Decode avg` for per-token latency, not the per-iteration `FWD_PASS` lines.
First run JIT-compiles (couple of minutes); a run past ~3 minutes is hanging —
`pkill -9 -f demo.py` and `rm -f /tmp/gpucore*`.

New env flags must be registered in **two** places or they are silently ignored:
`python/mirage/mpk/persistent_kernel.py` (adds the `-D` to the compile) and
`demo/gpt_oss/env_common.sh` `MPK_FORWARD_VARS` (forwards it to the child).

## Correctness

Full guide: **[`docs/mpk/correctness-infra.md`](docs/mpk/correctness-infra.md)**.
Three gates, cheapest first:

```bash
# 1. end-to-end tokens (seconds)
MODEL_PATH=$MODEL_PATH bash tests/ci-tests/run_ci_tests_gpt_oss.sh

# 2. per-stage layer compare -- names the op that diverged (~4 min)
MODEL_PATH=$MODEL_PATH bash tests/ci-tests/run_ci_tests_gpt_oss_layer_compare.sh

# 3. perplexity vs the matched Torch reference (~10 min/arm)
MODEL_PATH=$MODEL_PATH bash tests/ci-tests/run_ci_tests_gpt_oss_perplexity.sh
```

Prove a gate still bites before trusting it. `MPK_LSE_LOG_BUG=1` rebuilds with a
real historical defect injected; the layer-compare run **must** go red:

```bash
MPK_LSE_LOG_BUG=1 MODEL_PATH=$MODEL_PATH \
  bash tests/ci-tests/run_ci_tests_gpt_oss_layer_compare.sh   # expect FAIL
```

## Rules learned the hard way

Each of these cost a day or more. They are not style preferences.

- **Never quote a latency number without checking the generated text.** A
  faulted kernel, or one built from a stale `permanent_output_dir`, still prints
  a plausible `avg_ms`. Hash the output; `d41d8cd98f00` means empty.
- **Never run two MPK arms concurrently.** Two `--use-mirage` processes on
  different GPUs, with different build dirs and different `.so` md5s, have been
  observed emitting bit-identical logits. `persistent_kernel.py` hardcodes
  `tempdir = "./permanent_output_dir/"`; a separate cwd is not isolation.
  Serialize, then prove determinism by repeating one arm and diffing.
- **Change one variable per run.** But when an isolated change regresses, check
  whether it only pays off combined with another — several wins here were
  invisible alone.
- **Never size an accuracy change on one perplexity window.** The four
  1024-token windows of this corpus score 41/54/100/104 on the *same* reference
  and the MPK-minus-reference delta sign-flips between them. Use `PPL_SLICE=k`
  over >=4 windows and a paired per-position NLL t-test.
- **Know the noise floor before calling something a defect.** MPK is
  bit-deterministic, so the floor is FP-reordering sensitivity, not run
  variance. Probe it with `CK_FMHA_NUM_KV_CHUNKS=8` vs `16` — mathematically
  exact, so whatever it moves is noise (currently +0.0093 +/- 0.0073 nats).
- **Cross-correlate buffers, do not pair them by name.** MPK's `attn_out` is
  pre-o_proj and head-major; the reference's `self_attn` hook is post-o_proj.
  Name-matching them gives cos 0.03 and reads as catastrophe.
- **A gate nobody has watched go red is not known to work.** Hence
  `MPK_LSE_LOG_BUG`.

## Docs

- [`docs/mpk/correctness-infra.md`](docs/mpk/correctness-infra.md) — the three gates, in detail
- [`docs/mpk/splitkv-merge.md`](docs/mpk/splitkv-merge.md) — split-KV merge, and how a profiler misread hid 0.024 ms
- [`docs/mpk/row-symmetry-b-gt-1.md`](docs/mpk/row-symmetry-b-gt-1.md) — the B>1 row-symmetry defect and its fix; what bs>1 is and is not gated for
- [`demo/gpt_oss/README.md`](demo/gpt_oss/README.md) — flags, build variants, prompt handling
