# Running GPT-OSS 120B with Mirage

GPT-OSS 120B (MoE, FP8 activations) as a Mirage Persistent Kernel (MegaKernel),
**batch size 1**, on AMD MI350 / MI355 (gfx950). Reference decode latency:
**~2.0 ms/tok**.

Set these once (adjust to your environment):

```bash
export MIRAGE_HOME=$(pwd)                 # repo root
export MODEL_PATH=/path/to/gpt-oss-120b   # local GPT-OSS 120B weights
export GPU=0                              # target GPU id
```

All commands below run from the repo root.

## Standard run (batch size 1)

```bash
rm -rf demo/gpt_oss/permanent_output_dir
USE_FP8_ACT=1 HIP_VISIBLE_DEVICES=$GPU \
python3 demo/gpt_oss/demo.py --use-mirage \
  --model-path "$MODEL_PATH" \
  --prompt "Tell me the history of america" \
  --max-seq-length 512
```

Read the **`Decode avg`** line from the output for the per-token latency (TPOT) —
not the per-iteration `FWD_PASS` lines. To run the PyTorch reference instead of
the megakernel, drop `--use-mirage`.

> **Note:** For GPT-OSS 120B, **only batch size 1 is supported.** The
> `--max-num-batched-requests` / `--max-num-batched-tokens` flags exist for other
> models; do not expect bs > 1 to work for GPT-OSS 120B.

## Setting the prompt

The input prompt is set with `--prompt` (wrap it in double quotes):

```bash
rm -rf demo/gpt_oss/permanent_output_dir
USE_FP8_ACT=1 HIP_VISIBLE_DEVICES=$GPU \
python3 demo/gpt_oss/demo.py --use-mirage \
  --model-path "$MODEL_PATH" \
  --prompt "Explain how transformers work in simple terms" \
  --max-seq-length 512
```

Notes:
- If `--prompt` is omitted, the default is `"The capital of France is"`.
- Keep the prompt **fixed** when comparing performance numbers — different prompts
  change the token count and make `Decode avg` non-comparable.
- `--max-seq-length` is the total sequence length (prompt + generated tokens). Keep
  it at `512` for benchmarking; shorter values give artificially low TPOT.

## Required environment

| Variable | Value | Why |
|----------|-------|-----|
| `MIRAGE_HOME` | repo root | Mirage repo root. |
| `USE_FP8_ACT` | `1` | FP8 activations (the benchmark config). |
| `HIP_VISIBLE_DEVICES` | GPU id | Target GPU. |

## Key flags

| Flag | Meaning |
|------|---------|
| `--use-mirage` | Run the Mirage persistent megakernel (vs. the PyTorch reference path). |
| `--model-path` | Path to a local GPT-OSS 120B model directory. |
| `--prompt` | Input prompt. Keep fixed for comparable numbers. |
| `--max-seq-length 512` | Sequence length. Keep at 512 — shorter prompts give artificially low TPOT. |

## Build variants

Prepend these environment variables to the command:

| Variable | Effect |
|----------|--------|
| `MPK_LAYER_ORACLE=1` | Use the 256-thread oracle (non-decoupled) fused-layer build. |
| `MPK_SUBPHASE_TIMING=1` | Per-subphase device timing breakdown. |
| `MPK_GAP_TIMING=1` | Inter-task gap timing. |

The default (no flag) build is the **decoupled 512-thread** (4 memory + 4 compute
wave) path. The oracle build is the working batch-1 reference.

## Notes

- **Always `rm -rf demo/gpt_oss/permanent_output_dir` after any kernel/header
  change** — header edits do not trigger a rebuild on their own. The kernel
  compiles into `demo/gpt_oss/permanent_output_dir`.
- The first mirage run JIT-compiles the persistent kernel (can exceed a couple of
  minutes); later runs reuse the cached kernel.
- If a run takes longer than ~3 minutes it is hanging — kill it
  (`pkill -9 -f demo.py`) and clean up core dumps (`rm -f /tmp/gpucore*`).
- To capture output to a file instead of watching live, append `> /tmp/run.txt 2>&1`
  and read the file.
