#!/usr/bin/env bash
# GPT-OSS 120B per-stage layer comparison: dump MPK and matched-Torch
# intermediates at depth 1 and 2, then gate on cosine similarity
# (tests/ci-tests/test_gpt_oss_layer_compare.py).
#
# Depth 1 pins the comparison to layer 0 (sliding_attention) and depth 2 to
# layer 1 (full_attention) -- MPK's stage buffers are single-token scratch and
# hold the LAST layer's values, so --max-layers selects which layer is under
# test. Both attention kinds are covered.
#
# The reference arm runs with PPL_MXFP4_MATCH=1 PPL_FP8_ACT=1 so it matches
# MPK on BOTH weights (MXFP4) and activations (FP8). A weights-only baseline
# charges ~2.3% per-GEMM activation quantization error to "kernel error" and
# forces uselessly loose thresholds.
#
# Env (override as needed):
#   MODEL_PATH            local GPT-OSS 120B dir or HF repo id (required)
#   HIP_VISIBLE_DEVICES   target GPU (default 0)
#   STAGE_TOKENS          corpus tokens to run (default 64)
#   PPL_CORPUS            corpus file, or "wikitext2" to download the split
#                         (default: wikitext2). A local file is preferred on
#                         runners without network -- demo.py's load_ppl_corpus
#                         takes either.
#   MPK_LSE_LOG_BUG=1     fault injection: rebuild MPK with the pre-49f446b
#                         split-KV LSE unit bug so the gate can be watched
#                         going red. See docs/mpk/correctness-infra.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MIRAGE_HOME="${MIRAGE_HOME:-$ROOT}"
export PYTHONPATH="$ROOT/python:${PYTHONPATH:-}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export PPL_MODE=1

MODEL_PATH="${MODEL_PATH:-${GPT_OSS_MODEL_PATH:-openai/gpt-oss-120b}}"
STAGE_TOKENS="${STAGE_TOKENS:-64}"
PPL_CORPUS="${PPL_CORPUS:-wikitext2}"
STAGE_DIR="${GPT_OSS_STAGE_DIR:-$ROOT/outputs/gpt_oss/stage}"
DEMO="$ROOT/demo/gpt_oss/demo.py"

mkdir -p "$STAGE_DIR"

# Row -1 = the last scored position, which is the one whose values are still
# live in MPK's scratch when the run ends.
export PPL_STAGE_ROW="${PPL_STAGE_ROW:--1}"

echo "MIRAGE_HOME=$MIRAGE_HOME  HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
echo "MODEL_PATH=$MODEL_PATH  tokens=$STAGE_TOKENS  out=$STAGE_DIR"

for L in 1 2; do
  echo "=== depth $L: Torch reference (MXFP4 weights + FP8 activations) ==="
  PPL_MXFP4_MATCH=1 PPL_FP8_ACT=1 \
  PPL_STAGE_DUMP="$STAGE_DIR/ref_L${L}.pt" \
    python3 "$DEMO" --model-path "$MODEL_PATH" --max-layers "$L" \
      --ppl-corpus "$PPL_CORPUS" --ppl-max-tokens "$STAGE_TOKENS" \
      --ppl-out "$STAGE_DIR/ref_L${L}.json"

  # MPK arms MUST be serial. Two concurrent --use-mirage runs contaminate each
  # other's results even on different GPUs with different build dirs -- they
  # emit bit-identical logits from genuinely different binaries. The build dir
  # (./permanent_output_dir) is also hardcoded, hence the rm.
  echo "=== depth $L: MPK ==="
  rm -rf "$ROOT/demo/gpt_oss/permanent_output_dir"
  PPL_STAGE_DUMP="$STAGE_DIR/mpk_L${L}.pt" \
    python3 "$DEMO" --model-path "$MODEL_PATH" --use-mirage --max-layers "$L" \
      --ppl-corpus "$PPL_CORPUS" --ppl-max-tokens "$STAGE_TOKENS" \
      --ppl-out "$STAGE_DIR/mpk_L${L}.json"
done

echo "=== comparing stages ==="
GPT_OSS_STAGE_DIR="$STAGE_DIR" \
  pytest -q -s "$ROOT/tests/ci-tests/test_gpt_oss_layer_compare.py"
