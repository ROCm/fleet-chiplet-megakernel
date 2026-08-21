#!/usr/bin/env bash
# GPT-OSS 120B perplexity: score a WikiText-2 slice with the Torch reference and
# with Mirage (MPK), then compare (tests/ci-tests/test_gpt_oss_perplexity.py).
#
# PPL_MODE=1 loads the corpus as one long prompt and runs prefill only. The
# megakernel does not overwrite tokens[] inside the prompt, so every position
# conditions on the reference prefix -- prefill is teacher forcing. The LM head
# scores one row per iteration, hence --max-num-batched-tokens 1.
#
# Env (override as needed):
#   MODEL_PATH            local GPT-OSS 120B dir or HF repo id (required)
#   HIP_VISIBLE_DEVICES   target GPU (default 0)
#   PPL_MAX_TOKENS        corpus tokens to score (default 512)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MIRAGE_HOME="${MIRAGE_HOME:-$ROOT}"
# MUST come first: another mirage checkout at ~/mirage otherwise wins the
# `import mirage` and is missing this tree's kernel APIs.
export PYTHONPATH="$ROOT/python:${PYTHONPATH:-}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export PPL_MODE=1
MODEL_PATH="${MODEL_PATH:-${GPT_OSS_MODEL_PATH:-openai/gpt-oss-120b}}"
PPL_MAX_TOKENS="${PPL_MAX_TOKENS:-512}"
OUT_DIR="${GPT_OSS_OUTPUT_DIR:-$ROOT/outputs/gpt_oss}"

DEMO="$ROOT/demo/gpt_oss/demo.py"
COMMON=(--model-path "$MODEL_PATH" --max-num-batched-tokens 1
        --ppl-max-tokens "$PPL_MAX_TOKENS")

echo "MIRAGE_HOME=$MIRAGE_HOME  HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES  MODEL_PATH=$MODEL_PATH"
echo "Scoring with the Torch reference..."
python3 "$DEMO" "${COMMON[@]}" --ppl-out "$OUT_DIR/torch_ppl.json"
echo "Scoring with Mirage (MPK)..."
python3 "$DEMO" --use-mirage "${COMMON[@]}" --ppl-out "$OUT_DIR/mpk_ppl.json"
echo "Comparing perplexity..."
GPT_OSS_OUTPUT_DIR="$OUT_DIR" pytest -q -s "$ROOT/tests/ci-tests/test_gpt_oss_perplexity.py"
