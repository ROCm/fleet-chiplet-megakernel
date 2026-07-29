#!/usr/bin/env bash
# GPT-OSS 120B bs=1 end-to-end correctness: run the Torch reference and the
# Mirage (MPK) path, dump their generated tokens, then compare with the
# vLLM-style tolerant check (tests/ci-tests/test_gpt_oss_inference_output.py).
#
# Env (override as needed):
#   MODEL_PATH            local GPT-OSS 120B dir or HF repo id (required)
#   HIP_VISIBLE_DEVICES   target GPU (default 0)
#   GPT_OSS_PROMPT        prompt (default "Tell me the history of america")
#   GPT_OSS_MAX_SEQ_LEN   sequence length (default 512)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MIRAGE_HOME="${MIRAGE_HOME:-$ROOT}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export USE_FP8_ACT="${USE_FP8_ACT:-1}"
MODEL_PATH="${MODEL_PATH:-${GPT_OSS_MODEL_PATH:-openai/gpt-oss-120b}}"
PROMPT="${GPT_OSS_PROMPT:-Tell me the history of america}"
MAX_SEQ_LEN="${GPT_OSS_MAX_SEQ_LEN:-512}"

DEMO="$ROOT/demo/gpt_oss/demo.py"
COMMON=(--model-path "$MODEL_PATH" --prompt "$PROMPT" --max-seq-length "$MAX_SEQ_LEN" --save-tokens)

echo "MIRAGE_HOME=$MIRAGE_HOME  HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES  MODEL_PATH=$MODEL_PATH"
echo "Running Torch reference..."
python3 "$DEMO" "${COMMON[@]}"
echo "Running Mirage (MPK)..."
python3 "$DEMO" --use-mirage "${COMMON[@]}"
echo "Comparing outputs..."
pytest -q -s "$ROOT/tests/ci-tests/test_gpt_oss_inference_output.py"
