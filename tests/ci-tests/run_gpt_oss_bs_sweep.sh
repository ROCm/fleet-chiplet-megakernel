#!/usr/bin/env bash
# GPT-OSS 120B decode-latency sweep across batch size (concurrent requests).
#
# WHY THIS IS NOT `Decode avg`
#
# demo.py derives prefill_iterations as ceil(prompt_len / max_num_batched_tokens),
# which is only correct at B=1. At B=3 that is ceil(512/3)=171, but three
# 512-token prompts actually cost ~512 iterations, so every iteration in between
# is misclassified as decode and the printed `Decode avg` is a prefill/decode
# blend -- it has read 5.345 ms against a true 2.069 ms/token.
#
# So TPOT here is a two-point difference on the device-side [FWD_PASS_TOTAL]
# clock at two decode lengths. Both points pay exactly the same prefill and the
# same setup, so both cancel:
#
#   ms_per_step  = (total_ms(HI) - total_ms(LO)) / (HI - LO)
#   ms_per_token = ms_per_step / B
#
# Requests are given DISTINCT prompts. With identical prompts a request reading
# another request's KV cache still produces the right answer, so identical
# prompts cannot gate the batched path. --ignore-eos keeps every request alive
# for the full decode length, which is what makes the two points differenceable.
#
# Runs are strictly serial: two --use-mirage processes contaminate each other
# even on different GPUs, because the build dir is hardcoded.
#
# Env:
#   MODEL_PATH            GPT-OSS 120B dir (required)
#   HIP_VISIBLE_DEVICES   target GPU (default 0; NOTE this is a HIP index and
#                         need not match rocm-smi's GPU numbering)
#   BATCH_SIZES           concurrent requests to sweep (default 1 2 4 8)
#   GEN_LO / GEN_HI       decode lengths to difference (default 128 / 256)
#   BS_OUT                log/result dir (default outputs/gpt_oss/bs_sweep)
#   TOKENS_PER_ITER       if set, run the token-PACKING sweep instead: one
#                         request prefilling a 512-token prompt at
#                         --max-num-batched-tokens T for each T listed. This
#                         measures prefill, not decode, so it does not use the
#                         two-point difference -- see the comment on that loop.
#   PROMPT_DIR            make_long_prompts.py output, for the packing sweep
#                         (default /tmp/prompts)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MIRAGE_HOME="${MIRAGE_HOME:-$ROOT}"
# MUST come first: any other mirage checkout on PYTHONPATH, or an editable
# install pointing at one, is what `import mirage` resolves to otherwise --
# and benchmarking it would silently measure the wrong repo.
export PYTHONPATH="$ROOT/python:${PYTHONPATH:-}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the gpt-oss-120b directory}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8}"
GEN_LO="${GEN_LO:-128}"
GEN_HI="${GEN_HI:-256}"
BS_OUT="${BS_OUT:-$ROOT/outputs/gpt_oss/bs_sweep}"
PROMPT_DIR="${PROMPT_DIR:-/tmp/prompts}"
DEMO="$ROOT/demo/gpt_oss/demo.py"

mkdir -p "$BS_OUT"

# Eight distinct prompts, cycled if B is smaller. Distinct is the point: see
# the header.
PROMPTS=(
  "What is the capital of Japan?"
  "Who wrote Romeo and Juliet?"
  "Name three prime numbers."
  "What is 12 times 12?"
  "Explain photosynthesis in one sentence."
  "What year did the Apollo 11 mission land?"
  "List the primary colors."
  "What is the boiling point of water?"
)

run_point() {
  local B=$1 GEN=$2 LOG=$3
  # Header edits do not trigger a rebuild, and the build dir is shared.
  rm -rf "$ROOT/demo/gpt_oss/permanent_output_dir"
  set +e
  timeout 3600 python3 "$DEMO" --use-mirage \
    --model-path "$MODEL_PATH" \
    --prompts "${PROMPTS[@]:0:$B}" \
    --max-num-batched-requests "$B" \
    --max-num-batched-tokens "$B" \
    --max-seq-length 512 \
    --max-new-tokens "$GEN" \
    --ignore-eos \
    > "$LOG" 2>&1
  local RC=$?
  set -e
  return $RC
}

# --- token-packing sweep (prefill), TOKENS_PER_ITER set ---------------------
#
# Different measurement from the decode sweep below. Here there is ONE request
# and we vary how many of its prompt tokens the scheduler packs into a single
# megakernel pass. Prefill dominates the run, so instead of differencing two
# decode lengths we subtract the decode tail from the accumulator:
#
#   prefill_ms  = total_ms - generated_tokens * median_decode_ms
#   prefill_it  = iters    - generated_tokens
#
# [FWD_PASS] alone will NOT do: its ring buffer truncates, so for a 512-token
# prompt it retains only the last few dozen iterations and the per-iteration
# medians it yields are drawn from an arbitrary window. [FWD_PASS_TOTAL] is the
# untruncated accumulator. --ignore-eos keeps the 8-token tail a real 8 tokens.
if [[ -n "${TOKENS_PER_ITER:-}" ]]; then
  PROMPT="$PROMPT_DIR/prompt_512.txt"
  if [[ ! -f "$PROMPT" ]]; then
    echo "MISSING $PROMPT -- run tests/ci-tests/make_long_prompts.py first" >&2
    exit 1
  fi
  for T in $TOKENS_PER_ITER; do
    LOG="$BS_OUT/pack_t${T}.log"
    rm -rf "$ROOT/demo/gpt_oss/permanent_output_dir"
    echo "=== packing T=$T ==="
    set +e
    timeout 3600 python3 "$DEMO" --use-mirage \
      --model-path "$MODEL_PATH" \
      --prompt-file "$PROMPT" \
      --max-num-batched-requests 1 \
      --max-num-batched-tokens "$T" \
      --max-seq-length 600 \
      --max-new-tokens 8 \
      --ignore-eos \
      > "$LOG" 2>&1
    RC=$?
    set -e
    if [[ $RC -eq 0 ]]; then
      grep -E '^\[FWD_PASS_TOTAL\]' "$LOG" | tail -1 | sed 's/^/  /'
    else
      echo "  FAILED T=$T (see $LOG)"
    fi
  done
  echo
  echo "=== summary ==="
  python3 "$ROOT/tests/ci-tests/summarize_bs_sweep.py" --packing "$BS_OUT"
  exit 0
fi

for B in $BATCH_SIZES; do
  echo "=== Fleet B=$B  gen ${GEN_LO} then ${GEN_HI} ==="
  for GEN in "$GEN_LO" "$GEN_HI"; do
    LOG="$BS_OUT/fleet_b${B}_g${GEN}.log"
    if run_point "$B" "$GEN" "$LOG"; then
      grep -E '^\[FWD_PASS_TOTAL\]' "$LOG" | tail -1 | sed 's/^/  /'
    else
      echo "  FAILED B=$B gen=$GEN (see $LOG)"
    fi
  done
done

echo
echo "=== summary ==="
python3 "$ROOT/tests/ci-tests/summarize_bs_sweep.py" "$BS_OUT"
