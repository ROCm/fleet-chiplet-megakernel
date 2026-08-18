#!/usr/bin/env bash
# GPT-OSS 120B decode-latency sweep across context length, batch 1.
#
# For each target context N, feeds a prompt of exactly N tokens (built by
# make_long_prompts.py, which compensates for the chat template) and decodes on
# top of it, so the reported per-token time is the cost of attending over ~N
# tokens of KV.
#
# TPOT is the MEDIAN of the decode-phase [FWD_PASS] ring samples, computed by
# summarize_seqlen_sweep.py -- see that file's docstring for why, and for the
# two methods that came before it and produced wrong numbers that looked fine.
# The short version: [FWD_PASS_TOTAL] avg_ms blends prefill into decode, and a
# two-point differential on total_ms subtracts two prefills whose ~1% variance
# is ~107 ms at ctx 4096, against ~1000 ms of signal.
#
# GEN_LO/GEN_HI are therefore no longer differential endpoints. Each context is
# still run at both lengths, for two reasons:
#
#   * The long run is what TPOT comes from -- more ring samples survive and
#     they sit further from the prefill boundary.
#   * The short run is a correctness check. Decode is greedy, so its
#     continuation must be a prefix of the long run's; if it is not, the extra
#     decode steps perturbed earlier ones and the summarizer says so.
#
# Prefill stays at 1 token/iteration (--max-num-batched-tokens 1). Batching it
# 16-wide is much faster per prompt token but changes the compiled shape and
# measurably changes decode too (3.87 vs 2.40 ms/tok at ctx 512), so it is not
# a free speedup for a decode benchmark. Cost: prefill is ~2.4ms x N, which is
# ~5 min at 128k. That is the price of an honest decode number.
#
# Runs are strictly serial: two --use-mirage processes contaminate each other
# even on different GPUs, because the build dir is hardcoded.
#
# Env:
#   MODEL_PATH            GPT-OSS 120B dir (required)
#   HIP_VISIBLE_DEVICES   target GPU (default 0)
#   SEQ_LENS              context lengths (default 512 ... 131072)
#   GEN_LO / GEN_HI       decode lengths, short then long (default 4 / 400)
#   PROMPT_DIR            output of make_long_prompts.py (default /tmp/prompts)
#   SWEEP_OUT             log/result dir (default outputs/gpt_oss/seqlen_sweep)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MIRAGE_HOME="${MIRAGE_HOME:-$ROOT}"
# MUST come first: there is another mirage checkout at ~/mirage that `import
# mirage` resolves to otherwise, and it is missing this tree's kernel APIs.
# Benchmarking it silently would measure the wrong repo.
export PYTHONPATH="$ROOT/python:${PYTHONPATH:-}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
MODEL_PATH="${MODEL_PATH:-/root/schowdha/models/gpt-oss-120b}"
# 65536 and 131072 are deliberately not in the default: prefill is 1 tok/iter,
# so 128k costs ~5 min of prefill per point. Pass SEQ_LENS explicitly for those.
SEQ_LENS="${SEQ_LENS:-512 1024 2048 4096 8192 32768}"
GEN_LO="${GEN_LO:-4}"
GEN_HI="${GEN_HI:-400}"
PROMPT_DIR="${PROMPT_DIR:-/tmp/prompts}"
SWEEP_OUT="${SWEEP_OUT:-$ROOT/outputs/gpt_oss/seqlen_sweep}"
DEMO="$ROOT/demo/gpt_oss/demo.py"
PAGE_SIZE=4096

mkdir -p "$SWEEP_OUT"

run_point() {
  local N=$1 GEN=$2 LOG=$3
  local MSL=$((N + GEN))
  local PAGES=$(( (MSL + PAGE_SIZE - 1) / PAGE_SIZE ))
  # Header edits do not trigger a rebuild, and the build dir is shared.
  rm -rf "$ROOT/demo/gpt_oss/permanent_output_dir"
  set +e
  timeout 3600 python3 "$DEMO" --use-mirage \
    --model-path "$MODEL_PATH" \
    --prompt-file "$PROMPT_DIR/prompt_${N}.txt" \
    --max-seq-length "$MSL" \
    --max-num-pages "$PAGES" \
    --page-size "$PAGE_SIZE" \
    --max-num-batched-tokens 1 \
    --ignore-eos \
    --save-tokens "${LOG%.log}_tokens.json" \
    > "$LOG" 2>&1
  local RC=$?
  set -e
  return $RC
}

for N in $SEQ_LENS; do
  if [[ ! -f "$PROMPT_DIR/prompt_${N}.txt" ]]; then
    echo "MISSING $PROMPT_DIR/prompt_${N}.txt -- run make_long_prompts.py first" >&2
    exit 1
  fi
  echo "=== Fleet ctx=$N  gen ${GEN_LO} then ${GEN_HI} ==="
  for GEN in "$GEN_LO" "$GEN_HI"; do
    LOG="$SWEEP_OUT/fleet_${N}_g${GEN}.log"
    if run_point "$N" "$GEN" "$LOG"; then
      grep -E '^\[(WALL|FWD_PASS_TOTAL)\]' "$LOG" | sed 's/^/  /'
    else
      echo "  FAILED gen=$GEN (see $LOG)"
    fi
  done
done

echo
echo "=== summary ==="
python3 "$ROOT/tests/ci-tests/summarize_seqlen_sweep.py" "$SWEEP_OUT"
