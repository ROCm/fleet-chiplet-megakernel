#!/usr/bin/env bash
# Perplexity vs sequence length for GPT-OSS 120B, 512 -> 32768 tokens.
#
# Scores a WikiText-2 prefix at each length with Mirage (MPK), and with the
# Torch reference wherever the reference can actually run. The reference
# attention in demo/gpt_oss/models/modeling_gpt_oss.py materializes a
# [64, n, n] float32 score matrix (:172), which is 64 GB per copy at 16k and
# 256 GB at 32k -- it OOMs well before MPK does. So Torch anchors the short
# end and the long end is MPK-only. TORCH_MAX_LEN sets that cutoff.
#
# Env:
#   MODEL_PATH            local GPT-OSS 120B dir or HF repo id
#   HIP_VISIBLE_DEVICES   target GPU (default 0)
#   PPL_LENS              lengths to sweep (default "512 1024 2048 4096 8192 16384 32768")
#   TORCH_MAX_LEN         longest length to also score with Torch (default 4096)
#   PPL_SWEEP_OUT         results dir (default outputs/gpt_oss/ppl_sweep)
set -uo pipefail   # NOT -e: a single length OOMing must not kill the sweep

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MIRAGE_HOME="${MIRAGE_HOME:-$ROOT}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export USE_FP8_ACT="${USE_FP8_ACT:-1}"
export PPL_MODE=1
MODEL_PATH="${MODEL_PATH:-${GPT_OSS_MODEL_PATH:-openai/gpt-oss-120b}}"
LENS="${PPL_LENS:-512 1024 2048 4096 8192 16384 32768}"
TORCH_MAX_LEN="${TORCH_MAX_LEN:-4096}"
OUT="${PPL_SWEEP_OUT:-$ROOT/outputs/gpt_oss/ppl_sweep}"
mkdir -p "$OUT"

DEMO="$ROOT/demo/gpt_oss/demo.py"
LOG="$OUT/sweep.log"
: > "$LOG"

echo "MODEL_PATH=$MODEL_PATH  GPU=$HIP_VISIBLE_DEVICES  lengths: $LENS" | tee -a "$LOG"
echo "Torch reference up to $TORCH_MAX_LEN tokens (longer OOMs the reference attention)" | tee -a "$LOG"

run_one() {  # $1=len $2=mpk|torch
  local n="$1" mode="$2"
  local tag="${mode}_${n}"
  local extra=()
  [ "$mode" = "mpk" ] && extra=(--use-mirage)
  echo "=== $mode n=$n ===" | tee -a "$LOG"
  timeout 3600 python3 "$DEMO" --model-path "$MODEL_PATH" "${extra[@]}" \
      --max-num-batched-tokens 1 --ppl-max-tokens "$n" \
      --ppl-out "$OUT/$tag.json" > "$OUT/$tag.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "  FAILED rc=$rc (see $OUT/$tag.log)" | tee -a "$LOG"
    tail -5 "$OUT/$tag.log" | sed 's/^/    /' | tee -a "$LOG"
  else
    grep -E "perplexity|mean entropy|top-1 accuracy|zero columns|per-worker-argmax|FWD_PASS_TOTAL" \
        "$OUT/$tag.log" | sed 's/^/  /' | tee -a "$LOG"
  fi
}

for n in $LENS; do
  run_one "$n" mpk
  if [ "$n" -le "$TORCH_MAX_LEN" ]; then run_one "$n" torch; fi
done

echo "" | tee -a "$LOG"
python3 "$ROOT/tests/ci-tests/summarize_ppl_sweep.py" "$OUT" 2>&1 | tee -a "$LOG"
echo "Results in $OUT (per-run logs: $OUT/<mode>_<len>.log)" | tee -a "$LOG"
