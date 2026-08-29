#!/bin/bash
# Run one benchmark arm N times against a given .so, sequentially.
#
# Fleet MK is non-deterministic by design (MoE float reduction order), so a single
# run cannot resolve a sub-1% latency delta -- always run >=3 and feed the logs
# to tools/compare_runs.py, which reports within-arm spread next to the
# between-arm delta.
#
# Usage:
#   tools/run_arm.sh <arm-name> <path-to-.so> [runs]
#
# Writes /tmp/<arm-name>_run<N>.log. Then:
#   python3 tools/compare_runs.py --arm base /tmp/base_run*.log \
#                                 --arm new  /tmp/new_run*.log
#
# NEVER run two arms concurrently -- they contend for the GPU and inflate both.

set -e

ARM="${1:?usage: run_arm.sh <arm-name> <so-path> [runs]}"
SO="${2:?usage: run_arm.sh <arm-name> <so-path> [runs]}"
RUNS="${3:-3}"

FLEET_MK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LIVE_SO="$FLEET_MK_DIR/generated/gpt_oss_120b.so"
MODEL="${MODEL_PATH:-/home/claudeuser/models/gpt-oss-120b}"

# The prompt and --max-seq-length are load-bearing: shorter prompts give
# artificially low numbers. Do not "simplify" them.
PROMPT="Tell me the history of america"
MAXLEN=512

if [ "$(readlink -f "$SO")" != "$(readlink -f "$LIVE_SO")" ]; then
    echo "[run_arm] installing $SO -> $LIVE_SO"
    cp "$SO" "$LIVE_SO"
fi
echo "[run_arm] arm=$ARM md5=$(md5sum "$LIVE_SO" | cut -d' ' -f1) runs=$RUNS"

for i in $(seq 1 "$RUNS"); do
    LOG="/tmp/${ARM}_run${i}.log"
    echo "[run_arm] $ARM run $i/$RUNS -> $LOG"
    # A run over ~3 min is hanging, not slow. Bound it and move on.
    cd "$FLEET_MK_DIR" && HIP_VISIBLE_DEVICES=0 timeout 280 python3 \
        demo_gpt_oss_120b.py --model-path "$MODEL" \
        --prompt "$PROMPT" --max-seq-length "$MAXLEN" > "$LOG" 2>&1 || {
        echo "[run_arm] run $i exited $? (see $LOG)"
        pkill -9 -f demo_gpt_oss_120b 2>/dev/null || true
        sleep 3
    }
    grep -h "Non-outlier avg" "$LOG" || echo "[run_arm]   (no avg line)"
done

# Core dumps from a crashed run fill the disk fast (it sits at ~94%).
rm -f /tmp/gpucore* "$FLEET_MK_DIR"/gpucore* 2>/dev/null || true
echo "[run_arm] done: /tmp/${ARM}_run*.log"
