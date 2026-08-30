#!/bin/bash
# Interleaved A/B: alternate two .so files run-by-run in ONE session.
#
# Why interleave instead of tools/run_arm.sh twice
# ------------------------------------------------
# run_arm.sh runs one arm N times consecutively. That is fine for a large
# effect and wrong for a small one, because everything that drifts over a
# session -- clock/power state, HBM temperature, page placement, whatever else
# the box is doing -- is CONFOUNDED with the arm. Measure arm A this morning
# and arm B this afternoon and a 0.3% "regression" is indistinguishable from
# the afternoon.
#
# Alternating A B A B splits that drift across both arms instead of loading it
# onto one. It does not remove the drift; it stops the drift from having a
# preferred arm. The paired per-round deltas are then the thing to read: if the
# sign flips round to round, the effect is noise however consistent the means
# look.
#
# This is the instrument for a sub-1% delta on a kernel that is
# non-deterministic by design. For a delta that is supposed to be exactly zero
# -- a rename, a reformat -- prefer tools/check_isa_neutral.py, which settles
# it in a minute without running the model at all.
#
# Usage:
#   tools/ab_interleave.sh <name-a> <so-a> <name-b> <so-b> [rounds]
#
# Writes /tmp/<name>_run<N>.log, then:
#   python3 tools/compare_runs.py --arm <name-a> /tmp/<name-a>_run*.log \
#                                 --arm <name-b> /tmp/<name-b>_run*.log

set -e

NAME_A="${1:?usage: ab_interleave.sh <name-a> <so-a> <name-b> <so-b> [rounds]}"
SO_A="${2:?missing so-a}"
NAME_B="${3:?missing name-b}"
SO_B="${4:?missing so-b}"
ROUNDS="${5:-3}"

MK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LIVE_SO="$MK_DIR/generated/gpt_oss_120b.so"
MODEL="${MODEL_PATH:-/home/claudeuser/models/gpt-oss-120b}"

# Load-bearing, same as run_arm.sh: a shorter prompt gives artificially low
# numbers. Do not "simplify" them.
PROMPT="Tell me the history of america"
MAXLEN=512

# The live .so is about to be overwritten repeatedly. Put it back on exit so an
# interrupted A/B does not leave the tree holding whichever arm ran last.
RESTORE="$(mktemp /tmp/ab_live_XXXX.so)"
cp "$LIVE_SO" "$RESTORE"
cleanup() {
    cp "$RESTORE" "$LIVE_SO" && rm -f "$RESTORE"
    rm -f "${STAGE_A:-}" "${STAGE_B:-}" 2>/dev/null || true
    rm -f /tmp/gpucore* "$MK_DIR"/gpucore* 2>/dev/null || true
    echo "[ab] restored original $LIVE_SO"
}
trap cleanup EXIT

for so in "$SO_A" "$SO_B"; do
    [ -f "$so" ] || { echo "[ab] no such .so: $so" >&2; exit 2; }
done
echo "[ab] A=$NAME_A md5=$(md5sum "$SO_A" | cut -d' ' -f1)"
echo "[ab] B=$NAME_B md5=$(md5sum "$SO_B" | cut -d' ' -f1)"

# Stage both arms outside the tree BEFORE the loop. The natural way to invoke
# this is with one arm already installed as the live .so -- and then `cp` sees
# source and destination as the same file, fails, and `set -e` aborts the whole
# A/B one run in. Staging first also means the live .so being overwritten mid
# run cannot corrupt an arm we still need.
STAGE_A="$(mktemp /tmp/ab_a_XXXX.so)"
STAGE_B="$(mktemp /tmp/ab_b_XXXX.so)"
cp "$SO_A" "$STAGE_A"
cp "$SO_B" "$STAGE_B"

one_run() {
    local name="$1" so="$2" i="$3"
    local log="/tmp/${name}_run${i}.log"
    cp "$so" "$LIVE_SO"
    echo "[ab] round $i: $name -> $log"
    # Over ~3 min is hanging, not slow. Bound it and move on.
    cd "$MK_DIR" && HIP_VISIBLE_DEVICES=0 timeout 280 python3 \
        demo_gpt_oss_120b.py --model-path "$MODEL" \
        --prompt "$PROMPT" --max-seq-length "$MAXLEN" > "$log" 2>&1 || {
        echo "[ab]   run exited $? (see $log)"
        pkill -9 -f demo_gpt_oss_120b 2>/dev/null || true
        sleep 3
    }
    grep -h "Non-outlier avg" "$log" || echo "[ab]   (no avg line)"
    # Cores from a crashed run fill the disk fast (it sits at ~95%).
    rm -f /tmp/gpucore* "$MK_DIR"/gpucore* 2>/dev/null || true
}

for i in $(seq 1 "$ROUNDS"); do
    # Alternate which arm leads each round, so neither arm always occupies the
    # cold slot right after the other's teardown.
    if [ $((i % 2)) -eq 1 ]; then
        one_run "$NAME_A" "$STAGE_A" "$i"
        one_run "$NAME_B" "$STAGE_B" "$i"
    else
        one_run "$NAME_B" "$STAGE_B" "$i"
        one_run "$NAME_A" "$STAGE_A" "$i"
    fi
done

echo "[ab] done. Compare with:"
echo "  python3 tools/compare_runs.py --arm $NAME_A /tmp/${NAME_A}_run*.log \\"
echo "                                --arm $NAME_B /tmp/${NAME_B}_run*.log"
