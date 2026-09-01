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
#
# Host-coupled arms: DRIVER_A / DRIVER_B
# --------------------------------------
# Swapping only the .so assumes the driver is the same for both arms. That
# holds for every flag whose effect is confined to device code, and it is FALSE
# for a host-coupled one -- MPK_OPROJ_KMAJOR is the live example: the flag
# rewrites the kernel's LDS address arithmetic AND the driver must repack the
# O-proj tile to match. Pair the new driver with the old .so and you have not
# measured the baseline, you have measured the broken half-configuration, which
# does not crash and does not fail to build -- every lane reads a valid byte of
# the tile, just the wrong one. It comes back as fluent garbage and a first
# token that differs across arms.
#
# So for those, give each arm its own driver:
#   DRIVER_A=/tmp/old_demo.py DRIVER_B=demo_gpt_oss_120b.py \
#       tools/ab_interleave.sh base /tmp/base.so kmajor /tmp/kmajor.so 2
#
# Unset means "use the driver already in the tree" -- the previous behaviour,
# so existing invocations are unaffected. The live driver is restored on exit
# exactly like the live .so.
#
# Either way, read the per-run "iter 1" line this prints. Two arms that
# disagree on the first token are not two measurements of the same model.

set -e

NAME_A="${1:?usage: ab_interleave.sh <name-a> <so-a> <name-b> <so-b> [rounds]}"
SO_A="${2:?missing so-a}"
NAME_B="${3:?missing name-b}"
SO_B="${4:?missing so-b}"
ROUNDS="${5:-3}"

MK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LIVE_SO="$MK_DIR/generated/gpt_oss_120b.so"
LIVE_DRIVER="$MK_DIR/demo_gpt_oss_120b.py"
MODEL="${MODEL_PATH:-/home/claudeuser/models/gpt-oss-120b}"

# Load-bearing, same as run_arm.sh: a shorter prompt gives artificially low
# numbers. Do not "simplify" them.
PROMPT="Tell me the history of america"
MAXLEN=512

# The live .so is about to be overwritten repeatedly. Put it back on exit so an
# interrupted A/B does not leave the tree holding whichever arm ran last.
RESTORE="$(mktemp /tmp/ab_live_XXXX.so)"
cp "$LIVE_SO" "$RESTORE"
RESTORE_DRIVER="$(mktemp /tmp/ab_live_XXXX.py)"
cp "$LIVE_DRIVER" "$RESTORE_DRIVER"
cleanup() {
    cp "$RESTORE" "$LIVE_SO" && rm -f "$RESTORE"
    cp "$RESTORE_DRIVER" "$LIVE_DRIVER" && rm -f "$RESTORE_DRIVER"
    rm -f "${STAGE_A:-}" "${STAGE_B:-}" 2>/dev/null || true
    rm -f "${STAGE_DA:-}" "${STAGE_DB:-}" 2>/dev/null || true
    rm -f /tmp/gpucore* "$MK_DIR"/gpucore* 2>/dev/null || true
    echo "[ab] restored original $LIVE_SO and $LIVE_DRIVER"
}
trap cleanup EXIT

for so in "$SO_A" "$SO_B"; do
    [ -f "$so" ] || { echo "[ab] no such .so: $so" >&2; exit 2; }
done
for drv in "${DRIVER_A:-}" "${DRIVER_B:-}"; do
    [ -z "$drv" ] || [ -f "$drv" ] || { echo "[ab] no such driver: $drv" >&2; exit 2; }
done
echo "[ab] A=$NAME_A md5=$(md5sum "$SO_A" | cut -d' ' -f1) driver=${DRIVER_A:-<tree>}"
echo "[ab] B=$NAME_B md5=$(md5sum "$SO_B" | cut -d' ' -f1) driver=${DRIVER_B:-<tree>}"

# Stage both arms outside the tree BEFORE the loop. The natural way to invoke
# this is with one arm already installed as the live .so -- and then `cp` sees
# source and destination as the same file, fails, and `set -e` aborts the whole
# A/B one run in. Staging first also means the live .so being overwritten mid
# run cannot corrupt an arm we still need.
STAGE_A="$(mktemp /tmp/ab_a_XXXX.so)"
STAGE_B="$(mktemp /tmp/ab_b_XXXX.so)"
cp "$SO_A" "$STAGE_A"
cp "$SO_B" "$STAGE_B"

# Same staging argument for the drivers: one arm's driver is typically the one
# already in the tree, and cp would see source and destination as one file.
STAGE_DA=""
STAGE_DB=""
if [ -n "${DRIVER_A:-}" ]; then
    STAGE_DA="$(mktemp /tmp/ab_da_XXXX.py)"; cp "$DRIVER_A" "$STAGE_DA"
fi
if [ -n "${DRIVER_B:-}" ]; then
    STAGE_DB="$(mktemp /tmp/ab_db_XXXX.py)"; cp "$DRIVER_B" "$STAGE_DB"
fi

one_run() {
    local name="$1" so="$2" i="$3" drv="${4:-}"
    local log="/tmp/${name}_run${i}.log"
    cp "$so" "$LIVE_SO"
    [ -z "$drv" ] || cp "$drv" "$LIVE_DRIVER"
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
    # A host-coupled arm fails as wrong numerics, not as a crash, so surface the
    # first token every run rather than only at compare time.
    grep -h -m1 "iter 1: next_token" "$log" || echo "[ab]   (no iter 1 line)"
    # Cores from a crashed run fill the disk fast (it sits at ~95%).
    rm -f /tmp/gpucore* "$MK_DIR"/gpucore* 2>/dev/null || true
}

for i in $(seq 1 "$ROUNDS"); do
    # Alternate which arm leads each round, so neither arm always occupies the
    # cold slot right after the other's teardown.
    if [ $((i % 2)) -eq 1 ]; then
        one_run "$NAME_A" "$STAGE_A" "$i" "$STAGE_DA"
        one_run "$NAME_B" "$STAGE_B" "$i" "$STAGE_DB"
    else
        one_run "$NAME_B" "$STAGE_B" "$i" "$STAGE_DB"
        one_run "$NAME_A" "$STAGE_A" "$i" "$STAGE_DA"
    fi
done

echo "[ab] done. Compare with:"
echo "  python3 tools/compare_runs.py --arm $NAME_A /tmp/${NAME_A}_run*.log \\"
echo "                                --arm $NAME_B /tmp/${NAME_B}_run*.log"
