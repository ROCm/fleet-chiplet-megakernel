#!/bin/bash
# Capture a rocprofv3 kernel + HIP-runtime trace of one decode run, for
# tools/trace_dispatch_gaps.py.
#
# Why a script and not a command line
# -----------------------------------
# Three things about this capture are load-bearing and were each got wrong at
# least once:
#
#   1. --kernel-trace AND --hip-runtime-trace. The kernel trace alone gives
#      dispatch begin/end, from which the between-dispatch gap falls out, but
#      not what the host was doing during it. Both, on one clock, is the whole
#      point -- a difference of two instruments is an inference, not a
#      measurement.
#   2. --output-format csv. The default also writes a .pftrace, which is 24 MB
#      per capture, is not read by any tool here, and this box sits at 95% disk.
#   3. The prompt and --max-seq-length are the ones from CLAUDE.md. A shorter
#      prompt gives an artificially low number, so a trace taken with a
#      different one cannot be compared against any latency this project has
#      recorded.
#
# The FLEET_MK_PERSIST value is passed through, so the N=1 and N=8 arms are the
# same binary and the same command, differing only in the knob under test.
#
# Usage:
#   tools/capture_trace.sh <arm-name> [persist-N]
#
# Writes /tmp/trace_<arm-name>/ and /tmp/trace_<arm-name>.log (the run's own
# stdout -- read it to confirm tok1=35644 and coherent text BEFORE trusting any
# timing out of the trace). Then:
#
#   python3 tools/trace_dispatch_gaps.py --arm n1 /tmp/trace_n1 \
#                                        --arm n8 /tmp/trace_n8
#
# Tracing inflates the run. Never quote a latency number from a traced run --
# same rule as FLEET_MK_WORKER_STATE. The trace is for the SHAPE (dispatch
# count, where time sits), not the magnitude.

set -e

ARM="${1:?usage: capture_trace.sh <arm-name> [persist-N]}"
PERSIST="${2:-1}"

FLEET_MK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${MODEL_PATH:-/home/claudeuser/models/gpt-oss-120b}"
OUT="/tmp/trace_${ARM}"
LOG="/tmp/trace_${ARM}.log"

PROMPT="Tell me the history of america"
MAXLEN=512

rm -rf "$OUT"
mkdir -p "$OUT"

echo "[capture] arm=$ARM persist=$PERSIST"
echo "[capture] so md5=$(md5sum "$FLEET_MK_DIR/generated/gpt_oss_120b.so" | cut -d' ' -f1)"

# Core dumps from a crashed traced run fill the disk fast.
ulimit -c 0

cd "$FLEET_MK_DIR"
HIP_VISIBLE_DEVICES=0 FLEET_MK_PERSIST="$PERSIST" timeout 280 rocprofv3 \
    --kernel-trace \
    --hip-runtime-trace \
    --output-format csv \
    -d "$OUT" -o "$ARM" \
    -- python3 demo_gpt_oss_120b.py --model-path "$MODEL" \
       --prompt "$PROMPT" --max-seq-length "$MAXLEN" > "$LOG" 2>&1 || {
    echo "[capture] exited $? (see $LOG)"
    pkill -9 -f demo_gpt_oss_120b 2>/dev/null || true
    sleep 3
}

rm -f /tmp/gpucore* "$FLEET_MK_DIR"/gpucore* 2>/dev/null || true

echo "[capture] $OUT:"
ls -la "$OUT" || true
echo "[capture] run log: $LOG  -- CHECK tok1 and the generated text there."
