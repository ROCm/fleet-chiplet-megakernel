#!/usr/bin/env bash
# Run one vLLM arm (stock or fleet_mk) and leave the GPU clean for the next one.
#
# Why this exists rather than a bare python invocation: a timed-out or hung run
# leaves an orphaned VLLM::EngineCore child holding the whole KV pool (227 GiB),
# and killing the parent does NOT reap it. The next run then dies at init with
# "Free memory on device cuda:0 ... is less than desired GPU memory utilization",
# which reads like a fleet_mk bug and is not one. Two runs were lost to that
# before this script existed. A fixed `sleep` is not enough either -- VRAM comes
# back asynchronously after SIGKILL, so we poll until it is actually free.
#
# Usage: run_vllm_arm.sh <arm> <max_tokens> [timeout_s]
#   arm: "stock" (VLLM_PLUGINS empty) or "fleet_mk"
# Env passthrough: FLEET_MK_SO pins a specific .so (for bisecting builds).
# Writes /tmp/vllm_<arm>_<max_tokens>.log and echoes the rc.
set -u

ARM="${1:?arm: stock|fleet_mk}"
NTOK="${2:-10}"
TMO="${3:-150}"
MODEL="${FLEET_MK_MODEL:-/home/claudeuser/models/gpt-oss-120b}"
LOG="/tmp/vllm_${ARM}_${NTOK}.log"

case "$ARM" in
  stock)    PLUGINS="" ;;
  fleet_mk) PLUGINS="fleet_mk" ;;
  *) echo "unknown arm: $ARM (want stock|fleet_mk)" >&2; exit 2 ;;
esac

reap() {
  pkill -9 -f 'VLLM::Engine[C]ore' 2>/dev/null
  pkill -9 -f 'fleet_megakernel[_]vllm' 2>/dev/null
  # Poll VRAM rather than sleeping a guessed interval. rocm-smi reports the
  # allocation percentage per device; anything under 5% is "nobody is holding
  # the pool". 60 x 2 s is far longer than a teardown has ever taken.
  for _ in $(seq 60); do
    sleep 2
    local used
    used=$(rocm-smi --showmemuse 2>/dev/null \
             | awk '/GPU Memory Allocated/ {print $NF; exit}')
    [ -z "$used" ] && break
    [ "$used" -lt 5 ] 2>/dev/null && break
  done
  rm -f /tmp/gpucore* gpucore* 2>/dev/null
}

reap
cd "$(dirname "$0")/.." || exit 1
ulimit -c 0
HIP_VISIBLE_DEVICES=0 \
VLLM_PLUGINS="$PLUGINS" \
FLEET_MK_TEMP="${FLEET_MK_TEMP:-0}" \
FLEET_MK_MAX_TOKENS="$NTOK" \
FLEET_MK_MODEL="$MODEL" \
  timeout "$TMO" /home/claudeuser/venv-vllm027/bin/python3 \
  -m fleet_megakernel_vllm.harness > "$LOG" 2>&1
RC=$?
echo "rc=$RC" >> "$LOG"
reap
echo "$ARM ntok=$NTOK rc=$RC log=$LOG"
