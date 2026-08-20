#!/usr/bin/env bash
# vLLM decode-latency sweep across context length, batch 1. Companion arm to
# run_gpt_oss_seqlen_sweep.sh; feed both to plot_seqlen_sweep.py.
#
# `vllm bench latency` reports END-TO-END latency for one batch, prefill
# included, so TPOT is a two-point difference at the same input length:
# (latency@HI - latency@LO) / (HI - LO). Both points pay the same prefill, so
# it cancels. This is the same reason the Fleet sweep does NOT use a
# differential -- there, per-iteration decode samples are available directly
# and differencing two totals would subtract two noisy prefills instead.
#
# Two environment notes, both of which silently change the number:
#
#   * PYTHONNOUSERSITE=1 is set below. A user-site `transformers` newer than
#     the one vLLM pins makes tokenizer init fail outright
#     (`TokenizersBackend has no attribute all_special_tokens_extended`).
#   * The MXFP4 MoE backend is whatever the installed vLLM selects, and on
#     ROCm several versions have no AITER FP4 path at all -- they dequantize
#     to BF16 and run BF16 GEMMs. Grep the log for the "Using <X> backend"
#     line and report it alongside the latency; "Triton" is not native FP4.
#
# Env:
#   MODEL_PATH            GPT-OSS 120B dir (required)
#   HIP_VISIBLE_DEVICES   target GPU (default 0)
#   SEQ_LENS              context lengths (default 512 ... 32768)
#   LO / HI               output lengths for the difference (default 1 / 129)
#   OUT                   result dir (default outputs/gpt_oss/vllm_sweep)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${MODEL_PATH:?set MODEL_PATH}"
OUT="${OUT:-$ROOT/outputs/gpt_oss/vllm_sweep}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export PYTHONNOUSERSITE=1
LO="${LO:-1}"
HI="${HI:-129}"
ITERS="${ITERS:-10}"
WARM="${WARM:-3}"
SEQ_LENS="${SEQ_LENS:-512 1024 2048 4096 8192 32768}"
mkdir -p "$OUT"

for N in $SEQ_LENS; do
  for G in "$LO" "$HI"; do
    J="$OUT/vllm_${N}_o${G}.json"
    L="$OUT/vllm_${N}_o${G}.log"
    [[ -s "$J" ]] && { echo "skip $J"; continue; }
    echo "=== vllm ctx=$N output-len=$G ==="
    timeout 3600 vllm bench latency --model "$MODEL" \
      --input-len "$N" --output-len "$G" --batch-size 1 \
      --num-iters "$ITERS" --num-iters-warmup "$WARM" \
      --max-model-len $((N + G + 16)) \
      --gpu-memory-utilization "${GPU_MEM_UTIL:-0.6}" \
      --disable-detokenize \
      --output-json "$J" > "$L" 2>&1
    echo "  rc=$? $(grep -oE 'Avg latency: [0-9.]+ seconds' "$L" | tail -1)"
    grep -oE 'Using [A-Za-z]+ backend' "$L" | tail -1 | sed 's/^/  MoE /'
  done
done

echo
echo "=== summary ==="
python3 "$ROOT/tests/ci-tests/plot_seqlen_sweep.py" \
  --fleet "${FLEET_SUMMARY:-$ROOT/outputs/gpt_oss/seqlen_sweep/summary.json}" \
  --vllm "$OUT" --vllm-lo "$LO" --vllm-hi "$HI"
