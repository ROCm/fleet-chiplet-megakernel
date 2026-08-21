#!/bin/bash
# Multi-prompt / long-output correctness sweep.
#
# Runs the same prompt set through a given launcher (run_1gpu.sh, run_mp2.sh,
# ...) and dumps each prompt's generated token_ids to JSON. compare_tokens.py
# then diffs a multi-GPU run against the 1-GPU reference.
#
# Every prompt costs a full megakernel recompile (persistent_kernel.py compiles
# unconditionally), so this is minutes per prompt, not seconds. That is why it
# writes results incrementally -- a run that dies at prompt 5 still leaves the
# first four comparable.
#
#   ./run_correctness_suite.sh run_1gpu.sh 1gpu
#   ./run_correctness_suite.sh run_mp2.sh  mp2
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

LAUNCHER="${1:?usage: run_correctness_suite.sh <launcher.sh> <tag>}"
TAG="${2:?usage: run_correctness_suite.sh <launcher.sh> <tag>}"

OUT_DIR="${OUT_DIR:-/tmp/gptoss_correctness}"
mkdir -p "$OUT_DIR"

# Long output: 256 new tokens is well past the point where a subtly wrong
# reduction shows up as divergence, while 512 positions keeps the KV cache
# inside the 16x4096-page budget.
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
# demo.py caps the token dump at 100 for the CI test; raise it or a divergence
# past token 100 never reaches the JSON and the comparison silently passes.
export MAX_SAVE_TOKENS="${MAX_SAVE_TOKENS:-$((MAX_NEW_TOKENS + 8))}"

PROMPTS=(
  "The capital of France is"
  "Write a short paragraph explaining why the sky appears blue."
  "List the first ten prime numbers and explain what makes a number prime."
  "Explain the difference between a stack and a queue in computer science."
)

for i in "${!PROMPTS[@]}"; do
  p="${PROMPTS[$i]}"
  dst="$OUT_DIR/${TAG}_p${i}.json"
  log="$OUT_DIR/${TAG}_p${i}.log"
  echo "=== [$TAG] prompt $i: $p"
  ./"$LAUNCHER" --prompt "$p" --save-tokens "$dst" > "$log" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$dst" ]; then
    echo "    FAILED (rc=$rc) -- see $log"
  else
    echo "    ok -> $dst"
  fi
done
