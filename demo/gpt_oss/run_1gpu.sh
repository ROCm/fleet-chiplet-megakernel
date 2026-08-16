#!/bin/bash
# Single-GPU gpt-oss-120b baseline. This is the 2.1 ms/token number that the
# 2-GPU path has to beat, so run it on the same machine and the same weights
# as the multi-GPU scripts -- not from a saved log.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env_common.sh

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-6}"

if [ "${KEEP_BUILD:-0}" != "1" ]; then
  rm -rf permanent_output_dir permanent_output_dir_rank*
fi

python demo.py --use-mirage \
  --max-seq-length "${MAX_SEQ_LENGTH:-128}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-16}" \
  --ignore-eos \
  --model-path "$MODEL_PATH" "$@"
echo "EXIT=$?"
