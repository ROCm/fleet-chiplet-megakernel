#!/bin/bash
# 4-GPU gpt-oss-120b: DP attention + EP MoE, full fusion, ONE expert per rank.
#
# Same configuration as run_mp2_dp_ep_fused.sh at world size 4. With top_k=4
# and EP_SLOT=1 the activated list splits 1/1/1/1, so each rank runs exactly
# one expert -- the finest slice this model admits at bs=1.
#
# Two things get worse at world 4, both in Phase 9 and both structural:
#   1. The direct peer store is EP_WORLD_SIZE == 2 only (one peer, one delta).
#      World 4 falls back to the staged rocSHMEM putmem_signal, once per peer,
#      i.e. 3 sequential work-group collectives instead of one fused store.
#   2. The 9d wait is per-peer, so a worker waits on 3 signal lines and the
#      slowest rank sets the layer time. Straggler exposure grows with ranks.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env_common.sh

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-4,5,6,7}"
export ROCSHMEM_MAX_NUM_CONTEXTS="${ROCSHMEM_MAX_NUM_CONTEXTS:-4}"
export ATTN_DP=1
export MOE_EP=1
export EP_SLOT="${EP_SLOT:-1}"
export W13_OPW="${W13_OPW:-64}"
export FUSE_FULL_LAYER="${FUSE_FULL_LAYER:-1}"
export PRECOMPUTED_DISPATCH="${PRECOMPUTED_DISPATCH:-1}"

if [ "${KEEP_BUILD:-0}" != "1" ]; then
  rm -rf permanent_output_dir permanent_output_dir_rank*
fi

mpirun -np 4 --tag-output --allow-run-as-root \
  $(mpk_x_args) \
  python demo.py --use-mirage \
    --max-seq-length "${MAX_SEQ_LENGTH:-128}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-16}" \
    --ignore-eos \
    --model-path "$MODEL_PATH" "$@"
echo "MPIRUN_EXIT=$?"
