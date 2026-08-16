#!/bin/bash
# Shared environment for the gpt-oss-120b megakernel runs.
# Sourced by run_1gpu.sh / run_mp2.sh / run_mp2_ep.sh -- not run directly.

FLEET_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# MIRAGE_HOME decides which tree the megakernel compiles against, and
# PYTHONPATH decides which one `import mirage` resolves to. A stale editable
# install (~/.local/.../__editable__.mirage_project*.pth) points at a
# *different* checkout, so both must be pinned or the run silently exercises
# the wrong codebase.
export MIRAGE_HOME="$FLEET_HOME"
export PYTHONPATH="$FLEET_HOME/python:${PYTHONPATH:-}"

export USE_FP8_ACT=1

export MODEL_PATH="${MODEL_PATH:-/root/schowdha/models/gpt-oss-120b}"

# rocSHMEM + the MPI it was built against. rocSHMEM's IPC backend only needs
# MPI for bootstrap (rank exchange), not for the data path.
export ROCSHMEM_INC_PATH="${ROCSHMEM_INC_PATH:-/home/claudeuser/rocshmem/include}"
export ROCSHMEM_LIB_PATH="${ROCSHMEM_LIB_PATH:-/home/claudeuser/rocshmem/lib}"
if [ -d /home/claudeuser/ompi/lib ]; then
  export MPI_INC_PATH="${MPI_INC_PATH:-/home/claudeuser/ompi/include}"
  export MPI_LIB_PATH="${MPI_LIB_PATH:-/home/claudeuser/ompi/lib}"
  export PATH="/home/claudeuser/ompi/bin:$PATH"
  export LD_LIBRARY_PATH="/home/claudeuser/ompi/lib:/home/claudeuser/ucx/lib:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
else
  export MPI_INC_PATH="${MPI_INC_PATH:-/usr/lib/x86_64-linux-gnu/openmpi/include}"
  export MPI_LIB_PATH="${MPI_LIB_PATH:-/usr/lib/x86_64-linux-gnu/openmpi/lib}"
  export LD_LIBRARY_PATH="$MPI_LIB_PATH:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
fi

# Every knob the multi-GPU path reads, forwarded to both ranks by mpirun.
# Listing them unconditionally is deliberate: -x on an unset variable is a
# no-op, so a knob set in the caller's shell reaches rank 1 without editing
# this list.
MPK_FORWARD_VARS=(
  MIRAGE_HOME PYTHONPATH USE_FP8_ACT MODEL_PATH
  HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
  ROCSHMEM_INC_PATH ROCSHMEM_LIB_PATH MPI_INC_PATH MPI_LIB_PATH
  LD_LIBRARY_PATH PATH
  ROCSHMEM_MAX_NUM_CONTEXTS
  MOE_EP EP_NOAR EP_NOSLICE EP_SLOT EP_FOLD_RANK
  W13_OPW W2_OPW
  MPK_EP_ABLATE MPK_EP_SIG_DBG MPK_WORKER_STATE MPK_EP9_ONLY MPK_EP_SKEW_PROBE MPK_DEVICE_TIMING
  MPK_W2_HALFK MPK_W2_SPLITK MPK_EP_WAIT_AT_USE MPK_MOE_NOPAD
  MPK_W2_CONSUMER_GATE MPK_NO_LAYER_BARRIER MPK_WIDE_FP8_QUANT MPK_ABLATE_W13_QUANT
  MPK_OPROJ_NO_WB MPK_OPROJ_INNER_TIMING MPK_MOE_INNER_TIMING
  MPK_XCD_LOCAL_BARRIER MPK_MFMA_PINGPONG_SCHED MPK_OPROJ_TREE_BARRIER
  MPK_WORKERS_PER_XCD MPK_DRAIN_OVERLAP MPK_W13_LINEAR_LOAD MPK_W2_LINEAR_LOAD
  MPK_W13_T1_LINEAR_LOAD MPK_MOE_INNER_WIDE MPK_NDEBUG MPK_NO_SW_MASK MPK_LSE_LOG_BUG MPK_INTERLAYER_FENCE MPK_ONE_NORM_WRITER
  MPK_W13_T1_EARLY_SCALE_LOAD MPK_QKV_PREFETCH_SCALES MPK_W13_T1_SPLIT_LDS_STAGE
  MPK_SYS_POLL_LOAD MPK_QKV_GATE_NO_AGENT_FENCE MPK_NARROW_GATE_POLL MPK_W13_BIAS_PREFETCH
  MPK_W13_BIAS_PF_T0_ONLY MPK_LEAN_ARRIVE MPK_W2_ONLY_ARRIVE MPK_ROUTER_FUSED_DP MPK_QKV_PF_WAVE_SPLIT MPK_OPROJ_LEAN_ACQUIRE MPK_MOE_PAD_ROUND MPK_W13_PREQUANT MPK_OPROJ_ARRIVE_ONLY MPK_ROUTING_LANE_RELEASE MPK_ROUTING_DERIVED_EPOCH MPK_MERGE_KV_OUTER MPK_MERGE_TWO_PASS
  MPK_W13_T1_SPLIT_NOSUFFIX MPK_W13_T1_SPLIT_STAGE_PROBE MPK_W13_T1_SPLIT_PROBE_DRAIN MPK_W13_T1_SPLIT_READ_PROBE MPK_W13_T1_SPLIT_CMP MPK_W13_T1_STAGED_ROWS_OVERRIDE MPK_W13_T1_STAGE_CHUNKS_OVERRIDE
  ATTN_DP MPK_INLINE_AR2 INLINE_AR2 FUSED_MGPU_NOAR
  AR1_ELEMS_PER_BLOCK AR1_TARGET_GRID
  AR2_ELEMS_PER_BLOCK AR2_TARGET_GRID AR2_PREFETCH AR2_PREFETCH_BLOCKS
  MPK_SPAN_TIMING MPK_SUBPHASE_TIMING MPK_MOE_SUBPHASE
  MPK_OVERLAP_XGPU MPK_DEVICE_ACCUM
  FUSE_FULL_LAYER FUSE_FULL_LAYER_MGPU FUSE_QKV_ATTN FUSE_OPROJ_MOE
  FUSE_OPROJ_TOPK FUSE_TAIL
  PRECOMPUTED_DISPATCH USE_GANG PPL_MODE PPL_MXFP4_MATCH
  MLP_DBG MLP_FINAL_IDX
  MAX_SAVE_TOKENS
)

mpk_x_args() {
  local v
  for v in "${MPK_FORWARD_VARS[@]}"; do printf -- '-x %s ' "$v"; done
}
