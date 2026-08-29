import os

import torch

# This function returns the shared memory limit (in bytes)
# for the given GPU hardware architecture
def get_shared_memory_capacity(target_cc):
    if target_cc == 80:
        # A100 GPUs
        return 163 * 1024
    elif target_cc == 86:
        # A5000 GPUs
        return 99 * 1024
    elif target_cc == 89:
        # A6000 GPUs
        return 99 * 1024
    elif target_cc == 90:
        # H100 GPUs
        return 223 * 1024
    elif target_cc == 100:
        # B200 GPUs
        return 227 * 1024
    else:
        assert False, "Unsupported compute capacity: {}".format(target_cc)


def get_scheduler(sm_cnt, worker):
    scheduler = sm_cnt - worker
    assert scheduler > 0, "worker count is not compatible with sm count on"
    "the GPU"
    return sm_cnt - worker

# MAX_NUM_WORKERS must match C++ runtime_header.h (used in MPK device asserts).
# Increased to 304 to support full CU utilization on AMD MI300X (304 CUs)
MAX_NUM_WORKERS = 304


# Shipped optimizations that only build at batch 1. Each of these is a
# *compile error* on the packed multi-row MoE path, not a mistune, so an
# unconditional default would break every bs>1 build:
#
#   MPK_W13_PREQUANT           static_assert "single-token only (BATCH_SIZE
#                              == 1)" -- its publication is one row, and the
#                              router's writer election has no notion of which
#                              tokens an expert drew.
#   MPK_W13_T1_SPLIT_LDS_STAGE static_assert "split W13 tile-1 stage exceeds
#                              the worker's LDS budget" -- the stage buffer is
#                              11 KiB x 4 waves on top of a W13 tile that the
#                              multi-row path has already widened.
#
# Only the DEFAULT is narrowed. Setting one explicitly at bs>1 still reaches
# the static_assert, which is the intended loud failure rather than a knob
# that silently does nothing.
_BS1_ONLY_OPTS = (
    "MPK_W13_PREQUANT",
    "MPK_W13_T1_SPLIT_LDS_STAGE",
    # Rides on MPK_W13_PREQUANT (it reorders that flag's handoff), so it has
    # to go off wherever that one does.
    "MPK_W13_T0_COUNTED_HANDOFF",
    # Canonical K128 recycle is batch-1 / OPW=128 only (kernel static_asserts).
    "MPK_W13_KMAJOR_RECYCLE",
    # Rides on the batch-1-only recycle schedule.
    "MPK_W13_T1_BIAS_EARLY",
    # Redline-style next-group LM-head pipeline has a batch-1 LDS layout.
    "MPK_LM_HEAD_GROUP_PIPELINE",
)


def mpk_opt(name, batch_size=1):
    """Is a shipped optimization enabled? Default ON; MPK_<name>=0 disables.

    Only for the knobs that ship on. Ablations, probes and fault injection are
    read directly with a "0" default at their own sites.

    `batch_size` narrows the default for the bs=1-only flags above; pass the
    build's max_num_batched_tokens wherever it is known.
    """
    default = "0" if (batch_size != 1 and name in _BS1_ONLY_OPTS) else "1"
    return int(os.environ.get(name, default)) == 1


def mpk_w13_prequant(batch_size):
    """Is the FP8 W13 activation prequant active for this build?

    Separate from mpk_opt() because it has a second dependency: the kernel
    #errors without MPK_ROUTER_FUSED_DP (it publishes from that flag's elected
    norm writer), so disabling that disables this.

    Callers outside the compiler need this because the flag changes the wire
    format of `rmsnorm_out_moe` from BF16 to FP8 E4M3 + one E8M0 byte per 128
    elements. Anything reading that buffer back on the host has to know which.
    """
    if not mpk_opt("MPK_ROUTER_FUSED_DP"):
        return False
    return mpk_opt("MPK_W13_PREQUANT", batch_size)


def mpk_workers_per_xcd():
    """Worker blocks per XCD on MI350.

    31 is the shipped default: 31*8 workers + 8 scheduler blocks = 256, exactly
    the CU count. 30 was the MI300X-compatible value and left 8 CUs idle for
    the whole run; it is still reachable as MPK_WORKERS_PER_XCD=30. 32 would
    make a scheduler share a CU with a worker.

    Read through this helper, never inline: the value also sizes the MoE tile
    padding round, and the host and device sides disagreeing there hangs the W2
    workers (see PAD_MULTIPLE in gang_moe_fused_mxfp4_mi300.cuh).
    """
    return int(os.environ.get("MPK_WORKERS_PER_XCD", "31"))


# This method auto probe GPUs and return the worker and scheduler count for
# them.
def get_configurations_from_gpu(rank):
    # Reference: https://github.com/mirage-project/mirage/issues/354
    props = torch.cuda.get_device_properties(rank)
    sm_cnt = props.multi_processor_count
    print("sm_cnt: ", sm_cnt)

    # Check if this is an AMD GPU (ROCm/HIP)
    is_amd = hasattr(torch.version, "hip") and torch.version.hip is not None

    worker = 0
    if is_amd:
        # AMD MI300X configuration (split_worker_scheduler mode)
        # XCD-aligned scheduling: 1 scheduler per XCD (8 total for MI300X).
        # Each scheduler handles all workers on its XCD via stride-based mapping.
        # scheduler_kernel launches num_schedulers blocks with 1 warp (32 threads) each.
        if sm_cnt >= 300:
            num_xcds = 8
            worker = sm_cnt - num_xcds  # 296 workers, use all CUs
            scheduler = num_xcds
            worker = min(worker, MAX_NUM_WORKERS)
            print(f"AMD config: workers={worker}, schedulers={scheduler}, "
                  f"physical_blocks={worker + scheduler}, CUs={sm_cnt}")
            return worker, scheduler
        elif sm_cnt >= 200:
            # MI350: 256 CUs, 8 XCDs (32 CUs/XCD)
            num_xcds = 8
            workers_per_xcd = mpk_workers_per_xcd()
            worker = num_xcds * workers_per_xcd
            scheduler = num_xcds
            worker = min(worker, MAX_NUM_WORKERS)
            print(f"AMD MI350 config: workers={worker}, schedulers={scheduler}, "
                  f"physical_blocks={worker + scheduler}, CUs={sm_cnt}")
            return worker, scheduler
        elif sm_cnt >= 120:
            worker = 96   # 96 + 96 = 192 -> scaled for 120+ CUs
        elif sm_cnt >= 60:
            worker = 48
        else:
            worker = 24
    else:
        # NVIDIA GPU configuration (unchanged)
        if sm_cnt >= 160:
            worker = 144  # Blackwell B200
        elif sm_cnt >= 132:
            worker = 128  # Hopper H100
        elif sm_cnt >= 108:
            worker = 96   # Ampere A100
        elif sm_cnt >= 68:
            worker = 64
        elif sm_cnt >= 40:
            worker = 30
        else:
            worker = 20

    # Cap workers at MAX_NUM_WORKERS
    worker = min(worker, MAX_NUM_WORKERS)
    scheduler = get_scheduler(sm_cnt, worker)

    if is_amd:
        # In split mode, each scheduler block has 1 warp (32 threads).
        # Physical blocks = workers + schedulers = sm_cnt (all CUs used).
        print(f"AMD config: workers={worker}, schedulers={scheduler}, "
              f"physical_blocks={worker + scheduler}, CUs={sm_cnt}")

    return worker, scheduler
