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
            worker = 240  # Same as MI300X for compatibility
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
