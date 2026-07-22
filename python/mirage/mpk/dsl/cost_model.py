"""Cost model for auto-selecting linear strategy on chiplet GPUs."""

import os


def select_strategy(output_size, reduction_size, batch_size, target_cc,
                    num_xcds=8, l2_per_xcd=4 * 1024 * 1024):
    """Select the best linear strategy for given dimensions and hardware.

    Returns one of: "standard", "gang", "gang_coop", "gang_splitk"
    """
    # Non-MI300X: always standard
    if target_cc != 94:
        return "standard"

    # Gang requires output divisible by num_xcds and chunk by 64
    if output_size % num_xcds != 0:
        return "standard"
    chunk_n = output_size // num_xcds
    if chunk_n % 64 != 0:
        return "standard"

    # Environment variable overrides (backward compat with demo.py)
    if os.environ.get("USE_GANG_COOP", "0") == "1":
        return "gang_coop"
    if os.environ.get("USE_GANG_SPLITK", "0") == "1":
        return "gang_splitk"

    # Default: gang (matches demo.py default behavior)
    return "gang"


def select_k_splits(reduction_size, m_tiles, workers_per_xcd=30,
                    k_per_block=256):
    """Select K_SPLITS so that tiles_per_n >= workers_per_xcd.

    Constraint: K_per_split = reduction_size / K_SPLITS must be
    divisible by k_per_block (256) for CK GEMM alignment.
    """
    if m_tiles >= workers_per_xcd:
        return 1

    for ks in [1, 2, 4, 8, 16]:
        k_per_split = reduction_size // ks
        if (reduction_size % ks == 0
                and k_per_split % k_per_block == 0
                and m_tiles * ks >= workers_per_xcd):
            return ks

    # Fallback: just use 1
    return 1
