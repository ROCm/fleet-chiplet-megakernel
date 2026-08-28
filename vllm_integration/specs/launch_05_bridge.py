# Step 6, batch 5: the BridgeArgs field comments.
#
# BridgeArgs carries page_size and moe_barrier_ints as runtime values -- Python
# passes them in, so nothing here breaks if they change. What breaks is the
# reader: four comments state the GPT-OSS values as fact, and a stale comment
# on a struct whose whole job is cross-kernel state hand-off is worse than no
# comment.
#
# 2048 is the driver's `16 * NUM_EXPERTS`. That expression stays the authority
# (see specs/driver_04_counters.py for why it is not collapsed to a literal
# there); these comments are derived from the same two numbers rather than
# from a second definition of the size.
#
# The "/ 256 threads = 8 per thread" arithmetic is emitted as division, so the
# quotient cannot drift from the dividend. 256 itself stays literal: it is
# NUM_THREADS, a property of the device-function library's wave layout, and
# the dense arm leaves it literal too -- see specs/launch_03_grid.py.

SUBS = [
    ("    int       *moe_barrier;     // write: zero 2048 ints",
     "    int       *moe_barrier;     // write: zero "
     "{16 * cfg.num_experts} ints", 1),
    ("    int        page_size;       // const: 4096",
     "    int        page_size;       // const: {cfg.page_size}", 1),
    ("    int        moe_barrier_ints;// const: 2048",
     "    int        moe_barrier_ints;// const: {16 * cfg.num_experts}", 1),
    ("    // All threads: zero MoE barrier "
     "(2048 ints / 256 threads = 8 per thread)",
     "    // All threads: zero MoE barrier ({16 * cfg.num_experts} ints / 256 "
     "threads = {16 * cfg.num_experts // 256} per thread)", 1),
]
