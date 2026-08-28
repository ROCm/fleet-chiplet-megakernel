# Step 5, batch 4: the trailing counter regions.
#
# The driver's counter_total_ints and the kernel's EMBED_BARRIER_BASE are the
# same running sum in two languages. This batch points the driver at
# TRAILING_COUNTERS; kernel_08 points EMBED_BARRIER_BASE at the same list, so
# the offset the kernel writes to and the size the driver allocates move
# together.

SUBS = [
    ("    RANK_COUNTER_INTS = NUM_XCDS * 16\n"
     "    DECODE_ITER_COUNTER_INTS = 16  "
     "# 1 cache line for decode iteration barrier\n"
     "    # Embedding-write barrier: 1 global cache line + NUM_XCDS per-XCD lines.\n"
     "    # Kept outside the per-layer blocks, whose 103 cache lines are fully used.\n"
     "    # Must match SLOT_EMBED_DONE / SLOT_EMBED_LOCAL in the kernel.\n"
     "    EMBED_BARRIER_INTS = (1 + NUM_XCDS) * 16\n"
     "    # Slack for the TITAN_ILB_TIMING diagnostic probe barrier (compiled out in\n"
     "    # production). Always allocated so a timing build needs no Python change.\n"
     "    ILB_PROBE_INTS = 20 * 16\n"
     "    counter_total_ints = (NUM_LAYERS * COUNTERS_PER_LAYER + RANK_COUNTER_INTS\n"
     "                          + DECODE_ITER_COUNTER_INTS + EMBED_BARRIER_INTS\n"
     "                          + ILB_PROBE_INTS)",
     "{emit_trailing_counters(cfg)}", 1),

    # buf_moe_barrier is deliberately NOT substituted. It reads
    # `16 * NUM_EXPERTS`, and NUM_EXPERTS is already emitted from cfg in
    # batch 1 -- the expression tracks the config on its own. Replacing it
    # with a computed literal would make the file less readable and gain
    # nothing.
]
