# Step 4 follow-up: EMBED_BARRIER_BASE from TRAILING_COUNTERS.
#
# The other half of specs/driver_04_counters.py. This offset is the sum of
# every trailing region declared before EMBED_BARRIER_INTS; the driver's
# counter_total_ints is the sum of all of them. Emitting both from one list
# is the only way the barrier the kernel writes to is guaranteed to be inside
# the buffer the driver allocated -- an overrun here corrupts whatever torch
# put next in the heap, with no crash.

SUBS = [
    ("    NUM_LAYERS * titan::COUNTERS_PER_LAYER + NUM_XCDS * 16 + 16;",
     "{emit_embed_barrier_base()}", 1),
]
