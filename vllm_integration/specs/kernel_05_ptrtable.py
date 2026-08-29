# Step 4, batch 5: pointer-table counts and the EMBED_BARRIER_BASE comment.
#
# The three counts here are the kernel's view of MIRAGE_IN / MIRAGE_OUT in
# fleet_mk_generate.py; the demo's table build (step 5) is the other view. Driving
# both from len() is the point -- an off-by-one between them does not fail to
# build, it hands mirage the wrong buffer and the tokens come out garbage.
#
# EMBED_BARRIER_BASE itself is already symbolic in the target
# (NUM_LAYERS * fleet_mk::COUNTERS_PER_LAYER + NUM_XCDS * 16 + 16), so only its
# explanatory comment needs the 103 pulled from the header.

SUBS = [
    ("// 24 mirage_in + 11 mirage_out + 1 layer_output = 36",
     "// {len(MIRAGE_IN)} mirage_in + {len(MIRAGE_OUT)} mirage_out "
     "+ 1 layer_output = {MIRAGE_PTRS_PER_LAYER}", 1),
    ("static constexpr int MIRAGE_IN_COUNT = 24;",
     "static constexpr int MIRAGE_IN_COUNT = {len(MIRAGE_IN)};", 1),
    ("static constexpr int MIRAGE_OUT_COUNT = 11;",
     "static constexpr int MIRAGE_OUT_COUNT = {len(MIRAGE_OUT)};", 1),
    ("static constexpr int PTRS_PER_LAYER = MIRAGE_IN_COUNT + MIRAGE_OUT_COUNT + 1;  // 36",
     "static constexpr int PTRS_PER_LAYER = MIRAGE_IN_COUNT + MIRAGE_OUT_COUNT"
     " + 1;  // {MIRAGE_PTRS_PER_LAYER}", 1),

    ("// mirage_in[0..23] are at slots 0..23\n"
     "// mirage_out[0..10] are at slots 24..34\n"
     "// layer_output is at slot 35\n"
     "static constexpr int SLOT_LAYER_OUTPUT = MIRAGE_IN_COUNT + MIRAGE_OUT_COUNT; // 35",
     "// mirage_in[0..{len(MIRAGE_IN) - 1}] are at slots 0..{len(MIRAGE_IN) - 1}\n"
     "// mirage_out[0..{len(MIRAGE_OUT) - 1}] are at slots {len(MIRAGE_IN)}.."
     "{len(MIRAGE_IN) + len(MIRAGE_OUT) - 1}\n"
     "// layer_output is at slot {len(MIRAGE_IN) + len(MIRAGE_OUT)}\n"
     "static constexpr int SLOT_LAYER_OUTPUT = MIRAGE_IN_COUNT + "
     "MIRAGE_OUT_COUNT; // {len(MIRAGE_IN) + len(MIRAGE_OUT)}", 1),

    # counters_per_layer is parsed out of kernels/device_functions.cuh, so this
    # comment cannot drift from the header it is quoting.
    ("// decode-iter counters. The per-layer block's 103 cache lines are fully",
     "// decode-iter counters. The per-layer block's "
     "{cfg.counters_per_layer // 16} cache lines are fully", 1),
]
