# Step 4, batch 4: the per-layer counter layout.
#
# Highest silent-corruption risk in the kernel. These offsets are duplicated in
# demo_gpt_oss_120b.py (which sizes the counter buffer) and consumed by mirage.
# A mismatch does not fail to build and does not crash -- it corrupts a live
# barrier and produces garbage tokens.
#
# So neither block below is substituted line-by-line. Both are replaced by a
# call into COUNTER_REGIONS, the single running-sum table in titan_generate.py,
# which is the only place a cache-line count is written down. The comment map
# and the SLOT_* declarations are in DIFFERENT orders in the target file, and
# the emitters preserve that -- the map is layout order, the declarations lead
# with the end-of-layer barrier slots.

SUBS = [
    # ---- Comment map (layout order) ----
    ("//   [0..9*16-1]    : OProj Mechanism C (10 cache lines)\n"
     "//   [10*16..18*16] : routing_ready (9 cache lines)\n"
     "//   [19*16]        : attn_global_counter\n"
     "//   [20*16..27*16] : qkv_epoch (8 per-XCD)\n"
     "//   [28*16..35*16] : chunk_barrier (8 per-XCD)\n"
     "//   [36*16..43*16] : attn_release (8 per-XCD)\n"
     "//   [44*16..51*16] : qkv_barrier arrival (input_ptrs[7], 8 ints padded "
     "to 8 cache lines)\n"
     "//   [52*16]        : layer_done (end-of-layer barrier)\n"
     "//   [53*16..60*16] : layer_local (per-XCD local arrive, 8 cache lines)\n"
     "//   [61*16]        : tail_lmhead\n"
     "//   [62*16..64*16] : tail_argmax",
     "{emit_counter_map(cfg)}", 1),

    # ---- SLOT_* declarations (emission order, not layout order) ----
    ("static constexpr int SLOT_LAYER_DONE_NEW    = 52 * 16;  "
     "// global arrival counter (1 cache line)\n"
     "static constexpr int SLOT_LAYER_LOCAL_NEW   = 53 * 16;  "
     "// per-XCD local arrival (8 cache lines: 53..60)\n"
     "static constexpr int SLOT_TAIL_LMHEAD_NEW   = 61 * 16;\n"
     "static constexpr int SLOT_TAIL_ARGMAX_NEW   = 62 * 16;\n"
     "static constexpr int SLOT_QKV_BARRIER_NEW   = 44 * 16;\n"
     "static constexpr int SLOT_LAYER_RELEASE_NEW = 65 * 16;  "
     "// per-XCD release flags (8 cache lines: 65..72)\n"
     "static constexpr int SLOT_LAYER_DONE_GLOBAL = 73 * 16;  "
     "// global done counter for release (1 cache line)",
     "{emit_counter_slots(cfg)}", 1),
]
