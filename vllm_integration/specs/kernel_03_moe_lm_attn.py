# Step 4, batch 3: MoE tile constants, router, LM head, and attention runtime
# parameters.
#
# W13_OPW and W2_OPW are MEASURED, not derived, and they pull in OPPOSITE
# directions -- W13 wants fat tiles (LDS weight reuse across 2 MFMA rounds),
# W2 wants thin ones (parallelism hides HBM latency). The emitted comments
# carry the cost of the alternative so the next reader does not re-run the
# experiment. See configs/gpt_oss_120b.yaml `measured:`.

SUBS = [
    # ---- MoE expert GEMMs ----
    ("static constexpr int NUM_EXPERTS = 128;",
     "static constexpr int NUM_EXPERTS = {cfg.num_experts};", 1),
    ("static constexpr int NUM_TOPK = 4;",
     "static constexpr int NUM_TOPK = {cfg.num_experts_per_tok};", 1),
    ("static constexpr int MOE_INTERMEDIATE_SIZE = 2944;",
     "static constexpr int MOE_INTERMEDIATE_SIZE = "
     "{cfg.padded_moe_intermediate_size};", 1),

    ("static constexpr int W13_OPW = 128;",
     "static constexpr int W13_OPW = {cfg.w13_output_per_wg};", 1),
    ("static constexpr int W13_OUTPUT_SIZE = 2 * MOE_INTERMEDIATE_SIZE;      // 5888",
     "static constexpr int W13_OUTPUT_SIZE = 2 * MOE_INTERMEDIATE_SIZE;      "
     "// {cfg.w13_output_size}", 1),
    ("static constexpr int W13_N_WGS = W13_OUTPUT_SIZE / W13_OPW;           // 46",
     "static constexpr int W13_N_WGS = W13_OUTPUT_SIZE / W13_OPW;           "
     "// {cfg.w13_n_wgs}", 1),
    ("static constexpr int W2_OPW = 64;",
     "static constexpr int W2_OPW = {cfg.w2_output_per_wg};", 1),
    ("static constexpr int W2_N_WGS = HIDDEN_SIZE / W2_OPW;                 // 46",
     "static constexpr int W2_N_WGS = HIDDEN_SIZE / W2_OPW;                 "
     "// {cfg.w2_n_wgs}", 1),

    # ---- Router ----
    ("static constexpr int ROUTER_TILE_N = NUM_EXPERTS / NUM_XCDS;  // 16",
     "static constexpr int ROUTER_TILE_N = NUM_EXPERTS / NUM_XCDS;  "
     "// {cfg.num_experts // cfg.num_xcds}", 1),
    ("static constexpr int TOTAL_TOPK_TILES = NUM_EXPERTS;           // 128",
     "static constexpr int TOTAL_TOPK_TILES = NUM_EXPERTS;           "
     "// {cfg.num_experts}", 1),

    # ---- Fused MoE tile count. Cross-checked against the derivation in
    # load_and_validate; changing either OPW changes this number. ----
    ("// Fused MoE tile count per XCD (matches mirage: OPW=128 for W13)\n"
     "static constexpr int MOE_TOTAL_TILES_PER_XCD = 53;",
     "// Fused MoE tile count per XCD (matches mirage: "
     "OPW={cfg.w13_output_per_wg} for W13)\n"
     "static constexpr int MOE_TOTAL_TILES_PER_XCD = "
     "{cfg.moe_total_tiles_per_xcd};", 1),

    # ---- LM head ----
    ("static constexpr int LM_N_WGS = PADDED_VOCAB_SIZE / OUTPUT_PER_WG;    // 3144",
     "static constexpr int LM_N_WGS = PADDED_VOCAB_SIZE / OUTPUT_PER_WG;    "
     "// {cfg.lm_n_wgs}", 1),
    ("static constexpr int LM_N_WGS_PER_XCD = LM_N_WGS / NUM_XCDS;         // 393",
     "static constexpr int LM_N_WGS_PER_XCD = LM_N_WGS / NUM_XCDS;         "
     "// {cfg.lm_n_wgs_per_xcd}", 1),

    # ---- Attention runtime parameters ----
    ("static constexpr int PAGE_SIZE = 4096;",
     "static constexpr int PAGE_SIZE = {cfg.page_size};", 1),
    ("// 128-token window, odd layers are \"full_attention\" (window 0 = unlimited).\n"
     "static constexpr int SLIDING_WINDOW = 128;",
     "// {cfg.sliding_window}-token window, odd layers are \"full_attention\" "
     "(window 0 = unlimited).\n"
     "static constexpr int SLIDING_WINDOW = {cfg.sliding_window};", 1),
    ("static constexpr int NUM_KV_CHUNKS = 8;",
     "static constexpr int NUM_KV_CHUNKS = {cfg.num_kv_chunks};", 1),
    ("static constexpr int MAX_SEQ_LEN = 512;",
     "static constexpr int MAX_SEQ_LEN = {cfg.max_seq_len};", 1),
    ("static constexpr int KV_CACHE_STRIDE = NUM_KV_HEADS * HEAD_DIM;       // 512",
     "static constexpr int KV_CACHE_STRIDE = NUM_KV_HEADS * HEAD_DIM;       "
     "// {cfg.kv_cache_stride}", 1),
    ("static constexpr int Q_WORKSPACE_STRIDE = NUM_Q_PER_KV * HEAD_DIM;    // 512",
     "static constexpr int Q_WORKSPACE_STRIDE = NUM_Q_PER_KV * HEAD_DIM;    "
     "// {cfg.q_workspace_stride}", 1),
]
