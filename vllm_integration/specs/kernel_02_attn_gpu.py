# Step 4, batch 2: attention shape, GPU layout, and the QKV/OProj tile
# constants. The trailing `// N` comments are emitted too -- leaving them as
# literals is how a generated file grows lies.

SUBS = [
    # ---- Attention shape ----
    ("static constexpr int NUM_Q_HEADS = 64;",
     "static constexpr int NUM_Q_HEADS = {cfg.num_q_heads};", 1),
    ("static constexpr int NUM_KV_HEADS = 8;",
     "static constexpr int NUM_KV_HEADS = {cfg.num_kv_heads};", 1),
    ("static constexpr int HEAD_DIM = 64;",
     "static constexpr int HEAD_DIM = {cfg.head_dim};", 1),
    ("static constexpr int NUM_Q_PER_KV = 8;  // 64 / 8",
     "static constexpr int NUM_Q_PER_KV = {cfg.q_per_kv};  "
     "// {cfg.num_q_heads} / {cfg.num_kv_heads}", 1),

    ("// QKV GEMM output: Q(4096) + K(512) + V(512) = 5120",
     "// QKV GEMM output: Q({cfg.oproj_reduction}) + K({cfg.kv_cache_stride})"
     " + V({cfg.kv_cache_stride}) = {cfg.qkv_output_size}", 1),
    ("                                     + 2 * NUM_KV_HEADS * HEAD_DIM;  // 5120",
     "                                     + 2 * NUM_KV_HEADS * HEAD_DIM;  "
     "// {cfg.qkv_output_size}", 1),

    # ---- GPU layout ----
    ("static constexpr int NUM_XCDS = 8;",
     "static constexpr int NUM_XCDS = {cfg.num_xcds};", 1),
    ("static constexpr int WORKERS_PER_XCD = 30;",
     "static constexpr int WORKERS_PER_XCD = {cfg.workers_per_xcd};", 1),

    ("// All GEMMs use OPW=64 for QKV, OPW=16 for OProj",
     "// All GEMMs use OPW={cfg.output_per_wg} for QKV, "
     "OPW={cfg.oproj_opw} for OProj", 1),
    ("static constexpr int OUTPUT_PER_WG = 64;",
     "static constexpr int OUTPUT_PER_WG = {cfg.output_per_wg};", 1),

    # ---- QKV GEMM tiles ----
    ("static constexpr int QKV_N_WGS = QKV_OUTPUT_SIZE / OUTPUT_PER_WG;     // 80",
     "static constexpr int QKV_N_WGS = QKV_OUTPUT_SIZE / OUTPUT_PER_WG;     "
     "// {cfg.qkv_n_wgs}", 1),
    ("static constexpr int QKV_N_WGS_PER_XCD = QKV_N_WGS / NUM_XCDS;       // 10",
     "static constexpr int QKV_N_WGS_PER_XCD = QKV_N_WGS / NUM_XCDS;       "
     "// {cfg.qkv_n_wgs_per_xcd}", 1),

    # ---- OProj tiles. OPROJ_OPW is MEASURED: auto-derivation picks 64 and is
    # wrong (see configs/gpt_oss_120b.yaml `measured:`). ----
    ("static constexpr int OPROJ_REDUCTION = NUM_Q_HEADS * HEAD_DIM;         // 4096",
     "static constexpr int OPROJ_REDUCTION = NUM_Q_HEADS * HEAD_DIM;         "
     "// {cfg.oproj_reduction}", 1),
    ("static constexpr int OPROJ_OPW = 16;",
     "static constexpr int OPROJ_OPW = {cfg.oproj_opw};", 1),
    ("static constexpr int OPROJ_N_WGS = HIDDEN_SIZE / OPROJ_OPW;            // 184",
     "static constexpr int OPROJ_N_WGS = HIDDEN_SIZE / OPROJ_OPW;            "
     "// {cfg.oproj_n_wgs}", 1),
    ("static constexpr int OPROJ_N_WGS_PER_XCD = OPROJ_N_WGS / NUM_XCDS;    // 23",
     "static constexpr int OPROJ_N_WGS_PER_XCD = OPROJ_N_WGS / NUM_XCDS;    "
     "// {cfg.oproj_n_wgs_per_xcd}", 1),
    ("static constexpr int OPROJ_TILES_PER_XCD = OPROJ_N_WGS_PER_XCD;       // 23",
     "static constexpr int OPROJ_TILES_PER_XCD = OPROJ_N_WGS_PER_XCD;       "
     "// {cfg.oproj_n_wgs_per_xcd}", 1),
]
