# Step 5, batch 1: the driver's constants header.
#
# Every constant here has a twin in generated/gpt_oss_120b_kernel.cuh. The
# three *_OPW values are the dangerous ones: they control WEIGHT PACKING here
# and GEMM TILING there. Change one file only and the build succeeds, the run
# does not crash, and the tokens are garbage. Driving both from one cfg is the
# entire point of the exercise.
#
# The two MEASURED comment blocks (W13_OPW, W2_OPW) record expensive negative
# results and are preserved verbatim -- only the values they annotate are
# substituted.

SUBS = [
    # ---- Model dimensions ----
    ("HIDDEN_SIZE = 2944",
     "HIDDEN_SIZE = {cfg.padded_hidden_size}", 1),
    ("ACTUAL_HIDDEN_DIM = 2880  # for RMSNorm mean",
     "ACTUAL_HIDDEN_DIM = {cfg.hidden_size}  # for RMSNorm mean", 1),
    # Leading newline anchors this to a line start: without it the `old` is
    # also a suffix of MOE_INTERMEDIATE_SIZE, and these are two different
    # config fields that merely coincide at 2944 for GPT-OSS.
    ("\nINTERMEDIATE_SIZE = 2944",
     "\nINTERMEDIATE_SIZE = {cfg.padded_intermediate_size}", 1),
    ("VOCAB_SIZE = 201088",
     "VOCAB_SIZE = {cfg.vocab_size}", 1),
    ("PADDED_VOCAB_SIZE = 201216  # next multiple for LM head split",
     "PADDED_VOCAB_SIZE = {cfg.padded_vocab_size}  "
     "# next multiple for LM head split", 1),
    ("NUM_LAYERS = 36",
     "NUM_LAYERS = {cfg.num_layers}", 1),

    # ---- Attention shape ----
    ("NUM_Q_HEADS = 64",
     "NUM_Q_HEADS = {cfg.num_q_heads}", 1),
    ("NUM_KV_HEADS = 8",
     "NUM_KV_HEADS = {cfg.num_kv_heads}", 1),
    ("HEAD_DIM = 64",
     "HEAD_DIM = {cfg.head_dim}", 1),
    ("Q_PER_KV = NUM_Q_HEADS // NUM_KV_HEADS  # 8",
     "Q_PER_KV = NUM_Q_HEADS // NUM_KV_HEADS  # {cfg.q_per_kv}", 1),
    ("QKV_OUTPUT_SIZE = NUM_Q_HEADS * HEAD_DIM + 2 * NUM_KV_HEADS * HEAD_DIM  # 5120",
     "QKV_OUTPUT_SIZE = NUM_Q_HEADS * HEAD_DIM + 2 * NUM_KV_HEADS * HEAD_DIM  "
     "# {cfg.qkv_output_size}", 1),

    # ---- GPU layout ----
    ("# All GEMMs use OPW=64\nOUTPUT_PER_WG = 64",
     "# All GEMMs use OPW={cfg.output_per_wg}\n"
     "OUTPUT_PER_WG = {cfg.output_per_wg}", 1),
    ("NUM_XCDS = 8",
     "NUM_XCDS = {cfg.num_xcds}", 1),
    ("WORKERS_PER_XCD = 30",
     "WORKERS_PER_XCD = {cfg.workers_per_xcd}", 1),
    ("NUM_KV_CHUNKS = 8",
     "NUM_KV_CHUNKS = {cfg.num_kv_chunks}", 1),

    # ---- Per-XCD workgroup counts ----
    ("QKV_N_WGS_PER_XCD = (QKV_OUTPUT_SIZE // OUTPUT_PER_WG) // NUM_XCDS  # 10",
     "QKV_N_WGS_PER_XCD = (QKV_OUTPUT_SIZE // OUTPUT_PER_WG) // NUM_XCDS  "
     "# {cfg.qkv_n_wgs_per_xcd}", 1),
    ("OPROJ_REDUCTION = NUM_Q_HEADS * HEAD_DIM  # 4096",
     "OPROJ_REDUCTION = NUM_Q_HEADS * HEAD_DIM  # {cfg.oproj_reduction}", 1),
    ("OPROJ_OPW = 16",
     "OPROJ_OPW = {cfg.oproj_opw}", 1),
    ("OPROJ_N_WGS = HIDDEN_SIZE // OPROJ_OPW  # 184 (global distribution)",
     "OPROJ_N_WGS = HIDDEN_SIZE // OPROJ_OPW  "
     "# {cfg.oproj_n_wgs} (global distribution)", 1),
    ("OPROJ_N_WGS_PER_XCD = OPROJ_N_WGS // NUM_XCDS  # 23",
     "OPROJ_N_WGS_PER_XCD = OPROJ_N_WGS // NUM_XCDS  "
     "# {cfg.oproj_n_wgs_per_xcd}", 1),

    # ---- MoE ----
    ("NUM_EXPERTS = 128",
     "NUM_EXPERTS = {cfg.num_experts}", 1),
    ("NUM_TOPK = 4",
     "NUM_TOPK = {cfg.num_experts_per_tok}", 1),
    ("MOE_INTERMEDIATE_SIZE = 2944",
     "MOE_INTERMEDIATE_SIZE = {cfg.padded_moe_intermediate_size}", 1),
    ("W13_OPW = 128",
     "W13_OPW = {cfg.w13_output_per_wg}", 1),
    ("W13_N_WGS = W13_OUTPUT_SIZE // W13_OPW  # 46",
     "W13_N_WGS = W13_OUTPUT_SIZE // W13_OPW  # {cfg.w13_n_wgs}", 1),
    ("W2_OPW = 64",
     "W2_OPW = {cfg.w2_output_per_wg}", 1),
    ("W2_N_WGS = HIDDEN_SIZE // W2_OPW  # 46",
     "W2_N_WGS = HIDDEN_SIZE // W2_OPW  # {cfg.w2_n_wgs}", 1),
    ("ROUTER_OUTPUT_SIZE = 128",
     "ROUTER_OUTPUT_SIZE = {cfg.num_experts}", 1),
    ("ROUTER_N_WGS = ROUTER_OUTPUT_SIZE // OUTPUT_PER_WG  # 2",
     "ROUTER_N_WGS = ROUTER_OUTPUT_SIZE // OUTPUT_PER_WG  "
     "# {cfg.router_n_wgs}", 1),
    ("LM_N_WGS_PER_XCD = (PADDED_VOCAB_SIZE // OUTPUT_PER_WG) // NUM_XCDS  # 393",
     "LM_N_WGS_PER_XCD = (PADDED_VOCAB_SIZE // OUTPUT_PER_WG) // NUM_XCDS  "
     "# {cfg.lm_n_wgs_per_xcd}", 1),
    ("OPROJ_WG_BYTES = OPROJ_OPW * (OPROJ_REDUCTION // 2 + OPROJ_REDUCTION // 32)  # 34816 for OPW=16",
     "OPROJ_WG_BYTES = OPROJ_OPW * (OPROJ_REDUCTION // 2 + "
     "OPROJ_REDUCTION // 32)  # {cfg.oproj_wg_bytes} for OPW={cfg.oproj_opw}",
     1),

    # ---- Pointer table: the demo's view of MIRAGE_IN / MIRAGE_OUT. The
    # kernel's view (MIRAGE_IN_COUNT etc.) comes from the same two lists. ----
    ("# 24 mirage_in + 11 mirage_out + 1 layer_output = 36 per layer per XCD\n"
     "MIRAGE_IN = 24\n"
     "MIRAGE_OUT = 11\n"
     "PTRS_PER_LAYER = MIRAGE_IN + MIRAGE_OUT + 1  # 36",
     "# {len(MIRAGE_IN)} mirage_in + {len(MIRAGE_OUT)} mirage_out "
     "+ 1 layer_output = {MIRAGE_PTRS_PER_LAYER} per layer per XCD\n"
     "MIRAGE_IN = {len(MIRAGE_IN)}\n"
     "MIRAGE_OUT = {len(MIRAGE_OUT)}\n"
     "PTRS_PER_LAYER = MIRAGE_IN + MIRAGE_OUT + 1  "
     "# {MIRAGE_PTRS_PER_LAYER}", 1),

    # ---- Counters: same COUNTER_REGIONS table and same parsed header the
    # kernel emitter uses, so the two files cannot disagree. ----
    ("SLOT_QKV_BARRIER = 44 * 16  # per-XCD QKV barrier arrival",
     "SLOT_QKV_BARRIER = {counter_slots()['SLOT_QKV_BARRIER_NEW'][0]} * 16  "
     "# per-XCD QKV barrier arrival", 1),
    ("COUNTERS_PER_LAYER = 103 * 16  # 84 base + 10 fused oproj + 9 routing ready",
     "COUNTERS_PER_LAYER = {cfg.counters_per_layer // 16} * 16  "
     "# 84 base + 10 fused oproj + 9 routing ready", 1),
]
