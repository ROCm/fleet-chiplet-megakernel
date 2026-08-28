# Step 5, batch 5: attention runtime parameters.
#
# page_size and max_seq_len have twins in the kernel (PAGE_SIZE, MAX_SEQ_LEN),
# already emitted from cfg there. The two strides are derived expressions whose
# trailing `// 512` comments are the part that goes stale.
#
# max_num_pages is left at the literal 16 on purpose: it is not a config field,
# it is `ceil(max_seq_len / page_size)` rounded up to a comfortable slack, and
# 512/4096 rounds to 1. Deriving it would shrink the allocation. Left alone
# until there is a measured reason to size it.

SUBS = [
    ("        --prompt \"Tell me the history of america\" --max-seq-length 512",
     "        --prompt \"Tell me the history of america\" "
     "--max-seq-length {cfg.max_seq_len}", 1),
    ("    parser.add_argument(\"--max-seq-length\", type=int, default=512)",
     "    parser.add_argument(\"--max-seq-length\", type=int, "
     "default={cfg.max_seq_len})", 1),
    ("    page_size = 4096",
     "    page_size = {cfg.page_size}", 1),
    ("    q_ws_stride = Q_PER_KV * HEAD_DIM  # 512",
     "    q_ws_stride = Q_PER_KV * HEAD_DIM  # {cfg.q_workspace_stride}", 1),
    ("    kv_cache_stride = NUM_KV_HEADS * HEAD_DIM  # 512",
     "    kv_cache_stride = NUM_KV_HEADS * HEAD_DIM  # {cfg.kv_cache_stride}",
     1),
]
