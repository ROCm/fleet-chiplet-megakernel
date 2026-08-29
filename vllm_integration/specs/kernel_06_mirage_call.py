# Step 4, batch 6: the gang_full_layer_fused_kernel_mi300 call.
#
# The highest-risk site in the file. 23 template parameters, all int or bool;
# 24 runtime arguments, all int/float/pointer. Swap any two and it compiles
# clean, runs, and emits garbage -- there is no type to catch it.
#
# So the whole 48-line block is replaced by one emitter call. Parameter ORDER
# becomes data (MIRAGE_TEMPLATE_PARAMS / MIRAGE_RUNTIME_ARGS in
# fleet_mk_generate.py, with module-level asserts on the arity mirage declares),
# and the layer_sliding_window initializer above it comes from the config's
# sliding_window_pattern rather than being hardcoded to GPT-OSS's alternation.

SUBS = [
    ("            int const layer_sliding_window = (layer & 1) == 0 ? "
     "SLIDING_WINDOW : 0;",
     "            int const layer_sliding_window = "
     "{emit_layer_sliding_window(cfg)};", 1),

    ("            kernel::gang_full_layer_fused_kernel_mi300<\n"
     "                /*QKV_BATCH_SIZE=*/1,\n"
     "                /*QKV_OUTPUT_PER_WG=*/OUTPUT_PER_WG,\n"
     "                /*QKV_REDUCTION_SIZE=*/HIDDEN_SIZE,\n"
     "                /*ACTUAL_HIDDEN_DIM=*/ACTUAL_HIDDEN_DIM,\n"
     "                /*HEAD_DIM=*/HEAD_DIM,\n"
     "                /*NUM_Q_PER_KV=*/NUM_Q_PER_KV,\n"
     "                /*PAGE_SIZE=*/PAGE_SIZE,\n"
     "                /*MAX_SEQ_LEN=*/MAX_SEQ_LEN,\n"
     "                /*NUM_KV_CHUNKS=*/NUM_KV_CHUNKS,\n"
     "                /*Q_WORKSPACE_STRIDE=*/Q_WORKSPACE_STRIDE,\n"
     "                /*KV_CACHE_STRIDE=*/KV_CACHE_STRIDE,\n"
     "                /*NUM_KV_HEADS=*/NUM_KV_HEADS,\n"
     "                /*SLIDING_WINDOW=*/0,\n"
     "                /*HAS_SINKS=*/1,\n"
     "                /*OPROJ_OUTPUT_PER_WG=*/OPROJ_OPW,\n"
     "                /*OPROJ_REDUCTION_SIZE=*/OPROJ_REDUCTION,\n"
     "                /*NUM_EXPERTS=*/NUM_EXPERTS,\n"
     "                /*TOPK_K=*/NUM_TOPK,\n"
     "                /*MOE_INTERMEDIATE_SIZE=*/MOE_INTERMEDIATE_SIZE,\n"
     "                /*MOE_HIDDEN_SIZE=*/HIDDEN_SIZE,\n"
     "                /*MOE_W13_OUTPUT_PER_WG=*/W13_OPW,\n"
     "                /*MOE_W2_OUTPUT_PER_WG=*/W2_OPW,\n"
     "                /*DECODE_ONLY=*/true>(\n"
     "                mirage_in,\n"
     "                mirage_out,\n"
     "                config.cos_ptr,\n"
     "                config.sin_ptr,\n"
     "                config.qo_indptr,\n"
     "                config.kv_indptr,\n"
     "                config.kv_indices,\n"
     "                config.kv_last_page_len,\n"
     "                config.num_active_tokens,\n"
     "                QKV_N_WGS_PER_XCD,\n"
     "                KV_CACHE_STRIDE,\n"
     "                Q_WORKSPACE_STRIDE,\n"
     "                config.attn_scale,\n"
     "                QKV_N_WGS_PER_XCD,\n"
     "                OPROJ_N_WGS_PER_XCD,\n"
     "                HIDDEN_SIZE,\n"
     "                ROUTER_TILE_N,\n"
     "                OPROJ_N_WGS,\n"
     "                TOTAL_TOPK_TILES,\n"
     "                OPROJ_TILES_PER_XCD,\n"
     "                MOE_TOTAL_TILES_PER_XCD,\n"
     "                WORKERS_PER_XCD,\n"
     "                tile_idx,\n"
     "                layer_sliding_window);",
     "{emit_mirage_call(cfg)}", 1),
]
