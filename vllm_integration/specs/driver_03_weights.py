# Step 5, batch 3: the per-layer weight slot table.
#
# 14 scattered appends, one count, and one 14-line unpack block -- three views
# of a single ordered list. The failure mode is the same as the pointer table:
# insert an append in the middle and every later `base + N` silently addresses
# the previous slot. The total-length assert still passes.
#
# The appends cannot be replaced as one block (each follows the packing code
# that builds its tensor), so each is substituted by name. emit_weight_append
# looks the index up from WEIGHT_SLOTS, so the printed [N] cannot drift from
# the offset the unpack block uses.

SUBS = [
    ("        weight_tensors.append(qkv_packed)  # [0] qkv_weight",
     "{emit_weight_append(cfg, 'qkv_weight_base')}", 1),
    ("        weight_tensors.append(qkv_bias)  # [1] qkv_bias",
     "{emit_weight_append(cfg, 'qkv_bias')}", 1),
    ("        weight_tensors.append(o_packed)  # [2] oproj_weight (OPW=16)",
     "{emit_weight_append(cfg, 'oproj_weight_base')}", 1),
    ("        weight_tensors.append(o_bias)  # [3] oproj_bias",
     "{emit_weight_append(cfg, 'oproj_bias')}", 1),
    ("        weight_tensors.append(norm_w1.contiguous())  # [4] norm_weight_pre",
     "{emit_weight_append(cfg, 'norm_w1')}", 1),
    ("        weight_tensors.append(norm_w2.contiguous())  # [5] norm_weight_post",
     "{emit_weight_append(cfg, 'norm_w2')}", 1),
    ("        weight_tensors.append(r_packed)  # [6] router_weight (MXFP4)",
     "{emit_weight_append(cfg, 'router_weight_base')}", 1),
    ("        weight_tensors.append(r_bias)  # [7] router_bias",
     "{emit_weight_append(cfg, 'router_bias')}", 1),
    ("        weight_tensors.append(w_router_bf16.contiguous())  "
     "# [8] router_weight_bf16",
     "{emit_weight_append(cfg, 'router_weight_bf16')}", 1),
    ("        weight_tensors.append(gu_packed)  "
     "# [9] w13_weight [E, w13_n_wgs, wg_bytes]",
     "{emit_weight_append(cfg, 'w13_weight_base')}", 1),
    ("        weight_tensors.append(w13_bias)  # [10] w13_bias",
     "{emit_weight_append(cfg, 'w13_bias')}", 1),
    ("        weight_tensors.append(dp_packed)  "
     "# [11] w2_weight [E, w2_n_wgs, wg_bytes]",
     "{emit_weight_append(cfg, 'w2_weight_base')}", 1),
    ("        weight_tensors.append(w2_bias)  # [12] w2_bias",
     "{emit_weight_append(cfg, 'w2_bias')}", 1),
    ("        weight_tensors.append(w_sinks)  # [13] attn_sinks",
     "{emit_weight_append(cfg, 'attn_sinks')}", 1),

    ("    WEIGHTS_PER_LAYER = 14",
     "    WEIGHTS_PER_LAYER = {len(WEIGHT_SLOTS)}", 1),

    ("            qkv_weight_base = weight_ptrs_host[base + 0]\n"
     "            qkv_bias = weight_ptrs_host[base + 1]\n"
     "            oproj_weight_base = weight_ptrs_host[base + 2]\n"
     "            oproj_bias = weight_ptrs_host[base + 3]\n"
     "            norm_w1 = weight_ptrs_host[base + 4]\n"
     "            norm_w2 = weight_ptrs_host[base + 5]\n"
     "            router_weight_base = weight_ptrs_host[base + 6]\n"
     "            router_bias = weight_ptrs_host[base + 7]\n"
     "            router_weight_bf16 = weight_ptrs_host[base + 8]\n"
     "            w13_weight_base = weight_ptrs_host[base + 9]\n"
     "            w13_bias = weight_ptrs_host[base + 10]\n"
     "            w2_weight_base = weight_ptrs_host[base + 11]\n"
     "            w2_bias = weight_ptrs_host[base + 12]\n"
     "            attn_sinks = weight_ptrs_host[base + 13]",
     "{emit_weight_unpack()}", 1),
]
