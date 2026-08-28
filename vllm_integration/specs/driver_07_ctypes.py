# Step 5, batch 7: ctypes argtypes from the shared ABI table.
#
# The plan's warning for this batch, verbatim: a signature/ctypes mismatch is
# stack corruption reproducible only under one of four env-var-selected paths.
# ctypes packs whatever the Python list says; there is no compiler between the
# list and the .so. So the four argtypes blocks and the four C signatures in
# generate_launch_fused_moe (step 6) are emitted from ONE table, ABI_CORE plus
# ABI_TAILS, and can no longer drift.
#
# Each block is replaced whole -- an argtypes list is exactly the contiguous
# region a single substitution should own.

SUBS = [
    ('        ctypes.c_int,       # num_active_tokens' "\n"
     '        ctypes.c_float,     # attn_scale' "\n"
     '        ctypes.c_void_p,    # cos_ptr' "\n"
     '        ctypes.c_void_p,    # sin_ptr' "\n"
     '        ctypes.c_void_p,    # qo_indptr' "\n"
     '        ctypes.c_void_p,    # kv_indptr' "\n"
     '        ctypes.c_void_p,    # kv_indices' "\n"
     '        ctypes.c_void_p,    # kv_last_page_len' "\n"
     '        ctypes.c_void_p,    # ptr_table' "\n"
     '        ctypes.c_void_p,    # counter_buf' "\n"
     '        ctypes.c_void_p,    # lm_norm_weight' "\n"
     '        ctypes.c_void_p,    # lm_norm_scratch' "\n"
     '        ctypes.c_void_p,    # lm_mxfp4_weight' "\n"
     '        ctypes.c_void_p,    # lm_bias' "\n"
     '        ctypes.c_void_p,    # argmax_output' "\n"
     '        ctypes.c_void_p,    # timing_buf' "\n"
     '        ctypes.c_void_p,    # embed_weight' "\n"
     '        ctypes.c_int,       # cur_token_id' "\n"
     '        ctypes.c_void_p,    # decode_ctrl (GPU pointer to DecodeControl struct)' "\n"
     '        ctypes.c_void_p,    # stream',
     "{emit_ctypes_argtypes('launch')}", 1),

    ('        ctypes.c_int,       # num_active_tokens' "\n"
     '        ctypes.c_float,     # attn_scale' "\n"
     '        ctypes.c_void_p,    # cos_ptr' "\n"
     '        ctypes.c_void_p,    # sin_ptr' "\n"
     '        ctypes.c_void_p,    # qo_indptr' "\n"
     '        ctypes.c_void_p,    # kv_indptr' "\n"
     '        ctypes.c_void_p,    # kv_indices' "\n"
     '        ctypes.c_void_p,    # kv_last_page_len' "\n"
     '        ctypes.c_void_p,    # ptr_table' "\n"
     '        ctypes.c_void_p,    # counter_buf' "\n"
     '        ctypes.c_void_p,    # lm_norm_weight' "\n"
     '        ctypes.c_void_p,    # lm_norm_scratch' "\n"
     '        ctypes.c_void_p,    # lm_mxfp4_weight' "\n"
     '        ctypes.c_void_p,    # lm_bias' "\n"
     '        ctypes.c_void_p,    # argmax_output' "\n"
     '        ctypes.c_void_p,    # timing_buf' "\n"
     '        ctypes.c_void_p,    # embed_weight' "\n"
     '        ctypes.c_int,       # cur_token_id' "\n"
     '        ctypes.c_void_p,    # decode_ctrl' "\n"
     '        ctypes.c_void_p,    # stream' "\n"
     '        # Zero buffer info' "\n"
     '        ctypes.c_void_p,    # counter_buf_raw' "\n"
     '        ctypes.c_size_t,    # counter_buf_bytes' "\n"
     '        ctypes.c_void_p,    # workspace_buf_raw' "\n"
     '        ctypes.c_size_t,    # workspace_buf_bytes' "\n"
     '        ctypes.c_void_p,    # moe_barrier_buf_raw' "\n"
     '        ctypes.c_size_t,    # moe_barrier_buf_bytes',
     "{emit_ctypes_argtypes('graph_capture')}", 1),

    ('        ctypes.c_int,       # num_active_tokens' "\n"
     '        ctypes.c_float,     # attn_scale' "\n"
     '        ctypes.c_void_p,    # cos_ptr' "\n"
     '        ctypes.c_void_p,    # sin_ptr' "\n"
     '        ctypes.c_void_p,    # qo_indptr' "\n"
     '        ctypes.c_void_p,    # kv_indptr' "\n"
     '        ctypes.c_void_p,    # kv_indices' "\n"
     '        ctypes.c_void_p,    # kv_last_page_len' "\n"
     '        ctypes.c_void_p,    # ptr_table' "\n"
     '        ctypes.c_void_p,    # counter_buf' "\n"
     '        ctypes.c_void_p,    # lm_norm_weight' "\n"
     '        ctypes.c_void_p,    # lm_norm_scratch' "\n"
     '        ctypes.c_void_p,    # lm_mxfp4_weight' "\n"
     '        ctypes.c_void_p,    # lm_bias' "\n"
     '        ctypes.c_void_p,    # argmax_output' "\n"
     '        ctypes.c_void_p,    # timing_buf' "\n"
     '        ctypes.c_void_p,    # embed_weight' "\n"
     '        ctypes.c_int,       # cur_token_id' "\n"
     '        ctypes.c_void_p,    # decode_ctrl' "\n"
     '        ctypes.c_void_p,    # stream' "\n"
     '        # Bridge kernel args' "\n"
     '        ctypes.c_void_p,    # moe_barrier_raw' "\n"
     '        ctypes.c_int,       # moe_barrier_ints' "\n"
     '        ctypes.c_void_p,    # cur_token_ptr' "\n"
     '        ctypes.c_void_p,    # token_output_buf' "\n"
     '        ctypes.c_void_p,    # step_counter' "\n"
     '        ctypes.c_int,       # start_pos' "\n"
     '        ctypes.c_int,       # page_size' "\n"
     '        # Workspace memset' "\n"
     '        ctypes.c_void_p,    # workspace_buf_raw' "\n"
     '        ctypes.c_size_t,    # workspace_buf_bytes',
     "{emit_ctypes_argtypes('pipe_capture')}", 1),

    ('        ctypes.c_int,       # num_active_tokens' "\n"
     '        ctypes.c_float,     # attn_scale' "\n"
     '        ctypes.c_void_p,    # cos_ptr' "\n"
     '        ctypes.c_void_p,    # sin_ptr' "\n"
     '        ctypes.c_void_p,    # qo_indptr' "\n"
     '        ctypes.c_void_p,    # kv_indptr' "\n"
     '        ctypes.c_void_p,    # kv_indices' "\n"
     '        ctypes.c_void_p,    # kv_last_page_len' "\n"
     '        ctypes.c_void_p,    # ptr_table' "\n"
     '        ctypes.c_void_p,    # counter_buf' "\n"
     '        ctypes.c_void_p,    # lm_norm_weight' "\n"
     '        ctypes.c_void_p,    # lm_norm_scratch' "\n"
     '        ctypes.c_void_p,    # lm_mxfp4_weight' "\n"
     '        ctypes.c_void_p,    # lm_bias' "\n"
     '        ctypes.c_void_p,    # argmax_output' "\n"
     '        ctypes.c_void_p,    # timing_buf' "\n"
     '        ctypes.c_void_p,    # embed_weight' "\n"
     '        ctypes.c_int,       # first_token_id' "\n"
     '        ctypes.c_void_p,    # decode_ctrl' "\n"
     '        ctypes.c_void_p,    # stream' "\n"
     '        # Decode loop params' "\n"
     '        ctypes.c_int,       # total_steps' "\n"
     '        ctypes.c_int,       # start_pos' "\n"
     '        ctypes.c_int,       # page_size' "\n"
     '        ctypes.c_void_p,    # token_output_buf (host-pinned)',
     "{emit_ctypes_argtypes('decode_loop')}", 1),

]
