# Step 6, batch 1: the C entrypoint signatures.
#
# The other half of specs/driver_07_ctypes.py. These four signatures and those
# four ctypes lists are one ABI written twice in two languages with no compiler
# in between. Both now come out of ABI_CORE + ABI_TAILS, so a parameter added
# to one is added to the other in the same edit -- which is the only way to
# make the failure mode impossible rather than merely unlikely.
#
# Each signature is replaced whole: it is a contiguous region, and a partial
# replacement is exactly the drift being prevented.

SUBS = [
    ('    int num_active_tokens,' "\n"
     '    float attn_scale,' "\n"
     '    void *cos_ptr,' "\n"
     '    void *sin_ptr,' "\n"
     '    int *qo_indptr,' "\n"
     '    int *kv_indptr,' "\n"
     '    int *kv_indices,' "\n"
     '    int *kv_last_page_len,' "\n"
     '    void **ptr_table,' "\n"
     '    void **counter_buf_vp,' "\n"
     '    void *lm_norm_weight,' "\n"
     '    void *lm_norm_scratch,' "\n"
     '    void *lm_mxfp4_weight,' "\n"
     '    void *lm_bias,' "\n"
     '    void *argmax_output,' "\n"
     '    unsigned long long *timing_buf,' "\n"
     '    void *embed_weight,' "\n"
     '    int cur_token_id,' "\n"
     '    DecodeControl *decode_ctrl,' "\n"
     '    hipStream_t stream)',
     "{emit_c_signature('launch')}", 1),

    ('    int num_active_tokens,' "\n"
     '    float attn_scale,' "\n"
     '    void *cos_ptr,' "\n"
     '    void *sin_ptr,' "\n"
     '    int *qo_indptr,' "\n"
     '    int *kv_indptr,' "\n"
     '    int *kv_indices,' "\n"
     '    int *kv_last_page_len,' "\n"
     '    void **ptr_table,' "\n"
     '    void **counter_buf_vp,' "\n"
     '    void *lm_norm_weight,' "\n"
     '    void *lm_norm_scratch,' "\n"
     '    void *lm_mxfp4_weight,' "\n"
     '    void *lm_bias,' "\n"
     '    void *argmax_output,' "\n"
     '    unsigned long long *timing_buf,' "\n"
     '    void *embed_weight,' "\n"
     '    int cur_token_id,' "\n"
     '    DecodeControl *decode_ctrl,' "\n"
     '    hipStream_t stream,' "\n"
     '    // Zero buffer info' "\n"
     '    void *counter_buf_raw,' "\n"
     '    size_t counter_buf_bytes,' "\n"
     '    void *workspace_buf_raw,' "\n"
     '    size_t workspace_buf_bytes,' "\n"
     '    void *moe_barrier_buf_raw,' "\n"
     '    size_t moe_barrier_buf_bytes)',
     "{emit_c_signature('graph_capture')}", 1),

    ('    int num_active_tokens,' "\n"
     '    float attn_scale,' "\n"
     '    void *cos_ptr,' "\n"
     '    void *sin_ptr,' "\n"
     '    int *qo_indptr,' "\n"
     '    int *kv_indptr,' "\n"
     '    int *kv_indices,' "\n"
     '    int *kv_last_page_len,' "\n"
     '    void **ptr_table,' "\n"
     '    void **counter_buf_vp,' "\n"
     '    void *lm_norm_weight,' "\n"
     '    void *lm_norm_scratch,' "\n"
     '    void *lm_mxfp4_weight,' "\n"
     '    void *lm_bias,' "\n"
     '    void *argmax_output,' "\n"
     '    unsigned long long *timing_buf,' "\n"
     '    void *embed_weight,' "\n"
     '    int cur_token_id,' "\n"
     '    DecodeControl *decode_ctrl,' "\n"
     '    hipStream_t stream,' "\n"
     '    // Bridge kernel args' "\n"
     '    void *moe_barrier_raw,' "\n"
     '    int moe_barrier_ints,' "\n"
     '    int *cur_token_ptr,' "\n"
     '    int *token_output_buf,' "\n"
     '    int *step_counter,' "\n"
     '    int start_pos,' "\n"
     '    int page_size,' "\n"
     '    // Workspace memset' "\n"
     '    void *workspace_buf_raw,' "\n"
     '    size_t workspace_buf_bytes)',
     "{emit_c_signature('pipe_capture')}", 1),

    ('    int num_active_tokens,' "\n"
     '    float attn_scale,' "\n"
     '    void *cos_ptr,' "\n"
     '    void *sin_ptr,' "\n"
     '    int *qo_indptr,' "\n"
     '    int *kv_indptr,' "\n"
     '    int *kv_indices,' "\n"
     '    int *kv_last_page_len,' "\n"
     '    void **ptr_table,' "\n"
     '    void **counter_buf_vp,' "\n"
     '    void *lm_norm_weight,' "\n"
     '    void *lm_norm_scratch,' "\n"
     '    void *lm_mxfp4_weight,' "\n"
     '    void *lm_bias,' "\n"
     '    void *argmax_output,' "\n"
     '    unsigned long long *timing_buf,' "\n"
     '    void *embed_weight,' "\n"
     '    int first_token_id,' "\n"
     '    DecodeControl *decode_ctrl,' "\n"
     '    hipStream_t stream,' "\n"
     '    // Decode loop params' "\n"
     '    int total_steps,' "\n"
     '    int start_pos,' "\n"
     '    int page_size,' "\n"
     '    int *token_output_buf)  // host-pinned or device buffer for output tokens',
     "{emit_c_signature('decode_loop')}", 1),

]
