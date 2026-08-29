# Step 5, batch 2: the mirage pointer table build.
#
# The highest-value substitution in the plan. This 35-entry table and the
# kernel's MIRAGE_IN_COUNT / MIRAGE_OUT_COUNT / SLOT_LAYER_OUTPUT were
# maintained by hand in two files. mirage indexes input_ptrs[]/output_ptrs[]
# positionally, so an entry inserted on one side and not the other does not
# fail an assert -- every length still checks out -- it just hands mirage a
# different buffer than the kernel thinks it passed. Output is garbage and
# nothing says why.
#
# After this both files read from MIRAGE_IN / MIRAGE_OUT in fleet_mk_generate.py,
# which is also where the two asserts below get their numbers, so the assert
# can no longer pass while the table is wrong.

SUBS = [
    ("            # Pre-computed mirage format: 24 mirage_in + 11 mirage_out "
     "+ 1 counter = 36 ptrs",
     "            # Pre-computed mirage format: {len(MIRAGE_IN)} mirage_in + "
     "{len(MIRAGE_OUT)} mirage_out + 1 counter = {MIRAGE_PTRS_PER_LAYER} ptrs",
     1),

    ("                buf_moe_workspace_f32.data_ptr(),  # [0]  workspace_f32\n"
     "                buf_residual.data_ptr() if li == 0 else buf_oproj_out.data_ptr(),  # [1] residual\n"
     "                norm_w1,                           # [2]  norm_weight_pre\n"
     "                buf_norm_scratch1.data_ptr(),      # [3]  norm_scratch_pre\n"
     "                qkv_weight_xcd,                    # [4]  qkv_weight\n"
     "                qkv_bias_xcd,                      # [5]  qkv_bias\n"
     "                attn_sinks,                        # [6]  attn_sinks (per-layer, bf16)\n"
     "                qkv_barrier_ptr,                   # [7]  qkv_barrier\n"
     "                buf_lse_acc.data_ptr(),             # [8]  lse_acc\n"
     "                oproj_weight_xcd,                  # [9]  oproj_weight\n"
     "                oproj_bias_xcd,                    # [10] oproj_bias\n"
     "                norm_w2,                           # [11] norm_weight_post\n"
     "                buf_norm_scratch2.data_ptr(),      # [12] norm_scratch_post\n"
     "                router_bf16_xcd,                   # [13] router_weight (bf16 per-XCD)\n"
     "                router_bias_xcd,                   # [14] router_bias (per-XCD)\n"
     "                logits_scratch_xcd,                # [15] logits_scratch (per-XCD)\n"
     "                counter_ptr,                       # [16] oproj_counters (counter base)\n"
     "                w13_weight_base,                   # [17] moe_gate_up_weight\n"
     "                w2_weight_base,                    # [18] moe_down_weight\n"
     "                w13_bias,                          # [19] moe_w13_bias\n"
     "                w2_bias,                           # [20] moe_w2_bias\n"
     "                buf_moe_barrier.data_ptr(),        # [21] moe_barrier\n"
     "                buf_swiglu_out.data_ptr(),         # [22] moe_swiglu_out\n"
     "                buf_o_acc_f32.data_ptr(),           # [23] o_acc_f32",
     "{emit_mirage_ptr_list('in')}", 1),

    ("                buf_x_output.data_ptr(),           # [0]  x_output (separate intermediate, NOT residual)\n"
     "                buf_k_cache[li].data_ptr(),        # [1]  k_cache\n"
     "                buf_v_cache[li].data_ptr(),        # [2]  v_cache\n"
     "                buf_q_workspace.data_ptr(),        # [3]  q_workspace\n"
     "                buf_attn_out.data_ptr(),           # [4]  o_acc (attn output)\n"
     "                buf_oproj_out.data_ptr(),          # [5]  attn_proj_out (= next layer's residual)\n"
     "                buf_topk_weight.data_ptr(),        # [6]  topk_weight\n"
     "                buf_routing_indices.data_ptr(),    # [7]  routing_indices\n"
     "                buf_active_expert_ids.data_ptr(),  # [8]  active_expert_ids\n"
     "                buf_topk_weight.data_ptr(),        # [9]  moe_routing_weight (= topk_weight)\n"
     "                buf_moe_workspace_f32.data_ptr(),  # [10] moe_workspace_f32",
     "{emit_mirage_ptr_list('out')}", 1),

    ("            assert len(mirage_in) == 24\n"
     "            assert len(mirage_out) == 11",
     "            assert len(mirage_in) == {len(MIRAGE_IN)}\n"
     "            assert len(mirage_out) == {len(MIRAGE_OUT)}", 1),
]
