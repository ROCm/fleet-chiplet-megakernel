"""Qwen3 demo using the Gang DSL.

This is functionally equivalent to demo.py but uses mirage.mpk.dsl
to eliminate workspace allocation boilerplate and manual strategy dispatch.
The model graph construction shrinks from ~600 lines to ~150 lines.

Usage:
    # Same arguments as demo.py
    HIP_VISIBLE_DEVICES=2 python3 demo_dsl.py --use-mirage --max-num-batched-tokens 8 --max-num-batched-requests 1
"""

from models.modeling_qwen3 import Qwen3ForCausalLM
from transformers import AutoTokenizer, AutoConfig
import torch
import torch.distributed as dist
import argparse
import os, json, math

import mirage as mi
import mirage.mpk.dsl as dsl

DEFAULT_SAVE_DIR = os.path.join("outputs", "qwen3")
MAX_SAVE_TOKENS = 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-num-batched-tokens", default=8, type=int)
    parser.add_argument("--max-num-batched-requests", default=1, type=int)
    parser.add_argument("--page-size", default=4096, type=int)
    parser.add_argument("--max-num-pages", default=16, type=int)
    parser.add_argument("--output-dir", help="Output files directory")
    parser.add_argument("--trace-name", default="")
    parser.add_argument("--profiling", action="store_true")
    parser.add_argument("--max-seq-length", default=512, type=int)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model", type=str, default='Qwen/Qwen3-8B')
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt", type=str,
                        default="Give me a short introduction to large language model.")
    parser.add_argument("--split-kv-cache", action="store_true", default=True)
    parser.add_argument("--no-split-kv-cache", action="store_false", dest="split_kv_cache")
    parser.add_argument("--save-tokens", nargs="?", const="auto", default=None)
    args = parser.parse_args()

    # Force cutlass off on ROCm/MI300X
    is_rocm = bool(getattr(torch.version, "hip", None))

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        world_size = comm.Get_size()
        rank = comm.Get_rank()
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
    except ImportError:
        world_size = 1
        rank = 0

    if args.save_tokens:
        if args.save_tokens == "auto":
            save_path = os.path.join(DEFAULT_SAVE_DIR, "mpk_dsl_output.json")
        else:
            save_path = args.save_tokens
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    else:
        save_path = None

    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
    global print
    if rank != 0:
        print = lambda *_, **__: None

    print("Input arguments:", args)
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(rank)

    with torch.device("cuda"):
        if args.model_path is not None:
            model = Qwen3ForCausalLM.from_pretrained(
                args.model_path, world_size,
                max_num_pages=args.max_num_pages, page_size=args.page_size,
            ).to("cuda")
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        else:
            model = Qwen3ForCausalLM.from_pretrained(
                args.model, world_size,
                max_num_pages=args.max_num_pages, page_size=args.page_size,
            ).to("cuda")
            tokenizer = AutoTokenizer.from_pretrained(args.model)

    total_num_requests = args.max_num_batched_requests
    tokens = torch.full((total_num_requests, args.max_seq_length), 0,
                        dtype=torch.long, device="cuda")
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": args.prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    for r in range(total_num_requests):
        for i in range(model_inputs.input_ids.shape[-1]):
            tokens[r, i] = model_inputs.input_ids[0, i]
    prompt_lengths = torch.full((total_num_requests,), model_inputs.input_ids.shape[-1],
                                dtype=torch.int, device="cuda")
    positions = torch.arange(32768).unsqueeze(0).to(model.device)
    position_embeddings = model.model.rotary_emb(positions)

    input_tokens = torch.full((args.max_num_batched_tokens, 1), 0,
                              dtype=torch.long, device="cuda")
    output_tokens = torch.full((args.max_num_batched_tokens, 1), 0,
                               dtype=torch.long, device="cuda")
    step = torch.full((total_num_requests,), 0, dtype=torch.int32, device="cuda")
    num_new_tokens = torch.full((total_num_requests,), 1, dtype=torch.int32, device="cuda")
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # ---- Model dimensions ----
    hidden_size = model.config.hidden_size
    intermediate_size = model.config.intermediate_size
    num_q_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    num_local_q_heads = num_q_heads // world_size
    num_local_kv_heads = num_kv_heads // world_size
    head_dim = model.config.head_dim
    fused_outdim_1 = (num_q_heads + 2 * num_kv_heads) * head_dim
    fused_outdim_2 = 2 * intermediate_size
    num_kv_cache_chunks = max(1, (args.max_seq_length + 127) // 128)

    lm_head_weight = torch.cat((
        model.lm_head.weight,
        torch.full((153600 - model.config.vocab_size, hidden_size), 0, device="cuda"),
    ), 0)
    vocab_size = 153600

    if args.profiling:
        profiler_tensor = torch.zeros(30000 * 128, dtype=torch.uint64, device="cuda").contiguous()
    else:
        profiler_tensor = None

    num_workers, num_schedulers = mi.get_configurations_from_gpu(rank)
    qo_indptr_buffer = torch.empty(args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
    paged_kv_indptr_buffer = torch.empty(args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
    paged_kv_indices_buffer = torch.empty(args.max_num_pages, dtype=torch.int32, device="cuda")
    paged_kv_last_page_len_buffer = torch.empty(args.max_num_batched_requests, dtype=torch.int32, device="cuda")

    mpk = mi.PersistentKernel(
        mode="offline",
        world_size=world_size, mpi_rank=rank,
        num_workers=num_workers, num_local_schedulers=num_schedulers,
        num_remote_schedulers=0,
        max_seq_length=args.max_seq_length,
        max_num_batched_requests=args.max_num_batched_requests,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_pages=args.max_num_pages, page_size=args.page_size,
        eos_token_id=model.config.eos_token_id if not args.ignore_eos else 0x7FFFFFFF,
        meta_tensors={
            "step": step, "tokens": tokens,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "num_new_tokens": num_new_tokens, "prompt_lengths": prompt_lengths,
            "qo_indptr_buffer": qo_indptr_buffer,
            "paged_kv_indptr_buffer": paged_kv_indptr_buffer,
            "paged_kv_indices_buffer": paged_kv_indices_buffer,
            "paged_kv_last_page_len_buffer": paged_kv_last_page_len_buffer,
        },
        profiler_tensor=profiler_tensor,
        trace_name=args.trace_name,
        spec_decode_config=None,
        use_cutlass_kernel=False,
    )

    # ================================================================
    # DSL: Model graph construction
    # ================================================================
    g = dsl.Graph(mpk)

    # Global inputs
    x_in = g.input(input_tokens, "input_token")
    cos_pe = g.input(position_embeddings[0][0, :4096, :], "cos_position_embedding")
    sin_pe = g.input(position_embeddings[1][0, :4096, :], "sin_position_embedding")

    # Shared intermediate tensors
    bs = args.max_num_batched_tokens
    y = g.intermediate((bs, hidden_size), name="embed_out")
    rmsnorm_out = g.intermediate((bs, hidden_size), name="rmsnorm_out")
    attn_in = g.intermediate((bs, fused_outdim_1 // world_size), name="attn_in")
    lse = dsl.Tensor(
        mpk.new_tensor(
            dims=(bs, num_kv_cache_chunks * num_local_q_heads // num_local_kv_heads, num_local_kv_heads),
            strides=(num_kv_cache_chunks * num_local_q_heads, 1,
                     num_kv_cache_chunks * num_local_q_heads // num_local_kv_heads),
            dtype=mi.float32, name="lse", io_category="cuda_tensor",
        ), "lse", g)
    attn_out_tmp = dsl.Tensor(
        mpk.new_tensor(
            dims=(bs, num_kv_cache_chunks * num_local_q_heads // num_local_kv_heads * head_dim, num_local_kv_heads),
            strides=(num_kv_cache_chunks * num_local_q_heads, 1,
                     num_kv_cache_chunks * num_local_q_heads // num_local_kv_heads * head_dim),
            dtype=mi.bfloat16, name="attn_out_tmp", io_category="cuda_tensor",
        ), "attn_out_tmp", g)
    attn_out = g.intermediate((bs, num_local_q_heads * head_dim), name="attn_out")

    # CK FMHA workspace tensors
    use_ck_fmha = is_rocm
    ck_fmha_num_kv_chunks = 1
    if use_ck_fmha:
        num_qo_per_kv = num_local_q_heads // num_local_kv_heads
        ck_fmha_q_ws_torch = torch.zeros(bs, num_local_q_heads * head_dim,
                                         dtype=torch.bfloat16, device="cuda")
        ck_fmha_q_ws = dsl.Tensor(
            mpk.attach_input(torch_tensor=ck_fmha_q_ws_torch,
                             name="ck_fmha_q_workspace"),
            "ck_fmha_q_ws", g)
        o_acc_dim1 = num_local_kv_heads * ck_fmha_num_kv_chunks * num_qo_per_kv * head_dim
        ck_fmha_o_acc_torch = torch.zeros(bs, o_acc_dim1,
                                          dtype=torch.float32, device="cuda")
        ck_fmha_o_acc = dsl.Tensor(
            mpk.attach_input(torch_tensor=ck_fmha_o_acc_torch,
                             name="ck_fmha_o_acc"),
            "ck_fmha_o_acc", g)
        lse_dim1 = num_local_kv_heads * ck_fmha_num_kv_chunks * num_qo_per_kv
        ck_fmha_lse_torch = torch.zeros(bs, lse_dim1,
                                        dtype=torch.float32, device="cuda")
        ck_fmha_lse = dsl.Tensor(
            mpk.attach_input(torch_tensor=ck_fmha_lse_torch,
                             name="ck_fmha_lse_acc"),
            "ck_fmha_lse", g)

    io_cat = "nvshmem_tensor" if world_size > 1 else "cuda_tensor"
    attn_proj_out = dsl.Tensor(
        mpk.new_tensor(dims=(bs, hidden_size), dtype=mi.bfloat16,
                       name="attn_proj_out", io_category=io_cat),
        "attn_proj_out", g)
    mlp_mid = g.intermediate((bs, fused_outdim_2 // world_size), name="mlp_mid")
    silu_mul_out = g.intermediate((bs, intermediate_size // world_size), name="silu_mul_out")
    mlp_out = dsl.Tensor(
        mpk.new_tensor(dims=(bs, hidden_size), dtype=mi.bfloat16,
                       name="mlp_out", io_category=io_cat),
        "mlp_out", g)
    argmax_in = g.intermediate((bs, vocab_size), name="argmax_in")
    argmax_tasks = mpk.num_workers if vocab_size % mpk.num_workers == 0 else 240
    argmax_pv = dsl.Tensor(
        mpk.new_tensor(dims=(bs, argmax_tasks), dtype=mi.bfloat16,
                       name="argmax_part_value", io_category="cuda_tensor"),
        "argmax_part_value", g)
    argmax_pi = dsl.Tensor(
        mpk.new_tensor(dims=(bs, argmax_tasks), dtype=mi.int64,
                       name="argmax_part_index", io_category="cuda_tensor"),
        "argmax_part_index", g)
    argmax_out = dsl.Tensor(
        mpk.attach_input(torch_tensor=output_tokens, name="output_token"),
        "argmax_out", g)

    if world_size > 1:
        allreduce_buf = dsl.Tensor(
            mpk.new_tensor(dims=(world_size, bs, hidden_size), dtype=mi.bfloat16,
                           name="all_reduce_buf", io_category=io_cat),
            "allreduce_buf", g)
        attn_allreduce_out = dsl.Tensor(
            mpk.new_tensor(dims=(bs, hidden_size), dtype=mi.bfloat16,
                           name="attn_allreduce_out", io_category=io_cat),
            "attn_allreduce_out", g)
        mlp_final = dsl.Tensor(
            mpk.new_tensor(dims=(bs, hidden_size), dtype=mi.bfloat16,
                           name="mlp_final", io_category=io_cat),
            "mlp_final", g)

    # Embedding
    w_embed = g.input(model.model.embed_tokens.weight, "embed_tokens")
    dsl.embed(x_in, w_embed, y, input_source=1)
    x = y

    # ---- Transformer layers ----
    use_gang = is_rocm and os.environ.get("USE_GANG", "1") == "1"
    qkv_size = fused_outdim_1 // world_size
    gateup_size = fused_outdim_2 // world_size

    for i, layer in enumerate(model.model.layers):
        # RMSNorm → QKV linear
        w_norm = g.input(layer.input_layernorm.weight, f"layer_{i}_input_layernorm")
        w_q = g.input(layer.self_attn.q_proj.weight, f"layer_{i}_q_proj")
        w_k = g.input(layer.self_attn.k_proj.weight, f"layer_{i}_k_proj")
        w_v = g.input(layer.self_attn.v_proj.weight, f"layer_{i}_v_proj")
        w_qkv = g.shuffle_inputs(
            [w_q.dt, w_k.dt, w_v.dt],
            num_groups=num_kv_heads // world_size,
            name=f"layer_{i}_qkv_proj",
        )

        dsl.rmsnorm(x, w_norm, rmsnorm_out)
        dsl.linear(rmsnorm_out, w_qkv, attn_in, output_stride=qkv_size)

        # Attention
        w_q_norm = g.input(layer.self_attn.q_norm.weight, f"layer_{i}_q_norm")
        w_k_norm = g.input(layer.self_attn.k_norm.weight, f"layer_{i}_k_norm")
        k_cache = g.input(model.model.kv_cache[0][i], f"layer_{i}_k_cache")
        v_cache = g.input(model.model.kv_cache[1][i], f"layer_{i}_v_cache")

        if use_ck_fmha and args.split_kv_cache:
            dsl.paged_attention_ck_fmha(
                attn_in, k_cache, v_cache, w_q_norm, w_k_norm,
                cos_pe, sin_pe, ck_fmha_q_ws, ck_fmha_o_acc, ck_fmha_lse,
                attn_out,
                num_q_heads=num_local_q_heads,
                num_kv_heads=num_local_kv_heads,
                head_dim=head_dim,
                num_kv_chunks=ck_fmha_num_kv_chunks,
            )
        elif args.split_kv_cache:
            dsl.paged_attention_split_kv(
                attn_in, k_cache, v_cache, w_q_norm, w_k_norm,
                cos_pe, sin_pe, lse, attn_out_tmp, attn_out,
                num_q_heads=num_local_q_heads,
                num_kv_chunks=num_kv_cache_chunks,
                num_kv_heads=num_local_kv_heads,
                head_dim=head_dim,
                gang=False,
            )
        else:
            mpk.paged_attention_layer(
                input=attn_in.dt, k_cache=k_cache.dt, v_cache=v_cache.dt,
                q_norm=w_q_norm.dt, k_norm=w_k_norm.dt,
                cos_pos_embed=cos_pe.dt, sin_pos_embed=sin_pe.dt,
                output=attn_out.dt,
                grid_dim=(mpk.max_num_batched_requests, num_local_kv_heads, 1),
                block_dim=(128, 1, 1),
            )

        # O-proj with residual
        w_o = g.input(layer.self_attn.o_proj.weight, f"layer_{i}_o_proj")
        dsl.linear(attn_out, w_o, attn_proj_out, residual=x)
        x = attn_proj_out

        if world_size > 1:
            mpk.allreduce_layer(
                input=attn_proj_out.dt, buffer=allreduce_buf.dt,
                output=attn_allreduce_out.dt,
                grid_dim=(hidden_size // 64, 1, 1), block_dim=(128, 1, 1),
            )
            x = attn_allreduce_out

        # RMSNorm → Gate+Up linear
        w_norm2 = g.input(layer.post_attention_layernorm.weight,
                          f"layer_{i}_post_attn_layernorm")
        w_gate = g.input(layer.mlp.gate_proj.weight, f"layer_{i}_gate_proj")
        w_up = g.input(layer.mlp.up_proj.weight, f"layer_{i}_up_proj")
        gateup_tasks = gateup_size // 128 if is_rocm else gateup_size // 64
        w_gatedup = g.shuffle_inputs(
            [w_gate.dt, w_up.dt],
            num_groups=gateup_tasks // 2,
            name=f"layer_{i}_gatedup_proj",
        )

        dsl.rmsnorm(x, w_norm2, rmsnorm_out)
        if use_gang:
            # Fused gate+up GEMM + SiLU+mul → output is [bs, inter_size]
            dsl.linear_silu(rmsnorm_out, w_gatedup, silu_mul_out,
                            output_stride=intermediate_size // world_size)
        else:
            dsl.linear(rmsnorm_out, w_gatedup, mlp_mid,
                       output_stride=gateup_size)
            dsl.silu_mul(mlp_mid, silu_mul_out)

        # Down-proj with residual
        w_down = g.input(layer.mlp.down_proj.weight, f"layer_{i}_down_proj")
        dsl.linear(silu_mul_out, w_down, mlp_out, residual=x)
        x = mlp_out

        if world_size > 1:
            mpk.allreduce_layer(
                input=mlp_out.dt, buffer=allreduce_buf.dt,
                output=mlp_final.dt,
                grid_dim=(hidden_size // 64, 1, 1), block_dim=(128, 1, 1),
            )
            x = mlp_final

    # Final RMSNorm → LM head → Argmax
    w_final_norm = g.input(model.model.norm.weight, "model_norm_weight")
    w_lm_head = g.input(lm_head_weight, "lm_head")
    dsl.rmsnorm(x, w_final_norm, rmsnorm_out)
    dsl.linear(rmsnorm_out, w_lm_head, argmax_in, strategy="standard",
               num_tasks=argmax_tasks)
    dsl.argmax(argmax_in, argmax_pv, argmax_pi, argmax_out, num_tasks=argmax_tasks)

    # ================================================================
    # Compile and run
    # ================================================================
    results = mpk.kn_graph.generate_task_graph(num_gpus=world_size, my_gpu_id=rank)
    with open(f"task_graph_{rank}.json", "w") as f:
        f.write(results["json_file"])
    with open(f"kernel_{rank}.cu", "w") as f:
        f.write(results["cuda_code"])

    mpk.compile(output_dir=args.output_dir)

    starter.record()
    mpk()
    ender.record()
    torch.cuda.synchronize()
    run_time = starter.elapsed_time(ender)

    print("tokens.shape =", tokens.shape, flush=True)
    for r in range(total_num_requests):
        generated_ids = tokens[r, : step[r] + 1]
        valid_ids = generated_ids[generated_ids >= 0]
        response = tokenizer.decode(valid_ids, skip_special_tokens=True)
        print(response)

    prompt_len = prompt_lengths[0].item()
    total_tokens = step.max().item() + 1
    generated_tokens = total_tokens - prompt_len
    prefill_iterations = math.ceil(prompt_len / args.max_num_batched_tokens)
    decode_iterations = generated_tokens
    total_iterations = prefill_iterations + decode_iterations

    if total_iterations > 0:
        avg_time_per_iter = run_time / total_iterations
        prefill_time_est = prefill_iterations * avg_time_per_iter
        decode_time_est = decode_iterations * avg_time_per_iter
    else:
        prefill_time_est = decode_time_est = 0

    print("=" * 80)
    print(f"Prefill: {prompt_len} tokens ≈ {prefill_time_est:.1f}ms ({prefill_time_est/max(prompt_len,1):.3f}ms/token)")
    print(f"Decode: {generated_tokens} tokens ≈ {decode_time_est:.1f}ms ({decode_time_est/max(generated_tokens,1):.3f}ms/token)")
    print(f"Combined: {total_tokens} tokens, per-token latency: {run_time / total_tokens:.3f} ms")
    print("=" * 80)

    if save_path and rank == 0:
        end_idx = step[0].item() + 1
        tokens_generated = max(0, end_idx - prompt_len)
        per_tok_ms = run_time / max(tokens_generated, 1)
        slice_end = min(end_idx, prompt_len + MAX_SAVE_TOKENS)
        token_ids = tokens[0, prompt_len:slice_end].tolist()
        all_toks = tokens[0, :end_idx]
        valid_toks = all_toks[all_toks >= 0]
        out = {
            "token_ids": token_ids,
            "text": tokenizer.decode(valid_toks, skip_special_tokens=True),
            "latency_ms_per_token": per_tok_ms,
            "prompt_length": prompt_len,
            "generate_length": tokens_generated,
            "mode": "mpk_dsl",
        }
        with open(save_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved tokens to {save_path}")

    if world_size > 1:
        dist.destroy_process_group()
