from models.modeling_qwen3 import Qwen3ForCausalLM
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_model
import torch
import torch.distributed as dist
import argparse
import os, json
import math

DEFAULT_SAVE_DIR = os.path.join("outputs", "qwen3")
MAX_SAVE_TOKENS = 100

# print limitation
# torch.set_printoptions(threshold=2000)

def grid_for_rmsnorm_linear_layer(size: int, use_cutlass_kernel: bool = True, batch_size: int = 0):
    # 96 and 64 are enough to cover all Qwen3 model? Please update the method
    # if you meet any incompatibility.
    if size % 64 == 0 and not use_cutlass_kernel:
        return size // 64
    if size / 96 > 400:
        # TODO: An add-hoc workaround for linear kernel, both MPK ptx and
        # cutlass version will output unexpected result (not same out put for
        # same prompt) if the OUTPUT_SIZE is too big, try to figure it out.
        assert size % 256 == 0, "FATAL: Linear layer size not support, it's {size}."
        return size // 256
    if size % 96 == 0:
        return 96
    elif size % 64 == 0:
        return 64
    
def compute_dynamic_splitk(output_size, n_per_block, reduction_size, num_workers, k_per_block=256):
    """Compute optimal split-K to maximize CU utilization.

    Targets total_tasks >= num_workers. Prefers slight over-subscription
    over under-utilization (ceil instead of round).
    Ensures K_per_split is divisible by k_per_block for MFMA alignment.
    """
    import math
    tile_num = output_size // n_per_block
    if tile_num >= num_workers:
        return 1  # Already fully utilizing workers

    ideal = max(1, math.ceil(num_workers / tile_num))
    k_tiles = reduction_size // k_per_block  # Total K-tiles

    # Find nearest valid split-K that divides k_tiles
    # On tie, prefer larger split-K (better utilization)
    best = 1
    best_diff = abs(1 - ideal)
    for s in range(2, k_tiles + 1):
        if k_tiles % s == 0:
            diff = abs(s - ideal)
            if diff < best_diff or (diff == best_diff and s > best):
                best = s
                best_diff = diff

    return best

# Return the largest factor of m that is less than or equal to n
# This is used to determine the grid size
def max_factor_leq_n(m: int, n: int) -> int:
    max_factor = 1
    i = 1
    while i * i <= m:
        if m % i == 0:
            if i <= n:
                max_factor = max(max_factor, i)
            if m // i <= n:
                max_factor = max(max_factor, m // i)
        i += 1
    return max_factor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-mirage", action="store_true", help="Use Mirage kernels")
    parser.add_argument("--max-num-batched-tokens", default=8, type=int, help="Max number of tokens in a batch")
    parser.add_argument("--max-num-batched-requests", default=1, type=int, help="Max number of requests in a batch")
    parser.add_argument("--page-size", default=4096, type=int, help="Page size")
    parser.add_argument("--max-num-pages", default=16, type=int, help="Max num pages")
    parser.add_argument("--output-dir", help="Output files directory")
    parser.add_argument("--trace-name", default="", help="Perfetto trace output name")
    parser.add_argument(
        "--profiling", action="store_true", help="Use Profiler to generate trace"
    )
    parser.add_argument(
        "--num-layers", default=None, type=int, help="Override number of layers (for profiling)"
    )
    # lookahead or promptlookup
    parser.add_argument(
        "--spec-decode",
        default=None,
        choices=["promptlookup", "lookahead"],
        help="Enable speculative decoding with 'lookahead' or 'promptlookup' mode.",
    )
    parser.add_argument(
        "--ngram-size",
        default=3,
        type=int,
        help="Ngram size for lookahead spec decode",
    )
    parser.add_argument(
        "--max-seq-length",
        default=512,
        type=int,
        help="Max sequence length for lookahead spec decode",
    )
    parser.add_argument(
        "--spec-length",
        default=3,
        type=int,
        help="Spec length for lookahead spec decode",
    )

    parser.add_argument("--model-path", type=str, default=None, help="Path to a local model (necessary for multi-GPU demo)")
    parser.add_argument(
        "--model", type=str, default='Qwen/Qwen3-8B', help="Model path on hugging face"
    )
    parser.add_argument(
        "--no-use-cutlass-kernel",
        action="store_false",
        dest="use_cutlass_kernel",
        default=True,
        help="Not use the cutlass version kernel.",
    )
    parser.add_argument("--ignore-eos", action="store_true", help="Ignore eos token during generation")

    # -------- Args for CI tests ----------
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Decode cap for CI determinism")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--do-sample", dest="do_sample", action="store_true", help="Enable sampling (default off)")
    parser.add_argument(
        "--save-tokens",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Optionally dump first N generated token_ids, text, and latency to JSON. "
            "If path omitted, saves to outputs/qwen3/{torch_output.json|mpk_output.json}."
        ),
    )
    parser.add_argument("--prompt",
        type=str,
        default="Give me a short introduction to large language model.",
        help="Custom prompt text to generate from.",
    )

    parser.add_argument("--linear-only", action="store_true", help="Only register linear tasks (skip attention/silu) for L2 profiling")
    parser.add_argument("--split-kv-cache", action="store_true", default=True, help="Use split-kv cache (default: enabled)")
    parser.add_argument("--no-split-kv-cache", action="store_false", dest="split_kv_cache", help="Disable split-kv cache")
    args = parser.parse_args()
    # Force cutlass off on ROCm/MI300X — CUTLASS requires CUDA tensor cores.
    # This also fixes grid_for_rmsnorm_linear_layer to return size//64,
    # maximizing CU utilization (e.g. Gate+Up: 96→384 blocks on 304 CUs).
    if getattr(torch.version, "hip", None):
        args.use_cutlass_kernel = False
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
            filename = "mpk_output.json" if args.use_mirage else "torch_output.json"
            save_path = os.path.join(DEFAULT_SAVE_DIR, filename)
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
    print(f"world_size({world_size}) rank({rank})")
    model_name = args.model
    torch.set_default_dtype(torch.bfloat16)

    torch.cuda.set_device(rank)
    with torch.device("cuda"):
        if args.model_path is not None:
            # load model locally (necessary for multi-GPU case)
            print(f"Load model from model path: {args.model_path}")
            config = AutoConfig.from_pretrained(args.model_path)
            model = Qwen3ForCausalLM(config, world_size, args.max_num_pages, args.page_size)
            # load_model(
            #     model, f"{args.model_path}/model{rank}-mp{world_size}.safetensors"
            # )
            model = Qwen3ForCausalLM.from_pretrained(args.model_path, world_size, max_num_pages=args.max_num_pages, page_size=args.page_size).to("cuda")
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        else:
            extra_kwargs = {}
            if args.num_layers is not None:
                extra_kwargs["num_hidden_layers"] = args.num_layers
            model = Qwen3ForCausalLM.from_pretrained(model_name, world_size, max_num_pages=args.max_num_pages, page_size=args.page_size, **extra_kwargs).to("cuda")
            tokenizer = AutoTokenizer.from_pretrained(model_name)

    total_num_requests = 1 if not args.use_mirage else args.max_num_batched_requests
    # get all model weight tensors
    tokens = torch.full((total_num_requests, args.max_seq_length), 0, dtype=torch.long, device="cuda")

    prompt = args.prompt
    # This prompt is copied from https://github.com/apoorvumang/prompt-lookup-decoding/blob/main/demo-pld.ipynb
    code_text = """import numpy as np
                import matplotlib.pyplot as plt

                # Calculate the average
                average_throughput = np.mean(tokens_per_sec_arr)
                print(f"Average Throughput: {average_throughput} tokens/sec")

                # Plotting the histogram
                plt.hist(tokens_per_sec_arr, bins=20, color='blue', edgecolor='black', alpha=0.7)
                plt.title('Histogram of Throughput Values')
                plt.xlabel('Tokens per Second')
                plt.ylabel('Frequency')
                plt.axvline(average_throughput, color='red', linestyle='dashed', linewidth=1)
                plt.text(average_throughput*0.9, max(plt.ylim())*0.9, f'Average: {average_throughput:.2f}', color = 'red')
                plt.show()
                """
    #question = "Can you please change x axis to start from 0"
    #prompt = code_text + "\n" + question
    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    for r in range(total_num_requests):
        for i in range(model_inputs.input_ids.shape[-1]):
            tokens[r, i] = model_inputs.input_ids[0, i]
    prompt_lengths = torch.full((total_num_requests,), model_inputs.input_ids.shape[-1], dtype=torch.int, device="cuda")
    positions = torch.arange(32768).unsqueeze(0).to(model.device)
    position_embeddings = model.model.rotary_emb(positions)

    # get all model weight tensors
    input_tokens = torch.full((args.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda")
    output_tokens = torch.full((args.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda")
    prev_pos = 0

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    step = torch.full((total_num_requests, ), 0, dtype=torch.int32, device="cuda")
    num_new_tokens = torch.full((total_num_requests, ), 1, dtype=torch.int32, device="cuda")

    if args.use_mirage:
        import mirage as mi

        hidden_size = model.config.hidden_size
        intermediate_size = model.config.intermediate_size
        # pad vocab_size to facilitate task graph creation
        lm_head_weight = torch.cat(
            (
                model.lm_head.weight,
                torch.full(
                    (153600 - model.config.vocab_size, hidden_size), 0, device="cuda"
                ),
            ),
            0,
        )
        assert lm_head_weight.stride()[0] == hidden_size
        vocab_size = 153600
        num_q_heads = model.config.num_attention_heads
        num_kv_heads = model.config.num_key_value_heads
        num_local_q_heads = num_q_heads // world_size
        num_local_kv_heads = num_kv_heads // world_size
        head_dim = model.config.head_dim
        fused_outdim_1 = (num_q_heads + 2 * num_kv_heads) * head_dim
        fused_outdim_2 = 2 * intermediate_size
        num_kv_cache_chunks = max(1, (args.max_seq_length + 127) // 128)
        use_ck_fmha = int(os.environ.get("USE_CK_FMHA", "1")) == 1
        ck_fmha_num_kv_chunks = int(os.environ.get("CK_FMHA_CHUNKS", "1"))

        if args.profiling:
            profiler_tensor = torch.zeros(
                30000 * 1280, dtype=torch.uint64, device="cuda"
            ).contiguous()
        else:
            profiler_tensor = None
            
        spec_decode_config = mi.mpk.spec_decode_class(
            args.spec_decode,
            ngram_size=args.ngram_size,
            spec_length=args.spec_length,
        )
            
        num_workers, num_schedulers = mi.get_configurations_from_gpu(rank)
        qo_indptr_buffer = torch.empty(
            args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
        paged_kv_indptr_buffer = torch.empty(
            args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
        paged_kv_indices_buffer = torch.empty(
            args.max_num_pages, dtype=torch.int32, device="cuda")
        paged_kv_last_page_len_buffer = torch.empty(
            args.max_num_batched_requests, dtype=torch.int32, device="cuda")
        mpk = mi.PersistentKernel(
            mode="offline",
            world_size=world_size,
            mpi_rank=rank,
            num_workers=num_workers,
            num_local_schedulers=num_schedulers,
            num_remote_schedulers=0,
            max_seq_length=args.max_seq_length,
            max_num_batched_requests=args.max_num_batched_requests,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_pages=args.max_num_pages,
            page_size=args.page_size,
            eos_token_id=model.config.eos_token_id if not args.ignore_eos else 0x7FFFFFFF,
            meta_tensors={
                "step": step,
                "tokens": tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "num_new_tokens": num_new_tokens,
                "prompt_lengths": prompt_lengths,
                "qo_indptr_buffer": qo_indptr_buffer,
                "paged_kv_indptr_buffer": paged_kv_indptr_buffer,
                "paged_kv_indices_buffer": paged_kv_indices_buffer,
                "paged_kv_last_page_len_buffer": paged_kv_last_page_len_buffer,
            },
            profiler_tensor=profiler_tensor,
            trace_name=args.trace_name,
            spec_decode_config=spec_decode_config,
            use_cutlass_kernel=args.use_cutlass_kernel
        )
        
        if spec_decode_config and spec_decode_config.method == "promptlookup":
            all_tokens = mpk.attach_input(torch_tensor=tokens, name="all_tokens")
            num_tokens_extend = spec_decode_config.spec_length + 1
        else:
            num_tokens_extend = 1
        
        # TODO: Make the code run well even if 96 % max_num_batched_tokens != 0
        # assert(96 % args.max_num_batched_tokens == 0)
        
        x = mpk.attach_input(torch_tensor=input_tokens, name="input_token")
        cos_pos_embed = mpk.attach_input(
            torch_tensor=position_embeddings[0][0, :4096, :],
            name="cos_position_embedding",
        )
        sin_pos_embed = mpk.attach_input(
            torch_tensor=position_embeddings[1][0, :4096, :],
            name="sin_position_embedding",
        )

        y = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, hidden_size),
            dtype=mi.bfloat16,
            name="embed_out",
            io_category="cuda_tensor",
        )
        rmsnorm_out = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, hidden_size),
            dtype=mi.bfloat16,
            name="rmsnorm_out",
            io_category="cuda_tensor",
        )
        attn_in = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, fused_outdim_1 // world_size), # [6, 6144]
            dtype=mi.bfloat16,
            name="attn_in",
            io_category="cuda_tensor",
        )
        lse = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, num_kv_cache_chunks * num_local_q_heads // num_local_kv_heads, num_local_kv_heads),
            strides=(num_kv_cache_chunks * num_local_q_heads, 1, num_kv_cache_chunks * num_local_q_heads // num_local_kv_heads),
            dtype=mi.float32,
            name="lse",
            io_category="cuda_tensor",
        )
        attn_out_tmp = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, num_kv_cache_chunks * num_local_q_heads // num_local_kv_heads * head_dim, num_local_kv_heads),
            strides=(num_kv_cache_chunks * num_local_q_heads, 1, num_kv_cache_chunks * num_local_q_heads // num_local_kv_heads * head_dim),
            dtype=mi.bfloat16,
            name="attn_out_tmp",
            io_category="cuda_tensor",
        )
        attn_out = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, num_local_q_heads * head_dim),
            dtype=mi.bfloat16,
            name="attn_out",
            io_category="cuda_tensor",
        )
        if use_ck_fmha:
            num_qo_per_kv = num_local_q_heads // num_local_kv_heads
            q_ws_stride = num_local_q_heads * head_dim
            ck_fmha_q_ws_tensor = torch.zeros(
                args.max_num_batched_tokens, q_ws_stride,
                dtype=torch.bfloat16, device="cuda")
            ck_fmha_q_ws = mpk.attach_input(
                torch_tensor=ck_fmha_q_ws_tensor, name="ck_fmha_q_workspace")
            o_acc_dim1 = num_local_kv_heads * ck_fmha_num_kv_chunks * num_qo_per_kv * head_dim
            ck_fmha_o_acc_tensor = torch.zeros(
                args.max_num_batched_tokens, o_acc_dim1,
                dtype=torch.float32, device="cuda")
            ck_fmha_o_acc = mpk.attach_input(
                torch_tensor=ck_fmha_o_acc_tensor, name="ck_fmha_o_acc")
            lse_dim1 = num_local_kv_heads * ck_fmha_num_kv_chunks * num_qo_per_kv
            ck_fmha_lse_acc_tensor = torch.zeros(
                args.max_num_batched_tokens, lse_dim1,
                dtype=torch.float32, device="cuda")
            ck_fmha_lse_acc = mpk.attach_input(
                torch_tensor=ck_fmha_lse_acc_tensor, name="ck_fmha_lse_acc")
        attn_proj_out = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, hidden_size),
            dtype=mi.bfloat16,
            name="attn_proj_out",
            io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
        )
        allreduce_buf = mpk.new_tensor(
            dims=(world_size, args.max_num_batched_tokens, hidden_size),
            dtype=mi.bfloat16,
            name="all_reduce_buf",
            io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
        )
        attn_allreduce_out = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, hidden_size),
            dtype=mi.bfloat16,
            name="attn_allreduce_out",
            io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
        )
        mlp_mid = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, fused_outdim_2 // world_size),
            dtype=mi.bfloat16,
            name="mlp_mid",
            io_category="cuda_tensor",
        )
        silu_mul_out = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, intermediate_size // world_size),
            dtype=mi.bfloat16,
            name="silu_mul_out",
            io_category="cuda_tensor",
        )
        # Split-K workspace and done_counter for MI300X linear_with_residual
        is_rocm = bool(getattr(torch.version, "hip", None))
        USE_GANG = is_rocm and int(os.environ.get("USE_GANG", "0")) == 1
        USE_FUSED_SILU = is_rocm and int(os.environ.get("USE_FUSED_SILU", "0")) == 1
        USE_GANG_SPLITK = USE_GANG and int(os.environ.get("USE_GANG_SPLITK", "0")) == 1
        USE_GANG_KSPLIT = USE_GANG and int(os.environ.get("USE_GANG_KSPLIT", "0")) == 1
        # USE_GANG_N_TILE: route gang_linear and gang_linear_with_residual to
        # their N-tiling sister kernels (N as fast tile index). Used to A/B
        # against the default M-tiling variant to isolate L2 weight reuse.
        USE_GANG_N_TILE = USE_GANG and int(os.environ.get("USE_GANG_N_TILE", "0")) == 1
        USE_GANG_M_SPLIT = USE_GANG and int(os.environ.get("USE_GANG_M_SPLIT", "0")) == 1
        GANG_K_SPLITS = int(os.environ.get("GANG_K_SPLITS", "4"))
        GANG_TILE_N = int(os.environ.get("GANG_TILE_N", "64"))
        GANG_WGM = int(os.environ.get("GANG_WGM", "0"))  # HipKittens Algorithm 1 window height (0=full M-major)
        # M-tiles for gang linear:
        # bs>=64: m_tiles=1, m_per_tile=bs → 64×64×128 tile (32×32 MFMA, 2.4x faster)
        # bs<64:  m_per_tile=16 chunks → 16×64×256 tile (16×16 MFMA)
        if USE_GANG:
            bs = args.max_num_batched_tokens
            GANG_M_TILES = int(os.environ.get("GANG_M_TILES_OVERRIDE", 0)) or max(1, bs // 16)
            # Per-op override: use bigger tile for GATE_UP (most weight, benefits from larger tile)
            GANG_M_TILES_GATEUP = int(os.environ.get("GANG_M_TILES_GATEUP", 0)) or GANG_M_TILES
        else:
            GANG_M_TILES = 1
            GANG_M_TILES_GATEUP = 1
        if is_rocm:
            n_blocks = hidden_size // 64
            splitk_ws_torch = torch.zeros(
                (args.max_num_batched_tokens, hidden_size),
                dtype=torch.float32, device="cuda",
            )
            splitk_dc_torch = torch.zeros(
                (n_blocks, 1), dtype=torch.int32, device="cuda",
            )
            splitk_workspace = mpk.attach_input(
                torch_tensor=splitk_ws_torch, name="splitk_workspace",
            )
            splitk_done_counter = mpk.attach_input(
                torch_tensor=splitk_dc_torch, name="splitk_done_counter",
            )
            # gang_splitk reuses splitk_workspace and splitk_done_counter
        mlp_out = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, hidden_size),
            dtype=mi.bfloat16,
            name="mlp_out",
            io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
        )
        mlp_final = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, hidden_size),
            dtype=mi.bfloat16,
            name="mlp_final",
            io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
        )
        argmax_in = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, vocab_size),
            dtype=mi.bfloat16,
            name="argmax_in",
            io_category="cuda_tensor",
        )
        argmax_tasks = mpk.num_workers if vocab_size % mpk.num_workers == 0 else 240
        argmax_part_value = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, argmax_tasks),
            dtype=mi.bfloat16,
            name="argmax_part_value",
            io_category="cuda_tensor",
        )
        argmax_part_index = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, argmax_tasks),
            dtype=mi.int64,
            name="argmax_part_index",
            io_category="cuda_tensor",
        )
        argmax_out = mpk.attach_input(torch_tensor=output_tokens, name="output_token")
        #argmax_out = mpk.new_tensor(
        #    dims=(args.max_num_batched_tokens, 1),
        #    dtype=mi.int64,
        #    name="argmax_out",
        #    io_category="cuda_tensor",
        #)

        # add spec tokens layer
        if spec_decode_config:
            spec_tokens = mpk.draft_forward_layer_dispatcher(
                spec_decode_config = spec_decode_config, 
                tokens = all_tokens,
                grid_dim=(96, 1, 1),
                block_dim=(128, 1, 1),
            )
            x = spec_tokens
        # Add Embed
        w = mpk.attach_input(
            torch_tensor=model.model.embed_tokens.weight, name="embed_tokens"
        )
        
        mpk.embed_layer(
            input=x, 
            weight=w, 
            output=y, 
            # grid_dim=(max_factor_leq_n(hidden_size, 96 // args.max_num_batched_tokens), total_tokens_per_iter, 1), 
            grid_dim=(1, 1, 1), 
            block_dim=(128, 1, 1),
            input_source=1,
        )
        x = y
        for i, layer in enumerate(model.model.layers):
            # if i > 0:
            #     break
            # add rmsnorm + linear
            w_norm = mpk.attach_input(
                torch_tensor=layer.input_layernorm.weight,
                name=f"layer_{i}_input_layernorm",
            )
            w_q = mpk.attach_input(
                torch_tensor=layer.self_attn.q_proj.weight, name=f"layer_{i}_q_proj"
            )
            w_k = mpk.attach_input(
                torch_tensor=layer.self_attn.k_proj.weight, name=f"layer_{i}_k_proj"
            )
            w_v = mpk.attach_input(
                torch_tensor=layer.self_attn.v_proj.weight, name=f"layer_{i}_v_proj"
            )
            w_qkv = mpk.shuffle_tensors(
                inputs=[w_q, w_k, w_v],
                shuffled_dim=0,
                num_groups=model.config.num_key_value_heads // world_size,
                name=f"layer_{i}_qkv_proj",
            )
            if USE_GANG:
                mpk.gang_rmsnorm_layer(
                    input=x,
                    weight=w_norm,
                    output=rmsnorm_out,
                )
            else:
                mpk.rmsnorm_layer(
                    input=x,
                    weight=w_norm,
                    output=rmsnorm_out,
                    grid_dim=(mpk.max_num_batched_tokens, 1, 1),
                    block_dim=(128, 1, 1),
                )
            if USE_GANG_M_SPLIT:
                mpk.gang_linear_msplit_layer(
                    input=rmsnorm_out, weight=w_qkv, output=attn_in,
                    tile_n=GANG_TILE_N, output_stride=w_qkv.dim(0), m_tiles=GANG_M_TILES,
                    block_dim=(256, 1, 1))
            elif USE_GANG_N_TILE:
                mpk.gang_linear_n_tiling_layer(
                    input=rmsnorm_out,
                    weight=w_qkv,
                    output=attn_in,
                    tile_n=GANG_TILE_N,
                    output_stride=w_qkv.dim(0),
                    m_tiles=GANG_M_TILES,
                    wgn=GANG_WGM,
                    block_dim=(256, 1, 1),
                )
            elif USE_GANG:
                mpk.gang_linear_layer(
                    input=rmsnorm_out,
                    weight=w_qkv,
                    output=attn_in,
                    tile_n=GANG_TILE_N,
                    output_stride=w_qkv.dim(0),
                    m_tiles=GANG_M_TILES,
                    wgm=GANG_WGM,
                    block_dim=(256, 1, 1),
                )
            else:
                mpk.linear_layer(
                    input=rmsnorm_out,
                    weight=w_qkv,
                    output=attn_in,
                    grid_dim=(grid_for_rmsnorm_linear_layer(w_qkv.dim(0), args.use_cutlass_kernel), 1, 1),
                    block_dim=(128, 1, 1),
                )
            #mpk.rmsnorm_linear_layer(
            #    input=x,
            #    weight_norm=w_norm,
            #    weight_linear=w_qkv,
            #    output=attn_in,
            #    grid_dim=(grid_for_rmsnorm_linear_layer(w_qkv.dim(0)), 1, 1),
            #    block_dim=(128, 1, 1),
            #)
            if not args.linear_only:
              # add attention
              w_q_norm = mpk.attach_input(
                torch_tensor=layer.self_attn.q_norm.weight, name=f"layer_{i}_q_norm"
              )
              w_k_norm = mpk.attach_input(
                torch_tensor=layer.self_attn.k_norm.weight, name=f"layer_{i}_k_norm"
              )
              k_cache = mpk.attach_input(
                torch_tensor=model.model.kv_cache[0][i], name=f"layer_{i}_k_cache"
              )
              v_cache = mpk.attach_input(
                torch_tensor=model.model.kv_cache[1][i], name=f"layer_{i}_v_cache"
              )
              if spec_decode_config:
                mpk.single_batch_extend_attention_layer(
                    input=attn_in,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    q_norm=w_q_norm,
                    k_norm=w_k_norm,
                    cos_pos_embed=cos_pos_embed,
                    sin_pos_embed=sin_pos_embed,
                    output=attn_out,
                    grid_dim=(1, num_local_kv_heads, 1),
                    block_dim=(128, 1, 1),
                )
              elif use_ck_fmha and args.split_kv_cache:
                if USE_GANG and ck_fmha_num_kv_chunks == 1:
                    # Gang CK FMHA: fuses KV cache update + attention into 8 gang tasks
                    # With 1 chunk, writes bf16 directly to attn_out (no merge needed)
                    mpk.gang_paged_attention_split_kv_layer(
                        input=attn_in,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        q_norm=w_q_norm,
                        k_norm=w_k_norm,
                        cos_pos_embed=cos_pos_embed,
                        sin_pos_embed=sin_pos_embed,
                        lse=ck_fmha_lse_acc,
                        output=attn_out,
                        q_workspace=ck_fmha_q_ws,
                        attention_params=(num_local_q_heads,
                                         ck_fmha_num_kv_chunks,
                                         q_ws_stride),
                        block_dim=(256, 1, 1),
                    )
                elif USE_GANG and ck_fmha_num_kv_chunks > 1:
                    # Gang CK FMHA with split-KV: writes float32 to o_acc, then merge
                    mpk.gang_paged_attention_split_kv_layer(
                        input=attn_in,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        q_norm=w_q_norm,
                        k_norm=w_k_norm,
                        cos_pos_embed=cos_pos_embed,
                        sin_pos_embed=sin_pos_embed,
                        lse=ck_fmha_lse_acc,
                        output=ck_fmha_o_acc,
                        q_workspace=ck_fmha_q_ws,
                        attention_params=(num_local_q_heads,
                                         ck_fmha_num_kv_chunks,
                                         q_ws_stride),
                        block_dim=(256, 1, 1),
                    )
                    # Gang merge: reduce float32 chunks to bf16 output
                    mpk.gang_paged_attention_split_kv_merge_layer(
                        lse=ck_fmha_lse_acc,
                        output_tmp=ck_fmha_o_acc,
                        output=attn_out,
                        attention_params=(num_local_q_heads, head_dim,
                                         num_local_kv_heads,
                                         ck_fmha_num_kv_chunks),
                        block_dim=(256, 1, 1),
                    )
                else:
                    # Non-gang: separate KV cache update + CK FMHA
                    # Phase A: KV cache update + Q preprocessing (per-request)
                    mpk.kv_cache_update_layer(
                        input=attn_in,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        q_norm=w_q_norm,
                        k_norm=w_k_norm,
                        cos_pos_embed=cos_pos_embed,
                        sin_pos_embed=sin_pos_embed,
                        q_workspace=ck_fmha_q_ws,
                        grid_dim=(mpk.max_num_batched_requests, num_local_kv_heads, 1),
                        block_dim=(256, 1, 1),
                    )
                    # Phase B: CK FMHA attention
                    if ck_fmha_num_kv_chunks == 1:
                        # Direct bf16 output — kernel writes to attn_out, no merge needed
                        mpk.paged_attention_ck_fmha_layer(
                            q_workspace=ck_fmha_q_ws,
                            k_cache=k_cache,
                            v_cache=v_cache,
                            o_acc=attn_out,
                            lse_acc=ck_fmha_lse_acc,
                            attention_params=(num_local_q_heads,
                                             ck_fmha_num_kv_chunks,
                                             mpk.max_num_batched_requests),
                            grid_dim=(mpk.max_num_batched_requests, num_local_kv_heads, ck_fmha_num_kv_chunks),
                            block_dim=(256, 1, 1),
                        )
                    else:
                        # Float32 accumulator + merge
                        mpk.paged_attention_ck_fmha_layer(
                            q_workspace=ck_fmha_q_ws,
                            k_cache=k_cache,
                            v_cache=v_cache,
                            o_acc=ck_fmha_o_acc,
                            lse_acc=ck_fmha_lse_acc,
                            attention_params=(num_local_q_heads,
                                             ck_fmha_num_kv_chunks,
                                             mpk.max_num_batched_requests),
                            grid_dim=(mpk.max_num_batched_requests, num_local_kv_heads, ck_fmha_num_kv_chunks),
                            block_dim=(256, 1, 1),
                        )
                        # Phase C: Merge split-KV partial results
                        mpk.paged_attention_ck_fmha_merge_layer(
                            lse=ck_fmha_lse_acc,
                            output_tmp=ck_fmha_o_acc,
                            output=attn_out,
                            attention_params=(num_local_q_heads, head_dim,
                                             ck_fmha_num_kv_chunks,
                                             num_local_kv_heads),
                            grid_dim=(mpk.max_num_batched_requests, num_local_kv_heads, 1),
                            block_dim=(256, 1, 1),
                        )
              elif args.split_kv_cache:
                mpk.paged_attention_split_kv_layer(
                    input=attn_in,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    q_norm=w_q_norm,
                    k_norm=w_k_norm,
                    cos_pos_embed=cos_pos_embed,
                    sin_pos_embed=sin_pos_embed,
                    lse=lse,
                    output=attn_out_tmp,
                    attention_params=(num_local_q_heads, num_kv_cache_chunks),
                    grid_dim=(mpk.max_num_batched_requests, num_local_kv_heads, num_kv_cache_chunks),
                    block_dim=(128, 1, 1),
                )

                mpk.paged_attention_split_kv_merge_layer(
                    lse=lse,
                    output_tmp=attn_out_tmp,
                    output=attn_out,
                    attention_params=(num_local_q_heads, head_dim),
                    grid_dim=(mpk.max_num_batched_requests, num_local_kv_heads, 1),
                    block_dim=(128, 1, 1),
                )
              else:
                mpk.paged_attention_layer(
                    input=attn_in,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    q_norm=w_q_norm,
                    k_norm=w_k_norm,
                    cos_pos_embed=cos_pos_embed,
                    sin_pos_embed=sin_pos_embed,
                    output=attn_out,
                    grid_dim=(mpk.max_num_batched_requests, num_local_kv_heads, 1),
                    block_dim=(128, 1, 1),
                )
            else:
                # linear_only: skip attention, feed QKV output to o_proj
                attn_out = attn_in

            # add linear w/ residual
            w = mpk.attach_input(
                torch_tensor=layer.self_attn.o_proj.weight, name=f"layer_{i}_o_proj"
            )
            if USE_GANG_KSPLIT:
                mpk.gang_ksplit_linear_with_residual_layer(
                    input=attn_out,
                    weight=w,
                    residual=x,
                    workspace=splitk_workspace,
                    output=attn_proj_out,
                    tile_n=GANG_TILE_N,
                    output_stride=hidden_size,
                    k_splits=8,
                )
            elif USE_GANG_SPLITK:
                mpk.gang_splitk_linear_with_residual_layer(
                    input=attn_out,
                    weight=w,
                    residual=x,
                    workspace=splitk_workspace,
                    output=attn_proj_out,
                    tile_n=GANG_TILE_N,
                    output_stride=hidden_size,
                    k_splits=GANG_K_SPLITS,
                )
            elif USE_GANG_M_SPLIT:
                mpk.gang_linear_with_residual_msplit_layer(
                    input=attn_out, weight=w, residual=x, output=attn_proj_out,
                    tile_n=GANG_TILE_N, output_stride=hidden_size, m_tiles=GANG_M_TILES,
                    block_dim=(256, 1, 1))
            elif USE_GANG_N_TILE:
                mpk.gang_linear_with_residual_n_tiling_layer(
                    input=attn_out,
                    weight=w,
                    residual=x,
                    output=attn_proj_out,
                    tile_n=GANG_TILE_N,
                    output_stride=hidden_size,
                    m_tiles=GANG_M_TILES,
                    wgn=GANG_WGM,
                    block_dim=(256, 1, 1),
                )
            elif USE_GANG and not int(os.environ.get("GANG_SPLITK_RES", "0")):
                mpk.gang_linear_with_residual_layer(
                    input=attn_out,
                    weight=w,
                    residual=x,
                    output=attn_proj_out,
                    tile_n=GANG_TILE_N,
                    output_stride=hidden_size,
                    m_tiles=GANG_M_TILES,
                    wgm=GANG_WGM,
                    block_dim=(256, 1, 1),
                )
            elif (is_rocm or (USE_GANG and int(os.environ.get("GANG_SPLITK_RES", "0")))) and not int(os.environ.get("NOGANG_NOSPLITK", "0")):
                o_proj_n_blocks = hidden_size // 64
                o_proj_k_splits = compute_dynamic_splitk(
                    hidden_size, 64, hidden_size, mpk.num_workers)
                mpk.splitk_linear_res_atomic_layer(
                    input=attn_out,
                    weight=w,
                    residual=x,
                    workspace=splitk_workspace,
                    done_counter=splitk_done_counter,
                    output=attn_proj_out,
                    k_splits=o_proj_k_splits,
                    grid_dim=(o_proj_n_blocks, o_proj_k_splits, 1),
                    block_dim=(256, 1, 1),
                )
            else:
                mpk.linear_with_residual_layer(
                    input=attn_out,
                    weight=w,
                    residual=x,
                    output=attn_proj_out,
                    grid_dim=(hidden_size // 64, 1, 1),
                    block_dim=(256, 1, 1),
                )
            # reset residual input as x
            x = attn_proj_out
            # add allreduce if needed
            if world_size > 1:
                mpk.allreduce_layer(
                    input=attn_proj_out,
                    buffer=allreduce_buf,
                    output=attn_allreduce_out,
                    grid_dim=(hidden_size // 64, 1, 1),
                    block_dim=(128, 1, 1),
                )
                x = attn_allreduce_out
            # add rmsnorm_linear layer
            w_norm = mpk.attach_input(
                torch_tensor=layer.post_attention_layernorm.weight,
                name=f"layer_{i}_post_attn_layernorm",
            )
            w_gate_proj = mpk.attach_input(
                torch_tensor=layer.mlp.gate_proj.weight, name=f"layer_{i}_gate_proj"
            )
            w_up_proj = mpk.attach_input(
                torch_tensor=layer.mlp.up_proj.weight, name=f"layer_{i}_up_proj"
            )
            gateup_size = w_gate_proj.dim(0) + w_up_proj.dim(0)
            if is_rocm:
                # Use 128 cols/block to fit GateUp in 1 round (224 tasks ≤ 240 workers)
                rmsnorm_num_tasks = gateup_size // 128
            else:
                rmsnorm_num_tasks = grid_for_rmsnorm_linear_layer(gateup_size, args.use_cutlass_kernel)
            w_gatedup = mpk.shuffle_tensors(
                inputs=[w_gate_proj, w_up_proj],
                shuffled_dim=0,
                num_groups=rmsnorm_num_tasks//2,
                name=f"layer_{i}_gatedup_proj",
            )
            if USE_GANG:
                mpk.gang_rmsnorm_layer(
                    input=x,
                    weight=w_norm,
                    output=rmsnorm_out,
                )
            else:
                mpk.rmsnorm_layer(
                    input=x,
                    weight=w_norm,
                    output=rmsnorm_out,
                    grid_dim=(mpk.max_num_batched_tokens, 1, 1),
                    block_dim=(128, 1, 1),
                )
            if USE_FUSED_SILU and USE_GANG:
                # Fused gate_up GEMM + SiLU + mul → output is [bs, inter_size]
                mpk.gang_linear_silu_layer(
                    input=rmsnorm_out,
                    weight=w_gatedup,
                    output=silu_mul_out,
                    tile_n=GANG_TILE_N,
                    output_stride=intermediate_size // world_size,
                    m_tiles=GANG_M_TILES_GATEUP,
                    wgm=GANG_WGM,
                    block_dim=(256, 1, 1),
                )
            elif USE_FUSED_SILU:
                # CU-task fused gate_up GEMM + SiLU + mul
                mpk.linear_silu_layer(
                    input=rmsnorm_out,
                    weight=w_gatedup,
                    output=silu_mul_out,
                    output_stride=intermediate_size // world_size,
                    block_dim=(256, 1, 1),
                )
            elif USE_GANG_M_SPLIT:
                mpk.gang_linear_msplit_layer(
                    input=rmsnorm_out, weight=w_gatedup, output=mlp_mid,
                    tile_n=GANG_TILE_N, output_stride=w_gatedup.dim(0), m_tiles=GANG_M_TILES,
                    block_dim=(256, 1, 1))
            elif USE_GANG_N_TILE:
                mpk.gang_linear_n_tiling_layer(
                    input=rmsnorm_out,
                    weight=w_gatedup,
                    output=mlp_mid,
                    tile_n=GANG_TILE_N,
                    output_stride=w_gatedup.dim(0),
                    m_tiles=GANG_M_TILES,
                    wgn=GANG_WGM,
                    block_dim=(256, 1, 1),
                )
            elif USE_GANG:
                mpk.gang_linear_layer(
                    input=rmsnorm_out,
                    weight=w_gatedup,
                    output=mlp_mid,
                    tile_n=GANG_TILE_N,
                    output_stride=w_gatedup.dim(0),
                    m_tiles=GANG_M_TILES_GATEUP,
                    wgm=GANG_WGM,
                    block_dim=(256, 1, 1),
                )
            else:
                nogang_gateup_tiles = gateup_size // 64  # Match gang tile size (64 cols)
                mpk.linear_layer(
                    input=rmsnorm_out,
                    weight=w_gatedup,
                    output=mlp_mid,
                    grid_dim=(nogang_gateup_tiles, 1, 1),
                    block_dim=(128, 1, 1),
                )
            #mpk.rmsnorm_linear_layer(
            #    input=x,
            #    weight_norm=w_norm,
            #    weight_linear=w_gatedup,
            #    output=mlp_mid,
            #    grid_dim=(rmsnorm_num_tasks, 1, 1),
            #    block_dim=(128, 1, 1),
            #)
            # Toggle: fused vs unfused MLP down_proj
            USE_FUSED_SILU_MUL_LINEAR = False
            w = mpk.attach_input(
                torch_tensor=layer.mlp.down_proj.weight, name=f"layer_{i}_down_proj"
            )
            if USE_FUSED_SILU_MUL_LINEAR:
                # Fused silu_mul_linear_with_residual: eliminates intermediate buffer
                mpk.silu_mul_linear_with_residual_layer(
                    input=mlp_mid,
                    weight=w,
                    residual=x,
                    output=mlp_out,
                    grid_dim=(hidden_size // 64, 1, 1),
                    block_dim=(256, 1, 1),
                )
            else:
                # Original: separate silu_mul + linear_with_residual
                if not USE_FUSED_SILU:
                    if not args.linear_only:
                        mpk.silu_mul_layer(
                            input=mlp_mid,
                            output=silu_mul_out,
                            grid_dim=(rmsnorm_num_tasks//2, 1, 1),
                            block_dim=(128, 1, 1),
                        )
                    else:
                        silu_mul_out = mlp_mid
                if USE_GANG_KSPLIT:
                    mpk.gang_ksplit_linear_with_residual_layer(
                        input=silu_mul_out,
                        weight=w,
                        residual=x,
                        workspace=splitk_workspace,
                        output=mlp_out,
                        tile_n=GANG_TILE_N,
                        output_stride=hidden_size,
                        k_splits=8,
                    )
                elif USE_GANG_SPLITK:
                    mpk.gang_splitk_linear_with_residual_layer(
                        input=silu_mul_out,
                        weight=w,
                        residual=x,
                        workspace=splitk_workspace,
                        output=mlp_out,
                        tile_n=GANG_TILE_N,
                        output_stride=hidden_size,
                        k_splits=GANG_K_SPLITS,
                    )
                elif USE_GANG_M_SPLIT:
                    mpk.gang_linear_with_residual_msplit_layer(
                        input=silu_mul_out, weight=w, residual=x, output=mlp_out,
                        tile_n=GANG_TILE_N, output_stride=hidden_size, m_tiles=GANG_M_TILES,
                        block_dim=(256, 1, 1))
                elif USE_GANG_N_TILE:
                    mpk.gang_linear_with_residual_n_tiling_layer(
                        input=silu_mul_out,
                        weight=w,
                        residual=x,
                        output=mlp_out,
                        tile_n=GANG_TILE_N,
                        output_stride=hidden_size,
                        m_tiles=GANG_M_TILES,
                        wgn=GANG_WGM,
                        block_dim=(256, 1, 1),
                    )
                elif USE_GANG and not int(os.environ.get("GANG_SPLITK_RES", "0")):
                    mpk.gang_linear_with_residual_layer(
                        input=silu_mul_out,
                        weight=w,
                        residual=x,
                        output=mlp_out,
                        tile_n=GANG_TILE_N,
                        output_stride=hidden_size,
                        m_tiles=GANG_M_TILES,
                        wgm=GANG_WGM,
                        block_dim=(256, 1, 1),
                    )
                elif (is_rocm or (USE_GANG and int(os.environ.get("GANG_SPLITK_RES", "0")))) and not int(os.environ.get("NOGANG_NOSPLITK", "0")):
                    down_n_blocks = hidden_size // 64
                    down_reduction = model.config.intermediate_size // world_size
                    down_k_splits = compute_dynamic_splitk(
                        hidden_size, 64, down_reduction, mpk.num_workers)
                    mpk.splitk_linear_res_atomic_layer(
                        input=silu_mul_out,
                        weight=w,
                        residual=x,
                        workspace=splitk_workspace,
                        done_counter=splitk_done_counter,
                        output=mlp_out,
                        k_splits=down_k_splits,
                        grid_dim=(down_n_blocks, down_k_splits, 1),
                        block_dim=(256, 1, 1),
                    )
                else:
                    mpk.linear_with_residual_layer(
                        input=silu_mul_out,
                        weight=w,
                        residual=x,
                        output=mlp_out,
                        grid_dim=(hidden_size // 64, 1, 1),
                        block_dim=(256, 1, 1),
                    )
            # reset residual input as x
            x = mlp_out
            if world_size > 1:
                mpk.allreduce_layer(
                    input=mlp_out,
                    buffer=allreduce_buf,
                    output=mlp_final,
                    grid_dim=(hidden_size // 64, 1, 1),
                    block_dim=(128, 1, 1),
                )
                x = mlp_final

        # add rmsnorm_linear layer
        w_norm = mpk.attach_input(
            torch_tensor=model.model.norm.weight, name="model_norm_weight"
        )
        w_proj = mpk.attach_input(torch_tensor=lm_head_weight, name="lm_head")
        if USE_GANG:
            mpk.gang_rmsnorm_layer(
                input=x,
                weight=w_norm,
                output=rmsnorm_out,
            )
        else:
            mpk.rmsnorm_layer(
                input=x,
                weight=w_norm,
                output=rmsnorm_out,
                grid_dim=(mpk.max_num_batched_tokens, 1, 1),
                block_dim=(128, 1, 1),
            )
        mpk.linear_layer(
            input=rmsnorm_out,
            weight=w_proj,
            output=argmax_in,
            grid_dim=(argmax_tasks, 1, 1),
            block_dim=(128, 1, 1),
        )
        #mpk.rmsnorm_linear_layer(
        #    input=x,
        #    weight_norm=w_norm,
        #    weight_linear=w_proj,
        #    output=argmax_in,
        #    grid_dim=(grid_for_rmsnorm_linear_layer(w_proj.dim(0)), 1, 1),
        #    block_dim=(128, 1, 1),
        #)
        # add argmax layer
        if spec_decode_config and spec_decode_config.method == "promptlookup":
            argmax_partial_grid_dim = (max_factor_leq_n(153600, 96 // (spec_decode_config.spec_length + 1)), 
                                       spec_decode_config.spec_length + 1, 
                                       1)
            argmax_reduce_grid_dim = (1, spec_decode_config.spec_length + 1, 1)
        else:
            argmax_partial_grid_dim = (argmax_tasks, 1, 1)
            argmax_reduce_grid_dim = (1, 1, 1)
        mpk.argmax_partial_layer(
            input=argmax_in,
            output=(argmax_part_value, argmax_part_index),
            grid_dim=argmax_partial_grid_dim,
            block_dim=(128, 1, 1),
        )
        mpk.argmax_reduce_layer(
            input=(argmax_part_value, argmax_part_index),
            output=argmax_out,
            grid_dim=argmax_reduce_grid_dim,
            block_dim=(128, 1, 1),
        )
        if spec_decode_config:
            verify_out = mpk.verify_layer_dispatcher(
                spec_decode_config = spec_decode_config,
                spec_tokens = spec_tokens,
                target_output = argmax_out,
                grid_dim = (1, 1, 1),
                block_dim = (128, 1, 1),
            )

        results = mpk.kn_graph.generate_task_graph(num_gpus=world_size, my_gpu_id=rank)
        with open(f"task_graph_{rank}.json", "w") as f:
            f.write(results["json_file"])
        with open(f"kernel_{rank}.cu", "w") as f:
            f.write(results["cuda_code"])

        mpk.compile(output_dir=args.output_dir)

    # g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    warmup = 0
    # Decode up to user cap or buffer size
    output_len = args.max_new_tokens if args.max_new_tokens is not None else (tokens.size(1) - prompt_lengths[0].item())
    output_len = max(0, min(output_len, tokens.size(1) - prompt_lengths[0].item()))
    if not args.use_mirage:
        prompt_len = prompt_lengths[0].item()
        decode_limit = prompt_len + output_len
        for cur_pos in range(prompt_len, decode_limit):
            step.fill_(cur_pos - 1)
            input_ids = tokens[:, prev_pos:cur_pos]
            cos_embeddings = position_embeddings[0][:, prev_pos:cur_pos]
            sin_embeddings = position_embeddings[1][:, prev_pos:cur_pos]
            logits = model.forward(
                input_ids=input_ids,
                position_embeddings=(cos_embeddings, sin_embeddings),
                step=step,
                stream=stream,
            )
            next_token = logits.argmax(dim=-1)
            next_token = next_token[0, -1]
            tokens[0, cur_pos] = next_token
            prev_pos = cur_pos
            if next_token == model.config.eos_token_id:
                break
            if cur_pos == prompt_len + warmup:
                torch.cuda.synchronize()
                starter.record()

        ender.record()
        torch.cuda.synchronize()
        run_time = starter.elapsed_time(ender)

        end_idx = prev_pos + 1
        generated_ids = tokens[:, :end_idx]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
        print(
            "Prompt length {}, generate length {}, per-token latency {} ms".format(
                prompt_len, cur_pos - prompt_len, run_time / (cur_pos - prompt_len)
            )
        )
        
        # -------- CI dumps outputs to json files ----------
        if save_path and rank == 0:
            tokens_generated = max(0, end_idx - prompt_len)
            per_tok_ms = run_time / max(tokens_generated, 1)
            slice_end = min(end_idx, prompt_len + MAX_SAVE_TOKENS)
            token_ids = tokens[0, prompt_len:slice_end].tolist()
            out = {
                "token_ids": token_ids,
                "text": tokenizer.decode(tokens[0, :end_idx], skip_special_tokens=True),
                "latency_ms_per_token": per_tok_ms,
                "prompt_length": prompt_len,
                "generate_length": tokens_generated,
                "mode": "torch",
            }
            with open(save_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Saved tokens to {save_path}")

    else:
        starter.record()
        mpk()
        ender.record()
        torch.cuda.synchronize()
        run_time = starter.elapsed_time(ender)

        # Read and print XCD affinity verification stats
        import sys
        print("[DEMO] About to call read_xcd_verify_func", flush=True)
        sys.stdout.flush()
        if hasattr(mpk, 'read_xcd_verify_func'):
            print("[DEMO] Calling read_xcd_verify_func now", flush=True)
            sys.stdout.flush()
            mpk.read_xcd_verify_func()
            sys.stdout.flush()
            print("[DEMO] read_xcd_verify_func completed", flush=True)
        else:
            print("[DEMO] read_xcd_verify_func not found", flush=True)

        print("tokens.shape = ", tokens.shape, flush=True)
        for r in range(total_num_requests):
            generated_ids = tokens[r, : step[r] + 1]
            # Filter out invalid token IDs (e.g. -1 sentinel) before decoding
            valid_ids = generated_ids[generated_ids >= 0]
            response = tokenizer.decode(valid_ids, skip_special_tokens=True)
            print(response)
        
        # Verify outputs
        prompt_len = prompt_lengths[0].item()

        # 1) Cross-batch comparison (batch 0 vs all others — catches gang bugs)
        if total_num_requests > 1:
            ref_len = step[0].item() + 1
            ref_tokens = tokens[0, :ref_len]
            all_match = True
            for r in range(1, total_num_requests):
                r_len = step[r].item() + 1
                if r_len != ref_len:
                    print(f"[VERIFY] batch {r} vs batch 0: length mismatch ({r_len} vs {ref_len})")
                    all_match = False
                    continue
                m = (tokens[r, :ref_len] != ref_tokens).nonzero(as_tuple=True)[0]
                if len(m) > 0:
                    first = m[0].item()
                    print(f"[VERIFY] batch {r} vs batch 0: MISMATCH at token {first}/{ref_len} "
                          f"(batch0={ref_tokens[first].item()}, batch{r}={tokens[r, first].item()}) "
                          f"total mismatches={len(m)}")
                    all_match = False
                else:
                    print(f"[VERIFY] batch {r} vs batch 0: OK ({ref_len} tokens match)")
            print(f"Cross-batch correctness: {all_match}")

        # Calculate separate prefill and decode metrics
        prompt_len = prompt_lengths[0].item()
        total_tokens = step.max().item() + 1
        generated_tokens = total_tokens - prompt_len

        # Estimate prefill vs decode iterations
        # Prefill processes max_num_batched_tokens per iteration
        # Decode processes 1 token per iteration
        prefill_iterations = math.ceil(prompt_len / args.max_num_batched_tokens)
        decode_iterations = generated_tokens
        total_iterations = prefill_iterations + decode_iterations

        # Estimate time split (rough approximation)
        if total_iterations > 0:
            avg_time_per_iter = run_time / total_iterations
            prefill_time_est = prefill_iterations * avg_time_per_iter
            decode_time_est = decode_iterations * avg_time_per_iter
        else:
            prefill_time_est = 0
            decode_time_est = 0

        print("=" * 80)
        print(f"Prefill: {prompt_len} tokens in {prefill_iterations} iterations ≈ {prefill_time_est:.1f}ms total ({prefill_time_est/max(prompt_len,1):.3f}ms/token)")
        print(f"Decode: {generated_tokens} tokens in {decode_iterations} iterations ≈ {decode_time_est:.1f}ms total ({decode_time_est/max(generated_tokens,1):.3f}ms/token)")
        print(f"Combined: {total_tokens} tokens, per-token latency: {run_time / total_tokens:.3f} ms")
        print("=" * 80)

        # -------- CI dumps outputs to json files ----------
        if save_path and rank == 0:
            end_idx = step[0].item() + 1
            prompt_len = prompt_lengths[0].item()
            tokens_generated = max(0, end_idx - prompt_len)
            per_tok_ms = run_time / max(tokens_generated, 1)
            slice_end = min(end_idx, prompt_len + MAX_SAVE_TOKENS)
            token_ids = tokens[0, prompt_len:slice_end].tolist()

            # Filter out invalid token IDs (e.g. -1 sentinel) before decoding
            all_tokens = tokens[0, :end_idx]
            valid_tokens = all_tokens[all_tokens >= 0]
            response_text = tokenizer.decode(valid_tokens, skip_special_tokens=True)
            out = {
                "token_ids": token_ids,
                "text": response_text,
                "latency_ms_per_token": per_tok_ms,
                "prompt_length": prompt_len,
                "generate_length": tokens_generated,
                "mode": "mpk",
            }
            with open(save_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Saved tokens to {save_path}")

    if world_size > 1:
        dist.destroy_process_group()
