from safetensors.torch import load_model
import torch

from ..utils import grid_for_rmsnorm_linear_layer, shuffle_tensors, inplace_shuffle_tensors
from ..graph_builder import GraphBuilder, MirageModelConfig
from ...persistent_kernel import PersistentKernel
from ...model_registry import register_model_builder
from ....core import bfloat16, int64

from typing import Optional
import os

@register_model_builder("Qwen3", "Qwen/Qwen3-8B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-14B", "Qwen/Qwen3-0.6B")
class Qwen3Builder(GraphBuilder):
    def __init__(self, mpk: PersistentKernel, weights: Optional[dict] = None):
        super().__init__(mpk, weights)
        self.max_num_pages = mpk.max_num_pages
        self.page_size = mpk.page_size
        self.world_size = mpk.world_size
        self.input_tokens = mpk.meta_tensors["input_tokens"]
        self.output_tokens = mpk.meta_tensors["output_tokens"]
        self.tokenizer = None
        self.model_name: str = None
        self.model_path: str = None
        self.shuffled_tensors = {}
        self.rank = mpk.mpi_rank
        self.eos_token_id = 151645 # default eos token id for Qwen3

    def build_from_config(self, 
                              model_config: MirageModelConfig):
        self.position_embeddings = model_config.position_embeddings
        
        self.k_cache = model_config.k_cache # (num_layers, max_num_pages, page_size, num_kv_heads // world_size, head_dim)
        self.v_cache = model_config.v_cache # (num_layers, max_num_pages, page_size, num_kv_heads // world_size, head_dim)
        
        self.hidden_size = model_config.hidden_size
        self.intermediate_size = model_config.intermediate_size
        
        self.vocab_size = model_config.vocab_size
        self.padded_vocab_size = 153600 #TODO: A better way to decide?
        self.num_local_q_heads = model_config.local_num_q_heads
        self.num_local_kv_heads = model_config.local_num_kv_heads
        self.head_dim = model_config.head_dim
        self.fused_outdim_1 = (self.num_local_q_heads + 2 * self.num_local_kv_heads) * self.head_dim
        self.fused_outdim_2 = 2 * self.intermediate_size
        
        self.num_layers = model_config.num_layers
        self.build_from_dict(model_config.state_dict, model_config.with_lm_head)

    def build_from_model(self, model_name: str, model_path: str | None = None):
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
        from transformers import AutoTokenizer, AutoConfig
        with torch.device("cuda"):
            if model_path is not None:
                self.model_path = model_path
                print(f"Loading model from model path: {model_path}")
                self.config = AutoConfig.from_pretrained(model_path)
                self.model = Qwen3ForCausalLM(self.config)
                load_model(self.model, f"{model_path}/model{self.rank}-mp{self.world_size}.safetensors")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            elif model_name is not None:
                self.model_name = model_name
                self.model = Qwen3ForCausalLM.from_pretrained(self.model_name).to("cuda")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            else:
                raise ValueError("model_name or model_path is required")
            
        dummy_x = torch.empty(0, dtype=torch.bfloat16, device="cuda")
        
        self.positions = torch.arange(32768).unsqueeze(0).to(self.model.device)
        self.position_embeddings = self.model.model.rotary_emb(dummy_x, self.positions)

        self.hidden_size = self.model.config.hidden_size
        self.intermediate_size = self.model.config.intermediate_size
        
        self.vocab_size = self.model.config.vocab_size
        self.padded_vocab_size = 153600
        self.num_q_heads = self.model.config.num_attention_heads
        self.num_kv_heads = self.model.config.num_key_value_heads
        self.num_local_q_heads = self.num_q_heads // self.world_size
        self.num_local_kv_heads = self.num_kv_heads // self.world_size
        self.head_dim = self.model.config.head_dim
        self.fused_outdim_1 = (self.num_local_q_heads + 2 * self.num_local_kv_heads) * self.head_dim
        self.fused_outdim_2 = 2 * self.intermediate_size
        
        self.num_layers = len(self.model.model.layers)
        
        self.k_cache = torch.empty(
            (
                self.num_layers,
                self.max_num_pages,
                self.page_size,
                self.num_local_kv_heads,
                self.head_dim,
            ),
            dtype=torch.bfloat16,
            device="cuda",
        )
        self.v_cache = torch.empty(
            (
                self.num_layers,
                self.max_num_pages,
                self.page_size,
                self.num_local_kv_heads,
                self.head_dim,
            ),
            dtype=torch.bfloat16,
            device="cuda",
        )
        
        print(f"build_from_model: Model name: {self.model_name}, num_layers: {self.num_layers}, hidden_size: {self.hidden_size}, intermediate_size: {self.intermediate_size}, vocab_size: {self.vocab_size}, num_q_heads: {self.num_q_heads}, num_kv_heads: {self.num_kv_heads}, num_local_q_heads: {self.num_local_q_heads}, num_local_kv_heads: {self.num_local_kv_heads}, head_dim: {self.head_dim}, fused_outdim_1: {self.fused_outdim_1}, fused_outdim_2: {self.fused_outdim_2}")
        
        self.build_from_dict(self.model.state_dict(), True)
        
    def new_intermediate_tensors(self):
        if self.mpk.mode == "online_notoken":
            fixed_tensor = True
        else:
            fixed_tensor = False
        self.max_num_batched_tokens = self.mpk.max_num_batched_tokens
        if fixed_tensor:
            self.y_tensor = torch.zeros(self.max_num_batched_tokens, self.hidden_size, dtype=torch.bfloat16, device="cuda")
            self.y = self.mpk.attach_input(torch_tensor=self.y_tensor, name="embed_out")
            
            # Allocate a torch tensor to store the returned hidden state
            self.returned_hidden_state = torch.zeros(self.max_num_batched_tokens, self.hidden_size, dtype=torch.bfloat16, device="cuda")
            self.rmsnorm_out = self.mpk.attach_input(torch_tensor=self.returned_hidden_state, name="rmsnorm_out")
            
            self.attn_in_tensor = torch.zeros(self.max_num_batched_tokens, self.fused_outdim_1, dtype=torch.bfloat16, device="cuda")
            self.attn_in = self.mpk.attach_input(torch_tensor=self.attn_in_tensor, name="attn_in")
            
            self.attn_out_tensor = torch.zeros(self.max_num_batched_tokens, self.num_local_q_heads * self.head_dim, dtype=torch.bfloat16, device="cuda")
            self.attn_out = self.mpk.attach_input(torch_tensor=self.attn_out_tensor, name="attn_out")
            
            self.attn_proj_out_tensor = torch.zeros(self.max_num_batched_tokens, self.hidden_size, dtype=torch.bfloat16, device="cuda")
            self.attn_proj_out = self.mpk.attach_input(torch_tensor=self.attn_proj_out_tensor, name="attn_proj_out")
            
            self.allreduce_buf_tensor = torch.zeros(self.world_size, self.max_num_batched_tokens, self.hidden_size, dtype=torch.bfloat16, device="cuda")
            self.allreduce_buf = self.mpk.attach_input(torch_tensor=self.allreduce_buf_tensor, name="all_reduce_buf")
            
            self.attn_allreduce_out_tensor = torch.zeros(self.max_num_batched_tokens, self.hidden_size, dtype=torch.bfloat16, device="cuda")
            self.attn_allreduce_out = self.mpk.attach_input(torch_tensor=self.attn_allreduce_out_tensor, name="attn_allreduce_out")
            
            self.mlp_mid_tensor = torch.zeros(self.max_num_batched_tokens, self.fused_outdim_2 // self.world_size, dtype=torch.bfloat16, device="cuda")
            self.mlp_mid = self.mpk.attach_input(torch_tensor=self.mlp_mid_tensor, name="mlp_mid")
            
            self.silu_mul_out_tensor = torch.zeros(self.max_num_batched_tokens, self.intermediate_size // self.world_size, dtype=torch.bfloat16, device="cuda")
            self.silu_mul_out = self.mpk.attach_input(torch_tensor=self.silu_mul_out_tensor, name="silu_mul_out")
            
            self.mlp_out_tensor = torch.zeros(self.max_num_batched_tokens, self.hidden_size, dtype=torch.bfloat16, device="cuda")
            self.mlp_out = self.mpk.attach_input(torch_tensor=self.mlp_out_tensor, name="mlp_out")
            
            self.mlp_final_tensor = torch.zeros(self.max_num_batched_tokens, self.hidden_size, dtype=torch.bfloat16, device="cuda")
            self.mlp_final = self.mpk.attach_input(torch_tensor=self.mlp_final_tensor, name="mlp_final")
            
            if self.mpk.mode != "online_notoken":
                self.argmax_in_tensor = torch.zeros(self.max_num_batched_tokens, self.padded_vocab_size, dtype=torch.bfloat16, device="cuda")
                self.argmax_in = self.mpk.attach_input(torch_tensor=self.argmax_in_tensor, name="argmax_in")
                
                self.argmax_part_value_tensor = torch.zeros(self.max_num_batched_tokens, self.mpk.num_workers, dtype=torch.bfloat16, device="cuda")
                self.argmax_part_value = self.mpk.attach_input(torch_tensor=self.argmax_part_value_tensor, name="argmax_part_value")
                
                self.argmax_part_index_tensor = torch.zeros(self.max_num_batched_tokens, self.mpk.num_workers, dtype=torch.int64, device="cuda")
                self.argmax_part_index = self.mpk.attach_input(torch_tensor=self.argmax_part_index_tensor, name="argmax_part_index")

        else:
            self.y = self.mpk.new_tensor(
                dims=(self.max_num_batched_tokens, self.hidden_size),
                dtype=bfloat16,
                name="embed_out",
                io_category="cuda_tensor",
            )
            self.rmsnorm_out = self.mpk.new_tensor(
                dims=(self.max_num_batched_tokens, self.hidden_size),
                dtype=bfloat16,
                name="rmsnorm_out",
                io_category="cuda_tensor",
            )
            self.attn_in = self.mpk.new_tensor(
                dims=(self.max_num_batched_tokens, self.fused_outdim_1), # [6, 6144]
                dtype=bfloat16,
                name="attn_in",
                io_category="cuda_tensor",
            )
            self.attn_out = self.mpk.new_tensor(
                dims=(self.max_num_batched_tokens, self.num_local_q_heads * self.head_dim),
                dtype=bfloat16,
                name="attn_out",
                io_category="cuda_tensor",
            )
            self.attn_proj_out = self.mpk.new_tensor(
                dims=(self.max_num_batched_tokens, self.hidden_size),
                dtype=bfloat16,
                name="attn_proj_out",
                io_category="nvshmem_tensor" if self.world_size > 1 else "cuda_tensor",
            )
            self.allreduce_buf = self.mpk.new_tensor(
                dims=(self.world_size, self.max_num_batched_tokens, self.hidden_size),
                dtype=bfloat16,
                name="all_reduce_buf",
                io_category="nvshmem_tensor" if self.world_size > 1 else "cuda_tensor",
            )
            self.attn_allreduce_out = self.mpk.new_tensor(
                dims=(self.max_num_batched_tokens, self.hidden_size),
                dtype=bfloat16,
                name="attn_allreduce_out",
                io_category="nvshmem_tensor" if self.world_size > 1 else "cuda_tensor",
            )
            self.mlp_mid = self.mpk.new_tensor(
                dims=(self.max_num_batched_tokens, self.fused_outdim_2 // self.world_size),
                dtype=bfloat16,
                name="mlp_mid",
                io_category="cuda_tensor",
            )
            self.silu_mul_out = self.mpk.new_tensor(
                dims=(self.max_num_batched_tokens, self.intermediate_size // self.world_size),
                dtype=bfloat16,
                name="silu_mul_out",
                io_category="cuda_tensor",
            )
            self.mlp_out = self.mpk.new_tensor(
                dims=(self.max_num_batched_tokens, self.hidden_size),
                dtype=bfloat16,
                name="mlp_out",
                io_category="nvshmem_tensor" if self.world_size > 1 else "cuda_tensor",
            )
            self.mlp_final = self.mpk.new_tensor(
                dims=(self.max_num_batched_tokens, self.hidden_size),
                dtype=bfloat16,
                name="mlp_final",
                io_category="nvshmem_tensor" if self.world_size > 1 else "cuda_tensor",
            )
            if self.mpk.mode != "online_notoken":
                self.argmax_in = self.mpk.new_tensor(
                    dims=(self.max_num_batched_tokens, self.padded_vocab_size),
                    dtype=bfloat16,
                    name="argmax_in",
                    io_category="cuda_tensor",
                )
                self.argmax_part_value = self.mpk.new_tensor(
                    dims=(self.max_num_batched_tokens, self.mpk.num_workers),
                    dtype=bfloat16,
                    name="argmax_part_value",
                    io_category="cuda_tensor",
                )
                self.argmax_part_index = self.mpk.new_tensor(
                    dims=(self.max_num_batched_tokens, self.mpk.num_workers),
                    dtype=int64,
                    name="argmax_part_index",
                    io_category="cuda_tensor",
                )
        
    def build_layers(self, 
                     state_dict: dict):
        # add rmsnorm + linear
        target_cc = torch.cuda.get_device_properties(0).major * 10 + torch.cuda.get_device_properties(0).minor
        # Split-K for B200 (SM100) and MI300X (target_cc=94)
        use_splitk = (target_cc == 100 or target_cc == 94)
        # Allocate float32 workspace for MI300X split-K (workspace+reduce approach)
        if use_splitk and target_cc == 94:
            o_red = self.hidden_size // self.world_size
            k_tiles_o = o_red // 128
            self.splitk_k_sp = min(4, k_tiles_o)
            while k_tiles_o % self.splitk_k_sp != 0 and self.splitk_k_sp > 1:
                self.splitk_k_sp -= 1
            self.splitk_ws_tensor = torch.zeros(
                self.splitk_k_sp * self.mpk.max_num_batched_tokens, self.hidden_size,
                dtype=torch.float32, device="cuda")
            self.splitk_ws = self.mpk.attach_input(
                torch_tensor=self.splitk_ws_tensor, name="splitk_workspace")
            # QKV split-K: workspace for partial GEMM results
            self.splitk_ws_qkv_tensor = torch.zeros(
                self.splitk_k_sp * self.mpk.max_num_batched_tokens, self.fused_outdim_1,
                dtype=torch.float32, device="cuda")
            self.splitk_ws_qkv = self.mpk.attach_input(
                torch_tensor=self.splitk_ws_qkv_tensor, name="splitk_workspace_qkv")
            # Zero residual for QKV reduce (QKV has no residual connection)
            self.qkv_zero_res_tensor = torch.zeros(
                self.mpk.max_num_batched_tokens, self.fused_outdim_1,
                dtype=torch.bfloat16, device="cuda")
            self.qkv_zero_res = self.mpk.attach_input(
                torch_tensor=self.qkv_zero_res_tensor, name="qkv_zero_residual")
        # CK FMHA batch-independent attention workspace
        self.use_ck_fmha = (target_cc in (94, 95) and int(os.environ.get("USE_CK_FMHA", "0")) == 1)
        if self.use_ck_fmha:
            print(f"[CK FMHA] Batch-independent attention enabled (target_cc={target_cc})")
        if self.use_ck_fmha:
            self.ck_fmha_num_kv_chunks = int(os.environ.get("CK_FMHA_KV_CHUNKS", "1"))
            num_qo_per_kv = self.num_local_q_heads // self.num_local_kv_heads
            # Q workspace: processed Q with QK norm + RoPE applied
            # Layout: [num_tokens, num_q_heads * head_dim]
            q_ws_stride = self.num_local_q_heads * self.head_dim
            self.ck_fmha_q_ws_tensor = torch.zeros(
                self.mpk.max_num_batched_tokens, q_ws_stride,
                dtype=torch.bfloat16, device="cuda")
            self.ck_fmha_q_ws = self.mpk.attach_input(
                torch_tensor=self.ck_fmha_q_ws_tensor, name="ck_fmha_q_workspace")
            # O_acc workspace: float32 partial output (interleaved layout matching merge kernel)
            # Layout: [num_tokens, num_kv_heads * num_kv_chunks * num_qo_per_kv * head_dim]
            num_qo_per_kv = self.num_local_q_heads // self.num_local_kv_heads
            o_acc_dim1 = self.num_local_kv_heads * self.ck_fmha_num_kv_chunks * num_qo_per_kv * self.head_dim
            self.ck_fmha_o_acc_tensor = torch.zeros(
                self.mpk.max_num_batched_tokens, o_acc_dim1,
                dtype=torch.float32, device="cuda")
            self.ck_fmha_o_acc = self.mpk.attach_input(
                torch_tensor=self.ck_fmha_o_acc_tensor, name="ck_fmha_o_acc")
            # LSE_acc workspace: float32 partial LSE
            # Layout: [num_tokens, num_kv_heads * num_kv_chunks * num_qo_per_kv]
            lse_dim1 = self.num_local_kv_heads * self.ck_fmha_num_kv_chunks * num_qo_per_kv
            self.ck_fmha_lse_acc_tensor = torch.zeros(
                self.mpk.max_num_batched_tokens, lse_dim1,
                dtype=torch.float32, device="cuda")
            self.ck_fmha_lse_acc = self.mpk.attach_input(
                torch_tensor=self.ck_fmha_lse_acc_tensor, name="ck_fmha_lse_acc")
        for i in range(self.num_layers):
            prefix = f"model.layers.{i}."
            w_norm = self.mpk.attach_input(
                torch_tensor=state_dict[f"{prefix}input_layernorm.weight"],
                name=f"layer_{i}_input_layernorm",
            )
            if (f"{prefix}self_attn.qkv_proj.weight" in state_dict) and (f"{prefix}self_attn.q_proj.weight" in state_dict):
                    # Shuffle on CPU in place for qkv_proj.weight tensor
                    inplace_shuffle_tensors(
                        [
                            state_dict[f"{prefix}self_attn.q_proj.weight"], # views
                            state_dict[f"{prefix}self_attn.k_proj.weight"],
                            state_dict[f"{prefix}self_attn.v_proj.weight"],
                        ],
                        state_dict[f"{prefix}self_attn.qkv_proj.weight"], # target tensor
                        self.num_local_kv_heads,
                        0,
                    )
                    w_qkv = self.mpk.attach_input(
                        torch_tensor=state_dict[f"{prefix}self_attn.qkv_proj.weight"], name=f"layer_{i}_qkv_proj"
                    )
            elif f"{prefix}self_attn.q_proj.weight" in state_dict:
                if self.mpk.mode == "online_notoken":
                    self.w_qkv_tensor = shuffle_tensors(
                        [
                            state_dict[f"{prefix}self_attn.q_proj.weight"],
                            state_dict[f"{prefix}self_attn.k_proj.weight"],
                            state_dict[f"{prefix}self_attn.v_proj.weight"],
                        ],
                        self.num_local_kv_heads,
                        0,
                    )
                    assert self.w_qkv_tensor.is_contiguous(), "qkv tensor should be contiguous"
                    # We need to self maintain the shuffled tensor to avoid recycling
                    self.shuffled_tensors[f"layer_{i}_qkv_proj"] = self.w_qkv_tensor
                    w_qkv = self.mpk.attach_input(
                        torch_tensor=self.w_qkv_tensor, name=f"layer_{i}_qkv_proj"
                    )
                else:
                    w_q = self.mpk.attach_input(
                        torch_tensor=state_dict[f"{prefix}self_attn.q_proj.weight"], name=f"layer_{i}_q_proj"
                    )
                    w_k = self.mpk.attach_input(
                        torch_tensor=state_dict[f"{prefix}self_attn.k_proj.weight"], name=f"layer_{i}_k_proj"
                    )
                    w_v = self.mpk.attach_input(
                        torch_tensor=state_dict[f"{prefix}self_attn.v_proj.weight"], name=f"layer_{i}_v_proj"
                    )
                    w_qkv = self.mpk.shuffle_tensors(
                        inputs=[w_q, w_k, w_v],
                        shuffled_dim=0,
                        num_groups=self.num_local_kv_heads,
                        name=f"layer_{i}_qkv_proj",
                    )
            elif f"{prefix}self_attn.qkv_proj.weight" in state_dict:
                w_qkv = self.mpk.attach_input(
                    torch_tensor=state_dict[f"{prefix}self_attn.qkv_proj.weight"], name=f"layer_{i}_qkv_proj"
                )
            else:
                raise ValueError(f"No qkv projection weight found for layer {i}")
            self.mpk.rmsnorm_layer(
                input=self.x,
                weight=w_norm,
                output=self.rmsnorm_out,
                grid_dim=(self.mpk.max_num_batched_tokens, 1, 1),
                block_dim=(128, 1, 1),
            )
            output_size = w_qkv.dim(0)
            if use_splitk and target_cc == 94:
                # MI300X: split-K for QKV to increase CU utilization
                # grid=(N_blocks, K_splits, 1) e.g. (48, 4, 1) = 192 tasks
                k_sp = self.splitk_k_sp
                print(f"DEBUG QKV split-K: output_size={output_size}, k_sp={k_sp}, n_blocks={output_size // 128}")
                qkv_n_blocks = output_size // 128
                assert output_size % 128 == 0, f"QKV output_size={output_size} not divisible by 128"
                sk_grid = (qkv_n_blocks, k_sp, 1)
                self.mpk.splitk_linear_layer(
                    input=self.rmsnorm_out,
                    weight=w_qkv,
                    output=self.splitk_ws_qkv,
                    grid_dim=sk_grid,
                    block_dim=(256, 1, 1),
                )
                self.mpk.splitk_reduce_layer(
                    workspace=self.splitk_ws_qkv,
                    residual=self.qkv_zero_res,
                    output=self.attn_in,
                    k_splits=k_sp,
                    grid_dim=(qkv_n_blocks, 1, 1),
                    block_dim=(256, 1, 1),
                )
            else:
                grid_x = grid_for_rmsnorm_linear_layer(output_size)
                assert grid_x is not None, f"grid_for_rmsnorm_linear_layer returned None for size {output_size}"
                if output_size % grid_x != 0:
                    for d in [96, 64, 32, 16, 8, 4, 2, 1]:
                        if output_size % d == 0:
                            grid_x = d
                            break
                assert output_size % grid_x == 0, f"grid_x={grid_x} does not divide output_size={output_size}"
                self.mpk.linear_layer(
                    input=self.rmsnorm_out,
                    weight=w_qkv,
                    output=self.attn_in,
                    grid_dim=(grid_x, 1, 1),
                    block_dim=(128, 1, 1),
                )

            # add attention
            w_q_norm = self.mpk.attach_input(
                torch_tensor=state_dict[f"{prefix}self_attn.q_norm.weight"], name=f"layer_{i}_q_norm"
            )
            w_k_norm = self.mpk.attach_input(
                torch_tensor=state_dict[f"{prefix}self_attn.k_norm.weight"], name=f"layer_{i}_k_norm"
            )
            # TODO: KV cache handling
            k_cache = self.mpk.attach_input(
                torch_tensor=self.k_cache[i], name=f"layer_{i}_k_cache"
            )
            v_cache = self.mpk.attach_input(
                torch_tensor=self.v_cache[i], name=f"layer_{i}_v_cache"
            )
            
            # TODO(Jianan Ji): spec_decode_config handling (see previous implementation)
            # if spec_decode_config:
            #     self.mpk.single_batch_extend_attention_layer(
            #         input=attn_in,
            #         k_cache=k_cache,
            #         v_cache=v_cache,
            #         q_norm=w_q_norm,
            #         k_norm=w_k_norm,
            #         cos_pos_embed=cos_pos_embed,
            #         sin_pos_embed=sin_pos_embed,
            #         output=attn_out,
            #         grid_dim=(1, num_local_kv_heads, 1), #TODO: further divide across batch dim
            #         block_dim=(128, 1, 1),
            #     )
            # else:
            if self.use_ck_fmha:
                # Phase A: KV cache update + Q preprocessing (per-request)
                self.mpk.kv_cache_update_layer(
                    input=self.attn_in,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    q_norm=w_q_norm,
                    k_norm=w_k_norm,
                    cos_pos_embed=self.cos_pos_embed,
                    sin_pos_embed=self.sin_pos_embed,
                    q_workspace=self.ck_fmha_q_ws,
                    grid_dim=(self.mpk.max_num_batched_requests, self.num_local_kv_heads, 1),
                    block_dim=(256, 1, 1),
                )
                # Phase B: CK FMHA attention
                num_qo_per_kv = self.num_local_q_heads // self.num_local_kv_heads
                if self.ck_fmha_num_kv_chunks == 1:
                    # Direct bf16 output — kernel writes to attn_out, no merge needed
                    self.mpk.paged_attention_ck_fmha_layer(
                        q_workspace=self.ck_fmha_q_ws,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        o_acc=self.attn_out,
                        lse_acc=self.ck_fmha_lse_acc,
                        attention_params=(self.num_local_q_heads,
                                         self.ck_fmha_num_kv_chunks,
                                         self.mpk.max_num_batched_requests),
                        grid_dim=(self.mpk.max_num_batched_requests, self.num_local_kv_heads, self.ck_fmha_num_kv_chunks),
                        block_dim=(256, 1, 1),
                    )
                else:
                    # Float32 accumulator + merge
                    self.mpk.paged_attention_ck_fmha_layer(
                        q_workspace=self.ck_fmha_q_ws,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        o_acc=self.ck_fmha_o_acc,
                        lse_acc=self.ck_fmha_lse_acc,
                        attention_params=(self.num_local_q_heads,
                                         self.ck_fmha_num_kv_chunks,
                                         self.mpk.max_num_batched_requests),
                        grid_dim=(self.mpk.max_num_batched_requests, self.num_local_kv_heads, self.ck_fmha_num_kv_chunks),
                        block_dim=(256, 1, 1),
                    )
                    # Phase C: Merge split-KV partial results
                    self.mpk.paged_attention_ck_fmha_merge_layer(
                        lse=self.ck_fmha_lse_acc,
                        output_tmp=self.ck_fmha_o_acc,
                        output=self.attn_out,
                        attention_params=(self.num_local_q_heads, self.head_dim,
                                         self.ck_fmha_num_kv_chunks,
                                         self.num_local_kv_heads),
                        grid_dim=(self.mpk.max_num_batched_requests, self.num_local_kv_heads, 1),
                        block_dim=(256, 1, 1),
                    )
            else:
                self.mpk.paged_attention_layer(
                    input=self.attn_in,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    q_norm=w_q_norm,
                    k_norm=w_k_norm,
                    cos_pos_embed=self.cos_pos_embed,
                    sin_pos_embed=self.sin_pos_embed,
                    output=self.attn_out,
                    grid_dim=(self.mpk.max_num_batched_requests, self.num_local_kv_heads, 1),
                    block_dim=(128, 1, 1),
                )
            # add linear w/ residual
            self.w = self.mpk.attach_input(
                torch_tensor=state_dict[f"{prefix}self_attn.o_proj.weight"], name=f"layer_{i}_o_proj"
            )
            if use_splitk:
                if target_cc == 94:
                    # MI300X: workspace+reduce split-K
                    k_sp = self.splitk_k_sp
                    sk_grid = (self.hidden_size // 128, k_sp, 1)
                    self.mpk.splitk_linear_layer(
                        input=self.attn_out,
                        weight=self.w,
                        output=self.splitk_ws,
                        grid_dim=sk_grid,
                        block_dim=(256, 1, 1),
                    )
                    self.mpk.splitk_reduce_layer(
                        workspace=self.splitk_ws,
                        residual=self.x,
                        output=self.attn_proj_out,
                        k_splits=k_sp,
                        grid_dim=(self.hidden_size // 128, 1, 1),
                        block_dim=(256, 1, 1),
                    )
                else:
                    self.attn_proj_out = self.x
                    sk_grid = (self.hidden_size // 128, 128 * 128 // self.hidden_size, 1)
                    self.mpk.splitk_linear_layer(
                        input=self.attn_out,
                        weight=self.w,
                        output=self.attn_proj_out,
                        grid_dim=sk_grid,
                        block_dim=(256, 1, 1),
                    )
            else:
                self.mpk.linear_with_residual_layer(
                    input=self.attn_out,
                    weight=self.w,
                    residual=self.x,
                    output=self.attn_proj_out,
                    grid_dim=(self.hidden_size // 64, 1, 1),
                    block_dim=(128, 1, 1),
                )
            # reset residual input as x
            self.x = self.attn_proj_out
            # add allreduce if needed
            if self.world_size > 1:
                self.mpk.allreduce_layer(
                    input=self.attn_proj_out,
                    buffer=self.allreduce_buf,
                    output=self.attn_allreduce_out,
                    grid_dim=(self.hidden_size // 64, 1, 1),
                    block_dim=(128, 1, 1),
                )
                self.x = self.attn_allreduce_out
            # add rmsnorm_linear layer
            w_norm = self.mpk.attach_input(
                torch_tensor=state_dict[f"{prefix}post_attention_layernorm.weight"],
                name=f"layer_{i}_post_attn_layernorm",
            )
            # 
            if (f"{prefix}mlp.gate_proj.weight" in state_dict) and (f"{prefix}mlp.gate_up_proj.weight" in state_dict):
                rmsnorm_num_tasks = grid_for_rmsnorm_linear_layer(state_dict[f"{prefix}mlp.gate_up_proj.weight"].shape[0])
                inplace_shuffle_tensors(
                    [
                        state_dict[f"{prefix}mlp.gate_proj.weight"], # views
                        state_dict[f"{prefix}mlp.up_proj.weight"],
                    ],
                    state_dict[f"{prefix}mlp.gate_up_proj.weight"], # target tensor
                    rmsnorm_num_tasks//2,
                    0,
                )
                w_gatedup = self.mpk.attach_input(
                    torch_tensor=state_dict[f"{prefix}mlp.gate_up_proj.weight"], name=f"layer_{i}_gatedup_proj"
                )
            elif f"{prefix}mlp.gate_proj.weight" in state_dict:
                rmsnorm_num_tasks = grid_for_rmsnorm_linear_layer(
                    state_dict[f"{prefix}mlp.gate_proj.weight"].shape[0] 
                    + state_dict[f"{prefix}mlp.up_proj.weight"].shape[0]
                )
                if self.mpk.mode == "online_notoken":
                    self.w_gatedup_tensor = shuffle_tensors(
                        [
                            state_dict[f"{prefix}mlp.gate_proj.weight"],
                            state_dict[f"{prefix}mlp.up_proj.weight"],
                        ],
                        rmsnorm_num_tasks//2,
                        0,
                    )
                    assert self.w_gatedup_tensor.is_contiguous(), "gatedup tensor should be contiguous"
                    # We need to self maintain the shuffled tensor to avoid recycling
                    self.shuffled_tensors[f"layer_{i}_gatedup_proj"] = self.w_gatedup_tensor
                    w_gatedup = self.mpk.attach_input(
                        torch_tensor=self.w_gatedup_tensor, name=f"layer_{i}_gatedup_proj"
                    )
                else:
                    w_gate_proj = self.mpk.attach_input(
                        torch_tensor=state_dict[f"{prefix}mlp.gate_proj.weight"], name=f"layer_{i}_gate_proj"
                    )
                    w_up_proj = self.mpk.attach_input(
                        torch_tensor=state_dict[f"{prefix}mlp.up_proj.weight"], name=f"layer_{i}_up_proj"
                    )                    
                    w_gatedup = self.mpk.shuffle_tensors(
                        inputs=[w_gate_proj, w_up_proj],
                        shuffled_dim=0,
                        num_groups=rmsnorm_num_tasks//2,
                        name=f"layer_{i}_gatedup_proj",
                    )
            elif f"{prefix}mlp.gate_up_proj.weight" in state_dict:
                rmsnorm_num_tasks = grid_for_rmsnorm_linear_layer(state_dict[f"{prefix}mlp.gate_up_proj.weight"].shape[0])
                w_gatedup = self.mpk.attach_input(
                    torch_tensor=state_dict[f"{prefix}mlp.gate_up_proj.weight"], name=f"layer_{i}_gatedup_proj"
                )
            else:
                raise ValueError(f"No gate or up projection weight found for layer {i}")
            self.mpk.rmsnorm_layer(
                input=self.x,
                weight=w_norm,
                output=self.rmsnorm_out,
                grid_dim=(self.mpk.max_num_batched_tokens, 1, 1),
                block_dim=(128, 1, 1),
            )
            self.mpk.linear_layer(
                input=self.rmsnorm_out,
                weight=w_gatedup,
                output=self.mlp_mid,
                grid_dim=(rmsnorm_num_tasks, 1, 1),
                block_dim=(128, 1, 1),
            )

            self.w = self.mpk.attach_input(
                torch_tensor=state_dict[f"{prefix}mlp.down_proj.weight"], name=f"layer_{i}_down_proj"
            )
            if target_cc == 94:
                # MI300: fused silu_mul_linear_with_residual
                # Skips separate silu_mul + splitk_linear + splitk_reduce
                self.mpk.silu_mul_linear_with_residual_layer(
                    input=self.mlp_mid,
                    weight=self.w,
                    residual=self.x,
                    output=self.mlp_out,
                    grid_dim=(self.hidden_size // 64, 1, 1),
                    block_dim=(256, 1, 1),
                )
            else:
                self.mpk.silu_mul_layer(
                    input=self.mlp_mid,
                    output=self.silu_mul_out,
                    grid_dim=(rmsnorm_num_tasks//2, 1, 1),
                    block_dim=(128, 1, 1),
                )
                if use_splitk:
                    self.mlp_out = self.x
                    sk_grid = (self.hidden_size // 128, 128 * 128 // self.hidden_size, 1)
                    self.mpk.splitk_linear_layer(
                        input=self.silu_mul_out,
                        weight=self.w,
                        output=self.mlp_out,
                        grid_dim=sk_grid,
                        block_dim=(256, 1, 1),
                    )
                else:
                    self.mpk.linear_with_residual_layer(
                        input=self.silu_mul_out,
                        weight=self.w,
                        residual=self.x,
                        output=self.mlp_out,
                        grid_dim=(self.hidden_size // 64, 1, 1),
                        block_dim=(128, 1, 1),
                    )
            # reset residual input as x
            self.x = self.mlp_out
            if self.world_size > 1:
                self.mpk.allreduce_layer(
                    input=self.mlp_out,
                    buffer=self.allreduce_buf,
                    output=self.mlp_final,
                    grid_dim=(self.hidden_size // 64, 1, 1),
                    block_dim=(128, 1, 1),
                )
                self.x = self.mlp_final
        
    def build_from_dict(self, 
                        state_dict: dict,
                        with_lm_head: bool,
                        ):
        # pad vocab_size to facilitate task graph creation
        if with_lm_head:
            self.lm_head_weight = torch.cat(
                (
                    state_dict["lm_head.weight"],
                    torch.full(
                        (self.padded_vocab_size - self.vocab_size, self.hidden_size), 0, device="cuda"
                    ),
                ),
                0,
            )
            assert self.lm_head_weight.stride()[0] == self.hidden_size
        # TODO(Jianan Ji): spec_decode_config handling (see previous implementation)
        # if spec_decode_config and spec_decode_config.method == "promptlookup":
        #     all_tokens = mpk.attach_input(torch_tensor=tokens, name="all_tokens")
        #     num_tokens_extend = spec_decode_config.spec_length + 1
        # else:
        #     num_tokens_extend = 1
            
        self.x = self.mpk.attach_input(torch_tensor=self.input_tokens, name="input_token")
        self.cos_pos_embed = self.mpk.attach_input(
            torch_tensor=self.position_embeddings[0][0, :4096, :],
            name="cos_position_embedding",
        )
        self.sin_pos_embed = self.mpk.attach_input(
            torch_tensor=self.position_embeddings[1][0, :4096, :],
            name="sin_position_embedding",
        )
        
        self.new_intermediate_tensors()
        
        argmax_out = self.mpk.attach_input(torch_tensor=self.output_tokens, name="output_token")
        # TODO(Jianan Ji): spec_decode_config handling (see previous implementation)
        # add spec tokens layer
        # if spec_decode_config:
        #     spec_tokens = self.mpk.draft_forward_layer_dispatcher(
        #         spec_decode_config = spec_decode_config, 
        #         tokens = all_tokens,
        #         grid_dim=(96, 1, 1),
        #         block_dim=(128, 1, 1),
        #     )
        #     x = spec_tokens
        # Add Embed
        
        self.w = self.mpk.attach_input(
            torch_tensor=state_dict["model.embed_tokens.weight"], name="embed_tokens"
        )
        
        self.mpk.embed_layer(
            input=self.x, 
            weight=self.w, 
            output=self.y, 
            # grid_dim=(max_factor_leq_n(hidden_size, 96 // args.max_num_batched_tokens), total_tokens_per_iter, 1), 
            grid_dim=(1, 1, 1), 
            block_dim=(128, 1, 1),
            input_source=1,
        )
        self.x = self.y
        
        self.build_layers(state_dict)

        # add rmsnorm_linear layer
        self.w_norm = self.mpk.attach_input(
            torch_tensor=state_dict["model.norm.weight"], name="model_norm_weight"
        )
        self.mpk.rmsnorm_layer(
            input=self.x,
            weight=self.w_norm,
            output=self.rmsnorm_out,
            grid_dim=(self.mpk.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )
        if with_lm_head:
            self.w_proj = self.mpk.attach_input(torch_tensor=self.lm_head_weight, name="lm_head")
            self.mpk.linear_layer(
                input=self.rmsnorm_out,
                weight=self.w_proj,
                output=self.argmax_in,
                grid_dim=(grid_for_rmsnorm_linear_layer(self.w_proj.dim(0)), 1, 1),
                block_dim=(128, 1, 1),
            )

            # add argmax layer
            # TODO(Jianan Ji): spec_decode_config handling (see previous implementation)
            # if spec_decode_config and spec_decode_config.method == "promptlookup":
            #     argmax_partial_grid_dim = (max_factor_leq_n(153600, 96 // (spec_decode_config.spec_length + 1)), 
            #                                spec_decode_config.spec_length + 1, 
            #                                1)
            #     argmax_reduce_grid_dim = (1, spec_decode_config.spec_length + 1, 1)
            # else:
            argmax_partial_grid_dim = (self.mpk.num_workers, 1, 1)
            argmax_reduce_grid_dim = (1, 1, 1)
            self.mpk.argmax_partial_layer(
                input=self.argmax_in,
                output=(self.argmax_part_value, self.argmax_part_index),
                grid_dim=argmax_partial_grid_dim,
                block_dim=(128, 1, 1),
            )
            self.mpk.argmax_reduce_layer(
                input=(self.argmax_part_value, self.argmax_part_index),
                output=argmax_out,
                grid_dim=argmax_reduce_grid_dim,
                block_dim=(128, 1, 1),
            )
            # TODO(Jianan Ji): spec_decode_config handling (see previous implementation)
            # if spec_decode_config:
            #     verify_out = self.mpk.verify_layer_dispatcher(
            #         spec_decode_config = spec_decode_config,
            #         spec_tokens = spec_tokens,
            #         target_output = argmax_out,
            #         grid_dim = (1, 1, 1),
            #         block_dim = (128, 1, 1),
            #     )
            
    def encode(self, text: str):
        return self.tokenizer.encode(text, add_special_tokens=True)
        
    def decode(self, ids: torch.Tensor):
        return self.tokenizer.decode(ids, skip_special_tokens=True)