"""fleet_megakernel_vllm: spec-driven weight-packing + megakernel launch glue for fleet_mk's
fused decode kernels (dense Qwen3, MoE GPT-OSS), used by the vLLM plugin.
"""

from .spec import ModelSpec, load_spec, supported_arches
from .packing import import_mirage_packers, pack_layer_weights, pack_lm_head
from .runtime import (
    FleetMKBuffers, FleetMKDecoder, build_ptr_table, build_ptr_table_dense,
    load_kernel, load_qwen3_kernel,
)

__all__ = [
    "ModelSpec",
    "load_spec",
    "supported_arches",
    "import_mirage_packers",
    "pack_layer_weights",
    "pack_lm_head",
    "FleetMKBuffers",
    "FleetMKDecoder",
    "build_ptr_table",
    "build_ptr_table_dense",
    "load_kernel",
    "load_qwen3_kernel",
]
