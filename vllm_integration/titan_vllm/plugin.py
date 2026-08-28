"""vLLM general plugin entry point for titan.

vLLM calls register() once at engine startup (discovered via the
`vllm.general_plugins` entry point in pyproject.toml). We override each stock
causal-LM titan supports with a titan subclass that keeps vLLM's prefill path but
routes decode through titan's fused megakernel. vLLM instantiates whichever
architecture matches the loaded model, so registering both is harmless when only
one model is served.
"""


def register():
    from vllm import ModelRegistry

    from .model import TitanQwen3ForCausalLM, _make_gptoss_class

    # vLLM 0.26+ moved the stock ROCm aiter KV pool to a packed layout that
    # titan's flat view cannot alias. Registering TitanAttentionBackend restores
    # the split layout; it must happen before any Attention layer is built, and
    # the plugin's register() is the earliest hook vLLM offers. Registration
    # alone does not *select* it -- the engine must also be launched with
    # attention_backend=CUSTOM (see harness.py). Absent on 0.11.x, where the
    # stock aiter backends already allocate titan's layout.
    try:
        from .backend import register_titan_backend
        register_titan_backend()
    except ImportError as e:  # pragma: no cover - depends on installed vLLM
        import logging
        logging.getLogger(__name__).info(
            "titan: CUSTOM attention backend not registered (%s); this vLLM "
            "predates the registry and should already use titan's KV layout.",
            e)

    ModelRegistry.register_model("Qwen3ForCausalLM", TitanQwen3ForCausalLM)

    # GPT-OSS lives in a separate vLLM module that may be absent in older builds;
    # only register it if importable so the dense path never breaks.
    try:
        ModelRegistry.register_model("GptOssForCausalLM", _make_gptoss_class())
    except Exception as e:  # pragma: no cover - depends on installed vLLM
        import logging
        logging.getLogger(__name__).info(
            "titan: GptOssForCausalLM not registered (%s); dense path unaffected.",
            e)
