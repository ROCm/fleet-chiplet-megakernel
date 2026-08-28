"""Probe the installed vLLM for every API seam titan_vllm depends on.

The plugin touches vLLM in about a dozen places -- a model registry, a plugin
entry point, three model methods it overrides, the forward-context attention
metadata, and the KV cache layout. Each of those is an independent way a version
bump can break, and the failure mode for most of them is *not* an import error:
a renamed `compute_logits` parameter or a moved `block_table` silently falls
through to the stock path and you measure vLLM, not titan.

So enumerate them explicitly and print a table, rather than discovering them one
traceback at a time. Run this against any candidate vLLM before porting:

    python -m titan_vllm.probe_vllm_api

Prints one line per seam: OK / MISSING / CHANGED, with the observed signature.
Exit code is the number of non-OK seams, so it works as a CI gate.
"""

import importlib
import inspect
import sys


def _sig(fn):
    try:
        return str(inspect.signature(fn))
    except (TypeError, ValueError):
        return "<no signature>"


class Probe:
    def __init__(self):
        self.rows = []

    def record(self, name, status, detail=""):
        self.rows.append((name, status, detail))

    def attr(self, name, modpath, *attrs, expect_params=None):
        """Check module.attr[.attr...] exists; optionally check its parameters."""
        try:
            obj = importlib.import_module(modpath)
        except Exception as e:
            self.record(name, "MISSING", f"import {modpath}: {e}")
            return None
        where = modpath
        for a in attrs:
            if not hasattr(obj, a):
                self.record(name, "MISSING", f"{where} has no {a!r}")
                return None
            obj = getattr(obj, a)
            where = f"{where}.{a}"
        detail = _sig(obj) if callable(obj) else repr(obj)[:80]
        if expect_params is not None and callable(obj):
            try:
                params = set(inspect.signature(obj).parameters)
            except (TypeError, ValueError):
                params = set()
            missing = set(expect_params) - params
            if missing:
                self.record(name, "CHANGED",
                            f"missing params {sorted(missing)}; got {detail}")
                return obj
        self.record(name, "OK", detail)
        return obj

    def any_of(self, name, candidates):
        """One of several spellings must exist -- report which, count once.

        Renames (get_input_embeddings -> embed_input_ids, vllm.attention ->
        vllm.model_executor.layers.attention) are the common shape of vLLM
        drift. Probing each spelling separately would score a clean rename as
        two failures and one pass; what matters is whether *some* spelling is
        available and which one to bind to.
        """
        found = []
        for modpath, *attrs in candidates:
            try:
                obj = importlib.import_module(modpath)
                ok = True
                for a in attrs:
                    if not hasattr(obj, a):
                        ok = False
                        break
                    obj = getattr(obj, a)
            except Exception:
                ok = False
            if ok:
                found.append(".".join([modpath, *attrs]))
        if found:
            self.record(name, "OK", " | ".join(found))
        else:
            spellings = [".".join([m, *a]) for m, *a in candidates]
            self.record(name, "MISSING", f"none of {spellings}")
        return found

    def metadata_fields(self, name, modpath, clsname, expect):
        """Check a dataclass still carries the fields the mixin reads."""
        import dataclasses
        try:
            cls = getattr(importlib.import_module(modpath), clsname)
        except Exception as e:
            self.record(name, "MISSING", f"{modpath}.{clsname}: {e}")
            return
        if not dataclasses.is_dataclass(cls):
            self.record(name, "CHANGED", "not a dataclass")
            return
        have = {f.name for f in dataclasses.fields(cls)}
        missing = [f for f in expect if f not in have]
        if missing:
            self.record(name, "CHANGED", f"missing fields {missing}")
        else:
            self.record(name, "OK", f"has {expect}")

    def report(self):
        w = max(len(r[0]) for r in self.rows)
        bad = 0
        for name, status, detail in self.rows:
            if status != "OK":
                bad += 1
            print(f"{name:<{w}}  {status:<8} {detail}")
        print(f"\n{len(self.rows) - bad}/{len(self.rows)} seams OK")
        return bad


def main():
    import vllm

    print(f"vLLM {vllm.__version__}\n")
    p = Probe()

    # -- plugin registration --------------------------------------------------
    p.attr("ModelRegistry.register_model", "vllm", "ModelRegistry",
           "register_model")

    # -- model classes we subclass --------------------------------------------
    p.attr("Qwen3ForCausalLM", "vllm.model_executor.models.qwen3",
           "Qwen3ForCausalLM")
    p.attr("GptOssForCausalLM", "vllm.model_executor.models.gpt_oss",
           "GptOssForCausalLM")

    # -- the three methods the mixin overrides --------------------------------
    # Signature drift here is the dangerous kind: super() calls break loudly,
    # but a *renamed* parameter that vLLM passes by keyword breaks at runtime
    # only on the path that uses it.
    for cls_name, modpath in (("Qwen3ForCausalLM",
                               "vllm.model_executor.models.qwen3"),
                              ("GptOssForCausalLM",
                               "vllm.model_executor.models.gpt_oss")):
        p.attr(f"{cls_name}.forward", modpath, cls_name, "forward",
               expect_params=["input_ids", "positions"])
        p.attr(f"{cls_name}.load_weights", modpath, cls_name, "load_weights")
        p.attr(f"{cls_name}.compute_logits", modpath, cls_name,
               "compute_logits")
        # Token embedding lookup. Renamed get_input_embeddings ->
        # embed_input_ids in 0.27.x; report which spelling this vLLM has.
        p.any_of(f"{cls_name} embed lookup",
                 [(modpath, cls_name, "get_input_embeddings"),
                  (modpath, cls_name, "embed_input_ids")])

    # -- forward context / attention metadata ---------------------------------
    p.attr("get_forward_context", "vllm.forward_context",
           "get_forward_context")

    # -- attention backend registry (the CUSTOM-backend seam) -----------------
    p.attr("register_backend", "vllm.v1.attention.backends.registry",
           "register_backend")
    p.attr("AttentionBackendEnum", "vllm.v1.attention.backends.registry",
           "AttentionBackendEnum")
    p.attr("AttentionBackendEnum.CUSTOM", "vllm.v1.attention.backends.registry",
           "AttentionBackendEnum", "CUSTOM")

    # -- the aiter backend we subclass for the KV shape override --------------
    p.attr("RocmAiterUnifiedAttentionBackend",
           "vllm.v1.attention.backends.rocm_aiter_unified_attn",
           "RocmAiterUnifiedAttentionBackend")
    p.attr("RocmAiterUnifiedAttentionBackend.get_kv_cache_shape",
           "vllm.v1.attention.backends.rocm_aiter_unified_attn",
           "RocmAiterUnifiedAttentionBackend", "get_kv_cache_shape")

    # -- Attention module: where titan reads the KV tensor from ---------------
    # Moved vllm.attention -> vllm.model_executor.layers.attention in 0.27.x.
    p.any_of("Attention",
             [("vllm.attention", "Attention"),
              ("vllm.model_executor.layers.attention", "Attention")])

    # -- attention metadata fields the mixin reads ----------------------------
    # forward() gates on max_query_len and indexes block_table. If either is
    # renamed the gate silently sends every decode down the stock path -- you
    # get correct output and titan never runs, which benchmarks as a
    # "regression" with no error anywhere.
    p.metadata_fields("RocmAttentionMetadata",
                      "vllm.v1.attention.backends.rocm_attn",
                      "RocmAttentionMetadata",
                      ["max_query_len", "block_table", "slot_mapping",
                       "seq_lens"])

    # -- Attention.kv_cache shape contract ------------------------------------
    # 0.11.1 bound a per-virtual-engine LIST (kv_cache[0] == the layer's cache);
    # 0.27.x binds the tensor directly, so kv_cache[0] is the first *block*.
    # Both index cleanly and neither raises -- the failure is silent wrong KV.
    p.any_of("Attention.bind_kv_cache",
             [("vllm.attention", "Attention", "bind_kv_cache"),
              ("vllm.model_executor.layers.attention", "Attention",
               "bind_kv_cache")])

    bad = p.report()
    sys.exit(min(bad, 100))


if __name__ == "__main__":
    main()
