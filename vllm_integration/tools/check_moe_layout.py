#!/usr/bin/env python3
"""Check MoeLayout against the standalone driver's arithmetic, knob by knob.

`titan_vllm/moe_layout.py` re-derives, for the vLLM plugin, geometry that
`demo_gpt_oss_120b.py` already derives for the standalone driver: K row pitch, N
expert pitch, section split, workgroup and expert byte strides. Two derivations
of one layout is exactly the drift pattern that has bitten this tree before --
and this instance is worse than most, because the kernel is compiled against
`-DMPK_MOE_*` flags and nothing at runtime can notice that Python disagrees with
it. A wrong pitch does not fault; it reads real weight bytes belonging to the
wrong rows and emits fluent garbage.

So rather than trust that the two agree, compute both and compare. The driver's
constants are read by executing its module-level prologue under the same
environment the layout sees -- not by importing it, which would try to load a
model. That keeps the driver as the reference definition without duplicating its
expressions here.

Checked, per knob combination (default / K-stride / +split / +N-stride / alias):

  1. `pack_kwargs` matches the driver's pack call arguments exactly.
  2. Expert pitch in BYTES matches the driver's W13_EXPERT_BYTES / W2_EXPERT_BYTES
     -- this is the number the pointer table walks and the kernel's buffer-rsrc
     extent bounds against, so it is the one that turns a mistake into a
     silent out-of-range read rather than a zero.
  3. W13's expert pitch is twice W2's whenever a foreign N stride is set (W13 is
     gate+up interleaved, so its axis holds two intermediate rows).
  4. The layout rejects the combinations the kernel cannot honour.

Run:
  python3 tools/check_moe_layout.py
"""

import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

DRIVER = os.path.join(_ROOT, "demo_gpt_oss_120b.py")

#: The driver's prologue runs from the top down to its first import of torch-heavy
#: machinery; everything this checker needs is defined before that. Cutting at the
#: model-loading section keeps the exec cheap and side-effect free.
_CUT = "# Pointer table layout"


def _driver_constants(env):
    """The driver's MoE constants, under `env`, by executing its prologue.

    Executed rather than imported: importing runs argparse and model loading.
    Executed rather than re-typed: re-typing is the drift this file exists to
    catch.
    """
    src = open(DRIVER).read()
    cut = src.index(_CUT)
    # __file__ must be present: the prologue puts its own directory on sys.path.
    ns = {"__name__": "_driver_prologue", "__file__": DRIVER, "os": os}
    old = dict(os.environ)
    try:
        for k in ("TITAN_MOE_K_STRIDE", "TITAN_MOE_SPLIT_SCALES",
                  "TITAN_MOE_N_STRIDE", "TITAN_MOE_SPLIT_BUFFERS",
                  "TITAN_MOE_ALIAS_VLLM"):
            os.environ.pop(k, None)
        os.environ.update(env)
        exec(compile(src[:cut], DRIVER, "exec"), ns)
    finally:
        os.environ.clear()
        os.environ.update(old)
    return ns


class _FakeSpec:
    """Just the fields MoeLayout reads. Values are GPT-OSS 120B's."""
    padded_hidden_size = 2944
    gemm = {"w13_output_per_wg": 128, "w2_output_per_wg": 64}


def _layout(env):
    """A MoeLayout built under `env`, importing fresh so module state cannot leak."""
    from titan_vllm.moe_layout import MoeLayout
    old = dict(os.environ)
    try:
        for k in ("TITAN_MOE_K_STRIDE", "TITAN_MOE_SPLIT_SCALES",
                  "TITAN_MOE_N_STRIDE", "TITAN_MOE_ALIAS_VLLM"):
            os.environ.pop(k, None)
        os.environ.update(env)
        return MoeLayout(_FakeSpec())
    finally:
        os.environ.clear()
        os.environ.update(old)


def _say(label, cond):
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}")
    return cond


def _expert_bytes(lay, which):
    """Bytes from one expert's data to the next, the layout's way.

    Deliberately recomputed here from the layout's public attributes rather than
    exposed as a method: this checker's job is to confirm the driver's expression
    and the layout's inputs produce the same number, and a shared helper would
    make that trivially true.
    """
    opw = lay.w13_opw if which == "w13" else lay.w2_opw
    stride = lay.w13_out_stride if which == "w13" else lay.w2_out_stride
    wg_data = opw * (lay.k_stride // 2)
    wg_scale = opw * (lay.hidden // 32)
    wg_bytes = wg_data if lay.split_scales else wg_data + wg_scale
    return (stride // opw) * wg_bytes


def check(tag, env):
    print(f"--- {tag}: {env or 'defaults'}")
    d = _driver_constants(env)
    lay = _layout(env)
    ok = True

    ok &= _say(f"{tag}: k_stride_blocks matches driver",
               lay.k_stride_blocks == d["MOE_K_STRIDE_BLOCKS"])
    ok &= _say(f"{tag}: split_scales matches driver",
               lay.split_scales == d["MOE_SPLIT_SCALES"])
    ok &= _say(f"{tag}: W13 expert row pitch matches driver",
               lay.w13_out_stride == d["W13_N_STRIDE"])
    ok &= _say(f"{tag}: W2 expert row pitch matches driver",
               lay.w2_out_stride == d["W2_N_STRIDE"])

    # The number the pointer table walks and the buffer-rsrc extent bounds
    # against. Everything above is an input to it; this is the output that
    # matters.
    ok &= _say(f"{tag}: W13 expert BYTES match driver "
               f"({_expert_bytes(lay, 'w13')})",
               _expert_bytes(lay, "w13") == d["W13_EXPERT_BYTES"])
    ok &= _say(f"{tag}: W2 expert BYTES match driver "
               f"({_expert_bytes(lay, 'w2')})",
               _expert_bytes(lay, "w2") == d["W2_EXPERT_BYTES"])

    # The packer must be handed exactly what the driver hands it.
    for which, opw_key, out_key, stride_key in (
            ("w13", "W13_OPW", "W13_OUTPUT_SIZE", "W13_N_STRIDE"),
            ("w2", "W2_OPW", "HIDDEN_SIZE", "W2_N_STRIDE")):
        kw = lay.pack_kwargs(which)
        want = dict(
            output_per_wg=d[opw_key],
            target_out_dim=d[out_key],
            target_num_blocks=d["HIDDEN_SIZE"] // 32,
            row_stride_blocks=d["MOE_K_STRIDE_BLOCKS"],
            out_stride_rows=d[stride_key] if d["MOE_N_STRIDE"] else None,
            split_scales=d["MOE_SPLIT_SCALES"])
        ok &= _say(f"{tag}: {which} pack_kwargs match the driver's pack call",
                   kw == want)

    # W13 is gate+up interleaved: a foreign N stride counts intermediate rows,
    # and W13's axis holds two of them. The kernel's W13_N_STRIDE = 2 *
    # MPK_MOE_N_STRIDE is the same claim on the other side of the ABI.
    if lay.n_stride:
        ok &= _say(f"{tag}: W13 pitch is exactly twice W2's",
                   lay.w13_out_stride == 2 * lay.w2_out_stride)
    return ok


#: vLLM's own per-expert storage strides for GPT-OSS 120B, in bytes, as measured
#: on the live model by `tools/probe_vllm_expert_layout.py` AFTER
#: `process_weights_after_loading`:
#:
#:   w13_weight  shape (128, 1536, 6144)  stride (9437184, 1, 1536)
#:   w2_weight   shape (128, 1536, 3072)  stride (4718592, 1, 1536)
#:
#: The shapes read transposed and the strides are why: the middle axis has
#: stride 1 and the last has stride 1536, so these are transposed *views* over
#: `[E, rows, 1536]` row-major storage -- 6144 rows for w13, 3072 for w2. A
#: linear reader sees titan's own order. Recorded as literals rather than
#: re-derived so this stays a comparison against a measurement.
VLLM_EXPERT_BYTES = {"w13": 9437184, "w2": 4718592}


def check_against_vllm():
    """The aliased layout's expert pitch must equal vLLM's actual allocation stride.

    Every other check in this file compares titan against titan: the plugin's
    layout against the driver's arithmetic. Both could agree and both be wrong
    about the buffer they are aliasing, and the symptom would be reading real
    weight bytes at a drifting expert offset -- fluent output from the wrong
    expert, worse for later experts than earlier ones.

    This is the one check with an external referent. If it passes, titan's
    addressing walks vLLM's buffer expert-for-expert.
    """
    print("--- vs vLLM's measured storage")
    lay = _layout({"TITAN_MOE_K_STRIDE": "3072", "TITAN_MOE_SPLIT_SCALES": "1",
                   "TITAN_MOE_N_STRIDE": "3072", "TITAN_MOE_ALIAS_VLLM": "1"})
    ok = True
    for which, want in VLLM_EXPERT_BYTES.items():
        got = _expert_bytes(lay, which)
        ok &= _say(f"{which} expert pitch == vLLM's stride(0) ({want})",
                   got == want)

    # The pitch could match by luck with a compensating error in the two factors,
    # so check them separately: rows from the N knob, row bytes from the K knob.
    ok &= _say("w13 row bytes == vLLM's stride(1)-to-stride(2) pitch (1536)",
               lay.k_stride // 2 == 1536)
    ok &= _say("w13 rows == vLLM's stored row count (6144)",
               lay.w13_out_stride == 6144)
    ok &= _say("w2 rows == vLLM's stored row count (3072)",
               lay.w2_out_stride == 3072)
    return ok


def check_rejects():
    """The layout must refuse combinations the kernel cannot honour.

    Each of these is silent if it gets through: a foreign pitch without the
    split reads titan's interleaved scales at vLLM's row spacing, and aliasing
    without the pitches points the kernel at vLLM's memory while addressing it
    as titan's.
    """
    print("--- rejects")
    cases = [
        ("N stride without split", dict(TITAN_MOE_N_STRIDE="3072")),
        ("alias without any knobs", dict(TITAN_MOE_ALIAS_VLLM="1")),
        ("alias without N stride", dict(TITAN_MOE_ALIAS_VLLM="1",
                                        TITAN_MOE_SPLIT_SCALES="1",
                                        TITAN_MOE_K_STRIDE="3072")),
        ("alias without K stride", dict(TITAN_MOE_ALIAS_VLLM="1",
                                        TITAN_MOE_SPLIT_SCALES="1",
                                        TITAN_MOE_N_STRIDE="3072")),
        ("N stride below the computed rows",
         dict(TITAN_MOE_SPLIT_SCALES="1", TITAN_MOE_N_STRIDE="2048")),
        ("K stride not a multiple of 32", dict(TITAN_MOE_K_STRIDE="3000")),
    ]
    ok = True
    for label, env in cases:
        try:
            _layout(env)
            ok &= _say(f"rejects {label}", False)
        except AssertionError:
            ok &= _say(f"rejects {label}", True)
    return ok


def main():
    K, S, N = "TITAN_MOE_K_STRIDE", "TITAN_MOE_SPLIT_SCALES", "TITAN_MOE_N_STRIDE"
    ok = True
    ok &= check("default", {})
    ok &= check("kstride", {K: "3072"})
    ok &= check("split", {S: "1"})
    ok &= check("kstride+split", {K: "3072", S: "1"})
    ok &= check("all", {K: "3072", S: "1", N: "3072"})
    ok &= check("alias", {K: "3072", S: "1", N: "3072",
                          "TITAN_MOE_ALIAS_VLLM": "1"})
    ok &= check_against_vllm()
    ok &= check_rejects()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
