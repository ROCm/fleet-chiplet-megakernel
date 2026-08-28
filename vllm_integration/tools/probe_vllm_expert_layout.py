"""What do vLLM's MoE expert tensors actually look like, and where do they live?

The single-copy question turns on one fact that cannot be settled by reading
code: after `process_weights_after_loading` runs, are the expert *weight bytes*
vLLM keeps for the process lifetime still the plain `[E, 2I, K/2]` checkpoint
order at the same address -- or has vLLM re-packed them into a backend format?

Reading suggests the former on ROCm: `_swizzle_mxfp4`
(quantization/utils/mxfp4_utils.py:60) picks `value_layout = StridedLayout` for
`current_platform.is_rocm()`, which is identity, and the only shape change is a
`transpose(-2, -1)` view. But `convert_layout` may still materialise a copy, the
selected backend may not be TRITON at all, and the *scales* take a different
path (`should_use_cdna4_mx_scale_swizzle()` is true on gfx950 at TP<=2, so they
ARE swizzled). Guessing any of that wrong sends the kernel at the wrong bytes,
which is silently wrong output rather than a crash.

So measure it. This hooks `Mxfp4MoEMethod.process_weights_after_loading`, snaps
every expert tensor's identity before and after, and prints whether the storage
survived -- by `data_ptr`, which is the only thing a pointer table can carry.

    HIP_VISIBLE_DEVICES=0 /home/claudeuser/venv-vllm027/bin/python3 \
        tools/probe_vllm_expert_layout.py > /tmp/expert_layout.txt 2>&1

Reports, per tensor: shape, dtype, contiguity, nbytes, and the before/after
data_ptr with a SAME/MOVED/GONE verdict. Any tensor marked SAME is one a
pointer table could alias without copying it.
"""

import os
import sys

os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
os.environ.setdefault("VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION", "1")
# The engine core normally runs in a CHILD process, where a hook installed here
# does not exist -- the probe would report "hook never fired" and look like a
# vLLM-changed-shape result rather than a plumbing mistake. Force in-process.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch  # noqa: E402

MODEL = os.environ.get("TITAN_MODEL", "/home/claudeuser/models/gpt-oss-120b")

#: Names probed on the experts module. w13/w2 are the block data; *_scale the
#: E8M0 exponents; *_bias the bf16 per-output biases.
NAMES = ("w13_weight", "w13_weight_scale", "w13_bias",
         "w2_weight", "w2_weight_scale", "w2_bias")

records = []


def _unwrap(obj):
    """The underlying torch.Tensor, through triton_kernels' wrapper types.

    The TRITON path reassigns `layer.w13_weight` to a `triton_kernels.tensor`
    wrapper rather than a Parameter, so a plain isinstance check reports the
    tensor as missing when it is merely boxed. Try the documented spellings and
    fall back to reporting the type, which is itself the useful answer.
    """
    if obj is None:
        return None, "None"
    if isinstance(obj, torch.Tensor):
        return obj, type(obj).__name__
    for attr in ("data", "storage", "base", "_data"):
        inner = getattr(obj, attr, None)
        if isinstance(inner, torch.Tensor):
            return inner, f"{type(obj).__name__}.{attr}"
        # triton_kernels wraps twice: Tensor.storage.data
        if inner is not None and not isinstance(inner, torch.Tensor):
            deeper = getattr(inner, "data", None)
            if isinstance(deeper, torch.Tensor):
                return deeper, f"{type(obj).__name__}.{attr}.data"
    return None, type(obj).__name__


def _snap(mod, keep_bytes=False):
    """Identity of every expert tensor on `mod`, keyed by name.

    `keep_bytes` clones the raw storage so the after-snapshot can prove the
    CONTENTS survived, not just the address. That distinction is load-bearing:
    the AITER_MXFP4_BF16 path de-interleaves w13 with an in-place
    `w13_weight.view(torch.uint8).copy_(...)` (oracle/mxfp4.py:1027), which
    keeps data_ptr identical while rewriting every byte. An address-only probe
    would call that SAME and send a pointer-aliasing kernel at permuted rows --
    wrong output, no crash. ~1.7 GiB per layer, one layer only.
    """
    out = {}
    for n in NAMES:
        t, how = _unwrap(getattr(mod, n, None))
        if t is None:
            out[n] = dict(present=False, how=how)
        else:
            # Compare the STORAGE base, not data_ptr: a transposed view has the
            # same data_ptr but so would a fresh allocation that happens to be
            # reused, and a sliced view would differ while sharing the buffer.
            rec = dict(
                present=True, how=how, ptr=t.untyped_storage().data_ptr(), t=t,
                shape=tuple(t.shape), dtype=str(t.dtype),
                stride=tuple(t.stride()), contig=t.is_contiguous(),
                nbytes=t.numel() * t.element_size())
            if keep_bytes:
                rec["bytes"] = _flat_bytes(t).clone()
            out[n] = rec
    return out


def _flat_bytes(t):
    """The tensor's STORAGE as a flat uint8 vector, in physical memory order.

    Deliberately not `.contiguous()`. After the TRITON swizzle w13 is a
    transposed *view* of the same allocation: `.contiguous()` would materialise
    that transpose and compare a permuted copy against the original, reporting
    REWRITTEN for a buffer nothing touched. (This probe did exactly that on its
    first run.) The question is what a kernel reading the allocation linearly
    sees, which is the untyped storage.
    """
    s = t.detach().untyped_storage()
    return torch.empty(0, dtype=torch.uint8, device=t.device).set_(
        s, 0, (s.nbytes(),))


def _same_bytes(before, after):
    """Did the storage keep its contents? None if not comparable."""
    b = before.get("bytes")
    if b is None or not after.get("present"):
        return None
    a = _flat_bytes(after["t"])
    if a.numel() != b.numel():
        return None
    return bool(torch.equal(a, b))


def _install_hook():
    from vllm.model_executor.layers.quantization import mxfp4 as m

    # Do NOT hardcode a class name. GPT-OSS on 0.27.1 goes through
    # `GptOssMxfp4MoEMethod`, not `Mxfp4MoEMethod` -- an earlier version of this
    # probe guessed the latter, hooked a class the model never instantiates, and
    # reported "hook never fired", which reads as a finding rather than as the
    # plumbing bug it was. Hook every class in the module that defines the
    # method and let the run report which one actually fired.
    hooked = []
    for cls_name in dir(m):
        cls = getattr(m, cls_name)
        if (isinstance(cls, type)
                and "process_weights_after_loading" in vars(cls)):
            _wrap(cls)
            hooked.append(cls_name)
    if not hooked:
        raise SystemExit("no class in quantization.mxfp4 defines "
                         "process_weights_after_loading; vLLM layout changed")
    print(f"[probe] hooked: {', '.join(hooked)}", flush=True)


def _wrap(cls):
    orig = cls.process_weights_after_loading

    def wrapped(self, layer):
        # Record the first call that is actually about EXPERTS. Hooking every
        # class in the module also catches UnquantizedLinearMethod, which runs
        # first and carries none of these tensors -- taking "first call" would
        # report six GONE rows and conclude vLLM freed weights it never held.
        # Only one layer is captured: all 36 take the identical path.
        want = not records and any(hasattr(layer, n) for n in NAMES)
        before = _snap(layer, keep_bytes=True) if want else None
        out = orig(self, layer)
        if want:
            records.append(dict(
                cls=cls.__name__,
                backend=str(getattr(self, "mxfp4_backend", "?")),
                before=before, after=_snap(layer)))
        return out

    cls.process_weights_after_loading = wrapped


def _report():
    if not records:
        print("[probe] hook never fired -- this model's MoE is not going "
              "through Mxfp4MoEMethod, so nothing was measured.")
        return 1
    r = records[0]
    print(f"\nfired on: {r['cls']}   backend selected: {r['backend']}\n")
    hdr = f"{'tensor':<18} {'addr':<6} {'bytes':<9} {'shape':<22} " \
          f"{'dtype':<16} {'MiB':>8}"
    print(hdr)
    print("-" * len(hdr))
    aliasable = 0
    for n in NAMES:
        b, a = r["before"][n], r["after"][n]
        if not a["present"]:
            print(f"{n:<18} {'GONE':<6} {'-':<9} (deleted; was {b.get('shape')})")
            continue
        addr = ("NEW" if not b["present"]
                else "same" if a["ptr"] == b["ptr"] else "MOVED")
        eq = _same_bytes(b, a)
        byts = {True: "identical", False: "REWRITTEN", None: "?"}[eq]
        print(f"{n:<18} {addr:<6} {byts:<9} {str(a['shape']):<22} "
              f"{a['dtype']:<16} {a['nbytes'] / 2**20:>8.1f}")
        if addr == "MOVED":
            print(f"{'':<18} {'':<6} {'':<9} was @ {b['ptr']:#x} "
                  f"-> now {a['ptr']:#x}")
        print(f"{'':<18} {'':<6} {'':<9} via {a['how']}, "
              f"stride={a['stride']}, contig={a['contig']}")
        if addr == "same" and eq is True:
            aliasable += a["nbytes"]

    print(f"\nper-layer bytes that are BOTH at the original address AND "
          f"unchanged: {aliasable / 2**20:.1f} MiB")
    print(f"across 36 layers: {aliasable * 36 / 2**30:.2f} GiB")
    print("\nOnly 'same + identical' can be aliased by a pointer table. "
          "'same + REWRITTEN' is the dangerous case: the address survives "
          "an in-place permute, so an address-only check would pass while "
          "the bytes moved underneath it.")
    _report_pad(r)
    return 0


#: True hidden/intermediate for GPT-OSS 120B, and the reduction length the
#: kernel would actually run if it aliased vLLM's buffer. vLLM pads K to 3072;
#: 2880 is unusable because the MFMA is 16x16x128 and 2880/128 = 22.5, so the
#: nearest legal reduction is 2944 (23 iters, odd -- which is also what
#: MPK_MFMA_PINGPONG_SCHED's static_assert requires).
TRUE_K = 2880
REDUCE_K = 2944


def _report_pad(r):
    """Is vLLM's K-axis pad zero, and is the 2880..2944 slice we would newly
    reduce over also zero?

    This is the fact that decides whether titan can alias vLLM's buffer while
    keeping K=2944. Aliasing means the row stride becomes vLLM's 3072/2 bytes
    but the MFMA still runs 23 iterations = 2944 columns, so columns
    [2880, 2944) -- pad, never part of the real weight -- get folded into every
    expert's dot product. If they are zero that is exactly a no-op. If they are
    not, the result is silently wrong: no crash, no NaN, just 64 columns of
    garbage summed into all 128 experts of all 36 layers.

    Reported per tensor as three counts so a nonzero answer says WHERE.
    """
    print("\n\n--- K-axis pad: can the reduction run to 2944 over vLLM's "
          "buffer? ---")
    for n in ("w13_weight", "w2_weight"):
        a = r["after"][n]
        if not a["present"]:
            print(f"{n:<14} absent")
            continue
        t = a["t"]
        # Physical layout is [E, out, K/2] regardless of the transposed view
        # the swizzle leaves behind: one uint8 holds two MXFP4 values, so the
        # byte index for column c is c//2.
        flat = _flat_bytes(t)
        kbytes = 3072 // 2
        rows = flat.numel() // kbytes
        m = flat[:rows * kbytes].view(rows, kbytes)
        real = m[:, :TRUE_K // 2]
        newly = m[:, TRUE_K // 2:REDUCE_K // 2]   # folded in by aliasing
        beyond = m[:, REDUCE_K // 2:]             # stays outside the reduction
        print(f"{n:<14} rows={rows}  "
              f"nonzero: real[0:{TRUE_K}]={int((real != 0).sum())}  "
              f"NEWLY-REDUCED[{TRUE_K}:{REDUCE_K}]={int((newly != 0).sum())}  "
              f"beyond[{REDUCE_K}:3072]={int((beyond != 0).sum())}")
    print("\nNEWLY-REDUCED must be 0. It is the only column range that "
          "changes meaning when titan aliases vLLM's buffer at stride 3072 "
          "while keeping a 2944-long reduction.")


def main():
    _install_hook()
    from vllm import LLM, SamplingParams

    # Stock path only: the point is to observe what vLLM does to its OWN
    # weights, so titan's plugin must NOT be loaded. One token is enough --
    # process_weights_after_loading runs during engine init. No CUSTOM
    # attention backend either: that exists to give titan an aliasable KV
    # layout, and registering it needs the plugin this probe deliberately
    # disables. The stock backend reaches the same MoE weight-load path.
    os.environ["VLLM_PLUGINS"] = ""
    llm = LLM(model=MODEL, max_num_seqs=1, enforce_eager=True,
              max_model_len=2048, dtype="bfloat16", block_size=16,
              disable_hybrid_kv_cache_manager=True,
              gpu_memory_utilization=float(
                  os.environ.get("TITAN_GPU_MEM_UTIL", "0.9")))
    llm.generate(["hi"], SamplingParams(temperature=0.0, max_tokens=1))
    sys.exit(_report())


if __name__ == "__main__":
    main()
