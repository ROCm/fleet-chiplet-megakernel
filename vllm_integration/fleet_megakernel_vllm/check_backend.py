"""Verify FleetMKAttentionBackend's KV layout binds zero-copy and round-trips.

This is the gate that must pass before the backend is wired into a model. It
checks the things that can each independently be wrong, in the order that
localises a failure fastest:

  1. the declared shape is fleet_mk's `(2, nb, bs, n_kv, hs)`;
  2. `_split_kv_cache` returns contiguous K and V that are *views* of the pool
     (a copy here silently duplicates the cache and decode reads stale KV);
  3. fleet_mk's flat `[entries, kv_cache_stride]` reshape of each half is likewise
     a view -- same data_ptr;
  4. a real `reshape_and_cache_flash` write followed by a manual read of the
     same slots returns bit-identical values, on fleet_mk's layout AND on stock's
     packed layout, so a mismatch can be attributed to the layout rather than
     to the harness;
  5. the impl actually *constructs* -- see below;
  6. attention over fleet_mk's layout agrees numerically with attention over the
     stock packed layout holding the same logical KV.

Check 5 exists because of a real escape. The original gate exercised
`_split_kv_cache` unbound (`Impl._split_kv_cache(None, pool)`) rather than on an
instance, so it never ran `__init__` -- and `__init__` is exactly where the
parent class imports aiter, which is ABI-broken against this venv's torch. The
gate passed clean and the failure surfaced minutes later at engine startup, in a
traceback pointing at vLLM rather than at this file. A gate that skips the
constructor is not testing the class it claims to test.

Check 6 is what makes the layout swap trustworthy rather than plausible.
Checks 1-4 say the bytes land in the right places; only 6 says the *kernel*
reads them the same way, and it is the only check that exercises the descale
expansion in `_UnifiedAttn`.

Run:  python -m fleet_megakernel_vllm.check_backend
Exit code 0 iff every check passes.
"""

import sys

import torch


def _fail(msg):
    print(f"FAIL: {msg}")
    return False


def check_shape():
    from .backend import FleetMKAttentionBackend as B

    nb, bs, nkv, hs = 100, 16, 8, 64
    shape = B.get_kv_cache_shape(nb, bs, nkv, hs)
    want = (2, nb, bs, nkv, hs)
    print(f"  declared shape: {shape}")
    if shape != want:
        return _fail(f"shape {shape} != {want}")
    try:
        B.get_kv_cache_shape(nb, 15, nkv, hs)
    except ValueError:
        pass
    else:
        return _fail("block_size 15 should be rejected")
    print("  OK  shape is fleet_mk's split layout; non-multiple-of-16 rejected")
    return True


def _make_impl(num_heads=64, nkv=8, hs=64, sliding_window=None, sinks=None):
    """Construct a real FleetMKUnifiedAttentionImpl, GPT-OSS shaped by default."""
    from .backend import FleetMKUnifiedAttentionImpl

    return FleetMKUnifiedAttentionImpl(
        num_heads=num_heads,
        head_size=hs,
        scale=hs ** -0.5,
        num_kv_heads=nkv,
        alibi_slopes=None,
        sliding_window=sliding_window,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        sinks=sinks,
    )


def check_impl_constructs():
    """__init__ must complete without aiter, and bind a callable kernel.

    The parent's __init__ does `from aiter.ops.triton.unified_attention import
    unified_attention`; the installed aiter is a prebuilt extension linked
    against an older torch and raises `undefined symbol:
    _ZN3c103hip21warn_or_error_on_syncEv` under this venv. FleetMKUnifiedAttentionImpl
    bypasses that level of super() and binds vLLM's in-tree Triton kernel
    instead, so this must construct on a box with no working aiter at all.
    """
    impl = _make_impl()
    if not callable(getattr(impl, "unified_attention", None)):
        return _fail("impl has no callable unified_attention")
    # Attributes the parent's forward() reads; a skipped super() would drop them.
    for attr in ("num_heads", "num_kv_heads", "head_size", "scale", "sinks",
                 "sliding_window", "kv_cache_dtype", "fp8_dtype",
                 "logits_soft_cap", "num_queries_per_kv"):
        if not hasattr(impl, attr):
            return _fail(f"impl missing {attr!r} -- __init__ chain is wrong")
    if "aiter" in sys.modules:
        print("  note: aiter is importable here; the no-aiter path is still "
              "what this impl uses")
    # The fused rope paths would call into aiter ops at runtime; must be off.
    if impl.fused_rope_kvcache_supported():
        return _fail("fused_rope_kvcache_supported must be False (aiter op)")
    if impl.fused_qk_norm_rope_kvcache_supported():
        return _fail("fused_qk_norm_rope_kvcache_supported must be False")
    print("  OK  impl constructs with no aiter; Triton kernel bound, "
          "aiter fused-rope paths declined")
    return True


def check_split_is_view():
    """K/V halves, and fleet_mk's flat reshape of them, must alias the pool."""
    nb, bs, nkv, hs = 100, 16, 8, 64
    pool = torch.zeros(2, nb, bs, nkv, hs, dtype=torch.bfloat16, device="cuda")
    k, v = _make_impl(nkv=nkv, hs=hs)._split_kv_cache(pool)

    if k.data_ptr() != pool[0].data_ptr() or v.data_ptr() != pool[1].data_ptr():
        return _fail("split returned copies, not views")
    if not (k.is_contiguous() and v.is_contiguous()):
        return _fail(f"split halves not contiguous: {k.stride()}, {v.stride()}")
    print(f"  split -> K{tuple(k.shape)} stride{k.stride()} contiguous")

    entries, stride = nb * bs, nkv * hs
    kf, vf = k.reshape(entries, stride), v.reshape(entries, stride)
    if kf.data_ptr() != k.data_ptr() or vf.data_ptr() != v.data_ptr():
        return _fail("fleet_mk flat reshape is a copy -- zero-copy broken")
    print(f"  OK  fleet_mk flat view [{entries}, {stride}] aliases the pool")
    return True


def _roundtrip(layout):
    """Write via reshape_and_cache_flash, read back by hand, compare.

    Returns max abs diff over K and V, or None if the layout is unsupported.
    """
    from vllm import _custom_ops as ops

    nb, bs, nkv, hs = 64, 16, 8, 64
    n_tok = 24
    dev, dt = "cuda", torch.bfloat16
    torch.manual_seed(0)
    key = torch.randn(n_tok, nkv, hs, dtype=dt, device=dev)
    val = torch.randn(n_tok, nkv, hs, dtype=dt, device=dev)
    # Distinct, non-contiguous slots so a stride bug cannot cancel out.
    slots = torch.arange(n_tok, dtype=torch.long, device=dev) * 7 + 3

    if layout == "fleet_mk":
        pool = torch.zeros(2, nb, bs, nkv, hs, dtype=dt, device=dev)
        kc, vc = pool[0], pool[1]
    elif layout == "packed":
        pool = torch.zeros(nb, nkv, bs, 2 * hs, dtype=dt, device=dev)
        kc, vc = pool.transpose(1, 2).split(hs, dim=-1)
    else:
        raise ValueError(layout)

    one = torch.tensor(1.0, dtype=torch.float32, device=dev)
    ops.reshape_and_cache_flash(key, val, kc, vc, slots.to(torch.int64),
                                "auto", one, one)

    # Read back independently of the write path: index the block/offset the
    # slot maps to. If this agrees, the writer and fleet_mk's reader see the same
    # bytes at the same addresses.
    blk, off = slots // bs, slots % bs
    got_k = kc[blk, off]      # [n_tok, nkv, hs]
    got_v = vc[blk, off]
    return max((got_k - key).abs().max().item(),
               (got_v - val).abs().max().item())


def check_roundtrip():
    ok = True
    for layout in ("fleet_mk", "packed"):
        d = _roundtrip(layout)
        status = "OK " if d == 0.0 else "FAIL"
        print(f"  {status} {layout:>6} layout: max abs diff {d}")
        ok = ok and d == 0.0
    if ok:
        print("  both layouts bit-identical -- reshape_and_cache_flash is "
              "stride-aware, as stock vLLM's own packed path requires")
    return ok


def _attend(layout, sinks, seed=0):
    """Run one decode step of unified attention over `layout`; return output.

    Both layouts are filled from the *same* logical KV, so the two outputs must
    agree. Uses the same kernel entry the impl binds, with the same descale
    expansion, so this exercises _UnifiedAttn rather than reimplementing it.
    """
    nb, bs, nkv, hs = 8, 16, 8, 64
    n_q_heads, ctx = 64, 40
    dev, dt = "cuda", torch.bfloat16
    torch.manual_seed(seed)

    q = torch.randn(1, n_q_heads, hs, dtype=dt, device=dev)
    kv = torch.randn(2, ctx, nkv, hs, dtype=dt, device=dev)  # logical K/V

    if layout == "fleet_mk":
        pool = torch.zeros(2, nb, bs, nkv, hs, dtype=dt, device=dev)
        kc, vc = pool[0], pool[1]
    else:
        pool = torch.zeros(nb, nkv, bs, 2 * hs, dtype=dt, device=dev)
        kc, vc = pool.transpose(1, 2).split(hs, dim=-1)

    # Scatter the same logical KV into both, via the block/offset mapping the
    # kernel will use to read it back.
    pos = torch.arange(ctx, device=dev)
    kc[pos // bs, pos % bs] = kv[0]
    vc[pos // bs, pos % bs] = kv[1]

    out = torch.empty(1, n_q_heads, hs, dtype=dt, device=dev)
    block_table = torch.arange(nb, dtype=torch.int32, device=dev).view(1, nb)
    one = torch.tensor(1.0, dtype=torch.float32, device=dev)

    impl = _make_impl(num_heads=n_q_heads, nkv=nkv, hs=hs, sinks=sinks)
    impl.unified_attention(
        q=q, k=kc, v=vc, out=out,
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device=dev),
        max_seqlen_q=1,
        seqused_k=torch.tensor([ctx], dtype=torch.int32, device=dev),
        max_seqlen_k=ctx,
        softmax_scale=hs ** -0.5, causal=True, alibi_slopes=None,
        window_size=impl.sliding_window, block_table=block_table, softcap=0,
        q_descale=None, k_descale=one, v_descale=one,
        sinks=impl.sinks, output_scale=None,
    )
    return out


def check_attention_equivalence():
    """Attention over fleet_mk's layout == attention over the packed layout.

    Checks 1-4 only establish that bytes land where fleet_mk expects. This is the
    one that says the *kernel* agrees -- and the only one that exercises the
    k_descale/v_descale expansion, which is the single behavioural difference
    between the aiter call signature and vLLM's Triton one. Run with sinks too,
    since GPT-OSS uses them and they are the reason RocmAttentionBackend is
    unusable here.
    """
    ok = True
    for label, sinks in (("no sinks", None),
                         ("with sinks", torch.randn(64, dtype=torch.float32,
                                                    device="cuda"))):
        a = _attend("fleet_mk", sinks)
        b = _attend("packed", sinks)
        d = (a.float() - b.float()).abs().max().item()
        # Same kernel, same values, different strides: expect exact equality.
        # Any nonzero difference means the two layouts are not being read as
        # the same tensor, which is the failure this whole file is about.
        status = "OK " if d == 0.0 else "FAIL"
        print(f"  {status} {label:>10}: max abs diff vs packed layout {d}")
        ok = ok and d == 0.0
    if ok:
        print("  fleet_mk's layout is numerically indistinguishable from stock's")
    return ok


def main():
    if not torch.cuda.is_available():
        print("no GPU; cannot check the write path")
        sys.exit(1)
    import vllm
    print(f"vLLM {vllm.__version__}\n")

    checks = [("declared KV shape", check_shape),
              ("impl constructs without aiter", check_impl_constructs),
              ("split is a zero-copy view", check_split_is_view),
              ("reshape_and_cache_flash round-trip", check_roundtrip),
              ("attention matches stock layout", check_attention_equivalence)]
    bad = 0
    for name, fn in checks:
        print(f"[{name}]")
        try:
            if not fn():
                bad += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"FAIL: {name}: {e}")
            bad += 1
        print()
    print(f"{len(checks) - bad}/{len(checks)} checks passed")
    sys.exit(bad)


if __name__ == "__main__":
    main()
