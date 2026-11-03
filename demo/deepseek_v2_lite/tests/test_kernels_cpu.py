"""kernels/*.cuh run on the CPU shim (tests/cpu_shim.h) vs fleet_ds.reference at real shapes.
Builds tests/cpu_harness.cc with clang++ into a temp dir and drives it through ctypes (no pybind).
Gates = docs/design.md 5.2: per-op max|d| <= 2e-2*max|ref| + 1e-3, routing exact, layer 5% + cosine >= 0.999."""
import ctypes, math, os, shutil, subprocess, sys, tempfile
import pytest, torch, torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "ref"), os.path.join(ROOT, "tests")]
from fleet_ds import reference as R  # noqa: E402
from fleet_ds.graph import DIMS  # noqa: E402
from fleet_ds.sim import CPU_FLAGS  # noqa: E402
from test_reference import REAL, make_layer  # noqa: E402

H, DN, DR, DC, DV, TMAX, CHUNK, NCHUNK = 16, 128, 64, 512, 128, 1152, 128, 9
E, TOPK, HID, VCHUNK, VPARTS = 64, 6, 2048, 1024, 100
EPS = ctypes.c_float(1e-6)
BF = torch.bfloat16
pytestmark = pytest.mark.skipif(shutil.which("clang++") is None, reason="clang++ not installed")


HARNESS = os.path.join(ROOT, "tests", "cpu_harness.cc")  # kernels: <repo>/include/mirage/persistent_kernel/kernels


def build(src, defs=()):
    """clang++ the harness (or a reference copy) into its own .so. Inline functions are hidden so two builds of the same
    kernels with different switch -D flags cannot interpose each other's instantiations."""
    out = os.path.join(tempfile.mkdtemp(prefix="fleet_k_"), "k.so")
    subprocess.check_call(["clang++", "-x", "c++", "-std=c++17", "-O2", "-fPIC", "-shared", "-fvisibility-inlines-hidden",
                           *CPU_FLAGS, *defs, src, "-o", out])
    lib = ctypes.CDLL(out)
    if src == HARNESS:
        lib.k_switches.restype = ctypes.c_char_p
    return lib


@pytest.fixture(scope="module")
def K():
    return build(HARNESS)


@pytest.fixture(scope="module")
def K1():
    """The pre-vectorisation gemv (tests/gemv_bf16_v1.cc) in its own .so: the bit-identity oracle for the new one."""
    return build(os.path.join(ROOT, "tests", "gemv_bf16_v1.cc"))


def P(t, off=0):  # data pointer (+ element offset)
    return ctypes.c_void_p(t.data_ptr() + off * t.element_size())


def close(out, ref, tol=2e-2, name=""):
    out, ref = out.float(), ref.float()
    err = (out - ref).abs().max().item()
    bound = tol * ref.abs().max().item() + 1e-3
    print(f"  {name}: max|d|={err:.3e} bound={bound:.3e} max|ref|={ref.abs().max().item():.3e}")
    assert err <= bound, (name, err, bound)
    return err


def interleave(g, u, blk):  # gate/up rows in alternating `blk`-row blocks (design.md 4.1)
    K_ = g.shape[1]
    return torch.stack([g.reshape(-1, blk, K_), u.reshape(-1, blk, K_)], 1).reshape(-1, K_).contiguous()


def gemv(K, x, w, epi=0, res=None, rows=64, fp32=False):
    N = w.shape[0]
    out = torch.empty(N // 2 if epi == 2 else N, dtype=torch.float32 if fp32 else BF)
    for t in range(N // rows):
        r = K.k_gemv(P(x), P(w, t * rows * w.shape[1]), P(out, t * (rows // 2 if epi == 2 else rows)),
                     P(res, t * rows) if res is not None else None, w.shape[1], rows, epi, int(fp32))
        assert r == 0
    return out


# ---- per-kernel tests --------------------------------------------------------------------------------

@pytest.mark.parametrize("Kdim", [512, 1408, 2048, 2816, 3648, 10944])
def test_gemv_plain(K, Kdim):
    torch.manual_seed(Kdim)
    x, w = torch.randn(Kdim).to(BF), (torch.randn(64, Kdim) / math.sqrt(Kdim)).to(BF)
    close(gemv(K, x, w), F.linear(x, w), name=f"gemv K={Kdim}")
    close(gemv(K, x, w, fp32=True), F.linear(x.float(), w.float()), name=f"gemv fp32 out K={Kdim}")


def test_gemv_epilogues(K):
    torch.manual_seed(1)
    x = torch.randn(2048).to(BF)
    w, res = (torch.randn(128, 2048) / 45).to(BF), torch.randn(128).to(BF)
    close(gemv(K, x, w, epi=1, res=res), res + F.linear(x, w), name="gemv +residual")
    g, u = (torch.randn(1408, 2048) / 45).to(BF), (torch.randn(1408, 2048) / 45).to(BF)
    ref = F.silu(F.linear(x, g)) * F.linear(x, u)
    close(gemv(K, x, interleave(g, u, 32), epi=2), ref, name="gemv silu 64-row tiles")
    close(gemv(K, x, interleave(g, u, 64), epi=2, rows=128), ref, name="gemv silu 128-row tiles")


@pytest.mark.parametrize("Kdim", [512, 1408, 2048, 2736, 2816, 3648, 10944])
def test_gemv_bit_identical_to_v1(K, K1, Kdim):
    """Vectorised gemv (4 rows per wave in flight, all loads up front) == the one-row-per-wave original, bit for bit,
    at every K and epilogue (design D10d: GEMV vectorisation must stay bit-identical)."""
    torch.manual_seed(Kdim + 1)
    x = torch.randn(Kdim).to(BF)
    for rows, epi, fp32 in ((64, 0, 0), (64, 0, 1), (64, 1, 0), (64, 2, 0), (128, 2, 0), (88, 2, 0)):
        w, res = (torch.randn(2 * rows, Kdim) / math.sqrt(Kdim)).to(BF), torch.randn(2 * rows).to(BF)
        a = gemv(K, x, w, epi, res if epi == 1 else None, rows, fp32)
        b = gemv(K1, x, w, epi, res if epi == 1 else None, rows, fp32)
        assert torch.equal(a, b), (Kdim, rows, epi, fp32)


def test_gemv_4row_tiles(K):
    """L5 gate GEMV as 16 tiles x 4 rows (one row per wave in flight) == the 64-row fp32 gemv, bit for bit."""
    torch.manual_seed(5)
    x, gate = torch.randn(HID).to(BF), (torch.randn(E, HID) / 45).to(BF)
    assert torch.equal(gemv(K, x, gate, rows=4, fp32=True), gemv(K, x, gate, fp32=True))


def test_splitk(K):
    torch.manual_seed(2)
    h, wd, res = torch.randn(10944).to(BF), (torch.randn(2048, 10944) / 105).to(BF), torch.randn(2048).to(BF)
    ws = torch.empty(4, 2048)
    kc = 10944 // 4
    for s in range(4):
        wslice = wd[:, s * kc:(s + 1) * kc].contiguous()
        ws[s] = gemv(K, h[s * kc:(s + 1) * kc].contiguous(), wslice, fp32=True)
    out = torch.empty(2048, dtype=BF)
    K.k_splitk_reduce(P(ws), P(res), P(out), 2048)
    close(out, res + F.linear(h, wd), name="split-K 4 + reduce + residual")


@pytest.mark.parametrize("N", [2048, 512])
def test_rmsnorm(K, N):
    torch.manual_seed(N)
    x, w = torch.randn(N).to(BF), torch.empty(N).uniform_(0.5, 1.5).to(BF)
    out = torch.empty(N, dtype=BF)
    K.k_rmsnorm(P(x), P(w), P(out), N, EPS)
    ref = R.rmsnorm(x, w, EPS.value)
    close(out, ref, name=f"rmsnorm {N}")
    assert torch.equal(out, ref), "rmsnorm should be bit-exact (same op order as HF)"


def rope_tables(cfg):
    cs, sn = zip(*[R.rope_cos_sin(p, cfg, BF) for p in range(TMAX * 8)])
    return torch.stack(cs).contiguous(), torch.stack(sn).contiguous()


@pytest.fixture(scope="module")
def moe():
    cfg, layer = make_layer(REAL, BF, dense=False)
    return cfg, layer, layer.state_dict()


@pytest.mark.parametrize("pos", [1024, 5000])
def test_mla_prep(K, moe, pos):
    cfg, _, w = moe
    torch.manual_seed(pos)
    xn = torch.randn(HID).to(BF)
    qkva = F.linear(xn, torch.cat([w["self_attn.q_proj.weight"], w["self_attn.kv_a_proj_with_mqa.weight"]]))
    cs, sn = rope_tables(cfg)
    ckvn, q_rope, kc = torch.empty(DC, dtype=BF), torch.empty(H, DN + DR, dtype=BF), torch.zeros(H, TMAX, DN + DR, dtype=BF)
    K.k_mla_prep(P(qkva), P(w["self_attn.kv_a_layernorm.weight"]), P(cs), P(sn), P(ckvn), P(q_rope), P(kc), pos % TMAX, pos, EPS)
    q_nope, q_pe = R.q_rope(xn, w, cfg, pos)
    c_ref, k_pe_ref = R.kv_new(xn, w, cfg, pos)
    close(q_rope[:, :DN], q_nope, name="q_nope copy"); assert torch.equal(q_rope[:, :DN], q_nope)
    close(q_rope[:, DN:], q_pe, name=f"q_pe rope pos {pos}")
    close(ckvn, c_ref, name="c_kv rmsnorm 512")
    close(kc[:, pos % TMAX, DN:], k_pe_ref.expand(H, -1), name=f"k_pe rope pos {pos}")
    assert torch.equal(q_rope[:, DN:], q_pe) and torch.equal(kc[0, pos % TMAX, DN:], k_pe_ref), "rope mirrors HF op order"
    assert kc[:, :pos % TMAX].abs().sum() == 0 and kc[:, pos % TMAX, :DN].abs().sum() == 0, "only k_pe slice written"


def attention_kernels(K, q_rope, kc, vc, T, scale):
    o_part, lse, o = torch.zeros(NCHUNK * H * DV), torch.zeros(NCHUNK * H), torch.empty(H * DV, dtype=BF)
    for h in range(H):
        for c in range(NCHUNK):
            K.k_attn_chunk(P(q_rope), P(kc), P(vc), P(o_part), P(lse), h, c, T, ctypes.c_float(scale))
        K.k_attn_merge(P(o_part), P(lse), P(o), h)
    return o, lse


@pytest.mark.parametrize("T", [1, 301, 1056])
def test_attention(K, moe, T):
    cfg = moe[0]
    torch.manual_seed(T)
    kc, vc = torch.randn(H, TMAX, DN + DR).to(BF), torch.randn(H, TMAX, DV).to(BF)
    q = (torch.randn(H, DN + DR) * 0.5).to(BF)
    o, lse = attention_kernels(K, q, kc, vc, T, R.softmax_scale(cfg))
    ref = R.attention(q[:, :DN], q[:, DN:], None, cfg, kc[:, :T], vc[:, :T])
    close(o.view(H, DV), ref, name=f"attention T={T}")
    live = -(-T // CHUNK)
    assert torch.isneginf(lse.view(NCHUNK, H)[live:]).all() and torch.isfinite(lse.view(NCHUNK, H)[:live]).all()


def test_moe_gate_topk(K, moe):
    """One task: gate GEMV (fp32 logits, same accumulation as a plain fp32-out gemv) + wave-parallel top-6."""
    cfg, _, w = moe
    logits, wt, idx = torch.empty(E), torch.empty(TOPK), torch.empty(TOPK, dtype=torch.int32)
    for seed in range(5):
        torch.manual_seed(seed)
        xn = torch.randn(HID).to(BF)
        K.k_moe_gate_topk(P(xn), P(w["mlp.gate.weight"]), P(logits), P(wt), P(idx))
        close(logits, F.linear(xn.float(), w["mlp.gate.weight"].float()), name=f"gate logits seed {seed}")
        assert torch.equal(logits, gemv(K, xn, w["mlp.gate.weight"], fp32=True)), "gate part == the fp32-out gemv, bit for bit"
        tw, ti = R.route(xn, w, cfg)
        assert sorted(idx.tolist()) == sorted(ti.tolist())
        assert torch.allclose(wt, tw[[ti.tolist().index(i) for i in idx.tolist()]], atol=1e-6)
        assert wt.tolist() == sorted(wt.tolist(), reverse=True)  # slot order = descending score
    gate = torch.zeros(E, HID, dtype=BF)
    gate[[3, 9, 17, 25, 40, 50, 60]] = xn  # 7 identical rows -> 7-way exact tie at the top: lowest 6 indices win
    K.k_moe_gate_topk(P(xn), P(gate), P(logits), P(wt), P(idx))
    assert idx.tolist() == [3, 9, 17, 25, 40, 50] and torch.allclose(wt, torch.softmax(logits, 0)[idx.long()], atol=1e-6)


def gemv_norm(K, x, nw, w, epi=0, rows=64, fp32=False):
    """NORM-prologue tiles (11b / 12e L6): every tile recomputes rmsnorm(x, nw) into its LDS before the GEMV."""
    N = w.shape[0]
    out = torch.empty(N // 2 if epi == 2 else N, dtype=torch.float32 if fp32 else BF)
    for t in range(N // rows):
        assert K.k_gemv_norm(P(x), P(nw), P(w, t * rows * w.shape[1]), P(out, t * (rows // 2 if epi == 2 else rows)), None,
                             w.shape[1], rows, epi, int(fp32), EPS) == 0
    return out


def rmsnorm_task(K, x, nw):
    out = torch.empty(x.numel(), dtype=BF)
    K.k_rmsnorm(P(x), P(nw), P(out), x.numel(), EPS)
    return out


@pytest.mark.parametrize("rows,epi,fp32,n,k", [(64, 0, False, 3648, 2048), (64, 2, False, 256, 2048), (176, 2, False, 352, 2048),
                                               (512, 0, True, 1024, 2048), (64, 0, False, 256, 512), (16, 0, False, 128, 2048),
                                               (32, 0, False, 128, 2048)],
                         ids=["qkv", "gate_up_silu", "w13_shared_silu", "lm_head", "k512", "qkv16", "qkv32"])
def test_fold_norm_gemv(K, rows, epi, fp32, n, k):
    """L6: RMS_NORM task + GEMV == the NORM-prologue GEMV, bit for bit (same fp32 norm loops per task)."""
    torch.manual_seed(n + k)
    x, nw, w = torch.randn(k).to(BF), torch.empty(k).uniform_(0.5, 1.5).to(BF), (torch.randn(n, k) / 40).to(BF)
    assert torch.equal(gemv_norm(K, x, nw, w, epi, rows, fp32), gemv(K, rmsnorm_task(K, x, nw), w, epi, rows=rows, fp32=fp32))


def test_fold_gate_norm(K):
    """L6: norm2 prologue + gate + top-k == RMS_NORM task + the gate + top-k task; xn2 published bit for bit."""
    torch.manual_seed(12)
    nw, gate = torch.empty(HID).uniform_(0.5, 1.5).to(BF), (torch.randn(E, HID) / 45).to(BF)
    l1, w1, i1 = torch.empty(E), torch.empty(TOPK), torch.empty(TOPK, dtype=torch.int32)
    l2, w2, i2, xn2 = torch.empty(E), torch.empty(TOPK), torch.empty(TOPK, dtype=torch.int32), torch.empty(HID, dtype=BF)
    for seed in range(3):
        x = torch.randn(HID).to(BF)
        xn = rmsnorm_task(K, x, nw)
        K.k_moe_gate_topk(P(xn), P(gate), P(l1), P(w1), P(i1))
        K.k_moe_gate_topk_norm(P(x), P(nw), P(gate), P(l2), P(w2), P(i2), P(xn2), EPS)
        assert torch.equal(xn2, xn) and torch.equal(l1, l2) and torch.equal(w1, w2) and torch.equal(i1, i2), seed


def attn_partials(K, q, kc, vc, T, scale=0.114721388):
    o_part, lse = torch.zeros(NCHUNK * H * DV), torch.zeros(NCHUNK * H)
    for h in range(H):
        for c in range(NCHUNK):
            K.k_attn_chunk(P(q), P(kc), P(vc), P(o_part), P(lse), h, c, T, ctypes.c_float(scale))
    return o_part, lse


def o_proj_merge(K, o_part, lse, wo, x, T):
    out = torch.empty(HID, dtype=BF)
    for t in range(32):
        K.k_o_proj_merge(P(o_part), P(lse), P(wo, t * 64 * HID), P(out, t * 64), P(x, t * 64), T)
    return out


@pytest.mark.parametrize("T", [1, 129, 301, 1056])
def test_fold_attn_merge(K, T):
    """L6: ATTN_MERGE (16 tasks) + O_PROJ_RESIDUAL == O_PROJ_RESIDUAL tiles with the merge prologue, bit for bit."""
    torch.manual_seed(T)
    kc, vc = torch.randn(H, TMAX, DN + DR).to(BF), torch.randn(H, TMAX, DV).to(BF)
    q = (torch.randn(H, DN + DR) * 0.5).to(BF)
    o_part, lse = attn_partials(K, q, kc, vc, T)
    o = torch.empty(H * DV, dtype=BF)
    for h in range(H):
        K.k_attn_merge(P(o_part), P(lse), P(o), h)
    x, wo = torch.randn(HID).to(BF), (torch.randn(HID, HID) / 45).to(BF)
    ref = gemv(K, o, wo, epi=1, res=x)
    assert torch.equal(o_proj_merge(K, o_part, lse, wo, x, T), ref)
    for rows in (16, 32):  # L4 O_PROJ tile rows with the merge prologue
        out = torch.empty(HID, dtype=BF)
        for t in range(HID // rows):
            assert K.k_o_proj_merge_rows(P(o_part), P(lse), P(wo, t * rows * HID), P(out, t * rows), P(x, t * rows), T, rows) == 0
        assert torch.equal(out, ref), rows


def test_fold_mla_prep(K, moe):
    """D11b: MLA_PREP + KV_B_PROJ + ATTN_CHUNK == KV_B_PROJ tiles (kv_a norm prologue, bit-identical to the RMSNorm task; k_pe
    from tile 4h) + ATTN_CHUNK (q RoPE prologue): identical K/V cache and attention partials; one writer per cache element."""
    cfg, _, w = moe
    torch.manual_seed(11)
    slot, T = 300, 301
    qkva, wkv = torch.randn(H * (DN + DR) + DC + DR).to(BF), w["self_attn.kv_b_proj.weight"]
    nw = w["self_attn.kv_a_layernorm.weight"]
    cs, sn = rope_tables(cfg)
    kc0, vc0 = torch.randn(H, TMAX, DN + DR).to(BF), torch.randn(H, TMAX, DV).to(BF)
    kc1, vc1, kc2, vc2 = kc0.clone(), vc0.clone(), kc0.clone(), vc0.clone()
    ckvn, q_rope = torch.empty(DC, dtype=BF), torch.empty(H, DN + DR, dtype=BF)
    K.k_mla_prep(P(qkva), P(nw), P(cs), P(sn), P(ckvn), P(q_rope), P(kc1), slot, slot, EPS)
    writes = torch.zeros(H, DN + DR + DV, dtype=torch.int32)
    for n in range(64):
        h, sub = n // 4, n % 4
        o1 = P(kc1, (h * TMAX + slot) * (DN + DR) + sub * 64) if sub < 2 else P(vc1, (h * TMAX + slot) * DV + (sub - 2) * 64)
        o2 = P(kc2, (h * TMAX + slot) * (DN + DR) + sub * 64) if sub < 2 else P(vc2, (h * TMAX + slot) * DV + (sub - 2) * 64)
        assert K.k_gemv(P(ckvn), P(wkv, n * 64 * DC), o1, None, DC, 64, 0, 0) == 0
        before = torch.cat([kc2[:, slot], vc2[:, slot]], 1).clone()
        K.k_kv_b_proj(P(qkva), P(nw), P(wkv, n * 64 * DC), P(cs), P(sn), o2, P(kc2), n, slot, slot, EPS)
        writes += (torch.cat([kc2[:, slot], vc2[:, slot]], 1) != before).int()
    assert torch.equal(kc1, kc2) and torch.equal(vc1, vc2)
    assert writes.max() <= 1, "one writer per cache element"
    sc = ctypes.c_float(R.softmax_scale(cfg))
    p1, l1, p2, l2 = torch.zeros(NCHUNK * H * DV), torch.zeros(NCHUNK * H), torch.zeros(NCHUNK * H * DV), torch.zeros(NCHUNK * H)
    for h in range(H):
        for c in range(NCHUNK):
            K.k_attn_chunk(P(q_rope), P(kc1), P(vc1), P(p1), P(l1), h, c, T, sc)
            K.k_attn_chunk_rope(P(qkva), P(kc2), P(vc2), P(p2), P(l2), h, c, T, sc, P(cs), P(sn), slot)
    assert torch.equal(p1, p2) and torch.equal(l1, l2)


def test_moe_combine(K):
    torch.manual_seed(3)
    e_out, res = torch.randn(E + 1, HID).to(BF), torch.randn(HID).to(BF)
    wt, idx = torch.rand(TOPK) / 4, torch.tensor([3, 17, 64 - 1, 40, 8, 22], dtype=torch.int32)
    out = torch.empty(HID, dtype=BF)
    K.k_moe_combine(P(e_out), P(wt), P(idx), P(res), P(out))
    y = torch.zeros(HID)
    for k in range(TOPK):
        y += wt[k] * e_out[idx[k]].float()
    ref = res + (y.to(BF) + e_out[E])
    close(out, ref, name="moe combine")
    assert torch.equal(out, ref), "combine is HF's exact order -> bit-exact"


def test_embed_argmax(K):
    torch.manual_seed(4)
    table, toks = torch.randn(1000, HID).to(BF), torch.tensor([5, 77, 999], dtype=torch.int64)
    out = torch.empty(HID, dtype=BF)
    K.k_embed(P(table), P(toks), 2, P(out))
    assert torch.equal(out, table[999])
    logits = torch.randn(VPARTS * VCHUNK)
    logits[70000] = logits.max() + 1; logits[70001] = logits[70000]  # tie: lowest index must win
    vals, idxs, tok = torch.empty(VPARTS), torch.empty(VPARTS, dtype=torch.int64), torch.empty(1, dtype=torch.int64)
    for c in range(VPARTS):
        K.k_argmax_partial(P(logits, c * VCHUNK), P(vals, c), P(idxs, c))
    K.k_argmax_reduce(P(vals), P(idxs), P(tok))
    assert tok.item() == torch.argmax(logits).item() == 70000


# ---- one full layer through the naive-path kernels vs reference.decode_step ------------------------------

def run_layer(K, cfg, w, x, kc, vc, slot, dense):
    """Ops exactly as docs/design.md 2.4 / 2.5 (kernel calls stand in for tasks; pointers pre-offset)."""
    cs, sn = rope_tables(cfg)
    T = slot + 1
    xn = torch.empty(HID, dtype=BF); K.k_rmsnorm(P(x), P(w["input_layernorm.weight"]), P(xn), HID, EPS)     # 1
    qkva = gemv(K, xn, torch.cat([w["self_attn.q_proj.weight"], w["self_attn.kv_a_proj_with_mqa.weight"]]))  # 2 (57)
    wkv, nkv = w["self_attn.kv_b_proj.weight"], w["self_attn.kv_a_layernorm.weight"]
    for n in range(64):                                                                                    # 3 (64, scattered; kv_a norm + k_pe, 11b)
        h, sub = n // 4, n % 4
        out = P(kc, (h * TMAX + slot) * (DN + DR) + sub * 64) if sub < 2 else P(vc, (h * TMAX + slot) * DV + (sub - 2) * 64)
        K.k_kv_b_proj(P(qkva), P(nkv), P(wkv, n * 64 * DC), P(cs), P(sn), out, P(kc), n, slot, slot, EPS)
    o_part, lse, o = torch.zeros(NCHUNK * H * DV), torch.zeros(NCHUNK * H), torch.empty(H * DV, dtype=BF)
    for h in range(H):                                                                                     # 4 (144, q RoPE) + 5 (16)
        for c in range(NCHUNK):
            K.k_attn_chunk_rope(P(qkva), P(kc), P(vc), P(o_part), P(lse), h, c, T, ctypes.c_float(R.softmax_scale(cfg)), P(cs), P(sn), slot)
        K.k_attn_merge(P(o_part), P(lse), P(o), h)
    x1 = gemv(K, o, w["self_attn.o_proj.weight"], epi=1, res=x)                                            # 7 (32)
    xn2 = torch.empty(HID, dtype=BF); K.k_rmsnorm(P(x1), P(w["post_attention_layernorm.weight"]), P(xn2), HID, EPS)  # 8
    if dense:
        hbuf = gemv(K, xn2, interleave(w["mlp.gate_proj.weight"], w["mlp.up_proj.weight"], 32), epi=2)   # 342
        wd, kc4 = w["mlp.down_proj.weight"], 10944 // 4
        ws = torch.empty(4, HID)
        for s in range(4):                                                                                 # 32 x 4
            ws[s] = gemv(K, hbuf[s * kc4:(s + 1) * kc4].contiguous(), wd[:, s * kc4:(s + 1) * kc4].contiguous(), fp32=True)
        x2 = torch.empty(HID, dtype=BF); K.k_splitk_reduce(P(ws), P(x1), P(x2), HID)                      # 1
        return x2, None
    logits, wt, idx = torch.empty(E), torch.empty(TOPK), torch.empty(TOPK, dtype=torch.int32)
    K.k_moe_gate_topk(P(xn2), P(w["mlp.gate.weight"]), P(logits), P(wt), P(idx))                           # 9 (gate + top-k)
    hbuf, e_out = torch.zeros(E + 1, 2816, dtype=BF), torch.zeros(E + 1, HID, dtype=BF)
    slots = idx.tolist() + [E]  # slot s -> expert idx[s] read at run time (10a); slot 6 = the shared expert
    n13, n2 = DIMS["w13_tiles"], DIMS["w2_tiles"]
    for e in slots:                                                                                        # 10 (w13_tiles x 7)
        rows = (5632 if e == E else 2816) // n13
        w13 = interleave(*R.expert_weights(w, cfg, e)[:2], rows // 2)
        for t in range(n13):
            assert K.k_gemv(P(xn2), P(w13, t * rows * HID), P(hbuf, e * 2816 + t * rows // 2), None, HID, rows, 2, 0) == 0
    for e in slots:                                                                                        # 11 (w2_tiles x 7)
        Kd, w2, rows = (2816 if e == E else 1408), R.expert_weights(w, cfg, e)[2], HID // n2
        for t in range(n2):
            assert K.k_gemv(P(hbuf, e * 2816), P(w2, t * rows * Kd), P(e_out, e * HID + t * rows), None, Kd, rows, 0, 0) == 0
    x2 = torch.empty(HID, dtype=BF)
    K.k_moe_combine(P(e_out), P(wt), P(idx), P(x1), P(x2))                                                 # 12
    off = [e for e in range(E) if e not in slots]
    assert (hbuf[off] == 0).all() and (e_out[off] == 0).all(), "unselected experts never touched"
    return x2, (wt, idx)


@pytest.mark.parametrize("dense", [False, True], ids=["moe_layer", "dense_layer0"])
def test_layer_chain(K, moe, dense):
    if dense:
        cfg, layer = make_layer(REAL, BF, dense=True); w = layer.state_dict()
    else:
        cfg, _, w = moe
    torch.manual_seed(7)
    slot = 300  # T = 301 -> chunks 0,1 full, chunk 2 partial, 3..8 empty
    x = torch.randn(HID).to(BF)
    kc, vc = torch.randn(H, TMAX, DN + DR).to(BF), torch.randn(H, TMAX, DV).to(BF)
    ref_k, ref_v = kc[:, :slot + 1].clone(), vc[:, :slot + 1].clone()
    out, routing = run_layer(K, cfg, w, x, kc, vc, slot, dense)
    ref, ref_routing = R.decode_step(x, w, cfg, ref_k, ref_v, slot, dense)
    err = close(out, ref, tol=5e-2, name=f"layer output ({'dense' if dense else 'moe'})")
    cos = F.cosine_similarity(out.float(), ref.float(), dim=0).item()
    print(f"  cosine={cos:.6f}")
    assert cos >= 0.999
    close(kc[:, slot], ref_k[:, slot], name="new K row"); close(vc[:, slot], ref_v[:, slot], name="new V row")
    assert torch.equal(kc[:, slot, DN:], ref_k[:, slot, DN:]), "k_pe cache row bit-exact"
    if routing is not None:
        assert sorted(routing[1].tolist()) == sorted(ref_routing[1].tolist())


def test_gate_split_and_kv_b_rows(K, moe):
    """12e L5: the gate GEMV as 8/16/32-row fp32 tiles + the top-k task == the fused gate+top-k task, bit for bit.
    L4: KV_B_PROJ tiles of 32 / 16 rows write exactly the K/V rows (incl. k_pe) of the 64-row tiles."""
    cfg, _, w = moe
    gate = w["mlp.gate.weight"]
    for seed in range(3):
        torch.manual_seed(seed)
        xn = torch.randn(HID).to(BF)
        l0, w0, i0 = torch.empty(E), torch.empty(TOPK), torch.empty(TOPK, dtype=torch.int32)
        K.k_moe_gate_topk(P(xn), P(gate), P(l0), P(w0), P(i0))
        for rows in (8, 16, 32):
            l1, w1, i1 = torch.empty(E), torch.empty(TOPK), torch.empty(TOPK, dtype=torch.int32)
            for t in range(E // rows):
                assert K.k_gemv(P(xn), P(gate, t * rows * HID), P(l1, t * rows), None, HID, rows, 0, 1) == 0
            K.k_moe_topk(P(l1), P(w1), P(i1))
            assert torch.equal(l0, l1) and torch.equal(w0, w1) and torch.equal(i0, i1), rows
    torch.manual_seed(12)
    slot, ref = 300, None
    qkva, wkv, nw = torch.randn(H * (DN + DR) + DC + DR).to(BF), w["self_attn.kv_b_proj.weight"], w["self_attn.kv_a_layernorm.weight"]
    cs, sn = rope_tables(cfg)
    for rows in (64, 32, 16):
        kc, vc, tph = torch.zeros(H, TMAX, DN + DR, dtype=BF), torch.zeros(H, TMAX, DV, dtype=BF), (DN + DV) // rows
        for n in range(4096 // rows):
            h, sub = n // tph, n % tph
            out = P(kc, (h * TMAX + slot) * (DN + DR) + sub * rows) if sub < tph // 2 else \
                P(vc, (h * TMAX + slot) * DV + (sub - tph // 2) * rows)
            assert K.k_kv_b_proj_rows(P(qkva), P(nw), P(wkv, n * rows * DC), P(cs), P(sn), out, P(kc), n, slot, slot, EPS, rows) == 0
        ref = ref or (kc, vc)
        assert kc[:, slot].abs().sum() > 0 and torch.equal(kc, ref[0]) and torch.equal(vc, ref[1]), rows


# ---- kernel switches (design D12e-g, kernels/switches.h): every value == the default kernels -----

DEFAULT_SWITCHES = {"DSK_COMBINE_BLOCK": 1, "DSK_NORM_FIXED": 1, "DSK_MERGE_FIXED": 2, "DSK_ATTN_SCORE_HOIST": 0,
                    "DSK_GEMV_RPW": 0}  # switches.h since 2026-09-15 = the session-8 measured best
SWITCH_SETS = {  # each set is one extra harness build; its outputs must equal the default build's bit for bit
    "off": {"DSK_COMBINE_BLOCK": 0, "DSK_NORM_FIXED": 0, "DSK_MERGE_FIXED": 0},  # the kernels measured before session 8
    "on1": {"DSK_COMBINE_BLOCK": 1, "DSK_NORM_FIXED": 1, "DSK_MERGE_FIXED": 1, "DSK_ATTN_SCORE_HOIST": 1,
            "DSK_GEMV_RPW": 4},
    "on2": {"DSK_COMBINE_BLOCK": 1, "DSK_NORM_FIXED": 1, "DSK_MERGE_FIXED": 2, "DSK_ATTN_SCORE_HOIST": 1,
            "DSK_GEMV_RPW": 8},
    "rpw16": {"DSK_GEMV_RPW": 16},
}


def switch_outputs(K):
    """Every entry point a switch touches, on fixed random inputs (bit identity does not need realistic values)."""
    g = torch.Generator().manual_seed(27)
    bf = lambda *s: torch.randn(*s, generator=g).to(BF)
    o = {}
    e_out, res, wt = bf(E + 1, HID), bf(HID), torch.rand(TOPK, generator=g) / 4
    for i, ids in enumerate(([3, 17, 63, 40, 8, 22], [0, 1, 2, 3, 4, 63], [63, 63, 0, 0, 5, 5])):
        idx, out = torch.tensor(ids, dtype=torch.int32), torch.empty(HID, dtype=BF)
        K.k_moe_combine(P(e_out), P(wt), P(idx), P(res), P(out))
        o[f"combine{i}"] = out
    for n in (2048, 512):
        x, nw, out = bf(n), bf(n), torch.empty(n, dtype=BF)
        K.k_rmsnorm(P(x), P(nw), P(out), n, EPS)
        o[f"rmsnorm{n}"] = out
    slot, nw = 300, bf(DC)
    qkva, cs, sn = bf(H * (DN + DR) + DC + DR), bf(TMAX, DR), bf(TMAX, DR)
    ckvn, q_rope, kc = torch.empty(DC, dtype=BF), torch.empty(H, DN + DR, dtype=BF), torch.zeros(H, TMAX, DN + DR, dtype=BF)
    K.k_mla_prep(P(qkva), P(nw), P(cs), P(sn), P(ckvn), P(q_rope), P(kc), slot, slot, EPS)
    o["mla_prep_ckvn"], o["mla_prep_q"], o["mla_prep_kc"] = ckvn, q_rope, kc
    wkv, kc, vc = (torch.randn(64 * 64, DC, generator=g) / 20).to(BF), torch.zeros(H, TMAX, DN + DR, dtype=BF), torch.zeros(H, TMAX, DV, dtype=BF)
    for n in range(64):
        h, sub = n // 4, n % 4
        out = P(kc, (h * TMAX + slot) * (DN + DR) + sub * 64) if sub < 2 else P(vc, (h * TMAX + slot) * DV + (sub - 2) * 64)
        K.k_kv_b_proj(P(qkva), P(nw), P(wkv, n * 64 * DC), P(cs), P(sn), out, P(kc), n, slot, slot, EPS)
    o["kv_b_kc"], o["kv_b_vc"] = kc, vc
    for rows, epi, fp32, n, k in ((64, 0, 0, 128, 2048), (64, 2, 0, 128, 2048), (176, 2, 0, 352, 2048), (512, 0, 1, 512, 2048),
                                  (64, 0, 0, 128, 512)):
        x, nw, w = bf(k), bf(k), (torch.randn(n, k, generator=g) / 40).to(BF)
        o[f"gemv_norm_{rows}_{epi}_{k}"] = gemv_norm(K, x, nw, w, epi, rows, bool(fp32))
    x, nw, gate = bf(HID), bf(HID), (torch.randn(E, HID, generator=g) / 45).to(BF)
    l, wt, idx, xn2 = torch.empty(E), torch.empty(TOPK), torch.empty(TOPK, dtype=torch.int32), torch.empty(HID, dtype=BF)
    K.k_moe_gate_topk_norm(P(x), P(nw), P(gate), P(l), P(wt), P(idx), P(xn2), EPS)
    o["gate_norm_logits"], o["gate_norm_w"], o["gate_norm_idx"], o["gate_norm_xn2"] = l, wt, idx, xn2
    kc, vc, q = bf(H, TMAX, DN + DR), bf(H, TMAX, DV), bf(H, DN + DR) * 0.5
    x, wo = bf(HID), (torch.randn(HID, HID, generator=g) / 45).to(BF)
    for T in (1, 129, 301, 1056):
        o_part, lse = attn_partials(K, q, kc, vc, T)
        o[f"attn_opart_{T}"], o[f"attn_lse_{T}"] = o_part, lse
        o[f"o_proj_merge_{T}"] = o_proj_merge(K, o_part, lse, wo, x, T)
    for T in (129, 1056):  # q RoPE prologue variant (ATTN_CHUNK as the graph runs it)
        o_part, lse = torch.zeros(NCHUNK * H * DV), torch.zeros(NCHUNK * H)
        for h in range(H):
            for c in range(NCHUNK):
                K.k_attn_chunk_rope(P(qkva), P(kc), P(vc), P(o_part), P(lse), h, c, T, ctypes.c_float(0.114721388), P(cs), P(sn), T - 1)
        o[f"attn_rope_opart_{T}"], o[f"attn_rope_lse_{T}"] = o_part, lse
    for rows, epi, fp32, k in ((4, 0, 1, 2048), (16, 0, 0, 2048), (64, 0, 0, 2048), (64, 0, 1, 2048), (64, 1, 0, 2048), (64, 2, 0, 2048), (88, 2, 0, 2048),
                               (176, 2, 0, 2048), (128, 0, 0, 2048), (64, 0, 0, 512), (64, 0, 0, 1408), (64, 0, 0, 2816),
                               (64, 0, 1, 2736), (512, 0, 1, 2048)):
        x, w, res = bf(k), (torch.randn(2 * rows, k, generator=g) / 40).to(BF), bf(2 * rows)
        o[f"gemv_{rows}_{epi}_{fp32}_{k}"] = gemv(K, x, w, epi, res if epi == 1 else None, rows, bool(fp32))
    x, gate = bf(HID), (torch.randn(E, HID, generator=g) / 45).to(BF)
    l, wt, idx = torch.empty(E), torch.empty(TOPK), torch.empty(TOPK, dtype=torch.int32)
    K.k_moe_gate_topk(P(x), P(gate), P(l), P(wt), P(idx))
    o["gate_logits"], o["gate_w"], o["gate_idx"] = l, wt, idx
    return o


@pytest.mark.parametrize("sw", SWITCH_SETS)
def test_switches_bit_identical(K, sw):
    defs = SWITCH_SETS[sw]
    Ks = build(HARNESS, [f"-D{k}={v}" for k, v in defs.items()])
    sws = lambda lib: {k: int(v) for k, v in (kv.split("=") for kv in lib.k_switches().decode().split())}
    got = sws(Ks)
    assert sws(K) == DEFAULT_SWITCHES
    assert all(got[k] == v for k, v in defs.items()), got
    a, b = switch_outputs(K), switch_outputs(Ks)
    bad = [k for k in a if not torch.equal(a[k], b[k])]
    assert not bad, f"switch set {sw} changes {bad}"
