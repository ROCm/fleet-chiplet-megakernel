"""AID-local placement for MoE expert weights on MI350/MI355.

MI350 in NPS1 interleaves memory over 7 HBM stacks (one is harvested; each is
36 GB, and 7*36 = 252 GB matches the visible capacity). The owning stack is a
mod-7 hash of the *physical* 4 KiB page number, and the two AIDs own 3 and 4 of
those stacks. Three consequences shape everything here:

* A contiguous weight slab is never AID-local. The kernel reads 200192 B per
  workgroup, 49 pages, and the longest single-AID run is 6 pages -- measured 0
  of 10699 slabs on one AID. So no tile map or reader-steering scheme can help;
  the bytes have to be placed.
* There is no compile-time map, but there is a cheap one. Homing follows the
  physical page, so two allocations agree only ~71% at the same offset and the
  map still has to be established at load. It does not have to be measured page
  by page: the label sequence repeats every 7 MiB (1792 pages) at 100.00% over
  1 GiB, and every allocation's map is a pure rotation of one 224-byte pattern
  (100.00% agreement at k=768 and k=1536, rotations always a multiple of the
  1 MiB interleave unit, so there are only 7 of them). Probing two periods and
  tiling turns a ~14 s classification of 63 GiB into a few ms.
* Work must be split 3:4, not evenly. Handing both AIDs the same number of
  slabs starves the 4-stack side behind the 3-stack side, which looks like a
  10% locality penalty but is really load imbalance. Which stack is harvested
  is per-part, so the ratio is measured rather than assumed.

Measured on one layer of gpt-oss-120b W13, this is worth 1.09x on the weight
read at fleet's operating point (and nothing at all once saturated, which is
why it has to be measured in the regime fleet actually runs in).
"""

import ctypes
import os

import torch

_PAGE = 4096
_lib = None


def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    for cand in (
        os.environ.get("MPK_AID_LAYOUT_LIB"),
        os.path.join(root, "libaid_pattern.so"),
        os.path.join(root, "libaid_layout.so"),
        "/tmp/libaid_layout.so",
    ):
        if cand and os.path.exists(cand):
            lib = ctypes.CDLL(cand)
            lib.aid_classify_pages.restype = ctypes.c_int
            lib.aid_classify_pages.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_ubyte),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_double),
            ]
            if hasattr(lib, "aid_classify_fast"):
                lib.aid_classify_fast.restype = ctypes.c_int
                lib.aid_classify_fast.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_ubyte),
                    ctypes.POINTER(ctypes.c_int),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                ]
            lib.aid_scatter_copy.restype = ctypes.c_int
            lib.aid_scatter_copy.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            _lib = lib
            return _lib
    raise RuntimeError(
        "libaid_layout.so not found; build it with\n"
        "  hipcc -O3 -std=c++17 --offload-arch=gfx950 -fPIC -shared \\\n"
        "    benchmark/aid_remote_hbm/aid_layout.hip -o libaid_layout.so\n"
        "or point MPK_AID_LAYOUT_LIB at it"
    )


def classify(buf: torch.Tensor):
    """Label every 4 KiB page of a device buffer by home AID.

    Destroys the buffer's contents: the classifier ping-pongs a flag in each
    page to time it. Call before the weights are written.

    Returns (side, n_aid0, seconds) with side a uint8 CPU tensor of 0/1.
    """
    lib = _load_lib()
    nbytes = buf.numel() * buf.element_size()
    assert nbytes % _PAGE == 0, "buffer must be a whole number of 4 KiB pages"
    npages = nbytes // _PAGE
    side = (ctypes.c_ubyte * npages)()
    n0 = ctypes.c_int(0)
    secs = ctypes.c_double(0.0)
    rc = lib.aid_classify_pages(
        ctypes.c_void_p(buf.data_ptr()),
        ctypes.c_size_t(nbytes),
        side,
        ctypes.byref(n0),
        ctypes.byref(secs),
    )
    if rc != 0:
        raise RuntimeError(f"aid_classify_pages failed with {rc}")
    return torch.frombuffer(memoryview(side), dtype=torch.uint8).clone(), n0.value, secs.value


# 7 MiB. The label sequence repeats at exactly this stride -- measured 100.00%
# over 1 GiB, and every allocation's map is a rotation of one pattern -- because
# the mod-7 stack hash runs over a 1 MiB interleave unit, so the cycle is
# 7 * 256 pages rather than the 7 the page-level rule suggests.
_AID_PERIOD_PAGES = 1792


def classify_fast(buf: torch.Tensor):
    """classify() by probing two periods instead of the whole buffer.

    Same return shape as classify(). Reads 14 MiB regardless of buffer size, so
    a 63 GiB weight buffer costs a few ms rather than ~14 s, and build_layout's
    sizing loop can retry without the retries dominating model load.

    The second period is checked against the first, so a buffer that is not
    periodic is reported rather than silently mislabelled; the caller falls
    back to the full classifier in that case.
    """
    lib = _load_lib()
    if not hasattr(lib, "aid_classify_fast"):
        return classify(buf)
    nbytes = buf.numel() * buf.element_size()
    npages = nbytes // _PAGE
    if npages < 2 * _AID_PERIOD_PAGES:
        return classify(buf)
    side = (ctypes.c_ubyte * npages)()
    n0 = ctypes.c_int(0)
    secs = ctypes.c_double(0.0)
    pm = ctypes.c_double(0.0)
    rc = lib.aid_classify_fast(
        ctypes.c_void_p(buf.data_ptr()),
        ctypes.c_size_t(nbytes),
        side,
        ctypes.byref(n0),
        ctypes.byref(secs),
        ctypes.byref(pm),
    )
    if rc == 2:
        return classify(buf)
    if rc != 0:
        raise RuntimeError(f"aid_classify_fast failed with {rc}")
    return (
        torch.frombuffer(memoryview(side), dtype=torch.uint8).clone(),
        n0.value,
        secs.value,
    )


def moe_slab_aid(num_experts: int, wgs_per_expert: int, frac_aid0: float):
    """Assign MoE weight slabs to AIDs so the split holds under any routing.

    The obvious assignment -- slabs in order, first 3/7 to AID0 -- puts whole
    experts on one AID. Routing picks top-k of 128 experts per token and the
    active set changes every step, so the AID split of *active* tiles would be
    43/57 only on average and would swing hard token to token, which is the
    load imbalance this whole scheme exists to avoid.

    Splitting on the workgroup index inside each expert instead makes every
    expert carry the same 3:4 proportion, so any subset of experts is still
    split 3:4.
    """
    # 46 workgroups cannot be split 3:7 exactly -- 46*3/7 is 19.71 -- and
    # rounding every expert the same way to 20 overshoots AID0's page supply.
    # Carrying the remainder across experts instead makes the cut alternate
    # 19/20 so the total lands exactly on the ratio.
    out = torch.empty(num_experts * wgs_per_expert, dtype=torch.uint8)
    wg = torch.arange(wgs_per_expert)
    prev = 0
    for e in range(num_experts):
        cur = round((e + 1) * wgs_per_expert * frac_aid0)
        cut = cur - prev
        prev = cur
        out[e * wgs_per_expert : (e + 1) * wgs_per_expert] = (wg >= cut).to(
            torch.uint8
        )
    return out


def pad_slabs_to_pages(packed: torch.Tensor, tile_bytes: int, n_tiles: int):
    """Re-block [nslabs, slab_bytes] so every tile starts on a 4 KiB page.

    The kernel resolves a chunk's page by indexing a register array, and a
    register array can only be indexed with a constant. Page-aligning the tiles
    is what makes that index constant: chunk c of a tile is then always page
    c/4 at byte offset (c%4)*1024, regardless of which tile it is.

    Without this the index depends on the tile number, the entry comes back as
    a per-chunk `flat_load_dword` + `s_waitcnt vmcnt(0)`, and the 23 serialized
    round trips in front of each tile cost far more than locality returns --
    measured 2.078 ms against a 1.837 ms baseline.

    Costs 4.3% more footprint on the gpt-oss-120b shape, in padding that is
    never read.
    """
    nslabs, slab_bytes = packed.shape
    data_bytes = tile_bytes * n_tiles
    scale_bytes = slab_bytes - data_bytes
    assert scale_bytes > 0, "slab is not data followed by scales"
    tile_pages = (tile_bytes + _PAGE - 1) // _PAGE
    scale_pages = (scale_bytes + _PAGE - 1) // _PAGE
    slab_pages = n_tiles * tile_pages + scale_pages

    out = torch.zeros(nslabs, slab_pages * _PAGE, dtype=torch.uint8,
                      device=packed.device)
    view = out.view(nslabs, slab_pages, _PAGE)
    tiles = view[:, : n_tiles * tile_pages].reshape(
        nslabs, n_tiles, tile_pages * _PAGE
    )
    tiles[:, :, :tile_bytes] = packed[:, :data_bytes].view(
        nslabs, n_tiles, tile_bytes
    )
    scales = view[:, n_tiles * tile_pages :].reshape(nslabs, scale_pages * _PAGE)
    scales[:, :scale_bytes] = packed[:, data_bytes:]
    return out


def place_xcd_striped(packed: torch.Tensor, tile_bytes: int, n_tiles: int,
                      n_xcds: int = 8, verbose: bool = True):
    """Place an XCD-striped [n_wgs, slab_bytes] tensor onto each XCD's AID.

    Rows are already partitioned `n_wgs % n_xcds == 0` with XCD i reading
    rows [i*wgs_x, (i+1)*wgs_x). XCDs 0-3 are AID0, 4-7 are AID1. Same
    self-describing buffer as build_layout, so the kernel finds the map at
    page 1 of the weight pointer -- which must be the *full* buffer, not a
    dim-0 slice (new_input map (-1,-1,-1), address via xcd_id).
    """
    packed = packed.reshape(-1, packed.shape[-1]).contiguous()
    n_wgs = packed.shape[0]
    assert n_wgs % n_xcds == 0, (n_wgs, n_xcds)
    wgs_x = n_wgs // n_xcds
    padded = pad_slabs_to_pages(packed, tile_bytes, n_tiles)
    sa = torch.zeros(n_wgs, dtype=torch.uint8)
    # 4 XCDs per AID on this part. AID1 owns more stacks, so XCDs 4-7 get the
    # larger pool; the 3:4 page split still comes from classify, not from this.
    sa[4 * wgs_x :] = 1
    return build_layout(padded, slab_aid=sa, per_expert=None, verbose=verbose)


def swap_remote(slab_aid: torch.Tensor, num_experts: int, wgs_per_expert: int,
                k: int, reader_cut: int = None):
    """Make k workgroups per expert read across the AID boundary.

    Swaps in pairs -- one workgroup off each AID -- so every AID keeps exactly
    the page count it had. The byte split therefore stays at the stack ratio
    and only the local/remote mix moves, which is what separates the effect of
    locality from the effect of load balance.

    Used to measure how decode time responds to the remote fraction. If the
    weight loads are pipelined behind the MFMA there is slack to absorb some
    latency, and the response is flat until that slack runs out.
    """
    sa = slab_aid.clone().reshape(num_experts, wgs_per_expert)
    if reader_cut is None:
        reader_cut = wgs_per_expert // 2
    if k <= 0:
        return sa.reshape(-1)
    for e in range(num_experts):
        row = sa[e]
        # Read by an AID0 XCD and placed on AID0, so currently local.
        a = [w for w in range(reader_cut) if row[w] == 0][:k]
        b = [w for w in range(reader_cut, wgs_per_expert) if row[w] == 1][:k]
        for w0, w1 in zip(a, b):
            row[w0] = 1
            row[w1] = 0
    return sa.reshape(-1)


def build_layout(packed: torch.Tensor, slab_aid: torch.Tensor = None,
                 per_expert=None, steer: bool = True, verbose: bool = True):
    """Place a packed weight tensor AID-locally.

    `packed` is [num_slabs, slab_bytes] uint8 (or anything reshapeable to it),
    matching the kernel's per-workgroup slab layout. Returns:

      dest       uint8, the self-describing placed-weight buffer:
                   page 0        header, word 0 = buffer size in bytes
                   pages 1..T    int32 table, slab page -> buffer page
                   pages T+1..   the slab pages, in placement order
      table      [num_slabs * pages_per_slab] int32, a device copy of the table
      slab_aid   [num_slabs] uint8, which AID each slab was placed on
      info       dict of timings and ratios

    Carrying the table inside the buffer is what keeps this a one-tensor
    substitution on the host: the kernel finds its own map from the weight
    pointer, so no task-graph input slot has to be added for it.

    Slabs are assigned to AIDs in proportion to each AID's page count, which is
    the stack ratio, so both sides finish together.
    """
    assert packed.is_cuda and packed.dtype == torch.uint8
    packed = packed.reshape(-1, packed.shape[-1]).contiguous()
    nslabs, slab_bytes = packed.shape
    pages_per_slab = (slab_bytes + _PAGE - 1) // _PAGE

    if steer is False:
        # Same buffer, same table, same indirection in the kernel, but pages
        # handed out in order so every slab keeps the natural 3:4 interleave.
        # Isolates what the indirection costs from what the steering buys.
        table_pages = (nslabs * pages_per_slab * 4 + _PAGE - 1) // _PAGE
        reserved = 1 + table_pages
        total_pages = reserved + nslabs * pages_per_slab
        dest = torch.empty(total_pages * _PAGE, dtype=torch.uint8,
                           device=packed.device)
        table = (
            torch.arange(nslabs * pages_per_slab, dtype=torch.int32) + reserved
        )
        table_dev = table.to(packed.device)
        dest[:4].copy_(
            torch.tensor([total_pages * _PAGE], dtype=torch.int32)
            .view(torch.uint8).flatten()
        )
        dest[_PAGE : _PAGE + table.numel() * 4].copy_(
            table.view(torch.uint8).flatten()
        )
        lib = _load_lib()
        rc = lib.aid_scatter_copy(
            ctypes.c_void_p(dest.data_ptr()),
            ctypes.c_void_p(packed.data_ptr()),
            ctypes.c_void_p(table_dev.data_ptr()),
            ctypes.c_int(nslabs),
            ctypes.c_int(slab_bytes),
            ctypes.c_int(pages_per_slab),
        )
        if rc != 0:
            raise RuntimeError(f"aid_scatter_copy failed with {rc}")
        if verbose:
            print("  AID layout: steering disabled, pages in buffer order")
        return dest, table_dev, torch.zeros(nslabs, dtype=torch.uint8), {
            "pages_per_slab": pages_per_slab,
            "bytes": total_pages * _PAGE,
        }

    # Slack slabs. Each AID's page pool has to round down to a whole number of
    # slabs, so an exactly-sized buffer loses a slab to fragmentation on both
    # sides at once. The split also wobbles by a page or two between runs, since
    # classification is a timing measurement.
    slack = max(8, nslabs // 256)
    table_pages = (nslabs * pages_per_slab * 4 + _PAGE - 1) // _PAGE
    reserved = 1 + table_pages
    data_pages = (nslabs + slack) * pages_per_slab

    # The tile map splits work evenly between the two AIDs, but the AIDs own 3
    # and 4 stacks, so an exactly-sized buffer only ever yields 43% of its
    # pages on AID0 and the even split will not fit. Size up until the scarcer
    # side has room. The surplus on the other side is never touched.
    want0 = None
    if slab_aid is not None:
        want0 = int((slab_aid.cpu().reshape(-1) == 0).sum())
    elif per_expert is None:
        want0 = 0  # proportional default always fits

    secs = 0.0
    for _ in range(4):
        total_pages = reserved + data_pages
        dest = torch.empty(
            total_pages * _PAGE, dtype=torch.uint8, device=packed.device
        )
        side, _, s = classify_fast(dest)
        secs += s
        # The header and table pages are read scalar-side once per workgroup,
        # so their own homing does not matter; exclude them from the pools.
        side[:reserved] = 2
        free0 = torch.nonzero(side == 0, as_tuple=False).flatten()
        free1 = torch.nonzero(side == 1, as_tuple=False).flatten()
        npages = int((side != 2).sum())
        if want0 is None:
            break
        need0, need1 = want0 * pages_per_slab, (nslabs - want0) * pages_per_slab
        if free0.numel() >= need0 and free1.numel() >= need1:
            break
        grow = max(
            need0 / max(free0.numel(), 1), need1 / max(free1.numel(), 1)
        )
        del dest, side, free0, free1
        data_pages = int(data_pages * grow * 1.02) + pages_per_slab

    cap0, cap1 = free0.numel() // pages_per_slab, free1.numel() // pages_per_slab
    if slab_aid is not None:
        slab_aid = slab_aid.cpu().to(torch.uint8).reshape(-1)
        assert slab_aid.numel() == nslabs
    elif per_expert is not None:
        # The ratio has to come from the measurement rather than a constant:
        # which stack is harvested is per-part, and the split wobbles a little
        # run to run because classification is a timing measurement.
        num_experts, wgs_per_expert = per_expert
        assert num_experts * wgs_per_expert == nslabs
        slab_aid = moe_slab_aid(
            num_experts, wgs_per_expert, free0.numel() / npages
        )
    else:
        n0 = min(int(nslabs * free0.numel() / npages), cap0)
        slab_aid = torch.zeros(nslabs, dtype=torch.uint8)
        slab_aid[n0:] = 1
    n0 = int((slab_aid == 0).sum())
    n1 = nslabs - n0
    if n0 > cap0 or n1 > cap1:
        raise RuntimeError(
            f"cannot place {n0}/{n1} slabs: capacity {cap0}/{cap1}"
        )

    # Pages are handed out in buffer order within each AID, so slabs that are
    # adjacent in the original tensor stay roughly adjacent in the new one.
    table = torch.empty(nslabs, pages_per_slab, dtype=torch.int32)
    which0 = torch.nonzero(slab_aid == 0, as_tuple=False).flatten()
    which1 = torch.nonzero(slab_aid == 1, as_tuple=False).flatten()
    table[which0] = (
        free0[: which0.numel() * pages_per_slab]
        .to(torch.int32)
        .reshape(-1, pages_per_slab)
    )
    table[which1] = (
        free1[: which1.numel() * pages_per_slab]
        .to(torch.int32)
        .reshape(-1, pages_per_slab)
    )
    table = table.reshape(-1)

    table_dev = table.to(packed.device)
    # Header and table go in before the weights: the scatter writes only data
    # pages, which the pools above excluded from.
    dest[: 4].copy_(
        torch.tensor(
            [total_pages * _PAGE], dtype=torch.int32
        ).view(torch.uint8).flatten()
    )
    dest[_PAGE : _PAGE + table.numel() * 4].copy_(
        table.view(torch.uint8).flatten()
    )
    lib = _load_lib()
    rc = lib.aid_scatter_copy(
        ctypes.c_void_p(dest.data_ptr()),
        ctypes.c_void_p(packed.data_ptr()),
        ctypes.c_void_p(table_dev.data_ptr()),
        ctypes.c_int(nslabs),
        ctypes.c_int(slab_bytes),
        ctypes.c_int(pages_per_slab),
    )
    if rc != 0:
        raise RuntimeError(f"aid_scatter_copy failed with {rc}")

    info = {
        "classify_s": secs,
        "pages": npages,
        "aid0_pages": int(free0.numel()),
        "aid1_pages": int(free1.numel()),
        "stack_ratio": (
            7.0 * free0.numel() / npages,
            7.0 * free1.numel() / npages,
        ),
        "slabs_aid0": n0,
        "slabs_aid1": n1,
        "pages_per_slab": pages_per_slab,
        "bytes": total_pages * _PAGE,
        "pad_overhead": total_pages * _PAGE / (nslabs * slab_bytes) - 1.0,
    }
    if verbose:
        r0, r1 = info["stack_ratio"]
        print(
            f"  AID layout: {npages} pages, {free0.numel()} on AID0 / "
            f"{free1.numel()} on AID1 (stacks {r0:.2f}:{r1:.2f}), "
            f"slabs {n0}/{n1}, classify {secs:.2f}s, "
            f"footprint +{100 * info['pad_overhead']:.1f}%"
        )
    return dest, table_dev, slab_aid, info


def build_layout_invariant(packed: torch.Tensor, num_experts: int,
                           wgs_per_expert: int, slab_aid: torch.Tensor = None,
                           periods_per_expert: int = 2, verbose: bool = True):
    """Place AID-locally with the page offsets identical for every expert.

    build_layout() puts each slab wherever the next free page of its AID
    happens to be, so every slab lands at a different phase of the 1792-page
    cycle and carries its own 49 offsets. The kernel then has to fetch those
    offsets from HBM on every invocation, which measured 1.67 us/layer -- more
    than the placement returns, so the whole scheme ran net negative.

    The offsets only differ because the phase does. Give every expert a stride
    that is a whole number of periods and every expert sees the same AID
    pattern, so a workgroup's 49 offsets are the same for all 128 experts and
    all 36 layers. `wg_idx` is fixed per workgroup in a persistent kernel, so
    the workgroup resolves them once at kernel start and decode-time lookup
    goes away entirely; addressing becomes

        page = expert_base(expert_id) + off[j]

    with expert_base a multiply-add and off[j] a constant.

    Costs footprint: 46 slabs need 2254 pages and the stride rounds up to
    2*1792, so ~59% of the tensor is padding that is never read. Returns the
    same tuple as build_layout(), plus `rel_off` -- the [wgs, pages_per_slab]
    offsets the kernel resolves once.
    """
    assert packed.is_cuda and packed.dtype == torch.uint8
    packed = packed.reshape(-1, packed.shape[-1]).contiguous()
    nslabs, slab_bytes = packed.shape
    assert nslabs == num_experts * wgs_per_expert
    pages_per_slab = (slab_bytes + _PAGE - 1) // _PAGE

    stride = periods_per_expert * _AID_PERIOD_PAGES
    need = wgs_per_expert * pages_per_slab
    if need > stride:
        raise RuntimeError(
            f"expert needs {need} pages but stride is only {stride}; "
            f"raise periods_per_expert"
        )

    table_pages = (nslabs * pages_per_slab * 4 + _PAGE - 1) // _PAGE
    # Start the expert regions on a period boundary so expert 0 is not a
    # special case; everything after it is then a pure multiple of the period.
    reserved = 1 + table_pages
    reserved = ((reserved + _AID_PERIOD_PAGES - 1) // _AID_PERIOD_PAGES
                * _AID_PERIOD_PAGES)
    total_pages = reserved + num_experts * stride

    dest = torch.empty(total_pages * _PAGE, dtype=torch.uint8,
                       device=packed.device)
    side, _, secs = classify_fast(dest)

    if slab_aid is None:
        n0 = int((side[reserved:reserved + stride] == 0).sum())
        slab_aid = moe_slab_aid(num_experts, wgs_per_expert, n0 / stride)
    slab_aid = slab_aid.cpu().to(torch.uint8).reshape(num_experts, wgs_per_expert)

    # Lay out one expert region; periodicity makes it valid for all of them.
    # Both AIDs draw from the same window and use disjoint pages, so the two
    # sets interleave instead of each needing their own space.
    win = side[reserved:reserved + stride]
    free = {0: torch.nonzero(win == 0, as_tuple=False).flatten().tolist(),
            1: torch.nonzero(win == 1, as_tuple=False).flatten().tolist()}
    take = {0: 0, 1: 0}
    # Every expert must use the same offsets, so a workgroup's AID has to be
    # the same for every expert. Where moe_slab_aid alternates the 19/20 cut
    # across experts to hit the ratio exactly, settle it by majority.
    wg_aid = (slab_aid.float().mean(0) >= 0.5).to(torch.uint8)
    rel_off = torch.empty(wgs_per_expert, pages_per_slab, dtype=torch.int32)
    for wg in range(wgs_per_expert):
        a = int(wg_aid[wg])
        pool = free[a]
        if take[a] + pages_per_slab > len(pool):
            raise RuntimeError(
                f"AID{a} exhausted at wg {wg}: {len(pool)} pages for "
                f"{take[a] + pages_per_slab} needed"
            )
        rel_off[wg] = torch.tensor(pool[take[a]:take[a] + pages_per_slab],
                                   dtype=torch.int32)
        take[a] += pages_per_slab

    # Same offsets, shifted by the expert stride.
    e_base = (reserved + torch.arange(num_experts, dtype=torch.int32) * stride)
    table = (e_base.view(-1, 1, 1) + rel_off.view(1, wgs_per_expert, -1))
    table = table.reshape(-1).contiguous()

    # The periodicity claim is load-bearing: if it were wrong every expert but
    # the first would read the wrong AID and the placement would be silently
    # useless, so check the pages really are where they are meant to be.
    got = side[table.long()].reshape(num_experts, wgs_per_expert, pages_per_slab)
    want = wg_aid.view(1, -1, 1).expand_as(got)
    frac_local = float((got == want).float().mean())

    table_dev = table.to(packed.device)
    # Header word 0 is the buffer size, as in build_layout. Word 1 carries the
    # expert stride so the kernel can reach expert e's pages from the single
    # expert-invariant row of offsets rather than reading a row per expert.
    dest[:8].copy_(
        torch.tensor([total_pages * _PAGE, stride], dtype=torch.int32)
        .view(torch.uint8).flatten()
    )
    dest[_PAGE:_PAGE + table.numel() * 4].copy_(
        table.view(torch.uint8).flatten()
    )
    lib = _load_lib()
    rc = lib.aid_scatter_copy(
        ctypes.c_void_p(dest.data_ptr()),
        ctypes.c_void_p(packed.data_ptr()),
        ctypes.c_void_p(table_dev.data_ptr()),
        ctypes.c_int(nslabs),
        ctypes.c_int(slab_bytes),
        ctypes.c_int(pages_per_slab),
    )
    if rc != 0:
        raise RuntimeError(f"aid_scatter_copy failed with {rc}")

    info = {
        "classify_s": secs,
        "pages_per_slab": pages_per_slab,
        "bytes": total_pages * _PAGE,
        "expert_stride_pages": stride,
        "reserved_pages": reserved,
        "frac_local": frac_local,
        "slabs_aid0": int((wg_aid == 0).sum()) * num_experts,
        "slabs_aid1": int((wg_aid == 1).sum()) * num_experts,
        "pad_overhead": total_pages * _PAGE / (nslabs * slab_bytes) - 1.0,
    }
    if verbose:
        print(
            f"  AID layout (phase-invariant): stride {stride} pages/expert, "
            f"wg split {int((wg_aid == 0).sum())}/{int((wg_aid == 1).sum())}, "
            f"pages on intended AID {100 * frac_local:.2f}%, "
            f"classify {secs:.2f}s, footprint +{100 * info['pad_overhead']:.1f}%"
        )
    return dest, table_dev, wg_aid, rel_off, info


def verify(dest: torch.Tensor, table: torch.Tensor, slab_aid: torch.Tensor,
           pages_per_slab: int) -> float:
    """Re-classify and report the fraction of slab pages on their intended AID.

    Destroys `dest`, so this is a debugging aid to run on a throwaway copy.
    """
    side, _, _ = classify(dest)
    tbl = table.cpu().reshape(-1, pages_per_slab)
    want = slab_aid.cpu().reshape(-1, 1).expand_as(tbl)
    got = side[tbl.reshape(-1).long()].reshape(tbl.shape)
    return float((got == want).float().mean())
