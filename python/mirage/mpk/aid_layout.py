"""AID-local placement for MoE expert weights on MI350/MI355.

MI350 in NPS1 interleaves memory over 7 HBM stacks (one is harvested; each is
36 GB, and 7*36 = 252 GB matches the visible capacity). The owning stack is a
mod-7 hash of the *physical* 4 KiB page number, and the two AIDs own 3 and 4 of
those stacks. Three consequences shape everything here:

* A contiguous weight slab is never AID-local. The kernel reads 200192 B per
  workgroup, 49 pages, and the longest single-AID run is 6 pages -- measured 0
  of 10699 slabs on one AID. So no tile map or reader-steering scheme can help;
  the bytes have to be placed.
* There is no compile-time map. Homing follows the physical page, so two
  allocations agree only ~71% at the same offset. Classification has to run at
  load, every load, costing ~14 s for 63 GiB.
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


def build_layout(packed: torch.Tensor, slab_aid: torch.Tensor = None,
                 verbose: bool = True):
    """Place a packed weight tensor AID-locally.

    `packed` is [num_slabs, slab_bytes] uint8 (or anything reshapeable to it),
    matching the kernel's per-workgroup slab layout. Returns:

      dest       [num_slabs * pages_per_slab * 4096] uint8, the placed weights
      table      [num_slabs * pages_per_slab] int32, slab page -> buffer page
      slab_aid   [num_slabs] uint8, which AID each slab was placed on
      info       dict of timings and ratios

    Slabs are assigned to AIDs in proportion to each AID's page count, which is
    the stack ratio, so both sides finish together.
    """
    assert packed.is_cuda and packed.dtype == torch.uint8
    packed = packed.reshape(-1, packed.shape[-1]).contiguous()
    nslabs, slab_bytes = packed.shape
    pages_per_slab = (slab_bytes + _PAGE - 1) // _PAGE

    # Slack slabs. Each AID's page pool has to round down to a whole number of
    # slabs, so an exactly-sized buffer loses a slab to fragmentation on both
    # sides at once. The split also wobbles by a page or two between runs, since
    # classification is a timing measurement.
    slack = max(8, nslabs // 256)
    dest = torch.empty(
        (nslabs + slack) * pages_per_slab * _PAGE,
        dtype=torch.uint8,
        device=packed.device,
    )
    side, n_aid0, secs = classify(dest)

    free0 = torch.nonzero(side == 0, as_tuple=False).flatten()
    free1 = torch.nonzero(side == 1, as_tuple=False).flatten()
    npages = side.numel()

    cap0, cap1 = free0.numel() // pages_per_slab, free1.numel() // pages_per_slab
    if slab_aid is None:
        n0 = min(int(nslabs * free0.numel() / npages), cap0)
        slab_aid = torch.zeros(nslabs, dtype=torch.uint8)
        slab_aid[n0:] = 1
    else:
        slab_aid = slab_aid.cpu().to(torch.uint8).reshape(-1)
        assert slab_aid.numel() == nslabs
    n0 = int((slab_aid == 0).sum())
    n1 = nslabs - n0
    if n0 > cap0 or n1 > cap1:
        raise RuntimeError(
            f"cannot place {n0}/{n1} slabs: capacity {cap0}/{cap1}"
        )

    # Pages are handed out in buffer order within each AID, so slabs that are
    # adjacent in the original tensor stay roughly adjacent in the new one.
    table = torch.empty(nslabs * pages_per_slab, dtype=torch.int32)
    take0 = take1 = 0
    order0 = free0.to(torch.int32)
    order1 = free1.to(torch.int32)
    for s in range(nslabs):
        lo = s * pages_per_slab
        if slab_aid[s] == 0:
            table[lo : lo + pages_per_slab] = order0[take0 : take0 + pages_per_slab]
            take0 += pages_per_slab
        else:
            table[lo : lo + pages_per_slab] = order1[take1 : take1 + pages_per_slab]
            take1 += pages_per_slab

    table_dev = table.to(packed.device)
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
        "pad_overhead": (nslabs + slack)
        * pages_per_slab
        * _PAGE
        / (nslabs * slab_bytes)
        - 1.0,
    }
    if verbose:
        r0, r1 = info["stack_ratio"]
        print(
            f"  AID layout: {npages} pages, {free0.numel()} on AID0 / "
            f"{free1.numel()} on AID1 (stacks {r0:.2f}:{r1:.2f}), "
            f"slabs {n0}/{n1}, classify {secs:.2f}s, "
            f"pad +{100 * info['pad_overhead']:.2f}%"
        )
    return dest, table_dev, slab_aid, info


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
