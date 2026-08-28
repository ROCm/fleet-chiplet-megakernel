#!/usr/bin/env python3
"""Identify which MoE knob combination a built .so was compiled with.

The three MPK_MOE_* knobs are invisible from the outside -- the banner titan
prints comes from the *env*, not the binary, so a .so and its env knobs can
disagree silently (that mismatch is the whole reason moe_layout.py prints the
"<-- .so MUST be built" line). This reads the answer out of the device code
instead of trusting a filename.

The tell is the pair of buffer-descriptor extents the kernel materializes for
the W13 and W2 expert slabs -- `s_mov_b32 sN, 0x...` immediately feeding
make_w_buffer_rsrc. Both are pure functions of the knobs:

    W13_EXPERT_BYTES = (W13_N_STRIDE / W13_OPW) * W13_OPW * (W13_K_STRIDE/2)
    W2_EXPERT_BYTES  = (W2_N_STRIDE  / W2_OPW)  * W2_OPW  * (W2_K_STRIDE/2)

so each (K_STRIDE, N_STRIDE) pair has a distinct signature. SPLIT_SCALES does
not move the data extent (it only removes the interleaved scale bytes from
WG_BYTES), so it shows up as a *third* distinct value rather than a flag, and
is reported when it can be told apart.

Usage:
    python3 tools/id_moe_so.py generated/gpt_oss_120b.so /tmp/*.so.*
"""
import argparse
import re
import subprocess
import sys
import tempfile
import os

OBJDUMP = "/opt/rocm/llvm/bin/llvm-objdump"

# Model geometry -- must match the kernel's compiled constants.
W13_OPW, W2_OPW = 128, 64
W13_OUT, W2_OUT = 5888, 2944
K_REDUCE = 2944
NUM_BLK32 = K_REDUCE // 32


def extents(k_stride, n_stride_w13, n_stride_w2, split):
    """(W13_EXPERT_BYTES, W2_EXPERT_BYTES) for one knob combination."""
    def one(opw, n_stride):
        wg_data = opw * (k_stride // 2)
        wg_bytes = wg_data if split else wg_data + opw * NUM_BLK32
        return (n_stride // opw) * wg_bytes
    return one(W13_OPW, n_stride_w13), one(W2_OPW, n_stride_w2)


def combos():
    """Every knob combination titan can currently be built with."""
    out = {}
    for k in (2944, 3072):
        for n in (0, 3072):
            for split in (False, True):
                n13 = 2 * n if n else W13_OUT
                n2 = n if n else W2_OUT
                out[extents(k, n13, n2, split)] = (
                    f"K_STRIDE={k or 'default'} "
                    f"N_STRIDE={n or 'default'} "
                    f"SPLIT_SCALES={int(split)}")
    return out


def device_consts(so_path):
    """Every `s_mov_b32 sN, 0x...` immediate in the device code, as a set."""
    with tempfile.TemporaryDirectory() as td:
        # The device ELF lives inside the .hip_fatbin section of the host .so.
        hdr = subprocess.run([OBJDUMP, "-h", so_path],
                             capture_output=True, text=True).stdout
        off = size = None
        for line in hdr.splitlines():
            f = line.split()
            if len(f) >= 4 and f[1] == ".hip_fatbin":
                size, off = int(f[2], 16), int(f[3], 16)
                break
        if off is None:
            return None
        with open(so_path, "rb") as fh:
            fh.seek(off)
            blob = fh.read(size)
        # Fatbin wraps one or more ELFs; take the first ELF magic onwards.
        i = blob.find(b"\x7fELF")
        if i < 0:
            return None
        dev = os.path.join(td, "dev.elf")
        with open(dev, "wb") as fh:
            fh.write(blob[i:])
        dis = subprocess.run(
            [OBJDUMP, "-d", "--triple=amdgcn-amd-amdhsa", "--mcpu=gfx950", dev],
            capture_output=True, text=True).stdout
        consts = set(int(m, 16) for m in
                     re.findall(r"s_mov_b32 s\d+, (0x[0-9a-f]+)", dis))
        # Count direct-to-LDS buffer loads. These are NOT unique to
        # MPK_W13_LDS_PREFETCH -- the QKV/OProj/W2 prefetch paths emit a fixed
        # 57 of them unconditionally. Turning W13 prefetch on adds its own 48,
        # for 105. So the flag is read as a COUNT, not a presence test:
        # 57 = off, 105 = on. Checking presence alone reports every build as
        # "on", which is how a prefetch-off build got mistaken for one twice.
        n_lds = len(re.findall(r"buffer_load_dwordx4 [^\n]*\blds\b", dis))
        return consts, n_lds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("so", nargs="+")
    a = ap.parse_args()

    table = combos()
    print(f"{'file':<44}  {'W13 extent':>12}  {'W2 extent':>11}  "
          f"{'prefetch':>8}  knobs")
    print("-" * 122)
    for so in a.so:
        got = device_consts(so)
        if got is None:
            print(f"{os.path.basename(so):<44}  {'-- no device code --':>26}")
            continue
        consts, n_lds = got
        pf = {105: "ON", 57: "OFF"}.get(n_lds, f"?({n_lds})")
        hits = [(w13, w2, desc) for (w13, w2), desc in table.items()
                if w13 in consts and w2 in consts]
        if not hits:
            print(f"{os.path.basename(so):<44}  "
                  f"{'-- no MoE extent pair matched --':>26}  {pf:>8}")
            continue
        for w13, w2, desc in hits:
            print(f"{os.path.basename(so):<44}  {w13:>12,}  {w2:>11,}  "
                  f"{pf:>8}  {desc}")


if __name__ == "__main__":
    main()
