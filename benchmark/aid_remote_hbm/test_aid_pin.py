"""Validate libaid_pin against one layer of gpt-oss-120b MoE weights.

Checks the three things that have to hold before this is worth wiring into the
demo: the setup cost is what aid_bulk_pin predicted, the pages actually land on
the side the layout asked for, and the bytes survive the round trip.

  hipcc -O3 -std=c++17 --offload-arch=gfx950 -Wno-unused-result -fPIC -shared \
    benchmark/aid_remote_hbm/aid_pin.hip -o /tmp/libaid_pin.so
  HIP_VISIBLE_DEVICES=1 python3 benchmark/aid_remote_hbm/test_aid_pin.py
"""

import ctypes
import os
import sys

import torch

LIB = os.environ.get("AID_PIN_LIB", "/tmp/libaid_pin.so")

# gpt-oss-120b, W13_OPW=128 / W2_OPW=64 after padding to 2944.
W13 = dict(name="gate_up", n_experts=128, n_wgs=46, wg_bytes=200192)
W2 = dict(name="down", n_experts=128, n_wgs=46, wg_bytes=100096)


def main():
    lib = ctypes.CDLL(LIB)
    lib.aid_pin_alloc.restype = ctypes.c_void_p
    lib.aid_pin_alloc.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    hip = ctypes.CDLL("libamdhip64.so")

    torch.cuda.init()
    free, total = torch.cuda.mem_get_info()
    print(f"device free {free/2**30:.1f} GiB / {total/2**30:.1f} GiB\n")

    which = sys.argv[1] if len(sys.argv) > 1 else "down"
    spec = W13 if which == "gate_up" else W2
    nbytes = spec["n_experts"] * spec["n_wgs"] * spec["wg_bytes"]
    half = spec["n_wgs"] // 2
    print(f"{spec['name']}: [{spec['n_experts']}, {spec['n_wgs']}, "
          f"{spec['wg_bytes']}] = {nbytes/2**20:.0f} MiB, "
          f"half_wgs={half} (wg<{half} -> AID0)")
    print(f"wg_bytes % 4096 = {spec['wg_bytes'] % 4096} "
          f"({'page-aligned' if spec['wg_bytes'] % 4096 == 0 else 'NOT page-aligned'})\n")

    # Deterministic pattern so the round trip is checkable.
    src = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device="cuda")
    ref = src[::1048573].clone().cpu()  # sparse sample, prime stride

    setup = ctypes.c_double(0.0)
    frac = ctypes.c_double(0.0)
    invert = int(os.environ.get("AID_PIN_INVERT", "0"))
    ptr = lib.aid_pin_alloc(
        ctypes.c_void_p(src.data_ptr()), nbytes, spec["wg_bytes"],
        spec["n_wgs"], half, invert, ctypes.byref(setup), ctypes.byref(frac))
    if not ptr:
        print("FAILED: aid_pin_alloc returned NULL")
        return 1

    print(f"\nsetup      : {setup.value:.1f} s "
          f"({setup.value*1e6/(nbytes/4096):.1f} us/page)")
    print(f"verified   : {frac.value*100:.1f}% of pages on the intended side")
    print(f"36 layers  : {setup.value * 36 * (1178730496+589365248)/nbytes/60:.1f} min "
          f"(both tensors, extrapolated)")

    # Round trip: copy the pinned buffer back and compare the sparse sample.
    out = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    rc = hip.hipMemcpy(ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(ptr),
                       ctypes.c_size_t(nbytes), ctypes.c_int(3))  # D2D
    hip.hipDeviceSynchronize()
    got = out[::1048573].cpu()
    ok = torch.equal(ref, got)
    print(f"round trip : {'OK' if ok else 'CORRUPT'} "
          f"(hipMemcpy rc={rc}, {ref.numel()} sampled bytes)")

    lib.aid_pin_free_all()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
