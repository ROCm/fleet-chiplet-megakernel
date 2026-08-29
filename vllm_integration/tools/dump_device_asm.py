"""Disassemble the GPU code inside a fleet_mk .so.

`llvm-objdump` on the .so itself yields only x86 host code -- the device image
is a clang offload bundle in the `.hip_fatbin` section, and under `-fgpu-rdc`
(which fleet_mk uses) that bundle holds a linked ELF at a 4096-byte alignment
offset rather than at byte 0. Disassembling the bundle directly reports zero
instructions, which reads as "the kernel vanished" rather than "wrong input".

Used to prove a refactor is byte-identical: dump before, dump after, diff.

    python3 tools/dump_device_asm.py generated/gpt_oss_120b.so /tmp/asm_base.s

Prints an instruction-mix summary so an accidental codegen change is visible
even when the diff is too large to read.

With two .so arguments it diffs them by FULL mnemonic histogram:

    python3 tools/dump_device_asm.py --diff a.so b.so

A raw `diff` of the two dumps is useless for this: every instruction after the
first inserted one shifts address, so a 4-instruction change reports thousands of
differing lines. The histogram is shift-invariant. It also covers every mnemonic
rather than the eight CLASSES below, so a change that swaps, say, a v_lshl_add
for a v_mul cannot hide in an unlisted category.

Two per-symbol modes, because a whole-binary count once gave the OPPOSITE
answer to the truth (a dead function grew 1 -> 966 lines and swamped the
totals). Attribute to a symbol before concluding anything:

    python3 tools/dump_device_asm.py --symbols /tmp/asm.s [substr]
    python3 tools/dump_device_asm.py --diff-fn /tmp/a.s /tmp/b.s <substr>

`--symbols` lists instruction and byte counts per function, largest first.
`--diff-fn` compares the mnemonic STREAMS of matching symbols across two
dumps and reports the first differing index -- it answers "is this the same
code?", which equal counts cannot.
"""

import re
import subprocess
import sys
from pathlib import Path

OBJCOPY = "/opt/rocm/llvm/bin/llvm-objcopy"
OBJDUMP = "/opt/rocm/llvm/bin/llvm-objdump"
MCPU = "gfx950"

#: Counted per dump. MFMA is the math; the buffer/LDS ops are what a
#: stride-vs-reduction split would perturb if it were not neutral; s_nop is
#: hazard padding and moves when scheduling changes.
CLASSES = ("v_mfma", "buffer_load", "ds_read", "ds_write", "s_waitcnt",
           "s_nop", "global_load", "v_accvgpr")


def dump(so_path, out_path):
    so_path, out_path = Path(so_path), Path(out_path)
    fatbin = out_path.with_suffix(".fatbin.tmp")
    elf = out_path.with_suffix(".elf.tmp")

    subprocess.run(
        [OBJCOPY, f"--dump-section=.hip_fatbin={fatbin}", str(so_path),
         "/dev/null"], check=True, capture_output=True)

    blob = fatbin.read_bytes()
    # The bundle header is followed by the device ELF. Locate it rather than
    # hardcoding 4096: the offset is an alignment artifact, not a contract.
    marks = [m.start() for m in re.finditer(b"\x7fELF", blob)]
    if not marks:
        raise SystemExit(f"no device ELF inside {so_path}'s .hip_fatbin "
                         f"({len(blob)} bytes) -- offload bundle format changed")
    elf.write_bytes(blob[marks[0]:])

    asm = subprocess.run([OBJDUMP, "-d", f"--mcpu={MCPU}", str(elf)],
                         check=True, capture_output=True, text=True).stdout
    out_path.write_text(asm)
    fatbin.unlink()
    elf.unlink()

    counts = {c: len(re.findall(rf"\b{c}", asm)) for c in CLASSES}
    # Instruction lines are "\t<mnemonic> ...  // <addr>: <encoding>" -- the
    # address is a TRAILING comment, not a prefix, so an anchored address match
    # counts zero and reports an empty kernel.
    total = sum(1 for ln in asm.splitlines()
                if ln.startswith("\t") and "//" in ln)
    print(f"{so_path.name} -> {out_path}  ({total} instructions)")
    for c, n in counts.items():
        print(f"  {c:<14} {n}")
    return counts


def histogram(asm_path):
    """Count every mnemonic in a dump. Shift-invariant, unlike a line diff."""
    hist = {}
    for ln in Path(asm_path).read_text().splitlines():
        if not ln.startswith("\t") or "//" not in ln:
            continue
        mnemonic = ln.strip().split()[0]
        hist[mnemonic] = hist.get(mnemonic, 0) + 1
    return hist


def functions(asm_path):
    """Split a dump into {symbol: [instruction lines]}.

    Whole-binary counts can give the OPPOSITE answer to the truth: a dead
    function that grew 1 -> 966 lines once dominated the totals and reversed
    the sign of a real per-function improvement. Always attribute to a symbol
    before concluding anything from an instruction count.
    """
    out, cur = {}, None
    for ln in Path(asm_path).read_text().splitlines():
        m = re.match(r"^[0-9a-f]+ <(.+)>:$", ln)
        if m:
            cur = m.group(1)
            out.setdefault(cur, [])
        elif cur is not None and ln.startswith("\t") and "//" in ln:
            out[cur].append(ln.strip())
    return out


def symbols(asm_path, substr=None):
    """Print per-symbol instruction and byte counts, largest first.

    Byte size comes from the trailing `// <addr>: <encoding>` comment rather
    than from the ELF symbol table, so it reflects what was actually
    disassembled.
    """
    fns = functions(asm_path)
    rows = []
    for name, lines in fns.items():
        if substr and substr not in name:
            continue
        addrs = []
        for ln in lines:
            m = re.search(r"//\s*([0-9A-Fa-f]+):\s*([0-9A-Fa-f ]+)$", ln)
            if m:
                addrs.append((int(m.group(1), 16),
                              len(m.group(2).replace(" ", "")) // 2))
        nbytes = (max(a + w for a, w in addrs) - min(a for a, _ in addrs)
                  if addrs else 0)
        rows.append((len(lines), nbytes, name))
    rows.sort(reverse=True)
    print(f"{'insns':>7} {'bytes':>8}  symbol")
    for n, nb, name in rows:
        print(f"{n:>7} {nb:>8}  {name}")
    return 0


def diff_fn(asm_a, asm_b, substr):
    """Compare the mnemonic streams of matching symbols in two dumps.

    Answers the question a histogram cannot: are two bodies the SAME code, or
    merely the same size? Reports the first differing index, which localizes a
    codegen divergence that a count-only comparison reports as identical.
    """
    fa = {k: v for k, v in functions(asm_a).items() if substr in k}
    fb = {k: v for k, v in functions(asm_b).items() if substr in k}
    print(f"A: {Path(asm_a).name}  {len(fa)} matching symbols")
    print(f"B: {Path(asm_b).name}  {len(fb)} matching symbols")
    for name in sorted(set(fa) | set(fb)):
        la, lb = fa.get(name, []), fb.get(name, [])
        ma = [x.split()[0] for x in la]
        mb = [x.split()[0] for x in lb]
        if ma == mb:
            print(f"  SAME  {len(ma):>6} insns  {name[:110]}")
            continue
        first = next((i for i, (x, y) in enumerate(zip(ma, mb)) if x != y),
                     min(len(ma), len(mb)))
        print(f"  DIFF  {len(ma):>6} vs {len(mb):<6} first@{first}  "
              f"{name[:100]}")
    return 0


def diff(so_a, so_b):
    a = histogram(_dump_quiet(so_a, "/tmp/_asm_a.s"))
    b = histogram(_dump_quiet(so_b, "/tmp/_asm_b.s"))
    keys = sorted(set(a) | set(b))
    moved = [(k, a.get(k, 0), b.get(k, 0)) for k in keys
             if a.get(k, 0) != b.get(k, 0)]
    print(f"A: {Path(so_a).name}  {sum(a.values())} instructions, {len(a)} mnemonics")
    print(f"B: {Path(so_b).name}  {sum(b.values())} instructions, {len(b)} mnemonics")
    if not moved:
        print("IDENTICAL instruction mix")
        return 0
    print(f"{'mnemonic':<28} {'A':>6} {'B':>6} {'delta':>7}")
    for k, na, nb in moved:
        print(f"{k:<28} {na:>6} {nb:>6} {nb - na:>+7}")
    print(f"net {sum(b.values()) - sum(a.values()):+d} instructions "
          f"across {len(moved)} mnemonics")
    return 0


def _dump_quiet(so_path, out_path):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        dump(so_path, out_path)
    return out_path


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--diff":
        sys.exit(diff(sys.argv[2], sys.argv[3]))
    if sys.argv[1:2] == ["--symbols"] and len(sys.argv) in (3, 4):
        sys.exit(symbols(sys.argv[2], sys.argv[3] if len(sys.argv) == 4 else None))
    if sys.argv[1:2] == ["--diff-fn"] and len(sys.argv) == 5:
        sys.exit(diff_fn(sys.argv[2], sys.argv[3], sys.argv[4]))
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    dump(sys.argv[1], sys.argv[2])
