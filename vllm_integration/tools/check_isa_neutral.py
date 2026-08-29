#!/usr/bin/env python3
"""Assert two builds of the same kernel differ only in NAMES, not in code.

The right gate for any change that is supposed to be semantically empty: a
rename, a comment sweep, a reformat, a header-path flip. Those all claim "this
cannot change the machine code" -- this proves it instead of asserting it.

Why the ordinary gates are not enough. check_roundtrip.py compares emitted
SOURCE, so it is blind to anything the compiler does downstream. The decode run
compares TEXT, and GPT-OSS MoE is legitimately non-deterministic (float
atomicAdd over experts, see check_determinism.py), so a sub-1% latency or token
delta cannot be resolved by running it more times. The disassembly is the only
artifact that is both deterministic and downstream of the compiler.

Usage:
    ./check_isa_neutral.py --old OLDNAME --new NEWNAME <reference> <candidate>

<reference> and <candidate> are each a .so or a .s. A .so is unbundled to its
gfx950 code object and disassembled; a .s is used as-is, so a reference dump
taken before the change survives after its .so has been overwritten by the
rebuild. Capture one with:

    llvm-objcopy --dump-section=.hip_fatbin=fat.bin <so> /dev/null
    clang-offload-bundler --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx950 \\
        --input=fat.bin --output=co.o --unbundle
    llvm-objdump -d --mcpu=gfx950 co.o > isa_before.s

WHAT LEGITIMATELY DIFFERS AFTER A RENAME
----------------------------------------
Four categories, and nothing else:

  1. the objdump filename line ("co.o:  file format elf64-amdgpu")
  2. mangled symbols, and the branch annotations that cite them
  3. internal-linkage symbol hashes (.intern.<hex>) -- these are a digest OF
     the source identifiers, so they MUST change; an unchanged hash after a
     real rename would be the surprising result
  4. string-literal LENGTH immediates. A renamed printf format is a different
     number of bytes, and that length is an immediate operand:
       s_mov_b64 s[10:11], 0x42   ->   0x45
     This script only accepts such a delta when it equals len(new)-len(old)
     exactly. An immediate that moves by any other amount is not a string
     length, and needs its own explanation before you believe it.

The real gate is the last line of the report: the OPCODE SEQUENCE. Mnemonics
are stripped of operands and encodings and compared positionally. If that
matches and the line counts match, no instruction was added, removed, or
reordered, whatever the symbol text happens to say.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

ROCM_BIN = "/opt/rocm/llvm/bin"
TARGET = "hipv4-amdgcn-amd-amdhsa--gfx950"


def tool(name):
    p = os.path.join(ROCM_BIN, name)
    return p if os.path.exists(p) else name


def disassemble(so_path, workdir):
    """Extract the fatbin, unbundle the gfx950 code object, disassemble."""
    os.makedirs(workdir, exist_ok=True)
    fat = os.path.join(workdir, "fat.bin")
    obj = os.path.join(workdir, "co.o")
    subprocess.run([tool("llvm-objcopy"), "--dump-section=.hip_fatbin=" + fat,
                    so_path, os.devnull], check=True)
    subprocess.run([tool("clang-offload-bundler"), "--type=o",
                    "--targets=" + TARGET, "--input=" + fat,
                    "--output=" + obj, "--unbundle"], check=True)
    return subprocess.run([tool("llvm-objdump"), "-d", "--mcpu=gfx950", obj],
                          check=True, capture_output=True,
                          text=True).stdout.split("\n")


def load(path, workdir, tag):
    if path.endswith(".s"):
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read().split("\n")
    return disassemble(path, os.path.join(workdir, tag))


def opcodes(lines):
    """Mnemonics only -- no operands, no encodings, no symbol annotations."""
    out = []
    for line in lines:
        m = re.match(r"\t(\S+)", line)
        if m:
            out.append(m.group(1))
    return out


def classify(a, b, old, new):
    if "file format" in a:
        return "objdump filename line"
    if ".intern." in a:
        return "internal-linkage symbol hash"
    # Itanium mangling spells a namespace as <length><name>, so the renamed
    # symbol is matched by the name alone appearing in either side.
    if re.search(r"_ZN\d+(%s|%s)" % (re.escape(old), re.escape(new)), a + b):
        return "mangled symbol / branch annotation"
    ia = re.findall(r"0x[0-9a-f]+", a)
    ib = re.findall(r"0x[0-9a-f]+", b)
    if len(ia) == len(ib):
        deltas = [int(y, 16) - int(x, 16) for x, y in zip(ia, ib)
                  if int(x, 16) != int(y, 16)]
        if deltas and all(d == len(new) - len(old) for d in deltas):
            return ("string-literal length (%+d = len(%s)-len(%s))"
                    % (len(new) - len(old), new, old))
    return "UNEXPLAINED"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True,
                    help="identifier being renamed away from")
    ap.add_argument("--new", required=True, help="identifier being renamed to")
    ap.add_argument("reference", help=".so or .s captured before the change")
    ap.add_argument("candidate", help=".so or .s built after the change")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        ref = load(args.reference, td, "ref")
        cand = load(args.candidate, td, "cand")

    ok = True
    if len(ref) != len(cand):
        print("FAIL: line count differs (%d vs %d) -- instructions were "
              "added or removed" % (len(ref), len(cand)))
        ok = False

    cats, unexplained = {}, []
    for i, (a, b) in enumerate(zip(ref, cand), 1):
        if a == b:
            continue
        c = classify(a, b, args.old, args.new)
        cats[c] = cats.get(c, 0) + 1
        if c == "UNEXPLAINED":
            unexplained.append((i, a, b))

    print("reference: %s (%d lines)" % (args.reference, len(ref)))
    print("candidate: %s (%d lines)" % (args.candidate, len(cand)))
    print("\ndiffering lines by category:")
    for k, v in sorted(cats.items(), key=lambda kv: -kv[1]):
        print("  %4d  %s" % (v, k))
    if not cats:
        print("  (none -- disassembly is identical)")

    same_ops = opcodes(ref) == opcodes(cand)
    print("\nopcode sequence identical: %s" % same_ops)
    if not same_ops:
        ok = False

    if unexplained:
        ok = False
        print("\n%d UNEXPLAINED line(s):" % len(unexplained))
        for i, a, b in unexplained[:40]:
            print("  line %d\n    - %s\n    + %s" % (i, a, b))

    print("\n%s" % ("PASS (ISA-neutral: differences are name-derived only)"
                   if ok else
                   "FAIL (a real instruction delta -- find it before shipping)"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
