#!/usr/bin/env python3
"""Gated literal substitution inside a backwards-built emitter.

Step 4/5 of the generator re-convergence work: the hand-written artifact was
pasted into an f-string verbatim, and is now parameterized one literal at a
time. Each substitution must leave the emitter byte-identical to its target.

This driver makes that loop safe:
  * edits are confined to ONE function's body, so a literal like `64` cannot
    accidentally be rewritten somewhere else in fleet_mk_generate.py;
  * every `old` must occur an exact expected number of times -- a count that
    drifted means the substitution is hitting something it should not;
  * the round-trip gate runs after the write, and the file is restored if it
    goes red.

Usage:
    ./subst.py --fn generate_kernel_fused_moe --spec specs/kernel_dims.py

A spec file defines SUBS = [(old, new, count), ...] and is exec'd, not
imported, so it can be a plain literal list with comments.
"""

import argparse
import ast
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(REPO, "fleet_mk_generate.py")


def function_span(text, name):
    """(start, end) character offsets of the whole `def name(...)` body.

    Parsed with ast rather than scanned for the next "\\ndef ". The emitters
    hold pasted artifacts inside f-strings, and demo_gpt_oss_120b.py has its
    own top-level `def`s -- a textual scan stops at the first one and silently
    hands back a fraction of the function, so a substitution that should hit
    once reports zero and the refusal looks like a bad pattern rather than a
    bad span.
    """
    tree = ast.parse(text)
    lines = text.split("\n")
    # Character offset of the start of each 1-indexed line.
    offs, acc = [0, 0], 0
    for ln in lines:
        acc += len(ln) + 1
        offs.append(acc)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == name:
            return offs[node.lineno], offs[node.end_lineno + 1]
    raise SystemExit(f"REFUSED: no top-level function named {name!r}.")


def apply(subs, fn_name, text):
    start, end = function_span(text, fn_name)
    body = text[start:end]
    for old, new, count in subs:
        got = body.count(old)
        if got != count:
            raise SystemExit(
                f"REFUSED: expected {count} occurrence(s) of {old!r} in "
                f"{fn_name}, found {got}. The substitution is not hitting what "
                f"you think it is.")
        body = body.replace(old, new)
    return text[:start] + body + text[end:]


def gate(strict=True, only=None):
    cmd = [sys.executable, os.path.join(REPO, "check_roundtrip.py")]
    if strict:
        cmd.append("--strict")
    for label in (only or []):
        cmd += ["--only", label]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    return r.returncode, r.stdout + r.stderr


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fn", required=True, help="function whose body to edit")
    ap.add_argument("--spec", required=True, help="python file defining SUBS")
    ap.add_argument("--no-strict", action="store_true",
                    help="gate without --strict (driver not built yet)")
    ap.add_argument("--only", action="append", metavar="LABEL",
                    help="pass through to check_roundtrip.py --only")
    args = ap.parse_args()

    ns = {}
    with open(args.spec, encoding="utf-8") as f:
        exec(compile(f.read(), args.spec, "exec"), ns)
    subs = ns["SUBS"]

    with open(GEN, encoding="utf-8") as f:
        before = f.read()

    after = apply(subs, args.fn, before)
    if after == before:
        raise SystemExit("REFUSED: substitution changed nothing.")

    backup = tempfile.mktemp(suffix=".fleet_mk_generate.py")
    shutil.copy2(GEN, backup)
    with open(GEN, "w", encoding="utf-8") as f:
        f.write(after)

    rc, out = gate(strict=not args.no_strict, only=args.only)
    print(out)
    if rc != 0:
        shutil.copy2(backup, GEN)
        print(f"GATE RED -- reverted {len(subs)} substitution(s). "
              f"Backup kept at {backup}")
        return 1
    os.unlink(backup)
    print(f"GATE GREEN -- {len(subs)} substitution(s) kept.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
