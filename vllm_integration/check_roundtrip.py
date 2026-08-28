#!/usr/bin/env python3
"""Round-trip tripwire for titan_generate.py.

Asserts that regenerating a model's artifacts from its YAML config reproduces
the on-disk file byte-for-byte. Sub-second, no GPU, no build. Run this after
every edit to titan_generate.py.

WHY THIS EXISTS
---------------
The generator silently diverged from the hand-written GPT-OSS 120B kernel that
actually performs (2.520 ms/tok). Nothing regenerated the artifact, so nothing
caught the drift. This test makes "the generator works" falsifiable.

The dense path (qwen3_8b, llama3_8b) already round-trips byte-identically for
kernel/launch/build. That is the regression baseline: any refactor of the
generator must leave it green.

  ./check_roundtrip.py            # dense = hard gate, gptoss = informational
  ./check_roundtrip.py --strict   # everything is a hard gate (use from step 8)

EXIT CODE
---------
0 = every hard-gated check passed. Non-zero = at least one failed.
"""

import argparse
import contextlib
import io
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))

# (config stem, generator fn suffix -> on-disk path relative to REPO)
#
# Each entry is (label, generator_attr, path_template). path_template is
# formatted with name_clean.
ARTIFACTS = [
    ("kernel", "generate_kernel", "generated/{nc}_kernel.cuh"),
    ("launch", "generate_launch", "generated/{nc}_launch.hip"),
    ("build",  "generate_build",  "build_{nc}.sh"),
    ("driver", "generate_driver", "demo_{nc}.py"),
]

# Models whose artifacts must match byte-for-byte, and which artifacts are
# gated. Anything listed in `skip` is checked and reported but never fails the
# run -- with the reason recorded here so it does not become folklore.
MODELS = [
    {
        "config": "configs/qwen3_8b.yaml",
        # driver is gated as of the COUNTERS_PER_LAYER fix: the on-disk demo
        # hardcoded 40 * 16 while the header had moved to 103 * 16, so the
        # kernel's rank_counters landed ~36000 ints past the allocation. That
        # is an out-of-bounds atomicAdd with no diagnostic -- it presented as a
        # memory access fault at a plausible-looking address. Regenerating is
        # what fixed it, which is exactly the drift this gate exists to catch.
        "gated": {"kernel", "launch", "build", "driver"},
        "skip": {},
    },
    {
        "config": "configs/llama3_8b.yaml",
        "gated": {"kernel", "launch", "build"},
        "skip": {
            # Same stale device_map drift qwen3_8b had, still unregenerated --
            # llama3_8b has no weights here to verify a regenerated driver
            # against, and promoting one blind is how the counter bug shipped.
            # Regenerate and re-gate when the weights are available.
            "driver": 'stale on disk (device_map "cuda" vs "cpu")',
        },
    },
    {
        "config": "configs/gpt_oss_120b.yaml",
        # Byte-identity scope for the fused MoE path is the kernel .cuh and the
        # demo .py. launch.hip and build.sh are gated on building and hitting
        # 2.52 ms, not on bytes.
        # launch.hip is only required to build and measure 2.52 ms, but the
        # backwards construction reaches byte-identity for free, so it is gated
        # here too -- a weaker gate is not worth the drift it lets through. If
        # a future change to the launch wrapper is a deliberate improvement,
        # move it back to `skip` with the reason, do not loosen it silently.
        # build.sh was also planned as a semantic gate only. Driving -O3,
        # -fgpu-rdc and -DMPK_W13_LDS_PREFETCH from the YAML `build:` section
        # reached byte-identity, so it is gated on bytes as well. The three
        # flags are the ones commit f2354a7 records as affecting the 2.520 ms
        # result -- byte drift here is a silent performance change.
        "gated": {"kernel", "driver", "launch", "build"},
        "skip": {},
        # Until the backwards-built emitters land (plan steps 4-5), these two
        # are expected to fail. --strict flips them to hard gates.
        "expect_fail_until_strict": True,
    },
]


def first_diff(a, b):
    """Human-readable description of where two strings first differ."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    if i == n and len(a) == len(b):
        return "identical"
    line = a.count("\n", 0, i) + 1
    col = i - (a.rfind("\n", 0, i) + 1) + 1
    if i == n:
        longer = "generated" if len(a) > len(b) else "on-disk"
        return (f"byte {i} (line {line}, col {col}): {longer} is longer "
                f"({len(a)} vs {len(b)} bytes)")
    return (f"byte {i} (line {line}, col {col}): "
            f"generated {a[i]!r} vs on-disk {b[i]!r} "
            f"({len(a)} vs {len(b)} bytes)")


def hygiene(text):
    """Whitespace drift that `diff` hides but `cmp` catches.

    The byte target for every emitted file: no CRLF, no tabs, no trailing
    whitespace, and exactly one final newline. An f-string emitter picks these
    up silently from a careless paste.
    """
    problems = []
    if "\r" in text:
        problems.append(f"{text.count(chr(13))} CR")
    if "\t" in text:
        problems.append(f"{text.count(chr(9))} tab")
    trailing = sum(1 for ln in text.split("\n") if ln != ln.rstrip())
    if trailing:
        problems.append(f"{trailing} trailing-ws lines")
    if not text.endswith("\n"):
        problems.append("no final newline")
    elif text.endswith("\n\n"):
        problems.append("multiple final newlines")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true",
                    help="Gate on every artifact listed in 'gated', including "
                         "GPT-OSS. Use once the fused emitters are built.")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Show the generator's own config banner.")
    ap.add_argument("--only", action="append", metavar="LABEL",
                    help="Restrict to these artifact labels (kernel, launch, "
                         "build, driver). Repeatable. Use while an emitter is "
                         "still being built backwards, so its not-yet-written "
                         "sibling does not mask the gate.")
    args = ap.parse_args()

    sys.path.insert(0, REPO)
    import titan_generate as tg

    failures = []
    lines = []

    for spec in MODELS:
        cfg_path = os.path.join(REPO, spec["config"])
        banner = io.StringIO()
        with contextlib.redirect_stdout(banner):
            cfg = tg.load_and_validate(cfg_path)
        if args.verbose:
            lines.append(banner.getvalue().rstrip())

        soft = spec.get("expect_fail_until_strict", False) and not args.strict
        lines.append(f"{cfg.name_clean}  (arch={cfg.arch})"
                     + ("   [informational until --strict]" if soft else ""))

        for label, attr, tmpl in ARTIFACTS:
            if args.only and label not in args.only:
                continue
            path = os.path.join(REPO, tmpl.format(nc=cfg.name_clean))
            if label in spec.get("skip", {}):
                lines.append(f"    skip  {label:7s} -- {spec['skip'][label]}")
                continue
            if label not in spec["gated"]:
                continue
            if not os.path.exists(path):
                lines.append(f"    MISS  {label:7s} {path} does not exist")
                if not soft:
                    failures.append(f"{cfg.name_clean}/{label}: missing {path}")
                continue

            try:
                generated = getattr(tg, attr)(cfg)
            except NotImplementedError as e:
                # An emitter still being built backwards. Report it, but let the
                # dense checks in this same run still speak.
                lines.append(f"    TODO  {label:7s} {e}")
                if not soft:
                    failures.append(f"{cfg.name_clean}/{label}: not implemented")
                continue

            # Explicit utf-8: both GPT-OSS targets contain em-dash and arrow
            # characters, so a C-locale environment would decode them wrong and
            # produce a spurious diff. main() must write with the same encoding.
            with open(path, encoding="utf-8") as f:
                on_disk = f.read()

            if generated == on_disk:
                warn = hygiene(generated)
                suffix = ("   [hygiene: " + ", ".join(warn) + "]") if warn else ""
                lines.append(f"    OK    {label:7s} {len(on_disk)} bytes{suffix}")
            else:
                lines.append(f"    FAIL  {label:7s} {first_diff(generated, on_disk)}")
                if not soft:
                    failures.append(f"{cfg.name_clean}/{label}")

    print("\n".join(lines))
    print()
    if failures:
        print(f"FAILED ({len(failures)}): " + ", ".join(failures))
        return 1
    print("PASS" + (" (strict)" if args.strict else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
