#!/usr/bin/env python3
"""Census of fleet's per-token HBM reads, split by how localizable they are.

Reads a task_graph JSON and, for every tensor a task reads, works out whether
the byte range is owned by one XCD or shared by all of them. That decides
whether AID pinning can help:

  private   each XCD reads a disjoint range, fixed at graph-build time
            -> pinnable today, currently ~50% remote by luck of hipMalloc
  routed    disjoint per XCD, but which range depends on runtime MoE routing
            -> pinnable only after the tile->XCD map is made expert-invariant
  shared    every XCD reads the same bytes
            -> not pinnable; one AID is always remote (or replicate it)

Usage: python3 analyze_locality.py [task_graph_0.json]
"""

import collections
import json
import math
import sys

DT_BYTES = {930: 1, 935: 1, 936: 1, 940: 2, 941: 2, 945: 2, 946: 2,
            950: 4, 955: 4, 956: 4}

NUM_XCDS = 8
XCDS_PER_AID = 4
TOP_K = 4          # gpt-oss-120b num_experts_per_tok
NUM_EXPERTS = 128  # num_local_experts


def extent(inp):
    """(first_byte, last_byte+1) of the region this input descriptor covers."""
    esz = DT_BYTES[inp["data_type"]]
    dims, strides = inp["dims"], inp["strides"]
    if len(strides) < len(dims):
        strides = [1] * len(dims)
    span = sum((d - 1) * s for d, s in zip(dims, strides)) + 1
    return inp["offset"] * esz, (inp["offset"] + span) * esz


def dense_bytes(inp):
    return math.prod(inp["dims"]) * DT_BYTES[inp["data_type"]]


def main(path):
    tasks = json.load(open(path))["all_tasks"]
    layer_tasks = [t for t in tasks if t["task_type"] == 216]
    assert len(layer_tasks) % NUM_XCDS == 0, len(layer_tasks)
    n_layers = len(layer_tasks) // NUM_XCDS
    print(f"{path}: {len(tasks)} tasks, {n_layers} layer groups "
          f"x {NUM_XCDS} XCDs\n")

    # Group the 8 sibling tasks of layer 0 and compare what each one reads.
    # Task index within the group is the XCD id (global_tile = tile*8 + xcd).
    group = layer_tasks[:NUM_XCDS]
    per_tensor = collections.defaultdict(list)
    for xcd, t in enumerate(group):
        for inp in t["inputs"]:
            per_tensor[inp["base_ptr"]].append((xcd, extent(inp),
                                                dense_bytes(inp)))

    # A tensor is "private" when the 8 XCDs' ranges are pairwise disjoint.
    classes, bytes_per_layer = {}, {}
    for name, entries in per_tensor.items():
        ranges = [e[1] for e in entries]
        disjoint = all(
            ranges[i][1] <= ranges[j][0] or ranges[j][1] <= ranges[i][0]
            for i in range(len(ranges)) for j in range(i + 1, len(ranges)))
        # MoE expert weights are handed over whole because routing is dynamic.
        routed = ("gate_up_proj" in name or "down_proj" in name
                  or "gate_up_bias" in name or "down_bias" in name)
        if routed:
            classes[name] = "routed"
            # Only TOP_K of NUM_EXPERTS experts are touched per token, and each
            # active expert's slab is read in full across the 8 XCDs.
            inp = next(i for i in group[0]["inputs"] if i["base_ptr"] == name)
            slab = dense_bytes(inp) // inp["dims"][0]  # bytes for one expert
            bytes_per_layer[name] = slab * TOP_K
        elif disjoint:
            classes[name] = "private"
            bytes_per_layer[name] = sum(e[2] for e in entries)
        else:
            classes[name] = "shared"
            bytes_per_layer[name] = sum(e[2] for e in entries)

    order = {"private": 0, "routed": 1, "shared": 2}
    rows = sorted(bytes_per_layer.items(),
                  key=lambda kv: (order[classes[kv[0]]], -kv[1]))
    print(f"{'tensor':32s} {'class':8s} {'MiB/layer/token':>16s}")
    print("-" * 60)
    for name, b in rows:
        if b < 1024:
            continue
        print(f"{name:32s} {classes[name]:8s} {b / 2**20:16.3f}")

    tot = collections.Counter()
    for name, b in bytes_per_layer.items():
        tot[classes[name]] += b
    grand = sum(tot.values())

    print("\n" + "=" * 60)
    print(f"per token, all {n_layers} layers (bytes read from HBM)")
    print("=" * 60)
    print(f"{'class':10s} {'GiB/token':>11s} {'share':>8s}")
    for k in ("private", "routed", "shared"):
        g = tot[k] * n_layers / 2**30
        print(f"{k:10s} {g:11.3f} {100 * tot[k] / grand:7.1f}%")
    print(f"{'total':10s} {grand * n_layers / 2**30:11.3f}")

    # Locality outcome. hipMalloc spreads pages ~50/50 over the two AIDs, so
    # half of every read is remote regardless of who issues it.
    print("\n" + "=" * 60)
    print("remote fraction of that traffic")
    print("=" * 60)
    gib = lambda b: b * n_layers / 2**30
    priv, rout, shar = tot["private"], tot["routed"], tot["shared"]

    print(f"{'scenario':38s} {'local':>9s} {'remote':>9s} {'remote%':>8s}")
    print("-" * 68)
    rem = 0.5 * grand
    print(f"{'today (hipMalloc, ~50/50 spread)':38s} "
          f"{gib(grand - rem):9.3f} {gib(rem):9.3f} {100 * rem / grand:7.1f}%")

    rem = 0.5 * rout + 0.5 * shar
    print(f"{'pin private tensors only':38s} "
          f"{gib(grand - rem):9.3f} {gib(rem):9.3f} {100 * rem / grand:7.1f}%")

    rem = 0.5 * shar
    print(f"{'+ expert-invariant MoE tile map':38s} "
          f"{gib(grand - rem):9.3f} {gib(rem):9.3f} {100 * rem / grand:7.1f}%")

    print(f"\nshared traffic is {gib(shar) * 1024:.1f} MiB/token "
          f"({100 * shar / grand:.2f}%) — replicating it per AID would cost "
          f"that much extra memory to remove the last remote reads.")

    moe_tile_map()


def moe_tile_map(w13_wgs=92, w2_wgs=46):
    """Why MoE weights are 'routed'.

    gang_moe_fused_mxfp4_mi300.cuh decodes tiles as
        global_tile = tile_idx * 8 + xcd_id
        expert_idx  = global_tile / TILES ; wg_idx = global_tile % TILES
    so the XCD that reads workgroup w of an expert is (p*TILES + w) % 8, where
    p is that expert's *position in the activated list* — which changes every
    token. A weight page can only be pinned if its reading AID is the same for
    every position it might land in.
    """
    print("\n" + "=" * 60)
    print("MoE tile map: is the reading AID fixed per weight page?")
    print("=" * 60)
    for phase, tiles in (("W13", w13_wgs), ("W2", w2_wgs)):
        stable = sum(
            len({((p * tiles + w) % NUM_XCDS) // XCDS_PER_AID
                 for p in range(TOP_K)}) == 1
            for w in range(tiles))
        print(f"  {phase}: TILES={tiles}, TILES%8={tiles % NUM_XCDS} -> "
              f"{stable}/{tiles} workgroups keep one AID across the {TOP_K} "
              f"expert positions ({100 * stable / tiles:.0f}%)")
    print("  Fix: make the tile map expert-invariant (xcd = wg_idx % 8) so a\n"
          "  workgroup is always read by the same XCD regardless of routing.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "demo/gpt_oss/task_graph_0.json")
