# Local vs remote AID access on MI350/MI355

This machine is one MI350 package: 8 XCDs (chiplets) split into two groups of
four, with HBM stacks attached to the AIDs. Traffic that has to reach the other
side crosses the inter-AID interconnect. This is the effect GTAC'25 paper 48
(Starscream) motivates its CPX/NPS4 partitioning with; the paper measures it on
MI300X (4 AIDs), where 75% of a CU's traffic is remote under SPX/NPS1.

The GPU here runs SPX/NPS1 and cannot be put in NPS2 without also splitting
compute (`amd-smi partition --accelerator`: SPX offers NPS1 only; NPS2 needs
DPX/QPX/CPX). The question these benchmarks answer is what can be done *without*
touching partition modes.

**Answer: memory can be placed AID-locally from software.** The home AID is a
property of each 4 KiB page, it is stable and probeable at runtime, so an
allocator can hand out pages whose home matches the chiplets that will read
them. Details below.

## Tools

| file | what it does |
|---|---|
| `aid_remote_hbm.hip` | flag ping-pong between two named chiplets, plus NT STREAM phases |
| `xcd_pair_matrix.hip` | round-trip latency for all 28 chiplet pairs, then a flag-address sweep |
| `aid_home_map.hip` | classifies the home side of each address granule; finds the granularity |
| `aid_page_map.hip` | single-chiplet atomic latency probe — **negative result, see caveat** |

All of them read `HW_REG_XCC_ID` so each phase runs only on the chiplet it
names, and they allocate probe flags with `hipDeviceMallocUncached` so the
traffic actually reaches the fabric instead of sitting in L2.

## What the chiplet-pair matrix shows

`xcd_pair_matrix` on an idle GPU (`0000:06:00.0`), round-trip ns:

```
     XCD0 XCD1 XCD2 XCD3 XCD4 XCD5 XCD6 XCD7
XCD0    . 1090 1152 1109  984  956  999  972
XCD1 1090    . 1111 1085  954  814  970  815
XCD2 1152 1111    . 1089  916  818 1010  922
XCD3 1109 1085 1089    .  950  817  951  889
XCD4  984  954  916  950    .  731  765  687
XCD5  956  814  818  817  731    .  742  626
XCD6  999  970 1010  951  765  742    .  749
XCD7  972  815  922  889  687  626  749    .
```

Pair cost behaves like `c_a + c_b`: every chiplet has its own distance to the
flag, and the pair pays both. For *this* flag the {4,5,6,7} group is close and
{0,1,2,3} is far. Grouping the pairs as same-AID vs cross-AID gives a 1.01x
"penalty", i.e. nothing — the structure is about where the *flag* lives, not
about which pairs span a boundary.

## The home AID is an address property

Sweeping the flag address (same pair, different offsets) flips which group is
fast:

| flag offset | XCD0-1 | XCD4-5 |
|---|---|---|
| 1 MiB | 713 | 1164 |
| 3 MiB | 1210 | 682 |
| 1024 MiB | 704 | 1067 |
| 1024 MiB + 4 KiB | 1060 | 707 |

Two addresses 4 KiB apart land on opposite sides. `aid_home_map` turns this into
a classifier — the fast/slow split is wide and bimodal (692 vs 1180 ns), so one
ping-pong run classifies one address, and the XCD4-XCD5 pair inverts it as a
check (8/8 agreement).

Granularity, from the classifier:

| stride | pattern | reading |
|---|---|---|
| 256 B x32 (within a page) | all identical | a 4 KiB page has one home |
| 1 KiB x32 | transitions only every 4th sample | boundaries land on 4 KiB |
| 4 KiB x64 | 38/64 on side 1, 32 transitions | per-page, hashed, ~50/50 |
| 2 MiB x32 | 19/32 on side 1, 18 transitions | same at coarse stride |

So the assignment is per 4 KiB page and looks hashed rather than a simple
round-robin, but it is deterministic: re-probing pages 0-7 in a later kernel
launch reproduced `1 1 0 1 1 0 0 1` exactly.

## How to use it

Probe once at init, then allocate accordingly:

1. Allocate the pool (physical pages are pinned for its lifetime).
2. Classify each 4 KiB page with one ping-pong between a chiplet from each
   group. The map follows the physical address, so it must be re-probed per
   allocation, not cached across runs.
3. Hand out only pages whose home matches the chiplet group that will consume
   them.

What is not yet measured: whether this pays for bulk streaming. The 1.7x
measured here is a coherence round trip on a single flag, which is the
sync-primitive case. The payoff for weight streaming needs a bandwidth run over
a working set larger than the 224 MiB L3, comparing local-page and remote-page
buffers read from one chiplet group.

## Caveat on `aid_page_map` (negative result, and why it is wrong)

`aid_page_map` times a dependent chain of agent-scope atomics that stays on one
address, probed from each chiplet in turn. It reports every chiplet as
equidistant from every address (all 88-102 ns, AIDs agreeing within 0.5 ns), and
concluding from it that no allocation can be AID-local would be wrong.

The probe is blind by construction: when a single chiplet hammers one address,
the line stays owned by that chiplet and the atomic never has to travel to the
home node. The home only becomes visible when ownership has to move, which is
why the two-chiplet ping-pong sees a 1.7x split on the same addresses. The file
is kept because the failure mode is worth knowing: single-agent latency probes
cannot map NUMA structure on this part.

## Power

Socket `power1_input` moves only 281 W -> 284 W between the local and remote
ping-pong cases against a ~239 W idle. Two spinning waves are far too small a
load for a 1000 W package, and the counter is whole-socket rather than per-AID,
so nothing here substantiates the paper's AID-power claim. The latency figures
are the usable signal.

## Build

```bash
for f in aid_remote_hbm xcd_pair_matrix aid_home_map aid_page_map; do
  hipcc -O3 -std=c++17 --offload-arch=gfx950 \
    benchmark/aid_remote_hbm/$f.hip -o /tmp/$f
done
HIP_VISIBLE_DEVICES=1 /tmp/aid_home_map
```

Run on an idle GPU; a neighbouring process on the same package perturbs the
latency split.
