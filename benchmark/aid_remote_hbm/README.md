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
| `hipmalloc_pages.hip` | how hipMalloc lays out pages, and the AID granularity/periodicity |
| `aid_alloc.hip` | AID-pinned allocator built on 4 KiB VMM handles, plus the bandwidth payoff |
| `aid_gemv.hip` | batch-1 GEMV on AID-pinned weights — the end-to-end latency result |
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

## Pinning allocations to one AID (`aid_alloc.hip`)

This works, with no partition mode change and no root. HIP's virtual memory API
reports a 4 KiB allocation granularity on this part, which is exactly the
granularity at which homing changes, so pages can be selected individually and
re-mapped into a contiguous virtual range:

1. `hipMemCreate` one 4 KiB physical handle per page, map them all into a
   scratch VA range.
2. Classify every page in a single kernel launch (one wave on XCD0, one on
   XCD1, bouncing a flag inside each page in turn).
3. `hipMemAddressReserve` a fresh range and `hipMemMap` only the matching
   handles into it, contiguously.

The caller gets a plain `void*`. Classification runs at **45 us/page**, so a
256 MiB buffer costs about 9 s of probing over a 768 MiB candidate pool. The
split is roughly 43/57 rather than exactly even, so oversample by 3x.

Measured on `0000:06:00.0`, 256 MiB buffers, re-probing the finished buffers:

```
local  buffer: 4096/4096 pages on the expected side, mean  728 ns
remote buffer: 4091/4096 pages on the expected side, mean 1091 ns
```

### Does it pay?

NT streaming of 256 MiB, restricted to one chiplet group (GB/s):

| reader | pages homed XCD0-3 | pages homed XCD4-7 | prefers |
|---|---|---|---|
| XCD0-3 | **89.0** | 77.9 | its own, 1.14x |
| XCD4-7 | 73.2 | **83.2** | its own, 1.14x |

Both groups prefer their own pages by the same 1.14x. The crossover is the
control: if one buffer were simply faster, both rows would pick the same column.
That is a real 14% of read bandwidth from allocation placement alone, in the
range of the 20% the paper attributes to avoiding remote-AID access.

Two caveats worth keeping:

- **The working set has to exceed the 224 MiB L3.** At 16 MiB the same test
  gives 0.94x, i.e. nothing, because the memory-side cache absorbs the traffic
  and the home stops mattering.
- **The map follows physical addresses**, so it must be re-probed per
  allocation and cannot be cached across runs.

## Latency on a real kernel (`aid_gemv.hip`)

STREAM is the friendliest possible case, so the same question on a GEMV.
Batch-1 decode is the shape that matters here: `y[n] = dot(W[n][:], x[:])`
touches every weight exactly once and reuses nothing, so it is pure weight
streaming and it is what the megakernel spends its time on. `N=16384, K=8192`
fp32 puts W at 512 MiB, comfortably past the 224 MiB L3.

Phase 1, only XCD0-3 working, over 256 MiB of W:

| W homed | latency | GB/s |
|---|---|---|
| on its own side | **1.62 ms** | 165.3 |
| on the far side | 2.00 ms | 134.4 |

23% more latency for the same arithmetic, purely from where the pages live.

Phase 2 is the version you would actually ship: all 8 chiplets working, each
group taking half the rows, with that half homed on its own side.

| W allocation | latency | GB/s |
|---|---|---|
| plain `hipMalloc` | 1.81 ms | 296.9 |
| AID-pinned | **1.64 ms** | 327.2 |
| inverted (control) | 2.02 ms | 266.3 |

**1.10x end-to-end** against stock `hipMalloc`, and 1.23x against the inverted
control. The control is the load-bearing part: it reuses the exact same VMM
pages and only swaps which group reads which half, so the gain cannot be an
artifact of 4 KiB VMM pages differing from `hipMalloc` pages. `hipMalloc`
landing between the two is what its ~50/50 split predicts.

Checksums are identical across all five configurations. Reproduced across three
runs, spread under 1%.

The honest framing of the 1.10x: half the win is already there by accident,
because `hipMalloc`'s even spread means half of each group's reads are local
anyway. Pinning converts the remaining half. The 1.23x pinned-vs-inverted gap is
the full size of the effect.

Probe cost is 8.8 s for the 768 MiB candidate pool, which only makes sense for
allocations that live for the whole process — model weights, not activations.

## How hipMalloc lays out pages (`hipmalloc_pages.hip`)

Observed mechanics:

- Allocations up to 1 MiB all come back at the *same* address after being
  freed: ROCr suballocates them from a larger block. That is the fragment
  allocator, and `HSA_DISABLE_FRAGMENT_ALLOCATOR` turns it off.
- Allocations of 8 MiB and up are 2 MiB aligned. The kernel-side VRAM manager
  is a buddy allocator with 4 KiB chunks (`amdgpu_vram_mm` in debugfs).
- The home side is a deterministic function of the physical address: two
  separate 16 MiB allocations line up at **100%** agreement under a pure page
  shift (-768, +1536 and -512 pages seen on different runs).

Is it round-robin across the stacks? Partly.

- **Even spread, yes.** Every 16 MiB allocation lands 2341/4096 pages on one
  side, run after run. A big `hipMalloc` buffer is ~50/50 across the AIDs, which
  is exactly why an unmodified allocation is half-remote to whichever chiplets
  read it.
- **But the AID bit is not a simple modulo.** 4 KiB round-robin over 8 stacks
  with stacks 0-3 on one AID would give runs of exactly 4 pages and a period of
  8. Measured run lengths are 615x1, 1049x2, 347x3, 61x4, 16x5, 3x6, and the
  match at P=8 is 0.47, i.e. chance (P=2 0.28, P=4 0.59, P=16 0.52, P=512 0.43).
  It behaves like an address hash.
- **Granularity is 4 KiB, not 256 B.** Probing 16 sub-page offsets in each of 8
  pages gives a uniform side within every page while the side varies between
  pages:

```
page 0: 1111111111111111   uniform
page 1: 0000000000000000   uniform
page 2: 1111111111111111   uniform
...
```

  So the AID selection changes only at page boundaries. Fine-grained striping
  across the four channels *inside* an AID is not ruled out — this probe only
  reports which side of the machine an address lives on, never which of the 8
  stacks — but whatever the hardware does below 4 KiB does not move the AID bit.

The period search reports a 100% match at some large period (8960 pages in one
run, 14336 in the next). That number moves between runs, so it describes how the
buddy allocator laid out that particular allocation rather than a hardware
period. There is no short intrinsic period, which is why `aid_alloc` probes
rather than predicts.

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
for f in aid_remote_hbm xcd_pair_matrix aid_home_map aid_alloc aid_gemv aid_page_map; do
  hipcc -O3 -std=c++17 --offload-arch=gfx950 \
    benchmark/aid_remote_hbm/$f.hip -o /tmp/$f
done
HIP_VISIBLE_DEVICES=1 /tmp/aid_alloc 256   # argument is the buffer size in MiB
HIP_VISIBLE_DEVICES=1 /tmp/aid_gemv        # ~90 s, most of it page probing
```

Run on an idle GPU; a neighbouring process on the same package perturbs the
latency split.
