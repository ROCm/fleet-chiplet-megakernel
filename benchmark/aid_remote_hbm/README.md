# Local vs remote AID access on MI350/MI355

This machine is one MI350 package: 8 XCDs (chiplets) split into two groups of
four, with HBM stacks attached to the AIDs. Traffic that has to reach the other
side crosses the inter-AID interconnect. This is the effect GTAC'25 paper 48
(Starscream) motivates its CPX/NPS4 partitioning with; the paper measures it on
MI300X (4 AIDs), where 75% of a CU's traffic is remote under SPX/NPS1.

The GPU here runs SPX/NPS1 and cannot be put in NPS2 without also splitting
compute. `sudo amd-smi partition --accelerator` is the authoritative view:
profile 0 is SPX with `MEMORY_PARTITION_CAPS: NPS1`, and only DPX (4 XCC), QPX
(2 XCC) and CPX (1 XCC) list `NPS1,NPS2`. The per-GPU `NPS1,NPS2` printed by
plain `amd-smi partition` is the union over all profiles and does *not* mean SPX
can take NPS2 -- confirmed the hard way, see "What NPS2 actually costs" at the
end. So NPS2 is never free here: it is paid for with half the GPU or worse. The
question these benchmarks answer is what can be done *without* touching
partition modes.

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
| `analyze_locality.py` | census of fleet's per-token reads: how much is local, remote, pinnable |
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

## How much of fleet's traffic is remote (`analyze_locality.py`)

The benchmarks above say a remote read costs ~23% more. This says how much of
fleet's traffic is exposed to that. The script walks `demo/gpt_oss` task graph
0 (gpt-oss-120b, batch 1 decode) and, for every tensor a task reads, checks
whether the 8 sibling tasks of a layer touch disjoint byte ranges. Task index
within a layer group is the XCD, since `global_tile = tile_idx * 8 + xcd_id`.

Per token, across all 36 layers:

| class | GiB/token | share | meaning |
|---|---|---|---|
| private | 0.509 | 20.8% | QKV, o_proj, gate — each XCD reads a disjoint, build-time-fixed range |
| routed | 1.855 | 75.8% | MoE experts — disjoint per XCD, but which range depends on runtime routing |
| shared | 0.083 | 3.4% | norms, barriers, accumulators — every XCD reads the same bytes |
| total | 2.447 | | |

The 2.447 GiB/token checks out against the model: 128 experts x 138 wgs x
100096 B x 36 layers is 63.6 GB of MXFP4 weights, and top-4-of-128 routing
touches ~2.4 GiB of it per token.

Today `hipMalloc`'s ~50/50 spread means **half of all 2.447 GiB is read from the
far AID**, private and routed alike. Where that can go:

| scenario | local | remote | remote % |
|---|---|---|---|
| today (hipMalloc) | 1.223 | 1.223 | 50.0% |
| pin private tensors only | 1.478 | 0.969 | 39.6% |
| + expert-invariant MoE tile map | 2.405 | 0.041 | **1.7%** |

Pinning alone only gets 50% down to 39.6%, because the 76% of traffic that is
MoE weights cannot be pinned as the kernel is written today. The tile decode is

    global_tile = tile_idx * 8 + xcd_id
    expert_idx  = global_tile / TILES ;  wg_idx = global_tile % TILES

so workgroup `w` of an expert is read by XCD `(p * TILES + w) % 8`, where `p` is
that expert's **position in the activated list** — which changes every token.
With `W13_TILES = 92` (`92 % 8 = 4`) and `W2_TILES = 46` (`46 % 8 = 6`), the
script finds **0 of 92** and **0 of 46** workgroups keep a single AID across the
4 possible positions. Not one weight page has a stable home, so no static
placement can help.

Making the map expert-invariant is what takes remote traffic to 1.7%. The
obvious `xcd = wg_idx % 8` works but costs load balance, because neither 92 nor
46 divides by 8. The better map uses the fact that **only the AID has to be
invariant, not the XCD** — there are 2 memory domains, not 8, so which of the
4 XCDs inside an AID takes a tile is still free to rotate:

    aid = (w < TILES/2) ? 0 : 1                  // fixed per workgroup
    xcd = aid * 4 + ((p * (TILES/2) + w) % 4)    // rotates within the AID

| map | AID-invariant wgs | XCD load imbalance |
|---|---|---|
| current `(p*TILES + w) % 8` | 0/92, 0/46 | 1.00x |
| naive `w % 8` | 92/92, 46/46 | 1.09x (W13), 1.20x (W2) |
| AID-split + rotate within AID | 92/92, 46/46 | **1.00x** |

The halves divide evenly (92 -> 46 per AID, 46 -> 23) and 4 expert positions
over 4 XCDs lands exactly, so this buys full pinnability at no balance cost.

The residual 1.7% is the 85 MiB/token of genuinely shared tensors. Removing it
would mean replicating them per AID, which is cheap in bytes but only worth it
after the first two steps land.

Two things this analysis does *not* establish. It counts bytes demanded, not
bytes that miss the 224 MiB L3, so the shared tensors in particular are likely
already cached and cheaper than their row suggests. And fleet's MoE GEMMs
overlap MFMA with the weight stream, so the end-to-end gain is bounded by the
1.10x measured on a pure-streaming GEMV rather than equal to it. No end-to-end
number is claimed here because none has been measured.

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

## Does it scale to the whole model? (`aid_bulk_pin.hip`, `aid_pin.hip`)

`aid_alloc` pins a 256 MiB buffer. The MoE weights are 36 layers x 1.77 GB =
64 GB, or 16.7M pages, so the question is whether the same trick survives a
250x scale-up. Half of it does.

The classifier does. It was serial -- one wave pair walking every page at
45 us/page -- but pages are independent addresses, so N pairs need no barrier
between them. Two XCD pairs x 32 wave pairs give **0.8 us/page at 100%
agreement** with the serial reference and no timeouts, turning 12 minutes of
probing into 14 seconds.

The VMM calls do not, and there is no way around them:

| call | rate | for 16.7M pages |
|---|---|---|
| `hipMemCreate` 4 KiB | 47 us (2K live) ... 238 us (202K live) | 66 min+ |
| `hipMemMap` + `SetAccess` 4 KiB | 10 us ... 186 us | 52 min+ |
| classify | 0.8 us | 14 s |

Three escapes were tried and all are closed:

- **Large handles.** Homing is uniform inside a 4 KiB handle and nothing else:
  0 of 128 64 KiB handles, 0 of 32 256 KiB handles and 0 of 4 2 MiB handles sit
  on a single AID. The interleave is per 4 KiB whatever the allocation size, so
  the cheap 2 MiB create rate is unreachable.
- **Sub-handle mapping.** `hipMemMap` rejects a non-zero offset into a handle,
  so a large handle cannot be dealt out page by page either.
- **Host threads.** The calls anti-scale: 47 -> 112 -> 204 -> 429 us/handle at
  1/4/16/32 threads. That is a global lock, not work.

`aid_pin.hip` is the working allocator anyway, as a C ABI for demo.py, and it
is correct: one 562 MiB `down_proj` pins with **100% of pages on the intended
side** and a clean round trip. It takes **116 s**. Extrapolated to 64 GB that
is ~3.7 hours, and that is the optimistic reading, because the per-call rate
grows with the live handle count and the final state holds 16.7M live handles.

The map cannot be cached across runs either -- not because the weights change,
but because the AID is a property of the *physical* page, and `hipMemCreate`
returns different physical pages each run. The stable thing is the physical
address -> AID hash, which userspace cannot see. Caching would save only the
14 s of classification regardless; the cost is create and map.

## What is the ceiling? (`MPK_PHASE_SLOTS`)

Per-layer phase breakdown of the 1.825 ms decode, averaged over the 240
compute workers (`MPK_PHASE_SLOTS=1 MPK_PHASE_START_ITER=20`):

| slot | ns | share | phase |
|---|---|---|---|
| 6 | 12980 | 23.9% | O-proj + RMSNorm + Router + TopK |
| 8 | 12927 | **23.8%** | MoE (W13+SwiGLU+W2) |
| 7 | 7608 | 14.0% | O-proj barrier |
| 5 | 7372 | 13.6% | cross-XCD attention barrier |
| 0 | 5550 | 10.2% | inter-layer span |
| others | 7822 | 14.5% | QKV, attention, layer barrier, tail |

The MoE phase moves 52.7 MiB in 12.9 us, i.e. **4.27 TB/s, 53% of the 8 TB/s
peak** -- genuinely bandwidth-sensitive. But it is only 23.8% of the layer, so
crediting the *whole* phase with the 1.11x that AID-local placement is worth
against an interleaved baseline gives 1.782 ms, **+2.4%**. Even the 1.235x
inverted-control contrast -- which is not achievable, being local-vs-remote
rather than local-vs-interleaved -- only reaches 1.742 ms. Both are upper
bounds: they credit the full memory speedup to SwiGLU, LDS traffic and MFMA
too, none of which speed up.

**Caveat that matters more than the number.** The 1.10x/1.23x above were
measured by `aid_gemv` at **296.9 GB/s**. The real MoE phase runs at
**4270 GB/s** -- a 14x different operating point. Those figures come from a
regime where the memory system is nearly idle and latency dominates; at 53% of
peak the inter-AID interconnect is loaded and the penalty may be larger or
smaller. Until the local-vs-remote gap is re-measured at ~4 TB/s, the 2.4%
ceiling should be read as provisional.

For scale, the same breakdown puts **38% of every layer in barriers and
inter-layer wait** (13.6 + 14.0 + 10.2), and Phase 7 alone is as expensive as
the entire MoE. Those are far larger targets than AID locality.

## The gap at fleet's real operating point (`aid_moe_bw.hip`)

Every local/remote figure above -- 1.14x on NT streaming, 1.10x and 1.23x on
`aid_gemv` -- was measured with the memory system nearly idle. `aid_gemv`'s
shipped phase runs at **296.9 GB/s**; fleet's MoE phase runs at **4270 GB/s**.
That is a 14x different operating point, so the gap was re-measured on a stream
shaped like the MoE itself: 200192-byte MXFP4 slabs read exactly once, the
invariant map's `w < 23 -> AID0` assignment, and a 509 MiB working set so the
224 MiB L3 cannot absorb it.

The arms matter. Comparing a pinned VMM buffer against `hipMalloc` confounds
*homing* with *page size*, because VMM forces 4 KiB mappings while `hipMalloc`
gets 2 MiB pages. So there is a fourth arm: `VMM-mixed`, the same VMM machinery
with pages dealt alternately instead of by AID.

| grid | hipMalloc | VMM-mixed | VMM-local | VMM-remote | loc/mix | loc/rem |
|---|---|---|---|---|---|---|
| 240 | 1422 | 1415 | 1970 | 1610 | 1.392x | 1.224x |
| 960 | 5404 | 4814 | 4921 | 2881 | **1.022x** | 1.708x |
| 1920 | 5480 | 4462 | 4525 | 2811 | **1.014x** | 1.610x |
| 3840 | 5981 | 4518 | 4832 | 2916 | **1.070x** | 1.657x |

Two results, and they point the same way.

**The remote penalty grows under load, but the achievable gain shrinks.**
AID-local beats AID-remote by 1.61-1.71x here, far worse than the 1.23x seen at
297 GB/s -- so a loaded inter-AID link really does hurt more. But it beats a
50/50 mix by only **1.02-1.07x**, and 50/50 is what `hipMalloc` already gives.
Spreading traffic balances it across both AIDs' HBM stacks and both directions
of the link; at saturation the bottleneck is aggregate HBM bandwidth, not the
interconnect, so removing crossings buys almost nothing. The all-remote column
is a pathology no allocator produces.

**The mechanism costs 32%.** VMM-mixed (4518) against `hipMalloc` (5981) is the
price of 130K separate 4 KiB mappings on the TLB -- an order of magnitude more
than the 2-7% locality is worth. Folding both into the MoE phase's 23.8% share
of the layer:

| | end to end |
|---|---|
| locality alone, if 4 KiB pages were free | 1.825 -> 1.80 ms (+0.5% to +1.6%) |
| locality as actually implementable | 1.825 -> ~1.93 ms (**5% slower**) |

So AID-pinning the MoE weights through the VMM path would make fleet slower,
independently of the 3.7 hours of startup it would cost. The grid-240 row does
invert (1.392x), but it only reaches 1970 GB/s; fleet's 4270 GB/s sits in the
saturated regime, so the 1.02x rows are the ones that apply.

This retires the pinning line. What survives is the tile map, which is free and
is a prerequisite for anything that later wants a stable AID per workgroup --
partition modes included, since NPS2 gives AID-local memory in hardware with no
4 KiB page penalty and no startup cost.

## Locality with the mechanism cost removed (`aid_pure_locality.hip`)

Measuring locality *through* the VMM path was a methodological error: it charged
the 4 KiB TLB penalty to locality's account and capped every arm before locality
could show anything. `aid_pure_locality` removes the confound. A plain
`hipMalloc` buffer already has mixed homing, so instead of moving pages to match
the readers it classifies the pages and moves the *readers* to match the pages:
each XCD group gathers only pages already homed on its own AID. Ordinary 2 MiB
backing, no VMM, no handles, no startup cost, and all arms share one gather
kernel so the only difference is homing.

Sweeping the remote fraction is the point, because AID count sets where a part
sits on this curve (grid 1920, 1 GiB, reproduced on two different GPUs):

| remote | GB/s | gain from moving to 0% | |
|---|---|---|---|
| 0% | 5190 | 1.000x | |
| 25% | 5184 | 1.001x | |
| 50% | 4947 | **1.049x** | MI350 SPX/NPS1 -- this part |
| 75% | 3810 | **1.362x** | MI300X SPX/NPS1 -- the paper's baseline |
| 100% | 2965 | 1.751x | no allocator produces this |

The curve is strongly non-linear, and that single fact reconciles the paper with
these measurements. MI300X has 4 AIDs, so 75% of its traffic is remote and it
sits on the steep part with 1.36x available; GTAC'25 captured 20% of that. This
part has 2 AIDs and starts at 50% remote, where the curve is still flat and only
1.05x exists to be captured. The technique is sound in both cases -- the
headroom is a property of AID count, not of the method.

Note this also corrects the "5% slower" row above: the 32% was the VMM
mechanism, not locality. Locality is worth ~1.05x here; it is the *delivery* of
it that has, so far, cost more than it returns.

## Why no software scheme can deliver it under NPS1 (`aid_interleave.hip`)

The obvious way to avoid remapping entirely is to leave pages where they are and
choose the *reader* instead: pick which XCD reads which expert so that every
workgroup reads a slab already homed on its own AID. That requires slabs to be
single-AID to begin with. They are not.

Classifying every 4 KiB page of a `hipMalloc` buffer and taking run lengths:

| run | count share of buffer |
|---|---|
| 2 pages (8 KiB) | 51.3% |
| 3 pages (12 KiB) | 25.3% |
| 1 page (4 KiB) | 15.1% |
| 4 pages (16 KiB) | 6.0% |
| 5-6 pages | 2.3% |

Longest run anywhere: **6 pages (24 KiB)**. Homing is a deterministic hash of the
physical address, not a stripe -- the 42.9/57.1 split and every run-length share
reproduce to two decimals across a 512 MiB and a 2 GiB buffer. Aligned-block
purity collapses immediately: 100% at 4 KiB, 42.9% at 8 KiB, and **0.0% at
16 KiB and above**.

A fleet MoE workgroup slab is 200192 B, about 49 pages. **0 of 10699 slabs are
single-AID.** Not a small fraction -- zero, and necessarily so, since a 49-page
span cannot fit inside a 6-page run.

So under NPS1 no tile map, expert assignment, or reader-steering scheme can
produce an AID-local weight read, because no contiguous weight slab is AID-local
to begin with. This is also why the AID-invariant tile map measured neutral
(+0.30%, inside run-to-run noise): it was never in a position to matter. Only
three mechanisms remain, and two are worse than the 1.05x they chase:

| mechanism | status |
|---|---|
| VMM 4 KiB page remap | costs 32% to buy 5% -- retired |
| kernel-side page-granular gather | needs indirection in the weight load path |
| NPS2 hardware repartition | free at runtime, but see below |

## What NPS2 actually costs (`aid_nps_compare.hip`)

The partition modes were measured directly, which settles the mechanism question
without any of the software workarounds above. Getting there needs a host-side
`amdgpu` reload -- the mode only commits on driver reload, all 8 cards are one
XGMI hive, and from inside a container it cannot be done at all (`amd-smi reset
-r` returns `AMDSMI_STATUS_AMDGPU_RESTART_ERR`, `modprobe -r amdgpu` exits 1
with `refcnt` at 1192; `cap_sys_module` is in the bounding set but the module
belongs to the host).

Comparing the modes fairly takes care, because they do not offer the same
machine: NPS2 forces DPX or narrower, so a partition has 4 XCDs and 126 GiB
against SPX's 8 and 252. `aid_nps_compare` therefore runs work on **XCDs 0-3
only in both modes** -- blocks landing on 4-7 exit -- and sizes the grid
`nxcd*K` so every XCD gets K blocks. Active workers, bytes touched and access
pattern are then identical, and homing is the only thing left that differs. Same
binary, same card:

| workers/XCD | NPS1 latency | NPS2 latency | NPS1 GB/s | NPS2 GB/s |
|---|---|---|---|---|
| 64 | 400.0 ns | **340.8 ns** | **3533** | 1802 |
| 128 | 401.9 ns | **352.6 ns** | **3541** | 1798 |
| 256 | 414.8 ns | **367.3 ns** | **3553** | 1782 |
| 512 | 417.0 ns | **395.5 ns** | **3537** | 1796 |

**Locality is worth 15-18% of memory latency.** 400 -> 341 ns, and the p90/p10
spread tightens from 1.05-1.18x to 1.03-1.09x: the bimodal near/far split of
NPS1 pages collapses to a single mode. This is the effect with no mechanism
attached at all, and it is the strongest evidence in this directory that AID
locality is real.

The bandwidth columns, by contrast, **do not measure locality and must not be
read as such.** NPS2 delivers 1.97x less at identical worker counts (3540 ->
1798), and full-card peak shows 5941 GB/s under SPX/NPS1 against 3540 for both
NPS2 partitions combined. But the two arms do not see the same memory system:
under NPS1 those 4 XCDs pull from all 8 HBM stacks, while NPS2 confines them to
their partition's 4. Halving the stacks predicts about the 2x observed on its
own, so this says nothing about crossings. An earlier version of this file
concluded "NPS2 costs half the bandwidth" from these columns; that was a
confound between stack count and locality, and it is withdrawn. See the next
section for the controlled measurement, which points the other way.

That is moot in any case, because **fleet cannot run under NPS2**. The MoE
kernel hardcodes 8 XCDs: `global_tile = tile_idx * 8 + xcd_id` cannot address
tiles with `global_tile % 8` in {4..7} when `xcd_id` only reaches 3, so half the
tile space is never claimed, and the barrier polls a fixed 8 per-XCD release
slots (`for (int _x = 0; _x < 8; _x++)`, `MOE_BAR_COUNTER_SLOT = 8`) whose
arrival count can then never be reached. It deadlocks. Making the tile decode
and barrier geometry XCD-count-parametric is a prerequisite for even trying, and
the numbers above say it would not be worth it.

The SPX/NPS1 decode baseline was re-confirmed at **1.821 ms** after two driver
reloads, against 1.825 ms before, so nothing here rests on a shifted baseline.

## Locality at constant stack count (`aid_stack_controlled.hip`)

To measure crossings rather than stack count, hold the stacks fixed. Everything
runs under SPX/NPS1 on XCDs 0-3, which are AID0, and those same 4 XCDs read
either only AID0 pages or only AID1 pages. Four XCDs and four stacks in both
arms, identical page counts and gather pattern; the crossing is the only
variable.

| K/XCD | local (AID0) | remote (AID1) | loc/rem |
|---|---|---|---|
| 32 | 2629 GB/s | 1978 | 1.329x |
| 64 | 2661 GB/s | 2040 | 1.305x |
| 128 | 2633 GB/s | 2035 | 1.293x |
| 256 | 2621 GB/s | 2037 | 1.287x |

**Locality is worth 1.29-1.33x of bandwidth once stack count is controlled** --
a real gain, not the loss the confounded comparison suggested.

Three ratios now exist in this directory and they are not contradictory; they
describe different situations, and picking the wrong one is how the earlier
mistakes happened:

| ratio | what it compares | when it applies |
|---|---|---|
| 1.05x | 50% remote -> 0%, 8 XCDs over all 8 stacks | **fleet**: every XCD active, whole memory system |
| 1.29x | 100% remote -> 0%, 4 XCDs over 4 stacks | a partition-shaped workload |
| 1.75x | 100% remote -> 0%, 8 XCDs over all 8 stacks | all-remote pathology, no allocator does this |

fleet's case is the first row: it runs all 8 XCDs against all 8 stacks, and
`hipMalloc` already gives it a ~50/50 split, so 1.05x is the headroom. The
larger ratios need a baseline that is further from local than anything fleet
actually has. Note also that NPS1-local at 4 stacks (2628 GB/s) and the NPS2
partition (1798) were measured with *different kernels* -- page-gather here,
contiguous stream there -- and are not comparable; settling that would need this
binary run under NPS2.
