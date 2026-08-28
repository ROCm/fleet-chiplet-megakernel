# Local vs remote AID access on MI350/MI355

This machine is one MI350 package: 8 XCDs (chiplets) grouped onto 2 AIDs, with
the HBM stacks attached to the AIDs. A chiplet reaching memory owned by the
*other* AID crosses the inter-AID interconnect, which costs latency and
interconnect power. This is the effect GTAC'25 paper 48 (Starscream) motivates
its CPX/NPS4 partitioning with; the paper measures it on MI300X (4 AIDs), where
75% of a CU's traffic is remote under SPX/NPS1.

## Build and run

```bash
hipcc -O3 -std=c++17 --offload-arch=gfx950 \
  benchmark/aid_remote_hbm/aid_remote_hbm.hip -o /tmp/aid_remote_hbm
HIP_VISIBLE_DEVICES=0 /tmp/aid_remote_hbm
```

The benchmark reads `HW_REG_XCC_ID` to learn which chiplet each block landed on,
so every phase runs only on the chiplet it names. The ping-pong flag is
allocated with `hipDeviceMallocUncached` so the round trip actually reaches
HBM/fabric instead of hitting L2.

## Result (GPU `0000:76:00.0`, NPS1/SPX, idle 239 W)

Uncached HBM flag ping-pong, one wave per chiplet, 2M rounds:

| pair        | topology  | ns/round    | socket power | wall   |
|-------------|-----------|-------------|--------------|--------|
| XCD0 ↔ XCD1 | same AID  | 768         | 281 W        | 1.54 s |
| XCD0 ↔ XCD4 | other AID | 961 (1.25×) | 284 W        | 1.92 s |

Crossing to the other AID costs **25% more latency** per round trip.

Socket `power1_input` is nearly flat between the two cases (281 W vs 284 W).
Two spinning waves are far too small a load to move a 1000 W package rail, and
the sysfs counter is whole-socket, not per-AID, so this benchmark cannot resolve
the paper's AID-power claim. The latency gap is the clean signal here.

## STREAM does not split under NPS1

Phases 2 and 3 stream 4 GiB from XCD 0 with non-temporal loads, selecting
address subsets by 256 B channel group and by 2 MiB page parity:

| selection                | GB/s  |
|--------------------------|-------|
| all stacks               | 17.5  |
| stacks 0-3 / stacks 4-7  | 8.6 / 8.6 |
| even / odd 2 MiB pages   | 142.3 / 143.9 |

Neither coloring separates the AIDs. NPS1 hashes physical addresses across all
stacks on both AIDs, so no address-bit trick pins a buffer to one AID's memory —
that needs a memory partitioning mode (NPS2 on this part), which is a
driver-level reconfiguration of the whole GPU.

The low absolute bandwidth in phase 2 is expected: it is one chiplet out of
eight, and the strided variants touch one 256 B line per 2 KiB of address space,
so they are latency-bound rather than bandwidth-bound.
