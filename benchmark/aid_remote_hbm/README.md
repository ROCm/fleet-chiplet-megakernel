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

# Can an allocation be pinned to the local AID? (`aid_page_map.hip`)

No, not in the mode this GPU runs in. `aid_page_map` times a dependent chain of
agent-scope atomics that stays on one address, so each step is a fabric round
trip to whatever owns that address. Plain loads are useless here: they sit in
L1/L2 at ~56 ns and hide the topology entirely. Atomics land at ~94 ns.

Measured on an idle GPU (`0000:06:00.0`), latency in ns from every chiplet to a
fixed address:

| offset    | XCD0 | XCD1 | XCD2 | XCD3 | XCD4 | XCD5 | XCD6 | XCD7 | AID0 avg | AID1 avg |
|-----------|------|------|------|------|------|------|------|------|----------|----------|
| 0 KiB     | 99   | 101  | 97   | 88   | 99   | 97   | 88   | 99   | 96.1     | 95.6     |
| 4 KiB     | 95   | 97   | 95   | 84   | 97   | 95   | 84   | 97   | 92.9     | 93.4     |
| 1 MiB     | 99   | 101  | 97   | 88   | 99   | 97   | 88   | 99   | 96.1     | 95.7     |
| 17 MiB    | 101  | 102  | 99   | 88   | 101  | 99   | 88   | 101  | 97.4     | 97.0     |
| 129 MiB   | 99   | 101  | 97   | 87   | 99   | 97   | 88   | 99   | 95.9     | 95.6     |
| 400 MiB   | 101  | 102  | 99   | 87   | 101  | 99   | 88   | 101  | 97.2     | 97.0     |

The two AIDs agree to within 0.5 ns on every address. The spread that does exist
(88-102 ns) repeats with period 4 across the chiplets and is identical for all
six addresses, so it is a property of the chiplet, not of where the data lives.

Sweeping 128 addresses at 256 B, 4 KiB and 2 MiB strides gives the same answer:
the XCD0-vs-XCD4 delta is 0 +/- 2 ns with no bimodality, no single address bit
correlates with it, and the same-AID XCD0-vs-XCD1 control is just as flat.

So under NPS1 every address is equidistant from both AIDs: physical memory is
interleaved across all stacks on both AIDs at a granularity finer than a cache
line. No page coloring, no `hipMalloc` trick, and no address-bit filter can make
an allocation AID-local.

## What would actually work

`amd-smi partition --accelerator` on this part:

| compute mode | XCCs per partition | memory modes |
|--------------|--------------------|--------------|
| SPX (current)| 8                  | NPS1 only    |
| DPX          | 4                  | NPS1, NPS2   |
| QPX          | 2                  | NPS1, NPS2   |
| CPX          | 1                  | NPS1, NPS2   |

AID-local memory requires NPS2, and NPS2 is not offered under SPX. Getting
locality therefore means giving up the single-device view and splitting compute
too, which is exactly the trade the Starscream paper takes on: it runs CPX/NPS4
and then has to repair the tensor-parallel partitioning that the split breaks.
Switching modes is a `sudo amd-smi set -M` reconfiguration of an idle GPU, not
something a process can request for one allocation.
