---
name: cdna4-expert
description: AMD CDNA4 (gfx950 / MI350X) architecture expert. Covers hardware overview (XCDs, CUs, LDS, register file, cache hierarchy, MFMA throughput), memory coherency and synchronization (cache flags sc0/sc1/nt, cache management buffer_wbl2/buffer_inv, waits, fences, atomics, buffer descriptors V#, cross-lane ops) with worked HIP+asm examples, and the full CDNA4 ISA manual (per-opcode semantics and binary encodings). Use when writing, reviewing, optimizing, or debugging HIP or assembly for CDNA4, or when you need authoritative ISA or hardware facts.
---

# CDNA4 Expert (gfx950 / MI350X)

Authoritative knowledge for AMD CDNA4 GPUs. Route to the right reference:

- Hardware overview (XCD/CU layout, LDS, register file, cache hierarchy, MFMA throughput, key optimization parameters): [`hardware-overview.md`](../../references/gpu-architectures/cdna4/hardware-overview.md)
- Memory coherency, cache control flags (`sc0`/`sc1`/`nt`), cache management (`buffer_wbl2`/`buffer_inv`), fences, waits, atomics, buffer descriptors (V#), and cross-lane ops -- curated with worked HIP+asm examples: [`memory-and-sync.md`](../../references/gpu-architectures/cdna4/memory-and-sync.md)
- Any other instruction (scalar/vector ALU, transcendentals, conversions, matrix/MFMA) or the authoritative per-opcode semantics and binary encoding of any instruction -- the full ISA manual: [`isa-manual/README.md`](../../references/gpu-architectures/cdna4/isa-manual/README.md)
- Cross-architecture comparison (CDNA3 vs CDNA4 vs RDNA3 vs RDNA4): [`comparison.md`](../../references/gpu-architectures/comparison.md)
