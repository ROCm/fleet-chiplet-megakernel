# CDNA4 Instruction Set Architecture Reference

Full AMD CDNA4 (MI350-series, gfx950) ISA spec, split by chapter for targeted lookup. Open the file that matches your need rather than loading the whole spec.

Source: *CDNA4 Instruction Set Architecture: Reference Guide* (AMD, Aug 2025), converted from the official PDF.

**Which reference do I want?** For *memory coherency, cache control, and synchronization* specifically — cache flags (`sc0`/`sc1`/`nt`), `buffer_wbl2`/`buffer_inv`, wait counters, fences, atomics, buffer descriptors, cross-lane ops — the curated [../memory-and-sync.md](../memory-and-sync.md) is more practical, with worked HIP+asm examples. For everything else (any compute/ALU/matrix instruction) and for the authoritative per-opcode semantics or binary encoding of *any* instruction, use this collection.

| File | Use when you need |
|------|-------------------|
| [CDNA4 ISA: Programming Model & Kernel State](00-programming-model.md) | Architecture model, wavefronts, GPRs/LDS allocation, EXEC mask, status/mode registers, traps, branching, and dependency/wait-state rules. |
| [CDNA4 ISA: Scalar & Vector ALU Operations](01-scalar-vector-alu.md) | How SALU and VALU instructions work: operands, data types, carry/SCC, literal constants, DPP/SDWA modifiers, rounding and denormal behavior. |
| [CDNA4 ISA: Matrix Arithmetic (MFMA)](02-matrix-mfma.md) | Matrix-fused-multiply-add (MFMA/WMMA) register layouts, supported data types (FP4/FP6/FP8/BF16/etc.), sparsity, and scale/conversion semantics. |
| [CDNA4 ISA: Memory Operations (Scalar, Vector, Flat, LDS)](03-memory-operations.md) | Memory access model for scalar memory, vector memory, flat/global/scratch, and the Local Data Share — addressing, coherency, and out-of-range rules. |
| [CDNA4 ISA Instructions: Scalar (SOP2/SOPK/SOP1/SOPC/SOPP)](04-instr-scalar.md) | Per-opcode reference for scalar ALU and control instructions: the S_* family (arith, logic, bitfield, branches, waits, messages). |
| [CDNA4 ISA Instructions: Scalar Memory (SMEM)](05-instr-smem.md) | Per-opcode reference for scalar memory loads/stores/atomics (S_LOAD_*, S_STORE_*, S_ATOMIC_*, cache/scope control). |
| [CDNA4 ISA Instructions: Vector (VOP2/VOP1/VOPC)](06-instr-vector.md) | Per-opcode reference for the common vector ALU encodings: 2-operand, 1-operand, and vector compares (V_*). |
| [CDNA4 ISA Instructions: Packed Vector (VOP3P)](07-instr-vop3p.md) | Per-opcode reference for packed-math VOP3P instructions (dual 16-bit / packed FP ops, V_PK_* and friends). |
| [CDNA4 ISA Instructions: VOP3A & VOP3B](08-instr-vop3.md) | Per-opcode reference for the 3-operand vector encoding — the largest instruction family (transcendentals, conversions, bitfield, FMA, etc.). |
| [CDNA4 ISA Instructions: LDS / Data Share (DS)](09-instr-lds.md) | Per-opcode reference for Local Data Share instructions (DS_*): loads, stores, atomics, permutes, and append/consume. |
| [CDNA4 ISA Instructions: Buffer & Flat (MUBUF/MTBUF/FLAT/Scratch/Global)](10-instr-buffer-flat.md) | Per-opcode reference for typed/untyped buffer access and flat/global/scratch memory instructions, plus instruction limitations. |
| [CDNA4 ISA: Microcode Formats (Binary Encodings)](11-microcode-formats.md) | Bit-field layouts for every instruction encoding (SOP2, VOP3, MUBUF, ...): field names, bit ranges, operand codes, and opcode tables. |

## Finding an instruction

Chapter 12 (the per-opcode reference) is organized by **encoding family**, not alphabetically. Each instruction file lists its mnemonics at the top. To locate an opcode by name, grep the collection:

```bash
grep -rl '#### V_FMA_F32' .agents/references/gpu-architectures/cdna4/isa-manual/
```

Rough guide to families: `S_*` scalar ALU/control → `04-instr-scalar`; `S_LOAD/S_STORE/S_ATOMIC` → `05-instr-smem`; common `V_*` → `06-instr-vector`; packed `V_PK_*` → `07-instr-vop3p`; 3-operand `V_*` (most transcendentals, conversions) → `08-instr-vop3`; `DS_*` → `09-instr-lds`; `BUFFER_*`/`TBUFFER_*`/`FLAT_*`/`GLOBAL_*`/`SCRATCH_*` → `10-instr-buffer-flat`. Binary bit-field encodings for all families are in `11-microcode-formats`.

