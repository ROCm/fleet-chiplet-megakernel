# CDNA4 ISA: Programming Model & Kernel State

> Part of the **CDNA4 Instruction Set Architecture** reference. See the [index](README.md) for other sections.

## Sections in this file

- [Preface](#preface)
  - [About This Document](#about-this-document)
  - [Audience](#audience)
  - [Organization](#organization)
  - [Conventions](#conventions)
  - [Contact Information](#contact-information)
- [Chapter 1. Introduction](#chapter-1-introduction)
  - [1.1. Terminology](#11-terminology)
- [Chapter 2. Program Organization](#chapter-2-program-organization)
  - [2.1. Compute Shaders](#21-compute-shaders)
  - [2.2. Data Sharing](#22-data-sharing)
  - [2.3. Device Memory](#23-device-memory)
- [Chapter 3. Kernel State](#chapter-3-kernel-state)
  - [3.1. State Overview](#31-state-overview)
  - [3.2. Program Counter (PC)](#32-program-counter-pc)
  - [3.3. EXECute Mask](#33-execute-mask)
  - [3.4. Status registers](#34-status-registers)
  - [3.5. Mode register](#35-mode-register)
  - [3.6. GPRs and LDS](#36-gprs-and-lds)
  - [3.7. M0 Memory Descriptor](#37-m0-memory-descriptor)
  - [3.8. SCC: Scalar Condition code](#38-scc-scalar-condition-code)
  - [3.9. Vector Compares: VCC and VCCZ](#39-vector-compares-vcc-and-vccz)
  - [3.10. Trap and Exception registers](#310-trap-and-exception-registers)
  - [3.11. Memory Violations](#311-memory-violations)
  - [3.12. Hardware ID Registers](#312-hardware-id-registers)
  - [3.13. GPR Initialization](#313-gpr-initialization)
- [Chapter 4. Program Flow Control](#chapter-4-program-flow-control)
  - [4.1. Program Control](#41-program-control)
  - [4.2. Branching](#42-branching)
  - [4.3. Workgroups](#43-workgroups)
  - [4.4. Data Dependency Resolution](#44-data-dependency-resolution)
  - [4.5. Manually Inserted Wait States (NOPs)](#45-manually-inserted-wait-states-nops)
  - [4.6. Arbitrary Divergent Control Flow](#46-arbitrary-divergent-control-flow)

---

## Preface

### About This Document

This document describes the current environment, organization and program state of AMD CDNA4 devices. It
details the instruction set and the microcode formats native to this family of processors that are accessible to
programmers and compilers.

The document specifies the instructions (including the format of each type of instruction) and the relevant
program state (including how the program state interacts with the instructions). Some instruction fields are
mutually dependent; not all possible settings for all fields are legal. This document specifies the valid
combinations.

The main purposes of this document are to:

```
 1. Specify the language constructs and behavior, including the organization of each type of instruction in
    both text syntax and binary format.
 2. Provide a reference of instruction operation that compiler writers can use to maximize performance of the
    processor.
```

### Audience

This document is intended for programmers writing application and system software, including operating
systems, compilers, loaders, linkers, device drivers, and system utilities. It assumes that programmers are
writing compute-intensive parallel applications (streaming applications) and assumes an understanding of
requisite programming practices.

### Organization

This document begins with an overview of the AMD CDNA processors' hardware and programming
environment (Chapter 1).
Chapter 2 describes the organization of CDNA programs.
Chapter 3 describes the program state that is maintained.
Chapter 4 describes the program flow.
Chapter 5 describes the scalar ALU operations.
Chapter 6 describes the vector ALU operations.
Chapter 7 describes the vector Matrix ALU operations.
Chapter 8 describes the scalar memory operations.
Chapter 9 describes the vector memory operations.
Chapter 10 provides information about the flat memory instructions.
Chapter 11 describes the data share operations.
Chapter 12 describes instruction details, first by the microcode format to which they belong, then in alphabetic
order.
Finally, Chapter 13 provides a detailed specification of each microcode format.

### Conventions

The following conventions are used in this document:

```
mono-spaced font                     A filename, file path or code.
*                                    Any number of alphanumeric characters in the name of a code format, parameter, or
                                     instruction.
<>                                   Angle brackets denote streams.
[1,2)                                A range that includes the left-most value (in this case, 1), but excludes the right-most
                                     value (in this case, 2).
[1,2]                                A range that includes both the left-most and right-most values.
{x | y}                              One of the multiple options listed. In this case, X or Y.
0.0                                  A single-precision (32-bit) floating-point value.
1011b                                A binary value, in this example a 4-bit value.
7:4                                  A bit range, from bit 7 to bit 4, inclusive. The high-order bit is shown first.
italicized word or phrase            The first use of a term or concept basic to the understanding of stream computing.
```

### Contact Information

For information concerning AMD Accelerated Parallel Processing development, please see:
http://developer.amd.com/

## Chapter 1. Introduction

AMD CDNA processors implement a parallel micro-architecture that is designed to provide an excellent
platform for general-purpose data parallel applications. Data-intensive applications that require high
bandwidth or are computationally intensive are a candidate for running on an AMD CDNA processor.

The figure below shows a block diagram of the AMD CDNA Generation series processors

**Figure 1. AMD CDNA Generation Series Block Diagram**

The CDNA device includes a data-parallel processor (DPP) array, a command processor, a memory controller,
and other logic (not shown). The CDNA command processor reads commands that the host has written to
memory-mapped CDNA registers in the system-memory address space. The command processor sends
hardware-generated interrupts to the host when the command is completed. The CDNA memory controller has
direct access to all CDNA device memory and the host-specified areas of system memory. To satisfy read and
write requests, the memory controller performs the functions of a direct-memory access (DMA) controller,
including computing memory-address offsets based on the format of the requested data in memory. In the
CDNA environment, a complete application includes two parts:

- a program running on the host processor, and
- programs, called kernels, running on the CDNA processor.

The CDNA programs are controlled by host commands that

- set CDNA internal base-address and other configuration registers,
- specify the data domain on which the CDNA Accelerator is to operate,
- invalidate and flush caches on the CDNA Accelerator, and
- cause the CDNA Accelerator to begin execution of a program.

The CDNA driver program runs on the host.

The DPP array is the heart of the CDNA processor. The array is organized as a set of compute unit pipelines,
each independent from the others, that are designed to operate in parallel on streams of floating-point or
integer data. The compute unit pipelines can process data or, through the memory controller, transfer data to,
or from, memory. Computation in a compute unit pipeline can be made conditional. Outputs written to
memory can also be made conditional.

When it receives a request, the compute unit pipeline loads instructions and data from memory, begins
execution, and continues until the end of the kernel. As kernels are running, the CDNA hardware automatically
fetches instructions from memory into on-chip caches; CDNA software plays no role in this. CDNA kernels can
load data from off-chip memory into on-chip general-purpose registers (GPRs) and caches.

The AMD CDNA devices can detect floating point exceptions and can generate interrupts. In particular, they
can detect IEEE floating-point exceptions in hardware; these can be recorded for post-execution analysis. The
software interrupts shown in the previous figure from the command processor to the host represent hardware-
generated interrupts for signaling command-completion and related management functions.

The CDNA processor is designed to hide memory latency by keeping track of potentially hundreds of work-
items in different stages of execution, and by overlapping compute operations with memory-access operations.

### 1.1. Terminology

**Table 1. Basic Terms**

```
Term               Description
CDNA Processor     The shader processor is a scalar and vector ALU capable of running complex programs on behalf of a
                   wavefront.
Dispatch           A dispatch launches a 1D, 2D, or 3D grid of work to the CDNA processor array.
Workgroup          A workgroup is a collection of wavefronts that have the ability to synchronize with each other quickly;
                   they also can share data through the Local Data Share.
Wavefront          A collection of 64 work-items that execute in parallel on a single CDNA processor.
Work-item          A single element of work: one element from the dispatch grid, or in graphics a pixel or vertex.
Literal Constant   A 32-bit integer or float constant that is placed in the instruction stream.
Scalar ALU (SALU) The scalar ALU operates on one value per wavefront and manages all control flow.
Vector ALU (VALU) The vector ALU maintains Vector GPRs that are unique for each work item and execute arithmetic
                  operations uniquely on each work-item.
Microcode format   The microcode format describes the bit patterns used to encode instructions. Each instruction is
                   either 32 or 64 bits.
Instruction        An instruction is the basic unit of the kernel. Instructions include: vector ALU, scalar ALU, memory
                   transfer, and control flow operations.
Buffer Resource    A buffer resource descriptor describes a buffer in memory: address, data format, stride, etc.
(V#)
```

## Chapter 2. Program Organization

CDNA kernels are programs executed by the CDNA processor. Conceptually, the kernel is executed
independently on every work-item, but in reality the CDNA processor groups 64 work-items into a wavefront,
which executes the kernel on all 64 work-items in one pass.

The CDNA processor consists of:

- A scalar ALU, which operates on one value per wavefront (common to all work items).
- A vector ALU, which operates on unique values per work-item.
- Local data storage, which allows work-items within a workgroup to communicate and share data.
- Scalar memory, which can transfer data between SGPRs and memory through a cache.
- Vector memory, which can transfer data between VGPRs and memory

All kernel control flow is handled using scalar ALU instructions. This includes if/else, branches and looping.
Scalar ALU (SALU) and memory instructions work on an entire wavefront and operate on up to two SGPRs, as
well as literal constants.

Vector memory and ALU instructions operate on all work-items in the wavefront at one time. In order to
support branching and conditional execute, every wavefront has an EXECute mask that determines which
work-items are active at that moment, and which are dormant. Active work-items execute the vector
instruction, and dormant ones treat the instruction as a NOP. The EXEC mask can be changed at any time by
Scalar ALU instructions.

Vector ALU instructions can take up to three arguments, which can come from VGPRs, SGPRs, or literal
constants that are part of the instruction stream. They operate on all work-items enabled by the EXEC mask.
Vector compare and add with- carryout return a bit-per-work-item mask back to the SGPRs to indicate, per
work-item, which had a "true" result from the compare or generated a carry-out.

Vector memory instructions transfer data between VGPRs and memory. Each work-item supplies its own
memory address and supplies or receives unique data. These instructions are also subject to the EXEC mask.

### 2.1. Compute Shaders

Compute kernels (shaders) are generic programs that can run on the CDNA processor, taking data from
memory, processing it, and writing results back to memory. Compute kernels are created by a dispatch, which
causes the CDNA processors to run the kernel over all of the work-items in a 1D, 2D, or 3D grid of data. The
CDNA processor walks through this grid and generates wavefronts, which then run the compute kernel. Each
work-item is initialized with its unique address (index) within the grid. Based on this index, the work-item
computes the address of the data it is required to work on and what to do with the results.

### 2.2. Data Sharing

The AMD CDNA stream processors can share data between different work-items. Data sharing can significantly
boost performance. The figure below shows the memory hierarchy that is available to each work-item.

**Figure 2. Shared Memory Hierarchy**

#### 2.2.1. Local Data Share (LDS)

Each compute unit has a 160 kB memory space that enables low-latency communication between work-items
within a work-group, or the work-items within a wavefront; this is the local data share (LDS). This memory is
configured with 64 banks, each with 640 entries of 4 bytes. The shared memory contains 32 integer atomic
units designed to enable fast, unordered atomic operations. This memory can be used as a software cache for
predictable re-use of data, a data exchange machine for the work-items of a work-group, or as a cooperative
way to enable efficient access to off-chip memory.

### 2.3. Device Memory

The AMD CDNA devices offer several methods for access to off-chip memory from the processing elements
(PE) within each compute unit. On the primary read path, the device consists of multiple channels of L2 read-
write cache that provides data to an L1 cache for each compute unit. Specific cache-less load instructions can
force data to be retrieved from device memory during an execution of a load clause. Load requests that overlap
within the clause are cached with respect to each other. The output cache is formed by two levels of cache: the
first for write-combining cache (collect scatter and store operations and combine them to provide good access
patterns to memory); the second is a read/write cache with atomic units that lets each processing element
complete unordered atomic accesses that return the initial value. Each processing element provides the
destination address on which the atomic operation acts, the data to be used in the atomic operation, and a
return address for the read/write atomic unit to store the pre-op value in memory. Each store or atomic
operation can be set up to return an acknowledgment to the requesting PE upon write confirmation of the
return value (pre-atomic op value at destination) being stored to device memory.

This acknowledgment has two purposes:

- enabling a PE to recover the pre-op value from an atomic operation by performing a cache-less load from its return address after receipt of the write confirmation acknowledgment, and
- enabling the system to maintain a relaxed consistency model.

Each scatter write from a given PE to a given memory channel maintains order. The acknowledgment enables
one processing element to implement a fence to maintain serial consistency by ensuring all writes have been
posted to memory prior to completing a subsequent write. In this manner, the system can maintain a relaxed
consistency model between all parallel work-items operating on the system.

## Chapter 3. Kernel State

This chapter describes the kernel states visible to the shader program.

### 3.1. State Overview

The table below shows all of the hardware states readable or writable by a shader program.

**Table 2. Readable and Writable Hardware States**

```
Abbrev.               Name                            Size      Description
                                                      (bits)
PC                    Program Counter                 48        Points to the memory address of the next shader
                                                                instruction to execute.
V0-V255               VGPR                            32        Vector general-purpose register ("architectural VGPRs").
AV0-AV255             AccVGPR                         32        Matrix Accumulation Vector general-purpose register.
S0-S103               SGPR                            32        Vector general-purpose register.
LDS                   Local Data Share                160kB     Local data share is a scratch RAM with built-in
                                                                arithmetic capabilities that allow data to be shared
                                                                between threads in a workgroup.
EXEC                  Execute Mask                    64        A bit mask with one bit per thread, which is applied to
                                                                vector instructions and controls that threads execute and
                                                                that ignore the instruction.
EXECZ                 EXEC is zero                    1         A single bit flag indicating that the EXEC mask is all
                                                                zeros.
VCC                   Vector Condition Code           64        A bit mask with one bit per thread; it holds the result of a
                                                                vector compare operation.
VCCZ                  VCC is zero                     1         A single bit-flag indicating that the VCC mask is all zeros.
SCC                   Scalar Condition Code           1         Result from a scalar ALU comparison instruction.
FLAT_SCRATCH          Flat scratch address            64        The 64-bit base address of scratch memory, in
                                                                NumSGPRs-5 and -6. Read Only.
XNACK_MASK            Address translation failure.    64        Bit mask of threads that have failed their address
                                                                translation.
STATUS                Status                          32        Read-only shader status bits.
MODE                  Mode                            32        Writable shader mode bits.
M0                    Memory Reg                      32        A temporary register that has various uses, including
                                                                GPR indexing and bounds checking.
HW_ID                 Hardware ID                     32        Read-only status register that has various wave ID state.
XCC_ID                Compute ID                      32        Read-only status register that contains the compute
                                                                device ID.
TRAPSTS               Trap Status                     32        Holds information about exceptions and pending traps.
TBA                   Trap Base Address               64        Holds the pointer to the current trap handler program.
TMA                   Trap Memory Address             64        Temporary register for shader operations. For example,
                                                                can hold a pointer to memory used by the trap handler.
TTMP0-TTMP15          Trap Temporary SGPRs            32        16 SGPRs available only to the Trap Handler for
                                                                temporary storage.
VMCNT                 Vector memory instruction count 6         Counts the number of VMEM instructions issued but not
                                                                yet completed.
EXPCNT                Export Count                    3         Unused
```

```
Abbrev.             Name                               Size      Description
                                                       (bits)
LGKMCNT             LDS, Constant and Message count 4            Counts the number of LDS, constant-fetch (scalar
                                                                 memory read), and message instructions issued but not
                                                                 yet completed.
```

### 3.2. Program Counter (PC)

The program counter (PC) is a byte address pointing to the next instruction to execute. When a wavefront is
created, the PC is initialized to the first instruction in the program.

The PC interacts with three instructions: S_GET_PC, S_SET_PC, S_SWAP_PC. These transfer the PC to, and
from, an even-aligned SGPR pair.

Branches jump to (PC_of_the_instruction_after_the_branch + offset). The shader program cannot directly read
from, or write to, the PC. Branches, GET_PC and SWAP_PC, are PC-relative to the next instruction, not the
current one. S_TRAP saves the PC of the S_TRAP instruction itself.

### 3.3. EXECute Mask

The Execute mask (64-bit) determines which threads in the vector are executed:
1 = execute, 0 = do not execute.

EXEC can be read from, and written to, through scalar instructions; it also can be written as a result of a vector-
ALU compare. This mask affects vector-ALU, vector-memory, and LDS instructions. It does not affect scalar
execution or branches.

A helper bit (EXECZ) can be used as a condition for branches to skip code when EXEC is zero.

> This Accelerator does no optimization when EXEC = 0. The shader hardware executes every
> instruction, wasting instruction issue bandwidth. Use CBRANCH or VSKIP to rapidly skip
> over code when it is likely that the EXEC mask is zero.

### 3.4. Status registers

Status register fields can be read, but not written to, by the shader. These bits are initialized at wavefront-
creation time. The table below lists and briefly describes the status register fields. The status register fields may
be written when PRIV=1. Some fields are set as a result of shader instructions.

**Table 3. Status Register Fields**

```
Field                       Bit          Description
                            Position
SCC                         1            Scalar condition code. Used as a carry-out bit. For a comparison instruction, this
                                         bit indicates failure or success. For logical operations, this is 1 if the result was
                                         non-zero.
```

```
Field                        Bit          Description
                             Position
SPI_PRIO                     2:1          Wavefront priority set by the shader processor interpolator (SPI) when the
                                          wavefront is created. See the S_SETPRIO instruction (page 12-49) for details. 0 is
                                          lowest, 3 is highest priority.
WAVE_PRIO                    4:3          Wavefront priority set by the shader program. See the S_SETPRIO instruction
                                          (page 12-49) for details.
PRIV                         5            Privileged mode. Can only be active when in the trap handler. Gives write access
                                          to the TTMP, TMA, and TBA registers.
TRAP_EN                      6            Indicates that a trap handler is present. When set to zero, traps are not taken.
EXECZ                        9            Exec mask is zero.
VCCZ                         10           Vector condition code is zero.
IN_TG                        11           Wavefront is a member of a work-group of more than one wavefront.
IN_BARRIER                   12           Wavefront is waiting at a barrier.
HALT                         13           Wavefront is halted or scheduled to halt. HALT can be set by the host through
                                          wavefront-control messages, or by the shader. This bit is ignored while in the
                                          trap handler (PRIV = 1); it also is ignored if a host-initiated trap is received
                                          (request to enter the trap handler).
TRAP                         14           Wavefront is flagged to enter the trap handler as soon as possible.
VALID                        16           Wavefront is active (has been created and not yet ended).
ECC_ERR                      17           An ECC error has occurred.
PERF_EN                      19           Performance counters are enabled for this wavefront.
COND_DBG_USER                20           Conditional debug indicator for user mode
COND_DBG_SYS                 21           Conditional debug indicator for system mode.
ALLOW_REPLAY                 22           Indicates that ATC replay is enabled.
FATAL_HALT                   23           Indicates a fatal halt has occurred.
SCRATCH_EN                   28           1 = wave has scratch space allocated; 0 = does not.
IDLE                         31           Indicates wave is idle - has no outstanding instructions.
```

### 3.5. Mode register

Mode register fields can be read from, and written to, by the shader through scalar instructions. The table
below lists and briefly describes the mode register fields.

**Table 4. Mode Register Fields**

```
Field                    Bit            Description
                         Position
FP_ROUND                 3:0            [1:0] Single precision round mode. [3:2] Double/Half precision round mode.
                                        Round Modes: 0=nearest even, 1= +infinity, 2= -infinity, 3= toward zero.
FP_DENORM                7:4            [1:0] Single precision denormal mode. [3:2] Double/Half precision denormal
                                        mode. Denorm modes:
                                        0 = flush input and output denorms.
                                        1 = allow input denorms, flush output denorms.
                                        2 = flush input denorms, allow output denorms.
                                        3 = allow input and output denorms.
DX10_CLAMP               8              Used by the vector ALU to force DX10-style treatment of NaNs: when set, clamp
                                        NaN to zero; otherwise, pass NaN through.
IEEE                     9              Floating point opcodes that support exception flag gathering quiet and propagate
                                        signaling NaN inputs per IEEE 754-2008. Min_dx10 and max_dx10 become IEEE
                                        754-2008 compliant due to signaling NaN propagation and quieting.
```

```
Field                      Bit          Description
                           Position
DEBUG                      11           Forces the wavefront to jump to the exception handler after each instruction is
                                        executed (but not after ENDPGM). Only works if TRAP_EN = 1.
EXCP_EN                    18:12        Enable mask for exceptions. Enabled means if the exception occurs and
                                        TRAP_EN==1, a trap is taken.
                                        [12] : invalid.
                                        [13] : inputDenormal.
                                        [14] : float_div0.
                                        [15] : overflow.
                                        [16] : underflow.
                                        [17] : inexact.
                                        [18] : int_div0.
                                        [19] : address watch
                                        [20] : memory violation
                                        [20] : trap on wave end
FP16_OVFL                  23           If set, an overflowed FP16 result is clamped to +/- MAX_FP16, regardless of round
                                        mode, while still preserving true INF values.
DISABLE_PERF               26           1 = disable performance counting for this wave
GPR_IDX_EN                 27           GPR index enable.
VSKIP                      28           0 = normal operation. 1 = skip (do not execute) any vector instructions: valu,
                                        vmem, lds. "Skipping" instructions occurs at high-speed (10 wavefronts per clock
                                        cycle can skip one instruction). This is much faster than issuing and discarding
                                        instructions.
CSP                        31:29        Conditional branch stack pointer.
```

### 3.6. GPRs and LDS

This section describes how GPR and LDS space is allocated to a wavefront, as well as how out-of-range and
misaligned accesses are handled.

#### 3.6.1. Out-of-Range behavior

This section defines the behavior when a source or destination GPR or memory address is outside the legal
range for a wavefront.

Out-of-range can occur through GPR-indexing or bad programming. It is illegal to index from one register type
into another (for example: SGPRs into trap registers or inline constants). It is also illegal to index within inline
constants.

The following describe the out-of-range behavior for various storage types.

- SGPRs
  - Source or destination out-of-range = (sgpr < 0 || (sgpr >= sgpr_size)).
  - Source out-of-range: returns the value of SGPR0 (not the value 0).
  - Destination out-of-range: instruction writes no SGPR result.
- VGPRs
  - Similar to SGPRs. It is illegal to index from SGPRs into VGPRs, or vice versa.
  - Out-of-range = (vgpr < 0 || (vgpr >= vgpr_size))

  - If a source VGPR is out of range, VGPR0 is used.
  - If a destination VGPR is out-of-range, the instruction is ignored (treated as an NOP).
- LDS
  - If the LDS-ADDRESS is out-of-range (addr < 0 or >= (MIN(lds_size, m0)):
    - Writes out-of-range are discarded; it is undefined if SIZE is not a multiple of write-data-size.
    - Reads return the value zero.
  - If any source-VGPR is out-of-range, use the VGPR0 value is used.
  - If the dest-VGPR is out of range, nullify the instruction (issue with exec=0)
- Memory, LDS: Reads and atomics with returns.
  - If any source VGPR or SGPR is out-of-range, the data value is undefined.
  - If any destination VGPR is out-of-range, the operation is nullified by issuing the instruction as if the EXEC mask were cleared to 0.
    - This out-of-range check must check all VGPRs that can be returned (for example: VDST to VDST+3 for a BUFFER_LOAD_DWORDx4).
    - Atomic operations with out-of-range destination VGPRs are nullified: issued, but with exec mask of zero.

Instructions with multiple destinations (for example: V_ADDC): if any destination is out-of-range, no results
are written.

#### 3.6.2. SGPR Allocation and storage

A wavefront can be allocated 16 to 102 SGPRs, in units of 16 GPRs (Dwords). These are logically viewed as
SGPRs 0-101. The VCC is physically stored as part of the wavefront's SGPRs in the highest numbered two SGPRs
(SGPR 106 and 107; the source/destination VCC is an alias for those two SGPRs). When a trap handler is present,
16 additional SGPRs are reserved after VCC to hold the trap addresses, as well as saved-PC and trap-handler
temps. These all are privileged (cannot be written to unless privilege is set). Note that if a wavefront allocates
16 SGPRs, 2 SGPRs are typically used as VCC, the remaining 14 are available to the shader. Shader hardware
does not prevent use of all 16 SGPRs.

#### 3.6.3. SGPR Alignment

Even-aligned SGPRs are required in the following cases.

- When 64-bit data is used. This is required for moves to/from 64-bit registers, including the PC.
- When scalar memory reads that the address-base comes from an SGPR-pair (either in SGPR).

Quad-alignment is required for the data-GPR when a scalar memory read returns four or more Dwords. When a
64-bit quantity is stored in SGPRs, the LSBs are in SGPR[n], and the MSBs are in SGPR[n+1].

#### 3.6.4. VGPR Allocation and Alignment

VGPRs are allocated in groups of eight Dwords.

VGPRs are allocated out of two pools: regular VGPRs and accumulation VGPRs. Accumulation VGPRs are used
with matrix VALU instructions, and can also be loaded directly from memory. A wave may have up to 512 total

VGPRs, 256 of each type. When a wave has fewer than 512 total VGPRs, the number of each type is flexible - it is
not required to be equal numbers of both types.

Instructions which operate on 64-bit data must use aligned (i.e. even) VGPRs. This applies to ALU and memory
instructions.

#### 3.6.5. LDS Allocation and Clamping

LDS is allocated per work-group or per-wavefront when work-groups are not in use. LDS space is allocated to a
work-group or wavefront in contiguous blocks of 1280 bytes on 1280-byte alignment. LDS allocations do not
wrap around the LDS storage. All accesses to LDS are restricted to the space allocated to that wavefront/work-
group.

Clamping of LDS reads and writes is controlled by two size registers, which contain values for the size of the
LDS space allocated by SPI to this wavefront or work-group, and a possibly smaller value specified in the LDS
instruction (size is held in M0). The LDS operations use the smaller of these two sizes to determine how to
clamp the read/write addresses.

### 3.7. M0 Memory Descriptor

There is one 32-bit M0 register per wavefront, which can be used for:

- Local Data Share (LDS)
  - LDS addressing for Memory/Vfetch → LDS: {14'h0, lds_offset[17:0]} // in bytes
  - { base[5:0], 16'h0}
- Indirect GPR addressing for both vector and scalar instructions. M0 is an unsigned index.

### 3.8. SCC: Scalar Condition code

Most scalar ALU instructions set the Scalar Condition Code (SCC) bit, indicating the result of the operation.

```
  Compare operations: 1 = true
  Arithmetic operations: 1 = carry out
  Bit/logical operations: 1 = result was not zero
  Move: does not alter SCC
```

The SCC can be used as the carry-in for extended-precision integer arithmetic, as well as the selector for
conditional moves and branches.

### 3.9. Vector Compares: VCC and VCCZ

Vector ALU comparisons set the Vector Condition Code (VCC) register (1=pass, 0=fail). Also, vector compares
have the option of setting EXEC to the VCC value.

There is also a VCC summary bit (vccz) that is set to 1 when the VCC result is zero. This is useful for early-exit
branch tests. VCC is also set for selected integer ALU operations (carry-out).

Vector compares have the option of writing the result to VCC (32-bit instruction encoding) or to any SGPR (64-
bit instruction encoding). VCCZ is updated every time VCC is updated: vector compares and scalar writes to
VCC.

The EXEC mask determines which threads execute an instruction. The VCC indicates which executing threads
passed the conditional test, or which threads generated a carry-out from an integer add or subtract.

```
  V_CMP_* ⇒ VCC[n] = EXEC[n] & (test passed for thread[n])
```

VCC is fully written; there are no partial mask updates.

> VCC physically resides in the SGPR register file, so when an instruction sources VCC, that
> counts against the limit on the total number of SGPRs that can be sourced for a given
> instruction. VCC physically resides in the highest two user SGPRs.

Shader Hazard with VCC The user/compiler must prevent a scalar-ALU write to the SGPR holding VCC,
immediately followed by a conditional branch using VCCZ. The hardware cannot detect this, and inserts the
one required wait state (hardware does detect it when the SALU writes to VCC, it only fails to do this when the
SALU instruction references the SGPRs that happen to hold VCC).

### 3.10. Trap and Exception registers

Each type of exception can be enabled or disabled independently by setting, or clearing, bits in the TRAPSTS
register's EXCP_EN field. This section describes the registers which control and report kernel exceptions.

All Trap temporary SGPRs (TTMP*) are privileged for writes - they can be written only when in the trap handler
(status.priv = 1). When not privileged, writes to these are ignored and reads return zero. TMA and TBA are
read-only; they can be accessed through S_GETREG_B32.

When a trap is taken (either user initiated, exception or host initiated), the shader hardware generates an
S_TRAP instruction. This loads trap information into a pair of SGPRS:

```
  {TTMP1, TTMP0} = {3'h0, pc_rewind[3:0], HT[0],trapID[7:0], PC[47:0]}.
```

HT is set to one for host initiated traps, and zero for user traps (s_trap) or exceptions. TRAP_ID is zero for
exceptions, or the user/host trapID for those traps. When the trap handler is entered, the PC of the faulting
instruction is: (PC - PC_rewind*4).

STATUS . TRAP_EN - This bit indicates to the shader whether or not a trap handler is present. When one is not
present, traps are not taken, no matter whether they're floating point, user-, or host-initiated traps. When the
trap handler is present, the wavefront uses an extra 16 SGPRs for trap processing. If trap_en == 0, all traps and
exceptions are ignored, and s_trap is converted by hardware to NOP.

MODE . EXCP_EN[8:0] - Floating point exception enables. Defines which exceptions and events cause a trap.

```
                 Bit                 Exception
                 0                   Invalid
                 1                   Input Denormal
                 2                   Divide by zero
                 3                   Overflow
                 4                   Underflow
                 5                   Inexact
                 6                   Integer divide by zero
                 7                   Address Watch - TC (L1) has witnessed a thread access to an 'address
                                     of interest'
```

#### 3.10.1. Trap Status register

The trap status register records previously seen traps or exceptions. It can be read and written by the kernel.

**Table 5. Exception Field Bits**

```
Field                     Bits       Description
EXCP                      8:0        Status bits of which exceptions have occurred. These bits are sticky and accumulate
                                     results until the shader program clears them. These bits are accumulated regardless
                                     of the setting of EXCP_EN. These can be read or written without shader privilege. Bit
                                     Exception 0 invalid
                                     1 Input Denormal
                                     2 Divide by zero
                                     3 overflow
                                     4 underflow
                                     5 inexact
                                     6 integer divide by zero
                                     7 address watch
                                     8 memory violation
SAVECTX                   10         A bit set by the host command indicating that this wave must jump to its trap handler
                                     and save its context. This bit must be cleared by the trap handler using S_SETREG.
                                     Note - a shader can set this bit to 1 to cause a save-context trap, and due to hardware
                                     latency the shader may execute up to 2 additional instructions before taking the trap.
ILLEGAL_INST              11         An illegal instruction has been detected.
ADDR_WATCH1-3             14:12      Indicates that address watch 1, 2, or 3 has been hit. Bit 12 is address watch 1; bit 13 is
                                     2; bit 14 is 3.
EXCP_CYCLE                21:16      When a float exception occurs, this tells the trap handler on which cycle the exception
                                     occurred on. 0-3 for normal float operations, 0-7 for double float add, and 0-15 for
                                     double float muladd or transcendentals. This register records the cycle number of the
                                     first occurrence of an enabled (unmasked) exception. EXCP_CYCLE[1:0] Phase:
                                     threads 0-15 are in phase 0, 48-63 in phase 3.
                                     EXCP_CYCLE[3:2] Multi-slot pass.
                                     EXCP_CYCLE[5:4] Hybrid pass: used for machines running at lower rates.
DP_RATE                   31:29      Determines how the shader interprets the TRAP_STS.cycle. Different Vector Shader
                                     Processors (VSP) process instructions at different rates.
```

### 3.11. Memory Violations

A Memory Violation is reported from:

- LDS alignment error.
- Memory read/write/atomic alignment error.
- Flat access where the address is invalid (does not fall in any aperture).
- Write to a read-only memory address.

Memory violations are not reported for instruction or scalar-data accesses.

Memory Buffer to LDS does NOT return a memory violation if the LDS address is out of range, but masks off
EXEC bits of threads that would go out of range.

When a memory access is in violation, the appropriate memory (LDS or TC) returns MEM_VIOL to the wave.
This is stored in the wave's TRAPSTS.mem_viol bit. This bit is sticky, so once set to 1, it remains at 1 until the
user clears it.

There is a corresponding exception enable bit (EXCP_EN.mem_viol). If this bit is set when the memory returns
with a violation, the wave jumps to the trap handler.

Memory violations are not precise. The violation is reported when the LDS or TC processes the address; during
this time, the wave may have processed many more instructions. When a mem_viol is reported, the Program
Counter saved is that of the next instruction to execute; it has no relationship the faulting instruction.

### 3.12. Hardware ID Registers

The values below indicate where a wave is currently execution. It is not safe to rely on these values as they may
change over the lifetime of a wave.

**Table 6. Hardware ID (HW_ID)**

```
Field                  Bits          Description
WAVE_ID                3:0           Wave buffer slot number
SIMD_ID                5:4           SIMD which the wave is assigned to within the CU
PIPE_ID                7:6           Pipeline from which the wave was dispatched
CU_ID                  11:8          Compute Unit the wave is assigned to
SH_ID                  12            Shader Array (within an SE) the wave is assigned to. Is set to zero.
SE_ID                  15:13         Shader Engine the wave is assigned to
TG_ID                  19:16         Thread-group ID
VM_ID                  23:20         Virtual Memory ID
QUEUE_ID               26:24         Queue from which this wave was dispatched
STATE_ID               29:27         State ID (UNUSED)
ME_ID                  31:30         Micro-engine ID
```

**Table 7. XCC ID (XCC_ID)**

```
Field                  Bits          Description
XCC_ID                 3:0           ID of this XCC
```

### 3.13. GPR Initialization

When a compute shader wave is launched VGPR0 and a number of SGPRs are initialized.

Compute shaders have VGPR0 initialized with the X, Y and Z index within the workgroup: { 2'b00, Z[9:0], Y[9:0],
X[9:0] }.

**Table 8. CS SGPR Load**

```
SGPR Order        Description                                        Enable
First 0.. 16 of   User data registers                                COMPUTE_PGM_RSRC2.user_sgpr
then              work_group_id0[31:0]                               COMPUTE_PGM_RSRC2.tgid_x_en
then              work_group_id1[31:0]                               COMPUTE_PGM_RSRC2.tgid_y_en
then              work_group_id2[31:0]                               COMPUTE_PGM_RSRC2.tgid_z_en
then              {first_wave, 6'h00, wave_id_in_group[4:0], 2'h0,   COMPUTE_PGM_RSRC2.tg_size_en
                  14'h0, work-group_size_in_waves[5:0]}
TTMP4,5           0
TTMP6             dispatch packet addr lo
TTMP7             dispatch packet addr hi
TTMP8             dispatch grid X[31:0]
TTMP9             dispatch grid Y[31:0]
TTMP10            dispatch grid Z[31:0]
TTMP11            { 26'b0, wave_id_in_workgroup[5:0] }
```

Other TTMPs are not initialized.

## Chapter 4. Program Flow Control

All program flow control is programmed using scalar ALU instructions. This includes loops, branches,
subroutine calls, and traps. The program uses SGPRs to store branch conditions and loop counters. Constants
can be fetched from the scalar constant cache directly into SGPRs.

### 4.1. Program Control

The instructions in the table below control the priority and termination of a shader program, as well as provide
support for trap handlers.

**Table 9. Control Instructions**

```
Instructions       Description
S_ENDPGM           Terminates the wavefront. It can appear anywhere in the kernel and can appear multiple times.
S_ENDPGM_SAVE Terminates the wavefront due to context save. It can appear anywhere in the kernel and can appear
D             multiple times.
S_NOP              Does nothing; it can be repeated in hardware up to 16 times.
S_TRAP             Jumps to the trap handler.
S_RFE              Returns from the trap handler
S_SETPRIO          Modifies the priority of this wavefront: 0=lowest, 3 = highest.
S_SLEEP            Causes the wavefront to sleep for 64 - 8128 clock cycles.
S_SENDMSG          Sends a message (typically an interrupt) to the host CPU.
S_WAKEUP           Causes one wave in a work-group to signal all other waves in the same work-group to wake up from
                   S_SLEEP early. If waves are not sleeping, they are not affected by this instruction.
```

### 4.2. Branching

Branching is done using one of the following scalar ALU instructions.

**Table 10. Branch Instructions**

```
Instructions                          Description
S_BRANCH                              Unconditional branch.
S_CBRANCH_<test>                      Conditional branch. Branch only if <test> is true. Tests are VCCZ, VCCNZ, EXECZ,
                                      EXECNZ, SCCZ, and SCCNZ.
S_CBRANCH_CDBGSYS                     Conditional branch, taken if the COND_DBG_SYS status bit is set.
S_CBRANCH_CDBGUSER                    Conditional branch, taken if the COND_DBG_USER status bit is set.
S_CBRANCH_CDBGSYS_AND_USER            Conditional branch, taken only if both COND_DBG_SYS and COND_DBG_USER are
                                      set.
S_SETPC                               Directly set the PC from an SGPR pair.
S_SWAPPC                              Swap the current PC with an address in an SGPR pair.
S_GETPC                               Retrieve the current PC value (does not cause a branch).
S_CBRANCH_{G,I}_FORK and              Conditional branch for complex branching.
S_CBRANCH_JOIN
S_SETVSKIP                            Set a bit that causes all vector instructions to be ignored. Useful alternative to
                                      branching.
```

```
Instructions                         Description
S_CALL_B64                           Jump to a subroutine, and save return address. SGPR_pair = PC+4; PC =
                                     PC+4+SIMM16*4.
```

For conditional branches, the branch condition can be determined by either scalar or vector operations. A
scalar compare operation sets the Scalar Condition Code (SCC), which then can be used as a conditional branch
condition. Vector compare operations set the VCC mask, and VCCZ or VCCNZ then can be used to determine
branching.

### 4.3. Workgroups

Work-groups are collections of wavefronts running on the same compute unit which can synchronize and
share data. Up to 16 wavefronts (1024 work-items) can be combined into a work-group. When multiple
wavefronts are in a workgroup, the S_BARRIER instruction can be used to force each wavefront to wait until all
other wavefronts reach the same instruction; then, all wavefronts continue. Any wavefront can terminate early
using S_ENDPGM, and the barrier is considered satisfied when the remaining live waves reach their barrier
instruction.

### 4.4. Data Dependency Resolution

Shader hardware resolves most data dependencies, but a few cases must be explicitly handled by the shader
program. In these cases, the program must insert S_WAITCNT instructions to ensure that previous operations
have completed before continuing.

The shader has three counters that track the progress of issued instructions. S_WAITCNT waits for the values
of these counters to be at, or below, specified values before continuing.

These allow the shader writer to schedule long-latency instructions, execute unrelated work, and specify when
results of long-latency operations are needed.

Instructions of a given type return in order, but instructions of different types can complete out-of-order.

- VM_CNT: Vector memory count. Determines when memory reads have returned data to VGPRs, or memory writes have completed.
  - Incremented every time a vector-memory read or write (MUBUF, MTBUF, or FLAT format) instruction is issued.
  - Decremented for reads when the data has been written back to the VGPRs, and for writes when the data has been written to the L2 cache. Ordering: Memory reads and writes return in the order they were issued, including mixing reads and writes.
- LGKM_CNT: (LDS, (K)constant, (M)essage) Determines when one of these low-latency instructions have completed.
  - Incremented by 1 for every LDS instruction issued, as well as by Dword-count for scalar-memory reads (1 for 1-dword loads, 2 for 2-dword or larger loads). S_memtime counts the same as an s_load_dwordx2.
  - Incremented by 1 for every FLAT instruction issued.
  - Decremented by 1 for LDS reads or atomic-with-return when the data has been returned to VGPRs.
  - Incremented by 1 for each S_SENDMSG issued. Decremented by 1 when message is sent out.
  - 

Decremented by 1 for LDS writes when the data has been written to LDS.
  - Decremented by 1 for each Dword returned from the data-cache (SMEM). Ordering:
    - Instructions of different types are returned out-of-order.
    - Instructions of the same type are returned in the order they were issued, except scalar-memory- reads, which can return out-of-order (in which case only S_WAITCNT 0 is the only legitimate value).
- EXP_CNT: VGPR-export count: unused

### 4.5. Manually Inserted Wait States (NOPs)

The hardware does not check for the following dependencies; they must be resolved by inserting NOPs or
independent instructions.

**Table 11. Required Software-inserted Wait States**

```
First Instruction                           Second Instruction                Wait      Notes
S_SETREG <*>                                S_GETREG <same reg>               2
S_SETREG <*>                                S_SETREG <same reg>               2
SET_VSKIP                                   S_GETREG MODE                     2         Reads VSKIP from MODE.
S_SETREG MODE.vskip                         any vector op                     2         Requires two nops or non-vector
                                                                                        instructions.
VALU that sets VCC or EXEC                  VALU that uses EXECZ or VCCZ as 5
                                            a data source
VALU writes SGPR/VCC (readlane, cmp,        V_{READ,WRITE}LANE using that 4
add/sub, div_scale)                         SGPR/VCC as the lane select
VALU writes VCC (including v_div_scale) V_DIV_FMAS                            4
FLAT_STORE_X3                               Write VGPRs holding writedata     1         BUFFER_STORE_* operations that
FLAT_STORE_X4                               from those instructions.                    use an SGPR for "offset" do not
FLAT_ATOMIC_{F}CMPSWAP_X2                                                               require any wait states.
(and global & scratch stores/atomics)
BUFFER_STORE_DWORD_X3
BUFFER_STORE_DWORD_X4
BUFFER_STORE_FORMAT_XYZ
BUFFER_STORE_FORMAT_XYZW
BUFFER_ATOMIC_{F}CMPSWAP_X2
FLAT_STORE_X3                               VALU writes VGPRs holding          2        BUFFER_STORE_* operations that
FLAT_STORE_X4                               writedata from those instructions.          use an SGPR for "offset" do not
(and global & scratch stores/atomics)                                                   require any wait states.
FLAT_ATOMIC_{F}CMPSWAP_X2
BUFFER_STORE_DWORD_X3
BUFFER_STORE_DWORD_X4
BUFFER_STORE_FORMAT_XYZ
BUFFER_STORE_FORMAT_XYZW
BUFFER_ATOMIC_{F}CMPSWAP_X2
VALU writes SGPR                            VMEM reads that SGPR              5         Hardware assumes that there is no
                                                                                        dependency here. If the VALU
                                                                                        writes the SGPR that is used by a
                                                                                        VMEM, the user must add five wait
                                                                                        states.
SALU writes M0                              S_SENDMSG                         1
```

```
First Instruction                            Second Instruction                  Wait   Notes
VALU writes VGPR                             VALU DPP reads that VGPR            2
VALU writes EXEC                             VALU DPP op                         5      ALU does not forward EXEC to
                                                                                        DPP.
Mixed use of VCC: alias vs                   VALU which reads VCC as a            1     VCC can be accessed by name or
SGPR#                                        constant (not as a carry-in which is       by the logical SGPR which holds
v_readlane, v_readfirstlane                  0 wait states).                            VCC. The data dependency check
v_cmp                                                                                   logic does not understand that
v_add*i/u                                                                               these are the same register and do
v_sub*_i/u                                                                              not prevent races.
v_div_scale* (writes vcc)
S_SETREG TRAPSTS                             RFE, RFE_restore                    1
SALU writes M0                               LDS "add-TID" instruction,      1
                                             buffer_store_LDS_dword, scratch
                                             or global with LDS = 1
SALU writes M0                               S_MOVEREL                           1
VALU writes SGPR/VCC:                        VALU reads SGPR as constant         2
v_readlane, v_readfirstlane, v_cmp,          VALU reads SGPR as carry-in         0
v_add*_i/u, v_sub*_i/u, v_div_scale*         v_readlane, v_writelane reads       4
                                             SGPR as lane-select
v_cmpx                                       VALU reads EXEC as constant         2
                                             V_readlane, v_readfirstlane,        4
                                             v_writelane
                                             Other VALU                          0
VALU writes VGPRn                            v_readlane vsrc0 reads VGPRn        1
VALU op which uses OPSEL or SDWA             VALU op consumes result of that     1
with changes the result's bit position       op
VALU Trans op                                Non-trans VALU op consumes          1
                                             result of that op
V_CMPX (writes exec)                         V_PERMLANE*                         4
VALU* writes vdst                            V_PERMLANE* reads vdst              2
```

**Table 12. Trans Ops**

```
V_EXP_F32             V_LOG_F32              V_RCP_F32                    V_RCP_IFLAG_F32
V_RSQ_F32             V_RCP_F64              V_RSQ_F64                    V_SQRT_F32
V_SQRT_F64            V_SIN_F32              V_COS_F32                    V_RCP_F16
V_SQRT_F16            V_RSQ_F16              V_LOG_F16                    V_EXP_F16
V_SIN_F16             V_COS_F16              V_EXP_LEGACY_F32             V_LOG_LEGACY_F32
```

### 4.6. Arbitrary Divergent Control Flow

In the CDNA architecture, conditional branches are handled in one of the following ways.

```
 1. S_CBRANCH This case is used for simple control flow, where the decision to take a branch is based on a
    previous compare operation. This is the most common method for conditional branching.
 2. S_CBRANCH_I/G_FORK and S_CBRANCH_JOIN This method, intended for complex, irreducible control
    flow graphs, is described in the rest of this section. The performance of this method is lower than that for
    S_CBRANCH on simple flow control; use it only when necessary.
```

Conditional Branch (CBR) graphs are grouped into self-contained code blocks, denoted by FORK at the
entrance point, and JOIN and the exit point. The shader compiler must add these instructions into the code.
This method uses a six-deep stack and requires three SGPRs for each fork/join block. Fork/Join blocks can be
hierarchically nested to any depth (subject to SGPR requirements); they also can coexist with other conditional
flow control or computed jumps.

**Figure 3. Example of Complex Control Flow Graph**

The register requirements per wavefront are:

- CSP [2:0] - control stack pointer.
- Six stack entries of 128-bits each, stored in SGPRS: { exec[63:0], PC[47:2] }

This method compares how many of the 64 threads go down the PASS path instead of the FAIL path; then, it
selects the path with the fewer number of threads first. This means at most 50% of the threads are active, and
this limits the necessary stack depth to Log264 = 6.

The following pseudo-code shows the details of CBRANCH Fork and Join operations.

```
  S_CBRANCH_G_FORK arg0, arg1
      // arg1 is an sgpr-pair which holds 64bit (48bit) target address
```

```
  S_CBRANCH_I_FORK arg0, #target_addr_offset[17:2]
      // target_addr_offset: 16b signed immediate offset
```

```
  // PC: in this pseudo-code is pointing to the cbranch_*_fork instruction
  mask_pass = SGPR[arg0] & exec
  mask_fail = ~SGPR[arg0] & exec
```

```
  if (mask_pass == exec)
      I_FORK : PC += 4 + target_addr_offset
      G_FORK: PC = SGPR[arg1]
  else if (mask_fail == exec)
      PC += 4
```

```
  else if (bitcount(mask_fail) < bitcount(mask_pass))
      exec = mask_fail
      I_FORK : SGPR[CSP*4] = { (pc + 4 + target_addr_offset), mask_pass }
      G_FORK: SGPR[CSP*4] = { SGPR[arg1], mask_pass }
      CSP++
      PC += 4
  else
      exec = mask_pass
      SGPR[CSP*4] = { (pc+4), mask_fail }
      CSP++
      I_FORK : PC += 4 + target_addr_offset
      G_FORK: PC = SGPR[arg1]
```

```
  S_CBRANCH_JOIN arg0
  if (CSP == SGPR[arg0]) // SGPR[arg0] holds the CSP value when the FORK started
      PC += 4 // this is the 2nd time to JOIN: continue with pgm
  else
      CSP -- // this is the 1st time to JOIN: jump to other FORK path
      {PC, EXEC} = SGPR[CSP*4] // read 128-bits from 4 consecutive SGPRs
```
