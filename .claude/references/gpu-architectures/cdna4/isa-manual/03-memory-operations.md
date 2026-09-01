# CDNA4 ISA: Memory Operations (Scalar, Vector, Flat, LDS)

> Part of the **CDNA4 Instruction Set Architecture** reference. See the [index](README.md) for other sections.

## Sections in this file

- [Chapter 8. Scalar Memory Operations](#chapter-8-scalar-memory-operations)
  - [8.1. Microcode Encoding](#81-microcode-encoding)
  - [8.2. Operations](#82-operations)
  - [8.3. Dependency Checking](#83-dependency-checking)
  - [8.4. Alignment and Bounds Checking](#84-alignment-and-bounds-checking)
- [Chapter 9. Vector Memory Operations](#chapter-9-vector-memory-operations)
  - [9.1. Vector Memory Buffer Instructions](#91-vector-memory-buffer-instructions)
  - [9.2. Float Memory Atomics](#92-float-memory-atomics)
- [Chapter 10. Flat Memory Instructions](#chapter-10-flat-memory-instructions)
  - [10.1. Flat Memory Instruction](#101-flat-memory-instruction)
  - [10.2. Instructions](#102-instructions)
  - [10.3. Addressing](#103-addressing)
  - [10.4. Global](#104-global)
  - [10.5. Scratch](#105-scratch)
  - [10.6. Data](#106-data)
  - [10.7. Scratch Space (Private)](#107-scratch-space-private)
- [Chapter 11. Data Share Operations](#chapter-11-data-share-operations)
  - [11.1. Overview](#111-overview)
  - [11.2. Dataflow in Memory Hierarchy](#112-dataflow-in-memory-hierarchy)
  - [11.3. LDS Access](#113-lds-access)
  - [11.4. MFMA Transpose Load from LDS](#114-mfma-transpose-load-from-lds)

---

## Chapter 8. Scalar Memory Operations

Scalar Memory Read (SMEM) instructions allow a shader program to load data from memory into SGPRs
through the Scalar Data Cache, or write data from SGPRs to memory through the Scalar Data Cache.
Instructions can read from 1 to 16 Dwords, or write 1 to 4 Dwords at a time. Data is read directly into SGPRs
without any format conversion.

The scalar unit reads and writes consecutive Dwords between memory and the SGPRs. This is intended
primarily for loading ALU constants. No data formatting is supported, nor is byte or short data.

### 8.1. Microcode Encoding

Scalar memory read, write and atomic instructions are encoded using the SMEM microcode format.

The fields are described in the table below:

**Table 39. SMEM Encoding Field Descriptions**

```
Field   Size Description
OP      8    Opcode.
IMM     1    Determines how the OFFSET field is interpreted.
             IMM=1 : Offset is a 21-bit unsigned byte offset to the address.
             IMM=0 : Offset[6:0] specifies an SGPR or M0 which provides an unsigned byte offset (for stores, must be
             M0). STORE and ATOMIC instructions cannot use an SGPR: only imm or M0.
GLC     1    Globally Coherent.
             For loads, controls L1 cache policy: 0=hit_lru, 1=miss_evict.
             For stores, controls L1 cache bypass: 0=write-combine, 1=write-thru.
             For atomics, "1" indicates that the atomic returns the pre-op value.
SDATA 7      SGPRs to return read data to, or to source write-data from.
             Reads of two Dwords must have an even SDST-sgpr.
             Reads of four or more Dwords must have their DST-gpr aligned to a multiple of 4.
             SDATA must be: SGPR or VCC. Not: exec or m0.
SBASE   6    SGPR-pair (SBASE has an implied LSB of zero) which provides a base address, or for BUFFER instructions, a
             set of 4 SGPRs (4-sgpr aligned) which hold the resource constant. For BUFFER instructions, the only
             resource fields used are: base, stride, num_records.
OFFSET 21    An unsigned byte offset, or the address of an SGPR holding the offset. Writes and atomics: M0 or immediate
             only, not SGPR.
NV      1    Non-volatile.
SOE     1    Scalar Offset Enable.
```

See Memory Scope and Temporal Control for more information on the GLC bit.

### 8.2. Operations

#### 8.2.1. S_LOAD_DWORD, S_STORE_DWORD

These instructions load 1-16 Dwords or store 1-4 Dwords between SGPRs and memory. The data in SGPRs is
specified in SDATA, and the address is composed of the SBASE, OFFSET, and SOFFSET fields.

8.2.1.1. Scalar Memory Addressing
S_LOAD / S_STORE / S_DCACHE_DISCARD:

```
    ADDR = SGPR[base] + inst_offset + { M0 or SGPR[offset] or zero }
```

S_SCRATCH_LOAD / S_SCRATCH_STORE:

```
    ADDR = SGPR[base] + inst_offset + { M0 or SGPR[offset] or zero } * 64
```

Use of offset fields:

```
IMM SOFFSET_EN (SOE)        Address
0      0                    SGPR[base] + (SGPR[offset] or M0)
0      1                    SGPR[base] + (SGPR[soffset] or M0)
1      0                    SGPR[base] + inst_offset
1      1                    SGPR[base] + inst_offset + (SGPR[soffset] or M0)
```

All components of the address (base, offset, inst_offset, M0) are in bytes, but the two LSBs are ignored and
treated as if they were zero. S_DCACHE_DISCARD ignores the six LSBs to make the address 64-byte-aligned.

It is illegal and undefined if the inst_offset is negative and the resulting
(inst_offset + (M0 or SGPR[offset])) is negative.

Scalar access to private space must either use a buffer constant or manually convert the address:

```
    Addr = Addr - private_base + private_base_addr + scratch_baseOffset_for_this_wave
```

"Hidden private base" is not available to the shader through hardware: It must be preloaded into an SGPR or
made available through a constant buffer. This is equivalent to what the driver must do to calculate the base
address from scratch for buffer constants.

A scalar instruction must not overwrite its own source registers because the possibility of the instruction being
replayed due to an ATC XNACK. Similarly, instructions in scalar memory clauses must not overwrite the
sources of any of the instructions in the clause. A clause is defined as a string of memory instructions of the
same type. A clause is broken by any non-memory instruction. One exception to this rule is a single SMEM
instruction in a clause by itself which loads a single DWORD may legally overwrite its own source SGPRs. (This

instruction either completely succeeds to execute and continue, or completely fail; it does not overwrite just part of one
DWORD).

Atomics are unusual because they are naturally aligned and they must be in a single-instruction clause. By
definition, an atomic that returns the pre-op value overwrites its data source, which is acceptable.

Reads/Writes/Atomics using Buffer Constant

Buffer constant fields used: base_address, stride, num_records, NV. Other fields are ignored.

Scalar memory read/write does not support "swizzled" buffers. Stride is used only for memory address bounds
checking, not for computing the address to access.

The SMEM supplies only a SBASE address (byte) and an offset (byte or Dword). Any "index * stride" must be
calculated manually in shader code and added to the offset prior to the SMEM.

The two LSBs of V#.base and of the final address are ignored to force Dword alignment.

```
  "m_*" components come from the buffer constant (V#):
     offset       = IMM ? OFFSET : SGPR[OFFSET]
     m_base       = { SGPR[SBASE * 2 +1][15:0], SGPR[SBASE] }
     m_stride     = SGPR[SBASE * 2 +1][31:16]
     m_num_records = SGPR[SBASE * 2 + 2]
     m_size       = (m_stride == 0) ? 1 : m_num_records
     m_addr       = (SGPR[SBASE * 2] + offset) & ~0x3
     SGPR[SDST] = read_Dword_from_dcache(m_base, offset, m_size)
```

```
     If more than 1 dword is being read, it is returned to SDST+1, SDST+2, etc,
     and the offset is incremented by 4 bytes per DWORD.
```

#### 8.2.2. Scalar Atomic Operations

The scalar memory unit supports the same set of memory atomics as the vector memory unit. Addressing is the
same as for scalar memory loads and stores. Like the vector memory atomics, scalar atomic operations can
return the "pre-operation value" to the SDATA SGPRs. This is enabled by setting the microcode GLC bit to 1.

#### 8.2.3. S_DCACHE_INV, S_DCACHE_WB

This instruction invalidates, or does a "write back" of dirty data, for the entire scalar data cache. It does not
return anything to SDST.

#### 8.2.4. S_MEMTIME

This instruction reads a 64-bit clock counter into a pair of SGPRs: SDST and SDST+1.

#### 8.2.5. S_MEMREALTIME

This instruction reads a 64-bit "real time-counter" and returns the value into a pair of SGPRS: SDST and SDST+1.
The time value is from a constant 100MHz clock (not affected by power modes or core clock frequency
changes).

### 8.3. Dependency Checking

Scalar memory reads and writes can return data out-of-order from how they were issued; they can return
partial results at different times when the read crosses two cache lines. The shader program uses the
LGKM_CNT counter to determine when the data has been returned to the SDST SGPRs. This is done as follows.

- LGKM_CNT is incremented by 1 for every fetch of a single Dword.
- LGKM_CNT is incremented by 2 for every fetch of two or more Dwords.
- LGKM_CNT is decremented by an equal amount when each instruction completes.

Because the instructions can return out-of-order, the only sensible way to use this counter is to implement
S_WAITCNT 0; this imposes a wait for all data to return from previous SMEMs before continuing.

### 8.4. Alignment and Bounds Checking

```
SDST
   The value of SDST must be even for fetches of two Dwords (including S_MEMTIME), or a multiple of four
   for larger fetches. If this rule is not followed, invalid data can result. If SDST is out-of-range, the instruction
   is not executed.
```

```
SBASE
   The value of SBASE must be even for S_BUFFER_LOAD (specifying the address of an SGPR which is a
   multiple of four). If SBASE is out-of-range, the value from SGPR0 is used.
```

```
OFFSET
   The value of OFFSET has no alignment restrictions.
```

Memory Address : If the memory address is out-of-range (clamped), the operation is not performed for any
Dwords that are out-of-range.

## Chapter 9. Vector Memory Operations

Vector Memory (VMEM) instructions read or write one piece of data separately for each work-item in a
wavefront into, or out of, VGPRs. This is in contrast to Scalar Memory instructions, which move a single piece
of data that is shared by all threads in the wavefront. All Vector Memory (VM) operations are processed by the
texture cache system (level 1 and level 2 caches).

Software initiates a load, store or atomic operation through the texture cache through one of these types of
VMEM instructions:

- MTBUF: Memory typed-buffer operations.
- MUBUF: Memory untyped-buffer operations.

The instruction defines which VGPR(s) supply the addresses for the operation, which VGPRs supply or receive
data from the operation, and a series of SGPRs that contain the memory buffer descriptor (V#).

### 9.1. Vector Memory Buffer Instructions

Vector-memory (VM) operations transfer data between the VGPRs and buffer objects in memory through the
texture cache (TC). Vector means that one or more piece of data is transferred uniquely for every thread in the
wavefront, in contrast to scalar memory reads, which transfer only one value that is shared by all threads in
the wavefront.

Buffer reads have the option of returning data to VGPRs or directly into LDS.

Examples of buffer objects are vertex buffers, raw buffers, stream-out buffers, and structured buffers.

Buffer objects support both homogeneous and heterogeneous data, but no filtering of read-data. Buffer
instructions are divided into two groups:

- MUBUF: Untyped buffer objects.
  - Data format is specified in the resource constant.
  - Load, store, atomic operations, with or without data format conversion.
- MTBUF: Typed buffer objects.
  - Data format is specified in the instruction.
  - The only operations are Load and Store, both with data format conversion.

Atomic operations take data from VGPRs and combine them arithmetically with data already in memory.
Optionally, the value that was in memory before the operation took place can be returned to the shader.

All VM operations use a buffer resource constant (V#) which is a 128-bit value in SGPRs. This constant is sent to
the texture cache when the instruction is executed. This constant defines the address and characteristics of the
buffer in memory. Typically, these constants are fetched from memory using scalar memory reads prior to
executing VM instructions, but these constants also can be generated within the shader.

#### 9.1.1. Simplified Buffer Addressing

The equation below shows how the hardware calculates the memory address for a buffer access.

#### 9.1.2. Buffer Instructions

Buffer instructions (MTBUF and MUBUF) allow the shader program to read from, and write to, linear buffers
in memory. These operations can operate on data as small as one byte, and up to four Dwords per work-item.
Atomic arithmetic operations are provided that can operate on the data values in memory and, optionally,
return the value that was in memory before the arithmetic operation was performed.

The D16 instruction variants convert the results to packed 16-bit values. For example,
BUFFER_LOAD_FORMAT_D16_XYZW writes two VGPRs.

**Table 40. Buffer Instructions**

```
Instruction                                      Description
MTBUF Instructions
TBUFFER_LOAD_FORMAT_{x,xy,xyz,xyzw}              Read from, or write to, a typed buffer object. Also used for a vertex
TBUFFER_STORE_FORMAT_{x,xy,xyz,xyzw}             fetch.
MUBUF Instructions
BUFFER_LOAD_FORMAT_{x,xy,xyz,xyzw}               Read to, or write from, an untyped buffer object.
BUFFER_STORE_FORMAT_{x,xy,xyz,xyzw}              <size> = byte, ubyte, short, ushort, Dword, Dwordx2, Dwordx3,
BUFFER_LOAD_<size>                               Dwordx4
BUFFER_STORE_<size>
BUFFER_ATOMIC_<op>                               Buffer object atomic operation. Globally coherent. Operates on 32-bit or
BUFFER_ATOMIC_<op>_ x2                           64-bit values (x2 = 64 bits).
```

**Table 41. Microcode Formats**

```
Field    Bit Size Description
OP       4       MTBUF: Opcode for Typed buffer instructions.
         7       MUBUF: Opcode for Untyped buffer instructions.
VADDR    8       Address of VGPR to supply first component of address (offset or index). When both index and offset are
                 used, index is in the first VGPR, offset in the second.
VDATA    8       Address of VGPR to supply first component of write data or receive first component of read-data.
SOFFSET 8        SGPR to supply unsigned byte offset. Must be an SGPR, M0, or inline constant.
SRSRC    5       Specifies which SGPR supplies T# (resource constant) in four or eight consecutive SGPRs. This field is
                 missing the two LSBs of the SGPR address, since this address must be aligned to a multiple of four
                 SGPRs.
```

```
Field     Bit Size Description
DFMT      4       Data Format of data in memory buffer:
                  0 invalid
                  18
                  2 16
                  3 8_8
                  4 32
                  5 16_16
                  6 10_11_11
                  7 11_11_10
                  8 10_10_10_2
                  9 2_10_10_10
                  10 8_8_8_8
                  11 32_32
                  12 16_16_16_16
                  13 32_32_32
                  14 32_32_32_32
                  15 reserved
NFMT      3       Numeric format of data in memory:
                  0 unorm
                  1 snorm
                  2 uscaled
                  3 sscaled
                  4 uint
                  5 sint
                  6 reserved
                  7 float
OFFSET 12         Unsigned byte offset.
OFFEN     1       1 = Supply an offset from VGPR (VADDR). 0 = Do not (offset = 0).
IDXEN     1       1 = Supply an index from VGPR (VADDR). 0 = Do not (index = 0).
SC0       1       Scope bit 0
NT        1       Non-Temporal
ACC       1       VDATA is Accumulation VGPR
SC1       1       Scope bit 1
LDS       1       MUBUF-ONLY: 0 = Return read-data to VGPRs. 1 = Return read-data to LDS instead of VGPRs.
```

#### 9.1.3. VGPR Usage

VGPRs supply address and write-data; also, they can be the destination for return data (the other option is
LDS).

```
Address
     Zero, one or two VGPRs are used, depending of the offset-enable (OFFEN) and index-enable (IDXEN) in the
     instruction word, as shown in the table below:
```

**Table 42. Address VGPRs**

```
                                          IDXEN OFFEN VGPRn             VGPRn+1
                                          0      0        nothing
                                          0      1        uint offset
                                          1      0        uint index
                                          1      1        uint index uint offset
```

Write Data : N consecutive VGPRs, starting at VDATA. The data format specified in the instruction word
(NFMT, DFMT for MTBUF, or encoded in the opcode field for MUBUF) determines how many Dwords to write.

Read Data : Same as writes. Data is returned to consecutive GPRs.

Read Data Format : Read data is 32 bits, based on the data format in the instruction or resource. Float or
normalized data is returned as floats; integer formats are returned as integers (signed or unsigned, same type
as the memory storage format). Memory reads of data in memory that is 32 or 64 bits do not undergo any
format conversion.

Atomics with Return : Data is read out of the VGPR(s) starting at VDATA to supply to the atomic operation. If
the atomic returns a value to VGPRs, that data is returned to those same VGPRs starting at VDATA.

#### 9.1.4. Buffer Data

The amount and type of data that is read or written is controlled by the following: data-format (dfmt), numeric-
format (nfmt), destination-component-selects (dst_sel), and the opcode. Dfmt and nfmt can come from the
resource, instruction fields, or the opcode itself. Dst_sel comes from the resource, but is ignored for many
operations.

**Table 43. Buffer Instructions**

```
                       Instruction                    Data Format    Num Format    DST SEL
                       TBUFFER_LOAD_FORMAT_*          instruction    instruction   identity
                       TBUFFER_STORE_FORMAT_*         instruction    instruction   identity
                       BUFFER_LOAD_<type>             derived        derived       identity
                       BUFFER_STORE_<type>            derived        derived       identity
                       BUFFER_LOAD_FORMAT_*           resource       resource      resource
                       BUFFER_STORE_FORMAT_*          resource       resource      resource
                       BUFFER_ATOMIC_*                derived        derived       identity
```

Instruction : The instruction's dfmt and nfmt fields are used instead of the resource's fields.

Data format derived : The data format is derived from the opcode and ignores the resource definition. For
example, buffer_load_ubyte sets the data-format to 8 and number-format to uint.

> The resource's data format must not be INVALID; that format has specific meaning
> (unbound resource), and for that case the data format is not replaced by the instruction's
> implied data format.

DST_SEL identity : Depending on the number of components in the data-format, this is: X000, XY00, XYZ0, or
XYZW.

The MTBUF derives the data format from the instruction. The MUBUF BUFFER_LOAD_FORMAT and
BUFFER_STORE_FORMAT instructions use dst_sel from the resource; other MUBUF instructions derive data-
format from the instruction itself.

D16 Instructions : Load-format and store-format instructions also come in a "d16" variant. For stores, each 32-
bit VGPR holds two 16-bit data elements that are passed to the texture unit. This texture unit converts them to
the texture format before writing to memory. For loads, data returned from the texture unit is converted to 16

bits, and a pair of data are stored in each 32-bit VGPR (LSBs first, then MSBs). Control over int vs. float is
controlled by NFMT.

#### 9.1.5. Buffer Addressing

A buffer is a data structure in memory that is addressed with an index and an offset. The index points to a
particular record of size stride bytes, and the offset is the byte-offset within the record. The stride comes from
the resource, the index from a VGPR (or zero), and the offset from an SGPR or VGPR and also from the
instruction itself.

**Table 44. BUFFER Instruction Fields for Addressing**

```
Field        Size Description
inst_offset 12   Literal byte offset from the instruction.
inst_idxen 1     Boolean: get index from VGPR when true, or no index when false.
inst_offen 1     Boolean: get offset from VGPR when true, or no offset when false. Note that inst_offset is present,
                 regardless of this bit.
```

The "element size" for a buffer instruction is the amount of data the instruction transfers. It is determined by
the DFMT field for MTBUF instructions, or from the opcode for MUBUF instructions. It can be 1, 2, 4, 8, or 16
bytes.

**Table 45. V# Buffer Resource Constant Fields for Addressing**

```
Field                Size Description
const_base           48   Base address, in bytes, of the buffer resource.
const_stride         14   Stride of the record in bytes (0 to 16,383 bytes, or 0 to 262,143 bytes). Normally 14 bits, but is
                     or   extended to 18-bits when:
                     18   const_add_tid_enable = true used with MUBUF instructions which are not format types (or
                          cache invalidate/WB).
                          This is extension intended for use with scratch (private) buffers.
```

```
                             If (const_add_tid_enable && MUBUF-non-format instr.)
                                const_stride [17:0] = { V#.DFMT[3:0],
                                                         V#.const_stride[13:0] }
                             else
                                const_stride is 14 bits: {4'b0, V#.const_stride[13:0]}
```

```
const_num_record 32       Number of records in the buffer.
s                         In units of Bytes for raw buffers, units of Stride for structured buffers, and ignored for private
                          (scratch) buffers.
                          In units of: (inst_idxen == 1) ? Bytes : Stride
const_add_tid_ena 1       Boolean. Add thread_ID within the wavefront to the index when true.
ble
const_swizzle_enab 1      Boolean. Indicates that the surface is swizzled when true.
le
const_element_size 2      Used only when const_swizzle_en = true. Number of contiguous bytes of a record for a given
                          index (2, 4, 8, or 16 bytes).
                          Must be >= the maximum element size in the structure. const_stride must be an integer multiple
                          of const_element_size.
const_index_stride 2      Used only when const_swizzle_en = true. Number of contiguous indices for a single element (of
                          const_element_size) before switching to the next element. There are 8, 16, 32, or 64 indices.
```

**Table 46. Address Components from GPRs**

```
Field         Size Description
SGPR_offset   32    An unsigned byte-offset to the address. Comes from an SGPR or M0.
VGPR_offset   32    An optional unsigned byte-offset. It is per-thread, and comes from a VGPR.
VGPR_index    32    An optional index value. It is per-thread and comes from a VGPR.
```

The final buffer memory address is composed of three parts:

- the base address from the buffer resource (V#),
- the offset from the SGPR, and
- a buffer-offset that is calculated differently, depending on whether the buffer is linearly addressed (a simple Array-of-Structures calculation) or is swizzled.

**Figure 4. Address Calculation for a Linear Buffer**

9.1.5.1. Range Checking
Addresses can be checked to see if they are in or out of range. When an address is out of range, reads return
zero, and writes and atomics are dropped. The address range check method depends on the buffer type.

```
Private (Scratch) Buffer
   Used when: AddTID==1 && IdxEn==0
   For this buffer, there is no range checking.
```

```
Raw Buffer
   Used when: AddTID==0 && SWizzleEn==0 && IdxEn==0
   Out of Range if: (InstOffset + (OffEN ? vgpr_offset : 0)) >= NumRecords
```

```
Structured Buffer
   Used when: AddTID==0 && Stride!=0 && IdxEn==1
   Out of Range if: Index(vgpr) >= NumRecords
```

Notes:

1. Reads that go out-of-range return zero (except for components with V#.dst_sel = SEL_1 that return 1).

2. Writes that are out-of-range do not write anything.
3. Load/store-format-* instruction and atomics are range-checked "all or nothing" - either entirely in or out.
4. Load/store-Dword-x{2,3,4} and range-check per component.

9.1.5.2. Swizzled Buffer Addressing
Swizzled addressing rearranges the data in the buffer to help provide improved cache locality for arrays of
structures. Swizzled addressing also requires Dword-aligned accesses. A single fetch instruction cannot
attempt to fetch a unit larger than const-element-size. The buffer's STRIDE must be a multiple of element_size.

```
  Index = (inst_idxen ? vgpr_index : 0) +
           (const_add_tid_enable ? thread_id[5:0] : 0)
```

```
  Offset = (inst_offen ? vgpr_offset : 0) + inst_offset
```

```
  index_msb = index / const_index_stride
  index_lsb = index % const_index_stride
  offset_msb = offset / const_element_size
  offset_lsb = offset % const_element_size
```

```
  buffer_offset = (index_msb * const_stride + offset_msb *
                     const_element_size) * const_index_stride + index_lsb *
                     const_element_size + offset_lsb
```

```
  Final Address = const_base + sgpr_offset + buffer_offset
```

Remember that the "sgpr_offset" is not a part of the "offset" term in the above equations.

**Figure 5. Example of Buffer Swizzling**

9.1.5.3. Proposed Use Cases for Swizzled Addressing
Here are few proposed uses of swizzled addressing in common graphics buffers.

**Table 47. Swizzled Buffer Use Cases**

```
                    DX11 Raw             Dx11 Structured       Dx11 Structured      Scratch       Ring /       Const Buffer
                    Uav OpenCL           (literal offset)      (gpr offset)                       stream-out
                    Buffer Object
inst_vgpr_offset_e T                     F                     T                    T             T            T
n
inst_vgpr_index_en F                     T                     T                    F             F            F
const_stride        na                   <api>                 <api>                scratchSize   na           na
const_add_tid_ena F                      F                     F                    T             T            F
ble
const_buffer_swizz F                     T                     T                    T             F            F
le
const_elem_size     na                   4                     4                    4 or 16       na           4
const_index_stride na                    16                    16                   64
```

#### 9.1.6. 16-bit Memory Operations

The D16 buffer instructions allow a kernel to load or store just 16 bits per work item between VGPRs and
memory. There are two variants of these instructions:

- D16 loads data into or stores data from the lower 16 bits of a VGPR.
- D16_HI loads data into or stores data from the upper 16 bits of a VGPR.

For example, BUFFER_LOAD_UBYTE_D16 reads a byte per work-item from memory, converts it to a 16-bit
integer, then loads it into the lower 16 bits of the data VGPR.

When ECC is enabled 16-bit memory loads write the full 32-bit VGPR. Unused bits are set to zero.

#### 9.1.7. Alignment

For Dword or larger reads or writes, the two LSBs of the byte-address are ignored, thus forcing Dword
alignment.

#### 9.1.8. Buffer Resource

The buffer resource describes the location of a buffer in memory and the format of the data in the buffer. It is
specified in four consecutive SGPRs (four aligned SGPRs) and sent to the texture cache with each buffer
instruction.

The table below details the fields that make up the buffer resource descriptor.

**Table 48. Buffer Resource Descriptor**

```
Bits           Size       Name                Description
47:0           48         Base address        Byte address.
61:48          14         Stride              Bytes 0 to 16383
62             1          Cache swizzle       Buffer access. Optionally, swizzle texture cache TC L1 cache banks.
63             1          Swizzle enable      Swizzle AOS according to stride, index_stride, and element_size, else
                                              linear (stride * index + offset).
95:64          32         Num_records         In units of stride or bytes.
98:96          3          Dst_sel_x           Destination channel select:
101:99         3          Dst_sel_y           0=0, 1=1, 4=R, 5=G, 6=B, 7=A
104:102        3          Dst_sel_z
107:105        3          Dst_sel_w
110:108        3          Num format          Numeric data type (float, int, …). See instruction encoding for values.
114:111        4          Data format         Number of fields and size of each field. See instruction encoding for
                                              values. For MUBUF instructions with ADD_TID_EN = 1. This field holds
                                              Stride [17:14].
115            1          User VM Enable      Resource is mapped via tiled pool / heap.
116            1          User VM mode        Unmapped behavior: 0: null (return 0 / drop write); 1:invalid (results in
                                              error)
118:117        2          Index stride        8, 16, 32, or 64. Used for swizzled buffer addressing.
119            1          Add tid enable      Add thread ID to the index for to calculate the address.
122:120        3          RSVD                Reserved. Must be set to zero.
123            1          NV                  Non-volatile (0=volatile)
125:124        2          RSVD                Reserved. Must be set to zero.
127:126        2          Type                Value == 0 for buffer. Overlaps upper two bits of four-bit TYPE field in
                                              128-bit T# resource.
```

A resource set to all zeros acts as an unbound texture or buffer (return 0,0,0,0).

#### 9.1.9. Memory Buffer Load to LDS

The MUBUF instruction format allows reading data from a memory buffer directly into LDS without passing
through VGPRs. This is supported for the following subset of MUBUF instructions.

- BUFFER_LOAD_{ubyte, sbyte, ushort, sshort, dword, dwordX3, dwordX4, format_x}.

```
     LDS_offset = 18-bit unsigned byte offset from M0[17:0].
     Mem_offset = 32-bit unsigned byte offset from an SGPR (the SOFFSET SGPR).
     idx_vgpr = index value from a VGPR (located at VADDR). (Zero if idxen=0.)
     off_vgpr = offset value from a VGPR (located at VADDR or VADDR+1). (Zero if offen=0.)
```

The figure below shows the components of the LDS and memory address calculation:

TIDinWave is only added if the resource (T#) has the ADD_TID_ENABLE field set to 1, whereas LDS adds it. The
MEM_ADDR M0 is in the VDATA field; it specifies M0.

For loads to LDS with data-size of 3 or 4 dwords, the equation is modified to be: (TIDinWAve * 16).
Note that LOAD_DWORDX3 writes 3 dwords and skips the 4th.

9.1.9.1. Clamping Rules
Memory address clamping follows the same rules as any other buffer fetch. LDS address clamping: the return
data must not be written outside the LDS space allocated to this wave.

- Set the active-mask to limit buffer reads to those threads that return data to a legal LDS location.
- The LDSbase (alloc) is in units of 32 Dwords, as is LDSsize.

#### 9.1.10. Memory Scope and Temporal Control

9.1.10.1. Scalar Memory
Scalar Memory instructions have a single memory control bit: GLC (Globally Coherent). The GLC bit means
different things for loads, stores, and atomic ops.

READ
GLC = 0 Reads can hit on the L1 and persist across wavefronts
GLC = 1 Reads miss the L1 and L2 and force fetch to the data fabric. No L1 persistence across waves.
WRITE
GLC = 0 Writes miss the L1, write through to L2, and persist in L1 across wavefronts.
GLC = 1 Writes miss the L1, write through to L2. No persistence across wavefronts.
ATOMIC
GLC = 0 Previous data value is not returned. No L1 persistence across wavefronts.
GLC = 1 Previous data value is returned. No L1 persistence across wavefronts.
Note: GLC means "return pre-op value" for atomics.

9.1.10.2. Vector Memory
Vector Memory instructions (Flat, Global, Scratch, and Buffer) have 3 bits to control scope and cacheability:

- SC[1:0] System Cache level: 0=wave, 1=group, 2=device, 3=system

- NT Non-Temporal: 0=expect temporal reuse; 1=do not expect temporal reuse

Loads

**Table 49. Load Controls**

```
Scope SC1 SC0 NT CU Cache                L2 Cache Behavior                                       Last-level Cache
                 Behavior                                                                        Behavior
Wave    0   0    0   Hit LRU             Hit LRU                                                 Hit LRU
                 1   Miss Evict          Hit Stream                                              Hit Evict
Group 0     1    0   Hit LRU             Hit LRU                                                 Hit Evict
                 1   Miss Evict          Hit Stream                                              Hit Evict
Device 1    0    0   Miss Evict          (1 L2 cache): Hit LRU; (>1 L2 cache): Coherent Cache    Hit LRU
                                         Bypass
                 1   Miss Evict          (1 L2 cache): Hit Stream; (>1 L2 cache): Coherent Cache Hit Evict
                                         Bypass
Syste   1   1    0   Miss Evict          Coherent Cache Bypass                                   Hit LRU
m                1   Miss Evict          Coherent Cache Bypass                                   Hit Evict
```

Note that if TG_SPLIT mode is active, SC1==0, SC0==1, NT==0 case becomes: Miss LRU, Hit LRU, Hit LRU.

Stores

**Table 50. Store Controls**

```
Scope SC1 SC0 NT CU Cache                L2 Cache Behavior                                       Last-level Cache
                 Behavior                                                                        Behavior
Wave    0   0    0   Miss LRU            Hit LRU                                                 Hit LRU
                 1   Miss Evict          Hit Stream                                              Hit Evict
Group 0     1    0   Miss LRU            Hit LRU                                                 Hit LRU
                 1   Miss Evict          Hit Stream                                              Hit Evict
Device 1    0    0   Miss Evict          (1 L2 cache): Hit LRU; (>1 L2 cache): Coherent Cache    Hit LRU
                                         Bypass
                 1   Miss Evict          (1 L2 cache): Hit Stream; (>1 L2 cache): Coherent Cache Hit Evict
                                         Bypass
Syste   1   1    0   Miss Evict          Coherent Cache Bypass                                   Hit LRU
m                1   Miss Evict          Coherent Cache Bypass                                   Hit Evict
```

Atomics

For atomics, SC0 indicates whether or not to return the "pre-op" memory value (the value in memory before
the atomic operation was performed). 0 = no return, 1 = return pre-op value.

SC1 : 0 = device scope atomic, 1 = system scope atomic

NT : 0 = last level cache "allocate" policy; 1 = "no Allocate" policy

Invalidate and Writeback

**Table 51. BUFFER_WBL2**

```
SC1 SC0 L2 Cache Behavior
0    any NOP
1    0    (1 L2 cache): NOP,
          (>1 L2 cache) Write-back dirty data
1    1    Write back dirty data
```

**Table 52. BUFFER_INV**

```
SC1 SC0 CU Cache Behavior                    L2 Cache Behavior
0    0    NOP                                NOP
0    1    (if TG Split): Invalidate cache,   NOP
          (If not TG Split): NOP
1    0    Invalidate cache                   (1 L2 cache): NOP,
                                             (>1 L2 cache) Invalidate non-coherently cached lines
1    1    Invalidate cache                   Invalidate non-coherently cached lines
```

#### 9.1.11. Data Formats

The table below shows the buffer data formats:

### 9.2. Float Memory Atomics

Floating point memory atomics are executed in LDS and in the L2 cache. They can be issued as LDS, Buffer,
Flat, Global, and Scratch instructions.

This chapter explains the rules for rounding, denormals and NaN for floating point atomics.

#### 9.2.1. Rounding of Float Atomics

All float atomic ADD opcodes use "Round to Nearest-Even" rounding.

#### 9.2.2. Denormal (Subnormal) Handling

When atomics operate on floating point data, there is the possibility of the data containing denormal numbers,
or the operation producing a denormal.

Denormals: The floating point atomic instructions have the option of passing denormal values through, or
flushing them to zero. This is controlled with the MODE.denorm bits which also control VALU denormal
behavior. As with VALU ops, "denorm_single" affects F32 ops and "denorm_double" affects F64 and F16. Some
atomics have fixed denormal handling behavior.

LDS instructions allows denormals to be passed through or flushed to zero based on the MODE.denormal wave-
state register.
- Float 16 and 32 bit Adder uses both input and output denorm flush controls from MODE
- Float 64 bit adder does not flush denorms
- Float CMP, MIN and MAX use only the "input denormal" flushing control
  - Each input to the comparisons flushes the mantissa of both operands to zero before the compare if the exponent is zero and the flush denorm control is active. For Min and Max the actual result returned is the selected non-flushed input.
  - CompareStore ("compare swap") flushes the result when input denormal flushing occurs.

**Table 53. Denorm Handling Rules for Memory Ops**

```
Atomic type                 LDS Handling           L2 Cache Handling
PK_ADD_F16 / BF16           Mode                   Do not Flush Denorms
ADD_F32                     Mode                   Flush Denorms
Min/MAX_F32                 Mode                   N/A
CMPST_F32                   Mode                   N/A
MIN/MAX_F64                 Mode                   Do not Flush Denorms
CMPST_F64                   Mode                   N/A
ADD_F64                     Do not Flush Denorms   Do not Flush Denorms
```

- "Flush Denorms" = flush all input denorm
- "Do not Flush" = don't flush input denorm
- "Mode" = denormal flush controlled by bit from shader's "MODE . fp_denorm" register
- "Mode + reg" = "Mode" from above, but there exists an override register to flush output or not.

Note that MIN and MAX when flushing denormals only do it for the comparison, but the result is an
unmodified copy of one of the sources. CompareStore ("compare swap") flushes the result when input
denormal flushing occurs.

#### 9.2.3. NaN Handling

Not A Number ("NaN") is a IEEE-754 value representing a result which cannot be computed.

There two types of NaN: quiet and signaling
- Quiet NaN Exponent=0xFF, Mantissa MSB=1
- Signaling NaN Exponent=0xFF, Mantissa MSB=0 and at least one other mantissa bit ==1

The LDS does not produce any exception or "signal" due to a signaling NaN.

DS_ADD_F32 can create a quiet NaN, or propagate NaN from its inputs: if either input is a NaN, the output is
that same NaN, and if both inputs are NaN, the NaN from the first input is selected as the output. Signaling NaN
is converted to Quiet NaN.

Floating point atomics (CMPSWAP, MIN, MAX) flush input denormals only when
MODE (allow_input_denorm)=0, otherwise values are passed through without modification. When flushing,
denorms are flushed before the operation (i.e. before the comparison).

FP Max Selection Rules:

```
   if (src0 == SNaN) result = QNaN (src0)
   else if (src1 == SNaN) result = QNaN (src1)
   else result = larger of (src0, src1)
   "Larger" order from smallest to largest: QNaN, -inf, -float, -denorm, -0, +0, +denorm, +float, +inf
```

FP Min Selection Rules:

```
   if (src0 == SNaN) result = QNaN (src0)
   else if (src1 == SNaN) result = QNaN (src1)
   else result = smaller of (src0, src1)
   "Smaller" order from smallest to largest: -inf, -float, -denorm, -0, +0, +denorm, +float, +inf, QNaN
```

FP Compare Swap: only swap if the compare condition (==) is true, treating +0 and -0 as equal

```
   doSwap = (src0 != NaN) && (src1 != NaN) && (src0 == src1) // allow +0 == -0
```

Float Add rules:
1. -INF + INF = QNAN (mantissa is all zeros except MSB)
2. +/-INF + NAN = QNAN (NAN input is copied to output but made quiet NAN)
3. -0 + 0 = +0
4. INF + (float, +0, -0) = INF, with infinity sign preserved
5. NaN + NaN = SRC0's NaN, converted to QNaN

## Chapter 10. Flat Memory Instructions

Flat Memory instructions read, or write, one piece of data into, or out of, VGPRs; they do this separately for
each work-item in a wavefront. Unlike buffer or image instructions, Flat instructions do not use a resource
constant to define the base address of a surface. Instead, Flat instructions use a single flat address from the
VGPR; this addresses memory as a single flat memory space. This memory space includes video memory,
system memory, LDS memory, and scratch (private) memory. Parts of the flat memory space may not map to
any real memory, and accessing these regions generates a memory-violation error. The determination of the
memory space to which an address maps is controlled by a set of "memory aperture" base and size registers.

### 10.1. Flat Memory Instruction

Flat memory instructions let the kernel read or write data in memory, or perform atomic operations on data
already in memory. These operations occur through the texture L2 cache. The instruction declares which
VGPR holds the address (either 32- or 64-bit, depending on the memory configuration), the VGPR which sends
and the VGPR which receives data. Flat instructions also use M0 as described in the table below:

**Table 54. Flat, Global and Scratch Microcode Formats**

```
Field   Bit Size Description
OP      7        Opcode. Can be Flat, Scratch or Global instruction. See next table.
ADDR    8        VGPR which holds the address. For 64-bit addresses, ADDR has the LSBs, and ADDR+1 has the MSBs.
DATA    8        VGPR which holds the first Dword of data. Instructions can use 0-4 Dwords.
VDST    8        VGPR destination for data returned to the kernel, either from LOADs or Atomics with SC[0]=1 (return pre-
                 op value).
SC      2        Memory Scope
NT      1        Non-Temporal
ACC     1        DATA is Accumulation VGPR
SEG     2        Memory Segment: 0=FLAT, 1=SCRATCH, 2=GLOBAL, 3=reserved.
SVE     1        Scratch VGPR Enable - indicates if a VGPR contributes to calculating scratch memory addresses.
NV      1        Non-volatile. When set, the read/write is operating on non-volatile memory.
OFFSET 13        Address offset.
                 Scratch, Global: 13-bit signed byte offset.
                 Flat: 12-bit unsigned offset (MSB is ignored).
SADDR 7          Scalar SGPR that provides an offset address. To disable, set this field to 0x7F. Meaning of this field is
                 different for Scratch and Global:
                 Flat: Unused.
                 Scratch: Use an SGPR (instead of VGPR) for the address.
                 Global: Use the SGPR to provide a base address; the VGPR provides a 32-bit offset.
M0      16       Implied use of M0 for SCRATCH and GLOBAL only when LDS=1. Provides the LDS address-offset.
```

**Table 55. Flat, Global and Scratch Opcodes**

```
Flat Opcodes                         Global Opcodes                          Scratch Opcodes
FLAT                                 GLOBAL                                  SCRATCH
FLAT_LOAD_UBYTE                      GLOBAL_LOAD_UBYTE                       SCRATCH_LOAD_UBYTE
FLAT_LOAD_UBYTE_D16                  GLOBAL_LOAD_UBYTE_D16                   SCRATCH_LOAD_UBYTE_D16
FLAT_LOAD_UBYTE_D16_HI               GLOBAL_LOAD_UBYTE_D16_HI                SCRATCH_LOAD_UBYTE_D16_HI
FLAT_LOAD_SBYTE                      GLOBAL_LOAD_SBYTE                       SCRATCH_LOAD_SBYTE
FLAT_LOAD_SBYTE_D16                  GLOBAL_LOAD_SBYTE_D16                   SCRATCH_LOAD_SBYTE_D16
```

```
Flat Opcodes                         Global Opcodes                          Scratch Opcodes
FLAT_LOAD_SBYTE_D16_HI               GLOBAL_LOAD_SBYTE_D16_HI                SCRATCH_LOAD_SBYTE_D16_HI
FLAT_LOAD_USHORT                     GLOBAL_LOAD_USHORT                      SCRATCH_LOAD_USHORT
FLAT_LOAD_SSHORT                     GLOBAL_LOAD_SSHORT                      SCRATCH_LOAD_SSHORT
FLAT_LOAD_SHORT_D16                  GLOBAL_LOAD_SHORT_D16                   SCRATCH_LOAD_SHORT_D16
FLAT_LOAD_SHORT_D16_HI               GLOBAL_LOAD_SHORT_D16_HI                SCRATCH_LOAD_SHORT_D16_HI
FLAT_LOAD_DWORD                      GLOBAL_LOAD_DWORD                       SCRATCH_LOAD_DWORD
FLAT_LOAD_DWORDX2                    GLOBAL_LOAD_DWORDX2                     SCRATCH_LOAD_DWORDX2
FLAT_LOAD_DWORDX3                    GLOBAL_LOAD_DWORDX3                     SCRATCH_LOAD_DWORDX3
FLAT_LOAD_DWORDX4                    GLOBAL_LOAD_DWORDX4                     SCRATCH_LOAD_DWORDX4
FLAT_STORE_BYTE                      GLOBAL_STORE_BYTE                       SCRATCH_STORE_BYTE
FLAT_STORE_BYTE_D16_HI               GLOBAL_STORE_BYTE_D16_HI                SCRATCH_STORE_BYTE_D16_HI
FLAT_STORE_SHORT                     GLOBAL_STORE_SHORT                      SCRATCH_STORE_SHORT
FLAT_STORE_SHORT_D16_HI              GLOBAL_STORE_SHORT_D16_HI               SCRATCH_STORE_SHORT_D16_HI
FLAT_STORE_DWORD                     GLOBAL_STORE_DWORD                      SCRATCH_STORE_DWORD
FLAT_STORE_DWORDX2                   GLOBAL_STORE_DWORDX2                    SCRATCH_STORE_DWORDX2
FLAT_STORE_DWORDX3                   GLOBAL_STORE_DWORDX3                    SCRATCH_STORE_DWORDX3
FLAT_STORE_DWORDX4                   GLOBAL_STORE_DWORDX4                    SCRATCH_STORE_DWORDX4
FLAT_ATOMIC_SWAP                     GLOBAL_ATOMIC_SWAP                      none
FLAT_ATOMIC_CMPSWAP                  GLOBAL_ATOMIC_CMPSWAP                   none
FLAT_ATOMIC_ADD                      GLOBAL_ATOMIC_ADD                       none
FLAT_ATOMIC_SUB                      GLOBAL_ATOMIC_SUB                       none
FLAT_ATOMIC_SMIN                     GLOBAL_ATOMIC_SMIN                      none
FLAT_ATOMIC_UMIN                     GLOBAL_ATOMIC_UMIN                      none
FLAT_ATOMIC_SMAX                     GLOBAL_ATOMIC_SMAX                      none
FLAT_ATOMIC_UMAX                     GLOBAL_ATOMIC_UMAX                      none
FLAT_ATOMIC_AND                      GLOBAL_ATOMIC_AND                       none
FLAT_ATOMIC_OR                       GLOBAL_ATOMIC_OR                        none
FLAT_ATOMIC_XOR                      GLOBAL_ATOMIC_XOR                       none
FLAT_ATOMIC_INC                      GLOBAL_ATOMIC_INC                       none
FLAT_ATOMIC_DEC                      GLOBAL_ATOMIC_DEC                       none
FLAT_ATOMIC_ADD_F32                  GLOBAL_ATOMIC_ADD_F32                   none
FLAT_ATOMIC_PK_ADD_F16               GLOBAL_ATOMIC_PK_ADD_F16                none
FLAT_ATOMIC_PK_ADD_BF16              GLOBAL_ATOMIC_PK_ADD_BF16               none
FLAT_ATOMIC_ADD_F64                  GLOBAL_ATOMIC_ADD_F64                   none
FLAT_ATOMIC_MIN_F64                  GLOBAL_ATOMIC_MIN_F64                   none
FLAT_ATOMIC_MAX_F64                  GLOBAL_ATOMIC_MAX_F64                   none
                                     GLOBAL_LOAD_LDS_UBYTE                   SCRATCH_LOAD_LDS_UBYTE
                                     GLOBAL_LOAD_LDS_SBYTE                   SCRATCH_LOAD_LDS_SBYTE
                                     GLOBAL_LOAD_LDS_USHORT                  SCRATCH_LOAD_LDS_USHORT
                                     GLOBAL_LOAD_LDS_SSHORT                  SCRATCH_LOAD_LDS_SSHORT
                                     GLOBAL_LOAD_LDS_DWORD                   SCRATCH_LOAD_LDS_DWORD
                                     GLOBAL_LOAD_LDS_DWORDX3
                                     GLOBAL_LOAD_LDS_DWORDX4
The non-float atomic instructions above are also available in "_X2" versions (64-bit).
```

**Table 56. SVE Bit**

```
SADDR         SVE Mode
==EXEC_HI     0      ST : addr = flat_scratch + swizzle(inst.offset, threadID)
!=EXEC_HI     0      SS : addr = flat_scratch + swizzle(sgpr_offset + inst.offset, threadID)
==EXEC_HI     1      SV : addr = flat_scratch + swizzle(vgpr_offset + inst.offset, threadID)
!=EXEC_HI     1      SVS : addr = flat_scratch + swizzle(sgpr_offset + vgpr_offset + inst.offset, threadID)
```

### 10.2. Instructions

The FLAT instruction set is nearly identical to the Buffer instruction set, but without the FORMAT reads and
writes. Unlike Buffer instructions, FLAT instructions cannot return data directly to LDS, but only to VGPRs.

FLAT instructions do not use a resource constant (V#), however, they do require a specific SGPR-pair to hold
scratch-space information in case any threads' address resolves to scratch space. See the Scratch section for
details.

Internally, FLAT instruction are executed as both an LDS and a Buffer instruction; so, they increment both
VM_CNT and LGKM_CNT and are not considered done until both have been decremented. There is no way
beforehand to determine whether a FLAT instruction uses only LDS or TA memory space.

#### 10.2.1. Ordering

Flat instructions can complete out of order with each other. If one flat instruction finds all of its data in Texture
cache, and the next finds all of its data in LDS, the second instruction might complete first. If the two fetches
return data to the same VGPR, the result are unknown.

#### 10.2.2. Important Timing Consideration

Since the data for a FLAT load can come from either LDS or the texture cache, and because these units have
different latencies, there is a potential race condition with respect to the VM_CNT and LGKM_CNT counters.
Because of this, the only sensible S_WAITCNT value to use after FLAT instructions is zero.

### 10.3. Addressing

FLAT instructions support both 64- and 32-bit addressing. The address size is set using a mode register (PTR32),
and a local copy of the value is stored per wave.

The addresses for the aperture check differ in 32- and 64-bit mode; however, this is not covered here.

64-bit addresses are stored with the LSBs in the VGPR at ADDR, and the MSBs in the VGPR at ADDR+1.

For scratch space, the texture unit takes the address from the VGPR and does the following.

```
  Address = VGPR[addr] + TID_in_wave * Size
              - private aperture base (in SH_MEM_BASES)
              + offset (from flat_scratch)
```

Instructions which return data to LDS address LDS as:

```
  DWORDX1:     LDS_ADDR = LDSbase(hw alloc) + LDSoffset(M0[17:2] * 4) + INST.OFFSET + ThreadID*4
  DWORDX4:     LDS_ADDR = LDSbase(hw alloc) + LDSoffset(M0[17:2] * 4) + INST.OFFSET + ThreadID*16
```

#### 10.3.1. Atomics

Float atomics must set SC[0]=0 (no return value).

Memory atomics are performed in the data fabric so they are known to be atomic with host memory access.

FP32 atomic operations flush denormals to zero, and both FP64 and FP16 atomic do not flush denormals. The
rounding mode is fixed and "round to nearest even".

### 10.4. Global

Global instructions are similar to Flat instructions, but the programmer must ensure that no threads access
LDS space; thus, no LDS bandwidth is used by global instructions.

Global instructions offer two types of addressing:

- Memory_addr = VGPR-address + instruction offset.
- Memory_addr = SGPR-address + VGPR-offset + instruction offset.

The size of the address component is dependent on ADDRESS_MODE: 32-bits or 64-bit pointers. The VGPR-
offset is 32 bits.

These instructions also allow direct data movement between LDS and memory without going through VGPRs.

Since these instructions do not access LDS, only VM_CNT is used, not LGKM_CNT. If a global instruction does
attempt to access LDS, the instruction returns MEM_VIOL.

### 10.5. Scratch

Scratch instructions are similar to Flat, but the programmer must ensure that no threads access LDS space, and
the memory space is swizzled. Thus, no LDS bandwidth is used by scratch instructions.

Scratch instructions also support multi-Dword access and mis-aligned access (although mis-aligned is slower).

Scratch instructions use the following addressing:

- Memory_addr = flat_scratch.addr + swizzle(V/SGPR_offset + inst_offset, threadID)
- The offset can come from either an SGPR or a VGPR, and is a 32- bit unsigned byte.

The size of the address component is dependent on the ADDRESS_MODE: 32-bits or 64-bit pointers. The VGPR-
offset is 32 bits.

These instructions also allow direct data movement between LDS and memory without going through VGPRs.

Since these instructions do not access LDS, only VM_CNT is used, not LGKM_CNT. It is not possible for a
Scratch instruction to access LDS; thus, no error or aperture checking is done.

### 10.6. Data

FLAT instructions can use zero to four consecutive Dwords of data in VGPRs and/or memory. The DATA field
determines which VGPR(s) supply source data (if any), and the VDST VGPRs hold return data (if any). No data-
format conversion is done.

### 10.7. Scratch Space (Private)

Scratch (thread-private memory) is an area of memory defined by the aperture registers. When an address falls
in scratch space, additional address computation is automatically performed by the hardware. The kernel must
provide additional information for this computation to occur in the form of the FLAT_SCRATCH register.

The FLAT_SCRATCH address is automatically sent with every FLAT request.

FLAT_SCRATCH is a 64-bit, byte address. The shader composes the value by adding together two separate
values: the base address, which can be passed in via an initialized SGPR, or perhaps through a constant buffer,
and the per-wave allocation offset (also initialized in an SGPR).

## Chapter 11. Data Share Operations

Local data share (LDS) is a very low-latency, RAM scratchpad for temporary data with at least one order of
magnitude higher effective bandwidth than direct, uncached global memory. It permits sharing of data
between work-items in a work-group. Unlike read-only caches, the LDS permits high-speed write-to-read re-
use of the memory space (full gather/read/load and scatter/write/store operations).

### 11.1. Overview

The figure below shows the conceptual framework of the LDS is integration into the memory of AMD
Accelerators using OpenCL.

**Figure 6. High-Level Memory Configuration**

Physically located on-chip, directly next to the ALUs, the LDS can be approximately one order of magnitude
faster than global memory (assuming no bank conflicts).

There are 160 kB memory per compute unit, segmented into 64 banks of 640 Dwords, each bank being 32bits
wide. Dwords are placed in the banks serially, but all banks can execute a store or load simultaneously. One
work-group can request up to 160 kB memory. Reads across wavefront are dispatched over four cycles in
waterfall.

The high bandwidth of the LDS memory is achieved not only through its proximity to the ALUs, but also
through simultaneous access to its memory banks. Thus, it is possible to concurrently execute 32 write or read
instructions, each nominally 32-bits; extended instructions, read2/write2, can be 64-bits each. If, however,
more than one access attempt is made to the same bank at the same time, a bank conflict occurs. In this case,
for indexed and atomic operations, hardware prevents the attempted concurrent accesses to the same bank by

turning them into serial accesses. This can decrease the effective bandwidth of the LDS. To help achieve
optimal throughput (optimal efficiency), therefore, it is important to avoid bank conflicts. A knowledge of
request scheduling and address mapping can be key to help achieving this.

### 11.2. Dataflow in Memory Hierarchy

The figure below is a conceptual diagram of the dataflow within the memory structure.

To load data into LDS from global memory, it is read from global memory and placed into the work-item's
registers; then, a store is performed to LDS. Similarly, to store data into global memory, data is read from LDS
and placed into the workitem's registers, then placed into global memory. To make effective use of the LDS, a
kernel must perform many operations on what is transferred between global memory and LDS. It also is
possible to load data from a memory buffer directly into LDS, bypassing VGPRs.

LDS atomics are performed in the LDS hardware. (Thus, although ALUs are not directly used for these
operations, latency is incurred by the LDS executing this function.)

### 11.3. LDS Access

The LDS is accessed via indexed or atomic instructions.

#### 11.3.1. Data Share Indexed and Atomic Access

Indexed and atomic operations supply a unique address per work-item from the VGPRs to the LDS, and supply
or return unique data per work-item back to VGPRs. Due to the internal banked structure of LDS, operations
can complete in as little as two cycles, or take as many 64 cycles, depending upon the number of bank conflicts
(addresses that map to the same memory bank).

Indexed operations are simple LDS load and store operations that read data from, and return data to, VGPRs.

Atomic operations are arithmetic operations that combine data from VGPRs and data in LDS, and write the
result back to LDS. Atomic operations have the option of returning the LDS "pre-op" value to VGPRs.

The table below lists and briefly describes the LDS instruction fields.

**Table 57. LDS Instruction Fields**

```
Field                                             Size Description
OP                                                7       LDS opcode.
GDS                                               1       0 = LDS, 1 = Reserved.
OFFSET0                                           8       Immediate offset, in bytes. Instructions with one address combine
                                                          the offset fields into a single 16-bit unsigned offset: {offset1,
OFFSET1                                           8       offset0}. Instructions with two addresses (for example: READ2) use
                                                          the offsets separately as two 8- bit unsigned offsets.
                                                  VDS 8
                                                  T
VGPR to which result is written: either from      ADD 8
LDS-load or atomic return value.                  R
VGPR that supplies the byte address offset.       DAT 8
                                                  A0
VGPR that supplies first data source.             DAT 8
                                                  A1
VGPR that supplies second data source.            ( M0 32 Implied use of M0. M0[16:0] contains the byte-size of the LDS
                                                  )    segment. This is used to clamp the final address.
```

All LDS operations require that M0 be initialized prior to use. M0 contains a size value that can be used to
restrict access to a subset of the allocated LDS range. If no clamping is wanted, set M0 to 0xFFFFFFFF.

**Table 58. LDS Indexed Load/Store**

```
Load / Store                                          Description
DS_READ_{B32,B64,B96,B128,U8,I8,U16,I16}              Read one value per thread; sign extend to Dword, if signed.
DS_READ2_{B32,B64}                                    Read two values at unique addresses.
DS_READ2ST64_{B32,B64}                                Read 2 values at unique addresses; offset *= 64.
DS_WRITE_{B32,B64,B96,B128,B8,B16}                    Write one value.
DS_WRITE2_{B32,B64}                                   Write two values.
DS_WRITE2ST64_{B32,B64}                               Write two values, offset *= 64.
DS_WRXCHG2_RTN_{B32,B64}                              Exchange GPR with LDS-memory.
DS_WRXCHG2ST64_RTN_{B32,B64}                          Exchange GPR with LDS-memory; offset *= 64.
DS_PERMUTE_B32                                        Forward permute. Does not write any LDS memory.
                                                      LDS[dst] = src0
                                                      returnVal = LDS[thread_id]
                                                      where thread_id is 0..63.
DS_BPERMUTE_B32                                       Backward permute. Does not actually write any LDS memory.
                                                      LDS[thread_id] = src0
                                                      where thread_id is 0..63, and returnVal = LDS[dst].
```

Single Address Instructions

```
  LDS_Addr = LDS_BASE + VGPR[ADDR] + {InstrOffset1,InstrOffset0}
```

Double Address Instructions

```
  LDS_Addr0 = LDS_BASE + VGPR[ADDR] + InstrOffset0*ADJ +
  LDS_Addr1 = LDS_BASE + VGPR[ADDR] + InstrOffset1*ADJ
     Where ADJ = 4 for 8, 16 and 32-bit data types; and ADJ = 8 for 64-bit.
```

Note that LDS_ADDR1 is used only for READ2*, WRITE2*, and WREXCHG2*.

The address comes from VGPR, and both ADDR and InstrOffset are byte addresses.

At the time of wavefront creation, LDS_BASE is assigned to the physical LDS region owned by this wavefront or
work-group.

Specify only one address by setting both offsets to the same value. This causes only one read or write to occur
and uses only the first DATA0.

LDS Atomic Ops

DS_<atomicOp> OP, OFFSET0, OFFSET1, VDST, ADDR, Data0, Data1

Data size is encoded in atomicOp: byte, word, Dword, or double.

```
  LDS_Addr0 = LDS_BASE + VGPR[ADDR] + {InstrOffset1,InstrOffset0}
```

ADDR is a Dword address. VGPRs 0,1 and dst are double-GPRs for doubles data.

VGPR data sources can only be VGPRs or constant values, not SGPRs.

Denormal behavior for floating point atomics is controlled via the MODE register's FP_DENORM field. The
rounding mode is fixed at "round to nearest even".

### 11.4. MFMA Transpose Load from LDS

These instructions allow the user to perform matrix transpose while transferring 16, 8, 6 or 4-bit data from LDS
to VGPRs. The operation takes two instructions with different LDS addresses and VGPR destinations.

Prior to executing these instructions the EXEC mask must be set to all 1's. It is required that the LDS address be
aligned to the data size. Any DS op reading or writing 64-bit or larger data must use an even-aligned VGPR,
however DS_READ_B96_TR_B6 does not require even-VGPR alignment.

```
DS_READ_B64_TR_B16           Used for either column major matrix A or row major matrix B data load to 2 VGPRs.
                             Element size is 16b. Two instruction load a complete matrix. The first loads K=0..3 and
                             K=8..11 into two VGPRs, and the next loads K=4..7 and 12..15. Each lane (one VGPR) holds 4
                             consecutive M or N values.
DS_READ_B64_TR_B8            Used for either column major matrix A or row major matrix B data load to 2 VGPRs.
                             Element size is 8b. Two instruction load a complete matrix. The first loads K values (0..7,
                             16..23, 32..39 and 48..55) into two VGPRs, and the next instruction loads the other K values.
DS_READ_B64_TR_B4            Used for either column major matrix A or row major matrix B data load to 2 VGPRs.
                             Element size is 4b. Two instruction load a complete matrix. The first loads K values (0..15,
                             32..47) into two VGPRs, and the next instruction loads the other K values.
```

```
DS_READ_B96_TR_B6            Used for either column major matrix A or row major matrix B data load to 3 VGPRs.
                             Element size is 6b. Two instruction load a complete matrix. The first loads K values (0..15,
                             32..47) into three VGPRs, and the next instruction loads the other K values.
```
