# CDNA4 ISA Instructions: Buffer & Flat (MUBUF/MTBUF/FLAT/Scratch/Global)

> Part of the **CDNA4 Instruction Set Architecture** reference. See the [index](README.md) for other sections.

## Sections in this file

  - [12.13. MUBUF Instructions](#1213-mubuf-instructions)
  - [12.14. MTBUF Instructions](#1214-mtbuf-instructions)
  - [12.15. FLAT, Scratch and Global Instructions](#1215-flat-scratch-and-global-instructions)
  - [12.16. Instruction Limitations](#1216-instruction-limitations)

## Instruction mnemonics in this file

- **12.13. MUBUF Instructions**: BUFFER_LOAD_FORMAT_X, BUFFER_LOAD_FORMAT_XY, BUFFER_LOAD_FORMAT_XYZ, BUFFER_LOAD_FORMAT_XYZW, BUFFER_STORE_FORMAT_X, BUFFER_STORE_FORMAT_XY, BUFFER_STORE_FORMAT_XYZ, BUFFER_STORE_FORMAT_XYZW, BUFFER_LOAD_FORMAT_D16_X, BUFFER_LOAD_FORMAT_D16_XY, BUFFER_LOAD_FORMAT_D16_XYZ, BUFFER_LOAD_FORMAT_D16_XYZW, BUFFER_STORE_FORMAT_D16_X, BUFFER_STORE_FORMAT_D16_XY, BUFFER_STORE_FORMAT_D16_XYZ, BUFFER_STORE_FORMAT_D16_XYZW, BUFFER_LOAD_UBYTE, BUFFER_LOAD_SBYTE, BUFFER_LOAD_USHORT, BUFFER_LOAD_SSHORT, BUFFER_LOAD_DWORD, BUFFER_LOAD_DWORDX2, BUFFER_LOAD_DWORDX3, BUFFER_LOAD_DWORDX4, BUFFER_STORE_BYTE, BUFFER_STORE_BYTE_D16_HI, BUFFER_STORE_SHORT, BUFFER_STORE_SHORT_D16_HI, BUFFER_STORE_DWORD, BUFFER_STORE_DWORDX2, BUFFER_STORE_DWORDX3, BUFFER_STORE_DWORDX4, BUFFER_LOAD_UBYTE_D16, BUFFER_LOAD_UBYTE_D16_HI, BUFFER_LOAD_SBYTE_D16, BUFFER_LOAD_SBYTE_D16_HI, BUFFER_LOAD_SHORT_D16, BUFFER_LOAD_SHORT_D16_HI, BUFFER_LOAD_FORMAT_D16_HI_X, BUFFER_STORE_FORMAT_D16_HI_X, BUFFER_WBL2, BUFFER_INV, BUFFER_ATOMIC_SWAP, BUFFER_ATOMIC_CMPSWAP, BUFFER_ATOMIC_ADD, BUFFER_ATOMIC_SUB, BUFFER_ATOMIC_SMIN, BUFFER_ATOMIC_UMIN, BUFFER_ATOMIC_SMAX, BUFFER_ATOMIC_UMAX, BUFFER_ATOMIC_AND, BUFFER_ATOMIC_OR, BUFFER_ATOMIC_XOR, BUFFER_ATOMIC_INC, BUFFER_ATOMIC_DEC, BUFFER_ATOMIC_ADD_F32, BUFFER_ATOMIC_PK_ADD_F16, BUFFER_ATOMIC_ADD_F64, BUFFER_ATOMIC_MIN_F64, BUFFER_ATOMIC_MAX_F64, BUFFER_ATOMIC_PK_ADD_BF16, BUFFER_ATOMIC_SWAP_X2, BUFFER_ATOMIC_CMPSWAP_X2, BUFFER_ATOMIC_ADD_X2, BUFFER_ATOMIC_SUB_X2, BUFFER_ATOMIC_SMIN_X2, BUFFER_ATOMIC_UMIN_X2, BUFFER_ATOMIC_SMAX_X2, BUFFER_ATOMIC_UMAX_X2, BUFFER_ATOMIC_AND_X2, BUFFER_ATOMIC_OR_X2, BUFFER_ATOMIC_XOR_X2, BUFFER_ATOMIC_INC_X2, BUFFER_ATOMIC_DEC_X2
- **12.14. MTBUF Instructions**: TBUFFER_LOAD_FORMAT_X, TBUFFER_LOAD_FORMAT_XY, TBUFFER_LOAD_FORMAT_XYZ, TBUFFER_LOAD_FORMAT_XYZW, TBUFFER_STORE_FORMAT_X, TBUFFER_STORE_FORMAT_XY, TBUFFER_STORE_FORMAT_XYZ, TBUFFER_STORE_FORMAT_XYZW, TBUFFER_LOAD_FORMAT_D16_X, TBUFFER_LOAD_FORMAT_D16_XY, TBUFFER_LOAD_FORMAT_D16_XYZ, TBUFFER_LOAD_FORMAT_D16_XYZW, TBUFFER_STORE_FORMAT_D16_X, TBUFFER_STORE_FORMAT_D16_XY, TBUFFER_STORE_FORMAT_D16_XYZ, TBUFFER_STORE_FORMAT_D16_XYZW
- **12.15. FLAT, Scratch and Global Instructions**: FLAT_LOAD_UBYTE, FLAT_LOAD_SBYTE, FLAT_LOAD_USHORT, FLAT_LOAD_SSHORT, FLAT_LOAD_DWORD, FLAT_LOAD_DWORDX2, FLAT_LOAD_DWORDX3, FLAT_LOAD_DWORDX4, FLAT_STORE_BYTE, FLAT_STORE_BYTE_D16_HI, FLAT_STORE_SHORT, FLAT_STORE_SHORT_D16_HI, FLAT_STORE_DWORD, FLAT_STORE_DWORDX2, FLAT_STORE_DWORDX3, FLAT_STORE_DWORDX4, FLAT_LOAD_UBYTE_D16, FLAT_LOAD_UBYTE_D16_HI, FLAT_LOAD_SBYTE_D16, FLAT_LOAD_SBYTE_D16_HI, FLAT_LOAD_SHORT_D16, FLAT_LOAD_SHORT_D16_HI, FLAT_ATOMIC_SWAP, FLAT_ATOMIC_CMPSWAP, FLAT_ATOMIC_ADD, FLAT_ATOMIC_SUB, FLAT_ATOMIC_SMIN, FLAT_ATOMIC_UMIN, FLAT_ATOMIC_SMAX, FLAT_ATOMIC_UMAX, FLAT_ATOMIC_AND, FLAT_ATOMIC_OR, FLAT_ATOMIC_XOR, FLAT_ATOMIC_INC, FLAT_ATOMIC_DEC, FLAT_ATOMIC_ADD_F32, FLAT_ATOMIC_PK_ADD_F16, FLAT_ATOMIC_ADD_F64, FLAT_ATOMIC_MIN_F64, FLAT_ATOMIC_MAX_F64, FLAT_ATOMIC_PK_ADD_BF16, FLAT_ATOMIC_SWAP_X2, FLAT_ATOMIC_CMPSWAP_X2, FLAT_ATOMIC_ADD_X2, FLAT_ATOMIC_SUB_X2, FLAT_ATOMIC_SMIN_X2, FLAT_ATOMIC_UMIN_X2, FLAT_ATOMIC_SMAX_X2, FLAT_ATOMIC_UMAX_X2, FLAT_ATOMIC_AND_X2, FLAT_ATOMIC_OR_X2, FLAT_ATOMIC_XOR_X2, FLAT_ATOMIC_INC_X2, FLAT_ATOMIC_DEC_X2, SCRATCH_LOAD_UBYTE, SCRATCH_LOAD_SBYTE, SCRATCH_LOAD_USHORT, SCRATCH_LOAD_SSHORT, SCRATCH_LOAD_DWORD, SCRATCH_LOAD_DWORDX2, SCRATCH_LOAD_DWORDX3, SCRATCH_LOAD_DWORDX4, SCRATCH_STORE_BYTE, SCRATCH_STORE_BYTE_D16_HI, SCRATCH_STORE_SHORT, SCRATCH_STORE_SHORT_D16_HI, SCRATCH_STORE_DWORD, SCRATCH_STORE_DWORDX2, SCRATCH_STORE_DWORDX3, SCRATCH_STORE_DWORDX4, SCRATCH_LOAD_UBYTE_D16, SCRATCH_LOAD_UBYTE_D16_HI, SCRATCH_LOAD_SBYTE_D16, SCRATCH_LOAD_SBYTE_D16_HI, SCRATCH_LOAD_SHORT_D16, SCRATCH_LOAD_SHORT_D16_HI, SCRATCH_LOAD_LDS_UBYTE, SCRATCH_LOAD_LDS_SBYTE, SCRATCH_LOAD_LDS_USHORT, SCRATCH_LOAD_LDS_SSHORT, SCRATCH_LOAD_LDS_DWORD, GLOBAL_LOAD_UBYTE, GLOBAL_LOAD_SBYTE, GLOBAL_LOAD_USHORT, GLOBAL_LOAD_SSHORT, GLOBAL_LOAD_DWORD, GLOBAL_LOAD_DWORDX2, GLOBAL_LOAD_DWORDX3, GLOBAL_LOAD_DWORDX4, GLOBAL_STORE_BYTE, GLOBAL_STORE_BYTE_D16_HI, GLOBAL_STORE_SHORT, GLOBAL_STORE_SHORT_D16_HI, GLOBAL_STORE_DWORD, GLOBAL_STORE_DWORDX2, GLOBAL_STORE_DWORDX3, GLOBAL_STORE_DWORDX4, GLOBAL_LOAD_UBYTE_D16, GLOBAL_LOAD_UBYTE_D16_HI, GLOBAL_LOAD_SBYTE_D16, GLOBAL_LOAD_SBYTE_D16_HI, GLOBAL_LOAD_SHORT_D16, GLOBAL_LOAD_SHORT_D16_HI, GLOBAL_LOAD_LDS_UBYTE, GLOBAL_LOAD_LDS_SBYTE, GLOBAL_LOAD_LDS_USHORT, GLOBAL_LOAD_LDS_SSHORT, GLOBAL_LOAD_LDS_DWORD, GLOBAL_ATOMIC_SWAP, GLOBAL_ATOMIC_CMPSWAP, GLOBAL_ATOMIC_ADD, GLOBAL_ATOMIC_SUB, GLOBAL_ATOMIC_SMIN, GLOBAL_ATOMIC_UMIN, GLOBAL_ATOMIC_SMAX, GLOBAL_ATOMIC_UMAX, GLOBAL_ATOMIC_AND, GLOBAL_ATOMIC_OR, GLOBAL_ATOMIC_XOR, GLOBAL_ATOMIC_INC, GLOBAL_ATOMIC_DEC, GLOBAL_ATOMIC_ADD_F32, GLOBAL_ATOMIC_PK_ADD_F16, GLOBAL_ATOMIC_ADD_F64, GLOBAL_ATOMIC_MIN_F64, GLOBAL_ATOMIC_MAX_F64, GLOBAL_ATOMIC_PK_ADD_BF16, GLOBAL_ATOMIC_SWAP_X2, GLOBAL_ATOMIC_CMPSWAP_X2, GLOBAL_ATOMIC_ADD_X2, GLOBAL_ATOMIC_SUB_X2, GLOBAL_ATOMIC_SMIN_X2, GLOBAL_ATOMIC_UMIN_X2, GLOBAL_ATOMIC_SMAX_X2, GLOBAL_ATOMIC_UMAX_X2, GLOBAL_ATOMIC_AND_X2, GLOBAL_ATOMIC_OR_X2, GLOBAL_ATOMIC_XOR_X2, GLOBAL_ATOMIC_INC_X2, GLOBAL_ATOMIC_DEC_X2, GLOBAL_LOAD_LDS_DWORDX4, GLOBAL_LOAD_LDS_DWORDX3

---

### 12.13. MUBUF Instructions

The bitfield map of the MUBUF format is:

```
      where:
```

```
      OFFSET   = Unsigned immediate byte offset.
      OFFEN    = Send offset either as VADDR or as zero..
      IDXEN    = Send index either as VADDR or as zero.
      LDS      = Data read from/written to LDS or VGPR.
      OP       = Instruction Opcode.
      VADDR    = VGPR address source.
      VDATA    = Destination vector GPR.
      SRSRC    = Scalar GPR that specifies resource constant.
      ACC      = Return to ACC VGPRs
      SC       = Scope
      NT       = Non-Temporal
      SOFFSET = Byte offset added to the memory address of an SGPR.
```

#### BUFFER_LOAD_FORMAT_X  (opcode 0)

Load 1-component formatted data from a buffer surface, convert the data to 32 bit integral or floating point
format, then store the result into a vector register. The resource descriptor specifies the data format of the
surface.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  VDATA[31 : 0].b32 = ConvertFromFormat(MEM[addr + ChannelOffsetX()]);
  // Mem access size depends on format
```

#### BUFFER_LOAD_FORMAT_XY  (opcode 1)

Load 2-component formatted data from a buffer surface, convert the data to 32 bit integral or floating point
format, then store the result into a vector register. The resource descriptor specifies the data format of the
surface.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  VDATA[31 : 0].b32 = ConvertFromFormat(MEM[addr + ChannelOffsetX()]);
  // Mem access size depends on format
  VDATA[63 : 32].b32 = ConvertFromFormat(MEM[addr + ChannelOffsetY()])
```

#### BUFFER_LOAD_FORMAT_XYZ  (opcode 2)

Load 3-component formatted data from a buffer surface, convert the data to 32 bit integral or floating point
format, then store the result into a vector register. The resource descriptor specifies the data format of the
surface.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  VDATA[31 : 0].b32 = ConvertFromFormat(MEM[addr + ChannelOffsetX()]);
  // Mem access size depends on format
  VDATA[63 : 32].b32 = ConvertFromFormat(MEM[addr + ChannelOffsetY()]);
  VDATA[95 : 64].b32 = ConvertFromFormat(MEM[addr + ChannelOffsetZ()])
```

#### BUFFER_LOAD_FORMAT_XYZW  (opcode 3)

Load 4-component formatted data from a buffer surface, convert the data to 32 bit integral or floating point
format, then store the result into a vector register. The resource descriptor specifies the data format of the
surface.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  VDATA[31 : 0].b32 = ConvertFromFormat(MEM[addr + ChannelOffsetX()]);
  // Mem access size depends on format
  VDATA[63 : 32].b32 = ConvertFromFormat(MEM[addr + ChannelOffsetY()]);
  VDATA[95 : 64].b32 = ConvertFromFormat(MEM[addr + ChannelOffsetZ()]);
  VDATA[127 : 96].b32 = ConvertFromFormat(MEM[addr + ChannelOffsetW()])
```

#### BUFFER_STORE_FORMAT_X  (opcode 4)

Convert 32 bits of data from vector input registers into 1-component formatted data and store the data into a
buffer surface. The instruction specifies the data format of the surface, overriding the resource descriptor.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  MEM[addr + ChannelOffsetX()] = ConvertToFormat(VDATA[31 : 0].b32);
  // Mem access size depends on format
```

#### BUFFER_STORE_FORMAT_XY  (opcode 5)

Convert 64 bits of data from vector input registers into 2-component formatted data and store the data into a
buffer surface. The instruction specifies the data format of the surface, overriding the resource descriptor.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  MEM[addr + ChannelOffsetX()] = ConvertToFormat(VDATA[31 : 0].b32);
  // Mem access size depends on format
  MEM[addr + ChannelOffsetY()] = ConvertToFormat(VDATA[63 : 32].b32)
```

#### BUFFER_STORE_FORMAT_XYZ  (opcode 6)

Convert 96 bits of data from vector input registers into 3-component formatted data and store the data into a
buffer surface. The instruction specifies the data format of the surface, overriding the resource descriptor.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  MEM[addr + ChannelOffsetX()] = ConvertToFormat(VDATA[31 : 0].b32);
  // Mem access size depends on format
  MEM[addr + ChannelOffsetY()] = ConvertToFormat(VDATA[63 : 32].b32);
  MEM[addr + ChannelOffsetZ()] = ConvertToFormat(VDATA[95 : 64].b32)
```

#### BUFFER_STORE_FORMAT_XYZW  (opcode 7)

Convert 128 bits of data from vector input registers into 4-component formatted data and store the data into a
buffer surface. The instruction specifies the data format of the surface, overriding the resource descriptor.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  MEM[addr + ChannelOffsetX()] = ConvertToFormat(VDATA[31 : 0].b32);
  // Mem access size depends on format
  MEM[addr + ChannelOffsetY()] = ConvertToFormat(VDATA[63 : 32].b32);
  MEM[addr + ChannelOffsetZ()] = ConvertToFormat(VDATA[95 : 64].b32);
  MEM[addr + ChannelOffsetW()] = ConvertToFormat(VDATA[127 : 96].b32)
```

#### BUFFER_LOAD_FORMAT_D16_X  (opcode 8)

Load 1-component formatted data from a buffer surface, convert the data to packed 16 bit integral or floating
point format, then store the result into the low 16 bits of a 32-bit vector register. The resource descriptor
specifies the data format of the surface.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  VDATA[15 : 0].b16 = 16'B(ConvertFromFormat(MEM[addr + ChannelOffsetX()]));
  // Mem access size depends on format
  // VDATA[31:16].b16 is preserved.
```

#### BUFFER_LOAD_FORMAT_D16_XY  (opcode 9)

Load 2-component formatted data from a buffer surface, convert the data to packed 16 bit integral or floating
point format, then store the result into a vector register. The resource descriptor specifies the data format of
the surface.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  VDATA[15 : 0].b16 = 16'B(ConvertFromFormat(MEM[addr + ChannelOffsetX()]));
```

```
  // Mem access size depends on format
  VDATA[31 : 16].b16 = 16'B(ConvertFromFormat(MEM[addr + ChannelOffsetY()]))
```

#### BUFFER_LOAD_FORMAT_D16_XYZ  (opcode 10)

Load 3-component formatted data from a buffer surface, convert the data to packed 16 bit integral or floating
point format, then store the result into a vector register. The resource descriptor specifies the data format of
the surface.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  VDATA[15 : 0].b16 = 16'B(ConvertFromFormat(MEM[addr + ChannelOffsetX()]));
  // Mem access size depends on format
  VDATA[31 : 16].b16 = 16'B(ConvertFromFormat(MEM[addr + ChannelOffsetY()]));
  VDATA[47 : 32].b16 = 16'B(ConvertFromFormat(MEM[addr + ChannelOffsetZ()]));
  // VDATA[63:48].b16 is preserved.
```

#### BUFFER_LOAD_FORMAT_D16_XYZW  (opcode 11)

Load 4-component formatted data from a buffer surface, convert the data to packed 16 bit integral or floating
point format, then store the result into a vector register. The resource descriptor specifies the data format of
the surface.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  VDATA[15 : 0].b16 = 16'B(ConvertFromFormat(MEM[addr + ChannelOffsetX()]));
  // Mem access size depends on format
  VDATA[31 : 16].b16 = 16'B(ConvertFromFormat(MEM[addr + ChannelOffsetY()]));
  VDATA[47 : 32].b16 = 16'B(ConvertFromFormat(MEM[addr + ChannelOffsetZ()]));
  VDATA[63 : 48].b16 = 16'B(ConvertFromFormat(MEM[addr + ChannelOffsetW()]))
```

#### BUFFER_STORE_FORMAT_D16_X  (opcode 12)

Convert 16 bits of data from the low 16 bits of a 32-bit vector input register into 1-component formatted data
and store the data into a buffer surface. The instruction specifies the data format of the surface, overriding the
resource descriptor.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  MEM[addr + ChannelOffsetX()] = ConvertToFormat(32'B(VDATA[15 : 0].b16));
  // Mem access size depends on format
```

#### BUFFER_STORE_FORMAT_D16_XY  (opcode 13)

Convert 32 bits of data from vector input registers into 2-component formatted data and store the data into a
buffer surface. The instruction specifies the data format of the surface, overriding the resource descriptor.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  MEM[addr + ChannelOffsetX()] = ConvertToFormat(32'B(VDATA[15 : 0].b16));
  // Mem access size depends on format
  MEM[addr + ChannelOffsetY()] = ConvertToFormat(32'B(VDATA[31 : 16].b16))
```

#### BUFFER_STORE_FORMAT_D16_XYZ  (opcode 14)

Convert 48 bits of data from vector input registers into 3-component formatted data and store the data into a
buffer surface. The instruction specifies the data format of the surface, overriding the resource descriptor.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  MEM[addr + ChannelOffsetX()] = ConvertToFormat(32'B(VDATA[15 : 0].b16));
  // Mem access size depends on format
  MEM[addr + ChannelOffsetY()] = ConvertToFormat(32'B(VDATA[31 : 16].b16));
  MEM[addr + ChannelOffsetZ()] = ConvertToFormat(32'B(VDATA[47 : 32].b16))
```

#### BUFFER_STORE_FORMAT_D16_XYZW  (opcode 15)

Convert 64 bits of data from vector input registers into 4-component formatted data and store the data into a
buffer surface. The instruction specifies the data format of the surface, overriding the resource descriptor.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  MEM[addr + ChannelOffsetX()] = ConvertToFormat(32'B(VDATA[15 : 0].b16));
  // Mem access size depends on format
  MEM[addr + ChannelOffsetY()] = ConvertToFormat(32'B(VDATA[31 : 16].b16));
  MEM[addr + ChannelOffsetZ()] = ConvertToFormat(32'B(VDATA[47 : 32].b16));
  MEM[addr + ChannelOffsetW()] = ConvertToFormat(32'B(VDATA[63 : 48].b16))
```

#### BUFFER_LOAD_UBYTE  (opcode 16)

Load 8 bits of unsigned data from a buffer surface, zero extend to 32 bits and store the result into a vector
register.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  VDATA.u32 = 32'U({ 24'0U, MEM[addr].u8 })
```

#### BUFFER_LOAD_SBYTE  (opcode 17)

Load 8 bits of signed data from a buffer surface, sign extend to 32 bits and store the result into a vector register.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  VDATA.i32 = 32'I(signext(MEM[addr].i8))
```

#### BUFFER_LOAD_USHORT  (opcode 18)

Load 16 bits of unsigned data from a buffer surface, zero extend to 32 bits and store the result into a vector
register.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  VDATA.u32 = 32'U({ 16'0U, MEM[addr].u16 })
```

#### BUFFER_LOAD_SSHORT  (opcode 19)

Load 16 bits of signed data from a buffer surface, sign extend to 32 bits and store the result into a vector
register.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  VDATA.i32 = 32'I(signext(MEM[addr].i16))
```

#### BUFFER_LOAD_DWORD  (opcode 20)

Load 32 bits of data from a buffer surface into a vector register.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  VDATA[31 : 0] = MEM[addr].b32
```

#### BUFFER_LOAD_DWORDX2  (opcode 21)

Load 64 bits of data from a buffer surface into a vector register.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  VDATA[31 : 0] = MEM[addr].b32;
  VDATA[63 : 32] = MEM[addr + 4U].b32
```

#### BUFFER_LOAD_DWORDX3  (opcode 22)

Load 96 bits of data from a buffer surface into a vector register.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  VDATA[31 : 0] = MEM[addr].b32;
  VDATA[63 : 32] = MEM[addr + 4U].b32;
  VDATA[95 : 64] = MEM[addr + 8U].b32
```

#### BUFFER_LOAD_DWORDX4  (opcode 23)

Load 128 bits of data from a buffer surface into a vector register.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  VDATA[31 : 0] = MEM[addr].b32;
  VDATA[63 : 32] = MEM[addr + 4U].b32;
  VDATA[95 : 64] = MEM[addr + 8U].b32;
  VDATA[127 : 96] = MEM[addr + 12U].b32
```

#### BUFFER_STORE_BYTE  (opcode 24)

Store 8 bits of data from a vector register into a buffer surface.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  MEM[addr].b8 = VDATA[7 : 0]
```

#### BUFFER_STORE_BYTE_D16_HI  (opcode 25)

Store 8 bits of data from the high 16 bits of a 32-bit vector register into a buffer surface.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  MEM[addr].b8 = VDATA[23 : 16]
```

#### BUFFER_STORE_SHORT  (opcode 26)

Store 16 bits of data from a vector register into a buffer surface.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
```

```
  MEM[addr].b16 = VDATA[15 : 0]
```

#### BUFFER_STORE_SHORT_D16_HI  (opcode 27)

Store 16 bits of data from the high 16 bits of a 32-bit vector register into a buffer surface.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  MEM[addr].b16 = VDATA[31 : 16]
```

#### BUFFER_STORE_DWORD  (opcode 28)

Store 32 bits of data from vector input registers into a buffer surface.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  MEM[addr].b32 = VDATA[31 : 0]
```

#### BUFFER_STORE_DWORDX2  (opcode 29)

Store 64 bits of data from vector input registers into a buffer surface.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  MEM[addr].b32 = VDATA[31 : 0];
  MEM[addr + 4U].b32 = VDATA[63 : 32]
```

#### BUFFER_STORE_DWORDX3  (opcode 30)

Store 96 bits of data from vector input registers into a buffer surface.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  MEM[addr].b32 = VDATA[31 : 0];
  MEM[addr + 4U].b32 = VDATA[63 : 32];
  MEM[addr + 8U].b32 = VDATA[95 : 64]
```

#### BUFFER_STORE_DWORDX4  (opcode 31)

Store 128 bits of data from vector input registers into a buffer surface.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  MEM[addr].b32 = VDATA[31 : 0];
  MEM[addr + 4U].b32 = VDATA[63 : 32];
  MEM[addr + 8U].b32 = VDATA[95 : 64];
  MEM[addr + 12U].b32 = VDATA[127 : 96]
```

#### BUFFER_LOAD_UBYTE_D16  (opcode 32)

Load 8 bits of unsigned data from a buffer surface, zero extend to 16 bits and store the result into the low 16 bits
of a 32-bit vector register.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  VDATA[15 : 0].u16 = 16'U({ 8'0U, MEM[addr].u8 });
  // VDATA[31:16] is preserved.
```

#### BUFFER_LOAD_UBYTE_D16_HI  (opcode 33)

Load 8 bits of unsigned data from a buffer surface, zero extend to 16 bits and store the result into the high 16
bits of a 32-bit vector register.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  VDATA[31 : 16].u16 = 16'U({ 8'0U, MEM[addr].u8 });
  // VDATA[15:0] is preserved.
```

#### BUFFER_LOAD_SBYTE_D16  (opcode 34)

Load 8 bits of signed data from a buffer surface, sign extend to 16 bits and store the result into the low 16 bits of
a 32-bit vector register.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  VDATA[15 : 0].i16 = 16'I(signext(MEM[addr].i8));
  // VDATA[31:16] is preserved.
```

#### BUFFER_LOAD_SBYTE_D16_HI  (opcode 35)

Load 8 bits of signed data from a buffer surface, sign extend to 16 bits and store the result into the high 16 bits
of a 32-bit vector register.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
```

```
  VDATA[31 : 16].i16 = 16'I(signext(MEM[addr].i8));
  // VDATA[15:0] is preserved.
```

#### BUFFER_LOAD_SHORT_D16  (opcode 36)

Load 16 bits of unsigned data from a buffer surface and store the result into the low 16 bits of a 32-bit vector
register.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  VDATA[15 : 0].b16 = MEM[addr].b16;
  // VDATA[31:16] is preserved.
```

#### BUFFER_LOAD_SHORT_D16_HI  (opcode 37)

Load 16 bits of unsigned data from a buffer surface and store the result into the high 16 bits of a 32-bit vector
register.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  VDATA[31 : 16].b16 = MEM[addr].b16;
  // VDATA[15:0] is preserved.
```

#### BUFFER_LOAD_FORMAT_D16_HI_X  (opcode 38)

Load 1-component formatted data from a buffer surface, convert the data to packed 16 bit integral or floating
point format, then store the result into the high 16 bits of a 32-bit vector register. The resource descriptor
specifies the data format of the surface.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  VDATA[31 : 16].b16 = 16'B(ConvertFromFormat(MEM[addr + ChannelOffsetX()]));
  // Mem access size depends on format
  // VDATA[15:0].b16 is preserved.
```

#### BUFFER_STORE_FORMAT_D16_HI_X  (opcode 39)

Convert 16 bits of data from the high 16 bits of a 32-bit vector input register into 1-component formatted data
and store the data into a buffer surface. The instruction specifies the data format of the surface, overriding the
resource descriptor.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  MEM[addr + ChannelOffsetX()] = ConvertToFormat(32'B(VDATA[31 : 16].b16));
```

```
  // Mem access size depends on format
```

#### BUFFER_WBL2  (opcode 40)

Write back L2 cache. Returns ACK to shader.

#### BUFFER_INV  (opcode 41)

Invalidate CU and/or L2 cache depending on sc0 and sc1 bits. Returns ACK to shader.

#### BUFFER_ATOMIC_SWAP  (opcode 64)

Swap an unsigned 32-bit integer value in the data register with a location in a buffer surface. Store the original
value from buffer surface into a vector register iff the SC0 bit is set.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = DATA.b32;
  RETURN_DATA.b32 = tmp
```

#### BUFFER_ATOMIC_CMPSWAP  (opcode 65)

Compare two unsigned 32-bit integer values stored in the data comparison register and a location in a buffer
surface. Modify the memory location with a value in the data source register iff the comparison is equal. Store
the original value from buffer surface into a vector register iff the SC0 bit is set.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  tmp = MEM[addr].u32;
  src = DATA[31 : 0].u32;
  cmp = DATA[63 : 32].u32;
  MEM[addr].u32 = tmp == cmp ? src : tmp;
  RETURN_DATA.u32 = tmp
```

#### BUFFER_ATOMIC_ADD  (opcode 66)

Add two unsigned 32-bit integer values stored in the data register and a location in a buffer surface. Store the
original value from buffer surface into a vector register iff the SC0 bit is set.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
```

```
  tmp = MEM[addr].u32;
  MEM[addr].u32 += DATA.u32;
  RETURN_DATA.u32 = tmp
```

#### BUFFER_ATOMIC_SUB  (opcode 67)

Subtract an unsigned 32-bit integer value stored in the data register from a value stored in a location in a buffer
surface. Store the original value from buffer surface into a vector register iff the SC0 bit is set.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  tmp = MEM[addr].u32;
  MEM[addr].u32 -= DATA.u32;
  RETURN_DATA.u32 = tmp
```

#### BUFFER_ATOMIC_SMIN  (opcode 68)

Select the minimum of two signed 32-bit integer inputs, given two values stored in the data register and a
location in a buffer surface. Update the buffer surface with the selected value. Store the original value from
buffer surface into a vector register iff the SC0 bit is set.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  tmp = MEM[addr].i32;
  src = DATA.i32;
  MEM[addr].i32 = src < tmp ? src : tmp;
  RETURN_DATA.i32 = tmp
```

#### BUFFER_ATOMIC_UMIN  (opcode 69)

Select the minimum of two unsigned 32-bit integer inputs, given two values stored in the data register and a
location in a buffer surface. Update the buffer surface with the selected value. Store the original value from
buffer surface into a vector register iff the SC0 bit is set.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = src < tmp ? src : tmp;
  RETURN_DATA.u32 = tmp
```

#### BUFFER_ATOMIC_SMAX  (opcode 70)

Select the maximum of two signed 32-bit integer inputs, given two values stored in the data register and a

location in a buffer surface. Update the buffer surface with the selected value. Store the original value from
buffer surface into a vector register iff the SC0 bit is set.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  tmp = MEM[addr].i32;
  src = DATA.i32;
  MEM[addr].i32 = src >= tmp ? src : tmp;
  RETURN_DATA.i32 = tmp
```

#### BUFFER_ATOMIC_UMAX  (opcode 71)

Select the maximum of two unsigned 32-bit integer inputs, given two values stored in the data register and a
location in a buffer surface. Update the buffer surface with the selected value. Store the original value from
buffer surface into a vector register iff the SC0 bit is set.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = src >= tmp ? src : tmp;
  RETURN_DATA.u32 = tmp
```

#### BUFFER_ATOMIC_AND  (opcode 72)

Calculate bitwise AND given two unsigned 32-bit integer values stored in the data register and a location in a
buffer surface. Store the original value from buffer surface into a vector register iff the SC0 bit is set.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = (tmp & DATA.b32);
  RETURN_DATA.b32 = tmp
```

#### BUFFER_ATOMIC_OR  (opcode 73)

Calculate bitwise OR given two unsigned 32-bit integer values stored in the data register and a location in a
buffer surface. Store the original value from buffer surface into a vector register iff the SC0 bit is set.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = (tmp | DATA.b32);
  RETURN_DATA.b32 = tmp
```

#### BUFFER_ATOMIC_XOR  (opcode 74)

Calculate bitwise XOR given two unsigned 32-bit integer values stored in the data register and a location in a
buffer surface. Store the original value from buffer surface into a vector register iff the SC0 bit is set.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = (tmp ^ DATA.b32);
  RETURN_DATA.b32 = tmp
```

#### BUFFER_ATOMIC_INC  (opcode 75)

Increment an unsigned 32-bit integer value from a location in a buffer surface with wraparound to 0 if the
value exceeds a value in the data register. Store the original value from buffer surface into a vector register iff
the SC0 bit is set.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = tmp >= src ? 0U : tmp + 1U;
  RETURN_DATA.u32 = tmp
```

#### BUFFER_ATOMIC_DEC  (opcode 76)

Decrement an unsigned 32-bit integer value from a location in a buffer surface with wraparound to a value in
the data register if the decrement yields a negative value. Store the original value from buffer surface into a
vector register iff the SC0 bit is set.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = ((tmp == 0U) || (tmp > src)) ? src : tmp - 1U;
  RETURN_DATA.u32 = tmp
```

#### BUFFER_ATOMIC_ADD_F32  (opcode 77)

Add a single-precision float value in the data register to a location in a buffer surface. Store the original value
from buffer surface into a vector register iff the SC0 bit is set.

```
  tmp = MEM[ADDR].f32;
  MEM[ADDR].f32 += DATA.f32;
  RETURN_DATA = tmp
```

Notes

Floating-point addition handles NAN/INF/denorm.

#### BUFFER_ATOMIC_PK_ADD_F16  (opcode 78)

Add a packed 2-component half-precision float value in the data register to a location in a buffer surface. Store
the original value from buffer surface into a vector register iff the SC0 bit is set.

```
  tmp = MEM[ADDR];
  src = DATA;
  dst[31 : 16].f16 = tmp[31 : 16].f16 + src[31 : 16].f16;
  dst[15 : 0].f16 = tmp[15 : 0].f16 + src[15 : 0].f16;
  MEM[ADDR] = dst.b32;
  RETURN_DATA = tmp
```

Notes

Floating-point addition handles NAN/INF/denorm.

#### BUFFER_ATOMIC_ADD_F64  (opcode 79)

Add a double-precision float value in the data register to a location in a buffer surface. Store the original value
from buffer surface into a vector register iff the SC0 bit is set.

```
  tmp = MEM[ADDR].f64;
  MEM[ADDR].f64 += DATA.f64;
  RETURN_DATA = tmp
```

Notes

Floating-point addition handles NAN/INF/denorm.

#### BUFFER_ATOMIC_MIN_F64  (opcode 80)

Select the minimum of two double-precision float inputs, given two values stored in the data register and a
location in a buffer surface. Update the buffer surface with the selected value. Store the original value from
buffer surface into a vector register iff the SC0 bit is set.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  tmp = MEM[addr].f64;
  src = DATA.f64;
  MEM[addr].f64 = src < tmp ? src : tmp;
```

```
  RETURN_DATA.f64 = tmp
```

#### BUFFER_ATOMIC_MAX_F64  (opcode 81)

Select the maximum of two double-precision float inputs, given two values stored in the data register and a
location in a buffer surface. Update the buffer surface with the selected value. Store the original value from
buffer surface into a vector register iff the SC0 bit is set.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  tmp = MEM[addr].f64;
  src = DATA.f64;
  MEM[addr].f64 = src > tmp ? src : tmp;
  RETURN_DATA.f64 = tmp
```

#### BUFFER_ATOMIC_PK_ADD_BF16  (opcode 82)

Add a packed 2-component BF16 float value in the data register to a location in a buffer surface. Store the
original value from buffer surface into a vector register iff the SC0 bit is set.

```
  tmp = MEM[ADDR];
  src = DATA;
  dst[31 : 16].bf16 = tmp[31 : 16].bf16 + src[31 : 16].bf16;
  dst[15 : 0].bf16 = tmp[15 : 0].bf16 + src[15 : 0].bf16;
  MEM[ADDR] = dst.b32;
  RETURN_DATA = tmp
```

Notes

Floating-point addition handles NAN/INF/denorm.

#### BUFFER_ATOMIC_SWAP_X2  (opcode 96)

Swap an unsigned 64-bit integer value in the data register with a location in a buffer surface. Store the original
value from buffer surface into a vector register iff the SC0 bit is set.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  tmp = MEM[addr].b64;
  MEM[addr].b64 = DATA.b64;
  RETURN_DATA.b64 = tmp
```

#### BUFFER_ATOMIC_CMPSWAP_X2  (opcode 97)

Compare two unsigned 64-bit integer values stored in the data comparison register and a location in a buffer
surface. Modify the memory location with a value in the data source register iff the comparison is equal. Store
the original value from buffer surface into a vector register iff the SC0 bit is set.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  tmp = MEM[addr].u64;
  src = DATA[63 : 0].u64;
  cmp = DATA[127 : 64].u64;
  MEM[addr].u64 = tmp == cmp ? src : tmp;
  RETURN_DATA.u64 = tmp
```

#### BUFFER_ATOMIC_ADD_X2  (opcode 98)

Add two unsigned 64-bit integer values stored in the data register and a location in a buffer surface. Store the
original value from buffer surface into a vector register iff the SC0 bit is set.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  tmp = MEM[addr].u64;
  MEM[addr].u64 += DATA.u64;
  RETURN_DATA.u64 = tmp
```

#### BUFFER_ATOMIC_SUB_X2  (opcode 99)

Subtract an unsigned 64-bit integer value stored in the data register from a value stored in a location in a buffer
surface. Store the original value from buffer surface into a vector register iff the SC0 bit is set.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  tmp = MEM[addr].u64;
  MEM[addr].u64 -= DATA.u64;
  RETURN_DATA.u64 = tmp
```

#### BUFFER_ATOMIC_SMIN_X2  (opcode 100)

Select the minimum of two signed 64-bit integer inputs, given two values stored in the data register and a
location in a buffer surface. Update the buffer surface with the selected value. Store the original value from
buffer surface into a vector register iff the SC0 bit is set.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  tmp = MEM[addr].i64;
  src = DATA.i64;
  MEM[addr].i64 = src < tmp ? src : tmp;
  RETURN_DATA.i64 = tmp
```

#### BUFFER_ATOMIC_UMIN_X2  (opcode 101)

Select the minimum of two unsigned 64-bit integer inputs, given two values stored in the data register and a
location in a buffer surface. Update the buffer surface with the selected value. Store the original value from
buffer surface into a vector register iff the SC0 bit is set.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  tmp = MEM[addr].u64;
  src = DATA.u64;
  MEM[addr].u64 = src < tmp ? src : tmp;
  RETURN_DATA.u64 = tmp
```

#### BUFFER_ATOMIC_SMAX_X2  (opcode 102)

Select the maximum of two signed 64-bit integer inputs, given two values stored in the data register and a
location in a buffer surface. Update the buffer surface with the selected value. Store the original value from
buffer surface into a vector register iff the SC0 bit is set.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  tmp = MEM[addr].i64;
  src = DATA.i64;
  MEM[addr].i64 = src >= tmp ? src : tmp;
  RETURN_DATA.i64 = tmp
```

#### BUFFER_ATOMIC_UMAX_X2  (opcode 103)

Select the maximum of two unsigned 64-bit integer inputs, given two values stored in the data register and a
location in a buffer surface. Update the buffer surface with the selected value. Store the original value from
buffer surface into a vector register iff the SC0 bit is set.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  tmp = MEM[addr].u64;
  src = DATA.u64;
  MEM[addr].u64 = src >= tmp ? src : tmp;
  RETURN_DATA.u64 = tmp
```

#### BUFFER_ATOMIC_AND_X2  (opcode 104)

Calculate bitwise AND given two unsigned 64-bit integer values stored in the data register and a location in a
buffer surface. Store the original value from buffer surface into a vector register iff the SC0 bit is set.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  tmp = MEM[addr].b64;
  MEM[addr].b64 = (tmp & DATA.b64);
  RETURN_DATA.b64 = tmp
```

#### BUFFER_ATOMIC_OR_X2  (opcode 105)

Calculate bitwise OR given two unsigned 64-bit integer values stored in the data register and a location in a
buffer surface. Store the original value from buffer surface into a vector register iff the SC0 bit is set.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  tmp = MEM[addr].b64;
  MEM[addr].b64 = (tmp | DATA.b64);
  RETURN_DATA.b64 = tmp
```

#### BUFFER_ATOMIC_XOR_X2  (opcode 106)

Calculate bitwise XOR given two unsigned 64-bit integer values stored in the data register and a location in a
buffer surface. Store the original value from buffer surface into a vector register iff the SC0 bit is set.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  tmp = MEM[addr].b64;
  MEM[addr].b64 = (tmp ^ DATA.b64);
  RETURN_DATA.b64 = tmp
```

#### BUFFER_ATOMIC_INC_X2  (opcode 107)

Increment an unsigned 64-bit integer value from a location in a buffer surface with wraparound to 0 if the
value exceeds a value in the data register. Store the original value from buffer surface into a vector register iff
the SC0 bit is set.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  tmp = MEM[addr].u64;
  src = DATA.u64;
  MEM[addr].u64 = tmp >= src ? 0ULL : tmp + 1ULL;
  RETURN_DATA.u64 = tmp
```

#### BUFFER_ATOMIC_DEC_X2  (opcode 108)

Decrement an unsigned 64-bit integer value from a location in a buffer surface with wraparound to a value in

the data register if the decrement yields a negative value. Store the original value from buffer surface into a
vector register iff the SC0 bit is set.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  tmp = MEM[addr].u64;
  src = DATA.u64;
  MEM[addr].u64 = ((tmp == 0ULL) || (tmp > src)) ? src : tmp - 1ULL;
  RETURN_DATA.u64 = tmp
```

### 12.14. MTBUF Instructions

The bitfield map of the MTBUF format is:

```
      where:
```

```
      OFFSET   = Unsigned immediate byte offset.
      OFFEN    = Send offset either as VADDR or as zero.
      IDXEN    = Send index either as VADDR or as zero.
      LDS      = Data is transferred between LDS and Memory, not VGPRs.
      OP       = Instruction Opcode.
      DFMT     = Data format for typed buffer.
      NFMT     = Number format for typed buffer.
      VADDR    = VGPR address source.
      VDATA    = Vector GPR for read/write result.
      SRSRC    = Scalar GPR that specifies resource constant.
      SOFFSET = Unsigned byte offset from an SGPR.
      SC       = Scope
      NT       = Non-Temporal
      ACC      = Return to ACC VGPRs
```

#### TBUFFER_LOAD_FORMAT_X  (opcode 0)

Load 1-component formatted data from a buffer surface, convert the data to 32 bit integral or floating point
format, then store the result into a vector register. The instruction specifies the data format of the surface,
overriding the resource descriptor.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  VDATA[31 : 0].b32 = ConvertFromFormat(MEM[addr + ChannelOffsetX()]);
  // Mem access size depends on format
```

#### TBUFFER_LOAD_FORMAT_XY  (opcode 1)

Load 2-component formatted data from a buffer surface, convert the data to 32 bit integral or floating point
format, then store the result into a vector register. The instruction specifies the data format of the surface,
overriding the resource descriptor.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  VDATA[31 : 0].b32 = ConvertFromFormat(MEM[addr + ChannelOffsetX()]);
  // Mem access size depends on format
  VDATA[63 : 32].b32 = ConvertFromFormat(MEM[addr + ChannelOffsetY()])
```

#### TBUFFER_LOAD_FORMAT_XYZ  (opcode 2)

Load 3-component formatted data from a buffer surface, convert the data to 32 bit integral or floating point
format, then store the result into a vector register. The instruction specifies the data format of the surface,
overriding the resource descriptor.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  VDATA[31 : 0].b32 = ConvertFromFormat(MEM[addr + ChannelOffsetX()]);
  // Mem access size depends on format
  VDATA[63 : 32].b32 = ConvertFromFormat(MEM[addr + ChannelOffsetY()]);
  VDATA[95 : 64].b32 = ConvertFromFormat(MEM[addr + ChannelOffsetZ()])
```

#### TBUFFER_LOAD_FORMAT_XYZW  (opcode 3)

Load 4-component formatted data from a buffer surface, convert the data to 32 bit integral or floating point
format, then store the result into a vector register. The instruction specifies the data format of the surface,
overriding the resource descriptor.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  VDATA[31 : 0].b32 = ConvertFromFormat(MEM[addr + ChannelOffsetX()]);
  // Mem access size depends on format
  VDATA[63 : 32].b32 = ConvertFromFormat(MEM[addr + ChannelOffsetY()]);
  VDATA[95 : 64].b32 = ConvertFromFormat(MEM[addr + ChannelOffsetZ()]);
  VDATA[127 : 96].b32 = ConvertFromFormat(MEM[addr + ChannelOffsetW()])
```

#### TBUFFER_STORE_FORMAT_X  (opcode 4)

Convert 32 bits of data from vector input registers into 1-component formatted data and store the data into a
buffer surface. The instruction specifies the data format of the surface, overriding the resource descriptor.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  MEM[addr + ChannelOffsetX()] = ConvertToFormat(VDATA[31 : 0].b32);
  // Mem access size depends on format
```

#### TBUFFER_STORE_FORMAT_XY  (opcode 5)

Convert 64 bits of data from vector input registers into 2-component formatted data and store the data into a
buffer surface. The instruction specifies the data format of the surface, overriding the resource descriptor.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  MEM[addr + ChannelOffsetX()] = ConvertToFormat(VDATA[31 : 0].b32);
  // Mem access size depends on format
```

```
  MEM[addr + ChannelOffsetY()] = ConvertToFormat(VDATA[63 : 32].b32)
```

#### TBUFFER_STORE_FORMAT_XYZ  (opcode 6)

Convert 96 bits of data from vector input registers into 3-component formatted data and store the data into a
buffer surface. The instruction specifies the data format of the surface, overriding the resource descriptor.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  MEM[addr + ChannelOffsetX()] = ConvertToFormat(VDATA[31 : 0].b32);
  // Mem access size depends on format
  MEM[addr + ChannelOffsetY()] = ConvertToFormat(VDATA[63 : 32].b32);
  MEM[addr + ChannelOffsetZ()] = ConvertToFormat(VDATA[95 : 64].b32)
```

#### TBUFFER_STORE_FORMAT_XYZW  (opcode 7)

Convert 128 bits of data from vector input registers into 4-component formatted data and store the data into a
buffer surface. The instruction specifies the data format of the surface, overriding the resource descriptor.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  MEM[addr + ChannelOffsetX()] = ConvertToFormat(VDATA[31 : 0].b32);
  // Mem access size depends on format
  MEM[addr + ChannelOffsetY()] = ConvertToFormat(VDATA[63 : 32].b32);
  MEM[addr + ChannelOffsetZ()] = ConvertToFormat(VDATA[95 : 64].b32);
  MEM[addr + ChannelOffsetW()] = ConvertToFormat(VDATA[127 : 96].b32)
```

#### TBUFFER_LOAD_FORMAT_D16_X  (opcode 8)

Load 1-component formatted data from a buffer surface, convert the data to packed 16 bit integral or floating
point format, then store the result into a vector register. The instruction specifies the data format of the
surface, overriding the resource descriptor.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  VDATA[15 : 0].b16 = 16'B(ConvertFromFormat(MEM[addr + ChannelOffsetX()]));
  // Mem access size depends on format
  // VDATA[31:16].b16 is preserved.
```

#### TBUFFER_LOAD_FORMAT_D16_XY  (opcode 9)

Load 2-component formatted data from a buffer surface, convert the data to packed 16 bit integral or floating
point format, then store the result into a vector register. The instruction specifies the data format of the
surface, overriding the resource descriptor.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  VDATA[15 : 0].b16 = 16'B(ConvertFromFormat(MEM[addr + ChannelOffsetX()]));
  // Mem access size depends on format
  VDATA[31 : 16].b16 = 16'B(ConvertFromFormat(MEM[addr + ChannelOffsetY()]))
```

#### TBUFFER_LOAD_FORMAT_D16_XYZ  (opcode 10)

Load 3-component formatted data from a buffer surface, convert the data to packed 16 bit integral or floating
point format, then store the result into a vector register. The instruction specifies the data format of the
surface, overriding the resource descriptor.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  VDATA[15 : 0].b16 = 16'B(ConvertFromFormat(MEM[addr + ChannelOffsetX()]));
  // Mem access size depends on format
  VDATA[31 : 16].b16 = 16'B(ConvertFromFormat(MEM[addr + ChannelOffsetY()]));
  VDATA[47 : 32].b16 = 16'B(ConvertFromFormat(MEM[addr + ChannelOffsetZ()]));
  // VDATA[63:48].b16 is preserved.
```

#### TBUFFER_LOAD_FORMAT_D16_XYZW  (opcode 11)

Load 4-component formatted data from a buffer surface, convert the data to packed 16 bit integral or floating
point format, then store the result into a vector register. The instruction specifies the data format of the
surface, overriding the resource descriptor.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  VDATA[15 : 0].b16 = 16'B(ConvertFromFormat(MEM[addr + ChannelOffsetX()]));
  // Mem access size depends on format
  VDATA[31 : 16].b16 = 16'B(ConvertFromFormat(MEM[addr + ChannelOffsetY()]));
  VDATA[47 : 32].b16 = 16'B(ConvertFromFormat(MEM[addr + ChannelOffsetZ()]));
  VDATA[63 : 48].b16 = 16'B(ConvertFromFormat(MEM[addr + ChannelOffsetW()]))
```

#### TBUFFER_STORE_FORMAT_D16_X  (opcode 12)

Convert 16 bits of data from vector input registers into 1-component formatted data and store the data into a
buffer surface. The instruction specifies the data format of the surface, overriding the resource descriptor.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  MEM[addr + ChannelOffsetX()] = ConvertToFormat(32'B(VDATA[15 : 0].b16));
  // Mem access size depends on format
```

#### TBUFFER_STORE_FORMAT_D16_XY  (opcode 13)

Convert 32 bits of data from vector input registers into 2-component formatted data and store the data into a
buffer surface. The instruction specifies the data format of the surface, overriding the resource descriptor.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  MEM[addr + ChannelOffsetX()] = ConvertToFormat(32'B(VDATA[15 : 0].b16));
  // Mem access size depends on format
  MEM[addr + ChannelOffsetY()] = ConvertToFormat(32'B(VDATA[31 : 16].b16))
```

#### TBUFFER_STORE_FORMAT_D16_XYZ  (opcode 14)

Convert 48 bits of data from vector input registers into 3-component formatted data and store the data into a
buffer surface. The instruction specifies the data format of the surface, overriding the resource descriptor.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  MEM[addr + ChannelOffsetX()] = ConvertToFormat(32'B(VDATA[15 : 0].b16));
  // Mem access size depends on format
  MEM[addr + ChannelOffsetY()] = ConvertToFormat(32'B(VDATA[31 : 16].b16));
  MEM[addr + ChannelOffsetZ()] = ConvertToFormat(32'B(VDATA[47 : 32].b16))
```

#### TBUFFER_STORE_FORMAT_D16_XYZW  (opcode 15)

Convert 64 bits of data from vector input registers into 4-component formatted data and store the data into a
buffer surface. The instruction specifies the data format of the surface, overriding the resource descriptor.

```
  addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
  MEM[addr + ChannelOffsetX()] = ConvertToFormat(32'B(VDATA[15 : 0].b16));
  // Mem access size depends on format
  MEM[addr + ChannelOffsetY()] = ConvertToFormat(32'B(VDATA[31 : 16].b16));
  MEM[addr + ChannelOffsetZ()] = ConvertToFormat(32'B(VDATA[47 : 32].b16));
  MEM[addr + ChannelOffsetW()] = ConvertToFormat(32'B(VDATA[63 : 48].b16))
```

### 12.15. FLAT, Scratch and Global Instructions

The bitfield map of the FLAT format is:

```
      where:
```

```
      OP       = Instruction Opcode.
      ADDR     = Source of flat address VGPR.
      DATA     = Source data.
      VDST     = Destination VGPR.
      NV       = Access to non-volatile memory.
      SADDR    = SGPR holding address or offset
      SEG      = Instruction type: Flat, Scratch, or Global
      LDS      = Data is transferred between LDS and Memory, not VGPRs.
      OFFSET = Immediate address byte-offset.
      SC        = Scope
      NT        = Non-Temporal
```

#### 12.15.1. Flat Instructions

Flat instructions look at the per-workitem address and determine for each work item if the target memory
address is in global, private or scratch memory.

#### FLAT_LOAD_UBYTE  (opcode 16)

Load 8 bits of unsigned data from the flat aperture, zero extend to 32 bits and store the result into a vector
register.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  VDATA.u32 = 32'U({ 24'0U, MEM[addr].u8 })
```

#### FLAT_LOAD_SBYTE  (opcode 17)

Load 8 bits of signed data from the flat aperture, sign extend to 32 bits and store the result into a vector
register.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  VDATA.i32 = 32'I(signext(MEM[addr].i8))
```

#### FLAT_LOAD_USHORT  (opcode 18)

Load 16 bits of unsigned data from the flat aperture, zero extend to 32 bits and store the result into a vector
register.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  VDATA.u32 = 32'U({ 16'0U, MEM[addr].u16 })
```

#### FLAT_LOAD_SSHORT  (opcode 19)

Load 16 bits of signed data from the flat aperture, sign extend to 32 bits and store the result into a vector
register.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  VDATA.i32 = 32'I(signext(MEM[addr].i16))
```

#### FLAT_LOAD_DWORD  (opcode 20)

Load 32 bits of data from the flat aperture into a vector register.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  VDATA[31 : 0] = MEM[addr].b32
```

#### FLAT_LOAD_DWORDX2  (opcode 21)

Load 64 bits of data from the flat aperture into a vector register.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  VDATA[31 : 0] = MEM[addr].b32;
  VDATA[63 : 32] = MEM[addr + 4U].b32
```

#### FLAT_LOAD_DWORDX3  (opcode 22)

Load 96 bits of data from the flat aperture into a vector register.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  VDATA[31 : 0] = MEM[addr].b32;
  VDATA[63 : 32] = MEM[addr + 4U].b32;
```

```
  VDATA[95 : 64] = MEM[addr + 8U].b32
```

#### FLAT_LOAD_DWORDX4  (opcode 23)

Load 128 bits of data from the flat aperture into a vector register.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  VDATA[31 : 0] = MEM[addr].b32;
  VDATA[63 : 32] = MEM[addr + 4U].b32;
  VDATA[95 : 64] = MEM[addr + 8U].b32;
  VDATA[127 : 96] = MEM[addr + 12U].b32
```

#### FLAT_STORE_BYTE  (opcode 24)

Store 8 bits of data from a vector register into the flat aperture.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  MEM[addr].b8 = VDATA[7 : 0]
```

#### FLAT_STORE_BYTE_D16_HI  (opcode 25)

Store 8 bits of data from the high 16 bits of a 32-bit vector register into the flat aperture.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  MEM[addr].b8 = VDATA[23 : 16]
```

#### FLAT_STORE_SHORT  (opcode 26)

Store 16 bits of data from a vector register into the flat aperture.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  MEM[addr].b16 = VDATA[15 : 0]
```

#### FLAT_STORE_SHORT_D16_HI  (opcode 27)

Store 16 bits of data from the high 16 bits of a 32-bit vector register into the flat aperture.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  MEM[addr].b16 = VDATA[31 : 16]
```

#### FLAT_STORE_DWORD  (opcode 28)

Store 32 bits of data from vector input registers into the flat aperture.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  MEM[addr].b32 = VDATA[31 : 0]
```

#### FLAT_STORE_DWORDX2  (opcode 29)

Store 64 bits of data from vector input registers into the flat aperture.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  MEM[addr].b32 = VDATA[31 : 0];
  MEM[addr + 4U].b32 = VDATA[63 : 32]
```

#### FLAT_STORE_DWORDX3  (opcode 30)

Store 96 bits of data from vector input registers into the flat aperture.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  MEM[addr].b32 = VDATA[31 : 0];
  MEM[addr + 4U].b32 = VDATA[63 : 32];
  MEM[addr + 8U].b32 = VDATA[95 : 64]
```

#### FLAT_STORE_DWORDX4  (opcode 31)

Store 128 bits of data from vector input registers into the flat aperture.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  MEM[addr].b32 = VDATA[31 : 0];
  MEM[addr + 4U].b32 = VDATA[63 : 32];
  MEM[addr + 8U].b32 = VDATA[95 : 64];
  MEM[addr + 12U].b32 = VDATA[127 : 96]
```

#### FLAT_LOAD_UBYTE_D16  (opcode 32)

Load 8 bits of unsigned data from the flat aperture, zero extend to 16 bits and store the result into the low 16
bits of a 32-bit vector register.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  VDATA[15 : 0].u16 = 16'U({ 8'0U, MEM[addr].u8 });
  // VDATA[31:16] is preserved.
```

#### FLAT_LOAD_UBYTE_D16_HI  (opcode 33)

Load 8 bits of unsigned data from the flat aperture, zero extend to 16 bits and store the result into the high 16
bits of a 32-bit vector register.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  VDATA[31 : 16].u16 = 16'U({ 8'0U, MEM[addr].u8 });
  // VDATA[15:0] is preserved.
```

#### FLAT_LOAD_SBYTE_D16  (opcode 34)

Load 8 bits of signed data from the flat aperture, sign extend to 16 bits and store the result into the low 16 bits
of a 32-bit vector register.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  VDATA[15 : 0].i16 = 16'I(signext(MEM[addr].i8));
  // VDATA[31:16] is preserved.
```

#### FLAT_LOAD_SBYTE_D16_HI  (opcode 35)

Load 8 bits of signed data from the flat aperture, sign extend to 16 bits and store the result into the high 16 bits
of a 32-bit vector register.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  VDATA[31 : 16].i16 = 16'I(signext(MEM[addr].i8));
  // VDATA[15:0] is preserved.
```

#### FLAT_LOAD_SHORT_D16  (opcode 36)

Load 16 bits of unsigned data from the flat aperture and store the result into the low 16 bits of a 32-bit vector

register.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  VDATA[15 : 0].b16 = MEM[addr].b16;
  // VDATA[31:16] is preserved.
```

#### FLAT_LOAD_SHORT_D16_HI  (opcode 37)

Load 16 bits of unsigned data from the flat aperture and store the result into the high 16 bits of a 32-bit vector
register.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  VDATA[31 : 16].b16 = MEM[addr].b16;
  // VDATA[15:0] is preserved.
```

#### FLAT_ATOMIC_SWAP  (opcode 64)

Swap an unsigned 32-bit integer value in the data register with a location in the flat aperture. Store the original
value from flat aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = DATA.b32;
  RETURN_DATA.b32 = tmp
```

#### FLAT_ATOMIC_CMPSWAP  (opcode 65)

Compare two unsigned 32-bit integer values stored in the data comparison register and a location in the flat
aperture. Modify the memory location with a value in the data source register iff the comparison is equal. Store
the original value from flat aperture into a vector register iff the SC0 bit is set.

NOTE: RETURN_DATA[1] is not modified.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  tmp = MEM[addr].u32;
  src = DATA[31 : 0].u32;
  cmp = DATA[63 : 32].u32;
  MEM[addr].u32 = tmp == cmp ? src : tmp;
  RETURN_DATA.u32 = tmp
```

#### FLAT_ATOMIC_ADD  (opcode 66)

Add two unsigned 32-bit integer values stored in the data register and a location in the flat aperture. Store the
original value from flat aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  tmp = MEM[addr].u32;
  MEM[addr].u32 += DATA.u32;
  RETURN_DATA.u32 = tmp
```

#### FLAT_ATOMIC_SUB  (opcode 67)

Subtract an unsigned 32-bit integer value stored in the data register from a value stored in a location in the flat
aperture. Store the original value from flat aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  tmp = MEM[addr].u32;
  MEM[addr].u32 -= DATA.u32;
  RETURN_DATA.u32 = tmp
```

#### FLAT_ATOMIC_SMIN  (opcode 68)

Select the minimum of two signed 32-bit integer inputs, given two values stored in the data register and a
location in the flat aperture. Update the flat aperture with the selected value. Store the original value from flat
aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  tmp = MEM[addr].i32;
  src = DATA.i32;
  MEM[addr].i32 = src < tmp ? src : tmp;
  RETURN_DATA.i32 = tmp
```

#### FLAT_ATOMIC_UMIN  (opcode 69)

Select the minimum of two unsigned 32-bit integer inputs, given two values stored in the data register and a
location in the flat aperture. Update the flat aperture with the selected value. Store the original value from flat
aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = src < tmp ? src : tmp;
```

```
  RETURN_DATA.u32 = tmp
```

#### FLAT_ATOMIC_SMAX  (opcode 70)

Select the maximum of two signed 32-bit integer inputs, given two values stored in the data register and a
location in the flat aperture. Update the flat aperture with the selected value. Store the original value from flat
aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  tmp = MEM[addr].i32;
  src = DATA.i32;
  MEM[addr].i32 = src >= tmp ? src : tmp;
  RETURN_DATA.i32 = tmp
```

#### FLAT_ATOMIC_UMAX  (opcode 71)

Select the maximum of two unsigned 32-bit integer inputs, given two values stored in the data register and a
location in the flat aperture. Update the flat aperture with the selected value. Store the original value from flat
aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = src >= tmp ? src : tmp;
  RETURN_DATA.u32 = tmp
```

#### FLAT_ATOMIC_AND  (opcode 72)

Calculate bitwise AND given two unsigned 32-bit integer values stored in the data register and a location in the
flat aperture. Store the original value from flat aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = (tmp & DATA.b32);
  RETURN_DATA.b32 = tmp
```

#### FLAT_ATOMIC_OR  (opcode 73)

Calculate bitwise OR given two unsigned 32-bit integer values stored in the data register and a location in the
flat aperture. Store the original value from flat aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = (tmp | DATA.b32);
  RETURN_DATA.b32 = tmp
```

#### FLAT_ATOMIC_XOR  (opcode 74)

Calculate bitwise XOR given two unsigned 32-bit integer values stored in the data register and a location in the
flat aperture. Store the original value from flat aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = (tmp ^ DATA.b32);
  RETURN_DATA.b32 = tmp
```

#### FLAT_ATOMIC_INC  (opcode 75)

Increment an unsigned 32-bit integer value from a location in the flat aperture with wraparound to 0 if the
value exceeds a value in the data register. Store the original value from flat aperture into a vector register iff
the SC0 bit is set.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = tmp >= src ? 0U : tmp + 1U;
  RETURN_DATA.u32 = tmp
```

#### FLAT_ATOMIC_DEC  (opcode 76)

Decrement an unsigned 32-bit integer value from a location in the flat aperture with wraparound to a value in
the data register if the decrement yields a negative value. Store the original value from flat aperture into a
vector register iff the SC0 bit is set.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = ((tmp == 0U) || (tmp > src)) ? src : tmp - 1U;
  RETURN_DATA.u32 = tmp
```

#### FLAT_ATOMIC_ADD_F32  (opcode 77)

Add a single-precision float value in the data register to a location in the flat aperture. Store the original value
from flat aperture into a vector register iff the SC0 bit is set.

```
  tmp = MEM[ADDR].f32;
  MEM[ADDR].f32 += DATA.f32;
  RETURN_DATA = tmp
```

Notes

Floating-point addition handles NAN/INF/denorm.

#### FLAT_ATOMIC_PK_ADD_F16  (opcode 78)

Add a packed 2-component half-precision float value in the data register to a location in the flat aperture. Store
the original value from flat aperture into a vector register iff the SC0 bit is set.

```
  tmp = MEM[ADDR];
  src = DATA;
  dst[31 : 16].f16 = tmp[31 : 16].f16 + src[31 : 16].f16;
  dst[15 : 0].f16 = tmp[15 : 0].f16 + src[15 : 0].f16;
  MEM[ADDR] = dst.b32;
  RETURN_DATA = tmp
```

Notes

Floating-point addition handles NAN/INF/denorm.

#### FLAT_ATOMIC_ADD_F64  (opcode 79)

Add a double-precision float value in the data register to a location in the flat aperture. Store the original value
from flat aperture into a vector register iff the SC0 bit is set.

```
  tmp = MEM[ADDR].f64;
  MEM[ADDR].f64 += DATA.f64;
  RETURN_DATA = tmp
```

Notes

Floating-point addition handles NAN/INF/denorm.

#### FLAT_ATOMIC_MIN_F64  (opcode 80)

Select the minimum of two double-precision float inputs, given two values stored in the data register and a

location in the flat aperture. Update the flat aperture with the selected value. Store the original value from flat
aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  tmp = MEM[addr].f64;
  src = DATA.f64;
  MEM[addr].f64 = src < tmp ? src : tmp;
  RETURN_DATA.f64 = tmp
```

#### FLAT_ATOMIC_MAX_F64  (opcode 81)

Select the maximum of two double-precision float inputs, given two values stored in the data register and a
location in the flat aperture. Update the flat aperture with the selected value. Store the original value from flat
aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  tmp = MEM[addr].f64;
  src = DATA.f64;
  MEM[addr].f64 = src > tmp ? src : tmp;
  RETURN_DATA.f64 = tmp
```

#### FLAT_ATOMIC_PK_ADD_BF16  (opcode 82)

Add a packed 2-component BF16 float value in the data register to a location in the flat aperture. Store the
original value from flat aperture into a vector register iff the SC0 bit is set.

```
  tmp = MEM[ADDR];
  src = DATA;
  dst[31 : 16].bf16 = tmp[31 : 16].bf16 + src[31 : 16].bf16;
  dst[15 : 0].bf16 = tmp[15 : 0].bf16 + src[15 : 0].bf16;
  MEM[ADDR] = dst.b32;
  RETURN_DATA = tmp
```

Notes

Floating-point addition handles NAN/INF/denorm.

#### FLAT_ATOMIC_SWAP_X2  (opcode 96)

Swap an unsigned 64-bit integer value in the data register with a location in the flat aperture. Store the original
value from flat aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
```

```
  tmp = MEM[addr].b64;
  MEM[addr].b64 = DATA.b64;
  RETURN_DATA.b64 = tmp
```

#### FLAT_ATOMIC_CMPSWAP_X2  (opcode 97)

Compare two unsigned 64-bit integer values stored in the data comparison register and a location in the flat
aperture. Modify the memory location with a value in the data source register iff the comparison is equal. Store
the original value from flat aperture into a vector register iff the SC0 bit is set.

NOTE: RETURN_DATA[2:3] is not modified.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  tmp = MEM[addr].u64;
  src = DATA[63 : 0].u64;
  cmp = DATA[127 : 64].u64;
  MEM[addr].u64 = tmp == cmp ? src : tmp;
  RETURN_DATA.u64 = tmp
```

#### FLAT_ATOMIC_ADD_X2  (opcode 98)

Add two unsigned 64-bit integer values stored in the data register and a location in the flat aperture. Store the
original value from flat aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  tmp = MEM[addr].u64;
  MEM[addr].u64 += DATA.u64;
  RETURN_DATA.u64 = tmp
```

#### FLAT_ATOMIC_SUB_X2  (opcode 99)

Subtract an unsigned 64-bit integer value stored in the data register from a value stored in a location in the flat
aperture. Store the original value from flat aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  tmp = MEM[addr].u64;
  MEM[addr].u64 -= DATA.u64;
  RETURN_DATA.u64 = tmp
```

#### FLAT_ATOMIC_SMIN_X2  (opcode 100)

Select the minimum of two signed 64-bit integer inputs, given two values stored in the data register and a
location in the flat aperture. Update the flat aperture with the selected value. Store the original value from flat
aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  tmp = MEM[addr].i64;
  src = DATA.i64;
  MEM[addr].i64 = src < tmp ? src : tmp;
  RETURN_DATA.i64 = tmp
```

#### FLAT_ATOMIC_UMIN_X2  (opcode 101)

Select the minimum of two unsigned 64-bit integer inputs, given two values stored in the data register and a
location in the flat aperture. Update the flat aperture with the selected value. Store the original value from flat
aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  tmp = MEM[addr].u64;
  src = DATA.u64;
  MEM[addr].u64 = src < tmp ? src : tmp;
  RETURN_DATA.u64 = tmp
```

#### FLAT_ATOMIC_SMAX_X2  (opcode 102)

Select the maximum of two signed 64-bit integer inputs, given two values stored in the data register and a
location in the flat aperture. Update the flat aperture with the selected value. Store the original value from flat
aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  tmp = MEM[addr].i64;
  src = DATA.i64;
  MEM[addr].i64 = src >= tmp ? src : tmp;
  RETURN_DATA.i64 = tmp
```

#### FLAT_ATOMIC_UMAX_X2  (opcode 103)

Select the maximum of two unsigned 64-bit integer inputs, given two values stored in the data register and a
location in the flat aperture. Update the flat aperture with the selected value. Store the original value from flat
aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  tmp = MEM[addr].u64;
  src = DATA.u64;
```

```
  MEM[addr].u64 = src >= tmp ? src : tmp;
  RETURN_DATA.u64 = tmp
```

#### FLAT_ATOMIC_AND_X2  (opcode 104)

Calculate bitwise AND given two unsigned 64-bit integer values stored in the data register and a location in the
flat aperture. Store the original value from flat aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  tmp = MEM[addr].b64;
  MEM[addr].b64 = (tmp & DATA.b64);
  RETURN_DATA.b64 = tmp
```

#### FLAT_ATOMIC_OR_X2  (opcode 105)

Calculate bitwise OR given two unsigned 64-bit integer values stored in the data register and a location in the
flat aperture. Store the original value from flat aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  tmp = MEM[addr].b64;
  MEM[addr].b64 = (tmp | DATA.b64);
  RETURN_DATA.b64 = tmp
```

#### FLAT_ATOMIC_XOR_X2  (opcode 106)

Calculate bitwise XOR given two unsigned 64-bit integer values stored in the data register and a location in the
flat aperture. Store the original value from flat aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  tmp = MEM[addr].b64;
  MEM[addr].b64 = (tmp ^ DATA.b64);
  RETURN_DATA.b64 = tmp
```

#### FLAT_ATOMIC_INC_X2  (opcode 107)

Increment an unsigned 64-bit integer value from a location in the flat aperture with wraparound to 0 if the
value exceeds a value in the data register. Store the original value from flat aperture into a vector register iff
the SC0 bit is set.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
```

```
  tmp = MEM[addr].u64;
  src = DATA.u64;
  MEM[addr].u64 = tmp >= src ? 0ULL : tmp + 1ULL;
  RETURN_DATA.u64 = tmp
```

#### FLAT_ATOMIC_DEC_X2  (opcode 108)

Decrement an unsigned 64-bit integer value from a location in the flat aperture with wraparound to a value in
the data register if the decrement yields a negative value. Store the original value from flat aperture into a
vector register iff the SC0 bit is set.

```
  addr = CalcFlatAddr(ADDR.b32, OFFSET.b32);
  tmp = MEM[addr].u64;
  src = DATA.u64;
  MEM[addr].u64 = ((tmp == 0ULL) || (tmp > src)) ? src : tmp - 1ULL;
  RETURN_DATA.u64 = tmp
```

#### 12.15.2. Scratch Instructions

Scratch instructions are like Flat, but assume all workitem addresses fall in scratch (private) space.

#### SCRATCH_LOAD_UBYTE  (opcode 16)

Load 8 bits of unsigned data from the scratch aperture, zero extend to 32 bits and store the result into a vector
register.

```
  addr = CalcScratchAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  VDATA.u32 = 32'U({ 24'0U, MEM[addr].u8 })
```

#### SCRATCH_LOAD_SBYTE  (opcode 17)

Load 8 bits of signed data from the scratch aperture, sign extend to 32 bits and store the result into a vector
register.

```
  addr = CalcScratchAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  VDATA.i32 = 32'I(signext(MEM[addr].i8))
```

#### SCRATCH_LOAD_USHORT  (opcode 18)

Load 16 bits of unsigned data from the scratch aperture, zero extend to 32 bits and store the result into a vector
register.

```
  addr = CalcScratchAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  VDATA.u32 = 32'U({ 16'0U, MEM[addr].u16 })
```

#### SCRATCH_LOAD_SSHORT  (opcode 19)

Load 16 bits of signed data from the scratch aperture, sign extend to 32 bits and store the result into a vector
register.

```
  addr = CalcScratchAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  VDATA.i32 = 32'I(signext(MEM[addr].i16))
```

#### SCRATCH_LOAD_DWORD  (opcode 20)

Load 32 bits of data from the scratch aperture into a vector register.

```
  addr = CalcScratchAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  VDATA[31 : 0] = MEM[addr].b32
```

#### SCRATCH_LOAD_DWORDX2  (opcode 21)

Load 64 bits of data from the scratch aperture into a vector register.

```
  addr = CalcScratchAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  VDATA[31 : 0] = MEM[addr].b32;
  VDATA[63 : 32] = MEM[addr + 4U].b32
```

#### SCRATCH_LOAD_DWORDX3  (opcode 22)

Load 96 bits of data from the scratch aperture into a vector register.

```
  addr = CalcScratchAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  VDATA[31 : 0] = MEM[addr].b32;
  VDATA[63 : 32] = MEM[addr + 4U].b32;
  VDATA[95 : 64] = MEM[addr + 8U].b32
```

#### SCRATCH_LOAD_DWORDX4  (opcode 23)

Load 128 bits of data from the scratch aperture into a vector register.

```
  addr = CalcScratchAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  VDATA[31 : 0] = MEM[addr].b32;
  VDATA[63 : 32] = MEM[addr + 4U].b32;
  VDATA[95 : 64] = MEM[addr + 8U].b32;
  VDATA[127 : 96] = MEM[addr + 12U].b32
```

#### SCRATCH_STORE_BYTE  (opcode 24)

Store 8 bits of data from a vector register into the scratch aperture.

```
  addr = CalcScratchAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  MEM[addr].b8 = VDATA[7 : 0]
```

#### SCRATCH_STORE_BYTE_D16_HI  (opcode 25)

Store 8 bits of data from the high 16 bits of a 32-bit vector register into the scratch aperture.

```
  addr = CalcScratchAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  MEM[addr].b8 = VDATA[23 : 16]
```

#### SCRATCH_STORE_SHORT  (opcode 26)

Store 16 bits of data from a vector register into the scratch aperture.

```
  addr = CalcScratchAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  MEM[addr].b16 = VDATA[15 : 0]
```

#### SCRATCH_STORE_SHORT_D16_HI  (opcode 27)

Store 16 bits of data from the high 16 bits of a 32-bit vector register into the scratch aperture.

```
  addr = CalcScratchAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  MEM[addr].b16 = VDATA[31 : 16]
```

#### SCRATCH_STORE_DWORD  (opcode 28)

Store 32 bits of data from vector input registers into the scratch aperture.

```
  addr = CalcScratchAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  MEM[addr].b32 = VDATA[31 : 0]
```

#### SCRATCH_STORE_DWORDX2  (opcode 29)

Store 64 bits of data from vector input registers into the scratch aperture.

```
  addr = CalcScratchAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  MEM[addr].b32 = VDATA[31 : 0];
  MEM[addr + 4U].b32 = VDATA[63 : 32]
```

#### SCRATCH_STORE_DWORDX3  (opcode 30)

Store 96 bits of data from vector input registers into the scratch aperture.

```
  addr = CalcScratchAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  MEM[addr].b32 = VDATA[31 : 0];
  MEM[addr + 4U].b32 = VDATA[63 : 32];
  MEM[addr + 8U].b32 = VDATA[95 : 64]
```

#### SCRATCH_STORE_DWORDX4  (opcode 31)

Store 128 bits of data from vector input registers into the scratch aperture.

```
  addr = CalcScratchAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  MEM[addr].b32 = VDATA[31 : 0];
  MEM[addr + 4U].b32 = VDATA[63 : 32];
  MEM[addr + 8U].b32 = VDATA[95 : 64];
  MEM[addr + 12U].b32 = VDATA[127 : 96]
```

#### SCRATCH_LOAD_UBYTE_D16  (opcode 32)

Load 8 bits of unsigned data from the scratch aperture, zero extend to 16 bits and store the result into the low
16 bits of a 32-bit vector register.

```
  addr = CalcScratchAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  VDATA[15 : 0].u16 = 16'U({ 8'0U, MEM[addr].u8 });
  // VDATA[31:16] is preserved.
```

#### SCRATCH_LOAD_UBYTE_D16_HI  (opcode 33)

Load 8 bits of unsigned data from the scratch aperture, zero extend to 16 bits and store the result into the high
16 bits of a 32-bit vector register.

```
  addr = CalcScratchAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  VDATA[31 : 16].u16 = 16'U({ 8'0U, MEM[addr].u8 });
  // VDATA[15:0] is preserved.
```

#### SCRATCH_LOAD_SBYTE_D16  (opcode 34)

Load 8 bits of signed data from the scratch aperture, sign extend to 16 bits and store the result into the low 16

bits of a 32-bit vector register.

```
  addr = CalcScratchAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  VDATA[15 : 0].i16 = 16'I(signext(MEM[addr].i8));
  // VDATA[31:16] is preserved.
```

#### SCRATCH_LOAD_SBYTE_D16_HI  (opcode 35)

Load 8 bits of signed data from the scratch aperture, sign extend to 16 bits and store the result into the high 16
bits of a 32-bit vector register.

```
  addr = CalcScratchAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  VDATA[31 : 16].i16 = 16'I(signext(MEM[addr].i8));
  // VDATA[15:0] is preserved.
```

#### SCRATCH_LOAD_SHORT_D16  (opcode 36)

Load 16 bits of unsigned data from the scratch aperture and store the result into the low 16 bits of a 32-bit
vector register.

```
  addr = CalcScratchAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  VDATA[15 : 0].b16 = MEM[addr].b16;
  // VDATA[31:16] is preserved.
```

#### SCRATCH_LOAD_SHORT_D16_HI  (opcode 37)

Load 16 bits of unsigned data from the scratch aperture and store the result into the high 16 bits of a 32-bit
vector register.

```
  addr = CalcScratchAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  VDATA[31 : 16].b16 = MEM[addr].b16;
  // VDATA[15:0] is preserved.
```

#### SCRATCH_LOAD_LDS_UBYTE  (opcode 38)

Load 8 bits of untyped data from the scratch aperture, zero extend to 32 bits and store the result into a data
share.

#### SCRATCH_LOAD_LDS_SBYTE  (opcode 39)

Load 8 bits of untyped data from the scratch aperture, sign extend to 32 bits and store the result into a data
share.

#### SCRATCH_LOAD_LDS_USHORT  (opcode 40)

Load 16 bits of untyped data from the scratch aperture, zero extend to 32 bits and store the result into a data
share.

#### SCRATCH_LOAD_LDS_SSHORT  (opcode 41)

Load 16 bits of untyped data from the scratch aperture, sign extend to 32 bits and store the result into a data
share.

#### SCRATCH_LOAD_LDS_DWORD  (opcode 42)

Load 32 bits of untyped data from the scratch aperture and store the result into a data share.

#### 12.15.3. Global Instructions

Global instructions are like Flat, but assume all workitem addresses fall in global memory space.

#### GLOBAL_LOAD_UBYTE  (opcode 16)

Load 8 bits of unsigned data from the global aperture, zero extend to 32 bits and store the result into a vector
register.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  VDATA.u32 = 32'U({ 24'0U, MEM[addr].u8 })
```

#### GLOBAL_LOAD_SBYTE  (opcode 17)

Load 8 bits of signed data from the global aperture, sign extend to 32 bits and store the result into a vector
register.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  VDATA.i32 = 32'I(signext(MEM[addr].i8))
```

#### GLOBAL_LOAD_USHORT  (opcode 18)

Load 16 bits of unsigned data from the global aperture, zero extend to 32 bits and store the result into a vector
register.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  VDATA.u32 = 32'U({ 16'0U, MEM[addr].u16 })
```

#### GLOBAL_LOAD_SSHORT  (opcode 19)

Load 16 bits of signed data from the global aperture, sign extend to 32 bits and store the result into a vector
register.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  VDATA.i32 = 32'I(signext(MEM[addr].i16))
```

#### GLOBAL_LOAD_DWORD  (opcode 20)

Load 32 bits of data from the global aperture into a vector register.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  VDATA[31 : 0] = MEM[addr].b32
```

#### GLOBAL_LOAD_DWORDX2  (opcode 21)

Load 64 bits of data from the global aperture into a vector register.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  VDATA[31 : 0] = MEM[addr].b32;
  VDATA[63 : 32] = MEM[addr + 4U].b32
```

#### GLOBAL_LOAD_DWORDX3  (opcode 22)

Load 96 bits of data from the global aperture into a vector register.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  VDATA[31 : 0] = MEM[addr].b32;
  VDATA[63 : 32] = MEM[addr + 4U].b32;
  VDATA[95 : 64] = MEM[addr + 8U].b32
```

#### GLOBAL_LOAD_DWORDX4  (opcode 23)

Load 128 bits of data from the global aperture into a vector register.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  VDATA[31 : 0] = MEM[addr].b32;
  VDATA[63 : 32] = MEM[addr + 4U].b32;
  VDATA[95 : 64] = MEM[addr + 8U].b32;
  VDATA[127 : 96] = MEM[addr + 12U].b32
```

#### GLOBAL_STORE_BYTE  (opcode 24)

Store 8 bits of data from a vector register into the global aperture.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  MEM[addr].b8 = VDATA[7 : 0]
```

#### GLOBAL_STORE_BYTE_D16_HI  (opcode 25)

Store 8 bits of data from the high 16 bits of a 32-bit vector register into the global aperture.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  MEM[addr].b8 = VDATA[23 : 16]
```

#### GLOBAL_STORE_SHORT  (opcode 26)

Store 16 bits of data from a vector register into the global aperture.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  MEM[addr].b16 = VDATA[15 : 0]
```

#### GLOBAL_STORE_SHORT_D16_HI  (opcode 27)

Store 16 bits of data from the high 16 bits of a 32-bit vector register into the global aperture.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  MEM[addr].b16 = VDATA[31 : 16]
```

#### GLOBAL_STORE_DWORD  (opcode 28)

Store 32 bits of data from vector input registers into the global aperture.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  MEM[addr].b32 = VDATA[31 : 0]
```

#### GLOBAL_STORE_DWORDX2  (opcode 29)

Store 64 bits of data from vector input registers into the global aperture.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  MEM[addr].b32 = VDATA[31 : 0];
  MEM[addr + 4U].b32 = VDATA[63 : 32]
```

#### GLOBAL_STORE_DWORDX3  (opcode 30)

Store 96 bits of data from vector input registers into the global aperture.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  MEM[addr].b32 = VDATA[31 : 0];
  MEM[addr + 4U].b32 = VDATA[63 : 32];
  MEM[addr + 8U].b32 = VDATA[95 : 64]
```

#### GLOBAL_STORE_DWORDX4  (opcode 31)

Store 128 bits of data from vector input registers into the global aperture.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  MEM[addr].b32 = VDATA[31 : 0];
  MEM[addr + 4U].b32 = VDATA[63 : 32];
  MEM[addr + 8U].b32 = VDATA[95 : 64];
  MEM[addr + 12U].b32 = VDATA[127 : 96]
```

#### GLOBAL_LOAD_UBYTE_D16  (opcode 32)

Load 8 bits of unsigned data from the global aperture, zero extend to 16 bits and store the result into the low 16
bits of a 32-bit vector register.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  VDATA[15 : 0].u16 = 16'U({ 8'0U, MEM[addr].u8 });
  // VDATA[31:16] is preserved.
```

#### GLOBAL_LOAD_UBYTE_D16_HI  (opcode 33)

Load 8 bits of unsigned data from the global aperture, zero extend to 16 bits and store the result into the high 16
bits of a 32-bit vector register.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  VDATA[31 : 16].u16 = 16'U({ 8'0U, MEM[addr].u8 });
  // VDATA[15:0] is preserved.
```

#### GLOBAL_LOAD_SBYTE_D16  (opcode 34)

Load 8 bits of signed data from the global aperture, sign extend to 16 bits and store the result into the low 16

bits of a 32-bit vector register.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  VDATA[15 : 0].i16 = 16'I(signext(MEM[addr].i8));
  // VDATA[31:16] is preserved.
```

#### GLOBAL_LOAD_SBYTE_D16_HI  (opcode 35)

Load 8 bits of signed data from the global aperture, sign extend to 16 bits and store the result into the high 16
bits of a 32-bit vector register.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  VDATA[31 : 16].i16 = 16'I(signext(MEM[addr].i8));
  // VDATA[15:0] is preserved.
```

#### GLOBAL_LOAD_SHORT_D16  (opcode 36)

Load 16 bits of unsigned data from the global aperture and store the result into the low 16 bits of a 32-bit vector
register.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  VDATA[15 : 0].b16 = MEM[addr].b16;
  // VDATA[31:16] is preserved.
```

#### GLOBAL_LOAD_SHORT_D16_HI  (opcode 37)

Load 16 bits of unsigned data from the global aperture and store the result into the high 16 bits of a 32-bit
vector register.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  VDATA[31 : 16].b16 = MEM[addr].b16;
  // VDATA[15:0] is preserved.
```

#### GLOBAL_LOAD_LDS_UBYTE  (opcode 38)

Load 8 bits of untyped data from the global aperture, zero extend to 32 bits and store the result into a data
share.

#### GLOBAL_LOAD_LDS_SBYTE  (opcode 39)

Load 8 bits of untyped data from the global aperture, sign extend to 32 bits and store the result into a data
share.

#### GLOBAL_LOAD_LDS_USHORT  (opcode 40)

Load 16 bits of untyped data from the global aperture, zero extend to 32 bits and store the result into a data
share.

#### GLOBAL_LOAD_LDS_SSHORT  (opcode 41)

Load 16 bits of untyped data from the global aperture, sign extend to 32 bits and store the result into a data
share.

#### GLOBAL_LOAD_LDS_DWORD  (opcode 42)

Load 32 bits of untyped data from the global aperture and store the result into a data share.

#### GLOBAL_ATOMIC_SWAP  (opcode 64)

Swap an unsigned 32-bit integer value in the data register with a location in the global aperture. Store the
original value from global aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = DATA.b32;
  RETURN_DATA.b32 = tmp
```

#### GLOBAL_ATOMIC_CMPSWAP  (opcode 65)

Compare two unsigned 32-bit integer values stored in the data comparison register and a location in the global
aperture. Modify the memory location with a value in the data source register iff the comparison is equal. Store
the original value from global aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  tmp = MEM[addr].u32;
  src = DATA[31 : 0].u32;
  cmp = DATA[63 : 32].u32;
  MEM[addr].u32 = tmp == cmp ? src : tmp;
```

```
  RETURN_DATA.u32 = tmp
```

#### GLOBAL_ATOMIC_ADD  (opcode 66)

Add two unsigned 32-bit integer values stored in the data register and a location in the global aperture. Store
the original value from global aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  tmp = MEM[addr].u32;
  MEM[addr].u32 += DATA.u32;
  RETURN_DATA.u32 = tmp
```

#### GLOBAL_ATOMIC_SUB  (opcode 67)

Subtract an unsigned 32-bit integer value stored in the data register from a value stored in a location in the
global aperture. Store the original value from global aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  tmp = MEM[addr].u32;
  MEM[addr].u32 -= DATA.u32;
  RETURN_DATA.u32 = tmp
```

#### GLOBAL_ATOMIC_SMIN  (opcode 68)

Select the minimum of two signed 32-bit integer inputs, given two values stored in the data register and a
location in the global aperture. Update the global aperture with the selected value. Store the original value
from global aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  tmp = MEM[addr].i32;
  src = DATA.i32;
  MEM[addr].i32 = src < tmp ? src : tmp;
  RETURN_DATA.i32 = tmp
```

#### GLOBAL_ATOMIC_UMIN  (opcode 69)

Select the minimum of two unsigned 32-bit integer inputs, given two values stored in the data register and a
location in the global aperture. Update the global aperture with the selected value. Store the original value
from global aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = src < tmp ? src : tmp;
  RETURN_DATA.u32 = tmp
```

#### GLOBAL_ATOMIC_SMAX  (opcode 70)

Select the maximum of two signed 32-bit integer inputs, given two values stored in the data register and a
location in the global aperture. Update the global aperture with the selected value. Store the original value
from global aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  tmp = MEM[addr].i32;
  src = DATA.i32;
  MEM[addr].i32 = src >= tmp ? src : tmp;
  RETURN_DATA.i32 = tmp
```

#### GLOBAL_ATOMIC_UMAX  (opcode 71)

Select the maximum of two unsigned 32-bit integer inputs, given two values stored in the data register and a
location in the global aperture. Update the global aperture with the selected value. Store the original value
from global aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = src >= tmp ? src : tmp;
  RETURN_DATA.u32 = tmp
```

#### GLOBAL_ATOMIC_AND  (opcode 72)

Calculate bitwise AND given two unsigned 32-bit integer values stored in the data register and a location in the
global aperture. Store the original value from global aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = (tmp & DATA.b32);
  RETURN_DATA.b32 = tmp
```

#### GLOBAL_ATOMIC_OR  (opcode 73)

Calculate bitwise OR given two unsigned 32-bit integer values stored in the data register and a location in the
global aperture. Store the original value from global aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = (tmp | DATA.b32);
  RETURN_DATA.b32 = tmp
```

#### GLOBAL_ATOMIC_XOR  (opcode 74)

Calculate bitwise XOR given two unsigned 32-bit integer values stored in the data register and a location in the
global aperture. Store the original value from global aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = (tmp ^ DATA.b32);
  RETURN_DATA.b32 = tmp
```

#### GLOBAL_ATOMIC_INC  (opcode 75)

Increment an unsigned 32-bit integer value from a location in the global aperture with wraparound to 0 if the
value exceeds a value in the data register. Store the original value from global aperture into a vector register iff
the SC0 bit is set.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = tmp >= src ? 0U : tmp + 1U;
  RETURN_DATA.u32 = tmp
```

#### GLOBAL_ATOMIC_DEC  (opcode 76)

Decrement an unsigned 32-bit integer value from a location in the global aperture with wraparound to a value
in the data register if the decrement yields a negative value. Store the original value from global aperture into a
vector register iff the SC0 bit is set.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = ((tmp == 0U) || (tmp > src)) ? src : tmp - 1U;
```

```
  RETURN_DATA.u32 = tmp
```

#### GLOBAL_ATOMIC_ADD_F32  (opcode 77)

Add a single-precision float value in the data register to a location in the global aperture. Store the original
value from global aperture into a vector register iff the SC0 bit is set.

```
  tmp = MEM[ADDR].f32;
  MEM[ADDR].f32 += DATA.f32;
  RETURN_DATA = tmp
```

Notes

Floating-point addition handles NAN/INF/denorm.

#### GLOBAL_ATOMIC_PK_ADD_F16  (opcode 78)

Add a packed 2-component half-precision float value in the data register to a location in the global aperture.
Store the original value from global aperture into a vector register iff the SC0 bit is set.

```
  tmp = MEM[ADDR];
  src = DATA;
  dst[31 : 16].f16 = tmp[31 : 16].f16 + src[31 : 16].f16;
  dst[15 : 0].f16 = tmp[15 : 0].f16 + src[15 : 0].f16;
  MEM[ADDR] = dst.b32;
  RETURN_DATA = tmp
```

Notes

Floating-point addition handles NAN/INF/denorm.

#### GLOBAL_ATOMIC_ADD_F64  (opcode 79)

Add a double-precision float value in the data register to a location in the global aperture. Store the original
value from global aperture into a vector register iff the SC0 bit is set.

```
  tmp = MEM[ADDR].f64;
  MEM[ADDR].f64 += DATA.f64;
  RETURN_DATA = tmp
```

Notes

Floating-point addition handles NAN/INF/denorm.

#### GLOBAL_ATOMIC_MIN_F64  (opcode 80)

Select the minimum of two double-precision float inputs, given two values stored in the data register and a
location in the global aperture. Update the global aperture with the selected value. Store the original value
from global aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  tmp = MEM[addr].f64;
  src = DATA.f64;
  MEM[addr].f64 = src < tmp ? src : tmp;
  RETURN_DATA.f64 = tmp
```

#### GLOBAL_ATOMIC_MAX_F64  (opcode 81)

Select the maximum of two double-precision float inputs, given two values stored in the data register and a
location in the global aperture. Update the global aperture with the selected value. Store the original value
from global aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  tmp = MEM[addr].f64;
  src = DATA.f64;
  MEM[addr].f64 = src > tmp ? src : tmp;
  RETURN_DATA.f64 = tmp
```

#### GLOBAL_ATOMIC_PK_ADD_BF16  (opcode 82)

Add a packed 2-component BF16 float value in the data register to a location in the global aperture. Store the
original value from global aperture into a vector register iff the SC0 bit is set.

```
  tmp = MEM[ADDR];
  src = DATA;
  dst[31 : 16].bf16 = tmp[31 : 16].bf16 + src[31 : 16].bf16;
  dst[15 : 0].bf16 = tmp[15 : 0].bf16 + src[15 : 0].bf16;
  MEM[ADDR] = dst.b32;
  RETURN_DATA = tmp
```

Notes

Floating-point addition handles NAN/INF/denorm.

#### GLOBAL_ATOMIC_SWAP_X2  (opcode 96)

Swap an unsigned 64-bit integer value in the data register with a location in the global aperture. Store the
original value from global aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  tmp = MEM[addr].b64;
  MEM[addr].b64 = DATA.b64;
  RETURN_DATA.b64 = tmp
```

#### GLOBAL_ATOMIC_CMPSWAP_X2  (opcode 97)

Compare two unsigned 64-bit integer values stored in the data comparison register and a location in the global
aperture. Modify the memory location with a value in the data source register iff the comparison is equal. Store
the original value from global aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  tmp = MEM[addr].u64;
  src = DATA[63 : 0].u64;
  cmp = DATA[127 : 64].u64;
  MEM[addr].u64 = tmp == cmp ? src : tmp;
  RETURN_DATA.u64 = tmp
```

#### GLOBAL_ATOMIC_ADD_X2  (opcode 98)

Add two unsigned 64-bit integer values stored in the data register and a location in the global aperture. Store
the original value from global aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  tmp = MEM[addr].u64;
  MEM[addr].u64 += DATA.u64;
  RETURN_DATA.u64 = tmp
```

#### GLOBAL_ATOMIC_SUB_X2  (opcode 99)

Subtract an unsigned 64-bit integer value stored in the data register from a value stored in a location in the
global aperture. Store the original value from global aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  tmp = MEM[addr].u64;
  MEM[addr].u64 -= DATA.u64;
  RETURN_DATA.u64 = tmp
```

#### GLOBAL_ATOMIC_SMIN_X2  (opcode 100)

Select the minimum of two signed 64-bit integer inputs, given two values stored in the data register and a
location in the global aperture. Update the global aperture with the selected value. Store the original value
from global aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  tmp = MEM[addr].i64;
  src = DATA.i64;
  MEM[addr].i64 = src < tmp ? src : tmp;
  RETURN_DATA.i64 = tmp
```

#### GLOBAL_ATOMIC_UMIN_X2  (opcode 101)

Select the minimum of two unsigned 64-bit integer inputs, given two values stored in the data register and a
location in the global aperture. Update the global aperture with the selected value. Store the original value
from global aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  tmp = MEM[addr].u64;
  src = DATA.u64;
  MEM[addr].u64 = src < tmp ? src : tmp;
  RETURN_DATA.u64 = tmp
```

#### GLOBAL_ATOMIC_SMAX_X2  (opcode 102)

Select the maximum of two signed 64-bit integer inputs, given two values stored in the data register and a
location in the global aperture. Update the global aperture with the selected value. Store the original value
from global aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  tmp = MEM[addr].i64;
  src = DATA.i64;
  MEM[addr].i64 = src >= tmp ? src : tmp;
  RETURN_DATA.i64 = tmp
```

#### GLOBAL_ATOMIC_UMAX_X2  (opcode 103)

Select the maximum of two unsigned 64-bit integer inputs, given two values stored in the data register and a
location in the global aperture. Update the global aperture with the selected value. Store the original value
from global aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  tmp = MEM[addr].u64;
  src = DATA.u64;
  MEM[addr].u64 = src >= tmp ? src : tmp;
  RETURN_DATA.u64 = tmp
```

#### GLOBAL_ATOMIC_AND_X2  (opcode 104)

Calculate bitwise AND given two unsigned 64-bit integer values stored in the data register and a location in the
global aperture. Store the original value from global aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  tmp = MEM[addr].b64;
  MEM[addr].b64 = (tmp & DATA.b64);
  RETURN_DATA.b64 = tmp
```

#### GLOBAL_ATOMIC_OR_X2  (opcode 105)

Calculate bitwise OR given two unsigned 64-bit integer values stored in the data register and a location in the
global aperture. Store the original value from global aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  tmp = MEM[addr].b64;
  MEM[addr].b64 = (tmp | DATA.b64);
  RETURN_DATA.b64 = tmp
```

#### GLOBAL_ATOMIC_XOR_X2  (opcode 106)

Calculate bitwise XOR given two unsigned 64-bit integer values stored in the data register and a location in the
global aperture. Store the original value from global aperture into a vector register iff the SC0 bit is set.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  tmp = MEM[addr].b64;
  MEM[addr].b64 = (tmp ^ DATA.b64);
  RETURN_DATA.b64 = tmp
```

#### GLOBAL_ATOMIC_INC_X2  (opcode 107)

Increment an unsigned 64-bit integer value from a location in the global aperture with wraparound to 0 if the
value exceeds a value in the data register. Store the original value from global aperture into a vector register iff

the SC0 bit is set.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  tmp = MEM[addr].u64;
  src = DATA.u64;
  MEM[addr].u64 = tmp >= src ? 0ULL : tmp + 1ULL;
  RETURN_DATA.u64 = tmp
```

#### GLOBAL_ATOMIC_DEC_X2  (opcode 108)

Decrement an unsigned 64-bit integer value from a location in the global aperture with wraparound to a value
in the data register if the decrement yields a negative value. Store the original value from global aperture into a
vector register iff the SC0 bit is set.

```
  addr = CalcGlobalAddr(ADDR.b32, SADDR.b32, OFFSET.b32);
  tmp = MEM[addr].u64;
  src = DATA.u64;
  MEM[addr].u64 = ((tmp == 0ULL) || (tmp > src)) ? src : tmp - 1ULL;
  RETURN_DATA.u64 = tmp
```

#### GLOBAL_LOAD_LDS_DWORDX4  (opcode 125)

Untyped buffer load 4 dwords, store result into data share.

#### GLOBAL_LOAD_LDS_DWORDX3  (opcode 126)

Untyped buffer load 3 dwords, store result into data share.

### 12.16. Instruction Limitations

#### 12.16.1. DPP

The following instructions cannot use DPP:

- V_MADMK_F32
- V_MADAK_F32
- V_MADMK_F16
- V_MADAK_F16
- V_READFIRSTLANE_B32
- V_CVT_I32_F64
- V_CVT_F64_I32
- V_CVT_F32_F64
- V_CVT_F64_F32
- V_CVT_U32_F64
- V_CVT_F64_U32
- V_TRUNC_F64
- V_CEIL_F64
- V_RNDNE_F64
- V_FLOOR_F64
- V_RCP_F64
- V_RSQ_F64
- V_SQRT_F64
- V_FREXP_EXP_I32_F64
- V_FREXP_MANT_F64
- V_FRACT_F64
- V_CLREXCP
- V_SWAP_B32
- V_CMP_CLASS_F64
- V_CMPX_CLASS_F64
- V_CMP_*_F64
- V_CMPX_*_F64
- V_CMP_*_I64
- V_CMP_*_U64
- V_CMPX_*_I64
- V_CMPX_*_U64

#### 12.16.2. SDWA

The following instructions cannot use SDWA:

- V_MAC_F32
- V_MADMK_F32

- V_MADAK_F32
- V_MAC_F16
- V_MADMK_F16
- V_MADAK_F16
- V_FMAC_F32
- V_READFIRSTLANE_B32
- V_CLREXCP
- V_SWAP_B32
