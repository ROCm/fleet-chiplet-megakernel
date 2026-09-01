# CDNA4 ISA Instructions: Scalar Memory (SMEM)

> Part of the **CDNA4 Instruction Set Architecture** reference. See the [index](README.md) for other sections.

## Sections in this file

  - [12.6. SMEM Instructions](#126-smem-instructions)

## Instruction mnemonics in this file

- **12.6. SMEM Instructions**: S_LOAD_DWORD, S_LOAD_DWORDX2, S_LOAD_DWORDX4, S_LOAD_DWORDX8, S_LOAD_DWORDX16, S_SCRATCH_LOAD_DWORD, S_SCRATCH_LOAD_DWORDX2, S_SCRATCH_LOAD_DWORDX4, S_BUFFER_LOAD_DWORD, S_BUFFER_LOAD_DWORDX2, S_BUFFER_LOAD_DWORDX4, S_BUFFER_LOAD_DWORDX8, S_BUFFER_LOAD_DWORDX16, S_STORE_DWORD, S_STORE_DWORDX2, S_STORE_DWORDX4, S_SCRATCH_STORE_DWORD, S_SCRATCH_STORE_DWORDX2, S_SCRATCH_STORE_DWORDX4, S_BUFFER_STORE_DWORD, S_BUFFER_STORE_DWORDX2, S_BUFFER_STORE_DWORDX4, S_DCACHE_INV, S_DCACHE_WB, S_DCACHE_INV_VOL, S_DCACHE_WB_VOL, S_MEMTIME, S_MEMREALTIME, S_DCACHE_DISCARD, S_DCACHE_DISCARD_X2, S_BUFFER_ATOMIC_SWAP, S_BUFFER_ATOMIC_CMPSWAP, S_BUFFER_ATOMIC_ADD, S_BUFFER_ATOMIC_SUB, S_BUFFER_ATOMIC_SMIN, S_BUFFER_ATOMIC_UMIN, S_BUFFER_ATOMIC_SMAX, S_BUFFER_ATOMIC_UMAX, S_BUFFER_ATOMIC_AND, S_BUFFER_ATOMIC_OR, S_BUFFER_ATOMIC_XOR, S_BUFFER_ATOMIC_INC, S_BUFFER_ATOMIC_DEC, S_BUFFER_ATOMIC_SWAP_X2, S_BUFFER_ATOMIC_CMPSWAP_X2, S_BUFFER_ATOMIC_ADD_X2, S_BUFFER_ATOMIC_SUB_X2, S_BUFFER_ATOMIC_SMIN_X2, S_BUFFER_ATOMIC_UMIN_X2, S_BUFFER_ATOMIC_SMAX_X2, S_BUFFER_ATOMIC_UMAX_X2, S_BUFFER_ATOMIC_AND_X2, S_BUFFER_ATOMIC_OR_X2, S_BUFFER_ATOMIC_XOR_X2, S_BUFFER_ATOMIC_INC_X2, S_BUFFER_ATOMIC_DEC_X2, S_ATOMIC_SWAP, S_ATOMIC_CMPSWAP, S_ATOMIC_ADD, S_ATOMIC_SUB, S_ATOMIC_SMIN, S_ATOMIC_UMIN, S_ATOMIC_SMAX, S_ATOMIC_UMAX, S_ATOMIC_AND, S_ATOMIC_OR, S_ATOMIC_XOR, S_ATOMIC_INC, S_ATOMIC_DEC, S_ATOMIC_SWAP_X2, S_ATOMIC_CMPSWAP_X2, S_ATOMIC_ADD_X2, S_ATOMIC_SUB_X2, S_ATOMIC_SMIN_X2, S_ATOMIC_UMIN_X2, S_ATOMIC_SMAX_X2, S_ATOMIC_UMAX_X2, S_ATOMIC_AND_X2, S_ATOMIC_OR_X2, S_ATOMIC_XOR_X2, S_ATOMIC_INC_X2, S_ATOMIC_DEC_X2

---

### 12.6. SMEM Instructions

#### S_LOAD_DWORD  (opcode 0)

Load 32 bits of data from the scalar memory into a scalar register.

```
  addr = CalcScalarGlobalAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  SDATA[31 : 0] = MEM[addr].b32
```

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

#### S_LOAD_DWORDX2  (opcode 1)

Load 64 bits of data from the scalar memory into a scalar register.

```
  addr = CalcScalarGlobalAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  SDATA[31 : 0] = MEM[addr].b32;
  SDATA[63 : 32] = MEM[addr + 4U].b32
```

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

#### S_LOAD_DWORDX4  (opcode 2)

Load 128 bits of data from the scalar memory into a scalar register.

```
  addr = CalcScalarGlobalAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  SDATA[31 : 0] = MEM[addr].b32;
  SDATA[63 : 32] = MEM[addr + 4U].b32;
  SDATA[95 : 64] = MEM[addr + 8U].b32;
  SDATA[127 : 96] = MEM[addr + 12U].b32
```

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

#### S_LOAD_DWORDX8  (opcode 3)

Load 256 bits of data from the scalar memory into a scalar register.

```
  addr = CalcScalarGlobalAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  SDATA[31 : 0] = MEM[addr].b32;
  SDATA[63 : 32] = MEM[addr + 4U].b32;
  SDATA[95 : 64] = MEM[addr + 8U].b32;
  SDATA[127 : 96] = MEM[addr + 12U].b32;
  SDATA[159 : 128] = MEM[addr + 16U].b32;
  SDATA[191 : 160] = MEM[addr + 20U].b32;
  SDATA[223 : 192] = MEM[addr + 24U].b32;
  SDATA[255 : 224] = MEM[addr + 28U].b32
```

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

#### S_LOAD_DWORDX16  (opcode 4)

Load 512 bits of data from the scalar memory into a scalar register.

```
  addr = CalcScalarGlobalAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  SDATA[31 : 0] = MEM[addr].b32;
  SDATA[63 : 32] = MEM[addr + 4U].b32;
```

```
  SDATA[95 : 64] = MEM[addr + 8U].b32;
  SDATA[127 : 96] = MEM[addr + 12U].b32;
  SDATA[159 : 128] = MEM[addr + 16U].b32;
  SDATA[191 : 160] = MEM[addr + 20U].b32;
  SDATA[223 : 192] = MEM[addr + 24U].b32;
  SDATA[255 : 224] = MEM[addr + 28U].b32;
  SDATA[287 : 256] = MEM[addr + 32U].b32;
  SDATA[319 : 288] = MEM[addr + 36U].b32;
  SDATA[351 : 320] = MEM[addr + 40U].b32;
  SDATA[383 : 352] = MEM[addr + 44U].b32;
  SDATA[415 : 384] = MEM[addr + 48U].b32;
  SDATA[447 : 416] = MEM[addr + 52U].b32;
  SDATA[479 : 448] = MEM[addr + 56U].b32;
  SDATA[511 : 480] = MEM[addr + 60U].b32
```

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

#### S_SCRATCH_LOAD_DWORD  (opcode 5)

Load 32 bits of data from the scalar scratch aperture into a scalar register.

```
  addr = CalcScalarScratchAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  SDATA[31 : 0] = MEM[addr].b32
```

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED 64-byte offset, consistent with other
scratch operations.

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

#### S_SCRATCH_LOAD_DWORDX2  (opcode 6)

Load 64 bits of data from the scalar scratch aperture into a scalar register.

```
  addr = CalcScalarScratchAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  SDATA[31 : 0] = MEM[addr].b32;
  SDATA[63 : 32] = MEM[addr + 4U].b32
```

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED 64-byte offset, consistent with other
scratch operations.

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

#### S_SCRATCH_LOAD_DWORDX4  (opcode 7)

Load 128 bits of data from the scalar scratch aperture into a scalar register.

```
  addr = CalcScalarScratchAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  SDATA[31 : 0] = MEM[addr].b32;
  SDATA[63 : 32] = MEM[addr + 4U].b32;
  SDATA[95 : 64] = MEM[addr + 8U].b32;
  SDATA[127 : 96] = MEM[addr + 12U].b32
```

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED 64-byte offset, consistent with other
scratch operations.

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

#### S_BUFFER_LOAD_DWORD  (opcode 8)

Load 32 bits of data from a scalar buffer surface into a scalar register.

```
  addr = CalcScalarBufferAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  SDATA[31 : 0] = MEM[addr].b32
```

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

#### S_BUFFER_LOAD_DWORDX2  (opcode 9)

Load 64 bits of data from a scalar buffer surface into a scalar register.

```
  addr = CalcScalarBufferAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  SDATA[31 : 0] = MEM[addr].b32;
  SDATA[63 : 32] = MEM[addr + 4U].b32
```

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

#### S_BUFFER_LOAD_DWORDX4  (opcode 10)

Load 128 bits of data from a scalar buffer surface into a scalar register.

```
  addr = CalcScalarBufferAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  SDATA[31 : 0] = MEM[addr].b32;
  SDATA[63 : 32] = MEM[addr + 4U].b32;
  SDATA[95 : 64] = MEM[addr + 8U].b32;
  SDATA[127 : 96] = MEM[addr + 12U].b32
```

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

#### S_BUFFER_LOAD_DWORDX8  (opcode 11)

Load 256 bits of data from a scalar buffer surface into a scalar register.

```
  addr = CalcScalarBufferAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  SDATA[31 : 0] = MEM[addr].b32;
  SDATA[63 : 32] = MEM[addr + 4U].b32;
  SDATA[95 : 64] = MEM[addr + 8U].b32;
  SDATA[127 : 96] = MEM[addr + 12U].b32;
  SDATA[159 : 128] = MEM[addr + 16U].b32;
  SDATA[191 : 160] = MEM[addr + 20U].b32;
  SDATA[223 : 192] = MEM[addr + 24U].b32;
  SDATA[255 : 224] = MEM[addr + 28U].b32
```

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

#### S_BUFFER_LOAD_DWORDX16  (opcode 12)

Load 512 bits of data from a scalar buffer surface into a scalar register.

```
  addr = CalcScalarBufferAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  SDATA[31 : 0] = MEM[addr].b32;
  SDATA[63 : 32] = MEM[addr + 4U].b32;
```

```
  SDATA[95 : 64] = MEM[addr + 8U].b32;
  SDATA[127 : 96] = MEM[addr + 12U].b32;
  SDATA[159 : 128] = MEM[addr + 16U].b32;
  SDATA[191 : 160] = MEM[addr + 20U].b32;
  SDATA[223 : 192] = MEM[addr + 24U].b32;
  SDATA[255 : 224] = MEM[addr + 28U].b32;
  SDATA[287 : 256] = MEM[addr + 32U].b32;
  SDATA[319 : 288] = MEM[addr + 36U].b32;
  SDATA[351 : 320] = MEM[addr + 40U].b32;
  SDATA[383 : 352] = MEM[addr + 44U].b32;
  SDATA[415 : 384] = MEM[addr + 48U].b32;
  SDATA[447 : 416] = MEM[addr + 52U].b32;
  SDATA[479 : 448] = MEM[addr + 56U].b32;
  SDATA[511 : 480] = MEM[addr + 60U].b32
```

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

#### S_STORE_DWORD  (opcode 16)

Store 32 bits of data from a scalar register into the scalar memory.

```
  addr = CalcScalarGlobalAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  MEM[addr].b32 = SDATA[31 : 0]
```

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

#### S_STORE_DWORDX2  (opcode 17)

Store 64 bits of data from a scalar register into the scalar memory.

```
  addr = CalcScalarGlobalAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  MEM[addr].b32 = SDATA[31 : 0];
  MEM[addr + 4U].b32 = SDATA[63 : 32]
```

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

#### S_STORE_DWORDX4  (opcode 18)

Store 128 bits of data from a scalar register into the scalar memory.

```
  addr = CalcScalarGlobalAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  MEM[addr].b32 = SDATA[31 : 0];
  MEM[addr + 4U].b32 = SDATA[63 : 32];
  MEM[addr + 8U].b32 = SDATA[95 : 64];
  MEM[addr + 12U].b32 = SDATA[127 : 96]
```

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

#### S_SCRATCH_STORE_DWORD  (opcode 21)

Store 32 bits of data from a scalar register into the scalar scratch aperture.

```
  addr = CalcScalarScratchAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  MEM[addr].b32 = SDATA[31 : 0]
```

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED 64-byte offset, consistent with other
scratch operations.

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

#### S_SCRATCH_STORE_DWORDX2  (opcode 22)

Store 64 bits of data from a scalar register into the scalar scratch aperture.

```
  addr = CalcScalarScratchAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  MEM[addr].b32 = SDATA[31 : 0];
  MEM[addr + 4U].b32 = SDATA[63 : 32]
```

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED 64-byte offset, consistent with other
scratch operations.

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

#### S_SCRATCH_STORE_DWORDX4  (opcode 23)

Store 128 bits of data from a scalar register into the scalar scratch aperture.

```
  addr = CalcScalarScratchAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  MEM[addr].b32 = SDATA[31 : 0];
  MEM[addr + 4U].b32 = SDATA[63 : 32];
  MEM[addr + 8U].b32 = SDATA[95 : 64];
  MEM[addr + 12U].b32 = SDATA[127 : 96]
```

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED 64-byte offset, consistent with other
scratch operations.

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

#### S_BUFFER_STORE_DWORD  (opcode 24)

Store 32 bits of data from a scalar register into a scalar buffer surface.

```
  addr = CalcScalarBufferAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  MEM[addr].b32 = SDATA[31 : 0]
```

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

#### S_BUFFER_STORE_DWORDX2  (opcode 25)

Store 64 bits of data from a scalar register into a scalar buffer surface.

```
  addr = CalcScalarBufferAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  MEM[addr].b32 = SDATA[31 : 0];
  MEM[addr + 4U].b32 = SDATA[63 : 32]
```

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

#### S_BUFFER_STORE_DWORDX4  (opcode 26)

Store 128 bits of data from a scalar register into a scalar buffer surface.

```
  addr = CalcScalarBufferAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  MEM[addr].b32 = SDATA[31 : 0];
  MEM[addr + 4U].b32 = SDATA[63 : 32];
  MEM[addr + 8U].b32 = SDATA[95 : 64];
  MEM[addr + 12U].b32 = SDATA[127 : 96]
```

Notes

If the offset is specified as an SGPR, the SGPR contains an UNSIGNED BYTE offset (the 2 LSBs are ignored).

If the offset is specified as an immediate 21-bit constant, the constant is a SIGNED BYTE offset.

#### S_DCACHE_INV  (opcode 32)

Invalidate the scalar (L0) data cache.

#### S_DCACHE_WB  (opcode 33)

Write back dirty data in the scalar (L0) data cache.

#### S_DCACHE_INV_VOL  (opcode 34)

Invalidate the scalar (L0) data cache volatile lines.

#### S_DCACHE_WB_VOL  (opcode 35)

Write back dirty data in the scalar (L0) data cache volatile lines.

#### S_MEMTIME  (opcode 36)

Return current 64-bit timestamp.

#### S_MEMREALTIME  (opcode 37)

Return current 64-bit RTC.

#### S_DCACHE_DISCARD  (opcode 40)

Discard one dirty scalar (L0) data cache line. A cache line is 64 bytes.

Typically, dirty cachelines (one which have been written by the shader) are written back to memory, but this
instruction allows the shader to invalidate and not write back cachelines which it has previously written. This
is a performance optimization to be used when the shader knows it no longer needs that data.

Address is calculated the same as S_STORE_DWORD, except the 6 LSBs are ignored to get the 64 byte aligned
address. LGKM count is incremented by 1 for this opcode.

#### S_DCACHE_DISCARD_X2  (opcode 41)

Discard two consecutive dirty scalar (L0) data cache lines. A cache line is 64 bytes.

Typically, dirty cachelines (one which have been written by the shader) are written back to memory, but this
instruction allows the shader to invalidate and not write back cachelines which it has previously written. This
is a performance optimization to be used when the shader knows it no longer needs that data.

Address is calculated the same as S_STORE_DWORD, except the 6 LSBs are ignored to get the 64 byte aligned
address. LGKM count is incremented by 2 for this opcode.

#### S_BUFFER_ATOMIC_SWAP  (opcode 64)

Swap an unsigned 32-bit integer value in the data register with a location in a scalar buffer surface.

```
  addr = CalcScalarBufferAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = DATA.b32;
  RETURN_DATA.b32 = tmp
```

#### S_BUFFER_ATOMIC_CMPSWAP  (opcode 65)

Compare two unsigned 32-bit integer values stored in the data comparison register and a location in a scalar
buffer surface. Modify the memory location with a value in the data source register iff the comparison is equal.

```
  addr = CalcScalarBufferAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].u32;
  src = DATA[31 : 0].u32;
```

```
  cmp = DATA[63 : 32].u32;
  MEM[addr].u32 = tmp == cmp ? src : tmp;
  RETURN_DATA.u32 = tmp
```

#### S_BUFFER_ATOMIC_ADD  (opcode 66)

Add two unsigned 32-bit integer values stored in the data register and a location in a scalar buffer surface.

```
  addr = CalcScalarBufferAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].u32;
  MEM[addr].u32 += DATA.u32;
  RETURN_DATA.u32 = tmp
```

#### S_BUFFER_ATOMIC_SUB  (opcode 67)

Subtract an unsigned 32-bit integer value stored in the data register from a value stored in a location in a scalar
buffer surface.

```
  addr = CalcScalarBufferAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].u32;
  MEM[addr].u32 -= DATA.u32;
  RETURN_DATA.u32 = tmp
```

#### S_BUFFER_ATOMIC_SMIN  (opcode 68)

Select the minimum of two signed 32-bit integer inputs, given two values stored in the data register and a
location in a scalar buffer surface. Update the scalar buffer with the selected value.

```
  addr = CalcScalarBufferAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].i32;
  src = DATA.i32;
  MEM[addr].i32 = src < tmp ? src : tmp;
  RETURN_DATA.i32 = tmp
```

#### S_BUFFER_ATOMIC_UMIN  (opcode 69)

Select the minimum of two unsigned 32-bit integer inputs, given two values stored in the data register and a
location in a scalar buffer surface. Update the scalar buffer with the selected value.

```
  addr = CalcScalarBufferAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
```

```
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = src < tmp ? src : tmp;
  RETURN_DATA.u32 = tmp
```

#### S_BUFFER_ATOMIC_SMAX  (opcode 70)

Select the maximum of two signed 32-bit integer inputs, given two values stored in the data register and a
location in a scalar buffer surface. Update the scalar buffer with the selected value.

```
  addr = CalcScalarBufferAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].i32;
  src = DATA.i32;
  MEM[addr].i32 = src >= tmp ? src : tmp;
  RETURN_DATA.i32 = tmp
```

#### S_BUFFER_ATOMIC_UMAX  (opcode 71)

Select the maximum of two unsigned 32-bit integer inputs, given two values stored in the data register and a
location in a scalar buffer surface. Update the scalar buffer with the selected value.

```
  addr = CalcScalarBufferAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = src >= tmp ? src : tmp;
  RETURN_DATA.u32 = tmp
```

#### S_BUFFER_ATOMIC_AND  (opcode 72)

Calculate bitwise AND given two unsigned 32-bit integer values stored in the data register and a location in a
scalar buffer surface.

```
  addr = CalcScalarBufferAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = (tmp & DATA.b32);
  RETURN_DATA.b32 = tmp
```

#### S_BUFFER_ATOMIC_OR  (opcode 73)

Calculate bitwise OR given two unsigned 32-bit integer values stored in the data register and a location in a
scalar buffer surface.

```
  addr = CalcScalarBufferAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = (tmp | DATA.b32);
  RETURN_DATA.b32 = tmp
```

#### S_BUFFER_ATOMIC_XOR  (opcode 74)

Calculate bitwise XOR given two unsigned 32-bit integer values stored in the data register and a location in a
scalar buffer surface.

```
  addr = CalcScalarBufferAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = (tmp ^ DATA.b32);
  RETURN_DATA.b32 = tmp
```

#### S_BUFFER_ATOMIC_INC  (opcode 75)

Increment an unsigned 32-bit integer value from a location in a scalar buffer surface with wraparound to 0 if
the value exceeds a value in the data register.

```
  addr = CalcScalarBufferAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = tmp >= src ? 0U : tmp + 1U;
  RETURN_DATA.u32 = tmp
```

#### S_BUFFER_ATOMIC_DEC  (opcode 76)

Decrement an unsigned 32-bit integer value from a location in a scalar buffer surface with wraparound to a
value in the data register if the decrement yields a negative value.

```
  addr = CalcScalarBufferAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = ((tmp == 0U) || (tmp > src)) ? src : tmp - 1U;
  RETURN_DATA.u32 = tmp
```

#### S_BUFFER_ATOMIC_SWAP_X2  (opcode 96)

Swap an unsigned 64-bit integer value in the data register with a location in a scalar buffer surface.

```
  addr = CalcScalarBufferAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].b64;
  MEM[addr].b64 = DATA.b64;
  RETURN_DATA.b64 = tmp
```

#### S_BUFFER_ATOMIC_CMPSWAP_X2  (opcode 97)

Compare two unsigned 64-bit integer values stored in the data comparison register and a location in a scalar
buffer surface. Modify the memory location with a value in the data source register iff the comparison is equal.

```
  addr = CalcScalarBufferAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].u64;
  src = DATA[63 : 0].u64;
  cmp = DATA[127 : 64].u64;
  MEM[addr].u64 = tmp == cmp ? src : tmp;
  RETURN_DATA.u64 = tmp
```

#### S_BUFFER_ATOMIC_ADD_X2  (opcode 98)

Add two unsigned 64-bit integer values stored in the data register and a location in a scalar buffer surface.

```
  addr = CalcScalarBufferAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].u64;
  MEM[addr].u64 += DATA.u64;
  RETURN_DATA.u64 = tmp
```

#### S_BUFFER_ATOMIC_SUB_X2  (opcode 99)

Subtract an unsigned 64-bit integer value stored in the data register from a value stored in a location in a scalar
buffer surface.

```
  addr = CalcScalarBufferAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].u64;
  MEM[addr].u64 -= DATA.u64;
  RETURN_DATA.u64 = tmp
```

#### S_BUFFER_ATOMIC_SMIN_X2  (opcode 100)

Select the minimum of two signed 64-bit integer inputs, given two values stored in the data register and a
location in a scalar buffer surface. Update the scalar buffer with the selected value.

```
  addr = CalcScalarBufferAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].i64;
  src = DATA.i64;
  MEM[addr].i64 = src < tmp ? src : tmp;
  RETURN_DATA.i64 = tmp
```

#### S_BUFFER_ATOMIC_UMIN_X2  (opcode 101)

Select the minimum of two unsigned 64-bit integer inputs, given two values stored in the data register and a
location in a scalar buffer surface. Update the scalar buffer with the selected value.

```
  addr = CalcScalarBufferAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].u64;
  src = DATA.u64;
  MEM[addr].u64 = src < tmp ? src : tmp;
  RETURN_DATA.u64 = tmp
```

#### S_BUFFER_ATOMIC_SMAX_X2  (opcode 102)

Select the maximum of two signed 64-bit integer inputs, given two values stored in the data register and a
location in a scalar buffer surface. Update the scalar buffer with the selected value.

```
  addr = CalcScalarBufferAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].i64;
  src = DATA.i64;
  MEM[addr].i64 = src >= tmp ? src : tmp;
  RETURN_DATA.i64 = tmp
```

#### S_BUFFER_ATOMIC_UMAX_X2  (opcode 103)

Select the maximum of two unsigned 64-bit integer inputs, given two values stored in the data register and a
location in a scalar buffer surface. Update the scalar buffer with the selected value.

```
  addr = CalcScalarBufferAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].u64;
  src = DATA.u64;
  MEM[addr].u64 = src >= tmp ? src : tmp;
  RETURN_DATA.u64 = tmp
```

#### S_BUFFER_ATOMIC_AND_X2  (opcode 104)

Calculate bitwise AND given two unsigned 64-bit integer values stored in the data register and a location in a
scalar buffer surface.

```
  addr = CalcScalarBufferAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].b64;
  MEM[addr].b64 = (tmp & DATA.b64);
  RETURN_DATA.b64 = tmp
```

#### S_BUFFER_ATOMIC_OR_X2  (opcode 105)

Calculate bitwise OR given two unsigned 64-bit integer values stored in the data register and a location in a
scalar buffer surface.

```
  addr = CalcScalarBufferAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].b64;
  MEM[addr].b64 = (tmp | DATA.b64);
  RETURN_DATA.b64 = tmp
```

#### S_BUFFER_ATOMIC_XOR_X2  (opcode 106)

Calculate bitwise XOR given two unsigned 64-bit integer values stored in the data register and a location in a
scalar buffer surface.

```
  addr = CalcScalarBufferAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].b64;
  MEM[addr].b64 = (tmp ^ DATA.b64);
  RETURN_DATA.b64 = tmp
```

#### S_BUFFER_ATOMIC_INC_X2  (opcode 107)

Increment an unsigned 64-bit integer value from a location in a scalar buffer surface with wraparound to 0 if
the value exceeds a value in the data register.

```
  addr = CalcScalarBufferAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].u64;
  src = DATA.u64;
  MEM[addr].u64 = tmp >= src ? 0ULL : tmp + 1ULL;
  RETURN_DATA.u64 = tmp
```

#### S_BUFFER_ATOMIC_DEC_X2  (opcode 108)

Decrement an unsigned 64-bit integer value from a location in a scalar buffer surface with wraparound to a
value in the data register if the decrement yields a negative value.

```
  addr = CalcScalarBufferAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].u64;
  src = DATA.u64;
  MEM[addr].u64 = ((tmp == 0ULL) || (tmp > src)) ? src : tmp - 1ULL;
  RETURN_DATA.u64 = tmp
```

#### S_ATOMIC_SWAP  (opcode 128)

Swap an unsigned 32-bit integer value in the data register with a location in the scalar memory.

```
  addr = CalcScalarGlobalAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = DATA.b32;
  RETURN_DATA.b32 = tmp
```

#### S_ATOMIC_CMPSWAP  (opcode 129)

Compare two unsigned 32-bit integer values stored in the data comparison register and a location in the scalar
memory. Modify the memory location with a value in the data source register iff the comparison is equal.

```
  addr = CalcScalarGlobalAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].u32;
  src = DATA[31 : 0].u32;
  cmp = DATA[63 : 32].u32;
  MEM[addr].u32 = tmp == cmp ? src : tmp;
  RETURN_DATA.u32 = tmp
```

#### S_ATOMIC_ADD  (opcode 130)

Add two unsigned 32-bit integer values stored in the data register and a location in the scalar memory.

```
  addr = CalcScalarGlobalAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].u32;
  MEM[addr].u32 += DATA.u32;
  RETURN_DATA.u32 = tmp
```

#### S_ATOMIC_SUB  (opcode 131)

Subtract an unsigned 32-bit integer value stored in the data register from a value stored in a location in the
scalar memory.

```
  addr = CalcScalarGlobalAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].u32;
  MEM[addr].u32 -= DATA.u32;
  RETURN_DATA.u32 = tmp
```

#### S_ATOMIC_SMIN  (opcode 132)

Select the minimum of two signed 32-bit integer inputs, given two values stored in the data register and a
location in the scalar memory. Update the scalar memory with the selected value.

```
  addr = CalcScalarGlobalAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].i32;
  src = DATA.i32;
  MEM[addr].i32 = src < tmp ? src : tmp;
  RETURN_DATA.i32 = tmp
```

#### S_ATOMIC_UMIN  (opcode 133)

Select the minimum of two unsigned 32-bit integer inputs, given two values stored in the data register and a
location in the scalar memory. Update the scalar memory with the selected value.

```
  addr = CalcScalarGlobalAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = src < tmp ? src : tmp;
  RETURN_DATA.u32 = tmp
```

#### S_ATOMIC_SMAX  (opcode 134)

Select the maximum of two signed 32-bit integer inputs, given two values stored in the data register and a
location in the scalar memory. Update the scalar memory with the selected value.

```
  addr = CalcScalarGlobalAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].i32;
  src = DATA.i32;
  MEM[addr].i32 = src >= tmp ? src : tmp;
  RETURN_DATA.i32 = tmp
```

#### S_ATOMIC_UMAX  (opcode 135)

Select the maximum of two unsigned 32-bit integer inputs, given two values stored in the data register and a
location in the scalar memory. Update the scalar memory with the selected value.

```
  addr = CalcScalarGlobalAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = src >= tmp ? src : tmp;
  RETURN_DATA.u32 = tmp
```

#### S_ATOMIC_AND  (opcode 136)

Calculate bitwise AND given two unsigned 32-bit integer values stored in the data register and a location in the
scalar memory.

```
  addr = CalcScalarGlobalAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = (tmp & DATA.b32);
  RETURN_DATA.b32 = tmp
```

#### S_ATOMIC_OR  (opcode 137)

Calculate bitwise OR given two unsigned 32-bit integer values stored in the data register and a location in the
scalar memory.

```
  addr = CalcScalarGlobalAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = (tmp | DATA.b32);
  RETURN_DATA.b32 = tmp
```

#### S_ATOMIC_XOR  (opcode 138)

Calculate bitwise XOR given two unsigned 32-bit integer values stored in the data register and a location in the
scalar memory.

```
  addr = CalcScalarGlobalAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].b32;
  MEM[addr].b32 = (tmp ^ DATA.b32);
  RETURN_DATA.b32 = tmp
```

#### S_ATOMIC_INC  (opcode 139)

Increment an unsigned 32-bit integer value from a location in the scalar memory with wraparound to 0 if the
value exceeds a value in the data register.

```
  addr = CalcScalarGlobalAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = tmp >= src ? 0U : tmp + 1U;
  RETURN_DATA.u32 = tmp
```

#### S_ATOMIC_DEC  (opcode 140)

Decrement an unsigned 32-bit integer value from a location in the scalar memory with wraparound to a value
in the data register if the decrement yields a negative value.

```
  addr = CalcScalarGlobalAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].u32;
  src = DATA.u32;
  MEM[addr].u32 = ((tmp == 0U) || (tmp > src)) ? src : tmp - 1U;
  RETURN_DATA.u32 = tmp
```

#### S_ATOMIC_SWAP_X2  (opcode 160)

Swap an unsigned 64-bit integer value in the data register with a location in the scalar memory.

```
  addr = CalcScalarGlobalAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].b64;
  MEM[addr].b64 = DATA.b64;
  RETURN_DATA.b64 = tmp
```

#### S_ATOMIC_CMPSWAP_X2  (opcode 161)

Compare two unsigned 64-bit integer values stored in the data comparison register and a location in the scalar
memory. Modify the memory location with a value in the data source register iff the comparison is equal.

```
  addr = CalcScalarGlobalAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].u64;
  src = DATA[63 : 0].u64;
  cmp = DATA[127 : 64].u64;
  MEM[addr].u64 = tmp == cmp ? src : tmp;
  RETURN_DATA.u64 = tmp
```

#### S_ATOMIC_ADD_X2  (opcode 162)

Add two unsigned 64-bit integer values stored in the data register and a location in the scalar memory.

```
  addr = CalcScalarGlobalAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].u64;
  MEM[addr].u64 += DATA.u64;
  RETURN_DATA.u64 = tmp
```

#### S_ATOMIC_SUB_X2  (opcode 163)

Subtract an unsigned 64-bit integer value stored in the data register from a value stored in a location in the
scalar memory.

```
  addr = CalcScalarGlobalAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].u64;
  MEM[addr].u64 -= DATA.u64;
  RETURN_DATA.u64 = tmp
```

#### S_ATOMIC_SMIN_X2  (opcode 164)

Select the minimum of two signed 64-bit integer inputs, given two values stored in the data register and a
location in the scalar memory. Update the scalar memory with the selected value.

```
  addr = CalcScalarGlobalAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].i64;
  src = DATA.i64;
  MEM[addr].i64 = src < tmp ? src : tmp;
  RETURN_DATA.i64 = tmp
```

#### S_ATOMIC_UMIN_X2  (opcode 165)

Select the minimum of two unsigned 64-bit integer inputs, given two values stored in the data register and a
location in the scalar memory. Update the scalar memory with the selected value.

```
  addr = CalcScalarGlobalAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].u64;
  src = DATA.u64;
  MEM[addr].u64 = src < tmp ? src : tmp;
  RETURN_DATA.u64 = tmp
```

#### S_ATOMIC_SMAX_X2  (opcode 166)

Select the maximum of two signed 64-bit integer inputs, given two values stored in the data register and a
location in the scalar memory. Update the scalar memory with the selected value.

```
  addr = CalcScalarGlobalAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].i64;
  src = DATA.i64;
  MEM[addr].i64 = src >= tmp ? src : tmp;
  RETURN_DATA.i64 = tmp
```

#### S_ATOMIC_UMAX_X2  (opcode 167)

Select the maximum of two unsigned 64-bit integer inputs, given two values stored in the data register and a
location in the scalar memory. Update the scalar memory with the selected value.

```
  addr = CalcScalarGlobalAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].u64;
  src = DATA.u64;
  MEM[addr].u64 = src >= tmp ? src : tmp;
  RETURN_DATA.u64 = tmp
```

#### S_ATOMIC_AND_X2  (opcode 168)

Calculate bitwise AND given two unsigned 64-bit integer values stored in the data register and a location in the
scalar memory.

```
  addr = CalcScalarGlobalAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].b64;
  MEM[addr].b64 = (tmp & DATA.b64);
  RETURN_DATA.b64 = tmp
```

#### S_ATOMIC_OR_X2  (opcode 169)

Calculate bitwise OR given two unsigned 64-bit integer values stored in the data register and a location in the
scalar memory.

```
  addr = CalcScalarGlobalAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].b64;
  MEM[addr].b64 = (tmp | DATA.b64);
```

```
  RETURN_DATA.b64 = tmp
```

#### S_ATOMIC_XOR_X2  (opcode 170)

Calculate bitwise XOR given two unsigned 64-bit integer values stored in the data register and a location in the
scalar memory.

```
  addr = CalcScalarGlobalAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].b64;
  MEM[addr].b64 = (tmp ^ DATA.b64);
  RETURN_DATA.b64 = tmp
```

#### S_ATOMIC_INC_X2  (opcode 171)

Increment an unsigned 64-bit integer value from a location in the scalar memory with wraparound to 0 if the
value exceeds a value in the data register.

```
  addr = CalcScalarGlobalAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].u64;
  src = DATA.u64;
  MEM[addr].u64 = tmp >= src ? 0ULL : tmp + 1ULL;
  RETURN_DATA.u64 = tmp
```

#### S_ATOMIC_DEC_X2  (opcode 172)

Decrement an unsigned 64-bit integer value from a location in the scalar memory with wraparound to a value
in the data register if the decrement yields a negative value.

```
  addr = CalcScalarGlobalAddr(SBASE.b32, SOFFSET.b32, OFFSET.i32);
  tmp = MEM[addr].u64;
  src = DATA.u64;
  MEM[addr].u64 = ((tmp == 0ULL) || (tmp > src)) ? src : tmp - 1ULL;
  RETURN_DATA.u64 = tmp
```
