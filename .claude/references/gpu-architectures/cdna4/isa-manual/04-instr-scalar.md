# CDNA4 ISA Instructions: Scalar (SOP2/SOPK/SOP1/SOPC/SOPP)

> Part of the **CDNA4 Instruction Set Architecture** reference. See the [index](README.md) for other sections.

## Sections in this file

- [Chapter 12. Instructions](#chapter-12-instructions)
  - [12.1. SOP2 Instructions](#121-sop2-instructions)
  - [12.2. SOPK Instructions](#122-sopk-instructions)
  - [12.3. SOP1 Instructions](#123-sop1-instructions)
  - [12.4. SOPC Instructions](#124-sopc-instructions)
  - [12.5. SOPP Instructions](#125-sopp-instructions)

## Instruction mnemonics in this file

- **12.1. SOP2 Instructions**: S_ADD_U32, S_SUB_U32, S_ADD_I32, S_SUB_I32, S_ADDC_U32, S_SUBB_U32, S_MIN_I32, S_MIN_U32, S_MAX_I32, S_MAX_U32, S_CSELECT_B32, S_CSELECT_B64, S_AND_B32, S_AND_B64, S_OR_B32, S_OR_B64, S_XOR_B32, S_XOR_B64, S_ANDN2_B32, S_ANDN2_B64, S_ORN2_B32, S_ORN2_B64, S_NAND_B32, S_NAND_B64, S_NOR_B32, S_NOR_B64, S_XNOR_B32, S_XNOR_B64, S_LSHL_B32, S_LSHL_B64, S_LSHR_B32, S_LSHR_B64, S_ASHR_I32, S_ASHR_I64, S_BFM_B32, S_BFM_B64, S_MUL_I32, S_BFE_U32, S_BFE_I32, S_BFE_U64, S_BFE_I64, S_CBRANCH_G_FORK, S_ABSDIFF_I32, S_MUL_HI_U32, S_MUL_HI_I32, S_LSHL1_ADD_U32, S_LSHL2_ADD_U32, S_LSHL3_ADD_U32, S_LSHL4_ADD_U32, S_PACK_LL_B32_B16, S_PACK_LH_B32_B16, S_PACK_HH_B32_B16
- **12.2. SOPK Instructions**: S_MOVK_I32, S_CMOVK_I32, S_CMPK_EQ_I32, S_CMPK_LG_I32, S_CMPK_GT_I32, S_CMPK_GE_I32, S_CMPK_LT_I32, S_CMPK_LE_I32, S_CMPK_EQ_U32, S_CMPK_LG_U32, S_CMPK_GT_U32, S_CMPK_GE_U32, S_CMPK_LT_U32, S_CMPK_LE_U32, S_ADDK_I32, S_MULK_I32, S_CBRANCH_I_FORK, S_GETREG_B32, S_SETREG_B32, S_SETREG_IMM32_B32, S_CALL_B64
- **12.3. SOP1 Instructions**: S_MOV_B32, S_MOV_B64, S_CMOV_B32, S_CMOV_B64, S_NOT_B32, S_NOT_B64, S_WQM_B32, S_WQM_B64, S_BREV_B32, S_BREV_B64, S_BCNT0_I32_B32, S_BCNT0_I32_B64, S_BCNT1_I32_B32, S_BCNT1_I32_B64, S_FF0_I32_B32, S_FF0_I32_B64, S_FF1_I32_B32, S_FF1_I32_B64, S_FLBIT_I32_B32, S_FLBIT_I32_B64, S_FLBIT_I32, S_FLBIT_I32_I64, S_SEXT_I32_I8, S_SEXT_I32_I16, S_BITSET0_B32, S_BITSET0_B64, S_BITSET1_B32, S_BITSET1_B64, S_GETPC_B64, S_SETPC_B64, S_SWAPPC_B64, S_RFE_B64, S_AND_SAVEEXEC_B64, S_OR_SAVEEXEC_B64, S_XOR_SAVEEXEC_B64, S_ANDN2_SAVEEXEC_B64, S_ORN2_SAVEEXEC_B64, S_NAND_SAVEEXEC_B64, S_NOR_SAVEEXEC_B64, S_XNOR_SAVEEXEC_B64, S_QUADMASK_B32, S_QUADMASK_B64, S_MOVRELS_B32, S_MOVRELS_B64, S_MOVRELD_B32, S_MOVRELD_B64, S_CBRANCH_JOIN, S_ABS_I32, S_SET_GPR_IDX_IDX, S_ANDN1_SAVEEXEC_B64, S_ORN1_SAVEEXEC_B64, S_ANDN1_WREXEC_B64, S_ANDN2_WREXEC_B64, S_BITREPLICATE_B64_B32
- **12.4. SOPC Instructions**: S_CMP_EQ_I32, S_CMP_LG_I32, S_CMP_GT_I32, S_CMP_GE_I32, S_CMP_LT_I32, S_CMP_LE_I32, S_CMP_EQ_U32, S_CMP_LG_U32, S_CMP_GT_U32, S_CMP_GE_U32, S_CMP_LT_U32, S_CMP_LE_U32, S_BITCMP0_B32, S_BITCMP1_B32, S_BITCMP0_B64, S_BITCMP1_B64, S_SETVSKIP, S_SET_GPR_IDX_ON, S_CMP_EQ_U64, S_CMP_LG_U64
- **12.5. SOPP Instructions**: S_NOP, S_ENDPGM, S_BRANCH, S_WAKEUP, S_CBRANCH_SCC0, S_CBRANCH_SCC1, S_CBRANCH_VCCZ, S_CBRANCH_VCCNZ, S_CBRANCH_EXECZ, S_CBRANCH_EXECNZ, S_BARRIER, S_SETKILL, S_WAITCNT, S_SETHALT, S_SLEEP, S_SETPRIO, S_SENDMSG, S_SENDMSGHALT, S_TRAP, S_ICACHE_INV, S_INCPERFLEVEL, S_DECPERFLEVEL, S_TTRACEDATA, S_CBRANCH_CDBGSYS, S_CBRANCH_CDBGUSER, S_CBRANCH_CDBGSYS_OR_USER, S_CBRANCH_CDBGSYS_AND_USER, S_ENDPGM_SAVED, S_SET_GPR_IDX_OFF, S_SET_GPR_IDX_MODE

---

## Chapter 12. Instructions

This chapter lists, and provides descriptions for, all instructions in the CDNA Generation environment.
Instructions are grouped according to their format.

Instruction suffixes have the following definitions:

- B32 Bitfield (untyped data) 32-bit
- B64 Bitfield (untyped data) 64-bit
- F32 floating-point 32-bit (IEEE 754 single-precision float)
- F64 floating-point 64-bit (IEEE 754 double-precision float)
- BF16 floating-point 16 bit (Bfloat16 format)
- I8 signed 8-bit integer
- I16 signed 16-bit integer
- I32 signed 32-bit integer
- I64 signed 64-bit integer
- U32 unsigned 32-bit integer
- U64 unsigned 64-bit integer

If an instruction has two suffixes (for example, _I32_F32), the first suffix indicates the destination type, the
second the source type.

The following abbreviations are used in instruction definitions:

- D = destination
- U = unsigned integer
- S = source
- SCC = scalar condition code
- I = signed integer
- B = bitfield

Note: .u or .i specifies to interpret the argument as an unsigned or signed float.

Note: Rounding and Denormal modes apply to all floating-point operations unless otherwise specified in the
instruction description.

### 12.1. SOP2 Instructions

Instructions in this format may use a 32-bit literal constant which occurs immediately after the instruction.

#### S_ADD_U32  (opcode 0)

Add two unsigned 32-bit integer inputs, store the result into a scalar register and store the carry-out bit into
SCC.

```
  tmp = 64'U(S0.u32) + 64'U(S1.u32);
  SCC = tmp >= 0x100000000ULL ? 1'1U : 1'0U;
  // unsigned overflow or carry-out for S_ADDC_U32.
  D0.u32 = tmp.u32
```

#### S_SUB_U32  (opcode 1)

Subtract the second unsigned 32-bit integer input from the first input, store the result into a scalar register and
store the carry-out bit into SCC.

```
  tmp = S0.u32 - S1.u32;
  SCC = S1.u32 > S0.u32 ? 1'1U : 1'0U;
  // unsigned overflow or carry-out for S_SUBB_U32.
  D0.u32 = tmp.u32
```

#### S_ADD_I32  (opcode 2)

Add two signed 32-bit integer inputs, store the result into a scalar register and store the carry-out bit into SCC.

```
  tmp = S0.i32 + S1.i32;
  SCC = ((S0.u32[31] == S1.u32[31]) && (S0.u32[31] != tmp.u32[31]));
  // signed overflow.
  D0.i32 = tmp.i32
```

Notes

This opcode is not suitable for use with S_ADDC_U32 for implementing 64-bit operations.

#### S_SUB_I32  (opcode 3)

Subtract the second signed 32-bit integer input from the first input, store the result into a scalar register and
store the carry-out bit into SCC.

```
  tmp = S0.i32 - S1.i32;
  SCC = ((S0.u32[31] != S1.u32[31]) && (S0.u32[31] != tmp.u32[31]));
  // signed overflow.
  D0.i32 = tmp.i32
```

Notes

This opcode is not suitable for use with S_SUBB_U32 for implementing 64-bit operations.

#### S_ADDC_U32  (opcode 4)

Add two unsigned 32-bit integer inputs and a carry-in bit from SCC, store the result into a scalar register and
store the carry-out bit into SCC.

```
  tmp = 64'U(S0.u32) + 64'U(S1.u32) + SCC.u64;
  SCC = tmp >= 0x100000000ULL ? 1'1U : 1'0U;
  // unsigned overflow or carry-out for S_ADDC_U32.
  D0.u32 = tmp.u32
```

#### S_SUBB_U32  (opcode 5)

Subtract the second unsigned 32-bit integer input from the first input, subtract the carry-in bit, store the result
into a scalar register and store the carry-out bit into SCC.

```
  tmp = S0.u32 - S1.u32 - SCC.u32;
  SCC = 64'U(S1.u32) + SCC.u64 > 64'U(S0.u32) ? 1'1U : 1'0U;
  // unsigned overflow or carry-out for S_SUBB_U32.
  D0.u32 = tmp.u32
```

#### S_MIN_I32  (opcode 6)

Select the minimum of two signed 32-bit integer inputs, store the selected value into a scalar register and set
SCC iff the first value is selected.

```
  SCC = S0.i32 < S1.i32;
  D0.i32 = SCC ? S0.i32 : S1.i32
```

#### S_MIN_U32  (opcode 7)

Select the minimum of two unsigned 32-bit integer inputs, store the selected value into a scalar register and set
SCC iff the first value is selected.

```
  SCC = S0.u32 < S1.u32;
  D0.u32 = SCC ? S0.u32 : S1.u32
```

#### S_MAX_I32  (opcode 8)

Select the maximum of two signed 32-bit integer inputs, store the selected value into a scalar register and set
SCC iff the first value is selected.

```
  SCC = S0.i32 >= S1.i32;
  D0.i32 = SCC ? S0.i32 : S1.i32
```

#### S_MAX_U32  (opcode 9)

Select the maximum of two unsigned 32-bit integer inputs, store the selected value into a scalar register and set
SCC iff the first value is selected.

```
  SCC = S0.u32 >= S1.u32;
  D0.u32 = SCC ? S0.u32 : S1.u32
```

#### S_CSELECT_B32  (opcode 10)

Select the first input if SCC is true otherwise select the second input, then store the selected input into a scalar
register.

```
  D0.u32 = SCC ? S0.u32 : S1.u32
```

#### S_CSELECT_B64  (opcode 11)

Select the first input if SCC is true otherwise select the second input, then store the selected input into a scalar
register.

```
  D0.u64 = SCC ? S0.u64 : S1.u64
```

#### S_AND_B32  (opcode 12)

Calculate bitwise AND on two scalar inputs, store the result into a scalar register and set SCC iff the result is
nonzero.

```
  D0.u32 = (S0.u32 & S1.u32);
  SCC = D0.u32 != 0U
```

#### S_AND_B64  (opcode 13)

Calculate bitwise AND on two scalar inputs, store the result into a scalar register and set SCC iff the result is
nonzero.

```
  D0.u64 = (S0.u64 & S1.u64);
  SCC = D0.u64 != 0ULL
```

#### S_OR_B32  (opcode 14)

Calculate bitwise OR on two scalar inputs, store the result into a scalar register and set SCC iff the result is
nonzero.

```
  D0.u32 = (S0.u32 | S1.u32);
  SCC = D0.u32 != 0U
```

#### S_OR_B64  (opcode 15)

Calculate bitwise OR on two scalar inputs, store the result into a scalar register and set SCC iff the result is
nonzero.

```
  D0.u64 = (S0.u64 | S1.u64);
  SCC = D0.u64 != 0ULL
```

#### S_XOR_B32  (opcode 16)

Calculate bitwise XOR on two scalar inputs, store the result into a scalar register and set SCC iff the result is
nonzero.

```
  D0.u32 = (S0.u32 ^ S1.u32);
  SCC = D0.u32 != 0U
```

#### S_XOR_B64  (opcode 17)

Calculate bitwise XOR on two scalar inputs, store the result into a scalar register and set SCC iff the result is
nonzero.

```
  D0.u64 = (S0.u64 ^ S1.u64);
```

```
  SCC = D0.u64 != 0ULL
```

#### S_ANDN2_B32  (opcode 18)

Calculate bitwise AND with the first input and the negation of the second input, store the result into a scalar
register and set SCC if the result is nonzero.

```
  D0.u32 = (S0.u32 & ~S1.u32);
  SCC = D0.u32 != 0U
```

#### S_ANDN2_B64  (opcode 19)

Calculate bitwise AND with the first input and the negation of the second input, store the result into a scalar
register and set SCC if the result is nonzero.

```
  D0.u64 = (S0.u64 & ~S1.u64);
  SCC = D0.u64 != 0ULL
```

#### S_ORN2_B32  (opcode 20)

Calculate bitwise OR with the first input and the negation of the second input, store the result into a scalar
register and set SCC if the result is nonzero.

```
  D0.u32 = (S0.u32 | ~S1.u32);
  SCC = D0.u32 != 0U
```

#### S_ORN2_B64  (opcode 21)

Calculate bitwise OR with the first input and the negation of the second input, store the result into a scalar
register and set SCC if the result is nonzero.

```
  D0.u64 = (S0.u64 | ~S1.u64);
  SCC = D0.u64 != 0ULL
```

#### S_NAND_B32  (opcode 22)

Calculate bitwise NAND on two scalar inputs, store the result into a scalar register and set SCC if the result is

nonzero.

```
  D0.u32 = ~(S0.u32 & S1.u32);
  SCC = D0.u32 != 0U
```

#### S_NAND_B64  (opcode 23)

Calculate bitwise NAND on two scalar inputs, store the result into a scalar register and set SCC if the result is
nonzero.

```
  D0.u64 = ~(S0.u64 & S1.u64);
  SCC = D0.u64 != 0ULL
```

#### S_NOR_B32  (opcode 24)

Calculate bitwise NOR on two scalar inputs, store the result into a scalar register and set SCC if the result is
nonzero.

```
  D0.u32 = ~(S0.u32 | S1.u32);
  SCC = D0.u32 != 0U
```

#### S_NOR_B64  (opcode 25)

Calculate bitwise NOR on two scalar inputs, store the result into a scalar register and set SCC if the result is
nonzero.

```
  D0.u64 = ~(S0.u64 | S1.u64);
  SCC = D0.u64 != 0ULL
```

#### S_XNOR_B32  (opcode 26)

Calculate bitwise XNOR on two scalar inputs, store the result into a scalar register and set SCC if the result is
nonzero.

```
  D0.u32 = ~(S0.u32 ^ S1.u32);
  SCC = D0.u32 != 0U
```

#### S_XNOR_B64  (opcode 27)

Calculate bitwise XNOR on two scalar inputs, store the result into a scalar register and set SCC if the result is
nonzero.

```
  D0.u64 = ~(S0.u64 ^ S1.u64);
  SCC = D0.u64 != 0ULL
```

#### S_LSHL_B32  (opcode 28)

Given a shift count in the second scalar input, calculate the logical shift left of the first scalar input, store the
result into a scalar register and set SCC iff the result is nonzero.

```
  D0.u32 = (S0.u32 << S1[4 : 0].u32);
  SCC = D0.u32 != 0U
```

#### S_LSHL_B64  (opcode 29)

Given a shift count in the second scalar input, calculate the logical shift left of the first scalar input, store the
result into a scalar register and set SCC iff the result is nonzero.

```
  D0.u64 = (S0.u64 << S1[5 : 0].u32);
  SCC = D0.u64 != 0ULL
```

#### S_LSHR_B32  (opcode 30)

Given a shift count in the second scalar input, calculate the logical shift right of the first scalar input, store the
result into a scalar register and set SCC iff the result is nonzero.

```
  D0.u32 = (S0.u32 >> S1[4 : 0].u32);
  SCC = D0.u32 != 0U
```

#### S_LSHR_B64  (opcode 31)

Given a shift count in the second scalar input, calculate the logical shift right of the first scalar input, store the
result into a scalar register and set SCC iff the result is nonzero.

```
  D0.u64 = (S0.u64 >> S1[5 : 0].u32);
```

```
  SCC = D0.u64 != 0ULL
```

#### S_ASHR_I32  (opcode 32)

Given a shift count in the second scalar input, calculate the arithmetic shift right (preserving sign bit) of the
first scalar input, store the result into a scalar register and set SCC iff the result is nonzero.

```
  D0.i32 = 32'I(signext(S0.i32) >> S1[4 : 0].u32);
  SCC = D0.i32 != 0
```

#### S_ASHR_I64  (opcode 33)

Given a shift count in the second scalar input, calculate the arithmetic shift right (preserving sign bit) of the
first scalar input, store the result into a scalar register and set SCC iff the result is nonzero.

```
  D0.i64 = (signext(S0.i64) >> S1[5 : 0].u32);
  SCC = D0.i64 != 0LL
```

#### S_BFM_B32  (opcode 34)

Calculate a bitfield mask given a field offset and size and store the result in a scalar register.

```
  D0.u32 = (((1U << S0[4 : 0].u32) - 1U) << S1[4 : 0].u32)
```

#### S_BFM_B64  (opcode 35)

Calculate a bitfield mask given a field offset and size and store the result in a scalar register.

```
  D0.u64 = (((1ULL << S0[5 : 0].u32) - 1ULL) << S1[5 : 0].u32)
```

#### S_MUL_I32  (opcode 36)

Multiply two signed 32-bit integer inputs and store the result into a scalar register.

```
  D0.i32 = S0.i32 * S1.i32
```

#### S_BFE_U32  (opcode 37)

Extract an unsigned bitfield from the first input using field offset and size encoded in the second input, store
the result into a scalar register and set SCC iff the result is nonzero.

```
  D0.u32 = ((S0.u32 >> S1[4 : 0].u32) & ((1U << S1[22 : 16].u32) - 1U));
  SCC = D0.u32 != 0U
```

#### S_BFE_I32  (opcode 38)

Extract a signed bitfield from the first input using field offset and size encoded in the second input, store the
result into a scalar register and set SCC iff the result is nonzero.

```
  tmp.i32 = ((S0.i32 >> S1[4 : 0].u32) & ((1 << S1[22 : 16].u32) - 1));
  D0.i32 = signext_from_bit(tmp.i32, S1[22 : 16].u32);
  SCC = D0.i32 != 0
```

#### S_BFE_U64  (opcode 39)

Extract an unsigned bitfield from the first input using field offset and size encoded in the second input, store
the result into a scalar register and set SCC iff the result is nonzero.

```
  D0.u64 = ((S0.u64 >> S1[5 : 0].u32) & ((1ULL << S1[22 : 16].u32) - 1ULL));
  SCC = D0.u64 != 0ULL
```

#### S_BFE_I64  (opcode 40)

Extract a signed bitfield from the first input using field offset and size encoded in the second input, store the
result into a scalar register and set SCC iff the result is nonzero.

```
  tmp.i64 = ((S0.i64 >> S1[5 : 0].u32) & ((1LL << S1[22 : 16].u32) - 1LL));
  D0.i64 = signext_from_bit(tmp.i64, S1[22 : 16].u32);
  SCC = D0.i64 != 0LL
```

#### S_CBRANCH_G_FORK  (opcode 41)

Conditional branch using branch-stack.

S0 = compare mask (VCC or any SGPR) and S1 = 64-bit byte address of target instruction. See also
S_CBRANCH_JOIN.

```
  mask_pass = (S0.u64 & EXEC.u64);
  mask_fail = (~S0.u64 & EXEC.u64);
  if mask_pass == EXEC.u64 then
        PC = 64'I(S1.u64)
  elsif mask_fail == EXEC.u64 then
        PC += 4LL
  elsif bitCount(mask_fail.b64) < bitCount(mask_pass.b64) then
        EXEC = mask_fail.b64;
        SGPR[WAVE_MODE.CSP.u32 * 4U].b128 = { S1.u64, mask_pass };
        WAVE_MODE.CSP += 3'1U;
        PC += 4LL
  else
        EXEC = mask_pass.b64;
        SGPR[WAVE_MODE.CSP.u32 * 4U].b128 = { (PC + 4LL), mask_fail };
        WAVE_MODE.CSP += 3'1U;
        PC = 64'I(S1.u64)
  endif
```

#### S_ABSDIFF_I32  (opcode 42)

Calculate the absolute value of difference between two scalar inputs, store the result into a scalar register and
set SCC iff the result is nonzero.

```
  D0.i32 = S0.i32 - S1.i32;
  if D0.i32 < 0 then
        D0.i32 = -D0.i32
  endif;
  SCC = D0.i32 != 0
```

Notes

Functional examples:

```
  S_ABSDIFF_I32(0x00000002, 0x00000005) => 0x00000003
  S_ABSDIFF_I32(0xffffffff, 0x00000000) => 0x00000001
  S_ABSDIFF_I32(0x80000000, 0x00000000) => 0x80000000        // Note: result is negative!
  S_ABSDIFF_I32(0x80000000, 0x00000001) => 0x7fffffff
  S_ABSDIFF_I32(0x80000000, 0xffffffff) => 0x7fffffff
  S_ABSDIFF_I32(0x80000000, 0xfffffffe) => 0x7ffffffe
```

#### S_MUL_HI_U32  (opcode 44)

Multiply two unsigned integers and store the high 32 bits of the result into a scalar register.

```
  D0.u32 = 32'U((64'U(S0.u32) * 64'U(S1.u32)) >> 32U)
```

#### S_MUL_HI_I32  (opcode 45)

Multiply two signed integers and store the high 32 bits of the result into a scalar register.

```
  D0.i32 = 32'I((64'I(S0.i32) * 64'I(S1.i32)) >> 32U)
```

#### S_LSHL1_ADD_U32  (opcode 46)

Calculate the logical shift left of the first input by 1, then add the second input, store the result into a scalar
register and set SCC iff the summation results in an unsigned overflow.

```
  tmp = (64'U(S0.u32) << 1U) + 64'U(S1.u32);
  SCC = tmp >= 0x100000000ULL ? 1'1U : 1'0U;
  // unsigned overflow.
  D0.u32 = tmp.u32
```

#### S_LSHL2_ADD_U32  (opcode 47)

Calculate the logical shift left of the first input by 2, then add the second input, store the result into a scalar
register and set SCC iff the summation results in an unsigned overflow.

```
  tmp = (64'U(S0.u32) << 2U) + 64'U(S1.u32);
  SCC = tmp >= 0x100000000ULL ? 1'1U : 1'0U;
  // unsigned overflow.
  D0.u32 = tmp.u32
```

#### S_LSHL3_ADD_U32  (opcode 48)

Calculate the logical shift left of the first input by 3, then add the second input, store the result into a scalar
register and set SCC iff the summation results in an unsigned overflow.

```
  tmp = (64'U(S0.u32) << 3U) + 64'U(S1.u32);
  SCC = tmp >= 0x100000000ULL ? 1'1U : 1'0U;
  // unsigned overflow.
  D0.u32 = tmp.u32
```

#### S_LSHL4_ADD_U32  (opcode 49)

Calculate the logical shift left of the first input by 4, then add the second input, store the result into a scalar
register and set SCC iff the summation results in an unsigned overflow.

```
  tmp = (64'U(S0.u32) << 4U) + 64'U(S1.u32);
  SCC = tmp >= 0x100000000ULL ? 1'1U : 1'0U;
  // unsigned overflow.
  D0.u32 = tmp.u32
```

#### S_PACK_LL_B32_B16  (opcode 50)

Pack two 16-bit scalar values into a scalar register.

```
  D0 = { S1[15 : 0].u16, S0[15 : 0].u16 }
```

#### S_PACK_LH_B32_B16  (opcode 51)

Pack two 16-bit scalar values into a scalar register.

```
  D0 = { S1[31 : 16].u16, S0[15 : 0].u16 }
```

#### S_PACK_HH_B32_B16  (opcode 52)

Pack two 16-bit scalar values into a scalar register.

```
  D0 = { S1[31 : 16].u16, S0[31 : 16].u16 }
```

### 12.2. SOPK Instructions

Instructions in this format may use a 32-bit literal constant which occurs immediately after the instruction.

#### S_MOVK_I32  (opcode 0)

Sign extend a literal 16-bit constant and store the result into a scalar register.

```
  D0.i32 = 32'I(signext(S0.i16))
```

#### S_CMOVK_I32  (opcode 1)

Move the sign extension of a literal 16-bit constant into a scalar register iff SCC is nonzero.

```
  if SCC then
       D0.i32 = 32'I(signext(S0.i16))
  endif
```

#### S_CMPK_EQ_I32  (opcode 2)

Set SCC to 1 iff scalar input is equal to the sign extension of a literal 16-bit constant.

```
  SCC = S0.i32 == 32'I(signext(S1.i16))
```

#### S_CMPK_LG_I32  (opcode 3)

Set SCC to 1 iff scalar input is less than or greater than the sign extension of a literal 16-bit constant.

```
  SCC = S0.i32 != 32'I(signext(S1.i16))
```

#### S_CMPK_GT_I32  (opcode 4)

Set SCC to 1 iff scalar input is greater than the sign extension of a literal 16-bit constant.

```
  SCC = S0.i32 > 32'I(signext(S1.i16))
```

#### S_CMPK_GE_I32  (opcode 5)

Set SCC to 1 iff scalar input is greater than or equal to the sign extension of a literal 16-bit constant.

```
  SCC = S0.i32 >= 32'I(signext(S1.i16))
```

#### S_CMPK_LT_I32  (opcode 6)

Set SCC to 1 iff scalar input is less than the sign extension of a literal 16-bit constant.

```
  SCC = S0.i32 < 32'I(signext(S1.i16))
```

#### S_CMPK_LE_I32  (opcode 7)

Set SCC to 1 iff scalar input is less than or equal to the sign extension of a literal 16-bit constant.

```
  SCC = S0.i32 <= 32'I(signext(S1.i16))
```

#### S_CMPK_EQ_U32  (opcode 8)

Set SCC to 1 iff scalar input is equal to the zero extension of a literal 16-bit constant.

```
  SCC = S0.u32 == 32'U(S1.u16)
```

#### S_CMPK_LG_U32  (opcode 9)

Set SCC to 1 iff scalar input is less than or greater than the zero extension of a literal 16-bit constant.

```
  SCC = S0.u32 != 32'U(S1.u16)
```

#### S_CMPK_GT_U32  (opcode 10)

Set SCC to 1 iff scalar input is greater than the zero extension of a literal 16-bit constant.

```
  SCC = S0.u32 > 32'U(S1.u16)
```

#### S_CMPK_GE_U32  (opcode 11)

Set SCC to 1 iff scalar input is greater than or equal to the zero extension of a literal 16-bit constant.

```
  SCC = S0.u32 >= 32'U(S1.u16)
```

#### S_CMPK_LT_U32  (opcode 12)

Set SCC to 1 iff scalar input is less than the zero extension of a literal 16-bit constant.

```
  SCC = S0.u32 < 32'U(S1.u16)
```

#### S_CMPK_LE_U32  (opcode 13)

Set SCC to 1 iff scalar input is less than or equal to the zero extension of a literal 16-bit constant.

```
  SCC = S0.u32 <= 32'U(S1.u16)
```

#### S_ADDK_I32  (opcode 14)

Add a scalar input and the sign extension of a literal 16-bit constant, store the result into a scalar register and
store the carry-out bit into SCC.

```
  tmp = D0.i32;
  // Save value to check sign bits for overflow later.
  D0.i32 = D0.i32 + 32'I(signext(S0.i16));
  SCC = ((tmp[31] == S0.i16[15]) && (tmp[31] != D0.i32[31]));
  // signed overflow.
```

#### S_MULK_I32  (opcode 15)

Multiply a scalar input with the sign extension of a literal 16-bit constant and store the result into a scalar
register.

```
  D0.i32 = D0.i32 * 32'I(signext(S0.i16))
```

#### S_CBRANCH_I_FORK  (opcode 16)

Conditional branch using branch-stack.

S0 = compare mask (VCC or any SGPR), and SIMM16 = signed DWORD branch offset relative to next
instruction. See also S_CBRANCH_JOIN.

```
  // Initial setup.
  mask_pass = (S0.u64 & EXEC.u64);
  mask_fail = (~S0.u64 & EXEC.u64);
  target_addr = PC + signext(SIMM16.i32 * 4) + 4LL;
  // Decide where to jump to.
  if mask_pass == EXEC.u64 then
      PC = target_addr
  elsif mask_fail == EXEC.u64 then
      PC += 4LL
  elsif bitCount(mask_fail.b64) < bitCount(mask_pass.b64) then
      EXEC = mask_fail.b64;
      SGPR[WAVE_MODE.CSP.u32 * 4U].b128 = { target_addr, mask_pass };
      WAVE_MODE.CSP += 3'1U;
      PC += 4LL
  else
      EXEC = mask_pass.b64;
      SGPR[WAVE_MODE.CSP.u32 * 4U].b128 = { (PC + 4LL), mask_fail };
      WAVE_MODE.CSP += 3'1U;
      PC = target_addr
  endif
```

#### S_GETREG_B32  (opcode 17)

Read some or all of a hardware register into the LSBs of destination.

```
  hwRegId = SIMM16.u16[5 : 0];
  offset = SIMM16.u16[10 : 6];
  size = SIMM16.u16[15 : 11].u32 + 1U;
  // logical size is in range 1:32
  value = HW_REGISTERS[hwRegId];
  D0.u32 = 32'U(32'I(value >> offset.u32) & ((1 << size) - 1))
```

#### S_SETREG_B32  (opcode 18)

Write some or all of the LSBs of source argument into a hardware register.

```
  hwRegId = SIMM16.u16[5 : 0];
  offset = SIMM16.u16[10 : 6];
  size = SIMM16.u16[15 : 11].u32 + 1U;
  // logical size is in range 1:32
  mask = (1 << size) - 1;
  mask = (mask << offset.u32);
  mask = (mask & HwRegWriteMask(hwRegId, WAVE_STATUS.PRIV));
  // Mask of bits that can be modified
  value = ((S0.u32 << offset.u32) & mask.u32);
  value = (value | 32'U(HW_REGISTERS[hwRegId].i32 & ~mask));
  HW_REGISTERS[hwRegId] = value.b32;
  // Side-effects may trigger here if certain bits are modified
```

#### S_SETREG_IMM32_B32  (opcode 20)

Write some or all of the LSBs of a 32-bit literal constant into a hardware register; this instruction requires a 32-
bit literal constant.

```
  hwRegId = SIMM16.u16[5 : 0];
  offset = SIMM16.u16[10 : 6];
  size = SIMM16.u16[15 : 11].u32 + 1U;
  // logical size is in range 1:32
  mask = (1 << size) - 1;
  mask = (mask << offset.u32);
  mask = (mask & HwRegWriteMask(hwRegId, WAVE_STATUS.PRIV));
  // Mask of bits that can be modified
  value = ((SIMM32.u32 << offset.u32) & mask.u32);
  value = (value | 32'U(HW_REGISTERS[hwRegId].i32 & ~mask));
  HW_REGISTERS[hwRegId] = value.b32;
  // Side-effects may trigger here if certain bits are modified
```

#### S_CALL_B64  (opcode 21)

Store the address of the next instruction to a scalar register and then jump to a constant offset relative to the
current PC.

The literal argument is a signed DWORD offset relative to the PC of the next instruction. The byte address of
the instruction immediately following this instruction is saved to the destination.

```
  D0.i64 = PC + 4LL;
  PC = PC + signext(SIMM16.i16 * 16'4) + 4LL
```

Notes

This implements a short subroutine call where the return address (the next instruction after the S_CALL_B64)
is saved to D. Long calls should consider S_SWAPPC_B64 instead.

This instruction must be 4 bytes.

### 12.3. SOP1 Instructions

Instructions in this format may use a 32-bit literal constant which occurs immediately after the instruction.

#### S_MOV_B32  (opcode 0)

Move scalar input into a scalar register.

```
  D0.b32 = S0.b32
```

#### S_MOV_B64  (opcode 1)

Move scalar input into a scalar register.

```
  D0.b64 = S0.b64
```

#### S_CMOV_B32  (opcode 2)

Move scalar input into a scalar register iff SCC is nonzero.

```
  if SCC then
      D0.b32 = S0.b32
  endif
```

#### S_CMOV_B64  (opcode 3)

Move scalar input into a scalar register iff SCC is nonzero.

```
  if SCC then
      D0.b64 = S0.b64
  endif
```

#### S_NOT_B32  (opcode 4)

Calculate bitwise negation on a scalar input, store the result into a scalar register and set SCC iff the result is
nonzero.

```
  D0.u32 = ~S0.u32;
  SCC = D0.u32 != 0U
```

#### S_NOT_B64  (opcode 5)

Calculate bitwise negation on a scalar input, store the result into a scalar register and set SCC iff the result is
nonzero.

```
  D0.u64 = ~S0.u64;
  SCC = D0.u64 != 0ULL
```

#### S_WQM_B32  (opcode 6)

Given an active pixel mask in a scalar input, calculate whole quad mode mask for that input, store the result
into a scalar register and set SCC iff the result is nonzero.

In whole quad mode, if any pixel in a quad is active then all pixels of the quad are marked active.

```
  tmp = 0U;
  declare i : 6'U;
  for i in 6'0U : 6'31U do
      tmp[i] = S0.u32[i & 6'60U +: 6'4U] != 0U
  endfor;
  D0.u32 = tmp;
  SCC = D0.u32 != 0U
```

#### S_WQM_B64  (opcode 7)

Given an active pixel mask in a scalar input, calculate whole quad mode mask for that input, store the result
into a scalar register and set SCC iff the result is nonzero.

In whole quad mode, if any pixel in a quad is active then all pixels of the quad are marked active.

```
  tmp = 0ULL;
  declare i : 6'U;
  for i in 6'0U : 6'63U do
      tmp[i] = S0.u64[i & 6'60U +: 6'4U] != 0ULL
  endfor;
  D0.u64 = tmp;
  SCC = D0.u64 != 0ULL
```

#### S_BREV_B32  (opcode 8)

Reverse the order of bits in a scalar input and store the result into a scalar register.

```
  D0.u32[31 : 0] = S0.u32[0 : 31]
```

#### S_BREV_B64  (opcode 9)

Reverse the order of bits in a scalar input and store the result into a scalar register.

```
  D0.u64[63 : 0] = S0.u64[0 : 63]
```

#### S_BCNT0_I32_B32  (opcode 10)

Count the number of "0" bits in a scalar input, store the result into a scalar register and set SCC iff the result is
nonzero.

```
  tmp = 0;
  for i in 0 : 31 do
        tmp += S0.u32[i] == 1'0U ? 1 : 0
  endfor;
  D0.i32 = tmp;
  SCC = D0.u32 != 0U
```

Notes

Functional examples:

```
  S_BCNT0_I32_B32(0x00000000) => 32
  S_BCNT0_I32_B32(0xcccccccc) => 16
  S_BCNT0_I32_B32(0xffffffff) => 0
```

#### S_BCNT0_I32_B64  (opcode 11)

Count the number of "0" bits in a scalar input, store the result into a scalar register and set SCC iff the result is
nonzero.

```
  tmp = 0;
  for i in 0 : 63 do
        tmp += S0.u64[i] == 1'0U ? 1 : 0
  endfor;
```

```
  D0.i32 = tmp;
  SCC = D0.u64 != 0ULL
```

#### S_BCNT1_I32_B32  (opcode 12)

Count the number of "1" bits in a scalar input, store the result into a scalar register and set SCC iff the result is
nonzero.

```
  tmp = 0;
  for i in 0 : 31 do
        tmp += S0.u32[i] == 1'1U ? 1 : 0
  endfor;
  D0.i32 = tmp;
  SCC = D0.u32 != 0U
```

Notes

Functional examples:

```
  S_BCNT1_I32_B32(0x00000000) => 0
  S_BCNT1_I32_B32(0xcccccccc) => 16
  S_BCNT1_I32_B32(0xffffffff) => 32
```

#### S_BCNT1_I32_B64  (opcode 13)

Count the number of "1" bits in a scalar input, store the result into a scalar register and set SCC iff the result is
nonzero.

```
  tmp = 0;
  for i in 0 : 63 do
        tmp += S0.u64[i] == 1'1U ? 1 : 0
  endfor;
  D0.i32 = tmp;
  SCC = D0.u64 != 0ULL
```

#### S_FF0_I32_B32  (opcode 14)

Count the number of trailing "1" bits before the first "0" in a scalar input and store the result into a scalar
register. Store -1 if there are no "0" bits in the input.

```
  tmp = -1;
  // Set if no zeros are found
```

```
  for i in 0 : 31 do
        // Search from LSB
        if S0.u32[i] == 1'0U then
            tmp = i;
            break
        endif
  endfor;
  D0.i32 = tmp
```

Notes

Functional examples:

```
  S_FF0_I32_B32(0xaaaaaaaa) => 0
  S_FF0_I32_B32(0x55555555) => 1
  S_FF0_I32_B32(0x00000000) => 0
  S_FF0_I32_B32(0xffffffff) => 0xffffffff
  S_FF0_I32_B32(0xfffeffff) => 16
```

#### S_FF0_I32_B64  (opcode 15)

Count the number of trailing "1" bits before the first "0" in a scalar input and store the result into a scalar
register. Store -1 if there are no "0" bits in the input.

```
  tmp = -1;
  // Set if no zeros are found
  for i in 0 : 63 do
        // Search from LSB
        if S0.u64[i] == 1'0U then
            tmp = i;
            break
        endif
  endfor;
  D0.i32 = tmp
```

#### S_FF1_I32_B32  (opcode 16)

Count the number of trailing "0" bits before the first "1" in a scalar input and store the result into a scalar
register. Store -1 if there are no "1" bits in the input.

```
  tmp = -1;
  // Set if no ones are found
  for i in 0 : 31 do
        // Search from LSB
        if S0.u32[i] == 1'1U then
            tmp = i;
            break
```

```
        endif
  endfor;
  D0.i32 = tmp
```

Notes

Functional examples:

```
  S_FF1_I32_B32(0xaaaaaaaa) => 1
  S_FF1_I32_B32(0x55555555) => 0
  S_FF1_I32_B32(0x00000000) => 0xffffffff
  S_FF1_I32_B32(0xffffffff) => 0
  S_FF1_I32_B32(0x00010000) => 16
```

#### S_FF1_I32_B64  (opcode 17)

Count the number of trailing "0" bits before the first "1" in a scalar input and store the result into a scalar
register. Store -1 if there are no "1" bits in the input.

```
  tmp = -1;
  // Set if no ones are found
  for i in 0 : 63 do
        // Search from LSB
        if S0.u64[i] == 1'1U then
            tmp = i;
            break
        endif
  endfor;
  D0.i32 = tmp
```

#### S_FLBIT_I32_B32  (opcode 18)

Count the number of leading "0" bits before the first "1" in a scalar input and store the result into a scalar
register. Store -1 if there are no "1" bits.

```
  tmp = -1;
  // Set if no ones are found
  for i in 0 : 31 do
        // Search from MSB
        if S0.u32[31 - i] == 1'1U then
            tmp = i;
            break
        endif
  endfor;
  D0.i32 = tmp
```

Notes

Functional examples:

```
  S_FLBIT_I32_B32(0x00000000) => 0xffffffff
  S_FLBIT_I32_B32(0x0000cccc) => 16
  S_FLBIT_I32_B32(0xffff3333) => 0
  S_FLBIT_I32_B32(0x7fffffff) => 1
  S_FLBIT_I32_B32(0x80000000) => 0
  S_FLBIT_I32_B32(0xffffffff) => 0
```

#### S_FLBIT_I32_B64  (opcode 19)

Count the number of leading "0" bits before the first "1" in a scalar input and store the result into a scalar
register. Store -1 if there are no "1" bits.

```
  tmp = -1;
  // Set if no ones are found
  for i in 0 : 63 do
        // Search from MSB
        if S0.u64[63 - i] == 1'1U then
            tmp = i;
            break
        endif
  endfor;
  D0.i32 = tmp
```

#### S_FLBIT_I32  (opcode 20)

Count the number of leading bits that are the same as the sign bit of a scalar input and store the result into a
scalar register. Store -1 if all input bits are the same.

```
  tmp = -1;
  // Set if all bits are the same
  for i in 1 : 31 do
        // Search from MSB
        if S0.u32[31 - i] != S0.u32[31] then
            tmp = i;
            break
        endif
  endfor;
  D0.i32 = tmp
```

Notes

Functional examples:

```
  S_FLBIT_I32(0x00000000) => 0xffffffff
  S_FLBIT_I32(0x0000cccc) => 16
  S_FLBIT_I32(0xffff3333) => 16
  S_FLBIT_I32(0x7fffffff) => 1
  S_FLBIT_I32(0x80000000) => 1
  S_FLBIT_I32(0xffffffff) => 0xffffffff
```

#### S_FLBIT_I32_I64  (opcode 21)

Count the number of leading bits that are the same as the sign bit of a scalar input and store the result into a
scalar register. Store -1 if all input bits are the same.

```
  tmp = -1;
  // Set if all bits are the same
  for i in 1 : 63 do
       // Search from MSB
       if S0.u64[63 - i] != S0.u64[63] then
            tmp = i;
            break
       endif
  endfor;
  D0.i32 = tmp
```

#### S_SEXT_I32_I8  (opcode 22)

Sign extend a signed 8 bit scalar input to 32 bits and store the result into a scalar register.

```
  D0.i32 = 32'I(signext(S0.i8))
```

#### S_SEXT_I32_I16  (opcode 23)

Sign extend a signed 16 bit scalar input to 32 bits and store the result into a scalar register.

```
  D0.i32 = 32'I(signext(S0.i16))
```

#### S_BITSET0_B32  (opcode 24)

Given a bit offset in a scalar input, set the indicated bit in the destination scalar register to 0.

```
  D0.u32[S0.u32[4 : 0]] = 1'0U
```

#### S_BITSET0_B64  (opcode 25)

Given a bit offset in a scalar input, set the indicated bit in the destination scalar register to 0.

```
  D0.u64[S0.u32[5 : 0]] = 1'0U
```

#### S_BITSET1_B32  (opcode 26)

Given a bit offset in a scalar input, set the indicated bit in the destination scalar register to 1.

```
  D0.u32[S0.u32[4 : 0]] = 1'1U
```

#### S_BITSET1_B64  (opcode 27)

Given a bit offset in a scalar input, set the indicated bit in the destination scalar register to 1.

```
  D0.u64[S0.u32[5 : 0]] = 1'1U
```

#### S_GETPC_B64  (opcode 28)

Store the address of the next instruction to a scalar register.

The byte address of the instruction immediately following this instruction is saved to the destination.

```
  D0.i64 = PC + 4LL
```

Notes

This instruction must be 4 bytes.

#### S_SETPC_B64  (opcode 29)

Jump to an address specified in a scalar register.

The argument is a byte address of the instruction to jump to.

```
  PC = S0.i64
```

#### S_SWAPPC_B64  (opcode 30)

Store the address of the next instruction to a scalar register and then jump to an address specified in the scalar
input.

The argument is a byte address of the instruction to jump to. The byte address of the instruction immediately
following this instruction is saved to the destination.

```
  jump_addr = S0.i64;
  D0.i64 = PC + 4LL;
  PC = jump_addr.i64
```

Notes

This instruction must be 4 bytes.

#### S_RFE_B64  (opcode 31)

Return from the exception handler. Clear the wave's PRIV bit and then jump to an address specified by the
scalar input.

The argument is a byte address of the instruction to jump to; this address is likely derived from the state passed
into the trap handler.

This instruction may only be used within a trap handler.

```
  WAVE_STATUS.PRIV = 1'0U;
  PC = S0.i64
```

#### S_AND_SAVEEXEC_B64  (opcode 32)

Calculate bitwise AND on the scalar input and the EXEC mask, store the calculated result into the EXEC mask,
set SCC iff the calculated result is nonzero and store the original value of the EXEC mask into the scalar
destination register.

The original EXEC mask is saved to the destination SGPRs before the bitwise operation is performed.

```
  saveexec = EXEC.u64;
```

```
  EXEC.u64 = (S0.u64 & EXEC.u64);
  D0.u64 = saveexec.u64;
  SCC = EXEC.u64 != 0ULL
```

#### S_OR_SAVEEXEC_B64  (opcode 33)

Calculate bitwise OR on the scalar input and the EXEC mask, store the calculated result into the EXEC mask, set
SCC iff the calculated result is nonzero and store the original value of the EXEC mask into the scalar destination
register.

The original EXEC mask is saved to the destination SGPRs before the bitwise operation is performed.

```
  saveexec = EXEC.u64;
  EXEC.u64 = (S0.u64 | EXEC.u64);
  D0.u64 = saveexec.u64;
  SCC = EXEC.u64 != 0ULL
```

#### S_XOR_SAVEEXEC_B64  (opcode 34)

Calculate bitwise XOR on the scalar input and the EXEC mask, store the calculated result into the EXEC mask,
set SCC iff the calculated result is nonzero and store the original value of the EXEC mask into the scalar
destination register.

The original EXEC mask is saved to the destination SGPRs before the bitwise operation is performed.

```
  saveexec = EXEC.u64;
  EXEC.u64 = (S0.u64 ^ EXEC.u64);
  D0.u64 = saveexec.u64;
  SCC = EXEC.u64 != 0ULL
```

#### S_ANDN2_SAVEEXEC_B64  (opcode 35)

Calculate bitwise AND on the scalar input and the negation of the EXEC mask, store the calculated result into
the EXEC mask, set SCC iff the calculated result is nonzero and store the original value of the EXEC mask into
the scalar destination register.

The original EXEC mask is saved to the destination SGPRs before the bitwise operation is performed.

```
  saveexec = EXEC.u64;
  EXEC.u64 = (S0.u64 & ~EXEC.u64);
  D0.u64 = saveexec.u64;
  SCC = EXEC.u64 != 0ULL
```

#### S_ORN2_SAVEEXEC_B64  (opcode 36)

Calculate bitwise OR on the scalar input and the negation of the EXEC mask, store the calculated result into the
EXEC mask, set SCC iff the calculated result is nonzero and store the original value of the EXEC mask into the
scalar destination register.

The original EXEC mask is saved to the destination SGPRs before the bitwise operation is performed.

```
  saveexec = EXEC.u64;
  EXEC.u64 = (S0.u64 | ~EXEC.u64);
  D0.u64 = saveexec.u64;
  SCC = EXEC.u64 != 0ULL
```

#### S_NAND_SAVEEXEC_B64  (opcode 37)

Calculate bitwise NAND on the scalar input and the EXEC mask, store the calculated result into the EXEC mask,
set SCC iff the calculated result is nonzero and store the original value of the EXEC mask into the scalar
destination register.

```
  saveexec = EXEC.u64;
  EXEC.u64 = ~(S0.u64 & EXEC.u64);
  D0.u64 = saveexec.u64;
  SCC = EXEC.u64 != 0ULL
```

#### S_NOR_SAVEEXEC_B64  (opcode 38)

Calculate bitwise NOR on the scalar input and the EXEC mask, store the calculated result into the EXEC mask,
set SCC iff the calculated result is nonzero and store the original value of the EXEC mask into the scalar
destination register.

```
  saveexec = EXEC.u64;
  EXEC.u64 = ~(S0.u64 | EXEC.u64);
  D0.u64 = saveexec.u64;
  SCC = EXEC.u64 != 0ULL
```

#### S_XNOR_SAVEEXEC_B64  (opcode 39)

Calculate bitwise XNOR on the scalar input and the EXEC mask, store the calculated result into the EXEC mask,
set SCC iff the calculated result is nonzero and store the original value of the EXEC mask into the scalar
destination register.

```
  saveexec = EXEC.u64;
  EXEC.u64 = ~(S0.u64 ^ EXEC.u64);
  D0.u64 = saveexec.u64;
  SCC = EXEC.u64 != 0ULL
```

#### S_QUADMASK_B32  (opcode 40)

Reduce a pixel mask from the scalar input into a quad mask, store the result in a scalar register and set SCC iff
the result is nonzero.

```
  tmp = 0U;
  for i in 0 : 7 do
        tmp[i] = S0.u32[i * 4 +: 4] != 0U
  endfor;
  D0.u32 = tmp;
  SCC = D0.u32 != 0U
```

Notes

To perform the inverse operation see S_BITREPLICATE_B64_B32.

#### S_QUADMASK_B64  (opcode 41)

Reduce a pixel mask from the scalar input into a quad mask, store the result in a scalar register and set SCC iff
the result is nonzero.

```
  tmp = 0ULL;
  for i in 0 : 15 do
        tmp[i] = S0.u64[i * 4 +: 4] != 0ULL
  endfor;
  D0.u64 = tmp;
  SCC = D0.u64 != 0ULL
```

Notes

To perform the inverse operation see S_BITREPLICATE_B64_B32.

#### S_MOVRELS_B32  (opcode 42)

Move data from a relatively-indexed scalar register into another scalar register.

```
  addr = SRC0.u32;
  // Raw value from instruction
```

```
  addr += M0.u32[31 : 0];
  D0.b32 = SGPR[addr].b32
```

Notes

Example: The following instruction sequence performs the move s5 <= s17:

```
        s_mov_b32 m0, 10
        s_movrels_b32 s5, s7
```

#### S_MOVRELS_B64  (opcode 43)

Move data from a relatively-indexed scalar register into another scalar register.

The index in M0.u and the operand address in SRC0.u must be even for this operation.

```
  addr = SRC0.u32;
  // Raw value from instruction
  addr += M0.u32[31 : 0];
  D0.b64 = SGPR[addr].b64
```

#### S_MOVRELD_B32  (opcode 44)

Move data from a scalar input into a relatively-indexed scalar register.

```
  addr = DST.u32;
  // Raw value from instruction
  addr += M0.u32[31 : 0];
  SGPR[addr].b32 = S0.b32
```

Notes

Example: The following instruction sequence performs the move s15 <= s7:

```
        s_mov_b32 m0, 10
        s_movreld_b32 s5, s7
```

#### S_MOVRELD_B64  (opcode 45)

Move data from a scalar input into a relatively-indexed scalar register.

The index in M0.u and the operand address in DST.u must be even for this operation.

```
  addr = DST.u32;
  // Raw value from instruction
  addr += M0.u32[31 : 0];
  SGPR[addr].b64 = S0.b64
```

#### S_CBRANCH_JOIN  (opcode 46)

Conditional branch join point (end of conditional branch block).

S0 is saved CSP value. See S_CBRANCH_G_FORK and S_CBRANCH_I_FORK for related instructions.

```
  saved_csp = S0.u32;
  if WAVE_MODE.CSP.u32 == saved_csp then
        PC += 4LL;
        // Second time to JOIN: continue with program.
  else
        WAVE_MODE.CSP -= 3'1U;
        // First time to JOIN; jump to other FORK path.
        { PC, EXEC } = SGPR[WAVE_MODE.CSP.u32 * 4U].b128;
        // Read 128 bits from 4 consecutive SGPRs.
  endif
```

#### S_ABS_I32  (opcode 48)

Compute the absolute value of a scalar input, store the result into a scalar register and set SCC iff the result is
nonzero.

```
  D0.i32 = S0.i32 < 0 ? -S0.i32 : S0.i32;
  SCC = D0.i32 != 0
```

Notes

Functional examples:

```
  S_ABS_I32(0x00000001) => 0x00000001
  S_ABS_I32(0x7fffffff) => 0x7fffffff
  S_ABS_I32(0x80000000) => 0x80000000        // Note this is negative!
  S_ABS_I32(0x80000001) => 0x7fffffff
  S_ABS_I32(0x80000002) => 0x7ffffffe
  S_ABS_I32(0xffffffff) => 0x00000001
```

#### S_SET_GPR_IDX_IDX  (opcode 50)

Set the index used in vector GPR indexing.

S_SET_GPR_IDX_ON, S_SET_GPR_IDX_OFF, S_SET_GPR_IDX_MODE and S_SET_GPR_IDX_IDX are related
instructions.

```
  M0[7 : 0] = S0.u32[7 : 0].b8
```

#### S_ANDN1_SAVEEXEC_B64  (opcode 51)

Calculate bitwise AND on the EXEC mask and the negation of the scalar input, store the calculated result into
the EXEC mask, set SCC iff the calculated result is nonzero and store the original value of the EXEC mask into
the scalar destination register.

The original EXEC mask is saved to the destination SGPRs before the bitwise operation is performed.

```
  saveexec = EXEC.u64;
  EXEC.u64 = (~S0.u64 & EXEC.u64);
  D0.u64 = saveexec.u64;
  SCC = EXEC.u64 != 0ULL
```

#### S_ORN1_SAVEEXEC_B64  (opcode 52)

Calculate bitwise OR on the EXEC mask and the negation of the scalar input, store the calculated result into the
EXEC mask, set SCC iff the calculated result is nonzero and store the original value of the EXEC mask into the
scalar destination register.

The original EXEC mask is saved to the destination SGPRs before the bitwise operation is performed.

```
  saveexec = EXEC.u64;
  EXEC.u64 = (~S0.u64 | EXEC.u64);
  D0.u64 = saveexec.u64;
  SCC = EXEC.u64 != 0ULL
```

#### S_ANDN1_WREXEC_B64  (opcode 53)

Calculate bitwise AND on the EXEC mask and the negation of the scalar input, store the calculated result into
the EXEC mask and also into the scalar destination register, and set SCC iff the calculated result is nonzero.

Unlike the SAVEEXEC series of opcodes, the value written to destination SGPRs is the result of the bitwise-op
result. EXEC and the destination SGPRs have the same value at the end of this instruction. This instruction is
intended to help accelerate waterfalling.

```
  EXEC.u64 = (~S0.u64 & EXEC.u64);
  D0.u64 = EXEC.u64;
  SCC = EXEC.u64 != 0ULL
```

#### S_ANDN2_WREXEC_B64  (opcode 54)

Calculate bitwise AND on the scalar input and the negation of the EXEC mask, store the calculated result into
the EXEC mask and also into the scalar destination register, and set SCC iff the calculated result is nonzero.

Unlike the SAVEEXEC series of opcodes, the value written to destination SGPRs is the result of the bitwise-op
result. EXEC and the destination SGPRs have the same value at the end of this instruction. This instruction is
intended to help accelerate waterfalling.

```
  EXEC.u64 = (S0.u64 & ~EXEC.u64);
  D0.u64 = EXEC.u64;
  SCC = EXEC.u64 != 0ULL
```

Notes

In particular, the following sequence of waterfall code is optimized by using a WREXEC instead of two separate
scalar ops:

```
        // V0 holds the index value per lane
        // save exec mask for restore at the end
        s_mov_b64 s2, exec
        // exec mask of remaining (unprocessed) threads
        s_mov_b64 s4, exec
        loop:
        // get the index value for the first active lane
        v_readfirstlane_b32      s0, v0
        // find all other lanes with same index value
        v_cmpx_eq s0, v0
        <OP>        // do the operation using the current EXEC mask. S0 holds the index.
        // mask out thread that was just executed
        // s_andn2_b64    s4, s4, exec
        // s_mov_b64      exec, s4
        s_andn2_wrexec_b64 s4, s4         // replaces above 2 ops
        // repeat until EXEC==0
        s_cbranch_scc1    loop
        s_mov_b64      exec, s2
```

#### S_BITREPLICATE_B64_B32  (opcode 55)

Substitute each bit of a 32 bit scalar input with two instances of itself and store the result into a 64 bit scalar
register.

```
  tmp = S0.u32;
  for i in 0 : 31 do
        D0.u64[i * 2] = tmp[i];
        D0.u64[i * 2 + 1] = tmp[i]
  endfor
```

Notes

This opcode can be used to convert a quad mask into a pixel mask; given quad mask in s0, the following
sequence produces a pixel mask in s2:

```
        s_bitreplicate_b64 s2, s0
        s_bitreplicate_b64 s2, s2
```

To perform the inverse operation see S_QUADMASK_B64.

### 12.4. SOPC Instructions

Instructions in this format may use a 32-bit literal constant which occurs immediately after the instruction.

#### S_CMP_EQ_I32  (opcode 0)

Set SCC to 1 iff the first scalar input is equal to the second scalar input.

```
  SCC = S0.i32 == S1.i32
```

Notes

Note that S_CMP_EQ_I32 and S_CMP_EQ_U32 are identical opcodes, but both are provided for symmetry.

#### S_CMP_LG_I32  (opcode 1)

Set SCC to 1 iff the first scalar input is less than or greater than the second scalar input.

```
  SCC = S0.i32 <> S1.i32
```

Notes

Note that S_CMP_LG_I32 and S_CMP_LG_U32 are identical opcodes, but both are provided for symmetry.

#### S_CMP_GT_I32  (opcode 2)

Set SCC to 1 iff the first scalar input is greater than the second scalar input.

```
  SCC = S0.i32 > S1.i32
```

#### S_CMP_GE_I32  (opcode 3)

Set SCC to 1 iff the first scalar input is greater than or equal to the second scalar input.

```
  SCC = S0.i32 >= S1.i32
```

#### S_CMP_LT_I32  (opcode 4)

Set SCC to 1 iff the first scalar input is less than the second scalar input.

```
  SCC = S0.i32 < S1.i32
```

#### S_CMP_LE_I32  (opcode 5)

Set SCC to 1 iff the first scalar input is less than or equal to the second scalar input.

```
  SCC = S0.i32 <= S1.i32
```

#### S_CMP_EQ_U32  (opcode 6)

Set SCC to 1 iff the first scalar input is equal to the second scalar input.

```
  SCC = S0.u32 == S1.u32
```

Notes

Note that S_CMP_EQ_I32 and S_CMP_EQ_U32 are identical opcodes, but both are provided for symmetry.

#### S_CMP_LG_U32  (opcode 7)

Set SCC to 1 iff the first scalar input is less than or greater than the second scalar input.

```
  SCC = S0.u32 <> S1.u32
```

Notes

Note that S_CMP_LG_I32 and S_CMP_LG_U32 are identical opcodes, but both are provided for symmetry.

#### S_CMP_GT_U32  (opcode 8)

Set SCC to 1 iff the first scalar input is greater than the second scalar input.

```
  SCC = S0.u32 > S1.u32
```

#### S_CMP_GE_U32  (opcode 9)

Set SCC to 1 iff the first scalar input is greater than or equal to the second scalar input.

```
  SCC = S0.u32 >= S1.u32
```

#### S_CMP_LT_U32  (opcode 10)

Set SCC to 1 iff the first scalar input is less than the second scalar input.

```
  SCC = S0.u32 < S1.u32
```

#### S_CMP_LE_U32  (opcode 11)

Set SCC to 1 iff the first scalar input is less than or equal to the second scalar input.

```
  SCC = S0.u32 <= S1.u32
```

#### S_BITCMP0_B32  (opcode 12)

Extract a bit from the first scalar input based on an index in the second scalar input, and set SCC to 1 iff the
extracted bit is equal to 0.

```
  SCC = S0.u32[S1.u32[4 : 0]] == 1'0U
```

#### S_BITCMP1_B32  (opcode 13)

Extract a bit from the first scalar input based on an index in the second scalar input, and set SCC to 1 iff the
extracted bit is equal to 1.

```
  SCC = S0.u32[S1.u32[4 : 0]] == 1'1U
```

#### S_BITCMP0_B64  (opcode 14)

Extract a bit from the first scalar input based on an index in the second scalar input, and set SCC to 1 iff the
extracted bit is equal to 0.

```
  SCC = S0.u64[S1.u32[5 : 0]] == 1'0U
```

#### S_BITCMP1_B64  (opcode 15)

Extract a bit from the first scalar input based on an index in the second scalar input, and set SCC to 1 iff the
extracted bit is equal to 1.

```
  SCC = S0.u64[S1.u32[5 : 0]] == 1'1U
```

#### S_SETVSKIP  (opcode 16)

Enables or disables VSKIP mode.

When VSKIP is enabled, no VOP*/M*BUF/MIMG/DS/FLAT instructions are issued. Note that VSKIPped memory
instructions do not manipulate the waitcnt counters; as a result, if there are outstanding memory requests the
shader may want to issue S_WAITCNT 0 prior to enabling VSKIP, otherwise the shader must be careful not to
count VSKIPped instructions in waitcnt calculations.

```
  VSKIP = S0.u32[S1.u32[4 : 0]]
```

Notes

Functional examples:

```
        s_setvskip 1, 0     // Enable vskip mode.
        s_setvskip 0, 0     // Disable vskip mode.
```

#### S_SET_GPR_IDX_ON  (opcode 17)

Enable GPR indexing mode.

Vector operations after this perform relative GPR addressing based on the contents of M0. The index is
specified in the SRC0 operand. The raw bits of the SRC1 field are read and used to set the enable bits. S1[0] =
VSRC0_REL, S1[1] = VSRC1_REL, S1[2] = VSRC2_REL and S1[3] = VDST_REL.

S_SET_GPR_IDX_ON, S_SET_GPR_IDX_OFF, S_SET_GPR_IDX_MODE and S_SET_GPR_IDX_IDX are related
instructions.

```
  WAVE_MODE.GPR_IDX_EN = 1'1U;
  M0[7 : 0] = S0.u32[7 : 0].b8;
  M0[15 : 12] = SRC1.u32[3 : 0].b4;
  // this is the direct content of raw S1 field
  // Remaining bits of M0 are unmodified.
```

#### S_CMP_EQ_U64  (opcode 18)

Set SCC to 1 iff the first scalar input is equal to the second scalar input.

```
  SCC = S0.u64 == S1.u64
```

#### S_CMP_LG_U64  (opcode 19)

Set SCC to 1 iff the first scalar input is less than or greater than the second scalar input.

```
  SCC = S0.u64 <> S1.u64
```

### 12.5. SOPP Instructions

#### S_NOP  (opcode 0)

Do nothing. Delay issue of next instruction by a small, fixed amount.

Insert 0..15 wait states based on SIMM16[3:0]. 0x0 means the next instruction can issue on the next clock, 0xf
means the next instruction can issue 16 clocks later.

```
  for i in 0U : SIMM16.u16[3 : 0].u32 do
        nop()
  endfor
```

Notes

Examples:

```
        s_nop 0           // Wait 1 cycle.
        s_nop 0xf         // Wait 16 cycles.
```

#### S_ENDPGM  (opcode 1)

End of program; terminate wavefront.

The hardware implicitly executes S_WAITCNT 0 before executing this instruction. See S_ENDPGM_SAVED for
the context-switch version of this instruction.

#### S_BRANCH  (opcode 2)

Jump to a constant offset relative to the current PC.

The literal argument is a signed DWORD offset relative to the PC of the next instruction.

```
  PC = PC + signext(SIMM16.i16 * 16'4) + 4LL;
  // short jump.
```

Notes

For a long jump or an indirect jump use S_SETPC_B64.

Examples:

```
      s_branch label      // Set SIMM16 = +4 = 0x0004
      s_nop 0      // 4 bytes
  label:
      s_nop 0      // 4 bytes
      s_branch label      // Set SIMM16 = -8 = 0xfff8
```

#### S_WAKEUP  (opcode 3)

Allow a wave to 'ping' all the other waves in its threadgroup to force them to wake up early from an S_SLEEP
instruction.

The ping is ignored if the waves are not sleeping. This allows for efficient polling on a memory location. The
waves which are polling can sit in a long S_SLEEP between memory reads, but the wave which writes the value
can tell them all to wake up early now that the data is available. This method is also safe from races since any
waves that miss the ping resume when they complete their S_SLEEP.

If the wave executing S_WAKEUP is in a threadgroup (in_tg set), then it wakes up all waves associated with the
same threadgroup ID. Otherwise, S_WAKEUP is treated as an S_NOP.

#### S_CBRANCH_SCC0  (opcode 4)

If SCC is 0 then jump to a constant offset relative to the current PC.

The literal argument is a signed DWORD offset relative to the PC of the next instruction.

```
  if SCC == 1'0U then
      PC = PC + signext(SIMM16.i16 * 16'4) + 4LL
  else
      PC = PC + 4LL
  endif
```

#### S_CBRANCH_SCC1  (opcode 5)

If SCC is 1 then jump to a constant offset relative to the current PC.

The literal argument is a signed DWORD offset relative to the PC of the next instruction.

```
  if SCC == 1'1U then
      PC = PC + signext(SIMM16.i16 * 16'4) + 4LL
  else
      PC = PC + 4LL
  endif
```

#### S_CBRANCH_VCCZ  (opcode 6)

If VCCZ is 1 then jump to a constant offset relative to the current PC.

The literal argument is a signed DWORD offset relative to the PC of the next instruction.

```
  if VCCZ.u1 == 1'1U then
      PC = PC + signext(SIMM16.i16 * 16'4) + 4LL
  else
      PC = PC + 4LL
  endif
```

#### S_CBRANCH_VCCNZ  (opcode 7)

If VCCZ is 0 then jump to a constant offset relative to the current PC.

The literal argument is a signed DWORD offset relative to the PC of the next instruction.

```
  if VCCZ.u1 == 1'0U then
      PC = PC + signext(SIMM16.i16 * 16'4) + 4LL
  else
      PC = PC + 4LL
  endif
```

#### S_CBRANCH_EXECZ  (opcode 8)

If EXECZ is 1 then jump to a constant offset relative to the current PC.

The literal argument is a signed DWORD offset relative to the PC of the next instruction.

```
  if EXECZ.u1 == 1'1U then
      PC = PC + signext(SIMM16.i16 * 16'4) + 4LL
  else
      PC = PC + 4LL
  endif
```

#### S_CBRANCH_EXECNZ  (opcode 9)

If EXECZ is 0 then jump to a constant offset relative to the current PC.

The literal argument is a signed DWORD offset relative to the PC of the next instruction.

```
  if EXECZ.u1 == 1'0U then
```

```
      PC = PC + signext(SIMM16.i16 * 16'4) + 4LL
  else
      PC = PC + 4LL
  endif
```

#### S_BARRIER  (opcode 10)

Synchronize waves within a threadgroup.

If not all waves of the threadgroup have been created yet, waits for entire group before proceeding. If some
waves in the threadgroup have already terminated, this waits on only the surviving waves. Barriers are legal
inside trap handlers.

Barrier instructions do not wait for any counters to go to zero before issuing. If the barrier is being used to
protect an outstanding memory operation use the appropriate S_WAITCNT instruction before the barrier.

#### S_SETKILL  (opcode 11)

Kill this wave if the least significant bit of the immediate constant is 1.

Used primarily for debugging kill wave host command behavior.

#### S_WAITCNT  (opcode 12)

Wait for the counts of outstanding local data share, vector memory and export instructions to be at or below
the specified levels.

```
  SIMM16[3:0] = vmcount (vector memory operations) lower bits [3:0],
```

```
  SIMM16[6:4] = export/mem-write-data count,
```

```
  SIMM16[11:8] = LGKMcnt (scalar-mem/GDS/LDS count),
```

```
  SIMM16[15:14] = vmcount (vector memory operations) upper bits [5:4].
```

#### S_SETHALT  (opcode 13)

Set or clear the HALT status bit.

Set HALT bit to value of SIMM16[0]; 1 = halt, 0 = clear HALT bit. The halt flag is ignored while PRIV == 1 (inside
trap handlers) but the shader halts after the handler returns if HALT is still set at that time.

#### S_SLEEP  (opcode 14)

Cause a wave to sleep for up to ~8000 clocks.

The wave sleeps for (64*(SIMM16[6:0]-1) .. 64*SIMM16[6:0]) clocks. The exact amount of delay is approximate.
Compare with S_NOP. When SIMM16[6:0] is zero then no sleep occurs.

Notes

Examples:

```
        s_sleep 0         // Wait for 0 clocks.
        s_sleep 1         // Wait for 1-64 clocks.
        s_sleep 2         // Wait for 65-128 clocks.
```

#### S_SETPRIO  (opcode 15)

Change wave user priority.

User settable wave priority is set to SIMM16[1:0]. 0 = lowest, 3 = highest. The overall wave priority is
{SPIPrio[1:0], UserPrio[1:0], WaveAge[3:0]}.

#### S_SENDMSG  (opcode 16)

Send a message upstream to graphics control hardware.

SIMM16[9:0] contains the message type.

Notes

#### S_SENDMSGHALT  (opcode 17)

Send a message to upstream control hardware and then HALT the wavefront; see S_SENDMSG for details.

#### S_TRAP  (opcode 18)

Enter the trap handler.

This instruction may be generated internally as well in response to a host trap (HT = 1) or an exception. TrapID

0 is reserved for hardware use and should not be used in a shader-generated trap.

```
  TrapID = SIMM16.u16[7 : 0];
  "Wait for all instructions to complete";
  // PC passed into trap handler points to S_TRAP itself,
  // *not* to the next instruction.
  { TTMP[1], TTMP[0] } = { 3'0, PCRewind[3 : 0], HT[0], TrapID[7 : 0], PC[47 : 0] };
  PC = TBA.i64;
  // trap base address
  WAVE_STATUS.PRIV = 1'1U
```

#### S_ICACHE_INV  (opcode 19)

Invalidate entire first level instruction cache.

There must be 16 separate S_NOP instructions or a jump/branch instruction after this instruction to ensure the
internal instruction buffers are also invalidated.

#### S_INCPERFLEVEL  (opcode 20)

Increment performance counter specified in SIMM16[3:0] by 1.

#### S_DECPERFLEVEL  (opcode 21)

Decrement performance counter specified in SIMM16[3:0] by 1.

#### S_TTRACEDATA  (opcode 22)

Send M0 as user data to the thread trace stream.

#### S_CBRANCH_CDBGSYS  (opcode 23)

If the system debug flag is set then jump to a constant offset relative to the current PC.

The literal argument is a signed DWORD offset relative to the PC of the next instruction.

```
  if WAVE_STATUS.COND_DBG_SYS.u32 != 0U then
      PC = PC + signext(SIMM16.i16 * 16'4) + 4LL
  else
      PC = PC + 4LL
  endif
```

#### S_CBRANCH_CDBGUSER  (opcode 24)

If the user debug flag is set then jump to a constant offset relative to the current PC.

The literal argument is a signed DWORD offset relative to the PC of the next instruction.

```
  if WAVE_STATUS.COND_DBG_USER.u32 != 0U then
      PC = PC + signext(SIMM16.i16 * 16'4) + 4LL
  else
      PC = PC + 4LL
  endif
```

#### S_CBRANCH_CDBGSYS_OR_USER  (opcode 25)

If either the system debug flag or the user debug flag is set then jump to a constant offset relative to the current
PC.

The literal argument is a signed DWORD offset relative to the PC of the next instruction.

```
  if (WAVE_STATUS.COND_DBG_SYS || WAVE_STATUS.COND_DBG_USER) then
      PC = PC + signext(SIMM16.i16 * 16'4) + 4LL
  else
      PC = PC + 4LL
  endif
```

#### S_CBRANCH_CDBGSYS_AND_USER  (opcode 26)

If both the system debug flag and the user debug flag are set then jump to a constant offset relative to the
current PC.

The literal argument is a signed DWORD offset relative to the PC of the next instruction.

```
  if (WAVE_STATUS.COND_DBG_SYS && WAVE_STATUS.COND_DBG_USER) then
      PC = PC + signext(SIMM16.i16 * 16'4) + 4LL
  else
      PC = PC + 4LL
  endif
```

#### S_ENDPGM_SAVED  (opcode 27)

End of program; signal that a wave has been saved by the context-switch trap handler and terminate
wavefront.

The hardware implicitly executes S_WAITCNT 0 before executing this instruction.

See S_ENDPGM for additional variants.

#### S_SET_GPR_IDX_OFF  (opcode 28)

Clear GPR indexing mode.

Vector operations after this do not perform relative GPR addressing regardless of the contents of M0. This
instruction does not modify M0.

S_SET_GPR_IDX_ON, S_SET_GPR_IDX_OFF, S_SET_GPR_IDX_MODE and S_SET_GPR_IDX_IDX are related
instructions.

```
  WAVE_MODE.GPR_IDX_EN = 1'0U
```

#### S_SET_GPR_IDX_MODE  (opcode 29)

Modify the mode used for vector GPR indexing.

The raw contents of the source field are read and used to set the enable bits. SIMM16[0] = VSRC0_REL,
SIMM16[1] = VSRC1_REL, SIMM16[2] = VSRC2_REL and SIMM16[3] = VDST_REL.

S_SET_GPR_IDX_ON, S_SET_GPR_IDX_OFF, S_SET_GPR_IDX_MODE and S_SET_GPR_IDX_IDX are related
instructions.

```
  M0[15 : 12] = SIMM16.u16[3 : 0].b4
```

#### 12.5.1. Send Message

The S_SENDMSG instruction encodes the message type in M0, and can also send data from the SIMM16 field
and in some cases from EXEC.

```
Message          SIMM16[3:0]      SIMM16[6:4]    Payload
none             0                -              illegal
Interrupt        1                -              M0[23:0] carries data payload
Save wave        4                -              used in context switching
Stall Wave Gen   5                -              stop new wave generation
Halt Waves       6                -              halt all running waves of this vmid
Get Doorbell ID 10                -              Returns doorbell into EXEC, with the doorbell physical address in bits
                                                 [12:3].
```
