# CDNA4 ISA Instructions: Vector (VOP2/VOP1/VOPC)

> Part of the **CDNA4 Instruction Set Architecture** reference. See the [index](README.md) for other sections.

## Sections in this file

  - [12.7. VOP2 Instructions](#127-vop2-instructions)
  - [12.8. VOP1 Instructions](#128-vop1-instructions)
  - [12.9. VOPC Instructions](#129-vopc-instructions)

## Instruction mnemonics in this file

- **12.7. VOP2 Instructions**: V_CNDMASK_B32, V_ADD_F32, V_SUB_F32, V_SUBREV_F32, V_FMAC_F64, V_MUL_F32, V_MUL_I32_I24, V_MUL_HI_I32_I24, V_MUL_U32_U24, V_MUL_HI_U32_U24, V_MIN_F32, V_MAX_F32, V_MIN_I32, V_MAX_I32, V_MIN_U32, V_MAX_U32, V_LSHRREV_B32, V_ASHRREV_I32, V_LSHLREV_B32, V_AND_B32, V_OR_B32, V_XOR_B32, V_DOT2C_F32_BF16, V_FMAMK_F32, V_FMAAK_F32, V_ADD_CO_U32, V_SUB_CO_U32, V_SUBREV_CO_U32, V_ADDC_CO_U32, V_SUBB_CO_U32, V_SUBBREV_CO_U32, V_ADD_F16, V_SUB_F16, V_SUBREV_F16, V_MUL_F16, V_MAC_F16, V_MADMK_F16, V_MADAK_F16, V_ADD_U16, V_SUB_U16, V_SUBREV_U16, V_MUL_LO_U16, V_LSHLREV_B16, V_LSHRREV_B16, V_ASHRREV_I16, V_MAX_F16, V_MIN_F16, V_MAX_U16, V_MAX_I16, V_MIN_U16, V_MIN_I16, V_LDEXP_F16, V_ADD_U32, V_SUB_U32, V_SUBREV_U32, V_DOT2C_F32_F16, V_DOT2C_I32_I16, V_DOT4C_I32_I8, V_DOT8C_I32_I4, V_FMAC_F32, V_PK_FMAC_F16, V_XNOR_B32
- **12.8. VOP1 Instructions**: V_NOP, V_MOV_B32, V_READFIRSTLANE_B32, V_CVT_I32_F64, V_CVT_F64_I32, V_CVT_F32_I32, V_CVT_F32_U32, V_CVT_U32_F32, V_CVT_I32_F32, V_CVT_F16_F32, V_CVT_F32_F16, V_CVT_RPI_I32_F32, V_CVT_FLR_I32_F32, V_CVT_OFF_F32_I4, V_CVT_F32_F64, V_CVT_F64_F32, V_CVT_F32_UBYTE0, V_CVT_F32_UBYTE1, V_CVT_F32_UBYTE2, V_CVT_F32_UBYTE3, V_CVT_U32_F64, V_CVT_F64_U32, V_TRUNC_F64, V_CEIL_F64, V_RNDNE_F64, V_FLOOR_F64, V_FRACT_F32, V_TRUNC_F32, V_CEIL_F32, V_RNDNE_F32, V_FLOOR_F32, V_EXP_F32, V_LOG_F32, V_RCP_F32, V_RCP_IFLAG_F32, V_RSQ_F32, V_RCP_F64, V_RSQ_F64, V_SQRT_F32, V_SQRT_F64, V_SIN_F32, V_COS_F32, V_NOT_B32, V_BFREV_B32, V_FFBH_U32, V_FFBL_B32, V_FFBH_I32, V_FREXP_EXP_I32_F64, V_FREXP_MANT_F64, V_FRACT_F64, V_FREXP_EXP_I32_F32, V_FREXP_MANT_F32, V_CLREXCP, V_MOV_B64, V_CVT_F16_U16, V_CVT_F16_I16, V_CVT_U16_F16, V_CVT_I16_F16, V_RCP_F16, V_SQRT_F16, V_RSQ_F16, V_LOG_F16, V_EXP_F16, V_FREXP_MANT_F16, V_FREXP_EXP_I16_F16, V_FLOOR_F16, V_CEIL_F16, V_TRUNC_F16, V_RNDNE_F16, V_FRACT_F16, V_SIN_F16, V_COS_F16, V_CVT_NORM_I16_F16, V_CVT_NORM_U16_F16, V_SAT_PK_U8_I16, V_SWAP_B32, V_ACCVGPR_MOV_B32, V_CVT_F32_FP8, V_CVT_F32_BF8, V_CVT_PK_F32_FP8, V_CVT_PK_F32_BF8, V_PRNG_B32, V_PERMLANE16_SWAP_B32, V_PERMLANE32_SWAP_B32, V_CVT_F32_BF16
- **12.9. VOPC Instructions**: V_CMP_CLASS_F32, V_CMPX_CLASS_F32, V_CMP_CLASS_F64, V_CMPX_CLASS_F64, V_CMP_CLASS_F16, V_CMPX_CLASS_F16, V_CMP_F_F16, V_CMP_LT_F16, V_CMP_EQ_F16, V_CMP_LE_F16, V_CMP_GT_F16, V_CMP_LG_F16, V_CMP_GE_F16, V_CMP_O_F16, V_CMP_U_F16, V_CMP_NGE_F16, V_CMP_NLG_F16, V_CMP_NGT_F16, V_CMP_NLE_F16, V_CMP_NEQ_F16, V_CMP_NLT_F16, V_CMP_TRU_F16, V_CMPX_F_F16, V_CMPX_LT_F16, V_CMPX_EQ_F16, V_CMPX_LE_F16, V_CMPX_GT_F16, V_CMPX_LG_F16, V_CMPX_GE_F16, V_CMPX_O_F16, V_CMPX_U_F16, V_CMPX_NGE_F16, V_CMPX_NLG_F16, V_CMPX_NGT_F16, V_CMPX_NLE_F16, V_CMPX_NEQ_F16, V_CMPX_NLT_F16, V_CMPX_TRU_F16, V_CMP_F_F32, V_CMP_LT_F32, V_CMP_EQ_F32, V_CMP_LE_F32, V_CMP_GT_F32, V_CMP_LG_F32, V_CMP_GE_F32, V_CMP_O_F32, V_CMP_U_F32, V_CMP_NGE_F32, V_CMP_NLG_F32, V_CMP_NGT_F32, V_CMP_NLE_F32, V_CMP_NEQ_F32, V_CMP_NLT_F32, V_CMP_TRU_F32, V_CMPX_F_F32, V_CMPX_LT_F32, V_CMPX_EQ_F32, V_CMPX_LE_F32, V_CMPX_GT_F32, V_CMPX_LG_F32, V_CMPX_GE_F32, V_CMPX_O_F32, V_CMPX_U_F32, V_CMPX_NGE_F32, V_CMPX_NLG_F32, V_CMPX_NGT_F32, V_CMPX_NLE_F32, V_CMPX_NEQ_F32, V_CMPX_NLT_F32, V_CMPX_TRU_F32, V_CMP_F_F64, V_CMP_LT_F64, V_CMP_EQ_F64, V_CMP_LE_F64, V_CMP_GT_F64, V_CMP_LG_F64, V_CMP_GE_F64, V_CMP_O_F64, V_CMP_U_F64, V_CMP_NGE_F64, V_CMP_NLG_F64, V_CMP_NGT_F64, V_CMP_NLE_F64, V_CMP_NEQ_F64, V_CMP_NLT_F64, V_CMP_TRU_F64, V_CMPX_F_F64, V_CMPX_LT_F64, V_CMPX_EQ_F64, V_CMPX_LE_F64, V_CMPX_GT_F64, V_CMPX_LG_F64, V_CMPX_GE_F64, V_CMPX_O_F64, V_CMPX_U_F64, V_CMPX_NGE_F64, V_CMPX_NLG_F64, V_CMPX_NGT_F64, V_CMPX_NLE_F64, V_CMPX_NEQ_F64, V_CMPX_NLT_F64, V_CMPX_TRU_F64, V_CMP_F_I16, V_CMP_LT_I16, V_CMP_EQ_I16, V_CMP_LE_I16, V_CMP_GT_I16, V_CMP_NE_I16, V_CMP_GE_I16, V_CMP_T_I16, V_CMP_F_U16, V_CMP_LT_U16, V_CMP_EQ_U16, V_CMP_LE_U16, V_CMP_GT_U16, V_CMP_NE_U16, V_CMP_GE_U16, V_CMP_T_U16, V_CMPX_F_I16, V_CMPX_LT_I16, V_CMPX_EQ_I16, V_CMPX_LE_I16, V_CMPX_GT_I16, V_CMPX_NE_I16, V_CMPX_GE_I16, V_CMPX_T_I16, V_CMPX_F_U16, V_CMPX_LT_U16, V_CMPX_EQ_U16, V_CMPX_LE_U16, V_CMPX_GT_U16, V_CMPX_NE_U16, V_CMPX_GE_U16, V_CMPX_T_U16, V_CMP_F_I32, V_CMP_LT_I32, V_CMP_EQ_I32, V_CMP_LE_I32, V_CMP_GT_I32, V_CMP_NE_I32, V_CMP_GE_I32, V_CMP_T_I32, V_CMP_F_U32, V_CMP_LT_U32, V_CMP_EQ_U32, V_CMP_LE_U32, V_CMP_GT_U32, V_CMP_NE_U32, V_CMP_GE_U32, V_CMP_T_U32, V_CMPX_F_I32, V_CMPX_LT_I32, V_CMPX_EQ_I32, V_CMPX_LE_I32, V_CMPX_GT_I32, V_CMPX_NE_I32, V_CMPX_GE_I32, V_CMPX_T_I32, V_CMPX_F_U32, V_CMPX_LT_U32, V_CMPX_EQ_U32, V_CMPX_LE_U32, V_CMPX_GT_U32, V_CMPX_NE_U32, V_CMPX_GE_U32, V_CMPX_T_U32, V_CMP_F_I64, V_CMP_LT_I64, V_CMP_EQ_I64, V_CMP_LE_I64, V_CMP_GT_I64, V_CMP_NE_I64, V_CMP_GE_I64, V_CMP_T_I64, V_CMP_F_U64, V_CMP_LT_U64, V_CMP_EQ_U64, V_CMP_LE_U64, V_CMP_GT_U64, V_CMP_NE_U64, V_CMP_GE_U64, V_CMP_T_U64, V_CMPX_F_I64, V_CMPX_LT_I64, V_CMPX_EQ_I64, V_CMPX_LE_I64, V_CMPX_GT_I64, V_CMPX_NE_I64, V_CMPX_GE_I64, V_CMPX_T_I64, V_CMPX_F_U64, V_CMPX_LT_U64, V_CMPX_EQ_U64, V_CMPX_LE_U64, V_CMPX_GT_U64, V_CMPX_NE_U64, V_CMPX_GE_U64, V_CMPX_T_U64

---

### 12.7. VOP2 Instructions

Instructions in this format may use a 32-bit literal constant, DPP or SDWA which occurs immediately after the
instruction.

#### V_CNDMASK_B32  (opcode 0)

Copy data from one of two inputs based on the per-lane condition code and store the result into a vector
register.

```
  D0.u32 = VCC.u64[laneId] ? S1.u32 : S0.u32
```

Notes

In VOP3 the VCC source may be a scalar GPR specified in S2.

Floating-point modifiers are valid for this instruction if S0 and S1 are 32-bit floating point values. This
instruction is suitable for negating or taking the absolute value of a floating-point value.

#### V_ADD_F32  (opcode 1)

Add two floating point inputs and store the result into a vector register.

```
  D0.f32 = S0.f32 + S1.f32
```

Notes

0.5ULP precision, denormals are supported.

#### V_SUB_F32  (opcode 2)

Subtract the second floating point input from the first input and store the result into a vector register.

```
  D0.f32 = S0.f32 - S1.f32
```

Notes

0.5ULP precision, denormals are supported.

#### V_SUBREV_F32  (opcode 3)

Subtract the first floating point input from the second input and store the result into a vector register.

```
  D0.f32 = S1.f32 - S0.f32
```

Notes

0.5ULP precision, denormals are supported.

#### V_FMAC_F64  (opcode 4)

Multiply two floating point inputs and accumulate the result into the destination register using fused multiply
add.

```
  D0.f64 = fma(S0.f64, S1.f64, D0.f64)
```

#### V_MUL_F32  (opcode 5)

Multiply two floating point inputs and store the result into a vector register.

```
  D0.f32 = S0.f32 * S1.f32
```

Notes

0.5ULP precision, denormals are supported.

#### V_MUL_I32_I24  (opcode 6)

Multiply two signed 24-bit integer inputs and store the result as a signed 32-bit integer into a vector register.

```
  D0.i32 = 32'I(S0.i24) * 32'I(S1.i24)
```

Notes

This opcode is expected to be as efficient as basic single-precision opcodes since it utilizes the single-precision
floating point multiplier. See also V_MUL_HI_I32_I24.

#### V_MUL_HI_I32_I24  (opcode 7)

Multiply two signed 24-bit integer inputs and store the high 32 bits of the result as a signed 32-bit integer into a
vector register.

```
  D0.i32 = 32'I((64'I(S0.i24) * 64'I(S1.i24)) >> 32U)
```

Notes

See also V_MUL_I32_I24.

#### V_MUL_U32_U24  (opcode 8)

Multiply two unsigned 24-bit integer inputs and store the result as an unsigned 32-bit integer into a vector
register.

```
  D0.u32 = 32'U(S0.u24) * 32'U(S1.u24)
```

Notes

This opcode is expected to be as efficient as basic single-precision opcodes since it utilizes the single-precision
floating point multiplier. See also V_MUL_HI_U32_U24.

#### V_MUL_HI_U32_U24  (opcode 9)

Multiply two unsigned 24-bit integer inputs and store the high 32 bits of the result as an unsigned 32-bit integer
into a vector register.

```
  D0.u32 = 32'U((64'U(S0.u24) * 64'U(S1.u24)) >> 32U)
```

Notes

See also V_MUL_U32_U24.

#### V_MIN_F32  (opcode 10)

Select the minimum of two single-precision float inputs and store the result into a vector register.

```
  if (WAVE_MODE.IEEE && isSignalNAN(64'F(S0.f32))) then
        D0.f32 = 32'F(cvtToQuietNAN(64'F(S0.f32)))
  elsif (WAVE_MODE.IEEE && isSignalNAN(64'F(S1.f32))) then
        D0.f32 = 32'F(cvtToQuietNAN(64'F(S1.f32)))
  elsif isNAN(64'F(S0.f32)) then
        D0.f32 = S1.f32
```

```
  elsif isNAN(64'F(S1.f32)) then
      D0.f32 = S0.f32
  elsif ((64'F(S0.f32) == +0.0) && (64'F(S1.f32) == -0.0)) then
      D0.f32 = S1.f32
  elsif ((64'F(S0.f32) == -0.0) && (64'F(S1.f32) == +0.0)) then
      D0.f32 = S0.f32
  else
      // Note: there's no IEEE case here like there is for V_MAX_F32.
      D0.f32 = S0.f32 < S1.f32 ? S0.f32 : S1.f32
  endif
```

#### V_MAX_F32  (opcode 11)

Select the maximum of two single-precision float inputs and store the result into a vector register.

```
  if (WAVE_MODE.IEEE && isSignalNAN(64'F(S0.f32))) then
      D0.f32 = 32'F(cvtToQuietNAN(64'F(S0.f32)))
  elsif (WAVE_MODE.IEEE && isSignalNAN(64'F(S1.f32))) then
      D0.f32 = 32'F(cvtToQuietNAN(64'F(S1.f32)))
  elsif isNAN(64'F(S0.f32)) then
      D0.f32 = S1.f32
  elsif isNAN(64'F(S1.f32)) then
      D0.f32 = S0.f32
  elsif ((64'F(S0.f32) == +0.0) && (64'F(S1.f32) == -0.0)) then
      D0.f32 = S0.f32
  elsif ((64'F(S0.f32) == -0.0) && (64'F(S1.f32) == +0.0)) then
      D0.f32 = S1.f32
  elsif WAVE_MODE.IEEE then
      D0.f32 = S0.f32 >= S1.f32 ? S0.f32 : S1.f32
  else
      D0.f32 = S0.f32 > S1.f32 ? S0.f32 : S1.f32
  endif
```

#### V_MIN_I32  (opcode 12)

Select the minimum of two signed 32-bit integer inputs and store the selected value into a vector register.

```
  D0.i32 = S0.i32 < S1.i32 ? S0.i32 : S1.i32
```

#### V_MAX_I32  (opcode 13)

Select the maximum of two signed 32-bit integer inputs and store the selected value into a vector register.

```
  D0.i32 = S0.i32 >= S1.i32 ? S0.i32 : S1.i32
```

#### V_MIN_U32  (opcode 14)

Select the minimum of two unsigned 32-bit integer inputs and store the selected value into a vector register.

```
  D0.u32 = S0.u32 < S1.u32 ? S0.u32 : S1.u32
```

#### V_MAX_U32  (opcode 15)

Select the maximum of two unsigned 32-bit integer inputs and store the selected value into a vector register.

```
  D0.u32 = S0.u32 >= S1.u32 ? S0.u32 : S1.u32
```

#### V_LSHRREV_B32  (opcode 16)

Given a shift count in the first vector input, calculate the logical shift right of the second vector input and store
the result into a vector register.

```
  D0.u32 = (S1.u32 >> S0[4 : 0].u32)
```

#### V_ASHRREV_I32  (opcode 17)

Given a shift count in the first vector input, calculate the arithmetic shift right (preserving sign bit) of the second
vector input and store the result into a vector register.

```
  D0.i32 = (S1.i32 >> S0[4 : 0].u32)
```

#### V_LSHLREV_B32  (opcode 18)

Given a shift count in the first vector input, calculate the logical shift left of the second vector input and store the
result into a vector register.

```
  D0.u32 = (S1.u32 << S0[4 : 0].u32)
```

#### V_AND_B32  (opcode 19)

Calculate bitwise AND on two vector inputs and store the result into a vector register.

```
  D0.u32 = (S0.u32 & S1.u32)
```

Notes

Input and output modifiers not supported.

#### V_OR_B32  (opcode 20)

Calculate bitwise OR on two vector inputs and store the result into a vector register.

```
  D0.u32 = (S0.u32 | S1.u32)
```

Notes

Input and output modifiers not supported.

#### V_XOR_B32  (opcode 21)

Calculate bitwise XOR on two vector inputs and store the result into a vector register.

```
  D0.u32 = (S0.u32 ^ S1.u32)
```

Notes

Input and output modifiers not supported.

#### V_DOT2C_F32_BF16  (opcode 22)

Compute the dot product of two packed 2-D BF16 float inputs in the single-precision float domain and
accumulate with the single-precision float value in the destination register.

```
  tmp = D0.f32;
  tmp += bf16_to_f32(S0[15 : 0].bf16) * bf16_to_f32(S1[15 : 0].bf16);
  tmp += bf16_to_f32(S0[31 : 16].bf16) * bf16_to_f32(S1[31 : 16].bf16);
  D0.f32 = tmp
```

Notes

ABS[1:0] are used as NEG_HI[1:0] during translation.

NEG and ABS input modifiers do not affect S2.

#### V_FMAMK_F32  (opcode 23)

Multiply a single-precision float input with a literal constant and add a second single-precision float input using
fused multiply add, and store the result into a vector register.

```
  D0.f32 = fma(S0.f32, SIMM32.f32, S1.f32)
```

Notes

This opcode cannot use the VOP3 encoding and cannot use input/output modifiers.

#### V_FMAAK_F32  (opcode 24)

Multiply two single-precision float inputs and add a literal constant using fused multiply add, and store the
result into a vector register.

```
  D0.f32 = fma(S0.f32, S1.f32, SIMM32.f32)
```

Notes

This opcode cannot use the VOP3 encoding and cannot use input/output modifiers.

#### V_ADD_CO_U32  (opcode 25)

Add two unsigned 32-bit integer inputs, store the result into a vector register and store the carry-out mask into
a scalar register.

```
  tmp = 64'U(S0.u32) + 64'U(S1.u32);
  VCC.u64[laneId] = tmp >= 0x100000000ULL ? 1'1U : 1'0U;
  // VCC is an UNSIGNED overflow/carry-out for V_ADDC_CO_U32.
  D0.u32 = tmp.u32
```

Notes

In VOP3 the VCC destination may be an arbitrary SGPR-pair.

Supports saturation (unsigned 32-bit integer domain).

#### V_SUB_CO_U32  (opcode 26)

Subtract the second unsigned 32-bit integer input from the first input, store the result into a vector register and
store the carry-out mask into a scalar register.

```
  tmp = S0.u32 - S1.u32;
  VCC.u64[laneId] = S1.u32 > S0.u32 ? 1'1U : 1'0U;
  // VCC is an UNSIGNED overflow/carry-out for V_SUBB_CO_U32.
  D0.u32 = tmp.u32
```

Notes

In VOP3 the VCC destination may be an arbitrary SGPR-pair.

Supports saturation (unsigned 32-bit integer domain).

#### V_SUBREV_CO_U32  (opcode 27)

Subtract the first unsigned 32-bit integer input from the second input, store the result into a vector register and
store the carry-out mask into a scalar register.

```
  tmp = S1.u32 - S0.u32;
  VCC.u64[laneId] = S0.u32 > S1.u32 ? 1'1U : 1'0U;
  // VCC is an UNSIGNED overflow/carry-out for V_SUBB_CO_U32.
  D0.u32 = tmp.u32
```

Notes

In VOP3 the VCC destination may be an arbitrary SGPR-pair.

Supports saturation (unsigned 32-bit integer domain).

#### V_ADDC_CO_U32  (opcode 28)

Add two unsigned 32-bit integer inputs and a bit from a carry-in mask, store the result into a vector register and
store the carry-out mask into a scalar register.

```
  tmp = 64'U(S0.u32) + 64'U(S1.u32) + VCC.u64[laneId].u64;
  VCC.u64[laneId] = tmp >= 0x100000000ULL ? 1'1U : 1'0U;
  // VCC is an UNSIGNED overflow/carry-out for V_ADDC_CO_U32.
  D0.u32 = tmp.u32
```

Notes

In VOP3 the VCC destination may be an arbitrary SGPR-pair, and the VCC source comes from the SGPR-pair at

S2.u.

Supports saturation (unsigned 32-bit integer domain).

#### V_SUBB_CO_U32  (opcode 29)

Subtract the second unsigned 32-bit integer input from the first input, subtract a bit from the carry-in mask,
store the result into a vector register and store the carry-out mask into a scalar register.

```
  tmp = S0.u32 - S1.u32 - VCC.u64[laneId].u32;
  VCC.u64[laneId] = 64'U(S1.u32) + VCC.u64[laneId].u64 > 64'U(S0.u32) ? 1'1U : 1'0U;
  // VCC is an UNSIGNED overflow/carry-out for V_SUBB_CO_U32.
  D0.u32 = tmp.u32
```

Notes

In VOP3 the VCC destination may be an arbitrary SGPR-pair, and the VCC source comes from the SGPR-pair at
S2.u.

Supports saturation (unsigned 32-bit integer domain).

#### V_SUBBREV_CO_U32  (opcode 30)

Subtract the first unsigned 32-bit integer input from the second input, subtract a bit from the carry-in mask,
store the result into a vector register and store the carry-out mask into a scalar register.

```
  tmp = S1.u32 - S0.u32 - VCC.u64[laneId].u32;
  VCC.u64[laneId] = 64'U(S0.u32) + VCC.u64[laneId].u64 > 64'U(S1.u32) ? 1'1U : 1'0U;
  // VCC is an UNSIGNED overflow/carry-out for V_SUBB_CO_U32.
  D0.u32 = tmp.u32
```

Notes

In VOP3 the VCC destination may be an arbitrary SGPR-pair, and the VCC source comes from the SGPR-pair at
S2.u.

Supports saturation (unsigned 32-bit integer domain).

#### V_ADD_F16  (opcode 31)

Add two floating point inputs and store the result into a vector register.

```
  D0.f16 = S0.f16 + S1.f16
```

Notes

0.5ULP precision. Supports denormals, round mode, exception flags and saturation.

#### V_SUB_F16  (opcode 32)

Subtract the second floating point input from the first input and store the result into a vector register.

```
  D0.f16 = S0.f16 - S1.f16
```

Notes

0.5ULP precision. Supports denormals, round mode, exception flags and saturation.

#### V_SUBREV_F16  (opcode 33)

Subtract the first floating point input from the second input and store the result into a vector register.

```
  D0.f16 = S1.f16 - S0.f16
```

Notes

0.5ULP precision. Supports denormals, round mode, exception flags and saturation.

#### V_MUL_F16  (opcode 34)

Multiply two floating point inputs and store the result into a vector register.

```
  D0.f16 = S0.f16 * S1.f16
```

Notes

0.5ULP precision. Supports denormals, round mode, exception flags and saturation.

#### V_MAC_F16  (opcode 35)

Multiply two floating point inputs and accumulate the result into the destination register. Implements IEEE
rules and non-standard rule for OPSEL.

```
  tmp = S0.f16 * S1.f16 + D0.f16;
```

```
  if OPSEL.u4[3] then
        D0 = { tmp.f16, D0[15 : 0] }
  else
        D0 = { 16'0, tmp.f16 }
  endif
```

Notes

Supports round mode, exception flags, saturation.

#### V_MADMK_F16  (opcode 36)

Multiply a floating point input with a literal constant and add a second floating point input, and store the result
into a vector register. Implements IEEE rules.

```
  tmp = S0.f16 * SIMM16.f16 + S1.f16;
  D0 = { 16'0, tmp.f16 }
```

Notes

This opcode cannot use the VOP3 encoding and cannot use input/output modifiers. Supports round mode,
exception flags, saturation.

#### V_MADAK_F16  (opcode 37)

Multiply two floating point inputs and add a literal constant, and store the result into a vector register.
Implements IEEE rules.

```
  tmp = S0.f16 * S1.f16 + SIMM16.f16;
  D0 = { 16'0, tmp.f16 }
```

Notes

This opcode cannot use the VOP3 encoding and cannot use input/output modifiers. Supports round mode,
exception flags, saturation.

#### V_ADD_U16  (opcode 38)

Add two unsigned 16-bit integer inputs and store the result into a vector register. No carry-in or carry-out
support.

```
  D0.u16 = S0.u16 + S1.u16
```

Notes

Supports saturation (unsigned 16-bit integer domain).

#### V_SUB_U16  (opcode 39)

Subtract the second unsigned 16-bit integer input from the first input and store the result into a vector register.
No carry-in or carry-out support.

```
  D0.u16 = S0.u16 - S1.u16
```

Notes

Supports saturation (unsigned 16-bit integer domain).

#### V_SUBREV_U16  (opcode 40)

Subtract the first unsigned 16-bit integer input from the second input and store the result into a vector register.
No carry-in or carry-out support.

```
  D0.u16 = S1.u16 - S0.u16
```

Notes

Supports saturation (unsigned 16-bit integer domain).

#### V_MUL_LO_U16  (opcode 41)

Multiply two unsigned 16-bit integer inputs and store the low bits of the result into a vector register.

```
  D0.u16 = S0.u16 * S1.u16
```

Notes

Supports saturation (unsigned 16-bit integer domain).

#### V_LSHLREV_B16  (opcode 42)

Given a shift count in the first vector input, calculate the logical shift left of the second vector input and store the
result into a vector register.

```
  D0.u16 = (S1.u16 << S0[3 : 0].u32)
```

#### V_LSHRREV_B16  (opcode 43)

Given a shift count in the first vector input, calculate the logical shift right of the second vector input and store
the result into a vector register.

```
  D0.u16 = (S1.u16 >> S0[3 : 0].u32)
```

#### V_ASHRREV_I16  (opcode 44)

Given a shift count in the first vector input, calculate the arithmetic shift right (preserving sign bit) of the second
vector input and store the result into a vector register.

```
  D0.i16 = (S1.i16 >> S0[3 : 0].u32)
```

#### V_MAX_F16  (opcode 45)

Select the maximum of two half-precision float inputs and store the result into a vector register.

```
  if (WAVE_MODE.IEEE && isSignalNAN(64'F(S0.f16))) then
        D0.f16 = 16'F(cvtToQuietNAN(64'F(S0.f16)))
  elsif (WAVE_MODE.IEEE && isSignalNAN(64'F(S1.f16))) then
        D0.f16 = 16'F(cvtToQuietNAN(64'F(S1.f16)))
  elsif isNAN(64'F(S0.f16)) then
        D0.f16 = S1.f16
  elsif isNAN(64'F(S1.f16)) then
        D0.f16 = S0.f16
  elsif ((64'F(S0.f16) == +0.0) && (64'F(S1.f16) == -0.0)) then
        D0.f16 = S0.f16
  elsif ((64'F(S0.f16) == -0.0) && (64'F(S1.f16) == +0.0)) then
        D0.f16 = S1.f16
  elsif WAVE_MODE.IEEE then
        D0.f16 = S0.f16 >= S1.f16 ? S0.f16 : S1.f16
  else
        D0.f16 = S0.f16 > S1.f16 ? S0.f16 : S1.f16
  endif
```

Notes

IEEE compliant. Supports denormals, round mode, exception flags, saturation.

#### V_MIN_F16  (opcode 46)

Select the minimum of two half-precision float inputs and store the result into a vector register.

```
  if (WAVE_MODE.IEEE && isSignalNAN(64'F(S0.f16))) then
        D0.f16 = 16'F(cvtToQuietNAN(64'F(S0.f16)))
  elsif (WAVE_MODE.IEEE && isSignalNAN(64'F(S1.f16))) then
        D0.f16 = 16'F(cvtToQuietNAN(64'F(S1.f16)))
  elsif isNAN(64'F(S0.f16)) then
        D0.f16 = S1.f16
  elsif isNAN(64'F(S1.f16)) then
        D0.f16 = S0.f16
  elsif ((64'F(S0.f16) == +0.0) && (64'F(S1.f16) == -0.0)) then
        D0.f16 = S1.f16
  elsif ((64'F(S0.f16) == -0.0) && (64'F(S1.f16) == +0.0)) then
        D0.f16 = S0.f16
  else
        // Note: there's no IEEE case here like there is for V_MAX_F16.
        D0.f16 = S0.f16 < S1.f16 ? S0.f16 : S1.f16
  endif
```

Notes

IEEE compliant. Supports denormals, round mode, exception flags, saturation.

#### V_MAX_U16  (opcode 47)

Select the maximum of two unsigned 16-bit integer inputs and store the selected value into a vector register.

```
  D0.u16 = S0.u16 >= S1.u16 ? S0.u16 : S1.u16
```

#### V_MAX_I16  (opcode 48)

Select the maximum of two signed 16-bit integer inputs and store the selected value into a vector register.

```
  D0.i16 = S0.i16 >= S1.i16 ? S0.i16 : S1.i16
```

#### V_MIN_U16  (opcode 49)

Select the minimum of two unsigned 16-bit integer inputs and store the selected value into a vector register.

```
  D0.u16 = S0.u16 < S1.u16 ? S0.u16 : S1.u16
```

#### V_MIN_I16  (opcode 50)

Select the minimum of two signed 16-bit integer inputs and store the selected value into a vector register.

```
  D0.i16 = S0.i16 < S1.i16 ? S0.i16 : S1.i16
```

#### V_LDEXP_F16  (opcode 51)

Multiply the first input, a floating point value, by an integral power of 2 specified in the second input, a signed
integer value, and store the floating point result into a vector register.

```
  D0.f16 = S0.f16 * 16'F(2.0F ** 32'I(S1.i16))
```

Notes

Compare with the ldexp() function in C. Note that the S1 has a format of f16 since floating point literal
constants are interpreted as 16 bit value for this opcode.

#### V_ADD_U32  (opcode 52)

Add two unsigned 32-bit integer inputs and store the result into a vector register. No carry-in or carry-out
support.

```
  D0.u32 = S0.u32 + S1.u32
```

Notes

Supports saturation (unsigned 32-bit integer domain).

#### V_SUB_U32  (opcode 53)

Subtract the second unsigned 32-bit integer input from the first input and store the result into a vector register.
No carry-in or carry-out support.

```
  D0.u32 = S0.u32 - S1.u32
```

Notes

Supports saturation (unsigned 32-bit integer domain).

#### V_SUBREV_U32  (opcode 54)

Subtract the first unsigned 32-bit integer input from the second input and store the result into a vector register.
No carry-in or carry-out support.

```
  D0.u32 = S1.u32 - S0.u32
```

Notes

Supports saturation (unsigned 32-bit integer domain).

#### V_DOT2C_F32_F16  (opcode 55)

Compute the dot product of two packed 2-D half-precision float inputs in the single-precision float domain and
accumulate with the single-precision float value in the destination register.

```
  tmp = D0.f32;
  tmp += f16_to_f32(S0[15 : 0].f16) * f16_to_f32(S1[15 : 0].f16);
  tmp += f16_to_f32(S0[31 : 16].f16) * f16_to_f32(S1[31 : 16].f16);
  D0.f32 = tmp
```

#### V_DOT2C_I32_I16  (opcode 56)

Compute the dot product of two packed 2-D signed 16-bit integer inputs in the signed 32-bit integer domain and
accumulate with the signed 32-bit integer value in the destination register.

```
  tmp = D0.i32;
  tmp += i16_to_i32(S0[15 : 0].i16) * i16_to_i32(S1[15 : 0].i16);
  tmp += i16_to_i32(S0[31 : 16].i16) * i16_to_i32(S1[31 : 16].i16);
  D0.i32 = tmp
```

#### V_DOT4C_I32_I8  (opcode 57)

Compute the dot product of two packed 4-D signed 8-bit integer inputs in the signed 32-bit integer domain and
accumulate with the signed 32-bit integer value in the destination register.

```
  tmp = D0.i32;
  tmp += i8_to_i32(S0[7 : 0].i8) * i8_to_i32(S1[7 : 0].i8);
  tmp += i8_to_i32(S0[15 : 8].i8) * i8_to_i32(S1[15 : 8].i8);
  tmp += i8_to_i32(S0[23 : 16].i8) * i8_to_i32(S1[23 : 16].i8);
  tmp += i8_to_i32(S0[31 : 24].i8) * i8_to_i32(S1[31 : 24].i8);
  D0.i32 = tmp
```

#### V_DOT8C_I32_I4  (opcode 58)

Compute the dot product of two packed 8-D signed 4-bit integer inputs in the signed 32-bit integer domain and
accumulate with the signed 32-bit integer value in the destination register.

```
  tmp = D0.i32;
  tmp += i4_to_i32(S0[3 : 0].i4) * i4_to_i32(S1[3 : 0].i4);
  tmp += i4_to_i32(S0[7 : 4].i4) * i4_to_i32(S1[7 : 4].i4);
  tmp += i4_to_i32(S0[11 : 8].i4) * i4_to_i32(S1[11 : 8].i4);
  tmp += i4_to_i32(S0[15 : 12].i4) * i4_to_i32(S1[15 : 12].i4);
  tmp += i4_to_i32(S0[19 : 16].i4) * i4_to_i32(S1[19 : 16].i4);
  tmp += i4_to_i32(S0[23 : 20].i4) * i4_to_i32(S1[23 : 20].i4);
  tmp += i4_to_i32(S0[27 : 24].i4) * i4_to_i32(S1[27 : 24].i4);
  tmp += i4_to_i32(S0[31 : 28].i4) * i4_to_i32(S1[31 : 28].i4);
  D0.i32 = tmp
```

#### V_FMAC_F32  (opcode 59)

Multiply two floating point inputs and accumulate the result into the destination register using fused multiply
add.

```
  D0.f32 = fma(S0.f32, S1.f32, D0.f32)
```

#### V_PK_FMAC_F16  (opcode 60)

Multiply two packed half-precision float inputs component-wise and accumulate the result into the destination
register using fused multiply add.

```
  D0[15 : 0].f16 = fma(S0[15 : 0].f16, S1[15 : 0].f16, D0[15 : 0].f16);
  D0[31 : 16].f16 = fma(S0[31 : 16].f16, S1[31 : 16].f16, D0[31 : 16].f16)
```

#### V_XNOR_B32  (opcode 61)

Calculate bitwise XNOR on two vector inputs and store the result into a vector register.

```
  D0.u32 = ~(S0.u32 ^ S1.u32)
```

Notes

Input and output modifiers not supported.

#### 12.7.1. VOP2 using VOP3 encoding

Instructions in this format may also be encoded as VOP3. This allows access to the extra control bits (e.g. ABS,
OMOD) in exchange for not being able to use a literal constant. The VOP3 opcode is: VOP2 opcode + 0x100.

### 12.8. VOP1 Instructions

Instructions in this format may use a 32-bit literal constant, DPP or SDWA which occurs immediately after the
instruction.

#### V_NOP  (opcode 0)

Do nothing.

Notes

This instruction can be used to insert a single-cycle bubble in the vector ALU pipeline. For multiple cycles
repeat this opcode.

#### V_MOV_B32  (opcode 1)

Move 32-bit data from a vector input into a vector register.

```
  D0.b32 = S0.b32
```

Notes

Floating-point modifiers are valid for this instruction if S0 is a 32-bit floating point value. This instruction is
suitable for negating or taking the absolute value of a floating-point value.

Functional examples:

```
        v_mov_b32 v0, v1    // Move into v0 from v1
        v_mov_b32 v0, -v1   // Set v0 to the negation of v1
```

```
        v_mov_b32 v0, abs(v1)   // Set v0 to the absolute value of v1
```

#### V_READFIRSTLANE_B32  (opcode 2)

Read the scalar value in the lowest active lane of the input vector register and store it into a scalar register.

```
  declare lane : 32'I;
  if EXEC == 0x0LL then
        lane = 0;
        // Force lane 0 if all lanes are disabled
  else
        lane = s_ff1_i32_b64(EXEC);
        // Lowest active lane
  endif;
  D0.b32 = VGPR[lane][SRC0.u32]
```

Notes

Overrides EXEC mask for the VGPR read. Input and output modifiers not supported; this is an untyped
operation.

#### V_CVT_I32_F64  (opcode 3)

Convert from a double-precision float input to a signed 32-bit integer value and store the result into a vector
register.

```
  D0.i32 = f64_to_i32(S0.f64)
```

Notes

0.5ULP accuracy, out-of-range floating point values (including infinity) saturate. NAN is converted to 0.

Generation of the INEXACT exception is controlled by the CLAMP bit. INEXACT exceptions are enabled for this
conversion iff CLAMP == 1.

#### V_CVT_F64_I32  (opcode 4)

Convert from a signed 32-bit integer input to a double-precision float value and store the result into a vector
register.

```
  D0.f64 = i32_to_f64(S0.i32)
```

Notes

0ULP accuracy.

#### V_CVT_F32_I32  (opcode 5)

Convert from a signed 32-bit integer input to a single-precision float value and store the result into a vector
register.

```
  D0.f32 = i32_to_f32(S0.i32)
```

Notes

0.5ULP accuracy.

#### V_CVT_F32_U32  (opcode 6)

Convert from an unsigned 32-bit integer input to a single-precision float value and store the result into a vector
register.

```
  D0.f32 = u32_to_f32(S0.u32)
```

Notes

0.5ULP accuracy.

#### V_CVT_U32_F32  (opcode 7)

Convert from a single-precision float input to an unsigned 32-bit integer value and store the result into a vector
register.

```
  D0.u32 = f32_to_u32(S0.f32)
```

Notes

1ULP accuracy, out-of-range floating point values (including infinity) saturate. NAN is converted to 0.

Generation of the INEXACT exception is controlled by the CLAMP bit. INEXACT exceptions are enabled for this
conversion iff CLAMP == 1.

#### V_CVT_I32_F32  (opcode 8)

Convert from a single-precision float input to a signed 32-bit integer value and store the result into a vector
register.

```
  D0.i32 = f32_to_i32(S0.f32)
```

Notes

1ULP accuracy, out-of-range floating point values (including infinity) saturate. NAN is converted to 0.

Generation of the INEXACT exception is controlled by the CLAMP bit. INEXACT exceptions are enabled for this
conversion iff CLAMP == 1.

#### V_CVT_F16_F32  (opcode 10)

Convert from a single-precision float input to a half-precision float value and store the result into a vector
register.

```
  D0.f16 = f32_to_f16(S0.f32)
```

Notes

0.5ULP accuracy, supports input modifiers and creates FP16 denormals when appropriate. Flush denorms on
output if specified based on DP denorm mode. Output rounding based on DP rounding mode.

#### V_CVT_F32_F16  (opcode 11)

Convert from a half-precision float input to a single-precision float value and store the result into a vector
register.

```
  D0.f32 = f16_to_f32(S0.f16)
```

Notes

0ULP accuracy, FP16 denormal inputs are accepted. Flush denorms on input if specified based on DP denorm
mode.

#### V_CVT_RPI_I32_F32  (opcode 12)

Convert from a single-precision float input to a signed 32-bit integer value using round to nearest integer
semantics (ignore the default rounding mode) and store the result into a vector register.

```
  D0.i32 = f32_to_i32(floor(S0.f32 + 0.5F))
```

Notes

0.5ULP accuracy, denormals are supported.

#### V_CVT_FLR_I32_F32  (opcode 13)

Convert from a single-precision float input to a signed 32-bit integer value using round-down semantics (ignore
the default rounding mode) and store the result into a vector register.

```
  D0.i32 = f32_to_i32(floor(S0.f32))
```

Notes

1ULP accuracy, denormals are supported.

#### V_CVT_OFF_F32_I4  (opcode 14)

Convert from a signed 4-bit integer input to a single-precision float value using an offset table and store the
result into a vector register.

Used for interpolation in shader. Lookup table on S0[3:0]:

S0 binary Result
1000 -0.5000f
1001 -0.4375f
1010 -0.3750f
1011 -0.3125f
1100 -0.2500f
1101 -0.1875f
1110 -0.1250f
1111 -0.0625f
0000 +0.0000f
0001 +0.0625f
0010 +0.1250f
0011 +0.1875f
0100 +0.2500f
0101 +0.3125f
0110 +0.3750f
0111 +0.4375f

```
  declare CVT_OFF_TABLE : 32'F[16];
  D0.f32 = CVT_OFF_TABLE[S0.u32[3 : 0]]
```

#### V_CVT_F32_F64  (opcode 15)

Convert from a double-precision float input to a single-precision float value and store the result into a vector
register.

```
  D0.f32 = f64_to_f32(S0.f64)
```

Notes

0.5ULP accuracy, denormals are supported.

#### V_CVT_F64_F32  (opcode 16)

Convert from a single-precision float input to a double-precision float value and store the result into a vector
register.

```
  D0.f64 = f32_to_f64(S0.f32)
```

Notes

0ULP accuracy, denormals are supported.

#### V_CVT_F32_UBYTE0  (opcode 17)

Convert an unsigned byte in byte 0 of the input to a single-precision float value and store the result into a vector
register.

```
  D0.f32 = u32_to_f32(S0[7 : 0].u32)
```

#### V_CVT_F32_UBYTE1  (opcode 18)

Convert an unsigned byte in byte 1 of the input to a single-precision float value and store the result into a vector
register.

```
  D0.f32 = u32_to_f32(S0[15 : 8].u32)
```

#### V_CVT_F32_UBYTE2  (opcode 19)

Convert an unsigned byte in byte 2 of the input to a single-precision float value and store the result into a vector
register.

```
  D0.f32 = u32_to_f32(S0[23 : 16].u32)
```

#### V_CVT_F32_UBYTE3  (opcode 20)

Convert an unsigned byte in byte 3 of the input to a single-precision float value and store the result into a vector
register.

```
  D0.f32 = u32_to_f32(S0[31 : 24].u32)
```

#### V_CVT_U32_F64  (opcode 21)

Convert from a double-precision float input to an unsigned 32-bit integer value and store the result into a
vector register.

```
  D0.u32 = f64_to_u32(S0.f64)
```

Notes

0.5ULP accuracy, out-of-range floating point values (including infinity) saturate. NAN is converted to 0.

Generation of the INEXACT exception is controlled by the CLAMP bit. INEXACT exceptions are enabled for this
conversion iff CLAMP == 1.

#### V_CVT_F64_U32  (opcode 22)

Convert from an unsigned 32-bit integer input to a double-precision float value and store the result into a
vector register.

```
  D0.f64 = u32_to_f64(S0.u32)
```

Notes

0ULP accuracy.

#### V_TRUNC_F64  (opcode 23)

Compute the integer part of a double-precision float input using round toward zero semantics and store the
result in floating point format into a vector register.

```
  D0.f64 = trunc(S0.f64)
```

#### V_CEIL_F64  (opcode 24)

Round the double-precision float input up to next integer and store the result in floating point format into a
vector register.

```
  D0.f64 = trunc(S0.f64);
  if ((S0.f64 > 0.0) && (S0.f64 != D0.f64)) then
      D0.f64 += 1.0
  endif
```

#### V_RNDNE_F64  (opcode 25)

Round the double-precision float input to the nearest even integer and store the result in floating point format
into a vector register.

```
  D0.f64 = floor(S0.f64 + 0.5);
  if (isEven(floor(S0.f64)) && (fract(S0.f64) == 0.5)) then
      D0.f64 -= 1.0
  endif
```

#### V_FLOOR_F64  (opcode 26)

Round the double-precision float input down to previous integer and store the result in floating point format
into a vector register.

```
  D0.f64 = trunc(S0.f64);
  if ((S0.f64 < 0.0) && (S0.f64 != D0.f64)) then
      D0.f64 += -1.0
  endif
```

#### V_FRACT_F32  (opcode 27)

Compute the fractional portion of a single-precision float input and store the result in floating point format into
a vector register.

```
  D0.f32 = S0.f32 + -floor(S0.f32)
```

Notes

0.5ULP accuracy, denormals are accepted.

This is intended to comply with the DX specification of fract where the function behaves like an extension of
integer modulus; be aware this may differ from how fract() is defined in other domains. For example: fract(-
1.2) = 0.8 in DX.

Obey round mode, result clamped to 0x3f7fffff.

#### V_TRUNC_F32  (opcode 28)

Compute the integer part of a single-precision float input using round toward zero semantics and store the
result in floating point format into a vector register.

```
  D0.f32 = trunc(S0.f32)
```

#### V_CEIL_F32  (opcode 29)

Round the single-precision float input up to next integer and store the result in floating point format into a
vector register.

```
  D0.f32 = trunc(S0.f32);
  if ((S0.f32 > 0.0F) && (S0.f32 != D0.f32)) then
        D0.f32 += 1.0F
  endif
```

#### V_RNDNE_F32  (opcode 30)

Round the single-precision float input to the nearest even integer and store the result in floating point format
into a vector register.

```
  D0.f32 = floor(S0.f32 + 0.5F);
  if (isEven(64'F(floor(S0.f32))) && (fract(S0.f32) == 0.5F)) then
        D0.f32 -= 1.0F
  endif
```

#### V_FLOOR_F32  (opcode 31)

Round the single-precision float input down to previous integer and store the result in floating point format
into a vector register.

```
  D0.f32 = trunc(S0.f32);
  if ((S0.f32 < 0.0F) && (S0.f32 != D0.f32)) then
        D0.f32 += -1.0F
  endif
```

#### V_EXP_F32  (opcode 32)

Calculate 2 raised to the power of the single-precision float input and store the result into a vector register.

```
  D0.f32 = pow(2.0F, S0.f32)
```

Notes

1ULP accuracy, denormals are flushed.

Functional examples:

```
  V_EXP_F32(0xff800000) => 0x00000000        // exp(-INF) = 0
  V_EXP_F32(0x80000000) => 0x3f800000        // exp(-0.0) = 1
  V_EXP_F32(0x7f800000) => 0x7f800000        // exp(+INF) = +INF
```

#### V_LOG_F32  (opcode 33)

Calculate the base 2 logarithm of the single-precision float input and store the result into a vector register.

```
  D0.f32 = log2(S0.f32)
```

Notes

1ULP accuracy, denormals are flushed.

Functional examples:

```
  V_LOG_F32(0xff800000) => 0xffc00000        // log(-INF) = NAN
  V_LOG_F32(0xbf800000) => 0xffc00000        // log(-1.0) = NAN
  V_LOG_F32(0x80000000) => 0xff800000        // log(-0.0) = -INF
  V_LOG_F32(0x00000000) => 0xff800000        // log(+0.0) = -INF
  V_LOG_F32(0x3f800000) => 0x00000000        // log(+1.0) = 0
```

```
  V_LOG_F32(0x7f800000) => 0x7f800000       // log(+INF) = +INF
```

#### V_RCP_F32  (opcode 34)

Calculate the reciprocal of the single-precision float input using IEEE rules and store the result into a vector
register.

```
  D0.f32 = 1.0F / S0.f32
```

Notes

1ULP accuracy. Accuracy converges to < 0.5ULP when using the Newton-Raphson method and 2 FMA
operations. Denormals are flushed.

Functional examples:

```
  V_RCP_F32(0xff800000) => 0x80000000       // rcp(-INF) = -0
  V_RCP_F32(0xc0000000) => 0xbf000000       // rcp(-2.0) = -0.5
  V_RCP_F32(0x80000000) => 0xff800000       // rcp(-0.0) = -INF
  V_RCP_F32(0x00000000) => 0x7f800000       // rcp(+0.0) = +INF
  V_RCP_F32(0x7f800000) => 0x00000000       // rcp(+INF) = +0
```

#### V_RCP_IFLAG_F32  (opcode 35)

Calculate the reciprocal of the vector float input in a manner suitable for integer division and store the result
into a vector register. This opcode is intended for use as part of an integer division macro.

```
  D0.f32 = 1.0F / S0.f32;
  // Can only raise integer DIV_BY_ZERO exception
```

Notes

Can raise integer DIV_BY_ZERO exception but cannot raise floating-point exceptions. To be used in an integer
reciprocal macro by the compiler with one of the sequences listed below (depending on signed or unsigned
operation).

Unsigned usage:
CVT_F32_U32
RCP_IFLAG_F32
MUL_F32 (2**32 - 1)
CVT_U32_F32

Signed usage:
CVT_F32_I32

RCP_IFLAG_F32
MUL_F32 (2**31 - 1)
CVT_I32_F32

#### V_RSQ_F32  (opcode 36)

Calculate the reciprocal of the square root of the single-precision float input using IEEE rules and store the
result into a vector register.

```
  D0.f32 = 1.0F / sqrt(S0.f32)
```

Notes

1ULP accuracy, denormals are flushed.

Functional examples:

```
  V_RSQ_F32(0xff800000) => 0xffc00000       // rsq(-INF) = NAN
  V_RSQ_F32(0x80000000) => 0xff800000       // rsq(-0.0) = -INF
  V_RSQ_F32(0x00000000) => 0x7f800000       // rsq(+0.0) = +INF
  V_RSQ_F32(0x40800000) => 0x3f000000       // rsq(+4.0) = +0.5
  V_RSQ_F32(0x7f800000) => 0x00000000       // rsq(+INF) = +0
```

#### V_RCP_F64  (opcode 37)

Calculate the reciprocal of the double-precision float input using IEEE rules and store the result into a vector
register.

```
  D0.f64 = 1.0 / S0.f64
```

Notes

This opcode has (2**29)ULP accuracy and supports denormals.

#### V_RSQ_F64  (opcode 38)

Calculate the reciprocal of the square root of the double-precision float input using IEEE rules and store the
result into a vector register.

```
  D0.f64 = 1.0 / sqrt(S0.f64)
```

Notes

This opcode has (2**29)ULP accuracy and supports denormals.

#### V_SQRT_F32  (opcode 39)

Calculate the square root of the single-precision float input using IEEE rules and store the result into a vector
register.

```
  D0.f32 = sqrt(S0.f32)
```

Notes

1ULP accuracy, denormals are flushed.

Functional examples:

```
  V_SQRT_F32(0xff800000) => 0xffc00000       // sqrt(-INF) = NAN
  V_SQRT_F32(0x80000000) => 0x80000000       // sqrt(-0.0) = -0
  V_SQRT_F32(0x00000000) => 0x00000000       // sqrt(+0.0) = +0
  V_SQRT_F32(0x40800000) => 0x40000000       // sqrt(+4.0) = +2.0
  V_SQRT_F32(0x7f800000) => 0x7f800000       // sqrt(+INF) = +INF
```

#### V_SQRT_F64  (opcode 40)

Calculate the square root of the double-precision float input using IEEE rules and store the result into a vector
register.

```
  D0.f64 = sqrt(S0.f64)
```

Notes

This opcode has (2**29)ULP accuracy and supports denormals.

#### V_SIN_F32  (opcode 41)

Calculate the trigonometric sine of a single-precision float value using IEEE rules and store the result into a
vector register. The operand is calculated by scaling the vector input by 2 PI.

```
  D0.f32 = sin(S0.f32 * 32'F(PI * 2.0))
```

Notes

Denormals are supported. Full range input is supported.

Functional examples:

```
  V_SIN_F32(0xff800000) => 0xffc00000       // sin(-INF) = NAN
  V_SIN_F32(0xff7fffff) => 0x00000000       // -MaxFloat, finite
  V_SIN_F32(0x80000000) => 0x80000000       // sin(-0.0) = -0
  V_SIN_F32(0x3e800000) => 0x3f800000       // sin(0.25) = 1
  V_SIN_F32(0x7f800000) => 0xffc00000       // sin(+INF) = NAN
```

#### V_COS_F32  (opcode 42)

Calculate the trigonometric cosine of a single-precision float value using IEEE rules and store the result into a
vector register. The operand is calculated by scaling the vector input by 2 PI.

```
  D0.f32 = cos(S0.f32 * 32'F(PI * 2.0))
```

Notes

Denormals are supported. Full range input is supported.

Functional examples:

```
  V_COS_F32(0xff800000) => 0xffc00000       // cos(-INF) = NAN
  V_COS_F32(0xff7fffff) => 0x3f800000       // -MaxFloat, finite
  V_COS_F32(0x80000000) => 0x3f800000       // cos(-0.0) = 1
  V_COS_F32(0x3e800000) => 0x00000000       // cos(0.25) = 0
  V_COS_F32(0x7f800000) => 0xffc00000       // cos(+INF) = NAN
```

#### V_NOT_B32  (opcode 43)

Calculate bitwise negation on a vector input and store the result into a vector register.

```
  D0.u32 = ~S0.u32
```

Notes

Input and output modifiers not supported.

#### V_BFREV_B32  (opcode 44)

Reverse the order of bits in a vector input and store the result into a vector register.

```
  D0.u32[31 : 0] = S0.u32[0 : 31]
```

Notes

Input and output modifiers not supported.

#### V_FFBH_U32  (opcode 45)

Count the number of leading "0" bits before the first "1" in a vector input and store the result into a vector
register. Store -1 if there are no "1" bits.

```
  D0.i32 = -1;
  // Set if no ones are found
  for i in 0 : 31 do
        // Search from MSB
        if S0.u32[31 - i] == 1'1U then
            D0.i32 = i;
            break
        endif
  endfor
```

Notes

Functional examples:

```
  V_FFBH_U32(0x00000000) => 0xffffffff
  V_FFBH_U32(0x800000ff) => 0
  V_FFBH_U32(0x100000ff) => 3
  V_FFBH_U32(0x0000ffff) => 16
  V_FFBH_U32(0x00000001) => 31
```

#### V_FFBL_B32  (opcode 46)

Count the number of trailing "0" bits before the first "1" in a vector input and store the result into a vector
register. Store -1 if there are no "1" bits in the input.

```
  D0.i32 = -1;
  // Set if no ones are found
  for i in 0 : 31 do
        // Search from LSB
        if S0.u32[i] == 1'1U then
            D0.i32 = i;
            break
        endif
```

```
  endfor
```

Notes

Functional examples:

```
  V_FFBL_B32(0x00000000) => 0xffffffff
  V_FFBL_B32(0xff000001) => 0
  V_FFBL_B32(0xff000008) => 3
  V_FFBL_B32(0xffff0000) => 16
  V_FFBL_B32(0x80000000) => 31
```

#### V_FFBH_I32  (opcode 47)

Count the number of leading bits that are the same as the sign bit of a vector input and store the result into a
vector register. Store -1 if all input bits are the same.

```
  D0.i32 = -1;
  // Set if all bits are the same
  for i in 1 : 31 do
        // Search from MSB
        if S0.i32[31 - i] != S0.i32[31] then
            D0.i32 = i;
            break
        endif
  endfor
```

Notes

Functional examples:

```
  V_FFBH_I32(0x00000000) => 0xffffffff
  V_FFBH_I32(0x40000000) => 1
  V_FFBH_I32(0x80000000) => 1
  V_FFBH_I32(0x0fffffff) => 4
  V_FFBH_I32(0xffff0000) => 16
  V_FFBH_I32(0xfffffffe) => 31
  V_FFBH_I32(0xffffffff) => 0xffffffff
```

#### V_FREXP_EXP_I32_F64  (opcode 48)

Extract the exponent of a double-precision float input and store the result as a signed 32-bit integer into a
vector register.

```
  if ((S0.f64 == +INF) || (S0.f64 == -INF) || isNAN(S0.f64)) then
```

```
        D0.i32 = 0
  else
        D0.i32 = exponent(S0.f64) - 1023 + 1
  endif
```

Notes

This operation satisfies the invariant S0.f64 = significand * (2 ** exponent). See also V_FREXP_MANT_F64,
which returns the significand. See the C library function frexp() for more information.

#### V_FREXP_MANT_F64  (opcode 49)

Extract the binary significand, or mantissa, of a double-precision float input and store the result as a double-
precision float into a vector register.

```
  if ((S0.f64 == +INF) || (S0.f64 == -INF) || isNAN(S0.f64)) then
        D0.f64 = S0.f64
  else
        D0.f64 = mantissa(S0.f64)
  endif
```

Notes

This operation satisfies the invariant S0.f64 = significand * (2 ** exponent). Result range is in (-1.0,-0.5][0.5,1.0)
in normal cases. See also V_FREXP_EXP_I32_F64, which returns integer exponent. See the C library function
frexp() for more information.

#### V_FRACT_F64  (opcode 50)

Compute the fractional portion of a double-precision float input and store the result in floating point format
into a vector register.

```
  D0.f64 = S0.f64 + -floor(S0.f64)
```

Notes

0.5ULP accuracy, denormals are accepted.

This is intended to comply with the DX specification of fract where the function behaves like an extension of
integer modulus; be aware this may differ from how fract() is defined in other domains. For example: fract(-
1.2) = 0.8 in DX.

Obey round mode, result clamped to 0x3fefffffffffffff.

#### V_FREXP_EXP_I32_F32  (opcode 51)

Extract the exponent of a single-precision float input and store the result as a signed 32-bit integer into a vector
register.

```
  if ((64'F(S0.f32) == +INF) || (64'F(S0.f32) == -INF) || isNAN(64'F(S0.f32))) then
        D0.i32 = 0
  else
        D0.i32 = exponent(S0.f32) - 127 + 1
  endif
```

Notes

This operation satisfies the invariant S0.f32 = significand * (2 ** exponent). See also V_FREXP_MANT_F32,
which returns the significand. See the C library function frexp() for more information.

#### V_FREXP_MANT_F32  (opcode 52)

Extract the binary significand, or mantissa, of a single-precision float input and store the result as a single-
precision float into a vector register.

```
  if ((64'F(S0.f32) == +INF) || (64'F(S0.f32) == -INF) || isNAN(64'F(S0.f32))) then
        D0.f32 = S0.f32
  else
        D0.f32 = mantissa(S0.f32)
  endif
```

Notes

This operation satisfies the invariant S0.f32 = significand * (2 ** exponent). Result range is in (-1.0,-0.5][0.5,1.0)
in normal cases. See also V_FREXP_EXP_I32_F32, which returns integer exponent. See the C library function
frexp() for more information.

#### V_CLREXCP  (opcode 53)

Clear this wave's exception state in the vector ALU.

#### V_MOV_B64  (opcode 56)

Move data from a 64-bit vector input into a vector register.

```
  D0.b64 = S0.b64
```

Notes

Floating-point modifiers are valid for this instruction if S0.u64 is a 64-bit floating point value. This instruction is
suitable for negating or taking the absolute value of a floating-point value.

#### V_CVT_F16_U16  (opcode 57)

Convert from an unsigned 16-bit integer input to a half-precision float value and store the result into a vector
register.

```
  D0.f16 = u16_to_f16(S0.u16)
```

Notes

0.5ULP accuracy, supports denormals, rounding, exception flags and saturation.

#### V_CVT_F16_I16  (opcode 58)

Convert from a signed 16-bit integer input to a half-precision float value and store the result into a vector
register.

```
  D0.f16 = i16_to_f16(S0.i16)
```

Notes

0.5ULP accuracy, supports denormals, rounding, exception flags and saturation.

#### V_CVT_U16_F16  (opcode 59)

Convert from a half-precision float input to an unsigned 16-bit integer value and store the result into a vector
register.

```
  D0.u16 = f16_to_u16(S0.f16)
```

Notes

1ULP accuracy, supports rounding, exception flags and saturation. FP16 denormals are accepted. Conversion
is done with truncation.

Generation of the INEXACT exception is controlled by the CLAMP bit. INEXACT exceptions are enabled for this
conversion iff CLAMP == 1.

#### V_CVT_I16_F16  (opcode 60)

Convert from a half-precision float input to a signed 16-bit integer value and store the result into a vector
register.

```
  D0.i16 = f16_to_i16(S0.f16)
```

Notes

1ULP accuracy, supports rounding, exception flags and saturation. FP16 denormals are accepted. Conversion
is done with truncation.

Generation of the INEXACT exception is controlled by the CLAMP bit. INEXACT exceptions are enabled for this
conversion iff CLAMP == 1.

#### V_RCP_F16  (opcode 61)

Calculate the reciprocal of the half-precision float input using IEEE rules and store the result into a vector
register.

```
  D0.f16 = 16'1.0 / S0.f16
```

Notes

1ULP accuracy.

Functional examples:

```
  V_RCP_F16(0xfc00) => 0x8000        // rcp(-INF) = -0
  V_RCP_F16(0xc000) => 0xb800        // rcp(-2.0) = -0.5
  V_RCP_F16(0x8000) => 0xfc00        // rcp(-0.0) = -INF
  V_RCP_F16(0x0000) => 0x7c00        // rcp(+0.0) = +INF
  V_RCP_F16(0x7c00) => 0x0000        // rcp(+INF) = +0
```

#### V_SQRT_F16  (opcode 62)

Calculate the square root of the half-precision float input using IEEE rules and store the result into a vector
register.

```
  D0.f16 = sqrt(S0.f16)
```

Notes

1ULP accuracy, denormals are supported.

Functional examples:

```
  V_SQRT_F16(0xfc00) => 0xfe00       // sqrt(-INF) = NAN
  V_SQRT_F16(0x8000) => 0x8000       // sqrt(-0.0) = -0
  V_SQRT_F16(0x0000) => 0x0000       // sqrt(+0.0) = +0
  V_SQRT_F16(0x4400) => 0x4000       // sqrt(+4.0) = +2.0
  V_SQRT_F16(0x7c00) => 0x7c00       // sqrt(+INF) = +INF
```

#### V_RSQ_F16  (opcode 63)

Calculate the reciprocal of the square root of the half-precision float input using IEEE rules and store the result
into a vector register.

```
  D0.f16 = 16'1.0 / sqrt(S0.f16)
```

Notes

1ULP accuracy, denormals are supported.

Functional examples:

```
  V_RSQ_F16(0xfc00) => 0xfe00        // rsq(-INF) = NAN
  V_RSQ_F16(0x8000) => 0xfc00        // rsq(-0.0) = -INF
  V_RSQ_F16(0x0000) => 0x7c00        // rsq(+0.0) = +INF
  V_RSQ_F16(0x4400) => 0x3800        // rsq(+4.0) = +0.5
  V_RSQ_F16(0x7c00) => 0x0000        // rsq(+INF) = +0
```

#### V_LOG_F16  (opcode 64)

Calculate the base 2 logarithm of the half-precision float input and store the result into a vector register.

```
  D0.f16 = log2(S0.f16)
```

Notes

1ULP accuracy, denormals are supported.

Functional examples:

```
  V_LOG_F16(0xfc00) => 0xfe00        // log(-INF) = NAN
  V_LOG_F16(0xbc00) => 0xfe00        // log(-1.0) = NAN
```

```
  V_LOG_F16(0x8000) => 0xfc00        // log(-0.0) = -INF
  V_LOG_F16(0x0000) => 0xfc00        // log(+0.0) = -INF
  V_LOG_F16(0x3c00) => 0x0000        // log(+1.0) = 0
  V_LOG_F16(0x7c00) => 0x7c00        // log(+INF) = +INF
```

#### V_EXP_F16  (opcode 65)

Calculate 2 raised to the power of the half-precision float input and store the result into a vector register.

```
  D0.f16 = pow(16'2.0, S0.f16)
```

Notes

1ULP accuracy, denormals are supported.

Functional examples:

```
  V_EXP_F16(0xfc00) => 0x0000        // exp(-INF) = 0
  V_EXP_F16(0x8000) => 0x3c00        // exp(-0.0) = 1
  V_EXP_F16(0x7c00) => 0x7c00        // exp(+INF) = +INF
```

#### V_FREXP_MANT_F16  (opcode 66)

Extract the binary significand, or mantissa, of a half-precision float input and store the result as a half-
precision float into a vector register.

```
  if ((64'F(S0.f16) == +INF) || (64'F(S0.f16) == -INF) || isNAN(64'F(S0.f16))) then
        D0.f16 = S0.f16
  else
        D0.f16 = mantissa(S0.f16)
  endif
```

Notes

This operation satisfies the invariant S0.f16 = significand * (2 ** exponent). Result range is in (-1.0,-0.5][0.5,1.0)
in normal cases. See also V_FREXP_EXP_I16_F16, which returns integer exponent. See the C library function
frexp() for more information.

#### V_FREXP_EXP_I16_F16  (opcode 67)

Extract the exponent of a half-precision float input and store the result as a signed 16-bit integer into a vector
register.

```
  if ((64'F(S0.f16) == +INF) || (64'F(S0.f16) == -INF) || isNAN(64'F(S0.f16))) then
        D0.i16 = 16'0
  else
        D0.i16 = 16'I(exponent(S0.f16) - 15 + 1)
  endif
```

Notes

This operation satisfies the invariant S0.f16 = significand * (2 ** exponent). See also V_FREXP_MANT_F16,
which returns the significand. See the C library function frexp() for more information.

#### V_FLOOR_F16  (opcode 68)

Round the half-precision float input down to previous integer and store the result in floating point format into
a vector register.

```
  D0.f16 = trunc(S0.f16);
  if ((S0.f16 < 16'0.0) && (S0.f16 != D0.f16)) then
        D0.f16 += -16'1.0
  endif
```

#### V_CEIL_F16  (opcode 69)

Round the half-precision float input up to next integer and store the result in floating point format into a vector
register.

```
  D0.f16 = trunc(S0.f16);
  if ((S0.f16 > 16'0.0) && (S0.f16 != D0.f16)) then
        D0.f16 += 16'1.0
  endif
```

#### V_TRUNC_F16  (opcode 70)

Compute the integer part of a half-precision float input using round toward zero semantics and store the result
in floating point format into a vector register.

```
  D0.f16 = trunc(S0.f16)
```

#### V_RNDNE_F16  (opcode 71)

Round the half-precision float input to the nearest even integer and store the result in floating point format
into a vector register.

```
  D0.f16 = floor(S0.f16 + 16'0.5);
  if (isEven(64'F(floor(S0.f16))) && (fract(S0.f16) == 16'0.5)) then
        D0.f16 -= 16'1.0
  endif
```

#### V_FRACT_F16  (opcode 72)

Compute the fractional portion of a half-precision float input and store the result in floating point format into a
vector register.

```
  D0.f16 = S0.f16 + -floor(S0.f16)
```

Notes

0.5ULP accuracy, denormals are accepted.

This is intended to comply with the DX specification of fract where the function behaves like an extension of
integer modulus; be aware this may differ from how fract() is defined in other domains. For example: fract(-
1.2) = 0.8 in DX.

#### V_SIN_F16  (opcode 73)

Calculate the trigonometric sine of a half-precision float value using IEEE rules and store the result into a
vector register. The operand is calculated by scaling the vector input by 2 PI.

```
  D0.f16 = sin(S0.f16 * 16'F(PI * 2.0))
```

Notes

Denormals are supported. Full range input is supported.

Functional examples:

```
  V_SIN_F16(0xfc00) => 0xfe00        // sin(-INF) = NAN
  V_SIN_F16(0xfbff) => 0x0000        // Most negative finite FP16
  V_SIN_F16(0x8000) => 0x8000        // sin(-0.0) = -0
  V_SIN_F16(0x3400) => 0x3c00        // sin(0.25) = 1
  V_SIN_F16(0x7bff) => 0x0000        // Most positive finite FP16
  V_SIN_F16(0x7c00) => 0xfe00        // sin(+INF) = NAN
```

#### V_COS_F16  (opcode 74)

Calculate the trigonometric cosine of a half-precision float value using IEEE rules and store the result into a
vector register. The operand is calculated by scaling the vector input by 2 PI.

```
  D0.f16 = cos(S0.f16 * 16'F(PI * 2.0))
```

Notes

Denormals are supported. Full range input is supported.

Functional examples:

```
  V_COS_F16(0xfc00) => 0xfe00        // cos(-INF) = NAN
  V_COS_F16(0xfbff) => 0x3c00        // Most negative finite FP16
  V_COS_F16(0x8000) => 0x3c00        // cos(-0.0) = 1
  V_COS_F16(0x3400) => 0x0000        // cos(0.25) = 0
  V_COS_F16(0x7bff) => 0x3c00        // Most positive finite FP16
  V_COS_F16(0x7c00) => 0xfe00        // cos(+INF) = NAN
```

#### V_CVT_NORM_I16_F16  (opcode 77)

Convert from a half-precision float input to a signed normalized short and store the result into a vector
register.

```
  D0.i16 = f16_to_snorm(S0.f16)
```

Notes

0.5ULP accuracy, supports rounding, exception flags and saturation, denormals are supported.

#### V_CVT_NORM_U16_F16  (opcode 78)

Convert from a half-precision float input to an unsigned normalized short and store the result into a vector
register.

```
  D0.u16 = f16_to_unorm(S0.f16)
```

Notes

0.5ULP accuracy, supports rounding, exception flags and saturation, denormals are supported.

#### V_SAT_PK_U8_I16  (opcode 79)

Given 2 signed 16-bit integer inputs, saturate each input over an unsigned 8-bit integer range, pack the
resulting values into a packed 16-bit value and store the result into a vector register.

```
  SAT8 = lambda(n) (
        if n <= 16'0 then
            return 8'0U
        elsif n >= 16'255 then
            return 8'255U
        else
            return n[7 : 0].u8
        endif);
  tmp = 16'0;
  tmp[7 : 0].u8 = SAT8(S0[15 : 0].i16);
  tmp[15 : 8].u8 = SAT8(S0[31 : 16].i16);
  D0.b16 = tmp.b16
```

Notes

Used for 4x16bit data packed as 4x8bit data.

#### V_SWAP_B32  (opcode 81)

Swap the values in two vector registers.

```
  tmp = D0.b32;
  D0.b32 = S0.b32;
  S0.b32 = tmp
```

Notes

Input and output modifiers not supported; this is an untyped operation.

#### V_ACCVGPR_MOV_B32  (opcode 82)

Move data from one accumulator register to another accumulator register.

#### V_CVT_F32_FP8  (opcode 84)

Convert from an FP8 float input to a single-precision float value and store the result into a vector register.

```
  if SDWA_SRC0_SEL == BYTE1.b3 then
        D0.f32 = fp8_to_f32(S0[15 : 8].fp8)
  elsif SDWA_SRC0_SEL == BYTE2.b3 then
```

```
        D0.f32 = fp8_to_f32(S0[23 : 16].fp8)
  elsif SDWA_SRC0_SEL == BYTE3.b3 then
        D0.f32 = fp8_to_f32(S0[31 : 24].fp8)
  else
        // BYTE0 implied
        D0.f32 = fp8_to_f32(S0[7 : 0].fp8)
  endif
```

Notes

SDWA encoding allows SRC0_SEL to control which byte of S0 is converted. Only the BYTE selects of SRC0_SEL
are legal. If this instruction is not encoded in SDWA then BYTE0 is implied.

#### V_CVT_F32_BF8  (opcode 85)

Convert from a BF8 float input to a single-precision float value and store the result into a vector register.

```
  if SDWA_SRC0_SEL == BYTE1.b3 then
        D0.f32 = bf8_to_f32(S0[15 : 8].bf8)
  elsif SDWA_SRC0_SEL == BYTE2.b3 then
        D0.f32 = bf8_to_f32(S0[23 : 16].bf8)
  elsif SDWA_SRC0_SEL == BYTE3.b3 then
        D0.f32 = bf8_to_f32(S0[31 : 24].bf8)
  else
        // BYTE0 implied
        D0.f32 = bf8_to_f32(S0[7 : 0].bf8)
  endif
```

Notes

SDWA encoding allows SRC0_SEL to control which byte of S0 is converted. Only the BYTE selects of SRC0_SEL
are legal. If this instruction is not encoded in SDWA then BYTE0 is implied.

#### V_CVT_PK_F32_FP8  (opcode 86)

Convert from a packed 2-component FP8 float input to a packed single-precision float value and store the result
into a vector register.

```
  tmp = SDWA_SRC0_SEL[1 : 0] == WORD1.b2 ? S0[31 : 16] : S0[15 : 0];
  D0[31 : 0].f32 = fp8_to_f32(tmp[7 : 0].fp8);
  D0[63 : 32].f32 = fp8_to_f32(tmp[15 : 8].fp8)
```

Notes

SDWA encoding allows SRC0_SEL to control which word of S0 is converted. Only the WORD selects of
SRC0_SEL are legal. If this instruction is not encoded in SDWA then WORD0 is implied.

#### V_CVT_PK_F32_BF8  (opcode 87)

Convert from a packed 2-component BF8 float input to a packed single-precision float value and store the result
into a vector register.

```
  tmp = SDWA_SRC0_SEL[1 : 0] == WORD1.b2 ? S0[31 : 16] : S0[15 : 0];
  D0[31 : 0].f32 = bf8_to_f32(tmp[7 : 0].bf8);
  D0[63 : 32].f32 = bf8_to_f32(tmp[15 : 8].bf8)
```

Notes

SDWA encoding allows SRC0_SEL to control which word of S0 is converted. Only the WORD selects of
SRC0_SEL are legal. If this instruction is not encoded in SDWA then WORD0 is implied.

#### V_PRNG_B32  (opcode 88)

Generate a pseudorandom number using an LFSR (linear feedback shift register) seeded with the vector input,
then store the result into a vector register.

```
  in = S0.u32;
  D0.u32 = ((in << 1U) ^ (in[31] ? 197U : 0U))
```

Notes

This function produces a sequence of pseudorandom numbers with period 2**32 - 1 unless the input is zero, in
which case the period is 1.

#### V_PERMLANE16_SWAP_B32  (opcode 89)

Swap data between two vector registers. Odd rows of the first operand are swapped with even rows of the
second operand (one row is 16 lanes).

```
  for pass in 0 : 1 do
        for lane in 0 : 15 do
            tmp = VGPR[pass * 32 + lane][SRC0.u32];
            VGPR[pass * 32 + lane][SRC0.u32] = VGPR[pass * 32 + lane + 16][VDST.u32];
            VGPR[pass * 32 + lane + 16][VDST.u32] = tmp
        endfor
  endfor
```

Notes

ABS, NEG and OMOD modifiers should all be zeroed for this instruction.

This instruction is useful for BFP data conversions.

#### V_PERMLANE32_SWAP_B32  (opcode 90)

Swap data between two vector registers. Rows 2 and 3 of the first operand are swapped with rows 0 and 1 of the
second operand (one row is 16 lanes).

```
  for lane in 0 : 31 do
        tmp = VGPR[lane][SRC0.u32];
        VGPR[lane][SRC0.u32] = VGPR[lane + 32][VDST.u32];
        VGPR[lane + 32][VDST.u32] = tmp
  endfor
```

Notes

ABS, NEG and OMOD modifiers should all be zeroed for this instruction.

This instruction is useful for BFP data conversions.

#### V_CVT_F32_BF16  (opcode 91)

Convert from a BF16 float input to a single-precision float value and store the result into a vector register.

```
  D0.f32 = 32'F({ S0.b16, 16'0U })
```

#### 12.8.1. VOP1 using VOP3 encoding

Instructions in this format may also be encoded as VOP3. This allows access to the extra control bits (e.g. ABS,
OMOD) in exchange for not being able to use a literal constant. The VOP3 opcode is: VOP2 opcode + 0x140.

### 12.9. VOPC Instructions

The bitfield map for VOPC is:

```
        where:
```

```
        SRC0   = First operand for instruction.
        VSRC1 = Second operand for instruction.
        OP     = Instructions.
        All VOPC instructions can alternatively be encoded in the VOP3A format.
```

Compare instructions perform the same compare operation on each lane (workItem or thread) using that
lane's private data, and producing a 1 bit result per lane into VCC or EXEC.

Instructions in this format may use a 32-bit literal constant which occurs immediately after the instruction.

Most compare instructions fall into one of two categories:

- Those which can use one of 16 compare operations (floating point types). "{COMPF}"
- Those which can use one of 8 compare operations (integer types). "{COMPI}"

The opcode number is such that for these the opcode number can be calculated from a base opcode number
for the data type, plus an offset for the specific compare operation.

**Table 59. Float Compare Operations**

```
Compare Operation         Opcode Offset   Description
F                         0               D.u = 0
LT                        1               D.u = (S0 < S1)
EQ                        2               D.u = (S0 == S1)
LE                        3               D.u = (S0 <= S1)
GT                        4               D.u = (S0 > S1)
LG                        5               D.u = (S0 <> S1)
```

```
Compare Operation         Opcode Offset      Description
GE                        6                  D.u = (S0 >= S1)
O                         7                  D.u = (!isNaN(S0) && !isNaN(S1))
U                         8                  D.u = (!isNaN(S0) || !isNaN(S1))
NGE                       9                  D.u = !(S0 >= S1)
NLG                       10                 D.u = !(S0 <> S1)
NGT                       11                 D.u = !(S0 > S1)
NLE                       12                 D.u = !(S0 <= S1)
NEQ                       13                 D.u = !(S0 == S1)
NLT                       14                 D.u = !(S0 < S1)
TRU                       15                 D.u = 1
```

**Table 60. Instructions with Sixteen Compare Operations**

```
Instruction                     Description                                     Hex Range
V_CMP_{COMPF}_F16               16-bit float compare.                           0x20 to 0x2F
V_CMPX_{COMPF}_F16              16-bit float compare. Also writes EXEC.         0x30 to 0x3F
V_CMP_{COMPF}_F32               32-bit float compare.                           0x40 to 0x4F
V_CMPX_{COMPF}_F32              32-bit float compare. Also writes EXEC.         0x50 to 0x5F
V_CMPS_{COMPF}_F64              64-bit float compare.                           0x60 to 0x6F
V_CMPSX_{COMPF}_F64             64-bit float compare. Also writes EXEC.         0x70 to 0x7F
```

**Table 61. Integer Compare Operations**

```
Compare Operation         Opcode Offset      Description
F                         0                  D.u = 0
LT                        1                  D.u = (S0 < S1)
EQ                        2                  D.u = (S0 == S1)
LE                        3                  D.u = (S0 <= S1)
GT                        4                  D.u = (S0 > S1)
LG                        5                  D.u = (S0 <> S1)
GE                        6                  D.u = (S0 >= S1)
TRU                       7                  D.u = 1
```

**Table 62. Instructions with Eight Compare Operations**

```
Instruction                    Description                                                     Hex Range
V_CMP_{COMPI}_I16              16-bit signed integer compare.                                  0xA0 - 0xA7
V_CMP_{COMPI}_U16              16-bit signed integer compare.                                  0xA8 - 0xAF
V_CMPX_{COMPI}_I16             16-bit unsigned integer compare. Also writes EXEC.              0xB0 - 0xB7
V_CMPX_{COMPI}_U16             16-bit unsigned integer compare. Also writes EXEC.              0xB8 - 0xBF
V_CMP_{COMPI}_I32              32-bit signed integer compare.                                  0xC0 - 0xC7
V_CMP_{COMPI}_U32              32-bit signed integer compare.                                  0xC8 - 0xCF
V_CMPX_{COMPI}_I32             32-bit unsigned integer compare. Also writes EXEC.              0xD0 - 0xD7
V_CMPX_{COMPI}_U32             32-bit unsigned integer compare. Also writes EXEC.              0xD8 - 0xDF
V_CMP_{COMPI}_I64              64-bit signed integer compare.                                  0xE0 - 0xE7
V_CMP_{COMPI}_U64              64-bit signed integer compare.                                  0xE8 - 0xEF
V_CMPX_{COMPI}_I64             64-bit unsigned integer compare. Also writes EXEC.              0xF0 - 0xF7
V_CMPX_{COMPI}_U64             64-bit unsigned integer compare. Also writes EXEC.              0xF8 - 0xFF
```

**Table 63. VOPC Compare Opcodes**

#### V_CMP_CLASS_F32  (opcode 16)

Evaluate the IEEE numeric class function specified as a 10 bit mask in the second input on the first input, a
single-precision float, and set the per-lane condition code to the result. Store the result into VCC or a scalar
register.

The function reports true if the floating point value is any of the numeric types selected in the 10 bit mask
according to the following list:

S1.u[0] value is a signaling NAN.
S1.u[1] value is a quiet NAN.
S1.u[2] value is negative infinity.
S1.u[3] value is a negative normal value.
S1.u[4] value is a negative denormal value.
S1.u[5] value is negative zero.
S1.u[6] value is positive zero.
S1.u[7] value is a positive denormal value.
S1.u[8] value is a positive normal value.
S1.u[9] value is positive infinity.

```
  declare result : 1'U;
  if isSignalNAN(64'F(S0.f32)) then
      result = S1.u32[0]
  elsif isQuietNAN(64'F(S0.f32)) then
      result = S1.u32[1]
  elsif exponent(S0.f32) == 255 then
      // +-INF
      result = S1.u32[sign(S0.f32) ? 2 : 9]
  elsif exponent(S0.f32) > 0 then
      // +-normal value
      result = S1.u32[sign(S0.f32) ? 3 : 8]
  elsif 64'F(abs(S0.f32)) > 0.0 then
      // +-denormal value
      result = S1.u32[sign(S0.f32) ? 4 : 7]
  else
      // +-0.0
      result = S1.u32[sign(S0.f32) ? 5 : 6]
  endif;
  D0.u64[laneId] = result;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_CLASS_F32  (opcode 17)

Evaluate the IEEE numeric class function specified as a 10 bit mask in the second input on the first input, a
single-precision float, and set the per-lane condition code to the result. Store the result into the EXEC mask and
to VCC or a scalar register.

The function reports true if the floating point value is any of the numeric types selected in the 10 bit mask
according to the following list:

S1.u[0] value is a signaling NAN.
S1.u[1] value is a quiet NAN.
S1.u[2] value is negative infinity.
S1.u[3] value is a negative normal value.
S1.u[4] value is a negative denormal value.
S1.u[5] value is negative zero.
S1.u[6] value is positive zero.
S1.u[7] value is a positive denormal value.
S1.u[8] value is a positive normal value.
S1.u[9] value is positive infinity.

```
  declare result : 1'U;
  if isSignalNAN(64'F(S0.f32)) then
      result = S1.u32[0]
  elsif isQuietNAN(64'F(S0.f32)) then
      result = S1.u32[1]
  elsif exponent(S0.f32) == 255 then
      // +-INF
      result = S1.u32[sign(S0.f32) ? 2 : 9]
  elsif exponent(S0.f32) > 0 then
      // +-normal value
      result = S1.u32[sign(S0.f32) ? 3 : 8]
  elsif 64'F(abs(S0.f32)) > 0.0 then
      // +-denormal value
      result = S1.u32[sign(S0.f32) ? 4 : 7]
  else
      // +-0.0
      result = S1.u32[sign(S0.f32) ? 5 : 6]
  endif;
  EXEC.u64[laneId] = D0.u64[laneId] = result
```

#### V_CMP_CLASS_F64  (opcode 18)

Evaluate the IEEE numeric class function specified as a 10 bit mask in the second input on the first input, a
double-precision float, and set the per-lane condition code to the result. Store the result into VCC or a scalar
register.

The function reports true if the floating point value is any of the numeric types selected in the 10 bit mask
according to the following list:

S1.u[0] value is a signaling NAN.
S1.u[1] value is a quiet NAN.
S1.u[2] value is negative infinity.
S1.u[3] value is a negative normal value.
S1.u[4] value is a negative denormal value.
S1.u[5] value is negative zero.
S1.u[6] value is positive zero.
S1.u[7] value is a positive denormal value.
S1.u[8] value is a positive normal value.
S1.u[9] value is positive infinity.

```
  declare result : 1'U;
  if isSignalNAN(S0.f64) then
      result = S1.u32[0]
  elsif isQuietNAN(S0.f64) then
      result = S1.u32[1]
  elsif exponent(S0.f64) == 2047 then
      // +-INF
      result = S1.u32[sign(S0.f64) ? 2 : 9]
  elsif exponent(S0.f64) > 0 then
      // +-normal value
      result = S1.u32[sign(S0.f64) ? 3 : 8]
  elsif abs(S0.f64) > 0.0 then
      // +-denormal value
      result = S1.u32[sign(S0.f64) ? 4 : 7]
  else
      // +-0.0
      result = S1.u32[sign(S0.f64) ? 5 : 6]
  endif;
  D0.u64[laneId] = result;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_CLASS_F64  (opcode 19)

Evaluate the IEEE numeric class function specified as a 10 bit mask in the second input on the first input, a
double-precision float, and set the per-lane condition code to the result. Store the result into the EXEC mask
and to VCC or a scalar register.

The function reports true if the floating point value is any of the numeric types selected in the 10 bit mask
according to the following list:

S1.u[0] value is a signaling NAN.
S1.u[1] value is a quiet NAN.
S1.u[2] value is negative infinity.
S1.u[3] value is a negative normal value.
S1.u[4] value is a negative denormal value.
S1.u[5] value is negative zero.
S1.u[6] value is positive zero.
S1.u[7] value is a positive denormal value.
S1.u[8] value is a positive normal value.
S1.u[9] value is positive infinity.

```
  declare result : 1'U;
  if isSignalNAN(S0.f64) then
      result = S1.u32[0]
  elsif isQuietNAN(S0.f64) then
      result = S1.u32[1]
  elsif exponent(S0.f64) == 2047 then
      // +-INF
      result = S1.u32[sign(S0.f64) ? 2 : 9]
  elsif exponent(S0.f64) > 0 then
      // +-normal value
```

```
      result = S1.u32[sign(S0.f64) ? 3 : 8]
  elsif abs(S0.f64) > 0.0 then
      // +-denormal value
      result = S1.u32[sign(S0.f64) ? 4 : 7]
  else
      // +-0.0
      result = S1.u32[sign(S0.f64) ? 5 : 6]
  endif;
  EXEC.u64[laneId] = D0.u64[laneId] = result
```

#### V_CMP_CLASS_F16  (opcode 20)

Evaluate the IEEE numeric class function specified as a 10 bit mask in the second input on the first input, a
half-precision float, and set the per-lane condition code to the result. Store the result into VCC or a scalar
register.

The function reports true if the floating point value is any of the numeric types selected in the 10 bit mask
according to the following list:

S1.u[0] value is a signaling NAN.
S1.u[1] value is a quiet NAN.
S1.u[2] value is negative infinity.
S1.u[3] value is a negative normal value.
S1.u[4] value is a negative denormal value.
S1.u[5] value is negative zero.
S1.u[6] value is positive zero.
S1.u[7] value is a positive denormal value.
S1.u[8] value is a positive normal value.
S1.u[9] value is positive infinity.

```
  declare result : 1'U;
  if isSignalNAN(64'F(S0.f16)) then
      result = S1.u32[0]
  elsif isQuietNAN(64'F(S0.f16)) then
      result = S1.u32[1]
  elsif exponent(S0.f16) == 31 then
      // +-INF
      result = S1.u32[sign(S0.f16) ? 2 : 9]
  elsif exponent(S0.f16) > 0 then
      // +-normal value
      result = S1.u32[sign(S0.f16) ? 3 : 8]
  elsif 64'F(abs(S0.f16)) > 0.0 then
      // +-denormal value
      result = S1.u32[sign(S0.f16) ? 4 : 7]
  else
      // +-0.0
      result = S1.u32[sign(S0.f16) ? 5 : 6]
  endif;
  D0.u64[laneId] = result;
  // D0 = VCC in VOPC encoding.
```

Notes

Note that the S1 has a format of f16 since floating point literal constants are interpreted as 16 bit value for this
opcode.

#### V_CMPX_CLASS_F16  (opcode 21)

Evaluate the IEEE numeric class function specified as a 10 bit mask in the second input on the first input, a
half-precision float, and set the per-lane condition code to the result. Store the result into the EXEC mask and
to VCC or a scalar register.

The function reports true if the floating point value is any of the numeric types selected in the 10 bit mask
according to the following list:

S1.u[0] value is a signaling NAN.
S1.u[1] value is a quiet NAN.
S1.u[2] value is negative infinity.
S1.u[3] value is a negative normal value.
S1.u[4] value is a negative denormal value.
S1.u[5] value is negative zero.
S1.u[6] value is positive zero.
S1.u[7] value is a positive denormal value.
S1.u[8] value is a positive normal value.
S1.u[9] value is positive infinity.

```
  declare result : 1'U;
  if isSignalNAN(64'F(S0.f16)) then
        result = S1.u32[0]
  elsif isQuietNAN(64'F(S0.f16)) then
        result = S1.u32[1]
  elsif exponent(S0.f16) == 31 then
        // +-INF
        result = S1.u32[sign(S0.f16) ? 2 : 9]
  elsif exponent(S0.f16) > 0 then
        // +-normal value
        result = S1.u32[sign(S0.f16) ? 3 : 8]
  elsif 64'F(abs(S0.f16)) > 0.0 then
        // +-denormal value
        result = S1.u32[sign(S0.f16) ? 4 : 7]
  else
        // +-0.0
        result = S1.u32[sign(S0.f16) ? 5 : 6]
  endif;
  EXEC.u64[laneId] = D0.u64[laneId] = result
```

Notes

Note that the S1 has a format of f16 since floating point literal constants are interpreted as 16 bit value for this
opcode.

#### V_CMP_F_F16  (opcode 32)

Set the per-lane condition code to 0. Store the result into VCC or a scalar register.

```
  D0.u64[laneId] = 1'0U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_LT_F16  (opcode 33)

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into VCC or a
scalar register.

```
  D0.u64[laneId] = S0.f16 < S1.f16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_EQ_F16  (opcode 34)

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into VCC or a
scalar register.

```
  D0.u64[laneId] = S0.f16 == S1.f16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_LE_F16  (opcode 35)

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into VCC or a scalar register.

```
  D0.u64[laneId] = S0.f16 <= S1.f16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_GT_F16  (opcode 36)

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into VCC
or a scalar register.

```
  D0.u64[laneId] = S0.f16 > S1.f16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_LG_F16  (opcode 37)

Set the per-lane condition code to 1 iff the first input is less than or greater than the second input. Store the
result into VCC or a scalar register.

```
  D0.u64[laneId] = S0.f16 <> S1.f16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_GE_F16  (opcode 38)

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into VCC or a scalar register.

```
  D0.u64[laneId] = S0.f16 >= S1.f16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_O_F16  (opcode 39)

Set the per-lane condition code to 1 iff the first input is orderable to the second input. Store the result into VCC
or a scalar register.

```
  D0.u64[laneId] = (!isNAN(64'F(S0.f16)) && !isNAN(64'F(S1.f16)));
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_U_F16  (opcode 40)

Set the per-lane condition code to 1 iff the first input is not orderable to the second input. Store the result into
VCC or a scalar register.

```
  D0.u64[laneId] = (isNAN(64'F(S0.f16)) || isNAN(64'F(S1.f16)));
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_NGE_F16  (opcode 41)

Set the per-lane condition code to 1 iff the first input is not greater than or equal to the second input. Store the
result into VCC or a scalar register.

```
  D0.u64[laneId] = !(S0.f16 >= S1.f16);
  // With NAN inputs this is not the same operation as <
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_NLG_F16  (opcode 42)

Set the per-lane condition code to 1 iff the first input is not less than or greater than the second input. Store the
result into VCC or a scalar register.

```
  D0.u64[laneId] = !(S0.f16 <> S1.f16);
  // With NAN inputs this is not the same operation as ==
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_NGT_F16  (opcode 43)

Set the per-lane condition code to 1 iff the first input is not greater than the second input. Store the result into
VCC or a scalar register.

```
  D0.u64[laneId] = !(S0.f16 > S1.f16);
  // With NAN inputs this is not the same operation as <=
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_NLE_F16  (opcode 44)

Set the per-lane condition code to 1 iff the first input is not less than or equal to the second input. Store the
result into VCC or a scalar register.

```
  D0.u64[laneId] = !(S0.f16 <= S1.f16);
  // With NAN inputs this is not the same operation as >
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_NEQ_F16  (opcode 45)

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into VCC
or a scalar register.

```
  D0.u64[laneId] = !(S0.f16 == S1.f16);
  // With NAN inputs this is not the same operation as !=
```

```
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_NLT_F16  (opcode 46)

Set the per-lane condition code to 1 iff the first input is not less than the second input. Store the result into VCC
or a scalar register.

```
  D0.u64[laneId] = !(S0.f16 < S1.f16);
  // With NAN inputs this is not the same operation as >=
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_TRU_F16  (opcode 47)

Set the per-lane condition code to 1. Store the result into VCC or a scalar register.

```
  D0.u64[laneId] = 1'1U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_F_F16  (opcode 48)

Set the per-lane condition code to 0. Store the result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = 1'0U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_LT_F16  (opcode 49)

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into the
EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.f16 < S1.f16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_EQ_F16  (opcode 50)

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into the EXEC
mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.f16 == S1.f16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_LE_F16  (opcode 51)

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.f16 <= S1.f16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_GT_F16  (opcode 52)

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into the
EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.f16 > S1.f16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_LG_F16  (opcode 53)

Set the per-lane condition code to 1 iff the first input is less than or greater than the second input. Store the
result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.f16 <> S1.f16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_GE_F16  (opcode 54)

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.f16 >= S1.f16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_O_F16  (opcode 55)

Set the per-lane condition code to 1 iff the first input is orderable to the second input. Store the result into the
EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = (!isNAN(64'F(S0.f16)) && !isNAN(64'F(S1.f16)));
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_U_F16  (opcode 56)

Set the per-lane condition code to 1 iff the first input is not orderable to the second input. Store the result into
the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = (isNAN(64'F(S0.f16)) || isNAN(64'F(S1.f16)));
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_NGE_F16  (opcode 57)

Set the per-lane condition code to 1 iff the first input is not greater than or equal to the second input. Store the
result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = !(S0.f16 >= S1.f16);
  // With NAN inputs this is not the same operation as <
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_NLG_F16  (opcode 58)

Set the per-lane condition code to 1 iff the first input is not less than or greater than the second input. Store the
result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = !(S0.f16 <> S1.f16);
  // With NAN inputs this is not the same operation as ==
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_NGT_F16  (opcode 59)

Set the per-lane condition code to 1 iff the first input is not greater than the second input. Store the result into
the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = !(S0.f16 > S1.f16);
  // With NAN inputs this is not the same operation as <=
```

```
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_NLE_F16  (opcode 60)

Set the per-lane condition code to 1 iff the first input is not less than or equal to the second input. Store the
result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = !(S0.f16 <= S1.f16);
  // With NAN inputs this is not the same operation as >
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_NEQ_F16  (opcode 61)

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into the
EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = !(S0.f16 == S1.f16);
  // With NAN inputs this is not the same operation as !=
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_NLT_F16  (opcode 62)

Set the per-lane condition code to 1 iff the first input is not less than the second input. Store the result into the
EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = !(S0.f16 < S1.f16);
  // With NAN inputs this is not the same operation as >=
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_TRU_F16  (opcode 63)

Set the per-lane condition code to 1. Store the result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = 1'1U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_F_F32  (opcode 64)

Set the per-lane condition code to 0. Store the result into VCC or a scalar register.

```
  D0.u64[laneId] = 1'0U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_LT_F32  (opcode 65)

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into VCC or a
scalar register.

```
  D0.u64[laneId] = S0.f32 < S1.f32;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_EQ_F32  (opcode 66)

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into VCC or a
scalar register.

```
  D0.u64[laneId] = S0.f32 == S1.f32;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_LE_F32  (opcode 67)

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into VCC or a scalar register.

```
  D0.u64[laneId] = S0.f32 <= S1.f32;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_GT_F32  (opcode 68)

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into VCC
or a scalar register.

```
  D0.u64[laneId] = S0.f32 > S1.f32;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_LG_F32  (opcode 69)

Set the per-lane condition code to 1 iff the first input is less than or greater than the second input. Store the
result into VCC or a scalar register.

```
  D0.u64[laneId] = S0.f32 <> S1.f32;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_GE_F32  (opcode 70)

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into VCC or a scalar register.

```
  D0.u64[laneId] = S0.f32 >= S1.f32;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_O_F32  (opcode 71)

Set the per-lane condition code to 1 iff the first input is orderable to the second input. Store the result into VCC
or a scalar register.

```
  D0.u64[laneId] = (!isNAN(64'F(S0.f32)) && !isNAN(64'F(S1.f32)));
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_U_F32  (opcode 72)

Set the per-lane condition code to 1 iff the first input is not orderable to the second input. Store the result into
VCC or a scalar register.

```
  D0.u64[laneId] = (isNAN(64'F(S0.f32)) || isNAN(64'F(S1.f32)));
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_NGE_F32  (opcode 73)

Set the per-lane condition code to 1 iff the first input is not greater than or equal to the second input. Store the
result into VCC or a scalar register.

```
  D0.u64[laneId] = !(S0.f32 >= S1.f32);
```

```
  // With NAN inputs this is not the same operation as <
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_NLG_F32  (opcode 74)

Set the per-lane condition code to 1 iff the first input is not less than or greater than the second input. Store the
result into VCC or a scalar register.

```
  D0.u64[laneId] = !(S0.f32 <> S1.f32);
  // With NAN inputs this is not the same operation as ==
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_NGT_F32  (opcode 75)

Set the per-lane condition code to 1 iff the first input is not greater than the second input. Store the result into
VCC or a scalar register.

```
  D0.u64[laneId] = !(S0.f32 > S1.f32);
  // With NAN inputs this is not the same operation as <=
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_NLE_F32  (opcode 76)

Set the per-lane condition code to 1 iff the first input is not less than or equal to the second input. Store the
result into VCC or a scalar register.

```
  D0.u64[laneId] = !(S0.f32 <= S1.f32);
  // With NAN inputs this is not the same operation as >
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_NEQ_F32  (opcode 77)

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into VCC
or a scalar register.

```
  D0.u64[laneId] = !(S0.f32 == S1.f32);
  // With NAN inputs this is not the same operation as !=
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_NLT_F32  (opcode 78)

Set the per-lane condition code to 1 iff the first input is not less than the second input. Store the result into VCC
or a scalar register.

```
  D0.u64[laneId] = !(S0.f32 < S1.f32);
  // With NAN inputs this is not the same operation as >=
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_TRU_F32  (opcode 79)

Set the per-lane condition code to 1. Store the result into VCC or a scalar register.

```
  D0.u64[laneId] = 1'1U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_F_F32  (opcode 80)

Set the per-lane condition code to 0. Store the result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = 1'0U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_LT_F32  (opcode 81)

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into the
EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.f32 < S1.f32;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_EQ_F32  (opcode 82)

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into the EXEC
mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.f32 == S1.f32;
```

```
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_LE_F32  (opcode 83)

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.f32 <= S1.f32;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_GT_F32  (opcode 84)

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into the
EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.f32 > S1.f32;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_LG_F32  (opcode 85)

Set the per-lane condition code to 1 iff the first input is less than or greater than the second input. Store the
result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.f32 <> S1.f32;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_GE_F32  (opcode 86)

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.f32 >= S1.f32;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_O_F32  (opcode 87)

Set the per-lane condition code to 1 iff the first input is orderable to the second input. Store the result into the

EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = (!isNAN(64'F(S0.f32)) && !isNAN(64'F(S1.f32)));
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_U_F32  (opcode 88)

Set the per-lane condition code to 1 iff the first input is not orderable to the second input. Store the result into
the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = (isNAN(64'F(S0.f32)) || isNAN(64'F(S1.f32)));
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_NGE_F32  (opcode 89)

Set the per-lane condition code to 1 iff the first input is not greater than or equal to the second input. Store the
result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = !(S0.f32 >= S1.f32);
  // With NAN inputs this is not the same operation as <
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_NLG_F32  (opcode 90)

Set the per-lane condition code to 1 iff the first input is not less than or greater than the second input. Store the
result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = !(S0.f32 <> S1.f32);
  // With NAN inputs this is not the same operation as ==
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_NGT_F32  (opcode 91)

Set the per-lane condition code to 1 iff the first input is not greater than the second input. Store the result into
the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = !(S0.f32 > S1.f32);
  // With NAN inputs this is not the same operation as <=
```

```
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_NLE_F32  (opcode 92)

Set the per-lane condition code to 1 iff the first input is not less than or equal to the second input. Store the
result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = !(S0.f32 <= S1.f32);
  // With NAN inputs this is not the same operation as >
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_NEQ_F32  (opcode 93)

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into the
EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = !(S0.f32 == S1.f32);
  // With NAN inputs this is not the same operation as !=
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_NLT_F32  (opcode 94)

Set the per-lane condition code to 1 iff the first input is not less than the second input. Store the result into the
EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = !(S0.f32 < S1.f32);
  // With NAN inputs this is not the same operation as >=
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_TRU_F32  (opcode 95)

Set the per-lane condition code to 1. Store the result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = 1'1U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_F_F64  (opcode 96)

Set the per-lane condition code to 0. Store the result into VCC or a scalar register.

```
  D0.u64[laneId] = 1'0U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_LT_F64  (opcode 97)

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into VCC or a
scalar register.

```
  D0.u64[laneId] = S0.f64 < S1.f64;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_EQ_F64  (opcode 98)

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into VCC or a
scalar register.

```
  D0.u64[laneId] = S0.f64 == S1.f64;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_LE_F64  (opcode 99)

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into VCC or a scalar register.

```
  D0.u64[laneId] = S0.f64 <= S1.f64;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_GT_F64  (opcode 100)

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into VCC
or a scalar register.

```
  D0.u64[laneId] = S0.f64 > S1.f64;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_LG_F64  (opcode 101)

Set the per-lane condition code to 1 iff the first input is less than or greater than the second input. Store the
result into VCC or a scalar register.

```
  D0.u64[laneId] = S0.f64 <> S1.f64;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_GE_F64  (opcode 102)

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into VCC or a scalar register.

```
  D0.u64[laneId] = S0.f64 >= S1.f64;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_O_F64  (opcode 103)

Set the per-lane condition code to 1 iff the first input is orderable to the second input. Store the result into VCC
or a scalar register.

```
  D0.u64[laneId] = (!isNAN(S0.f64) && !isNAN(S1.f64));
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_U_F64  (opcode 104)

Set the per-lane condition code to 1 iff the first input is not orderable to the second input. Store the result into
VCC or a scalar register.

```
  D0.u64[laneId] = (isNAN(S0.f64) || isNAN(S1.f64));
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_NGE_F64  (opcode 105)

Set the per-lane condition code to 1 iff the first input is not greater than or equal to the second input. Store the
result into VCC or a scalar register.

```
  D0.u64[laneId] = !(S0.f64 >= S1.f64);
```

```
  // With NAN inputs this is not the same operation as <
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_NLG_F64  (opcode 106)

Set the per-lane condition code to 1 iff the first input is not less than or greater than the second input. Store the
result into VCC or a scalar register.

```
  D0.u64[laneId] = !(S0.f64 <> S1.f64);
  // With NAN inputs this is not the same operation as ==
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_NGT_F64  (opcode 107)

Set the per-lane condition code to 1 iff the first input is not greater than the second input. Store the result into
VCC or a scalar register.

```
  D0.u64[laneId] = !(S0.f64 > S1.f64);
  // With NAN inputs this is not the same operation as <=
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_NLE_F64  (opcode 108)

Set the per-lane condition code to 1 iff the first input is not less than or equal to the second input. Store the
result into VCC or a scalar register.

```
  D0.u64[laneId] = !(S0.f64 <= S1.f64);
  // With NAN inputs this is not the same operation as >
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_NEQ_F64  (opcode 109)

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into VCC
or a scalar register.

```
  D0.u64[laneId] = !(S0.f64 == S1.f64);
  // With NAN inputs this is not the same operation as !=
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_NLT_F64  (opcode 110)

Set the per-lane condition code to 1 iff the first input is not less than the second input. Store the result into VCC
or a scalar register.

```
  D0.u64[laneId] = !(S0.f64 < S1.f64);
  // With NAN inputs this is not the same operation as >=
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_TRU_F64  (opcode 111)

Set the per-lane condition code to 1. Store the result into VCC or a scalar register.

```
  D0.u64[laneId] = 1'1U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_F_F64  (opcode 112)

Set the per-lane condition code to 0. Store the result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = 1'0U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_LT_F64  (opcode 113)

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into the
EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.f64 < S1.f64;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_EQ_F64  (opcode 114)

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into the EXEC
mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.f64 == S1.f64;
```

```
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_LE_F64  (opcode 115)

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.f64 <= S1.f64;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_GT_F64  (opcode 116)

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into the
EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.f64 > S1.f64;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_LG_F64  (opcode 117)

Set the per-lane condition code to 1 iff the first input is less than or greater than the second input. Store the
result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.f64 <> S1.f64;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_GE_F64  (opcode 118)

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.f64 >= S1.f64;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_O_F64  (opcode 119)

Set the per-lane condition code to 1 iff the first input is orderable to the second input. Store the result into the

EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = (!isNAN(S0.f64) && !isNAN(S1.f64));
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_U_F64  (opcode 120)

Set the per-lane condition code to 1 iff the first input is not orderable to the second input. Store the result into
the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = (isNAN(S0.f64) || isNAN(S1.f64));
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_NGE_F64  (opcode 121)

Set the per-lane condition code to 1 iff the first input is not greater than or equal to the second input. Store the
result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = !(S0.f64 >= S1.f64);
  // With NAN inputs this is not the same operation as <
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_NLG_F64  (opcode 122)

Set the per-lane condition code to 1 iff the first input is not less than or greater than the second input. Store the
result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = !(S0.f64 <> S1.f64);
  // With NAN inputs this is not the same operation as ==
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_NGT_F64  (opcode 123)

Set the per-lane condition code to 1 iff the first input is not greater than the second input. Store the result into
the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = !(S0.f64 > S1.f64);
  // With NAN inputs this is not the same operation as <=
```

```
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_NLE_F64  (opcode 124)

Set the per-lane condition code to 1 iff the first input is not less than or equal to the second input. Store the
result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = !(S0.f64 <= S1.f64);
  // With NAN inputs this is not the same operation as >
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_NEQ_F64  (opcode 125)

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into the
EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = !(S0.f64 == S1.f64);
  // With NAN inputs this is not the same operation as !=
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_NLT_F64  (opcode 126)

Set the per-lane condition code to 1 iff the first input is not less than the second input. Store the result into the
EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = !(S0.f64 < S1.f64);
  // With NAN inputs this is not the same operation as >=
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_TRU_F64  (opcode 127)

Set the per-lane condition code to 1. Store the result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = 1'1U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_F_I16  (opcode 160)

Set the per-lane condition code to 0. Store the result into VCC or a scalar register.

```
  D0.u64[laneId] = 1'0U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_LT_I16  (opcode 161)

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into VCC or a
scalar register.

```
  D0.u64[laneId] = S0.i16 < S1.i16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_EQ_I16  (opcode 162)

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into VCC or a
scalar register.

```
  D0.u64[laneId] = S0.i16 == S1.i16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_LE_I16  (opcode 163)

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into VCC or a scalar register.

```
  D0.u64[laneId] = S0.i16 <= S1.i16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_GT_I16  (opcode 164)

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into VCC
or a scalar register.

```
  D0.u64[laneId] = S0.i16 > S1.i16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_NE_I16  (opcode 165)

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into VCC
or a scalar register.

```
  D0.u64[laneId] = S0.i16 <> S1.i16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_GE_I16  (opcode 166)

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into VCC or a scalar register.

```
  D0.u64[laneId] = S0.i16 >= S1.i16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_T_I16  (opcode 167)

Set the per-lane condition code to 1. Store the result into VCC or a scalar register.

```
  D0.u64[laneId] = 1'1U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_F_U16  (opcode 168)

Set the per-lane condition code to 0. Store the result into VCC or a scalar register.

```
  D0.u64[laneId] = 1'0U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_LT_U16  (opcode 169)

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into VCC or a
scalar register.

```
  D0.u64[laneId] = S0.u16 < S1.u16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_EQ_U16  (opcode 170)

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into VCC or a
scalar register.

```
  D0.u64[laneId] = S0.u16 == S1.u16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_LE_U16  (opcode 171)

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into VCC or a scalar register.

```
  D0.u64[laneId] = S0.u16 <= S1.u16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_GT_U16  (opcode 172)

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into VCC
or a scalar register.

```
  D0.u64[laneId] = S0.u16 > S1.u16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_NE_U16  (opcode 173)

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into VCC
or a scalar register.

```
  D0.u64[laneId] = S0.u16 <> S1.u16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_GE_U16  (opcode 174)

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into VCC or a scalar register.

```
  D0.u64[laneId] = S0.u16 >= S1.u16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_T_U16  (opcode 175)

Set the per-lane condition code to 1. Store the result into VCC or a scalar register.

```
  D0.u64[laneId] = 1'1U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_F_I16  (opcode 176)

Set the per-lane condition code to 0. Store the result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = 1'0U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_LT_I16  (opcode 177)

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into the
EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.i16 < S1.i16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_EQ_I16  (opcode 178)

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into the EXEC
mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.i16 == S1.i16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_LE_I16  (opcode 179)

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result

into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.i16 <= S1.i16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_GT_I16  (opcode 180)

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into the
EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.i16 > S1.i16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_NE_I16  (opcode 181)

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into the
EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.i16 <> S1.i16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_GE_I16  (opcode 182)

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.i16 >= S1.i16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_T_I16  (opcode 183)

Set the per-lane condition code to 1. Store the result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = 1'1U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_F_U16  (opcode 184)

Set the per-lane condition code to 0. Store the result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = 1'0U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_LT_U16  (opcode 185)

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into the
EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.u16 < S1.u16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_EQ_U16  (opcode 186)

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into the EXEC
mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.u16 == S1.u16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_LE_U16  (opcode 187)

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.u16 <= S1.u16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_GT_U16  (opcode 188)

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into the
EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.u16 > S1.u16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_NE_U16  (opcode 189)

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into the
EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.u16 <> S1.u16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_GE_U16  (opcode 190)

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.u16 >= S1.u16;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_T_U16  (opcode 191)

Set the per-lane condition code to 1. Store the result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = 1'1U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_F_I32  (opcode 192)

Set the per-lane condition code to 0. Store the result into VCC or a scalar register.

```
  D0.u64[laneId] = 1'0U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_LT_I32  (opcode 193)

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into VCC or a
scalar register.

```
  D0.u64[laneId] = S0.i32 < S1.i32;
```

```
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_EQ_I32  (opcode 194)

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into VCC or a
scalar register.

```
  D0.u64[laneId] = S0.i32 == S1.i32;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_LE_I32  (opcode 195)

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into VCC or a scalar register.

```
  D0.u64[laneId] = S0.i32 <= S1.i32;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_GT_I32  (opcode 196)

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into VCC
or a scalar register.

```
  D0.u64[laneId] = S0.i32 > S1.i32;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_NE_I32  (opcode 197)

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into VCC
or a scalar register.

```
  D0.u64[laneId] = S0.i32 <> S1.i32;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_GE_I32  (opcode 198)

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the

result into VCC or a scalar register.

```
  D0.u64[laneId] = S0.i32 >= S1.i32;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_T_I32  (opcode 199)

Set the per-lane condition code to 1. Store the result into VCC or a scalar register.

```
  D0.u64[laneId] = 1'1U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_F_U32  (opcode 200)

Set the per-lane condition code to 0. Store the result into VCC or a scalar register.

```
  D0.u64[laneId] = 1'0U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_LT_U32  (opcode 201)

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into VCC or a
scalar register.

```
  D0.u64[laneId] = S0.u32 < S1.u32;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_EQ_U32  (opcode 202)

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into VCC or a
scalar register.

```
  D0.u64[laneId] = S0.u32 == S1.u32;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_LE_U32  (opcode 203)

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into VCC or a scalar register.

```
  D0.u64[laneId] = S0.u32 <= S1.u32;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_GT_U32  (opcode 204)

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into VCC
or a scalar register.

```
  D0.u64[laneId] = S0.u32 > S1.u32;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_NE_U32  (opcode 205)

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into VCC
or a scalar register.

```
  D0.u64[laneId] = S0.u32 <> S1.u32;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_GE_U32  (opcode 206)

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into VCC or a scalar register.

```
  D0.u64[laneId] = S0.u32 >= S1.u32;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_T_U32  (opcode 207)

Set the per-lane condition code to 1. Store the result into VCC or a scalar register.

```
  D0.u64[laneId] = 1'1U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_F_I32  (opcode 208)

Set the per-lane condition code to 0. Store the result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = 1'0U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_LT_I32  (opcode 209)

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into the
EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.i32 < S1.i32;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_EQ_I32  (opcode 210)

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into the EXEC
mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.i32 == S1.i32;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_LE_I32  (opcode 211)

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.i32 <= S1.i32;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_GT_I32  (opcode 212)

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into the
EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.i32 > S1.i32;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_NE_I32  (opcode 213)

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into the
EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.i32 <> S1.i32;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_GE_I32  (opcode 214)

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.i32 >= S1.i32;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_T_I32  (opcode 215)

Set the per-lane condition code to 1. Store the result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = 1'1U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_F_U32  (opcode 216)

Set the per-lane condition code to 0. Store the result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = 1'0U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_LT_U32  (opcode 217)

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into the
EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.u32 < S1.u32;
```

```
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_EQ_U32  (opcode 218)

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into the EXEC
mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.u32 == S1.u32;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_LE_U32  (opcode 219)

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.u32 <= S1.u32;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_GT_U32  (opcode 220)

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into the
EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.u32 > S1.u32;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_NE_U32  (opcode 221)

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into the
EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.u32 <> S1.u32;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_GE_U32  (opcode 222)

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the

result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.u32 >= S1.u32;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_T_U32  (opcode 223)

Set the per-lane condition code to 1. Store the result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = 1'1U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_F_I64  (opcode 224)

Set the per-lane condition code to 0. Store the result into VCC or a scalar register.

```
  D0.u64[laneId] = 1'0U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_LT_I64  (opcode 225)

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into VCC or a
scalar register.

```
  D0.u64[laneId] = S0.i64 < S1.i64;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_EQ_I64  (opcode 226)

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into VCC or a
scalar register.

```
  D0.u64[laneId] = S0.i64 == S1.i64;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_LE_I64  (opcode 227)

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into VCC or a scalar register.

```
  D0.u64[laneId] = S0.i64 <= S1.i64;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_GT_I64  (opcode 228)

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into VCC
or a scalar register.

```
  D0.u64[laneId] = S0.i64 > S1.i64;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_NE_I64  (opcode 229)

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into VCC
or a scalar register.

```
  D0.u64[laneId] = S0.i64 <> S1.i64;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_GE_I64  (opcode 230)

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into VCC or a scalar register.

```
  D0.u64[laneId] = S0.i64 >= S1.i64;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_T_I64  (opcode 231)

Set the per-lane condition code to 1. Store the result into VCC or a scalar register.

```
  D0.u64[laneId] = 1'1U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_F_U64  (opcode 232)

Set the per-lane condition code to 0. Store the result into VCC or a scalar register.

```
  D0.u64[laneId] = 1'0U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_LT_U64  (opcode 233)

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into VCC or a
scalar register.

```
  D0.u64[laneId] = S0.u64 < S1.u64;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_EQ_U64  (opcode 234)

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into VCC or a
scalar register.

```
  D0.u64[laneId] = S0.u64 == S1.u64;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_LE_U64  (opcode 235)

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into VCC or a scalar register.

```
  D0.u64[laneId] = S0.u64 <= S1.u64;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_GT_U64  (opcode 236)

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into VCC
or a scalar register.

```
  D0.u64[laneId] = S0.u64 > S1.u64;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_NE_U64  (opcode 237)

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into VCC
or a scalar register.

```
  D0.u64[laneId] = S0.u64 <> S1.u64;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_GE_U64  (opcode 238)

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into VCC or a scalar register.

```
  D0.u64[laneId] = S0.u64 >= S1.u64;
  // D0 = VCC in VOPC encoding.
```

#### V_CMP_T_U64  (opcode 239)

Set the per-lane condition code to 1. Store the result into VCC or a scalar register.

```
  D0.u64[laneId] = 1'1U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_F_I64  (opcode 240)

Set the per-lane condition code to 0. Store the result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = 1'0U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_LT_I64  (opcode 241)

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into the
EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.i64 < S1.i64;
```

```
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_EQ_I64  (opcode 242)

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into the EXEC
mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.i64 == S1.i64;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_LE_I64  (opcode 243)

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.i64 <= S1.i64;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_GT_I64  (opcode 244)

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into the
EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.i64 > S1.i64;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_NE_I64  (opcode 245)

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into the
EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.i64 <> S1.i64;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_GE_I64  (opcode 246)

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the

result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.i64 >= S1.i64;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_T_I64  (opcode 247)

Set the per-lane condition code to 1. Store the result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = 1'1U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_F_U64  (opcode 248)

Set the per-lane condition code to 0. Store the result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = 1'0U;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_LT_U64  (opcode 249)

Set the per-lane condition code to 1 iff the first input is less than the second input. Store the result into the
EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.u64 < S1.u64;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_EQ_U64  (opcode 250)

Set the per-lane condition code to 1 iff the first input is equal to the second input. Store the result into the EXEC
mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.u64 == S1.u64;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_LE_U64  (opcode 251)

Set the per-lane condition code to 1 iff the first input is less than or equal to the second input. Store the result
into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.u64 <= S1.u64;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_GT_U64  (opcode 252)

Set the per-lane condition code to 1 iff the first input is greater than the second input. Store the result into the
EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.u64 > S1.u64;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_NE_U64  (opcode 253)

Set the per-lane condition code to 1 iff the first input is not equal to the second input. Store the result into the
EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.u64 <> S1.u64;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_GE_U64  (opcode 254)

Set the per-lane condition code to 1 iff the first input is greater than or equal to the second input. Store the
result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = S0.u64 >= S1.u64;
  // D0 = VCC in VOPC encoding.
```

#### V_CMPX_T_U64  (opcode 255)

Set the per-lane condition code to 1. Store the result into the EXEC mask and to VCC or a scalar register.

```
  EXEC.u64[laneId] = D0.u64[laneId] = 1'1U;
  // D0 = VCC in VOPC encoding.
```

#### 12.9.1. VOPC using VOP3A encoding

Instructions in this format may also be encoded as VOP3A. This allows access to the extra control bits (e.g. ABS,
OMOD) in exchange for not being able to use a literal constant. The VOP3 opcode is: VOP2 opcode + 0x000.

When the CLAMP microcode bit is set to 1, these compare instructions signal an exception when either of the
inputs is NaN. When CLAMP is set to zero, NaN does not signal an exception. The second eight VOPC
instructions have {OP8} embedded in them. This refers to each of the compare operations listed below.

```
  where:
```

```
    VDST = Destination for instruction in the VGPR.
    ABS = Floating-point absolute value.
    CLMP = Clamp output.
    OP = Instructions.
    SRC0 = First operand for instruction.
    SRC1 = Second operand for instruction.
    SRC2 = Third operand for instruction. Unused in VOPC instructions.
    OMOD = Output modifier for instruction. Unused in VOPC instructions.
    NEG = Floating-point negation.
```
