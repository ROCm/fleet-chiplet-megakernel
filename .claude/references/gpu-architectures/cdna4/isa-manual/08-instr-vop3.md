# CDNA4 ISA Instructions: VOP3A & VOP3B

> Part of the **CDNA4 Instruction Set Architecture** reference. See the [index](README.md) for other sections.

## Sections in this file

  - [12.11. VOP3A & VOP3B Instructions](#1211-vop3a-vop3b-instructions)

## Instruction mnemonics in this file

- **12.11. VOP3A & VOP3B Instructions**: V_NOP, V_MOV_B32, V_READFIRSTLANE_B32, V_CVT_I32_F64, V_CVT_F64_I32, V_CVT_F32_I32, V_CVT_F32_U32, V_CVT_U32_F32, V_CVT_I32_F32, V_CVT_F16_F32, V_CVT_F32_F16, V_CVT_RPI_I32_F32, V_CVT_FLR_I32_F32, V_CVT_OFF_F32_I4, V_CVT_F32_F64, V_CVT_F64_F32, V_CVT_F32_UBYTE0, V_CVT_F32_UBYTE1, V_CVT_F32_UBYTE2, V_CVT_F32_UBYTE3, V_CVT_U32_F64, V_CVT_F64_U32, V_TRUNC_F64, V_CEIL_F64, V_RNDNE_F64, V_FLOOR_F64, V_FRACT_F32, V_TRUNC_F32, V_CEIL_F32, V_RNDNE_F32, V_FLOOR_F32, V_EXP_F32, V_LOG_F32, V_RCP_F32, V_RCP_IFLAG_F32, V_RSQ_F32, V_RCP_F64, V_RSQ_F64, V_SQRT_F32, V_SQRT_F64, V_SIN_F32, V_COS_F32, V_NOT_B32, V_BFREV_B32, V_FFBH_U32, V_FFBL_B32, V_FFBH_I32, V_FREXP_EXP_I32_F64, V_FREXP_MANT_F64, V_FRACT_F64, V_FREXP_EXP_I32_F32, V_FREXP_MANT_F32, V_CLREXCP, V_MOV_B64, V_CVT_F16_U16, V_CVT_F16_I16, V_CVT_U16_F16, V_CVT_I16_F16, V_RCP_F16, V_SQRT_F16, V_RSQ_F16, V_LOG_F16, V_EXP_F16, V_FREXP_MANT_F16, V_FREXP_EXP_I16_F16, V_FLOOR_F16, V_CEIL_F16, V_TRUNC_F16, V_RNDNE_F16, V_FRACT_F16, V_SIN_F16, V_COS_F16, V_CVT_NORM_I16_F16, V_CVT_NORM_U16_F16, V_SAT_PK_U8_I16, V_SWAP_B32, V_ACCVGPR_MOV_B32, V_CVT_F32_FP8, V_CVT_F32_BF8, V_CVT_PK_F32_FP8, V_CVT_PK_F32_BF8, V_PRNG_B32, V_PERMLANE16_SWAP_B32, V_PERMLANE32_SWAP_B32, V_CVT_F32_BF16, V_CNDMASK_B32, V_ADD_F32, V_SUB_F32, V_SUBREV_F32, V_FMAC_F64, V_MUL_F32, V_MUL_I32_I24, V_MUL_HI_I32_I24, V_MUL_U32_U24, V_MUL_HI_U32_U24, V_MIN_F32, V_MAX_F32, V_MIN_I32, V_MAX_I32, V_MIN_U32, V_MAX_U32, V_LSHRREV_B32, V_ASHRREV_I32, V_LSHLREV_B32, V_AND_B32, V_OR_B32, V_XOR_B32, V_DOT2C_F32_BF16, V_ADD_CO_U32, V_SUB_CO_U32, V_SUBREV_CO_U32, V_ADDC_CO_U32, V_SUBB_CO_U32, V_SUBBREV_CO_U32, V_ADD_F16, V_SUB_F16, V_SUBREV_F16, V_MUL_F16, V_MAC_F16, V_ADD_U16, V_SUB_U16, V_SUBREV_U16, V_MUL_LO_U16, V_LSHLREV_B16, V_LSHRREV_B16, V_ASHRREV_I16, V_MAX_F16, V_MIN_F16, V_MAX_U16, V_MAX_I16, V_MIN_U16, V_MIN_I16, V_LDEXP_F16, V_ADD_U32, V_SUB_U32, V_SUBREV_U32, V_DOT2C_F32_F16, V_DOT2C_I32_I16, V_DOT4C_I32_I8, V_DOT8C_I32_I4, V_FMAC_F32, V_PK_FMAC_F16, V_XNOR_B32, V_MAD_I32_I24, V_MAD_U32_U24, V_CUBEID_F32, V_CUBESC_F32, V_CUBETC_F32, V_CUBEMA_F32, V_BFE_U32, V_BFE_I32, V_BFI_B32, V_FMA_F32, V_FMA_F64, V_LERP_U8, V_ALIGNBIT_B32, V_ALIGNBYTE_B32, V_MIN3_F32, V_MIN3_I32, V_MIN3_U32, V_MAX3_F32, V_MAX3_I32, V_MAX3_U32, V_MED3_F32, V_MED3_I32, V_MED3_U32, V_SAD_U8, V_SAD_HI_U8, V_SAD_U16, V_SAD_U32, V_CVT_PK_U8_F32, V_DIV_FIXUP_F32, V_DIV_FIXUP_F64, V_DIV_SCALE_F32, V_DIV_SCALE_F64, V_DIV_FMAS_F32, V_DIV_FMAS_F64, V_MSAD_U8, V_QSAD_PK_U16_U8, V_MQSAD_PK_U16_U8, V_MQSAD_U32_U8, V_MAD_U64_U32, V_MAD_I64_I32, V_MAD_LEGACY_F16, V_MAD_LEGACY_U16, V_MAD_LEGACY_I16, V_PERM_B32, V_FMA_LEGACY_F16, V_DIV_FIXUP_LEGACY_F16, V_CVT_PKACCUM_U8_F32, V_MAD_U32_U16, V_MAD_I32_I16, V_XAD_U32, V_MIN3_F16, V_MIN3_I16, V_MIN3_U16, V_MAX3_F16, V_MAX3_I16, V_MAX3_U16, V_MED3_F16, V_MED3_I16, V_MED3_U16, V_LSHL_ADD_U32, V_ADD_LSHL_U32, V_ADD3_U32, V_LSHL_OR_B32, V_AND_OR_B32, V_OR3_B32, V_MAD_F16, V_MAD_U16, V_MAD_I16, V_FMA_F16, V_DIV_FIXUP_F16, V_LSHL_ADD_U64, V_BITOP3_B16, V_BITOP3_B32, V_CVT_SCALEF32_PK_FP8_F32, V_CVT_SCALEF32_PK_BF8_F32, V_CVT_SCALEF32_SR_FP8_F32, V_CVT_SCALEF32_SR_BF8_F32, V_CVT_SCALEF32_PK_F32_FP8, V_CVT_SCALEF32_PK_F32_BF8, V_CVT_SCALEF32_F32_FP8, V_CVT_SCALEF32_F32_BF8, V_CVT_SCALEF32_PK_FP4_F32, V_CVT_SCALEF32_SR_PK_FP4_F32, V_CVT_SCALEF32_PK_F32_FP4, V_CVT_SCALEF32_PK_FP8_F16, V_CVT_SCALEF32_PK_BF8_F16, V_CVT_SCALEF32_SR_FP8_F16, V_CVT_SCALEF32_SR_BF8_F16, V_CVT_SCALEF32_PK_FP8_BF16, V_CVT_SCALEF32_PK_BF8_BF16, V_CVT_SCALEF32_SR_FP8_BF16, V_CVT_SCALEF32_SR_BF8_BF16, V_CVT_SCALEF32_PK_F16_FP8, V_CVT_SCALEF32_PK_F16_BF8, V_CVT_SCALEF32_F16_FP8, V_CVT_SCALEF32_F16_BF8, V_CVT_SCALEF32_PK_FP4_F16, V_CVT_SCALEF32_PK_FP4_BF16, V_CVT_SCALEF32_SR_PK_FP4_F16, V_CVT_SCALEF32_SR_PK_FP4_BF16, V_CVT_SCALEF32_PK_F16_FP4, V_CVT_SCALEF32_PK_BF16_FP4, V_CVT_SCALEF32_2XPK16_FP6_F32, V_CVT_SCALEF32_2XPK16_BF6_F32, V_CVT_SCALEF32_SR_PK32_FP6_F32, V_CVT_SCALEF32_SR_PK32_BF6_F32, V_CVT_SCALEF32_PK32_F32_FP6, V_CVT_SCALEF32_PK32_F32_BF6, V_CVT_SCALEF32_PK32_FP6_F16, V_CVT_SCALEF32_PK32_FP6_BF16, V_CVT_SCALEF32_PK32_BF6_F16, V_CVT_SCALEF32_PK32_BF6_BF16, V_CVT_SCALEF32_SR_PK32_FP6_F16, V_CVT_SCALEF32_SR_PK32_FP6_BF16, V_CVT_SCALEF32_SR_PK32_BF6_F16, V_CVT_SCALEF32_SR_PK32_BF6_BF16, V_CVT_SCALEF32_PK32_F16_FP6, V_CVT_SCALEF32_PK32_BF16_FP6, V_CVT_SCALEF32_PK32_F16_BF6, V_CVT_SCALEF32_PK32_BF16_BF6, V_ASHR_PK_I8_I32, V_ASHR_PK_U8_I32, V_CVT_PK_F16_F32, V_CVT_PK_BF16_F32, V_CVT_SCALEF32_PK_BF16_FP8, V_CVT_SCALEF32_PK_BF16_BF8, V_ADD_F64, V_MUL_F64, V_MIN_F64, V_MAX_F64, V_LDEXP_F64, V_MUL_LO_U32, V_MUL_HI_U32, V_MUL_HI_I32, V_LDEXP_F32, V_READLANE_B32, V_WRITELANE_B32, V_BCNT_U32_B32, V_MBCNT_LO_U32_B32, V_MBCNT_HI_U32_B32, V_LSHLREV_B64, V_LSHRREV_B64, V_ASHRREV_I64, V_TRIG_PREOP_F64, V_BFM_B32, V_CVT_PKNORM_I16_F32, V_CVT_PKNORM_U16_F32, V_CVT_PKRTZ_F16_F32, V_CVT_PK_U16_U32, V_CVT_PK_I16_I32, V_CVT_PKNORM_I16_F16, V_CVT_PKNORM_U16_F16, V_ADD_I32, V_SUB_I32, V_ADD_I16, V_SUB_I16, V_PACK_B32_F16, V_MUL_LEGACY_F32, V_CVT_PK_FP8_F32, V_CVT_PK_BF8_F32, V_CVT_SR_FP8_F32, V_CVT_SR_BF8_F32, V_CVT_SR_F16_F32, V_CVT_SR_BF16_F32, V_MINIMUM3_F32, V_MAXIMUM3_F32, V_CMP_CLASS_F32, V_CMPX_CLASS_F32, V_CMP_CLASS_F64, V_CMPX_CLASS_F64, V_CMP_CLASS_F16, V_CMPX_CLASS_F16, V_CMP_F_F16, V_CMP_LT_F16, V_CMP_EQ_F16, V_CMP_LE_F16, V_CMP_GT_F16, V_CMP_LG_F16, V_CMP_GE_F16, V_CMP_O_F16, V_CMP_U_F16, V_CMP_NGE_F16, V_CMP_NLG_F16, V_CMP_NGT_F16, V_CMP_NLE_F16, V_CMP_NEQ_F16, V_CMP_NLT_F16, V_CMP_TRU_F16, V_CMPX_F_F16, V_CMPX_LT_F16, V_CMPX_EQ_F16, V_CMPX_LE_F16, V_CMPX_GT_F16, V_CMPX_LG_F16, V_CMPX_GE_F16, V_CMPX_O_F16, V_CMPX_U_F16, V_CMPX_NGE_F16, V_CMPX_NLG_F16, V_CMPX_NGT_F16, V_CMPX_NLE_F16, V_CMPX_NEQ_F16, V_CMPX_NLT_F16, V_CMPX_TRU_F16, V_CMP_F_F32, V_CMP_LT_F32, V_CMP_EQ_F32, V_CMP_LE_F32, V_CMP_GT_F32, V_CMP_LG_F32, V_CMP_GE_F32, V_CMP_O_F32, V_CMP_U_F32, V_CMP_NGE_F32, V_CMP_NLG_F32, V_CMP_NGT_F32, V_CMP_NLE_F32, V_CMP_NEQ_F32, V_CMP_NLT_F32, V_CMP_TRU_F32, V_CMPX_F_F32, V_CMPX_LT_F32, V_CMPX_EQ_F32, V_CMPX_LE_F32, V_CMPX_GT_F32, V_CMPX_LG_F32, V_CMPX_GE_F32, V_CMPX_O_F32, V_CMPX_U_F32, V_CMPX_NGE_F32, V_CMPX_NLG_F32, V_CMPX_NGT_F32, V_CMPX_NLE_F32, V_CMPX_NEQ_F32, V_CMPX_NLT_F32, V_CMPX_TRU_F32, V_CMP_F_F64, V_CMP_LT_F64, V_CMP_EQ_F64, V_CMP_LE_F64, V_CMP_GT_F64, V_CMP_LG_F64, V_CMP_GE_F64, V_CMP_O_F64, V_CMP_U_F64, V_CMP_NGE_F64, V_CMP_NLG_F64, V_CMP_NGT_F64, V_CMP_NLE_F64, V_CMP_NEQ_F64, V_CMP_NLT_F64, V_CMP_TRU_F64, V_CMPX_F_F64, V_CMPX_LT_F64, V_CMPX_EQ_F64, V_CMPX_LE_F64, V_CMPX_GT_F64, V_CMPX_LG_F64, V_CMPX_GE_F64, V_CMPX_O_F64, V_CMPX_U_F64, V_CMPX_NGE_F64, V_CMPX_NLG_F64, V_CMPX_NGT_F64, V_CMPX_NLE_F64, V_CMPX_NEQ_F64, V_CMPX_NLT_F64, V_CMPX_TRU_F64, V_CMP_F_I16, V_CMP_LT_I16, V_CMP_EQ_I16, V_CMP_LE_I16, V_CMP_GT_I16, V_CMP_NE_I16, V_CMP_GE_I16, V_CMP_T_I16, V_CMP_F_U16, V_CMP_LT_U16, V_CMP_EQ_U16, V_CMP_LE_U16, V_CMP_GT_U16, V_CMP_NE_U16, V_CMP_GE_U16, V_CMP_T_U16, V_CMPX_F_I16, V_CMPX_LT_I16, V_CMPX_EQ_I16, V_CMPX_LE_I16, V_CMPX_GT_I16, V_CMPX_NE_I16, V_CMPX_GE_I16, V_CMPX_T_I16, V_CMPX_F_U16, V_CMPX_LT_U16, V_CMPX_EQ_U16, V_CMPX_LE_U16, V_CMPX_GT_U16, V_CMPX_NE_U16, V_CMPX_GE_U16, V_CMPX_T_U16, V_CMP_F_I32, V_CMP_LT_I32, V_CMP_EQ_I32, V_CMP_LE_I32, V_CMP_GT_I32, V_CMP_NE_I32, V_CMP_GE_I32, V_CMP_T_I32, V_CMP_F_U32, V_CMP_LT_U32, V_CMP_EQ_U32, V_CMP_LE_U32, V_CMP_GT_U32, V_CMP_NE_U32, V_CMP_GE_U32, V_CMP_T_U32, V_CMPX_F_I32, V_CMPX_LT_I32, V_CMPX_EQ_I32, V_CMPX_LE_I32, V_CMPX_GT_I32, V_CMPX_NE_I32, V_CMPX_GE_I32, V_CMPX_T_I32, V_CMPX_F_U32, V_CMPX_LT_U32, V_CMPX_EQ_U32, V_CMPX_LE_U32, V_CMPX_GT_U32, V_CMPX_NE_U32, V_CMPX_GE_U32, V_CMPX_T_U32, V_CMP_F_I64, V_CMP_LT_I64, V_CMP_EQ_I64, V_CMP_LE_I64, V_CMP_GT_I64, V_CMP_NE_I64, V_CMP_GE_I64, V_CMP_T_I64, V_CMP_F_U64, V_CMP_LT_U64, V_CMP_EQ_U64, V_CMP_LE_U64, V_CMP_GT_U64, V_CMP_NE_U64, V_CMP_GE_U64, V_CMP_T_U64, V_CMPX_F_I64, V_CMPX_LT_I64, V_CMPX_EQ_I64, V_CMPX_LE_I64, V_CMPX_GT_I64, V_CMPX_NE_I64, V_CMPX_GE_I64, V_CMPX_T_I64, V_CMPX_F_U64, V_CMPX_LT_U64, V_CMPX_EQ_U64, V_CMPX_LE_U64, V_CMPX_GT_U64, V_CMPX_NE_U64, V_CMPX_GE_U64, V_CMPX_T_U64

---

### 12.11. VOP3A & VOP3B Instructions

VOP3 instructions use one of two encodings:

```
  VOP3B       this encoding allows specifying a unique scalar destination, and is used only for:
              V_ADD_CO_U32
              V_SUB_CO_U32
              V_SUBREV_CO_U32
              V_ADDC_CO_U32
              V_SUBB_CO_U32
              V_SUBBREV_CO_U32
              V_DIV_SCALE_F32
              V_DIV_SCALE_F64
              V_MAD_U64_U32
              V_MAD_I64_I32
```

```
  VOP3A       all other VALU instructions use this encoding
```

#### V_NOP  (opcode 384)

Do nothing.

Notes

This instruction can be used to insert a single-cycle bubble in the vector ALU pipeline. For multiple cycles
repeat this opcode.

#### V_MOV_B32  (opcode 385)

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
        v_mov_b32 v0, abs(v1)   // Set v0 to the absolute value of v1
```

#### V_READFIRSTLANE_B32  (opcode 386)

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

#### V_CVT_I32_F64  (opcode 387)

Convert from a double-precision float input to a signed 32-bit integer value and store the result into a vector
register.

```
  D0.i32 = f64_to_i32(S0.f64)
```

Notes

0.5ULP accuracy, out-of-range floating point values (including infinity) saturate. NAN is converted to 0.

Generation of the INEXACT exception is controlled by the CLAMP bit. INEXACT exceptions are enabled for this
conversion iff CLAMP == 1.

#### V_CVT_F64_I32  (opcode 388)

Convert from a signed 32-bit integer input to a double-precision float value and store the result into a vector
register.

```
  D0.f64 = i32_to_f64(S0.i32)
```

Notes

0ULP accuracy.

#### V_CVT_F32_I32  (opcode 389)

Convert from a signed 32-bit integer input to a single-precision float value and store the result into a vector
register.

```
  D0.f32 = i32_to_f32(S0.i32)
```

Notes

0.5ULP accuracy.

#### V_CVT_F32_U32  (opcode 390)

Convert from an unsigned 32-bit integer input to a single-precision float value and store the result into a vector
register.

```
  D0.f32 = u32_to_f32(S0.u32)
```

Notes

0.5ULP accuracy.

#### V_CVT_U32_F32  (opcode 391)

Convert from a single-precision float input to an unsigned 32-bit integer value and store the result into a vector
register.

```
  D0.u32 = f32_to_u32(S0.f32)
```

Notes

1ULP accuracy, out-of-range floating point values (including infinity) saturate. NAN is converted to 0.

Generation of the INEXACT exception is controlled by the CLAMP bit. INEXACT exceptions are enabled for this
conversion iff CLAMP == 1.

#### V_CVT_I32_F32  (opcode 392)

Convert from a single-precision float input to a signed 32-bit integer value and store the result into a vector
register.

```
  D0.i32 = f32_to_i32(S0.f32)
```

Notes

1ULP accuracy, out-of-range floating point values (including infinity) saturate. NAN is converted to 0.

Generation of the INEXACT exception is controlled by the CLAMP bit. INEXACT exceptions are enabled for this
conversion iff CLAMP == 1.

#### V_CVT_F16_F32  (opcode 394)

Convert from a single-precision float input to a half-precision float value and store the result into a vector
register.

```
  D0.f16 = f32_to_f16(S0.f32)
```

Notes

0.5ULP accuracy, supports input modifiers and creates FP16 denormals when appropriate. Flush denorms on
output if specified based on DP denorm mode. Output rounding based on DP rounding mode.

#### V_CVT_F32_F16  (opcode 395)

Convert from a half-precision float input to a single-precision float value and store the result into a vector
register.

```
  D0.f32 = f16_to_f32(S0.f16)
```

Notes

0ULP accuracy, FP16 denormal inputs are accepted. Flush denorms on input if specified based on DP denorm
mode.

#### V_CVT_RPI_I32_F32  (opcode 396)

Convert from a single-precision float input to a signed 32-bit integer value using round to nearest integer
semantics (ignore the default rounding mode) and store the result into a vector register.

```
  D0.i32 = f32_to_i32(floor(S0.f32 + 0.5F))
```

Notes

0.5ULP accuracy, denormals are supported.

#### V_CVT_FLR_I32_F32  (opcode 397)

Convert from a single-precision float input to a signed 32-bit integer value using round-down semantics (ignore
the default rounding mode) and store the result into a vector register.

```
  D0.i32 = f32_to_i32(floor(S0.f32))
```

Notes

1ULP accuracy, denormals are supported.

#### V_CVT_OFF_F32_I4  (opcode 398)

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

#### V_CVT_F32_F64  (opcode 399)

Convert from a double-precision float input to a single-precision float value and store the result into a vector
register.

```
  D0.f32 = f64_to_f32(S0.f64)
```

Notes

0.5ULP accuracy, denormals are supported.

#### V_CVT_F64_F32  (opcode 400)

Convert from a single-precision float input to a double-precision float value and store the result into a vector
register.

```
  D0.f64 = f32_to_f64(S0.f32)
```

Notes

0ULP accuracy, denormals are supported.

#### V_CVT_F32_UBYTE0  (opcode 401)

Convert an unsigned byte in byte 0 of the input to a single-precision float value and store the result into a vector
register.

```
  D0.f32 = u32_to_f32(S0[7 : 0].u32)
```

#### V_CVT_F32_UBYTE1  (opcode 402)

Convert an unsigned byte in byte 1 of the input to a single-precision float value and store the result into a vector
register.

```
  D0.f32 = u32_to_f32(S0[15 : 8].u32)
```

#### V_CVT_F32_UBYTE2  (opcode 403)

Convert an unsigned byte in byte 2 of the input to a single-precision float value and store the result into a vector
register.

```
  D0.f32 = u32_to_f32(S0[23 : 16].u32)
```

#### V_CVT_F32_UBYTE3  (opcode 404)

Convert an unsigned byte in byte 3 of the input to a single-precision float value and store the result into a vector
register.

```
  D0.f32 = u32_to_f32(S0[31 : 24].u32)
```

#### V_CVT_U32_F64  (opcode 405)

Convert from a double-precision float input to an unsigned 32-bit integer value and store the result into a
vector register.

```
  D0.u32 = f64_to_u32(S0.f64)
```

Notes

0.5ULP accuracy, out-of-range floating point values (including infinity) saturate. NAN is converted to 0.

Generation of the INEXACT exception is controlled by the CLAMP bit. INEXACT exceptions are enabled for this
conversion iff CLAMP == 1.

#### V_CVT_F64_U32  (opcode 406)

Convert from an unsigned 32-bit integer input to a double-precision float value and store the result into a
vector register.

```
  D0.f64 = u32_to_f64(S0.u32)
```

Notes

0ULP accuracy.

#### V_TRUNC_F64  (opcode 407)

Compute the integer part of a double-precision float input using round toward zero semantics and store the
result in floating point format into a vector register.

```
  D0.f64 = trunc(S0.f64)
```

#### V_CEIL_F64  (opcode 408)

Round the double-precision float input up to next integer and store the result in floating point format into a
vector register.

```
  D0.f64 = trunc(S0.f64);
  if ((S0.f64 > 0.0) && (S0.f64 != D0.f64)) then
        D0.f64 += 1.0
  endif
```

#### V_RNDNE_F64  (opcode 409)

Round the double-precision float input to the nearest even integer and store the result in floating point format
into a vector register.

```
  D0.f64 = floor(S0.f64 + 0.5);
  if (isEven(floor(S0.f64)) && (fract(S0.f64) == 0.5)) then
        D0.f64 -= 1.0
  endif
```

#### V_FLOOR_F64  (opcode 410)

Round the double-precision float input down to previous integer and store the result in floating point format
into a vector register.

```
  D0.f64 = trunc(S0.f64);
  if ((S0.f64 < 0.0) && (S0.f64 != D0.f64)) then
        D0.f64 += -1.0
  endif
```

#### V_FRACT_F32  (opcode 411)

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

#### V_TRUNC_F32  (opcode 412)

Compute the integer part of a single-precision float input using round toward zero semantics and store the
result in floating point format into a vector register.

```
  D0.f32 = trunc(S0.f32)
```

#### V_CEIL_F32  (opcode 413)

Round the single-precision float input up to next integer and store the result in floating point format into a
vector register.

```
  D0.f32 = trunc(S0.f32);
  if ((S0.f32 > 0.0F) && (S0.f32 != D0.f32)) then
        D0.f32 += 1.0F
  endif
```

#### V_RNDNE_F32  (opcode 414)

Round the single-precision float input to the nearest even integer and store the result in floating point format
into a vector register.

```
  D0.f32 = floor(S0.f32 + 0.5F);
```

```
  if (isEven(64'F(floor(S0.f32))) && (fract(S0.f32) == 0.5F)) then
        D0.f32 -= 1.0F
  endif
```

#### V_FLOOR_F32  (opcode 415)

Round the single-precision float input down to previous integer and store the result in floating point format
into a vector register.

```
  D0.f32 = trunc(S0.f32);
  if ((S0.f32 < 0.0F) && (S0.f32 != D0.f32)) then
        D0.f32 += -1.0F
  endif
```

#### V_EXP_F32  (opcode 416)

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

#### V_LOG_F32  (opcode 417)

Calculate the base 2 logarithm of the single-precision float input and store the result into a vector register.

```
  D0.f32 = log2(S0.f32)
```

Notes

1ULP accuracy, denormals are flushed.

Functional examples:

```
  V_LOG_F32(0xff800000) => 0xffc00000       // log(-INF) = NAN
  V_LOG_F32(0xbf800000) => 0xffc00000       // log(-1.0) = NAN
  V_LOG_F32(0x80000000) => 0xff800000       // log(-0.0) = -INF
  V_LOG_F32(0x00000000) => 0xff800000       // log(+0.0) = -INF
  V_LOG_F32(0x3f800000) => 0x00000000       // log(+1.0) = 0
  V_LOG_F32(0x7f800000) => 0x7f800000       // log(+INF) = +INF
```

#### V_RCP_F32  (opcode 418)

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

#### V_RCP_IFLAG_F32  (opcode 419)

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

#### V_RSQ_F32  (opcode 420)

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

#### V_RCP_F64  (opcode 421)

Calculate the reciprocal of the double-precision float input using IEEE rules and store the result into a vector
register.

```
  D0.f64 = 1.0 / S0.f64
```

Notes

This opcode has (2**29)ULP accuracy and supports denormals.

#### V_RSQ_F64  (opcode 422)

Calculate the reciprocal of the square root of the double-precision float input using IEEE rules and store the
result into a vector register.

```
  D0.f64 = 1.0 / sqrt(S0.f64)
```

Notes

This opcode has (2**29)ULP accuracy and supports denormals.

#### V_SQRT_F32  (opcode 423)

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

#### V_SQRT_F64  (opcode 424)

Calculate the square root of the double-precision float input using IEEE rules and store the result into a vector
register.

```
  D0.f64 = sqrt(S0.f64)
```

Notes

This opcode has (2**29)ULP accuracy and supports denormals.

#### V_SIN_F32  (opcode 425)

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

#### V_COS_F32  (opcode 426)

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

#### V_NOT_B32  (opcode 427)

Calculate bitwise negation on a vector input and store the result into a vector register.

```
  D0.u32 = ~S0.u32
```

Notes

Input and output modifiers not supported.

#### V_BFREV_B32  (opcode 428)

Reverse the order of bits in a vector input and store the result into a vector register.

```
  D0.u32[31 : 0] = S0.u32[0 : 31]
```

Notes

Input and output modifiers not supported.

#### V_FFBH_U32  (opcode 429)

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

#### V_FFBL_B32  (opcode 430)

Count the number of trailing "0" bits before the first "1" in a vector input and store the result into a vector
register. Store -1 if there are no "1" bits in the input.

```
  D0.i32 = -1;
  // Set if no ones are found
  for i in 0 : 31 do
```

```
        // Search from LSB
        if S0.u32[i] == 1'1U then
            D0.i32 = i;
            break
        endif
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

#### V_FFBH_I32  (opcode 431)

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

#### V_FREXP_EXP_I32_F64  (opcode 432)

Extract the exponent of a double-precision float input and store the result as a signed 32-bit integer into a
vector register.

```
  if ((S0.f64 == +INF) || (S0.f64 == -INF) || isNAN(S0.f64)) then
        D0.i32 = 0
  else
        D0.i32 = exponent(S0.f64) - 1023 + 1
  endif
```

Notes

This operation satisfies the invariant S0.f64 = significand * (2 ** exponent). See also V_FREXP_MANT_F64,
which returns the significand. See the C library function frexp() for more information.

#### V_FREXP_MANT_F64  (opcode 433)

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

#### V_FRACT_F64  (opcode 434)

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

#### V_FREXP_EXP_I32_F32  (opcode 435)

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

#### V_FREXP_MANT_F32  (opcode 436)

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

#### V_CLREXCP  (opcode 437)

Clear this wave's exception state in the vector ALU.

#### V_MOV_B64  (opcode 440)

Move data from a 64-bit vector input into a vector register.

```
  D0.b64 = S0.b64
```

Notes

Floating-point modifiers are valid for this instruction if S0.u64 is a 64-bit floating point value. This instruction is
suitable for negating or taking the absolute value of a floating-point value.

#### V_CVT_F16_U16  (opcode 441)

Convert from an unsigned 16-bit integer input to a half-precision float value and store the result into a vector
register.

```
  D0.f16 = u16_to_f16(S0.u16)
```

Notes

0.5ULP accuracy, supports denormals, rounding, exception flags and saturation.

#### V_CVT_F16_I16  (opcode 442)

Convert from a signed 16-bit integer input to a half-precision float value and store the result into a vector
register.

```
  D0.f16 = i16_to_f16(S0.i16)
```

Notes

0.5ULP accuracy, supports denormals, rounding, exception flags and saturation.

#### V_CVT_U16_F16  (opcode 443)

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

#### V_CVT_I16_F16  (opcode 444)

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

#### V_RCP_F16  (opcode 445)

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

#### V_SQRT_F16  (opcode 446)

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

#### V_RSQ_F16  (opcode 447)

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

#### V_LOG_F16  (opcode 448)

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
  V_LOG_F16(0x8000) => 0xfc00        // log(-0.0) = -INF
  V_LOG_F16(0x0000) => 0xfc00        // log(+0.0) = -INF
  V_LOG_F16(0x3c00) => 0x0000        // log(+1.0) = 0
  V_LOG_F16(0x7c00) => 0x7c00        // log(+INF) = +INF
```

#### V_EXP_F16  (opcode 449)

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

#### V_FREXP_MANT_F16  (opcode 450)

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

#### V_FREXP_EXP_I16_F16  (opcode 451)

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

#### V_FLOOR_F16  (opcode 452)

Round the half-precision float input down to previous integer and store the result in floating point format into
a vector register.

```
  D0.f16 = trunc(S0.f16);
  if ((S0.f16 < 16'0.0) && (S0.f16 != D0.f16)) then
        D0.f16 += -16'1.0
  endif
```

#### V_CEIL_F16  (opcode 453)

Round the half-precision float input up to next integer and store the result in floating point format into a vector
register.

```
  D0.f16 = trunc(S0.f16);
  if ((S0.f16 > 16'0.0) && (S0.f16 != D0.f16)) then
        D0.f16 += 16'1.0
  endif
```

#### V_TRUNC_F16  (opcode 454)

Compute the integer part of a half-precision float input using round toward zero semantics and store the result
in floating point format into a vector register.

```
  D0.f16 = trunc(S0.f16)
```

#### V_RNDNE_F16  (opcode 455)

Round the half-precision float input to the nearest even integer and store the result in floating point format
into a vector register.

```
  D0.f16 = floor(S0.f16 + 16'0.5);
  if (isEven(64'F(floor(S0.f16))) && (fract(S0.f16) == 16'0.5)) then
        D0.f16 -= 16'1.0
  endif
```

#### V_FRACT_F16  (opcode 456)

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

#### V_SIN_F16  (opcode 457)

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
```

```
  V_SIN_F16(0x7bff) => 0x0000        // Most positive finite FP16
  V_SIN_F16(0x7c00) => 0xfe00        // sin(+INF) = NAN
```

#### V_COS_F16  (opcode 458)

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

#### V_CVT_NORM_I16_F16  (opcode 461)

Convert from a half-precision float input to a signed normalized short and store the result into a vector
register.

```
  D0.i16 = f16_to_snorm(S0.f16)
```

Notes

0.5ULP accuracy, supports rounding, exception flags and saturation, denormals are supported.

#### V_CVT_NORM_U16_F16  (opcode 462)

Convert from a half-precision float input to an unsigned normalized short and store the result into a vector
register.

```
  D0.u16 = f16_to_unorm(S0.f16)
```

Notes

0.5ULP accuracy, supports rounding, exception flags and saturation, denormals are supported.

#### V_SAT_PK_U8_I16  (opcode 463)

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

#### V_SWAP_B32  (opcode 465)

Swap the values in two vector registers.

```
  tmp = D0.b32;
  D0.b32 = S0.b32;
  S0.b32 = tmp
```

Notes

Input and output modifiers not supported; this is an untyped operation.

#### V_ACCVGPR_MOV_B32  (opcode 466)

Move data from one accumulator register to another accumulator register.

#### V_CVT_F32_FP8  (opcode 468)

Convert from an FP8 float input to a single-precision float value and store the result into a vector register.

```
  if SDWA_SRC0_SEL == BYTE1.b3 then
        D0.f32 = fp8_to_f32(S0[15 : 8].fp8)
  elsif SDWA_SRC0_SEL == BYTE2.b3 then
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

#### V_CVT_F32_BF8  (opcode 469)

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

#### V_CVT_PK_F32_FP8  (opcode 470)

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

#### V_CVT_PK_F32_BF8  (opcode 471)

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

#### V_PRNG_B32  (opcode 472)

Generate a pseudorandom number using an LFSR (linear feedback shift register) seeded with the vector input,
then store the result into a vector register.

```
  in = S0.u32;
  D0.u32 = ((in << 1U) ^ (in[31] ? 197U : 0U))
```

Notes

This function produces a sequence of pseudorandom numbers with period 2**32 - 1 unless the input is zero, in
which case the period is 1.

#### V_PERMLANE16_SWAP_B32  (opcode 473)

Swap data between two vector registers. Odd rows of the first operand are swapped with even rows of the
second operand (one row is 16 lanes).

```
  for pass in 0 : 1 do
        for lane in 0 : 15 do
            tmp = VGPR[pass * 32 + lane][SRC0.u32];
            VGPR[pass * 32 + lane][SRC0.u32] = VGPR[pass * 32 + lane + 16][VDST.u32];
            VGPR[pass * 32 + lane + 16][VDST.u32] = tmp
        endfor
```

```
  endfor
```

Notes

ABS, NEG and OMOD modifiers should all be zeroed for this instruction.

This instruction is useful for BFP data conversions.

#### V_PERMLANE32_SWAP_B32  (opcode 474)

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

#### V_CVT_F32_BF16  (opcode 475)

Convert from a BF16 float input to a single-precision float value and store the result into a vector register.

```
  D0.f32 = 32'F({ S0.b16, 16'0U })
```

#### V_CNDMASK_B32  (opcode 256)

Copy data from one of two inputs based on the per-lane condition code and store the result into a vector
register.

```
  D0.u32 = VCC.u64[laneId] ? S1.u32 : S0.u32
```

Notes

In VOP3 the VCC source may be a scalar GPR specified in S2.

Floating-point modifiers are valid for this instruction if S0 and S1 are 32-bit floating point values. This
instruction is suitable for negating or taking the absolute value of a floating-point value.

#### V_ADD_F32  (opcode 257)

Add two floating point inputs and store the result into a vector register.

```
  D0.f32 = S0.f32 + S1.f32
```

Notes

0.5ULP precision, denormals are supported.

#### V_SUB_F32  (opcode 258)

Subtract the second floating point input from the first input and store the result into a vector register.

```
  D0.f32 = S0.f32 - S1.f32
```

Notes

0.5ULP precision, denormals are supported.

#### V_SUBREV_F32  (opcode 259)

Subtract the first floating point input from the second input and store the result into a vector register.

```
  D0.f32 = S1.f32 - S0.f32
```

Notes

0.5ULP precision, denormals are supported.

#### V_FMAC_F64  (opcode 260)

Multiply two floating point inputs and accumulate the result into the destination register using fused multiply
add.

```
  D0.f64 = fma(S0.f64, S1.f64, D0.f64)
```

#### V_MUL_F32  (opcode 261)

Multiply two floating point inputs and store the result into a vector register.

```
  D0.f32 = S0.f32 * S1.f32
```

Notes

0.5ULP precision, denormals are supported.

#### V_MUL_I32_I24  (opcode 262)

Multiply two signed 24-bit integer inputs and store the result as a signed 32-bit integer into a vector register.

```
  D0.i32 = 32'I(S0.i24) * 32'I(S1.i24)
```

Notes

This opcode is expected to be as efficient as basic single-precision opcodes since it utilizes the single-precision
floating point multiplier. See also V_MUL_HI_I32_I24.

#### V_MUL_HI_I32_I24  (opcode 263)

Multiply two signed 24-bit integer inputs and store the high 32 bits of the result as a signed 32-bit integer into a
vector register.

```
  D0.i32 = 32'I((64'I(S0.i24) * 64'I(S1.i24)) >> 32U)
```

Notes

See also V_MUL_I32_I24.

#### V_MUL_U32_U24  (opcode 264)

Multiply two unsigned 24-bit integer inputs and store the result as an unsigned 32-bit integer into a vector
register.

```
  D0.u32 = 32'U(S0.u24) * 32'U(S1.u24)
```

Notes

This opcode is expected to be as efficient as basic single-precision opcodes since it utilizes the single-precision
floating point multiplier. See also V_MUL_HI_U32_U24.

#### V_MUL_HI_U32_U24  (opcode 265)

Multiply two unsigned 24-bit integer inputs and store the high 32 bits of the result as an unsigned 32-bit integer
into a vector register.

```
  D0.u32 = 32'U((64'U(S0.u24) * 64'U(S1.u24)) >> 32U)
```

Notes

See also V_MUL_U32_U24.

#### V_MIN_F32  (opcode 266)

Select the minimum of two single-precision float inputs and store the result into a vector register.

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
        D0.f32 = S1.f32
  elsif ((64'F(S0.f32) == -0.0) && (64'F(S1.f32) == +0.0)) then
        D0.f32 = S0.f32
  else
        // Note: there's no IEEE case here like there is for V_MAX_F32.
        D0.f32 = S0.f32 < S1.f32 ? S0.f32 : S1.f32
  endif
```

#### V_MAX_F32  (opcode 267)

Select the maximum of two single-precision float inputs and store the result into a vector register.

```
  if (WAVE_MODE.IEEE && isSignalNAN(64'F(S0.f32))) then
        D0.f32 = 32'F(cvtToQuietNAN(64'F(S0.f32)))
  elsif (WAVE_MODE.IEEE && isSignalNAN(64'F(S1.f32))) then
        D0.f32 = 32'F(cvtToQuietNAN(64'F(S1.f32)))
```

```
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

#### V_MIN_I32  (opcode 268)

Select the minimum of two signed 32-bit integer inputs and store the selected value into a vector register.

```
  D0.i32 = S0.i32 < S1.i32 ? S0.i32 : S1.i32
```

#### V_MAX_I32  (opcode 269)

Select the maximum of two signed 32-bit integer inputs and store the selected value into a vector register.

```
  D0.i32 = S0.i32 >= S1.i32 ? S0.i32 : S1.i32
```

#### V_MIN_U32  (opcode 270)

Select the minimum of two unsigned 32-bit integer inputs and store the selected value into a vector register.

```
  D0.u32 = S0.u32 < S1.u32 ? S0.u32 : S1.u32
```

#### V_MAX_U32  (opcode 271)

Select the maximum of two unsigned 32-bit integer inputs and store the selected value into a vector register.

```
  D0.u32 = S0.u32 >= S1.u32 ? S0.u32 : S1.u32
```

#### V_LSHRREV_B32  (opcode 272)

Given a shift count in the first vector input, calculate the logical shift right of the second vector input and store
the result into a vector register.

```
  D0.u32 = (S1.u32 >> S0[4 : 0].u32)
```

#### V_ASHRREV_I32  (opcode 273)

Given a shift count in the first vector input, calculate the arithmetic shift right (preserving sign bit) of the second
vector input and store the result into a vector register.

```
  D0.i32 = (S1.i32 >> S0[4 : 0].u32)
```

#### V_LSHLREV_B32  (opcode 274)

Given a shift count in the first vector input, calculate the logical shift left of the second vector input and store the
result into a vector register.

```
  D0.u32 = (S1.u32 << S0[4 : 0].u32)
```

#### V_AND_B32  (opcode 275)

Calculate bitwise AND on two vector inputs and store the result into a vector register.

```
  D0.u32 = (S0.u32 & S1.u32)
```

Notes

Input and output modifiers not supported.

#### V_OR_B32  (opcode 276)

Calculate bitwise OR on two vector inputs and store the result into a vector register.

```
  D0.u32 = (S0.u32 | S1.u32)
```

Notes

Input and output modifiers not supported.

#### V_XOR_B32  (opcode 277)

Calculate bitwise XOR on two vector inputs and store the result into a vector register.

```
  D0.u32 = (S0.u32 ^ S1.u32)
```

Notes

Input and output modifiers not supported.

#### V_DOT2C_F32_BF16  (opcode 278)

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

#### V_ADD_CO_U32  (opcode 281)

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

#### V_SUB_CO_U32  (opcode 282)

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

#### V_SUBREV_CO_U32  (opcode 283)

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

#### V_ADDC_CO_U32  (opcode 284)

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

#### V_SUBB_CO_U32  (opcode 285)

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

#### V_SUBBREV_CO_U32  (opcode 286)

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

#### V_ADD_F16  (opcode 287)

Add two floating point inputs and store the result into a vector register.

```
  D0.f16 = S0.f16 + S1.f16
```

Notes

0.5ULP precision. Supports denormals, round mode, exception flags and saturation.

#### V_SUB_F16  (opcode 288)

Subtract the second floating point input from the first input and store the result into a vector register.

```
  D0.f16 = S0.f16 - S1.f16
```

Notes

0.5ULP precision. Supports denormals, round mode, exception flags and saturation.

#### V_SUBREV_F16  (opcode 289)

Subtract the first floating point input from the second input and store the result into a vector register.

```
  D0.f16 = S1.f16 - S0.f16
```

Notes

0.5ULP precision. Supports denormals, round mode, exception flags and saturation.

#### V_MUL_F16  (opcode 290)

Multiply two floating point inputs and store the result into a vector register.

```
  D0.f16 = S0.f16 * S1.f16
```

Notes

0.5ULP precision. Supports denormals, round mode, exception flags and saturation.

#### V_MAC_F16  (opcode 291)

Multiply two floating point inputs and accumulate the result into the destination register. Implements IEEE

rules and non-standard rule for OPSEL.

```
  tmp = S0.f16 * S1.f16 + D0.f16;
  if OPSEL.u4[3] then
        D0 = { tmp.f16, D0[15 : 0] }
  else
        D0 = { 16'0, tmp.f16 }
  endif
```

Notes

Supports round mode, exception flags, saturation.

#### V_ADD_U16  (opcode 294)

Add two unsigned 16-bit integer inputs and store the result into a vector register. No carry-in or carry-out
support.

```
  D0.u16 = S0.u16 + S1.u16
```

Notes

Supports saturation (unsigned 16-bit integer domain).

#### V_SUB_U16  (opcode 295)

Subtract the second unsigned 16-bit integer input from the first input and store the result into a vector register.
No carry-in or carry-out support.

```
  D0.u16 = S0.u16 - S1.u16
```

Notes

Supports saturation (unsigned 16-bit integer domain).

#### V_SUBREV_U16  (opcode 296)

Subtract the first unsigned 16-bit integer input from the second input and store the result into a vector register.
No carry-in or carry-out support.

```
  D0.u16 = S1.u16 - S0.u16
```

Notes

Supports saturation (unsigned 16-bit integer domain).

#### V_MUL_LO_U16  (opcode 297)

Multiply two unsigned 16-bit integer inputs and store the low bits of the result into a vector register.

```
  D0.u16 = S0.u16 * S1.u16
```

Notes

Supports saturation (unsigned 16-bit integer domain).

#### V_LSHLREV_B16  (opcode 298)

Given a shift count in the first vector input, calculate the logical shift left of the second vector input and store the
result into a vector register.

```
  D0.u16 = (S1.u16 << S0[3 : 0].u32)
```

#### V_LSHRREV_B16  (opcode 299)

Given a shift count in the first vector input, calculate the logical shift right of the second vector input and store
the result into a vector register.

```
  D0.u16 = (S1.u16 >> S0[3 : 0].u32)
```

#### V_ASHRREV_I16  (opcode 300)

Given a shift count in the first vector input, calculate the arithmetic shift right (preserving sign bit) of the second
vector input and store the result into a vector register.

```
  D0.i16 = (S1.i16 >> S0[3 : 0].u32)
```

#### V_MAX_F16  (opcode 301)

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

#### V_MIN_F16  (opcode 302)

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

#### V_MAX_U16  (opcode 303)

Select the maximum of two unsigned 16-bit integer inputs and store the selected value into a vector register.

```
  D0.u16 = S0.u16 >= S1.u16 ? S0.u16 : S1.u16
```

#### V_MAX_I16  (opcode 304)

Select the maximum of two signed 16-bit integer inputs and store the selected value into a vector register.

```
  D0.i16 = S0.i16 >= S1.i16 ? S0.i16 : S1.i16
```

#### V_MIN_U16  (opcode 305)

Select the minimum of two unsigned 16-bit integer inputs and store the selected value into a vector register.

```
  D0.u16 = S0.u16 < S1.u16 ? S0.u16 : S1.u16
```

#### V_MIN_I16  (opcode 306)

Select the minimum of two signed 16-bit integer inputs and store the selected value into a vector register.

```
  D0.i16 = S0.i16 < S1.i16 ? S0.i16 : S1.i16
```

#### V_LDEXP_F16  (opcode 307)

Multiply the first input, a floating point value, by an integral power of 2 specified in the second input, a signed
integer value, and store the floating point result into a vector register.

```
  D0.f16 = S0.f16 * 16'F(2.0F ** 32'I(S1.i16))
```

Notes

Compare with the ldexp() function in C. Note that the S1 has a format of f16 since floating point literal
constants are interpreted as 16 bit value for this opcode.

#### V_ADD_U32  (opcode 308)

Add two unsigned 32-bit integer inputs and store the result into a vector register. No carry-in or carry-out
support.

```
  D0.u32 = S0.u32 + S1.u32
```

Notes

Supports saturation (unsigned 32-bit integer domain).

#### V_SUB_U32  (opcode 309)

Subtract the second unsigned 32-bit integer input from the first input and store the result into a vector register.
No carry-in or carry-out support.

```
  D0.u32 = S0.u32 - S1.u32
```

Notes

Supports saturation (unsigned 32-bit integer domain).

#### V_SUBREV_U32  (opcode 310)

Subtract the first unsigned 32-bit integer input from the second input and store the result into a vector register.
No carry-in or carry-out support.

```
  D0.u32 = S1.u32 - S0.u32
```

Notes

Supports saturation (unsigned 32-bit integer domain).

#### V_DOT2C_F32_F16  (opcode 311)

Compute the dot product of two packed 2-D half-precision float inputs in the single-precision float domain and
accumulate with the single-precision float value in the destination register.

```
  tmp = D0.f32;
  tmp += f16_to_f32(S0[15 : 0].f16) * f16_to_f32(S1[15 : 0].f16);
  tmp += f16_to_f32(S0[31 : 16].f16) * f16_to_f32(S1[31 : 16].f16);
```

```
  D0.f32 = tmp
```

#### V_DOT2C_I32_I16  (opcode 312)

Compute the dot product of two packed 2-D signed 16-bit integer inputs in the signed 32-bit integer domain and
accumulate with the signed 32-bit integer value in the destination register.

```
  tmp = D0.i32;
  tmp += i16_to_i32(S0[15 : 0].i16) * i16_to_i32(S1[15 : 0].i16);
  tmp += i16_to_i32(S0[31 : 16].i16) * i16_to_i32(S1[31 : 16].i16);
  D0.i32 = tmp
```

#### V_DOT4C_I32_I8  (opcode 313)

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

#### V_DOT8C_I32_I4  (opcode 314)

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

#### V_FMAC_F32  (opcode 315)

Multiply two floating point inputs and accumulate the result into the destination register using fused multiply
add.

```
  D0.f32 = fma(S0.f32, S1.f32, D0.f32)
```

#### V_PK_FMAC_F16  (opcode 316)

Multiply two packed half-precision float inputs component-wise and accumulate the result into the destination
register using fused multiply add.

```
  D0[15 : 0].f16 = fma(S0[15 : 0].f16, S1[15 : 0].f16, D0[15 : 0].f16);
  D0[31 : 16].f16 = fma(S0[31 : 16].f16, S1[31 : 16].f16, D0[31 : 16].f16)
```

#### V_XNOR_B32  (opcode 317)

Calculate bitwise XNOR on two vector inputs and store the result into a vector register.

```
  D0.u32 = ~(S0.u32 ^ S1.u32)
```

Notes

Input and output modifiers not supported.

#### V_MAD_I32_I24  (opcode 450)

Multiply two signed 24-bit integer inputs in the signed 32-bit integer domain, add a signed 32-bit integer value
from a third input, and store the result as a signed 32-bit integer into a vector register.

```
  D0.i32 = 32'I(S0.i24) * 32'I(S1.i24) + S2.i32
```

#### V_MAD_U32_U24  (opcode 451)

Multiply two unsigned 24-bit integer inputs in the unsigned 32-bit integer domain, add a unsigned 32-bit
integer value from a third input, and store the result as an unsigned 32-bit integer into a vector register.

```
  D0.u32 = 32'U(S0.u24) * 32'U(S1.u24) + S2.u32
```

#### V_CUBEID_F32  (opcode 452)

Compute the cubemap face ID of a 3D coordinate specified as three single-precision float inputs. Store the
result in single-precision float format into a vector register.

```
  // Set D0.f = cubemap face ID ({0.0, 1.0, ..., 5.0}).
  // XYZ coordinate is given in (S0.f, S1.f, S2.f).
  // S0.f = x
  // S1.f = y
  // S2.f = z
  if ((abs(S2.f32) >= abs(S0.f32)) && (abs(S2.f32) >= abs(S1.f32))) then
      if S2.f32 < 0.0F then
          D0.f32 = 5.0F
      else
          D0.f32 = 4.0F
      endif
  elsif abs(S1.f32) >= abs(S0.f32) then
      if S1.f32 < 0.0F then
          D0.f32 = 3.0F
      else
          D0.f32 = 2.0F
      endif
  else
      if S0.f32 < 0.0F then
          D0.f32 = 1.0F
      else
          D0.f32 = 0.0F
      endif
  endif
```

#### V_CUBESC_F32  (opcode 453)

Compute the cubemap S coordinate of a 3D coordinate specified as three single-precision float inputs. Store the
result in single-precision float format into a vector register.

```
  // D0.f = cubemap S coordinate.
  // XYZ coordinate is given in (S0.f, S1.f, S2.f).
  // S0.f = x
  // S1.f = y
  // S2.f = z
  if ((abs(S2.f32) >= abs(S0.f32)) && (abs(S2.f32) >= abs(S1.f32))) then
      if S2.f32 < 0.0F then
          D0.f32 = -S0.f32
      else
          D0.f32 = S0.f32
      endif
  elsif abs(S1.f32) >= abs(S0.f32) then
      D0.f32 = S0.f32
  else
      if S0.f32 < 0.0F then
          D0.f32 = S2.f32
      else
```

```
          D0.f32 = -S2.f32
      endif
  endif
```

#### V_CUBETC_F32  (opcode 454)

Compute the cubemap T coordinate of a 3D coordinate specified as three single-precision float inputs. Store
the result in single-precision float format into a vector register.

```
  // D0.f = cubemap T coordinate.
  // XYZ coordinate is given in (S0.f, S1.f, S2.f).
  // S0.f = x
  // S1.f = y
  // S2.f = z
  if ((abs(S2.f32) >= abs(S0.f32)) && (abs(S2.f32) >= abs(S1.f32))) then
      D0.f32 = -S1.f32
  elsif abs(S1.f32) >= abs(S0.f32) then
      if S1.f32 < 0.0F then
          D0.f32 = -S2.f32
      else
          D0.f32 = S2.f32
      endif
  else
      D0.f32 = -S1.f32
  endif
```

#### V_CUBEMA_F32  (opcode 455)

Compute the cubemap major axis of a 3D coordinate specified as three single-precision float inputs. Store the
result in single-precision float format into a vector register.

```
  // D0.f = 2.0 * cubemap major axis.
  // XYZ coordinate is given in (S0.f, S1.f, S2.f).
  // S0.f = x
  // S1.f = y
  // S2.f = z
  if ((abs(S2.f32) >= abs(S0.f32)) && (abs(S2.f32) >= abs(S1.f32))) then
      D0.f32 = S2.f32 * 2.0F
  elsif abs(S1.f32) >= abs(S0.f32) then
      D0.f32 = S1.f32 * 2.0F
  else
      D0.f32 = S0.f32 * 2.0F
  endif
```

#### V_BFE_U32  (opcode 456)

Extract an unsigned bitfield from the first input using field offset from the second input and size from the third
input, then store the result into a vector register.

```
  D0.u32 = ((S0.u32 >> S1[4 : 0].u32) & ((1U << S2[4 : 0].u32) - 1U))
```

#### V_BFE_I32  (opcode 457)

Extract a signed bitfield from the first input using field offset from the second input and size from the third
input, then store the result into a vector register.

```
  tmp.i32 = ((S0.i32 >> S1[4 : 0].u32) & ((1 << S2[4 : 0].u32) - 1));
  D0.i32 = signext_from_bit(tmp.i32, S2[4 : 0].u32)
```

#### V_BFI_B32  (opcode 458)

Overwrite a bitfield in the third input with a bitfield from the second input using a mask from the first input,
then store the result into a vector register.

```
  D0.u32 = ((S0.u32 & S1.u32) | (~S0.u32 & S2.u32))
```

#### V_FMA_F32  (opcode 459)

Multiply two single-precision float inputs and add a third input using fused multiply add, and store the result
into a vector register.

```
  D0.f32 = fma(S0.f32, S1.f32, S2.f32)
```

Notes

0.5ULP accuracy, denormals are supported.

#### V_FMA_F64  (opcode 460)

Multiply two double-precision float inputs and add a third input using fused multiply add, and store the result
into a vector register.

```
  D0.f64 = fma(S0.f64, S1.f64, S2.f64)
```

Notes

0.5ULP accuracy, denormals are supported.

#### V_LERP_U8  (opcode 461)

Average two 4-D vectors stored as packed bytes in the first two inputs with rounding control provided by the
third input, then store the result into a vector register. Each byte in the third input acts as a rounding mode for
the corresponding element; if the LSB is set then 0.5 rounds up, otherwise 0.5 truncates.

```
  tmp = ((S0.u32[31 : 24] + S1.u32[31 : 24] + S2.u32[24].u8) >> 1U << 24U);
  tmp += ((S0.u32[23 : 16] + S1.u32[23 : 16] + S2.u32[16].u8) >> 1U << 16U);
  tmp += ((S0.u32[15 : 8] + S1.u32[15 : 8] + S2.u32[8].u8) >> 1U << 8U);
  tmp += ((S0.u32[7 : 0] + S1.u32[7 : 0] + S2.u32[0].u8) >> 1U);
  D0.u32 = tmp.u32
```

#### V_ALIGNBIT_B32  (opcode 462)

Align a 64-bit value encoded in the first two inputs to a bit position specified in the third input, then store the
result into a 32-bit vector register.

```
  D0.u32 = 32'U(({ S0.u32, S1.u32 } >> S2.u32[4 : 0]) & 0xffffffffLL)
```

Notes

> S0 carries the MSBs and S1 carries the LSBs of the value being aligned.

#### V_ALIGNBYTE_B32  (opcode 463)

Align a 64-bit value encoded in the first two inputs to a byte position specified in the third input, then store the
result into a 32-bit vector register.

```
  D0.u32 = 32'U(({ S0.u32, S1.u32 } >> (S2.u32[1 : 0] * 8U)) & 0xffffffffLL)
```

Notes

> S0 carries the MSBs and S1 carries the LSBs of the value being aligned.

#### V_MIN3_F32  (opcode 464)

Select the minimum of three single-precision float inputs and store the selected value into a vector register.

```
  D0.f32 = v_min_f32(v_min_f32(S0.f32, S1.f32), S2.f32)
```

#### V_MIN3_I32  (opcode 465)

Select the minimum of three signed 32-bit integer inputs and store the selected value into a vector register.

```
  D0.i32 = v_min_i32(v_min_i32(S0.i32, S1.i32), S2.i32)
```

#### V_MIN3_U32  (opcode 466)

Select the minimum of three unsigned 32-bit integer inputs and store the selected value into a vector register.

```
  D0.u32 = v_min_u32(v_min_u32(S0.u32, S1.u32), S2.u32)
```

#### V_MAX3_F32  (opcode 467)

Select the maximum of three single-precision float inputs and store the selected value into a vector register.

```
  D0.f32 = v_max_f32(v_max_f32(S0.f32, S1.f32), S2.f32)
```

#### V_MAX3_I32  (opcode 468)

Select the maximum of three signed 32-bit integer inputs and store the selected value into a vector register.

```
  D0.i32 = v_max_i32(v_max_i32(S0.i32, S1.i32), S2.i32)
```

#### V_MAX3_U32  (opcode 469)

Select the maximum of three unsigned 32-bit integer inputs and store the selected value into a vector register.

```
  D0.u32 = v_max_u32(v_max_u32(S0.u32, S1.u32), S2.u32)
```

#### V_MED3_F32  (opcode 470)

Select the median of three single-precision float values and store the selected value into a vector register.

```
  if (isNAN(64'F(S0.f32)) || isNAN(64'F(S1.f32)) || isNAN(64'F(S2.f32))) then
      D0.f32 = v_min3_f32(S0.f32, S1.f32, S2.f32)
  elsif v_max3_f32(S0.f32, S1.f32, S2.f32) == S0.f32 then
      D0.f32 = v_max_f32(S1.f32, S2.f32)
  elsif v_max3_f32(S0.f32, S1.f32, S2.f32) == S1.f32 then
      D0.f32 = v_max_f32(S0.f32, S2.f32)
  else
      D0.f32 = v_max_f32(S0.f32, S1.f32)
  endif
```

#### V_MED3_I32  (opcode 471)

Select the median of three signed 32-bit integer values and store the selected value into a vector register.

```
  if v_max3_i32(S0.i32, S1.i32, S2.i32) == S0.i32 then
      D0.i32 = v_max_i32(S1.i32, S2.i32)
  elsif v_max3_i32(S0.i32, S1.i32, S2.i32) == S1.i32 then
      D0.i32 = v_max_i32(S0.i32, S2.i32)
  else
      D0.i32 = v_max_i32(S0.i32, S1.i32)
  endif
```

#### V_MED3_U32  (opcode 472)

Select the median of three unsigned 32-bit integer values and store the selected value into a vector register.

```
  if v_max3_u32(S0.u32, S1.u32, S2.u32) == S0.u32 then
      D0.u32 = v_max_u32(S1.u32, S2.u32)
  elsif v_max3_u32(S0.u32, S1.u32, S2.u32) == S1.u32 then
      D0.u32 = v_max_u32(S0.u32, S2.u32)
  else
      D0.u32 = v_max_u32(S0.u32, S1.u32)
  endif
```

#### V_SAD_U8  (opcode 473)

Calculate the sum of absolute differences of elements in two packed 4-component unsigned 8-bit integer
inputs, add an unsigned 32-bit integer value from the third input and store the result into a vector register.

```
  ABSDIFF = lambda(x, y) (
        x > y ? x - y : y - x);
  // UNSIGNED comparison
  tmp = S2.u32;
  tmp += 32'U(ABSDIFF(S0.u32[7 : 0], S1.u32[7 : 0]));
  tmp += 32'U(ABSDIFF(S0.u32[15 : 8], S1.u32[15 : 8]));
  tmp += 32'U(ABSDIFF(S0.u32[23 : 16], S1.u32[23 : 16]));
  tmp += 32'U(ABSDIFF(S0.u32[31 : 24], S1.u32[31 : 24]));
  D0.u32 = tmp
```

Notes

Overflow into the upper bits is allowed.

#### V_SAD_HI_U8  (opcode 474)

Calculate the sum of absolute differences of elements in two packed 4-component unsigned 8-bit integer
inputs, shift the sum left by 16 bits, add an unsigned 32-bit integer value from the third input and store the
result into a vector register.

```
  D0.u32 = (32'U(v_sad_u8(S0, S1, 0U)) << 16U) + S2.u32
```

Notes

Overflow into the upper bits is allowed.

#### V_SAD_U16  (opcode 475)

Calculate the sum of absolute differences of elements in two packed 2-component unsigned 16-bit integer
inputs, add an unsigned 32-bit integer value from the third input and store the result into a vector register.

```
  ABSDIFF = lambda(x, y) (
        x > y ? x - y : y - x);
  // UNSIGNED comparison
  tmp = S2.u32;
  tmp += ABSDIFF(S0[15 : 0].u16, S1[15 : 0].u16);
  tmp += ABSDIFF(S0[31 : 16].u16, S1[31 : 16].u16);
  D0.u32 = tmp
```

#### V_SAD_U32  (opcode 476)

Calculate the absolute difference of two unsigned 32-bit integer inputs, add an unsigned 32-bit integer value
from the third input and store the result into a vector register.

```
  ABSDIFF = lambda(x, y) (
      x > y ? x - y : y - x);
  // UNSIGNED comparison
  D0.u32 = ABSDIFF(S0.u32, S1.u32) + S2.u32
```

#### V_CVT_PK_U8_F32  (opcode 477)

Convert a single-precision float value from the first input to an unsigned 8-bit integer value and pack the result
into one byte of the third input using the second input as a byte select. Store the result into a vector register.

```
  tmp = (S2.u32 & 32'U(~(0xff << (S1.u32[1 : 0].u32 * 8U))));
  tmp = (tmp | ((32'U(f32_to_u8(S0.f32)) & 255U) << (S1.u32[1 : 0].u32 * 8U)));
  D0.u32 = tmp
```

#### V_DIV_FIXUP_F32  (opcode 478)

Given a single-precision float quotient in the first input, a denominator in the second input and a numerator in
the third input, detect and apply corner cases related to division, including divide by zero, NaN inputs and
overflow, and modify the quotient accordingly. Generate any invalid, denormal and divide-by-zero exceptions
that are a result of the division. Store the modified quotient into a vector register.

This operation handles corner cases in a division macro such as divide by zero and NaN inputs. This operation
is well defined when the quotient is approximately equal to the numerator divided by the denominator. Other
inputs produce a predictable result but may not be mathematically useful.

```
  sign_out = (sign(S1.f32) ^ sign(S2.f32));
  if isNAN(64'F(S2.f32)) then
      D0.f32 = 32'F(cvtToQuietNAN(64'F(S2.f32)))
  elsif isNAN(64'F(S1.f32)) then
      D0.f32 = 32'F(cvtToQuietNAN(64'F(S1.f32)))
  elsif ((64'F(S1.f32) == 0.0) && (64'F(S2.f32) == 0.0)) then
      // 0/0
      D0.f32 = 32'F(0xffc00000)
  elsif ((64'F(abs(S1.f32)) == +INF) && (64'F(abs(S2.f32)) == +INF)) then
      // inf/inf
      D0.f32 = 32'F(0xffc00000)
  elsif ((64'F(S1.f32) == 0.0) || (64'F(abs(S2.f32)) == +INF)) then
      // x/0, or inf/y
      D0.f32 = sign_out ? -INF.f32 : +INF.f32
  elsif ((64'F(abs(S1.f32)) == +INF) || (64'F(S2.f32) == 0.0)) then
      // x/inf, 0/y
      D0.f32 = sign_out ? -0.0F : 0.0F
  elsif exponent(S2.f32) - exponent(S1.f32) < -150 then
      D0.f32 = sign_out ? -UNDERFLOW_F32 : UNDERFLOW_F32
  elsif exponent(S1.f32) == 255 then
      D0.f32 = sign_out ? -OVERFLOW_F32 : OVERFLOW_F32
  else
```

```
        D0.f32 = sign_out ? -abs(S0.f32) : abs(S0.f32)
  endif
```

Notes

This operation is the final step of a high precision division macro and handles all exceptional cases of division.

#### V_DIV_FIXUP_F64  (opcode 479)

Given a double-precision float quotient in the first input, a denominator in the second input and a numerator
in the third input, detect and apply corner cases related to division, including divide by zero, NaN inputs and
overflow, and modify the quotient accordingly. Generate any invalid, denormal and divide-by-zero exceptions
that are a result of the division. Store the modified quotient into a vector register.

This operation handles corner cases in a division macro such as divide by zero and NaN inputs. This operation
is well defined when the quotient is approximately equal to the numerator divided by the denominator. Other
inputs produce a predictable result but may not be mathematically useful.

```
  sign_out = (sign(S1.f64) ^ sign(S2.f64));
  if isNAN(S2.f64) then
        D0.f64 = cvtToQuietNAN(S2.f64)
  elsif isNAN(S1.f64) then
        D0.f64 = cvtToQuietNAN(S1.f64)
  elsif ((S1.f64 == 0.0) && (S2.f64 == 0.0)) then
        // 0/0
        D0.f64 = 64'F(0xfff8000000000000LL)
  elsif ((abs(S1.f64) == +INF) && (abs(S2.f64) == +INF)) then
        // inf/inf
        D0.f64 = 64'F(0xfff8000000000000LL)
  elsif ((S1.f64 == 0.0) || (abs(S2.f64) == +INF)) then
        // x/0, or inf/y
        D0.f64 = sign_out ? -INF : +INF
  elsif ((abs(S1.f64) == +INF) || (S2.f64 == 0.0)) then
        // x/inf, 0/y
        D0.f64 = sign_out ? -0.0 : 0.0
  elsif exponent(S2.f64) - exponent(S1.f64) < -1075 then
        D0.f64 = sign_out ? -UNDERFLOW_F64 : UNDERFLOW_F64
  elsif exponent(S1.f64) == 2047 then
        D0.f64 = sign_out ? -OVERFLOW_F64 : OVERFLOW_F64
  else
        D0.f64 = sign_out ? -abs(S0.f64) : abs(S0.f64)
  endif
```

Notes

This operation is the final step of a high precision division macro and handles all exceptional cases of division.

#### V_DIV_SCALE_F32  (opcode 480)

Given a single-precision float value to scale in the first input, a denominator in the second input and a
numerator in the third input, scale the first input for division if required to avoid subnormal terms appearing
during application of the Newton-Raphson correction method. Store the scaled result into a vector register and
set the vector condition code iff post-scaling is required.

This operation is designed for use in a high precision division macro. The first input should be the same value
as either the second or third input; other scale values produce predictable results but may not be
mathematically useful. The vector condition code is used by V_DIV_FMAS_F32 to determine if the quotient
requires post-scaling.

```
  VCC = 0x0LL;
  if ((64'F(S2.f32) == 0.0) || (64'F(S1.f32) == 0.0)) then
        D0.f32 = NAN.f32
  elsif exponent(S2.f32) - exponent(S1.f32) >= 96 then
        // N/D near MAX_FLOAT_F32
        VCC = 0x1LL;
        if S0.f32 == S1.f32 then
            // Only scale the denominator
            D0.f32 = ldexp(S0.f32, 64)
        endif
  elsif S1.f32 == DENORM.f32 then
        D0.f32 = ldexp(S0.f32, 64)
  elsif ((1.0 / 64'F(S1.f32) == DENORM.f64) && (S2.f32 / S1.f32 == DENORM.f32)) then
        VCC = 0x1LL;
        if S0.f32 == S1.f32 then
            // Only scale the denominator
            D0.f32 = ldexp(S0.f32, 64)
        endif
  elsif 1.0 / 64'F(S1.f32) == DENORM.f64 then
        D0.f32 = ldexp(S0.f32, -64)
  elsif S2.f32 / S1.f32 == DENORM.f32 then
        VCC = 0x1LL;
        if S0.f32 == S2.f32 then
            // Only scale the numerator
            D0.f32 = ldexp(S0.f32, 64)
        endif
  elsif exponent(S2.f32) <= 23 then
        // Numerator is tiny
        D0.f32 = ldexp(S0.f32, 64)
  endif
```

Notes

V_DIV_SCALE_F32, V_DIV_FMAS_F32 and V_DIV_FIXUP_F32 are all designed for use in a high precision
division macro that utilizes V_RCP_F32 and V_MUL_F32 to compute the approximate result and then applies
two steps of the Newton-Raphson method to converge to the quotient. If subnormal terms appear during this
calculation then a loss of precision occurs. This loss of precision can be avoided by scaling the inputs and then
post-scaling the quotient after Newton-Raphson is applied.

#### V_DIV_SCALE_F64  (opcode 481)

Given a double-precision float value to scale in the first input, a denominator in the second input and a

numerator in the third input, scale the first input for division if required to avoid subnormal terms appearing
during application of the Newton-Raphson correction method. Store the scaled result into a vector register and
set the vector condition code iff post-scaling is required.

This operation is designed for use in a high precision division macro. The first input should be the same value
as either the second or third input; other scale values produce predictable results but may not be
mathematically useful. The vector condition code is used by V_DIV_FMAS_F64 to determine if the quotient
requires post-scaling.

```
  VCC = 0x0LL;
  if ((S2.f64 == 0.0) || (S1.f64 == 0.0)) then
        D0.f64 = NAN.f64
  elsif exponent(S2.f64) - exponent(S1.f64) >= 768 then
        // N/D near MAX_FLOAT_F64
        VCC = 0x1LL;
        if S0.f64 == S1.f64 then
            // Only scale the denominator
            D0.f64 = ldexp(S0.f64, 128)
        endif
  elsif S1.f64 == DENORM.f64 then
        D0.f64 = ldexp(S0.f64, 128)
  elsif ((1.0 / S1.f64 == DENORM.f64) && (S2.f64 / S1.f64 == DENORM.f64)) then
        VCC = 0x1LL;
        if S0.f64 == S1.f64 then
            // Only scale the denominator
            D0.f64 = ldexp(S0.f64, 128)
        endif
  elsif 1.0 / S1.f64 == DENORM.f64 then
        D0.f64 = ldexp(S0.f64, -128)
  elsif S2.f64 / S1.f64 == DENORM.f64 then
        VCC = 0x1LL;
        if S0.f64 == S2.f64 then
            // Only scale the numerator
            D0.f64 = ldexp(S0.f64, 128)
        endif
  elsif exponent(S2.f64) <= 53 then
        // Numerator is tiny
        D0.f64 = ldexp(S0.f64, 128)
  endif
```

Notes

V_DIV_SCALE_F64, V_DIV_FMAS_F64 and V_DIV_FIXUP_F64 are all designed for use in a high precision
division macro that utilizes V_RCP_F64 and V_MUL_F64 to compute the approximate result and then applies
two steps of the Newton-Raphson method to converge to the quotient. If subnormal terms appear during this
calculation then a loss of precision occurs. This loss of precision can be avoided by scaling the inputs and then
post-scaling the quotient after Newton-Raphson is applied.

#### V_DIV_FMAS_F32  (opcode 482)

Multiply two single-precision float inputs and add a third input using fused multiply add, then scale the
exponent of the result by a fixed factor if the vector condition code is set. Store the result into a vector register.

This operation is designed for use in floating point division macros and relies on V_DIV_SCALE_F32 to set the
vector condition code iff the quotient requires post-scaling.

```
  if VCC.u64[laneId] then
        D0.f32 = 2.0F ** 32 * fma(S0.f32, S1.f32, S2.f32)
  else
        D0.f32 = fma(S0.f32, S1.f32, S2.f32)
  endif
```

Notes

Input denormals are not flushed but output flushing is allowed.

V_DIV_SCALE_F32, V_DIV_FMAS_F32 and V_DIV_FIXUP_F32 are all designed for use in a high precision
division macro that utilizes V_RCP_F32 and V_MUL_F32 to compute the approximate result and then applies
two steps of the Newton-Raphson method to converge to the quotient. If subnormal terms appear during this
calculation then a loss of precision occurs. This loss of precision can be avoided by scaling the inputs and then
post-scaling the quotient after Newton-Raphson is applied.

#### V_DIV_FMAS_F64  (opcode 483)

Multiply two double-precision float inputs and add a third input using fused multiply add, then scale the
exponent of the result by a fixed factor if the vector condition code is set. Store the result into a vector register.

This operation is designed for use in floating point division macros and relies on V_DIV_SCALE_F64 to set the
vector condition code iff the quotient requires post-scaling.

```
  if VCC.u64[laneId] then
        D0.f64 = 2.0 ** 64 * fma(S0.f64, S1.f64, S2.f64)
  else
        D0.f64 = fma(S0.f64, S1.f64, S2.f64)
  endif
```

Notes

Input denormals are not flushed but output flushing is allowed.

V_DIV_SCALE_F64, V_DIV_FMAS_F64 and V_DIV_FIXUP_F64 are all designed for use in a high precision
division macro that utilizes V_RCP_F64 and V_MUL_F64 to compute the approximate result and then applies
two steps of the Newton-Raphson method to converge to the quotient. If subnormal terms appear during this
calculation then a loss of precision occurs. This loss of precision can be avoided by scaling the inputs and then
post-scaling the quotient after Newton-Raphson is applied.

#### V_MSAD_U8  (opcode 484)

Calculate the sum of absolute differences of elements in two packed 4-component unsigned 8-bit integer

inputs, except that elements where the second input (known as the reference input) is zero are not included in
the sum. Add an unsigned 32-bit integer value from the third input and store the result into a vector register.

```
  ABSDIFF = lambda(x, y) (
        x > y ? x - y : y - x);
  // UNSIGNED comparison
  tmp = S2.u32;
  tmp += S1.u32[7 : 0] == 8'0U ? 0U : 32'U(ABSDIFF(S0.u32[7 : 0], S1.u32[7 : 0]));
  tmp += S1.u32[15 : 8] == 8'0U ? 0U : 32'U(ABSDIFF(S0.u32[15 : 8], S1.u32[15 : 8]));
  tmp += S1.u32[23 : 16] == 8'0U ? 0U : 32'U(ABSDIFF(S0.u32[23 : 16], S1.u32[23 : 16]));
  tmp += S1.u32[31 : 24] == 8'0U ? 0U : 32'U(ABSDIFF(S0.u32[31 : 24], S1.u32[31 : 24]));
  D0.u32 = tmp
```

Notes

Overflow into the upper bits is allowed.

#### V_QSAD_PK_U16_U8  (opcode 485)

Perform the V_SAD_U8 operation four times using different slices of the first array, all entries of the second
array and each entry of the third array. Truncate each result to 16 bits, pack the values into a 4-entry array and
store the array into a vector register. The first input is an 8-entry array of unsigned 8-bit integers, the second
input is a 4-entry array of unsigned 8-bit integers and the third input is a 4-entry array of unsigned 16-bit
integers.

```
  tmp[63 : 48] = 16'B(v_sad_u8(S0[55 : 24], S1[31 : 0], S2[63 : 48].u32));
  tmp[47 : 32] = 16'B(v_sad_u8(S0[47 : 16], S1[31 : 0], S2[47 : 32].u32));
  tmp[31 : 16] = 16'B(v_sad_u8(S0[39 : 8], S1[31 : 0], S2[31 : 16].u32));
  tmp[15 : 0] = 16'B(v_sad_u8(S0[31 : 0], S1[31 : 0], S2[15 : 0].u32));
  D0.b64 = tmp.b64
```

#### V_MQSAD_PK_U16_U8  (opcode 486)

Perform the V_MSAD_U8 operation four times using different slices of the first array, all entries of the second
array and each entry of the third array. Truncate each result to 16 bits, pack the values into a 4-entry array and
store the array into a vector register. The first input is an 8-entry array of unsigned 8-bit integers, the second
input is a 4-entry array of unsigned 8-bit integers and the third input is a 4-entry array of unsigned 16-bit
integers.

```
  tmp[63 : 48] = 16'B(v_msad_u8(S0[55 : 24], S1[31 : 0], S2[63 : 48].u32));
  tmp[47 : 32] = 16'B(v_msad_u8(S0[47 : 16], S1[31 : 0], S2[47 : 32].u32));
  tmp[31 : 16] = 16'B(v_msad_u8(S0[39 : 8], S1[31 : 0], S2[31 : 16].u32));
  tmp[15 : 0] = 16'B(v_msad_u8(S0[31 : 0], S1[31 : 0], S2[15 : 0].u32));
  D0.b64 = tmp.b64
```

#### V_MQSAD_U32_U8  (opcode 487)

Perform the V_MSAD_U8 operation four times using different slices of the first array, all entries of the second
array and each entry of the third array. Pack each 32-bit value into a 4-entry array and store the array into a
vector register. The first input is an 8-entry array of unsigned 8-bit integers, the second input is a 4-entry array
of unsigned 8-bit integers and the third input is a 4-entry array of unsigned 32-bit integers.

```
  tmp[127 : 96] = 32'B(v_msad_u8(S0[55 : 24], S1[31 : 0], S2[127 : 96].u32));
  tmp[95 : 64] = 32'B(v_msad_u8(S0[47 : 16], S1[31 : 0], S2[95 : 64].u32));
  tmp[63 : 32] = 32'B(v_msad_u8(S0[39 : 8], S1[31 : 0], S2[63 : 32].u32));
  tmp[31 : 0] = 32'B(v_msad_u8(S0[31 : 0], S1[31 : 0], S2[31 : 0].u32));
  D0.b128 = tmp.b128
```

#### V_MAD_U64_U32  (opcode 488)

Multiply two unsigned integer inputs, add a third unsigned integer input, store the result into a 64-bit vector
register and store the overflow/carryout into a scalar mask register.

```
  { D1.u1, D0.u64 } = 65'B(65'U(S0.u32) * 65'U(S1.u32) + 65'U(S2.u64))
```

Notes

In VOP3 the VCC destination may be an arbitrary SGPR-pair.

#### V_MAD_I64_I32  (opcode 489)

Multiply two signed integer inputs, add a third signed integer input, store the result into a 64-bit vector register
and store the overflow/carryout into a scalar mask register.

```
  { D1.i1, D0.i64 } = 65'B(65'I(S0.i32) * 65'I(S1.i32) + 65'I(S2.i64))
```

Notes

In VOP3 the VCC destination may be an arbitrary SGPR-pair.

#### V_MAD_LEGACY_F16  (opcode 490)

Multiply add of FP16 values. Implements IEEE rules and non-standard rule for OPSEL.

```
  tmp = S0.f16 * S1.f16 + S2.f16;
  if OPSEL.u4[3] then
        D0 = { tmp.f16, D0[15 : 0] }
```

```
  else
        D0 = { 16'0, tmp.f16 }
  endif
```

Notes

Supports round mode, exception flags, saturation.

If OPSEL[3] is 0 Result is written to 16 LSBs of destination VGPR and hi 16 bits are written as 0 (this is different
from V_MAD_F16).

If OPSEL[3] is 1 Result is written to 16 MSBs of destination VGPR and lo 16 bits are preserved.

#### V_MAD_LEGACY_U16  (opcode 491)

Multiply add of unsigned short values. Has non-standard rule for OPSEL.

```
  tmp = S0.u16 * S1.u16 + S2.u16;
  if OPSEL.u4[3] then
        D0 = { tmp.u16, D0[15 : 0] }
  else
        D0 = { 16'0, tmp.u16 }
  endif
```

Notes

Supports saturation (unsigned 16-bit integer domain).

If OPSEL[3] is 0 Result is written to 16 LSBs of destination VGPR and hi 16 bits are written as 0 (this is different
from V_MAD_U16).

If OPSEL[3] is 1 Result is written to 16 MSBs of destination VGPR and lo 16 bits are preserved.

#### V_MAD_LEGACY_I16  (opcode 492)

Multiply add of signed short values. Has non-standard rule for OPSEL.

```
  tmp = S0.i16 * S1.i16 + S2.i16;
  if OPSEL.u4[3] then
        D0 = { tmp.i16, D0[15 : 0] }
  else
        D0 = { 16'0, tmp.i16 }
  endif
```

Notes

Supports saturation (signed 16-bit integer domain).

If OPSEL[3] is 0 Result is written to 16 LSBs of destination VGPR and hi 16 bits are written as 0 (this is different
from V_MAD_I16).

If OPSEL[3] is 1 Result is written to 16 MSBs of destination VGPR and lo 16 bits are preserved.

#### V_PERM_B32  (opcode 493)

Permute a 64-bit value constructed from two vector inputs (most significant bits come from the first input)
using a per-lane selector from the third input. The lane selector allows each byte of the result to choose from
any of the 8 input bytes, perform sign extension or pad with 0/1 bits. Store the result into a vector register.

```
  BYTE_PERMUTE = lambda(data, sel) (
        declare in : 8'B[8];
        for i in 0 : 7 do
            in[i] = data[i * 8 + 7 : i * 8].b8
        endfor;
        if sel.u32 >= 13U then
            return 8'0xff
        elsif sel.u32 == 12U then
            return 8'0x0
        elsif sel.u32 == 11U then
            return in[7][7].b8 * 8'0xff
        elsif sel.u32 == 10U then
            return in[5][7].b8 * 8'0xff
        elsif sel.u32 == 9U then
            return in[3][7].b8 * 8'0xff
        elsif sel.u32 == 8U then
            return in[1][7].b8 * 8'0xff
        else
            return in[sel]
        endif);
  D0[31 : 24] = BYTE_PERMUTE({ S0.u32, S1.u32 }, S2.u32[31 : 24]);
  D0[23 : 16] = BYTE_PERMUTE({ S0.u32, S1.u32 }, S2.u32[23 : 16]);
  D0[15 : 8] = BYTE_PERMUTE({ S0.u32, S1.u32 }, S2.u32[15 : 8]);
  D0[7 : 0] = BYTE_PERMUTE({ S0.u32, S1.u32 }, S2.u32[7 : 0])
```

Notes

Selects 0 through 7 select the corresponding byte of the 64-bit input value.

Selects 8 through 11 are useful in modeling sign extension of a smaller-precision signed integer to a larger-
precision result by replicating the leading bit of a selected byte.

Selects 12 and 13 return padding values of 0 and 1 bits respectively.

Note the MSBs of the 64-bit value being selected are stored in S0. This is counterintuitive for a little-endian
architecture.

#### V_FMA_LEGACY_F16  (opcode 494)

Fused half precision multiply add. Implements IEEE rules and non-standard rule for OPSEL.

```
  tmp = fma(S0.f16, S1.f16, S2.f16);
  if OPSEL.u4[3] then
      D0 = { tmp.f16, D0[15 : 0] }
  else
      D0 = { 16'0, tmp.f16 }
  endif
```

#### V_DIV_FIXUP_LEGACY_F16  (opcode 495)

Half precision division fixup. Has non-standard rule for OPSEL.

S0 = Quotient, S1 = Denominator, S2 = Numerator.

Given a numerator, denominator, and quotient from a divide, this opcode detects and applies specific case
numerics, touching up the quotient if necessary. This opcode also generates invalid, denorm and divide by
zero exceptions caused by the division.

```
  sign_out = (sign(S1.f16) ^ sign(S2.f16));
  if isNAN(64'F(S2.f16)) then
      tmp = cvtToQuietNAN(64'F(S2.f16))
  elsif isNAN(64'F(S1.f16)) then
      tmp = cvtToQuietNAN(64'F(S1.f16))
  elsif ((64'F(S1.f16) == 0.0) && (64'F(S2.f16) == 0.0)) then
      // 0/0
      tmp = 16'F(0xfe00)
  elsif ((64'F(abs(S1.f16)) == +INF) && (64'F(abs(S2.f16)) == +INF)) then
      // inf/inf
      tmp = 16'F(0xfe00)
  elsif ((64'F(S1.f16) == 0.0) || (64'F(abs(S2.f16)) == +INF)) then
      // x/0, or inf/y
      tmp = sign_out ? -INF : +INF
  elsif ((64'F(abs(S1.f16)) == +INF) || (64'F(S2.f16) == 0.0)) then
      // x/inf, 0/y
      tmp = sign_out ? -0.0 : 0.0
  else
      tmp = sign_out ? -abs(S0.f16) : abs(S0.f16)
  endif;
  if OPSEL.u4[3] then
      D0 = { tmp.f16, D0[15 : 0] }
  else
      D0 = { 16'0, tmp.f16 }
  endif
```

#### V_CVT_PKACCUM_U8_F32  (opcode 496)

Convert a single-precision float value in the first input to an unsigned 8-bit integer value and store the result
into one byte of the destination register using the second input as a byte select.

```
  byte = S1.u32[1 : 0];
  bit = byte.u32 * 8U;
  D0.u32[bit + 7U : bit] = 32'U(f32_to_u8(S0.f32))
```

Notes

This opcode uses src_c to pass destination in as a source.

#### V_MAD_U32_U16  (opcode 497)

Multiply two unsigned 16-bit integer inputs in the unsigned 32-bit integer domain, add an unsigned 32-bit
integer value from a third input, and store the result as an unsigned 32-bit integer into a vector register.

```
  D0.u32 = 32'U(S0.u16) * 32'U(S1.u16) + S2.u32
```

#### V_MAD_I32_I16  (opcode 498)

Multiply two signed 16-bit integer inputs in the signed 32-bit integer domain, add a signed 32-bit integer value
from a third input, and store the result as a signed 32-bit integer into a vector register.

```
  D0.i32 = 32'I(S0.i16) * 32'I(S1.i16) + S2.i32
```

#### V_XAD_U32  (opcode 499)

Calculate bitwise XOR of the first two vector inputs, then add the third vector input to the intermediate result,
then store the final result into a vector register.

```
  D0.u32 = (S0.u32 ^ S1.u32) + S2.u32
```

Notes

No carryin/carryout and no saturation. This opcode is designed to help accelerate the SHA256 hash algorithm.

#### V_MIN3_F16  (opcode 500)

Select the minimum of three half-precision float inputs and store the selected value into a vector register.

```
  D0.f16 = v_min_f16(v_min_f16(S0.f16, S1.f16), S2.f16)
```

#### V_MIN3_I16  (opcode 501)

Select the minimum of three signed 16-bit integer inputs and store the selected value into a vector register.

```
  D0.i16 = v_min_i16(v_min_i16(S0.i16, S1.i16), S2.i16)
```

#### V_MIN3_U16  (opcode 502)

Select the minimum of three unsigned 16-bit integer inputs and store the selected value into a vector register.

```
  D0.u16 = v_min_u16(v_min_u16(S0.u16, S1.u16), S2.u16)
```

#### V_MAX3_F16  (opcode 503)

Select the maximum of three half-precision float inputs and store the selected value into a vector register.

```
  D0.f16 = v_max_f16(v_max_f16(S0.f16, S1.f16), S2.f16)
```

#### V_MAX3_I16  (opcode 504)

Select the maximum of three signed 16-bit integer inputs and store the selected value into a vector register.

```
  D0.i16 = v_max_i16(v_max_i16(S0.i16, S1.i16), S2.i16)
```

#### V_MAX3_U16  (opcode 505)

Select the maximum of three unsigned 16-bit integer inputs and store the selected value into a vector register.

```
  D0.u16 = v_max_u16(v_max_u16(S0.u16, S1.u16), S2.u16)
```

#### V_MED3_F16  (opcode 506)

Select the median of three half-precision float values and store the selected value into a vector register.

```
  if (isNAN(64'F(S0.f16)) || isNAN(64'F(S1.f16)) || isNAN(64'F(S2.f16))) then
      D0.f16 = v_min3_f16(S0.f16, S1.f16, S2.f16)
  elsif v_max3_f16(S0.f16, S1.f16, S2.f16) == S0.f16 then
      D0.f16 = v_max_f16(S1.f16, S2.f16)
  elsif v_max3_f16(S0.f16, S1.f16, S2.f16) == S1.f16 then
      D0.f16 = v_max_f16(S0.f16, S2.f16)
  else
      D0.f16 = v_max_f16(S0.f16, S1.f16)
  endif
```

#### V_MED3_I16  (opcode 507)

Select the median of three signed 16-bit integer values and store the selected value into a vector register.

```
  if v_max3_i16(S0.i16, S1.i16, S2.i16) == S0.i16 then
      D0.i16 = v_max_i16(S1.i16, S2.i16)
  elsif v_max3_i16(S0.i16, S1.i16, S2.i16) == S1.i16 then
      D0.i16 = v_max_i16(S0.i16, S2.i16)
  else
      D0.i16 = v_max_i16(S0.i16, S1.i16)
  endif
```

#### V_MED3_U16  (opcode 508)

Select the median of three unsigned 16-bit integer values and store the selected value into a vector register.

```
  if v_max3_u16(S0.u16, S1.u16, S2.u16) == S0.u16 then
      D0.u16 = v_max_u16(S1.u16, S2.u16)
  elsif v_max3_u16(S0.u16, S1.u16, S2.u16) == S1.u16 then
      D0.u16 = v_max_u16(S0.u16, S2.u16)
  else
      D0.u16 = v_max_u16(S0.u16, S1.u16)
  endif
```

#### V_LSHL_ADD_U32  (opcode 509)

Given a shift count in the second input, calculate the logical shift left of the first input, then add the third input
to the intermediate result, then store the final result into a vector register.

```
  D0.u32 = (S0.u32 << S1.u32[4 : 0].u32) + S2.u32
```

#### V_ADD_LSHL_U32  (opcode 510)

Add the first two integer inputs, then given a shift count in the third input, calculate the logical shift left of the
intermediate result, then store the final result into a vector register.

```
  D0.u32 = ((S0.u32 + S1.u32) << S2.u32[4 : 0].u32)
```

#### V_ADD3_U32  (opcode 511)

Add three unsigned inputs and store the result into a vector register. No carry-in or carry-out support.

```
  D0.u32 = S0.u32 + S1.u32 + S2.u32
```

#### V_LSHL_OR_B32  (opcode 512)

Given a shift count in the second input, calculate the logical shift left of the first input, then calculate the
bitwise OR of the intermediate result and the third input, then store the final result into a vector register.

```
  D0.u32 = ((S0.u32 << S1.u32[4 : 0].u32) | S2.u32)
```

#### V_AND_OR_B32  (opcode 513)

Calculate bitwise AND on the first two vector inputs, then compute the bitwise OR of the intermediate result
and the third vector input, then store the final result into a vector register.

```
  D0.u32 = ((S0.u32 & S1.u32) | S2.u32)
```

Notes

Input and output modifiers not supported.

#### V_OR3_B32  (opcode 514)

Calculate the bitwise OR of three vector inputs and store the result into a vector register.

```
  D0.u32 = (S0.u32 | S1.u32 | S2.u32)
```

Notes

Input and output modifiers not supported.

#### V_MAD_F16  (opcode 515)

Multiply two half-precision float inputs and add a third input, and store the result into a vector register.

```
  D0.f16 = S0.f16 * S1.f16 + S2.f16
```

Notes

Supports round mode, exception flags, saturation. 1ULP accuracy, denormals are flushed.

If OPSEL[3] is 0 Result is written to 16 LSBs of destination VGPR and hi 16 bits are preserved.

If OPSEL[3] is 1 Result is written to 16 MSBs of destination VGPR and lo 16 bits are preserved.

#### V_MAD_U16  (opcode 516)

Multiply two unsigned 16-bit integer inputs, add an unsigned 16-bit integer value from a third input, and store
the result into a vector register.

```
  D0.u16 = S0.u16 * S1.u16 + S2.u16
```

Notes

Supports saturation (unsigned 16-bit integer domain).

If OPSEL[3] is 0 the result is written to 16 LSBs of destination VGPR and the high 16 bits are preserved.

If OPSEL[3] is 1 the result is written to 16 MSBs of destination VGPR and the low 16 bits are preserved.

#### V_MAD_I16  (opcode 517)

Multiply two signed 16-bit integer inputs, add a signed 16-bit integer value from a third input, and store the
result into a vector register.

```
  D0.i16 = S0.i16 * S1.i16 + S2.i16
```

Notes

Supports saturation (signed 16-bit integer domain).

If OPSEL[3] is 0 the result is written to 16 LSBs of destination VGPR and the high 16 bits are preserved.

If OPSEL[3] is 1 the result is written to 16 MSBs of destination VGPR and the low 16 bits are preserved.

#### V_FMA_F16  (opcode 518)

Multiply two half-precision float inputs and add a third input using fused multiply add, and store the result into
a vector register.

```
  D0.f16 = fma(S0.f16, S1.f16, S2.f16)
```

Notes

0.5ULP accuracy, denormals are supported.

If OPSEL[3] is 0 Result is written to 16 LSBs of destination VGPR and hi 16 bits are preserved.

If OPSEL[3] is 1 Result is written to 16 MSBs of destination VGPR and lo 16 bits are preserved.

#### V_DIV_FIXUP_F16  (opcode 519)

Given a half-precision float quotient in the first input, a denominator in the second input and a numerator in
the third input, detect and apply corner cases related to division, including divide by zero, NaN inputs and
overflow, and modify the quotient accordingly. Generate any invalid, denormal and divide-by-zero exceptions
that are a result of the division. Store the modified quotient into a vector register.

This operation handles corner cases in a division macro such as divide by zero and NaN inputs. This operation
is well defined when the quotient is approximately equal to the numerator divided by the denominator. Other
inputs produce a predictable result but may not be mathematically useful.

```
  sign_out = (sign(S1.f16) ^ sign(S2.f16));
  if isNAN(64'F(S2.f16)) then
        D0.f16 = 16'F(cvtToQuietNAN(64'F(S2.f16)))
  elsif isNAN(64'F(S1.f16)) then
        D0.f16 = 16'F(cvtToQuietNAN(64'F(S1.f16)))
  elsif ((64'F(S1.f16) == 0.0) && (64'F(S2.f16) == 0.0)) then
        // 0/0
        D0.f16 = 16'F(0xfe00)
  elsif ((64'F(abs(S1.f16)) == +INF) && (64'F(abs(S2.f16)) == +INF)) then
        // inf/inf
        D0.f16 = 16'F(0xfe00)
  elsif ((64'F(S1.f16) == 0.0) || (64'F(abs(S2.f16)) == +INF)) then
        // x/0, or inf/y
        D0.f16 = sign_out ? -INF.f16 : +INF.f16
  elsif ((64'F(abs(S1.f16)) == +INF) || (64'F(S2.f16) == 0.0)) then
        // x/inf, 0/y
        D0.f16 = sign_out ? -16'0.0 : 16'0.0
  else
```

```
        D0.f16 = sign_out ? -abs(S0.f16) : abs(S0.f16)
  endif
```

Notes

This operation is the final step of a high precision division macro and handles all exceptional cases of division.

If OPSEL[3] is 0 Result is written to 16 LSBs of destination VGPR and hi 16 bits are preserved.

If OPSEL[3] is 1 Result is written to 16 MSBs of destination VGPR and lo 16 bits are preserved.

#### V_LSHL_ADD_U64  (opcode 520)

Given a shift count in the second input, calculate the logical shift left of the first input, then add the third input
to the intermediate result, then store the final result into a vector register.

For this opcode the shift count must be between 0 and 4, higher shift counts are unsupported.

```
  D0.u64 = (S0.u64 << S1.u32[2 : 0].u32) + S2.u64
```

Notes

The design treats unsupported shift counts as a shift of zero.

#### V_BITOP3_B16  (opcode 563)

Calculate the generic bitwise operation of three 16-bit vector inputs using a truth table encoded in the
instruction and store the result into a vector register.

```
  TTBL = { INST.OMOD[1 : 0], INST.ABS[2 : 0], INST.NEG[2 : 0] };
  tmp = 16'0U;
  tmp = (tmp | (32'I(TTBL.b32 & 0x1) != 0 ? 16'U(~S0.b16 & ~S1.b16 & ~S2.b16) : 16'0U));
  tmp = (tmp | (32'I(TTBL.b32 & 0x2) != 0 ? 16'U(~S0.b16 & ~S1.b16 & S2.b16) : 16'0U));
  tmp = (tmp | (32'I(TTBL.b32 & 0x4) != 0 ? 16'U(~S0.b16 & S1.b16 & ~S2.b16) : 16'0U));
  tmp = (tmp | (32'I(TTBL.b32 & 0x8) != 0 ? 16'U(~S0.b16 & S1.b16 & S2.b16) : 16'0U));
  tmp = (tmp | (32'I(TTBL.b32 & 0x10) != 0 ? 16'U(S0.b16 & ~S1.b16 & ~S2.b16) : 16'0U));
  tmp = (tmp | (32'I(TTBL.b32 & 0x20) != 0 ? 16'U(S0.b16 & ~S1.b16 & S2.b16) : 16'0U));
  tmp = (tmp | (32'I(TTBL.b32 & 0x40) != 0 ? 16'U(S0.b16 & S1.b16 & ~S2.b16) : 16'0U));
  tmp = (tmp | (32'I(TTBL.b32 & 0x80) != 0 ? 16'U(S0.b16 & S1.b16 & S2.b16) : 16'0U));
  D.b16 = tmp.b16
```

Notes

The truth table is encoded as a SIMM8 value overloading the OMOD, ABS and NEG fields of the instruction.
Normal output modifier, absolute value and negation controls are disabled for this instruction. The truth table
is encoded as

```
  { OMOD[1:0], ABS[2:0], NEG[2:0] }
```

Given the i'th bit of inputs S0, S1 and S2, the i'th bit of the result D0 can be determined by looking up a bit in
TTBL:

```
  D0[i] = TTBL[{S0[i], S1[i], S2[i]}]
```

Equivalencies with other bitwise ops:

Opcode S0 S1 S2 TTBL
V_AND A B 0 0x40
V_OR A B 0 0x54
V_XOR A B 0 0x14
V_XNOR A B 0 0x41
V_NOT A 0 0 0x01
V_AND_OR A B C 0xea
V_OR3 A B C 0xfe
V_XOR3 A B C 0x96

#### V_BITOP3_B32  (opcode 564)

Calculate the generic bitwise operation of three 32-bit vector inputs using a truth table encoded in the
instruction and store the result into a vector register.

```
  TTBL = { INST.OMOD[1 : 0], INST.ABS[2 : 0], INST.NEG[2 : 0] };
  tmp = 0U;
  tmp = (tmp | (32'I(TTBL.b32 & 0x1) != 0 ? 32'U(~S0.b32 & ~S1.b32 & ~S2.b32) : 0U));
  tmp = (tmp | (32'I(TTBL.b32 & 0x2) != 0 ? 32'U(~S0.b32 & ~S1.b32 & S2.b32) : 0U));
  tmp = (tmp | (32'I(TTBL.b32 & 0x4) != 0 ? 32'U(~S0.b32 & S1.b32 & ~S2.b32) : 0U));
  tmp = (tmp | (32'I(TTBL.b32 & 0x8) != 0 ? 32'U(~S0.b32 & S1.b32 & S2.b32) : 0U));
  tmp = (tmp | (32'I(TTBL.b32 & 0x10) != 0 ? 32'U(S0.b32 & ~S1.b32 & ~S2.b32) : 0U));
  tmp = (tmp | (32'I(TTBL.b32 & 0x20) != 0 ? 32'U(S0.b32 & ~S1.b32 & S2.b32) : 0U));
  tmp = (tmp | (32'I(TTBL.b32 & 0x40) != 0 ? 32'U(S0.b32 & S1.b32 & ~S2.b32) : 0U));
  tmp = (tmp | (32'I(TTBL.b32 & 0x80) != 0 ? 32'U(S0.b32 & S1.b32 & S2.b32) : 0U));
  D.b32 = tmp.b32
```

Notes

The truth table is encoded as a SIMM8 value overloading the OMOD, ABS and NEG fields of the instruction.
Normal output modifier, absolute value and negation controls are disabled for this instruction. The truth table
is encoded as

```
  { OMOD[1:0], ABS[2:0], NEG[2:0] }
```

Given the i'th bit of inputs S0, S1 and S2, the i'th bit of the result D0 can be determined by looking up a bit in

TTBL:

```
  D0[i] = TTBL[{S0[i], S1[i], S2[i]}]
```

Equivalencies with other bitwise ops:

Opcode S0 S1 S2 TTBL
V_AND A B 0 0x40
V_OR A B 0 0x54
V_XOR A B 0 0x14
V_XNOR A B 0 0x41
V_NOT A 0 0 0x01
V_AND_OR A B C 0xea
V_OR3 A B C 0xfe
V_XOR3 A B C 0x96

#### V_CVT_SCALEF32_PK_FP8_F32  (opcode 565)

Scale two single-precision float inputs using the exponent provided by the third single-precision float input,
then convert the values to a packed FP8 float value with round toward nearest even semantics. Store the result
into 16 bits of a vector register using OPSEL.

```
  scale = 32'U(exponent(S2.f32));
  tmp0 = f32_to_fp8_scale(S0.f32, scale.u8);
  tmp1 = f32_to_fp8_scale(S1.f32, scale.u8);
  dstword = OPSEL[3].i32 * 16;
  VGPR[laneId][VDST.u32][dstword + 15 : dstword].b16 = { tmp1, tmp0 };
  // Other destination bits are preserved
```

#### V_CVT_SCALEF32_PK_BF8_F32  (opcode 566)

Scale two single-precision float inputs using the exponent provided by the third single-precision float input,
then convert the values to a packed BF8 float value with round toward nearest even semantics. Store the result
into 16 bits of a vector register using OPSEL.

```
  scale = 32'U(exponent(S2.f32));
  tmp0 = f32_to_bf8_scale(S0.f32, scale.u8);
  tmp1 = f32_to_bf8_scale(S1.f32, scale.u8);
  dstword = OPSEL[3].i32 * 16;
  VGPR[laneId][VDST.u32][dstword + 15 : dstword].b16 = { tmp1, tmp0 };
  // Other destination bits are preserved
```

#### V_CVT_SCALEF32_SR_FP8_F32  (opcode 567)

Scale a single-precision float input using the exponent provided by the third single-precision float input, then
convert the values to an FP8 float value with stochastic rounding using seed data from the second input. Store
the result into 8 bits of a vector register using OPSEL[3:2] to determine which byte of the destination to
overwrite.

```
  scale = 32'U(exponent(S2.f32));
  tmp = f32_to_fp8_sr_scale(S0.f32, S1.u32, scale.u8);
  dstbyte = OPSEL[3 : 2].i32 * 8;
  VGPR[laneId][VDST.u32][dstbyte + 7 : dstbyte].fp8 = tmp;
  // Other destination bits are preserved
```

#### V_CVT_SCALEF32_SR_BF8_F32  (opcode 568)

Scale a single-precision float input using the exponent provided by the third single-precision float input, then
convert the values to a BF8 float value with stochastic rounding using seed data from the second input. Store
the result into 8 bits of a vector register using OPSEL[3:2] to determine which byte of the destination to
overwrite.

```
  scale = 32'U(exponent(S2.f32));
  tmp = f32_to_bf8_sr_scale(S0.f32, S1.u32, scale.u8);
  dstbyte = OPSEL[3 : 2].i32 * 8;
  VGPR[laneId][VDST.u32][dstbyte + 7 : dstbyte].bf8 = tmp;
  // Other destination bits are preserved
```

#### V_CVT_SCALEF32_PK_F32_FP8  (opcode 569)

Convert from a packed 2-component FP8 float input to a packed single-precision float value, then scale the
packed values using the exponent provided by the second single-precision float input. Store the result into a
vector register.

```
  scale = 32'U(exponent(S1.f32));
  srcword = OPSEL[0].i32 * 16;
  src = VGPR[laneId][SRC0.u32][srcword + 15 : srcword].b16;
  tmp0 = fp8_to_f32_scale(src[7 : 0].fp8, scale.u8);
  tmp1 = fp8_to_f32_scale(src[15 : 8].fp8, scale.u8);
  D0[31 : 0].f32 = tmp0;
  D0[63 : 32].f32 = tmp1
```

#### V_CVT_SCALEF32_PK_F32_BF8  (opcode 570)

Convert from a packed 2-component BF8 float input to a packed single-precision float value, then scale the
packed values using the exponent provided by the second single-precision float input. Store the result into a
vector register.

```
  scale = 32'U(exponent(S1.f32));
  srcword = OPSEL[0].i32 * 16;
  src = VGPR[laneId][SRC0.u32][srcword + 15 : srcword].b16;
  tmp0 = bf8_to_f32_scale(src[7 : 0].bf8, scale.u8);
  tmp1 = bf8_to_f32_scale(src[15 : 8].bf8, scale.u8);
  D0[31 : 0].f32 = tmp0;
  D0[63 : 32].f32 = tmp1
```

#### V_CVT_SCALEF32_F32_FP8  (opcode 571)

Convert from an FP8 float input to a single-precision float value, then scale the value using the exponent
provided by the second single-precision float input. Store the result into a vector register. The value to convert
is loaded from 8 bits of the input using OPSEL[1:0] to determine which byte to read.

```
  scale = 32'U(exponent(S1.f32));
  srcbyte = OPSEL[1 : 0].i32 * 8;
  src = VGPR[laneId][SRC0.u32][srcbyte + 7 : srcbyte].fp8;
  tmp = fp8_to_f32_scale(src, scale.u8);
  D0 = tmp.b32
```

#### V_CVT_SCALEF32_F32_BF8  (opcode 572)

Convert from a BF8 float input to a single-precision float value, then scale the value using the exponent
provided by the second single-precision float input. Store the result into a vector register. The value to convert
is loaded from 8 bits of the input using OPSEL[1:0] to determine which byte to read.

```
  scale = 32'U(exponent(S1.f32));
  srcbyte = OPSEL[1 : 0].i32 * 8;
  src = VGPR[laneId][SRC0.u32][srcbyte + 7 : srcbyte].bf8;
  tmp = bf8_to_f32_scale(src, scale.u8);
  D0 = tmp.b32
```

#### V_CVT_SCALEF32_PK_FP4_F32  (opcode 573)

Scale two single-precision float inputs using the exponent provided by the third single-precision float input,
then convert the values to a packed FP4 float value with round toward nearest even semantics. Store the result
into 8 bits of a vector register using OPSEL[3:2] to determine which byte of the destination to overwrite.

```
  scale = 32'U(exponent(S2.f32));
  tmp0 = f32_to_fp4_scale(S0.f32, scale.u8);
  tmp1 = f32_to_fp4_scale(S1.f32, scale.u8);
  dstbyte = OPSEL[3 : 2].i32 * 8;
  VGPR[laneId][VDST.u32][dstbyte + 7 : dstbyte].b8 = { tmp1, tmp0 };
```

```
  // Other destination bits are preserved
```

#### V_CVT_SCALEF32_SR_PK_FP4_F32  (opcode 574)

Scale a packed 2-component single-precision float input using the exponent provided by the third single-
precision float input, then convert the values to a packed FP4 float value with stochastic rounding using seed
data from the second input. Store the result into 8 bits of a vector register using OPSEL[3:2] to determine which
byte of the destination to overwrite.

```
  scale = 32'U(exponent(S2.f32));
  randomVal = S1.u32;
  tmp0 = f32_to_fp4_sr_scale(S0[31 : 0].f32, randomVal, scale.u8);
  randomVal = 32'U(v_prng_b32(randomVal.b32));
  tmp1 = f32_to_fp4_sr_scale(S0[63 : 32].f32, randomVal, scale.u8);
  dstbyte = OPSEL[3 : 2].i32 * 8;
  VGPR[laneId][VDST.u32][dstbyte + 7 : dstbyte].b8 = { tmp1, tmp0 };
  // Other destination bits are preserved
```

#### V_CVT_SCALEF32_PK_F32_FP4  (opcode 575)

Convert from a packed 2-component FP4 float input to a packed single-precision float value, then scale the
packed values using the exponent provided by the second single-precision float input. Store the result into a
vector register. The value to convert is loaded from 8 bits of the input using OPSEL[1:0] to determine which
byte to read.

```
  scale = 32'U(exponent(S1.f32));
  srcbyte = OPSEL[1 : 0].i32 * 8;
  src = VGPR[laneId][SRC0.u32][srcbyte + 7 : srcbyte].b8;
  tmp0 = fp4_to_f32_scale(src[3 : 0].fp4, scale.u8);
  tmp1 = fp4_to_f32_scale(src[7 : 4].fp4, scale.u8);
  D0[31 : 0].f32 = tmp0;
  D0[63 : 32].f32 = tmp1
```

#### V_CVT_SCALEF32_PK_FP8_F16  (opcode 576)

Scale a packed 2-component half-precision float input using the exponent provided by the second single-
precision float input, then convert the values to a packed FP8 float value with round toward nearest even
semantics. Store the result into 16 bits of a vector register using OPSEL.

```
  scale = 32'U(exponent(S1.f32));
  tmp0 = f16_to_fp8_scale(S0[15 : 0].f16, scale.u8);
  tmp1 = f16_to_fp8_scale(S0[31 : 16].f16, scale.u8);
  dstword = OPSEL[3].i32 * 16;
  VGPR[laneId][VDST.u32][dstword + 15 : dstword].b16 = { tmp1, tmp0 };
```

```
  // Other destination bits are preserved
```

#### V_CVT_SCALEF32_PK_BF8_F16  (opcode 577)

Scale a packed 2-component half-precision float input using the exponent provided by the second single-
precision float input, then convert the values to a packed BF8 float value with round toward nearest even
semantics. Store the result into 16 bits of a vector register using OPSEL.

```
  scale = 32'U(exponent(S1.f32));
  tmp0 = f16_to_bf8_scale(S0[15 : 0].f16, scale.u8);
  tmp1 = f16_to_bf8_scale(S0[31 : 16].f16, scale.u8);
  dstword = OPSEL[3].i32 * 16;
  VGPR[laneId][VDST.u32][dstword + 15 : dstword].b16 = { tmp1, tmp0 };
  // Other destination bits are preserved
```

#### V_CVT_SCALEF32_SR_FP8_F16  (opcode 578)

Scale a half-precision float input using the exponent provided by the third single-precision float input, then
convert the values to an FP8 float value with stochastic rounding using seed data from the second input. Store
the result into 8 bits of a vector register using OPSEL[3:2] to determine which byte of the destination to
overwrite.

```
  scale = 32'U(exponent(S2.f32));
  tmp = f16_to_fp8_sr_scale(S0.f16, S1.u32, scale.u8);
  dstbyte = OPSEL[3 : 2].i32 * 8;
  VGPR[laneId][VDST.u32][dstbyte + 7 : dstbyte].fp8 = tmp;
  // Other destination bits are preserved
```

#### V_CVT_SCALEF32_SR_BF8_F16  (opcode 579)

Scale a half-precision float input using the exponent provided by the third single-precision float input, then
convert the values to a BF8 float value with stochastic rounding using seed data from the second input. Store
the result into 8 bits of a vector register using OPSEL[3:2] to determine which byte of the destination to
overwrite.

```
  scale = 32'U(exponent(S2.f32));
  tmp = f16_to_bf8_sr_scale(S0.f16, S1.u32, scale.u8);
  dstbyte = OPSEL[3 : 2].i32 * 8;
  VGPR[laneId][VDST.u32][dstbyte + 7 : dstbyte].bf8 = tmp;
  // Other destination bits are preserved
```

#### V_CVT_SCALEF32_PK_FP8_BF16  (opcode 580)

Scale a packed 2-component BF16 float input using the exponent provided by the second single-precision float
input, then convert the values to a packed FP8 float value with round toward nearest even semantics. Store the
result into 16 bits of a vector register using OPSEL.

```
  scale = 32'U(exponent(S1.f32));
  tmp0 = bf16_to_fp8_scale(S0[15 : 0].bf16, scale.u8);
  tmp1 = bf16_to_fp8_scale(S0[31 : 16].bf16, scale.u8);
  dstword = OPSEL[3].i32 * 16;
  VGPR[laneId][VDST.u32][dstword + 15 : dstword].b16 = { tmp1, tmp0 };
  // Other destination bits are preserved
```

#### V_CVT_SCALEF32_PK_BF8_BF16  (opcode 581)

Scale a packed 2-component BF16 float input using the exponent provided by the second single-precision float
input, then convert the values to a packed BF8 float value with round toward nearest even semantics. Store the
result into 16 bits of a vector register using OPSEL.

```
  scale = 32'U(exponent(S1.f32));
  tmp0 = bf16_to_bf8_scale(S0[15 : 0].bf16, scale.u8);
  tmp1 = bf16_to_bf8_scale(S0[31 : 16].bf16, scale.u8);
  dstword = OPSEL[3].i32 * 16;
  VGPR[laneId][VDST.u32][dstword + 15 : dstword].b16 = { tmp1, tmp0 };
  // Other destination bits are preserved
```

#### V_CVT_SCALEF32_SR_FP8_BF16  (opcode 582)

Scale a BF16 float input using the exponent provided by the third single-precision float input, then convert the
values to an FP8 float value with stochastic rounding using seed data from the second input. Store the result
into 8 bits of a vector register using OPSEL[3:2] to determine which byte of the destination to overwrite.

```
  scale = 32'U(exponent(S2.f32));
  tmp = bf16_to_fp8_sr_scale(S0.bf16, S1.u32, scale.u8);
  dstbyte = OPSEL[3 : 2].i32 * 8;
  VGPR[laneId][VDST.u32][dstbyte + 7 : dstbyte].fp8 = tmp;
  // Other destination bits are preserved
```

#### V_CVT_SCALEF32_SR_BF8_BF16  (opcode 583)

Scale a BF16 float input using the exponent provided by the third single-precision float input, then convert the
values to a BF8 float value with stochastic rounding using seed data from the second input. Store the result into
8 bits of a vector register using OPSEL[3:2] to determine which byte of the destination to overwrite.

```
  scale = 32'U(exponent(S2.f32));
  tmp = bf16_to_bf8_sr_scale(S0.bf16, S1.u32, scale.u8);
  dstbyte = OPSEL[3 : 2].i32 * 8;
  VGPR[laneId][VDST.u32][dstbyte + 7 : dstbyte].bf8 = tmp;
  // Other destination bits are preserved
```

#### V_CVT_SCALEF32_PK_F16_FP8  (opcode 584)

Convert from a packed 2-component FP8 float input to a packed half-precision float value, then scale the
packed values using the exponent provided by the second single-precision float input. Store the result into a
vector register.

```
  scale = 32'U(exponent(S1.f32));
  srcword = OPSEL[0].i32 * 16;
  src = VGPR[laneId][SRC0.u32][srcword + 15 : srcword].b16;
  tmp0 = fp8_to_f16_scale(src[7 : 0].fp8, scale.u8);
  tmp1 = fp8_to_f16_scale(src[15 : 8].fp8, scale.u8);
  D0[15 : 0].f16 = tmp0;
  D0[31 : 16].f16 = tmp1
```

#### V_CVT_SCALEF32_PK_F16_BF8  (opcode 585)

Convert from a packed 2-component BF8 float input to a packed half-precision float value, then scale the
packed values using the exponent provided by the second single-precision float input. Store the result into a
vector register.

```
  scale = 32'U(exponent(S1.f32));
  srcword = OPSEL[0].i32 * 16;
  src = VGPR[laneId][SRC0.u32][srcword + 15 : srcword].b16;
  tmp0 = bf8_to_f16_scale(src[7 : 0].bf8, scale.u8);
  tmp1 = bf8_to_f16_scale(src[15 : 8].bf8, scale.u8);
  D0[15 : 0].f16 = tmp0;
  D0[31 : 16].f16 = tmp1
```

#### V_CVT_SCALEF32_F16_FP8  (opcode 586)

Convert from an FP8 float input to a half-precision float value, then scale the value using the exponent
provided by the second single-precision float input. Store the result into a vector register. The value to convert
is loaded from 8 bits of the input using OPSEL[1:0] to determine which byte to read.

```
  scale = 32'U(exponent(S1.f32));
  srcbyte = OPSEL[1 : 0].i32 * 8;
  src = VGPR[laneId][SRC0.u32][srcbyte + 7 : srcbyte].fp8;
```

```
  tmp = fp8_to_f16_scale(src, scale.u8);
  // OPSEL[3] controls destination hi/lo
  D0 = tmp.b32
```

#### V_CVT_SCALEF32_F16_BF8  (opcode 587)

Convert from a BF8 float input to a half-precision float value, then scale the value using the exponent provided
by the second single-precision float input. Store the result into a vector register. The value to convert is loaded
from 8 bits of the input using OPSEL[1:0] to determine which byte to read.

```
  scale = 32'U(exponent(S1.f32));
  srcbyte = OPSEL[1 : 0].i32 * 8;
  src = VGPR[laneId][SRC0.u32][srcbyte + 7 : srcbyte].bf8;
  tmp = bf8_to_f16_scale(src, scale.u8);
  // OPSEL[3] controls destination hi/lo
  D0 = tmp.b32
```

#### V_CVT_SCALEF32_PK_FP4_F16  (opcode 588)

Scale a packed 2-component half-precision float input using the exponent provided by the second single-
precision float input, then convert the values to a packed FP4 float value with round toward nearest even
semantics. Store the result into 8 bits of a vector register using OPSEL[3:2] to determine which byte of the
destination to overwrite.

```
  scale = 32'U(exponent(S1.f32));
  tmp0 = f16_to_fp4_scale(S0[15 : 0].f16, scale.u8);
  tmp1 = f16_to_fp4_scale(S0[31 : 16].f16, scale.u8);
  dstbyte = OPSEL[3 : 2].i32 * 8;
  VGPR[laneId][VDST.u32][dstbyte + 7 : dstbyte].b8 = { tmp1, tmp0 };
  // Other destination bits are preserved
```

#### V_CVT_SCALEF32_PK_FP4_BF16  (opcode 589)

Scale a packed 2-component BF16 float input using the exponent provided by the second single-precision float
input, then convert the values to a packed FP4 float value with round toward nearest even semantics. Store the
result into 8 bits of a vector register using OPSEL[3:2] to determine which byte of the destination to overwrite.

```
  scale = 32'U(exponent(S1.f32));
  tmp0 = bf16_to_fp4_scale(S0[15 : 0].bf16, scale.u8);
  tmp1 = bf16_to_fp4_scale(S0[31 : 16].bf16, scale.u8);
  dstbyte = OPSEL[3 : 2].i32 * 8;
  VGPR[laneId][VDST.u32][dstbyte + 7 : dstbyte].b8 = { tmp1, tmp0 };
  // Other destination bits are preserved
```

#### V_CVT_SCALEF32_SR_PK_FP4_F16  (opcode 590)

Scale a packed 2-component half-precision float input using the exponent provided by the third single-
precision float input, then convert the values to a packed FP4 float value with stochastic rounding using seed
data from the second input. Store the result into 8 bits of a vector register using OPSEL[3:2] to determine which
byte of the destination to overwrite.

```
  scale = 32'U(exponent(S2.f32));
  randomVal = S1.u32;
  tmp0 = f16_to_fp4_sr_scale(S0[15 : 0].f16, randomVal, scale.u8);
  randomVal = 32'U(v_prng_b32(randomVal.b32));
  tmp1 = f16_to_fp4_sr_scale(S0[31 : 16].f16, randomVal, scale.u8);
  dstbyte = OPSEL[3 : 2].i32 * 8;
  VGPR[laneId][VDST.u32][dstbyte + 7 : dstbyte].b8 = { tmp1, tmp0 };
  // Other destination bits are preserved
```

#### V_CVT_SCALEF32_SR_PK_FP4_BF16  (opcode 591)

Scale a packed 2-component BF16 float input using the exponent provided by the third single-precision float
input, then convert the values to a packed FP4 float value with stochastic rounding using seed data from the
second input. Store the result into 8 bits of a vector register using OPSEL[3:2] to determine which byte of the
destination to overwrite.

```
  scale = 32'U(exponent(S2.f32));
  randomVal = S1.u32;
  tmp0 = bf16_to_fp4_sr_scale(S0[15 : 0].bf16, randomVal, scale.u8);
  randomVal = 32'U(v_prng_b32(randomVal.b32));
  tmp1 = bf16_to_fp4_sr_scale(S0[31 : 16].bf16, randomVal, scale.u8);
  dstbyte = OPSEL[3 : 2].i32 * 8;
  VGPR[laneId][VDST.u32][dstbyte + 7 : dstbyte].b8 = { tmp1, tmp0 };
  // Other destination bits are preserved
```

#### V_CVT_SCALEF32_PK_F16_FP4  (opcode 592)

Convert from a packed 2-component FP4 float input to a packed half-precision float value, then scale the
packed values using the exponent provided by the second single-precision float input. Store the result into a
vector register. The value to convert is loaded from 8 bits of the input using OPSEL[1:0] to determine which
byte to read.

```
  scale = 32'U(exponent(S1.f32));
  srcbyte = OPSEL[1 : 0].i32 * 8;
  src = VGPR[laneId][SRC0.u32][srcbyte + 7 : srcbyte].b8;
  tmp0 = fp4_to_f16_scale(src[3 : 0].fp4, scale.u8);
  tmp1 = fp4_to_f16_scale(src[7 : 4].fp4, scale.u8);
  D0[15 : 0].f16 = tmp0;
```

```
  D0[31 : 16].f16 = tmp1
```

#### V_CVT_SCALEF32_PK_BF16_FP4  (opcode 593)

Convert from a packed 2-component FP4 float input to a packed BF16 float value, then scale the packed values
using the exponent provided by the second single-precision float input. Store the result into a vector register.
The value to convert is loaded from 8 bits of the input using OPSEL[1:0] to determine which byte to read.

```
  scale = 32'U(exponent(S1.f32));
  srcbyte = OPSEL[1 : 0].i32 * 8;
  src = VGPR[laneId][SRC0.u32][srcbyte + 7 : srcbyte].b8;
  tmp0 = fp4_to_bf16_scale(src[3 : 0].fp4, scale.u8);
  tmp1 = fp4_to_bf16_scale(src[7 : 4].fp4, scale.u8);
  D0[15 : 0].bf16 = tmp0;
  D0[31 : 16].bf16 = tmp1
```

#### V_CVT_SCALEF32_2XPK16_FP6_F32  (opcode 594)

Scale packed 16-component single-precision float vectors from two source inputs using the exponent provided
by the third single-precision float input, then convert the values to a packed 32-component FP6 float value.
Store the result into a vector register.

```
  scale = 32'U(exponent(S2.f32));
  declare tmp : 192'B;
  for pass in 0 : 15 do
      dOffset = pass * 12;
      sOffset = pass * 32;
      // Note that S0 and S1 inputs are interleaved in the packed result.
      tmp[dOffset + 5 : dOffset].fp6 = f32_to_fp6_scale(S0[sOffset + 31 : sOffset].f32, scale.u8);
      tmp[dOffset + 11 : dOffset + 6].fp6 = f32_to_fp6_scale(S1[sOffset + 31 : sOffset].f32, scale.u8)
  endfor;
  D0[191 : 0] = tmp.b192
```

#### V_CVT_SCALEF32_2XPK16_BF6_F32  (opcode 595)

Scale packed 16-component single-precision float vectors from two source inputs using the exponent provided
by the third single-precision float input, then convert the values to a packed 32-component BF6 float value.
Store the result into a vector register.

```
  scale = 32'U(exponent(S2.f32));
  declare tmp : 192'B;
  for pass in 0 : 15 do
      dOffset = pass * 12;
      sOffset = pass * 32;
```

```
      // Note that S0 and S1 inputs are interleaved in the packed result.
      tmp[dOffset + 5 : dOffset].bf6 = f32_to_bf6_scale(S0[sOffset + 31 : sOffset].f32, scale.u8);
      tmp[dOffset + 11 : dOffset + 6].bf6 = f32_to_bf6_scale(S1[sOffset + 31 : sOffset].f32, scale.u8)
  endfor;
  D0[191 : 0] = tmp.b192
```

#### V_CVT_SCALEF32_SR_PK32_FP6_F32  (opcode 596)

Scale a packed 32-component single-precision float input using the exponent provided by the third single-
precision float input, then convert the values to a packed 32-component FP6 float value with stochastic
rounding using seed data from the second input. Store the result into a vector register.

```
  scale = 32'U(exponent(S2.f32));
  randomVal = S1.u32;
  declare tmp : 192'B;
  for pass in 0 : 31 do
      dOffset = pass * 6;
      sOffset = pass * 32;
      tmp[dOffset + 5 : dOffset].fp6 = f32_to_fp6_sr_scale(S0[sOffset + 31 : sOffset].f32, randomVal,
  scale.u8);
      randomVal = 32'U(v_prng_b32(randomVal.b32))
  endfor;
  D0[191 : 0] = tmp.b192
```

#### V_CVT_SCALEF32_SR_PK32_BF6_F32  (opcode 597)

Scale a packed 32-component single-precision float input using the exponent provided by the third single-
precision float input, then convert the values to a packed 32-component BF6 float value with stochastic
rounding using seed data from the second input. Store the result into a vector register.

```
  scale = 32'U(exponent(S2.f32));
  randomVal = S1.u32;
  declare tmp : 192'B;
  for pass in 0 : 31 do
      dOffset = pass * 6;
      sOffset = pass * 32;
      tmp[dOffset + 5 : dOffset].bf6 = f32_to_bf6_sr_scale(S0[sOffset + 31 : sOffset].f32, randomVal,
  scale.u8);
      randomVal = 32'U(v_prng_b32(randomVal.b32))
  endfor;
  D0[191 : 0] = tmp.b192
```

#### V_CVT_SCALEF32_PK32_F32_FP6  (opcode 598)

Convert from a packed 32-component FP6 float input to a packed single-precision float value, then scale the

packed values using the exponent provided by the second single-precision float input. Store the result into a
vector register.

```
  scale = 32'U(exponent(S1.f32));
  declare tmp : 1024'B;
  for pass in 0 : 31 do
      dOffset = pass * 32;
      sOffset = pass * 6;
      tmp[dOffset + 31 : dOffset].f32 = fp6_to_f32_scale(S0[sOffset + 5 : sOffset].fp6, scale.u8)
  endfor;
  D0[1023 : 0] = tmp.b1024
```

#### V_CVT_SCALEF32_PK32_F32_BF6  (opcode 599)

Convert from a packed 32-component BF6 float input to a packed single-precision float value, then scale the
packed values using the exponent provided by the second single-precision float input. Store the result into a
vector register.

```
  scale = 32'U(exponent(S1.f32));
  declare tmp : 1024'B;
  for pass in 0 : 31 do
      dOffset = pass * 32;
      sOffset = pass * 6;
      tmp[dOffset + 31 : dOffset].f32 = bf6_to_f32_scale(S0[sOffset + 5 : sOffset].bf6, scale.u8)
  endfor;
  D0[1023 : 0] = tmp.b1024
```

#### V_CVT_SCALEF32_PK32_FP6_F16  (opcode 600)

Scale a packed 32-component half-precision float input using the exponent provided by the second single-
precision float input, then convert the values to a packed 32-component FP6 float value. Store the result into a
vector register.

```
  scale = 32'U(exponent(S1.f32));
  declare tmp : 192'B;
  for pass in 0 : 31 do
      dOffset = pass * 6;
      sOffset = pass * 16;
      tmp[dOffset + 5 : dOffset].fp6 = f16_to_fp6_scale(S0[sOffset + 15 : sOffset].f16, scale.u8)
  endfor;
  D0[191 : 0] = tmp.b192
```

#### V_CVT_SCALEF32_PK32_FP6_BF16  (opcode 601)

Scale a packed 32-component BF16 float input using the exponent provided by the second single-precision float
input, then convert the values to a packed 32-component FP6 float value. Store the result into a vector register.

```
  scale = 32'U(exponent(S1.f32));
  declare tmp : 192'B;
  for pass in 0 : 31 do
      dOffset = pass * 6;
      sOffset = pass * 16;
      tmp[dOffset + 5 : dOffset].fp6 = bf16_to_fp6_scale(S0[sOffset + 15 : sOffset].bf16, scale.u8)
  endfor;
  D0[191 : 0] = tmp.b192
```

#### V_CVT_SCALEF32_PK32_BF6_F16  (opcode 602)

Scale a packed 32-component half-precision float input using the exponent provided by the second single-
precision float input, then convert the values to a packed 32-component BF6 float value. Store the result into a
vector register.

```
  scale = 32'U(exponent(S1.f32));
  declare tmp : 192'B;
  for pass in 0 : 31 do
      dOffset = pass * 6;
      sOffset = pass * 16;
      tmp[dOffset + 5 : dOffset].bf6 = f16_to_bf6_scale(S0[sOffset + 15 : sOffset].f16, scale.u8)
  endfor;
  D0[191 : 0] = tmp.b192
```

#### V_CVT_SCALEF32_PK32_BF6_BF16  (opcode 603)

Scale a packed 32-component BF16 float input using the exponent provided by the second single-precision float
input, then convert the values to a packed 32-component BF6 float value. Store the result into a vector register.

```
  scale = 32'U(exponent(S1.f32));
  declare tmp : 192'B;
  for pass in 0 : 31 do
      dOffset = pass * 6;
      sOffset = pass * 16;
      tmp[dOffset + 5 : dOffset].bf6 = bf16_to_bf6_scale(S0[sOffset + 15 : sOffset].bf16, scale.u8)
  endfor;
  D0[191 : 0] = tmp.b192
```

#### V_CVT_SCALEF32_SR_PK32_FP6_F16  (opcode 604)

Scale a packed 32-component half-precision float input using the exponent provided by the third single-

precision float input, then convert the values to a packed 32-component FP6 float value with stochastic
rounding using seed data from the second input. Store the result into a vector register.

```
  scale = 32'U(exponent(S2.f32));
  randomVal = S1.u32;
  declare tmp : 192'B;
  for pass in 0 : 31 do
      dOffset = pass * 6;
      sOffset = pass * 16;
      tmp[dOffset + 5 : dOffset].fp6 = f16_to_fp6_sr_scale(S0[sOffset + 15 : sOffset].f16, randomVal,
  scale.u8);
      randomVal = 32'U(v_prng_b32(randomVal.b32))
  endfor;
  D0[191 : 0] = tmp.b192
```

#### V_CVT_SCALEF32_SR_PK32_FP6_BF16  (opcode 605)

Scale a packed 32-component BF16 float input using the exponent provided by the third single-precision float
input, then convert the values to a packed 32-component FP6 float value with stochastic rounding using seed
data from the second input. Store the result into a vector register.

```
  scale = 32'U(exponent(S2.f32));
  randomVal = S1.u32;
  declare tmp : 192'B;
  for pass in 0 : 31 do
      dOffset = pass * 6;
      sOffset = pass * 16;
      tmp[dOffset + 5 : dOffset].fp6 = bf16_to_fp6_sr_scale(S0[sOffset + 15 : sOffset].bf16, randomVal,
  scale.u8);
      randomVal = 32'U(v_prng_b32(randomVal.b32))
  endfor;
  D0[191 : 0] = tmp.b192
```

#### V_CVT_SCALEF32_SR_PK32_BF6_F16  (opcode 606)

Scale a packed 32-component half-precision float input using the exponent provided by the third single-
precision float input, then convert the values to a packed 32-component BF6 float value with stochastic
rounding using seed data from the second input. Store the result into a vector register.

```
  scale = 32'U(exponent(S2.f32));
  randomVal = S1.u32;
  declare tmp : 192'B;
  for pass in 0 : 31 do
      dOffset = pass * 6;
      sOffset = pass * 16;
      tmp[dOffset + 5 : dOffset].bf6 = f16_to_bf6_sr_scale(S0[sOffset + 15 : sOffset].f16, randomVal,
  scale.u8);
      randomVal = 32'U(v_prng_b32(randomVal.b32))
```

```
  endfor;
  D0[191 : 0] = tmp.b192
```

#### V_CVT_SCALEF32_SR_PK32_BF6_BF16  (opcode 607)

Scale a packed 32-component BF16 float input using the exponent provided by the third single-precision float
input, then convert the values to a packed 32-component BF6 float value with stochastic rounding using seed
data from the second input. Store the result into a vector register.

```
  scale = 32'U(exponent(S2.f32));
  randomVal = S1.u32;
  declare tmp : 192'B;
  for pass in 0 : 31 do
      dOffset = pass * 6;
      sOffset = pass * 16;
      tmp[dOffset + 5 : dOffset].bf6 = bf16_to_bf6_sr_scale(S0[sOffset + 15 : sOffset].bf16, randomVal,
  scale.u8);
      randomVal = 32'U(v_prng_b32(randomVal.b32))
  endfor;
  D0[191 : 0] = tmp.b192
```

#### V_CVT_SCALEF32_PK32_F16_FP6  (opcode 608)

Convert from a packed 32-component FP6 float input to a packed half-precision float value, then scale the
packed values using the exponent provided by the second single-precision float input. Store the result into a
vector register.

```
  scale = 32'U(exponent(S1.f32));
  declare tmp : 512'B;
  for pass in 0 : 31 do
      dOffset = pass * 16;
      sOffset = pass * 6;
      tmp[dOffset + 15 : dOffset].f16 = fp6_to_f16_scale(S0[sOffset + 5 : sOffset].fp6, scale.u8)
  endfor;
  D0[511 : 0] = tmp.b512
```

#### V_CVT_SCALEF32_PK32_BF16_FP6  (opcode 609)

Convert from a packed 32-component FP6 float input to a packed BF16 float value, then scale the packed values
using the exponent provided by the second single-precision float input. Store the result into a vector register.

```
  scale = 32'U(exponent(S1.f32));
  declare tmp : 512'B;
  for pass in 0 : 31 do
```

```
      dOffset = pass * 16;
      sOffset = pass * 6;
      tmp[dOffset + 15 : dOffset].bf16 = fp6_to_bf16_scale(S0[sOffset + 5 : sOffset].fp6, scale.u8)
  endfor;
  D0[511 : 0] = tmp.b512
```

#### V_CVT_SCALEF32_PK32_F16_BF6  (opcode 610)

Convert from a packed 32-component BF6 float input to a packed half-precision float value, then scale the
packed values using the exponent provided by the second single-precision float input. Store the result into a
vector register.

```
  scale = 32'U(exponent(S1.f32));
  declare tmp : 512'B;
  for pass in 0 : 31 do
      dOffset = pass * 16;
      sOffset = pass * 6;
      tmp[dOffset + 15 : dOffset].f16 = bf6_to_f16_scale(S0[sOffset + 5 : sOffset].bf6, scale.u8)
  endfor;
  D0[511 : 0] = tmp.b512
```

#### V_CVT_SCALEF32_PK32_BF16_BF6  (opcode 611)

Convert from a packed 32-component BF6 float input to a packed BF16 float value, then scale the packed values
using the exponent provided by the second single-precision float input. Store the result into a vector register.

```
  scale = 32'U(exponent(S1.f32));
  declare tmp : 512'B;
  for pass in 0 : 31 do
      dOffset = pass * 16;
      sOffset = pass * 6;
      tmp[dOffset + 15 : dOffset].bf16 = bf6_to_bf16_scale(S0[sOffset + 5 : sOffset].bf6, scale.u8)
  endfor;
  D0[511 : 0] = tmp.b512
```

#### V_ASHR_PK_I8_I32  (opcode 613)

Given two signed 32-bit integers and a shift count, calculate the arithmetic shift right (preserving sign bit) of the
two integers, saturate the two results in the signed 8-bit interval [-128, 127], pack the bytes and store the result
into a vector register.

```
  SAT8 = lambda(n) (
      if n <= -128 then
            return 8'0x80
```

```
      elsif n >= 127 then
           return 8'0x7f
      else
           return n[7 : 0].b8
      endif);
  declare tmp : 16'B;
  tmp[7 : 0] = SAT8(S0.i32 >> S2[4 : 0].u32);
  tmp[15 : 8] = SAT8(S1.i32 >> S2[4 : 0].u32);
  D0[15 : 0] = tmp
```

#### V_ASHR_PK_U8_I32  (opcode 614)

Given two signed 32-bit integers and a shift count, calculate the arithmetic shift right (preserving sign bit) of the
two integers, saturate the two results in the unsigned 8-bit interval [0, 255], pack the bytes and store the result
into a vector register.

```
  SAT8 = lambda(n) (
      if n <= 0 then
           return 8'0x0
      elsif n >= 255 then
           return 8'0xff
      else
           return n[7 : 0].b8
      endif);
  declare tmp : 16'B;
  tmp[7 : 0] = SAT8(S0.i32 >> S2[4 : 0].u32);
  tmp[15 : 8] = SAT8(S1.i32 >> S2[4 : 0].u32);
  D0[15 : 0] = tmp
```

#### V_CVT_PK_F16_F32  (opcode 615)

Convert from two single-precision float inputs to a packed half-precision value and store the result into a vector
register.

```
  prev_mode = ROUND_MODE;
  ROUND_MODE = ROUND_NEAREST_EVEN;
  tmp[15 : 0].f16 = f32_to_f16(S0.f32);
  tmp[31 : 16].f16 = f32_to_f16(S1.f32);
  D0 = tmp.b32;
  ROUND_MODE = prev_mode
```

#### V_CVT_PK_BF16_F32  (opcode 616)

Convert from two single-precision float inputs to a packed BF16 value and store the result into a vector register.

```
  prev_mode = ROUND_MODE;
  ROUND_MODE = ROUND_NEAREST_EVEN;
  tmp[15 : 0].bf16 = f32_to_bf16(S0.f32);
  tmp[31 : 16].bf16 = f32_to_bf16(S1.f32);
  D0 = tmp.b32;
  ROUND_MODE = prev_mode
```

#### V_CVT_SCALEF32_PK_BF16_FP8  (opcode 617)

Convert from a packed 2-component FP8 float input to a packed BF16 float value, then scale the packed values
using the exponent provided by the second single-precision float input. Store the result into a vector register.

```
  scale = 32'U(exponent(S1.f32));
  srcword = OPSEL[0].i32 * 16;
  src = VGPR[laneId][SRC0.u32][srcword + 15 : srcword].b16;
  tmp0 = fp8_to_bf16_scale(src[7 : 0].fp8, scale);
  tmp1 = fp8_to_bf16_scale(src[15 : 8].fp8, scale);
  D0[15 : 0].bf16 = tmp0.bf16;
  D0[31 : 16].bf16 = tmp1.bf16
```

#### V_CVT_SCALEF32_PK_BF16_BF8  (opcode 618)

Convert from a packed 2-component BF8 float input to a packed BF16 float value, then scale the packed values
using the exponent provided by the second single-precision float input. Store the result into a vector register.

```
  scale = 32'U(exponent(S1.f32));
  srcword = OPSEL[0].i32 * 16;
  src = VGPR[laneId][SRC0.u32][srcword + 15 : srcword].b16;
  tmp0 = bf8_to_bf16_scale(src[7 : 0].bf8, scale);
  tmp1 = bf8_to_bf16_scale(src[15 : 8].bf8, scale);
  D0[15 : 0].bf16 = tmp0.bf16;
  D0[31 : 16].bf16 = tmp1.bf16
```

#### V_ADD_F64  (opcode 640)

Add two floating point inputs and store the result into a vector register.

```
  D0.f64 = S0.f64 + S1.f64
```

Notes

0.5ULP precision, denormals are supported.

#### V_MUL_F64  (opcode 641)

Multiply two floating point inputs and store the result into a vector register.

```
  D0.f64 = S0.f64 * S1.f64
```

Notes

0.5ULP precision, denormals are supported.

#### V_MIN_F64  (opcode 642)

Select the minimum of two double-precision float inputs and store the result into a vector register.

```
  if (WAVE_MODE.IEEE && isSignalNAN(S0.f64)) then
        D0.f64 = cvtToQuietNAN(S0.f64)
  elsif (WAVE_MODE.IEEE && isSignalNAN(S1.f64)) then
        D0.f64 = cvtToQuietNAN(S1.f64)
  elsif isNAN(S0.f64) then
        D0.f64 = S1.f64
  elsif isNAN(S1.f64) then
        D0.f64 = S0.f64
  elsif ((S0.f64 == +0.0) && (S1.f64 == -0.0)) then
        D0.f64 = S1.f64
  elsif ((S0.f64 == -0.0) && (S1.f64 == +0.0)) then
        D0.f64 = S0.f64
  else
        // Note: there's no IEEE case here like there is for V_MAX_F64.
        D0.f64 = S0.f64 < S1.f64 ? S0.f64 : S1.f64
  endif
```

#### V_MAX_F64  (opcode 643)

Select the maximum of two double-precision float inputs and store the result into a vector register.

```
  if (WAVE_MODE.IEEE && isSignalNAN(S0.f64)) then
        D0.f64 = cvtToQuietNAN(S0.f64)
  elsif (WAVE_MODE.IEEE && isSignalNAN(S1.f64)) then
        D0.f64 = cvtToQuietNAN(S1.f64)
  elsif isNAN(S0.f64) then
        D0.f64 = S1.f64
  elsif isNAN(S1.f64) then
        D0.f64 = S0.f64
  elsif ((S0.f64 == +0.0) && (S1.f64 == -0.0)) then
        D0.f64 = S0.f64
  elsif ((S0.f64 == -0.0) && (S1.f64 == +0.0)) then
```

```
        D0.f64 = S1.f64
  elsif WAVE_MODE.IEEE then
        D0.f64 = S0.f64 >= S1.f64 ? S0.f64 : S1.f64
  else
        D0.f64 = S0.f64 > S1.f64 ? S0.f64 : S1.f64
  endif
```

#### V_LDEXP_F64  (opcode 644)

Multiply the first input, a floating point value, by an integral power of 2 specified in the second input, a signed
integer value, and store the floating point result into a vector register.

```
  D0.f64 = S0.f64 * 2.0 ** S1.i32
```

Notes

Compare with the ldexp() function in C.

#### V_MUL_LO_U32  (opcode 645)

Multiply two unsigned 32-bit integer inputs and store the result into a vector register.

```
  D0.u32 = S0.u32 * S1.u32
```

Notes

To multiply integers with small magnitudes consider V_MUL_U32_U24, which is intended to be a more
efficient implementation.

#### V_MUL_HI_U32  (opcode 646)

Multiply two unsigned 32-bit integer inputs and store the high 32 bits of the result into a vector register.

```
  D0.u32 = 32'U((64'U(S0.u32) * 64'U(S1.u32)) >> 32U)
```

Notes

To multiply integers with small magnitudes consider V_MUL_HI_U32_U24, which is intended to be a more
efficient implementation.

#### V_MUL_HI_I32  (opcode 647)

Multiply two signed 32-bit integer inputs and store the high 32 bits of the result into a vector register.

```
  D0.i32 = 32'I((64'I(S0.i32) * 64'I(S1.i32)) >> 32U)
```

Notes

To multiply integers with small magnitudes consider V_MUL_HI_I32_I24, which is intended to be a more
efficient implementation.

#### V_LDEXP_F32  (opcode 648)

Multiply the first input, a floating point value, by an integral power of 2 specified in the second input, a signed
integer value, and store the floating point result into a vector register.

```
  D0.f32 = S0.f32 * 2.0F ** S1.i32
```

Notes

Compare with the ldexp() function in C.

#### V_READLANE_B32  (opcode 649)

Read the scalar value in the specified lane of the first input where the lane select is in the second input. Store
the result into a scalar register.

```
  lane = S1.u32[5 : 0];
  // Lane select
  D0.b32 = VGPR[lane][SRC0.u32]
```

Notes

Overrides EXEC mask for the VGPR read. Input and output modifiers not supported; this is an untyped
operation.

#### V_WRITELANE_B32  (opcode 650)

Write the scalar value in the first input into the specified lane of a vector register where the lane select is in the
second input.

```
  lane = S1.u32[5 : 0];
```

```
  // Lane select
  VGPR[lane][VDST.u32] = S0.b32
```

Notes

Overrides EXEC mask for the VGPR write. Input and output modifiers not supported; this is an untyped
operation.

#### V_BCNT_U32_B32  (opcode 651)

Count the number of "1" bits in the vector input and store the result into a vector register.

```
  tmp = S1.u32;
  for i in 0 : 31 do
        tmp += S0[i].u32;
        // count i'th bit
  endfor;
  D0.u32 = tmp
```

#### V_MBCNT_LO_U32_B32  (opcode 652)

For each lane 0 <= N < 32, examine the N least significant bits of the first input and count how many of those
bits are "1". For each lane 32 <= N < 64, all "1" bits in the first input are counted. Add this count to the value in
the second input and store the result into a vector register.

In conjunction with V_MBCNT_HI_U32_B32 and with a vector condition code as input, this counts the number
of lanes at or below the current lane number that have set their vector condition code bit.

```
  ThreadMask = (1LL << laneId.u32) - 1LL;
  MaskedValue = (S0.u32 & ThreadMask[31 : 0].u32);
  tmp = S1.u32;
  for i in 0 : 31 do
        tmp += MaskedValue[i] == 1'1U ? 1U : 0U
  endfor;
  D0.u32 = tmp
```

Notes

See also V_MBCNT_HI_U32_B32.

#### V_MBCNT_HI_U32_B32  (opcode 653)

For each lane 32 <= N < 64, examine the N least significant bits of the first input and count how many of those
bits are "1". For lane positions 0 <= N < 32 no bits are examined and the count is zero. Add this count to the

value in the second input and store the result into a vector register.

In conjunction with V_MBCNT_LO_U32_B32 and with a vector condition code as input, this counts the number
of lanes at or below the current lane number that have set their vector condition code bit.

```
  ThreadMask = (1LL << laneId.u32) - 1LL;
  MaskedValue = (S0.u32 & ThreadMask[63 : 32].u32);
  tmp = S1.u32;
  for i in 0 : 31 do
        tmp += MaskedValue[i] == 1'1U ? 1U : 0U
  endfor;
  D0.u32 = tmp
```

Notes

Example to compute each lane's position in 0..63:

```
        v_mbcnt_lo_u32_b32 v0, -1, 0
        v_mbcnt_hi_u32_b32 v0, -1, v0
        // v0 now contains laneId
```

Example to compute each lane's position in a list of all lanes whose VCC bits are set, where the first lane with
VCC set is assigned position 1, the second lane with VCC set is assigned position 2, etc.:

```
        v_mbcnt_lo_u32_b32 v0, vcc_lo, 0
        v_mbcnt_hi_u32_b32 v0, vcc_hi, v0 // Note vcc_hi is passed in for second instruction
        // v0 now contains position among lanes with VCC=1
```

See also V_MBCNT_LO_U32_B32.

#### V_LSHLREV_B64  (opcode 655)

Given a shift count in the first vector input, calculate the logical shift left of the second vector input and store the
result into a vector register.

```
  D0.u64 = (S1.u64 << S0[5 : 0].u32)
```

#### V_LSHRREV_B64  (opcode 656)

Given a shift count in the first vector input, calculate the logical shift right of the second vector input and store
the result into a vector register.

```
  D0.u64 = (S1.u64 >> S0[5 : 0].u32)
```

#### V_ASHRREV_I64  (opcode 657)

Given a shift count in the first vector input, calculate the arithmetic shift right (preserving sign bit) of the second
vector input and store the result into a vector register.

```
  D0.i64 = (S1.i64 >> S0[5 : 0].u32)
```

#### V_TRIG_PREOP_F64  (opcode 658)

Look up a 53-bit segment of 2/PI using an integer segment select in the second input. Scale the intermediate
result by the exponent from the first double-precision float input and store the double-precision float result
into a vector register.

This operation returns an aligned, double precision segment of 2/PI needed to do trigonometric argument
reduction on the floating point input. Multiple segments can be accessed using the first input. Rounding is
toward zero. Large floating point inputs (with an exponent > 1968) are scaled to avoid loss of precision through
denormalization.

```
  shift = 32'I(S1[4 : 0].u32) * 53;
  if exponent(S0.f64) > 1077 then
        shift += exponent(S0.f64) - 1077
  endif;
  // (2.0/PI) == 0.{b_1200, b_1199, b_1198, ..., b_1, b_0}
  // b_1200 is the MSB of the fractional part of 2.0/PI
  // Left shift operation indicates which bits are brought
  // into the whole part of the number.
  // Only whole part of result is kept.
  result = 64'F((1201'B(2.0 / PI)[1200 : 0] << shift.u32) & 1201'0x1fffffffffffff);
  scale = -53 - shift;
  if exponent(S0.f64) >= 1968 then
        scale += 128
  endif;
  D0.f64 = ldexp(result, scale)
```

Notes

For a more complete treatment of trigonometric argument reduction refer to Argument Reduction for Huge
Arguments: Good to the Last Bit, K. C. Ng et.al., March 1992, available online.

#### V_BFM_B32  (opcode 659)

Calculate a bitfield mask given a field offset and size and store the result into a vector register.

```
  D0.u32 = (((1U << S0[4 : 0].u32) - 1U) << S1[4 : 0].u32)
```

#### V_CVT_PKNORM_I16_F32  (opcode 660)

Convert from two single-precision float inputs to a packed signed normalized short and store the result into a
vector register.

```
  declare tmp : 32'B;
  tmp[15 : 0].i16 = f32_to_snorm(S0.f32);
  tmp[31 : 16].i16 = f32_to_snorm(S1.f32);
  D0 = tmp.b32
```

#### V_CVT_PKNORM_U16_F32  (opcode 661)

Convert from two single-precision float inputs to a packed unsigned normalized short and store the result into
a vector register.

```
  declare tmp : 32'B;
  tmp[15 : 0].u16 = f32_to_unorm(S0.f32);
  tmp[31 : 16].u16 = f32_to_unorm(S1.f32);
  D0 = tmp.b32
```

#### V_CVT_PKRTZ_F16_F32  (opcode 662)

Convert two single-precision float inputs to a packed half-precision float value using round toward zero
semantics (ignore the current rounding mode), and store the result into a vector register.

```
  prev_mode = ROUND_MODE;
  ROUND_MODE = ROUND_TOWARD_ZERO;
  tmp[15 : 0].f16 = f32_to_f16(S0.f32);
  tmp[31 : 16].f16 = f32_to_f16(S1.f32);
  D0 = tmp.b32;
  ROUND_MODE = prev_mode;
  // Round-toward-zero regardless of current round mode setting in hardware.
```

Notes

This opcode is intended for use with 16-bit compressed exports. See V_CVT_F16_F32 for a version that respects
the current rounding mode.

#### V_CVT_PK_U16_U32  (opcode 663)

Convert from two unsigned 32-bit integer inputs to a packed unsigned 16-bit integer value and store the result
into a vector register.

```
  declare tmp : 32'B;
  tmp[15 : 0].u16 = u32_to_u16(S0.u32);
  tmp[31 : 16].u16 = u32_to_u16(S1.u32);
  D0 = tmp.b32
```

#### V_CVT_PK_I16_I32  (opcode 664)

Convert from two signed 32-bit integer inputs to a packed signed 16-bit integer value and store the result into a
vector register.

```
  declare tmp : 32'B;
  tmp[15 : 0].i16 = i32_to_i16(S0.i32);
  tmp[31 : 16].i16 = i32_to_i16(S1.i32);
  D0 = tmp.b32
```

#### V_CVT_PKNORM_I16_F16  (opcode 665)

Convert from two half-precision float inputs to a packed signed normalized short and store the result into a
vector register.

```
  declare tmp : 32'B;
  tmp[15 : 0].i16 = f16_to_snorm(S0.f16);
  tmp[31 : 16].i16 = f16_to_snorm(S1.f16);
  D0 = tmp.b32
```

#### V_CVT_PKNORM_U16_F16  (opcode 666)

Convert from two half-precision float inputs to a packed unsigned normalized short and store the result into a
vector register.

```
  declare tmp : 32'B;
  tmp[15 : 0].u16 = f16_to_unorm(S0.f16);
  tmp[31 : 16].u16 = f16_to_unorm(S1.f16);
  D0 = tmp.b32
```

#### V_ADD_I32  (opcode 668)

Add two signed 32-bit integer inputs and store the result into a vector register. No carry-in or carry-out support.

```
  D0.i32 = S0.i32 + S1.i32
```

Notes

Supports saturation (signed 32-bit integer domain).

#### V_SUB_I32  (opcode 669)

Subtract the second signed 32-bit integer input from the first input and store the result into a vector register.
No carry-in or carry-out support.

```
  D0.i32 = S0.i32 - S1.i32
```

Notes

Supports saturation (signed 32-bit integer domain).

#### V_ADD_I16  (opcode 670)

Add two signed 16-bit integer inputs and store the result into a vector register. No carry-in or carry-out support.

```
  D0.i16 = S0.i16 + S1.i16
```

Notes

Supports saturation (signed 16-bit integer domain).

#### V_SUB_I16  (opcode 671)

Subtract the second signed 16-bit integer input from the first input and store the result into a vector register.
No carry-in or carry-out support.

```
  D0.i16 = S0.i16 - S1.i16
```

Notes

Supports saturation (signed 16-bit integer domain).

#### V_PACK_B32_F16  (opcode 672)

Pack two half-precision float values into a single 32-bit value and store the result into a vector register.

```
  D0[31 : 16].f16 = S1.f16;
  D0[15 : 0].f16 = S0.f16
```

#### V_MUL_LEGACY_F32  (opcode 673)

Multiply two floating point inputs and store the result into a vector register. Follows DX9 rules where 0.0 times
anything produces 0.0 (this differs from other APIs when the other input is infinity or NaN).

```
  if ((64'F(S0.f32) == 0.0) || (64'F(S1.f32) == 0.0)) then
        // DX9 rules, 0.0 * x = 0.0
        D0.f32 = 0.0F
  else
        D0.f32 = S0.f32 * S1.f32
  endif
```

#### V_CVT_PK_FP8_F32  (opcode 674)

Convert from two single-precision float inputs to a packed FP8 float value with round to nearest even semantics
and store the result into 16 bits of a vector register using OPSEL.

```
  prev_mode = ROUND_MODE;
  ROUND_MODE = ROUND_NEAREST_EVEN;
  if OPSEL[3].u32 == 0U then
        VGPR[laneId][VDST.u32][15 : 0].b16 = { f32_to_fp8(S1.f32), f32_to_fp8(S0.f32) };
        // D0[31:16] are preserved
  else
        VGPR[laneId][VDST.u32][31 : 16].b16 = { f32_to_fp8(S1.f32), f32_to_fp8(S0.f32) };
        // D0[15:0] are preserved
  endif;
  ROUND_MODE = prev_mode
```

Notes

Round to nearest even. Ignores OMOD and clamp.

#### V_CVT_PK_BF8_F32  (opcode 675)

Convert from two single-precision float inputs to a packed BF8 float value with round to nearest even

semantics and store the result into 16 bits of a vector register using OPSEL.

```
  prev_mode = ROUND_MODE;
  ROUND_MODE = ROUND_NEAREST_EVEN;
  if OPSEL[3].u32 == 0U then
        VGPR[laneId][VDST.u32][15 : 0].b16 = { f32_to_bf8(S1.f32), f32_to_bf8(S0.f32) };
        // D0[31:16] are preserved
  else
        VGPR[laneId][VDST.u32][31 : 16].b16 = { f32_to_bf8(S1.f32), f32_to_bf8(S0.f32) };
        // D0[15:0] are preserved
  endif;
  ROUND_MODE = prev_mode
```

Notes

Round to nearest even. Ignores OMOD and clamp.

#### V_CVT_SR_FP8_F32  (opcode 676)

Convert from a single-precision float input to an FP8 value with stochastic rounding using seed data from the
second input. Store the result into 8 bits of a vector register using OPSEL to determine which byte of the
destination to overwrite.

```
  prev_mode = ROUND_MODE;
  ROUND_MODE = ROUND_NEAREST_EVEN;
  s = sign(S0.f32);
  e = exponent(S0.f32);
  m = 23'U(32'U(23'B(mantissa(S0.f32))) + S1[31 : 12].u32);
  tmp = float32(s, e, m);
  // Add stochastic value to mantissa, wrap around on overflow
  if OPSEL[3 : 2].u2 == 2'0U then
        VGPR[laneId][VDST.u32][7 : 0].fp8 = f32_to_fp8(tmp.f32)
  elsif OPSEL[3 : 2].u2 == 2'1U then
        VGPR[laneId][VDST.u32][15 : 8].fp8 = f32_to_fp8(tmp.f32)
  elsif OPSEL[3 : 2].u2 == 2'2U then
        VGPR[laneId][VDST.u32][23 : 16].fp8 = f32_to_fp8(tmp.f32)
  else
        VGPR[laneId][VDST.u32][31 : 24].fp8 = f32_to_fp8(tmp.f32)
  endif;
  // Unwritten bytes of D are preserved.
  ROUND_MODE = prev_mode
```

Notes

Stochastic rounding. Ignores OMOD and clamp.

#### V_CVT_SR_BF8_F32  (opcode 677)

Convert from a single-precision float input to a BF8 value with stochastic rounding using seed data from the
second input. Store the result into 8 bits of a vector register using OPSEL to determine which byte of the
destination to overwrite.

```
  prev_mode = ROUND_MODE;
  ROUND_MODE = ROUND_NEAREST_EVEN;
  s = sign(S0.f32);
  e = exponent(S0.f32);
  m = 23'U(32'U(23'B(mantissa(S0.f32))) + S1[31 : 11].u32);
  tmp = float32(s, e, m);
  // Add stochastic value to mantissa, wrap around on overflow
  if OPSEL[3 : 2].u2 == 2'0U then
        VGPR[laneId][VDST.u32][7 : 0].bf8 = f32_to_bf8(tmp.f32)
  elsif OPSEL[3 : 2].u2 == 2'1U then
        VGPR[laneId][VDST.u32][15 : 8].bf8 = f32_to_bf8(tmp.f32)
  elsif OPSEL[3 : 2].u2 == 2'2U then
        VGPR[laneId][VDST.u32][23 : 16].bf8 = f32_to_bf8(tmp.f32)
  else
        VGPR[laneId][VDST.u32][31 : 24].bf8 = f32_to_bf8(tmp.f32)
  endif;
  // Unwritten bytes of D are preserved.
  ROUND_MODE = prev_mode
```

Notes

Stochastic rounding. Ignores OMOD and clamp.

#### V_CVT_SR_F16_F32  (opcode 678)

Convert from a single-precision float input to a half-precision value with stochastic rounding using seed data
from the second input. Store the result into 16 bits of a vector register using OPSEL to determine which word of
the destination to overwrite.

```
  prev_mode = ROUND_MODE;
  ROUND_MODE = ROUND_NEAREST_EVEN;
  if OPSEL[3].u2 == 2'0U then
        VGPR[laneId][VDST.u32][15 : 0].f16 = 16'F(f32_to_f16_SR(S0.f32, S1.u32))
  else
        VGPR[laneId][VDST.u32][31 : 16].f16 = 16'F(f32_to_f16_sr(S0.f32, S1.u32))
  endif;
  // Unwritten bytes of D are preserved.
  ROUND_MODE = prev_mode
```

Notes

Stochastic rounding. Ignores OMOD and clamp.

#### V_CVT_SR_BF16_F32  (opcode 679)

Convert from a single-precision float input to a BF16 value with stochastic rounding using seed data from the
second input. Store the result into 16 bits of a vector register using OPSEL to determine which word of the
destination to overwrite.

```
  prev_mode = ROUND_MODE;
  ROUND_MODE = ROUND_NEAREST_EVEN;
  if OPSEL[3].u2 == 2'0U then
        VGPR[laneId][VDST.u32][15 : 0].bf16 = 16'BF(f32_to_bf16_SR(S0.f32, S1.u32))
  else
        VGPR[laneId][VDST.u32][31 : 16].bf16 = 16'BF(f32_to_bf16_sr(S0.f32, S1.u32))
  endif;
  // Unwritten bytes of D are preserved.
  ROUND_MODE = prev_mode
```

Notes

Stochastic rounding. Ignores OMOD and clamp.

#### V_MINIMUM3_F32  (opcode 680)

Select the IEEE minimum() of three single-precision float inputs and store the result into a vector register.

A signaling NaN in either argument is propagated to the result.

```
  D0.f32 = 32'F(v_minimum_f32(v_minimum_f32(S0.f32, S1.f32), S2.f32))
```

Notes

DX10_CLAMP forces a NAN to zero. The IEEE mode bit is ignored (hardware forces it to 1 for this operation).

#### V_MAXIMUM3_F32  (opcode 681)

Select the IEEE maximum() of three single-precision float inputs and store the result into a vector register.

A signaling NaN in either argument is propagated to the result.

```
  D0.f32 = 32'F(v_maximum_f32(v_maximum_f32(S0.f32, S1.f32), S2.f32))
```

Notes

DX10_CLAMP forces a NAN to zero. The IEEE mode bit is ignored (hardware forces it to 1 for this operation).

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
```

```
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
      result = S1.u32[sign(S0.f64) ? 3 : 8]
  elsif abs(S0.f64) > 0.0 then
      // +-denormal value
```

```
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
```

```
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
```

```
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
