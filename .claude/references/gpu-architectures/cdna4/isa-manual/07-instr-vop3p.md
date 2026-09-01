# CDNA4 ISA Instructions: Packed Vector (VOP3P)

> Part of the **CDNA4 Instruction Set Architecture** reference. See the [index](README.md) for other sections.

## Sections in this file

  - [12.10. VOP3P Instructions](#1210-vop3p-instructions)

## Instruction mnemonics in this file

- **12.10. VOP3P Instructions**: V_PK_MAD_I16, V_PK_MUL_LO_U16, V_PK_ADD_I16, V_PK_SUB_I16, V_PK_LSHLREV_B16, V_PK_LSHRREV_B16, V_PK_ASHRREV_I16, V_PK_MAX_I16, V_PK_MIN_I16, V_PK_MAD_U16, V_PK_ADD_U16, V_PK_SUB_U16, V_PK_MAX_U16, V_PK_MIN_U16, V_PK_FMA_F16, V_PK_ADD_F16, V_PK_MUL_F16, V_PK_MIN_F16, V_PK_MAX_F16, V_DOT2_F32_BF16, V_PK_MINIMUM3_F16, V_PK_MAXIMUM3_F16, V_MAD_MIX_F32, V_MAD_MIXLO_F16, V_MAD_MIXHI_F16, V_DOT2_F32_F16, V_DOT2_I32_I16, V_DOT2_U32_U16, V_DOT4_I32_I8, V_DOT4_U32_U8, V_DOT8_I32_I4, V_DOT8_U32_U4, V_MFMA_F32_16X16X128_F8F6F4, V_MFMA_F32_32X32X64_F8F6F4, V_PK_FMA_F32, V_PK_MUL_F32, V_PK_ADD_F32, V_PK_MOV_B32, V_MFMA_F32_16X16X32_BF16, V_MFMA_I32_16X16X64_I8, V_MFMA_F32_32X32X16_BF16, V_MFMA_I32_32X32X32_I8, V_SMFMAC_F32_16X16X64_BF16, V_SMFMAC_I32_16X16X128_I8, V_SMFMAC_F32_16X16X128_BF8_BF8, V_SMFMAC_F32_16X16X128_BF8_FP8, V_SMFMAC_F32_16X16X128_FP8_BF8, V_MFMA_F32_32X32X1_2B_F32, V_MFMA_F32_16X16X1_4B_F32, V_MFMA_F32_4X4X1_16B_F32, V_SMFMAC_F32_16X16X128_FP8_FP8, V_MFMA_F32_32X32X2_F32, V_MFMA_F32_16X16X4_F32, V_SMFMAC_F32_32X32X32_BF16, V_SMFMAC_I32_32X32X64_I8, V_MFMA_F32_32X32X4_2B_F16, V_MFMA_F32_16X16X4_4B_F16, V_MFMA_F32_4X4X4_16B_F16, V_SMFMAC_F32_32X32X64_BF8_BF8, V_MFMA_F32_32X32X8_F16, V_MFMA_F32_16X16X16_F16, V_SMFMAC_F32_32X32X64_BF8_FP8, V_SMFMAC_F32_32X32X64_FP8_BF8, V_MFMA_I32_32X32X4_2B_I8, V_MFMA_I32_16X16X4_4B_I8, V_MFMA_I32_4X4X4_16B_I8, V_SMFMAC_F32_32X32X64_FP8_FP8, V_MFMA_F32_16X16X32_F16, V_MFMA_F32_32X32X16_F16, V_MFMA_I32_32X32X16_I8, V_MFMA_I32_16X16X32_I8, V_ACCVGPR_READ, V_ACCVGPR_WRITE, V_SMFMAC_F32_16X16X64_F16, V_SMFMAC_F32_32X32X32_F16, V_MFMA_F32_32X32X4_2B_BF16, V_MFMA_F32_16X16X4_4B_BF16, V_MFMA_F32_4X4X4_16B_BF16, V_MFMA_F32_32X32X8_BF16, V_MFMA_F32_16X16X16_BF16, V_SMFMAC_F32_16X16X32_F16, V_SMFMAC_F32_32X32X16_F16, V_SMFMAC_F32_16X16X32_BF16, V_SMFMAC_F32_32X32X16_BF16, V_SMFMAC_I32_16X16X64_I8, V_SMFMAC_I32_32X32X32_I8, V_MFMA_F64_16X16X4_F64, V_MFMA_F64_4X4X4_4B_F64, V_MFMA_F32_16X16X32_BF8_BF8, V_MFMA_F32_16X16X32_BF8_FP8, V_MFMA_F32_16X16X32_FP8_BF8, V_MFMA_F32_16X16X32_FP8_FP8, V_MFMA_F32_32X32X16_BF8_BF8, V_MFMA_F32_32X32X16_BF8_FP8, V_MFMA_F32_32X32X16_FP8_BF8, V_MFMA_F32_32X32X16_FP8_FP8, V_SMFMAC_F32_16X16X64_BF8_BF8, V_SMFMAC_F32_16X16X64_BF8_FP8, V_SMFMAC_F32_16X16X64_FP8_BF8, V_SMFMAC_F32_16X16X64_FP8_FP8, V_SMFMAC_F32_32X32X32_BF8_BF8, V_SMFMAC_F32_32X32X32_BF8_FP8, V_SMFMAC_F32_32X32X32_FP8_BF8, V_SMFMAC_F32_32X32X32_FP8_FP8

---

### 12.10. VOP3P Instructions

#### V_PK_MAD_I16  (opcode 0)

Multiply two packed signed 16-bit integer inputs component-wise, add a packed signed 16-bit integer value
from a third input component-wise, and store the result into a vector register.

```
  declare tmp : 32'B;
  tmp[15 : 0].i16 = S0[15 : 0].i16 * S1[15 : 0].i16 + S2[15 : 0].i16;
  tmp[31 : 16].i16 = S0[31 : 16].i16 * S1[31 : 16].i16 + S2[31 : 16].i16;
  D0.b32 = tmp
```

#### V_PK_MUL_LO_U16  (opcode 1)

Multiply two packed unsigned 16-bit integer inputs component-wise and store the low bits of each resulting
component into a vector register.

```
  tmp[31 : 16].u16 = S0[31 : 16].u16 * S1[31 : 16].u16;
  tmp[15 : 0].u16 = S0[15 : 0].u16 * S1[15 : 0].u16;
  D0.b32 = tmp.b32
```

#### V_PK_ADD_I16  (opcode 2)

Add two packed signed 16-bit integer inputs component-wise and store the result into a vector register. No
carry-in or carry-out support.

```
  declare tmp : 32'B;
  tmp[15 : 0].i16 = S0[15 : 0].i16 + S1[15 : 0].i16;
  tmp[31 : 16].i16 = S0[31 : 16].i16 + S1[31 : 16].i16;
  D0.b32 = tmp
```

#### V_PK_SUB_I16  (opcode 3)

Subtract the second packed signed 16-bit integer input from the first input component-wise and store the result
into a vector register. No carry-in or carry-out support.

```
  declare tmp : 32'B;
  tmp[15 : 0].i16 = S0[15 : 0].i16 - S1[15 : 0].i16;
  tmp[31 : 16].i16 = S0[31 : 16].i16 - S1[31 : 16].i16;
  D0.b32 = tmp
```

#### V_PK_LSHLREV_B16  (opcode 4)

Given a packed shift count in the first vector input, calculate the component-wise logical shift left of the second
packed vector input and store the result into a vector register.

```
  tmp[31 : 16].u16 = (S1[31 : 16].u16 << S0.u32[19 : 16].u32);
  tmp[15 : 0].u16 = (S1[15 : 0].u16 << S0.u32[3 : 0].u32);
  D0.b32 = tmp.b32
```

#### V_PK_LSHRREV_B16  (opcode 5)

Given a packed shift count in the first vector input, calculate the component-wise logical shift right of the
second packed vector input and store the result into a vector register.

```
  tmp[31 : 16].u16 = (S1[31 : 16].u16 >> S0.u32[19 : 16].u32);
  tmp[15 : 0].u16 = (S1[15 : 0].u16 >> S0.u32[3 : 0].u32);
  D0.b32 = tmp.b32
```

#### V_PK_ASHRREV_I16  (opcode 6)

Given a packed shift count in the first vector input, calculate the component-wise arithmetic shift right
(preserving sign bit) of the second packed vector input and store the result into a vector register.

```
  tmp[31 : 16].i16 = (S1[31 : 16].i16 >> S0.u32[19 : 16].u32);
  tmp[15 : 0].i16 = (S1[15 : 0].i16 >> S0.u32[3 : 0].u32);
  D0.b32 = tmp.b32
```

#### V_PK_MAX_I16  (opcode 7)

Select the component-wise maximum of two packed signed 16-bit integer inputs and store the selected values
into a vector register.

```
  declare tmp : 32'B;
  tmp[15 : 0].i16 = S0[15 : 0].i16 >= S1[15 : 0].i16 ? S0[15 : 0].i16 : S1[15 : 0].i16;
  tmp[31 : 16].i16 = S0[31 : 16].i16 >= S1[31 : 16].i16 ? S0[31 : 16].i16 : S1[31 : 16].i16;
  D0.b32 = tmp
```

#### V_PK_MIN_I16  (opcode 8)

Select the component-wise minimum of two packed signed 16-bit integer inputs and store the selected values
into a vector register.

```
  declare tmp : 32'B;
  tmp[15 : 0].i16 = S0[15 : 0].i16 < S1[15 : 0].i16 ? S0[15 : 0].i16 : S1[15 : 0].i16;
  tmp[31 : 16].i16 = S0[31 : 16].i16 < S1[31 : 16].i16 ? S0[31 : 16].i16 : S1[31 : 16].i16;
  D0.b32 = tmp
```

#### V_PK_MAD_U16  (opcode 9)

Multiply two packed unsigned 16-bit integer inputs component-wise, add a packed unsigned 16-bit integer
value from a third input component-wise, and store the result into a vector register.

```
  declare tmp : 32'B;
  tmp[15 : 0].u16 = S0[15 : 0].u16 * S1[15 : 0].u16 + S2[15 : 0].u16;
  tmp[31 : 16].u16 = S0[31 : 16].u16 * S1[31 : 16].u16 + S2[31 : 16].u16;
  D0.b32 = tmp
```

#### V_PK_ADD_U16  (opcode 10)

Add two packed unsigned 16-bit integer inputs component-wise and store the result into a vector register. No
carry-in or carry-out support.

```
  declare tmp : 32'B;
  tmp[15 : 0].u16 = S0[15 : 0].u16 + S1[15 : 0].u16;
  tmp[31 : 16].u16 = S0[31 : 16].u16 + S1[31 : 16].u16;
  D0.b32 = tmp
```

#### V_PK_SUB_U16  (opcode 11)

Subtract the second packed unsigned 16-bit integer input from the first input component-wise and store the
result into a vector register. No carry-in or carry-out support.

```
  declare tmp : 32'B;
  tmp[15 : 0].u16 = S0[15 : 0].u16 - S1[15 : 0].u16;
  tmp[31 : 16].u16 = S0[31 : 16].u16 - S1[31 : 16].u16;
  D0.b32 = tmp
```

#### V_PK_MAX_U16  (opcode 12)

Select the component-wise maximum of two packed unsigned 16-bit integer inputs and store the selected
values into a vector register.

```
  declare tmp : 32'B;
  tmp[15 : 0].u16 = S0[15 : 0].u16 >= S1[15 : 0].u16 ? S0[15 : 0].u16 : S1[15 : 0].u16;
  tmp[31 : 16].u16 = S0[31 : 16].u16 >= S1[31 : 16].u16 ? S0[31 : 16].u16 : S1[31 : 16].u16;
  D0.b32 = tmp
```

#### V_PK_MIN_U16  (opcode 13)

Select the component-wise minimum of two packed unsigned 16-bit integer inputs and store the selected
values into a vector register.

```
  declare tmp : 32'B;
  tmp[15 : 0].u16 = S0[15 : 0].u16 < S1[15 : 0].u16 ? S0[15 : 0].u16 : S1[15 : 0].u16;
  tmp[31 : 16].u16 = S0[31 : 16].u16 < S1[31 : 16].u16 ? S0[31 : 16].u16 : S1[31 : 16].u16;
  D0.b32 = tmp
```

#### V_PK_FMA_F16  (opcode 14)

Multiply two packed half-precision float inputs component-wise and add a third input component-wise using
fused multiply add, and store the result into a vector register.

```
  declare tmp : 32'B;
  tmp[15 : 0].f16 = fma(S0[15 : 0].f16, S1[15 : 0].f16, S2[15 : 0].f16);
  tmp[31 : 16].f16 = fma(S0[31 : 16].f16, S1[31 : 16].f16, S2[31 : 16].f16);
  D0.b32 = tmp
```

#### V_PK_ADD_F16  (opcode 15)

Add two packed half-precision float inputs component-wise and store the result into a vector register. No carry-
in or carry-out support.

```
  declare tmp : 32'B;
  tmp[15 : 0].f16 = S0[15 : 0].f16 + S1[15 : 0].f16;
  tmp[31 : 16].f16 = S0[31 : 16].f16 + S1[31 : 16].f16;
  D0.b32 = tmp
```

#### V_PK_MUL_F16  (opcode 16)

Multiply two packed half-precision float inputs component-wise and store the result into a vector register.

```
  declare tmp : 32'B;
  tmp[15 : 0].f16 = S0[15 : 0].f16 * S1[15 : 0].f16;
  tmp[31 : 16].f16 = S0[31 : 16].f16 * S1[31 : 16].f16;
  D0.b32 = tmp
```

#### V_PK_MIN_F16  (opcode 17)

Select the component-wise minimum of two packed half-precision float inputs and store the result into a vector
register.

```
  declare tmp : 32'B;
  tmp[15 : 0].f16 = v_min_f16(S0[15 : 0].f16, S1[15 : 0].f16);
  tmp[31 : 16].f16 = v_min_f16(S0[31 : 16].f16, S1[31 : 16].f16);
  D0.b32 = tmp
```

#### V_PK_MAX_F16  (opcode 18)

Select the component-wise maximum of two packed half-precision float inputs and store the result into a
vector register.

```
  declare tmp : 32'B;
  tmp[15 : 0].f16 = v_max_f16(S0[15 : 0].f16, S1[15 : 0].f16);
  tmp[31 : 16].f16 = v_max_f16(S0[31 : 16].f16, S1[31 : 16].f16);
  D0.b32 = tmp
```

#### V_DOT2_F32_BF16  (opcode 26)

Calculate the dot product of BF16 float 2-vectors from the first and second inputs, convert the product to single-
precision float format, add the third input and store the result into a vector register.

```
  tmp = 32'F(S0[15 : 0].bf16) * 32'F(S1[15 : 0].bf16);
  tmp += 32'F(S0[31 : 16].bf16) * 32'F(S1[31 : 16].bf16);
  tmp += S2.f32;
  D0.f32 = tmp
```

Notes

NEG and ABS input modifiers do not affect S2.

#### V_PK_MINIMUM3_F16  (opcode 27)

Select the component-wise IEEE minimum() of three half-precision float inputs and store the result into a
vector register.

A signaling NaN in either argument is propagated to the result.

```
  tmp[31 : 16].f16 = 16'F(v_minimum3_f16(S0[31 : 16].f16, S1[31 : 16].f16, S2[31 : 16].f16));
  tmp[15 : 0].f16 = 16'F(v_minimum3_f16(S0[15 : 0].f16, S1[15 : 0].f16, S2[15 : 0].f16));
  D0.b32 = tmp.b32
```

Notes

DX10_CLAMP forces a NAN to zero. The IEEE mode bit is ignored (hardware forces it to 1 for this operation).

#### V_PK_MAXIMUM3_F16  (opcode 28)

Select the component-wise IEEE maximum() of three half-precision float inputs and store the result into a
vector register.

A signaling NaN in either argument is propagated to the result.

```
  tmp[31 : 16].f16 = 16'F(v_maximum3_f16(S0[31 : 16].f16, S1[31 : 16].f16, S2[31 : 16].f16));
  tmp[15 : 0].f16 = 16'F(v_maximum3_f16(S0[15 : 0].f16, S1[15 : 0].f16, S2[15 : 0].f16));
  D0.b32 = tmp.b32
```

Notes

DX10_CLAMP forces a NAN to zero. The IEEE mode bit is ignored (hardware forces it to 1 for this operation).

#### V_MAD_MIX_F32  (opcode 32)

Multiply two inputs and add a third input where the inputs are a mix of half-precision float and single-
precision float values. Store the result into a vector register.

Size and location of the three inputs are controlled by { OPSEL_HI[i], OPSEL[i] }: 0=src[31:0], 1=src[31:0],
2=src[15:0], 3=src[31:16]. For MIX opcodes the NEG_HI instruction field acts as an absolute-value modifier
for the three inputs.

```
  declare in : 32'F[3];
  declare S : 32'B[3];
  for i in 0 : 2 do
        if !OPSEL_HI.u3[i] then
            in[i] = S[i].f32
        elsif OPSEL.u3[i] then
            in[i] = f16_to_f32(S[i][31 : 16].f16)
        else
            in[i] = f16_to_f32(S[i][15 : 0].f16)
        endif
  endfor;
  D0[31 : 0].f32 = in[0] * in[1] + in[2]
```

Notes

#### V_MAD_MIXLO_F16  (opcode 33)

Multiply two inputs and add a third input where the inputs are a mix of half-precision float and single-
precision float values. Convert the result to a half-precision float. Store the result into the low bits of a vector

register.

Size and location of the three inputs are controlled by { OPSEL_HI[i], OPSEL[i] }: 0=src[31:0], 1=src[31:0],
2=src[15:0], 3=src[31:16]. For MIX opcodes the NEG_HI instruction field acts as an absolute-value modifier
for the three inputs.

```
  declare in : 32'F[3];
  declare S : 32'B[3];
  for i in 0 : 2 do
        if !OPSEL_HI.u3[i] then
            in[i] = S[i].f32
        elsif OPSEL.u3[i] then
            in[i] = f16_to_f32(S[i][31 : 16].f16)
        else
            in[i] = f16_to_f32(S[i][15 : 0].f16)
        endif
  endfor;
  D0[15 : 0].f16 = f32_to_f16(in[0] * in[1] + in[2])
```

Notes

#### V_MAD_MIXHI_F16  (opcode 34)

Multiply two inputs and add a third input where the inputs are a mix of half-precision float and single-
precision float values. Convert the result to a half-precision float. Store the result into the high bits of a vector
register.

Size and location of the three inputs are controlled by { OPSEL_HI[i], OPSEL[i] }: 0=src[31:0], 1=src[31:0],
2=src[15:0], 3=src[31:16]. For MIX opcodes the NEG_HI instruction field acts as an absolute-value modifier
for the three inputs.

```
  declare in : 32'F[3];
  declare S : 32'B[3];
  for i in 0 : 2 do
        if !OPSEL_HI.u3[i] then
            in[i] = S[i].f32
        elsif OPSEL.u3[i] then
            in[i] = f16_to_f32(S[i][31 : 16].f16)
        else
            in[i] = f16_to_f32(S[i][15 : 0].f16)
        endif
  endfor;
  D0[31 : 16].f16 = f32_to_f16(in[0] * in[1] + in[2])
```

Notes

#### V_DOT2_F32_F16  (opcode 35)

Compute the dot product of two packed 2-D half-precision float inputs in the single-precision float domain, add
a single-precision float value from the third input and store the result into a vector register.

```
  tmp = S2.f32;
  tmp += f16_to_f32(S0[15 : 0].f16) * f16_to_f32(S1[15 : 0].f16);
  tmp += f16_to_f32(S0[31 : 16].f16) * f16_to_f32(S1[31 : 16].f16);
  D0.f32 = tmp
```

#### V_DOT2_I32_I16  (opcode 38)

Compute the dot product of two packed 2-D signed 16-bit integer inputs in the signed 32-bit integer domain,
add a signed 32-bit integer value from the third input and store the result into a vector register.

```
  tmp = S2.i32;
  tmp += i16_to_i32(S0[15 : 0].i16) * i16_to_i32(S1[15 : 0].i16);
  tmp += i16_to_i32(S0[31 : 16].i16) * i16_to_i32(S1[31 : 16].i16);
  D0.i32 = tmp
```

#### V_DOT2_U32_U16  (opcode 39)

Compute the dot product of two packed 2-D unsigned 16-bit integer inputs in the unsigned 32-bit integer
domain, add an unsigned 32-bit integer value from the third input and store the result into a vector register.

```
  tmp = S2.u32;
  tmp += u16_to_u32(S0[15 : 0].u16) * u16_to_u32(S1[15 : 0].u16);
  tmp += u16_to_u32(S0[31 : 16].u16) * u16_to_u32(S1[31 : 16].u16);
  D0.u32 = tmp
```

#### V_DOT4_I32_I8  (opcode 40)

Compute the dot product of two packed 4-D signed 8-bit integer inputs in the signed 32-bit integer domain, add
a signed 32-bit integer value from the third input and store the result into a vector register.

```
  tmp = S2.i32;
  tmp += i8_to_i32(S0[7 : 0].i8) * i8_to_i32(S1[7 : 0].i8);
  tmp += i8_to_i32(S0[15 : 8].i8) * i8_to_i32(S1[15 : 8].i8);
  tmp += i8_to_i32(S0[23 : 16].i8) * i8_to_i32(S1[23 : 16].i8);
  tmp += i8_to_i32(S0[31 : 24].i8) * i8_to_i32(S1[31 : 24].i8);
  D0.i32 = tmp
```

#### V_DOT4_U32_U8  (opcode 41)

Compute the dot product of two packed 4-D unsigned 8-bit integer inputs in the unsigned 32-bit integer
domain, add an unsigned 32-bit integer value from the third input and store the result into a vector register.

```
  tmp = S2.u32;
  tmp += u8_to_u32(S0[7 : 0].u8) * u8_to_u32(S1[7 : 0].u8);
  tmp += u8_to_u32(S0[15 : 8].u8) * u8_to_u32(S1[15 : 8].u8);
  tmp += u8_to_u32(S0[23 : 16].u8) * u8_to_u32(S1[23 : 16].u8);
  tmp += u8_to_u32(S0[31 : 24].u8) * u8_to_u32(S1[31 : 24].u8);
  D0.u32 = tmp
```

#### V_DOT8_I32_I4  (opcode 42)

Compute the dot product of two packed 8-D signed 4-bit integer inputs in the signed 32-bit integer domain, add
a signed 32-bit integer value from the third input and store the result into a vector register.

```
  tmp = S2.i32;
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

#### V_DOT8_U32_U4  (opcode 43)

Compute the dot product of two packed 8-D unsigned 4-bit integer inputs in the unsigned 32-bit integer
domain, add an unsigned 32-bit integer value from the third input and store the result into a vector register.

```
  tmp = S2.u32;
  tmp += u4_to_u32(S0[3 : 0].u4) * u4_to_u32(S1[3 : 0].u4);
  tmp += u4_to_u32(S0[7 : 4].u4) * u4_to_u32(S1[7 : 4].u4);
  tmp += u4_to_u32(S0[11 : 8].u4) * u4_to_u32(S1[11 : 8].u4);
  tmp += u4_to_u32(S0[15 : 12].u4) * u4_to_u32(S1[15 : 12].u4);
  tmp += u4_to_u32(S0[19 : 16].u4) * u4_to_u32(S1[19 : 16].u4);
  tmp += u4_to_u32(S0[23 : 20].u4) * u4_to_u32(S1[23 : 20].u4);
  tmp += u4_to_u32(S0[27 : 24].u4) * u4_to_u32(S1[27 : 24].u4);
  tmp += u4_to_u32(S0[31 : 28].u4) * u4_to_u32(S1[31 : 28].u4);
  D0.u32 = tmp
```

#### V_MFMA_F32_16X16X128_F8F6F4  (opcode 45)

Multiply the 16x128 matrix in the first input by the 128x16 matrix in the second input and add the 16x16 matrix
in the third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (16x128) * B (128x16) + C (16x16)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A may be one of several small float formats determined by CBSZ. Matrix B may be one of several small
float formats determined by BLGP. The formats are selected via the following table:

0 FP8 E4M3
1 FP8 E5M2
2 FP6 E2M3
3 FP6 E3M2
4 FP4 E2M1

Matrices C and D are single-precision float format.

Notes

CBSZ and BLGP are overridden to control the A and B matrix formats. NEG[1:0] and ABS[1:0] must be zero.
NEG[2] and ABS[2] may be used to control matrix C. CLAMP is not supported. Round toward nearest even
semantics.

Matrix size requirements:

```
  A, B: (16*128) elements * 8 bits/element = 16384 bits, divide by 64 lanes * 32 bits/register-lane = 8
  registers C, D: (16*16) elements * 32 bits/element = 8192 bits, divide by 64 lanes * 32 bits/register-
  lane = 4 registers
```

#### V_MFMA_F32_32X32X64_F8F6F4  (opcode 46)

Multiply the 32x64 matrix in the first input by the 64x32 matrix in the second input and add the 32x32 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (32x64) * B (64x32) + C (32x32)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A may be one of several small float formats determined by CBSZ. Matrix B may be one of several small

float formats determined by BLGP. The formats are selected via the following table:

0 FP8 E4M3
1 FP8 E5M2
2 FP6 E2M3
3 FP6 E3M2
4 FP4 E2M1

Matrices C and D are single-precision float format.

Notes

CBSZ and BLGP are overridden to control the A and B matrix formats. NEG[1:0] and ABS[1:0] must be zero.
NEG[2] and ABS[2] may be used to control matrix C. CLAMP is not supported. Round toward nearest even
semantics.

Matrix size requirements:

```
  A, B: (32*64) elements * 8 bits/element = 16384 bits, divide by 64 lanes * 32 bits/register-lane = 8
  registers C, D: (32*32) elements * 32 bits/element = 32768 bits, divide by 64 lanes * 32 bits/register-
  lane = 16 registers
```

#### V_PK_FMA_F32  (opcode 48)

Multiply two packed single-precision float inputs component-wise and add a third input component-wise using
fused multiply add, and store the result into a vector register.

```
  declare tmp : 64'B;
  tmp[31 : 0].f32 = fma(S0[31 : 0].f32, S1[31 : 0].f32, S2[31 : 0].f32);
  tmp[63 : 32].f32 = fma(S0[63 : 32].f32, S1[63 : 32].f32, S2[63 : 32].f32);
  D0.b64 = tmp
```

#### V_PK_MUL_F32  (opcode 49)

Multiply two packed single-precision float inputs component-wise and store the result into a vector register.

```
  declare tmp : 64'B;
  tmp[31 : 0].f32 = S0[31 : 0].f32 * S1[31 : 0].f32;
  tmp[63 : 32].f32 = S0[63 : 32].f32 * S1[63 : 32].f32;
  D0.b64 = tmp
```

#### V_PK_ADD_F32  (opcode 50)

Add two packed single-precision float inputs component-wise and store the result into a vector register. No
carry-in or carry-out support.

```
  declare tmp : 64'B;
  tmp[31 : 0].f32 = S0[31 : 0].f32 + S1[31 : 0].f32;
  tmp[63 : 32].f32 = S0[63 : 32].f32 + S1[63 : 32].f32;
  D0.b64 = tmp
```

#### V_PK_MOV_B32  (opcode 51)

Move data from two vector inputs into two vector registers.

```
  tmp0.u32 = S0.u32[OPSEL[0].i32 * 32 + 31 : OPSEL[0].i32 * 32];
  tmp1.u32 = S1.u32[OPSEL[1].i32 * 32 + 31 : OPSEL[1].i32 * 32];
  D0.u32[31 : 0] = tmp0.u32;
  D0.u32[63 : 32] = tmp1.u32
```

Notes

The source operands are treated as 64 bit and are subject to alignment restrictions for both SGPR and VGPR.

For two VGPR inputs this opcode can be used as an arbitrary gather by using OP_SEL to select either the even
VGPR specified or the next odd VGPR.

```
        v_pk_mov_b32 v0, v2, v4 op_sel:[0,1] // evaluates v0 <- v2 and v1 <- v5.
```

Due to scalar broadcast restrictions if two SGPRs are specified as operands, they must be the same SGPR.

```
        v_pk_mov_b32 v0, s6, s6 op_sel:[0,1] // 64-bit move from scalar s[6:7].
```

#### V_MFMA_F32_16X16X32_BF16  (opcode 53)

Multiply the 16x32 matrix in the first input by the 32x16 matrix in the second input and add the 16x16 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (16x32) * B (32x16) + C (16x16)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrices A and B are BF16 float format. Matrices C and D are single-precision float format.

Notes

NEG[1:0] and ABS[1:0] must be zero. NEG[2] and ABS[2] may be used to control matrix C. CLAMP is not
supported. Round toward nearest even semantics.

#### V_MFMA_I32_16X16X64_I8  (opcode 54)

Multiply the 16x64 matrix in the first input by the 64x16 matrix in the second input and add the 16x16 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (16x64) * B (64x16) + C (16x16)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrices A and B are signed 8-bit integer format. Matrices C and D are signed 32-bit integer format.

Notes

NEG[1:0] and ABS[1:0] must be zero. NEG[2] and ABS[2] may be used to control matrix C. CLAMP is not
supported. Round toward nearest even semantics.

#### V_MFMA_F32_32X32X16_BF16  (opcode 55)

Multiply the 32x16 matrix in the first input by the 16x32 matrix in the second input and add the 32x32 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (32x16) * B (16x32) + C (32x32)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrices A and B are BF16 float format. Matrices C and D are single-precision float format.

Notes

NEG[1:0] and ABS[1:0] must be zero. NEG[2] and ABS[2] may be used to control matrix C. CLAMP is not
supported. Round toward nearest even semantics.

#### V_MFMA_I32_32X32X32_I8  (opcode 56)

Multiply the 32x32 matrix in the first input by the 32x32 matrix in the second input and add the 32x32 matrix in

the third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (32x32) * B (32x32) + C (32x32)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrices A and B are signed 8-bit integer format. Matrices C and D are signed 32-bit integer format.

Notes

NEG[1:0] and ABS[1:0] must be zero. NEG[2] and ABS[2] may be used to control matrix C. CLAMP is not
supported. Round toward nearest even semantics.

#### V_SMFMAC_F32_16X16X64_BF16  (opcode 57)

Multiply the 16x64 sparse matrix in the first input by the 64x16 matrix in the second input and accumulate the
result into the 16x16 matrix stored in the destination registers using fused multiply add. Sparse indexes for the
first matrix are given in the third input.

```
  D = A (sparse 16x64) * B (64x16) + D (16x16)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is a sparse matrix in BF16 float format, consuming half the physical storage of a dense matrix with
same dimensions. Matrix B is a dense matrix in BF16 float format. Matrix D is single-precision float format and
is both the output and the accumulate input.

2 out of every 4 elements on the K axis of matrix A are zero. The sparse indexes are used to determine which 2
elements are zero.

Notes

NEG[1:0] and ABS[1:0] must be zero. CLAMP is not supported. Round toward nearest even semantics.

#### V_SMFMAC_I32_16X16X128_I8  (opcode 58)

Multiply the 16x128 sparse matrix in the first input by the 128x16 matrix in the second input and accumulate
the result into the 16x16 matrix stored in the destination registers using fused multiply add. Sparse indexes for
the first matrix are given in the third input.

```
  D = A (sparse 16x128) * B (128x16) + D (16x16)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is a sparse matrix in signed 8-bit integer format, consuming half the physical storage of a dense
matrix with same dimensions. Matrix B is a dense matrix in signed 8-bit integer format. Matrix D is signed 32-
bit integer format and is both the output and the accumulate input.

2 out of every 4 elements on the K axis of matrix A are zero. The sparse indexes are used to determine which 2
elements are zero.

Notes

NEG[1:0] and ABS[1:0] must be zero. CLAMP is not supported. Round toward nearest even semantics.

#### V_SMFMAC_F32_16X16X128_BF8_BF8  (opcode 59)

Multiply the 16x128 sparse matrix in the first input by the 128x16 matrix in the second input and accumulate
the result into the 16x16 matrix stored in the destination registers using fused multiply add. Sparse indexes for
the first matrix are given in the third input.

```
  D = A (sparse 16x128) * B (128x16) + D (16x16)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is a sparse matrix in BF8 float format, consuming half the physical storage of a dense matrix with
same dimensions. Matrix B is a dense matrix in BF8 float format. Matrix D is single-precision float format and
is both the output and the accumulate input.

2 out of every 4 elements on the K axis of matrix A are zero. The sparse indexes are used to determine which 2
elements are zero.

Notes

NEG[1:0] and ABS[1:0] must be zero. CLAMP is not supported. Round toward nearest even semantics.

#### V_SMFMAC_F32_16X16X128_BF8_FP8  (opcode 60)

Multiply the 16x128 sparse matrix in the first input by the 128x16 matrix in the second input and accumulate
the result into the 16x16 matrix stored in the destination registers using fused multiply add. Sparse indexes for
the first matrix are given in the third input.

```
  D = A (sparse 16x128) * B (128x16) + D (16x16)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is a sparse matrix in BF8 float format, consuming half the physical storage of a dense matrix with
same dimensions. Matrix B is a dense matrix in FP8 float format. Matrix D is single-precision float format and
is both the output and the accumulate input.

2 out of every 4 elements on the K axis of matrix A are zero. The sparse indexes are used to determine which 2
elements are zero.

Notes

NEG[1:0] and ABS[1:0] must be zero. CLAMP is not supported. Round toward nearest even semantics.

#### V_SMFMAC_F32_16X16X128_FP8_BF8  (opcode 61)

Multiply the 16x128 sparse matrix in the first input by the 128x16 matrix in the second input and accumulate
the result into the 16x16 matrix stored in the destination registers using fused multiply add. Sparse indexes for
the first matrix are given in the third input.

```
  D = A (sparse 16x128) * B (128x16) + D (16x16)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is a sparse matrix in FP8 float format, consuming half the physical storage of a dense matrix with
same dimensions. Matrix B is a dense matrix in BF8 float format. Matrix D is single-precision float format and
is both the output and the accumulate input.

2 out of every 4 elements on the K axis of matrix A are zero. The sparse indexes are used to determine which 2
elements are zero.

Notes

NEG[1:0] and ABS[1:0] must be zero. CLAMP is not supported. Round toward nearest even semantics.

#### V_MFMA_F32_32X32X1_2B_F32  (opcode 64)

Multiply the 32x1 matrix in the first input by the 1x32 matrix in the second input and add the 32x32 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (32x1) * B (1x32) + C (32x32)
```

This instruction performs 2 matrix multiplies. Each operand contains 2 matrices back to back, and each matrix
has elements distributed across all lanes of the wave. Each matrix multiple is computed and the row-column
dot products are distributed across the vector ALU for higher performance. The result matrices are stored
back-to-back in the destination vector registers.

Matrices A and B are single-precision float format. Matrices C and D are single-precision float format.

Notes

This instruction performs 16 passes.

#### V_MFMA_F32_16X16X1_4B_F32  (opcode 65)

Multiply the 16x1 matrix in the first input by the 1x16 matrix in the second input and add the 16x16 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (16x1) * B (1x16) + C (16x16)
```

This instruction performs 4 matrix multiplies. Each operand contains 4 matrices back to back, and each matrix
has elements distributed across all lanes of the wave. Each matrix multiple is computed and the row-column
dot products are distributed across the vector ALU for higher performance. The result matrices are stored
back-to-back in the destination vector registers.

Matrices A and B are single-precision float format. Matrices C and D are single-precision float format.

Notes

This instruction performs 8 passes.

#### V_MFMA_F32_4X4X1_16B_F32  (opcode 66)

Multiply the 4x1 matrix in the first input by the 1x4 matrix in the second input and add the 4x4 matrix in the
third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (4x1) * B (1x4) + C (4x4)
```

This instruction performs 16 matrix multiplies. Each operand contains 16 matrices back to back, and each
matrix has elements distributed across all lanes of the wave. Each matrix multiple is computed and the row-
column dot products are distributed across the vector ALU for higher performance. The result matrices are
stored back-to-back in the destination vector registers.

Matrices A and B are single-precision float format. Matrices C and D are single-precision float format.

Notes

This instruction performs 2 passes.

#### V_SMFMAC_F32_16X16X128_FP8_FP8  (opcode 67)

Multiply the 16x128 sparse matrix in the first input by the 128x16 matrix in the second input and accumulate
the result into the 16x16 matrix stored in the destination registers using fused multiply add. Sparse indexes for
the first matrix are given in the third input.

```
  D = A (sparse 16x128) * B (128x16) + D (16x16)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is a sparse matrix in FP8 float format, consuming half the physical storage of a dense matrix with
same dimensions. Matrix B is a dense matrix in FP8 float format. Matrix D is single-precision float format and
is both the output and the accumulate input.

2 out of every 4 elements on the K axis of matrix A are zero. The sparse indexes are used to determine which 2
elements are zero.

Notes

NEG[1:0] and ABS[1:0] must be zero. CLAMP is not supported. Round toward nearest even semantics.

#### V_MFMA_F32_32X32X2_F32  (opcode 68)

Multiply the 32x2 matrix in the first input by the 2x32 matrix in the second input and add the 32x32 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (32x2) * B (2x32) + C (32x32)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrices A and B are single-precision float format. Matrices C and D are single-precision float format.

Notes

This instruction performs 16 passes.

#### V_MFMA_F32_16X16X4_F32  (opcode 69)

Multiply the 16x4 matrix in the first input by the 4x16 matrix in the second input and add the 16x16 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (16x4) * B (4x16) + C (16x16)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrices A and B are single-precision float format. Matrices C and D are single-precision float format.

Notes

This instruction performs 8 passes.

#### V_SMFMAC_F32_32X32X32_BF16  (opcode 70)

Multiply the 32x32 sparse matrix in the first input by the 32x32 matrix in the second input and accumulate the
result into the 32x32 matrix stored in the destination registers using fused multiply add. Sparse indexes for the
first matrix are given in the third input.

```
  D = A (sparse 32x32) * B (32x32) + D (32x32)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is a sparse matrix in BF16 float format, consuming half the physical storage of a dense matrix with
same dimensions. Matrix B is a dense matrix in BF16 float format. Matrix D is single-precision float format and
is both the output and the accumulate input.

2 out of every 4 elements on the K axis of matrix A are zero. The sparse indexes are used to determine which 2
elements are zero.

Notes

NEG[1:0] and ABS[1:0] must be zero. CLAMP is not supported. Round toward nearest even semantics.

#### V_SMFMAC_I32_32X32X64_I8  (opcode 71)

Multiply the 32x64 sparse matrix in the first input by the 64x32 matrix in the second input and accumulate the
result into the 32x32 matrix stored in the destination registers using fused multiply add. Sparse indexes for the
first matrix are given in the third input.

```
  D = A (sparse 32x64) * B (64x32) + D (32x32)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is a sparse matrix in signed 8-bit integer format, consuming half the physical storage of a dense
matrix with same dimensions. Matrix B is a dense matrix in signed 8-bit integer format. Matrix D is signed 32-
bit integer format and is both the output and the accumulate input.

2 out of every 4 elements on the K axis of matrix A are zero. The sparse indexes are used to determine which 2
elements are zero.

Notes

NEG[1:0] and ABS[1:0] must be zero. CLAMP is not supported. Round toward nearest even semantics.

#### V_MFMA_F32_32X32X4_2B_F16  (opcode 72)

Multiply the 32x4 matrix in the first input by the 4x32 matrix in the second input and add the 32x32 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (32x4) * B (4x32) + C (32x32)
```

This instruction performs 2 matrix multiplies. Each operand contains 2 matrices back to back, and each matrix
has elements distributed across all lanes of the wave. Each matrix multiple is computed and the row-column
dot products are distributed across the vector ALU for higher performance. The result matrices are stored
back-to-back in the destination vector registers.

Matrices A and B are half-precision float format. Matrices C and D are single-precision float format.

Notes

This instruction performs 16 passes.

#### V_MFMA_F32_16X16X4_4B_F16  (opcode 73)

Multiply the 16x4 matrix in the first input by the 4x16 matrix in the second input and add the 16x16 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (16x4) * B (4x16) + C (16x16)
```

This instruction performs 4 matrix multiplies. Each operand contains 4 matrices back to back, and each matrix
has elements distributed across all lanes of the wave. Each matrix multiple is computed and the row-column

dot products are distributed across the vector ALU for higher performance. The result matrices are stored
back-to-back in the destination vector registers.

Matrices A and B are half-precision float format. Matrices C and D are single-precision float format.

Notes

This instruction performs 8 passes.

#### V_MFMA_F32_4X4X4_16B_F16  (opcode 74)

Multiply the 4x4 matrix in the first input by the 4x4 matrix in the second input and add the 4x4 matrix in the
third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (4x4) * B (4x4) + C (4x4)
```

This instruction performs 16 matrix multiplies. Each operand contains 16 matrices back to back, and each
matrix has elements distributed across all lanes of the wave. Each matrix multiple is computed and the row-
column dot products are distributed across the vector ALU for higher performance. The result matrices are
stored back-to-back in the destination vector registers.

Matrices A and B are half-precision float format. Matrices C and D are single-precision float format.

Notes

This instruction performs 2 passes.

#### V_SMFMAC_F32_32X32X64_BF8_BF8  (opcode 75)

Multiply the 32x64 sparse matrix in the first input by the 64x32 matrix in the second input and accumulate the
result into the 32x32 matrix stored in the destination registers using fused multiply add. Sparse indexes for the
first matrix are given in the third input.

```
  D = A (sparse 32x64) * B (64x32) + D (32x32)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is a sparse matrix in BF8 float format, consuming half the physical storage of a dense matrix with
same dimensions. Matrix B is a dense matrix in BF8 float format. Matrix D is single-precision float format and
is both the output and the accumulate input.

2 out of every 4 elements on the K axis of matrix A are zero. The sparse indexes are used to determine which 2
elements are zero.

Notes

NEG[1:0] and ABS[1:0] must be zero. CLAMP is not supported. Round toward nearest even semantics.

#### V_MFMA_F32_32X32X8_F16  (opcode 76)

Multiply the 32x8 matrix in the first input by the 8x32 matrix in the second input and add the 32x32 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (32x8) * B (8x32) + C (32x32)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrices A and B are half-precision float format. Matrices C and D are single-precision float format.

Notes

This instruction performs 8 passes.

#### V_MFMA_F32_16X16X16_F16  (opcode 77)

Multiply the 16x16 matrix in the first input by the 16x16 matrix in the second input and add the 16x16 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (16x16) * B (16x16) + C (16x16)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrices A and B are half-precision float format. Matrices C and D are single-precision float format.

Notes

This instruction performs 4 passes.

#### V_SMFMAC_F32_32X32X64_BF8_FP8  (opcode 78)

Multiply the 32x64 sparse matrix in the first input by the 64x32 matrix in the second input and accumulate the
result into the 32x32 matrix stored in the destination registers using fused multiply add. Sparse indexes for the
first matrix are given in the third input.

```
  D = A (sparse 32x64) * B (64x32) + D (32x32)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is a sparse matrix in BF8 float format, consuming half the physical storage of a dense matrix with
same dimensions. Matrix B is a dense matrix in FP8 float format. Matrix D is single-precision float format and
is both the output and the accumulate input.

2 out of every 4 elements on the K axis of matrix A are zero. The sparse indexes are used to determine which 2
elements are zero.

Notes

NEG[1:0] and ABS[1:0] must be zero. CLAMP is not supported. Round toward nearest even semantics.

#### V_SMFMAC_F32_32X32X64_FP8_BF8  (opcode 79)

Multiply the 32x64 sparse matrix in the first input by the 64x32 matrix in the second input and accumulate the
result into the 32x32 matrix stored in the destination registers using fused multiply add. Sparse indexes for the
first matrix are given in the third input.

```
  D = A (sparse 32x64) * B (64x32) + D (32x32)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is a sparse matrix in FP8 float format, consuming half the physical storage of a dense matrix with
same dimensions. Matrix B is a dense matrix in BF8 float format. Matrix D is single-precision float format and
is both the output and the accumulate input.

2 out of every 4 elements on the K axis of matrix A are zero. The sparse indexes are used to determine which 2
elements are zero.

Notes

NEG[1:0] and ABS[1:0] must be zero. CLAMP is not supported. Round toward nearest even semantics.

#### V_MFMA_I32_32X32X4_2B_I8  (opcode 80)

Multiply the 32x4 matrix in the first input by the 4x32 matrix in the second input and add the 32x32 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (32x4) * B (4x32) + C (32x32)
```

This instruction performs 2 matrix multiplies. Each operand contains 2 matrices back to back, and each matrix
has elements distributed across all lanes of the wave. Each matrix multiple is computed and the row-column
dot products are distributed across the vector ALU for higher performance. The result matrices are stored
back-to-back in the destination vector registers.

Matrices A and B are signed 8-bit integer format. Matrices C and D are signed 32-bit integer format.

Notes

This instruction performs 16 passes.

#### V_MFMA_I32_16X16X4_4B_I8  (opcode 81)

Multiply the 16x4 matrix in the first input by the 4x16 matrix in the second input and add the 16x16 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (16x4) * B (4x16) + C (16x16)
```

This instruction performs 4 matrix multiplies. Each operand contains 4 matrices back to back, and each matrix
has elements distributed across all lanes of the wave. Each matrix multiple is computed and the row-column
dot products are distributed across the vector ALU for higher performance. The result matrices are stored
back-to-back in the destination vector registers.

Matrices A and B are signed 8-bit integer format. Matrices C and D are signed 32-bit integer format.

Notes

This instruction performs 8 passes.

#### V_MFMA_I32_4X4X4_16B_I8  (opcode 82)

Multiply the 4x4 matrix in the first input by the 4x4 matrix in the second input and add the 4x4 matrix in the
third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (4x4) * B (4x4) + C (4x4)
```

This instruction performs 16 matrix multiplies. Each operand contains 16 matrices back to back, and each
matrix has elements distributed across all lanes of the wave. Each matrix multiple is computed and the row-
column dot products are distributed across the vector ALU for higher performance. The result matrices are
stored back-to-back in the destination vector registers.

Matrices A and B are signed 8-bit integer format. Matrices C and D are signed 32-bit integer format.

Notes

This instruction performs 2 passes.

#### V_SMFMAC_F32_32X32X64_FP8_FP8  (opcode 83)

Multiply the 32x64 sparse matrix in the first input by the 64x32 matrix in the second input and accumulate the
result into the 32x32 matrix stored in the destination registers using fused multiply add. Sparse indexes for the
first matrix are given in the third input.

```
  D = A (sparse 32x64) * B (64x32) + D (32x32)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is a sparse matrix in FP8 float format, consuming half the physical storage of a dense matrix with
same dimensions. Matrix B is a dense matrix in FP8 float format. Matrix D is single-precision float format and
is both the output and the accumulate input.

2 out of every 4 elements on the K axis of matrix A are zero. The sparse indexes are used to determine which 2
elements are zero.

Notes

NEG[1:0] and ABS[1:0] must be zero. CLAMP is not supported. Round toward nearest even semantics.

#### V_MFMA_F32_16X16X32_F16  (opcode 84)

Multiply the 16x32 matrix in the first input by the 32x16 matrix in the second input and add the 16x16 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (16x32) * B (32x16) + C (16x16)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrices A and B are half-precision float format. Matrices C and D are single-precision float format.

Notes

This instruction performs 4 passes.

#### V_MFMA_F32_32X32X16_F16  (opcode 85)

Multiply the 32x16 matrix in the first input by the 16x32 matrix in the second input and add the 32x32 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (32x16) * B (16x32) + C (32x32)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrices A and B are half-precision float format. Matrices C and D are single-precision float format.

Notes

This instruction performs 8 passes.

#### V_MFMA_I32_32X32X16_I8  (opcode 86)

Multiply the 32x16 matrix in the first input by the 16x32 matrix in the second input and add the 32x32 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (32x16) * B (16x32) + C (32x32)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrices A and B are signed 8-bit integer format. Matrices C and D are signed 32-bit integer format.

Notes

This instruction performs 8 passes.

#### V_MFMA_I32_16X16X32_I8  (opcode 87)

Multiply the 16x32 matrix in the first input by the 32x16 matrix in the second input and add the 16x16 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (16x32) * B (32x16) + C (16x16)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrices A and B are signed 8-bit integer format. Matrices C and D are signed 32-bit integer format.

Notes

This instruction performs 4 passes.

#### V_ACCVGPR_READ  (opcode 88)

Move 32 bits of data from an accumulator vector register into an architectural vector register.

#### V_ACCVGPR_WRITE  (opcode 89)

Move 32 bits of data from an architectural vector register into an accumulator vector register.

#### V_SMFMAC_F32_16X16X64_F16  (opcode 90)

Multiply the 16x64 sparse matrix in the first input by the 64x16 matrix in the second input and accumulate the
result into the 16x16 matrix stored in the destination registers using fused multiply add. Sparse indexes for the
first matrix are given in the third input.

```
  D = A (sparse 16x64) * B (64x16) + D (16x16)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is a sparse matrix in half-precision float format, consuming half the physical storage of a dense
matrix with same dimensions. Matrix B is a dense matrix in half-precision float format. Matrix D is single-
precision float format and is both the output and the accumulate input.

2 out of every 4 elements on the K axis of matrix A are zero. The sparse indexes are used to determine which 2
elements are zero.

Notes

This instruction performs 4 passes.

#### V_SMFMAC_F32_32X32X32_F16  (opcode 91)

Multiply the 32x32 sparse matrix in the first input by the 32x32 matrix in the second input and accumulate the
result into the 32x32 matrix stored in the destination registers using fused multiply add. Sparse indexes for the
first matrix are given in the third input.

```
  D = A (sparse 32x32) * B (32x32) + D (32x32)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is a sparse matrix in half-precision float format, consuming half the physical storage of a dense
matrix with same dimensions. Matrix B is a dense matrix in half-precision float format. Matrix D is single-
precision float format and is both the output and the accumulate input.

2 out of every 4 elements on the K axis of matrix A are zero. The sparse indexes are used to determine which 2
elements are zero.

Notes

This instruction performs 8 passes.

#### V_MFMA_F32_32X32X4_2B_BF16  (opcode 93)

Multiply the 32x4 matrix in the first input by the 4x32 matrix in the second input and add the 32x32 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (32x4) * B (4x32) + C (32x32)
```

This instruction performs 2 matrix multiplies. Each operand contains 2 matrices back to back, and each matrix
has elements distributed across all lanes of the wave. Each matrix multiple is computed and the row-column
dot products are distributed across the vector ALU for higher performance. The result matrices are stored
back-to-back in the destination vector registers.

Matrices A and B are BF16 float format. Matrices C and D are single-precision float format.

Notes

This instruction performs 16 passes.

#### V_MFMA_F32_16X16X4_4B_BF16  (opcode 94)

Multiply the 16x4 matrix in the first input by the 4x16 matrix in the second input and add the 16x16 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (16x4) * B (4x16) + C (16x16)
```

This instruction performs 4 matrix multiplies. Each operand contains 4 matrices back to back, and each matrix
has elements distributed across all lanes of the wave. Each matrix multiple is computed and the row-column

dot products are distributed across the vector ALU for higher performance. The result matrices are stored
back-to-back in the destination vector registers.

Matrices A and B are BF16 float format. Matrices C and D are single-precision float format.

Notes

This instruction performs 8 passes.

#### V_MFMA_F32_4X4X4_16B_BF16  (opcode 95)

Multiply the 4x4 matrix in the first input by the 4x4 matrix in the second input and add the 4x4 matrix in the
third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (4x4) * B (4x4) + C (4x4)
```

This instruction performs 16 matrix multiplies. Each operand contains 16 matrices back to back, and each
matrix has elements distributed across all lanes of the wave. Each matrix multiple is computed and the row-
column dot products are distributed across the vector ALU for higher performance. The result matrices are
stored back-to-back in the destination vector registers.

Matrices A and B are BF16 float format. Matrices C and D are single-precision float format.

Notes

This instruction performs 2 passes.

#### V_MFMA_F32_32X32X8_BF16  (opcode 96)

Multiply the 32x8 matrix in the first input by the 8x32 matrix in the second input and add the 32x32 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (32x8) * B (8x32) + C (32x32)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrices A and B are BF16 float format. Matrices C and D are single-precision float format.

Notes

This instruction performs 8 passes.

#### V_MFMA_F32_16X16X16_BF16  (opcode 97)

Multiply the 16x16 matrix in the first input by the 16x16 matrix in the second input and add the 16x16 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (16x16) * B (16x16) + C (16x16)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrices A and B are BF16 float format. Matrices C and D are single-precision float format.

Notes

This instruction performs 4 passes.

#### V_SMFMAC_F32_16X16X32_F16  (opcode 98)

Multiply the 16x32 sparse matrix in the first input by the 32x16 matrix in the second input and accumulate the
result into the 16x16 matrix stored in the destination registers using fused multiply add. Sparse indexes for the
first matrix are given in the third input.

```
  D = A (sparse 16x32) * B (32x16) + D (16x16)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is a sparse matrix in half-precision float format, consuming half the physical storage of a dense
matrix with same dimensions. Matrix B is a dense matrix in half-precision float format. Matrix D is single-
precision float format and is both the output and the accumulate input.

2 out of every 4 elements on the K axis of matrix A are zero. The sparse indexes are used to determine which 2
elements are zero.

Notes

This instruction performs 4 passes.

#### V_SMFMAC_F32_32X32X16_F16  (opcode 100)

Multiply the 32x16 sparse matrix in the first input by the 16x32 matrix in the second input and accumulate the
result into the 32x32 matrix stored in the destination registers using fused multiply add. Sparse indexes for the
first matrix are given in the third input.

```
  D = A (sparse 32x16) * B (16x32) + D (32x32)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is a sparse matrix in half-precision float format, consuming half the physical storage of a dense
matrix with same dimensions. Matrix B is a dense matrix in half-precision float format. Matrix D is single-
precision float format and is both the output and the accumulate input.

2 out of every 4 elements on the K axis of matrix A are zero. The sparse indexes are used to determine which 2
elements are zero.

Notes

This instruction performs 8 passes.

#### V_SMFMAC_F32_16X16X32_BF16  (opcode 102)

Multiply the 16x32 sparse matrix in the first input by the 32x16 matrix in the second input and accumulate the
result into the 16x16 matrix stored in the destination registers using fused multiply add. Sparse indexes for the
first matrix are given in the third input.

```
  D = A (sparse 16x32) * B (32x16) + D (16x16)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is a sparse matrix in BF16 float format, consuming half the physical storage of a dense matrix with
same dimensions. Matrix B is a dense matrix in BF16 float format. Matrix D is single-precision float format and
is both the output and the accumulate input.

2 out of every 4 elements on the K axis of matrix A are zero. The sparse indexes are used to determine which 2
elements are zero.

Notes

This instruction performs 4 passes.

#### V_SMFMAC_F32_32X32X16_BF16  (opcode 104)

Multiply the 32x16 sparse matrix in the first input by the 16x32 matrix in the second input and accumulate the
result into the 32x32 matrix stored in the destination registers using fused multiply add. Sparse indexes for the
first matrix are given in the third input.

```
  D = A (sparse 32x16) * B (16x32) + D (32x32)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is a sparse matrix in BF16 float format, consuming half the physical storage of a dense matrix with
same dimensions. Matrix B is a dense matrix in BF16 float format. Matrix D is single-precision float format and
is both the output and the accumulate input.

2 out of every 4 elements on the K axis of matrix A are zero. The sparse indexes are used to determine which 2
elements are zero.

Notes

This instruction performs 8 passes.

#### V_SMFMAC_I32_16X16X64_I8  (opcode 106)

Multiply the 16x64 sparse matrix in the first input by the 64x16 matrix in the second input and accumulate the
result into the 16x16 matrix stored in the destination registers using fused multiply add. Sparse indexes for the
first matrix are given in the third input.

```
  D = A (sparse 16x64) * B (64x16) + D (16x16)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is a sparse matrix in signed 8-bit integer format, consuming half the physical storage of a dense
matrix with same dimensions. Matrix B is a dense matrix in signed 8-bit integer format. Matrix D is signed 32-
bit integer format and is both the output and the accumulate input.

2 out of every 4 elements on the K axis of matrix A are zero. The sparse indexes are used to determine which 2
elements are zero.

Notes

This instruction performs 4 passes.

#### V_SMFMAC_I32_32X32X32_I8  (opcode 108)

Multiply the 32x32 sparse matrix in the first input by the 32x32 matrix in the second input and accumulate the
result into the 32x32 matrix stored in the destination registers using fused multiply add. Sparse indexes for the
first matrix are given in the third input.

```
  D = A (sparse 32x32) * B (32x32) + D (32x32)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is a sparse matrix in signed 8-bit integer format, consuming half the physical storage of a dense
matrix with same dimensions. Matrix B is a dense matrix in signed 8-bit integer format. Matrix D is signed 32-
bit integer format and is both the output and the accumulate input.

2 out of every 4 elements on the K axis of matrix A are zero. The sparse indexes are used to determine which 2
elements are zero.

Notes

This instruction performs 8 passes.

#### V_MFMA_F64_16X16X4_F64  (opcode 110)

Multiply the 16x4 matrix in the first input by the 4x16 matrix in the second input and add the 16x16 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (16x4) * B (4x16) + C (16x16)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrices A and B are double-precision float format. Matrices C and D are double-precision float format.

Notes

This instruction performs 16 passes.

#### V_MFMA_F64_4X4X4_4B_F64  (opcode 111)

Multiply the 4x4 matrix in the first input by the 4x4 matrix in the second input and add the 4x4 matrix in the
third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (4x4) * B (4x4) + C (4x4)
```

This instruction performs 4 matrix multiplies. Each operand contains 4 matrices back to back, and each matrix
has elements distributed across all lanes of the wave. Each matrix multiple is computed and the row-column
dot products are distributed across the vector ALU for higher performance. The result matrices are stored

back-to-back in the destination vector registers.

Matrices A and B are double-precision float format. Matrices C and D are double-precision float format.

Notes

This instruction performs 4 passes.

#### V_MFMA_F32_16X16X32_BF8_BF8  (opcode 112)

Multiply the 16x32 matrix in the first input by the 32x16 matrix in the second input and add the 16x16 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (16x32) * B (32x16) + C (16x16)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is BF8 float format. Matrix B is BF8 float format. Matrices C and D are single-precision float format.

Notes

This instruction performs 4 passes.

#### V_MFMA_F32_16X16X32_BF8_FP8  (opcode 113)

Multiply the 16x32 matrix in the first input by the 32x16 matrix in the second input and add the 16x16 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (16x32) * B (32x16) + C (16x16)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is BF8 float format. Matrix B is FP8 float format. Matrices C and D are single-precision float format.

Notes

This instruction performs 4 passes.

#### V_MFMA_F32_16X16X32_FP8_BF8  (opcode 114)

Multiply the 16x32 matrix in the first input by the 32x16 matrix in the second input and add the 16x16 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (16x32) * B (32x16) + C (16x16)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is FP8 float format. Matrix B is BF8 float format. Matrices C and D are single-precision float format.

Notes

This instruction performs 4 passes.

#### V_MFMA_F32_16X16X32_FP8_FP8  (opcode 115)

Multiply the 16x32 matrix in the first input by the 32x16 matrix in the second input and add the 16x16 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (16x32) * B (32x16) + C (16x16)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is FP8 float format. Matrix B is FP8 float format. Matrices C and D are single-precision float format.

Notes

This instruction performs 4 passes.

#### V_MFMA_F32_32X32X16_BF8_BF8  (opcode 116)

Multiply the 32x16 matrix in the first input by the 16x32 matrix in the second input and add the 32x32 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (32x16) * B (16x32) + C (32x32)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is BF8 float format. Matrix B is BF8 float format. Matrices C and D are single-precision float format.

Notes

This instruction performs 8 passes.

#### V_MFMA_F32_32X32X16_BF8_FP8  (opcode 117)

Multiply the 32x16 matrix in the first input by the 16x32 matrix in the second input and add the 32x32 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (32x16) * B (16x32) + C (32x32)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is BF8 float format. Matrix B is FP8 float format. Matrices C and D are single-precision float format.

Notes

This instruction performs 8 passes.

#### V_MFMA_F32_32X32X16_FP8_BF8  (opcode 118)

Multiply the 32x16 matrix in the first input by the 16x32 matrix in the second input and add the 32x32 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (32x16) * B (16x32) + C (32x32)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is FP8 float format. Matrix B is BF8 float format. Matrices C and D are single-precision float format.

Notes

This instruction performs 8 passes.

#### V_MFMA_F32_32X32X16_FP8_FP8  (opcode 119)

Multiply the 32x16 matrix in the first input by the 16x32 matrix in the second input and add the 32x32 matrix in
the third input using fused multiply add. Store the resulting matrix into vector registers.

```
  D = A (32x16) * B (16x32) + C (32x32)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is FP8 float format. Matrix B is FP8 float format. Matrices C and D are single-precision float format.

Notes

This instruction performs 8 passes.

#### V_SMFMAC_F32_16X16X64_BF8_BF8  (opcode 120)

Multiply the 16x64 sparse matrix in the first input by the 64x16 matrix in the second input and accumulate the
result into the 16x16 matrix stored in the destination registers using fused multiply add. Sparse indexes for the
first matrix are given in the third input.

```
  D = A (sparse 16x64) * B (64x16) + D (16x16)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is a sparse matrix in BF8 float format, consuming half the physical storage of a dense matrix with
same dimensions. Matrix B is a dense matrix in BF8 float format. Matrix D is single-precision float format and
is both the output and the accumulate input.

2 out of every 4 elements on the K axis of matrix A are zero. The sparse indexes are used to determine which 2
elements are zero.

Notes

This instruction performs 4 passes.

#### V_SMFMAC_F32_16X16X64_BF8_FP8  (opcode 121)

Multiply the 16x64 sparse matrix in the first input by the 64x16 matrix in the second input and accumulate the
result into the 16x16 matrix stored in the destination registers using fused multiply add. Sparse indexes for the
first matrix are given in the third input.

```
  D = A (sparse 16x64) * B (64x16) + D (16x16)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single

matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is a sparse matrix in BF8 float format, consuming half the physical storage of a dense matrix with
same dimensions. Matrix B is a dense matrix in FP8 float format. Matrix D is single-precision float format and
is both the output and the accumulate input.

2 out of every 4 elements on the K axis of matrix A are zero. The sparse indexes are used to determine which 2
elements are zero.

Notes

This instruction performs 4 passes.

#### V_SMFMAC_F32_16X16X64_FP8_BF8  (opcode 122)

Multiply the 16x64 sparse matrix in the first input by the 64x16 matrix in the second input and accumulate the
result into the 16x16 matrix stored in the destination registers using fused multiply add. Sparse indexes for the
first matrix are given in the third input.

```
  D = A (sparse 16x64) * B (64x16) + D (16x16)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is a sparse matrix in FP8 float format, consuming half the physical storage of a dense matrix with
same dimensions. Matrix B is a dense matrix in BF8 float format. Matrix D is single-precision float format and
is both the output and the accumulate input.

2 out of every 4 elements on the K axis of matrix A are zero. The sparse indexes are used to determine which 2
elements are zero.

Notes

This instruction performs 4 passes.

#### V_SMFMAC_F32_16X16X64_FP8_FP8  (opcode 123)

Multiply the 16x64 sparse matrix in the first input by the 64x16 matrix in the second input and accumulate the
result into the 16x16 matrix stored in the destination registers using fused multiply add. Sparse indexes for the
first matrix are given in the third input.

```
  D = A (sparse 16x64) * B (64x16) + D (16x16)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single

matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is a sparse matrix in FP8 float format, consuming half the physical storage of a dense matrix with
same dimensions. Matrix B is a dense matrix in FP8 float format. Matrix D is single-precision float format and
is both the output and the accumulate input.

2 out of every 4 elements on the K axis of matrix A are zero. The sparse indexes are used to determine which 2
elements are zero.

Notes

This instruction performs 4 passes.

#### V_SMFMAC_F32_32X32X32_BF8_BF8  (opcode 124)

Multiply the 32x32 sparse matrix in the first input by the 32x32 matrix in the second input and accumulate the
result into the 32x32 matrix stored in the destination registers using fused multiply add. Sparse indexes for the
first matrix are given in the third input.

```
  D = A (sparse 32x32) * B (32x32) + D (32x32)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is a sparse matrix in BF8 float format, consuming half the physical storage of a dense matrix with
same dimensions. Matrix B is a dense matrix in BF8 float format. Matrix D is single-precision float format and
is both the output and the accumulate input.

2 out of every 4 elements on the K axis of matrix A are zero. The sparse indexes are used to determine which 2
elements are zero.

Notes

This instruction performs 8 passes.

#### V_SMFMAC_F32_32X32X32_BF8_FP8  (opcode 125)

Multiply the 32x32 sparse matrix in the first input by the 32x32 matrix in the second input and accumulate the
result into the 32x32 matrix stored in the destination registers using fused multiply add. Sparse indexes for the
first matrix are given in the third input.

```
  D = A (sparse 32x32) * B (32x32) + D (32x32)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single

matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is a sparse matrix in BF8 float format, consuming half the physical storage of a dense matrix with
same dimensions. Matrix B is a dense matrix in FP8 float format. Matrix D is single-precision float format and
is both the output and the accumulate input.

2 out of every 4 elements on the K axis of matrix A are zero. The sparse indexes are used to determine which 2
elements are zero.

Notes

This instruction performs 8 passes.

#### V_SMFMAC_F32_32X32X32_FP8_BF8  (opcode 126)

Multiply the 32x32 sparse matrix in the first input by the 32x32 matrix in the second input and accumulate the
result into the 32x32 matrix stored in the destination registers using fused multiply add. Sparse indexes for the
first matrix are given in the third input.

```
  D = A (sparse 32x32) * B (32x32) + D (32x32)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single
matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is a sparse matrix in FP8 float format, consuming half the physical storage of a dense matrix with
same dimensions. Matrix B is a dense matrix in BF8 float format. Matrix D is single-precision float format and
is both the output and the accumulate input.

2 out of every 4 elements on the K axis of matrix A are zero. The sparse indexes are used to determine which 2
elements are zero.

Notes

This instruction performs 8 passes.

#### V_SMFMAC_F32_32X32X32_FP8_FP8  (opcode 127)

Multiply the 32x32 sparse matrix in the first input by the 32x32 matrix in the second input and accumulate the
result into the 32x32 matrix stored in the destination registers using fused multiply add. Sparse indexes for the
first matrix are given in the third input.

```
  D = A (sparse 32x32) * B (32x32) + D (32x32)
```

Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single

matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher
performance.

Matrix A is a sparse matrix in FP8 float format, consuming half the physical storage of a dense matrix with
same dimensions. Matrix B is a dense matrix in FP8 float format. Matrix D is single-precision float format and
is both the output and the accumulate input.

2 out of every 4 elements on the K axis of matrix A are zero. The sparse indexes are used to determine which 2
elements are zero.

Notes

This instruction performs 8 passes.
