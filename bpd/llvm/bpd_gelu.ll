// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
; BPD gelu: x * 0.5 * (1 + erf(x * M_SQRT1_2))
; A&S 7.1.26 erf polynomial with CORRECT float32 constants
; Constants verified against PyTorch vec256_float.h
define void @bpd_gelu(i32 %n, float* %out, float* %in) {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop, label %exit

loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %idx = sext i32 %i to i64
  %inptr = getelementptr float, float* %in, i64 %idx
  %x = load float, float* %inptr, align 4

  ; erf_arg = x * M_SQRT1_2
  %erf_arg = fmul float %x, 0x3FE6A09E60000000

  ; abs and sign
  %bits = bitcast float %erf_arg to i32
  %abs_bits = and i32 %bits, 2147483647
  %abs_val = bitcast i32 %abs_bits to float
  %sign_bits = and i32 %bits, -2147483648

  ; t = 1.0 / (p * |x| + 1.0)  p = 0.3275911
  %pt = fmul float %abs_val, 0x3FD4F740A0000000
  %pt1 = fadd float %pt, 1.0
  %t = fdiv float 1.0, %pt1

  ; Horner: ((((p5*t + p4)*t + p3)*t + p2)*t + p1) * t
  ; p5 = 1.061405429
  %h1 = fmul float 0x3FF0FB8440000000, %t
  ; + p4 = -1.453152027
  %h2 = fadd float %h1, 0xBFF7401C60000000
  %h3 = fmul float %h2, %t
  ; + p3 = 1.421413741
  %h4 = fadd float %h3, 0x3FF6BE1C60000000
  %h5 = fmul float %h4, %t
  ; + p2 = -0.284496736
  %h6 = fadd float %h5, 0xBFD23531C0000000
  %h7 = fmul float %h6, %t
  ; + p1 = 0.254829592
  %h8 = fadd float %h7, 0x3FD04F20C0000000
  %r = fmul float %h8, %t

  ; exp(-erf_arg^2)
  %x2 = fmul float %erf_arg, %erf_arg
  %negx2 = fsub float -0.0, %x2
  %expval = call float @expf(float %negx2)

  ; erf_abs = 1 - r * expval
  %rexp = fmul float %r, %expval
  %erf_abs = fsub float 1.0, %rexp

  ; apply sign
  %erf_bits = bitcast float %erf_abs to i32
  %signed_bits = xor i32 %erf_bits, %sign_bits
  %erf_result = bitcast i32 %signed_bits to float

  ; gelu = x * 0.5 * (1 + erf)
  %half_x = fmul float %x, 5.000000e-01
  %one_plus_erf = fadd float %erf_result, 1.0
  %result = fmul float %half_x, %one_plus_erf

  %outptr = getelementptr float, float* %out, i64 %idx
  store float %result, float* %outptr, align 4
  %next = add i32 %i, 1
  %done = icmp sge i32 %next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

declare float @expf(float)
