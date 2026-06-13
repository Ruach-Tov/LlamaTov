// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
; q8_0_dot INTRINSIC (matches ggml AVX instruction-for-instruction)
target triple = "x86_64-unknown-linux-gnu"

declare <16 x i8> @llvm.x86.ssse3.psign.b.128(<16 x i8>, <16 x i8>)
declare <8 x i16> @llvm.x86.ssse3.pmadd.ub.sw.128(<16 x i8>, <16 x i8>)
declare <4 x i32> @llvm.x86.sse2.pmadd.wd(<8 x i16>, <8 x i16>)

define float @bpd_q8_0_dot(i8* noalias %wq, float* noalias %wd, i8* noalias %aq, float* noalias %ad, i32 %nb) {
entry:
  br label %hdr

hdr:
  %b = phi i32 [ 0, %entry ], [ %bn, %lat ]
  %acc = phi <8 x float> [ zeroinitializer, %entry ], [ %accn, %lat ]
  %b1 = add i32 %b, 1
  %c = icmp slt i32 %b1, %nb
  br i1 %c, label %body, label %tail

body:
  ; --- block b -> p_lo (4 i32 lanes) ---
  %b_off = mul i32 %b, 32
  %b_o64 = sext i32 %b_off to i64
  %b_wp0 = getelementptr i8, i8* %wq, i64 %b_o64
  %b_wp0b = getelementptr i8, i8* %b_wp0, i64 0
  %b_ap0 = getelementptr i8, i8* %aq, i64 %b_o64
  %b_ap0b = getelementptr i8, i8* %b_ap0, i64 0
  %b_wv0 = bitcast i8* %b_wp0b to <16 x i8>*
  %b_av0 = bitcast i8* %b_ap0b to <16 x i8>*
  %b_x0 = load <16 x i8>, <16 x i8>* %b_wv0, align 1
  %b_y0 = load <16 x i8>, <16 x i8>* %b_av0, align 1
  %b_ax0 = call <16 x i8> @llvm.x86.ssse3.psign.b.128(<16 x i8> %b_x0, <16 x i8> %b_x0)
  %b_sy0 = call <16 x i8> @llvm.x86.ssse3.psign.b.128(<16 x i8> %b_y0, <16 x i8> %b_x0)
  %b_p16_0 = call <8 x i16> @llvm.x86.ssse3.pmadd.ub.sw.128(<16 x i8> %b_ax0, <16 x i8> %b_sy0)
  %b_p32_0 = call <4 x i32> @llvm.x86.sse2.pmadd.wd(<8 x i16> %b_p16_0, <8 x i16> <i16 1,i16 1,i16 1,i16 1,i16 1,i16 1,i16 1,i16 1>)
  %b_wp1 = getelementptr i8, i8* %wq, i64 %b_o64
  %b_wp1b = getelementptr i8, i8* %b_wp1, i64 16
  %b_ap1 = getelementptr i8, i8* %aq, i64 %b_o64
  %b_ap1b = getelementptr i8, i8* %b_ap1, i64 16
  %b_wv1 = bitcast i8* %b_wp1b to <16 x i8>*
  %b_av1 = bitcast i8* %b_ap1b to <16 x i8>*
  %b_x1 = load <16 x i8>, <16 x i8>* %b_wv1, align 1
  %b_y1 = load <16 x i8>, <16 x i8>* %b_av1, align 1
  %b_ax1 = call <16 x i8> @llvm.x86.ssse3.psign.b.128(<16 x i8> %b_x1, <16 x i8> %b_x1)
  %b_sy1 = call <16 x i8> @llvm.x86.ssse3.psign.b.128(<16 x i8> %b_y1, <16 x i8> %b_x1)
  %b_p16_1 = call <8 x i16> @llvm.x86.ssse3.pmadd.ub.sw.128(<16 x i8> %b_ax1, <16 x i8> %b_sy1)
  %b_p32_1 = call <4 x i32> @llvm.x86.sse2.pmadd.wd(<8 x i16> %b_p16_1, <8 x i16> <i16 1,i16 1,i16 1,i16 1,i16 1,i16 1,i16 1,i16 1>)
  %p_lo = add <4 x i32> %b_p32_0, %b_p32_1
  ; --- block b1 -> p_hi (4 i32 lanes) ---
  %b1_off = mul i32 %b1, 32
  %b1_o64 = sext i32 %b1_off to i64
  %b1_wp0 = getelementptr i8, i8* %wq, i64 %b1_o64
  %b1_wp0b = getelementptr i8, i8* %b1_wp0, i64 0
  %b1_ap0 = getelementptr i8, i8* %aq, i64 %b1_o64
  %b1_ap0b = getelementptr i8, i8* %b1_ap0, i64 0
  %b1_wv0 = bitcast i8* %b1_wp0b to <16 x i8>*
  %b1_av0 = bitcast i8* %b1_ap0b to <16 x i8>*
  %b1_x0 = load <16 x i8>, <16 x i8>* %b1_wv0, align 1
  %b1_y0 = load <16 x i8>, <16 x i8>* %b1_av0, align 1
  %b1_ax0 = call <16 x i8> @llvm.x86.ssse3.psign.b.128(<16 x i8> %b1_x0, <16 x i8> %b1_x0)
  %b1_sy0 = call <16 x i8> @llvm.x86.ssse3.psign.b.128(<16 x i8> %b1_y0, <16 x i8> %b1_x0)
  %b1_p16_0 = call <8 x i16> @llvm.x86.ssse3.pmadd.ub.sw.128(<16 x i8> %b1_ax0, <16 x i8> %b1_sy0)
  %b1_p32_0 = call <4 x i32> @llvm.x86.sse2.pmadd.wd(<8 x i16> %b1_p16_0, <8 x i16> <i16 1,i16 1,i16 1,i16 1,i16 1,i16 1,i16 1,i16 1>)
  %b1_wp1 = getelementptr i8, i8* %wq, i64 %b1_o64
  %b1_wp1b = getelementptr i8, i8* %b1_wp1, i64 16
  %b1_ap1 = getelementptr i8, i8* %aq, i64 %b1_o64
  %b1_ap1b = getelementptr i8, i8* %b1_ap1, i64 16
  %b1_wv1 = bitcast i8* %b1_wp1b to <16 x i8>*
  %b1_av1 = bitcast i8* %b1_ap1b to <16 x i8>*
  %b1_x1 = load <16 x i8>, <16 x i8>* %b1_wv1, align 1
  %b1_y1 = load <16 x i8>, <16 x i8>* %b1_av1, align 1
  %b1_ax1 = call <16 x i8> @llvm.x86.ssse3.psign.b.128(<16 x i8> %b1_x1, <16 x i8> %b1_x1)
  %b1_sy1 = call <16 x i8> @llvm.x86.ssse3.psign.b.128(<16 x i8> %b1_y1, <16 x i8> %b1_x1)
  %b1_p16_1 = call <8 x i16> @llvm.x86.ssse3.pmadd.ub.sw.128(<16 x i8> %b1_ax1, <16 x i8> %b1_sy1)
  %b1_p32_1 = call <4 x i32> @llvm.x86.sse2.pmadd.wd(<8 x i16> %b1_p16_1, <8 x i16> <i16 1,i16 1,i16 1,i16 1,i16 1,i16 1,i16 1,i16 1>)
  %p_hi = add <4 x i32> %b1_p32_0, %b1_p32_1
  %pv32 = shufflevector <4 x i32> %p_lo, <4 x i32> %p_hi, <8 x i32> <i32 0,i32 1,i32 2,i32 3,i32 4,i32 5,i32 6,i32 7>
  %pf = sitofp <8 x i32> %pv32 to <8 x float>
  %bi = sext i32 %b to i64
  %b1i = sext i32 %b1 to i64
  %b_si = sext i32 %b to i64
  %wdp0 = getelementptr float, float* %wd, i64 %b_si
  %adp0 = getelementptr float, float* %ad, i64 %b_si
  %wdv0 = load float, float* %wdp0, align 4
  %adv0 = load float, float* %adp0, align 4
  %sc0 = fmul float %wdv0, %adv0
  %b1_si = sext i32 %b1 to i64
  %wdp1 = getelementptr float, float* %wd, i64 %b1_si
  %adp1 = getelementptr float, float* %ad, i64 %b1_si
  %wdv1 = load float, float* %wdp1, align 4
  %adv1 = load float, float* %adp1, align 4
  %sc1 = fmul float %wdv1, %adv1
  %d0 = insertelement <8 x float> undef, float %sc0, i32 0
  %d1 = insertelement <8 x float> %d0, float %sc0, i32 1
  %d2 = insertelement <8 x float> %d1, float %sc0, i32 2
  %d3 = insertelement <8 x float> %d2, float %sc0, i32 3
  %d4 = insertelement <8 x float> %d3, float %sc1, i32 4
  %d5 = insertelement <8 x float> %d4, float %sc1, i32 5
  %d6 = insertelement <8 x float> %d5, float %sc1, i32 6
  %deltas = insertelement <8 x float> %d6, float %sc1, i32 7
  %prod = fmul <8 x float> %deltas, %pf
  %accn = fadd <8 x float> %acc, %prod
  br label %lat

lat:
  %bn = add i32 %b, 2
  br label %hdr

tail:
  %lo = shufflevector <8 x float> %acc, <8 x float> undef, <4 x i32> <i32 0,i32 1,i32 2,i32 3>
  %hi = shufflevector <8 x float> %acc, <8 x float> undef, <4 x i32> <i32 4,i32 5,i32 6,i32 7>
  %r4 = fadd <4 x float> %lo, %hi
  %mhl = shufflevector <4 x float> %r4, <4 x float> undef, <4 x i32> <i32 2,i32 3,i32 2,i32 3>
  %r2 = fadd <4 x float> %r4, %mhl
  %e0 = extractelement <4 x float> %r2, i32 0
  %e1 = extractelement <4 x float> %r2, i32 1
  %s = fadd float %e0, %e1
  ret float %s
}
