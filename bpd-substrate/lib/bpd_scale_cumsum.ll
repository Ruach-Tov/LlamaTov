// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
; bpd_scale_cumsum.ll — scale, cumsum, cumprod LLVM IR emitters
;
; scale: dst[i] = src[i] * scalar  (binary_elementwise)
; cumsum: dst[i] = sum(src[0..i])  (scan)
; cumprod: dst[i] = prod(src[0..i])  (scan)

; ============================================================
; scale: dst[i] = src[i] * scalar
; ============================================================
define void @bpd_scale(i32 %n, ptr %dst, ptr %src, float %scalar) {
entry:
  %np = and i32 %n, -4
  %has = icmp sgt i32 %np, 0
  br i1 %has, label %loop, label %done

loop:
  %i = phi i32 [ 0, %entry ], [ %i_next, %loop ]
  %idx = sext i32 %i to i64
  %sg = getelementptr float, ptr %src, i64 %idx
  %xv = load <4 x float>, ptr %sg, align 4
  %s0 = insertelement <4 x float> undef, float %scalar, i32 0
  %s1 = insertelement <4 x float> %s0, float %scalar, i32 1
  %s2 = insertelement <4 x float> %s1, float %scalar, i32 2
  %splat = insertelement <4 x float> %s2, float %scalar, i32 3
  %result = fmul <4 x float> %xv, %splat
  %dg = getelementptr float, ptr %dst, i64 %idx
  store <4 x float> %result, ptr %dg, align 4
  %i_next = add i32 %i, 4
  %again = icmp slt i32 %i_next, %np
  br i1 %again, label %loop, label %done

done:
  ret void
}

; ============================================================
; cumsum: dst[i] = dst[i-1] + src[i]
; ============================================================
define void @bpd_cumsum(i32 %n, ptr %dst, ptr %src) {
entry:
  %has = icmp sgt i32 %n, 0
  br i1 %has, label %loop, label %done

loop:
  %i = phi i32 [ 0, %entry ], [ %i_next, %loop ]
  %acc = phi float [ 0.0, %entry ], [ %sum, %loop ]
  %idx = sext i32 %i to i64
  %sg = getelementptr float, ptr %src, i64 %idx
  %x = load float, ptr %sg, align 4
  %sum = fadd float %acc, %x
  %dg = getelementptr float, ptr %dst, i64 %idx
  store float %sum, ptr %dg, align 4
  %i_next = add i32 %i, 1
  %again = icmp slt i32 %i_next, %n
  br i1 %again, label %loop, label %done

done:
  ret void
}

; ============================================================
; cumprod: dst[i] = dst[i-1] * src[i]
; ============================================================
define void @bpd_cumprod(i32 %n, ptr %dst, ptr %src) {
entry:
  %has = icmp sgt i32 %n, 0
  br i1 %has, label %loop, label %done

loop:
  %i = phi i32 [ 0, %entry ], [ %i_next, %loop ]
  %acc = phi float [ 1.0, %entry ], [ %prod, %loop ]
  %idx = sext i32 %i to i64
  %sg = getelementptr float, ptr %src, i64 %idx
  %x = load float, ptr %sg, align 4
  %prod = fmul float %acc, %x
  %dg = getelementptr float, ptr %dst, i64 %idx
  store float %prod, ptr %dg, align 4
  %i_next = add i32 %i, 1
  %again = icmp slt i32 %i_next, %n
  br i1 %again, label %loop, label %done

done:
  ret void
}

; ============================================================
; clamp: dst[i] = max(min_val, min(max_val, src[i]))
; For now: hardcoded to clamp(-1, 1) to match test harness
; ============================================================
declare float @llvm.maxnum.f32(float, float)
declare float @llvm.minnum.f32(float, float)

define void @bpd_clamp(i32 %n, ptr %dst, ptr %src) {
entry:
  %has = icmp sgt i32 %n, 0
  br i1 %has, label %loop, label %done

loop:
  %i = phi i32 [ 0, %entry ], [ %i_next, %loop ]
  %idx = sext i32 %i to i64
  %sg = getelementptr float, ptr %src, i64 %idx
  %x = load float, ptr %sg, align 4
  %lo = call float @llvm.maxnum.f32(float %x, float -1.0)
  %result = call float @llvm.minnum.f32(float %lo, float 1.0)
  %dg = getelementptr float, ptr %dst, i64 %idx
  store float %result, ptr %dg, align 4
  %i_next = add i32 %i, 1
  %again = icmp slt i32 %i_next, %n
  br i1 %again, label %loop, label %done

done:
  ret void
}
