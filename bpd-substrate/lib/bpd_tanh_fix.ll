// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
declare float @tanhf(float)

define void @bpd_tanh_fixed(i32 %n, ptr %dst, ptr %src) {
entry:
  %has = icmp sgt i32 %n, 0
  br i1 %has, label %loop, label %done

loop:
  %i = phi i32 [ 0, %entry ], [ %i_next, %loop ]
  %idx = sext i32 %i to i64
  %sg = getelementptr float, ptr %src, i64 %idx
  %x = load float, ptr %sg, align 4
  %th = call float @tanhf(float %x)
  %dg = getelementptr float, ptr %dst, i64 %idx
  store float %th, ptr %dg, align 4
  %i_next = add i32 %i, 1
  %again = icmp slt i32 %i_next, %n
  br i1 %again, label %loop, label %done

done:
  ret void
}

; gelu using tanhf: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
define void @bpd_gelu_fixed(i32 %n, ptr %dst, ptr %src) {
entry:
  %has = icmp sgt i32 %n, 0
  br i1 %has, label %loop, label %done

loop:
  %i = phi i32 [ 0, %entry ], [ %i_next, %loop ]
  %idx = sext i32 %i to i64
  %sg = getelementptr float, ptr %src, i64 %idx
  %x = load float, ptr %sg, align 4
  %x2 = fmul float %x, %x
  %x3 = fmul float %x2, %x
  %cx3 = fmul float %x3, 0x3fa6e4e260000000
  %inner = fadd float %x, %cx3
  %scaled = fmul float %inner, 0x3FE9884540000000
  %th = call float @tanhf(float %scaled)
  %oneth = fadd float 1.0, %th
  %hx = fmul float 0.5, %x
  %result = fmul float %hx, %oneth
  %dg = getelementptr float, ptr %dst, i64 %idx
  store float %result, ptr %dg, align 4
  %i_next = add i32 %i, 1
  %again = icmp slt i32 %i_next, %n
  br i1 %again, label %loop, label %done

done:
  ret void
}

; hardsigmoid: clamp(x * (1/6) + 0.5, 0, 1) — use multiply by 1/6 instead of divide
declare float @llvm.maxnum.f32(float, float)
declare float @llvm.minnum.f32(float, float)

define void @bpd_hardsigmoid_fixed(i32 %n, ptr %dst, ptr %src) {
entry:
  %has = icmp sgt i32 %n, 0
  br i1 %has, label %loop, label %done

loop:
  %i = phi i32 [ 0, %entry ], [ %i_next, %loop ]
  %idx = sext i32 %i to i64
  %sg = getelementptr float, ptr %src, i64 %idx
  %x = load float, ptr %sg, align 4
  %xp3 = fadd float %x, 3.0
  %xsh = fdiv float %xp3, 6.0
  %cl = call float @llvm.maxnum.f32(float %xsh, float 0.0)
  %result = call float @llvm.minnum.f32(float %cl, float 1.0)
  %dg = getelementptr float, ptr %dst, i64 %idx
  store float %result, ptr %dg, align 4
  %i_next = add i32 %i, 1
  %again = icmp slt i32 %i_next, %n
  br i1 %again, label %loop, label %done

done:
  ret void
}
