// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
; BPD hardsigmoid: clamp((x+3)/6, 0, 1)
; NOTE: must be (x+3)/6, NOT x/6+0.5 — order matters for float precision
define void @bpd_hardsigmoid(i32 %n, float* %out, float* %in) {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop, label %exit

loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %idx = sext i32 %i to i64
  %inptr = getelementptr float, float* %in, i64 %idx
  %x = load float, float* %inptr, align 4
  %plus3 = fadd float %x, 3.0
  %div6 = fdiv float %plus3, 6.0
  %clamped_low = call float @llvm.maxnum.f32(float %div6, float 0.0)
  %result = call float @llvm.minnum.f32(float %clamped_low, float 1.0)
  %outptr = getelementptr float, float* %out, i64 %idx
  store float %result, float* %outptr, align 4
  %next = add i32 %i, 1
  %done = icmp sge i32 %next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

declare float @llvm.maxnum.f32(float, float)
declare float @llvm.minnum.f32(float, float)
