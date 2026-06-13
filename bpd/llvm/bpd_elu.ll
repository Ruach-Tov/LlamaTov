// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
; BPD elu: elu(x, alpha=1.0) = x if x > 0, expm1(x) if x <= 0
; Uses expm1f for precision near zero — matches PyTorch MKL
define void @bpd_elu(i32 %n, float* %out, float* %in) {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop, label %exit

loop:
  %i = phi i32 [ 0, %entry ], [ %next, %store ]
  %idx = sext i32 %i to i64
  %inptr = getelementptr float, float* %in, i64 %idx
  %x = load float, float* %inptr, align 4
  %positive = fcmp ogt float %x, 0.0
  br i1 %positive, label %passthrough, label %compute

compute:
  %result_neg = call float @expm1f(float %x)
  br label %store

passthrough:
  br label %store

store:
  %result = phi float [ %x, %passthrough ], [ %result_neg, %compute ]
  %outptr = getelementptr float, float* %out, i64 %idx
  store float %result, float* %outptr, align 4
  %next = add i32 %i, 1
  %done = icmp sge i32 %next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

declare float @expm1f(float)
