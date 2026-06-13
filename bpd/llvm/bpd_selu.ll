// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
; BPD selu: lambda * (x if x>0, alpha * expm1(x) if x<=0)
; Precomputed: lambda*alpha = 1.7580993 (0x3FFC212CC0000000)
define void @bpd_selu(i32 %n, float* %out, float* %in) {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop, label %exit

loop:
  %i = phi i32 [ 0, %entry ], [ %next, %store ]
  %idx = sext i32 %i to i64
  %inptr = getelementptr float, float* %in, i64 %idx
  %x = load float, float* %inptr, align 4
  %positive = fcmp ogt float %x, 0.0
  br i1 %positive, label %pos, label %neg

pos:
  %pos_result = fmul float %x, 0x3FF0CFABE0000000
  br label %store

neg:
  %expm1_val = call float @expm1f(float %x)
  %neg_result = fmul float %expm1_val, 0x3FFC212CC0000000
  br label %store

store:
  %result = phi float [ %pos_result, %pos ], [ %neg_result, %neg ]
  %outptr = getelementptr float, float* %out, i64 %idx
  store float %result, float* %outptr, align 4
  %next = add i32 %i, 1
  %done = icmp sge i32 %next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

declare float @expm1f(float)
