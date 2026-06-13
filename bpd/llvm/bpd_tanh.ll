// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
; BPD tanh: calls scalar tanhf from libm — matches PyTorch exactly
define void @bpd_tanh(i32 %n, float* %out, float* %in) {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop, label %exit

loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %idx = sext i32 %i to i64
  %inptr = getelementptr float, float* %in, i64 %idx
  %x = load float, float* %inptr, align 4
  %result = call float @tanhf(float %x)
  %outptr = getelementptr float, float* %out, i64 %idx
  store float %result, float* %outptr, align 4
  %next = add i32 %i, 1
  %done = icmp sge i32 %next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

declare float @tanhf(float)
