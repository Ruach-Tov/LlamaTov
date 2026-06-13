// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
define void @bpd_softplus(i32 %n, float* %out, float* %in) {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop, label %exit

loop:
  %i = phi i32 [ 0, %entry ], [ %next, %store ]
  %idx = sext i32 %i to i64
  %inptr = getelementptr float, float* %in, i64 %idx
  %x = load float, float* %inptr, align 4
  %above_threshold = fcmp ogt float %x, 2.000000e+01
  br i1 %above_threshold, label %passthrough, label %compute

compute:
  %expx = call float @expf(float %x)
  %logval = call float @log1pf(float %expx)
  br label %store

passthrough:
  br label %store

store:
  %result = phi float [ %logval, %compute ], [ %x, %passthrough ]
  %outptr = getelementptr float, float* %out, i64 %idx
  store float %result, float* %outptr, align 4
  %next = add i32 %i, 1
  %done = icmp sge i32 %next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

declare float @expf(float)
declare float @log1pf(float)
