// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
declare <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float>, <4 x float>)

define void @bpd_sum_sse3(i32 %n, ptr %s, ptr %x) {
entry:
  %np = and i32 %n, -32
  %has = icmp sgt i32 %np, 0
  br i1 %has, label %loop, label %done

loop:
  %i = phi i32 [ 0, %entry ], [ %i_next, %loop ]
  %s0 = phi <4 x float> [ zeroinitializer, %entry ], [ %a0, %loop ]
  %s1 = phi <4 x float> [ zeroinitializer, %entry ], [ %a1, %loop ]
  %s2 = phi <4 x float> [ zeroinitializer, %entry ], [ %a2, %loop ]
  %s3 = phi <4 x float> [ zeroinitializer, %entry ], [ %a3, %loop ]
  %s4 = phi <4 x float> [ zeroinitializer, %entry ], [ %a4, %loop ]
  %s5 = phi <4 x float> [ zeroinitializer, %entry ], [ %a5, %loop ]
  %s6 = phi <4 x float> [ zeroinitializer, %entry ], [ %a6, %loop ]
  %s7 = phi <4 x float> [ zeroinitializer, %entry ], [ %a7, %loop ]

  %i0 = sext i32 %i to i64
  %g0 = getelementptr float, ptr %x, i64 %i0
  %v0 = load <4 x float>, ptr %g0, align 4
  %a0 = fadd <4 x float> %s0, %v0

  %off1 = add i32 %i, 4
  %i1 = sext i32 %off1 to i64
  %g1 = getelementptr float, ptr %x, i64 %i1
  %v1 = load <4 x float>, ptr %g1, align 4
  %a1 = fadd <4 x float> %s1, %v1

  %off2 = add i32 %i, 8
  %i2 = sext i32 %off2 to i64
  %g2 = getelementptr float, ptr %x, i64 %i2
  %v2 = load <4 x float>, ptr %g2, align 4
  %a2 = fadd <4 x float> %s2, %v2

  %off3 = add i32 %i, 12
  %i3 = sext i32 %off3 to i64
  %g3 = getelementptr float, ptr %x, i64 %i3
  %v3 = load <4 x float>, ptr %g3, align 4
  %a3 = fadd <4 x float> %s3, %v3

  %off4 = add i32 %i, 16
  %i4 = sext i32 %off4 to i64
  %g4 = getelementptr float, ptr %x, i64 %i4
  %v4 = load <4 x float>, ptr %g4, align 4
  %a4 = fadd <4 x float> %s4, %v4

  %off5 = add i32 %i, 20
  %i5 = sext i32 %off5 to i64
  %g5 = getelementptr float, ptr %x, i64 %i5
  %v5 = load <4 x float>, ptr %g5, align 4
  %a5 = fadd <4 x float> %s5, %v5

  %off6 = add i32 %i, 24
  %i6 = sext i32 %off6 to i64
  %g6 = getelementptr float, ptr %x, i64 %i6
  %v6 = load <4 x float>, ptr %g6, align 4
  %a6 = fadd <4 x float> %s6, %v6

  %off7 = add i32 %i, 28
  %i7 = sext i32 %off7 to i64
  %g7 = getelementptr float, ptr %x, i64 %i7
  %v7 = load <4 x float>, ptr %g7, align 4
  %a7 = fadd <4 x float> %s7, %v7

  %i_next = add i32 %i, 32
  %again = icmp slt i32 %i_next, %np
  br i1 %again, label %loop, label %reduce

reduce:
  %r0 = fadd <4 x float> %a0, %a4
  %r1 = fadd <4 x float> %a1, %a5
  %r2 = fadd <4 x float> %a2, %a6
  %r3 = fadd <4 x float> %a3, %a7
  %r4 = fadd <4 x float> %r0, %r2
  %r5 = fadd <4 x float> %r1, %r3
  %r6 = fadd <4 x float> %r4, %r5
  %h1 = call <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float> %r6, <4 x float> %r6)
  %h2 = call <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float> %h1, <4 x float> %h1)
  %sum = extractelement <4 x float> %h2, i64 0
  store float %sum, ptr %s, align 4
  ret void

done:
  store float 0.0, ptr %s, align 4
  ret void
}
