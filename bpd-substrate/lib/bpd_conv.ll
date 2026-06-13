// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
; ============================================================
; im2col_1d: rearrange input for convolution
; void im2col_1d(i32 n_out, i32 kernel, i32 stride, ptr col, ptr src)
; col layout: [n_out × kernel] row-major
; ============================================================
define void @bpd_im2col_1d(i32 %n_out, i32 %kernel, i32 %stride, ptr %col, ptr %src) {
entry:
  %has = icmp sgt i32 %n_out, 0
  br i1 %has, label %outer, label %done

outer:
  %oi = phi i32 [ 0, %entry ], [ %oi_next, %outer_end ]
  %base = mul i32 %oi, %stride
  %col_row = mul i32 %oi, %kernel
  br label %inner

inner:
  %ki = phi i32 [ 0, %outer ], [ %ki_next, %inner ]
  %src_idx = add i32 %base, %ki
  %si64 = sext i32 %src_idx to i64
  %sg = getelementptr float, ptr %src, i64 %si64
  %val = load float, ptr %sg, align 4
  %col_idx = add i32 %col_row, %ki
  %ci64 = sext i32 %col_idx to i64
  %cg = getelementptr float, ptr %col, i64 %ci64
  store float %val, ptr %cg, align 4
  %ki_next = add i32 %ki, 1
  %ki_done = icmp sge i32 %ki_next, %kernel
  br i1 %ki_done, label %outer_end, label %inner

outer_end:
  %oi_next = add i32 %oi, 1
  %oi_done = icmp sge i32 %oi_next, %n_out
  br i1 %oi_done, label %done, label %outer

done:
  ret void
}

; ============================================================
; conv1d: im2col then dot product each output with kernel weights
; void conv1d(i32 n_out, i32 kernel, i32 stride, ptr dst, ptr src, ptr weights)
; ============================================================
define void @bpd_conv1d(i32 %n_out, i32 %kernel, i32 %stride, ptr %dst, ptr %src, ptr %weights) {
entry:
  %has = icmp sgt i32 %n_out, 0
  br i1 %has, label %outer, label %done

outer:
  %oi = phi i32 [ 0, %entry ], [ %oi_next, %outer_end ]
  %base = mul i32 %oi, %stride
  br label %inner

inner:
  %ki = phi i32 [ 0, %outer ], [ %ki_next, %inner ]
  %acc = phi float [ 0.0, %outer ], [ %acc_next, %inner ]
  %src_idx = add i32 %base, %ki
  %si64 = sext i32 %src_idx to i64
  %sg = getelementptr float, ptr %src, i64 %si64
  %sv = load float, ptr %sg, align 4
  %ki64 = sext i32 %ki to i64
  %wg = getelementptr float, ptr %weights, i64 %ki64
  %wv = load float, ptr %wg, align 4
  %prod = fmul float %sv, %wv
  %acc_next = fadd float %acc, %prod
  %ki_next = add i32 %ki, 1
  %ki_done = icmp sge i32 %ki_next, %kernel
  br i1 %ki_done, label %outer_end, label %inner

outer_end:
  %oi64 = sext i32 %oi to i64
  %dg = getelementptr float, ptr %dst, i64 %oi64
  store float %acc_next, ptr %dg, align 4
  %oi_next = add i32 %oi, 1
  %oi_done = icmp sge i32 %oi_next, %n_out
  br i1 %oi_done, label %done, label %outer

done:
  ret void
}
