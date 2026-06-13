// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
declare <4 x float> @llvm.maxnum.v4f32(<4 x float>, <4 x float>)

; ============================================================
; max_pool_1d: for each output position, take max over kernel window
; void pool_max_1d(i32 n_out, i32 kernel, i32 stride, ptr dst, ptr src)
; ============================================================
define void @bpd_pool_max_1d(i32 %n_out, i32 %kernel, i32 %stride, ptr %dst, ptr %src) {
entry:
  %has = icmp sgt i32 %n_out, 0
  br i1 %has, label %outer, label %done

outer:
  %oi = phi i32 [ 0, %entry ], [ %oi_next, %outer_end ]
  %base = mul i32 %oi, %stride
  br label %inner

inner:
  %ki = phi i32 [ 0, %outer ], [ %ki_next, %inner ]
  %cur_max = phi float [ 0xFFF0000000000000, %outer ], [ %new_max, %inner ]
  %src_idx = add i32 %base, %ki
  %idx64 = sext i32 %src_idx to i64
  %sg = getelementptr float, ptr %src, i64 %idx64
  %val = load float, ptr %sg, align 4
  %cmp = fcmp ogt float %val, %cur_max
  %new_max = select i1 %cmp, float %val, float %cur_max
  %ki_next = add i32 %ki, 1
  %ki_done = icmp sge i32 %ki_next, %kernel
  br i1 %ki_done, label %outer_end, label %inner

outer_end:
  %oi64 = sext i32 %oi to i64
  %dg = getelementptr float, ptr %dst, i64 %oi64
  store float %new_max, ptr %dg, align 4
  %oi_next = add i32 %oi, 1
  %oi_done = icmp sge i32 %oi_next, %n_out
  br i1 %oi_done, label %done, label %outer

done:
  ret void
}

; ============================================================
; avg_pool_1d: for each output position, take mean over kernel window
; void pool_avg_1d(i32 n_out, i32 kernel, i32 stride, ptr dst, ptr src)
; ============================================================
define void @bpd_pool_avg_1d(i32 %n_out, i32 %kernel, i32 %stride, ptr %dst, ptr %src) {
entry:
  %kf = sitofp i32 %kernel to float
  %has = icmp sgt i32 %n_out, 0
  br i1 %has, label %outer, label %done

outer:
  %oi = phi i32 [ 0, %entry ], [ %oi_next, %outer_end ]
  %base = mul i32 %oi, %stride
  br label %inner

inner:
  %ki = phi i32 [ 0, %outer ], [ %ki_next, %inner ]
  %sum = phi float [ 0.0, %outer ], [ %new_sum, %inner ]
  %src_idx = add i32 %base, %ki
  %idx64 = sext i32 %src_idx to i64
  %sg = getelementptr float, ptr %src, i64 %idx64
  %val = load float, ptr %sg, align 4
  %new_sum = fadd float %sum, %val
  %ki_next = add i32 %ki, 1
  %ki_done = icmp sge i32 %ki_next, %kernel
  br i1 %ki_done, label %outer_end, label %inner

outer_end:
  %avg = fdiv float %new_sum, %kf
  %oi64 = sext i32 %oi to i64
  %dg = getelementptr float, ptr %dst, i64 %oi64
  store float %avg, ptr %dg, align 4
  %oi_next = add i32 %oi, 1
  %oi_done = icmp sge i32 %oi_next, %n_out
  br i1 %oi_done, label %done, label %outer

done:
  ret void
}
