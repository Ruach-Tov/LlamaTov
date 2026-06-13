// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
declare <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float>, <4 x float>)
declare <4 x float> @llvm.exp.v4f32(<4 x float>)
declare <4 x float> @llvm.log.v4f32(<4 x float>)
declare <4 x float> @llvm.fabs.v4f32(<4 x float>)
declare <4 x float> @llvm.maxnum.v4f32(<4 x float>, <4 x float>)

; ============================================================
; mse_loss: mean((pred - target)^2)
; ============================================================
define float @bpd_mse_loss(i32 %n, ptr %pred, ptr %target) {
entry:
  %np = and i32 %n, -4
  %has = icmp sgt i32 %np, 0
  br i1 %has, label %loop, label %done
loop:
  %i = phi i32 [ 0, %entry ], [ %i_next, %loop ]
  %acc = phi <4 x float> [ zeroinitializer, %entry ], [ %acc_next, %loop ]
  %idx = sext i32 %i to i64
  %pg = getelementptr float, ptr %pred, i64 %idx
  %tg = getelementptr float, ptr %target, i64 %idx
  %pv = load <4 x float>, ptr %pg, align 4
  %tv = load <4 x float>, ptr %tg, align 4
  %diff = fsub <4 x float> %pv, %tv
  %sq = fmul <4 x float> %diff, %diff
  %acc_next = fadd <4 x float> %acc, %sq
  %i_next = add i32 %i, 4
  %again = icmp slt i32 %i_next, %np
  br i1 %again, label %loop, label %reduce
reduce:
  %h1 = call <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float> %acc_next, <4 x float> %acc_next)
  %h2 = call <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float> %h1, <4 x float> %h1)
  %sum = extractelement <4 x float> %h2, i64 0
  %nf = sitofp i32 %n to float
  %mean = fdiv float %sum, %nf
  ret float %mean
done:
  ret float 0.0
}

; ============================================================
; cross_entropy_loss: -mean(target * log(pred))
; ============================================================
define float @bpd_cross_entropy_loss(i32 %n, ptr %pred, ptr %target) {
entry:
  %np = and i32 %n, -4
  %has = icmp sgt i32 %np, 0
  br i1 %has, label %loop, label %done
loop:
  %i = phi i32 [ 0, %entry ], [ %i_next, %loop ]
  %acc = phi <4 x float> [ zeroinitializer, %entry ], [ %acc_next, %loop ]
  %idx = sext i32 %i to i64
  %pg = getelementptr float, ptr %pred, i64 %idx
  %tg = getelementptr float, ptr %target, i64 %idx
  %pv = load <4 x float>, ptr %pg, align 4
  %tv = load <4 x float>, ptr %tg, align 4
  %logp = call <4 x float> @llvm.log.v4f32(<4 x float> %pv)
  %tlogp = fmul <4 x float> %tv, %logp
  %acc_next = fadd <4 x float> %acc, %tlogp
  %i_next = add i32 %i, 4
  %again = icmp slt i32 %i_next, %np
  br i1 %again, label %loop, label %reduce
reduce:
  %h1 = call <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float> %acc_next, <4 x float> %acc_next)
  %h2 = call <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float> %h1, <4 x float> %h1)
  %sum = extractelement <4 x float> %h2, i64 0
  %nf = sitofp i32 %n to float
  %neg_mean = fdiv float %sum, %nf
  %loss = fsub float 0.0, %neg_mean
  ret float %loss
done:
  ret float 0.0
}

; ============================================================
; hinge_loss: mean(max(0, 1 - target * pred))
; ============================================================
define float @bpd_hinge_loss(i32 %n, ptr %pred, ptr %target) {
entry:
  %np = and i32 %n, -4
  %has = icmp sgt i32 %np, 0
  br i1 %has, label %loop, label %done
loop:
  %i = phi i32 [ 0, %entry ], [ %i_next, %loop ]
  %acc = phi <4 x float> [ zeroinitializer, %entry ], [ %acc_next, %loop ]
  %idx = sext i32 %i to i64
  %pg = getelementptr float, ptr %pred, i64 %idx
  %tg = getelementptr float, ptr %target, i64 %idx
  %pv = load <4 x float>, ptr %pg, align 4
  %tv = load <4 x float>, ptr %tg, align 4
  %tp = fmul <4 x float> %tv, %pv
  %ones = insertelement <4 x float> insertelement(<4 x float> insertelement(<4 x float> insertelement(<4 x float> undef, float 1.0, i32 0), float 1.0, i32 1), float 1.0, i32 2), float 1.0, i32 3
  %margin = fsub <4 x float> %ones, %tp
  
  %hinge = call <4 x float> @llvm.maxnum.v4f32(<4 x float> %margin, <4 x float> zeroinitializer)
  %acc_next = fadd <4 x float> %acc, %hinge
  %i_next = add i32 %i, 4
  %again = icmp slt i32 %i_next, %np
  br i1 %again, label %loop, label %reduce
reduce:
  %h1 = call <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float> %acc_next, <4 x float> %acc_next)
  %h2 = call <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float> %h1, <4 x float> %h1)
  %sum = extractelement <4 x float> %h2, i64 0
  %nf = sitofp i32 %n to float
  %mean = fdiv float %sum, %nf
  ret float %mean
done:
  ret float 0.0
}

; ============================================================
; kl_div_loss: mean(target * (log(target) - log(pred)))
; ============================================================
define float @bpd_kl_div_loss(i32 %n, ptr %pred, ptr %target) {
entry:
  %np = and i32 %n, -4
  %has = icmp sgt i32 %np, 0
  br i1 %has, label %loop, label %done
loop:
  %i = phi i32 [ 0, %entry ], [ %i_next, %loop ]
  %acc = phi <4 x float> [ zeroinitializer, %entry ], [ %acc_next, %loop ]
  %idx = sext i32 %i to i64
  %pg = getelementptr float, ptr %pred, i64 %idx
  %tg = getelementptr float, ptr %target, i64 %idx
  %pv = load <4 x float>, ptr %pg, align 4
  %tv = load <4 x float>, ptr %tg, align 4
  %logp = call <4 x float> @llvm.log.v4f32(<4 x float> %pv)
  %logt = call <4 x float> @llvm.log.v4f32(<4 x float> %tv)
  %diff = fsub <4 x float> %logt, %logp
  %kl = fmul <4 x float> %tv, %diff
  %acc_next = fadd <4 x float> %acc, %kl
  %i_next = add i32 %i, 4
  %again = icmp slt i32 %i_next, %np
  br i1 %again, label %loop, label %reduce
reduce:
  %h1 = call <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float> %acc_next, <4 x float> %acc_next)
  %h2 = call <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float> %h1, <4 x float> %h1)
  %sum = extractelement <4 x float> %h2, i64 0
  %nf = sitofp i32 %n to float
  %mean = fdiv float %sum, %nf
  ret float %mean
done:
  ret float 0.0
}
