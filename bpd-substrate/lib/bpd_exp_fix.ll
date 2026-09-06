declare float @expf(float)
declare float @logf(float)
declare float @log1pf(float)

; elu_fixed: x > 0 ? x : exp(x) - 1  (calls expf directly)
define void @bpd_elu_fixed(i32 %n, ptr %dst, ptr %src) {
entry:
  %has = icmp sgt i32 %n, 0
  br i1 %has, label %loop, label %done
loop:
  %i = phi i32 [ 0, %entry ], [ %i_next, %loop ]
  %idx = sext i32 %i to i64
  %sg = getelementptr float, ptr %src, i64 %idx
  %x = load float, ptr %sg, align 4
  %ex = call float @expf(float %x)
  %em1 = fsub float %ex, 1.0
  %pos = fcmp ogt float %x, 0.0
  %result = select i1 %pos, float %x, float %em1
  %dg = getelementptr float, ptr %dst, i64 %idx
  store float %result, ptr %dg, align 4
  %i_next = add i32 %i, 1
  %again = icmp slt i32 %i_next, %n
  br i1 %again, label %loop, label %done
done:
  ret void
}

; softplus_fixed: log(1 + exp(x))  (calls expf + logf directly)
define void @bpd_softplus_fixed(i32 %n, ptr %dst, ptr %src) {
entry:
  %has = icmp sgt i32 %n, 0
  br i1 %has, label %loop, label %done
loop:
  %i = phi i32 [ 0, %entry ], [ %i_next, %loop ]
  %idx = sext i32 %i to i64
  %sg = getelementptr float, ptr %src, i64 %idx
  %x = load float, ptr %sg, align 4
  %ex = call float @expf(float %x)
  %result = call float @log1pf(float %ex)
  %dg = getelementptr float, ptr %dst, i64 %idx
  store float %result, ptr %dg, align 4
  %i_next = add i32 %i, 1
  %again = icmp slt i32 %i_next, %n
  br i1 %again, label %loop, label %done
done:
  ret void
}

; selu_fixed: scale * (x > 0 ? x : alpha*(exp(x)-1))
; scale = 1.0507009873554804934193349852946
; alpha = 1.6732632423543772848170429916717
define void @bpd_selu_fixed(i32 %n, ptr %dst, ptr %src) {
entry:
  %has = icmp sgt i32 %n, 0
  br i1 %has, label %loop, label %done
loop:
  %i = phi i32 [ 0, %entry ], [ %i_next, %loop ]
  %idx = sext i32 %i to i64
  %sg = getelementptr float, ptr %src, i64 %idx
  %x = load float, ptr %sg, align 4
  %ex = call float @expf(float %x)
  %em1 = fsub float %ex, 1.0
  %aem1 = fmul float %em1, 0x3FFAC5AFA0000000
  %pos = fcmp ogt float %x, 0.0
  %branch = select i1 %pos, float %x, float %aem1
  %result = fmul float %branch, 0x3FF0CFABE0000000
  %dg = getelementptr float, ptr %dst, i64 %idx
  store float %result, ptr %dg, align 4
  %i_next = add i32 %i, 1
  %again = icmp slt i32 %i_next, %n
  br i1 %again, label %loop, label %done
done:
  ret void
}
