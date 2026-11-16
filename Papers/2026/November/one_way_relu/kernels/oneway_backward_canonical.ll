; CANONICAL one-way-leaky backward — LLVM IR, emitted from op_expr (var2) by expr_ir.
; Semantics: grad_in = grad_out * ((x>0)?1 : (x<=0 && grad_out<0)?alpha : 0)
; Convention: x=0 DEAD (x<=0); -0.0<=0 IEEE-true -> dead; NaN x -> ogt false -> dead.
define void @oneway_backward(float* noalias %x, float* noalias %g, float* noalias %dst, i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [0, %entry], [%inext, %body]
  %cond = icmp slt i64 %i, %n
  br i1 %cond, label %body, label %done
body:
  %xp = getelementptr float, float* %x, i64 %i
  %v0 = load float, float* %xp, align 4
  %gp = getelementptr float, float* %g, i64 %i
  %v1 = load float, float* %gp, align 4
  %l0 = fadd float 0.0, 0.0
  %l1 = fcmp ogt float %v0, %l0
  %l2 = fadd float 0.0, 1.0
  %l3 = fadd float 0.0, 0.0
  %l4 = fcmp olt float %v1, %l3
  %l5 = fadd float 0.0, 0x3F847AE140000000
  %l6 = fadd float 0.0, 0.0
  %l7 = select i1 %l4, float %l5, float %l6
  %l8 = select i1 %l1, float %l2, float %l7
  %l9 = fmul float %v1, %l8
  %dp = getelementptr float, float* %dst, i64 %i
  store float %l9, float* %dp, align 4
  %inext = add i64 %i, 1
  br label %loop
done:
  ret void
}
