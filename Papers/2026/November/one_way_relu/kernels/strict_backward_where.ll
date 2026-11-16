; CANONICAL strict_backward_where backward emitted from op_expr (var2) by expr_ir.
define void @strict_backward_where(float* noalias %x, float* noalias %g, float* noalias %dst, i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [0,%entry],[%inext,%body]
  %c = icmp slt i64 %i, %n
  br i1 %c, label %body, label %done
body:
  %xp = getelementptr float, float* %x, i64 %i
  %v0 = load float, float* %xp, align 4
  %gp = getelementptr float, float* %g, i64 %i
  %v1 = load float, float* %gp, align 4
  %l0 = fadd float 0.0, 0.0
  %l1 = fcmp ogt float %v0, %l0
  %l2 = fadd float 0.0, 0.0
  %l3 = select i1 %l1, float %v1, float %l2
  %dp = getelementptr float, float* %dst, i64 %i
  store float %l3, float* %dp, align 4
  %inext = add i64 %i, 1
  br label %loop
done:
  ret void
}
