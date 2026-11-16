; CANONICAL relu FORWARD (sign-preserving) emitted from op_expr by expr_ir.
; MEASURED spec: select(x>=0, x, 0.0) reproduces torch-CPU relu(-0.0)=-0.0 (sign preserved).
; GE not GT (inverse of backward). §6: GPU clears -0 sign (forward-only divergence).
define void @relu_forward(float* noalias %x, float* noalias %dst, i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [0,%entry],[%inext,%body]
  %c = icmp slt i64 %i, %n
  br i1 %c, label %body, label %done
body:
  %xp = getelementptr float, float* %x, i64 %i
  %v0 = load float, float* %xp, align 4
  %l0 = fadd float 0.0, 0.0
  %l1 = fcmp oge float %v0, %l0
  %l2 = fadd float 0.0, 0.0
  %l3 = select i1 %l1, float %v0, float %l2
  %dp = getelementptr float, float* %dst, i64 %i
  store float %l3, float* %dp, align 4
  %inext = add i64 %i, 1
  br label %loop
done:
  ret void
}
