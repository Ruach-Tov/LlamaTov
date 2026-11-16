; PROTOTYPE one-way-leaky backward — CPU LLVM IR (bocher, 2026-11-16).
; Semantics (relu doc §7.3, torch convention: revival = grad_out < 0):
;   grad_in = grad_out * ( (x > 0) ? 1 : (x <= 0 && grad_out < 0) ? alpha : 0 )
; alpha baked = 0.01 (oneway_10 arm). Two-input elementwise — the var2
; extension this prototype motivates for expr_ir.pl (Iyun's canonical lane).
; Convention header (§6 contract): x=0 treated as DEAD side (x<=0);
; -0.0 <= 0 is IEEE-true -> dead side; NaN x -> mask 0 (matches strict).
define void @oneway_backward(float* noalias %x, float* noalias %g, float* noalias %dst, i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [0, %entry], [%inext, %body]
  %cond = icmp slt i64 %i, %n
  br i1 %cond, label %body, label %done
body:
  %xp = getelementptr float, float* %x, i64 %i
  %xv = load float, float* %xp, align 4
  %gp = getelementptr float, float* %g, i64 %i
  %gv = load float, float* %gp, align 4
  ; alive = x > 0 (ordered: NaN -> false -> dead path)
  %alive = fcmp ogt float %xv, 0.0
  ; revival = g < 0 (ordered)
  %revival = fcmp olt float %gv, 0.0
  ; dead-side factor: revival ? alpha : 0
  %deadf = select i1 %revival, float 0x3F847AE140000000, float 0.0
  ; mask = alive ? 1.0 : deadf
  %mask = select i1 %alive, float 1.0, float %deadf
  %res = fmul float %gv, %mask
  %dp = getelementptr float, float* %dst, i64 %i
  store float %res, float* %dp, align 4
  %inext = add i64 %i, 1
  br label %loop
done:
  ret void
}
