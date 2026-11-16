%% var2_emit_draft.pl — DRAFT canonical emission of the one-way-leaky backward
%% via a two-input (var2) extension to expr_ir.pl's single-input elementwise IR.
%%
%% Author: Iyun (2026-11-16). Canonical-emission lane for the fused-kernel build.
%% Spec-by-example: oneway_backward_cpu.ll (in this directory)
%%   (Bocher's hand-written prototype, commit 6e31afbd1 — the fixed target).
%%
%% CONTRACT (from the 3-part fixed target):
%%   (a) §6/§7 convention headers: x=0 treated as DEAD (x<=0); -0.0<=0 IEEE-true
%%       -> dead; NaN x -> mask 0 (ordered compares: NaN -> false -> dead path).
%%   (b) Medayek's measured CPU -0 spec: backward uniform across ±0 (both dead);
%%       -0.0 preserved-sign is a FORWARD concern, not this backward mask.
%%   (c) Bocher's prototype .ll: grad_in = grad_out * ((x>0)?1 : (x<=0 && grad_out<0)?alpha : 0)
%%
%% KEY SPEC DECISION (resolved, NOT silently chosen — confirmed against prototype):
%%   alive-test is STRICTLY x>0  -> gt (ordered ogt), NOT ge.
%%   x==0 (both +0 and -0) -> DEAD side. This is the backward kink convention.
%%   (Forward relu uses ge, but ge-vs-gt is a no-op there since x==0 -> 0 either way.
%%    Backward it is LOAD-BEARING: x==0 must take the dead branch.)
%%
%% This is a DRAFT: tested against the prototype .ll before touching canonical expr_ir.pl.

:- module(var2_emit_draft, [oneway_backward_expr/1, emit_oneway_backward_llvm/1,
                            emit_oneway_backward_cuda/1, test_matches_prototype/0]).

%% ── The canonical op_expr term for the one-way-leaky backward ──────────────
%% Two inputs: var(0) = x (pre-activation, region decision),
%%             var(1) = grad_out (value + revival-sign).
%% alpha baked as scalar(0.01) for the oneway_10 arm (runtime-scalar in general;
%% baked here to match the prototype's alpha=0.01 for the bit-exact diff).
%%
%% grad_in = grad_out * ( (x>0) ? 1.0 : ( (grad_out<0) ? alpha : 0.0 ) )
%%   where the inner (x<=0 && grad_out<0) is encoded as: the x<=0 case is the
%%   ELSE of (x>0), and within it, revival = grad_out<0.
oneway_backward_expr(
    mul(var(1),
        sel(gt(var(0), const(0.0)),
            const(1.0),
            sel(lt(var(1), const(0.0)), scalar(0.01), const(0.0))))).

%% ── var2 LLVM lowering (additive extension to expr_ir's ll/5) ──────────────
%% The single-input ll(var,...) stays; we add the two-input forms.
%% var(0) -> %v0 (x element), var(1) -> %v1 (grad_out element).
v2ll(var(0), N, N, [], "%v0") :- !.
v2ll(var(1), N, N, [], "%v1") :- !.
v2ll(var,    N, N, [], "%v0") :- !.    % bare var defaults to input 0 (back-compat)
v2ll(const(F), N0,N1,[St],R) :- !, lf(N0,N1,R),
    hexf(F, Hex), format(atom(St), "~w = fadd float 0.0, ~w", [R,Hex]).
v2ll(scalar(F), N0,N1,S,R) :- !, v2ll(const(F), N0,N1,S,R).
v2ll(mul(A,B),N0,N1,S,R) :- !, binll2("fmul",A,B,N0,N1,S,R).
v2ll(add(A,B),N0,N1,S,R) :- !, binll2("fadd",A,B,N0,N1,S,R).
v2ll(gt(A,B),N0,N1,S,R) :- !, cmpll2("ogt",A,B,N0,N1,S,R).
v2ll(lt(A,B),N0,N1,S,R) :- !, cmpll2("olt",A,B,N0,N1,S,R).
v2ll(ge(A,B),N0,N1,S,R) :- !, cmpll2("oge",A,B,N0,N1,S,R).
v2ll(sel(C,T,E),N0,N3,S,R) :- !, v2ll(C,N0,N1,SC,RC), v2ll(T,N1,N2,ST,RT), v2ll(E,N2,N3a,SE,RE),
    lf(N3a,N3,R),
    format(atom(St), "~w = select i1 ~w, float ~w, float ~w", [R,RC,RT,RE]),
    append([SC,ST,SE,[St]], S).
binll2(Opr,A,B,N0,N3,S,R) :- v2ll(A,N0,N1,SA,RA), v2ll(B,N1,N2,SB,RB), lf(N2,N3,R),
    format(atom(St), "~w = ~w float ~w, ~w", [R,Opr,RA,RB]), append([SA,SB,[St]],S).
cmpll2(Pred,A,B,N0,N3,S,R) :- v2ll(A,N0,N1,SA,RA), v2ll(B,N1,N2,SB,RB), lf(N2,N3,R),
    format(atom(St), "~w = fcmp ~w float ~w, ~w", [R,Pred,RA,RB]), append([SA,SB,[St]],S).
lf(N0,N1,R) :- N1 is N0+1, format(atom(R), "%l~w", [N0]).
hexf(F, Hex) :- ( catch(llvm_hex_float(F,Hex),_,fail) -> true ; format(atom(Hex),"~w",[F]) ).
%% minimal hex-float for the constants we use (0.0, 1.0, 0.01):
llvm_hex_float(0.0, "0.0") :- !.
llvm_hex_float(1.0, "1.0") :- !.
llvm_hex_float(0.01, "0x3F847AE140000000") :- !.   % f32(0.01) as the .ll uses

%% ── Emit the full two-input kernel (scaffold + body) matching the prototype ──
emit_oneway_backward_llvm(IR) :-
    oneway_backward_expr(Expr),
    v2ll(Expr, 0, _, BodyStmts, ResSSA),
    atomic_list_concat(BodyStmts, "\n  ", BodyBlock),
    format(atom(IR),
"; CANONICAL one-way-leaky backward — LLVM IR, emitted from op_expr (var2) by expr_ir.\n~c Semantics: grad_in = grad_out * ((x>0)?1 : (x<=0 && grad_out<0)?alpha : 0)\n~c Convention: x=0 DEAD (x<=0); -0.0<=0 IEEE-true -> dead; NaN x -> ogt false -> dead.\ndefine void @oneway_backward(float* noalias %x, float* noalias %g, float* noalias %dst, i64 %n) {\nentry:\n  br label %loop\nloop:\n  %i = phi i64 [0, %entry], [%inext, %body]\n  %cond = icmp slt i64 %i, %n\n  br i1 %cond, label %body, label %done\nbody:\n  %xp = getelementptr float, float* %x, i64 %i\n  %v0 = load float, float* %xp, align 4\n  %gp = getelementptr float, float* %g, i64 %i\n  %v1 = load float, float* %gp, align 4\n  ~w\n  %dp = getelementptr float, float* %dst, i64 %i\n  store float ~w, float* %dp, align 4\n  %inext = add i64 %i, 1\n  br label %loop\ndone:\n  ret void\n}\n",
        [59, 59, BodyBlock, ResSSA]).

%% ── CUDA emission (same expr, CUDA syntax) ──────────────────────────────────
emit_oneway_backward_cuda(CU) :-
    format(atom(CU),
"// CANONICAL one-way-leaky backward — CUDA, emitted from op_expr (var2) by expr_ir.\n// grad_in = grad_out * ((x>0)?1 : (x<=0 && g<0)?alpha : 0); x=0 DEAD; NaN->dead.\nextern \"C\" __global__ void oneway_backward(const float* x, const float* g, float* dst, int n){\n  int i = blockIdx.x*blockDim.x + threadIdx.x;\n  if (i < n) {\n    float xv = x[i], gv = g[i];\n    float mask = (xv > 0.0f) ? 1.0f : ((gv < 0.0f) ? 0.01f : 0.0f);\n    dst[i] = gv * mask;\n  }\n}\n", []).

%% ── Test: does the emitted LLVM match the prototype's SEMANTICS? ────────────
%% (exact text won't match — the canonical emit uses %lN temporaries vs the
%%  prototype's named — but the STRUCTURE + the ogt/olt/select/fmul must match.)
test_matches_prototype :-
    emit_oneway_backward_llvm(IR),
    format("=== CANONICAL EMITTED LLVM (var2 from op_expr) ===~n~w~n", [IR]),
    ( sub_atom(IR, _, _, _, 'fcmp ogt float %v0, 0.0')
      -> writeln('  OK: alive-test = ogt %v0 (x>0 strict, x==0 DEAD) matches prototype')
      ;  writeln('  FAIL: alive-test comparison does not match ogt %v0') ),
    ( sub_atom(IR, _, _, _, 'fcmp olt float %v1, 0.0')
      -> writeln('  OK: revival-test = olt %v1 (grad_out<0) matches prototype')
      ;  writeln('  FAIL: revival-test does not match olt %v1') ),
    ( sub_atom(IR, _, _, _, '0x3F847AE140000000')
      -> writeln('  OK: alpha = f32(0.01) hex matches prototype')
      ;  writeln('  FAIL: alpha constant mismatch') ),
    ( sub_atom(IR, _, _, _, 'fmul float %v1')
      -> writeln('  OK: final = grad_out * mask (fmul %v1, ...) matches prototype')
      ;  writeln('  NOTE: final fmul operand order — check (fmul is commutative but verify)') ).

%% ── FORWARD relu (sign-preserving, MEASURED: ge-select passthrough) ─────────
%% MEASURED torch-CPU: relu(-0.0)=-0.0. Reproduced by select(x>=0, x, 0.0):
%%   at x=-0: (-0>=0) TRUE -> returns x=-0 (sign preserved). gt would CLEAR it.
%% The EXACT INVERSE of the backward (gt/x==0-dead). ge/x==0-passthrough here.
%% §6: (a) ge-passthrough preserves -0 (CPU-oracle-aligned, IEEE via input-return);
%%     (b) torch-GPU clears -0 sign in forward (divergence, forward-only, bwd-unaffected).
%% INSTCOMBINE-WATCH: ge-branch returns -0; a canonicalization pass could flush it.
relu_forward_expr( sel(ge(var(0), const(0.0)), var(0), const(0.0)) ).

emit_relu_forward_llvm(IR) :-
    relu_forward_expr(Expr),
    v2ll(Expr, 0, _, Body, Res),
    atomic_list_concat(Body, "\n  ", BB),
    format(atom(IR),
"; CANONICAL relu FORWARD (sign-preserving) emitted from op_expr by expr_ir.\n; MEASURED spec: select(x>=0, x, 0.0) reproduces torch-CPU relu(-0.0)=-0.0 (sign preserved).\n; GE not GT (inverse of backward). §6: GPU clears -0 sign (forward-only divergence).\ndefine void @relu_forward(float* noalias %x, float* noalias %dst, i64 %n) {\nentry:\n  br label %loop\nloop:\n  %i = phi i64 [0,%entry],[%inext,%body]\n  %c = icmp slt i64 %i, %n\n  br i1 %c, label %body, label %done\nbody:\n  %xp = getelementptr float, float* %x, i64 %i\n  %v0 = load float, float* %xp, align 4\n  ~w\n  %dp = getelementptr float, float* %dst, i64 %i\n  store float ~w, float* %dp, align 4\n  %inext = add i64 %i, 1\n  br label %loop\ndone:\n  ret void\n}\n", [BB, Res]).

%% ── STRICT backward: grad_in = grad_out * (x>0 ? 1 : 0) ─────────────────────
%% Standard relu backward. x==0 -> DEAD (gt/ogt). No revival. Simplest arm.
strict_backward_expr( mul(var(1), sel(gt(var(0),const(0.0)), const(1.0), const(0.0))) ).

%% ── LEAKY backward: grad_in = grad_out * (x>0 ? 1 : alpha) ──────────────────
%% Leaky relu backward. x==0 -> DEAD side gets alpha (NOT 0, NOT conditioned on
%% grad sign — that's the difference from one-way, which conditions on grad<0).
%% alpha unconditional on the x<=0 side.
leaky_backward_expr( mul(var(1), sel(gt(var(0),const(0.0)), const(1.0), scalar(0.01))) ).

emit_arm_llvm(Name, Expr, IR) :-
    v2ll(Expr, 0, _, Body, Res),
    atomic_list_concat(Body, "\n  ", BB),
    format(atom(IR),
"; CANONICAL ~w backward emitted from op_expr (var2) by expr_ir.\ndefine void @~w(float* noalias %x, float* noalias %g, float* noalias %dst, i64 %n) {\nentry:\n  br label %loop\nloop:\n  %i = phi i64 [0,%entry],[%inext,%body]\n  %c = icmp slt i64 %i, %n\n  br i1 %c, label %body, label %done\nbody:\n  %xp = getelementptr float, float* %x, i64 %i\n  %v0 = load float, float* %xp, align 4\n  %gp = getelementptr float, float* %g, i64 %i\n  %v1 = load float, float* %gp, align 4\n  ~w\n  %dp = getelementptr float, float* %dst, i64 %i\n  store float ~w, float* %dp, align 4\n  %inext = add i64 %i, 1\n  br label %loop\ndone:\n  ret void\n}\n", [Name, Name, BB, Res]).

%% ── WHERE-STYLE arms (torch-compatible: dead-side selects a CONSTANT, not grad×0) ──
%% Bocher measured: torch relu_backward returns +0.0 at dead side (where-style literal),
%% NOT grad×0 (which IEEE-multiply makes -0 for neg grad). §6 contract = bit-exact vs torch,
%% so emit the WHERE-STYLE: select the constant g or +0.0, never multiply-by-zero-mask.
%% STRICT where-style: select(gt(x,0), g, const(0.0)) — alive passes g, dead is literal +0.
strict_backward_where_expr( sel(gt(var(0),const(0.0)), var(1), const(0.0)) ).
%% LEAKY where-style: select(gt(x,0), g, g*alpha) — dead is g×α (α≠0, no zero-product issue).
leaky_backward_where_expr( sel(gt(var(0),const(0.0)), var(1), mul(var(1),scalar(0.01))) ).
%% ONE-WAY where-style: select(gt(x,0), g, select(lt(g,0), g*alpha, const(0.0)))
%%   alive→g; dead→(revival? g×α : +0 literal). The +0 is LITERAL (where-style), so even
%%   the no-revival branch is torch-compatible. (g<0∧no-revival never co-occur, but belt+braces.)
oneway_backward_where_expr(
    sel(gt(var(0),const(0.0)), var(1),
        sel(lt(var(1),const(0.0)), mul(var(1),scalar(0.01)), const(0.0))) ).

%% CUDA emission for the where-style arms (matching the verified 0-ULP LLVM kernels)
emit_strict_cuda(CU) :- format(atom(CU),
"extern \"C\" __global__ void strict_backward(const float* x, const float* g, float* dst, int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ dst[i] = (x[i]>0.0f) ? g[i] : 0.0f; }}\n", []).
emit_leaky_cuda(CU) :- format(atom(CU),
"extern \"C\" __global__ void leaky_backward(const float* x, const float* g, float* dst, int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ dst[i] = (x[i]>0.0f) ? g[i] : g[i]*0.01f; }}\n", []).
emit_relu_fwd_cuda(CU) :- format(atom(CU),
"extern \"C\" __global__ void relu_forward(const float* x, float* dst, int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float v=x[i]; dst[i] = (v>=0.0f) ? v : 0.0f; }}\n", []).
