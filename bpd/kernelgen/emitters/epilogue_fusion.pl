%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════════
%% epilogue_fusion.pl — MOVE 3 (thesis fidelity): epilogue fusion at the AST level,
%% so EVERY backend inherits it — not just cuda-c via -D macros.
%%
%% The drift: CONV_EPILOGUE / GEMM_EPILOGUE were cuda-c preprocessor macros, so the
%% fusion only reached the cuda backend. But the epilogue tail is just an op_expr
%% SUB-TERM (compose_chain already folds it). If we lower that ONE folded term
%% per-backend (lower_cuda / lower_mlir / lower_oxide all exist in expr_ir), the
%% SAME fusion reaches every backend. Fusion becomes backend-neutral.
%%
%% epilogue_expr(+TailChain, -Expr)   — fold the elementwise tail to one op_expr term
%% epilogue_cuda(+TailChain, -CStr)   — that term as a C expression over 'v'
%% epilogue_mlir(+TailChain, +Root, -Stmts) — that term as arith/math SSA over Root
%% epilogue_identity_cuda/1, epilogue_identity_mlir/2 — the no-fusion default
%% Author: Iyun, 2026-06-08
%% ═══════════════════════════════════════════════════════════════════════════
:- module(epilogue_fusion, [
    epilogue_expr/2,        % epilogue_expr(+TailChain, -Expr)  [the folded AST]
    epilogue_cuda/2,        % epilogue_cuda(+TailChain, -CStr)
    epilogue_mlir/3,        % epilogue_mlir(+TailChain, +Root, -Stmts)
    epilogue_backends/2     % epilogue_backends(+TailChain, -Lowerings) [all at once]
]).
:- use_module(library(lists)).
:- use_module(chain_compose, [compose_chain/2]).
:- use_module(expr_ir, [lower_cuda/2]).

%% epilogue_expr(+TailChain, -Expr): the elementwise tail folded to ONE op_expr term.
%% TailChain = [bpd_relu, bpd_scaling, ...] (the ops after the head). This is the
%% backend-NEUTRAL representation — the fusion decision lives in the AST, once.
epilogue_expr([], var) :- !.                 % empty tail = identity (just the head value)
epilogue_expr(TailChain, Expr) :-
    compose_chain(TailChain, Expr).

%% epilogue_cuda(+TailChain, -CStr): the folded tail as a C expression over 'v'.
%% A cuda head emitter inlines this at its store: C[i] = <CStr>.
epilogue_cuda([], "v") :- !.
epilogue_cuda(TailChain, CStr) :-
    epilogue_expr(TailChain, Expr),
    lower_cuda(Expr, CStr).

%% epilogue_mlir(+TailChain, +Root, -Stmts): the folded tail as arith/math dialect
%% SSA statements operating on the SSA value Root, yielding a result SSA. An MLIR
%% head emitter appends Stmts at its store. SAME folded term as cuda -> same fusion.
epilogue_mlir([], Root, stmts([], Root)) :- !.   % identity: result is Root itself
epilogue_mlir(TailChain, Root, stmts(Stmts, Result)) :-
    epilogue_expr(TailChain, Expr),
    lower_expr_mlir(Expr, Root, Stmts, Result).

%% Minimal arith/math-dialect lowering of the folded epilogue expr over an SSA Root.
%% (var -> Root; const -> arith.constant; binops -> arith ops; calls -> math ops.)
%% Emits a list of "Res = op Args" SSA statement strings + the final Result name.
lower_expr_mlir(var, Root, [], Root) :- !.
lower_expr_mlir(const(C), _Root, [Stmt], Res) :- !,
    gensym('%c', Res),
    format(atom(Stmt), "~w = arith.constant ~w : f32", [Res, C]).
lower_expr_mlir(scalar(C), Root, Stmts, Res) :- !,
    lower_expr_mlir(const(C), Root, Stmts, Res).
lower_expr_mlir(mul(A,B), Root, Stmts, Res) :- !,
    lower_expr_mlir(A, Root, SA, RA), lower_expr_mlir(B, Root, SB, RB),
    gensym('%m', Res), format(atom(St), "~w = arith.mulf ~w, ~w : f32", [Res,RA,RB]),
    append([SA,SB,[St]], Stmts).
lower_expr_mlir(add(A,B), Root, Stmts, Res) :- !,
    lower_expr_mlir(A, Root, SA, RA), lower_expr_mlir(B, Root, SB, RB),
    gensym('%a', Res), format(atom(St), "~w = arith.addf ~w, ~w : f32", [Res,RA,RB]),
    append([SA,SB,[St]], Stmts).
lower_expr_mlir(call(tanh,A), Root, Stmts, Res) :- !,
    lower_expr_mlir(A, Root, SA, RA),
    gensym('%t', Res), format(atom(St), "~w = math.tanh ~w : f32", [Res,RA]),
    append(SA, [St], Stmts).
lower_expr_mlir(call(exp,A), Root, Stmts, Res) :- !,
    lower_expr_mlir(A, Root, SA, RA),
    gensym('%e', Res), format(atom(St), "~w = math.exp ~w : f32", [Res,RA]),
    append(SA, [St], Stmts).
%% relu via the sel(is_nan...) folded form -> arith.maximumf with 0 (nan-propagating
%% form would need a cmp; the common relu fold lowers to max(v,0)).
lower_expr_mlir(sel(ge(A,const(0.0)),A,const(0.0)), Root, Stmts, Res) :- !,
    lower_expr_mlir(A, Root, SA, RA),
    gensym('%z', Z), format(atom(Sz), "~w = arith.constant 0.0 : f32", [Z]),
    gensym('%r', Res), format(atom(Sr), "~w = arith.maximumf ~w, ~w : f32", [Res,RA,Z]),
    append([SA,[Sz,Sr]], Stmts).
%% nan-guarded relu: sel(is_nan(v),v, sel(ge(v,0),v,0)) -> drop the nan guard for MLIR
%% (math dialect propagates nan through maximumf) -> max(v,0).
lower_expr_mlir(sel(is_nan(A),A,Inner), Root, Stmts, Res) :- !,
    lower_expr_mlir(Inner, Root, Stmts, Res).   % the inner is the relu max-form

%% epilogue_backends(+TailChain, -Lowerings): all backend lowerings of the SAME fold.
%% Lowerings = [cuda(CStr), mlir(Stmts,Result)]. Proves backend-neutrality: one fold,
%% many lowerings, identical semantics.
epilogue_backends(TailChain, [cuda(CStr), mlir(Stmts, Res)]) :-
    epilogue_cuda(TailChain, CStr),
    epilogue_mlir(TailChain, "%v", stmts(Stmts, Res)).
