%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% op_classify.pl — derive an op's fusion CLASS from its op_expr AST structure.
%%
%% fusion_analyzer.pl classifies ops for fusion (elementwise/reduction/spatial/
%% normalization/matmul/layout) — but its classify_op/2 was a hardcoded table of
%% ggml op names. That couples the fusion analysis to the ggml backend. Our
%% op_expr facts are MORE GENERAL: the class is derivable from the AST node type,
%% backend-independent. This module provides that general classifier so
%% fusion_analyzer can operate on our lifted bpd ops (the single source of truth),
%% with ggml as just one downstream backend.
%%
%% classify_bpd_op(+BpdOp, -Class): look up op_expr(BpdOp, Term), classify Term.
%% classify_expr(+Term, -Class): the structural classifier (the general core).
%%
%% Author: Iyun, 2026-06-08 — porting fusion classification off ggml onto op_expr.

:- module(op_classify, [
    classify_bpd_op/2,        % classify_bpd_op(+BpdOp, -Class)
    classify_expr/2           % classify_expr(+Term, -Class)
]).

:- use_module('robust_op_match', []).

%% ── classify_expr: the AST node type determines the fusion class ─────────────
%% DETERMINISTic: compute the ONE class via classify_expr_/2 (first match wins),
%% then unify with the requested/output Class. This prevents a pre-bound MISMATCHED
%% class (e.g. classify_expr(conv(...), elementwise)) from falling through to the
%% compound catch-all — conv must classify as spatial ONLY. (Without this, asking
%% to PROVE classify_expr(conv,elementwise) wrongly succeeds via the catch-all,
%% because the cut clauses' heads don't unify when the class arg is pre-bound.)
classify_expr(Term, Class) :-
    classify_expr_(Term, Computed),
    !,
    Class = Computed.

%% Order matters: match the structural heads (conv/pool/norm/reduce/softmax)
%% before falling through to the elementwise arithmetic catch-all.
classify_expr_(conv(_,_,_,_,_,_),   spatial) :- !.
classify_expr_(conv(_,_,_,_,_,_,_), spatial) :- !.        % 7-arg transposed
classify_expr_(pool(_,_,_,_,_,_,_), spatial) :- !.
classify_expr_(batchnorm(_,_),      normalization) :- !.
classify_expr_(groupnorm(_,_,_),    normalization) :- !.
classify_expr_(stat_norm(_,_,_),    normalization) :- !.
classify_expr_(rmsnorm(_,_,_),      normalization) :- !.
classify_expr_(l1norm(_,_),         normalization) :- !.
classify_expr_(l2norm(_,_),         normalization) :- !.
classify_expr_(frobnorm(_),         normalization) :- !.
classify_expr_(softmax(_,_),        reduction) :- !.
classify_expr_(log_softmax(_,_),    reduction) :- !.
classify_expr_(logsumexp(_,_),      reduction) :- !.
classify_expr_(axis_reduce(_,_,_),  reduction) :- !.
classify_expr_(reduce(_,_,_,_,_),   matmul) :- !.         % the matmul reduce form
%% loss composites read 2-3 operands; treat as reduction (produce a scalar/few)
classify_expr_(cross_entropy(_,_),  reduction) :- !.
classify_expr_(huber(_,_),          reduction) :- !.
classify_expr_(hinge(_,_),          reduction) :- !.
classify_expr_(kl_div(_,_),         reduction) :- !.
classify_expr_(mse(_,_),            reduction) :- !.
classify_expr_(triplet(_,_,_),      reduction) :- !.
%% var passthrough = layout-neutral (identity/dropout)
classify_expr_(var,                 layout) :- !.
%% everything else (var-rooted arithmetic, sel, call(activation), scalar ops)
%% is elementwise — reads/writes each element independently.
classify_expr_(T, elementwise) :- compound(T), !.
classify_expr_(_, elementwise).

%% ── classify_bpd_op: resolve the op's op_expr term, then classify it ─────────
classify_bpd_op(BpdOp, Class) :-
    robust_op_match:op_expr(BpdOp, Term),
    classify_expr(Term, Class).
