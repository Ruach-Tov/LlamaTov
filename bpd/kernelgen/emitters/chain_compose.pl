%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% chain_compose.pl — compose an L2 op-chain into a single op_expr term.
%%
%% L2 problems are chains of L1 ops: x -> op1 -> op2 -> ... -> opN. Each op_expr
%% body is `var`-rooted (var is the input placeholder). A chain composes by
%% SUBSTITUTING each step's var with the prior step's full term:
%%   chain [relu, scaling, tanh]
%%     = subst(tanh_body, var, subst(scaling_body, var, relu_body))
%%     = call(tanh, mul(sel(...relu...), scalar(2.0)))
%% The result is ONE op_expr term for the whole fused chain — which lower_torch /
%% lower_cuda / etc. emit as a single fused kernel (the always-profitable
%% elementwise-chain fusion, realized as term composition).
%%
%% This is the L2 execution path: lift chain -> compose -> lower -> verify.
%% Author: Iyun, 2026-06-08.

:- module(chain_compose, [
    compose_chain/2,          % compose_chain(+OpList, -ComposedTerm)
    compose_chain_terms/2,    % compose_chain_terms(+TermList, -ComposedTerm)
    subst_var/3,              % subst_var(+Term, +Replacement, -Result)
    var_composable/1,         % var_composable(+Op) — folds into a var-rooted term
    split_chain/3,            % split_chain(+OpList, -HeadOps, -FusableTail)
    fused_tail_term/2         % fused_tail_term(+FusableTailOps, -OneTerm)
]).

:- use_module(expr_ir).
:- use_module('../../lib/robust_op_match', []).

%% subst_var(+Term, +Replacement, -Result):
%% replace every leaf `var` in Term with Replacement. Walks the AST recursively.
subst_var(var, R, R) :- !.
subst_var(T, _, T) :- atomic(T), !.                 % const atoms, scalars, numbers
subst_var(const(C), _, const(C)) :- !.
subst_var(scalar(S), _, scalar(S)) :- !.
subst_var(T, R, Out) :-
    compound(T), !,
    T =.. [F | Args],
    maplist(subst_one(R), Args, Args2),
    Out =.. [F | Args2].

subst_one(R, Arg, Out) :- subst_var(Arg, R, Out).

%% compose_chain_terms(+TermList, -Composed):
%% fold a list of var-rooted op_expr terms into one composed term.
%% The first term is the base (its var stays = the actual input x); each
%% subsequent term's var is replaced by the accumulated term.
compose_chain_terms([T], T) :- !.
compose_chain_terms([First | Rest], Composed) :-
    foldl(compose_step, Rest, First, Composed).

compose_step(NextTerm, Acc, Out) :-
    subst_var(NextTerm, Acc, Out).

%% compose_chain(+OpList, -Composed):
%% OpList = list of bpd_* op names; look up each op_expr body, then compose.
compose_chain(OpList, Composed) :-
    maplist(lookup_op_expr, OpList, Terms),
    compose_chain_terms(Terms, Composed).

lookup_op_expr(Op, Term) :-
    ( robust_op_match:op_expr(Op, Term) -> true
    ; throw(error(no_op_expr(Op), compose_chain)) ).

%% ── chain splitting: var-composable tail vs operand-bound heads ──────────────
%% An op is var-composable iff its op_expr body contains a `var` leaf — meaning
%% it folds into a single-input fused term. Operand-bound ops (conv: takes x,w;
%% matmul: uses elem(a)/elem(b)) have NO var leaf — they're chain HEADS that run
%% with their own operands, and the fusable tail composes AFTER them.
var_composable(Op) :-
    robust_op_match:op_expr(Op, Term),
    term_has_var(Term).

term_has_var(var) :- !.
term_has_var(T) :- compound(T), T =.. [_|Args], member(A, Args), term_has_var(A), !.

%% split_chain(+OpList, -HeadOps, -FusableTail):
%% partition the chain into the leading operand-bound HEAD ops (run individually)
%% and the trailing run of var-composable ops (fused into ONE kernel). Heads are
%% the ops up to the LAST operand-bound op; everything after composes.
%% (Conservative: any operand-bound op anywhere flushes the fusable run before it.)
split_chain(OpList, HeadOps, FusableTail) :-
    split_at_last_head(OpList, [], HeadOps, FusableTail).

split_at_last_head([], HAcc, HAcc, []).
split_at_last_head([Op|Rest], HAcc, Heads, Tail) :-
    ( var_composable(Op)
    -> ( all_composable(Rest)
       -> % Op + everything after composes -> this is the start of the tail
          Heads = HAcc, Tail = [Op|Rest]
       ;  % a later op is operand-bound -> Op belongs to the head region
          append(HAcc, [Op], HAcc2),
          split_at_last_head(Rest, HAcc2, Heads, Tail) )
    ;  % Op is operand-bound -> part of heads
       append(HAcc, [Op], HAcc2),
       split_at_last_head(Rest, HAcc2, Heads, Tail) ).

all_composable(Ops) :- forall(member(O, Ops), var_composable(O)).

%% fused_tail_term(+TailOps, -OneTerm): compose the fusable tail into one term.
%% (Identity for empty tail handled by caller.)
fused_tail_term([], var) :- !.
fused_tail_term(TailOps, Term) :- compose_chain(TailOps, Term).
