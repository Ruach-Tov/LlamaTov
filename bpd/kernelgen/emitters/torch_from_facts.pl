%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════════
%% torch_from_facts.pl — PyTorch backend (backend 5), generated from the SAME
%% neutral expression AST (expr_ir.pl op_expr/2 -> lower_torch/2). Emits a Python
%% module with a function op(x) computing the op in torch ops over a tensor x.
%% The differential referee runs it vs torch.<op> for verification.
%%
%% Author: Iyun, 2026-06-07 (backend 5 wired from the expression IR)
%% ═══════════════════════════════════════════════════════════════════════════

:- module(torch_from_facts, [emit_torch_from_fact/2, torch_supported_op/1, emit_torch_term/3]).

:- ( prolog_load_context(directory, ED),
     atomic_list_concat([ED, '/expr_ir.pl'], EP), exists_file(EP)
   -> use_module(EP, [op_expr/2, lower_torch/2])
   ;  exists_file('kernelgen/emitters/expr_ir.pl')
   -> use_module('kernelgen/emitters/expr_ir.pl', [op_expr/2, lower_torch/2])
   ;  exists_file('kernelgen/emitters/expr_ir.pl')
   -> use_module('kernelgen/emitters/expr_ir.pl', [op_expr/2, lower_torch/2])
   ;  true ).

torch_supported_op(Op) :- op_expr(Op, _).

%% emit_torch_from_fact(+Op, +OutFile): write a Python module: op(x) -> tensor.
%% loss ops take (predictions, targets) as the input tuple.
loss_expr(cross_entropy(_,_)).
loss_expr(smooth_l1(_,_)).
loss_expr(kl_div(_,_)).
loss_expr(hinge(_,_)).
loss_expr(mse(_,_)).

%% emit_torch_term(+Name, +Expr, +OutFile): emit a torch fn from an EXPLICIT expr
%% term (params already substituted) — used for per-problem parameterized ops
%% where the sweep extracts the actual stride/pad/kernel/dims from the KB Model.
emit_torch_term(Name, Expr, OutFile) :-
    lower_torch(Expr, PyExpr),
    open(OutFile, write, S),
    format(S, "# GENERATED (parameterized) via lower_torch — backend 5.~n", []),
    format(S, "import torch~n~n", []),
    format(S, "def ~w(x):~n", [Name]),
    ( (functor(Expr, conv, _))
    -> format(S, "    x, w, b = x~n    return ~w~n", [PyExpr])
    ;  Expr = triplet(_,_,_)
    -> format(S, "    anchor, positive, negative = x~n    return ~w~n", [PyExpr])
    ;  loss_expr(Expr)
    -> format(S, "    predictions, targets = x~n    return ~w~n", [PyExpr])
    ;  format(S, "    return ~w~n", [PyExpr]) ),
    close(S),
    format("Generated parameterized torch ~w -> ~w~n", [Name, OutFile]).

emit_torch_from_fact(Op, OutFile) :-
    op_expr(Op, Expr),
    lower_torch(Expr, PyExpr),
    (atom_concat('bpd_', Name, Op) -> true ; Name = Op),
    open(OutFile, write, S),
    format(S, "# GENERATED from op_expr(~w) via lower_torch — backend 5 (shared AST).~n", [Op]),
    format(S, "import torch~n~n", []),
    format(S, "def ~w(x):~n", [Name]),
    ( Name == matmul
    -> format(S, "    A, B = x~n    return ~w~n", [PyExpr])
    ;  functor(Expr, conv, _)              % conv ops take (input, weight, bias)
    -> format(S, "    x, w, b = x~n    return ~w~n", [PyExpr])
    ;  Expr = triplet(_,_,_)               % triplet takes (anchor,positive,negative)
    -> format(S, "    anchor, positive, negative = x~n    return ~w~n", [PyExpr])
    ;  loss_expr(Expr)                     % losses take (predictions, targets)
    -> format(S, "    predictions, targets = x~n    return ~w~n", [PyExpr])
    ;  format(S, "    return ~w~n", [PyExpr]) ),
    close(S),
    format("Generated PyTorch kernel ~w -> ~w~n", [Name, OutFile]).
