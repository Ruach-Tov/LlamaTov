%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% qkv_generator.pl — Generate C AST from QKV BPD facts.
%%
%% This is the dual of qkv_lifter.pl. Together they close the full
%% round-trip loop:
%%
%%   qwen2.cpp [real C source]
%%     → c_ast:c_parse_stmts_v2 → AST
%%     → qkv_lifter:lift_qkv_section → BPD facts
%%     → qkv_generator:generate_from_bpd → AST (regenerated)
%%     → c_ast:emit → C source (regenerated)
%%     → clang-format diff = ∅
%%
%% Per mavchin's direction: "verify the full QKV section round-trips.
%% Q+K+V projections, all conditional biases, all reshapes, all RoPE.
%% One shot, full section, zero diff. The paper writes itself from
%% the commit log."

:- module(qkv_generator, [
    generate_from_bpd/2,
    bpd_ops_to_ast/3,
    op_to_ast_stmt/3
]).

%% ────────────────────────────────────────────────────────────────────
%% Main entry point
%% ────────────────────────────────────────────────────────────────────
%%
%% generate_from_bpd(+BPDFacts, -ASTStmts)
%%   Given the BPD facts produced by qkv_lifter, generate the AST
%%   statement list that mavchin's c_ast:emit can serialize back to C.

generate_from_bpd(BPDFacts, ASTStmts) :-
    %% Index the facts by op name and by sequence
    findall(Op, member(op(Op), BPDFacts), Ops),
    sort_ops_by_sequence(Ops, BPDFacts, SortedOps),
    bpd_ops_to_ast(SortedOps, BPDFacts, ASTStmts).

%% sort_ops_by_sequence(+Ops, +Facts, -SortedOps)
%%   Sort the op list by their sequence/3 facts in the BPD.
sort_ops_by_sequence(Ops, Facts, SortedOps) :-
    findall(Seq-Op,
            ( member(Op, Ops),
              member(sequence(_, Op, Seq), Facts)
            ),
            Pairs),
    keysort(Pairs, SortedPairs),
    pairs_values(SortedPairs, SortedOps).

%% bpd_ops_to_ast(+SortedOps, +Facts, -ASTStmts)
%%   Convert ordered ops into AST statements, handling tensor_join
%%   facts to produce conditional structures.
bpd_ops_to_ast([], _, []).
bpd_ops_to_ast([Op | Rest], Facts, Stmts) :-
    %% Check if this op has a tensor_join showing it's the post-bias
    %% variant in a conditional. If so, skip it (it's emitted as part
    %% of the conditional structure).
    ( member(op_condition(Op, present(_)), Facts)
    -> %% This is the post-bias op; should be emitted inside an if-block
       %% along with its tensor_join. Skip here; handled below.
       bpd_ops_to_ast(Rest, Facts, Stmts)
    ;  %% Otherwise emit this op normally
       op_to_ast_stmt(Op, Facts, OpStmts),
       bpd_ops_to_ast(Rest, Facts, RestStmts),
       append(OpStmts, RestStmts, Stmts)
    ).

%% (No /2 wrapper — facts are needed for proper op-to-AST conversion.)

%% ────────────────────────────────────────────────────────────────────
%% Per-op AST generation
%% ────────────────────────────────────────────────────────────────────
%%
%% op_to_ast_stmt(+Op, +Facts, -ASTStmts)
%%   Generate the AST statement(s) for one op. May produce multiple
%%   statements when an op has a tensor_join (the conditional pattern).

op_to_ast_stmt(Op, Facts, Stmts) :-
    member(op_kind(Op, Kind), Facts),
    member(op_inputs(Op, Inputs), Facts),
    %% Build the call args by reconstructing the c_member patterns
    %% for parameter inputs.
    inputs_to_call_args(Inputs, Facts, CallArgs),
    %% Check if this op has a tensor_join (the conditional bias pattern)
    ( member(tensor_join(Op, [if(present(BiasParam), PostBiasOp, _Alt)]), Facts)
    -> %% Generate: Op = build_X(...); if (model.layers[il].bias) { Op = ggml_add(...); cb(...); }
       %% First: the main assignment
       MainStmt = c_assign(c_var(Op), c_call(Kind, CallArgs)),
       %% Then: the conditional
       member(op_kind(PostBiasOp, AddKind), Facts),
       member(op_inputs(PostBiasOp, AddInputs), Facts),
       inputs_to_call_args(AddInputs, Facts, AddArgs),
       %% Build the if-condition: c_member(c_index(c_member(c_var(model), layers), c_var(il)), bias)
       BiasMember = c_member(c_index(c_member(c_var(model), layers), c_var(il)),
                              BiasParam),
       %% The bias-add reassigns to Op (the original tensor name)
       AddStmt = c_assign(c_var(Op), c_call(AddKind, AddArgs)),
       %% The cb after — find the matching cb_after_v2 fact
       ( member(cb_after_v2(Op, Label, _), Facts)
       -> CbStmt = c_expr_stmt(c_call(cb, [c_var(Op), c_string(Label), c_var(il)]))
       ;  atom_string(Op, OpStr),
          CbStmt = c_expr_stmt(c_call(cb, [c_var(Op), c_string(OpStr), c_var(il)]))
       ),
       IfStmt = c_if(BiasMember, [AddStmt, CbStmt]),
       Stmts = [MainStmt, IfStmt]
    ;  %% Simple op (no conditional): just emit the assignment
       Stmts = [c_assign(c_var(Op), c_call(Kind, CallArgs))]
    ).

%% inputs_to_call_args(+Inputs, +Facts, -CallArgs)
%%   Convert lifted input names back to c_ast call arguments.
%%   Parameter inputs (named in `parameter` facts) become
%%   c_member(c_index(c_member(c_var(model), layers), c_var(il)), Name).
%%   Regular tensor inputs become c_var(Name).

inputs_to_call_args([], _, []).
inputs_to_call_args([Input | Rest], Facts, [Arg | RestArgs]) :-
    ( member(parameter(Input, layer(il), from_hparams), Facts)
    -> %% Architecture parameter — reconstruct the member access pattern
       Arg = c_member(c_index(c_member(c_var(model), layers), c_var(il)),
                       Input)
    ;  Input == 'NULL'
    -> %% Align with parser's behavior: NULL is treated as a regular var.
       %% (The parser doesn't special-case NULL as c_null.)
       Arg = c_var('NULL')
    ;  %% Plain tensor reference
       Arg = c_var(Input)
    ),
    inputs_to_call_args(Rest, Facts, RestArgs).
