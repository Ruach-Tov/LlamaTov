%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_l1_epilogue_ops.pl — verify new L1 elementwise epilogue clauses
%% emit syntactically valid CUDA when fused after a matmul.
%%
%% Per verification discipline: every claim about Scope B coverage gets
%% an empirical test. This test covers the activation ops added by
%% metayen 2026-05-15 for KernelBench L1.

:- set_prolog_flag(double_quotes, codes).
:- use_module('../lib/c_ast').
:- use_module('../lib/fusion_analyzer').

%% fusion_to_cuda.pl's initialization now checks the associated file
%% before running its demo, so consulting from another file is safe.
:- consult('fusion_to_cuda').

run_tests :-
    Tests = [
        test_classify_leaky_relu_elementwise,
        test_classify_selu_elementwise,
        test_classify_elu_elementwise,
        test_classify_hardsigmoid_elementwise,
        test_classify_softsign_elementwise,
        test_classify_pool_2d_spatial,
        test_classify_conv_2d_spatial,
        test_classify_mse_loss_reduction,
        test_classify_argmax_reduction,
        test_classify_group_norm_normalization,
        test_classify_flash_attn_special,
        test_emit_matmul_leaky_relu_chain,
        test_emit_matmul_gelu_chain,
        test_emit_matmul_selu_chain,
        test_emit_matmul_elu_chain,
        test_emit_matmul_hardsigmoid_chain,
        test_emit_matmul_softplus_chain,
        test_emit_matmul_softsign_chain,
        test_emit_matmul_neg_chain,
        test_emit_matmul_abs_chain,
        test_emit_matmul_sqrt_chain,
        test_emit_matmul_sqr_chain,
        test_emit_matmul_exp_chain,
        test_emit_matmul_log_chain,
        test_emit_matmul_div_chain
    ],
    run_each(Tests, 0, 0, P, F),
    format("~n=============================================~n", []),
    format("RESULTS: ~d passed, ~d failed~n", [P, F]),
    format("=============================================~n", []),
    ( F > 0 -> halt(1) ; true ).

run_each([], P, F, P, F).
run_each([T | Rest], P0, F0, P, F) :-
    ( catch(call(T), Err, (format("  FAIL ~w: error ~w~n", [T, Err]), fail))
    -> ( format("  PASS ~w~n", [T]), P1 is P0 + 1, F1 = F0 )
    ; ( format("  FAIL ~w~n", [T]), P1 = P0, F1 is F0 + 1 )
    ),
    run_each(Rest, P1, F1, P, F).

%% ─────────────────────────────────────────────────────────────────────
%% Classification tests
%% ─────────────────────────────────────────────────────────────────────

test_classify_leaky_relu_elementwise :-
    fusion_analyzer:classify_op(ggml_leaky_relu, elementwise).

test_classify_selu_elementwise :-
    fusion_analyzer:classify_op(ggml_selu, elementwise).

test_classify_elu_elementwise :-
    fusion_analyzer:classify_op(ggml_elu, elementwise).

test_classify_hardsigmoid_elementwise :-
    fusion_analyzer:classify_op(ggml_hardsigmoid, elementwise).

test_classify_softsign_elementwise :-
    fusion_analyzer:classify_op(ggml_softsign, elementwise).

test_classify_pool_2d_spatial :-
    fusion_analyzer:classify_op(ggml_pool_2d, spatial).

test_classify_conv_2d_spatial :-
    fusion_analyzer:classify_op(ggml_conv_2d, spatial).

test_classify_mse_loss_reduction :-
    fusion_analyzer:classify_op(ggml_mse_loss, reduction).

test_classify_argmax_reduction :-
    fusion_analyzer:classify_op(ggml_argmax, reduction).

test_classify_group_norm_normalization :-
    fusion_analyzer:classify_op(ggml_group_norm, normalization).

test_classify_flash_attn_special :-
    fusion_analyzer:classify_op(ggml_flash_attn_ext, special).

%% ─────────────────────────────────────────────────────────────────────
%% Epilogue emission tests
%% Each verifies that build_epilogue_stmts produces a non-empty AST
%% with the expected structure for a single-op chain.
%% ─────────────────────────────────────────────────────────────────────

test_emit_matmul_leaky_relu_chain :-
    build_epilogue_stmts([op(act, ggml_leaky_relu, 2)], Stmts),
    Stmts = [c_assign(c_var(sum), c_ternary(_, _, _))].

test_emit_matmul_gelu_chain :-
    build_epilogue_stmts([op(act, ggml_gelu, 2)], Stmts),
    Stmts = [c_assign(c_var(sum), _)],
    %% Verify it mentions erff (the GELU implementation)
    Stmts = [Stmt | _],
    contains_call(Stmt, erff).

test_emit_matmul_selu_chain :-
    build_epilogue_stmts([op(act, ggml_selu, 2)], Stmts),
    Stmts = [c_assign(c_var(sum), _)],
    Stmts = [Stmt | _],
    contains_call(Stmt, expf).

test_emit_matmul_elu_chain :-
    build_epilogue_stmts([op(act, ggml_elu, 2)], Stmts),
    Stmts = [c_assign(c_var(sum), _)],
    Stmts = [Stmt | _],
    contains_call(Stmt, expf).

test_emit_matmul_hardsigmoid_chain :-
    build_epilogue_stmts([op(act, ggml_hardsigmoid, 2)], Stmts),
    Stmts = [c_assign(c_var(sum), _)],
    Stmts = [Stmt | _],
    contains_call(Stmt, fmaxf),
    contains_call(Stmt, fminf).

test_emit_matmul_softplus_chain :-
    build_epilogue_stmts([op(act, ggml_softplus, 2)], Stmts),
    Stmts = [c_assign(c_var(sum), _)],
    Stmts = [Stmt | _],
    contains_call(Stmt, log1pf),
    contains_call(Stmt, expf).

test_emit_matmul_softsign_chain :-
    build_epilogue_stmts([op(act, ggml_softsign, 2)], Stmts),
    Stmts = [c_assign(c_var(sum), _)],
    Stmts = [Stmt | _],
    contains_call(Stmt, fabsf).

test_emit_matmul_neg_chain :-
    build_epilogue_stmts([op(act, ggml_neg, 2)], Stmts),
    Stmts = [c_assign(c_var(sum), c_unop('-', c_var(sum)))].

test_emit_matmul_abs_chain :-
    build_epilogue_stmts([op(act, ggml_abs, 2)], Stmts),
    Stmts = [c_assign(c_var(sum), c_call(fabsf, [c_var(sum)]))].

test_emit_matmul_sqrt_chain :-
    build_epilogue_stmts([op(act, ggml_sqrt, 2)], Stmts),
    Stmts = [c_assign(c_var(sum), c_call(sqrtf, [c_var(sum)]))].

test_emit_matmul_sqr_chain :-
    build_epilogue_stmts([op(act, ggml_sqr, 2)], Stmts),
    Stmts = [c_assign(c_var(sum), c_binop('*', c_var(sum), c_var(sum)))].

test_emit_matmul_exp_chain :-
    build_epilogue_stmts([op(act, ggml_exp, 2)], Stmts),
    Stmts = [c_assign(c_var(sum), c_call(expf, [c_var(sum)]))].

test_emit_matmul_log_chain :-
    build_epilogue_stmts([op(act, ggml_log, 2)], Stmts),
    Stmts = [c_assign(c_var(sum), c_call(logf, [c_var(sum)]))].

test_emit_matmul_div_chain :-
    build_epilogue_stmts([op(act, ggml_div, 2)], Stmts),
    Stmts = [c_assign(c_var(sum),
                       c_binop('/', c_var(sum),
                               c_index(c_var(operand), c_var(col))))].

%% ─────────────────────────────────────────────────────────────────────
%% Helper: recursively search an AST term for a c_call to a given function.
%% ─────────────────────────────────────────────────────────────────────

contains_call(c_call(FuncName, _Args), FuncName) :- !.
contains_call(c_call(_, Args), Target) :-
    member(A, Args), contains_call(A, Target), !.
contains_call(c_binop(_, L, R), Target) :-
    ( contains_call(L, Target) ; contains_call(R, Target) ), !.
contains_call(c_unop(_, X), Target) :-
    contains_call(X, Target), !.
contains_call(c_paren(X), Target) :-
    contains_call(X, Target), !.
contains_call(c_ternary(C, T, E), Target) :-
    ( contains_call(C, Target)
    ; contains_call(T, Target)
    ; contains_call(E, Target)
    ), !.
contains_call(c_assign(_, RHS), Target) :-
    contains_call(RHS, Target), !.
contains_call(c_index(L, R), Target) :-
    ( contains_call(L, Target) ; contains_call(R, Target) ), !.

:- initialization(run_tests, main).
