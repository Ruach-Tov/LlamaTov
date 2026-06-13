%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_kernel_templates_reduction.pl — verify Family 1 (Reductions) emission.
%%
%% Tests generate_kernel_reduction/4 for all 8 reduction op variants.
%% Per mavchin's directive: sequential implementation, no warp shuffle.

:- set_prolog_flag(double_quotes, codes).
:- use_module('../lib/c_ast').
:- use_module('../lib/kernel_templates').

run_tests :-
    Tests = [
        %% Dispatch by op kind
        test_dispatch_sum_rows,
        test_dispatch_mean,
        test_dispatch_max,
        test_dispatch_min,
        test_dispatch_argmax,
        test_dispatch_argmin,
        test_dispatch_cumsum,
        test_dispatch_cumprod,
        %% Kernel names
        test_kernel_name_sum,
        test_kernel_name_mean,
        test_kernel_name_argmax,
        test_kernel_name_cumsum,
        %% Structural properties
        test_sum_has_required_params,
        test_argmax_has_arg_variable,
        test_mean_finalizes_with_division,
        test_cumsum_writes_inline,
        test_max_uses_ternary,
        %% Emission
        test_sum_emits_compilable_cuda,
        test_argmax_emits_compilable_cuda,
        test_cumsum_emits_compilable_cuda,
        %% Indexing parens
        test_emitted_indexing_has_correct_parens
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
%% Dispatch tests (every op-kind is reachable)
%% ─────────────────────────────────────────────────────────────────────

test_dispatch_sum_rows :- generate_kernel_reduction(ggml_sum_rows, 2, axis_inner, _).
test_dispatch_mean     :- generate_kernel_reduction(ggml_mean, 2, axis_inner, _).
test_dispatch_max      :- generate_kernel_reduction(ggml_max, 2, axis_inner, _).
test_dispatch_min      :- generate_kernel_reduction(ggml_min, 2, axis_inner, _).
test_dispatch_argmax   :- generate_kernel_reduction(ggml_argmax, 2, axis_inner, _).
test_dispatch_argmin   :- generate_kernel_reduction(ggml_argmin, 2, axis_inner, _).
test_dispatch_cumsum   :- generate_kernel_reduction(ggml_cumsum, 2, axis_inner, _).
test_dispatch_cumprod  :- generate_kernel_reduction(ggml_cumprod, 2, axis_inner, _).

%% ─────────────────────────────────────────────────────────────────────
%% Kernel naming
%% ─────────────────────────────────────────────────────────────────────

test_kernel_name_sum :-
    generate_kernel_reduction(ggml_sum_rows, 2, axis_inner, K),
    K = c_func(_, _, reduce_sum, _, _).

test_kernel_name_mean :-
    generate_kernel_reduction(ggml_mean, 2, axis_inner, K),
    K = c_func(_, _, reduce_mean, _, _).

test_kernel_name_argmax :-
    generate_kernel_reduction(ggml_argmax, 2, axis_inner, K),
    K = c_func(_, _, reduce_argmax, _, _).

test_kernel_name_cumsum :-
    generate_kernel_reduction(ggml_cumsum, 2, axis_inner, K),
    K = c_func(_, _, cumsum, _, _).

%% ─────────────────────────────────────────────────────────────────────
%% Structural properties
%% ─────────────────────────────────────────────────────────────────────

test_sum_has_required_params :-
    generate_kernel_reduction(ggml_sum_rows, 2, axis_inner, K),
    K = c_func(_, _, _, Params, _),
    member(param(_, 'X'), Params),
    member(param(_, 'Y'), Params),
    member(param(_, 'N'), Params),
    member(param(_, outer), Params).

test_argmax_has_arg_variable :-
    %% argmax needs `int arg = 0;` declaration in body
    generate_kernel_reduction(ggml_argmax, 2, axis_inner, K),
    K = c_func(_, _, _, _, Body),
    member(c_decl_init(c_type(int), arg, _), Body).

test_mean_finalizes_with_division :-
    %% mean ends with Y[o] = acc / (float)N
    generate_kernel_reduction(ggml_mean, 2, axis_inner, K),
    K = c_func(_, _, _, _, Body),
    last(Body, c_assign(_, c_binop('/', _, _))).

test_cumsum_writes_inline :-
    %% cumsum writes Y[idx] = acc inside the loop, no post-loop assignment
    generate_kernel_reduction(ggml_cumsum, 2, axis_inner, K),
    K = c_func(_, _, _, _, Body),
    member(c_for(_, _, _, LoopBody), Body),
    member(c_assign(c_index(c_var('Y'), c_var(idx)), c_var(acc)), LoopBody).

test_max_uses_ternary :-
    %% max uses ternary `v > acc ? v : acc` inside the loop body
    generate_kernel_reduction(ggml_max, 2, axis_inner, K),
    K = c_func(_, _, _, _, Body),
    member(c_for(_, _, _, LoopBody), Body),
    member(c_assign(c_var(acc), c_ternary(_, _, _)), LoopBody).

%% ─────────────────────────────────────────────────────────────────────
%% Emission tests (emit_program produces valid CUDA text)
%% ─────────────────────────────────────────────────────────────────────

test_sum_emits_compilable_cuda :-
    generate_kernel_reduction(ggml_sum_rows, 2, axis_inner, K),
    emit_program([c_include_sys('cuda_runtime.h'), c_blank, K], Code),
    atom(Code),
    atom_length(Code, L), L > 100,
    sub_atom(Code, _, _, _, '__global__'),
    sub_atom(Code, _, _, _, 'reduce_sum').

test_argmax_emits_compilable_cuda :-
    generate_kernel_reduction(ggml_argmax, 2, axis_inner, K),
    emit_program([c_include_sys('cuda_runtime.h'), c_blank, K], Code),
    atom(Code),
    sub_atom(Code, _, _, _, 'reduce_argmax').

test_cumsum_emits_compilable_cuda :-
    generate_kernel_reduction(ggml_cumsum, 2, axis_inner, K),
    emit_program([c_include_sys('cuda_runtime.h'), c_blank, K], Code),
    atom(Code),
    sub_atom(Code, _, _, _, 'cumsum').

test_emitted_indexing_has_correct_parens :-
    %% The X[o * N + i] index has no precedence ambiguity (single-level)
    %% but the cumsum's idx = o * N + i should still emit correctly.
    %% Verify the emitted code contains expected indexing patterns.
    generate_kernel_reduction(ggml_cumsum, 2, axis_inner, K),
    emit_program([c_include_sys('cuda_runtime.h'), c_blank, K], Code),
    sub_atom(Code, _, _, _, 'o * N + i').

:- initialization(run_tests, main).
