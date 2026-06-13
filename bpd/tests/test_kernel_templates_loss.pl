%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_kernel_templates_loss.pl — verify Family 5 (Losses).
%%
%% Tests generate_kernel_loss/4 for all 6 loss op variants.
%% Per mavchin's note: "compose from existing templates" — each loss is
%% an element-wise op followed by a per-batch reduction.

:- set_prolog_flag(double_quotes, codes).
:- use_module('../lib/c_ast').
:- use_module('../lib/kernel_templates').

run_tests :-
    Tests = [
        %% Dispatch
        test_dispatch_mse,
        test_dispatch_cross_entropy,
        test_dispatch_huber,
        test_dispatch_kl_div,
        test_dispatch_hinge,
        test_dispatch_triplet_margin,
        %% Naming
        test_kernel_name_mse,
        test_kernel_name_huber,
        test_kernel_name_triplet,
        %% Two-input vs three-input structure
        test_mse_has_X_Y_params,
        test_huber_has_extra_delta_param,
        test_triplet_has_anchor_positive_negative,
        %% Element-wise op identification
        test_mse_emits_squared_diff,
        test_cross_entropy_uses_log,
        test_kl_div_uses_two_logs,
        test_hinge_uses_max_zero,
        test_huber_uses_ternary,
        %% Reduction mode
        test_mse_mean_divides_by_N,
        test_mse_sum_does_not_divide,
        %% Emission
        test_mse_emits_compilable_cuda,
        test_triplet_emits_compilable_cuda
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
%% Dispatch
%% ─────────────────────────────────────────────────────────────────────

test_dispatch_mse           :- generate_kernel_loss(ggml_mse_loss,           mean, [], _).
test_dispatch_cross_entropy :- generate_kernel_loss(ggml_cross_entropy_loss, mean, [], _).
test_dispatch_huber         :- generate_kernel_loss(ggml_huber_loss,         mean, [], _).
test_dispatch_kl_div        :- generate_kernel_loss(ggml_kl_div_loss,        sum,  [], _).
test_dispatch_hinge         :- generate_kernel_loss(ggml_hinge_loss,         mean, [], _).
test_dispatch_triplet_margin:- generate_kernel_loss(ggml_triplet_margin_loss, mean, [], _).

%% ─────────────────────────────────────────────────────────────────────
%% Naming
%% ─────────────────────────────────────────────────────────────────────

test_kernel_name_mse :-
    generate_kernel_loss(ggml_mse_loss, mean, [], K),
    K = c_func(_, _, loss_mse, _, _).

test_kernel_name_huber :-
    generate_kernel_loss(ggml_huber_loss, mean, [], K),
    K = c_func(_, _, loss_huber, _, _).

test_kernel_name_triplet :-
    generate_kernel_loss(ggml_triplet_margin_loss, mean, [], K),
    K = c_func(_, _, loss_triplet_margin, _, _).

%% ─────────────────────────────────────────────────────────────────────
%% Parameter shapes
%% ─────────────────────────────────────────────────────────────────────

test_mse_has_X_Y_params :-
    generate_kernel_loss(ggml_mse_loss, mean, [], K),
    K = c_func(_, _, _, Params, _),
    member(param(_, 'X'), Params),
    member(param(_, 'Y'), Params),
    member(param(_, 'Out'), Params).

test_huber_has_extra_delta_param :-
    generate_kernel_loss(ggml_huber_loss, mean, [], K),
    K = c_func(_, _, _, Params, _),
    member(param(_, huber_delta), Params).

test_triplet_has_anchor_positive_negative :-
    generate_kernel_loss(ggml_triplet_margin_loss, mean, [], K),
    K = c_func(_, _, _, Params, _),
    member(param(_, anchor), Params),
    member(param(_, positive), Params),
    member(param(_, negative), Params),
    member(param(_, margin), Params).

%% ─────────────────────────────────────────────────────────────────────
%% Element-wise op verification (via emission inspection)
%% ─────────────────────────────────────────────────────────────────────

test_mse_emits_squared_diff :-
    generate_kernel_loss(ggml_mse_loss, mean, [], K),
    emit_program([c_include_sys('cuda_runtime.h'), c_blank, K], Code),
    sub_atom(Code, _, _, _, '(x_i - y_i)').

test_cross_entropy_uses_log :-
    generate_kernel_loss(ggml_cross_entropy_loss, mean, [], K),
    emit_program([c_include_sys('cuda_runtime.h'), c_blank, K], Code),
    sub_atom(Code, _, _, _, 'logf').

test_kl_div_uses_two_logs :-
    %% KL divergence has TWO logf calls (one for y_i, one for x_i)
    generate_kernel_loss(ggml_kl_div_loss, sum, [], K),
    K = c_func(_, _, _, _, Body),
    count_calls_to(Body, logf, N),
    N >= 2.

test_hinge_uses_max_zero :-
    generate_kernel_loss(ggml_hinge_loss, mean, [], K),
    emit_program([c_include_sys('cuda_runtime.h'), c_blank, K], Code),
    sub_atom(Code, _, _, _, 'fmaxf').

test_huber_uses_ternary :-
    generate_kernel_loss(ggml_huber_loss, mean, [], K),
    K = c_func(_, _, _, _, Body),
    contains_in_term(Body, c_ternary(_, _, _)).

%% ─────────────────────────────────────────────────────────────────────
%% Reduction mode
%% ─────────────────────────────────────────────────────────────────────

test_mse_mean_divides_by_N :-
    generate_kernel_loss(ggml_mse_loss, mean, [], K),
    K = c_func(_, _, _, _, Body),
    %% last assignment should be Out[b] = acc / (float)N
    last(Body, c_assign(_, c_binop('/', _, _))).

test_mse_sum_does_not_divide :-
    generate_kernel_loss(ggml_mse_loss, sum, [], K),
    K = c_func(_, _, _, _, Body),
    %% last assignment should be Out[b] = acc (no division)
    last(Body, c_assign(_, c_var(acc))).

%% ─────────────────────────────────────────────────────────────────────
%% Emission
%% ─────────────────────────────────────────────────────────────────────

test_mse_emits_compilable_cuda :-
    generate_kernel_loss(ggml_mse_loss, mean, [], K),
    emit_program([c_include_sys('cuda_runtime.h'), c_blank, K], Code),
    atom(Code),
    atom_length(Code, L), L > 100,
    sub_atom(Code, _, _, _, '__global__'),
    sub_atom(Code, _, _, _, 'loss_mse').

test_triplet_emits_compilable_cuda :-
    generate_kernel_loss(ggml_triplet_margin_loss, mean, [], K),
    emit_program([c_include_sys('cuda_runtime.h'), c_blank, K], Code),
    atom(Code),
    sub_atom(Code, _, _, _, 'loss_triplet_margin'),
    sub_atom(Code, _, _, _, 'anchor'),
    sub_atom(Code, _, _, _, 'margin').

%% ─────────────────────────────────────────────────────────────────────
%% Helpers
%% ─────────────────────────────────────────────────────────────────────

contains_in_term(Term, Pattern) :-
    Term = Pattern, !.
contains_in_term(Term, Pattern) :-
    compound(Term),
    Term =.. [_ | Args],
    member(A, Args),
    contains_in_term(A, Pattern), !.
contains_in_term([H | _], Pattern) :-
    contains_in_term(H, Pattern), !.
contains_in_term([_ | T], Pattern) :-
    contains_in_term(T, Pattern).

count_calls_to(Term, FName, N) :-
    count_calls_helper(Term, FName, 0, N).

count_calls_helper(c_call(FName, _), FName, Acc, Acc1) :- !, Acc1 is Acc + 1.
count_calls_helper(T, FName, Acc, Out) :-
    compound(T),
    T =.. [_ | Args],
    !,
    count_calls_list(Args, FName, Acc, Out).
count_calls_helper([H | T], FName, Acc, Out) :- !,
    count_calls_helper(H, FName, Acc, Acc1),
    count_calls_helper(T, FName, Acc1, Out).
count_calls_helper(_, _, Acc, Acc).

count_calls_list([], _, Acc, Acc).
count_calls_list([A | Rest], FName, Acc, Out) :-
    count_calls_helper(A, FName, Acc, Acc1),
    count_calls_list(Rest, FName, Acc1, Out).

:- initialization(run_tests, main).
