%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_kernel_templates_pool.pl — verify Family 3 (Pooling) emission.
%%
%% Tests generate_kernel_pool/5 for all 6 pool variants (1D/2D/3D × max/avg).
%% 2D variants have full bodies; 1D and 3D are skeletons (B2-Pool-Extended).

:- set_prolog_flag(double_quotes, codes).
:- use_module('../lib/c_ast').
:- use_module('../lib/kernel_templates').

run_tests :-
    Tests = [
        %% Dispatch
        test_dispatch_pool_2d_max,
        test_dispatch_pool_2d_avg,
        test_dispatch_pool_1d_max,
        test_dispatch_pool_1d_avg,
        test_dispatch_pool_3d_max,
        test_dispatch_pool_3d_avg,
        %% Naming
        test_kernel_name_pool_2d_max,
        test_kernel_name_pool_2d_avg,
        %% Parameter shapes (2D)
        test_pool_2d_has_all_dim_params,
        %% Init value differs by pool kind
        test_max_uses_negative_infinity_init,
        test_avg_uses_zero_init,
        %% Accumulator differs by pool kind
        test_max_uses_ternary_accumulate,
        test_avg_uses_addition_accumulate,
        %% Finalize differs by pool kind
        test_max_finalize_is_acc_only,
        test_avg_finalize_divides_by_count,
        %% 2D body has window loop
        test_pool_2d_has_nested_window_loop,
        test_pool_2d_has_bounds_check,
        test_pool_2d_writes_output_with_correct_index,
        %% Emission
        test_pool_2d_max_emits_compilable_cuda,
        test_pool_2d_avg_emits_compilable_cuda,
        %% Indexing correctness
        test_emitted_input_index_has_proper_parens,
        test_emitted_output_index_has_proper_parens
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

test_dispatch_pool_2d_max :- generate_kernel_pool(ggml_pool_2d, 2, max, [], _).
test_dispatch_pool_2d_avg :- generate_kernel_pool(ggml_pool_2d, 2, avg, [], _).
test_dispatch_pool_1d_max :- generate_kernel_pool(ggml_pool_1d, 1, max, [], _).
test_dispatch_pool_1d_avg :- generate_kernel_pool(ggml_pool_1d, 1, avg, [], _).
test_dispatch_pool_3d_max :- generate_kernel_pool(ggml_pool_3d, 3, max, [], _).
test_dispatch_pool_3d_avg :- generate_kernel_pool(ggml_pool_3d, 3, avg, [], _).

%% ─────────────────────────────────────────────────────────────────────
%% Naming
%% ─────────────────────────────────────────────────────────────────────

test_kernel_name_pool_2d_max :-
    generate_kernel_pool(ggml_pool_2d, 2, max, [], K),
    K = c_func(_, _, pool_2d_max, _, _).

test_kernel_name_pool_2d_avg :-
    generate_kernel_pool(ggml_pool_2d, 2, avg, [], K),
    K = c_func(_, _, pool_2d_avg, _, _).

%% ─────────────────────────────────────────────────────────────────────
%% Parameter shapes (2D should have all required dim params)
%% ─────────────────────────────────────────────────────────────────────

test_pool_2d_has_all_dim_params :-
    generate_kernel_pool(ggml_pool_2d, 2, max, [], K),
    K = c_func(_, _, _, Params, _),
    member(param(_, 'B'), Params),
    member(param(_, 'C'), Params),
    member(param(_, 'inH'), Params),
    member(param(_, 'inW'), Params),
    member(param(_, 'outH'), Params),
    member(param(_, 'outW'), Params),
    member(param(_, 'kH'), Params),
    member(param(_, 'kW'), Params),
    member(param(_, stride_h), Params),
    member(param(_, stride_w), Params),
    member(param(_, pad_h), Params),
    member(param(_, pad_w), Params).

%% ─────────────────────────────────────────────────────────────────────
%% Init value differs by pool kind
%% ─────────────────────────────────────────────────────────────────────

test_max_uses_negative_infinity_init :-
    generate_kernel_pool(ggml_pool_2d, 2, max, [], K),
    K = c_func(_, _, _, _, Body),
    member(c_decl_init(c_type(float), acc, c_float(N)), Body),
    N < -1.0e29.

test_avg_uses_zero_init :-
    generate_kernel_pool(ggml_pool_2d, 2, avg, [], K),
    K = c_func(_, _, _, _, Body),
    member(c_decl_init(c_type(float), acc, c_float(0.0)), Body).

%% ─────────────────────────────────────────────────────────────────────
%% Accumulator differs by pool kind
%% ─────────────────────────────────────────────────────────────────────

test_max_uses_ternary_accumulate :-
    generate_kernel_pool(ggml_pool_2d, 2, max, [], K),
    K = c_func(_, _, _, _, Body),
    contains_in_term(Body,
        c_assign(c_var(acc), c_ternary(_, _, _))).

test_avg_uses_addition_accumulate :-
    generate_kernel_pool(ggml_pool_2d, 2, avg, [], K),
    K = c_func(_, _, _, _, Body),
    contains_in_term(Body,
        c_assign(c_var(acc), c_binop('+', c_var(acc), c_var(v)))).

%% ─────────────────────────────────────────────────────────────────────
%% Finalize differs by pool kind
%% ─────────────────────────────────────────────────────────────────────

test_max_finalize_is_acc_only :-
    %% Max: Y[idx] = acc (no division)
    generate_kernel_pool(ggml_pool_2d, 2, max, [], K),
    K = c_func(_, _, _, _, Body),
    last(Body, c_assign(_, c_var(acc))).

test_avg_finalize_divides_by_count :-
    %% Avg: Y[idx] = acc / (float)count
    generate_kernel_pool(ggml_pool_2d, 2, avg, [], K),
    K = c_func(_, _, _, _, Body),
    last(Body, c_assign(_, c_binop('/', _, _))).

%% ─────────────────────────────────────────────────────────────────────
%% 2D body has the right window structure
%% ─────────────────────────────────────────────────────────────────────

test_pool_2d_has_nested_window_loop :-
    %% Body has c_for (over kh) containing c_for (over kw)
    generate_kernel_pool(ggml_pool_2d, 2, max, [], K),
    K = c_func(_, _, _, _, Body),
    member(c_for(_, _, _, OuterBody), Body),
    member(c_for(_, _, _, _), OuterBody).

test_pool_2d_has_bounds_check :-
    %% Body has an early-return guard
    generate_kernel_pool(ggml_pool_2d, 2, max, [], K),
    K = c_func(_, _, _, _, Body),
    member(c_if(_, [c_return_void]), Body).

test_pool_2d_writes_output_with_correct_index :-
    %% The last statement assigns to Y[...] (the output)
    generate_kernel_pool(ggml_pool_2d, 2, max, [], K),
    K = c_func(_, _, _, _, Body),
    last(Body, c_assign(c_index(c_var('Y'), _), _)).

%% ─────────────────────────────────────────────────────────────────────
%% Emission
%% ─────────────────────────────────────────────────────────────────────

test_pool_2d_max_emits_compilable_cuda :-
    generate_kernel_pool(ggml_pool_2d, 2, max, [], K),
    emit_program([c_include_sys('cuda_runtime.h'), c_blank, K], Code),
    atom(Code),
    atom_length(Code, L), L > 100,
    sub_atom(Code, _, _, _, '__global__'),
    sub_atom(Code, _, _, _, 'pool_2d_max').

test_pool_2d_avg_emits_compilable_cuda :-
    generate_kernel_pool(ggml_pool_2d, 2, avg, [], K),
    emit_program([c_include_sys('cuda_runtime.h'), c_blank, K], Code),
    atom(Code),
    sub_atom(Code, _, _, _, 'pool_2d_avg'),
    sub_atom(Code, _, _, _, '(float)(count)').

%% ─────────────────────────────────────────────────────────────────────
%% Indexing parens (per the empirical c_paren wrap pattern)
%% ─────────────────────────────────────────────────────────────────────

test_emitted_input_index_has_proper_parens :-
    %% X[((b * C + c) * inH + ih) * inW + iw] should have correct parens
    generate_kernel_pool(ggml_pool_2d, 2, max, [], K),
    emit_program([c_include_sys('cuda_runtime.h'), c_blank, K], Code),
    sub_atom(Code, _, _, _, '((b * C + c) * inH + ih) * inW + iw').

test_emitted_output_index_has_proper_parens :-
    %% Y[((b * C + c) * outH + oh) * outW + ow] should have correct parens
    generate_kernel_pool(ggml_pool_2d, 2, max, [], K),
    emit_program([c_include_sys('cuda_runtime.h'), c_blank, K], Code),
    sub_atom(Code, _, _, _, '((b * C + c) * outH + oh) * outW + ow').

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

:- initialization(run_tests, main).
