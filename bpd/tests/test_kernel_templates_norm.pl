%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_kernel_templates_norm.pl — verify Family 2 (Normalizations).
%%
%% Tests generate_kernel_norm/4 for all 4 norm op variants × {affine, no-affine}.
%% Per mavchin's spec: arity 4 with Affine boolean parameter.

:- set_prolog_flag(double_quotes, codes).
:- use_module('../lib/c_ast').
:- use_module('../lib/kernel_templates').

run_tests :-
    Tests = [
        %% Dispatch by op kind
        test_dispatch_layer_norm,
        test_dispatch_l2_norm,
        test_dispatch_rms_norm,
        test_dispatch_group_norm,
        %% Kernel names
        test_kernel_name_layer,
        test_kernel_name_rms,
        test_kernel_name_l2,
        test_kernel_name_group,
        %% Affine vs no-affine parameter coverage
        test_layer_no_affine_has_5_params,
        test_layer_affine_has_7_params,
        test_rms_no_affine_has_5_params,
        test_rms_affine_has_7_params,
        %% Structural properties
        test_layer_has_two_pass_structure,
        test_rms_does_not_compute_mean,
        test_l2_does_not_normalize_by_N,
        test_group_uses_layer_stats,
        %% Statistics finalization includes sqrt
        test_layer_uses_sqrt_in_finalize,
        test_rms_uses_sqrt_in_finalize,
        test_l2_uses_sqrt_in_finalize,
        %% Emission
        test_layer_emits_compilable_cuda,
        test_rms_emits_compilable_cuda,
        test_l2_emits_compilable_cuda,
        test_layer_affine_includes_W_B_in_output
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

test_dispatch_layer_norm :- generate_kernel_norm(ggml_norm,       2, false, _).
test_dispatch_l2_norm    :- generate_kernel_norm(ggml_l2_norm,    2, false, _).
test_dispatch_rms_norm   :- generate_kernel_norm(ggml_rms_norm,   2, false, _).
test_dispatch_group_norm :- generate_kernel_norm(ggml_group_norm, 2, false, _).

%% ─────────────────────────────────────────────────────────────────────
%% Naming
%% ─────────────────────────────────────────────────────────────────────

test_kernel_name_layer :-
    generate_kernel_norm(ggml_norm, 2, false, K),
    K = c_func(_, _, norm_layer, _, _).

test_kernel_name_rms :-
    generate_kernel_norm(ggml_rms_norm, 2, false, K),
    K = c_func(_, _, norm_rms, _, _).

test_kernel_name_l2 :-
    generate_kernel_norm(ggml_l2_norm, 2, false, K),
    K = c_func(_, _, norm_l2, _, _).

test_kernel_name_group :-
    generate_kernel_norm(ggml_group_norm, 2, false, K),
    K = c_func(_, _, norm_group, _, _).

%% ─────────────────────────────────────────────────────────────────────
%% Affine vs no-affine parameter counts
%% ─────────────────────────────────────────────────────────────────────
%% Base params: X, Y, N, outer, eps (5)
%% With affine: + W, B (7 total)

test_layer_no_affine_has_5_params :-
    generate_kernel_norm(ggml_norm, 2, false, K),
    K = c_func(_, _, _, Params, _),
    length(Params, 5).

test_layer_affine_has_7_params :-
    generate_kernel_norm(ggml_norm, 2, true, K),
    K = c_func(_, _, _, Params, _),
    length(Params, 7).

test_rms_no_affine_has_5_params :-
    generate_kernel_norm(ggml_rms_norm, 2, false, K),
    K = c_func(_, _, _, Params, _),
    length(Params, 5).

test_rms_affine_has_7_params :-
    generate_kernel_norm(ggml_rms_norm, 2, true, K),
    K = c_func(_, _, _, Params, _),
    length(Params, 7).

%% ─────────────────────────────────────────────────────────────────────
%% Structural properties
%% ─────────────────────────────────────────────────────────────────────

test_layer_has_two_pass_structure :-
    %% Body should contain TWO c_for loops (pass 1: stats, pass 2: apply)
    generate_kernel_norm(ggml_norm, 2, false, K),
    K = c_func(_, _, _, _, Body),
    findall(F, member(F, Body), AllStmts),
    findall(L, (member(c_for(_, _, _, _), AllStmts)), Loops),
    %% This counts loops; should be exactly 2 top-level for-loops
    length(Loops, 2).

test_rms_does_not_compute_mean :-
    %% RMS norm has no `mean` variable declaration anywhere in the body
    generate_kernel_norm(ggml_rms_norm, 2, false, K),
    K = c_func(_, _, _, _, Body),
    \+ contains_decl_named(Body, mean).

test_l2_does_not_normalize_by_N :-
    %% L2 norm uses inv_norm, NOT inv_rms or inv_std
    generate_kernel_norm(ggml_l2_norm, 2, false, K),
    K = c_func(_, _, _, _, Body),
    contains_decl_named(Body, inv_norm),
    \+ contains_decl_named(Body, inv_rms),
    \+ contains_decl_named(Body, inv_std).

test_group_uses_layer_stats :-
    %% Group norm has mean variable (delegates to layer norm structure)
    generate_kernel_norm(ggml_group_norm, 2, false, K),
    K = c_func(_, _, _, _, Body),
    contains_decl_named(Body, mean).

%% ─────────────────────────────────────────────────────────────────────
%% Statistics finalization uses sqrt
%% ─────────────────────────────────────────────────────────────────────

test_layer_uses_sqrt_in_finalize :-
    generate_kernel_norm(ggml_norm, 2, false, K),
    K = c_func(_, _, _, _, Body),
    contains_call_to(Body, sqrtf).

test_rms_uses_sqrt_in_finalize :-
    generate_kernel_norm(ggml_rms_norm, 2, false, K),
    K = c_func(_, _, _, _, Body),
    contains_call_to(Body, sqrtf).

test_l2_uses_sqrt_in_finalize :-
    generate_kernel_norm(ggml_l2_norm, 2, false, K),
    K = c_func(_, _, _, _, Body),
    contains_call_to(Body, sqrtf).

%% ─────────────────────────────────────────────────────────────────────
%% Emission
%% ─────────────────────────────────────────────────────────────────────

test_layer_emits_compilable_cuda :-
    generate_kernel_norm(ggml_norm, 2, false, K),
    emit_program([c_include_sys('cuda_runtime.h'), c_blank, K], Code),
    atom(Code),
    atom_length(Code, L), L > 100,
    sub_atom(Code, _, _, _, '__global__'),
    sub_atom(Code, _, _, _, 'norm_layer'),
    sub_atom(Code, _, _, _, 'sqrtf').

test_rms_emits_compilable_cuda :-
    generate_kernel_norm(ggml_rms_norm, 2, false, K),
    emit_program([c_include_sys('cuda_runtime.h'), c_blank, K], Code),
    atom(Code),
    sub_atom(Code, _, _, _, 'norm_rms').

test_l2_emits_compilable_cuda :-
    generate_kernel_norm(ggml_l2_norm, 2, false, K),
    emit_program([c_include_sys('cuda_runtime.h'), c_blank, K], Code),
    atom(Code),
    sub_atom(Code, _, _, _, 'norm_l2').

test_layer_affine_includes_W_B_in_output :-
    %% Affine layer norm should reference W[i] and B[i] in the output computation
    generate_kernel_norm(ggml_norm, 2, true, K),
    emit_program([c_include_sys('cuda_runtime.h'), c_blank, K], Code),
    sub_atom(Code, _, _, _, 'W[i]'),
    sub_atom(Code, _, _, _, 'B[i]').

%% ─────────────────────────────────────────────────────────────────────
%% Helpers
%% ─────────────────────────────────────────────────────────────────────

contains_decl_named([], _) :- !, fail.
contains_decl_named([c_decl_init(_, Name, _) | _], Name) :- !.
contains_decl_named([c_for(_, _, _, Body) | Rest], Name) :-
    ( contains_decl_named(Body, Name) ; contains_decl_named(Rest, Name) ).
contains_decl_named([c_if(_, Body) | Rest], Name) :-
    ( contains_decl_named(Body, Name) ; contains_decl_named(Rest, Name) ).
contains_decl_named([_ | Rest], Name) :-
    contains_decl_named(Rest, Name).

contains_call_to(Term, FName) :-
    sub_term_contains_call(Term, FName), !.

sub_term_contains_call(c_call(FName, _), FName) :- !.
sub_term_contains_call(T, FName) :-
    compound(T),
    T =.. [_ | Args],
    member(A, Args),
    sub_term_contains_call(A, FName).

:- initialization(run_tests, main).
