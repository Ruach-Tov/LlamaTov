%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_kernel_templates_conv.pl — verify Family 4 (Convolutions) emission.
%%
%% Tests the generate_kernel_im2col/4 entry point for all 6 conv variants
%% (1D/2D/3D × forward/transpose). The 2D forward case has the full body
%% emitted; other variants are skeletons (full body in B2-Conv-Extended).

:- set_prolog_flag(double_quotes, codes).
:- use_module('../lib/c_ast').
:- use_module('../lib/kernel_templates').

run_tests :-
    Tests = [
        test_im2col_2d_forward_returns_kernel,
        test_im2col_2d_forward_has_correct_name,
        test_im2col_2d_forward_has_required_params,
        test_im2col_2d_forward_emits_compilable_cuda,
        test_im2col_2d_forward_includes_bounds_check,
        test_im2col_2d_forward_includes_triple_loop,
        test_im2col_2d_forward_includes_im2col_write,
        test_im2col_1d_forward_returns_skeleton,
        test_im2col_3d_forward_returns_skeleton,
        test_col2im_2d_transpose_returns_skeleton,
        test_dispatch_by_opkind_conv_2d,
        test_dispatch_by_opkind_conv_transpose_3d
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
%% 2D forward (the substantive case)
%% ─────────────────────────────────────────────────────────────────────

test_im2col_2d_forward_returns_kernel :-
    generate_kernel_im2col(ggml_conv_2d, 2, forward, Kernel),
    Kernel = c_func(_Qualifiers, _ReturnType, _Name, _Params, _Body).

test_im2col_2d_forward_has_correct_name :-
    generate_kernel_im2col(ggml_conv_2d, 2, forward, Kernel),
    Kernel = c_func(_, _, Name, _, _),
    Name == im2col_2d_forward.

test_im2col_2d_forward_has_required_params :-
    generate_kernel_im2col(ggml_conv_2d, 2, forward, Kernel),
    Kernel = c_func(_, _, _, Params, _),
    %% Must have X (input), Y (output), and dim/kernel/stride/pad/dilation params
    member(param(_, 'X'), Params),
    member(param(_, 'Y'), Params),
    member(param(_, 'kH'), Params),
    member(param(_, 'kW'), Params),
    member(param(_, stride_h), Params),
    member(param(_, stride_w), Params),
    member(param(_, pad_h), Params),
    member(param(_, pad_w), Params),
    member(param(_, dilation_h), Params),
    member(param(_, dilation_w), Params).

test_im2col_2d_forward_emits_compilable_cuda :-
    %% Verify that emit_program produces non-trivial CUDA text mentioning
    %% the kernel name and __global__ qualifier.
    %% emit_program returns an atom (via atom_codes in c_ast.pl).
    generate_kernel_im2col(ggml_conv_2d, 2, forward, Kernel),
    Program = [
        c_include_sys('cuda_runtime.h'),
        c_blank,
        Kernel
    ],
    emit_program(Program, Code),
    atom(Code),
    atom_length(Code, L),
    L > 100,
    sub_atom(Code, _, _, _, '__global__'),
    sub_atom(Code, _, _, _, 'im2col_2d_forward').

test_im2col_2d_forward_includes_bounds_check :-
    generate_kernel_im2col(ggml_conv_2d, 2, forward, Kernel),
    Kernel = c_func(_, _, _, _, Body),
    %% Body must include a c_if guarding against out-of-bounds
    member(c_if(_, _), Body).

test_im2col_2d_forward_includes_triple_loop :-
    %% Body must include the triple-nested for (over c, kh, kw)
    generate_kernel_im2col(ggml_conv_2d, 2, forward, Kernel),
    Kernel = c_func(_, _, _, _, Body),
    member(c_for(_, _, _, _), Body),
    %% Drill down to verify nesting
    nth0(_, Body, c_for(_, _, _, InnerStmts1)),
    member(c_for(_, _, _, InnerStmts2), InnerStmts1),
    member(c_for(_, _, _, _), InnerStmts2).

test_im2col_2d_forward_includes_im2col_write :-
    %% The im2col layout write to Y must be present somewhere in the body
    generate_kernel_im2col(ggml_conv_2d, 2, forward, Kernel),
    Kernel = c_func(_, _, _, _, Body),
    contains_assignment_to(Body, 'Y').

%% ─────────────────────────────────────────────────────────────────────
%% Other dim variants — skeleton only
%% ─────────────────────────────────────────────────────────────────────

test_im2col_1d_forward_returns_skeleton :-
    generate_kernel_im2col(ggml_conv_1d, 1, forward, Kernel),
    Kernel = c_func(_, _, im2col_1d_forward, Params, _Body),
    member(param(_, 'L'), Params).

test_im2col_3d_forward_returns_skeleton :-
    generate_kernel_im2col(ggml_conv_3d, 3, forward, Kernel),
    Kernel = c_func(_, _, im2col_3d_forward, Params, _Body),
    member(param(_, 'D'), Params).

test_col2im_2d_transpose_returns_skeleton :-
    generate_kernel_im2col(ggml_conv_transpose_2d, 2, transpose, Kernel),
    Kernel = c_func(_, _, col2im_2d_transpose, _, _).

%% ─────────────────────────────────────────────────────────────────────
%% Dispatch by op-kind: ensure every L1 conv op-kind is reachable
%% ─────────────────────────────────────────────────────────────────────

test_dispatch_by_opkind_conv_2d :-
    generate_kernel_im2col(ggml_conv_2d, 2, forward, _).

test_dispatch_by_opkind_conv_transpose_3d :-
    generate_kernel_im2col(ggml_conv_transpose_3d, 3, transpose, _).

%% ─────────────────────────────────────────────────────────────────────
%% Helpers
%% ─────────────────────────────────────────────────────────────────────

%% emit_program returns a list of codes; check for a substring.
%% Codes can be a list of char codes or a string depending on settings;
%% we accept both and check via string_chars.
sub_atom_or_codes_contains(Code, Substring) :-
    ( string(Code)
    -> sub_string(Code, _, _, _, Substring)
    ;  string_codes(S, Code),
       sub_string(S, _, _, _, Substring)
    ).

%% contains_assignment_to(+Body, +VarName)
%%   Recursively search Body for an assignment whose LHS indexes VarName.
contains_assignment_to([], _) :- !, fail.
contains_assignment_to([Stmt | Rest], VarName) :-
    ( stmt_assigns_to(Stmt, VarName)
    -> true
    ;  contains_assignment_to(Rest, VarName)
    ).

stmt_assigns_to(c_assign(c_index(c_var(VarName), _), _), VarName) :- !.
stmt_assigns_to(c_for(_, _, _, Body), VarName) :-
    contains_assignment_to(Body, VarName).
stmt_assigns_to(c_if(_, Body), VarName) :-
    contains_assignment_to(Body, VarName).

:- initialization(run_tests, main).
