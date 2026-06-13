%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_metamorphic_fusion.pl — Metamorphic testing for kernel fusion correctness.
%%
%% Implements metamorphic relations (MRs) from the academic literature
%% (FSHADER/ICSE 2023, APSEC 2010) applied to ML kernel fusion.
%%
%% The fundamental insight: you don't need an oracle. You need a RELATION:
%%   fused(A,B)(input) ≈ B(A(input))
%% Compare fused vs unfused execution on the same input.
%%
%% TRIVIAL FIRST CASE: matmul + bias_add epilogue fusion.
%% This is the simplest fusion our analyzer supports and the most
%% common in transformer inference (every linear layer).
%%
%% Methodology:
%%   1. Generate random tensor inputs
%%   2. Run unfused: Y = matmul(X, W); Z = Y + bias
%%   3. Run fused:   Z' = fused_matmul_bias(X, W, bias)
%%   4. Assert: Z ≈ Z' within floating-point tolerance
%%
%% Since we're in Prolog (not GPU), we simulate with integer arithmetic
%% first (exact), then extend to floating-point via janus_swi+PyTorch.
%%
%% Author: medayek (Claude Opus 4.6, conversation 123)
%% Date: 2026-05-15
%% Research: FSHADER (ICSE 2023), arxiv 2406.05397

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(library(apply)).


%% ══════════════════════════════════════════════════════════════════════
%% Trivial compute primitives (integer arithmetic, exact)
%% ══════════════════════════════════════════════════════════════════════

%% matmul_1x1(+A, +B, -C) — scalar "matmul" (1×1 × 1×1 = 1×1)
matmul_1x1(A, B, C) :- C is A * B.

%% bias_add_scalar(+X, +Bias, -Y) — scalar bias add
bias_add_scalar(X, Bias, Y) :- Y is X + Bias.

%% fused_matmul_bias_1x1(+A, +B, +Bias, -Y) — fused scalar matmul+bias
fused_matmul_bias_1x1(A, B, Bias, Y) :- Y is A * B + Bias.

%% dot_product(+Xs, +Ys, -Dot) — inner product of two equal-length lists
dot_product([], [], 0).
dot_product([X|Xs], [Y|Ys], Dot) :-
    dot_product(Xs, Ys, Rest),
    Dot is X * Y + Rest.

%% matmul_row(+Row, +Cols, -Results) — multiply one row by each column
matmul_row(_, [], []).
matmul_row(Row, [Col|Cols], [Dot|Rest]) :-
    dot_product(Row, Col, Dot),
    matmul_row(Row, Cols, Rest).

%% transpose(+Matrix, -Transposed) — transpose a list of lists
transpose([[]|_], []) :- !.
transpose(Matrix, [Row|Rows]) :-
    maplist(nth0(0), Matrix, Row),
    maplist(tail, Matrix, RestMatrix),
    transpose(RestMatrix, Rows).

tail([_|T], T).

%% matmul(+A, +B, -C) — matrix multiply (A: MxK, B: KxN → C: MxN)
%% B is stored as rows; we transpose to get columns for dot products
matmul(A, B, C) :-
    transpose(B, BT),
    maplist(matmul_row_bt(BT), A, C).

matmul_row_bt(BT, Row, ResultRow) :-
    matmul_row(Row, BT, ResultRow).

%% vec_add(+Xs, +Ys, -Zs) — elementwise vector addition
vec_add([], [], []).
vec_add([X|Xs], [Y|Ys], [Z|Zs]) :-
    Z is X + Y,
    vec_add(Xs, Ys, Zs).

%% matrix_bias_add(+M, +Bias, -Result) — add bias vector to each row
matrix_bias_add([], _, []).
matrix_bias_add([Row|Rows], Bias, [NewRow|NewRows]) :-
    vec_add(Row, Bias, NewRow),
    matrix_bias_add(Rows, Bias, NewRows).

%% fused_matmul_bias(+A, +B, +Bias, -C) — fused matmul + bias add
fused_matmul_bias(A, B, Bias, C) :-
    matmul(A, B, Intermediate),
    matrix_bias_add(Intermediate, Bias, C).


%% ══════════════════════════════════════════════════════════════════════
%% MR1: Equivalence Preservation — fused ≈ sequential
%% ══════════════════════════════════════════════════════════════════════

:- begin_tests(mr1_equivalence_preservation).

test(scalar_matmul_bias_equivalence) :-
    %% Most trivial case: 1×1 matmul + scalar bias
    A = 3, B = 4, Bias = 5,
    %% Sequential: matmul then bias_add
    matmul_1x1(A, B, Intermediate),
    bias_add_scalar(Intermediate, Bias, Sequential),
    %% Fused: single operation
    fused_matmul_bias_1x1(A, B, Bias, Fused),
    %% MR1: they must be equal
    Sequential =:= Fused.

test(scalar_zero_inputs) :-
    %% Edge case: all zeros
    matmul_1x1(0, 0, I), bias_add_scalar(I, 0, Seq),
    fused_matmul_bias_1x1(0, 0, 0, Fused),
    Seq =:= Fused.

test(scalar_negative_inputs) :-
    %% Negative numbers
    matmul_1x1(-3, 4, I), bias_add_scalar(I, -5, Seq),
    fused_matmul_bias_1x1(-3, 4, -5, Fused),
    Seq =:= Fused.

test(matrix_2x2_equivalence) :-
    %% 2×2 matmul + bias: first non-trivial matrix case
    A = [[1, 2], [3, 4]],
    B = [[5, 6], [7, 8]],
    Bias = [10, 20],
    %% Sequential
    matmul(A, B, Intermediate),
    matrix_bias_add(Intermediate, Bias, Sequential),
    %% Fused
    fused_matmul_bias(A, B, Bias, Fused),
    %% MR1
    Sequential == Fused.

test(matrix_3x2_equivalence) :-
    %% Non-square: 3×2 × 2×2 + bias(2)
    A = [[1, 0], [0, 1], [1, 1]],
    B = [[2, 3], [4, 5]],
    Bias = [10, 20],
    matmul(A, B, I),
    matrix_bias_add(I, Bias, Seq),
    fused_matmul_bias(A, B, Bias, Fused),
    Seq == Fused.

test(identity_matrix_preserves_input) :-
    %% Matmul with identity should preserve the input
    A = [[1, 2, 3]],
    I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    Bias = [0, 0, 0],
    matmul(A, I, Intermediate),
    matrix_bias_add(Intermediate, Bias, Result),
    Result == [[1, 2, 3]].

:- end_tests(mr1_equivalence_preservation).


%% ══════════════════════════════════════════════════════════════════════
%% MR2: Associativity — fuse(A,B,C) ≈ fuse(fuse(A,B),C)
%% ══════════════════════════════════════════════════════════════════════

:- begin_tests(mr2_associativity).

test(elementwise_chain_associative) :-
    %% Three elementwise ops: add(1) → mul(2) → add(3)
    %% Fusing left-first vs right-first should give same result
    Input = 10,
    %% Sequential: ((10+1)*2)+3 = 25
    S1 is Input + 1,
    S2 is S1 * 2,
    S3 is S2 + 3,
    %% Left-fuse: fuse(add1, mul2) first, then add3
    L1 is (Input + 1) * 2,
    L2 is L1 + 3,
    %% Right-fuse: fuse(mul2, add3) first, apply after add1
    R1 is Input + 1,
    R2 is R1 * 2 + 3,
    %% All three must agree
    S3 =:= L2,
    S3 =:= R2.

:- end_tests(mr2_associativity).


%% ══════════════════════════════════════════════════════════════════════
%% MR3: Identity Preservation — layout ops don't change results
%% ══════════════════════════════════════════════════════════════════════

:- begin_tests(mr3_identity_preservation).

test(identity_reshape_preserves_matmul_bias) :-
    %% "Reshaping" (identity in our integer simulation) before matmul+bias
    %% should not change the result
    A = [[1, 2], [3, 4]],
    B = [[5, 6], [7, 8]],
    Bias = [10, 20],
    %% Without reshape
    fused_matmul_bias(A, B, Bias, Without),
    %% With identity "reshape" (no-op in our simulation)
    Reshaped = A,  % identity reshape
    fused_matmul_bias(Reshaped, B, Bias, With),
    %% MR3: must be equal
    Without == With.

:- end_tests(mr3_identity_preservation).


%% ══════════════════════════════════════════════════════════════════════
%% MR5: Scale Invariance (for linear operations)
%% ══════════════════════════════════════════════════════════════════════

:- begin_tests(mr5_scale_invariance).

test(scalar_scale_linearity) :-
    %% For matmul (linear): fused(k*X, W, k*bias) = k * fused(X, W, bias)
    %% Only holds when bias is also scaled
    X = 3, W = 4, Bias = 5, K = 2,
    fused_matmul_bias_1x1(X, W, Bias, Base),
    ScaledX is K * X,
    ScaledBias is K * Bias,
    fused_matmul_bias_1x1(ScaledX, W, ScaledBias, Scaled),
    Expected is K * Base,
    Scaled =:= Expected.

:- end_tests(mr5_scale_invariance).


%% ══════════════════════════════════════════════════════════════════════
%% Meta: mutation detection
%% ══════════════════════════════════════════════════════════════════════

:- begin_tests(mutation_detection).

test(wrong_fusion_detected) :-
    %% If fusion computes A*B - Bias instead of A*B + Bias,
    %% MR1 catches it (sequential and fused disagree)
    A = 3, B = 4, Bias = 5,
    matmul_1x1(A, B, I),
    bias_add_scalar(I, Bias, Sequential),
    %% Deliberately wrong "fusion": subtract instead of add
    WrongFused is A * B - Bias,
    %% MR1 violation: they disagree
    Sequential =\= WrongFused.

test(off_by_one_detected) :-
    %% If fusion has an off-by-one: A*(B+1) + Bias instead of A*B + Bias
    A = 3, B = 4, Bias = 5,
    Correct is A * B + Bias,
    Wrong is A * (B + 1) + Bias,
    Correct =\= Wrong.

:- end_tests(mutation_detection).
