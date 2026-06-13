%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_fusion_profitability_gate.pl
%% Verifies that iterative_fusion's apply path gates on fusion_profitable:
%%   - VALID-but-unprofitable generator-prologue fusions (im2col) are DECLINED
%%   - always-profitable rules (elementwise/epilogue/layout) still apply
%%   - cost-dependent generator fusions follow the calibrated cost model
%% This closes the loop on the im2col conv finding: validity != profitability.

:- use_module('../lib/iterative_fusion').
:- use_module('../lib/fusion_cost').
:- use_module(library(plunit)).

:- begin_tests(fusion_profitability_gate).

%% im2col -> GEMM at M=128 was MEASURED slower than 2-stage on the P4.
%% The gate must DECLINE it.
test(im2col_m128_declined) :-
    NElems is 576*93312,
    Facts = [op_kind(im2c, im2col), op_output_elems(im2c, NElems),
             op_gemm_m(gm, 128), op_gemm_bm(gm, 64)],
    F = fusion(generator_prologue, [im2c, gm], bit_exact),
    \+ iterative_fusion:fusion_is_profitable(Facts, F).

test(im2col_m512_declined) :-
    NElems is 2304*5408,
    Facts = [op_kind(im2c, im2col), op_output_elems(im2c, NElems),
             op_gemm_m(gm, 512), op_gemm_bm(gm, 64)],
    F = fusion(generator_prologue, [im2c, gm], bit_exact),
    \+ iterative_fusion:fusion_is_profitable(Facts, F).

%% always-profitable rules pass the gate unchanged.
test(elementwise_chain_allowed) :-
    iterative_fusion:fusion_is_profitable([], fusion(elementwise_chain, [a,b], bit_exact)).

test(epilogue_allowed) :-
    iterative_fusion:fusion_is_profitable([], fusion(epilogue_matmul_elementwise, [a,b], bit_exact)).

test(layout_transparent_allowed) :-
    iterative_fusion:fusion_is_profitable([], fusion(layout_transparent, [a,b], bit_exact)).

%% a cheap generator (broadcast) with low reload factor IS profitable.
test(broadcast_factor1_allowed) :-
    NElems is 576*93312,
    Facts = [op_kind(bc, broadcast), op_output_elems(bc, NElems),
             op_gemm_m(g, 64), op_gemm_bm(g, 64)],
    iterative_fusion:fusion_is_profitable(Facts, fusion(generator_prologue, [bc, g], bit_exact)).

%% same cheap generator but reloaded 8x (large M) becomes unprofitable.
test(broadcast_factor8_declined) :-
    NElems is 576*93312,
    Facts = [op_kind(bc, broadcast), op_output_elems(bc, NElems),
             op_gemm_m(g, 512), op_gemm_bm(g, 64)],
    \+ iterative_fusion:fusion_is_profitable(Facts, fusion(generator_prologue, [bc, g], bit_exact)).

:- end_tests(fusion_profitability_gate).
