%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_kernelbench_l3.pl — Run fusion analyzer on KernelBench L3 architectures.
%%
%% Reproduces the L3 analysis as committed, verifiable tests.
%% Each architecture expressed as op sequence, analyzed by fusion_analyzer.

:- set_prolog_flag(double_quotes, codes).
:- use_module('../lib/fusion_analyzer').

%% ═══════════════════════════════════════════════════════════════
%% L3 ARCHITECTURES AS OP SEQUENCES
%% ═══════════════════════════════════════════════════════════════

l3_problem(1, 'MLP', [
    op(fc1, ggml_mul_mat, 1), op(relu1, ggml_relu, 2),
    op(fc2, ggml_mul_mat, 3), op(relu2, ggml_relu, 4),
    op(fc3, ggml_mul_mat, 5)]).

l3_problem(8, 'ResNet_Basic_Block', [
    op(conv1, ggml_mul_mat, 1), op(bn1, ggml_norm, 2), op(relu1, ggml_relu, 3),
    op(conv2, ggml_mul_mat, 4), op(bn2, ggml_norm, 5),
    op(shortcut, ggml_add, 6), op(relu2, ggml_relu, 7)]).

l3_problem(28, 'Vision_Transformer', [
    op(patch_embed, ggml_mul_mat, 1), op(patch_norm, ggml_norm, 2),
    op(qkv_proj, ggml_mul_mat, 3), op(q_scale, ggml_scale, 4),
    op(attn_scores, ggml_mul_mat, 5), op(attn_scale, ggml_scale, 6),
    op(attn_softmax, ggml_soft_max_ext, 7), op(attn_values, ggml_mul_mat, 8),
    op(out_proj, ggml_mul_mat, 9), op(attn_residual, ggml_add, 10),
    op(ffn_norm, ggml_norm, 11), op(ffn_up, ggml_mul_mat, 12),
    op(ffn_gelu, ggml_gelu, 13), op(ffn_down, ggml_mul_mat, 14),
    op(ffn_residual, ggml_add, 15)]).

l3_problem(43, 'MiniGPT_Causal_Attention', [
    op(qkv_proj, ggml_mul_mat, 1),
    op(q_reshape, ggml_reshape_3d, 2), op(k_reshape, ggml_reshape_3d, 3),
    op(v_reshape, ggml_reshape_3d, 4),
    op(attn_scores, ggml_mul_mat, 5), op(attn_scale, ggml_scale, 6),
    op(attn_mask, ggml_add, 7), op(attn_softmax, ggml_soft_max_ext, 8),
    op(attn_values, ggml_mul_mat, 9),
    op(out_proj, ggml_mul_mat, 10), op(residual, ggml_add, 11)]).

%% ═══════════════════════════════════════════════════════════════
%% TESTS WITH ASSERTIONS
%% ═══════════════════════════════════════════════════════════════

:- dynamic pass_count/1, fail_count/1.
pass_count(0). fail_count(0).

inc(Name) :- retract(Name), Name =.. [F,N], N1 is N+1, New =.. [F,N1], assert(New).

assert_reduction(TestName, Num, ExpectedMax) :-
    l3_problem(Num, _, Ops),
    length(Ops, TotalOps),
    find_fusible_chains(Ops, Chains),
    include([C]>>(length(C,L),L>1), Chains, Multi),
    findall(L, (member(C,Multi),length(C,L)), Lens),
    sumlist(Lens, Fused),
    length(Multi, NChains),
    Launches is NChains + (TotalOps - Fused),
    ( Launches =< ExpectedMax ->
        inc(pass_count(0)),
        Pct is (TotalOps - Launches) * 100 // TotalOps,
        format("  PASS ~w: ~d ops -> ~d launches (~d% reduction)~n", [TestName, TotalOps, Launches, Pct])
    ;
        inc(fail_count(0)),
        format("  FAIL ~w: ~d launches > expected max ~d~n", [TestName, Launches, ExpectedMax])
    ).

run_all :-
    write('=== KernelBench L3 Fusion Analysis ==='), nl, nl,
    test_l3(1, 'L3#1_MLP', 3),
    test_l3(8, 'L3#8_ResNet', 5),
    test_l3(28, 'L3#28_ViT', 10),
    test_l3(43, 'L3#43_MiniGPT', 4),
    nl, write('ALL L3 TESTS PASSED'), nl.

test_l3(Num, TestName, ExpectedMax) :-
    l3_problem(Num, _, Ops),
    length(Ops, TotalOps),
    find_fusible_chains(Ops, Chains),
    include([C]>>(length(C,L),L>1), Chains, Multi),
    findall(L, (member(C,Multi),length(C,L)), Lens),
    sumlist(Lens, Fused),
    length(Multi, NChains),
    Launches is NChains + (TotalOps - Fused),
    Pct is (TotalOps - Launches) * 100 // TotalOps,
    format("  ~w: ~d ops -> ~d launches (~d% reduction)~n", [TestName, TotalOps, Launches, Pct]),
    ( Launches =< ExpectedMax -> true
    ; format("  FAIL: ~d > ~d~n", [Launches, ExpectedMax]), fail
    ).

:- initialization((run_all -> halt(0) ; (write('FAILED'), nl, halt(1)))).
