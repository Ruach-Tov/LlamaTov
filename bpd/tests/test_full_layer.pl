%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_full_layer.pl — S4: Full transformer layer composition
%%
%% Wires S1(QKV) + S2(attention) + S3(FFN) + residuals into
%% a complete transformer layer and runs fusion analysis.

:- set_prolog_flag(double_quotes, codes).
:- use_module('../lib/fusion_analyzer').

%% ═══════════════════════════════════════════════════════════════
%% S4: Full Qwen2 transformer layer as op sequence
%% ═══════════════════════════════════════════════════════════════

full_layer_ops([
    %% S1: QKV block
    op(attn_norm, ggml_rms_norm, 1),
    op(wq_proj, ggml_mul_mat, 2),
    op(wq_bias, ggml_add, 3),
    op(wk_proj, ggml_mul_mat, 4),
    op(wk_bias, ggml_add, 5),
    op(wv_proj, ggml_mul_mat, 6),
    op(wv_bias, ggml_add, 7),
    op(q_reshape, ggml_reshape_3d, 8),
    op(k_reshape, ggml_reshape_3d, 9),
    op(v_reshape, ggml_reshape_3d, 10),
    op(q_rope, ggml_rope_ext, 11),
    op(k_rope, ggml_rope_ext, 12),
    
    %% S2: Attention block
    op(qk_matmul, ggml_mul_mat, 13),
    op(qk_scale, ggml_scale, 14),
    op(attn_softmax, ggml_soft_max_ext, 15),
    op(sv_matmul, ggml_mul_mat, 16),
    op(out_proj, ggml_mul_mat, 17),
    op(attn_residual, ggml_add, 18),
    
    %% S3: FFN block
    op(ffn_norm, ggml_rms_norm, 19),
    op(gate_proj, ggml_mul_mat, 20),
    op(silu_act, ggml_silu, 21),
    op(up_proj, ggml_mul_mat, 22),
    op(gate_mul, ggml_mul, 23),
    op(down_proj, ggml_mul_mat, 24),
    op(ffn_residual, ggml_add, 25)
]).

test_full_layer :-
    write('═══════════════════════════════════════════════════'), nl,
    write('S4: Full Transformer Layer Fusion Analysis'), nl,
    write('═══════════════════════════════════════════════════'), nl, nl,
    
    full_layer_ops(Ops),
    length(Ops, TotalOps),
    format("Total operations: ~d~n~n", [TotalOps]),
    
    find_fusible_chains(Ops, Chains),
    include([C]>>(length(C, L), L > 1), Chains, FusionChains),
    length(FusionChains, NumChains),
    
    write('Fusible chains found:'), nl,
    forall(member(Chain, FusionChains), (
        reverse(Chain, Fwd),
        findall(N, member(op(N,_,_), Fwd), Names),
        length(Fwd, Len),
        format("  [~d ops] ~w~n", [Len, Names])
    )),
    nl,
    
    %% Compute savings
    findall(L, (member(C, FusionChains), length(C, L)), Lens),
    sumlist(Lens, FusedOps),
    TotalKernels is NumChains + (TotalOps - FusedOps),
    Eliminated is TotalOps - TotalKernels,
    
    write('═══════════════════════════════════════════════════'), nl,
    write('FUSION SUMMARY FOR ONE TRANSFORMER LAYER'), nl,
    write('═══════════════════════════════════════════════════'), nl,
    format("  Operations:          ~d~n", [TotalOps]),
    format("  Fused chains:        ~d~n", [NumChains]),
    format("  Kernel launches:     ~d (was ~d)~n", [TotalKernels, TotalOps]),
    format("  Eliminated launches: ~d~n", [Eliminated]),
    format("  Reduction:           ~d%~n", [Eliminated * 100 // TotalOps]),
    nl,
    
    write('PER-MODEL SAVINGS (32 layers):'), nl,
    TotalElim is Eliminated * 32,
    TotalLaunches is TotalKernels * 32,
    TotalUnfused is TotalOps * 32,
    format("  Without fusion: ~d kernel launches~n", [TotalUnfused]),
    format("  With fusion:    ~d kernel launches~n", [TotalLaunches]),
    format("  Eliminated:     ~d VRAM round-trips~n", [TotalElim]),
    format("  At 0.1ms per round-trip: ~1fms saved per forward pass~n",
           [TotalElim * 0.1]),
    nl,
    
    write('SECTION BREAKDOWN:'), nl,
    %% Count fusions per section
    count_section_fusions(FusionChains, 1, 12, S1Fused),
    count_section_fusions(FusionChains, 13, 18, S2Fused),
    count_section_fusions(FusionChains, 19, 25, S3Fused),
    format("  S1 (QKV):       ~d ops fused~n", [S1Fused]),
    format("  S2 (Attention): ~d ops fused~n", [S2Fused]),
    format("  S3 (FFN):       ~d ops fused~n", [S3Fused]),
    nl.

count_section_fusions(Chains, MinSeq, MaxSeq, Count) :-
    findall(1, (
        member(Chain, Chains),
        member(op(_, _, Seq), Chain),
        Seq >= MinSeq, Seq =< MaxSeq
    ), Ones),
    length(Ones, Count).

:- initialization((test_full_layer -> halt(0) ; (write('FAILED'), nl, halt(1)))).
