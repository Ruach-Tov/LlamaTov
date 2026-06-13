%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_fusion_varied.pl — Test fusion analyzer on diverse KernelBench L2 patterns.

:- set_prolog_flag(double_quotes, codes).
:- use_module('../lib/fusion_analyzer').

test_all :-
    test_kb4,
    test_kb5, 
    test_kb27,
    test_kb32,
    test_kb62,
    test_kb70,
    test_transformer_ffn,
    test_transformer_attention.

%% KernelBench L2 #4: Conv2d → Mish → Mish
test_kb4 :-
    write('=== KB#4: Conv2d_Mish_Mish ==='), nl,
    Ops = [op(conv, ggml_mul_mat, 1),    % conv ≈ matmul for fusion purposes
           op(mish1, ggml_silu, 2),       % mish ≈ silu for classification
           op(mish2, ggml_silu, 3)],
    find_fusible_chains(Ops, Chains),
    print_chains(Chains),
    nl.

%% KernelBench L2 #5: ConvTranspose2d → Subtract → Tanh
test_kb5 :-
    write('=== KB#5: ConvTranspose2d_Subtract_Tanh ==='), nl,
    Ops = [op(conv_t, ggml_mul_mat, 1),
           op(subtract, ggml_sub, 2),
           op(tanh_op, ggml_tanh, 3)],
    find_fusible_chains(Ops, Chains),
    print_chains(Chains),
    nl.

%% KernelBench L2 #27: Conv3d → HardSwish → GroupNorm → Mean
test_kb27 :-
    write('=== KB#27: Conv3d_HardSwish_GroupNorm_Mean ==='), nl,
    Ops = [op(conv3d, ggml_mul_mat, 1),
           op(hardswish, ggml_silu, 2),    % hardswish ≈ silu
           op(group_norm, ggml_norm, 3),
           op(mean_op, ggml_mean, 4)],
    find_fusible_chains(Ops, Chains),
    print_chains(Chains),
    write('  Expected: conv+hardswish fusible, norm+mean NOT (reduction after norm)'), nl,
    nl.

%% KernelBench L2 #32: Conv2d → Scaling → Min
test_kb32 :-
    write('=== KB#32: Conv2d_Scaling_Min ==='), nl,
    Ops = [op(conv, ggml_mul_mat, 1),
           op(scale_op, ggml_scale, 2),
           op(min_op, ggml_sum_rows, 3)],  % min ≈ reduction
    find_fusible_chains(Ops, Chains),
    print_chains(Chains),
    nl.

%% KernelBench L2 #62: Matmul → GroupNorm → LeakyReLU → Sum
test_kb62 :-
    write('=== KB#62: Matmul_GroupNorm_LeakyReLU_Sum ==='), nl,
    Ops = [op(matmul, ggml_mul_mat, 1),
           op(gnorm, ggml_norm, 2),
           op(lrelu, ggml_relu, 3),
           op(sum_op, ggml_add, 4)],
    find_fusible_chains(Ops, Chains),
    print_chains(Chains),
    write('  Expected: matmul alone (can\'t fuse into norm),'), nl,
    write('            norm+relu+add fusible'), nl,
    nl.

%% KernelBench L2 #70: Gemm → Sigmoid → Scaling → ResidualAdd  
test_kb70 :-
    write('=== KB#70: Gemm_Sigmoid_Scaling_ResidualAdd ==='), nl,
    Ops = [op(gemm, ggml_mul_mat, 1),
           op(sigmoid, ggml_sigmoid, 2),
           op(scale_op, ggml_scale, 3),
           op(residual, ggml_add, 4)],
    find_fusible_chains(Ops, Chains),
    print_chains(Chains),
    write('  Expected: ALL FOUR fused (matmul + 3 elementwise epilogue)'), nl,
    nl.

%% Real transformer: FFN SwiGLU block
%% norm → gate_proj(matmul) → silu → up_proj(matmul) → mul → down_proj(matmul) → add
test_transformer_ffn :-
    write('=== Transformer FFN (SwiGLU) ==='), nl,
    Ops = [op(ffn_norm, ggml_rms_norm, 1),
           op(gate_proj, ggml_mul_mat, 2),
           op(silu_act, ggml_silu, 3),
           op(up_proj, ggml_mul_mat, 4),
           op(gate_mul, ggml_mul, 5),
           op(down_proj, ggml_mul_mat, 6),
           op(residual, ggml_add, 7)],
    find_fusible_chains(Ops, Chains),
    print_chains(Chains),
    write('  Expected: gate+silu fused, up alone,'), nl,
    write('            down+residual fused, norm standalone'), nl,
    nl.

%% Real transformer: full attention block
%% norm → Q(mm) → K(mm) → V(mm) → reshape×3 → rope×2 → attn(mm) → add
test_transformer_attention :-
    write('=== Transformer Attention Block ==='), nl,
    Ops = [op(attn_norm, ggml_rms_norm, 1),
           op(q_proj, ggml_mul_mat, 2),
           op(q_bias, ggml_add, 3),
           op(k_proj, ggml_mul_mat, 4),
           op(k_bias, ggml_add, 5),
           op(v_proj, ggml_mul_mat, 6),
           op(v_bias, ggml_add, 7),
           op(q_reshape, ggml_reshape_3d, 8),
           op(k_reshape, ggml_reshape_3d, 9),
           op(v_reshape, ggml_reshape_3d, 10),
           op(q_rope, ggml_rope_ext, 11),
           op(k_rope, ggml_rope_ext, 12),
           op(attn_scores, ggml_mul_mat, 13),
           op(attn_softmax, ggml_soft_max_ext, 14),
           op(attn_values, ggml_mul_mat, 15),
           op(out_proj, ggml_mul_mat, 16),
           op(residual, ggml_add, 17)],
    find_fusible_chains(Ops, Chains),
    print_chains(Chains),
    nl.

print_chains(Chains) :-
    forall(member(Chain, Chains), (
        reverse(Chain, Fwd),
        findall(N, member(op(N,_,_), Fwd), Names),
        length(Names, Len),
        format("  Chain (~d ops): ~w~n", [Len, Names])
    )).

:- initialization((test_all -> halt(0) ; (write('FAILED'), nl, halt(1)))).
