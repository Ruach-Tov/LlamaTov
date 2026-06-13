%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_fusion.pl — Test the fusion analyzer on the QKV compute graph.

:- set_prolog_flag(double_quotes, codes).
:- use_module('../lib/fusion_analyzer').

test_qkv_fusion :-
    %% The QKV graph from qkv.bpd, expressed as op/3 terms
    Ops = [
        op(qkv_norm, build_norm(rms), 1),
        op(wq_mul, build_lora_mm, 2),
        op(wq_bias, ggml_add, 3),
        op(wk_mul, build_lora_mm, 4),
        op(wk_bias, ggml_add, 5),
        op(wv_mul, build_lora_mm, 6),
        op(wv_bias, ggml_add, 7),
        op(q_reshape, ggml_reshape_3d, 8),
        op(k_reshape, ggml_reshape_3d, 9),
        op(v_reshape, ggml_reshape_3d, 10),
        op(q_rope, ggml_rope_ext, 11),
        op(k_rope, ggml_rope_ext, 12)
    ],
    
    write('=== Operation Classifications ==='), nl,
    forall(member(op(Name, Kind, _), Ops), (
        ( classify_op(Kind, Class) ->
            format("  ~w: ~w (~w)~n", [Name, Kind, Class])
        ;
            format("  ~w: ~w (UNCLASSIFIED)~n", [Name, Kind])
        )
    )),
    nl,
    
    write('=== Pairwise Fusion Analysis ==='), nl,
    forall((member(op(N1, K1, S1), Ops), member(op(N2, K2, S2), Ops), S2 =:= S1 + 1), (
        ( can_fuse(K1, K2, Reason) ->
            format("  ~w -> ~w: FUSIBLE (~w)~n", [N1, N2, Reason])
        ; cannot_fuse(K1, K2, Reason) ->
            format("  ~w -> ~w: BLOCKED (~w)~n", [N1, N2, Reason])
        ;
            format("  ~w -> ~w: no rule~n", [N1, N2])
        )
    )),
    nl,
    
    write('=== Fusible Chains ==='), nl,
    find_fusible_chains(Ops, Chains),
    forall(member(Chain, Chains), (
        reverse(Chain, Fwd),
        maplist([op(N,_,_)]>>true, Fwd),
        findall(N, member(op(N,_,_), Fwd), Names),
        format("  Chain: ~w~n", [Names])
    )),
    nl,
    
    %% Test KernelBench L2 #70
    write('=== KernelBench L2 #70: Gemm_Sigmoid_Scaling_ResidualAdd ==='), nl,
    KB70 = [op(gemm, ggml_mul_mat, 1),
            op(sigmoid, ggml_sigmoid, 2),
            op(scale, ggml_scale, 3),
            op(residual, ggml_add, 4)],
    find_fusible_chains(KB70, KB70Chains),
    forall(member(Chain, KB70Chains), (
        reverse(Chain, Fwd),
        findall(N, member(op(N,_,_), Fwd), Names),
        format("  Chain: ~w~n", [Names])
    )).

:- initialization((test_qkv_fusion -> halt(0) ; (write('FAILED'), nl, halt(1)))).
