%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_llama_helpers.pl — verify Llama-family helpers in llamatov_helpers.py
%%
%% Tests the helpers needed for Llama-architecture inference: RMSNorm,
%% RoPE, SwiGLU FFN, GQA attention. These compose with the existing
%% GPT-2 helpers (layer_norm, gelu, linear with bias) to extend the
%% substrate to most modern transformer architectures.

:- use_module(library(janus)).
:- py_add_lib_dir('lib').

run_tests :-
    Tests = [
        test_rms_norm_preserves_shape,
        test_rms_norm_is_not_identity,
        test_linear_no_bias_shape,
        test_llama_qkv_split_mha,
        test_llama_qkv_split_gqa,
        test_precompute_rope_returns_two_tensors,
        test_rope_preserves_shape,
        test_rope_is_not_identity,
        test_expand_kv_for_gqa,
        test_llama_causal_attention_gqa,
        test_swiglu_ffn,
        test_embed_tokens_no_position
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

%% ────────────────────────────────────────────────────────────────────
%% Tests
%% ────────────────────────────────────────────────────────────────────

test_rms_norm_preserves_shape :-
    py_call(torch:randn([1, 4, 8]), X),
    py_call(torch:ones([8]), W),
    py_call(llamatov_helpers:rms_norm(X, W, 1.0e-5), Result),
    py_call(llamatov_helpers:tensor_shape(Result), Shape),
    Shape =.. ['-', 1, 4, 8].

test_rms_norm_is_not_identity :-
    %% A constant tensor of 2.0 should NOT come out as 2.0 (RMS would
    %% normalize to magnitude 1.0).
    py_call(torch:full([1, 4, 8], 2.0), X),
    py_call(torch:ones([8]), W),
    py_call(llamatov_helpers:rms_norm(X, W, 1.0e-5), Result),
    py_call(torch:mean(Result), MeanT),
    py_call(MeanT:item(), Mean),
    %% After RMS norm: each value should be ~1.0 (since RMS of constant 2.0 is 2.0,
    %% so x/RMS = 1.0). With weight=1.0, result is 1.0.
    abs(Mean - 1.0) < 0.01.

test_linear_no_bias_shape :-
    py_call(torch:randn([2, 4]), X),
    py_call(torch:randn([4, 6]), W),
    py_call(llamatov_helpers:linear_no_bias(X, W), R),
    py_call(llamatov_helpers:tensor_shape(R), Shape),
    Shape == 2-6.

test_llama_qkv_split_mha :-
    %% Multi-head attention: n_heads == n_heads_kv = 4
    %% Q, K, V each have shape [batch=1, tokens=8, n_heads*head_dim=32]
    py_call(torch:randn([1, 8, 32]), Q),
    py_call(torch:randn([1, 8, 32]), K),
    py_call(torch:randn([1, 8, 32]), V),
    py_call(llamatov_helpers:llama_qkv_split(Q, K, V, 4, 4), Tuple),
    Tuple =.. ['-', Q1, K1, V1],
    py_call(llamatov_helpers:tensor_shape(Q1), QS),
    %% Expected: [1, 4, 8, 8]
    QS =.. ['-', 1, 4, 8, 8],
    py_call(llamatov_helpers:tensor_shape(K1), QS),
    py_call(llamatov_helpers:tensor_shape(V1), QS).

test_llama_qkv_split_gqa :-
    %% GQA: n_heads=4, n_heads_kv=2
    %% head_dim = 32 / 4 = 8
    %% Q has [1, 8, 32], K and V have [1, 8, 16] (n_heads_kv * head_dim)
    py_call(torch:randn([1, 8, 32]), Q),
    py_call(torch:randn([1, 8, 16]), K),
    py_call(torch:randn([1, 8, 16]), V),
    py_call(llamatov_helpers:llama_qkv_split(Q, K, V, 4, 2), Tuple),
    Tuple =.. ['-', Q1, K1, _V1],
    py_call(llamatov_helpers:tensor_shape(Q1), QShape),
    py_call(llamatov_helpers:tensor_shape(K1), KShape),
    QShape =.. ['-', 1, 4, 8, 8],
    KShape =.. ['-', 1, 2, 8, 8].

test_precompute_rope_returns_two_tensors :-
    py_call(llamatov_helpers:precompute_rope_cos_sin(64, 32, 10000.0),
            Tuple),
    %% 2-tuple should be `Cos - Sin`
    Tuple = (Cos - Sin),
    py_call(llamatov_helpers:tensor_shape(Cos), CS),
    py_call(llamatov_helpers:tensor_shape(Sin), CS),
    %% Cos and Sin should both be [64, 16] (16 = head_dim/2)
    CS == 64-16.

test_rope_preserves_shape :-
    py_call(torch:randn([1, 4, 8, 16]), X),
    py_call(llamatov_helpers:precompute_rope_cos_sin(8, 16, 10000.0),
            Tuple),
    Tuple = (Cos - Sin),
    py_call(llamatov_helpers:apply_rope(X, Cos, Sin), R),
    py_call(llamatov_helpers:tensor_shape(R), Shape),
    Shape =.. ['-', 1, 4, 8, 16].

test_rope_is_not_identity :-
    %% RoPE should change the values (except at position 0 where cos=1, sin=0)
    py_call(torch:randn([1, 1, 4, 16]), X),
    py_call(llamatov_helpers:precompute_rope_cos_sin(4, 16, 10000.0),
            Tuple),
    Tuple = (Cos - Sin),
    py_call(llamatov_helpers:apply_rope(X, Cos, Sin), R),
    %% At least one element should differ — use torch.not_equal for element-wise
    py_call(torch:not_equal(X, R), DiffMask),
    py_call(torch:any(DiffMask), AnyDiff),
    py_call(AnyDiff:item(), AnyDiffVal),
    AnyDiffVal == @(true).

test_expand_kv_for_gqa :-
    %% K, V with 2 heads → expand to 4 heads (each KV head shared by 2 Q heads)
    py_call(torch:randn([1, 2, 8, 16]), K),
    py_call(torch:randn([1, 2, 8, 16]), V),
    py_call(llamatov_helpers:expand_kv_for_gqa(K, V, 4, 2), Tuple),
    Tuple = (KExp - VExp),
    py_call(llamatov_helpers:tensor_shape(KExp), Shape),
    Shape =.. ['-', 1, 4, 8, 16],
    py_call(llamatov_helpers:tensor_shape(VExp), Shape).

test_llama_causal_attention_gqa :-
    %% GQA attention: Q has 4 heads, K/V have 2 heads
    py_call(torch:randn([1, 4, 8, 16]), Q),
    py_call(torch:randn([1, 2, 8, 16]), K),
    py_call(torch:randn([1, 2, 8, 16]), V),
    py_call(llamatov_helpers:llama_causal_attention(Q, K, V, 4, 2), Out),
    py_call(llamatov_helpers:tensor_shape(Out), Shape),
    Shape =.. ['-', 1, 4, 8, 16].

test_swiglu_ffn :-
    %% x @ gate → silu → * (x @ up) → @ down
    %% x: [1, 4, 8], gate_w: [8, 16], up_w: [8, 16], down_w: [16, 8]
    py_call(torch:randn([1, 4, 8]), X),
    py_call(torch:randn([8, 16]), GateW),
    py_call(torch:randn([8, 16]), UpW),
    py_call(torch:randn([16, 8]), DownW),
    py_call(llamatov_helpers:swiglu_ffn(X, GateW, UpW, DownW), Out),
    py_call(llamatov_helpers:tensor_shape(Out), Shape),
    Shape =.. ['-', 1, 4, 8].

test_embed_tokens_no_position :-
    %% GGUF convention: token_embd stored as [embed_dim, vocab].
    %% Embed shape [8, 100]; tokens [5, 10, 7] → result [1, 3, 8]
    py_call(torch:randn([8, 100]), Embd),
    py_call(llamatov_helpers:embed_tokens_no_position(Embd, [5, 10, 7]), X),
    py_call(llamatov_helpers:tensor_shape(X), Shape),
    Shape =.. ['-', 1, 3, 8].

:- initialization(run_tests, main).
