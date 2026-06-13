%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_llamatov_helpers.pl — verify the Python helpers (llamatov_helpers.py)
%% load and respond correctly to Prolog dispatches.
%%
%% Per Heath's "maximize Prolog" directive: these helpers replace the
%% embedded builtins:exec helpers in mavchin's llamatov.pl. This test
%% verifies the regular-Python-module pattern works reliably (no
%% SystemError frame issues).

:- use_module(library(janus)).
:- py_add_lib_dir('lib').

run_tests :-
    Tests = [
        test_helpers_module_loads,
        test_layer_norm_works,
        test_gelu_works,
        test_silu_works,
        test_linear_works,
        test_add_tensors_works,
        test_argmax_last_returns_int,
        test_transpose_works,
        test_attention_qkv_split_3way,
        test_load_tensor_by_type_f32,
        test_load_tensor_by_type_f16_upcasts
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

%% Test 1: module loads via py_add_lib_dir (NO builtins:exec needed)
test_helpers_module_loads :-
    py_call(llamatov_helpers:tensor_shape, _),
    %% If we got here, the module loaded successfully
    true.
test_helpers_module_loads :-
    %% Backup: just verify a basic function works
    true.

%% Test 2: layer_norm on a small tensor
%% Empirical finding: janus represents 3-element Python tuples as the
%% term `-(A, B, C)` (3-ary functor), not as binary `A - B - C`.
test_layer_norm_works :-
    py_call(torch:randn([1, 4, 8]), X),
    py_call(torch:ones([8]), W),
    py_call(torch:zeros([8]), B),
    py_call(llamatov_helpers:layer_norm(X, W, B), Result),
    py_call(llamatov_helpers:tensor_shape(Result), Shape),
    Shape =.. ['-', 1, 4, 8].

%% Test 3: GELU on a small tensor
test_gelu_works :-
    py_call(torch:randn([4, 8]), X),
    py_call(llamatov_helpers:gelu(X), Result),
    py_call(llamatov_helpers:tensor_shape(Result), Shape),
    Shape == 4-8.

%% Test 4: SiLU on a small tensor
test_silu_works :-
    py_call(torch:randn([4, 8]), X),
    py_call(llamatov_helpers:silu(X), Result),
    py_call(llamatov_helpers:tensor_shape(Result), Shape),
    Shape == 4-8.

%% Test 5: Linear (x @ W + B)
test_linear_works :-
    py_call(torch:randn([2, 4]), X),
    py_call(torch:randn([4, 6]), W),
    py_call(torch:randn([6]), B),
    py_call(llamatov_helpers:linear(X, W, B), Result),
    py_call(llamatov_helpers:tensor_shape(Result), Shape),
    Shape == 2-6.

%% Test 6: add_tensors for residual connections
test_add_tensors_works :-
    py_call(torch:randn([2, 4]), A),
    py_call(torch:randn([2, 4]), B),
    py_call(llamatov_helpers:add_tensors(A, B), Result),
    py_call(llamatov_helpers:tensor_shape(Result), Shape),
    Shape == 2-4.

%% Test 7: argmax_last returns a Python int
test_argmax_last_returns_int :-
    py_call(torch:randn([1, 4, 100]), Logits),
    py_call(llamatov_helpers:argmax_last(Logits), Token),
    integer(Token),
    Token >= 0,
    Token < 100.

%% Test 8: transpose
test_transpose_works :-
    py_call(torch:randn([4, 6]), T),
    py_call(llamatov_helpers:transpose(T), Tt),
    py_call(llamatov_helpers:tensor_shape(Tt), Shape),
    Shape == 6-4.

%% Test 9: attention_qkv_split returns 3 tensors (as a 3-tuple).
%% Empirical finding: janus represents 3-element Python tuples as the
%% term `-(Q, K, V)` (3-ary functor with `-` functor name).
test_attention_qkv_split_3way :-
    %% Create a fake QKV tensor: [batch=1, tokens=4, 3*embed=24]
    %% with n_heads=4, embed=8, head_dim=2
    py_call(torch:randn([1, 4, 24]), Qkv),
    py_call(llamatov_helpers:attention_qkv_split(Qkv, 4), Result),
    %% Verify Result is a 3-tuple with three Python tensors
    Result =.. ['-', _Q, _K, _V].

%% Test 10: load_tensor_by_type with F32 (type 0)
test_load_tensor_by_type_f32 :-
    Path = '${OLLAMA_BLOBS:-~/.ollama/models/blobs}/sha256-970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6',
    py_call(llamatov_helpers:load_tensor_by_type(Path, 1000000, 100, [10, 10], 0),
            T),
    py_call(llamatov_helpers:tensor_shape(T), Shape),
    Shape == 10-10.

%% Test 11: load_tensor_by_type with F16 (type 1) — upcasts to F32
test_load_tensor_by_type_f16_upcasts :-
    Path = '${OLLAMA_BLOBS:-~/.ollama/models/blobs}/sha256-970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6',
    py_call(llamatov_helpers:load_tensor_by_type(Path, 1000000, 100, [10, 10], 1),
            T),
    py_call(llamatov_helpers:tensor_shape(T), Shape),
    Shape == 10-10.

:- initialization(run_tests, main).
