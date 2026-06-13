%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% spike_load_real_tensor.pl
%%
%% Integration spike Stage 3: load a REAL tensor from a real GGUF file
%% via the llamatov_loader Python helper module, verify it's a single
%% Python object (not a Prolog list), run ops on it.
%%
%% This is the substrate piece mavchin needs for end-to-end inference.
%% By going through a Python helper that returns torch.from_numpy(arr),
%% we keep the tensor as ONE Python object — operations stay in Python
%% space rather than exploding through Prolog list conversion.
%%
%% Per Heath's bounded-as-urgent directive: produces the integration
%% piece mavchin requested for "GGUF in, token out" end-to-end work.

:- use_module(library(janus)).

%% Path to the helper module (relative to this script's directory)
:- py_add_lib_dir('lib').

%% Test file: nomic-embed-text 262 MB GGUF
gguf_test_file('${OLLAMA_BLOBS:-~/.ollama/models/blobs}/sha256-970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6').

run_tests :-
    Tests = [
        test_load_tensor_returns_python_handle,
        test_loaded_tensor_has_correct_shape,
        test_can_compute_sum_in_torch,
        test_can_run_matmul,
        test_can_run_silu
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

%% Test 1: load_tensor returns a Python object handle, NOT a Prolog list.
%% This is the KEY property — direct numpy.fromfile explodes large arrays
%% to Prolog lists; going through the helper keeps it as one object.
test_load_tensor_returns_python_handle :-
    gguf_test_file(Path),
    py_call(llamatov_loader:load_tensor(Path, 1000000, 589824, 'float32', [768, 768]),
            Tensor),
    \+ is_list(Tensor).

%% Test 2: the tensor's shape matches what we requested
test_loaded_tensor_has_correct_shape :-
    gguf_test_file(Path),
    py_call(llamatov_loader:load_tensor(Path, 1000000, 589824, 'float32', [768, 768]),
            Tensor),
    py_call(llamatov_loader:tensor_shape(Tensor), Shape),
    Shape == 768-768.

%% Test 3: torch operations work on the loaded tensor
test_can_compute_sum_in_torch :-
    gguf_test_file(Path),
    py_call(llamatov_loader:load_tensor(Path, 1000000, 589824, 'float32', [768, 768]),
            Tensor),
    py_call(llamatov_loader:tensor_sum(Tensor), Sum),
    float(Sum).

%% Test 4: matmul of loaded tensor with itself works
test_can_run_matmul :-
    gguf_test_file(Path),
    py_call(llamatov_loader:load_tensor(Path, 1000000, 589824, 'float32', [768, 768]),
            T),
    py_call(llamatov_loader:matmul(T, T), Result),
    py_call(llamatov_loader:tensor_shape(Result), Shape),
    Shape == 768-768.

%% Test 5: silu activation on loaded tensor
test_can_run_silu :-
    gguf_test_file(Path),
    py_call(llamatov_loader:load_tensor(Path, 1000000, 589824, 'float32', [768, 768]),
            T),
    py_call(llamatov_loader:silu(T), Result),
    py_call(llamatov_loader:tensor_shape(Result), Shape),
    Shape == 768-768.

:- initialization(run_tests, main).
