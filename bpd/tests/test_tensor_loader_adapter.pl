%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_tensor_loader_adapter.pl
%%
%% Verify the Prolog adapter loads tensors from the GGUF manifest with
%% Prolog-side orchestration (only torch ops happen in Python).
%%
%% Per Heath's directive: maximize Prolog. This test demonstrates the
%% pattern: Prolog reads manifest facts, computes call args, dispatches
%% to Python helper only for the actual data read + torch construction.

:- use_module(library(janus)).
:- py_add_lib_dir('lib').
:- use_module('../lib/tensor_loader_adapter').

%% Path to the nomic-embed-text GGUF
model_path('${OLLAMA_BLOBS:-~/.ollama/models/blobs}/sha256-970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6').

%% Sample tensor fact from the manifest emitter
%% (representing the first layer's attention output weight — F16)
sample_f16_tensor(tensor(
    name("blk.0.attn_output.weight"),
    dimensions([768, 768]),
    type_code(1),
    type_name('F16'),
    numpy_dtype(float16),
    file_offset(50433024),
    byte_size(1179648),
    relative_offset(49676320))).

%% Sample F32 tensor (a norm bias)
sample_f32_tensor(tensor(
    name("blk.0.attn_output_norm.bias"),
    dimensions([768]),
    type_code(0),
    type_name('F32'),
    numpy_dtype(float32),
    file_offset(65768448),
    byte_size(3072),
    relative_offset(65011744))).

run_tests :-
    Tests = [
        test_element_count_2d,
        test_element_count_1d,
        test_dtype_string_mapping,
        test_load_f16_tensor_upcasts_to_f32,
        test_load_f32_tensor_directly,
        test_default_mode_handles_f16,
        test_loaded_tensor_is_python_handle,
        test_can_use_loaded_tensor_in_matmul
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

test_element_count_2d :-
    tensor_loader_adapter:tensor_element_count([768, 2304], N),
    N == 1769472.

test_element_count_1d :-
    tensor_loader_adapter:tensor_element_count([768], N),
    N == 768.

test_dtype_string_mapping :-
    tensor_loader_adapter:tensor_dtype_string(float32, 'float32'),
    tensor_loader_adapter:tensor_dtype_string(float16, 'float16').

test_load_f16_tensor_upcasts_to_f32 :-
    model_path(Path),
    sample_f16_tensor(T),
    %% Use the 4-arg form: (ModelPath, TensorFact, Mode, Tensor)
    tensor_loader_adapter:load_tensor_from_manifest(Path, T, fp32_from_f16, Tensor),
    %% Verify it's a Python handle
    \+ is_list(Tensor),
    %% Verify shape via the helper
    py_call(llamatov_loader:tensor_shape(Tensor), Shape),
    Shape == 768-768.

test_load_f32_tensor_directly :-
    model_path(Path),
    sample_f32_tensor(T),
    tensor_loader_adapter:load_tensor_from_manifest(Path, T, fp32, Tensor),
    \+ is_list(Tensor),
    py_call(llamatov_loader:tensor_shape(Tensor), Shape),
    %% Janus represents Python 1-tuples as `-(N)` (the negative-prefix op term).
    %% Empirical finding: torch.Size([768]) → Shape = -(768) in Prolog.
    Shape = -(768).

test_default_mode_handles_f16 :-
    model_path(Path),
    sample_f16_tensor(T),
    %% Use the default-mode predicate (4-arity)
    tensor_loader_adapter:load_tensor_from_manifest(Path, T, Tensor),
    py_call(llamatov_loader:tensor_shape(Tensor), Shape),
    Shape == 768-768.

test_loaded_tensor_is_python_handle :-
    %% Critical property: tensor stays as Python object, never converts to Prolog list
    model_path(Path),
    sample_f16_tensor(T),
    tensor_loader_adapter:load_tensor_from_manifest(Path, T, Tensor),
    \+ is_list(Tensor).

test_can_use_loaded_tensor_in_matmul :-
    %% End-to-end: load tensor by manifest fact, run matmul with itself
    model_path(Path),
    sample_f16_tensor(T),
    tensor_loader_adapter:load_tensor_from_manifest(Path, T, Tensor),
    py_call(llamatov_loader:matmul(Tensor, Tensor), Result),
    py_call(llamatov_loader:tensor_shape(Result), Shape),
    Shape == 768-768.

:- initialization(run_tests, main).
