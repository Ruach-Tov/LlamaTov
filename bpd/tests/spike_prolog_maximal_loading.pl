%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% spike_prolog_maximal_loading.pl
%%
%% End-to-end demonstration of Prolog-maximal tensor loading from a
%% real GGUF file. Composes three substrate pieces:
%%
%%   1. gguf_emit_manifest.pl (must_close/boundary_dsl/) — Prolog parses
%%      GGUF file, emits manifest with absolute tensor offsets
%%   2. tensor_loader_adapter.pl (bpd/lib/) — Prolog consumes manifest
%%      facts, dispatches load_tensor calls to Python helper
%%   3. llamatov_loader.py (bpd/lib/) — Python helper: numpy.fromfile +
%%      torch.from_numpy, keeps tensor as single Python handle
%%
%% Per Heath's "maximize Prolog" directive: this demonstrates that
%% the Prolog substrate ALONE handles tensor selection, dtype
%% conversion, shape calculation, and offset calculation. Only the
%% bytes-and-torch step is Python.
%%
%% This is the reference implementation for mavchin's runner refactor.
%% The pattern shown here eliminates the need for embedded
%% builtins:exec Python helpers.

:- use_module(library(janus)).
:- py_add_lib_dir('lib').
:- use_module('../lib/tensor_loader_adapter').

%% Test fixture: Granite Embedding 30M (BERT architecture)
gguf_path('/tmp/test_model.gguf').
manifest_path('/tmp/test_manifest.pl').

%% Consult the manifest (produced by gguf_emit_manifest.pl earlier).
%% Use a separate module to avoid polluting top-level.
:- dynamic(tensor/8).
:- dynamic(gguf_header/4).
:- dynamic(metadata/3).

load_manifest :-
    manifest_path(Path),
    consult(Path).

run_tests :-
    load_manifest,
    Tests = [
        test_manifest_has_header,
        test_manifest_has_tensors,
        test_load_token_embd_norm_weight,
        test_load_position_embd_weight,
        test_load_f16_token_embd,
        test_pure_prolog_tensor_selection_by_name,
        test_full_layer_norm_weight_pair_load,
        test_dispatch_torch_ops_on_loaded_tensors
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
%% Helper: lookup tensor by name (pure Prolog)
%% ────────────────────────────────────────────────────────────────────

tensor_by_name(Name, TensorFact) :-
    TensorFact = tensor(name(Name), _Dims, _TypeCode, _TypeName,
                        _NumpyDtype, _FileOffset, _ByteSize, _RelOffset),
    call(TensorFact).

%% Load a tensor by name (pure Prolog dispatch via adapter)
load_by_name(ModelPath, Name, Tensor) :-
    tensor_by_name(Name, TensorFact),
    tensor_loader_adapter:load_tensor_from_manifest(ModelPath, TensorFact, Tensor).

%% ────────────────────────────────────────────────────────────────────
%% Tests
%% ────────────────────────────────────────────────────────────────────

test_manifest_has_header :-
    gguf_header(magic("GGUF"), version(3), tensor_count(TC), kv_count(_)),
    TC == 101.

test_manifest_has_tensors :-
    findall(N, tensor(name(N), _, _, _, _, _, _, _), Names),
    length(Names, NumTensors),
    NumTensors == 101.

test_load_token_embd_norm_weight :-
    %% 1D F32 tensor: a bias vector
    gguf_path(Path),
    load_by_name(Path, "token_embd_norm.weight", Tensor),
    \+ is_list(Tensor),
    py_call(llamatov_loader:tensor_shape(Tensor), Shape),
    %% 1D shape — janus represents 1-tuple as -(N)
    Shape = -(384).

test_load_position_embd_weight :-
    %% 2D F32 tensor: position embeddings, 384 x 512
    gguf_path(Path),
    load_by_name(Path, "position_embd.weight", Tensor),
    \+ is_list(Tensor),
    py_call(llamatov_loader:tensor_shape(Tensor), Shape),
    Shape == 384-512.

test_load_f16_token_embd :-
    %% F16 tensor that auto-upcasts to F32 via adapter default mode
    gguf_path(Path),
    load_by_name(Path, "token_embd.weight", Tensor),
    \+ is_list(Tensor),
    py_call(llamatov_loader:tensor_shape(Tensor), Shape),
    Shape == 384-50265.

test_pure_prolog_tensor_selection_by_name :-
    %% Demonstrate that selecting a tensor is pure Prolog
    %% (no py_call until the actual load)
    tensor_by_name("token_embd_norm.weight", TensorFact),
    %% Verify the fact structure
    TensorFact = tensor(name(Name), _, _, _, _, _, _, _),
    Name == "token_embd_norm.weight".

test_full_layer_norm_weight_pair_load :-
    %% Real-use pattern: load both weight and bias for a layer norm
    gguf_path(Path),
    load_by_name(Path, "token_embd_norm.weight", Weight),
    load_by_name(Path, "token_embd_norm.bias", Bias),
    \+ is_list(Weight),
    \+ is_list(Bias),
    %% Both should be the same shape (384,)
    py_call(llamatov_loader:tensor_shape(Weight), WShape),
    py_call(llamatov_loader:tensor_shape(Bias), BShape),
    WShape == BShape.

test_dispatch_torch_ops_on_loaded_tensors :-
    %% End-to-end: load tensors, run torch ops
    gguf_path(Path),
    load_by_name(Path, "token_embd_norm.weight", W),
    load_by_name(Path, "token_embd_norm.bias", B),
    %% torch.add the weight and bias
    py_call(llamatov_loader:add(W, B), Result),
    py_call(llamatov_loader:tensor_shape(Result), Shape),
    Shape = -(384).

:- initialization(run_tests, main).
