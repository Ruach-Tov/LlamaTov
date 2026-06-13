%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% spike_prolog_drives_torch.pl
%%
%% Integration spike: Prolog drives a PyTorch matmul end-to-end.
%%
%% Per Heath's "decomposed end-to-end trivial case" methodology:
%% the smallest spike that proves Prolog CAN drive real PyTorch
%% computation, without requiring GGUF integration yet. This is the
%% trivial-case foothold for the full Prolog → GGUF → PyTorch pipeline.
%%
%% What this spike proves:
%%  1. janus_swi loads and Prolog can call Python
%%  2. Prolog constructs a PyTorch tensor from a Prolog list
%%  3. Prolog dispatches torch.matmul on Python tensor handles
%%  4. Prolog materializes the result back as a Prolog value
%%
%% Once this works, extending to real GGUF tensors is mechanical:
%% replace the Prolog list with a numpy.memmap on the GGUF file at
%% the offset computed by our existing parser.

:- use_module(library(janus)).

%% ────────────────────────────────────────────────────────────────────
%% Compute a small matmul end-to-end via Prolog → PyTorch
%% ────────────────────────────────────────────────────────────────────

%% Sample 4x4 identity matrix as Prolog list of lists
identity_4x4([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
]).

%% Sample 4x4 weight matrix
weight_4x4([
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0, 16.0]
]).

%% prolog_drives_matmul/1: returns the scalar sum of the matmul result.
%% Expected: matmul(I, W) = W, sum(W) = 1+2+...+16 = 136
prolog_drives_matmul(SumScalar) :-
    identity_4x4(I),
    weight_4x4(W),
    %% Step 1: Construct PyTorch tensors via janus
    py_call(torch:tensor(I), ITensor),
    py_call(torch:tensor(W), WTensor),
    %% Step 2: Dispatch matmul
    py_call(torch:matmul(ITensor, WTensor), Result),
    %% Step 3: Sum the result
    py_call(torch:sum(Result), SumTensor),
    %% Step 4: Materialize the scalar back to Prolog
    py_call(SumTensor:item(), SumScalar).

%% ────────────────────────────────────────────────────────────────────
%% Test runner
%% ────────────────────────────────────────────────────────────────────

run_tests :-
    Tests = [
        test_prolog_drives_torch_matmul,
        test_prolog_drives_chained_ops,
        test_prolog_constructs_large_tensor
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

%% Test 1: Identity matmul should produce sum 136 (1+2+...+16)
test_prolog_drives_torch_matmul :-
    prolog_drives_matmul(Sum),
    Sum =:= 136.0.

%% Test 2: Chain operations — silu(matmul(A, B))
test_prolog_drives_chained_ops :-
    identity_4x4(I),
    weight_4x4(W),
    py_call(torch:tensor(I), ITensor),
    py_call(torch:tensor(W), WTensor),
    py_call(torch:matmul(ITensor, WTensor), MM),
    %% Apply SiLU (which is x * sigmoid(x))
    py_call(torch:nn:functional:silu(MM), SiLUResult),
    %% Just verify we got a tensor back, materialize shape
    py_call(SiLUResult:shape, Shape),
    Shape = 4-4.

%% Test 3: Construct a larger tensor (768x768) as Prolog might from
%% a GGUF tensor. Verify it works without errors.
test_prolog_constructs_large_tensor :-
    %% Build a 768x768 tensor as a flat numpy array via py_call
    py_call(torch:zeros([768, 768]), Tensor),
    py_call(Tensor:shape, Shape),
    Shape = 768-768.

:- initialization(run_tests, main).
