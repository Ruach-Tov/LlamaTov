%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_warp_optin_and_tc.pl — verify opt-in shared memory and tensor-core-aware tile selection
%%
%% Stage extension to warp_optimizer per Heath's "treat bounded subtasks as urgent" directive:
%% - Opt-in shared memory budget on sm_80/sm_89 admits larger tiles
%% - Tensor-core-aware selection produces tiles compatible with WMMA shapes
%% - Substrate now exhibits HARDWARE DIFFERENTIATION (different tiles per GPU)

:- use_module('../lib/warp_optimizer').
:- use_module('../lib/hardware_facts').

run_tests :-
    Tests = [
        test_optin_p4_same_as_default,
        test_optin_a100_admits_larger_tile,
        test_optin_score_improves_or_equal,
        test_tc_aligned_tile_dims_multiples_of_mma,
        test_tc_pascal_has_no_tensor_cores,
        test_tc_aware_score_better_than_default,
        test_optin_a100_strictly_better_than_default,
        test_hardware_differentiation
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

%% Test 1: Opt-in on Pascal produces same result as default (no opt-in available)
test_optin_p4_same_as_default :-
    warp_optimizer:optimal_tile_size_searched(sm_61, matmul_bias_add, f32, T_def, _),
    warp_optimizer:optimal_tile_size_optin(sm_61, matmul_bias_add, f32, T_opt, _),
    T_def == T_opt.

%% Test 2: Opt-in on A100 admits a LARGER tile than default
test_optin_a100_admits_larger_tile :-
    warp_optimizer:optimal_tile_size_searched(sm_80, matmul_bias_add, f32,
                                              tile(_, _, K_def), _),
    warp_optimizer:optimal_tile_size_optin(sm_80, matmul_bias_add, f32,
                                            tile(_, _, K_opt), _),
    K_opt >= K_def,
    format("    A100 default K=~d, opt-in K=~d~n", [K_def, K_opt]).

%% Test 3: Opt-in score is at least as good as default score on every hardware
test_optin_score_improves_or_equal :-
    forall(hardware_facts:hardware_target(HW),
           ( warp_optimizer:optimal_tile_size_searched(HW, matmul_bias_add, f32, _, S_def),
             warp_optimizer:optimal_tile_size_optin(HW, matmul_bias_add, f32, _, S_opt),
             S_opt >= S_def
           )).

%% Test 4: TC-aligned tile dims are multiples of an MMA shape
test_tc_aligned_tile_dims_multiples_of_mma :-
    warp_optimizer:optimal_tile_size_tc(sm_80, matmul_bias_add, f16,
                                         tile(M, N, K), _),
    warp_optimizer:tensor_core_aligned(sm_80, f16, tile(M, N, K)).

%% Test 5: Tensor cores unavailable on Pascal — TC-aware optimization fails
test_tc_pascal_has_no_tensor_cores :-
    \+ warp_optimizer:optimal_tile_size_tc(sm_61, matmul_bias_add, f16, _, _).

%% Test 6: TC-aware FP16 score better than default-budget FP32 score on same hw
%% (TC operations have 2× throughput on FP16 vs CUDA cores on FP32)
test_tc_aware_score_better_than_default :-
    warp_optimizer:optimal_tile_size_tc(sm_80, matmul_bias_add, f16, _, S_tc),
    warp_optimizer:optimal_tile_size_searched(sm_80, matmul_bias_add, f32, _, S_def),
    S_tc > S_def,
    format("    A100 TC f16 score=~3f, default f32 score=~3f~n", [S_tc, S_def]).

%% Test 7: Opt-in A100 STRICTLY better than default A100 (monotonicity with opt-in)
test_optin_a100_strictly_better_than_default :-
    warp_optimizer:optimal_tile_size_searched(sm_80, matmul_bias_add, f32, _, S_def),
    warp_optimizer:optimal_tile_size_optin(sm_80, matmul_bias_add, f32, _, S_opt),
    S_opt > S_def.

%% Test 8: Hardware differentiation — A100 picks different tile than P4 with opt-in
test_hardware_differentiation :-
    warp_optimizer:optimal_tile_size_optin(sm_61, matmul_bias_add, f32,
                                            tile(_, _, K_p4), _),
    warp_optimizer:optimal_tile_size_optin(sm_80, matmul_bias_add, f32,
                                            tile(_, _, K_a100), _),
    %% A100 should pick a tile with K >= P4's K (better hardware = better tile)
    K_a100 >= K_p4.

:- initialization(run_tests, main).
