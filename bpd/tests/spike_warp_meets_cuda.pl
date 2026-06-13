%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% spike_warp_meets_cuda.pl
%%
%% Composition spike: feed warp_optimizer's tile dimensions into
%% mavchin's CUDA AST primitives to emit a TILED fused kernel.
%%
%% This is the natural next step after:
%%   - mavchin's commit d0fd967c0: CUDA kernel emission works on Tesla P4
%%     (0.93x speed — needs shared memory tiling for speedup)
%%   - my warp_optimizer (commits 091c42d4c, 6fb1732c6, dc3d0a6b7):
%%     produces hardware-aware tile dimensions
%%
%% The composition: optimizer.tile_size → kernel.shared_memory_decl
%%                                       → kernel.tile_dimensions
%%                                       → kernel.threadblock_layout
%%
%% Once shared memory is tiled, the matmul reuses A and B from shared
%% memory across the K loop, eliminating most VRAM round-trips. Expected
%% speedup vs unfused: substantial (depends on K and tile size).

:- use_module('../lib/warp_optimizer').
:- use_module('../lib/hardware_facts').

%% ────────────────────────────────────────────────────────────────────
%% Demonstration: derive tile dimensions for a real fusion on P4
%% ────────────────────────────────────────────────────────────────────

run_tests :-
    Tests = [
        test_optimizer_provides_tile_for_p4,
        test_tile_fits_shared_memory_budget,
        test_threads_per_block_fits_hw_limit,
        test_arithmetic_intensity_improves_with_tiling,
        test_k_dim_increases_arithmetic_intensity
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

%% Test 1: optimizer produces a tile for the fused kernel pattern on P4
test_optimizer_provides_tile_for_p4 :-
    warp_optimizer:optimal_tile_size_searched(sm_61, matmul_bias_add, f32,
                                              tile(M, N, K), _Score),
    integer(M), integer(N), integer(K),
    M > 0, N > 0, K > 0.

%% Test 2: the tile fits the shared memory budget
test_tile_fits_shared_memory_budget :-
    warp_optimizer:optimal_tile_size_searched(sm_61, matmul_bias_add, f32,
                                              Tile, _),
    warp_optimizer:fits_shared_memory(sm_61, matmul_bias_add, Tile, f32).

%% Test 3: threads-per-block (M*N for matmul) fits P4's 1024 max
test_threads_per_block_fits_hw_limit :-
    warp_optimizer:optimal_tile_size_searched(sm_61, matmul_bias_add, f32,
                                              tile(M, N, _), _),
    hardware_facts:hw_threads_per_block_max(sm_61, MaxThreads),
    ThreadsPerBlock is M * N,
    ThreadsPerBlock =< MaxThreads,
    format("    Tile produces ~d threads/block (max ~d)~n",
           [ThreadsPerBlock, MaxThreads]).

%% Test 4: arithmetic intensity with optimizer's tile vs naive 1x1
test_arithmetic_intensity_improves_with_tiling :-
    warp_optimizer:optimal_tile_size_searched(sm_61, matmul_bias_add, f32,
                                              OptimalTile, _),
    warp_optimizer:arithmetic_intensity(OptimalTile, f32, OptimalAI),
    %% Naive baseline: tile(32, 32, 32) — what trivial strategy would pick
    warp_optimizer:arithmetic_intensity(tile(32, 32, 32), f32, NaiveAI),
    OptimalAI > NaiveAI,
    Improvement is (OptimalAI - NaiveAI) / NaiveAI * 100,
    format("    Optimal: ~3f FLOPs/byte vs Naive: ~3f (~3f%% better)~n",
           [OptimalAI, NaiveAI, Improvement]).

%% Test 5: increasing K improves arithmetic intensity
test_k_dim_increases_arithmetic_intensity :-
    warp_optimizer:arithmetic_intensity(tile(32, 32, 32),  f32, AI1),
    warp_optimizer:arithmetic_intensity(tile(32, 32, 128), f32, AI2),
    AI2 > AI1.

:- initialization(run_tests, main).
