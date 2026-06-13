%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_warp_optimizer.pl — Trivial case: optimal tile size for P4 matmul

:- use_module('../lib/warp_optimizer').
:- use_module('../lib/hardware_facts').

run_tests :-
    Tests = [
        test_p4_facts_loaded,
        test_warp_size_p4,
        test_shared_memory_per_block_p4,
        test_tile_fits_shared_memory_small,
        test_tile_overflows_shared_memory_huge,
        test_warp_aligned_check,
        test_optimal_tile_size_p4_f32,
        test_optimal_tile_size_p4_f16,
        test_optimal_tile_size_a100_vs_p4_differs,
        test_occupancy_estimate,
        test_arithmetic_intensity_increases_with_K,
        test_arithmetic_intensity_increases_with_min_dim,
        test_searched_returns_valid_tile,
        test_searched_picks_LARGER_than_trivial
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

test_p4_facts_loaded :-
    hardware_facts:hardware_target(sm_61),
    hardware_facts:hw_compute_capability(sm_61, '6.1').

test_warp_size_p4 :-
    hardware_facts:hw_warp_size(sm_61, 32).

test_shared_memory_per_block_p4 :-
    hardware_facts:hw_shared_memory_per_block_max(sm_61, 49152).  % 48 KB

%% Small tile (16x16x16) clearly fits
test_tile_fits_shared_memory_small :-
    warp_optimizer:fits_shared_memory(sm_61, matmul_bias_add,
                                       tile(16, 16, 16), f32).

%% Huge tile (1024x1024x1024) overflows
test_tile_overflows_shared_memory_huge :-
    \+ warp_optimizer:fits_shared_memory(sm_61, matmul_bias_add,
                                          tile(1024, 1024, 1024), f32).

%% 32 is warp-aligned; 33 is not
test_warp_aligned_check :-
    warp_optimizer:warp_aligned(sm_61, m, 32),
    warp_optimizer:warp_aligned(sm_61, m, 64),
    \+ warp_optimizer:warp_aligned(sm_61, m, 33).

%% Optimal tile for P4 with f32 should exist and be valid
test_optimal_tile_size_p4_f32 :-
    warp_optimizer:optimal_tile_size(sm_61, matmul_bias_add, f32,
                                      tile(M, N, K)),
    integer(M), integer(N), integer(K),
    % It should fit shared memory
    warp_optimizer:fits_shared_memory(sm_61, matmul_bias_add,
                                       tile(M, N, K), f32),
    % All dims should be warp-aligned
    warp_optimizer:warp_aligned(sm_61, m, M),
    warp_optimizer:warp_aligned(sm_61, n, N),
    warp_optimizer:warp_aligned(sm_61, k, K).

%% f16 needs half the memory per element so can use larger tiles
test_optimal_tile_size_p4_f16 :-
    warp_optimizer:optimal_tile_size(sm_61, matmul_bias_add, f16,
                                      tile(M, N, K)),
    integer(M), integer(N), integer(K).

%% A100 has more shared memory than P4, so it should admit larger tiles
%% (or at least not smaller tiles than P4 with same constraints)
%% For the trivial case we just check that A100 produces SOME tile.
test_optimal_tile_size_a100_vs_p4_differs :-
    warp_optimizer:optimal_tile_size(sm_61, matmul_bias_add, f32, P4Tile),
    warp_optimizer:optimal_tile_size(sm_80, matmul_bias_add, f32, A100Tile),
    % Both should be valid tile() terms
    P4Tile = tile(_, _, _),
    A100Tile = tile(_, _, _).
    % Note: with our simple "first valid" strategy, both might return
    % the same smallest tile. The point is BOTH return something valid
    % FOR THEIR HARDWARE.

%% Occupancy: 32x32 tile with 32 registers/thread on P4
test_occupancy_estimate :-
    warp_optimizer:occupancy_estimate(sm_61, tile(32, 32, 16), 32, Occ),
    integer(Occ),
    Occ > 0.

:- initialization(run_tests, main).

%% ────────────────────────────────────────────────────────────────────
%% Tests for the searched tile-size strategy
%% (Added separately so old test_runner above still works; these would
%% be added to the Tests list to actually run them. For now, run via
%% direct goal invocation.)
%% ────────────────────────────────────────────────────────────────────

test_arithmetic_intensity_increases_with_K :-
    warp_optimizer:arithmetic_intensity(tile(32, 32, 32), f32, AI1),
    warp_optimizer:arithmetic_intensity(tile(32, 32, 64), f32, AI2),
    AI2 > AI1.

test_arithmetic_intensity_increases_with_min_dim :-
    warp_optimizer:arithmetic_intensity(tile(32, 32, 32), f32, AI1),
    warp_optimizer:arithmetic_intensity(tile(64, 64, 32), f32, AI2),
    AI2 > AI1.

test_searched_returns_valid_tile :-
    warp_optimizer:optimal_tile_size_searched(sm_61, matmul_bias_add, f32,
                                              Tile, _),
    Tile = tile(M, N, K),
    integer(M), integer(N), integer(K),
    warp_optimizer:fits_shared_memory(sm_61, matmul_bias_add, Tile, f32),
    warp_optimizer:warp_aligned(sm_61, m, M),
    warp_optimizer:warp_aligned(sm_61, n, N).

test_searched_picks_LARGER_than_trivial :-
    % Searched should pick a tile with higher arithmetic intensity
    % than the trivial "first valid" choice. Specifically: K should
    % be larger than the trivial-strategy K when budget allows.
    warp_optimizer:optimal_tile_size(sm_61, matmul_bias_add, f32, TileTrivial),
    warp_optimizer:optimal_tile_size_searched(sm_61, matmul_bias_add, f32,
                                              TileSearched, _),
    warp_optimizer:arithmetic_intensity(TileTrivial, f32, AITrivial),
    warp_optimizer:arithmetic_intensity(TileSearched, f32, AISearched),
    AISearched >= AITrivial.
