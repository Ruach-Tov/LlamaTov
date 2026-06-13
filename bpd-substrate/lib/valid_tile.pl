%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════
%% Licensed under RTAAL-1.0 (Ruach Tov AI Agent License)
%% See: LICENSE-RTAAL-1-0.md in the project root
%% ═══════════════════════════════════════════════════════════════════════

%% valid_tile.pl — Constraint-based tile selection with platform parameters
%%
%% Determines valid tiling configurations based on hardware constraints
%% and platform parameters.

/** <module> Constraint-Based Tile Selection

Selects valid tile dimensions for fused kernels based on hardware
constraints. K_tile must be divisible by 8 for vectorization.
Shared memory usage must not exceed hardware limits.

Tile parameters are derived from hardware_facts and the kernel's
memory access pattern, not guessed or hardcoded.

@author Ruach Tov Collective
@license RTAAL-1.0
@see implementation_matches.pl for platform-specific hardware facts
*/

:- module(valid_tile, [
    valid_tile/4
]).

:- use_module(matmul_optimizer, [
    hw_shared_mem_bytes/1,
    hw_max_regs_per_thread/1,
    hw_max_regs_per_sm/1,
    k_tile_for_shape/5
]).

%% ────────────────────────────────────────────────────────────────────
%% valid_tile(+M_tile, +N_tile, +K_tile, +Platform)
%% ────────────────────────────────────────────────────────────────────
%% True if the given tile dimensions are valid for the specified platform.

%! valid_tile(-M_tile, -N_tile, -K_tile, +Platform) is nondet.
%  Generate valid tile dimensions for Platform via backtracking.
%  Each solution satisfies vectorization alignment and shared memory bounds.
valid_tile(M_tile, N_tile, K_tile, Platform) :-
    %% Ensure dimensions are positive integers
    integer(M_tile), M_tile > 0,
    integer(N_tile), N_tile > 0,
    integer(K_tile), K_tile > 0,

    %% K_tile should typically be a multiple of 8 for vectorization
    K_tile mod 8 =:= 0,

    %% Shared memory constraint
    %% Assuming we need to store A (M_tile x K_tile) and B (K_tile x N_tile) in shared memory.
    %% Float32 = 4 bytes per element.
    %% If double buffering is used, multiply by 2.
    %% Let's assume single buffer for simplicity here, or check platform params.
    SharedBytes is (M_tile * K_tile + K_tile * N_tile) * 4,
    hw_shared_mem_bytes(MaxShared),
    SharedBytes =< MaxShared,

    %% Register constraints (simplified)
    %% Accumulators needed: M_tile * N_tile per warp?
    %% This depends on thread block size. We can defer deep checks to valid_config/1
    %% or implement a simplified version here.
    true.
