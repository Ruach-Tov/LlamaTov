%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% warp_optimizer.pl — Per-hardware warp/tile optimization.
%%
%% Trivial case per Heath's decomposition methodology:
%%   - ONE hardware target (P4, sm_61)
%%   - ONE fused kernel (matmul + bias_add from QKV BPD)
%%   - ONE optimization (loop tile-size selection)
%%   - Constraint: tile must fit in shared memory budget
%%   - Output: derived tile dimensions
%%
%% Per Heath's empirical wisdom: "I have hand-warped kernels for 32x
%% gains more often than you would expect... we have to examine and
%% deterministically compile custom code to every hardware."
%%
%% This module implements that determinism: given a kernel and a
%% hardware target, derive the optimal tile size by Prolog query
%% over hardware facts.

:- module(warp_optimizer, [
    optimal_tile_size/4,
    optimal_tile_size_searched/5,
    optimal_tile_size_optin/5,
    optimal_tile_size_tc/5,
    fits_shared_memory/4,
    fits_shared_memory_optin/4,
    warp_aligned/3,
    occupancy_estimate/4,
    occupancy_estimate_optin/4,
    arithmetic_intensity/3,
    tile_score/4,
    tensor_core_aligned/3
]).

:- use_module(hardware_facts).

%% ────────────────────────────────────────────────────────────────────
%% Tile size selection
%% ────────────────────────────────────────────────────────────────────
%%
%% optimal_tile_size(+Hardware, +Kernel, +DataType, -TileSize)
%%   Hardware: e.g., sm_61
%%   Kernel: the fused operation being tiled (e.g., matmul_bias_add)
%%   DataType: f32 | f16
%%   TileSize: tile(TileM, TileN, TileK) — typed three-axis tile
%%
%% Constraints:
%%   - Tile fits in shared memory (with all loaded operands)
%%   - Tile dimensions are multiples of warp_size for coalescing
%%   - Tile produces reasonable occupancy
%%
%% Strategy: enumerate candidate sizes, filter by constraints, pick best.

optimal_tile_size(Hardware, matmul_bias_add, DataType,
                  tile(TileM, TileN, TileK)) :-
    % Get hardware facts (using mavchin-aligned hw_ prefix naming)
    hw_warp_size(Hardware, WarpSize),
    hw_shared_memory_per_block_max(Hardware, SharedBudget),

    % Element size in bytes
    bytes_per_element(DataType, ElementBytes),

    % Candidate tile dimensions: warp-aligned multiples.
    candidate_tile_dim(WarpSize, TileM),
    candidate_tile_dim(WarpSize, TileN),
    candidate_tile_dim(WarpSize, TileK),

    % Enforce warp alignment on memory-access dims.
    warp_aligned(Hardware, m, TileM),
    warp_aligned(Hardware, n, TileN),

    % EPILOGUE-IS-FREE (mavchin's KernelBench empirical finding):
    % For fused matmul + elementwise epilogue, the bias_add runs ON
    % THE SAME THREAD that computed the matmul output. The epilogue
    % uses registers, NOT shared memory. So the shared memory budget
    % only needs to accommodate the matmul's tile inputs, not the
    % bias or any other elementwise epilogue's working set.
    %
    % Before: TotalSharedBytes = (M*K + K*N + N) * ElementBytes
    % After:  TotalSharedBytes = (M*K + K*N) * ElementBytes
    % This admits LARGER tiles than the conservative formula.
    BytesA is TileM * TileK * ElementBytes,
    BytesB is TileK * TileN * ElementBytes,
    TotalSharedBytes is BytesA + BytesB,
    TotalSharedBytes =< SharedBudget,

    % Trivial-case strategy: first valid = smallest tile.
    % Real optimization would search for max arithmetic intensity.
    !.

%% Candidate tile dimensions: warp-aligned sizes from warp_size up to
%% a reasonable maximum. Generated from WarpSize so they scale to AMD
%% (wave_size 64) or other architectures.
candidate_tile_dim(WarpSize, WarpSize).
candidate_tile_dim(WarpSize, Dim) :- Dim is WarpSize * 2.
candidate_tile_dim(WarpSize, Dim) :- Dim is WarpSize * 4.
candidate_tile_dim(WarpSize, Dim) :- Dim is WarpSize * 8.
candidate_tile_dim(WarpSize, Dim) :- Dim is WarpSize * 16.

bytes_per_element(f32, 4).
bytes_per_element(f16, 2).
bytes_per_element(bf16, 2).
bytes_per_element(i32, 4).
bytes_per_element(i8, 1).
bytes_per_element(q4_0, 0.5).      % approximate; quantized
bytes_per_element(q8_0, 1).

%% ────────────────────────────────────────────────────────────────────
%% Shared memory fit check
%% ────────────────────────────────────────────────────────────────────
%%
%% fits_shared_memory(+Hardware, +Kernel, +TileSize, +DataType)
%%   Succeeds when the tile's working set fits in shared memory budget.
%%   Reports BYTES used (via the unbound call site) on success.

fits_shared_memory(Hardware, matmul_bias_add, tile(M, N, K), DataType) :-
    hw_shared_memory_per_block_max(Hardware, Budget),
    bytes_per_element(DataType, EB),
    % Epilogue-is-free: only matmul tiles count, not bias broadcast.
    Bytes is (M * K + K * N) * EB,
    Bytes =< Budget.

%% ────────────────────────────────────────────────────────────────────
%% Warp alignment check
%% ────────────────────────────────────────────────────────────────────
%%
%% warp_aligned(+Hardware, +Dim, +Value)
%%   Succeeds when Value is a multiple of warp_size for Dim.
%%   For coalesced memory access, the innermost (column) dim should
%%   be warp-aligned.

warp_aligned(Hardware, _Dim, Value) :-
    hw_warp_size(Hardware, W),
    0 is Value mod W.

%% ────────────────────────────────────────────────────────────────────
%% Occupancy estimation
%% ────────────────────────────────────────────────────────────────────
%%
%% occupancy_estimate(+Hardware, +TileSize, +RegPerThread, -Occupancy)
%%   Estimates blocks-per-SM for the proposed tile + register usage.
%%   Higher occupancy = better latency hiding.

occupancy_estimate(Hardware, tile(M, N, K), RegPerThread, Occupancy) :-
    hw_registers_per_sm(Hardware, RegPerSM),
    hw_shared_memory_per_sm(Hardware, SharedPerSM),
    hw_blocks_per_sm_max(Hardware, MaxBlocks),

    % Threads per block (one thread per output element tile)
    ThreadsPerBlock is M * N,

    % Constraint 1: registers
    RegPerBlock is ThreadsPerBlock * RegPerThread,
    BlocksByReg is RegPerSM // RegPerBlock,

    % Constraint 2: shared memory (epilogue-is-free: matmul tiles only)
    bytes_per_element(f32, EB),
    BytesPerBlock is (M * K + K * N) * EB,
    BlocksByShared is SharedPerSM // BytesPerBlock,

    % Constraint 3: max blocks
    Blocks0 is min(BlocksByReg, BlocksByShared),
    Occupancy is min(Blocks0, MaxBlocks).

%% ────────────────────────────────────────────────────────────────────
%% Better tile-size strategy: search space + score function
%% ────────────────────────────────────────────────────────────────────
%%
%% The trivial-case `optimal_tile_size/4` returns the FIRST VALID tile
%% (smallest, conservatively chosen). That's correct but suboptimal —
%% it doesn't exploit available shared memory budget. The "always
%% 32x32x32" output proves we're leaving hardware capability unused.
%%
%% The searched version generates all valid candidates and picks the
%% one that maximizes a scoring function:
%%   score = arithmetic_intensity * sqrt(occupancy)
%% Square-rooting occupancy reflects the diminishing-returns shape: going
%% from 2 to 4 blocks/SM helps a lot; 4 to 16 helps less.
%%
%% Arithmetic intensity for matmul: FLOPs / bytes_moved
%%   FLOPs = 2 * M * N * K  (multiply-add)
%%   bytes_moved = (M*K + K*N + M*N) * EB  (A + B in, C out)
%%   ratio = (2*M*N*K) / ((M*K + K*N + M*N) * EB)
%%
%% This favors LARGER tiles (more FLOPs per byte) up to memory budget.

%% arithmetic_intensity(+TileSize, +DataType, -Intensity)
%%   FLOPs per byte for the matmul portion of the tile.
arithmetic_intensity(tile(M, N, K), DataType, Intensity) :-
    bytes_per_element(DataType, EB),
    Flops is 2 * M * N * K,
    BytesMoved is (M*K + K*N + M*N) * EB,
    BytesMoved > 0,
    Intensity is Flops / BytesMoved.

%% tile_score(+Hardware, +TileSize, +DataType, -Score)
%%   Combined arithmetic intensity × sqrt(occupancy) — the objective
%%   we want to maximize. Higher score = better tile choice.
%%
%%   Uses 32 registers/thread as a typical mid-range estimate; real
%%   register pressure depends on the compiled kernel.
tile_score(Hardware, tile(M, N, K), DataType, Score) :-
    arithmetic_intensity(tile(M, N, K), DataType, AI),
    occupancy_estimate(Hardware, tile(M, N, K), 32, Occ),
    Occ > 0,
    Score is AI * sqrt(Occ).

%% optimal_tile_size_searched(+Hardware, +Kernel, +DataType, -BestTile, -Score)
%%   Generate all valid candidates, score each, return the winner.
%%   This is the substrate-honest tile selection: maximize the objective
%%   over the constrained candidate space.

optimal_tile_size_searched(Hardware, matmul_bias_add, DataType,
                           BestTile, BestScore) :-
    hw_warp_size(Hardware, WarpSize),
    % Generate all valid candidate tiles
    findall(
        score(Score, Tile),
        ( candidate_tile_dim(WarpSize, TileM),
          candidate_tile_dim(WarpSize, TileN),
          candidate_tile_dim(WarpSize, TileK),
          warp_aligned(Hardware, m, TileM),
          warp_aligned(Hardware, n, TileN),
          Tile = tile(TileM, TileN, TileK),
          fits_shared_memory(Hardware, matmul_bias_add, Tile, DataType),
          tile_score(Hardware, Tile, DataType, Score)
        ),
        Candidates
    ),
    Candidates \= [],
    % Pick max-scoring
    sort(0, @>=, Candidates, Sorted),
    Sorted = [score(BestScore, BestTile) | _].

%% ────────────────────────────────────────────────────────────────────
%% Opt-in shared memory + tensor-core-aware variants
%% ────────────────────────────────────────────────────────────────────
%%
%% Modern NVIDIA hardware (sm_80+) allows opt-in shared memory beyond
%% the default 48 KB via cudaFuncSetAttribute. Larger budget admits
%% larger tiles, which improve arithmetic intensity further.

%% fits_shared_memory_optin(+Hardware, +Kernel, +TileSize, +DataType)
%%   Succeeds when the tile fits in the OPT-IN shared memory budget.
%%   For Pascal (sm_61), opt-in budget = default budget (no opt-in available).
fits_shared_memory_optin(Hardware, matmul_bias_add, tile(M, N, K), DataType) :-
    hw_shared_memory_per_block_optin(Hardware, Budget),
    bytes_per_element(DataType, EB),
    Bytes is (M * K + K * N) * EB,
    Bytes =< Budget.

%% optimal_tile_size_optin(+Hardware, +Kernel, +DataType, -BestTile, -Score)
%%   Like optimal_tile_size_searched but uses opt-in shared memory budget.
%%   On Pascal this behaves identically to the default-budget search.
%%   On Ampere/Ada/Hopper, admits substantially larger tiles.
optimal_tile_size_optin(Hardware, matmul_bias_add, DataType,
                        BestTile, BestScore) :-
    hw_warp_size(Hardware, WarpSize),
    findall(
        score(Score, Tile),
        ( candidate_tile_dim_extended(WarpSize, TileM),
          candidate_tile_dim_extended(WarpSize, TileN),
          candidate_tile_dim_extended(WarpSize, TileK),
          warp_aligned(Hardware, m, TileM),
          warp_aligned(Hardware, n, TileN),
          Tile = tile(TileM, TileN, TileK),
          fits_shared_memory_optin(Hardware, matmul_bias_add, Tile, DataType),
          %% Score using opt-in budget for occupancy estimate
          arithmetic_intensity(Tile, DataType, AI),
          occupancy_estimate_optin(Hardware, Tile, 32, Occ),
          Occ > 0,
          Score is AI * sqrt(Occ)
        ),
        Candidates
    ),
    Candidates \= [],
    sort(0, @>=, Candidates, Sorted),
    Sorted = [score(BestScore, BestTile) | _].

%% Extended candidate tile dims: larger than the default candidate set
%% to admit the bigger tiles opt-in shared memory allows.
candidate_tile_dim_extended(WarpSize, WarpSize).
candidate_tile_dim_extended(WarpSize, Dim) :- Dim is WarpSize * 2.
candidate_tile_dim_extended(WarpSize, Dim) :- Dim is WarpSize * 4.
candidate_tile_dim_extended(WarpSize, Dim) :- Dim is WarpSize * 8.
candidate_tile_dim_extended(WarpSize, Dim) :- Dim is WarpSize * 16.
candidate_tile_dim_extended(WarpSize, Dim) :- Dim is WarpSize * 32.

%% occupancy_estimate_optin uses opt-in shared memory budget per SM
%% (total per-SM shared, not per-block opt-in).
occupancy_estimate_optin(Hardware, tile(M, N, K), RegPerThread, Occupancy) :-
    hw_registers_per_sm(Hardware, RegPerSM),
    hw_shared_memory_per_sm(Hardware, SharedPerSM),
    hw_blocks_per_sm_max(Hardware, MaxBlocks),
    ThreadsPerBlock is M * N,
    RegPerBlock is ThreadsPerBlock * RegPerThread,
    BlocksByReg is RegPerSM // RegPerBlock,
    bytes_per_element(f32, EB),
    BytesPerBlock is (M * K + K * N) * EB,
    BlocksByShared is SharedPerSM // BytesPerBlock,
    Blocks0 is min(BlocksByReg, BlocksByShared),
    Occupancy is min(Blocks0, MaxBlocks).

%% ────────────────────────────────────────────────────────────────────
%% Tensor core awareness
%% ────────────────────────────────────────────────────────────────────
%%
%% For hardware with tensor cores, tile dimensions should be multiples
%% of one of the supported MMA shapes. This produces tiles that map
%% cleanly to tensor core instructions.

%% tensor_core_aligned(+Hardware, +DataType, +TileSize)
%%   Succeeds when the tile dimensions are compatible with at least
%%   one supported MMA shape for this hardware+dtype.
tensor_core_aligned(Hardware, DataType, tile(M, N, K)) :-
    hw_tensor_core_shapes(Hardware, DataType, Shapes),
    member(mma_shape(MmaM, MmaN, MmaK), Shapes),
    0 is M mod MmaM,
    0 is N mod MmaN,
    0 is K mod MmaK.

%% optimal_tile_size_tc(+Hardware, +Kernel, +DataType, -BestTile, -Score)
%%   Tensor-core-aware tile selection. Only considers tiles whose
%%   dimensions are multiples of a supported MMA shape.
%%   Requires hw_supports_tensor_cores(Hardware, true).
optimal_tile_size_tc(Hardware, matmul_bias_add, DataType,
                     BestTile, BestScore) :-
    hw_supports_tensor_cores(Hardware, true),
    hw_warp_size(Hardware, WarpSize),
    findall(
        score(Score, Tile),
        ( candidate_tile_dim_extended(WarpSize, TileM),
          candidate_tile_dim_extended(WarpSize, TileN),
          candidate_tile_dim_extended(WarpSize, TileK),
          warp_aligned(Hardware, m, TileM),
          warp_aligned(Hardware, n, TileN),
          Tile = tile(TileM, TileN, TileK),
          tensor_core_aligned(Hardware, DataType, Tile),
          fits_shared_memory_optin(Hardware, matmul_bias_add, Tile, DataType),
          arithmetic_intensity(Tile, DataType, AI),
          occupancy_estimate_optin(Hardware, Tile, 32, Occ),
          Occ > 0,
          Score is AI * sqrt(Occ)
        ),
        Candidates
    ),
    Candidates \= [],
    sort(0, @>=, Candidates, Sorted),
    Sorted = [score(BestScore, BestTile) | _].
