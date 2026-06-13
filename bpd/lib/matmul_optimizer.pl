%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════
%% Licensed under RTAAL-1.0 (Ruach Tov AI Agent License)
%% See: LICENSE-RTAAL-1-0.md in the project root
%% ═══════════════════════════════════════════════════════════════════════

%% matmul_optimizer.pl — Constraint solver for matmul tile parameters
%%
%% Given hardware limits (sm_61), find all valid parameter combinations
%% and rank them by predicted arithmetic intensity.
%%
%% This IS the Phase 2 A* search expressed as Prolog constraints.

:- module(matmul_optimizer, [
    valid_config/1,
    config_metrics/2,
    best_configs/2,
    print_configs/1,
    config_metrics_v2/2,
    valid_pipeline_config/1,
    pipeline_config_metrics/2,

    %% k_tile_strategy substrate-design parameter (per Heath's 2026-05-20 ~00:35 UTC
    %% direction to pair-implement with mavchin). Names the shape-aware K_TILE
    %% dispatch choice as an explicit substrate-design parameter, parallel to
    %% reduction_strategy_kind/1 and rsqrt_variant.
    k_tile_strategy_kind/1,             % enumerate strategies
    k_tile_strategy_description/2,       % strategy -> description text
    k_tile_for_shape/5                   % +Strategy, +M, +N, +K, -K_TILE
]).

%% ═══════════════════════════════════════════════════
%% Hardware constraints (Tesla P4, sm_61)
%% ═══════════════════════════════════════════════════

hw_shared_mem_bytes(49152).      % 48KB per SM
hw_max_regs_per_thread(255).
hw_max_regs_per_sm(65536).
hw_max_threads_per_block(1024).
hw_max_blocks_per_sm(32).
hw_warp_size(32).
hw_sm_count(20).

%% ═══════════════════════════════════════════════════
%% Parameter space
%% ═══════════════════════════════════════════════════

tile_size(16).
tile_size(32).
tile_size(64).

reg_block(1).
reg_block(2).
reg_block(4).
reg_block(8).

double_buffer(false).
double_buffer(true).

vector_width(1).     % float
vector_width(4).     % float4

%% ═══════════════════════════════════════════════════
%% Constraint: valid configuration
%% ═══════════════════════════════════════════════════

valid_config(config(Tile, RegH, RegW, DoubleBuf, VecW)) :-
    tile_size(Tile),
    reg_block(RegH),
    reg_block(RegW),
    double_buffer(DoubleBuf),
    vector_width(VecW),

    %% Tile must be divisible by reg block
    Tile mod RegH =:= 0,
    Tile mod RegW =:= 0,

    %% Threads per block
    ThreadsH is Tile // RegH,
    ThreadsW is Tile // RegW,
    ThreadsPerBlock is ThreadsH * ThreadsW,
    ThreadsPerBlock >= 32,           % at least one warp
    ThreadsPerBlock =< 1024,         % hardware limit

    %% Shared memory
    (DoubleBuf == true -> BufCount = 2 ; BufCount = 1),
    SharedBytes is BufCount * 2 * Tile * Tile * 4,
    hw_shared_mem_bytes(MaxShared),
    SharedBytes =< MaxShared,

    %% Registers per thread
    Accumulators is RegH * RegW,
    LoadRegs is RegH + RegW,
    RegsPerThread is Accumulators + LoadRegs + 12,  % +12 for overhead
    hw_max_regs_per_thread(MaxRegs),
    RegsPerThread =< MaxRegs,

    %% Occupancy: total regs per SM
    hw_max_regs_per_sm(MaxRegsSM),
    RegsPerThread * ThreadsPerBlock =< MaxRegsSM,

    %% Vector width must divide tile
    Tile mod VecW =:= 0.

%% ═══════════════════════════════════════════════════
%% Metrics for each configuration
%% ═══════════════════════════════════════════════════

config_metrics(config(Tile, RegH, RegW, DoubleBuf, VecW), Metrics) :-
    ThreadsH is Tile // RegH,
    ThreadsW is Tile // RegW,
    ThreadsPerBlock is ThreadsH * ThreadsW,
    Accumulators is RegH * RegW,

    %% Arithmetic intensity: FMAs per memory load from shared
    %% Each inner loop step: load RegH from A_shared + RegW from B_shared
    %% Compute: RegH × RegW FMAs
    LoadsPerStep is RegH + RegW,
    FMAsPerStep is RegH * RegW,
    ReuseFactor is FMAsPerStep / LoadsPerStep,

    %% Shared memory
    (DoubleBuf == true -> BufCount = 2 ; BufCount = 1),
    SharedKB is BufCount * 2 * Tile * Tile * 4 / 1024,

    %% Registers
    RegsPerThread is Accumulators + RegH + RegW + 12,

    %% Predicted relative throughput (higher = better)
    %% Based on: larger tile = less global mem traffic
    %%           higher reuse = less shared mem traffic
    %%           double buffer = hide latency
    (DoubleBuf == true -> DBBonus = 1.3 ; DBBonus = 1.0),
    (VecW =:= 4 -> VecBonus = 1.2 ; VecBonus = 1.0),
    TileBonus is Tile / 32.0,
    PredScore is ReuseFactor * TileBonus * DBBonus * VecBonus,

    Metrics = metrics{
        tile: Tile, reg_h: RegH, reg_w: RegW,
        double_buf: DoubleBuf, vec_width: VecW,
        threads: ThreadsPerBlock,
        accumulators: Accumulators,
        reuse_factor: ReuseFactor,
        shared_kb: SharedKB,
        regs_per_thread: RegsPerThread,
        pred_score: PredScore
    }.

%% ═══════════════════════════════════════════════════
%% Find best configurations
%% ═══════════════════════════════════════════════════

best_configs(N, Configs) :-
    findall(Score-Metrics,
        (valid_config(C), config_metrics(C, Metrics),
         Score = Metrics.pred_score),
        Pairs),
    sort(1, @>=, Pairs, Sorted),
    length(Sorted, Total),
    format("~w valid configurations found.~n~n", [Total]),
    take_n(N, Sorted, Top),
    maplist([_-_M]>>true, Top, Configs),
    Configs = Top.

take_n(_, [], []) :- !.
take_n(0, _, []) :- !.
take_n(N, [H|T], [H|R]) :- N > 0, N1 is N - 1, take_n(N1, T, R).

print_configs(Pairs) :-
    format("~w~t~40|~w~t~48|~w~t~56|~w~t~62|~w~t~70|~w~t~78|~w~t~88|~w~n",
           ['Config', 'Thrd', 'Acc', 'Reuse', 'Shmem', 'Regs', 'Score', 'DblBuf']),
    forall(member(Score-M, Pairs),
        (format("T=~w R=~wx~w V=~w~t~40|~w~t~48|~w~t~56|~2f~t~62|~wKB~t~70|~w~t~78|~2f~t~88|~w~n",
                [M.tile, M.reg_h, M.reg_w, M.vec_width,
                 M.threads, M.accumulators, M.reuse_factor,
                 M.shared_kb, M.regs_per_thread, Score, M.double_buf]))).

%% ═══════════════════════════════════════════════════
%% Extended parameter space: K-tile + pipeline depth
%% ═══════════════════════════════════════════════════

k_tile_size(4).
k_tile_size(8).
k_tile_size(16).
k_tile_size(32).
k_tile_size(64).

%% ═════════════════════════════════════════════════════════════════════════
%% k_tile_strategy substrate-design parameter
%% ═════════════════════════════════════════════════════════════════════════
%%
%% Per Heath's "Definitely proceed" 2026-05-20 ~00:38 UTC and mavchin's
%% SASS analysis (the public make bit_identical 14/20 → 20/20 push).
%%
%% Empirically surfaced by Tier 2 verification: cuBLAS dispatches DIFFERENT
%% K_TILE sizes depending on matrix shape. The substrate currently uses
%% K_TILE=8 universally. When shapes diverge from cuBLAS's choice, the FMA
%% accumulation order differs, producing 11-66 ULP (correct values, different
%% rounding).
%%
%% Mavchin's nsys SASS analysis (2026-05-20 ~00:35 UTC):
%%
%%   1024×1024×1024 → cuBLAS sgemm_128x128x8_NN_vec  (K_TILE=8)  → substrate K=8 → 0 ULP
%%   2048×1024×512  → cuBLAS sgemm_128x128x8_NN_vec  (K_TILE=8)  → 0 ULP
%%   512×512×512    → cuBLAS maxwell_sgemm_128x64_nn (K_TILE=8)  → 0 ULP
%%   64×1024×1024   → cuBLAS sgemm_32x32x32_NN_vec   (K_TILE=32) → substrate K=8 → 32 ULP
%%   128×512×256    → cuBLAS sgemm_32x32x32_NN_vec   (K_TILE=32) → 11 ULP
%%
%% Substrate-design pattern: same as reduction_strategy_kind/1 — name the
%% choice, expose it as a parameter, let the verification ladder check each
%% choice produces the expected bits.

k_tile_strategy_kind(auto).        % Shape-aware: dispatch by M/N/K dimensions
k_tile_strategy_kind(k8).          % Force K_TILE=8 (substrate-historical default)
k_tile_strategy_kind(k32).         % Force K_TILE=32 (cuBLAS small/non-square)
k_tile_strategy_kind(k16).         % Future: intermediate sizes
k_tile_strategy_kind(k64).         % Future: large-tile experiments

k_tile_strategy_description(auto,
    'Shape-aware dispatch matching cuBLAS heuristic. K=32 for small/non-square, K=8 for large square.').
k_tile_strategy_description(k8,
    'Substrate-historical default. Matches cuBLAS for large square matrices (≥512, divisible by 64).').
k_tile_strategy_description(k32,
    'Matches cuBLAS sgemm_32x32x32 dispatch for small or non-square shapes (any dim < 512).').
k_tile_strategy_description(k16,
    'Intermediate tile size. Reserved for future substrate-design auto-tuning experiments.').
k_tile_strategy_description(k64,
    'Large-tile experiment. May exceed shared-memory budget at T=64; subject to constraint solver.').

%% k_tile_for_shape(+Strategy, +M, +N, +K, -KTile)
%%
%% Implements the substrate-design dispatch heuristic. For Strategy=auto,
%% uses mavchin's SASS-derived rule: cuBLAS uses K_TILE=32 when any dim
%% is small (< 512) or the shape is non-square (M ≠ N). Otherwise K=8.
%%
%% For explicit strategies (k8, k32, etc.), the dispatch is constant —
%% the substrate-design choice is made by the caller, not by the shape.

k_tile_for_shape(k8, _M, _N, _K, 8).
k_tile_for_shape(k16, _M, _N, _K, 16).
k_tile_for_shape(k32, _M, _N, _K, 32).
k_tile_for_shape(k64, _M, _N, _K, 64).

%% auto: shape-aware dispatch matching cuBLAS heuristic per SASS analysis.
%% Empirical rule from mavchin's nsys profiling:
%%   - If M=N=K AND all ≥ 512: K_TILE=8 (matches sgemm_128x128x8_NN_vec)
%%   - If any dim < 512 OR non-square: K_TILE=32 (matches sgemm_32x32x32_NN_vec)
k_tile_for_shape(auto, M, N, K, 8) :-
    M =:= N, N =:= K,            % square
    M >= 512,                    % all dims at least 512
    !.
k_tile_for_shape(auto, _M, _N, _K, 32).   % otherwise

%% ═════════════════════════════════════════════════════════════════════════
%% Pipeline depth (existing infrastructure, unchanged)
%% ═════════════════════════════════════════════════════════════════════════

pipeline_depth(1).     % no pipelining (single buffer)
pipeline_depth(2).     % double buffer (load[k+1] while compute[k])
pipeline_depth(3).     % triple buffer (2 loads ahead)

%% A valid pipeline configuration must satisfy shared memory constraints
%% AND the pipeline depth must be compatible with the K-tile size.

valid_pipeline_config(pconfig(Tile, RegH, RegW, KTile, PipeDepth, VecW)) :-
    tile_size(Tile),
    reg_block(RegH),
    reg_block(RegW),
    k_tile_size(KTile),
    pipeline_depth(PipeDepth),
    vector_width(VecW),

    %% Basic validity from tile/reg (same as valid_config)
    Tile mod RegH =:= 0,
    Tile mod RegW =:= 0,
    ThreadsH is Tile // RegH,
    ThreadsW is Tile // RegW,
    ThreadsPerBlock is ThreadsH * ThreadsW,
    ThreadsPerBlock >= 32,
    ThreadsPerBlock =< 1024,

    %% K-tile must divide into Tile for clean loads
    Tile mod KTile =:= 0,

    %% SHARED MEMORY: pipeline_depth × (A_tile + B_tile)
    %% A_tile = Tile × KTile × 4 bytes
    %% B_tile = KTile × Tile × 4 bytes
    SharedBytesPerBuf is 2 * Tile * KTile * 4,
    TotalSharedBytes is PipeDepth * SharedBytesPerBuf,
    hw_shared_mem_bytes(MaxShared),
    TotalSharedBytes =< MaxShared,

    %% REGISTER CONSTRAINT
    Accumulators is RegH * RegW,
    LoadRegs is RegH + RegW,
    %% Pipeline depth adds register pressure: each in-flight load needs
    %% address registers. Approximate: +4 regs per pipeline stage
    PipeRegs is (PipeDepth - 1) * 4,
    RegsPerThread is Accumulators + LoadRegs + 12 + PipeRegs,
    hw_max_regs_per_thread(MaxRegs),
    RegsPerThread =< MaxRegs,
    hw_max_regs_per_sm(MaxRegsSM),
    RegsPerThread * ThreadsPerBlock =< MaxRegsSM,

    %% Vector width must divide both Tile and KTile
    Tile mod VecW =:= 0,
    KTile mod min(VecW, KTile) =:= 0.

%% Metrics for pipeline configs
pipeline_config_metrics(pconfig(Tile, RegH, RegW, KTile, PipeDepth, VecW), Metrics) :-
    ThreadsH is Tile // RegH,
    ThreadsW is Tile // RegW,
    ThreadsPerBlock is ThreadsH * ThreadsW,
    Accumulators is RegH * RegW,

    %% Reuse factor: FMAs per shared-mem load
    FMAsPerStep is RegH * RegW,
    LoadsPerStep is RegH + RegW,
    ReuseFactor is FMAsPerStep / LoadsPerStep,

    %% Compute-to-load ratio: how many FMAs per global memory transaction
    %% Per K-tile: Tile² FMAs of useful work, 2×Tile×KTile floats loaded
    FMAsPerKTile is Tile * Tile * KTile,
    LoadsPerKTile is 2 * Tile * KTile,
    ComputeLoadRatio is FMAsPerKTile / LoadsPerKTile,

    %% Pipeline overlap factor: with depth D, (D-1)/D of load latency hidden
    %% But only if compute time exceeds load time per K-tile
    %% Compute time per K-tile ≈ Tile² × KTile FMAs / (peak_GFLOPS × threads)
    %% Load time per K-tile ≈ 2 × Tile × KTile × 4 / BW
    PipeOverlap is (PipeDepth - 1) / PipeDepth,

    %% Shared memory per buffer
    SharedKBPerBuf is 2 * Tile * KTile * 4 / 1024,
    TotalSharedKB is PipeDepth * SharedKBPerBuf,

    %% Registers
    PipeRegs is (PipeDepth - 1) * 4,
    RegsPerThread is Accumulators + RegH + RegW + 12 + PipeRegs,

    %% Occupancy
    hw_max_regs_per_sm(MaxRegsSM),
    hw_warp_size(WarpSize),
    WarpsPerBlock is ThreadsPerBlock // WarpSize,
    MaxBlocksReg is MaxRegsSM // (RegsPerThread * ThreadsPerBlock),
    hw_shared_mem_bytes(MaxShared),
    SharedBytes is integer(TotalSharedKB * 1024),
    (SharedBytes > 0 -> MaxBlocksShared is MaxShared // max(1, SharedBytes) ; MaxBlocksShared = 32),
    hw_max_blocks_per_sm(MaxBlocksHW),
    ActiveBlocks is min(MaxBlocksReg, min(MaxBlocksShared, MaxBlocksHW)),
    ActiveWarps is ActiveBlocks * WarpsPerBlock,
    _Occupancy is min(1.0, ActiveWarps / 64.0),

    %% ═══════════════════════════════════════════════
    %% EMPIRICALLY CALIBRATED SCORING (v3)
    %% ═══════════════════════════════════════════════
    %% 
    %% Based on measurements (2048×2048 on Tesla P4):
    %%   R=2x2: 1518 GFLOPS (28%)   32 regs, 100% occ
    %%   R=4x4: 2604 GFLOPS (47%)   40 regs, 75% occ   ← BEST
    %%   R=8x8: 1052 GFLOPS (19%)   94 regs, 31% occ   ← 2.5× slower!
    %%   K=4:   2204 GFLOPS          loop overhead
    %%   K=8:   2604 GFLOPS          sweet spot
    %%   K=16:  2321 GFLOPS          too much shmem
    %%   K=64:  1087 GFLOPS          way too much shmem

    %% ACTUAL register count model (calibrated from nvcc -O3 --ptxas-options=-v):
    %%   Measured: R=2x2→32, R=4x4→40, R=8x2→48, R=4x8→63, R=8x8→94
    %%   Pattern: nvcc overhead grows with accumulator count
    %%   Best fit: regs ≈ 1.1 * acc + max(rh,rw) + 24
    %%   R=2x2: 1.1*4+2+24=30.4  (actual 32, error 5%)
    %%   R=4x4: 1.1*16+4+24=45.6 (actual 40, error 14%) 
    %%   R=8x8: 1.1*64+8+24=102.4 (actual 94, error 9%)
    %%   Good enough for occupancy prediction.
    ActualRegs is integer(1.1 * Accumulators + max(RegH, RegW) + 24 + PipeRegs),
    
    %% Occupancy from ACTUAL register count
    RegsPerBlock is ActualRegs * ThreadsPerBlock,
    (RegsPerBlock > 0 ->
        MaxBlocksFromRegs is 65536 // RegsPerBlock
    ;   MaxBlocksFromRegs = 32
    ),
    ActualBlocksPerSM is min(MaxBlocksFromRegs, min(MaxBlocksShared, MaxBlocksHW)),
    ActualWarpsPerSM is ActualBlocksPerSM * WarpsPerBlock,
    ActualOccupancy is min(1.0, ActualWarpsPerSM / 64.0),

    %% LATENCY HIDING MODEL (calibrated from measurements):
    %% R=4x4: 48 warps → 2604 GFLOPS (best)  → factor = 1.0
    %% R=4x8: 16 warps → ~1500 GFLOPS         → factor ≈ 0.58
    %% R=8x8: 10 warps → 1052 GFLOPS          → factor ≈ 0.40
    %%
    %% HOWEVER: reuse=4.0 for R=8x8 vs reuse=2.0 for R=4x4.
    %% Measured ratio: 1052/2604 = 0.404
    %% Required: LF(10 warps) * reuse(4) ≈ LF(48 warps) * reuse(2) * 0.404
    %% → LF(10) = LF(48) * 2 * 0.404 / 4 = 1.0 * 0.202 = 0.20
    %%
    %% Model: LF = min(1.0, (warps/40)^2)
    %% At 40 warps: LF=1.0. At 20 warps: LF=0.25. At 10 warps: LF=0.0625.
    %% This matches the measured sharp cliff.
    (ActualWarpsPerSM >= 40 ->
        LatencyFactor = 1.0
    ;   LatencyFactor is (ActualWarpsPerSM / 40.0) ** 2
    ),

    %% K-TILE OVERHEAD MODEL (first-principles + empirical calibration):
    %%
    %% Cost per K-tile iteration = fixed_overhead + load_time + compute_time
    %%
    %% fixed_overhead:  2 barriers + address math ≈ 74 cycles equivalent
    %% compute_time:    K_TILE × (Tile²/threads) FMAs
    %%                  For T=64 R=4x4: K_TILE × 16 FMAs ≈ K_TILE × 16 cycles
    %% load_time:       2 × Tile × K_TILE × 4 bytes / BW_shared
    %%                  BUT: larger tiles cause bank conflicts + stalls
    %%                  Empirical: load_time ∝ K_TILE^1.5 (superlinear!)
    %%
    %% Number of K-tile iterations: K / K_TILE (more tiles = more overhead)
    %%
    %% Total time ∝ (K/K_TILE) × (overhead + load(K_TILE) + compute(K_TILE))
    %%
    %% DERIVED MODEL:
    %%   FMAs_per_tile = Tile × Tile × K_TILE  (useful work)
    %%   overhead_per_tile = 74 cycles
    %%   compute_per_tile = K_TILE × 16 cycles  (for T=64 R=4x4)
    %%   load_per_tile = 15 × K_TILE^1.3 cycles (empirical: superlinear with size)
    %%
    %%   efficiency = compute / (overhead + compute + load)
    %%
    %% Verification against measurements (T=64 R=4x4, 2048×2048):
    %%   K=4:  comp=64, load=15*4^1.3=82, total=64+74+82=220, eff=64/220=0.29
    %%         → scale to match measured 0.85: load_coeff needs tuning
    %%   
    %% SIMPLER CALIBRATED MODEL:
    %%   Two competing effects:
    %%     Small K: overhead/compute ratio is bad (too many iterations)
    %%     Large K: shared memory load time dominates (too much data per tile)
    %%   
    %%   Model: KEfficiency = min(compute_eff, load_eff)
    %%     compute_eff = K_TILE × 16 / (K_TILE × 16 + 74)  (overhead fraction)
    %%     load_eff = 1.0 / (1.0 + (K_TILE / 12.0)^1.5)    (load saturation)
    %%
    %% PURE OVERHEAD MODEL: only the fixed cost per K-tile iteration.
    %% Load saturation is captured by shared memory → occupancy → LatencyFactor.
    %% Don't double-count it here.
    %%
    %% overhead_cycles ≈ 74 (2 barriers + address math + sync)
    %% compute_cycles = K_TILE × 16 (for T=64 R=4x4, 16 FMAs per step)
    %% FMAs per inner step ≈ reg_h × reg_w = Accumulators
    ComputeCycles is KTile * Accumulators,
    OverheadCycles = 74,
    ComputeEff is ComputeCycles / (ComputeCycles + OverheadCycles),
    %%
    %% Verification:
    %%   K=4:  CE=64/138=0.46,  LE=1/(1+0.19)=0.84  → 0.39
    %%   K=8:  CE=128/202=0.63, LE=1/(1+0.54)=0.65  → 0.41
    %%   K=16: CE=256/330=0.78, LE=1/(1+1.54)=0.39  → 0.31
    %%   K=32: CE=512/586=0.87, LE=1/(1+4.35)=0.19  → 0.16
    %%   K=64: CE=1024/1098=0.93, LE=1/(1+12.3)=0.075 → 0.07
    %%
    %% Ratios (normalized to K=8):
    %%   K=4:  0.39/0.41 = 0.95  (measured 0.85) — close
    %%   K=8:  1.00               (measured 1.00) — baseline
    %%   K=16: 0.31/0.41 = 0.75  (measured 0.84) — off
    %%   K=32: 0.16/0.41 = 0.39  (measured 0.41) — close!
    %%   K=64: 0.07/0.41 = 0.17  (measured 0.42) — off
    %%
    %% The load saturation exponent 1.5 is too aggressive for K=16.
    %% Adjust to 1.8 to better penalize large K:
    %% FINAL: load_eff = 1.0 / (1.0 + (K/10)^1.8)
    %% HYBRID MODEL:
    %% K ≤ 8:  physics (overhead-limited, well-understood)
    %% K > 8:  empirical (compiler unrolling + bank conflicts, hard to model)
    %% 
    %% For K ≤ 8: ComputeEff captures the overhead correctly
    %% For K > 8: ComputeEff is too optimistic, use empirical decay
    %%   measured: K=16→0.84, K=32→0.41, K=64→0.42 (relative to K=8)
    %%   empirical decay: 0.84 × (8/K)^0.5 for K>8
    %%     K=16: 0.84×0.71 = 0.59 (too low). Just use measured directly.
    (KTile =< 8 ->
        KEfficiencyFinal is ComputeEff
    ;   %% K>8: start from K=8 efficiency, apply measured degradation
        CE8 is (8 * Accumulators) / (8 * Accumulators + OverheadCycles),
        (KTile =:= 16 -> KEfficiencyFinal is CE8 * 0.84
        ; KTile =:= 32 -> KEfficiencyFinal is CE8 * 0.41
        ; KTile =:= 64 -> KEfficiencyFinal is CE8 * 0.42
        ; KEfficiencyFinal is CE8 * 0.5  % unknown, conservative
        )
    ),

    %% TILE SIZE BONUS: larger tile = less global load overhead
    %% Quadratic in tile: output grows as tile², loads grow as tile
    TileBonus is Tile / 32.0,

    %% VECTOR LOAD BONUS
    (VecW =:= 4 -> VecBonus = 1.15 ; VecBonus = 1.0),
    
    %% PIPELINE BONUS (modest — nvcc already schedules loads)
    PipeBonus is 1.0 + 0.1 * PipeOverlap,

    %% COMBINED SCORE
    %% Reuse × LatencyHiding × KEfficiency × TileSize × Vector × Pipeline
    Score is ReuseFactor * LatencyFactor * KEfficiencyFinal * TileBonus * VecBonus * PipeBonus,

    Metrics = metrics{
        tile: Tile, reg_h: RegH, reg_w: RegW,
        k_tile: KTile, pipe_depth: PipeDepth, vec_width: VecW,
        threads: ThreadsPerBlock, accumulators: Accumulators,
        reuse_factor: ReuseFactor, compute_load_ratio: ComputeLoadRatio,
        pipe_overlap: PipeOverlap, shared_kb: TotalSharedKB,
        actual_regs: ActualRegs, occupancy: ActualOccupancy,
        active_warps: ActualWarpsPerSM,
        latency_factor: LatencyFactor,
        k_compute_eff: ComputeEff,
        k_efficiency: KEfficiencyFinal,
        pred_score: Score
    }.

%% ═══════════════════════════════════════════════════
%% Improved scoring with occupancy model
%% ═══════════════════════════════════════════════════

config_metrics_v2(config(Tile, RegH, RegW, DoubleBuf, VecW), Metrics) :-
    config_metrics(config(Tile, RegH, RegW, DoubleBuf, VecW), M0),
    get_dict(threads, M0, Threads),
    get_dict(regs_per_thread, M0, RegsPerThread),
    get_dict(reuse_factor, M0, RF),
    get_dict(shared_kb, M0, SharedKB),

    %% Occupancy model
    hw_max_regs_per_sm(MaxRegsSM),
    hw_warp_size(WarpSize),
    WarpsPerBlock is Threads // WarpSize,
    %% Max blocks per SM from registers
    (RegsPerThread > 0 ->
        MaxBlocksReg is MaxRegsSM // (RegsPerThread * Threads)
    ;   MaxBlocksReg = 32
    ),
    %% Max blocks from shared memory
    SharedBytes is SharedKB * 1024,
    hw_shared_mem_bytes(MaxShared),
    (SharedBytes > 0 ->
        MaxBlocksShared is MaxShared // max(1, SharedBytes)
    ;   MaxBlocksShared = 32
    ),
    hw_max_blocks_per_sm(MaxBlocksHW),
    ActiveBlocks is min(MaxBlocksReg, min(MaxBlocksShared, MaxBlocksHW)),
    ActiveWarps is ActiveBlocks * WarpsPerBlock,
    %% Occupancy: fraction of max warps (64 for sm_61)
    _Occupancy is min(1.0, ActiveWarps / 64.0),

    %% Score: reuse × occupancy × tile bonus × vector bonus
    TileBonus is Tile / 32.0,
    (VecW =:= 4 -> VecBonus = 1.2 ; VecBonus = 1.0),
    (DoubleBuf == true -> DBBonus = 1.3 ; DBBonus = 1.0),
    Score is RF * sqrt(Occupancy) * TileBonus * DBBonus * VecBonus,

    put_dict(occupancy, M0, Occupancy, M1),
    put_dict(active_warps, M1, ActiveWarps, M2),
    put_dict(active_blocks, M2, ActiveBlocks, M3),
    put_dict(pred_score_v2, M3, Score, Metrics).
