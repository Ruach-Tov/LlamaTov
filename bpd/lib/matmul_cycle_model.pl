%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% matmul_cycle_model.pl — Cycle-accurate performance prediction
%%
%% Predicts wall-clock time for any matmul configuration from
%% first principles: instruction costs × iteration counts.
%%
%% All constants measured on Tesla P4 (sm_61).

:- module(matmul_cycle_model, [
    predict_matmul/5,     % +Config, +M, +N, +K, -Prediction
    predict_gflops/5,     % +Config, +M, +N, +K, -GFLOPS
    compare_configs/4     % +ConfigList, +M, +N, +K
]).

%% ═══════════════════════════════════════
%% Hardware constants (measured on Tesla P4)
%% ═══════════════════════════════════════

hw_sm_count(20).
hw_fp32_per_sm(128).        % FP32 CUDA cores per SM
hw_clock_mhz(1400).         % approximate boost clock
hw_warp_size(32).
hw_max_regs_sm(65536).
hw_shared_bytes(49152).     % 48KB

%% Instruction latencies/throughputs (cycles)
cost_ffma_throughput(1).        % 1 FFMA per cycle per 32 cores
cost_ffma_latency(6).           % 6 cycle pipeline latency
cost_lds_latency(30).           % shared memory load latency
cost_sts_latency(30).           % shared memory store latency  
cost_ldg_latency(350).          % global memory load latency (DRAM)
cost_bar_sync(20).              % __syncthreads barrier
cost_loop_overhead(4).          % branch + counter increment per iteration
cost_launch_us(9.0).            % kernel launch overhead

%% Memory bandwidth
hw_dram_bw_gb(150.0).           % measured achievable peak

%% ═══════════════════════════════════════
%% Cycle-accurate prediction
%% ═══════════════════════════════════════

predict_matmul(config(Tile, RegH, RegW, KTile), M, N, K, Prediction) :-
    %% Thread geometry
    ThreadsH is Tile // RegH,
    ThreadsW is Tile // RegW,
    ThreadsPerBlock is ThreadsH * ThreadsW,
    WarpsPerBlock is ThreadsPerBlock // 32,
    Accumulators is RegH * RegW,

    %% Register-based occupancy
    ActualRegs is integer(1.1 * Accumulators + max(RegH, RegW) + 24),
    RegsPerBlock is ActualRegs * ThreadsPerBlock,
    hw_max_regs_sm(MaxRegs),
    BlocksPerSM_Reg is MaxRegs // max(1, RegsPerBlock),

    %% Shared memory occupancy  
    SharedPerBlock is 2 * Tile * KTile * 4,
    hw_shared_bytes(MaxShared),
    BlocksPerSM_Shmem is MaxShared // max(1, SharedPerBlock),

    %% Active blocks and warps
    BlocksPerSM is min(BlocksPerSM_Reg, min(BlocksPerSM_Shmem, 32)),
    ActiveWarps is BlocksPerSM * WarpsPerBlock,

    %% Number of output tiles
    TilesM is (M + Tile - 1) // Tile,
    TilesN is (N + Tile - 1) // Tile,
    TotalOutputTiles is TilesM * TilesN,

    %% Number of K-tile iterations per output tile
    KIterations is (K + KTile - 1) // KTile,

    %% ─── COST PER K-TILE ITERATION ───

    %% 1. Global load cost
    %%    Bytes to load: Tile × KTile × 4 (for A) + KTile × Tile × 4 (for B)
    GlobalBytes is 2 * Tile * KTile * 4,
    %%    Each thread loads GlobalBytes / ThreadsPerBlock bytes
    BytesPerThread is GlobalBytes / ThreadsPerBlock,
    %%    Transactions: each 32-byte sector = 1 transaction
    %%    Latency hidden by warp interleaving: effective = latency / active_warps
    cost_ldg_latency(LDGLat),
    (ActiveWarps > 0 ->
        LoadCycles is LDGLat / min(ActiveWarps, 8) + BytesPerThread / 4
    ;   LoadCycles is LDGLat + BytesPerThread / 4
    ),

    %% 2. Shared store cost
    cost_sts_latency(STSLat),
    StoreCycles is STSLat,

    %% 3. Barrier cost
    cost_bar_sync(BarCost),

    %% 4. Compute cost (the inner loop: KTile steps × RegH×RegW FMAs)
    %%    Per inner step: load RegH + RegW values from shared, do RegH×RegW FMAs
    %%    Shared loads: can issue 1 per cycle per bank (32 banks)
    %%    FMAs: 1 per cycle per warp (32 threads × 1 FFMA = 32 FP ops)
    %%    
    %%    Per warp per inner step:
    %%      LDS: RegH + RegW loads. If no bank conflict: (RegH+RegW) cycles
    %%      FMA: RegH × RegW FMAs. Throughput: RegH*RegW cycles
    %%      These overlap if LDS latency is hidden by FMA pipeline
    cost_lds_latency(_LDSLat),
    LDSPerStep is RegH + RegW,
    FMAPerStep is Accumulators,
    %% For the full inner loop (KTile steps):
    %% Compute cycles per warp = KTile × max(FMA_cycles, LDS_cycles + FMA_latency)
    %% Simplified: if FMAs dominate (RegH*RegW > RegH+RegW+6), compute-bound
    %% If LDS dominates: memory-bound on shared
    (FMAPerStep >= LDSPerStep + 6 ->
        %% Compute-bound: FMAs are the bottleneck
        InnerCyclesPerWarp is KTile * FMAPerStep
    ;   %% LDS-bound: shared memory loads are the bottleneck  
        InnerCyclesPerWarp is KTile * (LDSPerStep + 6)
    ),
    %% Multiple warps time-multiplex on the SM
    %% Total compute cycles = InnerCyclesPerWarp × WarpsPerBlock / SM_width
    hw_fp32_per_sm(FP32Units),
    WarpWidth is FP32Units // 32,  % How many warps can execute FMAs simultaneously
    ComputeCycles is InnerCyclesPerWarp * WarpsPerBlock / max(1, WarpWidth),

    %% 5. Loop overhead
    cost_loop_overhead(LoopOH),

    %% ─── TOTAL COST PER K-TILE ───
    %% Load and store overlap with compute on different warps
    %% BUT: __syncthreads forces serialization
    %% Actual: max(load+store, compute) + 2×barrier + loop_overhead
    TileIterCost is max(LoadCycles + StoreCycles, ComputeCycles) + 2 * BarCost + LoopOH,

    %% ─── TOTAL KERNEL TIME ───
    %% Each SM processes TotalOutputTiles / sm_count output tiles
    %% Each tile processes KIterations K-tiles
    hw_sm_count(SMCount),
    %% Account for multiple blocks per SM
    EffectiveSMs is SMCount * BlocksPerSM,
    TilesPerSM is (TotalOutputTiles + EffectiveSMs - 1) // EffectiveSMs,
    
    TotalCycles is TilesPerSM * KIterations * TileIterCost,

    %% Convert cycles to microseconds
    hw_clock_mhz(ClockMHz),
    TotalUS is TotalCycles / ClockMHz + 9.0,  % + launch overhead

    %% FLOPs
    TotalFLOPs is 2.0 * M * N * K,
    PredGFLOPS is TotalFLOPs / (TotalUS * 1000.0),
    PredPctPeak is PredGFLOPS / 5500.0 * 100.0,

    Prediction = prediction{
        config: config(Tile, RegH, RegW, KTile),
        total_cycles: TotalCycles,
        time_us: TotalUS,
        gflops: PredGFLOPS,
        pct_peak: PredPctPeak,
        tiles_per_sm: TilesPerSM,
        k_iterations: KIterations,
        cost_per_ktile: TileIterCost,
        load_cycles: LoadCycles,
        compute_cycles: ComputeCycles,
        active_warps: ActiveWarps,
        blocks_per_sm: BlocksPerSM,
        shared_kb: SharedPerBlock / 1024
    }.

predict_gflops(Config, M, N, K, GFLOPS) :-
    predict_matmul(Config, M, N, K, P),
    GFLOPS = P.gflops.

compare_configs(Configs, M, N, K) :-
    format("~nCycle-accurate prediction (M=~w N=~w K=~w):~n~n", [M, N, K]),
    format("~w~t~36|~w~t~44|~w~t~52|~w~t~60|~w~t~68|~w~n",
           ['Config', 'us', 'GFLOPS', '%peak', 'ldcy', 'ccyc']),
    forall(
        member(C, Configs),
        (predict_matmul(C, M, N, K, P),
         get_dict(time_us, P, US),
         get_dict(gflops, P, GF),
         get_dict(pct_peak, P, PP),
         get_dict(load_cycles, P, LC),
         get_dict(compute_cycles, P, CC),
         C = config(T, RH, RW, KT),
         format("T=~w R=~wx~w K=~w~t~36|~0f~t~44|~0f~t~52|~1f%~t~60|~0f~t~68|~0f~n",
                [T, RH, RW, KT, US, GF, PP, LC, CC]))
    ).
