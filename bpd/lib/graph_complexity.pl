%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% graph_complexity.pl — Compute graph complexity metrics from BPD facts.
%%
%% Quantifies the "seams of memory boundaries" in a compute graph.
%% Each metric measures a different dimension of complexity:
%%
%%   materialization_points  — mandatory DRAM writes (unfusible boundaries)
%%   kernel_launches         — total GPU dispatch count
%%   arithmetic_intensity    — FLOPs / bytes (roofline position)
%%   fusion_potential         — how much memory traffic fusion could eliminate
%%   memory_traffic_ratio     — actual / minimum memory traffic
%%
%% The key insight: each "seam" is a materialization point.
%% Graph complexity = how many seams are NECESSARY vs how many exist
%% due to lack of fusion.
%%
%% Author: medayek (Collective SME, Verification Methodology)
%% Date: 2026-05-19
%% Per Heath's directive: mathematically exact performance model.

:- module(graph_complexity, [
    graph_complexity/2,         % graph_complexity(+Graph, -Metrics)
    arithmetic_intensity/3,     % arithmetic_intensity(+Flops, +Bytes, -AI)
    materialization_points/2,   % materialization_points(+Graph, -Count)
    kernel_launches/2,          % kernel_launches(+Graph, -Count)
    total_flops/2,              % total_flops(+Graph, -Flops)
    total_memory_traffic/2,     % total_memory_traffic(+Graph, -Bytes)
    minimum_memory_traffic/2,   % minimum_memory_traffic(+Graph, -Bytes)
    fusion_potential/2,         % fusion_potential(+Graph, -Ratio)
    memory_traffic_ratio/2,     % memory_traffic_ratio(+Graph, -Ratio)
    roofline_bound/4,           % roofline_bound(+AI, +PeakGflops, +PeakBW, -Bound)
    kernel_complexity/2,        % kernel_complexity(+KernelFacts, -Metrics)
    op_flops/3,                 % op_flops(+OpKind, +Shape, -Flops)
    op_memory/3,                % op_memory(+OpKind, +Shape, -Bytes)
    tensor_bytes/2,             % tensor_bytes(+Shape, -Bytes)
    print_complexity/1          % print_complexity(+Metrics)
]).

%% ═══════════════════════════════════════════════════════════════════════
%% Hardware parameters (Tesla P4, sm_61)
%% These become configurable when we support multiple targets.
%% ═══════════════════════════════════════════════════════════════════════

%% Tesla P4 (GP104, sm_61)
%%   Peak FP32: 5.5 TFLOPS
%%   Peak memory bandwidth: 192 GB/s (GDDR5X)
%%   L2 cache: 2 MB
%%   Shared memory per SM: 48 KB (or 96 KB with reduced L1)
%%   SMs: 20
%%   Warp size: 32
%%   Kernel launch overhead: ~5-15 μs
hw_peak_gflops(5500.0).          % 5.5 TFLOPS = 5500 GFLOPS
hw_peak_bandwidth_gbs(192.0).    % 192 GB/s
hw_l2_cache_bytes(2097152).      % 2 MB
hw_shared_mem_bytes(49152).      % 48 KB per SM
hw_sm_count(20).
hw_warp_size(32).
hw_launch_overhead_us(10.0).     % ~10 μs per kernel launch


%% ═══════════════════════════════════════════════════════════════════════
%% Core complexity metric: graph_complexity/2
%% ═══════════════════════════════════════════════════════════════════════

%% graph_complexity(+Graph, -Metrics)
%%   Graph = list of op(OpKind, InputShapes, OutputShape) terms
%%   Metrics = complexity{...} dict with all computed metrics
graph_complexity(Graph, Metrics) :-
    materialization_points(Graph, MP),
    kernel_launches(Graph, KL),
    total_flops(Graph, Flops),
    total_memory_traffic(Graph, Traffic),
    minimum_memory_traffic(Graph, MinTraffic),
    ( Traffic > 0
    -> arithmetic_intensity(Flops, Traffic, AI),
       MTR is Traffic / max(MinTraffic, 1)
    ;  AI = inf, MTR = 1.0
    ),
    ( MinTraffic > 0
    -> FP is 1.0 - (MinTraffic / max(Traffic, 1))
    ;  FP = 0.0
    ),
    hw_peak_gflops(PeakGF),
    hw_peak_bandwidth_gbs(PeakBW),
    roofline_bound(AI, PeakGF, PeakBW, RooflineBound),
    hw_launch_overhead_us(LaunchUS),
    LaunchOverheadUS is KL * LaunchUS,
    Metrics = complexity{
        ops: Graph,
        num_ops: NumOps,
        materialization_points: MP,
        kernel_launches: KL,
        total_flops: Flops,
        total_memory_bytes: Traffic,
        minimum_memory_bytes: MinTraffic,
        arithmetic_intensity: AI,
        memory_traffic_ratio: MTR,
        fusion_potential: FP,
        roofline_bound_gflops: RooflineBound,
        launch_overhead_us: LaunchOverheadUS
    },
    length(Graph, NumOps).


%% ═══════════════════════════════════════════════════════════════════════
%% Roofline model
%% ═══════════════════════════════════════════════════════════════════════

%% arithmetic_intensity(+Flops, +Bytes, -AI)
%%   AI = Flops / Bytes (FLOPs per byte of memory traffic)
arithmetic_intensity(Flops, Bytes, AI) :-
    Bytes > 0,
    AI is Flops / Bytes.

%% roofline_bound(+AI, +PeakGflops, +PeakBandwidthGBs, -BoundGflops)
%%   The roofline: min(PeakCompute, AI * PeakBandwidth)
roofline_bound(AI, PeakGF, PeakBW, Bound) :-
    ( AI =:= inf
    -> Bound = PeakGF
    ;  BWBound is AI * PeakBW,
       Bound is min(PeakGF, BWBound)
    ).


%% ═══════════════════════════════════════════════════════════════════════
%% Per-operation FLOP and memory models
%% ═══════════════════════════════════════════════════════════════════════

%% op_flops(+OpKind, +Shape, -Flops)
%%   Shape conventions:
%%     [M, N]     — matrix dimensions
%%     [M, N, K]  — matmul: C[M,N] = A[M,K] * B[K,N]
%%     [N]        — vector of length N

% BLAS L3: GEMM — 2*M*N*K FLOPs (multiply + accumulate)
op_flops(gemm, [M, N, K], Flops) :- Flops is 2 * M * N * K.
op_flops(matmul, [M, N, K], Flops) :- Flops is 2 * M * N * K.

% BLAS L2: GEMV — 2*M*N FLOPs
op_flops(gemv, [M, N], Flops) :- Flops is 2 * M * N.
op_flops(vecmat, [M, N], Flops) :- Flops is 2 * M * N.

% BLAS L1: dot product — 2*N FLOPs (multiply + add)
op_flops(dot, [N], Flops) :- Flops is 2 * N.

% Elementwise unary: N FLOPs (one op per element)
op_flops(relu, [N], N).
op_flops(silu, [N], Flops) :- Flops is 4 * N.  % exp + div + mul + add
op_flops(gelu, [N], Flops) :- Flops is 8 * N.  % tanh approx polynomial
op_flops(sigmoid, [N], Flops) :- Flops is 3 * N.  % exp + add + div
op_flops(tanh, [N], Flops) :- Flops is 5 * N.  % exp(2x), div, sub, etc.
op_flops(exp, [N], Flops) :- Flops is 1 * N.
op_flops(log, [N], Flops) :- Flops is 1 * N.
op_flops(sqrt, [N], Flops) :- Flops is 1 * N.
op_flops(neg, [N], N).
op_flops(abs, [N], N).
op_flops(sqr, [N], N).

% Elementwise binary: N FLOPs
op_flops(add, [N], N).
op_flops(mul, [N], N).
op_flops(sub, [N], N).
op_flops(div, [N], N).
op_flops(scale, [N], N).

% Reductions: N FLOPs (one comparison/add per element)
op_flops(sum, [N], N).
op_flops(max, [N], N).
op_flops(min, [N], N).
op_flops(mean, [N], Flops) :- Flops is N + 1.  % sum + divide
op_flops(argmax, [N], N).

% Softmax: 5N FLOPs (max + exp + sum + div, with subtract)
op_flops(softmax, [N], Flops) :- Flops is 5 * N.

% RMS norm: 3N + 2 FLOPs (sqr + sum + rsqrt + mul*weight)
op_flops(rms_norm, [N], Flops) :- Flops is 3 * N + 2.

% Layer norm: 5N + 4 FLOPs
op_flops(layer_norm, [N], Flops) :- Flops is 5 * N + 4.

% RoPE: 6 FLOPs per pair (2 sin/cos + 4 mul/add)
op_flops(rope, [N], Flops) :- Flops is 3 * N.

% Attention: QK^T (2*T*d) + softmax (5*T) + attn*V (2*T*d)
op_flops(attention, [T, D], Flops) :- Flops is 4 * T * D + 5 * T.

% CFD stencil: ~20 FLOPs per interface (Roe flux)
op_flops(roe_flux, [N], Flops) :- Flops is 20 * N.

% Fallback: assume 1 FLOP per element
op_flops(_, [N], N).


%% op_memory(+OpKind, +Shape, -Bytes)
%%   Memory traffic in bytes (read + write, assuming F32 = 4 bytes/element)
%%   This models DRAM traffic, not cache hits.

% GEMM: read A[M,K] + B[K,N] + write C[M,N]
op_memory(gemm, [M, N, K], Bytes) :-
    Bytes is 4 * (M*K + K*N + M*N).
op_memory(matmul, [M, N, K], Bytes) :-
    Bytes is 4 * (M*K + K*N + M*N).

% GEMV: read A[M,N] + x[N] + write y[M]
op_memory(gemv, [M, N], Bytes) :-
    Bytes is 4 * (M*N + N + M).
op_memory(vecmat, [M, N], Bytes) :-
    Bytes is 4 * (M*N + N + M).

% Elementwise unary: read N + write N
op_memory(relu, [N], Bytes) :- Bytes is 4 * 2 * N.
op_memory(silu, [N], Bytes) :- Bytes is 4 * 2 * N.
op_memory(gelu, [N], Bytes) :- Bytes is 4 * 2 * N.
op_memory(sigmoid, [N], Bytes) :- Bytes is 4 * 2 * N.
op_memory(tanh, [N], Bytes) :- Bytes is 4 * 2 * N.
op_memory(exp, [N], Bytes) :- Bytes is 4 * 2 * N.
op_memory(neg, [N], Bytes) :- Bytes is 4 * 2 * N.
op_memory(abs, [N], Bytes) :- Bytes is 4 * 2 * N.
op_memory(sqr, [N], Bytes) :- Bytes is 4 * 2 * N.

% Elementwise binary: read 2N + write N
op_memory(add, [N], Bytes) :- Bytes is 4 * 3 * N.
op_memory(mul, [N], Bytes) :- Bytes is 4 * 3 * N.
op_memory(sub, [N], Bytes) :- Bytes is 4 * 3 * N.
op_memory(scale, [N], Bytes) :- Bytes is 4 * 2 * N.  % scalar + vector

% Reductions: read N + write 1
op_memory(sum, [N], Bytes) :- Bytes is 4 * (N + 1).
op_memory(max, [N], Bytes) :- Bytes is 4 * (N + 1).
op_memory(mean, [N], Bytes) :- Bytes is 4 * (N + 1).
op_memory(softmax, [N], Bytes) :- Bytes is 4 * 3 * N.  % read N, write N (+ internal max/sum)
op_memory(rms_norm, [N], Bytes) :- Bytes is 4 * 3 * N.  % read x+w, write y
op_memory(layer_norm, [N], Bytes) :- Bytes is 4 * 4 * N.  % read x+gamma+beta, write y

% Attention: read Q[T,D]+K[T,D]+V[T,D], write O[T,D], plus scores[T,T]
op_memory(attention, [T, D], Bytes) :-
    Bytes is 4 * (3*T*D + T*D + T*T).

% Fallback: assume read N + write N
op_memory(_, [N], Bytes) :- Bytes is 4 * 2 * N.


%% tensor_bytes(+Shape, -Bytes)
%%   Size of a tensor in bytes (F32)
tensor_bytes([], 4).
tensor_bytes([H|T], Bytes) :-
    tensor_bytes(T, Rest),
    Bytes is H * Rest.


%% ═══════════════════════════════════════════════════════════════════════
%% Graph-level metrics
%% ═══════════════════════════════════════════════════════════════════════

%% total_flops(+Graph, -TotalFlops)
total_flops([], 0).
total_flops([op(Kind, _, Shape)|Rest], Total) :-
    op_flops(Kind, Shape, F),
    total_flops(Rest, RestF),
    Total is F + RestF.

%% total_memory_traffic(+Graph, -TotalBytes)
%%   Sum of all per-op memory traffic (unfused — each op reads/writes DRAM)
total_memory_traffic([], 0).
total_memory_traffic([op(Kind, _, Shape)|Rest], Total) :-
    op_memory(Kind, Shape, M),
    total_memory_traffic(Rest, RestM),
    Total is M + RestM.

%% minimum_memory_traffic(+Graph, -MinBytes)
%%   Minimum possible: read graph inputs + write graph outputs only.
%%   All intermediates stay in registers/shared memory.
minimum_memory_traffic(Graph, MinBytes) :-
    graph_inputs(Graph, Inputs),
    graph_outputs(Graph, Outputs),
    sum_tensor_bytes(Inputs, InBytes),
    sum_tensor_bytes(Outputs, OutBytes),
    MinBytes is InBytes + OutBytes.

%% Helper: sum bytes across a list of shapes
sum_tensor_bytes([], 0).
sum_tensor_bytes([Shape|Rest], Total) :-
    tensor_bytes(Shape, B),
    sum_tensor_bytes(Rest, RestB),
    Total is B + RestB.

%% graph_inputs(+Graph, -InputShapes)
%%   Inputs = tensors that are read but never written by any op in the graph.
%%   Simplified: first op's inputs.
graph_inputs([op(_, Inputs, _)|_], Inputs).
graph_inputs([], []).

%% graph_outputs(+Graph, -OutputShapes)
%%   Outputs = last op's output.
%%   Simplified: last op's output shape.
graph_outputs(Graph, [Shape]) :-
    last(Graph, op(_, _, Shape)).
graph_outputs([], []).

%% materialization_points(+Graph, -Count)
%%   Count of mandatory DRAM writes (where fusion cannot eliminate the boundary).
%%   Unfusible: reductions that need global synchronization before next op.
materialization_points([], 0).
materialization_points([_], 0).  % single op: no materialization
materialization_points([op(Kind1, _, _), Op2 | Rest], Count) :-
    materialization_points([Op2 | Rest], RestCount),
    ( requires_materialization(Kind1)
    -> Count is RestCount + 1
    ;  Count is RestCount
    ).

%% requires_materialization(+OpKind)
%%   Ops that MUST write their result to DRAM before the next op can read it.
%%   These are global reductions that need cross-block synchronization.
requires_materialization(softmax).    % needs full-row max + sum
requires_materialization(rms_norm).   % needs full-row sum-of-squares
requires_materialization(layer_norm). % needs mean + variance
requires_materialization(mean).       % global reduction
requires_materialization(sum).        % global reduction
requires_materialization(max).        % global reduction
requires_materialization(argmax).     % global reduction

%% kernel_launches(+Graph, -Count)
%%   Number of kernel launches = materialization_points + 1
%%   (each materialization starts a new kernel)
kernel_launches(Graph, Count) :-
    materialization_points(Graph, MP),
    Count is MP + 1.

%% fusion_potential(+Graph, -Ratio)
%%   Ratio of memory traffic that fusion could eliminate.
%%   FP = 1 - (minimum_traffic / actual_traffic)
%%   FP = 0 means nothing to fuse; FP = 0.8 means 80% of traffic is eliminable.
fusion_potential(Graph, FP) :-
    total_memory_traffic(Graph, Traffic),
    minimum_memory_traffic(Graph, MinTraffic),
    ( Traffic > 0
    -> FP is 1.0 - (MinTraffic / Traffic)
    ;  FP = 0.0
    ).

%% memory_traffic_ratio(+Graph, -MTR)
%%   MTR = actual_traffic / minimum_traffic
%%   MTR = 1.0 is perfectly fused; MTR > 1.0 has fusion opportunity.
memory_traffic_ratio(Graph, MTR) :-
    total_memory_traffic(Graph, Traffic),
    minimum_memory_traffic(Graph, MinTraffic),
    ( MinTraffic > 0
    -> MTR is Traffic / MinTraffic
    ;  MTR = 1.0
    ).


%% ═══════════════════════════════════════════════════════════════════════
%% Per-kernel complexity (single kernel analysis)
%% ═══════════════════════════════════════════════════════════════════════

%% kernel_complexity(+KernelFacts, -Metrics)
%%   Analyze a single kernel's complexity from its BPD facts.
kernel_complexity(op(Kind, _, Shape), Metrics) :-
    op_flops(Kind, Shape, Flops),
    op_memory(Kind, Shape, MemBytes),
    arithmetic_intensity(Flops, MemBytes, AI),
    hw_peak_gflops(PeakGF),
    hw_peak_bandwidth_gbs(PeakBW),
    roofline_bound(AI, PeakGF, PeakBW, Bound),
    ( AI < PeakGF / PeakBW
    -> Regime = memory_bound
    ;  Regime = compute_bound
    ),
    Metrics = kernel_metrics{
        op: Kind,
        shape: Shape,
        flops: Flops,
        memory_bytes: MemBytes,
        arithmetic_intensity: AI,
        roofline_bound_gflops: Bound,
        regime: Regime
    }.


%% ═══════════════════════════════════════════════════════════════════════
%% Pretty-print
%% ═══════════════════════════════════════════════════════════════════════

print_complexity(complexity{
    num_ops: NumOps,
    materialization_points: MP,
    kernel_launches: KL,
    total_flops: Flops,
    total_memory_bytes: MemBytes,
    minimum_memory_bytes: MinBytes,
    arithmetic_intensity: AI,
    memory_traffic_ratio: MTR,
    fusion_potential: FP,
    roofline_bound_gflops: Bound,
    launch_overhead_us: LaunchUS
}) :-
    format("Graph Complexity Metrics:~n"),
    format("  Operations:              ~w~n", [NumOps]),
    format("  Materialization points:  ~w~n", [MP]),
    format("  Kernel launches:         ~w~n", [KL]),
    format("  Total FLOPs:             ~w~n", [Flops]),
    format("  Memory traffic (actual): ~w bytes~n", [MemBytes]),
    format("  Memory traffic (min):    ~w bytes~n", [MinBytes]),
    format("  Arithmetic intensity:    ~4f FLOPs/byte~n", [AI]),
    format("  Memory traffic ratio:    ~4f×~n", [MTR]),
    format("  Fusion potential:        ~1f%~n", [FP * 100]),
    format("  Roofline bound:          ~4f GFLOPS~n", [Bound]),
    format("  Launch overhead:         ~1f μs~n", [LaunchUS]).


%% ═══════════════════════════════════════════════════════════════════════
%% Example: LLM transformer layer complexity
%% ═══════════════════════════════════════════════════════════════════════

%% example_transformer_layer(-Graph)
%%   A simplified llama3.2:1b transformer layer (n_embd=2048, n_ff=8192)
example_transformer_layer(Graph) :-
    N = 2048,
    FF = 8192,
    Graph = [
        op(rms_norm, [[N]], [N]),           % attention norm
        op(gemv, [[N]], [N, N]),             % Q projection
        op(gemv, [[N]], [N, N]),             % K projection
        op(gemv, [[N]], [N, N]),             % V projection
        op(rope, [[N]], [N]),               % RoPE on Q
        op(rope, [[N]], [N]),               % RoPE on K
        op(attention, [[1, 64]], [1, 64]),   % per-head attention (simplified)
        op(gemv, [[N]], [N, N]),             % O projection
        op(add, [[N]], [N]),                % residual add
        op(rms_norm, [[N]], [N]),           % FFN norm
        op(gemv, [[N]], [N, FF]),           % FFN up
        op(silu, [[FF]], [FF]),             % SiLU activation
        op(gemv, [[N]], [FF, N]),           % FFN gate (simplified)
        op(mul, [[FF]], [FF]),              % gate * up
        op(gemv, [[N]], [FF, N]),           % FFN down
        op(add, [[N]], [N])                % residual add
    ].
