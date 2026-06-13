%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% gpu_roofline_analysis.pl — Comprehensive GPU performance counter project.
%%
%% Goal: explain every percentage point of the gap between
%% achieved bandwidth (42-62% of peak) and roofline (192 GB/s)
%% for ggml's Q8_0 matmul on the Tesla P4.
%%
%% Architecture: Prolog dispatches ggml matmul on GPU,
%% CUPTI event API reads exact hardware counter totals,
%% facts are stored for analysis and visualization.
%%
%% MEASUREMENT GROUPS (run kernel once per group, ~4 events max per domain):
%%
%% Group 1: MEMORY HIERARCHY
%%   - Where do the bytes come from? (DRAM vs L2 vs L1)
%%   - fb_read_sectors (DRAM reads)
%%   - l2_read_transactions (L2 reads)
%%   - tex_cache_transactions (L1/unified cache)
%%   - global_hit_rate (L1 hit rate)
%%   - l2_tex_read_hit_rate (L2 hit rate)
%%   → Answers: is the kernel DRAM-bound or cache-bound?
%%
%% Group 2: COALESCING EFFICIENCY
%%   - How efficiently do loads access memory?
%%   - gld_transactions_per_request (sectors per load)
%%   - gld_efficiency (useful bytes / loaded bytes)
%%   - gld_transactions (total global load transactions)
%%   - gld_requested_throughput vs gld_throughput
%%   → Answers: is the access pattern wasting bandwidth?
%%
%% Group 3: COMPUTE UTILIZATION
%%   - How busy are the SMs?
%%   - achieved_occupancy (warps in flight / max warps)
%%   - warp_execution_efficiency (active threads per warp)
%%   - inst_per_warp (instructions per warp)
%%   - inst_replay_overhead (replayed instructions)
%%   → Answers: is the kernel compute-starved?
%%
%% Group 4: THROUGHPUT BALANCE
%%   - Which subsystem is the bottleneck?
%%   - flop_count_sp (total FLOPs)
%%   - gld_throughput (GB/s achieved for loads)
%%   - l2_read_throughput (L2 bandwidth achieved)
%%   - shared_load_throughput (shared mem bandwidth)
%%   → Answers: compute-bound or memory-bound? Which memory level?
%%
%% Group 5: STALL ANALYSIS
%%   - Why are warps waiting?
%%   - PC sampling stall reasons (our existing CUPTI bridge)
%%   - active_cycles_pm vs elapsed_cycles_sm (SM utilization)
%%   - active_warps_pm (warp occupancy over time)
%%   → Answers: what's stalling the pipeline?
%%
%% Group 6: INTEGER/QUANTIZED COMPUTE
%%   - Q8_0 uses integer multiply-accumulate
%%   - flop_count_sp (should be LOW for integer MMQ)
%%   - inst_integer (integer instruction count)
%%   → Answers: is the kernel using integer or float pipeline?
%%
%% ANALYSIS FRAMEWORK:
%%
%%   The roofline gap decomposes into:
%%
%%   Peak BW (192 GB/s)
%%     × SM utilization (elapsed_cycles / active_cycles)
%%     × Occupancy (achieved_occupancy)
%%     × Memory efficiency (gld_efficiency)
%%     × Cache effects (1 - global_hit_rate if cache helps, or 1 if pure-stream)
%%     = Achieved BW
%%
%%   Each factor is measurable. The product explains the gap.
%%
%% Author: mavchin (2026-06-03)

:- module(gpu_roofline_analysis, [
    roofline_analysis/4,
    explain_gap/4
]).

%% measurement_group(Name, Description, Events).
measurement_group(memory_hierarchy, "Where do bytes come from?", [
    fb_subp0_read_sectors, fb_subp1_read_sectors,
    tex0_cache_sector_queries, tex0_cache_sector_misses,
    elapsed_cycles_sm, active_cycles_pm
]).

measurement_group(coalescing, "How efficiently do loads access memory?", [
    gld_inst_8bit, gld_inst_16bit, gld_inst_32bit,
    gld_inst_64bit, gld_inst_128bit,
    fb_subp0_read_sectors, fb_subp1_read_sectors
]).

measurement_group(compute, "How busy are the SMs?", [
    active_warps_pm, active_cycles_pm,
    elapsed_cycles_sm, elapsed_cycles_pm
]).

measurement_group(l2_cache, "L2 cache behavior", [
    l2_subp0_read_sector_misses, l2_subp1_read_sector_misses,
    l2_subp0_read_tex_sector_queries, l2_subp0_read_tex_hit_sectors
]).

%% roofline_analysis(M, K, N, Analysis)
%% Run all measurement groups, produce a structured analysis.
roofline_analysis(M, K, N, Analysis) :-
    findall(Group-Events-Values,
        (measurement_group(Group, _, Events),
         collect_group(M, K, N, Events, Values)),
        Measurements),
    compute_analysis(M, K, N, Measurements, Analysis).

%% The gap decomposition
explain_gap(M, K, N, Explanation) :-
    roofline_analysis(M, K, N, A),
    PeakBW = 192.0,  % GB/s
    member(achieved_bw-AchievedBW, A),
    Gap is PeakBW - AchievedBW,
    member(sm_utilization-SMUtil, A),
    member(cache_hit_rate-CacheHit, A),
    member(load_efficiency-LoadEff, A),
    Explanation = gap_analysis{
        peak_bw: PeakBW,
        achieved_bw: AchievedBW,
        gap_gbps: Gap,
        gap_pct: Gap / PeakBW * 100,
        factors: [
            sm_utilization: SMUtil,
            cache_hit_rate: CacheHit,
            load_efficiency: LoadEff
        ]
    }.

%% Placeholder for the CUPTI collection (wired via the C bridge)
collect_group(M, K, N, Events, Values) :-
    format("Collecting ~w for matmul [~w x ~w] x ~w~n", [Events, M, K, N]),
    Values = [].  % TODO: wire to gpu_profile_matmul

compute_analysis(_M, _K, _N, _Measurements, []) :-
    true.  % TODO: derive analysis from measurements
