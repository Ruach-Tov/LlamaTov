%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% cost_shape_cupti_validation.pl — Wire cost-shape PREDICTION to CUPTI MEASUREMENT
%%
%% The validation loop:
%%   1. cost-shape PREDICTS a memory bottleneck (eliminable {k}[N] materialization)
%%   2. CUPTI MEASURES actual stall reasons on the GPU
%%   3. If predicted == measured, the fusion is JUSTIFIED
%%   4. After fusion, CUPTI re-measures to confirm stall REDUCED
%%
%% Integration hooks (dynamic):
%%   run_kernel/2     — GPU launch primitive (cuda_launch PLF, mavchin)
%%   propose_fusion/2 — fusion analyzer (Iyun's greedy_fuse)
%%
%% Author: Iyun (2026-05-29), verified against live cupti_bridge.so on Tesla P4
%% Commit: a5796fa (iyun/cost-shape-cupti-validation branch)

:- module(cost_shape_cupti_validation, [
    predicted_bottleneck/2,
    profile_kernel/3,
    fusion_validated/3,
    stall_of/3
]).

%% PREDICTION (cost-shape, no GPU): a kernel with an eliminable {k}[N]
%% producer->consumer seam is memory-bound -> predicted stall = memory_dependency.
predicted_bottleneck(Kernel, memory_dependency) :-
    has_eliminable_materialization(Kernel).

:- dynamic has_eliminable_materialization/1.

%% MEASUREMENT (CUPTI): global reset->run->flush->report cycle.
%% run_kernel/2 = launch hook (to be wired to cuda_launch PLF).
profile_kernel(Kernel, Config, Stalls) :-
    cupti_init,
    cupti_reset,
    run_kernel(Kernel, Config),
    cupti_flush,
    cupti_stall_report(Stalls).

stall_of(Stalls, Reason, Pct) :-
    member(Reason-Pct, Stalls).

%% VALIDATION: predicted==measured==reduced-by-fusion (not just "faster").
%%
%% The validation is TIGHT:
%%   - cost-shape predicts memory_dependency bottleneck
%%   - CUPTI confirms memory_dependency > 30%
%%   - fusion is proposed from cost-shape analysis
%%   - CUPTI re-measures: memory_dependency DECREASED
%%
%% Returns validated(Before, After, Delta) as evidence.
fusion_validated(Kernel, Config, validated(P0, P1, Delta)) :-
    predicted_bottleneck(Kernel, memory_dependency),
    profile_kernel(Kernel, Config, S0),
    stall_of(S0, memory_dependency, P0),
    P0 > 30,
    propose_fusion(Kernel, Fused),
    profile_kernel(Fused, Config, S1),
    stall_of(S1, memory_dependency, P1),
    P1 < P0,
    Delta is P0 - P1.

%% Integration hooks — to be wired by mavchin (launch) and Iyun (fusion)
:- dynamic run_kernel/2.
:- dynamic propose_fusion/2.
