%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
:- module(cupti_profile, [
    optimization_needed/2,
    stall_threshold/3,
    profile_summary/1,
    optimal_config/3,
    assert_profile/3,
    query_profiles/2,
    structural_bottleneck/2
]).

/** <module> CUPTI GPU Kernel Profiling - Phase 4 continuous loop

@author Ruach Tov Collective
*/

stall_threshold(memory_dependency, 30, warp_shuffle).
stall_threshold(memory_dependency, 15, shared_memory).
stall_threshold(exec_dependency, 20, increase_ilp).
stall_threshold(sync, 15, reduce_barriers).
stall_threshold(memory_throttle, 10, coalesce_memory).
stall_threshold(constant_memory, 10, use_registers).

optimization_needed(StallList, Optimization) :-
    member(StallType-Pct, StallList),
    stall_threshold(StallType, Threshold, Optimization),
    Pct > Threshold.

profile_summary(StallList) :-
    format('~n=== Stall Analysis ===~n'),
    forall(member(Type-Pct, StallList), format('  ~w: ~1f~n', [Type, Pct])),
    format('~n=== Suggestions ===~n'),
    forall(optimization_needed(StallList, Opt), format('  -> ~w~n', [Opt])).

:- dynamic kernel_profile/4.

assert_profile(KernelName, Config, StallList) :-
    get_time(Now),
    assertz(kernel_profile(KernelName, Config, StallList, Now)),
    format('  Recorded: ~w @ ~w~n', [KernelName, Config]).

query_profiles(KernelName, Profiles) :-
    findall(profile(Config, StallList, Time),
        kernel_profile(KernelName, Config, StallList, Time), Profiles).

optimal_config(KernelName, BestConfig, BestIssuing) :-
    findall(Issue-Config,
        (kernel_profile(KernelName, Config, Stalls, _), member(none-Issue, Stalls)),
        Pairs),
    Pairs \= [],
    sort(0, @>=, Pairs, [BestIssuing-BestConfig|_]).

structural_bottleneck(KernelName, Diagnosis) :-
    findall(_, kernel_profile(KernelName, _, _, _), AllProfiles),
    length(AllProfiles, NumConfigs),
    NumConfigs > 0,
    findall(StallType,
        (kernel_profile(KernelName, _, Stalls, _),
         member(StallType-Pct, Stalls), Pct > 30),
        AllDominant),
    sort(AllDominant, UniqueTypes),
    member(DomType, UniqueTypes),
    findall(_, (kernel_profile(KernelName, _, S, _), member(DomType-P, S), P > 30), Matches),
    length(Matches, M),
    M >= NumConfigs,
    format(atom(Diagnosis),
        '~w is structural: dominates all ~w configs. Change the algorithm.',
        [DomType, NumConfigs]).
