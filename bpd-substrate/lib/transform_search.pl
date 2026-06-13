%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════
%% Licensed under RTAAL-1.0 (Ruach Tov AI Agent License)
%% See: LICENSE-RTAAL-1-0.md in the project root
%% ═══════════════════════════════════════════════════════════════════════

%% transform_search.pl — Search over model transformation space
%%
%% A framework for exploring different combinations and orderings of
%% model transformations to find the optimal rewritten graph.

:- module(transform_search, [
    optimal_transformed_graph/4
]).

:- use_module(model_transform).
:- use_module(graph_optimizer).

%% ────────────────────────────────────────────────────────────────────
%% optimal_transformed_graph(+AvailableTransforms, +InitialFacts, +Platform, -BestPlan)
%% ────────────────────────────────────────────────────────────────────
%% Searches over all valid subsets and permutations of AvailableTransforms,
%% applies them to InitialFacts, fuses the resulting graph, and selects
%% the one with the lowest cost.
%%
%% For this proof-of-concept, we just enumerate all permutations of subsets
%% and use a dummy cost function (e.g., number of remaining ops).

optimal_transformed_graph(AvailableTransforms, InitialFacts, Platform, BestPlan) :-
    findall(
        cost_plan(Cost, FusedFacts, Permutation),
        (
            %% 1. Pick a subset of transforms
            subset(AvailableTransforms, Subset),
            %% 2. Pick an ordering (permutation)
            permutation(Subset, Permutation),
            %% 3. Apply the transforms in that order
            apply_all_transforms(Permutation, InitialFacts, TransformedFacts),
            %% 4. Run the fusion optimizer on the rewritten graph
            fuse_graph(TransformedFacts, Platform, FusedFacts),
            %% 5. Evaluate the cost of the final fused graph
            evaluate_cost(FusedFacts, Cost)
        ),
        ScoredPlans
    ),
    %% Sort by cost (ascending) and pick the first one
    sort(ScoredPlans, [cost_plan(_MinCost, BestPlan, _BestPermutation)|_]).

%% Helper: generate all subsets
subset([], []).
subset([E|Tail], [E|NTail]) :- subset(Tail, NTail).
subset([_|Tail], NTail) :- subset(Tail, NTail).

%% Helper: Dummy cost evaluation
%% In a real implementation, this would sum the execution time of each op
%% based on the platform hardware parameters.
%% For now, fewer ops = lower cost.
evaluate_cost(Facts, Cost) :-
    findall(Op, member(op(Op), Facts), Ops),
    length(Ops, Cost).
