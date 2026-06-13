%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════
%% Licensed under RTAAL-1.0 (Ruach Tov AI Agent License)
%% See: LICENSE-RTAAL-1-0.md in the project root
%% ═══════════════════════════════════════════════════════════════════════

%% graph_optimizer.pl — Full optimizer that derives optimal fused plans
%%
%% Iteratively applies fusion rules to a compute graph until a fixed point
%% is reached.

/** <module> Fixed-Point Graph Optimizer

Iteratively applies fusion rules from fusible.pl to a compute graph
until no more fusions are possible. Produces a FusedPlan describing
which operations are fused and what tile parameters to use.

@author Ruach Tov Collective
@license RTAAL-1.0
@see fusible.pl for the fusion legality rules
@see valid_tile.pl for tile constraint solving
*/

:- module(graph_optimizer, [
    fuse_graph/3
]).

:- use_module(fusible, [fusible_pair/4]).
:- use_module(apply_fusion, [apply_fusion_to_facts/3]).

%% ────────────────────────────────────────────────────────────────────
%% fuse_graph(+GraphFacts, +Platform, -FusedPlan)
%% ────────────────────────────────────────────────────────────────────
%% Takes a list of BPD facts representing the compute graph,
%% applies valid fusions, and returns the optimized graph.

%! fuse_graph(+GraphFacts, +Platform, -FusedPlan) is det.
%  Optimize a compute graph for Platform by iteratively applying
%  fusion rules until fixed-point. Returns the optimized FusedPlan.
fuse_graph(GraphFacts, Platform, FusedPlan) :-
    fuse_graph_iter(GraphFacts, Platform, FusedPlan).

%% Iteratively find and apply fusions until no more are possible.
fuse_graph_iter(Facts, Platform, FinalFacts) :-
    %% Find the first valid fusion pair
    (   find_fusion(Facts, Fusion)
    ->  %% Apply it
        apply_fusion_to_facts(Facts, Fusion, NewFacts),
        %% Recurse
        fuse_graph_iter(NewFacts, Platform, FinalFacts)
    ;   %% Fixed point reached
        FinalFacts = Facts
    ).

%% Find a valid fusion pair in the graph.
find_fusion(Facts, Fusion) :-
    %% We need to pick two ops that are fusible.
    %% To avoid infinite loops, we just pick the first valid one.
    member(op(Op1), Facts),
    member(op(Op2), Facts),
    Op1 \= Op2,
    fusible_pair(Facts, Op1, Op2, Fusion),
    !. % Take the first one we find
