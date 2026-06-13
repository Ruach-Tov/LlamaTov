%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════
%% Licensed under RTAAL-1.0 (Ruach Tov AI Agent License)
%% See: LICENSE-RTAAL-1-0.md in the project root
%% ═══════════════════════════════════════════════════════════════════════

%% model_transform.pl — General framework for structural model rewriting
%%
%% A model transformation is a pattern-match-and-replace operation on BPD facts.
%% Unlike fusion, which merges ops for execution efficiency, transformations
%% change the semantic structure or mathematical operations of the model
%% (e.g., AttnRes, TurboQuant) before fusion begins.

:- module(model_transform, [
    apply_transform/4,        % +TransformName, +InputFacts, -OutputFacts, -Applied
    apply_all_transforms/3    % +Transforms, +InputFacts, -OutputFacts
]).

%% Multifile hooks for concrete transforms to implement
:- multifile transform_pattern/4.     % transform_pattern(Name, Facts, Subgraph, Context)
:- multifile transform_replacement/5. % transform_replacement(Name, Facts, Subgraph, Context, NewFacts)

%% ────────────────────────────────────────────────────────────────────
%% apply_transform(+TransformName, +Facts, -NewFacts, -Applied)
%% ────────────────────────────────────────────────────────────────────
%% Applies a single transform to the graph until a fixed point is reached.
%% Applied is true if at least one instance of the pattern was transformed.

apply_transform(TransformName, Facts, FinalFacts, Applied) :-
    (   transform_step(TransformName, Facts, NextFacts)
    ->  apply_transform(TransformName, NextFacts, FinalFacts, _)
    ,   Applied = true
    ;   FinalFacts = Facts
    ,   Applied = false
    ).

%% Single step of a transform
transform_step(TransformName, Facts, NextFacts) :-
    %% 1. Find a matching subgraph and its context
    transform_pattern(TransformName, Facts, Subgraph, Context),
    
    %% 2. Generate the replacement facts
    transform_replacement(TransformName, Facts, Subgraph, Context, ReplacementFacts),
    
    %% 3. Remove the old subgraph facts
    subtract(Facts, Subgraph, TempFacts),
    
    %% 4. Insert the new replacement facts
    append(TempFacts, ReplacementFacts, NextFacts),
    !. % Only apply one at a time to allow Facts list to update

%% ────────────────────────────────────────────────────────────────────
%% apply_all_transforms(+TransformsList, +Facts, -FinalFacts)
%% ────────────────────────────────────────────────────────────────────
%% Applies a list of transforms in order.
%% E.g. apply_all_transforms([turboquant, attnres], Facts, FinalFacts).

apply_all_transforms([], Facts, Facts).
apply_all_transforms([T|Ts], Facts, FinalFacts) :-
    apply_transform(T, Facts, NextFacts, _),
    apply_all_transforms(Ts, NextFacts, FinalFacts).

%% ────────────────────────────────────────────────────────────────────
%% Utility predicates for writing transforms
%% ────────────────────────────────────────────────────────────────────

%% Helper: Find an op by kind
find_op(Facts, Kind, Op) :-
    member(op(Op), Facts),
    member(op_kind(Op, Kind), Facts).

%% Helper: Get inputs of an op
get_inputs(Facts, Op, Inputs) :-
    member(op_inputs(Op, Inputs), Facts).

%% Helper: Get output of an op
get_output(Facts, Op, Output) :-
    member(op_output(Op, Output), Facts).
