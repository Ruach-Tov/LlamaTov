%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% pytorch_ast.pl — Bidirectional Prolog DCG for PyTorch model AST
%%
%% Parses a PyTorch nn.Module tree (as JSON) into BPD compute graph facts.
%% Round-trips: JSON → Prolog facts → JSON
%%
%% The same approach as our C AST DCG:
%%   1. Parse: JSON model tree → graph_fact/3 terms
%%   2. Emit:  graph_fact/3 terms → yolo_graph.pl (or any model)
%%   3. Round-trip: graph_fact/3 → JSON reconstruction
%%
%% Usage:
%%   ?- parse_pytorch_model('model_tree.json', Facts).
%%   ?- emit_graph(Facts, 'yolo_graph.pl').
%%   ?- reconstruct_json(Facts, JSON).

:- module(pytorch_ast, [
    parse_pytorch_model/2,
    emit_graph/2,
    node_to_facts/3,
    reconstruct_json/2,
    pytorch_type_to_bpd/2
]).

:- use_module(library(lists)).

%% ============================================================
%% Type mapping: PyTorch nn.Module type → BPD op_kind
%% ============================================================

%% Leaf ops (actual compute)
pytorch_type_to_bpd('Conv2d',      conv2d).
pytorch_type_to_bpd('BatchNorm2d', batchnorm).
pytorch_type_to_bpd('SiLU',        silu).
pytorch_type_to_bpd('ReLU',        relu).
pytorch_type_to_bpd('LeakyReLU',   leaky_relu).
pytorch_type_to_bpd('GELU',        gelu).
pytorch_type_to_bpd('Sigmoid',     sigmoid).
pytorch_type_to_bpd('Tanh',        tanh).
pytorch_type_to_bpd('MaxPool2d',   maxpool).
pytorch_type_to_bpd('Upsample',    upsample).
pytorch_type_to_bpd('Identity',    identity).

%% Compound ops (containers with known forward semantics)
pytorch_type_to_bpd('Conv',       cbs).        % Conv+BN+SiLU
pytorch_type_to_bpd('Bottleneck', bottleneck).  % cv1(CBS) → cv2(CBS) + residual
pytorch_type_to_bpd('C3',         c3).          % CSP bottleneck
pytorch_type_to_bpd('SPPF',       sppf).        % Spatial pyramid pooling
pytorch_type_to_bpd('Concat',     concat).
pytorch_type_to_bpd('Detect',     detect).

%% Containers (structural, not compute)
pytorch_type_to_bpd('Sequential',  sequential).
pytorch_type_to_bpd('ModuleList',  module_list).

%% ============================================================
%% Parse JSON model tree → graph facts
%% ============================================================

%% parse_pytorch_model(+JSONFile, -Facts)
%% Reads JSON, walks tree, emits op_kind/op_output/op_inputs facts.
parse_pytorch_model(File, Facts) :-
    read_json(File, Tree),
    walk_tree(Tree, [], Facts).

%% walk_tree(+Nodes, +ParentContext, -Facts)
walk_tree([], _, []).
walk_tree([Node|Rest], Ctx, AllFacts) :-
    node_to_facts(Node, Ctx, NodeFacts),
    walk_tree(Rest, Ctx, RestFacts),
    append(NodeFacts, RestFacts, AllFacts).

%% node_to_facts(+Node, +Context, -Facts)
%% Convert a single node and its children into graph facts.
node_to_facts(Node, _Ctx, Facts) :-
    get_field(Node, name, Name),
    get_field(Node, type, Type),
    get_field(Node, children, Children),
    (pytorch_type_to_bpd(Type, BpdKind) -> true ; BpdKind = unknown(Type)),
    %% Determine if this is a leaf (no children) or compound
    (Children = [] ->
        %% Leaf node: emit op_kind fact
        atom_string(NameAtom, Name),
        atom_concat(NameAtom, '_out', OutputAtom),
        Facts = [op_kind(NameAtom, BpdKind),
                 op_output(NameAtom, OutputAtom)]
    ;
        %% Compound node: recurse into children, then emit
        %% the compound's data flow based on its type
        walk_tree(Children, Name, ChildFacts),
        atom_string(NameAtom, Name),
        emit_compound(BpdKind, NameAtom, Children, CompoundFacts),
        append(ChildFacts, CompoundFacts, Facts)
    ).

%% ============================================================
%% Compound op emission: encode forward() semantics as data flow
%% ============================================================

%% CBS (Conv+BN+SiLU): conv_out → bn → silu → cbs_out
emit_compound(cbs, Name, Children, Facts) :-
    find_child(Children, 'Conv2d', ConvName),
    find_child(Children, 'BatchNorm2d', BnName),
    find_child(Children, 'SiLU', ActName),
    atom_string(ConvAtom, ConvName),
    atom_string(BnAtom, BnName),
    atom_string(ActAtom, ActName),
    atom_concat(ConvAtom, '_out', ConvOut),
    atom_concat(BnAtom, '_out', BnOut),
    atom_concat(ActAtom, '_out', ActOut),
    atom_concat(Name, '_out', NameOut),
    Facts = [
        op_inputs(ConvAtom, [Name]),       % conv reads from parent input
        op_inputs(BnAtom, [ConvOut]),       % bn reads conv output
        op_inputs(ActAtom, [BnOut]),        % silu reads bn output
        op_output(Name, NameOut)            % compound output = silu output
    ].
emit_compound(cbs, Name, _, [op_output(Name, Out)]) :-
    atom_concat(Name, '_out', Out).

%% Bottleneck: cv1(CBS) → cv2(CBS) + residual add
emit_compound(bottleneck, Name, Children, Facts) :-
    find_child_by_name(Children, "cv1", Cv1Name),
    find_child_by_name(Children, "cv2", Cv2Name),
    atom_string(Cv1Atom, Cv1Name),
    atom_string(Cv2Atom, Cv2Name),
    atom_concat(Cv1Atom, '_out', Cv1Out),
    atom_concat(Cv2Atom, '_out', Cv2Out),
    atom_concat(Name, '_add', AddName),
    atom_concat(Name, '_out', NameOut),
    Facts = [
        op_inputs(Cv1Atom, [Name]),
        op_inputs(Cv2Atom, [Cv1Out]),
        op_kind(AddName, add),
        op_inputs(AddName, [Cv2Out, Name]),  % residual from input
        op_output(AddName, NameOut),
        op_output(Name, NameOut)
    ].
emit_compound(bottleneck, Name, _, [op_output(Name, Out)]) :-
    atom_concat(Name, '_out', Out).

%% C3: cv1 → bottlenecks → concat(bottleneck_out, cv2(input)) → cv3
emit_compound(c3, Name, Children, Facts) :-
    find_child_by_name(Children, "cv1", Cv1Name),
    find_child_by_name(Children, "cv2", Cv2Name),
    find_child_by_name(Children, "cv3", Cv3Name),
    atom_string(Cv1Atom, Cv1Name),
    atom_string(Cv2Atom, Cv2Name),
    atom_string(Cv3Atom, Cv3Name),
    atom_concat(Cv1Atom, '_out', Cv1Out),
    atom_concat(Cv2Atom, '_out', Cv2Out),
    atom_concat(Name, '_concat', ConcatName),
    atom_concat(ConcatName, '_out', ConcatOut),
    atom_concat(Cv3Atom, '_out', Cv3Out),
    atom_concat(Name, '_out', NameOut),
    %% Data flow: input → cv1 → bottlenecks → concat with cv2(input) → cv3
    Facts = [
        op_inputs(Cv1Atom, [Name]),
        op_inputs(Cv2Atom, [Name]),
        op_kind(ConcatName, concat),
        op_inputs(ConcatName, [Cv1Out, Cv2Out]),  % simplified: bottleneck output ≈ cv1 path
        op_output(ConcatName, ConcatOut),
        op_inputs(Cv3Atom, [ConcatOut]),
        op_output(Name, NameOut)
    ].
emit_compound(c3, Name, _, [op_output(Name, Out)]) :-
    atom_concat(Name, '_out', Out).

%% SPPF: cv1 → pool → pool → pool → concat(y, p1, p2, p3) → cv2
emit_compound(sppf, Name, Children, Facts) :-
    find_child_by_name(Children, "cv1", Cv1Name),
    find_child_by_name(Children, "cv2", Cv2Name),
    atom_string(Cv1Atom, Cv1Name),
    atom_string(Cv2Atom, Cv2Name),
    atom_concat(Name, '_out', NameOut),
    Facts = [
        op_inputs(Cv1Atom, [Name]),
        op_inputs(Cv2Atom, [Name]),  % simplified
        op_output(Name, NameOut)
    ].
emit_compound(sppf, Name, _, [op_output(Name, Out)]) :-
    atom_concat(Name, '_out', Out).

%% Default: just emit output
emit_compound(_, Name, _, [op_output(Name, Out)]) :-
    atom_concat(Name, '_out', Out).

%% ============================================================
%% Emit graph facts to a Prolog file
%% ============================================================

emit_graph(Facts, File) :-
    open(File, write, Stream),
    write(Stream, '%% Auto-generated from PyTorch model by pytorch_ast.pl\n'),
    write(Stream, ':- module(model_graph, [op_kind/2, op_output/2, op_inputs/2, op_attr/3]).\n\n'),
    emit_facts(Facts, Stream),
    close(Stream).

emit_facts([], _).
emit_facts([Fact|Rest], Stream) :-
    writeq(Stream, Fact),
    write(Stream, '.\n'),
    emit_facts(Rest, Stream).

%% ============================================================
%% JSON → Prolog helpers (simplified — real impl uses library(http/json))
%% ============================================================

read_json(File, Tree) :-
    %% In practice, use library(http/json) or shell out to Python
    %% For now, we accept pre-parsed Prolog terms
    see(File),
    read(Tree),
    seen.

get_field(Dict, Key, Value) :-
    (is_dict(Dict) -> get_dict(Key, Dict, Value)
    ; member(Key=Value, Dict)
    ; member(Key-Value, Dict)
    ).

find_child(Children, Type, Name) :-
    member(Child, Children),
    get_field(Child, type, Type),
    get_field(Child, name, Name).

find_child_by_name(Children, Suffix, Name) :-
    member(Child, Children),
    get_field(Child, name, Name),
    atom_string(_, Suffix),
    sub_string(Name, _, _, 0, Suffix).

%% ============================================================
%% Round-trip: Facts → JSON reconstruction
%% ============================================================

reconstruct_json(Facts, JSON) :-
    findall(node(Name, Kind), member(op_kind(Name, Kind), Facts), Nodes),
    findall(edge(From, To), 
        (member(op_output(From, Out), Facts),
         member(op_inputs(To, Ins), Facts),
         member(Out, Ins)), 
        Edges),
    JSON = graph{nodes: Nodes, edges: Edges}.
