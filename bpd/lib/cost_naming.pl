%% SPDX-License-Identifier: LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% cost_naming.pl — the coordinate nomenclature for ops, models, and metas.
%% (Iyun, 2026-05-29, meta-aware from the start, per Heath.)
%%
%% ONE addressing scheme, THREE uses (the unifying insight):
%%   1. cost-model vocabulary   — graph_complexity scores coordinates
%%   2. fusion-planner addresses— fusion_analyzer fuses at coordinates
%%   3. contribution coordinates— revision-control + meta-composition address space
%%
%% A COORDINATE is a stable, canonical, SEMANTIC address for a structure —
%% NOT a file path. Contributions (op-deltas, metas, hardware-findings) are
%% addressed against coordinates, so they survive file-layout divergence
%% (the private/public fork problem dissolves: name the THING, not the location).
%%
%% Design is SMILES/InChI-style layered (Heath's nomenclature): a coordinate
%% is a path of typed segments; canonical (one name per structure); a meta
%% pattern-matches against coordinate TYPE + path pattern.
%%
%% STATUS (Satya): this lays the FOUNDATION + interface. The canonical-naming
%% algorithm over a full corpus and the meta-composition-VCS are the long-term
%% build; what's here is verified to load + the core queries work. Bridges to
%% the EXISTING meta substrate (model_transform.pl: transform_pattern/4 +
%% transform_replacement/5 — TurboQuant, AttnRes already live there).

:- module(cost_naming, [
    coordinate/3,            % coordinate(?Path, ?Type, ?Attrs)
    coordinate_type/2,       % coordinate_type(+Path, -Type)
    coordinate_canonical/2,  % coordinate_canonical(+Op, -CanonicalName)  (the InChI-style name)
    meta/4,                  % meta(?Name, ?AttachesAt, ?Transform, ?EquivalenceClass)
    meta_attaches/2,         % meta_attaches(+MetaName, -CoordinatePath)  (composition matching)
    meta_composable/3,       % meta_composable(+MetaA, +MetaB, -Verdict)
    contribution/4           % contribution(?Id, ?Coordinate, ?Delta, ?Contract)
]).

%% ─────────────────────────────────────────────────────────────────────────
%% COORDINATE TYPES — what KINDS of address exist (metas pattern-match on these)
%% Derived from the real op taxonomy (classify_op) + structural roles.
%% ─────────────────────────────────────────────────────────────────────────
coordinate_kind(op).               % a single operation
coordinate_kind(block).            % a composed block (attention, ffn, norm-act)
coordinate_kind(model).            % a whole model (llama, yolo)
coordinate_kind(projection).      % linear/matmul projection (TurboQuant attaches here)
coordinate_kind(residual).        % residual add (AttnRes attaches here)
coordinate_kind(reduction).       % a reduction (softmax, sum, mean)
coordinate_kind(quant).           % a quantization point (Q4_K, turboquant)

%% ─────────────────────────────────────────────────────────────────────────
%% COORDINATES — coordinate(Path, Type, Attrs)
%% Path is a list of typed segments: [llama, layer(I), attention, qkv].
%% These are the STABLE SEMANTIC ADDRESSES. A few exemplars (the full set is
%% DERIVED from a model's lifted graph — see coordinate_of_op/3 below).
%% ─────────────────────────────────────────────────────────────────────────
:- discontiguous coordinate/3.
coordinate([llama, layer(_I), attention, qkv],   projection, _).
coordinate([llama, layer(_I), attention, score], reduction, [reduces-seq_sq]).
coordinate([llama, layer(_I), residual],         residual, _).
coordinate([llama, layer(_I), ffn, swiglu],      block, _).
coordinate([yolo, layer(_I), conv_bn_silu],      block, [fused-cbs]).

%% coordinate_type(+Path, -Type)
coordinate_type(Path, Type) :- coordinate(Path, Type, _).

%% coordinate_of_op(+OpKind, +Role, -CoordSegment): map a lifted op to its
%% coordinate TYPE — the bridge from yolo_graph/llama op-facts to coordinates.
coordinate_of_op(K, projection) :- member(K, [conv2d, matmul, ggml_mul_mat]).
coordinate_of_op(K, reduction)  :- member(K, [ggml_soft_max_ext, ggml_sum_rows, ggml_mean, batchnorm]).
coordinate_of_op(K, residual)   :- member(K, [ggml_add]).
coordinate_of_op(_, op).        % default: everything is at least an op-coordinate

%% ─────────────────────────────────────────────────────────────────────────
%% CANONICAL NAMING — InChI/SMILES-style layered name (the nomenclature proper)
%% coordinate_canonical(Op, Name): Name = layers joined; one name per structure.
%% Layer order: topology / shape / cost / precision (cycle isolated to wrapper).
%% Foundation form here; the full canonicalization algorithm is the long build.
%% ─────────────────────────────────────────────────────────────────────────
coordinate_canonical(op(Kind, Shape, Precision), Name) :-
    coordinate_of_op(Kind, Type),
    format(atom(Name), '~w[~w]:~w:~w', [Type, Kind, Shape, Precision]).

%% ─────────────────────────────────────────────────────────────────────────
%% METAS — meta(Name, AttachesAt, Transform, EquivalenceClass)
%%   Name             : the meta's identifier (turboquant, attnres, nla, ...)
%%   AttachesAt       : a coordinate PATTERN (type or path-pattern) it matches
%%   Transform        : the transform predicate (in model_transform.pl)
%%   EquivalenceClass : what it PRESERVES — bit_exact | tolerance(Eps) |
%%                      mathematical | lossy(reason)  (medayek's field, generalized)
%%
%% This bridges to the EXISTING meta substrate: TurboQuant + AttnRes already
%% live as model_transform:transform_pattern/4. Here they get a COORDINATE
%% address + a declared contract, making them composable + verifiable.
%% ─────────────────────────────────────────────────────────────────────────
:- discontiguous meta/4.

%% TurboQuant: polar-quantizes K/V projections. Attaches at projection
%% coordinates feeding attention. Lossy by design (it's quantization).
meta(turboquant,
     attaches_at(coordinate_type(projection), role(kv_projection)),
     transform(model_transform, turboquant),
     lossy(polar_quantization_kv)).

%% AttnRes: replaces residual adds with learned attention over prior layers.
%% Attaches at residual coordinates. Changes the math (it's a new architecture).
meta(attnres,
     attaches_at(coordinate_type(residual), role(skip_connection)),
     transform(model_transform, attnres),
     mathematical(learned_attention_residual)).

%% NLA (Natural Language Autoencoder): an interpretability ASPECT — observes,
%% does not change the forward pass. Attaches at any op coordinate. bit_exact
%% (observation only). Placeholder for the Anthropic-NLA-style contributed meta.
meta(nla,
     attaches_at(coordinate_type(op), role(any)),
     transform(model_transform, nla_probe),
     bit_exact).  % observation-only metas preserve the computation exactly

%% meta_attaches(+MetaName, -CoordinatePath): which coordinates a meta composes
%% with — the composition-matching query. A meta attaches at every coordinate
%% whose TYPE matches its AttachesAt pattern.
meta_attaches(MetaName, Path) :-
    meta(MetaName, attaches_at(coordinate_type(Type), _Role), _, _),
    coordinate(Path, Type, _).

%% ─────────────────────────────────────────────────────────────────────────
%% META COMPOSITION — can two metas coexist at overlapping coordinates?
%% The composition algebra (foundation). Verdict the contract logic needs.
%% ─────────────────────────────────────────────────────────────────────────
%% meta_composable(+MetaA, +MetaB, -Verdict)
%%   compatible        : disjoint coordinates, or order-independent
%%   ordered(A,B)      : both attach at a shared coordinate; order matters
%%   conflict(Reason)  : both rewrite the SAME coordinate destructively
meta_composable(MetaA, MetaB, Verdict) :-
    findall(P, meta_attaches(MetaA, P), PsA),
    findall(P, meta_attaches(MetaB, P), PsB),
    ( \+ (member(X, PsA), member(X, PsB)) ->
        Verdict = compatible              % disjoint coordinates: freely composable
    ;   meta(MetaA, _, _, EqA), meta(MetaB, _, _, EqB),
        ( ( EqA = bit_exact ; EqB = bit_exact ) ->
            Verdict = ordered(observe_after_transform)  % observers compose after transforms
        ;   Verdict = conflict(both_rewrite_shared_coordinate)
        )
    ).

%% ─────────────────────────────────────────────────────────────────────────
%% CONTRIBUTIONS — the revision-control unit: an addressed delta with a contract
%% contribution(Id, Coordinate, Delta, Contract). The file tree is a PROJECTION;
%% THIS is the source of truth for what a contributor submitted + against what.
%%   Delta : op_impl(NewClause) | meta(MetaName) | finding(Data) | fusion_rule(Clause)
%% Acceptance gate (the build/test gate Heath specified): a contribution is
%% accepted iff composed at its Coordinate, the correctness harness still passes
%% per its declared Contract. (Gate lives in CI; this is the addressing layer.)
%% ─────────────────────────────────────────────────────────────────────────
:- discontiguous contribution/4.

%% Example: Collin's hardware-coverage finding, addressed at a coordinate.
contribution(collin_ampere_sgemm,
             coordinate([blas, sgemm, square(2048)], op, [arch-sm_86]),
             finding(ulp(data_dependent, [24576, 4864, 7168, 1152, 2560])),
             measured(rtx_3090)).
