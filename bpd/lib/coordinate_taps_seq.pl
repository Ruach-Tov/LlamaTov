%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% coordinate_taps_seq.pl — coordinate taps over the SEQUENCE op-representation.
%% (Iyun, 2026-05-29.) Generalizes the tap/coordinate system to the SECOND graph
%% form used in the codebase: op(Name, GgmlKind, Seq) lists (transformer layers,
%% KernelBench L2/L3, fusion_analyzer's input) — distinct from yolo_graph.pl's
%% op_kind(Id, Kind) + structured-ID form (handled by coordinate_taps.pl).
%%
%% WHY TWO: measured fact — YOLO is lifted as op_kind/2 facts w/ parseable IDs;
%% llama/L3/KernelBench are op(K,K,Seq) sequences. To take "all models up to
%% lifted-and-derived" (Heath), the deriver must serve BOTH. Same TAP TYPES +
%% same META attachment semantics; only the graph-walk differs.
%%
%% NOTE on full llama: a complete llama model graph is materialized by
%% llama_cpp_lifter:lift_arch_graph_ops/2, which needs the llama.cpp SOURCE path
%% (it lifts FROM source). Not assumed present. This file derives taps from the
%% op(K,K,Seq) form, exercised on the real L3 19-op transformer layer.

:- module(coordinate_taps_seq, [
    seq_tap/3,        % seq_tap(+Ops, ?Seq, ?TapType) — taps over an op-sequence
    seq_coordinate/3, % seq_coordinate(+Arch, +Op, -Path)
    seq_meta_taps/4,  % seq_meta_taps(+Arch, +MetaName, +Ops, -Taps)
    ggml_tap_type/2
]).

%% ── ggml op-kind -> tap type (the ggml_* vocabulary, vs YOLO's conv2d/silu) ──
ggml_tap_type(ggml_mul_mat,      projection).
ggml_tap_type(ggml_soft_max_ext, reduction).
ggml_tap_type(ggml_rms_norm,     reduction).    % norm carries a reduction
ggml_tap_type(ggml_norm,         reduction).
ggml_tap_type(ggml_sum_rows,     reduction).
ggml_tap_type(ggml_mean,         reduction).
ggml_tap_type(ggml_add,          residual).
ggml_tap_type(ggml_silu,         activation).
ggml_tap_type(ggml_gelu,         activation).
ggml_tap_type(ggml_mul,          elementwise).
ggml_tap_type(ggml_scale,        elementwise).
ggml_tap_type(ggml_reshape_2d,   reshape).
ggml_tap_type(ggml_reshape_3d,   reshape).

specific_ggml(K) :- ggml_tap_type(K, _).

%% ── seq_tap/3: one tap per op in the sequence, by its ggml kind ──
%% Ops is a list of op(Name, Kind, Seq). Deterministic specific type.
seq_tap(Ops, Seq, TapType) :-
    member(op(_, Kind, Seq), Ops),
    once(ggml_tap_type(Kind, TapType)).

%% ── seq_coordinate/3: semantic coordinate for a sequence op ──
%% [Arch, op(Seq), kind(K)] — a transformer layer has no rich nesting like YOLO's
%% IDs, so the coordinate is arch + sequence-position + kind. Stable + addressable.
seq_coordinate(Arch, op(_Name, Kind, Seq), [Arch, layer(0), op(Seq), Kind]).

%% ── seq_meta_taps/4: where a meta attaches in this op-sequence ──
%% Observer metas (tap-type op) attach at EVERY op (universal level);
%% typed metas attach at their specific tap-type. (Same two-level model as
%% coordinate_taps.pl — observers are universal, typed metas are specific.)
seq_meta_taps(Arch, MetaName, Ops, Taps) :-
    meta_tap_kind(MetaName, MetaKind),
    ( MetaKind == op ->
        findall(Coord,
            ( member(O, Ops), seq_coordinate(Arch, O, Coord) ), Taps)   % universal
    ;   findall(Coord,
            ( member(op(N,K,S), Ops), once(ggml_tap_type(K, MetaKind)),
              seq_coordinate(Arch, op(N,K,S), Coord) ), Taps)           % specific
    ).

%% which tap-kind each meta attaches at (shared intent w/ cost_naming meta/4)
meta_tap_kind(nla,        op).           % observer: every op
meta_tap_kind(probe,      op).
meta_tap_kind(turboquant, projection).   % the matmuls
meta_tap_kind(attnres,    residual).     % the adds
meta_tap_kind(flashattn,  reduction).    % attaches at softmax reductions
