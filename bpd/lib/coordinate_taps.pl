%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% coordinate_taps.pl — derive the coordinate system (taps/hooks) from LIFTED GRAPHS.
%% (Iyun, 2026-05-29.) The "taps" Heath wants: every op in a real model gets a
%% canonical, stable, SEMANTIC coordinate that metas (NLA, probes, quantizers,
%% experimental methods) can attach to — making experiments coordinate-addressed
%% and therefore REPRODUCIBLE ("attach NLA at yolo.layer(2).bot(0).cv(1).conv").
%%
%% Coordinates are DERIVED from the lifted graph (yolo_graph.pl op_kind/2 facts),
%% NOT hand-stubbed. The YOLOv5n op IDs already encode the hierarchy:
%%   l0_conv            -> [yolo, layer(0), conv]
%%   l2_bot0_cv1_conv   -> [yolo, layer(2), bot(0), cv(1), conv]
%% So we PARSE the op-ID structure into a coordinate path. This is the bridge
%% from "there is a model" to "here are its named attachment points."
%%
%% Pairs with cost_naming.pl (the schema): this POPULATES coordinates from a
%% real model. Verified against the actual 199-op YOLOv5n graph.

:- module(coordinate_taps, [
    tap/3,              % tap(?OpId, ?CoordinatePath, ?TapType)  — the derived hooks
    tap_type/2,         % tap_type(+OpKind, -TapType)
    parse_coordinate/2, % parse_coordinate(+OpId, -Path)
    meta_taps/3,        % meta_taps(+MetaName, -OpId, -Path) — where a meta attaches in THIS model
    model_taps/2        % model_taps(+Model, -Count) — how many attachment points
]).

%% requires yolo_graph loaded (op_kind/2, op_inputs/2, op_output/2).

%% ── tap_type: the KIND of attachment point (what metas pattern-match on) ──
tap_type(conv2d,    projection).   % conv = linear projection (quantizers attach)
tap_type(matmul,    projection).
tap_type(batchnorm, reduction).    % bn carries a reduction (mean/var)
tap_type(silu,      activation).
tap_type(add,       residual).     % residual add (AttnRes attaches)
tap_type(concat,    route).        % feature routing
tap_type(maxpool,   reduction).    % spatial reduction (SPPF)
tap_type(upsample,  resample).
tap_type(_,         unknown).      % unrecognized op kinds (deterministic via tap/3 cut below)

%% ── parse_coordinate: op-ID structure -> semantic coordinate path ──
%% l<N>_<segments...>  where final segment is the op-role.
%% Splits on "_", first segment l<N> -> layer(N), known role-segments map,
%% bot<K> -> bot(K), cv<K> -> cv(K), etc.
parse_coordinate(OpId, [yolo, layer(N) | Rest]) :-
    atom_codes(OpId, Codes),
    split_codes(Codes, 0'_, Segs0),
    Segs0 = [First | RestSegs],
    layer_seg(First, N),
    maplist(seg_term, RestSegs, Rest), !.
parse_coordinate(OpId, [yolo, raw(OpId)]).   % fallback: un-parseable -> raw addr

layer_seg(Seg, N) :- atom_codes(A, Seg), atom_concat(l, NumA, A), atom_number(NumA, N).

seg_term(Seg, Term) :-
    atom_codes(A, Seg),
    ( atom_concat(bot, K, A), atom_number(K, KN) -> Term = bot(KN)
    ; atom_concat(cv,  K, A), atom_number(K, KN) -> Term = cv(KN)
    ; Term = A ).

%% tiny code-splitter (split list of codes on a separator code)
split_codes(Codes, Sep, [Seg | Segs]) :-
    append(Seg, [Sep | Rest], Codes), \+ member(Sep, Seg), !,
    split_codes(Rest, Sep, Segs).
split_codes(Codes, _, [Codes]).

%% ── tap/3: the DERIVED coordinate system over the loaded model ──
%% Every op in the lifted graph becomes an addressable tap.
tap(OpId, Path, TapType) :-
    op_kind(OpId, Kind),
    parse_coordinate(OpId, Path),
    once(tap_type(Kind, TapType)).   % ONE specific type per op (199 taps, no double-count)

%% op_tap/2: the UNIVERSAL observer level. EVERY op has an op-coordinate that
%% observer-metas (NLA, probe) attach to, INDEPENDENT of its specific tap-type.
%% This is the two-level model: specific-type taps (for typed metas) + the
%% universal op-tap (for observers). They are different LEVELS of the same hook.
op_tap(OpId, Path) :- op_kind(OpId, _), parse_coordinate(OpId, Path).

%% ── meta_taps: where does a given meta attach IN THIS MODEL? ──
%% A meta declares a TapType it attaches to; meta_taps finds every real
%% coordinate of that type in the loaded graph. This is the reproducible
%% "attach X here" — the published experimental method.
meta_taps(MetaName, OpId, Path) :-
    meta_tap_type(MetaName, op), !,        % observer meta: attaches at EVERY op (universal level)
    op_tap(OpId, Path).
meta_taps(MetaName, OpId, Path) :-
    meta_tap_type(MetaName, TapType),
    TapType \= op,
    tap(OpId, Path, TapType).

%% which tap-type each meta attaches at (bridges cost_naming meta/4 intent)
meta_tap_type(turboquant, projection).   % quantize projections
meta_tap_type(attnres,    residual).     % rewrite residual adds
meta_tap_type(nla,        op).           % observer: attaches at EVERY op-tap
meta_tap_type(probe,      op).           % generic observer

%% ── model_taps: total attachment points (the surface a model exposes) ──
model_taps(yolo, Count) :-
    aggregate_all(count, tap(_, _, _), Count).
