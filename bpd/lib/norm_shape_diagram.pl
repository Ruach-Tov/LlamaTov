%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% norm_shape_diagram.pl — activation-record shape diagrams for the four
%% ggml norm ops (rms_norm, layer_norm, l2_norm, group_norm).
%%
%% Substrate-of-source: Iyun and mavhir's meeting decomposition 2026-05-29
%% 00:48 — 01:49 UTC, cross-referenced against canonical ggerganov/ggml
%% src/ggml.c.
%%
%% Substrate-of-rendering: visualizes the OPERAND SIGNATURE as geometric shapes,
%% per Heath's direction (2026-05-29): "see whether memory regions are
%% 1d vs 2d vs scalar". Memory regions render as labelled rectangles, sized
%% proportionally to their shape. Reductions render as arrows collapsing the
%% reduced dimension. Broadcasts render as arrows fanning out from a scalar.
%%
%% LCARS motif: rectangular regions in distinct colors per region-kind
%% (scalar, 1D, 2D), readable at-a-glance.
%%
%% Substrate-of-extension: this is the detailed per-op view. Iyun will fold
%% in a taxonomy-class view (map / scaled_map / reduce / binary / rowwise)
%% reading live from bpd/lib/op_signatures.pl (55 sigs, ce5c410b), as a
%% co-linear projection. See ls-by-origin-story --lang=... for the
%% co-linear-output pattern; same shape here.
%%
%% Render (default = all four norms to /tmp/output-only/):
%%   swipl -g main bpd/lib/norm_shape_diagram.pl
%% Render single op programmatically:
%%   ?- emit_norm_shape_svg(rms_norm, '/tmp/rms.svg').

:- module(norm_shape_diagram, [
    emit_norm_shape_svg/2,
    region_color/2,    %% LCARS palette shared with kernel_taxonomy_diagram
    main/0
]).

:- use_module(library(lists)).

%% ─────────────────────────────────────────────────────────────────────────────
%% Substrate-of-record: operand signatures per norm op
%% (Provenance: Iyun's meeting decomposition + canonical ggml verification)
%%
%% operand(OpName, OperandName, Role, Shape).
%%   Role: input | output | parameter | intermediate
%%   Shape: scalar | vec(Dim) | mat(Outer, Inner)
%% ─────────────────────────────────────────────────────────────────────────────

%% rms_norm: y = x * rsqrt(mean(x²) + eps)  (no gamma in canonical ggml op)
operand(rms_norm, x,       input,        mat(outer, n)).
operand(rms_norm, eps,     parameter,    scalar).
operand(rms_norm, mean_sq, intermediate, vec(outer)).  %% per-row scalar
operand(rms_norm, inv_rms, intermediate, vec(outer)).  %% per-row scalar
operand(rms_norm, y,       output,       mat(outer, n)).

%% layer_norm: y = (x - mean) * rsqrt(var + eps)  (no gamma/beta in canonical)
operand(layer_norm, x,       input,        mat(outer, n)).
operand(layer_norm, eps,     parameter,    scalar).
operand(layer_norm, mean,    intermediate, vec(outer)).
operand(layer_norm, var,     intermediate, vec(outer)).
operand(layer_norm, inv_std, intermediate, vec(outer)).
operand(layer_norm, y,       output,       mat(outer, n)).

%% l2_norm: y = x * rsqrt(SUM(x²) + eps)  (note: SUM not mean — l2/rms distinction)
operand(l2_norm, x,         input,        mat(outer, n)).
operand(l2_norm, eps,       parameter,    scalar).
operand(l2_norm, sum_sq,    intermediate, vec(outer)).
operand(l2_norm, inv_l2,    intermediate, vec(outer)).
operand(l2_norm, y,         output,       mat(outer, n)).

%% group_norm: per-group reduction over reshape [N, G, C/G]
%% Iyun's note: B2-Norm-Extended roadmap; n_groups is first-class ggml param
operand(group_norm, x,         input,        mat(outer, n)).        %% pre-reshape
operand(group_norm, n_groups,  parameter,    scalar).
operand(group_norm, eps,       parameter,    scalar).
operand(group_norm, mean,      intermediate, mat(outer, n_groups)). %% per-group scalar
operand(group_norm, var,       intermediate, mat(outer, n_groups)).
operand(group_norm, y,         output,       mat(outer, n)).

%% ─────────────────────────────────────────────────────────────────────────────
%% Reduction structure: which operand collapses along which axis to produce which
%% reduction(OpName, From, Axis, To, Kind).
%%   Kind: mean_of_squares | sum_of_squares | mean_centered_variance | mean
%% ─────────────────────────────────────────────────────────────────────────────

reduction(rms_norm,   x, inner, mean_sq, mean_of_squares).
reduction(layer_norm, x, inner, mean,    mean).
reduction(layer_norm, x, inner, var,     mean_centered_variance).
reduction(l2_norm,    x, inner, sum_sq,  sum_of_squares).
reduction(group_norm, x, group, mean,    mean).
reduction(group_norm, x, group, var,     mean_centered_variance).

%% Broadcast structure: which scalar/vector gets stretched back across which axis
broadcast(rms_norm,   inv_rms, outer_only, multiply).
broadcast(layer_norm, mean,    outer_only, subtract_then_multiply).
broadcast(layer_norm, inv_std, outer_only, multiply).
broadcast(l2_norm,    inv_l2,  outer_only, multiply).
broadcast(group_norm, mean,    group,      subtract_then_multiply).
broadcast(group_norm, var,     group,      multiply).


%% ─────────────────────────────────────────────────────────────────────────────
%% Visual constants — LCARS-aligned palette
%% ─────────────────────────────────────────────────────────────────────────────

region_color(input,        '#3a6b88').  %% deep blue — reading from
region_color(output,       '#3f7c5c').  %% green — produced
region_color(parameter,    '#daa520').  %% amber — controls
region_color(intermediate, '#7a5a8a').  %% violet — derived in flight
region_color(reduction,    '#8a3a2a').  %% red — collapses
region_color(broadcast,    '#5fa07c').  %% light green — stretches

bg_color('#f8f5ee').
ink_color('#3a2a1a').


%% ─────────────────────────────────────────────────────────────────────────────
%% SVG emitter: one diagram per op
%% ─────────────────────────────────────────────────────────────────────────────

emit_norm_shape_svg(Op, Path) :-
    SvgW = 1100, SvgH = 540,
    setup_call_cleanup(
        open(Path, write, S),
        ( bg_color(BG), ink_color(Ink),
          format(S, '<svg xmlns="http://www.w3.org/2000/svg" width="~w" height="~w" viewBox="0 0 ~w ~w" font-family="Georgia, serif">~n',
              [SvgW, SvgH, SvgW, SvgH]),
          format(S, '  <rect x="0" y="0" width="~w" height="~w" fill="~w"/>~n',
              [SvgW, SvgH, BG]),
          emit_arrow_defs(S),
          TitleX is SvgW // 2,
          format(S, '  <text x="~w" y="32" font-size="20" font-weight="bold" fill="~w" text-anchor="middle">~w</text>~n',
              [TitleX, Ink, Op]),
          format(S, '  <text x="~w" y="52" font-size="11" fill="#5a4a3a" text-anchor="middle">activation-record shape · DRAFT · Iyun + mavchin 2026-05-29</text>~n',
              [TitleX]),

          %% Layout: input on left, intermediates in middle, output on right
          %% Wider canvas (1000px) so title fits and there's room for arrows.

          InputX = 90, InputY = 120, InputW = 180, InputH = 280,
          IntermX = 450, IntermY = 130, IntermW = 200,
          OutputX = 830, OutputY = 120, OutputW = 180, OutputH = 280,

          %% Input region (2D)
          format(S, '  <rect class="m" data-dim="shape(op(~w),operand(input))" x="~w" y="~w" width="~w" height="~w" fill="~w" stroke="~w" stroke-width="2"/>~n',
              [Op, InputX, InputY, InputW, InputH, '#3a6b88', Ink]),
          IXMid is InputX + InputW // 2,
          IYTop is InputY + 20,
          format(S, '  <text x="~w" y="~w" font-size="14" font-weight="bold" fill="#f8f5ee" text-anchor="middle">x</text>~n', [IXMid, IYTop]),
          IYLbl is InputY + InputH + 18,
          format(S, '  <text x="~w" y="~w" font-size="11" fill="~w" text-anchor="middle">[outer × N]</text>~n', [IXMid, IYLbl, Ink]),
          %% Axis labels on the rectangle
          OuterLblX is InputX - 12,
          OuterLblY is InputY + InputH // 2,
          format(S, '  <text x="~w" y="~w" font-size="10" fill="~w" text-anchor="end" transform="rotate(-90 ~w ~w)">outer</text>~n',
              [OuterLblX, OuterLblY, Ink, OuterLblX, OuterLblY]),
          NLblX is InputX + InputW // 2,
          NLblY is InputY - 6,
          format(S, '  <text x="~w" y="~w" font-size="10" fill="~w" text-anchor="middle">N (inner, reduced)</text>~n', [NLblX, NLblY, Ink]),

          %% Output region (2D, same shape as input)
          format(S, '  <rect class="m" data-dim="shape(op(~w),operand(output))" x="~w" y="~w" width="~w" height="~w" fill="~w" stroke="~w" stroke-width="2"/>~n',
              [Op, OutputX, OutputY, OutputW, OutputH, '#3f7c5c', Ink]),
          OXMid is OutputX + OutputW // 2,
          OYTop is OutputY + 20,
          format(S, '  <text x="~w" y="~w" font-size="14" font-weight="bold" fill="#f8f5ee" text-anchor="middle">y</text>~n', [OXMid, OYTop]),
          OYLbl is OutputY + OutputH + 18,
          format(S, '  <text x="~w" y="~w" font-size="11" fill="~w" text-anchor="middle">[outer × N]</text>~n', [OXMid, OYLbl, Ink]),

          %% Intermediate region(s) — one small rect per intermediate, stacked vertically
          findall(I-S2, (operand(Op, I, intermediate, S2)), Intermediates),
          length(Intermediates, NI),
          ( NI =:= 0
          -> true
          ;  emit_intermediates(S, Op, Intermediates, IntermX, IntermY, IntermW, 260)
          ),

          %% Parameter region(s) — small at bottom
          findall(P-PS, (operand(Op, P, parameter, PS)), Params),
          emit_parameters(S, Op, Params, 70, 460),

          %% Reduction arrows: input.right → intermediates.left
          %% Anchor at mid-Y of input rectangle on left, mid-Y of intermediates block on right
          IntermMidY is IntermY + 130,  %% middle of the intermediate region
          InputMidY is InputY + InputH // 2,
          InputRightX is InputX + InputW,
          emit_reductions(S, Op, InputRightX, InputMidY, IntermX, IntermMidY),

          %% Broadcast arrows: intermediates.right → output.left
          IntermRightX is IntermX + IntermW,
          OutputMidY is OutputY + OutputH // 2,
          emit_broadcasts(S, Op, IntermRightX, IntermMidY, OutputX, OutputMidY),

          format(S, '</svg>~n', [])
        ),
        close(S)),
    format(user_error, "Wrote: ~w~n", [Path]).


%% Place each intermediate as a small box stacked vertically
emit_intermediates(_, _, [], _, _, _, _).
emit_intermediates(S, Op, Items, X, Y0, W, TotalH) :-
    length(Items, N),
    BoxH is min(48, TotalH // N - 8),
    Gap = 8,
    emit_intermediates_loop(S, Op, Items, X, Y0, W, BoxH, Gap, 0).

emit_intermediates_loop(_, _, [], _, _, _, _, _, _).
emit_intermediates_loop(S, Op, [Name-Shape|Rest], X, Y0, W, BoxH, Gap, Idx) :-
    Y is Y0 + Idx * (BoxH + Gap),
    region_color(intermediate, Color),
    ink_color(Ink),
    format(S, '  <rect class="m" data-dim="shape(op(~w),operand(~w))" x="~w" y="~w" width="~w" height="~w" fill="~w" stroke="~w" stroke-width="1.5" rx="3"/>~n',
        [Op, Name, X, Y, W, BoxH, Color, Ink]),
    XMid is X + W // 2,
    YMidText is Y + BoxH // 2 + 5,
    format(S, '  <text x="~w" y="~w" font-size="12" font-weight="bold" fill="#f8f5ee" text-anchor="middle">~w</text>~n',
        [XMid, YMidText, Name]),
    shape_label(Shape, ShapeText),
    YShapeText is Y + BoxH + 12,
    format(S, '  <text x="~w" y="~w" font-size="9" fill="~w" text-anchor="middle">~w</text>~n',
        [XMid, YShapeText, Ink, ShapeText]),
    Idx1 is Idx + 1,
    emit_intermediates_loop(S, Op, Rest, X, Y0, W, BoxH, Gap, Idx1).


emit_parameters(_, _, [], _, _).
emit_parameters(S, Op, Items, X, Y) :-
    emit_parameters_loop(S, Op, Items, X, Y, 0).

emit_parameters_loop(_, _, [], _, _, _).
emit_parameters_loop(S, Op, [Name-Shape|Rest], X, Y, Idx) :-
    BoxW = 80, BoxH = 28,
    Gap = 16,
    BX is X + Idx * (BoxW + Gap),
    region_color(parameter, Color),
    ink_color(Ink),
    format(S, '  <rect class="m" data-dim="shape(op(~w),operand(~w))" x="~w" y="~w" width="~w" height="~w" fill="~w" stroke="~w" stroke-width="1.5" rx="2"/>~n',
        [Op, Name, BX, Y, BoxW, BoxH, Color, Ink]),
    XMid is BX + BoxW // 2,
    YText is Y + BoxH // 2 + 5,
    format(S, '  <text x="~w" y="~w" font-size="11" font-weight="bold" fill="~w" text-anchor="middle">~w</text>~n',
        [XMid, YText, Ink, Name]),
    shape_label(Shape, ShapeText),
    YShape is Y + BoxH + 12,
    format(S, '  <text x="~w" y="~w" font-size="9" fill="~w" text-anchor="middle">~w</text>~n',
        [XMid, YShape, Ink, ShapeText]),
    Idx1 is Idx + 1,
    emit_parameters_loop(S, Op, Rest, X, Y, Idx1).


%% Emit the arrowhead marker definitions ONCE at the top of the SVG (caller
%% must call this before any reduction/broadcast emission).
emit_arrow_defs(S) :-
    format(S, '  <defs>~n', []),
    format(S, '    <marker id="arrowhead_red" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><polygon points="0,0 9,3 0,6" fill="#8a3a2a"/></marker>~n', []),
    format(S, '    <marker id="arrowhead_grn" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><polygon points="0,0 9,3 0,6" fill="#5fa07c"/></marker>~n', []),
    format(S, '  </defs>~n', []).

emit_reductions(S, Op, X0, Y0, X1, _Y1) :-
    findall(K, reduction(Op, _, _, _, K), Kinds),
    (   Kinds = []
    ->  true
    ;   list_to_set(Kinds, UniqKinds),
        atomic_list_concat(UniqKinds, ' + ', KindsText),
        ink_color(Ink),
        StartX is X0 + 4,
        EndX is X1 - 8,
        XMid is (StartX + EndX) // 2,
        format(S, '  <line x1="~w" y1="~w" x2="~w" y2="~w" stroke="~w" stroke-width="2.5" marker-end="url(#arrowhead_red)"/>~n',
            [StartX, Y0, EndX, Y0, '#8a3a2a']),
        YLbl is Y0 - 10,
        format(S, '  <text x="~w" y="~w" font-size="11" font-weight="bold" fill="#8a3a2a" text-anchor="middle">reduce: ~w</text>~n',
            [XMid, YLbl, KindsText]),
        YLbl2 is Y0 + 18,
        format(S, '  <text x="~w" y="~w" font-size="10" fill="~w" text-anchor="middle">eps inside rsqrt</text>~n',
            [XMid, YLbl2, Ink])
    ).


emit_broadcasts(S, Op, X0, _Y0, X1, Y1) :-
    findall(B, broadcast(Op, _, _, B), Bs),
    (   Bs = []
    ->  true
    ;   list_to_set(Bs, UniqBs),
        atomic_list_concat(UniqBs, ' + ', BsText),
        StartX is X0 + 4,
        EndX is X1 - 8,
        XMid is (StartX + EndX) // 2,
        format(S, '  <line x1="~w" y1="~w" x2="~w" y2="~w" stroke="~w" stroke-width="2.5" marker-end="url(#arrowhead_grn)"/>~n',
            [StartX, Y1, EndX, Y1, '#5fa07c']),
        YLbl is Y1 - 10,
        format(S, '  <text x="~w" y="~w" font-size="11" font-weight="bold" fill="#5fa07c" text-anchor="middle">broadcast: ~w</text>~n',
            [XMid, YLbl, BsText])
    ).


shape_label(scalar,            'scalar').
shape_label(vec(D),            Text) :- format(atom(Text), '[~w]', [D]).
shape_label(mat(A, B),         Text) :- format(atom(Text), '[~w × ~w]', [A, B]).


%% ─────────────────────────────────────────────────────────────────────────────
%% main: render all four ops
%% ─────────────────────────────────────────────────────────────────────────────

main :-
    Ops = [rms_norm, layer_norm, l2_norm, group_norm],
    forall(member(Op, Ops),
        ( format(atom(Path), '/tmp/output-only/~w_shape.o.svg', [Op]),
          emit_norm_shape_svg(Op, Path)
        )),
    halt(0).
