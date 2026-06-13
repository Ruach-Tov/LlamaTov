%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% divergence_heatmap.pl — Table(10011): Mistral bit-identity divergence heatmap.
%% (Iyun, 2026-05-29, plan f3dfd600.) Follows llvm_match_status.pl (mavchin) pattern EXACTLY:
%%   - facts module lib/divergence_map.pl (div_status/2, status_color/2, mistral_coordinate/2)
%%   - SVG projection derives from facts via swipl -g main
%%   - each cell carries watermarkup data-dim="cell(table(10011),column,row)"
%%   - format/2 direct SVG emission, numeric attrs evaluated BEFORE format (strict SVG parsers)
%%   - output is GENERATED: divergence_heatmap.o.svg (the .o = object, regenerable, not git)
%%
%% GRID: rows = layers (0..N-1) + global rows; cols = per-layer coordinates. Mistral has
%% REPEATED layer structure, so the grid shows PATTERNS (a red column = op diverges every layer).
%% Run: swipl -q -g "use_module(lib/divergence_map),consult(lib/divergence_heatmap),main" -- OUT.o.svg

:- module(divergence_heatmap, [ emit_divergence_svg/1, main/0, layer_coords/1, global_coords/1 ]).
:- use_module(library(lists)).
:- use_module(dashboard_common, [ freshness_stamp_svg/4 ]).   % shared freshness stamp (DRY, all 4 dashboards)

%% the per-layer coordinate columns (the repeated structure)
layer_coords([attn_norm, qkv, rope, score, o_proj, residual1, ffn_norm, swiglu, residual2]).
%% global (non-per-layer) coordinates — shown as their own rows
global_coords([embed, output_norm, logits, dequant]).

%% map a (layer-col) to the canonical coordinate path used in divergence_map
col_path(attn_norm, [mistral, layer(l), attn, norm]).
col_path(qkv,       [mistral, layer(l), attn, qkv]).
col_path(rope,      [mistral, layer(l), attn, rope]).
col_path(score,     [mistral, layer(l), attn, score]).
col_path(o_proj,    [mistral, layer(l), attn, o_proj]).
col_path(residual1, [mistral, layer(l), residual(1)]).
col_path(ffn_norm,  [mistral, layer(l), ffn, norm]).
col_path(swiglu,    [mistral, layer(l), ffn, swiglu]).
col_path(residual2, [mistral, layer(l), residual(2)]).
glob_path(embed,       [mistral, embed, token]).
glob_path(output_norm, [mistral, output, norm]).
glob_path(logits,      [mistral, output, logits]).
glob_path(dequant,     [mistral, weights, dequant]).

%% hex for the heatmap palette (Heath spec, via divergence_map:status_color/2 symbolic)
hex(blue,      '#2a5db0').
hex(green,     '#2e8b3d').
hex(yellow,    '#d6b800').
hex(red,       '#b02a2a').
hex(tan,       '#cdbb96').
hex(dark_grey, '#5a5a5a').
hex(violet,    '#7a4fa0').
hex(teal,      '#2a9d8f').
hex(light_blue, '#5b8dd9').
hex(steel_blue, '#4a7a9d').

%% status of a coordinate -> color hex (re-anchored/intrinsic via divergence_map)
coord_hex(Path, Hex) :-
    ( divergence_map:intrinsic_status(Path, S) -> true ; S = unverified ),
    ( divergence_map:status_color(S, Sym) -> true ; Sym = tan ),
    ( hex(Sym, Hex) -> true ; Hex = '#cdbb96' ).

%% NOTE: per-layer coords use path with layer(l) as the canonical "any-layer" form; until
%% per-layer measurements exist, all layers share the coordinate's status. When Mavhir reports
%% per-layer ULP, assert div_status([mistral,layer(N),...],S) and this lights individual cells.

emit_divergence_svg(Path) :-
    layer_coords(Cols), length(Cols, NCols),
    NLayers = 8,                      % display first 8 layers (Mistral=32; grid shows the pattern)
    global_coords(Globals), length(Globals, NGlob),
    CellW = 96, CellH = 34, HeaderH = 64, LabelW = 90, TitleH = 80,
    NRows is NLayers + NGlob + 1,     % layers + a separator + globals
    TableW is LabelW + NCols * CellW,
    TableH is HeaderH + NRows * CellH,
    FooterH = 112,   % room for legend + the freshness stamp line
    SvgW0 is TableW + 60, SvgW is max(SvgW0, 1100),
    SvgH is TitleH + TableH + FooterH,
    TableLeft is (SvgW - TableW) // 2, TitleX is SvgW // 2,
    setup_call_cleanup(open(Path, write, S),
        emit_body(S, Cols, NCols, NLayers, Globals, CellW, CellH, HeaderH, LabelW,
                  TitleH, SvgW, SvgH, TableLeft, TitleX, TableH),
        close(S)),
    format(user_error, "Wrote: ~w~n", [Path]).

emit_body(S, Cols, NCols, NLayers, Globals, CellW, CellH, HeaderH, LabelW,
          TitleH, SvgW, SvgH, TableLeft, TitleX, TableH) :-
    format(S, '<svg xmlns="http://www.w3.org/2000/svg" width="~w" height="~w" viewBox="0 0 ~w ~w" font-family="Georgia, serif">~n', [SvgW,SvgH,SvgW,SvgH]),
    format(S, '  <rect x="0" y="0" width="~w" height="~w" fill="#f8f5ee"/>~n', [SvgW,SvgH]),
    format(S, '  <text x="~w" y="36" font-size="20" font-weight="bold" fill="#3a2a1a" text-anchor="middle">Mistral bit-identity divergence (vs Ollama)</text>~n', [TitleX]),
    format(S, '  <text x="~w" y="60" font-size="11" fill="#5a4a3a" text-anchor="middle">Table(10011) \u00b7 rows=layers+globals \u00d7 cols=coordinates \u00b7 spec = LIFTED llama.cpp facts (Ollama runs these) \u00b7 green = bit-identical by faithful lift \u00b7 HF/pytorch = cross-checks \u00b7 then improve: faster/better \u00b7 from divergence_map.pl</text>~n', [TitleX]),
    %% column headers
    forall(nth0(I, Cols, CName),
        ( CX is TableLeft + LabelW + I*CellW + CellW//2, CY is TitleH + HeaderH - 18,
          format(S, '  <text x="~w" y="~w" font-size="10" font-weight="bold" fill="#3a2a1a" text-anchor="middle" transform="rotate(-30 ~w ~w)">~w</text>~n', [CX,CY,CX,CY,CName]) )),
    %% layer rows
    forall(between(0, NLayers, _), true),
    NL1 is NLayers - 1,
    forall(between(0, NL1, L),
        ( RY is TitleH + HeaderH + L*CellH, MidY is RY + CellH//2 + 4, LabelX is TableLeft + 8,
          format(S, '  <text x="~w" y="~w" font-size="11" fill="#3a2a1a">layer ~w</text>~n', [LabelX,MidY,L]),
          forall(nth0(C, Cols, CName),
              ( col_path(CName, P0), set_layer(P0, L, P), coord_hex(P, Hex),
                CX is TableLeft + LabelW + C*CellW, RX is CX+2, RYY is RY+2, RW is CellW-4, RH is CellH-4,
                format(S, '  <rect class="d" data-dim="cell(table(10011),column(~w),row(layer~w))" x="~w" y="~w" width="~w" height="~w" fill="~w" stroke="#3a2a1a" stroke-width="1"/>~n',
                    [CName,L,RX,RYY,RW,RH,Hex]) )) )),
    %% global rows (after a gap)
    GlobStartRow is NLayers + 1,
    forall(nth0(GI, Globals, GName),
        ( RowIdx is GlobStartRow + GI, RY is TitleH + HeaderH + RowIdx*CellH, MidY is RY + CellH//2 + 4, LabelX is TableLeft + 8,
          glob_path(GName, P), coord_hex(P, Hex),
          format(S, '  <text x="~w" y="~w" font-size="11" font-weight="bold" fill="#3a2a1a">~w</text>~n', [LabelX,MidY,GName]),
          CX is TableLeft + LabelW, RX is CX+2, RYY is RY+2, RW is (NCols*CellW)-4, RH is CellH-4,
          format(S, '  <rect class="d" data-dim="cell(table(10011),column(~w),row(global))" x="~w" y="~w" width="~w" height="~w" fill="~w" stroke="#3a2a1a" stroke-width="1"/>~n',
              [GName,RX,RYY,RW,RH,Hex]) )),
    %% legend
    LegY is TitleH + TableH + 36, LegX is TableLeft,
    legend(S, LegX, LegY),
    %% FRESHNESS STAMP — makes the dashboard PROVABLY LIVE on its own face: which commit it
    %% reflects + when generated. A viewer reads liveness off the artifact; no need to TEST it.
    StampY is LegY + 26,
    freshness_stamp_svg(S, LegX, StampY, 10),   % shared (dashboard_common) — DRY across all 4 dashboards
    format(S, '</svg>~n', []).

set_layer([mistral, layer(l) | Rest], _L, [mistral, layer(N) | Rest]) :- !, N = l.  % keep symbolic until per-layer facts exist
set_layer(P, _, P).

legend(S, X, Y) :-
    Items = ['#2a5db0'-'IR match', '#5b8dd9'-'reproduced (0-ULP + tick-identical)', '#2e8b3d'-'0 ULP', '#d6b800'-'small ULP', '#b02a2a'-'large ULP', '#cdbb96'-'untested', '#5a5a5a'-'blocked'],
    format(S, '  <text x="~w" y="~w" font-size="11" font-weight="bold" fill="#3a2a1a">Legend:</text>~n', [X,Y]),
    foldl(emit_legend_item(S, Y), Items, X+70, _).
emit_legend_item(S, Y, Hex-Label, X0, X1) :-
    ( integer(X0) -> X = X0 ; X is X0 ),
    SX is X, SY is Y - 11,
    format(S, '  <rect x="~w" y="~w" width="16" height="14" fill="~w" stroke="#3a2a1a"/>~n', [SX,SY,Hex]),
    TX is X + 20, format(S, '  <text x="~w" y="~w" font-size="11" fill="#3a2a1a">~w</text>~n', [TX,Y,Label]),
    atom_length(Label, LL), X1 is X + 20 + LL*7 + 24.

main :-
    ( current_prolog_flag(argv, [Out|_]) -> true ; Out = 'divergence_heatmap.o.svg' ),
    emit_divergence_svg(Out), halt.
