%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% llvm_match_status.pl — Table(10001): LLVM IR emitter ULP match status
%%
%% Per Heath's substrate-direction (via mavchin 2026-05-27): a cross-tab
%% showing how close our Prolog-generated LLVM IR gets to each reference
%% implementation, per emission pattern.
%%
%% Substrate-architecture mirrors lift_coverage.pl (Table 10000):
%%   - Prolog facts declare the substrate-of-record
%%   - SVG projection derives from facts via swipl -g main
%%   - Each cell carries watermarkup data-dim="cell(table(10001),column,row)"
%%   - Cells updated as new patterns × references verify

:- module(llvm_match_status, [
    ulp_match/4,
    emission_pattern/3,
    reference_implementation/2,
    ulp_color/2,
    table_id/3,
    cell_dim_expression/4,
    emit_llvm_match_svg/1,
    main/0
]).

:- use_module(library(lists)).


%% ─────────────────────────────────────────────────────────────────────────────
%% Document registry
%% ─────────────────────────────────────────────────────────────────────────────

table_id(10001, 'LLVM_IR_Match_Status',
         cross_tab_2d(emission_pattern, reference_implementation)).


%% ─────────────────────────────────────────────────────────────────────────────
%% Rows: 9 IR emission patterns (mavchin's kernel_patterns.pl decomposition)
%% emission_pattern(+PatternAtom, +DisplayName, +OpCount).
%% ─────────────────────────────────────────────────────────────────────────────

emission_pattern(unary_elementwise,         'unary elementwise',          12).
emission_pattern(binary_elementwise,        'binary elementwise',          1).
emission_pattern(reduction,                 'reduction',                   7).
emission_pattern(reduction_then_elementwise,'reduction + elementwise',     6).
emission_pattern(scan,                      'scan',                        2).
emission_pattern(conv_im2col,               'conv via im2col',             6).
emission_pattern(pool_reduce,               'pool / reduce',               3).
emission_pattern(loss_reduce,               'loss / reduce',               6).
emission_pattern(flash_attention,           'flash attention',             1).


%% ─────────────────────────────────────────────────────────────────────────────
%% Columns: reference implementations to match against
%% reference_implementation(+RefAtom, +DisplayName).
%% ─────────────────────────────────────────────────────────────────────────────

reference_implementation(ggml_sse3,    'ggml SSE3').
reference_implementation(ggml_avx2,    'ggml AVX2').
reference_implementation(ggml_avx512,  'ggml AVX512').
reference_implementation(ggml_neon,    'ggml NEON').
reference_implementation(pytorch_mkl,  'PyTorch MKL').
reference_implementation(pytorch_nnpack,'PyTorch NNPACK').


%% ─────────────────────────────────────────────────────────────────────────────
%% Cells: ulp_match(+Pattern, +Reference, +UlpValue, +Evidence).
%%   UlpValue: ulp(N) — measured at N ULP
%%             untested — not yet measured
%% ─────────────────────────────────────────────────────────────────────────────

%% reduction × ggml_sse3 → 0 ULP, verified 2026-05-27 (mavchin)
ulp_match(reduction, ggml_sse3, ulp(0),
    'vec_dot bit-identical vs ggml SSE3 — <4 x float>, 8 accumulators, hadd horizontal reduction (mavchin 2026-05-27)').

ulp_match(unary_elementwise, ggml_sse3, ulp(0),
    'silu verified 0 ULP vs scalar ref. 10 ops emitting: relu, silu, sigmoid, tanh, gelu, softplus, leaky_relu, elu, hardsigmoid, softsign (mavchin 2026-05-27, commit ce13789)').

%% All other cells: untested (default if no fact)
%% — emitter not yet built, or measurement not yet run.


%% ─────────────────────────────────────────────────────────────────────────────
%% ULP → color mapping (declarative thresholds)
%%
%% Substrate-design: ulp_threshold(MaxInclusive, Color) means
%%   "ULP values up to and including MaxInclusive get this color."
%% Evaluated in order; first match wins.
%% ─────────────────────────────────────────────────────────────────────────────

ulp_threshold(0,        '#2a5caa').   %% blue — IR-match (0 ULP)
ulp_threshold(2,        '#5fa07c').   %% light green — within rounding noise
ulp_threshold(10,       '#daa520').   %% amber — small drift
ulp_threshold(infinity, '#8a3a2a').   %% red — significant drift

untested_color('#6a6a6a').            %% grey — not yet measured


%% ulp_color(+UlpValue, -Color).
ulp_color(untested, Color) :- !, untested_color(Color).
ulp_color(ulp(N), Color) :-
    ulp_color_at(N, Color).

ulp_color_at(N, Color) :-
    ulp_threshold(Max, Color),
    (   Max == infinity
    ->  true
    ;   N =< Max
    ),
    !.


%% ─────────────────────────────────────────────────────────────────────────────
%% Watermarkup: canonical cell coordinate expression
%% ─────────────────────────────────────────────────────────────────────────────

cell_dim_expression(Table, Column, Row, DimExpr) :-
    format(atom(DimExpr),
        'cell(table(~w),column(~w),row(~w))',
        [Table, Column, Row]).


%% ─────────────────────────────────────────────────────────────────────────────
%% SVG projection
%% ─────────────────────────────────────────────────────────────────────────────

emit_llvm_match_svg(Path) :-
    findall(P-N-Ops, emission_pattern(P, N, Ops), Patterns),
    findall(R-N, reference_implementation(R, N), Refs),
    length(Patterns, NRows),
    length(Refs, NCols),

    CellW = 120, CellH = 38, HeaderH = 60, LabelW = 240,
    TableW is LabelW + NCols * CellW,
    TableH is HeaderH + NRows * CellH,
    TitleH = 80,
    LegendGutter = 32, LegendBlockH = 28, LegendBottomPad = 24,
    FooterH is LegendGutter + LegendBlockH + LegendBottomPad,
    SvgW0 is TableW + 60,
    SvgW is max(SvgW0, 1280),
    SvgH is TitleH + TableH + FooterH,
    TableLeft is (SvgW - TableW) // 2,
    TitleX is SvgW // 2,

    setup_call_cleanup(
        open(Path, write, S),
        emit_svg_body(S, Patterns, Refs, NRows, NCols, CellW, CellH,
                      HeaderH, LabelW, TitleH, LegendGutter, LegendBlockH,
                      SvgW, SvgH, TableLeft, TitleX, TableW, TableH),
        close(S)),
    format(user_error, "Wrote: ~w~n", [Path]).

emit_svg_body(S, Patterns, Refs, NRows, NCols, CellW, CellH,
              HeaderH, LabelW, TitleH, LegendGutter, _LegendBlockH,
              SvgW, SvgH, TableLeft, TitleX, _TableW, TableH) :-
    format(S,
        '<svg xmlns="http://www.w3.org/2000/svg" width="~w" height="~w" viewBox="0 0 ~w ~w" font-family="Georgia, serif">~n',
        [SvgW, SvgH, SvgW, SvgH]),
    format(S, '  <rect x="0" y="0" width="~w" height="~w" fill="#f8f5ee"/>~n',
        [SvgW, SvgH]),
    format(S, '  <text x="~w" y="36" font-size="20" font-weight="bold" fill="#3a2a1a" text-anchor="middle">LLVM IR emitter — ULP match status</text>~n',
        [TitleX]),
    format(S, '  <text x="~w" y="60" font-size="11" fill="#5a4a3a" text-anchor="middle">Table(10001) · ~w patterns × ~w reference implementations · regenerated from bpd/llvm_match_status.pl</text>~n',
        [TitleX, NRows, NCols]),

    %% Column headers
    forall(nth0(Idx, Refs, _-CName),
        ( CX is TableLeft + LabelW + Idx * CellW + CellW // 2,
          CYTop is TitleH + HeaderH - 16,
          format(S, '  <text x="~w" y="~w" font-size="12" font-weight="bold" fill="#3a2a1a" text-anchor="middle">~w</text>~n',
              [CX, CYTop, CName])
        )),

    %% Rows
    forall(nth0(RowIdx, Patterns, Pat-PName-Ops),
        ( RY is TitleH + HeaderH + RowIdx * CellH,
          RowMidY is RY + CellH // 2 + 4,
          LabelX is TableLeft + 10,
          format(S, '  <text x="~w" y="~w" font-size="13" fill="#3a2a1a">~w <tspan font-size="11" fill="#7a6a4a">(~w ops)</tspan></text>~n',
              [LabelX, RowMidY, PName, Ops]),
          %% Cells
          forall(nth0(ColIdx, Refs, Ref-_),
              ( CX is TableLeft + LabelW + ColIdx * CellW,
                ( ulp_match(Pat, Ref, UlpVal, _)
                -> true
                ;  UlpVal = untested
                ),
                ulp_color(UlpVal, Color),
                CellMidX is CX + CellW // 2,
                %% Evaluate numeric attributes BEFORE format/3 — strict SVG
                %% parsers (Firefox) reject 'x="255+2"' unevaluated arithmetic.
                RectX is CX + 2,
                RectY is RY + 2,
                RectW is CellW - 4,
                RectH is CellH - 4,
                cell_dim_expression(10001, Ref, Pat, DimExpr),
                format(S, '  <rect class="m" data-dim="~w" x="~w" y="~w" width="~w" height="~w" fill="~w" stroke="#3a2a1a" stroke-width="1"/>~n',
                    [DimExpr, RectX, RectY, RectW, RectH, Color]),
                ulp_glyph(UlpVal, Glyph),
                (UlpVal = ulp(0) -> GlyphSize = 10 ; GlyphSize = 14),
                format(S, '  <text x="~w" y="~w" font-size="~w" font-weight="bold" fill="#f8f5ee" text-anchor="middle">~w</text>~n',
                    [CellMidX, RowMidY, GlyphSize, Glyph])
              ))
        )),

    %% Legend
    %% Layout: [Legend:] [swatch] 0 ULP  [swatch] 1-2  [swatch] 3-10  [swatch] >10  [swatch] untested
    LegendTotalW is 50 + 32 + (22 + 100) + 28 + (22 + 50) + 28 + (22 + 60) + 28 + (22 + 50) + 28 + (22 + 80),
    LegendStartX is (SvgW - LegendTotalW) // 2,
    TableBottom is TitleH + TableH,
    LegendY is TableBottom + LegendGutter + 20,
    format(S, '  <text x="~w" y="~w" font-size="12" font-weight="bold" fill="#3a2a1a">Legend:</text>~n',
        [LegendStartX, LegendY]),
    L1 is LegendStartX + 50 + 32,
    L2 is L1 + 22 + 100 + 28,
    L3 is L2 + 22 + 50 + 28,
    L4 is L3 + 22 + 60 + 28,
    L5 is L4 + 22 + 50 + 28,
    emit_legend_swatch(S, 'IR-match (0 ULP)', '#2a5caa', L1, LegendY),
    emit_legend_swatch(S, '1–2 ULP',  '#5fa07c', L2, LegendY),
    emit_legend_swatch(S, '3–10 ULP', '#daa520', L3, LegendY),
    emit_legend_swatch(S, '>10 ULP',  '#8a3a2a', L4, LegendY),
    emit_legend_swatch(S, 'untested', '#6a6a6a', L5, LegendY),
    format(S, '</svg>~n', []).


ulp_glyph(untested, '○').
ulp_glyph(ulp(0), 'IR-match').
ulp_glyph(ulp(N), Glyph) :-
    N > 0,
    format(atom(Glyph), '~w', [N]).


emit_legend_swatch(S, Label, Color, X, Y) :-
    Y0 is Y - 12,
    format(S, '  <rect x="~w" y="~w" width="16" height="16" fill="~w" stroke="#3a2a1a"/>~n',
        [X, Y0, Color]),
    LabelX is X + 22,
    format(S, '  <text x="~w" y="~w" font-size="11" fill="#3a2a1a">~w</text>~n',
        [LabelX, Y, Label]).


%% ─────────────────────────────────────────────────────────────────────────────
%% main
%% ─────────────────────────────────────────────────────────────────────────────

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [OutPath|_] -> true
    ;   OutPath = '/tmp/llvm_match_status.o.svg'
    ),
    emit_llvm_match_svg(OutPath),
    halt(0).
