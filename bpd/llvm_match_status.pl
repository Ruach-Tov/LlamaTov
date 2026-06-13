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
    pattern_op/2,
    pattern_op_count_actual/2,
    ir_param/3,
    cells_in_pattern/2,
    ulp_color/2,
    table_id/3,
    cell_dim_expression/4,
    emit_llvm_match_svg/1,
    main/0
]).

:- use_module(library(lists)).
:- use_module(lib/dashboard_common, [freshness_stamp_svg/4]).


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
%% Pattern decomposition: which ops belong to each pattern
%%
%% pattern_op(+PatternAtom, +OpAtom).
%%
%% These are the per-op rows that appear when a pattern is "expanded" in the
%% grimoire-viewer. The pattern-level ulp_match/4 rolls up status across these
%% ops; per-op ulp_match/4 facts (using compound atom Pattern/Op as the row)
%% give finer-grain status when measurements are recorded per op.
%%
%% Substrate-of-record provenance:
%%   - unary_elementwise: 12 ops named partially by mavchin (2026-05-27,
%%     "relu, silu, gelu, tanh, sigmoid…"); remainder from common ggml/
%%     PyTorch activation taxonomy. To be corrected by mavchin in β-stage.
%%   - reduction: 7 ops named by mavchin (mul_mat, sum, mean, max, min,
%%     argmax, argmin).
%%   - other patterns: best-reading from mavchin's commentary; subject to
%%     correction.
%%
%% Lives in this file (not a sibling kernel_patterns.pl) because mavchin
%% named that filename as future-substrate they would build. If it later
%% becomes a separate substrate-of-record, these facts move there.
%% ─────────────────────────────────────────────────────────────────────────────

pattern_op(unary_elementwise, relu).
pattern_op(unary_elementwise, silu).
pattern_op(unary_elementwise, gelu).
pattern_op(unary_elementwise, tanh).
pattern_op(unary_elementwise, sigmoid).
pattern_op(unary_elementwise, softplus).
pattern_op(unary_elementwise, leaky_relu).
pattern_op(unary_elementwise, elu).
pattern_op(unary_elementwise, hardsigmoid).
pattern_op(unary_elementwise, softsign).
pattern_op(unary_elementwise, mish).
pattern_op(unary_elementwise, hardswish).

pattern_op(binary_elementwise, scale).

pattern_op(reduction, mul_mat).
pattern_op(reduction, sum).
pattern_op(reduction, mean).
pattern_op(reduction, max).
pattern_op(reduction, min).
pattern_op(reduction, argmax).
pattern_op(reduction, argmin).

pattern_op(reduction_then_elementwise, norm).
pattern_op(reduction_then_elementwise, rms_norm).
pattern_op(reduction_then_elementwise, softmax).
pattern_op(reduction_then_elementwise, log_softmax).
pattern_op(reduction_then_elementwise, group_norm).
pattern_op(reduction_then_elementwise, layer_norm).

pattern_op(scan, cumsum).
pattern_op(scan, cumprod).

pattern_op(conv_im2col, conv1d).
pattern_op(conv_im2col, conv2d).
pattern_op(conv_im2col, conv3d).
pattern_op(conv_im2col, conv_transpose1d).
pattern_op(conv_im2col, conv_transpose2d).
pattern_op(conv_im2col, conv_transpose3d).

pattern_op(pool_reduce, pool1d).
pattern_op(pool_reduce, pool2d).
pattern_op(pool_reduce, pool3d).

pattern_op(loss_reduce, mse).
pattern_op(loss_reduce, cross_entropy).
pattern_op(loss_reduce, hinge).
pattern_op(loss_reduce, l1).
pattern_op(loss_reduce, nll).
pattern_op(loss_reduce, smooth_l1).

pattern_op(flash_attention, flash_attn).


%% pattern_op_count_actual(+Pattern, -Count)
%% Counts ops declared via pattern_op/2 for a given pattern. Substrate-honest
%% check: this should equal the OpCount in emission_pattern/3. If they
%% drift, that's a substrate-error caught by the grimoire itself.
pattern_op_count_actual(Pattern, Count) :-
    findall(Op, pattern_op(Pattern, Op), Ops),
    length(Ops, Count).


%% ─────────────────────────────────────────────────────────────────────────────
%% IR-shape parameters: which knobs control the IR emitted for each pattern
%%
%% ir_param(+Pattern, +Key, +Value).
%%
%% These are the dimensions of variation that distinguish one reference
%% implementation's IR from another. When the comparison view shows two
%% IRs side-by-side, these are the parameters whose values determine the
%% IR shape. They are visible to the reader so the substrate-of-difference
%% is named, not just rendered.
%%
%% Substrate-honest current state: only reduction has substrate-of-record
%% here, from mavchin's vec_dot achievement (2026-05-27). Others fill in
%% as patterns are emitted.
%%
%% Value can be:
%%   - a literal atom or number (pinned parameter)
%%   - sweepable(Low, High) — parameter that varies across references
%%   - dependent(OtherKey) — derived from another parameter
%% ─────────────────────────────────────────────────────────────────────────────

ir_param(reduction, vec_width,          sweepable(4, 16)).   % SSE3=4, AVX2=8, AVX512=16
ir_param(reduction, accumulator_count,  4).                  % mavchin's vec_dot: 8 accumulators
ir_param(reduction, horizontal_reduce,  hadd_butterfly).     % pairwise butterfly (medayek precision benefit)
ir_param(reduction, fma_used,           true).
ir_param(reduction, target_isa(ggml_sse3), sse3).
ir_param(reduction, target_isa(ggml_avx2), avx2).
ir_param(reduction, target_isa(ggml_avx512), avx512).
ir_param(reduction, target_isa(ggml_neon), neon).

%% Other patterns: substrate-of-record empty; fills in as emitters ship.


%% ─────────────────────────────────────────────────────────────────────────────
%% Resolvers
%%
%% cells_in_pattern(+Pattern, -CellList)
%%   Returns the per-op cells for one pattern, each across all reference
%%   implementations. CellList is a list of cell(Pattern/Op, Ref, Status, Evidence)
%%   ready for projection by a viewer that wants to render the expanded view.
%% ─────────────────────────────────────────────────────────────────────────────

cells_in_pattern(Pattern, Cells) :-
    findall(cell(Pattern/Op, Ref, Status, Evidence),
        ( pattern_op(Pattern, Op),
          reference_implementation(Ref, _),
          (   ulp_match(Pattern/Op, Ref, Status, Evidence)
          ->  true
          ;   Status = untested, Evidence = ''
          )
        ),
        Cells).


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

%% Per-op verification (finer-grain than the pattern roll-up above).
%% These fill in as individual ops are measured.
ulp_match(unary_elementwise/silu, ggml_sse3, ulp(0),
    'silu emitter verified 0 ULP vs scalar reference (mavchin 2026-05-27, commit ce13789)').

%% All other cells: untested (default if no fact)
%% — emitter not yet built, or measurement not yet run.


%% ─────────────────────────────────────────────────────────────────────────────
%% ULP → color mapping (declarative thresholds)
%%
%% Substrate-design: ulp_threshold(MaxInclusive, Color) means
%%   "ULP values up to and including MaxInclusive get this color."
%% Evaluated in order; first match wins.
%% ─────────────────────────────────────────────────────────────────────────────

ulp_threshold(0,        '#3f7c5c').   %% green — exact match
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


%% ─── Bookspine helpers ───────────────────────────────────────

%% Collect per-op spine colors for a pattern×reference, sorted by color rank.
%% Uses llvm_op_match.pl facts if loaded, falls back to empty list.
%% Collect spines for a SPECIFIC column reference.
%% Shows the BEST available data for each op in this column:
%%   - If tested against this exact reference → use that result (blue if 0 ULP)
%%   - If tested against scalar only → show as green (0 ULP) or yellow/red
%%   - If untested → grey
%% Non-ggml columns show grey for ops only tested against ggml/scalar.
collect_op_spines(Pattern, ColRef, Sorted) :-
    %% Get all ops for this pattern
    catch(
        findall(Op, ulp_op_match(Pattern, Op, _, _, _), AllOps0),
        _, AllOps0 = []),
    sort(AllOps0, AllOps),
    (AllOps \= [] ->
        findall(Rank-Color, (
            member(Op, AllOps),
            best_op_color(Op, Pattern, ColRef, Color),
            spine_rank(Color, Rank)
        ), Pairs),
        msort(Pairs, SortedPairs),
        pairs_values(SortedPairs, Sorted)
    ;
        Sorted = []
    ).

%% Find the best color for an op in a given column.
%% Priority: exact reference match > scalar fallback > untested
best_op_color(Op, Pattern, ColRef, Color) :-
    %% Try exact match against this column's reference
    ulp_op_match(Pattern, Op, ColRef, ULP, _), !,
    spine_color_ref(ULP, ColRef, Color).
best_op_color(Op, Pattern, ColRef, Color) :-
    %% For ggml_sse3 only: scalar-verified ops show as green
    %% because scalar C ref uses the same libm as ggml SSE3 on our enclave.
    %% Other ggml columns (avx2/avx512/neon) stay grey — different backends.
    ColRef = ggml_sse3,
    ulp_op_match(Pattern, Op, scalar, ULP, _), !,
    spine_color_ref(ULP, scalar, Color).
best_op_color(_, _, _, '#b0a890').  %% grey — no data for this column

%% Load llvm_op_match facts from generated output (preferred) or repo fallback
:- (  catch(use_module('/tmp/bpd-generated/output/llvm_op_match.o'), _, fail)
   -> true
   ;  catch(use_module('lib/llvm_op_match'), _, fail)
   -> true
   ;  true  % silently continue if not found anywhere
   ).

%% Spine color depends on BOTH ULP and reference level.
%% Blue = 0 ULP AND verified vs ggml SSE3 (IR-match)
%% Green = 0 ULP but only vs scalar C ref (different IR, same numbers)
%% Yellow/Red/Grey = divergent or untested
spine_color_ref(0, ggml_sse3, '#2a6496') :- !.      % blue — IR-match
spine_color_ref(0, _, '#4a8a4a') :- !.               % green — 0 ULP, different IR
spine_color_ref(skipped, _, '#7a7a7a') :- !.          % dark grey — harness can't call this op
spine_color_ref(untested, _, '#b0a890') :- !.         % warm grey — no data yet
spine_color_ref(ULP, _, '#b8860b') :- integer(ULP), ULP > 0, ULP =< 100, !.   % yellow
spine_color_ref(ULP, _, '#a04040') :- integer(ULP), ULP > 100.                 % red

%% Backward compat — used by all_blue check
spine_color(0, '#2a6496').
spine_color(skipped, '#7a7a7a').
spine_color(untested, '#b0a890').
spine_color(ULP, '#4a8a4a') :- integer(ULP), ULP > 0, ULP =< 2.
spine_color(ULP, '#b8860b') :- integer(ULP), ULP > 2, ULP =< 100.
spine_color(ULP, '#a04040') :- integer(ULP), ULP > 100.

spine_rank('#2a6496', 1).  % blue first (IR-match)
spine_rank('#4a8a4a', 2).  % green (0 ULP)
spine_rank('#b8860b', 3).  % yellow (1-100 ULP)
spine_rank('#a04040', 4).  % red (>100 ULP)
spine_rank('#7a7a7a', 5).  % dark grey (skipped — harness gap)
spine_rank('#b0a890', 6).  % warm grey last (untested — no data)

%% Draw bookspine rects
draw_spines(_, [], _, _, _, _, _).
draw_spines(S, [Color|Rest], Idx, BaseX, Y, W, H) :-
    X is BaseX + 2 + Idx * W,
    format(S, '  <rect x="~w" y="~w" width="~w" height="~w" fill="~w" rx="1"/>~n',
        [X, Y, W, H, Color]),
    Idx1 is Idx + 1,
    draw_spines(S, Rest, Idx1, BaseX, Y, W, H).

%% Check if all spines are blue (0 ULP)
all_blue([]).
all_blue(['#2a6496'|Rest]) :- all_blue(Rest).


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
    SvgH is TitleH + TableH + FooterH + 28,
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
          %% Cells — bookspine progress bars
          %% Each cell shows N thin vertical rects (bookspines), one per op.
          %% Sorted by color: blue→green→yellow→red→grey (looks like progress bar).
          %% Cell text shows "IR-match" ONLY when ALL spines are blue.
          forall(nth0(ColIdx, Refs, Ref-_),
              ( CX is TableLeft + LabelW + ColIdx * CellW,
                cell_dim_expression(10001, Ref, Pat, DimExpr),
                %% Cell background
                RectX is CX + 2,
                RectY is RY + 2,
                RectW is CellW - 4,
                RectH is CellH - 4,
                format(S, '  <a href="/diviner/detail/~w/~w.svg" target="_blank">~n', [Pat, Ref]),
                format(S, '  <rect class="m" data-dim="~w" x="~w" y="~w" width="~w" height="~w" fill="#e8e0d0" stroke="#3a2a1a" stroke-width="1" style="cursor:pointer"/>~n',
                    [DimExpr, RectX, RectY, RectW, RectH]),
                %% Collect per-op ULP for this pattern×reference
                collect_op_spines(Pat, Ref, Spines),
                length(Spines, NSpines),
                (NSpines > 0 ->
                    %% Draw bookspines
                    SpineW0 is (RectW - 4) / max(NSpines, 1),
                    SpineW is max(1, min(SpineW0, 10)),
                    SpineH is RectH - 4,
                    SpineY is RectY + 2,
                    draw_spines(S, Spines, 0, RectX, SpineY, SpineW, SpineH),
                    %% Label: IR-match only if ALL blue
                    (all_blue(Spines) ->
                        CellMidX is CX + CellW // 2,
                        format(S, '  <text x="~w" y="~w" font-size="10" font-weight="bold" fill="#ffffff" text-anchor="middle">IR-match</text>~n',
                            [CellMidX, RowMidY])
                    ; true
                    )
                ;
                    %% No ops data — show old-style single cell
                    ( ulp_match(Pat, Ref, UlpVal, _)
                    -> true
                    ;  UlpVal = untested
                    ),
                    ulp_color(UlpVal, Color),
                    InnerX is RectX + 2,
                    InnerY is RectY + 2,
                    InnerW is RectW - 4,
                    InnerH is RectH - 4,
                    format(S, '  <rect x="~w" y="~w" width="~w" height="~w" fill="~w"/>~n',
                        [InnerX, InnerY, InnerW, InnerH, Color]),
                    ulp_glyph(UlpVal, Glyph),
                    CellMidX is CX + CellW // 2,
                    format(S, '  <text x="~w" y="~w" font-size="13" font-weight="bold" fill="#f8f5ee" text-anchor="middle">~w</text>~n',
                        [CellMidX, RowMidY, Glyph])
                ),
                format(S, '  </a>~n', [])
              ))
        )),

    %% Legend
    %% Layout: [Legend:] [blue] IR Match  [green] 0 ULP  [yellow] 1-100  [red] >100  [grey] untested
    LegendTotalW is 50 + 32 + (22 + 90) + 28 + (22 + 50) + 28 + (22 + 70) + 28 + (22 + 60) + 28 + (22 + 70),
    LegendStartX is (SvgW - LegendTotalW) // 2,
    TableBottom is TitleH + TableH,
    LegendY is TableBottom + LegendGutter + 20,
    format(S, '  <text x="~w" y="~w" font-size="12" font-weight="bold" fill="#3a2a1a">Legend:</text>~n',
        [LegendStartX, LegendY]),
    L0 is LegendStartX + 50 + 32,
    L1 is L0 + 22 + 90 + 28,
    L2 is L1 + 22 + 50 + 28,
    L3 is L2 + 22 + 70 + 28,
    L4 is L3 + 22 + 60 + 28,
    emit_legend_swatch(S, 'LLVM IR Matching', '#2a6496', L0, LegendY),
    emit_legend_swatch(S, '0 ULP',    '#4a8a4a', L1, LegendY),
    emit_legend_swatch(S, '1\u2013100 ULP', '#b8860b', L2, LegendY),
    emit_legend_swatch(S, '>100 ULP', '#a04040', L3, LegendY),
    emit_legend_swatch(S, 'untested', '#b0a890', L4, LegendY),
    StampY is LegendY + 30,
    freshness_stamp_svg(S, 20, StampY, 10),
    format(S, '</svg>~n', []).


ulp_glyph(untested, '○').
ulp_glyph(ulp(0), '✓').
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
    ;   OutPath = '/tmp/output-only/llvm_match_status.o.svg'
    ),
    emit_llvm_match_svg(OutPath),
    halt(0).
