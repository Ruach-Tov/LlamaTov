%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% cfd_retire_dashboard.pl — SVG dashboard for CFD retire/verify status
%% Table(10010) — generated from cfd_retire_status.pl facts
%%
%% Usage: swipl -g 'main, halt' cfd_retire_dashboard.pl -- output.o.svg

:- module(cfd_retire_dashboard, [main/0]).
:- use_module(library(lists)).
:- use_module(lib/dashboard_common, [freshness_stamp_svg/4]).

%% Load facts — try generated output first, then repo
:- (  catch(use_module('/tmp/bpd-generated/output/cfd_retire_status.o'), _, fail)
   -> true
   ;  catch(use_module('lib/cfd_retire_status'), _, fail)
   -> true
   ;  true
   ).

%% Colors
status_color(lifted_verified(0), '#2a6496').    % blue — verified 0 ULP
status_color(lifted_verified(U), '#4a8a4a') :- U > 0, U =< 10.  % green
status_color(lifted_verified(U), '#b8860b') :- U > 10, U =< 100. % yellow
status_color(lifted_divergent(_, _), '#a04040'). % red
status_color(needs_lift, '#b0a890').             % warm grey
status_color(frontier, '#7a7a7a').               % dark grey

status_label(lifted_verified(0), 'verified · 0 ULP').
status_label(lifted_verified(U), Label) :- U > 0, format(atom(Label), 'verified · ~w ULP', [U]).
status_label(lifted_divergent(U, Cause), Label) :- format(atom(Label), '~w ULP · ~w', [U, Cause]).
status_label(needs_lift, 'needs lift').
status_label(frontier, 'frontier').

text_color('#2a6496', '#ffffff').
text_color('#4a8a4a', '#ffffff').
text_color('#a04040', '#ffffff').
text_color('#7a7a7a', '#ffffff').
text_color(_, '#1a1a1a').

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [OutPath | _]
    ->  true
    ;   OutPath = '/tmp/bpd-generated/output/cfd_retire_status.o.svg'
    ),
    generate_svg(OutPath),
    halt.

generate_svg(OutPath) :-
    findall(Op-Status, cfd_retire_status:retire_status(Op, Status), Pairs),
    length(Pairs, N),
    
    %% Count by category
    include(is_verified, Pairs, Verified),
    include(is_divergent, Pairs, Divergent),
    include(is_needs_lift, Pairs, NeedsLift),
    include(is_frontier, Pairs, Frontier),
    length(Verified, NV), length(Divergent, ND),
    length(NeedsLift, NL), length(Frontier, NF),
    
    %% Layout
    CellW = 500, CellH = 40, LabelW = 220, Pad = 20,
    TitleH = 100,
    SvgW is LabelW + CellW + Pad * 3,
    SvgH is TitleH + N * (CellH + 4) + Pad * 2 + 108,
    MidX is SvgW // 2,
    
    open(OutPath, write, S),
    format(S, '<svg xmlns="http://www.w3.org/2000/svg" width="~w" height="~w" viewBox="0 0 ~w ~w" font-family="Georgia, serif">~n',
        [SvgW, SvgH, SvgW, SvgH]),
    format(S, '  <rect x="0" y="0" width="~w" height="~w" fill="#f8f5ee"/>~n', [SvgW, SvgH]),
    
    %% Title
    format(S, '  <text x="~w" y="36" font-size="20" font-weight="bold" fill="#3a2a1a" text-anchor="middle">CFD Stencil Operations — Retire/Verify Status</text>~n', [MidX]),
    format(S, '  <text x="~w" y="58" font-size="11" fill="#5a4a3a" text-anchor="middle">Table(10010) · ~w ops · ~w verified · ~w divergent · ~w needs lift · ~w frontier</text>~n',
        [MidX, N, NV, ND, NL, NF]),
    format(S, '  <text x="~w" y="78" font-size="11" fill="#5a4a3a" text-anchor="middle">regenerated from bpd/lib/cfd_retire_status.pl</text>~n', [MidX]),
    
    %% Back link
    BackY is SvgH - 50,
    format(S, '  <a href="/static/llvm_match_status.html">~n', []),
    format(S, '    <text x="~w" y="~w" font-size="12" fill="#2a6496" text-decoration="underline">← Table(10001) LLVM IR match</text>~n', [Pad, BackY]),
    format(S, '  </a>~n', []),
    BackY2 is BackY + 18,
    format(S, '  <a href="/static/lift_coverage.html">~n', []),
    format(S, '    <text x="~w" y="~w" font-size="12" fill="#2a6496" text-decoration="underline">← Table(10000) Lifting coverage</text>~n', [Pad, BackY2]),
    format(S, '  </a>~n', []),
    
    %% Rows
    Y0 is TitleH,
    emit_rows(S, Pairs, Y0, Pad, LabelW, CellW, CellH),
    
    %% Legend
    LegendY is TitleH + N * (CellH + 4) + Pad,
    format(S, '  <text x="~w" y="~w" font-size="12" font-weight="bold" fill="#3a2a1a">Legend:</text>~n', [Pad, LegendY]),
    LY2 is LegendY + 18,
    format(S, '  <rect x="~w" y="~w" width="16" height="12" fill="#2a6496" rx="2"/>~n', [Pad, LY2 - 10]),
    format(S, '  <text x="~w" y="~w" font-size="11" fill="#3a2a1a">verified 0 ULP</text>~n', [Pad + 22, LY2]),
    format(S, '  <rect x="~w" y="~w" width="16" height="12" fill="#a04040" rx="2"/>~n', [Pad + 140, LY2 - 10]),
    format(S, '  <text x="~w" y="~w" font-size="11" fill="#3a2a1a">divergent</text>~n', [Pad + 162, LY2]),
    format(S, '  <rect x="~w" y="~w" width="16" height="12" fill="#b0a890" rx="2"/>~n', [Pad + 250, LY2 - 10]),
    format(S, '  <text x="~w" y="~w" font-size="11" fill="#3a2a1a">needs lift</text>~n', [Pad + 272, LY2]),
    format(S, '  <rect x="~w" y="~w" width="16" height="12" fill="#7a7a7a" rx="2"/>~n', [Pad + 360, LY2 - 10]),
    format(S, '  <text x="~w" y="~w" font-size="11" fill="#3a2a1a">frontier</text>~n', [Pad + 382, LY2]),
    
    StampY is LY2 + 28,
    freshness_stamp_svg(S, Pad, StampY, 10),
    format(S, '</svg>~n', []),
    close(S),
    format("Wrote: ~w~n", [OutPath]).

emit_rows(_, [], _, _, _, _, _).
emit_rows(S, [Op-Status | Rest], Y, Pad, LabelW, CellW, CellH) :-
    status_color(Status, Color),
    status_label(Status, Label),
    text_color(Color, TxtColor),
    
    TextY is Y + CellH // 2 + 5,
    
    %% Op name
    format(S, '  <text x="~w" y="~w" font-size="13" fill="#3a2a1a">~w</text>~n', [Pad, TextY, Op]),
    
    %% Colored cell
    CellX is Pad + LabelW,
    format(S, '  <rect x="~w" y="~w" width="~w" height="~w" fill="~w" rx="4" stroke="#3a2a1a" stroke-width="1"/>~n',
        [CellX, Y, CellW, CellH, Color]),
    
    %% Label
    CellMidX is CellX + CellW // 2,
    format(S, '  <text x="~w" y="~w" font-size="13" font-weight="bold" fill="~w" text-anchor="middle">~w</text>~n',
        [CellMidX, TextY, TxtColor, Label]),
    
    Y1 is Y + CellH + 4,
    emit_rows(S, Rest, Y1, Pad, LabelW, CellW, CellH).

%% Helpers for counting
is_verified(_-lifted_verified(_)).
is_divergent(_-lifted_divergent(_, _)).
is_needs_lift(_-needs_lift).
is_frontier(_-frontier).
