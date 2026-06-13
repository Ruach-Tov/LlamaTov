%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% =============================================================================
%% warp_tile_diagram.pl — SVG for dimension 3: warp-tile sub-decomposition
%% =============================================================================
%%
%% Shows the hierarchical decomposition: threadblock → warps → threads.
%% GEMM-class kernels decompose further (warp-tile → thread-tile → mma); not
%% yet in our v1 substrate. L1 elementwise has thread-level only.
%%
%% Classifications:
%%   threadblock_only(N)        — single level, thread-only (L1 elementwise)
%%   threadblock_warp(W, S)     — two levels: W warps × S threads (sgemv)
%%   ineffable                  — substrate-honest about gap
%% =============================================================================

:- module(warp_tile_diagram,
    [kernel_warp_tile/2,
     emit_warp_tile_svg/2,
     write_warp_tile_diagram/2]).


%% clauses grouped by family, not contiguous — declared for warning-free consult.
:- discontiguous svg_body/2.
:- use_module(library(lists)).
:- use_module('kernel_templates_blas.pl').

kernel_warp_tile(KernelName, threadblock_only(256)) :-
    elem_op(KernelName, _, _, _), !.
kernel_warp_tile(sgemv_substrate_native, threadblock_warp(1, 32)) :- !.
kernel_warp_tile(sgemv_cublas_match, threadblock_warp(4, 32)) :- !.
kernel_warp_tile(_, ineffable).

write_warp_tile_diagram(K, Path) :-
    setup_call_cleanup(
        open(Path, write, Stream),
        emit_warp_tile_svg(K, Stream),
        close(Stream)),
    format(user_error, "Wrote diagram: ~w~n", [Path]).

emit_warp_tile_svg(K, Stream) :-
    kernel_warp_tile(K, Tile),
    svg_open(Stream, 1000, 520),
    svg_title(Stream, K, Tile),
    svg_body(Stream, Tile),
    svg_close(Stream).

svg_open(S, W, H) :-
    format(S, '<svg xmlns="http://www.w3.org/2000/svg" width="~w" height="~w" viewBox="0 0 ~w ~w" font-family="sans-serif">~n', [W,H,W,H]),
    format(S, '  <rect x="0" y="0" width="~w" height="~w" fill="#fafafa"/>~n', [W,H]).
svg_close(S) :- format(S, '</svg>~n', []).

svg_title(S, K, T) :-
    format(S, '  <text x="20" y="30" font-size="18" font-weight="bold" fill="#222">Warp-tile decomposition: ~w</text>~n', [K]),
    subtitle(T, Sub),
    format(S, '  <text x="20" y="52" font-size="13" fill="#555">~w</text>~n', [Sub]).

subtitle(threadblock_only(N), Sub) :-
    format(atom(Sub), 'Single level: threadblock of ~w threads. No warp-tile structure exposed.', [N]).
subtitle(threadblock_warp(W, S), Sub) :-
    Total is W * S,
    format(atom(Sub), 'Two levels: 1 threadblock (~w threads) = ~w warps × ~w threads/warp.', [Total, W, S]).
subtitle(ineffable, 'Warp-tile decomposition not classified for this kernel.').

svg_body(S, threadblock_only(_N)) :-
    %% Outer rect = threadblock. Inside: many small thread cells.
    %% Substrate-honest: no intermediate warp grouping shown.
    BX = 60, BY = 100, BW = 880, BH = 80,
    format(S, '  <rect x="~w" y="~w" width="~w" height="~w" fill="#fff5d6" stroke="#cc9" stroke-width="2" rx="6"/>~n', [BX,BY,BW,BH]),
    format(S, '  <text x="~w" y="~w" font-size="13" fill="#996">threadblock (256 threads, no warp-tile sub-structure)</text>~n', [BX+10, BY+18]),
    %% Show 32 mini-cells representing a sampling of threads
    InnerY is BY + 35, CellW = 24, CellH = 30,
    InnerX is BX + 12,
    forall(between(0, 31, I),
           (CX is InnerX + I * (CellW+2),
            format(S, '  <rect x="~w" y="~w" width="~w" height="~w" fill="#FFB347" stroke="#666" stroke-width="0.4"/>~n', [CX,InnerY,CellW,CellH]),
            TX is CX + CellW//2, TY is InnerY + CellH//2 + 3,
            format(S, '  <text x="~w" y="~w" font-size="9" fill="#222" text-anchor="middle">t~w</text>~n', [TX,TY,I]))),
    EllX is InnerX + 32 * (CellW+2) + 6, EllY is InnerY + CellH//2 + 5,
    format(S, '  <text x="~w" y="~w" font-size="18" fill="#999">…</text>~n', [EllX, EllY]),
    %% Footer note
    Y2 is BY + BH + 50,
    format(S, '  <text x="60" y="~w" font-size="14" fill="#444">Each thread independently computes one output element.</text>~n', [Y2]),
    Y3 is Y2 + 22,
    format(S, '  <text x="60" y="~w" font-size="13" fill="#777">No cross-thread cooperation; no warp-level structure needed.</text>~n', [Y3]).

svg_body(S, threadblock_warp(W, ThreadsPerWarp)) :-
    %% Outer rect = threadblock. Inside: W warp rects, each containing
    %% ThreadsPerWarp thread cells.
    BX = 60, BY = 100, BW = 880,
    WarpH = 60, WarpPad = 14,
    BH is 30 + W * (WarpH + WarpPad),
    format(S, '  <rect x="~w" y="~w" width="~w" height="~w" fill="#fff5d6" stroke="#cc9" stroke-width="2" rx="6"/>~n', [BX,BY,BW,BH]),
    Total is W * ThreadsPerWarp,
    format(S, '  <text x="~w" y="~w" font-size="13" fill="#996">threadblock (~w threads)</text>~n', [BX+10, BY+18, Total]),
    emit_warps(S, BX+14, BY+30, BW-28, WarpH, WarpPad, ThreadsPerWarp, 0, W),
    %% Footer
    Y2 is BY + BH + 30,
    format(S, '  <text x="60" y="~w" font-size="14" fill="#444">Each warp computes one output row independently; warp-shuffle reduces partial sums within the warp.</text>~n', [Y2]).

emit_warps(_, _, _, _, _, _, _, I, N) :- I >= N, !.
emit_warps(S, X, Y, W, H, Pad, TPW, I, N) :-
    WY is Y + I * (H + Pad),
    warp_color(I, Color),
    format(S, '  <rect x="~w" y="~w" width="~w" height="~w" fill="~w" stroke="#789" stroke-width="1.2" rx="4"/>~n', [X, WY, W, H, Color]),
    LY is WY + 16,
    format(S, '  <text x="~w" y="~w" font-size="12" fill="#234" font-weight="bold">warp ~w  →  y[bid*~w + ~w]</text>~n', [X+8, LY, I, N, I]),
    %% Inner thread cells
    InnerY is WY + 24, CellW = 22, CellH = 28,
    InnerX is X + 100,
    forall(between(0, 31, T),
           (T < TPW
           ->  CX is InnerX + T * (CellW+1),
               format(S, '  <rect x="~w" y="~w" width="~w" height="~w" fill="#ffffff" stroke="#789" stroke-width="0.4"/>~n', [CX,InnerY,CellW,CellH]),
               TX is CX + CellW//2, TY is InnerY + CellH//2 + 3,
               format(S, '  <text x="~w" y="~w" font-size="8" fill="#345" text-anchor="middle">t~w</text>~n', [TX,TY,T])
           ;   true)),
    I1 is I + 1,
    emit_warps(S, X, Y, W, H, Pad, TPW, I1, N).

svg_body(S, ineffable) :-
    format(S, '  <rect x="100" y="180" width="800" height="180" fill="#ffffff" stroke="#999" stroke-width="1" stroke-dasharray="6 4"/>~n', []),
    format(S, '  <text x="500" y="260" font-size="20" font-weight="bold" fill="#666" text-anchor="middle">This Node Intentionally Left Blank</text>~n', []),
    format(S, '  <text x="500" y="295" font-size="14" fill="#888" text-anchor="middle">(Warp-tile decomposition not classified for this kernel.)</text>~n', []).

warp_color(0, '#FFE5B4'). warp_color(1, '#C1E1C1'). warp_color(2, '#AEC6CF'). warp_color(3, '#E0BBE4').
warp_color(N, C) :- N >= 4, M is N mod 4, warp_color(M, C).
