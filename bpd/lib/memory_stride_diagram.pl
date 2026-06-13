%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% =============================================================================
%% memory_stride_diagram.pl — SVG for dimension 4: memory-stride / coalescing
%% =============================================================================
%%
%% Shows the address pattern that threads read from global memory.
%% Coalesced: 32 consecutive threads read 32 consecutive addresses → 1 cache line.
%% Strided: spread across multiple cache lines → multiple memory transactions.
%%
%% Classifications:
%%   coalesced(elements_per_warp)  — ideal pattern (sgemv x reads, elem_op reads)
%%   strided(stride)                — every Nth element (sgemv A reads in some impls)
%%   gather_scatter                 — irregular (deferred)
%%   ineffable                      — substrate-honest
%% =============================================================================

:- module(memory_stride_diagram,
    [kernel_memory_stride/2,
     emit_memory_stride_svg/2,
     write_memory_stride_diagram/2]).

:- use_module(library(lists)).
:- use_module('kernel_templates_blas.pl').

%% L1 elementwise: thread i reads address i → coalesced 32-wide per warp.
kernel_memory_stride(K, coalesced(32)) :-
    elem_op(K, _, _, _), !.

%% sgemv variants: substantively both coalesce x reads but differ on A.
%% substrate-native: 32 threads per row read A[row][k..k+31] coalesced.
%% cublas-match: 4 warps per block, each reads its row's A coalesced.
%% Both substantively coalesced at the read level.
kernel_memory_stride(sgemv_substrate_native, coalesced(32)) :- !.
kernel_memory_stride(sgemv_cublas_match, coalesced(32)) :- !.

kernel_memory_stride(_, ineffable).

write_memory_stride_diagram(K, Path) :-
    setup_call_cleanup(
        open(Path, write, Stream),
        emit_memory_stride_svg(K, Stream),
        close(Stream)),
    format(user_error, "Wrote diagram: ~w~n", [Path]).

emit_memory_stride_svg(K, S) :-
    kernel_memory_stride(K, Pattern),
    svg_open(S, 1100, 460),
    svg_title(S, K, Pattern),
    svg_body(S, Pattern),
    svg_close(S).

svg_open(S, W, H) :-
    format(S, '<svg xmlns="http://www.w3.org/2000/svg" width="~w" height="~w" viewBox="0 0 ~w ~w" font-family="sans-serif">~n', [W,H,W,H]),
    format(S, '  <rect x="0" y="0" width="~w" height="~w" fill="#fafafa"/>~n', [W,H]).
svg_close(S) :- format(S, '</svg>~n', []).

svg_title(S, K, P) :-
    format(S, '  <text x="20" y="30" font-size="18" font-weight="bold" fill="#222">Memory-stride / coalescing: ~w</text>~n', [K]),
    subtitle(P, Sub),
    format(S, '  <text x="20" y="52" font-size="13" fill="#555">~w</text>~n', [Sub]).

subtitle(coalesced(N), Sub) :-
    format(atom(Sub), 'Coalesced reads: ~w threads access ~w consecutive addresses → 1 cache line, 1 memory transaction.', [N, N]).
subtitle(strided(S), Sub) :-
    format(atom(Sub), 'Strided reads: stride = ~w elements. Multiple cache lines, multiple transactions.', [S]).
subtitle(ineffable, 'Memory-stride pattern not classified for this kernel.').

svg_body(S, coalesced(NThreads)) :-
    %% Show global memory as a horizontal strip with cache-line boundaries marked.
    %% Each cache line holds 32 floats (128 bytes at f32).
    %% Show 32 threads reading 32 consecutive addresses → one cache line, all green.
    XStart = 60, YGlobal = 100, CellW = 22, CellH = 26,
    %% Top: address strip
    format(S, '  <text x="~w" y="~w" font-size="13" fill="#444">global memory addresses</text>~n', [XStart, YGlobal - 8]),
    emit_addr_strip(S, XStart, YGlobal, CellW, CellH, NThreads),
    %% Cache line boundary marker
    YBoundary is YGlobal + CellH + 6,
    LineW is NThreads * CellW,
    format(S, '  <line x1="~w" y1="~w" x2="~w" y2="~w" stroke="#44aa44" stroke-width="2.5"/>~n', [XStart, YBoundary, XStart + LineW, YBoundary]),
    YLabel is YBoundary + 16,
    format(S, '  <text x="~w" y="~w" font-size="12" fill="#2a8" font-weight="bold">└── 1 cache line (~w × 4 bytes = ~w bytes) ──┘</text>~n', [XStart, YLabel, NThreads, NThreads*4]),
    %% Bottom: threads strip
    YThreads is YBoundary + 60,
    format(S, '  <text x="~w" y="~w" font-size="13" fill="#444">warp (~w threads, each reading one address above)</text>~n', [XStart, YThreads - 8, NThreads]),
    emit_thread_strip(S, XStart, YThreads, CellW, CellH, NThreads),
    %% Arrows from threads to addresses (sampling: every 4th)
    forall(member(I, [0, 4, 8, 12, 16, 20, 24, 28, 31]),
           (I < NThreads
           ->  AX is XStart + I * CellW + CellW//2,
               format(S, '  <line x1="~w" y1="~w" x2="~w" y2="~w" stroke="#88a" stroke-width="1" stroke-dasharray="2 2"/>~n', [AX, YThreads, AX, YGlobal + CellH])
           ;   true)),
    %% Footer
    Y2 is YThreads + CellH + 40,
    format(S, '  <text x="60" y="~w" font-size="14" fill="#2a6">✓ Coalesced: 1 cache line fetched, full bandwidth utilized.</text>~n', [Y2]),
    Y3 is Y2 + 20,
    format(S, '  <text x="60" y="~w" font-size="13" fill="#777">Each thread reads address (warp_base + thread_id). Adjacent threads → adjacent addresses.</text>~n', [Y3]).

svg_body(S, strided(Stride)) :-
    %% Spread addresses with gaps; show that they fall in multiple cache lines.
    XStart = 60, YGlobal = 100, CellW = 14, CellH = 26,
    NShown is 8 * Stride,
    format(S, '  <text x="~w" y="~w" font-size="13" fill="#444">global memory addresses (showing reads only — stride = ~w)</text>~n', [XStart, YGlobal-8, Stride]),
    emit_strided_strip(S, XStart, YGlobal, CellW, CellH, NShown, Stride),
    Y2 is YGlobal + CellH + 80,
    format(S, '  <text x="60" y="~w" font-size="14" fill="#a44">⚠ Non-coalesced: multiple cache lines fetched per warp, bandwidth wasted.</text>~n', [Y2]).

svg_body(S, ineffable) :-
    format(S, '  <rect x="100" y="180" width="800" height="160" fill="#ffffff" stroke="#999" stroke-width="1" stroke-dasharray="6 4"/>~n', []),
    format(S, '  <text x="500" y="255" font-size="20" font-weight="bold" fill="#666" text-anchor="middle">This Node Intentionally Left Blank</text>~n', []),
    format(S, '  <text x="500" y="290" font-size="14" fill="#888" text-anchor="middle">(Memory-stride pattern not classified for this kernel.)</text>~n', []).

emit_addr_strip(_, _, _, _, _, 0) :- !.
emit_addr_strip(S, X, Y, W, H, N) :-
    emit_addr_loop(S, X, Y, W, H, 0, N).
emit_addr_loop(_, _, _, _, _, I, N) :- I >= N, !.
emit_addr_loop(S, X, Y, W, H, I, N) :-
    CX is X + I * W,
    format(S, '  <rect x="~w" y="~w" width="~w" height="~w" fill="#c8e6c9" stroke="#4a4" stroke-width="0.5"/>~n', [CX,Y,W,H]),
    TX is CX + W//2, TY is Y + H//2 + 3,
    Addr is I,
    format(S, '  <text x="~w" y="~w" font-size="8" fill="#264" text-anchor="middle">~w</text>~n', [TX,TY,Addr]),
    I1 is I + 1, emit_addr_loop(S, X, Y, W, H, I1, N).

emit_thread_strip(_, _, _, _, _, 0) :- !.
emit_thread_strip(S, X, Y, W, H, N) :-
    emit_thr_loop(S, X, Y, W, H, 0, N).
emit_thr_loop(_, _, _, _, _, I, N) :- I >= N, !.
emit_thr_loop(S, X, Y, W, H, I, N) :-
    CX is X + I * W,
    format(S, '  <rect x="~w" y="~w" width="~w" height="~w" fill="#779ECB" stroke="#345" stroke-width="0.4"/>~n', [CX,Y,W,H]),
    TX is CX + W//2, TY is Y + H//2 + 3,
    format(S, '  <text x="~w" y="~w" font-size="8" fill="#fff" text-anchor="middle">t~w</text>~n', [TX,TY,I]),
    I1 is I + 1, emit_thr_loop(S, X, Y, W, H, I1, N).

emit_strided_strip(_, _, _, _, _, 0, _) :- !.
emit_strided_strip(S, X, Y, W, H, N, Stride) :-
    emit_sstr_loop(S, X, Y, W, H, 0, N, Stride).
emit_sstr_loop(_, _, _, _, _, I, N, _) :- I >= N, !.
emit_sstr_loop(S, X, Y, W, H, I, N, Stride) :-
    CX is X + I * W,
    (I mod Stride =:= 0
    ->  format(S, '  <rect x="~w" y="~w" width="~w" height="~w" fill="#779ECB" stroke="#345" stroke-width="0.4"/>~n', [CX,Y,W,H])
    ;   format(S, '  <rect x="~w" y="~w" width="~w" height="~w" fill="#eeeeee" stroke="#ccc" stroke-width="0.4"/>~n', [CX,Y,W,H])),
    I1 is I + 1, emit_sstr_loop(S, X, Y, W, H, I1, N, Stride).
