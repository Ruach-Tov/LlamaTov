%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% =============================================================================
%% shared_memory_diagram.pl — SVG diagrams of shared-memory loading patterns
%% =============================================================================
%%
%% Dimension (2) of the kernel-visualization atlas: shared-memory loading
%% pattern. Per Heath's 2026-05-25 ~08:30 UTC direction:
%%
%%   "be artistic the same way you were with thread mapping, and draw shared
%%    memory diagram as you envision it in your mind's eye."
%%
%% Substantively the shared-memory loading pattern shows:
%%   - Which threads cooperate to load data from global memory into shared
%%   - The shape of the shared-memory tile (a buffer reused by many threads)
%%   - Whether the kernel also uses shared memory for reduction (separate buffer)
%%   - Or whether reduction uses warp-shuffle (registers, no shared mem)
%%
%% Substrate-honest about classification:
%%
%%   no_shared_memory
%%       Kernel doesn't use shared memory at all. Most L1 elementwise
%%       kernels (k_vadd, k_saxpy, etc.). Substrate-honestly distinct from
%%       `ineffable`: this isn't ineffable, it's expressly empty.
%%
%%   cooperative_load(Source, BufferSize, BlockSize)
%%       Threads in block cooperatively load BufferSize elements from
%%       Source into shared mem. Stride pattern: each thread loads
%%       indices tid, tid+blockDim, tid+2*blockDim, ...
%%
%%   reduction_buffer(Size, Pattern)
%%       Shared memory used as reduction buffer. Pattern is the reduction
%%       tree shape: strided_pairs(8,4,2,1), butterfly, etc.
%%
%%   composed(LoadPattern, ReductionPattern)
%%       Both: cooperative load AND reduction buffer. The cuBLAS sgemv
%%       pattern uses cooperative load of x into shared, then warp-shuffle
%%       reduction (so ReductionPattern is warp_shuffle in this case).
%%
%%   warp_shuffle_only(WarpSize)
%%       Reduction via __shfl_xor_sync, no shared memory. Register-level
%%       cross-thread communication within a warp. The sgemv_substrate_native
%%       case (one warp per row, no cooperative load).
%%
%%   ineffable
%%       Mapping concept doesn't apply or is too complex for v1.
%%
%% =============================================================================

:- module(shared_memory_diagram,
    [kernel_shared_memory_pattern/2,
     emit_shared_memory_diagram_svg/2,
     write_shared_memory_diagram/2]).

:- use_module(library(lists)).
:- use_module('kernel_templates_blas.pl').


%% -----------------------------------------------------------------------------
%% kernel_shared_memory_pattern(+KernelName, -Pattern)
%% -----------------------------------------------------------------------------

%% L1 elementwise: explicitly no shared memory.
kernel_shared_memory_pattern(KernelName, no_shared_memory) :-
    elem_op(KernelName, _Params, _LHS, _RHS),
    !.

%% sgemv_substrate_native: warp-shuffle reduction, no shared mem proper.
kernel_shared_memory_pattern(sgemv_substrate_native,
    warp_shuffle_only(32)) :- !.

%% sgemv_cublas_match: cooperative load of x into shared sx, then warp-shuffle
%% reduction per row. Block size 128 = 32 threads × 4 rows.
kernel_shared_memory_pattern(sgemv_cublas_match,
    composed(
        cooperative_load(x, k_elements, 128),
        warp_shuffle(32, per_row(4)))) :- !.

%% Default: substrate-honest about the gap.
kernel_shared_memory_pattern(_, ineffable).


%% -----------------------------------------------------------------------------
%% Convenience: write to a file
%% -----------------------------------------------------------------------------

write_shared_memory_diagram(KernelName, OutPath) :-
    setup_call_cleanup(
        open(OutPath, write, Stream),
        emit_shared_memory_diagram_svg(KernelName, Stream),
        close(Stream)),
    format(user_error, "Wrote diagram: ~w~n", [OutPath]).


%% -----------------------------------------------------------------------------
%% SVG generation
%% -----------------------------------------------------------------------------

emit_shared_memory_diagram_svg(KernelName, Stream) :-
    kernel_shared_memory_pattern(KernelName, Pattern),
    svg_open(Stream, 1000, 580),
    svg_title(Stream, KernelName, Pattern),
    svg_body(Stream, KernelName, Pattern),
    svg_close(Stream).


%% -- SVG primitives -----------------------------------------------------------

svg_open(Stream, W, H) :-
    format(Stream,
        '<svg xmlns="http://www.w3.org/2000/svg" width="~w" height="~w" viewBox="0 0 ~w ~w" font-family="sans-serif">~n',
        [W, H, W, H]),
    format(Stream,
        '  <rect x="0" y="0" width="~w" height="~w" fill="#fafafa"/>~n',
        [W, H]).

svg_close(Stream) :-
    format(Stream, '</svg>~n', []).

svg_title(Stream, KernelName, Pattern) :-
    format(Stream,
        '  <text x="20" y="30" font-size="18" font-weight="bold" fill="#222">Shared-memory loading pattern: ~w</text>~n',
        [KernelName]),
    pattern_subtitle(Pattern, Subtitle),
    format(Stream,
        '  <text x="20" y="52" font-size="13" fill="#555">~w</text>~n',
        [Subtitle]).

pattern_subtitle(no_shared_memory,
    'No shared memory used. Threads work directly from global memory.').
pattern_subtitle(warp_shuffle_only(N), Sub) :-
    format(atom(Sub),
        'Warp-shuffle reduction across ~w threads. No shared memory used (registers only).',
        [N]).
pattern_subtitle(composed(cooperative_load(Src, _, BlockSize),
                          warp_shuffle(WarpSize, per_row(Rows))),
    Sub) :-
    format(atom(Sub),
        'Cooperative load of ~w into shared memory by ~w threads, then warp-shuffle reduction (~w threads × ~w rows per block).',
        [Src, BlockSize, WarpSize, Rows]).
pattern_subtitle(ineffable,
    'Shared-memory pattern not yet classified for this kernel.').


%% -- Body dispatch ------------------------------------------------------------

svg_body(Stream, _KernelName, no_shared_memory) :-
    %% Substrate-honest empty: distinct from ineffable.
    %% Show a visual "no shared memory" — a faint dashed shared-mem region
    %% that's empty, with an explanation.
    format(Stream,
        '  <text x="20" y="100" font-size="14" fill="#666">Each thread reads its input from global memory directly:</text>~n',
        []),
    %% Three threads, each with arrow to global → straight to output.
    %% No shared-memory intermediate.
    emit_no_shmem_illustration(Stream),
    format(Stream,
        '  <rect x="100" y="380" width="800" height="120" fill="#ffffff" stroke="#bbb" stroke-width="1" stroke-dasharray="6 4"/>~n',
        []),
    format(Stream,
        '  <text x="500" y="430" font-size="16" font-weight="bold" fill="#888" text-anchor="middle">No shared memory used by this kernel</text>~n',
        []),
    format(Stream,
        '  <text x="500" y="460" font-size="13" fill="#999" text-anchor="middle">L1 elementwise kernels are embarrassingly parallel; cooperation is unnecessary.</text>~n',
        []).

svg_body(Stream, _KernelName, warp_shuffle_only(N)) :-
    %% Show the warp as 32 thread-cells in a horizontal strip,
    %% with butterfly arrows depicting the warp shuffle pattern.
    emit_warp_shuffle_illustration(Stream, N).

svg_body(Stream, _KernelName, composed(cooperative_load(Src, _, BlockSize),
                                       warp_shuffle(WarpSize, per_row(Rows)))) :-
    %% The substantive case. Two regions:
    %%   1. Top: Global memory tensor (x) → cooperative load → shared memory tile (sx)
    %%      Multiple threads (colored) each contributing different elements
    %%   2. Bottom: Per-row warps reading from sx + global A, accumulating in
    %%      registers, then warp-shuffle reducing
    emit_cooperative_load_illustration(Stream, Src, BlockSize),
    emit_warp_compute_illustration(Stream, WarpSize, Rows).

svg_body(Stream, _KernelName, ineffable) :-
    format(Stream,
        '  <rect x="100" y="200" width="800" height="180" fill="#ffffff" stroke="#999" stroke-width="1" stroke-dasharray="6 4"/>~n',
        []),
    format(Stream,
        '  <text x="500" y="275" font-size="20" font-weight="bold" fill="#666" text-anchor="middle">This Node Intentionally Left Blank</text>~n',
        []),
    format(Stream,
        '  <text x="500" y="310" font-size="14" fill="#888" text-anchor="middle">(Shared-memory pattern not yet classified for this kernel.)</text>~n',
        []).


%% -- Illustrations ------------------------------------------------------------

%% no_shared_memory: show three threads each pulling from global → output, no smem
emit_no_shmem_illustration(Stream) :-
    %% Global memory strip at top
    format(Stream, '  <text x="60" y="140" font-size="12" fill="#444">global memory: input[i]</text>~n', []),
    emit_simple_strip(Stream, 60, 150, 28, 30, 24, 8, gray_palette),
    %% Output strip at bottom
    format(Stream, '  <text x="60" y="240" font-size="12" fill="#444">global memory: output[i]</text>~n', []),
    emit_simple_strip(Stream, 60, 250, 28, 30, 24, 8, gray_palette),
    %% Arrows showing direct read → write (a few exemplars)
    emit_direct_arrows(Stream, 60, 180, 28, 24).

emit_simple_strip(_Stream, _X, _Y, _W, _H, 0, _CPB, _Palette) :- !.
emit_simple_strip(Stream, X, Y, W, H, N, CPB, Palette) :-
    emit_simple_strip_loop(Stream, X, Y, W, H, 0, N, CPB, Palette).
emit_simple_strip_loop(_S, _X, _Y, _W, _H, I, N, _CPB, _P) :- I >= N, !.
emit_simple_strip_loop(Stream, X, Y, W, H, I, N, CPB, Palette) :-
    BlockIdx is I // CPB,
    palette_color(Palette, BlockIdx, Color),
    CX is X + I * W,
    format(Stream,
        '  <rect x="~w" y="~w" width="~w" height="~w" fill="~w" stroke="#444" stroke-width="0.5"/>~n',
        [CX, Y, W, H, Color]),
    TX is CX + W // 2, TY is Y + H // 2 + 4,
    format(Stream,
        '  <text x="~w" y="~w" font-size="10" fill="#333" text-anchor="middle">~w</text>~n',
        [TX, TY, I]),
    I1 is I + 1,
    emit_simple_strip_loop(Stream, X, Y, W, H, I1, N, CPB, Palette).

emit_direct_arrows(Stream, X, Y, CellW, NArrows) :-
    emit_direct_arrows_loop(Stream, X, Y, CellW, 0, NArrows).
emit_direct_arrows_loop(_S, _X, _Y, _W, I, N) :- I >= N, !.
emit_direct_arrows_loop(Stream, X, Y, CellW, I, N) :-
    %% Pick spaced-out indices for arrows
    Idx is I * 3,
    CX is X + Idx * CellW + CellW // 2,
    format(Stream,
        '  <line x1="~w" y1="~w" x2="~w" y2="~w" stroke="#888" stroke-width="1.5" marker-end="url(#arrowhead-gray)"/>~n',
        [CX, Y, CX, Y + 70]),
    I1 is I + 1,
    emit_direct_arrows_loop(Stream, X, Y, CellW, I1, N).


%% warp_shuffle_only: show the 32-thread warp as horizontal strip,
%% with butterfly arrows showing the reduction steps
emit_warp_shuffle_illustration(Stream, N) :-
    format(Stream, '  <text x="60" y="100" font-size="13" fill="#444">warp of ~w threads (per-thread partial sums in registers)</text>~n', [N]),
    %% Show all N threads in a strip
    CellW = 22, CellH = 28,
    XStart = 60, YStart = 110,
    emit_warp_cells(Stream, XStart, YStart, CellW, CellH, N),
    %% Show the butterfly reduction levels below
    Y2 is YStart + CellH + 40,
    format(Stream, '  <text x="60" y="~w" font-size="13" fill="#444">__shfl_xor_sync butterfly reduction:</text>~n', [Y2]),
    %% Four steps for 32-thread reduction (offset 16, 8, 4, 2, 1)
    emit_butterfly_steps(Stream, XStart, Y2 + 20, CellW, CellH, N, [16, 8, 4, 2, 1]).

emit_warp_cells(_S, _X, _Y, _W, _H, 0) :- !.
emit_warp_cells(Stream, X, Y, W, H, N) :-
    emit_warp_cells_loop(Stream, X, Y, W, H, 0, N).
emit_warp_cells_loop(_S, _X, _Y, _W, _H, I, N) :- I >= N, !.
emit_warp_cells_loop(Stream, X, Y, W, H, I, N) :-
    %% All warp threads in same warp color (steel blue)
    CX is X + I * W,
    format(Stream,
        '  <rect x="~w" y="~w" width="~w" height="~w" fill="#779ECB" stroke="#345" stroke-width="0.5"/>~n',
        [CX, Y, W, H]),
    TX is CX + W // 2, TY is Y + H // 2 + 4,
    format(Stream,
        '  <text x="~w" y="~w" font-size="9" fill="#fff" text-anchor="middle">t~w</text>~n',
        [TX, TY, I]),
    I1 is I + 1,
    emit_warp_cells_loop(Stream, X, Y, W, H, I1, N).

emit_butterfly_steps(_S, _X, _Y, _W, _H, _N, []) :- !.
emit_butterfly_steps(Stream, X, Y, CellW, CellH, N, [Offset|Rest]) :-
    format(Stream, '  <text x="~w" y="~w" font-size="11" fill="#666">offset=~w (~w → ~w pairs)</text>~n',
           [X, Y + 12, Offset, N, N // 2]),
    %% Draw a few sample arc-arrows for this offset level
    YArc is Y + 22,
    emit_butterfly_arcs_sample(Stream, X, YArc, CellW, Offset, N),
    Y1 is Y + 50,
    emit_butterfly_steps(Stream, X, Y1, CellW, CellH, N, Rest).

emit_butterfly_arcs_sample(Stream, X, Y, CellW, Offset, N) :-
    %% Draw arcs from thread i to thread i XOR Offset for a few i values
    %% (showing the butterfly without overcrowding)
    SampleIdxs = [0, 1, 2, 3],
    forall(member(I, SampleIdxs),
           ( I < N,
             J is I xor Offset,
             J < N
           ->  X1 is X + I * CellW + CellW // 2,
               X2 is X + J * CellW + CellW // 2,
               XM is (X1 + X2) // 2,
               YDip is Y + 8,
               format(Stream,
                   '  <path d="M ~w,~w Q ~w,~w ~w,~w" stroke="#cc6677" stroke-width="1" fill="none"/>~n',
                   [X1, Y, XM, YDip, X2, Y])
           ;   true
           )).


%% composed: cooperative load + warp shuffle (sgemv_cublas_match)
emit_cooperative_load_illustration(Stream, _Src, BlockSize) :-
    %% Top half: shows x in global memory, sx in shared, with colored
    %% arrows from each thread (color = thread) to the shared mem slots
    %% they load. With BlockSize=128 threads loading K elements (we show K=24
    %% to fit), each thread cycles through K with stride BlockSize.
    %%
    %% For visualization: show 24-element x and sx strips, with 8 threads
    %% colored, each loading 3 elements (stride 8 instead of 128) to keep
    %% the picture comprehensible.
    XStart = 60,

    %% x in global memory
    format(Stream, '  <text x="~w" y="90" font-size="13" fill="#444">x in global memory (K elements)</text>~n', [XStart]),
    emit_indexed_strip(Stream, XStart, 100, 30, 26, 24, thread_palette_24, 'x'),

    %% Arrows showing cooperative load
    Y1 is 100 + 26 + 12,
    format(Stream, '  <text x="~w" y="~w" font-size="12" fill="#666">cooperative load: each of ~w threads loads K/blockDim = ⌈K/~w⌉ elements (stride blockDim)</text>~n',
           [XStart, Y1, BlockSize, BlockSize]),
    emit_load_arrows(Stream, XStart, 100 + 26, 30, 24),

    %% sx in shared memory — same layout, colored by which thread loaded
    Y2 is Y1 + 50,
    format(Stream, '  <text x="~w" y="~w" font-size="13" fill="#444">sx in shared memory (cached x, fast access)</text>~n', [XStart, Y2]),
    Y2c is Y2 + 10,
    emit_indexed_strip(Stream, XStart, Y2c, 30, 26, 24, thread_palette_24, 'sx'),

    %% Highlight the shared region with a soft yellow background to mark it
    %% as "fast memory" — substrate-honest visual cue
    YBoxTop is Y2c - 8,
    BoxH = 50,
    format(Stream,
        '  <rect x="~w" y="~w" width="~w" height="~w" fill="#fff5d6" stroke="#dda" stroke-width="1" rx="4" opacity="0.4"/>~n',
        [XStart - 10, YBoxTop, 24 * 30 + 20, BoxH]),

    %% syncthreads barrier visualization — a horizontal "barrier" line
    YBar is Y2c + 26 + 16,
    format(Stream,
        '  <line x1="~w" y1="~w" x2="~w" y2="~w" stroke="#aa6644" stroke-width="2" stroke-dasharray="4 3"/>~n',
        [XStart, YBar, XStart + 24 * 30, YBar]),
    format(Stream, '  <text x="~w" y="~w" font-size="11" font-style="italic" fill="#aa6644">__syncthreads()  // barrier: all threads finish loading before any thread proceeds</text>~n',
           [XStart, YBar + 14]).

emit_indexed_strip(_S, _X, _Y, _W, _H, 0, _P, _L) :- !.
emit_indexed_strip(Stream, X, Y, W, H, N, Palette, _Label) :-
    emit_indexed_strip_loop(Stream, X, Y, W, H, 0, N, Palette).
emit_indexed_strip_loop(_S, _X, _Y, _W, _H, I, N, _P) :- I >= N, !.
emit_indexed_strip_loop(Stream, X, Y, W, H, I, N, Palette) :-
    %% For thread_palette_24, each thread loads every 8th element starting from tid.
    %% So index I is loaded by thread (I mod 8). Color by that.
    %% With 24 elements & 8 threads → 3 elements per thread.
    LoaderThread is I mod 8,
    palette_color(Palette, LoaderThread, Color),
    CX is X + I * W,
    format(Stream,
        '  <rect x="~w" y="~w" width="~w" height="~w" fill="~w" stroke="#333" stroke-width="0.5"/>~n',
        [CX, Y, W, H, Color]),
    TX is CX + W // 2, TY is Y + H // 2 + 4,
    format(Stream,
        '  <text x="~w" y="~w" font-size="10" fill="#222" text-anchor="middle">~w</text>~n',
        [TX, TY, I]),
    I1 is I + 1,
    emit_indexed_strip_loop(Stream, X, Y, W, H, I1, N, Palette).

emit_load_arrows(Stream, X, Y, CellW, _N) :-
    %% Draw arrows from global cells (above) to shared cells (below).
    %% For visual clarity, draw arrows only for thread 0's three loads (i=0, 8, 16).
    %% Plus a faint indication for other threads.
    %%
    %% Thread 0 loads global x[0], x[8], x[16] → shared sx[0], sx[8], sx[16].
    %% (Stride 8 in our reduced visualization; stride 128 in actual sgemv.)
    %%
    %% Use the t0 color (orange) for thread 0's arrows.
    YGlobalBottom is Y,
    YSharedTop is Y + 50,
    palette_color(thread_palette_24, 0, T0Color),
    forall(member(I, [0, 8, 16]),
           ( CX is X + I * CellW + CellW // 2,
             format(Stream,
                 '  <line x1="~w" y1="~w" x2="~w" y2="~w" stroke="~w" stroke-width="2" marker-end="url(#arrowhead-t0)"/>~n',
                 [CX, YGlobalBottom, CX, YSharedTop, T0Color])
           )),
    %% Faint arrows for thread 1 (color: green) — i=1, 9, 17
    palette_color(thread_palette_24, 1, T1Color),
    forall(member(I, [1, 9, 17]),
           ( CX is X + I * CellW + CellW // 2,
             format(Stream,
                 '  <line x1="~w" y1="~w" x2="~w" y2="~w" stroke="~w" stroke-width="1.2" opacity="0.5"/>~n',
                 [CX, YGlobalBottom, CX, YSharedTop, T1Color])
           )),
    %% Generic ellipsis indication
    XLabel is X + 18 * CellW,
    YLabel is YGlobalBottom + 26,
    format(Stream,
        '  <text x="~w" y="~w" font-size="11" font-style="italic" fill="#888">... 6 more threads load similarly ...</text>~n',
        [XLabel, YLabel]).


%% Bottom half: per-row warps doing the compute + warp-shuffle reduction
emit_warp_compute_illustration(Stream, WarpSize, Rows) :-
    XStart = 60,
    YStart = 360,
    format(Stream,
        '  <text x="~w" y="~w" font-size="13" fill="#444">Compute phase: ~w warps (~w threads each), one per output row of y</text>~n',
        [XStart, YStart, Rows, WarpSize]),
    Y1 is YStart + 16,
    format(Stream,
        '  <text x="~w" y="~w" font-size="12" fill="#666">Each warp reads its row of A (global) + sx (shared, cached above), accumulates in registers, then warp-shuffle reduces.</text>~n',
        [XStart, Y1]),
    %% Show 4 warps as horizontal bands, each labeled
    Y2 is Y1 + 20,
    emit_warp_bands(Stream, XStart, Y2, Rows, WarpSize).

emit_warp_bands(_S, _X, _Y, 0, _W) :- !.
emit_warp_bands(Stream, X, Y, NRows, WarpSize) :-
    emit_warp_bands_loop(Stream, X, Y, 0, NRows, WarpSize).
emit_warp_bands_loop(_S, _X, _Y, R, NR, _W) :- R >= NR, !.
emit_warp_bands_loop(Stream, X, Y, R, NR, WarpSize) :-
    BandY is Y + R * 32,
    %% Color band by warp index
    palette_color(warp_palette, R, BandColor),
    %% Background rectangle for the warp
    BandW is WarpSize * 18 + 100,
    format(Stream,
        '  <rect x="~w" y="~w" width="~w" height="28" fill="~w" stroke="#444" stroke-width="0.5" rx="3" opacity="0.7"/>~n',
        [X, BandY, BandW, BandColor]),
    %% Label
    LabelY is BandY + 18,
    format(Stream,
        '  <text x="~w" y="~w" font-size="11" fill="#222">warp ~w → y[bid*4 + ~w]</text>~n',
        [X + 8, LabelY, R, R]),
    %% Mini thread cells inside band
    InnerX is X + 110,
    forall(between(0, 31, T),
           ( CX is InnerX + T * 16,
             TY is BandY + 4,
             format(Stream,
                 '  <rect x="~w" y="~w" width="14" height="20" fill="#ffffff" stroke="#789" stroke-width="0.3" opacity="0.9"/>~n',
                 [CX, TY])
           )),
    %% Butterfly arrow on the right showing reduction
    ArrowX is X + BandW + 8,
    format(Stream,
        '  <text x="~w" y="~w" font-size="11" fill="#aa6644">→ __shfl_xor 16→8→4→2→1 → y[row]</text>~n',
        [ArrowX, LabelY]),
    R1 is R + 1,
    emit_warp_bands_loop(Stream, X, Y, R1, NR, WarpSize).


%% -- Color palettes -----------------------------------------------------------

%% Pastel-saturated, readable with black text.
palette_color(thread_palette_24, 0, '#FFB347'). % orange
palette_color(thread_palette_24, 1, '#77DD77'). % green
palette_color(thread_palette_24, 2, '#779ECB'). % blue
palette_color(thread_palette_24, 3, '#C49BBB'). % mauve
palette_color(thread_palette_24, 4, '#FDFD96'). % yellow
palette_color(thread_palette_24, 5, '#FFB6C1'). % pink
palette_color(thread_palette_24, 6, '#B0E0E6'). % powder blue
palette_color(thread_palette_24, 7, '#E6BE8A'). % tan
palette_color(thread_palette_24, N, C) :- N >= 8, M is N mod 8, palette_color(thread_palette_24, M, C).

palette_color(warp_palette, 0, '#FFE5B4'). % peach
palette_color(warp_palette, 1, '#C1E1C1'). % pale green
palette_color(warp_palette, 2, '#AEC6CF'). % light blue
palette_color(warp_palette, 3, '#E0BBE4'). % lavender
palette_color(warp_palette, N, C) :- N >= 4, M is N mod 4, palette_color(warp_palette, M, C).

palette_color(gray_palette, _N, '#dcdcdc').
