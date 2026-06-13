%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% =============================================================================
%% threadblock_diagram.pl — SVG diagrams of threadblock-to-tile mappings
%% =============================================================================
%%
%% Per Heath's 2026-05-25 ~00:43 UTC direction:
%%
%%   "Let's start with generating SVGs for only the dimension (1)
%%    threadblock-to-tile mapping. The diagram purpose will be only to
%%    show that mapping in a visual form. Try to use only prolog to
%%    process a kernel specified in BPD, and from that generate a
%%    diagram of this. If the mapping does not exist, the renderer
%%    could render 'This Node Intentionally Left Blank\n(Mapping does
%%    not exist for this kernel.)'  A 1:1 mapping can be represented,
%%    though. Only when the concept of the mapping is ineffable should
%%    the page be intentionally left blank."
%%
%% This is the first substrate-of-substrate-visualization. The same
%% Prolog facts that emit C/CUDA emit SVG via the same canonical-form
%% machinery. Visualization-as-code-generation, not visualization-as-
%% separate-rendering.
%%
%% STARTING SCOPE (substrate-honest about what's in v1):
%%
%%   Only dimension 1 of the kernel-visualization atlas:
%%       threadblock-to-output-tile mapping.
%%
%%   Future dimensions deferred:
%%       - shared-memory loading pattern
%%       - warp-tile sub-decomposition
%%       - memory-stride / coalescing pattern
%%       - K-axis iteration / accumulation state
%%       - precision / dtype overlay
%%       - verification-state overlay (the substantive payoff layer)
%%
%% Three classification outcomes per kernel:
%%
%%   one_to_one(N)
%%       Each thread computes one output element. Canonical for L1
%%       elementwise (elem_op/4 facts). The threadblock is a contiguous
%%       chunk of consecutive output indices; threadblocks tile the
%%       output array.
%%
%%   tiled(Spec)
%%       Non-trivial mapping. Spec carries the parameters required to
%%       render the specific mapping (which dim parallelizes, what
%%       threads-per-block, reduction shape, etc.). v1 supports
%%       sgemv-shaped (one block per output row, threads reduce K).
%%
%%   ineffable
%%       The mapping concept does not substantively apply to this
%%       kernel, OR is too complex for v1 to express faithfully.
%%       Renderer emits "This Node Intentionally Left Blank" per
%%       Heath's direction. Substrate-honest about the substrate-gap.
%%
%% =============================================================================

:- module(threadblock_diagram,
    [kernel_threadblock_mapping/2,
     emit_threadblock_diagram_svg/2,
     write_diagram_for_kernel/2]).

:- use_module(library(lists)).

%% Load the kernel-template substrate the same way generate_blas_kernels.pl does.
:- use_module('kernel_templates_blas.pl').


%% -----------------------------------------------------------------------------
%% kernel_threadblock_mapping(+KernelName, -Mapping)
%%
%% Classify a kernel by its threadblock-to-output-tile mapping.
%% -----------------------------------------------------------------------------

%% L1 elementwise kernels — all elem_op/4 facts share the canonical
%% 1:1 thread-per-output-element mapping. Threadblocks tile the
%% output array as contiguous chunks.
%%
%% Default block size for BPD-emitted elementwise kernels: 256.
%% (Matches the generate_blas_kernels.pl emission pattern.)
kernel_threadblock_mapping(KernelName, one_to_one(256)) :-
    elem_op(KernelName, _Params, _LHS, _RHS),
    !.

%% sgemv: one threadblock per row of the output vector y.
%% Threads within block stride across the K dimension and reduce.
%% Block size 32 (one warp); reduction via warp-shuffle.
kernel_threadblock_mapping(sgemv_substrate_native, tiled(
    block_per_row,
    block_size(32),
    reduction(warp_shuffle))) :- !.

kernel_threadblock_mapping(sgemv_cublas_match, tiled(
    block_per_row_group(8),
    block_size(128),
    reduction(strided_shared))) :- !.

%% Anything else: substrate-honest about the gap. Could be a kernel
%% whose mapping is genuinely ineffable, or just one we haven't yet
%% classified. Both produce the "intentionally left blank" diagram;
%% the distinction lives in whether we extend this predicate later.
kernel_threadblock_mapping(_, ineffable).


%% -----------------------------------------------------------------------------
%% SVG generation
%% -----------------------------------------------------------------------------

%% write_diagram_for_kernel(+KernelName, +OutPath)
%%
%% Convenience: open OutPath, emit the diagram, close.
write_diagram_for_kernel(KernelName, OutPath) :-
    setup_call_cleanup(
        open(OutPath, write, Stream),
        emit_threadblock_diagram_svg(KernelName, Stream),
        close(Stream)),
    format(user_error, "Wrote diagram: ~w~n", [OutPath]).


%% emit_threadblock_diagram_svg(+KernelName, +Stream)
%%
%% Emit the SVG to Stream, dispatched by mapping classification.
emit_threadblock_diagram_svg(KernelName, Stream) :-
    kernel_threadblock_mapping(KernelName, Mapping),
    svg_open(Stream, 800, 360),
    svg_title(Stream, KernelName, Mapping),
    svg_body(Stream, KernelName, Mapping),
    svg_close(Stream).


%% -- SVG primitives -----------------------------------------------------------

svg_open(Stream, W, H) :-
    format(Stream,
        '<svg xmlns="http://www.w3.org/2000/svg" width="~w" height="~w" viewBox="0 0 ~w ~w" font-family="sans-serif">~n',
        [W, H, W, H]),
    %% background
    format(Stream,
        '  <rect x="0" y="0" width="~w" height="~w" fill="#fafafa"/>~n',
        [W, H]).

svg_close(Stream) :-
    format(Stream, '</svg>~n', []).

svg_title(Stream, KernelName, Mapping) :-
    format(Stream,
        '  <text x="20" y="30" font-size="18" font-weight="bold" fill="#222">Threadblock-to-tile mapping: ~w</text>~n',
        [KernelName]),
    mapping_subtitle(Mapping, Subtitle),
    format(Stream,
        '  <text x="20" y="52" font-size="13" fill="#555">~w</text>~n',
        [Subtitle]).

mapping_subtitle(one_to_one(N),
    Sub) :-
    format(atom(Sub),
        '1:1 thread-per-element. Block size ~w. Threadblocks tile the output array.',
        [N]).
mapping_subtitle(tiled(block_per_row, block_size(B), reduction(Kind)),
    Sub) :-
    format(atom(Sub),
        'One block per output row. Block size ~w. Reduction: ~w.',
        [B, Kind]).
mapping_subtitle(tiled(block_per_row_group(G), block_size(B), reduction(Kind)),
    Sub) :-
    format(atom(Sub),
        'Block covers ~w rows. Block size ~w. Reduction: ~w.',
        [G, B, Kind]).
mapping_subtitle(ineffable,
    'Mapping does not exist for this kernel (or is not yet classified).').


%% -- Body: dispatch on mapping kind -------------------------------------------

svg_body(Stream, _KernelName, one_to_one(BlockSize)) :-
    %% Render the output array as a horizontal strip of cells.
    %% Color cells by threadblock assignment. Show enough cells to
    %% display 3 threadblocks worth, with the rest abbreviated as "...".
    NCells = 24,                   %% visible cells
    BlocksShown = 3,
    CellsPerBlock is NCells // BlocksShown,
    CellW = 28,
    CellH = 36,
    XStart = 40,
    YStart = 110,
    emit_strip_label(Stream, XStart, YStart, 'output[i]'),
    emit_strip_cells(Stream, XStart, YStart, CellW, CellH, NCells, CellsPerBlock),
    emit_block_braces(Stream, XStart, YStart, CellW, CellH, BlocksShown, CellsPerBlock),
    emit_strip_ellipsis(Stream, XStart, YStart, CellW, CellH, NCells),
    %% Equation legend
    YEq is YStart + CellH + 80,
    format(Stream,
        '  <text x="40" y="~w" font-size="14" fill="#333" font-family="monospace">i = blockIdx.x * blockDim.x + threadIdx.x</text>~n',
        [YEq]),
    YEq2 is YEq + 22,
    format(Stream,
        '  <text x="40" y="~w" font-size="13" fill="#555">Block ~w covers indices [k*blockDim.x .. (k+1)*blockDim.x - 1] (k = blockIdx.x); blockDim.x = ~w.</text>~n',
        [YEq2, k, BlockSize]).

svg_body(Stream, _KernelName, tiled(block_per_row, block_size(B), reduction(_Kind))) :-
    %% Render the output vector y as a vertical strip of cells.
    %% One block per cell (one block per row of A → one element of y).
    %% Show threads within one block as a labeled horizontal strip
    %% representing K iterations.
    NRows = 6,
    CellW = 50,
    CellH = 28,
    XStart = 60,
    YStart = 90,
    emit_strip_label_vertical(Stream, XStart, YStart, 'y[row]'),
    emit_vstrip_cells(Stream, XStart, YStart, CellW, CellH, NRows),
    %% Annotation: arrows from "block k" labels to each row cell
    emit_block_labels_vertical(Stream, XStart, YStart, CellW, CellH, NRows),
    %% Equation legend
    YEq is YStart + (NRows * CellH) + 50,
    format(Stream,
        '  <text x="40" y="~w" font-size="14" fill="#333" font-family="monospace">row = blockIdx.x;  threads tid in [0..~w) stride across K, reduce.</text>~n',
        [YEq, B]).

svg_body(Stream, _KernelName, tiled(block_per_row_group(G), block_size(B), reduction(_Kind))) :-
    %% Render y as vertical strip; group every G cells into one block.
    NRows = 24,
    CellW = 30,
    CellH = 20,
    XStart = 60,
    YStart = 90,
    emit_strip_label_vertical(Stream, XStart, YStart, 'y[row]'),
    emit_vstrip_cells_grouped(Stream, XStart, YStart, CellW, CellH, NRows, G),
    YEq is YStart + (NRows * CellH) + 50,
    format(Stream,
        '  <text x="40" y="~w" font-size="14" fill="#333" font-family="monospace">Block covers ~w rows; ~w threads per block.</text>~n',
        [YEq, G, B]).

svg_body(Stream, _KernelName, ineffable) :-
    %% This Node Intentionally Left Blank.
    format(Stream,
        '  <rect x="100" y="100" width="600" height="180" fill="#ffffff" stroke="#999" stroke-width="1" stroke-dasharray="6 4"/>~n',
        []),
    format(Stream,
        '  <text x="400" y="175" font-size="20" font-weight="bold" fill="#666" text-anchor="middle">This Node Intentionally Left Blank</text>~n',
        []),
    format(Stream,
        '  <text x="400" y="210" font-size="14" fill="#888" text-anchor="middle">(Mapping does not exist for this kernel.)</text>~n',
        []).


%% -- Helpers ------------------------------------------------------------------

emit_strip_label(Stream, X, Y, Label) :-
    YL is Y - 8,
    format(Stream, '  <text x="~w" y="~w" font-size="12" fill="#444">~w</text>~n', [X, YL, Label]).

emit_strip_label_vertical(Stream, X, Y, Label) :-
    YL is Y - 8,
    format(Stream, '  <text x="~w" y="~w" font-size="12" fill="#444">~w</text>~n', [X, YL, Label]).

emit_strip_cells(_Stream, _X, _Y, _W, _H, 0, _CPB) :- !.
emit_strip_cells(Stream, X, Y, W, H, N, CPB) :-
    N > 0,
    %% Index of this cell (0-based from start)
    %% We emit left-to-right; track using a counter.
    NumCells = N,                  %% expected total
    emit_strip_cells_loop(Stream, X, Y, W, H, 0, NumCells, CPB).

emit_strip_cells_loop(_Stream, _X, _Y, _W, _H, I, N, _CPB) :- I >= N, !.
emit_strip_cells_loop(Stream, X, Y, W, H, I, N, CPB) :-
    BlockIdx is I // CPB,
    block_color(BlockIdx, Color),
    CX is X + I * W,
    format(Stream,
        '  <rect x="~w" y="~w" width="~w" height="~w" fill="~w" stroke="#333" stroke-width="1"/>~n',
        [CX, Y, W, H, Color]),
    TX is CX + W // 2,
    TY is Y + H // 2 + 4,
    format(Stream,
        '  <text x="~w" y="~w" font-size="11" fill="#222" text-anchor="middle">~w</text>~n',
        [TX, TY, I]),
    I1 is I + 1,
    emit_strip_cells_loop(Stream, X, Y, W, H, I1, N, CPB).

emit_block_braces(Stream, X, Y, W, H, BlocksShown, CPB) :-
    YBr is Y + H + 10,
    YBrEnd is YBr + 16,
    YLbl is YBrEnd + 16,
    emit_block_braces_loop(Stream, X, YBr, YBrEnd, YLbl, W, BlocksShown, CPB, 0).

emit_block_braces_loop(_Stream, _X, _YBr, _YBrEnd, _YLbl, _W, BS, _CPB, I) :-
    I >= BS, !.
emit_block_braces_loop(Stream, X, YBr, YBrEnd, YLbl, W, BS, CPB, I) :-
    XL is X + (I * CPB) * W,
    XR is XL + CPB * W,
    XMid is (XL + XR) // 2,
    block_color(I, Color),
    %% Brace line + label
    format(Stream,
        '  <line x1="~w" y1="~w" x2="~w" y2="~w" stroke="~w" stroke-width="3"/>~n',
        [XL, YBr, XR, YBr, Color]),
    format(Stream,
        '  <text x="~w" y="~w" font-size="13" fill="#222" text-anchor="middle">block ~w</text>~n',
        [XMid, YLbl, I]),
    I1 is I + 1,
    emit_block_braces_loop(Stream, X, YBr, YBrEnd, YLbl, W, BS, CPB, I1).

emit_strip_ellipsis(Stream, X, Y, W, H, N) :-
    XE is X + N * W + 4,
    YE is Y + H // 2 + 5,
    format(Stream,
        '  <text x="~w" y="~w" font-size="20" fill="#999">…</text>~n',
        [XE, YE]).

emit_vstrip_cells(_Stream, _X, _Y, _W, _H, 0) :- !.
emit_vstrip_cells(Stream, X, Y, W, H, N) :-
    N > 0,
    emit_vstrip_cells_loop(Stream, X, Y, W, H, 0, N).

emit_vstrip_cells_loop(_Stream, _X, _Y, _W, _H, I, N) :- I >= N, !.
emit_vstrip_cells_loop(Stream, X, Y, W, H, I, N) :-
    block_color(I, Color),
    CY is Y + I * H,
    format(Stream,
        '  <rect x="~w" y="~w" width="~w" height="~w" fill="~w" stroke="#333" stroke-width="1"/>~n',
        [X, CY, W, H, Color]),
    TX is X + W // 2,
    TY is CY + H // 2 + 4,
    format(Stream,
        '  <text x="~w" y="~w" font-size="11" fill="#222" text-anchor="middle">row ~w</text>~n',
        [TX, TY, I]),
    I1 is I + 1,
    emit_vstrip_cells_loop(Stream, X, Y, W, H, I1, N).

emit_vstrip_cells_grouped(_Stream, _X, _Y, _W, _H, 0, _G) :- !.
emit_vstrip_cells_grouped(Stream, X, Y, W, H, N, G) :-
    emit_vstrip_cells_grouped_loop(Stream, X, Y, W, H, 0, N, G).

emit_vstrip_cells_grouped_loop(_Stream, _X, _Y, _W, _H, I, N, _G) :- I >= N, !.
emit_vstrip_cells_grouped_loop(Stream, X, Y, W, H, I, N, G) :-
    BlockIdx is I // G,
    block_color(BlockIdx, Color),
    CY is Y + I * H,
    format(Stream,
        '  <rect x="~w" y="~w" width="~w" height="~w" fill="~w" stroke="#333" stroke-width="1"/>~n',
        [X, CY, W, H, Color]),
    TX is X + W // 2,
    TY is CY + H // 2 + 3,
    format(Stream,
        '  <text x="~w" y="~w" font-size="9" fill="#222" text-anchor="middle">~w</text>~n',
        [TX, TY, I]),
    I1 is I + 1,
    emit_vstrip_cells_grouped_loop(Stream, X, Y, W, H, I1, N, G).

emit_block_labels_vertical(Stream, X, Y, W, H, N) :-
    XL is X + W + 16,
    emit_block_labels_vertical_loop(Stream, XL, Y, H, 0, N).

emit_block_labels_vertical_loop(_Stream, _X, _Y, _H, I, N) :- I >= N, !.
emit_block_labels_vertical_loop(Stream, X, Y, H, I, N) :-
    CY is Y + I * H + H // 2 + 4,
    format(Stream,
        '  <text x="~w" y="~w" font-size="12" fill="#555">← block ~w</text>~n',
        [X, CY, I]),
    I1 is I + 1,
    emit_block_labels_vertical_loop(Stream, X, Y, H, I1, N).


%% -- Color palette ------------------------------------------------------------
%%
%% Substantively the same threadblock-coloring vocabulary you'd see in
%% Rimika Dhara's tiling diagram, adapted for the threadblock-axis:
%% one color per block, cycling. The colors are pastel-saturated so
%% black labels remain readable.

block_color(0, '#FFB347').   %% orange
block_color(1, '#77DD77').   %% green
block_color(2, '#779ECB').   %% blue
block_color(3, '#C49BBB').   %% mauve
block_color(4, '#FDFD96').   %% yellow
block_color(5, '#FFB6C1').   %% pink
block_color(6, '#B0E0E6').   %% powder blue
block_color(7, '#E6BE8A').   %% tan
block_color(N, Color) :- N >= 8, M is N mod 8, block_color(M, Color).
