%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% memory_layout.pl — closed-form CALCULATION of a cache/page/GPU-aligned weight arena.
%%
%% Per Heath: don't just "make it contiguous" — CALCULATE the optimal layout in closed form
%% from the hardware alignment facts (cache-line, page, GPU coalesce width). The q8_0 block
%% is 34 bytes (2-byte f16 scale + 32 int8) — NOT a power of 2, so naive packing makes blocks
%% STRADDLE cache lines (1.88 blocks/line) and rows END mid-line, defeating prefetch/TLB.
%%
%% This module computes, for a given target + tensor shape, the padded row stride and arena
%% size so that: (a) every row starts cache-line aligned, (b) the arena is page-aligned
%% (huge-page-sized for TLB), (c) optionally GPU-coalesce-aligned (128 B).

:- module(memory_layout, [
    align_fact/3,            % align_fact(Target, Kind, Bytes)
    q8_0_block_bytes/1,
    row_stride/4,            % row_stride(Target, K_elems, AlignTo, StrideBytes)
    arena_layout/5,          % arena_layout(Target, Rows, K_elems, AlignTo, layout{...})
    best_alignment/2         % best_alignment(Target, Bytes)  -- the alignment to use
]).

%% ─── Alignment facts (from /proc, getconf, hardware_facts.pl) ───
%% CPU (ivy_bridge): 64B cache line, 4096B page, 2MB huge page.
align_fact(ivy_bridge_e5_2697_v2, cache_line, 64).
align_fact(ivy_bridge_e5_2697_v2, page, 4096).
align_fact(ivy_bridge_e5_2697_v2, huge_page, 2097152).
%% GPU (sm_61 / sm_89): 128B coalesced transaction / cache line.
align_fact(sm_61, cache_line, 128).
align_fact(sm_61, coalesce, 128).
align_fact(sm_89, cache_line, 128).
align_fact(sm_89, coalesce, 128).

%% q8_0 block: QK8_0=32 quants (int8) + 1 f16 scale = 32 + 2 = 34 bytes.
q8_0_block_bytes(34).

%% best_alignment/2: the row-alignment to use for a target = its cache_line (the unit a
%% single memory transaction fetches). For GPU, use the coalesce width (= cache_line = 128).
best_alignment(Target, Bytes) :-
    ( align_fact(Target, coalesce, Bytes) -> true
    ; align_fact(Target, cache_line, Bytes) ).

%% ── closed-form: round Up to a multiple of A ──
round_up(X, A, Y) :- Y is ((X + A - 1) // A) * A.

%% row_stride/4: bytes per padded weight row for K elements of q8_0, aligned to AlignTo.
%%   raw = (K/32) blocks * 34 bytes; padded up to a multiple of AlignTo so the NEXT row
%%   starts aligned. This is the closed form: stride = ceil(K/32 * 34 / AlignTo) * AlignTo.
row_stride(_Target, K, AlignTo, Stride) :-
    q8_0_block_bytes(B),
    NBlocks is K // 32,
    Raw is NBlocks * B,
    round_up(Raw, AlignTo, Stride).

%% arena_layout/5: full arena for Rows x K, returning stride, total size (page/huge-page
%% rounded), padding overhead, and the alignment used. Closed-form, no search.
arena_layout(Target, Rows, K, AlignTo, layout{
        align_to: AlignTo,
        row_stride: Stride,
        raw_row: Raw,
        row_pad: RowPad,
        arena_raw: ArenaRaw,
        arena_padded: ArenaPadded,
        pad_overhead_pct: PadPct,
        page_aligned_to: PageA
    }) :-
    q8_0_block_bytes(B),
    NBlocks is K // 32,
    Raw is NBlocks * B,
    row_stride(Target, K, AlignTo, Stride),
    RowPad is Stride - Raw,
    ArenaRaw is Rows * Stride,
    %% round the whole arena to a huge page (best TLB) if available, else page.
    ( align_fact(Target, huge_page, PageA) -> true
    ; align_fact(Target, page, PageA) -> true
    ; PageA = AlignTo ),
    round_up(ArenaRaw, PageA, ArenaPadded),
    PadPct is (ArenaPadded - (Rows * Raw)) * 100.0 / (Rows * Raw).

%% ─── pack_arena/4: lay out a LIST of tensors contiguously, each at a cache-line-aligned
%% offset, the whole arena huge-page-sized. This is the closed-form weight-arena packing
%% that fixes the scattered-allocation cold-streaming (the measured 66 ms/token loss).
%% Tensors: list of t(Name, Rows, K). Returns placements [p(Name, Offset, Bytes)] + ArenaSize.
pack_arena(Target, Tensors, Placements, ArenaSize) :-
    best_alignment(Target, A),
    pack_(Target, Tensors, A, 0, Placements, EndOff),
    ( align_fact(Target, huge_page, HP) -> true ; align_fact(Target, page, HP) ),
    round_up(EndOff, HP, ArenaSize).

pack_(_, [], _, Off, [], Off).
pack_(Target, [t(Name,Rows,K)|Rest], A, Off, [p(Name,Off,Bytes)|Ps], EndOff) :-
    row_stride(Target, K, A, Stride),
    Bytes is Rows * Stride,
    Next is Off + Bytes,
    round_up(Next, A, NextAligned),   % next tensor starts cache-line aligned
    pack_(Target, Rest, A, NextAligned, Ps, EndOff).
