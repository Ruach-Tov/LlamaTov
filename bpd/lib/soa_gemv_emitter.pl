%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════════
%% soa_gemv_emitter.pl — Emit a SoA Q8_0 matrix-vector product (gemv) in LLVM IR
%% by composing the PROVEN bit-identical q8_0 block-dot over rows.
%%
%% RUNG 2 of the bottom-up SoA chain (Heath's research plan):
%%   atom (q8_0 dot, prolog_to_llvm.pl emit_q8_0_dot) ──▶  THIS (gemv) ──▶ chain ──▶ layer ──▶ model
%%
%% PRINCIPLE — bit-identity by construction (the week's lesson):
%%   The block-dot (emit_q8_0_dot) is ALREADY SoA: weight quants (%wq) and weight
%%   scales (%wd) are SEPARATE pointers, and the per-block FP is mul-then-add
%%   (NOT fma) matching the ggml reference reduction. We DO NOT re-derive the dot.
%%   We COMPOSE it: loop the identical block-dot over `nrows`, with the SoA layout
%%   stride as a FACT, so each row's quant/scale base is computed declaratively.
%%   The hand-written gemv_soa dropped fusion + diverged in reduction order
%%   precisely because it re-implemented the dot. This composes the verified one.
%%
%% SHAPES (the trap that hid tonight's bug): the model uses TWO shapes —
%%   ncols_dst=2 (prefill) and ncols_dst=1 (decode, where tokens are produced).
%%   We emit a gemv parameterized by ncols_dst and MUST verify 0-ULP at BOTH.
%%
%% SoA LAYOUT FACT:
%%   Weight matrix is nrows × ncols (ncols = nb*32 quant lanes; nb blocks/row).
%%   SoA: all quants contiguous  → row r quants base = r * (nb*32)   [i8]
%%        all scales contiguous  → row r scales base = r * nb        [float]
%%   Activation (one column j): quants base = j * (nb*32), scales = j * nb.
%%   Output: dst[j*nrows + r] = dot(weight_row_r, activation_col_j).
%%
%% Verification target: for each (nrows, ncols, ncols_dst), the emitted gemv
%%   must be 0-ULP vs the CPU int-dot-then-float-accumulate reference AND vs
%%   ggml mul_mat_vec_q — at BOTH ncols_dst ∈ {1,2}.
%%
%% Author: Iyun (re-vector to the declarative pipeline, 2026-06-06)
%% Composes the q8_0 block-dot authored by medayek in prolog_to_llvm.pl.
%% ═══════════════════════════════════════════════════════════════════════════

:- module(soa_gemv_emitter, [
    emit_soa_gemv/2,        % emit_soa_gemv(+OutFile, +NColsDst)
    emit_soa_gemv/1         % emit_soa_gemv(+OutFile)  — defaults ncols_dst=1 (decode)
]).

:- use_module(library(lists)).

%% ───────────────────────────────────────────────────────────────────────────
%% SoA layout facts (the WHAT — declarative, not hand-coded strides)
%% ───────────────────────────────────────────────────────────────────────────

%! soa_layout(+Buffer, +Kind, -ElemStrideBlocks)
%  How to advance one "row"/"column" within a SoA buffer, in BLOCKS (nb units).
%  quants advance by nb*32 i8 lanes; scales advance by nb floats.
soa_layout(quants, i8,    lanes).    % stride = nb*32 i8 per row
soa_layout(scales, float, blocks).   % stride = nb  float per row

%! qk(-LanesPerBlock)   — Q8_0 block size (32 int8 quants per block).
qk(32).

%% ───────────────────────────────────────────────────────────────────────────
%% emit_soa_gemv — compose the block-dot over rows × ncols_dst columns
%% ───────────────────────────────────────────────────────────────────────────

emit_soa_gemv(OutFile) :- emit_soa_gemv(OutFile, 1).   % default: decode shape

emit_soa_gemv(OutFile, NColsDst) :-
    qk(QK),
    open(OutFile, write, S),
    format(S, '; ══════════════════════════════════════════════════════════════~n', []),
    format(S, '; SoA Q8_0 gemv  ncols_dst=~w  — composed from emit_q8_0_dot block-dot~n', [NColsDst]),
    format(S, '; Bit-identical by construction: reuses the verified block-dot reduction.~n', []),
    format(S, '; SoA layout: quants contiguous (row r at r*nb*~w i8), scales contiguous (row r at r*nb float).~n', [QK]),
    format(S, '; Output: dst[j*nrows + r] for j in 0..~w, r in 0..nrows.~n', [NColsDst]),
    format(S, '; ══════════════════════════════════════════════════════════════~n~n', []),
    %% the block-dot, declared (will be linked from emit_q8_0_dot output / or inlined)
    format(S, 'declare float @bpd_q8_0_dot(i8* noalias, float* noalias, i8* noalias, float* noalias, i32)~n~n', []),
    %% gemv signature:
    %%   %wq,%wd : full SoA weight quants + scales (nrows*nb*32 i8, nrows*nb float)
    %%   %aq,%ad : activation quants + scales (ncols_dst*nb*32 i8, ncols_dst*nb float)
    %%   %dst    : output (ncols_dst*nrows float)
    %%   %nrows, %nb : dimensions
    format(S, 'define void @bpd_soa_gemv_q8_0(~n', []),
    format(S, '    i8* noalias %wq, float* noalias %wd,~n', []),
    format(S, '    i8* noalias %aq, float* noalias %ad,~n', []),
    format(S, '    float* noalias %dst, i32 %nrows, i32 %nb) {~n', []),
    format(S, 'entry:~n', []),
    format(S, '  %qk = mul i32 %nb, ~w        ; lanes per row/col~n', [QK]),
    format(S, '  br label %col.header~n~n', []),
    %% ── column loop (j in 0..ncols_dst) — emitted explicitly, ncols_dst is small (1 or 2) ──
    emit_col_loop(S, NColsDst),
    format(S, 'exit:~n', []),
    format(S, '  ret void~n', []),
    format(S, '}~n', []),
    close(S),
    format("Emitted SoA gemv (ncols_dst=~w) to ~w~n", [NColsDst, OutFile]).

%% ── column loop: for small ncols_dst we fully unroll the columns (j=0..N-1),
%%    each column running a row-loop that calls the verified block-dot. ──
emit_col_loop(S, NColsDst) :-
    format(S, 'col.header:~n', []),
    format(S, '  ; ncols_dst=~w columns, unrolled; each column runs the row-loop.~n', [NColsDst]),
    format(S, '  br label %col0.row.header~n~n', []),
    NLast is NColsDst - 1,
    forall(between(0, NLast, J), emit_row_loop_for_col(S, J, NColsDst)).

%% ── row loop for column J: r in 0..nrows; dst[J*nrows + r] = dot(W_row_r, A_col_J) ──
emit_row_loop_for_col(S, J, NColsDst) :-
    qk(QK),
    prev_block(J, PrevLabel),          % resolve predecessor label to a concrete atom FIRST
    format(S, 'col~w.row.header:~n', [J]),
    format(S, '  %r~w = phi i32 [ 0, %~w ], [ %r~w.next, %col~w.row.body ]~n',
           [J, PrevLabel, J, J]),
    format(S, '  %rcmp~w = icmp slt i32 %r~w, %nrows~n', [J, J]),
    ( J =:= NColsDst - 1 -> NextBlk = 'exit' ; succ(J, JN), format(atom(NextBlk), 'col~w.row.header', [JN]) ),
    format(S, '  br i1 %rcmp~w, label %col~w.row.body, label %~w~n~n', [J, J, NextBlk]),
    format(S, 'col~w.row.body:~n', [J]),
    %% SoA weight row base: wq + r*(nb*32) i8 ; wd + r*nb float   (soa_layout fact)
    format(S, '  ; SoA fact: weight row ~w-th base = quants[r*nb*~w], scales[r*nb]~n', [J, QK]),
    format(S, '  %wq.off~w = mul i32 %r~w, %qk~n', [J, J]),
    format(S, '  %wq.off~w.i = sext i32 %wq.off~w to i64~n', [J, J]),
    format(S, '  %wq.row~w = getelementptr i8, i8* %wq, i64 %wq.off~w.i~n', [J, J]),
    format(S, '  %wd.off~w = mul i32 %r~w, %nb~n', [J, J]),
    format(S, '  %wd.off~w.i = sext i32 %wd.off~w to i64~n', [J, J]),
    format(S, '  %wd.row~w = getelementptr float, float* %wd, i64 %wd.off~w.i~n', [J, J]),
    %% activation column J base: aq + J*(nb*32) i8 ; ad + J*nb float
    format(S, '  ; activation column ~w base = quants[~w*nb*~w], scales[~w*nb]~n', [J, J, QK, J]),
    format(S, '  %aq.off~w = mul i32 ~w, %qk~n', [J, J]),
    format(S, '  %aq.off~w.i = sext i32 %aq.off~w to i64~n', [J, J]),
    format(S, '  %aq.col~w = getelementptr i8, i8* %aq, i64 %aq.off~w.i~n', [J, J]),
    format(S, '  %ad.off~w = mul i32 ~w, %nb~n', [J, J]),
    format(S, '  %ad.off~w.i = sext i32 %ad.off~w to i64~n', [J, J]),
    format(S, '  %ad.col~w = getelementptr float, float* %ad, i64 %ad.off~w.i~n', [J, J]),
    %% call the VERIFIED block-dot — this is the bit-identity-by-construction point
    format(S, '  ; ── compose verified block-dot (bit-identical reduction) ──~n', []),
    format(S, '  %dot~w = call float @bpd_q8_0_dot(i8* %wq.row~w, float* %wd.row~w, i8* %aq.col~w, float* %ad.col~w, i32 %nb)~n',
           [J, J, J, J, J]),
    %% store dst[J*nrows + r]
    format(S, '  %dst.off~w = mul i32 ~w, %nrows~n', [J, J]),
    format(S, '  %dst.off~w.r = add i32 %dst.off~w, %r~w~n', [J, J, J]),
    format(S, '  %dst.off~w.i = sext i32 %dst.off~w.r to i64~n', [J, J]),
    format(S, '  %dst.p~w = getelementptr float, float* %dst, i64 %dst.off~w.i~n', [J, J]),
    format(S, '  store float %dot~w, float* %dst.p~w, align 4~n', [J, J]),
    format(S, '  %r~w.next = add i32 %r~w, 1~n', [J, J]),
    format(S, '  br label %col~w.row.header~n~n', [J]).

%% helper: the predecessor block label for the phi node (initial r=0 value) of
%% column J's row loop. Column 0 is entered from %entry (via col.header). Column
%% J>0 is entered from col(J-1).row.header — the branch taken when the previous
%% column's row loop finishes (rcmp false → br ... label %colJ.row.header).
prev_block(0, 'col.header').      % entry → col.header → col0.row.header
prev_block(J, Label) :- J > 0, JP is J - 1, format(atom(Label), 'col~w.row.header', [JP]).
