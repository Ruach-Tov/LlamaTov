%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% lift_coverage.pl — Cross-tabulation of BPD-fact-lifting coverage across substrate-sources.
%%
%% Per Heath's 2026-05-26 substrate-direction: automated data product showing
%% which substrate-sources have working BPD-fact-lifting, with SVG tabular
%% projection (green/red cells).
%%
%% Substrate-vocabulary:
%%   substrate-source: a corpus of kernel implementations we want to lift
%%     (e.g., Stanford KernelBench L1, torch-cfd, llama.cpp/ggml).
%%   capability: a substrate-of-lifting operation
%%     (parse, lift to BPD facts, regenerate, execute via Prolog resolution,
%%      emit Rust+CUDA-oxide).
%%   cell: lift_coverage(Source, Capability, Status, Evidence)
%%     Status: yes | partial | no | planned
%%     Evidence: list of file paths or commits substrate-of-record.
%%
%% Two layers:
%%   1. lift_coverage_*/N facts: the substrate-of-record (this file, declarative).
%%   2. emit_lift_coverage_svg/1: the SVG projection (called via swipl -g).
%%
%% Both layers are substrate-honest: facts come from empirical substrate-audit,
%% the SVG is just a projection. Update the facts; regenerate the SVG.

:- module(lift_coverage, [
    lift_coverage/4,
    substrate_source/2,
    capability/2,
    status_color/2,
    table_id/3,
    cell_dim_expression/4,
    emit_lift_coverage_svg/1,
    main/0
]).


%% ─────────────────────────────────────────────────────────────────────────────
%% Document registry — canonical document numbering for declaratively-generated,
%% publicly-indexed artifacts. Per Heath's direction (2026-05-26):
%%   - Numbers start at 10000 to leave room for past artifacts to be retroactively
%%     numbered as 0-9999.
%%   - 10000 = this table, the BPD Lifting Coverage cross-tab.
%%   - A document is Table(N) only if it is both declaratively-generated from
%%     substrate-of-record AND publicly-indexed at a stable address.
%% ─────────────────────────────────────────────────────────────────────────────

%% table_id(+Number, +Name, +SchemaTerm).
table_id(10000, 'BPD_Lifting_Coverage',
         cross_tab_2d(capability, substrate_source.filter(public_kernel_corpus))).


%% cell_dim_expression(+Table, +Column, +Row, -DimExpr)
%% Build the watermarkup data-dim expression for a cell at canonical
%% coordinates. The expression is a nested term that an AI agent can
%% parse without lookup tables, per the Ruach Tov watermarkup convention
%% (blog post #10: The Web That Was Supposed to Be).
%%
%% Example:
%%   ?- cell_dim_expression(10000, emit_llvm, yolo, E).
%%   E = 'cell(table(10000),column(emit_llvm),row(yolo))'
cell_dim_expression(Table, Column, Row, DimExpr) :-
    format(atom(DimExpr),
        'cell(table(~w),column(~w),row(~w))',
        [Table, Column, Row]).

:- use_module(library(lists)).
:- use_module(lib/dashboard_common, [freshness_stamp_svg/4]).


%% ─────────────────────────────────────────────────────────────────────────────
%% Substrate sources (rows of the cross-tab)
%% ─────────────────────────────────────────────────────────────────────────────

%% substrate_source(+SourceAtom, +DisplayName).
substrate_source(kernelbench_l1, 'Stanford KernelBench L1').
substrate_source(kernelbench_l2, 'Stanford KernelBench L2').
substrate_source(kernelbench_l3, 'Stanford KernelBench L3').
substrate_source(torch_cfd,      'torch-cfd').
substrate_source(genomics,       'Genomics alignment').
substrate_source(ollama_models,  'Ollama models (GGUF)').
substrate_source(llama_cpp,      'llama.cpp / ggml').
substrate_source(yolo,           'YOLO (PyTorch)').
substrate_source(other_cfd,      'Other CFD').


%% ─────────────────────────────────────────────────────────────────────────────
%% Capabilities (columns of the cross-tab)
%% ─────────────────────────────────────────────────────────────────────────────

%% capability(+CapAtom, +DisplayName).
%%
%% Per medayek's β-stage substrate-correction (2026-05-26): the single
%% column 'BPD fact lift' substantively-conflated MANUAL lifting (where
%% substrate-of-record is hand-translated Python/C++ → Prolog facts)
%% with AUTOMATED parsing (where a parser-substrate substantively-ingests
%% source and emits facts).
%%
%% torch-cfd substantively-has 58 manually-lifted ops, no automated
%% parser. Substrate-schema substrate-needs to distinguish these.
capability(parse,         'Parse').
capability(lift_manual,   'Lift (manual)').
capability(lift_auto,     'Lift (auto)').
capability(verify,        'Verify').
capability(roundtrip,     'Roundtrip').
capability(execute,       'Execute').
capability(emit_cuda,     'CUDA').
capability(emit_llvm,     'LLVM IR').
capability(emit_rust,     'Rust+oxide').


%% ─────────────────────────────────────────────────────────────────────────────
%% Coverage cells (the substrate-of-record)
%% ─────────────────────────────────────────────────────────────────────────────
%%
%% lift_coverage(+Source, +Capability, +Status, +Evidence).
%%   Status: yes | partial | no | planned
%%   Evidence: a short atom or string naming the substrate-of-record file(s).

%% ── llama.cpp / ggml — the most-developed substrate-source ──
lift_coverage(llama_cpp, parse,       yes,
    'bpd/lib/c_preprocess.pl + bpd/lib/c_ast.pl (mavchin) — c_raw→c_ast quality improvements ongoing (boneh: commit 37bc01524 maxpool/avgpool)').
lift_coverage(llama_cpp, lift_manual, yes,
    'qkv_lifter.pl + ffn_expander.pl (manual mappings)').
lift_coverage(llama_cpp, lift_auto,   yes,
    'bpd/lib/llama_cpp_lifter.pl + arch_summary.pl — ingests llama.cpp source').
lift_coverage(llama_cpp, verify,      yes,
    'tests/test_qkv_roundtrip.pl + test_ffn_roundtrip.pl bit-identical').
lift_coverage(llama_cpp, roundtrip,   yes,
    'bpd/lib/arch_emit.pl — regenerate llama.cpp from BPD facts').
lift_coverage(llama_cpp, execute,     partial,
    'bpd/llamatov_llama.pl — Prolog-resolution execution operational').
lift_coverage(llama_cpp, emit_cuda,   yes,
    'bpd/lib/kernel_templates*.pl + generate_*_kernels.pl').
lift_coverage(llama_cpp, emit_llvm,   partial,
    'vec_dot 0 ULP vs ggml SSE3 (mavchin 2026-05-27); <4 x float>, 8 accumulators, hadd reduction. 1/9 IR emission patterns complete (reduction); 7 to build, 1 future per kernel_patterns.pl').
lift_coverage(llama_cpp, emit_rust,   partial,
    'bpd/rust/kernel-harness — emits Rust kernels but not full CUDA-oxide').

%% ── Stanford KernelBench L1 — single-op classification ──
lift_coverage(kernelbench_l1, parse,       yes,
    'bpd/tests/test_kernelbench_l1_structure.pl').
lift_coverage(kernelbench_l1, lift_manual, yes,
    'bpd/tests/kernelbench_l1_problems.pl (100 ops manually classified)').
lift_coverage(kernelbench_l1, lift_auto,   no,
    'no automated parser for KernelBench Python; all 100 ops hand-classified').
lift_coverage(kernelbench_l1, verify,      partial,
    'test_kernelbench_l1_semantic.py — semantic checks per problem').
lift_coverage(kernelbench_l1, roundtrip,   partial,
    'roundtrip via ggml taxonomy mapping, not byte-identical Python regen').
lift_coverage(kernelbench_l1, execute,     partial,
    'bpd/tests/test_kernelbench_l1_semantic.py + test_kernelbench_l1_cuda.pl').
lift_coverage(kernelbench_l1, emit_cuda,   yes,
    'test_kernelbench_l1_cuda.pl substantively-emits CUDA per problem').
lift_coverage(kernelbench_l1, emit_llvm,   partial,
    '1/9 IR emission patterns operational (reduction, 0 ULP) covering subset of 44 L1 ops; kernel_patterns.pl maps ops to patterns (mavchin 2026-05-27)').
lift_coverage(kernelbench_l1, emit_rust,   planned,
    'no L1-specific Rust emit yet').

%% ── Stanford KernelBench L2 — fusion patterns ──
lift_coverage(kernelbench_l2, parse,       yes,
    'bpd/tests/kernelbench_l2_problems.pl (mavchin)').
lift_coverage(kernelbench_l2, lift_manual, partial,
    'fusion rules exist in fusion_rules.pl; end-to-end roundtrip not verified on L2 set (mavchin β-correction)').
lift_coverage(kernelbench_l2, lift_auto,   no,
    'no automated PyTorch graph → BPD parser').
lift_coverage(kernelbench_l2, verify,      partial,
    'fusion-property tests partial').
lift_coverage(kernelbench_l2, roundtrip,   planned,
    'no PyTorch regen path from L2 facts yet').
lift_coverage(kernelbench_l2, execute,     partial,
    'bpd/tests/test_kernelbench_l2.pl exists').
lift_coverage(kernelbench_l2, emit_cuda,   partial,
    'partial via fusion_to_cuda.pl').
lift_coverage(kernelbench_l2, emit_llvm,   no, '').
lift_coverage(kernelbench_l2, emit_rust,   no, '').

%% ── Stanford KernelBench L3 — full models ──
lift_coverage(kernelbench_l3, parse,       partial,
    'bpd/tests/test_kernelbench_l3.pl + test_l3_readiness.pl').
lift_coverage(kernelbench_l3, lift_manual, partial,
    'partial coverage; models in L3 too large for full lift yet').
lift_coverage(kernelbench_l3, lift_auto,   no, '').
lift_coverage(kernelbench_l3, verify,      no, '').
lift_coverage(kernelbench_l3, roundtrip,   no, '').
lift_coverage(kernelbench_l3, execute,     no, '').
lift_coverage(kernelbench_l3, emit_cuda,   no, '').
lift_coverage(kernelbench_l3, emit_llvm,   no, '').
lift_coverage(kernelbench_l3, emit_rust,   no, '').

%% ── torch-cfd — substrate-corrected per medayek 2026-05-26 ──
lift_coverage(torch_cfd, parse,       yes,
    'bpd/docs/torch_cfd_lifting_catalog.py + tests/test_torch_cfd_stencils.py').
lift_coverage(torch_cfd, lift_manual, yes,
    'bpd/lib/torch_cfd_lifted.pl (medayek) — 58 ops manually lifted').
lift_coverage(torch_cfd, lift_auto,   no,
    'no pytorch_to_prolog.py parser; manual translation only (medayek)').
lift_coverage(torch_cfd, verify,      partial,
    '12/58 verified vs PyTorch · 7/8 stencils + 4/7 spectral at 0 ULP (medayek)').
lift_coverage(torch_cfd, roundtrip,   planned,
    'PyTorch regen path documented in torch_cfd_lifted.pl header, not implemented').
lift_coverage(torch_cfd, execute,     partial,
    'bpd/tests/test_cfd_substrate.py').
lift_coverage(torch_cfd, emit_cuda,   partial,
    '4 Sod shock tube kernels; stencils compiled on P4 (medayek)').
lift_coverage(torch_cfd, emit_llvm,   no,
    'torch-cfd ops not yet in prolog_to_llvm.pl (medayek)').
lift_coverage(torch_cfd, emit_rust,   planned,
    'documented in torch_cfd_lifted.pl as future-substrate').

%% ── YOLO (PyTorch) ── substantially corrected by mavchin 2026-05-26 β-stage
lift_coverage(yolo, parse,       yes,
    'pytorch_to_prolog.py parses YOLOv5n PyTorch source (mavchin, bpd-substrate commit 60fcacc)').
lift_coverage(yolo, lift_manual, yes,
    'yolo_graph.pl — manually-written substrate-of-record (mavchin)').
lift_coverage(yolo, lift_auto,   yes,
    'pytorch_to_prolog.py — 219 ops, 196 edges, 480 attrs for YOLOv5n (mavchin, commit 60fcacc)').
lift_coverage(yolo, verify,      yes,
    'round-trip verified at 0 ULP for 10 backbone layers (mavchin)').
lift_coverage(yolo, roundtrip,   yes,
    '0 ULP round-trip on 10 layers — substrate-of-record regenerates source-equivalent PyTorch graph').
lift_coverage(yolo, execute,     partial,
    'partial — Prolog resolution operational, full execution path WIP').
lift_coverage(yolo, emit_cuda,   yes,
    '49 CUDA kernels emitted, 25+ at 0 ULP (mavchin)').
lift_coverage(yolo, emit_llvm,   partial,
    'Prolog→LLVM emitter proven; vec_width sweep is next (mavchin)').
lift_coverage(yolo, emit_rust,   planned,
    'Rust+CUDA-oxide emission path — future substrate-direction (Heath)').

%% ── Ollama models (GGUF substrate) ──
lift_coverage(ollama_models, parse,       yes,
    'bpd/lib/model_zoo.pl + gguf_native_reader (test_gguf_native_reader.pl)').
lift_coverage(ollama_models, lift_manual, yes,
    'manually-derived from llama.cpp + GGUF metadata').
lift_coverage(ollama_models, lift_auto,   yes,
    'GGUF metadata + tensor layout → BPD facts (model_zoo + llama_cpp_lifter compose)').
lift_coverage(ollama_models, verify,      partial,
    'test_ollama_match.py — token-level compare against Ollama').
lift_coverage(ollama_models, roundtrip,   partial,
    'inference round-trip via test_ollama_match.py (token-level)').
lift_coverage(ollama_models, execute,     partial,
    'llamatov_inference.so — partial; test_ollama_match.py compares').
lift_coverage(ollama_models, emit_cuda,   partial,
    'via llama.cpp/ggml emission path').
lift_coverage(ollama_models, emit_llvm,   no, '').
lift_coverage(ollama_models, emit_rust,   no, '').

%% ── Genomics alignment ──
lift_coverage(genomics, parse,       no,
    'no genomics substrate-of-record found in BPD').
lift_coverage(genomics, lift_manual, no, '').
lift_coverage(genomics, lift_auto,   no, '').
lift_coverage(genomics, verify,      no, '').
lift_coverage(genomics, roundtrip,   no, '').
lift_coverage(genomics, execute,     no, '').
lift_coverage(genomics, emit_cuda,   no, '').
lift_coverage(genomics, emit_llvm,   no, '').
lift_coverage(genomics, emit_rust,   no, '').

%% ── Other CFD ──
lift_coverage(other_cfd, parse,       no,
    'only torch-cfd substantively-lifted; other CFD frameworks unattempted').
lift_coverage(other_cfd, lift_manual, no, '').
lift_coverage(other_cfd, lift_auto,   no, '').
lift_coverage(other_cfd, verify,      no, '').
lift_coverage(other_cfd, roundtrip,   no, '').
lift_coverage(other_cfd, execute,     no, '').
lift_coverage(other_cfd, emit_cuda,   no, '').
lift_coverage(other_cfd, emit_llvm,   no, '').
lift_coverage(other_cfd, emit_rust,   no, '').


%% ─────────────────────────────────────────────────────────────────────────────
%% Status → SVG color mapping
%% ─────────────────────────────────────────────────────────────────────────────

status_color(yes,     '#3f7c5c').   %% green
status_color(partial, '#daa520').   %% amber
status_color(no,      '#8a3a2a').   %% red-shadow
status_color(planned, '#6a6a6a').   %% slate-shadow (greyed)


%% ─────────────────────────────────────────────────────────────────────────────
%% SVG projection
%% ─────────────────────────────────────────────────────────────────────────────

emit_lift_coverage_svg(Path) :-
    findall(S-N, substrate_source(S, N), Sources),
    findall(C-N, capability(C, N), Capabilities),
    length(Sources, NRows),
    length(Capabilities, NCols),

    %% Layout: header column for source names + one column per capability.
    %% Cell dimensions. SvgW is sized to fit title at 22pt font (~600px wide)
    %% plus the table; whichever is wider wins.
    %% Per medayek's β-correction: 9 capability-columns now (was 6).
    %% Narrower cells to keep total canvas substrate-substrate-manageable.
    CellW = 110, CellH = 36, HeaderH = 70, LabelW = 220,
    TableW is LabelW + NCols * CellW,
    TableH is HeaderH + NRows * CellH,
    %% Legend lives in its own document-element below the table, with
%% a whitespace gutter separating it from the grid. The legend is
%% centered on document.center_x — not aligned to row-labels — because
%% it documents the color scheme (a document-level dimension), not the
%% per-source rows.
    TitleH = 80,
    LegendGutter = 32,    %% whitespace between table bottom and legend
    LegendBlockH = 28,    %% height the legend itself occupies
    LegendBottomPad = 24, %% breathing room below legend before SVG edge
    FooterH is LegendGutter + LegendBlockH + LegendBottomPad,
    %% Title at 20pt 'BPD lifting coverage' measures ~250px. Table width is
    %% LabelW (220) + NCols*CellW. For 9 capabilities at CellW=110 that's
    %% 220 + 990 = 1210 + 40 padding = 1250 minimum.
    SvgW0 is TableW + 60,
    SvgW is max(SvgW0, 1280),
    SvgH is TitleH + TableH + FooterH + 28,
    TableLeft is (SvgW - TableW) // 2,

    setup_call_cleanup(
        open(Path, write, S),
        ( format(S,
            '<svg xmlns="http://www.w3.org/2000/svg" width="~w" height="~w" viewBox="0 0 ~w ~w" font-family="Georgia, serif">~n',
            [SvgW, SvgH, SvgW, SvgH]),
          format(S,
            '  <rect x="0" y="0" width="~w" height="~w" fill="#f8f5ee"/>~n',
            [SvgW, SvgH]),

          %% Title
          TitleX is SvgW // 2,
          format(S, '  <text x="~w" y="36" font-size="20" font-weight="bold" fill="#3a2a1a" text-anchor="middle">BPD lifting coverage</text>~n',
            [TitleX]),
          format(S, '  <text x="~w" y="60" font-size="11" fill="#5a4a3a" text-anchor="middle">~w substrate-sources × ~w capabilities · regenerated from bpd/lift_coverage.pl</text>~n',
            [TitleX, NRows, NCols]),

          %% Column headers
          forall(nth0(Idx, Capabilities, _-CName),
              ( CX is TableLeft + LabelW + Idx * CellW + CellW // 2,
                CYTop is TitleH + HeaderH - 16,
                format(S, '  <text x="~w" y="~w" font-size="12" font-weight="bold" fill="#3a2a1a" text-anchor="middle">~w</text>~n',
                    [CX, CYTop, CName])
              )),

          %% Rows
          forall(nth0(RowIdx, Sources, Src-SName),
              ( RY is TitleH + HeaderH + RowIdx * CellH,
                RowMidY is RY + CellH // 2 + 4,
                %% Row label
                LabelX is TableLeft + 10,
                format(S, '  <text x="~w" y="~w" font-size="13" fill="#3a2a1a">~w</text>~n',
                    [LabelX, RowMidY, SName]),
                %% Cells — each carries watermarkup data-dim expression naming
                %% its canonical coordinate in the document registry, per the
                %% Ruach Tov inline-semantic-annotation convention.
                forall(nth0(ColIdx, Capabilities, Cap-_),
                    ( CX is TableLeft + LabelW + ColIdx * CellW,
                      ( lift_coverage(Src, Cap, Status, _)
                      -> status_color(Status, Color)
                      ;  Status = no, Color = '#cccccc'
                      ),
                      CellMidX is CX + CellW // 2,
                      %% Evaluate numeric attributes BEFORE format/3 — strict SVG
                      %% parsers (Firefox) reject 'x="255+2"' unevaluated arithmetic.
                      RectX is CX + 2,
                      RectY is RY + 2,
                      RectW is CellW - 4,
                      RectH is CellH - 4,
                      cell_dim_expression(10000, Cap, Src, DimExpr),
                      format(S, '  <rect class="m" data-dim="~w" x="~w" y="~w" width="~w" height="~w" fill="~w" stroke="#3a2a1a" stroke-width="1"/>~n',
                          [DimExpr, RectX, RectY, RectW, RectH, Color]),
                      status_glyph(Status, Glyph),
                      format(S, '  <text x="~w" y="~w" font-size="14" font-weight="bold" fill="#f8f5ee" text-anchor="middle">~w</text>~n',
                          [CellMidX, RowMidY, Glyph])
                    ))
              )),

          %% Legend — its own document-element. Centered on document.center_x
          %% with whitespace gutter above. Distinctive style: stands out
          %% by alignment-discipline (document-axis, not row-spine).
          %%
          %% Layout: [Legend:] [swatch] yes (working)  [swatch] partial  ...
          %% We measure approximate widths of each fragment, sum the total,
          %% then offset by (SvgW - LegendTotalW) // 2 to center.
          %%
          %% Approx widths at 12pt Georgia:
          %%   "Legend:"          ≈ 50
          %%   swatch (16) + gap  ≈ 22
          %%   "yes (working)"    ≈ 85
          %%   "partial"          ≈ 50
          %%   "no (not started)" ≈ 105
          %%   "planned"          ≈ 55
          %% Inter-group gap: 32 px
          LegendTotalW is 50 + 32 + (22 + 85) + 32 + (22 + 50) + 32 + (22 + 105) + 32 + (22 + 55),
          LegendStartX is (SvgW - LegendTotalW) // 2,
          %% Y baseline: below table bottom + gutter, vertically centered in LegendBlockH.
          TableBottom is TitleH + TableH,
          LegendY is TableBottom + LegendGutter + (LegendBlockH * 2 // 3),
          %% "Legend:" label
          format(S, '  <text x="~w" y="~w" font-size="12" font-weight="bold" fill="#3a2a1a">Legend:</text>~n',
              [LegendStartX, LegendY]),
          %% Four swatch+label pairs, positioned sequentially
          L1 is LegendStartX + 50 + 32,
          L2 is L1 + 22 + 85 + 32,
          L3 is L2 + 22 + 50 + 32,
          L4 is L3 + 22 + 105 + 32,
          emit_legend_cell(S, 'yes (working)',     yes,     L1, LegendY),
          emit_legend_cell(S, 'partial',           partial, L2, LegendY),
          emit_legend_cell(S, 'no (not started)',  no,      L3, LegendY),
          emit_legend_cell(S, 'planned',           planned, L4, LegendY),

          StampY is LegendY + 30,
          freshness_stamp_svg(S, 20, StampY, 10),
          format(S, '</svg>~n', [])
        ),
        close(S)),
    format(user_error, "Wrote: ~w~n", [Path]).

status_glyph(yes, '\u2713').      %% checkmark
status_glyph(partial, '\u25D0').  %% half-filled circle
status_glyph(no, '').
status_glyph(planned, '\u25CB').  %% empty circle

emit_legend_cell(S, Label, Status, X, Y) :-
    status_color(Status, Color),
    Y0 is Y - 12,
    format(S, '  <rect x="~w" y="~w" width="16" height="16" fill="~w" stroke="#3a2a1a"/>~n',
        [X, Y0, Color]),
    LabelX is X + 22,
    format(S, '  <text x="~w" y="~w" font-size="11" fill="#3a2a1a">~w</text>~n',
        [LabelX, Y, Label]).


%% ─────────────────────────────────────────────────────────────────────────────
%% main — run via: swipl -g main lift_coverage.pl -- [output_path]
%% ─────────────────────────────────────────────────────────────────────────────

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [OutPath|_] -> true
    ;   OutPath = '/tmp/output-only/lift_coverage.o.svg'
    ),
    emit_lift_coverage_svg(OutPath),
    halt(0).
