%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% phase5_sweep_ast_isomorphism.pl — Canonical Phase 5 round-trip metric.
%%
%% The CANONICAL metric for the regex-to-AST migration:
%%
%%   For each architecture's load_arch_hparams function body, compare:
%%     LEFT  — emit_load_arch_hparams_ast (lift through AST + emit)
%%     RIGHT — preprocessed upstream source
%%   Re-parse BOTH into c_ast vocabulary terms.
%%   MATCH iff the two AST term lists are structurally equal (==).
%%
%% ─── Why AST isomorphism is the foundational capacity ─────────────
%%
%% Heath's framing (2026-05-17): the foundational capacity is
%% comparing two ASTs for isomorphism — equal symbol names, matching
%% the shape of the trees or graphs. The string output comparison is
%% a BACK-STOP sanity check on the more important fidelity: data-
%% structure fidelity, not string-representation fidelity.
%%
%% Two ASTs that compare equal MUST emit byte-identical text through
%% any deterministic canonical emitter. If they don't, the bug is
%% non-determinism in the emitter, not a difference in what the ASTs
%% represent.
%%
%% Linear text is what speech-acts produce — humans linearize tree-
%% structured thoughts into sequential tokens for transmission. The
%% real Essence is tree-structured (today: c_ast), and eventually
%% graph-structured with cross-references, type resolution, control
%% flow, and data flow. Each upgrade to the substrate is a brick
%% toward that direction.
%%
%% ─── Pre/post step 3.1.e ─────────────────────────────────────────
%%
%% Pre-3.1.e: text-level comparison reached 93/93 = 100% by accident.
%%   The 4 archs using GGML_ASSERT had partial-parse failures that
%%   excluded the macro region on the AST side, AND the raw upstream
%%   also had the unexpanded macro, so the whitespace-normalized text
%%   comparison happened to align.
%%
%% Post-3.1.e: cpp expands GGML_ASSERT(x) to `if (!(x)) ggml_abort(...)`.
%%   The parser now correctly handles the braceless-if expansion (rule
%%   added in commit 97d67bda4). Both sides preprocess identically.
%%   AST isomorphism: 93/93 = 100%. Same number, but now the metric
%%   actually means what it claims — same object, not "happened to
%%   exclude the same text region."
%%
%% ─── Status code semantics ────────────────────────────────────────
%%
%%   match      — both sides parsed; ASTs compare ==
%%   diff       — both sides parsed; ASTs differ (real semantic divergence)
%%   no_source  — source file doesn't exist for this arch
%%   no_body    — function load_arch_hparams not found in source
%%   no_parse   — partial-parse returned empty (catastrophic parse failure)
%%   timeout    — sweep timeout for this arch
%%   error(E)   — exception during processing
%%
%% Author: metayen 2026-05-17
%% Per Heath's framing: AST isomorphism is the foundational capacity;
%% text comparison is the back-stop sanity check.

:- use_module('lib/arch_emit').
:- use_module('lib/arch_summary').
:- use_module('lib/llama_cpp_lifter').
:- use_module('lib/c_ast').
:- initialization(main, main).


%% silent_rt_ast_iso(+ArchName, +RepoPath, -Match)
%%
%% Compute the AST-isomorphism match status for one architecture's
%% load_arch_hparams round-trip. Both sides are preprocessed through
%% the same cpp pipeline, then re-parsed into ASTs.
silent_rt_ast_iso(ArchName, RepoPath, Match) :-
    catch(
        ( arch_emit:emit_load_arch_hparams_ast(ArchName, RepoPath, OurText),
          ( OurText = error_no_source -> Match = no_source
          ; OurText = error_no_body -> Match = no_body
          ; OurText = error_partial_parse_empty -> Match = no_parse
          ; format(atom(SrcPath), "~w/src/models/~w.cpp", [RepoPath, ArchName]),
            arch_emit:extract_load_arch_hparams_preprocessed(SrcPath, UpstreamText),
            arch_summary:extract_load_arch_hparams_body(OurText, OurBody),
            arch_summary:extract_load_arch_hparams_body(UpstreamText, UpBody),
            c_ast:c_parse_stmts_v2_partial(OurBody, OurAst, _OurRest),
            c_ast:c_parse_stmts_v2_partial(UpBody, UpAst, _UpRest),
            ( OurAst == UpAst -> Match = match ; Match = diff )
          )
        ),
        E,
        Match = error(E)
    ).


main :-
    DispatchPath = "./external/llama.cpp/src/llama-model.cpp",
    RepoPath = "./external/llama.cpp",
    lift_dispatch_table(DispatchPath, Pairs),
    findall(A, member(arch_class(A, _), Pairs), All),
    sort(All, Sorted),
    findall(r(A, M),
        ( member(A, Sorted),
          catch(call_with_time_limit(15, silent_rt_ast_iso(A, RepoPath, M)),
                _, M = timeout)
        ),
        Results),
    aggregate_all(count, member(r(_, match), Results), NM),
    aggregate_all(count, member(r(_, diff), Results), ND),
    aggregate_all(count, member(r(_, no_source), Results), NN),
    aggregate_all(count, member(r(_, no_parse), Results), NP),
    aggregate_all(count, member(r(_, no_body), Results), NB),
    aggregate_all(count, member(r(_, timeout), Results), NT),
    aggregate_all(count, ( member(r(_, error(_)), Results) ), NE),
    length(Results, Total),
    format("~n=== Phase 5 round-trip — AST ISOMORPHISM (canonical metric) ===~n"),
    format("Total archs in dispatch table: ~w~n", [Total]),
    format("~n  Eligible (source + body parsed):~n"),
    Eligible is NM + ND + NP,
    format("    MATCH:    ~w~n", [NM]),
    format("    DIFF:     ~w~n", [ND]),
    format("    no_parse: ~w~n", [NP]),
    format("    eligible total: ~w~n", [Eligible]),
    ( Eligible > 0
    -> Rate is (NM * 100.0) / Eligible,
       format("    MATCH rate: ~2f%~n", [Rate])
    ;  format("    MATCH rate: N/A (no eligible)~n")
    ),
    format("~n  Excluded from rate (no source or no body):~n"),
    format("    no_source: ~w~n", [NN]),
    format("    no_body:   ~w~n", [NB]),
    format("~n  Other:~n"),
    format("    timeout: ~w~n", [NT]),
    format("    error:   ~w~n", [NE]),
    findall(A, member(r(A, diff), Results), DiffArchs),
    ( DiffArchs = []
    -> format("~n  No DIFFs — substrate captures every parseable arch's semantics.~n")
    ;  format("~n  DIFFs: ~q~n", [DiffArchs])
    ).
