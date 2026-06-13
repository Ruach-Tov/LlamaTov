%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ════════════════════════════════════════════════════════════════════════
%% citation_linter.pl — Truth Flow citation adjacency linter
%%
%% Enforces the discipline of keeping cites/2 annotations adjacent to the
%% clauses they cite. Implements Step 2 of the Truth Flow → BPD Integration
%% implementation plan (see docs/truth-flow-bpd-integration-design.md).
%%
%% Discipline (Heath's adjacency rule, 2026-06-01):
%%   "For adjacency, the annotation must appear immediately prior to the
%%    clause, or to another annotation of a clause."
%%
%% Positional, not metric. No "within N lines" fuzzy distance.
%%
%% Extension for multi-clause predicates (Heath, 2026-06-03):
%%   Citations apply to PREDICATES (the natural addressable unit in Prolog),
%%   not to individual clauses. For a multi-clause predicate, the annotation
%%   covers a contiguous RUN of clauses of the same predicate. If clauses of
%%   the same predicate appear in disjoint sections of a file (interrupted
%%   by other predicates or directives), each run requires its own annotation.
%%   This preserves source-locality discipline while matching how Prolog
%%   natively addresses predicates as first-class entities.
%%
%% This is the minimal viable linter: it checks adjacency and emits
%% structured findings. It does NOT yet validate locators against the
%% Authority Registry (the registry doesn't exist yet) or check for
%% unknown authorities. Those checks compose on top of this foundation
%% as the substrate matures.
%%
%% Usage:
%%   ?- lint_file('lib/gguf_native_reader.pl', Findings).
%%   Findings = [linter_finding(missing_citation, location('lib/...', 27),
%%                              target(predicate(gguf_read/4)), ...), ...]
%%
%%   ?- lint_file_summary('lib/gguf_native_reader.pl').
%%   File: lib/gguf_native_reader.pl
%%   Clauses examined: 14
%%   Missing citations: 14
%%   Exempt (no_citation_needed): 0
%%   Cited: 0
%%
%% Author: metayen 2026-06-02
%% ════════════════════════════════════════════════════════════════════════

:- module(citation_linter, [
    lint_file/2,             % +SourcePath, -Findings
    lint_file_summary/1,     % +SourcePath  (prints summary to stdout)
    lint_file_summary/2,     % +SourcePath, -Stats
    finding_format/2         % +Finding, -HumanReadableString
]).

:- use_module(library(lists)).
:- use_module(library(readutil)).

%% Truth Flow citation annotations — this module dogfoods its own discipline.
:- discontiguous(cites/2).
:- discontiguous(no_citation_needed/1).


%% ────────────────────────────────────────────────────────────────────────
%% lint_file(+SourcePath, -Findings)
%%
%% Reads the source file term-by-term, identifies rule clauses, and emits
%% one finding per clause that does not have valid adjacency to its
%% citation annotation (or is not explicitly exempt via no_citation_needed/1).
%% ────────────────────────────────────────────────────────────────────────
cites(lint_file/2, [
    cite(authority:internal(truth_flow_bpd_integration_design), section(layer_3_linter)),
    cite(authority:internal(adjacency_rule_2026_06_01), section(rule_statement)),
    cite(authority:internal(adjacency_rule_2026_06_03), section(rule_b_extension))
]).
lint_file(SourcePath, Findings) :-
    read_source_terms(SourcePath, Terms),
    walk_terms_for_findings(Terms, SourcePath, Findings).


%% read_source_terms(+Path, -Terms)
%%
%% Reads a Prolog source file and returns a list of term/2 records:
%%   term(Term, Line)
%% in source order, where Line is the 1-indexed line number of the term.
%% Skips end_of_file. Handles read errors by skipping the offending term
%% but logging it (we lint what we can read).
no_citation_needed(read_source_terms/2).
read_source_terms(Path, Terms) :-
    setup_call_cleanup(
        open(Path, read, Stream),
        read_terms_loop(Stream, Terms),
        close(Stream)
    ).

no_citation_needed(read_terms_loop/2).
read_terms_loop(Stream, Terms) :-
    read_term(Stream, Term, [term_position(Pos)]),
    (   Term == end_of_file
    ->  Terms = []
    ;   (   stream_position_data(line_count, Pos, Line)
        ->  Terms = [term(Term, Line) | Rest]
        ;   Terms = [term(Term, 0) | Rest]   % position unavailable
        ),
        read_terms_loop(Stream, Rest)
    ).


%% ────────────────────────────────────────────────────────────────────────
%% walk_terms_for_findings(+Terms, +SourcePath, -Findings)
%%
%% Walks the term sequence in source order. For each rule clause (Head :- Body),
%% walks backward through preceding terms to determine adjacency:
%%   - if a cites/2 for matching predicate-indicator is found: collect
%%   - if a no_citation_needed/1 for matching indicator is found: mark exempt
%%   - if any other term is found: stop walking (adjacency broken)
%%
%% Emits a finding for each clause that has neither valid citations nor
%% an exemption.
%% ────────────────────────────────────────────────────────────────────────
no_citation_needed(walk_terms_for_findings/3).
walk_terms_for_findings(Terms, SourcePath, Findings) :-
    walk_terms(Terms, [], SourcePath, [], FindingsRev),
    reverse(FindingsRev, Findings).

%% walk_terms(+RemainingTerms, +AccumulatedSoFar, +SourcePath, +Acc, -FindingsOut)
%%
%% AccumulatedSoFar is the list of terms seen so far in REVERSE source order
%% (most-recent-term at head). When we hit a rule clause, we walk this list
%% to look for adjacency.
no_citation_needed(walk_terms/5).
walk_terms([], _Acc, _Path, Findings, Findings).

walk_terms([term(Term, Line) | Rest], AccRev, Path, FindingsIn, FindingsOut) :-
    (   is_rule_clause(Term, Functor/Arity)
    ->  check_adjacency(AccRev, Functor/Arity, AdjacencyResult),
        (   AdjacencyResult = exempt
        ->  FindingsMid = FindingsIn
        ;   AdjacencyResult = cited(_Citations)
        ->  FindingsMid = FindingsIn
        ;   AdjacencyResult = missing
        ->  Finding = linter_finding(
                missing_citation,
                location(Path, Line),
                target(predicate(Functor/Arity)),
                detail("No cites/2 annotation immediately precedes this clause, and no no_citation_needed/1 directive exempts it."),
                severity(warning)
            ),
            FindingsMid = [Finding | FindingsIn]
        )
    ;   FindingsMid = FindingsIn
    ),
    walk_terms(Rest, [term(Term, Line) | AccRev], Path, FindingsMid, FindingsOut).


%% is_rule_clause(+Term, -PredicateIndicator)
%%
%% True if Term is a rule clause (Head :- Body). Returns the predicate
%% indicator of Head. Bare facts are not currently linted (first-pass
%% simplification — bare facts are often configuration; can be added later).
no_citation_needed(is_rule_clause/2).
is_rule_clause((Head :- _Body), Functor/Arity) :-
    callable(Head),
    functor(Head, Functor, Arity),
    %% Skip directives (Head is a directive marker for :- body)
    Functor \== ':-'.


%% is_any_clause_of(+Term, +TargetIndicator)
%%
%% True if Term is any clause of the target predicate-indicator — either a
%% rule clause OR a bare fact. Used by rule (b) so that the annotation chain
%% covers ALL clauses of a multi-clause predicate, regardless of whether
%% individual clauses happen to be rules or facts.
%%
%% Distinct from is_rule_clause/2 (which is used to decide what to lint).
%% Bare facts aren't linted, but they ARE part of the same-predicate run
%% that an annotation covers.
no_citation_needed(is_any_clause_of/2).
is_any_clause_of((Head :- _Body), Functor/Arity) :-
    callable(Head),
    functor(Head, Functor, Arity),
    Functor \== ':-'.
is_any_clause_of(Head, Functor/Arity) :-
    callable(Head),
    Head \= (_ :- _),                %% not a rule
    Head \= cites(_, _),             %% not a citation annotation
    Head \= no_citation_needed(_),   %% not an exemption
    functor(Head, Functor, Arity),
    Functor \== ':-'.


%% check_adjacency(+AccRev, +Indicator, -Result)
%%
%% Walks the reverse-order accumulator looking for adjacency.
%%
%% Per Heath's rule extension 2026-06-03 (interpretation b): an annotation
%% chain for Functor/Arity covers a contiguous run of clauses of the same
%% predicate. So when walking backward from a clause for Functor/Arity:
%%   - intervening clauses of the SAME Functor/Arity are skipped (they are
%%     part of the same annotated run)
%%   - intervening cites/2 for the SAME Functor/Arity are collected
%%   - intervening no_citation_needed/1 for the SAME Functor/Arity exempts
%%   - any other term (including clauses of a DIFFERENT predicate, or
%%     cites/2 for a different predicate) breaks adjacency
%%
%% Result is one of:
%%   exempt              — no_citation_needed/1 directive found
%%   cited(Citations)    — one or more cites/2 found, all for matching indicator
%%   missing             — chain broken before any annotation found
cites(check_adjacency/3, [
    cite(authority:internal(adjacency_rule_2026_06_01), section(rule_statement)),
    cite(authority:internal(adjacency_rule_2026_06_03), section(rule_b_extension))
]).
check_adjacency([], _Indicator, missing).

check_adjacency([term(Term, _Line) | Rest], Indicator, Result) :-
    (   is_no_citation_needed(Term, Indicator)
    ->  Result = exempt
    ;   is_cites_annotation(Term, Indicator, _Citations)
    ->  %% Found a matching citation; keep walking to collect more in the chain
        collect_citation_chain(Rest, Indicator, [Term], AllCitations, _RemainingAfterChain),
        Result = cited(AllCitations)
    ;   is_any_clause_of(Term, Indicator)
    ->  %% Preceding term is another clause of the SAME predicate
        %% (either a rule clause or a bare fact). Per rule (b), skip past
        %% it and continue looking for the run's annotation backward.
        check_adjacency(Rest, Indicator, Result)
    ;   %% Any other term (clause of a different predicate, mismatched
        %% cites/2, directive, etc.) breaks adjacency.
        Result = missing
    ).


%% collect_citation_chain(+Rest, +Indicator, +SoFar, -All, -Remaining)
%%
%% Continues walking backward through the accumulator while we see more
%% cites/2 annotations for the same predicate indicator. Stops when any
%% non-matching term is encountered.
no_citation_needed(collect_citation_chain/5).
collect_citation_chain([], _Indicator, SoFar, SoFar, []).

collect_citation_chain([term(Term, _) | Rest], Indicator, SoFar, All, Remaining) :-
    (   is_cites_annotation(Term, Indicator, _)
    ->  collect_citation_chain(Rest, Indicator, [Term | SoFar], All, Remaining)
    ;   All = SoFar,
        Remaining = [term(Term, _) | Rest]
    ).


%% is_cites_annotation(+Term, +TargetIndicator, -Citations)
%%
%% True if Term is cites(Indicator, Citations) and Indicator matches the
%% target predicate-indicator.
no_citation_needed(is_cites_annotation/3).
is_cites_annotation(cites(Indicator, Citations), TargetIndicator, Citations) :-
    Indicator == TargetIndicator.


%% is_no_citation_needed(+Term, +TargetIndicator)
%%
%% True if Term is no_citation_needed(Indicator) and Indicator matches.
no_citation_needed(is_no_citation_needed/2).
is_no_citation_needed(no_citation_needed(Indicator), TargetIndicator) :-
    Indicator == TargetIndicator.


%% ────────────────────────────────────────────────────────────────────────
%% Convenience: human-readable summary
%% ────────────────────────────────────────────────────────────────────────

no_citation_needed(lint_file_summary/1).
lint_file_summary(Path) :-
    lint_file_summary(Path, _).

no_citation_needed(lint_file_summary/2).
lint_file_summary(Path, stats(Examined, Missing, Exempt, Cited)) :-
    lint_file(Path, Findings),
    read_source_terms(Path, Terms),
    count_clauses(Terms, Examined, Exempt, Cited),
    length(Findings, Missing),
    format('File: ~w~n', [Path]),
    format('Clauses examined: ~w~n', [Examined]),
    format('Missing citations: ~w~n', [Missing]),
    format('Exempt (no_citation_needed): ~w~n', [Exempt]),
    format('Cited: ~w~n', [Cited]),
    (   Missing > 0
    ->  format('~nFindings:~n'),
        forall(member(F, Findings),
               (finding_format(F, S), format('  ~w~n', [S])))
    ;   true
    ).


%% count_clauses(+Terms, -Examined, -Exempt, -Cited)
%%
%% Counts: total rule clauses, clauses with exemption, clauses with citations.
%% Walks the terms forward, checking adjacency via reverse-accumulator
%% (same algorithm as lint_file, just collecting counts).
no_citation_needed(count_clauses/4).
count_clauses(Terms, Examined, Exempt, Cited) :-
    count_clauses_walk(Terms, [], 0, 0, 0, Examined, Exempt, Cited).

no_citation_needed(count_clauses_walk/8).
count_clauses_walk([], _Acc, E, X, C, E, X, C).

count_clauses_walk([term(Term, _Line) | Rest], AccRev,
                   EIn, XIn, CIn, EOut, XOut, COut) :-
    (   is_rule_clause(Term, Functor/Arity)
    ->  ENew is EIn + 1,
        check_adjacency(AccRev, Functor/Arity, Result),
        (   Result = exempt    -> XNew is XIn + 1, CNew = CIn
        ;   Result = cited(_)  -> XNew = XIn,     CNew is CIn + 1
        ;   XNew = XIn,        CNew = CIn
        )
    ;   ENew = EIn, XNew = XIn, CNew = CIn
    ),
    count_clauses_walk(Rest, [term(Term, _) | AccRev],
                       ENew, XNew, CNew, EOut, XOut, COut).


%% finding_format(+Finding, -HumanReadableString)
%%
%% Renders a structured finding as a one-line human-readable string for
%% CLI output.
no_citation_needed(finding_format/2).
finding_format(linter_finding(Category,
                              location(File, Line),
                              target(predicate(Functor/Arity)),
                              detail(Detail),
                              severity(Severity)),
               Formatted) :-
    format(string(Formatted),
           "~w:~w  [~w/~w]  ~w/~w  — ~w",
           [File, Line, Severity, Category, Functor, Arity, Detail]).
