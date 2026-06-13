%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ════════════════════════════════════════════════════════════════════════
%% citation_extractor.pl — Extract cites/2 facts from annotated Prolog files
%%
%% Implements the producer side of "Citations survive projection" for
%% Truth Flow → BPD Integration. Given an annotated Prolog source file,
%% extracts all cites/2 and no_citation_needed/1 facts as queryable Prolog
%% terms, ready to be projected by an emitter into target-substrate-
%% appropriate citation form (Markdown bibliography, C comments, LLVM IR
%% metadata, SVG data-cites attributes).
%%
%% Architectural role: this module sits between the source file and the
%% emitter chain. It does NOT itself project; it produces an extracted
%% citation substrate that any emitter can consume.
%%
%% Usage:
%%   ?- extract_citations('lib/gguf_native_reader.pl', Citations, Exemptions).
%%   Citations = [
%%       citation(predicate(gguf_read/4),
%%                [cite(authority:internal(gguf_spec_v3),
%%                      section(file_layout)),
%%                 cite(authority:internal(llamacpp_gguf_reference),
%%                      section(reading_a_file))]),
%%       ...
%%   ],
%%   Exemptions = [
%%       exemption(predicate(gguf_data_start/2)),
%%       ...
%%   ].
%%
%% Author: metayen 2026-06-03
%% ════════════════════════════════════════════════════════════════════════

:- module(citation_extractor, [
    extract_citations/3,         % +SourcePath, -Citations, -Exemptions
    extract_citations_grouped/2  % +SourcePath, -CitationsByPredicate
]).

:- use_module(library(lists)).

%% Truth Flow citation annotations — this module dogfoods its own discipline.
:- discontiguous(cites/2).
:- discontiguous(no_citation_needed/1).


%% ────────────────────────────────────────────────────────────────────────
%% extract_citations(+SourcePath, -Citations, -Exemptions)
%%
%% Reads the source file and extracts all cites/2 and no_citation_needed/1
%% facts. Returns two flat lists. Multiple cites/2 for the same predicate
%% are returned as separate citation/2 entries (the emitter aggregates them
%% per-predicate via extract_citations_grouped/2 if needed).
%% ────────────────────────────────────────────────────────────────────────
cites(extract_citations/3, [
    cite(authority:internal(truth_flow_bpd_integration_design),
         section(layer_4_projection_preservation))
]).
extract_citations(SourcePath, Citations, Exemptions) :-
    read_source_terms(SourcePath, Terms),
    walk_terms_for_annotations(Terms, [], [], CitationsRev, ExemptionsRev),
    reverse(CitationsRev, Citations),
    reverse(ExemptionsRev, Exemptions).


%% read_source_terms(+Path, -Terms)
%%
%% Reads a Prolog file and returns a list of source-ordered terms.
%% Skips end_of_file. Does not need positional info for citation extraction
%% (we want the citations themselves, not their line locations).
no_citation_needed(read_source_terms/2).
read_source_terms(Path, Terms) :-
    setup_call_cleanup(
        open(Path, read, Stream),
        read_terms_loop(Stream, Terms),
        close(Stream)
    ).

no_citation_needed(read_terms_loop/2).
read_terms_loop(Stream, Terms) :-
    read_term(Stream, Term, []),
    (   Term == end_of_file
    ->  Terms = []
    ;   Terms = [Term | Rest],
        read_terms_loop(Stream, Rest)
    ).


%% walk_terms_for_annotations(+Terms, +CAcc, +EAcc, -COut, -EOut)
%%
%% Walks the term list collecting cites/2 and no_citation_needed/1 facts.
%% Other terms (rule clauses, directives, etc.) are ignored.
no_citation_needed(walk_terms_for_annotations/5).
walk_terms_for_annotations([], CAcc, EAcc, CAcc, EAcc).

walk_terms_for_annotations([Term | Rest], CAccIn, EAccIn, COut, EOut) :-
    (   Term = cites(Indicator, Citations)
    ->  CAccMid = [citation(predicate(Indicator), Citations) | CAccIn],
        EAccMid = EAccIn
    ;   Term = no_citation_needed(Indicator)
    ->  CAccMid = CAccIn,
        EAccMid = [exemption(predicate(Indicator)) | EAccIn]
    ;   CAccMid = CAccIn,
        EAccMid = EAccIn
    ),
    walk_terms_for_annotations(Rest, CAccMid, EAccMid, COut, EOut).


%% ────────────────────────────────────────────────────────────────────────
%% extract_citations_grouped(+SourcePath, -CitationsByPredicate)
%%
%% Like extract_citations/3 but groups citations per-predicate.
%% Multiple cites/2 for the same predicate are aggregated into one
%% predicate_citations/2 entry. Emitters that want one summary entry
%% per predicate (rather than per-cites/2-fact) use this form.
%%
%% Exemptions are surfaced as predicate_citations(predicate(P), exempt).
%% ────────────────────────────────────────────────────────────────────────
cites(extract_citations_grouped/2, [
    cite(authority:internal(truth_flow_bpd_integration_design),
         section(layer_4_projection_preservation))
]).
extract_citations_grouped(SourcePath, Grouped) :-
    extract_citations(SourcePath, Citations, Exemptions),
    %% Build a list of (predicate, citations) pairs, aggregating duplicates
    findall(predicate_citations(predicate(P), AllCites),
            ( setof(P, distinct_pred_with_citations(Citations, P), Preds),
              member(P, Preds),
              findall(Cs, member(citation(predicate(P), Cs), Citations),
                      CitationLists),
              flatten(CitationLists, AllCites)
            ),
            CitedGroup),
    findall(predicate_citations(predicate(P), exempt),
            member(exemption(predicate(P)), Exemptions),
            ExemptGroup),
    append(CitedGroup, ExemptGroup, Grouped).

no_citation_needed(distinct_pred_with_citations/2).
distinct_pred_with_citations(Citations, P) :-
    member(citation(predicate(P), _), Citations).
