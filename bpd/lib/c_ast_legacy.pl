%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% c_ast_legacy.pl — Deprecated v1 parsing entry points from c_ast.pl
%%
%% These predicates were identified as dead (zero external callers) by
%% medayek's static call-graph analysis on 2026-05-17 (commit 7687eb95c).
%% They are extracted here, intact and functional, per Heath's
%% "conservative factoring with library extraction" pattern:
%%
%%   When a substrate transformation makes some existing code redundant,
%%   factor the suspect code into a library rather than delete it. The
%%   library survives as (a) a fallback if our prediction was wrong,
%%   (b) documentation of what the substrate used to compensate for,
%%   (c) eventually a candidate for deletion once empirical evidence
%%   accumulates that nothing reaches it.
%%
%% The 8 predicates here are all v1 entry points superseded by v2/v3
%% parsers in c_ast.pl. They depend on internal DCG rules (parse_stmt,
%% parse_stmt_list, parse_full_expr, parse_chain, etc.) that remain in
%% c_ast.pl. Those internal rules become dead-by-association when these
%% entry points are removed from c_ast.pl, but are left in place for
%% now to avoid surprising downstream effects. A followup commit can
%% prune the dead-by-association DCG rules after empirical verification
%% that no v2/v3 path reaches them.
%%
%% USAGE:
%%
%%   :- use_module(library('c_ast')).
%%   :- use_module(library('c_ast_legacy')).
%%
%%   c_ast_legacy:c_parse_type(String, AST).
%%
%% No predicates from this module are imported into c_ast.pl. They are
%% available only to callers that explicitly load this legacy module.
%%
%% Author: metayen 2026-05-17
%% Detection by medayek (intercom 02:58 UTC + 03:01 UTC followup)
%% Pattern named by Heath ("conservative factoring with library extraction")

:- module(c_ast_legacy, [
    c_tokenize_enriched/2,    % v1, superseded by c_tokenize_enriched_v2
    c_enrich_tokens/2,        % helper for c_tokenize_enriched
    c_parse_stmts/2,          % v1, superseded by c_parse_stmts_v2 / _v3
    c_parse_stmt/2,           % v1, superseded by v2/v3 statement parser
    c_parse_type/2,           % zero callers ever found
    c_parse_tokens/2,         % zero callers ever found
    c_parse_chain/2,          % superseded by parse_chain_postfix in v2/v3
    c_parse_full_expr/2       % superseded by parse_expr in v2/v3
]).

:- use_module(c_ast).


%% ─── v1 entry points ─────────────────────────────────────────────
%% Each predicate calls a DCG rule that is still defined in c_ast.pl.
%% The DCG rules remain in c_ast.pl pending separate cleanup.

%% Enriched tokenizer (v1): classifies keywords and unambiguous operators.
%% Superseded by c_tokenize_enriched_v2 (which adds semicolon-as-own-token).
%% Reached only by these (now extracted) entry points; zero external callers.
c_tokenize_enriched(String, Tokens) :-
    c_ast:c_tokenize(String, RawTokens),
    c_enrich_tokens(RawTokens, Tokens).

%% Helper for c_tokenize_enriched (v1 enrichment rules).
%% Reclassifies id() → keyword() and punct() → operator() for unambiguous ops.
c_enrich_tokens([], []).
c_enrich_tokens([id(X)|Rest], [keyword(X)|ERest]) :-
    c_ast:c_keyword(X), !, c_enrich_tokens(Rest, ERest).
c_enrich_tokens([punct(Op)|Rest], [operator(Op)|ERest]) :-
    c_ast:c_unambiguous_op(Op), !, c_enrich_tokens(Rest, ERest).
c_enrich_tokens([T|Rest], [T|ERest]) :-
    c_enrich_tokens(Rest, ERest).

%% v1 single-statement parser entry. Note: c_ast.pl has TWO clauses
%% historically — one at line 756 and one at line 906. Both are equivalent
%% (both phrase parse_stmt). Both are extracted here as one predicate.
c_parse_stmt(String, AST) :-
    c_ast:c_tokenize(String, Tokens),
    phrase(c_ast:parse_stmt(AST), Tokens).

%% v1 multi-statement parser entry.
c_parse_stmts(String, ASTs) :-
    c_ast:c_tokenize(String, Tokens),
    phrase(c_ast:parse_stmt_list(ASTs), Tokens).

%% v1 type parser entry. Zero callers ever.
c_parse_type(String, AST) :-
    c_ast:c_tokenize(String, Tokens),
    phrase(c_ast:parse_type(AST), Tokens).

%% v1 raw-token interface. Zero callers ever.
c_parse_tokens(String, Tokens) :-
    c_ast:c_tokenize(String, Tokens).

%% v1 full-expression parser entry. Superseded by parse_expr in v2/v3.
c_parse_full_expr(String, AST) :-
    c_ast:c_tokenize(String, Tokens),
    phrase(c_ast:parse_full_expr(AST), Tokens).

%% v1 chained-expression parser entry. Superseded by parse_chain_postfix
%% in v2/v3.
c_parse_chain(String, AST) :-
    c_ast:c_tokenize(String, Tokens),
    phrase(c_ast:parse_chain(AST), Tokens).
