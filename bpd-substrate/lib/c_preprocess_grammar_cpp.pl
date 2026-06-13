%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% c_preprocess_grammar_cpp.pl — GCC cpp grammar in Prolog form
%%
%% This module owns the preprocessor-grammar concerns that were
%% previously smeared across c_ast.pl. Per Heath's architectural
%% directive (2026-05-17, via medayek 03:32 UTC + 03:37 UTC):
%%
%%   "Avoid future cross-concern entanglement of cpp rules with C
%%    parsing rules. A re-export would preserve exactly the entanglement
%%    we're cutting."
%%
%% MODULE NAMING: {purpose}_{specific_instance}.pl
%%   purpose: c_preprocess_grammar
%%   specific_instance: cpp (GCC's C/C++ preprocessor)
%% Future sibling modules can hold grammar for other preprocessor
%% variants (c_preprocess_grammar_mscpp.pl for Microsoft's, etc.).
%% A shared c_preprocess_grammar.pl can later hold rules common across
%% variants.
%%
%% RELATIONSHIP TO c_preprocess.pl:
%%   c_preprocess.pl is the runtime wrapper around the EXTERNAL GCC cpp.
%%   This module (c_preprocess_grammar_cpp.pl) is the Prolog grammar
%%   that, when complete, can replace the external dependency. They are
%%   different concerns: runtime wrapper vs. native grammar. They sit
%%   side-by-side until the native grammar is full enough to retire the
%%   wrapper. At that point c_preprocess.pl either retires or persists
%%   as a GCC-comparison oracle for testing.
%%
%% This is a brick in Heath's Subject Matter Guru (SMG) Quality World
%% vision: "we have a Prolog cpp that exactly matches several industry
%% COTS cpp implementations, with predicates to shift compatibility,
%% templatized on spec differences."
%%
%% AST TERMS owned by this module:
%%   c_include(Path)      — #include "path"  (quoted form)
%%   c_include_sys(Path)  — #include <path>  (system form)
%%   c_header_guard(Guard, Body) — #ifndef GUARD ... #define GUARD ... #endif
%%
%% TOKENIZER RULES owned by this module:
%%   c_line_comment//0       — `// ... \n`
%%   c_line_comment_body//0  — content within a line comment
%%   c_block_comment//0      — `/* ... */`
%%   c_block_comment_body//0 — content within a block comment
%%
%% EMIT GRAMMAR for the above AST terms is contributed to c_ast:emit//2
%% as MULTIFILE (not re-exported). c_ast.pl declares emit//2 multifile;
%% this module contributes clauses for preprocessor AST nodes. Neither
%% module imports the other's clauses; the runtime sees them as one
%% jointly-defined predicate.
%%
%% CALLER CONTRACT:
%%   - Production callers using c_include_sys etc. as AST term constructors
%%     (e.g., emit_program([c_include_sys('cuda_runtime.h'), ...], C))
%%     do NOT need to import this module. They use the multifile
%%     extension transparently via emit_program.
%%   - The c_ast.pl tokenizer privately imports the comment grammar
%%     for use in its c_ws whitespace rule until commit 3 of the
%%     refactor sequence ships (which will remove that dependency
%%     entirely once 3.1.e routes all source through c_preprocess).
%%
%% Author: metayen 2026-05-17
%% Per Heath's directive (via medayek). Pattern: conservative factoring
%% with library extraction, applied to API-surface direction.

:- module(c_preprocess_grammar_cpp, [
    c_line_comment//0,
    c_line_comment_body//0,
    c_block_comment//0,
    c_block_comment_body//0
]).

:- set_prolog_flag(double_quotes, codes).

%% Declare our contribution to c_ast's multifile emit//2.
%% This is NOT a re-export. We do not import c_ast's emit rules.
%% We contribute clauses; the runtime sees them as part of the same
%% predicate due to the multifile declaration in c_ast.pl.
:- multifile c_ast:emit//2.

:- discontiguous c_ast:emit//2.


%% ─── Emit grammar for preprocessor AST nodes ─────────────────────

c_ast:emit(c_include(Path), _Indent) -->
    "#include \"", c_ast:emit_atom(Path), "\"\n".

c_ast:emit(c_include_sys(Path), _Indent) -->
    "#include <", c_ast:emit_atom(Path), ">\n".

c_ast:emit(c_header_guard(Guard, Body), _Indent) -->
    "#ifndef ", c_ast:emit_atom(Guard), "\n",
    "#define ", c_ast:emit_atom(Guard), "\n\n",
    c_ast:emit_nodes(Body, 0),
    "\n#endif // ", c_ast:emit_atom(Guard), "\n".


%% ─── Tokenizer grammar for C/C++ comments ────────────────────────

%% Line comment: // followed by any characters up to and including \n.
%% Used by c_ast.pl's c_ws (whitespace) rule to skip comments during
%% tokenization of raw (unpreprocessed) input.
c_line_comment --> [0'/, 0'/], c_line_comment_body, [0'\n].

c_line_comment_body --> [].
c_line_comment_body --> [C], { C \= 0'\n }, c_line_comment_body.


%% Block comment: /* followed by any characters up to and including */.
%% Used by c_ast.pl's c_ws (whitespace) rule to skip comments during
%% tokenization of raw (unpreprocessed) input.
c_block_comment --> [0'/, 0'*], c_block_comment_body.

c_block_comment_body --> [0'*, 0'/].
c_block_comment_body --> [_], c_block_comment_body.
