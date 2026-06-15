%% c_ast.pl — Prolog C AST library for BPD code generators.
%%
%% Provides a term-level representation of C language constructs
%% and a DCG-based pretty-printer that emits valid C source code.
%%
%% Design principle: generators construct AST terms using these
%% predicates. They never contain C source strings. The pretty-printer
%% is the ONLY place that knows C syntax.
%%
%% Two initial consumers:
%%   1. Compute-graph builder generator (mavchin)
%%   2. GGUF C reader generator (metayen)
%%
%% Part of the BPD ecosystem.

:- module(c_ast, [
    emit_c/2,          % emit_c(+ASTNode, -CString)
    emit_program/2,    % emit_program(+Nodes, -CString)
    emit_to_file/2,    % emit_to_file(+Nodes, +FilePath)
    ast_uses_var/2     % ast_uses_var(+ASTTerm, +VarName) — does the AST
                       %   anywhere reference c_var(VarName)?
                       %   Used by generators to derive structural facts
                       %   (e.g., "does this op need an `arg` declaration?")
                       %   from the AST itself rather than via side tables.
]).

:- discontiguous emit_expr/3.
:- discontiguous emit_for_init/3.

:- set_prolog_flag(double_quotes, codes).
:- discontiguous emit_type//1.
:- discontiguous emit_stmt//2.
:- discontiguous c_punct//1.
:- discontiguous c_str_chars//1.
:- discontiguous parse_expr//1.
:- discontiguous parse_stmt//1.
:- discontiguous parse_stmt_v2//1.
:- discontiguous parse_chain_postfix//2.

%% emit//2 is jointly defined with c_preprocess_grammar_cpp.pl, which
%% contributes the clauses for preprocessor AST nodes (c_include,
%% c_include_sys, c_header_guard). Declared multifile here so the
%% runtime sees both modules' clauses as a single predicate. NOT a
%% re-export — each module owns its own clauses. See:
%%   lib/c_preprocess_grammar_cpp.pl
:- multifile emit//2.
:- discontiguous emit//2.

%% Load c_preprocess_grammar_cpp WITHOUT importing any of its symbols.
%% The empty import list means none of its predicates enter c_ast's
%% namespace, but the module IS loaded — which is what activates its
%% multifile clauses for c_ast:emit//2 (the emit rules for c_include,
%% c_include_sys, c_header_guard).
%%
%% Historical context: pre-step-3.1.purify, this directive imported
%% c_line_comment//0 and c_block_comment//0 so c_ws could call them
%% during raw-source tokenization. Post-3.1.e routes all source through
%% cpp (which strips comments), so c_ws no longer needs them. The
%% imports were dropped (3.1.purify), but the load remains because
%% emit_program(Program, _) where Program contains c_include_sys/1 etc.
%% needs the multifile clauses to be registered. Substrate-honest:
%% loading the module is structural (the multifile partition requires
%% it); importing its symbols was incidental (only c_ws used them).
:- use_module(c_preprocess_grammar_cpp, []).

%% ═══════════════════════════════════════════════════════════════
%% AST TERM DEFINITIONS (documentation — not enforced by types)
%% ═══════════════════════════════════════════════════════════════
%%
%% TYPES:
%%   c_type(int)
%%   c_type(float)
%%   c_type(void)
%%   c_type(ptr(BaseType))          — pointer to BaseType
%%   c_type(const(BaseType))        — const qualifier
%%   c_type(struct(Name))           — struct type
%%   c_type(array(BaseType, Size))  — array type
%%   c_type(named(Atom))            — typedef'd name like ggml_tensor
%%
%% EXPRESSIONS:
%%   c_int(Value)                   — integer literal
%%   c_float(Value)                 — float literal
%%   c_string(Value)                — string literal
%%   c_var(Name)                    — variable reference
%%   c_member(Expr, Field)          — struct.field
%%   c_arrow(Expr, Field)           — ptr->field
%%   c_index(Expr, IndexExpr)       — array[index]
%%   c_call(FuncName, ArgExprs)     — function call
%%   c_cast(Type, Expr)             — (type)expr
%%   c_binop(Op, Left, Right)       — left op right
%%   c_unop(Op, Expr)               — op expr  (prefix)
%%   c_sizeof(TypeOrExpr)           — sizeof(x)
%%   c_ternary(Cond, Then, Else)    — cond ? then : else
%%   c_nullptr                      — nullptr / NULL
%%   c_addr(Expr)                   — &expr (address-of)
%%   c_deref(Expr)                  — *expr (dereference)
%%   c_hex(Value)                   — hex literal (0xFF)
%%   c_ptr_arith(Base, Offset)      — base + offset (pointer arithmetic)
%%   c_fma(A, B, C)                 — __fmaf_rn(a, b, c) (CUDA FMA intrinsic)
%%   c_index2d(Arr, Row, Col)       — arr[row][col] (2D array index)
%%
%% STATEMENTS:
%%   c_expr_stmt(Expr)              — expr;
%%   c_assign(LHS, RHS)             — lhs = rhs;
%%   c_decl(Type, Name)             — type name;
%%   c_decl_init(Type, Name, Expr)  — type name = expr;
%%   c_return(Expr)                 — return expr;
%%   c_if(Cond, Then, Else)         — if (cond) { then } else { else }
%%   c_if(Cond, Then)               — if (cond) { then }
%%   c_for(Init, Cond, Step, Body)  — for (init; cond; step) { body }
%%   c_for_step(Init, Cond, StepStmt, Body) — for with compound-assign step
%%                                   e.g. c_compound_assign('+=', i, c_int(128))
%%   c_compound_assign(Op, LHS, RHS) — lhs op= rhs;  (e.g. i += 128, s >>= 1)
%%   c_block(Stmts)                 — { stmts }
%%   c_comment(Text)                — // text
%%   c_blank                        — empty line
%%   c_break                        — break;
%%   c_continue                     — continue;
%%
%% TOP-LEVEL:
%%   c_include(Path)                — #include "path"
%%   c_include_sys(Path)            — #include <path>
%%   c_func(RetType, Name, Params, Body)
%%   c_func(Qualifiers, RetType, Name, Params, Body) — with static/inline/etc
%%   c_struct_def(Name, Fields)     — struct Name { type1 field1; ... };
%%   c_typedef(Type, Name)          — typedef type name;
%%   c_typedef_struct(Name, Fields) — typedef struct { ... } Name;
%%   c_enum_def(Name, Values)       — enum Name { V1, V2, ... };
%%   c_header_guard(Guard, Body)    — #ifndef GUARD ... #endif

%% ═══════════════════════════════════════════════════════════════
%% PRETTY-PRINTER (DCG)
%% ═══════════════════════════════════════════════════════════════

%% Entry points
emit_c(Node, String) :-
    phrase(emit(Node, 0), Codes),
    atom_codes(String, Codes).

emit_program(Nodes, String) :-
    phrase(emit_nodes(Nodes, 0), Codes),
    atom_codes(String, Codes).

emit_to_file(Nodes, Path) :-
    emit_program(Nodes, String),
    open(Path, write, Stream),
    write(Stream, String),
    close(Stream).


%% ─── AST introspection ───

%% ast_uses_var(+ASTTerm, +VarName)
%%
%% Succeeds iff the AST term anywhere references c_var(VarName).
%% Walks the term recursively through compound subterms and lists.
%%
%% Used by generators to derive structural facts from the AST itself
%% rather than via parallel side tables. Example: a reduction template
%% can ask "does AccumStmt or FinalizeExpr use c_var(arg)?" and emit
%% the `int arg = 0;` declaration only when the answer is yes.
%%
%% The same query generalizes: ast_uses_var(Body, sum) tells you if
%% `sum` is referenced; ast_uses_var(Body, threadIdx) tells you if the
%% block uses thread coordinates; etc. Substrate-honest: the AST
%% already encodes the structural fact; we just need to ask.
ast_uses_var(c_var(V), V) :- !.
ast_uses_var(Term, V) :-
    compound(Term),
    Term \= c_var(_),
    Term =.. [_|Args],
    member(Arg, Args),
    ast_uses_var_member(Arg, V).

%% Helper: walk into list args as well as compound args.
ast_uses_var_member([H|T], V) :- !,
    ( ast_uses_var(H, V)
    ; ast_uses_var_member(T, V)
    ).
ast_uses_var_member(Term, V) :- ast_uses_var(Term, V).


%% ─── Types ───

emit_type(c_type(int)) --> "int".
emit_type(c_type(float)) --> "float".
emit_type(c_type(double)) --> "double".
emit_type(c_type(void)) --> "void".
emit_type(c_type(bool)) --> "bool".
emit_type(c_type(char)) --> "char".
emit_type(c_type(int32_t)) --> "int32_t".
emit_type(c_type(int64_t)) --> "int64_t".
emit_type(c_type(uint8_t)) --> "uint8_t".
emit_type(c_type(uint16_t)) --> "uint16_t".
emit_type(c_type(uint32_t)) --> "uint32_t".
emit_type(c_type(uint64_t)) --> "uint64_t".
emit_type(c_type(int8_t)) --> "int8_t".
emit_type(c_type(int16_t)) --> "int16_t".
emit_type(c_type(size_t)) --> "size_t".
emit_type(c_type(ptr(Base))) --> emit_type(Base), " *".
emit_type(c_type(const(Base))) --> "const ", emit_type(Base).
emit_type(c_type(const_ptr(Base))) --> "const ", emit_type(Base), " *".
emit_type(c_type(struct(Name))) --> "struct ", emit_atom(Name).
emit_type(c_type(named(Name))) --> emit_atom(Name).
emit_type(c_type(auto)) --> "auto".

%% ─── Expressions ───

emit_expr(c_int(V)) --> emit_number(V).
%% c_float_lit(Value, Suffix) - canonical float literal node. Models a C float literal
%% as value + optional suffix (none | f | l). Emits value via format_float_minimal
%% (precision-preserving) + suffix char. c_float/c_float_f are aliases. No raw text.
emit_expr(c_float_lit(V, Suffix)) --> emit_float(V), float_suffix_chars(Suffix).
float_suffix_chars(none) --> [].
float_suffix_chars('') --> [].
float_suffix_chars(f) --> "f".
float_suffix_chars('F') --> "F".
float_suffix_chars(l) --> "l".
float_suffix_chars('L') --> "L".
emit_expr(c_float(V)) --> emit_expr(c_float_lit(V, none)).
emit_expr(c_float_f(V)) --> emit_expr(c_float_lit(V, f)).
emit_expr(c_string(V)) --> "\"", emit_atom(V), "\"".
emit_expr(c_var(Name)) --> emit_atom(Name).
emit_expr(c_nullptr) --> "nullptr".
emit_expr(c_null) --> "NULL".
emit_expr(c_true) --> "true".
emit_expr(c_false) --> "false".

emit_expr(c_member(Expr, Field)) -->
    emit_expr(Expr), ".", emit_atom(Field).
emit_expr(c_arrow(Expr, Field)) -->
    emit_expr(Expr), "->", emit_atom(Field).
emit_expr(c_index(Expr, Idx)) -->
    emit_expr(Expr), "[", emit_expr(Idx), "]".

%% N-dimensional row-major index expression.
%% c_nd_index([Base, Dim1, Idx1, Dim2, Idx2, ...])
%% Emits: ((Base * Dim1 + Idx1) * Dim2 + Idx2) * ...
%% Left-fold: acc = Base, then acc = acc * DimK + IdxK for each pair.
%% Wraps each intermediate in c_paren for correct C precedence.
emit_expr(c_nd_index(Parts)) -->
    { nd_index_expr(Parts, Expr) },
    emit_expr(Expr).

%% Call expression. The function position can be:
%%   - an atom (plain function call: foo(args))
%%   - a c_member chain (method call: obj.method(args))
%%   - a c_qualified (namespaced call: ns::func(args))
%%   - any other expression (less common: function pointer call)
%% Dispatch by shape so parse and emit stay symmetric.
emit_expr(c_call(Func, Args)) -->
    { atom(Func) }, !,
    emit_atom(Func), "(", emit_expr_list(Args), ")".
emit_expr(c_call(Func, Args)) -->
    emit_expr(Func), "(", emit_expr_list(Args), ")".

%% C cast: (Type)Expr. The operand isn't wrapped in extra parens
%% unless the parser explicitly captured them (as c_paren); preserves
%% upstream's minimal-paren convention.
emit_expr(c_cast(Type, Expr)) -->
    "(", emit_type(Type), ")", emit_expr(Expr).

%% Binop: parenthesize operands that are THEMSELVES compound expressions
%% (binop/ternary/cast), so precedence is preserved. Without this,
%% c_binop(/, c_binop(+, n, 255), 256) emitted "n + 255 / 256" (= n + 0)
%% instead of "(n + 255) / 256". Simple operands (vars, ints, calls,
%% already-parenthesized) pass through unwrapped — preserving the
%% upstream minimal-paren convention.
emit_expr(c_binop(Op, L, R)) -->
    emit_binop_operand(L), " ", emit_atom(Op), " ", emit_binop_operand(R).
emit_expr(c_unop(Op, E)) -->
    emit_atom(Op), emit_binop_operand(E).

%% emit_binop_operand: wrap compound operands in parens; pass simple ones through.
emit_binop_operand(c_binop(Op, L, R)) -->
    "(", emit_expr(c_binop(Op, L, R)), ")".
emit_binop_operand(c_ternary(C, T, E)) -->
    "(", emit_expr(c_ternary(C, T, E)), ")".
emit_binop_operand(c_cast(Ty, E)) -->
    "(", emit_expr(c_cast(Ty, E)), ")".
emit_binop_operand(E) -->
    { \+ compound_binop_operand(E) },
    emit_expr(E).

compound_binop_operand(c_binop(_, _, _)).
compound_binop_operand(c_ternary(_, _, _)).
compound_binop_operand(c_cast(_, _)).

emit_expr(c_sizeof(E)) -->
    "sizeof(", emit_expr(E), ")".

emit_expr(c_ternary(Cond, Then, Else)) -->
    emit_expr(Cond), " ? ", emit_expr(Then), " : ", emit_expr(Else).

emit_expr(c_paren(E)) -->
    "(", emit_expr(E), ")".

%% Braced initializer list: { e1, e2, ... }
%% Symmetric with parse_atom_expr(c_init_list(...)).
emit_expr(c_init_list(Elements)) -->
    "{", emit_expr_list(Elements), "}".

emit_expr(c_addr(E)) -->
    "&", emit_expr(E).
emit_expr(c_deref(E)) -->
    "*", emit_expr(E).

%% c_ptr_arith(Base, Offset) — pointer arithmetic expression: base + offset
emit_expr(c_ptr_arith(Base, Offset)) -->
    emit_expr(Base), " + ", emit_expr(Offset).

%% c_fma(A, B, C) — CUDA fused multiply-add intrinsic __fmaf_rn(a, b, c)
emit_expr(c_fma(A, B, C)) -->
    "__fmaf_rn(", emit_expr(A), ", ", emit_expr(B), ", ", emit_expr(C), ")".

%% c_index2d(Arr, Row, Col) — 2D array indexing: arr[row][col]
emit_expr(c_index2d(Arr, Row, Col)) -->
    emit_expr(Arr), "[", emit_expr(Row), "][", emit_expr(Col), "]".

emit_expr(c_hex(V)) -->
    { format(atom(A), "0x~16r", [V]), atom_codes(A, Cs) }, Cs.

%% ─── Statements ───

emit_stmt(c_expr_stmt(Expr), Indent) -->
    emit_indent(Indent), emit_expr(Expr), ";\n".

emit_stmt(c_assign(LHS, RHS), Indent) -->
    emit_indent(Indent), emit_expr(LHS), " = ", emit_expr(RHS), ";\n".

emit_stmt(c_decl(Type, Name), Indent) -->
    emit_indent(Indent), emit_type(Type), " ", emit_atom(Name), ";\n".

emit_stmt(c_decl_init(Type, Name, Expr), Indent) -->
    emit_indent(Indent), emit_type(Type), " ", emit_atom(Name),
    " = ", emit_expr(Expr), ";\n".

emit_stmt(c_return(Expr), Indent) -->
    emit_indent(Indent), "return ", emit_expr(Expr), ";\n".

emit_stmt(c_comment(Text), Indent) -->
    emit_indent(Indent), "// ", emit_atom(Text), "\n".

emit_stmt(c_blank, _Indent) --> "\n".

emit_stmt(c_break, Indent) -->
    emit_indent(Indent), "break;\n".

emit_stmt(c_continue, Indent) -->
    emit_indent(Indent), "continue;\n".

emit_stmt(c_if(Cond, Then), Indent) -->
    { I1 is Indent + 1 },
    emit_indent(Indent), "if (", emit_expr(Cond), ") {\n",
    emit_stmts(Then, I1),
    emit_indent(Indent), "}
".

%% else-if: when the else body is a single c_if statement, render as
%% `} else if (...) { ... }` rather than `} else { if (...) { ... } }`.
%% Matches upstream's idiomatic else-if chains without changing the AST
%% representation (which uniformly uses c_if(Cond, Then, [InnerIf]) for
%% the chain shape).
emit_stmt(c_if(Cond, Then, [InnerIf]), Indent) -->
    { I1 is Indent + 1,
      (InnerIf = c_if(_, _) ; InnerIf = c_if(_, _, _)) },
    !,
    emit_indent(Indent), "if (", emit_expr(Cond), ") {\n",
    emit_stmts(Then, I1),
    emit_indent(Indent), "} else ",
    emit_stmt_inline(InnerIf, Indent).

%% Standard if-else with braced else body.
emit_stmt(c_if(Cond, Then, Else), Indent) -->
    { I1 is Indent + 1 },
    emit_indent(Indent), "if (", emit_expr(Cond), ") {\n",
    emit_stmts(Then, I1),
    emit_indent(Indent), "} else {\n",
    emit_stmts(Else, I1),
    emit_indent(Indent), "}
".

%% Inline statement emission: same as emit_stmt but without leading
%% indent (used after `} else ` where the leading `if` should be
%% on the same line).
emit_stmt_inline(c_if(Cond, Then), Indent) -->
    { I1 is Indent + 1 },
    "if (", emit_expr(Cond), ") {\n",
    emit_stmts(Then, I1),
    emit_indent(Indent), "}\n".
emit_stmt_inline(c_if(Cond, Then, [InnerIf]), Indent) -->
    { I1 is Indent + 1,
      (InnerIf = c_if(_, _) ; InnerIf = c_if(_, _, _)) },
    !,
    "if (", emit_expr(Cond), ") {\n",
    emit_stmts(Then, I1),
    emit_indent(Indent), "} else ",
    emit_stmt_inline(InnerIf, Indent).
emit_stmt_inline(c_if(Cond, Then, Else), Indent) -->
    { I1 is Indent + 1 },
    "if (", emit_expr(Cond), ") {\n",
    emit_stmts(Then, I1),
    emit_indent(Indent), "} else {\n",
    emit_stmts(Else, I1),
    emit_indent(Indent), "}\n".

emit_stmt(c_for(Init, Cond, Step, Body), Indent) -->
    { I1 is Indent + 1 },
    emit_indent(Indent), "for (",
    emit_for_init(Init), "; ", emit_expr(Cond), "; ", emit_expr(Step),
    ") {\n",
    emit_stmts(Body, I1),
    emit_indent(Indent), "}
".

%% For-init can be a declaration or an expression
emit_for_init(c_decl_init(Type, Name, Expr)) -->
    emit_type(Type), " ", emit_atom(Name), " = ", emit_expr(Expr).
emit_for_init(Expr) --> emit_expr(Expr).

emit_stmt(c_block(Stmts), Indent) -->
    { I1 is Indent + 1 },
    emit_indent(Indent), "{\n",
    emit_stmts(Stmts, I1),
    emit_indent(Indent), "}
".

%% ─── Top-level ───

%% emit(c_include(Path), _) and emit(c_include_sys(Path), _) — moved to
%% c_preprocess_grammar_cpp.pl as multifile contributions to c_ast:emit//2
%% (commit 2026-05-17 step 3.1.factor). The clauses there qualify as
%% `c_ast:emit(...)` so the runtime sees them as part of this predicate.

emit(c_func(RetType, Name, Params, Body), Indent) -->
    { I1 is Indent + 1 },
    emit_type(RetType), " ", emit_atom(Name), "(",
    emit_param_list(Params), ") {\n",
    emit_stmts(Body, I1),
    "}
".

emit(c_func(Quals, RetType, Name, Params, Body), Indent) -->
    { I1 is Indent + 1 },
    emit_qualifiers(Quals),
    emit_type(RetType), " ", emit_atom(Name), "(",
    emit_param_list(Params), ") {\n",
    emit_stmts(Body, I1),
    "}
".

emit(Stmt, Indent) --> emit_stmt(Stmt, Indent).

%% ─── Struct/typedef/enum/header guard ───

emit(c_struct_def(Name, Fields), Indent) -->
    { I1 is Indent + 1 },
    emit_indent(Indent), "struct ", emit_atom(Name), " {\n",
    emit_struct_fields(Fields, I1),
    emit_indent(Indent), "};\n".

emit(c_typedef(Type, Name), Indent) -->
    emit_indent(Indent), "typedef ", emit_type(Type), " ", emit_atom(Name), ";\n".

emit(c_typedef_struct(Name, Fields), Indent) -->
    { I1 is Indent + 1 },
    emit_indent(Indent), "typedef struct {\n",
    emit_struct_fields(Fields, I1),
    emit_indent(Indent), "} ", emit_atom(Name), ";\n".

emit(c_enum_def(Name, Values), Indent) -->
    emit_indent(Indent), "enum ", emit_atom(Name), " {\n",
    emit_enum_values(Values, Indent),
    emit_indent(Indent), "};\n".

%% emit(c_header_guard(Guard, Body), _) — moved to
%% c_preprocess_grammar_cpp.pl as multifile contribution to c_ast:emit//2.

%% ─── Switch/case ───

emit_stmt(c_switch(Expr, Cases), Indent) -->
    { I1 is Indent + 1, I2 is Indent + 2 },
    emit_indent(Indent), "switch (", emit_expr(Expr), ") {\n",
    emit_switch_cases(Cases, I1, I2),
    emit_indent(Indent), "}
".

emit_switch_cases([], _, _) --> [].
emit_switch_cases([c_case(Val, Stmts)|Rest], I1, I2) -->
    emit_indent(I1), "case ", emit_expr(Val), ":\n",
    emit_stmts(Stmts, I2),
    emit_switch_cases(Rest, I1, I2).
emit_switch_cases([c_default(Stmts)|Rest], I1, I2) -->
    emit_indent(I1), "default:\n",
    emit_stmts(Stmts, I2),
    emit_switch_cases(Rest, I1, I2).

%% ─── Struct field helpers ───

emit_struct_fields([], _) --> [].
emit_struct_fields([field(Type, Name)|Rest], Indent) -->
    emit_indent(Indent), emit_type(Type), " ", emit_atom(Name), ";\n",
    emit_struct_fields(Rest, Indent).

%% ─── Enum value helpers ───

emit_enum_values([], _) --> [].
emit_enum_values([V], Indent) -->
    { I1 is Indent + 1 },
    emit_indent(I1), emit_atom(V), "\n".
emit_enum_values([V|Vs], Indent) -->
    { I1 is Indent + 1 },
    emit_indent(I1), emit_atom(V), ",\n",
    emit_enum_values(Vs, Indent).

%% ─── Helpers ───

emit_nodes([], _) --> [].
emit_nodes([H|T], Indent) --> emit(H, Indent), emit_nodes(T, Indent).

emit_stmts([], _) --> [].
emit_stmts([H|T], Indent) -->
    
    emit_stmt(H, Indent),
    emit_stmts(T, Indent).

emit_expr_list([]) --> [].
emit_expr_list([E]) --> emit_expr(E).
emit_expr_list([E|Es]) --> emit_expr(E), ", ", emit_expr_list(Es).

emit_param_list([]) --> [].
emit_param_list([param(Type, Name)]) -->
    emit_type(Type), " ", emit_atom(Name).
emit_param_list([param(Type, Name)|Ps]) -->
    emit_type(Type), " ", emit_atom(Name), ", ", emit_param_list(Ps).

emit_indent(0) --> [].
emit_indent(N) --> { N > 0, N1 is N - 1 }, "    ", emit_indent(N1).

emit_atom(A) --> { atom_codes(A, Cs) }, Cs.
emit_number(N) --> { N < 0, Abs is abs(N), number_codes(Abs, Cs) }, "-", Cs.
emit_number(N) --> { N >= 0, number_codes(N, Cs) }, Cs.
emit_float(F) --> { format_float_minimal(F, A), atom_codes(A, Cs) }, Cs.

%% Render a float in the most-compact form that round-trips to the
%% same value. Strategy:
%%   1. Try ~f with 1 decimal (8.0, 100.0). If parsing it back gives
%%      the same value, use it.
%%   2. Otherwise use Prolog's default write_term which produces a
%%      precision-preserving representation.
%%
%% Examples:
%%   8.0                 → '8.0'
%%   0.1                 → '0.1'
%%   0.5773502691896257  → '0.5773502691896257'
%%   1.0e-5              → '1.0e-5'
%%   78.38367176906169   → '78.38367176906169'
format_float_minimal(F, Atom) :-
    %% Try the short form first; only use it if round-trip preserves value.
    short_float_form(F, ShortAtom),
    atom_number(ShortAtom, Reparsed),
    Reparsed =:= F,
    !,
    Atom = ShortAtom.
format_float_minimal(F, Atom) :-
    %% Fall back to Prolog's default float printing, which preserves
    %% precision. Convert any 'e+' to 'e' and 'e0X' to 'eX' for
    %% upstream-style scientific notation if needed.
    format(atom(Raw), "~w", [F]),
    atom_codes(Raw, Codes),
    fix_scientific_format(Codes, FixedCodes),
    atom_codes(Atom, FixedCodes).

%% Short form: format with ~f and strip trailing zeros after decimal.
short_float_form(F, Atom) :-
    format(atom(Raw), "~f", [F]),
    atom_codes(Raw, Codes),
    strip_trailing_zeros(Codes, Stripped),
    atom_codes(Atom, Stripped).

%% Strip zeros from the end up to (but not past) the decimal point + 1 digit.
strip_trailing_zeros(Codes, Stripped) :-
    reverse(Codes, Rev),
    strip_trailing_zeros_rev(Rev, RevStripped),
    reverse(RevStripped, Stripped).

strip_trailing_zeros_rev([0'0, 0'., D | Rest], [0'0, 0'., D | Rest]) :- !.
strip_trailing_zeros_rev([0'0 | Rest], Stripped) :- !,
    strip_trailing_zeros_rev(Rest, Stripped).
strip_trailing_zeros_rev(Codes, Codes).

%% Normalize scientific notation: 1.0e-05 → 1e-5 (upstream style).
%% Also handles 1.0e+05 → 1e5.
fix_scientific_format(Codes, Out) :-
    %% Find 'e' position
    ( append(Pre, [0'e | ExpPart], Codes)
    -> %% Strip ".0" or trailing zeros before e in mantissa
       strip_trailing_zeros_in_mantissa(Pre, FixedPre),
       %% Strip + and leading zeros in exponent
       fix_exponent(ExpPart, FixedExp),
       append(FixedPre, [0'e | FixedExp], Out)
    ;  Out = Codes
    ).

%% Strip ".0" suffix from mantissa (1.0 → 1) for scientific form.
strip_trailing_zeros_in_mantissa(Pre, Fixed) :-
    reverse(Pre, RevPre),
    ( RevPre = [0'0, 0'. | Rest]
    -> reverse(Rest, Fixed)
    ;  %% Keep stripping zeros after decimal
       strip_trailing_zeros_rev(RevPre, RevFixed),
       reverse(RevFixed, Fixed)
    ).

%% Fix exponent: strip leading + and leading zeros (but keep one digit).
fix_exponent([0'+ | Rest], Fixed) :- !, strip_exp_leading_zeros(Rest, Fixed).
fix_exponent([0'- | Rest], [0'- | Fixed]) :- !, strip_exp_leading_zeros(Rest, Fixed).
fix_exponent(Codes, Fixed) :- strip_exp_leading_zeros(Codes, Fixed).

strip_exp_leading_zeros([0'0, D | Rest], Fixed) :-
    D >= 0'0, D =< 0'9, !,
    strip_exp_leading_zeros([D | Rest], Fixed).
strip_exp_leading_zeros(Codes, Codes).

%% ─── Qualifiers ───
emit_qualifiers([]) --> [].
emit_qualifiers([Q|Qs]) --> emit_atom(Q), " ", emit_qualifiers(Qs).

%% ═══════════════════════════════════════════════════════════════
%% C PARSER (DCG-based, operates on token lists)
%% ═══════════════════════════════════════════════════════════════

%% Step 1: Tokenize a C string into a token list
c_tokenize(String, Tokens) :-
    atom_codes(String, Codes),
    phrase(c_lex(Tokens), Codes).

%% Lexer: character codes → token list
c_lex([]) --> c_ws.
c_lex([]) --> [].
c_lex(Tokens) --> c_ws, c_lex(Tokens).
c_lex([T|Ts]) --> c_token(T), c_lex(Ts).

c_ws --> [0' ], c_ws_rest.    % space
c_ws --> [0'\t], c_ws_rest.   % tab
c_ws --> [0'\n], c_ws_rest.   % newline
%% Comment-handling alternatives removed in step 3.1.purify. Source
%% always passes through cpp first (step 3.1.e); comments are gone by
%% the time the tokenizer runs. If a future caller needs raw-source
%% tokenization, load c_preprocess_grammar_cpp explicitly.
c_ws_rest --> [].
c_ws_rest --> c_ws.

%% c_line_comment//0 and c_block_comment//0 (with their _body helpers)
%% have moved to bpd/lib/c_preprocess_grammar_cpp.pl. They are imported
%% privately at the top of this file so c_ws (whitespace handling) can
%% still call them for the raw-source tokenization path. After step
%% 3.1.e routes all source through preprocess_file_segment, these
%% callsites in c_ws will become unreached and the import can be dropped
%% (step 3.1.purify).

c_token(punct(P)) --> c_punct(P).
%% Float literal: digits . digits (optional exponent) (optional suffix)
%% e.g. 1.0, 1.0f, 1.5e-3, 8.0f. Tries first because float is more
%% specific than an int followed by punct('.').
%%
%% The token carries (Value, Suffix, OriginalText) so the emit side
%% can reproduce the source's exact textual form (preserving scientific
%% notation, precision, etc.) for round-trip purposes.
c_token(floatnum(F, Suffix, RawAtom)) -->
    c_digits(IntCs),
    [0'.],
    c_digits(FracCs),
    c_float_exponent(ExpCs),
    c_float_suffix_chars(SuffixCs),
    { append(IntCs, [0'.|FracCs], A0),
      append(A0, ExpCs, A),
      number_codes(F, A),
      atom_codes(Suffix, SuffixCs),
      append(A, SuffixCs, RawCs),
      atom_codes(RawAtom, RawCs) }.
%% Scientific without decimal point: 1e-5, 2E10 — also float.
c_token(floatnum(F, Suffix, RawAtom)) -->
    c_digits(IntCs),
    [E], { (E = 0'e ; E = 0'E) },
    c_signed_digits(ExpDigCs),
    c_float_suffix_chars(SuffixCs),
    { append(IntCs, [E|ExpDigCs], A),
      number_codes(F, A),
      atom_codes(Suffix, SuffixCs),
      append(A, SuffixCs, RawCs),
      atom_codes(RawAtom, RawCs) }.
c_token(num(N)) --> c_digits(Cs), { number_codes(N, Cs) }.
c_token(str(S)) --> [0'"], c_str_chars(Cs), [0'"], { atom_codes(S, Cs) }.
c_token(id(A)) --> c_id_chars([C|Cs]), { atom_codes(A, [C|Cs]) }.

%% Optional float exponent: e+5, E-3, e10, or nothing.
c_float_exponent([E, S | DigCs]) -->
    [E], { (E = 0'e ; E = 0'E) },
    [S], { (S = 0'+ ; S = 0'-) },
    c_digits(DigCs).
c_float_exponent([E | DigCs]) -->
    [E], { (E = 0'e ; E = 0'E) },
    c_digits(DigCs).
c_float_exponent([]) --> [].

%% Optional float suffix: f, F, l, L, or nothing.
c_float_suffix_chars([C]) --> [C], { member(C, [0'f, 0'F, 0'l, 0'L]) }.
c_float_suffix_chars([]) --> [].

%% Signed digits for scientific notation exponent.
c_signed_digits([S | Cs]) --> [S], { (S = 0'+ ; S = 0'-) }, c_digits(Cs).
c_signed_digits(Cs) --> c_digits(Cs).

c_punct('!=') --> [0'!, 0'=].
c_punct('==') --> [0'=, 0'=].
c_punct('->') --> [0'-, 0'>].
c_punct('++') --> [0'+, 0'+].
c_punct('--') --> [0'-, 0'-].
c_punct('<=') --> [0'<, 0'=].
c_punct('>=') --> [0'>, 0'=].
c_punct('&&') --> [0'&, 0'&].
c_punct('||') --> [0'|, 0'|].
%% Compound assignment operators — multi-char so they must come
%% before single-char alternatives in the c_punct alternatives.
c_punct('+=') --> [0'+, 0'=].
c_punct('-=') --> [0'-, 0'=].
c_punct('*=') --> [0'*, 0'=].
c_punct('/=') --> [0'/, 0'=].
c_punct('%=') --> [0'%, 0'=].
%% C++ scope resolution operator
c_punct('::') --> [0':, 0':].
c_punct('(') --> [0'(].
c_punct(')') --> [0')].
c_punct('{') --> [0'{].
c_punct('}') --> [0'}].
c_punct('[') --> [0'[].
c_punct(']') --> [0']].
c_punct(';') --> [0';].
c_punct(',') --> [0',].
c_punct('*') --> [0'*].
c_punct('&') --> [0'&].
c_punct('.') --> [0'.].
c_punct('=') --> [0'=].
c_punct('<') --> [0'<].
c_punct('>') --> [0'>].
c_punct('+') --> [0'+].
c_punct('-') --> [0'-].
c_punct('/') --> [0'/].
c_punct('!') --> [0'!].
c_punct('%') --> [0'%].
c_punct(':') --> [0':].
c_punct('?') --> [0'?].
c_punct('#') --> [0'#].

c_digits([C|Cs]) --> [C], { C >= 0'0, C =< 0'9 }, c_digits_rest(Cs).
c_digits_rest([C|Cs]) --> [C], { C >= 0'0, C =< 0'9 }, c_digits_rest(Cs).
c_digits_rest([]) --> [].

c_id_chars([C|Cs]) --> [C], { c_id_start(C) }, c_id_rest(Cs).
c_id_rest([C|Cs]) --> [C], { c_id_cont(C) }, c_id_rest(Cs).
c_id_rest([]) --> [].

c_id_start(C) :- C >= 0'a, C =< 0'z.
c_id_start(C) :- C >= 0'A, C =< 0'Z.
c_id_start(0'_).

c_id_cont(C) :- c_id_start(C).
c_id_cont(C) :- C >= 0'0, C =< 0'9.

c_str_chars([]) --> [].
c_str_chars([0'\\, C|Cs]) --> [0'\\, C], c_str_chars(Cs).
c_str_chars([C|Cs]) --> [C], { C \= 0'" }, c_str_chars(Cs).

%% Step 2: Parse token list into AST terms

%% Types (token-level parsing)
parse_type(Type) --> parse_base_type(Base), parse_type_suffix(Base, Type).

parse_base_type(c_type(const(c_type(T)))) -->
    [id(const)], [id(T)], { known_type(T) }.
parse_base_type(c_type(T)) -->
    [id(T)], { known_type(T) }.

parse_type_suffix(Base, c_type(ptr(Base))) --> [punct('*')].
parse_type_suffix(Base, c_type(const_ptr(Inner))) -->
    { Base = c_type(const(Inner)) }, [punct('*')].
parse_type_suffix(Base, Base) --> [].

known_type(int).
known_type(float).
known_type(double).
known_type(void).
known_type(char).
known_type(bool).
known_type(int8_t).
known_type(int16_t).
known_type(int32_t).
known_type(int64_t).
known_type(uint8_t).
known_type(uint16_t).
known_type(uint32_t).
known_type(uint64_t).
known_type(size_t).

%% Expressions (token-level)
parse_expr(c_int(N)) --> [num(N)].
parse_expr(c_string(S)) --> [str(S)].
parse_expr(c_var(Name)) --> [id(Name)].
parse_expr(c_nullptr) --> [id(nullptr)].
parse_expr(c_null) --> [id('NULL')].

parse_expr(c_call(Func, Args)) -->
    [id(Func), punct('(')], parse_arg_list(Args), [punct(')')].

parse_expr(c_member(Expr, Field)) -->
    parse_primary(Expr), [punct('.')], [id(Field)].
parse_expr(c_arrow(Expr, Field)) -->
    parse_primary(Expr), [punct('->')], [id(Field)].
parse_expr(c_index(Expr, Idx)) -->
    parse_primary(Expr), [punct('[')], parse_expr(Idx), [punct(']')].

parse_expr(c_binop(Op, L, R)) -->
    parse_primary(L), [punct(Op)], parse_primary(R),
    { member(Op, ['!=', '==', '<', '>', '<=', '>=', '+', '-', '/', '*']) }.

parse_primary(c_int(N)) --> [num(N)].
parse_primary(c_string(S)) --> [str(S)].
parse_primary(c_var(Name)) --> [id(Name)].
parse_primary(c_nullptr) --> [id(nullptr)].
parse_primary(c_null) --> [id('NULL')].
parse_primary(c_call(F, Args)) -->
    [id(F), punct('(')], parse_arg_list(Args), [punct(')')].

parse_arg_list([]) --> [].
parse_arg_list([E]) --> parse_expr(E).
parse_arg_list([E|Es]) --> parse_expr(E), [punct(',')], parse_arg_list(Es).

%% Top-level: parse a C string into an AST term
c_parse_expr(String, AST) :-
    c_tokenize(String, Tokens),
    phrase(parse_expr(AST), Tokens).

%% c_parse_type/2 — extracted to c_ast_legacy.pl (2026-05-17, commit pending)
%% Zero callers ever found via static analysis (medayek).

%% Parse for-loop from tokens
parse_stmt(c_for(Init, Cond, Step, Body)) -->
    [id(for), punct('(')],
    parse_for_init(Init), [punct(';')],
    parse_expr(Cond), [punct(';')],
    parse_expr(Step),
    [punct(')')],
    parse_stmt_or_block(Body).

parse_for_init(c_decl_init(Type, Name, Expr)) -->
    parse_type(Type), [id(Name), punct('=')], parse_expr(Expr).

parse_stmt_or_block(Stmts) --> [punct('{')], parse_stmt_list(Stmts), [punct('}')].
parse_stmt_or_block([Stmt]) --> parse_stmt(Stmt).

parse_stmt(c_expr_stmt(Expr)) --> parse_expr(Expr), [punct(';')].

parse_stmt_list([]) --> [].
parse_stmt_list([S|Ss]) --> parse_stmt(S), parse_stmt_list(Ss).

%% Postfix expressions  
parse_expr(c_unop('++', E)) --> parse_primary(E), [punct('++')].
parse_expr(c_unop('--', E)) --> parse_primary(E), [punct('--')].

%% c_parse_stmt/2 (v1) — extracted to c_ast_legacy.pl
%% c_parse_tokens/2 — extracted to c_ast_legacy.pl
%% Both zero external callers (medayek static analysis, 2026-05-17).

%% ═══════════════════════════════════════════════════════════════
%% ENRICHED TOKENIZER — classifies keywords and operators
%% ═══════════════════════════════════════════════════════════════
%%
%% Safe classifications (no context needed):
%%   keyword(K)    — C reserved words
%%   id(X)         — identifiers that aren't keywords
%%   operator(Op)  — unambiguous operators
%%   punct(P)      — context-dependent punctuation (left for parser)
%%   num(N)        — numeric literals
%%   str(S)        — string literals

c_keyword(auto).
c_keyword(break).
c_keyword(case).
c_keyword(char).
c_keyword(const).
c_keyword(continue).
c_keyword(default).
c_keyword(do).
c_keyword(double).
c_keyword(else).
c_keyword(enum).
c_keyword(extern).
c_keyword(float).
c_keyword(for).
c_keyword(goto).
c_keyword(if).
c_keyword(inline).
c_keyword(int).
c_keyword(long).
c_keyword(register).
c_keyword(return).
c_keyword(short).
c_keyword(signed).
c_keyword(sizeof).
c_keyword(static).
c_keyword(struct).
c_keyword(switch).
c_keyword(typedef).
c_keyword(union).
c_keyword(unsigned).
c_keyword(void).
c_keyword(volatile).
c_keyword(while).
%% C99/C11 additions
c_keyword(bool).
c_keyword(int8_t).
c_keyword(int16_t).
c_keyword(int32_t).
c_keyword(int64_t).
c_keyword(uint8_t).
c_keyword(uint16_t).
c_keyword(uint32_t).
c_keyword(uint64_t).
c_keyword(size_t).
c_keyword(nullptr).
c_keyword('NULL').
c_keyword(true).
c_keyword(false).

%% Unambiguous operators (no context needed to classify)
c_unambiguous_op('!=').
c_unambiguous_op('==').
c_unambiguous_op('<=').
c_unambiguous_op('>=').
c_unambiguous_op('++').
c_unambiguous_op('--').
c_unambiguous_op('&&').
c_unambiguous_op('||').
c_unambiguous_op('->').
c_unambiguous_op('+').
c_unambiguous_op('/').
c_unambiguous_op('%').
c_unambiguous_op('=').
c_unambiguous_op('<').
c_unambiguous_op('>').
c_unambiguous_op('!').
%% NOT included: * (deref/multiply/pointer), & (address/bitand),
%% - (subtract/negate), ; ({ } [ ] ( ) , are structural punctuation

%% c_enrich_tokens/2 (v1) and c_tokenize_enriched/2 (v1) — extracted to
%% c_ast_legacy.pl. Superseded by c_enrich_tokens_v2 (defined below)
%% and c_tokenize_enriched_v2. Zero external callers (medayek 2026-05-17).

%% The semicolon gets its own token class — eponymy-tier in C.
c_enrich_tokens_v2([], []).
c_enrich_tokens_v2([id(X)|Rest], [keyword(X)|ERest]) :-
    c_keyword(X), !, c_enrich_tokens_v2(Rest, ERest).
c_enrich_tokens_v2([punct(Op)|Rest], [operator(Op)|ERest]) :-
    c_unambiguous_op(Op), !, c_enrich_tokens_v2(Rest, ERest).
c_enrich_tokens_v2([punct(';')|Rest], [semicolon|ERest]) :-
    !, c_enrich_tokens_v2(Rest, ERest).
c_enrich_tokens_v2([T|Rest], [T|ERest]) :-
    c_enrich_tokens_v2(Rest, ERest).

c_tokenize_enriched_v2(String, Tokens) :-
    c_tokenize(String, RawTokens),
    c_enrich_tokens_v2(RawTokens, Tokens).

%% ─── Statement-level parsing (token list → AST) ───

%% Declaration with init: type * name = expr ;
parse_stmt(c_decl_init(c_type(ptr(c_type(named(TypeName)))), Name, Expr)) -->
    [id(TypeName), punct('*'), id(Name), punct('=')],
    parse_expr(Expr),
    [punct(';')].

%% Declaration without init: type * name ;
parse_stmt(c_decl(c_type(ptr(c_type(named(TypeName)))), Name)) -->
    [id(TypeName), punct('*'), id(Name), punct(';')].

%% Assignment: name = expr ;
parse_stmt(c_assign(c_var(Name), Expr)) -->
    [id(Name), punct('=')], parse_expr(Expr), [punct(';')].

%% Expression statement: expr ;
parse_stmt(c_expr_stmt(Expr)) -->
    parse_expr(Expr), [punct(';')].

%% If statement: if ( expr ) { stmts }
parse_stmt(c_if(Cond, Then)) -->
    [id(if), punct('(')], parse_expr(Cond), [punct(')')],
    [punct('{')], parse_stmt_list(Then), [punct('}')].

%% If-else: if ( expr ) { stmts } else { stmts }
parse_stmt(c_if(Cond, Then, Else)) -->
    [id(if), punct('(')], parse_expr(Cond), [punct(')')],
    [punct('{')], parse_stmt_list(Then), [punct('}')],
    [id(else)],
    [punct('{')], parse_stmt_list(Else), [punct('}')].

%% c_parse_stmt/2 (v1, second clause) and c_parse_stmts/2 (v1) — extracted
%% to c_ast_legacy.pl. Zero external callers (medayek 2026-05-17).
%% Note: there were historically two definitions of c_parse_stmt; both
%% were equivalent. Both are represented by the single legacy predicate.

%% ─── Chained expression parser (handles a.b[c].d→e chains) ───
%% Parses a primary expression, then applies postfix operators left-to-right.

parse_chained_expr(Expr) -->
    parse_primary(Base),
    parse_postfix(Base, Expr).

parse_postfix(Acc, Expr) -->
    [punct('.')], [id(Field)],
    parse_postfix(c_member(Acc, Field), Expr).
parse_postfix(Acc, Expr) -->
    [punct('->')], [id(Field)],
    parse_postfix(c_arrow(Acc, Field), Expr).
parse_postfix(Acc, Expr) -->
    [punct('[')], parse_chained_expr(Idx), [punct(']')],
    parse_postfix(c_index(Acc, Idx), Expr).
parse_postfix(Acc, Acc) --> [].

%% Override parse_expr for member/index/arrow to use chained parser
%% (these need to come before the simple parse_expr rules)

%% Full expression parser with chained access support
parse_full_expr(Expr) -->
    parse_chained_expr(Left),
    ( [punct(Op)], { member(Op, ['!=', '==', '<', '>', '+', '-', '/', '*', '=']) },
      parse_full_expr(Right) ->
        { Expr = c_binop(Op, Left, Right) }
    ;
        { Expr = Left }
    ).

%% c_parse_full_expr/2 — extracted to c_ast_legacy.pl
%% Zero external callers (medayek 2026-05-17). Superseded by parse_expr in v2/v3.

%% ─── Fixed chained expression parser with function calls ───

parse_chain(Expr) -->
    parse_atom_expr(Base),
    parse_chain_postfix(Base, Expr).

%% Namespace-qualified call: std::runtime_error(args), foo::bar(x, y).
%% Two-segment qualification covers our actual use cases (no need for
%% deeper nesting yet).
parse_atom_expr(c_call(c_qualified(Ns, Name), Args)) -->
    [id(Ns), punct('::'), id(Name), punct('(')],
    parse_chain_args(Args),
    [punct(')')].

parse_atom_expr(c_call(F, Args)) -->
    [id(F), punct('(')], parse_chain_args(Args), [punct(')')].
%% Prefix increment/decrement: ++i, --i. The c_pre_inc/c_pre_dec
%% AST shape matches what the emit side already produces.
parse_atom_expr(c_pre_inc(E)) --> [punct('++')], parse_chain(E).
parse_atom_expr(c_pre_dec(E)) --> [punct('--')], parse_chain(E).
%% Unary not: !x. Matches the emit-side c_not shape.
parse_atom_expr(c_not(E)) --> [punct('!')], parse_chain(E).
%% Unary sign prefix: -x and +x, handled EQUALLY (Iyun 2026-05-29, Heath request).
%% At the atom layer feeding parse_chain, so it composes in binop RHS too (a + -1).
%% Emit side already handles c_unop(Op,E) (emit_expr/254), so this completes round-trip.
%% Distinct from BINARY minus (a - b), which is parse_binop_rest with a left operand;
%% here there is no left operand -> unary.
parse_atom_expr(c_unop('-', E)) --> [punct('-')], parse_chain(E).
parse_atom_expr(c_unop('+', E)) --> [punct('+')], parse_chain(E).
parse_atom_expr(c_int(N)) --> [num(N)].
%% Float literal: dispatches to c_float_f for f-suffix forms,
%% c_float for unsuffixed, c_float_suffix for other suffix chars.
%% Same AST shape that the existing emit rules produce in the inverse
%% direction (Satya: same vocabulary both ways).
%% Float parse rules produce c_float_lit(V, Suffix) modeling the value + suffix.
%% source's exact textual representation. The emit side renders it
%% verbatim. This preserves scientific notation, precision, and
%% suffix forms for round-trip.
parse_atom_expr(c_float_lit(V, Suffix)) --> [floatnum(V, Suffix, _Raw)].
parse_atom_expr(c_string(S)) --> [str(S)].
parse_atom_expr(c_var(Name)) --> [id(Name)].
%% C cast expression: (TypeName) expr — placed before c_paren so
%% the type-name guard can decide. Guard prevents ordinary parens
%% containing an expression from being mis-parsed as a cast.
parse_atom_expr(c_cast(c_type(named(TypeName)), Inner)) -->
    [punct('(')], [id(TypeName)],
    { c_value_type_name(TypeName) },
    [punct(')')],
    parse_chain(Inner).
%% Parenthesized expression — wraps a full expr_v2 (binop, ternary, etc.)
%% so the chained parser can carry it as an atom in larger expressions.
parse_atom_expr(c_paren(E)) -->
    [punct('(')], parse_expr_v2(E), [punct(')')].

%% Braced initializer list: { e1, e2, ... } or { } (empty).
%% Used in C/C++ as initializer for arrays, structs, and (in
%% llama.cpp) as the shape argument to create_tensor().
%% The c_init_list AST term carries the list of element expressions.
parse_atom_expr(c_init_list(Elements)) -->
    [punct('{')], parse_chain_args(Elements), [punct('}')].

%% Call arguments: each arg is a full expression (parse_expr_v2)
%% so binops, ternaries, parens, casts work inside call args.
%% E.g. GGML_ASSERT(a < b) needs the < to be recognized.
parse_chain_args([]) --> [].
parse_chain_args([E]) --> parse_expr_v2(E).
parse_chain_args([E|Es]) --> parse_expr_v2(E), [punct(',')], parse_chain_args(Es).

parse_chain_postfix(Acc, Expr) -->
    [punct('.')], [id(Field)],
    parse_chain_postfix(c_member(Acc, Field), Expr).
parse_chain_postfix(Acc, Expr) -->
    [punct('->')], [id(Field)],
    parse_chain_postfix(c_arrow(Acc, Field), Expr).
parse_chain_postfix(Acc, Expr) -->
    [punct('[')], parse_chain(Idx), [punct(']')],
    parse_chain_postfix(c_index(Acc, Idx), Expr).
%% Postfix call-application: handles method calls like hparams.n_head_kv()
%% where the receiver is itself a chain (member access).
parse_chain_postfix(Acc, Expr) -->
    [punct('(')], parse_chain_args(Args), [punct(')')],
    parse_chain_postfix(c_call(Acc, Args), Expr).
parse_chain_postfix(Acc, Acc) --> [].

%% c_parse_chain/2 — extracted to c_ast_legacy.pl
%% Zero external callers (medayek 2026-05-17). Superseded by parse_chain_postfix.

%% ─── Unified statement parser using chained expressions ───

parse_stmt_v2(c_decl_init(c_type(ptr(c_type(named(TypeName)))), Name, Expr)) -->
    [id(TypeName), punct('*'), id(Name), punct('=')],
    parse_chain(Expr), [punct(';')].

%% Plain (non-pointer) decl-init: `uint32_t n_vocab = 0;` and similar.
%% Recognizes a value-typed declaration with initializer. The type name
%% list is restricted to known scalar/value types to avoid greedy
%% matching of arbitrary identifier pairs (which would shadow legitimate
%% expression statements like `n_vocab = ...` — that's a c_assign).
parse_stmt_v2(c_decl_init(c_type(named(TypeName)), Name, Expr)) -->
    [id(TypeName)],
    { c_value_type_name(TypeName) },
    [id(Name), punct('=')],
    parse_expr_v2(Expr), [punct(';')].

%% Const-qualified decl-init: `const bool found_swa = expr;`
%% Wraps the type in c_type(const(...)) to preserve the qualifier; the
%% emit side can render it back if needed. Same value-type guard.
parse_stmt_v2(c_decl_init(c_type(const(c_type(named(TypeName)))), Name, Expr)) -->
    [id(const), id(TypeName)],
    { c_value_type_name(TypeName) },
    [id(Name), punct('=')],
    parse_expr_v2(Expr), [punct(';')].

%% Declaration without initializer: `uint32_t dec_start_token_id;`
%% Just a value-type declaration consumed as a no-op statement.
parse_stmt_v2(c_decl(c_type(named(TypeName)), Name)) -->
    [id(TypeName)],
    { c_value_type_name(TypeName) },
    [id(Name), punct(';')].

%% Known scalar/value type names that introduce a declaration.
%% Conservative list — adding here is bounded; the alternative would be
%% to introduce a general type-name grammar with declarator syntax,
%% which is substrate-deeper than needed for what we lift.
c_value_type_name(uint32_t).
c_value_type_name(int32_t).
c_value_type_name(uint64_t).
c_value_type_name(int64_t).
c_value_type_name(uint16_t).
c_value_type_name(int16_t).
c_value_type_name(uint8_t).
c_value_type_name(int8_t).
c_value_type_name(size_t).
c_value_type_name(bool).
c_value_type_name(int).
c_value_type_name(float).
c_value_type_name(double).
%% C++ auto: type deduced from initializer. Treated as a value type
%% for declaration purposes since it appears in the same syntactic
%% slot as concrete value types.
c_value_type_name(auto).

parse_stmt_v2(c_assign(Target, Expr)) -->
    parse_chain(Target), [punct('=')], parse_expr_v2(Expr), [punct(';')].

%% Compound assignment: x += expr, x -= expr, x *= expr, x /= expr, x %= expr.
parse_stmt_v2(c_compound_assign(Op, Target, Expr)) -->
    parse_chain(Target), [punct(Op)],
    { member(Op, ['+=', '-=', '*=', '/=', '%=']) },
    parse_expr_v2(Expr), [punct(';')].

parse_stmt_v2(c_expr_stmt(Expr)) -->
    parse_expr_v2(Expr), [punct(';')].

%% Bare block: introduces a new scope. Matches the emit-side
%% c_block(Stmts) AST shape for round-trip symmetry.
parse_stmt_v2(c_block(Stmts)) -->
    [punct('{')], parse_stmts_v2(Stmts), [punct('}')].

%% Throw statement: `throw expr;`. Captures the C++ exception throw
%% as an opaque statement so the parser can walk past it without
%% getting stuck. Used in mistral3.
parse_stmt_v2(c_throw(Expr)) -->
    [id(throw)], parse_expr_v2(Expr), [punct(';')].

%% C++17 if-with-init-statement:
%%   if (init ; cond) { body }
%% where `init` is typically a decl-init (`const auto x = expr`).
%% The init-stmt is parsed as a parse_stmt_v2 itself (which consumes
%% the trailing `;` of the init-decl); then we need the condition
%% terminating with `)` and the braced body. Tried BEFORE the
%% simple 2-arg if to avoid the greedy cut.
%%
%% NOTE: parse_stmt_v2(c_decl_init/3) above produces statements that
%% END WITH `;` already, so we don't add a second `;` here.
parse_stmt_v2(c_if_init(Init, Cond, Then)) -->
    [id(if), punct('(')],
    parse_decl_init_no_semi(Init), [punct(';')],
    parse_expr_v2(Cond), [punct(')')],
    [punct('{')], parse_stmts_v2(Then), [punct('}')].

%% Decl-init without the trailing `;` for use in for-init and if-init
%% positions. Mirrors the standalone parse_stmt_v2(c_decl_init/3)
%% clauses but drops the `;` terminator.
parse_decl_init_no_semi(c_decl_init(c_type(const(c_type(named(TypeName)))), Name, Expr)) -->
    [id(const), id(TypeName)],
    { c_value_type_name(TypeName) },
    [id(Name), punct('=')],
    parse_expr_v2(Expr).
parse_decl_init_no_semi(c_decl_init(c_type(named(TypeName)), Name, Expr)) -->
    [id(TypeName)],
    { c_value_type_name(TypeName) },
    [id(Name), punct('=')],
    parse_expr_v2(Expr).

%% If-else: try the 3-arg (with else) shape FIRST so the partial
%% parser's greedy cut doesn't commit to the 2-arg before seeing the
%% else clause.
parse_stmt_v2(c_if(Cond, Then, Else)) -->
    [id(if), punct('(')], parse_expr_v2(Cond), [punct(')')],
    [punct('{')], parse_stmts_v2(Then), [punct('}')],
    [id(else)],
    [punct('{')], parse_stmts_v2(Else), [punct('}')].

%% Else-if: in C, `else if` is just `else <if-stmt>`. The else body
%% is a single statement (the inner if), not a braced block.
%% Represent as c_if(Cond, Then, [InnerIf]) so the else is uniformly
%% a list of statements.
parse_stmt_v2(c_if(Cond, Then, [InnerIf])) -->
    [id(if), punct('(')], parse_expr_v2(Cond), [punct(')')],
    [punct('{')], parse_stmts_v2(Then), [punct('}')],
    [id(else)],
    parse_stmt_v2(InnerIf),
    { InnerIf = c_if(_, _) ; InnerIf = c_if(_, _, _) }.

parse_stmt_v2(c_if(Cond, Then)) -->
    [id(if), punct('(')], parse_expr_v2(Cond), [punct(')')],
    [punct('{')], parse_stmts_v2(Then), [punct('}')].

%% Braceless if: `if (cond) stmt;` — single statement body, no braces.
%% Found in real C++ post-cpp-expansion (e.g., GGML_ASSERT expands to
%% `if (!(x)) ggml_abort(...)`). Represented internally as c_if/2 with
%% Then = [SingleStmt] so emit and downstream walkers handle it
%% uniformly. The emit path will reproduce it with braces — that's
%% acceptable because the AST captures the semantics (one statement
%% in the then-branch); brace style is a surface convention.
parse_stmt_v2(c_if(Cond, [Then])) -->
    [id(if), punct('(')], parse_expr_v2(Cond), [punct(')')],
    parse_stmt_v2(Then).

%% Switch statement: switch (expr) { case_clauses }
%%
%% AST: c_switch(Discriminant, Cases) — 2-arg shape symmetric with the
%% existing emit_stmt(c_switch/2) rule earlier in this file.
%%
%% Cases is a heterogeneous list of:
%%   c_case(Value, BodyStmts)     — labeled case
%%   c_default(BodyStmts)         — default clause
%%
%% Each case body is the run of statements between its label and the
%% next case/default/}. Statements within a case can include break.
parse_stmt_v2(c_switch(Discriminant, Cases)) -->
    [id(switch), punct('(')], parse_expr_v2(Discriminant), [punct(')')],
    [punct('{')], parse_case_clauses(Cases), [punct('}')].

%% Parse a sequence of case and default clauses until }.
parse_case_clauses([c_case(Value, Body) | Rest]) -->
    [id(case)], parse_chain(Value), [punct(':')],
    parse_case_body(Body),
    parse_case_clauses(Rest).
parse_case_clauses([c_default(Body) | Rest]) -->
    [id(default), punct(':')],
    parse_case_body(Body),
    parse_case_clauses(Rest).
parse_case_clauses([]) --> [].

%% Parse the body of one case: statements until next case/default/}.
%% Uses fall-through to the empty clause when parse_stmt_v2 fails.
%% c_break matches the emit side's existing atom for break statements
%% (Satya symmetry: same AST term in both directions).
parse_case_body([c_break | Rest]) -->
    [id(break), punct(';')], parse_case_body(Rest).
parse_case_body([S | Rest]) -->
    parse_stmt_v2(S), parse_case_body(Rest).
parse_case_body([]) --> [].

parse_stmts_v2([S|Ss]) --> parse_stmt_v2(S), !, parse_stmts_v2(Ss).
parse_stmts_v2([]) --> [].



c_parse_stmts_v2(String, ASTs) :-
    c_tokenize(String, Tokens),
    phrase(parse_stmts_v2(ASTs), Tokens).

%% Partial parser: collects as many recognized statements as possible
%% from the prefix of the token stream, returns them along with the
%% remaining unparsed tokens. Use for lifting purposes where we want
%% to extract structured facts from the parseable portion of a body
%% and leave the rest opaque (or hand it to another lifter).
%%
%% Greedy left-to-right with cut: takes the first statement that
%% parses, commits, recurses on the tail. If no statement parses at
%% the current position, the [] base case captures whatever tokens
%% remain via the explicit Tokens/RestTokens argument plumbing in the
%% wrapper predicate.
c_parse_stmts_v2_partial(String, ASTs, RestTokens) :-
    c_tokenize(String, Tokens),
    parse_stmts_v2_greedy(ASTs, Tokens, RestTokens).

parse_stmts_v2_greedy([S|Ss], T0, T) :-
    phrase(parse_stmt_v2(S), T0, T1), !,
    parse_stmts_v2_greedy(Ss, T1, T).
parse_stmts_v2_greedy([], T, T).

%% ─── For-loop parsing (unified with chained expressions) ───

parse_stmt_v2(c_for(Init, Cond, Step, Body)) -->
    [id(for), punct('(')],
    parse_for_init_v2(Init), [punct(';')],
    parse_expr_v2(Cond), [punct(';')],
    parse_expr_v2(Step),
    [punct(')')],
    [punct('{')], parse_stmts_v2(Body), [punct('}')].

parse_for_init_v2(c_decl_init(Type, Name, Expr)) -->
    parse_type(Type), [id(Name), punct('=')], parse_chain(Expr).
parse_for_init_v2(c_assign(c_var(Name), Expr)) -->
    [id(Name), punct('=')], parse_chain(Expr).

%% Postfix increment/decrement in chained parser
parse_chain_postfix(Acc, Expr) -->
    [punct('++')],
    parse_chain_postfix(c_postfix('++', Acc), Expr).
parse_chain_postfix(Acc, Expr) -->
    [punct('--')],
    parse_chain_postfix(c_postfix('--', Acc), Expr).

%% ─── Unified expression parser: chained + binary operators ───
%% parse_chain handles postfix (., ->, [], ++)
%% parse_expr_v2 adds binary operators between chained sub-expressions

parse_expr_v2(Expr) -->
    parse_chain(Left),
    parse_binop_rest(Left, Mid),
    parse_ternary_rest(Mid, Expr).

parse_binop_rest(Left, Expr) -->
    [punct(Op)],
    { member(Op, ['!=', '==', '<', '>', '<=', '>=', '+', '-', '*', '/', '%', '&&', '||']) },
    parse_chain(Right),
    parse_binop_rest(c_binop(Op, Left, Right), Expr).
parse_binop_rest(Expr, Expr) --> [].

%% Ternary tail: ? expr_v2 : expr_v2  (optional, right-associative).
%% Recursing into parse_expr_v2 for the ELSE slot gives nested ternary
%% support: cond ? A : (cond2 ? B : C) parses as nested c_ternary terms.
parse_ternary_rest(Cond, c_ternary(Cond, Then, Else)) -->
    [punct('?')],
    parse_expr_v2(Then),
    [punct(':')],
    parse_expr_v2(Else).
parse_ternary_rest(Expr, Expr) --> [].

%% ─── For-loop with unified expression parsing ───

parse_stmt_v3(c_for(Init, Cond, Step, Body)) -->
    [id(for), punct('(')],
    parse_for_init_v3(Init), [punct(';')],
    parse_expr_v2(Cond), [punct(';')],
    parse_expr_v2(Step),
    [punct(')')],
    [punct('{')], parse_stmts_v3(Body), [punct('}')].

parse_stmt_v3(c_decl_init(c_type(ptr(c_type(named(TypeName)))), Name, Expr)) -->
    [id(TypeName), punct('*'), id(Name), punct('=')],
    parse_expr_v2(Expr), [punct(';')].

parse_stmt_v3(c_assign(Target, Expr)) -->
    parse_chain(Target), [punct('=')], parse_expr_v2(Expr), [punct(';')].

parse_stmt_v3(c_expr_stmt(Expr)) -->
    parse_expr_v2(Expr), [punct(';')].

parse_stmt_v3(c_if(Cond, Then)) -->
    [id(if), punct('(')], parse_expr_v2(Cond), [punct(')')],
    [punct('{')], parse_stmts_v3(Then), [punct('}')].

parse_for_init_v3(c_decl_init(Type, Name, Expr)) -->
    parse_type(Type), [id(Name), punct('=')], parse_expr_v2(Expr).

parse_stmts_v3([]) --> [].
parse_stmts_v3([S|Ss]) --> parse_stmt_v3(S), parse_stmts_v3(Ss).

c_parse_stmts_v3(String, ASTs) :-
    c_tokenize(String, Tokens),
    phrase(parse_stmts_v3(ASTs), Tokens).

%% Postfix unary operators (++ and -- are postfix when parsed from C)
emit_expr(c_postfix(Op, E)) -->
    emit_expr(E), emit_atom(Op).

%% ═══════════════════════════════════════════════════════════════
%% CUDA EXTENSIONS — CUDA is C + a few keywords
%% ═══════════════════════════════════════════════════════════════

%% CUDA qualifiers
%% c_func([__global__], c_type(void), kernel_name, Params, Body)
%% Already supported — __global__ is just a qualifier atom!

%% CUDA built-in variables
%% threadIdx.x, blockIdx.x, blockDim.x — these are member access:
%% c_member(c_var(threadIdx), x) → "threadIdx.x"
%% Already supported by emit_expr(c_member(...))!

%% CUDA kernel launch syntax: kernel<<<grid, block>>>(args)
%% This needs a new AST node:
emit_expr(c_cuda_launch(Kernel, Grid, Block, Args)) -->
    emit_atom(Kernel), "<<<", emit_expr(Grid), ", ", emit_expr(Block), ">>>(",
    emit_expr_list(Args), ")".

%% CUDA __shared__ memory declaration (1D)
emit_stmt(c_shared_decl(Type, Name, Size), Indent) -->
    emit_indent(Indent), "__shared__ ", emit_type(Type), " ", 
    emit_atom(Name), "[", emit_expr(Size), "];\n".

%% CUDA __shared__ memory declaration (2D)
%% Per Heath's kernel-lift trajectory 2026-05-19: matmul kernels use
%% 2D tiled shared memory like `__shared__ float sA[TILE][TILE]`.
%% This emits the substrate-historical form. For bank-conflict
%% optimization, callers should use 1D with manual indexing
%% (a future fix flag could expose this choice).
emit_stmt(c_shared_decl_2d(Type, Name, Rows, Cols), Indent) -->
    emit_indent(Indent), "__shared__ ", emit_type(Type), " ",
    emit_atom(Name), "[", emit_expr(Rows), "][", emit_expr(Cols),
    "];\n".

%% CUDA __shared__ scalar declaration (no array brackets)
%% Per Heath's kernel-lift trajectory 2026-05-19 ~06:30 UTC:
%% LayerNorm's reduction pattern uses scalar shared values like
%% `__shared__ float s_mean, s_inv_std;` — not arrays. This avoids
%% the syntactic divergence of using [1] form to emulate scalar.
%%
%% Use list of atoms for multiple-name form:
%%   c_shared_scalar_decl(c_type(float), [s_mean, s_inv_std])
%% Use single atom for single-name form:
%%   c_shared_scalar_decl(c_type(float), s_mean)
emit_stmt(c_shared_scalar_decl(Type, Names), Indent) -->
    { is_list(Names) }, !,
    emit_indent(Indent), "__shared__ ", emit_type(Type), " ",
    emit_atom_list_comma(Names), ";\n".

emit_stmt(c_shared_scalar_decl(Type, Name), Indent) -->
    { \+ is_list(Name) },
    emit_indent(Indent), "__shared__ ", emit_type(Type), " ",
    emit_atom(Name), ";\n".

%% Helper: emit list of atoms separated by ", "
emit_atom_list_comma([]) --> [].
emit_atom_list_comma([X]) --> emit_atom(X).
emit_atom_list_comma([X, Y | Rest]) -->
    emit_atom(X), ", ", emit_atom_list_comma([Y | Rest]).

%% CUDA __syncthreads()
emit_stmt(c_syncthreads, Indent) -->
    emit_indent(Indent), "__syncthreads();
".

%% Void return statement
emit_stmt(c_return_void, Indent) -->
    emit_indent(Indent), "return;\n".

%% Float with fewer decimals

%% ═══════════════════════════════════════════════════════════════
%% CUDA-specific AST nodes for kernel generation
%% ═══════════════════════════════════════════════════════════════

%% extern __shared__ (dynamically-sized shared memory)
emit_stmt(c_extern_shared(Type, Name), Indent) -->
    emit_indent(Indent), "extern __shared__ ", emit_type(Type), " ",
    emit_atom(Name), "[];\n".

%% CUDA kernel launch: kernel<<<grid, block, smem>>>(args)
emit_stmt(c_cuda_launch(Kernel, Grid, Block, Smem, Args), Indent) -->
    emit_indent(Indent), emit_atom(Kernel), "<<<",
    emit_expr(Grid), ", ", emit_expr(Block), ", ", emit_expr(Smem),
    ">>>(", emit_expr_list(Args), ");
".

%% CUDA kernel launch without shared memory
emit_stmt(c_cuda_launch(Kernel, Grid, Block, Args), Indent) -->
    emit_indent(Indent), emit_atom(Kernel), "<<<",
    emit_expr(Grid), ", ", emit_expr(Block),
    ">>>(", emit_expr_list(Args), ");
".

%% extern "C" block
emit_stmt(c_extern_c(Stmts), Indent) -->
    emit_indent(Indent), "extern \"C\" {
",
    emit_stmts(Stmts, Indent),
    emit_indent(Indent), "} // extern \"C\"\\n".

%% Top-level extern "C" (no indentation for contents)
emit_top(c_extern_c_begin) --> "extern \"C\" {
".
emit_top(c_extern_c_end) --> "} // extern \"C\"\\n".

%% Compound assignment operators (+=, -=, *=, /=, %=)
%% Parser produces c_compound_assign(Op, Target, Expr).
emit_stmt(c_compound_assign(Op, Target, Expr), Indent) -->
    emit_indent(Indent), emit_expr(Target), " ", emit_atom(Op), " ",
    emit_expr(Expr), ";\n".

%% Logical NOT
emit_expr(c_not(E)) --> "!", emit_expr(E).

%% Prefix increment/decrement
emit_expr(c_pre_inc(E)) --> "++", emit_expr(E).
emit_expr(c_pre_dec(E)) --> "--", emit_expr(E).

%% Namespace-qualified name: ns::Name (as a c_var-equivalent atom for use in calls)
emit_expr(c_qualified(Ns, Name)) -->
    emit_atom(Ns), "::", emit_atom(Name).

%% Bare declaration (no initializer): uint32_t x;
emit_stmt(c_decl(Type, Name), Indent) -->
    emit_indent(Indent), emit_type(Type), " ", emit_atom(Name), ";\n".

%% Throw statement
emit_stmt(c_throw(Expr), Indent) -->
    emit_indent(Indent), "throw ", emit_expr(Expr), ";\n".

%% Bare block as a statement
%% c_for_step: for-loop where the step is a compound-assign statement
%% (e.g. i += 128, s >>= 1, d *= 2) rather than a simple expression.
%% The Op atom is the full compound operator including '=': '+=', '>>=', '*=', etc.
%% Emits: for (init; cond; lhs op rhs) { body }
emit_stmt(c_for_step(Init, Cond, c_compound_assign(Op, LHS, RHS), Body), Indent) -->
    { I1 is Indent + 1 },
    emit_indent(Indent), "for (",
    emit_for_init(Init), "; ", emit_expr(Cond), "; ",
    emit_expr(LHS), " ", emit_atom(Op), " ", emit_expr(RHS),
    ") {\n",
    emit_stmts(Body, I1),
    emit_indent(Indent), "}\n".

%% c_compound_assign as a standalone statement: lhs op rhs;
%% Op is the full operator atom: '+=', '-=', '*=', '/=', '>>=', '<<=', '&=', '|='
emit_stmt(c_compound_assign(Op, LHS, RHS), Indent) -->
    emit_indent(Indent), emit_expr(LHS), " ", emit_atom(Op), " ", emit_expr(RHS), ";\n".

emit_stmt(c_block(Stmts), Indent) -->
    { I1 is Indent + 1 },
    emit_indent(Indent), "{\n",
    emit_stmts(Stmts, I1),
    emit_indent(Indent), "}\n".

%% break statement (when emitted as a top-level stmt rather than via
%% c_case's body where it's handled by emit_switch_cases)
emit_stmt(c_break, Indent) -->
    emit_indent(Indent), "break;\n".

%% For-loop
emit_stmt(c_for(Init, Cond, Step, Body), Indent) -->
    { I1 is Indent + 1 },
    emit_indent(Indent), "for (",
    emit_for_init(Init), "; ",
    emit_expr(Cond), "; ",
    emit_expr(Step), ") {\n",
    emit_stmts(Body, I1),
    emit_indent(Indent), "}\n".

%% C++17 if-with-init: if (init; cond) { body }
emit_stmt(c_if_init(Init, Cond, Then), Indent) -->
    { I1 is Indent + 1 },
    emit_indent(Indent), "if (",
    emit_for_init(Init), "; ",
    emit_expr(Cond), ") {\n",
    emit_stmts(Then, I1),
    emit_indent(Indent), "}\n".

%% For-init: a decl-init without the trailing semicolon.
emit_for_init(c_decl_init(Type, Name, Expr)) -->
    emit_type(Type), " ", emit_atom(Name), " = ", emit_expr(Expr).
emit_for_init(c_assign(Target, Expr)) -->
    emit_expr(Target), " = ", emit_expr(Expr).

%% For loop with compound step (i += stride)
emit_expr(c_compound_step(Var, Op, Val)) -->
    emit_expr(Var), " ", emit_atom(Op), " ", emit_expr(Val).

%% sizeof expression
emit_expr(c_sizeof(Type)) --> "sizeof(", emit_type(Type), ")".

%% min/max builtins
emit_expr(c_min(A, B)) --> "min(", emit_expr(A), ", ", emit_expr(B), ")".

%% CUDA half type
emit_type(c_type(half)) --> "half".
emit_type(c_type(const_uchar_ptr)) --> "const unsigned char *".
emit_type(c_type(uchar_ptr)) --> "unsigned char *".

%% __restrict__ qualifier
emit_type(c_type(restrict_ptr(Base))) --> emit_type(Base), " * __restrict__".
emit_type(c_type(const_restrict_ptr(Base))) --> "const ", emit_type(Base), " * __restrict__".

%% Float literal with 'f' suffix
%% Float with f-suffix: render with minimal trailing zeros to match
%% upstream's convention (8.0f, not 8.000000f).

%% (c_float_raw removed: float literals are modeled as c_float_lit(V,Suffix).)
%% Used by parse-side floats to preserve scientific notation and
%% exact precision for round-trip.

%% c_raw is DEAD. The substrate is 100% semantic.
%%
%% c_raw was the escape hatch: opaque C strings the substrate could not
%% read back, vary, or compose. Every c_raw was substrate debt —
%% "verbatim is correct" was rationalization; verbatim was the substrate
%% failing to comprehend what it emits.
%%
%% As of commit fa9c27e (2026-05-20), all 333 c_raw calls in
%% kernel_templates_blas.pl have been replaced with structural c_ast nodes.
%% The substrate sees ALL of itself. The optimizer can pattern-match on
%% every loop, every branch, every stride, every accumulation.
%%
%% c_raw now throws an error. If you need it, you need a new c_ast node.
%% File an issue. Don't reintroduce opacity.
emit(c_raw(Text), _Indent) -->
    { atom_concat('c_raw is dead. Use a structural c_ast node instead: ', Text, Msg),
      throw(error(c_raw_is_dead, context(Msg))) }.


%% Preprocessor directives (structurally expressed, not escape-hatched)
%% c_pragma(Text) — emits "#pragma <text>" as a complete preprocessor line
%% Used for compiler hints like "#pragma unroll", "#pragma once", etc.
emit_stmt(c_pragma(Text), Indent) -->
    emit_indent(Indent), "#pragma ", emit_atom(Text), "\n".

%% ═══════════════════════════════════════════════════════════════
%% nd_index_expr/2 — build N-dimensional strided index expression
%% ═══════════════════════════════════════════════════════════════
%%
%% nd_index_expr([Base, Dim1, Idx1, Dim2, Idx2, ...], Expr)
%%
%% Base case: single element → c_var(Base)
%% One pair:  Base*Dim1+Idx1 → c_binop('+', c_binop('*', Base, Dim1), Idx1)
%% Two pairs: (Base*D1+I1)*D2+I2 → left-fold with c_paren
%%
%% Atoms are wrapped in c_var/1. This lets callers write:
%%   c_nd_index([n, 'C_in', ci, 'H_in', hi, 'W_in', wi])
%% instead of deeply nested c_binop/c_paren trees.

nd_index_expr([X], c_var(X)).
nd_index_expr([Base, Dim, Idx | Rest], Expr) :-
    Acc = c_binop('+', c_binop('*', c_var(Base), c_var(Dim)), c_var(Idx)),
    nd_index_fold(Rest, Acc, Expr).

nd_index_fold([], Acc, Acc).
nd_index_fold([Dim, Idx | Rest], Acc, Expr) :-
    Next = c_binop('+', c_binop('*', c_paren(Acc), c_var(Dim)), c_var(Idx)),
    nd_index_fold(Rest, Next, Expr).




%% c_extern_c_open / c_extern_c_close — structural extern "C" block markers
%% Used by generators that build function lists incrementally.
%% For self-contained blocks, use c_extern_c(Stmts) instead.
emit(c_extern_c_open, _Indent) --> "extern \"C\" {\n".
emit(c_extern_c_close, _Indent) --> "}\n".
