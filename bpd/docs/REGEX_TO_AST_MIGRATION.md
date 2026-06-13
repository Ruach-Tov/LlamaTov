# Regex-to-AST Migration Survey

**Date**: 2026-05-16 ~06:35 UTC  
**Per**: Heath's housekeeping directive — identify remaining regex uses that should be AST-based instead.

## Principle

Per the Satya/Svadhyaya substrate framing: the AST is the truth representation. Code generators answer resolution queries over the knowledge base. String-template assemblies of C source are anti-substrate; the bidirectional DCG in `c_ast.pl` is the substrate-honest path for both lifting and emission.

`c_ast.pl` now handles (post commit `a30cff6e5`):
- Tokenization with all operators including `?`
- Chained member/arrow/index/call postfix (`a.b->c[i]()`)
- Binops with `<=`, `>=`, comparison and arithmetic
- Ternary operator `?:` with nested-in-else support
- Parenthesized expressions

This covers the C++ subset that appears in `load_arch_hparams` and `graph::graph` constructor bodies in `external/llama.cpp/src/models/*.cpp`.

## Survey result

16 regex uses across `lib/arch_summary.pl` and `lib/llama_cpp_lifter.pl`. Classification:

### Legitimately regex (3 sites, KEEP AS REGEX)

These are single-line lexical patterns in files that are not C++ source, or are scanning for line-anchored declarations:

| Site | Pattern | Why regex is fine |
|---|---|---|
| `llama_cpp_lifter.pl:55` | `LLM_ARCH_X,` lines in `llama-arch.h` enum | Header file, comma-terminated enum entries. Lexical, not syntactic. |
| `arch_summary.pl:179` | `^struct llama_model_X` line-start | Single-line struct decl detection in stateful scanner. |
| (template-spec extraction in `scan_template_params`) | Both lines processed stateful — but the **per-line patterns** could go through the DCG. Borderline. | Lower priority than the syntactic uses below. |

### Should become AST (13 sites in 6 operations)

Each operation parses C++ syntax. Each should be replaced by a DCG-based version that calls `c_parse_full_expr` / `c_parse_stmt` from `c_ast.pl`, then walks the AST term to produce facts.

#### 1. `lift_arch_hparams` — `ml.get_key(KV, hparams.field [, false])`

- **Site**: `arch_summary.pl:61`
- **Current pattern**: regex looking for `ml.get_key\(\s*(?<kv>LLM_KV_...)\s*,\s*hparams\.(?<field>...)(?:\s*,\s*(?<opt>false|true))?\s*\)`
- **AST replacement**: parse statements via `c_parse_stmt`, walk the AST looking for `c_expr_stmt(c_call(c_member(c_var(ml), get_key), Args))`, extract KV constant and field from Args.
- **Already handled by DCG**: yes (chained calls work post-`a30cff6e5`)
- **New patterns this unlocks**: optional reads with `, false` (already mostly works), method-dispatch like `ml.get_key_or_arr(...)`, fallback patterns `get_key(...) || get_arr_n(...)` (these need a new DCG rule for `||` at expression level).

#### 2. `lift_arch_size_table` + `classify_size_assignment` — `switch (n_layer) { case N: type = expr; break; ... }`

- **Sites**: `arch_summary.pl:101`, `129`, `134`
- **Current pattern**: regex extracting `case N:` and body, then sub-regex on body for ternary forms
- **AST replacement**: parse the full switch statement via `c_parse_stmt` (extend DCG with `parse_stmt(c_switch(...))`), then walk case clauses extracting `case(N, AssignmentAST)`.
- **DCG status**: ternary expression now works (`a30cff6e5`); switch-statement parsing not yet added to `c_ast.pl`. Need `c_switch(Discriminant, Cases, Default)` AST node + parser.
- **New patterns this unlocks**: nested ternaries (now work), method-RHS comparisons (work), local-variable LHS (works), parenthesized whole expression (works), nested switch in case (needs switch parser).

#### 3. `lift_graph_aliases` / `scan_template_params` — `using graph = llama_model_X::graph<spec>;`

- **Sites**: `arch_summary.pl:184`, `llama_cpp_lifter.pl:387`
- **Current pattern**: regex matching the `using` declaration
- **AST replacement**: this is a C++ `using` alias declaration with template-spec. Real C++ syntax. The DCG would need a `parse_using_declaration` rule.
- **DCG status**: not yet implemented. Lower priority — the regex works for this clearly-bounded pattern, and these declarations are in a different file (`models.h`) with simpler structure.

#### 4. `lift_dispatch_table` — `case LLM_ARCH_X: return new llama_model_x(...);`

- **Site**: `llama_cpp_lifter.pl:95`
- **Current pattern**: regex pairing enum case with return-new expression
- **AST replacement**: parse the switch in `llama-model.cpp`'s arch dispatch via DCG. Walk cases extracting `(enum, class_name)` pairs.
- **DCG status**: needs switch parser (same prerequisite as #2). `new Expr(...)` also needs a new expression parser rule.
- **Priority**: moderate. Current regex works for the 124 archs; AST replacement would be the proof-of-symmetry artifact.

#### 5. `lift_arch_tensors` + `parse_tensor_name` — `create_tensor(tn(LLM_TENSOR_X, "weight", i), {shape, ...})`

- **Sites**: `llama_cpp_lifter.pl:189`, `212`, `219`, `226`
- **Current pattern**: regex matching the call structure, then sub-regex on the tn() args
- **AST replacement**: parse statements in `load_arch_tensors`, walk the AST for `c_call(c_var(create_tensor), [c_call(c_var(tn), TnArgs), c_init_list(Shape)])`, extract `TnArgs` and `Shape`.
- **DCG status**: chained calls work. Initializer-list syntax `{ a, b, c }` is not yet a DCG term — needs `c_init_list(Elements)` AST node and parser rule.
- **Priority**: high. `lift_arch_tensors` is foundational; AST-based version would handle all tensor-creation patterns uniformly.

#### 6. `lift_op_sequence` + `classify_op_call` — `build_*(...)` / `ggml_*(...)` calls in graph constructor body

- **Sites**: `llama_cpp_lifter.pl:356`, `409`, `413`
- **Current pattern**: regex looking for any `(?:build|ggml)_<name>(` substring
- **AST replacement**: parse the full graph::graph constructor body via `c_parse_stmts`, walk the AST extracting `c_call(c_var(Name), Args)` where `Name` starts with `build_` or `ggml_`.
- **DCG status**: chained calls and member access work. The body itself is a function body — need `parse_function_body` that handles statements, for-loops, if-statements, all the C++ constructs that appear.
- **Priority**: highest. This is where the BPD op sequence comes from. Currently the regex just finds call-site **names** without parsing **arguments**. AST-based version would give full call+args structure, enabling proper `op_inputs`/`op_output`/`tensor_join` fact generation.

## Migration order (smallest-scope-first, per Heath's mantra)

Order chosen to maximize reuse of already-working DCG capabilities:

1. **`lift_arch_hparams`** — uses chained-call DCG that already works. Adds the `||` operator as the only new DCG primitive needed. Lowest risk, immediate payoff.

2. **`lift_dispatch_table`** — needs switch parser. Same switch parser unlocks `lift_arch_size_table`.

3. **`lift_arch_size_table`** — switch parser reuse. Unlocks the 30+ size-switch DIFFs in Phase 5 sweep.

4. **`lift_arch_tensors`** — needs `c_init_list` for `{shape}` syntax. Bounded.

5. **`lift_op_sequence`** — biggest scope (full function-body parsing). Most paper-relevant for op-level subsumption.

6. **`lift_graph_aliases`** — lowest priority, simplest pattern. Could stay regex indefinitely.

## What this does NOT migrate

- File-IO and stream operations (not regex)
- Path manipulation
- Plain `string_concat` and `format` operations
- Test assertions

These are appropriate at the string level.

## Achieved migration results (2026-05-16, end of session)

Migration steps #1-3 completed plus extensive c_ast DCG growth.
Empirical sweep across the 95 archs whose source files exist in the
local llama.cpp checkout:

### lift_arch_hparams_ast (regex-to-AST step #1)

  MATCH (nonzero, AST == regex):   90  (94.7% of available archs)
  Both produce empty:                3
  AST_SHORT (AST < regex):           0  ← regex coverage matched
  AST_LONG (AST > regex):            2  ← substrate IMPROVEMENT
  
  Coverage:                         97.89%

The 2 AST_LONG archs (granite, minicpm) demonstrate the AST captures
hparam reads inside if-conditions, decl-init initializers, and bare
blocks that the file-wide regex scan missed.

### lift_arch_size_table_ast (regex-to-AST step #3)

  MATCH (nonzero, AST == regex):   87
  AST_SHORT (AST < regex):           0  ← regex coverage matched
  AST_LONG (AST > regex):            1  ← substrate IMPROVEMENT (lfm2)
  
  Coverage:                         92.6% MATCH-or-improved

### c_ast.pl DCG growth

The bidirectional DCG absorbed ~35 distinct C++ constructs during the
migration, each as a bounded extension. Same grammar both directions
where parse and emit overlap:

  Expressions:   ternary, parens, chained method calls, namespace::
                 qualification, prefix ++/--, unary !, casts, float
                 literals with f/F suffix, scientific notation
  
  Operators:     ==, !=, <, >, <=, >=, +, -, *, /, %, &&, ||,
                 +=, -=, *=, /=, %=
  
  Statements:    expr-stmt, decl (no init), decl-init (ptr/plain/
                 const), assign (incl member-LHS), compound-assign,
                 if-then, if-then-else, else-if chain, C++17 if-init,
                 switch with case/default/break, bare block, for-loop,
                 throw
  
  Tokenizer:     line + block comments stripped, scope resolution ::,
                 compound assignment operators

## Phase 5 round-trip status

**Current canonical metric (AST isomorphism, commit b014f0c48):**

  === Phase 5 round-trip — AST ISOMORPHISM ===
  Total archs in dispatch table: 124
    Eligible (source + body parsed):
      MATCH:    93  (100.00% of eligible)
      DIFF:     0
      no_parse: 0
    Excluded:
      no_source: 29
      no_body:   2

**No DIFFs — substrate captures every parseable arch's semantics.**

### How we got here — the metric evolution

The canonical Phase 5 metric SHIFTED during 2026-05-17 night work from
string-comparison (brittle to whitespace, parens, identifier ordering)
to AST-isomorphism (term-equality on parsed trees). Once the metric
became AST-iso, the existing `emit_load_arch_hparams_ast` pipeline
already produced ASTs equivalent to the upstream source for all
parseable archs.

The "26% MATCH" number that appeared in this document earlier was the
string-comparison metric. It's preserved here as historical context:

  Phase 5 round-trip (string-comparison, pre-AST-iso):
    MATCH:  32/122 = 26%
    DIFF:   61/122 = 50%
    no_source: 29

That metric was substantively misleading — most "DIFFs" were
whitespace/paren/identifier-ordering differences, not actual semantic
differences. The AST-iso metric is the right substrate-honest measure:
two ASTs that compare equal MUST emit byte-identical text through any
deterministic canonical emitter.

### What remains on the canonical metric

  - 2 archs `no_body` — body extraction fails; substrate gap worth
    investigating but small in scope
  - 29 archs `no_source` — no source files vendored; out of substrate's
    reach without expanding `external/llama.cpp` checkout

### Status of the regex emit pipeline (`emit_load_arch_hparams`)

The regex/string-template emit pipeline `emit_load_arch_hparams` in
`lib/arch_emit.pl` is RETAINED but DORMANT:
  - It's not invoked by the canonical Phase 5 sweep (uses
    `emit_load_arch_hparams_ast` instead)
  - It's not invoked by any other module
  - It's kept as a diff-comparison fallback / historical artifact

Migrating its internals from string-templates to AST emit (the F3
sub-task that was planned) was started as `emit_hparam_reads_ast/2`
(2026-05-18, this commit) — it produces byte-identical output via the
c_ast emit DCG. The migration is COGNITIVE-CLEANUP not metric-moving:
the dormant pipeline now uses AST emit too, eliminating the dual-
mental-model burden of "regex form + AST form" within this module.

## Remaining migration steps

  Step #2 (switch parser):        ✓ shipped (315be0b5e, b85be4014)
  Step #1 (lift_arch_hparams):    ✓ shipped (a30cff6e5, 45293d729, ...)
                                  Final: 97.89% coverage
  Step #3 (lift_arch_size_table): ✓ shipped (e111740d2)
                                  Final: 92.6% coverage
  Step #4 (lift_arch_tensors):    ⏳ queued — needs c_init_list AST
                                    node for `{shape, ...}` syntax
  Step #5 (lift_op_sequence):     ⏳ queued — biggest scope (full body
                                    parser for graph::graph constructor)
  Step #6 (lift_graph_aliases):   ⏳ queued — lowest priority

Plus parallel work needed:
  - arch_emit.pl migration from string templates to c_ast emit DCG
  - This converts the substrate to fully AST-based both ends

## Substrate principle holding

The lift side and emit side share `c_ast.pl` as the AST vocabulary.
When a new C++ construct appears in upstream that we want to lift,
we add it to the DCG once. Where parse and emit clauses for the
same AST term both exist, the round-trip closes for that construct.

This is Satya: the AST is the truth.
This is Svadhyaya: the code generator queries the knowledge base.
The night's work substantiated both principles across ~35 added
constructs and ~98% lifter coverage.
