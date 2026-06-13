# Regex Lifter → AST Migration: Decomposition of the 9 Substantial Sites

**Date**: 2026-05-18 ~01:30 UTC
**Per**: Heath's "enumerate the 9 substantial Prolog regex uses, and consider for each how we would decompose it into subtasks" directive.

## Framing

The audit (commit 841469458) identified 9 substantial Prolog regex
uses across `lib/arch_summary.pl` and `lib/llama_cpp_lifter.pl`. Each
is a lifter — extracts structured data from C++ source via PCRE
pattern matching. The source code itself names the substrate-honest
direction: "Replace with AST-based lifter using tree-sitter-cpp."

The 9 sites group into **4 logical predicates**, each with 1-3 regex
branches. Migration happens per-predicate, not per-regex — the
branches share lifted-output shape and should migrate together.

## Group analysis

The 4 predicates and their substrate roles:

  Predicate                    | Sites | Role                              | Module
  -----------------------------|-------|-----------------------------------|---------------------
  classify_size_assignment     | 1, 2  | Lift llm_type size table          | arch_summary.pl
  scan_template_params         | 3, 4  | Lift template-param specialization| arch_summary.pl
  parse_tensor_name            | 5,6,7 | Lift LLM_TENSOR_* operand spec    | llama_cpp_lifter.pl
  classify_op_call             | 8, 9  | Lift call-site op kind            | llama_cpp_lifter.pl

Each predicate is one migration unit. Total: 4 migration units, not
9 — the unit-of-work boundary is per-predicate.

---

## Migration Unit 1: classify_size_assignment

### What it lifts

`size_rec(_, Cond, Type)` from a single C++ assignment statement
inside `llama_model::load_arch_hparams`. The body has the shape:

```cpp
switch (hparams.n_layer) {
    case 32: type = LLM_TYPE_7B; break;
    case 22:
        type = hparams.n_embd == 1536 ? LLM_TYPE_1B : LLM_TYPE_3B;
        break;
    default: type = LLM_TYPE_UNKNOWN; break;
}
```

The regex sees the per-case body string and produces one of:
- `size_rec(_, unconditional, 'LLM_TYPE_7B')`
- `size_rec(_, if_then_else(condition(n_embd, ==, 1536), 'LLM_TYPE_1B', 'LLM_TYPE_3B'), 'LLM_TYPE_1B')`
- `size_rec(_, complex(BodyStr), unknown)` — fallback

### C++ structure being matched

**Site 1 (unconditional)**: A single assignment `type = LLM_TYPE_X;`
- AST: `c_assign(c_var(type), c_var('LLM_TYPE_X'))`

**Site 2 (ternary)**: A ternary assignment
`type = hparams.LHS OP RHS ? LLM_TYPE_A : LLM_TYPE_B;`
- AST: `c_assign(c_var(type), c_ternary(c_binop(OP, c_member(c_var(hparams), LHS), RHS_expr), c_var('LLM_TYPE_A'), c_var('LLM_TYPE_B')))`

### c_ast vocabulary needed

All present: `c_assign`, `c_var`, `c_member`, `c_ternary`, `c_binop`, `c_int`, `c_call`, `c_paren`.

### Decomposition into subtasks (5 bounded steps)

1. **Parse the body string into c_ast statements** via `c_parse_stmts`.
   The body is short (one statement, possibly with `break`); should
   parse with current parser coverage. Verify on a representative
   sample of bodies (sweep through arch_summary's existing test
   corpus, see what %% currently parse).

2. **Pattern-match the AST for the unconditional case**:
   match `c_assign(c_var(type), c_var(LLMType))` and lift to
   `size_rec(_, unconditional, LLMType)`. Single Prolog clause.

3. **Pattern-match the AST for the ternary case**:
   match `c_assign(c_var(type), c_ternary(Cond, ThenExpr, ElseExpr))`
   and recursively classify Cond/ThenExpr/ElseExpr. Single Prolog
   clause + helper for Cond shape (extract LHS/Op/RHS from
   `c_binop(Op, c_member(c_var(hparams), LHS), RHS)`).

4. **Fallback case for unmatched ASTs**: `complex(SomeRepr)`. The
   complex case currently stores the raw BodyStr; the AST version
   can store the AST itself (more useful for downstream emit).

5. **Verify equivalence on existing corpus**: run both the regex
   and AST classifiers over arch_summary's currently-classified
   bodies; assert classification matches for every body. Empirical
   equivalence test before removing the regex form.

**Estimated effort**: 1-2 hours. Bounded by the parser's coverage of
the C++ body fragments — which is currently 100% on the Phase 5
eligible archs, so this should work.

**Substrate impact**: arch_summary.pl's per-case classifier no
longer head-compiles regex. Mental model: "this is a C statement
being structurally matched" instead of "this is a string matching
this PCRE pattern."

---

## Migration Unit 2: scan_template_params

### What it lifts

`template_param(ChildArch, ParentArch, Specialization)` facts from
`models.h`. The header has the shape:

```cpp
struct llama_model_llama {
    // ...
};

struct llama_model_baichuan : public llama_model_llama {
    using graph = llama_model_llama::graph;
};

struct llama_model_command_r : public llama_model_llama {
    using graph = llama_model_llama::graph<command_r_specialization>;
};
```

The regex sees lines and produces:
- `template_param(baichuan, llama, none)`
- `template_param(command_r, llama, template(command_r_specialization))`

### C++ structure being matched

**Site 3**: A `struct llama_model_X` declaration line.
- This is the OPENING of a top-level struct.
- AST: `c_struct_def('llama_model_X', ...)` — but we only care about
  the name, not the body, and the parser would need to consume the
  whole struct body to produce `c_struct_def`.

**Site 4**: A `using graph = llama_model_<parent>::graph<<spec>>;`
declaration inside a struct.
- AST: `c_using_decl(graph, c_qualified('llama_model_parent', graph,
  TemplateArgs))` — but our parser doesn't yet have `c_using_decl`
  or `c_qualified_template` nodes!

### c_ast vocabulary needed

**Partially present**: `c_struct_def` exists. **MISSING**:
- `c_using_decl(Name, Type)` — `using X = T;`
- `c_qualified_template(Namespace, Name, TemplateArgs)` —
  `Namespace::Name<TemplateArgs>`

### Decomposition into subtasks (8 steps — bigger because parser extension needed)

1. **Extend c_ast.pl to support `c_using_decl`**. Add the AST node
   definition + emit_stmt DCG rule. One commit.

2. **Extend c_ast.pl to support `c_qualified_template`**. Add the
   AST node + emit_expr DCG rule. May already partially exist as
   `c_qualified`; check and either extend or add new node.

3. **Extend the parser to recognize `using X = ...;`**.
   `c_parse_stmts` needs to lex `using` keyword, expect an
   identifier, `=`, a type expression, `;`. New grammar rule.

4. **Extend the parser to recognize `Namespace::Name<TemplateArgs>`**.
   When the parser sees `::`, it should consume the qualified name;
   when it sees `<`, it should consume template arguments. This
   may already exist if Phase 5 handles templated types.

5. **Verify parser on `models.h` body**: parse the whole header
   into a list of `c_struct_def` nodes; verify each has the
   expected name + parent + using-graph children.

6. **Pattern-match the parsed AST**: walk the `c_struct_def` nodes,
   find ones matching `llama_model_<child>`, look for `c_using_decl`
   inside, extract parent + spec. Produce `template_param` facts.

7. **Verify equivalence**: compare the AST lifter's output against
   the regex lifter's output on the current models.h. Should be
   bit-identical.

8. **Retire the regex lifter** after verification. One commit.

**Estimated effort**: 4-6 hours. The parser extension (steps 1-4)
is the substantial part. The pattern-match (step 6) is short
Prolog.

**Substrate impact**: c_ast.pl gains `c_using_decl` and
qualified-template support — generally useful beyond this lifter.
Future C++ parsing benefits.

---

## Migration Unit 3: parse_tensor_name

### What it lifts

`TensorRef` from a tensor-spec string like `'LLM_TENSOR_ATTN_NORM, "weight", i'`.

Three branches:
- `layer(i, attn_norm, weight)` — layer tensor with iteration index
- `global(tok_embd, weight)` — global tensor (single instance)
- `unparsed(Atom)` — fallback

### C++ structure being matched

The input is a string captured from a C++ macro invocation argument
like:
```cpp
TENSOR(LLM_TENSOR_ATTN_NORM, "weight", i)
TENSOR(LLM_TENSOR_TOK_EMBD, "weight")
```

The captured argument string then needs to be tokenized into:
1. The `LLM_TENSOR_X` constant name
2. The string literal `"weight"`
3. Optional trailing `i` (means "layer-iterated")

### c_ast vocabulary needed

All present. The "argument string" is really a comma-separated list
of expressions:
- `c_var('LLM_TENSOR_ATTN_NORM')`
- `c_string("weight")`
- `c_var(i)`

### Decomposition into subtasks (4 bounded steps)

1. **Parse the captured argument string** as a comma-separated
   expression list via `c_parse_args` (a new helper, or use an
   existing one in c_ast). Returns `[Expr1, Expr2, ...]`.

2. **Pattern-match the parsed args** for the three cases:
   - `[c_var(NameAtom), c_string(PartStr), c_var(i)]` → layer
   - `[c_var(NameAtom), c_string(PartStr)]` → global with part
   - `[c_var(NameAtom)]` → global with default 'weight' part

3. **Lowercase the name** (strip 'LLM_TENSOR_' prefix, downcase)
   — keep this Prolog-side; just `atom_concat` and `downcase_atom`.

4. **Verify equivalence**: run both lifters on the existing tensor
   spec corpus; assert TensorRef matches for every input.

**Estimated effort**: 2-3 hours. Bounded by whether `c_parse_args`
exists — if yes, this is short.

**Substrate impact**: tensor specs now structurally lifted. The
parser gains (or reuses) a "parse a comma-separated arg list" helper
that benefits any future macro-argument lifting.

---

## Migration Unit 4: classify_op_call

### What it lifts

`op(OpKind, OpName)` from a function-call string like `'build_attention('`
or `'ggml_mul_mat('`.

Two branches:
- `op(build_helper, attention)` — calls to `build_X` helper functions
- `op(ggml_op, mul_mat)` — calls to `ggml_X` primitives

### C++ structure being matched

The input is the prefix of a function call expression. The full
C++ would be like:
```cpp
cur = build_attention(ctx, ...);
cur = ggml_mul_mat(ctx, w, x);
```

The CallStr captured is just `build_attention(` or `ggml_mul_mat(`
— the regex peels off the open-paren.

### c_ast vocabulary needed

All present: `c_call(FuncName, Args)`. The `FuncName` is a `c_var`
or atom.

### Decomposition into subtasks (3 bounded steps)

1. **Pattern-match the function name prefix**. Given a `c_call(Name, _)`
   AST node, check whether `Name` starts with `build_` or `ggml_`
   atom-prefix. Use `atom_concat(build_, Suffix, Name)` or similar.

2. **Extract the suffix** as the OpName atom; map to OpKind.
   Three-line Prolog.

3. **Verify equivalence**: existing classify_op_call output should
   match the AST-based output on every existing call expression in
   the lifter's corpus.

**Estimated effort**: 30 min - 1 hour. By far the smallest. The
"regex" here is really just a string-prefix check; once we have
the call as an AST node (which we usually already do at the call
site), the migration is trivial.

**Substrate impact**: Tiny. classify_op_call becomes a one-line
predicate. The "regex" wasn't really regex-y to begin with — it
was just using regex syntax for prefix-matching atoms.

---

## Aggregate effort estimate

  Unit                       | Effort       | Sites
  ---------------------------|--------------|--------
  1. classify_size_assignment| 1-2 hours    | 2 (sites 1, 2)
  2. scan_template_params    | 4-6 hours    | 2 (sites 3, 4)
  3. parse_tensor_name       | 2-3 hours    | 3 (sites 5, 6, 7)
  4. classify_op_call        | 30 min - 1 hr| 2 (sites 8, 9)

Total: 8-12 hours of work across all 4 units. Migration unit 2
(scan_template_params) is the substantial one because it requires
extending the c_ast parser; the others are pattern-match-against-
existing-parser.

## Recommended migration order (by leverage per hour)

1. **Unit 4 (classify_op_call)** first — 30 min, retires 2 regex
   sites, smallest cognitive cost to plan.

2. **Unit 1 (classify_size_assignment)** second — 1-2 hours, retires
   2 regex sites, exercises the parser on a realistic case.

3. **Unit 3 (parse_tensor_name)** third — 2-3 hours, retires 3
   regex sites, depends on whether `c_parse_args` exists.

4. **Unit 2 (scan_template_params)** last — 4-6 hours, the
   substantial parser-extension work. Worth doing AFTER the
   easier units validate that the migration pattern works.

This ordering retires regex sites quickly first; saves the
substrate-extension work for after the pattern is empirically
validated.

## Methodology principle

Each migration unit is a **bounded substrate-extension** — gives the
substrate a new capability while retiring an old regex. The retirement
is the cognitive-load payoff; the new capability is the compound
benefit (other lifters can use the same parser extension).

Per Heath's framing: "the leverage from forgetting the currently
in-mind regexes will compound way more dI/dt than the time will cost
to do this." The compound: each unit retired means future readers
don't head-compile that regex grammar anymore. Permanent.

## What this decomposition does NOT decide

Whether to actually execute the migration. The decomposition exists
as a queued substrate plan. The decision to execute is per-frontier-
choice; if F4 (curriculum extension) or F5 (kernel fusion) produces
higher subsumption velocity per hour, the regex retirement can wait.

The plan exists so that when the right moment comes, the work is
already structurally scoped. Future-me on resume reads this doc and
knows exactly which subtasks to attempt.

Author: metayen 2026-05-18 ~01:35 UTC
Per Heath's "enumerate the 9 substantial Prolog regex uses, and
consider for each how we would decompose it into subtasks."
