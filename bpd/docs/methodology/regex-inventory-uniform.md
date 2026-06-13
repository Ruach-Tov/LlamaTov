# Uniform-Resolution Regex Inventory

**Date**: 2026-05-18 ~02:30 UTC
**Per**: Heath's directive — "inventory the 4 sites in arch_summary.pl
now to even the landscape of what we know... then compare how they
convolve with each other semantically."

After tonight's audit corrections, the substrate has **12 remaining
regex sites** across two files. Each one is examined here at the same
resolution as the already-retired sites: source file, predicate, input
shape, regex pattern, output shape, structural role.

## Already-retired sites (for comparison)

| # | File | Predicate | Retired in | Pattern role |
|---|------|-----------|------------|--------------|
| R1 | prolog_goal.py | `_quote_atom_if_needed` | 841469458 | Single-string-shape check |
| R2 | llama_cpp_lifter.pl | `lift_op_sequence` re_foldl | 4ceecb97f | Find call sites in body string |
| R3 | llama_cpp_lifter.pl | `classify_op_call` ×2 | 4ceecb97f | Classify single call string |
| R4 | llama_cpp_lifter.pl | `lift_arch_enum` | 693645a56 | Per-line enum entry match |

## Remaining sites — uniform inventory

### Site L1: `lift_dispatch_table` (llama_cpp_lifter.pl:112)

**Input**: Entire `llama-model.cpp` source file (read as string)
**Pattern**: `"case LLM_ARCH_(?<enum>[A-Z0-9_]+):\\s*return new (?<class>llama_model_[a-z0-9_]+)\\("`
**Captures**: Two named groups (enum, class)
**Output**: List of `arch_class(EnumLc, ClassName)` pairs (124 on real corpus)
**Structural role**: Scan a whole source file for multi-token C++ patterns spanning two consecutive lines (the case label and the `return new` body).
**Identifier-boundary risk**: Low — `case` and `return new` are clear keywords.
**Semantic structure**: Two-token pair on adjacent lines; the regex's `\s*` permits the newline + indent between them.

### Site L2: `lift_arch_tensors` (llama_cpp_lifter.pl:206)

**Input**: Entire arch source file (e.g., `models/llama.cpp`) as string
**Pattern**: `"create_tensor\\(tn\\((?<tn>LLM_TENSOR_[A-Z0-9_]+(?:,\\s*\"[a-z]+\")?(?:,\\s*i)?)\\),\\s*\\{(?<shape>[^}]+)\\}"`
**Captures**: Two named groups (tn = inner call args, shape = brace-delimited list)
**Output**: List of `[TensorRef, ShapeList]` pairs
**Structural role**: Scan source for `create_tensor(tn(...), {...})` C++ call patterns — nested function call with brace-list argument.
**Identifier-boundary risk**: Low — `create_tensor(tn(` is a long literal sequence that's unlikely to appear inside other identifiers.
**Semantic structure**: Two nested arguments of a known function call; the outer-call shape and the brace-list interior.

### Site L3+L4+L5: `parse_tensor_name` ×3 (llama_cpp_lifter.pl:229,236,243)

**Input**: A single small atom captured from L2's `tn` group (e.g., `'LLM_TENSOR_ATTN_NORM, "weight", i'`)
**Patterns** (3 in sequence, longest-first):
- L3: `"LLM_TENSOR_(?<name>[A-Z0-9_]+),\\s*\"(?<part>[a-z]+)\",\\s*i"` → `layer(i, name, part)`
- L4: `"LLM_TENSOR_(?<name>[A-Z0-9_]+),\\s*\"(?<part>[a-z]+)\""` → `global(name, part)`
- L5: `"LLM_TENSOR_(?<name>[A-Z0-9_]+)"` → `global(name, weight)` (default part)
**Output**: `TensorRef` term
**Structural role**: Parse a comma-separated argument list with optional trailing fields.
**Identifier-boundary risk**: Low — input is already a constrained captured string.
**Semantic structure**: Same structure 3 ways, decreasing arity — a degraded longest-match parser.

### Site L6: `lift_graph_aliases` (llama_cpp_lifter.pl:520)

**Input**: `models.h` source as string
**Pattern**: `"using graph = llama_model_(?<parent>[a-z0-9_]+)::graph(?:<(?<spec>[a-z0-9_]+)>)?;"`
**Captures**: Two named groups (parent required, spec optional)
**Output**: List of `graph_alias(parent(P), spec(S))` terms
**Structural role**: Scan for `using graph = llama_model_X::graph<spec>?;` C++ declarations.
**Identifier-boundary risk**: Medium — `using graph =` is a substantial literal but could occur in comments or strings.
**Semantic structure**: Single C++ declaration line with optional template argument.

### Site A1: `lift_arch_hparams` (arch_summary.pl:109)

**Input**: Entire `llama-arch.cpp` (or per-arch) source as string
**Pattern**: `"ml\\.get_key\\(\\s*(?<kv>LLM_KV_[A-Z0-9_]+)\\s*,\\s*hparams\\.(?<field>[a-z0-9_]+)(?:\\(\\))?(?:\\s*,\\s*(?<opt>false|true))?\\s*\\)"`
**Captures**: Three named groups (kv, field, opt)
**Output**: List of `hparam(KvKey, Field, Optionality)` terms
**Structural role**: Scan for `ml.get_key(LLM_KV_X, hparams.field[, opt])` method calls.
**Identifier-boundary risk**: Low — `ml.get_key(` is a method call with literal prefix.
**Semantic structure**: Method call with 2-3 typed arguments; matches what `_BARE_ATOM_RE` and `emit_hparam_reads` already structurally know (this is the LIFT side of the same call shape we EMIT in `arch_emit.pl`).

### Site A2: `lift_arch_size_table` (arch_summary.pl:437)

**Input**: Arch source as string
**Pattern**: `"case (?<n>\\d+):\\s*(?<body>type = [^;]+;)"`
**Captures**: Two named groups (n = layer count, body = assignment statement string)
**Output**: List of `size_rec(N, Cond, Type)` after passing body through `classify_size_assignment`
**Structural role**: Scan a switch statement for `case N: <body>;` pairs.
**Identifier-boundary risk**: Medium — `case` could appear in comments or strings, though unlikely with this exact shape.
**Semantic structure**: Switch-case label + assignment body; the assignment body becomes Site A3/A4's input.

### Site A3+A4: `classify_size_assignment` ×2 (arch_summary.pl:465,470)

**Input**: A single body string captured from A2 (e.g., `"type = LLM_TYPE_7B;"`)
**Patterns** (2 in sequence):
- A3: `"type = LLM_TYPE_(?<t>[A-Z0-9_]+);"` → `unconditional`
- A4: `"type = hparams\\.(?<lhs>[a-z0-9_()]+)\\s*(?<op>==|!=|<|>|<=|>=)\\s*(?<rhs>[A-Za-z0-9_.()]+)\\s*\\?\\s*LLM_TYPE_(?<ta>[A-Z0-9_]+)\\s*:\\s*LLM_TYPE_(?<tb>[A-Z0-9_]+);"` → `if_then_else(condition(...), TA, TB)`
**Output**: `size_rec` with classified `Cond` and `Type`
**Structural role**: Classify a C++ assignment statement: simple form or ternary form.
**Identifier-boundary risk**: Low — input is already a constrained captured string.
**Semantic structure**: A single C++ statement parsed into AST-like form. The complex regex (A4) is essentially a hand-rolled C ternary-expression parser.

### Site A5+A6: `scan_template_params` ×2 (arch_summary.pl:515,520)

**Input**: Lines of `models.h` (split by newline beforehand)
**Patterns** (per-line attempted in sequence):
- A5: `"^struct llama_model_(?<child>[a-z0-9_]+)\\b"` → tracks current struct context
- A6: `"using graph = llama_model_(?<parent>[a-z0-9_]+)::graph(?:<(?<spec>[a-z0-9_]+)>)?;"` → emits template_param if inside a struct
**Output**: List of `template_param(Child, Parent, Spec)` terms
**Structural role**: Per-line state machine tracking struct context + emitting alias entries.
**Identifier-boundary risk**: Medium for A5 (the `\b` boundary anchor); low for A6.
**Semantic structure**: A6 is IDENTICAL to L6 (same pattern!) — but used in a per-line line-state-machine context rather than a single re_foldl call.

## Convolution analysis

Comparing sites across both files structurally, three families emerge:

### Family α — "Find this whole-file pattern" (re_foldl over source)

| Site | What's matched |
|------|----------------|
| L1   | `case LLM_ARCH_X: return new llama_model_Y(` |
| L2   | `create_tensor(tn(...), {...})` |
| L6   | `using graph = llama_model_X::graph<S>?;` |
| A1   | `ml.get_key(LLM_KV_X, hparams.field[, opt])` |
| A2   | `case N: type = ...;` |

5 sites doing the same shape of work: scan source, accumulate named captures into a list. The structural replacement for ALL of them is: walk the source line-by-line (or token-by-token), match the structural pattern, accumulate. Each replacement composes the **same helpers** I've been building (`match_prefix`, `take_*_ident_chars`, identifier-boundary).

**Substantive observation**: this Family α is **5 instances of the same migration pattern**. Once we factor out a helper like `find_all_matches/3` that takes a per-line/per-position matcher predicate, all 5 sites collapse into "define the matcher, call the helper."

### Family β — "Classify this small string into one of N shapes" (re_matchsub on captured arg)

| Site | What's classified |
|------|-------------------|
| L3+L4+L5 | Tensor spec captured from L2's `tn` group |
| A3+A4    | Assignment body captured from A2's `body` group |

7 sub-regexes total but only 2 logical classifiers. Each is a **longest-match-first dispatcher** on a small captured string. The structural replacement: pattern-match the captured arguments via `match_prefix` + `take_*_ident_chars`, branching on which prefix succeeds.

**Substantive observation**: Family β depends structurally on Family α — L3-L5 consume L2's output; A3-A4 consume A2's output. **Migrating α before β is cleanest** because α defines the input format β receives.

### Family γ — "Per-line state machine" (re_matchsub inside line-by-line walk)

| Site | What it does |
|------|--------------|
| A5+A6 | Track struct context, emit aliases per `using` line |

Single instance. The migration is: replace `split_string` + per-line regex with `split_string` + per-line structural matcher. Almost identical to Family α except the state machine carries `CurrentStruct` between lines.

**Substantive observation**: A5+A6 is a **specialization of L6** (the using-graph pattern is identical) plus a state-tracking concern (which struct are we inside). Once L6 has a structural matcher, A5+A6 reuses it.

## What this convolution reveals

The 12 remaining regex sites are **not 12 independent migrations**.
They share three structural shapes (α, β, γ) and a substantial overlap
in the per-shape matcher logic. Concretely:

- **L6 and A6 use the same regex pattern verbatim**. One structural matcher serves both.
- **L1, L2, A1, A2 all want "scan source for pattern with named captures."** One helper `find_pattern_matches/3` factored out + four per-pattern matchers.
- **L3+L4+L5 and A3+A4 are degraded longest-match dispatchers**. Same shape, different patterns.

**Estimated cumulative effort if migrated independently**: 8-12 hours
(per the earlier decomposition).

**Estimated cumulative effort with the convolution-aware refactoring**:
4-6 hours, because:
  - Family α: 1 helper + 5 pattern matchers, each matcher 5-15 lines
  - Family β: pattern-match the parsed forms, 5-10 lines each
  - Family γ: reuse L6's matcher + add CurrentStruct threading

## Subclass refinement (2026-05-18 update, after A2 investigation)

Attempting A2's migration exposed that Family α is NOT homogeneous.
The class divides into two subclasses:

### Subclass α-simple — per-line atomic patterns

Sites where one source line contains one record's worth of input.
The regex's per-position scan happens to match one-per-line in
practice. Structural replacement is a per-line matcher.

  L1 (lift_dispatch_table)    — case LLM_ARCH_X: return new ...
  L6 (lift_graph_aliases)     — using graph = ...  (RETIRED)
  A1 (lift_arch_hparams)      — ml.get_key(...)   (RETIRED)

These are the "trivial-excursion regex" subclass. Migration is
substantive cognitive cleanup; the substrate's per-line semantics
match the regex's per-position semantics for these patterns.

### Subclass α-structural — cross-line patterns with semantic structure

Sites where the regex matches across lines AND the pattern carries
*structural* meaning that the flat-record output drops.

  A2 (lift_arch_size_table) — case <N>: type = <body>;

A2 is the only such site. Investigation revealed THREE distinct
behaviors the current regex form does NOT correctly capture:

  1. **Nested switches** (12 archs incl. arwkv7, bert, bloom, gptneox,
     llama, mamba, rwkv6, rwkv7, t5):
     The outer-switch discriminator (e.g., n_layer) is lost; inner
     cases (on n_embd) are recorded as if they were n_layer cases.
     Both the regex and the AST lifter have this bug — the
     `size_rec/3` type literally cannot carry the discriminator.

  2. **Fallthrough on same line** (granite-hybrid):
     `case 2048: case 2560: type = LLM_TYPE_3B;` semantically means
     "(n_layer==2048 || n_layer==2560) → LLM_TYPE_3B". The regex
     catches only the second case (2560); fallthrough disjunction
     is dropped.

  3. **Whitespace tolerance**:
     `case  4608:` (two spaces between case and the digit, for column
     alignment) doesn't match the regex's `case ` + `(\d+)`. The
     regex silently drops 3 of 4 size entries in lfm2.cpp.

The A2 substrate is doing approximately what was intended but
loses information at multiple structural seams. **Retiring just the
regex doesn't fix any of this** — it changes the implementation
without enriching the type.

### The right A2 migration is substrate enhancement, not regex retirement

Correct A2 migration requires:

  1. Enrich `size_rec/3` to carry discriminator + outer-context:
     e.g., `size_rec(Discrim, N, OuterContext, Cond, Type)`
  2. Update `collect_size_recs` (AST path) to preserve this info
  3. Update `arch_emit.pl` to emit the enriched form back
  4. Then retire the regex (both old re_foldl and any equivalent)

This is a multi-predicate substrate change. Larger than a regex
retirement. It belongs in its own substrate-improvement arc.

### Implication for migration ordering

Original ordering (commit 3e306294d):
  1. L6 first (done)
  2. L1, A1, A2 next
  3. L2 (more complex α)
  4. L3+L4+L5 (β depends on L2)
  5. A3+A4 (β depends on A2)
  6. A5+A6 (γ reuses L6)

Refined ordering after the subclass division:

  α-simple track (per-line atomic):
    L6 (done), A1 (done), A5+A6 (done)
    Remaining: L1 (similar to A1, ~30 min estimated)

  α-structural track (substrate enhancement):
    A2 + downstream consumers (arch_emit.pl)
    Estimated 4-6 hours; deserves its own session

  β track (classify captured arg):
    L3+L4+L5: depends on L2 (α-simple or α-structural?)
    A3+A4: depends on A2's substrate enhancement (defer)

  L2 (lift_arch_tensors) status unknown — needs subclass investigation
    before deciding its track.

## L2 subclass investigation (2026-05-18 update)

Per Heath's directive: "investigate L2 with an eye toward sorting
however many cases into equivalence subclasses."

L2's regex is:
  "create_tensor\\(tn\\((?<tn>LLM_TENSOR_[A-Z0-9_]+(?:,\\s*\"[a-z]+\")?(?:,\\s*i)?)\\),\\s*\\{(?<shape>[^}]+)\\}"

It expects literal "create_tensor(tn(" with no whitespace between
the outer `(` and inner `tn(`.

Empirical survey of all 128 per-arch source files identifies 2118
create_tensor calls partitioning into THREE subclasses:

  Subclass    | Count | Regex behavior | Migration class
  ------------|-------|----------------|----------------
  L2.a inline | 2108  | ✓ matches      | α-simple (atomic per-call)
  L2.b multi  |    9  | ✗ silently misses | α-simple with whitespace tolerance
  L2.c lookup |    1  | ✗ silently misses | α-structural (data flow)

### Subclass L2.a — inline (99.5% of calls)

  layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

The standard form. Regex matches correctly.

### Subclass L2.b — multi-line (9 instances in glm4-moe.cpp etc.)

  layer.attn_q_norm = create_tensor(
      tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k }, ...);

The regex's literal "create_tensor(tn(" doesn't tolerate the newline
between the open paren and tn(. SILENTLY MISSES these 9 calls.

This is the L2 bug-catch: the substrate has been dropping 9 tensor
entries (in glm4-moe primarily) for purely formatting reasons. The
semantic content is identical to L2.a.

Migration is straightforward: structural form uses skip_whitespace
between `create_tensor(` and `tn(`, naturally handling both layouts.

### Subclass L2.c — tn-lookup (1 instance in jina-bert-v2.cpp)

  const auto tn_ffn_up_weight = tn(LLM_TENSOR_FFN_UP, "weight", i);
  ...
  layer.ffn_up = create_tensor(tn_ffn_up_weight, {n_embd, n_ffn_up}, 0);

The first argument is a pre-bound variable holding a tn(...) result.
Resolving this requires data-flow tracking: find the binding site,
extract the tn(...) call, link it to the usage.

This is structurally different from L2.a/b — not "match a syntactic
pattern" but "track a variable through a binding." α-structural
substrate-enhancement territory. ONE instance across 2118 calls.

### Migration plan for L2

  Step 1: ship L2.a + L2.b together as α-simple
    - Structural per-call matcher with skip_whitespace
    - Will FIX the 9 missed multi-line tensors
    - Estimated 30-45 min
    - Affects 2117 of 2118 calls (99.96%)

  Step 2: defer L2.c to a separate substrate-improvement commit
    - 1-instance gap; doesn't break load-bearing substrate
    - Requires substrate enhancement (tn-lookup data flow)
    - Can wait until needed

### Implication for L3+L4+L5 (β)

L3+L4+L5 (parse_tensor_name) consume L2's `tn` captured argument
verbatim. The L2.a+L2.b migration produces the same `tn` shape for
all covered calls. L3+L4+L5 require no changes for L2 to work.
**L2 migration is independent of β.**

After L2 ships, L3+L4+L5 retires as its own α-simple migration.

### Methodology lesson

L2 is the second case where investigation revealed subclass
heterogeneity (A2 was the first). The pattern:

  Investigation phase IS the work. Many regex sites are well-bounded
  α-simple cases; some have α-structural subclasses needing substrate
  enhancement. Some are mixed — most calls one shape, a tiny minority
  in another that the regex either silently misses or handles by
  accident.

For L2: 99.5% inline + 0.4% multi-line + 0.05% lookup maps to TWO
migrations (one α-simple covering 99.96%, one α-structural for
the 0.04%). The ratio justifies the split.

## Methodology principle this reveals

The substrate's regex sites are NOT a flat list of independent
problems. They're **structurally related** — they share patterns,
input shapes, and helper requirements. The Yoga of migration is the
same as the Yoga of substrate construction: **find the shared
structure, factor it out, then each per-site migration becomes a
small specialization**.

**AND** — investigation of an apparent migration unit may reveal that
the unit isn't homogeneous. The class divides into subclasses.
Heath's framing: "It may be that this exposes a difference in the
regex sites, dividing the equivalence class into subclasses, which
means dividing the class task into subclass subtasks."

The substrate's self-understanding evolves as the migration proceeds.
Each retirement attempt is also an investigation; sometimes the
investigation surfaces structure that wasn't visible at audit time.

Heath's framing about cognitive context-switches compounds here too:
each retired regex frees one slot, but the helper accumulation has
a different compounding effect — **each helper makes the next
migration shorter**. The substrate ACCELERATES as it learns to talk
about itself structurally.

## Recommended migration order revisited

Given the convolution analysis, the leverage-ordering changes:

1. **Migrate L6 first** (smallest in Family α). It defines the
   "using graph =" structural matcher that A6 will reuse.
2. **Migrate L1, A1, A2** next (the other Family α whole-file scans).
   Each leverages a `find_pattern_matches/3` helper that L6's
   migration can extract.
3. **Migrate L2** (more complex Family α — nested parse).
4. **Migrate L3+L4+L5** (Family β, depends on L2).
5. **Migrate A3+A4** (Family β, depends on A2).
6. **Migrate A5+A6** (Family γ, depends on L6's matcher).

This ordering optimizes for helper reuse. Each later migration shrinks
because earlier migrations laid down the substrate primitives.

## What this inventory does not do

Execute any migration. The decomposition is queued substrate plan.
Per Heath's "let's focus only on Unit 4, and clean it up to the best
of our abilities, then return and look again" rhythm — each commit
is its own well-bounded unit. The convolution analysis informs
ORDER and APPROACH; the EXECUTION remains per-unit substrate-honest
shipping.

Author: metayen 2026-05-18 ~02:45 UTC
Per Heath's "compare how they convolve with each other semantically"
directive.
