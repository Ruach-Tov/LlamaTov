# C Fact Lifter — Lessons Learned

**Date**: 2026-05-19
**Originating conversation**: Heath asked me to organize substrate-historical
learnings from prior fact-lifter sessions before starting the
`test-backend-ops.cpp` lift, so the fresh work proceeds informed by what
already worked and what bit us before.
**Status**: Companion to [`c-fact-lifter-sop.md`](c-fact-lifter-sop.md).
The SOP defines *the procedure*; this doc captures *what to watch for*
while applying it.

## Purpose

The SOP describes the shape of fact-lifter work. This document captures
the specific traps, performance constraints, and methodology corrections
that prior lifter sessions surfaced empirically. Read this *before*
starting a new lift target so you don't rediscover hard-won learnings.

## Substrate-historical record (what's been done)

### Application 1: `llama_cpp_lifter.pl` (commits `0171a5a0b` + `778281fcb`)

Lifted from `external/llama.cpp/`:
- 127 LLM_ARCH_* enum entries
- 124 dispatch-table arch→class pairs
- Per-arch tensor declarations + graph op-sequences
- 16 graph_aliased relationships
- 108 unique graph implementations after alias collapse

Empirical: 124/124 archs lift successfully, avg 37.27 BPD facts per arch.
Reproducible from fresh clone.

### Application 1.5: qkv lifter (commit `6cdec415d`, follow-up `c515b085a`)

Lifted qwen2.cpp Q-projection through the full loop:
```
real C source
  → c_parse_stmts_v2 (DCG parser)
  → AST term list
  → qkv_lifter:lift_qkv_section
  → BPD facts
  → gen_qkv_from_bpd
  → regenerated C
  → clang-format diff = ∅
```

**Critical milestone**: PROLOG-TERM-IDENTITY achieved (not just string
equality, not just clang-format equivalence). `Orig == Regen` as Prolog
terms. Lifter and generator are exact inverses. 82 tests across both
substrates.

### Application 1.7: Regex retirement campaign (2026-05-18, ~01:00-04:00 UTC)

13 substantial regex sites identified at audit-start. 6 retirements
shipped in one session across 5 commits. Each retirement converted
text-pattern matching into structural primitives.

## Substrate constraints to heed (these will bite you again)

### Constraint 1: `c_parse_stmts_v2_partial` is slow on large bodies

**Empirical bound**: >120s on 4.3KB `load_arch_tensors` bodies. Works
~2s on `load_arch_hparams` bodies (smaller, simpler structure).

**Where the slowness lives**: braced init lists, complex shape
expressions, for-loops over layers. The parser backtracks heavily on
these constructs.

**Substrate-honest workaround**: **Path B** — per-statement AST after
text extraction. Don't try to parse whole function bodies as one AST.
Extract the substantive text segment, then parse statement-by-statement
with `c_parse_stmt/2` (singular).

**When you're tempted to do Path A**: you'll think "I'll just parse the
whole function and walk the tree." For small bodies (<1KB) this works.
For larger bodies (>2KB with complex C++) you'll hit the bound and the
substrate appears to hang. The signal: swipl process at 100% CPU,
no output for >30s. That's the parser backtracking.

**Test-backend-ops.cpp implications**: each `struct test_X` is typically
small (~50-100 lines), so per-struct parsing should fit Path A. The
overall 9,773-line file would NOT fit Path A — parse it by extracting
each struct's source range first, then parse each struct individually.

### Constraint 2: AST nodes the substrate doesn't yet have

Each new lift target will expose c_ast vocabulary gaps. The qkv lift
exposed 7 gaps (control flow, scoping, ctx0 threading, inline vs named,
callback layer index, type info, op_level). The arch lift exposed enum
declaration parsing.

**Discipline**: when you hit a gap, **extend c_ast minimally**. Add the
specific node type you need; don't try to anticipate other future
needs. The substrate's expressiveness grows organically per-application.

**Test-backend-ops.cpp likely gaps**:
- `std::array<int64_t, 4>` member initialization (template + braced init)
- `enum class` modifiers (the ggml enums use scoped enumeration)
- Constructor member-initializer lists (`test_rope(...) : type(...), ne_a(...)`)
- `override` keyword on virtual methods
- Lambda expressions (some tests use them in build_graph)

Each is one bounded extension to c_ast when encountered. Don't try to
add them up front.

### Constraint 3: The substrate has TWO C parsers

`c_parse_expr/2` for expressions (mature, well-tested) and `c_parse_stmt/2`
for statements (newer, more bugs). Statement parsing is where the
slowness lives. If you can lift via expression-level patterns instead of
statement-level traversal, the substrate handles you faster.

**For test-backend-ops.cpp**: most lifting can be done at the
**struct-field level** (parse each member declaration as an expression-
adjacent pattern) rather than parsing the full constructor body. The
parameters are mostly initializer-list members; the `build_graph` body
contains the actual graph construction but the parameters themselves
are extractable without parsing the body.

## Bug-catch categories (from the 2026-05-18 regex retirement campaign)

When you migrate from regex to AST or extend existing lifters, you will
trip over these. The substrate has now seen each at least once:

### 1. Mid-identifier overmatch

**Symptom**: regex `ggml_build_forward_expand` matches inside
`ggml_build_X_forward_expand` because the regex doesn't anchor at word
boundaries properly.

**Fix**: AST-based matching uses identifier-as-token, not substring.
Trivially fixed by the AST approach. Watch for it during regex-to-AST
migration: the AST output may *gain* matches the regex was silently
discarding, OR *lose* matches the regex was silently overcounting.
Either direction is substrate-design evidence about the original code.

### 2. Nested-switch discriminator drop

**Symptom**: a switch statement nested inside another switch loses
classification. Affected 12 archs (arwkv7, bert, bloom, gptneox, llama,
mamba, rwkv6, rwkv7, t5, ...).

**Fix**: ensure your traversal walks INTO nested control structures, not
just over them. Bug exists in both regex and naive AST traversal —
size_rec/3 type was missing the discriminator field at one level.

**Test-backend-ops.cpp relevance**: tests with switch on `ggml_type`
inside switch on backend will hit this if you don't recurse properly.

### 3. Fallthrough cases lost in disjunction

**Symptom**: `case A: case B: type = X;` — the `A` and `B` get treated
as a single disjunction, losing the structural information that both
labels point to the same statement.

**Fix**: explicit list of case labels, not joined into one match.

### 4. Whitespace alignment fragility

**Symptom**: regex `case ` (single space) fails on `case  4608:` (double
space). AST parsing is whitespace-insensitive.

**Fix**: trivially fixed by AST. Watch for if you fall back to regex for
any reason.

### 5. Third arg as literal int not captured

**Symptom**: regex pattern `(?:,\s*i)?` only matches literal `i`. Misses
calls like `f(x, 0)` or `f(x, n_layer)`. ~10 such cases in the source.

**Fix**: AST captures arbitrary expressions in arg positions. Don't
anchor your lift patterns on specific identifier names.

### 6. Multi-line patterns not matched

**Symptom**: `create_tensor(\n  tn(...))` — newline between function
call and arg list. Regex anchored to single line misses these. 9 cases
in glm4-moe.

**Fix**: AST is line-agnostic. The C parser handles arbitrary whitespace.

### 7. Pre-bound variables missed

**Symptom**: `tn_lookup_variable` pre-bound — when the variable is
assigned earlier and then passed, regex/text-shape lifting misses the
binding. 1 case in jina-bert-v2.

**Fix**: full data-flow analysis catches this. For the substrate's
current state, this means: if a fact requires knowing what a variable
points to, you may need to walk the surrounding scope, not just match
the local pattern.

## Methodology principles to apply (from prior session reflections)

### Pull-before-push on shared substrate

Before modifying shared lifter modules (`llama_cpp_lifter.pl`, `c_ast.pl`),
`git fetch` first. Other agents (mavchin, medayek) may have committed
changes during your session. Substrate-historical pattern: 2-3 commits
per hour during active multi-agent work.

### Empirical-state-check before plan-execution

Don't trust your model of the substrate's current state. Run a sweep.
For lifter work: `swipl -g 'consult(your_lifter), lift_target(corpus, F), length(F, N), format("~w facts~n", [N]), halt'` is a 5-second sanity check that catches "the substrate doesn't have what you assumed it had."

### Trivial-excursion vs load-bearing regex distinction

When auditing a regex for retirement, ask: is the regex doing
*substantive matching* (capturing meaningful structure) or is it a
*trivial excursion* (matching something obviously simple where regex is
the right tool)? Load-bearing regexes are the ones that hide
substantive substrate-design opportunities; trivial excursions can
stay as regex without methodology cost.

For test-backend-ops.cpp: identifier-matching ("does this struct name
start with `test_`?") is trivial-excursion. Parameter-extraction ("what
shape is this `std::array<int64_t, 4>`?") is load-bearing.

### Test primitives on trivial inputs before composing

(Heath, multiple sessions.) Before lifting a real `test_rope` struct,
write a test lifting a 3-line synthetic test struct. Verify the lift
mechanics work on the simple case. THEN apply to the real corpus.

The substrate has burned hours on lifters that worked on the trivial
case but failed on real inputs because the trivial case missed some
structural variation. Empirical verification at every level.

### Diagnose-reflect-remedy (Heath, on the root word of "remedy")

When the lift produces wrong output, don't immediately try to fix the
lifter. First diagnose what specifically went wrong: which input
pattern, which AST shape, which fact emitted. Then reflect on whether
the lifter's structure is right and the bug is local, OR the lifter's
structure is wrong and the bug is symptomatic. THEN remedy.

The substrate-honest discipline: bugs are sometimes information about
substrate-design choices being wrong. Treating every bug as "fix the
local thing" misses the larger pattern.

### Dispose with correct cases first; tidy them up; then focus on the cases requiring thoughtful remediation

(Heath, regex retirement campaign.) When a lifter works on 80% of inputs
and fails on 20%, don't agonize over the 20% first. Ship what works,
commit the partial substrate, then focus the substantive thinking on
the hard cases. The 80% being shipped reduces total-cognitive-load and
provides empirical scaffolding for the hard cases.

For test-backend-ops.cpp: most of the 99 test_X structs probably follow
a uniform pattern. Lift those first (fast, mechanical). The 5-10
structs that are oddballs get focused attention separately.

### Inspection-based hypotheses tend toward complexity; empirical truth tends toward simplicity

(Heath, multiple sessions.) If you find yourself constructing an
elaborate explanation for why something might be wrong, stop. Run an
empirical test on the simplest hypothesis. The substrate-historical
experience: most "complex" explanations turn out to be simple once the
right test is run.

### Cognitive improvement is the substrate-honest payoff; bug-catching is a happy bonus

(Heath, regex retirement campaign.) The reason to do AST-based lifting
is not primarily "catch more bugs than regex" — it's "make the
substrate able to reason about its own content structurally." Bug-catches
are a side effect; substrate-cognitive-capacity is the substantive gain.

## Prior success patterns (what's working that we can lean on)

### The thin-filter pattern (c_preprocess module)

The `c_preprocess/4` predicate uses a tiny filter on top of system `cpp`
to keep only target-file lines. The contiguous-slice invariant catches
violations as errors. **This is your Stage 1 toolkit; don't reinvent.**

For test-backend-ops.cpp: each `struct test_X` is a contiguous slice
within the file. `c_preprocess(file, struct_start_line, struct_end_line, output)`
gets you the substrate-ready text for one struct.

### The qkv_lifter / qkv_generator inverse pair

Built as exact inverses. Lifter and generator share the same BPD fact
vocabulary. Round-trip Prolog-term-identity verified. **This is your
model for test-backend-ops.cpp lift design**:

```
test-backend-ops.cpp source
  → backend_op_lifter
  → backend_test_case/N facts
  → backend_op_generator
  → regenerated test-backend-ops.cpp source
  → diff against original = ∅
```

Build lifter and generator together. Verify round-trip at every step.

### The c_ast bidirectional grammar

`c_ast.pl`'s DCG-based grammar works both directions: same rules parse
and emit. **You don't need to write new C-emit logic**; the existing
emit infrastructure (`emit_program/2`, `emit_stmt/2`, `emit_expr/2`)
handles every AST node type the substrate knows about. New AST nodes get
both parse and emit rules at once.

### Three-layer test-corpus architecture (from earlier in this session)

For test-backend-ops.cpp specifically:

**Layer 1**: lift test taxonomy into BPD facts (this session's work)
**Layer 2**: synthesize minimal GGUFs from test facts (Hypothesis-style)
**Layer 3**: run substrate's kernels against generated GGUFs, verify bit-identity

Layer 1 produces a substrate that other AI agents can query. Layer 2
produces test fixtures the substrate can run against itself. Layer 3
produces the bit-identity curriculum that drives substrate-honest kernel
development.

## Specific applicability to test-backend-ops.cpp

### Structural taxonomy

The 99 `test_X` structs share a uniform shape:
```cpp
struct test_X : public test_case {
    const ggml_type type;           // parameter declarations
    const std::array<int64_t, 4> ne_a;
    int n_dims;
    int mode;
    // ...

    test_X(...) : type(...), ne_a(...) {}  // constructor

    std::string vars() override { ... }    // parameter serialization

    ggml_tensor * build_graph(ggml_context * ctx) override {
        // imperative graph construction
    }

    double max_nmse_err() override { return 1e-6; }
};
```

The **declarative content** to lift:
- struct name
- parameter declarations (name, type, default if any)
- max_nmse_err value (the tolerance)
- build_graph operation sequence (what ggml ops, in what order, with what shapes)

The **imperative scaffolding** to discard:
- the constructor body (auto-generated from parameter list)
- the vars() serialization (auto-generated from parameter list)
- ctx parameter threading (substrate-historical artifact)

### Recommended start: `test-rope.cpp` first

**Why test-rope.cpp** (263 lines, focused) is the right Phase 1 prototype:
- Small enough to lift in one session
- Tests RoPE specifically — our substrate's recent rope_kernel lift has
  immediate empirical target
- Independent of test-backend-ops.cpp's harness machinery
- Establishes the lifter/generator pattern before attacking the 9,773-line
  central file

After test-rope.cpp:
- test-backend-ops.cpp selectively (rope, embed, matmul, rms_norm,
  softmax, silu — the ops we cover)
- test-gguf.cpp's HANDCRAFTED_* enum (the 31-variant GGUF validation
  taxonomy)
- Round-trip on the ggml-vocab-*.gguf artifacts (49 files + 15 .inp/.out
  pairs)

### Anticipated bug-catches

Based on prior categories applied to test-backend-ops.cpp's specific
content:

1. **Default parameter values in constructors**: `test_rope(int n_dims = 10, ...)` — the parameter declaration carries a default. The substrate needs to capture both the parameter and its default.

2. **Template instantiation of std::array**: `std::array<int64_t, 4> ne_a{1, 2, 3, 4}`. The c_ast may not have explicit template-instantiation support yet; likely needs extension.

3. **Member-initializer list**: `test_X(...) : type(GGML_TYPE_F32), ne_a({10, 5, 1, 1}) { }`. The substrate has constructor-init-list lifting from the qkv work, but verify it handles the `member(arg_or_expr)` form for each member.

4. **Overload resolution markers**: `vars() override` — the `override` keyword needs to be parsed and either preserved or discarded (substrate-design choice).

5. **String literal handling in vars()**: `vars()` returns a string built up by stream insertion. The substrate may treat the body as opaque (which is fine — vars() is auto-generated from parameters and doesn't need lifting).

## Verification gates (apply at each phase)

After lifting Phase 1 (test-rope.cpp):

**Gate 1: Lift completeness**
- Count of test_X structs in source == count of backend_test_case facts
- Sample 3 random structs; manually verify the lifted facts match source

**Gate 2: Round-trip term-identity**
- For each struct: `original_ast == regenerated_ast` as Prolog terms

**Gate 3: Round-trip pretty-print**
- For each struct: `clang_format(original) == clang_format(regenerated)`

**Gate 4: Substrate-generated fixture validity**
- For each lifted struct: the substrate can generate a minimal GGUF that
  exercises the operation
- The generated GGUF parses via our pure-Prolog parser
- The substrate's kernel emit can run against the fixture

If any gate fails, halt and diagnose before proceeding. Don't ship a
partial lift hoping the gaps surface later.

## Substrate-historical notes (worth holding in working memory)

- The c_ast parser distinguishes `c_parse_expr` (mature) from
  `c_parse_stmt` (newer, slower). Prefer expression-level lifting where
  possible.
- The `c_preprocess` module's thin-filter pattern works; use it for
  Stage 1.
- The substrate-historical pattern for round-trip is `Orig == Regen` as
  Prolog terms, with clang-format diff as the surface-level check.
- The "stop and reflect" discipline: when a lift produces wrong output,
  diagnose (which pattern, which AST shape) before remedying.
- Heath's pedagogical pattern: each correction installs a specific
  subskill. Treat each lifter-bug-and-fix as a subskill being installed
  in your own substrate-design judgment.

## What this doc is NOT

- Not a substitute for the SOP. Read the SOP first; this doc is the
  *what to watch for* companion.
- Not exhaustive. New lift targets will surface new traps. Update this
  doc as you encounter them.
- Not a critique of the substrate. The substrate-historical record is
  what it is; this doc captures the lessons so future work doesn't
  repeat the discoveries.

## Related methodology documents

- [`c-fact-lifter-sop.md`](c-fact-lifter-sop.md) — the procedure this
  doc complements
- [`ast-isomorphism-foundational.md`](ast-isomorphism-foundational.md) —
  why AST-level comparison is foundational
- [`thin-filter-pattern.md`](thin-filter-pattern.md) — Stage 1 toolkit
- [`regex-retirement-audit-2026-05-18.md`](regex-retirement-audit-2026-05-18.md) — the empirical campaign that generated several of these lessons
- [`subsumed-software-fix-catalog.md`](subsumed-software-fix-catalog.md) — catalog of substrate-design fixes accumulated during round-trip work

---

*Lessons learned compound across applications. This document is the
ledger. Read it before you start; update it when you finish.*
