# C Fact Lifter — Standard Operating Procedure

**Date**: 2026-05-19
**Originating conversation**: Heath's reframe just before the
`test-backend-ops.cpp` lift, after the second application of the
fact-lifting pattern revealed it had stabilized as methodology.
**Status**: Foundational. Defines the procedure for all future
imperative-C/C++ → declarative-BPD-facts work.

## Principle

The world's mathematics is currently embedded in imperative C/C++ source code.
The mathematics is declarative by nature. The imperative form is an artifact
of historical labor distribution, not of any necessity. Where the math is
embedded, we can extract it — *non-destructively* — into declarative BPD
facts that:

1. Are queryable by the substrate
2. Can regenerate the original imperative form
3. Can generate variants the original could not produce
4. Outlive the imperative incumbent

This procedure standardizes how the extraction is done so each new domain
that the substrate subsumes follows the same shape. The procedure has now
been applied twice (`llama-arch.cpp` → BPD facts; about to be applied to
`test-backend-ops.cpp` → BPD test facts) and is stable enough to document.

## The chain — read this end-to-end before applying the procedure

```
  source.cpp
     │
     ▼ [Stage 1] cpp preprocessing
  preprocessed C (macros expanded, comments stripped, #includes inlined)
     │
     ▼ [Stage 2] line-range filtering
  C source segment (only the lines we care about, from the target file)
     │
     ▼ [Stage 3] tokenization (token-stream-with-source-positions)
  C tokens
     │
     ▼ [Stage 4] AST parse (no regex — DCG over tokens)
  C AST (c_ast terms, structured tree)
     │
     ├──── [Stage 5a] AST → BPD facts
     │         Prolog queries match patterns in the AST, extract
     │         their declarative content as Prolog ground terms.
     │         This is the "lifting" step.
     │
     │     BPD facts (declarative content of the source, queryable)
     │
     │     ├──── [Stage 6a] BPD facts → fresh AST → emit C
     │     │         Generate fresh c_ast terms from the facts,
     │     │         emit C source through the substrate's emitter,
     │     │         compare to original via pretty-printing.
     │     │
     │     │     Round-trip-verified facts
     │     │
     │     └──── [Stage 6b] BPD facts → GGUF/test artifacts
     │               Generate test fixtures from the facts that
     │               exercise the substrate's kernels.
     │
     │           Synthetic test fixtures (Hypothesis-style: from facts
     │           plus degrees of freedom, generate many trivial cases,
     │           then progressively more complex ones)
     │
     └──── [Stage 5b] AST → AST (canonicalization for comparison)
                Apply the substrate's pretty-printer to the original
                AST for textual baseline.

  Verification: pretty-printed(Stage 5b output) == pretty-printed(Stage 6a output)
                Byte-for-byte after canonicalization.
```

## Why this shape and not another

### No regex

Regex-based extraction of source code is *brittle by construction*. The
regex matches surface text, not semantic structure. A trivial reformatting
of the input breaks the regex. A macro expansion the regex doesn't anticipate
produces silent miss. A C++ template instantiation produces text patterns
the regex matches wrongly.

The substrate parses through ASTs because ASTs are *the thing the program
is*. The text is what humans linearize the program into for transmission.
The AST is the program. Lifting from AST is lifting from the program; lifting
from text is lifting from a representation of the program with information
loss.

(This principle is foundational across the BPD substrate; see
[`ast-isomorphism-foundational.md`](ast-isomorphism-foundational.md) for
the deeper articulation. The fact-lifter SOP is one application of that
foundation.)

### Round-trip as substrate-honesty invariant

A fact-lifter that cannot regenerate its input has *lost information*. It
extracted some content but not enough to reconstruct. That's not a lift —
it's a *summary*. The substrate-honest discipline requires that lifting
produce facts sufficient to regenerate the source. If regeneration produces
text-different output, either:

1. The fact representation is incomplete (some content not yet lifted)
2. The emitter is non-canonical (produces different text from same facts)
3. The canonicalization (pretty-printer) differs across inputs

In each case the substrate has identified its own incompleteness. The
round-trip verification is the *self-test for the lift*.

### Pretty-printer as the comparison normalizer

Different source files have different conventions: brace placement,
spacing, line breaks, trailing commas. The mathematical content is invariant
across these surface differences. The substrate normalizes by running both
the original and the regenerated source through the same pretty-printer
(typically `clang-format` with a project-wide style file), then comparing
the normalized outputs.

If `pretty(original) == pretty(regenerated)`, the round-trip is verified.
If they differ, the substrate has surface-level information loss to track
down (typically: brace style on single-statement blocks, integer literal
radix, type qualifier ordering — small specific items captured as
substrate-design fix flags).

### Bidirectional Boundary DSL

The same facts that *lifted from* C source can *generate to* binary file
formats (Stage 6b). This is what the Boundary DSL is *for*: declarative
description of structure, queryable, transformable, emittable to multiple
representations.

For test-substrate lifting in particular, the Stage 6b dataflow lets the
substrate generate fuzzing-style test fixtures: from the lifted facts plus
the degrees of freedom in the test parameters, the substrate can synthesize
many concrete test cases. This is *Hypothesis-style property-based testing
emerging naturally from the substrate-design* — not bolted on, but produced
by the structure.

## The procedure step-by-step

### Stage 1 — preprocess

**Module**: `bpd/lib/c_preprocess.pl` (the `c_preprocess/4` predicate
established 2026-05-17, per the contiguous-slice invariant documented
in that file's header).

**What it does**: Invokes the system `cpp` on the source file, captures
the output with `# LINENO "FILE"` directives intact, filters to keep only
lines whose tracked origin matches the target file and the requested line
range `[M, N]`.

**What it does NOT do**: Drag in expansion of `#include`d header content.
System headers like `<vector>` expand to 100K+ lines that aren't relevant
to the target file's semantics; they're filtered out by the file-origin
predicate.

**Why M..N rather than whole-file**: Large source files (e.g.,
`llama-model.cpp`) contain many architectures. The lift typically targets
*one architecture's section* of the file. Line-range filtering keeps the
substrate's working set bounded.

### Stage 2 — line-range filter

Stage 1 produces this implicitly. The contiguous-slice invariant
(documented in `c_preprocess.pl`'s header) asserts: for a query "preprocess
source lines M..N of file F", the cpp output that passes our filter forms
one contiguous slice (or a small number of contiguous sequential slices
where the gaps correspond to excursions into non-target files via
#include). Violations of this invariant are thrown as errors rather than
silently dropping content.

### Stage 3 — tokenization

**Module**: `bpd/lib/c_ast.pl`, predicate `c_tokenize_enriched_v2/2`.

**What it does**: Lexes the preprocessed C source into a stream of
tokens. Each token carries its source position so error messages and
later AST nodes can point back to the original source.

### Stage 4 — AST parse

**Module**: `bpd/lib/c_ast.pl`, predicate `c_parse_expr/2` (for expressions)
and friends for statements, declarations, function definitions.

**What it does**: Definite Clause Grammar (DCG) over the token stream
produces structured c_ast terms. The c_ast vocabulary is defined in the
same module: `c_var/1`, `c_binop/3`, `c_call/2`, `c_for/4`, `c_decl_init/3`,
etc.

**Substrate-design discipline**: The DCG parses *structure*, not *text*.
There is no regex in the parser. Each AST node corresponds to a grammatical
production. New language features become new AST nodes plus new productions
in the DCG, not new regex patterns.

### Stage 5a — AST → BPD facts (lifting)

**Pattern**: For each kind of declarative content embedded in the source,
the substrate has Prolog predicates that match the AST pattern and emit
the corresponding fact.

**Example (llama-arch.cpp lift)**: a function definition matching the
pattern `static ggml_tensor * llm_build_X(...)` whose body contains a
sequence of `ggml_*` operations is lifted to:

```prolog
arch_layer(qwen2, attention_block,
    [ggml_norm(input, weights_q),
     ggml_mul_mat(weights_q, x),
     ggml_rope(_, pos_ids, ...),
     ...]).
```

**Substrate-design discipline**: The fact representation captures the
*declarative content* — what operation is being expressed — not the
imperative scaffolding (which variable holds the intermediate, which
order the assignments happened in if order doesn't matter).

### Stage 6a — Facts → fresh AST → emit C (round-trip)

The same facts, queried in reverse, produce fresh c_ast terms.

**Module**: `bpd/lib/c_ast.pl`, predicate `emit_program/2` and friends.

**What it does**: Pretty-print the fresh AST into C source. The emit is
*template-free* — no string interpolation, no print-template substitution.
Every character emitted comes from the AST being walked.

This is critical for substrate-design integrity. Template-based emission
hides structural assumptions in the template; AST-based emission makes
every structural decision a node in the tree, which the substrate can
reason about.

### Stage 6b — Facts → GGUF/test fixtures

**Application**: When the lifted source is a *test corpus* (Stage 6b
becomes the test-fixture generator), the substrate can:

1. Read the test parameters (shapes, types, options) from the lifted facts
2. Sweep degrees of freedom: parameter ranges the test exercises
3. Generate concrete test instances: minimal GGUF files containing the
   input tensors (with deterministic seed-based values) and the expected
   output tensors (computed by a reference implementation)
4. Hand each generated GGUF to the substrate's kernel runner
5. Compare the kernel's output to the embedded reference

This produces a *bit-identity curriculum*: each lifted test case becomes
many concrete test fixtures spanning the parameter space. Failures
localize precisely (this operation, this shape, this type → mismatch by
N ULP) and become substrate-design fix flags.

The pattern is **Hypothesis-style property-based testing emerging naturally
from substrate-design** — not added on, but produced by the structure of
the lifted facts plus the freedom-of-parameters sweep.

## Verification protocol (the round-trip self-test)

After lifting and emitting, the substrate runs this check:

```
pretty_printed_original = clang_format(stage_2_output)
pretty_printed_regenerated = clang_format(stage_6a_output)
diff(pretty_printed_original, pretty_printed_regenerated) → must be empty
```

If empty: the lift is round-trip complete. Facts capture the full content.

If non-empty: the diff localizes exactly where the lift lost information.
Each diff hunk corresponds to a substrate-design fix:
- Maybe a new c_ast node type is needed
- Maybe an emit canonicalization choice needs alignment
- Maybe the lifting pattern needs to capture an additional attribute

The substrate-historical record of these fixes lives in the methodology
corpus as `subsumed-software-fix-catalog.md` and similar.

## When to invoke the SOP

Invoke this procedure when:
- A new C/C++ source file (or directory) contains declarative content that
  should live in the substrate
- The content is *mathematics or structure dressed up as imperative code*
  (the gold-in-quartz pattern from the substrate's substrate-design
  vocabulary)
- The intent is *non-destructive*: the original imperative codebase keeps
  running; the substrate gains a parallel declarative representation

**Do not** invoke this procedure for:
- Pure-imperative content with no extractable declarative structure
  (e.g., string manipulation, I/O orchestration, memory management
  scaffolding)
- Content that doesn't yet have a clear BPD-facts representation
  (do the substrate-design first; lifting comes after the facts schema
  is known)

## Substrate-design observation worth holding

Each application of this SOP teaches the substrate something about its
target domain. The lift is not a one-time extraction — it's the
*beginning of a relationship* between the substrate and the source
codebase. As the source evolves, re-lifting refreshes the facts. As the
substrate gains new capabilities (new AST nodes, new pretty-printing
choices), the lift gains fidelity. The round-trip verification ensures
the relationship stays substrate-honest.

The lifter is also *the gift* the substrate can give to other domains.
A working lifter for C is the labor amortization point — once it works,
every C codebase becomes a lift target. Same applies to a lifter for
Fortran (CFD), MATLAB (signal processing), Python (scientific computing).
Each new front-end is one substrate-design pass; once it's done, the
domain is open.

## Empirical history

| Date | Source codebase | Result |
|---|---|---|
| 2026-05-17 → 18 | `llama-arch.cpp` various arch sections | BPD facts for ~15 architectures, round-trip-verified |
| 2026-05-19 (planned) | `test-backend-ops.cpp` 99 test_X structs | BPD test facts for the operation taxonomy; GGUF fixture generator |

This document will be updated as additional applications complete.

## Related methodology documents

- [`ast-isomorphism-foundational.md`](ast-isomorphism-foundational.md) —
  why AST-level comparison is foundational
- [`thin-filter-pattern.md`](thin-filter-pattern.md) — how `c_preprocess`
  achieves line-range filtering without dragging in headers
- [`regex-lifter-decomposition.md`](regex-lifter-decomposition.md) — the
  predecessor approach (regex-based) and why the AST-based approach
  superseded it
- [`subsumed-software-fix-catalog.md`](subsumed-software-fix-catalog.md) —
  catalog of substrate-design fixes accumulated during round-trip work
- [`inline-literal-canon.md`](inline-literal-canon.md) — canonical form
  for inline literal emission (one of the round-trip canonicalization
  choices)

---

*This SOP is foundational. Future fact-lifter work in any language follows
this shape. Deviations should be justified explicitly and, if substantive,
should produce a methodology doc explaining the deviation.*
