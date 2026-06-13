# AST Isomorphism: the Foundational Comparison Capacity

**Date**: 2026-05-17
**Originating conversation**: Heath's reframe of the Phase 5 metric question
after the post-3.1.e empirical results showed text-level DIFFs whose ASTs
compared as equal.

## The principle

When comparing two representations of a program, the foundational capacity
is **AST isomorphism** — equality of symbol names, equality of tree shape,
equality of substructure recursively. The capacity to do this comparison is
more important than the capacity to compare textual representations.

The string-output comparison is a **back-stop sanity check** on the more
important fidelity: data-structure fidelity, not string-representation
fidelity.

Two ASTs that compare equal MUST emit byte-identical text through any
deterministic canonical emitter. If they don't, the bug is non-determinism
in the emitter, not a difference in what the ASTs represent.

## Why this matters

Linear text is what speech-acts produce — humans linearize tree-structured
thoughts into sequential tokens for transmission. Two programs that mean
the same thing can have surface differences (brace placement, integer
literal radix, type qualifier ordering, trailing commas, etc.) that don't
affect what they mean. These surface differences appear as text-level DIFFs
but disappear at the AST level.

The Essence of a program lives in the **structure**: which expressions are
inside which statements, which statements are inside which functions, which
names refer to which declarations, which types resolve to which definitions.
The AST captures the first layer of this; eventually a graph representation
captures more (cross-references, type resolution, control flow, data flow).
Each upgrade preserves what came before and makes new questions askable.

## Empirical evidence from step 3.1.e

After integrating the c_preprocess pipeline into the c_ast extraction
layer, the Phase 5 round-trip showed 4 text-level DIFFs (bailingmoe2,
glm4, mimo2, phi3). Investigation revealed:

- Both sides preprocessed through the same GCC cpp.
- Upstream preprocessed text contained braceless `if (!(...)) ggml_abort(...);`
  (cpp's expansion of GGML_ASSERT).
- Our emit produced braced `if (!(...)) { ggml_abort(...); }` —
  semantically equivalent, textually different.

When we compared the ASTs directly (re-parse both texts, compare terms with
`==`), all 4 archs matched. Same AST term `c_if(Cond, [SingleStmt])` from
both sides. The substrate had captured the semantics correctly all along.

The 93/93 = 100% MATCH at the AST-isomorphism level is the truth. The
4 text-level DIFFs were a measurement artifact — comparing two surface
serializations of the same object.

## The two grades of representation faithfulness

This distinction is worth naming for future work:

### Grade A: Semantic faithfulness

The AST captures what the code MEANS. Two surface-different expressions
that mean the same thing become the same AST. Emit normalizes. Round-trip
is "semantically equivalent" not "textually identical."

**This is what the substrate provides today.** Sufficient for lifting,
transforming, generating, and inspecting code. The c_init_list term
doesn't remember whether you wrote `{a, b}` or `{a, b,}` — both are the
same list semantically.

### Grade B: Surface faithfulness

The AST captures the code's MEANING AND SURFACE. Two surface-different
expressions become different AST terms. Emit reproduces the original
surface choice. Round-trip is byte-equivalent modulo whitespace and
comments.

**This is what would be required for**:
- Patching upstream files via the substrate (emit back where we found it).
- Generating diffs against upstream that are reviewable line-by-line.
- Maintaining a fork using the substrate as the source of truth.

Grade B is a future direction, queued for when there's a concrete workload
that demands it. Adding it speculatively grows substrate complexity
without earning capability. Bounded extension per need.

## The canonical metric

Implementation: `bpd/phase5_sweep_ast_isomorphism.pl`.

Algorithm:
1. For each architecture's `load_arch_hparams`:
   a. Get our emit (via the AST lift+emit pipeline, which already
      preprocesses internally).
   b. Get upstream preprocessed text (via `extract_load_arch_hparams_preprocessed`).
   c. Re-parse both back into ASTs via `c_parse_stmts_v2_partial`.
   d. Compare AST term lists with `==`.
2. Report:
   - MATCH (ASTs equal)
   - DIFF (ASTs differ — real semantic divergence to investigate)
   - no_source / no_body / no_parse / timeout / error

Status codes:
- `match`: substrate captures the arch's semantics; both representations
  are the same object.
- `diff`: substrate captures DIFFERENT objects from the two
  representations. Real bug to investigate.
- `no_parse`: catastrophic parse failure (partial-parser returned empty).
  Indicates a c_ast vocabulary gap.
- `no_source` / `no_body`: arch doesn't exist in the corpus / function not
  found. Not a substrate concern.

## Why we don't enshrine text comparison as a separate metric

If the emitter is deterministic (same AST always produces the same text),
then text comparison of `emit(A)` vs `emit(A)` is trivially TRUE. It
doesn't tell us anything new.

If the emitter is non-deterministic, text comparison reveals it — but
that's a bug in the emitter, not a substrate-correctness question. We'd
fix the non-determinism, not measure it as a metric.

Text comparison between `emit(A)` and `text_source` (where text_source is
the original surface form) is what we'd want for Grade B faithfulness.
That's a different capability than the substrate currently provides; the
metric for it would live alongside the canonical AST isomorphism metric,
not replace it.

## Implications for substrate evolution

The substrate's primary commitment is **structural fidelity** — that the
AST term we lift from a program faithfully encodes the program's meaning.
Every test that exercises this — the c_preprocess invariant, the lifter
sweeps, Phase 5 AST isomorphism — is testing structural fidelity at some
level.

Future upgrades to the substrate that increase structural fidelity:

- **Cross-reference resolution**: link `c_var(name)` at use site to its
  declaration. Today: tree-local; same name in different scopes is the
  same term. Future: graph-shaped with edges.
- **Type resolution**: link `c_call(c_var(fname), args)` to the function
  declaration. Today: text-token. Future: edge to AST node.
- **Control flow**: capture the flow relationships between statements
  beyond textual order.
- **Data flow**: capture the dependencies between assignments and uses.

Each is a brick in the same direction Heath's SMG Quality World vision
names: a substrate that knows enough about code to answer questions about
it, not just emit and lift bytes.

The AST isomorphism comparison is the foundational capacity that every
upgrade preserves. As the substrate gets richer (tree → graph → graph with
cursors → activational), the comparison generalizes (term `==` →
graph isomorphism → semantic equivalence). The principle stays the same:
compare the structure, not the surface.

## See also

- `bpd/phase5_sweep_ast_isomorphism.pl` — the canonical metric implementation
- `bpd/docs/methodology/step-3.1-before-after.md` — quantitative before/after
  of step 3.1, including the moment the metric question was raised
- `bpd/docs/methodology/thin-filter-pattern.md` — the related Essence-school
  methodology for system tool integration

Author: metayen 2026-05-17
Per Heath's reframe: data-structure fidelity over string-representation
fidelity. The string output comparison is a back-stop on testing the more
important fidelity.
