# The Thin-Filter Pattern for System Tool Integration

A reusable methodology for integrating with complex system tools when we
need to lift facts from their output. Derived during step #4 of the
regex-to-AST migration (2026-05-17) while building `c_preprocess.pl` —
the layer that runs GCC's cpp and filters its output for our parser.

## TL;DR

When you need X from a complex system tool's output Y:

1. **Don't** reimplement the tool.
2. **Do** accept its output verbatim and write a thin streaming filter.
3. **Assert** at least one structural invariant the filtered data must satisfy.
4. **Validate** empirically against the actual corpus, not just synthetic
   tests.
5. **Iterate** — your specification converges through empirical signal.

The recipe is **generative**: it tells you how to find the right filter,
not just how to describe one you already found. The space of trivial
filters is small enough to enumerate exhaustively, prune implausible
candidates, empirically test survivors, formalize the winner.

## The shape of the problem

```
[opaque system tool] → [thin filter with invariant] → [our representation]
```

Three properties that make this work:

1. **The system tool is the oracle for its own correctness.** We don't
   reimplement, model internally, or duplicate its work. We accept its
   output as ground truth and only ask "of all the output it produced,
   which lines do we want?"

2. **The filter is structurally trivial.** Read line-by-line (or
   token-by-token, or record-by-record). Track a small explicit state.
   Decide per element whether to emit. No complex AST manipulation of
   the tool's output — just a state machine over its content stream.

3. **The invariant turns the tool's behavior into a falsifiable
   assertion.** We don't just trust the tool; we assert "the output we
   emit MUST have property P unless explicitly explained otherwise." If
   that assertion ever fails, we know either (a) the tool changed its
   behavior or (b) our model didn't cover a corner case. Either way,
   the failure is loud and points at the layer where the assumption
   broke.

The invariant doesn't have to be **comprehensive**. It just has to be
**violatable by interesting wrong behavior**.

## The seven-step recipe

1. **Accept the tool's output verbatim.** Don't parse, interpret, or
   reimplement.

2. **Define the minimal predicate that answers your question.** "Which
   lines do I want?" "Which records match my filter?" Stop when you can
   express your need in one sentence.

3. **Implement as a single-pass streaming filter with small explicit
   state.** Stateful where necessary, but the state should be small
   enough to fit in a single dict or term.

4. **Identify ONE structural property that must hold if your filter
   assumptions are correct.** Encode it as a runtime assertion. Examples:
   - "Output records are in monotonic order" (for sorted streams)
   - "Every emitted item has a tracked origin matching the target"
   - "Resource lifecycle is balanced (every open has a close)"

5. **Make assertion failure loud and diagnostic.** Throw with full
   context: what was expected, what was observed, what suggested-cause
   to investigate. The exception should give a future debugger enough
   information to find the divergence without rerunning the whole
   pipeline.

6. **Validate empirically against the actual corpus.** Synthetic tests
   verify mechanism; corpus sweep verifies assumptions. Synthetic tests
   are necessary but not sufficient — they only cover scenarios you
   thought to construct. Corpus sweep exposes scenarios you didn't
   know existed.

7. **Treat the corpus sweep as a regression-detection canary.** If the
   OK-rate drops on a future tool version, the wrapper surfaces the
   breakage before downstream code sees it.

## The generative move (Heath's key insight)

The recipe is **prospective**, not just retrospective. It generates
candidate solutions.

When faced with "we need X from complex tool output Y," the recipe
says: **enumerate the trivial filter candidates first**, before
designing anything specific.

For our cpp case, the candidate space was roughly:

| Candidate | State | Verdict |
|-----------|-------|---------|
| Identity (pass everything) | None | Rejected — 100k lines |
| Random-access by line | Buffer | Rejected — need structure |
| Streaming, stateless regex | None | Rejected — directives carry context |
| Streaming, file-tracking | (file) | Plausible |
| Streaming, file + line | (file, line) | **Chosen** |
| Streaming, full nesting | (stack) | Over-built |

Each candidate took maybe ten seconds to consider. The chosen one was
the cheapest filter that captured what we needed. Over-built candidates
got rejected because their additional complexity didn't earn additional
capability for our use case.

### Why enumeration is tractable

The trivial-filter space is finite-and-enumerable because the primitive
operations of a streaming filter are themselves few:

- **Pass-through** vs **drop** (per element)
- **Stateful** vs **stateless**
- **State updated by directives** vs **content** vs **both**
- **One-pass** vs **multi-pass**
- **Read all** vs **read up to threshold**

Cross-product these and you get maybe 30 candidate filter shapes. For
any given input/output type, most are immediately implausible and get
pruned. You're left with 2-5 candidates worth empirical inspection.

## The small-algebra observation

The filter is a **program transformation** over a small algebra. For
streaming-tagged-data, four operations are usually sufficient:

- **Project** to a subset of tags
- **Restrict** to a range
- **Reset** state at directive boundaries
- **Assert** invariants over the stream's structure

We used all four for the cpp preprocessor case. There weren't any other
categories we needed.

Other problem classes have their own small algebras:

- **Graph traversal**: project to relevant edges, restrict by
  reachability, summarize at nodes, assert acyclicity.
- **JSON tree extraction**: project to relevant paths, restrict by
  schema, normalize types, assert well-formedness.
- **Binary file decoding**: project to relevant offsets, restrict by
  section, decode primitives, assert checksum.
- **Syscall trace analysis**: project to relevant syscall family,
  restrict by PID, balance lifecycles, assert resource invariants.

The transformation space is small because each problem class has bounded
primitives. **Enumerate them, prune the implausible, empirically test
the survivors, formalize the winner.**

## Iterative self-education

You don't start with a specification and derive an implementation. You
start with intuition, implement a candidate, validate empirically, and
**discover what specification you actually wrote**.

For the cpp case, our journey was:

1. Picked the trivial filter intuitively.
2. Tested on a small example. (Pieces 1-2 unit tests.)
3. Swept across the corpus. (Piece 2 empirical sweep.)
4. Caught the case where intuition was wrong. (Suffix match false
   positive: "not_bert.cpp" matched "bert.cpp" because raw string
   suffix succeeded.)
5. Caught the case where the system tool's behavior wasn't what we
   modeled. (GCC emits `# N "FILE" 3 4` directives during macro
   expansion — directive lines were advancing OutputLine counter.)
6. Updated the filter's specification to match corpus-validated
   semantics. (Directives are zero-width in the OutputLine counter.)
7. Now we **know** what filter we wrote — not just what we
   **intended** to write.

The proof and the implementation co-evolved. The contiguous-slice
invariant went through 3 iterations before matching real cpp behavior.
The final invariant is provably what we needed; we arrived at it
through empirical signal, not formal derivation.

This is the substrate-honest principle: **specification converges,
doesn't get declared**. The corpus is the specification's source.

## The build-vs-buy inversion

Normally "build" means reimplementing the tool, "buy" means depending
on it. The thin-filter pattern is a **third option**: defer to the tool
for what it does well, wrap it with a thin verification layer for what
we need. Pay only for the wrapper.

The cost of the wrapper is small. `c_preprocess.pl` is about 230 lines
including comments and the invariant check. Reimplementing cpp would be
enormous — multi-year work to do well across vendor/version space (the
SMG end-state).

**Critically**, the wrapper composes with future replacement. When we
eventually build our own Prolog cpp, the wrapper interface
(`preprocess_file_segment/4`) stays exactly the same. We swap the
implementation. Tests still pass. The downstream parser doesn't notice.

That's the substrate-honest property — the abstraction is at the right
layer to survive both the cheap version and the expensive version.

## Connection to the SMG Quality World vision

Heath's "Subject Matter Guru" end-state — we know all about cpp
implementations across vendor/version/spec space, we cross-generate, we
templatize on spec differences, we have 100% branch coverage and
mutation testing — is built by walking exactly the thin-filter path.

You don't build SMG by reading specifications. You build SMG by:

- Writing thin filters
- Asserting invariants
- Sweeping corpora
- Watching what fires
- Refining

The path from cheap wrapper to SMG end-state runs through this
iterative empirical pattern. Each thin filter is a brick. The bricks
compose into a tower of vendor/version-aware substrate over time.

## When to apply this pattern

Any boundary where:

- You need a transformation an existing tool already does correctly
- The tool's output is too complex to consume directly (volume,
  format, metadata noise)
- You can express what you want as a small filter over the tool's
  output
- You can name at least one structural property that should hold if
  your filter assumptions are right

Concrete candidate problems where the pattern would fit:

- **`clang -ast-dump=json`**: filter to AST nodes from target file.
  Invariant: every node's `loc.file` matches the target.
- **`objdump -d`**: filter to a specific symbol. Invariant: addresses
  monotonically increasing within a function.
- **`git log --format=...`**: filter commits matching a pattern.
  Invariant: parent hashes form a connected DAG.
- **`strace -e ...`**: filter to syscalls relevant to an operation.
  Invariant: file descriptor lifecycle balanced.
- **`nvprof` / `ncu`**: filter to specific kernels. Invariant: total
  kernel time ≤ total wall clock time.

Each is the same shape: thin filter on opaque tool output, plus at
least one invariant that fails loud when assumptions break.

## When NOT to apply this pattern

- When you fundamentally need a different transformation than the tool
  provides (e.g., need lossless round-trip but tool is lossy).
- When the tool's output is not stable enough to rely on (rapidly
  changing format across minor versions).
- When the cost of running the tool is prohibitive (e.g., extracting
  one fact requires re-running a 10-minute compile).

In these cases, build the alternative — but use the same recipe to
build it (small state, structural invariants, empirical validation
against corpus).

## See also

- `bpd/lib/c_preprocess.pl` — first instance of the pattern in Ruach
  Tov's substrate
- `bpd/tests/test_c_preprocess.pl`, `bpd/tests/test_c_preprocess_piece2.pl`
  — empirical validation of the cpp wrapper
- Memory reflection `cc6d63dc-156b-4345-8979-e3620b171c7c` (companion
  memory `f9c3a938-310b-48af-a688-f0cf7c89fb1c`) — the discovery
  moment with the full memory mesh of associated semantic, procedural,
  meta, and episodic memories

## Provenance

Authored by metayen, 2026-05-17, after the empirical sweep showed
`c_preprocess.pl` working at 52/55 archs. Pattern named and elaborated
in conversation with Heath, who riding-along on the regex-to-AST
evolution offered the generative-move and program-transformation-algebra
insights that turned a war story into a recipe.
