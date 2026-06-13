# Regex Retirement Audit — 2026-05-18

**Per Heath's directive**: review each regex retirement and ask:
"did we fully understand the intended correct behavior of the regex
as applied to the problem of lifting facts from llama.cpp? Did we
make a fix that is smoothly aligned with the intended correct behavior?"

## The audit principle

Each regex retirement is TWO questions, not one:

  1. Does the regex capture the substrate's intent?
     (i.e., is the regex correct?)
  2. Do we capture the regex faithfully?
     (i.e., is our replacement equivalent?)

The substrate-honest answer requires YES to both. Tonight's retirements
optimized for #2 (regex-equivalent structural replacements) but didn't
consistently audit #1 (whether the regex was capturing the deeper intent).

## Tonight's six retirements, audited

### ✅ Aligned with intent (4 of 6)

**841469458 `_BARE_ATOM_RE` in prolog_goal.py** — The intent is "valid
Prolog atom shape." The Python form uses `str.isidentifier()` + lowercase-
start, structurally matching the intent. Replacement IS the intended
behavior.

**4ceecb97f Unit 4 op-call retirement** — Intent: "function calls to
build_X or ggml_X." The regex missed identifier boundaries (matched
`build_X` inside `ggml_build_X` substrings). Our structural form
respects boundaries — MORE aligned with intent than the regex was.
**Bug-catch.**

**5971bf2cf L6/A5/A6 combined retirement** — Intent: per-line
declarations in models.h (using-graph, struct llama_model_X). The
substrate's syntactic conventions keep these on single lines; structural
per-line matching IS the AST equivalent for this scope. Sweet spot.

**9c89099ed L1 lift_dispatch_table retirement** — Intent: model
construction dispatch. The regex's narrow `return new llama_model_y(`
filter IS the intended scope (excludes rope-type/memory dispatches).
Aligned with intent.

### ⚠️ Aligned with regex-approximation only (2 of 6)

**693645a56 lift_arch_enum retirement** — Intent: "lift the LLM_ARCH
enum members from `enum llm_arch { ... }`." The regex used a text-shape
heuristic: exact 4-space indent + LLM_ARCH_X + comma. This works on
current `llama-arch.h` formatting but isn't the deeper intent.

  Deeper intent: parse the enum DECLARATION as C, extract enumerators.

  Gap: regex (and our retirement) would silently miss entries with
  different indentation, trailing comments, or explicit values
  (`LLM_ARCH_X = 5,`).

**d956247d9 A1 lift_arch_hparams retirement** — Intent: "lift hparam
reads." The regex captures only `ml.get_key(KV, hparams.field, ...)`
(form 1). Real llama.cpp has 11+ form-2 reads where the second arg is
a local variable.

  Deeper intent: lift ALL `ml.get_key` calls regardless of second-arg
  shape; distinguish field-target from local-target in the output term.

  Gap: regex (and our retirement) miss form-2 reads entirely.

## The remedies investigated, and why they're queued not shipped tonight

### lift_arch_enum → AST-aligned remedy

Designed remedy: build `lift_arch_enum_ast/2` using `c_ast` to parse
the enum declaration, walk the enumerators, extract each LLM_ARCH_X
name.

  Empirical check: `c_parse_stmts_v2_partial` does NOT handle enum
  declarations. Returns 0 ASTs with the entire enum body as
  unconsumed tokens.

  Substrate constraint: parser would need extending to recognize
  `enum Name { Enumerators... };` and produce `c_enum_def(Name, ...)`
  AST nodes (the emit form exists; the parse form does not).

  Bounded for tonight? NO. Requires parser substrate work first.
  Estimated 2-4 hours minimum.

  Practical conclusion: queue as substrate enhancement.

### A1 lift_arch_hparams → AST-aligned remedy

Designed remedy: extend `lift_arch_hparams_ast/2` (already exists in
arch_summary.pl, parallels the regex form) with two more clauses for
the local-variable form. Then switch `lift_arch_full` to use the AST
version.

  Empirical check: parser handles the AST shape cleanly (40ms parse
  time on a real load_arch_hparams body). Pattern matching for form 2
  is straightforward additional clauses.

  But: real form-2 reads in llama.cpp are NOT simple `ml.get_key`
  calls. They're parts of compound expressions like:

    uint32_t n_vocab = 0;
    ml.get_key(LLM_KV_VOCAB_SIZE, n_vocab, false)
        || ml.get_arr_n(LLM_KV_TOKENIZER_LIST, n_vocab, false);

  The local variable declaration, the `||` short-circuit, and the
  fact that `n_vocab` is used elsewhere — these are substrate-design
  concerns that go beyond "lift this one call."

  Substrate constraint: arch_emit.pl currently can't emit form-2
  (would need to emit the local decl + the `||` expression too).
  If we lift form-2 without enhancing emit, the round-trip metric
  (currently 100% on 93 eligible archs) breaks for any arch with
  form-2 reads.

  Bounded for tonight? NO. The lift-side enhancement is small; the
  emit-side enhancement is substantial; both must ship together to
  preserve round-trip.

  Practical conclusion: queue as substrate enhancement.

## Substantive substrate-honest finding

The shipped retirements ARE the right code given the current substrate:
- They preserve the existing behavior of the regex AND the existing
  scope of the emit substrate.
- The round-trip metric is 100% on 93 eligible archs (commit b014f0c48,
  verified again in this audit).

The audit reveals two places where the retired regex captured the
substrate's ROUND-TRIP scope rather than the broader SEMANTIC intent.
The remedies for both are substrate-design enhancements, not regex-
retirement refinements.

## The recurring methodology principle

For future retirements:

  1. State the intended behavior in AST terms FIRST
  2. Verify the AST shape can be expressed cleanly (parser handles it)
  3. If yes: build AST matcher with one clause per semantic shape
  4. Verify the AST matcher captures a SUPERSET of regex behavior
  5. If superset: ship as deeper-intent-aligned retirement
  6. If parser doesn't support the shape: queue as substrate enhancement
  7. If emit substrate scope is narrower than lift: queue both lift+emit
     enhancement together to preserve round-trip

This is the **diagnose-reflect-remedy** pattern Heath named, applied at
the regex-retirement granularity:

  - Diagnose: what does the regex actually capture? What was it
    INTENDED to capture?
  - Reflect: where do these diverge? Is the divergence accidental
    (regex bugs) or scope-design (round-trip vs semantic-intent)?
  - Remedy: ship code that closes the gap if bounded; queue substrate
    enhancement if not.

## Queued substrate enhancements (post-audit)

  1. Parser extension: `c_ast` enum declaration recognition →
     enables lift_arch_enum AST-aligned remedy
  2. Form-2 hparam read: lift + emit coordination →
     enables lift_arch_hparams full-intent remedy
  3. L2 AST matcher (already prototyped) →
     replaces lift_arch_tensors regex with AST-aligned version
  4. A2 substrate enhancement: enrich `size_rec/3` to carry
     discriminator + outer context (per the A2 investigation,
     commit bb60a213e)

Each is bounded enough for a single focused session. Together they
close the substrate-honesty gaps the audit identified.

## What this audit produced

  - Methodology principle: "intent-vs-approximation" as the audit
    question (not just "regex-equivalent")
  - Four substrate enhancements queued with bounded scopes
  - Memories stored: methodology + episodic + per-commit reflections
  - This document as the persistent record

Author: metayen 2026-05-18 ~04:30 UTC
Per Heath's "review the git log and consider: did the applied fix
implement the intended correct behavior?" directive.
