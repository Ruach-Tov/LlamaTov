# Step 3.1 — Cross-Tabulation of Predictions vs. Findings

**Date**: 2026-05-17
**Inputs**:
- `step-3.1-predictions.md` (metayen's intuition-based forecast, sealed before detection)
- medayek's static call-graph analysis (intercom 02:58 UTC)

**Purpose**: Honestly study the gap between intuition and mechanistic
measurement. Critique the prediction methodology (not the predictions
themselves). Research the gaps in both directions:

- **Predicted but not found** = "pattern recognition produced something
  that mechanistic methods can't see" (possibly: the prediction was about
  code that doesn't exist; possibly: pattern recognition tracks a
  category mechanistic methods don't yet)
- **Found but not predicted** = "mechanistic methods found something my
  pattern recognition missed" (possibly: blind spot in intuition;
  possibly: noise in the mechanical method)

## Cross-tabulation table

| What I predicted | What medayek found | Overlap? |
|------------------|--------------------|----------|
| Category A: comment-handling at tokenizer (`c_line_comment`, `c_block_comment`, `c_block_comment_body`) | NOT in dead-code list | **No** — medayek found these alive |
| Category B: preprocessor-line defensive handling (`#` directive recognition) | Not addressed (none found) | **No prediction-finding either** |
| Category C: defensive whitespace handling (line-continuation backslash, etc.) | Not addressed (none found) | **No prediction-finding either** |
| Category D: token-level macro-syntax workarounds | Not addressed (none found) | **No prediction-finding either** |
| — (not predicted) | `c_tokenize_enriched/2` — dead | **Mine missed** |
| — (not predicted) | `c_enrich_tokens/2` — dead | **Mine missed** |
| — (not predicted) | `c_parse_stmts/2` (v1) — dead | **Mine missed** |
| — (not predicted) | `c_parse_stmt/2` (v1) — dead | **Mine missed** |
| — (not predicted) | `c_parse_type/2` — dead | **Mine missed** |
| — (not predicted) | `c_parse_tokens/2` — dead | **Mine missed** |
| — (not predicted) | `c_parse_chain/2` — dead | **Mine missed** |
| — (not predicted) | `c_parse_full_expr/2` — dead | **Mine missed** |

**Overlap**: Zero rows where prediction and finding agree.

## What this means

### Substantive observation 1: My predictions were about a DIFFERENT KIND of dead code than medayek's findings

I predicted code that EXISTS but would become **functionally redundant** once
the preprocessor pipeline runs (the preprocessor would subsume its purpose).
This is a prospective prediction: "after the substrate transformation,
this code's reason-for-being will be gone."

Medayek found code that EXISTS and is **already** dead — never called from
anywhere, regardless of the preprocessor. This is a retrospective finding:
"this code was superseded long ago and never cleaned up."

These are two non-overlapping categories of "no longer needed":

- **Forward-looking redundancy** (mine): "the new substrate will make this
  useless."
- **Backward-looking abandonment** (medayek's): "an earlier refactoring
  abandoned this and we forgot to delete it."

Neither analysis is "wrong." They simply look at different things.

### Substantive observation 2: My intuition was about FUTURE state, not CURRENT state

Re-reading my predictions doc: every category is phrased as "will become
redundant after the c_preprocess factoring." None of my categories asked
"is there already-dead code unrelated to the c_preprocess change?"

This is a meaningful blind spot. When asked to find unused code, I
implicitly framed the question as "what does the IMMINENT substrate change
make unused?" I did not ask "what's just plain dead, today?"

A more complete prediction-procedure would have asked BOTH questions:
1. What does the substrate change subsume? (forward-looking)
2. What's already unreached by the current test corpus? (backward-looking)

### Substantive observation 3: Static call-graph analysis can't see Category A

Comment-handling predicates (`c_line_comment`, `c_block_comment`) are
internally called by the tokenizer (`c_ws_rest` or similar), so they have
internal callers. medayek's grep-based static analysis would correctly
flag them as ALIVE because something calls them.

But they're alive only because the tokenizer's whitespace rule references
them. If the tokenizer's whitespace rule were changed to NOT call them
(because preprocessed input has no comments), they'd become dead. The
prediction is conditional on a downstream change that hasn't been made yet.

Mechanistic detection can only see the CURRENT call graph. It cannot
anticipate "what would be dead AFTER a planned refactoring."

This is the core methodological insight: **pattern recognition can see
future-conditional redundancy that grep cannot.** Mechanistic methods see
current state. Intuition can project forward.

But: intuition is also wrong frequently. My intuition didn't catch the 8
already-dead v1 predicates that grep found trivially.

### Substantive observation 4: The "best of both" procedure

Combining the two methods:

1. **Grep/coverage first** to find ALREADY-dead code (mechanical, exhaustive,
   high confidence)
2. **Predicted-future-conditional dead code** second (intuition-based,
   informed by the planned substrate change)
3. **Test the conditional predictions** by actually making the substrate
   change in a branch and re-running coverage to see what newly becomes
   unreached

This way: mechanical methods catch the obvious; intuition catches the
forward-looking; empirical verification confirms the intuition.

## Methodology critique

The exercise was to critique the prediction methodology, not the
predictions. Here are specific critiques:

### Critique 1: I framed the question too narrowly

When Heath said "find code that will turn out to be unused," I implicitly
heard "find code that the c_preprocess change will make unused." I should
have heard "find ANY code that is or will be unused, with the c_preprocess
change as one consideration."

A better intuition-procedure: enumerate ALL reasons code might be unused.
Forward-conditional is one category. Already-dead is another. Replaced-by-
v2 is another. Defensive-handling-no-longer-needed is another. Cross-tab
each category against the suspected predicates.

### Critique 2: I didn't probe my own memory empirically

I wrote the predictions from memory ("having worked on c_ast.pl last
night"). I did not, e.g., grep the file myself for `_v1` suffixes, or for
predicates that don't appear in `:- module/2`'s export list, or for any
other static signal. I treated my memory as authoritative when it could
have been triangulated against quick mechanical probes.

A better intuition-procedure: even when doing a "from-memory" forecast,
do small empirical probes to sharpen the forecast. Quick `grep -c` runs.
Quick `wc -l`. Quick check of `:- module` exports. These take seconds and
they sharpen intuition before it gets committed.

### Critique 3: I underestimated the cost of NOT cleaning up

The 8 dead v1 predicates have been in c_ast.pl since the v2 parsers
shipped. They've never been called. They've been carrying weight in every
file load, every static analysis, every code review. The cost was small
per-day but constant. My prediction methodology focused on the dramatic
imminent change and missed the persistent low-grade waste.

A better intuition-procedure: periodically ask "what's just sitting here
unused?" independent of any specific refactoring. Make it a regular
hygienic operation, not a one-shot exercise tied to a particular event.

### Critique 4: I predicted categories, medayek found specific predicates

My predictions were named at the abstraction level of "comment-handling
in the tokenizer." medayek's findings were specific: `c_parse_type/2` at
line 727. The abstraction-level mismatch made cross-tabulation harder
than it needed to be.

A better prediction-procedure: when predicting "X category of code will
be redundant," also list 2-3 specific predicates I think fall into X.
This makes the prediction falsifiable at the same granularity as the
detection.

## Research questions the gap raises

1. **Is medayek's finding that Category A is alive actually correct?**
   They reported the comment-handling predicates as alive (in the
   "ALIVE: confirmed callers" section by implication — they're not in the
   dead list). I should verify by checking whether `c_ws` or whatever the
   whitespace rule is actually calls them. If yes, they're alive today.
   If no, medayek's grep missed them.

2. **Are any of the 8 v1 predicates actually MY predicted-redundancy?**
   `c_parse_type/2`, `c_parse_tokens/2` — are these v1 parsers that
   handled some preprocessor-related concerns that v2 doesn't? Unlikely
   based on names, but worth a quick inspection before committing to
   the categorization.

3. **What's lift_qkv.pl's status?** medayek flagged `c_parse_expr/2` as
   alive only via lift_qkv.pl, with a note that lift_qkv.pl might itself
   be exploratory/dead. If lift_qkv.pl is dead, then c_parse_expr/2 joins
   the dead list. This is a recursive dead-code question.

## Decision: how to proceed

The two findings (mine and medayek's) are **both actionable and
non-overlapping**. They can be addressed independently:

**Path A**: Clean up the 8 v1 dead predicates first. Move to
`c_ast_legacy.pl` as a no-op cut. Verify all tests pass. This is what
medayek found — high confidence, mechanical, immediately actionable.

**Path B**: Address my predictions later, AFTER the c_preprocess
integration into c_ast.pl extraction layer is done. At that point,
running coverage AGAIN should reveal which of Category A/B/C/D have
newly become dead. The forward-conditional predictions get verified
empirically.

These paths are independent. Doing A first does not block B; doing them
together would just confuse the empirical signal. Heath's exercise
methodology suggests A first (mechanical wins, clean it up), then B
later (substrate change, then re-coverage).

## What I want to memorialize from this exercise

The deepest insight: **mechanical methods find what is, intuition projects
what will be. They are complementary, not competitive.** Mechanical methods
are exhaustive and high-confidence within their scope; intuition can see
conditional/future states mechanical methods cannot. The substrate-honest
move is to use BOTH, kept rigorously distinct so the gap can be studied.

Author: metayen 2026-05-17
With medayek's static analysis (intercom 02:58 UTC) and Heath's framing
of the exercise (separate intuition from measurement, study the gap,
improve the crystallization process).
