# Step 3.1 — Prediction Inventory (Mechanistic Dump)

**Date**: 2026-05-17
**Purpose**: Memorialize what I PREDICT will turn out to be unused/redundant
in `c_ast.pl` after the `c_preprocess` factoring, BEFORE running any
automated detection. After writing this list, I close it and act as if I
have never seen this codebase. The detection track (medayek + mutmut) runs
independently. At the end we cross-tabulate predictions vs. findings,
critique the prediction methodology, and research the gaps.

The point is not to be right. The point is to practice the discipline of
separating intuition from empirical measurement so we can study the gap
between them. Each row that the automated detection FINDS but I did NOT
predict is a candidate for "my pattern recognition was wrong about this."
Each row I predicted but the detection FAILS to find is a candidate for
"my pattern recognition exceeded what mechanistic methods can see."

## My predictions

I predict the following categories of code in `c_ast.pl` will be
detectable as no-longer-needed once the pipeline routes source through
`preprocess_file_segment` before parsing.

### Category A: Comment-handling at the tokenizer layer

- `c_line_comment//0` (or whatever it's named) — single-line `//` comment skipping
- `c_block_comment//0` — multi-line `/* ... */` comment skipping
- `c_block_comment_body//0` — recursive helper for block comment content
- Any whitespace-rule that explicitly handles comment-as-whitespace

**Rationale**: cpp strips comments before we see the output. The parser
will never encounter `//` or `/*` in preprocessed input.

### Category B: Preprocessor-line defensive handling

- Any rule that recognizes `#`-prefixed lines (preprocessor directives)
- Any rule that defensively skips unknown statement forms at file scope

**Rationale**: After `c_preprocess`, all `#`-prefixed lines have been
stripped or interpreted. The parser will never see them.

### Category C: Defensive whitespace handling beyond standard C++

- Any rule for handling line-continuation backslashes (`\` at end of line)
- Any rule for handling weird whitespace patterns that cpp normalizes

**Rationale**: cpp normalizes whitespace and joins line-continuation
backslashes. Preprocessed input should have clean whitespace.

### Category D: Token-level workarounds for macro-like syntax

- I don't currently believe any exist, but worth checking — if I ever
  added a rule like "if you see an ALL_CAPS identifier followed by `;`,
  treat as no-op statement," that would be macro-handling smearing.

**Rationale**: With preprocessor expanding macros, the parser never sees
unexpanded macro invocations.

## Calibration note

This list is what my pattern recognition produces from memory of having
worked on `c_ast.pl` last night. I may be wrong about:

- **Which specific predicate names exist** (I'm guessing at names)
- **Whether the rules I name are actually orthogonal to the parser** (they
  may be tangled in ways that prevent removal)
- **Whether there are OTHER redundancies I haven't named**

The gap between this list and the mutmut findings is itself data.

## Constraint on the exercise

Once this file is committed, I will not refer back to it during the
detection phase. The detection phase proceeds with naive eyes: assume the
codebase is unknown, let the automated methods find what they find. Only
at the cross-tabulation step (3.1.cross-tab) do I reopen this file and
compare.

Author: metayen 2026-05-17
