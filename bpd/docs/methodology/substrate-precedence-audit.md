# Substrate Precedence Audit — 2026-05-18

**Date**: 2026-05-18 ~14:30 UTC
**Performed by**: metayen
**Trigger**: bug caught during CFD C2.1 corrective work (commit 20a40f600)
**Status**: complete; all currently-shipped kernel emits PASS

## The bug that triggered this audit

During the C2.1 corrective work (commit 20a40f600), a precedence bug
was caught in the Harten entropy fix emit. The substrate's `c_binop/3`
emits without precedence-preserving parentheses, so:

```prolog
c_binop('/', N, c_binop('*', c_float_f(2.0), c_var(eps)))
```

emits as:

```c
N / 2.0f * eps
```

Under C operator precedence with left-to-right associativity, this
parses as `(N / 2.0f) * eps`, NOT as `N / (2.0f * eps)`. The
substantive intent was the latter; the emit was the former. **Silent
correctness bug.**

The fix in C2.1 was an explicit `c_paren` on the denominator:

```prolog
c_binop('/', N, c_paren(c_binop('*', c_float_f(2.0), c_var(eps))))
```

emits correctly as `N / (2.0f * eps)`.

After the fix shipped, an audit was performed across all
currently-shipped kernel emits to verify no other instances of this
bug pattern exist.

## The audit pattern

For each shipped kernel emit, the audit checks whether any binary
operator has a *right-side compound operand* whose operator has *same
or lower precedence* than the outer operator, without a `c_paren`
wrapping.

This is the precedence-collapse risk: when the right-side operator
binds the same as or weaker than the outer operator, C's left-to-right
associativity collapses the expression against intent.

The risk patterns by outer operator (only `/` and `-` have produced
real bugs; others are theoretically risky):

| Outer | Risky right-side operator | Why |
|---|---|---|
| `/` | `*`, `/`, `%` (same precedence) | `a / b * c` → `(a/b)*c`, not `a/(b*c)` |
| `-` | `+`, `-` (same precedence) | `a - b + c` → `(a-b)+c`, not `a-(b+c)` |
| `<` | `<`, `<=`, `>`, `>=` (same prec) | rare in kernel emits |
| `&&` | `\|\|` (lower prec) | `a && b \|\| c` → `(a&&b)\|\|c`; usually fine |

Safe right-side operands (no precedence collapse possible):

- `c_paren(_)` — already wrapped
- `c_var(_)`, `c_int(_)`, `c_float_f(_)`, `c_hex(_)` — atomic
- `c_index(_, _)` — `x[i]` has `[]` as syntax
- `c_member(_, _)` — `x.y` higher precedence than any binop
- `c_call(_, _)` — `f(...)` has `()` as syntax
- A right-side operator that *binds tighter* than the outer (e.g.,
  `a - b * c` is fine because `*` binds tighter than `-`)

## Audit method

For each kernel emit predicate currently in the substrate, the audit:

1. Emits the kernel to CUDA source via `phrase(c_ast:emit(K, 0), Cs)`.
2. Scans the source line-by-line for the regex pattern:
   ```
   /\s+[a-zA-Z_][a-zA-Z_0-9]*(?:\[[^\]]+\])?\s+[*/]\s+[a-zA-Z_(]
   ```
   This catches the specific pattern `/ identifier * something` and
   `/ identifier / something` (with optional array subscript), which
   is the load-bearing risky form.
3. Also visually inspects all subtractions and divisions for intent
   vs emit, given that the regex catches only the most common
   pattern.

## Audit results — 2026-05-18 ~14:30 UTC

All 11 currently-shipped kernel emits **PASS** the audit:

| Kernel | Domain | Status |
|---|---|---|
| k_compute_flux | CFD (Roe + Harten) | PASS |
| k_update_conservative | CFD (elementwise) | PASS |
| k_compute_primitives | CFD (elementwise) | PASS |
| k_cfl_condition | CFD (reduction) | PASS |
| rms_norm | ML (reduction) | PASS |
| softmax (default) | ML (reduction) | PASS |
| softmax (with fix_softmax_phase_inter_race) | ML | PASS |
| warp_reduce_sum | helper | PASS |
| warp_reduce_max | helper | PASS |
| block_reduce_sum | helper | PASS |
| block_reduce_max | helper | PASS |

The Harten entropy fix bug (caught and fixed in C2.1, commit
20a40f600) was the only instance of this bug pattern in shipped
substrate emits.

## Per-kernel notes

### k_compute_flux

Contains 11 division operations after the C2.1 fix. All have either:
- Simple variable denominators (`/ rho_l`, `/ a_roe_sq`, `/ denom`),
- OR `c_paren`-wrapped compound denominators (`/ (2.0f * eps)`,
  `/ (2.0f * a_roe_sq)`).

One subtraction line is worth noting:
```c
float alpha2 = drho - dp / a_roe_sq;
```

Under C precedence, this parses as `drho - (dp / a_roe_sq)` because
`/` binds tighter than `-`. The intent matches: mavchin's Python
reference computes `alpha2 = drho - dp / a_roe_sq` with the same
implicit grouping. **Substrate emit correct.**

### k_cfl_condition

```c
float a = sqrtf(1.4f * p / r);
```

Under C left-to-right at same precedence, this parses as
`sqrtf((1.4f * p) / r)`. Equivalent to `sqrtf(1.4f * (p / r))` in
exact arithmetic and very close in float32. The Python reference
matches. **Substrate emit correct.**

### Helper functions

All four reduction helpers (warp_reduce_*, block_reduce_*) have only
trivial divisions (`/ 32`, `tid / 32`, `tid % 32`). All safe.

### softmax

```c
float val = expf(x[col] - max_val);
```

Subtraction with simple operands. Safe.

```c
float inv_sum = 1.0f / tmp;
float val = y[col] * inv_sum;
```

Simple division. Safe.

## Substrate-design implication

The c_ast layer's `c_binop/3` is *not* precedence-aware. It emits the
operator with its operands flat:

```prolog
emit_expr(c_binop(Op, L, R)) -->
    emit_expr(L), " ", atom_codes(Op), " ", emit_expr(R).
```

This means **the substrate predicate author is responsible** for
inserting `c_paren` wrappers wherever C precedence would collapse the
expression against intent.

Two possible long-term substrate-design directions to remove this
manual-discipline requirement:

1. **Precedence-aware emit**: modify `c_binop/3` to know operator
   precedences and emit parens automatically when needed. Requires
   the emit pass to track context (the outer operator). Substantive
   c_ast refactor.

2. **Static analysis pass**: a separate Prolog predicate that walks
   the c_ast before emit and rewrites risky nestings to add `c_paren`.
   Could run during a build step. Non-invasive to c_ast itself.

Both directions are queued for future substrate-design sessions. For
now: manual discipline + this audit.

## Regression test

A pytest-based regression test (`bpd/tests/test_emit_precedence_audit.py`)
is being prepared by medayek (whose verification lane includes the
pytest+swipl fixture pattern).

### Why my first attempt failed

My initial fixture used a tempfile + `swipl FILE.pl` invocation with
all 11 emits in one `initialization/1` goal. When any single emit
failed, the whole goal failed, `halt` never fired, and swipl dropped
to the interactive REPL waiting for input — causing the subprocess
to hang at the 60-second timeout.

Wrapping each emit in `catch/3` did not help (the failure was upstream
of the catch).

### The substrate-honest fix

Per medayek 2026-05-18 ~14:40 UTC: **the fix is at the swipl
invocation layer**, not the script layer. Use `-t halt(1)` as the
toplevel fallback:

```bash
swipl -q -g 'Goal' -t 'halt(1)' script.pl
```

`-t halt(1)` means: if the goal succeeds, halt(0); if it fails or
errors, halt(1). No interactive mode ever. The subprocess always
exits.

Alternative: include this directive at the top of the script:

```prolog
:- set_prolog_flag(toplevel_goal, halt(1)).
```

Same effect.

This is durable substrate knowledge for all future Python+swipl test
infrastructure. The lesson: **`subprocess.run` with swipl needs a
toplevel-exit guarantee**, otherwise REPL-drop hangs the process.

### What the test will check

Same two patterns as the manual audit:

1. **Division precedence**: regex `/ id [*/] something` on each emit
2. **Brace balance**: `{`, `(`, `[` paired counts in the emit text

The regex catches the load-bearing pattern from the Harten fix bug.
Brace balance is a syntactic sanity check unrelated to precedence
but cheap to add.

For now: this doc remains the canonical audit record. The test
augments it as a regression check.

## Connection to other methodology principles

This audit operates under several established principles:

- **Comprehension over verbatim**: the substrate must comprehend what
  it emits, including operator-precedence semantics. Emitting text
  the substrate cannot reason about (in terms of C parsing) is debt.

- **Measure, don't assume**: my initial belief was "c_ast probably
  handles precedence automatically." The empirical check (look at the
  emit text, parse it as C, compare to intent) revealed the bug.

- **The framing question**: *"What is the correct form of this c_ast
  emit?"* Answer: the form where the emitted text parses as the
  algebraic expression the substrate intended. When that requires
  parens, the substrate must include them.

## Future kernels

When adding new kernel emit predicates to the substrate:

1. Identify any division where the denominator is a compound (binop)
   expression. Wrap with `c_paren`.
2. Identify any subtraction where the right operand is a compound
   `+` or `-` expression where C left-associativity would change the
   intent. Wrap with `c_paren`.
3. After emitting once, visually inspect the generated CUDA for the
   pattern `/ identifier [*/] something` or `- identifier [-+] something`.
4. Add to the audit table above if any non-trivial precedence
   wrapping was needed.

This audit will be re-run for each new kernel and as a periodic
maintenance task on existing kernels.
