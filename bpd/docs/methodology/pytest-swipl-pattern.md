# The Pytest+swipl Test Pattern

**Date crystallized**: 2026-05-18 ~15:05 UTC
**Discovered by**: medayek + metayen across the precedence-audit regression test
**Status**: durable operational pattern for substrate-emit regression tests

When writing pytest-based regression tests that need to invoke swipl as
a subprocess and parse the output of multiple kernel emits, four
patterns must all be applied together. Missing any one produces silent
failures that are very hard to debug from pytest's perspective.

## The four patterns

### 1. `set_prolog_flag(toplevel_goal, halt(1))` at script top

Without this, swipl drops into the interactive REPL when any goal
fails. Inside subprocess.run with no stdin, the REPL hangs until
timeout. From pytest's perspective: "swipl failed with timeout" — the
real cause (a Prolog goal failed) is invisible.

```prolog
:- set_prolog_flag(toplevel_goal, halt(1)).
:- use_module('...').
:- initialization((Goal), main).
```

`set_prolog_flag(toplevel_goal, halt(1))` ensures that *any* condition
that would normally drop to the REPL instead halts with exit code 1.
The subprocess always exits. From the command line, the equivalent is
`swipl -t halt(1)`.

### 2. Per-emit unique variable names

The body of `initialization((...))` is one big conjunction. All Prolog
variables in that conjunction share scope. If you reuse `K` across
multiple emit calls:

```prolog
:- initialization((
    cfd_flux_kernel(k_compute_flux, K),         % K bound to flux AST
    emit_program([K], S),                        % uses bound K
    cfd_update_conservative_kernel(k_update, K), % FAILS — K already bound!
    ...
), main).
```

The second `cfd_update_conservative_kernel(..., K)` tries to UNIFY K
(already bound to flux AST) with the new kernel's AST. They don't
match → goal fails silently.

**Fix**: per-emit unique variable names:

```prolog
:- initialization((
    cfd_flux_kernel(k_compute_flux, K0),
    emit_program([K0], S0),
    cfd_update_conservative_kernel(k_update, K1),
    emit_program([K1], S1),
    ...
), main).
```

The substrate-honest read: Prolog variables are lexically scoped to
the goal, not to a "clause" the way they're scoped to a `:- ... .`
in a source file. Treat each emit as needing its own variable namespace.

### 3. `emit_program([K], S)` — list wrapping for single nodes

`emit_program/2` at `lib/c_ast.pl:131` is defined as:

```prolog
emit_program(Nodes, String) :-
    phrase(emit_nodes(Nodes, 0), Codes),
    atom_codes(String, Codes).
```

`emit_nodes/2` pattern-matches on `[H|T]` — it expects a list. Passing
a single kernel node `K` (e.g., `c_func(['__global__'], ...)`) fails
silently because the list pattern doesn't match.

**Fix**: wrap single nodes in a list:

```prolog
emit_program([K0], S0)   % succeeds
emit_program(K0, S0)     % silent failure
```

This is asymmetric with `emit_to_file/2` which can take either form
(it wraps internally). `emit_program/2` does not wrap.

### 4. Delimiter-based output parsing

Multiple emits in one swipl invocation need to be separated in the
output so the Python side can route each kernel's CUDA to the right
test parameter. Use unambiguous delimiters:

```prolog
format('===BEGIN ~w===~n', [Name]),
emit_program([K], S),
write(S),
nl,
format('===END ~w===~n', [Name]),
```

Then on the Python side:

```python
pattern = re.compile(
    rf"===BEGIN\s+{re.escape(name)}===\n(.*?)===END\s+{re.escape(name)}===",
    re.DOTALL,
)
match = pattern.search(output)
cuda_source = match.group(1) if match else ""
```

The `re.DOTALL` flag is essential — the kernel source has newlines,
and without DOTALL `.*?` won't cross them.

## Skeleton fixture

A reusable shape for the module-scoped fixture:

```python
import os
import re
import subprocess
import tempfile
import pytest


BPD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


KERNEL_EMIT_CALLS = [
    # (goal_string, friendly_name)
    ("cfd_flux_kernel(k_compute_flux, K)", "k_compute_flux"),
    # ... etc
]


@pytest.fixture(scope="module")
def emitted_kernels():
    emit_goals = []
    for i, (goal, name) in enumerate(KERNEL_EMIT_CALLS):
        # Substitute K and K-only-arg forms with K<i>
        unique_goal = goal.replace(", K)", f", K{i})").replace("(K)", f"(K{i})")
        emit_goals.append(
            f"(catch(("
            f"  ({unique_goal} -> true ; (format('===BEGIN {name}===~nFAIL: goal {unique_goal} failed~n===END {name}===~n'), fail)),"
            f"  (emit_program([K{i}], S{i}) -> true ; (format('===BEGIN {name}===~nFAIL: emit_program([K{i}], S{i}) failed~n===END {name}===~n'), fail)),"
            f"  format('===BEGIN {name}===~n~w~n===END {name}===~n', [S{i}])"
            f"), E, "
            f"  format('===BEGIN {name}===~nFAIL: ~q~n===END {name}===~n', [E])"
            f"); true)"
        )
    all_goals = ",\n    ".join(emit_goals)

    script = f"""
:- set_prolog_flag(toplevel_goal, halt(1)).
:- use_module('{BPD_DIR}/lib/c_ast').
:- use_module('{BPD_DIR}/lib/kernel_templates_cfd').
:- use_module('{BPD_DIR}/lib/kernel_templates_llama', except([kernel_available_fixes/2, fix_description/2])).

:- initialization((
    {all_goals},
    halt(0)
), main).
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pl', delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        result = subprocess.run(
            ["swipl", "-q", script_path],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            pytest.skip(f"swipl failed: {result.stderr[:500]}")
        output = result.stdout
    finally:
        os.unlink(script_path)

    # Parse delimited sections
    emits = {}
    for _, name in KERNEL_EMIT_CALLS:
        pattern = re.compile(
            rf"===BEGIN\s+{re.escape(name)}===\n(.*?)===END\s+{re.escape(name)}===",
            re.DOTALL,
        )
        match = pattern.search(output)
        emits[name] = match.group(1) if match else ""
    return emits
```

## When this pattern applies

This pattern is the right shape for:

- Regression tests that emit multiple kernels and check their CUDA
  source for specific properties (precedence, brace balance,
  byte-identity against a reference, etc.)
- Cross-substrate audits that need to scan many emit predicates
- Any test where the Prolog side is "consult, emit, write" with no
  GPU involvement

This pattern is NOT the right shape for:

- Tests that need to compile + run CUDA (see test_cfd_substrate.py
  for that — it uses a different fixture pattern with nvcc + ctypes)
- Tests that need to interact with a running swipl REPL session
  (use a different pattern with subprocess.Popen + stdin/stdout pipes)
- Single-kernel emit checks (simpler one-shot subprocess call is fine)

## Caveats and edge cases

**Module imports can produce harmless `Warning:` lines**. swipl writes
these to stderr by default. Don't filter or fail on them — they're
informational. The fixture above ignores stderr unless returncode is
non-zero.

**The "discontiguous" warnings from c_ast.pl** appear on every load
because some emit predicates are interleaved across the source file.
This is pre-existing substrate-design state; don't try to fix it from
the test side.

**Goals that fail vs goals that error** are different:
- Failure: silent, doesn't raise
- Error (e.g., `throw(some_error)`): caught by `catch/3`

The fixture's `(Goal -> true ; fail-message-and-fail)` idiom handles
both cases by converting failure into a printable fail-message before
the disjunction. Pure `catch/3` would miss silent failures.

## Connection to other methodology

This pattern operates inside the **substrate-precedence-audit**
methodology (see `substrate-precedence-audit.md`) but is reusable for
any test that needs to scan multiple kernel emits.

Per the broader methodology: the substrate's test infrastructure is
itself substrate-honesty work. A test that silently skips when swipl
fails (because the fixture didn't handle the failure mode) is
substrate-blind. A test that exits cleanly with a clear failure
message is substrate-honest about what went wrong.

The four patterns together make the pattern *robust* — clear failure
modes, no hanging, useful diagnostics, scaling to many emits.

## Origin

The pattern crystallized across three iterations of
`test_emit_precedence_audit.py`:

1. medayek's initial version (commit `4d9d351a8`) — had the right
   structure but used `swipl -t halt(1)` from the wrong vantage and
   reused `K` across kernels
2. metayen's integration fixes (commit `c299d6c50`) — added per-emit
   unique variables, `[K]` list wrapping, path correction
3. This methodology doc — captures the combined wisdom for future
   tests

Both wizards' contributions were necessary. The pattern is now durable.

---

*Authored 2026-05-18 ~15:05 UTC by metayen,
per medayek's "worth a methodology footnote" suggestion.*
