# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""prolog_goal.py — Prolog goal AST and canonical serializer.

Build Prolog goals as TREE-STRUCTURED Python values, serialize to a
single canonical string for passing to `swipl -g <goal>`.

Replaces the previous f-string approach in kernel_emit_bridge.py.

## Why an AST

The previous f-string approach in kernel_emit_bridge.py had to manually
manage backslash escaping for embedded quotes:

    'emit_program([c_include_sys(\\"cuda_runtime.h\\"), ...], C)'

This worked but was fragile — a 2026-05-17 bug introduced an extra
escape level and caused swipl "End of file in quoted string" failures
that took several iterations to diagnose. The substrate-honest move,
per Heath's framing: build goals as Prolog ASTs and serialize them
canonically, the same discipline c_ast applies to C code.

The principle generalizes: any text generation should be tree-structured
intent serialized once, not surface-string concatenation. The bridge
already emits AST-emitted CUDA from c_ast terms; now it builds
AST-emitted Prolog from pg_* terms. Symmetric architecture.

## Term vocabulary

  pg_atom(name)        — Prolog atom (auto-quoted if non-canonical)
  pg_string(s)         — Prolog string in double quotes
  pg_var(name)         — Prolog variable (uppercase-starting)
  pg_int(n)            — Prolog integer literal
  pg_call(f, args)     — functor application: f(a1, a2, ...)
  pg_list(items)       — Prolog list: [a1, a2, ...]
  pg_seq(goals)        — comma-separated sequence of goals (the body
                         that `swipl -g` accepts directly)

Every term is a tagged tuple. Serialization is `serialize(term) -> str`.

## Quoting conventions

  Atoms: bare if matching [a-z][a-zA-Z0-9_]*, else single-quoted with
         internal ' doubled per Prolog convention.
  Strings: always double-quoted; internal " doubled.
  Numbers and variables: written directly.

## Example

  >>> goal = pg_seq([
  ...     pg_call('use_module', [pg_atom('lib/c_ast')]),
  ...     pg_call('use_module', [pg_atom('lib/kernel_templates_llama')]),
  ...     pg_call('unary_activation_kernel',
  ...             [pg_atom('k_silu'), pg_var('K')]),
  ...     pg_call('emit_program', [
  ...         pg_list([
  ...             pg_call('c_include_sys', [pg_string('cuda_runtime.h')]),
  ...             pg_call('c_include_sys', [pg_string('math.h')]),
  ...             pg_atom('c_blank'),
  ...             pg_var('K'),
  ...         ]),
  ...         pg_var('C'),
  ...     ]),
  ...     pg_call('write', [pg_var('C')]),
  ...     pg_atom('halt'),
  ... ])
  >>> serialize(goal)
  'use_module(lib/c_ast), use_module(lib/kernel_templates_llama), \\
   unary_activation_kernel(k_silu, K), \\
   emit_program([c_include_sys("cuda_runtime.h"), c_include_sys("math.h"), \\
                 c_blank, K], C), \\
   write(C), halt'

Author: metayen 2026-05-17
Per Heath's reframe: "replace the f-string parameterization with
something more similar to emitting from a parameterized AST."
The substrate-honest equivalent of c_ast for Prolog goal construction.
"""


from typing import List, Union

# A goal term is one of these tagged tuples.
# (We use plain tuples rather than dataclasses for minimal dependencies
# and to mirror the c_ast convention of Prolog-term-shaped Python values.)


def pg_atom(name: str) -> tuple:
    """Prolog atom. Auto-quoted at serialize time if non-canonical."""
    return ('pg_atom', name)


def pg_string(s: str) -> tuple:
    """Prolog string in double quotes."""
    return ('pg_string', s)


def pg_var(name: str) -> tuple:
    """Prolog variable. Must start with uppercase or _."""
    if not name or not (name[0].isupper() or name[0] == '_'):
        raise ValueError(
            f"Prolog variable must start with uppercase or _, got {name!r}"
        )
    return ('pg_var', name)


def pg_int(n: int) -> tuple:
    """Prolog integer literal."""
    return ('pg_int', int(n))


def pg_call(functor: str, args: List[tuple]) -> tuple:
    """Functor application: f(arg1, arg2, ...).

    For a 0-arg call (just the functor name), prefer pg_atom(functor).
    """
    return ('pg_call', functor, tuple(args))


def pg_list(items: List[tuple]) -> tuple:
    """Prolog list: [item1, item2, ...]."""
    return ('pg_list', tuple(items))


def pg_seq(goals: List[tuple]) -> tuple:
    """Comma-separated sequence of goals — the body for swipl -g."""
    return ('pg_seq', tuple(goals))


# ── Serializer ────────────────────────────────────────────────────────

# Canonical Prolog atom: starts lowercase, then alphanumerics/underscores.
# Also allows '/' for module paths like lib/c_ast (Prolog parses this as
# the infix operator '/' but it's syntactically valid in use_module(lib/c_ast)).
def _is_bare_atom(name: str) -> bool:
    """A Prolog atom that needs no quoting.

    Each /-separated segment must be a valid identifier (per
    str.isidentifier — letter or underscore start, alphanum body).
    The first segment additionally must start with a lowercase letter
    (Prolog atoms vs. variables).

    Empirically equivalent to the prior regex:
        r'^[a-z][a-zA-Z0-9_]*(/[a-zA-Z_][a-zA-Z0-9_]*)*$'
    on all tested inputs. Replaces the regex per Heath's 2026-05-18
    "retire trivial regex excursions" directive — eliminates one
    mental-grammar-context-switch per reader pass.
    """
    if not name:
        return False
    segments = name.split('/')
    if not segments[0] or not segments[0][0].islower():
        return False
    return all(seg.isidentifier() for seg in segments)


def _quote_atom_if_needed(name: str) -> str:
    """Bare atom if canonical; else single-quoted with internal ' doubled."""
    if _is_bare_atom(name):
        return name
    # Single-quote and escape internal single quotes per Prolog convention.
    escaped = name.replace("'", "''")
    return f"'{escaped}'"


def _quote_string(s: str) -> str:
    """Double-quote a Prolog string. Internal " is doubled per convention."""
    escaped = s.replace('"', '""')
    return f'"{escaped}"'


def serialize(term: Union[tuple, str]) -> str:
    """Render a Prolog goal AST term to its canonical string form.

    Returns a string suitable for `swipl -g <serialize(term)>`.
    """
    if not isinstance(term, tuple) or len(term) < 1:
        raise ValueError(
            f"serialize expects a Prolog goal AST tuple, got {term!r}"
        )

    tag = term[0]
    if tag == 'pg_atom':
        return _quote_atom_if_needed(term[1])
    if tag == 'pg_string':
        return _quote_string(term[1])
    if tag == 'pg_var':
        return term[1]
    if tag == 'pg_int':
        return str(term[1])
    if tag == 'pg_call':
        _, functor, args = term
        functor_text = _quote_atom_if_needed(functor)
        args_text = ', '.join(serialize(a) for a in args)
        return f'{functor_text}({args_text})'
    if tag == 'pg_list':
        items_text = ', '.join(serialize(i) for i in term[1])
        return f'[{items_text}]'
    if tag == 'pg_seq':
        return ', '.join(serialize(g) for g in term[1])
    raise ValueError(f"unknown Prolog goal term tag: {tag!r} in {term!r}")
