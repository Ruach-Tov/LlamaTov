# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Emit Precedence Audit — Regression test for c_ast operator precedence.

Catches missing c_paren on compound denominators/divisors in emitted CUDA.
Example bug: `1.0f / rho_l * u_l` when it should be `1.0f / (rho_l * u_l)`.

All 11 current kernel emit predicates are tested:
  - 4 CFD kernels (compute_flux, update_conservative, compute_primitives, cfl_condition)
  - 3 ML kernels (rms_norm, softmax, softmax with fix)
  - 4 helpers (warp_reduce_sum, warp_reduce_max, block_reduce_sum, block_reduce_max)

Author: medayek (Collective SME, Verification Methodology)
Date: 2026-05-18
Per metayen's request — regression test for C2.1 precedence bug.
"""

import pytest
import subprocess
import re
import os
import tempfile

# Path setup: __file__ is bpd/tests/test_emit_precedence_audit.py
# dirname -> bpd/tests, dirname again -> bpd/
# We want REPO_ROOT (parent of bpd) so use_module paths can be written
# as "bpd/lib/..." OR equivalently we can use BPD_DIR directly and
# write paths as "lib/...". Use BPD_DIR = .../Ruach-Tov/bpd and reference
# lib/* directly.
BPD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════════════
# Kernel emit predicates to audit
# ═══════════════════════════════════════════════════════════════════════

# Predicate names match the actual exports in kernel_templates_cfd.pl
# and kernel_templates_llama.pl. Substrate-honest naming: use what the
# substrate actually exports, not abbreviated forms.
KERNEL_EMIT_CALLS = [
    ("cfd_flux_kernel(k_compute_flux, K)", "k_compute_flux"),
    ("cfd_update_conservative_kernel(k_update_conservative, K)", "k_update_conservative"),
    ("cfd_compute_primitives_kernel(k_compute_primitives, K)", "k_compute_primitives"),
    ("cfd_cfl_condition_kernel(k_cfl_condition, K)", "k_cfl_condition"),
    ("rms_norm_kernel(K)", "k_rms_norm"),
    ("softmax_kernel(K)", "k_softmax"),
    ("softmax_kernel([fix_softmax_phase_inter_race], K)", "k_softmax_fixed"),
    ("warp_reduce_sum_helper(K)", "warp_reduce_sum"),
    ("warp_reduce_max_helper(K)", "warp_reduce_max"),
    ("block_reduce_sum_helper(K)", "block_reduce_sum"),
    ("block_reduce_max_helper(K)", "block_reduce_max"),
    # Cell-indexed Dirichlet 3-neighbor stencil family (added 2026-05-18 by metayen)
    ("jacobi1d_kernel(k_jacobi1d, K)", "k_jacobi1d"),
    # BLAS subsumption family (added 2026-05-18 by metayen)
    ("sgemv_kernel_substrate_native(k_sgemv_substrate_native, K)", "k_sgemv_substrate_native"),
    ("sgemv_kernel_cublas_match(k_sgemv_cublas_match, K)", "k_sgemv_cublas_match"),
]

# Regex for precedence risk: `/ identifier [*/] something`
# Catches: `/ a * b`, `/ x[i] / y`, etc.
PRECEDENCE_RISK_PATTERN = re.compile(
    r'/\s+[a-zA-Z_][a-zA-Z_0-9]*(?:\[[^\]]+\])?\s+[*/]\s+[a-zA-Z_(]'
)


# ═══════════════════════════════════════════════════════════════════════
# Fixture: emit all kernels via single swipl invocation
# ═══════════════════════════════════════════════════════════════════════

def swipl_available():
    """Check if swipl is available (directly or via nix-shell)."""
    try:
        result = subprocess.run(["swipl", "--version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.fixture(scope="module")
def emitted_kernels():
    """Emit all kernel predicates and return dict of {name: cuda_source}."""
    
    # Build Prolog script that emits each kernel with delimiters.
    # Two substrate-honest details (per metayen 2026-05-18 ~14:55 UTC):
    #
    # 1. emit_program/2 takes a LIST of nodes (it calls emit_nodes/2 which
    #    pattern-matches on [H|T]). A single kernel node K must be wrapped
    #    as [K] for the call to succeed.
    #
    # 2. Plain catch/3 only catches THROWN exceptions, not Prolog FAILURE.
    #    If emit_program fails (e.g., AST has an unrecognized node type),
    #    the whole conjunction fails silently. To get a useful error
    #    message AND continue to the next kernel, use catch/3 around a
    #    body that uses (Goal -> true ; fail-message-and-fail). The
    #    outer `(...; true)` ensures the per-kernel emit-or-fail-message
    #    always succeeds, so the conjunction of all kernels progresses.
    # Use a UNIQUE K variable per kernel (K0, K1, K2, ...) because the
    # conjunction inside initialization shares variable scope. If we use
    # the same K, the first kernel binds K, and subsequent kernels try
    # to unify their AST against K — which fails because K is already
    # bound to the first kernel's tree. Per-kernel S variable too (S0,
    # S1, ...) for the emit-program result.
    emit_goals = []
    for i, (goal, name) in enumerate(KERNEL_EMIT_CALLS):
        # Substitute the original 'K' in the goal with a unique 'K<i>'
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
:- use_module('{BPD_DIR}/lib/kernel_templates_stencil', except([kernel_available_fixes/2, fix_description/2])).
:- use_module('{BPD_DIR}/lib/kernel_templates_blas', except([kernel_available_fixes/2, fix_description/2])).

:- initialization((
    {all_goals},
    halt(0)
), main).
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pl', delete=False) as f:
        f.write(script)
        script_path = f.name
    
    try:
        # Try direct swipl
        result = subprocess.run(
            ["swipl", "-q", script_path],
            capture_output=True, text=True, timeout=30,
            cwd=BPD_DIR
        )
        if result.returncode != 0:
            # Try nix-shell
            result = subprocess.run(
                ["nix-shell", "-p", "swiProlog", "--run",
                 f"swipl -q {script_path}"],
                capture_output=True, text=True, timeout=60,
                cwd=BPD_DIR
            )
        
        if result.returncode != 0:
            pytest.skip(f"swipl failed: {result.stderr[:200]}")
        
        output = result.stdout
    finally:
        os.unlink(script_path)
    
    # Parse output into per-kernel dict
    kernels = {}
    for _, name in KERNEL_EMIT_CALLS:
        begin = f"===BEGIN {name}==="
        end = f"===END {name}==="
        start_idx = output.find(begin)
        end_idx = output.find(end)
        if start_idx >= 0 and end_idx >= 0:
            source = output[start_idx + len(begin):end_idx].strip()
            kernels[name] = source
        else:
            kernels[name] = None  # emit failed
    
    return kernels


# ═══════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("goal,name", KERNEL_EMIT_CALLS,
                         ids=[name for _, name in KERNEL_EMIT_CALLS])
def test_emit_succeeds(emitted_kernels, goal, name):
    """Each kernel emit predicate must succeed."""
    source = emitted_kernels.get(name)
    assert source is not None, f"{name} emit returned None"
    assert not source.startswith("FAIL:"), f"{name} emit failed: {source}"
    assert len(source) > 10, f"{name} emit produced suspiciously short output"


@pytest.mark.parametrize("goal,name", KERNEL_EMIT_CALLS,
                         ids=[name for _, name in KERNEL_EMIT_CALLS])
def test_division_precedence(emitted_kernels, goal, name):
    """No unparenthesized compound denominators in division expressions."""
    source = emitted_kernels.get(name)
    if source is None or source.startswith("FAIL:"):
        pytest.skip(f"{name} emit failed")
    
    matches = PRECEDENCE_RISK_PATTERN.findall(source)
    if matches:
        # Find line numbers for context
        lines = source.split('\n')
        risky_lines = []
        for i, line in enumerate(lines, 1):
            if PRECEDENCE_RISK_PATTERN.search(line):
                risky_lines.append(f"  L{i}: {line.strip()}")
        context = "\n".join(risky_lines[:5])
        pytest.fail(
            f"{name}: {len(matches)} precedence risk(s) found:\n{context}\n"
            f"Pattern: `/ id [*/] id` — may need c_paren wrapping"
        )


@pytest.mark.parametrize("goal,name", KERNEL_EMIT_CALLS,
                         ids=[name for _, name in KERNEL_EMIT_CALLS])
def test_brace_balance(emitted_kernels, goal, name):
    """Balanced braces, parens, and brackets in emitted CUDA."""
    source = emitted_kernels.get(name)
    if source is None or source.startswith("FAIL:"):
        pytest.skip(f"{name} emit failed")
    
    pairs = [('(', ')'), ('{', '}'), ('[', ']')]
    for open_c, close_c in pairs:
        opens = source.count(open_c)
        closes = source.count(close_c)
        assert opens == closes, \
            f"{name}: unbalanced '{open_c}{close_c}': {opens} opens vs {closes} closes"
