# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""test_kernelgen_lint.py — (4) Prolog LINT + (5) unified runner for the kernelgen
codegen test layer.

(4) LINT: consult each emitter / schedule module and flag swipl's free static
    analysis — singleton variables and undefined predicates (the typo/forgot-a-
    -clause class). swipl warns on these at load; we fail if any NEW ones appear.

(5) RUNNER: also drives the plunit backend-completeness check
    (test_backend_completeness.pl) from pytest, so ONE command runs the whole
    kernelgen codegen test layer:

      pytest bpd/tests/test_kernelgen_validity.py \
             bpd/tests/test_kernelgen_golden.py \
             bpd/tests/test_kernelgen_lint.py

    = structural validity (mlir-opt/nvcc) + golden snapshots + Prolog lint +
      backend completeness. Fast, mostly local, graceful skip without tools.

Known-benign warnings (load-order, not bugs) are allow-listed so the lint only
fails on NEW issues. Author: Iyun, 2026-06-08
"""
import os
import shutil
import subprocess

import pytest

_TESTS = os.path.dirname(os.path.abspath(__file__))
_BPD = os.path.dirname(_TESTS)
_REPO = os.path.dirname(_BPD)


def _swipl():
    return (shutil.which("swipl")
            or "/nix/store/jn4yixfq3qjdl3d4g6hfvl8nnn2pjhc5-swi-prolog-9.2.9/bin/swipl")


HAVE_SWIPL = os.path.exists(_swipl()) or shutil.which("swipl") is not None
skip_no_swipl = pytest.mark.skipif(not HAVE_SWIPL, reason="swipl required")

# Files to lint (the codegen emitters + the schedule IR layer).
EMITTERS = [
    "bpd/kernelgen/emitters/mlir_gpu_from_facts.pl",
    "bpd/kernelgen/schedule/schedule_ir.pl",
    "bpd/kernelgen/schedule/lower_schedule_cuda.pl",
    "bpd/kernelgen/schedule/lower_schedule_mlir.pl",
]

# Benign warnings (documented load-order, not bugs) — allow-listed substrings.
# Anything NOT matching these that says "Singleton" or "not defined" fails the lint.
ALLOW = [
    "op_expr/2",          # facts provided by caller (documented load-order)
    "op_expr",
]


def _lint(path):
    """Consult the file, capture swipl warnings. Return list of NON-benign warnings."""
    full = os.path.join(_REPO, path)
    r = subprocess.run([_swipl(), "-q", "-g", f"consult('{full}')", "-t", "halt"],
                       capture_output=True, text=True, timeout=30, cwd=_REPO)
    bad = []
    for ln in (r.stderr + r.stdout).splitlines():
        low = ln.lower()
        if "singleton" in low or ("not defined" in low) or ("undefined" in low):
            if not any(a in ln for a in ALLOW):
                bad.append(ln.strip())
    return bad


@skip_no_swipl
@pytest.mark.parametrize("path", EMITTERS)
def test_emitter_lints_clean(path):
    """No NEW singletons / undefined predicates (swipl check, the free linter)."""
    full = os.path.join(_REPO, path)
    if not os.path.exists(full):
        pytest.skip(f"{path} not present")
    bad = _lint(path)
    assert not bad, f"{path} has lint issues:\n  " + "\n  ".join(bad)


@skip_no_swipl
def test_backend_completeness_plunit():
    """(5) Drive the plunit backend-completeness check from pytest."""
    test_pl = os.path.join(_TESTS, "test_backend_completeness.pl")
    if not os.path.exists(test_pl):
        pytest.skip("test_backend_completeness.pl not present")
    r = subprocess.run([_swipl(), "-q", "-g", "run_tests", "-t", "halt", test_pl],
                       capture_output=True, text=True, timeout=60, cwd=_TESTS)
    # run_tests halt(1) on any failure -> nonzero rc
    assert r.returncode == 0, (
        f"backend completeness failed (rc={r.returncode}):\n{r.stdout[-1500:]}\n{r.stderr[-500:]}")
    assert "0 failed" in r.stdout, f"completeness did not report 0 failed:\n{r.stdout[-800:]}"
