# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""test_launch_contract.py — STATIC check that a generated kernel's LAUNCH contract
matches its actual thread-indexing, so a launch/kernel geometry MISMATCH (the CUPTI
'Not selected 31%' bug, 2026-06-08) is caught LOCALLY, before any GPU run.

THE BUG: maxpool K=4 emits the SIMPLE thread-per-output kernel, but an ad-hoc
harness launched it WARP-per-output (total*32 threads) -> 31/32 threads idle ->
~4x slower (and for some kernels, WRONG output). The launch contract was only a
prose comment; nothing checked it.

THE FIX (prevention, layered):
  1. Emitters now emit a machine-readable '// LAUNCH: <mode> ...' directive.
  2. THIS test verifies the directive AGREES with the kernel body's stride pattern:
       thread_per_output  <-> 'idx = ...blockIdx...threadIdx'  (1 thread/output)
       warp_per_output    <-> 'warp = (...) >> 5'              (1 warp/output)
     A kernel whose contract contradicts its code fails here.
  3. (runtime) callers compute geometry FROM the contract, not by hand — so they
     can't mismatch. (See referee/launch_from_contract.)

This is the 'detect + prevent' layer Heath asked for: the contract is the single
source of truth, and this test guards that the contract is honest.

Author: Iyun, 2026-06-08
"""
import os
import re
import shutil
import subprocess

import pytest

_TESTS = os.path.dirname(os.path.abspath(__file__))
_BPD = os.path.dirname(_TESTS)
_REPO = os.path.dirname(_BPD)
_EMIT = os.path.join(_BPD, "kernelgen", "emitters")
_FACTS = os.path.join(_BPD, "lib", "robust_op_match.pl")


def _swipl():
    return (shutil.which("swipl")
            or "/nix/store/jn4yixfq3qjdl3d4g6hfvl8nnn2pjhc5-swi-prolog-9.2.9/bin/swipl")


HAVE_SWIPL = os.path.exists(_swipl()) or shutil.which("swipl") is not None
skip_no_swipl = pytest.mark.skipif(not HAVE_SWIPL, reason="swipl required")


def _gen_pool(op, tmp):
    out = str(tmp / f"{op}.cu")
    goal = (f"use_module('{_FACTS}', [op_expr/2]), "
            f"consult('{_EMIT}/expr_ir.pl'), "
            f'emit_cuda_pool({op}, "v", "{out}"), halt')
    subprocess.run([_swipl(), "-q", "-g", goal, "-t", "halt"],
                   capture_output=True, text=True, timeout=60, cwd=_REPO)
    return open(out).read() if os.path.exists(out) else None


# (op, expected launch mode) — maxpool K=4 (KK=16<49) -> simple; avgpool K=11 -> warp
POOL_CASES = [
    ("bpd_maxpool2d", "thread_per_output"),
    ("bpd_avgpool2d", "warp_per_output"),
]


def _parse_contract(src):
    m = re.search(r"// LAUNCH:\s*(\w+)\s+total=(\S+)\s+threads=(\S+)", src)
    return m.groups() if m else None


def _detect_stride_class(src):
    """Infer the kernel's actual parallelization from its body."""
    # warp-per-output: 'warp = (...) >> 5'  (shift by 5 = /32)
    if re.search(r"\bwarp\s*=\s*\(.*\)\s*>>\s*5", src):
        return "warp_per_output"
    # thread-per-output: 'idx = ...blockIdx...threadIdx' with no warp shift
    if re.search(r"\bidx\s*=\s*\(long\)blockIdx", src) and ">> 5" not in src:
        return "thread_per_output"
    return "unknown"


@skip_no_swipl
@pytest.mark.parametrize("op,expected_mode", POOL_CASES)
def test_pool_has_launch_contract(op, expected_mode, tmp_path):
    src = _gen_pool(op, tmp_path)
    assert src, f"emit_cuda_pool({op}) produced nothing"
    contract = _parse_contract(src)
    assert contract, f"{op}: no machine-readable '// LAUNCH:' directive"
    mode, total, threads = contract
    assert mode == expected_mode, f"{op}: contract says {mode}, expected {expected_mode}"


@skip_no_swipl
@pytest.mark.parametrize("op,expected_mode", POOL_CASES)
def test_pool_contract_matches_body(op, expected_mode, tmp_path):
    """The contract must AGREE with the kernel's actual stride pattern — a kernel
    that claims thread_per_output but strides warp-wide (or vice versa) is the bug."""
    src = _gen_pool(op, tmp_path)
    assert src, f"emit_cuda_pool({op}) produced nothing"
    contract = _parse_contract(src)
    assert contract, f"{op}: no LAUNCH directive"
    declared = contract[0]
    actual = _detect_stride_class(src)
    assert actual == declared, (
        f"{op}: LAUNCH contract claims '{declared}' but the kernel body strides "
        f"'{actual}' — launch/kernel geometry mismatch (the CUPTI Not-selected bug).")


@skip_no_swipl
@pytest.mark.parametrize("op,expected_mode", POOL_CASES)
def test_threads_formula_matches_mode(op, expected_mode, tmp_path):
    """warp_per_output => threads=total*32 ; thread_per_output => threads=total."""
    src = _gen_pool(op, tmp_path)
    contract = _parse_contract(src)
    assert contract
    mode, total, threads = contract
    if mode == "warp_per_output":
        assert threads.endswith("*32"), f"{op}: warp mode but threads={threads} (expected total*32)"
    elif mode == "thread_per_output":
        assert threads == total, f"{op}: thread mode but threads={threads} != total={total}"
