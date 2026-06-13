# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""test_kernelgen_golden.py — GOLDEN-SNAPSHOT regression for generated kernels.

The "refactor freely" safety net (Heath, 2026-06-08). The validity suite
(test_kernelgen_validity.py) proves generated kernels are STRUCTURALLY valid;
this suite proves they HAVEN'T CHANGED unexpectedly. For each (emitter x op) it
regenerates the source and diffs against a committed golden file. Any change to
an emitter that alters output fails LOUDLY with a readable unified diff — so you
know EXACTLY what moved when you touch schedule_ir.pl or an emitter.

  * Pure SOURCE-TEXT regression — NO GPU, NO compile. Runs anywhere swipl exists.
  * Complements test_gpu_regression.py (which is OUTPUT-VALUE / 0-ULP, GPU-bound).
  * Goldens live in bpd/tests/golden/ and are committed.

WORKFLOW:
  pytest test_kernelgen_golden.py            # check against goldens (fails on diff)
  UPDATE_GOLDEN=1 pytest test_kernelgen_golden.py   # regenerate goldens (after an
                                                     # INTENTIONAL emitter change)

When a test fails: read the diff. If the change was intended, re-run with
UPDATE_GOLDEN=1 and commit the updated golden alongside the emitter change — so
the golden delta is reviewable in the same commit as the code delta.

Author: Iyun, 2026-06-08
"""
import difflib
import os
import shutil
import subprocess

import pytest

_TESTS = os.path.dirname(os.path.abspath(__file__))
_BPD = os.path.dirname(_TESTS)
_REPO = os.path.dirname(_BPD)
_EMIT = os.path.join(_BPD, "kernelgen", "emitters")
_SCHED = os.path.join(_BPD, "kernelgen", "schedule")
_FACTS = os.path.join(_BPD, "lib", "robust_op_match.pl")
_GOLDEN = os.path.join(_TESTS, "golden")

UPDATE = os.environ.get("UPDATE_GOLDEN", "") not in ("", "0", "false")


def _swipl():
    return shutil.which("swipl") or "/nix/store/jn4yixfq3qjdl3d4g6hfvl8nnn2pjhc5-swi-prolog-9.2.9/bin/swipl"


HAVE_SWIPL = os.path.exists(_swipl()) or shutil.which("swipl") is not None
skip_no_swipl = pytest.mark.skipif(not HAVE_SWIPL, reason="swipl required")


def _generate(consults, goal_call, tmp_out):
    """Consult modules+facts, run an emit goal writing tmp_out, return its text (or None)."""
    consult_terms = ", ".join(
        f"use_module('{_FACTS}', [op_expr/2])" if c == "FACTS"
        else f"consult('{c}')" for c in consults
    )
    goal = f"{consult_terms}, {goal_call}"
    subprocess.run([_swipl(), "-q", "-g", goal, "-t", "halt"],
                   capture_output=True, text=True, timeout=60, cwd=_REPO)
    if not os.path.exists(tmp_out):
        return None
    return open(tmp_out).read()


def _normalize(text):
    """Strip volatile bits (absolute paths in the '-> /tmp/...' generation banner)
    so goldens are path-independent. Keeps the kernel body verbatim."""
    lines = []
    for ln in text.splitlines():
        # generation banners sometimes echo the absolute out-path; drop after '->'
        if " -> /" in ln and (ln.startswith("//") or ln.startswith("/*")):
            ln = ln.split(" -> /")[0] + " -> <path>"
        lines.append(ln)
    return "\n".join(lines) + "\n"


def _check_golden(name, text):
    """Compare normalized text to golden file name. UPDATE_GOLDEN=1 rewrites it."""
    os.makedirs(_GOLDEN, exist_ok=True)
    path = os.path.join(_GOLDEN, name)
    norm = _normalize(text)
    if UPDATE or not os.path.exists(path):
        open(path, "w").write(norm)
        if not UPDATE:
            pytest.skip(f"golden {name} created (first run) — commit it")
        return
    golden = open(path).read()
    if norm != golden:
        diff = "\n".join(difflib.unified_diff(
            golden.splitlines(), norm.splitlines(),
            fromfile=f"golden/{name}", tofile="generated", lineterm=""))
        pytest.fail(f"generated output changed for {name}.\n"
                    f"If intentional: UPDATE_GOLDEN=1 pytest -k {name.split('.')[0]}\n\n{diff}")


# ── op sets ──────────────────────────────────────────────────────────────────
ELEMENTWISE = ["bpd_relu", "bpd_leaky_relu", "bpd_hardsigmoid", "bpd_softplus", "bpd_gelu"]
REDUCE = ["bpd_sum", "bpd_mean", "bpd_max", "bpd_min"]
POOL = ["bpd_maxpool2d", "bpd_avgpool2d"]


# ── MLIR-GPU goldens ─────────────────────────────────────────────────────────
@skip_no_swipl
@pytest.mark.parametrize("op", ELEMENTWISE)
def test_golden_mlir_elementwise(op, tmp_path):
    out = str(tmp_path / f"{op}.mlir")
    text = _generate(["FACTS", f"{_EMIT}/mlir_gpu_from_facts.pl"],
                     f'emit_mlir_gpu({op}, "{out}")', out)
    assert text, f"emit_mlir_gpu({op}) produced nothing"
    _check_golden(f"mlir_{op}.mlir", text)


@skip_no_swipl
@pytest.mark.parametrize("op", REDUCE)
def test_golden_mlir_reduce(op, tmp_path):
    out = str(tmp_path / f"{op}.mlir")
    text = _generate(["FACTS", f"{_EMIT}/mlir_gpu_from_facts.pl"],
                     f'emit_mlir_gpu_reduce({op}, "{out}")', out)
    assert text, f"emit_mlir_gpu_reduce({op}) produced nothing"
    _check_golden(f"mlir_reduce_{op}.mlir", text)


@skip_no_swipl
@pytest.mark.parametrize("op", POOL)
def test_golden_mlir_pool(op, tmp_path):
    out = str(tmp_path / f"{op}.mlir")
    text = _generate(["FACTS", f"{_EMIT}/mlir_gpu_from_facts.pl"],
                     f'emit_mlir_gpu_pool({op}, "{out}")', out)
    assert text, f"emit_mlir_gpu_pool({op}) produced nothing"
    _check_golden(f"mlir_pool_{op}.mlir", text)


@skip_no_swipl
def test_golden_mlir_conv(tmp_path):
    out = str(tmp_path / "conv.mlir")
    text = _generate(["FACTS", f"{_EMIT}/mlir_gpu_from_facts.pl"],
                     f'emit_mlir_gpu_conv(bpd_conv2d, "{out}")', out)
    assert text, "emit_mlir_gpu_conv produced nothing"
    _check_golden("mlir_conv2d.mlir", text)


@skip_no_swipl
def test_golden_mlir_matmul(tmp_path):
    out = str(tmp_path / "mm.mlir")
    text = _generate(["FACTS", f"{_EMIT}/mlir_gpu_from_facts.pl"],
                     f'emit_mlir_gpu_matmul(contract, "{out}")', out)
    assert text, "emit_mlir_gpu_matmul produced nothing"
    _check_golden("mlir_matmul.mlir", text)


# ── schedule-IR goldens (the new shared-tiling layer — most worth snapshotting) ──
@skip_no_swipl
@pytest.mark.parametrize("op", REDUCE)
def test_golden_schedule_cuda(op, tmp_path):
    out = str(tmp_path / f"{op}.cu")
    text = _generate(["FACTS", f"{_SCHED}/schedule_ir.pl", f"{_SCHED}/lower_schedule_cuda.pl"],
                     f'emit_schedule_cuda({op}, tiled_row_reduce, "{out}")', out)
    assert text, f"emit_schedule_cuda({op}) produced nothing"
    _check_golden(f"schedule_cuda_{op}.cu", text)


@skip_no_swipl
@pytest.mark.parametrize("op", REDUCE)
def test_golden_schedule_mlir(op, tmp_path):
    out = str(tmp_path / f"{op}.mlir")
    text = _generate(["FACTS", f"{_SCHED}/schedule_ir.pl", f"{_SCHED}/lower_schedule_mlir.pl"],
                     f'emit_schedule_mlir({op}, tiled_row_reduce, "{out}")', out)
    assert text, f"emit_schedule_mlir({op}) produced nothing"
    _check_golden(f"schedule_mlir_{op}.mlir", text)


# ── golden snapshots for the SCHEDULE-DERIVED kernels (infra-debt #5) ─────────
# These were generated this session but never snapshotted. Now a change to the
# flash schedule, the tiled_gemm lowering, or norm/softmax emitters fails loudly.

def test_golden_flash_schedule(tmp_path):
    out = str(tmp_path / "flash_sched.cu")
    text = _generate([f"{_EMIT}/flash_attention.pl"],
                     'recognize_attention([bpd_matmul,bpd_softmax,bpd_matmul], Spec), '
                     'flash_attn_schedule(tuned_d128, Sch), '
                     f'emit_flash_schedule(Spec, Sch, 128, "{out}")', out)
    assert text, "flash schedule produced nothing"
    _check_golden("flash_schedule_tuned_d128.cu", text)


def test_golden_tiled_gemm_cuda(tmp_path):
    out = str(tmp_path / "tg.cu")
    text = _generate(["FACTS", f"{_SCHED}/schedule_ir.pl", f"{_SCHED}/lower_schedule_cuda.pl"],
                     f'emit_schedule_cuda(bpd_matmul, tiled_gemm(128,128,32,8,4), "{out}")', out)
    assert text, "tiled_gemm cuda produced nothing"
    _check_golden("tiled_gemm_cuda_128.cu", text)


def test_golden_rmsnorm_from_fact(tmp_path):
    out = str(tmp_path / "rms.cu")
    text = _generate(["FACTS", f"{_EMIT}/norm_softmax_from_facts.pl"],
                     'op_expr(bpd_rmsnorm, R), '
                     f'emit_from_fact(R, [], "{out}")', out)
    assert text, "rmsnorm produced nothing"
    _check_golden("rmsnorm_from_fact.cu", text)


def test_golden_softmax_from_fact(tmp_path):
    out = str(tmp_path / "sm.cu")
    text = _generate(["FACTS", f"{_EMIT}/norm_softmax_from_facts.pl"],
                     'op_expr(bpd_softmax, Sm), '
                     f'emit_from_fact(Sm, [], "{out}")', out)
    assert text, "softmax produced nothing"
    _check_golden("softmax_from_fact.cu", text)


def test_golden_q8_0_scalar(tmp_path):
    out = str(tmp_path / "q8s.cu")
    text = _generate([f"{_EMIT}/q8_0_from_facts.pl"],
                     f'q8_0_op_expr(E), emit_from_fact(E, [mode(scalar)], "{out}")', out)
    assert text, "q8_0 scalar produced nothing"
    _check_golden("q8_0_gemv_scalar.cu", text)


def test_golden_q8_0_dp4a(tmp_path):
    out = str(tmp_path / "q8d.cu")
    text = _generate([f"{_EMIT}/q8_0_from_facts.pl"],
                     f'q8_0_op_expr(E), emit_from_fact(E, [mode(dp4a)], "{out}")', out)
    assert text, "q8_0 dp4a produced nothing"
    _check_golden("q8_0_gemv_dp4a.cu", text)
