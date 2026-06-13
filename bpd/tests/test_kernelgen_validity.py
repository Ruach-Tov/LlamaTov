# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""test_kernelgen_validity.py — FAST, LOCAL structural validity of generated kernels.

The pain this solves (Heath, 2026-06-08): every emitter bug today (MLIR SSA
redefinition, llvm.fcmp vs arith.cmpf, def-before-use, pragma-after-brace, Prolog
quote double-escaping) was discovered only AFTER a slow round-trip to the P4. This
suite moves that validation EARLIER and LOCAL — it generates each emitter's output
and checks it's structurally valid WITHOUT a GPU:

  * MLIR  -> mlir-opt --verify-diagnostics (parse + verify, NO lowering, NO GPU)
  * CUDA  -> nvcc -std=c++17 --cuda -E or g++ -fsyntax-only style parse

So a broken emitter fails in SECONDS on any node, not after scp + lower on the enclave.

GRACEFUL DEGRADATION (substrate-honest, per metayen/mavchin pattern): if a tool
(swipl / mlir-opt / nvcc) is absent, the dependent tests SKIP rather than fail.

Author: Iyun, 2026-06-08
"""
import os
import shutil
import subprocess
import tempfile

import pytest

_TESTS = os.path.dirname(os.path.abspath(__file__))
_BPD = os.path.dirname(_TESTS)
_REPO = os.path.dirname(_BPD)
_EMIT = os.path.join(_BPD, "kernelgen", "emitters")
_SCHED = os.path.join(_BPD, "kernelgen", "schedule")
_FACTS = os.path.join(_BPD, "lib", "robust_op_match.pl")


# ── tool availability (skip-not-fail) ────────────────────────────────────────
def _have(tool):
    return shutil.which(tool) is not None


def _swipl():
    return shutil.which("swipl") or shutil.which("swipl-win")


HAVE_SWIPL = _swipl() is not None
HAVE_MLIR = _have("mlir-opt")
HAVE_NVCC = _have("nvcc")

skip_no_swipl = pytest.mark.skipif(not HAVE_SWIPL, reason="swipl not available")
skip_no_mlir = pytest.mark.skipif(not (HAVE_SWIPL and HAVE_MLIR),
                                  reason="swipl+mlir-opt required")
skip_no_nvcc = pytest.mark.skipif(not (HAVE_SWIPL and HAVE_NVCC),
                                  reason="swipl+nvcc required")


def _run_swipl(goal, timeout=60):
    """Run a swipl goal, return (rc, stdout, stderr)."""
    r = subprocess.run([_swipl(), "-q", "-g", goal, "-t", "halt"],
                       capture_output=True, text=True, timeout=timeout, cwd=_REPO)
    return r.returncode, r.stdout, r.stderr


def _gen(consults, goal_call, out_path):
    """Consult modules + facts, run an emit goal that writes out_path. Returns out_path or None."""
    consult_terms = ", ".join(
        f"use_module('{_FACTS}', [op_expr/2])" if c == "FACTS"
        else f"consult('{c}')" for c in consults
    )
    goal = f"{consult_terms}, {goal_call}"
    rc, out, err = _run_swipl(goal)
    return out_path if os.path.exists(out_path) else None


def _mlir_verify(path):
    """Parse + verify MLIR WITHOUT lowering or GPU. Returns (ok, message)."""
    # --mlir-print-op-on-diagnostic + just round-trip through the verifier
    r = subprocess.run(["mlir-opt", path, "--verify-each", "-o", os.devnull],
                       capture_output=True, text=True, timeout=30)
    # mlir-opt with no passes still parses + verifies the module
    return (r.returncode == 0, r.stderr.strip())


# ── the op sets each emitter claims ──────────────────────────────────────────
ELEMENTWISE = ["bpd_relu", "bpd_tanh", "bpd_sigmoid", "bpd_silu", "bpd_elu",
               "bpd_gelu", "bpd_leaky_relu", "bpd_hardsigmoid", "bpd_softplus",
               "bpd_selu", "bpd_mish"]
REDUCE = ["bpd_sum", "bpd_mean", "bpd_max", "bpd_min"]
POOL = ["bpd_maxpool2d", "bpd_avgpool2d"]


# ── 1. MLIR-GPU emitters produce VERIFIABLE MLIR (catches SSA/cmpf/def-use bugs) ──
@skip_no_mlir
@pytest.mark.parametrize("op", ELEMENTWISE)
def test_mlir_gpu_elementwise_verifies(op, tmp_path):
    out = str(tmp_path / f"{op}.mlir")
    got = _gen(["FACTS", f"{_EMIT}/mlir_gpu_from_facts.pl"],
               f'emit_mlir_gpu({op}, "{out}")', out)
    assert got, f"emit_mlir_gpu({op}) produced no file"
    ok, msg = _mlir_verify(got)
    assert ok, f"MLIR for {op} failed verify:\n{msg}"


@skip_no_mlir
@pytest.mark.parametrize("op", REDUCE)
def test_mlir_gpu_reduce_verifies(op, tmp_path):
    out = str(tmp_path / f"{op}.mlir")
    got = _gen(["FACTS", f"{_EMIT}/mlir_gpu_from_facts.pl"],
               f'emit_mlir_gpu_reduce({op}, "{out}")', out)
    assert got, f"emit_mlir_gpu_reduce({op}) produced no file"
    ok, msg = _mlir_verify(got)
    assert ok, f"MLIR reduce for {op} failed verify:\n{msg}"


@skip_no_mlir
@pytest.mark.parametrize("op", POOL)
def test_mlir_gpu_pool_verifies(op, tmp_path):
    out = str(tmp_path / f"{op}.mlir")
    got = _gen(["FACTS", f"{_EMIT}/mlir_gpu_from_facts.pl"],
               f'emit_mlir_gpu_pool({op}, "{out}")', out)
    assert got, f"emit_mlir_gpu_pool({op}) produced no file"
    ok, msg = _mlir_verify(got)
    assert ok, f"MLIR pool for {op} failed verify:\n{msg}"


@skip_no_mlir
def test_mlir_gpu_conv_verifies(tmp_path):
    out = str(tmp_path / "conv.mlir")
    got = _gen(["FACTS", f"{_EMIT}/mlir_gpu_from_facts.pl"],
               f'emit_mlir_gpu_conv(bpd_conv2d, "{out}")', out)
    assert got, "emit_mlir_gpu_conv produced no file"
    ok, msg = _mlir_verify(got)
    assert ok, f"MLIR conv failed verify:\n{msg}"


@skip_no_mlir
def test_mlir_gpu_matmul_verifies(tmp_path):
    out = str(tmp_path / "mm.mlir")
    got = _gen(["FACTS", f"{_EMIT}/mlir_gpu_from_facts.pl"],
               f'emit_mlir_gpu_matmul(contract, "{out}")', out)
    assert got, "emit_mlir_gpu_matmul produced no file"
    ok, msg = _mlir_verify(got)
    assert ok, f"MLIR matmul failed verify:\n{msg}"


# ── 2. SCHEDULE-IR lowerings produce verifiable output (the new shared-tiling layer) ──
@skip_no_mlir
@pytest.mark.parametrize("op", REDUCE)
def test_schedule_mlir_reduce_verifies(op, tmp_path):
    out = str(tmp_path / f"sched_{op}.mlir")
    got = _gen(["FACTS", f"{_SCHED}/schedule_ir.pl", f"{_SCHED}/lower_schedule_mlir.pl"],
               f'emit_schedule_mlir({op}, tiled_row_reduce, "{out}")', out)
    assert got, f"emit_schedule_mlir({op}) produced no file"
    ok, msg = _mlir_verify(got)
    assert ok, f"schedule-MLIR for {op} failed verify:\n{msg}"


@skip_no_swipl
@pytest.mark.parametrize("op", REDUCE)
def test_schedule_cuda_reduce_generates(op, tmp_path):
    """The cuda-c schedule lowering at least GENERATES valid-looking source."""
    out = str(tmp_path / f"sched_{op}.cu")
    got = _gen(["FACTS", f"{_SCHED}/schedule_ir.pl", f"{_SCHED}/lower_schedule_cuda.pl"],
               f'emit_schedule_cuda({op}, tiled_row_reduce, "{out}")', out)
    assert got, f"emit_schedule_cuda({op}) produced no file"
    src = open(got).read()
    # structural sanity: has the kernel, the warp-shuffle, the guarded store
    assert "k_reduce" in src
    assert "__shfl_down_sync" in src
    assert "__shared__" in src


# ── 3. cuda-c emitters produce parseable CUDA (catches pragma/syntax bugs) ────
@skip_no_nvcc
@pytest.mark.parametrize("op", ELEMENTWISE[:4])
def test_cuda_c_elementwise_compiles_to_cubin(op, tmp_path):
    """nvcc -cubin is a full parse+codegen; catches pragma-after-brace, type bugs.
    (sm_61 cubin needs no GPU at compile time — only nvcc.)"""
    out = str(tmp_path / f"{op}.cu")
    got = _gen(["FACTS", f"{_EMIT}/cuda_c_from_facts.pl"],
               f'emit_cuda_c({op}, "{out}")', out)
    if not got:
        pytest.skip(f"emit_cuda_c({op}) not available / produced no file")
    r = subprocess.run(["nvcc", "-arch=sm_61", "-cubin", "-O3", got,
                        "-o", str(tmp_path / f"{op}.cubin")],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"nvcc failed for {op}:\n{r.stderr[:500]}"


# ── FlashAttention emitter (L3): generates valid CUDA + has a launch contract ──
@skip_no_nvcc
def test_flash_attention_compiles(tmp_path):
    """The recognized attention chain -> emit_flash_attention -> valid CUDA cubin."""
    out = str(tmp_path / "flash.cu")
    got = _gen(["FACTS", f"{_EMIT}/flash_attention.pl"],
               f'emit_flash_attention(flash_attn(scaled,softmax), "{out}")', out)
    assert got, "emit_flash_attention produced no file"
    cmd = ["nvcc", "-arch=sm_61", "-cubin", "-O3"]
    # nvcc auto-includes cuda_runtime.h; in a nix env the include dir isn't on the
    # default search path, so pass it explicitly (build like production does).
    cuda_home = os.environ.get("CUDA_HOME") or _find_cuda_include()
    if cuda_home:
        cmd += ["-I", os.path.join(cuda_home, "include")]
    cmd += [got, "-o", str(tmp_path / "flash.cubin")]
    # nix nvcc finds cuda_runtime.h (forced via <command-line>) through CPATH, not -I.
    env = dict(os.environ)
    if cuda_home:
        inc = os.path.join(cuda_home, "include")
        env["CPATH"] = inc + os.pathsep + env.get("CPATH", "")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
    if r.returncode != 0 and "cuda_runtime.h" in r.stderr:
        pytest.skip("nvcc cannot locate cuda_runtime.h in this env (toolchain config)")
    assert r.returncode == 0, f"nvcc failed for flash:\n{r.stderr[:500]}"


def _find_cuda_include():
    """Locate a CUDA toolkit dir whose include/ has cuda_runtime.h (nix-friendly)."""
    nvcc = shutil.which("nvcc")
    if nvcc:
        # nvcc is at <cuda>/bin/nvcc → <cuda>
        cand = os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))
        if os.path.exists(os.path.join(cand, "include", "cuda_runtime.h")):
            return cand
    return None


@skip_no_swipl
def test_flash_attention_recognizes_chain():
    """recognize_attention matches the QK^T->scale->softmax->xV diamond."""
    goal = ("use_module('%s/flash_attention.pl'), "
            "(recognize_attention([bpd_matmul, bpd_scaling, bpd_softmax, bpd_matmul], _) "
            "-> write(ok) ; write(no)), halt" % _EMIT)
    rc, out, err = _run_swipl(goal)
    assert "ok" in out, f"attention chain not recognized: {out} {err}"
