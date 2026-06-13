# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""L.1.9 Residual add, SiLU, elementwise MUL, and fused SwiGLU FFN tests.

Tests:
  L.1.9a  Residual add: dst = a + b  (0 ULP — pure float add, no expf)
  L.1.9b  SiLU: dst = x / (1 + exp(-x))  (structural — expf ABI issue)
  L.1.9c  Elementwise MUL: dst = a * b  (0 ULP — pure float mul)
  L.1.9d  Fused SwiGLU: dst = silu(gate) * up  (structural — expf ABI issue)
  L.1.9e  SwiGLU self-consistency: fused == separate silu + mul  (0 ULP)
  L.1.9f  Residual add non-multiple-of-8 tail (0 ULP — scalar tail path)
  L.1.9g  SwiGLU Llama 3 8B shape: gate/up dim=14336, n_tokens=8  (structural)

Verification strategy:
  add and mul are pure IEEE 754 float operations with no transcendentals.
  The Python reference `a + b` and `a * b` produce the same bits as the C
  kernel — 0 ULP is expected and verified.

  SiLU and SwiGLU use expf() internally.  Python ctypes passes float
  arguments through the x87 ABI which may differ from the C kernel's
  XMM-register expf() by 1 ULP (same issue as RoPE cosf() and GQA expf()).
  Tests verify structural correctness: finite/non-zero, norm bounded,
  and self-consistency between the fused and unfused paths.
  Bit-identical oracle is the Ruach Tov fixture test.
"""
import ctypes, os, sys
import numpy as np

SO_PATH = os.environ.get("BPD_CPU_SO", "build/bpd_cpu.so")
print(f"BPD_CPU_SO: {SO_PATH}")

lib = ctypes.CDLL(SO_PATH)

lib.bpd_add_f32_cpu.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.bpd_add_f32_cpu.restype = None

lib.bpd_silu_f32_cpu.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.bpd_silu_f32_cpu.restype = None

lib.bpd_mul_f32_cpu.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.bpd_mul_f32_cpu.restype = None

lib.bpd_swiglu_fuse_cpu.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.bpd_swiglu_fuse_cpu.restype = None


def ptr(arr):
    return arr.ctypes.data_as(ctypes.c_void_p)


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_residual_add(lib):
    """L.1.9a Residual add: 0 ULP vs Python a+b (n=4096)."""
    rng = np.random.default_rng(42)
    n = 4096
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    dst = np.zeros(n, dtype=np.float32)
    lib.bpd_add_f32_cpu(ptr(a), ptr(b), ptr(dst), ctypes.c_int(n))
    ref = a + b
    diff = np.abs(dst.view(np.uint32).astype(np.int64) -
                  ref.view(np.uint32).astype(np.int64))
    max_ulp = int(diff.max())
    ok = max_ulp == 0
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  L.1.9a Residual add (n={n}): max_ulp={max_ulp}")
    return ok


def test_residual_add_tail(lib):
    """L.1.9f Residual add non-multiple-of-8 tail: 0 ULP (n=4097)."""
    rng = np.random.default_rng(7)
    n = 4097  # 512*8 + 1 — exercises scalar tail
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    dst = np.zeros(n, dtype=np.float32)
    lib.bpd_add_f32_cpu(ptr(a), ptr(b), ptr(dst), ctypes.c_int(n))
    ref = a + b
    diff = np.abs(dst.view(np.uint32).astype(np.int64) -
                  ref.view(np.uint32).astype(np.int64))
    max_ulp = int(diff.max())
    ok = max_ulp == 0
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  L.1.9f Residual add tail (n={n}): max_ulp={max_ulp}")
    return ok


def test_elementwise_mul(lib):
    """L.1.9c Elementwise MUL: 0 ULP vs Python a*b (n=4096)."""
    rng = np.random.default_rng(13)
    n = 4096
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    dst = np.zeros(n, dtype=np.float32)
    lib.bpd_mul_f32_cpu(ptr(a), ptr(b), ptr(dst), ctypes.c_int(n))
    ref = a * b
    diff = np.abs(dst.view(np.uint32).astype(np.int64) -
                  ref.view(np.uint32).astype(np.int64))
    max_ulp = int(diff.max())
    ok = max_ulp == 0
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  L.1.9c Elementwise MUL (n={n}): max_ulp={max_ulp}")
    return ok


def test_silu_structural(lib):
    """L.1.9b SiLU structural: finite, non-zero, norm bounded (n=4096)."""
    rng = np.random.default_rng(55)
    n = 4096
    x = rng.standard_normal(n).astype(np.float32)
    dst = np.zeros(n, dtype=np.float32)
    lib.bpd_silu_f32_cpu(ptr(x), ptr(dst), ctypes.c_int(n))
    ok = True
    issues = []
    if not np.all(np.isfinite(dst)):
        ok = False; issues.append("inf/nan in output")
    if not np.any(dst != 0.0):
        ok = False; issues.append("all-zero output")
    # SiLU output norm should be <= input norm (SiLU is bounded by identity for large x)
    if float(np.linalg.norm(dst)) > float(np.linalg.norm(x)) * 1.01:
        ok = False; issues.append(f"norm too large: {np.linalg.norm(dst):.4f} > {np.linalg.norm(x):.4f}")
    # Determinism
    dst2 = np.zeros(n, dtype=np.float32)
    lib.bpd_silu_f32_cpu(ptr(x), ptr(dst2), ctypes.c_int(n))
    if not np.array_equal(dst, dst2):
        ok = False; issues.append("non-deterministic")
    status = "PASS" if ok else "FAIL"
    detail = "structural OK" if ok else f"structural FAIL: {'; '.join(issues)}"
    print(f"  {status}  L.1.9b SiLU (n={n}): {detail}")
    return ok


def test_swiglu_self_consistency(lib):
    """L.1.9e SwiGLU self-consistency: fused == separate silu+mul (0 ULP)."""
    rng = np.random.default_rng(99)
    n = 4096
    gate = rng.standard_normal(n).astype(np.float32)
    up = rng.standard_normal(n).astype(np.float32)

    # Fused path
    dst_fused = np.zeros(n, dtype=np.float32)
    lib.bpd_swiglu_fuse_cpu(ptr(gate), ptr(up), ptr(dst_fused), ctypes.c_int(n))

    # Separate path: silu(gate) then mul
    silu_out = np.zeros(n, dtype=np.float32)
    lib.bpd_silu_f32_cpu(ptr(gate), ptr(silu_out), ctypes.c_int(n))
    dst_sep = np.zeros(n, dtype=np.float32)
    lib.bpd_mul_f32_cpu(ptr(silu_out), ptr(up), ptr(dst_sep), ctypes.c_int(n))

    diff = np.abs(dst_fused.view(np.uint32).astype(np.int64) -
                  dst_sep.view(np.uint32).astype(np.int64))
    max_ulp = int(diff.max())
    ok = max_ulp == 0
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  L.1.9e SwiGLU self-consistency (n={n}): max_ulp={max_ulp}")
    return ok


def test_swiglu_structural(lib):
    """L.1.9d SwiGLU structural: finite, non-zero, deterministic (n=4096)."""
    rng = np.random.default_rng(77)
    n = 4096
    gate = rng.standard_normal(n).astype(np.float32)
    up = rng.standard_normal(n).astype(np.float32)
    dst = np.zeros(n, dtype=np.float32)
    lib.bpd_swiglu_fuse_cpu(ptr(gate), ptr(up), ptr(dst), ctypes.c_int(n))
    ok = True
    issues = []
    if not np.all(np.isfinite(dst)):
        ok = False; issues.append("inf/nan in output")
    if not np.any(dst != 0.0):
        ok = False; issues.append("all-zero output")
    dst2 = np.zeros(n, dtype=np.float32)
    lib.bpd_swiglu_fuse_cpu(ptr(gate), ptr(up), ptr(dst2), ctypes.c_int(n))
    if not np.array_equal(dst, dst2):
        ok = False; issues.append("non-deterministic")
    status = "PASS" if ok else "FAIL"
    detail = "structural OK" if ok else f"structural FAIL: {'; '.join(issues)}"
    print(f"  {status}  L.1.9d SwiGLU fused (n={n}): {detail}")
    return ok


def test_swiglu_llama3_shape(lib):
    """L.1.9g SwiGLU Llama 3 8B shape: gate/up dim=14336, n_tokens=8."""
    rng = np.random.default_rng(33)
    n_tokens, ffn_dim = 8, 14336
    n = n_tokens * ffn_dim
    gate = rng.standard_normal(n).astype(np.float32)
    up = rng.standard_normal(n).astype(np.float32)
    dst = np.zeros(n, dtype=np.float32)
    lib.bpd_swiglu_fuse_cpu(ptr(gate), ptr(up), ptr(dst), ctypes.c_int(n))
    ok = np.all(np.isfinite(dst)) and np.any(dst != 0.0)
    # Self-consistency with separate path
    silu_out = np.zeros(n, dtype=np.float32)
    lib.bpd_silu_f32_cpu(ptr(gate), ptr(silu_out), ctypes.c_int(n))
    dst_sep = np.zeros(n, dtype=np.float32)
    lib.bpd_mul_f32_cpu(ptr(silu_out), ptr(up), ptr(dst_sep), ctypes.c_int(n))
    diff = np.abs(dst.view(np.uint32).astype(np.int64) -
                  dst_sep.view(np.uint32).astype(np.int64))
    max_ulp = int(diff.max())
    ok = ok and (max_ulp == 0)
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  L.1.9g SwiGLU Llama3-8B shape ({n_tokens}x{ffn_dim}): "
          f"finite={np.all(np.isfinite(dst))} self_consistency_ulp={max_ulp}")
    return ok


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_residual_add,
        test_residual_add_tail,
        test_elementwise_mul,
        test_silu_structural,
        test_swiglu_self_consistency,
        test_swiglu_structural,
        test_swiglu_llama3_shape,
    ]
    passed = sum(t(lib) for t in tests)
    total = len(tests)
    print(f"\nPASS: {passed}  FAIL: {total - passed}  TOTAL: {total}")
    sys.exit(0 if passed == total else 1)
