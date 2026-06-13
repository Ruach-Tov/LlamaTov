#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""ir_submit_adapter.py — community kernel-exchange backend (MVP).

Receives LLVM-IR text (from a web form / agent / human), compiles it to a kernel via the llc
oracle, runs it through the 0-ULP test adapter against the reference, and returns a RATING.
This is the submission->test->rate loop that lets the whole community contribute kernels.

Rating dimensions (the leaderboard axes):
  - correctness: max-ULP vs reference (0 = bit-exact, the gold tier)
  - the kernel must EXPORT the expected symbol with the expected signature (per problem)
  - (future) speed: ns/element vs the reference

Security/license: IR is compiled in a sandboxed temp dir; submissions carry license terms.
This MVP proves the pipeline: ir_text -> llc -> .so -> referee -> {max_ulp, tier, accepted}.
"""
import subprocess, tempfile, os, ctypes, json, hashlib
import numpy as np

LLC = "llc"; CLANG = "clang"

def compile_ir(ir_text, symbol, workdir):
    """llc the IR oracle: ir_text -> .s -> .so. Returns (so_path, error)."""
    ll = os.path.join(workdir, "submit.ll")
    s  = os.path.join(workdir, "submit.s")
    so = os.path.join(workdir, "submit.so")
    open(ll, "w").write(ir_text)
    # llc: IR -> asm (the trusted lowering, no-AVX to match torch runtime)
    r1 = subprocess.run([LLC, "-O2", "-mattr=+sse2", "-o", s, ll],
                        capture_output=True, text=True, timeout=30)
    if r1.returncode != 0:
        return None, f"llc error: {r1.stderr[:300]}"
    # assemble+link to shared object
    r2 = subprocess.run([CLANG, "-shared", "-fPIC", "-O2", "-o", so, s, "-lm"],
                        capture_output=True, text=True, timeout=30)
    if r2.returncode != 0:
        return None, f"link error: {r2.stderr[:300]}"
    return so, None

def ulp(a, b):
    a = np.ascontiguousarray(a, np.float32).ravel(); b = np.ascontiguousarray(b, np.float32).ravel()
    n = min(a.size, b.size)
    ai = np.frombuffer(a[:n].tobytes(), np.int32).astype(np.int64)
    bi = np.frombuffer(b[:n].tobytes(), np.int32).astype(np.int64)
    u = np.abs(ai - bi); u[np.isnan(a[:n]) & np.isnan(b[:n])] = 0
    return int(u.max()), int((u > 0).sum())

def tier(max_ulp):
    if max_ulp == 0: return "GOLD (bit-exact 0-ULP)"
    if max_ulp <= 1: return "SILVER (1 ULP)"
    if max_ulp <= 4: return "BRONZE (<=4 ULP)"
    return f"UNRATED ({max_ulp} ULP)"

def test_unary_submission(ir_text, symbol, reference_fn, n=8192):
    """Test a unary-elementwise IR kernel: void sym(const float* x, float* y, int n)."""
    with tempfile.TemporaryDirectory() as wd:
        so, err = compile_ir(ir_text, symbol, wd)
        if err: return {"accepted": False, "error": err}
        try:
            lib = ctypes.CDLL(so)
            if not hasattr(lib, symbol):
                return {"accepted": False, "error": f"symbol '{symbol}' not exported"}
            fn = getattr(lib, symbol); fp = ctypes.c_void_p
            x = (np.random.RandomState(0).randn(n).astype(np.float32) * 4.0)
            ref = reference_fn(x)
            out = np.zeros(n, np.float32); xc = np.ascontiguousarray(x)
            fn(xc.ctypes.data_as(fp), out.ctypes.data_as(fp), ctypes.c_int(n))
            mx, nz = ulp(out, ref)
            return {"accepted": True, "max_ulp": mx, "nonzero": nz, "tier": tier(mx),
                    "ir_sha": hashlib.sha256(ir_text.encode()).hexdigest()[:12]}
        except Exception as e:
            return {"accepted": False, "error": f"runtime: {str(e)[:200]}"}

if __name__ == "__main__":
    # demo: a submitted ReLU kernel in LLVM-IR, tested against numpy relu
    import sys
    ir = open(sys.argv[1]).read() if len(sys.argv) > 1 else SAMPLE_RELU_IR
    sym = sys.argv[2] if len(sys.argv) > 2 else "submit_relu"
    res = test_unary_submission(ir, sym, lambda x: np.maximum(x, np.float32(0.0)))
    print(json.dumps(res, indent=2))
