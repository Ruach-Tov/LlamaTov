#!/usr/bin/env python3
"""Verification-side harness checks (Bocher's lane; Iyun's flags 1+4).

Check 1 — α=0 STRUCTURAL (anti-tautology): the emitted leaky/one-way
kernel source must contain NO branch on α's value: α enters as a
constant/scalar operand in the dataflow, never as a compile-time special
case. We grep the emitted source for conditional structures keyed on the
alpha constant; the general-arithmetic guarantee is textual, so grep IS
the right tool (claim is about the text).

Check 2 — EMITTED-vs-PROTOTYPE equivalence: compile both .ll files,
run both on the same edge-seeded inputs, demand 0-ULP agreement in all
four regions (x>0 / +0 / −0 / x<0). The canonical emit must reproduce
the hand-written spec-by-example bit-for-bit.

Usage: python3 harness_checks.py <emitted.ll> <prototype.ll>
"""
import ctypes, re, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

def check_alpha_structure(ll_path):
    src = Path(ll_path).read_text()
    # alpha appears as an fp constant operand (select/fmul), never in a
    # branch condition (br i1 ... derived from comparing ALPHA itself):
    alpha_hex = '0x3F847AE140000000'
    ok_uses = len(re.findall(rf'select .*{alpha_hex}|fmul .*{alpha_hex}|fadd .*{alpha_hex}', src))
    # any compare-against-alpha (would indicate special-casing on alpha value):
    bad = re.findall(rf'fcmp [a-z]+ (float|double) .*{alpha_hex}', src)
    return ok_uses > 0 and not bad, f'alpha operand uses={ok_uses}, compares-on-alpha={len(bad)}'

def compile_ll(ll_path, out_so):
    r = subprocess.run(['clang', '-O2', '-shared', '-fPIC', str(ll_path), '-o', str(out_so)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[:500])

def run_kernel(so_path, x, g):
    lib = ctypes.CDLL(str(so_path))
    out = np.empty_like(x)
    lib.oneway_backward(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        g.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int64(len(x)))
    return out

def inputs(n=8192):
    rng = np.random.default_rng(42)
    x = (rng.standard_normal(n) * 2).astype(np.float32)
    g = (rng.standard_normal(n) * 0.1).astype(np.float32)
    x[::7] = 0.0; x[3::14] = -0.0
    g[::5] = -np.abs(g[::5]); g[1::11] = 0.0
    x[1] = np.float32(1.17549435e-38); x[2] = -np.float32(1.17549435e-38)
    return x, g

def region_diff(a, b, x):
    ai = np.frombuffer(np.ascontiguousarray(a).tobytes(), np.int32)
    bi = np.frombuffer(np.ascontiguousarray(b).tobytes(), np.int32)
    exact = (ai == bi) | (np.isnan(a) & np.isnan(b))
    regions = {'x>0': x > 0, 'x==+0': (x == 0) & ~np.signbit(x),
               'x==-0': (x == 0) & np.signbit(x), 'x<0': x < 0}
    all_ok = True
    for name, m in regions.items():
        nd = int((~exact[m]).sum())
        all_ok &= (nd == 0)
        print(f'  {name:7s} n={int(m.sum()):5d} differ={nd} -> {"OK" if nd==0 else "FAIL"}')
    return all_ok

if __name__ == '__main__':
    emitted, proto = sys.argv[1], sys.argv[2]
    ok1, msg = check_alpha_structure(emitted)
    print(f'CHECK 1 (alpha structural): {msg} -> {"PASS" if ok1 else "FAIL"}')
    with tempfile.TemporaryDirectory() as td:
        so_e, so_p = Path(td)/'e.so', Path(td)/'p.so'
        compile_ll(emitted, so_e); compile_ll(proto, so_p)
        x, g = inputs()
        a, b = run_kernel(so_e, x, g), run_kernel(so_p, x, g)
        print('CHECK 2 (emitted vs prototype, 4-region bit-exact):')
        ok2 = region_diff(a, b, x)
    print(f'VERDICT: {"PASS" if (ok1 and ok2) else "FAIL"}')
