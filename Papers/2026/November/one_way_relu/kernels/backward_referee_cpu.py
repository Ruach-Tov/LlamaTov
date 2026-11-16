#!/usr/bin/env python3
"""Bit-exact referee: emitted one-way backward (.ll via clang) vs the
VERIFIED torch autograd implementation (activations.py — shakedown-passed).
Region-separated per Iyun's protocol: x>0 / x==+0 / x==-0 / x<0 columns.
"""
import ctypes, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent.parent))
from activations import one_way_leaky

def ulp(a, b):
    a = np.ascontiguousarray(a, np.float32).ravel(); b = np.ascontiguousarray(b, np.float32).ravel()
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    nan_mismatch = int((nan_a != nan_b).sum())
    both_zero = (a == 0) & (b == 0)
    ai = np.frombuffer(a.tobytes(), np.int32).astype(np.int64)
    bi = np.frombuffer(b.tobytes(), np.int32).astype(np.int64)
    signzero = int((both_zero & (ai != bi)).sum())
    mask = ~(nan_a | nan_b) & ~both_zero
    u = np.abs(ai - bi); u[~mask] = 0
    return int(u.max()), int((u > 0).sum()), nan_mismatch, signzero

def inputs(n=8192):
    rng = np.random.default_rng(42)
    x = (rng.standard_normal(n) * 2).astype(np.float32)
    g = (rng.standard_normal(n) * 0.1).astype(np.float32)
    # seed exact zeros, signed zeros, boundary structure in BOTH tensors:
    x[::7] = 0.0
    x[3::14] = -0.0
    g[::5] = -np.abs(g[::5])      # guaranteed revival-sign population
    g[1::11] = 0.0                 # zero-gradient edge
    x[1] = np.float32(1.17549435e-38); x[2] = -np.float32(1.17549435e-38)  # denormal-adjacent
    return x, g

def torch_oracle(x, g):
    xt = torch.from_numpy(x.copy()).requires_grad_(True)
    y = one_way_leaky(xt, 0.01)
    y.backward(torch.from_numpy(g.copy()))
    return xt.grad.numpy()

if __name__ == '__main__':
    so = sys.argv[1]
    x, g = inputs()
    lib = ctypes.CDLL(so)
    out = np.empty_like(x)
    lib.oneway_backward(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        g.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int64(len(x)))
    want = torch_oracle(x, g)
    regions = {
        'x>0':    x > 0,
        'x==+0':  (x == 0) & (np.signbit(x) == False),
        'x==-0':  (x == 0) & np.signbit(x),
        'x<0':    x < 0,
    }
    all_ok = True
    for name, m in regions.items():
        mx, nd, nanm, sz = ulp(out[m], want[m])
        ok = (mx == 0 and nd == 0 and nanm == 0 and sz == 0)
        all_ok &= ok
        print(f'  region {name:7s} n={int(m.sum()):5d}: maxULP={mx} differ={nd} nan={nanm} ±0={sz} -> {"OK" if ok else "FAIL"}')
    mx, nd, nanm, sz = ulp(out, want)
    print(f'  TOTAL: maxULP={mx} differ={nd} -> {"BIT-EXACT" if all_ok else "DIFFERS"}')
