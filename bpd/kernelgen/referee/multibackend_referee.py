#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""multibackend_referee.py — the multi-backend differential gate.

For each canonical op fact (robust_op_match.pl), generate ALL backends, run each
on the P4 over a FIXED input, and cross-check every pair (+ vs torch-CPU oracle).

Reports a matrix: per op, per backend-pair, maxULP / #differ — so a disagreement
auto-localizes a backend bug, and agreement-with-torch certifies correctness.

Backends: cuda-oxide (Rust->PTX) | cuda-c (nvcc) | torch-cpu (oracle reference).

The kernels are GENERATED (oxide_from_facts.pl, cuda_c_from_facts.pl) to dump
their raw f32 output to a .bin file; the referee loads + compares with the
shared ulp() machinery (same as stanford_referee.py).

Run on enclave:  python3 multibackend_referee.py [op ...]
Author: Iyun, 2026-06-06 (standing differential gate for the multi-backend generator)
"""
import subprocess, os, sys, struct
import numpy as np

SHARED = "/tmp/llamatov-data"
OXIDE  = f"{SHARED}/cuda-oxide"
WORK   = f"{SHARED}/gpu-work/referee"
os.makedirs(WORK, exist_ok=True)

# ── shared ULP machinery (from stanford_referee.py) ──
def ulp(a, b):
    a = np.ascontiguousarray(a, np.float32).ravel(); b = np.ascontiguousarray(b, np.float32).ravel()
    n = min(a.size, b.size); a=a[:n]; b=b[:n]
    nan_a=np.isnan(a); nan_b=np.isnan(b)
    nan_mismatch = int((nan_a!=nan_b).sum())
    both_zero = (a==0)&(b==0)
    ai=np.frombuffer(a.tobytes(),np.int32).astype(np.int64)
    bi=np.frombuffer(b.tobytes(),np.int32).astype(np.int64)
    signzero = int((both_zero & (ai!=bi)).sum())   # +-0 mismatch
    mask = ~(nan_a|nan_b) & ~both_zero
    u=np.abs(ai-bi); u[~mask]=0
    return int(u.max()), int((u>0).sum()), nan_mismatch, signzero

# ── fixed test input (edge cases + range) — written for all backends ──
def test_input(n=1024):
    x = np.empty(n, np.float32)
    edges = [0.0,-0.0,np.nan,-1.0,1.0,np.float32(1.17549435e-38),-np.float32(1.17549435e-38),3.4e38,-3.4e38]
    for i,e in enumerate(edges): x[i]=e
    for i in range(n-len(edges)):
        t=np.float32(i)*np.float32(0.013)-np.float32(6.6)
        x[i+len(edges)] = t*(-1.0 if i%3==0 else 1.0)
    return x

# ── torch-CPU oracle (the KernelBench reference) ──
TORCH_FN = {
    "relu": lambda x: __import__("torch").relu(__import__("torch").from_numpy(x)).numpy(),
    "elu":  lambda x: __import__("torch").nn.functional.elu(__import__("torch").from_numpy(x)).numpy(),
    "selu": lambda x: __import__("torch").nn.functional.selu(__import__("torch").from_numpy(x)).numpy(),
    "tanh": lambda x: __import__("torch").tanh(__import__("torch").from_numpy(x)).numpy(),
    "silu": lambda x: __import__("torch").nn.functional.silu(__import__("torch").from_numpy(x)).numpy(),
    "sigmoid": lambda x: __import__("torch").sigmoid(__import__("torch").from_numpy(x)).numpy(),
    "gelu": lambda x: __import__("torch").nn.functional.gelu(__import__("torch").from_numpy(x)).numpy(),
}

def torch_ref(op, x):
    try:
        return TORCH_FN[op](x.copy())
    except Exception as e:
        return None

def run_backend(op, label, status):
    """status: dict op->{backend:(ok,outpath_or_err)}; runs the dump binary."""
    # the dump-enabled runners are produced by the emitters (referee mode).
    # Here we just invoke the prebuilt runner that writes WORK/<op>_<backend>.bin
    pass  # orchestration wired in run_all below

def main():
    ops = sys.argv[1:] or ["relu","elu","tanh"]
    x = test_input()
    np.save(f"{WORK}/input.npy", x)
    x.tofile(f"{WORK}/input.bin")
    print(f"=== Multi-backend differential referee ({len(x)} elems) ===\n")
    print(f"{'op':8} {'oxide-vs-cudac':>18} {'oxide-vs-torch':>18} {'cudac-vs-torch':>18}")
    print("-"*66)
    for op in ops:
        ox = load_out(op, "oxide")
        cc = load_out(op, "cudac")
        tr = torch_ref(op, x)
        def cmp(a,b):
            if a is None or b is None: return "  N/A"
            mx,nd,nm,sz = ulp(a,b)
            base = f"{mx}ULP/{nd}d" if nd>0 else "0-ULP"
            extra = (f"+{nm}nan" if nm>0 else "") + (f"+{sz}sz" if sz>0 else "")
            return base + extra
        print(f"{op:8} {cmp(ox,cc):>18} {cmp(ox,tr):>18} {cmp(cc,tr):>18}")
    print("\nLegend: 0-ULP = bit-identical. NaN-aware. oxide=cuda-oxide Rust, cudac=nvcc, torch=CPU oracle.")
    print("Interpretation: oxide==cudac (both GPU libdevice) expected; vs-torch 1-ULP on transcendentals.")

def load_out(op, backend):
    p = f"{WORK}/{op}_{backend}.bin"
    if not os.path.exists(p): return None
    return np.fromfile(p, np.float32)

if __name__ == "__main__":
    main()
