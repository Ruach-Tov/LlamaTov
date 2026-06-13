#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""stanford_referee.py — authoritative L1 referee, registry v2.
Reference is ALWAYS prob.Model().forward() (both Stanford gate AND robust gate) — no hand-lambdas.
"""
import ctypes, os, sys, importlib.util
import numpy as np
import torch

SO = os.environ.get("BPD_CPU_SO", "")
L1_DIR = os.environ.get("KB_L1", "/tmp/KernelBench/KernelBench/level1")
lib = ctypes.CDLL(SO)
fpv = ctypes.c_void_p
FULL = "--full" in sys.argv

REG = [
  ("matmul", "1_Square_matrix_multiplication_.py", "bpd_mm_cpu"),
  ("unary", "19_ReLU.py",        "bpd_relu_cpu"),
  ("unary", "20_LeakyReLU.py",   "bpd_leaky_relu_cpu"),
  ("unary", "21_Sigmoid.py",     "bpd_sigmoid_cpu"),
  ("unary", "22_Tanh.py",        "bpd_tanh_cpu"),
  ("unary", "25_Swish.py",       "bpd_swish_cpu"),
  ("unary", "26_GELU_.py",       "bpd_gelu_cpu"),
  ("unary", "27_SELU_.py",       "bpd_selu_cpu"),
  ("unary", "28_HardSigmoid.py", "bpd_hardsigmoid_cpu"),
  ("unary", "29_Softplus.py",    "bpd_softplus_cpu"),
  ("unary", "30_Softsign.py",    "bpd_softsign_cpu"),
  ("unary", "31_ELU.py",         "bpd_elu_cpu"),
]

def ulp(a, b):
    a = np.ascontiguousarray(a, np.float32).ravel(); b = np.ascontiguousarray(b, np.float32).ravel()
    n = min(a.size, b.size); a=a[:n]; b=b[:n]
    ai = np.frombuffer(a.tobytes(), np.int32).astype(np.int64)
    bi = np.frombuffer(b.tobytes(), np.int32).astype(np.int64)
    u = np.abs(ai - bi); u[np.isnan(a)&np.isnan(b)] = 0
    return int(u.max()), int((u>0).sum())

def load_problem(fname):
    path = os.path.join(L1_DIR, fname)
    if not os.path.exists(path): return None
    spec = importlib.util.spec_from_file_location("kb_prob", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def call_unary(fn_name, x):
    if not hasattr(lib, fn_name): return None
    fn = getattr(lib, fn_name); out = np.zeros(x.size, np.float32)
    xc = np.ascontiguousarray(x, np.float32).ravel()
    fn(xc.ctypes.data_as(fpv), out.ctypes.data_as(fpv), ctypes.c_int(xc.size)); return out

def robust_vec(n=8192):
    edges = np.array([0.0,-0.0,1.0,-1.0,0.5,-0.5,88.0,-88.0,1e-30,-1e-30,
                      np.float32(1.4e-45),1e20,-1e20,10.0,-10.0], np.float32)
    return np.concatenate([np.random.RandomState(1).randn(n).astype(np.float32)*5.0, edges]).astype(np.float32)

def referee_unary(prob, fn_name):
    # Stanford gate: Model.forward() on Stanford-distribution input, small + (full) large
    shapes = [(64,1024),(256,1024)] + ([(512,4096)] if FULL else [])
    rows=[]
    for (r,c) in shapes:
        torch.manual_seed(0); x = torch.rand(r,c)
        ref = prob.Model().forward(x).detach().numpy()
        out = call_unary(fn_name, x.numpy())
        if out is None: return None, None
        rows.append((f"{r}x{c}", *ulp(out, ref)))
    # robust gate: SAME Model.forward() on the robust vector (NOT a hand lambda)
    rv = robust_vec(); xt = torch.from_numpy(rv.reshape(1,-1))
    rref = prob.Model().forward(xt).detach().numpy()
    rout = call_unary(fn_name, rv)
    rob = ulp(rout, rref) if rout is not None else None
    return rows, rob

def classify(rows, rob):
    if rows is None: return "MISSING_KERNEL", ""
    detail = " ".join(f"{sh}:{mx}" for (sh,mx,_) in rows)
    maxes = [mx for (_,mx,_) in rows]
    if all(m==0 for m in maxes):
        if rob is not None and rob[0] > 0:
            return "ROBUST_GAP", f"stanford 0-ULP; robust maxULP={rob[0]}"
        return "BIT_IDENTICAL", detail + (" +robust" if rob is not None else "")
    if rows[0][1]==0 and any(mx>0 for (_,mx,_) in rows):
        return "SHAPE_DIVERGENT", detail
    return "DIVERGENT", detail

def main():
    print(f"STANFORD REFEREE v2 (SO={os.path.basename(SO)}, {'FULL' if FULL else 'fast'})")
    print(f"{'problem':<26}{'status':<16} detail"); print("-"*72)
    npass=ntot=0
    for kind, sfile, kernel in REG:
        prob = load_problem(sfile); name = sfile.replace(".py","")
        if prob is None: print(f"{name:<26}{'NO_FILE':<16}"); continue
        ntot += 1
        if kind == "unary":
            rows, rob = referee_unary(prob, kernel)
            status, detail = classify(rows, rob)
        elif kind == "matmul":
            torch.manual_seed(0); N=512; A=torch.rand(N,N); B=torch.rand(N,N)
            ref = prob.Model().forward(A,B).detach().numpy()
            out = np.zeros((N,N),np.float32); Ac=np.ascontiguousarray(A.numpy()); Bc=np.ascontiguousarray(B.numpy())
            getattr(lib,kernel)(Ac.ctypes.data_as(fpv),Bc.ctypes.data_as(fpv),out.ctypes.data_as(fpv),ctypes.c_int(N),ctypes.c_int(N),ctypes.c_int(N))
            mx,nz = ulp(out, ref); status = "BIT_IDENTICAL" if mx==0 else "DIVERGENT"; detail=f"512^3:{mx}"
        else:
            status, detail = "NOT_ROUTED", ""
        if status == "BIT_IDENTICAL": npass += 1
        print(f"{name:<26}{status:<16} {detail}")
    print("-"*72); print(f"Stanford-refereed BIT_IDENTICAL (robust): {npass}/{ntot} routed")

if __name__ == "__main__":
    main()
