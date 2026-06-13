#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""probe_mkl_params.py — Empirically characterise what PyTorch MKL does for each
divergent kernel family so we can derive the correct platform_param facts for
the pytorch_cpu_mkl platform entry in implementation_matches.pl.

Run: BPD_CPU_SO=build/bpd_cpu.so python3 bench/probe_mkl_params.py
"""
import ctypes, os, subprocess, struct, math
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
torch.backends.mkldnn.enabled = False

SO = os.environ.get("BPD_CPU_SO", "build/bpd_cpu.so")
lib = ctypes.CDLL(SO)
RNG = np.random.default_rng(42)

# ─── helpers ──────────────────────────────────────────────────────────────────

def ulp(a, b):
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    BASE = np.int64(0x80000000)
    ai = np.where(ai < 0, BASE - ai, ai)
    bi = np.where(bi < 0, BASE - bi, bi)
    d = np.abs(ai - bi)
    return int(d.max()), int((d > 0).sum()), d.size

def ulp1(a, b):
    ai = np.float32(a).view(np.int32).astype(np.int64)
    bi = np.float32(b).view(np.int32).astype(np.int64)
    BASE = np.int64(0x80000000)
    ai = BASE - ai if ai < 0 else ai
    bi = BASE - bi if bi < 0 else bi
    return abs(int(ai) - int(bi))

def report(name, max_ulp, n_diff, n_total, note=""):
    status = "✅ 0 ULP" if max_ulp == 0 else f"❌ max={max_ulp} ULP  n={n_diff}/{n_total}"
    print(f"  {name:<55} {status}  {note}")

# ─── 1. Matrix-vector (kernel #4) ─────────────────────────────────────────────
print("\n══ 1. Matrix-vector (kernel #4) ══")
lib.bpd_mm_cpu.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*3
lib.bpd_mm_cpu.restype = None

for M_v, N_v, K_v in [(128, 1, 256), (64, 1, 128), (16, 1, 64)]:
    A = RNG.standard_normal((M_v, K_v)).astype(np.float32)
    B = RNG.standard_normal((K_v, N_v)).astype(np.float32)
    out = np.zeros((M_v, N_v), dtype=np.float32)
    lib.bpd_mm_cpu(A.ctypes.data, B.ctypes.data, out.ctypes.data, M_v, N_v, K_v)
    ref = torch.from_numpy(A).mm(torch.from_numpy(B)).numpy()
    report(f"bpd_mm_cpu GEMV M={M_v} K={K_v}", *ulp(ref, out))
    # scalar reference
    scalar = np.zeros((M_v, N_v), dtype=np.float32)
    for row in range(M_v):
        partial = np.float32(0.0)
        for k in range(K_v):
            partial = np.float32(partial + np.float32(A[row, k] * B[k, 0]))
        scalar[row, 0] = partial
    report(f"  scalar GEMV vs PT M={M_v} K={K_v}", *ulp(ref, scalar),
           note="0 ULP → PT uses same scalar order")

# ─── 2. Activations ───────────────────────────────────────────────────────────
print("\n══ 2. Activations ══")
x = torch.from_numpy(RNG.standard_normal(1024).astype(np.float32))
xn = x.numpy()

# Sigmoid: void bpd_sigmoid_cpu(const float* input, float* output, int n)
lib.bpd_sigmoid_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.bpd_sigmoid_cpu.restype = None
out_sig = np.zeros(1024, dtype=np.float32)
lib.bpd_sigmoid_cpu(xn.ctypes.data, out_sig.ctypes.data, 1024)
ref_sig = torch.sigmoid(x).numpy()
report("sigmoid", *ulp(ref_sig, out_sig))
sig_d = (1.0 / (1.0 + np.exp(-xn.astype(np.float64)))).astype(np.float32)
report("  sigmoid f64-exp vs PT", *ulp(ref_sig, sig_d), note="0 ULP → PT uses f64 exp")

# Tanh
lib.bpd_tanh_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.bpd_tanh_cpu.restype = None
out_tanh = np.zeros(1024, dtype=np.float32)
lib.bpd_tanh_cpu(xn.ctypes.data, out_tanh.ctypes.data, 1024)
ref_tanh = torch.tanh(x).numpy()
report("tanh", *ulp(ref_tanh, out_tanh))
tanh_d = np.tanh(xn.astype(np.float64)).astype(np.float32)
report("  tanh f64 vs PT", *ulp(ref_tanh, tanh_d), note="0 ULP → PT uses f64 tanh")

# GELU
lib.bpd_gelu_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.bpd_gelu_cpu.restype = None
out_gelu = np.zeros(1024, dtype=np.float32)
lib.bpd_gelu_cpu(xn.ctypes.data, out_gelu.ctypes.data, 1024)
ref_gelu = F.gelu(x).numpy()
report("gelu (exact erff)", *ulp(ref_gelu, out_gelu))
# PyTorch exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
gelu_d = np.array([0.5 * float(v) * (1.0 + math.erf(float(v) * 0.7071067811865476))
                   for v in xn], dtype=np.float32)
report("  gelu scalar-f64-erf vs PT", *ulp(ref_gelu, gelu_d), note="0 ULP → PT uses f64 erf scalar")
# Check: is PyTorch using SVML's vserf? Probe the ULP pattern
diffs_gelu = np.abs(ref_gelu.view(np.int32).astype(np.int64) -
                    out_gelu.view(np.int32).astype(np.int64))
print(f"    GELU ULP histogram: max={diffs_gelu.max()} "
      f"p99={np.percentile(diffs_gelu, 99):.0f} "
      f"p50={np.percentile(diffs_gelu, 50):.0f}")

# ELU
lib.bpd_elu_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.bpd_elu_cpu.restype = None
out_elu = np.zeros(1024, dtype=np.float32)
lib.bpd_elu_cpu(xn.ctypes.data, out_elu.ctypes.data, 1024)
ref_elu = F.elu(x).numpy()
report("elu (expm1f)", *ulp(ref_elu, out_elu))
elu_d = np.where(xn < 0, np.expm1(xn.astype(np.float64)).astype(np.float32), xn)
report("  elu f64-expm1 vs PT", *ulp(ref_elu, elu_d.astype(np.float32)),
       note="0 ULP → PT uses f64 expm1")

# SiLU
lib.bpd_silu_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.bpd_silu_cpu.restype = None
out_silu = np.zeros(1024, dtype=np.float32)
lib.bpd_silu_cpu(xn.ctypes.data, out_silu.ctypes.data, 1024)
ref_silu = F.silu(x).numpy()
report("silu", *ulp(ref_silu, out_silu))
silu_d = (xn.astype(np.float64) / (1.0 + np.exp(-xn.astype(np.float64)))).astype(np.float32)
report("  silu f64 vs PT", *ulp(ref_silu, silu_d), note="0 ULP → PT uses f64 sigmoid")

# Softmax: void bpd_softmax_cpu(const float* input, float* output, int rows, int cols)
lib.bpd_softmax_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.bpd_softmax_cpu.restype = None
x_sm = torch.from_numpy(RNG.standard_normal((8, 128)).astype(np.float32))
out_sm = np.zeros((8, 128), dtype=np.float32)
lib.bpd_softmax_cpu(x_sm.numpy().ctypes.data, out_sm.ctypes.data, 8, 128)
ref_sm = F.softmax(x_sm, dim=1).numpy()
report("softmax (8x128)", *ulp(ref_sm, out_sm))
xsm = x_sm.numpy().astype(np.float64)
xsm_max = xsm.max(axis=1, keepdims=True)
xsm_exp = np.exp(xsm - xsm_max)
sm_d = (xsm_exp / xsm_exp.sum(axis=1, keepdims=True)).astype(np.float32)
report("  softmax f64-exp vs PT", *ulp(ref_sm, sm_d), note="0 ULP → PT uses f64 exp")

# Softplus
if hasattr(lib, 'bpd_softplus_cpu'):
    lib.bpd_softplus_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.bpd_softplus_cpu.restype = None
    out_sp = np.zeros(1024, dtype=np.float32)
    lib.bpd_softplus_cpu(xn.ctypes.data, out_sp.ctypes.data, 1024)
    ref_sp = F.softplus(x).numpy()
    report("softplus", *ulp(ref_sp, out_sp))
    sp_d = np.log1p(np.exp(xn.astype(np.float64))).astype(np.float32)
    report("  softplus f64 vs PT", *ulp(ref_sp, sp_d), note="0 ULP → PT uses f64 log1p(exp)")

# LogSoftmax
if hasattr(lib, 'bpd_log_softmax_cpu'):
    lib.bpd_log_softmax_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.bpd_log_softmax_cpu.restype = None
    out_lsm = np.zeros((8, 128), dtype=np.float32)
    lib.bpd_log_softmax_cpu(x_sm.numpy().ctypes.data, out_lsm.ctypes.data, 8, 128)
    ref_lsm = F.log_softmax(x_sm, dim=1).numpy()
    report("log_softmax (8x128)", *ulp(ref_lsm, out_lsm))

# SELU
if hasattr(lib, 'bpd_selu_cpu'):
    lib.bpd_selu_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.bpd_selu_cpu.restype = None
    out_selu = np.zeros(1024, dtype=np.float32)
    lib.bpd_selu_cpu(xn.ctypes.data, out_selu.ctypes.data, 1024)
    ref_selu = F.selu(x).numpy()
    report("selu", *ulp(ref_selu, out_selu))
    ALPHA = 1.6732631921768188
    SCALE = 1.0507009029388428
    selu_d = np.where(xn >= 0,
                      (SCALE * xn).astype(np.float32),
                      (SCALE * ALPHA * (np.expm1(xn.astype(np.float64)))).astype(np.float32))
    report("  selu f64-expm1 vs PT", *ulp(ref_selu, selu_d.astype(np.float32)),
           note="0 ULP → PT uses f64 expm1")

# ─── 3. Norms ─────────────────────────────────────────────────────────────────
print("\n══ 3. Norms ══")

# InstanceNorm: void bpd_instancenorm_cpu(const float* input, float* output,
#   int N, int C, int H, int W, float eps)
lib.bpd_instancenorm_cpu.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]
lib.bpd_instancenorm_cpu.restype = None
x_in = torch.from_numpy(RNG.standard_normal((2, 4, 8, 8)).astype(np.float32))
out_in = np.zeros_like(x_in.numpy())
lib.bpd_instancenorm_cpu(x_in.numpy().ctypes.data, out_in.ctypes.data, 2, 4, 8, 8, np.float32(1e-5))
ref_in = F.instance_norm(x_in, weight=torch.ones(4), bias=torch.zeros(4), eps=1e-5).numpy()
report("instance_norm (2,4,8,8)", *ulp(ref_in, out_in))
# Probe: cascade(8) reduction for mean/var vs sequential
xin = x_in.numpy()
def cascade8(arr):
    arr = arr.astype(np.float32)
    n = len(arr); n8 = (n // 8) * 8
    acc = np.zeros(8, dtype=np.float32)
    for i in range(0, n8, 8):
        acc = (acc + arr[i:i+8]).astype(np.float32)
    total = np.float32(0.0)
    for a in acc: total = np.float32(total + a)
    for i in range(n8, n): total = np.float32(total + arr[i])
    return total

in_cascade = np.zeros_like(out_in)
for n in range(2):
    for c in range(4):
        patch = xin[n, c].ravel()
        spatial = len(patch)
        mean_c = cascade8(patch) / np.float32(spatial)
        diff2 = (patch - mean_c).astype(np.float32) ** 2
        var_c = cascade8(diff2) / np.float32(spatial)
        invstd = np.float32(1.0) / np.sqrt(var_c + np.float32(1e-5))
        bias_c = -mean_c * invstd
        in_cascade[n, c] = (patch * invstd + bias_c).reshape(8, 8)
report("  instance_norm cascade8 vs PT", *ulp(ref_in, in_cascade),
       note="0 ULP → PT uses cascade(8) reduction")

# RMSNorm: void bpd_rmsnorm_cpu(const float* input, float* output,
#   int N, int C, int H, int W, float eps)
# Harness shape: (N=2, C=8, H=4, W=4), eps=1e-5
# Reference: rms = sqrt(mean(x**2, dim=1, keepdim=True) + eps); out = x / rms
lib.bpd_rmsnorm_cpu.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]
lib.bpd_rmsnorm_cpu.restype = None
N_r, C_r, H_r, W_r = 2, 8, 4, 4
x_rms = torch.from_numpy(RNG.standard_normal((N_r, C_r, H_r, W_r)).astype(np.float32) * 2.0)
out_rms = np.zeros_like(x_rms.numpy())
lib.bpd_rmsnorm_cpu(x_rms.numpy().ctypes.data, out_rms.ctypes.data, N_r, C_r, H_r, W_r, np.float32(1e-5))
xt_rms = x_rms
rms_ref_t = torch.sqrt(torch.mean(xt_rms ** 2, dim=1, keepdim=True) + 1e-5)
ref_rms = (xt_rms / rms_ref_t).numpy()
report("rms_norm (2,8,4,4)", *ulp(ref_rms, out_rms))
# Probe: cascade8 vs pairwise for the C-dim reduction (per spatial position)
xrms = x_rms.numpy()
rms_cascade = np.zeros_like(out_rms)
for n in range(N_r):
    for h in range(H_r):
        for w in range(W_r):
            col = xrms[n, :, h, w]  # shape (C=8,)
            sq = (col ** 2).astype(np.float32)
            ms = cascade8(sq) / np.float32(C_r)
            rms_val = np.sqrt(ms + np.float32(1e-5))
            rms_cascade[n, :, h, w] = (col / rms_val).astype(np.float32)
report("  rms_norm cascade8 vs PT", *ulp(ref_rms, rms_cascade),
       note="0 ULP → PT uses cascade(8) for C-dim sum")

# L2Norm: void bpd_l2norm_cpu(const float* input, float* output, int rows, int cols)
lib.bpd_l2norm_cpu.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.bpd_l2norm_cpu.restype = None
x_l2 = torch.from_numpy(RNG.standard_normal((16, 64)).astype(np.float32))
out_l2 = np.zeros_like(x_l2.numpy())
lib.bpd_l2norm_cpu(x_l2.numpy().ctypes.data, out_l2.ctypes.data, 16, 64)
ref_l2 = F.normalize(x_l2, p=2, dim=1, eps=1e-12).numpy()
report("l2norm (16x64)", *ulp(ref_l2, out_l2))
xl2 = x_l2.numpy()
l2_cascade = np.zeros_like(out_l2)
for r in range(16):
    sq = (xl2[r] ** 2).astype(np.float32)
    ss = cascade8(sq)
    norm = np.sqrt(ss)
    l2_cascade[r] = (xl2[r] / norm).astype(np.float32)
report("  l2norm cascade8 vs PT", *ulp(ref_l2, l2_cascade),
       note="0 ULP → PT uses cascade(8) for sum-of-squares")

# ─── 4. Depthwise convolutions ────────────────────────────────────────────────
print("\n══ 4. Depthwise convolutions ══")

# Probe PyTorch depthwise conv accumulation order
N_dw, C, H, W, kH, kW, stride, pad = 1, 4, 8, 8, 3, 3, 1, 1
x_dw = torch.from_numpy(RNG.standard_normal((N_dw, C, H, W)).astype(np.float32))
w_dw = torch.from_numpy(RNG.standard_normal((C, 1, kH, kW)).astype(np.float32))
ref_dw = F.conv2d(x_dw, w_dw, groups=C, stride=stride, padding=pad).numpy()
H_out = (H + 2*pad - kH) // stride + 1
W_out = (W + 2*pad - kW) // stride + 1

# Scalar reference: kH-outer, kW-inner
scalar_dw = np.zeros((N_dw, C, H_out, W_out), dtype=np.float32)
xdw = x_dw.numpy(); wdw = w_dw.numpy()
for n in range(N_dw):
    for c in range(C):
        for oh in range(H_out):
            for ow in range(W_out):
                acc = np.float32(0.0)
                for kh in range(kH):
                    for kw in range(kW):
                        ih = oh * stride - pad + kh
                        iw = ow * stride - pad + kw
                        if 0 <= ih < H and 0 <= iw < W:
                            acc = np.float32(acc + np.float32(xdw[n,c,ih,iw] * wdw[c,0,kh,kw]))
                scalar_dw[n,c,oh,ow] = acc
report("scalar depthwise (kH-outer) vs PT", *ulp(ref_dw, scalar_dw),
       note="0 ULP → PT uses same scalar order")

# Check if bpd has a depthwise kernel
r = subprocess.run(['nm', SO], capture_output=True, text=True)
dw_lines = [l.strip() for l in r.stdout.split('\n') if 'depthwise' in l.lower()]
print(f"  Depthwise symbols in .so: {dw_lines or ['(none)']}")

# ─── 5. Losses ────────────────────────────────────────────────────────────────
print("\n══ 5. Losses ══")

# CrossEntropyLoss
lib.bpd_cross_entropy_loss_cpu.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.bpd_cross_entropy_loss_cpu.restype = None
logits = RNG.standard_normal((32, 10)).astype(np.float32)
targets = RNG.integers(0, 10, 32).astype(np.int64)
out_ce = np.zeros(1, dtype=np.float32)
lib.bpd_cross_entropy_loss_cpu(logits.ctypes.data, targets.ctypes.data,
                                out_ce.ctypes.data, 32, 10)
ref_ce = F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(targets)).numpy()
report("cross_entropy (32x10)", *ulp(ref_ce, out_ce))
# Probe: pairwise vs cascade8 for the final mean reduction
ce_losses = []
for b in range(32):
    row = logits[b]
    mx = row.max()
    exp_row = np.exp((row - mx).astype(np.float64))
    log_sum = math.log(exp_row.sum())
    t = int(targets[b])
    ce_losses.append(-(float(row[t]) - float(mx) - log_sum))
ce_mean_seq = np.float32(sum(ce_losses) / 32)
ce_mean_c8 = cascade8(np.array(ce_losses, dtype=np.float32)) / np.float32(32)
report("  CE pairwise-f64 vs PT", *ulp(ref_ce, np.array([ce_mean_seq])),
       note="0 ULP → PT uses sequential mean")
report("  CE cascade8-f64 vs PT", *ulp(ref_ce, np.array([ce_mean_c8])),
       note="0 ULP → PT uses cascade8 mean")

# SDPA
lib.bpd_scaled_dot_product_attention_cpu.argtypes = [
    ctypes.c_void_p]*4 + [ctypes.c_int]*4
lib.bpd_scaled_dot_product_attention_cpu.restype = None
B_s, H_s, S, D = 1, 2, 8, 16
q = RNG.standard_normal((B_s, H_s, S, D)).astype(np.float32)
k = RNG.standard_normal((B_s, H_s, S, D)).astype(np.float32)
v = RNG.standard_normal((B_s, H_s, S, D)).astype(np.float32)
out_sdpa = np.zeros_like(q)
lib.bpd_scaled_dot_product_attention_cpu(
    q.ctypes.data, k.ctypes.data, v.ctypes.data, out_sdpa.ctypes.data,
    B_s, H_s, S, D)
ref_sdpa = F.scaled_dot_product_attention(
    torch.from_numpy(q), torch.from_numpy(k), torch.from_numpy(v)).numpy()
report("scaled_dot_product_attention", *ulp(ref_sdpa, out_sdpa))

# ─── 6. GEMM tile strategy probe ──────────────────────────────────────────────
print("\n══ 6. MKL GEMM tile strategy probe ══")
lib.bpd_mm_cpu.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*3
lib.bpd_mm_cpu.restype = None

print("  K-block boundary scan (M=N=32):")
for K_test in [64, 128, 256, 384, 512, 768, 1024]:
    A_t = RNG.standard_normal((32, K_test)).astype(np.float32)
    B_t = RNG.standard_normal((K_test, 32)).astype(np.float32)
    out_t = np.zeros((32, 32), dtype=np.float32)
    lib.bpd_mm_cpu(A_t.ctypes.data, B_t.ctypes.data, out_t.ctypes.data, 32, 32, K_test)
    ref_t = torch.from_numpy(A_t).mm(torch.from_numpy(B_t)).numpy()
    max_ulp, n_diff, _ = ulp(ref_t, out_t)
    status = "✅" if max_ulp == 0 else f"❌ max={max_ulp}"
    print(f"    K={K_test:<6} {status}  n_diff={n_diff}/1024")

# ─── 7. Reduction strategy probe ──────────────────────────────────────────────
print("\n══ 7. Reduction strategy probe ══")
x_red = torch.from_numpy(RNG.standard_normal(1024).astype(np.float32))
xr = x_red.numpy()
pt_sum = float(x_red.sum())

seq = np.float32(0.0)
for v in xr:
    seq = np.float32(seq + np.float32(v))

def cascade(arr, vw):
    arr = arr.astype(np.float32)
    n = len(arr); n_vec = (n // vw) * vw
    acc = np.zeros(vw, dtype=np.float32)
    for i in range(0, n_vec, vw):
        acc = (acc + arr[i:i+vw]).astype(np.float32)
    total = np.float32(0.0)
    for a in acc: total = np.float32(total + a)
    for i in range(n_vec, n): total = np.float32(total + arr[i])
    return total

print(f"  PT sum:          {pt_sum:.10e}  ULP vs seq={ulp1(pt_sum, seq)}")
for vw in [4, 8, 16, 32]:
    cs = cascade(xr, vw)
    print(f"  Cascade(vw={vw:<2}): {float(cs):.10e}  ULP vs PT={ulp1(pt_sum, cs)}")

print("\nDone.")
