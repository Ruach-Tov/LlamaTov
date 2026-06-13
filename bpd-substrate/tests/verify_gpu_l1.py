# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
import numpy as np, ctypes, torch, torch.nn as nn
torch.set_num_threads(1)

print("PyTorch %s | CUDA %s | archs: %s | GPU: %s" % (
    torch.__version__, torch.version.cuda, torch.cuda.get_arch_list(), torch.cuda.get_device_name(0)))
print()

lib = ctypes.CDLL("build/bpd_gpu.so")
V, I, F = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
lib.gpu_alloc.argtypes=[I]; lib.gpu_alloc.restype=V
lib.gpu_free.argtypes=[V]; lib.gpu_free.restype=None
lib.gpu_h2d.argtypes=[V,V,I]; lib.gpu_h2d.restype=None
lib.gpu_d2h.argtypes=[V,V,I]; lib.gpu_d2h.restype=None
lib.gpu_sync.argtypes=[]; lib.gpu_sync.restype=None

# Setup all kernel argtypes
for name in ["bpd_relu_gpu", "bpd_silu_gpu", "bpd_sigmoid_gpu", "bpd_tanh_gpu",
             "bpd_gelu_gpu", "bpd_mish_gpu", "bpd_neg_gpu", "bpd_abs_gpu",
             "bpd_leaky_relu_gpu", "bpd_elu_gpu", "bpd_hardsigmoid_gpu", "bpd_softplus_gpu"]:
    getattr(lib, name).argtypes = [V,V,I]; getattr(lib, name).restype = None
lib.bpd_residual_add_gpu.argtypes = [V,V,V,I]; lib.bpd_residual_add_gpu.restype = None
lib.bpd_mul_gpu.argtypes = [V,V,V,I]; lib.bpd_mul_gpu.restype = None
lib.bpd_softmax_gpu.argtypes = [V,V,I,I]; lib.bpd_softmax_gpu.restype = None
lib.bpd_layernorm_gpu.argtypes = [V,V,V,V,I,I,F]; lib.bpd_layernorm_gpu.restype = None
lib.bpd_maxpool2d_gpu.argtypes = [V,V]+[I]*8; lib.bpd_maxpool2d_gpu.restype = None
lib.bpd_batchnorm_affine_gpu.argtypes = [V,V,V,V,I,I,I]; lib.bpd_batchnorm_affine_gpu.restype = None

np.random.seed(42)
n = 10000
x = np.random.randn(n).astype(np.float32) * 5.0
x2 = np.random.randn(n).astype(np.float32) * 5.0

d_in = lib.gpu_alloc(I(n*4)); d_in2 = lib.gpu_alloc(I(n*4)); d_out = lib.gpu_alloc(I(n*4))
lib.gpu_h2d(d_in, x.ctypes.data, I(n*4))
lib.gpu_h2d(d_in2, x2.ctypes.data, I(n*4))
xt = torch.from_numpy(x).cuda(); x2t = torch.from_numpy(x2).cuda()

def ulp(a, b):
    d = np.abs(a.view(np.int32).astype(np.int64) - b.view(np.int32).astype(np.int64))
    return int(d.max())

print("Stanford L1: BPD GPU vs PyTorch GPU (sm_61)")
print("%-15s %8s %s" % ("Op", "max_ULP", "Status"))
print("-" * 35)
n_pass = 0; n_total = 0

# Unary activations
for name, gpu_fn, pt_fn in [
    ("relu",        "bpd_relu_gpu",        lambda: torch.relu(xt)),
    ("silu",        "bpd_silu_gpu",        lambda: torch.nn.functional.silu(xt)),
    ("sigmoid",     "bpd_sigmoid_gpu",     lambda: torch.sigmoid(xt)),
    ("tanh",        "bpd_tanh_gpu",        lambda: torch.tanh(xt)),
    ("gelu",        "bpd_gelu_gpu",        lambda: torch.nn.functional.gelu(xt)),
    ("mish",        "bpd_mish_gpu",        lambda: torch.nn.functional.mish(xt)),
    ("leaky_relu",  "bpd_leaky_relu_gpu",  lambda: torch.nn.functional.leaky_relu(xt)),
    ("elu",         "bpd_elu_gpu",         lambda: torch.nn.functional.elu(xt)),
    ("neg",         "bpd_neg_gpu",         lambda: torch.neg(xt)),
    ("abs",         "bpd_abs_gpu",         lambda: torch.abs(xt)),
    ("hardsigmoid", "bpd_hardsigmoid_gpu", lambda: torch.nn.functional.hardsigmoid(xt)),
    ("softplus",    "bpd_softplus_gpu",    lambda: torch.nn.functional.softplus(xt)),
]:
    fn = getattr(lib, gpu_fn)
    fn(d_in, d_out, I(n)); lib.gpu_sync()
    out = np.zeros(n, dtype=np.float32)
    lib.gpu_d2h(out.ctypes.data, d_out, I(n*4))
    pt = pt_fn().cpu().numpy()
    mu = ulp(out, pt)
    if mu == 0: n_pass += 1
    n_total += 1
    print("%-15s %8d %s" % (name, mu, "0 ULP" if mu == 0 else "%d ULP" % mu))

# Binary ops
for name, gpu_call, pt_call in [
    ("add", lambda: lib.bpd_residual_add_gpu(d_in, d_in2, d_out, I(n)), lambda: xt + x2t),
    ("mul", lambda: lib.bpd_mul_gpu(d_in, d_in2, d_out, I(n)), lambda: xt * x2t),
]:
    gpu_call(); lib.gpu_sync()
    out = np.zeros(n, dtype=np.float32)
    lib.gpu_d2h(out.ctypes.data, d_out, I(n*4))
    pt = pt_call().cpu().numpy()
    mu = ulp(out, pt)
    if mu == 0: n_pass += 1
    n_total += 1
    print("%-15s %8d %s" % (name, mu, "0 ULP" if mu == 0 else "%d ULP" % mu))

lib.gpu_free(d_in); lib.gpu_free(d_in2); lib.gpu_free(d_out)

# Softmax
rows, cols = 64, 128
xs = np.random.randn(rows, cols).astype(np.float32)
d_xs = lib.gpu_alloc(I(rows*cols*4)); d_os = lib.gpu_alloc(I(rows*cols*4))
lib.gpu_h2d(d_xs, xs.ctypes.data, I(rows*cols*4))
lib.bpd_softmax_gpu(d_xs, d_os, I(rows), I(cols)); lib.gpu_sync()
gpu_sm = np.zeros_like(xs)
lib.gpu_d2h(gpu_sm.ctypes.data, d_os, I(rows*cols*4))
pt_sm = torch.softmax(torch.from_numpy(xs).cuda(), dim=1).cpu().numpy()
mu = ulp(gpu_sm, pt_sm)
if mu == 0: n_pass += 1
n_total += 1
print("%-15s %8d %s" % ("softmax", mu, "0 ULP" if mu == 0 else "%d ULP" % mu))
lib.gpu_free(d_xs); lib.gpu_free(d_os)

# LayerNorm
xln = np.random.randn(32, 512).astype(np.float32)
g = np.ones(512, dtype=np.float32); b = np.zeros(512, dtype=np.float32)
d_xln=lib.gpu_alloc(I(32*512*4)); d_oln=lib.gpu_alloc(I(32*512*4))
d_g=lib.gpu_alloc(I(512*4)); d_b=lib.gpu_alloc(I(512*4))
lib.gpu_h2d(d_xln,xln.ctypes.data,I(32*512*4))
lib.gpu_h2d(d_g,g.ctypes.data,I(512*4)); lib.gpu_h2d(d_b,b.ctypes.data,I(512*4))
lib.bpd_layernorm_gpu(d_xln,d_g,d_b,d_oln,I(32),I(512),F(1e-5)); lib.gpu_sync()
gpu_ln = np.zeros_like(xln)
lib.gpu_d2h(gpu_ln.ctypes.data,d_oln,I(32*512*4))
ln = nn.LayerNorm(512).cuda().eval()
pt_ln = ln(torch.from_numpy(xln).cuda()).detach().cpu().numpy()
mu = ulp(gpu_ln, pt_ln)
if mu == 0: n_pass += 1
n_total += 1
print("%-15s %8d %s" % ("layernorm", mu, "0 ULP" if mu == 0 else "%d ULP" % mu))
lib.gpu_free(d_xln); lib.gpu_free(d_oln); lib.gpu_free(d_g); lib.gpu_free(d_b)

# MaxPool2d
xmp = np.random.randn(1,16,16,16).astype(np.float32)
Ho=(16-2)//2+1  # k=2,s=2,p=0 -> 8
d_mp=lib.gpu_alloc(I(xmp.nbytes)); d_omp=lib.gpu_alloc(I(1*16*Ho*Ho*4))
lib.gpu_h2d(d_mp,xmp.ctypes.data,I(xmp.nbytes))
lib.bpd_maxpool2d_gpu(d_mp,d_omp,I(1),I(16),I(16),I(16),I(2),I(2),I(2),I(0)); lib.gpu_sync()
gpu_mp = np.zeros((1,16,Ho,Ho),dtype=np.float32)
lib.gpu_d2h(gpu_mp.ctypes.data,d_omp,I(gpu_mp.nbytes))
mp = nn.MaxPool2d(2,2).cuda()
pt_mp = mp(torch.from_numpy(xmp).cuda()).cpu().numpy()
mu = ulp(gpu_mp, pt_mp)
if mu == 0: n_pass += 1
n_total += 1
print("%-15s %8d %s" % ("maxpool2d", mu, "0 ULP" if mu == 0 else "%d ULP" % mu))
lib.gpu_free(d_mp); lib.gpu_free(d_omp)

print()
print("%d/%d L1 ops BIT_IDENTICAL on GPU" % (n_pass, n_total))
