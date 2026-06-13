# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
import numpy as np, ctypes, torch, torch.nn as nn, time

lib = ctypes.CDLL("build/bpd_gpu.so")
V,I,F = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
lib.gpu_alloc.argtypes=[I]; lib.gpu_alloc.restype=V
lib.gpu_free.argtypes=[V]; lib.gpu_free.restype=None
lib.gpu_h2d.argtypes=[V,V,I]; lib.gpu_h2d.restype=None
lib.gpu_d2h.argtypes=[V,V,I]; lib.gpu_d2h.restype=None
lib.gpu_sync.argtypes=[]; lib.gpu_sync.restype=None

for name in ["bpd_relu_gpu","bpd_silu_gpu","bpd_sigmoid_gpu","bpd_tanh_gpu",
             "bpd_gelu_gpu","bpd_mish_gpu","bpd_leaky_relu_gpu","bpd_elu_gpu",
             "bpd_neg_gpu","bpd_abs_gpu","bpd_hardsigmoid_gpu","bpd_softplus_gpu"]:
    getattr(lib, name).argtypes=[V,V,I]; getattr(lib, name).restype=None
lib.bpd_residual_add_gpu.argtypes=[V,V,V,I]; lib.bpd_residual_add_gpu.restype=None
lib.bpd_mul_gpu.argtypes=[V,V,V,I]; lib.bpd_mul_gpu.restype=None
lib.bpd_softmax_gpu.argtypes=[V,V,I,I]; lib.bpd_softmax_gpu.restype=None
lib.bpd_layernorm_gpu.argtypes=[V,V,V,V,I,I,F]; lib.bpd_layernorm_gpu.restype=None
lib.bpd_maxpool2d_gpu.argtypes=[V,V]+[I]*8; lib.bpd_maxpool2d_gpu.restype=None

np.random.seed(42)
n = 1000000  # 1M elements for timing
REPS = 100

x = np.random.randn(n).astype(np.float32)
x2 = np.random.randn(n).astype(np.float32)
d_in = lib.gpu_alloc(I(n*4)); d_in2 = lib.gpu_alloc(I(n*4)); d_out = lib.gpu_alloc(I(n*4))
lib.gpu_h2d(d_in, x.ctypes.data, I(n*4)); lib.gpu_h2d(d_in2, x2.ctypes.data, I(n*4))
xt = torch.from_numpy(x).cuda(); x2t = torch.from_numpy(x2).cuda()

print("GPU PERFORMANCE: BPD vs PyTorch (Tesla P4, n=%dK)" % (n//1000))
print("%-15s %8s %8s %8s" % ("Op", "BPD us", "PT us", "ratio"))
print("-" * 45)

unary_tests = [
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
]

bpd_wins = 0; pt_wins = 0; ties = 0
for name, gpu_fn, pt_fn in unary_tests:
    fn = getattr(lib, gpu_fn)
    # BPD warmup + time
    fn(d_in, d_out, I(n)); lib.gpu_sync()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        fn(d_in, d_out, I(n))
    lib.gpu_sync()
    bpd_us = (time.perf_counter() - t0) / REPS * 1e6
    
    # PyTorch warmup + time
    pt_fn(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        pt_fn()
    torch.cuda.synchronize()
    pt_us = (time.perf_counter() - t0) / REPS * 1e6
    
    ratio = pt_us / bpd_us
    if ratio > 1.05: bpd_wins += 1; tag = "BPD"
    elif ratio < 0.95: pt_wins += 1; tag = "PT"
    else: ties += 1; tag = "tie"
    print("%-15s %7.0f %7.0f %7.2fx  %s" % (name, bpd_us, pt_us, ratio, tag))

# Binary ops
for name, bpd_call, pt_call in [
    ("add", lambda: (lib.bpd_residual_add_gpu(d_in,d_in2,d_out,I(n)), lib.gpu_sync()), lambda: xt+x2t),
    ("mul", lambda: (lib.bpd_mul_gpu(d_in,d_in2,d_out,I(n)), lib.gpu_sync()), lambda: xt*x2t),
]:
    bpd_call(); torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(REPS): bpd_call()
    lib.gpu_sync()
    bpd_us=(time.perf_counter()-t0)/REPS*1e6
    pt_call(); torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(REPS): pt_call()
    torch.cuda.synchronize()
    pt_us=(time.perf_counter()-t0)/REPS*1e6
    ratio=pt_us/bpd_us
    if ratio > 1.05: bpd_wins += 1; tag = "BPD"
    elif ratio < 0.95: pt_wins += 1; tag = "PT"
    else: ties += 1; tag = "tie"
    print("%-15s %7.0f %7.0f %7.2fx  %s" % (name, bpd_us, pt_us, ratio, tag))

lib.gpu_free(d_in); lib.gpu_free(d_in2); lib.gpu_free(d_out)

# MaxPool2d
xmp = np.random.randn(1,64,112,112).astype(np.float32)
d_mp=lib.gpu_alloc(I(xmp.nbytes)); Ho=(112-2)//2+1; d_omp=lib.gpu_alloc(I(1*64*Ho*Ho*4))
lib.gpu_h2d(d_mp,xmp.ctypes.data,I(xmp.nbytes))
xmpt=torch.from_numpy(xmp).cuda(); mp=nn.MaxPool2d(2,2).cuda()
lib.bpd_maxpool2d_gpu(d_mp,d_omp,I(1),I(64),I(112),I(112),I(2),I(2),I(2),I(0)); lib.gpu_sync()
torch.cuda.synchronize()
t0=time.perf_counter()
for _ in range(REPS): lib.bpd_maxpool2d_gpu(d_mp,d_omp,I(1),I(64),I(112),I(112),I(2),I(2),I(2),I(0))
lib.gpu_sync()
bpd_us=(time.perf_counter()-t0)/REPS*1e6
mp(xmpt); torch.cuda.synchronize()
t0=time.perf_counter()
for _ in range(REPS): mp(xmpt)
torch.cuda.synchronize()
pt_us=(time.perf_counter()-t0)/REPS*1e6
ratio=pt_us/bpd_us
if ratio > 1.05: bpd_wins += 1; tag = "BPD"
elif ratio < 0.95: pt_wins += 1; tag = "PT"
else: ties += 1; tag = "tie"
print("%-15s %7.0f %7.0f %7.2fx  %s" % ("maxpool2d", bpd_us, pt_us, ratio, tag))
lib.gpu_free(d_mp); lib.gpu_free(d_omp)

# Softmax
rows,cols=256,1024
xs=np.random.randn(rows,cols).astype(np.float32)
d_xs=lib.gpu_alloc(I(rows*cols*4)); d_os=lib.gpu_alloc(I(rows*cols*4))
lib.gpu_h2d(d_xs,xs.ctypes.data,I(rows*cols*4))
xst=torch.from_numpy(xs).cuda()
lib.bpd_softmax_gpu(d_xs,d_os,I(rows),I(cols)); lib.gpu_sync()
torch.cuda.synchronize()
t0=time.perf_counter()
for _ in range(REPS): lib.bpd_softmax_gpu(d_xs,d_os,I(rows),I(cols))
lib.gpu_sync()
bpd_us=(time.perf_counter()-t0)/REPS*1e6
torch.softmax(xst,dim=1); torch.cuda.synchronize()
t0=time.perf_counter()
for _ in range(REPS): torch.softmax(xst,dim=1)
torch.cuda.synchronize()
pt_us=(time.perf_counter()-t0)/REPS*1e6
ratio=pt_us/bpd_us
if ratio > 1.05: bpd_wins += 1; tag = "BPD"
elif ratio < 0.95: pt_wins += 1; tag = "PT"
else: ties += 1; tag = "tie"
print("%-15s %7.0f %7.0f %7.2fx  %s" % ("softmax", bpd_us, pt_us, ratio, tag))
lib.gpu_free(d_xs); lib.gpu_free(d_os)

# LayerNorm
rows,cols=256,1024
xln=np.random.randn(rows,cols).astype(np.float32)
g=np.ones(cols,dtype=np.float32); b=np.zeros(cols,dtype=np.float32)
d_xln=lib.gpu_alloc(I(rows*cols*4)); d_oln=lib.gpu_alloc(I(rows*cols*4))
d_g=lib.gpu_alloc(I(cols*4)); d_b=lib.gpu_alloc(I(cols*4))
lib.gpu_h2d(d_xln,xln.ctypes.data,I(rows*cols*4))
lib.gpu_h2d(d_g,g.ctypes.data,I(cols*4)); lib.gpu_h2d(d_b,b.ctypes.data,I(cols*4))
xlnt=torch.from_numpy(xln).cuda(); lnm=nn.LayerNorm(cols).cuda().eval()
lib.bpd_layernorm_gpu(d_xln,d_g,d_b,d_oln,I(rows),I(cols),F(1e-5)); lib.gpu_sync()
torch.cuda.synchronize()
t0=time.perf_counter()
for _ in range(REPS): lib.bpd_layernorm_gpu(d_xln,d_g,d_b,d_oln,I(rows),I(cols),F(1e-5))
lib.gpu_sync()
bpd_us=(time.perf_counter()-t0)/REPS*1e6
lnm(xlnt); torch.cuda.synchronize()
t0=time.perf_counter()
for _ in range(REPS): lnm(xlnt)
torch.cuda.synchronize()
pt_us=(time.perf_counter()-t0)/REPS*1e6
ratio=pt_us/bpd_us
if ratio > 1.05: bpd_wins += 1; tag = "BPD"
elif ratio < 0.95: pt_wins += 1; tag = "PT"
else: ties += 1; tag = "tie"
print("%-15s %7.0f %7.0f %7.2fx  %s" % ("layernorm", bpd_us, pt_us, ratio, tag))
lib.gpu_free(d_xln); lib.gpu_free(d_oln); lib.gpu_free(d_g); lib.gpu_free(d_b)

print()
print("SCOREBOARD: BPD wins %d, PT wins %d, ties %d (of %d)" % (bpd_wins, pt_wins, ties, bpd_wins+pt_wins+ties))
