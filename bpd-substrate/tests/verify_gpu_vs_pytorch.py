# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
import numpy as np, ctypes, torch
torch.set_num_threads(1)

print("PyTorch %s | CUDA %s | archs: %s" % (torch.__version__, torch.version.cuda, torch.cuda.get_arch_list()))
print("GPU: %s" % torch.cuda.get_device_name(0))
print()

lib = ctypes.CDLL("build/bpd_gpu.so")
V, I = ctypes.c_void_p, ctypes.c_int
lib.gpu_alloc.argtypes=[I]; lib.gpu_alloc.restype=V
lib.gpu_free.argtypes=[V]; lib.gpu_free.restype=None
lib.gpu_h2d.argtypes=[V,V,I]; lib.gpu_h2d.restype=None
lib.gpu_d2h.argtypes=[V,V,I]; lib.gpu_d2h.restype=None
lib.gpu_sync.argtypes=[]; lib.gpu_sync.restype=None

for name in ["bpd_relu_gpu", "bpd_silu_gpu", "bpd_sigmoid_gpu", "bpd_tanh_gpu",
             "bpd_gelu_gpu", "bpd_mish_gpu", "bpd_neg_gpu", "bpd_abs_gpu"]:
    getattr(lib, name).argtypes = [V,V,I]; getattr(lib, name).restype = None
lib.bpd_residual_add_gpu.argtypes = [V,V,V,I]; lib.bpd_residual_add_gpu.restype = None

np.random.seed(42)
n = 100000
x = np.random.randn(n).astype(np.float32) * 5.0
x2 = np.random.randn(n).astype(np.float32) * 5.0

# BPD GPU
d_in = lib.gpu_alloc(I(n*4)); d_in2 = lib.gpu_alloc(I(n*4)); d_out = lib.gpu_alloc(I(n*4))
lib.gpu_h2d(d_in, x.ctypes.data, I(n*4))
lib.gpu_h2d(d_in2, x2.ctypes.data, I(n*4))

# PyTorch GPU (sm_61 build!)
xt = torch.from_numpy(x).cuda()
x2t = torch.from_numpy(x2).cuda()

def ulp(a, b):
    d = np.abs(a.view(np.int32).astype(np.int64) - b.view(np.int32).astype(np.int64))
    return int(d.max()), int((d > 0).sum())

print("BPD GPU vs PyTorch GPU (both on Tesla P4, sm_61)")
print("%-15s %8s %8s  %s" % ("Op", "max_ULP", "n_diffs", "Status"))
print("-" * 50)

tests = [
    ("relu",    "bpd_relu_gpu",    lambda: torch.relu(xt)),
    ("silu",    "bpd_silu_gpu",    lambda: torch.nn.functional.silu(xt)),
    ("sigmoid", "bpd_sigmoid_gpu", lambda: torch.sigmoid(xt)),
    ("tanh",    "bpd_tanh_gpu",    lambda: torch.tanh(xt)),
    ("gelu",    "bpd_gelu_gpu",    lambda: torch.nn.functional.gelu(xt)),
    ("mish",    "bpd_mish_gpu",    lambda: torch.nn.functional.mish(xt)),
    ("neg",     "bpd_neg_gpu",     lambda: torch.neg(xt)),
    ("abs",     "bpd_abs_gpu",     lambda: torch.abs(xt)),
]

n_pass = 0
for name, gpu_fn, pt_fn in tests:
    fn = getattr(lib, gpu_fn)
    fn(d_in, d_out, I(n)); lib.gpu_sync()
    bpd_out = np.zeros(n, dtype=np.float32)
    lib.gpu_d2h(bpd_out.ctypes.data, d_out, I(n*4))
    pt_out = pt_fn().cpu().numpy()
    mu, nd = ulp(bpd_out, pt_out)
    if mu == 0: n_pass += 1
    pct = 100.0 * nd / n if nd > 0 else 0
    print("%-15s %8d %8d  %s" % (name, mu, nd, "0 ULP" if mu == 0 else "%d ULP (%.1f%%)" % (mu, pct)))

# Add
lib.bpd_residual_add_gpu(d_in, d_in2, d_out, I(n)); lib.gpu_sync()
bpd_out = np.zeros(n, dtype=np.float32)
lib.gpu_d2h(bpd_out.ctypes.data, d_out, I(n*4))
pt_add = (xt + x2t).cpu().numpy()
mu, nd = ulp(bpd_out, pt_add)
if mu == 0: n_pass += 1
print("%-15s %8d %8d  %s" % ("add", mu, nd, "0 ULP" if mu == 0 else "%d ULP" % mu))

lib.gpu_free(d_in); lib.gpu_free(d_in2); lib.gpu_free(d_out)
print()
print("%d/%d ops BIT_IDENTICAL (BPD GPU vs PyTorch GPU)" % (n_pass, len(tests)+1))
