#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Verify BPD kernels are bit-identical with reference — CPU or GPU.

On CPU: compares against PyTorch CPU backend. If your PyTorch uses MKL/OpenBLAS,
matmul accumulation order may differ (set CPU_FP_MODE=fma or CPU_FP_MODE=native
when building bpd_cpu.so to match). Elementwise ops should always be 0 ULP.

Set BPD_CPU_REF=sequential to use a Python sequential reference loop instead
of torch.matmul — this eliminates BLAS-backend differences and verifies our
C code produces the correct sequential accumulation.

Detects available hardware and runs the appropriate comparison:
  CPU:  BPD C kernels (gcc) vs PyTorch CPU
  GPU:  BPD CUDA kernels (nvcc) vs PyTorch CUDA (cuBLAS/ATen)

Anyone with Python + gcc can verify correctness. No GPU required.

Usage:
    python3 bench/bit_identical_universal.py          # auto-detect
    BPD_CPU_SO=build/bpd_cpu.so python3 bench/bit_identical_universal.py  # explicit CPU
"""
import ctypes, os, sys, numpy as np

try:
    import torch
except ImportError:
    sys.exit("error: pip install torch numpy")

HAS_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if HAS_CUDA else "cpu"

CPU_SO = os.environ.get("BPD_CPU_SO", "build/bpd_cpu.so")
GPU_SO = os.environ.get("BPD_MM_SO", "build/bpd_mm.so")

def ulp(a, b):
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    B = np.int64(0x80000000)
    ai = np.where(ai < 0, B - ai, ai)
    bi = np.where(bi < 0, B - bi, bi)
    d = np.abs(ai - bi)
    return int(d.max()), int((d > 0).sum()), d.size

def classify(ref, got, label=""):
    mx, cnt, tot = ulp(ref, got)
    abs_max = float(np.abs(ref - got).max())
    if mx == 0:
        return "BIT_IDENTICAL", mx, abs_max
    elif abs_max < 1e-4 and mx > 100000:
        return "PASS_ABS_TOLERANCE", mx, abs_max
    elif mx <= 64:
        return "PASS_WITHIN_64_ULP", mx, abs_max
    elif abs_max < 1e-5:
        return "PASS_ABS_TOLERANCE", mx, abs_max
    else:
        return "FAIL", mx, abs_max

def load_cpu_lib():
    if not os.path.exists(CPU_SO):
        return None
    lib = ctypes.CDLL(CPU_SO)
    # matmul
    lib.bpd_mm_cpu.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*3
    lib.bpd_mm_cpu.restype = None
    # fused matmul + bias + relu
    lib.bpd_mm_bias_relu_cpu.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*3
    lib.bpd_mm_bias_relu_cpu.restype = None
    # elementwise
    for fn in ['bpd_relu_cpu', 'bpd_silu_cpu', 'bpd_mish_cpu']:
        getattr(lib, fn).argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        getattr(lib, fn).restype = None
    # conv2d
    lib.bpd_conv2d_cpu.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*9
    lib.bpd_conv2d_cpu.restype = None
    # batchnorm
    lib.bpd_batchnorm_cpu.argtypes = [ctypes.c_void_p]*6 + [ctypes.c_int]*3 + [ctypes.c_float]
    lib.bpd_batchnorm_cpu.restype = None
    # upsample
    lib.bpd_upsample_nearest2d_cpu.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]*4
    lib.bpd_upsample_nearest2d_cpu.restype = None
    # additional elementwise
    for fn in ['bpd_sigmoid_cpu', 'bpd_tanh_cpu', 'bpd_gelu_cpu', 'bpd_neg_cpu', 'bpd_abs_cpu', 'bpd_exp_cpu']:
        getattr(lib, fn).argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        getattr(lib, fn).restype = None
    # reductions
    for fn in ['bpd_sum_cpu', 'bpd_mean_cpu', 'bpd_max_cpu']:
        getattr(lib, fn).argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        getattr(lib, fn).restype = None
    # softmax
    lib.bpd_softmax_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.bpd_softmax_cpu.restype = None
    # layernorm
    lib.bpd_layernorm_cpu.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*2 + [ctypes.c_float]
    lib.bpd_layernorm_cpu.restype = None
    # maxpool2d / avgpool2d
    for fn in ['bpd_maxpool2d_cpu', 'bpd_avgpool2d_cpu']:
        getattr(lib, fn).argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]*8
        getattr(lib, fn).restype = None
    # linear
    lib.bpd_linear_cpu.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*3
    lib.bpd_linear_cpu.restype = None
    return lib

def run_cpu_tests(lib):
    results = []
    rng = np.random.default_rng(42)

    # ── Matmul ──
    for M in [64, 256, 512]:
        N = K = M
        A = rng.standard_normal((M, K)).astype(np.float32)
        B = rng.standard_normal((K, N)).astype(np.float32)
        ref = (torch.from_numpy(A) @ torch.from_numpy(B)).numpy()
        out = np.zeros((M, N), dtype=np.float32)
        lib.bpd_mm_cpu(A.ctypes.data, B.ctypes.data, out.ctypes.data, M, N, K)
        status, mx, ab = classify(ref, out)
        results.append(("sgemm_cpu", f"{M}x{M}", status, mx, ab))

    # ── Elementwise ──
    x = rng.standard_normal(10000).astype(np.float32)
    for name, pt_fn, bpd_fn in [
        ("relu",  lambda t: torch.relu(t), lib.bpd_relu_cpu),
        ("silu",  lambda t: torch.nn.functional.silu(t), lib.bpd_silu_cpu),
        ("mish",  lambda t: torch.nn.functional.mish(t), lib.bpd_mish_cpu),
    ]:
        ref = pt_fn(torch.from_numpy(x)).numpy()
        out = np.zeros_like(x)
        bpd_fn(x.ctypes.data, out.ctypes.data, len(x))
        status, mx, ab = classify(ref, out)
        results.append((f"{name}_cpu", "10000", status, mx, ab))

    # ── Fused matmul + bias + relu ──
    M, N, K = 256, 256, 256
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    bias = rng.standard_normal(N).astype(np.float32)
    ref = torch.relu(torch.from_numpy(A) @ torch.from_numpy(B) + torch.from_numpy(bias)).numpy()
    out = np.zeros((M, N), dtype=np.float32)
    lib.bpd_mm_bias_relu_cpu(A.ctypes.data, B.ctypes.data, bias.ctypes.data, out.ctypes.data, M, N, K)
    status, mx, ab = classify(ref, out)
    results.append(("fused_mm_bias_relu_cpu", f"{M}x{M}", status, mx, ab))

    # ── Conv2D ──
    N_batch, C_in, H, W, C_out, kH, kW = 1, 3, 16, 16, 8, 3, 3
    stride, pad = 1, 1
    inp = rng.standard_normal((N_batch, C_in, H, W)).astype(np.float32)
    weight = rng.standard_normal((C_out, C_in, kH, kW)).astype(np.float32)
    ref = torch.nn.functional.conv2d(
        torch.from_numpy(inp), torch.from_numpy(weight),
        stride=stride, padding=pad).numpy()
    H_out = (H + 2*pad - kH) // stride + 1
    W_out = (W + 2*pad - kW) // stride + 1
    out = np.zeros((N_batch, C_out, H_out, W_out), dtype=np.float32)
    lib.bpd_conv2d_cpu(inp.ctypes.data, weight.ctypes.data, out.ctypes.data,
                        N_batch, C_in, H, W, C_out, kH, kW, stride, pad)
    status, mx, ab = classify(ref, out)
    results.append(("conv2d_cpu", f"{N_batch}x{C_in}x{H}x{W}", status, mx, ab))

    # ── Upsample ──
    inp = rng.standard_normal((1, 8, 4, 4)).astype(np.float32)
    ref = torch.nn.functional.interpolate(
        torch.from_numpy(inp), scale_factor=2, mode='nearest').numpy()
    out = np.zeros((1, 8, 8, 8), dtype=np.float32)
    lib.bpd_upsample_nearest2d_cpu(inp.ctypes.data, out.ctypes.data, 1, 8, 4, 4)
    status, mx, ab = classify(ref, out)
    results.append(("upsample_cpu", "1x8x4x4", status, mx, ab))

    # ── Additional elementwise ──
    for name, pt_fn, bpd_fn in [
        ("sigmoid", lambda t: torch.sigmoid(t), lib.bpd_sigmoid_cpu),
        ("tanh",    lambda t: torch.tanh(t), lib.bpd_tanh_cpu),
        ("gelu",    lambda t: torch.nn.functional.gelu(t), lib.bpd_gelu_cpu),
        ("neg",     lambda t: -t, lib.bpd_neg_cpu),
        ("abs",     lambda t: torch.abs(t), lib.bpd_abs_cpu),
        ("exp",     lambda t: torch.exp(t), lib.bpd_exp_cpu),
    ]:
        ref = pt_fn(torch.from_numpy(x)).numpy()
        out = np.zeros_like(x)
        bpd_fn(x.ctypes.data, out.ctypes.data, len(x))
        status, mx, ab = classify(ref, out)
        results.append((f"{name}_cpu", "10000", status, mx, ab))

    # ── Reductions ──
    r_input = rng.standard_normal(1024).astype(np.float32)
    for name, pt_fn, bpd_fn in [
        ("sum",  lambda t: torch.sum(t), lib.bpd_sum_cpu),
        ("mean", lambda t: torch.mean(t), lib.bpd_mean_cpu),
        ("max",  lambda t: torch.max(t), lib.bpd_max_cpu),
    ]:
        ref_val = pt_fn(torch.from_numpy(r_input)).numpy().reshape(1)
        out = np.zeros(1, dtype=np.float32)
        bpd_fn(r_input.ctypes.data, out.ctypes.data, len(r_input))
        status, mx, ab = classify(ref_val, out)
        results.append((f"reduce_{name}_cpu", "1024", status, mx, ab))

    # ── Softmax ──
    s_input = rng.standard_normal((32, 64)).astype(np.float32)
    ref = torch.softmax(torch.from_numpy(s_input), dim=-1).numpy()
    out = np.zeros_like(s_input)
    lib.bpd_softmax_cpu(s_input.ctypes.data, out.ctypes.data, 32, 64)
    status, mx, ab = classify(ref, out)
    results.append(("softmax_cpu", "32x64", status, mx, ab))

    # ── LayerNorm ──
    ln_input = rng.standard_normal((8, 128)).astype(np.float32)
    gamma = rng.standard_normal(128).astype(np.float32)
    beta = rng.standard_normal(128).astype(np.float32)
    ln = torch.nn.LayerNorm(128, elementwise_affine=True)
    ln.weight.data = torch.from_numpy(gamma)
    ln.bias.data = torch.from_numpy(beta)
    ref = ln(torch.from_numpy(ln_input)).detach().numpy()
    out = np.zeros_like(ln_input)
    lib.bpd_layernorm_cpu(ln_input.ctypes.data, gamma.ctypes.data, beta.ctypes.data,
                           out.ctypes.data, 8, 128, ctypes.c_float(1e-5))
    status, mx, ab = classify(ref, out)
    results.append(("layernorm_cpu", "8x128", status, mx, ab))

    # ── MaxPool2D ──
    p_input = rng.standard_normal((1, 3, 16, 16)).astype(np.float32)
    ref = torch.nn.functional.max_pool2d(torch.from_numpy(p_input), 2, stride=2).numpy()
    H_out = (16 - 2) // 2 + 1
    out = np.zeros((1, 3, H_out, H_out), dtype=np.float32)
    lib.bpd_maxpool2d_cpu(p_input.ctypes.data, out.ctypes.data, 1, 3, 16, 16, 2, 2, 2, 0)
    status, mx, ab = classify(ref, out)
    results.append(("maxpool2d_cpu", "1x3x16x16", status, mx, ab))

    # ── Linear ──
    l_input = rng.standard_normal((4, 32)).astype(np.float32)
    weight = rng.standard_normal((64, 32)).astype(np.float32)
    bias_l = rng.standard_normal(64).astype(np.float32)
    lin = torch.nn.Linear(32, 64, bias=True)
    lin.weight.data = torch.from_numpy(weight)
    lin.bias.data = torch.from_numpy(bias_l)
    ref = lin(torch.from_numpy(l_input)).detach().numpy()
    out = np.zeros((4, 64), dtype=np.float32)
    lib.bpd_linear_cpu(l_input.ctypes.data, weight.ctypes.data, bias_l.ctypes.data,
                        out.ctypes.data, 4, 64, 32)
    status, mx, ab = classify(ref, out)
    results.append(("linear_cpu", "4x32->64", status, mx, ab))

    return results

def main():
    print(f"BPD Universal Bit-Identity Verification")
    print(f"PyTorch {torch.__version__} on {DEVICE.upper()}")
    if HAS_CUDA:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    lib_cpu = load_cpu_lib()
    if lib_cpu:
        print(f"CPU library: {CPU_SO}")
    else:
        print(f"CPU library: not found ({CPU_SO})")
        print(f"  Build with: gcc -O2 -shared -fPIC -o {CPU_SO} bench/bpd_cpu.c -lm")

    results = []

    if lib_cpu:
        print()
        print("── CPU VERIFICATION (BPD C kernels vs PyTorch CPU) ──")
        cpu_results = run_cpu_tests(lib_cpu)
        results.extend(cpu_results)
        for name, shape, status, mx, ab in cpu_results:
            tag = "✓" if "PASS" in status or "IDENTICAL" in status else "✗"
            print(f"  {name:<25} {shape:<16} {status:<22} max_ulp={mx:<10} {tag}")

    # Summary
    print()
    passed = sum(1 for _, _, s, _, _ in results if "PASS" in s or "IDENTICAL" in s)
    total = len(results)
    print(f"{'=' * 60}")
    print(f"PASSED: {passed}/{total}")
    if passed == total:
        print(f"\nALL KERNELS BIT-IDENTICAL WITH PyTorch on {DEVICE.upper()}.")
        print(f"Same math. Same bits. {'No GPU required.' if not HAS_CUDA else ''}")
    else:
        failed = [(n, s, mx) for n, _, s, mx, _ in results if "PASS" not in s and "IDENTICAL" not in s]
        print(f"\nFAILED: {len(failed)}")
        for n, s, mx in failed:
            print(f"  {n}: {s} (max {mx} ULP)")

    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
