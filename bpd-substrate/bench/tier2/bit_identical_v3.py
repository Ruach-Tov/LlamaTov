#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""bit_identical_v3.py — Extended sweep: norms + losses + pool + cumulative.

Per Heath's plan 8d65ba1c subtasks 2c through 2g (Track A). Extends the
bit_identical harness to cover all 28 validation cases. Each case follows
the same shape: input gen, reference compute, substrate compile+run,
classification.

This file deliberately keeps each case as a self-contained function so
substrate-design patterns surface clearly per kernel. Future refactoring
(after substantive findings cluster) can extract shared primitives.
"""
import ctypes
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/tmp")
from bit_identical_v1 import (
    compile_kernel, classify_result, BUILD_DIR, SEED, VALIDATION_DIR
)

# ────────────────────────────────────────────────────────────────────────
# Cumulative (cumsum, cumprod) — same (X, Y, N, outer) signature as reductions
# ────────────────────────────────────────────────────────────────────────

def cumulative_dispatch_body(kernel_name):
    return f"""
        float *d_X = nullptr, *d_Y = nullptr;
        size_t bytes = (size_t)outer * (size_t)N * sizeof(float);
        if (cudaMalloc(&d_X, bytes) != cudaSuccess) return 1;
        if (cudaMalloc(&d_Y, bytes) != cudaSuccess) {{ cudaFree(d_X); return 2; }}
        cudaMemcpy(d_X, h_X, bytes, cudaMemcpyHostToDevice);
        int block = 256;
        int grid = (outer + block - 1) / block;
        {kernel_name}<<<grid, block>>>(d_X, d_Y, N, outer);
        if (cudaGetLastError() != cudaSuccess) {{ cudaFree(d_X); cudaFree(d_Y); return 4; }}
        if (cudaDeviceSynchronize() != cudaSuccess) {{ cudaFree(d_X); cudaFree(d_Y); return 5; }}
        cudaMemcpy(h_Y, d_Y, bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_X); cudaFree(d_Y);
        return 0;
"""


def run_cumulative(case_name, kernel_name, op_kind):
    """Cumulative kernels write each step inline; output same shape as input."""
    torch.manual_seed(SEED)
    outer, N = 8, 32  # small to keep ref fast
    x = torch.randn(outer, N, dtype=torch.float32)

    # Reference: float32 cumulative scan with explicit casts
    x_np = x.numpy()
    y_ref = np.zeros((outer, N), dtype=np.float32)
    for o in range(outer):
        acc = np.float32(1.0) if op_kind == "cumprod" else np.float32(0.0)
        for i in range(N):
            if op_kind == "cumsum":
                acc = np.float32(acc + x_np[o, i])
            else:
                acc = np.float32(acc * x_np[o, i])
            y_ref[o, i] = acc

    x_flat = x.reshape(-1).contiguous().numpy()
    params = "const float *h_X, float *h_Y, int N, int outer"
    so = compile_kernel(case_name, kernel_name, params,
                       cumulative_dispatch_body(kernel_name))
    lib = ctypes.CDLL(str(so))
    fn = getattr(lib, f"run_{case_name}")
    fn.argtypes = [ctypes.POINTER(ctypes.c_float)]*2 + [ctypes.c_int]*2
    fn.restype = ctypes.c_int
    y_act = np.zeros(outer * N, dtype=np.float32)
    rc = fn(x_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y_act.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), N, outer)
    return classify_result(case_name, kernel_name, y_ref.reshape(-1), y_act, rc)


# ────────────────────────────────────────────────────────────────────────
# Norms — plain (X, Y, N, outer, eps) and affine (..., W, B)
# ────────────────────────────────────────────────────────────────────────

def norm_plain_dispatch_body(kernel_name):
    return f"""
        float *d_X = nullptr, *d_Y = nullptr;
        size_t bytes = (size_t)outer * (size_t)N * sizeof(float);
        if (cudaMalloc(&d_X, bytes) != cudaSuccess) return 1;
        if (cudaMalloc(&d_Y, bytes) != cudaSuccess) {{ cudaFree(d_X); return 2; }}
        cudaMemcpy(d_X, h_X, bytes, cudaMemcpyHostToDevice);
        int block = 256;
        int grid = (outer + block - 1) / block;
        {kernel_name}<<<grid, block>>>(d_X, d_Y, N, outer, eps);
        if (cudaGetLastError() != cudaSuccess) {{ cudaFree(d_X); cudaFree(d_Y); return 4; }}
        if (cudaDeviceSynchronize() != cudaSuccess) {{ cudaFree(d_X); cudaFree(d_Y); return 5; }}
        cudaMemcpy(h_Y, d_Y, bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_X); cudaFree(d_Y);
        return 0;
"""


def norm_affine_dispatch_body(kernel_name):
    return f"""
        float *d_X = nullptr, *d_Y = nullptr, *d_W = nullptr, *d_B = nullptr;
        size_t io_bytes = (size_t)outer * (size_t)N * sizeof(float);
        size_t wb_bytes = (size_t)N * sizeof(float);
        if (cudaMalloc(&d_X, io_bytes) != cudaSuccess) return 1;
        if (cudaMalloc(&d_Y, io_bytes) != cudaSuccess) {{ cudaFree(d_X); return 2; }}
        if (cudaMalloc(&d_W, wb_bytes) != cudaSuccess) {{ cudaFree(d_X); cudaFree(d_Y); return 21; }}
        if (cudaMalloc(&d_B, wb_bytes) != cudaSuccess) {{ cudaFree(d_X); cudaFree(d_Y); cudaFree(d_W); return 22; }}
        cudaMemcpy(d_X, h_X, io_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_W, h_W, wb_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, wb_bytes, cudaMemcpyHostToDevice);
        int block = 256;
        int grid = (outer + block - 1) / block;
        {kernel_name}<<<grid, block>>>(d_X, d_Y, N, outer, eps, d_W, d_B);
        if (cudaGetLastError() != cudaSuccess) {{ cudaFree(d_X); cudaFree(d_Y); cudaFree(d_W); cudaFree(d_B); return 4; }}
        if (cudaDeviceSynchronize() != cudaSuccess) {{ cudaFree(d_X); cudaFree(d_Y); cudaFree(d_W); cudaFree(d_B); return 5; }}
        cudaMemcpy(h_Y, d_Y, io_bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_X); cudaFree(d_Y); cudaFree(d_W); cudaFree(d_B);
        return 0;
"""


def _norm_layer_ref(x_np, eps):
    """LayerNorm: per-row, (x - mean) / sqrt(var + eps). float32 throughout."""
    outer, N = x_np.shape
    y = np.zeros_like(x_np)
    for o in range(outer):
        mean = np.float32(0.0)
        for i in range(N):
            mean = np.float32(mean + x_np[o, i])
        mean = np.float32(mean / N)
        var = np.float32(0.0)
        for i in range(N):
            d = np.float32(x_np[o, i] - mean)
            var = np.float32(var + np.float32(d * d))
        var = np.float32(var / N)
        inv = np.float32(1.0 / np.sqrt(var + eps))
        for i in range(N):
            y[o, i] = np.float32(np.float32(x_np[o, i] - mean) * inv)
    return y


def _norm_rms_ref(x_np, eps):
    """RMSNorm: x / sqrt(mean(x^2) + eps). float32."""
    outer, N = x_np.shape
    y = np.zeros_like(x_np)
    for o in range(outer):
        mean_sq = np.float32(0.0)
        for i in range(N):
            mean_sq = np.float32(mean_sq + np.float32(x_np[o, i] * x_np[o, i]))
        mean_sq = np.float32(mean_sq / N)
        inv = np.float32(1.0 / np.sqrt(mean_sq + eps))
        for i in range(N):
            y[o, i] = np.float32(x_np[o, i] * inv)
    return y


def _norm_l2_ref(x_np, eps):
    """L2Norm: x / sqrt(sum(x^2) + eps). float32."""
    outer, N = x_np.shape
    y = np.zeros_like(x_np)
    for o in range(outer):
        sum_sq = np.float32(0.0)
        for i in range(N):
            sum_sq = np.float32(sum_sq + np.float32(x_np[o, i] * x_np[o, i]))
        inv = np.float32(1.0 / np.sqrt(sum_sq + eps))
        for i in range(N):
            y[o, i] = np.float32(x_np[o, i] * inv)
    return y


def _norm_group_ref(x_np, eps):
    """GroupNorm = LayerNorm at this granularity (substrate emits the same template)."""
    return _norm_layer_ref(x_np, eps)


NORM_REFS = {
    "norm_layer_plain": (_norm_layer_ref, "norm_layer"),
    "norm_layer_affine": (_norm_layer_ref, "norm_layer"),  # affine adds W*y + B
    "norm_rms_plain":   (_norm_rms_ref,   "norm_rms"),
    "norm_rms_affine":  (_norm_rms_ref,   "norm_rms"),
    "norm_l2_plain":    (_norm_l2_ref,    "norm_l2"),
    "norm_l2_affine":   (_norm_l2_ref,    "norm_l2"),
    "norm_group_plain": (_norm_group_ref, "norm_group"),
}


def run_norm_plain(case_name):
    ref_fn, kernel_name = NORM_REFS[case_name]
    torch.manual_seed(SEED)
    outer, N = 8, 32
    eps = 1e-5
    x = torch.randn(outer, N, dtype=torch.float32)
    x_np = x.numpy()
    y_ref = ref_fn(x_np, eps)

    x_flat = x.reshape(-1).contiguous().numpy()
    params = "const float *h_X, float *h_Y, int N, int outer, float eps"
    so = compile_kernel(case_name, kernel_name, params,
                       norm_plain_dispatch_body(kernel_name))
    lib = ctypes.CDLL(str(so))
    fn = getattr(lib, f"run_{case_name}")
    fn.argtypes = [ctypes.POINTER(ctypes.c_float)]*2 + [ctypes.c_int]*2 + [ctypes.c_float]
    fn.restype = ctypes.c_int
    y_act = np.zeros(outer * N, dtype=np.float32)
    rc = fn(x_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y_act.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            N, outer, eps)
    return classify_result(case_name, kernel_name, y_ref.reshape(-1), y_act, rc)


def run_norm_affine(case_name):
    """Affine variant: apply W * y + B after norm."""
    ref_fn, kernel_name = NORM_REFS[case_name]
    torch.manual_seed(SEED)
    outer, N = 8, 32
    eps = 1e-5
    x = torch.randn(outer, N, dtype=torch.float32)
    W = torch.randn(N, dtype=torch.float32)
    B = torch.randn(N, dtype=torch.float32)
    x_np, W_np, B_np = x.numpy(), W.numpy(), B.numpy()
    y_normed = ref_fn(x_np, eps)
    # Apply affine: y = W * y_normed + B per element
    y_ref = np.zeros_like(y_normed)
    for o in range(outer):
        for i in range(N):
            y_ref[o, i] = np.float32(np.float32(W_np[i] * y_normed[o, i]) + B_np[i])

    x_flat = x.reshape(-1).contiguous().numpy()
    params = ("const float *h_X, float *h_Y, int N, int outer, float eps, "
              "const float *h_W, const float *h_B")
    so = compile_kernel(case_name, kernel_name, params,
                       norm_affine_dispatch_body(kernel_name))
    lib = ctypes.CDLL(str(so))
    fn = getattr(lib, f"run_{case_name}")
    fn.argtypes = ([ctypes.POINTER(ctypes.c_float)]*2 + [ctypes.c_int]*2 +
                   [ctypes.c_float] + [ctypes.POINTER(ctypes.c_float)]*2)
    fn.restype = ctypes.c_int
    y_act = np.zeros(outer * N, dtype=np.float32)
    rc = fn(x_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y_act.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            N, outer, eps,
            W_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            B_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    return classify_result(case_name, kernel_name, y_ref.reshape(-1), y_act, rc)


# ────────────────────────────────────────────────────────────────────────
# Loss kernels — (X, Y, Out, N, outer, ...) signatures
# ────────────────────────────────────────────────────────────────────────

def loss_basic_dispatch_body(kernel_name):
    """(X, Y, Out, N, outer) — mse, crossentropy, kldiv, hinge."""
    return f"""
        float *d_X = nullptr, *d_Y = nullptr, *d_Out = nullptr;
        size_t io_bytes = (size_t)outer * (size_t)N * sizeof(float);
        size_t out_bytes = (size_t)outer * sizeof(float);
        if (cudaMalloc(&d_X, io_bytes) != cudaSuccess) return 1;
        if (cudaMalloc(&d_Y, io_bytes) != cudaSuccess) {{ cudaFree(d_X); return 2; }}
        if (cudaMalloc(&d_Out, out_bytes) != cudaSuccess) {{ cudaFree(d_X); cudaFree(d_Y); return 3; }}
        cudaMemcpy(d_X, h_X, io_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Y, h_Y, io_bytes, cudaMemcpyHostToDevice);
        int block = 256;
        int grid = (outer + block - 1) / block;
        {kernel_name}<<<grid, block>>>(d_X, d_Y, d_Out, N, outer);
        if (cudaGetLastError() != cudaSuccess) {{ cudaFree(d_X); cudaFree(d_Y); cudaFree(d_Out); return 4; }}
        if (cudaDeviceSynchronize() != cudaSuccess) {{ cudaFree(d_X); cudaFree(d_Y); cudaFree(d_Out); return 5; }}
        cudaMemcpy(h_Out, d_Out, out_bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_X); cudaFree(d_Y); cudaFree(d_Out);
        return 0;
"""


def loss_huber_dispatch_body(kernel_name):
    """(X, Y, Out, N, outer, huber_delta)."""
    return f"""
        float *d_X = nullptr, *d_Y = nullptr, *d_Out = nullptr;
        size_t io_bytes = (size_t)outer * (size_t)N * sizeof(float);
        size_t out_bytes = (size_t)outer * sizeof(float);
        if (cudaMalloc(&d_X, io_bytes) != cudaSuccess) return 1;
        if (cudaMalloc(&d_Y, io_bytes) != cudaSuccess) {{ cudaFree(d_X); return 2; }}
        if (cudaMalloc(&d_Out, out_bytes) != cudaSuccess) {{ cudaFree(d_X); cudaFree(d_Y); return 3; }}
        cudaMemcpy(d_X, h_X, io_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Y, h_Y, io_bytes, cudaMemcpyHostToDevice);
        int block = 256;
        int grid = (outer + block - 1) / block;
        {kernel_name}<<<grid, block>>>(d_X, d_Y, d_Out, N, outer, huber_delta);
        if (cudaGetLastError() != cudaSuccess) {{ cudaFree(d_X); cudaFree(d_Y); cudaFree(d_Out); return 4; }}
        if (cudaDeviceSynchronize() != cudaSuccess) {{ cudaFree(d_X); cudaFree(d_Y); cudaFree(d_Out); return 5; }}
        cudaMemcpy(h_Out, d_Out, out_bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_X); cudaFree(d_Y); cudaFree(d_Out);
        return 0;
"""


def _loss_mse_ref(x_np, y_np, reduction):
    outer, N = x_np.shape
    out = np.zeros(outer, dtype=np.float32)
    for o in range(outer):
        s = np.float32(0.0)
        for i in range(N):
            d = np.float32(x_np[o, i] - y_np[o, i])
            s = np.float32(s + np.float32(d * d))
        out[o] = np.float32(s / N) if reduction == "mean" else s
    return out


def _loss_huber_ref(x_np, y_np, delta):
    outer, N = x_np.shape
    out = np.zeros(outer, dtype=np.float32)
    for o in range(outer):
        s = np.float32(0.0)
        for i in range(N):
            d = abs(np.float32(x_np[o, i] - y_np[o, i]))
            if d <= delta:
                s = np.float32(s + np.float32(np.float32(0.5) * np.float32(d * d)))
            else:
                s = np.float32(s + np.float32(delta) * np.float32(d - np.float32(0.5) * np.float32(delta)))
        out[o] = np.float32(s / N)
    return out


def _loss_crossentropy_ref(x_np, y_np):
    """Substrate emits: -y * log(x) summed and averaged (per ref source)."""
    outer, N = x_np.shape
    out = np.zeros(outer, dtype=np.float32)
    for o in range(outer):
        s = np.float32(0.0)
        for i in range(N):
            # Substrate's loss_crossentropy: -y * log(x), accumulate
            s = np.float32(s + np.float32(-y_np[o, i] * np.log(max(x_np[o, i], 1e-20))))
        out[o] = np.float32(s / N)
    return out


def _loss_kldiv_ref(x_np, y_np):
    outer, N = x_np.shape
    out = np.zeros(outer, dtype=np.float32)
    for o in range(outer):
        s = np.float32(0.0)
        for i in range(N):
            # KL: y * (log(y) - log(x))
            ly = np.log(max(y_np[o, i], 1e-20))
            lx = np.log(max(x_np[o, i], 1e-20))
            s = np.float32(s + np.float32(y_np[o, i] * np.float32(ly - lx)))
        out[o] = s
    return out


def _loss_hinge_ref(x_np, y_np):
    outer, N = x_np.shape
    out = np.zeros(outer, dtype=np.float32)
    for o in range(outer):
        s = np.float32(0.0)
        for i in range(N):
            # max(0, 1 - x*y)
            v = np.float32(np.float32(1.0) - np.float32(x_np[o, i] * y_np[o, i]))
            if v > 0:
                s = np.float32(s + v)
        out[o] = np.float32(s / N)
    return out


def run_loss(case_name):
    """Generic loss runner (handles mse_mean/sum, cross_entropy, kl_div, hinge)."""
    torch.manual_seed(SEED)
    outer, N = 8, 32
    x = torch.randn(outer, N, dtype=torch.float32)
    # Use positive y for log-using losses, otherwise random
    if "cross_entropy" in case_name or "kl_div" in case_name:
        x = torch.abs(x) + 0.01  # ensure positive
        y = torch.abs(torch.randn(outer, N, dtype=torch.float32)) + 0.01
    else:
        y = torch.randn(outer, N, dtype=torch.float32)
    x_np, y_np = x.numpy(), y.numpy()

    kernel_map = {
        "loss_mse_mean":      ("loss_mse", lambda: _loss_mse_ref(x_np, y_np, "mean")),
        "loss_mse_sum":       ("loss_mse", lambda: _loss_mse_ref(x_np, y_np, "sum")),
        "loss_cross_entropy": ("loss_crossentropy", lambda: _loss_crossentropy_ref(x_np, y_np)),
        "loss_kl_div":        ("loss_kldiv", lambda: _loss_kldiv_ref(x_np, y_np)),
        "loss_hinge":         ("loss_hinge", lambda: _loss_hinge_ref(x_np, y_np)),
    }
    kernel_name, ref_fn = kernel_map[case_name]
    y_ref = ref_fn()

    x_flat, y_flat = x.reshape(-1).contiguous().numpy(), y.reshape(-1).contiguous().numpy()
    params = "const float *h_X, const float *h_Y, float *h_Out, int N, int outer"
    so = compile_kernel(case_name, kernel_name, params,
                       loss_basic_dispatch_body(kernel_name))
    lib = ctypes.CDLL(str(so))
    fn = getattr(lib, f"run_{case_name}")
    fn.argtypes = [ctypes.POINTER(ctypes.c_float)]*3 + [ctypes.c_int]*2
    fn.restype = ctypes.c_int
    y_act = np.zeros(outer, dtype=np.float32)
    rc = fn(x_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y_act.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            N, outer)
    return classify_result(case_name, kernel_name, y_ref, y_act, rc)


def run_loss_huber():
    """Huber loss with delta parameter."""
    torch.manual_seed(SEED)
    outer, N = 8, 32
    delta = 1.0
    x = torch.randn(outer, N, dtype=torch.float32)
    y = torch.randn(outer, N, dtype=torch.float32)
    x_np, y_np = x.numpy(), y.numpy()
    y_ref = _loss_huber_ref(x_np, y_np, delta)
    x_flat, y_flat = x.reshape(-1).contiguous().numpy(), y.reshape(-1).contiguous().numpy()
    params = ("const float *h_X, const float *h_Y, float *h_Out, "
              "int N, int outer, float huber_delta")
    so = compile_kernel("loss_huber", "loss_huber", params,
                       loss_huber_dispatch_body("loss_huber"))
    lib = ctypes.CDLL(str(so))
    fn = lib.run_loss_huber
    fn.argtypes = [ctypes.POINTER(ctypes.c_float)]*3 + [ctypes.c_int]*2 + [ctypes.c_float]
    fn.restype = ctypes.c_int
    y_act = np.zeros(outer, dtype=np.float32)
    rc = fn(x_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y_act.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            N, outer, delta)
    return classify_result("loss_huber", "loss_huber", y_ref, y_act, rc)


def run_loss_triplet_margin():
    """Triplet margin: anchor, positive, negative inputs + margin param."""
    torch.manual_seed(SEED)
    outer, N = 8, 32
    margin = 1.0
    anchor = torch.randn(outer, N, dtype=torch.float32)
    pos = torch.randn(outer, N, dtype=torch.float32)
    neg = torch.randn(outer, N, dtype=torch.float32)
    a, p, n = anchor.numpy(), pos.numpy(), neg.numpy()

    # Reference: max(0, d(a, p) - d(a, n) + margin) per row, then mean
    out = np.zeros(outer, dtype=np.float32)
    for o in range(outer):
        dap = np.float32(0.0)
        dan = np.float32(0.0)
        for i in range(N):
            dap = np.float32(dap + np.float32(np.float32(a[o,i] - p[o,i]) ** 2))
            dan = np.float32(dan + np.float32(np.float32(a[o,i] - n[o,i]) ** 2))
        d = np.float32(np.float32(dap - dan) + np.float32(margin))
        out[o] = max(np.float32(0.0), d)

    dispatch = f"""
        float *d_A=nullptr, *d_P=nullptr, *d_N=nullptr, *d_Out=nullptr;
        size_t io = (size_t)outer*(size_t)N*sizeof(float);
        size_t ob = (size_t)outer*sizeof(float);
        cudaMalloc(&d_A, io); cudaMalloc(&d_P, io); cudaMalloc(&d_N, io); cudaMalloc(&d_Out, ob);
        cudaMemcpy(d_A, h_A, io, cudaMemcpyHostToDevice);
        cudaMemcpy(d_P, h_P, io, cudaMemcpyHostToDevice);
        cudaMemcpy(d_N, h_N, io, cudaMemcpyHostToDevice);
        int block=256, grid=(outer+block-1)/block;
        loss_triplet_margin<<<grid, block>>>(d_A, d_P, d_N, d_Out, N, outer, margin);
        cudaDeviceSynchronize();
        cudaMemcpy(h_Out, d_Out, ob, cudaMemcpyDeviceToHost);
        cudaFree(d_A); cudaFree(d_P); cudaFree(d_N); cudaFree(d_Out);
        return 0;
"""
    params = ("const float *h_A, const float *h_P, const float *h_N, "
              "float *h_Out, int N, int outer, float margin")
    so = compile_kernel("loss_triplet_margin", "loss_triplet_margin", params, dispatch)
    lib = ctypes.CDLL(str(so))
    fn = lib.run_loss_triplet_margin
    fn.argtypes = [ctypes.POINTER(ctypes.c_float)]*4 + [ctypes.c_int]*2 + [ctypes.c_float]
    fn.restype = ctypes.c_int
    a_flat, p_flat, n_flat = (anchor.reshape(-1).contiguous().numpy(),
                              pos.reshape(-1).contiguous().numpy(),
                              neg.reshape(-1).contiguous().numpy())
    y_act = np.zeros(outer, dtype=np.float32)
    rc = fn(a_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            p_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y_act.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            N, outer, margin)
    return classify_result("loss_triplet_margin", "loss_triplet_margin", out, y_act, rc)


# ────────────────────────────────────────────────────────────────────────
# Pool kernels — (X, Y, B, C, inH, inW, outH, outW, kH, kW, sH, sW, pH, pW)
# ────────────────────────────────────────────────────────────────────────

def pool_dispatch_body(kernel_name):
    """Substrate's pool kernel uses thread layout (per emit inspection 2g triage):
         blockIdx.z = batch (range 0..B)
         blockIdx.y = channel (range 0..C)
         blockIdx.x * blockDim.x + threadIdx.x = out_pos (range 0..outH*outW)
       Earlier dispatch was wrong (had (outH, outW, B*C) instead of (outH*outW, C, B))."""
    return f"""
        float *d_X = nullptr, *d_Y = nullptr;
        size_t in_bytes = (size_t)B*C*inH*inW*sizeof(float);
        size_t out_bytes = (size_t)B*C*outH*outW*sizeof(float);
        if (cudaMalloc(&d_X, in_bytes) != cudaSuccess) return 1;
        if (cudaMalloc(&d_Y, out_bytes) != cudaSuccess) {{ cudaFree(d_X); return 2; }}
        cudaMemcpy(d_X, h_X, in_bytes, cudaMemcpyHostToDevice);
        int out_pos_count = outH * outW;
        dim3 block(64, 1, 1), grid((out_pos_count+63)/64, C, B);
        {kernel_name}<<<grid, block>>>(d_X, d_Y, B, C, inH, inW, outH, outW,
                                       kH, kW, stride_h, stride_w, pad_h, pad_w);
        if (cudaGetLastError() != cudaSuccess) {{ cudaFree(d_X); cudaFree(d_Y); return 4; }}
        if (cudaDeviceSynchronize() != cudaSuccess) {{ cudaFree(d_X); cudaFree(d_Y); return 5; }}
        cudaMemcpy(h_Y, d_Y, out_bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_X); cudaFree(d_Y);
        return 0;
"""


def run_pool(case_name, op):
    """Triage 2g 2026-05-19 ~22:55 UTC: substrate matches PyTorch defaults:
       avg_pool: divide by IN-BOUNDS count, not kH*kW (count_include_pad=False)
       max_pool: leave acc at substrate's sentinel (-1e30) for fully-padded
                 positions (substrate doesn't clamp). Use same sentinel in ref.
       Earlier reference (divide by kH*kW, clamp -inf to 0) was wrong."""
    torch.manual_seed(SEED)
    B, C = 2, 3
    inH, inW = 8, 8
    kH, kW = 3, 3
    stride_h, stride_w = 1, 1
    pad_h, pad_w = 0, 0
    outH = (inH + 2*pad_h - kH)//stride_h + 1
    outW = (inW + 2*pad_w - kW)//stride_w + 1

    x = torch.randn(B, C, inH, inW, dtype=torch.float32)
    x_np = x.numpy()
    y_ref = np.zeros((B, C, outH, outW), dtype=np.float32)
    # Substrate's sentinel for max-pool initial: -1e30 (from emit inspection)
    NEG_SENTINEL = np.float32(-1.0e30)
    for b in range(B):
        for c in range(C):
            for h in range(outH):
                for w in range(outW):
                    if op == "max":
                        best = NEG_SENTINEL
                        for kh in range(kH):
                            for kw in range(kW):
                                ih, iw = h*stride_h - pad_h + kh, w*stride_w - pad_w + kw
                                if 0 <= ih < inH and 0 <= iw < inW:
                                    v = np.float32(x_np[b,c,ih,iw])
                                    if v > best:
                                        best = v
                        y_ref[b,c,h,w] = best  # substrate doesn't clamp
                    else:  # avg
                        s = np.float32(0.0); count = 0
                        for kh in range(kH):
                            for kw in range(kW):
                                ih, iw = h*stride_h - pad_h + kh, w*stride_w - pad_w + kw
                                if 0 <= ih < inH and 0 <= iw < inW:
                                    s = np.float32(s + x_np[b,c,ih,iw])
                                    count += 1
                        # Substrate divides by IN-BOUNDS count (matches PyTorch
                        # count_include_pad=False)
                        y_ref[b,c,h,w] = np.float32(s / count) if count > 0 else np.float32(0.0)
    x_flat = x.reshape(-1).contiguous().numpy()
    kernel_name = f"pool_2d_{op}"
    params = ("const float *h_X, float *h_Y, int B, int C, int inH, int inW, "
              "int outH, int outW, int kH, int kW, "
              "int stride_h, int stride_w, int pad_h, int pad_w")
    so = compile_kernel(case_name, kernel_name, params,
                       pool_dispatch_body(kernel_name))
    lib = ctypes.CDLL(str(so))
    fn = getattr(lib, f"run_{case_name}")
    fn.argtypes = [ctypes.POINTER(ctypes.c_float)]*2 + [ctypes.c_int]*12
    fn.restype = ctypes.c_int
    y_act = np.zeros(B*C*outH*outW, dtype=np.float32)
    rc = fn(x_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y_act.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            B, C, inH, inW, outH, outW, kH, kW,
            stride_h, stride_w, pad_h, pad_w)
    return classify_result(case_name, kernel_name, y_ref.reshape(-1), y_act, rc)


# ────────────────────────────────────────────────────────────────────────
# Main: run all 17 additional cases
# ────────────────────────────────────────────────────────────────────────

def main():
    print("=== Tier 2 / Track A — bit_identical_v3 (28-case extension) ===\n")
    results = []

    print("--- Cumulative ---")
    for case, kname, op in [("cumsum", "cumsum", "cumsum"),
                            ("cumprod", "cumprod", "cumprod")]:
        try:
            r = run_cumulative(case, kname, op)
        except Exception as e:
            r = {"case": case, "kernel": kname, "status": "HARNESS_ERROR", "detail": str(e)[:200]}
        results.append(r)
        print(f"  {r['status']:<32} {case:<32} {r.get('detail', '')}")

    print("\n--- Norms (plain) ---")
    for case in ["norm_layer_plain", "norm_rms_plain", "norm_l2_plain", "norm_group_plain"]:
        try:
            r = run_norm_plain(case)
        except Exception as e:
            r = {"case": case, "kernel": "?", "status": "HARNESS_ERROR", "detail": str(e)[:200]}
        results.append(r)
        print(f"  {r['status']:<32} {case:<32} {r.get('detail', '')}")

    print("\n--- Norms (affine) ---")
    for case in ["norm_layer_affine", "norm_rms_affine", "norm_l2_affine"]:
        try:
            r = run_norm_affine(case)
        except Exception as e:
            r = {"case": case, "kernel": "?", "status": "HARNESS_ERROR", "detail": str(e)[:200]}
        results.append(r)
        print(f"  {r['status']:<32} {case:<32} {r.get('detail', '')}")

    print("\n--- Losses ---")
    for case in ["loss_mse_mean", "loss_mse_sum", "loss_cross_entropy",
                 "loss_kl_div", "loss_hinge"]:
        try:
            r = run_loss(case)
        except Exception as e:
            r = {"case": case, "kernel": "?", "status": "HARNESS_ERROR", "detail": str(e)[:200]}
        results.append(r)
        print(f"  {r['status']:<32} {case:<32} {r.get('detail', '')}")

    print("\n--- Huber + Triplet ---")
    for runner in [run_loss_huber, run_loss_triplet_margin]:
        try:
            r = runner()
        except Exception as e:
            r = {"case": runner.__name__, "kernel": "?", "status": "HARNESS_ERROR", "detail": str(e)[:200]}
        results.append(r)
        print(f"  {r['status']:<32} {r.get('case', '?'):<32} {r.get('detail', '')}")

    print("\n--- Pool ---")
    for case, op in [("pool_2d_max", "max"), ("pool_2d_avg", "avg")]:
        try:
            r = run_pool(case, op)
        except Exception as e:
            r = {"case": case, "kernel": "?", "status": "HARNESS_ERROR", "detail": str(e)[:200]}
        results.append(r)
        print(f"  {r['status']:<32} {case:<32} {r.get('detail', '')}")

    print("\n=== Summary ===")
    by_status = {}
    for r in results:
        by_status.setdefault(r["status"], 0)
        by_status[r["status"]] += 1
    for s, n in sorted(by_status.items()):
        print(f"  {s:<32} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
