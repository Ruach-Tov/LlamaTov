#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""bit_identical_v2_with_conv.py — Extended sweep covering conv stubs.

Adds: conv_1d, conv_3d, conv_transpose_2d as test cases. These use a
generic "stub probe" pattern — we feed non-trivial input and check
whether output is all-zero. Stub kernels return all-zero (their body
doesn't write anything except maybe thread-setup variables).

Per Heath's plan 8d65ba1c subtask 2-inf-g-2.
"""
import ctypes
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

VALIDATION_DIR = Path("/tmp/l1_cuda_validation")
BUILD_DIR = Path("/tmp/tier2_build")
BUILD_DIR.mkdir(exist_ok=True)
SEED = 42

# Re-use the harness functions from v1
sys.path.insert(0, "/tmp")
from bit_identical_v1 import (
    compile_kernel, reduction_dispatch_body, run_reduction_case,
    classify_result, REDUCTION_CASES
)


def conv_2d_dispatch_body(kernel_name):
    """Dispatch for im2col_2d_forward: 16 params."""
    return f"""
        const int outH = (H + 2*pad_h - dilation_h*(kH-1) - 1)/stride_h + 1;
        const int outW = (W + 2*pad_w - dilation_w*(kW-1) - 1)/stride_w + 1;
        float *d_X = nullptr, *d_Y = nullptr;
        size_t in_bytes = (size_t)B*C*H*W*sizeof(float);
        size_t out_bytes = (size_t)B*outH*outW*C*kH*kW*sizeof(float);
        if (cudaMalloc(&d_X, in_bytes) != cudaSuccess) return 1;
        if (cudaMalloc(&d_Y, out_bytes) != cudaSuccess) {{ cudaFree(d_X); return 2; }}
        cudaMemset(d_Y, 0, out_bytes);  // Important: clear so stubs are detectable
        if (cudaMemcpy(d_X, h_X, in_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {{
            cudaFree(d_X); cudaFree(d_Y); return 3;
        }}
        dim3 block(16, 1, 1);
        dim3 grid((outW+15)/16, outH, B);
        {kernel_name}<<<grid, block>>>(d_X, d_Y, B, C, H, W, kH, kW, outH, outW,
                                       stride_h, stride_w, pad_h, pad_w,
                                       dilation_h, dilation_w);
        if (cudaGetLastError() != cudaSuccess) {{ cudaFree(d_X); cudaFree(d_Y); return 4; }}
        if (cudaDeviceSynchronize() != cudaSuccess) {{ cudaFree(d_X); cudaFree(d_Y); return 5; }}
        if (cudaMemcpy(h_Y, d_Y, out_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {{
            cudaFree(d_X); cudaFree(d_Y); return 6;
        }}
        cudaFree(d_X); cudaFree(d_Y);
        return 0;
"""


def conv_1d_dispatch_body(kernel_name):
    """Dispatch for im2col_1d_forward stub."""
    return f"""
        const int outL = (L + 2*pad - dilation*(kL-1) - 1)/stride + 1;
        float *d_X = nullptr, *d_Y = nullptr;
        size_t in_bytes = (size_t)B*C*L*sizeof(float);
        size_t out_bytes = (size_t)B*outL*C*kL*sizeof(float);
        if (cudaMalloc(&d_X, in_bytes) != cudaSuccess) return 1;
        if (cudaMalloc(&d_Y, out_bytes) != cudaSuccess) {{ cudaFree(d_X); return 2; }}
        cudaMemset(d_Y, 0, out_bytes);
        cudaMemcpy(d_X, h_X, in_bytes, cudaMemcpyHostToDevice);
        dim3 block(64), grid((outL+63)/64, B);
        {kernel_name}<<<grid, block>>>(d_X, d_Y, B, C, L, kL, outL, stride, pad, dilation);
        if (cudaGetLastError() != cudaSuccess) {{ cudaFree(d_X); cudaFree(d_Y); return 4; }}
        if (cudaDeviceSynchronize() != cudaSuccess) {{ cudaFree(d_X); cudaFree(d_Y); return 5; }}
        cudaMemcpy(h_Y, d_Y, out_bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_X); cudaFree(d_Y);
        return 0;
"""


def conv_3d_dispatch_body(kernel_name):
    """Dispatch for im2col_3d_forward — 22 params (full impl per 2-rs-i-2)."""
    return f"""
        float *d_X = nullptr, *d_Y = nullptr;
        size_t in_bytes = (size_t)B*C*D*H*W*sizeof(float);
        size_t out_bytes = (size_t)B*outD*outH*outW*C*kD*kH*kW*sizeof(float);
        if (cudaMalloc(&d_X, in_bytes) != cudaSuccess) return 1;
        if (cudaMalloc(&d_Y, out_bytes) != cudaSuccess) {{ cudaFree(d_X); return 2; }}
        cudaMemset(d_Y, 0, out_bytes);
        cudaMemcpy(d_X, h_X, in_bytes, cudaMemcpyHostToDevice);
        dim3 block(16, 1, 1), grid((outW+15)/16, outH, B*outD);
        {kernel_name}<<<grid, block>>>(d_X, d_Y, B, C, D, H, W, kD, kH, kW,
                                       outD, outH, outW,
                                       stride_d, stride_h, stride_w,
                                       pad_d, pad_h, pad_w,
                                       dilation_d, dilation_h, dilation_w);
        if (cudaGetLastError() != cudaSuccess) {{ cudaFree(d_X); cudaFree(d_Y); return 4; }}
        if (cudaDeviceSynchronize() != cudaSuccess) {{ cudaFree(d_X); cudaFree(d_Y); return 5; }}
        cudaMemcpy(h_Y, d_Y, out_bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_X); cudaFree(d_Y);
        return 0;
"""


def conv_transpose_1d_dispatch_body(kernel_name):
    """Dispatch for col2im_1d_transpose — gather pattern, 10 params."""
    return f"""
        float *d_X = nullptr, *d_Y = nullptr;
        size_t in_bytes = (size_t)B*L_in_col*C*kL*sizeof(float);
        size_t out_bytes = (size_t)B*C*L_out*sizeof(float);
        if (cudaMalloc(&d_X, in_bytes) != cudaSuccess) return 1;
        if (cudaMalloc(&d_Y, out_bytes) != cudaSuccess) {{ cudaFree(d_X); return 2; }}
        cudaMemset(d_Y, 0, out_bytes);
        cudaMemcpy(d_X, h_X, in_bytes, cudaMemcpyHostToDevice);
        dim3 block(32, 1, 1), grid((L_out+31)/32, C, B);
        {kernel_name}<<<grid, block>>>(d_X, d_Y, B, C, L_in_col, L_out, kL,
                                       stride, pad, dilation);
        if (cudaGetLastError() != cudaSuccess) {{ cudaFree(d_X); cudaFree(d_Y); return 4; }}
        if (cudaDeviceSynchronize() != cudaSuccess) {{ cudaFree(d_X); cudaFree(d_Y); return 5; }}
        cudaMemcpy(h_Y, d_Y, out_bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_X); cudaFree(d_Y);
        return 0;
"""


def conv_transpose_3d_dispatch_body(kernel_name):
    """Dispatch for col2im_3d_transpose — 22 params (3D analog of 2D)."""
    return f"""
        float *d_X = nullptr, *d_Y = nullptr;
        size_t in_bytes = (size_t)B*D_in*H_in*W_in*C*kD*kH*kW*sizeof(float);
        size_t out_bytes = (size_t)B*C*D_out*H_out*W_out*sizeof(float);
        if (cudaMalloc(&d_X, in_bytes) != cudaSuccess) return 1;
        if (cudaMalloc(&d_Y, out_bytes) != cudaSuccess) {{ cudaFree(d_X); return 2; }}
        cudaMemset(d_Y, 0, out_bytes);
        cudaMemcpy(d_X, h_X, in_bytes, cudaMemcpyHostToDevice);
        dim3 block(16, 1, 1), grid((W_out+15)/16, D_out*H_out, B*C);
        {kernel_name}<<<grid, block>>>(d_X, d_Y, B, C, D_in, H_in, W_in,
                                       D_out, H_out, W_out, kD, kH, kW,
                                       stride_d, stride_h, stride_w,
                                       pad_d, pad_h, pad_w,
                                       dilation_d, dilation_h, dilation_w);
        if (cudaGetLastError() != cudaSuccess) {{ cudaFree(d_X); cudaFree(d_Y); return 4; }}
        if (cudaDeviceSynchronize() != cudaSuccess) {{ cudaFree(d_X); cudaFree(d_Y); return 5; }}
        cudaMemcpy(h_Y, d_Y, out_bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_X); cudaFree(d_Y);
        return 0;
"""


def conv_transpose_2d_dispatch_body(kernel_name):
    """Dispatch for col2im_2d_transpose — full impl per 2-rs-i-4. 16 params."""
    return f"""
        float *d_X = nullptr, *d_Y = nullptr;
        size_t in_bytes = (size_t)B*H_in*W_in*C*kH*kW*sizeof(float);
        size_t out_bytes = (size_t)B*C*H_out*W_out*sizeof(float);
        if (cudaMalloc(&d_X, in_bytes) != cudaSuccess) return 1;
        if (cudaMalloc(&d_Y, out_bytes) != cudaSuccess) {{ cudaFree(d_X); return 2; }}
        cudaMemset(d_Y, 0, out_bytes);
        cudaMemcpy(d_X, h_X, in_bytes, cudaMemcpyHostToDevice);
        dim3 block(16, 1, 1), grid((W_out+15)/16, H_out, B*C);
        {kernel_name}<<<grid, block>>>(d_X, d_Y, B, C, H_in, W_in, H_out, W_out,
                                       kH, kW, stride_h, stride_w,
                                       pad_h, pad_w, dilation_h, dilation_w);
        if (cudaGetLastError() != cudaSuccess) {{ cudaFree(d_X); cudaFree(d_Y); return 4; }}
        if (cudaDeviceSynchronize() != cudaSuccess) {{ cudaFree(d_X); cudaFree(d_Y); return 5; }}
        cudaMemcpy(h_Y, d_Y, out_bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_X); cudaFree(d_Y);
        return 0;
"""


def run_conv_2d_case():
    """conv_2d_forward (im2col_2d_forward) — fully implemented; should pass."""
    torch.manual_seed(SEED)
    B, C, H, W = 2, 3, 8, 8
    kH, kW = 3, 3
    stride_h, stride_w = 1, 1
    pad_h, pad_w = 1, 1
    dilation_h, dilation_w = 1, 1
    outH = (H + 2*pad_h - dilation_h*(kH-1) - 1)//stride_h + 1
    outW = (W + 2*pad_w - dilation_w*(kW-1) - 1)//stride_w + 1

    x = torch.randn(B, C, H, W, dtype=torch.float32)
    x_flat = x.reshape(-1).contiguous().numpy()

    # PyTorch reference: torch.nn.functional.unfold for im2col
    y_ref_unfold = torch.nn.functional.unfold(
        x, kernel_size=(kH, kW), padding=(pad_h, pad_w),
        stride=(stride_h, stride_w), dilation=(dilation_h, dilation_w))
    # unfold output is (B, C*kH*kW, outH*outW); substrate emits (B, outH, outW, C, kH, kW)
    # Transpose to match substrate layout
    y_ref = y_ref_unfold.transpose(1, 2).reshape(B, outH, outW, C, kH, kW).contiguous()
    y_ref_np = y_ref.numpy().reshape(-1)

    params_decl = "const float *h_X, float *h_Y, int B, int C, int H, int W, int kH, int kW, int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w"
    so_path = compile_kernel("conv_2d_forward", "im2col_2d_forward",
                              params_decl, conv_2d_dispatch_body("im2col_2d_forward"))
    lib = ctypes.CDLL(str(so_path))
    fn = lib.run_conv_2d_forward
    fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)] + [ctypes.c_int]*12
    fn.restype = ctypes.c_int

    y_act = np.zeros(B*outH*outW*C*kH*kW, dtype=np.float32)
    rc = fn(
        x_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        y_act.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B, C, H, W, kH, kW, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w
    )
    return classify_result("conv_2d_forward", "im2col_2d_forward", y_ref_np, y_act, rc)


def run_conv_1d_case():
    """im2col_1d_forward — full body implemented per subtask 2-rs-i-1.
    Compare against torch.nn.functional.unfold (1D analog)."""
    torch.manual_seed(SEED)
    B, C, L = 2, 3, 16
    kL = 3
    stride, pad, dilation = 1, 1, 1
    outL = (L + 2*pad - dilation*(kL-1) - 1)//stride + 1
    x = torch.randn(B, C, L, dtype=torch.float32)
    x_flat = x.reshape(-1).contiguous().numpy()

    # PyTorch reference: unfold expects (B, C, L) → uses 2D unfold trick
    # Treat 1D as (B, C, L, 1) for unfold
    x_2d = x.unsqueeze(-1)  # (B, C, L, 1)
    y_unfold = torch.nn.functional.unfold(
        x_2d, kernel_size=(kL, 1), padding=(pad, 0),
        stride=(stride, 1), dilation=(dilation, 1))
    # y_unfold shape: (B, C*kL, outL)
    # Substrate emits (B, outL, C, kL): out_row=b*outL+ol, col=c*kL+kl
    # So Y[(b*outL+ol)*(C*kL) + (c*kL+kl)] = X[b,c,ol*stride-pad+kl*dilation]
    # Re-arrange unfold output to substrate's expected layout
    y_ref = y_unfold.transpose(1, 2).reshape(B, outL, C, kL).contiguous()
    y_ref_np = y_ref.numpy().reshape(-1)

    params_decl = "const float *h_X, float *h_Y, int B, int C, int L, int kL, int stride, int pad, int dilation"
    so_path = compile_kernel("conv_1d_forward", "im2col_1d_forward",
                              params_decl, conv_1d_dispatch_body("im2col_1d_forward"))
    lib = ctypes.CDLL(str(so_path))
    fn = lib.run_conv_1d_forward
    fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)] + [ctypes.c_int]*7
    fn.restype = ctypes.c_int

    y_act = np.zeros(B*outL*C*kL, dtype=np.float32)
    rc = fn(
        x_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        y_act.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B, C, L, kL, stride, pad, dilation
    )

    return classify_result("conv_1d_forward", "im2col_1d_forward",
                           y_ref_np, y_act, rc)


def run_conv_3d_case():
    """im2col_3d_forward — full body implemented per subtask 2-rs-i-2.
    Compare against torch.nn.functional.unfold via manual 3D unfold (PyTorch's
    unfold is 2D only; we implement the 3D reference inline)."""
    torch.manual_seed(SEED)
    B, C, D, H, W = 1, 2, 4, 4, 4
    kD, kH, kW = 2, 2, 2
    stride_d, stride_h, stride_w = 1, 1, 1
    pad_d, pad_h, pad_w = 0, 0, 0
    dilation_d, dilation_h, dilation_w = 1, 1, 1
    outD = (D + 2*pad_d - dilation_d*(kD-1) - 1)//stride_d + 1
    outH = (H + 2*pad_h - dilation_h*(kH-1) - 1)//stride_h + 1
    outW = (W + 2*pad_w - dilation_w*(kW-1) - 1)//stride_w + 1

    x = torch.randn(B, C, D, H, W, dtype=torch.float32)
    x_flat = x.reshape(-1).contiguous().numpy()

    # PyTorch reference: manual 3D unfold matching substrate's layout
    # Y[((b*outD+od)*outH+oh)*outW+ow, ((c*kD+kd)*kH+kh)*kW+kw] =
    #   X[b, c, od*stride_d - pad_d + kd*dilation_d, oh*..., ow*...] (or 0 if OOB)
    y_size = B*outD*outH*outW*C*kD*kH*kW
    y_ref = torch.zeros(y_size, dtype=torch.float32)
    for b in range(B):
        for od in range(outD):
            for oh in range(outH):
                for ow in range(outW):
                    out_row = ((b*outD + od)*outH + oh)*outW + ow
                    for c in range(C):
                        for kd in range(kD):
                            for kh in range(kH):
                                for kw in range(kW):
                                    id_ = od*stride_d - pad_d + kd*dilation_d
                                    ih = oh*stride_h - pad_h + kh*dilation_h
                                    iw = ow*stride_w - pad_w + kw*dilation_w
                                    col = ((c*kD + kd)*kH + kh)*kW + kw
                                    patch_size = C*kD*kH*kW
                                    idx = out_row * patch_size + col
                                    if 0 <= id_ < D and 0 <= ih < H and 0 <= iw < W:
                                        y_ref[idx] = x[b, c, id_, ih, iw]
    y_ref_np = y_ref.numpy()

    params_decl = ("const float *h_X, float *h_Y, "
                   "int B, int C, int D, int H, int W, "
                   "int kD, int kH, int kW, "
                   "int outD, int outH, int outW, "
                   "int stride_d, int stride_h, int stride_w, "
                   "int pad_d, int pad_h, int pad_w, "
                   "int dilation_d, int dilation_h, int dilation_w")
    so_path = compile_kernel("conv_3d_forward", "im2col_3d_forward",
                              params_decl, conv_3d_dispatch_body("im2col_3d_forward"))
    lib = ctypes.CDLL(str(so_path))
    fn = lib.run_conv_3d_forward
    fn.argtypes = ([ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)] +
                   [ctypes.c_int]*20)
    fn.restype = ctypes.c_int

    y_act = np.zeros(y_size, dtype=np.float32)
    rc = fn(
        x_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        y_act.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B, C, D, H, W, kD, kH, kW, outD, outH, outW,
        stride_d, stride_h, stride_w, pad_d, pad_h, pad_w,
        dilation_d, dilation_h, dilation_w
    )

    return classify_result("conv_3d_forward", "im2col_3d_forward",
                           y_ref_np, y_act, rc)


def run_conv_transpose_1d_case():
    """col2im_1d_transpose — full body per subtask 2-rs-i-3.
    Compare against manual Python reference implementing the inverse map."""
    torch.manual_seed(SEED)
    B, C = 2, 3
    L_in_col = 8
    kL = 3
    stride, pad, dilation = 1, 0, 1
    # Inverse formula for L_out given L_in_col and conv-transpose semantics:
    #   L_out = (L_in_col - 1)*stride - 2*pad + dilation*(kL-1) + 1
    L_out = (L_in_col - 1)*stride - 2*pad + dilation*(kL-1) + 1

    # Generate col matrix as input
    x = torch.randn(B, L_in_col, C, kL, dtype=torch.float32)
    x_flat = x.reshape(-1).contiguous().numpy()

    # Reference: for each (b, c, l), sum over kl where ol = (l+pad-kl*dilation)/stride
    # is an integer and 0 <= ol < L_in_col.
    # SUBSTANTIVE NOTE: accumulator must be float32 (matching kernel's `float acc`).
    # Using Python float (=double) accumulator would give silently higher precision
    # and produce 1-6 ULP divergence due to truncation at final cast.
    x_np = x.numpy()  # (B, L_in_col, C, kL), float32
    y_ref = np.zeros((B, C, L_out), dtype=np.float32)
    for b in range(B):
        for c in range(C):
            for l in range(L_out):
                acc = np.float32(0.0)
                for kl in range(kL):
                    numerator = l + pad - kl*dilation
                    if numerator % stride == 0:
                        ol = numerator // stride
                        if 0 <= ol < L_in_col:
                            acc = np.float32(acc + x_np[b, ol, c, kl])
                y_ref[b, c, l] = acc
    y_ref_np = y_ref.reshape(-1)

    params_decl = ("const float *h_X, float *h_Y, "
                   "int B, int C, int L_in_col, int L_out, int kL, "
                   "int stride, int pad, int dilation")
    so_path = compile_kernel("conv_transpose_1d", "col2im_1d_transpose",
                              params_decl, conv_transpose_1d_dispatch_body("col2im_1d_transpose"))
    lib = ctypes.CDLL(str(so_path))
    fn = lib.run_conv_transpose_1d
    fn.argtypes = ([ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)] +
                   [ctypes.c_int]*8)
    fn.restype = ctypes.c_int

    y_act = np.zeros(B*C*L_out, dtype=np.float32)
    rc = fn(
        x_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        y_act.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B, C, L_in_col, L_out, kL, stride, pad, dilation
    )

    return classify_result("conv_transpose_1d", "col2im_1d_transpose",
                           y_ref_np, y_act, rc)


def run_conv_transpose_2d_case():
    """col2im_2d_transpose — full body per subtask 2-rs-i-4.
    Compare against manual Python reference (float32 accumulator)."""
    torch.manual_seed(SEED)
    B, C = 2, 3
    H_in, W_in = 4, 5
    kH, kW = 3, 3
    stride_h, stride_w = 1, 1
    pad_h, pad_w = 0, 0
    dilation_h, dilation_w = 1, 1
    H_out = (H_in - 1)*stride_h - 2*pad_h + dilation_h*(kH-1) + 1
    W_out = (W_in - 1)*stride_w - 2*pad_w + dilation_w*(kW-1) + 1

    # Col matrix layout: [B*H_in*W_in, C*kH*kW]
    x = torch.randn(B, H_in, W_in, C, kH, kW, dtype=torch.float32)
    x_flat = x.reshape(-1).contiguous().numpy()
    x_np = x.numpy()

    # Reference: gather (oh, kh, ow, kw) -> col_row, col_col
    y_ref = np.zeros((B, C, H_out, W_out), dtype=np.float32)
    for b in range(B):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    acc = np.float32(0.0)
                    for kh in range(kH):
                        num_h = h + pad_h - kh*dilation_h
                        if num_h % stride_h != 0:
                            continue
                        oh = num_h // stride_h
                        if not (0 <= oh < H_in):
                            continue
                        for kw in range(kW):
                            num_w = w + pad_w - kw*dilation_w
                            if num_w % stride_w != 0:
                                continue
                            ow = num_w // stride_w
                            if not (0 <= ow < W_in):
                                continue
                            acc = np.float32(acc + x_np[b, oh, ow, c, kh, kw])
                    y_ref[b, c, h, w] = acc
    y_ref_np = y_ref.reshape(-1)

    params_decl = ("const float *h_X, float *h_Y, "
                   "int B, int C, int H_in, int W_in, int H_out, int W_out, "
                   "int kH, int kW, int stride_h, int stride_w, "
                   "int pad_h, int pad_w, int dilation_h, int dilation_w")
    so_path = compile_kernel("conv_transpose_2d", "col2im_2d_transpose",
                              params_decl, conv_transpose_2d_dispatch_body("col2im_2d_transpose"))
    lib = ctypes.CDLL(str(so_path))
    fn = lib.run_conv_transpose_2d
    fn.argtypes = ([ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)] +
                   [ctypes.c_int]*14)
    fn.restype = ctypes.c_int

    y_act = np.zeros(B*C*H_out*W_out, dtype=np.float32)
    rc = fn(
        x_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        y_act.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B, C, H_in, W_in, H_out, W_out, kH, kW,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w
    )

    return classify_result("conv_transpose_2d", "col2im_2d_transpose",
                           y_ref_np, y_act, rc)


def run_conv_transpose_3d_case():
    """col2im_3d_transpose — full body per subtask 2-rs-i-5.
    3D analog of 2D transpose, smallest test (3D shapes get big quickly)."""
    torch.manual_seed(SEED)
    B, C = 1, 2
    D_in, H_in, W_in = 3, 3, 3
    kD, kH, kW = 2, 2, 2
    stride_d, stride_h, stride_w = 1, 1, 1
    pad_d, pad_h, pad_w = 0, 0, 0
    dilation_d, dilation_h, dilation_w = 1, 1, 1
    D_out = (D_in - 1)*stride_d - 2*pad_d + dilation_d*(kD-1) + 1
    H_out = (H_in - 1)*stride_h - 2*pad_h + dilation_h*(kH-1) + 1
    W_out = (W_in - 1)*stride_w - 2*pad_w + dilation_w*(kW-1) + 1

    x = torch.randn(B, D_in, H_in, W_in, C, kD, kH, kW, dtype=torch.float32)
    x_flat = x.reshape(-1).contiguous().numpy()
    x_np = x.numpy()

    y_ref = np.zeros((B, C, D_out, H_out, W_out), dtype=np.float32)
    for b in range(B):
        for c in range(C):
            for d in range(D_out):
                for h in range(H_out):
                    for w in range(W_out):
                        acc = np.float32(0.0)
                        for kd in range(kD):
                            nd = d + pad_d - kd*dilation_d
                            if nd % stride_d != 0: continue
                            od = nd // stride_d
                            if not (0 <= od < D_in): continue
                            for kh in range(kH):
                                nh = h + pad_h - kh*dilation_h
                                if nh % stride_h != 0: continue
                                oh = nh // stride_h
                                if not (0 <= oh < H_in): continue
                                for kw in range(kW):
                                    nw = w + pad_w - kw*dilation_w
                                    if nw % stride_w != 0: continue
                                    ow = nw // stride_w
                                    if not (0 <= ow < W_in): continue
                                    acc = np.float32(acc + x_np[b, od, oh, ow, c, kd, kh, kw])
                        y_ref[b, c, d, h, w] = acc
    y_ref_np = y_ref.reshape(-1)

    params_decl = ("const float *h_X, float *h_Y, "
                   "int B, int C, int D_in, int H_in, int W_in, "
                   "int D_out, int H_out, int W_out, "
                   "int kD, int kH, int kW, "
                   "int stride_d, int stride_h, int stride_w, "
                   "int pad_d, int pad_h, int pad_w, "
                   "int dilation_d, int dilation_h, int dilation_w")
    so_path = compile_kernel("conv_transpose_3d", "col2im_3d_transpose",
                              params_decl, conv_transpose_3d_dispatch_body("col2im_3d_transpose"))
    lib = ctypes.CDLL(str(so_path))
    fn = lib.run_conv_transpose_3d
    fn.argtypes = ([ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)] +
                   [ctypes.c_int]*20)
    fn.restype = ctypes.c_int

    y_act = np.zeros(B*C*D_out*H_out*W_out, dtype=np.float32)
    rc = fn(
        x_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        y_act.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B, C, D_in, H_in, W_in, D_out, H_out, W_out, kD, kH, kW,
        stride_d, stride_h, stride_w, pad_d, pad_h, pad_w,
        dilation_d, dilation_h, dilation_w
    )

    return classify_result("conv_transpose_3d", "col2im_3d_transpose",
                           y_ref_np, y_act, rc)


def main():
    print("=== Tier 2 / Subtask 2-inf-g-2 — bit_identical v2 (reductions + conv probe) ===")
    print()

    results = []

    print("--- Reductions ---")
    for case_name, kernel_name, ref_fn in REDUCTION_CASES:
        try:
            r = run_reduction_case(case_name, kernel_name, ref_fn)
        except Exception as e:
            r = {"case": case_name, "kernel": kernel_name, "status": "HARNESS_ERROR",
                 "detail": str(e)[:200]}
        results.append(r)
        print(f"  {r['status']:<32} {case_name:<32} {r.get('detail', '')}")

    print()
    print("--- Conv (the substantive stub-detection test) ---")
    for runner in [run_conv_2d_case, run_conv_1d_case, run_conv_3d_case,
                   run_conv_transpose_1d_case, run_conv_transpose_2d_case,
                   run_conv_transpose_3d_case]:
        try:
            r = runner()
        except Exception as e:
            r = {"case": runner.__name__, "kernel": "?", "status": "HARNESS_ERROR",
                 "detail": str(e)[:200]}
        results.append(r)
        print(f"  {r['status']:<32} {r.get('case', '?'):<32} {r.get('detail', '')}")

    print()
    print("=== Summary ===")
    by_status = {}
    for r in results:
        by_status.setdefault(r["status"], 0)
        by_status[r["status"]] += 1
    for status, count in sorted(by_status.items()):
        print(f"  {status:<32} {count}")

    stub_count = sum(1 for r in results if r["status"] == "STUB_DETECTED")
    print()
    print(f"STUBS DETECTED: {stub_count}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
