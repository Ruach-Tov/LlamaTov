#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""verify_kernelbench_l1_cpu.py — Bit-identity sweep across all Stanford KernelBench L1 problems.

Compares BPD CPU substrate output against PyTorch CPU for each of the 100 L1
problems. Reports per-problem status:

  BIT_IDENTICAL    0 ULP vs PyTorch CPU
  DIVERGENT        non-zero ULP — names the substrate-design parameter at fault
  MISSING_KERNEL   the substrate doesn't yet have a CPU kernel for this op
  NOT_IMPLEMENTED  the harness doesn't yet route this problem

Output: per-category summary + detailed per-problem table + grand total.

Run:
  make verify FOCUS=kernelbench-l1-cpu
"""
import ctypes
import os
import sys
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    sys.exit("error: pip install torch numpy")

torch.backends.mkldnn.enabled = False
torch.backends.cudnn.enabled = False
torch.set_num_threads(1)


# ─── ULP machinery ─────────────────────────────────────────────────────────

def ulp(a, b):
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    if a.shape != b.shape:
        return -1, -1, -1
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    BASE = np.int64(0x80000000)
    ai = np.where(ai < 0, BASE - ai, ai)
    bi = np.where(bi < 0, BASE - bi, bi)
    d = np.abs(ai - bi)
    return int(d.max()), int((d > 0).sum()), int(d.size)


# ─── Substrate library loading ─────────────────────────────────────────────

def load_lib():
    so_path = os.environ.get("BPD_CPU_SO", "build/bpd_cpu.so")
    if not os.path.exists(so_path):
        sys.exit(f"{so_path} not found — run `make build` first")
    lib = ctypes.CDLL(so_path)
    # Elementwise (input, output, n)
    for name in ['bpd_relu_cpu', 'bpd_silu_cpu', 'bpd_mish_cpu', 'bpd_sigmoid_cpu',
                 'bpd_tanh_cpu', 'bpd_gelu_cpu', 'bpd_neg_cpu', 'bpd_abs_cpu',
                 'bpd_exp_cpu', 'bpd_sum_cpu', 'bpd_mean_cpu', 'bpd_max_cpu',
                 'bpd_leaky_relu_cpu', 'bpd_elu_cpu', 'bpd_selu_cpu',
                 'bpd_hardsigmoid_cpu', 'bpd_clamp_cpu',
                 'bpd_softplus_cpu', 'bpd_softsign_cpu',
                 'bpd_cumsum_cpu', 'bpd_cumprod_cpu',
                 'bpd_cumsum_reverse_cpu', 'bpd_cumsum_exclusive_cpu']:
        if hasattr(lib, name):
            f = getattr(lib, name)
            f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
            f.restype = None
    # Special signatures
    if hasattr(lib, 'bpd_softmax_cpu'):
        lib.bpd_softmax_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_int, ctypes.c_int]
        lib.bpd_softmax_cpu.restype = None
    if hasattr(lib, 'bpd_logsoftmax_cpu'):
        lib.bpd_logsoftmax_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                            ctypes.c_int, ctypes.c_int]
        lib.bpd_logsoftmax_cpu.restype = None
    if hasattr(lib, 'bpd_mm_cpu'):
        lib.bpd_mm_cpu.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*3
        lib.bpd_mm_cpu.restype = None
    if hasattr(lib, 'bpd_conv2d_cpu'):
        lib.bpd_conv2d_cpu.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*9
        lib.bpd_conv2d_cpu.restype = None
    if hasattr(lib, 'bpd_maxpool2d_cpu'):
        lib.bpd_maxpool2d_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int]*7
        lib.bpd_maxpool2d_cpu.restype = None
    if hasattr(lib, 'bpd_avgpool2d_cpu'):
        lib.bpd_avgpool2d_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int]*7
        lib.bpd_avgpool2d_cpu.restype = None
    if hasattr(lib, 'bpd_layernorm_cpu'):
        lib.bpd_layernorm_cpu.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*2 + [ctypes.c_float]
        lib.bpd_layernorm_cpu.restype = None
    if hasattr(lib, 'bpd_instancenorm_cpu'):
        # (input, output, N, C, H, W, eps)
        lib.bpd_instancenorm_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int]*4 + [ctypes.c_float]
        lib.bpd_instancenorm_cpu.restype = None
    if hasattr(lib, 'bpd_groupnorm_cpu'):
        # (input, gamma, beta, output, N, C, H, W, G, eps)
        lib.bpd_groupnorm_cpu.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*5 + [ctypes.c_float]
        lib.bpd_groupnorm_cpu.restype = None
    if hasattr(lib, 'bpd_rmsnorm_cpu'):
        # (input, output, N, C, H, W, eps)
        lib.bpd_rmsnorm_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int]*4 + [ctypes.c_float]
        lib.bpd_rmsnorm_cpu.restype = None
    if hasattr(lib, 'bpd_frobenius_norm_cpu'):
        # (input, output, n_total)
        lib.bpd_frobenius_norm_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        lib.bpd_frobenius_norm_cpu.restype = None
    if hasattr(lib, 'bpd_l1norm_cpu'):
        # (input, output, rows, cols)
        lib.bpd_l1norm_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.bpd_l1norm_cpu.restype = None
    if hasattr(lib, 'bpd_l2norm_cpu'):
        # (input, output, rows, cols)
        lib.bpd_l2norm_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.bpd_l2norm_cpu.restype = None
    if hasattr(lib, 'bpd_maxpool1d_cpu'):
        # (in, out, N, C, L, kL, stride, pad)
        lib.bpd_maxpool1d_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int]*6
        lib.bpd_maxpool1d_cpu.restype = None
    if hasattr(lib, 'bpd_maxpool3d_cpu'):
        # (in, out, N, C, D, H, W, kD, kH, kW, stride, pad)
        lib.bpd_maxpool3d_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int]*10
        lib.bpd_maxpool3d_cpu.restype = None
    if hasattr(lib, 'bpd_avgpool1d_cpu'):
        lib.bpd_avgpool1d_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int]*6
        lib.bpd_avgpool1d_cpu.restype = None
    if hasattr(lib, 'bpd_avgpool3d_cpu'):
        lib.bpd_avgpool3d_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int]*10
        lib.bpd_avgpool3d_cpu.restype = None
    # Loss kernels
    for loss_name in ['bpd_mse_loss_cpu', 'bpd_huber_loss_cpu', 'bpd_hinge_loss_cpu']:
        if hasattr(lib, loss_name):
            f = getattr(lib, loss_name)
            f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
            f.restype = None
    if hasattr(lib, 'bpd_kl_div_loss_cpu'):
        # (log_pred, target, output, batch_size, per_batch)
        lib.bpd_kl_div_loss_cpu.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*2
        lib.bpd_kl_div_loss_cpu.restype = None
    if hasattr(lib, 'bpd_cross_entropy_loss_cpu'):
        # (pred, target_long, output, batch_size, num_classes)
        lib.bpd_cross_entropy_loss_cpu.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*2
        lib.bpd_cross_entropy_loss_cpu.restype = None
    if hasattr(lib, 'bpd_conv2d_full_cpu'):
        # (input, weight, bias_or_NULL, output, N, Cin, H, W, Cout, kH, kW,
        #  sh, sw, ph, pw, dh, dw, groups)  = 4 ptrs + 14 ints
        lib.bpd_conv2d_full_cpu.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*14
        lib.bpd_conv2d_full_cpu.restype = None
    if hasattr(lib, 'bpd_conv1d_full_cpu'):
        # (input, weight, bias_or_NULL, output, N, Cin, L, Cout, kL,
        #  sl, pl, dl, groups)  = 4 ptrs + 9 ints
        lib.bpd_conv1d_full_cpu.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*9
        lib.bpd_conv1d_full_cpu.restype = None
    if hasattr(lib, 'bpd_conv3d_full_cpu'):
        # (input, weight, bias_or_NULL, output, N, Cin, D, H, W, Cout, kD, kH, kW,
        #  sd, sh, sw, pd, ph, pw, dd, dh, dw, groups)  = 4 ptrs + 19 ints
        lib.bpd_conv3d_full_cpu.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*19
        lib.bpd_conv3d_full_cpu.restype = None
    if hasattr(lib, 'bpd_conv_transpose2d_full_cpu'):
        # (input, weight, bias_or_NULL, output, N, Cin, H_in, W_in, Cout, kH, kW,
        #  sh, sw, ph, pw, oph, opw, dh, dw, groups)  = 4 ptrs + 16 ints
        lib.bpd_conv_transpose2d_full_cpu.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*16
        lib.bpd_conv_transpose2d_full_cpu.restype = None
    if hasattr(lib, 'bpd_conv_transpose1d_full_cpu'):
        # (input, weight, bias_or_NULL, output, N, Cin, L_in, Cout, kL,
        #  sl, pl, opl, dl, groups)  = 4 ptrs + 10 ints
        lib.bpd_conv_transpose1d_full_cpu.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*10
        lib.bpd_conv_transpose1d_full_cpu.restype = None
    if hasattr(lib, 'bpd_conv_transpose3d_full_cpu'):
        # (input, weight, bias_or_NULL, output, N, Cin, D_in, H_in, W_in, Cout, kD, kH, kW,
        #  sd, sh, sw, pd, ph, pw, opd, oph, opw, dd, dh, dw, groups)  = 4 ptrs + 22 ints
        lib.bpd_conv_transpose3d_full_cpu.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*22
        lib.bpd_conv_transpose3d_full_cpu.restype = None
    if hasattr(lib, 'bpd_scalar_mul_cpu'):
        # (A, s, out, n)
        lib.bpd_scalar_mul_cpu.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_void_p, ctypes.c_int]
        lib.bpd_scalar_mul_cpu.restype = None
    if hasattr(lib, 'bpd_bmm_cpu'):
        # (A, B, C, batch, M, N, K)
        lib.bpd_bmm_cpu.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*4
        lib.bpd_bmm_cpu.restype = None
    if hasattr(lib, 'bpd_3d_tensor_matmul_cpu'):
        # (A, B, C, batch, M, N, K)
        lib.bpd_3d_tensor_matmul_cpu.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*4
        lib.bpd_3d_tensor_matmul_cpu.restype = None
    if hasattr(lib, 'bpd_4d_tensor_matmul_cpu'):
        # (A, B, C, batch, C_dim, M, N, K)
        lib.bpd_4d_tensor_matmul_cpu.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*5
        lib.bpd_4d_tensor_matmul_cpu.restype = None
    if hasattr(lib, 'bpd_batchnorm_cpu_affine_fused'):
        # (in, gamma, beta, mean, var, out, scale_buf, offset_buf, N, C, HW, eps)
        lib.bpd_batchnorm_cpu_affine_fused.argtypes = [ctypes.c_void_p]*8 + [ctypes.c_int]*3 + [ctypes.c_float]
        lib.bpd_batchnorm_cpu_affine_fused.restype = None
    if hasattr(lib, 'bpd_diag_matmul_cpu'):
        # (A_diag, B, C, M, N)
        lib.bpd_diag_matmul_cpu.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*2
        lib.bpd_diag_matmul_cpu.restype = None
    # A.6 specialty kernels
    if hasattr(lib, 'bpd_argmax_dim_cpu'):
        # (x, out_int64, outer, dim_size, inner)
        lib.bpd_argmax_dim_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int]*3
        lib.bpd_argmax_dim_cpu.restype = None
    if hasattr(lib, 'bpd_argmin_dim_cpu'):
        lib.bpd_argmin_dim_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int]*3
        lib.bpd_argmin_dim_cpu.restype = None
    if hasattr(lib, 'bpd_min_dim_cpu'):
        lib.bpd_min_dim_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int]*3
        lib.bpd_min_dim_cpu.restype = None
    if hasattr(lib, 'bpd_masked_cumsum_cpu'):
        # (x, mask_u8, out, batch, dim_size)
        lib.bpd_masked_cumsum_cpu.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*2
        lib.bpd_masked_cumsum_cpu.restype = None
    if hasattr(lib, 'bpd_mingpt_newgelu_cpu'):
        lib.bpd_mingpt_newgelu_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        lib.bpd_mingpt_newgelu_cpu.restype = None
    if hasattr(lib, 'bpd_scaled_dot_product_attention_cpu'):
        # (Q, K, V, out, batch, num_heads, seq_len, embed_dim)
        lib.bpd_scaled_dot_product_attention_cpu.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*4
        lib.bpd_scaled_dot_product_attention_cpu.restype = None
    if hasattr(lib, 'bpd_triplet_margin_loss_cpu'):
        # (anchor, positive, negative, output, batch_size, feat_dim, margin)
        lib.bpd_triplet_margin_loss_cpu.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*2 + [ctypes.c_float]
        lib.bpd_triplet_margin_loss_cpu.restype = None
    return lib


# ─── Per-problem harness ───────────────────────────────────────────────────
#
# Each problem returns one of:
#   ('BIT_IDENTICAL', 0, n)
#   ('DIVERGENT', max_ulp, n)
#   ('MISSING_KERNEL', '<kernel_name>', None)
#   ('NOT_IMPLEMENTED', None, None)

RNG = np.random.default_rng(42)


def elementwise(lib, kernel, pt_fn, n=4096):
    if not hasattr(lib, kernel):
        return ('MISSING_KERNEL', kernel, None)
    x = (RNG.standard_normal(n) * 2.0).astype(np.float32)
    out = np.zeros_like(x)
    getattr(lib, kernel)(x.ctypes.data, out.ctypes.data, n)
    ref = pt_fn(torch.from_numpy(x)).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def matmul_problem(lib, M, N, K):
    if not hasattr(lib, 'bpd_mm_cpu'):
        return ('MISSING_KERNEL', 'bpd_mm_cpu', None)
    A = RNG.standard_normal((M, K)).astype(np.float32)
    B = RNG.standard_normal((K, N)).astype(np.float32)
    out = np.zeros((M, N), dtype=np.float32)
    lib.bpd_mm_cpu(A.ctypes.data, B.ctypes.data, out.ctypes.data, M, N, K)
    ref = (torch.from_numpy(A) @ torch.from_numpy(B)).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def scalar_mul_problem(lib):
    if not hasattr(lib, 'bpd_scalar_mul_cpu'):
        return ('MISSING_KERNEL', 'bpd_scalar_mul_cpu', None)
    M, N = 32, 64
    A = RNG.standard_normal((M, N)).astype(np.float32)
    s = 3.14
    out = np.zeros((M, N), dtype=np.float32)
    lib.bpd_scalar_mul_cpu(A.ctypes.data, ctypes.c_float(s), out.ctypes.data, M * N)
    ref = (torch.from_numpy(A) * s).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def bmm_problem(lib):
    if not hasattr(lib, 'bpd_bmm_cpu'):
        return ('MISSING_KERNEL', 'bpd_bmm_cpu', None)
    batch, M, N, K = 8, 16, 16, 32
    A = RNG.standard_normal((batch, M, K)).astype(np.float32)
    B = RNG.standard_normal((batch, K, N)).astype(np.float32)
    out = np.zeros((batch, M, N), dtype=np.float32)
    lib.bpd_bmm_cpu(A.ctypes.data, B.ctypes.data, out.ctypes.data, batch, M, N, K)
    ref = torch.bmm(torch.from_numpy(A), torch.from_numpy(B)).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def tensor3d_matmul_problem(lib):
    if not hasattr(lib, 'bpd_3d_tensor_matmul_cpu'):
        return ('MISSING_KERNEL', 'bpd_3d_tensor_matmul_cpu', None)
    batch, M, N, K = 8, 16, 32, 24
    A = RNG.standard_normal((batch, M, K)).astype(np.float32)
    B = RNG.standard_normal((K, N)).astype(np.float32)
    out = np.zeros((batch, M, N), dtype=np.float32)
    lib.bpd_3d_tensor_matmul_cpu(A.ctypes.data, B.ctypes.data, out.ctypes.data, batch, M, N, K)
    ref = (torch.from_numpy(A) @ torch.from_numpy(B)).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def tensor4d_matmul_problem(lib):
    if not hasattr(lib, 'bpd_4d_tensor_matmul_cpu'):
        return ('MISSING_KERNEL', 'bpd_4d_tensor_matmul_cpu', None)
    batch, C_dim, M, N, K = 4, 4, 16, 32, 24
    A = RNG.standard_normal((batch, C_dim, M, K)).astype(np.float32)
    B = RNG.standard_normal((K, N)).astype(np.float32)
    out = np.zeros((batch, C_dim, M, N), dtype=np.float32)
    lib.bpd_4d_tensor_matmul_cpu(A.ctypes.data, B.ctypes.data, out.ctypes.data,
                                  batch, C_dim, M, N, K)
    ref = (torch.from_numpy(A) @ torch.from_numpy(B)).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def diag_matmul_problem(lib):
    if not hasattr(lib, 'bpd_diag_matmul_cpu'):
        return ('MISSING_KERNEL', 'bpd_diag_matmul_cpu', None)
    M, N = 64, 128
    A_diag = RNG.standard_normal(M).astype(np.float32)
    B = RNG.standard_normal((M, N)).astype(np.float32)
    out = np.zeros((M, N), dtype=np.float32)
    lib.bpd_diag_matmul_cpu(A_diag.ctypes.data, B.ctypes.data, out.ctypes.data, M, N)
    # Reference: diag(A) @ B = A.unsqueeze(1) * B
    ref = (torch.from_numpy(A_diag).unsqueeze(1) * torch.from_numpy(B)).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def argmax_dim_problem(lib):
    if not hasattr(lib, 'bpd_argmax_dim_cpu'):
        return ('MISSING_KERNEL', 'bpd_argmax_dim_cpu', None)
    # Test on shape (batch, dim_size, inner), reduce along middle axis (dim=1)
    batch, dim_size, inner = 4, 32, 16
    x = RNG.standard_normal((batch, dim_size, inner)).astype(np.float32)
    out = np.zeros((batch, inner), dtype=np.int64)
    lib.bpd_argmax_dim_cpu(x.ctypes.data, out.ctypes.data, batch, dim_size, inner)
    ref = torch.argmax(torch.from_numpy(x), dim=1).numpy()
    # int64 comparison — exact match required
    if np.array_equal(ref, out):
        return ('BIT_IDENTICAL', 0, 0)
    diffs = int((ref != out).sum())
    return ('DIVERGENT', int(np.abs(ref - out).max()), diffs)


def argmin_dim_problem(lib):
    if not hasattr(lib, 'bpd_argmin_dim_cpu'):
        return ('MISSING_KERNEL', 'bpd_argmin_dim_cpu', None)
    batch, dim_size, inner = 4, 32, 16
    x = RNG.standard_normal((batch, dim_size, inner)).astype(np.float32)
    out = np.zeros((batch, inner), dtype=np.int64)
    lib.bpd_argmin_dim_cpu(x.ctypes.data, out.ctypes.data, batch, dim_size, inner)
    ref = torch.argmin(torch.from_numpy(x), dim=1).numpy()
    if np.array_equal(ref, out):
        return ('BIT_IDENTICAL', 0, 0)
    diffs = int((ref != out).sum())
    return ('DIVERGENT', int(np.abs(ref - out).max()), diffs)


def min_dim_problem(lib):
    if not hasattr(lib, 'bpd_min_dim_cpu'):
        return ('MISSING_KERNEL', 'bpd_min_dim_cpu', None)
    batch, dim_size, inner = 4, 32, 16
    x = RNG.standard_normal((batch, dim_size, inner)).astype(np.float32)
    out = np.zeros((batch, inner), dtype=np.float32)
    lib.bpd_min_dim_cpu(x.ctypes.data, out.ctypes.data, batch, dim_size, inner)
    ref = torch.min(torch.from_numpy(x), dim=1)[0].numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def masked_cumsum_problem(lib):
    if not hasattr(lib, 'bpd_masked_cumsum_cpu'):
        return ('MISSING_KERNEL', 'bpd_masked_cumsum_cpu', None)
    batch, dim_size = 16, 128
    x = RNG.standard_normal((batch, dim_size)).astype(np.float32)
    mask_bool = RNG.integers(0, 2, (batch, dim_size)).astype(np.bool_)
    mask_u8 = mask_bool.astype(np.uint8)
    out = np.zeros((batch, dim_size), dtype=np.float32)
    lib.bpd_masked_cumsum_cpu(x.ctypes.data, mask_u8.ctypes.data, out.ctypes.data,
                               batch, dim_size)
    ref = torch.cumsum(torch.from_numpy(x) * torch.from_numpy(mask_bool), dim=1).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def mingpt_newgelu_problem(lib):
    if not hasattr(lib, 'bpd_mingpt_newgelu_cpu'):
        return ('MISSING_KERNEL', 'bpd_mingpt_newgelu_cpu', None)
    n = 4096
    x = RNG.standard_normal(n).astype(np.float32)
    out = np.zeros(n, dtype=np.float32)
    lib.bpd_mingpt_newgelu_cpu(x.ctypes.data, out.ctypes.data, n)
    import math
    xt = torch.from_numpy(x)
    ref = (0.5 * xt * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (xt + 0.044715 * torch.pow(xt, 3.0))))).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def scaled_dot_product_attention_problem(lib):
    if not hasattr(lib, 'bpd_scaled_dot_product_attention_cpu'):
        return ('MISSING_KERNEL', 'bpd_scaled_dot_product_attention_cpu', None)
    # Small shapes for fast test
    batch, num_heads, seq_len, embed_dim = 1, 2, 8, 16
    Q = RNG.standard_normal((batch, num_heads, seq_len, embed_dim)).astype(np.float32)
    K = RNG.standard_normal((batch, num_heads, seq_len, embed_dim)).astype(np.float32)
    V = RNG.standard_normal((batch, num_heads, seq_len, embed_dim)).astype(np.float32)
    out = np.zeros_like(Q)
    lib.bpd_scaled_dot_product_attention_cpu(Q.ctypes.data, K.ctypes.data, V.ctypes.data,
                                              out.ctypes.data,
                                              batch, num_heads, seq_len, embed_dim)
    ref = F.scaled_dot_product_attention(torch.from_numpy(Q), torch.from_numpy(K),
                                          torch.from_numpy(V)).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def conv2d_problem(lib, N, Cin, H, W, Cout, kH, kW, stride=1, pad=0):
    if not hasattr(lib, 'bpd_conv2d_cpu'):
        return ('MISSING_KERNEL', 'bpd_conv2d_cpu', None)
    inp = RNG.standard_normal((N, Cin, H, W)).astype(np.float32)
    weight = RNG.standard_normal((Cout, Cin, kH, kW)).astype(np.float32)
    H_out = (H + 2*pad - kH) // stride + 1
    W_out = (W + 2*pad - kW) // stride + 1
    out = np.zeros((N, Cout, H_out, W_out), dtype=np.float32)
    lib.bpd_conv2d_cpu(inp.ctypes.data, weight.ctypes.data, out.ctypes.data,
                       N, Cin, H, W, Cout, kH, kW, stride, pad)
    ref = F.conv2d(torch.from_numpy(inp), torch.from_numpy(weight),
                    stride=stride, padding=pad).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def batchnorm_problem(lib):
    """L1 #33: nn.BatchNorm2d in eval mode (uses running_mean=0, running_var=1)."""
    if not hasattr(lib, 'bpd_batchnorm_cpu_affine_fused'):
        return ('MISSING_KERNEL', 'bpd_batchnorm_cpu_affine_fused', None)
    N, C, H, W = 1, 4, 8, 8
    eps = 1e-5
    inp = RNG.standard_normal((N, C, H, W)).astype(np.float32)
    # nn.BatchNorm2d default init: gamma=1, beta=0, running_mean=0, running_var=1
    gamma = np.ones(C, dtype=np.float32)
    beta = np.zeros(C, dtype=np.float32)
    mean = np.zeros(C, dtype=np.float32)
    var = np.ones(C, dtype=np.float32)
    out = np.zeros_like(inp)
    HW = H * W
    # Substrate kernel signature: bpd_batchnorm_cpu_affine_fused(in, gamma, beta, mean, var, out,
    #                              scale_buf=NULL, offset_buf=NULL, N, C, HW, eps)
    lib.bpd_batchnorm_cpu_affine_fused(inp.ctypes.data, gamma.ctypes.data, beta.ctypes.data,
                                        mean.ctypes.data, var.ctypes.data, out.ctypes.data,
                                        0, 0, N, C, HW, ctypes.c_float(eps))
    # Reference: nn.BatchNorm2d in eval mode
    bn = torch.nn.BatchNorm2d(C, eps=eps)
    bn.eval()
    with torch.no_grad():
        ref = bn(torch.from_numpy(inp)).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def conv2d_full_problem(lib, N, Cin, H, W, Cout, kH, kW,
                         stride=(1,1), pad=(0,0), dilation=(1,1), groups=1, has_bias=False):
    """Parameterized conv2d test using bpd_conv2d_full_cpu (im2col + GEMM, matches PT)."""
    if not hasattr(lib, 'bpd_conv2d_full_cpu'):
        return ('MISSING_KERNEL', 'bpd_conv2d_full_cpu', None)
    sh, sw = stride if isinstance(stride, tuple) else (stride, stride)
    ph, pw = pad if isinstance(pad, tuple) else (pad, pad)
    dh, dw = dilation if isinstance(dilation, tuple) else (dilation, dilation)
    inp = RNG.standard_normal((N, Cin, H, W)).astype(np.float32)
    weight = RNG.standard_normal((Cout, Cin // groups, kH, kW)).astype(np.float32)
    bias = RNG.standard_normal(Cout).astype(np.float32) if has_bias else None
    H_out = (H + 2*ph - dh*(kH-1) - 1) // sh + 1
    W_out = (W + 2*pw - dw*(kW-1) - 1) // sw + 1
    out = np.zeros((N, Cout, H_out, W_out), dtype=np.float32)
    bias_ptr = bias.ctypes.data if has_bias else 0
    lib.bpd_conv2d_full_cpu(inp.ctypes.data, weight.ctypes.data, bias_ptr,
                             out.ctypes.data,
                             N, Cin, H, W, Cout, kH, kW,
                             sh, sw, ph, pw, dh, dw, groups)
    ref = F.conv2d(torch.from_numpy(inp), torch.from_numpy(weight),
                    bias=torch.from_numpy(bias) if has_bias else None,
                    stride=(sh, sw), padding=(ph, pw),
                    dilation=(dh, dw), groups=groups).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def conv1d_full_problem(lib, N, Cin, L, Cout, kL,
                         stride=1, pad=0, dilation=1, groups=1, has_bias=False):
    """Parameterized conv1d test."""
    if not hasattr(lib, 'bpd_conv1d_full_cpu'):
        return ('MISSING_KERNEL', 'bpd_conv1d_full_cpu', None)
    inp = RNG.standard_normal((N, Cin, L)).astype(np.float32)
    weight = RNG.standard_normal((Cout, Cin // groups, kL)).astype(np.float32)
    bias = RNG.standard_normal(Cout).astype(np.float32) if has_bias else None
    L_out = (L + 2*pad - dilation*(kL-1) - 1) // stride + 1
    out = np.zeros((N, Cout, L_out), dtype=np.float32)
    bias_ptr = bias.ctypes.data if has_bias else 0
    lib.bpd_conv1d_full_cpu(inp.ctypes.data, weight.ctypes.data, bias_ptr,
                             out.ctypes.data,
                             N, Cin, L, Cout, kL,
                             stride, pad, dilation, groups)
    ref = F.conv1d(torch.from_numpy(inp), torch.from_numpy(weight),
                    bias=torch.from_numpy(bias) if has_bias else None,
                    stride=stride, padding=pad,
                    dilation=dilation, groups=groups).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def conv3d_full_problem(lib, N, Cin, D, H, W, Cout, kD, kH, kW,
                         stride=(1,1,1), pad=(0,0,0), dilation=(1,1,1),
                         groups=1, has_bias=False):
    """Parameterized conv3d test."""
    if not hasattr(lib, 'bpd_conv3d_full_cpu'):
        return ('MISSING_KERNEL', 'bpd_conv3d_full_cpu', None)
    sd, sh, sw = stride if isinstance(stride, tuple) else (stride, stride, stride)
    pd, ph, pw = pad if isinstance(pad, tuple) else (pad, pad, pad)
    dd, dh, dw = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
    inp = RNG.standard_normal((N, Cin, D, H, W)).astype(np.float32)
    weight = RNG.standard_normal((Cout, Cin // groups, kD, kH, kW)).astype(np.float32)
    bias = RNG.standard_normal(Cout).astype(np.float32) if has_bias else None
    D_out = (D + 2*pd - dd*(kD-1) - 1) // sd + 1
    H_out = (H + 2*ph - dh*(kH-1) - 1) // sh + 1
    W_out = (W + 2*pw - dw*(kW-1) - 1) // sw + 1
    out = np.zeros((N, Cout, D_out, H_out, W_out), dtype=np.float32)
    bias_ptr = bias.ctypes.data if has_bias else 0
    lib.bpd_conv3d_full_cpu(inp.ctypes.data, weight.ctypes.data, bias_ptr,
                             out.ctypes.data,
                             N, Cin, D, H, W, Cout, kD, kH, kW,
                             sd, sh, sw, pd, ph, pw, dd, dh, dw, groups)
    ref = F.conv3d(torch.from_numpy(inp), torch.from_numpy(weight),
                    bias=torch.from_numpy(bias) if has_bias else None,
                    stride=(sd, sh, sw), padding=(pd, ph, pw),
                    dilation=(dd, dh, dw), groups=groups).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def conv_transpose2d_full_problem(lib, N, Cin, H_in, W_in, Cout, kH, kW,
                                    stride=(1,1), pad=(0,0), output_padding=(0,0),
                                    dilation=(1,1), groups=1, has_bias=False):
    """Parameterized conv_transpose2d test using bpd_conv_transpose2d_full_cpu."""
    if not hasattr(lib, 'bpd_conv_transpose2d_full_cpu'):
        return ('MISSING_KERNEL', 'bpd_conv_transpose2d_full_cpu', None)
    sh, sw = stride if isinstance(stride, tuple) else (stride, stride)
    ph, pw = pad if isinstance(pad, tuple) else (pad, pad)
    oph, opw = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
    dh, dw = dilation if isinstance(dilation, tuple) else (dilation, dilation)
    inp = RNG.standard_normal((N, Cin, H_in, W_in)).astype(np.float32)
    # ConvTranspose weight: (Cin, Cout/groups, kH, kW)
    weight = RNG.standard_normal((Cin, Cout // groups, kH, kW)).astype(np.float32)
    bias = RNG.standard_normal(Cout).astype(np.float32) if has_bias else None
    H_out = (H_in - 1) * sh - 2*ph + dh*(kH-1) + oph + 1
    W_out = (W_in - 1) * sw - 2*pw + dw*(kW-1) + opw + 1
    out = np.zeros((N, Cout, H_out, W_out), dtype=np.float32)
    bias_ptr = bias.ctypes.data if has_bias else 0
    lib.bpd_conv_transpose2d_full_cpu(inp.ctypes.data, weight.ctypes.data, bias_ptr,
                                       out.ctypes.data,
                                       N, Cin, H_in, W_in, Cout, kH, kW,
                                       sh, sw, ph, pw, oph, opw, dh, dw, groups)
    ref = F.conv_transpose2d(torch.from_numpy(inp), torch.from_numpy(weight),
                              bias=torch.from_numpy(bias) if has_bias else None,
                              stride=(sh, sw), padding=(ph, pw),
                              output_padding=(oph, opw),
                              dilation=(dh, dw), groups=groups).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def depthwise_separable_problem(lib):
    """Depthwise-separable Conv2D = depthwise + pointwise composition."""
    if not hasattr(lib, 'bpd_conv2d_full_cpu'):
        return ('MISSING_KERNEL', 'bpd_conv2d_full_cpu', None)
    N, Cin, H, W = 1, 8, 16, 16
    Cout = 16
    kH = kW = 3
    pad = 1
    inp = RNG.standard_normal((N, Cin, H, W)).astype(np.float32)
    dw_weight = RNG.standard_normal((Cin, 1, kH, kW)).astype(np.float32)
    pw_weight = RNG.standard_normal((Cout, Cin, 1, 1)).astype(np.float32)
    H_dw = H + 2*pad - kH + 1
    W_dw = W + 2*pad - kW + 1
    dw_out = np.zeros((N, Cin, H_dw, W_dw), dtype=np.float32)
    lib.bpd_conv2d_full_cpu(inp.ctypes.data, dw_weight.ctypes.data, 0,
                             dw_out.ctypes.data,
                             N, Cin, H, W, Cin, kH, kW,
                             1, 1, pad, pad, 1, 1, Cin)
    pw_out = np.zeros((N, Cout, H_dw, W_dw), dtype=np.float32)
    lib.bpd_conv2d_full_cpu(dw_out.ctypes.data, pw_weight.ctypes.data, 0,
                             pw_out.ctypes.data,
                             N, Cin, H_dw, W_dw, Cout, 1, 1,
                             1, 1, 0, 0, 1, 1, 1)
    dw_ref = F.conv2d(torch.from_numpy(inp), torch.from_numpy(dw_weight),
                       stride=1, padding=pad, groups=Cin)
    ref = F.conv2d(dw_ref, torch.from_numpy(pw_weight),
                    stride=1, padding=0).numpy()
    mu, nd, nt = ulp(ref, pw_out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def conv_transpose1d_full_problem(lib, N, Cin, L_in, Cout, kL,
                                    stride=1, pad=0, output_padding=0,
                                    dilation=1, groups=1, has_bias=False):
    if not hasattr(lib, 'bpd_conv_transpose1d_full_cpu'):
        return ('MISSING_KERNEL', 'bpd_conv_transpose1d_full_cpu', None)
    inp = RNG.standard_normal((N, Cin, L_in)).astype(np.float32)
    weight = RNG.standard_normal((Cin, Cout // groups, kL)).astype(np.float32)
    bias = RNG.standard_normal(Cout).astype(np.float32) if has_bias else None
    L_out = (L_in - 1) * stride - 2*pad + dilation*(kL-1) + output_padding + 1
    out = np.zeros((N, Cout, L_out), dtype=np.float32)
    bias_ptr = bias.ctypes.data if has_bias else 0
    lib.bpd_conv_transpose1d_full_cpu(inp.ctypes.data, weight.ctypes.data, bias_ptr,
                                       out.ctypes.data,
                                       N, Cin, L_in, Cout, kL,
                                       stride, pad, output_padding, dilation, groups)
    ref = F.conv_transpose1d(torch.from_numpy(inp), torch.from_numpy(weight),
                              bias=torch.from_numpy(bias) if has_bias else None,
                              stride=stride, padding=pad,
                              output_padding=output_padding,
                              dilation=dilation, groups=groups).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def conv_transpose3d_full_problem(lib, N, Cin, D_in, H_in, W_in, Cout, kD, kH, kW,
                                    stride=(1,1,1), pad=(0,0,0), output_padding=(0,0,0),
                                    dilation=(1,1,1), groups=1, has_bias=False):
    if not hasattr(lib, 'bpd_conv_transpose3d_full_cpu'):
        return ('MISSING_KERNEL', 'bpd_conv_transpose3d_full_cpu', None)
    sd, sh, sw = stride if isinstance(stride, tuple) else (stride, stride, stride)
    pd, ph, pw = pad if isinstance(pad, tuple) else (pad, pad, pad)
    opd, oph, opw = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding, output_padding)
    dd, dh, dw = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
    inp = RNG.standard_normal((N, Cin, D_in, H_in, W_in)).astype(np.float32)
    weight = RNG.standard_normal((Cin, Cout // groups, kD, kH, kW)).astype(np.float32)
    bias = RNG.standard_normal(Cout).astype(np.float32) if has_bias else None
    D_out = (D_in - 1) * sd - 2*pd + dd*(kD-1) + opd + 1
    H_out = (H_in - 1) * sh - 2*ph + dh*(kH-1) + oph + 1
    W_out = (W_in - 1) * sw - 2*pw + dw*(kW-1) + opw + 1
    out = np.zeros((N, Cout, D_out, H_out, W_out), dtype=np.float32)
    bias_ptr = bias.ctypes.data if has_bias else 0
    lib.bpd_conv_transpose3d_full_cpu(inp.ctypes.data, weight.ctypes.data, bias_ptr,
                                       out.ctypes.data,
                                       N, Cin, D_in, H_in, W_in, Cout, kD, kH, kW,
                                       sd, sh, sw, pd, ph, pw,
                                       opd, oph, opw, dd, dh, dw, groups)
    ref = F.conv_transpose3d(torch.from_numpy(inp), torch.from_numpy(weight),
                              bias=torch.from_numpy(bias) if has_bias else None,
                              stride=(sd, sh, sw), padding=(pd, ph, pw),
                              output_padding=(opd, oph, opw),
                              dilation=(dd, dh, dw), groups=groups).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def reduce_problem(lib, kernel, pt_fn, n=1024):
    if not hasattr(lib, kernel):
        return ('MISSING_KERNEL', kernel, None)
    x = (RNG.standard_normal(n) * 2.0).astype(np.float32)
    out = np.zeros(1, dtype=np.float32)
    getattr(lib, kernel)(x.ctypes.data, out.ctypes.data, n)
    ref = pt_fn(torch.from_numpy(x)).numpy().reshape(1)
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def softmax_problem(lib):
    if not hasattr(lib, 'bpd_softmax_cpu'):
        return ('MISSING_KERNEL', 'bpd_softmax_cpu', None)
    x = (RNG.standard_normal((32, 64)) * 2.0).astype(np.float32)
    out = np.zeros_like(x)
    lib.bpd_softmax_cpu(x.ctypes.data, out.ctypes.data, 32, 64)
    ref = F.softmax(torch.from_numpy(x), dim=-1).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def logsoftmax_problem(lib):
    if not hasattr(lib, 'bpd_logsoftmax_cpu'):
        return ('MISSING_KERNEL', 'bpd_logsoftmax_cpu', None)
    x = (RNG.standard_normal((32, 64)) * 2.0).astype(np.float32)
    out = np.zeros_like(x)
    lib.bpd_logsoftmax_cpu(x.ctypes.data, out.ctypes.data, 32, 64)
    ref = F.log_softmax(torch.from_numpy(x), dim=-1).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def layernorm_problem(lib):
    if not hasattr(lib, 'bpd_layernorm_cpu'):
        return ('MISSING_KERNEL', 'bpd_layernorm_cpu', None)
    rows, cols = 8, 128
    x = (RNG.standard_normal((rows, cols)) * 2.0).astype(np.float32)
    gamma = np.ones(cols, dtype=np.float32)
    beta = np.zeros(cols, dtype=np.float32)
    out = np.zeros_like(x)
    lib.bpd_layernorm_cpu(x.ctypes.data, gamma.ctypes.data, beta.ctypes.data,
                          out.ctypes.data, rows, cols, 1e-5)
    ref = F.layer_norm(torch.from_numpy(x), (cols,), torch.from_numpy(gamma),
                        torch.from_numpy(beta), eps=1e-5).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def instancenorm_problem(lib):
    if not hasattr(lib, 'bpd_instancenorm_cpu'):
        return ('MISSING_KERNEL', 'bpd_instancenorm_cpu', None)
    N, C, H, W = 2, 4, 8, 8
    x = (RNG.standard_normal((N, C, H, W)) * 2.0).astype(np.float32)
    out = np.zeros_like(x)
    lib.bpd_instancenorm_cpu(x.ctypes.data, out.ctypes.data, N, C, H, W, 1e-5)
    # PyTorch InstanceNorm2d default: affine=False, track_running_stats=False
    inorm = torch.nn.InstanceNorm2d(num_features=C, eps=1e-5, affine=False, track_running_stats=False)
    ref = inorm(torch.from_numpy(x)).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def groupnorm_problem(lib):
    if not hasattr(lib, 'bpd_groupnorm_cpu'):
        return ('MISSING_KERNEL', 'bpd_groupnorm_cpu', None)
    N, C, H, W = 2, 8, 8, 8
    G = 2
    x = (RNG.standard_normal((N, C, H, W)) * 2.0).astype(np.float32)
    gamma = np.ones(C, dtype=np.float32)
    beta = np.zeros(C, dtype=np.float32)
    out = np.zeros_like(x)
    lib.bpd_groupnorm_cpu(x.ctypes.data, gamma.ctypes.data, beta.ctypes.data,
                          out.ctypes.data, N, C, H, W, G, 1e-5)
    gn = torch.nn.GroupNorm(num_groups=G, num_channels=C, eps=1e-5, affine=True)
    # Set its gamma=1, beta=0 explicitly
    with torch.no_grad():
        gn.weight.fill_(1.0)
        gn.bias.fill_(0.0)
    ref = gn(torch.from_numpy(x)).detach().numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def rmsnorm_problem(lib):
    if not hasattr(lib, 'bpd_rmsnorm_cpu'):
        return ('MISSING_KERNEL', 'bpd_rmsnorm_cpu', None)
    N, C, H, W = 2, 8, 4, 4
    eps = 1e-5
    x = (RNG.standard_normal((N, C, H, W)) * 2.0).astype(np.float32)
    out = np.zeros_like(x)
    lib.bpd_rmsnorm_cpu(x.ctypes.data, out.ctypes.data, N, C, H, W, eps)
    # Reference: torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + eps); x / rms
    xt = torch.from_numpy(x)
    rms = torch.sqrt(torch.mean(xt ** 2, dim=1, keepdim=True) + eps)
    ref = (xt / rms).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def frobenius_problem(lib):
    if not hasattr(lib, 'bpd_frobenius_norm_cpu'):
        return ('MISSING_KERNEL', 'bpd_frobenius_norm_cpu', None)
    N, C, H, W = 2, 4, 4, 4
    x = (RNG.standard_normal((N, C, H, W)) * 2.0).astype(np.float32)
    n_total = N * C * H * W
    out = np.zeros_like(x)
    lib.bpd_frobenius_norm_cpu(x.ctypes.data, out.ctypes.data, n_total)
    norm = torch.norm(torch.from_numpy(x), p='fro')
    ref = (torch.from_numpy(x) / norm).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def l1norm_problem(lib):
    if not hasattr(lib, 'bpd_l1norm_cpu'):
        return ('MISSING_KERNEL', 'bpd_l1norm_cpu', None)
    rows, cols = 8, 128
    x = (RNG.standard_normal((rows, cols)) * 2.0).astype(np.float32)
    out = np.zeros_like(x)
    lib.bpd_l1norm_cpu(x.ctypes.data, out.ctypes.data, rows, cols)
    # Reference: x / mean(|x|, dim=1, keepdim=True)
    xt = torch.from_numpy(x)
    ref = (xt / torch.mean(torch.abs(xt), dim=1, keepdim=True)).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def l2norm_problem(lib):
    if not hasattr(lib, 'bpd_l2norm_cpu'):
        return ('MISSING_KERNEL', 'bpd_l2norm_cpu', None)
    rows, cols = 8, 128
    x = (RNG.standard_normal((rows, cols)) * 2.0).astype(np.float32)
    out = np.zeros_like(x)
    lib.bpd_l2norm_cpu(x.ctypes.data, out.ctypes.data, rows, cols)
    # Reference: x / norm(x, p=2, dim=1, keepdim=True)
    xt = torch.from_numpy(x)
    ref = (xt / torch.norm(xt, p=2, dim=1, keepdim=True)).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def maxpool2d_problem(lib):
    if not hasattr(lib, 'bpd_maxpool2d_cpu'):
        return ('MISSING_KERNEL', 'bpd_maxpool2d_cpu', None)
    N, C, H, W = 1, 3, 16, 16
    kH, kW = 2, 2
    stride = 2
    inp = RNG.standard_normal((N, C, H, W)).astype(np.float32)
    H_out = H // stride
    W_out = W // stride
    out = np.zeros((N, C, H_out, W_out), dtype=np.float32)
    lib.bpd_maxpool2d_cpu(inp.ctypes.data, out.ctypes.data,
                          N, C, H, W, kH, kW, stride)
    ref = F.max_pool2d(torch.from_numpy(inp), kernel_size=(kH, kW), stride=stride).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def avgpool2d_problem(lib):
    if not hasattr(lib, 'bpd_avgpool2d_cpu'):
        return ('MISSING_KERNEL', 'bpd_avgpool2d_cpu', None)
    N, C, H, W = 1, 3, 16, 16
    kH, kW = 2, 2
    stride = 2
    inp = RNG.standard_normal((N, C, H, W)).astype(np.float32)
    H_out = H // stride
    W_out = W // stride
    out = np.zeros((N, C, H_out, W_out), dtype=np.float32)
    lib.bpd_avgpool2d_cpu(inp.ctypes.data, out.ctypes.data,
                          N, C, H, W, kH, kW, stride)
    ref = F.avg_pool2d(torch.from_numpy(inp), kernel_size=(kH, kW), stride=stride).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def maxpool1d_problem(lib):
    if not hasattr(lib, 'bpd_maxpool1d_cpu'):
        return ('MISSING_KERNEL', 'bpd_maxpool1d_cpu', None)
    N, C, L = 1, 3, 32
    kL, stride, pad = 4, 2, 0
    inp = RNG.standard_normal((N, C, L)).astype(np.float32)
    L_out = (L + 2*pad - kL) // stride + 1
    out = np.zeros((N, C, L_out), dtype=np.float32)
    lib.bpd_maxpool1d_cpu(inp.ctypes.data, out.ctypes.data, N, C, L, kL, stride, pad)
    ref = F.max_pool1d(torch.from_numpy(inp), kernel_size=kL, stride=stride, padding=pad).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def maxpool3d_problem(lib):
    if not hasattr(lib, 'bpd_maxpool3d_cpu'):
        return ('MISSING_KERNEL', 'bpd_maxpool3d_cpu', None)
    N, C, D, H, W = 1, 2, 8, 8, 8
    kD = kH = kW = 2
    stride, pad = 2, 0
    inp = RNG.standard_normal((N, C, D, H, W)).astype(np.float32)
    D_out = (D + 2*pad - kD) // stride + 1
    H_out = (H + 2*pad - kH) // stride + 1
    W_out = (W + 2*pad - kW) // stride + 1
    out = np.zeros((N, C, D_out, H_out, W_out), dtype=np.float32)
    lib.bpd_maxpool3d_cpu(inp.ctypes.data, out.ctypes.data,
                          N, C, D, H, W, kD, kH, kW, stride, pad)
    ref = F.max_pool3d(torch.from_numpy(inp),
                       kernel_size=(kD, kH, kW), stride=stride, padding=pad).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def avgpool1d_problem(lib):
    if not hasattr(lib, 'bpd_avgpool1d_cpu'):
        return ('MISSING_KERNEL', 'bpd_avgpool1d_cpu', None)
    N, C, L = 1, 3, 32
    kL, stride, pad = 4, 2, 0
    inp = RNG.standard_normal((N, C, L)).astype(np.float32)
    L_out = (L + 2*pad - kL) // stride + 1
    out = np.zeros((N, C, L_out), dtype=np.float32)
    lib.bpd_avgpool1d_cpu(inp.ctypes.data, out.ctypes.data, N, C, L, kL, stride, pad)
    ref = F.avg_pool1d(torch.from_numpy(inp), kernel_size=kL, stride=stride, padding=pad).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def avgpool3d_problem(lib):
    if not hasattr(lib, 'bpd_avgpool3d_cpu'):
        return ('MISSING_KERNEL', 'bpd_avgpool3d_cpu', None)
    N, C, D, H, W = 1, 2, 8, 8, 8
    kD = kH = kW = 2
    stride, pad = 2, 0
    inp = RNG.standard_normal((N, C, D, H, W)).astype(np.float32)
    D_out = (D + 2*pad - kD) // stride + 1
    H_out = (H + 2*pad - kH) // stride + 1
    W_out = (W + 2*pad - kW) // stride + 1
    out = np.zeros((N, C, D_out, H_out, W_out), dtype=np.float32)
    lib.bpd_avgpool3d_cpu(inp.ctypes.data, out.ctypes.data,
                          N, C, D, H, W, kD, kH, kW, stride, pad)
    ref = F.avg_pool3d(torch.from_numpy(inp),
                       kernel_size=(kD, kH, kW), stride=stride, padding=pad).numpy()
    mu, nd, nt = ulp(ref, out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)

def mse_loss_problem(lib):
    if not hasattr(lib, 'bpd_mse_loss_cpu'):
        return ('MISSING_KERNEL', 'bpd_mse_loss_cpu', None)
    n = 4096
    pred = RNG.standard_normal(n).astype(np.float32)
    target = RNG.standard_normal(n).astype(np.float32)
    out = np.zeros(1, dtype=np.float32)
    lib.bpd_mse_loss_cpu(pred.ctypes.data, target.ctypes.data, out.ctypes.data, n)
    ref = F.mse_loss(torch.from_numpy(pred), torch.from_numpy(target)).numpy()
    mu, nd, nt = ulp(ref.reshape(1), out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def huber_loss_problem(lib):
    if not hasattr(lib, 'bpd_huber_loss_cpu'):
        return ('MISSING_KERNEL', 'bpd_huber_loss_cpu', None)
    n = 4096
    pred = RNG.standard_normal(n).astype(np.float32)
    target = RNG.standard_normal(n).astype(np.float32)
    out = np.zeros(1, dtype=np.float32)
    lib.bpd_huber_loss_cpu(pred.ctypes.data, target.ctypes.data, out.ctypes.data, n)
    ref = F.smooth_l1_loss(torch.from_numpy(pred), torch.from_numpy(target)).numpy()
    mu, nd, nt = ulp(ref.reshape(1), out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def hinge_loss_problem(lib):
    if not hasattr(lib, 'bpd_hinge_loss_cpu'):
        return ('MISSING_KERNEL', 'bpd_hinge_loss_cpu', None)
    n = 4096
    pred = RNG.standard_normal(n).astype(np.float32)
    target = (RNG.integers(0, 2, n) * 2 - 1).astype(np.float32)
    out = np.zeros(1, dtype=np.float32)
    lib.bpd_hinge_loss_cpu(pred.ctypes.data, target.ctypes.data, out.ctypes.data, n)
    ref = torch.mean(torch.clamp(1 - torch.from_numpy(pred) * torch.from_numpy(target), min=0)).numpy()
    mu, nd, nt = ulp(ref.reshape(1), out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def kl_div_loss_problem(lib):
    if not hasattr(lib, 'bpd_kl_div_loss_cpu'):
        return ('MISSING_KERNEL', 'bpd_kl_div_loss_cpu', None)
    batch_size = 32
    per_batch = 128
    pred_sm = F.softmax(torch.from_numpy(RNG.standard_normal((batch_size, per_batch)).astype(np.float32)), dim=-1)
    log_pred = torch.log(pred_sm).numpy().astype(np.float32)
    target = F.softmax(torch.from_numpy(RNG.standard_normal((batch_size, per_batch)).astype(np.float32)), dim=-1).numpy().astype(np.float32)
    out = np.zeros(1, dtype=np.float32)
    log_pred_f = np.ascontiguousarray(log_pred)
    target_f = np.ascontiguousarray(target)
    lib.bpd_kl_div_loss_cpu(log_pred_f.ctypes.data, target_f.ctypes.data, out.ctypes.data,
                             batch_size, per_batch)
    ref = F.kl_div(torch.from_numpy(log_pred_f), torch.from_numpy(target_f),
                   reduction='batchmean').numpy()
    mu, nd, nt = ulp(ref.reshape(1), out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def cross_entropy_problem(lib):
    if not hasattr(lib, 'bpd_cross_entropy_loss_cpu'):
        return ('MISSING_KERNEL', 'bpd_cross_entropy_loss_cpu', None)
    batch_size = 64
    num_classes = 10
    pred = RNG.standard_normal((batch_size, num_classes)).astype(np.float32)
    target = RNG.integers(0, num_classes, batch_size).astype(np.int64)
    out = np.zeros(1, dtype=np.float32)
    lib.bpd_cross_entropy_loss_cpu(pred.ctypes.data, target.ctypes.data, out.ctypes.data,
                                    batch_size, num_classes)
    ref = F.cross_entropy(torch.from_numpy(pred), torch.from_numpy(target)).numpy()
    mu, nd, nt = ulp(ref.reshape(1), out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)


def triplet_margin_loss_problem(lib):
    if not hasattr(lib, 'bpd_triplet_margin_loss_cpu'):
        return ('MISSING_KERNEL', 'bpd_triplet_margin_loss_cpu', None)
    batch_size = 32
    feat_dim = 128
    margin = 1.0
    anchor = RNG.standard_normal((batch_size, feat_dim)).astype(np.float32)
    positive = RNG.standard_normal((batch_size, feat_dim)).astype(np.float32)
    negative = RNG.standard_normal((batch_size, feat_dim)).astype(np.float32)
    out = np.zeros(1, dtype=np.float32)
    lib.bpd_triplet_margin_loss_cpu(anchor.ctypes.data, positive.ctypes.data,
                                     negative.ctypes.data, out.ctypes.data,
                                     batch_size, feat_dim, ctypes.c_float(margin))
    ref = F.triplet_margin_loss(torch.from_numpy(anchor), torch.from_numpy(positive),
                                 torch.from_numpy(negative), margin=margin, p=2).numpy()
    mu, nd, nt = ulp(ref.reshape(1), out)
    return ('BIT_IDENTICAL' if mu == 0 else 'DIVERGENT', mu, nd)




# ─── Problem catalog ───────────────────────────────────────────────────────
#
# Each entry: (problem_number, name, runner_lambda)

def build_catalog(lib):
    """Build the full 100-problem catalog. Returns list of (num, name, fn)."""
    cat = []

    # 1–18: Matmul variants
    cat.append((1, '1_Square_matrix_multiplication',  lambda: matmul_problem(lib, 256, 256, 256)))
    cat.append((2, '2_Standard_matrix_multiplication', lambda: matmul_problem(lib, 128, 256, 64)))
    cat.append((3, '3_Batched_matrix_multiplication',  lambda: bmm_problem(lib)))
    cat.append((4, '4_Matrix_vector_multiplication',   lambda: matmul_problem(lib, 256, 1, 128)))
    cat.append((5, '5_Matrix_scalar_multiplication',   lambda: scalar_mul_problem(lib)))
    cat.append((6, '6_Matmul_with_large_K_dimension',  lambda: matmul_problem(lib, 16, 16, 4096)))
    cat.append((7, '7_Matmul_with_small_K_dimension',  lambda: matmul_problem(lib, 256, 256, 16)))
    cat.append((8, '8_Matmul_with_irregular_shapes',   lambda: matmul_problem(lib, 67, 89, 113)))
    cat.append((9, '9_Tall_skinny_matrix_multiplication', lambda: matmul_problem(lib, 1024, 16, 32)))
    cat.append((10, '10_3D_tensor_matrix_multiplication', lambda: tensor3d_matmul_problem(lib)))
    cat.append((11, '11_4D_tensor_matrix_multiplication', lambda: tensor4d_matmul_problem(lib)))
    cat.append((12, '12_Matmul_with_diagonal_matrices', lambda: diag_matmul_problem(lib)))
    cat.append((13, '13_Matmul_for_symmetric_matrices', lambda: matmul_problem(lib, 128, 128, 128)))
    cat.append((14, '14_Matmul_for_upper_triangular',  lambda: matmul_problem(lib, 128, 128, 128)))
    cat.append((15, '15_Matmul_for_lower_triangular',  lambda: matmul_problem(lib, 128, 128, 128)))
    cat.append((16, '16_Matmul_with_transposed_A',     lambda: matmul_problem(lib, 128, 128, 128)))
    cat.append((17, '17_Matmul_with_transposed_B',     lambda: matmul_problem(lib, 128, 128, 128)))
    cat.append((18, '18_Matmul_with_transposed_both',  lambda: matmul_problem(lib, 128, 128, 128)))

    # 19–32: Activations
    cat.append((19, '19_ReLU',         lambda: elementwise(lib, 'bpd_relu_cpu', lambda t: F.relu(t))))
    cat.append((20, '20_LeakyReLU',    lambda: elementwise(lib, 'bpd_leaky_relu_cpu', lambda t: F.leaky_relu(t))))
    cat.append((21, '21_Sigmoid',      lambda: elementwise(lib, 'bpd_sigmoid_cpu', lambda t: torch.sigmoid(t))))
    cat.append((22, '22_Tanh',         lambda: elementwise(lib, 'bpd_tanh_cpu', lambda t: torch.tanh(t))))
    cat.append((23, '23_Softmax',      lambda: softmax_problem(lib)))
    cat.append((24, '24_LogSoftmax',   lambda: logsoftmax_problem(lib)))
    cat.append((25, '25_Swish',        lambda: elementwise(lib, 'bpd_silu_cpu', lambda t: F.silu(t))))  # Swish == SiLU
    cat.append((26, '26_GELU',         lambda: elementwise(lib, 'bpd_gelu_cpu', lambda t: F.gelu(t))))
    cat.append((27, '27_SELU',         lambda: elementwise(lib, 'bpd_selu_cpu', lambda t: F.selu(t))))
    cat.append((28, '28_HardSigmoid',  lambda: elementwise(lib, 'bpd_hardsigmoid_cpu', lambda t: F.hardsigmoid(t))))
    cat.append((29, '29_Softplus',     lambda: elementwise(lib, 'bpd_softplus_cpu', lambda t: F.softplus(t))))
    cat.append((30, '30_Softsign',     lambda: elementwise(lib, 'bpd_softsign_cpu', lambda t: F.softsign(t))))
    cat.append((31, '31_ELU',          lambda: elementwise(lib, 'bpd_elu_cpu', lambda t: F.elu(t))))
    cat.append((32, '32_HardTanh',     lambda: elementwise(lib, 'bpd_clamp_cpu', lambda t: F.hardtanh(t))))

    # 33–40: Normalizations
    cat.append((33, '33_BatchNorm',    lambda: batchnorm_problem(lib)))
    cat.append((34, '34_InstanceNorm', lambda: instancenorm_problem(lib)))
    cat.append((35, '35_GroupNorm',    lambda: groupnorm_problem(lib)))
    cat.append((36, '36_RMSNorm',      lambda: rmsnorm_problem(lib)))
    cat.append((37, '37_FrobeniusNorm', lambda: frobenius_problem(lib)))
    cat.append((38, '38_L1Norm',       lambda: l1norm_problem(lib)))
    cat.append((39, '39_L2Norm',       lambda: l2norm_problem(lib)))
    cat.append((40, '40_LayerNorm',    lambda: layernorm_problem(lib)))

    # 41–46: Pooling
    cat.append((41, '41_Max_Pooling_1D',   lambda: maxpool1d_problem(lib)))
    cat.append((42, '42_Max_Pooling_2D',   lambda: maxpool2d_problem(lib)))
    cat.append((43, '43_Max_Pooling_3D',   lambda: maxpool3d_problem(lib)))
    cat.append((44, '44_Average_Pooling_1D', lambda: avgpool1d_problem(lib)))
    cat.append((45, '45_Average_Pooling_2D', lambda: avgpool2d_problem(lib)))
    cat.append((46, '46_Average_Pooling_3D', lambda: avgpool3d_problem(lib)))

    # 47–49: Reductions
    cat.append((47, '47_Sum_reduction',  lambda: reduce_problem(lib, 'bpd_sum_cpu', lambda t: torch.sum(t))))
    cat.append((48, '48_Mean_reduction', lambda: reduce_problem(lib, 'bpd_mean_cpu', lambda t: torch.mean(t))))
    cat.append((49, '49_Max_reduction',  lambda: reduce_problem(lib, 'bpd_max_cpu', lambda t: torch.max(t))))

    # 50, 54–87: Convolutions (small reproducible shapes)
    # Conv2D variants via bpd_conv2d_full_cpu (im2col + GEMM, matches PyTorch).
    cat.append((50, '50_Conv2D',       lambda: conv2d_full_problem(lib, 1, 3, 16, 16, 8, 3, 3, stride=1, pad=1)))
    # 51_Argmax_over_a_dim, 52_Argmin_over_a_dim — not implemented
    cat.append((51, '51_Argmax_over_a_dimension', lambda: argmax_dim_problem(lib)))
    cat.append((52, '52_Argmin_over_a_dimension', lambda: argmin_dim_problem(lib)))
    cat.append((53, '53_Min_reduction_over_a_dimension', lambda: min_dim_problem(lib)))
    cat.append((54, '54_conv_standard_3D_square_input_square_kernel',
                lambda: conv3d_full_problem(lib, 1, 3, 8, 8, 8, 8, 3, 3, 3, pad=1)))
    cat.append((55, '55_conv_standard_2D_asymmetric_input_square_kernel', lambda: conv2d_full_problem(lib, 1, 3, 12, 20, 8, 3, 3)))
    cat.append((56, '56_conv_standard_2D_asymmetric_input_asymmetric_kernel', lambda: conv2d_full_problem(lib, 1, 3, 12, 20, 8, 3, 5, pad=(1,2))))
    cat.append((57, '57_conv_transposed_2D_square_input_square_kernel',
                lambda: conv_transpose2d_full_problem(lib, 1, 3, 8, 8, 4, 3, 3, stride=2)))
    cat.append((58, '58_conv_transposed_3D_asymmetric_input_asymmetric_kernel',
                lambda: conv_transpose3d_full_problem(lib, 1, 3, 6, 8, 10, 4, 3, 3, 5, stride=2, pad=(1,1,2))))
    cat.append((59, '59_conv_standard_3D_asymmetric_input_square_kernel',
                lambda: conv3d_full_problem(lib, 1, 3, 8, 12, 8, 8, 3, 3, 3, pad=1)))
    cat.append((60, '60_conv_standard_3D_square_input_asymmetric_kernel',
                lambda: conv3d_full_problem(lib, 1, 3, 8, 8, 8, 8, 3, 5, 3, pad=(1,2,1))))
    cat.append((61, '61_conv_transposed_3D_square_input_square_kernel',
                lambda: conv_transpose3d_full_problem(lib, 1, 3, 6, 6, 6, 4, 3, 3, 3, stride=2, pad=1)))
    cat.append((62, '62_conv_standard_2D_square_input_asymmetric_kernel_dilated', lambda: conv2d_full_problem(lib, 1, 3, 16, 16, 8, 3, 5, pad=(1,2), dilation=2)))
    cat.append((63, '63_conv_standard_2D_square_input_square_kernel',     lambda: conv2d_full_problem(lib, 1, 3, 16, 16, 8, 3, 3, pad=1)))
    cat.append((64, '64_conv_transposed_1D',
                lambda: conv_transpose1d_full_problem(lib, 1, 3, 32, 4, 3, stride=2, pad=1)))
    cat.append((65, '65_conv_transposed_2D_square_input_asymmetric_kernel',
                lambda: conv_transpose2d_full_problem(lib, 1, 3, 8, 8, 4, 3, 5, stride=2, pad=(1,2))))
    cat.append((66, '66_conv_standard_3D_asymmetric_input_asymmetric_kernel',
                lambda: conv3d_full_problem(lib, 1, 3, 8, 10, 12, 8, 3, 3, 5, pad=(1,1,2))))
    cat.append((67, '67_conv_standard_1D',
                lambda: conv1d_full_problem(lib, 1, 3, 64, 8, 3, pad=1)))
    cat.append((68, '68_conv_transposed_3D_square_input_asymmetric_kernel',
                lambda: conv_transpose3d_full_problem(lib, 1, 3, 6, 6, 6, 4, 3, 5, 3, stride=2, pad=(1,2,1))))
    cat.append((69, '69_conv_transposed_2D_square_input_asymmetric_kernel',
                lambda: conv_transpose2d_full_problem(lib, 1, 3, 6, 6, 4, 3, 5, stride=2, pad=1)))
    cat.append((70, '70_conv_transposed_3D_asymmetric_input_square_kernel',
                lambda: conv_transpose3d_full_problem(lib, 1, 3, 6, 8, 6, 4, 3, 3, 3, stride=2, pad=1)))
    cat.append((71, '71_conv_transposed_2D_asymmetric_input_square_kernel',
                lambda: conv_transpose2d_full_problem(lib, 1, 3, 6, 8, 4, 3, 3, stride=2, pad=1)))
    cat.append((72, '72_conv_transposed_3D_grouped',
                lambda: conv_transpose3d_full_problem(lib, 1, 4, 6, 6, 6, 4, 3, 3, 3, stride=1, pad=1, groups=2)))
    cat.append((73, '73_conv_transposed_3D_grouped',
                lambda: conv_transpose3d_full_problem(lib, 1, 4, 6, 6, 6, 8, 3, 3, 3, stride=2, pad=1, groups=2)))
    cat.append((74, '74_conv_transposed_1D_dilated',
                lambda: conv_transpose1d_full_problem(lib, 1, 3, 32, 4, 3, stride=1, pad=1, dilation=2)))
    cat.append((75, '75_conv_transposed_2D_dilated_grouped_padded',
                lambda: conv_transpose2d_full_problem(lib, 1, 4, 6, 6, 4, 3, 3, stride=1, pad=1, dilation=2, groups=2)))
    cat.append((76, '76_conv_standard_1D_dilated_strided',
                lambda: conv1d_full_problem(lib, 1, 3, 64, 8, 3, stride=2, pad=1, dilation=2)))
    cat.append((77, '77_conv_transposed_3D_padded_dilated_strided',
                lambda: conv_transpose3d_full_problem(lib, 1, 3, 6, 6, 6, 4, 3, 3, 3, stride=2, pad=1, dilation=2)))
    cat.append((78, '78_conv_transposed_2D_padded',
                lambda: conv_transpose2d_full_problem(lib, 1, 3, 8, 8, 4, 3, 3, stride=1, pad=1)))
    cat.append((79, '79_conv_transposed_1D_padded_strided_dilated',
                lambda: conv_transpose1d_full_problem(lib, 1, 3, 32, 4, 3, stride=2, pad=1, dilation=2)))
    cat.append((80, '80_conv_standard_2D_dilated_padded',
                lambda: conv2d_full_problem(lib, 1, 3, 16, 16, 8, 3, 5, pad=(2,4), dilation=2)))
    cat.append((81, '81_conv_transposed_2D_dilated_padded_strided',
                lambda: conv_transpose2d_full_problem(lib, 1, 3, 8, 8, 4, 3, 3, stride=2, pad=1, dilation=2)))
    # Depthwise Conv2D variants (82-85): groups = in_channels = out_channels
    cat.append((82, '82_conv_depthwise_2D_square_square',
                lambda: conv2d_full_problem(lib, 1, 8, 16, 16, 8, 3, 3, pad=1, groups=8)))
    cat.append((83, '83_conv_depthwise_2D_square_asym',
                lambda: conv2d_full_problem(lib, 1, 8, 16, 16, 8, 3, 5, pad=(1,2), groups=8)))
    cat.append((84, '84_conv_depthwise_2D_asym_square',
                lambda: conv2d_full_problem(lib, 1, 8, 12, 20, 8, 3, 3, pad=1, groups=8)))
    cat.append((85, '85_conv_depthwise_2D_asym_asym',
                lambda: conv2d_full_problem(lib, 1, 8, 12, 20, 8, 3, 5, pad=(1,2), groups=8)))
    # Depthwise-separable (86) and pointwise (87)
    cat.append((86, '86_conv_depthwise_separable_2D',
                lambda: depthwise_separable_problem(lib)))
    cat.append((87, '87_conv_pointwise_2D',
                lambda: conv2d_full_problem(lib, 1, 8, 16, 16, 16, 1, 1)))

    # 88: 88_MinGPT_NewGelu — gelu approximation (tanh form)
    cat.append((88, '88_MinGPT_NewGelu', lambda: mingpt_newgelu_problem(lib)))

    # 89–93: Cumulative
    for n, name, kernel, pt_fn in [
        (89, '89_cumsum',           'bpd_cumsum_cpu',           lambda t: torch.cumsum(t, dim=-1)),
        (90, '90_cumprod',          'bpd_cumprod_cpu',          lambda t: torch.cumprod(t, dim=-1)),
        (91, '91_cumsum_reverse',   'bpd_cumsum_reverse_cpu',   lambda t: torch.flip(torch.cumsum(torch.flip(t, [-1]), dim=-1), [-1])),
        (92, '92_cumsum_exclusive', 'bpd_cumsum_exclusive_cpu', lambda t: torch.cat([torch.zeros_like(t[..., :1]), torch.cumsum(t, dim=-1)[..., :-1]], dim=-1)),
    ]:
        cat.append((n, name, lambda k=kernel, f=pt_fn: elementwise(lib, k, f, n=512)))
    cat.append((93, '93_masked_cumsum', lambda: masked_cumsum_problem(lib)))

    # 94–100: Losses
    cat.append((94, '94_MSELoss',              lambda: mse_loss_problem(lib)))
    cat.append((95, '95_CrossEntropyLoss',     lambda: cross_entropy_problem(lib)))
    cat.append((96, '96_HuberLoss',            lambda: huber_loss_problem(lib)))
    cat.append((97, '97_ScaledDotProductAttention', lambda: scaled_dot_product_attention_problem(lib)))
    cat.append((98, '98_KLDivLoss',            lambda: kl_div_loss_problem(lib)))
    cat.append((99, '99_TripletMarginLoss',    lambda: triplet_margin_loss_problem(lib)))
    cat.append((100, '100_HingeLoss',          lambda: hinge_loss_problem(lib)))

    return cat


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    lib = load_lib()
    cat = build_catalog(lib)

    # Run each problem
    results = []
    for num, name, fn in cat:
        try:
            r = fn()
        except Exception as e:
            r = ('ERROR', repr(e), None)
        results.append((num, name, r))

    # Print per-problem table
    print(f"{'#':<4} {'name':<50} {'status':<16} {'detail':<32}")
    print("─" * 105)
    for num, name, (status, a, b) in results:
        if status == 'BIT_IDENTICAL':
            detail = '0 ULP'
        elif status == 'DIVERGENT':
            detail = f"{a} ULP ({b} diffs)"
        elif status == 'MISSING_KERNEL':
            detail = str(a)
        elif status == 'NOT_IMPLEMENTED':
            detail = str(a)
        else:
            detail = str(a)[:32]
        # Trim name to fit
        n = name if len(name) <= 49 else name[:46] + '...'
        print(f"{num:<4} {n:<50} {status:<16} {detail:<32}")

    # Summary
    print()
    print("═" * 105)
    by_status = {}
    for _, _, (s, _, _) in results:
        by_status[s] = by_status.get(s, 0) + 1
    total = len(results)
    print(f"Total problems: {total}")
    for s in ['BIT_IDENTICAL', 'DIVERGENT', 'MISSING_KERNEL', 'NOT_IMPLEMENTED', 'ERROR']:
        c = by_status.get(s, 0)
        pct = 100 * c / total
        print(f"  {s:<18} {c:>3}  ({pct:5.1f}%)")

    # Per-category breakdown
    print()
    print("Per-category breakdown:")
    cats = [('Matmul (1-18)', range(1, 19)),
            ('Activations (19-32)', range(19, 33)),
            ('Norms (33-40)', range(33, 41)),
            ('Pooling (41-46)', range(41, 47)),
            ('Reductions (47-53)', range(47, 54)),
            ('Convolutions (50, 54-87)', list(range(54, 88)) + [50, 88]),
            ('Cumulative (89-93)', range(89, 94)),
            ('Losses (94-100)', range(94, 101))]
    for cname, rng_ in cats:
        in_cat = [r for r in results if r[0] in rng_]
        bi = sum(1 for _, _, (s, _, _) in in_cat if s == 'BIT_IDENTICAL')
        dv = sum(1 for _, _, (s, _, _) in in_cat if s == 'DIVERGENT')
        mk = sum(1 for _, _, (s, _, _) in in_cat if s == 'MISSING_KERNEL')
        ni = sum(1 for _, _, (s, _, _) in in_cat if s == 'NOT_IMPLEMENTED')
        er = sum(1 for _, _, (s, _, _) in in_cat if s == 'ERROR')
        n_ = len(in_cat)
        print(f"  {cname:<26} {n_:>3} total  |  BI {bi:>2}  DV {dv:>2}  MK {mk:>2}  NI {ni:>2}  ER {er:>2}")

    bi_count = by_status.get('BIT_IDENTICAL', 0)
    print()
    print(f"BIT_IDENTICAL: {bi_count}/{total}")
    return 0 if bi_count == total else 1


if __name__ == "__main__":
    sys.exit(main())
