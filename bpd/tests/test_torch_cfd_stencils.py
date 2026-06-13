#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""torch-cfd Stencil Kernel Lifting — BPD facts + C kernels + verification.

Lifts the core finite difference stencil operators from torch-cfd to:
  1. C kernel implementations (no PyTorch dependency)
  2. Verification harness comparing C output vs torch-cfd PyTorch output

Stencil operators lifted:
  - forward_difference(u, dim)  → (u[i+1] - u[i]) / dx
  - backward_difference(u, dim) → (u[i] - u[i-1]) / dx  
  - central_difference(u, dim)  → (u[i+1] - u[i-1]) / (2*dx)
  - laplacian(u)                → sum_d (u[i-1,d] - 2*u[i] + u[i+1,d]) / dx_d^2
  - divergence(v)               → sum_d backward_difference(v_d, d)

Each with periodic boundary conditions (wrap-around).

Author: medayek (Collective SME, Verification Methodology)
Date: 2026-05-22
Plan: 9da354ba Phase 2a
"""

import numpy as np
import ctypes
import tempfile
import subprocess
import os

# ═══════════════════════════════════════════════════════════════════════
# C kernel implementations — lifted from torch-cfd semantics
# ═══════════════════════════════════════════════════════════════════════

STENCIL_KERNELS_C = r"""
#include <math.h>
#include <string.h>

/*
 * Stencil operators for 2D grids with periodic boundary conditions.
 * Grid layout: row-major (ny, nx), operations along specified axis.
 * 
 * BPD facts that would generate these:
 *   stencil_op(forward_diff, periodic, 1).
 *   stencil_op(backward_diff, periodic, 1).
 *   stencil_op(central_diff, periodic, 2).
 *   stencil_op(laplacian_2d, periodic, 5).
 *
 * SUBSTRATE-DESIGN PARAMETER: division_strategy
 *   BPD_STENCIL_RECIPROCAL=0 (default): use direct division (a / dx)
 *     → bit-identical with PyTorch CPU (0 ULP)
 *   BPD_STENCIL_RECIPROCAL=1: use reciprocal multiply (a * (1/dx))
 *     → ~1 ULP divergence, but faster on GPU (div is ~10× slower than mul)
 *
 * The parameter is sweepable: set via env var at runtime.
 * Correctness gate: 0 ULP when RECIPROCAL=0, ≤1 ULP when RECIPROCAL=1.
 */

#include <stdlib.h>

static int _use_reciprocal = -1;
static int use_reciprocal(void) {
    if (_use_reciprocal < 0) {
        const char *env = getenv("BPD_STENCIL_RECIPROCAL");
        _use_reciprocal = (env && env[0] == '1') ? 1 : 0;
    }
    return _use_reciprocal;
}

/* Division helper: matches PyTorch when reciprocal=0, faster when reciprocal=1 */
static inline float stencil_div(float a, float b) {
    if (use_reciprocal()) {
        return a * (1.0f / b);
    }
    return a / b;
}

/* Forward difference along axis 0 (y-direction): (u[i+1,j] - u[i,j]) / dy */
void bpd_forward_diff_y(const float *u, float *out, int ny, int nx, float dy) {
    for (int i = 0; i < ny; i++) {
        int ip1 = (i + 1) % ny;
        for (int j = 0; j < nx; j++) {
            out[i * nx + j] = stencil_div(u[ip1 * nx + j] - u[i * nx + j], dy);
        }
    }
}

/* Forward difference along axis 1 (x-direction): (u[i,j+1] - u[i,j]) / dx */
void bpd_forward_diff_x(const float *u, float *out, int ny, int nx, float dx) {
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            int jp1 = (j + 1) % nx;
            out[i * nx + j] = stencil_div(u[i * nx + jp1] - u[i * nx + j], dx);
        }
    }
}

/* Backward difference along axis 0: (u[i,j] - u[i-1,j]) / dy */
void bpd_backward_diff_y(const float *u, float *out, int ny, int nx, float dy) {
    for (int i = 0; i < ny; i++) {
        int im1 = (i - 1 + ny) % ny;
        for (int j = 0; j < nx; j++) {
            out[i * nx + j] = stencil_div(u[i * nx + j] - u[im1 * nx + j], dy);
        }
    }
}

/* Backward difference along axis 1: (u[i,j] - u[i,j-1]) / dx */
void bpd_backward_diff_x(const float *u, float *out, int ny, int nx, float dx) {
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            int jm1 = (j - 1 + nx) % nx;
            out[i * nx + j] = stencil_div(u[i * nx + j] - u[i * nx + jm1], dx);
        }
    }
}

/* Central difference along axis 0: (u[i+1,j] - u[i-1,j]) / (2*dy) */
void bpd_central_diff_y(const float *u, float *out, int ny, int nx, float dy) {
    float two_dy = 2.0f * dy;
    for (int i = 0; i < ny; i++) {
        int ip1 = (i + 1) % ny;
        int im1 = (i - 1 + ny) % ny;
        for (int j = 0; j < nx; j++) {
            out[i * nx + j] = stencil_div(u[ip1 * nx + j] - u[im1 * nx + j], two_dy);
        }
    }
}

/* Central difference along axis 1: (u[i,j+1] - u[i,j-1]) / (2*dx) */
void bpd_central_diff_x(const float *u, float *out, int ny, int nx, float dx) {
    float two_dx = 2.0f * dx;
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            int jp1 = (j + 1) % nx;
            int jm1 = (j - 1 + nx) % nx;
            out[i * nx + j] = stencil_div(u[i * nx + jp1] - u[i * nx + jm1], two_dx);
        }
    }
}

/* 2D Laplacian with periodic BC — matches torch-cfd's computation:
   lap(u) = -2*u*(1/dx^2 + 1/dy^2) + (u[i-1]+u[i+1])/dy^2 + (u[j-1]+u[j+1])/dx^2 */
void bpd_laplacian_2d(const float *u, float *out, int ny, int nx, float dx, float dy) {
    float scale_x = 1.0f / (dx * dx);
    float scale_y = 1.0f / (dy * dy);
    float center_scale = -2.0f * (scale_x + scale_y);
    for (int i = 0; i < ny; i++) {
        int ip1 = (i + 1) % ny;
        int im1 = (i - 1 + ny) % ny;
        for (int j = 0; j < nx; j++) {
            int jp1 = (j + 1) % nx;
            int jm1 = (j - 1 + nx) % nx;
            out[i * nx + j] = 
                u[i * nx + j] * center_scale +
                (u[im1 * nx + j] + u[ip1 * nx + j]) * scale_y +
                (u[i * nx + jm1] + u[i * nx + jp1]) * scale_x;
        }
    }
}

/* 2D Divergence with backward differences (periodic BC):
   div(vx, vy) = (vx - roll(vx,1,x))/dx + (vy - roll(vy,1,y))/dy */
void bpd_divergence_2d(const float *vx, const float *vy, float *out,
                        int ny, int nx, float dx, float dy) {
    for (int i = 0; i < ny; i++) {
        int im1 = (i - 1 + ny) % ny;
        for (int j = 0; j < nx; j++) {
            int jm1 = (j - 1 + nx) % nx;
            float dvx_dx = stencil_div(vx[i * nx + j] - vx[i * nx + jm1], dx);
            float dvy_dy = stencil_div(vy[i * nx + j] - vy[im1 * nx + j], dy);
            out[i * nx + j] = dvx_dx + dvy_dy;
        }
    }
}
"""


def compile_stencil_kernels():
    """Compile stencil kernels to shared library."""
    with tempfile.NamedTemporaryFile(suffix='.c', mode='w', delete=False) as f:
        f.write(STENCIL_KERNELS_C)
        c_path = f.name
    so_path = c_path.replace('.c', '.so')
    result = subprocess.run(
        ['gcc', '-O2', '-shared', '-fPIC', '-o', so_path, c_path, '-lm'],
        capture_output=True, text=True, timeout=10
    )
    os.unlink(c_path)
    if result.returncode != 0:
        raise RuntimeError(f"Compile failed: {result.stderr}")
    
    lib = ctypes.CDLL(so_path)
    c_float_p = ctypes.POINTER(ctypes.c_float)
    
    # Set signatures
    for name in ['bpd_forward_diff_y', 'bpd_forward_diff_x',
                 'bpd_backward_diff_y', 'bpd_backward_diff_x',
                 'bpd_central_diff_y', 'bpd_central_diff_x']:
        fn = getattr(lib, name)
        fn.restype = None
        fn.argtypes = [c_float_p, c_float_p, ctypes.c_int, ctypes.c_int, ctypes.c_float]
    
    lib.bpd_laplacian_2d.restype = None
    lib.bpd_laplacian_2d.argtypes = [c_float_p, c_float_p, 
                                      ctypes.c_int, ctypes.c_int,
                                      ctypes.c_float, ctypes.c_float]
    
    lib.bpd_divergence_2d.restype = None
    lib.bpd_divergence_2d.argtypes = [c_float_p, c_float_p, c_float_p,
                                       ctypes.c_int, ctypes.c_int,
                                       ctypes.c_float, ctypes.c_float]
    
    return lib, so_path


# ═══════════════════════════════════════════════════════════════════════
# PyTorch reference implementations (matching torch-cfd exactly)
# ═══════════════════════════════════════════════════════════════════════

def torch_forward_diff(u, dim, dx):
    """torch-cfd forward_difference: (roll(-1, dim) - u) / dx"""
    import torch
    t = torch.from_numpy(u)
    return ((torch.roll(t, -1, dims=dim) - t) / dx).numpy()

def torch_backward_diff(u, dim, dx):
    """torch-cfd backward_difference: (u - roll(+1, dim)) / dx"""
    import torch
    t = torch.from_numpy(u)
    return ((t - torch.roll(t, 1, dims=dim)) / dx).numpy()

def torch_central_diff(u, dim, dx):
    """torch-cfd central_difference: (roll(-1) - roll(+1)) / (2*dx)"""
    import torch
    t = torch.from_numpy(u)
    return ((torch.roll(t, -1, dims=dim) - torch.roll(t, 1, dims=dim)) / (2 * dx)).numpy()

def torch_laplacian_2d(u, dx, dy):
    """torch-cfd laplacian: sum over dims of second-order central diff."""
    import torch
    t = torch.from_numpy(u)
    lap = (-2 * t * (1/dx**2 + 1/dy**2)
           + (torch.roll(t, -1, 0) + torch.roll(t, 1, 0)) / dy**2
           + (torch.roll(t, -1, 1) + torch.roll(t, 1, 1)) / dx**2)
    return lap.numpy()

def torch_divergence_2d(vx, vy, dx, dy):
    """torch-cfd divergence: backward_diff(vx, x) + backward_diff(vy, y)"""
    import torch
    tx = torch.from_numpy(vx)
    ty = torch.from_numpy(vy)
    dvx_dx = (tx - torch.roll(tx, 1, dims=1)) / dx
    dvy_dy = (ty - torch.roll(ty, 1, dims=0)) / dy
    return (dvx_dx + dvy_dy).numpy()


# ═══════════════════════════════════════════════════════════════════════
# Verification harness
# ═══════════════════════════════════════════════════════════════════════

def run_kernel(lib, func_name, u, *extra_args):
    """Run a C stencil kernel and return the output."""
    c_float_p = ctypes.POINTER(ctypes.c_float)
    ny, nx = u.shape
    inp = np.ascontiguousarray(u, dtype=np.float32)
    out = np.zeros_like(inp)
    fn = getattr(lib, func_name)
    
    if 'divergence' in func_name:
        # divergence takes two input arrays
        vx, vy = u, extra_args[0]
        vx = np.ascontiguousarray(vx, dtype=np.float32)
        vy = np.ascontiguousarray(vy, dtype=np.float32)
        fn(vx.ctypes.data_as(c_float_p),
           vy.ctypes.data_as(c_float_p),
           out.ctypes.data_as(c_float_p),
           ctypes.c_int(ny), ctypes.c_int(nx),
           *[ctypes.c_float(a) for a in extra_args[1:]])
    elif 'laplacian' in func_name:
        fn(inp.ctypes.data_as(c_float_p),
           out.ctypes.data_as(c_float_p),
           ctypes.c_int(ny), ctypes.c_int(nx),
           *[ctypes.c_float(a) for a in extra_args])
    else:
        fn(inp.ctypes.data_as(c_float_p),
           out.ctypes.data_as(c_float_p),
           ctypes.c_int(ny), ctypes.c_int(nx),
           *[ctypes.c_float(a) for a in extra_args])
    return out


def classify(name, ref, our):
    """Compare reference vs our output."""
    ref_flat = ref.flatten().astype(np.float32)
    our_flat = our.flatten().astype(np.float32)
    
    ref_bits = ref_flat.view(np.uint32)
    our_bits = our_flat.view(np.uint32)
    mismatches = np.sum(ref_bits != our_bits)
    
    if mismatches == 0:
        print(f"  ✅ {name:35s} BIT_IDENTICAL ({ref_flat.size} elements)")
        return True
    
    max_ulp = int(np.max(np.abs(ref_bits.astype(np.int64) - our_bits.astype(np.int64))))
    max_abs = float(np.max(np.abs(ref_flat - our_flat)))
    print(f"  ❌ {name:35s} MISMATCH max_ULP={max_ulp} max_abs={max_abs:.2e} ({mismatches}/{ref_flat.size})")
    return False


def main():
    print("=" * 70)
    print("torch-cfd Stencil Kernels — Bit-Identical Verification")
    print("=" * 70)
    
    # Compile our C kernels
    lib, so_path = compile_stencil_kernels()
    
    # Test grid
    np.random.seed(42)
    ny, nx = 64, 64
    dx, dy = 0.1, 0.1
    u = np.random.randn(ny, nx).astype(np.float32)
    vx = np.random.randn(ny, nx).astype(np.float32)
    vy = np.random.randn(ny, nx).astype(np.float32)
    
    print(f"\nGrid: {ny}×{nx}, dx={dx}, dy={dy}")
    print()
    
    results = []
    
    # Forward differences
    ref = torch_forward_diff(u, 0, dy)
    our = run_kernel(lib, 'bpd_forward_diff_y', u, dy)
    results.append(classify("forward_diff_y", ref, our))
    
    ref = torch_forward_diff(u, 1, dx)
    our = run_kernel(lib, 'bpd_forward_diff_x', u, dx)
    results.append(classify("forward_diff_x", ref, our))
    
    # Backward differences
    ref = torch_backward_diff(u, 0, dy)
    our = run_kernel(lib, 'bpd_backward_diff_y', u, dy)
    results.append(classify("backward_diff_y", ref, our))
    
    ref = torch_backward_diff(u, 1, dx)
    our = run_kernel(lib, 'bpd_backward_diff_x', u, dx)
    results.append(classify("backward_diff_x", ref, our))
    
    # Central differences
    ref = torch_central_diff(u, 0, dy)
    our = run_kernel(lib, 'bpd_central_diff_y', u, dy)
    results.append(classify("central_diff_y", ref, our))
    
    ref = torch_central_diff(u, 1, dx)
    our = run_kernel(lib, 'bpd_central_diff_x', u, dx)
    results.append(classify("central_diff_x", ref, our))
    
    # Laplacian
    ref = torch_laplacian_2d(u, dx, dy)
    our = run_kernel(lib, 'bpd_laplacian_2d', u, dx, dy)
    results.append(classify("laplacian_2d", ref, our))
    
    # Divergence
    ref = torch_divergence_2d(vx, vy, dx, dy)
    c_float_p = ctypes.POINTER(ctypes.c_float)
    vx_c = np.ascontiguousarray(vx, dtype=np.float32)
    vy_c = np.ascontiguousarray(vy, dtype=np.float32)
    out_div = np.zeros((ny, nx), dtype=np.float32)
    lib.bpd_divergence_2d(
        vx_c.ctypes.data_as(c_float_p),
        vy_c.ctypes.data_as(c_float_p),
        out_div.ctypes.data_as(c_float_p),
        ctypes.c_int(ny), ctypes.c_int(nx),
        ctypes.c_float(dx), ctypes.c_float(dy))
    results.append(classify("divergence_2d", ref, out_div))
    
    # Summary
    print()
    print("-" * 70)
    passed = sum(results)
    total = len(results)
    print(f"  BIT_IDENTICAL: {passed}/{total}")
    
    os.unlink(so_path)
    return passed == total


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
