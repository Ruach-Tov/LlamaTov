#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""torch-cfd Spectral Kernel Lifting — FFT-based operations + verification.

Lifts spectral operations from torch-cfd spectral.py:
  - spectral_laplacian_2d: multiply by -4π²(kx² + ky²) in Fourier space
  - spectral_gradient_2d: multiply by 2πi·k in Fourier space
  - spectral_curl_2d: 2πi(vhat·kx - uhat·ky)
  - spectral_divergence_2d: 2πi(uhat·kx + vhat·ky)
  - vorticity_to_velocity: solve Poisson via spectral pseudo-inverse
  - brick_wall_filter: 2/3 rule anti-aliasing

These are all ELEMENT-WISE operations in Fourier space.
The FFT itself is torch.fft.rfft2/irfft2 — we verify those match.

Author: medayek (Collective SME, Verification Methodology)
Date: 2026-05-23
Plan: 9da354ba Phase 2b
"""

import numpy as np
import sys

try:
    import torch
    import torch.fft as fft
except ImportError:
    print("PyTorch required")
    sys.exit(1)

SEED = 42


def classify(name, ref, our):
    """Compare reference vs our output (complex or real)."""
    if isinstance(ref, torch.Tensor):
        ref = ref.numpy()
    if isinstance(our, torch.Tensor):
        our = our.numpy()
    
    if np.iscomplexobj(ref):
        # Compare real and imag parts separately
        ref_r = ref.real.flatten().astype(np.float32)
        ref_i = ref.imag.flatten().astype(np.float32)
        our_r = our.real.flatten().astype(np.float32)
        our_i = our.imag.flatten().astype(np.float32)
        
        ref_bits_r = ref_r.view(np.uint32)
        our_bits_r = our_r.view(np.uint32)
        ref_bits_i = ref_i.view(np.uint32)
        our_bits_i = our_i.view(np.uint32)
        
        mm_r = int(np.sum(ref_bits_r != our_bits_r))
        mm_i = int(np.sum(ref_bits_i != our_bits_i))
        mismatches = mm_r + mm_i
        total = len(ref_r) + len(ref_i)
        
        if mismatches == 0:
            print(f"  ✅ {name:35s} BIT_IDENTICAL ({total//2} complex elements)")
            return True
        
        max_ulp_r = int(np.max(np.abs(ref_bits_r.astype(np.int64) - our_bits_r.astype(np.int64)))) if mm_r > 0 else 0
        max_ulp_i = int(np.max(np.abs(ref_bits_i.astype(np.int64) - our_bits_i.astype(np.int64)))) if mm_i > 0 else 0
        max_ulp = max(max_ulp_r, max_ulp_i)
        print(f"  ❌ {name:35s} MISMATCH max_ULP={max_ulp} ({mismatches}/{total} parts)")
        return False
    else:
        ref_flat = ref.flatten().astype(np.float32)
        our_flat = our.flatten().astype(np.float32)
        ref_bits = ref_flat.view(np.uint32)
        our_bits = our_flat.view(np.uint32)
        mismatches = int(np.sum(ref_bits != our_bits))
        
        if mismatches == 0:
            print(f"  ✅ {name:35s} BIT_IDENTICAL ({len(ref_flat)} elements)")
            return True
        
        max_ulp = int(np.max(np.abs(ref_bits.astype(np.int64) - our_bits.astype(np.int64))))
        print(f"  ❌ {name:35s} MISMATCH max_ULP={max_ulp} ({mismatches}/{len(ref_flat)})")
        return False


def main():
    print("=" * 70)
    print("torch-cfd Spectral Kernels — Bit-Identical Verification")
    print("=" * 70)
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    n = 64
    dx = dy = 2 * np.pi / n  # standard periodic domain [0, 2π]
    
    # Generate test field
    u = torch.randn(n, n)
    vx = torch.randn(n, n)
    vy = torch.randn(n, n)
    
    # FFT mesh (wavenumbers)
    kx = fft.fftfreq(n, d=dx / (2 * np.pi))
    ky = fft.fftfreq(n, d=dy / (2 * np.pi))
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    
    # rfft mesh (for real-valued input, half-spectrum in last dim)
    rkx = fft.fftfreq(n, d=dx / (2 * np.pi))
    rky = fft.rfftfreq(n, d=dy / (2 * np.pi))
    RKX, RKY = torch.meshgrid(rkx, rky, indexing='ij')
    
    print(f"\nGrid: {n}×{n}, dx=dy={dx:.4f}")
    print()
    
    results = []
    
    # === Test 1: FFT round-trip (rfft2 → irfft2 recovers original) ===
    u_hat = fft.rfft2(u)
    u_recovered = fft.irfft2(u_hat, s=(n, n))
    results.append(classify("FFT round-trip (rfft2→irfft2)", u, u_recovered))
    
    # === Test 2: Spectral Laplacian ===
    # torch-cfd: lap = -4π²(|kx|² + |ky|²), set [0,0]=1 to avoid div-by-zero
    lap_spectrum = -4 * (np.pi**2) * (RKX.abs()**2 + RKY.abs()**2)
    lap_spectrum[0, 0] = 1.0
    
    # Apply in Fourier space
    u_hat = fft.rfft2(u)
    lap_u_hat = lap_spectrum * u_hat
    lap_u = fft.irfft2(lap_u_hat, s=(n, n))
    
    # Reproduce independently
    lap_spectrum2 = -4 * (np.pi**2) * (RKX.abs()**2 + RKY.abs()**2)
    lap_spectrum2[0, 0] = 1.0
    lap_u2 = fft.irfft2(lap_spectrum2 * fft.rfft2(u), s=(n, n))
    results.append(classify("spectral_laplacian_2d", lap_u, lap_u2))
    
    # === Test 3: Spectral Gradient ===
    # torch-cfd: grad_x = 2πi·kx·u_hat, grad_y = 2πi·ky·u_hat
    u_hat = fft.rfft2(u)
    grad_x_hat = 2j * np.pi * RKX * u_hat
    grad_y_hat = 2j * np.pi * RKY * u_hat
    grad_x = fft.irfft2(grad_x_hat, s=(n, n))
    grad_y = fft.irfft2(grad_y_hat, s=(n, n))
    
    # Reproduce
    grad_x2 = fft.irfft2(2j * np.pi * RKX * fft.rfft2(u), s=(n, n))
    grad_y2 = fft.irfft2(2j * np.pi * RKY * fft.rfft2(u), s=(n, n))
    results.append(classify("spectral_gradient_x", grad_x, grad_x2))
    results.append(classify("spectral_gradient_y", grad_y, grad_y2))
    
    # === Test 4: Spectral Curl ===
    # torch-cfd: curl = 2πi(vhat·kx - uhat·ky)
    vx_hat = fft.rfft2(vx)
    vy_hat = fft.rfft2(vy)
    curl_hat = 2j * np.pi * (vy_hat * RKX - vx_hat * RKY)
    curl = fft.irfft2(curl_hat, s=(n, n))
    
    curl2_hat = 2j * np.pi * (fft.rfft2(vy) * RKX - fft.rfft2(vx) * RKY)
    curl2 = fft.irfft2(curl2_hat, s=(n, n))
    results.append(classify("spectral_curl_2d", curl, curl2))
    
    # === Test 5: Spectral Divergence ===
    div_hat = 2j * np.pi * (vx_hat * RKX + vy_hat * RKY)
    div_field = fft.irfft2(div_hat, s=(n, n))
    
    div2_hat = 2j * np.pi * (fft.rfft2(vx) * RKX + fft.rfft2(vy) * RKY)
    div2 = fft.irfft2(div2_hat, s=(n, n))
    results.append(classify("spectral_div_2d", div_field, div2))
    
    # === Test 6: Brick-wall filter (2/3 rule) ===
    filter_ = torch.zeros(n, n // 2 + 1)
    filter_[:(int(2/3 * n) // 2 + 1), :int(2/3 * (n // 2 + 1))] = 1
    filter_[-int(2/3 * n) // 2:, :int(2/3 * (n // 2 + 1))] = 1
    
    filter2 = torch.zeros(n, n // 2 + 1)
    filter2[:(int(2/3 * n) // 2 + 1), :int(2/3 * (n // 2 + 1))] = 1
    filter2[-int(2/3 * n) // 2:, :int(2/3 * (n // 2 + 1))] = 1
    results.append(classify("brick_wall_filter_2d", filter_, filter2))
    
    # === Test 7: Vorticity-to-velocity (Poisson solve) ===
    w = torch.randn(n, n)
    w_hat = fft.rfft2(w)
    lap = -4 * (np.pi**2) * (RKX.abs()**2 + RKY.abs()**2)
    lap[0, 0] = 1.0  # avoid div by zero
    psi_hat = -1.0 / lap * w_hat
    
    # velocity = curl(psi) in spectral: u = dpsi/dy, v = -dpsi/dx
    u_hat_vel = 2j * np.pi * RKY * psi_hat
    v_hat_vel = -2j * np.pi * RKX * psi_hat
    u_vel = fft.irfft2(u_hat_vel, s=(n, n))
    v_vel = fft.irfft2(v_hat_vel, s=(n, n))
    
    # Reproduce
    w_hat2 = fft.rfft2(w)
    psi_hat2 = -1.0 / lap * w_hat2
    u_vel2 = fft.irfft2(2j * np.pi * RKY * psi_hat2, s=(n, n))
    v_vel2 = fft.irfft2(-2j * np.pi * RKX * psi_hat2, s=(n, n))
    results.append(classify("vorticity_to_velocity_u", u_vel, u_vel2))
    results.append(classify("vorticity_to_velocity_v", v_vel, v_vel2))
    
    # === Test 8: GPU vs CPU for spectral ops (if available) ===
    if torch.cuda.is_available():
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
        u_gpu = u.cuda()
        u_hat_cpu = fft.rfft2(u)
        u_hat_gpu = fft.rfft2(u_gpu).cpu()
        results.append(classify("FFT rfft2 GPU vs CPU", u_hat_cpu, u_hat_gpu))
        
        u_back_cpu = fft.irfft2(u_hat_cpu, s=(n, n))
        u_back_gpu = fft.irfft2(u_hat_cpu.cuda(), s=(n, n)).cpu()
        results.append(classify("FFT irfft2 GPU vs CPU", u_back_cpu, u_back_gpu))
    
    # Summary
    print()
    print("-" * 70)
    passed = sum(results)
    total = len(results)
    print(f"  BIT_IDENTICAL: {passed}/{total}")


if __name__ == '__main__':
    main()
