# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""
Roe approximate Riemann solver — Python reference, float32 throughout.

Matches mavchin's hand-written k_roe_flux at /tmp/sod_gpu_kernels.cu:82-153.

Substantive differences from a "textbook Toro" Roe:
  - Harten entropy fix on the eigenvalues (eps = 0.1f * a_roe)
  - Wave strengths use rho_roe = sqrt_rl * sqrt_rr (not Toro's eq. 11.69-11.71)
  - State diffs are PRIMITIVE (drho, du, dp), not CONSERVATIVE (dU_0/1/2)

These choices are kernel-author choices; this Python reference deliberately
matches mavchin's CUDA so that the bit-identical (Level A) verification is
algorithmically meaningful.

Per medayek's methodology principle: "inspection-based hypothesis vs
empirical truth. Measure, don't assume. Align with the subsumption target."
"""
import numpy as np

GAMMA = np.float32(1.4)
GM1 = GAMMA - np.float32(1.0)  # = 0.4f


def roe_flux(rho_L, u_L, p_L, rho_R, u_R, p_R):
    """Compute the Roe flux at an interface given left/right primitive states.
    
    Matches mavchin's k_roe_flux exactly: Harten entropy fix with eps=0.1*a_roe,
    rho_roe wave strength formulation, primitive state diffs.
    
    Returns: (F_rho, F_rhou, F_E) = (mass flux, momentum flux, energy flux)
    """
    rho_l = np.float32(rho_L); u_l = np.float32(u_L); p_l = np.float32(p_L)
    rho_r = np.float32(rho_R); u_r = np.float32(u_R); p_r = np.float32(p_R)
    
    # Total energy per unit volume (conservative third component)
    E_l = p_l / GM1 + np.float32(0.5) * rho_l * u_l * u_l
    E_r = p_r / GM1 + np.float32(0.5) * rho_r * u_r * u_r
    
    # Total enthalpy: H = (E + p) / rho
    H_l = (E_l + p_l) / rho_l
    H_r = (E_r + p_r) / rho_r
    
    # Roe averages (density-weighted)
    sqrt_rl = np.sqrt(rho_l)
    sqrt_rr = np.sqrt(rho_r)
    denom = sqrt_rl + sqrt_rr
    u_roe = (sqrt_rl * u_l + sqrt_rr * u_r) / denom
    H_roe = (sqrt_rl * H_l + sqrt_rr * H_r) / denom
    rho_roe = sqrt_rl * sqrt_rr  # mavchin's rho_roe choice
    
    # Roe-averaged sound speed
    a_roe = np.sqrt(GM1 * (H_roe - np.float32(0.5) * u_roe * u_roe))
    
    # Eigenvalues (wave speeds)
    lam1 = u_roe - a_roe
    lam2 = u_roe
    lam3 = u_roe + a_roe
    
    # Harten entropy fix: smooth |lam_k| < eps via (lam^2 + eps^2)/(2*eps)
    eps = np.float32(0.1) * a_roe
    for_smooth = lambda lam: (lam * lam + eps * eps) / (np.float32(2.0) * eps)
    lam1 = for_smooth(lam1) if abs(lam1) < eps else abs(lam1)
    lam2 = for_smooth(lam2) if abs(lam2) < eps else abs(lam2)
    lam3 = for_smooth(lam3) if abs(lam3) < eps else abs(lam3)
    lam1 = np.float32(lam1); lam2 = np.float32(lam2); lam3 = np.float32(lam3)
    
    # Primitive state differences
    drho = rho_r - rho_l
    du = u_r - u_l
    dp = p_r - p_l
    
    # Wave strengths (mavchin's rho_roe formulation)
    a_roe_sq = a_roe * a_roe
    inv_two_a_sq = np.float32(1.0) / (np.float32(2.0) * a_roe_sq)
    alpha1 = (dp - rho_roe * a_roe * du) * inv_two_a_sq
    alpha2 = drho - dp / a_roe_sq
    alpha3 = (dp + rho_roe * a_roe * du) * inv_two_a_sq
    
    # Physical fluxes F(U) on left and right
    FL_rho  = rho_l * u_l
    FL_rhou = rho_l * u_l * u_l + p_l
    FL_E    = u_l * (E_l + p_l)
    FR_rho  = rho_r * u_r
    FR_rhou = rho_r * u_r * u_r + p_r
    FR_E    = u_r * (E_r + p_r)
    
    # Dissipation D = sum_k lam_k * alpha_k * r_k
    # Right eigenvectors for 1D Euler:
    #   r1 = (1, u-a, H-u*a)
    #   r2 = (1, u,   0.5*u*u)
    #   r3 = (1, u+a, H+u*a)
    D_rho  = lam1 * alpha1 + lam2 * alpha2 + lam3 * alpha3
    D_rhou = lam1 * alpha1 * (u_roe - a_roe) + lam2 * alpha2 * u_roe + lam3 * alpha3 * (u_roe + a_roe)
    D_E    = lam1 * alpha1 * (H_roe - u_roe * a_roe) + lam2 * alpha2 * np.float32(0.5) * u_roe * u_roe + lam3 * alpha3 * (H_roe + u_roe * a_roe)
    
    # Roe flux: F = 0.5*(FL + FR) - 0.5*D
    F_rho  = np.float32(0.5) * (FL_rho + FR_rho)   - np.float32(0.5) * D_rho
    F_rhou = np.float32(0.5) * (FL_rhou + FR_rhou) - np.float32(0.5) * D_rhou
    F_E    = np.float32(0.5) * (FL_E + FR_E)       - np.float32(0.5) * D_E
    
    return F_rho, F_rhou, F_E


#: Convenience alias requested by the verification harness
#: (test_cfd_substrate.py:91). Same function, more specific name.
#: Per medayek's 2026-05-18 ~18:48 UTC note: the harness expected
#: roe_flux_sod as the import symbol; aliasing here is the
#: substrate-honest fix.
roe_flux_sod = roe_flux


def test_roe_flux():
    """Verify the Roe flux on Sod's shock tube initial states."""
    # Sod's shock tube initial conditions
    rho_L, u_L, p_L = 1.0, 0.0, 1.0      # left state
    rho_R, u_R, p_R = 0.125, 0.0, 0.1    # right state
    
    F0, F1, F2 = roe_flux(rho_L, u_L, p_L, rho_R, u_R, p_R)
    
    print(f"Sod's shock tube — Roe flux at the diaphragm (mavchin-matched):")
    print(f"  F_rho (mass)       = {F0:.10g}")
    print(f"  F_rho_u (momentum) = {F1:.10g}")
    print(f"  F_E (energy)       = {F2:.10g}")
    
    # Test with symmetric (zero-jump) state: flux should equal physical flux
    F0s, F1s, F2s = roe_flux(1.0, 0.0, 1.0, 1.0, 0.0, 1.0)
    assert abs(F0s) < 1e-6, f"Mass flux for symmetric state should be 0, got {F0s}"
    assert abs(F1s - 1.0) < 1e-6, f"Momentum flux for symmetric state should be 1.0 (p), got {F1s}"
    assert abs(F2s) < 1e-6, f"Energy flux for symmetric state should be 0, got {F2s}"
    print(f"\nSymmetric state test passed: F = ({F0s}, {F1s}, {F2s})")


if __name__ == "__main__":
    test_roe_flux()
