# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Sod's Shock Tube — Exact Riemann Solver + Reference Implementation

The Sod shock tube problem (1978) is a standard 1D CFD benchmark with
an EXACT analytical solution. It tests: contact discontinuity, shock
wave, rarefaction fan.

Initial conditions:
  Left state  (x < 0.5): rho=1.0,  p=1.0,  u=0.0
  Right state (x > 0.5): rho=0.125, p=0.1, u=0.0
  gamma = 1.4 (ideal gas)

At t=0.2, the exact solution has:
  - Left rarefaction fan
  - Contact discontinuity
  - Right-moving shock

This module provides:
  1. exact_riemann_solution() — analytical solution at any time
  2. sod_initial_conditions() — IC arrays for numerical solver
  3. godunov_flux() — reference Godunov flux (exact Riemann at cell interface)
  4. euler_step() — one timestep of 1D Euler equations
  5. run_sod() — full simulation, returns (rho, u, p) at final time

Author: medayek (Collective SME, Verification Methodology)
Date: 2026-05-18
Per mavchin's request for CFD benchmark with exact analytical solution.
"""

import numpy as np
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════
# Physical parameters
# ═══════════════════════════════════════════════════════════════════════

GAMMA = 1.4  # Ratio of specific heats (ideal diatomic gas)

# Sod's initial conditions
RHO_L, U_L, P_L = 1.0, 0.0, 1.0      # Left state
RHO_R, U_R, P_R = 0.125, 0.0, 0.1    # Right state

# Domain
X_MIN, X_MAX = 0.0, 1.0
X_DIAPHRAGM = 0.5
T_FINAL = 0.2

# Default resolution
N_CELLS = 256


# ═══════════════════════════════════════════════════════════════════════
# Exact Riemann solver
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ExactSolution:
    """Exact solution at a given time."""
    x: np.ndarray
    rho: np.ndarray
    u: np.ndarray
    p: np.ndarray
    e: np.ndarray  # specific internal energy


def sound_speed(p, rho):
    """Sound speed for ideal gas."""
    return np.sqrt(GAMMA * p / rho)


def exact_riemann_solution(x, t, x0=X_DIAPHRAGM,
                            rho_l=RHO_L, u_l=U_L, p_l=P_L,
                            rho_r=RHO_R, u_r=U_R, p_r=P_R):
    """Compute exact Riemann solution at positions x and time t.
    
    Uses the iterative procedure from Toro (2009) Chapter 4.
    """
    if t <= 0:
        rho = np.where(x < x0, rho_l, rho_r)
        u = np.where(x < x0, u_l, u_r)
        p = np.where(x < x0, p_l, p_r)
        e = p / (rho * (GAMMA - 1))
        return ExactSolution(x=x, rho=rho, u=u, p=p, e=e)
    
    # Sound speeds
    a_l = sound_speed(p_l, rho_l)
    a_r = sound_speed(p_r, rho_r)
    
    # Solve for p_star (pressure in star region) via Newton iteration
    gm1 = GAMMA - 1
    gp1 = GAMMA + 1
    
    def f(p, rho_k, p_k, a_k):
        """Pressure function for Newton iteration."""
        if p > p_k:
            # Shock
            A = 2.0 / (gp1 * rho_k)
            B = gm1 / gp1 * p_k
            return (p - p_k) * np.sqrt(A / (p + B))
        else:
            # Rarefaction
            return 2 * a_k / gm1 * ((p / p_k) ** (gm1 / (2 * GAMMA)) - 1)
    
    def df(p, rho_k, p_k, a_k):
        """Derivative of pressure function."""
        if p > p_k:
            A = 2.0 / (gp1 * rho_k)
            B = gm1 / gp1 * p_k
            return np.sqrt(A / (p + B)) * (1 - (p - p_k) / (2 * (p + B)))
        else:
            return 1.0 / (rho_k * a_k) * (p / p_k) ** (-(gp1) / (2 * GAMMA))
    
    # Initial guess (linearized)
    p_star = 0.5 * (p_l + p_r)
    
    # Newton iteration
    for _ in range(50):
        fl = f(p_star, rho_l, p_l, a_l)
        fr = f(p_star, rho_r, p_r, a_r)
        dfl = df(p_star, rho_l, p_l, a_l)
        dfr = df(p_star, rho_r, p_r, a_r)
        
        dp = -(fl + fr + (u_r - u_l)) / (dfl + dfr)
        p_star = max(1e-10, p_star + dp)
        
        if abs(dp) < 1e-12 * p_star:
            break
    
    # Star region velocity
    u_star = 0.5 * (u_l + u_r) + 0.5 * (fr - fl)
    
    # Sample solution at x/t
    rho = np.zeros_like(x, dtype=np.float64)
    u = np.zeros_like(x, dtype=np.float64)
    p = np.zeros_like(x, dtype=np.float64)
    
    s = (x - x0) / t  # similarity variable
    
    for i in range(len(x)):
        si = s[i]
        
        if si < u_star:
            # Left of contact
            if p_star <= p_l:
                # Left rarefaction
                s_hl = u_l - a_l  # head speed
                s_tl = u_star - a_l * (p_star / p_l) ** (gm1 / (2 * GAMMA))  # tail speed
                
                if si <= s_hl:
                    rho[i], u[i], p[i] = rho_l, u_l, p_l
                elif si >= s_tl:
                    rho[i] = rho_l * (p_star / p_l) ** (1 / GAMMA)
                    u[i] = u_star
                    p[i] = p_star
                else:
                    # Inside fan
                    u[i] = 2 / gp1 * (a_l + gm1 / 2 * u_l + si)
                    a = 2 / gp1 * (a_l - gm1 / 2 * (si - u_l))
                    rho[i] = rho_l * (a / a_l) ** (2 / gm1)
                    p[i] = p_l * (a / a_l) ** (2 * GAMMA / gm1)
            else:
                # Left shock
                s_l = u_l - a_l * np.sqrt((gp1 * p_star / p_l + gm1) / (2 * GAMMA))
                if si <= s_l:
                    rho[i], u[i], p[i] = rho_l, u_l, p_l
                else:
                    rho[i] = rho_l * ((p_star / p_l + gm1 / gp1) /
                                       (gm1 / gp1 * p_star / p_l + 1))
                    u[i] = u_star
                    p[i] = p_star
        else:
            # Right of contact
            if p_star <= p_r:
                # Right rarefaction
                s_hr = u_r + a_r
                s_tr = u_star + a_r * (p_star / p_r) ** (gm1 / (2 * GAMMA))
                
                if si >= s_hr:
                    rho[i], u[i], p[i] = rho_r, u_r, p_r
                elif si <= s_tr:
                    rho[i] = rho_r * (p_star / p_r) ** (1 / GAMMA)
                    u[i] = u_star
                    p[i] = p_star
                else:
                    u[i] = 2 / gp1 * (-a_r + gm1 / 2 * u_r + si)
                    a = 2 / gp1 * (a_r + gm1 / 2 * (si - u_r))
                    rho[i] = rho_r * (a / a_r) ** (2 / gm1)
                    p[i] = p_r * (a / a_r) ** (2 * GAMMA / gm1)
            else:
                # Right shock
                s_r = u_r + a_r * np.sqrt((gp1 * p_star / p_r + gm1) / (2 * GAMMA))
                if si >= s_r:
                    rho[i], u[i], p[i] = rho_r, u_r, p_r
                else:
                    rho[i] = rho_r * ((p_star / p_r + gm1 / gp1) /
                                       (gm1 / gp1 * p_star / p_r + 1))
                    u[i] = u_star
                    p[i] = p_star
    
    e = p / (rho * (GAMMA - 1))
    return ExactSolution(x=x, rho=rho, u=u, p=p, e=e)


# ═══════════════════════════════════════════════════════════════════════
# Numerical solver (Godunov's method — reference implementation)
# ═══════════════════════════════════════════════════════════════════════

def sod_initial_conditions(n=N_CELLS):
    """Generate initial condition arrays."""
    dx = (X_MAX - X_MIN) / n
    x = np.linspace(X_MIN + dx/2, X_MAX - dx/2, n)  # cell centers
    
    rho = np.where(x < X_DIAPHRAGM, RHO_L, RHO_R)
    u = np.where(x < X_DIAPHRAGM, U_L, U_R)
    p = np.where(x < X_DIAPHRAGM, P_L, P_R)
    
    return x, rho, u, p


def conservative_from_primitive(rho, u, p):
    """Convert primitive (rho, u, p) to conservative (rho, rho*u, E)."""
    E = p / (GAMMA - 1) + 0.5 * rho * u**2
    return np.stack([rho, rho * u, E])


def primitive_from_conservative(U):
    """Convert conservative to primitive."""
    rho = U[0]
    u = U[1] / rho
    p = (GAMMA - 1) * (U[2] - 0.5 * rho * u**2)
    return rho, u, p


def flux(U):
    """Euler flux from conservative variables."""
    rho, u, p = primitive_from_conservative(U)
    E = U[2]
    return np.stack([rho * u, rho * u**2 + p, (E + p) * u])


def euler_step(U, dx, dt):
    """One Godunov timestep using exact Riemann solver at interfaces.
    
    THIS is the kernel that maps to GPU: for each cell interface,
    solve the Riemann problem and compute the flux.
    """
    n = U.shape[1]
    F = np.zeros((3, n + 1))  # fluxes at interfaces
    
    for i in range(n + 1):
        if i == 0:
            # Left boundary (transmissive)
            rho_l, u_l, p_l = primitive_from_conservative(U[:, 0])
            rho_r, u_r, p_r = rho_l, u_l, p_l
        elif i == n:
            # Right boundary (transmissive)
            rho_l, u_l, p_l = primitive_from_conservative(U[:, -1])
            rho_r, u_r, p_r = rho_l, u_l, p_l
        else:
            rho_l, u_l, p_l = primitive_from_conservative(U[:, i-1])
            rho_r, u_r, p_r = primitive_from_conservative(U[:, i])
        
        # Solve Riemann problem at x/t = 0 (interface)
        sol = exact_riemann_solution(
            np.array([0.0]), 1.0, x0=0.0,
            rho_l=rho_l, u_l=u_l, p_l=p_l,
            rho_r=rho_r, u_r=u_r, p_r=p_r
        )
        
        # Flux from the sampled state
        rho_s, u_s, p_s = sol.rho[0], sol.u[0], sol.p[0]
        E_s = p_s / (GAMMA - 1) + 0.5 * rho_s * u_s**2
        F[0, i] = rho_s * u_s
        F[1, i] = rho_s * u_s**2 + p_s
        F[2, i] = (E_s + p_s) * u_s
    
    # Conservative update
    U_new = U - dt / dx * (F[:, 1:] - F[:, :-1])
    return U_new


def run_sod(n=N_CELLS, t_final=T_FINAL, cfl=0.8):
    """Run the Sod shock tube to t_final.
    
    Returns (x, rho, u, p) at final time.
    """
    dx = (X_MAX - X_MIN) / n
    x, rho, u, p = sod_initial_conditions(n)
    U = conservative_from_primitive(rho, u, p)
    
    t = 0.0
    step = 0
    while t < t_final:
        # CFL condition
        rho, u, p = primitive_from_conservative(U)
        a = sound_speed(p, rho)
        s_max = np.max(np.abs(u) + a)
        dt = min(cfl * dx / s_max, t_final - t)
        
        U = euler_step(U, dx, dt)
        t += dt
        step += 1
    
    rho, u, p = primitive_from_conservative(U)
    return x, rho, u, p


# ═══════════════════════════════════════════════════════════════════════
# Verification
# ═══════════════════════════════════════════════════════════════════════

def verify_against_exact(n=N_CELLS, t_final=T_FINAL):
    """Run numerical solver and compare against exact solution.
    
    Returns L1 error norms for rho, u, p.
    """
    # Numerical solution
    x, rho_num, u_num, p_num = run_sod(n, t_final)
    
    # Exact solution
    exact = exact_riemann_solution(x, t_final)
    
    # L1 errors
    dx = (X_MAX - X_MIN) / n
    err_rho = np.sum(np.abs(rho_num - exact.rho)) * dx
    err_u = np.sum(np.abs(u_num - exact.u)) * dx
    err_p = np.sum(np.abs(p_num - exact.p)) * dx
    
    return {
        'n': n,
        'err_rho': err_rho,
        'err_u': err_u,
        'err_p': err_p,
        'x': x,
        'numerical': (rho_num, u_num, p_num),
        'exact': (exact.rho, exact.u, exact.p),
    }


if __name__ == "__main__":
    print("Sod's Shock Tube — Verification")
    print("=" * 50)
    
    for n in [64, 128, 256, 512]:
        result = verify_against_exact(n)
        print(f"  N={n:4d}: L1(rho)={result['err_rho']:.6f}  "
              f"L1(u)={result['err_u']:.6f}  L1(p)={result['err_p']:.6f}")
    
    print("\n  Expected: L1 errors decrease ~O(1/sqrt(N)) for Godunov scheme")
    print("  The SAME numerical solution on GPU must match CPU bit-for-bit")
