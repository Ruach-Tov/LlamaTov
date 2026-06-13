%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════════
%% torch_cfd_lifted.pl — Prolog canonical form for torch-cfd kernels
%%
%% Lifted from github.com/scaomath/torch-cfd via systematic parsing.
%% These facts are the CANONICAL representation — all target code
%% (CUDA, LLVM IR, Rust) is generated FROM these facts.
%%
%% Bidirectional:
%%   PyTorch .py → pytorch_to_prolog.py → THIS FILE
%%   THIS FILE → prolog_to_cuda.py → CUDA .cu
%%   THIS FILE → prolog_to_llvm.pl → LLVM IR .ll
%%   THIS FILE → (future) prolog_to_rust.pl → Rust .rs
%%
%% Author: medayek (Collective SME, Verification Methodology)
%% Source: torch-cfd finite_differences.py, spectral.py
%% Licensed under GPLv2
%% ═══════════════════════════════════════════════════════════════════════════

:- module(torch_cfd_lifted, [
    cfd_op/4,
    cfd_composition/3,
    cfd_stability/2
]).


%% clauses grouped by family, not contiguous — declared for warning-free consult.
:- discontiguous cfd_composition/3, cfd_op/4.
%% ═══════════════════════════════════════════════════════════════
%% LAYER 1: Primitive stencil operations (already verified)
%% ═══════════════════════════════════════════════════════════════

%! cfd_op(+Name, +Type, +Inputs, +Output)
%  Declares a CFD operation in canonical form.

% Forward difference: (u[i+1] - u[i]) / dx
cfd_op(forward_diff_x, stencil_1d, [u, dx], du_dx_fwd).
cfd_op(forward_diff_y, stencil_1d, [u, dy], du_dy_fwd).

% Backward difference: (u[i] - u[i-1]) / dx
cfd_op(backward_diff_x, stencil_1d, [u, dx], du_dx_bwd).
cfd_op(backward_diff_y, stencil_1d, [u, dy], du_dy_bwd).

% Central difference: (u[i+1] - u[i-1]) / (2*dx)
cfd_op(central_diff_x, stencil_1d, [u, dx], du_dx_cent).
cfd_op(central_diff_y, stencil_1d, [u, dy], du_dy_cent).

% Laplacian: (u[i-1] - 2*u[i] + u[i+1]) / dx²
cfd_op(laplacian_x, stencil_1d, [u, dx], d2u_dx2).
cfd_op(laplacian_y, stencil_1d, [u, dy], d2u_dy2).

%% ═══════════════════════════════════════════════════════════════
%% LAYER 2: Composed operations (need lifting + verification)
%% ═══════════════════════════════════════════════════════════════

%! cfd_composition(+Name, +Components, +Output)
%  Declares a composed operation built from primitives.

% Divergence: du/dx + dv/dy (forward differences)
% Source: finite_differences.py line 152
cfd_composition(divergence,
    [forward_diff_x(u, dx), forward_diff_y(v, dy), add],
    div_uv).

% Centered divergence: du/dx + dv/dy (central differences)
% Source: finite_differences.py line 163
cfd_composition(centered_divergence,
    [central_diff_x(u, dx), central_diff_y(v, dy), add],
    div_uv_cent).

% 2D Laplacian: d²u/dx² + d²u/dy²
% Source: finite_differences.py line 173
cfd_composition(laplacian_2d,
    [laplacian_x(u, dx), laplacian_y(u, dy), add],
    lap_u).

% Curl 2D: dv/dx - du/dy (forward differences)
% Source: finite_differences.py line 462
cfd_composition(curl_2d,
    [forward_diff_x(v, dx), forward_diff_y(u, dy), sub],
    curl_uv).

% Gradient tensor: [[du/dx, du/dy], [dv/dx, dv/dy]]
% Source: finite_differences.py line 442
cfd_composition(gradient_tensor,
    [forward_diff_x(u, dx), forward_diff_y(u, dy),
     forward_diff_x(v, dx), forward_diff_y(v, dy)],
    grad_uv).

%% ═══════════════════════════════════════════════════════════════
%% LAYER 3: Spectral operations
%% ═══════════════════════════════════════════════════════════════

% Spectral Laplacian: multiply by -(kx² + ky²) in Fourier space
% Source: spectral.py line 41
cfd_op(spectral_laplacian, elementwise_complex, [u_hat, k_sq], lap_hat).

% Spectral curl: i*(kx*vy_hat - ky*vx_hat)
% Source: spectral.py line 48
cfd_op(spectral_curl, elementwise_complex, [vx_hat, vy_hat, kx, ky], curl_hat).

% Spectral divergence: i*(kx*vx_hat + ky*vy_hat)
% Source: spectral.py line 58
cfd_op(spectral_div, elementwise_complex, [vx_hat, vy_hat, kx, ky], div_hat).

% Spectral gradient: [i*kx*u_hat, i*ky*u_hat]
% Source: spectral.py line 67
cfd_op(spectral_grad, elementwise_complex, [u_hat, kx, ky], grad_hat).

% FFT forward/inverse (reduction-sensitive!)
cfd_op(fft_2d, reduction, [u], u_hat).
cfd_op(ifft_2d, reduction, [u_hat], u).

%% ═══════════════════════════════════════════════════════════════
%% LAYER 4: Advection operations (lifted from advection.py)
%%
%% Architecture: 3-level composition
%%   Level A: Primitive interpolators (Upwind, LaxWendroff, Linear)
%%   Level B: Composed interpolators (TVD = Upwind + LaxWendroff + limiter)
%%   Level C: Full advection (interpolate c + v → aligned → flux → divergence)
%% ═══════════════════════════════════════════════════════════════

%% --- Level A: Primitive interpolators ---

% safe_div: x / where(y != 0, y, default)
% Source: advection.py line 42
% Stability: div_by_near_zero (denominator can be zero at extrema)
cfd_op(safe_div, elementwise_guarded, [x, y, default_num], result).

% Van Leer flux limiter: where(r > 0, 2*r / (1+r), 0)
% Source: advection.py line 47
% Uses safe_div internally. Non-negative output.
% Stability: div_by_near_zero via safe_div when r → ∞
cfd_op(van_leer_limiter, elementwise_conditional, [r], phi_r).

% Upwind interpolation: where(u > 0, c_floor, c_ceil)
% Source: advection.py line 52, forward() at line 95
% Pure conditional selection — no arithmetic rounding.
% Should be 0 ULP against any reference.
cfd_op(upwind_interp, stencil_conditional, [c, u, offset_delta], c_interp).

% Lax-Wendroff interpolation: upwind + second-order correction
% Source: advection.py line 128, forward() at line 163
% pos = c_floor + 0.5*(1-courant)*(c_ceil - c_floor)
% neg = c_ceil  - 0.5*(1+courant)*(c_ceil - c_floor)
% result = where(u > 0, pos, neg)
% Stability: catastrophic_cancellation when c_ceil ≈ c_floor (smooth flow)
%            fma_sensitive in the 0.5*(1±courant)*diff chain
cfd_op(lax_wendroff_interp, stencil_2nd_order, [c, u, dx, dt], c_interp_lw).

% Linear interpolation: weighted average along axis
% Source: finite_differences.py line 399 (called from advection)
% c_interp = alpha * c_floor + (1-alpha) * c_ceil
% where alpha = fractional offset distance
% Stability: fma_sensitive (weighted sum)
cfd_op(linear_interp, stencil_weighted, [c, alpha], c_interp_lin).

%% --- Level B: Composed interpolators ---

% TVD interpolation: low + limiter * (high - low)
% Source: advection.py line 343, forward() at line 385
% c_tvd = c_low + phi(r) * (c_high - c_low)
% where:
%   c_low  = upwind_interp(c, v, dt)
%   c_high = lax_wendroff_interp(c, v, dt)
%   r_pos  = (c - c_left) / (c_right - c)       ← gradient ratio
%   r_neg  = (c_next_right - c_right) / (c_right - c)
%   phi    = van_leer_limiter(where(u > 0, r_pos, r_neg))
% Stability: div_by_near_zero in gradient ratio (c_right ≈ c)
%            catastrophic_cancellation in c_high - c_low when they agree
cfd_composition(tvd_interp,
    [upwind_interp(c, v, dt),        % c_low
     lax_wendroff_interp(c, v, dt),  % c_high
     gradient_ratio(c),              % r
     van_leer_limiter(r),            % phi
     blend(c_low, c_high, phi)],     % c_low + phi*(c_high - c_low)
    c_interp_tvd).

%% --- Level C: Full advection operations ---

% AdvectAligned: flux = c * v, then -divergence(flux)
% Source: advection.py line 206, forward() at line 286
% This is the core: once c and v are aligned (same grid offsets),
% compute flux = c*v (element-wise multiply), then negative divergence.
cfd_composition(advect_aligned,
    [mul(c_aligned, v_aligned),    % flux = c * v
     divergence(flux),             % ∇·flux
     neg(div_flux)],               % -∇·flux (advection = negative divergence)
    advection_term).

% Full advection pipeline: interpolate → align → flux → divergence
% Source: advection.py line 422 (AdvectionBase.forward)
% 1. Interpolate v to control volume faces (linear interp)
% 2. Interpolate c to same faces (upwind/tvd/linear depending on scheme)
% 3. Compute aligned advection (flux = c*v, return -div(flux))
cfd_composition(advection_full,
    [linear_interp(v, target_offsets),      % step 1: align velocity
     flux_interp(c, v_aligned, dt),         % step 2: interpolate scalar
     advect_aligned(c_aligned, v_aligned)], % step 3: flux + divergence
    dc_dt_advection).

% Three advection schemes (differ only in flux_interp choice):
%   AdvectionLinear:   flux_interp = linear_interp
%   AdvectionUpwind:   flux_interp = upwind_interp
%   AdvectionVanLeer:  flux_interp = tvd_interp (van Leer limiter)

% ConvectionVector: velocity self-advection u·∇u
% Source: advection.py line 632
% Each component of v is advected by the full v field.
% conv_u = advection(u, v), conv_v = advection(v, v)
cfd_composition(convection_vector,
    [advection_full(u, v, dt),   % advect u-component by v
     advection_full(v, v, dt)],  % advect v-component by v
    conv_uv).

%% ═══════════════════════════════════════════════════════════════
%% LAYER 5: Solver operations (lifted from solvers.py)
%%
%% Architecture: 4 solver families
%%   A. Direct spectral (FFT-based, O(N log N))
%%   B. Jacobi/Gauss-Seidel (stationary iterative)
%%   C. Conjugate Gradient (Krylov, preconditioned)
%%   D. Multigrid (V-cycle with restrict/prolong)
%% ═══════════════════════════════════════════════════════════════

%% --- Family A: Spectral direct solve ---

% Separable Laplacian eigenvalue computation:
% eigenvalues = fft(operator_col_0) for each dimension
% summed = outer_sum(eigenvalues)
% inverse_diag = 1 / summed (with k=0 filtered)
% Source: solvers.py line 208 (PseudoInverseFFT)
cfd_op(compute_eigenvalues, fft_per_dim, [operators], eigenvalues).
cfd_op(outer_sum, reduction, [eigenvalues], summed_eigenvalues).
cfd_op(filter_eigenvalues, elementwise_guarded, [summed_eigenvalues], inverse_diag).

% Spectral Poisson solve: p = ifft(inverse_diag * fft(rhs))
% Source: solvers.py line 243 (PseudoInverseFFT.forward)
% This is O(N log N) — the fast path for periodic domains.
% Stability: reduction_sensitive (FFT), div_by_near_zero (k=0 mode)
cfd_composition(poisson_solve_fft,
    [fft_2d(rhs),                        % transform RHS to spectral
     mul(rhs_hat, inverse_diag),         % element-wise multiply
     ifft_2d(result_hat)],               % transform back to physical
    pressure).

%% --- Family B: Jacobi / Gauss-Seidel (stationary iterative) ---

% Apply separable Laplacian: Au = Lx @ u + u @ Ly^T
% Source: solvers.py line 466 (_apply_laplacian)
% Uses tensordot for each dimension — reduction_sensitive
cfd_op(apply_laplacian_sep, matmul_separable, [u, lx, ly], au).

% Residual: r = f - Au
% Source: solvers.py line 481
cfd_composition(compute_residual,
    [apply_laplacian_sep(u, lx, ly),     % Au
     sub(f, au)],                         % f - Au
    residual).

% Jacobi update: u_new = D^{-1} * (f - off_diag(A) * u)
% Source: solvers.py line 578 (Jacobi.update)
% Gathers 4 neighbors (left,right,down,up) weighted by operator diagonals
% Then: update = f - sum(neighbors), u_new = inv_diag * update
% Stability: div_by_near_zero (inv_diag near boundaries)
%            reduction_sensitive (neighbor sum, 4 terms)
cfd_op(jacobi_gather_neighbors, stencil_4neighbor, [u, lx, ly], neighbor_sum).
cfd_composition(jacobi_step,
    [jacobi_gather_neighbors(u, lx, ly),
     sub(f, neighbor_sum),                % update = f - neighbors
     mul(inv_diag, update)],              % u_new = D^{-1} * update
    u_new).

% Gauss-Seidel: same as Jacobi but uses red-black ordering
% Source: solvers.py line 614 (GaussSeidel extends Jacobi with checkerboard masks)
% Two masks: red cells (i+j even), black cells (i+j odd)
% Update red first using old black, then black using new red.
% Stability: same as Jacobi, plus ordering affects convergence rate
cfd_op(gauss_seidel_step, stencil_4neighbor_ordered, [u, f, lx, ly, mask], u_new_gs).

% Pure Neumann correction: u -= mean(u)
% Source: solvers.py line 607
% Stability: reduction_sensitive (mean over full grid)
cfd_op(neumann_mean_subtract, reduction, [u], u_corrected).

%% --- Family C: Conjugate Gradient ---

% CG one step (preconditioned):
% Source: solvers.py line 706 (ConjugateGradient.forward)
%
% Ap = apply_laplacian(p)
% pAp = sum(p * Ap)                   ← inner product (reduction!)
% alpha = rsold / (pAp + eps)         ← div_by_near_zero
% u += alpha * p                      ← solution update
% r -= alpha * Ap                     ← residual update
% z = preconditioner(r)               ← preconditioning (Jacobi or GS)
% rznew = sum(r * z)                  ← inner product (reduction!)
% beta = rznew / (rsold + eps)        ← div_by_near_zero
% p = z + beta * p                    ← search direction update

cfd_op(cg_inner_product, reduction, [a, b], dot_ab).
cfd_op(cg_alpha, elementwise_guarded, [rsold, pap, eps], alpha).
cfd_op(cg_beta, elementwise_guarded, [rznew, rsold, eps], beta).

cfd_composition(cg_step,
    [apply_laplacian_sep(p, lx, ly),          % Ap
     cg_inner_product(p, ap),                  % pAp = p·Ap
     cg_alpha(rsold, pap, eps),                % alpha = rsold / pAp
     mul_add(u, alpha, p),                     % u += alpha * p
     mul_sub(r, alpha, ap),                    % r -= alpha * Ap
     precondition(r),                          % z = M^{-1} r
     cg_inner_product(r, z),                   % rznew = r·z
     cg_beta(rznew, rsold, eps),               % beta = rznew / rsold
     mul_add(z, beta, p)],                     % p = z + beta * p
    cg_state).  % returns (u, r, p, rznew)

%% --- Family D: Multigrid V-cycle ---

% Restriction (fine → coarse): full-weighting 2D
% Source: solvers.py line 821 (MultigridSolver.restrict)
% r_coarse = 0.25 * (r[::2,::2] + r[1::2,::2] + r[::2,1::2] + r[1::2,1::2])
% This is a 4-point average at strided positions.
% Stability: reduction_sensitive (4-term average, mild)
cfd_op(mg_restrict, downsample_average, [r_fine], r_coarse).

% Prolongation (coarse → fine): transpose of restriction
% Source: solvers.py line 831 (MultigridSolver.prolong)
% Distributes each coarse point to a 2×2 fine grid region.
cfd_op(mg_prolong, upsample_distribute, [e_coarse], e_fine).

% Coarse direct solve: solve A_coarse * e = r via dense linear algebra
% Source: solvers.py line 867 (_coarse_solve, uses torch.linalg.solve)
% Stability: depends on condition number of A_coarse
cfd_op(mg_coarse_solve, dense_solve, [a_coarse, r_coarse], e_coarse).

% V-cycle: the recursive multigrid algorithm
% Source: solvers.py line 874 (MultigridSolver.v_cycle)
%
% v_cycle(level, f, u):
%   1. Pre-smooth: jacobi/GS iterations
%   2. Residual: r = f - A*u
%   3. If coarsest: direct solve
%      Else: restrict r → recurse → prolong correction
%   4. Correct: u += prolongated error
%   5. Post-smooth: jacobi/GS iterations
cfd_composition(mg_v_cycle,
    [gauss_seidel_step(u, f),             % pre-smooth
     compute_residual(f, u),               % r = f - Au
     mg_restrict(r),                       % r_c = restrict(r)
     mg_v_cycle_or_solve(r_c),            % e_c = recurse or direct
     mg_prolong(e_c),                      % e = prolong(e_c)
     add(u, e),                            % u += correction
     gauss_seidel_step(u_corrected, f)],  % post-smooth
    u_final).

%% ═══════════════════════════════════════════════════════════════
%% LAYER 6: Finite Volume Method — full solver compositions
%% (lifted from fvm.py)
%%
%% Architecture: Chorin's projection method on MAC grid
%%   1. Compute explicit terms (advection + diffusion + forcing)
%%   2. Time-step via Runge-Kutta
%%   3. Project velocity to divergence-free
%%
%% The fvm module orchestrates ALL lower layers.
%% ═══════════════════════════════════════════════════════════════

%% --- Pressure Projection ---

% Divergence-free projection: u_projected = u - ∇p
% Source: fvm.py line 303 (PressureProjection.forward)
%
% Steps:
%   1. rhs = divergence(v)               ← forward_diff divergence
%   2. rhs_transformed = transform(rhs)  ← boundary handling
%   3. q = solve(rhs_transformed)         ← Poisson solve (FFT/CG/multigrid)
%   4. q = impose_bc(q)                   ← pressure boundary conditions
%   5. grad_q = forward_difference(q)     ← pressure gradient
%   6. v_proj = v - grad_q                ← subtract gradient
%
% Stability: inherits from solver choice:
%   FFT: reduction_sensitive + div_by_near_zero
%   CG:  reduction_sensitive + div_by_near_zero (inner products)
%   Multigrid: reduction_sensitive (smoothing + restriction)
cfd_composition(pressure_projection,
    [divergence(v),                        % rhs = ∇·v
     transform_rhs(rhs, bc),               % boundary treatment
     poisson_solve(rhs_t, q0),             % solve ∇²q = rhs
     impose_bc(q, pressure_bc),            % apply pressure BC
     forward_diff_all(q),                  % ∇q (gradient)
     sub_vector(v, grad_q)],               % v - ∇q
    v_projected).

%% --- Explicit Terms (RHS of the ODE) ---

% Convection: -u·∇u (negative advection of velocity by itself)
% Source: fvm.py line 473 (_explicit_terms, calls _convect)
% Uses AdvectionVanLeer or AdvectionUpwind from advection module.
cfd_composition(convection_term,
    [convection_vector(v, v, dt)],         % from advection layer
    conv).

% Diffusion: ν∇²u (viscous term)
% Source: fvm.py line 459 (_diffusion)
% alpha = viscosity / density
% lap_v = alpha * laplacian(v) for each component
% Stability: catastrophic_cancellation (laplacian), overflow_risk (high Re)
cfd_composition(diffusion_term,
    [laplacian_2d(u),                      % ∇²u
     laplacian_2d(v_comp),                 % ∇²v
     scalar_mul(alpha, lap_u),             % ν/ρ * ∇²u
     scalar_mul(alpha, lap_v)],            % ν/ρ * ∇²v
    diffusion).

% Forcing: external body forces (e.g., Kolmogorov, pressure gradient)
% Source: fvm.py line 476 (forcing eval)
cfd_op(forcing_eval, external_function, [grid, v, t], force).

% Drag: -drag * v (linear damping)
% Source: fvm.py line 478
cfd_op(drag_term, elementwise, [v, drag_coeff], drag).

% Full explicit terms: du/dt = -convection + diffusion + forcing - drag
% Source: fvm.py line 467 (_explicit_terms)
cfd_composition(explicit_terms,
    [convection_term(v, dt),               % -u·∇u
     diffusion_term(v, viscosity),          % ν∇²u
     forcing_eval(grid, v, t),             % external forces
     drag_term(v, drag_coeff),             % linear damping
     sum_all(conv, diff, force, drag)],    % dv/dt
    dv_dt).

%% --- Runge-Kutta Time Stepping ---

% Butcher tableaux (sweepable parameter!)
% Source: fvm.py line 78 (RKStepper._METHOD_MAP)
%
% rk_method ∈ {forward_euler, midpoint, heun_rk2, classic_rk4}
%
% forward_euler: b = [1.0]                          (1 stage, 1st order)
% midpoint:      a = [[1/2]], b = [0, 1.0]          (2 stages, 2nd order)
% heun_rk2:      a = [[1.0]], b = [1/2, 1/2]        (2 stages, 2nd order)
% classic_rk4:   a = [[1/2],[0,1/2],[0,0,1]],       (4 stages, 4th order)
%                b = [1/6, 1/3, 1/3, 1/6]
%
% Each stage: k[i] = explicit_terms(u_stage)
%             u_stage = u0 + dt * sum(a[i][j] * k[j])
%             u_stage = pressure_projection(u_stage)
% Final:      u_new = u0 + dt * sum(b[j] * k[j])
%             u_new = pressure_projection(u_new)

:- dynamic rk_tableau/3.  % rk_tableau(Name, A_matrix, B_vector)
rk_tableau(forward_euler, [], [1.0]).
rk_tableau(midpoint, [[0.5]], [0.0, 1.0]).
rk_tableau(heun_rk2, [[1.0]], [0.5, 0.5]).
rk_tableau(classic_rk4, [[0.5],[0.0,0.5],[0.0,0.0,1.0]], [0.167,0.333,0.333,0.167]).

% One RK step (the full time integration)
% Source: fvm.py line 183 (RKStepper.forward)
% Stability: accumulates FMA from weighted sums of stages
%            each stage has a pressure projection (solver call)
cfd_composition(rk_step,
    [explicit_terms(u0, dt, t),             % k[0]
     rk_stage_loop(u0, k, alpha, dt),      % intermediate stages
     rk_final_combine(u0, k, beta, dt),    % u* = u0 + dt*sum(b*k)
     pressure_projection(u_star)],          % enforce ∇·u = 0
    u_new).

%% --- The Full Navier-Stokes 2D FVM Solver ---

% NavierStokes2DFVMProjection: one complete time step
% Source: fvm.py line 364 (NavierStokes2DFVMProjection)
%
% This is the TOP-LEVEL composition that orchestrates everything:
%   explicit_terms = -convection + diffusion + forcing - drag
%   time_step via RK (forward_euler / midpoint / heun_rk2 / classic_rk4)
%   pressure_projection via solver (FFT / CG / multigrid)
%
% Sweepable parameters at this level:
%   rk_method ∈ {forward_euler, midpoint, heun_rk2, classic_rk4}
%   solver_method ∈ {fft, cg, multigrid, jacobi}
%   advection_scheme ∈ {linear, upwind, van_leer}
%   viscosity (physical parameter)
%   drag (damping coefficient)
%   forcing (external body force)
cfd_composition(navier_stokes_2d_fvm_step,
    [rk_step(u, dt, explicit_terms, pressure_projection)],
    u_next).

% Navier-Stokes 2D Spectral: one time step (alternative formulation)
% Source: spectral.py line 463
% Uses IMEX (implicit-explicit) splitting:
%   Implicit: Crank-Nicolson on diffusion (spectral, exact)
%   Explicit: RK4 on advection + forcing
cfd_composition(navier_stokes_2d_spectral_step,
    [fft_2d(u), fft_2d(v),
     spectral_curl(vx_hat, vy_hat, kx, ky),
     spectral_laplacian(curl_hat, k_sq),
     crank_nicolson_diffusion(u_hat, k_sq, dt, nu),
     ifft_2d(result_hat)],
    u_new_spectral).

%% ═══════════════════════════════════════════════════════════════
%% STABILITY ANNOTATIONS (from numerical_stability.pl analysis)
%% ═══════════════════════════════════════════════════════════════

%! cfd_stability(+OpName, +Warning)
cfd_stability(laplacian_x, catastrophic_cancellation).
cfd_stability(laplacian_y, catastrophic_cancellation).
cfd_stability(laplacian_2d, catastrophic_cancellation).
cfd_stability(fft_2d, reduction_sensitive).
cfd_stability(ifft_2d, reduction_sensitive).
cfd_stability(pressure_poisson_fft, div_by_near_zero).  % k=0 mode
cfd_stability(van_leer_limiter, div_by_near_zero).      % gradient ratio
cfd_stability(safe_div, div_by_near_zero).              % guarded division
cfd_stability(lax_wendroff_interp, catastrophic_cancellation). % c_ceil ≈ c_floor
cfd_stability(lax_wendroff_interp, fma_sensitive).      % 0.5*(1±courant)*diff
cfd_stability(linear_interp, fma_sensitive).             % weighted sum
cfd_stability(tvd_interp, div_by_near_zero).            % gradient ratio in limiter
cfd_stability(tvd_interp, catastrophic_cancellation).   % c_high ≈ c_low
cfd_stability(cg_inner_product, reduction_sensitive).     % grid-scale dot product
cfd_stability(cg_alpha, div_by_near_zero).                % alpha = rsold/pAp
cfd_stability(cg_beta, div_by_near_zero).                 % beta = rznew/rsold
cfd_stability(jacobi_step, fma_sensitive).
cfd_stability(jacobi_gather_neighbors, reduction_sensitive). % 4-neighbor sum
cfd_stability(apply_laplacian_sep, reduction_sensitive).   % tensordot per dimension
cfd_stability(neumann_mean_subtract, reduction_sensitive). % mean over full grid
cfd_stability(poisson_solve_fft, reduction_sensitive).     % FFT
cfd_stability(poisson_solve_fft, div_by_near_zero).        % k=0 eigenvalue
cfd_stability(mg_restrict, reduction_sensitive).            % 4-point average
cfd_stability(mg_coarse_solve, fma_sensitive).             % dense linear solve
% FVM layer stability
cfd_stability(pressure_projection, reduction_sensitive).    % inherits from solver
cfd_stability(diffusion_term, catastrophic_cancellation).  % laplacian inside
cfd_stability(diffusion_term, overflow_risk).               % high Re → ν small, ∇²u large
cfd_stability(rk_step, fma_sensitive).                     % weighted stage combination
cfd_stability(rk_step, reduction_sensitive).                % each stage has projection

%% ═══════════════════════════════════════════════════════════════
%% VERIFICATION STATUS (empirical measurements)
%% ═══════════════════════════════════════════════════════════════

:- dynamic verified/3.  % verified(OpName, MaxULP, Reference)

verified(forward_diff_x, 0, pytorch_cpu).
verified(forward_diff_y, 0, pytorch_cpu).
verified(backward_diff_x, 0, pytorch_cpu).
verified(backward_diff_y, 0, pytorch_cpu).
verified(central_diff_x, 0, pytorch_cpu).
verified(central_diff_y, 0, pytorch_cpu).
verified(laplacian_x, 256, pytorch_cpu).   % catastrophic_cancellation
verified(divergence, 0, pytorch_cpu).
verified(spectral_laplacian, 0, pytorch_cpu).
verified(spectral_curl, 0, pytorch_cpu).
verified(spectral_div, 0, pytorch_cpu).
verified(spectral_grad, 0, pytorch_cpu).
