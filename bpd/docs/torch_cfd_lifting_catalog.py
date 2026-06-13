# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""torch-cfd Kernel Surface Catalog — systematic lifting plan.

Source: github.com/scaomath/torch-cfd (PyTorch port of Google's JAX-CFD)
Target: Prolog canonical form → CUDA / Rust / LLVM IR

Author: medayek (Collective SME, Verification Methodology)
Date: 2026-05-26

KERNEL SURFACE: 11 modules, ~80 liftable operations
====================================================

Module 1: finite_differences.py (CORE — lift first)
  Status: 7/8 stencils BIT_IDENTICAL in our test suite
  Operations:
    forward_difference(u, dim)     ← LIFTED (0 ULP)
    backward_difference(u, dim)    ← LIFTED (0 ULP)
    central_difference(u, dim)     ← LIFTED (0 ULP)
    laplacian(u)                   ← LIFTED (256 ULP — cancellation)
    divergence(v)                  ← LIFTED (0 ULP)
    centered_divergence(v)         ← need to lift
    gradient_tensor(v)             ← need to lift
    curl_2d(v)                     ← need to lift
    linear(c, offset, dim)         ← need to lift (interpolation)
    stencil_sum(*arrays)           ← utility
    trim_boundary(u)               ← utility

Module 2: spectral.py (CORE — lift second)
  Status: 8/11 spectral ops BIT_IDENTICAL
  Operations:
    spectral_laplacian_2d(fft_mesh)   ← LIFTED (0 ULP)
    spectral_curl_2d(vhat, mesh)      ← LIFTED (0 ULP)
    spectral_div_2d(vhat, mesh)       ← LIFTED (0 ULP)
    spectral_grad_2d(vhat, mesh)      ← LIFTED (0 ULP)
    spectral_rot_2d(vhat, mesh)       ← need to lift
    vorticity_to_velocity(w, grid)    ← need to lift
    brick_wall_filter_2d(grid)        ← need to lift
    stable_time_step(grid, v, visc)   ← need to lift (has CFL reduction)
    fft_mesh_2d(n, diam)              ← utility
    NavierStokes2DSpectral            ← full solver class
    IMEXStepper / RK4CrankNicolson   ← time integration

Module 3: advection.py (MEDIUM — lift third)
  Status: not started
  Operations:
    van_leer_limiter(r)             ← element-wise, should be 0 ULP
    Upwind.forward(u, v)            ← stencil + conditional
    LaxWendroff.forward(u, v)       ← stencil + quadratic
    AdvectAligned.forward(u, v)     ← semi-Lagrangian interpolation
    LinearInterpolation.forward()   ← interpolation kernel
    TVDInterpolation.forward()      ← limiter + interpolation
    AdvectionLinear.forward()       ← composition
    AdvectionUpwind.forward()       ← composition
    AdvectionVanLeer.forward()      ← composition
    ConvectionVector.forward()      ← velocity self-advection

Module 4: solvers.py (HARD — lift fourth)
  Status: not started
  Operations:
    PseudoInverseFFT.forward()      ← FFT-based Poisson solve (key!)
    PseudoInverseRFFT.forward()     ← real-FFT variant
    PseudoInverseMatmul.forward()   ← direct solve
    Jacobi.forward()                ← iterative solver
    GaussSeidel.forward()           ← iterative solver
    ConjugateGradient.forward()     ← CG solver (most complex)
    MultigridSolver.forward()       ← multigrid V/W cycle
    outer_sum(x)                    ← utility

Module 5: fvm.py (COMPOSITION — lift fifth)
  Status: not started
  Operations:
    ProjectionExplicitODE           ← pressure projection framework
    RKStepper                       ← RK time stepping
    PressureProjection              ← divergence-free projection
    NavierStokes2DFVMProjection     ← full FVM solver

Module 6: boundaries.py (INFRASTRUCTURE — lift as needed)
  Status: not started
  Many boundary condition types, pad operations

Module 7: grids.py (INFRASTRUCTURE)
  Status: partially understood
  Grid/GridVariable/GridVariableVector data structures

Module 8: forcings.py (DOMAIN-SPECIFIC)
  Kolmogorov forcing, pressure gradient forcing

Module 9: initial_conditions.py (DOMAIN-SPECIFIC)
  Random velocity fields, McWilliams spectrum

Module 10: tensor_utils.py (UTILITY)
  Slice/split operations

LIFTING METHODOLOGY:
====================

For EACH operation:
  1. Parse PyTorch source → extract tensor ops + shapes + dtypes
  2. Emit Prolog facts: kernel_op/4, stencil_expr/4, etc.
  3. Run numerical_stability.pl → flag hazards
  4. Generate target code (CUDA via c_ast, LLVM IR via prolog_to_llvm)
  5. Verify bit-identical against PyTorch output
  6. Record ULP + substrate-design parameters in atlas

BIDIRECTIONAL via Prolog:
  PyTorch .py → pytorch_to_prolog.py → Prolog facts
  Prolog facts → prolog_to_cuda.py → CUDA .cu
  Prolog facts → prolog_to_llvm.pl → LLVM IR .ll
  Prolog facts → (future) prolog_to_rust.pl → Rust .rs

The Prolog facts ARE the canonical form.
The Python parser and code generators are PROJECTIONS.
Same Prolog facts → any target language.

PRIORITY ORDER:
  1. finite_differences (4 remaining ops)
  2. spectral (3 remaining ops + full solver)
  3. advection (10 ops — the hard domain)
  4. solvers (7 ops — iterative methods)
  5. fvm (4 ops — composition layer)
  6. boundaries/grids/forcings/IC (as needed)

STABILITY WARNINGS (from our analyzer):
  - Laplacian: catastrophic_cancellation (known, 256 ULP)
  - FFT round-trip: reduction_sensitive (6777 ULP)
  - Poisson solve: div_by_near_zero (k=0 mode)
  - CFL reduction: reduction_sensitive (grid-scale sum)
  - van_leer_limiter: div_by_near_zero (gradient ratio at extrema)
  - CG solver: reduction_sensitive (inner products)
"""
