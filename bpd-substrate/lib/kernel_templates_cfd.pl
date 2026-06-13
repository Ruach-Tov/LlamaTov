%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% =============================================================================
%% kernel_templates_cfd.pl — CFD kernel templates
%% =============================================================================
%%
%% Substrate for emitting CUDA kernels for Computational Fluid Dynamics.
%% Mirrors the structure of kernel_templates_llama.pl (ML kernels) but
%% expresses the CFD domain: stencil operators, flux computations, time
%% stepping, and reduction-based CFL conditions.
%%
%% PER 2026-05-18 ~18:30 UTC, METHODOLOGY FRAMING (Heath + mavchin):
%%
%% Unlike the ML substrate (which subsumes llama.cpp and uses bug-for-bug
%% compatibility as evidence of comprehension), the CFD substrate verifies
%% against ANALYTICAL physics — Sod's shock tube has a known closed-form
%% Riemann solution. The reference IS the physics, not another code base.
%%
%% This is a STRONGER claim than bug-for-bug compatibility:
%%   - For ML/llama.cpp: "what was the originally intended behavior?" →
%%     "what llama.cpp does, including bugs we then optionally fix"
%%   - For CFD/analytical: "what was the originally intended behavior?" →
%%     "what the physics says, full stop"
%%
%% No fixes catalog needed for CFD-against-physics — there's no implementation
%% to inherit defects from. The fix-flag pattern (kernel_available_fixes/2,
%% fix_description/2) is READY for when we later subsume a real CFD code
%% (OpenFOAM, etc.) with known defects, but starts empty here.
%%
%% Verification convention (per mavchin):
%%   - Roe approximate Riemann solver in CUDA emit (algebraic, no iteration)
%%   - Reference solver in Python uses exact Riemann (Newton iteration)
%%   - Bit-identical comparison is Roe-vs-Roe across dispatch paths
%%     (CUDA-oxide, NVRTC, nvcc — same 6-column matrix as ML kernels)
%%   - Both converge to the same analytical solution as dx→0
%% =============================================================================

:- module(kernel_templates_cfd, [
    %% Facts (declarative kernel definitions)
    stencil_expr/4,            % +KName, +LeftVar, +RightVar, -BodyStmts
    cfd_kernel_signature/2,    % +KName, -ParamList for kernel signature

    %% Consumer predicates (emit c_ast trees)
    cfd_flux_kernel/2,         % +KName, -Kernel  (stencil-over-interfaces)
    cfd_flux_wrapper/2,        % +KName, -Wrapper
    cfd_update_conservative_kernel/2,    % C3: U -= dt_dx * (F[i+1] - F[i])
    cfd_update_conservative_wrapper/2,
    cfd_compute_primitives_kernel/2,     % C4: cons (rho, rho*u, E) -> prim (rho, u, p)
    cfd_compute_primitives_wrapper/2,
    cfd_cfl_condition_kernel/2,          % C5: max wavespeed reduction via block_reduce_max
    cfd_cfl_condition_wrapper/2,

    %% Fix-flag metadata (per substrate-honesty convention)
    kernel_available_fixes/2,  % +KernelPred, -FixList (always [] for CFD-against-physics)
    fix_description/2          % +FixAtom, -Description (none registered for CFD yet)
]).

:- use_module(c_ast).


%% =============================================================================
%% PREDICATE DECLARATIONS
%% =============================================================================
%%
%% Some exported predicates have no clauses yet — they're placeholders for
%% the next subtask. Declaring them dynamic + discontiguous keeps the module
%% loadable (no "undefined exported procedure" warnings) and signals their
%% future-population intent.

:- dynamic stencil_expr/4.
:- dynamic cfd_kernel_signature/2.
:- dynamic fix_description/2.
:- discontiguous stencil_expr/4.
:- discontiguous kernel_available_fixes/2.
:- discontiguous fix_description/2.


%% =============================================================================
%% PHYSICS CONSTANTS
%% =============================================================================
%%
%% These are bound to literal values in the emitted CUDA (not runtime params).
%% Per Sod's shock tube convention (medayek's sod_shock_tube.py:37):
%%   GAMMA = 1.4 (ratio of specific heats for ideal diatomic gas)
%%
%% Substrate convention: constants appear as c_float_f(value) in emit so
%% they pick up the f-suffix (matching the silu 1.0f literal lesson from
%% commit 6627e1b1e).

cfd_constant(gamma, c_float_f(1.4)).


%% =============================================================================
%% STENCIL-OVER-INTERFACES KERNELS
%% =============================================================================
%%
%% Convention (per medayek's sod_shock_tube.py:240-285):
%%
%%   - Grid has N cells: U[3 * N] in memory (component-major: rho, rho_u, E)
%%   - There are N+1 interfaces between cells
%%   - Flux F[3 * (N+1)] is computed at each interface
%%   - Boundary mode: TRANSMISSIVE
%%       i == 0   → left state from cell 0, right state from cell 0
%%       i == N   → left state from cell N-1, right state from cell N-1
%%       i ∈ 1..N-1 → left state from cell i-1, right state from cell i
%%
%% Index clamping idiom (avoids branchy code, deterministic across warps):
%%   left_idx  = max(0, i - 1)
%%   right_idx = min(N - 1, i)
%%
%% This produces the correct transmissive behavior:
%%   i == 0:   left_idx = max(0, -1) = 0,    right_idx = min(N-1, 0) = 0      ✓
%%   i == N:   left_idx = max(0, N-1) = N-1, right_idx = min(N-1, N) = N-1    ✓
%%   1 ≤ i ≤ N-1: left_idx = i-1, right_idx = i                                ✓


%% cfd_flux_kernel(+KName, -Kernel)
%%
%% Emits a __global__ kernel that loops over the N+1 interfaces of a 1D
%% grid, performs transmissive-BC index clamping to determine the left
%% and right neighbor cells, then splices in the kernel-specific flux
%% body from the stencil_expr/4 fact.
%%
%% Kernel signature:
%%   __global__ void k_KName(const float * __restrict__ U,
%%                           float * __restrict__ F,
%%                           int N);
%%
%% Body shape:
%%   int i = blockIdx.x * blockDim.x + threadIdx.x;
%%   if (i > N) return;              // bound: N+1 interfaces (0..N inclusive)
%%   int left_idx  = max(0, i - 1);
%%   int right_idx = min(N - 1, i);
%%   <body from stencil_expr — uses left_idx, right_idx to access U;
%%    writes 3 components of flux to F[0*Np1 + i], F[1*Np1 + i], F[2*Np1 + i]>
%%
%% Per substrate-honesty: the body stmts in stencil_expr/4 SHOULD reference
%% c_var(left_idx) and c_var(right_idx) for neighbor cell indexing, and
%% c_var(i) for the interface index. They have access to c_var(U), c_var(F),
%% c_var(N) from the kernel signature.

cfd_flux_kernel(KName, Kernel) :-
    stencil_expr(KName, _LeftVar, _RightVar, BodyStmts),
    Kernel = c_func(['__global__'], c_type(void), KName,
        [param(c_type(const_restrict_ptr(c_type(float))), 'U'),
         param(c_type(restrict_ptr(c_type(float))), 'F'),
         param(c_type(int), 'N')],
        [%% int i = blockIdx.x * blockDim.x + threadIdx.x;
         c_decl_init(c_type(int), i,
             c_binop('+',
                 c_binop('*',
                     c_member(c_var(blockIdx), x),
                     c_member(c_var(blockDim), x)),
                 c_member(c_var(threadIdx), x))),
         %% N+1 interfaces — bound is i > N (allow i == N for the rightmost)
         c_if(c_binop('>', c_var(i), c_var('N')), [c_return_void]),
         %% Transmissive BC via index clamping:
         %%   left_idx  = max(0, i - 1)
         %%   right_idx = min(N - 1, i)
         c_decl_init(c_type(int), left_idx,
             c_call(max, [c_int(0), c_binop('-', c_var(i), c_int(1))])),
         c_decl_init(c_type(int), right_idx,
             c_call(min, [c_binop('-', c_var('N'), c_int(1)), c_var(i)]))
         | BodyStmts]).


%% cfd_flux_wrapper(+KName, -Wrapper)
%%
%% C API wrapper for a flux kernel. Launches enough blocks to cover N+1
%% interfaces with 256 threads per block. Match the ML wrapper convention.
%%
%% Launch geometry: ceil_div(N+1, 256) blocks × 256 threads per block.
%%
%% Wrapper name: gpu_<KName_without_k_prefix>. For k_compute_flux:
%%   gpu_compute_flux(const float *U, float *F, int N).

cfd_flux_wrapper(KName, Wrapper) :-
    %% Derive wrapper name: strip "k_" prefix and prepend "gpu_"
    atom_concat('k_', Suffix, KName),
    atom_concat('gpu_', Suffix, WName),
    Wrapper = c_func(c_type(void), WName,
        [param(c_type(const_restrict_ptr(c_type(float))), 'U'),
         param(c_type(restrict_ptr(c_type(float))), 'F'),
         param(c_type(int), 'N')],
        [c_cuda_launch(KName,
            c_binop('/',
                c_paren(c_binop('+',
                    c_binop('+', c_var('N'), c_int(1)),
                    c_int(255))),
                c_int(256)),
            c_int(256),
            [c_var('U'), c_var('F'), c_var('N')])]).


%% =============================================================================
%% STENCIL_EXPR FACTS (per-kernel body definitions)
%% =============================================================================
%%
%% Each fact declares the WHAT of a flux kernel: the body c_ast statements
%% that consume left_idx/right_idx and produce the flux output.
%%
%% Stub for now — k_compute_flux's Roe-flux body is C2 (next subtask).
%% This fact will be expanded to encode the full Roe approximate Riemann
%% solver as a tree of c_ast statements:
%%
%%   1. Load 3-component conservative states from cells [left_idx], [right_idx]
%%   2. Convert to primitive variables (rho, u, p) via the
%%      primitive_from_conservative algorithm
%%   3. Compute Roe-averaged states (density-weighted means)
%%   4. Compute Roe eigenvalues (wave speeds)
%%   5. Compute the flux at the interface
%%   6. Store 3 components of flux to F[component*(N+1) + i]
%%
%% Approximately 30 c_ast statements per medayek's reference scale.

%% stencil_expr(k_compute_flux, left_idx, right_idx, Stmts)
%%
%% Per 2026-05-18 ~19:00 UTC substrate-honesty repair (C2.1):
%% Rewritten to match mavchin's hand-written k_roe_flux at
%% /tmp/sod_gpu_kernels.cu:82-153, which is the actual subsumption target.
%%
%% Substantive corrections from the prior C2 commit (b3ff1fefc):
%%   - Added Harten entropy fix on the eigenvalues:
%%       eps = 0.1f * a_roe
%%       if (|lam_k| < eps) lam_k = (lam_k^2 + eps^2) / (2*eps)
%%       else lam_k = |lam_k|
%%     Matters at sonic points (e.g., the rarefaction wave in Sod's tube
%%     where lambda_1 = u - a passes through zero).
%%
%%   - Switched wave-strength formula from Toro's eq. 11.69-11.71 to
%%     mavchin's rho_roe formulation:
%%       rho_roe = sqrt_rl * sqrt_rr (geometric mean of densities)
%%       alpha1 = (dp - rho_roe*a_roe*du) / (2*a_roe^2)
%%       alpha2 = drho - dp / a_roe^2
%%       alpha3 = (dp + rho_roe*a_roe*du) / (2*a_roe^2)
%%     Algebraically equivalent to Toro's form in exact arithmetic but
%%     DIFFERENT float32 rounding. Roe-vs-Roe bit-identical requires
%%     matching the formulation exactly.
%%
%%   - State diffs are PRIMITIVE (drho, du, dp), not CONSERVATIVE (dU_0/1/2).
%%
%%   - Dissipation uses post-Harten lams (already abs-valued by the fix).
%%
%% Per medayek's methodology principle established this session:
%% "Inspection-based hypothesis (my Toro formulation is correct) vs
%%  empirical truth (mavchin's kernel has Harten fix + different wave
%%  strength). Measure, don't assume. Align with the subsumption target."
%%
%% Per Heath's substrate-honesty framing question:
%%   "What was the originally intended purpose of this code,
%%    and what is the correct form of that code we should achieve today?"
%%   Answer: subsume mavchin's working k_roe_flux. The substrate's c_ast
%%   must match its algorithmic structure exactly for bit-identical Level A
%%   verification to be meaningful.
%%
%% Verified: /tmp/roe_reference.py (Python implementation of this same
%% algorithm, float32 throughout) produces F_rho = 0.396 at the Sod
%% diaphragm interface, matching mavchin's reported value.
%%
%% Computation in 12 sections (~67 c_ast statements total):
%%    1. Load conservative state from cells [left_idx], [right_idx]
%%    2. Primitive variables (rho_l, u_l, E_l, p_l, rho_r, u_r, E_r, p_r)
%%    3. Total enthalpy H = (E + p) / rho
%%    4. Roe averages (u_roe, H_roe, rho_roe = sqrt_rl * sqrt_rr)
%%    5. Roe sound speed a_roe
%%    6. Eigenvalues lambda_{1,2,3}
%%    7. Harten entropy fix on each lambda (smooth |lam| < eps)
%%    8. Primitive state differences (drho, du, dp)
%%    9. Wave strengths alpha_{1,2,3} via rho_roe formulation
%%   10. Physical fluxes F_L, F_R (both sides)
%%   11. Dissipation D = sum_k lam_k * alpha_k * r_k
%%       (right eigenvectors inlined to avoid 9 temp vars)
%%   12. Roe flux F = 0.5*(F_L + F_R) - 0.5*D, written to F[comp*(N+1) + i]
stencil_expr(k_compute_flux, left_idx, right_idx, [
        c_comment('Roe approximate Riemann solver — matches mavchin /tmp/sod_gpu_kernels.cu:82-153'),
        c_comment('--- 1. Load conservative state from left and right cells ---'),
        c_decl_init(c_type(float), 'U_L_0',
            c_index(c_var('U'),
                c_binop('+',
                    c_binop('*', c_int(0), c_var('N')),
                    c_var(left_idx)))),
        c_decl_init(c_type(float), 'U_L_1',
            c_index(c_var('U'),
                c_binop('+',
                    c_binop('*', c_int(1), c_var('N')),
                    c_var(left_idx)))),
        c_decl_init(c_type(float), 'U_L_2',
            c_index(c_var('U'),
                c_binop('+',
                    c_binop('*', c_int(2), c_var('N')),
                    c_var(left_idx)))),
        c_decl_init(c_type(float), 'U_R_0',
            c_index(c_var('U'),
                c_binop('+',
                    c_binop('*', c_int(0), c_var('N')),
                    c_var(right_idx)))),
        c_decl_init(c_type(float), 'U_R_1',
            c_index(c_var('U'),
                c_binop('+',
                    c_binop('*', c_int(1), c_var('N')),
                    c_var(right_idx)))),
        c_decl_init(c_type(float), 'U_R_2',
            c_index(c_var('U'),
                c_binop('+',
                    c_binop('*', c_int(2), c_var('N')),
                    c_var(right_idx)))),

        c_comment('--- 2. Primitive variables (rho, u, p) ---'),
        c_decl_init(c_type(float), rho_l, c_var('U_L_0')),
        c_decl_init(c_type(float), u_l,
            c_binop('/', c_var('U_L_1'), c_var(rho_l))),
        c_decl_init(c_type(float), 'E_l', c_var('U_L_2')),
        c_decl_init(c_type(float), p_l,
            c_binop('*', c_float_f(0.4),
                c_paren(c_binop('-',
                    c_var('E_l'),
                    c_binop('*',
                        c_binop('*', c_float_f(0.5), c_var(rho_l)),
                        c_binop('*', c_var(u_l), c_var(u_l))))))),
        c_decl_init(c_type(float), rho_r, c_var('U_R_0')),
        c_decl_init(c_type(float), u_r,
            c_binop('/', c_var('U_R_1'), c_var(rho_r))),
        c_decl_init(c_type(float), 'E_r', c_var('U_R_2')),
        c_decl_init(c_type(float), p_r,
            c_binop('*', c_float_f(0.4),
                c_paren(c_binop('-',
                    c_var('E_r'),
                    c_binop('*',
                        c_binop('*', c_float_f(0.5), c_var(rho_r)),
                        c_binop('*', c_var(u_r), c_var(u_r))))))),

        c_comment('--- 3. Total enthalpy: H = (E + p) / rho ---'),
        c_decl_init(c_type(float), 'H_l',
            c_binop('/',
                c_paren(c_binop('+', c_var('E_l'), c_var(p_l))),
                c_var(rho_l))),
        c_decl_init(c_type(float), 'H_r',
            c_binop('/',
                c_paren(c_binop('+', c_var('E_r'), c_var(p_r))),
                c_var(rho_r))),

        c_comment('--- 4. Roe averages (density-weighted) ---'),
        c_decl_init(c_type(float), sqrt_rl, c_call(sqrtf, [c_var(rho_l)])),
        c_decl_init(c_type(float), sqrt_rr, c_call(sqrtf, [c_var(rho_r)])),
        c_decl_init(c_type(float), denom,
            c_binop('+', c_var(sqrt_rl), c_var(sqrt_rr))),
        c_decl_init(c_type(float), u_roe,
            c_binop('/',
                c_paren(c_binop('+',
                    c_binop('*', c_var(sqrt_rl), c_var(u_l)),
                    c_binop('*', c_var(sqrt_rr), c_var(u_r)))),
                c_var(denom))),
        c_decl_init(c_type(float), 'H_roe',
            c_binop('/',
                c_paren(c_binop('+',
                    c_binop('*', c_var(sqrt_rl), c_var('H_l')),
                    c_binop('*', c_var(sqrt_rr), c_var('H_r')))),
                c_var(denom))),
        %% Roe-averaged density used in wave strengths (mavchin's choice):
        c_decl_init(c_type(float), rho_roe,
            c_binop('*', c_var(sqrt_rl), c_var(sqrt_rr))),

        c_comment('--- 5. Roe sound speed: a_roe = sqrt(GM1 * (H_roe - 0.5*u_roe^2)) ---'),
        c_decl_init(c_type(float), a_roe,
            c_call(sqrtf,
                [c_binop('*', c_float_f(0.4),
                    c_paren(c_binop('-',
                        c_var('H_roe'),
                        c_binop('*', c_float_f(0.5),
                            c_binop('*', c_var(u_roe), c_var(u_roe))))))])),

        c_comment('--- 6. Eigenvalues ---'),
        c_decl_init(c_type(float), lam1, c_binop('-', c_var(u_roe), c_var(a_roe))),
        c_decl_init(c_type(float), lam2, c_var(u_roe)),
        c_decl_init(c_type(float), lam3, c_binop('+', c_var(u_roe), c_var(a_roe))),

        c_comment('--- 7. Harten entropy fix (eps = 0.1f * a_roe) ---'),
        c_decl_init(c_type(float), eps,
            c_binop('*', c_float_f(0.1), c_var(a_roe))),
        %% For each lambda: if |lam| < eps, smooth via (lam^2 + eps^2)/(2*eps); else abs(lam)
        c_if(c_binop('<', c_call(fabsf, [c_var(lam1)]), c_var(eps)),
            [c_assign(c_var(lam1),
                c_binop('/',
                    c_paren(c_binop('+',
                        c_binop('*', c_var(lam1), c_var(lam1)),
                        c_binop('*', c_var(eps), c_var(eps)))),
                    c_paren(c_binop('*', c_float_f(2.0), c_var(eps)))))],
            [c_assign(c_var(lam1), c_call(fabsf, [c_var(lam1)]))]),
        c_if(c_binop('<', c_call(fabsf, [c_var(lam2)]), c_var(eps)),
            [c_assign(c_var(lam2),
                c_binop('/',
                    c_paren(c_binop('+',
                        c_binop('*', c_var(lam2), c_var(lam2)),
                        c_binop('*', c_var(eps), c_var(eps)))),
                    c_paren(c_binop('*', c_float_f(2.0), c_var(eps)))))],
            [c_assign(c_var(lam2), c_call(fabsf, [c_var(lam2)]))]),
        c_if(c_binop('<', c_call(fabsf, [c_var(lam3)]), c_var(eps)),
            [c_assign(c_var(lam3),
                c_binop('/',
                    c_paren(c_binop('+',
                        c_binop('*', c_var(lam3), c_var(lam3)),
                        c_binop('*', c_var(eps), c_var(eps)))),
                    c_paren(c_binop('*', c_float_f(2.0), c_var(eps)))))],
            [c_assign(c_var(lam3), c_call(fabsf, [c_var(lam3)]))]),

        c_comment('--- 8. Primitive state differences ---'),
        c_decl_init(c_type(float), drho, c_binop('-', c_var(rho_r), c_var(rho_l))),
        c_decl_init(c_type(float), du, c_binop('-', c_var(u_r), c_var(u_l))),
        c_decl_init(c_type(float), dp, c_binop('-', c_var(p_r), c_var(p_l))),

        c_comment('--- 9. Wave strengths (rho_roe formulation) ---'),
        c_decl_init(c_type(float), a_roe_sq,
            c_binop('*', c_var(a_roe), c_var(a_roe))),
        c_decl_init(c_type(float), inv_two_a_sq,
            c_binop('/',
                c_float_f(1.0),
                c_paren(c_binop('*', c_float_f(2.0), c_var(a_roe_sq))))),
        %% alpha1 = (dp - rho_roe*a_roe*du) * inv_two_a_sq
        c_decl_init(c_type(float), alpha1,
            c_binop('*',
                c_paren(c_binop('-',
                    c_var(dp),
                    c_binop('*',
                        c_binop('*', c_var(rho_roe), c_var(a_roe)),
                        c_var(du)))),
                c_var(inv_two_a_sq))),
        %% alpha2 = drho - dp / a_roe_sq
        c_decl_init(c_type(float), alpha2,
            c_binop('-',
                c_var(drho),
                c_binop('/', c_var(dp), c_var(a_roe_sq)))),
        %% alpha3 = (dp + rho_roe*a_roe*du) * inv_two_a_sq
        c_decl_init(c_type(float), alpha3,
            c_binop('*',
                c_paren(c_binop('+',
                    c_var(dp),
                    c_binop('*',
                        c_binop('*', c_var(rho_roe), c_var(a_roe)),
                        c_var(du)))),
                c_var(inv_two_a_sq))),

        c_comment('--- 10. Physical fluxes F(U) on left and right ---'),
        c_decl_init(c_type(float), 'FL_rho',
            c_binop('*', c_var(rho_l), c_var(u_l))),
        c_decl_init(c_type(float), 'FL_rhou',
            c_binop('+',
                c_binop('*',
                    c_binop('*', c_var(rho_l), c_var(u_l)),
                    c_var(u_l)),
                c_var(p_l))),
        c_decl_init(c_type(float), 'FL_E',
            c_binop('*',
                c_var(u_l),
                c_paren(c_binop('+', c_var('E_l'), c_var(p_l))))),
        c_decl_init(c_type(float), 'FR_rho',
            c_binop('*', c_var(rho_r), c_var(u_r))),
        c_decl_init(c_type(float), 'FR_rhou',
            c_binop('+',
                c_binop('*',
                    c_binop('*', c_var(rho_r), c_var(u_r)),
                    c_var(u_r)),
                c_var(p_r))),
        c_decl_init(c_type(float), 'FR_E',
            c_binop('*',
                c_var(u_r),
                c_paren(c_binop('+', c_var('E_r'), c_var(p_r))))),

        c_comment('--- 11. Dissipation D = sum_k lam_k * alpha_k * r_k (lams already abs-valued by Harten fix) ---'),
        %% Right eigenvectors r_k for 1D Euler (inlined into D sum):
        %%   r1 = (1, u-a, H-u*a)
        %%   r2 = (1, u,   0.5*u*u)
        %%   r3 = (1, u+a, H+u*a)
        c_decl_init(c_type(float), 'D_rho',
            c_binop('+',
                c_binop('+',
                    c_binop('*', c_var(lam1), c_var(alpha1)),
                    c_binop('*', c_var(lam2), c_var(alpha2))),
                c_binop('*', c_var(lam3), c_var(alpha3)))),
        c_decl_init(c_type(float), 'D_rhou',
            c_binop('+',
                c_binop('+',
                    c_binop('*',
                        c_binop('*', c_var(lam1), c_var(alpha1)),
                        c_paren(c_binop('-', c_var(u_roe), c_var(a_roe)))),
                    c_binop('*',
                        c_binop('*', c_var(lam2), c_var(alpha2)),
                        c_var(u_roe))),
                c_binop('*',
                    c_binop('*', c_var(lam3), c_var(alpha3)),
                    c_paren(c_binop('+', c_var(u_roe), c_var(a_roe)))))),
        c_decl_init(c_type(float), 'D_E',
            c_binop('+',
                c_binop('+',
                    c_binop('*',
                        c_binop('*', c_var(lam1), c_var(alpha1)),
                        c_paren(c_binop('-',
                            c_var('H_roe'),
                            c_binop('*', c_var(u_roe), c_var(a_roe))))),
                    c_binop('*',
                        c_binop('*', c_var(lam2), c_var(alpha2)),
                        c_binop('*', c_float_f(0.5),
                            c_binop('*', c_var(u_roe), c_var(u_roe))))),
                c_binop('*',
                    c_binop('*', c_var(lam3), c_var(alpha3)),
                    c_paren(c_binop('+',
                        c_var('H_roe'),
                        c_binop('*', c_var(u_roe), c_var(a_roe))))))),

        c_comment('--- 12. Roe flux: F = 0.5*(FL+FR) - 0.5*D ---'),
        c_assign(
            c_index(c_var('F'),
                c_binop('+',
                    c_binop('*', c_int(0),
                        c_paren(c_binop('+', c_var('N'), c_int(1)))),
                    c_var(i))),
            c_binop('-',
                c_binop('*', c_float_f(0.5),
                    c_paren(c_binop('+', c_var('FL_rho'), c_var('FR_rho')))),
                c_binop('*', c_float_f(0.5), c_var('D_rho')))),
        c_assign(
            c_index(c_var('F'),
                c_binop('+',
                    c_binop('*', c_int(1),
                        c_paren(c_binop('+', c_var('N'), c_int(1)))),
                    c_var(i))),
            c_binop('-',
                c_binop('*', c_float_f(0.5),
                    c_paren(c_binop('+', c_var('FL_rhou'), c_var('FR_rhou')))),
                c_binop('*', c_float_f(0.5), c_var('D_rhou')))),
        c_assign(
            c_index(c_var('F'),
                c_binop('+',
                    c_binop('*', c_int(2),
                        c_paren(c_binop('+', c_var('N'), c_int(1)))),
                    c_var(i))),
            c_binop('-',
                c_binop('*', c_float_f(0.5),
                    c_paren(c_binop('+', c_var('FL_E'), c_var('FR_E')))),
                c_binop('*', c_float_f(0.5), c_var('D_E'))))
    ]).



%% =============================================================================
%% KERNEL SIGNATURE METADATA
%% =============================================================================
%%
%% Currently unused — placeholder for future kernel-class generalization.
%% When CFD adds kernels beyond stencil-over-interfaces (e.g., elementwise
%% conservative update, primitive recovery), each kernel-class will have
%% its own consumer predicate (paralleling cfd_flux_kernel) AND signature
%% metadata.

%% Empty for now.


%% =============================================================================
%% C3 — k_update_conservative (Euler conservative update)
%% =============================================================================
%%
%% Per 2026-05-18 ~19:40 UTC subtask C3:
%% Emits the elementwise conservative-variable update at the heart of the
%% Godunov-type finite-volume timestep:
%%
%%   U_new = U - dt/dx * (F[i+1] - F[i])    componentwise
%%
%% Matches mavchin's k_euler_update at /tmp/sod_gpu_kernels.cu:178-187
%% algorithmically. Per medayek's "substrate keeps flat-array convention,
%% harness reshapes": mavchin uses 3 separate (rho, rhou, E) arrays; the
%% substrate uses one flat U[3 * N] with component-major SoA layout.
%% Same arithmetic, same output values, different calling convention.
%%
%% Signature:
%%   __global__ void k_update_conservative(float * __restrict__ U,
%%                                          const float * __restrict__ F,
%%                                          float dt_dx, int N);
%%   - U: in/out conservative state [3 * N], component-major SoA
%%   - F: flux array [3 * (N+1)], component-major SoA
%%   - dt_dx: dt/dx scalar (precomputed CFL timestep / cell spacing)
%%   - N: number of cells

cfd_update_conservative_kernel(KName, Kernel) :-
    Kernel = c_func(['__global__'], c_type(void), KName,
        [param(c_type(restrict_ptr(c_type(float))), 'U'),
         param(c_type(const_restrict_ptr(c_type(float))), 'F'),
         param(c_type(float), dt_dx),
         param(c_type(int), 'N')],
        [c_decl_init(c_type(int), i,
             c_binop('+',
                 c_binop('*',
                     c_member(c_var(blockIdx), x),
                     c_member(c_var(blockDim), x)),
                 c_member(c_var(threadIdx), x))),
         c_if(c_binop('<', c_var(i), c_var('N')),
              [%% For each component c=0,1,2:
               %%   U[c*N + i] -= dt_dx * (F[c*(N+1) + (i+1)] - F[c*(N+1) + i])
               c_compound_assign('-=',
                   c_index(c_var('U'),
                       c_binop('+',
                           c_binop('*', c_int(0), c_var('N')),
                           c_var(i))),
                   c_binop('*',
                       c_var(dt_dx),
                       c_paren(c_binop('-',
                           c_index(c_var('F'),
                               c_binop('+',
                                   c_binop('*', c_int(0),
                                       c_paren(c_binop('+', c_var('N'), c_int(1)))),
                                   c_paren(c_binop('+', c_var(i), c_int(1))))),
                           c_index(c_var('F'),
                               c_binop('+',
                                   c_binop('*', c_int(0),
                                       c_paren(c_binop('+', c_var('N'), c_int(1)))),
                                   c_var(i))))))),
               c_compound_assign('-=',
                   c_index(c_var('U'),
                       c_binop('+',
                           c_binop('*', c_int(1), c_var('N')),
                           c_var(i))),
                   c_binop('*',
                       c_var(dt_dx),
                       c_paren(c_binop('-',
                           c_index(c_var('F'),
                               c_binop('+',
                                   c_binop('*', c_int(1),
                                       c_paren(c_binop('+', c_var('N'), c_int(1)))),
                                   c_paren(c_binop('+', c_var(i), c_int(1))))),
                           c_index(c_var('F'),
                               c_binop('+',
                                   c_binop('*', c_int(1),
                                       c_paren(c_binop('+', c_var('N'), c_int(1)))),
                                   c_var(i))))))),
               c_compound_assign('-=',
                   c_index(c_var('U'),
                       c_binop('+',
                           c_binop('*', c_int(2), c_var('N')),
                           c_var(i))),
                   c_binop('*',
                       c_var(dt_dx),
                       c_paren(c_binop('-',
                           c_index(c_var('F'),
                               c_binop('+',
                                   c_binop('*', c_int(2),
                                       c_paren(c_binop('+', c_var('N'), c_int(1)))),
                                   c_paren(c_binop('+', c_var(i), c_int(1))))),
                           c_index(c_var('F'),
                               c_binop('+',
                                   c_binop('*', c_int(2),
                                       c_paren(c_binop('+', c_var('N'), c_int(1)))),
                                   c_var(i)))))))])]).


%% C3 wrapper: gpu_update_conservative
cfd_update_conservative_wrapper(KName, Wrapper) :-
    atom_concat('k_', Suffix, KName),
    atom_concat('gpu_', Suffix, WName),
    Wrapper = c_func(c_type(void), WName,
        [param(c_type(restrict_ptr(c_type(float))), 'U'),
         param(c_type(const_restrict_ptr(c_type(float))), 'F'),
         param(c_type(float), dt_dx),
         param(c_type(int), 'N')],
        [c_cuda_launch(KName,
            c_binop('/',
                c_paren(c_binop('+', c_var('N'), c_int(255))),
                c_int(256)),
            c_int(256),
            [c_var('U'), c_var('F'), c_var(dt_dx), c_var('N')])]).


%% =============================================================================
%% C4 — k_compute_primitives (conservative → primitive variable conversion)
%% =============================================================================
%%
%% Per 2026-05-18 ~19:40 UTC subtask C4:
%% Elementwise conservative-to-primitive conversion:
%%
%%   rho   = U_0
%%   u     = U_1 / rho
%%   p     = (gamma-1) * (U_2 - 0.5 * rho * u^2)
%%
%% Matches mavchin's k_cons_to_prim at /tmp/sod_gpu_kernels.cu:22-33
%% algorithmically. Same flat-array convention as C3 (medayek's choice).
%%
%% Signature:
%%   __global__ void k_compute_primitives(const float * __restrict__ U,
%%                                         float * __restrict__ prim,
%%                                         int N);
%%   - U: conservative state [3 * N], component-major SoA
%%   - prim: primitive state [3 * N] = (rho, u, p), component-major SoA

cfd_compute_primitives_kernel(KName, Kernel) :-
    Kernel = c_func(['__global__'], c_type(void), KName,
        [param(c_type(const_restrict_ptr(c_type(float))), 'U'),
         param(c_type(restrict_ptr(c_type(float))), prim),
         param(c_type(int), 'N')],
        [c_decl_init(c_type(int), i,
             c_binop('+',
                 c_binop('*',
                     c_member(c_var(blockIdx), x),
                     c_member(c_var(blockDim), x)),
                 c_member(c_var(threadIdx), x))),
         c_if(c_binop('<', c_var(i), c_var('N')),
              [%% float r = U[0*N + i];
               c_decl_init(c_type(float), r,
                   c_index(c_var('U'),
                       c_binop('+',
                           c_binop('*', c_int(0), c_var('N')),
                           c_var(i)))),
               %% float u = U[1*N + i] / r;
               c_decl_init(c_type(float), u,
                   c_binop('/',
                       c_index(c_var('U'),
                           c_binop('+',
                               c_binop('*', c_int(1), c_var('N')),
                               c_var(i))),
                       c_var(r))),
               %% float p = 0.4f * (U[2*N + i] - 0.5f * r * u * u);
               c_decl_init(c_type(float), p,
                   c_binop('*', c_float_f(0.4),
                       c_paren(c_binop('-',
                           c_index(c_var('U'),
                               c_binop('+',
                                   c_binop('*', c_int(2), c_var('N')),
                                   c_var(i))),
                           c_binop('*',
                               c_binop('*', c_float_f(0.5), c_var(r)),
                               c_binop('*', c_var(u), c_var(u))))))),
               %% Store primitives
               c_assign(
                   c_index(c_var(prim),
                       c_binop('+',
                           c_binop('*', c_int(0), c_var('N')),
                           c_var(i))),
                   c_var(r)),
               c_assign(
                   c_index(c_var(prim),
                       c_binop('+',
                           c_binop('*', c_int(1), c_var('N')),
                           c_var(i))),
                   c_var(u)),
               c_assign(
                   c_index(c_var(prim),
                       c_binop('+',
                           c_binop('*', c_int(2), c_var('N')),
                           c_var(i))),
                   c_var(p))])]).


cfd_compute_primitives_wrapper(KName, Wrapper) :-
    atom_concat('k_', Suffix, KName),
    atom_concat('gpu_', Suffix, WName),
    Wrapper = c_func(c_type(void), WName,
        [param(c_type(const_restrict_ptr(c_type(float))), 'U'),
         param(c_type(restrict_ptr(c_type(float))), prim),
         param(c_type(int), 'N')],
        [c_cuda_launch(KName,
            c_binop('/',
                c_paren(c_binop('+', c_var('N'), c_int(255))),
                c_int(256)),
            c_int(256),
            [c_var('U'), c_var(prim), c_var('N')])]).


%% =============================================================================
%% C5 — k_cfl_condition (CFL timestep reduction)
%% =============================================================================
%%
%% Per 2026-05-18 ~19:40 UTC subtask C5:
%% Computes the maximum wavespeed across all cells for the CFL condition:
%%
%%   max_wavespeed = max_i (|u_i| + a_i)    where a_i = sqrt(gamma * p_i / rho_i)
%%
%% The CFL timestep is then dt = CFL * dx / max_wavespeed (computed on host).
%%
%% Algorithm: per-thread strided max-accumulate over cells, then block_reduce_max
%% to get the block-level maximum, written to result[0].
%%
%% Uses block_reduce_max from kernel_templates_llama.pl (commit dc0b8be32) —
%% the same __device__ helper function for warp-shuffle + cross-warp shared mem
%% reduction. Cross-domain helper reuse at the program-assembly level.
%%
%% Matches mavchin's k_max_wavespeed at /tmp/sod_gpu_kernels.cu:62-...
%%
%% Signature:
%%   __global__ void k_cfl_condition(const float * __restrict__ prim,
%%                                    float * __restrict__ result,
%%                                    int N);
%%   - prim: primitive state [3 * N] = (rho, u, p), component-major SoA
%%   - result: single-element output array; result[0] = max_wavespeed
%%   - N: number of cells
%%
%% Takes PRIMITIVE state (not conservative). Matches mavchin's k_max_wavespeed
%% signature. Pipeline: k_compute_primitives runs first to produce prim from U;
%% then k_cfl_condition reads prim. Decoupling per medayek's "align with the
%% subsumption target" principle.

%% Per the medayek principle "align with the subsumption target":
%% takes PRIMITIVE state as input (matches mavchin's k_max_wavespeed),
%% not conservative. The harness runs k_compute_primitives first.
%% This decouples CFL from the conservative-to-primitive conversion,
%% matching mavchin's pipeline structure exactly.
cfd_cfl_condition_kernel(KName, Kernel) :-
    Kernel = c_func(['__global__'], c_type(void), KName,
        [param(c_type(const_restrict_ptr(c_type(float))), prim),
         param(c_type(restrict_ptr(c_type(float))), result),
         param(c_type(int), 'N')],
        [c_decl_init(c_type(int), tid, c_member(c_var(threadIdx), x)),
         %% Per-thread strided max-accumulate
         c_decl_init(c_type(float), local_max, c_float_f(0.0)),
         c_for(c_decl_init(c_type(int), i, c_var(tid)),
               c_binop('<', c_var(i), c_var('N')),
               c_compound_step(c_var(i), '+=', c_int(256)),
               [%% Read primitive variables (rho, u, p) for cell i
                c_decl_init(c_type(float), r,
                    c_index(c_var(prim),
                        c_binop('+',
                            c_binop('*', c_int(0), c_var('N')),
                            c_var(i)))),
                c_decl_init(c_type(float), u,
                    c_index(c_var(prim),
                        c_binop('+',
                            c_binop('*', c_int(1), c_var('N')),
                            c_var(i)))),
                c_decl_init(c_type(float), p,
                    c_index(c_var(prim),
                        c_binop('+',
                            c_binop('*', c_int(2), c_var('N')),
                            c_var(i)))),
                %% Sound speed: a = sqrt(gamma * p / r) = sqrt(1.4f * p / r)
                c_decl_init(c_type(float), a,
                    c_call(sqrtf,
                        [c_binop('/',
                            c_binop('*', c_float_f(1.4), c_var(p)),
                            c_var(r))])),
                %% Wavespeed: |u| + a
                c_decl_init(c_type(float), ws,
                    c_binop('+', c_call(fabsf, [c_var(u)]), c_var(a))),
                %% Accumulate max
                c_assign(c_var(local_max),
                    c_call(fmaxf, [c_var(local_max), c_var(ws)]))]),
         %% Block-level reduction via the substrate's block_reduce_max helper.
         %% The helper expects a shared buffer of size 8 (for 8 warps per 256-
         %% thread block). Declared inline at first use.
         c_shared_decl(c_type(float), buf_iw, c_int(8)),
         c_assign(c_var(local_max),
             c_call(block_reduce_max, [c_var(local_max), c_var(buf_iw)])),
         %% Thread 0 of block 0 writes the final result
         c_if(c_binop('==', c_var(tid), c_int(0)),
              [c_assign(c_index(c_var(result), c_int(0)), c_var(local_max))])]).


cfd_cfl_condition_wrapper(KName, Wrapper) :-
    atom_concat('k_', Suffix, KName),
    atom_concat('gpu_', Suffix, WName),
    Wrapper = c_func(c_type(void), WName,
        [param(c_type(const_restrict_ptr(c_type(float))), prim),
         param(c_type(restrict_ptr(c_type(float))), result),
         param(c_type(int), 'N')],
        [%% Single-block launch — reduces entire array in one block via
         %% strided per-thread accumulation + block reduce. Block size 256.
         c_cuda_launch(KName,
            c_int(1),
            c_int(256),
            [c_var(prim), c_var(result), c_var('N')])]).


%% =============================================================================
%% FIX-FLAG METADATA
%% =============================================================================
%%
%% Per the substrate-honesty principle established 2026-05-18 ~08:00 UTC:
%% the substrate exposes named, individually-toggleable defect-repairs as
%% first-class metadata for the bit-identical test harness to discover.
%%
%% For CFD-against-physics there are no inherited defects to fix (the
%% reference is mathematical truth, not another implementation). The
%% fix lists start empty. The pattern is ready for when we subsume a
%% real CFD code base (e.g., OpenFOAM, Athena++) with known numerical
%% defects.

kernel_available_fixes(k_compute_flux, []).
kernel_available_fixes(k_update_conservative, []).
kernel_available_fixes(k_compute_primitives, []).
kernel_available_fixes(k_cfl_condition, []).

%% No fix_description/2 facts registered yet for CFD.
