%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% =============================================================================
%% kernel_templates_stencil.pl — Cell-indexed Dirichlet 3-neighbor stencils
%% =============================================================================
%%
%% Substrate for emitting CUDA kernels in the cell-indexed Dirichlet
%% 3-neighbor stencil family. See:
%%   bpd/docs/methodology/cell-indexed-dirichlet-3-neighbor-stencil-family.md
%%
%% Family members all share the substrate-emit shape:
%%
%%   __global__ void k_<name>(const float * in, float * out, int N) {
%%       int i = blockIdx.x * blockDim.x + threadIdx.x;
%%       if (i == 0 || i >= N - 1) return;  // Dirichlet skip
%%       float L = in[i - 1];
%%       float C = in[i];
%%       float R = in[i + 1];
%%       out[i] = <family-specific combination of L, C, R>;
%%   }
%%
%% Members differ ONLY in the body expression. The loop, the bound, the
%% boundary handling, the memory layout, and the per-thread structure
%% are all shared.
%%
%% PER 2026-05-18 ~17:50 UTC SCOPE DECISION (metayen, post-reconnaissance):
%%
%% This file ships ONE family member as the minimum viable representative:
%% jacobi1d (the box-filter averaging kernel, weights (1/3, 1/3, 1/3)).
%%
%% The family taxonomy lists 15+ other members across PDE solvers, linear
%% algebra, signal processing, graph algorithms, and ML. Each would be a
%% separate predicate here following the same shape. The substrate-honest
%% choice is to add them as actual needs arise rather than parametrize
%% prematurely.
%%
%% FAMILY EXTENSION AXES (named in the taxonomy doc, NOT implemented here):
%%   1. Weight vector (w_L, w_C, w_R) — Gaussian, edge detection, etc.
%%   2. Variable coefficients per cell — tridiagonal matvec
%%   3. Stencil stride — cyclic reduction
%%   4. Multi-array reads — wave equation (needs prev), Poisson (needs f)
%%   5. Stencil width — wider stencils, 2D extensions (separate family)
%%
%% When a second family member is added, the substrate-design question
%% becomes whether to refactor toward a parametrized stencil_kernel/N
%% consumer. Until then: each family member is its own predicate, and
%% the redundancy is acceptable because the family is well-named.
%%
%% RELATIONSHIP TO CFD:
%%   The CFD beachhead established a DIFFERENT stencil family (interface-
%%   indexed transmissive 2-neighbor 3-component). The two families are
%%   genuinely distinct substrate-design targets — they share the substrate
%%   c_ast primitives but not the family pattern. CFD's cfd_flux_kernel/2
%%   is NOT reusable for stencils in this family.
%% =============================================================================

:- module(kernel_templates_stencil, [
    %% Family member consumers
    jacobi1d_kernel/2,         % +KName, -Kernel
    jacobi1d_wrapper/2,        % +KName, -Wrapper

    %% Fix-flag metadata (empty for now — PDE-against-analytical doesn't
    %% subsume software, so no defects to enumerate)
    kernel_available_fixes/2,
    fix_description/2
]).

:- use_module(c_ast).

:- dynamic kernel_available_fixes/2.
:- dynamic fix_description/2.
:- discontiguous kernel_available_fixes/2.
:- discontiguous fix_description/2.


%% =============================================================================
%% JACOBI1D — The family's minimum viable representative
%% =============================================================================
%%
%% Algorithm: out[i] = (in[i-1] + in[i] + in[i+1]) / 3
%%
%% This is the box-filter averaging stencil. It's the simplest member of
%% the family and serves as the substrate's verification that the family
%% pattern emits cleanly.
%%
%% Boundary handling: Dirichlet. Boundary cells (i=0, i=N-1) are NOT
%% written; they retain whatever values the input had at those positions.
%% The caller's responsibility to ensure boundary values are preserved
%% across iterations (e.g., by initializing both `in` and `out` buffers
%% with the same boundary state before the first iteration).
%%
%% Used in:
%%   - 1D smoothing iterations (PDE solver pre-conditioning)
%%   - Iterative Poisson solver as the relaxation step
%%   - 1D box filter in signal processing
%%   - 1D average pooling in ML (with kernel size 3, stride 1)
%%
%% Verification (Python reference + harness will verify):
%%   - Symmetric IC (constant value c) → out = c at every interior cell
%%   - Boundary preservation: out[0] and out[N-1] are not modified
%%   - Sinusoidal IC: analytical decay rate (2/3 + cos(2π/N)/3) per iteration

jacobi1d_kernel(KName, Kernel) :-
    Kernel = c_func(['__global__'], c_type(void), KName,
        [param(c_type(const_restrict_ptr(c_type(float))), 'in'),
         param(c_type(restrict_ptr(c_type(float))), out),
         param(c_type(int), 'N')],
        [%% int i = blockIdx.x * blockDim.x + threadIdx.x;
         c_decl_init(c_type(int), i,
             c_binop('+',
                 c_binop('*',
                     c_member(c_var(blockIdx), x),
                     c_member(c_var(blockDim), x)),
                 c_member(c_var(threadIdx), x))),
         %% Dirichlet skip: boundary cells (i=0 or i>=N-1) are not written.
         %% Combined into one return: if i <= 0 || i >= N - 1 then return.
         %% (i can be < 0 if launched with too-wide grid, defensively
         %% guarded.)
         c_if(c_binop('||',
                  c_binop('<=', c_var(i), c_int(0)),
                  c_binop('>=', c_var(i),
                      c_binop('-', c_var('N'), c_int(1)))),
              [c_return_void]),
         %% Family-pattern neighborhood reads:
         %%   float L = in[i - 1];   left neighbor
         %%   float C = in[i];       center
         %%   float R = in[i + 1];   right neighbor
         c_decl_init(c_type(float), 'L',
             c_index(c_var('in'), c_binop('-', c_var(i), c_int(1)))),
         c_decl_init(c_type(float), 'C',
             c_index(c_var('in'), c_var(i))),
         c_decl_init(c_type(float), 'R',
             c_index(c_var('in'), c_binop('+', c_var(i), c_int(1)))),
         %% Jacobi1D-specific body: out[i] = (L + C + R) / 3.0f
         %%
         %% Note the c_paren around the numerator. The substrate-precedence
         %% audit (commit 890bfc986) found that c_binop('/', N, D) emits
         %% without parens, so multi-term denominators (or numerators that
         %% combine with later context) need explicit c_paren. Here the
         %% denominator is a single literal, but the numerator MUST be
         %% paren'd because (L + C + R) / 3.0f and L + C + R / 3.0f parse
         %% differently in C.
         c_assign(c_index(c_var(out), c_var(i)),
             c_binop('/',
                 c_paren(c_binop('+',
                     c_binop('+', c_var('L'), c_var('C')),
                     c_var('R'))),
                 c_float_f(3.0)))]).


%% jacobi1d_wrapper(+KName, -Wrapper)
%%
%% C-API wrapper for ctypes loading via the harness. Matches the ML and
%% CFD wrapper conventions: gpu_<name> with launch geometry.

jacobi1d_wrapper(KName, Wrapper) :-
    atom_concat('k_', Suffix, KName),
    atom_concat('gpu_', Suffix, WName),
    Wrapper = c_func(c_type(void), WName,
        [param(c_type(const_restrict_ptr(c_type(float))), 'in'),
         param(c_type(restrict_ptr(c_type(float))), out),
         param(c_type(int), 'N')],
        [c_cuda_launch(KName,
            c_binop('/',
                c_paren(c_binop('+', c_var('N'), c_int(255))),
                c_int(256)),
            c_int(256),
            [c_var('in'), c_var(out), c_var('N')])]).


%% =============================================================================
%% FIX-FLAG METADATA
%% =============================================================================
%%
%% Per the substrate-honesty convention: stencils in this family are
%% PDE-against-analytical (or convolution-against-mathematical-definition,
%% depending on the member). The reference IS the mathematics, not another
%% implementation. No defects to inherit, no fixes to enumerate.
%%
%% The fix-flag mechanism is READY for if/when we later subsume a specific
%% production implementation (e.g., a PolyBench/GPU reference with known
%% numerical defects).

kernel_available_fixes(k_jacobi1d, []).

%% No fix_description/2 facts registered yet for this family.
