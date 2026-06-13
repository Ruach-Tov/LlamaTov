%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% iteration_lifting.pl
%% CANDIDATE for medayek review (Iyun, 2026-05-29)
%% branch iyun/elementwise-into-reduction-fusion. NOT merged; review first.
%%
%% Lifts the ITERATION LAYER that torch_cfd_lifted.pl left in Python source.
%% The acyclic loop BODIES are already lifted (cfd_composition/3); this file
%% adds the LOOP STRUCTURE that carries state across iterations -- the layer
%% Heath flagged and medayek confirmed is a deliberate "bodies first, loops
%% later" gap.
%%
%% WHY THIS IS NOT MERE BOOKKEEPING (the cost-leverage point, per Heath):
%%   The loop-back edge (e.g. u_new -> u, every iteration) is a FUSION SEAM.
%%   Carried state is written to DRAM at iteration i end, read back at i+1
%%   start -- EVERY iteration. Loop-boundary fusion ("temporal blocking":
%%   keep carried state resident in registers/shared/L2 across iterations)
%%   eliminates that round-trip. So the seam cost is:
%%
%%       loop_boundary_traffic = sizeof(Carried) * iteration_count
%%
%%   This is the DOMINANT energy term for iterative solvers -- a Jacobi solve
%%   over K iterations re-streams the [N] state K times unless fused. CG
%%   carries 4 vectors (u,r,p,rsold) so its loop-boundary seam is ~4x Jacobi
%%   -- the carried-state field below MAKES THAT LEVERAGE VISIBLE to the cost
%%   model (multiply Carried size by iteration_count -> the seam to fuse).
%%
%% LAYERED NAME (InChI-style, per agreement): the canonical kernel name is
%%   iterate[Kind, n=Iters]( <acyclic-body-name> / shape / cost / precision )
%% Cycle isolated to this OUTERMOST wrapper; inner layers stay acyclic-canonical.
%%
%% Depends on torch_cfd_lifted.pl vocabulary (cfd_composition/3 bodies).
%% Does NOT modify core. Reviewed -> folded by medayek or kept as extension.

%% iteration_kind/2 -- the 4 kinds the corpus actually exhibits (medayek-confirmed)
iteration_kind(fixed_point,    "converge to steady state (Jacobi, Gauss-Seidel)").
iteration_kind(krylov,         "converge to solution of Ax=b (CG, BiCGSTAB)").
iteration_kind(timestep,       "advance in time (RK, Euler)").
iteration_kind(autoregressive, "generate token by token (LLM decode)").

%% iteration(+Name, +Kind, +Body, +Carried, +Termination)
%%   Name        : iteration identifier
%%   Kind        : one of iteration_kind/2
%%   Body        : the acyclic loop-body, an already-lifted cfd_composition/3 name
%%   Carried     : list of state tensors crossing the iteration boundary
%%                 (THE cost-bearing field: sizeof(Carried)*iters = loop seam)
%%   Termination : convergence(Test) | fixed_count(N) | sequence_end(Token)

%% --- Family B: stationary iterative (fixed_point) ---
%% Jacobi: carries the single solution vector u; converges on residual norm.
iteration(jacobi_solve, fixed_point,
    body(jacobi_step),
    carried([u]),
    termination(convergence(residual_norm_below(tol)))).

%% Gauss-Seidel: same carried state; red-black ordering is a body detail.
iteration(gauss_seidel_solve, fixed_point,
    body(gauss_seidel_step),
    carried([u]),
    termination(convergence(residual_norm_below(tol)))).

%% --- Family C: Krylov (krylov) ---
%% CG carries FOUR vectors -- loop-boundary seam ~4x Jacobi. (Source comment
%% in torch_cfd_lifted.pl cg_step: "returns (u, r, p, rznew)".)
iteration(cg_solve, krylov,
    body(cg_step),
    carried([u, r, p, rsold]),
    termination(convergence(residual_norm_below(tol)))).

%% --- Family A is DIRECT (NOT iterated) ---
%% poisson_solve_fft is O(N log N) single-shot FFT->mul->IFFT: a CHAIN, no
%% iteration. Recorded as a non-iterating reference so the survey doesn't
%% misclassify it as a loop.
non_iterating(poisson_solve_fft, direct_spectral).

%% --- Family D: Multigrid V-cycle -- FRONTIER (recursive, not simple loop) ---
%% V-cycle is a TREE of iterations (restrict/smooth/prolong recursion), not a
%% flat carried-state loop. Named, deferred. Needs a recursive iteration term.
iteration_frontier(multigrid_vcycle, recursive_tree,
    "V-cycle: restrict->smooth->recurse->prolong; needs recursive iteration term").

%% --- cost-leverage helper: the loop-boundary seam the solver wants to fuse ---
%% loop_boundary_seam(+IterationName, -Carried, -Multiplier)
%%   INTERFACE (per medayek 2026-05-29): this predicate returns the WHAT --
%%   which tensors cross the iteration boundary -- and the symbolic Multiplier
%%   (iteration_count). It does NOT compute bytes. The HOW MUCH (bytes with
%%   alignment/padding/layout, bandwidth, roofline impact) is owned solely by
%%   graph_complexity, which CONSUMES this carried set via its own tensor_bytes.
%%   One authoritative byte-path, no drift. Separation of concerns: my facts
%%   say which tensors; his cost model says how many joules.
loop_boundary_seam(Name, Carried, iteration_count) :-
    iteration(Name, _Kind, _Body, carried(Carried), _Term).
