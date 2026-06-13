%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% cfd_retire_status.pl — CFD stencil operation retire/verify status
%% Table(10010) — queryable Prolog facts for the CFD dashboard
%%
%% Status types:
%%   lifted_verified(ULP)           — lifted to BPD, verified at ULP
%%   lifted_divergent(ULP, Cause)   — lifted but diverges from reference
%%   needs_lift                     — not yet lifted
%%   frontier                       — active work target

:- module(cfd_retire_status, [retire_status/2]).

%% Verified at 0 ULP — these are "blue" on the dashboard
retire_status(gradient_x,       lifted_verified(0)).
retire_status(gradient_y,       lifted_verified(0)).
retire_status(gradient_z,       lifted_verified(0)).
retire_status(divergence,       lifted_verified(0)).
retire_status(curl_x,           lifted_verified(0)).
retire_status(curl_y,           lifted_verified(0)).
retire_status(curl_z,           lifted_verified(0)).
retire_status(dot_product,      lifted_verified(0)).
retire_status(cross_product_x,  lifted_verified(0)).
retire_status(cross_product_y,  lifted_verified(0)).

%% Divergent — lifted but doesn't match reference
retire_status(laplacian,        lifted_divergent(256, reduction_order)).

%% Needs lift — not yet in BPD
retire_status(advection,        needs_lift).
retire_status(diffusion,        needs_lift).

%% Pending verification — lifted but not yet tested
retire_status(poisson_jacobi,   needs_lift).
retire_status(pressure_correct, needs_lift).
retire_status(vorticity,        needs_lift).
retire_status(strain_rate,      needs_lift).
retire_status(boundary_neumann, needs_lift).

%% Frontier — active work
retire_status(navier_stokes_rhs, frontier).
retire_status(turbulence_model,  frontier).
