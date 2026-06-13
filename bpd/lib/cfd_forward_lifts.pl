%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% cfd_forward_lifts.pl
%% CANDIDATE for medayek review (Iyun, 2026-05-29)
%% Forward-lift work: the "needs_lift" CFD ops, lifted to Python's ACTUAL
%% evaluation order (NOT textbook math) -- per the laplacian lesson that at
%% bit_exact, associativity IS the result. Each composes already-verified
%% primitives, so bit-identity should inherit + the explicit fold/operand order.
%%
%% Verify-then-mark discipline: these are lifted but NOT yet verified bit-identical
%% against the Python oracle (no GPU run from my host). Marked needs_verify, NOT
%% lifted_verified -- medayek's harness measures ULP, then we promote.
%%
%% Depends on torch_cfd_lifted.pl (central_difference, forward_difference -- both
%% lifted_verified(0)). Does NOT modify core.
%%
%% NOTE: the eval-order is carried as a proper TERM in cfd_eval_order/2 (the
%% laplacian-lesson associativity field). cfd_composition/3 keeps the existing
%% project shape (Name, ComponentList, Output); the ORDER lives in eval_order.

%% --- centered_divergence (finite_differences.py:163-171) ---
%% Python: reduce(operator.add, [central_difference(u,dim) for dim,u in enumerate(v)])
%% => LEFT-FOLD of central_difference over dimensions, in dim order:
%%    ((cd_0 + cd_1) + ...). reduce = left-associative; order is EXPLICIT.
%% Composes central_difference (already lifted_verified 0 ULP).
:- discontiguous cfd_composition/3, cfd_eval_order/2, lift_pending/2, cfd_lift_deferred/3, cfd_construction/2, cfd_bit_exact_reason/2, cfd_branched/2.

cfd_composition(centered_divergence,
    [central_difference(v, dim_each), fold_left_add_over_dims],
    div_v).
%% associativity field (the laplacian lesson): per-dimension central_difference,
%% combined by a LEFT fold of add, dimensions in ascending order.
cfd_eval_order(centered_divergence,
    fold_left(add, map(central_difference, dims_ascending))).

%% --- curl_2d (finite_differences.py:462-469) ---
%% Python: forward_difference(v[1], dim=-2) - forward_difference(v[0], dim=-1)
%% => single sub; OPERAND ORDER significant (minuend/subtrahend, axes -2/-1).
%% Composes forward_difference (already lifted_verified 0 ULP).
cfd_composition(curl_2d,
    [forward_difference(v1, minus2),
     forward_difference(v0, minus1),
     sub(fd_v1, fd_v0)],
    curl_v).
cfd_eval_order(curl_2d,
    sub(forward_difference(v1, minus2), forward_difference(v0, minus1))).

%% --- gradient_tensor: DEFERRED (needs more care) ---
%% finite_differences.py:442+ -- overloaded, returns a rank-2 GridTensor of all
%% d v_i / d x_j via recursive torch.stack over components. NOT a simple stencil
%% composition (tensor-valued, recursive). Marked deferred, not faked.
cfd_lift_deferred(gradient_tensor, tensor_valued_recursive,
    'rank-2 GridTensor of all dv_i/dx_j; recursive stack over components; needs tensor codomain').

%% --- lift status (needs_verify, NOT yet verified bit-identical) ---
%% These supersede the needs_lift entries in cfd_retire_status.pl ONCE medayek's
%% harness measures ULP and confirms bit-identity. Until then: needs_verify.
lift_pending(centered_divergence, needs_verify).
lift_pending(curl_2d,             needs_verify).

%% =====================================================================
%% BATCH 2 (Iyun, 2026-05-29) — spectral ops, method validated (batch 1
%% verified 0 ULP by medayek). Same discipline: Python's actual order.
%% =====================================================================

%% --- spectral_rot_2d (spectral.py:72-74) ---
%% Python: vgradx,vgrady = spectral_grad_2d(vhat,mesh); return vgrady, -vgradx
%% Composes spectral_grad_2d (lifted_verified 0 ULP). Output is a PAIR with
%% the 2nd component NEGATED. Order/sign explicit.
cfd_composition(spectral_rot_2d,
    [spectral_grad_2d(vhat, rfft_mesh), pair(vgrady, neg(vgradx))],
    rot_vhat).
cfd_eval_order(spectral_rot_2d, pair(vgrady, neg(vgradx))).

%% --- brick_wall_filter_2d (spectral.py:77-86) — 2/3 dealiasing rule ---
%% Python builds a STATIC MASK: zeros((n, m//2+1)), then sets two index-range
%% blocks to 1 (the 2/3 rule). NO float arithmetic -> trivially bit-exact
%% (values are exactly 0.0 / 1.0). It is mask CONSTRUCTION, not a stencil.
%% eval_order is N/A (no reduction); recorded as construction with the ranges.
cfd_construction(brick_wall_filter_2d,
    mask(shape(n, m_half_plus_1),
         set_ones([ block(rows(0, two_thirds_n_half_plus_1), cols(0, two_thirds_m_half)),
                    block(rows(neg_two_thirds_n_half, end),   cols(0, two_thirds_m_half)) ]))).
cfd_bit_exact_reason(brick_wall_filter_2d, no_float_arithmetic_values_0_or_1).

%% --- stable_time_step (spectral.py:120-149) — CFL ---
%% Python: dt_diffusion = dx  (or dx^2/(visc*2^ndim) if NOT implicit_diffusion);
%%         dt_advection = max_courant_number * dx / max_velocity;
%%         return min(dt_diffusion, dt_advection, dt)
%% KEY: the "CFL reduction" is a 3-way MIN. min is ORDER-INSENSITIVE (selects an
%% existing value, no fp arithmetic reordering) -> NO associativity hazard, unlike
%% a SUM reduction. So bit-safe regardless of min order. The arithmetic that DOES
%% matter: dt_advection = (max_courant_number * dx) / max_velocity (mul then div).
cfd_composition(stable_time_step,
    [cond(implicit_diffusion,
          dt_diffusion_is(dx),
          dt_diffusion_is(div(dx_sq, mul(viscosity, pow(2, ndim))))),
     dt_advection_is(div(mul(max_courant_number, dx), max_velocity)),
     min3(dt_diffusion, dt_advection, dt)],
    stable_dt).
cfd_eval_order(stable_time_step, min_reduction(order_insensitive,
    [dt_diffusion, advection(div(mul(max_courant_number, dx), max_velocity)), dt])).

%% --- vorticity_to_velocity: DEFERRED ---
%% spectral.py:89+ -- returns a CLOSURE (constructs a function), solves for a
%% stream function then computes velocity. Higher-order + internal solve, not a
%% simple stencil composition. Deferred like gradient_tensor.
cfd_lift_deferred(vorticity_to_velocity, higher_order_closure_with_solve,
    'returns a function; solves stream function then computes velocity; needs closure + solve handling').

%% --- lift status (needs_verify) ---
lift_pending(spectral_rot_2d,      needs_verify).
lift_pending(brick_wall_filter_2d, needs_verify).
lift_pending(stable_time_step,     needs_verify).

%% FINDING (Iyun): MIN/MAX reductions are BIT-SAFE (selection, no fp reorder);
%% SUM reductions carry the associativity hazard (laplacian). The eval_order
%% field should distinguish reduction KIND -- min/max order-insensitive vs
%% sum/product order-sensitive. Relevant to which lifts need careful ordering.

%% =====================================================================
%% BATCH 3 (Iyun, 2026-05-29) — linear_interpolation, the last needs_lift op.
%% WEIGHTED SUM => order-sensitive (unlike min). Captured Python's exact
%% weight derivation + operand order (finite_differences.py 366-398, 399-426).
%% =====================================================================

%% --- linear (finite_differences.py:399-426) — multi-linear interpolation ---
%% Python: interpolated = c; for dim,o in enumerate(offset):
%%             interpolated = _linear_along_axis(interpolated, o, dim)
%% => LEFT-FOLD of _linear_along_axis over dims, in order (like centered_divergence).
cfd_composition(linear_interpolation,
    [linear_along_axis(c, offset_each, dim_each), fold_left_over_dims],
    interp_c).
cfd_eval_order(linear_interpolation,
    fold_left(linear_along_axis, dims_ascending)).

%% --- _linear_along_axis (finite_differences.py:366-398) — the per-axis kernel ---
%% THREE branches (the lift must carry the branch structure):
%%   1. offset_delta == 0           -> return c unchanged (no arithmetic)
%%   2. integer offset_delta        -> shift only (no float arithmetic -> bit-exact)
%%   3. fractional offset_delta     -> WEIGHTED SUM (the bit-sensitive path):
%%        floor_weight = ceil - offset_delta          (NOT offset_delta - floor)
%%        ceil_weight  = 1.0 - floor_weight           (DERIVED from floor_weight)
%%        data = floor_weight*c.shift(floor) + ceil_weight*c.shift(ceil)  (floor term FIRST)
%% The weight derivation order matters: ceil_weight depends on floor_weight, so the
%% two weights round asymmetrically. Operand order floor-then-ceil is explicit.
cfd_branched(linear_along_axis,
    [ branch(offset_delta_zero,    identity(c)),
      branch(offset_delta_integer, shift_only(c, int(offset_delta), dim)),    % bit-exact, no fp
      branch(offset_delta_fractional,
             weighted_sum(
                 floor_weight = sub(ceil, offset_delta),
                 ceil_weight  = sub(1.0, floor_weight),
                 add(mul(floor_weight, shift(c, floor, dim)),
                     mul(ceil_weight,  shift(c, ceil,  dim))))) ]).
cfd_eval_order(linear_along_axis,
    weighted_sum_floor_first(
        floor_weight(sub(ceil, offset_delta)),
        ceil_weight(sub(1.0, floor_weight)),
        add(mul(floor_weight, c_floor), mul(ceil_weight, c_ceil)))).

lift_pending(linear_interpolation, needs_verify).

%% FINDING (Iyun): linear interpolation's two branches that AVOID the weighted sum
%% (offset unchanged; integer offset -> pure shift) are bit-exact trivially; only
%% the FRACTIONAL branch carries the sum-associativity hazard. So even within one
%% op, retire-readiness can be branch-dependent. The harness should exercise the
%% FRACTIONAL path specifically (the integer/zero paths can't diverge).
