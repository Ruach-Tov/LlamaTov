%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ulp_attribution.pl — Empirical ULP bounds + attribution substrate.
%%
%% Per Heath's F2 directive: build out the within-target substrate
%% self-knowledge frontier to its final shape. After F2 closure, every
%% cross-axis ULP in the matrix is attributable to a named tunable.
%%
%% This module is the structured Prolog substrate for those facts.
%% Companion to:
%%   docs/methodology/within-target-bit-identical-baseline.md
%%   docs/methodology/six-tunables-empirical.md
%%
%% ## Term shape
%%
%% within_target_invariant(Kernel, GroupName, MaxUlp).
%%   GroupName: cpu | gpu
%%   MaxUlp: max ULP observed within the target group across hosts.
%%   For F2-closed kernels, MaxUlp should be 0.
%%
%% cross_axis_bound(Kernel, MaxUlp, AttributionList).
%%   MaxUlp: empirical max ULP observed cross-axis (cell [2] vs cell [3]
%%           or equivalent CPU-vs-GPU comparison).
%%   AttributionList: [tunable(Name, Contribution), ...] explaining the bound.
%%
%% tunable_bound(TunableName, Description, EmpiricalBound).
%%   The six tunables and their measured impact on Tesla P4.
%%
%% ## Usage
%%
%%   ?- within_target_invariant(k_gelu_tanh, gpu, MaxUlp).
%%   MaxUlp = 0.
%%
%%   ?- cross_axis_bound(k_gelu_tanh, MaxUlp, Attribution).
%%   MaxUlp = 188,
%%   Attribution = [tunable(math_library, 'CUDA tanhf vs PyTorch CPU tanhf'),
%%                  tunable(fma_usage, 'polynomial expression FMA-eligible')].
%%
%% Future kernels added to the matrix register their bounds and
%% attributions via this substrate. When a kernel exceeds its bound,
%% the regression is structurally explicable.
%%
%% Author: metayen 2026-05-18
%% Per Heath's F2 directive. F2.c — attribution substrate.

:- module(ulp_attribution, [
    within_target_invariant/3,
    cross_axis_bound/3,
    tunable_bound/3,
    attribute_ulp/3,
    explain_divergence/2
]).

:- discontiguous within_target_invariant/3.
:- discontiguous cross_axis_bound/3.


%% ─────────────────────────────────────────────────────────────────
%% WITHIN-TARGET INVARIANTS (F2.a empirical results)
%%
%% Verified 2026-05-18 ~00:15 UTC, 36/36 (op, size) pairs MATCH:
%%   cell [2] (C host GPU) vs cell [4] (Python host GPU) for activations.
%% Verified earlier (mavchin 2026-05-17): k_add cells [1]-[6] BIT-IDENTICAL.
%% ─────────────────────────────────────────────────────────────────

%% Activation column — within-GPU verified across all 6 sizes per kernel.
within_target_invariant(k_silu,       gpu, 0).
within_target_invariant(k_sigmoid,    gpu, 0).
within_target_invariant(k_relu,       gpu, 0).
within_target_invariant(k_tanh,       gpu, 0).
within_target_invariant(k_gelu_tanh,  gpu, 0).
within_target_invariant(k_gelu_erf,   gpu, 0).

%% k_add — mavchin's anchor: 5 cells × all pairwise BIT-IDENTICAL.
within_target_invariant(k_add, cpu, 0).
within_target_invariant(k_add, gpu, 0).


%% ─────────────────────────────────────────────────────────────────
%% TUNABLE BOUNDS (F2.b empirical results)
%%
%% Each tunable's empirical impact on Tesla P4 with our current
%% kernel suite. Future kernels may surface different bounds for
%% the same tunable (e.g., reduction order at length 1024 vs 16).
%% ─────────────────────────────────────────────────────────────────

tunable_bound(math_library,
    'CUDA libcudart vs PyTorch CPU libm/ATen math primitives differ',
    bound(min:0, max:8050, depends_on:operation)).
%%   expf:  ≤2 ULP
%%   tanhf: ≤188 ULP (k_gelu_tanh tail)
%%   erff:  ≤8050 ULP (k_gelu_erf at N=1024)
%%   fmaxf: 0 ULP
%%   pure arithmetic: 0 ULP

tunable_bound(fma_usage,
    'nvcc --fmad=true (default) vs --fmad=false; CPU compilers vary',
    bound(min:0, max:36, depends_on:fma_eligible_pattern)).
%%   k_gelu_tanh: 0-36 ULP between FMA-on and FMA-off (within-GPU shift)
%%   Affects 1-6 elements per 1000-element vector
%%   Mean ULP ~0.02

tunable_bound(constant_precision,
    'double literals (0.5) vs float literals (0.5f) in CUDA source',
    bound(min:0, max:1, depends_on:affected_element_fraction)).
%%   k_silu dbl vs flt constants: exactly 1 ULP per affected element
%%   Affects 16-28% of elements (varies with input distribution)
%%   Substrate-protected via c_float_f in activation_expr facts.

tunable_bound(reduction_order,
    'linear accumulation vs tree reduction; fp32 add non-associative',
    bound(min:0, max:undetermined, depends_on:reduction_length)).
%%   ggml_sum_rows: 4 ULP on 16-element row (linear vs PyTorch tree)
%%   Scales with reduction length; bound grows as kernels reduce longer

tunable_bound(optimization_flags,
    'nvcc -O0 / -O2 / -O3; differs by kernel complexity',
    bound(min:0, max:0, depends_on:kernel_complexity)).
%%   k_silu_flt at O0/O2/O3: 0 ULP across all 3 levels
%%   NULL tunable for simple elementwise; may be non-null for matmul

tunable_bound(simd_strategy,
    'AVX/SSE/NEON vector loops vs scalar (CPU only)',
    bound(deferred, until:'cell [1] (C host CPU) implementation lands')).
%%   Will be characterized when matrix extends to cell [1] for reductions.
%%   Cross-axis CPU-vs-PyTorch-CPU divergence will surface SIMD effects.


%% ─────────────────────────────────────────────────────────────────
%% CROSS-AXIS BOUNDS WITH ATTRIBUTIONS
%%
%% For each kernel in the matrix, the empirical cell [2] vs cell [3]
%% bound and the tunable(s) responsible for it.
%% ─────────────────────────────────────────────────────────────────

%% Class 1: tight transcendental (CUDA + PyTorch implementations agree closely)
cross_axis_bound(k_silu,    2,
    [attribution(math_library, primary, 'expf precision difference')]).
cross_axis_bound(k_sigmoid, 2,
    [attribution(math_library, primary, 'expf precision difference')]).
cross_axis_bound(k_tanh,    2,
    [attribution(math_library, primary, 'tanhf for simple input range')]).

%% Class 2: loose transcendental (math library implementations diverge)
cross_axis_bound(k_gelu_tanh, 188,
    [attribution(math_library, primary, 'CUDA tanhf vs PyTorch CPU tanhf'),
     attribution(fma_usage, secondary, 'polynomial 1+0.044715*x^2 FMA-eligible')]).
cross_axis_bound(k_gelu_erf, 8050,
    [attribution(math_library, primary, 'CUDA erff vs PyTorch CPU erff near zero')]).

%% Class 3: conditional / selection (no math library, bit-identical)
cross_axis_bound(k_relu, 0,
    [attribution(none, primary, 'pure conditional fmaxf; no math library')]).

%% Reduction kernels
cross_axis_bound(ggml_sum_rows, 4,
    [attribution(reduction_order, primary, 'linear vs PyTorch pairwise tree')]).
cross_axis_bound(ggml_mean, 4,
    [attribution(reduction_order, primary, 'linear vs PyTorch pairwise tree')]).
cross_axis_bound(ggml_max, 0,
    [attribution(none, primary, 'selection op; no accumulation')]).
cross_axis_bound(ggml_min, 0,
    [attribution(none, primary, 'selection op; no accumulation')]).
cross_axis_bound(ggml_argmax, 0,
    [attribution(none, primary, 'selection op; no accumulation')]).
cross_axis_bound(ggml_argmin, 0,
    [attribution(none, primary, 'selection op; no accumulation')]).


%% ─────────────────────────────────────────────────────────────────
%% QUERY PREDICATES
%% ─────────────────────────────────────────────────────────────────

%% attribute_ulp(+Kernel, -MaxUlp, -PrimaryCause)
%%
%% Convenience predicate: given a kernel name, return its empirical
%% cross-axis max ULP and the primary attributed cause.
attribute_ulp(Kernel, MaxUlp, PrimaryCause) :-
    cross_axis_bound(Kernel, MaxUlp, Attribution),
    member(attribution(PrimaryCause, primary, _Reason), Attribution).


%% explain_divergence(+Kernel, -Explanation)
%%
%% Human-readable explanation of why a kernel diverges cross-axis.
%% Returns a list of strings naming the contributing tunables.
explain_divergence(Kernel, Explanation) :-
    cross_axis_bound(Kernel, MaxUlp, Attributions),
    findall(format(Reason, MaxUlp, Cause),
        member(attribution(Cause, _Priority, Reason), Attributions),
        Explanation).
