%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
:- module(derive_epilogues, [
    derive_epilogue/3,
    derive_chain/3,
    activation_def/2,
    demonstrate_derivation/0,
    emit_c/2
]).


%% clauses grouped by family, not contiguous — declared for warning-free consult.
:- discontiguous classify_bottleneck/2, codegen_strategy/2.
%% ═══════════════════════════════════════════════════════════════
%% Primitive C operations — the atoms of our expression language.
%% Each c_prim maps a symbolic function to a C AST node.
%% ═══════════════════════════════════════════════════════════════

c_prim(max, [A, B], c_call(fmaxf, [A, B])).
c_prim(min, [A, B], c_call(fminf, [A, B])).
c_prim(exp, [A],    c_call(expf, [A])).
c_prim(log, [A],    c_call(logf, [A])).
c_prim(log1p, [A],  c_call(log1pf, [A])).
c_prim(tanh, [A],   c_call(tanhf, [A])).
c_prim(abs, [A],    c_call(fabsf, [A])).
c_prim(erf, [A],    c_call(erff, [A])).
c_prim(expm1, [A],  c_call(expm1f, [A])).
c_prim(neg, [A],    c_unop('-', A)).
c_prim(add, [A, B], c_binop('+', A, B)).
c_prim(sub, [A, B], c_binop('-', A, B)).
c_prim(mul, [A, B], c_binop('*', A, B)).
c_prim(div, [A, B], c_binop('/', A, B)).
c_prim(gt, [A, B],  c_binop('>', A, B)).
c_prim(ternary, [C, T, F], c_ternary(C, T, F)).

%% ═══════════════════════════════════════════════════════════════
%% Activation definitions as symbolic math.
%% Each activation_def(Name, Expr) uses the variable 'x' as input.
%% ═══════════════════════════════════════════════════════════════

activation_def(relu,        max(const(0.0), x)).
activation_def(silu,        div(x, add(const(1.0), exp(neg(x))))).
activation_def(swish, E) :- activation_def(silu, E).
activation_def(sigmoid,     div(const(1.0), add(const(1.0), exp(neg(x))))).
activation_def(tanh,        tanh(x)).
%% GELU (exact): 0.5 * x * (1 + erf(x / sqrt(2)))
%% Matches PyTorch's default GELU which uses erff, not the tanh approximation.
activation_def(gelu,        mul(mul(const(0.5), x),
                                add(const(1.0),
                                    erf(mul(x, const(0.7071067811865476)))))).
activation_def(mish,        mul(x, tanh(log1p(exp(x))))).
activation_def(leaky_relu,  ternary(gt(x, const(0.0)), x, mul(const(0.01), x))).
%% ELU: use expm1 for numerical stability near zero (matches PyTorch)
activation_def(elu,         ternary(gt(x, const(0.0)), x, expm1(x))).
activation_def(hardswish,   mul(x, div(min(max(add(x, const(3.0)), const(0.0)), const(6.0)), const(6.0)))).
activation_def(hardsigmoid, div(min(max(add(x, const(3.0)), const(0.0)), const(6.0)), const(6.0))).
activation_def(hardtanh,    min(max(x, const(-1.0)), const(1.0))).
activation_def(abs,         abs(x)).
activation_def(neg,         neg(x)).
activation_def(softplus,    log1p(exp(x))).
%% SELU: matches PyTorch's float-truncated constants and expm1
activation_def(selu,        ternary(gt(x, const(0.0)),
                                mul(const(1.0507010), x),
                                mul(const(1.7580993), expm1(x)))).
%% Note: 1.7580993 = alpha * scale = 1.6732632 * 1.0507009
%% PyTorch computes negcoef = alpha * scale at float precision.

%% ═══════════════════════════════════════════════════════════════
%% The DERIVATION ENGINE: symbolic math → C AST
%% This is the core program transform.
%% ═══════════════════════════════════════════════════════════════

derive_expr(x, InputVar, InputVar).
derive_expr(const(V), _, c_float_f(V)).
derive_expr(Expr, InputVar, CAST) :-
    Expr \= x,
    Expr \= const(_),
    Expr =.. [Func | Args],
    maplist(derive_one(InputVar), Args, CArgs),
    c_prim(Func, CArgs, CAST).

derive_one(InputVar, Arg, CArg) :-
    derive_expr(Arg, InputVar, CArg).

%% Public API
derive_epilogue(Name, InputVar, CAST) :-
    activation_def(Name, MathExpr),
    derive_expr(MathExpr, InputVar, CAST).

derive_chain([], Var, Var).
derive_chain([Act | Rest], InputVar, CAST) :-
    derive_epilogue(Act, InputVar, Mid),
    derive_chain(Rest, Mid, CAST).

%% ═══════════════════════════════════════════════════════════════
%% Demonstration
%% ═══════════════════════════════════════════════════════════════

demonstrate_derivation :-
    format("=== Derived Epilogue Expressions ===~n~n"),
    forall(
        (   member(Act, [relu, silu, sigmoid, tanh, gelu, mish,
                         leaky_relu, elu, hardswish, hardsigmoid,
                         hardtanh, abs, neg, softplus, selu]),
            derive_epilogue(Act, c_var(x), CAST)),
        format("  ~w(x) = ~w~n", [Act, CAST])
    ),
    nl,
    format("=== Derived Chain Compositions ===~n~n"),
    forall(
        (   member((Label, Ops), [
                ("Conv+ReLU+HardSwish", [relu, hardswish]),
                ("Conv+Tanh+HardSwish", [tanh, hardswish]),
                ("Linear+Hardtanh+GELU", [hardtanh, gelu]),
                ("ReLU+Sigmoid+Neg", [relu, sigmoid, neg])
            ]),
            derive_chain(Ops, c_var(x), CAST)),
        format("  ~w:~n    ~w~n~n", [Label, CAST])
    ).

%% ═══════════════════════════════════════════════════════════════
%% C AST Pretty-Printer
%% ═══════════════════════════════════════════════════════════════

emit_c(S, c_var(Name)) :- format(S, "~w", [Name]).
emit_c(S, c_float_f(V)) :- format(S, "~wf", [V]).
emit_c(S, c_call(Func, Args)) :-
    format(S, "~w(", [Func]),
    emit_c_args(S, Args),
    write(S, ')').
emit_c(S, c_binop(Op, L, R)) :-
    write(S, '('),
    emit_c(S, L),
    format(S, " ~w ", [Op]),
    emit_c(S, R),
    write(S, ')').
emit_c(S, c_unop(Op, A)) :-
    format(S, "(~w", [Op]),
    emit_c(S, A),
    write(S, ')').
emit_c(S, c_ternary(Cond, T, F)) :-
    write(S, '('),
    emit_c(S, Cond),
    write(S, ' ? '),
    emit_c(S, T),
    write(S, ' : '),
    emit_c(S, F),
    write(S, ')').

emit_c_args(_, []).
emit_c_args(S, [A]) :- emit_c(S, A).
emit_c_args(S, [A, B|Rest]) :- emit_c(S, A), write(S, ', '), emit_c_args(S, [B|Rest]).

%% ═══════════════════════════════════════════════════════════════
%% Bottleneck classification for SIMD dispatch decisions.
%% Memory-bound ops benefit from SIMD (load/store throughput).
%% Transcendental-bound ops do NOT benefit from SIMD when the
%% SIMD path requires scalar extract/reload for libm calls.
%% ═══════════════════════════════════════════════════════════════

:- export(classify_bottleneck/2).
:- export(dispatch_simd/2).

%% Memory-bound: the computation is trivial, memory bandwidth limits.
%% SIMD gives 8x throughput on loads/stores/simple arithmetic.
classify_bottleneck(relu, memory).
classify_bottleneck(neg, memory).
classify_bottleneck(abs, memory).
classify_bottleneck(add, memory).
classify_bottleneck(mul, memory).
classify_bottleneck(sub, memory).
classify_bottleneck(div, memory).
classify_bottleneck(clamp, memory).
classify_bottleneck(hardtanh, memory).
classify_bottleneck(leaky_relu, memory).
classify_bottleneck(hardsigmoid, memory).
classify_bottleneck(hardswish, memory).
classify_bottleneck(dropout, memory).

%% Transcendental-bound: expf/tanhf/erff dominates.
%% SIMD extract/reload overhead exceeds arithmetic benefit.
classify_bottleneck(silu, transcendental).
classify_bottleneck(sigmoid, transcendental).
classify_bottleneck(tanh, transcendental).
classify_bottleneck(gelu, transcendental).
classify_bottleneck(mish, transcendental).
classify_bottleneck(softplus, transcendental).
classify_bottleneck(elu, transcendental).
classify_bottleneck(selu, transcendental).
classify_bottleneck(exp, transcendental).

%% Dispatch decision: use SIMD for memory-bound, scalar for transcendental.
dispatch_simd(Op, yes) :- classify_bottleneck(Op, memory).
dispatch_simd(Op, no)  :- classify_bottleneck(Op, transcendental).

%% ═══════════════════════════════════════════════════════════════
%% Code generation strategy: avx1 vs scalar
%%
%% This is a SWEEPABLE PARAMETER per op. The sweep harness:
%%   1. Generates BOTH versions (avx1 and scalar)
%%   2. Benchmarks both on the target hardware
%%   3. Records the winner as a Prolog fact
%%   4. Future code generation uses the recorded decision
%%
%% The decision is hardware-specific. On Ivy Bridge (AVX1, no FMA):
%%   memory-bound ops → avx1 wins
%%   transcendental ops → scalar wins
%% On Haswell+ (AVX2+FMA): both might benefit from SIMD polynomial exp.
%% ═══════════════════════════════════════════════════════════════

:- export(codegen_strategy/2).
:- export(sweep_codegen_strategy/3).

%% Default strategy: derived from bottleneck classification.
%% These can be OVERRIDDEN by sweep results stored as facts.
codegen_strategy(Op, avx1) :-
    classify_bottleneck(Op, memory), !.
codegen_strategy(Op, scalar) :-
    classify_bottleneck(Op, transcendental), !.
codegen_strategy(_, scalar).  % fallback

%% After sweeping, the harness asserts measured facts:
%% sweep_codegen_strategy(relu, avx1, 1.79).   % 1.79x faster than scalar
%% sweep_codegen_strategy(silu, scalar, 1.03).  % scalar 1.03x faster than avx1
%%
%% If a sweep fact exists, it overrides the default:
%% codegen_strategy(Op, Strategy) :-
%%     sweep_codegen_strategy(Op, Strategy, _), !.

%% The code generator queries:
%%   codegen_strategy(relu, Strategy),
%%   (Strategy = avx1 -> emit_avx1_loop(CAST) ; emit_scalar_loop(CAST))

%% Placeholder for sweep results (populated by benchmark harness)
:- discontiguous sweep_codegen_strategy/3.

%% ═══════════════════════════════════════════════════════════════
%% emit_loop/3 — unified code emitter with strategy parameter
%%
%% emit_loop(+Stream, +Strategy, +CAST)
%%   Strategy ∈ {avx1, scalar}
%%   CAST = the C AST expression from derive_epilogue
%%
%% Both strategies produce BIT_IDENTICAL output.
%% The choice is PURE PERFORMANCE — a sweepable parameter.
%% ═══════════════════════════════════════════════════════════════

:- export(emit_loop/3).
:- export(emit_kernel/4).

emit_loop(S, scalar, CAST) :-
    format(S, "    for (int i = 0; i < n; i++) {~n", []),
    format(S, "        float x = input[i];~n", []),
    format(S, "        output[i] = ", []),
    emit_c(S, CAST),
    format(S, ";~n    }~n", []).

emit_loop(S, avx1, CAST) :-
    format(S, "    int i = 0;~n", []),
    format(S, "#if BPD_HAVE_AVX1~n", []),
    format(S, "    for (; i + 7 < n; i += 8) {~n", []),
    format(S, "        __m256 x = _mm256_loadu_ps(input + i);~n", []),
    format(S, "        __m256 y = ", []),
    emit_c_avx(S, CAST),
    format(S, ";~n", []),
    format(S, "        _mm256_storeu_ps(output + i, y);~n", []),
    format(S, "    }~n", []),
    format(S, "#endif~n", []),
    format(S, "    for (; i < n; i++) {~n", []),
    format(S, "        float x = input[i];~n", []),
    format(S, "        output[i] = ", []),
    emit_c(S, CAST),
    format(S, ";~n    }~n", []).

%% emit_kernel/4 — generate a complete kernel function
%% emit_kernel(+Stream, +Name, +Strategy, +CAST)
emit_kernel(S, Name, Strategy, CAST) :-
    format(S, "void ~w(const float* input, float* output, int n) {~n", [Name]),
    emit_loop(S, Strategy, CAST),
    format(S, "}~n~n", []).

%% emit_c_avx/2 — emit AVX1 intrinsics from C AST
%% Maps scalar ops to AVX1 equivalents
emit_c_avx(S, c_var(x)) :- write(S, 'x').
emit_c_avx(S, c_float_f(V)) :- format(S, "_mm256_set1_ps(~wf)", [V]).
emit_c_avx(S, c_call(fmaxf, [A, B])) :-
    write(S, '_mm256_max_ps('),
    emit_c_avx(S, A), write(S, ', '),
    emit_c_avx(S, B), write(S, ')').
emit_c_avx(S, c_call(fminf, [A, B])) :-
    write(S, '_mm256_min_ps('),
    emit_c_avx(S, A), write(S, ', '),
    emit_c_avx(S, B), write(S, ')').
emit_c_avx(S, c_binop('+', A, B)) :-
    write(S, '_mm256_add_ps('),
    emit_c_avx(S, A), write(S, ', '),
    emit_c_avx(S, B), write(S, ')').
emit_c_avx(S, c_binop('-', A, B)) :-
    write(S, '_mm256_sub_ps('),
    emit_c_avx(S, A), write(S, ', '),
    emit_c_avx(S, B), write(S, ')').
emit_c_avx(S, c_binop('*', A, B)) :-
    write(S, '_mm256_mul_ps('),
    emit_c_avx(S, A), write(S, ', '),
    emit_c_avx(S, B), write(S, ')').
emit_c_avx(S, c_binop('/', A, B)) :-
    write(S, '_mm256_div_ps('),
    emit_c_avx(S, A), write(S, ', '),
    emit_c_avx(S, B), write(S, ')').
emit_c_avx(S, c_unop('-', A)) :-
    write(S, '_mm256_xor_ps('),
    emit_c_avx(S, A),
    write(S, ', _mm256_set1_ps(-0.0f))').
emit_c_avx(S, c_call(fabsf, [A])) :-
    write(S, '_mm256_and_ps('),
    emit_c_avx(S, A),
    write(S, ', _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)))').

%% ═══════════════════════════════════════════════════════════════
%% BLAS strategy classification
%%
%% BLAS-bound ops have multiple implementation tiers:
%%   scalar:   naive triple loop (reference, always bit-identical)
%%   tiled_v2: K-blocked with register tiling (our AVX1 GEMM)
%%   blas:     external BLAS library (MKL, OpenBLAS, BLIS)
%%
%% The strategy is a sweepable parameter per op AND per shape.
%% Different shapes may prefer different strategies:
%%   Small matmul (32x256x512): tiled_v2 may beat BLAS (no library overhead)
%%   Large matmul (1024²): BLAS wins (hand-tuned microkernel)
%% ═══════════════════════════════════════════════════════════════

:- export(blas_strategy/2).

classify_bottleneck(matmul, blas).
classify_bottleneck(linear, blas).
classify_bottleneck(conv2d, blas).    % im2col + GEMM
classify_bottleneck(conv3d, blas).
classify_bottleneck(conv_transpose2d, blas).
classify_bottleneck(batchnorm, memory). % per-channel affine — memory-bound

%% Available BLAS strategies (ordered by expected performance)
blas_strategy(scalar, reference).     % naive loop, always correct
blas_strategy(tiled_v2, substrate).   % our AVX1 K-blocked GEMM
blas_strategy(blas_openblas, external). % OpenBLAS (if available)
blas_strategy(blas_mkl, external).    % Intel MKL (if available)

%% Default: use our tiled GEMM (no external dependency)
codegen_strategy(matmul, tiled_v2) :- !.
codegen_strategy(linear, tiled_v2) :- !.
codegen_strategy(conv2d, tiled_v2) :- !.  % im2col + tiled GEMM

%% With MKL available:
%% codegen_strategy(matmul, blas_mkl) :- blas_available(mkl), !.
%% codegen_strategy(linear, blas_mkl) :- blas_available(mkl), !.

%% The emit_kernel for BLAS ops:
%% emit_kernel(S, Name, tiled_v2, matmul_args) →
%%     calls bpd_gemm_v2_full(A, B, C, M, N, K)
%% emit_kernel(S, Name, blas_mkl, matmul_args) →
%%     calls cblas_sgemm(CblasRowMajor, ...)
%% emit_kernel(S, Name, scalar, matmul_args) →
%%     naive triple loop

%% Shape-dependent strategy override (from sweep):
%% sweep_blas_strategy(matmul, shape(1, 2048, 2048), blas_mkl, 19.0).
%% sweep_blas_strategy(matmul, shape(32, 256, 512), tiled_v2, 1.2).
:- discontiguous sweep_blas_strategy/4.

%% ═══════════════════════════════════════════════════════════════
%% Division strategy: div vs mul-by-reciprocal
%%
%% div(x, const(6.0)) can be implemented two ways:
%%   x / 6.0f              — IEEE-754 division (exact)
%%   x * (1.0f / 6.0f)     — multiply by reciprocal (faster, 1 ULP different)
%%
%% Both are valid. The choice is a sweepable parameter:
%%   division_strategy ∈ {ieee_div, mul_recip}
%%
%% ieee_div:   bit-identical to scalar division
%% mul_recip:  ~1 ULP difference, but vdivps is slow on some hardware
%%             (Ivy Bridge: vdivps = 21-29 cycles, vmulps = 5 cycles)
%% ═══════════════════════════════════════════════════════════════

:- export(division_strategy/1).
:- export(emit_div_avx/4).

%% Default: ieee_div (matches PyTorch, bit-identical)
division_strategy(ieee_div).

%% The code generator queries this when emitting division:
%%
%% emit_div_avx(S, A_expr, B_const, Strategy)
%%   ieee_div:   _mm256_div_ps(A, _mm256_set1_ps(B))
%%   mul_recip:  _mm256_mul_ps(A, _mm256_set1_ps(1.0f/B))

emit_div_avx(S, A, B, ieee_div) :-
    format(S, "_mm256_div_ps(~w, _mm256_set1_ps(~wf))", [A, B]).
emit_div_avx(S, A, B, mul_recip) :-
    Recip is 1.0 / B,
    format(S, "_mm256_mul_ps(~w, _mm256_set1_ps(~wf))", [A, Recip]).

%% When generating the AVX version of a div node:
%% emit_c_avx(S, c_binop('/', A, c_float_f(V))) :-
%%     division_strategy(Strategy),
%%     emit_div_avx(S, ..., V, Strategy).

%% ═══════════════════════════════════════════════════════════════
%% Broadcast epilogue pattern — per-channel constants
%%
%% Some ops apply per-element math with CHANNEL-BROADCAST constants:
%%   batchnorm:  y = scale[c] * x + offset[c]
%%   groupnorm:  y = scale[g] * x + offset[g]
%%   instancenorm: y = scale[c] * x + offset[c]
%%
%% The code generator uses emit_broadcast_loop/4 for these:
%%   emit_broadcast_loop(Stream, Strategy, CAST, BroadcastVars)
%%
%% BroadcastVars = [scale_c, offset_c] — variables that change per-channel
%% The AVX1 version broadcasts these once per channel, then applies
%% the epilogue expression to all HW elements in that channel.
%% ═══════════════════════════════════════════════════════════════

:- export(broadcast_epilogue_def/2).
:- export(emit_broadcast_kernel/4).

%% Mathematical definitions of broadcast epilogues
broadcast_epilogue_def(batchnorm, add(mul(scale_c, x), offset_c)).
broadcast_epilogue_def(groupnorm, add(mul(scale_g, x), offset_g)).
broadcast_epilogue_def(instancenorm, add(mul(scale_c, x), offset_c)).

%% The code generator derives the AVX1 expression from the definition:
%%   broadcast_epilogue_def(batchnorm, add(mul(scale_c, x), offset_c))
%%   → derive_expr → c_binop('+', c_binop('*', c_var(scale_c), c_var(x)), c_var(offset_c))
%%   → emit_c_avx → _mm256_add_ps(_mm256_mul_ps(vs, x), vo)
%%
%% The broadcast variables (scale_c, offset_c) are set1_ps constants
%% that change per-channel in the outer loop.
%% The non-broadcast variable (x) is loaded per-element.
%%
%% This is the SAME derive_expr + emit_c_avx pipeline — the only
%% difference is that some c_var nodes are broadcast (set1_ps once
%% per channel) vs loaded (loadu_ps per element).

emit_broadcast_kernel(S, Name, Strategy, _BroadcastDef) :-
    broadcast_epilogue_def(Name, MathExpr),
    derive_expr(MathExpr, c_var(x), CAST),
    format(S, "/* Auto-generated broadcast epilogue for ~w */~n", [Name]),
    format(S, "/* Math: ~w */~n", [MathExpr]),
    format(S, "/* C AST: ~w */~n", [CAST]),
    format(S, "/* Strategy: ~w */~n", [Strategy]),
    format(S, "/* Broadcast vars change per-channel, x loads per-element */~n~n", []).
