%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% epilogue_generator.pl — Generate fused CUDA epilogue from BPD facts
%%
%% Takes a fusion plan from auto_fuser.pl and elem_op/4 facts,
%% generates the fused epilogue code as c_ast terms.

:- module(epilogue_generator, [
    generate_epilogue/3,       % +EpilogueOps, +InputVar, -ASTStatements
    generate_fused_kernel/3,   % +FusionPlan, +BaseKernelName, -KernelAST
    epilogue_expr/3,
    chain_ops/3,           % +OpName, +InputExpr, -OutputExpr
    demonstrate_epilogue/0
]).

%% ═══════════════════════════════════════
%% Epilogue expression for each operation
%% Maps op name → c_ast expression transform
%% ═══════════════════════════════════════

%% Each epilogue_expr takes an input expression and returns
%% the output expression. These compose by chaining.

%% Unary activations
epilogue_expr(relu, In, c_call(fmaxf, [c_float_f(0.0), In])).
epilogue_expr(silu, In, c_binop('/', In,
    c_paren(c_binop('+', c_float_f(1.0),
        c_call(expf, [c_unop('-', In)]))))).
epilogue_expr(swish, In, Expr) :- epilogue_expr(silu, In, Expr).  % swish = silu
epilogue_expr(sigmoid, In, c_binop('/', c_float_f(1.0),
    c_paren(c_binop('+', c_float_f(1.0),
        c_call(expf, [c_unop('-', In)]))))).
epilogue_expr(tanh, In, c_call(tanhf, [In])).
epilogue_expr(gelu, In,
    c_binop('*',
        c_binop('*', c_float_f(0.5), In),
        c_paren(c_binop('+', c_float_f(1.0),
            c_call(tanhf, [c_binop('*', c_float_f(0.7978845608),
                c_paren(c_binop('+', In,
                    c_binop('*', c_float_f(0.044715),
                        c_binop('*', In, c_binop('*', In, In))))))]))))).
epilogue_expr(neg, In, c_unop('-', In)).
epilogue_expr(abs, In, c_call(fabsf, [In])).
epilogue_expr(exp, In, c_call(expf, [In])).
epilogue_expr(log, In, c_call(logf, [In])).

%% Binary ops (need a second operand from memory)
epilogue_expr(bias_add, In, c_binop('+', In, c_index(c_var(bias), c_var(col)))).
epilogue_expr(scale, In, c_binop('*', In, c_var(alpha))).
epilogue_expr(add, In, c_binop('+', In, c_index(c_var(residual), c_var(idx)))).
epilogue_expr(mul, In, c_binop('*', In, c_index(c_var(gate), c_var(idx)))).

%% BN-affine-fused epilogue (per-channel scale+offset, precomputed on host).
%%
%% Per Heath's spell-of-computation direction 2026-05-20 ~03:40 UTC: this is
%% the substrate-design move that enables YOLO Conv+BN+activation fusion.
%%
%% In eval mode, BN reduces to per-channel affine:
%%   y = γ[c] / sqrt(σ²[c] + ε) * (x - μ[c]) + β[c]
%%
%% Algebraically collapses to:
%%   y = bn_scale[c] * x + bn_offset[c]
%% where:
%%   bn_scale[c]  = γ[c] / sqrt(σ²[c] + ε)         (precomputed on host)
%%   bn_offset[c] = β[c] - μ[c] * bn_scale[c]      (precomputed on host)
%%
%% Substantive substrate-design property: the per-channel scale/offset are
%% computed ONCE on the host before kernel launch, so the kernel sees only
%% two const float* arrays. The per-element math is 1 FMA + activation.
%%
%% The substrate-design vocabulary names this as bn_affine_fused/3 — it
%% uses 'c_out' as the channel index (matching conv-2d-forward kernel naming)
%% so the epilogue composes cleanly into the conv's output-write step.
epilogue_expr(bn_affine_fused, In,
    c_binop('+',
        c_binop('*', In, c_index(c_var(bn_scale), c_var(c_out))),
        c_index(c_var(bn_offset), c_var(c_out)))).

%% Clamping
epilogue_expr(clamp, In, c_call(fminf, [c_var(clamp_max),
    c_call(fmaxf, [c_var(clamp_min), In])])).
epilogue_expr(hardtanh, In, c_call(fminf, [c_float_f(1.0),
    c_call(fmaxf, [c_float_f(-1.0), In])])).

%% Mish: x * tanh(softplus(x)) = x * tanh(log1p(exp(x)))
%%
%% Per substrate-design correction 2026-05-20 ~03:50 UTC (spell-of-computation
%% inspection of fused Conv+BN+Mish chain): this fused-epilogue form must use
%% log1pf, not logf(1.0 + expf(x)). The standalone elem_op(k_mish_blas) was
%% corrected in commit adfd6c4 (539,225 ULP -> 0 ULP). The fused mish epilogue
%% needed the same correction to maintain bit-identity in chained kernels.
%%
%% Without this fix: Conv+BN+Mish (YOLOv4 CBA) would diverge from PyTorch by
%% ~6-figure ULP. With this fix: full YOLOv4 CBA chain remains BIT_IDENTICAL.
epilogue_expr(mish, In,
    c_binop('*', In,
        c_call(tanhf, [c_call(log1pf, [c_call(expf, [In])])]))).

%% ═══════════════════════════════════════
%% Chain epilogue expressions
%% ═══════════════════════════════════════

%% generate_epilogue(+Ops, +InputVar, -Statements)
%% Produces a list of c_ast statements that chain the operations.
%% Each op transforms a running 'val' variable.

generate_epilogue(Ops, InputExpr, Statements) :-
    %% Start: float val = InputExpr;
    chain_ops(Ops, InputExpr, FinalExpr),
    Statements = FinalExpr.

%% Chain: apply each op in sequence
%%
%% Per medayek's substrate-design discipline 2026-05-20 ~03:56 UTC:
%% wrap intermediate expressions in c_paren so C operator precedence
%% doesn't corrupt composition. The substrate-design property: the
%% composed expression must be ALGEBRAICALLY EQUIVALENT to the sequential
%% application of the same ops. Without parenthesization, an op that
%% expects to consume the full previous result can have its argument
%% silently truncated by precedence (e.g., 'x*scale + offset' consumed
%% as 'x*scale' when the consuming op has '+' at top level).
%%
%% This is the substrate-design fix the TDD spike surfaced. Tier 1.5
%% verification (fused == unfused) catches the corruption; chain_ops
%% prevents it.
chain_ops([], Expr, Expr).
chain_ops([Op|Rest], InExpr, FinalExpr) :-
    epilogue_expr(Op, InExpr, MidExpr),
    wrap_for_composition(MidExpr, WrappedMidExpr),
    chain_ops(Rest, WrappedMidExpr, FinalExpr).

%% wrap_for_composition: defensive parenthesization for chain composition.
%% Binary and unary operators need parens when consumed inside another
%% operator's expression (because C operator precedence can rebracket).
%% Atomic forms (variables, calls, indexes, literals) don't need wrapping.
wrap_for_composition(c_binop(O, A, B), c_paren(c_binop(O, A, B))).
wrap_for_composition(c_unop(O, A), c_paren(c_unop(O, A))).
wrap_for_composition(Expr, Expr).

%% ═══════════════════════════════════════
%% Generate fused kernel
%% ═══════════════════════════════════════

%% For a spatial_with_epilogue plan:
%% Takes the matmul output register c[r][cl] and applies epilogue ops.
generate_fused_kernel(kernel(spatial_with_epilogue, [_Spatial|EpiOps], _), _BaseName, EpilogueStatements) :-
    %% The input is the matmul accumulator (register variable)
    chain_ops(EpiOps, c_var(val), FusedExpr),
    %% Generate the store with fused expression
    EpilogueStatements = [
        c_raw('float val = c[r][cl];'),
        c_assign(c_var(val), FusedExpr),
        c_raw('C[(row0+r)*N + col0+cl] = val;')
    ].

%% ═══════════════════════════════════════
%% Demonstration
%% ═══════════════════════════════════════

demonstrate_epilogue :-
    format("═══ EPILOGUE GENERATOR ═══~n~n"),

    %% L2 #76: Gemm + bias_add + relu
    chain_ops([bias_add, relu], c_var(val), Expr1),
    format("L2 #76 (Gemm+Add+ReLU):~n"),
    format("  Epilogue expr: ~w~n~n", [Expr1]),

    %% L2 #59: Matmul + swish + scale
    chain_ops([swish, scale], c_var(val), Expr2),
    format("L2 #59 (Matmul+Swish+Scale):~n"),
    format("  Epilogue expr: ~w~n~n", [Expr2]),

    %% L2 #95: Matmul + bias_add + swish + tanh + gelu + hardtanh
    chain_ops([bias_add, swish, tanh, gelu, hardtanh], c_var(val), Expr3),
    format("L2 #95 (Matmul+5 ops):~n"),
    format("  Epilogue expr: ~w~n~n", [Expr3]),

    %% Now emit actual CUDA for L2 #76
    format("═══ GENERATED CUDA (L2 #76 epilogue) ═══~n~n"),
    chain_ops([bias_add, relu], c_var(val), CudaExpr),
    %% Use c_ast to emit
    format("  float val = c[r][cl];~n"),
    format("  val = ~w;  // ← the auto-fuser generates this~n", [CudaExpr]),
    format("  C[(row0+r)*N + col0+cl] = val;~n~n"),

    %% Show it would work with c_ast emit
    format("═══ WITH c_ast EMITTER ═══~n~n"),
    use_module('../lib/c_ast'),
    Stmts = [
        c_decl_init(c_type(float), val, c_index(c_var(c), c_var(acc_idx))),
        c_assign(c_var(val), CudaExpr),
        c_assign(c_index(c_var('C'), c_var(out_idx)), c_var(val))
    ],
    (emit_program(Stmts, Code) ->
        format("  ~w~n", [Code])
    ;
        format("  (c_ast emit pending — expression contains nested terms)~n")
    ).
