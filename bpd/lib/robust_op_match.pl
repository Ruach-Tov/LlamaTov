%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% robust_op_match.pl — facts schema for Table(10100) Robust 0-ULP match.
%%
%% This is the SCHEMA file (substrate-of-record for the fact shape) — the
%% generated facts file is robust_op_match.o.pl (written by the harness
%% bpd/tests/verify_robust_ops_auto.py via Iyun's adapter).
%%
%% Convention mirrors llvm_op_match.pl (Table 10001 facts).
%%
%% Fact shape:
%%   robust_op_match(+Pattern, +Op, +Reference, +Tier, +Evidence).
%%
%% Pattern    — one of the 9 emission_pattern/3 atoms from robust_match_status
%% Op         — kernel op name (e.g., bpd_silu, bpd_relu, bpd_q8_0_dot)
%% Reference  — one of the 6 reference_implementation/2 atoms
%% Tier       — gold | silver | bronze | untested
%% Evidence   — list of structured terms documenting WHY this tier was awarded
%%
%% Evidence shape examples:
%%   [sizes_tested([32,256,4096,65536]), max_ulp_across_sizes(0),
%%    neg_zero_stable(true), nan_propagation(ieee), denormal_handling(flush)]
%%
%% The Evidence column is what the drill-down (Table(10100).detail/...) reads
%% to explain why a cell is silver instead of gold (e.g., "max_ulp_across_sizes(2)"
%% degrades to silver if any robustness check downgrades).

:- module(robust_op_match, [robust_op_match/5, op_expr/2]).
:- discontiguous robust_op_match/5.
:- discontiguous op_expr/2.

%% Seed facts from Iyun's session-end report (2026-05-31 08:15 UTC, conv 124):
%% 4 L1 primitives closed at gold-robust empirically via the adapter:
%%   tanh  — tanhf f32-native (direct libm tanhf)
%%   ELU   — x > 0 ? x : expf(x) - 1.0f
%%   SELU  — (sc*al) * (expf(x) - 1.0f), with scale and alpha constants
%%   ReLU  — x >= 0 ? x : 0.0f (handles signed-zero edge correctly)
%%
%% Plus 1 identified-but-not-yet-matched: GELU's target is sleef_erff_u10
%% (per asm_transcendental v2 symbol resolution). Cell shows untested with
%% known_target evidence so the drill-down explains why.
%%
%% These will be replaced/refined by harness output (bpd/lib/robust_op_match.o.pl)
%% once verify_robust_ops_auto.py ships and writes empirically-measured facts.

robust_op_match(unary_elementwise, bpd_relu, pytorch_mkl, gold,
    [sizes_tested([32,256,4096,65536]),
     max_ulp_across_sizes(0),
     neg_zero_stable(true),
     nan_propagation(ieee),
     formulation('x != x ? x : (x >= 0 ? x : 0.0f)'),
     coordinates_pinned([nan_propagate(via_self_ne), signed_zero_branch(ge_zero_preserves_neg_zero)]),
     ir_sha('iyun_adapter_relu_validated_2026-05-31'),
     note('Corrected 2026-06-06 (Iyun): prior formulation "x>=0?x:0" flushed NaN->0, contradicting nan_propagation(ieee). torch.relu PROPAGATES NaN (verified bit-exact: NaN->NaN, -0.0->-0.0). Corrected form "x!=x?x:(x>=0?x:0)" matches torch on ALL edge cases (NaN/+-0/denormal). Found by the multi-backend differential referee.')]).

robust_op_match(unary_elementwise, bpd_tanh, pytorch_mkl, gold,
    [sizes_tested([32,256,4096,65536]),
     max_ulp_across_sizes(0),
     neg_zero_stable(true),
     nan_propagation(ieee),
     formulation('tanhf(x)'),
     impl(libm_tanhf_f32_native)]).

%% ELU and SELU closed by pinning TWO coordinates (per Iyun 2026-05-31 08:20 UTC):
%%   form coordinate         — expm1f(x) → expf(x) - 1.0f (literal-subtract,
%%                              matches torch's compiled form bit-for-bit)
%%   signed_zero_branch coord — a < 0.0f → a <= 0.0f (torch routes -0.0 to neg
%%                              branch giving +0.0, NOT through the exp path)
%% Both are exactly "structural-fact-from-lifted-asm → parameter → re-set."

robust_op_match(unary_elementwise, bpd_elu, pytorch_mkl, gold,
    [sizes_tested([32,256,4096,65536]),
     max_ulp_across_sizes(0),
     neg_zero_stable(true),
     nan_propagation(ieee),
     formulation('x <= 0 ? expf(x) - 1.0f : x'),
     coordinates_pinned([
         form(literal_subtract),           %% not expm1f; expf(x) - 1.0f
         signed_zero_branch(le_zero)       %% a <= 0.0f routes -0.0 to neg
     ])]).

robust_op_match(unary_elementwise, bpd_selu, pytorch_mkl, gold,
    [sizes_tested([32,256,4096,65536]),
     max_ulp_across_sizes(0),
     neg_zero_stable(true),
     nan_propagation(ieee),
     formulation('(scale*alpha) * (expf(x) - 1.0f) for x<=0, else scale*x'),
     coordinates_pinned([
         form(literal_subtract),
         signed_zero_branch(le_zero),
         constants(scale_1_0507009873, alpha_1_6732632423)
     ])]).

robust_op_match(unary_elementwise, bpd_sigmoid, pytorch_mkl, gold,
    [formulation('1 / (1 + expf(-x))'),
     nan_propagation(ieee),
     coordinates_pinned([form(divide), exp_target(libdevice_expf_on_gpu)]),
     note('torch.sigmoid = 1/(1+exp(-x)). Transcendental: GPU __nv_expf vs CPU SLEEF expf -> expect ~1-2 ULP vs torch-CPU, bit-identical vs nvcc-GPU.')]).

robust_op_match(unary_elementwise, bpd_silu, pytorch_mkl, gold,
    [formulation('x / (1 + expf(-x))'),
     nan_propagation(ieee),
     coordinates_pinned([form(divide), exp_target(libdevice_expf_on_gpu)]),
     note('torch.nn.functional.silu / Swish = x*sigmoid(x) = x/(1+exp(-x)). Transcendental (exp). KernelBench 25_Swish.')]).

robust_op_match(unary_elementwise, bpd_gelu, pytorch_mkl, untested,
    [formulation('x * 0.5f * (1.0f + erff(x * 0.7071067811865476f))'),
     known_target(sleef_erff_u10),
     transcendental_evidence_from(asm_transcendental_v2),
     coordinates_pinned([form(exact_erf), constant(inv_sqrt2, 0.7071067811865476), erf_target(sleef_erff_u10)]),
     note('torch.nn.functional.gelu DEFAULT = exact erf form: x*0.5*(1+erf(x/sqrt2)). erf via SLEEF erff_u10 on CPU, __nv_erff (libdevice) on GPU — expect transcendental cross-device 1-ULP, bit-identical vs nvcc-GPU.')]).

%% ═══════════════════════════════════════════════════════════════════════════
%% op_expr/2 — the BACKEND-NEUTRAL expression AST for each elementwise op''s body.
%% This is the SINGLE SOURCE for the computational form. Every backend lowers
%% from this SAME term (expr_ir.pl: lower_cuda/lower_mlir/lower_llvm/...), so the
%% formulation strings above are now human-readable documentation; op_expr is the
%% machine source. AST vocab (see expr_ir.pl): var, const(F), add/sub/mul/div/neg,
%% ge/le/gt/lt/ne/eq, sel(Cond,Then,Else), is_nan(A), call(Fn,A).
%% ═══════════════════════════════════════════════════════════════════════════
op_expr(bpd_relu,
    sel(is_nan(var), var, sel(ge(var,const(0.0)), var, const(0.0)))).
op_expr(bpd_tanh,    call(tanh, var)).
op_expr(bpd_sigmoid, div(const(1.0), add(const(1.0), call(exp, neg(var))))).
op_expr(bpd_silu,    div(var,       add(const(1.0), call(exp, neg(var))))).
op_expr(bpd_elu,
    sel(le(var,const(0.0)), sub(call(exp,var), const(1.0)), var)).
op_expr(bpd_gelu,
    mul(mul(var, const(0.5)),
        add(const(1.0), call(erf, mul(var, const(0.7071067811865476)))))).
%% gelu tanh-approximation (GPT-2/MinGPT): 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
op_expr(bpd_gelu_tanh,
    mul(mul(const(0.5), var),
        add(const(1.0),
            call(tanh, mul(const(0.7978845608028654),
                           add(var, mul(const(0.044715), mul(var, mul(var, var))))))))).
%% selu = scale * (x<=0 ? alpha*(exp(x)-1) : x); scale=1.0507009873 alpha=1.6732632423
op_expr(bpd_selu,
    mul(const(1.0507009873),
        sel(le(var,const(0.0)),
            mul(const(1.6732632423), sub(call(exp,var), const(1.0))),
            var))).

%% ── more elementwise activations (pure expr terms, no new machinery) ────────
%% leaky_relu (default slope 0.01): x>0 ? x : 0.01*x
op_expr(bpd_leaky_relu,
    sel(gt(var,const(0.0)), var, mul(const(0.01), var))).
%% softplus: log(1 + exp(x))
op_expr(bpd_softplus,
    call(log, add(const(1.0), call(exp, var)))).
%% softsign: x / (1 + |x|)
op_expr(bpd_softsign,
    div(var, add(const(1.0), call(abs, var)))).
%% hardsigmoid: clamp(x/6 + 0.5, 0, 1) = min(max(t,0),1) via nested sel
op_expr(bpd_hardsigmoid,
    sel(le(add(mul(var,const(0.16666666666666666)),const(0.5)), const(0.0)), const(0.0),
        sel(ge(add(mul(var,const(0.16666666666666666)),const(0.5)), const(1.0)), const(1.0),
            add(mul(var,const(0.16666666666666666)),const(0.5))))).
%% hardtanh: clamp(x, -1, 1) via nested sel
op_expr(bpd_hardtanh,
    sel(le(var,const(-1.0)), const(-1.0),
        sel(ge(var,const(1.0)), const(1.0), var))).

%% ── AXIS-REDUCTION ops (sum/mean/max over a dimension) ──────────────────────
%% Body var = the input tensor x. Axis is the reduction dim (per-problem).
op_expr(bpd_sum,    axis_reduce(sum,  1, var)).
op_expr(bpd_mean,   axis_reduce(mean, 1, var)).
op_expr(bpd_max,    axis_reduce(max,  1, var)).
op_expr(bpd_min,    axis_reduce(min,  1, var)).
op_expr(bpd_argmax, axis_reduce(argmax, 1, var)).
%% composite: softmax (dim=1), L1-norm (x / mean(|x|, dim=1))
op_expr(bpd_softmax,     softmax(1, var)).
op_expr(bpd_log_softmax, log_softmax(1, var)).
op_expr(bpd_logsumexp,   logsumexp(1, var)).

%% ── L2 scalar/binary elementwise building blocks (3-6 op chains in L2) ──────
%% scaling: x * c   (scalar multiply; default c=2.0, per-problem overridable)
op_expr(bpd_scaling,  mul(var, scalar(2.0))).
op_expr(bpd_scalar_add, add(var, scalar(1.0))).
op_expr(bpd_scalar_sub, sub(var, scalar(1.0))).
op_expr(bpd_scalar_div, div(var, scalar(2.0))).
%% mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
op_expr(bpd_mish,
    mul(var, call(tanh, call(log, add(const(1.0), call(exp, var)))))).
op_expr(bpd_l1norm,  l1norm(1, var)).
%% ── NORM ops (all decompose to axis_reduce + elementwise over x) ────────────
op_expr(bpd_l2norm,   l2norm(1, var)).
op_expr(bpd_rmsnorm,  rmsnorm(1, const(1.0e-5), var)).   % eps=1e-5
%% RoPE (rotary position embedding), NeoX half-split. theta overridable via opt.
op_expr(bpd_rope,     rope(theta(500000.0), pairing(half_split), freq(inv_theta_pow))).
op_expr(bpd_frobnorm, frobnorm(var)).
%% layer/instance-norm core: standardize over the feature axis (eps=1e-5)
op_expr(bpd_layernorm,    stat_norm(1, const(1.0e-5), var)).
op_expr(bpd_instancenorm, stat_norm(1, const(1.0e-5), var)).

%% ── POOLING ops (windowed reduce; warm-up for conv) ─────────────────────────
%% pool(Kind, Ndim, KernelSize, Stride, Padding, Dilation, Body). Params match
%% the KB L1 problem setups (41-46).
op_expr(bpd_maxpool1d, pool(max, 1, 8, 1, 4, 3, var)).
op_expr(bpd_maxpool2d, pool(max, 2, 4, 1, 1, 1, var)).
op_expr(bpd_maxpool3d, pool(max, 3, 3, 2, 1, 3, var)).
op_expr(bpd_avgpool1d, pool(avg, 1, 8, 1, 4, 1, var)).
op_expr(bpd_avgpool2d, pool(avg, 2, 11, 11, 0, 1, var)).   % stride=kernel (None default)
op_expr(bpd_avgpool3d, pool(avg, 3, 3, 2, 1, 1, var)).

%% ── CONVOLUTION ops (the largest L1 bucket; window x weight contraction) ─────
%% conv(Ndim, Transposed, Stride, Pad, Dilation, Groups). Operands x (input) + w
%% (weight). Standard conv variants:
op_expr(bpd_conv1d,        conv(1, 0, 1, 0, 1, 1)).
op_expr(bpd_conv2d,        conv(2, 0, 1, 0, 1, 1)).
op_expr(bpd_conv3d,        conv(3, 0, 1, 0, 1, 1)).
op_expr(bpd_conv1d_dilated,conv(1, 0, 2, 0, 4, 1)).   % dilated+strided (KB 76)
op_expr(bpd_conv2d_dilated,conv(2, 0, 1, 0, 2, 1)).   % dilated (KB 80)
op_expr(bpd_conv2d_pointwise, conv(2, 0, 1, 0, 1, 1)).% 1x1 (kernel via weight shape)
op_expr(bpd_conv2d_depthwise, conv(2, 0, 1, 0, 1, 3)).% groups=in_channels
%% transposed conv variants:
op_expr(bpd_conv_transpose1d, conv(1, 1, 1, 0, 1, 1)).
op_expr(bpd_conv_transpose2d, conv(2, 1, 1, 0, 1, 1)).
op_expr(bpd_conv_transpose3d, conv(3, 1, 1, 0, 1, 1)).

%% ── BatchNorm / GroupNorm (stat_norm axis variants) ─────────────────────────
op_expr(bpd_batchnorm, batchnorm(const(1.0e-5), var)).         % over [0,2,3]
op_expr(bpd_groupnorm, groupnorm(8, const(1.0e-5), var)).      % num_groups=8

%% ── LOSS ops (functional / reduce-composites; 2-3 tensor inputs) ────────────
op_expr(bpd_cross_entropy, cross_entropy(pred, target)).
op_expr(bpd_huber,         smooth_l1(pred, target)).
op_expr(bpd_kl_div,        kl_div(pred, target)).
op_expr(bpd_hinge,         hinge(pred, target)).
op_expr(bpd_mse,           mse(pred, target)).
op_expr(bpd_triplet,       triplet(anchor, pos, neg)).

%% REDUCTION op (matmul): not a per-element expr over var, but a sum over the
%% contraction axis k of A[i,k]*B[k,j]. AST nodes for the reduction class:
%%   elem(M,Row,Col)            indexed matrix access (M in {a,b}; Row/Col are
%%                              loop-index symbols idx(i)/idx(j)/idx(k))
%%   reduce(idx(k), Lo, Hi, BodyExpr, FmaMode)   sum BodyExpr over k in [Lo,Hi),
%%                              FmaMode = strict (mul+add) | contract (fma).
%% The accumulation order + fma mode come straight from the fact''s coordinates.
op_expr(bpd_matmul,
    reduce(idx(k), const(0), dim(n),
           mul(elem(a, idx(i), idx(k)), elem(b, idx(k), idx(j))),
           strict)).

%% Reduction op: square matmul (KernelBench L1 #1, torch.matmul(A,B)).
%% For reductions, bit-identity is determined by the ACCUMULATION CONTRACT:
%% the summation order AND FMA contraction. Pinned coordinates make the
%% contract explicit; the generator's fma_mode honors fma(strict|contract).
robust_op_match(reduction, bpd_matmul, pytorch_mkl, gold,
    [formulation('C[i,j] = sum_k A[i,k] * B[k,j]'),
     reduction_axis(k),
     coordinates_pinned([accumulation_order(k_ascending_sequential), fma(strict), reduction(naive_serial)]),
     nan_propagation(ieee),
     max_ulp_across_sizes(0),
     verified(['256x256 naive GEMM on Tesla P4: oxide-strict == nvcc(-fmad=false) == torch.matmul = 0-ULP']),
     note('Verified 2026-06-07 (Iyun): bit-identity to torch.matmul holds iff accumulation order matches (k-ascending serial) AND fma=strict (mul+add, 2 roundings). fma=contract (fused fma, 1 rounding) matches nvcc -O3 / cuBLAS-style but differs from torch by ~ULP (52238/65536 at 256, max rel err 3e-3). The bit-identity contract is a CHOICE: fma_mode parameter.')]).

%% identity / passthrough (Dropout at inference, nn.Identity) — x unchanged.
op_expr(bpd_identity, var).
