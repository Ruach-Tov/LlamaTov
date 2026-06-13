%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% op_signatures.pl — Canonical function signatures for BPD ops.
%%
%% The test harness queries these to determine calling conventions.
%% No more hardcoded conventions in Python. Prolog is the source of truth.
%%
%% Signature types:
%%   in       = float* input array (read-only)
%%   out      = float* output array (write)
%%   out_scalar = float* pointer to single output value
%%   n        = int32 array length
%%   scalar   = float scalar parameter
%%   eps      = float epsilon parameter
%%   groups   = int32 group count

:- module(op_signatures, [
    op_signature/2,    % op_signature(FnName, ArgList)
    op_pattern/2,      % op_pattern(FnName, PatternName)
    op_ggml_name/2     % op_ggml_name(FnName, GgmlOpName)
]).

:- discontiguous op_signature/2.
:- discontiguous op_pattern/2.
:- discontiguous op_ggml_name/2.

%% ============================================================
%% Unary elementwise: void fn(float* in, float* out, int n)
%% Convention: ion
%% ============================================================
op_signature(bpd_relu_cpu,        [in, out, n]).
op_signature(bpd_silu_cpu,        [in, out, n]).
op_signature(bpd_gelu_cpu,        [in, out, n]).
op_signature(bpd_elu_cpu,         [in, out, n]).
op_signature(bpd_selu_cpu,        [in, out, n]).
op_signature(bpd_leaky_relu_cpu,  [in, out, n]).
op_signature(bpd_sigmoid_cpu,     [in, out, n]).
op_signature(bpd_tanh_cpu,        [in, out, n]).
op_signature(bpd_hardsigmoid_cpu, [in, out, n]).
op_signature(bpd_softplus_cpu,    [in, out, n]).
op_signature(bpd_softsign_cpu,    [in, out, n]).
op_signature(bpd_clamp_cpu,       [in, out, n]).

%% LLVM-emitted unary: void fn(int n, float* out, float* in)
%% Convention: noi
op_signature(bpd_relu,        [n, out, in]).
op_signature(bpd_silu,        [n, out, in]).
op_signature(bpd_gelu,        [n, out, in]).
op_signature(bpd_gelu_fixed,        [n, out, in]).
op_signature(bpd_tanh_fixed,        [n, out, in]).
op_signature(bpd_hardsigmoid_fixed, [n, out, in]).
op_signature(bpd_sum_sse3,         [n, out_scalar, in]).
op_signature(bpd_scale,            [n, out, in, scalar]).
op_signature(bpd_cumsum,           [n, out, in]).
op_signature(bpd_cumprod,          [n, out, in]).
op_signature(bpd_clamp,            [n, out, in]).
op_signature(bpd_elu,              [n, out, in]).
op_signature(bpd_elu_fixed,        [n, out, in]).
op_signature(bpd_softplus_fixed,   [n, out, in]).
op_signature(bpd_leaky_relu,  [n, out, in]).
op_signature(bpd_sigmoid,     [n, out, in]).
op_signature(bpd_tanh,        [n, out, in]).
op_signature(bpd_hardsigmoid, [n, out, in]).
op_signature(bpd_softplus,    [n, out, in]).
op_signature(bpd_softsign,    [n, out, in]).

%% ============================================================
%% Binary elementwise: void fn(float* in, float* out, int n, float scalar)
%% Convention: ions
%% ============================================================
op_signature(bpd_scalar_mul_cpu, [in, out, n, scalar]).

%% ============================================================
%% Reduction (scalar output): void fn(float* in, float* out_scalar, int n)
%% Convention: ios (output is a single float*)
%% ============================================================
op_signature(bpd_sum_cpu,  [in, out_scalar, n]).
op_signature(bpd_mean_cpu, [in, out_scalar, n]).
op_signature(bpd_max_cpu,  [in, out_scalar, n]).
op_signature(bpd_min_cpu,  [in, out_scalar, n]).

%% ============================================================
%% Reduction-then-elementwise: void fn(float* in, float* out, int n)
%% Convention: ion (same as unary — array in, array out)
%% ============================================================
%% layernorm needs (in, gamma, beta, out, N, D, eps) — too complex, skip
%% rmsnorm needs (in, out, N, C, H, W, eps) — too complex, skip

%% These need extra params:
%% groupnorm needs (in, gamma, beta, out, N, C, H, W, G, eps) — too complex, skip

%% ============================================================
%% Scan: void fn(float* in, float* out, int n)
%% Convention: ion
%% ============================================================
op_signature(bpd_cumsum_cpu,  [in, out, n]).
op_signature(bpd_cumprod_cpu, [in, out, n]).

%% ============================================================
%% Pattern and ggml name mappings
%% ============================================================
%% LLVM-emitted ops
op_pattern(bpd_relu, unary_elementwise).
op_pattern(bpd_silu, unary_elementwise).
op_pattern(bpd_gelu, unary_elementwise).
op_pattern(bpd_gelu_fixed, unary_elementwise).
op_pattern(bpd_tanh_fixed, unary_elementwise).
op_pattern(bpd_hardsigmoid_fixed, unary_elementwise).
op_pattern(bpd_elu, unary_elementwise).
op_pattern(bpd_elu_fixed, unary_elementwise).
op_pattern(bpd_softplus_fixed, unary_elementwise).
op_pattern(bpd_leaky_relu, unary_elementwise).
op_pattern(bpd_sigmoid, unary_elementwise).
op_pattern(bpd_tanh, unary_elementwise).
op_pattern(bpd_hardsigmoid, unary_elementwise).
op_pattern(bpd_softplus, unary_elementwise).
op_pattern(bpd_softsign, unary_elementwise).

%% CPU reference ops
op_pattern(bpd_relu_cpu, unary_elementwise).
op_pattern(bpd_silu_cpu, unary_elementwise).
op_pattern(bpd_gelu_cpu, unary_elementwise).
op_pattern(bpd_elu_cpu, unary_elementwise).
op_pattern(bpd_selu_cpu, unary_elementwise).
op_pattern(bpd_leaky_relu_cpu, unary_elementwise).
op_pattern(bpd_sigmoid_cpu, unary_elementwise).
op_pattern(bpd_tanh_cpu, unary_elementwise).
op_pattern(bpd_hardsigmoid_cpu, unary_elementwise).
op_pattern(bpd_softplus_cpu, unary_elementwise).
op_pattern(bpd_softsign_cpu, unary_elementwise).
op_pattern(bpd_clamp_cpu, unary_elementwise).
op_pattern(bpd_scalar_mul_cpu, binary_elementwise).
op_pattern(bpd_sum_cpu, reduction).
op_pattern(bpd_mean_cpu, reduction).
op_pattern(bpd_max_cpu, reduction).
op_pattern(bpd_min_cpu, reduction).
op_pattern(bpd_layernorm_cpu, reduction_then_elementwise).
op_pattern(bpd_rmsnorm_cpu, reduction_then_elementwise).
op_pattern(bpd_softmax_cpu, reduction_then_elementwise).
op_pattern(bpd_logsoftmax_cpu, reduction_then_elementwise).
op_pattern(bpd_groupnorm_cpu, reduction_then_elementwise).
op_pattern(bpd_l2norm_cpu, reduction_then_elementwise).
op_pattern(bpd_cumsum_cpu, scan).
op_pattern(bpd_cumprod_cpu, scan).

%% LLVM-emitted ops
op_ggml_name(bpd_relu, ggml_relu).
op_ggml_name(bpd_silu, ggml_silu).
op_ggml_name(bpd_gelu, ggml_gelu).
op_ggml_name(bpd_gelu_fixed, ggml_gelu).
op_ggml_name(bpd_tanh_fixed, ggml_tanh).
op_ggml_name(bpd_hardsigmoid_fixed, ggml_hardsigmoid).
op_ggml_name(bpd_elu, ggml_elu).
op_ggml_name(bpd_elu_fixed, ggml_elu).
op_ggml_name(bpd_softplus_fixed, ggml_softplus).
op_ggml_name(bpd_leaky_relu, ggml_leaky_relu).
op_ggml_name(bpd_sigmoid, ggml_sigmoid).
op_ggml_name(bpd_tanh, ggml_tanh).
op_ggml_name(bpd_hardsigmoid, ggml_hardsigmoid).
op_ggml_name(bpd_softplus, ggml_softplus).
op_ggml_name(bpd_softsign, ggml_softsign).

%% CPU reference ops
op_ggml_name(bpd_relu_cpu, ggml_relu).
op_ggml_name(bpd_silu_cpu, ggml_silu).
op_ggml_name(bpd_gelu_cpu, ggml_gelu).
op_ggml_name(bpd_elu_cpu, ggml_elu).
op_ggml_name(bpd_selu_cpu, ggml_selu).
op_ggml_name(bpd_leaky_relu_cpu, ggml_leaky_relu).
op_ggml_name(bpd_sigmoid_cpu, ggml_sigmoid).
op_ggml_name(bpd_tanh_cpu, ggml_tanh).
op_ggml_name(bpd_hardsigmoid_cpu, ggml_hardsigmoid).
op_ggml_name(bpd_softplus_cpu, ggml_softplus).
op_ggml_name(bpd_softsign_cpu, ggml_softsign).
op_ggml_name(bpd_clamp_cpu, ggml_clamp).
op_ggml_name(bpd_scalar_mul_cpu, ggml_scale).
op_ggml_name(bpd_sum_cpu, ggml_sum).
op_ggml_name(bpd_mean_cpu, ggml_mean).
op_ggml_name(bpd_max_cpu, ggml_max).
op_ggml_name(bpd_min_cpu, ggml_min).
op_ggml_name(bpd_layernorm_cpu, ggml_norm).
op_ggml_name(bpd_rmsnorm_cpu, ggml_rms_norm).
op_ggml_name(bpd_softmax_cpu, ggml_soft_max).
op_ggml_name(bpd_logsoftmax_cpu, ggml_log_softmax).
op_ggml_name(bpd_groupnorm_cpu, ggml_group_norm).
op_ggml_name(bpd_l2norm_cpu, ggml_l2_norm).
op_ggml_name(bpd_cumsum_cpu, ggml_cumsum).
op_ggml_name(bpd_cumprod_cpu, ggml_cumprod).

%% ============================================================
%% Query helper: emit JSON for Python harness consumption
%% ============================================================
emit_test_manifest :-
    format('[~n'),
    forall(
        (op_signature(Fn, Sig), op_pattern(Fn, Pat), op_ggml_name(Fn, Ggml)),
        (
            term_to_atom(Sig, SigAtom),
            format('  {"fn": "~w", "pattern": "~w", "ggml": "~w", "sig": "~w"},~n',
                   [Fn, Pat, Ggml, SigAtom])
        )
    ),
    format(']~n').

main :- emit_test_manifest, halt.

%% SSE3-specific emitters
op_pattern(bpd_sum_sse3, reduction).
op_ggml_name(bpd_sum_sse3, ggml_sum).

%% LLVM-emitted: binary, scan, clamp
op_pattern(bpd_scale, binary_elementwise).
op_pattern(bpd_cumsum, scan).
op_pattern(bpd_cumprod, scan).
op_pattern(bpd_clamp, unary_elementwise).

op_ggml_name(bpd_scale, ggml_scale).
op_ggml_name(bpd_cumsum, ggml_cumsum).
op_ggml_name(bpd_cumprod, ggml_cumprod).
op_ggml_name(bpd_clamp, ggml_clamp).

%% Loss functions: (pred, target, out, n) — two-input convention
op_signature(bpd_mse_loss_cpu,             [in, in2, out, n]).
op_signature(bpd_cross_entropy_loss_cpu,   [in, in2, out, n]).
op_signature(bpd_hinge_loss_cpu,           [in, in2, out, n]).
op_signature(bpd_huber_loss_cpu,           [in, in2, out, n]).
op_signature(bpd_kl_div_loss_cpu,          [in, in2, out, n]).

op_pattern(bpd_mse_loss_cpu, loss_reduce).
op_pattern(bpd_cross_entropy_loss_cpu, loss_reduce).
op_pattern(bpd_hinge_loss_cpu, loss_reduce).
op_pattern(bpd_huber_loss_cpu, loss_reduce).
op_pattern(bpd_kl_div_loss_cpu, loss_reduce).

op_ggml_name(bpd_mse_loss_cpu, ggml_mse_loss).
op_ggml_name(bpd_cross_entropy_loss_cpu, ggml_cross_entropy_loss).
op_ggml_name(bpd_hinge_loss_cpu, ggml_hinge_loss).
op_ggml_name(bpd_huber_loss_cpu, ggml_huber_loss).
op_ggml_name(bpd_kl_div_loss_cpu, ggml_kl_div_loss).

%% Softmax/logsoftmax/l2norm: (in, out, 1, n) — treat as 1 row of n cols
op_signature(bpd_softmax_cpu,    [in, out, n_rows, n]).
op_signature(bpd_logsoftmax_cpu, [in, out, n_rows, n]).
op_signature(bpd_l2norm_cpu,     [in, out, n_rows, n]).

op_pattern(bpd_softmax_cpu, reduction_then_elementwise).
op_pattern(bpd_logsoftmax_cpu, reduction_then_elementwise).
op_pattern(bpd_l2norm_cpu, reduction_then_elementwise).

op_ggml_name(bpd_softmax_cpu, ggml_soft_max).
op_ggml_name(bpd_logsoftmax_cpu, ggml_log_softmax).
op_ggml_name(bpd_l2norm_cpu, ggml_l2_norm).

%% Ops with complex calling conventions — harness can't call yet
op_signature(bpd_layernorm_cpu, [unsupported]).
op_signature(bpd_rmsnorm_cpu, [unsupported]).
op_signature(bpd_groupnorm_cpu, [unsupported]).

%% ============================================================
%% Norm-op precision invariants (Iyun, 2026-05-29)
%% Grounding: ggml.c norm:3101, rms_norm:3132, group_norm:3183
%% ============================================================

%% Population variance (/n) — PyTorch ref needs unbiased=False
norm_variance_convention(layer_norm, population).
norm_variance_convention(group_norm, population).
norm_variance_convention(rms_norm,   population).

%% eps INSIDE sqrt, before (r)sqrt, every op. Default 1e-5.
norm_eps_placement(layer_norm, inside_sqrt).
norm_eps_placement(rms_norm,   inside_sqrt).
norm_eps_placement(l2_norm,    inside_sqrt).
norm_eps_placement(group_norm, inside_sqrt).
norm_eps_default(1.0e-5).

%% rsqrt (one rounding) NOT 1.0/sqrt (two) — ggml rsqrtf path.
norm_inv_sqrt_method(layer_norm, rsqrt).
norm_inv_sqrt_method(rms_norm,   rsqrt).
norm_inv_sqrt_method(l2_norm,    rsqrt).
norm_inv_sqrt_method(group_norm, rsqrt).

%% Reduction kind: rms=mean(x²), l2=sum(x²)
norm_reduction(rms_norm, mean_of_squares).
norm_reduction(l2_norm,  sum_of_squares).
norm_reduction(layer_norm, mean_centered_variance).
norm_reduction(group_norm, mean_centered_variance).

%% Affine (outside op): layer/group = weight+bias; rms/l2 = weight only
norm_affine(layer_norm, weight_and_bias).
norm_affine(rms_norm,   weight_only).
norm_affine(l2_norm,    weight_only).
norm_affine(group_norm, weight_and_bias).

%% ============================================================
%% RoPE (rotary position embedding) invariants — measured + GGUF-confirmed
%% from spec_dump_v2 model (Llama-3.2-1B-class, sha256-74701a8c...), 2026-05-30.
%% These are machine-checkable precision/config invariants for bit-exact rope.
%% ============================================================
%% freq_base (theta): Llama-3 uses 5e5 (NOT Mistral's 1e6). GGUF: llama.rope.freq_base.
rope_freq_base(llama3, 500000.0).
%% rope dimension = head_dim (full-dim rope, no partial). GGUF: llama.rope.dimension_count.
rope_dimension(llama3, 64).
%% rope mode: Llama-3 uses NEOX-style (split-half rotation), ggml GGML_ROPE_TYPE_NEOX.
rope_mode(llama3, neox).
%% freq_scale = 1.0 (no linear/yarn scaling for the base context).
rope_freq_scale(llama3, 1.0).
%% Attention config (GQA): n_head Q heads share n_head_kv KV heads (broadcast ratio n_head/n_head_kv).
attention_heads(llama3_1b, n_head(32), n_head_kv(8), head_dim(64)).
%% softmax scale for attention scores = 1/sqrt(head_dim).
attention_softmax_scale(head_dim(D), Scale) :- Scale is 1.0 / sqrt(D).
%% norm eps confirmed by bisection (eps=1e-6 fails, 1e-5 exact) on this model — validates norm_eps_default(1.0e-5).


%% ============================================================
%% LLVM-emitted loss functions: return float (scalar result)
%% Signature: float fn(int n, float* pred, float* target)
%% ============================================================
op_signature(bpd_mse_loss,             [n, in, in2]).
op_signature(bpd_cross_entropy_loss,   [n, in, in2]).
op_signature(bpd_hinge_loss,           [n, in, in2]).
op_signature(bpd_kl_div_loss,          [n, in, in2]).

op_pattern(bpd_mse_loss, loss_reduce).
op_pattern(bpd_cross_entropy_loss, loss_reduce).
op_pattern(bpd_hinge_loss, loss_reduce).
op_pattern(bpd_kl_div_loss, loss_reduce).

op_ggml_name(bpd_mse_loss, ggml_mse_loss).
op_ggml_name(bpd_cross_entropy_loss, ggml_cross_entropy_loss).
op_ggml_name(bpd_hinge_loss, ggml_hinge_loss).
op_ggml_name(bpd_kl_div_loss, ggml_kl_div_loss).

%% LLVM-emitted selu (calls expf directly)
op_signature(bpd_selu_fixed, [n, out, in]).
op_pattern(bpd_selu_fixed, unary_elementwise).
op_ggml_name(bpd_selu_fixed, ggml_selu).
