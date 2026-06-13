%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ggml_dispatch_table.pl — Maps BPD op_kind to ggml C function calls
%%
%% Usage:
%%   ?- ggml_dispatch(conv2d, FuncName, ArgPattern).
%%   FuncName = 'ggml_conv_2d',
%%   ArgPattern = [ctx, kernel, input, s0, s1, p0, p1, d0, d1].

:- module(ggml_dispatch_table, [
    ggml_dispatch/3,
    ggml_has_op/1
]).

%% ggml_dispatch(+BpdOpKind, -GgmlFuncName, -ArgPattern)
%% Maps BPD op_kind atoms to ggml function names and argument patterns.
%% ArgPattern describes what arguments the ggml function expects.

%% Elementwise activations: ggml_*(ctx, tensor)
ggml_dispatch(relu,        'ggml_relu',         [ctx, a]).
ggml_dispatch(leaky_relu,  'ggml_leaky_relu',   [ctx, a]).
ggml_dispatch(silu,        'ggml_silu',         [ctx, a]).
ggml_dispatch(gelu,        'ggml_gelu',         [ctx, a]).
ggml_dispatch(sigmoid,     'ggml_sigmoid',      [ctx, a]).
ggml_dispatch(tanh,        'ggml_tanh',         [ctx, a]).
ggml_dispatch(elu,         'ggml_elu',          [ctx, a]).
ggml_dispatch(hardswish,   'ggml_hardswish',    [ctx, a]).
ggml_dispatch(hardsigmoid, 'ggml_hardsigmoid',  [ctx, a]).

%% Elementwise binary: ggml_*(ctx, a, b)
ggml_dispatch(add,         'ggml_add',          [ctx, a, b]).
ggml_dispatch(sub,         'ggml_sub',          [ctx, a, b]).
ggml_dispatch(mul,         'ggml_mul',          [ctx, a, b]).
ggml_dispatch(div,         'ggml_div',          [ctx, a, b]).

%% Elementwise unary math
ggml_dispatch(abs,         'ggml_abs',          [ctx, a]).
ggml_dispatch(neg,         'ggml_neg',          [ctx, a]).
ggml_dispatch(sqrt,        'ggml_sqrt',         [ctx, a]).
ggml_dispatch(sqr,         'ggml_sqr',          [ctx, a]).

%% Matmul
ggml_dispatch(matmul,      'ggml_mul_mat',      [ctx, a, b]).
ggml_dispatch(linear,      'ggml_mul_mat',      [ctx, weight, input]).

%% Conv2d: ggml_conv_2d(ctx, kernel, data, s0, s1, p0, p1, d0, d1)
ggml_dispatch(conv2d,      'ggml_conv_2d',      [ctx, kernel, input, s0, s1, p0, p1, d0, d1]).

%% Normalization: ggml_norm(ctx, tensor, eps)
ggml_dispatch(batchnorm,   'ggml_norm',         [ctx, a, eps]).
ggml_dispatch(layernorm,   'ggml_norm',         [ctx, a, eps]).
ggml_dispatch(rmsnorm,     'ggml_rms_norm',     [ctx, a, eps]).
ggml_dispatch(groupnorm,   'ggml_group_norm',   [ctx, a, n_groups]).
ggml_dispatch(l2norm,      'ggml_l2_norm',      [ctx, a, eps]).

%% Pooling: ggml_pool_2d(ctx, a, op, k0, k1, s0, s1, p0, p1)
ggml_dispatch(maxpool,     'ggml_pool_2d',      [ctx, a, 'GGML_OP_POOL_MAX', k0, k1, s0, s1, p0, p1]).
ggml_dispatch(avgpool,     'ggml_pool_2d',      [ctx, a, 'GGML_OP_POOL_AVG', k0, k1, s0, s1, p0, p1]).

%% Reductions
ggml_dispatch(sum_reduce,  'ggml_sum',          [ctx, a]).
ggml_dispatch(mean_reduce, 'ggml_mean',         [ctx, a]).
ggml_dispatch(argmax,      'ggml_argmax',       [ctx, a]).

%% Structural ops
ggml_dispatch(concat,      'ggml_concat',       [ctx, a, b, dim]).
ggml_dispatch(upsample,    'ggml_upscale',      [ctx, a, scale_factor, mode]).
ggml_dispatch(reshape,     'ggml_reshape_4d',   [ctx, a, ne0, ne1, ne2, ne3]).

%% Softmax
ggml_dispatch(softmax,     'ggml_soft_max',     [ctx, a]).

%% Check if we have a ggml op for a given BPD kind
ggml_has_op(Kind) :- ggml_dispatch(Kind, _, _).

%% Ops that need BPD kernel fallback (no ggml equivalent)
%% ggml_dispatch(mish, ...) — NOT in ggml, use BPD
%% ggml_dispatch(selu, ...) — NOT in ggml, use BPD
%% ggml_dispatch(softsign, ...) — NOT in ggml, use BPD
%% ggml_dispatch(softplus, ...) — NOT in ggml, use BPD
%% ggml_dispatch(hardtanh, ...) — NOT in ggml, use BPD
