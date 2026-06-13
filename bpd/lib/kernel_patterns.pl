%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% kernel_patterns.pl — Categorize L1 ops into LLVM IR emission patterns.
%%
%% Each op maps to a pattern. Each pattern has one LLVM IR emitter.
%% 44 ops collapse to 6 patterns.
%%
%% The Prolog→LLVM emitter dispatches on pattern, not on op.
%% Same canonical form, different ops, same IR shape.

:- module(kernel_patterns, [
    kernel_pattern/2,     % kernel_pattern(Op, Pattern)
    pattern_desc/2,       % pattern_desc(Pattern, Description)
    pattern_ir_shape/2    % pattern_ir_shape(Pattern, IRShape)
]).


%% clauses grouped by family, not contiguous — declared for warning-free consult.
:- discontiguous kernel_pattern/2, pattern_desc/2, pattern_ir_shape/2.
%% ============================================================
%% Pattern 1: unary_elementwise
%% IR shape: loop over <4 x float>, apply f(x) → y
%% One input array, one output array.
%% ============================================================
pattern_desc(unary_elementwise,
    'Loop over <4 x float> vectors. Apply one function per element.').
pattern_ir_shape(unary_elementwise,
    'entry → loop(load x, f(x), store y) → done').

kernel_pattern(ggml_relu,        unary_elementwise).  % max(0, x)
kernel_pattern(ggml_silu,        unary_elementwise).  % x * sigmoid(x)
kernel_pattern(ggml_gelu,        unary_elementwise).  % x * Phi(x) approx
kernel_pattern(ggml_elu,         unary_elementwise).  % x>0 ? x : alpha*(exp(x)-1)
kernel_pattern(ggml_selu,        unary_elementwise).  % lambda * elu(x, alpha)
kernel_pattern(ggml_leaky_relu,  unary_elementwise).  % x>0 ? x : alpha*x
kernel_pattern(ggml_sigmoid,     unary_elementwise).  % 1/(1+exp(-x))
kernel_pattern(ggml_tanh,        unary_elementwise).  % tanh(x)
kernel_pattern(ggml_hardsigmoid, unary_elementwise).  % clamp(x/6 + 0.5, 0, 1)
kernel_pattern(ggml_softplus,    unary_elementwise).  % log(1+exp(x))
kernel_pattern(ggml_softsign,    unary_elementwise).  % x/(1+|x|)
kernel_pattern(ggml_clamp,       unary_elementwise).  % clamp(x, min, max)

%% ============================================================
%% Pattern 2: binary_elementwise
%% IR shape: loop over <4 x float>, apply f(x, y) → z
%% Two input arrays, one output array.
%% ============================================================
pattern_desc(binary_elementwise,
    'Loop over <4 x float> vectors. Apply binary op per element.').
pattern_ir_shape(binary_elementwise,
    'entry → loop(load x, load y, f(x,y), store z) → done').

kernel_pattern(ggml_scale,  binary_elementwise).  % x * scalar (broadcast)

%% ============================================================
%% Pattern 3: reduction
%% IR shape: accumulate <4 x float> across input, reduce to scalar.
%% Uses ARR accumulators + binary tree + hadd.
%% This is vec_dot — ALREADY 0 ULP.
%% ============================================================
pattern_desc(reduction,
    'Accumulate <4 x float> across input. Binary tree + hadd reduce.').
pattern_ir_shape(reduction,
    'entry → loop(load x, [load y], acc += x*y) → reduce(tree+hadd) → done').

kernel_pattern(ggml_mul_mat,   reduction).  % dot product inner loop
kernel_pattern(ggml_sum_rows,  reduction).  % sum elements (dot with ones)
kernel_pattern(ggml_mean,      reduction).  % sum / n
kernel_pattern(ggml_max,       reduction).  % max-reduce (use fmax instead of fadd)
kernel_pattern(ggml_min,       reduction).  % min-reduce (use fmin instead of fadd)
kernel_pattern(ggml_argmax,    reduction).  % max-reduce tracking index
kernel_pattern(ggml_argmin,    reduction).  % min-reduce tracking index

%% ============================================================
%% Pattern 4: reduction_then_elementwise
%% IR shape: first reduce (mean, variance), then elementwise normalize.
%% Two-pass: pass 1 accumulates, pass 2 applies.
%% ============================================================
pattern_desc(reduction_then_elementwise,
    'Two-pass: reduce (mean/var), then elementwise normalize.').
pattern_ir_shape(reduction_then_elementwise,
    'pass1: reduce(mean,var) → pass2: loop((x-mean)/sqrt(var+eps)) → done').

kernel_pattern(ggml_norm,       reduction_then_elementwise).  % layernorm
kernel_pattern(ggml_rms_norm,   reduction_then_elementwise).  % rms normalization
kernel_pattern(ggml_group_norm, reduction_then_elementwise).  % group normalization
kernel_pattern(ggml_l2_norm,    reduction_then_elementwise).  % L2 normalization
kernel_pattern(ggml_soft_max_ext, reduction_then_elementwise). % softmax: max-reduce, exp, sum, div
kernel_pattern(ggml_log_soft_max, reduction_then_elementwise). % log(softmax(x))

%% ============================================================
%% Pattern 5: scan (prefix/cumulative)
%% IR shape: sequential scan maintaining running accumulator.
%% Cannot be parallelized per-element.
%% ============================================================
pattern_desc(scan,
    'Sequential scan: running accumulator across elements.').
pattern_ir_shape(scan,
    'entry → loop(acc = f(acc, x[i]), out[i] = acc) → done').

kernel_pattern(ggml_cumsum,  scan).  % cumulative sum
kernel_pattern(ggml_cumprod, scan).  % cumulative product

%% ============================================================
%% Pattern 6: conv_im2col
%% IR shape: im2col rearrangement + reduction (mul_mat).
%% The reduction part reuses Pattern 3.
%% ============================================================
pattern_desc(conv_im2col,
    'im2col rearrangement + mul_mat reduction. Reuses Pattern 3.').
pattern_ir_shape(conv_im2col,
    'im2col(input, kernel) → mul_mat(col, weight) → [add bias]').

kernel_pattern(ggml_conv_1d,            conv_im2col).
kernel_pattern(ggml_conv_2d,            conv_im2col).
kernel_pattern(ggml_conv_3d,            conv_im2col).
kernel_pattern(ggml_conv_transpose_1d,  conv_im2col).
kernel_pattern(ggml_conv_transpose_2d,  conv_im2col).
kernel_pattern(ggml_conv_transpose_3d,  conv_im2col).

%% ============================================================
%% Pattern 7: pool_reduce
%% IR shape: sliding window reduction (max or avg over window).
%% ============================================================
pattern_desc(pool_reduce,
    'Sliding window reduction: max/avg over kernel-sized window.').
pattern_ir_shape(pool_reduce,
    'for each output position: reduce(window) → store').

kernel_pattern(ggml_pool_1d, pool_reduce).
kernel_pattern(ggml_pool_2d, pool_reduce).
kernel_pattern(ggml_pool_3d, pool_reduce).

%% ============================================================
%% Pattern 8: loss_reduce
%% IR shape: elementwise transform + reduction to scalar.
%% Similar to reduction_then_elementwise but inverse order.
%% ============================================================
pattern_desc(loss_reduce,
    'Elementwise transform then reduce to scalar loss.').
pattern_ir_shape(loss_reduce,
    'loop(transform(pred, target)) → reduce(sum/mean)').

kernel_pattern(ggml_mse_loss,             loss_reduce).
kernel_pattern(ggml_cross_entropy_loss,   loss_reduce).
kernel_pattern(ggml_hinge_loss,           loss_reduce).
kernel_pattern(ggml_huber_loss,           loss_reduce).
kernel_pattern(ggml_kl_div_loss,          loss_reduce).
kernel_pattern(ggml_triplet_margin_loss,  loss_reduce).

%% ============================================================
%% Pattern 9: flash_attention (special)
%% ============================================================
pattern_desc(flash_attention,
    'Tiled QKV attention with online softmax. Special pattern.').
pattern_ir_shape(flash_attention,
    'tile_loop(Q_tile @ K_tile^T → softmax → @ V_tile)').

kernel_pattern(ggml_flash_attn_ext, flash_attention).

%% ============================================================
%% Summary queries
%% ============================================================

%% Count ops per pattern
pattern_count(Pattern, Count) :-
    findall(Op, kernel_pattern(Op, Pattern), Ops),
    length(Ops, Count).

%% List all patterns with counts
list_patterns :-
    findall(P, pattern_desc(P, _), Ps),
    sort(Ps, Unique),
    format("~n~w~t~30|~w~t~40|~w~n", ['Pattern', 'Ops', 'IR Emitters Needed']),
    format("~`-t~50|~n", []),
    forall(member(P, Unique), (
        pattern_count(P, C),
        format("~w~t~30|~w~t~40|1~n", [P, C])
    )),
    findall(_, kernel_pattern(_, _), All),
    length(All, Total),
    length(Unique, NP),
    format("~`-t~50|~n"),
    format("Total: ~w ops → ~w patterns → ~w IR emitters~n", [Total, NP, NP]).
