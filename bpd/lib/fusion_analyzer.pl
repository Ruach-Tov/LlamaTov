%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% fusion_analyzer.pl — Prolog kernel fusion analyzer for BPD compute graphs.
%%
%% Discovers fusible operation chains in compute-graph BPD facts.
%% The analyzer queries the graph structure; it does NOT contain
%% target-language code. Fusion opportunities are FACTS that a
%% separate kernel generator consumes.
%%
%% Based on KernelBench L2 fusion patterns:
%%   - Epilogue fusion: matmul/conv + elementwise chain
%%   - Residual fusion: save input, compute, add back
%%   - Norm+activation fusion: layernorm + activation in one pass
%%
%% Part of the BPD ecosystem.

:- module(fusion_analyzer, [
    find_fusible_chains/2,     % find_fusible_chains(+GraphFacts, -Chains)
    classify_op/2,             % classify_op(+OpKind, -Class)
    can_fuse/3,                % can_fuse(+OpA, +OpB, -Reason)
    cannot_fuse/3              % cannot_fuse(+OpA, +OpB, -Reason)
]).

%% can_fuse/3 clauses are grouped by fusion KIND (epilogue, norm_activation, ...) and interleaved
%% with their cannot_fuse counterparts for readability — declare discontiguous (canary audibility).
:- discontiguous can_fuse/3, cannot_fuse/3.

%% General op_expr classifier — used to classify bpd_* ops (the general facts)
%% in addition to the ggml table below. (Port off ggml: op_expr is the source.)
:- use_module('op_classify', []).

%% ═══════════════════════════════════════════════════════════════
%% OPERATION CLASSIFICATION
%% ═══════════════════════════════════════════════════════════════
%%
%% Each operation is classified by its memory access pattern:
%%   elementwise — reads/writes each element independently
%%   reduction   — reads many elements, writes fewer (softmax, mean, sum)
%%   spatial     — reads a neighborhood (conv, pool)
%%   layout      — changes tensor shape without computing (reshape, view, permute)
%%   matmul      — matrix multiplication (producer of large intermediates)
%%   normalization — needs full-row/channel statistics (layernorm, rmsnorm)

classify_op(ggml_add, elementwise).
classify_op(ggml_mul, elementwise).
classify_op(ggml_sub, elementwise).
classify_op(ggml_div, elementwise).
classify_op(ggml_scale, elementwise).
classify_op(ggml_scale_bias, elementwise).
classify_op(ggml_silu, elementwise).
classify_op(ggml_gelu, elementwise).
classify_op(ggml_relu, elementwise).
classify_op(ggml_sigmoid, elementwise).
classify_op(ggml_tanh, elementwise).
classify_op(ggml_sqr, elementwise).
classify_op(ggml_sqrt, elementwise).
classify_op(ggml_exp, elementwise).
classify_op(ggml_clamp, elementwise).
classify_op(ggml_softplus, elementwise).
classify_op(ggml_xielu, elementwise).
classify_op(ggml_swiglu_split, elementwise).

classify_op(ggml_soft_max_ext, reduction).
classify_op(ggml_sum_rows, reduction).
classify_op(ggml_mean, reduction).
classify_op(ggml_cumsum, reduction).

classify_op(ggml_rms_norm, normalization).
classify_op(ggml_norm, normalization).
classify_op(ggml_l2_norm, normalization).

classify_op(ggml_mul_mat, matmul).
classify_op(build_lora_mm, matmul).

classify_op(ggml_reshape_1d, layout).
classify_op(ggml_reshape_2d, layout).
classify_op(ggml_reshape_3d, layout).
classify_op(ggml_reshape_4d, layout).
classify_op(ggml_view_1d, layout).
classify_op(ggml_view_2d, layout).
classify_op(ggml_view_3d, layout).
classify_op(ggml_view_4d, layout).
classify_op(ggml_cont, layout).
classify_op(ggml_cont_1d, layout).
classify_op(ggml_cont_2d, layout).
classify_op(ggml_cont_3d, layout).
classify_op(ggml_cont_4d, layout).
classify_op(ggml_permute, layout).
classify_op(ggml_transpose, layout).
classify_op(ggml_concat, layout).
classify_op(ggml_pad, layout).

classify_op(ggml_rope_ext, spatial).
classify_op(ggml_rope, spatial).
classify_op(ggml_rope_multi, spatial).
classify_op(ggml_conv_1d_ph, spatial).
classify_op(ggml_conv_1d_dw_ph, spatial).

classify_op(ggml_get_rows, gather).
classify_op(ggml_repeat, gather).
classify_op(ggml_cpy, gather).
classify_op(ggml_set_1d, gather).

classify_op(ggml_ssm_conv, ssm).
classify_op(ggml_ssm_scan, ssm).
classify_op(ggml_rwkv_wkv6, rwkv).
classify_op(ggml_rwkv_wkv7, rwkv).

classify_op(ggml_diag, special).
classify_op(ggml_tri, special).
classify_op(ggml_solve_tri, special).
classify_op(ggml_gated_linear_attn, special).

%% Builder-level ops (opaque — cannot fuse through without expansion)
classify_op(build_norm(_), builder).
classify_op(build_attn, builder).
classify_op(build_ffn, builder).
classify_op(build_moe_ffn, builder).
%% build_lora_mm is classified as matmul above (it's essentially matmul)

%% ═════════════════════════════════════════════════════════════════════
%% L1 op classifications (metayen 2026-05-15, Scope B for KernelBench L1)
%% ═════════════════════════════════════════════════════════════════════

%% Additional elementwise activations (each operates on one tensor element)
classify_op(ggml_leaky_relu,   elementwise).
classify_op(ggml_selu,         elementwise).
classify_op(ggml_elu,          elementwise).
classify_op(ggml_hardsigmoid,  elementwise).
classify_op(ggml_softsign,     elementwise).
classify_op(ggml_log_soft_max, reduction).      % needs row max + sum like softmax
classify_op(ggml_neg,          elementwise).
classify_op(ggml_abs,          elementwise).
classify_op(ggml_log,          elementwise).

%% Additional norms (require row reduction → divide)
classify_op(ggml_group_norm,   normalization).

%% Pooling — spatial reduction (different iteration space than matmul)
classify_op(ggml_pool_1d, spatial).
classify_op(ggml_pool_2d, spatial).
classify_op(ggml_pool_3d, spatial).

%% Reductions
classify_op(ggml_max,     reduction).
classify_op(ggml_min,     reduction).
classify_op(ggml_argmax,  reduction).
classify_op(ggml_argmin,  reduction).
classify_op(ggml_cumprod, reduction).

%% Convolutions (spatial; substrate-honest: each dimensionality is its own
%% iteration shape, but they share the convolution kernel family)
classify_op(ggml_conv_1d,           spatial).
classify_op(ggml_conv_2d,           spatial).
classify_op(ggml_conv_3d,           spatial).
classify_op(ggml_conv_transpose_1d, spatial).
classify_op(ggml_conv_transpose_2d, spatial).
classify_op(ggml_conv_transpose_3d, spatial).

%% Loss functions — produce a single scalar from two tensors.
%% Classified as reduction since they involve element-wise op then reduce.
classify_op(ggml_mse_loss,             reduction).
classify_op(ggml_cross_entropy_loss,   reduction).
classify_op(ggml_huber_loss,           reduction).
classify_op(ggml_kl_div_loss,          reduction).
classify_op(ggml_triplet_margin_loss,  reduction).
classify_op(ggml_hinge_loss,           reduction).

%% Flash attention (substrate-honest: this is its own kernel pattern,
%% not a simple op classification. Marked as 'special' for now;
%% future work will give it a dedicated category.)
classify_op(ggml_flash_attn_ext, special).

%% ── GENERAL op_expr fallback ─────────────────────────────────────────────────
%% The ggml clauses above are one backend's vocabulary. For our GENERAL lifted
%% facts (bpd_* ops), delegate to op_classify, which derives the class from the
%% op_expr AST structure — backend-independent. This is the port off ggml: the
%% fusion analyzer now reasons about the general op_expr facts (single source of
%% truth), with ggml as just one downstream backend. (Tried only if no ggml clause
%% above matched, so backward-compat with ggml op-lists is preserved.)
classify_op(BpdOp, Class) :-
    atom(BpdOp),
    atom_concat('bpd_', _, BpdOp),
    op_classify:classify_bpd_op(BpdOp, Class).

%% ═══════════════════════════════════════════════════════════════
%% FUSION RULES
%% ═══════════════════════════════════════════════════════════════
%%
%% A fusion opportunity exists when consecutive operations can be
%% combined into a single kernel, reducing VRAM traffic.
%%
%% KEY PRINCIPLE: fusion eliminates intermediate tensor writes.
%% The fused kernel reads input ONCE, applies all ops in registers
%% or shared memory, writes output ONCE.

%% Epilogue fusion: matmul/conv + chain of elementwise ops
%% This is the most common and highest-value pattern.
%% Example: matmul → bias_add → silu (3 VRAM accesses → 1)
can_fuse(A, B, epilogue) :-
    classify_op(A, matmul),
    classify_op(B, elementwise).
can_fuse(A, B, epilogue_chain) :-
    classify_op(A, elementwise),
    classify_op(B, elementwise).

%% Norm + activation fusion
%% Example: rmsnorm → silu (norm writes intermediate, activation reads it)
can_fuse(A, B, norm_activation) :-
    classify_op(A, normalization),
    classify_op(B, elementwise).

%% Layout ops are free to fuse through (no computation, just metadata)
%% BUT not through opaque builder boundaries
can_fuse(A, B, layout_transparent) :-
    classify_op(A, layout),
    classify_op(B, Class),
    Class \= builder.
can_fuse(A, B, layout_transparent) :-
    classify_op(A, Class),
    Class \= builder,
    classify_op(B, layout).

%% Elementwise → reduction (e.g., scale → sum_rows)
can_fuse(A, B, elementwise_reduction) :-
    classify_op(A, elementwise),
    classify_op(B, reduction).

%% Elementwise -> normalization (e.g., add -> rms_norm, the transformer residual pattern)
%% Iyun fusion_rule_elementwise_into_reduction identifies this; wired into analyzer.
can_fuse(A, B, elementwise_into_norm) :-
    classify_op(A, elementwise),
    classify_op(B, normalization).


%% Reduction → elementwise (e.g., softmax → scale)
%% The reduction output is smaller; elementwise on the result is cheap
can_fuse(A, B, reduce_epilogue) :-
    classify_op(A, reduction),
    classify_op(B, elementwise).

%% ═══════════════════════════════════════════════════════════════
%% THINGS THAT CANNOT BE FUSED
%% ═══════════════════════════════════════════════════════════════

%% Cannot fuse through builder-level ops (opaque)
cannot_fuse(_, B, opaque_builder) :-
    classify_op(B, builder).
cannot_fuse(A, _, opaque_builder) :-
    classify_op(A, builder).

%% Cannot fuse two matmuls (different iteration spaces)
cannot_fuse(A, B, incompatible_iteration_space) :-
    classify_op(A, matmul),
    classify_op(B, matmul).

%% Cannot fuse through a reduction into another reduction
cannot_fuse(A, B, double_reduction) :-
    classify_op(A, reduction),
    classify_op(B, reduction).

%% Elementwise -> normalization (e.g., add -> rms_norm, the transformer residual pattern)
%% Iyun fusion_rule_elementwise_into_reduction identifies this; wired into analyzer.
can_fuse(A, B, elementwise_into_norm) :-
    classify_op(A, elementwise),
    classify_op(B, normalization).


%% ═══════════════════════════════════════════════════════════════
%% CHAIN DISCOVERY
%% ═══════════════════════════════════════════════════════════════
%%
%% Given a list of operations in sequence, find maximal fusible chains.

%% Find all fusible chains in a graph described by BPD facts.
%% GraphFacts is a list of op(Name, Kind, SeqNum) terms.
find_fusible_chains(GraphFacts, Chains) :-
    sort(3, @=<, GraphFacts, Sorted),  % sort by sequence number
    find_chains(Sorted, [], Chains).

find_chains([], Acc, Acc).
find_chains([Op|Rest], Acc, Chains) :-
    extend_chain([Op], Rest, Chain, Remaining),
    ( length(Chain, Len), Len > 1 ->
        find_chains(Remaining, [Chain|Acc], Chains)
    ;
        find_chains(Rest, Acc, Chains)
    ).

extend_chain(Current, [], Current, []) :- !.
extend_chain(Current, [Next|Rest], Chain, Remaining) :-
    Current = [Last|_],
    Last = op(_, KindA, _),
    Next = op(_, KindB, _),
    %% Reason-aware terminal-fusion guard (medayek design, Iyun impl, 2026-05-29).
    %% The chain builds backward (right-to-left), so a "tail is reduction" check
    %% targets the wrong operand (broke L3 norm_activation). Instead: check the
    %% FUSION REASON. Folding an elementwise producer INTO a reduction/norm is a
    %% TERMINAL fusion -- add Next, then STOP (nothing fuses onto a reductions
    %% output). This stops chaining PAST reductions (L2 over-fusion) WITHOUT
    %% blocking legitimate norm->activation (which is a different reason).
    ( can_fuse(KindA, KindB, Reason) ->
        ( ( Reason == elementwise_reduction ; Reason == elementwise_into_norm ) ->
            %% INVARIANT (medayek 2026-05-29): a chain holds AT MOST ONE reduction.
            %% These fusions ADD a reduction (KindB). If Current already contains one
            %% (e.g. entered via reduce_epilogue from a prior reduction), adding a
            %% second would put two reductions in one kernel -- ILLEGAL. So:
            ( chain_has_reduction(Current) ->
                Chain = Current,             % already has a reduction; do NOT add second
                Remaining = [Next|Rest]
            ;
                Chain = [Next|Current],      % fold producer into reduction, then TERMINATE
                Remaining = Rest
            )
        ;
            extend_chain([Next|Current], Rest, Chain, Remaining)
        )
    ;
        Chain = Current,
        Remaining = [Next|Rest]
    ).

%% chain_has_reduction(+Chain): true if any op in the chain is a reduction.
%% (normalization counts too -- it carries a reduction internally.)
chain_has_reduction(Chain) :-
    member(op(_, K, _), Chain),
    ( classify_op(K, reduction) ; classify_op(K, normalization) ),
    !.

%% ═══════════════════════════════════════════════════════════════
%% TRANSFORMER-SPECIFIC FUSION PATTERNS
%% ═══════════════════════════════════════════════════════════════
%%
%% These patterns appear in every transformer forward pass.
%% Recognizing them is the primary value of the fusion analyzer.

%% Pattern: QKV projection + bias + reshape
%% In unfused form: 3 matmuls + 3 optional adds + 3 reshapes = 9 ops
%% Fused: 3 matmul-with-epilogue + 3 zero-cost reshapes = 3 kernel launches
transformer_qkv_fusion(Ops, FusionPlan) :-
    member(op(Q_mul, build_lora_mm, _), Ops),
    member(op(Q_bias, ggml_add, SQ), Ops),
    member(op(Q_reshape, ggml_reshape_3d, _), Ops),
    SQ > 0,  % bias after mul
    FusionPlan = fuse_epilogue(Q_mul, [Q_bias, Q_reshape], matmul_bias_reshape).

%% Pattern: FFN SwiGLU block
%% gate = silu(up * x) ; out = gate * down(x)
%% The silu activation can be fused into the matmul epilogue
transformer_ffn_silu_fusion(Ops, FusionPlan) :-
    member(op(Up, build_lora_mm, _), Ops),
    member(op(Silu, ggml_silu, _), Ops),
    FusionPlan = fuse_epilogue(Up, [Silu], matmul_silu).

%% ═══════════════════════════════════════════════════════════════
%% KERNELBENCH L2 PATTERN MATCHING
%% ═══════════════════════════════════════════════════════════════

%% KernelBench L2 #70: Gemm → Sigmoid → Scaling → ResidualAdd
%% All elementwise after matmul → single epilogue fusion
kernelbench_70(Ops, FusionPlan) :-
    Ops = [op(gemm, ggml_mul_mat, 1),
           op(sigmoid, ggml_sigmoid, 2),
           op(scale, ggml_scale, 3),
           op(residual, ggml_add, 4)],
    FusionPlan = fuse_epilogue(gemm, [sigmoid, scale, residual],
                               'Gemm_Sigmoid_Scaling_ResidualAdd').

%% KernelBench L2 #62: Matmul → GroupNorm → LeakyReLU → Sum
%% matmul + norm is NOT fusible (norm needs full-row stats)
%% But norm + leaky_relu + sum IS fusible
kernelbench_62(Ops, FusionPlan) :-
    Ops = [op(matmul, ggml_mul_mat, 1),
           op(gnorm, ggml_norm, 2),
           op(lrelu, ggml_relu, 3),  % leaky_relu ≈ relu for classification
           op(sum, ggml_add, 4)],
    FusionPlan = [
        no_fuse(matmul, gnorm, 'matmul→norm boundary: norm needs full row'),
        fuse_epilogue(gnorm, [lrelu, sum], 'norm_activation_residual')
    ].
