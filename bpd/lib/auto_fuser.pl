%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════
%% Licensed under RTAAL-1.0 (Ruach Tov AI Agent License)
%% See: LICENSE-RTAAL-1-0.md in the project root
%% ═══════════════════════════════════════════════════════════════════════

%% auto_fuser.pl — Automatic kernel fusion from BPD facts
%%
%% Takes a chain of operations expressed as BPD facts,
%% determines which can be fused, generates the fused kernel,
%% and predicts the pipeline depth required.

:- module(auto_fuser, [
    classify_op/2,           % +OpName, -Class (elementwise|reduction|spatial)
    fusion_plan/2,           % +OpChain, -FusionPlan
    pipeline_depth/2,        % +FusionPlan, -Depth
    generate_fused_chain/2,  % +OpChain, -FusedKernelExpr
    demonstrate_chains/0
]).

:- catch(use_module(veto_ledger), _, true).  % profitability citation (rec E)
:- discontiguous classify_op/2.

%% ═══════════════════════════════════════
%% Operation classification
%% ═══════════════════════════════════════

%% Elementwise: can be fused as epilogue, no pipeline cost
classify_op(add, elementwise).
classify_op(ggml_add, elementwise).
classify_op(ggml_mul, elementwise).
classify_op(ggml_silu, elementwise).
classify_op(ggml_gelu, elementwise).
classify_op(ggml_relu, elementwise).
classify_op(ggml_scale, elementwise).
classify_op(ggml_clamp, elementwise).
classify_op(sub, elementwise).
classify_op(mul, elementwise).
classify_op(div, elementwise).
classify_op(relu, elementwise).
classify_op(silu, elementwise).
classify_op(gelu, elementwise).
classify_op(sigmoid, elementwise).
classify_op(tanh, elementwise).
classify_op(neg, elementwise).
classify_op(abs, elementwise).
classify_op(scale, elementwise).
classify_op(bias_add, elementwise).
classify_op(clamp, elementwise).
classify_op(hardtanh, elementwise).
classify_op(mish, elementwise).
classify_op(swish, elementwise).
classify_op(dropout, elementwise).

%% Reduction: forces a materialization barrier
classify_op(layernorm, reduction).
classify_op(batchnorm, reduction).
classify_op(groupnorm, reduction).
classify_op(instancenorm, reduction).
classify_op(rmsnorm, reduction).
classify_op(softmax, reduction).
classify_op(sum, reduction).
classify_op(mean, reduction).
classify_op(max, reduction).
classify_op(logsumexp, reduction).

%% Spatial: heavyweight, usually standalone
classify_op(matmul, spatial).
classify_op(gemm, spatial).
classify_op(conv2d, spatial).
classify_op(conv3d, spatial).
classify_op(conv_transpose2d, spatial).
classify_op(conv_transpose3d, spatial).
classify_op(maxpool, spatial).
classify_op(avgpool, spatial).

%% ═══════════════════════════════════════
%% Fusion planning: group ops into kernels
%% ═══════════════════════════════════════

%% A fusion plan is a list of kernel_groups.
%% Each group: kernel(Type, Ops, PipelineDepth)
%%   Type: spatial_with_epilogue | reduction_with_epilogue | elementwise_chain
%%   Ops: the operations in this kernel
%%   PipelineDepth: how deep the pipeline needs to be

fusion_plan([], []).
fusion_plan([Op|Rest], Plan) :-
    classify_op(Op, Class),
    (Class = spatial ->
        %% Spatial op starts a new kernel. Absorb following elementwise as epilogue.
        absorb_epilogue(Rest, Epilogue, Remaining),
        length(Epilogue, _NEpi),
        %% Pipeline depth: spatial base (3 for bricklayer) + 0 for epilogue
        Depth = 3,
        fusion_plan(Remaining, RestPlan),
        Plan = [kernel(spatial_with_epilogue, [Op|Epilogue], Depth)|RestPlan]
    ; Class = reduction ->
        %% Reduction starts a new kernel. Absorb following elementwise as epilogue.
        absorb_epilogue(Rest, Epilogue, Remaining),
        %% Pipeline depth: 1 (reduction has no global memory pipeline)
        Depth = 1,
        fusion_plan(Remaining, RestPlan),
        Plan = [kernel(reduction_with_epilogue, [Op|Epilogue], Depth)|RestPlan]
    ; Class = elementwise ->
        %% Consecutive elementwise ops fuse into one kernel
        absorb_elementwise(Rest, MoreElem, Remaining),
        AllElem = [Op|MoreElem],
        Depth = 1,
        fusion_plan(Remaining, RestPlan),
        Plan = [kernel(elementwise_chain, AllElem, Depth)|RestPlan]
    ).

%% Absorb consecutive elementwise ops as epilogue
absorb_epilogue([], [], []).
absorb_epilogue([Op|Rest], [Op|More], Remaining) :-
    classify_op(Op, elementwise), !,
    absorb_epilogue(Rest, More, Remaining).
absorb_epilogue(Remaining, [], Remaining).

%% Absorb consecutive elementwise ops into a chain
absorb_elementwise([], [], []).
absorb_elementwise([Op|Rest], [Op|More], Remaining) :-
    classify_op(Op, elementwise), !,
    absorb_elementwise(Rest, More, Remaining).
absorb_elementwise(Remaining, [], Remaining).

%% ═══════════════════════════════════════
%% Pipeline depth for a fusion plan
%% ═══════════════════════════════════════

pipeline_depth(Plan, MaxDepth) :-
    maplist([kernel(_,_,D)]>>true, Plan, _),
    findall(D, member(kernel(_,_,D), Plan), Depths),
    max_list(Depths, MaxDepth).

%% ═══════════════════════════════════════
%% Generate fused expression from chain
%% ═══════════════════════════════════════

generate_fused_chain([], identity).
generate_fused_chain([Op], Op).
generate_fused_chain([Op|Rest], fused(Op, RestExpr)) :-
    generate_fused_chain(Rest, RestExpr).

%% ═══════════════════════════════════════
%% Demonstrate with real L2 chains
%% ═══════════════════════════════════════


%% cite_chain(+Chain): print the PROFITABILITY citation for a chain (validity != profitability, rec E).
%% Matmul/conv + elementwise epilogue is the always-profitable canonical case; anything matching a
%% known anti-fusion pattern is flagged DO-NOT-WIRE; else UNKNOWN (honest absence). The scanner reports
%% what CAN be fused; this says what SHOULD be — so a reader never wires a known 0.55x loser.
cite_chain(Chain) :-
    ( chain_pattern(Chain, Pat), catch(veto_ledger:cite(Pat, C), _, fail)
    -> format("  Profitability: ~w~n", [C])
    ;  format("  Profitability: UNKNOWN (validity only) — MEASURE before wiring~n", []) ).

%% chain_pattern(+Chain, -LedgerPattern): map an op chain to a veto-ledger pattern.
chain_pattern(Chain, epilogue_matmul_elementwise) :-
    Chain = [Head|Rest], member(Head, [gemm, matmul, conv2d]),
    forall(member(Op, Rest), elementwise_epilogue_op(Op)), !.
chain_pattern(Chain, store_side_silu) :-       % a known anti-fusion shape (silu feeding a store/requant)
    append(_, [silu, store|_], Chain), !.
chain_pattern(Chain, reciprocal_mul_softmax) :-
    member(softmax, Chain), member(scale, Chain), !.
chain_pattern(_, unknown_pattern).

elementwise_epilogue_op(Op) :- member(Op, [bias_add, add, relu, swish, silu, silu_mul, scale, tanh, gelu, hardtanh, mish, sigmoid, clamp]).


demonstrate_chains :-
    format("═══ AUTO-FUSER: L2 CHAIN ANALYSIS ═══~n~n"),

    %% L2 #76: Gemm + Add + ReLU
    Chain1 = [gemm, bias_add, relu],
    fusion_plan(Chain1, Plan1),
    format("L2 #76  Gemm+Add+ReLU:~n"),
    format("  Chain: ~w~n", [Chain1]),
    format("  Plan:  ~w~n", [Plan1]),
    cite_chain(Chain1),
    format("  → 1 kernel (gemm with add+relu epilogue), pipeline depth 3~n~n"),

    %% L2 #59: Matmul + Swish + Scaling
    Chain2 = [matmul, swish, scale],
    fusion_plan(Chain2, Plan2),
    format("L2 #59  Matmul+Swish+Scaling:~n"),
    format("  Chain: ~w~n", [Chain2]),
    format("  Plan:  ~w~n", [Plan2]),
    cite_chain(Chain2),
    format("  → 1 kernel (matmul with swish+scale epilogue)~n~n"),

    %% L2 #33: Gemm + Scale + BatchNorm
    Chain3 = [gemm, scale, batchnorm],
    fusion_plan(Chain3, Plan3),
    format("L2 #33  Gemm+Scale+BatchNorm:~n"),
    format("  Chain: ~w~n", [Chain3]),
    format("  Plan:  ~w~n", [Plan3]),
    cite_chain(Chain3),
    format("  → 2 kernels (gemm+scale, then batchnorm). Reduction forces split.~n~n"),

    %% L2 #84: Gemm + BatchNorm + Scaling + Softmax
    Chain4 = [gemm, batchnorm, scale, softmax],
    fusion_plan(Chain4, Plan4),
    format("L2 #84  Gemm+BatchNorm+Scaling+Softmax:~n"),
    format("  Chain: ~w~n", [Chain4]),
    format("  Plan:  ~w~n", [Plan4]),
    cite_chain(Chain4),
    format("  → 3 kernels. Two reductions force two splits.~n~n"),

    %% L2 #37: Matmul + Swish + Sum + GroupNorm
    Chain5 = [matmul, swish, sum, groupnorm],
    fusion_plan(Chain5, Plan5),
    format("L2 #37  Matmul+Swish+Sum+GroupNorm:~n"),
    format("  Chain: ~w~n", [Chain5]),
    format("  Plan:  ~w~n", [Plan5]),
    cite_chain(Chain5),
    format("  → 3 kernels. Matmul+swish fused, then sum, then groupnorm.~n~n"),

    %% L2 #4: Conv2d + Mish + Mish (pure elementwise after spatial)
    Chain6 = [conv2d, mish, mish],
    fusion_plan(Chain6, Plan6),
    format("L2 #4   Conv2d+Mish+Mish:~n"),
    format("  Chain: ~w~n", [Chain6]),
    format("  Plan:  ~w~n", [Plan6]),
    cite_chain(Chain6),
    format("  → 1 kernel (conv2d with mish+mish epilogue). ALL FUSED.~n~n"),

    %% L2 #95: Matmul+Add+Swish+Tanh+GELU+Hardtanh (6-deep elementwise)
    Chain7 = [matmul, bias_add, swish, tanh, gelu, hardtanh],
    fusion_plan(Chain7, Plan7),
    format("L2 #95  Matmul+Add+Swish+Tanh+GELU+Hardtanh:~n"),
    format("  Chain: ~w~n", [Chain7]),
    format("  Plan:  ~w~n", [Plan7]),
    cite_chain(Chain7),
    format("  → 1 kernel! 5 elementwise ops fuse into matmul epilogue.~n"),
    format("    Pipeline depth unchanged at 3 (epilogue is free).~n~n"),

    %% Answer Heath's question
    format("═══ PIPELINE DEPTH ANALYSIS ═══~n~n"),
    format("  Elementwise epilogue (any depth): pipeline depth UNCHANGED~n"),
    format("    Adding relu, silu, gelu, scale, clamp to any kernel = FREE~n"),
    format("    The ops execute in registers, no new pipeline stages.~n~n"),
    format("  Reduction barrier: forces +1 kernel (new pipeline context)~n"),
    format("    layernorm, softmax, batchnorm = NECESSARY materialization~n"),
    format("    The pipeline DRAINS, reduction runs, new pipeline starts.~n~n"),
    format("  Multiple reductions: each adds 1 kernel~n"),
    format("    gemm+batchnorm+softmax = 3 kernels minimum~n"),
    format("    No amount of fusion can merge across a reduction boundary.~n~n"),
    format("  ANSWER: deeper fusion does NOT need deeper pipelines.~n"),
    format("  It needs MORE kernels when reductions intervene,~n"),
    format("  but each kernel's pipeline depth stays at 3 (bricklayer).~n").

%% Extended op classifications for L2 fusion detection
classify_op(ggml_conv2d, spatial).
classify_op(ggml_conv3d, spatial).
classify_op(ggml_conv_transpose2d, spatial).
classify_op(ggml_conv_transpose3d, spatial).
classify_op(ggml_mul_mat, spatial).
classify_op(ggml_matmul, spatial).
classify_op(ggml_linear, spatial).
classify_op(linear, spatial).
classify_op(ggml_maxpool, spatial).
classify_op(ggml_avgpool, spatial).
classify_op(ggml_batchnorm, reduction).
classify_op(ggml_layernorm, reduction).
classify_op(ggml_groupnorm, reduction).
classify_op(ggml_instancenorm, reduction).
classify_op(ggml_hardswish, elementwise).
classify_op(hardswish, elementwise).
classify_op(ggml_hardsigmoid, elementwise).
classify_op(hardsigmoid, elementwise).
classify_op(ggml_leaky_relu, elementwise).
classify_op(leaky_relu, elementwise).
classify_op(ggml_elu, elementwise).
classify_op(elu, elementwise).
classify_op(ggml_selu, elementwise).
classify_op(selu, elementwise).
classify_op(ggml_softplus, elementwise).
classify_op(softplus, elementwise).
classify_op(ggml_sigmoid, elementwise).
classify_op(ggml_tanh, elementwise).
classify_op(ggml_mish, elementwise).
classify_op(ggml_softmax, reduction).
classify_op(softmax, reduction).

%% Engine-name aliases (production kernel vocabulary — name-drift fix, Bocher 2026-06-13).
%% The engine spells these rms_norm / silu_mul / q8_gemv; the scanner previously only
%% knew rmsnorm/silu/gemm. Aliases keep the scanner usable on real decode-graph op lists.
classify_op(rms_norm, reduction).
classify_op(silu_mul, elementwise).
classify_op(q8_gemv, spatial).
classify_op(q8_0_gemv, spatial).
classify_op(quant_q8, elementwise).
classify_op(rope, elementwise).
classify_op(argmax, reduction).
