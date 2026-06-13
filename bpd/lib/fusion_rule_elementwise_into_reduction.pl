%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% fusion_rule_elementwise_into_reduction.pl
%% CANDIDATE RULE for medayek review (Iyun, 2026-05-29) -- branch
%% iyun/elementwise-into-reduction-fusion. NOT merged; review before merge.
%%
%% Fills the gap medayek confirmed: fusion_rules.pl has reduction->elementwise
%% (epilogue) but NOT elementwise->reduction (the add;norm case, where a
%% POINTWISE elementwise PRODUCER is folded INTO a reduction/normalization
%% CONSUMER, eliminating one [N] materialization -- one fewer full DRAM pass,
%% which on bandwidth-bound norms is the energy win Heath named).
%%
%% Form matches fusion_rules.pl fusion_rule/4 exactly. Mirrors Rule 1
%% (epilogue_matmul_elementwise) and Rule 2 (elementwise_chain) but with the
%% consumer being a reduction/normalization instead of an elementwise.
%%
%% EQUIVALENCE CLASS = bit_exact, GUARDED:
%%   The reduction sums f(x[i]) over the SAME index range i=0..N in the SAME
%%   order; only the per-element VALUES change (x[i] -> f(x[i])). Summation
%%   ORDER is unchanged, so the accumulation is byte-identical -- PROVIDED the
%%   producer is POINTWISE (no cross-element dependence: f(x)[i] depends only
%%   on x[i]). A gather/permute/scan producer WOULD reorder and is excluded by
%%   the pointwise(EwKind) guard. (Per medayek: bit_exact only if the
%%   elementwise does not change the reduction summation order.)
%%
%% This file ADDS clauses; intended to be reviewed then folded into
%% fusion_rules.pl by medayek, or kept as an imported extension. Does NOT
%% modify core. Depends on fusion_rules.pl vocabulary: op_kind/2, op_class/2,
%% op_input/2, op_output/2, op_reads/3, op_writes/3, no_other_consumers/2.

%% Pointwise guard: producer kinds whose output element i depends ONLY on
%% input element i (no cross-element data flow). Conservative allow-list.
pointwise_elementwise(ggml_add).
pointwise_elementwise(ggml_mul).
pointwise_elementwise(ggml_sub).
pointwise_elementwise(ggml_silu).
pointwise_elementwise(ggml_gelu).
pointwise_elementwise(ggml_relu).
pointwise_elementwise(ggml_neg).
pointwise_elementwise(ggml_abs).

%% reduction/normalization consumer classes (norm family is the immediate target)
reduction_consumer_class(normalization).
reduction_consumer_class(reduction).

%% Rule: fold a pointwise elementwise PRODUCER into a reduction CONSUMER.
fusion_rule(
    elementwise_into_reduction,
    [   % Precondition
        op_kind(EwOp, EwKind),
        op_class(EwKind, elementwise),
        pointwise_elementwise(EwKind),           % GUARD: no cross-element dep
        op_output(EwOp, Intermediate),
        op_kind(RedOp, RedKind),
        op_class(RedKind, RedClass),
        reduction_consumer_class(RedClass),       % consumer is reduction/norm
        op_input(RedOp, Intermediate),
        op_writes(EwOp, Intermediate, region(elementwise, Shape)),
        op_reads(RedOp, Intermediate, region(reduction, Shape)),
        no_other_consumers(Intermediate, [RedOp])  % liveness: intermediate dead
    ],
    [   % Postcondition
        fused_op(FusedOp, [EwOp, RedOp]),
        op_kind(FusedOp, fused(EwKind, RedKind)),
        op_class(FusedOp, RedClass),   % fused kernel is still reduction-bound
        eliminate_tensor(Intermediate) % the [N] write-then-read disappears
    ],
    bit_exact
).
