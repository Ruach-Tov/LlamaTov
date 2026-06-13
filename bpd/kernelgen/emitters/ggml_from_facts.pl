%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ggml_from_facts.pl — the 6th backend: lower op_expr ops to ggml graph ops.
%%
%% The other 5 backends (cuda-oxide, cuda-c, MLIR, LLVM, torch) lower the op_expr
%% AST to a scalar expression / kernel. ggml is different: it's a coarse-grained
%% GRAPH-BUILDER API — each high-level op is one ggml node (ggml_leaky_relu,
%% ggml_gelu, ggml_conv_2d, ...), not a scalar expression. So this backend maps
%% at the OP level (bpd op name -> ggml op kind), emitting the op-list form that
%% bpd/tests/kernelbench_l2_problems.pl hand-wrote and bpd/lib/fusion_analyzer.pl
%% consumes:  [op(Name, GgmlKind, Index), ...]
%%
%% This makes "lower to ggml" just another backend — and lets the chain-lifter
%% AUTO-GENERATE the ggml op-lists (vs hand-encoding them), with our more-general
%% op_expr facts as the single source of truth.
%%
%% Author: Iyun, 2026-06-08 — ggml as backend #6.

:- module(ggml_from_facts, [
    lower_ggml_op/2,          % lower_ggml_op(+BpdOp, -GgmlKind)
    chain_to_ggml/2,          % chain_to_ggml(+OpList, -GgmlOpList)
    chain_to_ggml/3,          % chain_to_ggml(+Name, +OpList, -kb_problem_term)
    ggml_call_shape/2,        % ggml_call_shape(+GgmlKind, -Shape)
    emit_ggml_graph/3         % emit_ggml_graph(+FnName, +OpListWithParams, +OutFile)
]).

%% ── op_expr op -> ggml op kind ───────────────────────────────────────────────
%% Coarse-grained: each bpd op is one ggml graph node. (ggml fuses some of these
%% itself at graph-build; fusion_analyzer reasons about that.)
%% elementwise / activations
lower_ggml_op(bpd_relu,        ggml_relu).
lower_ggml_op(bpd_leaky_relu,  ggml_leaky_relu).
lower_ggml_op(bpd_gelu,        ggml_gelu).
lower_ggml_op(bpd_gelu_tanh,   ggml_gelu).
lower_ggml_op(bpd_silu,        ggml_silu).
lower_ggml_op(bpd_sigmoid,     ggml_sigmoid).
lower_ggml_op(bpd_tanh,        ggml_tanh).
lower_ggml_op(bpd_elu,         ggml_elu).
lower_ggml_op(bpd_softplus,    ggml_softplus).
lower_ggml_op(bpd_hardsigmoid, ggml_hardsigmoid).
lower_ggml_op(bpd_hardtanh,    ggml_clamp).         % hardtanh = clamp(min,max)
lower_ggml_op(bpd_selu,        ggml_elu).           % closest ggml primitive
lower_ggml_op(bpd_mish,        ggml_mish).
lower_ggml_op(bpd_softsign,    ggml_softsign).
lower_ggml_op(bpd_identity,    ggml_cont).          % passthrough/copy
%% scalar binary
lower_ggml_op(bpd_scaling,     ggml_scale).
lower_ggml_op(bpd_scalar_add,  ggml_add).
lower_ggml_op(bpd_scalar_sub,  ggml_sub).
lower_ggml_op(bpd_scalar_div,  ggml_div).
%% reductions / softmax
lower_ggml_op(bpd_softmax,     ggml_soft_max_ext).
lower_ggml_op(bpd_log_softmax, ggml_log_soft_max).
lower_ggml_op(bpd_logsumexp,   ggml_sum_rows).      % logsumexp via exp+sum+log graph
lower_ggml_op(bpd_sum,         ggml_sum_rows).
lower_ggml_op(bpd_mean,        ggml_mean).
lower_ggml_op(bpd_max,         ggml_argmax).        % (max value; argmax node family)
lower_ggml_op(bpd_min,         ggml_argmin).
lower_ggml_op(bpd_argmax,      ggml_argmax).
%% norms
lower_ggml_op(bpd_l1norm,      ggml_l2_norm).       % norm family
lower_ggml_op(bpd_l2norm,      ggml_l2_norm).
lower_ggml_op(bpd_rmsnorm,     ggml_rms_norm).
lower_ggml_op(bpd_frobnorm,    ggml_l2_norm).
lower_ggml_op(bpd_layernorm,   ggml_norm).
lower_ggml_op(bpd_instancenorm,ggml_norm).
lower_ggml_op(bpd_batchnorm,   ggml_norm).
lower_ggml_op(bpd_groupnorm,   ggml_group_norm).
%% pool
lower_ggml_op(bpd_maxpool2d,   ggml_pool_2d).
lower_ggml_op(bpd_avgpool2d,   ggml_pool_2d).
lower_ggml_op(bpd_maxpool1d,   ggml_pool_1d).
lower_ggml_op(bpd_avgpool1d,   ggml_pool_1d).
%% conv (ggml has the full family)
lower_ggml_op(bpd_conv2d,            ggml_conv_2d).
lower_ggml_op(bpd_conv1d,            ggml_conv_1d).
lower_ggml_op(bpd_conv3d,            ggml_conv_3d).
lower_ggml_op(bpd_conv_transpose2d,  ggml_conv_transpose_2d).
lower_ggml_op(bpd_conv_transpose1d,  ggml_conv_transpose_1d).
lower_ggml_op(bpd_conv_transpose3d,  ggml_conv_transpose_3d).
%% matmul / gemm
lower_ggml_op(bpd_matmul,      ggml_mul_mat).
%% losses
lower_ggml_op(bpd_cross_entropy, ggml_cross_entropy_loss).
lower_ggml_op(bpd_huber,         ggml_huber_loss).
lower_ggml_op(bpd_hinge,         ggml_hinge_loss).
lower_ggml_op(bpd_kl_div,        ggml_kl_div_loss).
lower_ggml_op(bpd_mse,           ggml_sub).         % mse = mean((p-t)^2) graph head
%% binary elementwise (tensor operands)
lower_ggml_op(bpd_bias_add,    ggml_add).
lower_ggml_op(bpd_residual_add,ggml_add).
lower_ggml_op(bpd_multiply,    ggml_mul).

%% ── chain_to_ggml(+OpList, -GgmlOpList) ──────────────────────────────────────
%% Map a lifted op-chain [bpd_matmul, bpd_scaling, ...] to the ggml op-list form
%%   [op(Name_1, ggml_mul_mat, 1), op(Name_2, ggml_scale, 2), ...]
%% matching kernelbench_l2_problems' encoding (which can now be GENERATED).
chain_to_ggml(OpList, GgmlOps) :-
    chain_to_ggml_(OpList, 1, GgmlOps).

chain_to_ggml_([], _, []).
chain_to_ggml_([Op|Rest], I, [op(StepName, Ggml, I) | T]) :-
    ( lower_ggml_op(Op, Ggml) -> true ; Ggml = ggml_unknown ),
    short_name(Op, Short),
    format(atom(StepName), "~w_~w", [Short, I]),
    I1 is I + 1,
    chain_to_ggml_(Rest, I1, T).

short_name(Op, Short) :-
    ( atom_concat('bpd_', S, Op) -> Short = S ; Short = Op ).

%% chain_to_ggml(+ProblemName, +OpList, -kb_problem(...)):
%% emit the full kb_problem/4 term so the generated ggml encoding is drop-in
%% compatible with kernelbench_l2_problems.pl.
chain_to_ggml(Name, OpList, kb_problem(Name, generated, Name, GgmlOps)) :-
    chain_to_ggml(OpList, GgmlOps).

%% ── (3) FULL emit_ggml: generate runnable ggml graph-builder C ───────────────
%% Emits a C function that builds the ggml compute graph for a chain:
%%   struct ggml_tensor* build_<fn>(struct ggml_context* ctx, struct ggml_tensor* x){
%%       struct ggml_tensor* t0 = x;
%%       struct ggml_tensor* t1 = ggml_<op>(ctx, t0 [,params]);
%%       ...
%%       return tN;
%%   }
%% This is the runnable artifact (vs the op-list, which is for fusion analysis).
%%
%% ggml_call_shape(+GgmlKind, -Shape): how to emit the call.
%%   unary       -> ggml_op(ctx, a)
%%   scalar(D)   -> ggml_op(ctx, a, <scalarParam|D>)
%%   leaky(D)    -> ggml_op(ctx, a, <slope|D>, false)
%%   binary      -> ggml_op(ctx, a, b)         (second operand: a named input)
%%   softmax     -> ggml_op(ctx, a, NULL, 1.0f, 0.0f)
%%   matmul      -> ggml_op(ctx, w, a)         (w = a named weight input)
ggml_call_shape(ggml_scale,        scalar(1.0)).
ggml_call_shape(ggml_leaky_relu,   leaky(0.01)).
ggml_call_shape(ggml_add,          binary).
ggml_call_shape(ggml_sub,          binary).
ggml_call_shape(ggml_mul,          binary).
ggml_call_shape(ggml_div,          binary).
ggml_call_shape(ggml_soft_max_ext, softmax).
ggml_call_shape(ggml_mul_mat,      matmul).
ggml_call_shape(ggml_conv_2d,      matmul).        % needs a weight input
ggml_call_shape(ggml_conv_transpose_2d, matmul).
ggml_call_shape(ggml_conv_transpose_3d, matmul).
ggml_call_shape(ggml_group_norm,   scalar(8)).     % n_groups
ggml_call_shape(_,                 unary).         % default: unary (ctx, a)

%% emit_ggml_graph(+FnName, +Ops, +OutFile):
%% Ops = list of op(Name, GgmlKind, Index) OR op(Name, GgmlKind, Index, Param).
emit_ggml_graph(FnName, Ops, OutFile) :-
    open(OutFile, write, S),
    format(S, "/* GENERATED ggml graph-builder for chain ~w (backend #6). */~n", [FnName]),
    format(S, "#include \"ggml.h\"~n~n", []),
    format(S, "struct ggml_tensor* build_~w(struct ggml_context* ctx, struct ggml_tensor* x) {~n", [FnName]),
    format(S, "    struct ggml_tensor* t0 = x;~n", []),
    emit_ggml_steps(S, Ops, 0),
    length(Ops, N),
    format(S, "    return t~w;~n}~n", [N]),
    close(S),
    format("Generated ggml graph-builder ~w -> ~w~n", [FnName, OutFile]).

emit_ggml_steps(_, [], _).
emit_ggml_steps(S, [Op|Rest], Prev) :-
    op_kind_param(Op, Kind, Param),
    Cur is Prev + 1,
    ggml_call_shape(Kind, Shape),
    ggml_call_c(Shape, Kind, Prev, Param, Call),
    format(S, "    struct ggml_tensor* t~w = ~w;~n", [Cur, Call]),
    emit_ggml_steps(S, Rest, Cur).

%% extract kind + optional param from an op(...) term (3- or 4-arg)
op_kind_param(op(_, Kind, _), Kind, none).
op_kind_param(op(_, Kind, _, P), Kind, P).

%% ggml_call_c(+Shape, +Kind, +PrevIdx, +Param, -CallString)
ggml_call_c(unary, K, P, _, C) :-
    format(atom(C), "~w(ctx, t~w)", [K, P]).
ggml_call_c(scalar(D), K, P, Param, C) :-
    ( Param == none -> V = D ; V = Param ),
    format(atom(C), "~w(ctx, t~w, ~w)", [K, P, V]).
ggml_call_c(leaky(D), K, P, Param, C) :-
    ( Param == none -> V = D ; V = Param ),
    format(atom(C), "~w(ctx, t~w, ~wf, false)", [K, P, V]).
ggml_call_c(binary, K, P, _, C) :-
    %% second operand is a named extra input (bias/residual) — caller binds 'b'
    format(atom(C), "~w(ctx, t~w, b)", [K, P]).
ggml_call_c(softmax, K, P, _, C) :-
    format(atom(C), "~w(ctx, t~w, NULL, 1.0f, 0.0f)", [K, P]).
ggml_call_c(matmul, K, P, _, C) :-
    %% weight is a named input 'w'; ggml_mul_mat(ctx, w, x)
    format(atom(C), "~w(ctx, w, t~w)", [K, P]).
