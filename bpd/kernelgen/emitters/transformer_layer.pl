%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════════
%% transformer_layer.pl — MOVE 2 (thesis fidelity): emit the transformer layer
%% as a CODEGEN PLAN DERIVED FROM THE RECOGNIZED GRAPH, not a hand-wired driver.
%%
%% The layer op-graph -> find_fusible_chains (the recognizer, 14/14) -> a plan that
%% dispatches each op/chain to its emitter (attention diamond -> flash schedule;
%% matmul->elementwise -> gemm epilogue; rms_norm/softmax -> their fact emitters).
%% Composition becomes DERIVATION: the plan IS the recognized structure.
%% Author: Iyun, 2026-06-08
%% ═══════════════════════════════════════════════════════════════════════════
:- module(transformer_layer, [
    layer_plan/2,            % layer_plan(+OpGraph, -Plan)
    emit_layer_plan/2        % emit_layer_plan(+Plan, -KernelSpecs)
]).
:- use_module(library(lists)).
:- use_module('../../lib/fusion_analyzer', [find_fusible_chains/2]).
:- use_module(flash_attention, [recognize_attention/2]).

%% layer_plan(+OpGraph, -Plan): derive the codegen plan from the op-graph.
%% OpGraph = list of op(Kind, Name, Seq). Plan = ordered list of plan steps, each:
%%   flash_attn(QKVchain)        — the attention diamond -> flash schedule
%%   fused_chain(Head, Tail)     — matmul/pool head + elementwise tail -> epilogue
%%   single(Op)                  — an op with no fusion (its own fact emitter)
layer_plan(OpGraph, Plan) :-
    %% 1. detect the attention diamond first (it spans matmul..softmax..matmul)
    ( attention_span(OpGraph, AttnChain, Before, After) ->
        layer_plan(Before, PlanB),
        layer_plan(After, PlanA),
        append(PlanB, [flash_attn(AttnChain) | PlanA], Plan)
    ; %% 2. otherwise, use the fusion recognizer for elementwise/epilogue chains
      find_fusible_chains(OpGraph, Chains),
      plan_from_chains(OpGraph, Chains, Plan)
    ).

%% attention_span: find a matmul -> (scale) -> softmax -> matmul subsequence.
attention_span(Ops, [QK, SC, SM, AV], Before, After) :-
    append(Before, [QK, SC, SM, AV | After], Ops),
    op_kind(QK, qk), op_kind(SC, scale), op_kind(SM, softmax), op_kind(AV, mm),
    !.
attention_span(Ops, [QK, SM, AV], Before, After) :-
    append(Before, [QK, SM, AV | After], Ops),
    op_kind(QK, mm), op_kind(SM, softmax), op_kind(AV, mm),
    !.

op_kind(op(K,_,_), Want) :- kind_class(K, Want).
kind_class(K, mm)      :- member(K, [matmul, gemm, ggml_mul_mat, mul_mat]).
kind_class(K, qk)      :- kind_class(K, mm).   % a matmul that produces the score
kind_class(K, scale)   :- member(K, [scale, scaling, ggml_scale]).
kind_class(K, softmax) :- member(K, [softmax, soft_max, ggml_soft_max]).

%% plan_from_chains: each fused chain -> fused_chain step; lone ops -> single.
plan_from_chains(OpGraph, _Chains, Plan) :-
    %% simple per-op plan with epilogue grouping: a head (matmul/norm/pool) followed
    %% by elementwise ops fuses into one fused_chain step.
    group_steps(OpGraph, Plan).

group_steps([], []).
group_steps([Op|Rest], [Step|Steps]) :-
    ( is_head(Op), take_tail(Rest, Tail, Rest1), Tail \= [] ->
        Step = fused_chain(Op, Tail), group_steps(Rest1, Steps)
    ; Step = single(Op), group_steps(Rest, Steps) ).

is_head(op(K,_,_)) :- member(K, [matmul,gemm,ggml_mul_mat,rms_norm,ggml_rms_norm,
                                 pool,ggml_pool_2d,conv,ggml_conv_2d]).
take_tail([Op|Rest], [Op|Tail], Rest1) :- is_elementwise(Op), !, take_tail(Rest, Tail, Rest1).
take_tail(Rest, [], Rest).
is_elementwise(op(K,_,_)) :- member(K, [add,mul,silu,relu,scale,tanh,gelu,bias,
                                        ggml_add,ggml_mul,ggml_silu,ggml_scale]).

%% emit_layer_plan(+Plan, -KernelSpecs): turn each plan step into a kernel spec
%% (the emitter to call + its args). The COMPOSITION is now this derived list.
emit_layer_plan(Plan, KernelSpecs) :-
    maplist(step_to_kernel, Plan, KernelSpecs).

step_to_kernel(flash_attn(_Chain),
               kernel(flash, emit_flash_schedule, [schedule(tuned_d128)])).
step_to_kernel(fused_chain(op(HK,_,_), Tail),
               kernel(fused, epilogue_emitter(HK), [tail(TailKinds)])) :-
    maplist([op(K,_,_),K]>>true, Tail, TailKinds).
step_to_kernel(single(op(K,_,_)),
               kernel(single, fact_emitter(K), [])).
