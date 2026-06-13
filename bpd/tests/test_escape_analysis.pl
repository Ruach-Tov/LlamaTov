%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_escape_analysis.pl — Test escape analysis on QKV graph.
%%
%% Uses op_inputs/op_output facts from the QKV BPD to detect
%% multi-consumer tensors that cannot be eliminated by fusion.

:- set_prolog_flag(double_quotes, codes).
:- use_module('../lib/fusion_analyzer').

%% ═══════════════════════════════════════════════════════════════
%% QKV GRAPH FACTS (from qkv.bpd — data flow)
%% ═══════════════════════════════════════════════════════════════

%% Operations with inputs and outputs
op_def(qkv_norm,    build_norm(rms), 1, [inpL, attn_norm_w],     cur_after_norm).
op_def(wq_mul,      build_lora_mm,   2, [wq, cur_after_norm],    'Qcur_pre_bias').
op_def(wq_bias_add, ggml_add,        3, ['Qcur_pre_bias', bq],   'Qcur_post_bias').
op_def(wk_mul,      build_lora_mm,   4, [wk, cur_after_norm],    'Kcur_pre_bias').
op_def(wk_bias_add, ggml_add,        5, ['Kcur_pre_bias', bk],   'Kcur_post_bias').
op_def(wv_mul,      build_lora_mm,   6, [wv, cur_after_norm],    'Vcur_pre_bias').
op_def(wv_bias_add, ggml_add,        7, ['Vcur_pre_bias', bv],   'Vcur_post_bias').

%% Tensor joins (SSA-phi for conditional bias)
tensor_join('Qcur', 'Qcur_post_bias').  % simplified: assume bias present
tensor_join('Kcur', 'Kcur_post_bias').
tensor_join('Vcur', 'Vcur_post_bias').

op_def(q_reshape,   ggml_reshape_3d, 8,  ['Qcur', n_embd_head, n_head, n_tokens], 'Qcur_3d').
op_def(k_reshape,   ggml_reshape_3d, 9,  ['Kcur', n_embd_head, n_head_kv, n_tokens], 'Kcur_3d').
op_def(v_reshape,   ggml_reshape_3d, 10, ['Vcur', n_embd_head, n_head_kv, n_tokens], 'Vcur_3d').
op_def(q_rope,      ggml_rope_ext,   11, ['Qcur_3d', inp_pos], 'Qcur_post_rope').
op_def(k_rope,      ggml_rope_ext,   12, ['Kcur_3d', inp_pos], 'Kcur_post_rope').

%% ═══════════════════════════════════════════════════════════════
%% ESCAPE ANALYSIS
%% ═══════════════════════════════════════════════════════════════

%% A tensor "escapes" a chain if it's consumed by ops OUTSIDE the chain.
%% Multi-consumer tensors MUST be written to VRAM — fusion can't eliminate them.

%% Find all consumers of a tensor
tensor_consumers(Tensor, Consumers) :-
    findall(OpName, (op_def(OpName, _, _, Inputs, _), member(Tensor, Inputs)), Consumers).

%% Find consumers through tensor_join (option b — join-aware)
tensor_consumers_with_joins(Tensor, AllConsumers) :-
    tensor_consumers(Tensor, Direct),
    findall(JoinConsumers, (
        tensor_join(JoinedName, Tensor),
        tensor_consumers(JoinedName, JoinConsumers)
    ), JoinConsumerLists),
    append([Direct|JoinConsumerLists], AllConsumers).

%% A tensor is a fusion barrier if it has >1 consumer
is_fusion_barrier(Tensor) :-
    tensor_consumers_with_joins(Tensor, Consumers),
    length(Consumers, N),
    N > 1.

%% Validate a chain: check that no INTERMEDIATE tensor escapes
%% (first op's inputs and last op's output are allowed to be multi-consumer)
validate_chain(Chain) :-
    Chain = [_First|Rest],
    ( Rest = [] -> true  % single-op chain is always valid
    ;
        last(Rest, _Last),
        append([_First], Intermediates, Chain),
        append(Intermediates, [_Last], Chain2), % intermediates = all but first and last
        % Actually: intermediate OUTPUTS of all but the last op
        Chain2 = Chain, % (this is just for clarity)
        forall((member(OpName, Intermediates),
                op_def(OpName, _, _, _, Output)),
               \+ is_fusion_barrier(Output))
    ).

%% ═══════════════════════════════════════════════════════════════
%% TESTS
%% ═══════════════════════════════════════════════════════════════

test :-
    write('=== Multi-Consumer Tensor Detection ==='), nl,
    forall(op_def(_, _, _, _, Output), (
        tensor_consumers_with_joins(Output, Consumers),
        length(Consumers, N),
        ( N > 1 ->
            format("  ~w: ~d consumers ~w  *** BARRIER ***~n", [Output, N, Consumers])
        ; N =:= 1 ->
            format("  ~w: ~d consumer~n", [Output, N])
        ;
            format("  ~w: ~d consumers (terminal)~n", [Output, N])
        )
    )),
    nl,

    write('=== Fusion Chain Validation ==='), nl,
    %% Build the ops list for chain discovery
    findall(op(Name, Kind, Seq), op_def(Name, Kind, Seq, _, _), Ops),
    find_fusible_chains(Ops, Chains),
    forall(member(Chain, Chains), (
        reverse(Chain, Fwd),
        findall(N, member(op(N,_,_), Fwd), Names),
        %% Check intermediate escapes
        findall(N, member(op(N,_,_), Fwd), OpNames),
        ( length(OpNames, 1) ->
            format("  Chain ~w: trivial (1 op)~n", [Names])
        ;
            %% Get intermediate ops (all but first and last)
            append([_|Intermediates], [_], OpNames),
            findall(Esc, (
                member(IntOp, Intermediates),
                op_def(IntOp, _, _, _, IntOutput),
                is_fusion_barrier(IntOutput)
            ), Escapes),
            ( Escapes = [] ->
                format("  Chain ~w: VALID (no escapes)~n", [Names])
            ;
                format("  Chain ~w: INVALID (escapes: ~w)~n", [Names, Escapes])
            )
        )
    )),
    nl.

:- initialization((test -> halt(0) ; (write('FAILED'), nl, halt(1)))).
