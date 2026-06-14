%% ─────────────────────────────────────────────────────────────────────────────
%% transform_bridge.pl — role-based model transformation: model_transform(Model, Strategy).
%%
%% The join the substrate was missing. coordinate_taps gives `type(projection)`;
%% cost_naming declares metas by `role(kv_projection)`; but nothing INFERRED the role from
%% the graph. This file is that inference: op_role/3 labels each op with a semantic ROLE
%% (kv_projection, q_projection, ffn_projection, skip_connection) derived from DATAFLOW —
%% transitively, through role-transparent ops (bias add, rope, reshape) — never from a
%% tensor name. A transform declared as `attaches_at(role(kv_projection))` then finds its
%% attachment points in ANY model whose graph the map can label, by role, not by matching a
%% hardcoded subgraph. Verified on the live qwen2 graph (gguf_to_graph.py): turboquant
%% attaches at exactly the 48 K/V projections, attnres at the 48 true residuals (the 72
%% qkv-bias adds correctly excluded). Pairs with gguf_to_graph.py (the live-GGUF graph
%% deriver) and the meta contracts in cost_naming.pl.
%%
%% Building toward model_transform(Model, Strategy) — the declarative model-to-model
%% rewriting milestone. RTAAL-1.1-only (the model-transformation capability).
%%
%% SPDX-License-Identifier: LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ─────────────────────────────────────────────────────────────────────────────
:- module(transform_bridge, [
    op_role/3,            % op_role(+Graph, ?OpId, ?Role)         — the inferred semantic role
    coordinate_map/2,     % coordinate_map(+Graph, -Map)          — the full labelled map
    meta_attach_points/3, % meta_attach_points(+Graph, +Role, -OpIds) — where a role-meta attaches
    model_transform/4,    %% model_transform(+Graph,+Strategy,-NewGraph,-Applied)
    strategy_role/3,
    model_transform_q8/3,  %% kv_quantize_q8: real insert-mode Q8_0 transform
    print_map/1
]).

%% ── a tap_type by op kind (the ggml vocabulary, from coordinate_taps_seq) ──
tap_type(ggml_mul_mat, projection).
tap_type(matmul,       projection).
tap_type(ggml_add,     residual).
tap_type(ggml_rms_norm, reduction).
tap_type(ggml_soft_max_ext, reduction).
tap_type(ggml_silu,    activation).
tap_type(ggml_mul,     elementwise).
tap_type(flash_attention, attention).
tap_type(_, unknown).

%% ── op_role/3: the ROLE inference — type + DATAFLOW. ────────────────────────
%% A projection's ROLE is determined by WHAT IT FEEDS, not its name. This is what
%% makes it portable: a new model with a differently-NAMED k_proj still gets
%% role(kv_projection) if its output flows into attention as K or V.
%% Graph is a list of: op(OpId, Kind, Inputs, Output).

op_kind_in(Graph, OpId, Kind)  :- member(op(OpId, Kind, _, _), Graph).
op_output_in(Graph, OpId, Out) :- member(op(OpId, _, _, Out), Graph).
op_inputs_in(Graph, OpId, Ins) :- member(op(OpId, _, Ins, _), Graph).

%% find the attention op + its [Q,K,V] inputs (by position).
attention_qkv(Graph, AttnId, Q, K, V) :-
    member(op(AttnId, flash_attention, [Q,K,V|_], _), Graph).

%% ── TRANSITIVE dataflow: does tensor T reach attention slot Target (Q|K|V)? ──
%% A projection rarely feeds attention DIRECTLY on a real model — its output passes
%% through bias adds and rope first. "Reaches" follows those position/element-local
%% ops (which preserve the tensor's ROLE) until it hits the attention input, with no
%% other projection (mul_mat) intervening (a new mul_mat would start a new role).
%% role_transparent: ops that pass a tensor's attention-role through unchanged.
role_transparent(ggml_add).    %% bias add
role_transparent(ggml_rope).   %% positional rotation
role_transparent(ggml_rms_norm).  %% per-head Q/K norm (Gemma2/Qwen3) — role passes through
role_transparent(ggml_reshape_2d). role_transparent(ggml_reshape_3d).
role_transparent(ggml_cont). role_transparent(ggml_permute). role_transparent(ggml_view).

reaches(_Graph, T, T).                               %% base: T is the slot itself
reaches(Graph, T, Target) :-                         %% step: T feeds a transparent op -> its output
    member(op(_, Kind, Ins, Out), Graph),
    role_transparent(Kind),
    Ins = [First|_], First == T,    %% trace through the ACTIVATION (first) input only, not weights
    Out \== T,
    reaches(Graph, Out, Target).

%% kv_projection: a projection whose output TRANSITIVELY reaches attention's K or V slot.
op_role(Graph, OpId, kv_projection) :-
    op_kind_in(Graph, OpId, Kind), tap_type(Kind, projection),
    op_output_in(Graph, OpId, Out),
    attention_qkv(Graph, _Attn, _Q, K, V),
    ( reaches(Graph, Out, K) ; reaches(Graph, Out, V) ).

%% q_projection: projection reaching attention's Q slot.
op_role(Graph, OpId, q_projection) :-
    op_kind_in(Graph, OpId, Kind), tap_type(Kind, projection),
    op_output_in(Graph, OpId, Out),
    attention_qkv(Graph, _Attn, Q, _K, _V),
    reaches(Graph, Out, Q).

%% ffn_projection: a projection reaching NO attention slot.
op_role(Graph, OpId, ffn_projection) :-
    op_kind_in(Graph, OpId, Kind), tap_type(Kind, projection),
    op_output_in(Graph, OpId, Out),
    \+ ( attention_qkv(Graph, _, Q, K, V), member(S, [Q,K,V]), reaches(Graph, Out, S) ).

%% skip_connection: a TRUE residual add (AttnRes attaches), NOT a weight-bias add.
%% Distinguisher: a residual add adds two ACTIVATIONS; a bias add adds an activation +
%% a PARAMETER (weight/bias). An activation is produced by an op OR is the residual-stream
%% graph input (consumed but never produced, and used in an activation position). A parameter
%% is a leaf consumed ONLY in a weight position: the 2nd arg of a projection (mul_mat) or the
%% 2nd arg of a bias add. (Position-based, name-free -> still portable.)
produced(Graph, T) :- op_output_in(Graph, _, T).

%% used_in_weight_position: T is the 2nd input of a mul_mat (a weight) or 2nd of an add (a bias).
used_in_weight_position(Graph, T) :-
    member(op(_, ggml_mul_mat, [_, T], _), Graph).
used_in_weight_position(Graph, T) :-
    member(op(_, ggml_add, [_, T], _), Graph),
    \+ produced(Graph, T).          %% only a LEAF 2nd-arg of add is a bias; an activation 2nd-arg is fine

%% used_in_activation_position: T appears as a 1st input anywhere (the data/residual stream).
used_in_activation_position(Graph, T) :-
    member(op(_, _, [T|_], _), Graph).

is_parameter(Graph, T) :-
    \+ produced(Graph, T),                       %% a leaf
    used_in_weight_position(Graph, T),
    \+ used_in_activation_position(Graph, T).    %% and NEVER used as data (excludes the residual stream)

is_activation(Graph, T) :-
    ( produced(Graph, T) -> true                 %% an op output is an activation
    ; \+ is_parameter(Graph, T) ).               %% or a leaf that is not a parameter (the residual input)

op_role(Graph, OpId, skip_connection) :-
    member(op(OpId, ggml_add, [A, B], _), Graph),
    is_activation(Graph, A),
    is_activation(Graph, B).      %% both inputs are activations -> a real residual, not a bias

%% ── coordinate_map/2: every op -> (kind, type, role) ────────────────────────
coordinate_map(Graph, Map) :-
    findall(coord(OpId, Kind, Type, Role),
            ( member(op(OpId, Kind, _, _), Graph),
              once(tap_type(Kind, Type)),
              ( op_role(Graph, OpId, Role) -> true ; Role = none ) ),
            Map).

%% ── meta_attach_points/3: where does a role-declared meta attach? ───────────
%% THIS is the portable query: "turboquant attaches at role(kv_projection)" ->
%% give me every op in THIS graph with that role. Works on any labelled model.
meta_attach_points(Graph, Role, OpIds) :-
    findall(OpId, op_role(Graph, OpId, Role), OpIds0),
    sort(OpIds0, OpIds).

print_map(Graph) :-
    coordinate_map(Graph, Map),
    format("~n=== COORDINATE MAP ===~n"),
    forall(member(coord(Id, Kind, Type, Role), Map),
           format("  ~w~t~22|kind=~w~t~46|type=~w~t~62|role=~w~n", [Id, Kind, Type, Role])).

%% ─────────────────────────────────────────────────────────────────────────────
%% Milestone 2: model_transform/3 — APPLY a strategy by ROLE.
%%   model_transform(+Graph, +Strategy, -NewGraph, -Applied)
%% Looks up the strategy's declared role (its meta/4 contract), finds the attach
%% points in THIS graph by role, and rewrites them. Portable: same call works on
%% any model the map can label. This is `model_transform(llama3, turbo_quant)`.
%% ─────────────────────────────────────────────────────────────────────────────

%% strategy_role/3: the meta contract (mirrors cost_naming.pl meta/4). A strategy
%% attaches at a role and applies a per-op rewrite. (In production these come from
%% cost_naming:meta/4; inlined here so the prototype is self-contained.)
strategy_role(turboquant, kv_projection, encode(turboquant)).  % polar-quantize K/V projections
strategy_role(attnres,    skip_connection, rewrite(attn_residual)). % residual -> learned attention

%% rewrite_op/4: apply a strategy's effect to one op, producing the new op + a provenance fact.
%% encode(E): tag the op's output tensor with a tensor_encoding (lossy, but structure-preserving).
apply_effect(encode(E), op(Id,Kind,Ins,Out), op(Id,Kind,Ins,Out), tensor_encoding(Out, E)).
%% rewrite(NewKind): change the op kind (a structural/mathematical transform).
apply_effect(rewrite(NewKind), op(Id,_Kind,Ins,Out), op(Id,NewKind,Ins,Out), transform_applied(Id, NewKind)).

model_transform(Graph, Strategy, NewGraph, Applied) :-
    strategy_role(Strategy, Role, Effect),
    meta_attach_points(Graph, Role, Points),
    foldl(apply_one(Strategy, Effect, Points), Graph, [], RevNewWithProv),
    %% collect provenance separately
    findall(P, member(prov(P), RevNewWithProv), Provs),
    findall(O, member(opx(O), RevNewWithProv), RevOps),
    reverse(RevOps, NewGraph),
    Applied = applied(Strategy, at(Points), provenance(Provs)).

apply_one(_Strategy, Effect, Points, op(Id,Kind,Ins,Out), Acc, [opx(NewOp), prov(Prov) | Acc]) :-
    memberchk(Id, Points), !,
    apply_effect(Effect, op(Id,Kind,Ins,Out), NewOp, Prov).
apply_one(_Strategy, _Effect, _Points, Op, Acc, [opx(Op) | Acc]).  % untouched op


%% ─────────────────────────────────────────────────────────────────────────────
%% kv_quantize_q8: a REAL, referee-verifiable model transformation.
%% Unlike the tag-only encode(turboquant), this INSERTS actual quantize+dequant ops
%% into the graph at each K/V projection output, so the K/V genuinely round-trips
%% through Q8_0. The correctness contract is CHECKABLE: each element's error <= d/2
%% (half the per-32-block quantization step, d=amax/127). Verified by the referee
%% against the engine's trusted Q8_0 arithmetic (kv_quant_ref.py).
%%
%% model_transform_q8/4: insert q8_quantize -> q8_dequant after each kv_projection's
%% OUTPUT, and rewire every downstream consumer of that output to read the
%% reconstructed tensor. The role-found points come from the SAME op_role/3, so this
%% is portable to any model the map can label.
%% ─────────────────────────────────────────────────────────────────────────────

%% the new activation symbols for an inserted quant/dequant pair on tensor T.
q8_syms(T, Q, D) :-
    format(atom(Q), '~w_q8', [T]),     % the quantized int8+scale tensor
    format(atom(D), '~w_dq', [T]).     % the dequantized (reconstructed) tensor

%% rewire(+Old, +New, +Op, -Op2): replace Old with New in an op's INPUT list.
rewire(Old, New, op(Id,K,Ins,Out), op(Id,K,Ins2,Out)) :-
    maplist([I,O2]>>(I == Old -> O2 = New ; O2 = I), Ins, Ins2).

model_transform_q8(Graph, NewGraph, Applied) :-
    %% find the kv_projection ops and their output tensors.
    findall(Out-Id,
            ( op_role(Graph, Id, kv_projection), op_output_in(Graph, Id, Out) ),
            KVPairs0),
    sort(KVPairs0, KVPairs),
    findall(Out, member(Out-_, KVPairs), KVOuts),
    %% for each KV output T: the quant/dequant insertions + provenance.
    findall(insert(T, op(quant(T), q8_quantize, [T], Q), op(dequant(T), q8_dequant, [Q], D), D),
            ( member(T, KVOuts), q8_syms(T, Q, D) ),
            Inserts),
    %% rewire every op that CONSUMES a KV output (but is not the projection itself, and
    %% not a quant op we just made) to read the dequantized tensor instead.
    rewire_consumers(Graph, Inserts, Rewired),
    %% splice the quant+dequant ops in right after each projection.
    splice_inserts(Rewired, Inserts, NewGraph),
    findall(prov(q8_inserted(T)), member(insert(T,_,_,_), Inserts), Provs),
    Applied = applied(kv_quantize_q8, at(KVOuts), provenance(Provs)).

%% rewire_consumers: any op reading a KV output T -> reads its dequantized form D instead.
%% (The quant op itself still reads the raw T; everything downstream reads D.)
rewire_consumers(Graph, Inserts, Out) :-
    foldl(rewire_one_kv(Inserts), Graph, [], RevOut), reverse(RevOut, Out).
rewire_one_kv(Inserts, Op, Acc, [Op2|Acc]) :-
    foldl(maybe_rewire(Op), Inserts, Op, Op2).
maybe_rewire(_OrigOp, insert(T,_,_,D), OpIn, OpOut) :-
    OpIn = op(Id,K,Ins,O),
    ( memberchk(T, Ins), Id \== quant(T)   %% consumes T and isn't the quantizer
    -> rewire(T, D, op(Id,K,Ins,O), OpOut)
    ;  OpOut = OpIn ).

%% splice_inserts: place each (q8_quantize, q8_dequant) pair immediately after the op
%% that PRODUCES T (the projection), preserving order.
splice_inserts(Graph, Inserts, Out) :-
    foldl(splice_one(Inserts), Graph, [], RevOut), reverse(RevOut, FlatRev),
    flatten_ops(FlatRev, Out).
splice_one(Inserts, Op, Acc, [Group|Acc]) :-
    Op = op(_,_,_,Produced),
    ( member(insert(Produced, QOp, DOp, _), Inserts)
    -> Group = [Op, QOp, DOp]
    ;  Group = [Op] ).
flatten_ops(Groups, Flat) :- foldl([G,A,B]>>append(A,G,B), Groups, [], Flat).
