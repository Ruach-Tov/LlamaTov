%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% tensor_types.pl — Type derivation engine for tensor operations.
%%
%% SHARED SCHEMA (agreed with Iyun):
%%   tensor(Name, dtype(D), shape(Dims), layout(L))
%%   op(NodeId, Coord, OpType, in(Srcs), out(Dst))
%%   op_role(NodeId, RoleTerm)
%%   op_verdict(NodeId, measured(...))
%%
%% This module owns the MIDDLE LAYER:
%%   - Type derivation: derive output tensor from input + op
%%   - Pipeline verification: check a sequence of ops is well-typed
%%   - Bandwidth computation: bytes read/written per op
%%   - Static checks: catch shape/dtype/layout mismatches
%%
%% Iyun owns BOTTOM (spec extraction, measurement → op_verdict facts)
%% Heath owns TOP (visual rendering, bandwidth overlays)
%%
%% Author: mavchin (2026-06-02)

:- module(tensor_types, [
    derive_type/3,
    verify_pipeline/3,
    op_bandwidth/5,
    tensor_bytes/2,
    bytes_per_element/2,
    bandwidth_class/2,
    describe_op/2,
    describe_role/2,
    check_pipeline/2
]).

%% ============================================================
%% Dtype → bytes per element
%% ============================================================

bytes_per_element(f32, 4).
bytes_per_element(f16, 2).
bytes_per_element(bf16, 2).
bytes_per_element(q8_0, 1.0625).   %% 34 bytes per 32 elements
bytes_per_element(q4_0, 0.5625).   %% 18 bytes per 32 elements
bytes_per_element(q4_k, 0.5625).
bytes_per_element(i32, 4).
bytes_per_element(i16, 2).
bytes_per_element(i8, 1).

%% ============================================================
%% Tensor size in bytes
%% ============================================================

tensor_bytes(tensor(_, dtype(D), shape(Dims), _), Bytes) :-
    product(Dims, Elems),
    bytes_per_element(D, BPE),
    Bytes is Elems * BPE.

product([], 1).
product([D|Ds], P) :-
    product(Ds, Rest),
    (   integer(D) -> P is D * Rest
    ;   P = D * Rest   %% symbolic — keep as term
    ).

%% ============================================================
%% Type derivation — derive output type from input type + op
%% ============================================================

%% reshape: same dtype, same element count, new shape, becomes contiguous
derive_type(reshape(NewShape),
            tensor(N, D, shape(OldShape), _L),
            tensor(N, D, shape(NewShape), contiguous)) :-
    product(OldShape, P1),
    product(NewShape, P2),
    (   integer(P1), integer(P2) -> P1 =:= P2
    ;   true  %% symbolic dims — trust the programmer
    ).

%% transpose: swap dimensions according to permutation, becomes strided
derive_type(transpose(Perm),
            tensor(N, D, shape(Shape), _L),
            tensor(N, D, shape(PermShape), strided)) :-
    permute_list(Shape, Perm, PermShape).

%% transpose/0: swap last two dimensions (common case)
derive_type(transpose,
            tensor(N, D, shape(Shape), _L),
            tensor(N, D, shape(TransShape), strided)) :-
    append(Prefix, [A, B], Shape),
    append(Prefix, [B, A], TransShape).

%% cont: make contiguous (copies data if strided)
derive_type(cont,
            tensor(N, D, shape(S), _L),
            tensor(N, D, shape(S), contiguous)).

%% view: take a subregion — REQUIRES contiguous input
derive_type(view(Region),
            tensor(N, D, shape(S), contiguous),
            tensor(N, D, shape(ViewShape), contiguous)) :-
    apply_region(S, Region, ViewShape).

%% view on non-contiguous → ERROR (caught by unification failure)
%% derive_type(view(_), tensor(_, _, _, strided), _) :- fail.

%% cast: change dtype, preserve shape and layout
derive_type(cast(NewDType),
            tensor(N, dtype(_OldD), shape(S), L),
            tensor(N, dtype(NewDType), shape(S), L)).

%% permute: reorder dimensions
derive_type(permute(Perm),
            tensor(N, D, shape(Shape), _L),
            tensor(N, D, shape(PermShape), strided)) :-
    permute_list(Shape, Perm, PermShape).

%% mul_mat: matrix multiply — C = A · B
%% A: [K, M], B: [K, N] → C: [M, N] (ggml convention: src0=A, src1=B)
derive_type(mul_mat,
            [tensor(_, dtype(DA), shape([K, M|RestA]), _),
             tensor(_, dtype(_DB), shape([K, N|RestB]), _)],
            tensor(result, dtype(f32), shape([M, N|RestOut]), contiguous)) :-
    (   DA = f32 ; DA = f16  %% output is always f32
    ),
    %% Batch dims must match
    (   RestA = RestB -> RestOut = RestA
    ;   RestOut = []
    ).

%% rope: in-place rotation, preserves everything
derive_type(rope(_Mode, _Theta),
            tensor(N, D, shape(S), L),
            tensor(N, D, shape(S), L)).

%% soft_max: in-place, preserves type/shape
derive_type(soft_max,
            tensor(N, D, shape(S), L),
            tensor(N, D, shape(S), L)).

%% add: element-wise, preserves type/shape (broadcast rules apply)
derive_type(add,
            [tensor(_, D, shape(S), _), tensor(_, D, shape(S), _)],
            tensor(result, D, shape(S), contiguous)).

%% rms_norm: reduces last dim, preserves others
derive_type(rms_norm,
            tensor(N, D, shape(S), _L),
            tensor(N, D, shape(S), contiguous)).

%% silu: element-wise activation
derive_type(silu,
            tensor(N, D, shape(S), L),
            tensor(N, D, shape(S), L)).

%% ============================================================
%% Pipeline verification — check a sequence of ops
%% ============================================================

verify_pipeline([], T, T).
verify_pipeline([Op|Ops], Tin, Tout) :-
    (   derive_type(Op, Tin, Tmid)
    ->  verify_pipeline(Ops, Tmid, Tout)
    ;   format(atom(Msg), "Type error: ~w cannot accept ~w", [Op, Tin]),
        throw(type_error(Op, Tin, Msg))
    ).

%% check_pipeline/2 — verify and report, don't throw
check_pipeline(Ops, Tin) :-
    (   verify_pipeline(Ops, Tin, Tout)
    ->  format("Pipeline OK: ~w → ~w~n", [Tin, Tout])
    ;   format("Pipeline FAILED at some op~n")
    ).

%% ============================================================
%% Bandwidth computation
%% ============================================================

%% bandwidth_class: how much data does each op type move?
%%   metadata_only: 0 bytes (reshape, view — just pointer arithmetic)
%%   full_copy: read all + write all (cont, cast)
%%   in_place: read + write same data (rope, soft_max, silu)
%%   compute: read inputs + write output (mul_mat)

bandwidth_class(reshape(_), metadata_only).
bandwidth_class(reshape, metadata_only).
bandwidth_class(view(_), metadata_only).
bandwidth_class(view, metadata_only).
bandwidth_class(permute(_), metadata_only).
bandwidth_class(permute, metadata_only).
bandwidth_class(transpose(_), metadata_only).
bandwidth_class(transpose, metadata_only).
bandwidth_class(cont, full_copy).
bandwidth_class(cast(_), full_copy).
bandwidth_class(cpy, full_copy).
bandwidth_class(rope(_, _), in_place).
bandwidth_class(rope, in_place).
bandwidth_class(soft_max, in_place).
bandwidth_class(silu, in_place).
bandwidth_class(rms_norm, in_place).
bandwidth_class(mul, in_place).
bandwidth_class(add, compute).
bandwidth_class(mul_mat, compute).
bandwidth_class(none, metadata_only).

%% op_bandwidth(Op, TensorIn, TensorOut, Read, Write)
op_bandwidth(Op, Tin, Tout, read(R), write(W)) :-
    bandwidth_class(Op, Class),
    bandwidth_from_class(Class, Tin, Tout, R, W).

bandwidth_from_class(metadata_only, _, _, 0, 0).

bandwidth_from_class(full_copy, Tin, Tout, R, W) :-
    tensor_bytes(Tin, R),
    tensor_bytes(Tout, W).

bandwidth_from_class(in_place, Tin, _, R, W) :-
    tensor_bytes(Tin, R),
    W = R.  %% read and write same data

bandwidth_from_class(compute, _Tin, Tout, R, W) :-
    %% For mul_mat: read = src0_bytes + src1_bytes, write = dst_bytes
    %% Simplified: just report output bytes as write
    tensor_bytes(Tout, W),
    R = W.  %% placeholder — real compute bandwidth needs both inputs

%% ============================================================
%% Human-readable descriptions (template-driven, never stored prose)
%% ============================================================

describe_op(reshape(Shape), Desc) :-
    format(atom(Desc), "reshape to ~w", [Shape]).
describe_op(transpose, "transpose").
describe_op(transpose(Perm), Desc) :-
    format(atom(Desc), "transpose ~w", [Perm]).
describe_op(cont, "contiguous").
describe_op(view(Region), Desc) :-
    format(atom(Desc), "view [~w]", [Region]).
describe_op(cast(DType), Desc) :-
    format(atom(Desc), "cast to ~w", [DType]).
describe_op(permute(Perm), Desc) :-
    format(atom(Desc), "permute ~w", [Perm]).
describe_op(mul_mat, "matmul").
describe_op(rope(Mode, theta(T)), Desc) :-
    format(atom(Desc), "RoPE (~w, θ=~w)", [Mode, T]).
describe_op(soft_max, "softmax").
describe_op(silu, "SiLU").
describe_op(rms_norm, "RMSNorm").
describe_op(add, "add").

%% Role-level descriptions (from op_role terms)
describe_role(projection(Q, W), Desc) :-
    format(atom(Desc), "~w projection (x·~w)", [Q, W]).
describe_role(rope(Mode, _), Desc) :-
    format(atom(Desc), "RoPE on ~w", [Mode]).
describe_role(attention_scores, "QK^T (scores)").
describe_role(attention_softmax, "softmax(scores)").
describe_role(v_sum, "V-sum (weights·V)").
describe_role(cache_store(Which, From, To), Desc) :-
    format(atom(Desc), "~w(~w) → ~w(~w_cache)", [From, Which, To, Which]).

%% ============================================================
%% Helpers
%% ============================================================

permute_list(List, Perm, Permuted) :-
    maplist(nth0_of(List), Perm, Permuted).

nth0_of(List, Idx, Elem) :- nth0(Idx, List, Elem).

apply_region(Shape, Region, ViewShape) :-
    maplist(apply_one_region, Shape, Region, ViewShape).

apply_one_region(_Dim, Start:End, Size) :-
    (   integer(Start), integer(End) -> Size is End - Start
    ;   Size = End - Start   %% symbolic
    ).
apply_one_region(Dim, all, Dim).
