%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% tensor_schema.pl — SHARED fact-schema declarations for the tensor-op type system.
%%
%% This module is the single source of truth for the SHAPE of the fact base. Both the
%% spec-extractor (Iyun, bottom layer) and the type-derivation engine (mavchin,
%% tensor_types.pl, middle layer) import it so the relations are declared in exactly
%% one place.
%%
%% PROVENANCE PRINCIPLE (quality-critical): every fact must derive from one of three
%% sanctioned sources — the program SPECIFICATION (lifted ggml graph), the CI/regression
%% TEST infrastructure (conformance harness), or PROFILING/PERFORMANCE measurement
%% (CUPTI-from-prolog / NVML-via-ioctl). No fact is authored as free prose. In particular
%% description strings are PRODUCED from op_role/2 via describe/2 templates, never stored;
%% and semantic_dims/3 carries a provenance term so a layout_equiv verdict is auditable.

:- module(tensor_schema,
    [ tensor/4,            % tensor(Name, dtype(D), shape(Dims), layout(L))
      op/5,                % op(NodeId, Coord, OpType, in(Srcs), out(Dst))
      op_role/2,           % op_role(NodeId, RoleTerm)
      op_verdict/2,        % op_verdict(NodeId, measured(...))
      semantic_dims/3,     % semantic_dims(Name, Dims, Provenance)
      layout_relation/4,   % layout_relation(NodeId, OurLayout, GgmlLayout, EquivProof)
      describe/2           % describe(RoleTerm, EnglishAtom)
    ]).

%% ── The fact relations (populated by the spec-extractor + measurement harness) ──
:- discontiguous tensor/4.
:- discontiguous op/5.
:- discontiguous op_role/2.
:- discontiguous op_verdict/2.
:- discontiguous semantic_dims/3.
:- discontiguous layout_relation/4.

:- dynamic tensor/4, op/5, op_role/2, op_verdict/2, semantic_dims/3, layout_relation/4.

%% ── Schema documentation (the contract) ───────────────────────────────────────
%%
%% tensor(Name, dtype(D), shape(Dims), layout(L))
%%   Name   : atom (ggml tensor name, or 'optype@node' for unnamed)
%%   D      : f32 | f16 | q8_0 | i32 | ...
%%   Dims   : list of physical dimensions (trailing 1s trimmed)
%%   L      : contiguous | strided(Kind) | permuted(transpose) | permuted(axes)
%%
%% op(NodeId, Coord, OpType, in(Srcs), out(Dst))
%%   NodeId : atom node id, e.g. '0024'
%%   Coord  : the dashboard coordinate path, e.g. [mistral,layer(l),attn,qkv]
%%   OpType : STRUCTURAL term — mul_mat | rope | soft_max | cpy | cont | reshape |
%%            view | permute | transpose | add | mul | none | get_rows
%%   Srcs   : list of src(ProducerNode, TensorName)  (dataflow edges from lineage)
%%   Dst    : output tensor Name
%%
%% op_role(NodeId, RoleTerm)
%%   RoleTerm : structured semantics — projection(X,W) | rope(X) | reshape(T) |
%%              view(T) | transpose(T) | permute(T) | cont(T) | cast(From,To,Dst) |
%%              copy(Src,Dst) | matmul(T) | softmax(T) | elementwise(Op,T) | leaf(T)
%%
%% op_verdict(NodeId, measured(max_ulp(U), ndiff(N), max_abs(A), elems(E), ref(R), test(T)))
%%   The conformance result from the measurement harness. ref(R): ollama | huggingface.
%%   test(T): the harness invocation that produced it (provenance, REQUIRED).
%%
%% semantic_dims(Name, Dims, Provenance)
%%   Dims       : the LOGICAL dimensions the physical layout encodes, e.g.
%%                [n_kv, n_heads, head_dim]
%%   Provenance : measured(Perm, test(T))  — empirically discovered permutation, OR
%%                derived(graph, lineage)   — from the ggml ne[] + op lineage, OR
%%                lifted(source(File,Line)) — from our C index expression.
%%   A semantic_dims fact WITHOUT provenance is forbidden; absence => verdict unverified.
%%
%% layout_relation(NodeId, our_layout(L1, semantic(P1)), ggml_layout(L2, semantic(P2)),
%%                 equiv_proof(Basis, Note))
%%   Records WHY a byte-divergent layout op is still correct. Basis is the downstream
%%   measurement that proves logical equivalence, e.g. v_sum_0ulp.

%% ── describe/2: English produced from RoleTerm via FIXED templates ─────────────
%% These format specs are the ONLY place a description string lives. Same RoleTerm
%% functor => structurally identical phrasing everywhere.
describe(projection(X, W), Desc) :-
    upcase_atom(X, XU),
    format(atom(Desc), '~w projection (x\u00b7~w)', [XU, W]).
describe(rope(X), Desc) :-
    format(atom(Desc), 'RoPE on ~w', [X]).
describe(reshape(T), Desc) :-
    format(atom(Desc), 'Reshape ~w', [T]).
describe(view(T), Desc) :-
    format(atom(Desc), 'View into ~w', [T]).
describe(transpose(T), Desc) :-
    format(atom(Desc), 'Transpose ~w', [T]).
describe(permute(T), Desc) :-
    format(atom(Desc), 'Permute ~w', [T]).
describe(cont(T), Desc) :-
    format(atom(Desc), 'Contiguous ~w', [T]).
describe(cast(From, To, Dst), Desc) :-
    format(atom(Desc), '~w \u2192 ~w cast into ~w', [From, To, Dst]).
describe(copy(Src, Dst), Desc) :-
    format(atom(Desc), 'Copy ~w \u2192 ~w', [Src, Dst]).
describe(matmul(T), Desc) :-
    format(atom(Desc), 'Matmul \u2192 ~w', [T]).
describe(softmax(T), Desc) :-
    format(atom(Desc), 'Softmax(~w)', [T]).
describe(elementwise(Op, T), Desc) :-
    cap_atom(Op, OpC),
    format(atom(Desc), '~w \u2192 ~w', [OpC, T]).

%% cap_atom: capitalize the first letter of an atom (sentence-case prose helper)
cap_atom(A, C) :- sub_atom(A, 0, 1, _, F), upcase_atom(F, FU),
                  sub_atom(A, 1, _, 0, Rest), atom_concat(FU, Rest, C).
describe(leaf(T), Desc) :-
    format(atom(Desc), 'Leaf ~w', [T]).
%% off-page connectors (drill-down filmstrip) — derived from layer_coords adjacency
describe(continues(Next), Desc) :-
    format(atom(Desc), 'CONTINUES (~w) \u25b6', [Next]).
describe(previous(Prev), Desc) :-
    format(atom(Desc), '\u25c0 PREVIOUS (~w)', [Prev]).
