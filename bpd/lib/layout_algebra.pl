%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% layout_algebra.pl — algebraic layout transforms over the ggml tensor model.
%% (Iyun, 2026-05-29, Heath's instinct: declare transpose/scatter/gather weaves AS ALGEBRA,
%%  not comments; then SOLVE kernel chains BACKWARD from desired output shape to required input.)
%%
%% Builds on compute_graph_invariants' tensor model: a tensor is t(Ne, Nb, Offset) where
%%   Ne = element dims  [d0,d1,d2,d3]   Nb = byte strides [s0,s1,s2,s3]   Offset = byte offset.
%% A LAYOUT TRANSFORM is a RELATION layout/3 over (Kind, TensorIn, TensorOut). Because they are
%% relations, running with TensorOut bound + TensorIn free DERIVES the input layout = the
%% backward solve (Prolog's natural mode). Composition = a chain; solving backward = weaving
%% inputs back through the chain from a desired output.

:- module(layout_algebra,
    [ layout/3, layout_chain/3, solve_backward/3, transpose_dims/4, is_contiguous/2,
      inverse/2, inverse_chain/2 ]).
:- use_module(library(lists)).

%% ── primitive layout transforms: layout(Kind, t(NeIn,NbIn,OffIn), t(NeOut,NbOut,OffOut)) ──

%% TRANSPOSE(I,J): swap two dims. Permutes BOTH Ne and Nb at positions I,J. Self-inverse.
%% This is the dequant case: we store (in,out), gguf-py (out,in) = transpose(0,1).
layout(transpose(I,J), t(NeIn,NbIn,Off), t(NeOut,NbOut,Off)) :-
    swap_at(NeIn, I, J, NeOut),
    swap_at(NbIn, I, J, NbOut).

%% PERMUTE(Perm): general dim permutation (Perm is a list of source indices).
layout(permute(Perm), t(NeIn,NbIn,Off), t(NeOut,NbOut,Off)) :-
    ( nonvar(NeIn)
    -> permute_by(Perm, NeIn, NeOut), permute_by(Perm, NbIn, NbOut)   % forward
    ;  inverse_perm(Perm, Inv),                                        % backward: derive IN from OUT
       permute_by(Inv, NeOut, NeIn), permute_by(Inv, NbOut, NbIn) ).

%% RESHAPE(NewNe): change dims, recompute CONTIGUOUS strides. Requires equal element count.
%% Reversible: with NeOut bound and NeIn free, you can reshape back (given NeIn).
layout(reshape(NewNe), t(NeIn,_NbIn,Off), t(NewNe,NbOut,Off)) :-
    prod(NeIn, P), prod(NewNe, P),            % element-count preserved (the constraint)
    contiguous_strides(NewNe, NbOut).

%% VIEW(Off2,Ne2,Nb2): a sub-tensor (offset/shape/stride into the same base). Carries through.
layout(view(Off2,Ne2,Nb2), t(_,_,_), t(Ne2,Nb2,Off2)).

%% GATHER(IndexCount): out[i] = in[idx[i]] — embedding lookup (get_rows), repeat, cpy.
%% (fusion_analyzer classifies ggml_get_rows/repeat/cpy as gather.) The output's leading dim
%% is the index count N; the trailing dims are the gathered row's shape. Element-count is
%% N * row_elements (NOT preserved from input — gather selects/replicates rows).
%% get_rows: weight [n_embd, vocab] gathered by N token ids -> [n_embd, N].
layout(gather(N), t([RowDim, _Vocab], _NbIn, Off), t([RowDim, N], NbOut, Off)) :-
    contiguous_strides([RowDim, N], NbOut).

%% ── algebraic LAWS (make backward-solving provably sound) ──
%% transpose is SELF-INVERSE: transpose(I,J) then transpose(I,J) = identity.
inverse(transpose(I,J), transpose(I,J)).
%% permute's inverse: Inv[P[i]] = i.
inverse(permute(P), permute(Inv)) :- inverse_perm(P, Inv).
%% reshape's inverse needs the original shape; view/gather are NOT generally invertible (lossy) —
%% no inverse/2 clause -> the algebra correctly reports them irreversible.
inverse_perm(P, Inv) :-
    findall(Pos-Idx, nth0(Idx, P, Pos), Pairs), keysort(Pairs, Sorted),
    findall(I, member(_-I, Sorted), Inv).

%% inverse chain: reverse the chain AND invert each step (for backward solving non-trivial chains).
inverse_chain([], []).
inverse_chain([K|Ks], InvChain) :- inverse(K, KI), inverse_chain(Ks, RestI), append(RestI, [KI], InvChain).

%% ── composition: apply a chain of transforms left-to-right ──
layout_chain([], T, T).
layout_chain([K|Ks], T0, T) :- layout(K, T0, T1), layout_chain(Ks, T1, T).

%% ── THE BACKWARD SOLVE: given desired OUTPUT and a transform chain, derive REQUIRED INPUT. ──
%% Run the chain with Out bound, In free. (transpose/permute are fully reversible; reshape needs
%% the input Ne to be known or derivable; view is a projection.) This is "start from output
%% shape, weave inputs backward."
solve_backward(Chain, TOut, TIn) :- layout_chain(Chain, TIn, TOut).

%% ── helpers ──
swap_at(L, I, J, R) :- nth0(I, L, Ei), nth0(J, L, Ej),
    set_at(L, I, Ej, L1), set_at(L1, J, Ei, R).
set_at(L, I, V, R) :- nth0(I, L, _, Rest), nth0(I, R, V, Rest).
permute_by(Perm, L, R) :- maplist([Idx,E]>>nth0(Idx,L,E), Perm, R).
prod([], 1). prod([H|T], P) :- prod(T, P0), P is H*P0.
contiguous_strides(Ne, Nb) :- reverse(Ne, RNe), cstr(RNe, 1, RNb), reverse(RNb, Nb).
cstr([], _, []). cstr([D|Ds], Acc, [Acc|Rest]) :- Acc1 is Acc*D, cstr(Ds, Acc1, Rest).
transpose_dims(I, J, t(Ne,Nb,O), t(Ne2,Nb2,O)) :- layout(transpose(I,J), t(Ne,Nb,O), t(Ne2,Nb2,O)).
is_contiguous(t(Ne,Nb,_), Yes) :- ( contiguous_strides(Ne, Nb) -> Yes = true ; Yes = false ).

%% ── the dequant transpose, CODIFIED (instance #1, grounded in real measurement) ──
%% Our dq4k returns logical (in,out)=(4096,32768) contiguous; gguf-py returns (out,in)=(32768,4096).
%% layout(transpose(0,1), ours, ref) relates them. Values verified 0-ULP identical (test_dequant_vs_gguf_py.py).
dequant_layout_relation(Ours, Ref) :-
    Ours = t([4096,32768], NbO, 0), contiguous_strides([4096,32768], NbO),
    layout(transpose(0,1), Ours, Ref).
