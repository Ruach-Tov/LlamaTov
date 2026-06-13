%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════
%% compute_graph_invariants.pl — Self-diagnosing compute graph verification
%%
%% Catches structural bugs in compute graphs BEFORE dispatch.
%% Each invariant is a Prolog rule that either succeeds (valid)
%% or fails with a diagnostic explaining what is wrong and why.
%%
%% Licensed under GPLv2
%% ═══════════════════════════════════════════════════════════════════════

/** <module> Compute Graph Invariant Checker

Verifies structural properties of compute graphs that must hold
for correct execution. Catches bugs at graph construction time,
before any kernel runs.

Key invariant classes:
  - View coherence: views into base tensors have valid offsets and strides
  - Write-read coherence: data written to a cache is reachable by readers
  - Temporal ordering: writes happen before reads in the evaluation order
  - Shape consistency: connected operations have compatible dimensions
  - Stride alignment: view offsets are element-aligned

Example:
==
?- assert_tensor(k_cache_full, base, 32768, f16, [512,64,1,1], [2,1024,128,1], 0),
   assert_tensor(k_cache_view, view(k_cache_full), 3072, f16, [64,8,6,1], [2,128,1024,1], 6144),
   check_view_containment(k_cache_view, Result).

Result = ok  %% or error(view_exceeds_base, Details)
==

@author Ruach Tov Collective
@see lib/cupti_profile.pl for performance invariants
*/

:- module(compute_graph_invariants, [
    assert_tensor/7,
    assert_op/4,
    check_all_invariants/1,
    check_view_containment/2,
    check_view_alignment/2,
    check_stride_consistency/2,
    check_write_read_coherence/3,
    check_temporal_ordering/3,
    check_shape_compatibility/3,
    diagnose_graph/1,
    clear_graph/0,
    check_dtype_coherence/3,
    check_dtype_flow/3,
    check_dtype_chain/2,
    trace_dtype_chain/2,
    check_scale_coherence/3,
    assert_scale_convention/2,
    scale_application_path/2,
    scale_matches_oracle/2
]).

%% ═══════════════════════════════════════════════════════════════════════
%% Dynamic facts: the compute graph
%% ═══════════════════════════════════════════════════════════════════════

:- dynamic tensor/7.
%% tensor(Name, Kind, NumElements, Dtype, Ne, Nb, ByteOffset)
%%   Kind = base | view(BaseName)
%%   Ne = [ne0, ne1, ne2, ne3]  (dimensions)
%%   Nb = [nb0, nb1, nb2, nb3]  (byte strides)
%%   ByteOffset = offset into base tensor (0 for base tensors)

:- dynamic op/4.
%% op(Name, OpType, Inputs, Outputs)
%%   OpType = cpy | matmul | attention | rope | norm | ...
%%   Inputs = [TensorName, ...]
%%   Outputs = [TensorName, ...]

:- dynamic eval_order/2.
%% eval_order(OpName, Index)
%%   Index = position in evaluation sequence (0-based)

%! assert_tensor(+Name, +Kind, +NumElements, +Dtype, +Ne, +Nb, +ByteOffset) is det.
assert_tensor(Name, Kind, NumElements, Dtype, Ne, Nb, ByteOffset) :-
    retractall(tensor(Name, _, _, _, _, _, _)),
    assertz(tensor(Name, Kind, NumElements, Dtype, Ne, Nb, ByteOffset)).

%! assert_op(+Name, +OpType, +Inputs, +Outputs) is det.
assert_op(Name, OpType, Inputs, Outputs) :-
    retractall(op(Name, _, _, _)),
    assertz(op(Name, OpType, Inputs, Outputs)).

%! clear_graph/0 is det.
clear_graph :-
    retractall(tensor(_, _, _, _, _, _, _)),
    retractall(op(_, _, _, _)),
    retractall(eval_order(_, _)).

%% ═══════════════════════════════════════════════════════════════════════
%% Invariant 1: View containment
%% A view tensor must fit entirely within its base tensor.
%% ═══════════════════════════════════════════════════════════════════════

%! check_view_containment(+ViewName, -Result) is det.
check_view_containment(ViewName, Result) :-
    tensor(ViewName, view(BaseName), VElems, Dtype, _Ne, _Nb, Offset),
    !,
    tensor(BaseName, base, _BElems, Dtype, _BNe, _BNb, _),
    dtype_bytes(Dtype, ElemSize),
    ViewBytes is VElems * ElemSize,
    tensor(BaseName, base, BElems, _, _, _, _),
    BaseBytes is BElems * ElemSize,
    EndByte is Offset + ViewBytes,
    (EndByte =< BaseBytes ->
        Result = ok
    ;
        format(atom(Msg),
            'View ~w exceeds base ~w: offset=~w + size=~w = ~w > base_size=~w',
            [ViewName, BaseName, Offset, ViewBytes, EndByte, BaseBytes]),
        Result = error(view_exceeds_base, Msg)
    ).
check_view_containment(Name, ok) :-
    tensor(Name, base, _, _, _, _, _), !.
check_view_containment(Name, error(not_found, Name)).

%% ═══════════════════════════════════════════════════════════════════════
%% Invariant 2: View alignment
%% View offset must be aligned to element size.
%% ═══════════════════════════════════════════════════════════════════════

%! check_view_alignment(+ViewName, -Result) is det.
check_view_alignment(ViewName, Result) :-
    tensor(ViewName, view(_), _, Dtype, _, [Nb0|_], Offset),
    !,
    dtype_bytes(Dtype, ElemSize),
    (Offset mod ElemSize =:= 0 ->
        (Offset mod Nb0 =:= 0 ->
            Result = ok
        ;
            format(atom(Msg),
                'View ~w offset ~w not aligned to stride nb0=~w',
                [ViewName, Offset, Nb0]),
            Result = error(stride_misalignment, Msg)
        )
    ;
        format(atom(Msg),
            'View ~w offset ~w not aligned to element size ~w bytes',
            [ViewName, Offset, ElemSize]),
        Result = error(element_misalignment, Msg)
    ).
check_view_alignment(Name, ok) :-
    tensor(Name, base, _, _, _, _, _), !.

%% ═══════════════════════════════════════════════════════════════════════
%% Invariant 3: Stride consistency
%% View strides must be consistent with base tensor strides,
%% unless the view is an explicit permutation.
%% ═══════════════════════════════════════════════════════════════════════

%! check_stride_consistency(+ViewName, -Result) is det.
check_stride_consistency(ViewName, Result) :-
    tensor(ViewName, view(BaseName), _, _, _VNe, VNb, _),
    tensor(BaseName, base, _, _, _BNe, BNb, _),
    !,
    (VNb = BNb ->
        Result = ok
    ;
        %% Strides differ — check if it's a valid permutation
        (is_permutation_of_strides(VNb, BNb) ->
            Result = ok  %% permuted view is valid
        ;
            format(atom(Msg),
                'View ~w strides ~w differ from base ~w strides ~w (not a permutation)',
                [ViewName, VNb, BaseName, BNb]),
            Result = warning(stride_mismatch, Msg)
        )
    ).
check_stride_consistency(Name, ok) :-
    tensor(Name, base, _, _, _, _, _), !.

%% Check if one stride list is a permutation of another
is_permutation_of_strides(A, B) :-
    msort(A, Sorted),
    msort(B, Sorted).

%% ═══════════════════════════════════════════════════════════════════════
%% Invariant 4: Write-read coherence
%% Data written by a CPY must be reachable by attention READ.
%% ═══════════════════════════════════════════════════════════════════════

%! check_write_read_coherence(+WriteOp, +ReadOp, -Result) is det.
check_write_read_coherence(WriteOp, ReadOp, Result) :-
    op(WriteOp, cpy, _, [WriteTensor]),
    op(ReadOp, attention, ReadInputs, _),
    member(ReadTensor, ReadInputs),
    tensor(WriteTensor, view(WriteBase), _, _, _, _, WriteOffset),
    tensor(ReadTensor, view(ReadBase), _, _, _, _, ReadOffset),
    !,
    (WriteBase = ReadBase ->
        %% Same base tensor — check offset overlap
        tensor(WriteTensor, _, WElems, Dtype, _, _, _),
        dtype_bytes(Dtype, ElemSize),
        WriteEnd is WriteOffset + WElems * ElemSize,
        tensor(ReadTensor, _, RElems, _, _, _, _),
        ReadEnd is ReadOffset + RElems * ElemSize,
        (WriteOffset >= ReadOffset, WriteOffset < ReadEnd ->
            Result = ok  %% write region overlaps read region
        ; WriteEnd > ReadOffset, WriteEnd =< ReadEnd ->
            Result = ok
        ; ReadOffset >= WriteOffset, ReadOffset < WriteEnd ->
            Result = ok
        ;
            format(atom(Msg),
                'Write ~w [~w..~w] does not overlap read ~w [~w..~w] in base ~w',
                [WriteOp, WriteOffset, WriteEnd, ReadOp, ReadOffset, ReadEnd, WriteBase]),
            Result = error(write_read_disjoint, Msg)
        )
    ;
        format(atom(Msg),
            'Write ~w (base=~w) and read ~w (base=~w) use DIFFERENT base tensors',
            [WriteOp, WriteBase, ReadOp, ReadBase]),
        Result = error(different_base_tensors, Msg)
    ).
check_write_read_coherence(_, _, ok).  %% non-view tensors are trivially coherent

%% ═══════════════════════════════════════════════════════════════════════
%% Invariant 5: Temporal ordering
%% Writes must precede reads in evaluation order.
%% ═══════════════════════════════════════════════════════════════════════

%! check_temporal_ordering(+WriteOp, +ReadOp, -Result) is det.
check_temporal_ordering(WriteOp, ReadOp, Result) :-
    eval_order(WriteOp, WIdx),
    eval_order(ReadOp, RIdx),
    !,
    (WIdx < RIdx ->
        Result = ok
    ;
        format(atom(Msg),
            'Write ~w (eval_order=~w) does NOT precede read ~w (eval_order=~w)',
            [WriteOp, WIdx, ReadOp, RIdx]),
        Result = error(temporal_violation, Msg)
    ).
check_temporal_ordering(_, _, warning(no_eval_order, 'Eval order not specified')).

%% ═══════════════════════════════════════════════════════════════════════
%% Invariant 6: Shape compatibility
%% Connected operations must have compatible dimensions.
%% ═══════════════════════════════════════════════════════════════════════

%! check_shape_compatibility(+Op1, +Op2, -Result) is det.
check_shape_compatibility(Op1, Op2, Result) :-
    op(Op1, _, _, [OutputTensor|_]),
    op(Op2, _, [InputTensor|_], _),
    tensor(OutputTensor, _, _, _, OutNe, _, _),
    tensor(InputTensor, _, _, _, InNe, _, _),
    !,
    (OutNe = InNe ->
        Result = ok
    ;
        %% Check if shapes are broadcast-compatible or matmul-compatible
        (shapes_compatible(OutNe, InNe) ->
            Result = ok
        ;
            format(atom(Msg),
                'Shape mismatch: ~w output ~w has ne=~w, ~w input ~w has ne=~w',
                [Op1, OutputTensor, OutNe, Op2, InputTensor, InNe]),
            Result = error(shape_mismatch, Msg)
        )
    ).
check_shape_compatibility(_, _, ok).

shapes_compatible([N|_], [N|_]) :- !.  %% leading dimension matches
shapes_compatible([1|Rest1], [_|Rest2]) :- shapes_compatible(Rest1, Rest2).
shapes_compatible([_|Rest1], [1|Rest2]) :- shapes_compatible(Rest1, Rest2).
shapes_compatible([], _).
shapes_compatible(_, []).

%% ═══════════════════════════════════════════════════════════════════════
%% Master checker: run ALL invariants and collect diagnostics
%% ═══════════════════════════════════════════════════════════════════════

%! check_all_invariants(-Diagnostics) is det.
check_all_invariants(Diagnostics) :-
    findall(Diag, check_one_invariant(Diag), Diagnostics).

check_one_invariant(diag(view_containment, Name, Result)) :-
    tensor(Name, view(_), _, _, _, _, _),
    check_view_containment(Name, Result),
    Result \= ok.

check_one_invariant(diag(view_alignment, Name, Result)) :-
    tensor(Name, view(_), _, _, _, _, _),
    check_view_alignment(Name, Result),
    Result \= ok.

check_one_invariant(diag(stride_consistency, Name, Result)) :-
    tensor(Name, view(_), _, _, _, _, _),
    check_stride_consistency(Name, Result),
    Result \= ok.

check_one_invariant(diag(write_read, WriteOp-ReadOp, Result)) :-
    op(WriteOp, cpy, _, _),
    op(ReadOp, attention, _, _),
    check_write_read_coherence(WriteOp, ReadOp, Result),
    Result \= ok.

check_one_invariant(diag(temporal, WriteOp-ReadOp, Result)) :-
    op(WriteOp, cpy, _, _),
    op(ReadOp, attention, _, _),
    check_temporal_ordering(WriteOp, ReadOp, Result),
    Result \= ok.

%% ═══════════════════════════════════════════════════════════════════════
%% Self-diagnosis: human-readable report
%% ═══════════════════════════════════════════════════════════════════════

%! diagnose_graph(-Report) is det.
diagnose_graph(Report) :-
    check_all_invariants(Diagnostics),
    length(Diagnostics, N),
    (N =:= 0 ->
        Report = 'All invariants satisfied. Graph is structurally valid.'
    ;
        findall(Line,
            (member(diag(Class, Subject, Result), Diagnostics),
             format(atom(Line), '  ~w [~w]: ~w', [Class, Subject, Result])),
            Lines),
        atomic_list_concat(['Graph has structural issues:'|Lines], '\n', Report)
    ).

%% ═══════════════════════════════════════════════════════════════════════
%% Utility: dtype sizes
%% ═══════════════════════════════════════════════════════════════════════

dtype_bytes(f32, 4).
dtype_bytes(f16, 2).
dtype_bytes(i32, 4).
dtype_bytes(i16, 2).
dtype_bytes(i8, 1).
dtype_bytes(q8_0, 34).  %% 32 int8 quants + 1 f16 scale = 34 bytes per block
dtype_bytes(q4_k, 144). %% K-quant block size

%% ═══════════════════════════════════════════════════════════════════════
%% Invariant 7: Dtype coherence
%% Connected tensors must have compatible dtypes.
%% An op that outputs f32 feeding into an op expecting f16 is a bug
%% UNLESS there is an explicit cast/convert operation between them.
%% ═══════════════════════════════════════════════════════════════════════

%! check_dtype_coherence(+Op1, +Op2, -Result) is det.
%  Verify that the output dtype of Op1 matches the expected input dtype of Op2.
check_dtype_coherence(Op1, Op2, Result) :-
    op(Op1, Op1Type, _, Outputs),
    op(Op2, Op2Type, Inputs, _),
    member(OutTensor, Outputs),
    member(InTensor, Inputs),
    OutTensor = InTensor,  %% same tensor connects them
    tensor(OutTensor, _, _, OutDtype, _, _, _),
    !,
    %% Check if Op2 has an expected input dtype
    (expected_input_dtype(Op2Type, ExpectedDtype) ->
        (OutDtype = ExpectedDtype ->
            Result = ok
        ;
            format(atom(Msg),
                'Dtype mismatch: ~w outputs ~w as ~w, but ~w (~w) expects ~w',
                [Op1, OutTensor, OutDtype, Op2, Op2Type, ExpectedDtype]),
            Result = error(dtype_mismatch, Msg)
        )
    ;
        Result = ok  %% no expected dtype constraint
    ).
check_dtype_coherence(_, _, ok).

%% For ops that connect via DIFFERENT tensor names (e.g., output feeds input
%% through a view or reshape), check by eval order adjacency
check_dtype_flow(Op1, Op2, Result) :-
    op(Op1, _, _, [OutTensor|_]),
    op(Op2, _, [InTensor|_], _),
    tensor(OutTensor, _, _, OutDtype, _, _, _),
    tensor(InTensor, _, _, InDtype, _, _, _),
    !,
    (OutDtype = InDtype ->
        Result = ok
    ;
        (is_explicit_cast(Op1, OutDtype, InDtype) ->
            Result = ok
        ; is_explicit_cast(Op2, OutDtype, InDtype) ->
            Result = ok
        ;
            format(atom(Msg),
                'Dtype flow error: ~w produces ~w (~w) but ~w consumes ~w (~w). Missing cast?',
                [Op1, OutTensor, OutDtype, Op2, InTensor, InDtype]),
            Result = error(dtype_flow_mismatch, Msg)
        )
    ).
check_dtype_flow(_, _, ok).

%% Known explicit cast operations
is_explicit_cast(Op, _From, _To) :-
    op(Op, cast, _, _).
is_explicit_cast(Op, _From, _To) :-
    op(Op, cpy, _, _).  %% CPY can convert dtypes in ggml
is_explicit_cast(Op, _From, _To) :-
    op(Op, convert, _, _).

%% Expected input dtypes for specific operations
%% NOTE: attention accepts MIXED dtypes (Q=f32, K/V=f16).
%% We only flag when ALL inputs are wrong, not mixed cases.
%% Use check_attention_dtypes/2 for the mixed-dtype check.
expected_input_dtype(rope, f32).           %% RoPE operates on f32
expected_input_dtype(norm, f32).           %% RMSNorm operates on f32
expected_input_dtype(softmax, f32).        %% Softmax needs f32 precision

%% ═══════════════════════════════════════════════════════════════════════
%% Invariant 8: Dtype chain tracing
%% Trace the dtype through a chain of operations and flag where
%% an unexpected dtype transition occurs without an explicit cast.
%% ═══════════════════════════════════════════════════════════════════════

%! trace_dtype_chain(+StartTensor, -Chain) is det.
%  Follow a tensor through operations, recording dtype at each step.
trace_dtype_chain(StartTensor, Chain) :-
    tensor(StartTensor, _, _, StartDtype, _, _, _),
    trace_dtype_chain_(StartTensor, StartDtype, [], Chain).

trace_dtype_chain_(Tensor, Dtype, Acc, Chain) :-
    %% Find an op that consumes this tensor
    op(OpName, OpType, Inputs, Outputs),
    member(Tensor, Inputs),
    Outputs = [OutTensor|_],
    tensor(OutTensor, _, _, OutDtype, _, _, _),
    Step = step(Tensor, Dtype, OpName, OpType, OutTensor, OutDtype),
    \+ member(Step, Acc),  %% prevent cycles
    !,
    trace_dtype_chain_(OutTensor, OutDtype, [Step|Acc], Chain).
trace_dtype_chain_(_, _, Acc, Chain) :-
    reverse(Acc, Chain).

%! check_dtype_chain(+StartTensor, -Violations) is det.
%  Trace dtype chain and find all implicit dtype transitions.
check_dtype_chain(StartTensor, Violations) :-
    trace_dtype_chain(StartTensor, Chain),
    findall(
        violation(Op, OpType, InDtype, OutDtype),
        (member(step(_, InDtype, Op, OpType, _, OutDtype), Chain),
         InDtype \= OutDtype,
         \+ is_explicit_cast(Op, InDtype, OutDtype)),
        Violations).

%% ═══════════════════════════════════════════════════════════════════════
%% Extended master checker: include dtype invariants
%% ═══════════════════════════════════════════════════════════════════════

check_one_invariant(diag(dtype_coherence, Op1-Op2, Result)) :-
    op(Op1, _, _, Outputs),
    op(Op2, _, Inputs, _),
    member(T, Outputs),
    member(T, Inputs),
    check_dtype_coherence(Op1, Op2, Result),
    Result \= ok.

check_one_invariant(diag(dtype_flow, Op1-Op2, Result)) :-
    eval_order(Op1, Idx1),
    eval_order(Op2, Idx2),
    Idx2 =:= Idx1 + 1,
    check_dtype_flow(Op1, Op2, Result),
    Result \= ok.

%% ═══════════════════════════════════════════════════════════════════════
%% Invariant 9: Scale coherence
%% When a matmul output feeds into softmax (attention QK^T path),
%% verify that the scaling convention matches the expected convention.
%%
%% Two valid conventions:
%%   pre_scaled:  Q' = Q / sqrt(d), QK^T = Q' · K^T (scale fused into matmul)
%%   post_scaled: QK^T = Q · K^T, then S = QK^T / sqrt(d) (scale applied after)
%%
%% The convention must be CONSISTENT: if the matmul produces pre-scaled
%% output, the softmax must expect pre-scaled input (no separate scale op).
%% If the matmul produces raw output, there must be a scale op before softmax.
%% ═══════════════════════════════════════════════════════════════════════

:- dynamic scale_convention/2.
%% scale_convention(OpName, pre_scaled | post_scaled | raw)

%! assert_scale_convention(+OpName, +Convention) is det.
assert_scale_convention(OpName, Convention) :-
    retractall(scale_convention(OpName, _)),
    assertz(scale_convention(OpName, Convention)).

%! check_scale_coherence(+MatmulOp, +SoftmaxOp, -Result) is det.
%  Verify that the scaling convention between matmul and softmax is consistent.
check_scale_coherence(MatmulOp, SoftmaxOp, Result) :-
    op(MatmulOp, _, _, [MatmulOut|_]),
    op(SoftmaxOp, softmax, [SoftmaxIn|_], _),
    !,
    %% Check if there's a scale op between matmul and softmax
    (MatmulOut = SoftmaxIn ->
        %% Direct connection: matmul → softmax (no intermediate scale)
        %% This means matmul MUST be producing pre-scaled output,
        %% OR the scale is missing (bug)
        (scale_convention(MatmulOp, pre_scaled) ->
            Result = ok  %% Explicitly marked as pre-scaled
        ; scale_convention(MatmulOp, raw) ->
            format(atom(Msg),
                'Scale missing: ~w produces raw QK^T but feeds directly to ~w (softmax). Need scale op between them, or declare pre_scaled.',
                [MatmulOp, SoftmaxOp]),
            Result = error(scale_missing, Msg)
        ;
            %% No convention declared — warn
            format(atom(Msg),
                'Scale convention unknown for ~w → ~w. Declare scale_convention(~w, pre_scaled|post_scaled|raw).',
                [MatmulOp, SoftmaxOp, MatmulOp]),
            Result = warning(scale_undeclared, Msg)
        )
    ;
        %% Not direct — check if there's a scale op in between
        (op(ScaleOp, scale, [MatmulOut], [SoftmaxIn]) ->
            Result = ok  %% Explicit scale op between them
        ; op(ScaleOp, mul, [MatmulOut|_], [SoftmaxIn]) ->
            Result = ok  %% Multiplication used as scale
        ;
            %% Different tensors, no scale op found
            %% This might be fine if matmul is pre-scaled
            (scale_convention(MatmulOp, pre_scaled) ->
                Result = ok
            ;
                format(atom(Msg),
                    'Cannot verify scale path from ~w to ~w: different tensors (~w vs ~w) with no scale op between.',
                    [MatmulOp, SoftmaxOp, MatmulOut, SoftmaxIn]),
                Result = warning(scale_path_unclear, Msg)
            )
        )
    ).
check_scale_coherence(_, _, ok).

%% Add to the master checker
check_one_invariant(diag(scale_coherence, MatmulOp-SoftmaxOp, Result)) :-
    op(MatmulOp, matmul, _, _),
    op(SoftmaxOp, softmax, _, _),
    check_scale_coherence(MatmulOp, SoftmaxOp, Result),
    Result \= ok.

%% Also detect when QK matmul feeds softmax with no declared convention
check_one_invariant(diag(scale_coherence, MatmulOp-SoftmaxOp, Result)) :-
    op(MatmulOp, attention, _, [Out|_]),
    op(SoftmaxOp, softmax, [Out|_], _),
    \+ scale_convention(MatmulOp, _),
    format(atom(Msg),
        'Attention ~w feeds softmax ~w but scale_convention not declared. Is QK^T pre-scaled or raw?',
        [MatmulOp, SoftmaxOp]),
    Result = warning(scale_undeclared, Msg).

%% ═══════════════════════════════════════════════════════════════════════
%% Sweepable parameter: scale_application_path
%% ═══════════════════════════════════════════════════════════════════════

%! scale_application_path(+Convention, -Description) is det.
%  Both conventions are mathematically correct. They produce different bits.
%  The substrate sweeps both; the verification gate picks the matching one.

scale_application_path(pre_scaled,
    'Fuse 1/sqrt(d) into Q before matmul: Q\' = Q/sqrt(d), QK^T = Q\' * K^T. Fewer ops.').
scale_application_path(post_scaled,
    'Raw QK^T then scale: S = (Q * K^T) / sqrt(d). Matches ggml convention.').

%! scale_matches_oracle(+Convention, +Oracle) is det.
%  Check if the chosen convention matches the oracle (ggml, PyTorch, etc.)
scale_matches_oracle(post_scaled, ggml).      %% ggml uses post-scaled
scale_matches_oracle(pre_scaled, pytorch).     %% PyTorch FlashAttention uses pre-scaled
scale_matches_oracle(post_scaled, cudasw4).    %% CUDASW4 uses post-scaled (N/A but pattern)

%% ═══════════════════════════════════════════════════════════════════════
%% Invariant 10: Unique tensor naming
%% Ambiguous tensor names cause source-linking errors in verifiers.
%% ═══════════════════════════════════════════════════════════════════════

%! check_unique_names(-Violations) is det.
%  Find tensor names that appear more than once with different indices.
check_unique_names(Violations) :-
    findall(
        ambiguous(Name, Indices),
        (   tensor(Name, _, _, _, _, _, _),
            findall(Idx, 
                (tensor(Name, _, _, _, _, _, _), op(_, _, _, [Name|_]), true),
                Indices),
            length(Indices, N),
            N > 1
        ),
        Violations
    ).

check_one_invariant(diag(unique_names, Name, warning(ambiguous_name, Msg))) :-
    tensor(Name, _, _, _, _, _, _),
    findall(Name, tensor(Name, _, _, _, _, _, _), Matches),
    length(Matches, N),
    N > 1,
    format(atom(Msg),
        'Tensor name "~w" appears ~w times. Source-linking verifiers may pick the wrong one.',
        [Name, N]).

%% ═══════════════════════════════════════════════════════════════════════
%% Sweepable parameter: exp_implementation
%% ═══════════════════════════════════════════════════════════════════════

%! exp_implementation(+Method, -Description, -MaxError) is det.
%  Controls which exp() is used in softmax and activation functions.

exp_implementation(libm_expf,
    'Standard IEEE-754 correctly-rounded expf from libm. Honest.',
    '0.5 ULP (correctly rounded)').
exp_implementation(ggml_polynomial,
    'ARM-derived polynomial approximation used by ggml. Fast, 1.45 ULP max error. Matches ggml oracle.',
    '1.45 ULP + 0.5 ULP').
exp_implementation(sleef_polynomial,
    'SLEEF library polynomial. Used by PyTorch for GELU/SiLU.',
    '1.0 ULP').

%! exp_matches_oracle(+Method, +Oracle) is det.
exp_matches_oracle(ggml_polynomial, ggml).
exp_matches_oracle(sleef_polynomial, pytorch).
exp_matches_oracle(libm_expf, reference).
