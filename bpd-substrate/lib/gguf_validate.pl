%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% gguf_validate.pl — Pre-load GGUF validation with structured error reporting
%%
%% Validates a GGUF file BEFORE any tensor data is mapped to GPU.
%% Every incompatibility caught at parse time is a runtime CUDA error prevented.
%%
%% Uses safe_read.pl for byte-ownership tracking (crossword-puzzle defense)
%% and gguf_native_reader.pl for structural parsing.
%%
%% Usage:
%%   ?- gguf_validate('/path/to/model.gguf').           % prints results
%%   ?- gguf_validate('/path/to/model.gguf', Results).  % returns list

/** <module> GGUF Pre-Load Validation

Validates GGUF model files BEFORE tensor data is mapped to GPU.
Every incompatibility caught at parse time is a runtime CUDA error prevented.

Uses safe_read.pl for byte-ownership tracking (crossword-puzzle defense)
and gguf_native_reader.pl for structural parsing.

Five validation checks:
  1. tensor_overlap — detects overlapping tensor data regions
  2. quant_version — K-quant types require version >= 3
  3. type_consistency — no mixed quantization within a layer
  4. rope_params — required for llama/falcon/mistral/etc.
  5. known_types — all tensor types recognized by substrate

@author Ruach Tov Collective
@license RTAAL-1.0 for optimizer components; GPLv2 for this module
@see safe_read.pl for byte-ownership primitives
@see gguf_native_reader.pl for GGUF parsing
*/

:- module(gguf_validate, [
    gguf_validate/1,
    gguf_validate/2
]).

:- use_module('lib/safe_read').
:- use_module('lib/gguf_native_reader').

%% ═══════════════════════════════════════════════════════════════
%% Top-level validator
%% ═══════════════════════════════════════════════════════════════

%! gguf_validate(+Path) is det.
%  Validate a GGUF file and print results to stdout.
%  Runs all 5 structural checks. Prints PASS/FAIL for each.
gguf_validate(Path) :-
    gguf_validate(Path, Results),
    report_results(Path, Results).

%! gguf_validate(+Path, -Results) is det.
%  Validate a GGUF file and return structured results.
%  Results is a list of pass(Check) or fail(Check, Detail) terms.
gguf_validate(Path, Results) :-
    catch(
        gguf_validate_inner(Path, Results),
        Error,
        Results = [error(parse_failure, Error)]
    ).

gguf_validate_inner(Path, Results) :-
    gguf_read(Path, Header, Metadata, TensorInfos),
    findall(Result, (
        validate_check(Header, Metadata, TensorInfos, Result)
    ), Results).

%% ═══════════════════════════════════════════════════════════════
%% Validation checks — each produces pass(Check) or fail(Check, Detail)
%% ═══════════════════════════════════════════════════════════════

validate_check(_, _, _, Result) :- Result = pass(placeholder), fail.  % anchor for findall

%% CHECK 1: Tensor data regions must not overlap each other
validate_check(_, _, TensorInfos, Result) :-
    check_tensor_overlap(TensorInfos, Result).

%% CHECK 2: Quantization version must match tensor types
validate_check(_, Metadata, TensorInfos, Result) :-
    check_quant_version(Metadata, TensorInfos, Result).

%% CHECK 3: Tensor types should be consistent within logical layers
validate_check(_, _, TensorInfos, Result) :-
    check_type_consistency(TensorInfos, Result).

%% CHECK 4: Rope parameters present if architecture requires them
validate_check(_, Metadata, _, Result) :-
    check_rope_params(Metadata, Result).

%% CHECK 5: All tensor types are recognized
validate_check(_, _, TensorInfos, Result) :-
    check_known_types(TensorInfos, Result).

%% ═══════════════════════════════════════════════════════════════
%% CHECK 1: Tensor overlap detection
%% ═══════════════════════════════════════════════════════════════

check_tensor_overlap(TensorInfos, Result) :-
    findall(Name-Start-End, (
        member(tensor_info(Name, Dims, Type, Offset), TensorInfos),
        tensor_byte_size(Dims, Type, Size),
        Start = Offset,
        End is Offset + Size
    ), Regions),
    (find_overlap(Regions, NameA, NameB, OverlapStart, OverlapEnd)
     -> Result = fail(tensor_overlap,
            overlap(NameA, NameB, bytes(OverlapStart, OverlapEnd)))
     ;  Result = pass(tensor_overlap)
    ).

find_overlap(Regions, NameA, NameB) :-
    member(NameA-SA-EA, Regions),
    member(NameB-SB-EB, Regions),
    NameA @< NameB,  % avoid duplicate pairs
    SA < EB, EA > SB.  % overlap condition

find_overlap(Regions, NameA, NameB, OverlapStart, OverlapEnd) :-
    member(NameA-SA-EA, Regions),
    member(NameB-SB-EB, Regions),
    NameA @< NameB,
    SA < EB, EA > SB,
    OverlapStart is max(SA, SB),
    OverlapEnd is min(EA, EB).

%% ═══════════════════════════════════════════════════════════════
%% CHECK 2: Quantization version vs tensor types
%% ═══════════════════════════════════════════════════════════════

check_quant_version(Metadata, TensorInfos, Result) :-
    (member('general.quantization_version'-Version, Metadata)
     -> (has_kquant_tensors(TensorInfos)
         -> (Version >= 3
             -> Result = pass(quant_version)
             ;  Result = fail(quant_version,
                    version_mismatch(declared(Version), requires(3),
                        reason('K-quant tensor types (Q4_K, Q5_K, Q6_K) require quantization_version >= 3'))))
         ;  Result = pass(quant_version))
     ;  (has_kquant_tensors(TensorInfos)
         -> Result = fail(quant_version,
                missing_version(reason('K-quant tensors present but general.quantization_version absent')))
         ;  Result = pass(quant_version))
    ).

has_kquant_tensors(TensorInfos) :-
    member(tensor_info(_, _, Type, _), TensorInfos),
    kquant_type(Type).

%% K-quant type codes (from gguf spec)
kquant_type(12).  % Q4_K
kquant_type(13).  % Q5_K
kquant_type(14).  % Q6_K
kquant_type(15).  % Q8_K

%% ═══════════════════════════════════════════════════════════════
%% CHECK 3: Type consistency within layers
%% ═══════════════════════════════════════════════════════════════

check_type_consistency(TensorInfos, Result) :-
    findall(Layer-Types, (
        extract_layer_name(TensorInfos, Layer),
        findall(Type, (
            member(tensor_info(Name, _, Type, _), TensorInfos),
            atom_concat(Layer, _, Name)
        ), TypeList),
        sort(TypeList, Types),
        length(Types, Len),
        Len > 1
    ), Inconsistencies),
    (Inconsistencies = []
     -> Result = pass(type_consistency)
     ;  Result = fail(type_consistency, mixed_types(Inconsistencies))
    ).

extract_layer_name(TensorInfos, Layer) :-
    member(tensor_info(Name, _, _, _), TensorInfos),
    %% Extract layer prefix: everything up to .weight/.bias
    (sub_atom(Name, Before, _, _, '.weight')
     -> sub_atom(Name, 0, Before, _, Layer)
     ;  sub_atom(Name, Before, _, _, '.bias')
     -> sub_atom(Name, 0, Before, _, Layer)
     ;  fail
    ).

%% ═══════════════════════════════════════════════════════════════
%% CHECK 4: Rope parameters for architectures that need them
%% ═══════════════════════════════════════════════════════════════

check_rope_params(Metadata, Result) :-
    (member('general.architecture'-Arch, Metadata)
     -> (requires_rope(Arch)
         -> (has_rope_params(Metadata)
             -> Result = pass(rope_params)
             ;  Result = fail(rope_params,
                    missing_rope(architecture(Arch),
                        reason('Architecture requires rope parameters but none found in metadata'))))
         ;  Result = pass(rope_params))
     ;  Result = pass(rope_params)  % no architecture key — can't validate
    ).

requires_rope(llama).
requires_rope(falcon).
requires_rope(mistral).
requires_rope(qwen).
requires_rope(qwen2).
requires_rope(phi).
requires_rope(phi3).
requires_rope(gemma).
requires_rope(gemma2).
requires_rope(starcoder2).

has_rope_params(Metadata) :-
    (member('llama.rope.freq_base'-_, Metadata) ;
     member('rope.freq_base'-_, Metadata) ;
     member('llama.rope.dimension_count'-_, Metadata) ;
     member('rope.dimension_count'-_, Metadata)).

%% ═══════════════════════════════════════════════════════════════
%% CHECK 5: All tensor types recognized
%% ═══════════════════════════════════════════════════════════════

check_known_types(TensorInfos, Result) :-
    findall(Name-Type, (
        member(tensor_info(Name, _, Type, _), TensorInfos),
        \+ known_type(Type)
    ), Unknown),
    (Unknown = []
     -> Result = pass(known_types)
     ;  Result = fail(known_types, unknown_tensor_types(Unknown))
    ).

known_type(0).   % F32
known_type(1).   % F16
known_type(2).   % Q4_0
known_type(3).   % Q4_1
known_type(6).   % Q5_0
known_type(7).   % Q5_1
known_type(8).   % Q8_0
known_type(9).   % Q8_1
known_type(10).  % Q2_K
known_type(11).  % Q3_K
known_type(12).  % Q4_K
known_type(13).  % Q5_K
known_type(14).  % Q6_K
known_type(15).  % Q8_K
known_type(24).  % I8
known_type(25).  % I16
known_type(26).  % I32
known_type(27).  % I64
known_type(28).  % F64
known_type(30).  % BF16

%% ═══════════════════════════════════════════════════════════════
%% Helpers
%% ═══════════════════════════════════════════════════════════════

tensor_byte_size(Dims, Type, Size) :-
    foldl([D, Acc0, Acc1]>>(Acc1 is Acc0 * D), Dims, 1, NumElements),
    type_element_size(Type, ElemSize),
    Size is NumElements * ElemSize.

type_element_size(0, 4).   % F32
type_element_size(1, 2).   % F16
type_element_size(30, 2).  % BF16
type_element_size(24, 1).  % I8
type_element_size(25, 2).  % I16
type_element_size(26, 4).  % I32
type_element_size(27, 8).  % I64
type_element_size(28, 8).  % F64
type_element_size(_, 1).   % quantized types: approximate

%% ═══════════════════════════════════════════════════════════════
%% Result reporting
%% ═══════════════════════════════════════════════════════════════

report_results(Path, Results) :-
    format("~n╔══════════════════════════════════════════════════╗~n"),
    format("║  GGUF Pre-Load Validation                        ║~n"),
    format("╚══════════════════════════════════════════════════╝~n"),
    format("  File: ~w~n~n", [Path]),
    include(is_pass, Results, Passes),
    include(is_fail, Results, Fails),
    length(Passes, NPass),
    length(Fails, NFail),
    forall(member(R, Results), report_one(R)),
    format("~n  ══════════════════════════════════════~n"),
    format("  PASSED: ~d  FAILED: ~d~n", [NPass, NFail]),
    (NFail =:= 0
     -> format("  ✓ FILE IS SAFE TO MAP TO GPU~n")
     ;  format("  ✗ DO NOT MAP TO GPU — validation failures detected~n")
    ).

is_pass(pass(_)).
is_fail(fail(_, _)).
is_fail(error(_, _)).

report_one(pass(Check)) :-
    format("  ✓ ~w~n", [Check]).
report_one(fail(Check, Detail)) :-
    format("  ✗ ~w: ~w~n", [Check, Detail]).
report_one(error(Check, Detail)) :-
    format("  ✗ ~w: ~w~n", [Check, Detail]).
