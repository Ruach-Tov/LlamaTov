#!/usr/bin/env swipl
%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
% gguf_dump_tensors.pl — Dump the tensor_info section of a GGUF file.
%
% V.D.G1 step 7: empirical verification that the new interior-repeated
% emission path works on real GGUF data. Reads the metadata K-V section
% to determine where it ends, then parses the tensor_info section
% (112 tensor info records for nomic-embed-text-v1.5).
%
% Each tensor_info_record has:
%   - name (length-prefixed string)
%   - n_dimensions (u32)
%   - dimensions (n_dimensions u64 values — the INTERIOR REPEATED ARRAY)
%   - type (u32 enum: F32, F16, Q4_0, ...)
%   - offset (u64 — tensor data offset)
%
% Usage:
%   ./gguf_dump_tensors.pl [path-to-file.gguf]

:- consult('ir/include/byte_readers.pl').
:- consult('ir/include/landmarks.pl').
:- consult('ir/include/repeated_records.pl').
:- consult('file-formats/gguf/output/gguf.pl').
:- consult('ir/include/prolog_reader_generator.pl').

default_gguf_file(
    '${OLLAMA_BLOBS:-~/.ollama/models/blobs}/\c
sha256-970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6').

read_file_partial(Path, MaxBytes, Str) :-
    open(Path, read, Stream, [type(binary)]),
    setup_call_cleanup(true,
        read_string(Stream, MaxBytes, Str),
        close(Stream)).

main :-
    default_gguf_file(P),
    main(P).

main(Path) :-
    format("~n=== GGUF tensor_info dump ===~n~n"),
    format("File: ~w~n~n", [Path]),

    % Read first 4MB — sufficient for header + metadata + tensor_info
    % even for moderately large models.
    read_file_partial(Path, 4_000_000, Bytes),
    string_length(Bytes, BytesLen),
    format("Loaded ~d bytes (header + metadata + tensor_info expected)~n~n",
           [BytesLen]),

    emit_section_reader(header),
    emit_section_reader(metadata_kv_pair),
    emit_section_reader(tensor_info_record),

    statistics(real_time, [T0|_]),
    with_parse_session(dump, (
        % Header
        read_section(header, Bytes, HeaderPairs),
        member(field(metadata_kv_count, KVCount), HeaderPairs),
        member(field(tensor_count, TensorCount), HeaderPairs),
        format("Header: tensor_count=~d  metadata_kv_count=~d~n~n",
               [TensorCount, KVCount]),

        % Walk through metadata pairs to find where they end
        bytes_slice_from(Bytes, 24, MetaBytes),
        walk_metadata_pairs(MetaBytes, 0, 1, KVCount, MetadataEnd),

        % tensor_info_section starts immediately after metadata
        TensorSectionStart is 24 + MetadataEnd,
        format("Metadata ends at file offset ~d~n", [MetadataEnd + 24]),
        format("Tensor info section starts at ~d~n~n", [TensorSectionStart]),

        % Parse all TensorCount tensor_info_records
        iter_tensors(Bytes, TensorSectionStart, 1, TensorCount, [], Summary)
    )),
    statistics(real_time, [T1|_]),
    ParseElapsed is T1 - T0,
    format("~nAll parses complete in ~3f sec~n", [ParseElapsed]),

    length(Summary, N),
    format("Tensors parsed: ~d / ~d~n~n", [N, TensorCount]),

    % Aggregate by tensor type
    aggregate_types(Summary, TypeCounts),
    format("Tensor type distribution:~n"),
    forall(member(T-Count, TypeCounts),
           ( type_name(T, Name),
             format("  type ~d (~w): ~d tensor(s)~n", [T, Name, Count]) )).

walk_metadata_pairs(_, Offset, N, Max, Offset) :- N > Max, !.
walk_metadata_pairs(Bytes, Offset, N, Max, FinalOffset) :-
    read_section_at(metadata_kv_pair, Bytes, Offset, _, Consumed),
    NextOffset is Offset + Consumed,
    NextN is N + 1,
    walk_metadata_pairs(Bytes, NextOffset, NextN, Max, FinalOffset).

iter_tensors(_, _Offset, N, Max, Acc, Summary) :-
    N > Max, !,
    reverse(Acc, Summary).
iter_tensors(Bytes, Offset, N, Max, Acc, Summary) :-
    catch(
        read_section_at(tensor_info_record, Bytes, Offset, Pairs, Consumed),
        E,
        ( format("  tensor ~d at offset ~d: EXCEPTION ~w~n", [N, Offset, E]),
          fail )
    ),
    member(field(name, NameStr), Pairs),
    member(field(n_dimensions, NDims), Pairs),
    member(field(dimensions, Dims), Pairs),
    member(field(type, TypeCode), Pairs),
    member(field(offset, TensorOffset), Pairs),

    % Compact display
    ( N =< 10
    -> format("  tensor ~d: name='~w' shape=~w type=~d offset=~d (consumed=~d)~n",
              [N, NameStr, Dims, TypeCode, TensorOffset, Consumed])
    ; N =:= 11
    -> format("  ... (~d more tensors, summary at end)~n", [Max - 10])
    ; true
    ),

    NextOffset is Offset + Consumed,
    NextN is N + 1,
    Acc2 = [tensor(N, NameStr, NDims, TypeCode, TensorOffset, Consumed) | Acc],
    iter_tensors(Bytes, NextOffset, NextN, Max, Acc2, Summary).

% Aggregate types
aggregate_types(Summary, Counts) :-
    findall(T, member(tensor(_, _, _, T, _, _), Summary), Types),
    msort(Types, Sorted),
    count_runs(Sorted, Counts).

count_runs([], []).
count_runs([X | Rest], [X-Count | RestCounts]) :-
    count_run(X, Rest, 1, Count, Tail),
    count_runs(Tail, RestCounts).

count_run(_, [], Acc, Acc, []).
count_run(X, [X | Rest], Acc, Count, Tail) :-
    !,
    Acc1 is Acc + 1,
    count_run(X, Rest, Acc1, Count, Tail).
count_run(_, Tail, Acc, Acc, Tail).

% GGUF tensor type codes (subset)
type_name(0, 'F32').
type_name(1, 'F16').
type_name(2, 'Q4_0').
type_name(3, 'Q4_1').
type_name(6, 'Q5_0').
type_name(7, 'Q5_1').
type_name(8, 'Q8_0').
type_name(9, 'Q8_1').
type_name(10, 'Q2_K').
type_name(11, 'Q3_K').
type_name(12, 'Q4_K').
type_name(13, 'Q5_K').
type_name(14, 'Q6_K').
type_name(15, 'Q8_K').
type_name(_, unknown).

:- initialization(cli_main, main).

cli_main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [PathArg | _]
    ->  ( atom(PathArg) -> Path = PathArg ; atom_string(Path, PathArg) ),
        main(Path)
    ;   default_gguf_file(P),
        main(P)
    ).
