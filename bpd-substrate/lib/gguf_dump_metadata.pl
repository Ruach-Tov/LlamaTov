#!/usr/bin/env swipl
%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
% gguf_dump_metadata.pl — Iterate ALL metadata K-V pairs in a real GGUF
% file. Closes V.C end-to-end: previously demonstrated isolated array
% parsing; this proves the substrate handles a full real-world metadata
% section, all pair shapes mixed, offsets cleanly partitioning the section.
%
% Output per pair: number, key, value_type, consumed bytes, value summary.
% Summaries are intentionally compact — arrays show "array of N x type"
% rather than dumping 30K elements.
%
% Usage:
%   ./gguf_dump_metadata.pl [path-to-file.gguf]
%
% Default: nomic-embed-text-v1.5 (262MB, GGUF v3, 24 K-V pairs).

:- consult('ir/include/byte_readers.pl').
:- consult('ir/include/landmarks.pl').
:- consult('ir/include/repeated_records.pl').
:- consult('file-formats/gguf/output/gguf.pl').
:- consult('ir/include/prolog_reader_generator.pl').

default_gguf_file(
    '${OLLAMA_BLOBS:-~/.ollama/models/blobs}/\c
sha256-970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6').

% Read only first MaxBytes — sufficient for typical metadata sections.
% GGUF metadata usually fits within first ~1-5 MB even for large models.
% For files where it doesn't, pass a larger MaxBytes or use read_full_file.

read_file_string(Path, Str) :-
    read_file_partial(Path, 4_000_000, Str).

read_file_partial(Path, MaxBytes, Str) :-
    open(Path, read, Stream, [type(binary)]),
    setup_call_cleanup(true,
        read_string(Stream, MaxBytes, Str),
        close(Stream)).

main :-
    default_gguf_file(P),
    main(P).

main(Path) :-
    format("~n=== GGUF full metadata K-V dump ===~n~n"),
    format("File: ~w~n~n", [Path]),

    statistics(real_time, [T0|_]),
    read_file_string(Path, Bytes),
    statistics(real_time, [T1|_]),
    string_length(Bytes, Total),
    LoadElapsed is T1 - T0,
    format("Loaded ~d bytes in ~3f sec~n~n", [Total, LoadElapsed]),

    emit_section_reader(header),
    emit_section_reader(metadata_kv_pair),

    statistics(real_time, [T2|_]),
    with_parse_session(dump, (
        read_section(header, Bytes, HeaderPairs),
        format("Header: ~w~n~n", [HeaderPairs]),
        member(field(metadata_kv_count, KVCount), HeaderPairs),
        member(field(tensor_count, TensorCount), HeaderPairs),

        bytes_slice_from(Bytes, 24, MetaBytes),
        format("Iterating ~d K-V pairs (~d tensors follow)...~n~n",
               [KVCount, TensorCount]),

        iter_pairs(MetaBytes, 0, 1, KVCount, [], Summary, FinalOffset)
    )),
    statistics(real_time, [T3|_]),
    ParseElapsed is T3 - T2,
    format("~nAll pairs parsed in ~3f sec~n", [ParseElapsed]),
    format("Metadata section ends at offset ~d~n", [FinalOffset]),
    AbsEnd is 24 + FinalOffset,
    format("(absolute file offset: ~d / 0x~16r)~n~n", [AbsEnd, AbsEnd]),

    length(Summary, NParsed),
    format("Pairs parsed: ~d / ~d~n~n", [NParsed, KVCount]),

    % Cross-check: count types
    aggregate_types(Summary, TypeCounts),
    format("Value type distribution:~n"),
    forall(member(VT-Count, TypeCounts),
           ( type_name(VT, Name),
             format("  type ~d (~w): ~d pair(s)~n", [VT, Name, Count]) )).

% iter_pairs: walk all N pairs, accumulating compact summaries.

iter_pairs(_, Offset, N, Max, Acc, Summary, Offset) :-
    N > Max, !,
    reverse(Acc, Summary).
iter_pairs(Bytes, Offset, N, Max, Acc, Summary, FinalOffset) :-
    % Peek key + value_type without parsing the value
    read_u64_le(Bytes, Offset, KeyLen),
    KeyStart is Offset + 8,
    KeyEnd is KeyStart + KeyLen,
    bytes_slice(Bytes, KeyStart, KeyLen, KeyBytes),
    bytes_to_atom(KeyBytes, Key),
    read_u32_le(Bytes, KeyEnd, ValueType),

    % Full parse to verify Consumed cleanly partitions the section
    catch(
        read_section_at(metadata_kv_pair, Bytes, Offset, Pairs, Consumed),
        E,
        ( format("  pair ~d at offset ~d: key='~w' EXCEPTION ~w~n",
                 [N, Offset, Key, E]),
          fail )
    ),

    summarize_value(ValueType, Pairs, ValueSummary),
    format("  pair ~d: key='~w' [type ~d] consumed=~d  ~w~n",
           [N, Key, ValueType, Consumed, ValueSummary]),

    NextOffset is Offset + Consumed,
    NextN is N + 1,
    Acc2 = [pair(N, Key, ValueType, Consumed) | Acc],
    iter_pairs(Bytes, NextOffset, NextN, Max, Acc2, Summary, FinalOffset).

% bytes_to_atom: convert key bytes (string or list) to an atom for display
bytes_to_atom(Bytes, Atom) :-
    string(Bytes), !,
    atom_string(Atom, Bytes).
bytes_to_atom(Bytes, Atom) :-
    is_list(Bytes), !,
    string_codes(S, Bytes),
    atom_string(Atom, S).

% summarize_value/3: compact value summary for display
summarize_value(VType, Pairs, Summary) :-
    member(field(value, V), Pairs),
    summarize_value_inner(VType, V, Summary).

% Scalar types: just show value
summarize_value_inner(4, V, scalar_u32(V)) :- !.
summarize_value_inner(5, V, scalar_i32(V)) :- !.
summarize_value_inner(6, V, scalar_f32(V)) :- !.
summarize_value_inner(7, 0, false) :- !.
summarize_value_inner(7, 1, true) :- !.
summarize_value_inner(10, V, scalar_u64(V)) :- !.
summarize_value_inner(11, V, scalar_i64(V)) :- !.
summarize_value_inner(12, V, scalar_f64(V)) :- !.

% String (type 8): show length
summarize_value_inner(8, V, string(L)) :-
    !, value_length(V, L).

% Array (type 9): unpack and show count + element type
% V is a sub-record: [field(element_type, T), field(element_count, N), field(items, ...)]
summarize_value_inner(9, SubPairs, array(N, ElementType)) :-
    member(field(element_type, ElementType), SubPairs),
    member(field(element_count, N), SubPairs), !.

summarize_value_inner(VT, _, opaque(VT)).

% length of value: string length, list length, or 'unknown'
value_length(V, L) :- string(V), !, string_length(V, L).
value_length(V, L) :- is_list(V), !, length(V, L).
value_length(_, unknown).

% Aggregate types into [VT-Count, ...]
aggregate_types(Summary, Counts) :-
    findall(VT, member(pair(_, _, VT, _), Summary), Types),
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

type_name(1, i8).
type_name(2, u16).
type_name(3, i16).
type_name(4, u32).
type_name(5, i32).
type_name(6, f32).
type_name(7, bool).
type_name(8, string).
type_name(9, array).
type_name(10, u64).
type_name(11, i64).
type_name(12, f64).
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
