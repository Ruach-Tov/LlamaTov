#!/usr/bin/env swipl
%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
% gguf_print_general.pl — Print the general.* metadata K-V pairs from a GGUF file.
%
% Usage:
%   gguf_print_general.pl <path-to-file.gguf>   (executable invocation)
%   gguf_print_general.pl                       (uses the built-in default file)
%
% Or interactively:
%   swipl -q -g 'consult("gguf_print_general.pl"), main("/path/to/file.gguf")' -t halt
%   swipl -q -g 'consult("gguf_print_general.pl"), main' -t halt   (default)
%
% Demonstrates: the BPD substrate (V.A + V.B + R-008 + cycle guard +
% landmark scoping) correctly parses real GGUF metadata. This script
% iterates through the metadata K-V pairs from the start of the file
% and prints any pair whose key starts with "general.".
%
% Stops iteration at the first ARRAY-valued pair (type 9). Real GGUF
% files put `tokenizer.ggml.tokens` (a 30K+ element array) before the
% rest of the metadata, which is past the general.* prefix anyway.
% Iterating the full file requires solving the 25× memory amplification
% (R-009, future).

:- consult('ir/include/byte_readers.pl').
:- consult('ir/include/landmarks.pl').
:- consult('ir/include/repeated_records.pl').
:- consult('file-formats/gguf/output/gguf.pl').
:- consult('ir/include/prolog_reader_generator.pl').

% Default file: nomic-embed-text-v1.5 (262MB; first 64KB has all general.*)
default_gguf_file(
    '${OLLAMA_BLOBS:-~/.ollama/models/blobs}/\c
sha256-970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6').

read_first_bytes(Path, N, Bytes) :-
    open(Path, read, Stream, [type(binary)]),
    setup_call_cleanup(
        true,
        read_n_bytes(Stream, N, Bytes),
        close(Stream)
    ).
read_n_bytes(_, 0, []) :- !.
read_n_bytes(S, N, [B | Rest]) :-
    N > 0,
    get_byte(S, B),
    (   B == -1
    ->  Rest = []
    ;   N1 is N - 1,
        read_n_bytes(S, N1, Rest)
    ).

% --- main ----------------------------------------------------------

main :-
    default_gguf_file(Path),
    main(Path).

main(Path) :-
    format("~n=== GGUF general.* metadata ===~n~nFile: ~w~n~n", [Path]),

    % Read first 64KB — enough for all general.* metadata in typical files
    read_first_bytes(Path, 65536, Bytes),

    % Compile readers ONCE before the parse session.
    emit_section_reader(header),
    emit_section_reader(metadata_kv_pair),

    % Single parse session so the header's landmarks (specifically
    % metadata_kv_count and field_end(magic) etc.) persist when we
    % iterate metadata_kv_pairs. Wrapping in ONE session also gives
    % clean landmark cleanup on exit.
    with_parse_session(gguf_general, (
        read_section(header, Bytes, HeaderPairs),
        format("Header: ~w~n~n", [HeaderPairs]),
        member(field(metadata_kv_count, KVCount), HeaderPairs),

        % Metadata section starts immediately after the 24-byte header
        bytes_slice_from(Bytes, 24, MetaBytes),

        % Iterate pairs, accumulating the general.* ones
        iter_general(MetaBytes, 0, 1, KVCount, [], GeneralPairs)
    )),

    % Pretty-print what we found
    format("~n--- general.* metadata ---~n~n"),
    forall(member(Key-Value, GeneralPairs),
           format("  ~w = ~w~n", [Key, Value])),
    length(GeneralPairs, NG),
    format("~n(~d general.* pair(s))~n~n", [NG]).

% iter_general(+Bytes, +Offset, +N, +Max, +Acc, -Result)
% Iterate K-V pairs. Stop early when we hit an array (type 9) — beyond
% V.C's current scale handling.

iter_general(_, _, N, Max, Acc, Result) :- N > Max, !, reverse(Acc, Result).
iter_general(Bytes, Offset, N, Max, Acc, Result) :-
    % Pre-peek the value_type WITHOUT fully parsing the value, so we
    % can short-circuit on arrays (type 9) which are beyond our
    % current scale model (R-009 future work: streaming + reduced
    % memory amplification).
    %
    % metadata_kv_pair layout: key (length_prefixed u64+u8) then
    % value_type (u32). Read the key length, skip the key bytes,
    % then read the value_type at the next u32.
    read_u64_le(Bytes, Offset, KeyLen),
    KeyStart is Offset + 8,
    KeyEnd is KeyStart + KeyLen,
    bytes_slice(Bytes, KeyStart, KeyLen, KeyBytes),
    string_codes(KeyStr, KeyBytes),
    atom_string(Key, KeyStr),
    read_u32_le(Bytes, KeyEnd, ValueType),

    (   ValueType =:= 9
    ->  format("  [pair ~d: '~w' is an ARRAY (type 9). Stopping iteration here.]~n",
               [N, Key]),
        reverse(Acc, Result)
    ;   % Read the full pair via the V.C-validated substrate.
        read_section_at(metadata_kv_pair, Bytes, Offset, Pairs, Consumed),
        extract_pair(Pairs, _, ValueType, ValueRaw),
        format_value(ValueType, ValueRaw, ValueDisplay),
        (   starts_with_general(Key)
        ->  Acc2 = [Key-ValueDisplay | Acc]
        ;   Acc2 = Acc
        ),
        NextOffset is Offset + Consumed,
        NextN is N + 1,
        iter_general(Bytes, NextOffset, NextN, Max, Acc2, Result)
    ).

% extract_pair(+Pairs, -KeyAtom, -ValueType, -ValueRaw)
extract_pair(Pairs, KeyAtom, ValueType, ValueRaw) :-
    member(field(key, KeyBytes), Pairs),
    member(field(value_type, ValueType), Pairs),
    member(field(value, ValueRaw), Pairs),
    string_codes(KeyStr, KeyBytes),
    atom_string(KeyAtom, KeyStr).

% starts_with_general(+Key) — true if Key begins with "general."
starts_with_general(Key) :-
    sub_atom(Key, 0, 8, _, 'general.').

% format_value(+ValueType, +Raw, -Display)
format_value(8, RawBytes, Display) :-
    !,
    % String: convert byte list to string atom
    string_codes(S, RawBytes),
    atom_string(Display, S).
format_value(7, 0, false) :- !.
format_value(7, 1, true) :- !.
format_value(_, Raw, Raw).


% --- CLI entry point ---
%
% Per initialization(_, main) the script invokes cli_main automatically
% when run as `swipl /path/to/gguf_print_general.pl <args>` or via the
% #!/usr/bin/env swipl shebang line. SWI-Prolog provides the script-arg
% argv via current_prolog_flag(argv, _).

:- initialization(cli_main, main).

cli_main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [PathArg | _]
    ->  ( atom(PathArg) -> Path = PathArg ; atom_string(Path, PathArg) ),
        main(Path)
    ;   format("(no path given — using built-in default)~n", []),
        default_gguf_file(P),
        main(P)
    ).
