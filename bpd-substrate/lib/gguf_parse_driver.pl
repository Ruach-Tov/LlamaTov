%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
% gguf_parse_driver.pl — drive Prolog GGUF parsing against a real file.
%
% Usage:
%
%   swipl -q -g 'consult("gguf_parse_driver.pl"), main("/path/to/file.gguf")' -t halt
%
% Or with the default file:
%
%   swipl -q -g 'consult("gguf_parse_driver.pl"), main' -t halt
%
% Sequence:
%   1. Consult byte_readers.pl (runtime byte primitives)
%   2. Consult the GGUF format IR (field/5, field_byte_size/3,
%      format_endianness/2, etc.)
%   3. Consult prolog_reader_generator.pl (the generator predicates)
%   4. Call emit_section_reader(header) to assert the read_section
%      clause for the GGUF header
%   5. Read the first 24 bytes of the .gguf file as a byte list
%   6. Invoke read_section(header, Bytes, Pairs)
%   7. Print the field name → value pairs
%
% Per Heath's architecture: the spec (gguf.pl) is consulted; the
% generator runs as Prolog; the generated code is asserted; the
% generated code is invoked. All in Prolog, all in one process.
%
% Author: metayen, 2026-05-12.

% Per R-005 (fixed in 6fa01878b): bpd_to_prolog.py renames the BPD
% `format/1` predicate to `bpd_format/1` during IR compilation,
% avoiding the collision with SWI-Prolog's built-in format/1
% (output formatting). The old redefine_system_predicate workaround
% is no longer needed.

:- consult('ir/include/byte_readers.pl').
:- consult('ir/include/prolog_reader_generator.pl').


% Default file: the first GGUF blob in the local Ollama cache.
% Override by passing a path argument to main/1.

default_gguf_file(
    '${OLLAMA_BLOBS:-~/.ollama/models/blobs}/\c
sha256-0577f52a4edfd5e48bb59c296bb4f40328161ecc3d0aa4398b3cb6b2b7367cac').


% read_first_n_bytes(+Path, +N, -Bytes)
%
% Reads the first N bytes of a binary file into a list of integers
% in the range 0..255. Uses SWI-Prolog's binary stream support.

read_first_n_bytes(Path, N, Bytes) :-
    setup_call_cleanup(
        open(Path, read, Stream, [type(binary)]),
        read_n_bytes_from_stream(Stream, N, Bytes),
        close(Stream)
    ).

read_n_bytes_from_stream(_Stream, 0, []) :- !.
read_n_bytes_from_stream(Stream, N, [B | Rest]) :-
    N > 0,
    get_byte(Stream, B),
    B \= -1,  % -1 indicates EOF
    N1 is N - 1,
    read_n_bytes_from_stream(Stream, N1, Rest).


% print_field_pair(+Pair)
%
% Pretty-print one field(Name, Value) pair. For byte sequences, show
% the bytes in hex; for integers, show decimal.

print_field_pair(field(Name, Value)) :-
    (   is_list(Value)
    ->  length(Value, ByteCount),
        format("  ~w: ~w (~d bytes)~n", [Name, Value, ByteCount]),
        bytes_as_hex_string(Value, Hex),
        format("    hex: ~w~n", [Hex])
    ;   format("  ~w: ~w~n", [Name, Value])
    ).

bytes_as_hex_string([], "").
bytes_as_hex_string([B], Hex) :- !,
    format(atom(Hex), "~`0t~16r~2|", [B]).
bytes_as_hex_string([B | Rest], Hex) :-
    format(atom(HexB), "~`0t~16r~2|", [B]),
    bytes_as_hex_string(Rest, RestHex),
    atom_concat(HexB, " ", HexB1),
    atom_concat(HexB1, RestHex, Hex).


% main(+Path) — entry point.

main(Path) :-
    format("~n=== GGUF parser (Prolog-generated) ===~n~n"),
    format("File: ~w~n~n", [Path]),

    % Consult the format IR
    consult('file-formats/gguf/output/gguf.pl'),

    % Generate the section reader from the IR
    emit_section_reader(header),

    % Read the first 24 bytes of the GGUF file (the header)
    read_first_n_bytes(Path, 24, Bytes),

    % Parse the header
    read_section(header, Bytes, HeaderPairs),
    format("Header:~n"),
    forall(member(P, HeaderPairs), print_field_pair(P)),

    % Metadata K-V parsing requires generator support for `repeated/3`
    % (count-from-field iteration over N records) and `sub_record/2`
    % (recursive parsing for array values). Both vocabulary features
    % are declared in gguf.bpd's metadata_kv_section but not yet handled
    % by the generator. The example library at examples/ exercises the
    % vocabulary features that ARE supported (length_prefixed,
    % byte_offset(after(...)), dispatch/4 — see examples/*/spec.bpd).
    member(field(metadata_kv_count, MetadataCount), HeaderPairs),
    format("~nMetadata: ~d K-V pairs declared by header, but the generator~n",
           [MetadataCount]),
    format("does not yet emit Prolog clauses for repeated/3 + sub_record/2~n"),
    format("vocabulary features used by metadata_kv_section. Tracked as~n"),
    format("future generator extensions; see examples/BUGS_AND_REFACTORS.md.~n"),

    format("~nDone.~n").

main :-
    default_gguf_file(Path),
    main(Path).
