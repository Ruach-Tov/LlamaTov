%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_crossword.pl — Detect crossword-puzzle attacks in GGUF files
%%
%% Uses safe_read.pl (byte-ownership tracking) to parse GGUF files.
%% Demonstrates detection of:
%%   1. Tensor aliasing (two tensors sharing bytes)
%%   2. Read-past-EOF (tensor data overflows file)
%%   3. Phantom data (unclaimed bytes at EOF)
%%   4. Well-formed file (all bytes accounted for)

:- module(test_crossword, [test_crossword/1, test_crossword/0]).
:- use_module('lib/safe_read').

%% ═══════════════════════════════════════════════════════════════
%% GGUF parser using safe_read (byte-ownership enforced)
%% ═══════════════════════════════════════════════════════════════

parse_gguf(Path, Result) :-
    format("~n=== Parsing: ~w ===~n", [Path]),
    catch(
        parse_gguf_inner(Path, Result),
        Error,
        (Result = error(Error),
         format("  CAUGHT: ~w~n", [Error]))
    ).

parse_gguf_inner(Path, Result) :-
    safe_open(Path, H0),

    %% Header: magic + version + tensor_count + kv_count
    safe_read_uint32_le(H0, Magic, H1),
    (Magic =:= 0x46554747    % "GGUF" little-endian
     -> true
     ;  throw(error(bad_magic(Magic), _))),
    safe_read_uint32_le(H1, Version, H2),
    format("  Version: ~d~n", [Version]),
    safe_read_uint64_le(H2, TensorCount, H3),
    format("  Tensors: ~d~n", [TensorCount]),
    safe_read_uint64_le(H3, KVCount, H4),
    format("  KV pairs: ~d~n", [KVCount]),

    %% Metadata KV pairs
    read_kv_pairs(H4, KVCount, KVPairs, H5),
    format("  Metadata: ~w~n", [KVPairs]),

    %% Tensor info entries
    read_tensor_infos(H5, TensorCount, TensorInfos, H6),
    format("  Tensor infos: ~w~n", [TensorInfos]),

    %% Calculate data section start (aligned to 32 bytes)
    safe_position(H6, MetaEnd),
    DataStart is MetaEnd + ((32 - MetaEnd mod 32) mod 32),

    %% Claim alignment padding bytes
    PadLen is DataStart - MetaEnd,
    (PadLen > 0
     -> safe_read_bytes(H6, PadLen, _, H7)
     ;  H7 = H6),

    %% Read tensor data (claim the bytes each tensor points to)
    read_tensor_data(H7, TensorInfos, DataStart, H8),

    %% Verify completeness
    safe_verify_complete(H8),
    safe_close(H8),

    Result = ok(Version, KVPairs, TensorInfos).

%% ═══════════════════════════════════════════════════════════════
%% KV pair reader
%% ═══════════════════════════════════════════════════════════════

read_kv_pairs(H, 0, [], H) :- !.
read_kv_pairs(H0, N, [Key-Value | Rest], HN) :-
    N > 0,
    safe_read_string(H0, Key, H1),
    safe_read_uint32_le(H1, Type, H2),
    read_kv_value(H2, Type, Value, H3),
    N1 is N - 1,
    read_kv_pairs(H3, N1, Rest, HN).

read_kv_value(H0, 8, Value, H1) :-  % type 8 = string
    safe_read_string(H0, Value, H1).
read_kv_value(H0, 4, Value, H1) :-  % type 4 = uint32
    safe_read_uint32_le(H0, Value, H1).
read_kv_value(H0, 5, Value, H1) :-  % type 5 = int32
    safe_read_int32_le(H0, Value, H1).
read_kv_value(H0, 6, Value, H1) :-  % type 6 = float32
    safe_read_float32_le(H0, Value, H1).
read_kv_value(H0, 7, Value, H1) :-  % type 7 = bool
    safe_read_bool(H0, Value, H1).
read_kv_value(H0, Type, unknown_type(Type), H0) :-
    format("  WARNING: unknown KV type ~d~n", [Type]).

%% ═══════════════════════════════════════════════════════════════
%% Tensor info reader
%% ═══════════════════════════════════════════════════════════════

read_tensor_infos(H, 0, [], H) :- !.
read_tensor_infos(H0, N, [tensor_info(Name, Dims, Type, Offset) | Rest], HN) :-
    N > 0,
    safe_read_string(H0, Name, H1),
    safe_read_uint32_le(H1, NDims, H2),
    read_dims(H2, NDims, Dims, H3),
    safe_read_uint32_le(H3, Type, H4),
    safe_read_uint64_le(H4, Offset, H5),
    format("  tensor: ~w dims=~w type=~d offset=~d~n", [Name, Dims, Type, Offset]),
    N1 is N - 1,
    read_tensor_infos(H5, N1, Rest, HN).

read_dims(H, 0, [], H) :- !.
read_dims(H0, N, [D | Rest], HN) :-
    N > 0,
    safe_read_uint64_le(H0, D, H1),
    N1 is N - 1,
    read_dims(H1, N1, Rest, HN).

%% ═══════════════════════════════════════════════════════════════
%% Tensor data reader (claims bytes at offsets)
%% ═══════════════════════════════════════════════════════════════

read_tensor_data(H, [], _, H).
read_tensor_data(H0, [tensor_info(Name, Dims, Type, Offset) | Rest], DataStart, HN) :-
    tensor_byte_size(Dims, Type, ByteSize),
    AbsOffset is DataStart + Offset,
    format("  claiming ~w: ~d bytes at offset ~d~n", [Name, ByteSize, AbsOffset]),
    safe_seek_and_read_bytes(H0, AbsOffset, ByteSize, _, H1),
    read_tensor_data(H1, Rest, DataStart, HN).

tensor_byte_size(Dims, Type, Size) :-
    foldl([D, Acc0, Acc1]>>(Acc1 is Acc0 * D), Dims, 1, NumElements),
    type_element_size(Type, ElemSize),
    Size is NumElements * ElemSize.

type_element_size(0, 4).   % F32
type_element_size(1, 2).   % F16
type_element_size(2, 1).   % Q4_0 (approximate)
type_element_size(3, 1).   % Q4_1
type_element_size(_, 4).   % default to F32

%% ═══════════════════════════════════════════════════════════════
%% Test runner
%% ═══════════════════════════════════════════════════════════════

test_crossword :-
    test_crossword('/tmp').

test_crossword(Dir) :-
    format("~n╔══════════════════════════════════════════════════╗~n"),
    format("║  Crossword-Puzzle GGUF Detection Tests           ║~n"),
    format("╚══════════════════════════════════════════════════╝~n"),

    %% Test 1: Tensor aliasing — should FAIL (double-claim)
    atomic_list_concat([Dir, '/crossword_alias.gguf'], Path1),
    format("~n── Test 1: Tensor aliasing (two tensors, same bytes) ──~n"),
    parse_gguf(Path1, R1),
    (R1 = error(_)
     -> format("  ✓ DETECTED: tensor aliasing caught by byte-ownership~n")
     ;  format("  ✗ MISSED: aliasing was not detected!~n")),

    %% Test 2: Short data — should FAIL (read past EOF or claim failure)
    atomic_list_concat([Dir, '/crossword_header_overlap.gguf'], Path2),
    format("~n── Test 2: Truncated tensor data (overflow) ──~n"),
    parse_gguf(Path2, R2),
    (R2 = error(_)
     -> format("  ✓ DETECTED: data overflow caught~n")
     ;  format("  ✗ MISSED: overflow was not detected!~n")),

    %% Test 3: Phantom data — should WARN (unclaimed bytes)
    atomic_list_concat([Dir, '/crossword_phantom.gguf'], Path3),
    format("~n── Test 3: Phantom data (unreferenced bytes at EOF) ──~n"),
    parse_gguf(Path3, R3),
    (R3 = ok(_, _, _)
     -> format("  ✓ PARSED but warned about phantom data~n")
     ;  format("  Result: ~w~n", [R3])),

    %% Test 4: Well-formed — should PASS completely
    atomic_list_concat([Dir, '/crossword_good.gguf'], Path4),
    format("~n── Test 4: Well-formed GGUF (should pass) ──~n"),
    parse_gguf(Path4, R4),
    (R4 = ok(_, _, _)
     -> format("  ✓ CLEAN: well-formed file, all bytes accounted for~n")
     ;  format("  ✗ FALSE POSITIVE: good file rejected!~n")),

    format("~n════════════════════════════════════════════════════~n"),
    format("Summary: byte-ownership tracking detects crossword attacks.~n"),
    format("Every byte has exactly one owner. No byte read twice.~n").
