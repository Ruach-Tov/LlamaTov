%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% spike_prolog_reads_gguf.pl
%%
%% Integration spike Stage 2: Prolog uses numpy via janus_swi to read
%% bytes from a real GGUF file, verifying the magic bytes match "GGUF".
%%
%% This is one more step up from spike_prolog_drives_torch.pl: now we
%% have a real GGUF file on disk being read by Python (numpy.fromfile)
%% under Prolog's direction. The Prolog substrate doesn't need to know
%% how Python reads the file — it just orchestrates.
%%
%% What this proves:
%%   1. Prolog can direct Python to read a specific file
%%   2. Prolog can get back the bytes as a structure it can verify
%%   3. The pipeline Prolog→numpy→bytes is wired and works
%%
%% Smallest viable cross-substrate operation. Once this works, reading
%% a specific tensor at a known offset is mechanically the same shape:
%% just change the offset and dtype.

:- use_module(library(janus)).

%% Path to the nomic-embed-text GGUF file used throughout this codebase
gguf_test_file('${OLLAMA_BLOBS:-~/.ollama/models/blobs}/sha256-970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6').

%% ────────────────────────────────────────────────────────────────────
%% Read the first 4 bytes of the GGUF file and verify magic
%% ────────────────────────────────────────────────────────────────────

%% read_magic_bytes(-MagicString) — directs numpy to read first 4 bytes
%% of the GGUF file and return them as a Python string.
read_magic_bytes(MagicString) :-
    gguf_test_file(Path),
    %% Janus returns small numpy arrays as Prolog lists. Read each
    %% byte and convert to its ASCII char.
    py_call(numpy:fromfile(Path, dtype = 'uint8', count = 4), [B1, B2, B3, B4]),
    py_call(B1:'__int__'(), I1),
    py_call(B2:'__int__'(), I2),
    py_call(B3:'__int__'(), I3),
    py_call(B4:'__int__'(), I4),
    atom_codes(MagicString, [I1, I2, I3, I4]).

%% read_gguf_header_int(+ByteOffset, +Width, -Value)
%%   Read a little-endian integer at given byte offset of given width.
%%   GGUF uses u32 (4 bytes) and u64 (8 bytes) in header fields.
read_gguf_header_int(ByteOffset, 4, Value) :-
    gguf_test_file(Path),
    py_call(numpy:fromfile(Path,
                            dtype = 'uint32',
                            count = 1,
                            offset = ByteOffset),
            [Single]),
    py_call(Single:'__int__'(), Value).

read_gguf_header_int(ByteOffset, 8, Value) :-
    gguf_test_file(Path),
    py_call(numpy:fromfile(Path,
                            dtype = 'uint64',
                            count = 1,
                            offset = ByteOffset),
            [Single]),
    py_call(Single:'__int__'(), Value).

%% ────────────────────────────────────────────────────────────────────
%% Test runner
%% ────────────────────────────────────────────────────────────────────

run_tests :-
    Tests = [
        test_read_gguf_magic,
        test_read_gguf_version,
        test_read_gguf_tensor_count,
        test_read_gguf_metadata_kv_count
    ],
    run_each(Tests, 0, 0, P, F),
    format("~n=============================================~n", []),
    format("RESULTS: ~d passed, ~d failed~n", [P, F]),
    format("=============================================~n", []),
    ( F > 0 -> halt(1) ; true ).

run_each([], P, F, P, F).
run_each([T | Rest], P0, F0, P, F) :-
    ( catch(call(T), Err, (format("  FAIL ~w: error ~w~n", [T, Err]), fail))
    -> ( format("  PASS ~w~n", [T]), P1 is P0 + 1, F1 = F0 )
    ; ( format("  FAIL ~w~n", [T]), P1 = P0, F1 is F0 + 1 )
    ),
    run_each(Rest, P1, F1, P, F).

%% ────────────────────────────────────────────────────────────────────
%% Tests
%% ────────────────────────────────────────────────────────────────────

%% Test 1: First 4 bytes of GGUF should be "GGUF" magic
test_read_gguf_magic :-
    read_magic_bytes(Magic),
    Magic == 'GGUF'.

%% Test 2: Bytes 4-7 are u32 version. GGUF v3 is current.
test_read_gguf_version :-
    read_gguf_header_int(4, 4, Version),
    Version == 3.

%% Test 3: Bytes 8-15 are u64 tensor count. Nomic-embed has 112 tensors.
test_read_gguf_tensor_count :-
    read_gguf_header_int(8, 8, TensorCount),
    TensorCount == 112.

%% Test 4: Bytes 16-23 are u64 metadata KV count. Nomic-embed has 24.
test_read_gguf_metadata_kv_count :-
    read_gguf_header_int(16, 8, KVCount),
    KVCount == 24.

:- initialization(run_tests, main).
