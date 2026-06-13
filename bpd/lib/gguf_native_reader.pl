%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% gguf_native_reader.pl — Native Prolog GGUF parser using safe_read.pl
%%
%% Reads GGUF files directly in Prolog with byte-ownership tracking.
%% No shell-out. No Python. No C. Pure Prolog binary I/O.
%%
%% Usage:
%%   ?- gguf_read(Path, Header, Metadata, TensorInfos).
%%   ?- gguf_architecture_native(Path, Arch).
%%   ?- gguf_tensor_list(Path, Tensors).

:- module(gguf_native_reader, [
    gguf_read/4,
    gguf_read/5,
    gguf_read_full/4,
    gguf_read_full/5,
    gguf_data_start/2,
    gguf_architecture_native/2,
    gguf_tensor_list/2
]).

:- discontiguous read_kv_value/4.

:- use_module('lib/safe_read').

%% ═══════════════════════════════════════════════════════════════
%% Top-level reader
%% ═══════════════════════════════════════════════════════════════

gguf_read(Path, header(Version, TensorCount, KVCount), Metadata, TensorInfos) :-
    safe_open(Path, H0),
    catch(
        gguf_read_inner(H0, Version, TensorCount, KVCount, Metadata, TensorInfos, HN),
        Error,
        (safe_close(H0), throw(Error))
    ),
    safe_close(HN).

%% gguf_read/5 \u2014 like gguf_read/4 but also returns the absolute file offset
%% of the tensor-data region. Tensor data lives at file position
%% DataStart + Tensor.Offset.
%%
%% DataStart is the position right after the tensor_info table, padded
%% up to the alignment boundary (32 bytes default, or general.alignment KV).
gguf_read(Path, Header, Metadata, TensorInfos, DataStart) :-
    safe_open(Path, H0),
    catch(
        (gguf_read_inner(H0, Version, TensorCount, KVCount, Metadata, TensorInfos, HN),
         Header = header(Version, TensorCount, KVCount),
         %% Position after reading tensor_info section: read from the
         %% underlying stream via safe_position/2 (NOT from the safe_handle's
         %% third arg, which is the claimed-ranges list, not a byte position).
         safe_position(HN, PosAfter),
         %% Alignment: prefer general.alignment metadata if present, else 32
         (member('general.alignment'-A, Metadata) -> Alignment = A ; Alignment = 32),
         Pad is (Alignment - (PosAfter mod Alignment)) mod Alignment,
         DataStart is PosAfter + Pad),
        Error,
        (safe_close(H0), throw(Error))
    ),
    safe_close(HN).

%% gguf_data_start/2 \u2014 convenience: just the data_start offset.
gguf_data_start(Path, DataStart) :-
    gguf_read(Path, _, _, _, DataStart).

gguf_read_inner(H0, Version, TensorCount, KVCount, Metadata, TensorInfos, H6) :-
    %% Magic: "GGUF" = 0x46554747 little-endian
    safe_read_uint32_le(H0, Magic, H1),
    (Magic =:= 0x46554747 -> true ; throw(error(bad_gguf_magic(Magic), _))),
    safe_read_uint32_le(H1, Version, H2),
    safe_read_uint64_le(H2, TensorCount, H3),
    safe_read_uint64_le(H3, KVCount, H4),
    %% Metadata KV pairs
    read_kv_pairs(H4, KVCount, Metadata, H5),
    %% Tensor info entries
    read_tensor_infos(H5, TensorCount, TensorInfos, H6).

%% ═══════════════════════════════════════════════════════════════
%% KV pair reader
%% ═══════════════════════════════════════════════════════════════

read_kv_pairs(H, 0, [], H) :- !.
read_kv_pairs(H0, N, [Key-Value | Rest], HN) :-
    N > 0,
    safe_read_string(H0, Key, H1),
    safe_read_uint32_le(H1, Type, H2),
    read_kv_value(Type, H2, Value, H3),
    N1 is N - 1,
    read_kv_pairs(H3, N1, Rest, HN).

%% Type dispatch (GGUF spec: 13 value types)
read_kv_value(0, H0, Value, H1) :- safe_read_uint8(H0, Value, H1).      % uint8
read_kv_value(1, H0, Value, H1) :- safe_read_int8(H0, Value, H1).       % int8
read_kv_value(2, H0, Value, H1) :- safe_read_uint16_le(H0, Value, H1).  % uint16
read_kv_value(3, H0, Value, H1) :- safe_read_uint16_le(H0, V, H1),      % int16
    (V > 32767 -> Value is V - 65536 ; Value = V).
read_kv_value(4, H0, Value, H1) :- safe_read_uint32_le(H0, Value, H1).  % uint32
read_kv_value(5, H0, Value, H1) :- safe_read_int32_le(H0, Value, H1).   % int32
read_kv_value(6, H0, Value, H1) :- safe_read_float32_le(H0, Value, H1). % float32
read_kv_value(7, H0, Value, H1) :- safe_read_bool(H0, Value, H1).       % bool
read_kv_value(8, H0, Value, H1) :- safe_read_string(H0, Value, H1).     % string
read_kv_value(9, H0, array(ElemType, Count, Values), HN) :-               % array
    safe_read_uint32_le(H0, ElemType, H1),
    safe_read_uint64_le(H1, Count, H2),
    %% For large arrays (tokenizer vocab etc.), normally skip past without parsing
    %% to avoid claimed-ranges blowup. But if force_full_arrays flag is set,
    %% materialize them (needed for tokenizer use).
    ((Count > 1000, \+ gguf_force_full_arrays_flag)
     -> Values = skipped(Count, ElemType),
        skip_large_array(ElemType, Count, H2, HN)
     ;  read_array_elements(ElemType, Count, H2, Values, HN)
    ).

%% Thread-local flag: when true, large arrays are materialized in full.
%% Default: not set (so behavior matches the old skip-large-arrays default).
:- dynamic(gguf_force_full_arrays_flag/0).

%% gguf_read_full/4: like gguf_read/4 but reads all arrays, including large
%% tokenizer ones. Use when you need vocab/merges as Prolog lists.
gguf_read_full(Path, Header, Metadata, TensorInfos) :-
    asserta(gguf_force_full_arrays_flag),
    catch(
        (gguf_read(Path, Header, Metadata, TensorInfos),
         retractall(gguf_force_full_arrays_flag)),
        E,
        (retractall(gguf_force_full_arrays_flag), throw(E))).

%% gguf_read_full/5: like gguf_read/5 but reads all arrays in full.
gguf_read_full(Path, Header, Metadata, TensorInfos, DataStart) :-
    asserta(gguf_force_full_arrays_flag),
    catch(
        (gguf_read(Path, Header, Metadata, TensorInfos, DataStart),
         retractall(gguf_force_full_arrays_flag)),
        E,
        (retractall(gguf_force_full_arrays_flag), throw(E))).
read_kv_value(10, H0, Value, H1) :- safe_read_uint64_le(H0, Value, H1). % uint64
read_kv_value(11, H0, Value, H1) :-                                      % int64
    safe_read_uint64_le(H0, V, H1),
    (V > 9223372036854775807 -> Value is V - 18446744073709551616 ; Value = V).
read_kv_value(12, H0, Value, H1) :-                                      % float64
    safe_read_bytes(H0, 8, _Bytes, H1), Value = 0.0. % TODO: decode f64
read_kv_value(Type, H, unknown_type(Type), H).

read_array_elements(_, 0, H, [], H) :- !.
read_array_elements(Type, N, H0, [V|Rest], HN) :-
    N > 0,
    read_kv_value(Type, H0, V, H1),
    N1 is N - 1,
    read_array_elements(Type, N1, H1, Rest, HN).

%% Skip a large array efficiently. Records the whole region as one claimed
%% block. Dispatches by element type so we use seek/4 for fixed-size types
%% and per-element length-then-seek for variable-length strings.
skip_large_array(ElemType, Count, safe_handle(S, FS, C0), safe_handle(S, FS, C1)) :-
    safe_read:byte_count(S, StartPos),
    skip_array_elems(ElemType, Count, S),
    safe_read:byte_count(S, EndPos),
    safe_read:claim_range(StartPos, EndPos, C0, C1).

%% Fixed-size element sizes (bytes per element).
elem_size(0, 1).    %% uint8
elem_size(1, 1).    %% int8
elem_size(2, 2).    %% uint16
elem_size(3, 2).    %% int16
elem_size(4, 4).    %% uint32
elem_size(5, 4).    %% int32
elem_size(6, 4).    %% float32
elem_size(7, 1).    %% bool
elem_size(10, 8).   %% uint64
elem_size(11, 8).   %% int64
elem_size(12, 8).   %% float64

%% Skip Count elements of given type, advancing the stream.
skip_array_elems(8, Count, S) :- !,
    %% Type 8 = string: per-element length-prefix + content
    skip_n_strings(S, Count).
skip_array_elems(ElemType, Count, S) :-
    elem_size(ElemType, ElemBytes), !,
    TotalBytes is Count * ElemBytes,
    seek(S, TotalBytes, current, _).
skip_array_elems(ElemType, _Count, _S) :-
    throw(error(unsupported_array_elem_type(ElemType), _)).

%% Skip Count length-prefixed strings. Each = 8-byte u64 length + content.
skip_n_strings(_, 0) :- !.
skip_n_strings(S, N) :-
    N > 0,
    get_byte(S, B0), get_byte(S, B1), get_byte(S, B2), get_byte(S, B3),
    get_byte(S, B4), get_byte(S, B5), get_byte(S, B6), get_byte(S, B7),
    Len is B0 + B1*256 + B2*65536 + B3*16777216
         + B4*4294967296 + B5*1099511627776
         + B6*281474976710656 + B7*72057594037927936,
    seek(S, Len, current, _),
    N1 is N - 1,
    skip_n_strings(S, N1).

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
    N1 is N - 1,
    read_tensor_infos(H5, N1, Rest, HN).

read_dims(H, 0, [], H) :- !.
read_dims(H0, N, [D|Rest], HN) :-
    N > 0,
    safe_read_uint64_le(H0, D, H1),
    N1 is N - 1,
    read_dims(H1, N1, Rest, HN).

%% ═══════════════════════════════════════════════════════════════
%% Convenience predicates
%% ═══════════════════════════════════════════════════════════════

%% Extract architecture from GGUF metadata
gguf_architecture_native(Path, Arch) :-
    gguf_read(Path, _, Metadata, _),
    member('general.architecture'-ArchRaw, Metadata),
    %% Normalize: llama.cpp uses dashes, we use underscores
    atom_chars(ArchRaw, Chars),
    maplist([C, D]>>(C = '-' -> D = '_' ; D = C), Chars, NewChars),
    atom_chars(Arch, NewChars).

%% List all tensors with their shapes and types
gguf_tensor_list(Path, Tensors) :-
    gguf_read(Path, _, _, TensorInfos),
    maplist([tensor_info(N,D,T,O), tensor(N,D,T,O)]>>true, TensorInfos, Tensors).
