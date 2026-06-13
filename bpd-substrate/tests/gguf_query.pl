%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% gguf_query.pl \u2014 query a tensor's offset/size from a GGUF file.
%%
%% Usage: swipl -q -g "consult('tests/gguf_query.pl'), gguf_query_main" -- <gguf_path> <tensor_name>
%%
%% Outputs (machine-readable, single line):
%%   ABS_OFFSET=<n> SIZE=<n> TYPE=<n> DIMS=<comma-separated>
%%
%% Returns success (0) or failure (1) if the tensor doesn't exist.

:- use_module(library(lists)).
:- use_module('lib/gguf_native_reader').

gguf_query_main :-
    current_prolog_flag(argv, Args),
    (Args = [Path, NameAtom]
     -> gguf_query_run(Path, NameAtom)
     ;  format(user_error, 'usage: gguf_query.pl <gguf_path> <tensor_name>~n', []),
        halt(2)).

gguf_query_run(Path, NameAtom) :-
    gguf_read(Path, _Header, _Metadata, Tensors, DataStart),
    !,
    (member(tensor_info(NameAtom, Dims, Type, Offset), Tensors)
     -> AbsOffset is DataStart + Offset,
        compute_size(Type, Dims, Size),
        atomic_list_concat(Dims, ',', DimsStr),
        format('ABS_OFFSET=~w SIZE=~w TYPE=~w DIMS=~w~n',
               [AbsOffset, Size, Type, DimsStr]),
        halt(0)
     ;  format(user_error, 'tensor not found: ~w~n', [NameAtom]),
        halt(1)).

%% Element count = product of dims.
elem_count([], 1).
elem_count([D|Ds], N) :- elem_count(Ds, M), N is D * M.

%% Compute byte size based on ggml type code.
%%   0=F32 (4 bytes/elem)
%%   1=F16 (2 bytes/elem)
%%   8=Q8_0 (34 bytes per 32-element block)
%%  12=Q4_K (144 bytes per 256-element block; the K-quants)
%%  others: best-effort, may not be exact
compute_size(0, Dims, Size) :- elem_count(Dims, N), Size is N * 4.
compute_size(1, Dims, Size) :- elem_count(Dims, N), Size is N * 2.
compute_size(8, Dims, Size) :-   %% Q8_0: 34 bytes per 32 elements
    elem_count(Dims, N),
    Blocks is N // 32,
    Size is Blocks * 34.
compute_size(12, Dims, Size) :-  %% Q4_K: 144 bytes per 256 elements
    elem_count(Dims, N),
    Blocks is N // 256,
    Size is Blocks * 144.
compute_size(Type, Dims, Size) :-
    %% Fallback: assume 1 byte per element (caller should check Type)
    elem_count(Dims, N),
    Size = N,
    format(user_error, 'WARN: unknown size for ggml_type=~w; reporting elem count~n', [Type]).
