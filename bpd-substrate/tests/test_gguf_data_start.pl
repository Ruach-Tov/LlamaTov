%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
:- use_module(library(lists)).
:- use_module('lib/gguf_native_reader').

main :-
    Path = '/tmp/llamatov-data/ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45',
    format('Reading GGUF...~n'),
    gguf_read(Path, header(V, TC, KVC), _Metadata, Tensors, DataStart),
    format('Version: ~w  TensorCount: ~w  KVCount: ~w~n', [V, TC, KVC]),
    format('DataStart: ~w~n', [DataStart]),
    %% Find blk.0.attn_k.weight
    member(tensor_info('blk.0.attn_k.weight', Dims, Type, Offset), Tensors),
    !,
    AbsOffset is DataStart + Offset,
    format('blk.0.attn_k.weight: dims=~w type=~w rel_offset=~w abs_offset=~w~n',
           [Dims, Type, Offset, AbsOffset]).

:- main.
:- halt.
