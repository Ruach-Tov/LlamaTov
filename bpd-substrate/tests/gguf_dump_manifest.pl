%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% gguf_dump_manifest.pl \u2014 dump GGUF metadata + tensor offsets in JSON for inference orchestration.
%%
%% Usage: swipl -q -g "consult('tests/gguf_dump_manifest.pl'), gguf_dump_main" -- <gguf_path> <output_json>
%%
%% Output JSON contains:
%%   - architecture, n_layers, n_heads, n_kv_heads, embed_dim, ffn_dim, vocab_size,
%%     rope_freq_base, rms_eps, head_dim, max_context, bos_token_id, eos_token_id
%%   - tokens: list of (id, text) for full vocab
%%   - merges: list of BPE merge rules (string pairs)
%%   - tensors: list of {name, dims, type, abs_offset, size_bytes}
%%
%% Outputs ONE LINE of pure JSON, no surrounding text.

:- use_module(library(lists)).
:- use_module(library(http/json)).
:- use_module('lib/gguf_native_reader').

gguf_dump_main :-
    current_prolog_flag(argv, Args),
    (Args = [Path, OutPath]
     -> gguf_dump_run(Path, OutPath)
     ;  format(user_error, 'usage: <gguf_path> <output_json>~n', []),
        halt(2)).

gguf_dump_run(Path, OutPath) :-
    gguf_read_full(Path, _Header, Metadata, Tensors, DataStart),
    !,
    %% Extract scalar config values
    metadata_value(Metadata, 'general.architecture', Arch, llama),
    metadata_value(Metadata, 'llama.block_count', NLayers, 16),
    metadata_value(Metadata, 'llama.attention.head_count', NHeads, 32),
    metadata_value(Metadata, 'llama.attention.head_count_kv', NKVHeads, 8),
    metadata_value(Metadata, 'llama.embedding_length', EmbedDim, 2048),
    metadata_value(Metadata, 'llama.feed_forward_length', FfnDim, 8192),
    metadata_value(Metadata, 'llama.attention.layer_norm_rms_epsilon', RmsEps, 1.0e-5),
    metadata_value(Metadata, 'llama.rope.freq_base', RopeBase, 500000.0),
    metadata_value(Metadata, 'llama.context_length', MaxContext, 131072),
    metadata_value(Metadata, 'llama.rope.dimension_count', RopeDim, 64),
    metadata_value(Metadata, 'tokenizer.ggml.bos_token_id', BosId, 128000),
    metadata_value(Metadata, 'tokenizer.ggml.eos_token_id', EosId, 128009),
    HeadDim is EmbedDim // NHeads,

    %% Vocab size from tokens list (already in metadata)
    %% array(_, Count, Values) is the full materialized form
    (member('tokenizer.ggml.tokens'-array(_, VocabSize, Tokens), Metadata)
     -> true
     ;  Tokens = [], VocabSize = 128256),

    %% Merges
    (member('tokenizer.ggml.merges'-array(_, _MCount, Merges), Metadata)
     -> true
     ;  Merges = []),

    %% Build tensor entries
    findall(TJ,
            (member(tensor_info(Name, Dims, Type, Offset), Tensors),
             AbsOff is DataStart + Offset,
             compute_size(Type, Dims, SizeBytes),
             TJ = _{name: Name, dims: Dims, type: Type, abs_offset: AbsOff, size_bytes: SizeBytes}),
            TensorJsons),

    %% Build tokens with explicit IDs
    build_token_records(Tokens, 0, TokenRecords),

    %% Build merges list of strings
    MergesList = Merges,

    %% Final JSON object
    JsonObj = _{
        architecture: Arch,
        n_layers: NLayers,
        n_heads: NHeads,
        n_kv_heads: NKVHeads,
        head_dim: HeadDim,
        embed_dim: EmbedDim,
        ffn_dim: FfnDim,
        vocab_size: VocabSize,
        rms_eps: RmsEps,
        rope_base: RopeBase,
        rope_dim: RopeDim,
        max_context: MaxContext,
        bos_token_id: BosId,
        eos_token_id: EosId,
        tensors: TensorJsons,
        tokens: TokenRecords,
        merges: MergesList,
        data_start: DataStart
    },

    %% Write JSON
    open(OutPath, write, Stream),
    json_write(Stream, JsonObj, [width(0)]),
    close(Stream),
    format('wrote manifest: ~w (~w tensors, ~w tokens, ~w merges)~n',
           [OutPath, length(TensorJsons), VocabSize, length(MergesList)]),
    halt(0).

%% metadata_value(+Metadata, +Key, -Value, +Default)
metadata_value(Metadata, Key, Value, _Default) :-
    member(Key-V, Metadata), !,
    Value = V.
metadata_value(_, _, Default, Default).

%% Build [{"id": 0, "text": "<|begin_of_text|>"}, ...]
build_token_records([], _, []).
build_token_records([T|Rest], Idx, [_{id: Idx, text: T}|RestRecs]) :-
    Next is Idx + 1,
    build_token_records(Rest, Next, RestRecs).

%% Compute byte size per ggml type
compute_size(0, Dims, Size) :- elem_count(Dims, N), Size is N * 4.    %% F32
compute_size(1, Dims, Size) :- elem_count(Dims, N), Size is N * 2.    %% F16
compute_size(8, Dims, Size) :-   %% Q8_0
    elem_count(Dims, N),
    Blocks is N // 32,
    Size is Blocks * 34.
compute_size(12, Dims, Size) :-  %% Q4_K
    elem_count(Dims, N),
    Blocks is N // 256,
    Size is Blocks * 144.
compute_size(_, Dims, Size) :- elem_count(Dims, Size).  %% fallback

elem_count([], 1).
elem_count([D|Ds], N) :- elem_count(Ds, M), N is D * M.
