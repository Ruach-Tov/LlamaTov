%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% llamatov_v2.pl — Prolog-maximal LlamaTov runner
%%
%% Per Heath's "maximize Prolog" directive: the SECOND version of the
%% runner, eliminating the builtins:exec fragility of llamatov.pl (commit
%% cce8c99ad) and using the existing Prolog GGUF substrate.
%%
%% ARCHITECTURE:
%%   - Prolog parses GGUF (via gguf_emit_manifest.pl → manifest facts)
%%   - Prolog orchestrates every layer (atom_concat builds weight names,
%%     load_weight dispatches by manifest fact)
%%   - Python helpers in bpd/lib/llamatov_helpers.py (regular module, NOT
%%     embedded via builtins:exec) execute GPU operations
%%   - Final argmax dispatched (returns Python int)
%%
%% NO builtins:exec. NO embedded Python definitions. The Python surface
%% is a regular module file loaded via py_add_lib_dir.
%%
%% This composes with the reference implementation in
%% bpd/tests/spike_prolog_maximal_loading.pl (commit e6d684a77).

:- use_module(library(janus)).
:- py_add_lib_dir('lib').
:- use_module('lib/tensor_loader_adapter').

%% Manifest facts loaded into a private namespace via consult/2.
:- dynamic(tensor/8).
:- dynamic(gguf_header/4).
:- dynamic(metadata/3).
:- dynamic(tensor_data_section_start/1).

%% ────────────────────────────────────────────────────────────────────
%% Top-level entry point
%% ────────────────────────────────────────────────────────────────────

%% run(+GgufPath, +InputTokenIds, -OutputToken)
%%   Run end-to-end inference. Loads the GGUF, runs all layers, returns
%%   the argmax of the final token's logits.
%%
%%   GgufPath: path to GGUF file
%%   InputTokenIds: list of input token IDs (e.g., [15496, 11] for "Hello,")
%%   OutputToken: predicted next token ID (Python int)
%%
%%   Note: requires a manifest file already produced via:
%%     swipl gguf_emit_manifest.pl <gguf_path> <manifest_path>
%%   then consulted before calling run/3, OR run/3 will emit one
%%   on-the-fly to a tmp location.

run(GgufPath, InputTokenIds, OutputToken) :-
    %% Step 1: ensure manifest exists (emit if needed)
    ensure_manifest(GgufPath, ManifestPath),
    consult(ManifestPath),

    %% Step 2: read architecture parameters from metadata
    architecture(Arch),
    block_count(NLayers),
    head_count(NHeads),
    format("Architecture: ~w  layers: ~d  heads: ~d~n",
           [Arch, NLayers, NHeads]),

    %% Step 3: load embedding tensors
    format("Loading embeddings...~n"),
    load_weight(GgufPath, "token_embd.weight", TokEmbd),
    load_weight(GgufPath, "position_embd.weight", PosEmbd),

    %% Step 4: embed input tokens
    py_call(llamatov_helpers:transpose(TokEmbd), TokEmbdT),
    py_call(llamatov_helpers:transpose(PosEmbd), PosEmbdT),
    py_call(llamatov_helpers:embed_tokens(TokEmbdT, PosEmbdT, InputTokenIds), X0),

    %% Step 5: run all transformer layers
    format("Running ~d layers...~n", [NLayers]),
    run_layers(GgufPath, NHeads, NLayers, 0, X0, XFinal),

    %% Step 6: final layer norm + output projection
    format("Final norm + output projection...~n"),
    load_weight(GgufPath, "output_norm.weight", OutNormW),
    load_weight(GgufPath, "output_norm.bias", OutNormB),
    py_call(llamatov_helpers:layer_norm(XFinal, OutNormW, OutNormB), XNormed),
    load_weight(GgufPath, "output.weight", OutW),
    py_call(llamatov_helpers:matmul(XNormed, OutW), Logits),

    %% Step 7: argmax of last token's logits
    py_call(llamatov_helpers:argmax_last(Logits), OutputToken),
    format("Output token: ~w~n", [OutputToken]).

%% ────────────────────────────────────────────────────────────────────
%% Manifest management
%% ────────────────────────────────────────────────────────────────────

%% ensure_manifest(+GgufPath, -ManifestPath)
%%   Emits a manifest to /tmp if one doesn't exist for this GGUF.
ensure_manifest(GgufPath, ManifestPath) :-
    file_base_name(GgufPath, Base),
    format(atom(ManifestPath), '/tmp/~w.manifest.pl', [Base]),
    ( exists_file(ManifestPath)
    -> format("Using existing manifest: ~w~n", [ManifestPath])
    ;  format("Emitting manifest: ~w~n", [ManifestPath]),
       %% Call the emitter (uses must_close/boundary_dsl path)
       emit_manifest_external(GgufPath, ManifestPath)
    ).

%% Stub: shells out to swipl with gguf_emit_manifest.
%% (Could also use process_create or python; this is the simplest
%% path with predictable behavior.)
emit_manifest_external(GgufPath, ManifestPath) :-
    format(atom(Cmd),
           'cd <repo>/must_close/boundary_dsl && swipl -q -g "consult(gguf_emit_manifest), emit_manifest(\'~w\', \'~w\'), halt"',
           [GgufPath, ManifestPath]),
    shell(Cmd, _).

%% ────────────────────────────────────────────────────────────────────
%% Architecture parameter accessors (from metadata facts)
%% ────────────────────────────────────────────────────────────────────

architecture(Arch) :-
    metadata("general.architecture", _, ArchStr),
    atom_string(Arch, ArchStr).

block_count(N) :-
    architecture(Arch),
    atom_concat(Arch, '.block_count', Key),
    atom_string(Key, KeyStr),
    metadata(KeyStr, _, N).

head_count(N) :-
    architecture(Arch),
    atom_concat(Arch, '.attention.head_count', Key),
    atom_string(Key, KeyStr),
    metadata(KeyStr, _, N).

%% ────────────────────────────────────────────────────────────────────
%% Weight loading by name (uses tensor_loader_adapter)
%% ────────────────────────────────────────────────────────────────────

load_weight(GgufPath, Name, Tensor) :-
    %% Manifest fields are wrapped: dimensions(D), type_code(T),
    %% file_offset(O), etc. Unwrap here for the helper; pass the
    %% wrapped fact to the adapter for non-quantized loads.
    tensor(name(Name), dimensions(Dims), type_code(TypeCode), TypeNameTerm,
           NumpyDtypeTerm, file_offset(FileOffset), ByteSizeTerm, RelOffsetTerm),
    ( quantized_type(TypeCode)
    -> n_elements(Dims, NEl),
       py_call(llamatov_helpers:load_tensor_by_type(
                   GgufPath, FileOffset, NEl, Dims, TypeCode),
               Tensor)
    ;  TensorFact = tensor(name(Name), dimensions(Dims), type_code(TypeCode),
                            TypeNameTerm, NumpyDtypeTerm,
                            file_offset(FileOffset), ByteSizeTerm, RelOffsetTerm),
       tensor_loader_adapter:load_tensor_from_manifest(GgufPath, TensorFact, Tensor)
    ).

quantized_type(10).   % Q2_K
quantized_type(11).   % Q3_K
quantized_type(14).   % Q6_K

n_elements([], 1).
n_elements([D | Rest], N) :-
    n_elements(Rest, RestN),
    N is D * RestN.

%% ────────────────────────────────────────────────────────────────────
%% Layer iteration (recursive Prolog)
%% ────────────────────────────────────────────────────────────────────

run_layers(_, _, NLayers, I, Cur, Cur) :-
    I >= NLayers, !.
run_layers(GgufPath, NHeads, NLayers, I, Cur, Final) :-
    I < NLayers,
    ( 0 =:= I mod 4
    -> format("  Layer ~d~n", [I])
    ;  true
    ),
    run_one_layer(GgufPath, I, NHeads, Cur, NextCur),
    I1 is I + 1,
    run_layers(GgufPath, NHeads, NLayers, I1, NextCur, Final).

%% ────────────────────────────────────────────────────────────────────
%% Single layer (GPT-2 style: norm → QKV → attn → FFN → residual)
%% ────────────────────────────────────────────────────────────────────

run_one_layer(GgufPath, I, NHeads, X, Output) :-
    %% Build weight name prefix
    format(atom(Prefix), 'blk.~d', [I]),
    atom_string(Prefix, PrefixStr),

    %% Attn norm + bias
    weight_name(PrefixStr, ".attn_norm.weight", AttnNormWName),
    weight_name(PrefixStr, ".attn_norm.bias",   AttnNormBName),
    load_weight(GgufPath, AttnNormWName, AttnNormW),
    load_weight(GgufPath, AttnNormBName, AttnNormB),
    py_call(llamatov_helpers:layer_norm(X, AttnNormW, AttnNormB), LN1),

    %% QKV projection (combined matmul)
    weight_name(PrefixStr, ".attn_qkv.weight", QkvWName),
    weight_name(PrefixStr, ".attn_qkv.bias",   QkvBName),
    load_weight(GgufPath, QkvWName, QkvW),
    load_weight(GgufPath, QkvBName, QkvB),
    py_call(llamatov_helpers:linear(LN1, QkvW, QkvB), Qkv),

    %% Split QKV → q, k, v per head
    %% Empirical: janus represents Python 3-tuples as -(A, B, C).
    py_call(llamatov_helpers:attention_qkv_split(Qkv, NHeads), QKV_tuple),
    QKV_tuple =.. ['-', Q, K, V],

    %% Causal attention
    py_call(llamatov_helpers:causal_attention(Q, K, V), AttOut),
    py_call(llamatov_helpers:merge_heads(AttOut, NHeads), AttOutMerged),

    %% Output projection
    weight_name(PrefixStr, ".attn_output.weight", AttnOutWName),
    weight_name(PrefixStr, ".attn_output.bias",   AttnOutBName),
    load_weight(GgufPath, AttnOutWName, AttnOutW),
    load_weight(GgufPath, AttnOutBName, AttnOutB),
    py_call(llamatov_helpers:linear(AttOutMerged, AttnOutW, AttnOutB), AttnOutProj),

    %% First residual
    py_call(llamatov_helpers:add_tensors(X, AttnOutProj), XPostAttn),

    %% FFN norm + bias
    weight_name(PrefixStr, ".ffn_norm.weight", FfnNormWName),
    weight_name(PrefixStr, ".ffn_norm.bias",   FfnNormBName),
    load_weight(GgufPath, FfnNormWName, FfnNormW),
    load_weight(GgufPath, FfnNormBName, FfnNormB),
    py_call(llamatov_helpers:layer_norm(XPostAttn, FfnNormW, FfnNormB), LN2),

    %% FFN up
    weight_name(PrefixStr, ".ffn_up.weight", FfnUpWName),
    weight_name(PrefixStr, ".ffn_up.bias",   FfnUpBName),
    load_weight(GgufPath, FfnUpWName, FfnUpW),
    load_weight(GgufPath, FfnUpBName, FfnUpB),
    py_call(llamatov_helpers:linear(LN2, FfnUpW, FfnUpB), FfnUpResult),
    py_call(llamatov_helpers:gelu(FfnUpResult), FfnUpActivated),

    %% FFN down
    weight_name(PrefixStr, ".ffn_down.weight", FfnDownWName),
    weight_name(PrefixStr, ".ffn_down.bias",   FfnDownBName),
    load_weight(GgufPath, FfnDownWName, FfnDownW),
    load_weight(GgufPath, FfnDownBName, FfnDownB),
    py_call(llamatov_helpers:linear(FfnUpActivated, FfnDownW, FfnDownB), FfnOut),

    %% Second residual
    py_call(llamatov_helpers:add_tensors(XPostAttn, FfnOut), Output).

weight_name(Prefix, Suffix, FullName) :-
    string_concat(Prefix, Suffix, FullName).
