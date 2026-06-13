%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% llamatov_llama.pl — Prolog-maximal LlamaTov runner for LLAMA family
%%
%% Companion to llamatov_v2.pl (GPT-2 family). This file handles
%% Llama-architecture models: TinyLlama, Llama 1/2/3, Mistral, Qwen,
%% etc. — anything with separate Q/K/V matrices, RMSNorm, RoPE, SwiGLU,
%% and optionally Grouped-Query Attention.
%%
%% Per Heath's "maximize Prolog" directive: every layer expressed
%% explicitly in Prolog. Python only dispatches the actual GPU ops
%% (rms_norm, linear_no_bias, apply_rope, llama_causal_attention,
%%  swiglu_ffn) via py_call to llamatov_helpers.py.
%%
%% Architecture detection: the manifest's "general.architecture"
%% metadata determines which runner to use. Llama-family architectures
%% include: llama, mistral, qwen, qwen2, etc.

:- use_module(library(janus)).
:- py_add_lib_dir('lib').
:- use_module('lib/tensor_loader_adapter').

%% Manifest facts loaded via consult.
:- dynamic(tensor/8).
:- dynamic(gguf_header/4).
:- dynamic(metadata/3).
:- dynamic(tensor_data_section_start/1).

%% ────────────────────────────────────────────────────────────────────
%% Top-level entry: run/3
%% ────────────────────────────────────────────────────────────────────

run(GgufPath, InputTokenIds, OutputToken) :-
    %% Step 1: ensure manifest exists
    ensure_manifest(GgufPath, ManifestPath),
    consult(ManifestPath),

    %% Step 2: read architecture parameters from metadata
    architecture(Arch),
    block_count(NLayers),
    head_count(NHeads),
    head_count_kv(NHeadsKv),
    embedding_length(EmbedDim),
    rope_freq_base(RopeFreqBase),
    rms_norm_eps(RmsEps),
    format("Architecture: ~w~n", [Arch]),
    format("  Layers:        ~d~n", [NLayers]),
    format("  Heads (Q):     ~d~n", [NHeads]),
    format("  Heads (KV):    ~d  (", [NHeadsKv]),
    gqa_or_mha(NHeads, NHeadsKv),
    format(")~n"),
    format("  Embed dim:     ~d~n", [EmbedDim]),
    format("  RoPE freq base: ~w~n", [RopeFreqBase]),
    format("  RMSNorm eps:   ~w~n", [RmsEps]),

    HeadDim is EmbedDim // NHeads,
    SeqLen is 64,   %% Precompute RoPE for sequences up to this length
    format("  Head dim:      ~d~n", [HeadDim]),

    %% Step 3: load token embedding
    format("Loading embeddings...~n"),
    load_weight(GgufPath, "token_embd.weight", TokEmbd),

    %% Step 4: embed input tokens (no separate position embedding)
    py_call(llamatov_helpers:embed_tokens_no_position(TokEmbd, InputTokenIds),
            X0),

    %% Step 5: precompute RoPE cos/sin tables
    py_call(llamatov_helpers:precompute_rope_cos_sin(SeqLen, HeadDim,
                                                      RopeFreqBase),
            RopeTuple),
    %% janus 2-tuple: cos and sin tensors
    RopeTuple = (Cos - Sin),

    %% Step 6: run all layers
    format("Running ~d layers...~n", [NLayers]),
    run_layers(GgufPath, NHeads, NHeadsKv, RmsEps, Cos, Sin,
               NLayers, 0, X0, XFinal),

    %% Step 7: final RMSNorm
    format("Final RMSNorm + output projection...~n"),
    load_weight(GgufPath, "output_norm.weight", FinalNormW),
    py_call(llamatov_helpers:rms_norm(XFinal, FinalNormW, RmsEps), XNormed),

    %% Step 8: output projection
    %% Some Llama models tie output.weight to token_embd.weight; otherwise
    %% there's a separate output.weight.
    ( tensor(name("output.weight"), _, _, _, _, _, _, _)
    -> load_weight(GgufPath, "output.weight", OutW)
    ;  %% Tied embedding: use token_embd.weight transposed
       py_call(llamatov_helpers:transpose(TokEmbd), OutW)
    ),
    py_call(llamatov_helpers:matmul(XNormed, OutW), Logits),

    %% Step 9: argmax of last token's logits
    py_call(llamatov_helpers:argmax_last(Logits), OutputToken),
    format("Output token: ~w~n", [OutputToken]).

%% ────────────────────────────────────────────────────────────────────
%% Architecture metadata accessors
%% ────────────────────────────────────────────────────────────────────

architecture(Arch) :-
    metadata("general.architecture", _, ArchStr),
    atom_string(Arch, ArchStr).

metadata_int(Suffix, Value) :-
    architecture(Arch),
    atom_concat(Arch, Suffix, KeyAtom),
    atom_string(KeyAtom, Key),
    metadata(Key, _, Value).

metadata_float(Suffix, Value) :-
    architecture(Arch),
    atom_concat(Arch, Suffix, KeyAtom),
    atom_string(KeyAtom, Key),
    metadata(Key, _, Value).

block_count(N) :- metadata_int('.block_count', N).
head_count(N)  :- metadata_int('.attention.head_count', N).
embedding_length(N) :- metadata_int('.embedding_length', N).

%% GQA: head_count_kv defaults to head_count if not specified
head_count_kv(N) :-
    metadata_int('.attention.head_count_kv', N), !.
head_count_kv(N) :-
    head_count(N).

%% RoPE base frequency: defaults to 10000.0 if not specified
rope_freq_base(F) :-
    metadata_float('.rope.freq_base', F), !.
rope_freq_base(10000.0).

%% RMSNorm epsilon: defaults to 1.0e-5
rms_norm_eps(E) :-
    metadata_float('.attention.layer_norm_rms_epsilon', E), !.
rms_norm_eps(1.0e-5).

%% gqa_or_mha/2 — return the atom 'gqa' or 'mha' based on head counts.
%% Used as a predicate, but written here as a function-like helper.
gqa_or_mha(N, N) :- !, write(mha).
gqa_or_mha(_, _) :- write(gqa).

%% ────────────────────────────────────────────────────────────────────
%% Manifest management (shared with v2)
%% ────────────────────────────────────────────────────────────────────

ensure_manifest(GgufPath, ManifestPath) :-
    file_base_name(GgufPath, Base),
    format(atom(ManifestPath), '/tmp/~w.manifest.pl', [Base]),
    ( exists_file(ManifestPath)
    -> format("Using existing manifest: ~w~n", [ManifestPath])
    ;  format("Emitting manifest: ~w~n", [ManifestPath]),
       emit_manifest_external(GgufPath, ManifestPath)
    ).

emit_manifest_external(GgufPath, ManifestPath) :-
    format(atom(Cmd),
           'cd <repo>/must_close/boundary_dsl && swipl -q -g "consult(gguf_emit_manifest), emit_manifest(\'~w\', \'~w\'), halt"',
           [GgufPath, ManifestPath]),
    shell(Cmd, _).

%% ────────────────────────────────────────────────────────────────────
%% Weight loading (uses adapter + quantized helpers)
%% ────────────────────────────────────────────────────────────────────

load_weight(GgufPath, Name, Tensor) :-
    %% Manifest fields are wrapped: dimensions(D), type_code(T),
    %% file_offset(O), etc. We unwrap here to get raw values for the
    %% Python helper, and pass the WRAPPED fact to the adapter.
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

quantized_type(2).    % Q4_0
quantized_type(10).   % Q2_K
quantized_type(11).   % Q3_K
quantized_type(12).   % Q4_K
quantized_type(14).   % Q6_K

n_elements([], 1).
n_elements([D | Rest], N) :-
    n_elements(Rest, RestN),
    N is D * RestN.

%% ────────────────────────────────────────────────────────────────────
%% Layer iteration (recursive Prolog)
%% ────────────────────────────────────────────────────────────────────

run_layers(_, _, _, _, _, _, NLayers, I, Cur, Cur) :-
    I >= NLayers, !.
run_layers(GgufPath, NHeads, NHeadsKv, RmsEps, Cos, Sin,
           NLayers, I, Cur, Final) :-
    I < NLayers,
    ( 0 =:= I mod 4
    -> format("  Layer ~d~n", [I])
    ;  true
    ),
    run_one_llama_layer(GgufPath, I, NHeads, NHeadsKv, RmsEps,
                         Cos, Sin, Cur, NextCur),
    I1 is I + 1,
    run_layers(GgufPath, NHeads, NHeadsKv, RmsEps, Cos, Sin,
               NLayers, I1, NextCur, Final).

%% ────────────────────────────────────────────────────────────────────
%% Single Llama layer
%% ────────────────────────────────────────────────────────────────────

run_one_llama_layer(GgufPath, I, NHeads, NHeadsKv, RmsEps,
                     Cos, Sin, X, Output) :-
    format(atom(Prefix), 'blk.~d', [I]),
    atom_string(Prefix, PrefixStr),

    %% Pre-attention RMSNorm
    weight_name(PrefixStr, ".attn_norm.weight", AttnNormName),
    load_weight(GgufPath, AttnNormName, AttnNormW),
    py_call(llamatov_helpers:rms_norm(X, AttnNormW, RmsEps), Normed1),

    %% Separate Q, K, V projections (Llama-style)
    weight_name(PrefixStr, ".attn_q.weight", QName),
    weight_name(PrefixStr, ".attn_k.weight", KName),
    weight_name(PrefixStr, ".attn_v.weight", VName),
    load_weight(GgufPath, QName, QW),
    load_weight(GgufPath, KName, KW),
    load_weight(GgufPath, VName, VW),
    py_call(llamatov_helpers:linear_no_bias(Normed1, QW), QProj),
    py_call(llamatov_helpers:linear_no_bias(Normed1, KW), KProj),
    py_call(llamatov_helpers:linear_no_bias(Normed1, VW), VProj),

    %% Split into per-head
    py_call(llamatov_helpers:llama_qkv_split(QProj, KProj, VProj,
                                               NHeads, NHeadsKv),
            QkvTuple),
    QkvTuple =.. ['-', Q, K, V],

    %% Apply RoPE to Q and K
    py_call(llamatov_helpers:apply_rope(Q, Cos, Sin), QRope),
    py_call(llamatov_helpers:apply_rope(K, Cos, Sin), KRope),

    %% Causal attention with GQA expansion
    py_call(llamatov_helpers:llama_causal_attention(
                QRope, KRope, V, NHeads, NHeadsKv),
            AttOut),

    %% Merge heads
    py_call(llamatov_helpers:merge_heads(AttOut, NHeads), AttMerged),

    %% Output projection
    weight_name(PrefixStr, ".attn_output.weight", OutName),
    load_weight(GgufPath, OutName, OutW),
    py_call(llamatov_helpers:linear_no_bias(AttMerged, OutW), AttProj),

    %% First residual
    py_call(llamatov_helpers:add_tensors(X, AttProj), XPostAttn),

    %% Pre-FFN RMSNorm
    weight_name(PrefixStr, ".ffn_norm.weight", FfnNormName),
    load_weight(GgufPath, FfnNormName, FfnNormW),
    py_call(llamatov_helpers:rms_norm(XPostAttn, FfnNormW, RmsEps),
            Normed2),

    %% SwiGLU FFN
    weight_name(PrefixStr, ".ffn_gate.weight", GateName),
    weight_name(PrefixStr, ".ffn_up.weight",   UpName),
    weight_name(PrefixStr, ".ffn_down.weight", DownName),
    load_weight(GgufPath, GateName, GateW),
    load_weight(GgufPath, UpName, UpW),
    load_weight(GgufPath, DownName, DownW),
    py_call(llamatov_helpers:swiglu_ffn(Normed2, GateW, UpW, DownW),
            FfnOut),

    %% Second residual
    py_call(llamatov_helpers:add_tensors(XPostAttn, FfnOut), Output).

weight_name(Prefix, Suffix, FullName) :-
    string_concat(Prefix, Suffix, FullName).
