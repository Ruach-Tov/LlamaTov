%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% model_zoo.pl — Knowledge resource for our local Model Zoo coverage.
%%
%% Per Heath's directive (2026-05-16 ~02:50 UTC): build an information
%% resource for future agents that answers "what coverage of the 124-127
%% lifted architectures do we have GGUFs for, and what's missing?"
%%
%% This is a first-principle calculation, not test-driven inference:
%%   - Lift the canonical arch table from llama.cpp source (ground truth)
%%   - Scan the on-disk Model Zoo (Ollama blobs + any other GGUFs)
%%   - Read general.architecture from each GGUF
%%   - Join: report covered, count by arch, list missing archs
%%
%% Usage patterns:
%%   ?- model_zoo:list_zoo(Models).
%%   ?- model_zoo:coverage_report.
%%   ?- model_zoo:missing_archs(Missing).
%%   ?- model_zoo:huggingface_source_hint(Arch, Hint).
%%
%% Author: metayen 2026-05-16
%% Composes with: llama_cpp_lifter.pl (Phase 1+2+3c arch enumeration)

:- module(model_zoo, [
    list_zoo/1,                   % -Models: [zoo_model(Name, Arch, SizeBytes, BlobPath)]
    scan_ollama_blobs/1,          % -Models: scan via Ollama manifest
    gguf_architecture/2,          % +BlobPath, -Arch (reads general.architecture)
    coverage_report/0,            %  Prints the substantive coverage report
    missing_archs/1,              % -Archs: list of arches with no zoo representative
    covered_archs/1,              % -Archs: list of arches with >= 1 representative
    arch_representatives/2,       % +Arch, -Models: zoo models for that arch
    huggingface_source_hint/2,    % +Arch, -Hint (where to find a GGUF for this arch)
    canonicalize_arch/2,          % +AnyArchName, -LlamaCppCanonical
    alias_resolution/2,           % back-compat alias for canonicalize_arch/2
    terminology_change/2,         % world:Name vs llamacpp:Name
    graph_aliased/2               % llamacpp:Child uses llamacpp:Parent's graph
]).

:- use_module(library(filesex)).
:- use_module(library(http/json)).
:- use_module(library(readutil)).
:- use_module('lib/gguf_native_reader').
:- use_module('lib/gguf_arch_fast').
:- use_module(library(lists)).
:- use_module(llama_cpp_lifter).

%% ═════════════════════════════════════════════════════════════════════
%% Configuration
%% ═════════════════════════════════════════════════════════════════════

ollama_root(Root) :- getenv('OLLAMA_MODELS', Root), !.
ollama_root('/usr/share/ollama/.ollama/models').
ollama_blobs_dir(Path) :-
    ollama_root(Root),
    directory_file_path(Root, blobs, Path).
ollama_manifests_root(Path) :-
    ollama_root(Root),
    atomic_list_concat([Root, '/manifests/registry.ollama.ai/library'], Path).

llama_cpp_root('../external/llama.cpp').


%% ═════════════════════════════════════════════════════════════════════
%% GGUF architecture extraction
%% ═════════════════════════════════════════════════════════════════════
%%
%% GGUF format: 4-byte magic ("GGUF"), 4-byte version, 8-byte tensor_count,
%% 8-byte kv_count, then KV pairs as length-prefixed strings + typed values.
%%
%% We use a substrate-honest shortcut: the first ~32KB of any non-corrupt
%% GGUF file contains general.architecture as one of the first KV pairs.
%% We extract it via shell `strings` since the value is plain UTF-8 and
%% comes immediately after the literal key "general.architecture".

%% Native Prolog GGUF reader — no shell-out, byte-ownership tracked.
%% Uses safe_read.pl for crossword-puzzle defense.
gguf_architecture(BlobPath, Arch) :-
    exists_file(BlobPath),
    catch(
        gguf_arch_only:gguf_architecture_fast(BlobPath, Arch),
        _Error,
        %% Fallback to shell method if native reader fails
        %% (e.g., for GGUF versions we don't yet support)
        gguf_architecture_shell(BlobPath, Arch)
    ).

%% Shell fallback (legacy — will be removed once native reader handles all formats)
gguf_architecture_shell(BlobPath, Arch) :-
    setup_call_cleanup(
        process_create(path(sh),
            ['-c', "head -c 32768 \"$0\" | strings -n 3 | awk '/^general\\.architecture$/ {getline; print; exit}'", BlobPath],
            [stdout(pipe(Out)), process(_PID)]),
        ( read_string(Out, _, ArchStr),
          string_concat(ArchTrim, "\n", ArchStr),
          ArchTrim \= "",
          atom_string(ArchAtom, ArchTrim),
          dash_to_underscore(ArchAtom, Arch)
        ),
        close(Out)
    ).

dash_to_underscore(In, Out) :-
    atom_chars(In, Chars),
    maplist([C, D]>>(C = '-' -> D = '_' ; D = C), Chars, NewChars),
    atom_chars(Out, NewChars).


%% ═════════════════════════════════════════════════════════════════════
%% Ollama manifest scanning
%% ═════════════════════════════════════════════════════════════════════
%%
%% Ollama manifests live at:
%%   ~/.ollama/models/manifests/registry.ollama.ai/library/<model>/<tag>
%% Each manifest is a JSON file with a "layers" array containing
%% mediaType + digest pointing to a blob in ~/.ollama/models/blobs/.
%% The model weights are in the layer with mediaType
%%   application/vnd.ollama.image.model

scan_ollama_blobs(Models) :-
    ollama_manifests_root(MRoot),
    ( exists_directory(MRoot)
    -> directory_files(MRoot, Entries),
       findall(zoo_model(Name, Arch, Size, BlobPath),
           ( member(Name, Entries),
             \+ member(Name, ['.', '..']),
             directory_file_path(MRoot, Name, ModelDir),
             exists_directory(ModelDir),
             directory_files(ModelDir, Tags),
             member(Tag, Tags),
             \+ member(Tag, ['.', '..']),
             directory_file_path(ModelDir, Tag, ManifestPath),
             exists_file(ManifestPath),
             ollama_model_blob(ManifestPath, BlobPath),
             size_file(BlobPath, Size),
             ( gguf_architecture(BlobPath, Arch)
             -> true
             ;  Arch = unknown
             )
           ),
           Models)
    ;  Models = []
    ).

%% Read manifest JSON, find the "model" layer, return absolute blob path.
ollama_model_blob(ManifestPath, BlobPath) :-
    setup_call_cleanup(
        open(ManifestPath, read, Stream),
        json_read_dict(Stream, Manifest),
        close(Stream)
    ),
    get_dict(layers, Manifest, Layers),
    member(Layer, Layers),
    get_dict(mediaType, Layer, "application/vnd.ollama.image.model"),
    get_dict(digest, Layer, DigestStr),
    %% Convert "sha256:abc..." → "sha256-abc..." (Ollama's blob naming)
    atom_string(DigestAtom, DigestStr),
    atom_chars(DigestAtom, DChars),
    maplist([C, D]>>(C = ':' -> D = '-' ; D = C), DChars, NDChars),
    atom_chars(BlobName, NDChars),
    ollama_blobs_dir(BlobsDir),
    directory_file_path(BlobsDir, BlobName, BlobPath),
    !.   % first model layer only


%% ═════════════════════════════════════════════════════════════════════
%% Top-level zoo listing
%% ═════════════════════════════════════════════════════════════════════
%%
%% list_zoo/1 unifies Ollama-pulled models with any standalone GGUFs.
%% For now this is Ollama-only; extend with additional sources later.

list_zoo(Models) :-
    scan_ollama_blobs(Models).


%% ═════════════════════════════════════════════════════════════════════
%% Architecture relationships — two distinct kinds
%% ═════════════════════════════════════════════════════════════════════
%%
%% Per Heath's substrate-honest precision directive (2026-05-16 ~03:15 UTC):
%% conflating these as "arch_alias" loses information about WHO says WHAT
%% to mean WHAT. We separate them by namespace:
%%
%% KIND 1 — terminology_change/2: cross-world naming
%%   The SAME architecture has different names in different vocabularies.
%%   Format: terminology_change(world:GgufName, llamacpp:CppName)
%%   Example: terminology_change(world:gptoss, llamacpp:openai_moe)
%%     "The GGUF metadata key general.architecture says 'gptoss', but
%%      llama.cpp's internal enum name (LLM_ARCH_OPENAI_MOE) is
%%      openai_moe. Same architecture, two namespaces' names for it."
%%
%% KIND 2 — graph_aliased/2: structural graph reuse within llama.cpp
%%   Within llama.cpp, a derived arch's `using graph = ParentClass::graph;`
%%   declaration means it IS structurally the parent's graph builder.
%%   Format: graph_aliased(llamacpp:Child, llamacpp:Parent)
%%   Example: graph_aliased(llamacpp:jina_bert_v2, llamacpp:bert)
%%     "In llama.cpp's world, jina_bert_v2's graph is bert's graph."
%%
%% These are substantively different relationships and the paper's
%% subsumption claim depends on tracking them separately. Terminology
%% changes are translation-at-boundary; graph aliases are structural
%% subsumption-by-construction.

%% KIND 1 facts: terminology changes across vocabularies.
%% Currently only one known mismatch between GGUF and llama.cpp:
terminology_change(world:gptoss, llamacpp:openai_moe).

%% KIND 2 facts: graph-level structural aliasing within llama.cpp.
%% Lifted from src/models/models.h `using graph = ...` declarations.
%% (Hand-curated bootstrap; long-term auto-generated from
%% llama_cpp_lifter:lift_graph_aliases/2.)
graph_aliased(llamacpp:jina_bert_v2, llamacpp:bert).
graph_aliased(llamacpp:jina_bert_v3, llamacpp:bert).
graph_aliased(llamacpp:nomic_bert, llamacpp:bert).
graph_aliased(llamacpp:nomic_bert_moe, llamacpp:bert).
graph_aliased(llamacpp:phimoe, llamacpp:phi3).
graph_aliased(llamacpp:llama_embed, llamacpp:llama).
graph_aliased(llamacpp:deepseek2ocr, llamacpp:deepseek2).
graph_aliased(llamacpp:glm_dsa, llamacpp:deepseek2).
graph_aliased(llamacpp:mistral4, llamacpp:deepseek2).
graph_aliased(llamacpp:t5encoder, llamacpp:t5).
graph_aliased(llamacpp:nemotron_h_moe, llamacpp:nemotron_h).
graph_aliased(llamacpp:granite_moe, llamacpp:granite).
graph_aliased(llamacpp:minicpm, llamacpp:granite).
graph_aliased(llamacpp:hunyuan_dense, llamacpp:hunyuan_vl).
graph_aliased(llamacpp:lfm2moe, llamacpp:lfm2).
graph_aliased(llamacpp:mamba2, llamacpp:mamba).

%% Iyun 2026-05-31 (Heath terminology_change directive): mistral and llama3 are
%% SYNONYMOUS jargon for the llama graph. llama.cpp has NO separate LLM_ARCH_MISTRAL
%% -- mistral GGUFs report general.architecture="llama" and build via llm_build_llama
%% (llama-model.cpp:4475, dispatch at :12903 case LLM_ARCH_LLAMA). Verified by source
%% inspection: the build switch routes LLAMA (no MISTRAL case exists). So the heatmap's
%% [mistral, ...] coordinates ARE [llama, ...] coordinates; Iyun's llama3 0-ULP
%% reproduction (full 16-layer) verifies the SAME graph.
graph_aliased(llamacpp:mistral, llamacpp:llama).
graph_aliased(llamacpp:llama3,  llamacpp:llama).
graph_aliased(llamacpp:llama32, llamacpp:llama).


%% canonicalize_arch/2: given an arch name (from any source), return its
%% llama.cpp canonical name after applying BOTH KINDS of relationships
%% in order: first translate vocabulary (world: → llamacpp:), then
%% resolve graph aliasing within llamacpp.
%%
%% Heath's directive substantively makes this composition explicit.
canonicalize_arch(Arch, Canonical) :-
    %% Step 1: vocabulary translation (world → llamacpp if needed)
    ( terminology_change(world:Arch, llamacpp:Translated)
    -> InLlamaCpp = Translated
    ;  InLlamaCpp = Arch
    ),
    %% Step 2: graph-aliasing resolution within llamacpp
    ( graph_aliased(llamacpp:InLlamaCpp, llamacpp:Canonical)
    -> true
    ;  Canonical = InLlamaCpp
    ),
    !.

%% Back-compat alias for the old name (used by callers below).
alias_resolution(Arch, Canonical) :- canonicalize_arch(Arch, Canonical).


%% ═════════════════════════════════════════════════════════════════════
%% Coverage queries
%% ═════════════════════════════════════════════════════════════════════

%% covered_archs/1: which canonical arches have at least one zoo model?
covered_archs(Archs) :-
    list_zoo(Models),
    findall(Canonical,
        ( member(zoo_model(_, RawArch, _, _), Models),
          alias_resolution(RawArch, Canonical),
          Canonical \= unknown
        ),
        Raw),
    sort(Raw, Archs).

%% missing_archs/1: which canonical arches in llama.cpp have NO zoo model?
%% Uses the dispatch table (Phase 2) as the universe of "things we care
%% about" — the 124 archs with real builder classes.
missing_archs(Missing) :-
    llama_cpp_root(Root),
    atomic_list_concat([Root, '/src/llama-model.cpp'], ModelCpp),
    lift_dispatch_table(ModelCpp, Pairs),
    findall(A, member(arch_class(A, _), Pairs), AllRaw),
    %% Canonicalize the universe through aliasing
    findall(C, ( member(A, AllRaw), alias_resolution(A, C) ), CanonRaw),
    sort(CanonRaw, AllCanon),
    covered_archs(Covered),
    subtract(AllCanon, Covered, Missing).

%% arch_representatives/2: which zoo models map to this canonical arch?
arch_representatives(Arch, Models) :-
    list_zoo(All),
    findall(Name-Size,
        ( member(zoo_model(Name, RawArch, Size, _), All),
          alias_resolution(RawArch, Arch)
        ),
        Models).


%% ═════════════════════════════════════════════════════════════════════
%% HuggingFace source hints (where to find GGUFs for missing arches)
%% ═════════════════════════════════════════════════════════════════════
%%
%% Hand-curated bootstrap. Each entry: huggingface_source_hint(Arch, Hint)
%% where Hint is a typical HF org/user that ships GGUFs for that arch.
%%
%% Coverage priority: most-deployed first. The 124-arch trajectory
%% should prioritize archs by deployment frequency, not enumeration order.

hf_hint_fact(llama,        'bartowski/Meta-Llama-3-8B-Instruct-GGUF').
hf_hint_fact(qwen2,        'Qwen/Qwen2.5-7B-Instruct-GGUF').
hf_hint_fact(qwen3,        'Qwen/Qwen3-8B-GGUF').
hf_hint_fact(gemma,        'google/gemma-7b-GGUF (or bartowski/)').
hf_hint_fact(gemma2,       'google/gemma-2-9b-GGUF').
hf_hint_fact(gemma3,       'google/gemma-3-4b-GGUF').
hf_hint_fact(phi3,         'microsoft/Phi-3-mini-4k-instruct-gguf').
hf_hint_fact(bert,         'leliuga/all-MiniLM-L6-v2-GGUF (sentence-bert)').
hf_hint_fact(nomic_bert,   'nomic-ai/nomic-embed-text-v1.5-GGUF').
hf_hint_fact(modern_bert,  'lightonai/modernbert-embed-large-gguf').
hf_hint_fact(mamba,        'state-spaces/mamba-2.8b-hf-GGUF (community ports)').
hf_hint_fact(mamba2,       'state-spaces/mamba2-2.7b (needs GGUF conversion)').
hf_hint_fact(jamba,        'ai21labs/Jamba-v0.1-GGUF (community)').
hf_hint_fact(rwkv6,        'RWKV/v6-Finch-7B-HF-GGUF').
hf_hint_fact(rwkv7,        'RWKV/v7-Goose-1.5B-HF-GGUF').
hf_hint_fact(mpt,          'mosaicml/mpt-7b (needs GGUF conversion)').
hf_hint_fact(falcon,       'tiiuae/falcon-7b-instruct-GGUF').
hf_hint_fact(falcon_h1,    'tiiuae/Falcon-H1-1B-Instruct-GGUF').
hf_hint_fact(starcoder,    'bigcode/starcoderbase-GGUF').
hf_hint_fact(starcoder2,   'bigcode/starcoder2-7b-GGUF').
hf_hint_fact(gpt2,         'openai-community/gpt2 (needs GGUF conversion via convert_hf_to_gguf.py)').
hf_hint_fact(gptneox,      'EleutherAI/gpt-neox-20b (needs GGUF conversion)').
hf_hint_fact(bloom,        'bigscience/bloom-7b1 (needs GGUF conversion)').
hf_hint_fact(stablelm,     'stabilityai/stablelm-2-1_6b-GGUF').
hf_hint_fact(orion,        'OrionStarAI/Orion-14B-GGUF').
hf_hint_fact(internlm2,    'internlm/internlm2-7b-GGUF').
hf_hint_fact(deepseek2,    'deepseek-ai/DeepSeek-V2-Lite-GGUF').
hf_hint_fact(chatglm,      'THUDM/chatglm3-6b-GGUF').
hf_hint_fact(glm4,         'THUDM/glm-4-9b-chat-GGUF').
hf_hint_fact(command_r,    'CohereForAI/c4ai-command-r-v01-GGUF').
hf_hint_fact(cohere2,      'CohereForAI/c4ai-command-r-08-2024-GGUF').
hf_hint_fact(dbrx,         'databricks/dbrx-instruct (needs GGUF conversion)').
hf_hint_fact(olmo,         'allenai/OLMo-1B-hf-GGUF').
hf_hint_fact(olmoe,        'allenai/OLMoE-1B-7B-0924-GGUF').
hf_hint_fact(openelm,      'apple/OpenELM-1_1B (needs GGUF conversion)').
hf_hint_fact(arctic,       'Snowflake/snowflake-arctic-instruct (needs GGUF conversion)').
hf_hint_fact(t5,           'google/flan-t5-large (needs GGUF conversion)').
hf_hint_fact(jais,         'inceptionai/jais-13b-chat-GGUF').
hf_hint_fact(nemotron,     'nvidia/Nemotron-4-340B-Instruct (needs GGUF conversion)').
hf_hint_fact(exaone,       'LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-GGUF').
hf_hint_fact(granite,      'ibm-granite/granite-3.1-8b-instruct-GGUF').
hf_hint_fact(chameleon,    'facebook/chameleon-7b (needs GGUF conversion)').
hf_hint_fact(grok,         'xai-org/grok-1 (needs GGUF conversion)').
hf_hint_fact(deci,         'Deci/DeciLM-7B (needs GGUF conversion)').
hf_hint_fact(baichuan,     'baichuan-inc/Baichuan-7B (needs GGUF conversion)').
hf_hint_fact(qwen,         'Qwen/Qwen-7B (needs GGUF conversion)').
hf_hint_fact(plamo,        'pfnet/plamo-13b (needs GGUF conversion)').
hf_hint_fact(codeshell,    'WisdomShell/CodeShell-7B-GGUF').
hf_hint_fact(refact,       'smallcloudai/Refact-1_6B-fim-GGUF').
hf_hint_fact(xverse,       'xverse/XVERSE-7B-Chat-GGUF').
hf_hint_fact(openai_moe,   'gpt-oss/* (e.g. via Ollama, what we have)').
hf_hint_fact(qwen2vl,      'Qwen/Qwen2-VL-7B-Instruct-GGUF').
hf_hint_fact(qwen2moe,     'Qwen/Qwen1.5-MoE-A2.7B-Chat-GGUF').
hf_hint_fact(bitnet,       'microsoft/bitnet-b1.58-2B-4T-GGUF').
hf_hint_fact(plamo2,       'pfnet/plamo-2-1b (needs GGUF conversion)').
hf_hint_fact(plamo3,       'pfnet/plamo-3-2b (needs GGUF conversion)').
hf_hint_fact(hunyuan_vl,   'tencent/Hunyuan-A13B-Instruct (needs GGUF conversion)').
hf_hint_fact(hunyuan_moe,  'tencent/Hunyuan-A13B-MoE (needs GGUF conversion)').
hf_hint_fact(smollm3,      'HuggingFaceTB/SmolLM3-3B-GGUF').
hf_hint_fact(arcee,        'arcee-ai/AFM-4.5B-Preview-GGUF').
hf_hint_fact(afmoe,        'arcee-ai/AFM-MoE-Preview-GGUF').
hf_hint_fact(ernie4_5,     'baidu/ERNIE-4.5-0.3B-PT-GGUF').
hf_hint_fact(ernie4_5_moe, 'baidu/ERNIE-4.5-21B-A3B-PT-GGUF').


%% API: huggingface_source_hint(+Arch, -Hint)
%%
%% Deterministic. Tries the declared facts first; falls back to a
%% search-suggestion string if no specific source is known. Uses
%% once/1 to ensure a single binding even if multiple facts somehow
%% match (defensive — facts should be unique by Arch).
huggingface_source_hint(Arch, Hint) :-
    ( hf_hint_fact(Arch, H0)
    -> Hint = H0
    ;  format(atom(Hint), 'unknown — search "huggingface.co ~w GGUF"', [Arch])
    ),
    !.


%% ═════════════════════════════════════════════════════════════════════
%% Coverage report
%% ═════════════════════════════════════════════════════════════════════

coverage_report :-
    format("=== Model Zoo Coverage Report ===~n~n", []),
    list_zoo(Models),
    length(Models, NTotal),
    format("Local Model Zoo: ~d entries~n", [NTotal]),
    format("~n--- Zoo contents (canonical arch resolution applied) ---~n", []),
    forall(member(zoo_model(Name, RawArch, Size, _), Models),
        ( alias_resolution(RawArch, Canonical),
          SizeMB is Size / 1048576,
          ( RawArch = Canonical
          -> format("  ~w: ~w (~0fMB)~n", [Name, RawArch, SizeMB])
          ;  format("  ~w: ~w → ~w (~0fMB)~n", [Name, RawArch, Canonical, SizeMB])
          )
        )),
    covered_archs(Covered),
    length(Covered, NCovered),
    format("~nUnique canonical architectures covered: ~d~n", [NCovered]),
    format("  ~w~n", [Covered]),
    missing_archs(Missing),
    length(Missing, NMissing),
    sort(Covered, CoveredSorted),
    sort(Missing, MissingSorted),
    append(CoveredSorted, MissingSorted, _AllArchs),
    NTotalArchs is NCovered + NMissing,
    Percentage is (NCovered * 100.0) / NTotalArchs,
    format("Total canonical archs in llama.cpp: ~d~n", [NTotalArchs]),
    format("Coverage: ~2f% (~d/~d)~n", [Percentage, NCovered, NTotalArchs]),
    format("~n--- Missing architectures (with HF source hints) ---~n", []),
    forall(member(M, MissingSorted),
        ( huggingface_source_hint(M, Hint),
          format("  ~w: ~w~n", [M, Hint])
        )).
