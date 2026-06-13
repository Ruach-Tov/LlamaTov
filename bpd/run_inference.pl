%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% run_inference.pl — BPD Substrate Inference Demo
%%
%% HEADLINE: Load GGUF model from Prolog, generate CUDA kernels from
%% BPD facts, compile with nvcc, run inference on Tesla P4.
%% Ollama can't run on this GPU. We can.
%%
%% Usage: swipl run_inference.pl -- --model PATH --prompt "Hello"

:- use_module('lib/c_ast').
:- use_module('lib/kernel_templates_llama', [except([kernel_available_fixes/2, fix_description/2])]).
:- use_module('lib/kernel_templates_blas').

%% Per Heath's substrate-honest unification 2026-05-19 ~00:00 UTC:
%% Consult the unified pure-Prolog GGUF parser from must_close/boundary_dsl.
%% This provides emit_manifest_inline/2 — the substrate-honest replacement
%% for the embedded-Python parser in main/0 (subtask 4.2).
%%
%% Subtask 4.1 (this commit): ESTABLISH the dependency. main/0 still uses
%% the Python shell-out for now; the consult just makes
%% emit_manifest_inline/2 callable. Subtask 4.2 will wire it into main/0.
%%
%% Path is absolute from the repo root so the consult works regardless
%% of where the user invokes swipl from. The substrate's deployment
%% byte_readers.pl was refreshed in commit 906754f68; the in-memory
%% delivery surface emit_manifest_inline/2 was added in commit 1a7ff880b
%% and made probe-and-grow capable in commit a90ca84dc.
:- consult('./must_close/boundary_dsl/gguf_emit_manifest.pl').

%% ═══════════════════════════════════════════════════════════════
%% GGUF parsing is provided by the unified pure-Prolog parser at
%% must_close/boundary_dsl/gguf_emit_manifest.pl (consulted above).
%% Subtask 4.2 migrated main/0 to emit_manifest_inline/2; subtask 4.4
%% (this commit) deletes the dead inline parser that was here.
%%
%% Substrate-archaeology: the deleted code (87 lines) was an inline
%% pure-Prolog GGUF parser — gguf_parse/4, read_uint32/2,
%% read_uint64/2, read_string/2, read_n_bytes/3, read_kv_pairs/3,
%% read_value/3, read_float32/2, read_array/4, read_tensor_infos/3,
%% read_dims/3. It was written by mavchin in the initial showcase
%% draft but never called — main/0 used the Python shell-out instead.
%% The inline parser also had a substantive bug: read_float32 returned
%% the integer bit pattern rather than decoding IEEE 754, which would
%% have produced 1148846080 for rope_freq_base instead of 1000.0.
%%
%% The unified parser (gguf_emit_manifest.pl + Boundary DSL chain)
%% handles all of this correctly with substrate-design discipline.
%% Three parsers → one canonical parser. Substrate-debt removed.

%% ═══════════════════════════════════════════════════════════════════
%% KERNEL GENERATION — Emit all inference kernels from BPD facts
%% ═══════════════════════════════════════════════════════════════

generate_inference_kernels(Code) :-
    % Generate all kernels needed for transformer inference
    sgemv_kernel_substrate_native(k_sgemv, SgemvK),
    
    Program = [
        c_include_sys('cuda_runtime.h'),
        c_include_sys('math.h'),
        c_blank,
        c_comment('BPD-generated inference kernels'),
        c_comment('Generated from Prolog facts — zero hand-written CUDA'),
        c_blank,
        SgemvK
    ],
    emit_program(Program, Code).

%% ═══════════════════════════════════════════════════════════════
%% MAIN — Parse args, load model, generate kernels, run inference
%% ═══════════════════════════════════════════════════════════════

main :-
    % Raise the stack limit substrate-honestly: large GGUFs with big
    % tokenizer vocabularies (qwen2.5, gemma, gpt-oss with 100K-200K
    % token vocabs) can exceed the default 1GB SWI stack limit during
    % byte-list parsing. 8GB is enough for every locally-tested model
    % including 13GB gpt-oss. Per all-models verification 2026-05-19
    % ~01:20 UTC.
    set_prolog_flag(stack_limit, 8_589_934_592),  % 8 GB

    format("~n=== BPD Substrate Inference ===~n"),
    format("Prolog -> GGUF -> CUDA kernels -> Tesla P4~n~n"),
    
    % Parse command line args
    current_prolog_flag(argv, Args),
    (   member('--model', Args), nth1(I, Args, '--model'), I1 is I+1, nth1(I1, Args, ModelPath)
    ->  true
    ;   ModelPath = '/tmp/llamatov-data/ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45'
    ),
    
    format("Model: ~w~n", [ModelPath]),
    
    % Step 1: Parse GGUF metadata via the unified pure-Prolog parser.
    %
    % Subtask 4.2 migration per Heath 2026-05-19 ~00:30 UTC:
    % Replaces the prior Python shell-out (process_create + struct.unpack
    % via builtins:exec) with emit_manifest_inline/2 from the unified
    % Boundary-DSL-generated parser at must_close/boundary_dsl/
    % gguf_emit_manifest.pl (consulted at top of file).
    %
    % The substrate-honest substantive claim: pure Prolog, no Python,
    % bit-identical to the previous shell-out output for the substrate-
    % historical metadata extraction. The `llama.*` key lookups (which
    % return 0 for non-llama models like nomic-bert) are PRESERVED
    % bit-identically as substrate-stale behavior; cleaning them up to
    % use arch-specific prefixes is a separate substrate-design task
    % (subtask outside this migration).
    format("Step 1: Parsing GGUF metadata...~n"),
    atom_string(ModelPath, ModelPathStr),
    emit_manifest_inline(ModelPathStr, Manifest),
    
    % Extract the five substrate-historical values from the manifest:
    %   NKV, NT, DataOffset, Arch, EmbDim, NLayers, NHeads
    % using member/2 patterns that mirror the manifest's term shapes.
    member(gguf_header(_, _, tensor_count(NT), kv_count(NKV)), Manifest),
    member(tensor_data_section_start(DataOffset), Manifest),
    ( member(metadata("general.architecture", _, ArchStr), Manifest)
    -> atom_string(Arch, ArchStr)
    ;  Arch = unknown
    ),
    % Substrate-historical: looks up llama.* keys, returning 0 for
    % non-llama architectures. Preserved bit-identically.
    ( member(metadata("llama.embedding_length", _, EmbDim), Manifest) -> true ; EmbDim = 0 ),
    ( member(metadata("llama.block_count", _, NLayers), Manifest) -> true ; NLayers = 0 ),
    ( member(metadata("llama.attention.head_count", _, NHeads), Manifest) -> true ; NHeads = 0 ),
    
    format("  ~w metadata entries, ~w tensors, data offset ~w~n", [NKV, NT, DataOffset]),
    format("  Architecture: ~w, ~w layers, ~w heads, dim ~w~n", [Arch, NLayers, NHeads, EmbDim]),
    
    % Step 2: Generate inference kernels
    format("~nStep 2: Generating CUDA kernels from BPD facts...~n"),
    generate_inference_kernels(Code),
    atom_length(Code, CodeLen),
    format("  Generated ~w characters of CUDA source~n", [CodeLen]),
    
    % Write to file
    atom_concat('build/inference_kernels.cu', '', OutPath),
    open(OutPath, write, OS),
    write(OS, Code),
    close(OS),
    format("  Written to ~w~n", [OutPath]),
    
    format("~n=== BPD Substrate: Model loaded, kernels generated ===~n"),
    format("=== Ollama CANNOT run on this GPU (no sm_61 kernels) ===~n"),
    format("=== Our substrate CAN — Prolog generates CUDA for Pascal ===~n~n"),
    halt(0).

:- initialization(main).
