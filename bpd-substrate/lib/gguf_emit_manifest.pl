#!/usr/bin/env -S swipl -q -g main -t halt -s
%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
% gguf_emit_manifest.pl — Emit a tensor-manifest for a GGUF file as Prolog terms.
%
% Per Heath's guideline (2026-05-15): never use string literals containing
% program code for a target language unless tokenizer-symbol-only or in Lisp.
%
% This script avoids the JSON anti-pattern by using SWI-Prolog's native
% writeq/2 to emit Prolog ground terms directly. The intermediate format
% IS our substrate's native format. Same architectural pattern as GGUF
% itself (manifest + data; here the manifest is Prolog terms and the
% data is the original GGUF binary, accessed via mmap from offsets the
% manifest provides).
%
% Output format: each line is one Prolog ground term followed by '.':
%
%   gguf_file('/path/to/file.gguf').
%   gguf_header(magic('GGUF'), version(3), tensor_count(112), kv_count(24)).
%   tensor_data_section_start(749984).
%   metadata(KeyAtom, ValueTypeCode, Value).
%   tensor(Name, Dimensions, TypeName, NumpyDtype, FileOffset, ByteSize).
%
% Python reads with a simple line-oriented parser (no external dep).
%
% Usage:
%   ./gguf_emit_manifest.pl /path/to/file.gguf [output.pl]
%   (defaults to stdout if output.pl not provided)
%
% Authored: metayen, 2026-05-15, MVP 1 substrate-honest restart after
% Heath caught the JSON anti-pattern.
%
% Consult-safety: metayen + Heath, 2026-05-19 ~00:30 UTC, per Heath's
% substrate-design principle: "every Prolog ruleset should be usable
% via consult() with no side effects." The previous form had
% `:- initialization(main, main)` which fired main/0 automatically
% whenever the file was consulted, causing double-execution when
% another file (e.g. bpd/run_inference.pl) tried to consult it.
%
% The fix: remove `:- initialization` entirely. main/0 is still
% defined and callable; direct script invocation uses the shebang
% line `#!/usr/bin/env -S swipl -q -g main` which provides explicit
% `-g main` invocation. Consumers who consult get only the predicates,
% never the side effect of main/0 running.
%
% Full module-ization (proper `:- module/2` declaration with explicit
% exports) was attempted and reverted because the substrate-historical
% generated-code path (emit_section_reader/1 produces clauses at
% runtime that call predicates unqualified) is deeply coupled to
% `user:` namespace assumptions. Module-izing this file requires
% module-izing the runtime libraries too — substantive substrate-
% design scope not appropriate for tonight's showcase migration.
%
% The substrate-honest minimum: drop the `:- initialization` so
% consult is side-effect-free. main/0 remains explicitly callable.

:- consult('ir/include/byte_readers.pl').
:- consult('ir/include/landmarks.pl').
:- consult('ir/include/repeated_records.pl').
:- consult('file-formats/gguf/output/gguf.pl').
:- consult('ir/include/prolog_reader_generator.pl').

default_gguf_file(
    '${OLLAMA_BLOBS:-~/.ollama/models/blobs}/\c
sha256-970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6').

% GGUF tensor type codes → numpy dtype names + bytes_per_element
% (0 means quantized; size depends on block layout).
type_info(0,  'F32',  'float32', 4).
type_info(1,  'F16',  'float16', 2).
type_info(2,  'Q4_0', 'q4_0',    0).
type_info(3,  'Q4_1', 'q4_1',    0).
type_info(6,  'Q5_0', 'q5_0',    0).
type_info(7,  'Q5_1', 'q5_1',    0).
type_info(8,  'Q8_0', 'q8_0',    0).
type_info(9,  'Q8_1', 'q8_1',    0).
type_info(10, 'Q2_K', 'q2_k',    0).
type_info(11, 'Q3_K', 'q3_k',    0).
type_info(12, 'Q4_K', 'q4_k',    0).
type_info(13, 'Q5_K', 'q5_k',    0).
type_info(14, 'Q6_K', 'q6_k',    0).
type_info(15, 'Q8_K', 'q8_k',    0).
type_info(16, 'IQ2_XXS', 'iq2_xxs', 0).
type_info(17, 'IQ2_XS',  'iq2_xs',  0).
type_info(18, 'IQ3_XXS', 'iq3_xxs', 0).
type_info(19, 'IQ1_S',   'iq1_s',   0).
type_info(20, 'IQ4_NL',  'iq4_nl',  0).
type_info(21, 'IQ3_S',   'iq3_s',   0).
type_info(22, 'IQ2_S',   'iq2_s',   0).
type_info(23, 'IQ4_XS',  'iq4_xs',  0).
type_info(24, 'I8',      'int8',    1).   % was incorrectly at 28
type_info(25, 'I16',     'int16',   2).   % was incorrectly at 29
type_info(26, 'I32',     'int32',   4).   % was incorrectly at 30
type_info(27, 'I64',     'int64',   8).   % was incorrectly at 31
type_info(28, 'F64',     'float64', 8).   % was incorrectly named 'I8'
type_info(29, 'IQ1_M',   'iq1_m',   0).   % was incorrectly named 'I16'
type_info(30, 'BF16',    'bfloat16',2).   % was incorrectly named 'I32'
type_info(34, 'TQ1_0',   'tq1_0',   0).
type_info(35, 'TQ2_0',   'tq2_0',   0).
type_info(39, 'MXFP4',   'mxfp4',   0).
type_info(_,  'unknown', 'unknown', 0).

tensor_byte_size(TypeCode, Shape, ByteSize) :-
    type_info(TypeCode, _, _, BytesPerElement),
    (   BytesPerElement > 0
    ->  product_list(Shape, NumElements),
        ByteSize is NumElements * BytesPerElement
    ;   quantized_block_info(TypeCode, ElementsPerBlock, BytesPerBlock)
    ->  product_list(Shape, NumElements),
        NumBlocks is NumElements // ElementsPerBlock,
        ByteSize is NumBlocks * BytesPerBlock
    ;   ByteSize = 0  % truly unknown type
    ).

%% quantized_block_info(+TypeCode, -ElementsPerBlock, -BytesPerBlock)
%%
%% Each GGUF quantization format packs N elements into M bytes per block.
%% Values cross-checked against llama.cpp ggml-common.h block definitions.
%%
%% Block sizes (verified against llama.cpp source):
%%   Q4_0:  32 elements per block, 18 bytes (2 scale + 16 quants)
%%   Q4_1:  32 elements per block, 20 bytes (4 scale/min + 16 quants)
%%   Q5_0:  32 elements per block, 22 bytes (2 scale + 4 hibits + 16 quants)
%%   Q5_1:  32 elements per block, 24 bytes (4 scale/min + 4 hibits + 16 quants)
%%   Q8_0:  32 elements per block, 34 bytes (2 scale + 32 quants)
%%   Q8_1:  32 elements per block, 36 bytes (4 scale/min + 32 quants)
%%   Q2_K: 256 elements per block, 84 bytes (matches dequant_q2_k in mavchin's runner)
%%   Q3_K: 256 elements per block, 110 bytes (matches dequant_q3_k)
%%   Q4_K: 256 elements per block, 144 bytes
%%   Q5_K: 256 elements per block, 176 bytes
%%   Q6_K: 256 elements per block, 210 bytes (matches dequant_q6_k)
%%   Q8_K: 256 elements per block, 292 bytes
quantized_block_info(2,  32,  18).   % Q4_0
quantized_block_info(3,  32,  20).   % Q4_1
quantized_block_info(6,  32,  22).   % Q5_0
quantized_block_info(7,  32,  24).   % Q5_1
quantized_block_info(8,  32,  34).   % Q8_0
quantized_block_info(9,  32,  36).   % Q8_1
quantized_block_info(10, 256, 84).   % Q2_K
quantized_block_info(11, 256, 110).  % Q3_K
quantized_block_info(12, 256, 144).  % Q4_K
quantized_block_info(13, 256, 176).  % Q5_K
quantized_block_info(14, 256, 210).  % Q6_K
quantized_block_info(15, 256, 292).  % Q8_K

product_list([], 1).
product_list([H | T], P) :-
    product_list(T, P0),
    P is H * P0.

%% read_file_partial(+Path, +MaxBytes, -Str)
%%
%% Read up to MaxBytes bytes from Path as a binary string. Substrate-
%% historical predicate; preserved for any callers that want fixed-size
%% reads. Returns whatever's available, may be shorter than MaxBytes
%% if file is smaller.
read_file_partial(Path, MaxBytes, Str) :-
    open(Path, read, Stream, [type(binary)]),
    setup_call_cleanup(true,
        read_string(Stream, MaxBytes, Str),
        close(Stream)).


%% header_str_to_counts(+HeaderStr, -TensorCount, -KVCount)
%%
%% Parse the 24-byte GGUF header to extract just TensorCount and
%% KVCount. We don't need the full header parse for the size estimate
%% — just the counts.
%%
%% GGUF header layout (24 bytes):
%%   bytes 0-3:   magic "GGUF"  (validated separately)
%%   bytes 4-7:   version (u32 LE)  (validated separately)
%%   bytes 8-15:  tensor_count (u64 LE)
%%   bytes 16-23: metadata_kv_count (u64 LE)
header_str_to_counts(HeaderStr, TensorCount, KVCount) :-
    %% Convert string to byte list for read_u64_le
    string_codes(HeaderStr, HeaderBytes),
    read_u64_le(HeaderBytes, 8, TensorCount),
    read_u64_le(HeaderBytes, 16, KVCount).

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [PathArg | Rest]
    ->  ( atom(PathArg) -> Path = PathArg ; atom_string(Path, PathArg) ),
        ( Rest = [OutArg | _]
        -> ( atom(OutArg) -> Out = OutArg ; atom_string(Out, OutArg) )
        ; Out = '-' )
    ;   default_gguf_file(Path), Out = '-'
    ),
    emit_manifest(Path, Out).

%% emit_manifest(+Path, +Out)
%%
%% Stream-output form: parse Path, emit Prolog ground terms to Out
%% (which can be '-' for user_output or a filesystem path).
%%
%% Substrate-historical entry point; preserved bit-identically for
%% backwards compatibility. Implementation factored to share parsing
%% with emit_manifest_inline/2 via parse_gguf_manifest/2.
emit_manifest(Path, Out) :-
    parse_gguf_manifest(Path, Manifest),
    open_output(Out, OutStream),
    write_manifest_terms(OutStream, Manifest),
    close_output(Out, OutStream).

%% emit_manifest_inline(+Path, -ManifestTerms)
%%
%% In-memory form: parse Path, return the manifest as a list of
%% Prolog ground terms identical to what emit_manifest/2 would write
%% to a stream. Consumers can member/2 over the list to look up
%% specific facts:
%%
%%   ?- emit_manifest_inline('/path/to/file.gguf', Terms),
%%      member(metadata("general.architecture", _, Arch), Terms),
%%      member(metadata("nomic-bert.rope.freq_base", _, FreqBase), Terms),
%%      member(gguf_header(magic(M), version(V), _, _), Terms).
%%
%% The term list mirrors emit_manifest/2's stream output exactly:
%%   [ gguf_file(Path),
%%     gguf_header(magic(M), version(V), tensor_count(NT), kv_count(NK)),
%%     tensor_info_section_start(InfoStart),
%%     tensor_data_section_start(DataStart),
%%     metadata(Key, VType, Value), ...,
%%     tensor(name(N), dimensions(D), type_code(TC), type_name(TN),
%%            numpy_dtype(ND), file_offset(FO), byte_size(BS),
%%            relative_offset(RO)), ...
%%   ]
%%
%% Same total elements: 1 + 1 + 1 + 1 + KVCount + TensorCount.
%%
%% Used by run_inference.pl (after migration) and any future
%% consumer that wants in-memory manifest without subprocess overhead.
%%
%% Per Heath's substrate-honest unification 2026-05-18 ~23:30 UTC.
emit_manifest_inline(Path, ManifestTerms) :-
    parse_gguf_manifest(Path, Manifest),
    manifest_to_terms(Manifest, ManifestTerms).

%% parse_gguf_manifest(+Path, -Manifest)
%%
%% Internal: parse a GGUF file, return the structured Manifest:
%%
%%   manifest(
%%       path(Path),
%%       header(magic(M), version(V), tensor_count(NT), kv_count(NK)),
%%       section_starts(info(IS), data(DS)),
%%       metadata([kv(Key, VType, Value), ...]),
%%       tensors([tensor_full(Name, Dims, TypeCode, RelOff, AbsOff, BS), ...])
%%   )
%%
%% This is the substrate-internal representation. emit_manifest/2
%% and emit_manifest_inline/2 both consume from it, ensuring they
%% produce IDENTICAL output (modulo serialization vs term construction).
%% must_succeed(:Goal)
%%
%% Substrate-design utility: turn predicate failure into a typed error
%% so downstream catch/3 blocks can handle "Goal failed" symmetrically
%% with "Goal raised error." Useful when fail-by-false vs error are
%% both signals the caller needs to distinguish from success.
%%
%% Substrate-honest pattern per Heath 2026-05-19 ~00:35 UTC:
%% "we could possibly wrap it with a throw-on-false-return wrapper."
%% This IS that wrapper.
%%
%% Local to this file for now; if a second substrate consumer needs
%% the same wrapper, lift to a substrate-utility module (e.g.,
%% bpd/lib/substrate_utils.pl). Substrate-honest "wait for the
%% second data point" discipline.
must_succeed(Goal) :-
    ( call(Goal)
    -> true
    ;  throw(error(must_succeed_failed(Goal), _))
    ).


parse_gguf_manifest(Path, Manifest) :-
    %% Substrate-honest dynamic read sizing per Heath 2026-05-19 ~00:10 UTC.
    %% Replaces the prior 8 MB hard cap.
    %%
    %% Strategy:
    %%   1. Compute initial estimate from header (TensorCount, KVCount).
    %%   2. Try parse via must_succeed/1 wrapper — converts the
    %%      substrate-historical fail-by-false to a typed error.
    %%   3. If parse threw (truncated buffer), double size and retry.
    %%   4. If parse threw at full file size, re-throw — that's a
    %%      genuine parse error (corrupted GGUF or unsupported version).
    %%
    %% SUBSTRATE-HONEST DISCOVERY (2026-05-19 ~00:30 UTC during
    %% Step 3 investigation): the parser already fails cleanly with
    %% `false` on truncated input — no parser-side fail-fast logic
    %% needed. Just wrap with must_succeed/1 and the wrapper makes
    %% the error path symmetric with "real" parse errors.
    %%
    %% Per Heath's "throw-on-false-return wrapper" suggestion: this
    %% is substantively cleaner than the prior (Goal ; FallbackGoal)
    %% pattern because catch/3 handles both failure modes uniformly.
    size_file(Path, FileSize),
    initial_read_estimate(Path, InitialEstimate),
    InitialSize is min(InitialEstimate, FileSize),
    parse_gguf_manifest_grow(Path, InitialSize, FileSize, Manifest).


%% parse_gguf_manifest_grow(+Path, +Size, +FileSize, -Manifest)
%%
%% Try parsing with buffer of Size bytes. The actual parse is wrapped
%% in must_succeed/1 so that failure-by-false propagates as a
%% must_succeed_failed/1 error, caught here for the retry decision.
%%
%% On retry:
%%   - If we're already at full file size: re-throw (substantive parse error).
%%   - Otherwise: double Size and recurse.
%%
%% Worst-case attempts: ceil(log2(FileSize / InitialSize)) — for a
%% 4 GB file with 4 MB initial estimate, that's 10 retries.
parse_gguf_manifest_grow(Path, Size, FileSize, Manifest) :-
    ReadSize is min(Size, FileSize),
    read_file_partial(Path, ReadSize, Bytes),
    catch(
        must_succeed(do_parse_gguf_manifest(Path, Bytes, Manifest)),
        error(must_succeed_failed(_), _),
        handle_parse_growable(Path, Size, FileSize, Manifest, ReadSize)
    ).


%% handle_parse_growable(+Path, +Size, +FileSize, -Manifest, +ReadSize)
%%
%% Called when must_succeed catches a failure from do_parse_gguf_manifest.
%% Either retry with a larger buffer, or surface a real parse error.
handle_parse_growable(Path, _Size, FileSize, _Manifest, ReadSize) :-
    ReadSize >= FileSize,
    !,
    %% Already read the whole file; failure is substantive
    throw(error(gguf_parse_failed(Path, file_size(FileSize)), _)).
handle_parse_growable(Path, Size, FileSize, Manifest, _ReadSize) :-
    Doubled is Size * 2,
    parse_gguf_manifest_grow(Path, Doubled, FileSize, Manifest).


%% initial_read_estimate(+Path, -Estimate)
%%
%% Compute the initial buffer estimate from the GGUF header. This is
%% an OPTIMISTIC sizing: tight enough to be efficient on small models,
%% with the probe-and-grow loop handling underestimates automatically.
%%
%% Heuristic (tuned empirically against nomic-embed-text and llama-class
%% models 2026-05-19):
%%   - Header: 24 bytes
%%   - Per-KV typical: 512 bytes (most KVs are small)
%%   - Per-tensor record: 256 bytes
%%   - Vocab allowance: 1 MB (covers most tokenizer vocabs up to
%%     ~50K tokens at ~20 bytes/token avg; grow handles bigger)
%%   - Safety margin: 128 KB
%%
%% Empirical: for nomic-embed (24 KV, 112 tensors), this yields
%% ~1.2 MB initial estimate vs the ~757 KB actual need. One-pass
%% parse, no grow retry on the common case.
%%
%% For models with bigger vocabs (Llama 128K-token vocab averaging
%% ~10 bytes/token = 1.3 MB), the 1 MB allowance is tight — one
%% grow doubles to 2 MB which suffices. Substrate-honest: tight
%% enough to be efficient on small models, with grow handling the
%% next size class.
initial_read_estimate(Path, Estimate) :-
    open(Path, read, Stream, [type(binary)]),
    read_string(Stream, 24, HeaderStr),
    close(Stream),
    string_length(HeaderStr, HLen),
    ( HLen < 24
    -> throw(error(gguf_truncated(Path, header_only(HLen)), _))
    ;  true
    ),
    header_str_to_counts(HeaderStr, TensorCount, KVCount),
    Estimate is 24
              + KVCount * 512
              + 1048576             % 1 MB vocab allowance
              + TensorCount * 256
              + 131072.             % 128 KB safety margin


%% do_parse_gguf_manifest(+Path, +Bytes, -Manifest)
%%
%% The actual parsing logic. Lifted from the original emit_manifest/2
%% body. Operates on a pre-read Bytes buffer.
do_parse_gguf_manifest(Path, Bytes, Manifest) :-
    emit_section_reader(header),
    emit_section_reader(metadata_kv_pair),
    emit_section_reader(tensor_info_record),

    with_parse_session(manifest, (
        read_section(header, Bytes, HeaderPairs),
        member(field(magic, Magic), HeaderPairs),
        member(field(version, Version), HeaderPairs),
        member(field(tensor_count, TensorCount), HeaderPairs),
        member(field(metadata_kv_count, KVCount), HeaderPairs),

        bytes_slice_from(Bytes, 24, MetaBytes),
        walk_metadata_pairs(MetaBytes, 0, 1, KVCount, MetaList, MetaEnd),

        TensorSectionStart is 24 + MetaEnd,
        iter_tensors(Bytes, TensorSectionStart, 1, TensorCount, [],
                     TensorList, TensorInfoEnd),

        TensorDataAlign = 32,
        TensorDataStart_unaligned = TensorInfoEnd,
        TensorDataStart is ((TensorDataStart_unaligned + TensorDataAlign - 1)
                            // TensorDataAlign) * TensorDataAlign,

        compute_tensor_file_offsets(TensorList, TensorDataStart,
                                     TensorListWithFileOffsets)
    )),

    Manifest = manifest(
        path(Path),
        header(magic(Magic), version(Version),
               tensor_count(TensorCount), kv_count(KVCount)),
        section_starts(info(TensorSectionStart), data(TensorDataStart)),
        metadata(MetaList),
        tensors(TensorListWithFileOffsets)
    ).

%% manifest_to_terms(+Manifest, -Terms)
%%
%% Convert structured Manifest into flat term list matching the
%% emit_manifest/2 stream output. Each term in the result is exactly
%% one of: gguf_file/1, gguf_header/4, tensor_info_section_start/1,
%% tensor_data_section_start/1, metadata/3, tensor/8.
manifest_to_terms(Manifest, Terms) :-
    Manifest = manifest(
        path(Path),
        header(magic(M), version(V), tensor_count(NT), kv_count(NK)),
        section_starts(info(IS), data(DS)),
        metadata(MetaList),
        tensors(TensorList)
    ),
    %% Build the prelude terms
    Prelude = [
        gguf_file(Path),
        gguf_header(magic(M), version(V), tensor_count(NT), kv_count(NK)),
        tensor_info_section_start(IS),
        tensor_data_section_start(DS)
    ],
    %% Convert metadata K-V pairs to metadata/3 terms
    findall(metadata(Key, VType, Value),
            member(kv(Key, VType, Value), MetaList),
            MetaTerms),
    %% Convert tensors to tensor/8 terms (matching emit_terms exactly).
    %%
    %% Use once/1 wrapping type_info/4 because the type_info table
    %% has a catch-all clause type_info(_, 'unknown', 'unknown', 0)
    %% that matches any TypeCode. Without once/1, findall enumerates
    %% TWO solutions per tensor (specific + catch-all), producing
    %% 2× the expected tensor count. emit_terms/10 uses forall+member
    %% which sees both solutions but the side-effect of write_term_dot
    %% emits each tensor twice — the substrate-historical code path
    %% has the same latent issue, but masked because forall+writeq
    %% to stdout effectively serializes only the first solution per
    %% backtrack. findall surfaces it.
    findall(tensor(
                name(Name),
                dimensions(Dims),
                type_code(TC),
                type_name(TypeName),
                numpy_dtype(NumpyDtype),
                file_offset(AbsOff),
                byte_size(BS),
                relative_offset(RelOff)
            ),
            ( member(tensor_full(Name, Dims, TC, RelOff, AbsOff, BS),
                     TensorList),
              once(type_info(TC, TypeName, NumpyDtype, _)) ),
            TensorTerms),
    append([Prelude, MetaTerms, TensorTerms], Terms).

%% write_manifest_terms(+Stream, +Manifest)
%%
%% Serialize structured Manifest to Stream in the canonical
%% emit_manifest/2 form. This is just manifest_to_terms/2 followed
%% by writeq for each term, but kept as a separate predicate to
%% preserve the substrate-historical formatting of emit_terms/10.
write_manifest_terms(Stream, Manifest) :-
    Manifest = manifest(
        path(Path),
        header(magic(M), version(V), tensor_count(NT), kv_count(NK)),
        section_starts(info(IS), data(DS)),
        metadata(MetaList),
        tensors(TensorList)
    ),
    emit_terms(Stream, Path, M, V, NT, NK, IS, DS, MetaList, TensorList).

walk_metadata_pairs(_, Offset, N, Max, [], Offset) :- N > Max, !.
walk_metadata_pairs(Bytes, Offset, N, Max, [KV | Rest], FinalOffset) :-
    read_section_at(metadata_kv_pair, Bytes, Offset, Pairs, Consumed),
    member(field(key, KeyStr), Pairs),
    member(field(value_type, VType), Pairs),
    ( member(field(value, V), Pairs) -> Value = V ; Value = unknown ),
    KV = kv(KeyStr, VType, Value),
    NextOffset is Offset + Consumed,
    NextN is N + 1,
    walk_metadata_pairs(Bytes, NextOffset, NextN, Max, Rest, FinalOffset).

iter_tensors(_, Offset, N, Max, Acc, Tensors, Offset) :-
    N > Max, !,
    reverse(Acc, Tensors).
iter_tensors(Bytes, Offset, N, Max, Acc, Tensors, FinalOffset) :-
    read_section_at(tensor_info_record, Bytes, Offset, Pairs, Consumed),
    member(field(name, NameStr), Pairs),
    member(field(n_dimensions, _NDims), Pairs),
    member(field(dimensions, Dims), Pairs),
    member(field(type, TypeCode), Pairs),
    member(field(offset, TensorRelOffset), Pairs),
    tensor_byte_size(TypeCode, Dims, ByteSize),
    T = tensor_raw(NameStr, Dims, TypeCode, TensorRelOffset, ByteSize),
    NextOffset is Offset + Consumed,
    NextN is N + 1,
    iter_tensors(Bytes, NextOffset, NextN, Max, [T | Acc], Tensors, FinalOffset).

compute_tensor_file_offsets([], _, []).
compute_tensor_file_offsets([tensor_raw(N, Dims, TC, RelOff, BS) | Rest],
                             SectionStart,
                             [tensor_full(N, Dims, TC, RelOff, AbsOff, BS) | Rest2]) :-
    AbsOff is SectionStart + RelOff,
    compute_tensor_file_offsets(Rest, SectionStart, Rest2).

open_output('-', user_output) :- !.
open_output(Path, Stream) :- open(Path, write, Stream).

close_output('-', _) :- !.
close_output(_, Stream) :- close(Stream).

% Emit Prolog ground terms with writeq/2 — this is the substrate-honest
% serializer. writeq/2 quotes atoms requiring quoting and emits proper
% Prolog syntax for lists, integers, strings.
emit_terms(S, Path, Magic, Version, TensorCount, KVCount,
           TensorSectionStart, TensorDataStart, MetaList, TensorList) :-
    % File path
    write_term_dot(S, gguf_file(Path)),

    % Header summary
    write_term_dot(S, gguf_header(
        magic(Magic),
        version(Version),
        tensor_count(TensorCount),
        kv_count(KVCount)
    )),

    % Section starts
    write_term_dot(S, tensor_info_section_start(TensorSectionStart)),
    write_term_dot(S, tensor_data_section_start(TensorDataStart)),

    % Metadata K-V pairs
    forall(member(kv(Key, VType, Value), MetaList),
           write_term_dot(S, metadata(Key, VType, Value))),

    % Tensor manifest entries
    forall(member(tensor_full(Name, Dims, TC, RelOff, AbsOff, BS), TensorList),
           ( type_info(TC, TypeName, NumpyDtype, _),
             write_term_dot(S, tensor(
                 name(Name),
                 dimensions(Dims),
                 type_code(TC),
                 type_name(TypeName),
                 numpy_dtype(NumpyDtype),
                 file_offset(AbsOff),
                 byte_size(BS),
                 relative_offset(RelOff)
             ))
           )).

write_term_dot(S, T) :-
    writeq(S, T),
    write(S, '.'),
    nl(S).

%% NO :- initialization directive.
%%
%% Per Heath's substrate-design principle 2026-05-19 ~00:30 UTC:
%% "every Prolog ruleset should be usable via consult() with no side
%% effects." This file is now consult-safe — loading it provides the
%% predicates but does NOT execute main/0.
%%
%% To run as a script: the shebang `#!/usr/bin/env -S swipl -q -g main`
%% provides explicit `-g main` invocation. Direct shell invocation:
%%
%%   ./gguf_emit_manifest.pl /path/to/file.gguf [output.pl]
%%
%% will run main/0 because the shebang requests it. Without the
%% shebang, callers must explicitly invoke main/0:
%%
%%   swipl -q -g 'consult(gguf_emit_manifest), main' -- /path/file.gguf
%%
%% or use any specific predicate they want:
%%
%%   swipl -q -g 'consult(gguf_emit_manifest),
%%               emit_manifest_inline("/path/file.gguf", M),
%%               write(M), halt.'
