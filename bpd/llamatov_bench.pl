%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% llamatov_bench.pl — benchmark harness for tok/s comparison
%%
%% Per Heath's directive 2: "generate CUDA for the Stanford L1, then the L2
%% is the same as the end-to-end benchmark — we have to generate CUDA for
%% the L2 problems and the L3 problems, and compare with ollama in tok/s."
%%
%% This is the SUBSTRATE for the comparison. Even before CUDA emission is
%% wired into the runner, having the harness ready means we can immediately
%% measure: baseline (unfused, CPU) → fused-CPU → fused-GPU progression.
%%
%% USAGE:
%%   bench_llamatov(+ModelPath, +InputTokens, +NumIterations, -Stats)
%%   bench_ollama(+ModelName, +Prompt, +NumIterations, -Stats)
%%   compare_bench(+ModelPath, +InputTokens, +OllamaModel, +Prompt, +N, -Comparison)
%%
%% Stats is a dict of: total_seconds, tok_per_sec, mean_per_iter, ...

:- use_module(library(janus)).
:- py_add_lib_dir('lib').
:- use_module('lib/tensor_loader_adapter').

%% ────────────────────────────────────────────────────────────────────
%% bench_llamatov(+ModelPath, +InputTokens, +NumIterations, -Stats)
%% ────────────────────────────────────────────────────────────────────
%%
%% Run llamatov_llama inference NumIterations times. Returns timing
%% statistics. Each iteration generates ONE next token (no continuation).
%%
%% Stats fields:
%%   total_seconds      — total elapsed
%%   mean_seconds       — average per iteration
%%   tok_per_sec        — tokens generated per second (rough proxy)
%%   tokens_output      — list of all generated next-tokens (sanity check)
%%   model_path         — for the record

bench_llamatov(ModelPath, InputTokens, NumIterations, Stats) :-
    %% Load the runner once
    consult('llamatov_llama'),

    format("=== bench_llamatov ===~n", []),
    format("  Model: ~w~n", [ModelPath]),
    format("  Input: ~w~n", [InputTokens]),
    format("  Iterations: ~d~n", [NumIterations]),

    %% Run N times, collecting times and outputs
    get_time(StartTime),
    bench_loop(ModelPath, InputTokens, NumIterations, [], [], Times, Tokens),
    get_time(EndTime),
    TotalSeconds is EndTime - StartTime,
    MeanSeconds is TotalSeconds / NumIterations,
    TokPerSec is NumIterations / TotalSeconds,
    format("~n=== Results ===~n", []),
    format("  Total seconds:    ~3f~n", [TotalSeconds]),
    format("  Mean per iter:    ~3f~n", [MeanSeconds]),
    format("  Tok/s (1 each):   ~3f~n", [TokPerSec]),
    format("  Generated:        ~w~n", [Tokens]),
    Stats = stats(
        total_seconds(TotalSeconds),
        mean_seconds(MeanSeconds),
        tok_per_sec(TokPerSec),
        tokens_output(Tokens),
        per_iter_times(Times),
        model_path(ModelPath)
    ).

bench_loop(_, _, 0, AccT, AccTokens, AccT, AccTokens) :- !.
bench_loop(ModelPath, InputTokens, N, AccT, AccTokens, Times, Tokens) :-
    N > 0,
    get_time(T0),
    user:run(ModelPath, InputTokens, OutputToken),
    get_time(T1),
    DT is T1 - T0,
    format("  iter ~d: ~3fs → token ~w~n", [N, DT, OutputToken]),
    N1 is N - 1,
    bench_loop(ModelPath, InputTokens, N1,
                [DT | AccT], [OutputToken | AccTokens],
                Times, Tokens).

%% ────────────────────────────────────────────────────────────────────
%% bench_ollama(+ModelName, +Prompt, +NumIterations, -Stats)
%% ────────────────────────────────────────────────────────────────────
%%
%% Run ollama inference NumIterations times using ollama's HTTP API.
%% Returns timing statistics for comparison with bench_llamatov.
%%
%% Note: ollama's eval_count + eval_duration response fields provide
%% the actual tokens-per-second the user experiences.

bench_ollama(ModelName, Prompt, NumIterations, Stats) :-
    format("=== bench_ollama ===~n", []),
    format("  Model:  ~w~n", [ModelName]),
    format("  Prompt: ~w~n", [Prompt]),
    format("  Iters:  ~d~n", [NumIterations]),
    get_time(T0),
    ollama_loop(ModelName, Prompt, NumIterations, [], [], Times, Outputs),
    get_time(T1),
    TotalSeconds is T1 - T0,
    MeanSeconds is TotalSeconds / NumIterations,
    %% Average tok/s from ollama's own measurement
    average_ollama_tok_per_sec(Outputs, AvgTokPerSec),
    format("~n=== Ollama Results ===~n", []),
    format("  Total seconds:    ~3f~n", [TotalSeconds]),
    format("  Mean per iter:    ~3f~n", [MeanSeconds]),
    format("  Avg ollama tok/s: ~3f~n", [AvgTokPerSec]),
    Stats = stats(
        total_seconds(TotalSeconds),
        mean_seconds(MeanSeconds),
        avg_ollama_tok_per_sec(AvgTokPerSec),
        per_iter_outputs(Outputs),
        per_iter_times(Times),
        model_name(ModelName)
    ).

ollama_loop(_, _, 0, AccT, AccO, AccT, AccO) :- !.
ollama_loop(Model, Prompt, N, AccT, AccO, Times, Outputs) :-
    N > 0,
    get_time(T0),
    ollama_generate(Model, Prompt, Response),
    get_time(T1),
    DT is T1 - T0,
    %% Extract eval_count and eval_duration from response
    extract_ollama_stats(Response, EvalCount, EvalNs),
    OllamaTokPerSec is EvalCount / (EvalNs / 1_000_000_000),
    format("  iter ~d: ~3fs total, ~d tokens, ~3f tok/s (ollama)~n",
           [N, DT, EvalCount, OllamaTokPerSec]),
    N1 is N - 1,
    ollama_loop(Model, Prompt, N1,
                 [DT | AccT],
                 [response(EvalCount, EvalNs, OllamaTokPerSec) | AccO],
                 Times, Outputs).

%% ollama_generate(+Model, +Prompt, -ResponseDict)
%% Calls ollama's /api/generate endpoint via curl. Returns the parsed
%% response (which is JSONL — we read the final line for stats).
ollama_generate(Model, Prompt, Response) :-
    %% Use stream=false so we get a single JSON response with stats
    format(atom(Body),
           '{"model": "~w", "prompt": "~w", "stream": false}',
           [Model, Prompt]),
    format(atom(Cmd),
           'curl -s http://localhost:11434/api/generate -d \'~w\'',
           [Body]),
    setup_call_cleanup(
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Out)), stderr(null), process(_)]),
        read_string(Out, _, ResponseStr),
        close(Out)
    ),
    %% Parse the response JSON. We'll just regex-extract the fields
    %% we need; full JSON parsing isn't needed.
    Response = response_str(ResponseStr).

%% extract_ollama_stats(+Response, -EvalCount, -EvalDurationNs)
%% Pulls eval_count and eval_duration from ollama's JSON response.
%% Uses Python's json module via janus for robust parsing — substrate-honest
%% rather than fragile string surgery.
%% Use a small Python helper to parse and extract the two fields.
%% Helper lives in llamatov_helpers.py (already on py path).
extract_ollama_stats(response_str(Str), EvalCount, EvalDuration) :-
    py_call(llamatov_helpers:extract_ollama_eval_stats(Str), Tuple),
    %% Janus represents Python 2-tuples as `A - B`.
    Tuple = (EvalCount - EvalDuration),
    !.
extract_ollama_stats(_, 0, 1).

average_ollama_tok_per_sec([], 0.0).
average_ollama_tok_per_sec(Outputs, Avg) :-
    findall(R, member(response(_, _, R), Outputs), Rates),
    sumlist(Rates, Sum),
    length(Rates, N),
    Avg is Sum / N.

%% ────────────────────────────────────────────────────────────────────
%% compare_bench: side-by-side llamatov vs ollama
%% ────────────────────────────────────────────────────────────────────

compare_bench(ModelPath, InputTokens, OllamaModel, Prompt, N, Comparison) :-
    format("~n╔══════════════════════════════════════════════════════════════╗~n", []),
    format("║  COMPARISON: llamatov (Prolog-maximal) vs ollama (native)    ║~n", []),
    format("╚══════════════════════════════════════════════════════════════╝~n", []),
    bench_llamatov(ModelPath, InputTokens, N, LStats),
    bench_ollama(OllamaModel, Prompt, N, OStats),
    LStats = stats(total_seconds(LT), _, tok_per_sec(LTPS), _, _, _),
    OStats = stats(total_seconds(OT), _, avg_ollama_tok_per_sec(OTPS), _, _, _),
    Ratio is OTPS / LTPS,
    format("~n=== SIDE-BY-SIDE ===~n", []),
    format("  llamatov:  ~3fs total, ~3f tok/s~n", [LT, LTPS]),
    format("  ollama:    ~3fs total, ~3f tok/s~n", [OT, OTPS]),
    format("  ollama is ~3fx faster than llamatov~n", [Ratio]),
    Comparison = comparison(
        llamatov(LStats),
        ollama(OStats),
        ollama_speedup_factor(Ratio)
    ).

%% ────────────────────────────────────────────────────────────────────
%% Top-level convenience: run a comparison with TinyLlama
%% ────────────────────────────────────────────────────────────────────

bench_tinyllama(N, Comparison) :-
    TinyLlamaPath = '${OLLAMA_BLOBS:-~/.ollama/models/blobs}/sha256-2af3b81862c6be03c769683af18efdadb2c33f60ff32ab6f83e42c043d6c7816',
    InputTokens = [1, 15043],   %% BOS + "He"
    OllamaModel = 'tinyllama',
    Prompt = 'He',
    compare_bench(TinyLlamaPath, InputTokens, OllamaModel, Prompt, N, Comparison).

%% (No module declaration — load this as a script with swipl -l llamatov_bench.pl)
