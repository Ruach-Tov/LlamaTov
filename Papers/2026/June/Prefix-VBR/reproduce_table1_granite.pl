#!/usr/bin/env swipl
%% reproduce_table1_granite.pl — Reproduce Table 1 from granite embeddings.
%%
%% Two-step process:
%%   1. Extract embeddings: psql -d claude_conversations -c "COPY ..." > embeddings.csv
%%   2. Analyze: swipl reproduce_table1_granite.pl embeddings.csv
%%
%% Or run the helper script which does both:
%%   ./run_table1.sh
%%
%% Reads embedding vectors as comma-separated floats, one per line.
%% Applies per-block normalization (block_size=32) and computes distribution.

:- use_module(library(csv)).
:- use_module(library(readutil)).
:- use_module(library(lists)).

block_size(32).

main :-
    current_prolog_flag(argv, Args),
    (Args = [Path|_] -> true ; Path = 'embeddings.csv'),
    format("Prefix-VBR Table 1 Reproduction~n"),
    format("Source: ~w~n~n", [Path]),
    
    %% Step 1: Read all float values from the CSV
    format("Loading embeddings...~n"),
    read_all_floats(Path, AllWeights),
    length(AllWeights, TotalWeights),
    format("Total weight values: ~w~n", [TotalWeights]),
    
    %% Basic stats
    min_list(AllWeights, MinW), max_list(AllWeights, MaxW),
    maplist([W, A]>>(A is abs(W)), AllWeights, AbsAll),
    max_list(AbsAll, AbsMax),
    format("Range: [~6f, ~6f]  |max|=~6f~n~n", [MinW, MaxW, AbsMax]),
    
    %% Step 2: Per-block normalize
    block_size(BS),
    format("Per-block normalizing (block_size=~w)...~n", [BS]),
    per_block_normalize(AllWeights, BS, NormWeights),
    length(NormWeights, NNorm),
    format("Normalized values: ~w~n", [NNorm]),
    
    %% Step 3: Histogram
    Bins = [0-32, 32-64, 64-96, 96-128],
    compute_histogram(NormWeights, Bins, Results),
    
    %% Step 4: Print
    format("~n==============================================~n"),
    format("  TABLE 1: Per-block normalized |w_norm|~n"),
    format("  Source: granite-embedding, 384 dims~n"),
    format("  Block size: ~w~n", [BS]),
    format("==============================================~n"),
    format("  ~w~t~25|~w~t~40|~w~n", ['Range', 'Fraction', 'Cumulative']),
    format("  ~`-t~50|~n"),
    print_results(Results, 0.0),
    format("~n"),
    halt.

main :-
    format("Error~n"), halt(1).

%% ---- Read floats from CSV (one embedding per line, values comma-separated) ----
read_all_floats(Path, AllFloats) :-
    read_file_to_string(Path, Content, []),
    split_string(Content, "\n", "\n", Lines),
    maplist(parse_float_line, Lines, NestedFloats),
    append(NestedFloats, AllFloats).

parse_float_line(Line, Floats) :-
    split_string(Line, ",", " \t", Tokens),
    exclude(=(""), Tokens, NonEmpty),
    maplist(to_float, NonEmpty, Floats).

to_float(S, F) :- number_string(F, S).

%% ---- Per-block normalization ----
per_block_normalize(Weights, BlockSize, NormWeights) :-
    chunk_list(Weights, BlockSize, Blocks),
    include(full_block(BlockSize), Blocks, FullBlocks),
    maplist(normalize_block, FullBlocks, NormBlocks),
    append(NormBlocks, NormWeights).

full_block(BS, Block) :- length(Block, BS).

normalize_block(Block, NormBlock) :-
    maplist([W, A]>>(A is abs(W)), Block, AbsBlock),
    max_list(AbsBlock, MaxAbs),
    (MaxAbs =:= 0.0
    ->  NormBlock = Block
    ;   Scale is MaxAbs / 127.0,
        maplist({Scale}/[W, NW]>>(NW is W / Scale), Block, NormBlock)
    ).

chunk_list([], _, []) :- !.
chunk_list(List, N, [Chunk|Rest]) :-
    length(Chunk, N),
    append(Chunk, Remaining, List), !,
    chunk_list(Remaining, N, Rest).
chunk_list(Remainder, _, [Remainder]).

%% ---- Histogram ----
compute_histogram(Values, Bins, Results) :-
    length(Values, Total),
    maplist(count_bin(Values, Total), Bins, Results).

count_bin(Values, Total, Lo-Hi, bin(Lo, Hi, Pct)) :-
    include({Lo, Hi}/[V]>>(Abs is abs(V), Abs >= Lo, Abs < Hi), Values, InBin),
    length(InBin, Count),
    Pct is (Count / Total) * 100.

print_results([], _).
print_results([bin(Lo, Hi, Pct)|Rest], Cumul) :-
    NewCumul is Cumul + Pct,
    format("  [~w, ~w)~t~25|~1f%~t~40|~1f%~n", [Lo, Hi, Pct, NewCumul]),
    print_results(Rest, NewCumul).

:- initialization(main, main).
