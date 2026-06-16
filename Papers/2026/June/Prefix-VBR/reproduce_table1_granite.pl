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

%% ---- Read floats from CSV or FP16 binary ----
read_all_floats(Path, AllFloats) :-
    (  sub_atom(Path, _, _, 0, '.f16')
    -> read_fp16_binary(Path, AllFloats)
    ;  read_csv_floats(Path, AllFloats)
    ).

%% CSV: one embedding per line, values comma-separated
read_csv_floats(Path, AllFloats) :-
    read_file_to_string(Path, Content, []),
    split_string(Content, "\n", "\n", Lines),
    maplist(parse_float_line, Lines, NestedFloats),
    append(NestedFloats, AllFloats).

parse_float_line(Line, Floats) :-
    split_string(Line, ",", " \t", Tokens),
    exclude(=(""), Tokens, NonEmpty),
    maplist(to_float, NonEmpty, Floats).

to_float(S, F) :- number_string(F, S).

%% FP16 binary: raw IEEE 754 half-precision, little-endian
read_fp16_binary(Path, AllFloats) :-
    open(Path, read, Stream, [type(binary)]),
    read_fp16_values(Stream, AllFloats),
    close(Stream).

read_fp16_values(Stream, Floats) :-
    (  get_byte(Stream, B0),
       B0 \= -1,
       get_byte(Stream, B1)
    -> Raw is B0 + (B1 << 8),
       Sign is (Raw >> 15) /\ 1,
       Exp is (Raw >> 10) /\ 0x1F,
       Mant is Raw /\ 0x3FF,
       fp16_decode(Sign, Exp, Mant, F),
       Floats = [F|Rest],
       read_fp16_values(Stream, Rest)
    ;  Floats = []
    ).

fp16_decode(_, 0, 0, 0.0) :- !.
fp16_decode(Sign, 0, Mant, F) :- !,
    F is (-1)**Sign * (Mant / 1024) * 2**(-14).
fp16_decode(_, 31, _, 0.0) :- !.  % inf/nan -> 0 for safety
fp16_decode(Sign, Exp, Mant, F) :-
    F is (-1)**Sign * (1 + Mant / 1024) * 2**(Exp - 15).

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
