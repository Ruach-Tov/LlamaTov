#!/usr/bin/env swipl
%% reproduce_table2_mse.pl — Reproduce Table 2: MSE comparison.
%%
%% Reads embedding data (FP16 binary or CSV), quantizes with Q8_0
%% and Prefix-VBR, measures round-trip MSE for each scheme.
%%
%% Usage (from Papers/2026/June/Prefix-VBR/):
%%   swipl --stack_limit=16G reproduce_table2_mse.pl embeddings_granite_65580x384.f16
%%
%% For quick testing:
%%   swipl --stack_limit=2G reproduce_table2_mse.pl embeddings_sample_100.csv

:- use_module(library(readutil)).
:- use_module(library(lists)).

block_size(32).

main :-
    current_prolog_flag(argv, Args),
    (Args = [Path|_] -> true ; Path = 'embeddings_sample_100.csv'),
    format("Prefix-VBR Table 2 Reproduction (MSE Comparison)~n"),
    format("Source: ~w~n~n", [Path]),

    %% Step 1: Read weights
    format("Loading embeddings...~n"),
    read_all_floats(Path, AllWeights),
    length(AllWeights, TotalWeights),
    format("Total weight values: ~w~n~n", [TotalWeights]),

    %% Step 2: Block the weights
    block_size(BS),
    chunk_list(AllWeights, BS, Blocks0),
    include(full_block(BS), Blocks0, Blocks),
    length(Blocks, NBlocks),
    format("Blocks of ~w: ~w~n~n", [BS, NBlocks]),

    %% Step 3: Quantize + measure each scheme
    format("Quantizing with Q8_0...~n"),
    maplist(q8_0_roundtrip, Blocks, Q8Recovered),
    append(Blocks, OrigFlat),
    append(Q8Recovered, Q8Flat),
    compute_mse(OrigFlat, Q8Flat, MSE_Q8),
    compute_snr(OrigFlat, MSE_Q8, SNR_Q8),

    format("Quantizing with Prefix-VBR...~n"),
    maplist(prefix_vbr_roundtrip, Blocks, VBRRecovered),
    append(VBRRecovered, VBRFlat),
    compute_mse(OrigFlat, VBRFlat, MSE_VBR),
    compute_snr(OrigFlat, MSE_VBR, SNR_VBR),

    format("Quantizing with mu-law (mu=15)...~n"),
    maplist(mulaw_roundtrip(15), Blocks, MuRecovered),
    append(MuRecovered, MuFlat),
    compute_mse(OrigFlat, MuFlat, MSE_Mu),
    compute_snr(OrigFlat, MSE_Mu, SNR_Mu),

    %% Step 4: Print Table 2
    Ratio_VBR is MSE_Q8 / MSE_VBR,
    Ratio_Mu is MSE_Q8 / MSE_Mu,
    format("~n================================================~n"),
    format("  TABLE 2: Quantization MSE comparison~n"),
    format("  All schemes: per-block scaling, 8 bits/weight~n"),
    format("================================================~n"),
    format("  ~w~t~25|~w~t~45|~w~t~55|~w~n",
           ['Scheme', 'MSE', 'SNR(dB)', 'vs Q8_0']),
    format("  ~`-t~65|~n"),
    format("  ~w~t~25|~e~t~45|~1f~t~55|~2fx~n",
           ['Prefix-VBR', MSE_VBR, SNR_VBR, Ratio_VBR]),
    format("  ~w~t~25|~e~t~45|~1f~t~55|~w~n",
           ['Q8_0 (baseline)', MSE_Q8, SNR_Q8, '1.00x']),
    format("  ~w~t~25|~e~t~45|~1f~t~55|~2fx~n",
           ['mu-law (mu=15)', MSE_Mu, SNR_Mu, Ratio_Mu]),
    format("~n"),
    halt.

main :-
    format("Error~n"), halt(1).

%% ================================================================
%% Q8_0: scale = max(|block|)/127, quantize to int8, dequantize
%% Matches ggml exactly: scale is truncated to FP16 for storage,
%% and dequantization uses the FP16-truncated scale.
%% ================================================================
q8_0_roundtrip(Block, Recovered) :-
    maplist([W, A]>>(A is abs(W)), Block, AbsBlock),
    max_list(AbsBlock, MaxAbs),
    (MaxAbs =:= 0.0
    ->  Recovered = Block
    ;   ScaleF64 is MaxAbs / 127.0,
        fp16_roundtrip(ScaleF64, ScaleF16),
        InvScale is 1.0 / ScaleF64,         % quantize with FP32-precision inverse
        maplist({InvScale, ScaleF16}/[W, R]>>(
            Q is max(-127, min(127, round(W * InvScale))),
            R is Q * ScaleF16                % dequantize with FP16-truncated scale
        ), Block, Recovered)
    ).

%% fp16_roundtrip(+Float64, -Float64RoundedToFP16)
%% Simulates IEEE 754 half-precision: 1 sign + 5 exponent + 10 mantissa.
%% Rounds the mantissa to 10 bits (round-to-nearest-even).
fp16_roundtrip(0.0, 0.0) :- !.
fp16_roundtrip(X, Result) :-
    Abs is abs(X),
    Sign is sign(X),
    %% Find the exponent
    Exp is floor(log(Abs) / log(2)),
    %% FP16 exponent range: -14 to 15 (biased: 1-30, 0=subnormal, 31=inf)
    (Exp < -14
    ->  %% Subnormal: mantissa = Abs / 2^(-14), quantize to 10 bits
        Mantissa is Abs / (2 ** (-14)),
        RoundedMantissa is round(Mantissa * 1024) / 1024,
        Result is Sign * RoundedMantissa * (2 ** (-14))
    ;   Exp > 15
    ->  Result is Sign * (1.0/0.0)   % infinity
    ;   %% Normal: mantissa in [1, 2), quantize fractional part to 10 bits
        Mantissa is Abs / (2 ** Exp),   % in [1.0, 2.0)
        Frac is Mantissa - 1.0,         % in [0.0, 1.0)
        RoundedFrac is round(Frac * 1024) / 1024,
        Result is Sign * (1.0 + RoundedFrac) * (2 ** Exp)
    ).

%% ================================================================
%% Prefix-VBR: per-block scale, then prefix-encode magnitude
%%   0xxxxxxx: 7-bit mantissa, [0, 32)   128 levels
%%   10xxxxxx: 6-bit mantissa, [32, 64)   64 levels
%%   110xxxxx: 5-bit mantissa, [64, 96)   32 levels
%%   1110xxxx: 4-bit mantissa, [96, 127)  16 levels
%% ================================================================
prefix_vbr_roundtrip(Block, Recovered) :-
    maplist([W, A]>>(A is abs(W)), Block, AbsBlock),
    max_list(AbsBlock, MaxAbs),
    (MaxAbs =:= 0.0
    ->  Recovered = Block
    ;   Scale is MaxAbs / 127.0,
        maplist({Scale}/[W, R]>>(
            Norm is W / Scale,
            AbsNorm is abs(Norm),
            vbr_quantize_dequant(AbsNorm, RecNorm),
            (Norm < 0 -> R is -(RecNorm * Scale) ; R is RecNorm * Scale)
        ), Block, Recovered)
    ).

vbr_quantize_dequant(AbsNorm, Recovered) :-
    (  AbsNorm < 32
    -> Q is max(0, min(127, round(AbsNorm * (127.0 / 32.0)))),
       Recovered is Q * (32.0 / 127.0)
    ;  AbsNorm < 64
    -> Q is max(0, min(63, round((AbsNorm - 32) * (63.0 / 32.0)))),
       Recovered is 32.0 + Q * (32.0 / 63.0)
    ;  AbsNorm < 96
    -> Q is max(0, min(31, round((AbsNorm - 64) * (31.0 / 32.0)))),
       Recovered is 64.0 + Q * (32.0 / 31.0)
    ;  Q is max(0, min(15, round((AbsNorm - 96) * (15.0 / 31.0)))),
       Recovered is 96.0 + Q * (31.0 / 15.0)
    ).

%% ================================================================
%% mu-law companding: compress, quantize to int8, expand
%% ================================================================
mulaw_roundtrip(Mu, Block, Recovered) :-
    maplist([W, A]>>(A is abs(W)), Block, AbsBlock),
    max_list(AbsBlock, MaxAbs),
    (MaxAbs =:= 0.0
    ->  Recovered = Block
    ;   Scale is MaxAbs / 127.0,
        maplist({Scale, Mu}/[W, R]>>(
            Norm is W / Scale,
            AbsNorm is abs(Norm) / 127.0,
            %% compress
            Compressed is sign(Norm) * log(1 + Mu * AbsNorm) / log(1 + Mu),
            %% quantize
            Q is max(-127, min(127, round(Compressed * 127))),
            %% decompress
            Decompressed is Q / 127.0,
            Expanded is sign(Decompressed) * ((1 + Mu) ** abs(Decompressed) - 1) / Mu * 127.0,
            R is Expanded * Scale
        ), Block, Recovered)
    ).

%% ================================================================
%% MSE and SNR
%% ================================================================
compute_mse(Orig, Recovered, MSE) :-
    maplist([O, R, E]>>(D is O - R, E is D * D), Orig, Recovered, Errors),
    sumlist(Errors, SumErr),
    length(Errors, N),
    MSE is SumErr / N.

compute_snr(Orig, MSE, SNR) :-
    maplist([O, S]>>(S is O * O), Orig, Squares),
    sumlist(Squares, SumSq),
    length(Orig, N),
    Var is SumSq / N,
    (MSE > 0 -> SNR is 10 * log10(Var / MSE) ; SNR = 999.0).

%% ================================================================
%% File reading (CSV or FP16 binary)
%% ================================================================
read_all_floats(Path, AllFloats) :-
    (  sub_atom(Path, _, _, 0, '.f16')
    -> read_fp16_binary(Path, AllFloats)
    ;  read_csv_floats(Path, AllFloats)
    ).

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
fp16_decode(_, 31, _, 0.0) :- !.
fp16_decode(Sign, Exp, Mant, F) :-
    F is (-1)**Sign * (1 + Mant / 1024) * 2**(Exp - 15).

%% ================================================================
%% Utilities
%% ================================================================
chunk_list([], _, []) :- !.
chunk_list(List, N, [Chunk|Rest]) :-
    length(Chunk, N),
    append(Chunk, Remaining, List), !,
    chunk_list(Remaining, N, Rest).
chunk_list(Remainder, _, [Remainder]).

full_block(BS, Block) :- length(Block, BS).

:- initialization(main, main).
