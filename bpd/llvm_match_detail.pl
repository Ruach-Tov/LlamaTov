%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% llvm_match_detail.pl — Detail view for a single pattern × reference cell.
%%
%% Generates an SVG showing each op in the pattern as its own full-sized
%% colored block with the op name and ULP value.
%%
%% Usage:
%%   swipl -g "main" llvm_match_detail.pl -- unary_elementwise ggml_sse3 /tmp/detail.o.svg

:- module(llvm_match_detail, [main/0, emit_detail_svg/3]).

:- use_module(library(lists)).

%% Load the facts
:- (  catch(use_module('/tmp/bpd-generated/output/llvm_op_match.o'), _, fail)
   -> true
   ;  catch(use_module('lib/llvm_op_match'), _, fail)
   -> true
   ;  true
   ).

%% Color scheme (same as main dashboard)
detail_color(0, ggml_sse3, '#2a6496') :- !.   % blue = IR-match (verified via IR diff)
detail_color(0, _, '#4a8a4a') :- !.            % green = 0 ULP (value comparison only)
detail_color(skipped, _, '#7a7a7a') :- !.      % dark grey
detail_color(untested, _, '#b0a890') :- !.     % warm grey
detail_color(ULP, _, '#b8860b') :- integer(ULP), ULP > 0, ULP =< 100, !.  % yellow
detail_color(ULP, _, '#a04040') :- integer(ULP), ULP > 100, !.            % red
detail_color(_, _, '#b0a890').                  % fallback grey

detail_text_color('#2a6496', '#ffffff') :- !.   % white on blue
detail_text_color('#4a8a4a', '#ffffff') :- !.   % white on green
detail_text_color('#a04040', '#ffffff') :- !.   % white on red
detail_text_color('#7a7a7a', '#ffffff') :- !.   % white on dark grey
detail_text_color(_, '#1a1a1a').                % dark text otherwise

ulp_label(0, ggml_sse3, 'IR-match') :- !.   %% blue — LLVM IR structurally verified
ulp_label(0, _, '0 ULP') :- !.              %% green — value comparison only
ulp_label(skipped, _, 'skipped') :- !.
ulp_label(untested, _, 'untested') :- !.
ulp_label(ULP, _, Label) :- integer(ULP), format(atom(Label), '~w ULP', [ULP]).

%% All ops per pattern (canonical order)
pattern_ops(unary_elementwise, [ggml_relu, ggml_silu, ggml_sigmoid, ggml_tanh,
    ggml_gelu, ggml_softplus, ggml_leaky_relu, ggml_elu,
    ggml_hardsigmoid, ggml_softsign, ggml_selu, ggml_clamp]).
pattern_ops(binary_elementwise, [ggml_scale]).
pattern_ops(reduction, [ggml_mul_mat, ggml_sum, ggml_mean, ggml_max,
    ggml_min, ggml_argmax, ggml_argmin]).
pattern_ops(reduction_then_elementwise, [ggml_norm, ggml_rms_norm, ggml_soft_max,
    ggml_log_softmax, ggml_group_norm, ggml_l2_norm]).
pattern_ops(scan, [ggml_cumsum, ggml_cumprod]).
pattern_ops(conv_im2col, [ggml_conv_1d, ggml_conv_2d, ggml_conv_3d,
    ggml_conv_transpose_1d, ggml_conv_transpose_2d, ggml_conv_transpose_3d]).
pattern_ops(pool_reduce, [ggml_pool_1d, ggml_pool_2d, ggml_pool_3d]).
pattern_ops(loss_reduce, [ggml_mse_loss, ggml_cross_entropy_loss, ggml_hinge_loss,
    ggml_huber_loss, ggml_kl_div_loss, ggml_triplet_margin_loss]).
pattern_ops(flash_attention, [ggml_flash_attn_ext]).

%% Get the best available fact for an op
best_fact(Pattern, Op, Ref, ULP, ActualRef) :-
    ulp_op_match(Pattern, Op, Ref, ULP, _), !,
    ActualRef = Ref.
best_fact(Pattern, Op, _Ref, ULP, scalar) :-
    ulp_op_match(Pattern, Op, scalar, ULP, _), !.
best_fact(_Pattern, _Op, _Ref, untested, untested).

emit_detail_svg(Pattern, Ref, OutPath) :-
    pattern_ops(Pattern, Ops),
    length(Ops, N),
    
    %% Layout
    CellW = 500, CellH = 40,
    LabelW = 220,
    Pad = 20,
    TitleH = 80,
    SvgW is LabelW + CellW + Pad * 3,
    SvgH is TitleH + N * (CellH + 4) + Pad * 2 + 40,
    
    open(OutPath, write, S),
    format(S, '<svg xmlns="http://www.w3.org/2000/svg" width="~w" height="~w" viewBox="0 0 ~w ~w" font-family="Georgia, serif">~n',
        [SvgW, SvgH, SvgW, SvgH]),
    format(S, '  <rect x="0" y="0" width="~w" height="~w" fill="#f8f5ee"/>~n', [SvgW, SvgH]),
    
    %% Title
    MidX is SvgW // 2,
    BackY is SvgH - 15,
    format(S, '  <text x="~w" y="36" font-size="18" font-weight="bold" fill="#3a2a1a" text-anchor="middle">~w × ~w</text>~n',
        [MidX, Pattern, Ref]),
    format(S, '  <text x="~w" y="56" font-size="11" fill="#5a4a3a" text-anchor="middle">Detail view · one cell per op · click browser back to return</text>~n',
        [MidX]),
    
    %% Back link
    format(S, '  <a href="/static/llvm_match_status.html">~n', []),
    format(S, '    <text x="~w" y="~w" font-size="12" fill="#2a6496" text-decoration="underline">← back to dashboard</text>~n',
        [Pad, BackY]),
    format(S, '  </a>~n', []),
    
    %% Ops
    Y0 is TitleH,
    emit_op_rows(S, Ops, Pattern, Ref, Y0, Pad, LabelW, CellW, CellH),
    
    format(S, '</svg>~n', []),
    close(S),
    format("Wrote: ~w~n", [OutPath]).

emit_op_rows(_, [], _, _, _, _, _, _, _).
emit_op_rows(S, [Op|Rest], Pattern, Ref, Y, Pad, LabelW, CellW, CellH) :-
    best_fact(Pattern, Op, Ref, ULP, ActualRef),
    detail_color(ULP, ActualRef, Color),
    detail_text_color(Color, TextColor),
    ulp_label(ULP, ActualRef, Label),
    
    %% Op name label
    LabelX is Pad,
    TextY is Y + CellH // 2 + 5,
    format(S, '  <text x="~w" y="~w" font-size="13" fill="#3a2a1a">~w</text>~n',
        [LabelX, TextY, Op]),
    
    %% Colored cell
    CellX is Pad + LabelW,
    format(S, '  <rect x="~w" y="~w" width="~w" height="~w" fill="~w" rx="4" stroke="#3a2a1a" stroke-width="1"/>~n',
        [CellX, Y, CellW, CellH, Color]),
    
    %% Label inside cell
    CellMidX is CellX + CellW // 2,
    format(S, '  <text x="~w" y="~w" font-size="13" font-weight="bold" fill="~w" text-anchor="middle">~w</text>~n',
        [CellMidX, TextY, TextColor, Label]),
    
    %% Evidence text (small, right-aligned)
    (   ulp_op_match(Pattern, Op, _, _, Evidence),
        Evidence \= ''
    ->  EvidX is CellX + CellW - 5,
        format(S, '  <text x="~w" y="~w" font-size="9" fill="~w" text-anchor="end" opacity="0.7">~w</text>~n',
            [EvidX, TextY, TextColor, Evidence])
    ;   true
    ),
    
    Y1 is Y + CellH + 4,
    emit_op_rows(S, Rest, Pattern, Ref, Y1, Pad, LabelW, CellW, CellH).

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [PatternAtom, RefAtom, OutPath | _]
    ->  atom_string(Pattern, PatternAtom),
        atom_string(Ref, RefAtom),
        emit_detail_svg(Pattern, Ref, OutPath)
    ;   format("Usage: swipl -g main llvm_match_detail.pl -- <pattern> <ref> <outpath>~n"),
        halt(1)
    ),
    halt.
