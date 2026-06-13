%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ffn_expander.pl — Expand opaque build_ffn ops into primitive BPD facts.
%%
%% The C source for Qwen2 FFN uses the OPAQUE `build_ffn(...)` builder
%% call which encapsulates the SwiGLU pattern. The lifter (qkv_lifter)
%% preserves this as a single op for round-trip fidelity. But fusion
%% analysis needs the EXPANDED primitives:
%%
%%   gate_out = build_lora_mm(ffn_gate, input)
%%   up_out   = build_lora_mm(ffn_up, input)
%%   silu_out = ggml_silu(gate_out)
%%   inner    = ggml_mul(silu_out, up_out)
%%   output   = build_lora_mm(ffn_down, inner)
%%
%% This module bridges the two views: take lifted facts with an opaque
%% build_ffn op, produce expanded facts with primitives.
%%
%% Per Heath's "proceed to other rule kinds" + "FFN BPD vocabulary" path.

:- module(ffn_expander, [
    expand_ffn_ops/2,
    expand_one_ffn/3
]).

%% ────────────────────────────────────────────────────────────────────
%% expand_ffn_ops(+InputFacts, -ExpandedFacts)
%%
%% Find all opaque build_ffn ops in the input facts and replace each
%% with its expanded primitive form. Non-FFN facts are preserved
%% unchanged.
%%
%% Assumes Qwen2-style SwiGLU FFN (LLM_FFN_SILU + LLM_FFN_PAR variant).
%% Other FFN variants (gelu, relu, geglu) would need their own expanders.

expand_ffn_ops(InputFacts, ExpandedFacts) :-
    %% Find all build_ffn ops
    findall(Op,
            ( member(op(Op), InputFacts),
              member(op_kind(Op, build_ffn), InputFacts)
            ),
            FfnOps),
    expand_ffn_ops_iter(FfnOps, InputFacts, ExpandedFacts).

expand_ffn_ops_iter([], Facts, Facts).
expand_ffn_ops_iter([Op | Rest], Facts, Final) :-
    expand_one_ffn(Op, Facts, Facts1),
    expand_ffn_ops_iter(Rest, Facts1, Final).

%% ────────────────────────────────────────────────────────────────────
%% expand_one_ffn(+OpName, +InputFacts, -OutputFacts)
%%
%% Expand a single build_ffn op into its constituent primitives.
%% The original opaque op's facts are removed; primitive facts are added.
%%
%% Input op_inputs convention (matches Qwen2 build_ffn call signature):
%%   [pre_norm_out, ffn_up_w, ffn_up_b, ffn_up_s,
%%                  ffn_gate_w, ffn_gate_b, ffn_gate_s,
%%                  ffn_down_w, ffn_down_b, ffn_down_s,
%%    ffn_act, type_op, type_gate, il]
%% For optional fields (biases, scales), NULL is the convention.
%%
%% Output op_output: the FFN block's final tensor (post-down projection).

expand_one_ffn(OpName, InputFacts, OutputFacts) :-
    %% Get the original op's properties
    member(op_inputs(OpName, OrigInputs), InputFacts),
    member(op_output(OpName, FinalOutput), InputFacts),
    member(sequence(Block, OpName, BaseSeq), InputFacts),

    %% Extract the architecture-parameter inputs.
    %% The first input is the normed activation; the rest are weights/biases.
    %% Conservative: assume positions match build_ffn convention.
    OrigInputs = [InputTensor | RestArgs],
    extract_ffn_params(RestArgs, FfnUpW, FfnGateW, FfnDownW),

    %% Generate primitive op names based on the original op
    atom_concat(OpName, '_gate_mul', GateMulName),
    atom_concat(OpName, '_up_mul', UpMulName),
    atom_concat(OpName, '_silu', SiluName),
    atom_concat(OpName, '_swiglu_mul', SwigluMulName),
    atom_concat(OpName, '_down_mul', DownMulName),

    %% Intermediate tensor names
    atom_concat(OpName, '_gate_out', GateOut),
    atom_concat(OpName, '_up_out', UpOut),
    atom_concat(OpName, '_gate_silu', GateSilu),
    atom_concat(OpName, '_inner', Inner),

    %% Sequence numbers: insert 5 primitives where 1 opaque op was
    Seq1 is BaseSeq,
    Seq2 is BaseSeq + 1,
    Seq3 is BaseSeq + 2,
    Seq4 is BaseSeq + 3,
    Seq5 is BaseSeq + 4,

    %% Generate the expanded primitive facts
    ExpandedOps = [
        %% Gate projection
        op(GateMulName),
        op_kind(GateMulName, build_lora_mm),
        op_inputs(GateMulName, [FfnGateW, InputTensor]),
        op_output(GateMulName, GateOut),
        op_level(GateMulName, builder),
        sequence(Block, GateMulName, Seq1),

        %% Up projection
        op(UpMulName),
        op_kind(UpMulName, build_lora_mm),
        op_inputs(UpMulName, [FfnUpW, InputTensor]),
        op_output(UpMulName, UpOut),
        op_level(UpMulName, builder),
        sequence(Block, UpMulName, Seq2),

        %% SiLU activation
        op(SiluName),
        op_kind(SiluName, ggml_silu),
        op_inputs(SiluName, [GateOut]),
        op_output(SiluName, GateSilu),
        op_level(SiluName, primitive),
        sequence(Block, SiluName, Seq3),

        %% Element-wise multiply (SwiGLU)
        op(SwigluMulName),
        op_kind(SwigluMulName, ggml_mul),
        op_inputs(SwigluMulName, [GateSilu, UpOut]),
        op_output(SwigluMulName, Inner),
        op_level(SwigluMulName, primitive),
        sequence(Block, SwigluMulName, Seq4),

        %% Down projection
        op(DownMulName),
        op_kind(DownMulName, build_lora_mm),
        op_inputs(DownMulName, [FfnDownW, Inner]),
        op_output(DownMulName, FinalOutput),
        op_level(DownMulName, builder),
        sequence(Block, DownMulName, Seq5),

        %% Provenance: record the expansion
        expanded_from(GateMulName, OpName),
        expanded_from(UpMulName, OpName),
        expanded_from(SiluName, OpName),
        expanded_from(SwigluMulName, OpName),
        expanded_from(DownMulName, OpName)
    ],

    %% Remove the original opaque op's facts
    remove_facts_for_op(OpName, InputFacts, FactsAfterRemoval),

    %% Append expanded facts
    append(ExpandedOps, FactsAfterRemoval, OutputFacts).

%% extract_ffn_params(+RestArgs, -UpW, -GateW, -DownW)
%%   From build_ffn's call args, extract the weight tensor names.
%%   Order matches llama.cpp's build_ffn signature:
%%     up_w, up_b, up_s, gate_w, gate_b, gate_s, down_w, down_b, down_s, ...
%%   We skip biases and scales (NULL for standard FFN).
extract_ffn_params([UpW, _UpB, _UpS,
                    GateW, _GateB, _GateS,
                    DownW | _Rest], UpW, GateW, DownW) :- !.
%% Fallback for shorter arg lists (test fixtures)
extract_ffn_params([UpW, GateW, DownW | _], UpW, GateW, DownW).

%% remove_facts_for_op(+Op, +InputFacts, -OutputFacts)
%%   Remove all facts mentioning Op as the subject.
remove_facts_for_op(Op, Facts, Cleaned) :-
    exclude(fact_about_op(Op), Facts, Cleaned).

fact_about_op(Op, op(Op)).
fact_about_op(Op, op_kind(Op, _)).
fact_about_op(Op, op_inputs(Op, _)).
fact_about_op(Op, op_output(Op, _)).
fact_about_op(Op, op_level(Op, _)).
fact_about_op(Op, sequence(_, Op, _)).
