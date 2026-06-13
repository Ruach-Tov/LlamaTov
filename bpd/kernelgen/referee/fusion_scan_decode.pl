%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
:- use_module('lib/fusion_rules').
:- dynamic op_kind/2, op_input/2, op_output/2, op_reads/3, op_writes/3.
fusion_rules:op_kind(Op, K) :- op_kind(Op, K).
fusion_rules:op_input(Op, T) :- op_input(Op, T).
fusion_rules:op_output(Op, T) :- op_output(Op, T).
fusion_rules:op_reads(Op, T, R) :- op_reads(Op, T, R).
fusion_rules:op_writes(Op, T, R) :- op_writes(Op, T, R).

%% ── CURRENT decode graph (one layer + logits), CORRECT region terms ──
%% Encodes the ops we actually launch (winning config: QFUSED/ADDRES OFF, so quant and gemv are
%% SEPARATE kernels, residual-add is a SEPARATE k_add). The scanner finds fusion REMAINING.
%% region terms: quant->gemv = region(quantized_activation,_); matmul->elementwise = matmul_output/elementwise.

%% --- attention ---
op_kind(attn_rms, build_norm(rms)). op_output(attn_rms, x1). op_writes(attn_rms, x1, region(norm_output, e896)).
op_kind(q_quant, k_quant_q8). op_input(q_quant, x1). op_reads(q_quant, x1, region(norm_output,e896)).
  op_output(q_quant, xqa). op_writes(q_quant, xqa, region(quantized_activation, e896)).
%% xqa feeds q,k,v gemvs = THREE consumers -> quant_into_gemv must NOT fire (no_other_consumers fails)
op_kind(q_gemv, k_q8_0_gemv). op_input(q_gemv, xqa). op_reads(q_gemv, xqa, region(quantized_activation,e896)). op_output(q_gemv, q).
op_kind(k_gemv, k_q8_0_gemv). op_input(k_gemv, xqa). op_reads(k_gemv, xqa, region(quantized_activation,e896)). op_output(k_gemv, kk).
op_kind(v_gemv, k_q8_0_gemv). op_input(v_gemv, xqa). op_reads(v_gemv, xqa, region(quantized_activation,e896)). op_output(v_gemv, vv).
op_kind(rope_q, ggml_rope_ext). op_input(rope_q, q). op_reads(rope_q, q, region(elementwise,e896)). op_output(rope_q, qr).
  op_writes(q_gemv, q, region(matmul_output, e896)).
op_kind(o_quant, k_quant_q8). op_output(o_quant, xqo). op_writes(o_quant, xqo, region(quantized_activation,e896)).
op_kind(o_gemv, k_q8_0_gemv). op_input(o_gemv, xqo). op_reads(o_gemv, xqo, region(quantized_activation,e896)). op_output(o_gemv, ap).
  op_writes(o_quant, xqo, region(quantized_activation,e896)).
op_kind(attn_add, ggml_add). op_input(attn_add, ap). op_reads(attn_add, ap, region(elementwise,e896)). op_output(attn_add, rmid).
  op_writes(o_gemv, ap, region(matmul_output,e896)).

%% --- ffn ---
op_kind(ffn_rms, build_norm(rms)). op_output(ffn_rms, x2). op_writes(ffn_rms, x2, region(norm_output,e896)).
op_kind(ffn_quant, k_quant_q8). op_input(ffn_quant, x2). op_reads(ffn_quant, x2, region(norm_output,e896)).
  op_output(ffn_quant, xqf). op_writes(ffn_quant, xqf, region(quantized_activation, e896)).
%% xqf feeds gate,up = TWO consumers -> no fusion
op_kind(gate_gemv, k_q8_0_gemv). op_input(gate_gemv, xqf). op_reads(gate_gemv, xqf, region(quantized_activation,e896)). op_output(gate_gemv, gate).
  op_writes(gate_gemv, gate, region(matmul_output, e4864)).
op_kind(up_gemv, k_q8_0_gemv). op_input(up_gemv, xqf). op_reads(up_gemv, xqf, region(quantized_activation,e896)). op_output(up_gemv, up).
  op_writes(up_gemv, up, region(matmul_output, e4864)).
%% silu(gate) -> single consumer (silu_mul). The gate_gemv -> silu is matmul->elementwise (epilogue!)
op_kind(silu, ggml_silu). op_input(silu, gate). op_reads(silu, gate, region(elementwise, e4864)). op_output(silu, gact).
  op_writes(silu, gact, region(elementwise, e4864)).
op_kind(silu_mul, ggml_mul). op_input(silu_mul, gact). op_reads(silu_mul, gact, region(elementwise, e4864)). op_output(silu_mul, fmid).
op_kind(down_quant, k_quant_q8). op_input(down_quant, fmid). op_reads(down_quant, fmid, region(elementwise,e4864)).
  op_output(down_quant, xqd). op_writes(down_quant, xqd, region(quantized_activation, e4864)).
%% xqd feeds ONLY down_gemv = single consumer -> quant_into_gemv CAN fire
op_kind(down_gemv, k_q8_0_gemv). op_input(down_gemv, xqd). op_reads(down_gemv, xqd, region(quantized_activation,e4864)). op_output(down_gemv, fp).
  op_writes(down_gemv, fp, region(matmul_output, e896)).
op_kind(ffn_add, ggml_add). op_input(ffn_add, fp). op_reads(ffn_add, fp, region(elementwise,e896)). op_output(ffn_add, rout).

%% --- logits (folded device path) ---
op_kind(final_rms, build_norm(rms)). op_output(final_rms, xf). op_writes(final_rms, xf, region(norm_output,e896)).
op_kind(logit_quant, k_quant_q8). op_input(logit_quant, xf). op_reads(logit_quant, xf, region(norm_output,e896)).
  op_output(logit_quant, xql). op_writes(logit_quant, xql, region(quantized_activation, e896)).
%% xql feeds ONLY vocab_gemv = single consumer -> quant_into_gemv CAN fire
op_kind(vocab_gemv, k_q8_0_gemv). op_input(vocab_gemv, xql). op_reads(vocab_gemv, xql, region(quantized_activation,e896)). op_output(vocab_gemv, logits).

:- initialization(main).
main :-
    Rules = [epilogue_matmul_elementwise, elementwise_chain, layout_transparent, quant_into_gemv],
    enumerate_valid_fusions(Rules, Fs), sort(Fs, U), length(U,N),
    format("~n=== FUSION SCAN (current decode, winning config) -> ~w opportunities ===~n",[N]),
    forall(member(fusion(R,Ops,Eq),U), format("  [~w] ~w (~w)~n",[R,Ops,Eq])),
    nl, halt.
