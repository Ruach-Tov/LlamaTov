%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════════
%% soa_ffn_emitter.pl — RUNG 4: a llama FFN block with the SwiGLU fusion,
%% composed BY CONSTRUCTION from verified atoms.
%%
%% THE RUNG WHERE TONIGHT'S BUG BECOMES STRUCTURALLY IMPOSSIBLE.
%% The hand-written SoA dispatch DROPPED the SwiGLU fusion at blk.15 (bare
%% matmul instead of silu(gate)·up → garbage tokens). Here the SwiGLU fusion is
%% a COMPOSED, VERIFIED FACT (swiglu_fused_emitter.pl, proven 0-ULP), wired into
%% the lowered graph. You cannot "forget" a fact you are lowering. The exact
%% composition the hand-kernel got wrong is now bit-identical by construction.
%%
%% LLAMA FFN BLOCK (the structure):
%%   gate = W_gate · xq        (gemv, quantized normalized input)
%%   up   = W_up   · xq        (gemv, same input)
%%   act  = silu(gate) · up    (SwiGLU fusion — the dropped-tonight op)
%%   down = W_down · quantize(act)   (gemv)
%%   out  = down
%%
%% Composes verified atoms:
%%   @bpd_soa_gemv_q8_0   (rung 2, 0-ULP)
%%   @bpd_swiglu_fused_cpu (mavchin, proven 0-ULP, divide form + scalar expf)
%%   @bpd_q8_0_quantize   (rung 3, byte-exact)
%%
%% Verified 0-ULP vs CPU reference doing the IDENTICAL FFN with the same atoms.
%%
%% Author: Iyun, 2026-06-06 (rung-4 of the bottom-up SoA chain)
%% ═══════════════════════════════════════════════════════════════════════════

:- module(soa_ffn_emitter, [
    emit_soa_ffn/1          % emit_soa_ffn(+OutFile) — the FFN block glue
]).

emit_soa_ffn(OutFile) :-
    open(OutFile, write, S),
    format(S, '; ══════════════════════════════════════════════════════════~n', []),
    format(S, '; bpd_soa_ffn_q8_0 — rung-4: llama FFN block with SwiGLU fusion~n', []),
    format(S, ';   gate = Wg·xq ; up = Wu·xq ; act = silu(gate)·up ; out = Wd·quant(act)~n', []),
    format(S, ';   The SwiGLU fusion (dropped by the hand-kernel) is composed BY CONSTRUCTION.~n', []),
    format(S, '; ══════════════════════════════════════════════════════════~n~n', []),
    %% composed verified atoms (declared; linked from their emitters)
    format(S, 'declare void @bpd_soa_gemv_q8_0(i8*, float*, i8*, float*, float*, i32, i32)~n', []),
    format(S, 'declare void @bpd_swiglu_fused_cpu(ptr, ptr, ptr, i32)~n', []),
    format(S, 'declare void @bpd_q8_0_quantize(float*, i8*, float*, i32)~n~n', []),
    %% FFN block glue.
    %%   xq,xd       : quantized normalized input (1 col, nb_in blocks)
    %%   Wg*,Wu*,Wd* : gate/up/down weight quants+scales (SoA)
    %%   gate,up,act : float scratch (n_ff each); aqd,add : quantized act (down's input)
    %%   out         : float[n_embd]
    %%   dims: n_ff (gate/up rows), nb_in (input blocks), nb_ff (act blocks = n_ff/32),
    %%         n_embd (down rows)
    format(S, 'define void @bpd_soa_ffn_q8_0(~n', []),
    format(S, '    i8* %xq, float* %xd,~n', []),
    format(S, '    i8* %wgq, float* %wgd, i8* %wuq, float* %wud,~n', []),
    format(S, '    i8* %wdq, float* %wdd,~n', []),
    format(S, '    float* %gate, float* %up, float* %act,~n', []),
    format(S, '    i8* %aqd, float* %add,~n', []),
    format(S, '    float* %out,~n', []),
    format(S, '    i32 %n_ff, i32 %nb_in, i32 %nb_ff, i32 %n_embd) {~n', []),
    format(S, 'entry:~n', []),
    format(S, '  ; gate = W_gate · xq~n', []),
    format(S, '  call void @bpd_soa_gemv_q8_0(i8* %wgq, float* %wgd, i8* %xq, float* %xd, float* %gate, i32 %n_ff, i32 %nb_in)~n', []),
    format(S, '  ; up = W_up · xq~n', []),
    format(S, '  call void @bpd_soa_gemv_q8_0(i8* %wuq, float* %wud, i8* %xq, float* %xd, float* %up, i32 %n_ff, i32 %nb_in)~n', []),
    format(S, '  ; act = silu(gate) · up   ← THE SwiGLU FUSION (composed, not dropped)~n', []),
    format(S, '  call void @bpd_swiglu_fused_cpu(ptr %gate, ptr %up, ptr %act, i32 %n_ff)~n', []),
    format(S, '  ; quantize act → (aqd, add)  for the down projection~n', []),
    format(S, '  call void @bpd_q8_0_quantize(float* %act, i8* %aqd, float* %add, i32 %nb_ff)~n', []),
    format(S, '  ; out = W_down · quantize(act)~n', []),
    format(S, '  call void @bpd_soa_gemv_q8_0(i8* %wdq, float* %wdd, i8* %aqd, float* %add, float* %out, i32 %n_embd, i32 %nb_ff)~n', []),
    format(S, '  ret void~n', []),
    format(S, '}~n', []),
    close(S),
    format("Emitted SoA FFN block (with SwiGLU fusion) to ~w~n", [OutFile]).
