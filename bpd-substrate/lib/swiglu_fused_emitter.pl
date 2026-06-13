%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% swiglu_fused_emitter.pl — Emit fused SwiGLU (silu×mul) LLVM IR.
%%
%% THE INAUGURAL FUSED KERNEL FROM FACTS.
%%
%% Fuses: bpd_silu_cpu + bpd_mul_cpu into ONE pass.
%% Eliminates: writing silu_out to memory + re-reading for mul.
%% Saves: 2×N×4 bytes of memory traffic (write+read of intermediate).
%%
%% CRITICAL BIT-IDENTITY DETAILS (from Iyun):
%%   - Uses SCALAR libm expf (NOT polynomial approximation)
%%   - Uses DIVIDE form: x / (1+exp(-x)) NOT reciprocal-mul x*(1/(1+exp(-x)))
%%   - The two forms differ by 1 ULP — must use divide form to match
%%   - The fusion win is the eliminated buffer, NOT exp vectorization
%%
%% Verification: fused == unfused == ggml (transitive 0 ULP)
%%
%% Author: mavchin (2026-06-03)

:- module(swiglu_fused_emitter, [emit_swiglu_fused/1]).

emit_swiglu_fused(OutFile) :-
    open(OutFile, write, S),
    format(S, '; SwiGLU fused kernel: silu(gate) * up in one pass~n', []),
    format(S, '; Generated from kernel_spec facts by swiglu_fused_emitter.pl~n', []),
    format(S, '; Eliminates intermediate buffer write+read (2*N*4 bytes saved)~n', []),
    format(S, '; BIT-IDENTICAL to bpd_silu_cpu + bpd_mul_cpu (divide form, scalar expf)~n~n', []),
    format(S, 'declare float @expf(float)~n~n', []),
    emit_fused_function(S),
    close(S),
    format("Emitted fused SwiGLU to ~w~n", [OutFile]).

emit_fused_function(S) :-
    format(S, '; Fused: for each element, compute silu(gate[i]) * up[i] in one pass.~n', []),
    format(S, '; No intermediate silu_out buffer — result goes directly to output.~n', []),
    format(S, '; Uses divide form: x / (1 + expf(-x)) to match bpd_silu_cpu exactly.~n', []),
    format(S, 'define void @bpd_swiglu_fused_cpu(ptr %gate, ptr %up, ptr %out, i32 %n) {~n', []),
    format(S, 'entry:~n', []),
    format(S, '  %cmp = icmp sgt i32 %n, 0~n', []),
    format(S, '  br i1 %cmp, label %loop, label %done~n~n', []),
    format(S, 'loop:~n', []),
    format(S, '  %i = phi i32 [ 0, %entry ], [ %i_next, %loop ]~n', []),
    format(S, '~n', []),
    %% Load gate[i]
    format(S, '  ; Load gate[i] and up[i]~n', []),
    format(S, '  %gate_ptr = getelementptr float, ptr %gate, i32 %i~n', []),
    format(S, '  %g = load float, ptr %gate_ptr~n', []),
    format(S, '  %up_ptr = getelementptr float, ptr %up, i32 %i~n', []),
    format(S, '  %u = load float, ptr %up_ptr~n', []),
    format(S, '~n', []),
    %% Compute silu(gate[i]) = gate[i] / (1 + expf(-gate[i]))
    %% MUST use divide form + scalar expf for bit-identity
    format(S, '  ; silu(g) = g / (1 + expf(-g))  [DIVIDE form, scalar expf]~n', []),
    format(S, '  %neg_g = fneg float %g~n', []),
    format(S, '  %exp_neg = call float @expf(float %neg_g)~n', []),
    format(S, '  %one_plus_exp = fadd float 1.0, %exp_neg~n', []),
    format(S, '  %silu_g = fdiv float %g, %one_plus_exp~n', []),
    format(S, '~n', []),
    %% Multiply by up[i] — the fusion point
    format(S, '  ; Fused: silu(gate[i]) * up[i] — no intermediate buffer!~n', []),
    format(S, '  %result = fmul float %silu_g, %u~n', []),
    format(S, '~n', []),
    %% Store result
    format(S, '  ; Store directly to output~n', []),
    format(S, '  %out_ptr = getelementptr float, ptr %out, i32 %i~n', []),
    format(S, '  store float %result, ptr %out_ptr~n', []),
    format(S, '~n', []),
    %% Loop
    format(S, '  %i_next = add i32 %i, 1~n', []),
    format(S, '  %done_cmp = icmp sge i32 %i_next, %n~n', []),
    format(S, '  br i1 %done_cmp, label %done, label %loop~n~n', []),
    format(S, 'done:~n', []),
    format(S, '  ret void~n', []),
    format(S, '}~n', []).
