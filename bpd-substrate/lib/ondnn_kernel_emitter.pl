%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ondnn_kernel_emitter.pl — Generate LLVM IR from JIT instruction facts.
%%
%% GENERIC emitter: walks jit_insn/4 facts from any oneDNN kernel
%% and generates matching vectorized LLVM IR.
%%
%% To match a NEW oneDNN kernel:
%%   1. Disassemble the JIT binary
%%   2. Write jit_insn/4 facts (one per instruction)
%%   3. Write jit_const/3 facts (the constant table)
%%   4. Call emit_kernel/2 — LLVM IR is generated automatically
%%
%% Usage:
%%   swipl -g 'emit_kernel(gelu_erf, "gelu_ondnn.ll"), halt' ondnn_kernel_emitter.pl
%%
%% Author: mavchin (2026-06-01)

:- module(ondnn_kernel_emitter, [emit_kernel/2]).

:- use_module(llvm_emit).

%% ============================================================
%% Kernel registry — which fact module defines each kernel
%% ============================================================

:- use_module(ondnn_jit_facts).

%% ============================================================
%% LLVM IR type for our vectors
%% ============================================================

vec_type('<8 x float>').
ivec_type('<8 x i32>').

%% ============================================================
%% SSA name generation from fact names
%% ============================================================

ssa(Name, SSA) :- format(atom(SSA), '%~w', [Name]).

%% ============================================================
%% Emit a complete kernel .ll file
%% ============================================================

emit_kernel(KernelName, OutFile) :-
    open(OutFile, write, S),
    emit_header(S, "x86_64-unknown-linux-gnu"),
    format(S, '; Kernel ~w — generated from jit_insn/4 facts~n', [KernelName]),
    format(S, '; Each LLVM instruction corresponds to one jit_insn fact.~n~n', []),
    %% Declares
    format(S, 'declare <8 x float> @llvm.floor.v8f32(<8 x float>)~n', []),
    format(S, 'declare <8 x float> @llvm.minnum.v8f32(<8 x float>, <8 x float>)~n', []),
    format(S, 'declare <8 x float> @llvm.maxnum.v8f32(<8 x float>, <8 x float>)~n', []),
    format(S, 'declare float @erff(float)~n~n', []),
    %% The vectorized kernel function
    emit_vec_function(S, KernelName),
    %% The loop wrapper
    emit_loop_wrapper(S, KernelName),
    close(S),
    format("Emitted kernel ~w to ~w~n", [KernelName, OutFile]).

%% ============================================================
%% Emit the vectorized function from jit_insn facts
%% ============================================================

emit_vec_function(S, _KernelName) :-
    format(S, 'define void @kernel_vec8(ptr %in_ptr, ptr %out_ptr) #0 {~n', []),
    format(S, 'entry:~n', []),
    %% First: emit all constant broadcasts
    format(S, '  ; --- constant broadcasts ---~n', []),
    forall(jit_const(_, Name, HexVal),
           emit_const_broadcast(S, Name, HexVal)),
    format(S, '~n', []),
    %% Walk instructions in offset order
    findall(Off-Op-Dst-Src, jit_insn(Off, Op, Dst, Src), Insns),
    msort(Insns, Sorted),
    format(S, '  ; --- kernel body (from jit_insn facts) ---~n', []),
    emit_insns(S, Sorted),
    format(S, '~n', []),
    %% Store result
    format(S, '  store <8 x float> %result, ptr %out_ptr, align 4~n', []),
    format(S, '  ret void~n', []),
    format(S, '}~n~n', []),
    format(S, 'attributes #0 = { "target-features"="+avx" }~n~n', []).

%% ============================================================
%% Constant broadcast emission
%% ============================================================

emit_const_broadcast(S, Name, HexVal) :-
    atom_string(HexVal, HexStr),
    %% Integer constants (bias, masks)
    (   sub_string(HexStr, 0, _, _, "i32")
    ->  format(S, '  %c_~w = bitcast <8 x i32> <~w, ~w, ~w, ~w, ~w, ~w, ~w, ~w> to <8 x i32>~n',
               [Name, HexVal, HexVal, HexVal, HexVal, HexVal, HexVal, HexVal, HexVal])
    ;   %% Float constants — splat via insertelement + shufflevector
        format(S, '  %c_~w_s = insertelement <8 x float> undef, float ~w, i32 0~n', [Name, HexVal]),
        format(S, '  %c_~w = shufflevector <8 x float> %c_~w_s, <8 x float> undef, <8 x i32> zeroinitializer~n', [Name, Name])
    ).

%% ============================================================
%% Instruction emission — dispatch on operation type
%% ============================================================

emit_insns(_, []).
emit_insns(S, [Off-Op-Dst-Src | Rest]) :-
    format(S, '  ; [0x~|~`0t~3r] ~n', [Off]),
    emit_one_insn(S, Op, Dst, Src),
    emit_insns(S, Rest).

%% --- load from input ---
emit_one_insn(S, load, Dst, '[input]') :-
    format(S, '  %~w = load <8 x float>, ptr %in_ptr, align 4~n', [Dst]).

%% --- load constant (already broadcast as %c_Name) ---
emit_one_insn(S, load, Dst, ConstName) :-
    ConstName \= '[input]',
    format(S, '  %~w = fadd <8 x float> %c_~w, <float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0>~n', [Dst, ConstName]).
    %% Using fadd 0.0 as a copy — LLVM will optimize to a mov

%% --- mov (copy) ---
emit_one_insn(S, mov, Dst, Src) :-
    format(S, '  %~w = bitcast <8 x float> %~w to <8 x float>~n', [Dst, Src]).

%% --- Helper: resolve operand (constant or SSA variable) ---
resolve_operand(X, Resolved) :-
    (   jit_const(_, X, _)
    ->  format(atom(Resolved), '%c_~w', [X])
    ;   format(atom(Resolved), '%~w', [X])
    ).

%% --- fmul: A * B (either may be a constant) ---
emit_one_insn(S, fmul, Dst, A * B) :-
    resolve_operand(A, AS),
    resolve_operand(B, BS),
    format(S, '  %~w = fmul <8 x float> ~w, ~w~n', [Dst, AS, BS]).

%% --- fadd: A + B (either may be a constant) ---
emit_one_insn(S, fadd, Dst, A + B) :-
    resolve_operand(A, AS),
    resolve_operand(B, BS),
    format(S, '  %~w = fadd <8 x float> ~w, ~w~n', [Dst, AS, BS]).

%% --- fsub: A - B ---
emit_one_insn(S, fsub, Dst, A - B) :-
    resolve_operand(A, AS),
    resolve_operand(B, BS),
    format(S, '  %~w = fsub <8 x float> ~w, ~w~n', [Dst, AS, BS]).

%% --- fdiv: A / B (either may be a constant) ---
emit_one_insn(S, fdiv, Dst, A / B) :-
    (jit_const(_, A, _) -> format(atom(AS), '%c_~w', [A]) ; format(atom(AS), '%~w', [A])),
    (jit_const(_, B, _) -> format(atom(BS), '%c_~w', [B]) ; format(atom(BS), '%~w', [B])),
    format(S, '  %~w = fdiv <8 x float> ~w, ~w~n', [Dst, AS, BS]).

%% --- and_abs: abs via AND with 0x7FFFFFFF ---
%% Input is Src (same as Dst from the mov), output goes to Dst
emit_one_insn(S, and_abs, Dst, Src * abs_mask) :-
    format(S, '  %~w_pre_i = bitcast <8 x float> %~w to <8 x i32>~n', [Dst, Src]),
    format(S, '  %~w_anded = and <8 x i32> %~w_pre_i, %c_abs_mask~n', [Dst, Dst]),
    format(S, '  %~w = bitcast <8 x i32> %~w_anded to <8 x float>~n', [Dst, Dst]).

%% --- and_sign: extract sign bit (result is <8 x i32>, kept as int for later xor) ---
emit_one_insn(S, and_sign, Dst, Src * sign_mask) :-
    format(S, '  %~w_src_i = bitcast <8 x float> %~w to <8 x i32>~n', [Dst, Src]),
    format(S, '  %~w = and <8 x i32> %~w_src_i, %c_sign_mask~n', [Dst, Dst]).

%% --- xor_neg: negate via XOR with sign bit ---
emit_one_insn(S, xor_neg, Dst, Src * sign_mask) :-
    format(S, '  %~w_src_i = bitcast <8 x float> %~w to <8 x i32>~n', [Dst, Src]),
    format(S, '  %~w_xored = xor <8 x i32> %~w_src_i, %c_sign_mask~n', [Dst, Dst]),
    format(S, '  %~w = bitcast <8 x i32> %~w_xored to <8 x float>~n', [Dst, Dst]).

%% --- xor_sign: apply saved sign (SignReg is <8 x i32> from and_sign) ---
emit_one_insn(S, xor_sign, Dst, Src * SignReg) :-
    format(S, '  %~w_src_i = bitcast <8 x float> %~w to <8 x i32>~n', [Dst, Src]),
    format(S, '  %~w_xored = xor <8 x i32> %~w_src_i, %~w~n', [Dst, Dst, SignReg]),
    format(S, '  %~w = bitcast <8 x i32> %~w_xored to <8 x float>~n', [Dst, Dst]).

%% --- fcmplt: compare less-than ---
emit_one_insn(S, fcmplt, Dst, A < B) :-
    format(S, '  %~w = fcmp olt <8 x float> %~w, %~w~n', [Dst, A, B]).

%% --- fmin / fmax: clamp ---
emit_one_insn(S, fmin, Dst, A * B) :-
    resolve_operand(A, AS), resolve_operand(B, BS),
    format(S, '  %~w = call <8 x float> @llvm.minnum.v8f32(<8 x float> ~w, <8 x float> ~w)~n', [Dst, AS, BS]).

emit_one_insn(S, fmax, Dst, A * B) :-
    resolve_operand(A, AS), resolve_operand(B, BS),
    format(S, '  %~w = call <8 x float> @llvm.maxnum.v8f32(<8 x float> ~w, <8 x float> ~w)~n', [Dst, AS, BS]).

%% --- floor ---
emit_one_insn(S, floor, Dst, Src) :-
    format(S, '  %~w = call <8 x float> @llvm.floor.v8f32(<8 x float> %~w)~n', [Dst, Src]).

%% --- cvt: float to int ---
emit_one_insn(S, cvt, Dst, Src) :-
    format(S, '  %~w = fptosi <8 x float> %~w to <8 x i32>~n', [Dst, Src]).

%% --- iadd: integer add ---
emit_one_insn(S, iadd, Dst, A + ConstName) :-
    format(S, '  %~w = add <8 x i32> %~w, %c_~w~n', [Dst, A, ConstName]).

%% --- ishl: integer shift left ---
emit_one_insn(S, ishl, Dst, Src * ShiftAmt) :-
    format(S, '  %~w = shl <8 x i32> %~w, <i32 ~w, i32 ~w, i32 ~w, i32 ~w, i32 ~w, i32 ~w, i32 ~w, i32 ~w>~n',
           [Dst, Src, ShiftAmt, ShiftAmt, ShiftAmt, ShiftAmt, ShiftAmt, ShiftAmt, ShiftAmt, ShiftAmt]).

%% --- extract/insert for AVX1 128-bit halves ---
emit_one_insn(S, extract, Dst, Src) :-
    format(S, '  ; extract/insert handled by LLVM for <8 x i32>~n', []).

emit_one_insn(S, insert, Dst, _ + _) :-
    format(S, '  ; extract/insert handled by LLVM for <8 x i32>~n', []).

%% --- bitcast_i2f: reinterpret i32 vector as float ---
emit_one_insn(S, bitcast_i2f, Dst, Src) :-
    format(S, '  %~w = bitcast <8 x i32> %~w to <8 x float>~n', [Dst, Src]).

%% ============================================================
%% Loop wrapper
%% ============================================================

emit_loop_wrapper(S, _KernelName) :-
    format(S, 'define void @bpd_gelu_ondnn_cpu(ptr %input, ptr %output, i32 %n) {~n', []),
    format(S, 'entry:~n', []),
    format(S, '  br label %check~n~n', []),
    format(S, 'check:~n', []),
    format(S, '  %i = phi i32 [ 0, %entry ], [ %i8, %vec_body ]~n', []),
    format(S, '  %rem = sub i32 %n, %i~n', []),
    format(S, '  %can = icmp sge i32 %rem, 8~n', []),
    format(S, '  br i1 %can, label %vec_body, label %scalar~n~n', []),
    format(S, 'vec_body:~n', []),
    format(S, '  %ip = getelementptr float, ptr %input, i32 %i~n', []),
    format(S, '  %op = getelementptr float, ptr %output, i32 %i~n', []),
    format(S, '  call void @kernel_vec8(ptr %ip, ptr %op)~n', []),
    format(S, '  %i8 = add i32 %i, 8~n', []),
    format(S, '  br label %check~n~n', []),
    format(S, 'scalar:~n', []),
    format(S, '  %j = phi i32 [ %i, %check ], [ %jn, %scalar_body ]~n', []),
    format(S, '  %sd = icmp sge i32 %j, %n~n', []),
    format(S, '  br i1 %sd, label %done, label %scalar_body~n~n', []),
    format(S, 'scalar_body:~n', []),
    format(S, '  %sp = getelementptr float, ptr %input, i32 %j~n', []),
    format(S, '  %sx = load float, ptr %sp~n', []),
    format(S, '  %sv = fmul float %sx, 0x3FE6A09E60000000~n', []),
    format(S, '  %se = call float @erff(float %sv)~n', []),
    format(S, '  %sa = fadd float 1.0, %se~n', []),
    format(S, '  %st = fmul float %sx, %sa~n', []),
    format(S, '  %sr = fmul float %st, 0.5~n', []),
    format(S, '  %so = getelementptr float, ptr %output, i32 %j~n', []),
    format(S, '  store float %sr, ptr %so~n', []),
    format(S, '  %jn = add i32 %j, 1~n', []),
    format(S, '  br label %scalar~n~n', []),
    format(S, 'done:~n', []),
    format(S, '  ret void~n', []),
    format(S, '}~n', []).
