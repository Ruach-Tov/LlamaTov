%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════════
%% llvm_from_facts.pl — LLVM IR backend (backend 4), generated from the SAME
%% neutral expression AST (expr_ir.pl op_expr/2 -> lower_llvm/3) that drives
%% cuda_c, mlir, and torch. Emits a complete CPU kernel (.ll) that clang compiles.
%%
%% Wraps lower_llvm's per-element SSA body in a function:
%%   void @<op>(float* %src, float* %dst, i64 %n)
%%   loop i in [0,n): %v = load src[i]; <body SSA -> %res>; store %res -> dst[i]
%%
%% Verified by the differential referee like the other backends. CPU target (the
%% honest LLVM analog of MLIR-CPU); a NVPTX variant is a follow-up.
%%
%% Author: Iyun, 2026-06-07 (backend 4 wired from the expression IR)
%% ═══════════════════════════════════════════════════════════════════════════

:- module(llvm_from_facts, [emit_llvm_from_fact/2, llvm_supported_op/1,
                            emit_llvm_nvptx_from_fact/2, emit_llvm_nvptx_matmul/2]).

%% import the shared IR (op_expr + lower_llvm), resolved relative to this file
:- ( prolog_load_context(directory, ED),
     atomic_list_concat([ED, '/expr_ir.pl'], EP), exists_file(EP)
   -> use_module(EP, [op_expr/2, lower_llvm/3])
   ;  exists_file('kernelgen/emitters/expr_ir.pl')
   -> use_module('kernelgen/emitters/expr_ir.pl', [op_expr/2, lower_llvm/3])
   ;  exists_file('kernelgen/emitters/expr_ir.pl')
   -> use_module('kernelgen/emitters/expr_ir.pl', [op_expr/2, lower_llvm/3])
   ;  true ).

llvm_supported_op(Op) :- op_expr(Op, E), E \= reduce(_,_,_,_,_).  % elementwise only (for now)

%% emit_llvm_from_fact(+Op, +OutFile): write a .ll module with the op kernel.
emit_llvm_from_fact(Op, OutFile) :-
    op_expr(Op, Expr),
    lower_llvm(Expr, BodySSA, ResSSA),
    (atom_concat('bpd_', Name, Op) -> true ; Name = Op),
    %% collect the math intrinsics used, so we can declare them
    findall(Decl, (member(St, BodySSA),
                   sub_atom(St, _, _, _, '@llvm.'),
                   intrinsic_decl(St, Decl)),
            Decls0),
    sort(Decls0, Decls),
    open(OutFile, write, S),
    format(S, "; GENERATED from op_expr(~w) via lower_llvm — backend 4 (shared AST).~n", [Op]),
    forall(member(D, Decls), format(S, "~w~n", [D])),
    format(S, "define void @~w(float* noalias %src, float* noalias %dst, i64 %n) {~n", [Name]),
    format(S, "entry:~n  br label %loop~n", []),
    format(S, "loop:~n", []),
    format(S, "  %i = phi i64 [0, %entry], [%inext, %body]~n", []),
    format(S, "  %cond = icmp slt i64 %i, %n~n", []),
    format(S, "  br i1 %cond, label %body, label %done~n", []),
    format(S, "body:~n", []),
    format(S, "  %sp = getelementptr float, float* %src, i64 %i~n", []),
    format(S, "  %v = load float, float* %sp, align 4~n", []),
    forall(member(St, BodySSA), format(S, "  ~w~n", [St])),
    format(S, "  %dp = getelementptr float, float* %dst, i64 %i~n", []),
    format(S, "  store float ~w, float* %dp, align 4~n", [ResSSA]),
    format(S, "  %inext = add i64 %i, 1~n", []),
    format(S, "  br label %loop~n", []),
    format(S, "done:~n  ret void~n}~n", []),
    close(S),
    format("Generated LLVM-IR kernel ~w -> ~w~n", [Name, OutFile]).

%% derive the `declare` line for a math intrinsic referenced in an SSA stmt.
%% A call line looks like:  %lN = call float @llvm.exp.f32(float %lM)
%% Extract the @llvm.<...>.f32 token (between "call float " and "(").
intrinsic_decl(St, Decl) :-
    sub_atom(St, Pre, _, Post, '@llvm.'),
    Start is Pre,
    sub_atom(St, Start, _, 0, FromAt),           % "@llvm.exp.f32(float %lM)"
    sub_atom(FromAt, Before, _, _, '('), !,
    sub_atom(FromAt, 0, Before, _, FnName),       % "@llvm.exp.f32"
    format(atom(Decl), "declare float ~w(float)", [FnName]),
    Post >= 0.

%% ── NVPTX variant: the LLVM backend targeting the GPU (P4) ──────────────────
%% Emits an NVPTX64 kernel from the SAME lower_llvm body. Thread index via the
%% nvvm sreg intrinsics; marked a kernel via !nvvm.annotations. llc -march=nvptx64
%% -mcpu=sm_61 -> PTX, then the cubin launcher runs it on the P4. This makes
%% backend 4 reach the same hardware as cuda_c + mlir_gpu.
emit_llvm_nvptx_from_fact(Op, OutFile) :-
    op_expr(Op, Expr),
    lower_llvm(Expr, BodySSA, ResSSA),
    (atom_concat('bpd_', Name, Op) -> true ; Name = Op),
    findall(D, (member(St, BodySSA), sub_atom(St,_,_,_,'@llvm.'), intrinsic_decl(St,D)), D0),
    sort(D0, Decls),
    open(OutFile, write, S),
    format(S, "; GENERATED NVPTX from op_expr(~w) via lower_llvm — backend 4 on the P4.~n", [Op]),
    format(S, "target triple = \"nvptx64-nvidia-cuda\"~n", []),
    format(S, "target datalayout = \"e-i64:64-i128:128-v16:16-v32:32-n16:32:64\"~n", []),
    forall(member(D, Decls), format(S, "~w~n", [D])),
    format(S, "declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()~n", []),
    format(S, "declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()~n", []),
    format(S, "declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()~n", []),
    %% kernel: src/dst are addrspace(1) global pointers
    format(S, "define void @~w(float addrspace(1)* %src, float addrspace(1)* %dst, i64 %n) {~n", [Name]),
    format(S, "entry:~n", []),
    format(S, "  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()~n", []),
    format(S, "  %ctaid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()~n", []),
    format(S, "  %ntid = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()~n", []),
    format(S, "  %mul = mul i32 %ctaid, %ntid~n", []),
    format(S, "  %gid32 = add i32 %mul, %tid~n", []),
    format(S, "  %i = sext i32 %gid32 to i64~n", []),
    format(S, "  %inb = icmp slt i64 %i, %n~n", []),
    format(S, "  br i1 %inb, label %body, label %done~n", []),
    format(S, "body:~n", []),
    format(S, "  %sp = getelementptr float, float addrspace(1)* %src, i64 %i~n", []),
    format(S, "  %v = load float, float addrspace(1)* %sp, align 4~n", []),
    forall(member(St, BodySSA), format(S, "  ~w~n", [St])),
    format(S, "  %dp = getelementptr float, float addrspace(1)* %dst, i64 %i~n", []),
    format(S, "  store float ~w, float addrspace(1)* %dp, align 4~n", [ResSSA]),
    format(S, "  br label %done~n", []),
    format(S, "done:~n  ret void~n}~n", []),
    %% mark as a kernel
    format(S, "!nvvm.annotations = !{!0}~n", []),
    format(S, "!0 = !{void (float addrspace(1)*, float addrspace(1)*, i64)* @~w, !\"kernel\", i32 1}~n", [Name]),
    close(S),
    format("Generated NVPTX LLVM kernel ~w -> ~w~n", [Name, OutFile]).

%% ── NVPTX MATMUL kernel (reduction class on the P4) ─────────────────────────
%% Wraps the matmul reduce in a 2-D-grid kernel: each thread computes C[row,col]
%% = sum_k A[row,k]*B[k,col]. addrspace(1) globals; row from (ctaid.y*ntid.y+tid.y),
%% col from x. fma_mode: strict (fmul+fadd) | contract (@llvm.fma.f32).
%% llc -march=nvptx64 -mcpu=sm_61 -> PTX; run via a 2-D-grid launcher.
emit_llvm_nvptx_matmul(FmaMode, OutFile) :-
    matmul_mac_llvm(FmaMode, Mac),
    open(OutFile, write, S),
    format(S, "; GENERATED NVPTX matmul (fma=~w) from op_expr(bpd_matmul) — backend 4 on P4.~n", [FmaMode]),
    format(S, "target triple = \"nvptx64-nvidia-cuda\"~n", []),
    format(S, "target datalayout = \"e-i64:64-i128:128-v16:16-v32:32-n16:32:64\"~n", []),
    ( FmaMode == contract -> format(S, "declare float @llvm.fma.f32(float, float, float)~n", []) ; true ),
    format(S, "declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()~n", []),
    format(S, "declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()~n", []),
    format(S, "declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()~n", []),
    format(S, "declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()~n", []),
    format(S, "declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()~n", []),
    format(S, "declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y()~n", []),
    format(S, "define void @gemm(float addrspace(1)* %A, float addrspace(1)* %B, float addrspace(1)* %C, i64 %n) {~n", []),
    format(S, "entry:~n", []),
    format(S, "  %tx = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()~n  %ty = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()~n", []),
    format(S, "  %bx = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()~n  %by = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()~n", []),
    format(S, "  %nx = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()~n  %ny = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()~n", []),
    format(S, "  %cx = mul i32 %bx, %nx~n  %col32 = add i32 %cx, %tx~n  %col = sext i32 %col32 to i64~n", []),
    format(S, "  %cy = mul i32 %by, %ny~n  %row32 = add i32 %cy, %ty~n  %row = sext i32 %row32 to i64~n", []),
    format(S, "  %rok = icmp slt i64 %row, %n~n  %cok = icmp slt i64 %col, %n~n  %ok = and i1 %rok, %cok~n", []),
    format(S, "  br i1 %ok, label %body, label %done~n", []),
    format(S, "body:~n  br label %rloop~n", []),
    format(S, "rloop:~n  %k = phi i64 [0, %body], [%knext, %rbody]~n  %acc = phi float [0.0, %body], [%nacc, %rbody]~n", []),
    format(S, "  %rc = icmp slt i64 %k, %n~n  br i1 %rc, label %rbody, label %rdone~n", []),
    format(S, "rbody:~n", []),
    format(S, "  %ai = mul i64 %row, %n~n  %aidx = add i64 %ai, %k~n  %ap = getelementptr float, float addrspace(1)* %A, i64 %aidx~n  %a = load float, float addrspace(1)* %ap, align 4~n", []),
    format(S, "  %bi = mul i64 %k, %n~n  %bidx = add i64 %bi, %col~n  %bp = getelementptr float, float addrspace(1)* %B, i64 %bidx~n  %b = load float, float addrspace(1)* %bp, align 4~n", []),
    format(S, "~w  %knext = add i64 %k, 1~n  br label %rloop~n", [Mac]),
    format(S, "rdone:~n  %ci = mul i64 %row, %n~n  %cidx = add i64 %ci, %col~n  %cp = getelementptr float, float addrspace(1)* %C, i64 %cidx~n  store float %acc, float addrspace(1)* %cp, align 4~n  br label %done~n", []),
    format(S, "done:~n  ret void~n}~n", []),
    format(S, "!nvvm.annotations = !{!0}~n", []),
    format(S, "!0 = !{void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i64)* @gemm, !\"kernel\", i32 1}~n", []),
    close(S),
    format("Generated NVPTX matmul (fma=~w) -> ~w~n", [FmaMode, OutFile]).
matmul_mac_llvm(contract, M) :- format(atom(M), "  %nacc = call float @llvm.fma.f32(float %a, float %b, float %acc)~n", []).
matmul_mac_llvm(_, M) :- format(atom(M), "  %p = fmul float %a, %b~n  %nacc = fadd float %acc, %p~n", []).
