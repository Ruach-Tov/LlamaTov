%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════════
%% mlir_gpu_from_facts.pl - generate MLIR gpu-dialect kernels from canonical
%% facts, for the MLIR→NVVM→PTX→P4 backend (verified by the differential referee).
%%
%% Emits a gpu.module containing a gpu.func kernel that applies the fact's
%% per-element op over a memref<?xf32>. Lowered by:
%%   mlir-opt --convert-scf-to-cf --nvvm-attach-target="chip=sm_61 O=3"
%%            --convert-gpu-to-nvvm --reconcile-unrealized-casts
%%            --gpu-module-to-binary="format=isa"
%% -> PTX -> ptxas -> driver-API launch on the P4.
%%
%% Same per-element semantics as the other backends (relu/tanh/elu/gelu);
%% the gpu kernel body is the only varying part. fast nan_mode for now (oge);
%% nan_mode(uno guard) + fma_mode wire in next, parallel to the other emitters.
%%
%% Author: Iyun, 2026-06-07 (MLIR-GPU emitter; 4th GPU-reaching backend)
%% ═══════════════════════════════════════════════════════════════════════════

:- module(mlir_gpu_from_facts, [emit_mlir_gpu/2, mlir_gpu_supported_op/1, emit_mlir_gpu_matmul/2,
                                emit_mlir_gpu_reduce/2, emit_mlir_gpu_pool/2, emit_mlir_gpu_conv/2]).
:- use_module(library(lists)).

%% Load the canonical facts (robust_op_match/5), resolved relative to this
%% emitter's location (bpd/kernelgen/emitters/ -> ../../lib/robust_op_match.pl),
%% with absolute fallbacks for the enclave/laptop checkouts.
:- ( prolog_load_context(directory, Dir),
     atomic_list_concat([Dir, '/../../lib/robust_op_match.pl'], RelPath),
     exists_file(RelPath)
   -> consult(RelPath)
   ;  exists_file('lib/robust_op_match.pl')
   -> consult('lib/robust_op_match.pl')
   ;  exists_file('lib/robust_op_match.pl')
   -> consult('lib/robust_op_match.pl')
   ;  exists_file('robust_op_match.pl')
   -> consult('robust_op_match.pl')
   ;  true ).

%% per-op MLIR body — NO LONGER a hand-written table. The body is GENERATED from
%% the op's neutral expression IR (expr_ir.pl op_expr/2) via lower_mlir/3, the
%% SAME term cuda_c lowers from. One expr per op -> both backends. Adding an op =
%% one expr term (in expr_ir.pl, ideally the fact itself); improving lower_mlir
%% improves EVERY op''s MLIR. Resolve expr_ir relative to this file.
:- ( prolog_load_context(directory, ED),
     atomic_list_concat([ED, '/expr_ir.pl'], EP), exists_file(EP)
   -> consult(EP)
   ;  exists_file('kernelgen/emitters/expr_ir.pl')
   -> consult('kernelgen/emitters/expr_ir.pl')
   ;  exists_file('kernelgen/emitters/expr_ir.pl')
   -> consult('kernelgen/emitters/expr_ir.pl')
   ;  true ).

%% gpu_body(+Op, -Stmts, -ResSSA): the generated MLIR body for Op, from its expr.
gpu_body(Op, Stmts, Res) :-
    op_expr(Op, Expr),
    lower_mlir(Expr, Stmts, Res).

mlir_gpu_supported_op(Op) :- op_expr(Op, _).

%% emit the gpu kernel for an op - PLAIN-POINTER signature (src, dst, n).
%%
%% Why plain pointers, not memref<?xf32>: a memref arg lowers to a 5-field
%% descriptor (allocated_ptr, aligned_ptr, offset, size, stride), so two
%% memrefs => 10 positional PTX params (relu_param_0..9) of which the kernel
%% only uses 3 (src ptr, n, dst ptr). The plain-pointer signature collapses
%% that to exactly 3 params - matching a hand-written CUDA kernel.
%%
%% Note on names: the NVPTX backend ALWAYS numbers params positionally as
%% <func>_param_N (it is the PTX ABI convention - nvcc does the same and does
%% not preserve source argument names). So the PTX shows relu_param_0/1/2; the
%% header comment documents which is which. The MLIR SSA names (%src/%dst/%n)
%% are still meaningful in the .mlir and the generated LLVM IR.
emit_mlir_gpu(Op, OutFile) :-
    gpu_body(Op, Stmts, Res),
    (atom_concat('bpd_', Name, Op) -> true ; Name = Op),
    open(OutFile, write, S),
    format(S, "// GENERATED from robust_op_match(~w) - MLIR gpu/llvm dialect (->NVVM->PTX->P4).~n", [Op]),
    format(S, "// Kernel params (PTX ABI numbers them ~w_param_0..2 positionally):~n", [Name]),
    format(S, "//   param_0 = src  (const float*)  - input pointer~n", []),
    format(S, "//   param_1 = dst  (float*)        - output pointer~n", []),
    format(S, "//   param_2 = n    (i64)           - element count~n", []),
    format(S, "module attributes {gpu.container_module} {~n", []),
    format(S, "  gpu.module @kernels {~n", []),
    format(S, "    llvm.func @~w(%src: !llvm.ptr, %dst: !llvm.ptr, %n: i64) attributes {gpu.kernel, nvvm.kernel} {~n", [Name]),
    format(S, "      %bid = nvvm.read.ptx.sreg.ctaid.x : i32~n", []),
    format(S, "      %bdim = nvvm.read.ptx.sreg.ntid.x : i32~n", []),
    format(S, "      %tid = nvvm.read.ptx.sreg.tid.x : i32~n", []),
    format(S, "      %t1 = llvm.mul %bid, %bdim : i32~n", []),
    format(S, "      %i32 = llvm.add %t1, %tid : i32~n", []),
    format(S, "      %i = llvm.sext %i32 : i32 to i64~n", []),
    format(S, "      %inb = llvm.icmp \"slt\" %i, %n : i64~n", []),
    format(S, "      llvm.cond_br %inb, ^body, ^done~n", []),
    format(S, "    ^body:~n", []),
    format(S, "      %sp = llvm.getelementptr %src[%i] : (!llvm.ptr, i64) -> !llvm.ptr, f32~n", []),
    format(S, "      %v = llvm.load %sp : !llvm.ptr -> f32~n", []),
    forall(member(St, Stmts), format(S, "      ~w~n", [St])),
    format(S, "      %dp = llvm.getelementptr %dst[%i] : (!llvm.ptr, i64) -> !llvm.ptr, f32~n", []),
    format(S, "      llvm.store ~w, %dp : f32, !llvm.ptr~n", [Res]),
    format(S, "      llvm.br ^done~n", []),
    format(S, "    ^done:~n      llvm.return~n    }~n  }~n}~n", []),
    close(S),
    format("Generated MLIR-GPU ~w (3 named params) -> ~w~n", [Name, OutFile]).

%% ── MLIR-GPU MATMUL kernel (reduction class on the P4) ──────────────────────
%% Same llvm-dialect path as the elementwise wrapper (proven through the pipeline),
%% but a 2-D grid: each thread computes C[row,col]=sum_k A[row,k]*B[k,col]. The
%% k-loop is an llvm-dialect block-arg loop (phi via ^kloop block args). fma_mode:
%% strict (fmul+fadd) | contract (llvm.intr.fma). Params A,B,C : !llvm.ptr, n : i64.
emit_mlir_gpu_matmul(FmaMode, OutFile) :-
    mm_mac_mlir(FmaMode, Mac),
    open(OutFile, write, S),
    format(S, "// GENERATED MLIR-GPU matmul (fma=~w) from op_expr(bpd_matmul) -> NVVM -> PTX -> P4.~n", [FmaMode]),
    format(S, "// params: param_0=A param_1=B param_2=C (!llvm.ptr), param_3=n (i64)~n", []),
    format(S, "module attributes {gpu.container_module} {~n", []),
    format(S, "  gpu.module @kernels {~n", []),
    format(S, "    llvm.func @gemm(%A: !llvm.ptr, %B: !llvm.ptr, %C: !llvm.ptr, %n: i64) attributes {gpu.kernel, nvvm.kernel} {~n", []),
    %% col = ctaid.x*ntid.x + tid.x ; row = ctaid.y*ntid.y + tid.y
    format(S, "      %bx = nvvm.read.ptx.sreg.ctaid.x : i32~n      %nx = nvvm.read.ptx.sreg.ntid.x : i32~n      %tx = nvvm.read.ptx.sreg.tid.x : i32~n", []),
    format(S, "      %by = nvvm.read.ptx.sreg.ctaid.y : i32~n      %ny = nvvm.read.ptx.sreg.ntid.y : i32~n      %ty = nvvm.read.ptx.sreg.tid.y : i32~n", []),
    format(S, "      %cxm = llvm.mul %bx, %nx : i32~n      %col32 = llvm.add %cxm, %tx : i32~n      %col = llvm.sext %col32 : i32 to i64~n", []),
    format(S, "      %cym = llvm.mul %by, %ny : i32~n      %row32 = llvm.add %cym, %ty : i32~n      %row = llvm.sext %row32 : i32 to i64~n", []),
    format(S, "      %rok = llvm.icmp \"slt\" %row, %n : i64~n      %cok = llvm.icmp \"slt\" %col, %n : i64~n      %ok = llvm.and %rok, %cok : i1~n", []),
    format(S, "      llvm.cond_br %ok, ^body, ^done~n", []),
    format(S, "    ^body:~n", []),
    format(S, "      %zero = llvm.mlir.constant(0 : i64) : i64~n      %one = llvm.mlir.constant(1 : i64) : i64~n      %fz = llvm.mlir.constant(0.0 : f32) : f32~n", []),
    format(S, "      llvm.br ^kloop(%zero, %fz : i64, f32)~n", []),
    format(S, "    ^kloop(%k: i64, %acc: f32):~n", []),
    format(S, "      %kc = llvm.icmp \"slt\" %k, %n : i64~n      llvm.cond_br %kc, ^kbody, ^kdone~n", []),
    format(S, "    ^kbody:~n", []),
    format(S, "      %ar = llvm.mul %row, %n : i64~n      %ai = llvm.add %ar, %k : i64~n      %ap = llvm.getelementptr %A[%ai] : (!llvm.ptr, i64) -> !llvm.ptr, f32~n      %a = llvm.load %ap : !llvm.ptr -> f32~n", []),
    format(S, "      %br = llvm.mul %k, %n : i64~n      %bi = llvm.add %br, %col : i64~n      %bp = llvm.getelementptr %B[%bi] : (!llvm.ptr, i64) -> !llvm.ptr, f32~n      %b = llvm.load %bp : !llvm.ptr -> f32~n", []),
    format(S, "~w", [Mac]),
    format(S, "      %knext = llvm.add %k, %one : i64~n      llvm.br ^kloop(%knext, %nacc : i64, f32)~n", []),
    format(S, "    ^kdone:~n", []),
    format(S, "      %cr = llvm.mul %row, %n : i64~n      %ci = llvm.add %cr, %col : i64~n      %cp = llvm.getelementptr %C[%ci] : (!llvm.ptr, i64) -> !llvm.ptr, f32~n      llvm.store %acc, %cp : f32, !llvm.ptr~n", []),
    format(S, "      llvm.br ^done~n", []),
    format(S, "    ^done:~n      llvm.return~n    }~n  }~n}~n", []),
    close(S),
    format("Generated MLIR-GPU matmul (fma=~w) -> ~w~n", [FmaMode, OutFile]).
mm_mac_mlir(contract, M) :- format(atom(M), "      %nacc = llvm.intr.fma(%a, %b, %acc) : (f32, f32, f32) -> f32~n", []).
mm_mac_mlir(_, M) :- format(atom(M), "      %p = llvm.fmul %a, %b : f32~n      %nacc = llvm.fadd %acc, %p : f32~n", []).

%% ── MLIR-GPU AXIS-REDUCE kernel (reduction class, 2nd op class on the P4) ───────
%% emit_mlir_gpu_reduce(+Op, +OutFile): naive one-thread-per-row reduce over a
%% RxC row-major matrix. Each thread r<R folds columns 0..C-1 into an accumulator
%% via an llvm-dialect block-arg loop (phi via ^rloop args), writes out[r].
%% Correctness-first (matches the OLD pre-tiling cuda-c reduce); the warp-shuffle
%% tiled version (nvvm.shfl.sync) is a follow-up. Same op_expr-derived init/combine/
%% final as the cuda-c reduce (reduce_kernel_parts). ABI: k_reduce(x,out,R,C),
%% 1-D grid, one thread per row.
emit_mlir_gpu_reduce(Op, OutFile) :-
    op_expr(Op, axis_reduce(Kind, _Axis, _Body)),
    mlir_reduce_parts(Kind, InitConst, Comb, FinalDiv),
    (atom_concat('bpd_', Name, Op) -> true ; Name = Op),
    open(OutFile, write, S),
    format(S, "// GENERATED MLIR-GPU axis-reduce (~w) from op_expr(~w) -> NVVM -> PTX -> P4.~n", [Kind, Op]),
    format(S, "// params: param_0=x param_1=out (!llvm.ptr), param_2=R param_3=C (i64). 1 thread/row.~n", []),
    format(S, "module attributes {gpu.container_module} {~n", []),
    format(S, "  gpu.module @kernels {~n", []),
    format(S, "    llvm.func @k_reduce(%x: !llvm.ptr, %out: !llvm.ptr, %R: i64, %C: i64) attributes {gpu.kernel, nvvm.kernel} {~n", []),
    format(S, "      %bid = nvvm.read.ptx.sreg.ctaid.x : i32~n      %bdim = nvvm.read.ptx.sreg.ntid.x : i32~n      %tid = nvvm.read.ptx.sreg.tid.x : i32~n", []),
    format(S, "      %t1 = llvm.mul %bid, %bdim : i32~n      %r32 = llvm.add %t1, %tid : i32~n      %r = llvm.sext %r32 : i32 to i64~n", []),
    format(S, "      %rok = llvm.icmp \"slt\" %r, %R : i64~n      llvm.cond_br %rok, ^body, ^done~n", []),
    format(S, "    ^body:~n", []),
    format(S, "      %zero = llvm.mlir.constant(0 : i64) : i64~n      %one = llvm.mlir.constant(1 : i64) : i64~n", []),
    format(S, "      %init = llvm.mlir.constant(~w : f32) : f32~n", [InitConst]),
    format(S, "      %base = llvm.mul %r, %C : i64~n", []),   % row start offset
    format(S, "      llvm.br ^rloop(%zero, %init : i64, f32)~n", []),
    format(S, "    ^rloop(%c: i64, %acc: f32):~n", []),
    format(S, "      %cc = llvm.icmp \"slt\" %c, %C : i64~n      llvm.cond_br %cc, ^cbody, ^cdone~n", []),
    format(S, "    ^cbody:~n", []),
    format(S, "      %off = llvm.add %base, %c : i64~n", []),
    format(S, "      %xp = llvm.getelementptr %x[%off] : (!llvm.ptr, i64) -> !llvm.ptr, f32~n      %v = llvm.load %xp : !llvm.ptr -> f32~n", []),
    format(S, Comb, []),   % produces %nacc (Comb has ~n directives) from %acc and %v
    format(S, "      %cnext = llvm.add %c, %one : i64~n      llvm.br ^rloop(%cnext, %nacc : i64, f32)~n", []),
    format(S, "    ^cdone:~n", []),
    mlir_reduce_final(FinalDiv, S),   % %res = acc or acc/C
    format(S, "      %op = llvm.getelementptr %out[%r] : (!llvm.ptr, i64) -> !llvm.ptr, f32~n      llvm.store %res, %op : f32, !llvm.ptr~n", []),
    format(S, "      llvm.br ^done~n    ^done:~n      llvm.return~n    }~n  }~n}~n", []),
    close(S),
    format("Generated MLIR-GPU axis-reduce (~w) ~w -> ~w~n", [Kind, Name, OutFile]).

%% per-kind: init constant, the combine MLIR (acc,v -> %nacc), and final divide flag
mlir_reduce_parts(sum,  "0.0", "      %nacc = llvm.fadd %acc, %v : f32~n", nodiv).
mlir_reduce_parts(mean, "0.0", "      %nacc = llvm.fadd %acc, %v : f32~n", div).
mlir_reduce_parts(max, "-3.40282347E+38", "      %gt = arith.cmpf ogt, %v, %acc : f32~n      %nacc = arith.select %gt, %v, %acc : f32~n", nodiv).
mlir_reduce_parts(min,  "3.40282347E+38", "      %lt = arith.cmpf olt, %v, %acc : f32~n      %nacc = arith.select %lt, %v, %acc : f32~n", nodiv).

%% nodiv (sum/max/min): result is the accumulator. div (mean): acc / C.
%% (MLIR SSA is single-assignment; alias %acc to %res via a no-op add of 0.0.)
mlir_reduce_final(nodiv, S) :- !,
    format(S, "      %fz = llvm.mlir.constant(0.0 : f32) : f32~n      %res = llvm.fadd %acc, %fz : f32~n", []).
mlir_reduce_final(div, S) :-
    format(S, "      %Cf = llvm.sitofp %C : i64 to f32~n      %res = llvm.fdiv %acc, %Cf : f32~n", []).

%% ── MLIR-GPU 2D POOL kernel (pooling class, 3rd op class) ───────────────────────
%% emit_mlir_gpu_pool(+Op, +OutFile): one-thread-per-output 2D pool (NCHW). Each
%% thread idx<NC*Hout*Wout decodes (nc,oh,ow), folds the KxK window (with padding
%% bounds-checks) into an accumulator (max or sum), writes out[idx] (avg divides by
%% the count of valid window elements). llvm-dialect nested block-arg loops over
%% kh,kw. ABI: k_pool(x,out,NC,H,W,Hout,Wout), 1-D grid one thread/output.
emit_mlir_gpu_pool(Op, OutFile) :-
    op_expr(Op, pool(Kind, 2, K, St, Pad, Dil, _Body)),
    mlir_pool_parts(Kind, InitConst, Comb, FinalAvg),
    (atom_concat('bpd_', Name, Op) -> true ; Name = Op),
    open(OutFile, write, S),
    format(S, "// GENERATED MLIR-GPU 2D pool (~w K=~w S=~w P=~w D=~w) from op_expr(~w) -> NVVM -> PTX -> P4.~n", [Kind, K, St, Pad, Dil, Op]),
    format(S, "// params: x,out (!llvm.ptr); NC,H,W,Hout,Wout (i64). 1 thread/output.~n", []),
    format(S, "module attributes {gpu.container_module} {~n  gpu.module @kernels {~n", []),
    format(S, "    llvm.func @k_pool(%x: !llvm.ptr, %out: !llvm.ptr, %NC: i64, %H: i64, %W: i64, %Hout: i64, %Wout: i64) attributes {gpu.kernel, nvvm.kernel} {~n", []),
    format(S, "      %bid = nvvm.read.ptx.sreg.ctaid.x : i32~n      %bdim = nvvm.read.ptx.sreg.ntid.x : i32~n      %tid = nvvm.read.ptx.sreg.tid.x : i32~n", []),
    format(S, "      %t1 = llvm.mul %bid, %bdim : i32~n      %g32 = llvm.add %t1, %tid : i32~n      %idx = llvm.sext %g32 : i32 to i64~n", []),
    %% total = NC*Hout*Wout ; bounds
    format(S, "      %nho = llvm.mul %NC, %Hout : i64~n      %total = llvm.mul %nho, %Wout : i64~n", []),
    format(S, "      %ok = llvm.icmp \"slt\" %idx, %total : i64~n      llvm.cond_br %ok, ^body, ^done~n    ^body:~n", []),
    %% decode idx -> ow, oh, nc   (row-major NC x Hout x Wout)
    format(S, "      %ow = llvm.srem %idx, %Wout : i64~n      %t2 = llvm.sdiv %idx, %Wout : i64~n", []),
    format(S, "      %oh = llvm.srem %t2, %Hout : i64~n      %nc = llvm.sdiv %t2, %Hout : i64~n", []),
    %% window start (oh*St - Pad), (ow*St - Pad)
    format(S, "      %St = llvm.mlir.constant(~w : i64) : i64~n      %Pad = llvm.mlir.constant(~w : i64) : i64~n      %Dil = llvm.mlir.constant(~w : i64) : i64~n      %K = llvm.mlir.constant(~w : i64) : i64~n", [St, Pad, Dil, K]),
    format(S, "      %hs0 = llvm.mul %oh, %St : i64~n      %hs = llvm.sub %hs0, %Pad : i64~n", []),
    format(S, "      %ws0 = llvm.mul %ow, %St : i64~n      %ws = llvm.sub %ws0, %Pad : i64~n", []),
    format(S, "      %z = llvm.mlir.constant(0 : i64) : i64~n      %one = llvm.mlir.constant(1 : i64) : i64~n", []),
    format(S, "      %init = llvm.mlir.constant(~w : f32) : f32~n      %fz = llvm.mlir.constant(0.0 : f32) : f32~n", [InitConst]),
    format(S, "      %nc_h = llvm.mul %nc, %H : i64~n", []),  %% nc*H (for input index base)
    %% kh loop: (acc, cnt) carried
    format(S, "      llvm.br ^khloop(%z, %init, %z : i64, f32, i64)~n", []),
    format(S, "    ^khloop(%kh: i64, %acch: f32, %cnth: i64):~n", []),
    format(S, "      %khok = llvm.icmp \"slt\" %kh, %K : i64~n      llvm.cond_br %khok, ^khb, ^khd(%acch, %cnth : f32, i64)~n    ^khb:~n", []),
    format(S, "      %ih0 = llvm.mul %kh, %Dil : i64~n      %ih = llvm.add %hs, %ih0 : i64~n", []),
    %% inner kw loop
    format(S, "      llvm.br ^kwloop(%z, %acch, %cnth : i64, f32, i64)~n", []),
    format(S, "    ^kwloop(%kw: i64, %acc: f32, %cnt: i64):~n", []),
    format(S, "      %kwok = llvm.icmp \"slt\" %kw, %K : i64~n      llvm.cond_br %kwok, ^kwb, ^kwd(%acc, %cnt : f32, i64)~n    ^kwb:~n", []),
    format(S, "      %iw0 = llvm.mul %kw, %Dil : i64~n      %iw = llvm.add %ws, %iw0 : i64~n", []),
    %% bounds: ih in [0,H) and iw in [0,W)
    format(S, "      %ihp = llvm.icmp \"sge\" %ih, %z : i64~n      %ihq = llvm.icmp \"slt\" %ih, %H : i64~n      %iwp = llvm.icmp \"sge\" %iw, %z : i64~n      %iwq = llvm.icmp \"slt\" %iw, %W : i64~n", []),
    format(S, "      %a1 = llvm.and %ihp, %ihq : i1~n      %a2 = llvm.and %iwp, %iwq : i1~n      %inb = llvm.and %a1, %a2 : i1~n", []),
    format(S, "      llvm.cond_br %inb, ^load, ^skip~n    ^load:~n", []),
    %% input offset ((nc*H + ih)*W + iw)
    format(S, "      %r0 = llvm.add %nc_h, %ih : i64~n      %r1 = llvm.mul %r0, %W : i64~n      %off = llvm.add %r1, %iw : i64~n", []),
    format(S, "      %xp = llvm.getelementptr %x[%off] : (!llvm.ptr, i64) -> !llvm.ptr, f32~n      %v = llvm.load %xp : !llvm.ptr -> f32~n", []),
    format(S, Comb, []),   %% %nacc from %acc,%v
    format(S, "      %ncnt = llvm.add %cnt, %one : i64~n      llvm.br ^kwnext(%nacc, %ncnt : f32, i64)~n", []),
    format(S, "    ^skip:~n      llvm.br ^kwnext(%acc, %cnt : f32, i64)~n", []),
    format(S, "    ^kwnext(%acc2: f32, %cnt2: i64):~n      %kwn = llvm.add %kw, %one : i64~n      llvm.br ^kwloop(%kwn, %acc2, %cnt2 : i64, f32, i64)~n", []),
    format(S, "    ^kwd(%accw: f32, %cntw: i64):~n      %khn = llvm.add %kh, %one : i64~n      llvm.br ^khloop(%khn, %accw, %cntw : i64, f32, i64)~n", []),
    format(S, "    ^khd(%accf: f32, %cntf: i64):~n", []),
    mlir_pool_final(FinalAvg, S),  %% %res from %acch,%cnth
    format(S, "      %op = llvm.getelementptr %out[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, f32~n      llvm.store %res, %op : f32, !llvm.ptr~n", []),
    format(S, "      llvm.br ^done~n    ^done:~n      llvm.return~n    }~n  }~n}~n", []),
    close(S),
    format("Generated MLIR-GPU 2D pool (~w) ~w -> ~w~n", [Kind, Name, OutFile]).

mlir_pool_parts(max, "-3.40282347E+38", "      %gt = arith.cmpf ogt, %v, %acc : f32~n      %nacc = arith.select %gt, %v, %acc : f32~n", nomean).
mlir_pool_parts(avg, "0.0", "      %nacc = llvm.fadd %acc, %v : f32~n", mean).

%% final: max -> result is accumulator; avg -> acc / count(valid)
mlir_pool_final(nomean, S) :- !,
    format(S, "      %fz2 = llvm.mlir.constant(0.0 : f32) : f32~n      %res = llvm.fadd %accf, %fz2 : f32~n", []).
mlir_pool_final(mean, S) :-
    format(S, "      %cf = llvm.sitofp %cntf : i64 to f32~n      %res = llvm.fdiv %accf, %cf : f32~n", []).

%% ── MLIR-GPU 2D CONV kernel (convolution class, 4th/final op class, G=1) ─────────
%% emit_mlir_gpu_conv(+Op, +OutFile): naive direct conv, one thread per output
%% (n,oc,oh,ow), triple-nested block-arg loop (ic,kh,kw) accumulating x*w with
%% padding bounds-checks. Mirrors the cuda-c emit_cuda_conv. ABI:
%% k_conv(x,w,out, N,Cin,H,W,Cout,KH,KW,Hout,Wout), 1-D grid one thread/output.
emit_mlir_gpu_conv(Op, OutFile) :-
    op_expr(Op, conv(2, Pad, St, _, Dil, _)),
    open(OutFile, write, S),
    format(S, "// GENERATED MLIR-GPU 2D conv (St=~w Pad=~w Dil=~w G=1) from op_expr(~w) -> NVVM -> PTX -> P4.~n", [St, Pad, Dil, Op]),
    format(S, "module attributes {gpu.container_module} {~n  gpu.module @kernels {~n", []),
    format(S, "    llvm.func @k_conv(%x: !llvm.ptr, %w: !llvm.ptr, %out: !llvm.ptr, %N: i64, %Cin: i64, %H: i64, %W: i64, %Cout: i64, %KH: i64, %KW: i64, %Hout: i64, %Wout: i64) attributes {gpu.kernel, nvvm.kernel} {~n", []),
    format(S, "      %bid = nvvm.read.ptx.sreg.ctaid.x : i32~n      %bdim = nvvm.read.ptx.sreg.ntid.x : i32~n      %tid = nvvm.read.ptx.sreg.tid.x : i32~n", []),
    format(S, "      %m1 = llvm.mul %bid, %bdim : i32~n      %g32 = llvm.add %m1, %tid : i32~n      %idx = llvm.sext %g32 : i32 to i64~n", []),
    format(S, "      %nc = llvm.mul %N, %Cout : i64~n      %nch = llvm.mul %nc, %Hout : i64~n      %total = llvm.mul %nch, %Wout : i64~n", []),
    format(S, "      %ok = llvm.icmp \"slt\" %idx, %total : i64~n      llvm.cond_br %ok, ^body, ^done~n    ^body:~n", []),
    %% decode idx -> ow, oh, oc, n
    format(S, "      %ow = llvm.srem %idx, %Wout : i64~n      %d1 = llvm.sdiv %idx, %Wout : i64~n", []),
    format(S, "      %oh = llvm.srem %d1, %Hout : i64~n      %d2 = llvm.sdiv %d1, %Hout : i64~n", []),
    format(S, "      %oc = llvm.srem %d2, %Cout : i64~n      %n = llvm.sdiv %d2, %Cout : i64~n", []),
    format(S, "      %St = llvm.mlir.constant(~w : i64) : i64~n      %Pad = llvm.mlir.constant(~w : i64) : i64~n      %Dil = llvm.mlir.constant(~w : i64) : i64~n", [St, Pad, Dil]),
    format(S, "      %z = llvm.mlir.constant(0 : i64) : i64~n      %one = llvm.mlir.constant(1 : i64) : i64~n      %i0 = llvm.mlir.constant(0.0 : f32) : f32~n", []),
    format(S, "      %hs0 = llvm.mul %oh, %St : i64~n      %hs = llvm.sub %hs0, %Pad : i64~n", []),
    format(S, "      %ws0 = llvm.mul %ow, %St : i64~n      %ws = llvm.sub %ws0, %Pad : i64~n", []),
    %% ic loop
    format(S, "      llvm.br ^icloop(%z, %i0 : i64, f32)~n", []),
    format(S, "    ^icloop(%ic: i64, %acc_i: f32):~n", []),
    format(S, "      %icok = llvm.icmp \"slt\" %ic, %Cin : i64~n      llvm.cond_br %icok, ^icb, ^icd(%acc_i : f32)~n    ^icb:~n", []),
    %% kh loop
    format(S, "      llvm.br ^khloop(%z, %acc_i : i64, f32)~n", []),
    format(S, "    ^khloop(%kh: i64, %acc_h: f32):~n", []),
    format(S, "      %khok = llvm.icmp \"slt\" %kh, %KH : i64~n      llvm.cond_br %khok, ^khb, ^khd(%acc_h : f32)~n    ^khb:~n", []),
    format(S, "      %ih0 = llvm.mul %kh, %Dil : i64~n      %ih = llvm.add %hs, %ih0 : i64~n", []),
    %% kw loop
    format(S, "      llvm.br ^kwloop(%z, %acc_h : i64, f32)~n", []),
    format(S, "    ^kwloop(%kw: i64, %acc: f32):~n", []),
    format(S, "      %kwok = llvm.icmp \"slt\" %kw, %KW : i64~n      llvm.cond_br %kwok, ^kwb, ^kwd(%acc : f32)~n    ^kwb:~n", []),
    format(S, "      %iw0 = llvm.mul %kw, %Dil : i64~n      %iw = llvm.add %ws, %iw0 : i64~n", []),
    %% bounds check ih,iw
    format(S, "      %p1 = llvm.icmp \"sge\" %ih, %z : i64~n      %p2 = llvm.icmp \"slt\" %ih, %H : i64~n      %p3 = llvm.icmp \"sge\" %iw, %z : i64~n      %p4 = llvm.icmp \"slt\" %iw, %W : i64~n", []),
    format(S, "      %b1 = llvm.and %p1, %p2 : i1~n      %b2 = llvm.and %p3, %p4 : i1~n      %inb = llvm.and %b1, %b2 : i1~n", []),
    format(S, "      llvm.cond_br %inb, ^load, ^skip~n    ^load:~n", []),
    %% x offset ((n*Cin + ic)*H + ih)*W + iw
    format(S, "      %xa = llvm.mul %n, %Cin : i64~n      %xb = llvm.add %xa, %ic : i64~n      %xc = llvm.mul %xb, %H : i64~n      %xd = llvm.add %xc, %ih : i64~n      %xe = llvm.mul %xd, %W : i64~n      %xoff = llvm.add %xe, %iw : i64~n", []),
    format(S, "      %xp = llvm.getelementptr %x[%xoff] : (!llvm.ptr, i64) -> !llvm.ptr, f32~n      %xv = llvm.load %xp : !llvm.ptr -> f32~n", []),
    %% w offset ((oc*Cin + ic)*KH + kh)*KW + kw
    format(S, "      %wa = llvm.mul %oc, %Cin : i64~n      %wb = llvm.add %wa, %ic : i64~n      %wc = llvm.mul %wb, %KH : i64~n      %wd = llvm.add %wc, %kh : i64~n      %we = llvm.mul %wd, %KW : i64~n      %woff = llvm.add %we, %kw : i64~n", []),
    format(S, "      %wp = llvm.getelementptr %w[%woff] : (!llvm.ptr, i64) -> !llvm.ptr, f32~n      %wv = llvm.load %wp : !llvm.ptr -> f32~n", []),
    %% fma: acc + xv*wv
    format(S, "      %prod = llvm.fmul %xv, %wv : f32~n      %nacc = llvm.fadd %acc, %prod : f32~n", []),
    format(S, "      llvm.br ^kwnext(%nacc : f32)~n    ^skip:~n      llvm.br ^kwnext(%acc : f32)~n", []),
    format(S, "    ^kwnext(%accw: f32):~n      %kwn = llvm.add %kw, %one : i64~n      llvm.br ^kwloop(%kwn, %accw : i64, f32)~n", []),
    format(S, "    ^kwd(%acckw: f32):~n      %khn = llvm.add %kh, %one : i64~n      llvm.br ^khloop(%khn, %acckw : i64, f32)~n", []),
    format(S, "    ^khd(%acckh: f32):~n      %icn = llvm.add %ic, %one : i64~n      llvm.br ^icloop(%icn, %acckh : i64, f32)~n", []),
    format(S, "    ^icd(%accf: f32):~n", []),
    format(S, "      %op = llvm.getelementptr %out[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, f32~n      llvm.store %accf, %op : f32, !llvm.ptr~n", []),
    format(S, "      llvm.br ^done~n    ^done:~n      llvm.return~n    }~n  }~n}~n", []),
    close(S),
    format("Generated MLIR-GPU 2D conv (G=1) ~w -> ~w~n", [Op, OutFile]).
