%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════════
%% lower_schedule_mlir.pl — lower a SCHEDULE to a TILED MLIR-GPU reduce.
%% THE PAYOFF: the same schedule that drove the tuned cuda-c reduce produces a
%% TILED MLIR reduce (block-per-row + thread-strided coalesced + nvvm.shfl warp
%% tree reduce), replacing the naive 1-thread-per-row MLIR reduce.
%% v1: single-warp block (blockDim=32) — block-per-row, 32 threads stride the row,
%% warp-shuffle tree reduce, lane0 stores. No shared mem needed (one warp). This
%% proves schedule->tiled-MLIR with the warp-shuffle (the key coalescing+tree win).
%% Multi-warp (shared-mem cross-warp) is the v2 extension. Author: Iyun, 2026-06-08
%% ═══════════════════════════════════════════════════════════════════════════
:- module(lower_schedule_mlir, [emit_schedule_mlir/3]).
:- use_module(library(lists)).
:- use_module(schedule_ir).

emit_schedule_mlir(Op, SchedName, OutFile) :-
    tile_schedule(Op, SchedName, schedule(reduce, Kind, _Prims)),
    schedule_combine(Kind, InitC, Comb),
    mlir_init_lit(InitC, InitLit),
    open(OutFile, write, S),
    format(S, "// GENERATED from SCHEDULE-IR (tiled_row_reduce, ~w) -> MLIR-GPU. One schedule, N backends.~n", [Kind]),
    format(S, "// TILED v1: block-per-row, 32-thread strided coalesced, nvvm.shfl warp tree reduce.~n", []),
    format(S, "module attributes {gpu.container_module} {~n  gpu.module @kernels {~n", []),
    format(S, "    llvm.func @k_reduce(%x: !llvm.ptr, %out: !llvm.ptr, %R: i32, %C: i32) attributes {gpu.kernel, nvvm.kernel} {~n", []),
    format(S, "      %r = nvvm.read.ptx.sreg.ctaid.x : i32~n      %t = nvvm.read.ptx.sreg.tid.x : i32~n      %bdim = nvvm.read.ptx.sreg.ntid.x : i32~n", []),
    format(S, "      %rok = llvm.icmp \"slt\" %r, %R : i32~n      llvm.cond_br %rok, ^body, ^done~n    ^body:~n", []),
    format(S, "      %r64 = llvm.sext %r : i32 to i64~n      %C64 = llvm.sext %C : i32 to i64~n      %base = llvm.mul %r64, %C64 : i64~n", []),
    format(S, "      %init = ~w~n", [InitLit]),
    %% thread-strided accumulate: for(c=t; c<C; c+=bdim)
    format(S, "      llvm.br ^sloop(%t, %init : i32, f32)~n    ^sloop(%c: i32, %acc: f32):~n", []),
    format(S, "      %cok = llvm.icmp \"slt\" %c, %C : i32~n      llvm.cond_br %cok, ^sb, ^sd(%acc : f32)~n    ^sb:~n", []),
    format(S, "      %c64 = llvm.sext %c : i32 to i64~n      %off = llvm.add %base, %c64 : i64~n", []),
    format(S, "      %xp = llvm.getelementptr %x[%off] : (!llvm.ptr, i64) -> !llvm.ptr, f32~n      %v = llvm.load %xp : !llvm.ptr -> f32~n", []),
    mlir_comb(Comb, "%acc", "%v", "%nacc", CS), format(S, "~w", [CS]),
    format(S, "      %cn = llvm.add %c, %bdim : i32~n      llvm.br ^sloop(%cn, %nacc : i32, f32)~n    ^sd(%acc1: f32):~n", []),
    %% warp-shuffle tree reduce (16,8,4,2,1)
    format(S, "      %mask = llvm.mlir.constant(-1 : i32) : i32~n      %wc = llvm.mlir.constant(31 : i32) : i32~n", []),
    mlir_shfl_tree("w", "%acc1", Comb, Red, ST), format(S, "~w", [ST]),
    %% finalize (mean / C) + lane0 store
    mlir_final(Kind, Red, "%C64", "%res", FS), format(S, "~w", [FS]),
    format(S, "      %fc0 = llvm.mlir.constant(0 : i32) : i32~n      %lane = llvm.and %t, %wc : i32~n      %lz = llvm.icmp \"eq\" %lane, %fc0 : i32~n", []),
    format(S, "      llvm.cond_br %lz, ^store, ^done~n    ^store:~n", []),
    format(S, "      %op = llvm.getelementptr %out[%r64] : (!llvm.ptr, i64) -> !llvm.ptr, f32~n      llvm.store %res, %op : f32, !llvm.ptr~n      llvm.br ^done~n", []),
    format(S, "    ^done:~n      llvm.return~n    }~n  }~n}~n", []),
    close(S),
    format("Generated SCHEDULE-IR tiled MLIR reduce (~w) -> ~w~n", [Kind, OutFile]).

mlir_init_lit("0.0",             "llvm.mlir.constant(0.0 : f32) : f32").
mlir_init_lit("-3.40282347e+38", "llvm.mlir.constant(-3.40282347E+38 : f32) : f32").
mlir_init_lit("3.40282347e+38",  "llvm.mlir.constant(3.40282347E+38 : f32) : f32").

mlir_comb(add,  A, V, O, S) :- format(atom(S), "      ~w = llvm.fadd ~w, ~w : f32~n", [O, A, V]).
mlir_comb(maxf, A, V, O, S) :- format(atom(S), "      ~w_c = arith.cmpf ogt, ~w, ~w : f32~n      ~w = arith.select ~w_c, ~w, ~w : f32~n", [O, V, A, O, O, V, A]).
mlir_comb(minf, A, V, O, S) :- format(atom(S), "      ~w_c = arith.cmpf olt, ~w, ~w : f32~n      ~w = arith.select ~w_c, ~w, ~w : f32~n", [O, V, A, O, O, V, A]).

mlir_shfl_tree(Pfx, In, Comb, Out, Stmt) :- shfl_steps([16,8,4,2,1], Pfx, In, Comb, Out, "", Stmt).
shfl_steps([], _, Cur, _, Cur, Acc, Acc).
shfl_steps([Off|T], Pfx, Cur, Comb, Out, Acc, Stmt) :-
    format(atom(OC), "%~w_o~w", [Pfx, Off]),
    format(atom(NV), "%~w_s~w", [Pfx, Off]),
    format(atom(R),  "%~w_r~w", [Pfx, Off]),
    format(atom(S1), "      ~w = llvm.mlir.constant(~w : i32) : i32~n      ~w = nvvm.shfl.sync down %mask, ~w, ~w, %wc : f32 -> f32~n", [OC, Off, NV, Cur, OC]),
    mlir_comb(Comb, Cur, NV, R, CombS),
    atom_concat(Acc, S1, A1), atom_concat(A1, CombS, A2),
    shfl_steps(T, Pfx, R, Comb, Out, A2, Stmt).

mlir_final(mean, Acc, C, Out, S) :- !, format(atom(S), "      ~w_cf = llvm.sitofp ~w : i64 to f32~n      ~w = llvm.fdiv ~w, ~w_cf : f32~n", [Out, C, Out, Acc, Out]).
mlir_final(_,    Acc, _, Out, S) :- format(atom(S), "      ~w_z = llvm.mlir.constant(0.0 : f32) : f32~n      ~w = llvm.fadd ~w, ~w_z : f32~n", [Out, Out, Acc, Out]).

%% ── tiled_gemm schedule -> hand-written llvm-dialect MLIR ───────────────────
%% emit_schedule_mlir(Op, tiled_gemm(BM,BN,BK,TM,TN), OutFile): the validated
%% register-blocked shared-memory tiled GEMM (proto bit-exact on P4, memory
%% d998f1b7). The TMxTN register tile = TM*TN loop-carried block-arg accumulators
%% with DISTINCT names per block (kacc/acc/dacc/facc). Reference: /tmp/gen_mlir_gemm.py.
emit_schedule_mlir(Op, tiled_gemm(BM,BN,BK,TM,TN), OutFile) :-
    ( op_expr(Op, _) ; true ), !,
    NTH is (BM // TM) * (BN // TN),
    NAS is BM * BK, NBS is BK * BN,
    acc_names(TM, TN, "acc", AccBind, AccList),
    acc_names(TM, TN, "kacc", KAccBind, KAccList),
    acc_names(TM, TN, "dacc", DAccBind, _DAccList0),
    dacc_list(TM, TN, DAccList),
    acc_names(TM, TN, "facc", FAccBind, _),
    f32s(TM, TN, F32s),
    zeros(TM, TN, Zeros),
    open(OutFile, write, S),
    format(S, "// GENERATED from SCHEDULE-IR tiled_gemm(BM=~w BN=~w BK=~w TM=~w TN=~w) -> MLIR. NTH=~w.~n", [BM,BN,BK,TM,TN,NTH]),
    format(S, "// LAUNCH: thread_tile threads=~w grid=(N/~w, M/~w) block=~w~n", [NTH, BN, BM, NTH]),
    format(S, "module attributes {gpu.container_module} {~n  gpu.module @kernels {~n", []),
    format(S, "    llvm.mlir.global internal @As() {addr_space = 3 : i32} : !llvm.array<~w x f32>~n", [NAS]),
    format(S, "    llvm.mlir.global internal @Bs() {addr_space = 3 : i32} : !llvm.array<~w x f32>~n", [NBS]),
    format(S, "    llvm.func @k_gemm(%A: !llvm.ptr, %B: !llvm.ptr, %C: !llvm.ptr, %M: i32, %N: i32, %K: i32) attributes {gpu.kernel, nvvm.kernel} {~n", []),
    format(S, "      %cBM = llvm.mlir.constant(~w : i32) : i32~n      %cBN = llvm.mlir.constant(~w : i32) : i32~n", [BM,BN]),
    format(S, "      %cBK = llvm.mlir.constant(~w : i32) : i32~n      %cTM = llvm.mlir.constant(~w : i32) : i32~n      %cTN = llvm.mlir.constant(~w : i32) : i32~n", [BK,TM,TN]),
    format(S, "      %cNTH = llvm.mlir.constant(~w : i32) : i32~n      %c0 = llvm.mlir.constant(0 : i32) : i32~n      %c1 = llvm.mlir.constant(1 : i32) : i32~n      %f0 = llvm.mlir.constant(0.0 : f32) : f32~n", [NTH]),
    format(S, "      %tid = nvvm.read.ptx.sreg.tid.x : i32~n      %bx = nvvm.read.ptx.sreg.ctaid.x : i32~n      %by = nvvm.read.ptx.sreg.ctaid.y : i32~n", []),
    format(S, "      %bnt = llvm.sdiv %cBN, %cTN : i32~n      %tRow = llvm.sdiv %tid, %bnt : i32~n      %tCol = llvm.srem %tid, %bnt : i32~n", []),
    format(S, "      %blockRow = llvm.mul %by, %cBM : i32~n      %blockCol = llvm.mul %bx, %cBN : i32~n", []),
    format(S, "      %As = llvm.mlir.addressof @As : !llvm.ptr<3>~n      %Bs = llvm.mlir.addressof @Bs : !llvm.ptr<3>~n", []),
    format(S, "      llvm.br ^kloop(%c0, ~w : i32, ~w)~n", [Zeros, F32s]),
    format(S, "    ^kloop(%k0: i32, ~w):~n", [KAccBind]),
    format(S, "      %kok = llvm.icmp \"slt\" %k0, %K : i32~n", []),
    format(S, "      llvm.cond_br %kok, ^kbody, ^kdone(~w : ~w)~n    ^kbody:~n", [KAccList, F32s]),
    %% load As
    format(S, "      %nAs = llvm.mlir.constant(~w : i32) : i32~n      llvm.br ^loadA(%tid : i32)~n    ^loadA(%ia: i32):~n", [NAS]),
    format(S, "      %iaok = llvm.icmp \"slt\" %ia, %nAs : i32~n      llvm.cond_br %iaok, ^loadAb, ^loadAd~n    ^loadAb:~n", []),
    format(S, "      %ar = llvm.sdiv %ia, %cBK : i32~n      %ac = llvm.srem %ia, %cBK : i32~n      %agr = llvm.add %blockRow, %ar : i32~n      %agc = llvm.add %k0, %ac : i32~n", []),
    format(S, "      %arok = llvm.icmp \"slt\" %agr, %M : i32~n      %acok = llvm.icmp \"slt\" %agc, %K : i32~n      %ainb = llvm.and %arok, %acok : i1~n", []),
    format(S, "      %aoff0 = llvm.mul %agr, %K : i32~n      %aoff = llvm.add %aoff0, %agc : i32~n      %aoff64 = llvm.sext %aoff : i32 to i64~n", []),
    format(S, "      %ap = llvm.getelementptr %A[%aoff64] : (!llvm.ptr, i64) -> !llvm.ptr, f32~n      %aval0 = llvm.load %ap : !llvm.ptr -> f32~n      %aval = arith.select %ainb, %aval0, %f0 : f32~n", []),
    format(S, "      %ia64 = llvm.sext %ia : i32 to i64~n      %asp = llvm.getelementptr %As[%ia64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32~n      llvm.store %aval, %asp : f32, !llvm.ptr<3>~n", []),
    format(S, "      %ian = llvm.add %ia, %cNTH : i32~n      llvm.br ^loadA(%ian : i32)~n    ^loadAd:~n", []),
    %% load Bs
    format(S, "      %nBs = llvm.mlir.constant(~w : i32) : i32~n      llvm.br ^loadB(%tid : i32)~n    ^loadB(%ib: i32):~n", [NBS]),
    format(S, "      %ibok = llvm.icmp \"slt\" %ib, %nBs : i32~n      llvm.cond_br %ibok, ^loadBb, ^loadBd~n    ^loadBb:~n", []),
    format(S, "      %br = llvm.sdiv %ib, %cBN : i32~n      %bc = llvm.srem %ib, %cBN : i32~n      %bgr = llvm.add %k0, %br : i32~n      %bgc = llvm.add %blockCol, %bc : i32~n", []),
    format(S, "      %brok = llvm.icmp \"slt\" %bgr, %K : i32~n      %bcok = llvm.icmp \"slt\" %bgc, %N : i32~n      %binb = llvm.and %brok, %bcok : i1~n", []),
    format(S, "      %boff0 = llvm.mul %bgr, %N : i32~n      %boff = llvm.add %boff0, %bgc : i32~n      %boff64 = llvm.sext %boff : i32 to i64~n", []),
    format(S, "      %bp = llvm.getelementptr %B[%boff64] : (!llvm.ptr, i64) -> !llvm.ptr, f32~n      %bval0 = llvm.load %bp : !llvm.ptr -> f32~n      %bval = arith.select %binb, %bval0, %f0 : f32~n", []),
    format(S, "      %ib64 = llvm.sext %ib : i32 to i64~n      %bsp = llvm.getelementptr %Bs[%ib64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32~n      llvm.store %bval, %bsp : f32, !llvm.ptr<3>~n", []),
    format(S, "      %ibn = llvm.add %ib, %cNTH : i32~n      llvm.br ^loadB(%ibn : i32)~n    ^loadBd:~n      nvvm.barrier0~n", []),
    %% kk inner loop
    format(S, "      llvm.br ^kkloop(%c0, ~w : i32, ~w)~n    ^kkloop(%kk: i32, ~w):~n", [KAccList, F32s, AccBind]),
    format(S, "      %kkok = llvm.icmp \"slt\" %kk, %cBK : i32~n      llvm.cond_br %kkok, ^kkbody, ^kkdone(~w : ~w)~n    ^kkbody:~n", [AccList, F32s]),
    emit_areg_loads(S, TM),
    emit_breg_loads(S, TN),
    emit_fma(S, TM, TN),
    nacc_list(TM, TN, NAccList),
    format(S, "      %kkn = llvm.add %kk, %c1 : i32~n      llvm.br ^kkloop(%kkn, ~w : i32, ~w)~n", [NAccList, F32s]),
    format(S, "    ^kkdone(~w):~n      nvvm.barrier0~n      %k0n = llvm.add %k0, %cBK : i32~n", [DAccBind]),
    format(S, "      llvm.br ^kloop(%k0n, ~w : i32, ~w)~n", [DAccList, F32s]),
    format(S, "    ^kdone(~w):~n", [FAccBind]),
    emit_gemm_store(S, TM, TN),
    format(S, "      llvm.return~n    }~n  }~n}~n", []),
    close(S),
    format("Generated SCHEDULE-IR tiled MLIR GEMM (~wx~w, TM~w TN~w) -> ~w~n", [BM,BN,TM,TN,OutFile]).

%% ── helpers: TMxTN accumulator name generation ──────────────────────────────
acc_names(TM, TN, Pfx, Bind, List) :-
    findall(B-L, (between(0,TM,Ti), Ti < TM, between(0,TN,Tj), Tj < TN,
                  format(atom(B), "%~w_~w_~w: f32", [Pfx,Ti,Tj]),
                  format(atom(L), "%~w_~w_~w", [Pfx,Ti,Tj])), Pairs),
    pairs_keys_values(Pairs, Binds, Lists),
    atomic_list_concat(Binds, ', ', Bind),
    atomic_list_concat(Lists, ', ', List).
dacc_list(TM, TN, List) :- acc_names(TM, TN, "dacc", _, List).
nacc_list(TM, TN, List) :- acc_names(TM, TN, "nacc", _, List).
f32s(TM, TN, F32s) :- N is TM*TN, length(L, N), maplist(=("f32"), L), atomic_list_concat(L, ', ', F32s).
zeros(TM, TN, Zeros) :- N is TM*TN, length(L, N), maplist(=("%f0"), L), atomic_list_concat(L, ', ', Zeros).

emit_areg_loads(S, TM) :-
    forall((between(0,TM,I), I < TM),
      ( format(S, "      %ci_~w = llvm.mlir.constant(~w : i32) : i32~n      %arow~wa = llvm.mul %tRow, %cTM : i32~n      %arow~w = llvm.add %arow~wa, %ci_~w : i32~n", [I,I,I,I,I,I]),
        format(S, "      %aidx~wa = llvm.mul %arow~w, %cBK : i32~n      %aidx~w = llvm.add %aidx~wa, %kk : i32~n      %aidx~w64 = llvm.sext %aidx~w : i32 to i64~n", [I,I,I,I,I,I]),
        format(S, "      %areg~wp = llvm.getelementptr %As[%aidx~w64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32~n      %areg~w = llvm.load %areg~wp : !llvm.ptr<3> -> f32~n", [I,I,I,I]) )).
emit_breg_loads(S, TN) :-
    forall((between(0,TN,J), J < TN),
      ( format(S, "      %cj_~w = llvm.mlir.constant(~w : i32) : i32~n      %bcol~wa = llvm.mul %tCol, %cTN : i32~n      %bcol~w = llvm.add %bcol~wa, %cj_~w : i32~n", [J,J,J,J,J,J]),
        format(S, "      %bidx~wa = llvm.mul %kk, %cBN : i32~n      %bidx~w = llvm.add %bidx~wa, %bcol~w : i32~n      %bidx~w64 = llvm.sext %bidx~w : i32 to i64~n", [J,J,J,J,J,J]),
        format(S, "      %breg~wp = llvm.getelementptr %Bs[%bidx~w64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32~n      %breg~w = llvm.load %breg~wp : !llvm.ptr<3> -> f32~n", [J,J,J,J]) )).
emit_fma(S, TM, TN) :-
    forall((between(0,TM,I), I < TM, between(0,TN,J), J < TN),
      format(S, "      %prod~w_~w = llvm.fmul %areg~w, %breg~w : f32~n      %nacc_~w_~w = llvm.fadd %acc_~w_~w, %prod~w_~w : f32~n", [I,J,I,J,I,J,I,J,I,J])).
emit_gemm_store(S, TM, TN) :-
    forall((between(0,TM,I), I < TM, between(0,TN,J), J < TN),
      ( format(S, "      %cgi_~w_~w = llvm.mlir.constant(~w : i32) : i32~n      %cgj_~w_~w = llvm.mlir.constant(~w : i32) : i32~n", [I,J,I,I,J,J]),
        format(S, "      %grr~w_~wa = llvm.mul %tRow, %cTM : i32~n      %grr~w_~wb = llvm.add %grr~w_~wa, %cgi_~w_~w : i32~n      %gr~w_~w = llvm.add %blockRow, %grr~w_~wb : i32~n", [I,J,I,J,I,J,I,J,I,J,I,J]),
        format(S, "      %gcc~w_~wa = llvm.mul %tCol, %cTN : i32~n      %gcc~w_~wb = llvm.add %gcc~w_~wa, %cgj_~w_~w : i32~n      %gc~w_~w = llvm.add %blockCol, %gcc~w_~wb : i32~n", [I,J,I,J,I,J,I,J,I,J,I,J]),
        format(S, "      %grok~w_~w = llvm.icmp \"slt\" %gr~w_~w, %M : i32~n      %gcok~w_~w = llvm.icmp \"slt\" %gc~w_~w, %N : i32~n      %gok~w_~w = llvm.and %grok~w_~w, %gcok~w_~w : i1~n", [I,J,I,J,I,J,I,J,I,J,I,J,I,J]),
        format(S, "      llvm.cond_br %gok~w_~w, ^st~w_~w, ^sk~w_~w~n    ^st~w_~w:~n", [I,J,I,J,I,J,I,J]),
        format(S, "      %coff~w_~wa = llvm.mul %gr~w_~w, %N : i32~n      %coff~w_~w = llvm.add %coff~w_~wa, %gc~w_~w : i32~n      %coff~w_~w64 = llvm.sext %coff~w_~w : i32 to i64~n", [I,J,I,J,I,J,I,J,I,J,I,J,I,J]),
        format(S, "      %cp~w_~w = llvm.getelementptr %C[%coff~w_~w64] : (!llvm.ptr, i64) -> !llvm.ptr, f32~n      llvm.store %facc_~w_~w, %cp~w_~w : f32, !llvm.ptr~n      llvm.br ^sk~w_~w~n    ^sk~w_~w:~n", [I,J,I,J,I,J,I,J,I,J,I,J]) )).
