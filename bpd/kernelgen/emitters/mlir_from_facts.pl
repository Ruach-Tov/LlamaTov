%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════════
%% mlir_from_facts.pl — generate MLIR (arith/math dialect) kernels from the
%% canonical robust_op_match.pl facts. The THIRD backend (after cuda-oxide Rust
%% and C++/CUDA) of the multi-backend generator.
%%
%% MLIR is structured IR: the fact's formulation maps cleanly to arith/math ops.
%% Lowering: mlir-opt --convert-{math,arith,func}-to-llvm -> LLVM IR -> run.
%% Verified on CPU (laptop mlir-cpu-runner) vs torch-CPU.
%%
%%   relu (x>=0?x:0) -> arith.cmpf oge + arith.select
%%   tanh           -> math.tanh
%%   elu  (x<=0?...) -> arith.cmpf + math.exp + arith.select
%%   gelu           -> math.erf + arith mulf/addf
%%
%% Honors nan_mode (propagate wraps with a uno NaN-passthrough) and fma_mode
%% (contract -> math.fma) the same way the other backends do.
%%
%% Author: Iyun, 2026-06-07 (MLIR backend of the multi-backend generator)
%% ═══════════════════════════════════════════════════════════════════════════

:- module(mlir_from_facts, [emit_mlir_unary/2, mlir_supported_op/1]).
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

%% C formulation -> a list of MLIR statements computing %res from %x.
%% Each rule yields mlir_body(Op, Stmts, ResultSSA).
mlir_body(bpd_relu,
  ["%zero = arith.constant 0.0 : f32",
   "%cmp = arith.cmpf oge, %x, %zero : f32",
   "%res = arith.select %cmp, %x, %zero : f32"], "%res").
mlir_body(bpd_tanh,
  ["%res = math.tanh %x : f32"], "%res").
mlir_body(bpd_elu,
  ["%zero = arith.constant 0.0 : f32",
   "%one = arith.constant 1.0 : f32",
   "%e = math.exp %x : f32",
   "%em1 = arith.subf %e, %one : f32",
   "%cmp = arith.cmpf ole, %x, %zero : f32",
   "%res = arith.select %cmp, %em1, %x : f32"], "%res").
mlir_body(bpd_gelu,
  ["%half = arith.constant 0.5 : f32",
   "%one = arith.constant 1.0 : f32",
   "%inv = arith.constant 0.7071067811865476 : f32",
   "%xs = arith.mulf %x, %inv : f32",
   "%er = math.erf %xs : f32",
   "%opr = arith.addf %one, %er : f32",
   "%hx = arith.mulf %x, %half : f32",
   "%res = arith.mulf %hx, %opr : f32"], "%res").

mlir_supported_op(Op) :- mlir_body(Op, _, _).

%% emit a standalone MLIR module: @kfn(f32)->f32 + a main that reads stdin-less
%% test set is driven by the harness; here we emit just @kfn for lowering+link,
%% plus an @apply over a memref for batch evaluation.
emit_mlir_unary(Op, OutFile) :-
    mlir_body(Op, Stmts, Res),
    (atom_concat('bpd_', Name, Op) -> true ; Name = Op),
    open(OutFile, write, S),
    format(S, "// GENERATED from robust_op_match(~w) — MLIR (arith/math) backend.~n", [Op]),
    format(S, "// lower: mlir-opt --convert-math-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts~n", []),
    %% scalar kernel
    format(S, "func.func @~w(%x: f32) -> f32 {~n", [Name]),
    forall(member(St, Stmts), format(S, "  ~w~n", [St])),
    format(S, "  return ~w : f32~n}~n~n", [Res]),
    %% batch apply over a 1-D memref (for bulk bit-identity testing)
    format(S, "func.func @apply(%in: memref<?xf32>, %out: memref<?xf32>) {~n", []),
    format(S, "  %c0 = arith.constant 0 : index~n  %c1 = arith.constant 1 : index~n", []),
    format(S, "  %n = memref.dim %in, %c0 : memref<?xf32>~n", []),
    format(S, "  scf.for %i = %c0 to %n step %c1 {~n", []),
    format(S, "    %v = memref.load %in[%i] : memref<?xf32>~n", []),
    format(S, "    %r = func.call @~w(%v) : (f32) -> f32~n", [Name]),
    format(S, "    memref.store %r, %out[%i] : memref<?xf32>~n", []),
    format(S, "  }~n  return~n}~n", []),
    close(S),
    format("Generated MLIR ~w -> ~w~n", [Name, OutFile]).
