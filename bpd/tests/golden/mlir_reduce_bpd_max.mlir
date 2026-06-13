// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
// GENERATED MLIR-GPU axis-reduce (max) from op_expr(bpd_max) -> NVVM -> PTX -> P4.
// params: param_0=x param_1=out (!llvm.ptr), param_2=R param_3=C (i64). 1 thread/row.
module attributes {gpu.container_module} {
  gpu.module @kernels {
    llvm.func @k_reduce(%x: !llvm.ptr, %out: !llvm.ptr, %R: i64, %C: i64) attributes {gpu.kernel, nvvm.kernel} {
      %bid = nvvm.read.ptx.sreg.ctaid.x : i32
      %bdim = nvvm.read.ptx.sreg.ntid.x : i32
      %tid = nvvm.read.ptx.sreg.tid.x : i32
      %t1 = llvm.mul %bid, %bdim : i32
      %r32 = llvm.add %t1, %tid : i32
      %r = llvm.sext %r32 : i32 to i64
      %rok = llvm.icmp "slt" %r, %R : i64
      llvm.cond_br %rok, ^body, ^done
    ^body:
      %zero = llvm.mlir.constant(0 : i64) : i64
      %one = llvm.mlir.constant(1 : i64) : i64
      %init = llvm.mlir.constant(-3.40282347E+38 : f32) : f32
      %base = llvm.mul %r, %C : i64
      llvm.br ^rloop(%zero, %init : i64, f32)
    ^rloop(%c: i64, %acc: f32):
      %cc = llvm.icmp "slt" %c, %C : i64
      llvm.cond_br %cc, ^cbody, ^cdone
    ^cbody:
      %off = llvm.add %base, %c : i64
      %xp = llvm.getelementptr %x[%off] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %v = llvm.load %xp : !llvm.ptr -> f32
      %gt = arith.cmpf ogt, %v, %acc : f32
      %nacc = arith.select %gt, %v, %acc : f32
      %cnext = llvm.add %c, %one : i64
      llvm.br ^rloop(%cnext, %nacc : i64, f32)
    ^cdone:
      %fz = llvm.mlir.constant(0.0 : f32) : f32
      %res = llvm.fadd %acc, %fz : f32
      %op = llvm.getelementptr %out[%r] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %res, %op : f32, !llvm.ptr
      llvm.br ^done
    ^done:
      llvm.return
    }
  }
}
