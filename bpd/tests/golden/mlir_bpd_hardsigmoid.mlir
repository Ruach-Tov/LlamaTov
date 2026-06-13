// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
// GENERATED from robust_op_match(bpd_hardsigmoid) - MLIR gpu/llvm dialect (->NVVM->PTX->P4).
// Kernel params (PTX ABI numbers them hardsigmoid_param_0..2 positionally):
//   param_0 = src  (const float*)  - input pointer
//   param_1 = dst  (float*)        - output pointer
//   param_2 = n    (i64)           - element count
module attributes {gpu.container_module} {
  gpu.module @kernels {
    llvm.func @hardsigmoid(%src: !llvm.ptr, %dst: !llvm.ptr, %n: i64) attributes {gpu.kernel, nvvm.kernel} {
      %bid = nvvm.read.ptx.sreg.ctaid.x : i32
      %bdim = nvvm.read.ptx.sreg.ntid.x : i32
      %tid = nvvm.read.ptx.sreg.tid.x : i32
      %t1 = llvm.mul %bid, %bdim : i32
      %i32 = llvm.add %t1, %tid : i32
      %i = llvm.sext %i32 : i32 to i64
      %inb = llvm.icmp "slt" %i, %n : i64
      llvm.cond_br %inb, ^body, ^done
    ^body:
      %sp = llvm.getelementptr %src[%i] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %v = llvm.load %sp : !llvm.ptr -> f32
      %e0 = arith.constant 0.16666666666666666 : f32
      %e1 = arith.mulf %v, %e0 : f32
      %e2 = arith.constant 0.5 : f32
      %e3 = arith.addf %e1, %e2 : f32
      %e4 = arith.constant 0.0 : f32
      %e5 = arith.cmpf ole, %e3, %e4 : f32
      %e6 = arith.constant 0.0 : f32
      %e7 = arith.constant 0.16666666666666666 : f32
      %e8 = arith.mulf %v, %e7 : f32
      %e9 = arith.constant 0.5 : f32
      %e10 = arith.addf %e8, %e9 : f32
      %e11 = arith.constant 1.0 : f32
      %e12 = arith.cmpf oge, %e10, %e11 : f32
      %e13 = arith.constant 1.0 : f32
      %e14 = arith.constant 0.16666666666666666 : f32
      %e15 = arith.mulf %v, %e14 : f32
      %e16 = arith.constant 0.5 : f32
      %e17 = arith.addf %e15, %e16 : f32
      %e18 = arith.select %e12, %e13, %e17 : f32
      %e19 = arith.select %e5, %e6, %e18 : f32
      %dp = llvm.getelementptr %dst[%i] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %e19, %dp : f32, !llvm.ptr
      llvm.br ^done
    ^done:
      llvm.return
    }
  }
}
