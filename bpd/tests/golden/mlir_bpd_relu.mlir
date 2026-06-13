// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
// GENERATED from robust_op_match(bpd_relu) - MLIR gpu/llvm dialect (->NVVM->PTX->P4).
// Kernel params (PTX ABI numbers them relu_param_0..2 positionally):
//   param_0 = src  (const float*)  - input pointer
//   param_1 = dst  (float*)        - output pointer
//   param_2 = n    (i64)           - element count
module attributes {gpu.container_module} {
  gpu.module @kernels {
    llvm.func @relu(%src: !llvm.ptr, %dst: !llvm.ptr, %n: i64) attributes {gpu.kernel, nvvm.kernel} {
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
      %e0 = arith.cmpf uno, %v, %v : f32
      %e1 = arith.constant 0.0 : f32
      %e2 = arith.cmpf oge, %v, %e1 : f32
      %e3 = arith.constant 0.0 : f32
      %e4 = arith.select %e2, %v, %e3 : f32
      %e5 = arith.select %e0, %v, %e4 : f32
      %dp = llvm.getelementptr %dst[%i] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %e5, %dp : f32, !llvm.ptr
      llvm.br ^done
    ^done:
      llvm.return
    }
  }
}
