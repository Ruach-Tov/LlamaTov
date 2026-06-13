// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
// GENERATED from robust_op_match(bpd_softplus) - MLIR gpu/llvm dialect (->NVVM->PTX->P4).
// Kernel params (PTX ABI numbers them softplus_param_0..2 positionally):
//   param_0 = src  (const float*)  - input pointer
//   param_1 = dst  (float*)        - output pointer
//   param_2 = n    (i64)           - element count
module attributes {gpu.container_module} {
  gpu.module @kernels {
    llvm.func @softplus(%src: !llvm.ptr, %dst: !llvm.ptr, %n: i64) attributes {gpu.kernel, nvvm.kernel} {
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
      %e0 = arith.constant 1.0 : f32
      %e1 = math.exp %v : f32
      %e2 = arith.addf %e0, %e1 : f32
      %e3 = math.log %e2 : f32
      %dp = llvm.getelementptr %dst[%i] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %e3, %dp : f32, !llvm.ptr
      llvm.br ^done
    ^done:
      llvm.return
    }
  }
}
