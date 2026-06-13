// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
// GENERATED MLIR-GPU matmul (fma=contract) from op_expr(bpd_matmul) -> NVVM -> PTX -> P4.
// params: param_0=A param_1=B param_2=C (!llvm.ptr), param_3=n (i64)
module attributes {gpu.container_module} {
  gpu.module @kernels {
    llvm.func @gemm(%A: !llvm.ptr, %B: !llvm.ptr, %C: !llvm.ptr, %n: i64) attributes {gpu.kernel, nvvm.kernel} {
      %bx = nvvm.read.ptx.sreg.ctaid.x : i32
      %nx = nvvm.read.ptx.sreg.ntid.x : i32
      %tx = nvvm.read.ptx.sreg.tid.x : i32
      %by = nvvm.read.ptx.sreg.ctaid.y : i32
      %ny = nvvm.read.ptx.sreg.ntid.y : i32
      %ty = nvvm.read.ptx.sreg.tid.y : i32
      %cxm = llvm.mul %bx, %nx : i32
      %col32 = llvm.add %cxm, %tx : i32
      %col = llvm.sext %col32 : i32 to i64
      %cym = llvm.mul %by, %ny : i32
      %row32 = llvm.add %cym, %ty : i32
      %row = llvm.sext %row32 : i32 to i64
      %rok = llvm.icmp "slt" %row, %n : i64
      %cok = llvm.icmp "slt" %col, %n : i64
      %ok = llvm.and %rok, %cok : i1
      llvm.cond_br %ok, ^body, ^done
    ^body:
      %zero = llvm.mlir.constant(0 : i64) : i64
      %one = llvm.mlir.constant(1 : i64) : i64
      %fz = llvm.mlir.constant(0.0 : f32) : f32
      llvm.br ^kloop(%zero, %fz : i64, f32)
    ^kloop(%k: i64, %acc: f32):
      %kc = llvm.icmp "slt" %k, %n : i64
      llvm.cond_br %kc, ^kbody, ^kdone
    ^kbody:
      %ar = llvm.mul %row, %n : i64
      %ai = llvm.add %ar, %k : i64
      %ap = llvm.getelementptr %A[%ai] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %a = llvm.load %ap : !llvm.ptr -> f32
      %br = llvm.mul %k, %n : i64
      %bi = llvm.add %br, %col : i64
      %bp = llvm.getelementptr %B[%bi] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %b = llvm.load %bp : !llvm.ptr -> f32
      %nacc = llvm.intr.fma(%a, %b, %acc) : (f32, f32, f32) -> f32
      %knext = llvm.add %k, %one : i64
      llvm.br ^kloop(%knext, %nacc : i64, f32)
    ^kdone:
      %cr = llvm.mul %row, %n : i64
      %ci = llvm.add %cr, %col : i64
      %cp = llvm.getelementptr %C[%ci] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %acc, %cp : f32, !llvm.ptr
      llvm.br ^done
    ^done:
      llvm.return
    }
  }
}
