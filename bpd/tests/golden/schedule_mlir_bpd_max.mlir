// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
// GENERATED from SCHEDULE-IR (tiled_row_reduce, max) -> MLIR-GPU. One schedule, N backends.
// TILED v1: block-per-row, 32-thread strided coalesced, nvvm.shfl warp tree reduce.
module attributes {gpu.container_module} {
  gpu.module @kernels {
    llvm.func @k_reduce(%x: !llvm.ptr, %out: !llvm.ptr, %R: i32, %C: i32) attributes {gpu.kernel, nvvm.kernel} {
      %r = nvvm.read.ptx.sreg.ctaid.x : i32
      %t = nvvm.read.ptx.sreg.tid.x : i32
      %bdim = nvvm.read.ptx.sreg.ntid.x : i32
      %rok = llvm.icmp "slt" %r, %R : i32
      llvm.cond_br %rok, ^body, ^done
    ^body:
      %r64 = llvm.sext %r : i32 to i64
      %C64 = llvm.sext %C : i32 to i64
      %base = llvm.mul %r64, %C64 : i64
      %init = llvm.mlir.constant(-3.40282347E+38 : f32) : f32
      llvm.br ^sloop(%t, %init : i32, f32)
    ^sloop(%c: i32, %acc: f32):
      %cok = llvm.icmp "slt" %c, %C : i32
      llvm.cond_br %cok, ^sb, ^sd(%acc : f32)
    ^sb:
      %c64 = llvm.sext %c : i32 to i64
      %off = llvm.add %base, %c64 : i64
      %xp = llvm.getelementptr %x[%off] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %v = llvm.load %xp : !llvm.ptr -> f32
      %nacc_c = arith.cmpf ogt, %v, %acc : f32
      %nacc = arith.select %nacc_c, %v, %acc : f32
      %cn = llvm.add %c, %bdim : i32
      llvm.br ^sloop(%cn, %nacc : i32, f32)
    ^sd(%acc1: f32):
      %mask = llvm.mlir.constant(-1 : i32) : i32
      %wc = llvm.mlir.constant(31 : i32) : i32
      %w_o16 = llvm.mlir.constant(16 : i32) : i32
      %w_s16 = nvvm.shfl.sync down %mask, %acc1, %w_o16, %wc : f32 -> f32
      %w_r16_c = arith.cmpf ogt, %w_s16, %acc1 : f32
      %w_r16 = arith.select %w_r16_c, %w_s16, %acc1 : f32
      %w_o8 = llvm.mlir.constant(8 : i32) : i32
      %w_s8 = nvvm.shfl.sync down %mask, %w_r16, %w_o8, %wc : f32 -> f32
      %w_r8_c = arith.cmpf ogt, %w_s8, %w_r16 : f32
      %w_r8 = arith.select %w_r8_c, %w_s8, %w_r16 : f32
      %w_o4 = llvm.mlir.constant(4 : i32) : i32
      %w_s4 = nvvm.shfl.sync down %mask, %w_r8, %w_o4, %wc : f32 -> f32
      %w_r4_c = arith.cmpf ogt, %w_s4, %w_r8 : f32
      %w_r4 = arith.select %w_r4_c, %w_s4, %w_r8 : f32
      %w_o2 = llvm.mlir.constant(2 : i32) : i32
      %w_s2 = nvvm.shfl.sync down %mask, %w_r4, %w_o2, %wc : f32 -> f32
      %w_r2_c = arith.cmpf ogt, %w_s2, %w_r4 : f32
      %w_r2 = arith.select %w_r2_c, %w_s2, %w_r4 : f32
      %w_o1 = llvm.mlir.constant(1 : i32) : i32
      %w_s1 = nvvm.shfl.sync down %mask, %w_r2, %w_o1, %wc : f32 -> f32
      %w_r1_c = arith.cmpf ogt, %w_s1, %w_r2 : f32
      %w_r1 = arith.select %w_r1_c, %w_s1, %w_r2 : f32
      %res_z = llvm.mlir.constant(0.0 : f32) : f32
      %res = llvm.fadd %w_r1, %res_z : f32
      %fc0 = llvm.mlir.constant(0 : i32) : i32
      %lane = llvm.and %t, %wc : i32
      %lz = llvm.icmp "eq" %lane, %fc0 : i32
      llvm.cond_br %lz, ^store, ^done
    ^store:
      %op = llvm.getelementptr %out[%r64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %res, %op : f32, !llvm.ptr
      llvm.br ^done
    ^done:
      llvm.return
    }
  }
}
