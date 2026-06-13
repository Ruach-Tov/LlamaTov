// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
// GENERATED from robust_op_match(bpd_tanh) — MLIR gpu-dialect (->NVVM->PTX->P4).
module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @tanh(%x: memref<?xf32>, %c: memref<?xf32>) kernel {
      %c0 = arith.constant 0 : index
      %bid = gpu.block_id x
      %bdim = gpu.block_dim x
      %tid = gpu.thread_id x
      %tmp = arith.muli %bid, %bdim : index
      %i = arith.addi %tmp, %tid : index
      %n = memref.dim %x, %c0 : memref<?xf32>
      %inb = arith.cmpi slt, %i, %n : index
      scf.if %inb {
        %v = memref.load %x[%i] : memref<?xf32>
        %r = math.tanh %v : f32
        memref.store %r, %c[%i] : memref<?xf32>
      }
      gpu.return
    }
  }
}
