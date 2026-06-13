// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
// GENERATED MLIR-GPU 2D pool (avg K=11 S=11 P=0 D=1) from op_expr(bpd_avgpool2d) -> NVVM -> PTX -> P4.
// params: x,out (!llvm.ptr); NC,H,W,Hout,Wout (i64). 1 thread/output.
module attributes {gpu.container_module} {
  gpu.module @kernels {
    llvm.func @k_pool(%x: !llvm.ptr, %out: !llvm.ptr, %NC: i64, %H: i64, %W: i64, %Hout: i64, %Wout: i64) attributes {gpu.kernel, nvvm.kernel} {
      %bid = nvvm.read.ptx.sreg.ctaid.x : i32
      %bdim = nvvm.read.ptx.sreg.ntid.x : i32
      %tid = nvvm.read.ptx.sreg.tid.x : i32
      %t1 = llvm.mul %bid, %bdim : i32
      %g32 = llvm.add %t1, %tid : i32
      %idx = llvm.sext %g32 : i32 to i64
      %nho = llvm.mul %NC, %Hout : i64
      %total = llvm.mul %nho, %Wout : i64
      %ok = llvm.icmp "slt" %idx, %total : i64
      llvm.cond_br %ok, ^body, ^done
    ^body:
      %ow = llvm.srem %idx, %Wout : i64
      %t2 = llvm.sdiv %idx, %Wout : i64
      %oh = llvm.srem %t2, %Hout : i64
      %nc = llvm.sdiv %t2, %Hout : i64
      %St = llvm.mlir.constant(11 : i64) : i64
      %Pad = llvm.mlir.constant(0 : i64) : i64
      %Dil = llvm.mlir.constant(1 : i64) : i64
      %K = llvm.mlir.constant(11 : i64) : i64
      %hs0 = llvm.mul %oh, %St : i64
      %hs = llvm.sub %hs0, %Pad : i64
      %ws0 = llvm.mul %ow, %St : i64
      %ws = llvm.sub %ws0, %Pad : i64
      %z = llvm.mlir.constant(0 : i64) : i64
      %one = llvm.mlir.constant(1 : i64) : i64
      %init = llvm.mlir.constant(0.0 : f32) : f32
      %fz = llvm.mlir.constant(0.0 : f32) : f32
      %nc_h = llvm.mul %nc, %H : i64
      llvm.br ^khloop(%z, %init, %z : i64, f32, i64)
    ^khloop(%kh: i64, %acch: f32, %cnth: i64):
      %khok = llvm.icmp "slt" %kh, %K : i64
      llvm.cond_br %khok, ^khb, ^khd(%acch, %cnth : f32, i64)
    ^khb:
      %ih0 = llvm.mul %kh, %Dil : i64
      %ih = llvm.add %hs, %ih0 : i64
      llvm.br ^kwloop(%z, %acch, %cnth : i64, f32, i64)
    ^kwloop(%kw: i64, %acc: f32, %cnt: i64):
      %kwok = llvm.icmp "slt" %kw, %K : i64
      llvm.cond_br %kwok, ^kwb, ^kwd(%acc, %cnt : f32, i64)
    ^kwb:
      %iw0 = llvm.mul %kw, %Dil : i64
      %iw = llvm.add %ws, %iw0 : i64
      %ihp = llvm.icmp "sge" %ih, %z : i64
      %ihq = llvm.icmp "slt" %ih, %H : i64
      %iwp = llvm.icmp "sge" %iw, %z : i64
      %iwq = llvm.icmp "slt" %iw, %W : i64
      %a1 = llvm.and %ihp, %ihq : i1
      %a2 = llvm.and %iwp, %iwq : i1
      %inb = llvm.and %a1, %a2 : i1
      llvm.cond_br %inb, ^load, ^skip
    ^load:
      %r0 = llvm.add %nc_h, %ih : i64
      %r1 = llvm.mul %r0, %W : i64
      %off = llvm.add %r1, %iw : i64
      %xp = llvm.getelementptr %x[%off] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %v = llvm.load %xp : !llvm.ptr -> f32
      %nacc = llvm.fadd %acc, %v : f32
      %ncnt = llvm.add %cnt, %one : i64
      llvm.br ^kwnext(%nacc, %ncnt : f32, i64)
    ^skip:
      llvm.br ^kwnext(%acc, %cnt : f32, i64)
    ^kwnext(%acc2: f32, %cnt2: i64):
      %kwn = llvm.add %kw, %one : i64
      llvm.br ^kwloop(%kwn, %acc2, %cnt2 : i64, f32, i64)
    ^kwd(%accw: f32, %cntw: i64):
      %khn = llvm.add %kh, %one : i64
      llvm.br ^khloop(%khn, %accw, %cntw : i64, f32, i64)
    ^khd(%accf: f32, %cntf: i64):
      %cf = llvm.sitofp %cntf : i64 to f32
      %res = llvm.fdiv %accf, %cf : f32
      %op = llvm.getelementptr %out[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %res, %op : f32, !llvm.ptr
      llvm.br ^done
    ^done:
      llvm.return
    }
  }
}
