// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
// GENERATED MLIR-GPU 2D conv (St=1 Pad=0 Dil=1 G=1) from op_expr(bpd_conv2d) -> NVVM -> PTX -> P4.
module attributes {gpu.container_module} {
  gpu.module @kernels {
    llvm.func @k_conv(%x: !llvm.ptr, %w: !llvm.ptr, %out: !llvm.ptr, %N: i64, %Cin: i64, %H: i64, %W: i64, %Cout: i64, %KH: i64, %KW: i64, %Hout: i64, %Wout: i64) attributes {gpu.kernel, nvvm.kernel} {
      %bid = nvvm.read.ptx.sreg.ctaid.x : i32
      %bdim = nvvm.read.ptx.sreg.ntid.x : i32
      %tid = nvvm.read.ptx.sreg.tid.x : i32
      %m1 = llvm.mul %bid, %bdim : i32
      %g32 = llvm.add %m1, %tid : i32
      %idx = llvm.sext %g32 : i32 to i64
      %nc = llvm.mul %N, %Cout : i64
      %nch = llvm.mul %nc, %Hout : i64
      %total = llvm.mul %nch, %Wout : i64
      %ok = llvm.icmp "slt" %idx, %total : i64
      llvm.cond_br %ok, ^body, ^done
    ^body:
      %ow = llvm.srem %idx, %Wout : i64
      %d1 = llvm.sdiv %idx, %Wout : i64
      %oh = llvm.srem %d1, %Hout : i64
      %d2 = llvm.sdiv %d1, %Hout : i64
      %oc = llvm.srem %d2, %Cout : i64
      %n = llvm.sdiv %d2, %Cout : i64
      %St = llvm.mlir.constant(1 : i64) : i64
      %Pad = llvm.mlir.constant(0 : i64) : i64
      %Dil = llvm.mlir.constant(1 : i64) : i64
      %z = llvm.mlir.constant(0 : i64) : i64
      %one = llvm.mlir.constant(1 : i64) : i64
      %i0 = llvm.mlir.constant(0.0 : f32) : f32
      %hs0 = llvm.mul %oh, %St : i64
      %hs = llvm.sub %hs0, %Pad : i64
      %ws0 = llvm.mul %ow, %St : i64
      %ws = llvm.sub %ws0, %Pad : i64
      llvm.br ^icloop(%z, %i0 : i64, f32)
    ^icloop(%ic: i64, %acc_i: f32):
      %icok = llvm.icmp "slt" %ic, %Cin : i64
      llvm.cond_br %icok, ^icb, ^icd(%acc_i : f32)
    ^icb:
      llvm.br ^khloop(%z, %acc_i : i64, f32)
    ^khloop(%kh: i64, %acc_h: f32):
      %khok = llvm.icmp "slt" %kh, %KH : i64
      llvm.cond_br %khok, ^khb, ^khd(%acc_h : f32)
    ^khb:
      %ih0 = llvm.mul %kh, %Dil : i64
      %ih = llvm.add %hs, %ih0 : i64
      llvm.br ^kwloop(%z, %acc_h : i64, f32)
    ^kwloop(%kw: i64, %acc: f32):
      %kwok = llvm.icmp "slt" %kw, %KW : i64
      llvm.cond_br %kwok, ^kwb, ^kwd(%acc : f32)
    ^kwb:
      %iw0 = llvm.mul %kw, %Dil : i64
      %iw = llvm.add %ws, %iw0 : i64
      %p1 = llvm.icmp "sge" %ih, %z : i64
      %p2 = llvm.icmp "slt" %ih, %H : i64
      %p3 = llvm.icmp "sge" %iw, %z : i64
      %p4 = llvm.icmp "slt" %iw, %W : i64
      %b1 = llvm.and %p1, %p2 : i1
      %b2 = llvm.and %p3, %p4 : i1
      %inb = llvm.and %b1, %b2 : i1
      llvm.cond_br %inb, ^load, ^skip
    ^load:
      %xa = llvm.mul %n, %Cin : i64
      %xb = llvm.add %xa, %ic : i64
      %xc = llvm.mul %xb, %H : i64
      %xd = llvm.add %xc, %ih : i64
      %xe = llvm.mul %xd, %W : i64
      %xoff = llvm.add %xe, %iw : i64
      %xp = llvm.getelementptr %x[%xoff] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %xv = llvm.load %xp : !llvm.ptr -> f32
      %wa = llvm.mul %oc, %Cin : i64
      %wb = llvm.add %wa, %ic : i64
      %wc = llvm.mul %wb, %KH : i64
      %wd = llvm.add %wc, %kh : i64
      %we = llvm.mul %wd, %KW : i64
      %woff = llvm.add %we, %kw : i64
      %wp = llvm.getelementptr %w[%woff] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %wv = llvm.load %wp : !llvm.ptr -> f32
      %prod = llvm.fmul %xv, %wv : f32
      %nacc = llvm.fadd %acc, %prod : f32
      llvm.br ^kwnext(%nacc : f32)
    ^skip:
      llvm.br ^kwnext(%acc : f32)
    ^kwnext(%accw: f32):
      %kwn = llvm.add %kw, %one : i64
      llvm.br ^kwloop(%kwn, %accw : i64, f32)
    ^kwd(%acckw: f32):
      %khn = llvm.add %kh, %one : i64
      llvm.br ^khloop(%khn, %acckw : i64, f32)
    ^khd(%acckh: f32):
      %icn = llvm.add %ic, %one : i64
      llvm.br ^icloop(%icn, %acckh : i64, f32)
    ^icd(%accf: f32):
      %op = llvm.getelementptr %out[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %accf, %op : f32, !llvm.ptr
      llvm.br ^done
    ^done:
      llvm.return
    }
  }
}
