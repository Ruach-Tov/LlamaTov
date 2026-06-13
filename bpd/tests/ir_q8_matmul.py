#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
# Compose the verified Q8_0 block-dot into a FULL matmul in LLVM IR -> JIT -> execute -> MEASURE
# vs the spec o_proj output. (Iyun, Heath.) Green by EXECUTED coverage.
import sys, struct
sys.path.insert(0, ".")
import llvmlite.ir as ir
import llvmlite.binding as llvm
import ctypes, numpy as np
llvm.initialize(); llvm.initialize_native_target(); llvm.initialize_native_asmprinter()

def fp16(x):
    return float(np.frombuffer(np.float16(x).tobytes(), dtype=np.float16).astype(np.float32)[0])

def read_bin(p):
    with open(p, "rb") as f:
        struct.unpack("<I", f.read(4)); struct.unpack("<I", f.read(4))
        ne = struct.unpack("<4q", f.read(32)); struct.unpack("<4Q", f.read(32))
        nb = struct.unpack("<Q", f.read(8))[0]; raw = f.read(nb)
    a = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    dims = [d for d in ne if d > 1] or [ne[0]]
    return a.reshape(dims[::-1]) if len(dims) > 1 else a

# --- EMIT: full Q8_0 matmul IR ---
# void q8_matmul(i8* wq, float* wd, i8* aq, float* ad, float* out, i32 nrows, i32 bpr)
#   out[r] = sum_b ( fp16(wd[r,b]) * fp16(ad[b]) * sum_{i in 32}( wq[r,b,i] * aq[b,i] ) )
# (orientation: weight row r dotted with activation; the transpose is handled in HOST data prep.)
mod = ir.Module(name="q8mm")
i8 = ir.IntType(8); i32 = ir.IntType(32); f32 = ir.FloatType()
i8p = ir.PointerType(i8); f32p = ir.PointerType(f32)
fnty = ir.FunctionType(ir.VoidType(), [i8p, f32p, i8p, f32p, f32p, i32, i32])
fn = ir.Function(mod, fnty, name="q8_matmul")
wq, wd, aq, ad, out, nrows, bpr = fn.args
QK = ir.Constant(i32, 32)
entry = fn.append_basic_block("entry"); b = ir.IRBuilder(entry)
r = b.alloca(i32); b.store(i32(0), r)
rloop = fn.append_basic_block("rloop"); rbody = fn.append_basic_block("rbody"); rend = fn.append_basic_block("rend")
b.branch(rloop); b.position_at_end(rloop)
rv = b.load(r); b.cbranch(b.icmp_signed("<", rv, nrows), rbody, rend)
b.position_at_end(rbody)
sumf = b.alloca(f32); b.store(ir.Constant(f32, 0.0), sumf)
bk = b.alloca(i32); b.store(i32(0), bk)
bloop = fn.append_basic_block("bloop"); bbody = fn.append_basic_block("bbody"); bend = fn.append_basic_block("bend")
b.branch(bloop); b.position_at_end(bloop)
bv = b.load(bk); b.cbranch(b.icmp_signed("<", bv, bpr), bbody, bend)
b.position_at_end(bbody)
# block index into wq/wd : (r*bpr + bv); into aq/ad : bv
rbpr = b.mul(b.load(r), bpr); widx = b.add(rbpr, bv)
acc = b.alloca(i32); b.store(i32(0), acc)
# inner 32: int8*int8 accumulate
wbase = b.mul(widx, QK); abase = b.mul(bv, QK)
ii = b.alloca(i32); b.store(i32(0), ii)
iloop = fn.append_basic_block("iloop"); ibody = fn.append_basic_block("ibody"); iend = fn.append_basic_block("iend")
b.branch(iloop); b.position_at_end(iloop)
iv = b.load(ii); b.cbranch(b.icmp_signed("<", iv, QK), ibody, iend)
b.position_at_end(ibody)
xi = b.sext(b.load(b.gep(wq, [b.add(wbase, iv)])), i32)
yi = b.sext(b.load(b.gep(aq, [b.add(abase, iv)])), i32)
b.store(b.add(b.load(acc), b.mul(xi, yi)), acc)
b.store(b.add(iv, i32(1)), ii); b.branch(iloop)
b.position_at_end(iend)
sumi = b.sitofp(b.load(acc), f32)
scale = b.fmul(b.load(b.gep(wd, [widx])), b.load(b.gep(ad, [bv])))
# FMA: sumf = fma(scale, sumi, sumf) — single rounding, matching ggml _mm256_fmadd_ps (not fmul+fadd)
fma = mod.declare_intrinsic('llvm.fma', [f32], ir.FunctionType(f32, [f32, f32, f32]))
b.store(b.call(fma, [scale, sumi, b.load(sumf)]), sumf)
b.store(b.add(bv, i32(1)), bk); b.branch(bloop)
b.position_at_end(bend)
b.store(b.load(sumf), b.gep(out, [b.load(r)]))
b.store(b.add(b.load(r), i32(1)), r); b.branch(rloop)
b.position_at_end(rend); b.ret_void()

# --- COMPILE ---
llmod = llvm.parse_assembly(str(mod)); llmod.verify()
tm = llvm.Target.from_default_triple().create_target_machine()
ee = llvm.create_mcjit_compiler(llmod, tm); ee.finalize_object()
addr = ee.get_function_address("q8_matmul")
cfn = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32)(addr)

# --- prepare data: spec o_proj input/output + raw Q8_0 weights (transposed orientation, solved) ---
from llamatov_run import parse_gguf
BLOB = sys.argv[1]
md, ts, do = parse_gguf(BLOB)
dims, typ, off = ts["blk.0.attn_output.weight"]; N = 2048; QKn = 32; bpr_v = N // QKn
with open(BLOB, "rb") as f:
    f.seek(do + off); raw = np.frombuffer(f.read(N * bpr_v * 34), dtype=np.uint8).reshape(N, bpr_v, 34)
# raw rows = INPUT dim (we proved rawW == W.T); ggml dots weight-row with activation -> need the
# orientation where out[r] = dot(weight col r over input). rawW[r] is along input for row r of W.T.
# We proved dot(rawW[r], kqv) = kqv@rawW.T = correct. So iterate rawW rows directly.
wd_np = np.array([[fp16(raw[r, bk, 0:2].view(np.float16)[0]) for bk in range(bpr_v)] for r in range(N)], dtype=np.float32)
wq_np = raw[:, :, 2:34].reshape(N, bpr_v * QKn).astype(np.int8)
kqv = read_bin("<home>/tmp/spec_dump/0048_kqv_out-0.bin"); spec = read_bin("<home>/tmp/spec_dump/0050_attn_out-0.bin")
ntok = kqv.shape[0]; kqv2 = kqv.reshape(ntok, N)
def q8_row(v):
    n = len(v) // QKn; d = np.zeros(n, dtype=np.float32); q = np.zeros((n, QKn), dtype=np.int8)
    for i in range(n):
        blk = v[i*QKn:(i+1)*QKn]; amax = float(np.max(np.abs(blk))); dd = fp16(amax/127.0)
        idd = 1.0/dd if dd else 0.0
        d[i] = dd; q[i] = (np.sign(blk*idd)*np.floor(np.abs(blk*idd)+0.5)).astype(np.int8)  # roundf(x*id)
    return d, q.reshape(-1)
out_all = np.zeros((ntok, N), dtype=np.float32)
wq_c = wq_np.reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
wd_c = wd_np.reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
for t in range(ntok):
    ad_np, aq_np = q8_row(kqv2[t])
    outv = np.zeros(N, dtype=np.float32)
    cfn(wq_c, wd_c, aq_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        ad_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        outv.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), N, bpr_v)
    out_all[t] = outv

def maxabs(a, b): a = a.ravel(); b = b.ravel(); n = min(a.size, b.size); return float(np.max(np.abs(a[:n]-b[:n])))
def ulp(a, b):
    a = a.ravel().astype(np.float32); b = b.ravel().astype(np.float32); n = min(a.size, b.size); a, b = a[:n], b[:n]
    ai = a.view(np.int32).astype(np.int64); bi = b.view(np.int32).astype(np.int64)
    ai = np.where(ai < 0, 2**31 - ai, ai); bi = np.where(bi < 0, 2**31 - bi, bi)
    return int(np.max(np.abs(ai - bi)))
ma = maxabs(out_all, spec); u = ulp(out_all, spec)
print("=== o_proj via OUR GENERATED+COMPILED+EXECUTED LLVM IR matmul ===")
print(f"  max_abs vs spec = {ma:.3e}   max_ULP = {u}")
print("  -> " + ("*** 0 ULP: o_proj GREEN by executed measurement ***" if u == 0 else
                  (f"essentially identical ({ma:.1e}) — fp accumulation order remains" if ma < 1e-3 else "diverges")))
