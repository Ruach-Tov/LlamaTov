#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
# Matmul IR replicating ggml's EXACT AVX2 q8_0_q8_0 reduction order (Iyun, Heath/medayek mandate).
# Target = the ggml/llama.cpp dump (Ollama replication). numpy = corroboration only.
# Sequence (from ggml-cpu-quants.c AVX2 path):
#   acc[8] = 0
#   per block b: lane_sum[l] = sum of 4 int8 products (maddubs+madd, EXACT integer);
#                d = fp16(wd[b])*fp16(ad[b]);  acc[l] = fma(d, lane_sum[l], acc[l])   (8-lane FMA)
#   hsum_float_8: res[4] = acc[0:4] + acc[4:8]; res[0]+=res[2]; res[1]+=res[3]; ret res[0]+res[1]
import sys, struct, glob
sys.path.insert(0, ".")
import llvmlite.ir as ir, llvmlite.binding as llvm
import ctypes, numpy as np
llvm.initialize(); llvm.initialize_native_target(); llvm.initialize_native_asmprinter()

def fp16(x): return float(np.frombuffer(np.float16(x).tobytes(), dtype=np.float16).astype(np.float32)[0])
def read_bin(p):
    with open(p, "rb") as f:
        struct.unpack("<I", f.read(4)); struct.unpack("<I", f.read(4))
        ne = struct.unpack("<4q", f.read(32)); struct.unpack("<4Q", f.read(32))
        nb = struct.unpack("<Q", f.read(8))[0]; raw = f.read(nb)
    a = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    d = [x for x in ne if x > 1] or [ne[0]]
    return a.reshape(d[::-1]) if len(d) > 1 else a

# ── EMIT IR: float q8_matmul_row(i8* wq, f32* wd, i32* lanesums_scratch, ...) ──
# To keep IR clean, the host precomputes the per-block 8-lane INTEGER sums (exact, no rounding),
# and the IR does ONLY the fp-order-critical part: 8-lane FMA accumulate + exact hsum tree.
# (The integer dot is associative/exact; the fp accumulation order is what must match ggml.)
mod = ir.Module(name="q8mm_simd")
f32 = ir.FloatType(); i32 = ir.IntType(32); f32p = ir.PointerType(f32)
# q8_dot_blocks(f32* lane_sums[nb*8], f32* dscales[nb], i32 nb) -> f32  (one output element)
fn = ir.Function(mod, ir.FunctionType(f32, [f32p, f32p, i32]), name="q8_reduce")
lane_sums, dscales, nb = fn.args
fma = mod.declare_intrinsic("llvm.fma", [f32], ir.FunctionType(f32, [f32, f32, f32]))
b = ir.IRBuilder(fn.append_basic_block("entry"))
# 8 lane accumulators
acc = [b.alloca(f32, name=f"acc{l}") for l in range(8)]
for l in range(8): b.store(ir.Constant(f32, 0.0), acc[l])
bk = b.alloca(i32); b.store(i32(0), bk)
loop = fn.append_basic_block("loop"); body = fn.append_basic_block("body"); done = fn.append_basic_block("done")
b.branch(loop); b.position_at_end(loop)
bv = b.load(bk); b.cbranch(b.icmp_signed("<", bv, nb), body, done)
b.position_at_end(body)
d = b.load(b.gep(dscales, [bv]))                       # block scale d = fp16(wd)*fp16(ad)
base = b.mul(bv, i32(8))
for l in range(8):                                     # acc[l] = fma(d, lane_sum[b,l], acc[l])
    ls = b.load(b.gep(lane_sums, [b.add(base, i32(l))]))
    b.store(b.call(fma, [d, ls, b.load(acc[l])]), acc[l])
b.store(b.add(bv, i32(1)), bk); b.branch(loop)
b.position_at_end(done)
# hsum_float_8 EXACT: res[l] = acc[l] + acc[l+4]  (l=0..3); then res0+=res2; res1+=res3; ret res0+res1
A = [b.load(acc[l]) for l in range(8)]
res = [b.fadd(A[l], A[l + 4]) for l in range(4)]       # _mm_add_ps(hi,lo)
r02 = b.fadd(res[0], res[2])                            # movehl add: res[0]+=res[2]
r13 = b.fadd(res[1], res[3])                            #             res[1]+=res[3]
b.ret(b.fadd(r02, r13))                                 # movehdup add_ss: res0+res1
fn.attributes.add("noinline")

# ── COMPILE ──
llmod = llvm.parse_assembly(str(mod)); llmod.verify()
tm = llvm.Target.from_default_triple().create_target_machine()
ee = llvm.create_mcjit_compiler(llmod, tm); ee.finalize_object()
cfn = ctypes.CFUNCTYPE(ctypes.c_float, ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_float), ctypes.c_int32)(ee.get_function_address("q8_reduce"))

# ── DATA: o_proj from the CLEAN ggml dump ──
from llamatov_run import parse_gguf
BLOB = sys.argv[1]
md, ts, do = parse_gguf(BLOB); dims, typ, off = ts["blk.0.attn_output.weight"]; N = 2048; QK = 32; bpr = N // QK
with open(BLOB, "rb") as fh:
    fh.seek(do + off); raw = np.frombuffer(fh.read(N * bpr * 34), dtype=np.uint8).reshape(N, bpr, 34)
wd = np.array([[fp16(raw[r, bb, 0:2].view(np.float16)[0]) for bb in range(bpr)] for r in range(N)], dtype=np.float32)
wq = raw[:, :, 2:34].view(np.int8).reshape(N, bpr, QK)
D = "<home>/tmp/spec_dump_v2/"
def fnd(idx, nm): g = [x for x in glob.glob(D + f"{idx}_*{nm}*.bin") if "_src" not in x]; return read_bin(sorted(g)[0])
kqv = fnd("0048", "kqv_out-0").reshape(-1, N); spec = fnd("0050", "attn_out-0"); ntok = kqv.shape[0]
def roundf(x): return np.sign(x) * np.floor(np.abs(x) + 0.5)
def q8(v):
    n = len(v) // QK; dd = np.zeros(n, dtype=np.float32); q = np.zeros((n, QK), dtype=np.int8)
    for i in range(n):
        blk = v[i*QK:(i+1)*QK].astype(np.float32); amax = np.float32(np.max(np.abs(blk)))
        d = np.float32(amax/np.float32(127.0)); idd = np.float32(1.0/d) if d>0 else np.float32(0)  # id from FP32 d
        dd[i] = fp16(d); q[i] = roundf(blk*idd).astype(np.int8)   # store fp16(d), quants use fp32 id
    return dd, q
def lane8(wqr, aq):  # 8 lanes, 4 products each (maddubs/madd grouping: contiguous groups of 4)
    p = (wqr.astype(np.int32) * aq.astype(np.int32))  # 32 products
    return p.reshape(8, 4).sum(axis=1).astype(np.float32)  # 8 int sums (exact) -> fp32
out = np.zeros((ntok, N), dtype=np.float32)
for t in range(ntok):
    ad, aq = q8(kqv[t])
    for r in range(N):
        ls = np.zeros(bpr * 8, dtype=np.float32); ds = np.zeros(bpr, dtype=np.float32)
        for bb in range(bpr):
            ls[bb*8:bb*8+8] = lane8(wq[r, bb], aq[bb]); ds[bb] = np.float32(wd[r, bb] * ad[bb])
        out[t, r] = cfn(ls.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        ds.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), bpr)
def ma(a, b): a = a.ravel(); b = b.ravel(); n = min(a.size, b.size); return float(np.max(np.abs(a[:n]-b[:n])))
def ulp(a, b):
    a = a.ravel().astype(np.float32); b = b.ravel().astype(np.float32); n = min(a.size, b.size); a, b = a[:n], b[:n]
    ai = a.view(np.int32).astype(np.int64); bi = b.view(np.int32).astype(np.int64)
    ai = np.where(ai < 0, 2**31 - ai, ai); bi = np.where(bi < 0, 2**31 - bi, bi)
    return int(np.max(np.abs(ai - bi)))
m = ma(out, spec); u = ulp(out, spec)
print(f"=== o_proj via IR with ggml EXACT 8-lane-FMA + hsum-tree, vs CLEAN ggml dump ===")
print(f"  max_abs = {m:.3e}   max_ULP = {u}")
print(f"  -> " + ("*** 0 ULP: o_proj GREEN vs ggml/Ollama by executed coverage ***" if u == 0
                   else f"residual {m:.1e} (lane-grouping or scale-order still differs from ggml)"))
