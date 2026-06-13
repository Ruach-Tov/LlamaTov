#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
# EMIT LLVM IR for the Q8_0 dot (from verified closed-form spec) -> JIT compile -> execute -> measure
# (Iyun, Heath). The compiler pipeline: our GENERATED IR runs and is measured vs the spec.
import llvmlite.ir as ir
import llvmlite.binding as llvm
import ctypes, numpy as np
llvm.initialize(); llvm.initialize_native_target(); llvm.initialize_native_asmprinter()

# --- BUILD the IR: float q8_dot_block(i8* xqs, float xd, i8* yqs, float yd) for one block of 32 ---
# closed form: (xd*yd) * sum_{i=0..31}(xqs[i]*yqs[i])  ; int32 accumulate then fp scale
mod = ir.Module(name="q8dot")
i8 = ir.IntType(8); i32 = ir.IntType(32); f32 = ir.FloatType(); i8p = ir.PointerType(i8)
fnty = ir.FunctionType(f32, [i8p, f32, i8p, f32])
fn = ir.Function(mod, fnty, name="q8_dot_block")
xqs, xd, yqs, yd = fn.args
bb = fn.append_basic_block("entry"); b = ir.IRBuilder(bb)
acc = b.alloca(i32, name="acc"); b.store(i32(0), acc)
for i in range(32):
    xi = b.sext(b.load(b.gep(xqs, [i32(i)])), i32)
    yi = b.sext(b.load(b.gep(yqs, [i32(i)])), i32)
    prod = b.mul(xi, yi)
    b.store(b.add(b.load(acc), prod), acc)
sumi = b.sitofp(b.load(acc), f32)
scale = b.fmul(xd, yd)
res = b.fmul(scale, sumi)
b.ret(res)

print("=== EMITTED LLVM IR ===")
print(str(mod))

# --- COMPILE (JIT) ---
llmod = llvm.parse_assembly(str(mod)); llmod.verify()
target = llvm.Target.from_default_triple(); tm = target.create_target_machine()
ee = llvm.create_mcjit_compiler(llmod, tm); ee.finalize_object()
addr = ee.get_function_address("q8_dot_block")
cfn = ctypes.CFUNCTYPE(ctypes.c_float, ctypes.POINTER(ctypes.c_int8), ctypes.c_float,
                       ctypes.POINTER(ctypes.c_int8), ctypes.c_float)(addr)

# --- EXECUTE on a known block + MEASURE vs hand-computed closed form ---
def fp16(x):
    return float(np.frombuffer(np.float16(x).tobytes(), dtype=np.float16).astype(np.float32)[0])
x = np.array([(i - 16) * 0.1 for i in range(32)], dtype=np.float32)
y = np.array([(15 - i) * 0.07 for i in range(32)], dtype=np.float32)
def q8(v):
    amax = float(np.max(np.abs(v))); d = fp16(amax / 127.0)
    q = np.round(v / d).astype(np.int8); return d, q
xd_v, xq = q8(x); yd_v, yq = q8(y)
xqc = (ctypes.c_int8 * 32)(*xq.tolist()); yqc = (ctypes.c_int8 * 32)(*yq.tolist())
ir_result = cfn(xqc, ctypes.c_float(xd_v), yqc, ctypes.c_float(yd_v))
cf = xd_v * yd_v * float(np.dot(xq.astype(np.int32), yq.astype(np.int32)))
print("=== LLVM-IR q8_dot (EMITTED -> JIT-COMPILED -> EXECUTED) ===")
print(f"  IR-generated result   = {ir_result:.8f}")
print(f"  closed-form reference = {cf:.8f}")
ulp = abs(int(np.float32(ir_result).view(np.int32)) - int(np.float32(cf).view(np.int32)))
ok = (np.float32(ir_result) == np.float32(cf))
print(f"  max_abs={abs(ir_result-cf):.2e}  ULP={ulp}  -> " +
      ("*** 0 ULP: our generated IR matches the spec, MEASURED ***" if ok else "diverges"))
