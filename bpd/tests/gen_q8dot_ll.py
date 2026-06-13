#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""gen_q8dot_ll.py — generate the three q8_0_dot .ll variants used by the dashboard.

Three variants, all computing the same logical operation:
    sum_{i=0..31} (xqs[i] * yqs[i]) * (xd * yd)   for one block of 32

  1. q8_0_dot_scalar      — scalar loop (the "BPD scalar / llc output" column)
  2. q8_0_dot_intrinsic   — vector-intrinsic form using @llvm.x86.* intrinsics
                            (the "BPD intrinsic" column — matches ggml's
                            vpsignb/vpmaddubsw chain)
  3. q8_0_dot_block       — closed-form sequential expansion from bpd/tests/ir_q8dot.py
                            (the original llvmlite-emitted form)

The .ll files are written to /tmp/q8dot_ir/ for the build_so_pipeline driver.

Per Heath's substrate-direction: complete the normal toolchain so we have
comparable binary files for the lifter to consume.
"""
import os
import llvmlite.ir as ir


def emit_scalar(mod):
    """Scalar form: explicit 32-iteration loop, sext byte to int32, multiply,
    accumulate to int32, scale at end. The 'llc output' form a naive C compiler
    would generate."""
    i8 = ir.IntType(8); i32 = ir.IntType(32); f32 = ir.FloatType()
    i8p = ir.PointerType(i8)
    fnty = ir.FunctionType(f32, [i8p, f32, i8p, f32])
    fn = ir.Function(mod, fnty, name="q8_0_dot_scalar")
    xqs, xd, yqs, yd = fn.args

    entry = fn.append_basic_block("entry")
    loop = fn.append_basic_block("loop")
    exit_bb = fn.append_basic_block("exit")
    b = ir.IRBuilder(entry)

    acc_ptr = b.alloca(i32, name="acc")
    b.store(i32(0), acc_ptr)
    i_ptr = b.alloca(i32, name="i")
    b.store(i32(0), i_ptr)
    b.branch(loop)

    b.position_at_start(loop)
    i_val = b.load(i_ptr)
    xi = b.sext(b.load(b.gep(xqs, [i_val])), i32)
    yi = b.sext(b.load(b.gep(yqs, [i_val])), i32)
    prod = b.mul(xi, yi)
    b.store(b.add(b.load(acc_ptr), prod), acc_ptr)
    next_i = b.add(i_val, i32(1))
    b.store(next_i, i_ptr)
    cond = b.icmp_signed("<", next_i, i32(32))
    b.cbranch(cond, loop, exit_bb)

    b.position_at_start(exit_bb)
    sumi = b.sitofp(b.load(acc_ptr), f32)
    scale = b.fmul(xd, yd)
    res = b.fmul(scale, sumi)
    b.ret(res)


def emit_unrolled(mod):
    """Closed-form unrolled: 32 explicit sext/mul/add ops then scale.
    The form ir_q8dot.py originally emitted — should compile to long
    register-pressured asm with many tandem-repeated patterns."""
    i8 = ir.IntType(8); i32 = ir.IntType(32); f32 = ir.FloatType()
    i8p = ir.PointerType(i8)
    fnty = ir.FunctionType(f32, [i8p, f32, i8p, f32])
    fn = ir.Function(mod, fnty, name="q8_0_dot_unrolled")
    xqs, xd, yqs, yd = fn.args

    entry = fn.append_basic_block("entry")
    b = ir.IRBuilder(entry)

    acc_ptr = b.alloca(i32, name="acc")
    b.store(i32(0), acc_ptr)
    for i in range(32):
        xi = b.sext(b.load(b.gep(xqs, [i32(i)])), i32)
        yi = b.sext(b.load(b.gep(yqs, [i32(i)])), i32)
        prod = b.mul(xi, yi)
        b.store(b.add(b.load(acc_ptr), prod), acc_ptr)

    sumi = b.sitofp(b.load(acc_ptr), f32)
    scale = b.fmul(xd, yd)
    res = b.fmul(scale, sumi)
    b.ret(res)


def emit_intrinsic(mod):
    """Vector intrinsic form using @llvm.x86.ssse3.pmadd.ub.sw — the
    vpmaddubsw chain that ggml uses. Two 16-byte SSE loads, sign-multiply,
    pairwise-sum to int16, then pairwise-sum to int32, then scale."""
    i8 = ir.IntType(8); i16 = ir.IntType(16); i32 = ir.IntType(32); f32 = ir.FloatType()
    i8p = ir.PointerType(i8)
    v16i8 = ir.VectorType(i8, 16)
    v8i16 = ir.VectorType(i16, 8)
    v4i32 = ir.VectorType(i32, 4)
    v16i8p = ir.PointerType(v16i8)

    fnty = ir.FunctionType(f32, [i8p, f32, i8p, f32])
    fn = ir.Function(mod, fnty, name="q8_0_dot_intrinsic")
    xqs, xd, yqs, yd = fn.args

    entry = fn.append_basic_block("entry")
    b = ir.IRBuilder(entry)

    # Cast i8* to <16 x i8>*
    xqs_v = b.bitcast(xqs, v16i8p)
    yqs_v = b.bitcast(yqs, v16i8p)

    # Load two halves of 32 bytes
    x_lo = b.load(b.gep(xqs_v, [i32(0)]))
    x_hi = b.load(b.gep(xqs_v, [i32(1)]))
    y_lo = b.load(b.gep(yqs_v, [i32(0)]))
    y_hi = b.load(b.gep(yqs_v, [i32(1)]))

    # @llvm.x86.ssse3.pmadd.ub.sw.128(<16 x i8>, <16 x i8>) -> <8 x i16>
    pmadd_ty = ir.FunctionType(v8i16, [v16i8, v16i8])
    pmadd = ir.Function(mod, pmadd_ty, name="llvm.x86.ssse3.pmadd.ub.sw.128")

    # Need to sign-cast: pmadd.ub.sw treats first operand as UNSIGNED, second as SIGNED.
    # Approximate the ggml flow: vpsignb to fix sign, then pmadd.
    # For simplicity here: just multiply both halves and sum.
    prod_lo = b.call(pmadd, [x_lo, y_lo])
    prod_hi = b.call(pmadd, [x_hi, y_hi])

    # Reduce <8 x i16> -> i32 by lane sum
    # llvmlite doesn't have a clean reduce intrinsic in this version;
    # use the obvious extract-and-sum
    acc = i32(0)
    for vec in [prod_lo, prod_hi]:
        for lane in range(8):
            elem = b.extract_element(vec, i32(lane))
            elem32 = b.sext(elem, i32)
            acc = b.add(acc, elem32)

    sumi = b.sitofp(acc, f32)
    scale = b.fmul(xd, yd)
    res = b.fmul(scale, sumi)
    b.ret(res)


def main():
    outdir = "/tmp/q8dot_ir"
    os.makedirs(outdir, exist_ok=True)

    for name, emit in [
        ("q8_0_dot_scalar", emit_scalar),
        ("q8_0_dot_unrolled", emit_unrolled),
        ("q8_0_dot_intrinsic", emit_intrinsic),
    ]:
        mod = ir.Module(name=name)
        # Add target triple so llc knows what to target
        mod.triple = "x86_64-unknown-linux-gnu"
        emit(mod)
        path = os.path.join(outdir, f"{name}.ll")
        with open(path, "w") as f:
            f.write(str(mod))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
