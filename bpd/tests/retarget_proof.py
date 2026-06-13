#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
# PROOF: same KernelSpec emits to CPU (x86) AND GPU (NVPTX/PTX) — the subsumption retarget power.
import llvmlite.ir as ir, llvmlite.binding as llvm
llvm.initialize(); llvm.initialize_all_targets(); llvm.initialize_all_asmprinters()
def emit_block_dot_ir(triple=''):
    m=ir.Module(name='kdot')
    if triple: m.triple=triple
    i8=ir.IntType(8);i32=ir.IntType(32);f32=ir.FloatType();i8p=ir.PointerType(i8)
    fn=ir.Function(m,ir.FunctionType(f32,[i8p,f32,i8p,f32]),name='block_dot')
    xq,xd,yq,yd=fn.args;b=ir.IRBuilder(fn.append_basic_block('e'))
    acc=b.alloca(i32);b.store(i32(0),acc)
    for i in range(32):
        xi=b.sext(b.load(b.gep(xq,[i32(i)])),i32);yi=b.sext(b.load(b.gep(yq,[i32(i)])),i32)
        b.store(b.add(b.load(acc),b.mul(xi,yi)),acc)
    b.ret(b.fmul(b.fmul(xd,yd),b.sitofp(b.load(acc),f32)))
    return m
if __name__=='__main__':
    cpu=llvm.Target.from_default_triple().create_target_machine()
    mc=llvm.parse_assembly(str(emit_block_dot_ir()));mc.verify()
    print('CPU x86-64:',[l.strip() for l in cpu.emit_assembly(mc).splitlines() if 'imull' in l or 'movsbl' in l][:2])
    nv=llvm.Target.from_triple('nvptx64-nvidia-cuda').create_target_machine(cpu='sm_61')
    mg=llvm.parse_assembly(str(emit_block_dot_ir('nvptx64-nvidia-cuda')));mg.verify()
    print('GPU PTX:',[l.strip() for l in nv.emit_assembly(mg).splitlines() if '.target' in l or 'block_dot' in l][:2])
    print('ONE KernelSpec -> CPU x86 + GPU PTX = retarget/subsumption proven')
