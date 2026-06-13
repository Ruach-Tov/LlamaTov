#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""asm_jit_lifter.py — Lift JIT binary into jit_insn/4 and jit_const/3 Prolog facts.

Takes a raw binary (JIT code dump) and produces the declarative facts
that ondnn_kernel_emitter.pl can walk to generate 0-ULP LLVM IR.

The lifter:
  1. Disassembles with objdump
  2. Tracks register assignments (data flow)
  3. Identifies constant table references (r9+offset patterns)
  4. Classifies each instruction into a semantic operation
  5. Assigns SSA names to each destination
  6. Extracts f32 constants from the data section

Usage:
  python3 asm_jit_lifter.py /tmp/ondnn_jit.bin > ondnn_gelu_facts.pl

Author: mavchin (2026-06-01)
"""

import subprocess
import re
import struct
import sys
from collections import OrderedDict

def disassemble(binpath):
    """Disassemble a raw binary with objdump."""
    result = subprocess.run(
        ['objdump', '-D', '-b', 'binary', '-m', 'i386:x86-64', '-M', 'intel', binpath],
        capture_output=True, text=True, timeout=10
    )
    insns = []
    for line in result.stdout.splitlines():
        m = re.match(r'\s+([0-9a-f]+):\s+([0-9a-f ]+?)\s{2,}(\S+)\s*(.*)', line)
        if m:
            offset = int(m.group(1), 16)
            raw_bytes = m.group(2).strip()
            mnem = m.group(3)
            operands = m.group(4).strip()
            insns.append({
                'offset': offset,
                'mnem': mnem,
                'operands': operands,
                'raw': f'{mnem} {operands}'.strip()
            })
    return insns

def extract_constants(binpath, table_base=0x400):
    """Extract f32 constants from the data section."""
    with open(binpath, 'rb') as f:
        data = f.read()
    
    constants = OrderedDict()
    # Scan in 32-byte steps (8 x f32 = one AVX broadcast)
    offset = table_base
    while offset + 32 <= len(data):
        # Read 8 floats
        vals = struct.unpack('<8f', data[offset:offset+32])
        # Check if all 8 are the same (broadcast constant)
        if all(v == vals[0] for v in vals):
            val = vals[0]
            import math
            if not math.isnan(val) and not math.isinf(val):
                constants[offset - table_base] = val
        offset += 32
    return constants

def parse_table_ref(operand_str):
    """Extract r9+0xNNN table offset from an operand."""
    m = re.search(r'\[r9\+0x([0-9a-f]+)\]', operand_str)
    if m:
        return int(m.group(1), 16)
    return None

def parse_ymm(op):
    """Extract ymm register number."""
    m = re.match(r'ymm(\d+)', op)
    return int(m.group(1)) if m else None

# Known constant names by table offset
CONST_NAMES = {
    0x080: 'half', 0x0a0: 'one', 0x0c0: 'two',
    0x120: 'ln2', 0x140: 'abs_mask', 0x160: 'sign_mask',
    0x180: 'bias_127', 0x1a0: 'log2e', 0x1c0: 'exp_hi', 0x1e0: 'exp_lo',
    0x200: 'exp_c1', 0x220: 'exp_c2', 0x240: 'exp_c3', 0x260: 'exp_c4', 0x280: 'exp_c5',
    0x2a0: 'erf_p', 0x2c0: 'sqrt2_inv',
    0x300: 'erf_a1', 0x320: 'erf_a2', 0x340: 'erf_a3', 0x360: 'erf_a4', 0x380: 'erf_a5',
}

# Mnemonic to semantic operation mapping
MNEM_MAP = {
    'vmulps': 'fmul',
    'vaddps': 'fadd',
    'vsubps': 'fsub',
    'vdivps': 'fdiv',
    'vmovups': 'mov',
    'vmovaps': 'mov',
    'vandps': 'and',
    'vandnps': 'andnot',
    'vxorps': 'xor',
    'vminps': 'fmin',
    'vmaxps': 'fmax',
    'vroundps': 'floor',  # mode-dependent, usually floor for exp
    'vcvtps2dq': 'cvt_f2i',
    'vcmpltps': 'fcmplt',
    'vblendvps': 'blend',
    'vextractf128': 'extract128',
    'vinsertf128': 'insert128',
    'vpaddd': 'iadd',
    'vpslld': 'ishl',
}

def lift_instructions(insns, code_start=0x38, code_end=0x1f0):
    """Lift instructions into semantic facts with SSA naming."""
    # Register state: ymm_N -> SSA name
    reg_state = {}
    ssa_counter = {}
    facts = []
    
    def fresh_name(base):
        """Generate a unique SSA name."""
        if base not in ssa_counter:
            ssa_counter[base] = 0
            return base
        ssa_counter[base] += 1
        return f"{base}_{ssa_counter[base]}"
    
    def reg_name(ymm_str):
        """Get current SSA name for a register."""
        return reg_state.get(ymm_str, ymm_str)
    
    def set_reg(ymm_str, ssa_name):
        """Assign new SSA name to register."""
        reg_state[ymm_str] = ssa_name
    
    for insn in insns:
        off = insn['offset']
        if off < code_start or off >= code_end:
            continue
        
        mnem = insn['mnem']
        ops_str = insn['operands']
        ops = [o.strip() for o in ops_str.split(',')]
        
        sem_op = MNEM_MAP.get(mnem)
        if sem_op is None:
            facts.append(f"%% 0x{off:03x}: {insn['raw']}  (not lifted)")
            continue
        
        # Parse operands
        table_ref = parse_table_ref(ops_str)
        dst_ymm = ops[0] if ops else None
        
        if sem_op in ('fmul', 'fadd', 'fsub', 'fdiv'):
            src1 = reg_name(ops[1]) if len(ops) > 1 else '?'
            if table_ref is not None:
                src2 = CONST_NAMES.get(table_ref, f'table_0x{table_ref:03x}')
            elif len(ops) > 2:
                src2 = reg_name(ops[2])
            else:
                src2 = '?'
            
            # Determine SSA name based on context
            dst_name = fresh_name(f'v{off:03x}')
            set_reg(dst_ymm, dst_name)
            
            op_sym = {'fmul': '*', 'fadd': '+', 'fsub': '-', 'fdiv': '/'}[sem_op]
            facts.append(f"jit_insn(0x{off:03x}, {sem_op}, {dst_name}, {src1} {op_sym} {src2}).")
        
        elif sem_op == 'mov':
            if table_ref is not None:
                const_name = CONST_NAMES.get(table_ref, f'table_0x{table_ref:03x}')
                dst_name = fresh_name(f'v{off:03x}')
                set_reg(dst_ymm, dst_name)
                facts.append(f"jit_insn(0x{off:03x}, load, {dst_name}, {const_name}).")
            elif '[' in ops_str and 'rax' in ops_str:
                dst_name = fresh_name('x')
                set_reg(dst_ymm, dst_name)
                facts.append(f"jit_insn(0x{off:03x}, load, {dst_name}, '[input]').")
            else:
                src = reg_name(ops[1]) if len(ops) > 1 else '?'
                dst_name = fresh_name(f'v{off:03x}')
                set_reg(dst_ymm, dst_name)
                facts.append(f"jit_insn(0x{off:03x}, mov, {dst_name}, {src}).")
        
        elif sem_op == 'and':
            src1 = reg_name(ops[1]) if len(ops) > 1 else '?'
            if table_ref == 0x140:  # abs_mask
                dst_name = fresh_name(f'v{off:03x}')
                set_reg(dst_ymm, dst_name)
                facts.append(f"jit_insn(0x{off:03x}, and_abs, {dst_name}, {src1} * abs_mask).")
            elif table_ref == 0x160:  # sign_mask
                dst_name = fresh_name(f'v{off:03x}')
                set_reg(dst_ymm, dst_name)
                facts.append(f"jit_insn(0x{off:03x}, and_sign, {dst_name}, {src1} * sign_mask).")
            else:
                dst_name = fresh_name(f'v{off:03x}')
                set_reg(dst_ymm, dst_name)
                facts.append(f"jit_insn(0x{off:03x}, and, {dst_name}, {src1}).")
        
        elif sem_op == 'xor':
            src1 = reg_name(ops[1]) if len(ops) > 1 else '?'
            if table_ref == 0x160:  # sign_mask -> negate
                dst_name = fresh_name(f'v{off:03x}')
                set_reg(dst_ymm, dst_name)
                facts.append(f"jit_insn(0x{off:03x}, xor_neg, {dst_name}, {src1} * sign_mask).")
            else:
                # xor with another register (sign application)
                src2 = reg_name(ops[2]) if len(ops) > 2 else '?'
                dst_name = fresh_name(f'v{off:03x}')
                set_reg(dst_ymm, dst_name)
                facts.append(f"jit_insn(0x{off:03x}, xor_sign, {dst_name}, {src1} * {src2}).")
        
        elif sem_op == 'fmin':
            src = reg_name(ops[1]) if len(ops) > 1 else '?'
            const = CONST_NAMES.get(table_ref, 'unknown') if table_ref else '?'
            dst_name = fresh_name(f'v{off:03x}')
            set_reg(dst_ymm, dst_name)
            facts.append(f"jit_insn(0x{off:03x}, fmin, {dst_name}, {src} * {const}).")
        
        elif sem_op == 'fmax':
            src = reg_name(ops[1]) if len(ops) > 1 else '?'
            const = CONST_NAMES.get(table_ref, 'unknown') if table_ref else '?'
            dst_name = fresh_name(f'v{off:03x}')
            set_reg(dst_ymm, dst_name)
            facts.append(f"jit_insn(0x{off:03x}, fmax, {dst_name}, {src} * {const}).")
        
        elif sem_op == 'floor':
            src = reg_name(ops[1]) if len(ops) > 1 else '?'
            dst_name = fresh_name(f'v{off:03x}')
            set_reg(dst_ymm, dst_name)
            facts.append(f"jit_insn(0x{off:03x}, floor, {dst_name}, {src}).")
        
        elif sem_op == 'cvt_f2i':
            src = reg_name(ops[1]) if len(ops) > 1 else '?'
            dst_name = fresh_name(f'vi{off:03x}')
            set_reg(dst_ymm, dst_name)
            facts.append(f"jit_insn(0x{off:03x}, cvt, {dst_name}, {src}).")
        
        elif sem_op in ('iadd', 'ishl'):
            src = reg_name(ops[1]) if len(ops) > 1 else '?'
            if table_ref:
                const = CONST_NAMES.get(table_ref, f'table_0x{table_ref:03x}')
            elif len(ops) > 2:
                const = ops[2].strip()
            else:
                const = '?'
            dst_name = fresh_name(f'vi{off:03x}')
            set_reg(dst_ymm, dst_name)
            if sem_op == 'iadd':
                facts.append(f"jit_insn(0x{off:03x}, iadd, {dst_name}, {src} + {const}).")
            else:
                facts.append(f"jit_insn(0x{off:03x}, ishl, {dst_name}, {src} * {const}).")
        
        elif sem_op in ('extract128', 'insert128'):
            facts.append(f"%% 0x{off:03x}: {insn['raw']}  (128-bit shuffle, LLVM handles)")
        
        elif sem_op == 'blend':
            facts.append(f"%% 0x{off:03x}: {insn['raw']}  (blend for underflow)")
        
        else:
            facts.append(f"%% 0x{off:03x}: {insn['raw']}  (unhandled: {sem_op})")
    
    return facts

def emit_const_facts(constants):
    """Emit jit_const/3 facts from extracted constants."""
    facts = []
    for offset, val in sorted(constants.items()):
        name = CONST_NAMES.get(offset, f'c_0x{offset:03x}')
        # Convert f32 to f64 hex for LLVM IR
        f32_bytes = struct.pack('<f', val)
        f64_val = struct.unpack('<f', f32_bytes)[0]
        f64_hex = struct.unpack('<Q', struct.pack('<d', float(f64_val)))[0]
        facts.append(f"jit_const(0x{offset:03x}, {name}, '0x{f64_hex:016X}').  %% {val}")
    return facts

def main():
    if len(sys.argv) < 2:
        print("Usage: asm_jit_lifter.py <binary> [code_start] [code_end]", file=sys.stderr)
        sys.exit(1)
    
    binpath = sys.argv[1]
    code_start = int(sys.argv[2], 0) if len(sys.argv) > 2 else 0x38
    code_end = int(sys.argv[3], 0) if len(sys.argv) > 3 else 0x1f0
    
    insns = disassemble(binpath)
    constants = extract_constants(binpath)
    
    print(f"%% Auto-lifted from {binpath} by asm_jit_lifter.py")
    print(f"%% Code range: 0x{code_start:x}-0x{code_end:x}")
    print(f"%% {len(insns)} instructions disassembled")
    print()
    print(":- module(lifted_jit_facts, [jit_insn/4, jit_const/3]).")
    print()
    
    print("%% === Constants (from data section) ===")
    for fact in emit_const_facts(constants):
        print(fact)
    print()
    
    print("%% === Instructions (lifted from disassembly) ===")
    lifted = lift_instructions(insns, code_start, code_end)
    for fact in lifted:
        print(fact)

if __name__ == '__main__':
    main()
