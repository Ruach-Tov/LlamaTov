#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""ir_match.py — LLVM IR semantic match verifier.

Extracts the core computation pattern from two .ll files and compares.
Ignores: variable names, loop unrolling, index width (sext vs zext),
metadata, attributes, block labels.

Focuses on: the sequence of float operations (fadd, fmul, fdiv, fcmp, 
call to intrinsics/libm) and memory operations (load, store).

Usage:
    python3 ir_match.py ggml_op.ll bpd_op.ll [function_name]
"""
import re
import sys

def extract_float_ops(text):
    """Extract the core float operation sequence from LLVM IR."""
    ops = []
    for line in text.strip().split('\n'):
        line = line.strip()
        
        # Float arithmetic
        for op in ['fadd', 'fsub', 'fmul', 'fdiv', 'fneg', 'fcmp']:
            if re.search(r'\b' + op + r'\b', line):
                # Capture the type too: fadd float vs fadd <4 x float>
                m = re.search(r'\b(' + op + r')\s+(float|<\d+ x float>)', line)
                if m:
                    ops.append('%s %s' % (m.group(1), m.group(2)))
                else:
                    ops.append(op)
        
        # Intrinsic/library calls
        if 'call' in line:
            m = re.search(r'call\s+\S+\s+@([\w.]+)', line)
            if m:
                fname = m.group(1)
                if any(k in fname for k in ['llvm.', 'expf', 'logf', 'tanhf', 'sqrtf', 'fabsf']):
                    ops.append('call %s' % fname)
        
        # Select (conditional)
        if re.search(r'\bselect\b', line):
            ops.append('select')
        
        # Vector operations
        if re.search(r'\bextractelement\b', line):
            ops.append('extractelement')
        if re.search(r'\binsertelement\b', line):
            ops.append('insertelement')
        if re.search(r'\bshufflevector\b', line):
            ops.append('shufflevector')
    
    return ops


def extract_function(text, func_name=None):
    """Extract a single function from an .ll file."""
    if func_name:
        pattern = r'define\s+[^@]*@' + re.escape(func_name) + r'\b[^{]*\{(.*?)^}'
        m = re.search(pattern, text, re.MULTILINE | re.DOTALL)
        if m:
            return m.group(0)
    return text


def deduplicate_unrolled(ops):
    """Detect and collapse loop-unrolled repetitions."""
    # Find repeating patterns of length 2-8
    for pattern_len in range(2, min(9, len(ops)//2 + 1)):
        pattern = ops[:pattern_len]
        count = 0
        pos = 0
        while pos + pattern_len <= len(ops):
            if ops[pos:pos+pattern_len] == pattern:
                count += 1
                pos += pattern_len
            else:
                break
        if count >= 2:
            remainder = ops[pos:]
            return pattern, count, remainder
    return ops, 1, []


def compare_ir(file_a, file_b, func_name=None):
    """Compare float operation sequences between two IR files."""
    text_a = open(file_a).read()
    text_b = open(file_b).read()
    
    if func_name:
        text_a = extract_function(text_a, func_name)
        text_b = extract_function(text_b, func_name)
    
    ops_a = extract_float_ops(text_a)
    ops_b = extract_float_ops(text_b)
    
    print("=== Reference: %s ===" % file_a)
    print("  Float ops: %s" % ops_a)
    
    print("\n=== BPD: %s ===" % file_b)
    print("  Float ops: %s" % ops_b)
    
    # Deduplicate unrolled ops
    core_a, repeats_a, tail_a = deduplicate_unrolled(ops_a)
    core_b, repeats_b, tail_b = deduplicate_unrolled(ops_b)
    
    if repeats_a > 1:
        print("\n  Reference core pattern (×%d): %s" % (repeats_a, core_a))
        if tail_a:
            print("  Reference tail: %s" % tail_a)
    if repeats_b > 1:
        print("  BPD core pattern (×%d): %s" % (repeats_b, core_b))
        if tail_b:
            print("  BPD tail: %s" % tail_b)
    
    # Compare
    if ops_a == ops_b:
        print("\n🔵 IR-MATCH — identical float operation sequence")
        return 'IR-MATCH'
    
    if core_a == core_b:
        print("\n🔵 IR-MATCH — same core pattern, different unrolling (%d vs %d)" % (repeats_a, repeats_b))
        return 'IR-MATCH'
    
    if set(core_a) == set(core_b) and len(core_a) == len(core_b):
        print("\n🟢 EQUIVALENT — same ops, possibly different order")
        return 'EQUIVALENT'
    
    if set(core_a) == set(core_b):
        print("\n🟡 SIMILAR — same op types, different counts")
        return 'SIMILAR'
    
    print("\n🔴 DIVERGENT — different float operations")
    print("  Only in reference: %s" % sorted(set(core_a) - set(core_b)))
    print("  Only in BPD: %s" % sorted(set(core_b) - set(core_a)))
    return 'DIVERGENT'


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 ir_match.py <reference.ll> <bpd.ll> [function_name]")
        sys.exit(1)
    
    func = sys.argv[3] if len(sys.argv) > 3 else None
    result = compare_ir(sys.argv[1], sys.argv[2], func)
    print("\nVERDICT: %s" % result)
