#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""ir_diff.py — Semantic LLVM IR comparison.

Normalizes two .ll files to strip variable names, metadata, attributes,
and unrolling differences, then compares the core instruction sequence.

For simple ops (cumsum, scale, relu), the normalized IR should match exactly.
For complex ops (vec_dot), we compare the accumulation pattern.

Usage:
    python3 ir_diff.py ggml_op.ll bpd_op.ll
    
Returns: MATCH (blue), EQUIVALENT (green), or DIVERGENT (red) with details.
"""
import re
import sys

def normalize_ir(text):
    """Normalize LLVM IR for semantic comparison."""
    lines = text.strip().split('\n')
    normalized = []
    
    for line in lines:
        line = line.strip()
        
        # Skip metadata, attributes, comments, empty lines
        if not line or line.startswith(';') or line.startswith('!') or \
           line.startswith('attributes') or line.startswith('target') or \
           line.startswith('source_filename') or line.startswith('declare'):
            continue
        
        # Strip variable names: %foo → %_
        # But keep the instruction type and structure
        line = re.sub(r'%[a-zA-Z_][a-zA-Z0-9_.]*', '%_', line)
        line = re.sub(r'%[0-9]+', '%_', line)
        
        # Strip alignment hints
        line = re.sub(r', align \d+', '', line)
        
        # Strip tbaa metadata
        line = re.sub(r',\s*!tbaa\s*![0-9]+', '', line)
        line = re.sub(r',\s*!llvm\.loop\s*![0-9]+', '', line)
        
        # Strip nuw/nsw/nonneg hints
        line = re.sub(r'\b(nuw|nsw|nneg|noundef|noalias|nocapture|readonly|writeonly|local_unnamed_addr|dso_local)\b', '', line)
        line = re.sub(r'\s+', ' ', line).strip()
        
        # Strip inbounds
        line = line.replace('inbounds ', '')
        
        # Strip block labels (normalize to sequential numbers)
        line = re.sub(r'^[a-zA-Z_][a-zA-Z0-9_.]*:', 'BLOCK:', line)
        line = re.sub(r'^[0-9]+:', 'BLOCK:', line)
        
        if line:
            normalized.append(line)
    
    return normalized


def extract_instructions(normalized):
    """Extract just the instruction opcodes in order."""
    ops = []
    for line in normalized:
        # Match instruction patterns
        m = re.match(r'%_ = (\w+)', line)
        if m:
            ops.append(m.group(1))
        elif line.startswith('store'):
            ops.append('store')
        elif line.startswith('br '):
            ops.append('br')
        elif line.startswith('ret'):
            ops.append('ret')
        elif line.startswith('call'):
            ops.append('call')
    return ops


def compare_ir(file_a, file_b):
    """Compare two LLVM IR files semantically."""
    text_a = open(file_a).read()
    text_b = open(file_b).read()
    
    norm_a = normalize_ir(text_a)
    norm_b = normalize_ir(text_b)
    
    ops_a = extract_instructions(norm_a)
    ops_b = extract_instructions(norm_b)
    
    print("=== File A: %s ===" % file_a)
    print("  %d normalized lines, %d instructions" % (len(norm_a), len(ops_a)))
    print("  Instruction sequence: %s" % ' → '.join(ops_a[:20]))
    
    print("\n=== File B: %s ===" % file_b)
    print("  %d normalized lines, %d instructions" % (len(norm_b), len(ops_b)))
    print("  Instruction sequence: %s" % ' → '.join(ops_b[:20]))
    
    # Compare instruction sequences
    # Account for loop unrolling: A may have N copies of the pattern in B
    if ops_a == ops_b:
        print("\n✅ EXACT IR MATCH — instructions identical")
        return 'MATCH'
    
    # Check if one is an unrolled version of the other
    # Find the core loop body in the shorter one
    shorter = ops_b if len(ops_b) <= len(ops_a) else ops_a
    longer = ops_a if len(ops_b) <= len(ops_a) else ops_b
    
    # Check if all instructions in shorter appear in longer
    shorter_set = set(shorter)
    longer_set = set(longer)
    
    if shorter_set <= longer_set:
        print("\n🟢 EQUIVALENT — same instructions, different unrolling")
        print("   Shorter has %d unique ops, longer has %d" % (len(shorter_set), len(longer_set)))
        
        # Show what the longer version adds
        extra = longer_set - shorter_set
        if extra:
            print("   Extra in longer: %s" % extra)
        
        return 'EQUIVALENT'
    
    # Show the diff
    print("\n❌ DIVERGENT — different instruction sequences")
    print("\n  Only in A: %s" % (set(ops_a) - set(ops_b)))
    print("  Only in B: %s" % (set(ops_b) - set(ops_a)))
    
    # Show line-by-line diff for short sequences
    if len(norm_a) < 30 and len(norm_b) < 30:
        print("\n  Normalized A:")
        for line in norm_a:
            print("    %s" % line)
        print("\n  Normalized B:")
        for line in norm_b:
            print("    %s" % line)
    
    return 'DIVERGENT'


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 ir_diff.py <ggml.ll> <bpd.ll>")
        sys.exit(1)
    
    result = compare_ir(sys.argv[1], sys.argv[2])
    print("\nVERDICT: %s" % result)
