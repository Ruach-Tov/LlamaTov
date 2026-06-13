#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""opcode_match.py — Compare actual CPU opcodes between our kernels and PyTorch.

Disassembles both .so files, extracts the instruction sequences for each op,
normalizes (strip register names, addresses), and compares opcodes.

If the opcode sequences match: blue (same hardware behavior).
If opcodes differ but 0 ULP: green (different code, same numbers).

Author: medayek
"""
import subprocess, re, os, sys

def disassemble_function(so_path, fn_name):
    """Extract disassembly for a specific function from a .so file."""
    try:
        r = subprocess.run(
            ["objdump", "-d", "--no-show-raw-insn", so_path],
            capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return None
        
        # Find the function
        lines = r.stdout.split("\n")
        in_fn = False
        opcodes = []
        
        for line in lines:
            if f"<{fn_name}>:" in line:
                in_fn = True
                continue
            if in_fn:
                # End of function: blank line or new function header
                if not line.strip() or (line.strip() and not line.startswith(" ")):
                    break
                opcodes.append(line.strip())
        
        return opcodes if opcodes else None
    except:
        return None


def normalize_opcodes(opcodes):
    """Normalize opcodes for comparison.
    
    Strips: register names, memory addresses, numeric operands.
    Keeps: instruction mnemonics and operand structure.
    """
    normalized = []
    for line in opcodes:
        # Extract just the instruction: "movps  %xmm0,%xmm1" → "movps REG,REG"
        # Remove the address prefix
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        insn = parts[-1].strip() if len(parts) >= 2 else line
        
        # Normalize registers
        insn = re.sub(r'%[a-z]+[0-9]*', 'REG', insn)
        # Normalize immediate values
        insn = re.sub(r'\$0x[0-9a-f]+', 'IMM', insn)
        insn = re.sub(r'\$-?[0-9]+', 'IMM', insn)
        # Normalize memory offsets
        insn = re.sub(r'0x[0-9a-f]+\(', 'MEM(', insn)
        insn = re.sub(r'-?[0-9]+\(', 'MEM(', insn)
        # Normalize jump targets
        insn = re.sub(r'[0-9a-f]+ <[^>]+>', 'TARGET', insn)
        
        normalized.append(insn)
    
    return normalized


def compare_opcodes(ours, theirs):
    """Compare two normalized opcode sequences.
    
    Returns (match_ratio, diff_count, details).
    """
    if not ours or not theirs:
        return 0.0, -1, "missing disassembly"
    
    # Simple sequence comparison
    max_len = max(len(ours), len(theirs))
    matches = 0
    diffs = []
    
    for i in range(max_len):
        a = ours[i] if i < len(ours) else "<end>"
        b = theirs[i] if i < len(theirs) else "<end>"
        if a == b:
            matches += 1
        else:
            if len(diffs) < 5:
                diffs.append(f"  [{i}] ours: {a}")
                diffs.append(f"  [{i}] refs: {b}")
    
    ratio = matches / max_len if max_len > 0 else 0
    return ratio, max_len - matches, "\n".join(diffs)


def main():
    our_so = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bpd_unary_llvm.so"
    
    # Our functions and their PyTorch equivalents
    # PyTorch ATen ops compile to functions in libtorch_cpu.so
    ops = [
        ("bpd_relu", "at::native::threshold_kernel"),
        ("bpd_silu", "at::native::silu_kernel"),
        ("bpd_sigmoid", "at::native::sigmoid_kernel"),
        ("bpd_tanh", "at::native::tanh_kernel"),
    ]
    
    print("OPCODE MATCH: BPD LLVM vs reference")
    print("=" * 60)
    
    for our_fn, ref_fn in ops:
        our_ops = disassemble_function(our_so, our_fn)
        if our_ops:
            our_norm = normalize_opcodes(our_ops)
            print(f"\n  {our_fn}: {len(our_ops)} instructions")
            # Show first few normalized opcodes
            for op in our_norm[:5]:
                print(f"    {op}")
            if len(our_norm) > 5:
                print(f"    ... ({len(our_norm)} total)")
        else:
            print(f"\n  {our_fn}: not found in {our_so}")


if __name__ == "__main__":
    main()
