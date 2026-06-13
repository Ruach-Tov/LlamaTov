#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""test_dequant_subnormal.py — Verify Q8_0 dequant with subnormal F16 scales.

Q8_0 blocks have a 2-byte F16 scale followed by 32 int8 quants.
When the F16 scale is subnormal (exponent=0, mantissa!=0), the
f16_to_f32 conversion must handle it correctly.

This test constructs synthetic Q8_0 blocks with known subnormal
scales and verifies the dequantized output matches the expected
F32 values exactly.

Regression: catches f16_to_f32 bugs that only manifest for
subnormal values (rare in practice but present in real models).

Extracted from check_dequant_subnormal.py.
Author: medayek
"""
import sys, ctypes, numpy as np, struct


def main():
    so_path = sys.argv[1] if len(sys.argv) > 1 else "build/bpd_cpu.so"
    lib = ctypes.CDLL(so_path)
    
    if not hasattr(lib, 'bpd_dequant_q8_0_cpu'):
        print("SKIP: bpd_dequant_q8_0_cpu not in .so")
        return
    
    lib.bpd_dequant_q8_0_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.bpd_dequant_q8_0_cpu.restype = None
    
    # Subnormal F16 values to test
    # F16 subnormal: exponent=0, mantissa!=0 → value = mantissa * 2^-24
    subnormal_cases = [
        (0x0001, "smallest subnormal"),      # 5.96e-8
        (0x0002, "2nd smallest"),             # 1.19e-7
        (0x00c2, "typical subnormal"),        # 1.156e-5
        (0x03FF, "largest subnormal"),        # 6.098e-5
        (0x0400, "smallest normal"),          # 6.104e-5
        (0x3C00, "1.0 (normal)"),             # 1.0
        (0x7BFF, "max finite f16"),           # 65504.0
    ]
    
    all_pass = True
    for f16_bits, desc in subnormal_cases:
        # Build a Q8_0 block: 2-byte F16 scale + 32 int8 quants (all = 1)
        block = bytearray()
        block.append(f16_bits & 0xFF)
        block.append((f16_bits >> 8) & 0xFF)
        block.extend([1] * 32)
        
        block_np = np.array(list(block), dtype=np.uint8)
        out = np.zeros(32, dtype=np.float32)
        
        lib.bpd_dequant_q8_0_cpu(block_np.ctypes.data, out.ctypes.data, ctypes.c_int(1))
        
        # Reference: numpy f16→f32
        scale_f16 = np.frombuffer(bytes([f16_bits & 0xFF, (f16_bits >> 8) & 0xFF]), dtype=np.float16)[0]
        expected = np.float32(scale_f16)
        
        # All 32 outputs should be scale * 1 = scale
        our_val = out[0]
        
        our_bits = struct.unpack('<I', struct.pack('<f', float(our_val)))[0]
        ref_bits = struct.unpack('<I', struct.pack('<f', float(expected)))[0]
        
        if our_bits == ref_bits:
            print(f"  PASS  0x{f16_bits:04x} ({desc:25s}): {float(our_val):.6e}")
        else:
            ulp = abs(our_bits - ref_bits)
            print(f"  FAIL  0x{f16_bits:04x} ({desc:25s}): ours={float(our_val):.6e} ref={float(expected):.6e} ULP={ulp}")
            all_pass = False
    
    print()
    if all_pass:
        print("PASS  dequant_subnormal: all F16 scales handled correctly")
    else:
        print("FAIL  dequant_subnormal: subnormal handling diverges")
        sys.exit(1)


if __name__ == "__main__":
    main()
