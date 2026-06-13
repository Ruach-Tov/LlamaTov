# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Direct test of bpd_dequant_q8_0_cpu on a single block with a known subnormal F16 scale."""
import ctypes
import numpy as np
SO = "/tmp/bpd_test/build/bpd_cpu.so"
lib = ctypes.CDLL(SO)
lib.bpd_dequant_q8_0_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.bpd_dequant_q8_0_cpu.restype = None

# Construct one Q8_0 block: F16 scale = 0x00c2 (subnormal, expected F32 = 1.156e-5)
# 32 int8 quants = all 1
block = bytearray()
block.append(0xc2)
block.append(0x00)
block.extend([1] * 32)
block_np = np.array(list(block), dtype=np.uint8)
out = np.zeros(32, dtype=np.float32)

lib.bpd_dequant_q8_0_cpu(block_np.ctypes.data, out.ctypes.data, ctypes.c_int(1))
print(f"Input: F16 0x00c2 (subnormal), int8 = all 1s")
print(f"Expected: out[i] = 1.156e-5 for all i")
print(f"Our C output: {out[:5]}")
print()

# numpy reference
scale_f16 = np.frombuffer(bytes([0xc2, 0x00]), dtype=np.float16)[0]
scale_f32 = np.float32(scale_f16)
print(f"numpy: scale_f16 = {scale_f16}, scale_f32 = {scale_f32}")
ref = scale_f32 * np.ones(32, dtype=np.float32)
print(f"numpy expected: {ref[:5]}")
print()
print(f"diff (ours - numpy): {(out - ref)[:5]}")
