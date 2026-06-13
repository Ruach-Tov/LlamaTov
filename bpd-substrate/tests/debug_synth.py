# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Debug: what F16 scale is in block 0 of attn_q.weight, and what does our kernel produce?"""
import sys, os, ctypes
sys.path.insert(0, "/tmp/bpd_test/bench")
import numpy as np
from gguf_helper import query_tensor, read_tensor_bytes

SO = "/tmp/bpd_test/build/bpd_cpu.so"
lib = ctypes.CDLL(SO)
lib.bpd_dequant_q8_0_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.bpd_dequant_q8_0_cpu.restype = None

gguf = "/tmp/llamatov-data/ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45"
info = query_tensor(gguf, "blk.0.attn_q.weight")
raw = read_tensor_bytes(gguf, info)
print(f"raw type: {type(raw)}, dtype: {raw.dtype}")
print(f"raw[0:5] = {list(raw[:5])}")

# Reconstruct scale_u16 the way check_f16_to_f32.py did
scale_u16 = int(raw[0]) | (int(raw[1]) << 8)
print(f"scale_u16 = {scale_u16:#06x}")

# Now what does dequant output?
block_np = np.ascontiguousarray(raw[:34], dtype=np.uint8)
out = np.zeros(32, dtype=np.float32)
lib.bpd_dequant_q8_0_cpu(block_np.ctypes.data, out.ctypes.data, ctypes.c_int(1))
print(f"Our dequant output[:5]: {out[:5]}")

# What does numpy say?
scale_f16 = np.frombuffer(bytes([raw[0], raw[1]]), dtype=np.float16)[0]
scale_f32 = np.float32(scale_f16)
quants_i8 = np.frombuffer(bytes(raw[2:34]), dtype=np.int8)
ref = quants_i8.astype(np.float32) * scale_f32
print(f"numpy ref[:5]: {ref[:5]}")

print(f"diff: {(out - ref)[:5]}")
