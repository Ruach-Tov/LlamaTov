# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Verify our f16_to_f32 matches numpy's exactly for all F16 scales in attn_q.weight."""
import sys, os, ctypes
sys.path.insert(0, "/tmp/bpd_test/bench")
import numpy as np
from gguf_helper import query_tensor, read_tensor_bytes

gguf = "/tmp/llamatov-data/ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45"
info = query_tensor(gguf, "blk.0.attn_q.weight")
raw = read_tensor_bytes(gguf, info)
print(f"Got {len(raw)} bytes, {len(raw)//34} Q8_0 blocks")

# Extract all F16 scales (first 2 bytes of each 34-byte block)
n_blocks = len(raw) // 34
scales_u16 = np.zeros(n_blocks, dtype=np.uint16)
for i in range(n_blocks):
    scales_u16[i] = raw[i*34] | (raw[i*34 + 1] << 8)

# numpy reference: view as F16 then convert to F32
scales_f16 = scales_u16.view(np.float16)
scales_f32_numpy = scales_f16.astype(np.float32)

# Our reference: must implement f16_to_f32 in Python or use a ctypes call
# Build a small C bridge
import ctypes
SO = "/tmp/bpd_test/build/bpd_cpu.so"
lib = ctypes.CDLL(SO)
# Use bpd_dequant_q8_0_cpu on a synthetic block where the int8s are all 1
# Then the output = (1.0) * scale, so we can read scales out of output[0]
synthetic_blocks = bytearray()
for i in range(n_blocks):
    synthetic_blocks.extend([raw[i*34], raw[i*34+1]])  # F16 scale
    synthetic_blocks.extend([1] * 32)  # all 1s
synthetic_raw = np.array(synthetic_blocks, dtype=np.uint8)
synthetic_raw_c = np.ascontiguousarray(synthetic_raw)
out = np.zeros(n_blocks * 32, dtype=np.float32)
lib.bpd_dequant_q8_0_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.bpd_dequant_q8_0_cpu.restype = None
lib.bpd_dequant_q8_0_cpu(synthetic_raw_c.ctypes.data, out.ctypes.data, ctypes.c_int(n_blocks))
# Our scales: every 32nd output should equal scale * 1 = scale
our_scales = out[::32]

# Compare
diff_mask = scales_f32_numpy != our_scales
n_diff = diff_mask.sum()
print(f"Scales total: {n_blocks}")
print(f"Where our_f16_to_f32 != numpy's: {n_diff}")
if n_diff > 0:
    diff_idx = np.where(diff_mask)[0][:5]
    for i in diff_idx:
        print(f"  block {i}: F16 raw {scales_u16[i]:#06x}, numpy={scales_f32_numpy[i]:.10g}, ours={our_scales[i]:.10g}")
else:
    print("ALL F16 -> F32 conversions match numpy exactly.")
