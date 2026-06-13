# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""check_qdot.py — debug L.1.10 by isolating just one (m, n) dot product."""
import sys, os, ctypes
sys.path.insert(0, "/tmp/bpd_test/bench")
import numpy as np
from llama_fixture_loader import load_manifest, find_op
from gguf_helper import query_tensor, read_tensor_bytes

SO = "/tmp/bpd_test/build/bpd_cpu.so"
lib = ctypes.CDLL(SO)
lib.bpd_quant_q8_0_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.bpd_quant_q8_0_cpu.restype = None
lib.bpd_qdot_q8_0_q8_0_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.bpd_qdot_q8_0_q8_0_cpu.restype = ctypes.c_float

tensors = load_manifest("/tmp/llama_dump_layer0")
attn_norm = find_op(tensors, name_substring="attn_norm-0", op_desc="MUL")
qcur = find_op(tensors, name_substring="Qcur-0", op_desc="MUL_MAT")
X = np.ascontiguousarray(attn_norm.as_numpy(), dtype=np.float32)
ref_full = np.ascontiguousarray(qcur.as_numpy(), dtype=np.float32)
print(f"X shape: {X.shape}, ref shape: {ref_full.shape}")
print(f"ref[0, 0] = {ref_full[0, 0]} (the value we want to reproduce)")

gguf = "/tmp/llamatov-data/ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45"
info = query_tensor(gguf, "blk.0.attn_q.weight")
W_raw = read_tensor_bytes(gguf, info)
bytes_per_row = (info.dims[0] // 32) * 34
print(f"bytes_per_row = {bytes_per_row}")
W_row0 = np.ascontiguousarray(W_raw[:bytes_per_row], dtype=np.uint8)

X_row0 = np.ascontiguousarray(X[0], dtype=np.float32)
X_q8_0 = np.zeros(bytes_per_row, dtype=np.uint8)
lib.bpd_quant_q8_0_cpu(X_row0.ctypes.data, X_q8_0.ctypes.data, ctypes.c_int(len(X_row0)))

n_blocks = info.dims[0] // 32
our_result = lib.bpd_qdot_q8_0_q8_0_cpu(W_row0.ctypes.data, X_q8_0.ctypes.data, ctypes.c_int(n_blocks))
print(f"Our C result for out[0, 0]:  {our_result}")
print(f"ggml ref for ref[0, 0]:      {ref_full[0, 0]}")
print(f"diff: {our_result - ref_full[0, 0]:.6e}")

def py_qdot(w_bytes, a_bytes, n_blocks):
    """Pure Python mirror of ggml's scalar reference."""
    sumf = np.float32(0.0)
    for ib in range(n_blocks):
        wb = w_bytes[ib*34:(ib+1)*34]
        ab = a_bytes[ib*34:(ib+1)*34]
        wq = np.frombuffer(wb[2:], dtype=np.int8)
        aq = np.frombuffer(ab[2:], dtype=np.int8)
        sumi = int(np.sum(wq.astype(np.int64) * aq.astype(np.int64)))
        wd_u16 = wb[:2].view(np.uint16)[0]
        ad_u16 = ab[:2].view(np.uint16)[0]
        wd = np.frombuffer(np.uint16(wd_u16).tobytes(), dtype=np.float16)[0]
        ad = np.frombuffer(np.uint16(ad_u16).tobytes(), dtype=np.float16)[0]
        wd_f32 = np.float32(wd)
        ad_f32 = np.float32(ad)
        # ggml's exact line: sumf += sumi * (wd * ad)
        block_contrib = np.float32(sumi) * (np.float32(wd_f32) * np.float32(ad_f32))
        sumf = np.float32(sumf + block_contrib)
    return float(sumf)

py_result = py_qdot(W_row0, X_q8_0, n_blocks)
print(f"Python ggml-mirror result:   {py_result}")
print(f"  vs our C:    diff={py_result - our_result:.6e}")
print(f"  vs ggml fix: diff={py_result - ref_full[0, 0]:.6e}")
