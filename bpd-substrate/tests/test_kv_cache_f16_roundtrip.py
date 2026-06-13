# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""test_kv_cache_f16_roundtrip.py — Isolated test for f16 KV cache write+read.

Tests the exact sequence: f32 → bpd_kv_cache_write_f16_cpu → f16 cache → dequant → f32
with the same parameters as Llama 3.2 1B forward pass.

If this passes, the bug is in the block function's wiring, not the cache ops.
If this fails, the bug is in the write or dequant functions.
"""
import ctypes, numpy as np, sys, os

c_float_p = ctypes.POINTER(ctypes.c_float)
c_uint8_p = ctypes.POINTER(ctypes.c_uint8)
c_uint16_p = ctypes.POINTER(ctypes.c_ushort)
c_int32_p = ctypes.POINTER(ctypes.c_int)

lib = ctypes.CDLL(sys.argv[1] if len(sys.argv) > 1 else "build/bpd_cpu.so")

# Setup argtypes
lib.bpd_kv_cache_write_f16_cpu.argtypes = [
    c_uint16_p, c_float_p, c_int32_p, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.bpd_kv_cache_write_f16_cpu.restype = None

lib.bpd_kv_cache_write_cpu.argtypes = [
    c_float_p, c_float_p, c_int32_p, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.bpd_kv_cache_write_cpu.restype = None

# Llama 3.2 1B parameters
n_kv_heads = 8
head_dim = 64
max_seq_len = 128
n_tokens = 2
row_stride = n_kv_heads * head_dim  # 512

print("=== KV Cache F16 Roundtrip Test ===")
print(f"n_kv_heads={n_kv_heads} head_dim={head_dim} max_seq_len={max_seq_len}")
print(f"n_tokens={n_tokens} row_stride={row_stride}")
print()

# Generate realistic K values (from a RoPE'd projection)
np.random.seed(42)
k_src = np.random.randn(n_tokens * row_stride).astype(np.float32) * 3.0
pos_ids = np.arange(n_tokens, dtype=np.int32)

# Test 1: F32 cache write + read
print("--- F32 cache path ---")
cache_f32 = np.zeros(max_seq_len * row_stride, dtype=np.float32)
lib.bpd_kv_cache_write_cpu(
    cache_f32.ctypes.data_as(c_float_p),
    k_src.ctypes.data_as(c_float_p),
    pos_ids.ctypes.data_as(c_int32_p),
    n_tokens, n_kv_heads, head_dim, max_seq_len)

# Read back: positions 0..n_tokens-1
readback_f32 = cache_f32[:n_tokens * row_stride].copy()
diff_f32 = np.abs(readback_f32 - k_src).max()
print(f"  Write+read roundtrip: max_diff={diff_f32:.2e}")
print(f"  Nonzero in cache: {(cache_f32 != 0).sum()}/{len(cache_f32)}")

# Test 2: F16 cache write + dequant read
print("\n--- F16 cache path ---")
cache_f16 = np.zeros(max_seq_len * row_stride, dtype=np.float16)
lib.bpd_kv_cache_write_f16_cpu(
    cache_f16.ctypes.data_as(c_uint16_p),
    k_src.ctypes.data_as(c_float_p),
    pos_ids.ctypes.data_as(c_int32_p),
    n_tokens, n_kv_heads, head_dim, max_seq_len)

print(f"  Nonzero in f16 cache: {(cache_f16 != 0).sum()}/{len(cache_f16)}")
print(f"  First 5 f16 values: {cache_f16[:5]}")
print(f"  First 5 f16 bits: {['0x%04x' % int(np.array(v).view(np.uint16)) for v in cache_f16[:5]]}")

# Dequant: f16 → f32 (same as the block function does)
n_kv = n_tokens  # kv_pos=0, so n_kv = kv_pos + n_tokens = n_tokens
kv_elems = n_kv * n_kv_heads * head_dim
cache_f16_raw = cache_f16.view(np.uint16)[:kv_elems]
readback_f16 = cache_f16_raw.view(np.float16).astype(np.float32)  # view=reinterpret bits, not convert integers

print(f"  Readback first 5: {readback_f16[:5]}")
print(f"  Nonzero readback: {(readback_f16 != 0).sum()}/{len(readback_f16)}")

# Compare f32 vs f16 roundtrip
f32_roundtrip = k_src[:kv_elems]
f16_roundtrip = readback_f16
diff_roundtrip = np.abs(f32_roundtrip - f16_roundtrip).max()
print(f"  F32 vs F16 roundtrip max_diff: {diff_roundtrip:.2e}")

# Test 3: Multi-layer simulation (the actual failure case)
print("\n--- Multi-layer cache test ---")
n_layers = 3
kv_total = n_layers * max_seq_len * row_stride
cache_ml = np.zeros(kv_total, dtype=np.float16)

for layer in range(n_layers):
    # Each layer writes different values
    k_layer = np.random.randn(n_tokens * row_stride).astype(np.float32) * 3.0
    layer_offset = layer * max_seq_len * row_stride
    layer_cache = cache_ml[layer_offset:layer_offset + max_seq_len * row_stride]
    
    lib.bpd_kv_cache_write_f16_cpu(
        layer_cache.ctypes.data_as(c_uint16_p),
        k_layer.ctypes.data_as(c_float_p),
        pos_ids.ctypes.data_as(c_int32_p),
        n_tokens, n_kv_heads, head_dim, max_seq_len)
    
    # Read back and verify
    readback = layer_cache[:kv_elems].view(np.uint16).astype(np.float16).astype(np.float32)
    nz = (readback != 0).sum()
    maxv = np.abs(readback).max()
    status = "✅" if nz == kv_elems and maxv < 100 else "❌"
    print(f"  Layer {layer}: nonzero={nz}/{kv_elems} max={maxv:.4f} {status}")

    # Check: does reading via C pointer arithmetic match?
    # Simulate what the forward pass does:
    # layer_k = (char*)k_cache + layer * kv_layer_stride * sizeof(uint16_t)
    byte_offset = layer * max_seq_len * row_stride * 2  # uint16_t = 2 bytes
    direct_read = np.frombuffer(cache_ml.tobytes()[byte_offset:byte_offset + kv_elems * 2], 
                                dtype=np.float16).astype(np.float32)
    nz2 = (direct_read != 0).sum()
    match = np.allclose(readback, direct_read)
    print(f"         C-style read: nonzero={nz2}/{kv_elems} matches_numpy: {match}")

print("\nDone.")
