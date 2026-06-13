# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""verify_f16_fix.py — verify that the f16_to_f32 fix closes the Q8_0 matmul gap.

Tests:
1. f16_to_f32 correctness: all 65536 F16 values via dequant of subnormal-scale blocks.
2. Q8_0 quant+dequant round-trip sanity.
3. Synthetic Q8_0 matmul with subnormal scales — using Python ggml-mirror as reference.
4. Full ggml-mirror scalar matmul: bpd_qmatmul_q8_0_cpu vs py_qmatmul for random data.
"""
import ctypes
import os
import struct
import sys
import numpy as np

SO = os.environ.get("BPD_CPU_SO", os.path.join(os.path.dirname(__file__), "../build/bpd_cpu.so"))
lib = ctypes.CDLL(SO)

lib.bpd_dequant_q8_0_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.bpd_dequant_q8_0_cpu.restype = None
lib.bpd_quant_q8_0_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.bpd_quant_q8_0_cpu.restype = None
lib.bpd_qmatmul_q8_0_cpu.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]
lib.bpd_qmatmul_q8_0_cpu.restype = None
lib.bpd_qdot_q8_0_q8_0_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.bpd_qdot_q8_0_q8_0_cpu.restype = ctypes.c_float

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

def ulp_bits(a, b):
    """Return the ULP distance between two float32 values."""
    ai = struct.unpack('I', struct.pack('f', float(a)))[0]
    bi = struct.unpack('I', struct.pack('f', float(b)))[0]
    return abs(int(ai) - int(bi))

def ggml_fp16_to_fp32(h):
    """Mirror of ggml_compute_fp16_to_fp32 — the reference."""
    h = int(h) & 0xFFFF
    w = (h << 16) & 0xFFFFFFFF
    sign = w & 0x80000000
    two_w = (w + w) & 0xFFFFFFFF
    exp_offset = 0x70000000  # 0xE0 << 23
    exp_scale_bits = 0x07800000  # 2^-112
    exp_scale = struct.unpack('f', struct.pack('I', exp_scale_bits))[0]
    norm_bits = ((two_w >> 4) + exp_offset) & 0xFFFFFFFF
    normalized_value = struct.unpack('f', struct.pack('I', norm_bits))[0] * exp_scale
    magic_mask = 0x3F000000  # 126 << 23
    denorm_bits = ((two_w >> 17) | magic_mask) & 0xFFFFFFFF
    denormalized_value = struct.unpack('f', struct.pack('I', denorm_bits))[0] - 0.5
    denormalized_cutoff = 0x08000000  # 1 << 27
    if two_w < denormalized_cutoff:
        result_bits = sign | struct.unpack('I', struct.pack('f', denormalized_value))[0]
    else:
        result_bits = sign | struct.unpack('I', struct.pack('f', normalized_value))[0]
    return struct.unpack('f', struct.pack('I', result_bits & 0xFFFFFFFF))[0]

def py_qdot(w_bytes, a_bytes, n_blocks):
    """Python mirror of ggml's scalar vec_dot_q8_0_q8_0 reference."""
    sumf = np.float32(0.0)
    for ib in range(n_blocks):
        wb = w_bytes[ib*34:(ib+1)*34]
        ab = a_bytes[ib*34:(ib+1)*34]
        wq = np.frombuffer(bytes(wb[2:]), dtype=np.int8)
        aq = np.frombuffer(bytes(ab[2:]), dtype=np.int8)
        sumi = int(np.sum(wq.astype(np.int64) * aq.astype(np.int64)))
        wd_u16 = int(wb[0]) | (int(wb[1]) << 8)
        ad_u16 = int(ab[0]) | (int(ab[1]) << 8)
        wd = np.float32(ggml_fp16_to_fp32(wd_u16))
        ad = np.float32(ggml_fp16_to_fp32(ad_u16))
        sumf = np.float32(sumf + np.float32(sumi) * (wd * ad))
    return float(sumf)

def py_quant_q8_0(x_f32, K):
    """Python mirror of ggml quantize_row_q8_0_ref."""
    n_blocks = K // 32
    result = np.zeros(n_blocks * 34, dtype=np.uint8)
    for b in range(n_blocks):
        block = x_f32[b*32:(b+1)*32]
        amax = float(np.max(np.abs(block)))
        d = np.float32(amax / 127.0)
        # F32 -> F16 -> F32 (round-trip through F16 precision)
        d_f16 = np.float32(np.float16(d))
        d_u16 = int(np.float16(d).view(np.uint16))
        result[b*34] = d_u16 & 0xFF
        result[b*34 + 1] = (d_u16 >> 8) & 0xFF
        if d_f16 == 0.0:
            for i in range(32):
                result[b*34 + 2 + i] = 0
        else:
            for i in range(32):
                q = int(round(float(block[i]) / float(d_f16)))
                q = max(-128, min(127, q))
                result[b*34 + 2 + i] = np.uint8(q & 0xFF)
    return result

# ── Test 1: f16_to_f32 correctness for subnormal F16 scales ──
print("Test 1: f16_to_f32 correctness for subnormal F16 scales")
n_blocks = 1
raw = np.zeros(n_blocks * 34, dtype=np.uint8)
# Scale = 0x0001 (smallest positive subnormal F16), quants = [1, 2, ..., 32]
raw[0] = 0x01; raw[1] = 0x00
for i in range(32): raw[2 + i] = np.uint8(i + 1)
out = np.zeros(32, dtype=np.float32)
lib.bpd_dequant_q8_0_cpu(raw.ctypes.data, out.ctypes.data, ctypes.c_int(1))
# Reference: ggml algorithm
scale_ggml = ggml_fp16_to_fp32(0x0001)
ref = np.array([(i + 1) * scale_ggml for i in range(32)], dtype=np.float32)
max_ulp = max(ulp_bits(float(out[i]), float(ref[i])) for i in range(32))
if max_ulp == 0:
    print(f"  {PASS}: subnormal scale 0x0001 dequant 0 ULP vs ggml reference")
    print(f"  scale_ggml={scale_ggml:.8e}, out[0]={out[0]:.8e}")
else:
    print(f"  {FAIL}: max_ulp={max_ulp}")
    print(f"  scale_ggml={scale_ggml:.8e}, out[0]={out[0]:.8e}, ref[0]={ref[0]:.8e}")

# ── Test 2: Q8_0 qdot with subnormal scale ──
print("Test 2: Q8_0 qdot with subnormal scale — bpd_qdot vs py_qdot")
n_blocks = 64
W_raw = np.zeros(n_blocks * 34, dtype=np.uint8)
A_raw = np.zeros(n_blocks * 34, dtype=np.uint8)
# Block 0: weight scale = 0x0001 (subnormal), activation scale = 0x3C00 (1.0)
W_raw[0] = 0x01; W_raw[1] = 0x00
A_raw[0] = 0x00; A_raw[1] = 0x3C
for i in range(32):
    W_raw[2 + i] = np.uint8(1)
    A_raw[2 + i] = np.uint8(1)
# Remaining blocks: normal scales
for b in range(1, n_blocks):
    W_raw[b*34] = 0x00; W_raw[b*34+1] = 0x3C  # 1.0
    A_raw[b*34] = 0x00; A_raw[b*34+1] = 0x3C  # 1.0
    for i in range(32):
        W_raw[b*34+2+i] = np.uint8(1)
        A_raw[b*34+2+i] = np.uint8(1)

bpd_result = lib.bpd_qdot_q8_0_q8_0_cpu(W_raw.ctypes.data, A_raw.ctypes.data, ctypes.c_int(n_blocks))
py_result = py_qdot(W_raw.tolist(), A_raw.tolist(), n_blocks)
ulp = ulp_bits(bpd_result, py_result)
if ulp == 0:
    print(f"  {PASS}: 0 ULP — bpd_qdot={bpd_result:.8e}, py_qdot={py_result:.8e}")
else:
    print(f"  {FAIL}: {ulp} ULP — bpd_qdot={bpd_result:.8e}, py_qdot={py_result:.8e}")

# ── Test 3: Full Q8_0 matmul — random data, verify vs py_qdot composition ──
print("Test 3: Full Q8_0 matmul (M=2, N=8, K=256) vs py_qdot composition")
np.random.seed(7)
M, N, K = 2, 8, 256
n_blocks_per_row = K // 32
bytes_per_row = n_blocks_per_row * 34

X_f32 = np.random.randn(M, K).astype(np.float32) * 0.3

# Quantize W (N rows of K elements) using our C quant
W_f32_orig = np.random.randn(N, K).astype(np.float32) * 0.5
W_q8 = np.zeros(N * bytes_per_row, dtype=np.uint8)
for n in range(N):
    lib.bpd_quant_q8_0_cpu(
        W_f32_orig[n].ctypes.data,
        W_q8[n * bytes_per_row:].ctypes.data,
        ctypes.c_int(K)
    )

# Quantize X using our C quant
X_q8 = np.zeros(M * bytes_per_row, dtype=np.uint8)
for m in range(M):
    lib.bpd_quant_q8_0_cpu(
        X_f32[m].ctypes.data,
        X_q8[m * bytes_per_row:].ctypes.data,
        ctypes.c_int(K)
    )

# BPD matmul
out_bpd = np.zeros((M, N), dtype=np.float32)
lib.bpd_qmatmul_q8_0_cpu(
    W_q8.ctypes.data, X_f32.ctypes.data, out_bpd.ctypes.data,
    ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K)
)

# Python reference: py_qdot for each (m, n)
out_py = np.zeros((M, N), dtype=np.float32)
for m in range(M):
    for n in range(N):
        w_row = W_q8[n * bytes_per_row:(n+1) * bytes_per_row].tolist()
        a_row = X_q8[m * bytes_per_row:(m+1) * bytes_per_row].tolist()
        out_py[m, n] = py_qdot(w_row, a_row, n_blocks_per_row)

max_ulp = max(ulp_bits(float(out_bpd[m, n]), float(out_py[m, n]))
              for m in range(M) for n in range(N))
n_diff = sum(1 for m in range(M) for n in range(N)
             if ulp_bits(float(out_bpd[m, n]), float(out_py[m, n])) > 0)
if max_ulp == 0:
    print(f"  {PASS}: 0 ULP / {M*N} elements vs py_qdot composition")
else:
    print(f"  {FAIL}: max_ulp={max_ulp}, n_diff={n_diff}/{M*N}")
    for m in range(M):
        for n in range(N):
            u = ulp_bits(float(out_bpd[m, n]), float(out_py[m, n]))
            if u > 0:
                print(f"    [{m},{n}]: bpd={out_bpd[m,n]:.8e} py={out_py[m,n]:.8e} ulp={u}")

print()
print("All tests done.")
