#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Single-row test (#1) — CORRECTED offset (off = data_off + ro).
Compare bpd_q8_0_dot on weight-row0 . activation-tok0 vs dump[row0,tok0]."""
import sys, os, ctypes
import numpy as np
sys.path.insert(0, _BPD)
from llamatov_run import parse_gguf
import os as _os, sys as _sys
def _bpd_root(_p=_os.path.dirname(_os.path.abspath(__file__))):
    while _p != '/' and _os.path.basename(_p) != 'bpd':
        _p = _os.path.dirname(_p)
    return _p if _os.path.basename(_p) == 'bpd' else _os.path.dirname(_os.path.abspath(__file__))
_BPD = _bpd_root()


BLOB = "/tmp/llamatov-data/ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45"
DUMP = "<home>/tmp/spec_dump_v2"
QK, K, NB = 32, 2048, 64
SO = "/tmp/bpd_dot.so"
HDR = 80  # dump header bytes

# ---- 1. weight row 0 (q8_0), CORRECTED offset ----
meta, tens, data_off = parse_gguf(BLOB)
shape, dtype, ro = tens["blk.0.attn_q.weight"]
assert dtype == 8
off = data_off + ro  # FIX: absolute = data_off + relative
ROW_BYTES = NB * 34
wq = np.empty(K, dtype=np.int8); wd = np.empty(NB, dtype=np.float32)
with open(BLOB, "rb") as f:
    f.seek(off); raw = f.read(ROW_BYTES)
for b in range(NB):
    blk = raw[b*34:(b+1)*34]
    wd[b] = np.frombuffer(blk[:2], dtype=np.float16)[0].astype(np.float32)
    wq[b*QK:(b+1)*QK] = np.frombuffer(blk[2:34], dtype=np.int8)
print(f"weight row0: blk0 d={wd[0]:.6f} q[:8]={wq[:8].tolist()}  wd finite={np.isfinite(wd).all()}")

# ---- 2. activation token 0 (f32 -> q8_0) ----
with open(os.path.join(DUMP, "0004_MUL_attn_norm-0.bin"), "rb") as f:
    f.read(HDR); act = np.frombuffer(f.read(), dtype=np.float32)
act0 = act[:K].astype(np.float32).copy()
print(f"act0[:4]={act0[:4].tolist()}  finite={np.isfinite(act0).all()}")
aq = np.empty(K, dtype=np.int8); ad = np.empty(NB, dtype=np.float32)
for b in range(NB):
    x = act0[b*QK:(b+1)*QK].astype(np.float32)
    amax = np.float32(0.0)
    for v in x:
        av = np.float32(abs(v))
        if av > amax: amax = av
    d = np.float32(amax / np.float32(127.0))           # ggml: d = amax/127 (fp32)
    idv = np.float32(1.0 / d) if d != 0 else np.float32(0.0)  # ggml: id = 1.0f/d (fp32)
    d_fp16 = np.float32(np.float16(d))                  # y.d = GGML_FP32_TO_FP16(d)
    x0 = (x * idv).astype(np.float32)                   # x*id in fp32
    q = np.rint(x0)                                     # round half to EVEN (SIMD path)
    aq[b*QK:(b+1)*QK] = np.clip(q, -127, 127).astype(np.int8)
    ad[b] = d_fp16
print(f"act0 q8: blk0 d={ad[0]:.6e} q[:8]={aq[:8].tolist()}")

# ---- 2b. pure-python reference dot (sanity) ----
acc = 0.0
for b in range(NB):
    isum = int(np.sum(wq[b*QK:(b+1)*QK].astype(np.int32) * aq[b*QK:(b+1)*QK].astype(np.int32)))
    acc += float(wd[b]) * float(ad[b]) * isum
print(f"pure-python dot = {acc!r}")

# ---- 3. kernel call ----
lib = ctypes.CDLL(SO)
lib.bpd_q8_0_dot.restype = ctypes.c_float
lib.bpd_q8_0_dot.argtypes = [ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_float),
                              ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_float),
                              ctypes.c_int32]
got = lib.bpd_q8_0_dot(wq.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                       wd.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                       aq.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                       ad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), NB)

# ---- 4. expected dump[row0,tok0] ----
with open(os.path.join(DUMP, "0007_MUL_MAT_Qcur-0.bin"), "rb") as f:
    f.read(HDR); out = np.frombuffer(f.read(), dtype=np.float32)
expected = float(out[0])

# ---- 5. compare ----
print("\n=== RESULT ===")
print(f"kernel   = {got!r}")
print(f"purepy   = {acc!r}")
print(f"expected = {expected!r}")
print(f"abs diff (kernel vs expected) = {abs(got-expected):.6e}")
gi = np.frombuffer(np.float32(got).tobytes(), dtype=np.int32)[0]
ei = np.frombuffer(np.float32(expected).tobytes(), dtype=np.int32)[0]
print(f"ULP dist (kernel vs expected) = {abs(int(gi)-int(ei))}")
