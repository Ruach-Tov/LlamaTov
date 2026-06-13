#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Sweep GPU Q4_K dequant parameters, verify 0 ULP at each setting."""
import ctypes, numpy as np, struct, time, sys

# Load CPU reference
cpu = ctypes.CDLL("/tmp/bpd_quant_cpu.so")
cpu.bpd_dequant_q4k.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

# Load GPU sweep lib
gpu = ctypes.CDLL("/tmp/bpd_quant_gpu_sweep.so")
for name in ['dequant_v1_256', 'dequant_v1_128', 'dequant_v1_64', 'dequant_v1_32',
             'dequant_ilp2', 'dequant_multi2', 'dequant_multi4', 'dequant_multi8']:
    fn = getattr(gpu, name)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
gpu.sgpu_alloc.restype = ctypes.c_void_p
gpu.sgpu_alloc.argtypes = [ctypes.c_int]
gpu.sgpu_free.argtypes = [ctypes.c_void_p]
gpu.sgpu_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
gpu.sgpu_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
gpu.sgpu_sync.argtypes = []

# Load real data
path = "${OLLAMA_BLOBS:-~/.ollama/models/blobs}/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f"
with open(path, "rb") as f:
    f.read(4); f.read(4)
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    for _ in range(n_kv):
        klen = struct.unpack("<Q", f.read(8))[0]; f.read(klen)
        vtype = struct.unpack("<I", f.read(4))[0]
        if vtype == 8: slen = struct.unpack("<Q", f.read(8))[0]; f.read(slen)
        elif vtype in (4,5,6): f.read(4)
        elif vtype == 7: f.read(1)
        elif vtype == 10: f.read(8)
        elif vtype == 9:
            etype = struct.unpack("<I", f.read(4))[0]
            count = struct.unpack("<Q", f.read(8))[0]
            if etype == 8:
                for _ in range(count): sl = struct.unpack("<Q", f.read(8))[0]; f.read(sl)
            elif etype in (0,1,7): f.read(count)
            elif etype in (4,5,6): f.read(count*4)
            elif etype in (10,11): f.read(count*8)
            elif etype in (2,3): f.read(count*2)
    tensors = []
    for _ in range(n_tensors):
        nlen = struct.unpack("<Q", f.read(8))[0]; name = f.read(nlen).decode()
        ndims = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(ndims)]
        ttype = struct.unpack("<I", f.read(4))[0]; offset = struct.unpack("<Q", f.read(8))[0]
        tensors.append((name, dims, ttype, offset))
    meta_end = f.tell(); data_start = meta_end + ((32 - meta_end % 32) % 32)
    for name, dims, ttype, offset in tensors:
        if "ffn_gate" in name and ttype == 12:
            n_blocks = min(4096, (dims[0]*dims[1]) // 256)
            f.seek(data_start + offset)
            qdata = np.frombuffer(f.read(n_blocks * 144), dtype=np.uint8).copy()
            break

K = n_blocks * 256
print("Sweeping GPU Q4_K dequant: %d blocks (%d elements)" % (n_blocks, K))

# CPU reference
cpu_out = np.zeros(K, dtype=np.float32)
cpu.bpd_dequant_q4k(qdata.ctypes.data, cpu_out.ctypes.data, n_blocks)

# GPU setup
d_q = gpu.sgpu_alloc(len(qdata))
d_out = gpu.sgpu_alloc(K * 4)
gpu.sgpu_h2d(d_q, qdata.ctypes.data, len(qdata))
gpu.sgpu_sync()

def ulp_max(a, b):
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    B = np.int64(0x80000000)
    ai = np.where(ai < 0, B - ai, ai)
    bi = np.where(bi < 0, B - bi, bi)
    return int(np.abs(ai - bi).max())

print("\n%-25s %12s %12s %6s" % ("Variant", "Time (us)", "M elem/s", "ULP"))
print("=" * 60)

variants = [
    ("v1 block=256", gpu.dequant_v1_256),
    ("v1 block=128", gpu.dequant_v1_128),
    ("v1 block=64",  gpu.dequant_v1_64),
    ("v1 block=32",  gpu.dequant_v1_32),
    ("ILP=2 (128t)", gpu.dequant_ilp2),
    ("multi=2",      gpu.dequant_multi2),
    ("multi=4",      gpu.dequant_multi4),
    ("multi=8",      gpu.dequant_multi8),
]

for name, fn in variants:
    # Warmup
    fn(d_q, d_out, n_blocks); gpu.sgpu_sync()
    # Benchmark
    iters = 500
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(d_q, d_out, n_blocks)
    gpu.sgpu_sync()
    t1 = time.perf_counter()
    us = (t1 - t0) / iters * 1e6
    meps = K / us
    # Verify
    gpu_out = np.zeros(K, dtype=np.float32)
    gpu.sgpu_d2h(gpu_out.ctypes.data, d_out, K * 4)
    ulp = ulp_max(cpu_out, gpu_out)
    tag = " ** BEST **" if us < 20 else ""
    print("%-25s %10.1f %10.0f %6d%s" % (name, us, meps, ulp, tag))

gpu.sgpu_free(d_q)
gpu.sgpu_free(d_out)
