#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Head-to-head: BPD vs llama.cpp GPU Q4_K dequant on real Mistral data."""
import ctypes, numpy as np, struct, time

gpu = ctypes.CDLL("/tmp/gpu_h2h.so")
for fn in ['run_llamacpp', 'run_bpd']:
    getattr(gpu, fn).argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
gpu.galloc.restype = ctypes.c_void_p
gpu.galloc.argtypes = [ctypes.c_int]
gpu.gfree.argtypes = [ctypes.c_void_p]
gpu.gh2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
gpu.gd2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
gpu.gsync.argtypes = []

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
            etype = struct.unpack("<I", f.read(4))[0]; count = struct.unpack("<Q", f.read(8))[0]
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
print("GPU head-to-head: %d blocks (%d elements)" % (n_blocks, K))
print("Real Mistral 7B Q4_K data (blk.0.ffn_gate.weight)")

d_q = gpu.galloc(len(qdata))
d_ll = gpu.galloc(K * 4)
d_bpd = gpu.galloc(K * 4)
gpu.gh2d(d_q, qdata.ctypes.data, len(qdata))
gpu.gsync()

# Warmup both
gpu.run_llamacpp(d_q, d_ll, n_blocks); gpu.gsync()
gpu.run_bpd(d_q, d_bpd, n_blocks); gpu.gsync()

# Benchmark llama.cpp
iters = 2000
t0 = time.perf_counter()
for _ in range(iters): gpu.run_llamacpp(d_q, d_ll, n_blocks)
gpu.gsync()
t1 = time.perf_counter()
ll_us = (t1 - t0) / iters * 1e6

# Benchmark BPD
t0 = time.perf_counter()
for _ in range(iters): gpu.run_bpd(d_q, d_bpd, n_blocks)
gpu.gsync()
t1 = time.perf_counter()
bpd_us = (t1 - t0) / iters * 1e6

print("\nllama.cpp GPU: %.1f us  (%.0f M elem/s)" % (ll_us, K / ll_us))
print("BPD GPU:       %.1f us  (%.0f M elem/s)" % (bpd_us, K / bpd_us))
print("Ratio:         %.3fx (BPD/llama.cpp)" % (bpd_us / ll_us))

# Verify bit-identical
ll_out = np.zeros(K, dtype=np.float32)
bpd_out = np.zeros(K, dtype=np.float32)
gpu.gd2h(ll_out.ctypes.data, d_ll, K * 4)
gpu.gd2h(bpd_out.ctypes.data, d_bpd, K * 4)

ai = ll_out.view(np.int32).astype(np.int64)
bi = bpd_out.view(np.int32).astype(np.int64)
B = np.int64(0x80000000)
ai2 = np.where(ai < 0, B - ai, ai)
bi2 = np.where(bi < 0, B - bi, bi)
ulp = int(np.abs(ai2 - bi2).max())
diffs = int((np.abs(ai2 - bi2) > 0).sum())
print("\nULP: max=%d  diffs=%d/%d" % (ulp, diffs, K))
if ulp == 0:
    print("BIT-IDENTICAL with llama.cpp GPU kernel")

gpu.gfree(d_q); gpu.gfree(d_ll); gpu.gfree(d_bpd)
