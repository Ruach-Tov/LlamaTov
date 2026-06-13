#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Test batched GPU SW: correctness + throughput vs single-alignment."""
import ctypes, random, time, os

os.system("gcc -O2 -shared -fPIC -o /tmp/bpd_sw_cpu.so bench/bpd_smith_waterman.c")
cpu = ctypes.CDLL("/tmp/bpd_sw_cpu.so")
cpu.bpd_sw_score_cpu.argtypes = [
    ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
cpu.bpd_sw_score_cpu.restype = ctypes.c_int

try:
    batch = ctypes.CDLL("/tmp/bpd_sw_batch.so")
    batch.bpd_sw_batch_gpu.argtypes = [
        ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.c_char_p, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    HAS_BATCH = True
    print("Batch GPU loaded")
except:
    HAS_BATCH = False
    print("Batch GPU not available")
    exit(1)

rng = random.Random(42)
bases = "ACGT"

# Generate reference
ref_len = 1000
ref = ''.join(rng.choice(bases) for _ in range(ref_len))

# === Correctness ===
print("\n=== Correctness: batch vs CPU (per-query) ===")
n_queries = 50
queries = []
for _ in range(n_queries):
    qlen = rng.randint(50, 200)
    queries.append(''.join(rng.choice(bases) for _ in range(qlen)))

# CPU scores
cpu_scores = []
for q in queries:
    s = cpu.bpd_sw_score_cpu(q.encode(), len(q), ref.encode(), ref_len, 2, -1, 3, 1)
    cpu_scores.append(s)

# Batch GPU scores
q_ptrs = (ctypes.c_char_p * n_queries)(*[q.encode() for q in queries])
q_lens = (ctypes.c_int * n_queries)(*[len(q) for q in queries])
gpu_scores = (ctypes.c_int * n_queries)()

batch.bpd_sw_batch_gpu(q_ptrs, q_lens, n_queries,
    ref.encode(), ref_len, 2, -1, 3, 1, 128, gpu_scores)

matches = sum(1 for c, g in zip(cpu_scores, list(gpu_scores)) if c == g)
print("  %d/%d BIT-IDENTICAL" % (matches, n_queries))
if matches < n_queries:
    for i in range(min(5, n_queries)):
        if cpu_scores[i] != gpu_scores[i]:
            print("    DIFF query %d (len=%d): cpu=%d gpu=%d" % (i, len(queries[i]), cpu_scores[i], gpu_scores[i]))

# === Throughput ===
print("\n=== Throughput: batch GPU vs sequential ===")

try:
    v3 = ctypes.CDLL("/tmp/bpd_sw_gpu_v3.so")
    v3.bpd_sw_score_gpu_v3.argtypes = [
        ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    v3.bpd_sw_score_gpu_v3.restype = ctypes.c_int
    HAS_V3 = True
except:
    HAS_V3 = False

for n_q in [20, 50, 100, 200, 500]:
    qlen = 150  # Illumina read length
    qs = [''.join(rng.choice(bases) for _ in range(qlen)) for _ in range(n_q)]
    
    # Sequential v3
    if HAS_V3:
        # Warmup
        v3.bpd_sw_score_gpu_v3(qs[0].encode(), qlen, ref.encode(), ref_len, 2, -1, 3, 1, 256)
        t0 = time.perf_counter()
        for q in qs:
            v3.bpd_sw_score_gpu_v3(q.encode(), qlen, ref.encode(), ref_len, 2, -1, 3, 1, 256)
        t_seq = (time.perf_counter() - t0) * 1000
    
    # Batch
    q_ptrs = (ctypes.c_char_p * n_q)(*[q.encode() for q in qs])
    q_lens = (ctypes.c_int * n_q)(*[qlen] * n_q)
    gpu_scores = (ctypes.c_int * n_q)()
    
    # Warmup
    batch.bpd_sw_batch_gpu(q_ptrs, q_lens, n_q, ref.encode(), ref_len, 2, -1, 3, 1, 128, gpu_scores)
    
    t0 = time.perf_counter()
    iters = max(1, 5)
    for _ in range(iters):
        batch.bpd_sw_batch_gpu(q_ptrs, q_lens, n_q, ref.encode(), ref_len, 2, -1, 3, 1, 128, gpu_scores)
    t_batch = (time.perf_counter() - t0) / iters * 1000
    
    total_cells = n_q * qlen * ref_len
    gcups_batch = total_cells / (t_batch / 1000) / 1e9
    reads_per_sec = n_q / (t_batch / 1000)
    
    if HAS_V3:
        speedup = t_seq / t_batch
        print("  %3d reads: seq=%.1fms batch=%.1fms  %.1fx  %.2f GCUPS  %d reads/s" % (
            n_q, t_seq, t_batch, speedup, gcups_batch, int(reads_per_sec)))
    else:
        print("  %3d reads: batch=%.1fms  %.2f GCUPS  %d reads/s" % (
            n_q, t_batch, gcups_batch, int(reads_per_sec)))

print()
