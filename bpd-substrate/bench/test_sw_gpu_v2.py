#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Verify GPU SW v2 (single-launch) against CPU + benchmark vs v1."""
import ctypes, os, random, time

# Build
os.system("gcc -O2 -shared -fPIC -o /tmp/bpd_sw_cpu.so bench/bpd_smith_waterman.c")

cpu = ctypes.CDLL("/tmp/bpd_sw_cpu.so")
cpu.bpd_sw_score_cpu.argtypes = [
    ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
cpu.bpd_sw_score_cpu.restype = ctypes.c_int

try:
    v1 = ctypes.CDLL("/tmp/bpd_sw_gpu.so")
    v1.bpd_sw_score_gpu.argtypes = [
        ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    v1.bpd_sw_score_gpu.restype = ctypes.c_int
    HAS_V1 = True
except:
    HAS_V1 = False

try:
    v2 = ctypes.CDLL("/tmp/bpd_sw_gpu_v2.so")
    v2.bpd_sw_score_gpu_v2.argtypes = [
        ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    v2.bpd_sw_score_gpu_v2.restype = ctypes.c_int
    HAS_V2 = True
    print("GPU v2 (single-launch) loaded")
except:
    HAS_V2 = False
    print("GPU v2 not available")
    exit(1)

rng = random.Random(42)
bases = "ACGT"
passed = 0
failed = 0

def check(name, cond):
    global passed, failed
    if cond: passed += 1; print("  PASS %s" % name)
    else: failed += 1; print("  FAIL %s" % name)

# === Correctness: v2 vs CPU ===
print("\n=== Correctness: GPU v2 vs CPU ===")
for trial in range(20):
    qlen = rng.randint(10, 500)
    rlen = rng.randint(10, 500)
    q = ''.join(rng.choice(bases) for _ in range(qlen))
    r = ''.join(rng.choice(bases) for _ in range(rlen))

    cpu_score = cpu.bpd_sw_score_cpu(q.encode(), qlen, r.encode(), rlen, 2, -1, 3, 1)
    v2_score = v2.bpd_sw_score_gpu_v2(q.encode(), qlen, r.encode(), rlen, 2, -1, 3, 1, 256)

    if cpu_score != v2_score and trial < 5:
        print("  DIFF trial %d (%dx%d): cpu=%d v2=%d" % (trial, qlen, rlen, cpu_score, v2_score))

match_count = 0
for trial in range(20):
    qlen = rng.randint(10, 500)
    rlen = rng.randint(10, 500)
    q = ''.join(rng.choice(bases) for _ in range(qlen))
    r = ''.join(rng.choice(bases) for _ in range(rlen))
    cpu_score = cpu.bpd_sw_score_cpu(q.encode(), qlen, r.encode(), rlen, 2, -1, 3, 1)
    v2_score = v2.bpd_sw_score_gpu_v2(q.encode(), qlen, r.encode(), rlen, 2, -1, 3, 1, 256)
    if cpu_score == v2_score: match_count += 1

check("GPU v2 vs CPU: %d/20 BIT-IDENTICAL" % match_count, match_count == 20)

# === Performance: v1 vs v2 vs CPU ===
print("\n=== Performance: v1 (multi-launch) vs v2 (single-launch) ===")
print("%-8s %10s %10s %10s %10s %8s" % ("Size", "CPU", "GPU v1", "GPU v2", "SIMD", "v2/v1"))

for size in [100, 300, 500, 1000]:
    q = ''.join(rng.choice(bases) for _ in range(size))
    r = ''.join(rng.choice(bases) for _ in range(size))
    qb, rb = q.encode(), r.encode()
    iters = max(3, min(50, 500000 // (size * size)))

    # CPU
    t0 = time.perf_counter()
    for _ in range(iters): cpu.bpd_sw_score_cpu(qb, size, rb, size, 2, -1, 3, 1)
    t_cpu = (time.perf_counter() - t0) / iters * 1000

    # GPU v1
    if HAS_V1:
        v1.bpd_sw_score_gpu(qb, size, rb, size, 2, -1, 3, 1, 128)  # warmup
        t0 = time.perf_counter()
        for _ in range(iters): v1.bpd_sw_score_gpu(qb, size, rb, size, 2, -1, 3, 1, 128)
        t_v1 = (time.perf_counter() - t0) / iters * 1000
    else:
        t_v1 = 0

    # GPU v2
    v2.bpd_sw_score_gpu_v2(qb, size, rb, size, 2, -1, 3, 1, 256)  # warmup
    t0 = time.perf_counter()
    for _ in range(iters): v2.bpd_sw_score_gpu_v2(qb, size, rb, size, 2, -1, 3, 1, 256)
    t_v2 = (time.perf_counter() - t0) / iters * 1000

    speedup = t_v1 / t_v2 if t_v2 > 0 and t_v1 > 0 else 0
    print("%-8s %8.2fms %8.2fms %8.2fms %10s %7.1fx" % (
        "%dbp" % size, t_cpu, t_v1, t_v2, "N/A", speedup))

# === GCUPS comparison ===
print("\n=== GCUPS ===")
for size in [300, 500, 1000]:
    q = ''.join(rng.choice(bases) for _ in range(size))
    r = ''.join(rng.choice(bases) for _ in range(size))
    qb, rb = q.encode(), r.encode()
    cells = size * size
    iters = max(3, 20)

    v2.bpd_sw_score_gpu_v2(qb, size, rb, size, 2, -1, 3, 1, 256)
    t0 = time.perf_counter()
    for _ in range(iters): v2.bpd_sw_score_gpu_v2(qb, size, rb, size, 2, -1, 3, 1, 256)
    gcups = cells / ((time.perf_counter() - t0) / iters) / 1e9
    print("  %4dbp: %.2f GCUPS" % (size, gcups))

print("\nPASSED: %d  FAILED: %d" % (passed, failed))
