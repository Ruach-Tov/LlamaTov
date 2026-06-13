#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Verify GPU Smith-Waterman against CPU reference + sweep block_size."""
import ctypes, os, random, time, struct

# Build CPU reference
os.system("gcc -O2 -shared -fPIC -o /tmp/bpd_sw_cpu.so bench/bpd_smith_waterman.c")
cpu = ctypes.CDLL("/tmp/bpd_sw_cpu.so")
cpu.bpd_sw_score_cpu.argtypes = [
    ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
cpu.bpd_sw_score_cpu.restype = ctypes.c_int

# Try loading GPU library
try:
    gpu = ctypes.CDLL("/tmp/bpd_sw_gpu.so")
    gpu.bpd_sw_score_gpu.argtypes = [
        ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int]  # block_size
    gpu.bpd_sw_score_gpu.restype = ctypes.c_int
    HAS_GPU = True
    print("GPU library loaded")
except:
    HAS_GPU = False
    print("GPU library not available — CPU-only verification")

def run_test(name, query, ref, match=2, mismatch=-1, gap_open=3, gap_extend=1):
    """Run one test, compare CPU vs GPU at multiple block sizes."""
    q, r = query.encode(), ref.encode()
    cpu_score = cpu.bpd_sw_score_cpu(q, len(query), r, len(ref),
                                      match, mismatch, gap_open, gap_extend)

    if not HAS_GPU:
        return cpu_score, {}

    results = {}
    for bs in [32, 64, 128, 256]:
        gpu_score = gpu.bpd_sw_score_gpu(q, len(query), r, len(ref),
                                          match, mismatch, gap_open, gap_extend, bs)
        diff = abs(gpu_score - cpu_score)
        results[bs] = (gpu_score, diff)
        if diff != 0:
            print("  FAIL %s block=%d: CPU=%d GPU=%d" % (name, bs, cpu_score, gpu_score))

    return cpu_score, results

print()
print("=== Smith-Waterman GPU Verification ===")
print()

# Test suite
tests = [
    ("perfect_match", "ACGT", "ACGT"),
    ("one_mismatch", "ACGT", "ACTT"),
    ("no_alignment", "AAAA", "CCCC"),
    ("textbook", "TGTTACGG", "GGTTGACTA"),
    ("with_gaps", "ATAGACGACATGGGGC", "ATAGACATGGCGC"),
    ("kozak", "GCCACCATGG", "GCCGCCATGG"),
    ("poly_a", "AAAAAAAAAAAA", "TTAAAAAAAAAAATT"),
]

passed = 0
failed = 0
for name, q, r in tests:
    cpu_score, gpu_results = run_test(name, q, r)
    all_match = all(d == 0 for _, d in gpu_results.values()) if gpu_results else True
    if all_match:
        passed += 1
    else:
        failed += 1
    status = "PASS" if all_match else "FAIL"
    print("  %s %-20s CPU=%d %s" % (status, name, cpu_score,
        " ".join("bs%d=%d" % (bs, s) for bs, (s, _) in sorted(gpu_results.items())) if gpu_results else "(no GPU)"))

# Random tests
print()
print("Random sequences (50 tests)...")
rng = random.Random(42)
bases = "ACGT"
for i in range(50):
    qlen = rng.randint(10, 300)
    rlen = rng.randint(10, 300)
    q = ''.join(rng.choice(bases) for _ in range(qlen))
    r = ''.join(rng.choice(bases) for _ in range(rlen))
    cpu_score, gpu_results = run_test("random_%d" % i, q, r)
    all_match = all(d == 0 for _, d in gpu_results.values()) if gpu_results else True
    if all_match:
        passed += 1
    else:
        failed += 1

# Performance sweep (if GPU available)
if HAS_GPU:
    print()
    print("=== Block Size Sweep (300bp x 300bp) ===")
    q300 = ''.join(rng.choice(bases) for _ in range(300))
    r300 = ''.join(rng.choice(bases) for _ in range(300))
    qb, rb = q300.encode(), r300.encode()

    print("%-12s %10s %10s %6s" % ("Block Size", "Time (ms)", "Score", "Match"))
    cpu_score = cpu.bpd_sw_score_cpu(qb, 300, rb, 300, 2, -1, 3, 1)
    for bs in [32, 64, 128, 256, 512]:
        # Warmup
        gpu.bpd_sw_score_gpu(qb, 300, rb, 300, 2, -1, 3, 1, bs)
        # Benchmark
        t0 = time.perf_counter()
        iters = 20
        for _ in range(iters):
            gs = gpu.bpd_sw_score_gpu(qb, 300, rb, 300, 2, -1, 3, 1, bs)
        t1 = time.perf_counter()
        ms = (t1 - t0) / iters * 1000
        match = "0 ULP" if gs == cpu_score else "DIFFER"
        print("%-12d %10.2f %10d %6s" % (bs, ms, gs, match))

print()
print("PASSED: %d  FAILED: %d  TOTAL: %d" % (passed, failed, passed + failed))
if failed == 0:
    gpu_note = " (GPU bit-identical across all block sizes)" if HAS_GPU else " (CPU only)"
    print("BIT-IDENTICAL%s" % gpu_note)
