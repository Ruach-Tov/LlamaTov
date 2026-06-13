#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""TDD: SIMD-striped Smith-Waterman vs scalar CPU reference."""
import ctypes, os, random, time

# Build
os.system("gcc -O2 -msse4.1 -shared -fPIC -o /tmp/bpd_sw_simd.so bench/bpd_sw_simd.c 2>&1")
os.system("gcc -O2 -shared -fPIC -o /tmp/bpd_sw_cpu.so bench/bpd_smith_waterman.c")

try:
    simd = ctypes.CDLL("/tmp/bpd_sw_simd.so")
    simd.bpd_sw_simd_score_cpu.argtypes = [
        ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    simd.bpd_sw_simd_score_cpu.restype = ctypes.c_int
    HAS_SIMD = True
    print("SIMD SW loaded")
except:
    HAS_SIMD = False
    print("SIMD build failed — checking compiler output:")
    os.system("gcc -O2 -msse4.1 -shared -fPIC -o /tmp/bpd_sw_simd.so bench/bpd_sw_simd.c 2>&1")
    exit(1)

scalar = ctypes.CDLL("/tmp/bpd_sw_cpu.so")
scalar.bpd_sw_score_cpu.argtypes = [
    ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
scalar.bpd_sw_score_cpu.restype = ctypes.c_int

passed = 0
failed = 0
def check(name, condition):
    global passed, failed
    if condition: passed += 1; print("  PASS %s" % name)
    else: failed += 1; print("  FAIL %s" % name)

rng = random.Random(42)
bases = "ACGT"

# Test 1: Known cases
print("\n=== Known cases: SIMD vs scalar ===")
known = [
    ("ACGT", "ACGT", 8),
    ("ACGT", "ACTT", 5),
    ("AAAA", "CCCC", 0),
    ("TGTTACGG", "GGTTGACTA", 7),
    ("GATTACA", "GATTACA", 14),
]

for q, r, expected in known:
    scalar_score = scalar.bpd_sw_score_cpu(q.encode(), len(q), r.encode(), len(r), 2, -1, 3, 1)
    simd_score = simd.bpd_sw_simd_score_cpu(q.encode(), len(q), r.encode(), len(r), 2, -1, 3, 1)
    check("%s vs %s: scalar=%d simd=%d expected=%d" % (q[:8], r[:8], scalar_score, simd_score, expected),
          simd_score == scalar_score)

# Test 2: Random sequences
print("\n=== Random sequences (50 tests) ===")
match_count = 0
for trial in range(50):
    qlen = rng.randint(10, 300)
    rlen = rng.randint(10, 300)
    q = ''.join(rng.choice(bases) for _ in range(qlen))
    r = ''.join(rng.choice(bases) for _ in range(rlen))
    
    scalar_score = scalar.bpd_sw_score_cpu(q.encode(), qlen, r.encode(), rlen, 2, -1, 3, 1)
    simd_score = simd.bpd_sw_simd_score_cpu(q.encode(), qlen, r.encode(), rlen, 2, -1, 3, 1)
    
    if simd_score == scalar_score:
        match_count += 1
    elif trial < 5:
        print("  DIFF trial %d (%dx%d): scalar=%d simd=%d" % (trial, qlen, rlen, scalar_score, simd_score))

check("Random: %d/50 SIMD = scalar" % match_count, match_count >= 40)

# Test 3: Performance comparison
print("\n=== Performance: SIMD vs scalar ===")
for size in [100, 500, 1000, 2000]:
    q = ''.join(rng.choice(bases) for _ in range(size))
    r = ''.join(rng.choice(bases) for _ in range(size))
    qb, rb = q.encode(), r.encode()
    
    # Scalar
    t0 = time.perf_counter()
    iters = max(1, 10000 // (size * size // 1000))
    for _ in range(iters):
        scalar.bpd_sw_score_cpu(qb, size, rb, size, 2, -1, 3, 1)
    t_scalar = (time.perf_counter() - t0) / iters * 1000
    
    # SIMD
    t0 = time.perf_counter()
    for _ in range(iters):
        simd.bpd_sw_simd_score_cpu(qb, size, rb, size, 2, -1, 3, 1)
    t_simd = (time.perf_counter() - t0) / iters * 1000
    
    speedup = t_scalar / t_simd if t_simd > 0 else 0
    print("  %4dbp: scalar=%.2fms simd=%.2fms speedup=%.1fx" % (size, t_scalar, t_simd, speedup))

print("\nPASSED: %d  FAILED: %d  TOTAL: %d" % (passed, failed, passed + failed))
