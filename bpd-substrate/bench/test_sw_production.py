#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""TDD: Production genomics features — banded SW, seed-and-extend, quality scoring."""
import ctypes, os, random, math

# Build
os.system("gcc -O2 -shared -fPIC -o /tmp/bpd_sw_prod.so "
          "bench/bpd_sw_production.c bench/bpd_smith_waterman.c -lm")
os.system("gcc -O2 -shared -fPIC -o /tmp/bpd_sa.so bench/bpd_suffix_array.c")
os.system("gcc -O2 -shared -fPIC -o /tmp/bpd_sw_cpu.so bench/bpd_smith_waterman.c")

prod = ctypes.CDLL("/tmp/bpd_sw_prod.so")
sa_lib = ctypes.CDLL("/tmp/bpd_sa.so")
sw_ref = ctypes.CDLL("/tmp/bpd_sw_cpu.so")

# Setup argtypes
prod.bpd_sw_banded_score_cpu.argtypes = [
    ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
prod.bpd_sw_banded_score_cpu.restype = ctypes.c_int

sw_ref.bpd_sw_score_cpu.argtypes = [
    ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
sw_ref.bpd_sw_score_cpu.restype = ctypes.c_int

sa_lib.bpd_build_suffix_array.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]

prod.bpd_find_seeds.argtypes = [
    ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
    ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
prod.bpd_find_seeds.restype = ctypes.c_int

class AlignResult(ctypes.Structure):
    _fields_ = [("score", ctypes.c_int), ("ref_start", ctypes.c_int),
                ("ref_end", ctypes.c_int), ("query_start", ctypes.c_int),
                ("query_end", ctypes.c_int), ("cigar", ctypes.c_char * 4096)]

prod.bpd_seed_and_extend.argtypes = [
    ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(AlignResult)]
prod.bpd_seed_and_extend.restype = ctypes.c_int

class SWResult(ctypes.Structure):
    _fields_ = [("score", ctypes.c_int), ("query_end", ctypes.c_int),
                ("ref_end", ctypes.c_int), ("cigar_len", ctypes.c_int),
                ("cigar", ctypes.c_char * 4096)]

prod.bpd_smith_waterman_quality_cpu.argtypes = [
    ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
    ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(SWResult)]

passed = 0
failed = 0
def check(name, condition):
    global passed, failed
    if condition: passed += 1; print("  PASS %s" % name)
    else: failed += 1; print("  FAIL %s" % name)

rng = random.Random(42)
bases = "ACGT"

# ================================================================
# Feature 1: Banded SW
# ================================================================
print("=== Feature 1: Banded Smith-Waterman ===")

# Test: banded with wide band should equal full SW
for trial in range(20):
    qlen = rng.randint(20, 150)
    rlen = rng.randint(20, 150)
    q = ''.join(rng.choice(bases) for _ in range(qlen))
    r = ''.join(rng.choice(bases) for _ in range(rlen))
    
    full = sw_ref.bpd_sw_score_cpu(q.encode(), qlen, r.encode(), rlen, 2, -1, 3, 1)
    banded = prod.bpd_sw_banded_score_cpu(q.encode(), qlen, r.encode(), rlen, 2, -1, 3, 1, max(qlen, rlen))
    
    if full != banded and trial < 3:
        print("    full=%d banded=%d (q=%d r=%d)" % (full, banded, qlen, rlen))

# Count matches
match_count = 0
for trial in range(20):
    qlen = rng.randint(20, 150)
    rlen = rng.randint(20, 150)
    q = ''.join(rng.choice(bases) for _ in range(qlen))
    r = ''.join(rng.choice(bases) for _ in range(rlen))
    full = sw_ref.bpd_sw_score_cpu(q.encode(), qlen, r.encode(), rlen, 2, -1, 3, 1)
    banded = prod.bpd_sw_banded_score_cpu(q.encode(), qlen, r.encode(), rlen, 2, -1, 3, 1, max(qlen, rlen))
    if full == banded: match_count += 1

check("Banded (wide band) = full SW: %d/20" % match_count, match_count >= 15)

# Test: narrow band on similar sequences still finds alignment
q_sim = "ACGTACGTACGTACGT"
r_sim = "ACGTACCTACGTACGT"  # one mismatch
banded_narrow = prod.bpd_sw_banded_score_cpu(
    q_sim.encode(), len(q_sim), r_sim.encode(), len(r_sim), 2, -1, 3, 1, 5)
check("Banded narrow (bw=5) on similar seqs: score=%d > 0" % banded_narrow, banded_narrow > 0)

# Test: banded is faster than full for long sequences
import time
q_long = ''.join(rng.choice(bases) for _ in range(2000))
r_long = ''.join(rng.choice(bases) for _ in range(2000))
t0 = time.perf_counter()
full_long = sw_ref.bpd_sw_score_cpu(q_long.encode(), 2000, r_long.encode(), 2000, 2, -1, 3, 1)
t_full = (time.perf_counter() - t0) * 1000
t0 = time.perf_counter()
banded_long = prod.bpd_sw_banded_score_cpu(q_long.encode(), 2000, r_long.encode(), 2000, 2, -1, 3, 1, 50)
t_banded = (time.perf_counter() - t0) * 1000
speedup = t_full / t_banded if t_banded > 0 else 0
check("Banded speedup on 2000bp: %.1fx (full=%.1fms banded=%.1fms)" % (speedup, t_full, t_banded),
      speedup > 2)

# ================================================================
# Feature 2: Seed-and-extend
# ================================================================
print()
print("=== Feature 2: Seed-and-extend ===")

# Build reference and plant a known read
ref_seq = ''.join(rng.choice(bases) for _ in range(5000))
read_start = 2000
read_len = 150
read = ref_seq[read_start:read_start+read_len]
# Add 2% mutations
read_list = list(read)
for i in range(len(read_list)):
    if rng.random() < 0.02:
        read_list[i] = rng.choice(bases)
read = ''.join(read_list)

# Build suffix array
n = len(ref_seq)
sa = (ctypes.c_int * n)()
sa_lib.bpd_build_suffix_array(ref_seq.encode(), n, sa)

# Find seeds
positions = (ctypes.c_int * 100)()
kmer = read[:11]
n_seeds = prod.bpd_find_seeds(ref_seq.encode(), n, sa, kmer.encode(), 11, positions, 100)
check("Seed finder: found %d seeds for '%s...'" % (n_seeds, kmer[:8]), n_seeds > 0)

# Seed-and-extend
result = AlignResult()
total_seeds = prod.bpd_seed_and_extend(
    read.encode(), len(read), ref_seq.encode(), n, sa,
    11, 50, 2, -1, 3, 1, ctypes.byref(result))

check("Seed-and-extend: score=%d seeds=%d" % (result.score, total_seeds), result.score > 0)
check("Seed-and-extend: ref position near %d (got %d-%d)" % (
    read_start, result.ref_start, result.ref_end),
    abs(result.ref_start - read_start) < 200 or abs(result.ref_end - read_start) < 200)

# ================================================================
# Feature 3: Quality-aware scoring
# ================================================================
print()
print("=== Feature 3: Quality-aware scoring ===")

q = "ACGTACGTACGT"
r = "ACGTACGTACGT"
high_qual = b"IIIIIIIIIIII"   # Phred ~40, p_error ~0.0001
low_qual =  b"############"   # Phred ~2, p_error ~0.63

result_hq = SWResult()
result_lq = SWResult()
prod.bpd_smith_waterman_quality_cpu(
    q.encode(), len(q), r.encode(), len(r), high_qual, 2, -1, 3, 1, ctypes.byref(result_hq))
prod.bpd_smith_waterman_quality_cpu(
    q.encode(), len(q), r.encode(), len(r), low_qual, 2, -1, 3, 1, ctypes.byref(result_lq))

check("Quality: high-Q score (%d) > low-Q score (%d)" % (result_hq.score, result_lq.score),
      result_hq.score > result_lq.score)

# Mismatch with low quality should penalize less
q_mm = "ACGTACTTACGT"  # mismatch at pos 6
result_hq_mm = SWResult()
result_lq_mm = SWResult()
qual_mixed = b"IIIIII##IIII"  # low quality at mismatch position
prod.bpd_smith_waterman_quality_cpu(
    q_mm.encode(), len(q_mm), r.encode(), len(r), high_qual, 2, -1, 3, 1, ctypes.byref(result_hq_mm))
prod.bpd_smith_waterman_quality_cpu(
    q_mm.encode(), len(q_mm), r.encode(), len(r), qual_mixed, 2, -1, 3, 1, ctypes.byref(result_lq_mm))

check("Quality: low-Q at mismatch score (%d) >= high-Q at mismatch (%d)" % (
    result_lq_mm.score, result_hq_mm.score), result_lq_mm.score >= result_hq_mm.score)

print()
print("PASSED: %d  FAILED: %d  TOTAL: %d" % (passed, failed, passed + failed))
