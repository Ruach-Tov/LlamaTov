#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Verify BPD Smith-Waterman C implementation against pure Python reference.

Tests across:
  - Known textbook cases
  - Random sequences (many lengths)
  - Edge cases (empty, single base, all same, all different)
  - Varied gap penalties
  - Real-ish biological motifs

Every test compares C score against Python reference.
0 ULP = identical integer scores.
"""
import ctypes, os, random

os.system("gcc -O2 -shared -fPIC -o /tmp/bpd_sw.so bench/bpd_smith_waterman.c")
lib = ctypes.CDLL("/tmp/bpd_sw.so")
lib.bpd_sw_score_cpu.argtypes = [
    ctypes.c_char_p, ctypes.c_int,
    ctypes.c_char_p, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int
]
lib.bpd_sw_score_cpu.restype = ctypes.c_int

def c_sw(query, ref, match=2, mismatch=-1, gap_open=3, gap_extend=1):
    return lib.bpd_sw_score_cpu(
        query.encode(), len(query),
        ref.encode(), len(ref),
        match, mismatch, gap_open, gap_extend)

def py_sw(query, ref, match=2, mismatch=-1, gap_open=3, gap_extend=1):
    m, n = len(query), len(ref)
    H = [[0]*(n+1) for _ in range(m+1)]
    E = [[0]*(n+1) for _ in range(m+1)]
    F = [[0]*(n+1) for _ in range(m+1)]
    max_score = 0
    for i in range(1, m+1):
        for j in range(1, n+1):
            s = match if query[i-1] == ref[j-1] else mismatch
            E[i][j] = max(H[i][j-1] - gap_open, E[i][j-1] - gap_extend)
            F[i][j] = max(H[i-1][j] - gap_open, F[i-1][j] - gap_extend)
            H[i][j] = max(0, H[i-1][j-1] + s, E[i][j], F[i][j])
            if H[i][j] > max_score: max_score = H[i][j]
    return max_score

passed = 0
failed = 0

def check(name, query, ref, **kwargs):
    global passed, failed
    c = c_sw(query, ref, **kwargs)
    p = py_sw(query, ref, **kwargs)
    if c == p:
        passed += 1
    else:
        failed += 1
        print("  FAIL %s: C=%d Python=%d  q=%s r=%s" % (name, c, p, query[:30], ref[:30]))

print("=== Smith-Waterman: C vs Python Reference ===")
print()

# 1. Textbook cases
print("Textbook cases...")
check("perfect_match", "ACGT", "ACGT")
check("one_mismatch", "ACGT", "ACTT")
check("no_alignment", "AAAA", "CCCC")
check("textbook_1", "TGTTACGG", "GGTTGACTA")
check("gap_in_query", "ATAGACGACATGGGGC", "ATAGACATGGCGC")

# 2. Edge cases
print("Edge cases...")
check("single_match", "A", "A")
check("single_mismatch", "A", "C")
check("query_longer", "ACGTACGTACGT", "ACGT")
check("ref_longer", "ACGT", "ACGTACGTACGT")
check("all_A", "AAAAAA", "AAAAAA")
check("poly_vs_poly", "AAAAAA", "TTTTTT")
check("alternating", "ACACAC", "ACACAC")
check("reverse_complement", "ACGT", "TGCA")

# 3. Random sequences — many sizes
print("Random sequences (100 tests)...")
rng = random.Random(42)
bases = "ACGT"
for i in range(100):
    qlen = rng.randint(10, 200)
    rlen = rng.randint(10, 200)
    q = ''.join(rng.choice(bases) for _ in range(qlen))
    r = ''.join(rng.choice(bases) for _ in range(rlen))
    check("random_%d_%dx%d" % (i, qlen, rlen), q, r)

# 4. Varied gap penalties
print("Varied gap penalties...")
q = "ACGTACGTACGTACGT"
r = "ACGTAAAACGTACGT"
for go in [1, 2, 3, 5, 10]:
    for ge in [1, 2, 3]:
        check("gaps_%d_%d" % (go, ge), q, r, gap_open=go, gap_extend=ge)

# 5. Varied scoring
print("Varied scoring...")
q = "ACGTACGT"
r = "ACTTACGT"
for m in [1, 2, 3, 5]:
    for mm in [-1, -2, -3]:
        check("score_%d_%d" % (m, mm), q, r, match=m, mismatch=mm)

# 6. Biological motifs
print("Biological motifs...")
check("kozak", "GCCACCATGG", "GCCGCCATGG")
check("tata_box", "TATAAAT", "TATAAATG")
check("splice_donor", "GTAAGT", "GTATGT")
check("poly_a", "AAAAAAAAAAAA", "TTAAAAAAAAAAATT")
check("microsatellite", "CAGCAGCAGCAG", "CAGCAGCAGCAGCAG")

print()
print("PASSED: %d  FAILED: %d  TOTAL: %d" % (passed, failed, passed + failed))
if failed == 0:
    print("BIT-IDENTICAL with Python reference across all %d tests" % passed)
