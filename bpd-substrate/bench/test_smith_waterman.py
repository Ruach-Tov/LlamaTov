#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Verify Smith-Waterman CPU implementation against known results."""
import ctypes, os

os.system("gcc -O2 -shared -fPIC -o /tmp/bpd_sw.so bench/bpd_smith_waterman.c")

lib = ctypes.CDLL("/tmp/bpd_sw.so")
lib.bpd_sw_score_cpu.argtypes = [
    ctypes.c_char_p, ctypes.c_int,
    ctypes.c_char_p, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int
]
lib.bpd_sw_score_cpu.restype = ctypes.c_int

class SWResult(ctypes.Structure):
    _fields_ = [
        ("score", ctypes.c_int),
        ("query_end", ctypes.c_int),
        ("ref_end", ctypes.c_int),
        ("cigar_len", ctypes.c_int),
        ("cigar", ctypes.c_char * 4096),
    ]

lib.bpd_smith_waterman_cpu.argtypes = [
    ctypes.c_char_p, ctypes.c_int,
    ctypes.c_char_p, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(SWResult)
]

def sw_align(query, ref, match=2, mismatch=-1, gap_open=3, gap_extend=1):
    result = SWResult()
    lib.bpd_smith_waterman_cpu(
        query.encode(), len(query),
        ref.encode(), len(ref),
        match, mismatch, gap_open, gap_extend,
        ctypes.byref(result))
    return result.score, result.query_end, result.ref_end, result.cigar.decode()

print("=== Smith-Waterman CPU Verification ===")
print()

# Test 1: Perfect match
score, qe, re, cigar = sw_align("ACGT", "ACGT")
print("Test 1: Perfect match")
print("  query=ACGT  ref=ACGT")
print("  score=%d  cigar=%s  PASS=%s" % (score, cigar, score == 8))

# Test 2: One mismatch
score, qe, re, cigar = sw_align("ACGT", "ACTT")
print("Test 2: One mismatch")
print("  query=ACGT  ref=ACTT")
print("  score=%d  cigar=%s" % (score, cigar))

# Test 3: Longer alignment with gaps
score, qe, re, cigar = sw_align("ATAGACGACATGGGGC", "ATAGACATGGCGC")
print("Test 3: With gaps")
print("  query=ATAGACGACATGGGGC  ref=ATAGACATGGCGC")
print("  score=%d  cigar=%s" % (score, cigar))

# Test 4: No alignment
score, qe, re, cigar = sw_align("AAAA", "CCCC")
print("Test 4: No alignment")
print("  query=AAAA  ref=CCCC")
print("  score=%d  PASS=%s" % (score, score == 0))

# Test 5: Classic textbook example
query = "TGTTACGG"
ref   = "GGTTGACTA"
score, qe, re, cigar = sw_align(query, ref)
print("Test 5: Textbook (TGTTACGG vs GGTTGACTA)")
print("  score=%d  cigar=%s  qend=%d  rend=%d" % (score, cigar, qe, re))

# Test 6: Longer real-ish sequences
import random
random.seed(42)
bases = "ACGT"
q = ''.join(random.choice(bases) for _ in range(100))
r = ''.join(random.choice(bases) for _ in range(120))
score, qe, re, cigar = sw_align(q, r)
print("Test 6: Random 100bp vs 120bp")
print("  score=%d  cigar_len=%d" % (score, len(cigar)))

print()
print("=== All tests complete ===")
