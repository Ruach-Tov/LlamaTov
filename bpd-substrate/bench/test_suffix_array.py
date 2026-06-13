#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Verify suffix array + LCP construction against Python reference."""
import ctypes, os

os.system("gcc -O2 -shared -fPIC -o /tmp/bpd_sa.so bench/bpd_suffix_array.c")
lib = ctypes.CDLL("/tmp/bpd_sa.so")

lib.bpd_build_suffix_array.argtypes = [ctypes.c_char_p, ctypes.c_int,
                                        ctypes.POINTER(ctypes.c_int)]
lib.bpd_build_lcp_array.argtypes = [ctypes.c_char_p, ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_int),
                                     ctypes.POINTER(ctypes.c_int)]

class MotifCandidate(ctypes.Structure):
    _fields_ = [("sa_start", ctypes.c_int), ("sa_end", ctypes.c_int),
                ("length", ctypes.c_int), ("count", ctypes.c_int)]

lib.bpd_find_repeat_candidates.argtypes = [
    ctypes.c_char_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(MotifCandidate), ctypes.c_int]
lib.bpd_find_repeat_candidates.restype = ctypes.c_int

lib.bpd_get_candidate_string.argtypes = [
    ctypes.c_char_p, ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(MotifCandidate), ctypes.c_char_p, ctypes.c_int]

def py_suffix_array(text):
    """Pure Python suffix array for verification."""
    n = len(text)
    return sorted(range(n), key=lambda i: text[i:])

def py_lcp(text, sa):
    """Pure Python LCP array."""
    n = len(sa)
    rank = [0] * n
    for i in range(n): rank[sa[i]] = i
    lcp = [0] * n
    k = 0
    for i in range(n):
        if rank[i] == 0: k = 0; continue
        j = sa[rank[i] - 1]
        while i+k < n and j+k < n and text[i+k] == text[j+k]: k += 1
        lcp[rank[i]] = k
        if k > 0: k -= 1
    return lcp

def c_suffix_array(text):
    n = len(text)
    sa = (ctypes.c_int * n)()
    lib.bpd_build_suffix_array(text.encode(), n, sa)
    return list(sa)

def c_lcp(text, sa):
    n = len(text)
    sa_arr = (ctypes.c_int * n)(*sa)
    lcp = (ctypes.c_int * n)()
    lib.bpd_build_lcp_array(text.encode(), n, sa_arr, lcp)
    return list(lcp)

def c_find_repeats(text, sa, lcp, min_len=3, min_count=2):
    n = len(text)
    sa_arr = (ctypes.c_int * n)(*sa)
    lcp_arr = (ctypes.c_int * n)(*lcp)
    max_cands = 1000
    cands = (MotifCandidate * max_cands)()
    found = lib.bpd_find_repeat_candidates(
        text.encode(), n, sa_arr, lcp_arr, min_len, min_count, cands, max_cands)
    
    results = []
    buf = ctypes.create_string_buffer(256)
    for i in range(found):
        lib.bpd_get_candidate_string(text.encode(), sa_arr,
                                      ctypes.byref(cands[i]), buf, 256)
        results.append((buf.value.decode(), cands[i].count, cands[i].length))
    return results

print("=== Suffix Array + LCP Verification ===")
print()

passed = 0
failed = 0

# Test 1: Simple string
for text in ["banana", "mississippi", "abcabcabc", "ACGTACGTACGT",
             "ATAGACGACATGGGGCATAGACATGGCGC"]:
    py_sa = py_suffix_array(text)
    c_sa = c_suffix_array(text)
    py_l = py_lcp(text, py_sa)
    c_l = c_lcp(text, c_sa)
    
    sa_match = (py_sa == c_sa)
    lcp_match = (py_l == c_l)
    
    if sa_match and lcp_match:
        passed += 1
        status = "PASS"
    else:
        failed += 1
        status = "FAIL"
    print("  %s %-30s SA=%s LCP=%s" % (status, text[:25], sa_match, lcp_match))

# Test 2: Motif discovery on sequences with planted motifs
print()
print("=== Motif Candidate Discovery ===")

# Plant motif "GATTACA" three times in random sequence
import random
rng = random.Random(42)
bases = "ACGT"
bg = ''.join(rng.choice(bases) for _ in range(200))
planted = bg[:50] + "GATTACA" + bg[50:100] + "GATTACA" + bg[100:150] + "GATTACA" + bg[150:]
sa = c_suffix_array(planted)
lcp = c_lcp(planted, sa)
repeats = c_find_repeats(planted, sa, lcp, min_len=5, min_count=3)

found_gattaca = any("GATTACA" in r[0] for r in repeats)
print("  Planted motif: GATTACA (3 copies in 221bp)")
print("  Found repeats (len>=5, count>=3):")
for motif, count, length in repeats[:10]:
    tag = " ← PLANTED MOTIF" if "GATTACA" in motif else ""
    print("    '%s' (len=%d, count=%d)%s" % (motif, length, count, tag))
if found_gattaca:
    passed += 1
    print("  PASS: planted motif discovered")
else:
    failed += 1
    print("  FAIL: planted motif not found")

# Test 3: Real E. coli 16S — find repeated subsequences
print()
print("=== E. coli 16S rRNA — repeat discovery ===")
try:
    ecoli = ""
    with open("/tmp/genomics_test/ecoli_16s_rrna.fasta") as f:
        for line in f:
            if not line.startswith(">"): ecoli += line.strip().upper()
    ecoli = ecoli.replace("N", "A")
    
    sa = c_suffix_array(ecoli)
    lcp = c_lcp(ecoli, sa)
    repeats = c_find_repeats(ecoli, sa, lcp, min_len=8, min_count=3)
    
    print("  Sequence: %d bp" % len(ecoli))
    print("  Repeats found (len>=8, count>=3): %d" % len(repeats))
    for motif, count, length in repeats[:8]:
        print("    '%s' (len=%d, count=%d)" % (motif[:30], length, count))
    if repeats:
        passed += 1
        print("  PASS: repeats found in real data")
    else:
        failed += 1
except FileNotFoundError:
    print("  SKIP: E. coli FASTA not found (run on enclave)")

print()
print("PASSED: %d  FAILED: %d" % (passed, failed))
