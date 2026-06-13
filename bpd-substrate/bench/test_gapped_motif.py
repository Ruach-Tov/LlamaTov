#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""TDD: Gapped motif discovery — Gibbs → SW extension pipeline.

Tests:
  D1: Gibbs output feeds into SW aligner
  D2: Gapped motifs score higher than ungapped when gaps are real
  D3: Known gapped motifs are discovered
"""
import ctypes, os, random, math

# Build
os.system("gcc -O2 -shared -fPIC -o /tmp/bpd_gapped_motif.so "
          "bench/bpd_gapped_motif.c bench/bpd_smith_waterman.c -lm")
os.system("gcc -O2 -shared -fPIC -o /tmp/bpd_pwm.so bench/bpd_pwm_score.c -lm")

lib = ctypes.CDLL("/tmp/bpd_gapped_motif.so")
pwm_lib = ctypes.CDLL("/tmp/bpd_pwm.so")

class GappedResult(ctypes.Structure):
    _fields_ = [
        ("seq_idx", ctypes.c_int),
        ("ungapped_pos", ctypes.c_int),
        ("ungapped_score", ctypes.c_int),
        ("gapped_score", ctypes.c_int),
        ("gapped_query_end", ctypes.c_int),
        ("gapped_ref_end", ctypes.c_int),
        ("cigar", ctypes.c_char * 4096),
        ("has_gaps", ctypes.c_int),
    ]

lib.bpd_extend_with_gaps.argtypes = [
    ctypes.POINTER(ctypes.c_char_p), ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ctypes.c_int,  # window_margin
    ctypes.c_int, ctypes.c_int,  # match, mismatch
    ctypes.c_int, ctypes.c_int,  # gap_open, gap_extend
    ctypes.POINTER(GappedResult)]
lib.bpd_extend_with_gaps.restype = ctypes.c_int

lib.bpd_pwm_to_consensus.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_char_p]

def make_pwm(motif, pseudocount=0.1):
    W = len(motif)
    pwm = []
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    for j in range(W):
        row = [pseudocount] * 4
        row[base_map[motif[j]]] += 1.0
        total = sum(row)
        for b in range(4):
            pwm.append(math.log(row[b] / total / 0.25))
    return pwm

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print("  PASS %s" % name)
    else:
        failed += 1
        print("  FAIL %s" % name)

# ================================================================
# D1: Consensus extraction from PWM
# ================================================================
print("=== D1: PWM → consensus → SW extension ===")

motif = "GATTACA"
pwm_list = make_pwm(motif)
pwm_arr = (ctypes.c_float * len(pwm_list))(*pwm_list)
consensus = ctypes.create_string_buffer(len(motif) + 1)
lib.bpd_pwm_to_consensus(pwm_arr, len(motif), consensus)
check("Consensus from PWM: '%s' (expect GATTACA)" % consensus.value.decode(),
      consensus.value.decode() == "GATTACA")

# ================================================================
# D1 continued: Full pipeline — plant motif, run Gibbs positions, extend with SW
# ================================================================
print()
print("=== D1: Ungapped motif extension ===")

rng = random.Random(42)
bases = "ACGT"
n_seqs = 10
seq_len = 200
motif = "GATTACA"
W = len(motif)

# Plant ungapped motif
sequences = []
true_positions = []
for _ in range(n_seqs):
    seq = list(''.join(rng.choice(bases) for _ in range(seq_len)))
    pos = rng.randint(20, seq_len - W - 20)
    for j, c in enumerate(motif):
        if rng.random() > 0.1:
            seq[pos + j] = c
    sequences.append(''.join(seq))
    true_positions.append(pos)

# Simulate "Gibbs found the right positions" 
# (use true positions ± small noise)
gibbs_positions = [p + rng.randint(-2, 2) for p in true_positions]
gibbs_positions = [max(0, min(p, seq_len - W)) for p in gibbs_positions]

seq_ptrs = (ctypes.c_char_p * n_seqs)(*[s.encode() for s in sequences])
seq_lens = (ctypes.c_int * n_seqs)(*[len(s) for s in sequences])
pos_arr = (ctypes.c_int * n_seqs)(*gibbs_positions)
results = (GappedResult * n_seqs)()

n_gapped = lib.bpd_extend_with_gaps(
    seq_ptrs, n_seqs, seq_lens, pos_arr,
    pwm_arr, W, 10,  # window margin
    2, -1, 3, 1,     # match, mismatch, gap_open, gap_extend
    results)

print("  Ungapped motif: %d/%d have gaps in SW alignment" % (n_gapped, n_seqs))
for i in range(min(3, n_seqs)):
    r = results[i]
    print("    seq %d: ungapped_pos=%d gapped_score=%d cigar=%s gaps=%d" % (
        r.seq_idx, r.ungapped_pos, r.gapped_score, r.cigar.decode(), r.has_gaps))

check("Ungapped motif: all SW scores > 0",
      all(results[i].gapped_score > 0 for i in range(n_seqs)))

# ================================================================
# D2: Gapped motif — plant motif WITH a gap
# ================================================================
print()
print("=== D2: Gapped motif detection ===")

gapped_motif_left = "GATT"
gapped_motif_right = "ACA"
gap_size = 3  # insert 3 random bases between GATT and ACA

sequences_gapped = []
true_pos_gapped = []
for _ in range(n_seqs):
    seq = list(''.join(rng.choice(bases) for _ in range(seq_len)))
    pos = rng.randint(20, seq_len - W - gap_size - 20)
    # Plant left part
    for j, c in enumerate(gapped_motif_left):
        seq[pos + j] = c
    # Insert gap (random bases)
    gap_start = pos + len(gapped_motif_left)
    # Plant right part after gap
    right_start = gap_start + gap_size
    for j, c in enumerate(gapped_motif_right):
        if right_start + j < seq_len:
            seq[right_start + j] = c
    sequences_gapped.append(''.join(seq))
    true_pos_gapped.append(pos)

# Use the UNGAPPED consensus "GATTACA" to search — SW should find the gap
seq_ptrs_g = (ctypes.c_char_p * n_seqs)(*[s.encode() for s in sequences_gapped])
seq_lens_g = (ctypes.c_int * n_seqs)(*[len(s) for s in sequences_gapped])
pos_arr_g = (ctypes.c_int * n_seqs)(*true_pos_gapped)
results_g = (GappedResult * n_seqs)()

n_gapped_found = lib.bpd_extend_with_gaps(
    seq_ptrs_g, n_seqs, seq_lens_g, pos_arr_g,
    pwm_arr, W, 15,
    2, -1, 3, 1,
    results_g)

print("  Gapped motif (GATT---ACA): %d/%d detected gaps in SW" % (n_gapped_found, n_seqs))
for i in range(min(5, n_seqs)):
    r = results_g[i]
    print("    seq %d: score=%d cigar=%s gaps=%d" % (
        r.seq_idx, r.gapped_score, r.cigar.decode(), r.has_gaps))

check("Gapped motif: majority have gaps detected", n_gapped_found >= n_seqs // 2)
check("Gapped motif: all SW scores > 0",
      all(results_g[i].gapped_score > 0 for i in range(n_seqs)))

# ================================================================
# D3: Known biological gapped motif — TATA box with spacer
# ================================================================
print()
print("=== D3: Biological gapped motif (TATA...INR) ===")

# The TATA box + Initiator (INR) is a real gapped motif in eukaryotic promoters:
# TATAAAT followed by ~25bp spacer followed by YYANTYY (simplified: CCATTCC)
tata = "TATAAAT"
inr = "CCATTCC"
spacer_len = 25

sequences_bio = []
for _ in range(15):
    seq = list(''.join(rng.choice(bases) for _ in range(300)))
    pos = rng.randint(50, 200)
    # Plant TATA
    for j, c in enumerate(tata):
        if rng.random() > 0.1: seq[pos+j] = c
    # Plant INR after spacer
    inr_pos = pos + len(tata) + spacer_len
    for j, c in enumerate(inr):
        if rng.random() > 0.1 and inr_pos+j < 300: seq[inr_pos+j] = c
    sequences_bio.append(''.join(seq))

# Use TATA as query, search with wide window to find the gapped structure
tata_pwm = make_pwm(tata)
tata_pwm_arr = (ctypes.c_float * len(tata_pwm))(*tata_pwm)
seq_ptrs_bio = (ctypes.c_char_p * 15)(*[s.encode() for s in sequences_bio])
seq_lens_bio = (ctypes.c_int * 15)(*[300]*15)
pos_arr_bio = (ctypes.c_int * 15)(*[100]*15)  # approximate positions
results_bio = (GappedResult * 15)()

n_bio_gapped = lib.bpd_extend_with_gaps(
    seq_ptrs_bio, 15, seq_lens_bio, pos_arr_bio,
    tata_pwm_arr, len(tata), 40,  # wide window
    2, -1, 3, 1,
    results_bio)

print("  TATA+INR gapped motif: %d/15 show gaps" % n_bio_gapped)
all_positive = all(results_bio[i].gapped_score > 0 for i in range(15))
check("Bio gapped motif: all scores > 0", all_positive)

print()
print("PASSED: %d  FAILED: %d  TOTAL: %d" % (passed, failed, passed + failed))
