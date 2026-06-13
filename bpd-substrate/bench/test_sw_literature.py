#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Test Smith-Waterman on well-known biological sequence pairs."""
import ctypes, os, time

os.system("gcc -O2 -shared -fPIC -o /tmp/bpd_sw_cpu.so bench/bpd_smith_waterman.c")
cpu = ctypes.CDLL("/tmp/bpd_sw_cpu.so")
cpu.bpd_sw_score_cpu.argtypes = [
    ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
cpu.bpd_sw_score_cpu.restype = ctypes.c_int

try:
    gpu = ctypes.CDLL("/tmp/bpd_sw_gpu.so")
    gpu.bpd_sw_score_gpu.argtypes = [
        ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    gpu.bpd_sw_score_gpu.restype = ctypes.c_int
    HAS_GPU = True
except:
    HAS_GPU = False

def read_fasta(path):
    seq = []
    with open(path) as f:
        for line in f:
            if not line.startswith(">"): seq.append(line.strip().upper())
    return "".join(seq).replace("N", "A")

def test_pair(name, path_a, path_b, match=2, mismatch=-1, gap_open=3, gap_extend=1):
    a = read_fasta(path_a)
    b = read_fasta(path_b)
    ab, bb = a.encode(), b.encode()
    
    t0 = time.perf_counter()
    cpu_score = cpu.bpd_sw_score_cpu(ab, len(a), bb, len(b), match, mismatch, gap_open, gap_extend)
    t_cpu = (time.perf_counter() - t0) * 1000
    
    gpu_str = ""
    if HAS_GPU:
        t0 = time.perf_counter()
        gpu_score = gpu.bpd_sw_score_gpu(ab, len(a), bb, len(b), match, mismatch, gap_open, gap_extend, 32)
        t_gpu = (time.perf_counter() - t0) * 1000
        match_str = "0 ULP" if gpu_score == cpu_score else "DIFFER"
        gpu_str = "  GPU: %.1fms %s" % (t_gpu, match_str)
    
    print("  %-40s %4dbp x %4dbp  score=%5d  CPU: %.1fms%s" % (
        name, len(a), len(b), cpu_score, t_cpu, gpu_str))

print("=== Literature Test Vectors ===")
print()

base = "/tmp/genomics_test"

test_pair("E.coli vs Salmonella 16S rRNA",
    base+"/ecoli_16s_rrna.fasta", base+"/salmonella_16s_rrna.fasta")

test_pair("BRCA1 human 1kb self-align",
    base+"/brca1_human_1kb.fasta", base+"/brca1_human_1kb.fasta")

test_pair("Sperm whale vs Human myoglobin",
    base+"/human_myoglobin.fasta", base+"/whale_myoglobin.fasta")

test_pair("Human vs Bovine insulin",
    base+"/human_insulin.fasta", base+"/bovine_insulin.fasta")

test_pair("E.coli 16S vs Human insulin (neg ctrl)",
    base+"/ecoli_16s_rrna.fasta", base+"/human_insulin.fasta")

print()
print("=== Complete ===")
