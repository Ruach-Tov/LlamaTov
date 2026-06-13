#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Verify collapsed Gibbs motif discovery — CPU vs GPU.

Plants a known motif in random sequences, runs Gibbs sampling,
verifies the motif is discovered. Compares CPU and GPU results.
"""
import ctypes, os, random, time, math

# Build CPU reference
os.system("gcc -O2 -shared -fPIC -o /tmp/bpd_pwm.so bench/bpd_pwm_score.c -lm")
cpu = ctypes.CDLL("/tmp/bpd_pwm.so")

cpu.bpd_pwm_score_all.argtypes = [
    ctypes.c_char_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ctypes.POINTER(ctypes.c_float)]

cpu.bpd_gibbs_step.argtypes = [
    ctypes.POINTER(ctypes.c_char_p), ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_float,
    ctypes.POINTER(ctypes.c_uint32)]
cpu.bpd_gibbs_step.restype = ctypes.c_int

# Try GPU
try:
    gpu = ctypes.CDLL("/tmp/bpd_gibbs_gpu.so")

    class GibbsConfig(ctypes.Structure):
        _fields_ = [("n_seqs", ctypes.c_int),
                     ("motif_width", ctypes.c_int),
                     ("max_seq_len", ctypes.c_int),
                     ("pseudocount", ctypes.c_float),
                     ("block_size", ctypes.c_int)]

    gpu.bpd_gibbs_round_gpu.argtypes = [
        ctypes.c_char_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(GibbsConfig)]

    gpu.bpd_gibbs_sampler_gpu.argtypes = [
        ctypes.c_char_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(GibbsConfig), ctypes.c_int]

    HAS_GPU = True
    print("GPU Gibbs sampler loaded")
except:
    HAS_GPU = False
    print("GPU not available")

def plant_motif(n_seqs, seq_len, motif, mutation_rate=0.1, seed=42):
    """Generate random sequences with a planted motif."""
    rng = random.Random(seed)
    bases = "ACGT"
    sequences = []
    true_positions = []
    
    for _ in range(n_seqs):
        seq = list(''.join(rng.choice(bases) for _ in range(seq_len)))
        pos = rng.randint(0, seq_len - len(motif))
        # Plant motif with mutations
        for j, base in enumerate(motif):
            if rng.random() < mutation_rate:
                seq[pos + j] = rng.choice(bases)
            else:
                seq[pos + j] = base
        sequences.append(''.join(seq))
        true_positions.append(pos)
    
    return sequences, true_positions

def run_cpu_gibbs(sequences, motif_width, n_iterations, seed=123):
    """Run CPU Gibbs sampling."""
    n = len(sequences)
    rng = random.Random(seed)
    
    # Initialize random positions
    positions = [rng.randint(0, len(s) - motif_width) for s in sequences]
    background = [0.25, 0.25, 0.25, 0.25]
    bg_arr = (ctypes.c_float * 4)(*background)
    rng_state = ctypes.c_uint32(seed)
    
    seq_ptrs = (ctypes.c_char_p * n)(*[s.encode() for s in sequences])
    seq_lens = (ctypes.c_int * n)(*[len(s) for s in sequences])
    pos_arr = (ctypes.c_int * n)(*positions)
    
    for it in range(n_iterations):
        for hold_out in range(n):
            new_pos = cpu.bpd_gibbs_step(
                seq_ptrs, n, seq_lens, pos_arr,
                motif_width, hold_out,
                bg_arr, ctypes.c_float(0.1),
                ctypes.byref(rng_state))
            pos_arr[hold_out] = new_pos
    
    return list(pos_arr)

def run_gpu_gibbs(sequences, motif_width, n_iterations, seed=123):
    """Run GPU collapsed Gibbs sampling."""
    n = len(sequences)
    rng = random.Random(seed)
    
    # Concatenate sequences
    concat = ''.join(sequences)
    offsets = []
    off = 0
    for s in sequences:
        offsets.append(off)
        off += len(s)
    
    max_len = max(len(s) for s in sequences)
    positions = [rng.randint(0, len(s) - motif_width) for s in sequences]
    rng_states = [seed + i for i in range(n)]
    
    d_offsets = (ctypes.c_int * n)(*offsets)
    d_lengths = (ctypes.c_int * n)(*[len(s) for s in sequences])
    d_positions = (ctypes.c_int * n)(*positions)
    d_rng = (ctypes.c_uint32 * n)(*rng_states)
    
    config = GibbsConfig(n_seqs=n, motif_width=motif_width,
                          max_seq_len=max_len, pseudocount=0.1,
                          block_size=128)
    
    gpu.bpd_gibbs_sampler_gpu(
        concat.encode(), len(concat),
        d_offsets, d_lengths, d_positions, d_rng,
        ctypes.byref(config), n_iterations)
    
    return list(d_positions)

print()
print("=== Test 1: Planted motif discovery (CPU) ===")
motif = "GATTACA"
seqs, true_pos = plant_motif(10, 200, motif, mutation_rate=0.1)
print("  Planted '%s' in 10 × 200bp sequences" % motif)
print("  True positions: %s" % true_pos)

t0 = time.perf_counter()
cpu_pos = run_cpu_gibbs(seqs, len(motif), 100)
t_cpu = (time.perf_counter() - t0) * 1000
print("  CPU discovered:  %s (%.1f ms)" % (cpu_pos, t_cpu))

# Check how many are within 2bp of true position
cpu_hits = sum(1 for cp, tp in zip(cpu_pos, true_pos) if abs(cp - tp) <= 2)
print("  CPU accuracy: %d/%d within 2bp of true position" % (cpu_hits, len(seqs)))

if HAS_GPU:
    print()
    print("=== Test 2: Planted motif discovery (GPU) ===")
    t0 = time.perf_counter()
    gpu_pos = run_gpu_gibbs(seqs, len(motif), 100)
    t_gpu = (time.perf_counter() - t0) * 1000
    print("  GPU discovered:  %s (%.1f ms)" % (gpu_pos, t_gpu))
    
    gpu_hits = sum(1 for gp, tp in zip(gpu_pos, true_pos) if abs(gp - tp) <= 2)
    print("  GPU accuracy: %d/%d within 2bp of true position" % (gpu_hits, len(seqs)))
    print("  Speedup: %.1fx" % (t_cpu / t_gpu) if t_gpu > 0 else "  N/A")

    print()
    print("=== Test 3: Stronger motif, more sequences ===")
    motif2 = "TATAAAT"  # TATA box — biologically real
    seqs2, true_pos2 = plant_motif(20, 500, motif2, mutation_rate=0.05)
    
    t0 = time.perf_counter()
    gpu_pos2 = run_gpu_gibbs(seqs2, len(motif2), 200)
    t_gpu2 = (time.perf_counter() - t0) * 1000
    
    gpu_hits2 = sum(1 for gp, tp in zip(gpu_pos2, true_pos2) if abs(gp - tp) <= 2)
    print("  TATA box in 20 × 500bp: %d/%d found (%.1f ms)" % (gpu_hits2, len(seqs2), t_gpu2))

    print()
    print("=== Test 4: Performance scaling ===")
    for n_seqs in [10, 50, 100]:
        seqs_perf, _ = plant_motif(n_seqs, 500, "GATTACA", seed=n_seqs)
        
        t0 = time.perf_counter()
        run_gpu_gibbs(seqs_perf, 7, 50)
        t = (time.perf_counter() - t0) * 1000
        print("  %3d seqs × 500bp × 50 iters: %.1f ms" % (n_seqs, t))

print()
print("=== Complete ===")
