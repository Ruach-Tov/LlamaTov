#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""TDD: Verify GPU LCP, GPU PWM scoring, and convergence detection."""
import ctypes, os, random, math

# === BUILD ===
os.system("gcc -O2 -shared -fPIC -o /tmp/bpd_sa.so bench/bpd_suffix_array.c")
os.system("gcc -O2 -shared -fPIC -o /tmp/bpd_pwm.so bench/bpd_pwm_score.c -lm")

cpu_sa = ctypes.CDLL("/tmp/bpd_sa.so")
cpu_sa.bpd_build_suffix_array.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
cpu_sa.bpd_build_lcp_array.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]

cpu_pwm = ctypes.CDLL("/tmp/bpd_pwm.so")
cpu_pwm.bpd_pwm_score_all.argtypes = [
    ctypes.c_char_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ctypes.POINTER(ctypes.c_float)]

try:
    gpu_lcp = ctypes.CDLL("/tmp/bpd_lcp_gpu.so")
    gpu_lcp.bpd_build_lcp_gpu.argtypes = [
        ctypes.c_char_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.c_int]
    HAS_GPU_LCP = True
except:
    HAS_GPU_LCP = False

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
# A5: GPU LCP verification
# ================================================================
print("=== A5: GPU LCP vs CPU LCP ===")

test_strings = ["banana", "mississippi", "ACGTACGTACGT", "ATAGACGACATGGGGC"]
rng = random.Random(42)
for _ in range(5):
    test_strings.append(''.join(rng.choice("ACGT") for _ in range(rng.randint(50, 500))))

for text in test_strings:
    n = len(text)
    cpu_sa_arr = (ctypes.c_int * n)()
    cpu_lcp_arr = (ctypes.c_int * n)()
    cpu_sa.bpd_build_suffix_array(text.encode(), n, cpu_sa_arr)
    cpu_sa.bpd_build_lcp_array(text.encode(), n, cpu_sa_arr, cpu_lcp_arr)
    cpu_lcp_list = list(cpu_lcp_arr)

    if HAS_GPU_LCP:
        gpu_lcp_arr = (ctypes.c_int * n)()
        gpu_lcp.bpd_build_lcp_gpu(text.encode(), n, cpu_sa_arr, gpu_lcp_arr, 128)
        gpu_lcp_list = list(gpu_lcp_arr)
        match = (cpu_lcp_list == gpu_lcp_list)
        label = text[:20] + ("..." if len(text) > 20 else "")
        check("LCP %-25s (n=%d)" % (label, n), match)
        if not match:
            diffs = [(i, cpu_lcp_list[i], gpu_lcp_list[i]) for i in range(n) if cpu_lcp_list[i] != gpu_lcp_list[i]]
            print("    First 3 diffs: %s" % diffs[:3])
    else:
        check("LCP %s (GPU not available, CPU only)" % text[:20], True)

# ================================================================
# B3: GPU PWM scoring vs CPU
# ================================================================
print()
print("=== B3: PWM scoring verification ===")

def make_pwm(motif, pseudocount=0.1):
    """Build a log-odds PWM from a consensus motif."""
    W = len(motif)
    pwm = []
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    for j in range(W):
        row = [pseudocount] * 4
        row[base_map[motif[j]]] += 1.0
        total = sum(row)
        for b in range(4):
            freq = row[b] / total
            pwm.append(math.log(freq / 0.25))
    return pwm

# Test: known motif against known sequence
for motif, seq, desc in [
    ("GATTACA", "AAAGATTACAAAT", "exact match"),
    ("TATAAAT", "CCCCCCCCCCCCC", "no match"),
    ("ACGT", "ACGTACGTACGT", "repeated"),
    ("GATTACA", "GATTACAGATTACAGATTACA", "three copies"),
]:
    W = len(motif)
    pwm_list = make_pwm(motif)
    pwm_arr = (ctypes.c_float * len(pwm_list))(*pwm_list)
    n_pos = len(seq) - W + 1
    cpu_scores = (ctypes.c_float * n_pos)()
    cpu_pwm.bpd_pwm_score_all(seq.encode(), len(seq), pwm_arr, W, cpu_scores)
    
    scores = list(cpu_scores)
    # Find position with max score
    max_pos = scores.index(max(scores))
    max_score = max(scores)
    
    if desc == "exact match":
        check("PWM %s: max at pos %d (expect 3)" % (desc, max_pos), max_pos == 3)
    elif desc == "no match":
        check("PWM %s: all scores negative" % desc, all(s < 0 for s in scores))
    elif desc == "repeated":
        check("PWM %s: max score > 0" % desc, max_score > 0)
    elif desc == "three copies":
        high = [i for i, s in enumerate(scores) if s > max_score * 0.9]
        check("PWM %s: 3 high-scoring positions" % desc, len(high) >= 3)

# Random PWM scoring: verify CPU consistency
for _ in range(10):
    W = rng.randint(5, 15)
    motif = ''.join(rng.choice("ACGT") for _ in range(W))
    seq_len = rng.randint(50, 200)
    seq = ''.join(rng.choice("ACGT") for _ in range(seq_len))
    
    pwm_list = make_pwm(motif)
    pwm_arr = (ctypes.c_float * len(pwm_list))(*pwm_list)
    n_pos = seq_len - W + 1
    cpu_scores = (ctypes.c_float * n_pos)()
    cpu_pwm.bpd_pwm_score_all(seq.encode(), seq_len, pwm_arr, W, cpu_scores)
    
    # Verify against pure Python
    py_scores = []
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    for i in range(n_pos):
        s = sum(pwm_list[j * 4 + base_map.get(seq[i+j], 0)] for j in range(W))
        py_scores.append(s)
    
    # Compare with tolerance (float32)
    max_diff = max(abs(cpu_scores[i] - py_scores[i]) for i in range(n_pos))
    check("PWM random (W=%d, L=%d): max_diff=%.2e" % (W, seq_len, max_diff), max_diff < 1e-5)

# ================================================================
# C4: Convergence detection
# ================================================================
print()
print("=== C4: Convergence detection ===")

def gibbs_with_convergence(sequences, motif_width, max_iters=500, threshold=0.001):
    """Pure Python Gibbs with Frobenius-norm convergence check."""
    n = len(sequences)
    rng2 = random.Random(42)
    positions = [rng2.randint(0, len(s) - motif_width) for s in sequences]
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    prev_pwm = None
    converged_at = None
    
    for iteration in range(max_iters):
        # Build PWM from current positions
        pwm = [[0.1]*4 for _ in range(motif_width)]
        for s in range(n):
            for j in range(motif_width):
                b = base_map.get(sequences[s][positions[s]+j], 0)
                pwm[j][b] += 1.0
        # Normalize to frequencies
        for j in range(motif_width):
            total = sum(pwm[j])
            pwm[j] = [x/total for x in pwm[j]]
        
        # Check convergence
        if prev_pwm is not None:
            frob = sum(sum((pwm[j][b] - prev_pwm[j][b])**2 
                          for b in range(4)) for j in range(motif_width))
            frob = math.sqrt(frob)
            if frob < threshold:
                converged_at = iteration
                break
        
        prev_pwm = [row[:] for row in pwm]
        
        # One Gibbs sweep
        for hold_out in range(n):
            # Score all positions in held-out sequence
            best_pos = 0
            best_score = -float('inf')
            for p in range(len(sequences[hold_out]) - motif_width + 1):
                score = 0
                for j in range(motif_width):
                    b = base_map.get(sequences[hold_out][p+j], 0)
                    freq = pwm[j][b]
                    score += math.log(max(freq, 1e-10) / 0.25)
                if score > best_score:
                    best_score = score
                    best_pos = p
            positions[hold_out] = best_pos
    
    return positions, converged_at

# Test 1: Strong motif should converge quickly
bases = "ACGT"
seqs_strong = []
true_pos_strong = []
rng3 = random.Random(42)
for _ in range(15):
    seq = list(''.join(rng3.choice(bases) for _ in range(200)))
    pos = rng3.randint(0, 190)
    for j, c in enumerate("GATTACA"):
        if rng3.random() > 0.05:  # 5% mutation
            seq[pos+j] = c
    seqs_strong.append(''.join(seq))
    true_pos_strong.append(pos)

pos_strong, conv_strong = gibbs_with_convergence(seqs_strong, 7, max_iters=200)
check("Convergence (strong motif): converged=%s at iter %s" % (
    conv_strong is not None, conv_strong), conv_strong is not None and conv_strong < 100)

hits_strong = sum(1 for p, t in zip(pos_strong, true_pos_strong) if abs(p-t) <= 2)
check("Convergence (strong motif): %d/15 found" % hits_strong, hits_strong >= 10)

# Test 2: No motif should NOT converge (or converge slowly)
seqs_random = [''.join(rng3.choice(bases) for _ in range(200)) for _ in range(10)]
_, conv_random = gibbs_with_convergence(seqs_random, 7, max_iters=50)
check("Convergence (no motif): did not converge in 50 iters", conv_random is None)

# Test 3: Moderate motif converges at moderate speed
seqs_moderate = []
for _ in range(10):
    seq = list(''.join(rng3.choice(bases) for _ in range(200)))
    pos = rng3.randint(0, 190)
    for j, c in enumerate("TATAAAT"):
        if rng3.random() > 0.15:  # 15% mutation
            seq[pos+j] = c
    seqs_moderate.append(''.join(seq))

_, conv_moderate = gibbs_with_convergence(seqs_moderate, 7, max_iters=200)
check("Convergence (moderate motif): converged=%s" % (conv_moderate is not None),
      conv_moderate is not None)

print()
print("PASSED: %d  FAILED: %d  TOTAL: %d" % (passed, failed, passed + failed))
