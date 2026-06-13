#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Verify GPU radix sort and GPU suffix array against CPU reference."""
import ctypes, os, time, random

# Build CPU reference
os.system("gcc -O2 -shared -fPIC -o /tmp/bpd_sa.so bench/bpd_suffix_array.c")
cpu = ctypes.CDLL("/tmp/bpd_sa.so")
cpu.bpd_build_suffix_array.argtypes = [ctypes.c_char_p, ctypes.c_int,
                                        ctypes.POINTER(ctypes.c_int)]

# Try loading GPU
try:
    gpu = ctypes.CDLL("/tmp/bpd_radix_sort_gpu.so")
    gpu.bpd_radix_sort_gpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_int),
        ctypes.c_int, ctypes.c_int]
    gpu.bpd_build_suffix_array_gpu.argtypes = [
        ctypes.c_char_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    HAS_GPU = True
    print("GPU radix sort loaded")
except:
    HAS_GPU = False
    print("GPU not available — CPU only")

print()
print("=== Test 1: GPU Radix Sort (raw key-value sort) ===")

if HAS_GPU:
    passed = 0
    rng = random.Random(42)
    for trial in range(10):
        n = rng.randint(100, 5000)
        keys_list = [rng.randint(0, 2**32-1) for _ in range(n)]
        vals_list = list(range(n))

        keys = (ctypes.c_uint32 * n)(*keys_list)
        vals = (ctypes.c_int * n)(*vals_list)

        gpu.bpd_radix_sort_gpu(keys, vals, n, 128)

        # Verify sorted
        sorted_keys = list(keys)
        is_sorted = all(sorted_keys[i] <= sorted_keys[i+1] for i in range(n-1))
        # Verify values are a permutation
        is_perm = sorted(list(vals)) == list(range(n))
        # Verify key-value correspondence
        kv_correct = all(keys_list[vals[i]] == sorted_keys[i] for i in range(n))

        if is_sorted and is_perm and kv_correct:
            passed += 1
        else:
            print("  FAIL trial %d (n=%d): sorted=%s perm=%s kv=%s" % (
                trial, n, is_sorted, is_perm, kv_correct))

    print("  %d/10 random sort tests PASSED" % passed)

print()
print("=== Test 2: GPU Suffix Array vs CPU Reference ===")

if HAS_GPU:
    passed = 0
    tests = ["banana", "mississippi", "ACGTACGTACGT", "ATAGACGACATGGGGC"]

    # Add random DNA sequences
    rng = random.Random(42)
    for _ in range(5):
        slen = rng.randint(50, 500)
        tests.append(''.join(rng.choice("ACGT") for _ in range(slen)))

    for text in tests:
        n = len(text)
        cpu_sa = (ctypes.c_int * n)()
        gpu_sa = (ctypes.c_int * n)()

        cpu.bpd_build_suffix_array(text.encode(), n, cpu_sa)
        gpu.bpd_build_suffix_array_gpu(text.encode(), n, gpu_sa, 128)

        cpu_list = list(cpu_sa)
        gpu_list = list(gpu_sa)

        if cpu_list == gpu_list:
            passed += 1
            status = "PASS"
        else:
            # Check how many positions match
            matches = sum(1 for a, b in zip(cpu_list, gpu_list) if a == b)
            status = "PARTIAL (%d/%d match)" % (matches, n)

        label = text[:25] + ("..." if len(text) > 25 else "")
        print("  %s %-30s (n=%d)" % (status, label, n))

    print("  %d/%d suffix array tests" % (passed, len(tests)))

print()
print("=== Test 3: Performance — GPU radix sort ===")

if HAS_GPU:
    for n in [1000, 10000, 100000]:
        keys_list = [rng.randint(0, 2**32-1) for _ in range(n)]
        vals_list = list(range(n))
        keys = (ctypes.c_uint32 * n)(*keys_list)
        vals = (ctypes.c_int * n)(*vals_list)

        # Warmup
        gpu.bpd_radix_sort_gpu(keys, vals, n, 128)

        # Benchmark
        keys = (ctypes.c_uint32 * n)(*keys_list)
        vals = (ctypes.c_int * n)(*vals_list)
        t0 = time.perf_counter()
        iters = 5
        for _ in range(iters):
            keys_copy = (ctypes.c_uint32 * n)(*keys_list)
            vals_copy = (ctypes.c_int * n)(*vals_list)
            gpu.bpd_radix_sort_gpu(keys_copy, vals_copy, n, 128)
        t1 = time.perf_counter()
        ms = (t1 - t0) / iters * 1000
        meps = n / ((t1 - t0) / iters) / 1e6
        print("  n=%6d  %.2f ms  %.1f M elem/s" % (n, ms, meps))

print()
print("=== Complete ===")
