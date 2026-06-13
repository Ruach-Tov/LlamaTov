#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Test GPU radix sort v2 (split-based, no atomicAdd race)."""
import ctypes, random, time

try:
    gpu = ctypes.CDLL("/tmp/bpd_radix_sort_v2.so")
    gpu.bpd_radix_sort_gpu_v2.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_int),
        ctypes.c_int, ctypes.c_int]
    print("GPU radix sort v2 loaded")
except:
    print("GPU not available")
    exit(1)

rng = random.Random(42)
passed = 0

print("\n=== Correctness (20 random tests) ===")
for trial in range(20):
    n = rng.randint(32, 2000)
    keys_list = [rng.randint(0, 2**32-1) for _ in range(n)]
    vals_list = list(range(n))

    keys = (ctypes.c_uint32 * n)(*keys_list)
    vals = (ctypes.c_int * n)(*vals_list)

    gpu.bpd_radix_sort_gpu_v2(keys, vals, n, 256)

    sorted_keys = list(keys)
    is_sorted = all(sorted_keys[i] <= sorted_keys[i+1] for i in range(n-1))
    is_perm = sorted(list(vals)) == list(range(n))
    kv_ok = all(keys_list[vals[i]] == sorted_keys[i] for i in range(n))

    if is_sorted and is_perm and kv_ok:
        passed += 1
    else:
        print("  FAIL trial %d (n=%d): sorted=%s perm=%s kv=%s" % (
            trial, n, is_sorted, is_perm, kv_ok))

print("  %d/20 PASSED" % passed)

if passed == 20:
    print("\n=== Performance ===")
    for n in [1000, 10000, 100000]:
        keys_list = [rng.randint(0, 2**32-1) for _ in range(n)]
        # Warmup
        keys = (ctypes.c_uint32 * n)(*keys_list)
        vals = (ctypes.c_int * n)(*list(range(n)))
        gpu.bpd_radix_sort_gpu_v2(keys, vals, n, 256)
        # Benchmark
        t0 = time.perf_counter()
        iters = 5
        for _ in range(iters):
            keys = (ctypes.c_uint32 * n)(*keys_list)
            vals = (ctypes.c_int * n)(*list(range(n)))
            gpu.bpd_radix_sort_gpu_v2(keys, vals, n, 256)
        t1 = time.perf_counter()
        ms = (t1 - t0) / iters * 1000
        print("  n=%6d  %.2f ms  %.1f M elem/s" % (n, ms, n / ((t1-t0)/iters) / 1e6))
