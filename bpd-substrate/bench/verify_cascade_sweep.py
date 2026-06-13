#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""verify_cascade_sweep.py — Bit-identity sweep across all cascade-reduction
pattern instantiations.

Per Heath's direction 2026-05-20 ~20:00 UTC: "make porting the full
SIMD-8 × ILP-4 × 4-level cascade implementation as a sweepable pattern
for our code generator/optimizer."

For each valid (SimdWidth, IlpFactor, CascadeDepth, CascadeBase) tuple,
this harness:
  1. Calls the generated C kernel by name (via ctypes dispatch)
  2. Compares the kernel's output to PyTorch CPU's torch.sum at multiple sizes
  3. Reports the per-size ULP table

The instantiation that's 0 ULP at every tested size is the one that matches
PyTorch's exact algorithm for the current CPU's SIMD capabilities. PyTorch
CPU on AVX1 hardware (Tesla P4 enclave) should match cascade(8, 4, 4, 16).

Output:
  - Per-pattern row: pattern | n=100 | n=256 | n=1000 | n=1024 | n=4096 | n=16384
  - Summary: how many patterns hit 0 ULP at all sizes
  - Highlighted: the matching pattern for this CPU

Usage:
    make bench/cascade_kernels_generated.c    # generate the C
    gcc -O2 -shared -fPIC -o build/cascade.so bench/cascade_kernels_generated.c
    python3 bench/verify_cascade_sweep.py
"""
import ctypes
import os
import sys
import numpy as np

try:
    import torch
except ImportError:
    sys.exit("error: pip install torch numpy")

torch.backends.mkldnn.enabled = False
torch.backends.cudnn.enabled = False
torch.set_num_threads(1)


def ulp(a, b):
    """Sign-magnitude ULP distance between two float32 scalars."""
    a = np.asarray(a, dtype=np.float32).reshape(1)
    b = np.asarray(b, dtype=np.float32).reshape(1)
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    BASE = np.int64(0x80000000)
    ai = np.where(ai < 0, BASE - ai, ai)
    bi = np.where(bi < 0, BASE - bi, bi)
    return int(np.abs(ai - bi).max())


def load_dispatch(so_path):
    """Load the cascade library and bind the dispatch table.

    Returns (lib, kernel_count, names_list, dispatch_array)
    where dispatch_array is a ctypes array of function pointers.
    """
    lib = ctypes.CDLL(so_path)

    # cascade_dispatch_count is an int symbol
    count_sym = ctypes.c_int.in_dll(lib, "cascade_dispatch_count")
    kernel_count = count_sym.value

    # cascade_dispatch_names is char**[kernel_count]
    names_arr = (ctypes.c_char_p * kernel_count).in_dll(lib, "cascade_dispatch_names")
    names = [names_arr[i].decode() for i in range(kernel_count)]

    # cascade_dispatch is fnptr[kernel_count]
    # The function signature is: float (*)(const float* data, int n)
    KERNEL_FN = ctypes.CFUNCTYPE(ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_int)
    dispatch_arr = (KERNEL_FN * kernel_count).in_dll(lib, "cascade_dispatch")
    return lib, kernel_count, names, dispatch_arr


def parse_pattern(name):
    """Parse 'cascade_sum_simd8_ilp4_depth4_base16' → (8, 4, 4, 16)."""
    parts = name.split("_")
    sw = int(parts[2].removeprefix("simd"))
    ilp = int(parts[3].removeprefix("ilp"))
    cd = int(parts[4].removeprefix("depth"))
    cb = int(parts[5].removeprefix("base"))
    return (sw, ilp, cd, cb)


def main():
    so_path = os.environ.get("CASCADE_SO", "build/cascade.so")
    if not os.path.exists(so_path):
        sys.exit(f"error: {so_path} not found. Build it first:\n"
                 f"  python3 bench/generate_cascade_kernels.py > bench/cascade_kernels_generated.c\n"
                 f"  gcc -O2 -shared -fPIC -o {so_path} bench/cascade_kernels_generated.c -lm")

    print(f"Loading {so_path}...")
    lib, count, names, dispatch = load_dispatch(so_path)
    print(f"  {count} cascade kernels available")

    # Test sizes — cover small (n=100, no SIMD path), boundary (n=256, 1000),
    # power-of-2 (1024, 4096), and large (16384).
    sizes = [100, 256, 1000, 1024, 4096, 16384]

    # Pre-compute reference inputs and PyTorch outputs
    refs = {}
    rng = np.random.default_rng(42)
    for n in sizes:
        x = rng.standard_normal(n).astype(np.float32)
        # PyTorch CPU reference
        pt = torch.sum(torch.from_numpy(x)).numpy()
        refs[n] = (x, pt)

    # Build the table
    print()
    header_sizes = "  ".join(f"n={n:>5}" for n in sizes)
    print(f"{'pattern':<48}  {header_sizes}  match")
    print("─" * (48 + 2 + len(header_sizes) + 8))

    matches_all = []
    matches_powers = []  # match only at power-of-2

    for k in range(count):
        name = names[k]
        pattern = parse_pattern(name)
        ulps = []
        for n in sizes:
            x, pt = refs[n]
            xp = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            result = dispatch[k](xp, n)
            ulps.append(ulp(pt, result))

        all_zero = all(u == 0 for u in ulps)
        pow_zero = all(u == 0 for n, u in zip(sizes, ulps) if (n & (n-1)) == 0)
        marker = " ← BIT_IDENTICAL_ALL" if all_zero else (" (power-of-2 only)" if pow_zero and not all_zero else "")
        ulps_str = "  ".join(f"{u:>7}" for u in ulps)
        # Trim the name to fit
        short_name = name.replace("cascade_sum_", "")
        print(f"{short_name:<48}  {ulps_str}  {marker}")

        if all_zero:
            matches_all.append((name, pattern))
        elif pow_zero:
            matches_powers.append((name, pattern))

    print()
    print("═" * 80)
    print(f"Summary:")
    print(f"  {len(matches_all)} patterns BIT_IDENTICAL at all sizes")
    print(f"  {len(matches_powers)} patterns BIT_IDENTICAL only at power-of-2 sizes")
    print(f"  {count - len(matches_all) - len(matches_powers)} patterns DIVERGENT")
    print()
    if matches_all:
        print(f"Patterns matching PyTorch CPU bit-for-bit at every tested size:")
        for name, pat in matches_all:
            print(f"  cascade(SW={pat[0]}, ILP={pat[1]}, CD={pat[2]}, CB={pat[3]})  — {name}")
    return 0 if matches_all else 1


if __name__ == "__main__":
    sys.exit(main())
