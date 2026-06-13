#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""matrix_verify.py — Cross-backend output comparison utility.

Part of the cross-language correctness matrix harness. Used by per-
backend tests to compare a computed output against either:
  - An expected fixture file (PyTorch reference)
  - Another backend's computed output (cross-backend pairing)

THE MATRIX HAS TWO ORTHOGONAL CORRECTNESS CONTRACTS:

  1. allclose (default): NUMERICAL correctness. Tolerant of fp32
     reordering. Detects catastrophic numerical bugs (NaN, scale
     errors, sign errors). Passes when implementations compute the
     same FUNCTION within fp32 precision but possibly different
     accumulation orders.

  2. --strict (bit-identical): SEMANTIC correctness. Compares uint32
     IEEE 754 bits via XOR. Detects layout/order/algorithm divergence
     that allclose silently absorbs. Passes only when implementations
     compute the exact same operation in the exact same order on the
     exact same data layout.

Per mavchin's empirical finding (intercom 11:10 UTC 2026-05-17): a
matrix transpose bug was caught by bit-identical comparison against
Ollama — allclose missed it because the transpose-wrong outputs were
still in plausible numerical range. The two contracts serve different
correctness questions.

Usage:
  python3 matrix_verify.py PATH_A PATH_B
    Default: allclose mode (rtol=1e-5, atol=1e-6).
    Exit 0 if np.allclose(a, b), else 1.

  python3 matrix_verify.py --strict PATH_A PATH_B
    Bit-identical mode: XOR the uint32 bits. Exit 0 iff EVERY element
    is bit-equal. Diagnostic shows per-element XOR + ULP distance on
    mismatch via mavchin's ieee754_hex.dump_comparison.

  python3 matrix_verify.py --reference PATH_REF PATH_A [PATH_B ...]
    Verify each PATH_A, PATH_B, ... against PATH_REF (allclose mode).
    Add --strict for bit-identical reference check.

Author: metayen 2026-05-17
Per Heath's cross-language correctness matrix vision (1.c.i) and
mavchin's bit-identical spike (intercom 11:10 UTC). The --strict mode
delegates to mavchin's authoritative ieee754_hex.py utilities at the
bpd/ root rather than duplicating the bit-inspection logic.
"""

import os
import sys
import argparse
import numpy as np

# Add bpd/ to path so we can import mavchin's ieee754_hex module.
_BPD_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if _BPD_DIR not in sys.path:
    sys.path.insert(0, _BPD_DIR)
import ieee754_hex


# Tolerance constants for the allclose contract. Matches
# cpu_references.py and the existing matrix conventions for fp32
# numerical comparison.
RTOL = 1e-5
ATOL = 1e-6


def compare_arrays(a: np.ndarray, b: np.ndarray, label_a: str = 'a',
                   label_b: str = 'b') -> bool:
    """Compare two arrays under the ALLCLOSE contract.

    Returns True if numerically equal within (rtol=1e-5, atol=1e-6).
    Prints a diagnostic on mismatch. See compare_arrays_strict for the
    bit-identical contract.
    """
    if a.shape != b.shape:
        print(f"  SHAPE MISMATCH: {label_a}={a.shape}  {label_b}={b.shape}")
        return False
    if a.dtype != b.dtype:
        print(f"  DTYPE MISMATCH: {label_a}={a.dtype}  {label_b}={b.dtype}")
        # Not necessarily fatal — try comparison anyway with cast.
    if np.allclose(a, b, rtol=RTOL, atol=ATOL):
        return True

    # Mismatch — print diagnostic.
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    max_diff_idx = np.unravel_index(diff.argmax(), diff.shape)
    print(f"  NUMERICAL MISMATCH (rtol={RTOL}, atol={ATOL})")
    print(f"    max abs diff:     {diff.max():.6e}")
    print(f"    at index:         {max_diff_idx}")
    print(f"    {label_a}[{max_diff_idx}]: {a[max_diff_idx]:.6e}")
    print(f"    {label_b}[{max_diff_idx}]: {b[max_diff_idx]:.6e}")
    n_diverge = (diff > (ATOL + RTOL * np.abs(b))).sum()
    print(f"    elements diverging: {n_diverge} / {a.size}")
    return False


def ulp_distance_f32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-element ULP distance for two float32 arrays.

    ULP (Units in the Last Place) is the canonical human-readable
    metric for "how far apart are these floats?" Two adjacent
    representable float32 values differ by exactly 1 ULP, regardless
    of magnitude. Two identical floats are 0 ULPs apart.

    Algorithm: treat the bit pattern as a signed integer in
    sign-magnitude form, then take the integer difference.
    Implementation detail: IEEE 754 floats with the same sign are
    monotonic when their bits are reinterpreted as uint32, so
    |bits_a - bits_b| gives the ULP count for same-sign values.
    Different-sign requires the sign-magnitude conversion.

    Returns ndarray of int64 ULP distances, same shape as a.
    """
    # Reinterpret as int32 sign-magnitude.
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    # Convert sign-magnitude to two's complement equivalent (so subtraction
    # gives correct ULP count across sign boundaries).
    SIGN_BIT = np.int64(0x80000000)
    ai = np.where(ai < 0, SIGN_BIT - ai, ai)
    bi = np.where(bi < 0, SIGN_BIT - bi, bi)
    return np.abs(ai - bi)


def compare_arrays_strict(a: np.ndarray, b: np.ndarray,
                          label_a: str = 'a', label_b: str = 'b',
                          ulp_tolerance: int = 0) -> bool:
    """Compare two arrays under the BIT-IDENTICAL or BOUNDED-ULP contract.

    Returns True iff:
      - ulp_tolerance=0 (default): every element is bit-equal
      - ulp_tolerance=N: every element's ULP distance is ≤ N

    Layout, dtype, and shape must match exactly in all cases.

    The ULP-tolerance variant captures the THIRD correctness contract
    (between allclose and bit-identical) named by mavchin's empirical
    boundary: transcendentals (silu/gelu/tanh) are ≤2 ULP against
    reference (SFU precision differs CPU vs GPU); rms_norm/layer_norm
    are ≤13 ULP (reduction order). Bit-equality is physically
    impossible for these families; ULP-bounded is the right contract.

    On mismatch, prints diagnostic showing per-element XOR (via
    mavchin's ieee754_hex.dump_comparison) and per-element ULP
    distance for the worst offenders.
    """
    if a.shape != b.shape:
        print(f"  SHAPE MISMATCH: {label_a}={a.shape}  {label_b}={b.shape}")
        return False
    if a.dtype != b.dtype:
        print(f"  DTYPE MISMATCH: {label_a}={a.dtype}  {label_b}={b.dtype}")
        return False

    # Bit-level comparison via numpy's view trick.
    if a.dtype == np.float32:
        a_bits = a.view(np.uint32)
        b_bits = b.view(np.uint32)
        ulps = ulp_distance_f32(a, b)
    elif a.dtype == np.float64:
        a_bits = a.view(np.uint64)
        b_bits = b.view(np.uint64)
        # ULP for f64 not implemented; bit-identical only for now.
        ulps = None
    else:
        # For non-float types, byte equality is the contract.
        if np.array_equal(a, b):
            return True
        print(f"  BIT MISMATCH on non-float dtype {a.dtype}")
        return False

    if ulp_tolerance == 0:
        # Bit-identical contract.
        if np.array_equal(a_bits, b_bits):
            return True
        contract_label = "BIT-IDENTICAL"
    else:
        # Bounded-ULP contract.
        if ulps is None:
            print(f"  ULP-bounded contract not yet supported for {a.dtype}")
            return False
        if (ulps <= ulp_tolerance).all():
            return True
        contract_label = f"ULP≤{ulp_tolerance}"

    # Mismatch — diagnostic.
    print(f"  {contract_label} MISMATCH ({label_a} vs {label_b})")
    ieee754_hex.dump_comparison(f"{label_a} ⊕ {label_b}", a, b, max_show=5)
    if ulps is not None:
        worst_ulp = ulps.max()
        n_over = (ulps > ulp_tolerance).sum() if ulp_tolerance > 0 \
                                              else (a_bits != b_bits).sum()
        print(f"    max ULP distance: {worst_ulp}")
        print(f"    elements over tolerance: {n_over} / {a.size}")
    else:
        n_diverge = (a_bits != b_bits).sum()
        print(f"    elements diverging: {n_diverge} / {a.size}")
    return False


def compare_files(path_a: str, path_b: str, strict: bool = False,
                  ulp_tolerance: int = 0) -> bool:
    """Compare two .npy files. Returns True if equal under the chosen contract.

    Args:
        path_a, path_b: paths to .npy files to compare.
        strict: if True, use bit-identical (or ULP-bounded) contract;
                else allclose contract.
        ulp_tolerance: if strict and >0, accept up to N ULPs difference.
                       Use 0 for true bit-identity; N>0 for the
                       bounded-ULP contract (e.g., 2 for transcendentals,
                       13 for normalization per mavchin's measurements).
    """
    a = np.load(path_a)
    b = np.load(path_b)
    if strict:
        contract = f"ULP≤{ulp_tolerance}" if ulp_tolerance > 0 else "BIT-IDENTICAL"
    else:
        contract = "allclose"
    print(f"Comparing ({contract}):")
    print(f"  {path_a}  shape={a.shape}  dtype={a.dtype}")
    print(f"  {path_b}  shape={b.shape}  dtype={b.dtype}")
    if strict:
        return compare_arrays_strict(a, b, path_a, path_b,
                                      ulp_tolerance=ulp_tolerance)
    else:
        return compare_arrays(a, b, path_a, path_b)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--reference', type=str, default=None,
                        help='Reference .npy file; all PATHS are compared to it')
    parser.add_argument('--strict', action='store_true',
                        help='Use bit-identical contract (uint32 XOR) instead '
                             'of the default allclose contract. Catches '
                             'layout/order/algorithm divergence that allclose '
                             'absorbs.')
    parser.add_argument('--ulp', type=int, default=0, metavar='N',
                        help='Bounded-ULP contract: accept up to N ULPs '
                             'difference (implies --strict). 0 = bit-identical. '
                             'Per mavchin\'s empirical kernel-family boundary: '
                             'use 2 for transcendentals (silu/gelu/tanh), '
                             '13 for normalization (rms_norm/layer_norm). '
                             'Bit-equality is physically impossible for these '
                             'families; the bounded-ULP contract is correct.')
    parser.add_argument('paths', nargs='+', type=str,
                        help='One or more .npy file paths (interpretation '
                             'depends on --reference)')
    args = parser.parse_args()

    # --ulp implies --strict (ULP-bounded is a strict-side contract).
    if args.ulp > 0:
        args.strict = True

    if args.strict:
        contract = f"ULP≤{args.ulp}" if args.ulp > 0 else "BIT-IDENTICAL"
    else:
        contract = "allclose"

    if args.reference is not None:
        # Compare each path to the reference.
        if len(args.paths) < 1:
            parser.error("--reference requires at least one PATH")
        ref = np.load(args.reference)
        print(f"Reference ({contract}): {args.reference}  "
              f"shape={ref.shape}  dtype={ref.dtype}")
        all_pass = True
        for path in args.paths:
            print()
            other = np.load(path)
            print(f"  vs {path}  shape={other.shape}  dtype={other.dtype}")
            if args.strict:
                ok = compare_arrays_strict(ref, other,
                                            label_a='reference', label_b=path,
                                            ulp_tolerance=args.ulp)
            else:
                ok = compare_arrays(ref, other,
                                    label_a='reference', label_b=path)
            print(f"  → {'MATCH' if ok else 'DIVERGE'}")
            all_pass = all_pass and ok
        sys.exit(0 if all_pass else 1)
    else:
        # Pairwise: exactly 2 paths.
        if len(args.paths) != 2:
            parser.error("without --reference, exactly 2 PATHS required")
        ok = compare_files(args.paths[0], args.paths[1],
                           strict=args.strict, ulp_tolerance=args.ulp)
        print(f"→ {'MATCH' if ok else 'DIVERGE'}")
        sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
