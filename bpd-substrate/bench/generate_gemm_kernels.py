#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""generate_gemm_kernels.py — Emit C for every valid gemm_pattern instantiation.

Per Heath's direction 2026-05-20 ~22:00 UTC: "By making a sweepable
generator that subsumes even our kernel BPD, we consolidated the
intelligence into tighter crystalline structures."

This is the substrate's parametric Goto GEMM generator. One C function
per (P, Q, UM, UN, SIMD, KRule) instantiation. The sweep harness
(bench/verify_gemm_sweep.py) compiles each kernel and runs it against
cblas_sgemm at multiple (M, N, K) shapes to:

  1. Identify which patterns are BIT_IDENTICAL with OpenBLAS at all shapes
  2. Identify configurations OpenBLAS never explored that achieve the
     same bit-exact result (substrate-design alternates)
  3. Measure GFLOPS per pattern for performance comparison

PyTorch CPU on AVX1 (Tesla P4 enclave) = gemm(768, 384, 16, 4, 8, adaptive_half).
Verified at commit 50f1f4d.

The parameter space is declared in lib/gemm_kernel.pl gemm_pattern/6.
This generator mirrors that parameter space.

Output: bench/gemm_kernels_generated.c (one function per pattern + dispatch table)

Usage:
    python3 bench/generate_gemm_kernels.py > bench/gemm_kernels_generated.c
"""
import sys
from itertools import product


# ─── Parameter space ──────────────────────────────────────────────────────
#
# Mirror of lib/gemm_kernel.pl gemm_pattern/6.

PS = [1, 64, 128, 256, 384, 512, 768, 1024]
QS = [1, 64, 128, 192, 240, 248, 256, 384, 512]
UMS = [1, 4, 8, 16, 32]
UNS = [1, 4, 8, 12, 16]
SIMDS = [1, 4, 8, 16]
KRULES = ['single_block', 'fixed_q', 'adaptive_half', 'equal_split']


def is_valid(p, q, um, un, simd, krule):
    """Mirror of valid_gemm_pattern/6 in lib/gemm_kernel.pl."""
    if p % um != 0:
        return False
    if um % simd != 0:
        return False
    row_regs = um // simd
    tile_regs = row_regs * un + 2 + un
    if tile_regs > 16:
        return False
    l1_bytes = q * (um + un) * 4
    if l1_bytes > 32768:
        return False
    l2_bytes = p * q * 4
    if l2_bytes > 4194304:
        return False
    return True


def enumerate_patterns():
    for p, q, um, un, simd, krule in product(PS, QS, UMS, UNS, SIMDS, KRULES):
        if is_valid(p, q, um, un, simd, krule):
            yield (p, q, um, un, simd, krule)


def kernel_name(p, q, um, un, simd, krule):
    return f"gemm_p{p}_q{q}_um{um}_un{un}_simd{simd}_{krule}"


# ─── K-block tiling rule emitters ─────────────────────────────────────────


def emit_k_setup(q, um, krule):
    """Emit C code that sets `min_l` for the current K-block iteration.

    Returns indented C that runs inside the `while (ls < K)` loop body.
    Uses `rem = K - ls` already in scope.
    """
    if krule == 'single_block':
        return "        int min_l = rem;  /* single_block */"
    if krule == 'fixed_q':
        return f"        int min_l = rem > {q} ? {q} : rem;  /* fixed_q */"
    if krule == 'adaptive_half':
        return (
            "        int min_l;\n"
            f"        if (rem >= 2 * {q}) {{ min_l = {q}; }}\n"
            f"        else if (rem > {q}) {{ min_l = ((rem / 2 + {um} - 1) / {um}) * {um}; }}\n"
            "        else { min_l = rem; }"
        )
    if krule == 'equal_split':
        return (
            "        int min_l;\n"
            "        {\n"
            f"            int nblocks = (K + {q} - 1) / {q};\n"
            "            int per_block = (K + nblocks - 1) / nblocks;\n"
            f"            per_block = ((per_block + {um} - 1) / {um}) * {um};\n"
            "            if (per_block > rem) per_block = rem;\n"
            "            min_l = per_block;\n"
            "        }"
        )
    raise ValueError(f"Unknown krule: {krule}")


def emit_kernel(p, q, um, un, simd, krule):
    """Emit one Goto GEMM kernel for the given pattern.

    Scalar-mimic form (deterministic bits). SimdWidth, UM, UN affect
    only loop nesting and would matter for performance but not for ULP.
    """
    name = kernel_name(p, q, um, un, simd, krule)
    k_setup = emit_k_setup(q, um, krule)
    return f"""// {name}
//   P={p} Q={q} UM={um} UN={un} SIMD={simd} KRule={krule}
void {name}(const float* A, const float* B, float* C,
            int M, int N, int K) {{
    for (int i = 0; i < M * N; ++i) C[i] = 0.0f;
    int ls = 0;
    while (ls < K) {{
        int rem = K - ls;
{k_setup}
        for (int row = 0; row < M; ++row) {{
            for (int col = 0; col < N; ++col) {{
                float partial = 0.0f;
                for (int k = ls; k < ls + min_l; ++k) {{
                    partial += A[row * K + k] * B[k * N + col];
                }}
                C[row * N + col] += partial;
            }}
        }}
        ls += min_l;
    }}
}}
"""


def emit_dispatch_table(patterns):
    names = [kernel_name(*p) for p in patterns]
    lines = [
        "typedef void (*gemm_kernel_fn)(const float* A, const float* B,",
        "                               float* C, int M, int N, int K);",
        "",
        f"const int gemm_dispatch_count = {len(patterns)};",
        "",
        "gemm_kernel_fn gemm_dispatch[] = {",
    ]
    lines.extend(f"    {n}," for n in names)
    lines.append("};")
    lines.append("")
    lines.append("const char* gemm_dispatch_names[] = {")
    lines.extend(f'    "{n}",' for n in names)
    lines.append("};")
    return "\n".join(lines) + "\n"


def main():
    patterns = list(enumerate_patterns())
    print(f"// {len(patterns)} gemm kernel instantiations", file=sys.stderr)
    sys.stdout.write("// =================================================================\n")
    sys.stdout.write("// gemm_kernels_generated.c — auto-generated, do not edit.\n")
    sys.stdout.write("// Generator: bench/generate_gemm_kernels.py\n")
    sys.stdout.write("// Parameter space: lib/gemm_kernel.pl gemm_pattern/6\n")
    sys.stdout.write(f"// Total kernels: {len(patterns)}\n")
    sys.stdout.write("//\n")
    sys.stdout.write("// PyTorch CPU on AVX1 (Tesla P4 enclave) bit-identical match:\n")
    sys.stdout.write("//   gemm_p768_q384_um16_un4_simd8_adaptive_half\n")
    sys.stdout.write("// Verified at commit 50f1f4d.\n")
    sys.stdout.write("// =================================================================\n\n")
    for pat in patterns:
        sys.stdout.write(emit_kernel(*pat))
        sys.stdout.write("\n")
    sys.stdout.write(emit_dispatch_table(patterns))


if __name__ == "__main__":
    main()
