#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""generate_cascade_kernels.py — Emit C for every valid cascade-reduction pattern.

Per Heath's direction 2026-05-20 ~20:00 UTC: "make porting the full
SIMD-8 × ILP-4 × 4-level cascade implementation as a sweepable pattern
for our code generator/optimizer."

This generator emits one C function per valid (SimdWidth, IlpFactor,
CascadeDepth, CascadeBase) instantiation. The function is the cascade
reduction algorithm specialized for that pattern. PyTorch CPU on AVX1
hardware corresponds to (8, 4, 4, 16); other instantiations explore the
parameter space.

The parameter space is declared in lib/reduction_kernel.pl as
reduction_pattern/4. This generator enumerates it via swi-prolog and
emits the matching C source.

Output: bench/cascade_kernels_generated.c

Usage:
    python3 bench/generate_cascade_kernels.py > bench/cascade_kernels_generated.c
"""
import subprocess
import sys
from itertools import product


# ─── Parameter space ──────────────────────────────────────────────────────
#
# Mirror of lib/reduction_kernel.pl's reduction_pattern/4. Kept here in
# Python so the generator runs without requiring swipl on PATH (the
# Prolog declaration is the single source of truth for documentation;
# the Python list here is the operational version).

SIMD_WIDTHS = [1, 4, 8, 16]
ILP_FACTORS = [1, 2, 4, 8]
CASCADE_DEPTHS = [1, 2, 4, 8]
CASCADE_BASES = [0, 16, 32, 64]


def is_valid_pattern(sw, ilp, cd, cb):
    """Validity: CascadeBase = 0 iff CascadeDepth = 1."""
    if cd == 1:
        return cb == 0
    return cb > 0


def enumerate_patterns():
    """Yield all valid (sw, ilp, cd, cb) tuples."""
    for sw, ilp, cd, cb in product(SIMD_WIDTHS, ILP_FACTORS, CASCADE_DEPTHS, CASCADE_BASES):
        if is_valid_pattern(sw, ilp, cd, cb):
            yield (sw, ilp, cd, cb)


def kernel_name(sw, ilp, cd, cb):
    return f"cascade_sum_simd{sw}_ilp{ilp}_depth{cd}_base{cb}"


# ─── C templates ──────────────────────────────────────────────────────────
#
# Three template variants — chosen by pattern shape:
#   1. cascade(1, 1, 1, 0) — naive sequential
#   2. cascade(1, ILP, 1, 0) — scalar ILP-only (no cascade, no SIMD)
#   3. cascade(SW, ILP, CD, CB) for SW > 1 or CD > 1 — full algorithm


def emit_naive(sw, ilp, cd, cb):
    """cascade(1, 1, 1, 0) — naive sequential summation."""
    name = kernel_name(sw, ilp, cd, cb)
    return f"""float {name}(const float* data, int n) {{
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += data[i];
    return s;
}}
"""


def emit_ilp_only(sw, ilp, cd, cb):
    """cascade(1, ILP, 1, 0) — ILP parallel accumulators, no SIMD, no cascade."""
    name = kernel_name(sw, ilp, cd, cb)
    # ILP load lines: acc[0] += data[i*ILP+0]; acc[1] += data[i*ILP+1]; ...
    loads = "\n".join(
        f"        acc[{k}] += data[i * {ilp} + {k}];" for k in range(ilp)
    )
    # Horizontal collapse: acc[0] += acc[k] for k=1..ILP-1
    collapse = "\n".join(f"    acc[0] += acc[{k}];" for k in range(1, ilp))
    return f"""float {name}(const float* data, int n) {{
    float acc[{ilp}] = {{0}};
    int ilp_size = n / {ilp};
    for (int i = 0; i < ilp_size; ++i) {{
{loads}
    }}
    for (int i = ilp_size * {ilp}; i < n; ++i) acc[0] += data[i];
{collapse}
    return acc[0];
}}
"""


def emit_simd_no_cascade(sw, ilp, cd, cb):
    """cascade(SW>1, ILP, 1, 0) — SIMD-aware ILP-only, no cascade promotion.

    Used when CascadeDepth=1 and we still want SIMD parallelism. Algorithm:
      - Group input into (size_ilp × ILP × SW) blocks
      - Each block: accumulate into acc[ilp][s] (ILP × SW grid)
      - Tail SIMD blocks added to acc[0][s]
      - ILP horizontal collapse: acc[0][s] += acc[k][s]
      - Final: scalar tail + SIMD partials
    """
    name = kernel_name(sw, ilp, cd, cb)
    stride = sw * ilp
    return f"""float {name}(const float* data, int n) {{
    // cascade(SW={sw}, ILP={ilp}, CD=1, CB=0) — SIMD ILP-only
    if (n < {sw}) {{
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }}
    int vec_size = n / {sw};
    int size_ilp = vec_size / {ilp};
    int simd_processed = vec_size * {sw};
    float acc[{ilp}][{sw}] = {{{{0}}}};

    // Main loop: full ILP-groups
    for (int i = 0; i < size_ilp; ++i) {{
        const float* base = data + i * {stride};
        for (int ilp_lane = 0; ilp_lane < {ilp}; ++ilp_lane) {{
            const float* src = base + ilp_lane * {sw};
            for (int s = 0; s < {sw}; ++s) {{
                acc[ilp_lane][s] += src[s];
            }}
        }}
    }}

    // Tail SIMD blocks (didn't fill ILP)
    for (int v = size_ilp * {ilp}; v < vec_size; ++v) {{
        const float* src = data + v * {sw};
        for (int s = 0; s < {sw}; ++s) {{
            acc[0][s] += src[s];
        }}
    }}

    // ILP collapse
    for (int k = 1; k < {ilp}; ++k) {{
        for (int s = 0; s < {sw}; ++s) {{
            acc[0][s] += acc[k][s];
        }}
    }}

    // Final scalar tail + SIMD sum
    float final_acc = 0.0f;
    for (int i = simd_processed; i < n; ++i) {{
        final_acc += data[i];
    }}
    for (int s = 0; s < {sw}; ++s) {{
        final_acc += acc[0][s];
    }}
    return final_acc;
}}
"""


def emit_full_cascade(sw, ilp, cd, cb):
    """cascade(SW, ILP, CD, CB) — full SIMD × ILP × cascade-depth algorithm.

    For SW=1: scalar lanes, but still ILP × cascade.
    For SW>1: full PyTorch-style algorithm with SIMD-lane parallelism.
    """
    name = kernel_name(sw, ilp, cd, cb)
    # Pre-computed constants
    cb_log = cb.bit_length() - 1  # log2(CB) for power-of-2 CB
    cb_mask = cb - 1
    stride = sw * ilp

    # The cascade load: each iteration adds (sw * ilp) floats to acc[0].
    # acc has shape [cd][ilp][sw].
    if sw == 1:
        # Scalar lanes: acc[level][ilp_lane] (no third dim)
        load_loop = f"""
        for (int j = 0; j < level_step; ++j, ++i) {{
            const float* base = data + i * {ilp};
            for (int ilp_lane = 0; ilp_lane < {ilp}; ++ilp_lane) {{
                acc[0][ilp_lane] += base[ilp_lane];
            }}
        }}"""
        tail_loop = f"""
    for (; i < size_ilp; ++i) {{
        const float* base = data + i * {ilp};
        for (int ilp_lane = 0; ilp_lane < {ilp}; ++ilp_lane) {{
            acc[0][ilp_lane] += base[ilp_lane];
        }}
    }}"""
        promote_inner = f"""
            for (int ilp_lane = 0; ilp_lane < {ilp}; ++ilp_lane) {{
                acc[level][ilp_lane] += acc[level-1][ilp_lane];
                acc[level-1][ilp_lane] = 0.0f;
            }}"""
        final_collapse_inner = f"""
            for (int ilp_lane = 0; ilp_lane < {ilp}; ++ilp_lane) {{
                acc[0][ilp_lane] += acc[level][ilp_lane];
            }}"""
        ilp_collapse = f"""
    for (int k = 1; k < {ilp}; ++k) {{
        acc[0][0] += acc[0][k];
    }}"""
        final_add = "    final_acc += acc[0][0];"
        acc_decl = f"float acc[{cd}][{ilp}] = {{{{0}}}};"
        # Scalar path: no SIMD blocks, the "vec" notion collapses to elements.
        # We treat each input element as one "SIMD block" of width 1.
        scalar_fallback = ""
        size_calcs = f"""
    int vec_size = n;           // SW=1: each element is its own block
    int size_ilp = n / {ilp};
    int simd_processed = size_ilp * {ilp};"""
        simd_tail_loop = ""  # no SIMD tail for SW=1
        scalar_tail_n = "simd_processed"
    else:
        # SIMD-aware: acc[cd][ilp][sw]
        load_loop = f"""
        for (int j = 0; j < level_step; ++j, ++i) {{
            const float* base = data + i * {stride};
            for (int ilp_lane = 0; ilp_lane < {ilp}; ++ilp_lane) {{
                const float* src = base + ilp_lane * {sw};
                for (int s = 0; s < {sw}; ++s) {{
                    acc[0][ilp_lane][s] += src[s];
                }}
            }}
        }}"""
        tail_loop = f"""
    for (; i < size_ilp; ++i) {{
        const float* base = data + i * {stride};
        for (int ilp_lane = 0; ilp_lane < {ilp}; ++ilp_lane) {{
            const float* src = base + ilp_lane * {sw};
            for (int s = 0; s < {sw}; ++s) {{
                acc[0][ilp_lane][s] += src[s];
            }}
        }}
    }}"""
        promote_inner = f"""
            for (int ilp_lane = 0; ilp_lane < {ilp}; ++ilp_lane) {{
                for (int s = 0; s < {sw}; ++s) {{
                    acc[level][ilp_lane][s] += acc[level-1][ilp_lane][s];
                    acc[level-1][ilp_lane][s] = 0.0f;
                }}
            }}"""
        final_collapse_inner = f"""
            for (int ilp_lane = 0; ilp_lane < {ilp}; ++ilp_lane) {{
                for (int s = 0; s < {sw}; ++s) {{
                    acc[0][ilp_lane][s] += acc[level][ilp_lane][s];
                }}
            }}"""
        ilp_collapse = f"""
    for (int k = 1; k < {ilp}; ++k) {{
        for (int s = 0; s < {sw}; ++s) {{
            acc[0][0][s] += acc[0][k][s];
        }}
    }}"""
        final_add = f"""    for (int s = 0; s < {sw}; ++s) {{
        final_acc += acc[0][0][s];
    }}"""
        acc_decl = f"float acc[{cd}][{ilp}][{sw}] = {{{{{{0}}}}}};"
        scalar_fallback = f"""
    if (n < {sw}) {{
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += data[i];
        return s;
    }}"""
        size_calcs = f"""
    int vec_size = n / {sw};
    int size_ilp = vec_size / {ilp};
    int simd_processed = vec_size * {sw};"""
        simd_tail_loop = f"""
    for (int v = size_ilp * {ilp}; v < vec_size; ++v) {{
        const float* src = data + v * {sw};
        for (int s = 0; s < {sw}; ++s) {{
            acc[0][0][s] += src[s];
        }}
    }}"""
        scalar_tail_n = "simd_processed"

    return f"""float {name}(const float* data, int n) {{
    // cascade(SW={sw}, ILP={ilp}, CD={cd}, CB={cb}){scalar_fallback}
{size_calcs}
    {acc_decl}
    int level_step = {cb};
    int level_mask = {cb_mask};
    int lp = {cb_log};

    int i = 0;
    for (; i + level_step <= size_ilp;) {{{load_loop}
        // Cascade promotion
        for (int level = 1; level < {cd}; ++level) {{{promote_inner}
            int mask = level_mask << (level * lp);
            if ((i & mask) != 0) break;
        }}
    }}{tail_loop}

    // Final per-(ilp,simd) cascade collapse: levels 1..CD-1 → level 0
    for (int level = 1; level < {cd}; ++level) {{{final_collapse_inner}
    }}
{simd_tail_loop}
    // ILP horizontal collapse: lane 0 += lane k for k=1..ILP-1
{ilp_collapse}

    // Scalar tail + final SIMD horizontal sum
    float final_acc = 0.0f;
    for (int i = {scalar_tail_n}; i < n; ++i) {{
        final_acc += data[i];
    }}
{final_add}
    return final_acc;
}}
"""


def emit_kernel(sw, ilp, cd, cb):
    """Dispatch to the right template based on pattern shape."""
    if sw == 1 and ilp == 1 and cd == 1:
        return emit_naive(sw, ilp, cd, cb)
    if sw == 1 and cd == 1:
        return emit_ilp_only(sw, ilp, cd, cb)
    if sw > 1 and cd == 1:
        return emit_simd_no_cascade(sw, ilp, cd, cb)
    return emit_full_cascade(sw, ilp, cd, cb)


def emit_dispatch_table(patterns):
    """Emit a name→pointer dispatch table so the sweep harness can call
    each kernel by string name via ctypes."""
    lines = [
        "// Dispatch table: cascade_dispatch[i] is the i-th kernel function pointer.",
        "// cascade_dispatch_names[i] is the matching string name.",
        "// cascade_dispatch_count is the total number of kernels.",
        "",
        "typedef float (*cascade_kernel_fn)(const float* data, int n);",
        "",
    ]
    names = [kernel_name(*p) for p in patterns]
    lines.append(f"const int cascade_dispatch_count = {len(patterns)};")
    lines.append("")
    lines.append("cascade_kernel_fn cascade_dispatch[] = {")
    for n in names:
        lines.append(f"    {n},")
    lines.append("};")
    lines.append("")
    lines.append("const char* cascade_dispatch_names[] = {")
    for n in names:
        lines.append(f'    "{n}",')
    lines.append("};")
    return "\n".join(lines) + "\n"


# ─── Main ──────────────────────────────────────────────────────────────────


def main():
    patterns = list(enumerate_patterns())
    print(f"// Generated by bench/generate_cascade_kernels.py — DO NOT EDIT", file=sys.stderr)
    print(f"// {len(patterns)} cascade reduction kernel instantiations", file=sys.stderr)

    # Header
    sys.stdout.write("// =================================================================\n")
    sys.stdout.write("// cascade_kernels_generated.c — auto-generated, do not edit.\n")
    sys.stdout.write("//\n")
    sys.stdout.write("// One C function per valid cascade(SW, ILP, CD, CB) instantiation.\n")
    sys.stdout.write("// Generator: bench/generate_cascade_kernels.py\n")
    sys.stdout.write("// Parameter space: lib/reduction_kernel.pl reduction_pattern/4\n")
    sys.stdout.write(f"// Total kernels: {len(patterns)}\n")
    sys.stdout.write("//\n")
    sys.stdout.write("// Each kernel sums an array of `n` floats. The sweep harness\n")
    sys.stdout.write("// (bench/verify_cascade_sweep.py) compiles this file and runs each\n")
    sys.stdout.write("// instantiation against PyTorch CPU at multiple input sizes to\n")
    sys.stdout.write("// determine which (SW, ILP, CD, CB) matches PyTorch's bit pattern.\n")
    sys.stdout.write("// =================================================================\n\n")

    for p in patterns:
        sys.stdout.write(emit_kernel(*p))
        sys.stdout.write("\n")

    sys.stdout.write(emit_dispatch_table(patterns))


if __name__ == "__main__":
    main()
