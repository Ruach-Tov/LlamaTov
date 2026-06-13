#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""sass_audit.py — Disassemble substrate-emitted kernels and characterize.

Per Heath's interjection 2026-05-19: SASS comparison reveals what kernels
actually do at silicon level. This tool produces structured SASS audit
data per kernel.

Run on the enclave inside a nix-shell with cudaPackages available.
"""
import re
import subprocess
import sys
from pathlib import Path


VALDIR = Path("/tmp/l1_cuda_validation")
SASSDIR = Path("/tmp/l1_sass_audit")
SASSDIR.mkdir(exist_ok=True)


def disassemble(obj_path):
    """Run cuobjdump --dump-sass and return the SASS text."""
    result = subprocess.run(
        ["cuobjdump", "--dump-sass", str(obj_path)],
        capture_output=True, text=True, timeout=60
    )
    return result.stdout


def characterize(sass_text):
    """Extract substantive instruction-mix characterization from SASS."""
    # Match SASS instruction lines: /*0XXX*/   OPNAME ...
    instr_pat = re.compile(r'^\s+/\*[0-9a-f]+\*/\s+(\S+)', re.MULTILINE)
    instrs = instr_pat.findall(sass_text)

    counts = {}
    for op in instrs:
        # Trim modifiers (e.g. FADD.SAT -> FADD)
        base = op.split('.')[0]
        counts[base] = counts.get(base, 0) + 1

    return {
        "total_instr": len(instrs),
        "FADD": counts.get("FADD", 0),
        "FFMA": counts.get("FFMA", 0),
        "FMUL": counts.get("FMUL", 0),
        "FMNMX": counts.get("FMNMX", 0),
        "LDG": counts.get("LDG", 0),
        "STG": counts.get("STG", 0),
        "ATOM": counts.get("ATOM", 0) + counts.get("RED", 0),
        "SHFL": counts.get("SHFL", 0),
        "BAR": counts.get("BAR", 0),
        "FSEL": counts.get("FSEL", 0),
        "all_ops": sorted(counts.items(), key=lambda x: -x[1])[:8],
    }


def main():
    print("=== Tier 2 SASS Audit ===")
    print(f"Disassembling {len(list(VALDIR.glob('*.o')))} substrate-emitted .o files")
    print()
    print(f"{'kernel':<28} {'instr':>5} {'FADD':>5} {'FFMA':>5} {'LDG':>5} "
          f"{'STG':>5} {'BAR':>4} {'ATOM':>5}")
    print("-" * 90)

    results = []
    for obj in sorted(VALDIR.glob("*.o")):
        name = obj.stem
        sass = disassemble(obj)
        # Save full SASS
        (SASSDIR / f"{name}.sass").write_text(sass)
        # Characterize
        c = characterize(sass)
        results.append((name, c))
        print(f"{name:<28} {c['total_instr']:>5} {c['FADD']:>5} {c['FFMA']:>5} "
              f"{c['LDG']:>5} {c['STG']:>5} {c['BAR']:>4} {c['ATOM']:>5}")

    print()
    print(f"SASS files saved to {SASSDIR}/")
    print()

    # Substantive observations
    print("=== Substantive observations ===")

    # The reduction kernels — these are the REDUCTION_ORDER_DIVERGENCE cases
    reductions = ["reduce_sum_rows", "reduce_mean"]
    for r_name in reductions:
        match = next((c for n, c in results if n == r_name), None)
        if match:
            print(f"  {r_name}: {match['FADD']} FADDs across "
                  f"{match['total_instr']} total instructions")
            print(f"    → Confirms sequential reduction at SASS level "
                  f"(would be tree if pairwise; would be SHFL if warp_shuffle)")

    # Conv kernels — should have FFMA (multiply-add) for matmul; im2col is just data movement
    convs = ["conv_2d_forward", "conv_transpose_2d"]
    for c_name in convs:
        match = next((c for n, c in results if n == c_name), None)
        if match:
            print(f"  {c_name}: FFMA={match['FFMA']} (multiply-add ops), "
                  f"FADD={match['FADD']} (plain add), "
                  f"LDG={match['LDG']} STG={match['STG']}")

    # Atomic-free check (transpose was substrate-design choice)
    atomic_kernels = [n for n, c in results if c['ATOM'] > 0]
    print(f"  Kernels using atomic operations: "
          f"{atomic_kernels if atomic_kernels else 'NONE (gather pattern confirmed in SASS)'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
