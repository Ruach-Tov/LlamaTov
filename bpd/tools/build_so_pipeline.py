#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
import os as _os, sys as _sys
import os as _os2
_REPO = _os2.environ.get("LLAMATOV_ROOT") or _os2.path.abspath(_os2.path.join(_os2.path.dirname(_os2.path.abspath(__file__)), *[".."]*8))

def _bpd_root(_p=_os.path.dirname(_os.path.abspath(__file__))):
    while _p != '/' and _os.path.basename(_p) != 'bpd':
        _p = _os.path.dirname(_p)
    return _p if _os.path.basename(_p) == 'bpd' else _os.path.dirname(_os.path.abspath(__file__))
_BPD = _bpd_root()

"""build_so_pipeline.py — uniform .ll -> .so -> objdump -> lift pipeline.

Per Heath's direction (2026-05-31): "compile to a .so and extract with the
same code we use to extract from pytorch .so or llama.cpp -> .so, then lift
with the same lifter. Just complete the normal toolchain execution until
you drop a comparable binary file."

This driver takes a .ll file and produces:
  - The compiled .so (cached in workdir)
  - The objdump-format asm text per function
  - The pcode chain per function
  - Unroll detection result per function

Used by the dashboard to feed the three q8_0_dot cases (ggml, BPD intrinsic,
BPD scalar) into a uniform pipeline.

Usage:
    python3 build_so_pipeline.py LL_FILE [--workdir DIR] [--json]
"""
import argparse
import os
import sys
import subprocess
import json
import re

sys.path.insert(0, _os2.path.join(_REPO, "bpd/tools"))
import asm_pcode
import asm_unroll


def build_so_from_ll(ll_path, workdir, cpu="skylake-avx512"):
    """llc + clang -shared. Returns .so path.

    cpu: -mcpu flag for llc. Default 'skylake-avx512' enables SSSE3+/AVX/AVX2/AVX512
    so pmadd/pmaddubsw intrinsics resolve. Use 'x86-64' for baseline.
    """
    name = os.path.splitext(os.path.basename(ll_path))[0]
    obj_path = os.path.join(workdir, f"{name}.o")
    so_path = os.path.join(workdir, f"lib{name}.so")

    # llc: .ll -> .o
    subprocess.run(
        ["llc", f"-mcpu={cpu}", "-filetype=obj", "-relocation-model=pic",
         "-o", obj_path, ll_path],
        check=True, capture_output=True, text=True,
    )
    # clang -shared: .o -> .so
    subprocess.run(
        ["clang", "-shared", "-o", so_path, obj_path],
        check=True, capture_output=True, text=True,
    )
    return so_path


def list_functions(so_path):
    """Return [(symbol, start_addr, size)] for each function in .text."""
    out = subprocess.run(
        ["objdump", "-t", so_path], capture_output=True, text=True, check=True,
    ).stdout
    functions = []
    for line in out.splitlines():
        parts = line.split()
        # symbol-table: ADDR FLAGS F SECTION SIZE NAME
        # e.g.: 0000000000001100 g     F .text	0000000000000063 unrolled4
        if len(parts) >= 6 and parts[3] == ".text":
            try:
                addr = int(parts[0], 16)
                size = int(parts[4], 16)
                # 'F' marker is parts[2] in 6-field form, or the type may vary
                name = parts[-1]
                if size > 0 and not name.startswith("_"):
                    functions.append((name, addr, size))
            except (ValueError, IndexError):
                continue
    return functions


def objdump_text(so_path, start, end):
    """Return the asm text for one address range."""
    return subprocess.run(
        ["objdump", "-d", "--no-show-raw-insn",
         f"--start-address={start}", f"--stop-address={end}", so_path],
        capture_output=True, text=True, check=True,
    ).stdout


def analyze_function(so_path, symbol, start, size):
    """Full pipeline: objdump -> lift -> unroll detect -> structured result."""
    end = start + size
    asm = objdump_text(so_path, f"0x{start:x}", f"0x{end:x}")
    chain = asm_pcode.lift_from_text(asm)
    segment = asm_unroll.detect_unroll_segment(chain)

    return {
        "symbol": symbol,
        "start": f"0x{start:x}",
        "end": f"0x{end:x}",
        "size": size,
        "instruction_count": len(chain),
        "asm_text": asm,
        "unroll": {
            "factor": segment["factor"],
            "rolled_body_insns": len(segment["rolled_body"]),
            "preamble_insns": len(segment["preamble"]),
            "epilogue_insns": len(segment["epilogue"]),
            "segment_start_idx": segment["segment_start"],
            "segment_end_idx": segment["segment_end"],
            "rolled_body_text": [
                asm_unroll.render_fp(asm_unroll.insn_fingerprint(i))
                for i in segment["rolled_body"]
            ],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ll_file", help="Path to .ll file")
    ap.add_argument("--workdir", help="Workdir (default: /tmp/build_so_<name>)")
    ap.add_argument("--json", action="store_true", help="Emit JSON")
    args = ap.parse_args()

    name = os.path.splitext(os.path.basename(args.ll_file))[0]
    workdir = args.workdir or f"/tmp/build_so_{name}"
    os.makedirs(workdir, exist_ok=True)

    so_path = build_so_from_ll(args.ll_file, workdir)
    functions = list_functions(so_path)
    results = [analyze_function(so_path, sym, start, size)
               for (sym, start, size) in functions]

    if args.json:
        print(json.dumps({
            "ll_file": args.ll_file,
            "so_path": so_path,
            "workdir": workdir,
            "functions": results,
        }, indent=2))
    else:
        print(f"Built {so_path}")
        print(f"Workdir: {workdir}")
        print()
        for r in results:
            u = r["unroll"]
            print(f"=== {r['symbol']} @ {r['start']}..{r['end']} ({r['instruction_count']} insns) ===")
            print(f"  Unroll: factor={u['factor']}, "
                  f"rolled_body={u['rolled_body_insns']} insns, "
                  f"preamble={u['preamble_insns']}, "
                  f"epilogue={u['epilogue_insns']}")
            if u['factor'] >= 2:
                print(f"  Rolled body:")
                for line in u['rolled_body_text']:
                    print(f"    {line}")


if __name__ == "__main__":
    main()
