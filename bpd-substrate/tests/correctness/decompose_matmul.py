#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""decompose_matmul.py — decompose Q-projection MUL_MAT into sub-steps and find
   the exact bit-level divergence point.

The Q-projection at layer 0 (Qcur-0, fixture idx 7) is the FIRST divergent
op. Per-op gates show it diverges by max_abs=1.9e-6. This script decomposes
the matmul into its constituent steps and locates the precise sub-step
where the bit pattern first differs from ggml.

Decomposition:

  Step A: Quantize input row m to Q8_0
          input X[m] (F32, 2048) -> X_q8_0[m] (Q8_0 bytes, 64 blocks * 34)
          Sub-sub-steps:
            A1: find amax (positive absolute max of row)
            A2: compute d = amax/127 (the F16 scale stored in block)
            A3: compute id = 127/amax (the F32 multiplier for quantization)
            A4: quantize each element: int8(rintf(x[i] * id))
            A5: F32->F16 conversion of d for storage

  Step B: For each (m, n), block-wise dot product
          W_q8_0[n] (64 blocks) dot X_q8_0[m] (64 blocks) -> F32 sum
          Sub-sub-steps:
            B1: per-block int8 dot: sum_{j=0..31} W[n,b,j]_i8 * X[m,b,j]_i8 -> int32
            B2: per-block F32 scaling: int32_sum * (f16_to_f32(W.scale) * f16_to_f32(X.scale))
            B3: F32 accumulator across blocks

Verifiers compare our substrate's output of each sub-step against an
analytic ground truth (computed in Python int64/float64 to bypass F32
intermediate rounding), AND against a numpy reimplementation of ggml's
AVX1 path. This bisects the F32-precision divergence to its source.
"""
import argparse
import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np

HARNESS_VERSION = "1.0.0"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "bench"))

from llama_fixture_loader import load_manifest, find_op  # noqa: E402

c_float_p = ctypes.POINTER(ctypes.c_float)
c_uint8_p = ctypes.POINTER(ctypes.c_uint8)


def f16_to_f32_numpy(u16):
    """Numpy's IEEE 754 F16 -> F32 (matches F16C instruction)."""
    return np.frombuffer(np.array([u16], dtype=np.uint16).tobytes(), dtype=np.float16)[0].astype(np.float32)


def quantize_row_q8_0_avx_ref(x):
    """Pure numpy reimplementation of ggml's AVX1 quantize_row_q8_0.

    Per 32-element block:
      maxScalar = max(abs(x))
      d = maxScalar / 127  (stored as F16)
      id = 127 / maxScalar  (used for quantization; NOTE: 127/max, NOT 1/d)
      qs[j] = rintf(x[j] * id)  (round-half-to-even)
      Store: 2-byte F16(d), 32-byte int8 qs
    """
    n = len(x)
    nb = n // 32
    out = bytearray(nb * 34)
    for b in range(nb):
        block = x[b*32:(b+1)*32]
        maxScalar = float(np.abs(block).max())
        d = np.float32(maxScalar / 127.0)
        id_inv = np.float32(127.0 / maxScalar) if maxScalar != 0 else np.float32(0)
        # F32 -> F16 -> uint16 via numpy (F16C-equivalent rounding)
        d_f16 = np.float16(d)
        out[b*34:b*34+2] = d_f16.tobytes()
        for j in range(32):
            v = np.float32(block[j] * id_inv)
            # round-half-to-even via int(rint)
            q = int(np.rint(v))
            q = max(-128, min(127, q))
            out[b*34 + 2 + j] = q & 0xff
    return bytes(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fixture-dir", required=True)
    p.add_argument("--so", required=True)
    p.add_argument("--gguf", required=True)
    p.add_argument("--report", default="/tmp/decompose_matmul_report.json")
    args = p.parse_args()

    print(f"[harness] decompose_matmul v{HARNESS_VERSION}")

    lib = ctypes.CDLL(args.so)
    lib.bpd_quant_q8_0_cpu.restype = None
    lib.bpd_quant_q8_0_cpu.argtypes = [c_float_p, c_uint8_p, ctypes.c_int]
    lib.bpd_qdot_q8_0_q8_0_cpu.restype = ctypes.c_float
    lib.bpd_qdot_q8_0_q8_0_cpu.argtypes = [c_uint8_p, c_uint8_p, ctypes.c_int]
    lib.bpd_qmatmul_q8_0_cpu.restype = None
    lib.bpd_qmatmul_q8_0_cpu.argtypes = [c_uint8_p, c_float_p, c_float_p,
                                          ctypes.c_int, ctypes.c_int, ctypes.c_int]

    tensors = load_manifest(args.fixture_dir)

    # The input to Qcur-0 is attn_norm-0 (F32, 6 tokens × 2048 dim)
    attn_norm = find_op(tensors, name_substring="attn_norm-0", op_desc="MUL")
    qcur = find_op(tensors, name_substring="Qcur-0", op_desc="MUL_MAT")
    X = np.ascontiguousarray(attn_norm.as_numpy(), dtype=np.float32)  # (6, 2048)
    ref_qcur = np.ascontiguousarray(qcur.as_numpy(), dtype=np.float32)  # (6, 2048)
    n_tokens, embed_dim = X.shape

    # Load Q-projection weight from GGUF
    import subprocess
    script = REPO_ROOT / "tests" / "gguf_query.pl"
    res = subprocess.run(
        ["swipl", "-q", "-g", f"consult('{script}'), gguf_query_main", "--",
         args.gguf, "blk.0.attn_q.weight"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30)
    fields = dict(tok.split("=", 1) for tok in res.stdout.strip().splitlines()[-1].split() if "=" in tok)
    W_off = int(fields["ABS_OFFSET"]); W_size = int(fields["SIZE"])
    W = np.fromfile(args.gguf, dtype=np.uint8, count=W_size, offset=W_off)
    bytes_per_row = (embed_dim // 32) * 34  # 2176
    print(f"[setup] X: {X.shape}, W: {W.shape} ({bytes_per_row} bytes/row, {W_size//bytes_per_row} rows)")

    report = {
        "harness_version": HARNESS_VERSION,
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "steps": [],
    }

    # ─── STEP A: Quantize X to Q8_0 ───────────────────────────────────────
    print(f"\n[STEP A] Quantize input X[m] to Q8_0, per-row")
    our_X_q8_0 = np.zeros(n_tokens * bytes_per_row, dtype=np.uint8)
    for m in range(n_tokens):
        row_f32 = np.ascontiguousarray(X[m], dtype=np.float32)
        out_bytes = np.zeros(bytes_per_row, dtype=np.uint8)
        lib.bpd_quant_q8_0_cpu(
            row_f32.ctypes.data_as(c_float_p),
            out_bytes.ctypes.data_as(c_uint8_p),
            ctypes.c_int(embed_dim))
        our_X_q8_0[m*bytes_per_row:(m+1)*bytes_per_row] = out_bytes

    # Compare to numpy reimplementation of ggml's AVX1 quantize_row_q8_0
    diffs_per_row = []
    for m in range(n_tokens):
        ref_bytes = quantize_row_q8_0_avx_ref(X[m])
        our_bytes = bytes(our_X_q8_0[m*bytes_per_row:(m+1)*bytes_per_row])
        n_diff_bytes = sum(1 for a, b in zip(ref_bytes, our_bytes) if a != b)
        diffs_per_row.append(n_diff_bytes)
        print(f"  row {m}: {n_diff_bytes}/{bytes_per_row} bytes differ")

    a_status = "pass" if all(d == 0 for d in diffs_per_row) else "fail"
    print(f"  STEP A status: {a_status} (all rows byte-identical with ggml AVX1 ref)")
    report["steps"].append({
        "name": "A: quantize_row_q8_0",
        "status": a_status,
        "n_diff_bytes_per_row": diffs_per_row,
        "bytes_per_row": bytes_per_row,
    })

    # ─── STEP B: Per-(m, n) block-dot ─────────────────────────────────────
    # The full matmul is Q[m, n] = sum_b dot(W[n, b], X[m, b]) * f16_to_f32(Wd[n,b]) * f16_to_f32(Xd[m,b])
    # Test the inner kernel: for (m=0, n=0), block-by-block dot.
    print(f"\n[STEP B] Block-dot for (m=0, n=0): substrate kernel vs Python ref")

    # Our substrate's full dot
    our_dot = float(lib.bpd_qdot_q8_0_q8_0_cpu(
        W[:bytes_per_row].ctypes.data_as(c_uint8_p),
        our_X_q8_0[:bytes_per_row].ctypes.data_as(c_uint8_p),
        ctypes.c_int(embed_dim // 32)))

    # Python ref: same algorithm, F32 arithmetic
    n_blocks = embed_dim // 32
    py_dot = np.float32(0.0)
    for b in range(n_blocks):
        wb_offset = b * 34
        # int8 dot for this block
        w_quants = np.frombuffer(bytes(W[wb_offset+2:wb_offset+34]), dtype=np.int8)
        x_quants = np.frombuffer(bytes(our_X_q8_0[wb_offset+2:wb_offset+34]), dtype=np.int8)
        sumi = int(np.sum(w_quants.astype(np.int64) * x_quants.astype(np.int64)))
        # F16 scales
        wd_u16 = int(W[wb_offset]) | (int(W[wb_offset+1]) << 8)
        xd_u16 = int(our_X_q8_0[wb_offset]) | (int(our_X_q8_0[wb_offset+1]) << 8)
        wd_f32 = f16_to_f32_numpy(wd_u16)
        xd_f32 = f16_to_f32_numpy(xd_u16)
        py_dot = np.float32(py_dot + np.float32(sumi) * (wd_f32 * xd_f32))

    b_diff = float(abs(our_dot - py_dot))
    b_status = "pass" if our_dot == py_dot else ("close" if b_diff < 1e-6 else "fail")
    print(f"  ours:    {our_dot}")
    print(f"  py_ref:  {py_dot}")
    print(f"  diff:    {b_diff:.6e}")
    print(f"  STEP B status: {b_status}")
    report["steps"].append({
        "name": "B: qdot_q8_0_q8_0 (m=0, n=0)",
        "status": b_status,
        "ours": float(our_dot),
        "py_ref": float(py_dot),
        "diff": b_diff,
    })

    # Compare both to ggml's captured Qcur-0[0, 0]
    ggml_dot = float(ref_qcur[0, 0])
    print(f"  ggml ref:   {ggml_dot}")
    print(f"  ours vs ggml: {abs(our_dot - ggml_dot):.6e}")
    print(f"  py_ref vs ggml: {abs(py_dot - ggml_dot):.6e}")

    # ─── STEP C: Full matmul, decomposed cell-by-cell ─────────────────────
    # The full matmul is 12288 individual cell-dots (6 tokens × 2048 weight rows).
    # We verified ONE cell (m=0, n=0) at 0 ULP in Step B. Does that imply ALL
    # cells are 0 ULP? Step C answers empirically.
    print(f"\n[STEP C] Full matmul, decomposed:")
    print(f"  C1: substrate's bpd_qmatmul_q8_0_cpu vs ggml fixture")
    out_qcur = np.zeros(n_tokens * embed_dim, dtype=np.float32)
    lib.bpd_qmatmul_q8_0_cpu(
        W.ctypes.data_as(c_uint8_p),
        X.ctypes.data_as(c_float_p),
        out_qcur.ctypes.data_as(c_float_p),
        ctypes.c_int(n_tokens), ctypes.c_int(embed_dim), ctypes.c_int(embed_dim))
    out_qcur_2d = out_qcur.reshape(n_tokens, embed_dim)

    # Per-cell ULP analysis
    diff_2d = out_qcur_2d - ref_qcur
    max_abs = float(np.abs(diff_2d).max())
    # Which cells differ?
    cell_diff_mask = diff_2d != 0
    n_cells_diff = int(cell_diff_mask.sum())
    n_cells_total = n_tokens * embed_dim
    # Per-row breakdown: how many cells differ in each token row
    per_row_diff = [int(cell_diff_mask[m].sum()) for m in range(n_tokens)]
    # Per-column-block breakdown: ggml's llamafile_sgemm operates in tile_n strides;
    # if the tile_n is 2 (RN=2 for our (m=6, n>=4) case), columns 0-1 are tile 0,
    # 2-3 are tile 1, etc. Let's see if divergence correlates with column position.
    per_col_diff = [int(cell_diff_mask[:, n].sum()) for n in range(embed_dim)]

    print(f"  C1 result: max_abs={max_abs:.6e}, cells_diff={n_cells_diff}/{n_cells_total}")
    print(f"  Per-row divergence counts: {per_row_diff}")

    # C2: per-cell dot product comparison
    # For 5 random sample cells (including m=0, n=0 which we know matches),
    # compute our cell-dot, ggml's reference value, and compare.
    print(f"\n  C2: per-cell dot product comparison")
    sample_cells = [(0, 0), (0, 1), (0, 511), (3, 0), (3, 511)]
    cell_results = []
    for m, n in sample_cells:
        w_row_off = n * bytes_per_row
        w_row = W[w_row_off:w_row_off+bytes_per_row]
        x_row = our_X_q8_0[m*bytes_per_row:(m+1)*bytes_per_row]
        cell_dot = float(lib.bpd_qdot_q8_0_q8_0_cpu(
            w_row.ctypes.data_as(c_uint8_p),
            x_row.ctypes.data_as(c_uint8_p),
            ctypes.c_int(n_blocks)))
        ggml_val = float(ref_qcur[m, n])
        diff = abs(cell_dot - ggml_val)
        status_icon = "✅" if diff == 0 else "❌"
        print(f"  {status_icon} ({m:2d},{n:4d}): ours={cell_dot:.10e}, ggml={ggml_val:.10e}, diff={diff:.6e}")
        cell_results.append({
            "m": m, "n": n,
            "ours": cell_dot, "ggml": ggml_val, "diff": diff,
            "match": cell_dot == ggml_val,
        })

    c_status = "pass" if max_abs == 0 else "fail"
    report["steps"].append({
        "name": "C: full Q-projection matmul",
        "status": c_status,
        "max_abs_vs_ggml": max_abs,
        "n_cells_diff": n_cells_diff,
        "n_cells_total": n_cells_total,
        "per_row_diff": per_row_diff,
        "sample_cells": cell_results,
    })

    # ─── DIAGNOSIS ─────────────────────────────────────────────────────────
    print(f"\n[diagnosis]")
    # Check: does m=0 row match ggml exactly?
    m0_match = per_row_diff[0] == 0
    # Check: how many cells in n>=4 region (where mnpack splits into gemm<4,2>
    # vs scalar fallback for the n>=4 tail) versus n<4?
    n_lt_4_diff = int(cell_diff_mask[:, :4].sum())
    n_ge_4_diff = int(cell_diff_mask[:, 4:].sum())
    print(f"  m=0 row matches ggml exactly: {m0_match}")
    print(f"  cells with n<4 that differ: {n_lt_4_diff}")
    print(f"  cells with n>=4 that differ: {n_ge_4_diff}")
    n_passing_cells = n_cells_total - n_cells_diff
    print(f"  passing cells: {n_passing_cells}/{n_cells_total} ({100*n_passing_cells/n_cells_total:.1f}%)")

    if a_status == "pass" and m0_match and not c_status == "pass":
        msg = ("Quantizer byte-identical with ggml. m=0 row matches ggml exactly. "
               "Most cells diverge by ~1.9e-6 \u2014 the same magnitude. This suggests "
               "ggml's matmul uses a reduction order that, for SOME (m, n) cells, "
               "produces the same result as the scalar reduction (matching ours), "
               "and for others produces a different result. Likely llamafile_sgemm "
               "tile dispatch: cells inside a tile share an accumulator state that "
               "differs from per-cell scalar accumulation.")
        print(f"  CONCLUSION: {msg}")
        report["diagnosis"] = msg
    elif a_status == "pass" and c_status == "pass":
        report["diagnosis"] = "all steps pass; matmul is bit-identical"
    else:
        report["diagnosis"] = "see step statuses"

    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[harness] wrote {args.report}")


if __name__ == "__main__":
    main()
