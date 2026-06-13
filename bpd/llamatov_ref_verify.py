# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""llamatov_ref_verify.py — Kernel-level Verification Harness

Given the C inference loop's per-stage output (dumped to numpy files
in the same layout as llamatov_ref_dump.py produces), compare against
the reference and report any stages that diverge beyond threshold.

USAGE:

  # First, generate the reference (CPU+KV path):
  python3 llamatov_ref_dump.py model.gguf --tokens 5 --out-dir /tmp/llamatov_ref

  # Then run the C inference loop with intermediate capture enabled,
  # dumping to /tmp/llamatov_c_inference/ in the same layout.

  # Then compare:
  python3 llamatov_ref_verify.py \\
      --ref /tmp/llamatov_ref \\
      --c-out /tmp/llamatov_c_inference

OUTPUT:

  Per stage, per layer, per token: cosine_sim and max_abs_drift.
  Summary: pass/fail for the whole run + the first failing stage.

THRESHOLD:

  cosine_sim > 0.9999 with max_abs_drift bounded.
  Defaults are conservative; relax via flags if the C side uses fp16
  or other reduced precision.

CONTRACT VS C INFERENCE LOOP:

For the C inference loop to be considered correct, EVERY stage at
EVERY layer at EVERY position must pass the threshold. Any divergence
beyond threshold is a kernel bug (or a verification harness bug, in
which case investigate the threshold).

Author: metayen 2026-05-16
Per Heath's "correct first" mantra. The kernel-level verification
predicate that mavchin and I will use during C-loop development.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np


# Stages dumped per layer (must match llamatov_ref_dump.py).
LAYER_STAGES = [
    "pre_attn",
    "h_attn",
    "q_pre_rope",
    "k_pre_rope",
    "v",
    "q_post_rope",
    "k_post_rope",
    "k_cache",
    "v_cache",
    "scores",
    "probs",
    "attn_out",
    "post_attn",
    "h_ffn",
    "gate",
    "up",
    "ffn_pre_down",
    "ffn",
    "post_ffn",
]

PER_TOKEN_STAGES = ["embed", "logits"]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between flattened tensors. Returns 1.0 if both zero."""
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    na = np.linalg.norm(af)
    nb = np.linalg.norm(bf)
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(af, bf) / (na * nb))


def compare_stage(ref_path: Path, c_path: Path, cos_threshold: float,
                  abs_threshold: float):
    """Compare one stage's reference and C output.

    Returns dict with keys: status (pass/fail/missing/shape_mismatch),
    cos, max_abs, ref_path, c_path.
    """
    if not ref_path.exists():
        return {"status": "missing_ref", "ref_path": str(ref_path)}
    if not c_path.exists():
        return {"status": "missing_c", "c_path": str(c_path)}

    try:
        ref = np.load(ref_path)
        cval = np.load(c_path)
    except Exception as e:
        return {"status": "load_error", "error": str(e)}

    if ref.shape != cval.shape:
        return {
            "status": "shape_mismatch",
            "ref_shape": list(ref.shape),
            "c_shape": list(cval.shape),
        }

    cs = cosine_sim(ref, cval)
    abs_diff = float(np.max(np.abs(ref - cval)))

    status = "pass" if (cs >= cos_threshold and abs_diff <= abs_threshold) else "fail"
    return {
        "status": status,
        "cos": cs,
        "max_abs": abs_diff,
    }


def verify(ref_root: Path, c_root: Path, cos_threshold: float,
           abs_threshold: float, verbose: bool = False):
    """Walk both directory trees and compare every stage.

    Returns a summary dict.
    """
    # Load reference meta
    ref_meta = json.loads((ref_root / "meta.json").read_text())
    n_layers = ref_meta["n_layers"]

    # Find all token directories in reference
    token_dirs = sorted([d for d in ref_root.iterdir() if d.is_dir() and d.name.startswith("tok")])

    total_stages = 0
    pass_stages = 0
    failures = []  # list of (token_step, layer, stage, info)

    for tok_dir in token_dirs:
        tok_name = tok_dir.name
        c_tok_dir = c_root / tok_name

        # Per-token stages
        for stage in PER_TOKEN_STAGES:
            ref_path = tok_dir / f"{stage}.npy"
            if not ref_path.exists():
                continue
            c_path = c_tok_dir / f"{stage}.npy"
            total_stages += 1
            result = compare_stage(ref_path, c_path, cos_threshold, abs_threshold)
            if result.get("status") == "pass":
                pass_stages += 1
                if verbose:
                    print(f"  {tok_name}/{stage}: PASS  cos={result['cos']:.6f}  max_abs={result['max_abs']:.2e}")
            else:
                failures.append((tok_name, None, stage, result))
                print(f"  {tok_name}/{stage}: {result.get('status').upper()}  {result}")

        # Per-layer stages
        for il in range(n_layers):
            layer_name = f"layer{il:02d}"
            layer_dir = tok_dir / layer_name
            if not layer_dir.exists():
                continue
            c_layer_dir = c_tok_dir / layer_name

            for stage in LAYER_STAGES:
                ref_path = layer_dir / f"{stage}.npy"
                if not ref_path.exists():
                    continue
                c_path = c_layer_dir / f"{stage}.npy"
                total_stages += 1
                result = compare_stage(ref_path, c_path, cos_threshold, abs_threshold)
                if result.get("status") == "pass":
                    pass_stages += 1
                    if verbose:
                        print(f"  {tok_name}/{layer_name}/{stage}: PASS  cos={result['cos']:.6f}  max_abs={result['max_abs']:.2e}")
                else:
                    failures.append((tok_name, layer_name, stage, result))
                    # Print only the first few failures by default
                    if len(failures) <= 10 or verbose:
                        print(f"  {tok_name}/{layer_name}/{stage}: {result.get('status').upper()}  {result}")

    summary = {
        "total_stages": total_stages,
        "pass_stages": pass_stages,
        "fail_stages": total_stages - pass_stages,
        "first_failure": failures[0] if failures else None,
        "all_pass": len(failures) == 0,
    }

    print()
    print(f"Total stages compared: {total_stages}")
    print(f"Pass: {pass_stages}")
    print(f"Fail: {len(failures)}")
    if failures:
        print(f"First failure: {failures[0][0]}/{failures[0][1] or '_'}/{failures[0][2]} → {failures[0][3].get('status')}")
        if len(failures) > 10 and not verbose:
            print(f"... ({len(failures) - 10} more failures suppressed; use --verbose to see all)")
    print(f"VERDICT: {'PASS' if summary['all_pass'] else 'FAIL'}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Verify C inference loop output against reference")
    parser.add_argument("--ref", required=True, help="Reference dump directory (from llamatov_ref_dump.py)")
    parser.add_argument("--c-out", required=True, help="C inference loop output directory (same layout)")
    parser.add_argument("--cos", type=float, default=0.9999, help="Minimum cosine similarity (default 0.9999)")
    parser.add_argument("--abs", type=float, default=1e-4, help="Maximum absolute drift (default 1e-4)")
    parser.add_argument("--verbose", action="store_true", help="Print every stage comparison (not just failures)")
    args = parser.parse_args()

    ref_root = Path(args.ref)
    c_root = Path(args.c_out)

    if not ref_root.exists():
        print(f"ERROR: reference directory not found: {ref_root}")
        sys.exit(2)
    if not c_root.exists():
        print(f"ERROR: C output directory not found: {c_root}")
        sys.exit(2)

    summary = verify(ref_root, c_root, args.cos, args.abs, args.verbose)
    sys.exit(0 if summary["all_pass"] else 1)


if __name__ == "__main__":
    main()
