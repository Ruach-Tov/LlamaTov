#!/usr/bin/env python3
"""Level-1 Byte-Identity Harness: per-layer intermediate comparison.

Complements bench/bit_identical_whole_model.py (Level-3, whole-model logits)
by measuring drift at each layer boundary through the forward-pass. When
Level-3 shows composite drift, Level-1 localizes WHICH layer(s) contribute.

Approach:
    1. Load both sides' per-layer intermediate dumps via
       bench/llama_fixture_loader.py (Doresh's format, June 2026).
    2. For each layer boundary (l_out-0 .. l_out-15), apply the dual-contract
       classify.
    3. Also compare final-tail tensors (norm, result_norm, result_output).
    4. Report the layer-by-layer curve + summary.

Scope (per Iyun's scout-boundary):
    - MEASURES ONLY. Does not modify libllama.so, llama.cpp source (Doresh's
      debug.cpp patch is his substrate work), our forward-pass instrumentation
      (his bpd_layer_bisect work), or L2 emit vocabulary.
    - Prototype quality: reports what's found, not what's hoped.

Build-on-don't-redo:
    - Reuses bench/llama_fixture_loader.py (Doresh's fixture format + loader,
      June 2026 — already in the tree).
    - Reuses bit_identical.py + bit_identical_whole_model.py's dual-contract
      classify.
    - New: name-filtered layer curve reporting + verdict interpretation.

Independence structure:
    - Comparison logic is ours (dual-contract classify).
    - Reference fixture comes from ggml's actual output via Doresh's debug.cpp
      patch — external truth, not derived from shared tooling.
    - Cross-validation with Level-3 harness preserved: both use ggml's actual
      output as reference.

Naming gotcha (per Doresh 41ef3ea5):
    - Per-layer: `l_out-{N}` (N=0..15), `norm-{N}`, `attn_norm-{N}`, etc.
    - FINAL tail (post all layers) has NO suffix: exact match "norm",
      "result_norm", "result_output". Do NOT substring-match "norm" or you
      grab per-layer ones.

Usage:
    # Compare pre-existing fixture directories
    python3 bench/bit_identical_per_layer.py \\
        --ours-fixture-dir /path/to/bpd/dumps \\
        --ref-fixture-dir /path/to/ggml/dumps \\
        --label pre-dot-product-fix
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Import Doresh's fixture loader from the same directory
sys.path.insert(0, str(Path(__file__).parent))
from llama_fixture_loader import load_manifest, FixtureTensor


# ─── Dual-contract classify (mirrors bit_identical.py + Level-3 harness) ───

def ulp_delta_f32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IEEE 754 sign-magnitude ULP distance array, per-element."""
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    B = np.int64(0x80000000)
    ai = np.where(ai < 0, B - ai, ai)
    bi = np.where(bi < 0, B - bi, bi)
    return np.abs(ai - bi)


def classify_layer(ours: np.ndarray, theirs: np.ndarray, tensor_name: str,
                   abs_tol: float = 1e-5) -> dict:
    """Apply the dual-contract classify to one layer's intermediate output.

    Contracts (same as Level-3 harness):
      - BIT_IDENTICAL: 0 ULP everywhere
      - PASS_ABS_TOLERANCE: max abs delta < abs_tol
      - FAIL: neither
    """
    if ours.shape != theirs.shape:
        return {
            "tensor_name": tensor_name,
            "error": f"shape mismatch: {ours.shape} vs {theirs.shape}",
        }
    if ours.dtype != np.float32 or theirs.dtype != np.float32:
        return {
            "tensor_name": tensor_name,
            "error": f"dtype: ours={ours.dtype}, theirs={theirs.dtype}",
        }

    ulp = ulp_delta_f32(ours.flatten(), theirs.flatten())
    abs_diff = np.abs(ours.astype(np.float64) - theirs.astype(np.float64)).flatten()

    bit_identical_mask = ulp == 0
    pass_abs_mask = abs_diff < abs_tol

    return {
        "tensor_name": tensor_name,
        "n_elements": int(ours.size),
        "shape": list(ours.shape),
        "bit_identical": int(bit_identical_mask.sum()),
        "pass_abs_tolerance": int(pass_abs_mask.sum() - bit_identical_mask.sum()),
        "fail": int((~pass_abs_mask).sum()),
        "max_ulp": int(ulp.max()),
        "median_ulp": int(np.median(ulp)),
        "max_abs_diff": float(abs_diff.max()),
        "median_abs_diff": float(np.median(abs_diff)),
        "mean_abs_diff": float(abs_diff.mean()),
    }


# ─── Fixture filtering ────────────────────────────────────────────────────

def filter_layer_tensors(tensors: list, name_pattern: str = "l_out-") -> list:
    """Filter tensors matching a per-layer name pattern like 'l_out-0'..'l_out-15'.

    Returns list sorted by layer number.
    """
    matching = []
    for t in tensors:
        if not t.name.startswith(name_pattern):
            continue
        # Extract layer number after the prefix
        suffix = t.name[len(name_pattern):]
        try:
            layer_num = int(suffix)
        except ValueError:
            continue
        matching.append((layer_num, t))
    matching.sort(key=lambda pair: pair[0])
    return [t for _, t in matching]


def find_exact_name(tensors: list, name: str) -> FixtureTensor:
    """Find a tensor by exact name match. Returns None if not found or ambiguous."""
    matches = [t for t in tensors if t.name == name]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) == 0:
        return None
    else:
        # Ambiguous — the last (most recent) one is likely correct, but flag it
        print(f"  WARN: {len(matches)} tensors named exactly '{name}'; using last (idx={matches[-1].idx})")
        return matches[-1]


# ─── Curve reporting ───────────────────────────────────────────────────────

def print_curve(per_layer: list[dict], final_tail: list[dict] = None):
    """Print the layer-by-layer curve in human-readable form."""
    print(f"{'Layer':>25} {'BitId':>15} {'PassAbs':>10} {'Fail':>10} "
          f"{'MaxULP':>14} {'MaxAbsDiff':>13} {'MedAbsDiff':>13}")
    print("-" * 105)
    for layer in per_layer:
        if "error" in layer:
            print(f"{layer['tensor_name']:>25}  ERROR: {layer['error']}")
            continue
        bit_id_str = f"{layer['bit_identical']:>7}/{layer['n_elements']:<7}"
        print(f"{layer['tensor_name']:>25} "
              f"{bit_id_str:>15} "
              f"{layer['pass_abs_tolerance']:>10} "
              f"{layer['fail']:>10} "
              f"{layer['max_ulp']:>14,} "
              f"{layer['max_abs_diff']:>13.3e} "
              f"{layer['median_abs_diff']:>13.3e}")

    if final_tail:
        print("-" * 105)
        print("Final tail:")
        for layer in final_tail:
            if "error" in layer:
                print(f"{layer['tensor_name']:>25}  ERROR: {layer['error']}")
                continue
            bit_id_str = f"{layer['bit_identical']:>7}/{layer['n_elements']:<7}"
            print(f"{layer['tensor_name']:>25} "
                  f"{bit_id_str:>15} "
                  f"{layer['pass_abs_tolerance']:>10} "
                  f"{layer['fail']:>10} "
                  f"{layer['max_ulp']:>14,} "
                  f"{layer['max_abs_diff']:>13.3e} "
                  f"{layer['median_abs_diff']:>13.3e}")


def summarize_curve(per_layer: list[dict], final_tail: list[dict] = None) -> dict:
    """Extract summary for delta comparison."""
    valid_layers = [l for l in per_layer if "error" not in l]
    all_tensors = valid_layers + [l for l in (final_tail or []) if "error" not in l]

    if not all_tensors:
        return {
            "n_valid_layers": 0,
            "error": "no valid layer comparisons",
        }

    max_abs_diffs = [l["max_abs_diff"] for l in valid_layers]
    max_ulps = [l["max_ulp"] for l in valid_layers]

    return {
        "n_valid_layers": len(valid_layers),
        "n_final_tail": len([l for l in (final_tail or []) if "error" not in l]),
        "max_abs_diff_per_layer": max_abs_diffs,
        "max_ulp_per_layer": max_ulps,
        "layer_with_max_abs_diff": (
            int(np.argmax(max_abs_diffs)) if max_abs_diffs else -1
        ),
        "max_abs_diff_overall": (
            float(max(l["max_abs_diff"] for l in all_tensors)) if all_tensors else 0.0
        ),
        "max_ulp_overall": (
            int(max(l["max_ulp"] for l in all_tensors)) if all_tensors else 0
        ),
        "all_layers_bit_identical": all(
            l["bit_identical"] == l["n_elements"] for l in all_tensors
        ),
        "all_final_tail_bit_identical": all(
            l["bit_identical"] == l["n_elements"]
            for l in (final_tail or []) if "error" not in l
        ),
    }


# ─── Verdict interpretation ─────────────────────────────────────────────────

def compare_curves(baseline: dict, current: dict) -> dict:
    """Compare two per-layer summaries; report delta."""
    delta = {}
    if current.get("all_layers_bit_identical") and current.get("all_final_tail_bit_identical", True):
        delta["interpretation"] = (
            "FLAT-0 ACHIEVED: every layer AND final tail bit-identical vs reference"
        )
    else:
        baseline_curve = baseline.get("max_abs_diff_per_layer", [])
        current_curve = current.get("max_abs_diff_per_layer", [])
        if baseline_curve and current_curve and len(baseline_curve) == len(current_curve):
            reductions = [
                (b - c) / b if b > 0 else 0.0
                for b, c in zip(baseline_curve, current_curve)
            ]
            mean_reduction = sum(reductions) / len(reductions)
            first_fail = next(
                (i for i, layer_diff in enumerate(current_curve) if layer_diff > 1e-6),
                len(current_curve),
            )
            delta["mean_reduction_across_layers"] = mean_reduction
            delta["first_layer_with_drift"] = first_fail
            if first_fail > 0 and mean_reduction > 0:
                delta["interpretation"] = (
                    f"PARTIAL: first drift at layer {first_fail}, "
                    f"{mean_reduction:.1%} mean reduction across all layers"
                )
            elif mean_reduction > 0:
                delta["interpretation"] = (
                    f"REDUCED: drift present from layer 0, "
                    f"{mean_reduction:.1%} mean reduction across layers"
                )
            elif mean_reduction < 0:
                delta["interpretation"] = (
                    f"REGRESSION: drift increased by {-mean_reduction:.1%} mean"
                )
            else:
                delta["interpretation"] = "SAME SHAPE: curve unchanged"
        elif len(baseline_curve) != len(current_curve):
            delta["interpretation"] = (
                f"LAYER COUNT MISMATCH: baseline={len(baseline_curve)}, "
                f"current={len(current_curve)}"
            )
        else:
            delta["interpretation"] = "INCOMPLETE: no per-layer curves to compare"

    delta["baseline_max_abs_diff"] = baseline.get("max_abs_diff_overall")
    delta["current_max_abs_diff"] = current.get("max_abs_diff_overall")
    delta["baseline_max_ulp"] = baseline.get("max_ulp_overall")
    delta["current_max_ulp"] = current.get("max_ulp_overall")
    return delta


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ours-fixture-dir", required=True,
                   help="Directory containing our BPD forward-pass per-layer dumps "
                        "(manifest.tsv + per-tensor .bin files)")
    p.add_argument("--ref-fixture-dir", required=True,
                   help="Directory containing ggml reference per-layer dumps "
                        "(same format)")
    p.add_argument("--layer-name-pattern", default="l_out-",
                   help="Pattern for per-layer tensor names (default: 'l_out-')")
    p.add_argument("--final-tail-names", nargs="*",
                   default=["norm", "result_norm", "result_output"],
                   help="Exact names of final-tail tensors to compare "
                        "(default: norm result_norm result_output)")
    p.add_argument("--abs-tol", type=float, default=1e-5,
                   help="Absolute tolerance for PASS_ABS_TOLERANCE (default: 1e-5)")
    p.add_argument("--baseline", default=None,
                   help="Path to baseline summary JSON for delta comparison")
    p.add_argument("--label", default=None,
                   help="Label for this run (saves labeled result JSON)")
    p.add_argument("--out", default="/tmp/l1_curve_result.json",
                   help="Path to save the full result")
    args = p.parse_args()

    print(f"Level-1 per-layer comparison")
    print(f"  ours:          {args.ours_fixture_dir}")
    print(f"  ref:           {args.ref_fixture_dir}")
    print(f"  name pattern:  {args.layer_name_pattern}")
    print(f"  final tail:    {args.final_tail_names}")
    print(f"  abs_tol:       {args.abs_tol}")
    print()

    # Load both sides via Doresh's llama_fixture_loader
    print("Loading fixtures...")
    ours_tensors = load_manifest(args.ours_fixture_dir)
    ref_tensors = load_manifest(args.ref_fixture_dir)
    print(f"  ours: {len(ours_tensors)} tensors from manifest")
    print(f"  ref:  {len(ref_tensors)} tensors from manifest")

    # Filter per-layer tensors
    ours_layers = filter_layer_tensors(ours_tensors, args.layer_name_pattern)
    ref_layers = filter_layer_tensors(ref_tensors, args.layer_name_pattern)
    print(f"  ours: {len(ours_layers)} per-layer tensors matching '{args.layer_name_pattern}*'")
    print(f"  ref:  {len(ref_layers)} per-layer tensors matching '{args.layer_name_pattern}*'")

    if len(ours_layers) != len(ref_layers):
        print(f"  WARN: layer count mismatch ({len(ours_layers)} vs {len(ref_layers)})")

    # Classify per layer (paired by name — safer than by index in case of gaps)
    print()
    print("Per-layer classification:")
    per_layer = []
    ref_by_name = {t.name: t for t in ref_layers}
    for our_t in ours_layers:
        their_t = ref_by_name.get(our_t.name)
        if their_t is None:
            per_layer.append({
                "tensor_name": our_t.name,
                "error": f"no matching tensor in reference fixture",
            })
            continue
        try:
            ours_arr = our_t.as_numpy().astype(np.float32)
            their_arr = their_t.as_numpy().astype(np.float32)
        except Exception as e:
            per_layer.append({
                "tensor_name": our_t.name,
                "error": f"as_numpy failed: {e}",
            })
            continue
        per_layer.append(classify_layer(ours_arr, their_arr, our_t.name, args.abs_tol))

    # Final tail
    final_tail = []
    for tail_name in args.final_tail_names:
        our_t = find_exact_name(ours_tensors, tail_name)
        their_t = find_exact_name(ref_tensors, tail_name)
        if our_t is None or their_t is None:
            missing_sides = []
            if our_t is None:
                missing_sides.append("ours")
            if their_t is None:
                missing_sides.append("ref")
            final_tail.append({
                "tensor_name": tail_name,
                "error": f"missing from: {', '.join(missing_sides)}",
            })
            continue
        try:
            ours_arr = our_t.as_numpy().astype(np.float32)
            their_arr = their_t.as_numpy().astype(np.float32)
        except Exception as e:
            final_tail.append({
                "tensor_name": tail_name,
                "error": f"as_numpy failed: {e}",
            })
            continue
        final_tail.append(classify_layer(ours_arr, their_arr, tail_name, args.abs_tol))

    print()
    print_curve(per_layer, final_tail)
    print()

    # Summarize
    summary = summarize_curve(per_layer, final_tail)
    print(f"Summary:")
    print(f"  n_valid_layers:            {summary['n_valid_layers']}")
    print(f"  layer_with_max_abs_diff:   {summary['layer_with_max_abs_diff']}")
    print(f"  max_abs_diff_overall:      {summary['max_abs_diff_overall']:.3e}")
    print(f"  max_ulp_overall:           {summary['max_ulp_overall']:,}")
    print(f"  all_layers_bit_identical:  {summary['all_layers_bit_identical']}")
    print(f"  all_final_tail_bit_id:     {summary.get('all_final_tail_bit_identical', 'n/a')}")
    print()

    # Verdict vs baseline
    if args.baseline and os.path.exists(args.baseline):
        with open(args.baseline) as f:
            baseline = json.load(f)
        base_summary = baseline.get("summary", baseline)
        delta = compare_curves(base_summary, summary)
        print(f"DELTA vs baseline ({args.baseline}):")
        for k, v in delta.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")
        print()

    # Save result
    result = {
        "args": vars(args),
        "per_layer": per_layer,
        "final_tail": final_tail,
        "summary": summary,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Full result saved to {args.out}")

    if args.label:
        labeled = Path(__file__).parent / f"l1_result_{args.label}.json"
        with open(labeled, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Labeled result saved to {labeled}")

    # Exit 0 if flat-0, 1 otherwise
    is_flat_zero = (
        summary.get("all_layers_bit_identical", False)
        and summary.get("all_final_tail_bit_identical", True)
    )
    sys.exit(0 if is_flat_zero else 1)


if __name__ == "__main__":
    main()
