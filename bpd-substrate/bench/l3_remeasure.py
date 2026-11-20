#!/usr/bin/env python3
"""Re-measurement wrapper for the Level-3 harness.

Purpose: given a substrate fix has landed on the current tree, rebuild
bpd_cpu.so, run bench/bit_identical_whole_model.py against the standard
llama-3.2-1b GGUF, and diff the result against a saved baseline.

Report:
  - If baseline was drift and current is 0-ULP: publish achievement
  - If baseline was drift and current is still drift: report the delta
    (did the shape change? did the max_ulp shrink?)
  - If baseline was 0-ULP and current is drift: regression, flag

Scope: measurement automation. Does not modify anything except rebuild
the C library. Same scout-boundary as bit_identical_whole_model.py.

Usage:
  python3 bench/l3_remeasure.py \\
    [--baseline bench/l3_baseline_pre_dispatcher_fix.json] \\
    [--gguf external/llama.cpp/models/llama-3.2-1b-instruct-q8_0.gguf] \\
    [--tokens '128000,9906,11,856,836,374'] \\
    [--n-generate 2] \\
    [--label descriptive-label-for-this-run]
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path("/home/heath/Ruach-Tov")
BPD_SUBSTRATE = REPO_ROOT / "bpd-substrate"
LLAMA_CPP = REPO_ROOT / "external/llama.cpp"

DEFAULT_BASELINE = BPD_SUBSTRATE / "bench/l3_baseline_pre_dispatcher_fix.json"
DEFAULT_GGUF = LLAMA_CPP / "models/llama-3.2-1b-instruct-q8_0.gguf"
DEFAULT_TOKENS = "128000,9906,11,856,836,374"
DEFAULT_N_GEN = 2

BPD_SO_PATH = "/tmp/bpd_test/build/bpd_cpu.so"
LLAMA_DUMP_BIN = "/tmp/llama_logits_dump"
HARNESS = BPD_SUBSTRATE / "bench/bit_identical_whole_model.py"


def sh(cmd, cwd=None, check=True, capture=True):
    """Run shell command, return (rc, stdout, stderr)."""
    r = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd,
                       capture_output=capture, text=True)
    if check and r.returncode != 0:
        print(f"  FAIL: {cmd}")
        print(f"  stderr: {r.stderr[:500]}")
        sys.exit(1)
    return r.returncode, r.stdout, r.stderr


def rebuild_bpd_so():
    """Rebuild bpd_cpu.so from current source at /tmp/bpd_test/build/bpd_cpu.so.
    
    Uses the same location bit_identical_whole_model.py expects. Copies from
    build/bpd_cpu.so (the substrate's local build) — assumes the substrate has
    already built its own copy. If not, tries a direct gcc build with the
    flags the substrate needs.
    """
    src_so = BPD_SUBSTRATE / "build/bpd_cpu.so"
    dst_so = Path(BPD_SO_PATH)
    dst_so.parent.mkdir(parents=True, exist_ok=True)
    
    if src_so.exists():
        # Compare mtimes: if src is newer, copy
        if not dst_so.exists() or src_so.stat().st_mtime > dst_so.stat().st_mtime:
            print(f"  [build] copying {src_so} → {dst_so}")
            sh(f"cp {src_so} {dst_so}")
        else:
            print(f"  [build] {dst_so} already up-to-date (mtime match)")
    else:
        print(f"  [build] {src_so} not found; running substrate's Makefile")
        rc, out, err = sh("make build/bpd_cpu.so", cwd=BPD_SUBSTRATE, check=False)
        if rc != 0:
            # Fall back to direct gcc with expected flags
            print(f"  [build] Makefile failed; trying direct gcc")
            sh(f"gcc -O2 -mavx2 -mfma -shared -fPIC "
               f"-o {dst_so} {BPD_SUBSTRATE}/bench/bpd_cpu.c -lm")
        elif src_so.exists():
            sh(f"cp {src_so} {dst_so}")
    
    if not dst_so.exists():
        print(f"  FAIL: {dst_so} still missing after build attempts")
        sys.exit(1)
    
    size = dst_so.stat().st_size
    print(f"  [build] bpd_cpu.so ready: {size:,} bytes at {dst_so}")


def ensure_llama_dump_bin():
    """Ensure /tmp/llama_logits_dump exists; rebuild if source is newer."""
    src_c = BPD_SUBSTRATE / "bench/llama_logits_dump.c"
    dst_bin = Path(LLAMA_DUMP_BIN)
    
    if dst_bin.exists() and src_c.stat().st_mtime <= dst_bin.stat().st_mtime:
        print(f"  [build] {dst_bin} already up-to-date")
        return
    
    print(f"  [build] compiling {src_c} → {dst_bin}")
    cmd = (
        f"gcc -O2 -o {dst_bin} {src_c} "
        f"-I {LLAMA_CPP}/include -I {LLAMA_CPP}/ggml/include "
        f"-L {LLAMA_CPP}/build/bin "
        f"-Wl,-rpath,{LLAMA_CPP}/build/bin -lllama"
    )
    sh(cmd)
    print(f"  [build] llama_logits_dump ready: {dst_bin.stat().st_size:,} bytes")


def run_harness(gguf: str, tokens: str, n_generate: int, out_path: str):
    """Run bit_identical_whole_model.py and return the parsed result JSON."""
    cmd = [
        sys.executable, str(HARNESS),
        "--gguf", gguf,
        "--tokens", tokens,
        "--n-generate", str(n_generate),
        "--out", out_path,
    ]
    print(f"  [harness] running: {' '.join(cmd[:6])} ...")
    r = subprocess.run(cmd, cwd=BPD_SUBSTRATE, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f"  FAIL: harness exit {r.returncode}")
        print(f"  stdout:\n{r.stdout[-1000:]}")
        print(f"  stderr:\n{r.stderr[-1000:]}")
        sys.exit(1)
    with open(out_path) as f:
        return json.load(f)


def summarize(name: str, result: dict) -> dict:
    """Extract compact summary for comparison."""
    if not result.get("per_step"):
        return {"name": name, "steps": 0}
    per_step = result["per_step"]
    max_ulp_across = max(s["max_ulp"] for s in per_step)
    total_vocab = per_step[0]["n_vocab"]
    fail_at_step0 = per_step[0]["fail"]
    bit_identical_at_step0 = per_step[0]["bit_identical"]
    all_argmax_match = all(s["argmax_match"] for s in per_step)
    max_abs_diff = max(s.get("max_abs_diff", 0.0) for s in per_step)
    return {
        "name": name,
        "steps": len(per_step),
        "vocab_size": total_vocab,
        "max_ulp_across_steps": max_ulp_across,
        "max_abs_diff_across_steps": max_abs_diff,
        "step0_bit_identical": bit_identical_at_step0,
        "step0_fail": fail_at_step0,
        "argmax_all_match": all_argmax_match,
        "verdict": (
            "0-ULP achieved" if all(
                s["bit_identical"] == s["n_vocab"] for s in per_step
            ) else f"drift detected (max ULP {max_ulp_across:.2e})"
        ),
    }


def compare(baseline: dict, current: dict) -> dict:
    """Compare two summaries and return a delta report."""
    delta = {
        "baseline_verdict": baseline["verdict"],
        "current_verdict": current["verdict"],
    }
    if current["vocab_size"] == baseline["vocab_size"]:
        delta["step0_bit_identical_delta"] = (
            current["step0_bit_identical"] - baseline["step0_bit_identical"]
        )
        delta["step0_fail_delta"] = current["step0_fail"] - baseline["step0_fail"]
    if baseline["max_ulp_across_steps"] > 0:
        ratio = current["max_ulp_across_steps"] / baseline["max_ulp_across_steps"]
        delta["max_ulp_ratio_current_over_baseline"] = ratio
    if baseline["max_abs_diff_across_steps"] > 0:
        ratio = current["max_abs_diff_across_steps"] / baseline["max_abs_diff_across_steps"]
        delta["max_abs_diff_ratio_current_over_baseline"] = ratio
    
    # Interpretation
    baseline_zero = "0-ULP" in baseline["verdict"]
    current_zero = "0-ULP" in current["verdict"]
    if not baseline_zero and current_zero:
        delta["interpretation"] = "IMPROVEMENT: drift closed → 0-ULP achieved"
    elif baseline_zero and current_zero:
        delta["interpretation"] = "NO CHANGE: both runs bit-identical"
    elif baseline_zero and not current_zero:
        delta["interpretation"] = "REGRESSION: baseline was 0-ULP, current drifts"
    else:
        # Both drift; compare shape
        if delta.get("step0_fail_delta", 0) < 0:
            delta["interpretation"] = (
                f"PARTIAL IMPROVEMENT: drift shrunk. step0_fail went from "
                f"{baseline['step0_fail']}/{baseline['vocab_size']} to "
                f"{current['step0_fail']}/{current['vocab_size']} "
                f"({-delta['step0_fail_delta']} more positions now match)"
            )
        elif delta.get("step0_fail_delta", 0) > 0:
            delta["interpretation"] = (
                f"WORSE: drift expanded. step0_fail went from "
                f"{baseline['step0_fail']} to {current['step0_fail']}"
            )
        else:
            delta["interpretation"] = "SAME SHAPE: drift structure unchanged"
    return delta


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--baseline", default=str(DEFAULT_BASELINE),
                   help=f"Path to baseline result JSON (default: {DEFAULT_BASELINE})")
    p.add_argument("--gguf", default=str(DEFAULT_GGUF),
                   help=f"Path to GGUF (default: {DEFAULT_GGUF})")
    p.add_argument("--tokens", default=DEFAULT_TOKENS,
                   help=f"Comma-separated prompt tokens (default: {DEFAULT_TOKENS})")
    p.add_argument("--n-generate", type=int, default=DEFAULT_N_GEN,
                   help=f"Number of tokens to generate (default: {DEFAULT_N_GEN})")
    p.add_argument("--label", default=None,
                   help="Label for this run (e.g., 'post-dispatcher-fix'). "
                        "If given, saves current result as "
                        "bench/l3_result_<label>.json for future comparisons.")
    p.add_argument("--out", default="/tmp/l3_remeasure_result.json",
                   help="Output path for the current run's raw harness result")
    args = p.parse_args()

    print("=" * 76)
    print(f"Level-3 Re-Measurement Wrapper")
    print(f"  timestamp:  {datetime.now().isoformat(timespec='seconds')}")
    print(f"  baseline:   {args.baseline}")
    print(f"  gguf:       {args.gguf}")
    print(f"  tokens:     {args.tokens}")
    print(f"  n_generate: {args.n_generate}")
    if args.label:
        print(f"  label:      {args.label}")
    print("=" * 76)
    print()

    # Load baseline
    if not os.path.exists(args.baseline):
        print(f"FAIL: baseline {args.baseline} not found")
        sys.exit(1)
    with open(args.baseline) as f:
        baseline_result = json.load(f)
    baseline_summary = summarize("baseline", baseline_result)
    print(f"Baseline: {baseline_summary['verdict']}")
    print(f"  step0: {baseline_summary['step0_bit_identical']}/{baseline_summary['vocab_size']} bit-identical, "
          f"{baseline_summary['step0_fail']} fail")
    print(f"  argmax across steps: {'match' if baseline_summary['argmax_all_match'] else 'DIVERGE'}")
    print()

    # Rebuild artifacts
    print("Building artifacts...")
    rebuild_bpd_so()
    ensure_llama_dump_bin()
    print()

    # Run harness
    print("Running Level-3 harness...")
    current_result = run_harness(args.gguf, args.tokens, args.n_generate, args.out)
    current_summary = summarize("current", current_result)
    print()
    print(f"Current: {current_summary['verdict']}")
    print(f"  step0: {current_summary['step0_bit_identical']}/{current_summary['vocab_size']} bit-identical, "
          f"{current_summary['step0_fail']} fail")
    print(f"  argmax across steps: {'match' if current_summary['argmax_all_match'] else 'DIVERGE'}")
    print()

    # Compare
    delta = compare(baseline_summary, current_summary)
    print("=" * 76)
    print("DELTA:")
    for k, v in delta.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    print("=" * 76)

    # Save labeled result if requested
    if args.label:
        labeled_path = BPD_SUBSTRATE / f"bench/l3_result_{args.label}.json"
        with open(labeled_path, "w") as f:
            json.dump({
                "label": args.label,
                "timestamp": datetime.now().isoformat(timespec='seconds'),
                "args": vars(args),
                "summary": current_summary,
                "delta_vs_baseline": delta,
                "raw_result": current_result,
            }, f, indent=2)
        print(f"\nLabeled result saved to {labeled_path}")

    # Exit code: 0 if improved or unchanged, non-zero if regressed
    if "REGRESSION" in delta.get("interpretation", ""):
        sys.exit(3)
    sys.exit(0)


if __name__ == "__main__":
    main()
