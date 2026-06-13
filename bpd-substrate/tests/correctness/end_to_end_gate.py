#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""end_to_end_gate.py — full-forward inference comparison vs ggml.

Runs our substrate's full forward pass on the same prompt as the captured
fixture. Compares the resulting logit vector to ggml's captured
result_output.bin. Emits a JSON report including:

  - argmax_match: did we pick the same top-1 token?
  - pearson_correlation: logit-vector correlation
  - cosine_similarity: angle between logit vectors
  - top_k_overlap: how many of our top-k overlap ggml's top-k
  - distribution_stats: mean/max/L2 diffs

Multi-sovereign-verifiable: anyone running this harness on the same
fixture should get the same numbers (modulo their substrate's drift).
Different verifiers comparing reports is the gold standard for the
LlamaTov correctness story.
"""
import argparse
import ctypes
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HARNESS_VERSION = "1.0.0"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "bench"))

from llama_fixture_loader import load_tensor  # noqa: E402


def gather_hardware_info():
    info = {"cpu_model": "unknown", "isa": [], "compiler": "unknown",
            "hostname": socket.gethostname(), "platform": platform.platform()}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name") and info["cpu_model"] == "unknown":
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                if line.startswith("flags") and not info["isa"]:
                    flags = line.split(":", 1)[1].strip().split()
                    info["isa"] = sorted(f for f in flags if f in
                        {"sse", "sse2", "sse3", "ssse3", "sse4_1", "sse4_2",
                         "avx", "avx2", "avx512f", "fma", "f16c"})
    except FileNotFoundError:
        pass
    try:
        gcc = subprocess.check_output(["gcc", "--version"], stderr=subprocess.STDOUT).decode()
        info["compiler"] = gcc.split("\n")[0]
    except Exception:
        pass
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fixture-dir", required=True,
                   help="LLAMA_DUMP_DIR containing ggml's result_output.bin")
    p.add_argument("--so", required=True, help="Path to bpd_cpu.so")
    p.add_argument("--gguf", required=True)
    p.add_argument("--tokens", required=True,
                   help="Comma-separated token IDs (must match the captured prompt)")
    p.add_argument("--verifier", default=None)
    p.add_argument("--report", default=f"/tmp/end_to_end_report_{os.getuid()}.json")
    p.add_argument("--orchestrator", default=str(REPO_ROOT / "bench" / "bpd_llamatov_infer.py"),
                   help="Path to the orchestrator (default: bench/bpd_llamatov_infer.py)")
    args = p.parse_args()

    print(f"[harness] LlamaTov end-to-end gate v{HARNESS_VERSION}")

    # 1. Find ggml's reference logits in the fixture
    fix_dir = Path(args.fixture_dir)
    result_files = sorted(fix_dir.glob("*result_output.bin"))
    if not result_files:
        print(f"[error] no result_output.bin in {fix_dir}", file=sys.stderr)
        sys.exit(2)
    ref_logits = load_tensor(str(result_files[-1])).as_numpy().reshape(-1)
    print(f"[harness] ggml ref_logits: shape={ref_logits.shape}, argmax={int(np.argmax(ref_logits))}")

    # 2. Run the substrate orchestrator with --dump-logits
    logits_path = f"/tmp/_e2e_logits_{os.getuid()}.npy"
    cmd = [
        sys.executable, args.orchestrator,
        "--gguf", args.gguf,
        "--so", args.so,
        "--tokens", args.tokens,
        "--n-generate", "1",
        "--dump-logits", logits_path,
        "--out", f"/tmp/_e2e_result_{os.getuid()}.json",
    ]
    # Delete stale logits BEFORE running to prevent false passes
    our_logits_path = logits_path.replace(".npy", "_step0.npy")
    for stale in [logits_path, our_logits_path]:
        if Path(stale).exists():
            Path(stale).unlink()
            print(f"[harness] deleted stale {stale}")

    print(f"[harness] running orchestrator: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    dt = time.time() - t0
    print(f"[harness] orchestrator completed in {dt:.1f}s (rc={result.returncode})")

    # Check return code BEFORE looking for logits
    if result.returncode != 0:
        print(f"[FATAL] orchestrator CRASHED (rc={result.returncode})", file=sys.stderr)
        print(f"[FATAL] stdout: {result.stdout[-500:]}", file=sys.stderr)
        print(f"[FATAL] stderr: {result.stderr[-500:]}", file=sys.stderr)
        sys.exit(3)

    # 3. Load our substrate's logits
    if not Path(our_logits_path).exists():
        print(f"[error] orchestrator did not produce {our_logits_path}", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(2)
    # Verify the logits file was created AFTER we started the orchestrator
    logits_mtime = Path(our_logits_path).stat().st_mtime
    if logits_mtime < t0:
        print(f"[FATAL] logits file is STALE (mtime={logits_mtime:.0f} < start={t0:.0f})", file=sys.stderr)
        print(f"[FATAL] This means the orchestrator did not produce fresh output.", file=sys.stderr)
        sys.exit(4)

    our_logits = np.load(our_logits_path)
    print(f"[harness] our logits: shape={our_logits.shape}, argmax={int(np.argmax(our_logits))}")

    # 4. Compute comparison metrics
    if our_logits.shape != ref_logits.shape:
        print(f"[error] shape mismatch: ours={our_logits.shape} ref={ref_logits.shape}", file=sys.stderr)
        sys.exit(2)

    diff = our_logits - ref_logits
    max_abs = float(np.abs(diff).max())
    mean_abs = float(np.abs(diff).mean())
    l2_diff = float(np.linalg.norm(diff))
    l2_ours = float(np.linalg.norm(our_logits))
    l2_ref = float(np.linalg.norm(ref_logits))
    cos_sim = float(np.dot(our_logits, ref_logits) / (l2_ours * l2_ref + 1e-12))
    pearson = float(np.corrcoef(our_logits, ref_logits)[0, 1])

    our_argmax = int(np.argmax(our_logits))
    ref_argmax = int(np.argmax(ref_logits))
    argmax_match = our_argmax == ref_argmax

    # Top-k overlap
    k = 10
    our_topk = set(np.argsort(our_logits)[-k:].tolist())
    ref_topk = set(np.argsort(ref_logits)[-k:].tolist())
    overlap = len(our_topk & ref_topk)

    # Per-token-position rank lookups
    our_argmax_in_ref_rank = int((ref_logits > ref_logits[our_argmax]).sum() + 1)
    ref_argmax_in_our_rank = int((our_logits > our_logits[ref_argmax]).sum() + 1)

    print(f"\n[results]")
    print(f"  argmax_match:     {argmax_match} (ours={our_argmax}, ref={ref_argmax})")
    print(f"  pearson_corr:     {pearson:.6f}")
    print(f"  cosine_sim:       {cos_sim:.6f}")
    print(f"  top_{k}_overlap:    {overlap}/{k}")
    print(f"  max_abs:          {max_abs:.6f}")
    print(f"  mean_abs:         {mean_abs:.6f}")
    print(f"  L2_diff:          {l2_diff:.6f}")
    print(f"  L2_ratio:         {l2_ours/l2_ref:.6f}")
    print(f"  our_argmax in ref dist: rank {our_argmax_in_ref_rank}")
    print(f"  ref_argmax in our dist: rank {ref_argmax_in_our_rank}")

    if argmax_match:
        verdict = "PASS_BIT_IDENTICAL_ARGMAX"
    elif pearson > 0.99:
        verdict = "PASS_HIGH_CORRELATION"
    elif pearson > 0.9:
        verdict = "PASS_GOOD_CORRELATION"
    else:
        verdict = "FAIL"
    print(f"\n[verdict] {verdict}")

    report = {
        "harness_version": HARNESS_VERSION,
        "verifier": args.verifier or os.environ.get("USER", "unknown"),
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fixture_dir": args.fixture_dir,
        "model": {"path": args.gguf},
        "hardware": gather_hardware_info(),
        "tokens": [int(t) for t in args.tokens.split(",")],
        "ours": {
            "argmax": our_argmax,
            "top_k_logits": {int(i): float(our_logits[i])
                              for i in sorted(our_topk, key=lambda j: -our_logits[j])},
        },
        "reference": {
            "argmax": ref_argmax,
            "top_k_logits": {int(i): float(ref_logits[i])
                              for i in sorted(ref_topk, key=lambda j: -ref_logits[j])},
        },
        "metrics": {
            "argmax_match": argmax_match,
            "pearson_correlation": pearson,
            "cosine_similarity": cos_sim,
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
            "L2_diff": l2_diff,
            "L2_ratio_ours_to_ref": l2_ours / l2_ref,
            f"top_{k}_overlap": overlap,
            f"top_{k}_overlap_pct": overlap / k,
            "our_argmax_rank_in_ref": our_argmax_in_ref_rank,
            "ref_argmax_rank_in_ours": ref_argmax_in_our_rank,
        },
        "verdict": verdict,
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[harness] wrote {args.report}")


if __name__ == "__main__":
    main()
