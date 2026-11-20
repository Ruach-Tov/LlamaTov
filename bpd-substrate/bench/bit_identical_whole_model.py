#!/usr/bin/env python3
"""Level-3 Byte-Identity Harness: whole-model logits diff BPD vs llama.cpp.

Motivation:
    bench/bit_identical.py verifies per-kernel bit-identity.
    bench/bpd_llamatov_infer.py runs the whole-model forward pass but
    only compares argmax token IDs against llama-cli reference.
    
    The gap: after the softmax f64 multi-dispatch mirror landed (Doresh's
    c28fb4bd9), we believe whole-model 0-ULP vs llama.cpp is now
    achievable. But belief is not measurement. This harness measures.

Level-3 in the scout deliverable's three-level taxonomy:
    Level 3: whole-model logits diff (cheapest first step, this file)
    Level 1: per-layer intermediate comparison (diagnostic if L3 shows drift)
    Level 2: per-sub-op comparison (deep diagnostic if L1 localizes drift)

Approach:
    1. Run our BPD forward-pass (bench/bpd_llamatov_infer.py) with
       --dump-logits on a fixed prompt, save per-step logits as .npy.
    2. Run llama.cpp's forward-pass via libllama.so ctypes bridge on
       the same prompt, capture per-step logits.
    3. Apply the dual-contract classify from bit_identical.py:
       BIT_IDENTICAL (0 ULP), PASS_ABS_TOLERANCE (small), or FAIL.
    4. Report per-token, per-vocab-position drift.

Scope (per Iyun's scout-boundary ruling):
    - MEASURES ONLY. Does not modify libllama.so, llama.cpp source,
      our forward-pass, or L2 emit vocabulary.
    - Prototype quality: reports what's found, not what's hoped.

Build-on-don't-redo:
    - Reuses bit_identical.py's dual-contract classify
    - Reuses bpd_llamatov_infer.py's --dump-logits path
    - New: libllama.so ctypes wrapper for reference logits

Usage:
    python3 bit_identical_whole_model.py \\
        --gguf /path/to/llama-3.2-1b.gguf \\
        --prompt "hello world" \\
        --n-generate 4 \\
        --llama-so external/llama.cpp/build/bin/libllama.so \\
        --bpd-so /tmp/bpd_test/build/bpd_cpu.so
"""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


# ─── Reference logits extraction via C companion tool ──────────────────────
#
# Design note: llama_model_default_params() and llama_context_default_params()
# return structs BY VALUE, which is painful to call correctly from ctypes
# across ABI boundaries. A small C companion (bench/llama_logits_dump.c)
# sidesteps the by-value struct problem entirely. Build:
#
#   gcc -O2 -o /tmp/llama_logits_dump bench/llama_logits_dump.c \
#       -I ../external/llama.cpp/include \
#       -I ../external/llama.cpp/ggml/include \
#       -L ../external/llama.cpp/build/bin \
#       -Wl,-rpath,$(pwd)/../external/llama.cpp/build/bin -lllama


def llama_extract_logits(
    dump_binary: str,
    gguf_path: str,
    tokens_csv: str,
    n_generate: int,
    workdir: str,
) -> list[np.ndarray]:
    """Extract per-step logits from llama.cpp via a small C companion tool.
    
    Design note: llama_model_default_params() and llama_context_default_params()
    return structs BY VALUE, which is painful across ctypes ABI. A ~200-line C
    helper (bench/llama_logits_dump.c) sidesteps the by-value struct problem
    entirely and is cleaner than fighting ctypes.
    
    Returns a list of np.ndarray, one per generated step, each shape
    [vocab_size] and dtype=float32.
    """
    prefix = os.path.join(workdir, "ref_logits")
    cmd = [
        dump_binary,
        "--gguf", gguf_path,
        "--tokens", tokens_csv,
        "--n-generate", str(n_generate),
        "--out-prefix", prefix,
    ]
    print(f"  [ref] running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(
            f"llama_logits_dump failed (exit {result.returncode}):\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    # Load per-step binary files (int32 vocab_size + vocab_size × float32)
    logits = []
    for step in range(n_generate):
        path = f"{prefix}_step{step}.bin"
        if not os.path.exists(path):
            print(f"  [ref] warning: {path} missing (may be OK if generation stopped early)")
            break
        with open(path, "rb") as f:
            vocab_size = np.frombuffer(f.read(4), dtype=np.int32)[0]
            step_logits = np.frombuffer(f.read(int(vocab_size) * 4), dtype=np.float32)
            if step_logits.shape[0] != vocab_size:
                raise RuntimeError(
                    f"short read: {path} has {step_logits.shape[0]}/{vocab_size} floats"
                )
            # Copy so the buffer is not tied to the file handle
            logits.append(np.array(step_logits, dtype=np.float32))
    return logits


def bpd_extract_logits(
    infer_script: str,
    gguf_path: str,
    bpd_so: str,
    prompt: str,
    n_generate: int,
    workdir: str,
) -> list[np.ndarray]:
    """Run bpd_llamatov_infer.py with --dump-logits and load the results.
    
    Returns a list of np.ndarray matching llama_extract_logits's format.
    """
    dump_prefix = os.path.join(workdir, "bpd_logits.npy")
    # bpd_llamatov_infer.py takes --tokens (comma-separated) not --prompt directly
    # per the current script. We tokenize via llama.cpp first for consistency.
    # For prototype: caller passes pre-tokenized as CSV in prompt arg.
    cmd = [
        sys.executable, infer_script,
        "--gguf", gguf_path,
        "--so", bpd_so,
        "--tokens", prompt,  # expect CSV of token IDs
        "--n-generate", str(n_generate),
        "--dump-logits", dump_prefix,
        "--out", os.path.join(workdir, "bpd_result.json"),
    ]
    print(f"  [bpd] running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(
            f"bpd_llamatov_infer.py failed (exit {result.returncode}):\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    # Load per-step logits
    logits = []
    for step in range(n_generate):
        path = dump_prefix.replace(".npy", f"_step{step}.npy")
        if not os.path.exists(path):
            print(f"  [bpd] warning: {path} missing (may be OK if generation stopped early)")
            break
        logits.append(np.load(path))
    return logits


# ─── Dual-contract classify (mirrors bit_identical.py) ─────────────────────

def ulp_delta_f32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IEEE 754 sign-magnitude ULP distance array, per-element."""
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    B = np.int64(0x80000000)
    ai = np.where(ai < 0, B - ai, ai)
    bi = np.where(bi < 0, B - bi, bi)
    return np.abs(ai - bi)


def classify_dual_contract(
    ours: np.ndarray,
    theirs: np.ndarray,
    abs_tol: float = 1e-5,
) -> dict:
    """Apply the dual-contract classify from bit_identical.py to a logits pair.
    
    Contracts:
      - BIT_IDENTICAL: 0 ULP delta everywhere
      - PASS_ABS_TOLERANCE: max abs delta < abs_tol (looser accept)
      - FAIL: neither
    
    Returns per-position classification + summary stats.
    """
    assert ours.shape == theirs.shape, f"shape mismatch: {ours.shape} vs {theirs.shape}"
    assert ours.dtype == np.float32 and theirs.dtype == np.float32
    
    ulp = ulp_delta_f32(ours, theirs)
    abs_diff = np.abs(ours.astype(np.float64) - theirs.astype(np.float64))
    
    bit_identical_mask = ulp == 0
    pass_abs_mask = abs_diff < abs_tol
    
    n_total = ours.size
    n_bit_identical = int(bit_identical_mask.sum())
    n_pass_abs = int(pass_abs_mask.sum())
    n_fail = int((~pass_abs_mask).sum())  # doesn't match either
    
    argmax_ours = int(np.argmax(ours))
    argmax_theirs = int(np.argmax(theirs))
    
    return {
        "n_vocab": n_total,
        "bit_identical": n_bit_identical,
        "pass_abs_tolerance": n_pass_abs - n_bit_identical,  # exclude bit-identical
        "fail": n_fail,
        "max_ulp": int(ulp.max()),
        "median_ulp": int(np.median(ulp)),
        "max_abs_diff": float(abs_diff.max()),
        "argmax_match": argmax_ours == argmax_theirs,
        "argmax_ours": argmax_ours,
        "argmax_theirs": argmax_theirs,
    }


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Level-3 harness: whole-model logits diff BPD vs llama.cpp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--gguf", required=True,
                   help="Path to the GGUF model file (e.g., llama-3.2-1b.gguf)")
    p.add_argument("--tokens", required=True,
                   help="Comma-separated pre-tokenized prompt (e.g., '128000,9906')")
    p.add_argument("--n-generate", type=int, default=4,
                   help="Number of tokens to generate (default: 4)")
    p.add_argument("--llama-dump-bin", default="/tmp/llama_logits_dump",
                   help="Path to compiled llama_logits_dump binary "
                        "(build with: gcc -O2 -o /tmp/llama_logits_dump "
                        "bench/llama_logits_dump.c -I../external/llama.cpp/include "
                        "-I../external/llama.cpp/ggml/include "
                        "-L../external/llama.cpp/build/bin "
                        "-Wl,-rpath,$(pwd)/../external/llama.cpp/build/bin -lllama)")
    p.add_argument("--bpd-so", default="/tmp/bpd_test/build/bpd_cpu.so",
                   help="Path to bpd_cpu.so (default: /tmp/bpd_test build)")
    p.add_argument("--infer-script", default="bench/bpd_llamatov_infer.py",
                   help="Path to bpd_llamatov_infer.py")
    p.add_argument("--abs-tol", type=float, default=1e-5,
                   help="Absolute tolerance for PASS_ABS_TOLERANCE (default: 1e-5)")
    p.add_argument("--out", default="/tmp/l3_harness_result.json",
                   help="Result JSON path")
    args = p.parse_args()

    print(f"Level-3 harness")
    print(f"  gguf:               {args.gguf}")
    print(f"  tokens:             {args.tokens}")
    print(f"  n_generate:         {args.n_generate}")
    print(f"  llama_logits_dump:  {args.llama_dump_bin}")
    print(f"  bpd_cpu.so:         {args.bpd_so}")
    print()

    with tempfile.TemporaryDirectory(prefix="l3harness_") as workdir:
        # Extract our BPD logits
        print("Step 1: Extract BPD logits...")
        try:
            bpd_logits = bpd_extract_logits(
                infer_script=args.infer_script,
                gguf_path=args.gguf,
                bpd_so=args.bpd_so,
                prompt=args.tokens,
                n_generate=args.n_generate,
                workdir=workdir,
            )
            print(f"  [bpd] extracted {len(bpd_logits)} step logits")
        except (subprocess.TimeoutExpired, RuntimeError, FileNotFoundError) as e:
            print(f"  [bpd] FAIL: {e}")
            print()
            print("HONEST PARTIAL (c): BPD extraction failed. "
                  "Cannot proceed with comparison until BPD forward-pass runs.")
            sys.exit(1)

        # Extract llama.cpp logits via C companion tool
        print("Step 2: Extract llama.cpp reference logits...")
        if not os.path.exists(args.llama_dump_bin):
            print(f"  [ref] FAIL: {args.llama_dump_bin} not found")
            print(f"  Build with: see --help for llama_logits_dump build command")
            sys.exit(1)
        try:
            ref_logits = llama_extract_logits(
                dump_binary=args.llama_dump_bin,
                gguf_path=args.gguf,
                tokens_csv=args.tokens,
                n_generate=args.n_generate,
                workdir=workdir,
            )
            print(f"  [ref] extracted {len(ref_logits)} step logits")
        except (subprocess.TimeoutExpired, RuntimeError) as e:
            print(f"  [ref] FAIL: {e}")
            print()
            print("HONEST PARTIAL (c): reference extraction failed. "
                  "See stderr for llama_logits_dump error.")
            sys.exit(1)

        # Compare
        print("Step 3: Dual-contract classify per step...")
        results = []
        for step, (ours, theirs) in enumerate(zip(bpd_logits, ref_logits)):
            r = classify_dual_contract(ours, theirs, abs_tol=args.abs_tol)
            r["step"] = step
            results.append(r)
            status = "✓ BIT-IDENTICAL" if r["bit_identical"] == r["n_vocab"] else \
                     f"⚠ drift: {r['fail']}/{r['n_vocab']} fail (max ULP: {r['max_ulp']})"
            argmax_status = "✓" if r["argmax_match"] else \
                            f"✗ (ours={r['argmax_ours']} vs theirs={r['argmax_theirs']})"
            print(f"  step {step}: {status}, argmax {argmax_status}")

        # Write result
        with open(args.out, "w") as f:
            json.dump({
                "args": vars(args),
                "per_step": results,
                "summary": {
                    "n_steps": len(results),
                    "all_bit_identical": all(
                        r["bit_identical"] == r["n_vocab"] for r in results
                    ),
                    "argmax_all_match": all(r["argmax_match"] for r in results),
                    "max_ulp_across_steps": max(r["max_ulp"] for r in results),
                },
            }, f, indent=2)
        print()
        print(f"Result written to {args.out}")

        # Verdict
        summary = results[0] if results else None
        if all(r["bit_identical"] == r["n_vocab"] for r in results):
            print("VERDICT: whole-model 0-ULP ACHIEVED vs llama.cpp")
        else:
            print(f"VERDICT: composite drift detected. Max ULP across steps: "
                  f"{max(r['max_ulp'] for r in results)}. See per-step detail.")


if __name__ == "__main__":
    main()
