#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""generate_fixtures.py — produce a substantively-substantive fixture suite for L.1 closure verification.

Generates op-by-op binary tensor dumps for each prompt in the test suite
by invoking the patched llama-eval-callback with LLAMA_DUMP_DIR set.

Each fixture lives in /tmp/llama_fixtures/<name>/ and contains:
  - All intermediate op outputs as .bin files
  - manifest.tsv listing op order
  - prompt.txt with the input text
  - tokens.txt with the tokenization

Why this matters: 2 fixtures ("hello", "Hello, my name is") don't exercise
the substrate's full code paths. Richer coverage tests:
  - Long context (KV cache F16 vs F32 matters more at length)
  - Diverse token distributions (code, numbers, unicode)
  - Different head usage patterns (not all heads activate for English prose)
  - Edge cases (single token, repeated tokens, extreme entropy)

This is the substrate-design discipline at the empirical-floor: more
diverse fixtures = more substantively-trustworthy verification claims.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Default GGUF on the enclave
DEFAULT_GGUF = "/tmp/llamatov-data/ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45"
DEFAULT_BINARY = "/tmp/llama_cpp_test/build/bin/llama-eval-callback"
DEFAULT_OUTPUT_BASE = "/tmp/llama_fixtures"


# ─── Fixture suite definition ─────────────────────────────────────────
#
# Each entry: (name, prompt, n_predict, why)
#
# Tier 1: Coverage tests (small, fast, broad)
TIER_1 = [
    ("bos_only", "", 1, "smallest case: BOS-only context, trivial softmax"),
    ("hello", "hello", 1, "current minimal case (compatibility with prior tests)"),
    ("hello_my_name", "Hello, my name is", 1, "current 6-token case (compatibility)"),
    ("code_python", "def fibonacci(n):", 1, "code tokens, distinct vocab subset"),
    ("numbers", "1 + 1 = 2, 2 + 2 = ", 1, "numeric tokens, arithmetic continuation"),
    ("punctuation", "What?! Really... yes; \"absolutely\".", 1, "punctuation-dense tokens"),
    ("repetition", "the the the the the the", 1, "same token repeated 6x; tests positional encoding"),
    ("unicode_jp", "東京は日本の首都です", 1, "Japanese multi-byte UTF-8; distinct token range"),
    ("unicode_emoji", "I love 🌅 sunset ⛵ sailing 🕯️ candle", 1, "emoji tokens; rare vocab + multi-byte"),
]

# Tier 2: Length stress tests (slower; substantively-revealing for KV cache)
# We construct longer prompts by repeating sentences with variation
LONG_TEXT_BASE = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! "
    "Sphinx of black quartz, judge my vow. "
    "Two driven jocks help fax my big quiz. "
    "Five quacking zephyrs jolt my wax bed. "
    "The five boxing wizards jump quickly. "
    "Jaded zombies acted quaintly but kept driving their oxen forward. "
)

TIER_2 = [
    ("length_32", LONG_TEXT_BASE * 1, 1, "~32 tokens; first multi-block softmax"),
    ("length_128", LONG_TEXT_BASE * 4, 1, "~128 tokens; common short-prompt length"),
    # ("length_512", LONG_TEXT_BASE * 16, 1, "~512 tokens; F16 cache precision matters"),
    # length_2048 would max out our 2048 context — skip for now to keep generation time bounded
]

# Tier 3: Diagnostic-targeted fixtures (single-prompt, short)
TIER_3 = [
    ("instruction", "Please write a short poem about", 1, "instruction-following prompt"),
    ("question_who", "Who is the president of France?", 1, "factual question, mid-context"),
    ("question_what", "What color is the sky?", 1, "simple factual; high-confidence prediction"),
]


def all_fixtures(tiers):
    """Yield (name, prompt, n_predict, why) for all selected tiers."""
    if "1" in tiers:
        yield from TIER_1
    if "2" in tiers:
        yield from TIER_2
    if "3" in tiers:
        yield from TIER_3


def generate_one(name, prompt, n_predict, gguf, binary, output_base, threads=1, timeout=180):
    """Generate one fixture by invoking patched llama-eval-callback.

    Returns (status, n_files, elapsed_seconds, error_msg).
    """
    fixture_dir = Path(output_base) / name
    if fixture_dir.exists():
        shutil.rmtree(fixture_dir)
    fixture_dir.mkdir(parents=True, exist_ok=True)

    # Persist prompt for posterity
    (fixture_dir / "prompt.txt").write_text(prompt if prompt else "<BOS-only>")

    env = os.environ.copy()
    env["LLAMA_DUMP_DIR"] = str(fixture_dir)

    # Build command. For BOS-only we pass empty string but llama-cli needs a -p.
    # The patched binary will still emit ops even for trivial input.
    cmd = [
        binary,
        "-m", gguf,
        "-p", prompt if prompt else " ",  # llama-cli rejects truly empty -p; use single space
        "-n", str(n_predict),
        "--threads", str(threads),
    ]

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
        elapsed = time.time() - t0
    except subprocess.TimeoutExpired:
        return ("timeout", 0, time.time() - t0, f"timeout after {timeout}s")

    if proc.returncode != 0:
        return ("error", 0, elapsed, f"rc={proc.returncode}: {proc.stderr[:200]}")

    # Count generated files
    files = sorted(fixture_dir.glob("*.bin"))
    n_files = len(files)
    manifest = fixture_dir / "manifest.tsv"
    has_manifest = manifest.exists()

    # Extract token IDs if visible in stderr (the binary prints them)
    # For now just record the prompt; tokenization will be redone at test time
    (fixture_dir / "n_predict.txt").write_text(str(n_predict))

    return ("ok" if has_manifest and n_files > 10 else "incomplete", n_files, elapsed, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", default=DEFAULT_GGUF, help="GGUF model path")
    parser.add_argument("--binary", default=DEFAULT_BINARY, help="patched llama-eval-callback path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_BASE, help="output directory base")
    parser.add_argument("--tiers", default="1,2,3", help="comma-separated tier numbers to generate (1,2,3)")
    parser.add_argument("--threads", type=int, default=1, help="threads for inference (more = faster but less deterministic)")
    parser.add_argument("--timeout", type=int, default=300, help="per-fixture timeout seconds")
    parser.add_argument("--dry-run", action="store_true", help="just list what would be generated")
    parser.add_argument("--only", help="generate only fixture with this name (overrides tier selection)")
    args = parser.parse_args()

    tiers = set(args.tiers.split(","))
    if args.only:
        # Find matching fixture in any tier
        all_specs = list(TIER_1) + list(TIER_2) + list(TIER_3)
        matching = [s for s in all_specs if s[0] == args.only]
        if not matching:
            print(f"[error] no fixture named {args.only!r}", file=sys.stderr)
            return 1
        specs = matching
    else:
        specs = list(all_fixtures(tiers))

    Path(args.output).mkdir(parents=True, exist_ok=True)

    print(f"[generate_fixtures] starting; {len(specs)} fixtures to generate")
    print(f"[generate_fixtures] gguf: {args.gguf}")
    print(f"[generate_fixtures] binary: {args.binary}")
    print(f"[generate_fixtures] output: {args.output}")
    print()

    if args.dry_run:
        for name, prompt, n_predict, why in specs:
            print(f"  {name:20s}  n_predict={n_predict}  why: {why}")
            print(f"  {'':20s}  prompt: {prompt[:80]!r}")
        return 0

    results = []
    total_t0 = time.time()
    for i, (name, prompt, n_predict, why) in enumerate(specs):
        print(f"[{i+1}/{len(specs)}] {name}: {prompt[:60]!r}")
        status, n_files, elapsed, err = generate_one(
            name, prompt, n_predict, args.gguf, args.binary, args.output,
            threads=args.threads, timeout=args.timeout,
        )
        marker = "✅" if status == "ok" else "❌"
        msg = f"  {marker} {status} ({n_files} files in {elapsed:.1f}s)"
        if err:
            msg += f" — {err}"
        print(msg)
        results.append((name, status, n_files, elapsed, err))

    total_elapsed = time.time() - total_t0

    print()
    print(f"=== Summary ===")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    ok_count = sum(1 for r in results if r[1] == "ok")
    print(f"Successful: {ok_count}/{len(results)}")
    print()
    print(f"{'Name':25s} {'Status':12s} {'Files':>8s} {'Time':>8s}")
    print(f"{'-'*25} {'-'*12} {'-'*8} {'-'*8}")
    for name, status, n_files, elapsed, err in results:
        marker = "✅" if status == "ok" else "❌"
        print(f"{name:25s} {marker} {status:10s} {n_files:>8d} {elapsed:>7.1f}s")
        if err:
            print(f"  └─ {err}")

    return 0 if ok_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
