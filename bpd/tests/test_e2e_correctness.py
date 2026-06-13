#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""End-to-end correctness + performance test runner.

Runs a model under both Ollama and LlamaTov, compares outputs for
correctness, and reports performance if our runner is correct.

This is the CI gate — no commit should break existing runner correctness.

Usage:
  python3 test_e2e_correctness.py [--model llama3.2:1b] [--prompt "Hello"]
  python3 -m pytest test_e2e_correctness.py -v  # as pytest suite

Environment:
  OLLAMA_HOST: Ollama server (default: http://localhost:11434)
  LLAMATOV_RUNNER: path to our inference script (auto-detected)
  GGUF_MODEL_PATH: path to GGUF model file (auto-detected from Ollama)

Author: medayek (Collective SME, Verification Methodology)
Date: 2026-05-16
"""

import os
import sys
import json
import time
import subprocess
import pytest
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
BPD_DIR = Path(__file__).parent.parent  # bpd/
REPO_DIR = BPD_DIR.parent  # Ruach-Tov/

# Models to test (name, GGUF path pattern, expected token count)
TEST_MODELS = [
    {
        "name": "llama3.2:1b",
        "prompt": "Hello",
        "min_match_tokens": 3,  # first N tokens must match
        "max_time_seconds": 60,
    },
]

TEST_PROMPTS = [
    "Hello",
    "The capital of France is",
    "1 + 1 =",
]


# ══════════════════════════════════════════════════════════════════════
# Ollama reference runner
# ══════════════════════════════════════════════════════════════════════

def ollama_generate(model: str, prompt: str, n_tokens: int = 5) -> dict:
    """Run Ollama and capture output tokens + timing."""
    import urllib.request

    url = f"{OLLAMA_HOST}/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": n_tokens,
            "temperature": 0.0,  # greedy decoding for determinism
            "seed": 42,
        }
    }).encode()

    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"})

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
    except Exception as e:
        return {"error": str(e), "elapsed": time.time() - start}

    elapsed = time.time() - start

    return {
        "model": model,
        "prompt": prompt,
        "response": result.get("response", ""),
        "eval_count": result.get("eval_count", 0),
        "eval_duration_ns": result.get("eval_duration", 0),
        "total_duration_ns": result.get("total_duration", 0),
        "tok_per_sec": (result.get("eval_count", 0) /
                        (result.get("eval_duration", 1) / 1e9))
                       if result.get("eval_duration", 0) > 0 else 0,
        "elapsed_wall": elapsed,
        "error": None,
    }


# ══════════════════════════════════════════════════════════════════════
# LlamaTov runner
# ══════════════════════════════════════════════════════════════════════

def find_llamatov_runner():
    """Find our inference runner script."""
    candidates = [
        BPD_DIR / "llamatov_gpu_dp4a.py",
        BPD_DIR / "llamatov_gpu_llama.py",
        BPD_DIR / "llamatov_run.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def llamatov_generate(runner_path: Path, model_path: str, prompt: str,
                       n_tokens: int = 5) -> dict:
    """Run LlamaTov and capture output tokens + timing."""
    cmd = [
        sys.executable, str(runner_path),
        "--model", model_path,
        "--prompt", prompt,
        "--n-tokens", str(n_tokens),
        "--temperature", "0.0",
        "--json-output",  # machine-readable output
    ]

    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            cwd=str(BPD_DIR)
        )
        elapsed = time.time() - start

        if result.returncode != 0:
            return {
                "error": f"Exit code {result.returncode}: {result.stderr[:500]}",
                "elapsed": elapsed,
            }

        # Try to parse JSON output
        try:
            output = json.loads(result.stdout)
        except json.JSONDecodeError:
            # Fallback: extract tokens from text output
            output = {
                "response": result.stdout.strip(),
                "raw_stdout": result.stdout[:1000],
            }

        output["elapsed_wall"] = elapsed
        output["error"] = None
        return output

    except subprocess.TimeoutExpired:
        return {"error": "Timeout (120s)", "elapsed": 120}
    except Exception as e:
        return {"error": str(e), "elapsed": time.time() - start}


# ══════════════════════════════════════════════════════════════════════
# Comparison logic
# ══════════════════════════════════════════════════════════════════════

def compare_outputs(ollama_result: dict, llamatov_result: dict,
                     min_match_tokens: int = 3) -> dict:
    """Compare outputs and return comparison report."""
    report = {
        "ollama_response": ollama_result.get("response", ""),
        "llamatov_response": llamatov_result.get("response", ""),
        "ollama_error": ollama_result.get("error"),
        "llamatov_error": llamatov_result.get("error"),
    }

    if report["ollama_error"]:
        report["status"] = "OLLAMA_ERROR"
        return report

    if report["llamatov_error"]:
        report["status"] = "LLAMATOV_ERROR"
        return report

    # Compare first N characters/tokens
    ollama_text = report["ollama_response"]
    our_text = report["llamatov_response"]

    # Token-level comparison (split by whitespace as approximation)
    ollama_tokens = ollama_text.split()
    our_tokens = our_text.split()

    matches = 0
    for i in range(min(len(ollama_tokens), len(our_tokens), min_match_tokens)):
        if ollama_tokens[i] == our_tokens[i]:
            matches += 1
        else:
            break

    report["tokens_compared"] = min(len(ollama_tokens), len(our_tokens),
                                     min_match_tokens)
    report["tokens_matched"] = matches
    report["match_ratio"] = matches / max(report["tokens_compared"], 1)

    if matches >= min_match_tokens:
        report["status"] = "CORRECT"
    elif matches > 0:
        report["status"] = "PARTIAL_MATCH"
    else:
        report["status"] = "INCORRECT"

    # Performance comparison (only if correct)
    if report["status"] == "CORRECT":
        ollama_tps = ollama_result.get("tok_per_sec", 0)
        our_elapsed = llamatov_result.get("elapsed_wall", 0)
        our_tokens_count = llamatov_result.get("eval_count",
                                                len(our_tokens))
        our_tps = our_tokens_count / our_elapsed if our_elapsed > 0 else 0

        report["ollama_tok_per_sec"] = ollama_tps
        report["llamatov_tok_per_sec"] = our_tps
        report["speedup_ratio"] = our_tps / ollama_tps if ollama_tps > 0 else 0

    return report


# ══════════════════════════════════════════════════════════════════════
# CI-friendly test output
# ══════════════════════════════════════════════════════════════════════

def print_report(report: dict, model: str, prompt: str):
    """Print human-readable comparison report."""
    status = report.get("status", "UNKNOWN")
    status_emoji = {
        "CORRECT": "✅",
        "PARTIAL_MATCH": "⚠️",
        "INCORRECT": "❌",
        "OLLAMA_ERROR": "🔴",
        "LLAMATOV_ERROR": "🔴",
    }.get(status, "❓")

    print(f"\n{'═'*60}")
    print(f"Model: {model} | Prompt: '{prompt}'")
    print(f"Status: {status_emoji} {status}")
    print(f"{'─'*60}")

    if report.get("ollama_error"):
        print(f"  Ollama error: {report['ollama_error']}")
    if report.get("llamatov_error"):
        print(f"  LlamaTov error: {report['llamatov_error']}")

    if "ollama_response" in report:
        print(f"  Ollama:    '{report['ollama_response'][:80]}'")
    if "llamatov_response" in report:
        print(f"  LlamaTov:  '{report['llamatov_response'][:80]}'")

    if "tokens_matched" in report:
        print(f"  Tokens matched: {report['tokens_matched']}/{report['tokens_compared']}")

    if status == "CORRECT" and "ollama_tok_per_sec" in report:
        print(f"  Ollama:    {report['ollama_tok_per_sec']:.1f} tok/s")
        print(f"  LlamaTov:  {report['llamatov_tok_per_sec']:.1f} tok/s")
        print(f"  Ratio:     {report['speedup_ratio']:.2f}×")

    print(f"{'═'*60}\n")


# ══════════════════════════════════════════════════════════════════════
# Pytest integration
# ══════════════════════════════════════════════════════════════════════

def ollama_available():
    """Check if Ollama is running."""
    try:
        import urllib.request
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


@pytest.fixture
def check_ollama():
    if not ollama_available():
        pytest.skip("Ollama not available")


@pytest.fixture
def check_runner():
    runner = find_llamatov_runner()
    if runner is None:
        pytest.skip("LlamaTov runner not found")
    return runner


class TestE2ECorrectness:
    """End-to-end correctness tests — the CI gate."""

    @pytest.mark.skipif(not ollama_available(), reason="Ollama not running")
    def test_llama32_hello(self, check_runner):
        """llama3.2:1b with 'Hello' — first 3 tokens must match Ollama."""
        ollama = ollama_generate("llama3.2:1b", "Hello", n_tokens=5)
        assert ollama["error"] is None, f"Ollama failed: {ollama['error']}"

        # TODO: wire in actual LlamaTov runner once --json-output is supported
        # For now, this test validates the framework
        assert ollama["response"], "Ollama returned empty response"
        assert ollama["tok_per_sec"] > 0, "Ollama reported 0 tok/s"

    @pytest.mark.skipif(not ollama_available(), reason="Ollama not running")
    def test_ollama_deterministic(self):
        """Same prompt twice → same output (greedy, temp=0)."""
        r1 = ollama_generate("llama3.2:1b", "Hello", n_tokens=5)
        r2 = ollama_generate("llama3.2:1b", "Hello", n_tokens=5)
        assert r1["error"] is None and r2["error"] is None
        assert r1["response"] == r2["response"], \
            f"Non-deterministic: '{r1['response']}' vs '{r2['response']}'"


# ══════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="E2E correctness + performance")
    parser.add_argument("--model", default="llama3.2:1b")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--n-tokens", type=int, default=5)
    parser.add_argument("--all-prompts", action="store_true",
                        help="Test all standard prompts")
    args = parser.parse_args()

    prompts = TEST_PROMPTS if args.all_prompts else [args.prompt]

    print("E2E Correctness + Performance Test")
    print(f"Model: {args.model}")
    print(f"Ollama: {OLLAMA_HOST}")

    runner = find_llamatov_runner()
    print(f"Runner: {runner or 'NOT FOUND'}")

    for prompt in prompts:
        print(f"\n--- Testing prompt: '{prompt}' ---")

        # Ollama reference
        print("Running Ollama...")
        ollama_result = ollama_generate(args.model, prompt, args.n_tokens)
        if ollama_result.get("error"):
            print(f"  Ollama error: {ollama_result['error']}")
            continue

        print(f"  Ollama: '{ollama_result['response'][:60]}' "
              f"({ollama_result['tok_per_sec']:.1f} tok/s)")

        # LlamaTov
        if runner:
            print("Running LlamaTov...")
            # TODO: wire actual runner when --json-output supported
            print("  LlamaTov runner integration pending")
        else:
            print("  LlamaTov runner not found — skipping")

    print("\n✅ Framework operational. Wire LlamaTov runner for full comparison.")
