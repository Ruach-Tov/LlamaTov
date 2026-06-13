#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""e2e_bench.py — END-TO-END tok/sec comparison capability for LlamaTov.

A PERMANENT, repeatable measurement tool (not a one-off). Measures whole-model
generation throughput via llama-bench, compares any variant build against a
baseline, reports tok/sec with variance, the ratio, and (optionally) a per-op
time profile. Designed so "compare our e2e tok/sec vs ollama/stock" is one call.

This is the foundational layer of the profiling capability:
  e2e_bench.py            -> tok/sec (this file)
  + cupti trace/roofline  -> per-kernel achieved-vs-peak (mavchin's GPU PMU)
  + referee_logits_0ulp   -> bit-identity gate (patched == stock)

Usage:
  e2e_bench.py --binary <llama-bench> --gguf <model> [--baseline-binary <stock>]
  e2e_bench.py --binary X --gguf M --label "SoA-vec128" --baseline-binary stock --baseline-label ollama-stock
  e2e_bench.py ... --json out.json    # structured output for the skyline plot
"""
import argparse, subprocess, json, os, re, statistics, sys

DEFAULT_GGUF = "/tmp/llamatov-data/ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45"
DRIVER = "/run/opengl-driver/lib"

def run_bench(binary, gguf, ngl, n_gen, reps, n_prompt, extra_ld):
    """Run llama-bench, parse the t/s table. Returns list of dicts per test row."""
    ld = ":".join([DRIVER, os.path.dirname(binary)] + ([extra_ld] if extra_ld else []))
    cmd = [binary, "-m", gguf, "-ngl", str(ngl), "-r", str(reps)]
    if n_gen:    cmd += ["-n", str(n_gen)]
    if n_prompt is not None: cmd += ["-p", str(n_prompt)]
    env = dict(os.environ, LD_LIBRARY_PATH=ld)
    out = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    rows = []
    for line in out.stdout.splitlines():
        # parse the markdown table rows: | model | size | params | backend | ngl | test | t/s |
        if "|" in line and ("tg" in line or "pp" in line) and "t/s" not in line:
            cols = [c.strip() for c in line.split("|") if c.strip()]
            if len(cols) >= 6:
                test = cols[-2]
                m = re.search(r"([\d.]+)\s*±\s*([\d.]+)", cols[-1])
                if m:
                    rows.append({"test": test, "tps": float(m.group(1)), "stddev": float(m.group(2))})
    if not rows:
        # surface errors to help debugging
        return [], out.stdout + "\n" + out.stderr
    return rows, None

def measure(label, binary, gguf, ngl, n_gen, reps, n_prompt, extra_ld):
    rows, err = run_bench(binary, gguf, ngl, n_gen, reps, n_prompt, extra_ld)
    if err:
        print(f"[{label}] FAILED: {err[:400]}", file=sys.stderr)
        return None
    res = {"label": label, "binary": binary, "tests": rows}
    for r in rows:
        print(f"  [{label}] {r['test']}: {r['tps']:.2f} ± {r['stddev']:.2f} tok/s")
    return res

def main():
    ap = argparse.ArgumentParser(description="e2e tok/sec comparison (measured, no projections)")
    ap.add_argument("--binary", required=True, help="llama-bench binary (the variant)")
    ap.add_argument("--label", default="variant")
    ap.add_argument("--baseline-binary", help="stock llama-bench (the denominator)")
    ap.add_argument("--baseline-label", default="stock")
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--n-gen", type=int, default=128, help="tokens to generate (tg)")
    ap.add_argument("--n-prompt", type=int, default=0, help="prompt tokens (pp); 0 = decode-only")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--extra-ld", default=None, help="extra LD_LIBRARY_PATH entry")
    ap.add_argument("--json", help="write structured results here (for skyline plots)")
    a = ap.parse_args()

    print(f"=== e2e tok/sec — MEASURED (llama-bench, -ngl {a.ngl} -n {a.n_gen} -r {a.reps}) ===")
    results = {}
    if a.baseline_binary:
        base = measure(a.baseline_label, a.baseline_binary, a.gguf, a.ngl, a.n_gen, a.reps, a.n_prompt, a.extra_ld)
        results["baseline"] = base
    var = measure(a.label, a.binary, a.gguf, a.ngl, a.n_gen, a.reps, a.n_prompt, a.extra_ld)
    results["variant"] = var

    # ratio per test (MEASURED only — no projection)
    if a.baseline_binary and base and var:
        print(f"\n--- MEASURED e2e ratio ({a.label} vs {a.baseline_label}) ---")
        bmap = {r["test"]: r for r in base["tests"]}
        for r in var["tests"]:
            if r["test"] in bmap:
                bt = bmap[r["test"]]["tps"]
                print(f"  {r['test']}: {r['tps']:.2f} vs {bt:.2f} tok/s = {r['tps']/bt:.4f}x MEASURED")
    if a.json:
        json.dump(results, open(a.json, "w"), indent=2)
        print(f"\n[wrote {a.json}]")

if __name__ == "__main__":
    main()
