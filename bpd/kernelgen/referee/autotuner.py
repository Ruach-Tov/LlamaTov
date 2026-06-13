#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""
autotuner.py — pick the best kernel config per (op, problem-size), with a cache.

The path past the single-fixed-kernel plateau (we hit ~70% of cuBLAS at N=1024
with ONE config; a per-size autotuner can do better because the optimal tile
shape shifts with the problem dimensions).

Design — a tuning CACHE keyed by (op, N):
  tune(op, N):
    1. cache hit  -> return the stored best config (instant)
    2. cache miss -> run a FOCUSED sweep over candidate configs at this N
                     (verify correctness gate + measure GFLOPS), store the
                     winner, return it.
  emit(op, N)   -> generate the kernel for the tuned config (ready to compile).

The cache makes tuning amortized: pay the sweep once per (op, N), serve instantly
after. The candidate set is the top configs from the global sweep (the optimum is
size-stable, so a focused sweep suffices — full 198-pt sweep at large N is too slow).

Reuses gemm_sweep.py's machinery (gen_compile, verify, measure, tcheck_v2).
Cache: SWEEP_WORK/autotune_cache.json.

Usage:
  autotuner.py tune  --op matmul --n 1024
  autotuner.py emit  --op matmul --n 1024 --out kernel.cu
  autotuner.py show                                  # dump the cache
Author: Iyun, 2026-06-07
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemm_sweep as gs   # reuse the sweep machinery
import numpy as np

CACHE = f"{gs.WORK}/autotune_cache.json"

# Candidate configs to try at each N — the top performers from the global v2+vec
# sweep (all vec=4, BK=32 dominate; include a few shapes for size adaptivity).
# (BM, BN, BK, TM, TN) — vec is always 4 here (proven best).
CANDIDATES = [
    (64, 64, 32, 8, 4), (32, 64, 32, 4, 4), (64, 128, 32, 8, 8),
    (128, 32, 32, 8, 4), (32, 128, 32, 8, 4), (64, 32, 32, 8, 4),
    (128, 64, 32, 8, 8), (128, 128, 32, 8, 8), (64, 128, 16, 8, 8),
]
VEC = 4

def load_cache():
    return json.load(open(CACHE)) if os.path.exists(CACHE) else {}

def save_cache(c):
    json.dump(c, open(CACHE, "w"), indent=2)

def key(op, n):
    return f"{op}:{n}"

def tune(op, n, verbose=True):
    """Return the best config for (op, n), sweeping + caching on miss."""
    cache = load_cache()
    k = key(op, n)
    if k in cache:
        if verbose: print(f"  [cache hit] {k} -> {cache[k]['config']} ({cache[k]['gflops']:.0f} GFLOPS)")
        return cache[k]

    if op != "matmul":
        raise SystemExit(f"autotuner currently tunes matmul only (got {op})")

    # focused sweep over candidates at this N
    import torch, warnings; warnings.filterwarnings("ignore")
    vn = min(512, n)   # verify at a small square (correctness is size-independent)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((vn, vn)).astype(np.float32)
    B = rng.standard_normal((vn, vn)).astype(np.float32)
    A.tofile(f"{gs.WORK}/A.bin"); B.tofile(f"{gs.WORK}/B.bin")
    ref = torch.matmul(torch.from_numpy(A), torch.from_numpy(B)).numpy().ravel()

    if verbose:
        print(f"  [cache miss] tuning {k}: sweeping {len(CANDIDATES)} candidates at N={n}...")
    best = None
    for p in CANDIDATES:
        cubin = gs.gen_compile(p, VEC)
        if not cubin:
            continue
        res = gs.verify(cubin, p, A, B, ref, vn)
        mean_rel, max_rel = (res if res is not None else (None, None))
        # accumulation-order-aware gate: small mean (systematic) AND bounded max
        # (no single wrong element). A reordered-but-correct GEMM passes; a bug fails.
        if mean_rel is None or mean_rel >= 1e-4 or max_rel >= 1e-2:
            os.remove(cubin); continue
        gflops, pct = gs.measure(cubin, p, n)
        os.remove(cubin)
        if gflops and (best is None or gflops > best["gflops"]):
            best = {"config": list(p), "vec": VEC, "gflops": gflops, "pct_peak": pct,
                    "mean_rel": mean_rel, "max_rel": max_rel}
        if verbose and gflops:
            print(f"    {p} vec={VEC}: {gflops:.0f} GFLOPS")
    if best is None:
        raise SystemExit("no candidate verified+measured")
    cache[k] = best
    save_cache(cache)
    if verbose: print(f"  [tuned] {k} -> {best['config']} ({best['gflops']:.0f} GFLOPS, {best['pct_peak']:.1f}% peak)")
    return best

def emit(op, n, out):
    """Generate the kernel .cu for the tuned config of (op, n)."""
    best = tune(op, n, verbose=False)
    BM, BN, BK, TM, TN = best["config"]
    import subprocess
    g = (f'consult("{gs.EMIT}/gemm_tiled_from_space.pl"), '
         f'emit_gemm_tiled({BM},{BN},{BK},{TM},{TN},true,{best["vec"]},0,"{out}"), halt')
    subprocess.run([gs.SWIPL, "-q", "-g", g], capture_output=True, text=True, timeout=60)
    print(f"  emitted tuned {op} N={n} config={best['config']} -> {out}")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tune"); t.add_argument("--op", default="matmul"); t.add_argument("--n", type=int, required=True)
    e = sub.add_parser("emit"); e.add_argument("--op", default="matmul"); e.add_argument("--n", type=int, required=True); e.add_argument("--out", required=True)
    sub.add_parser("show")
    a = ap.parse_args()
    if a.cmd == "tune": tune(a.op, a.n)
    elif a.cmd == "emit": emit(a.op, a.n, a.out)
    elif a.cmd == "show":
        c = load_cache()
        print(f"=== autotune cache ({len(c)} entries) ===")
        for k, v in sorted(c.items()):
            print(f"  {k:16} -> {v['config']} vec={v['vec']}  {v['gflops']:.0f} GFLOPS ({v['pct_peak']:.1f}% peak)")

if __name__ == "__main__":
    main()
