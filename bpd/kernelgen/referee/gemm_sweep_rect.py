#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
import os as _os, sys as _sys
def _bpd_root(_p=_os.path.dirname(_os.path.abspath(__file__))):
    while _p != '/' and _os.path.basename(_p) != 'bpd':
        _p = _os.path.dirname(_p)
    return _p if _os.path.basename(_p) == 'bpd' else _os.path.dirname(_os.path.abspath(__file__))
_BPD = _bpd_root()

"""gemm_sweep_rect — autotune RECTANGULAR GEMM C[MxN]=A[MxK]*B[KxN] over the
(BM,BN,BK,TM,TN) tile space. The general-case tuner (square gemm_sweep is the
special case). Built to chase cuBLAS's conv-GEMM strategy read from its SASS:
bigger register tiles (TM/TN up to 8 = 64-elem micro-tile) spread the reduction
across more accumulators -> better accuracy AND arithmetic intensity, all fp32.

For each tile config:
  EMIT    emit_gemm_tiled_rect(...,emit_fma=true,...) -> CUDA -> cubin
  VERIFY  tcheck_rect verify vs numpy ref, accumulation-aware gate
          (rel<1e-4 where |ref| large; abs<1e-3 where near-zero — the
           'deterministic but non-invariant' tolerance)
  PERF    tcheck_rect perf -> GFLOPS, % of cuBLAS

Usage: gemm_sweep_rect.py --m 128 --n 93312 --k 576 [--verify-m/-n/-k SMALL]
       [--limit N] [--out rect_sweep]
The conv-GEMM shape M=128 N=93312 K=576 is the default target.
By: Iyun, 2026-06-08
"""
import os, sys, json, argparse, subprocess
import numpy as np
import sys as _sys; _sys.path.insert(0, _os.path.join(_BPD, "lib"))
import toolchain as tc

REPO = "<repo>/Ruach-Tov"
EMIT = f"{REPO}/bpd/kernelgen/emitters"
WORK = "/tmp/gpu-work/sweep"
CUDA = tc.cuda_root()  # was hardcoded /nix path; now the shared toolchain (ENV-SHIFT defense)
REDIST = "/nix/store/560i0agldlr2h4h3bx6mq2lifw6w1iaa-cuda-native-redist-12.8/lib"
STUBS = "/nix/store/3n5kqxw44phkj9bcwdzdpj1z31q4ajg9-cuda_cudart-12.8.90-stubs/lib/stubs"
SWIPL = "/run/current-system/sw/bin/swipl"
ENV = tc.nvcc_env()  # shared toolchain env (driver + toolkit lib paths)
FIELDS = ["BM", "BN", "BK", "TM", "TN"]


def points():
    """Valid (BM,BN,BK,TM,TN) tile configs from the Prolog space."""
    g = (f'consult("{EMIT}/gemm_gpu_space.pl"), '
         'forall(gpu_gemm_point(BM,BN,BK,TM,TN), '
         'format("~w ~w ~w ~w ~w~n",[BM,BN,BK,TM,TN])), halt')
    out = subprocess.run([SWIPL, "-q", "-g", g], capture_output=True, text=True, env=ENV).stdout
    pts = []
    for ln in out.splitlines():
        p = ln.split()
        if len(p) == 5:
            pts.append(tuple(int(x) for x in p))
    return pts


def gen_compile(point):
    """emit rect kernel (emit_fma=true) + compile to cubin. Returns path or None."""
    BM, BN, BK, TM, TN = point
    cu = f"{WORK}/rect_{BM}_{BN}_{BK}_{TM}_{TN}.cu"
    g = (f'consult("{EMIT}/gemm_tiled_from_space.pl"), '
         f'emit_gemm_tiled_rect({BM},{BN},{BK},{TM},{TN},true,1,"{cu}"), halt')
    subprocess.run([SWIPL, "-q", "-g", g], capture_output=True, text=True, env=ENV)
    if not os.path.exists(cu):
        return None
    cubin = cu.replace(".cu", ".cubin")
    r = subprocess.run(tc.nvcc_compile_cmd(cu, cubin),
                       capture_output=True, text=True, env=ENV, timeout=120)
    return cubin if (r.returncode == 0 and os.path.exists(cubin)) else None


def nthreads(point):
    BM, BN, BK, TM, TN = point
    return (BM // TM) * (BN // TN)


def verify(cubin, point, M, N, K, A, B, ref):
    """Run tcheck_rect verify, return (mean_rel, n_genuinely_bad) or None.
    Accumulation-aware: an element is WRONG only if rel>1e-2 AND abs>1e-3 (so
    near-zero refs judged by absolute error, large refs by relative)."""
    BM, BN, BK, TM, TN = point
    NTH = nthreads(point)
    try: os.remove(f"{WORK}/Csw.bin")
    except FileNotFoundError: pass
    subprocess.run([f"{WORK}/tcheck_rect", cubin, str(NTH), str(BM), str(BN),
                    str(M), str(N), str(K), "verify"],
                   capture_output=True, text=True, env=ENV, cwd=WORK, timeout=60)
    if not os.path.exists(f"{WORK}/Csw.bin"):
        return None
    got = np.fromfile(f"{WORK}/Csw.bin", np.float32)
    if got.size != M * N:
        return None
    got = got.reshape(M, N)
    d = np.abs(got - ref)
    mean_rel = float((d / (np.abs(ref) + np.abs(got) + 1e-4)).mean())
    nbad = int(((d / (np.abs(ref) + 1e-30) > 1e-2) & (d > 1e-3)).sum())
    return mean_rel, nbad


def measure(cubin, point, M, N, K, iters=30):
    """tcheck_rect perf -> GFLOPS."""
    BM, BN, BK, TM, TN = point
    NTH = nthreads(point)
    r = subprocess.run([f"{WORK}/tcheck_rect", cubin, str(NTH), str(BM), str(BN),
                        str(M), str(N), str(K), "perf", str(iters)],
                       capture_output=True, text=True, env=ENV, cwd=WORK, timeout=60)
    for tok in r.stdout.split():
        if tok == "GFLOPS":  # "... %.1f GFLOPS ..."
            pass
    import re
    m = re.search(r"([\d.]+)\s+GFLOPS", r.stdout)
    return float(m.group(1)) if m else None


def cublas_gflops(M, N, K):
    """torch (cuBLAS) GFLOPS for the same shape — the reference to chase."""
    try:
        import torch, time
        dev = torch.device("cuda")
        a = torch.randn(M, K, device=dev); b = torch.randn(K, N, device=dev)
        for _ in range(5): a @ b
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(30): a @ b
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 30 * 1e3
        return 2.0 * M * N * K / (ms * 1e-3) / 1e9
    except Exception as e:
        print("cublas ref failed:", str(e)[:80]); return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--n", type=int, default=93312)
    ap.add_argument("--k", type=int, default=576)
    # verify on a SMALLER shape (same character) to keep verify fast
    ap.add_argument("--verify-m", type=int, default=128)
    ap.add_argument("--verify-n", type=int, default=1024)
    ap.add_argument("--verify-k", type=int, default=576)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=f"{WORK}/rect_sweep")
    args = ap.parse_args()

    M, N, K = args.m, args.n, args.k
    vM, vN, vK = args.verify_m, args.verify_n, args.verify_k
    print(f"Rectangular GEMM sweep: PERF on M={M} N={N} K={K}; VERIFY on M={vM} N={vN} K={vK}\n")

    cub_gf = cublas_gflops(M, N, K)
    print(f"cuBLAS (torch) reference: {cub_gf:.0f} GFLOPS\n" if cub_gf else "cuBLAS ref: n/a\n")

    # verify data (small shape)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((vM, vK)).astype(np.float32)
    B = rng.standard_normal((vK, vN)).astype(np.float32)
    ref = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)

    pts = points()
    if args.limit:
        pts = pts[:args.limit]
    print(f"{len(pts)} tile configs\n")

    rows = []
    for p in pts:
        tag = f"BM{p[0]} BN{p[1]} BK{p[2]} TM{p[3]} TN{p[4]} ({nthreads(p)}thr)"
        cubin = gen_compile(p)
        if not cubin:
            print(f"  {tag:34} compile-err"); continue
        # verify on small shape
        A.tofile(f"{WORK}/A.bin"); B.tofile(f"{WORK}/B.bin")
        vres = verify(cubin, p, vM, vN, vK, A, B, ref)
        if vres is None:
            print(f"  {tag:34} verify-NO-OUTPUT"); continue
        mean_rel, nbad = vres
        correct = (mean_rel < 1e-4 and nbad == 0)
        gf = measure(cubin, p, M, N, K) if correct else None
        pct = (100.0 * gf / cub_gf) if (gf and cub_gf) else None
        status = "ok" if correct else "WRONG"
        print(f"  {tag:34} {status:6} "
              f"{(f'{gf:.0f} GF' if gf else '-'):10} "
              f"{(f'{pct:.0f}%cuBLAS' if pct else ''):12} "
              f"rel={mean_rel:.1e} bad={nbad}")
        rows.append({**dict(zip(FIELDS, p)), "nthreads": nthreads(p),
                     "mean_rel": mean_rel, "nbad": nbad, "correct": correct,
                     "gflops": gf, "pct_cublas": pct})
        try: os.remove(cubin)
        except FileNotFoundError: pass

    good = [r for r in rows if r["correct"] and r["gflops"]]
    good.sort(key=lambda r: -r["gflops"])
    print(f"\n=== {len(good)}/{len(rows)} correct+measured. Top 5: ===")
    for r in good[:5]:
        print(f"  BM={r['BM']} BN={r['BN']} BK={r['BK']} TM={r['TM']} TN={r['TN']}: "
              f"{r['gflops']:.0f} GFLOPS ({r.get('pct_cublas') or 0:.0f}% cuBLAS)")
    if good and cub_gf:
        best = good[0]
        print(f"\n  BEST: {best['gflops']:.0f} GFLOPS  |  cuBLAS: {cub_gf:.0f}  |  "
              f"{100*best['gflops']/cub_gf:.0f}% of cuBLAS")
    with open(f"{args.out}.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nresults -> {args.out}.json")


if __name__ == "__main__":
    main()
