#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
import os as _os, sys as _sys
def _bpd_root(_p=_os.path.dirname(_os.path.abspath(__file__))):
    while _p != '/' and _os.path.basename(_p) != 'bpd':
        _p = _os.path.dirname(_p)
    return _p if _os.path.basename(_p) == 'bpd' else _os.path.dirname(_os.path.abspath(__file__))
_BPD = _bpd_root()

"""
gemm_sweep.py — sweep the GPU GEMM parameter space for best performance.

The GPU analog of bench/verify_gemm_sweep.py (which swept the CPU/AVX
gemm_pattern space). Enumerates the valid points of gemm_gpu_space.pl
(TILE,TM,TN,BLK,VEC,UNROLL), and for each:
  GENERATE  — project the point to a tiled CUDA kernel (gemm_tiled_from_space.pl)
  VERIFY    — run on the P4 over a fixed A,B; relative-error gate vs torch.matmul
              (reductions: rel-error is the correctness bar, not bit-identity)
  MEASURE   — perf_fixture GFLOPS vs the 5.70 TFLOPS P4 roofline
  RECORD    — (point -> correct?, GFLOPS, %peak)

Reports the Pareto-best correct points and the single fastest. This is the
sweepable-parameter performance search: the coordinate space is the design
space, the perf fixture is the objective, the rel-error gate is the constraint.

Compares the winner against cuBLAS (PyTorch's GPU backend) at the same size.

Usage: gemm_sweep.py [--n 1024] [--verify-n 512] [--limit N] [--out gemm_sweep]
Run on the GPU host (enclave).
Author: Iyun, 2026-06-07
"""
import os, sys, re, json, subprocess, argparse
import numpy as np
import sys as _sys; _sys.path.insert(0, _os.path.join(_BPD, "lib"))
import toolchain as tc

REPO   = os.environ.get("RUACHTOV_REPO", "")
EMIT   = f"{REPO}/bpd/kernelgen/emitters"
WORK   = os.environ.get("SWEEP_WORK", "")
CUDA   = tc.cuda_root()  # shared toolchain (ENV-SHIFT defense)
REDIST = "/nix/store/560i0agldlr2h4h3bx6mq2lifw6w1iaa-cuda-native-redist-12.8/lib"
STUBS  = "/nix/store/3n5kqxw44phkj9bcwdzdpj1z31q4ajg9-cuda_cudart-12.8.90-stubs/lib/stubs"
SWIPL  = "/run/current-system/sw/bin/swipl"
PERF   = f"{WORK}/perf_fixture"
ENV    = tc.nvcc_env()
os.makedirs(WORK, exist_ok=True)

def nthreads(p):
    BM, BN, BK, TM, TN = p
    return (BM // TM) * (BN // TN)

def enumerate_points():
    """Get the valid (BM,BN,BK,TM,TN) register-blocked points from the space."""
    g = ('consult("%s/gemm_gpu_space.pl"), forall(gpu_gemm_point(BM,BN,BK,TM,TN), '
         'format("~w ~w ~w ~w ~w~n",[BM,BN,BK,TM,TN])), halt' % EMIT)
    out = subprocess.run([SWIPL, "-q", "-g", g], capture_output=True, text=True, timeout=60).stdout
    pts = []
    for line in out.strip().splitlines():
        q = line.split()
        if len(q) == 5:
            pts.append(tuple(int(x) for x in q))
    return pts

def gen_compile(point, vec=1):
    """Project a register-blocked point -> kernel (vec=1 scalar or 4 float4 loads)
    -> cubin. Returns path or None."""
    BM, BN, BK, TM, TN = point
    cu = f"{WORK}/sw_{BM}_{BN}_{BK}_{TM}_{TN}_v{vec}.cu"
    cubin = cu.replace(".cu", ".cubin")
    g = (f'consult("{EMIT}/gemm_tiled_from_space.pl"), '
         f'emit_gemm_tiled({BM},{BN},{BK},{TM},{TN},true,{vec},"{cu}"), halt')
    subprocess.run([SWIPL, "-q", "-g", g], capture_output=True, text=True, timeout=60)
    if not os.path.exists(cu):
        return None
    subprocess.run(tc.nvcc_compile_cmd(cu, cubin),
                   capture_output=True, text=True, timeout=120, env=ENV)
    return cubin if os.path.exists(cubin) else None

def robust_rel(got, ref):
    """Accumulation-order-aware relative error for REDUCTIONS (GEMM/conv).

    A faster GEMM (tiled / split-K / register-blocked) LEGITIMATELY changes the
    reduction tree, so it diverges from a naive reference by a few accumulation-
    order ULP — this is 'deterministic but non-invariant' (arxiv 2606.00279),
    NOT a bug. So the correctness bar is rel-error, not bit-identity.

    The old metric |got-ref|/(|ref|+1e-30) had two failure modes for GEMM:
      (1) near-zero ref values EXPLODE the mean -> false FAIL on a correct kernel
          (the same bug fixed in l2_chain_verify today: a correct result read 1e14).
      (2) the MEAN can HIDE one catastrophically-wrong element among many exact
          ones -> false PASS on a real bug.
    Robust form: denom = |ref|+|got|+atol (allclose-style) floors near-zero AND
    bounds each element's error to ~[0,1], so a wrong element contributes ~1 and
    is not averaged away. We report BOTH mean (accumulation-order signal) and max
    (catches the single-wrong-element bug)."""
    got = np.asarray(got, np.float64); ref = np.asarray(ref, np.float64)
    denom = np.abs(ref) + np.abs(got) + 1e-4
    per = np.abs(got - ref) / denom
    return float(np.nanmean(per)), float(np.nanmax(per))


def verify(cubin, point, A, B, ref, n):
    """Launch via tcheck_v2 (1-D block=NThreads, 2-D grid); return mean rel err.
    Uses the accumulation-order-aware robust metric so a correct-but-reordered
    GEMM (different reduction tree) is not false-flagged as a bug."""
    BM, BN, BK, TM, TN = point
    NTH = nthreads(point)
    try: os.remove(f"{WORK}/Csw.bin")
    except FileNotFoundError: pass
    subprocess.run([f"{WORK}/tcheck_v2", cubin, str(NTH), str(BM), str(BN), str(n), "verify"],
                   capture_output=True, text=True, timeout=60, env=ENV, cwd=WORK)
    if not os.path.exists(f"{WORK}/Csw.bin"):
        return None
    got = np.fromfile(f"{WORK}/Csw.bin", np.float32)
    mean_rel, max_rel = robust_rel(got, ref)
    # mean is the accumulation-order signal; max guards against a single wrong
    # element being averaged away. A reordered-but-correct GEMM has small BOTH;
    # a real bug spikes max even if mean stays low.
    return mean_rel, max_rel

def measure(cubin, point, n):
    """GFLOPS via tcheck_v2 perf mode (it knows the v2 1-D-block launch geometry)."""
    BM, BN, BK, TM, TN = point
    NTH = nthreads(point)
    out = subprocess.run([f"{WORK}/tcheck_v2", cubin, str(NTH), str(BM), str(BN), str(n), "perf", "30"],
                         capture_output=True, text=True, timeout=180, env=ENV, cwd=WORK).stdout
    m = re.search(r"GFLOPS:\s*([\d.]+)\s*\(([\d.]+)%", out)
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)

def cublas_ref(n):
    out = subprocess.run([f"{WORK}/cublas_perf", str(n), "30"],
                         capture_output=True, text=True, timeout=120, env=ENV).stdout
    m = re.search(r"([\d.]+)\s*GFLOPS", out)
    return float(m.group(1)) if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024, help="perf size")
    ap.add_argument("--verify-n", type=int, default=512, help="correctness size")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=f"{WORK}/gemm_sweep")
    args = ap.parse_args()

    # fixed verify inputs + torch reference
    import torch, warnings; warnings.filterwarnings("ignore")
    vn = args.verify_n
    rng = np.random.default_rng(0)
    A = rng.standard_normal((vn, vn)).astype(np.float32)
    B = rng.standard_normal((vn, vn)).astype(np.float32)
    A.tofile(f"{WORK}/A.bin"); B.tofile(f"{WORK}/B.bin")
    ref = torch.matmul(torch.from_numpy(A), torch.from_numpy(B)).numpy().ravel()

    pts = enumerate_points()
    if args.limit: pts = pts[:args.limit]
    print(f"=== GPU GEMM sweep: {len(pts)} points, verify N={vn}, perf N={args.n} ===\n")
    print(f"{'BM BN BK TM TN (thr)':24} {'correct':10} {'GFLOPS':10} {'%peak'}")
    print("-" * 64)
    FIELDS = "BM BN BK TM TN".split()

    rows = []
    for p in pts:
        BM, BN, BK, TM, TN = p
        # try scalar (vec=1) and float4 (vec=4, when BK%4==0 and BN%4==0) loads
        vecs = [1] + ([4] if (BK % 4 == 0 and BN % 4 == 0) else [])
        for vec in vecs:
            tag = f"{BM:3} {BN:3} {BK:2} {TM:2} {TN:2} v{vec} ({nthreads(p):4})"
            cubin = gen_compile(p, vec)
            if not cubin:
                print(f"{tag:26} {'compile-err':10}"); rows.append({**dict(zip(FIELDS, p)), "vec": vec, "status": "compile_err"}); continue
            res = verify(cubin, p, A, B, ref, vn)
            mean_rel, max_rel = (res if res is not None else (None, None))
            # correct = small SYSTEMATIC error (mean, accumulation-order ok) AND no
            # single catastrophically-wrong element (max guards against averaging
            # a real bug away). Thresholds: mean<1e-4, max<1e-2 (a reordered GEMM
            # has tiny both; a bug spikes max).
            correct = (mean_rel is not None and mean_rel < 1e-4 and max_rel < 1e-2)
            gflops, pct = (None, None)
            if correct:
                gflops, pct = measure(cubin, p, args.n)
            print(f"{tag:26} {('ok' if correct else 'WRONG'):10} {(f'{gflops:.0f}' if gflops else '-'):10} {(f'{pct:.1f}%' if pct else '')}")
            rows.append({**dict(zip(FIELDS, p)), "vec": vec, "nthreads": nthreads(p),
                         "mean_rel": mean_rel, "max_rel": max_rel, "correct": correct,
                         "gflops": gflops, "pct_peak": pct})
            os.remove(cubin)  # keep WORK tidy

    # winner + cuBLAS comparison
    good = [r for r in rows if r.get("gflops")]
    good.sort(key=lambda r: r["gflops"], reverse=True)
    cub = cublas_ref(args.n)
    with open(f"{args.out}.json", "w") as fh: json.dump(rows, fh, indent=2, default=str)
    print(f"\n=== {len(good)}/{len(pts)} correct+measured. Top 5: ===")
    for r in good[:5]:
        print(f"  BM={r['BM']} BN={r['BN']} BK={r['BK']} TM={r['TM']} TN={r['TN']} vec={r.get('vec',1)} (thr={r['nthreads']}): "
              f"{r['gflops']:.0f} GFLOPS ({r['pct_peak']:.1f}% peak)")
    if good and cub:
        best = good[0]
        print(f"\n  BEST generated: {best['gflops']:.0f} GFLOPS  |  cuBLAS: {cub:.0f} GFLOPS  "
              f"|  ratio: {cub/best['gflops']:.1f}x  (we're at {best['gflops']/cub*100:.0f}% of cuBLAS)")

if __name__ == "__main__":
    main()
