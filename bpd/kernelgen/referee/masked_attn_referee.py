#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""masked_attn_referee.py — standing gate for the masked fixed-MAXT attention kernel.

The CUDA-graph keystone (4bc8f3e96): k_attn_decode_masked iterates a compile-fixed
MAXT, reads current length from a DEVICE int (len_ptr), masks positions >= L.
Before this kernel may be captured into a graph, it must survive:

  P. POISON TEST (HARD, the new axis): fill cache positions >= L with NaN, +inf,
     and garbage; output must be BIT-IDENTICAL to the clean-beyond-L run.
     "Masked == unmasked on valid data" only proves the numbers happened to agree;
     poison proves the -inf mask STRUCTURALLY SEVERS the masked lanes. Any
     sensitivity = mask leak, localized to the poisoned region that leaked.
     (Design: Bocher msg 1e1e0139; adopted by Iyun into pre-gate smoke 92cd1abd.)

  S. LENGTH SWEEP (HARD): correctness at the off-by-one corners where masked
     softmax dies: L in {1, 2, MAXT-1, MAXT}, plus a mid value.

  C. COUNTER CHECK (HARD): the device len_ptr value must equal the host-side
     cache.length exactly (integer), at every step of an incremental fill.

  A2. CROSS-IMPL (declared-soft): vs host torch attention on the [0:L] slice —
     reduction order differs (the tick-class home); expected ~1e-7, reported.

Run on enclave (P4):
    python3 masked_attn_referee.py [--maxt 64] [--seed 0] [--json out.json]

Idiom: sibling of decode_referee.py. Author: Bocher, 2026-06-10.
"""
import os, sys, json, argparse, ctypes
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "bpd", "lib"))
sys.path.insert(0, os.path.join(REPO, "bpd"))

import dev_residency as DR
import fact_dispatch as fd

def host_attention_ref(q, K, V, nh, nkv, hd, scale):
    """Reference: plain numpy attention over the valid slice. q:[nh*hd], K/V:[L,nkv,hd]."""
    L = K.shape[0]
    rep = nh // nkv
    out = np.empty(nh * hd, np.float32)
    for h in range(nh):
        hk = h // rep
        qh = q[h*hd:(h+1)*hd].astype(np.float64)
        scores = np.array([np.dot(qh, K[t, hk].astype(np.float64)) * scale for t in range(L)])
        m = scores.max()
        w = np.exp(scores - m); w /= w.sum()
        acc = np.zeros(hd, np.float64)
        for t in range(L):
            acc += w[t] * V[t, hk].astype(np.float64)
        out[h*hd:(h+1)*hd] = acc.astype(np.float32)
    return out

def fill_cache(cache, K, V):
    """Write [L,nkv,hd] K/V into the device cache from host, set length=L."""
    L = K.shape[0]
    kd = DR.DevTensor.from_host(np.ascontiguousarray(K.reshape(-1), dtype=np.float32))
    vd = DR.DevTensor.from_host(np.ascontiguousarray(V.reshape(-1), dtype=np.float32))
    cache.length = 0
    cache.append(kd.ptr, vd.ptr, count=L)
    return kd, vd  # keep alive

def poison_beyond(cache, L, maxt, mode):
    """Overwrite positions [L:maxt] of the device K and V buffers with poison."""
    n_pos = maxt - L
    if n_pos <= 0:
        return None
    width = cache.width
    if mode == "nan":
        poison = np.full(n_pos * width, np.nan, np.float32)
    elif mode == "inf":
        poison = np.full(n_pos * width, np.inf, np.float32)
    else:  # garbage: huge alternating values
        poison = (np.arange(n_pos * width, dtype=np.float32) % 7 - 3) * 1e30
    pd = DR.DevTensor.from_host(poison)
    cu = fd._libcuda()
    off = cache._row_off(L)
    nbytes = n_pos * width * cache.elem_bytes
    cu.cuMemcpyDtoD_v2(ctypes.c_void_p(cache.k_ptr.value + off), pd.ptr, nbytes)
    cu.cuMemcpyDtoD_v2(ctypes.c_void_p(cache.v_ptr.value + off), pd.ptr, nbytes)
    return pd

def run_masked(q_host, cache, nh, scale, maxt):
    qd = DR.DevTensor.from_host(q_host.astype(np.float32))
    len_ptr = cache._ensure_len_ptr()
    out = DR.attn_decode_from_cache_masked(qd.ptr, cache, nh, scale, len_ptr, maxt)
    fd._libcuda().cuCtxSynchronize()
    return out.to_host().copy()

def read_device_len(cache):
    cu = fd._libcuda(); cu.cuCtxSynchronize()
    v = ctypes.c_int(-1)
    cu.cuMemcpyDtoH_v2(ctypes.byref(v), cache.len_ptr, 4)
    return v.value

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxt", type=int, default=64)
    ap.add_argument("--nh", type=int, default=14)
    ap.add_argument("--nkv", type=int, default=2)
    ap.add_argument("--hd", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--a2-tol", type=float, default=5e-6,
                    help="declared-soft bound for cross-impl (reduction order)")
    ap.add_argument("--json", help="write per-check records")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    nh, nkv, hd, maxt = a.nh, a.nkv, a.hd, a.maxt
    scale = 1.0 / np.sqrt(hd)

    print(f"=== MASKED-ATTENTION REFEREE (keystone gate, MAXT={maxt}, P4) ===", flush=True)
    print(f"{'check':<34}{'L':<7}{'result':<14}{'detail':<28}{'verdict'}", flush=True)
    records, hard_fail = [], []

    def report(check, L, ok, result, detail):
        verdict = "PASS" if ok else "FAIL"
        if not ok:
            hard_fail.append((check, L))
        print(f"{check:<34}{L:<7}{result:<14}{detail:<28}{verdict}", flush=True)
        records.append({"check": check, "L": L, "ok": bool(ok),
                        "result": result, "detail": detail})

    lengths = sorted(set([1, 2, maxt // 2, maxt - 1, maxt]))
    for L in lengths:
        K = rng.standard_normal((L, nkv, hd)).astype(np.float32) * 0.4
        V = rng.standard_normal((L, nkv, hd)).astype(np.float32) * 0.4
        q = rng.standard_normal(nh * hd).astype(np.float32)

        # CLEAN baseline: cache with zeros beyond L (fresh alloc is not guaranteed
        # zero — explicitly zero the tail so "clean" is defined, not assumed)
        cache = DR.DeviceKVCache(maxt, nkv, hd)
        keep = fill_cache(cache, K, V)
        if L < maxt:
            z = np.zeros((maxt - L) * cache.width, np.float32)
            zd = DR.DevTensor.from_host(z); cu = fd._libcuda()
            off = cache._row_off(L); nb = (maxt - L) * cache.width * cache.elem_bytes
            cu.cuMemcpyDtoD_v2(ctypes.c_void_p(cache.k_ptr.value + off), zd.ptr, nb)
            cu.cuMemcpyDtoD_v2(ctypes.c_void_p(cache.v_ptr.value + off), zd.ptr, nb)
        out_clean = run_masked(q, cache, nh, scale, maxt)

        # C. counter integrity
        dl = read_device_len(cache)
        report("C:len_ptr==host.length", L, dl == L, f"dev={dl} host={L}", "integer equality")

        # sanity: output finite
        finite = np.isfinite(out_clean).all()
        report("S:clean-output-finite", L, finite,
               "all finite" if finite else "NaN/Inf!", f"max|y|={np.abs(out_clean).max():.3g}")

        # P. poison tests — three poisons, each must be BIT-identical to clean
        for mode in ("nan", "inf", "garbage"):
            keep_p = poison_beyond(cache, L, maxt, mode)
            cache.length = L; cache._ensure_len_ptr()   # length unchanged; refresh dev int
            out_p = run_masked(q, cache, nh, scale, maxt)
            bit_same = np.array_equal(out_clean.view(np.uint32), out_p.view(np.uint32))
            if bit_same:
                detail = "bit-identical"
            else:
                bad = np.nonzero(out_clean.view(np.uint32) != out_p.view(np.uint32))[0]
                detail = f"LEAK {len(bad)} elems, first@{bad[0]}(h{bad[0]//hd})"
            report(f"P:poison-{mode}-insensitive", L, bit_same, detail,
                   f"poisoned [{L}:{maxt}]")

        # A2. cross-impl vs host reference on valid slice (declared-soft, reported)
        ref = host_attention_ref(q, K, V, nh, nkv, hd, scale)
        d = float(np.abs(out_clean - ref).max())
        report("A2:vs-host-ref(soft)", L, d < a.a2_tol, f"max_abs={d:.2e}",
               f"declared bound {a.a2_tol:.0e}")

    # C2. incremental counter tracking: fill one position at a time
    cache = DR.DeviceKVCache(maxt, nkv, hd)
    okc, detail = True, ""
    for step in range(1, min(8, maxt) + 1):
        k1 = rng.standard_normal((1, nkv, hd)).astype(np.float32)
        v1 = rng.standard_normal((1, nkv, hd)).astype(np.float32)
        kd = DR.DevTensor.from_host(k1.reshape(-1)); vd = DR.DevTensor.from_host(v1.reshape(-1))
        cache.append(kd.ptr, vd.ptr, 1)
        cache._ensure_len_ptr()
        dl = read_device_len(cache)
        if dl != step:
            okc, detail = False, f"step {step}: dev={dl}"
            break
    report("C2:incremental-counter", min(8, maxt), okc,
           detail or f"8 steps exact", "append-by-1 tracking")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(records, f, indent=1)
    ok = not hard_fail
    print("\n" + ("MASKED ATTENTION VERIFIED ✓ — graph-capture eligible" if ok
                  else f"HARD FAIL: {hard_fail} ✗ — NOT graph-eligible"))
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
