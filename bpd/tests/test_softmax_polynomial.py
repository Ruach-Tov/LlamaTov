#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""test_softmax_polynomial.py — Verify ggml-matching polynomial exp at 0 ULP.

This is the regression test for the Cephes polynomial exp we extracted
from ggml's disassembly. If this test FAILS, our softmax no longer
matches ggml's and we have a regression.

Author: medayek
"""
import sys, ctypes, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "bench"))
from llama_fixture_loader import load_manifest

c_float_p = ctypes.POINTER(ctypes.c_float)

def main():
    so_path = sys.argv[1] if len(sys.argv) > 1 else "build/bpd_cpu.so"
    fixture_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/llama_dump_hello_8_v2"

    t = load_manifest(fixture_dir)
    
    # Find the first SOFT_MAX op
    sm_ops = [x for x in t if x.op_desc == "SOFT_MAX"]
    if not sm_ops:
        print("SKIP: no SOFT_MAX in fixture")
        return
    
    sm = sm_ops[0]
    sm_ref = sm.as_numpy()
    src_idx = sm.src_indices[0] if hasattr(sm, 'src_indices') and sm.src_indices else None
    if src_idx is None:
        print("SKIP: no source link for SOFT_MAX")
        return
    
    src_data = t[src_idx].as_numpy()
    ne = list(sm.ne)
    n_kv, n_q, n_heads = ne[0], ne[1], ne[2]
    scale = 1.0 / np.sqrt(64.0)
    
    lib = ctypes.CDLL(so_path)
    
    # Try ggml polynomial first, fall back to standard
    for fn_name in ['bpd_softmax_causal_ggml_cpu', 'bpd_softmax_causal_cpu']:
        if hasattr(lib, fn_name):
            fn = getattr(lib, fn_name)
            fn.restype = None
            fn.argtypes = [c_float_p, c_float_p,
                          ctypes.c_int, ctypes.c_int, ctypes.c_int,
                          ctypes.c_float, ctypes.c_int]
            break
    else:
        print("SKIP: no softmax kernel in .so")
        return
    
    inp = np.ascontiguousarray(src_data, dtype=np.float32)
    out = np.zeros_like(inp)
    
    fn(inp.ctypes.data_as(c_float_p),
       out.ctypes.data_as(c_float_p),
       ctypes.c_int(n_heads), ctypes.c_int(n_q), ctypes.c_int(n_kv),
       ctypes.c_float(scale), ctypes.c_int(0))
    
    ref_f = sm_ref.flatten().astype(np.float32)
    our_f = out.flatten().astype(np.float32)
    rb = ref_f.view(np.uint32)
    ob = our_f.view(np.uint32)
    mm = int(np.sum(rb != ob))
    
    if mm == 0:
        print(f"PASS  softmax_polynomial: BIT_IDENTICAL ({len(ref_f)} elements, 0 ULP)")
        print(f"  kernel: {fn_name}")
    else:
        max_ulp = int(np.max(np.abs(rb.astype(np.int64) - ob.astype(np.int64))))
        print(f"FAIL  softmax_polynomial: {mm}/{len(ref_f)} differ, max_ULP={max_ulp}")
        print(f"  kernel: {fn_name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
