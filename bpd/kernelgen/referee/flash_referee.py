#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""flash_referee.py — the standing differential gate for FlashAttention (fixes
infra-debt #2 + #4). Extends the referee family to cover the attention kernels that
the multibackend referee doesn't.

For each preserved flash kernel (runtime/flash_reference/) AND the schedule-derived
kernel, build it, run it on the P4 over a fixed deterministic input, and verify
against the attention oracle (softmax(scale·QK^T)·V) — using the SHARED kernel_harness
(no copy-paste). Reports per-kernel: ptxas spill, max_abs, within_tol, timing.

This makes the flash_reference kernels LIVE (not dead .cu artifacts) and brings the
session's attention work under a standing referee.

Run on enclave:  python3 flash_referee.py
Author: Iyun, 2026-06-08
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kernel_harness import (Env, build_cubin, run_kernel, verify, time_kernel,
                            attention_reference, deterministic_qkv)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
FLASH_REF = os.path.join(REPO, "bpd", "kernelgen", "runtime", "flash_reference")

# (file, kernel-name, launch-config-fn(S,D) -> (grid, block, shmem, args_after_O))
def _wpb_of(src):
    """Read the kernel's compiled-in WPB (#define WPB N) — the referee must launch
    with the geometry the kernel was built for (else a launch mismatch = wrong output,
    exactly the bug class our launch-contract prevention targets)."""
    import re
    m = re.search(r"#define\s+WPB\s+(\d+)", open(src).read())
    return int(m.group(1)) if m else 8

def cfg_shared(S, D, WPB, BC=32):   # warp+shared kernels: need shmem
    return ((S + WPB - 1)//WPB,), (32*WPB,), 2*BC*D*4

def cfg_nosh(S, D, WPB):            # warp-coop (no shared)
    return ((S + WPB - 1)//WPB,), (32*WPB,), 0

KERNELS = [
    ("flashv.cu",  "k_flash_vec",     cfg_shared, "vectorized warp+shared (10.25x)"),
    ("flashws.cu", "k_flash_warp_sh", cfg_shared, "warp+shared scalar (2.69x)"),
    ("flashw.cu",  "k_flash_warp",    cfg_nosh,   "warp-cooperative (1.65x)"),
]

def main():
    env = Env(work="/tmp/gpu-work/flash_referee")
    S, D = 512, 128
    Q, K, V = deterministic_qkv(S, D)
    scale = 1.0 / np.sqrt(D)
    ref = attention_reference(Q, K, V, scale)
    print(f"=== FLASH REFEREE (S={S} D={D}, P4) — all vs attention oracle ===")
    print(f"{'kernel':<16}{'desc':<34}{'max_abs':<11}{'within_tol':<11}{'ms':<9}{'verdict'}")
    all_ok = True
    for fname, kname, cfgfn, desc in KERNELS:
        src = os.path.join(FLASH_REF, fname)
        if not os.path.exists(src):
            print(f"{fname:<16}MISSING"); all_ok = False; continue
        local = os.path.join(env.work, fname)
        open(local, "w").write(open(src).read())
        cubin = build_cubin(env, local, ptxas_v=True)
        if not cubin:
            print(f"{fname:<16}BUILD-FAIL"); all_ok = False; continue
        wpb = _wpb_of(src)   # launch with the kernel's compiled-in geometry
        grid, block, shmem = cfgfn(S, D, wpb)
        args = [Q, K, V, "OUT", S, D, scale]
        try:
            out = run_kernel(env, cubin, kname, grid, block, args, out_idx=3,
                             out_shape=(S, D), shmem=shmem)
        except Exception as e:
            print(f"{fname:<16}RUN-FAIL {e}"); all_ok = False; continue
        rep = verify(out, ref)
        ms = time_kernel(env, cubin, kname, grid, block, args, out_idx=3,
                         out_shape=(S, D), shmem=shmem, iters=50)
        verdict = "PASS" if rep["within_tol"] else "FAIL"
        if not rep["within_tol"]:
            all_ok = False
        print(f"{fname:<16}{desc:<34}{rep['max_abs']:<11.6f}{str(rep['within_tol']):<11}{ms:<9.4f}{verdict}")
    print("\n" + ("ALL FLASH KERNELS VERIFIED ✓" if all_ok else "SOME FAILED ✗"))
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
