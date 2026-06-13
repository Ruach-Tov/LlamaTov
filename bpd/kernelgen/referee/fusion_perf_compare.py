#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
import os as _os, sys as _sys
def _bpd_root(_p=_os.path.dirname(_os.path.abspath(__file__))):
    while _p != '/' and _os.path.basename(_p) != 'bpd':
        _p = _os.path.dirname(_p)
    return _p if _os.path.basename(_p) == 'bpd' else _os.path.dirname(_os.path.abspath(__file__))
_BPD = _bpd_root()

"""fusion_perf_compare — measure FUSED vs NON-FUSED head+epilogue kernels,
gated on correctness (Heath, 2026-06-08: "convenient for collecting comparative
performance on fused vs non-fused kernels, once they are measured as bit-identical
or at least correct within contract").

For an L2 head+elementwise chain (e.g. Conv2D->ReLU):
  1. ask head_fusion.pl for the fuse_spec (the recognizer discovers the fusion)
  2. build the NON-FUSED path: conv kernel + a separate elementwise kernel
  3. build the FUSED path: conv kernel with -DCONV_EPILOGUE=<lowered tail>
  4. VERIFY both vs the torch oracle (bit-exact preferred; within-contract ok)
  5. only if BOTH correct, TIME both and report the fused speedup

The correctness gate is non-negotiable: a faster-but-wrong fused kernel is
rejected, never reported as a win. Output is a comparison row:
  chain | fused-correct | nonfused-correct | t_fused | t_nonfused | speedup

By: Iyun, 2026-06-08
"""
import os
import re
import subprocess
import sys

import numpy as np
import sys as _sys; _sys.path.insert(0, _os.path.join(_BPD, "lib"))
import toolchain as tc

REPO = "<repo>/Ruach-Tov"
WORK = "/tmp/gpu-work/sweep"
CUDA = tc.cuda_root()  # shared toolchain (ENV-SHIFT defense)
SWIPL = "/run/current-system/sw/bin/swipl"
HF = f"{REPO}/bpd/kernelgen/emitters/head_fusion.pl"
CONV = f"{REPO}/bpd/kernelgen/runtime/conv_implicit.cu"
ENV = tc.nvcc_env()

# conv-50 shape
N, Cin, H, W, Cout, KH, KW = 32, 64, 56, 56, 128, 3, 3
Ho, Wo = H - KH + 1, W - KW + 1


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, env=ENV, timeout=kw.get("t", 120),
                          cwd=kw.get("cwd", WORK))


def get_dflag(chain):
    """Ask the recognizer for the fused -DCONV_EPILOGUE flag (or None if no fusion)."""
    chain_pl = "[" + ",".join(chain) + "]"
    goal = (f"use_module('{HF}'), "
            f"(fuse_spec({chain_pl}, spec(_,_,_,fused(dflag(D),_),_)) "
            f"-> (write(D), nl) ; (write('NONE'), nl)), halt")
    r = _run([SWIPL, "-q", "-g", goal], cwd=REPO)
    for ln in r.stdout.splitlines():
        ln = ln.strip()
        if ln.startswith("-DCONV_EPILOGUE"):
            return ln
        if ln == "NONE":
            return None
    return None


def build_conv(cubin_name, extra_dflags=None):
    """Compile conv_implicit -> cubin, optionally with the epilogue -D flag."""
    out = f"{WORK}/{cubin_name}"
    # canonical compile (shared toolchain): source=CONV, out=cubin, extra=the epilogue -D flags.
    cmd = tc.nvcc_compile_cmd(CONV, out, extra=(extra_dflags or ()))
    r = _run(cmd)
    return out if (r.returncode == 0 and os.path.exists(out)) else (_ for _ in ()).throw(
        RuntimeError(f"build {cubin_name} failed: {r.stderr[:400]}"))


def conv_ref(x, w):
    """torch conv2d oracle."""
    import torch
    import torch.nn.functional as F
    return F.conv2d(torch.from_numpy(x), torch.from_numpy(w)).numpy()


def relu_np(a):
    return np.where(np.isnan(a), a, np.maximum(a, 0.0))


def run_conv(cubin, x, w, bm=128, bn=128, nth=512):
    """Launch k_conv_implicit, return the output array."""
    x.tofile(f"{WORK}/fx.bin")
    w.tofile(f"{WORK}/fw.bin")
    ws = Cout * Cin * KH * KW
    # reuse perf_kernel conv2d_implicit (writes Csw via a verify mode) — but here we
    # need the OUTPUT; use a tiny dedicated launcher embedded for clarity.
    # (perf_kernel gives timing; for correctness we read the output buffer.)
    raise NotImplementedError  # see _launch below


def main():
    chain = sys.argv[1].split(",") if len(sys.argv) > 1 else ["bpd_conv2d", "bpd_relu"]
    print(f"=== FUSION PERF COMPARE: {' -> '.join(chain)} ===\n")

    dflag = get_dflag(chain)
    if dflag is None:
        print(f"  no epilogue fusion discovered for {chain} — nothing to compare")
        return
    print(f"  recognizer discovered fusion. epilogue flag:\n    {dflag}\n")

    # build both
    nonfused = build_conv("conv_nonfused.cubin")
    fused = build_conv("conv_fused.cubin", extra_dflags=[dflag])
    print(f"  built: non-fused={os.path.basename(nonfused)}  fused={os.path.basename(fused)}\n")

    # correctness + timing handled by the companion C harness (fusion_run), which
    # launches both, applies the separate relu for non-fused, and compares to a
    # numpy/torch oracle. This script orchestrates; fusion_run.cu does the GPU work.
    print("  (build OK — correctness+timing via fusion_run harness)")


if __name__ == "__main__":
    main()
