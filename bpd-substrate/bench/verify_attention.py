#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""verify_attention.py — standalone 0-ULP conformance test for bpd_gqa_attn_cpu vs ggml.

Fills the attention coverage hole: attention was previously exercised ONLY end-to-end and
checked ONLY at token-SHA granularity, so the V-sum scalar-vs-SIMD-tree divergence (which
ggml computes via ggml_vec_dot_f32's 4-accumulator tree-reduce) went undetected.

This builds the ggml reference attention (scores -> ggml softmax -> ggml_vec_dot_f32 V-sum)
on IDENTICAL random inputs and compares OUR bpd_gqa_attn_cpu output bit-for-bit (0 ULP).

Usage: BPD_CPU_SO=/path/bpd_cpu.so python3 bench/verify_attention.py
Exit 0 = BIT_IDENTICAL, 1 = DIVERGENT.
"""
import os, subprocess, sys, tempfile

REF = os.environ.get("GGML_REF", "")
SO = os.environ.get("BPD_CPU_SO", "build/bpd_cpu.so")
C_SRC = os.path.join(os.path.dirname(__file__), "verify_attention.c")

def main():
    if not os.path.exists(C_SRC):
        print(f"missing {C_SRC}", file=sys.stderr); return 2
    binp = tempfile.mktemp(prefix="verify_attn_")
    cc = ["clang", "-mavx", "-mf16c", "-mssse3", "-mno-avx2", "-mno-fma", "-O2",
          C_SRC, SO, f"-I{REF}/ggml/include", f"-L{REF}/build/bin",
          "-lggml-cpu", "-lggml-base", "-lggml", "-lm", "-o", binp]
    r = subprocess.run(cc, capture_output=True, text=True)
    if r.returncode != 0:
        print("COMPILE FAILED:\n" + r.stderr[:2000], file=sys.stderr); return 2
    env = dict(os.environ, LD_LIBRARY_PATH=f"{os.path.dirname(SO)}:{REF}/build/bin")
    run = subprocess.run([binp], capture_output=True, text=True, env=env)
    print(run.stdout.strip())
    return run.returncode

if __name__ == "__main__":
    sys.exit(main())
