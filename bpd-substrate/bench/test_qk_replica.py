#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""test_qk_replica.py — TDD: test candidate QK-score algorithms against ggml node_20 ground truth.

This is the FAST inner-loop test: it does NOT rebuild the kernel. It takes ggml's exact
post-rope Q (0011) and K (0018) and the ground-truth scores (node_20), and tests several
candidate dot algorithms in numpy to find which one reproduces node_20 at 0 ULP. Once a
candidate passes here, implement it in C with confidence it CAN match.

Finding so far: naive per-pair f16 dot caps at ~1 ULP (ggml's own vec_dot_f16 does too),
proving node_20 is computed by ggml's tiled MUL_MAT, not a per-pair dot. This test will
identify the algorithm that DOES reproduce it.

Run: python3 bench/test_qk_replica.py
"""
import struct, glob, sys
import numpy as np

D = "/tmp/iyun_ggmldump"
def lg(p):
    raw = open(p, "rb").read(); nb = struct.unpack("<Q", raw[72:80])[0]
    return np.frombuffer(raw[80:80+nb], dtype=np.float32)

Q = lg(f"{D}/0011_ROPE_Qcur-0.bin").reshape(6, 32, 64)      # qpos, head, dim
K = lg(f"{D}/0018_ROPE_Kcur-0.bin").reshape(6, 8, 64)       # qpos(kvpos), kvhead, dim
S = lg(f"{D}/0041_MUL_MAT_node_20.bin").reshape(32, 6, 32)  # head, qpos, kv

def f16(x): return x.astype(np.float16).astype(np.float32)

def cmp(name, fn):
    """fn(q_vec, k_vec) -> score float. Test head0 over all (qpos,kv) valid pairs vs node_20."""
    h, kvh = 0, 0
    mu = 0; nd = 0; tot = 0
    for qp in range(6):
        for kv in range(qp + 1):
            g = np.float32(S[h, qp, kv])
            v = np.float32(fn(Q[qp, h], K[kv, kvh]))
            u = abs(int(v.view(np.int32)) - int(g.view(np.int32)))
            mu = max(mu, u); nd += (u != 0); tot += 1
    print(f"  {name:<32} maxULP={mu:<8} ndiff={nd}/{tot}  {'PASS' if mu==0 else 'fail'}")
    return mu == 0

# candidate algorithms
def naive_f32(q, k): return np.dot(q.astype(np.float64), k.astype(np.float64))
def f16_both_f64acc(q, k): return np.dot(f16(q).astype(np.float64), f16(k).astype(np.float64))
def f16_both_f32acc(q, k):
    s = np.float32(0)
    for i in range(64): s = np.float32(s + np.float32(f16(q)[i] * f16(k)[i]))
    return s
def f16_q_only(q, k): return np.dot(f16(q).astype(np.float64), k.astype(np.float64))

print("=== TDD: which QK algorithm reproduces ggml node_20 at 0 ULP? ===")
results = {
    "naive_f32": cmp("naive_f32 (current kernel)", naive_f32),
    "f16_both_f64acc": cmp("f16 both, f64 accumulate", f16_both_f64acc),
    "f16_both_f32acc": cmp("f16 both, f32 sequential", f16_both_f32acc),
    "f16_q_only": cmp("f16 Q only, f32 K", f16_q_only),
}
passed = [k for k, v in results.items() if v]
print(f"\nAlgorithms reaching 0-ULP: {passed if passed else 'NONE — node_20 needs ggml tiled mul_mat'}")
# exit 0 if at least one candidate is exact (we have a viable C target)
sys.exit(0 if passed else 1)
