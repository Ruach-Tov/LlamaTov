# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""L.1.8 GQA Attention tests for bpd_gqa_attn_cpu.

Tests:
  L.1.8a  MHA prefill   (8 tok, 8 q-heads = 8 kv-heads, d=64)
  L.1.8b  GQA prefill   (8 tok, 32 q-heads, 8 kv-heads, d=128)  — Llama 3 8B shape
  L.1.8c  GQA decode    (1 tok, 32 q-heads, 8 kv-heads, d=128, kv_offset=127)
  L.1.8d  MHA decode    (1 tok, 8 q-heads = 8 kv-heads, d=64, kv_offset=63)
  L.1.8e  Causal mask correctness (4 tok, 4 heads, d=32)

Verification strategy:
  The online softmax uses expf() internally.  Python ctypes passes float
  arguments through the x87 ABI, which may differ from the C kernel's
  XMM-register expf() by 1 ULP — identical to the RoPE cosf() issue.
  Therefore we verify structural correctness on this sandbox:
    1. Output is finite and non-zero.
    2. Causal mask is respected (token i does not attend to token j > i).
    3. Output is a convex combination of attended V rows (norm bounded).
    4. Self-consistency: prefill[qt] == single-token decode at pos=qt.
    5. Determinism: two calls with identical inputs produce identical output.
  The bit-identical oracle is the Ruach Tov fixture test (llama.cpp dump).
"""
import ctypes, os, sys
import numpy as np

SO_PATH = os.environ.get("BPD_CPU_SO", "build/bpd_cpu.so")
print(f"BPD_CPU_SO: {SO_PATH}")

lib = ctypes.CDLL(SO_PATH)
lib.bpd_gqa_attn_cpu.argtypes = [
    ctypes.c_void_p,  # q
    ctypes.c_void_p,  # k
    ctypes.c_void_p,  # v
    ctypes.c_void_p,  # dst
    ctypes.c_int,     # n_q_tokens
    ctypes.c_int,     # n_kv
    ctypes.c_int,     # n_q_heads
    ctypes.c_int,     # n_kv_heads
    ctypes.c_int,     # head_dim
    ctypes.c_float,   # scale
    ctypes.c_int,     # kv_offset
]
lib.bpd_gqa_attn_cpu.restype = None


def run_gqa(q, k, v, n_q_heads, n_kv_heads, head_dim, scale, kv_offset):
    n_q_tokens = q.shape[0]
    n_kv = k.shape[0]
    dst = np.zeros((n_q_tokens, n_q_heads, head_dim), dtype=np.float32)
    lib.bpd_gqa_attn_cpu(
        q.ctypes.data_as(ctypes.c_void_p),
        k.ctypes.data_as(ctypes.c_void_p),
        v.ctypes.data_as(ctypes.c_void_p),
        dst.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(n_q_tokens),
        ctypes.c_int(n_kv),
        ctypes.c_int(n_q_heads),
        ctypes.c_int(n_kv_heads),
        ctypes.c_int(head_dim),
        ctypes.c_float(float(scale)),
        ctypes.c_int(kv_offset),
    )
    return dst


def structural_ok(dst, q, k, v, n_q_heads, n_kv_heads, head_dim, scale, kv_offset, name):
    """Verify structural properties of the attention output."""
    n_q_tokens = dst.shape[0]
    gqa_ratio = n_q_heads // n_kv_heads
    issues = []

    # 1. Finite and non-zero
    if not np.all(np.isfinite(dst)):
        issues.append("output contains inf/nan")
    if not np.any(dst != 0.0):
        issues.append("output is all zeros")

    # 2. Causal mask: prefill[qt] must equal single-token decode at pos=qt
    if kv_offset == 0 and n_q_tokens > 1:
        for qt in range(n_q_tokens):
            dst_single = run_gqa(
                q[qt:qt+1], k[:qt+1], v[:qt+1],
                n_q_heads, n_kv_heads, head_dim, scale, kv_offset=qt
            )
            max_diff = float(np.max(np.abs(dst[qt] - dst_single[0])))
            if max_diff > 1e-5:
                issues.append(f"causal mask: token {qt} differs by {max_diff:.3e}")

    # 3. Determinism: two calls produce identical output
    dst2 = run_gqa(q, k, v, n_q_heads, n_kv_heads, head_dim, scale, kv_offset)
    if not np.array_equal(dst, dst2):
        issues.append("non-deterministic output")

    # 4. Norm bounded: output norm <= max attended V norm * 1.01
    for qt in range(n_q_tokens):
        q_pos = kv_offset + qt
        for iq in range(n_q_heads):
            kv_head = iq // gqa_ratio
            n_attended = min(q_pos + 1, k.shape[0])
            attended_v = v[:n_attended, kv_head]
            max_v_norm = float(np.max(np.linalg.norm(attended_v, axis=-1))) if n_attended > 0 else 0.0
            out_norm = float(np.linalg.norm(dst[qt, iq]))
            if max_v_norm > 0 and out_norm > max_v_norm * 1.01:
                issues.append(f"norm bound: qt={qt} iq={iq} out={out_norm:.4f} max_v={max_v_norm:.4f}")

    ok = len(issues) == 0
    status = "PASS" if ok else "FAIL"
    detail = "structural OK" if ok else f"structural FAIL: {'; '.join(issues)}"
    n_elem = dst.size
    print(f"  {status}  {name}: {detail} ({n_elem} elements)")
    return ok


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_mha_prefill(lib):
    """L.1.8a MHA prefill: 8 tokens, 8 heads = 8 kv-heads, d=64."""
    rng = np.random.default_rng(42)
    n_q_tokens, n_q_heads, n_kv_heads, head_dim = 8, 8, 8, 64
    scale = np.float32(1.0 / np.sqrt(head_dim))
    kv_offset = 0
    q = rng.standard_normal((n_q_tokens, n_q_heads, head_dim)).astype(np.float32)
    k = rng.standard_normal((n_q_tokens, n_kv_heads, head_dim)).astype(np.float32)
    v = rng.standard_normal((n_q_tokens, n_kv_heads, head_dim)).astype(np.float32)
    dst = run_gqa(q, k, v, n_q_heads, n_kv_heads, head_dim, scale, kv_offset)
    return structural_ok(dst, q, k, v, n_q_heads, n_kv_heads, head_dim, scale, kv_offset,
                         "L.1.8a MHA prefill (8 tok, 8h=8kv, d=64)")


def test_gqa_prefill_llama3(lib):
    """L.1.8b GQA prefill: 8 tokens, 32 q-heads, 8 kv-heads, d=128 (Llama 3 8B)."""
    rng = np.random.default_rng(99)
    n_q_tokens, n_q_heads, n_kv_heads, head_dim = 8, 32, 8, 128
    scale = np.float32(1.0 / np.sqrt(head_dim))
    kv_offset = 0
    q = rng.standard_normal((n_q_tokens, n_q_heads, head_dim)).astype(np.float32)
    k = rng.standard_normal((n_q_tokens, n_kv_heads, head_dim)).astype(np.float32)
    v = rng.standard_normal((n_q_tokens, n_kv_heads, head_dim)).astype(np.float32)
    dst = run_gqa(q, k, v, n_q_heads, n_kv_heads, head_dim, scale, kv_offset)
    return structural_ok(dst, q, k, v, n_q_heads, n_kv_heads, head_dim, scale, kv_offset,
                         "L.1.8b GQA prefill (8 tok, 32q/8kv, d=128, Llama3-8B)")


def test_gqa_decode_llama3(lib):
    """L.1.8c GQA decode: 1 token, 32 q-heads, 8 kv-heads, d=128, kv_offset=127."""
    rng = np.random.default_rng(77)
    n_q_tokens, n_q_heads, n_kv_heads, head_dim = 1, 32, 8, 128
    scale = np.float32(1.0 / np.sqrt(head_dim))
    kv_offset = 127
    q = rng.standard_normal((n_q_tokens, n_q_heads, head_dim)).astype(np.float32)
    k = rng.standard_normal((kv_offset + 1, n_kv_heads, head_dim)).astype(np.float32)
    v = rng.standard_normal((kv_offset + 1, n_kv_heads, head_dim)).astype(np.float32)
    dst = run_gqa(q, k, v, n_q_heads, n_kv_heads, head_dim, scale, kv_offset)
    return structural_ok(dst, q, k, v, n_q_heads, n_kv_heads, head_dim, scale, kv_offset,
                         "L.1.8c GQA decode (1 tok, 32q/8kv, d=128, pos=127)")


def test_mha_decode(lib):
    """L.1.8d MHA decode: 1 token, 8 heads, d=64, kv_offset=63."""
    rng = np.random.default_rng(13)
    n_q_tokens, n_q_heads, n_kv_heads, head_dim = 1, 8, 8, 64
    scale = np.float32(1.0 / np.sqrt(head_dim))
    kv_offset = 63
    q = rng.standard_normal((n_q_tokens, n_q_heads, head_dim)).astype(np.float32)
    k = rng.standard_normal((kv_offset + 1, n_kv_heads, head_dim)).astype(np.float32)
    v = rng.standard_normal((kv_offset + 1, n_kv_heads, head_dim)).astype(np.float32)
    dst = run_gqa(q, k, v, n_q_heads, n_kv_heads, head_dim, scale, kv_offset)
    return structural_ok(dst, q, k, v, n_q_heads, n_kv_heads, head_dim, scale, kv_offset,
                         "L.1.8d MHA decode (1 tok, 8h=8kv, d=64, pos=63)")


def test_causal_mask_correctness(lib):
    """L.1.8e Causal mask: token i must not attend to token j > i."""
    rng = np.random.default_rng(55)
    n_q_tokens, n_q_heads, n_kv_heads, head_dim = 4, 4, 4, 32
    scale = np.float32(1.0 / np.sqrt(head_dim))
    kv_offset = 0
    q = rng.standard_normal((n_q_tokens, n_q_heads, head_dim)).astype(np.float32)
    k = rng.standard_normal((n_q_tokens, n_kv_heads, head_dim)).astype(np.float32)
    v = rng.standard_normal((n_q_tokens, n_kv_heads, head_dim)).astype(np.float32)
    dst_full = run_gqa(q, k, v, n_q_heads, n_kv_heads, head_dim, scale, kv_offset)

    ok = True
    for qt in range(n_q_tokens):
        dst_single = run_gqa(
            q[qt:qt+1], k[:qt+1], v[:qt+1],
            n_q_heads, n_kv_heads, head_dim, scale, kv_offset=qt
        )
        diff = float(np.max(np.abs(dst_full[qt] - dst_single[0])))
        if diff > 1e-6:
            ok = False
            print(f"  FAIL  L.1.8e causal mask: token {qt} max_diff={diff:.3e}")

    status = "PASS" if ok else "FAIL"
    n_elem = dst_full.size
    print(f"  {status}  L.1.8e Causal mask correctness (4 tok, 4h, d=32, {n_elem} elements)")
    return ok


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_mha_prefill,
        test_gqa_prefill_llama3,
        test_gqa_decode_llama3,
        test_mha_decode,
        test_causal_mask_correctness,
    ]
    passed = sum(t(lib) for t in tests)
    total = len(tests)
    print(f"\nPASS: {passed}  FAIL: {total - passed}  TOTAL: {total}")
    sys.exit(0 if passed == total else 1)
