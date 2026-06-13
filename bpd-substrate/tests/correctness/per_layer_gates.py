#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""per_layer_gates.py — verify each transformer layer's output against ggml's.

For each of the N transformer layers, drive bpd_llama_block_cpu (the C
function) per-layer with real Q8_0 weights from the GGUF, compare the
output residual stream to the captured l_out-N fixture from
llama-eval-callback's trace.

Substrate-design discipline: the middle-resolution gate. Per-op gates
locate divergence to a specific sub-operation; per-layer gates show how
that divergence accumulates layer-by-layer. End-to-end gates show the
final logit-distribution effect.

Multi-sovereign-verifiable: produces a JSON report comparing our
per-layer outputs against ggml's. Different verifiers should produce
identical reports modulo hardware-specific drift (which we expect to be
0 ULP when the substrate-design discipline is honored).
"""
import argparse
import ctypes
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HARNESS_VERSION = "1.0.0"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "bench"))

from llama_fixture_loader import load_manifest, find_op  # noqa: E402
from bpd_llamatov_infer import (  # noqa: E402
    BpdLlamaConfig, BpdLlamaLayerWeights, BpdLlamaWeights,
    build_model, c_float_p, c_uint8_p, c_int32_p, c_long_p,
)


def ulp_distance(a, b):
    a = np.ascontiguousarray(a, dtype=np.float32).reshape(-1)
    b = np.ascontiguousarray(b, dtype=np.float32).reshape(-1)
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    BASE = np.int64(0x80000000)
    ai = np.where(ai < 0, BASE - ai, ai)
    bi = np.where(bi < 0, BASE - bi, bi)
    diffs = np.abs(ai - bi)
    return int(diffs.max()), int((diffs > 0).sum()), int(diffs.size)


def compare(ours, ref):
    ours_flat = np.ascontiguousarray(ours, dtype=np.float32).reshape(-1)
    ref_flat = np.ascontiguousarray(ref, dtype=np.float32).reshape(-1)
    if ours_flat.shape != ref_flat.shape:
        return {"status": "shape_mismatch",
                "ours_shape": list(ours.shape), "ref_shape": list(ref.shape)}
    max_abs = float(np.abs(ours_flat - ref_flat).max())
    max_ulp, n_diff, n_total = ulp_distance(ours_flat, ref_flat)
    cos_sim = float(
        np.dot(ours_flat, ref_flat) /
        (np.linalg.norm(ours_flat) * np.linalg.norm(ref_flat) + 1e-12))
    return {
        "status": "pass" if max_ulp == 0 else "fail",
        "max_abs": max_abs,
        "max_ulp": max_ulp,
        "n_diff": n_diff,
        "n_total": n_total,
        "cosine_similarity": cos_sim,
    }


def gather_hardware_info():
    info = {"cpu_model": "unknown", "isa": [], "compiler": "unknown",
            "hostname": socket.gethostname(), "platform": platform.platform()}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name") and info["cpu_model"] == "unknown":
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                if line.startswith("flags") and not info["isa"]:
                    flags = line.split(":", 1)[1].strip().split()
                    info["isa"] = sorted(f for f in flags if f in
                        {"sse", "sse2", "sse3", "ssse3", "sse4_1", "sse4_2",
                         "avx", "avx2", "avx512f", "fma", "f16c"})
    except FileNotFoundError:
        pass
    try:
        gcc = subprocess.check_output(["gcc", "--version"], stderr=subprocess.STDOUT).decode()
        info["compiler"] = gcc.split("\n")[0]
    except Exception:
        pass
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fixture-dir", required=True)
    p.add_argument("--so", required=True)
    p.add_argument("--gguf", required=True)
    p.add_argument("--tokens", required=True,
                   help="Comma-separated token IDs matching the captured fixture's prompt")
    p.add_argument("--verifier", default=None)
    p.add_argument("--report", default="/tmp/per_layer_report.json")
    args = p.parse_args()

    print(f"[harness] LlamaTov per-layer gates v{HARNESS_VERSION}")

    lib = ctypes.CDLL(args.so)
    lib.bpd_llama_block_cpu.restype = None
    lib.bpd_llama_block_cpu.argtypes = [
        c_float_p, ctypes.POINTER(BpdLlamaLayerWeights),
        ctypes.POINTER(BpdLlamaConfig), c_int32_p,
        ctypes.c_int, ctypes.c_int,
        c_float_p, c_float_p,
        c_float_p, c_float_p, c_float_p,
    ]
    lib.bpd_embed_lookup_q8_0_cpu.restype = None
    lib.bpd_embed_lookup_q8_0_cpu.argtypes = [
        c_uint8_p, c_int32_p, c_float_p, ctypes.c_int, ctypes.c_int,
    ]

    cfg, weights, loader = build_model(args.gguf)
    tensors = load_manifest(args.fixture_dir)
    print(f"[harness] cfg.n_layers={cfg.n_layers}, fixture tensors={len(tensors)}")

    prompt_tokens = [int(t) for t in args.tokens.split(",")]
    n_tokens = len(prompt_tokens)

    # Embed lookup
    token_ids = np.ascontiguousarray(prompt_tokens, dtype=np.int32)
    x = np.zeros(n_tokens * cfg.embed_dim, dtype=np.float32)
    lib.bpd_embed_lookup_q8_0_cpu(
        weights.token_embd, token_ids.ctypes.data_as(c_int32_p),
        x.ctypes.data_as(c_float_p),
        ctypes.c_int(n_tokens), ctypes.c_int(cfg.embed_dim))

    ref_inp = find_op(tensors, name_substring="inp_embd", op_desc="GET_ROWS")
    embed_result = compare(x.reshape(n_tokens, cfg.embed_dim), ref_inp.as_numpy())
    embed_result["layer"] = "embed"
    print(f"  ✅ embed: max_ulp={embed_result['max_ulp']}, n_diff={embed_result['n_diff']}/{embed_result['n_total']}")

    pos_ids = np.arange(n_tokens, dtype=np.int32)
    kv_per_layer = cfg.max_seq_len * cfg.n_kv_heads * cfg.head_dim
    k_cache_all = np.zeros(cfg.n_layers * kv_per_layer, dtype=np.float32)
    v_cache_all = np.zeros(cfg.n_layers * kv_per_layer, dtype=np.float32)
    max_dim = max(cfg.embed_dim, cfg.ffn_dim)
    max_proj = max(cfg.n_heads * cfg.head_dim, cfg.ffn_dim)
    s1 = np.zeros(n_tokens * max_dim, dtype=np.float32)
    s2 = np.zeros(n_tokens * max_proj, dtype=np.float32)
    s3 = np.zeros(n_tokens * max_proj, dtype=np.float32)

    per_layer_results = [embed_result]
    first_divergent = None
    for li in range(cfg.n_layers):
        s1[:] = 0; s2[:] = 0; s3[:] = 0
        k_cache_l = k_cache_all[li * kv_per_layer:(li + 1) * kv_per_layer]
        v_cache_l = v_cache_all[li * kv_per_layer:(li + 1) * kv_per_layer]
        lw_ptr = ctypes.byref(weights.layers[li])
        lib.bpd_llama_block_cpu(
            x.ctypes.data_as(c_float_p), lw_ptr, ctypes.byref(cfg),
            pos_ids.ctypes.data_as(c_int32_p),
            ctypes.c_int(n_tokens), ctypes.c_int(0),
            k_cache_l.ctypes.data_as(c_float_p),
            v_cache_l.ctypes.data_as(c_float_p),
            s1.ctypes.data_as(c_float_p),
            s2.ctypes.data_as(c_float_p),
            s3.ctypes.data_as(c_float_p))

        ref = find_op(tensors, name_substring=f"l_out-{li}")
        if ref is not None:
            res = compare(x.reshape(n_tokens, cfg.embed_dim), ref.as_numpy())
            res["layer"] = li
            per_layer_results.append(res)
            icon = "✅" if res["status"] == "pass" else "❌"
            print(f"  {icon} layer {li:2d}: max_abs={res.get('max_abs', 0):.4e}, "
                  f"max_ulp={res.get('max_ulp')}, cos_sim={res.get('cosine_similarity', 0):.6f}")
            if res["status"] == "fail" and first_divergent is None:
                first_divergent = li
        else:
            per_layer_results.append({"layer": li, "status": "fixture_missing"})

    report = {
        "harness_version": HARNESS_VERSION,
        "verifier": args.verifier or os.environ.get("USER", "unknown"),
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fixture_dir": args.fixture_dir,
        "model": {"path": args.gguf},
        "hardware": gather_hardware_info(),
        "tokens": prompt_tokens,
        "per_layer_results": per_layer_results,
        "summary": {
            "first_divergent_layer": first_divergent,
            "n_layers_pass": sum(1 for r in per_layer_results if r.get("status") == "pass"),
            "n_layers_fail": sum(1 for r in per_layer_results if r.get("status") == "fail"),
        },
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[harness] wrote {args.report}")
    print(f"[harness] first_divergent_layer = {first_divergent}")
    loader.close()


if __name__ == "__main__":
    main()
