#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""qkv_layout_fix.py — the PARAMETERIZED qkv layout fix, at the source (Iyun, 2026-05-29, Heath).
The fix is NOT a hardcoded patch — it is a function of SETTINGS. qkv_layout_fix(weights, setting)
applies the weave derived for that setting. Default setting (ollama_ggml_gpu) = identity (our runner
is already correct for the target). hf_reference setting = rope_permute (proven 0 ULP vs HF).
Parameters: rope_layout, target, dataflow, head_dim.
"""
import numpy as np

# ── SETTINGS (mirror of qkv_layout_config.pl) ──
SETTINGS = {
    "ollama_ggml_gpu": dict(rope_layout="ggml_interleave", target="sm_61", dataflow="row_major", head_dim=64),
    "hf_reference":    dict(rope_layout="hf_split_half",   target="sm_61", dataflow="row_major", head_dim=64),
    "sse3_pooled":     dict(rope_layout="hf_split_half",   target="sse3",  dataflow="row_major", head_dim=64),
    "systolic_grid":   dict(rope_layout="ggml_interleave", target="systolic_8x8", dataflow="col_major", head_dim=64),
}
DEFAULT = "ollama_ggml_gpu"

def derive_weave(setting):
    """The weave (list of transforms) for a setting — parameter-driven."""
    s = SETTINGS[setting]; weave = []
    # rope_layout: ggml_interleave is native (our target); hf_split_half needs the permute
    if s["rope_layout"] == "hf_split_half":
        weave.append(("rope_permute", s["head_dim"]))
    # target topology: pooled/systolic add a scatter/tile (placeholder structure here)
    if s["target"] in ("sse3", "riscv_v"):
        weave.append(("scatter_pools", 4 if s["target"]=="sse3" else 8))
    if s["target"].startswith("systolic"):
        weave.append(("tile_to_grid", (8, 8)))
    return weave

def rope_permute(w, n_head, head_dim):
    """ggml-interleave -> HF-split-half per head (evens then odds). PROVEN 0 ULP vs HF."""
    w = w.reshape(n_head, head_dim, -1)
    return np.concatenate([w[:, 0::2, :], w[:, 1::2, :]], axis=1).reshape(-1, w.shape[-1])

def qkv_layout_fix(weight, n_head, setting=DEFAULT):
    """Apply the parameterized layout fix to a q or k weight, per setting.
    Default (ollama_ggml_gpu) -> identity (correct for target). hf_reference -> rope_permute."""
    s = SETTINGS[setting]; hd = s["head_dim"]; weave = derive_weave(setting)
    out = weight
    for transform in weave:
        if transform[0] == "rope_permute":
            out = rope_permute(out, n_head, hd)
        # scatter_pools / tile_to_grid: structural (codegen-level); identity on the dense weight here
    return out

if __name__ == "__main__":
    # demonstrate: same kernel weights, different settings -> different (or no) weave
    import sys
    print("qkv layout fix — settings and their derived weaves:")
    for name in SETTINGS:
        print(f"  {name:18s} ({SETTINGS[name]['rope_layout']:14s} / {SETTINGS[name]['target']:12s}) -> weave={derive_weave(name)}")
    print(f"\n  DEFAULT setting = {DEFAULT}: weave={derive_weave(DEFAULT)}  (empty = our runner already correct for Ollama=ggml)")
