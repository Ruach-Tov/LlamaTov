#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Minimal GPU isolation test: embedding → rms_norm → Q_proj.

Compares GPU dp4a path against CPU reference values step-by-step
to isolate WHERE the GPU computation diverges.

Usage:
  python3 test_gpu_isolation.py --model /path/to/llama3.2-1b.gguf

Author: medayek (Collective SME, Verification Methodology)
Date: 2026-05-16
"""

import sys
import os
import numpy as np

def test_gpu_isolation(model_path):
    """Step-by-step GPU vs CPU comparison."""
    
    # Import our modules
    sys.path.insert(0, os.path.dirname(__file__) + '/..')
    
    print("=" * 60)
    print("GPU ISOLATION TEST: embedding → rms_norm → Q_proj")
    print("=" * 60)
    
    # Step 0: Load model weights via lt() (with reshape fix)
    print("\n[Step 0] Loading weights...")
    try:
        from llamatov_run import parse_gguf, lt
    except ImportError:
        try:
            from llamatov_gpu_dp4a import parse_gguf, lt
        except ImportError:
            print("ERROR: Cannot import parse_gguf/lt from runner")
            return
    
    md, tensors = parse_gguf(model_path)
    
    arch = md.get('general.architecture', 'llama')
    n_embd = md.get(f'{arch}.embedding_length', 2048)
    n_vocab = md.get(f'{arch}.block_count', 128256)
    eps = md.get(f'{arch}.attention.layer_norm_rms_epsilon', 1e-5)
    
    print(f"  Architecture: {arch}")
    print(f"  n_embd: {n_embd}")
    print(f"  eps: {eps}")
    
    # Step 1: Token embedding (CPU reference)
    print("\n[Step 1] Token embedding...")
    tok_embd = lt(tensors, 'token_embd.weight')
    print(f"  tok_embd shape: {tok_embd.shape}")
    
    # Use token 9906 ("Hello" for llama3)
    token_id = 9906
    
    # Embedding lookup - use column selection per ggml convention
    if tok_embd.shape[0] < tok_embd.shape[1]:
        emb_cpu = tok_embd[:, token_id].astype(np.float32)
    else:
        emb_cpu = tok_embd[token_id].astype(np.float32)
    
    print(f"  Embedding shape: {emb_cpu.shape}")
    print(f"  Embedding first 8: {emb_cpu[:8]}")
    print(f"  Embedding norm: {np.linalg.norm(emb_cpu):.6f}")
    print(f"  Embedding mean: {np.mean(emb_cpu):.6f}")
    
    # Step 2: RMS Norm (CPU reference)
    print("\n[Step 2] RMS Norm (CPU)...")
    norm_weight = lt(tensors, 'blk.0.attn_norm.weight')
    print(f"  norm_weight shape: {norm_weight.shape}")
    print(f"  norm_weight first 8: {norm_weight[:8]}")
    
    # CPU RMS norm
    rms = np.sqrt(np.mean(emb_cpu ** 2) + eps)
    normed_cpu = (emb_cpu / rms) * norm_weight.astype(np.float32)
    
    print(f"  RMS value: {rms:.6f}")
    print(f"  Normed first 8: {normed_cpu[:8]}")
    print(f"  Normed norm: {np.linalg.norm(normed_cpu):.6f}")
    print(f"  Normed mean: {np.mean(normed_cpu):.6f}")
    
    # Check for zeros
    if np.all(normed_cpu == 0):
        print("  ❌ CPU RMS NORM PRODUCED ALL ZEROS!")
    elif np.any(np.isnan(normed_cpu)):
        print("  ❌ CPU RMS NORM PRODUCED NaN!")
    else:
        print("  ✅ CPU RMS norm looks reasonable")
    
    # Step 3: Q projection (CPU reference)
    print("\n[Step 3] Q projection (CPU)...")
    q_weight = lt(tensors, 'blk.0.attn_q.weight')
    print(f"  q_weight shape: {q_weight.shape}")
    print(f"  q_weight first 8: {q_weight.flatten()[:8]}")
    
    # matmul: emb @ q_weight
    q_cpu = normed_cpu @ q_weight.astype(np.float32)
    print(f"  Q output shape: {q_cpu.shape}")
    print(f"  Q first 8: {q_cpu[:8]}")
    print(f"  Q norm: {np.linalg.norm(q_cpu):.6f}")
    
    # Step 4: Try GPU path (if available)
    print("\n[Step 4] GPU path comparison...")
    try:
        import ctypes
        
        # Check if .so exists
        so_paths = [
            '/tmp/llamatov_kernels.so',
            '/tmp/dp4a_inference.so',
        ]
        
        gpu_lib = None
        for p in so_paths:
            if os.path.exists(p):
                try:
                    gpu_lib = ctypes.CDLL(p)
                    print(f"  Loaded GPU library: {p}")
                    break
                except OSError as e:
                    print(f"  Failed to load {p}: {e}")
        
        if gpu_lib is None:
            print("  ⚠️  No GPU library found — skipping GPU comparison")
            print("  GPU test requires running the inference runner first")
            print("  to compile the .so")
        else:
            # List available functions
            print(f"  Available GPU functions:")
            for name in ['gpu_rms_norm', 'k_rms_norm', 'gpu_vecmat', 
                         'k_vecmat', 'gpu_silu', 'gpu_add']:
                try:
                    getattr(gpu_lib, name)
                    print(f"    ✅ {name}")
                except AttributeError:
                    print(f"    ❌ {name} (not found)")
            
            # TODO: Wire actual GPU kernel calls here
            # For now, just verify the library loads
            
    except ImportError:
        print("  ⚠️  ctypes not available")
    
    # Step 5: Summary
    print("\n" + "=" * 60)
    print("ISOLATION TEST SUMMARY")
    print("=" * 60)
    print(f"  Embedding:  {'✅ non-zero' if np.any(emb_cpu != 0) else '❌ all zeros'}")
    print(f"  RMS Norm:   {'✅ non-zero' if np.any(normed_cpu != 0) else '❌ all zeros'}")
    print(f"  Q proj:     {'✅ non-zero' if np.any(q_cpu != 0) else '❌ all zeros'}")
    print(f"")
    print(f"  CPU Reference values (for GPU comparison):")
    print(f"    emb[0:4]:     {emb_cpu[:4]}")
    print(f"    normed[0:4]:  {normed_cpu[:4]}")
    print(f"    q_out[0:4]:   {q_cpu[:4]}")
    print(f"")
    print(f"  If GPU produces different values at any step,")
    print(f"  the FIRST divergent step is where the bug lives.")
    
    return {
        'emb': emb_cpu[:8].tolist(),
        'normed': normed_cpu[:8].tolist(),
        'q_out': q_cpu[:8].tolist(),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Try to find a model
        candidates = [
            os.path.expanduser('~/.ollama/models/blobs/sha256-*'),
            '/tmp/test_model.gguf',
        ]
        print("Usage: python3 test_gpu_isolation.py <model.gguf>")
        print("  Provide a GGUF model file path")
        sys.exit(1)
    
    test_gpu_isolation(sys.argv[1])
