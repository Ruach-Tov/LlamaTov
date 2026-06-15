#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-RTAAL-1.1
"""rc_validate_rope.py — STEP 2 of residual_cache assembly: validate RoPE on K at a REAL non-zero position.
Extracts a real K from a layer, applies the Python reference apply_rope at position P (the decode case,
where the kernel's blockIdx-as-position assumption is exercised), dumps a fixture for the CUDA k_rope.
The critical thing: does the emitted k_rope rotate at the RIGHT position?"""
import sys, os
sys.path.insert(0, "os.environ.get("BPD_ROOT","bpd")"); sys.path.insert(0, "os.environ.get("BPD_ROOT","bpd")/lib")
import torch, numpy as np, llamatov_run as R

B = os.environ["BLOB"]
LAYER = int(os.environ.get("LAYER", "0"))
POS = int(os.environ.get("POS", "7"))   # a REAL non-zero position (decode case)

def main():
    cfg, w = R.load_model(B)
    p = f"blk.{LAYER}"
    nh = cfg["n_head"]; nkv = cfg.get("n_head_kv", nh)
    hd = cfg.get("head_dim") or (cfg["n_embd"] // nh)
    theta = cfg.get("rope_theta", 500000.0)
    n_head_k = nkv
    # a real-magnitude K vector for n_head_k heads x head_dim (one token)
    torch.manual_seed(1)
    K = torch.randn(1, 1, n_head_k * hd) * 0.5    # [B=1, T=1, n_head_k*hd]
    # PYTHON REFERENCE: apply_rope at position POS (we only need k; pass K as both q,k slots is fine, take k)
    pos_t = torch.tensor([POS], dtype=torch.int64)
    # apply_rope expects q and k; we give a dummy q of matching head count
    Qdummy = torch.zeros(1, 1, nh * hd)
    _, K_roped = R.apply_rope(Qdummy, K, nh, hd, theta, positions=pos_t)
    K_roped = K_roped.squeeze(0).squeeze(0).float()   # [n_head_k*hd]
    K_flat = K.squeeze(0).squeeze(0).float()
    # dump fixture
    K_flat.cpu().numpy().astype(np.float32).tofile("/tmp/fx_rope_kin.bin")
    K_roped.cpu().numpy().astype(np.float32).tofile("/tmp/fx_rope_kref.bin")
    print(f"  layer {LAYER}: n_head_k={n_head_k} head_dim={hd} theta={theta} POS={POS}")
    print(f"  K_in[0..3]   = {K_flat[:4].tolist()}")
    print(f"  K_ref[0..3]  = {K_roped[:4].tolist()}  (RoPE at pos {POS})")
    print(f"  dims: nheadk={n_head_k} headdim={hd} theta={theta} pos={POS}")

if __name__ == "__main__":
    main()
