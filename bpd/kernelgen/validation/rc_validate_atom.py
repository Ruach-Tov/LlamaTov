#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-RTAAL-1.1
"""rc_validate_atom.py — CAUTIOUS STEP 1 of residual_cache final assembly.
Extract a REAL (residual, attn_norm_w, W_k) from one layer of a model, compute the Python REFERENCE
recompute  K_ref = rms_norm(residual, attn_norm_w) @ W_k  (this is exactly what our validated engine does),
and dump residual+weights+K_ref to a binary fixture. The CUDA kv_direct_recompute kernel will then be run
on the SAME inputs and compared to K_ref (bounded-ULP gate). This validates the recompute ATOM before any
ggml integration — measure against expected values at each step."""
import sys, os, struct
sys.path.insert(0, "os.environ.get("BPD_ROOT","bpd")"); sys.path.insert(0, "os.environ.get("BPD_ROOT","bpd")/lib")
import torch, numpy as np, llamatov_run as R

B = os.environ["BLOB"]
LAYER = int(os.environ.get("LAYER", "0"))
OUT = os.environ.get("FIXTURE", "/tmp/rc_atom_fixture.npz")

def rms_norm(x, w, eps):
    # the exact reference rms_norm from the engine
    return R.rms_norm(x, w, eps)

def main():
    cfg, w = R.load_model(B)
    p = f"blk.{LAYER}"
    eps = cfg.get("norm_eps", 1e-5)
    attn_norm = w[f"{p}.attn_norm.weight"]           # [embd]
    Wk = w[f"{p}.attn_k.weight"]                       # [embd, k_out]
    embd = attn_norm.shape[0]
    k_out = Wk.shape[1]
    # a REAL residual: use the token embedding of a real token as a plausible residual-magnitude input.
    # (For the atom test, any real-magnitude vector exercises the same arithmetic; we use a fixed seed.)
    torch.manual_seed(0)
    emb = w.get("token_embd.weight", w.get("output.weight"))
    # emb may be [embd, vocab] or [vocab, embd]; pick the axis that is NOT embd as the token axis.
    if emb.shape[0] == embd:
        residual = emb[:, 100].clone().float()        # [embd] — token 100's embedding (column)
    else:
        residual = emb[100].clone().float()           # [embd] — token 100's embedding (row)
    # PYTHON REFERENCE recompute (exactly the engine's path):
    h = rms_norm(residual.unsqueeze(0), attn_norm, eps).squeeze(0)   # [embd]
    K_ref = (h @ Wk).float()                                          # [k_out]
    # dump fixture for the CUDA kernel to consume + compare
    np.savez(OUT,
             residual=residual.cpu().numpy().astype(np.float32),
             attn_norm_w=attn_norm.cpu().numpy().astype(np.float32),
             Wk=Wk.cpu().numpy().astype(np.float32),
             K_ref=K_ref.cpu().numpy().astype(np.float32),
             embd=np.int32(embd), k_out=np.int32(k_out), eps=np.float32(eps))
    print(f"  layer {LAYER}: embd={embd} k_out={k_out} eps={eps}")
    print(f"  residual: mean|.|={float(residual.abs().mean()):.5f}")
    print(f"  K_ref:   mean|.|={float(K_ref.abs().mean()):.5f}  [0..3]={K_ref[:4].tolist()}")
    print(f"  fixture -> {OUT}")

if __name__ == "__main__":
    main()
