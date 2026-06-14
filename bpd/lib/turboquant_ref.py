#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""turboquant_ref.py — reference implementation + verification of the TurboQuant CORE
(arXiv:2504.19874): random rotation -> per-coordinate scalar quantization.

The paper's mechanism: randomly rotate a vector (a structured orthogonal transform), which concentrates
each coordinate into a Beta distribution; in high dim the rotated coordinates are near-independent, so an
optimal per-coordinate scalar quantizer is near-optimal overall. The paper reports KV-cache quality-
neutral at 3.5 bits/channel. This measures whether OUR rotation+scalar-quant core hits the near-optimal
distortion BEFORE we wire it as a transform — measure-first, ground the math against the paper.

Scope: the MSE (single-stage) core. The 2-stage QJL inner-product debiasing is a further milestone.
"""
import numpy as np

def hadamard(n):
    """A 2^k Hadamard matrix (a fast, structured orthogonal rotation — the paper's 'random rotation'
    is typically a randomized Hadamard transform: sign-flip diag then Hadamard)."""
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H[:n, :n] / np.sqrt(n)

def randomized_rotation(d, seed):
    """Randomized orthogonal rotation R = H @ diag(s), s in {+-1}. R is orthonormal (R^T R = I)."""
    rng = np.random.default_rng(seed)
    n = 1
    while n < d: n <<= 1
    H = hadamard(n)
    s = rng.choice([-1.0, 1.0], size=n)
    R = H * s[None, :]          # H @ diag(s)
    return R[:d, :d] if n == d else (H[:d, :d] * s[:d][None, :])

def scalar_quantize(x, bits):
    """Optimal-ish uniform scalar quantizer over [-A, A] with A = max|x| (per the concentrated range).
    bits per coordinate -> 2^bits levels."""
    levels = (1 << bits) - 1
    A = np.abs(x).max() + 1e-9
    q = np.round((x / A) * (levels/2)) 
    q = np.clip(q, -(levels//2), levels//2)
    xr = q / (levels/2) * A
    return xr

def turboquant_core(x, bits, seed=0):
    """Random rotation -> per-coordinate scalar quant -> inverse rotation."""
    d = x.shape[0]
    n = 1
    while n < d: n <<= 1
    xp = np.zeros(n); xp[:d] = x          # pad to power of 2 for Hadamard
    R = randomized_rotation(n, seed)
    y = R @ xp                             # rotate
    yq = scalar_quantize(y, bits)          # per-coordinate scalar quant (rotated coords ~ Beta, near-indep)
    xr = R.T @ yq                          # inverse rotation (R orthonormal)
    return xr[:d]

def main():
    np.random.seed(0)
    print("=== TurboQuant core: random rotation + per-coordinate scalar quant ===")
    print("    (paper arXiv:2504.19874: KV-cache quality-neutral @ 3.5 bits/channel)\n")
    HD = 128  # qwen2 K/V dim (2 kv heads x 64)
    for bits in (8, 4, 3, 2):
        # compare TurboQuant (rotate+quant) vs PLAIN scalar quant (no rotation) — the paper's claim is
        # the rotation makes scalar quant near-optimal, so TQ should beat plain at low bits.
        tq_mse, plain_mse = [], []
        for _ in range(300):
            x = np.random.randn(HD).astype(np.float64)
            xr_tq = turboquant_core(x, bits, seed=np.random.randint(1<<30))
            xr_pl = scalar_quantize(x, bits)
            tq_mse.append(((xr_tq - x)**2).mean())
            plain_mse.append(((xr_pl - x)**2).mean())
        tq, pl = np.mean(tq_mse), np.mean(plain_mse)
        snr_tq = 10*np.log10(1.0/(tq+1e-30))   # signal var=1 (standard normal)
        print(f"  {bits} bits: TurboQuant MSE={tq:.5f} (SNR {snr_tq:5.1f}dB)  vs plain-scalar MSE={pl:.5f}  "
              f"-> TQ is {pl/tq:.2f}x {'better' if tq<pl else 'WORSE'}")
    print("\n-> The rotation should make low-bit scalar quant markedly better (the near-optimal-distortion claim).")

if __name__ == "__main__":
    main()
