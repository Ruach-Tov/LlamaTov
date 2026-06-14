#!/usr/bin/env python3
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
# SPDX-License-Identifier: LicenseRef-RTAAL-1.1
"""tq_innerprod.py — does the TurboQuant MSE core preserve the INNER PRODUCT (attention K.Q)?
The paper warns MSE quantizers BIAS inner products. Attention scores are q.k, so before wiring
turboquant into the K/V path we must measure: is q.(quant K) ~ q.k (unbiased), or systematically off?
This decides whether we need the 2-stage QJL debiasing before the transform is honest."""
import sys; import os as _o; sys.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
import numpy as np
from turboquant_ref import turboquant_core, scalar_quantize

def q8_roundtrip(x):
    n=(x.shape[0]//32)*32; xb=x[:n].reshape(-1,32); amax=np.abs(xb).max(1,keepdims=True)
    d=np.where(amax>0,amax/127,1); dh=d.astype(np.float16).astype(np.float64)
    q=np.round(xb/dh).clip(-127,127); return (q*dh).reshape(-1)

def main():
    np.random.seed(3); HD=128; N=2000
    print("=== inner-product fidelity: q.(quant K) vs q.k  (attention is an inner product) ===\n")
    for bits,label,quant in [(8,"Q8_0 (our shipped K/V quant)", lambda k:q8_roundtrip(k)),
                             (4,"TurboQuant 4-bit", lambda k:turboquant_core(k,4,np.random.randint(1<<30))),
                             (3,"TurboQuant 3-bit", lambda k:turboquant_core(k,3,np.random.randint(1<<30)))]:
        rel_err, bias = [], []
        for _ in range(N):
            # realistic K with an outlier channel + a query q
            k=np.r_[np.random.randn(HD-1)*0.4,[5.0]].astype(np.float64)
            q=np.random.randn(HD).astype(np.float64)*0.4
            true=q@k; approx=q@quant(k)
            rel_err.append(abs(approx-true)/(abs(true)+1e-6))
            bias.append(approx-true)
        mb=np.mean(bias)
        print(f"  {label:30}: mean|rel err|={np.mean(rel_err):.4f}  mean bias={mb:+.5f}  "
              f"({'~UNBIASED' if abs(mb)<0.01 else 'BIASED'})")
    print("\n-> If TurboQuant inner-product is BIASED, the 2-stage QJL residual is needed before wiring.")
    print("   If Q8_0 is already ~unbiased + accurate, kv_quantize_q8 stays the safe default; TurboQuant")
    print("   only wins at lower bits AND needs the debiasing to match attention fidelity.")

if __name__=="__main__":
    main()
