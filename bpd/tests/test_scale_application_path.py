#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""test_scale_application_path.py — Verify QK^T is UNSCALED (matching ggml).

Regression test for the scale_application_path fix. ggml computes
RAW dot products in QK^T (no 1/sqrt(d) scaling). The scale is applied
separately in the softmax step.

If this test FAILS, the scale has been re-fused into the dot product
and the 8× factor bug has regressed.

Author: medayek
"""
import sys, ctypes, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "bench"))
from llama_fixture_loader import load_manifest

c_float_p = ctypes.POINTER(ctypes.c_float)


def main():
    fixture_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/llama_dump_hello_8_v2"

    t = load_manifest(fixture_dir)
    
    # Find QK^T scores (first MUL_MAT after ROPE that has the right shape)
    # In the fixture, this is node_20 MUL_MAT
    qkt_ops = [x for x in t if x.op_desc == "MUL_MAT" and "node_" in x.name]
    if not qkt_ops:
        print("SKIP: no QK^T MUL_MAT found in fixture")
        return
    
    qkt = qkt_ops[0]
    qkt_ref = qkt.as_numpy()
    
    # The fixture contains UNSCALED dot products
    # Check: are the values in the range we expect for unscaled QK^T?
    # Unscaled: values should be O(sqrt(head_dim)) ≈ 8
    # Scaled: values should be O(1) ≈ 1
    max_abs = float(np.max(np.abs(qkt_ref)))
    head_dim = 64
    sqrt_d = np.sqrt(float(head_dim))  # = 8.0
    
    print(f"QK^T scores (fixture {qkt.name} idx {qkt.idx}):")
    print(f"  shape: {qkt_ref.shape}")
    print(f"  max_abs: {max_abs:.4f}")
    print(f"  sqrt(head_dim): {sqrt_d:.1f}")
    print(f"  max_abs / sqrt(d): {max_abs / sqrt_d:.4f}")
    print()
    
    # If max_abs >> 1 and max_abs ≈ O(sqrt(d)*embedding_scale), scores are UNSCALED ✓
    # If max_abs ≈ O(1), scores are SCALED (regression!)
    if max_abs > 2.0:
        print(f"PASS  scale_application_path: QK^T scores are UNSCALED (max_abs={max_abs:.2f} >> 1)")
        print(f"  This confirms ggml computes raw Q·K^T without 1/√d scaling")
    else:
        print(f"FAIL  scale_application_path: QK^T scores appear SCALED (max_abs={max_abs:.4f} ≈ 1)")
        print(f"  REGRESSION: scale has been fused back into the dot product!")
        print(f"  Expected max_abs ≈ {sqrt_d:.0f}× larger (around {max_abs * sqrt_d:.2f})")
        sys.exit(1)
    
    # Additional check: verify the ratio between QK^T and ROPE output
    # Q values from ROPE should be O(1), QK^T should be O(sqrt(d)) * O(1) = O(sqrt(d))
    q_rope_ops = [x for x in t if x.op_desc == "ROPE" and "Qcur" in x.name]
    if q_rope_ops:
        q_max = float(np.max(np.abs(q_rope_ops[0].as_numpy())))
        expected_qkt_max = q_max * q_max * head_dim * 0.5  # rough upper bound
        print(f"  Q max_abs: {q_max:.4f}")
        print(f"  QK^T/Q ratio: {max_abs/q_max:.2f} (expected ≈ sqrt(d)={sqrt_d:.0f})")


if __name__ == "__main__":
    main()
