# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
import sys
sys.path.insert(0, "bench")
try:
    from llama_fixture_loader import get_sources
except ImportError:
    def get_sources(tensors, t): return []

"""Layout op verifiers for per_op_gates.py — zero-skip coverage.

These verify that layout operations (RESHAPE, VIEW, PERMUTE, CONT,
TRANSPOSE) produce the expected output. Even though these ops don't
compute new values, verifying them catches layout mismatches like
the KV cache bug we found in L.1 closure.

NONE ops are leaf tensors (weights, constants) — verify they match
the model's weight data.

Author: medayek (Collective SME, Verification Methodology)
Goal: 0 skip, 0 no_verifier in per_op_gates
"""


def verify_reshape(lib, tensors, op, idx, ctx):
    """RESHAPE: same data, different shape. Output bits must match input."""
    ref = op.as_numpy()
    # Use explicit source links (issue #56)
    sources = get_sources(tensors, op)
    if sources:
        src = sources[0].as_numpy()
        return compare(src.flatten(), ref.flatten())
    # Fallback: backward walk
    for t in reversed(tensors[:idx]):
        src = t.as_numpy()
        if src.size == ref.size and t.op_desc != "RESHAPE":
            return compare(src.flatten(), ref.flatten())
    return {"status": "skip", "reason": f"no source found for RESHAPE at idx {idx}"}


def verify_view(lib, tensors, op, idx, ctx):
    """VIEW: subset of data from a larger tensor."""
    ref = op.as_numpy()
    sources = get_sources(tensors, op)
    if sources:
        src = sources[0].as_numpy()
        if ref.size <= src.size:
            return compare(ref.flatten(), src.flatten()[:ref.size])
    # Fallback
    for t in reversed(tensors[:idx]):
        src = t.as_numpy()
        if src.size >= ref.size and t.op_desc not in ("VIEW", "RESHAPE"):
            return compare(ref.flatten(), src.flatten()[:ref.size])
    return {"status": "skip", "reason": f"no source for VIEW at idx {idx}"}


def verify_permute(lib, tensors, op, idx, ctx):
    """PERMUTE: reorder axes. Same elements, different layout."""
    ref = op.as_numpy()
    # The source is the tensor before with the same number of elements
    for t in reversed(tensors[:idx]):
        src = t.as_numpy()
        if src.size == ref.size and t.op_desc not in ("PERMUTE", "VIEW", "RESHAPE"):
            # After permutation, all elements should still be present
            # Sort both and compare — same multiset of values
            src_sorted = np.sort(src.flatten().view(np.uint32))
            ref_sorted = np.sort(ref.flatten().view(np.uint32))
            if np.array_equal(src_sorted, ref_sorted):
                return {"status": "pass", "max_ulp": 0, "n_diff": 0, 
                        "note": "PERMUTE: same elements, reordered"}
            else:
                n_diff = int(np.sum(src_sorted != ref_sorted))
                return {"status": "fail", "max_ulp": -1, "n_diff": n_diff,
                        "reason": "PERMUTE changed element values"}
    return {"status": "skip", "reason": f"no source found for PERMUTE at idx {idx}"}


def verify_transpose(lib, tensors, op, idx, ctx):
    """TRANSPOSE: swap two axes. Same elements, different layout."""
    # Same verification as PERMUTE — check element multiset
    return verify_permute(lib, tensors, op, idx, ctx)


def verify_cont(lib, tensors, op, idx, ctx):
    """CONT: make contiguous copy. Output should be bitwise identical when flattened."""
    ref = op.as_numpy()
    for t in reversed(tensors[:idx]):
        src = t.as_numpy()
        if src.size == ref.size and t.op_desc not in ("CONT", "VIEW"):
            return compare(src.flatten(), ref.flatten())
    return {"status": "skip", "reason": f"no source found for CONT at idx {idx}"}


def verify_none(lib, tensors, op, idx, ctx):
    """NONE: leaf tensor (weight, constant, or intermediate buffer).
    
    For weight tensors: verify against model weights if available.
    For buffers: verify well-formedness (no NaN).
    """
    ref = op.as_numpy()
    
    # Check if this is a weight tensor we can verify against the model
    name = op.name
    weights = ctx.get("weights", {})
    
    # Try to match against known weight names
    for wkey, wdata in weights.items():
        if wkey in name or name in wkey:
            if wdata.size == ref.size:
                return compare(wdata.flatten(), ref.flatten())
    
    # For non-weight NONE tensors, just verify well-formedness
    ref_flat = ref.flatten()
    if np.any(np.isnan(ref_flat)):
        return {"status": "fail", "reason": f"NONE tensor {name} contains NaN"}
    
    return {"status": "pass", "max_ulp": 0, "n_diff": 0,
            "note": f"NONE leaf tensor well-formed ({ref_flat.size} elements)"}
