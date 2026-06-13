# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
import numpy as np
import ctypes
import sys
sys.path.insert(0, "bench")
try:
    from llama_fixture_loader import get_sources
except ImportError:
    def get_sources(tensors, t): return []

c_float_p = ctypes.POINTER(ctypes.c_float)
c_uint8_p = ctypes.POINTER(ctypes.c_uint8)
c_int32_p = ctypes.POINTER(ctypes.c_int32)


def compare(our, ref):
    our_flat = np.ascontiguousarray(our, dtype=np.float32).flatten()
    ref_flat = np.ascontiguousarray(ref, dtype=np.float32).flatten()
    if our_flat.shape != ref_flat.shape:
        return {"status": "fail", "reason": f"shape mismatch: {our_flat.shape} vs {ref_flat.shape}"}
    our_bits = our_flat.view(np.int32)
    ref_bits = ref_flat.view(np.int32)
    diffs = np.abs(our_bits.astype(np.int64) - ref_bits.astype(np.int64))
    max_ulp = int(diffs.max()) if diffs.size > 0 else 0
    n_diffs = int((diffs > 0).sum())
    max_abs = float(np.abs(our_flat - ref_flat).max())
    if max_ulp == 0:
        return {"status": "pass", "max_ulp": 0, "n_diffs": 0, "n_total": len(our_flat)}
    return {"status": "fail", "max_ulp": max_ulp, "n_diffs": n_diffs,
            "n_total": len(our_flat), "max_abs": max_abs}


def verify_add(lib, tensors, op, idx, ctx):
    """ADD: element-wise residual addition.
    
    In the transformer:
      ffn_inp-N  = attn_out-N + inp_embd (or prev layer out)  
      result-N   = ffn_out-N + ffn_inp-N
    
    Inputs found by name convention:
      ffn_inp-N: sources are attn_out-N (MUL_MAT) and the residual
      result-N:  sources are down-proj output and ffn_inp-N
    
    Since ADD is trivial (no rounding choices), we can verify by
    computing a + b and comparing to ref. But without src links,
    we verify that ref == a + b by finding the two most recent
    tensors before this ADD that have matching shapes.
    """
    if not hasattr(lib, 'bpd_add_f32_cpu'):
        return {"status": "skip", "reason": "bpd_add_f32_cpu not in substrate"}
    
    if not hasattr(lib.bpd_add_f32_cpu, '_argtypes_set'):
        lib.bpd_add_f32_cpu.restype = None
        lib.bpd_add_f32_cpu.argtypes = [c_float_p, c_float_p, c_float_p, ctypes.c_int]
        lib.bpd_add_f32_cpu._argtypes_set = True

    ref = op.as_numpy()
    ref_shape = tuple(ref.shape)
    n = ref.size

    # Find the two source tensors: walk backwards from idx to find
    # the two most recent tensors with matching total element count
    # Use explicit source links if available (issue #56)
    sources = get_sources(tensors, op)
    if len(sources) >= 2:
        candidates = sources[:2]
    else:
        # Fallback: backward walk
        candidates = []
        for t in reversed(tensors[:idx]):
            if t.as_numpy().size == n and t.op_desc not in ("RESHAPE", "VIEW", "PERMUTE", "CONT", "TRANSPOSE"):
                candidates.append(t)
                if len(candidates) >= 2:
                    break
    
    if len(candidates) < 2:
        return {"status": "skip", "reason": f"could not find 2 source tensors for ADD at idx {idx}"}
    
    a = np.ascontiguousarray(candidates[0].as_numpy(), dtype=np.float32).flatten()
    b = np.ascontiguousarray(candidates[1].as_numpy(), dtype=np.float32).flatten()
    
    out = np.zeros(n, dtype=np.float32)
    lib.bpd_add_f32_cpu(
        a.ctypes.data_as(c_float_p),
        b.ctypes.data_as(c_float_p),
        out.ctypes.data_as(c_float_p),
        ctypes.c_int(n))
    
    return compare(out.reshape(ref.shape), ref)


def verify_silu(lib, tensors, op, idx, ctx):
    """SILU: x * sigmoid(x).
    
    In SwiGLU FFN: gate = silu(x @ W_gate)
    Source: the MUL_MAT output immediately before this SILU.
    """
    if not hasattr(lib, 'bpd_silu_f32_cpu'):
        return {"status": "skip", "reason": "bpd_silu_f32_cpu not in substrate"}
    
    if not hasattr(lib.bpd_silu_f32_cpu, '_argtypes_set'):
        lib.bpd_silu_f32_cpu.restype = None
        lib.bpd_silu_f32_cpu.argtypes = [c_float_p, c_float_p, ctypes.c_int]
        lib.bpd_silu_f32_cpu._argtypes_set = True

    ref = op.as_numpy()
    n = ref.size

    # Find source: walk backwards to find the matching-size tensor
    # SILU's input is the gate projection output (a MUL_MAT)
    # Use explicit source links if available (issue #56)
    sources = get_sources(tensors, op)
    if sources:
        src = sources[0]
    else:
        # Fallback: backward walk
        src = None
        for t in reversed(tensors[:idx]):
            if t.as_numpy().size == n and t.op_desc not in ("RESHAPE", "VIEW", "PERMUTE", "CONT", "TRANSPOSE"):
                src = t
                break
    
    if src is None:
        return {"status": "skip", "reason": f"could not find source for SILU at idx {idx}"}
    
    inp = np.ascontiguousarray(src.as_numpy(), dtype=np.float32).flatten()
    out = np.zeros(n, dtype=np.float32)
    lib.bpd_silu_f32_cpu(
        inp.ctypes.data_as(c_float_p),
        out.ctypes.data_as(c_float_p),
        ctypes.c_int(n))
    
    return compare(out.reshape(ref.shape), ref)


def verify_soft_max(lib, tensors, op, idx, ctx):
    """SOFT_MAX: ggml applies scale (1/sqrt(d)) + causal mask internally.
    When SOFT_MAX has 2 sources: src[0]=QK^T, src[1]=causal mask tensor.
    When SOFT_MAX has 1 source: src[0]=QK^T, causal mask applied implicitly."""
    ref = np.ascontiguousarray(op.as_numpy(), dtype=np.float32)
    sources = get_sources(tensors, op)
    if not sources:
        return {"status": "skip", "reason": f"no source for SOFT_MAX at idx {idx}"}
    
    qk = np.ascontiguousarray(sources[0].as_numpy(), dtype=np.float32)
    ne = list(op.ne)
    n_kv, n_q, n_heads = ne[0], ne[1], ne[2]
    
    head_dim = ctx.get("head_dim", 64)
    scale = 1.0 / np.sqrt(float(head_dim))
    
    qk_3d = qk.reshape(n_heads, n_q, n_kv)
    
    # Check for explicit mask tensor (2nd source)
    if len(sources) >= 2:
        mask = np.ascontiguousarray(sources[1].as_numpy(), dtype=np.float32)
        # mask shape might be [n_kv, n_something, 1, 1] — broadcast
        mask_flat = mask.flatten()
        # The mask has 0 for valid and -inf for masked positions
        mask_row = mask_flat[:n_kv]  # first n_kv values are the mask pattern
    else:
        mask_row = None
    
    our_sm = np.zeros_like(qk_3d)
    for h in range(n_heads):
        for q in range(n_q):
            row = qk_3d[h, q, :].copy() * scale
            if mask_row is not None:
                # Apply explicit mask: mask has 0 for valid, -inf for masked
                row = row + mask_row
            else:
                # Apply implicit causal mask
                for k in range(n_kv):
                    if k > q:
                        row[k] = float('-inf')
            mx = row.max()
            if mx == float('-inf'):
                our_sm[h, q, :] = 0
            else:
                exp_row = np.exp(row - mx)
                s = exp_row.sum()
                our_sm[h, q, :] = exp_row / s if s > 0 else 0
    
    return compare(our_sm.flatten(), ref.flatten())


def verify_cpy(lib, tensors, op, idx, ctx):
    """CPY: tensor copy, potentially with dtype conversion (f32→f16 for KV cache).
    Uses source links to find the correct input tensor."""
    sources = get_sources(tensors, op)
    if not sources:
        return {"status": "skip", "reason": f"no source for CPY at idx {idx}"}
    
    src_tensor = sources[0]
    src_data = np.ascontiguousarray(src_tensor.as_numpy(), dtype=np.float32).flatten()
    
    # Check if this is a dtype conversion (f32→f16)
    if op.dtype_name == "f16" and src_tensor.dtype_name == "f32":
        # Compare as f16: convert source f32→f16 via numpy, compare against fixture f16
        n_f16 = len(op.data) // 2
        ggml_f16 = np.frombuffer(op.data, dtype=np.float16)
        our_f16 = src_data[:n_f16].astype(np.float16)
        # Compare the f16 bit patterns
        our_bits = our_f16.view(np.uint16).astype(np.int64)
        ggml_bits = ggml_f16.view(np.uint16).astype(np.int64)
        diffs = np.abs(our_bits - ggml_bits)
        max_ulp = int(diffs.max()) if diffs.size > 0 else 0
        n_diffs = int((diffs > 0).sum())
        if max_ulp == 0:
            return {"status": "pass", "max_ulp": 0, "n_diffs": 0, "n_total": len(ggml_f16)}
        return {"status": "fail", "max_ulp": max_ulp, "n_diffs": n_diffs,
                "n_total": len(ggml_f16), "max_abs": float(np.abs(our_f16.astype(np.float32) - ggml_f16.astype(np.float32)).max())}
    
    # Same-dtype copy: compare directly
    ref = np.ascontiguousarray(op.as_numpy(), dtype=np.float32).flatten()
    return compare(src_data[:len(ref)], ref)


