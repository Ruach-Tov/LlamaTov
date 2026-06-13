#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""
llamatov_inference.py — End-to-end inference: GGUF → tokens

The simplest possible path to "GGUF in, token out":
1. Parse GGUF header + tensor metadata
2. Memory-map tensor weights  
3. Execute transformer forward pass via PyTorch
4. Sample output token

No fusion yet — just get the pipeline working.
Fusion is an optimization applied AFTER correctness.
"""

import struct
import numpy as np
import sys
import os

# Check if torch available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not available, will use numpy only")

# ═══════════════════════════════════════════════════════════════
# GGUF PARSER (minimal — just enough to load tensors)
# ═══════════════════════════════════════════════════════════════

GGUF_MAGIC = 0x46554747  # "GGUF" as little-endian uint32
GGUF_VERSION = 3

# GGUF value types
GGUF_TYPE = {
    0: ('uint8', 'B', 1),
    1: ('int8', 'b', 1),
    2: ('uint16', 'H', 2),
    3: ('int16', 'h', 2),
    4: ('uint32', 'I', 4),
    5: ('int32', 'i', 4),
    6: ('float32', 'f', 4),
    7: ('bool', '?', 1),
    8: ('string', None, None),
    9: ('array', None, None),
    10: ('uint64', 'Q', 8),
    11: ('int64', 'q', 8),
    12: ('float64', 'd', 8),
}

# GGML tensor types
GGML_TYPE_SIZE = {
    0: ('F32', 4, 1),    # float32
    1: ('F16', 2, 1),    # float16
    2: ('Q4_0', 2+16, 32),  # quantized 4-bit (block of 32)
    3: ('Q4_1', 4+16, 32),
    6: ('Q5_0', 4+16, 32),
    7: ('Q5_1', 4+16, 32),
    8: ('Q8_0', 2+32, 32),
    9: ('Q8_1', 8+32, 32),
    10: ('Q2_K', None, None),  # k-quant
    11: ('Q3_K', None, None),
    12: ('Q4_K', None, None),
    13: ('Q5_K', None, None),
    14: ('Q6_K', None, None),
}

def read_string(f):
    """Read a GGUF string (length-prefixed)."""
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8')

def read_value(f, vtype):
    """Read a GGUF value of the given type."""
    if vtype == 8:  # string
        return read_string(f)
    elif vtype == 9:  # array
        atype = struct.unpack('<I', f.read(4))[0]
        count = struct.unpack('<Q', f.read(8))[0]
        return [read_value(f, atype) for _ in range(count)]
    else:
        name, fmt, size = GGUF_TYPE[vtype]
        return struct.unpack('<' + fmt, f.read(size))[0]

def parse_gguf(path):
    """Parse GGUF file, return metadata + tensor info."""
    with open(path, 'rb') as f:
        # Header
        magic = struct.unpack('<I', f.read(4))[0]
        assert magic == GGUF_MAGIC, f"Not a GGUF file (magic: {magic:#x})"
        
        version = struct.unpack('<I', f.read(4))[0]
        assert version == GGUF_VERSION, f"Unsupported GGUF version: {version}"
        
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]
        
        print(f"GGUF v{version}: {n_tensors} tensors, {n_kv} metadata entries")
        
        # Metadata key-value pairs
        metadata = {}
        for _ in range(n_kv):
            key = read_string(f)
            vtype = struct.unpack('<I', f.read(4))[0]
            value = read_value(f, vtype)
            metadata[key] = value
        
        # Tensor info
        tensors = {}
        for _ in range(n_tensors):
            name = read_string(f)
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            ttype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            tensors[name] = {
                'dims': dims,
                'type': ttype,
                'type_name': GGML_TYPE_SIZE.get(ttype, ('unknown',))[0],
                'offset': offset,
            }
        
        # Data starts at next alignment boundary
        alignment = metadata.get('general.alignment', 32)
        data_offset = f.tell()
        data_offset = (data_offset + alignment - 1) // alignment * alignment
        
    return metadata, tensors, data_offset

def load_tensor_f32(path, tensor_info, data_offset):
    """Load a tensor as float32 numpy array (dequantize if needed)."""
    info = tensor_info
    dims = info['dims']
    ttype = info['type']
    offset = data_offset + info['offset']
    
    if ttype == 0:  # F32
        count = 1
        for d in dims: count *= d
        with open(path, 'rb') as f:
            f.seek(offset)
            data = np.frombuffer(f.read(count * 4), dtype=np.float32)
        return data.reshape(dims)
    elif ttype == 1:  # F16
        count = 1
        for d in dims: count *= d
        with open(path, 'rb') as f:
            f.seek(offset)
            data = np.frombuffer(f.read(count * 2), dtype=np.float16)
        return data.astype(np.float32).reshape(dims)
    else:
        # For quantized types, we'd need dequantization
        # For now, return None and skip
        return None

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: llamatov_inference.py <gguf_file>")
        sys.exit(1)
    
    path = sys.argv[1]
    print(f"Loading {path}...")
    
    metadata, tensors, data_offset = parse_gguf(path)
    
    # Print key metadata
    arch = metadata.get('general.architecture', 'unknown')
    name = metadata.get('general.name', 'unknown')
    print(f"Architecture: {arch}")
    print(f"Model name: {name}")
    print(f"Tensors: {len(tensors)}")
    
    # Print tensor summary
    print(f"\nTensor inventory:")
    for tname, tinfo in sorted(tensors.items()):
        print(f"  {tname}: {tinfo['dims']} ({tinfo['type_name']})")
    
    # Try loading token embedding
    embd_key = 'token_embd.weight'
    if embd_key in tensors:
        print(f"\nLoading {embd_key}...")
        embd = load_tensor_f32(path, tensors[embd_key], data_offset)
        if embd is not None:
            print(f"  Shape: {embd.shape}, dtype: {embd.dtype}")
            print(f"  Sample values: {embd[0,:5]}")
            print(f"  GGUF → numpy tensor: SUCCESS")
        else:
            print(f"  Quantized ({tensors[embd_key]['type_name']}), dequant needed")
    
    print(f"\n=== GGUF parsing complete ===")
    print(f"Next step: execute forward pass through {arch} graph")

# ═══════════════════════════════════════════════════════════════
# DEQUANTIZATION (Q2_K, Q3_K, Q6_K)
# ═══════════════════════════════════════════════════════════════

def dequant_q2_k(data, shape):
    """Dequantize Q2_K format to float32."""
    n_elements = 1
    for d in shape: n_elements *= d
    
    block_size = 256  # elements per super-block
    n_blocks = n_elements // block_size
    bytes_per_block = 84  # 16 + 64 + 2 + 2
    
    result = np.zeros(n_elements, dtype=np.float32)
    
    for i in range(n_blocks):
        offset = i * bytes_per_block
        block = data[offset:offset + bytes_per_block]
        
        scales_bytes = block[0:16]
        qs = block[16:80]
        d = np.frombuffer(block[80:82], dtype=np.float16)[0].astype(np.float32)
        dmin = np.frombuffer(block[82:84], dtype=np.float16)[0].astype(np.float32)
        
        # Extract 4-bit scales (16 sub-blocks)
        scales = np.zeros(16, dtype=np.float32)
        mins = np.zeros(16, dtype=np.float32)
        for j in range(8):
            sc_byte = scales_bytes[j]
            scales[j] = d * (sc_byte & 0xF)
            mins[j] = dmin * (sc_byte >> 4)
        for j in range(8):
            sc_byte = scales_bytes[8 + j]
            scales[8 + j] = d * (sc_byte & 0xF)
            mins[8 + j] = dmin * (sc_byte >> 4)
        
        # Dequantize 2-bit values
        for j in range(64):
            q_byte = qs[j]
            base_idx = i * block_size
            for k in range(4):
                q_val = (q_byte >> (k * 2)) & 0x3
                elem_idx = j * 4 + k
                sub_block = elem_idx // 16
                result[base_idx + elem_idx] = scales[sub_block] * q_val - mins[sub_block]
    
    return result.reshape(shape)

def load_tensor_any(path, tensor_info, data_offset):
    """Load a tensor, dequantizing if needed."""
    info = tensor_info
    dims = info['dims']
    ttype = info['type']
    offset = data_offset + info['offset']
    
    if ttype == 0:  # F32
        count = 1
        for d in dims: count *= d
        with open(path, 'rb') as f:
            f.seek(offset)
            data = np.frombuffer(f.read(count * 4), dtype=np.float32)
        return data.reshape(dims)
    elif ttype == 1:  # F16
        count = 1
        for d in dims: count *= d
        with open(path, 'rb') as f:
            f.seek(offset)
            data = np.frombuffer(f.read(count * 2), dtype=np.float16)
        return data.astype(np.float32).reshape(dims)
    elif ttype == 10:  # Q2_K
        n_elements = 1
        for d in dims: n_elements *= d
        n_blocks = n_elements // 256
        byte_count = n_blocks * 84
        with open(path, 'rb') as f:
            f.seek(offset)
            raw = np.frombuffer(f.read(byte_count), dtype=np.uint8)
        return dequant_q2_k(raw, dims)
    else:
        return None  # Other quant types not yet implemented
