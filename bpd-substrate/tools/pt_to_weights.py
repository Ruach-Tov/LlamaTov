#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""pt_to_weights.py — Export PyTorch weights to a flat binary file for C loading.

Reads a PyTorch .pt checkpoint and writes a .bin file with:
  Header: int32 num_tensors
  Per tensor:
    int32 name_length
    char[name_length] name (C struct field name, null-terminated)
    int32 ndims
    int32[ndims] shape
    float32[...] data

Usage:
    python pt_to_weights.py yolov5n.pt weights.bin
"""
import sys, struct, numpy as np
sys.path.insert(0, "/tmp/yolov5")
import torch


def pt_key_to_c_name(key):
    """Convert PyTorch state_dict key to C struct field name.
    
    model.0.conv.weight → model_0_conv_weight
    model.2.m.0.cv1.bn.bias → model_2_bot0_cv1_bn_bias
    """
    parts = key.split('.')
    result = []
    i = 0
    while i < len(parts):
        p = parts[i]
        if p == 'm' and i + 1 < len(parts) and parts[i + 1].isdigit():
            # m.0 → bot0 (matching pytorch_to_prolog.py naming)
            result.append('bot' + parts[i + 1])
            i += 2
        elif p == 'num_batches_tracked':
            i += 1  # skip — not needed for inference
            return None
        else:
            result.append(p)
            i += 1
    return '_'.join(result)


def export_weights(pt_path, bin_path):
    ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
    sd = ckpt['model'].float().state_dict()
    
    # Filter and map
    tensors = []
    for key in sorted(sd.keys()):
        c_name = pt_key_to_c_name(key)
        if c_name is None:
            continue  # skip num_batches_tracked
        t = sd[key].numpy().astype(np.float32)
        tensors.append((c_name, t))
    
    print(f"Exporting {len(tensors)} tensors to {bin_path}")
    
    with open(bin_path, 'wb') as f:
        # Header
        f.write(struct.pack('i', len(tensors)))
        
        total_bytes = 4  # header
        for c_name, data in tensors:
            name_bytes = c_name.encode('utf-8') + b'\x00'
            ndims = len(data.shape)
            shape = list(data.shape)
            
            f.write(struct.pack('i', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('i', ndims))
            for s in shape:
                f.write(struct.pack('i', s))
            f.write(data.tobytes())
            
            tensor_bytes = 4 + len(name_bytes) + 4 + ndims * 4 + data.nbytes
            total_bytes += tensor_bytes
            
            print(f"  {c_name:45s} {str(list(data.shape)):20s} {data.nbytes:8d} bytes")
    
    print(f"\nTotal: {total_bytes:,d} bytes ({total_bytes/1e6:.1f} MB)")


if __name__ == '__main__':
    pt_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/yolo_canonical/yolov5n.pt'
    bin_path = sys.argv[2] if len(sys.argv) > 2 else '/tmp/yolov5n_weights.bin'
    export_weights(pt_path, bin_path)
