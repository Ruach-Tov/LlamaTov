#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""pt_to_weights.py v2 — Export weights with BN folded into Conv.

BN folding eliminates the BatchNorm ops entirely:
  w_folded = w * (gamma / sqrt(var + eps))
  b_folded = beta - mean * gamma / sqrt(var + eps)

The folded conv includes a bias term (original conv has no bias).
ggml_conv_2d doesn't support bias directly, so we emit the bias
as a separate ggml_add operation.
"""
import sys, struct, re
import numpy as np
sys.path.insert(0, "/tmp/yolov5")
import torch


def pt_key_to_c_name(key):
    parts = key.split('.')
    result = []
    i = 0
    while i < len(parts):
        p = parts[i]
        if p == 'm' and i + 1 < len(parts) and parts[i + 1].isdigit():
            result.append('bot' + parts[i + 1])
            i += 2
        elif p == 'num_batches_tracked':
            return None
        else:
            result.append(p)
            i += 1
    return '_'.join(result)


def fold_bn(conv_weight, bn_weight, bn_bias, bn_mean, bn_var, eps=0.001):
    """Fold BatchNorm into Conv weight and create bias.
    
    Returns (folded_weight, folded_bias).
    """
    # gamma / sqrt(var + eps)  — shape [Cout]
    scale = bn_weight / np.sqrt(bn_var + eps)
    
    # Reshape scale for broadcasting: [Cout, 1, 1, 1] for OIHW
    scale_shape = [conv_weight.shape[0]] + [1] * (conv_weight.ndim - 1)
    
    # w_folded = w * scale
    w_folded = conv_weight * scale.reshape(scale_shape)
    
    # b_folded = beta - mean * scale
    b_folded = bn_bias - bn_mean * scale
    
    return w_folded.astype(np.float32), b_folded.astype(np.float32)


def export_weights(pt_path, bin_path):
    ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
    sd = ckpt['model'].float().state_dict()
    
    # Group conv+bn pairs
    # Pattern: model.N.conv.weight + model.N.bn.{weight,bias,running_mean,running_var}
    # Or: model.N.cv1.conv.weight + model.N.cv1.bn.*
    
    conv_keys = sorted([k for k in sd if k.endswith('.conv.weight')])
    
    tensors = []  # (c_name, numpy_array)
    folded_convs = set()
    
    for conv_key in conv_keys:
        # Find matching BN
        prefix = conv_key.replace('.conv.weight', '')
        bn_w = prefix + '.bn.weight'
        bn_b = prefix + '.bn.bias'
        bn_m = prefix + '.bn.running_mean'
        bn_v = prefix + '.bn.running_var'
        
        if all(k in sd for k in [bn_w, bn_b, bn_m, bn_v]):
            # Fold BN into conv
            w_folded, b_folded = fold_bn(
                sd[conv_key].numpy(),
                sd[bn_w].numpy(),
                sd[bn_b].numpy(),
                sd[bn_m].numpy(),
                sd[bn_v].numpy(),
                eps=0.001
            )
            c_name = pt_key_to_c_name(conv_key)
            c_bias = c_name.replace('_weight', '_bias')
            tensors.append((c_name, w_folded))
            tensors.append((c_bias, b_folded))
            folded_convs.add(prefix)
        else:
            # No BN — just emit the conv weight directly
            c_name = pt_key_to_c_name(conv_key)
            if c_name:
                tensors.append((c_name, sd[conv_key].numpy()))
    
    # Also emit non-conv/non-bn tensors (detect heads have conv without BN)
    for key in sorted(sd.keys()):
        c_name = pt_key_to_c_name(key)
        if c_name is None:
            continue
        # Skip already-handled conv+bn
        prefix = key.rsplit('.', 2)[0] if '.conv.' in key or '.bn.' in key else ''
        if prefix in folded_convs:
            continue
        # Skip BN params (folded into conv)
        if '.bn.' in key and any(key.startswith(fc) for fc in folded_convs):
            continue
        tensors.append((c_name, sd[key].numpy().astype(np.float32)))
    
    # Remove duplicates
    seen = set()
    unique_tensors = []
    for name, data in tensors:
        if name not in seen:
            unique_tensors.append((name, data))
            seen.add(name)
    tensors = unique_tensors
    
    print("Exporting %d tensors (%d with BN folded)" % (len(tensors), len(folded_convs)))
    
    with open(bin_path, 'wb') as f:
        f.write(struct.pack('i', len(tensors)))
        for c_name, data in tensors:
            name_bytes = c_name.encode('utf-8') + b'\x00'
            f.write(struct.pack('i', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('i', len(data.shape)))
            for s in data.shape:
                f.write(struct.pack('i', s))
            f.write(data.tobytes())
    
    total = sum(d.nbytes for _, d in tensors)
    print("Written to %s (%.1f MB)" % (bin_path, total / 1e6))
    
    # Print folded convs
    for prefix in sorted(folded_convs):
        w = sd[prefix + '.conv.weight']
        print("  Folded: %s [%s] + BN → conv+bias" % (prefix, list(w.shape)))


if __name__ == '__main__':
    pt_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/yolo_canonical/yolov5n.pt'
    bin_path = sys.argv[2] if len(sys.argv) > 2 else '/tmp/yolov5n_weights_folded.bin'
    export_weights(pt_path, bin_path)
