#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Compare our GGUF weight loading against the official gguf-py library.

Loads the same model via both our code and gguf-py, compares weight
values to identify dequantization or layout bugs.

Usage:
  nix-shell -p python312Packages.gguf python312Packages.numpy --run \
    'python3 test_weight_comparison.py /path/to/model.gguf'

Author: medayek (Collective SME, Verification Methodology)
Date: 2026-05-16
"""

import sys
import struct
import numpy as np

def read_gguf_header(f):
    """Read GGUF header and return metadata + tensor info."""
    magic = f.read(4)
    assert magic == b'GGUF', f"Not a GGUF file: {magic}"
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_kv = struct.unpack('<Q', f.read(8))[0]
    return version, n_tensors, n_kv

def dequant_q8_0_block(block_data):
    """Dequantize one Q8_0 block (34 bytes = 1 f16 scale + 32 int8 values)."""
    # Q8_0: 32 int8 values with one f16 scale
    scale_bytes = block_data[:2]
    scale = np.frombuffer(scale_bytes, dtype=np.float16)[0].astype(np.float32)
    quants = np.frombuffer(block_data[2:34], dtype=np.int8).astype(np.float32)
    return quants * scale

def compare_weights_gguf_py(model_path, tensor_name=None, max_tensors=5):
    """Load model via gguf-py and compare against manual dequantization."""
    try:
        from gguf import GGUFReader
    except ImportError:
        print("ERROR: gguf library not available")
        print("Install: nix-shell -p python312Packages.gguf")
        return

    reader = GGUFReader(model_path)

    print(f"Model: {model_path}")
    print(f"Tensors: {len(reader.tensors)}")
    print(f"Architecture: {reader.fields.get('general.architecture', 'unknown')}")
    print()

    # Find interesting tensors
    tensors_to_check = []
    for tensor in reader.tensors:
        name = tensor.name
        if tensor_name and tensor_name not in name:
            continue
        tensors_to_check.append(tensor)
        if len(tensors_to_check) >= max_tensors:
            break

    for tensor in tensors_to_check:
        name = tensor.name
        shape = tuple(tensor.shape)
        dtype = tensor.tensor_type
        n_elements = tensor.n_elements
        data_offset = tensor.data_offset

        print(f"═══ {name} ═══")
        print(f"  Shape: {shape}")
        print(f"  Type: {dtype} (enum={dtype.value if hasattr(dtype, 'value') else dtype})")
        print(f"  Elements: {n_elements}")
        print(f"  Data offset: {data_offset}")

        # Get dequantized data from gguf-py
        try:
            # gguf-py provides the raw data; we need to interpret it
            raw_data = tensor.data
            print(f"  Raw data shape: {raw_data.shape}, dtype: {raw_data.dtype}")

            if hasattr(raw_data, 'astype'):
                # For F32 tensors, data is directly usable
                if raw_data.dtype == np.float32:
                    values = raw_data.flatten()
                elif raw_data.dtype == np.float16:
                    values = raw_data.astype(np.float32).flatten()
                else:
                    # For quantized, interpret as blocks
                    values = raw_data.flatten()

                print(f"  First 8 values: {values[:8]}")
                print(f"  Mean: {np.mean(values[:min(1000, len(values))]):.6f}")
                print(f"  Std:  {np.std(values[:min(1000, len(values))]):.6f}")
                print(f"  Min:  {np.min(values[:min(1000, len(values))]):.6f}")
                print(f"  Max:  {np.max(values[:min(1000, len(values))]):.6f}")

                # Manual Q8_0 dequant for comparison
                if 'Q8_0' in str(dtype):
                    block_size = 34  # 2 bytes scale + 32 bytes quants
                    n_blocks = len(raw_data) // block_size
                    manual_values = []
                    raw_bytes = raw_data.tobytes()
                    for b in range(min(4, n_blocks)):  # First 4 blocks
                        block = raw_bytes[b*block_size:(b+1)*block_size]
                        dequanted = dequant_q8_0_block(block)
                        manual_values.extend(dequanted.tolist())

                    if manual_values:
                        manual_arr = np.array(manual_values[:8])
                        print(f"  Manual dequant first 8: {manual_arr}")
                        if len(values) >= 8:
                            diff = np.abs(values[:8].astype(np.float32) - manual_arr)
                            print(f"  Diff (gguf-py vs manual): {diff}")
                            if np.max(diff) < 1e-6:
                                print(f"  ✅ MATCH")
                            else:
                                print(f"  ❌ MISMATCH — dequant differs!")
        except Exception as e:
            print(f"  Error reading tensor: {e}")

        print()


def dump_tensor_layout(model_path, tensor_name):
    """Dump the raw byte layout of a specific tensor for manual inspection."""
    try:
        from gguf import GGUFReader
    except ImportError:
        print("ERROR: gguf library not available")
        return

    reader = GGUFReader(model_path)

    for tensor in reader.tensors:
        if tensor_name in tensor.name:
            print(f"Tensor: {tensor.name}")
            print(f"Shape: {tuple(tensor.shape)}")
            print(f"Type: {tensor.tensor_type}")
            print(f"Offset: {tensor.data_offset}")

            raw = tensor.data
            print(f"Raw dtype: {raw.dtype}, shape: {raw.shape}")

            # Print first 128 bytes as hex
            raw_bytes = raw.tobytes()[:128]
            print(f"First 128 bytes (hex):")
            for i in range(0, min(128, len(raw_bytes)), 16):
                hex_str = ' '.join(f'{b:02x}' for b in raw_bytes[i:i+16])
                print(f"  {i:04x}: {hex_str}")

            # Interpret first Q8_0 block
            if 'Q8_0' in str(tensor.tensor_type):
                scale = np.frombuffer(raw_bytes[:2], dtype=np.float16)[0]
                quants = np.frombuffer(raw_bytes[2:34], dtype=np.int8)
                print(f"\nFirst Q8_0 block:")
                print(f"  Scale (f16): {scale} = {float(scale)}")
                print(f"  Quants (int8): {quants[:16]}...")
                print(f"  Dequantized: {(quants.astype(np.float32) * float(scale))[:8]}")

            return

    print(f"Tensor '{tensor_name}' not found")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_weight_comparison.py <model.gguf> [tensor_name]")
        print("Example: python3 test_weight_comparison.py model.gguf token_embd")
        sys.exit(1)

    model_path = sys.argv[1]
    tensor_name = sys.argv[2] if len(sys.argv) > 2 else None

    if tensor_name and tensor_name == '--dump':
        # Dump mode: show raw bytes
        dump_name = sys.argv[3] if len(sys.argv) > 3 else 'token_embd'
        dump_tensor_layout(model_path, dump_name)
    else:
        compare_weights_gguf_py(model_path, tensor_name, max_tensors=10)
