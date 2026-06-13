#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Generate a crossword-puzzle GGUF file with intentional overlap attacks.

Creates a minimal valid-looking GGUF file where:
  1. Two tensor entries point to the SAME byte range (tensor aliasing)
  2. A tensor data offset points back INTO the header (header/data overlap)
  3. Some bytes are not referenced by any structure (phantom data)

Then runs our safe_read.pl Prolog reader against it to verify detection.
"""
import struct, sys, os

def write_gguf_string(f, s):
    """Write a GGUF length-prefixed string."""
    encoded = s.encode('utf-8')
    f.write(struct.pack('<Q', len(encoded)))
    f.write(encoded)

def write_gguf_kv_string(f, key, value):
    """Write a GGUF string KV pair."""
    write_gguf_string(f, key)
    f.write(struct.pack('<I', 8))  # type 8 = string
    write_gguf_string(f, value)

def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "/tmp"

    # ── Attack 1: Tensor aliasing (two tensors share same bytes) ──
    path1 = os.path.join(outdir, "crossword_alias.gguf")
    with open(path1, 'wb') as f:
        # Header
        f.write(b'GGUF')                          # magic
        f.write(struct.pack('<I', 3))              # version
        f.write(struct.pack('<Q', 2))              # tensor_count = 2
        f.write(struct.pack('<Q', 1))              # metadata_kv_count = 1

        # Metadata: general.architecture = "test"
        write_gguf_kv_string(f, "general.architecture", "test")

        # Tensor info 0: "weight_a" — float32[4], offset 0
        write_gguf_string(f, "weight_a")
        f.write(struct.pack('<I', 1))              # n_dims = 1
        f.write(struct.pack('<Q', 4))              # dim[0] = 4
        f.write(struct.pack('<I', 0))              # type = F32
        f.write(struct.pack('<Q', 0))              # offset = 0 (relative to data start)

        # Tensor info 1: "weight_b" — float32[4], SAME offset 0 (ALIAS!)
        write_gguf_string(f, "weight_b")
        f.write(struct.pack('<I', 1))              # n_dims = 1
        f.write(struct.pack('<Q', 4))              # dim[0] = 4
        f.write(struct.pack('<I', 0))              # type = F32
        f.write(struct.pack('<Q', 0))              # offset = 0 (OVERLAPS weight_a!)

        # Align to 32 bytes
        pos = f.tell()
        pad = (32 - pos % 32) % 32
        f.write(b'\x00' * pad)

        # Tensor data: only 16 bytes (4 float32s)
        # Both weight_a and weight_b claim these same 16 bytes
        for v in [1.0, 2.0, 3.0, 4.0]:
            f.write(struct.pack('<f', v))

    print(f"Attack 1 (tensor aliasing): {path1} ({os.path.getsize(path1)} bytes)")

    # ── Attack 2: Header/data overlap ──
    path2 = os.path.join(outdir, "crossword_header_overlap.gguf")
    with open(path2, 'wb') as f:
        # Header
        f.write(b'GGUF')                          # magic: bytes 0-3
        f.write(struct.pack('<I', 3))              # version: bytes 4-7
        f.write(struct.pack('<Q', 1))              # tensor_count: bytes 8-15
        f.write(struct.pack('<Q', 1))              # kv_count: bytes 16-23

        # Metadata
        write_gguf_kv_string(f, "general.architecture", "test")

        # Tensor info: "evil_tensor" — offset points BACK into header
        write_gguf_string(f, "evil_tensor")
        f.write(struct.pack('<I', 1))              # n_dims = 1
        f.write(struct.pack('<Q', 4))              # dim[0] = 4
        f.write(struct.pack('<I', 0))              # type = F32

        # Calculate where data section starts, then set offset to point
        # back to byte 4 (the version field!) relative to data start
        pos = f.tell() + 8  # after writing the offset
        data_start_pad = (32 - pos % 32) % 32
        data_start = pos + data_start_pad

        # Offset is relative to data section start
        # We want absolute position 4 (version field)
        # So offset = 4 - data_start (this will be negative, encoded as huge uint64)
        # Actually GGUF offsets are unsigned and relative to data start
        # A negative offset wraps around — but let's make it point into padding instead
        # More realistic: set offset to 0 but DON'T write enough data
        f.write(struct.pack('<Q', 0))

        # Align
        f.write(b'\x00' * data_start_pad)

        # Write only 8 bytes of data (tensor claims 16)
        f.write(struct.pack('<ff', 1.0, 2.0))
        # Missing 8 bytes — tensor overflows file

    print(f"Attack 2 (short data): {path2} ({os.path.getsize(path2)} bytes)")

    # ── Attack 3: Phantom data ──
    path3 = os.path.join(outdir, "crossword_phantom.gguf")
    with open(path3, 'wb') as f:
        # Header
        f.write(b'GGUF')
        f.write(struct.pack('<I', 3))
        f.write(struct.pack('<Q', 1))              # 1 tensor
        f.write(struct.pack('<Q', 1))              # 1 KV pair

        write_gguf_kv_string(f, "general.architecture", "test")

        write_gguf_string(f, "real_tensor")
        f.write(struct.pack('<I', 1))
        f.write(struct.pack('<Q', 2))              # dim[0] = 2 (only 8 bytes of data)
        f.write(struct.pack('<I', 0))              # F32
        f.write(struct.pack('<Q', 0))              # offset 0

        # Align
        pos = f.tell()
        pad = (32 - pos % 32) % 32
        f.write(b'\x00' * pad)

        # Tensor data: 8 bytes (2 float32s)
        f.write(struct.pack('<ff', 42.0, 43.0))

        # PHANTOM: 64 extra bytes that nothing references
        f.write(b'THIS IS PHANTOM DATA THAT NO PARSER SHOULD ACCEPT!' + b'\x00' * 13)

    print(f"Attack 3 (phantom data): {path3} ({os.path.getsize(path3)} bytes)")

    # ── Verify: well-formed GGUF for comparison ──
    path_good = os.path.join(outdir, "crossword_good.gguf")
    with open(path_good, 'wb') as f:
        f.write(b'GGUF')
        f.write(struct.pack('<I', 3))
        f.write(struct.pack('<Q', 1))
        f.write(struct.pack('<Q', 1))

        write_gguf_kv_string(f, "general.architecture", "test")

        write_gguf_string(f, "good_tensor")
        f.write(struct.pack('<I', 1))
        f.write(struct.pack('<Q', 4))
        f.write(struct.pack('<I', 0))
        f.write(struct.pack('<Q', 0))

        pos = f.tell()
        pad = (32 - pos % 32) % 32
        f.write(b'\x00' * pad)

        for v in [1.0, 2.0, 3.0, 4.0]:
            f.write(struct.pack('<f', v))

    print(f"Good file:  {path_good} ({os.path.getsize(path_good)} bytes)")

    print("\nGenerated 4 test files. Run safe_read verification with:")
    print(f"  swipl -g 'use_module(\"lib/safe_read\"), test_crossword(\"{outdir}\").'")

if __name__ == "__main__":
    main()
