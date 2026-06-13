#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Generate malformed GGUF test files for gguf_validate/1 testing.

Each file is a valid-looking GGUF binary with a specific malformation
that gguf_validate/1 must detect and reject.

GGUF format reference:
  Header: magic(4) + version(4) + tensor_count(8) + metadata_kv_count(8)
  Metadata: key_len(8) + key(N) + value_type(4) + value(...)
  Tensor info: name_len(8) + name(N) + n_dims(4) + dims(8*n) + type(4) + offset(8)
  Tensor data: at alignment boundary after all tensor infos

Author: medayek (Collective SME, Verification Methodology)
Date: 2026-05-20
Per mavchin: crossword-puzzle attack files for issue #24.
"""

import struct
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'crossword_attacks')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GGUF constants
GGUF_MAGIC = b'GGUF'
GGUF_VERSION = 3
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_K = 12
ALIGNMENT = 32  # GGUF data alignment


def write_gguf_header(f, version, tensor_count, metadata_kv_count):
    f.write(GGUF_MAGIC)
    f.write(struct.pack('<I', version))
    f.write(struct.pack('<Q', tensor_count))
    f.write(struct.pack('<Q', metadata_kv_count))


def write_metadata_string(f, key, value):
    """Write a string metadata key-value pair."""
    key_bytes = key.encode('utf-8')
    val_bytes = value.encode('utf-8')
    f.write(struct.pack('<Q', len(key_bytes)))
    f.write(key_bytes)
    f.write(struct.pack('<I', 8))  # GGUF_TYPE_STRING = 8
    f.write(struct.pack('<Q', len(val_bytes)))
    f.write(val_bytes)


def write_metadata_uint32(f, key, value):
    """Write a uint32 metadata key-value pair."""
    key_bytes = key.encode('utf-8')
    f.write(struct.pack('<Q', len(key_bytes)))
    f.write(key_bytes)
    f.write(struct.pack('<I', 4))  # GGUF_TYPE_UINT32 = 4
    f.write(struct.pack('<I', value))


def write_tensor_info(f, name, dims, tensor_type, offset):
    """Write tensor info entry."""
    name_bytes = name.encode('utf-8')
    f.write(struct.pack('<Q', len(name_bytes)))
    f.write(name_bytes)
    f.write(struct.pack('<I', len(dims)))
    for d in dims:
        f.write(struct.pack('<Q', d))
    f.write(struct.pack('<I', tensor_type))
    f.write(struct.pack('<Q', offset))


def align_to(pos, alignment=ALIGNMENT):
    """Return next aligned position."""
    remainder = pos % alignment
    if remainder == 0:
        return pos
    return pos + (alignment - remainder)


def pad_to_alignment(f, alignment=ALIGNMENT):
    """Pad file to next alignment boundary."""
    pos = f.tell()
    aligned = align_to(pos, alignment)
    if aligned > pos:
        f.write(b'\x00' * (aligned - pos))


# ═══════════════════════════════════════════════════════════════════════
# Attack 1: Tensor-tensor overlap
# ═══════════════════════════════════════════════════════════════════════

def generate_tensor_overlap():
    """Two tensors with overlapping data regions."""
    path = os.path.join(OUTPUT_DIR, 'tensor_tensor_overlap.gguf')
    with open(path, 'wb') as f:
        write_gguf_header(f, GGUF_VERSION, 2, 1)
        write_metadata_string(f, 'general.architecture', 'llama')

        # Tensor A: 4 floats = 16 bytes, at offset 0
        write_tensor_info(f, 'tensor_a', [4], GGML_TYPE_F32, 0)
        # Tensor B: 4 floats = 16 bytes, at offset 8 (OVERLAPS with A!)
        write_tensor_info(f, 'tensor_b', [4], GGML_TYPE_F32, 8)

        pad_to_alignment(f)

        # Write 32 bytes of tensor data (but they overlap at bytes 8-15)
        f.write(struct.pack('<4f', 1.0, 2.0, 3.0, 4.0))  # tensor_a
        f.write(struct.pack('<4f', 5.0, 6.0, 7.0, 8.0))  # tensor_b (starts at +8 from data start... but offset says 8)

    print(f"  Generated: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════
# Attack 2: Tensor-header overlap
# ═══════════════════════════════════════════════════════════════════════

def generate_tensor_header_overlap():
    """Tensor whose data offset points back into header/metadata section."""
    path = os.path.join(OUTPUT_DIR, 'tensor_header_overlap.gguf')
    with open(path, 'wb') as f:
        write_gguf_header(f, GGUF_VERSION, 1, 1)
        write_metadata_string(f, 'general.architecture', 'llama')

        # The tensor's offset is relative to the data section start.
        # But we set a NEGATIVE-equivalent offset by using a very large number
        # that wraps around, or we set offset=0 but place the data section
        # overlapping with headers.
        #
        # Simpler attack: set offset to 0 but make the tensor so large
        # it would extend backwards into the header when read from data_start.
        # Actually, the simplest: the data_start IS the alignment point after
        # tensor infos. If we set offset to a value that, when added to
        # data_start, points BEFORE data_start (i.e., into the header).
        # GGUF offsets are relative to data_start, so offset=0 is fine.
        # We need: data_start + offset < header_end
        # Since offset is uint64, we can't go negative. But we CAN make
        # the tensor info claim a huge size that would extend past EOF.

        # Alternative: just use a legitimate offset that happens to be
        # within the metadata region when interpreted as absolute file offset
        # (a reader bug would interpret offset as absolute, not relative)
        write_tensor_info(f, 'evil_tensor', [1024], GGML_TYPE_F32, 0)

        # DON'T pad to alignment — the "data section" starts immediately
        # Write very little actual data (tensor claims 1024*4=4096 bytes but we write less)
        f.write(b'\x00' * 64)  # Only 64 bytes, tensor claims 4096

    print(f"  Generated: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════
# Attack 3: Wrong quantization version
# ═══════════════════════════════════════════════════════════════════════

def generate_wrong_quant_version():
    """Metadata says quant version 2 but tensors use Q4_K (needs version 3+)."""
    path = os.path.join(OUTPUT_DIR, 'wrong_quant_version.gguf')
    with open(path, 'wb') as f:
        write_gguf_header(f, GGUF_VERSION, 1, 2)
        write_metadata_string(f, 'general.architecture', 'llama')
        write_metadata_uint32(f, 'general.quantization_version', 2)  # Says v2

        # But tensor uses Q4_K which requires v3
        write_tensor_info(f, 'blk.0.attn_q.weight', [4096, 4096], GGML_TYPE_Q4_K, 0)

        pad_to_alignment(f)

        # Write minimal data (Q4_K block: 144 bytes per 256 elements)
        # 4096*4096 = 16M elements, needs ~9MB. Just write a small amount.
        f.write(b'\x00' * 1024)

    print(f"  Generated: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════
# Attack 4: Mixed quantization in same layer
# ═══════════════════════════════════════════════════════════════════════

def generate_mixed_quant_layer():
    """Same attention layer has Q4_0 weights and Q4_K biases."""
    path = os.path.join(OUTPUT_DIR, 'mixed_quant_layer.gguf')
    with open(path, 'wb') as f:
        write_gguf_header(f, GGUF_VERSION, 2, 1)
        write_metadata_string(f, 'general.architecture', 'llama')

        # Weight uses Q4_0
        write_tensor_info(f, 'blk.0.attn_q.weight', [4096, 4096], GGML_TYPE_Q4_0, 0)
        # Bias uses Q4_K (inconsistent with weight!)
        write_tensor_info(f, 'blk.0.attn_q.bias', [4096], GGML_TYPE_Q4_K, 1024)

        pad_to_alignment(f)
        f.write(b'\x00' * 2048)

    print(f"  Generated: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════
# Attack 5: Missing rope parameters
# ═══════════════════════════════════════════════════════════════════════

def generate_missing_rope():
    """Architecture is 'llama' but rope parameters are missing."""
    path = os.path.join(OUTPUT_DIR, 'missing_rope_params.gguf')
    with open(path, 'wb') as f:
        write_gguf_header(f, GGUF_VERSION, 1, 2)
        # Says llama architecture (requires rope)
        write_metadata_string(f, 'general.architecture', 'llama')
        # Has some metadata but NO rope.freq_base, NO rope.scaling.type
        write_metadata_uint32(f, 'llama.context_length', 4096)

        write_tensor_info(f, 'blk.0.attn_q.weight', [2048, 2048], GGML_TYPE_F16, 0)

        pad_to_alignment(f)
        f.write(b'\x00' * 512)

    print(f"  Generated: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════
# Attack 6: String length exceeds containing record
# ═══════════════════════════════════════════════════════════════════════

def generate_string_length_overflow():
    """Metadata string claims 10000 bytes but file has far less remaining.
    
    GGUF string format: length(uint64) + data(length bytes)
    Attack: length field says 10000 but only 5 bytes of actual string
    follow before the next record or EOF.
    
    A naive reader would:
      - Read length = 10000
      - Attempt to read 10000 bytes
      - Read past the metadata section into tensor info or EOF
      - Interpret garbage as string content
    """
    path = os.path.join(OUTPUT_DIR, 'string_length_overflow.gguf')
    with open(path, 'wb') as f:
        write_gguf_header(f, GGUF_VERSION, 1, 2)

        # First KV: normal architecture string
        write_metadata_string(f, 'general.architecture', 'llama')

        # Second KV: MALFORMED string — length claims 10000 bytes
        key = 'general.name'
        key_bytes = key.encode('utf-8')
        f.write(struct.pack('<Q', len(key_bytes)))
        f.write(key_bytes)
        f.write(struct.pack('<I', 8))  # GGUF_TYPE_STRING = 8
        # HERE IS THE ATTACK: claim 10000 bytes but write only 5
        f.write(struct.pack('<Q', 10000))  # length = 10000
        f.write(b'hello')                  # only 5 bytes of actual data

        # Tensor info follows immediately — a naive reader would
        # read 9995 bytes of this as "string content"
        write_tensor_info(f, 'blk.0.weight', [256], GGML_TYPE_F32, 0)

        pad_to_alignment(f)
        f.write(b'\x00' * 1024)

    print(f"  Generated: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════
# Attack 7: String length overflow into tensor data
# ═══════════════════════════════════════════════════════════════════════

def generate_string_into_tensor_data():
    """Metadata key string length extends past all metadata into tensor data.
    
    More subtle: the string length is carefully chosen to be exactly
    the right size to consume the remaining metadata + tensor info +
    alignment padding, landing in the tensor data section. A reader
    that doesn't bounds-check would parse tensor data as a string.
    """
    path = os.path.join(OUTPUT_DIR, 'string_into_tensor_data.gguf')
    with open(path, 'wb') as f:
        write_gguf_header(f, GGUF_VERSION, 1, 1)

        # Malformed: the VALUE string length extends past EOF
        key = 'general.architecture'
        key_bytes = key.encode('utf-8')
        f.write(struct.pack('<Q', len(key_bytes)))
        f.write(key_bytes)
        f.write(struct.pack('<I', 8))  # GGUF_TYPE_STRING
        # Claim string is 999999 bytes — file is ~200 bytes total
        f.write(struct.pack('<Q', 999999))
        f.write(b'llama')  # only 5 bytes

        # Minimal tensor info
        write_tensor_info(f, 'w', [4], GGML_TYPE_F32, 0)

        pad_to_alignment(f)
        f.write(struct.pack('<4f', 1.0, 2.0, 3.0, 4.0))

    print(f"  Generated: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════
# Attack 8: Key length overflow
# ═══════════════════════════════════════════════════════════════════════

def generate_key_length_overflow():
    """Metadata KEY length exceeds remaining file.
    
    The key name length is stored as uint64. If it claims to be
    larger than the remaining file, a naive reader reads past EOF.
    """
    path = os.path.join(OUTPUT_DIR, 'key_length_overflow.gguf')
    with open(path, 'wb') as f:
        write_gguf_header(f, GGUF_VERSION, 0, 1)

        # Key length claims 50000 bytes but only 10 bytes follow
        f.write(struct.pack('<Q', 50000))  # key_len = 50000
        f.write(b'short_key\x00')         # only 10 bytes
        f.write(struct.pack('<I', 8))      # type = string
        f.write(struct.pack('<Q', 5))      # value_len = 5
        f.write(b'llama')

    print(f"  Generated: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Generating crossword-puzzle GGUF attack files:")
    print()
    generate_tensor_overlap()
    generate_tensor_header_overlap()
    generate_wrong_quant_version()
    generate_mixed_quant_layer()
    generate_missing_rope()
    generate_string_length_overflow()
    generate_string_into_tensor_data()
    generate_key_length_overflow()
    print()
    print(f"All files in: {OUTPUT_DIR}")
    print("Each file has ONE specific malformation for gguf_validate/1 to catch.")
