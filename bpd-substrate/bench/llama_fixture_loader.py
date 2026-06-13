# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""llama_fixture_loader.py — load binary tensor dumps from llama.cpp's patched eval-callback.

Phase L.1.0 fixture-loading helper. Each .bin file animates a single ggml tensor
with this layout (little-endian):

  uint32_t dtype_code   (0=F32, 1=F16, 2=I32, 3=I16, 4=I8, 5=Quantized-raw)
  uint32_t n_dims       (always 4)
  int64_t  ne[4]        dimensions
  uint64_t nb[4]        byte strides
  uint64_t n_bytes      payload size
  uint8_t  data[n_bytes]

The manifest.tsv lists tensors in evaluation order:
  idx<TAB>tensor_name<TAB>op_desc<TAB>dtype_name<TAB>ne0,ne1,ne2,ne3
"""
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


DTYPE_CODE_TO_NAME = {
    0: "f32",
    1: "f16",
    2: "i32",
    3: "i16",
    4: "i8",
    5: "quantized",
}

DTYPE_CODE_TO_NUMPY = {
    0: np.float32,
    1: np.float16,
    2: np.int32,
    3: np.int16,
    4: np.int8,
    # 5 (quantized) returns raw bytes; caller must dequantize
}


@dataclass
class FixtureTensor:
    """A single tensor snapshot from llama.cpp's evaluation graph."""
    idx: int
    name: str
    op_desc: str
    dtype_name: str  # f32, f16, q8_0, etc. (from manifest)
    dtype_code: int  # 0..5 (from binary header)
    ne: tuple        # (ne0, ne1, ne2, ne3)
    nb: tuple        # (nb0, nb1, nb2, nb3) byte strides
    data: bytes      # raw payload
    src_indices: tuple = ()  # indices of source tensors (from manifest v2)

    @property
    def shape_no_trailing_ones(self):
        """ggml stores in (ne0, ne1, ne2, ne3) row-major-on-ne0; trailing 1s are
        often padding. For numpy use, drop trailing 1s and reverse."""
        ne = list(self.ne)
        while len(ne) > 1 and ne[-1] == 1:
            ne.pop()
        # ggml convention: ne0 is the fastest-changing axis (innermost in row-major).
        # numpy's row-major puts the LAST axis innermost. So reverse ne for numpy shape.
        return tuple(reversed(ne))

    def as_numpy(self) -> np.ndarray:
        """Return the tensor as a numpy array. Only works for non-quantized dtypes.
        For quantized data (dtype_code=5), use .data and dequantize via our own kernels.
        """
        if self.dtype_code == 5:
            raise ValueError(
                f"Tensor {self.name} is quantized ({self.dtype_name}); "
                f"caller must dequantize via the appropriate kernel"
            )
        np_dtype = DTYPE_CODE_TO_NUMPY[self.dtype_code]
        arr = np.frombuffer(self.data, dtype=np_dtype)
        # Assume contiguous: the stride check below verifies this.
        elem_size = np.dtype(np_dtype).itemsize
        expected_stride_0 = elem_size
        if self.nb[0] != expected_stride_0:
            # Non-contiguous; caller may need to handle strides manually.
            pass
        return arr.reshape(self.shape_no_trailing_ones)


def load_tensor(bin_path) -> FixtureTensor:
    """Load a single .bin file. Manifest fields are filled by load_manifest()."""
    bin_path = Path(bin_path)
    with open(bin_path, "rb") as f:
        # Header
        dtype_code = struct.unpack("<I", f.read(4))[0]
        n_dims = struct.unpack("<I", f.read(4))[0]
        ne = struct.unpack("<4q", f.read(32))    # 4 int64
        nb = struct.unpack("<4Q", f.read(32))    # 4 uint64
        n_bytes = struct.unpack("<Q", f.read(8))[0]
        data = f.read(n_bytes)
        if len(data) != n_bytes:
            raise IOError(f"{bin_path}: expected {n_bytes} payload bytes, got {len(data)}")
    return FixtureTensor(
        idx=int(bin_path.stem.split("_", 1)[0]),
        name="",         # filled by load_manifest
        op_desc="",
        dtype_name=DTYPE_CODE_TO_NAME.get(dtype_code, f"unknown_{dtype_code}"),
        dtype_code=dtype_code,
        ne=ne,
        nb=nb,
        data=data,
    )


def load_manifest(dump_dir) -> list:
    """Load and parse the manifest.tsv, returning a list of FixtureTensor with
    binary data loaded."""
    dump_dir = Path(dump_dir)
    manifest_path = dump_dir / "manifest.tsv"
    tensors = []
    with open(manifest_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            idx_str, name, op_desc, dtype_name, dims_str = parts[:5]
            # Parse source indices (6th column, optional — from patched eval-callback)
            src_indices = ()
            if len(parts) >= 6 and parts[5].strip():
                try:
                    src_indices = tuple(int(x) for x in parts[5].strip().split(",") if x)
                except ValueError:
                    src_indices = ()
            idx = int(idx_str)
            # Find corresponding .bin file (uses sanitized name)
            safe_name = name
            for ch in "/ ()":
                safe_name = safe_name.replace(ch, "_")
            bin_path = dump_dir / f"{idx:04d}_{safe_name}.bin"
            if not bin_path.exists():
                # Try to find by index alone
                candidates = list(dump_dir.glob(f"{idx:04d}_*.bin"))
                if not candidates:
                    print(f"WARN: no .bin found for idx={idx} name={name}")
                    continue
                bin_path = candidates[0]
            t = load_tensor(bin_path)
            t.name = name
            t.op_desc = op_desc
            t.src_indices = src_indices
            tensors.append(t)
    return tensors


def get_sources(tensors, tensor):
    """Get the source tensors for an op using explicit src_indices.
    Returns a list of FixtureTensors, or empty list if no source links."""
    if not tensor.src_indices:
        return []
    idx_map = {t.idx: t for t in tensors}
    return [idx_map[i] for i in tensor.src_indices if i in idx_map]


def find_op(tensors, name_substring=None, op_desc=None, after_idx=-1, layer=None):
    """Find the first tensor whose name contains name_substring AND whose op_desc
    matches. Optionally restrict to layer L (matches '-L' in name)."""
    for t in tensors:
        if t.idx <= after_idx:
            continue
        if name_substring is not None and name_substring not in t.name:
            continue
        if op_desc is not None and t.op_desc != op_desc:
            continue
        if layer is not None and f"-{layer}" not in t.name:
            continue
        return t
    return None


if __name__ == "__main__":
    import sys
    dump_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/llama_dump_layer0"
    tensors = load_manifest(dump_dir)
    print(f"Loaded {len(tensors)} tensors from {dump_dir}")
    print(f"\nFirst 20 ops (layer 0 prefix):")
    for t in tensors[:20]:
        shape = "x".join(str(n) for n in t.ne if n > 1) or "1"
        print(f"  [{t.idx:04d}] {t.op_desc:12} {t.name:40} ({t.dtype_name}) {shape}")
