# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""gguf_helper.py — thin Python helper that calls the canonical Prolog GGUF
reader (lib/gguf_native_reader.pl) via subprocess to fetch tensor info.

Per Heath: 'why build a python parser, when we have a prolog parser?'
The Prolog reader is the source of truth. This module is a thin animator
that invokes swipl and parses the single-line output.

Returns absolute file offsets so callers can read raw tensor bytes with
np.fromfile(offset=...).
"""
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


# Default location for the Prolog query script relative to repo root
_REPO_ROOT = Path(__file__).parent.parent
DEFAULT_QUERY_SCRIPT = _REPO_ROOT / "tests" / "gguf_query.pl"


@dataclass
class GGUFTensorInfo:
    """Animated tensor info as returned by gguf_query.pl."""
    name: str
    abs_offset: int   # absolute byte offset in the GGUF file
    size: int          # total bytes occupied by the tensor's data
    ggml_type: int    # 0=F32, 1=F16, 8=Q8_0, 12=Q4_K, etc.
    dims: tuple       # shape, as parsed from the comma-separated DIMS field


def query_tensor(gguf_path, tensor_name, query_script=None):
    """Invoke swipl with our gguf_query.pl to fetch tensor info.

    Returns GGUFTensorInfo on success, raises RuntimeError on failure.
    """
    query_script = Path(query_script) if query_script else DEFAULT_QUERY_SCRIPT
    if not query_script.exists():
        raise FileNotFoundError(f"gguf_query.pl not found at {query_script}")

    # Run swipl. Working directory should be the repo root so the script's
    # internal use_module('lib/gguf_native_reader') resolves correctly.
    cmd = [
        "swipl", "-q",
        "-g", f"consult('{query_script}'), gguf_query_main",
        "--",
        str(gguf_path), str(tensor_name),
    ]
    result = subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"gguf_query failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    # Parse the single line: "ABS_OFFSET=<n> SIZE=<n> TYPE=<n> DIMS=<csv>"
    line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if not line.startswith("ABS_OFFSET="):
        raise RuntimeError(f"unexpected gguf_query output: {result.stdout!r}")

    fields = {}
    for token in line.split():
        if "=" in token:
            k, v = token.split("=", 1)
            fields[k] = v

    return GGUFTensorInfo(
        name=tensor_name,
        abs_offset=int(fields["ABS_OFFSET"]),
        size=int(fields["SIZE"]),
        ggml_type=int(fields["TYPE"]),
        dims=tuple(int(d) for d in fields["DIMS"].split(",")),
    )


def read_tensor_bytes(gguf_path, info):
    """Read the raw bytes of a tensor from the GGUF file."""
    import numpy as np
    return np.fromfile(gguf_path, dtype=np.uint8, count=info.size, offset=info.abs_offset)


if __name__ == "__main__":
    import sys
    gguf = sys.argv[1] if len(sys.argv) > 1 else \
        "/tmp/llamatov-data/ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45"
    name = sys.argv[2] if len(sys.argv) > 2 else "blk.0.attn_k.weight"
    info = query_tensor(gguf, name)
    print(f"name={info.name}")
    print(f"abs_offset={info.abs_offset}")
    print(f"size={info.size}")
    print(f"ggml_type={info.ggml_type}")
    print(f"dims={info.dims}")
    # Sanity: read first 34 bytes (one Q8_0 block)
    raw = read_tensor_bytes(gguf, info)
    print(f"raw shape: {raw.shape}, first 34 bytes: {raw[:34].tolist()}")
