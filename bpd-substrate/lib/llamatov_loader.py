# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""LlamaTov GGUF tensor loader — minimal helper for janus_swi.

Provides load_tensor() that reads bytes from a GGUF file at a known
offset, reshapes, and returns a torch tensor. Keeps the tensor as a
single Python object (avoids janus_swi's small-array-to-list conversion).
"""

import numpy as np
import torch


def load_tensor(path, offset, count, dtype_str, shape):
    """Read `count` elements of `dtype_str` starting at `offset` bytes.

    Reshape to `shape` (tuple), convert to torch tensor, return.
    """
    arr = np.fromfile(path, dtype=dtype_str, count=count, offset=offset)
    arr = arr.reshape(shape)
    # copy() detaches from the memmap so the tensor owns its data
    return torch.from_numpy(arr.copy())


def load_tensor_f16(path, offset, count, shape):
    """Convenience: load FP16 and upcast to FP32 (P4 doesn't support FP16 ops well)."""
    arr = np.fromfile(path, dtype='float16', count=count, offset=offset)
    arr = arr.astype('float32').reshape(shape)
    return torch.from_numpy(arr.copy())


def matmul(a, b):
    """torch.matmul wrapper."""
    return torch.matmul(a, b)


def add(a, b):
    """torch.add wrapper."""
    return torch.add(a, b)


def silu(x):
    """SiLU activation."""
    return torch.nn.functional.silu(x)


def softmax(x, dim=-1):
    """Softmax over the given dim."""
    return torch.nn.functional.softmax(x, dim=dim)


def tensor_shape(t):
    """Return shape as a tuple."""
    return tuple(t.shape)


def tensor_sum(t):
    """Return the sum as a Python float (for verification)."""
    return float(t.sum().item())
