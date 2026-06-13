# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""_matrix_test_helpers.py — Shared measurement infrastructure for F2 tests.

Test files describe WHAT the substrate guarantees (assertions + bounds).
This module holds HOW to measure (kernel compilation, ctypes dispatch,
ULP distance, fixture loading). Per medayek's substrate-honest pattern
of separating measurement from assertion.

Tests import from here:
  measure_within_gpu_ulp(kernel, size) -> int
  measure_cross_axis_ulp(kernel, size, shape_suffix='') -> int
  ulp_distance(a, b) -> np.ndarray

The HOW lives in one place. Tests stay concise.

Author: metayen 2026-05-18
Per Heath's directive to elevate medayek's concise test style globally.
Pattern source: medayek's 3fdc9f239 test design.
"""

import ctypes
import os
import tempfile

import numpy as np

# conftest.py has already added bpd/lib/ and bpd/tests/ to sys.path.
# This module just needs the fixtures directory location.
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_DIR = os.path.join(_TESTS_DIR, 'fixtures')


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-element ULP distance via uint32 view (handles signed zero)."""
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    sign_flip = np.int64(0x80000000)
    ai = np.where(ai < 0, sign_flip - ai, ai)
    bi = np.where(bi < 0, sign_flip - bi, bi)
    return np.abs(ai - bi)


def _dispatch_via_ctypes(op_kind: str, x: np.ndarray, tmpdir: str) -> np.ndarray:
    """Emit kernel + launcher, compile to .so, dispatch from Python. Cell [4]."""
    import kernel_emit_bridge as keb
    from test_kernelbench_l1_semantic import compile_kernel_to_so

    combined, _kn, launcher_name = keb.emit_kernel_with_launcher('activation', op_kind)
    so_path = compile_kernel_to_so(combined, op_kind, tmpdir)
    lib = ctypes.CDLL(so_path)
    launcher = getattr(lib, launcher_name)
    launcher.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    launcher.restype = ctypes.c_int

    x_c = np.ascontiguousarray(x, dtype=np.float32)
    y_c = np.zeros_like(x_c)
    status = launcher(
        x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        y_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(x_c.shape[0]),
    )
    if status != 0:
        raise RuntimeError(f'Launcher {launcher_name} returned status {status}')
    return y_c


def measure_within_gpu_ulp(kernel: str, size: int):
    """Measure max ULP cell [2] vs cell [4]. Returns int or skips test.

    Returns 0 for bit-identical. Skips (via pytest) if fixtures missing.
    """
    import pytest

    input_path = os.path.join(FIXTURES_DIR, f'{kernel}_{size}.npy')
    cell2_path = f'/tmp/{kernel}_{size}_cell2_c_gpu.npy'
    if not os.path.exists(input_path):
        pytest.skip(f'No input fixture: {input_path}')
    if not os.path.exists(cell2_path):
        pytest.skip(f'No cell [2] output: {cell2_path}')

    x = np.load(input_path)
    cell2 = np.load(cell2_path)
    with tempfile.TemporaryDirectory() as tmpdir:
        cell4 = _dispatch_via_ctypes(kernel, x, tmpdir)
    return int(ulp_distance(cell2, cell4).max())


def measure_cross_axis_ulp(kernel: str, size, shape_suffix: str = ''):
    """Measure max ULP cell [2] vs cell [3]. Returns int or skips test.

    For activations (size is int): fixtures at <kernel>_<size>_*.npy
    For reductions (size is None, shape_suffix='_8x16'): fixtures at <kernel><shape>_*.npy
    """
    import pytest

    if shape_suffix:
        cell3_path = os.path.join(FIXTURES_DIR, f'{kernel}{shape_suffix}_cell3_python_cpu.npy')
        cell2_path = f'/tmp/{kernel}_cell2_c_gpu.npy'
    else:
        cell3_path = os.path.join(FIXTURES_DIR, f'{kernel}_{size}_cell3_python_cpu.npy')
        cell2_path = f'/tmp/{kernel}_{size}_cell2_c_gpu.npy'

    if not os.path.exists(cell3_path):
        pytest.skip(f'No cell [3] fixture: {cell3_path}')
    if not os.path.exists(cell2_path):
        pytest.skip(f'No cell [2] output: {cell2_path}')

    cell2 = np.load(cell2_path)
    cell3 = np.load(cell3_path)
    return int(ulp_distance(cell2, cell3).max())
