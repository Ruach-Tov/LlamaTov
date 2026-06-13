# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""conftest.py — Path setup for bpd/tests/ test suite.

Adds bpd/lib/ to sys.path so test modules can import curriculum_log,
kernel_emit_bridge, cpu_references, etc. without per-file path
manipulation.

Per medayek's substrate-honest pattern of "path setup in ONE place,
not scattered through every test file."

Author: metayen 2026-05-18
"""

import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_BPD_DIR = os.path.dirname(_TESTS_DIR)
_LIB_DIR = os.path.join(_BPD_DIR, 'lib')

# Make bpd/lib/ importable as top-level package directory.
if _LIB_DIR not in sys.path:
    sys.path.insert(0, _LIB_DIR)

# Make bpd/tests/ importable (for _matrix_test_helpers, etc.).
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
