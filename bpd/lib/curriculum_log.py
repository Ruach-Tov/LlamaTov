# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""curriculum_log.py — Substrate evolution tracking for the matrix harness.

Per medayek's 2026-05-18 00:23 UTC test-coverage design, F2 closure
requires both:
  - Hard invariant (assert max_ULP <= bound, fail on regression)
  - Characterization (log actual max_ULP, track substrate evolution)

This module provides the characterization side. Tests call
log_measurement(...) with the actual empirical observation; the
measurements accumulate in a JSON-lines log at
bpd/tests/ulp_measurements.jsonl.

Each line is one measurement record. Append-only; never edited
in place. Future analysis tools can query the JSON-lines stream
to track how kernel bounds evolved over time.

## Schema (one JSON object per line)

  {
    "timestamp": "2026-05-18T00:55:00Z",
    "kernel": "k_gelu_tanh",
    "size": 1024,
    "cell_pair": "cell2_vs_cell4",  // within-target
                                     // OR "cell2_vs_cell3" (cross-axis)
    "contract": "strict",            // strict | ulp_2 | allclose
    "actual_max_ulp": 0,
    "expected_bound": 0,
    "pass": true,
    "git_commit": "edfefa308...",    // current HEAD
    "host": "tesla_p4_enclave"
  }

When a test runs, ONE record is appended. Substrate's evolution
tracked through the stream of records over time.

## Why this and not just stdout

CI logs are ephemeral. The curriculum_log persists across CI runs
(stored in git? in S3? TBD by medayek's CI integration). A future
analysis tool can answer "what was k_gelu_erf's max ULP last week
vs today?" by replaying the jsonl stream.

Author: metayen 2026-05-18 ~01:00 UTC
Per medayek's hard-invariant + characterization pattern.
"""

import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Optional


# Resolve log path. Lives next to the tests so CI naturally finds it.
# __file__ is bpd/lib/curriculum_log.py; we want bpd/tests/ulp_measurements.jsonl.
_LIB_DIR = os.path.dirname(os.path.abspath(__file__))
_BPD_DIR = os.path.dirname(_LIB_DIR)
LOG_PATH = os.path.join(_BPD_DIR, 'tests', 'ulp_measurements.jsonl')


def _current_git_commit() -> Optional[str]:
    """Best-effort git HEAD hash for measurement provenance.

    Returns short hash if git is available and we're in a repo,
    else None. Never raises; never blocks the test.
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=2,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _host_identifier() -> str:
    """Best-effort host fingerprint for measurement provenance.

    Use HOSTNAME env var if set, else 'unknown'. Substrate-honest:
    don't probe nvidia-smi here (would fail on CPU-only nodes); the
    GPU_REQUIRED marker already separated those.
    """
    return os.environ.get('HOSTNAME', 'unknown')


def log_measurement(
    kernel: str,
    size: Optional[int],
    cell_pair: str,
    contract: str,
    actual_max_ulp: int,
    expected_bound: int,
    extra_fields: Optional[dict] = None,
) -> None:
    """Append one measurement record to the curriculum log.

    Args:
        kernel: kernel name (e.g., 'k_gelu_tanh')
        size: input size (or None for kernels with fixed-shape fixtures)
        cell_pair: 'cell2_vs_cell4' (within-target) or 'cell2_vs_cell3' (cross-axis)
        contract: 'strict' | 'ulp_2' | 'ulp_N' | 'allclose'
        actual_max_ulp: empirical measurement
        expected_bound: bound per substrate (lib/ulp_attribution.pl mirror)
        extra_fields: optional additional fields (e.g., FMA setting, opt level)

    Append-only. Never raises (failures here shouldn't fail tests).
    """
    record = {
        'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'kernel': kernel,
        'size': size,
        'cell_pair': cell_pair,
        'contract': contract,
        'actual_max_ulp': actual_max_ulp,
        'expected_bound': expected_bound,
        'pass': actual_max_ulp <= expected_bound,
        'git_commit': _current_git_commit(),
        'host': _host_identifier(),
    }
    if extra_fields:
        record.update(extra_fields)

    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, 'a') as f:
            f.write(json.dumps(record) + '\n')
    except (OSError, IOError):
        # Substrate-honest: don't fail tests because of log I/O.
        # If the log isn't writable, CI still functions; substrate
        # evolution tracking just temporarily stops accumulating.
        pass
