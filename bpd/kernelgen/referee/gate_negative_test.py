# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Negative test: the gate must REJECT a fusion that isn't bit-exact. Feed it two outputs that
differ by 1 ULP -> bit_exact must FAIL, tolerance(eps) must PASS. Proves the gate gates."""
import sys, numpy as np
sys.path.insert(0,_os2.path.join(_REPO, "bpd/kernelgen"))
from fusion_gate import compare_outputs
import os as _os, sys as _sys
import os as _os2
_REPO = _os2.environ.get("LLAMATOV_ROOT") or _os2.path.abspath(_os2.path.join(_os2.path.dirname(_os2.path.abspath(__file__)), *[".."]*8))

def _bpd_root(_p=_os.path.dirname(_os.path.abspath(__file__))):
    while _p != '/' and _os.path.basename(_p) != 'bpd':
        _p = _os.path.dirname(_p)
    return _p if _os.path.basename(_p) == 'bpd' else _os.path.dirname(_os.path.abspath(__file__))
_BPD = _bpd_root()

np.random.seed(1)
a=(np.random.randn(1000)*5).astype(np.float32)
# b = a perturbed by 1 ULP on half the elements (simulates an FMA-reorder drift)
b=a.copy(); bb=b.view(np.uint32); bb[::2]+=1  # flip 1 ULP on even indices
r_bit=compare_outputs(a,b,'bit_exact')
r_tol=compare_outputs(a,b,('tolerance',1e-3))
r_tol_tight=compare_outputs(a,b,('tolerance',1e-9))
print(f"1-ULP-drift vs bit_exact:        {r_bit}  -> {'PASS' if r_bit.passed else 'REJECT (correct!)'}", flush=True)
print(f"1-ULP-drift vs tolerance(1e-3):  {r_tol}  -> {'PASS (correct!)' if r_tol.passed else 'REJECT'}", flush=True)
print(f"1-ULP-drift vs tolerance(1e-9):  {r_tol_tight}  -> {'PASS' if r_tol_tight.passed else 'REJECT (correct!)'}", flush=True)
# and identical -> bit_exact passes
r_same=compare_outputs(a,a.copy(),'bit_exact')
print(f"identical vs bit_exact:          {r_same}  -> {'PASS (correct!)' if r_same.passed else 'REJECT'}", flush=True)
ok = (not r_bit.passed) and r_tol.passed and (not r_tol_tight.passed) and r_same.passed
print(f">>> GATE LOGIC CORRECT: {ok}", flush=True)
