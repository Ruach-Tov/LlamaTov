# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
import sys, numpy as np, torch
sys.path.insert(0,_BPD); sys.path.insert(0,_os.path.join(_BPD, "lib"))
import dev_residency as dr
import os as _os, sys as _sys
def _bpd_root(_p=_os.path.dirname(_os.path.abspath(__file__))):
    while _p != '/' and _os.path.basename(_p) != 'bpd':
        _p = _os.path.dirname(_p)
    return _p if _os.path.basename(_p) == 'bpd' else _os.path.dirname(_os.path.abspath(__file__))
_BPD = _bpd_root()

torch.manual_seed(0)
nh,nkv,hd,MAXT = 14,2,64,2048
scale = hd**-0.5
q = torch.randn(nh*hd)*0.3
qd = dr.DevTensor.from_host(q.numpy())

def build_cache(L, poison=False):
    cache = dr.DeviceKVCache(MAXT, nkv, hd)
    for t in range(L):
        k = torch.randn(nkv*hd)*0.3; v = torch.randn(nkv*hd)*0.3
        kd=dr.DevTensor.from_host(k.numpy()); vd=dr.DevTensor.from_host(v.numpy())
        cache.append(kd.ptr, vd.ptr, 1); kd.free(); vd.free()
    if poison:
        # write NaN/garbage into positions [L, MAXT) directly in the device buffers
        garbage = (np.full(nkv*hd, np.nan, np.float32))
        big = (np.full(nkv*hd, 1e30, np.float32))
        for t in range(L, min(L+50, MAXT)):  # poison the next 50 masked positions
            payload = garbage if (t%2==0) else big
            gd = dr.DevTensor.from_host(payload)
            off = t * cache.width * cache.elem_bytes
            import ctypes
            dr.cu.cuMemcpyDtoD_v2(ctypes.c_void_p(cache.k_ptr.value+off), gd.ptr, payload.nbytes)
            dr.cu.cuMemcpyDtoD_v2(ctypes.c_void_p(cache.v_ptr.value+off), gd.ptr, payload.nbytes)
            gd.free()
    return cache

print("=== POISON TEST (Bocher's gate axis): masked region must NOT leak ===", flush=True)
print("=== LENGTH SWEEP: L in {1, 2, 7, MAXT-1, MAXT} ===", flush=True)
worst=0.0; allok=True
for L in [1, 2, 7, MAXT-1, MAXT]:
    torch.manual_seed(L)  # same cache contents clean vs poisoned
    clean = build_cache(L, poison=False)
    lp=clean._ensure_len_ptr()
    out_clean = torch.from_numpy(dr.attn_decode_from_cache_masked(qd.ptr, clean, nh, scale, lp, MAXT).to_host())
    clean.free()
    torch.manual_seed(L)  # identical valid region
    pois = build_cache(L, poison=(L<MAXT))  # can't poison if L==MAXT (no masked region)
    lp2=pois._ensure_len_ptr()
    out_pois = torch.from_numpy(dr.attn_decode_from_cache_masked(qd.ptr, pois, nh, scale, lp2, MAXT).to_host())
    pois.free()
    d=(out_clean-out_pois).abs().max().item()
    nan = bool(torch.isnan(out_pois).any())
    ok = (d==0.0) and not nan
    allok = allok and ok
    print(f"  L={L:5d}: clean-vs-poisoned max_abs={d:.3e}  has_nan={nan}  {'OK' if ok else 'LEAK!'}", flush=True)
print(f">>> POISON+SWEEP: {'ALL PASS — mask structurally severs masked lanes' if allok else 'MASK LEAK DETECTED'}", flush=True)
