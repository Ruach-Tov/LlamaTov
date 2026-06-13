#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Profile YOLOv5n layer-by-layer through BPD — find optimization targets."""
import sys, os; sys.path.insert(0, "/tmp/yolov5")
import torch, numpy as np, ctypes, time
torch.set_num_threads(1); torch.backends.mkldnn.enabled = False

lib = ctypes.CDLL(os.environ.get("BPD_CPU_SO", "build/bpd_cpu.so"))
c_void=ctypes.c_void_p; c_int=ctypes.c_int; c_float=ctypes.c_float
lib.bpd_conv2d_full_cpu.argtypes=[c_void]*4+[c_int]*14; lib.bpd_conv2d_full_cpu.restype=None
lib.bpd_silu_cpu.argtypes=[c_void]*2+[c_int]; lib.bpd_silu_cpu.restype=None
lib.bpd_batchnorm_cpu_affine_fused.argtypes=[c_void]*8+[c_int]*3+[c_float]; lib.bpd_batchnorm_cpu_affine_fused.restype=None
lib.bpd_maxpool2d_cpu.argtypes=[c_void]*2+[c_int]*8; lib.bpd_maxpool2d_cpu.restype=None
lib.bpd_add_f32_cpu.argtypes=[c_void]*3+[c_int]; lib.bpd_add_f32_cpu.restype=None
lib.bpd_concat_cpu.argtypes=[c_void]*3+[c_int]*5; lib.bpd_concat_cpu.restype=None
lib.bpd_concat4_cpu.argtypes=[c_void]*5+[c_int]*7; lib.bpd_concat4_cpu.restype=None
lib.bpd_upsample_nearest2d_cpu.argtypes=[c_void]*2+[c_int]*4; lib.bpd_upsample_nearest2d_cpu.restype=None

# Timing wrapper
timings = {}
def timed(name, fn):
    t0 = time.perf_counter()
    result = fn()
    dt = (time.perf_counter() - t0) * 1000
    timings.setdefault(name, []).append(dt)
    return result

def bpd_conv(x, cm):
    w=np.ascontiguousarray(cm.weight.data.numpy(),dtype=np.float32)
    b=np.zeros(cm.out_channels,dtype=np.float32)
    N,Ci,H,W=x.shape; kH,kW=cm.kernel_size; sh,sw=cm.stride; ph,pw=cm.padding
    Ho=(H+2*ph-kH)//sh+1; Wo=(W+2*pw-kW)//sw+1
    co=np.zeros((N,cm.out_channels,Ho,Wo),dtype=np.float32)
    lib.bpd_conv2d_full_cpu(np.ascontiguousarray(x).ctypes.data,w.ctypes.data,b.ctypes.data,co.ctypes.data,
        c_int(N),c_int(Ci),c_int(H),c_int(W),c_int(cm.out_channels),c_int(kH),c_int(kW),
        c_int(sh),c_int(sw),c_int(ph),c_int(pw),c_int(1),c_int(1),c_int(cm.groups))
    return co

def bpd_bn(x, bm):
    g=bm.weight.data.numpy().astype(np.float32); bt=bm.bias.data.numpy().astype(np.float32)
    mn=bm.running_mean.data.numpy().astype(np.float32); vr=bm.running_var.data.numpy().astype(np.float32)
    N=x.shape[0]; C=bm.num_features; HW=int(np.prod(x.shape[2:]))
    bo=np.zeros_like(x); sb=np.zeros(C,dtype=np.float32); ob=np.zeros(C,dtype=np.float32)
    lib.bpd_batchnorm_cpu_affine_fused(x.ctypes.data,g.ctypes.data,bt.ctypes.data,
        mn.ctypes.data,vr.ctypes.data,bo.ctypes.data,sb.ctypes.data,ob.ctypes.data,
        c_int(N),c_int(C),c_int(HW),c_float(bm.eps))
    return bo

def bpd_silu(x):
    so=np.zeros_like(x); lib.bpd_silu_cpu(x.ctypes.data,so.ctypes.data,c_int(x.size)); return so

def bpd_cbs_profiled(x, cm, bm, layer_name):
    co = timed(f"{layer_name}/conv", lambda: bpd_conv(x, cm))
    bo = timed(f"{layer_name}/bn", lambda: bpd_bn(co, bm))
    so = timed(f"{layer_name}/silu", lambda: bpd_silu(bo))
    return so

def bpd_conv_bare(x, cm):
    w=np.ascontiguousarray(cm.weight.data.numpy(),dtype=np.float32)
    b=cm.bias.data.numpy().astype(np.float32) if cm.bias is not None else np.zeros(cm.out_channels,dtype=np.float32)
    N,Ci,H,W=x.shape; kH,kW=cm.kernel_size; sh,sw=cm.stride; ph,pw=cm.padding
    Ho=(H+2*ph-kH)//sh+1; Wo=(W+2*pw-kW)//sw+1
    out=np.zeros((N,cm.out_channels,Ho,Wo),dtype=np.float32)
    lib.bpd_conv2d_full_cpu(np.ascontiguousarray(x).ctypes.data,w.ctypes.data,b.ctypes.data,out.ctypes.data,
        c_int(N),c_int(Ci),c_int(H),c_int(W),c_int(cm.out_channels),c_int(kH),c_int(kW),
        c_int(sh),c_int(sw),c_int(ph),c_int(pw),c_int(1),c_int(1),c_int(cm.groups))
    return out

def bpd_bottleneck_profiled(x, bmod, name):
    y=bpd_cbs_profiled(x,bmod.cv1.conv,bmod.cv1.bn,f"{name}/bn_cv1")
    y=bpd_cbs_profiled(y,bmod.cv2.conv,bmod.cv2.bn,f"{name}/bn_cv2")
    if bmod.add:
        o=np.zeros_like(y)
        timed(f"{name}/add", lambda: lib.bpd_add_f32_cpu(y.ctypes.data,x.ctypes.data,o.ctypes.data,c_int(y.size)))
        return o
    return y

def bpd_c3_profiled(x, c3m, name):
    b1=bpd_cbs_profiled(x,c3m.cv1.conv,c3m.cv1.bn,f"{name}/cv1")
    for bi, bn in enumerate(c3m.m):
        b1=bpd_bottleneck_profiled(b1,bn,f"{name}/bot{bi}")
    b2=bpd_cbs_profiled(x,c3m.cv2.conv,c3m.cv2.bn,f"{name}/cv2")
    N,C1,H,W=b1.shape; _,C2,_,_=b2.shape
    cat=np.zeros((N,C1+C2,H,W),dtype=np.float32)
    timed(f"{name}/cat", lambda: lib.bpd_concat_cpu(b1.ctypes.data,b2.ctypes.data,cat.ctypes.data,c_int(N),c_int(C1),c_int(C2),c_int(H),c_int(W)))
    return bpd_cbs_profiled(cat,c3m.cv3.conv,c3m.cv3.bn,f"{name}/cv3")

def bpd_sppf_profiled(x, sm, name):
    y=bpd_cbs_profiled(x,sm.cv1.conv,sm.cv1.bn,f"{name}/cv1")
    N,C,H,W=y.shape
    y1=np.zeros_like(y);y2=np.zeros_like(y);y3=np.zeros_like(y)
    timed(f"{name}/mp1", lambda: lib.bpd_maxpool2d_cpu(y.ctypes.data,y1.ctypes.data,c_int(N),c_int(C),c_int(H),c_int(W),c_int(5),c_int(5),c_int(1),c_int(2)))
    timed(f"{name}/mp2", lambda: lib.bpd_maxpool2d_cpu(y1.ctypes.data,y2.ctypes.data,c_int(N),c_int(C),c_int(H),c_int(W),c_int(5),c_int(5),c_int(1),c_int(2)))
    timed(f"{name}/mp3", lambda: lib.bpd_maxpool2d_cpu(y2.ctypes.data,y3.ctypes.data,c_int(N),c_int(C),c_int(H),c_int(W),c_int(5),c_int(5),c_int(1),c_int(2)))
    cat=np.zeros((N,4*C,H,W),dtype=np.float32)
    timed(f"{name}/cat4", lambda: lib.bpd_concat4_cpu(y.ctypes.data,y1.ctypes.data,y2.ctypes.data,y3.ctypes.data,cat.ctypes.data,c_int(N),c_int(C),c_int(C),c_int(C),c_int(C),c_int(H),c_int(W)))
    return bpd_cbs_profiled(cat,sm.cv2.conv,sm.cv2.bn,f"{name}/cv2")

# Load and run
ckpt = torch.load("/tmp/yolo_canonical/yolov5n.pt", map_location="cpu", weights_only=False)
model = ckpt["model"].float().eval()
torch.manual_seed(42)
inp_np = (torch.randn(1, 3, 640, 640) * 0.1).numpy().astype(np.float32)

saved = {}; save_indices = set(model.save); x = inp_np.copy()
for i, layer in enumerate(model.model):
    lt = type(layer).__name__
    f = layer.f if hasattr(layer, "f") else -1
    if lt == "Detect":
        for di, src in enumerate([17, 20, 23]):
            timed(f"L24/det{di}", lambda s=src, d=di: bpd_conv_bare(saved[s], layer.m[d]))
        break
    if isinstance(f, list):
        a=x; b=saved[f[1]]; N,Ca,H,W=a.shape; _,Cb,_,_=b.shape
        x=np.zeros((N,Ca+Cb,H,W),dtype=np.float32)
        timed(f"L{i}/concat", lambda: lib.bpd_concat_cpu(a.ctypes.data,b.ctypes.data,x.ctypes.data,c_int(N),c_int(Ca),c_int(Cb),c_int(H),c_int(W)))
    elif lt=="Conv": x=bpd_cbs_profiled(x,layer.conv,layer.bn,f"L{i}")
    elif lt=="C3": x=bpd_c3_profiled(x,layer,f"L{i}")
    elif lt=="SPPF": x=bpd_sppf_profiled(x,layer,f"L{i}")
    elif lt=="Upsample":
        N,C,H,W=x.shape; o=np.zeros((N,C,2*H,2*W),dtype=np.float32)
        timed(f"L{i}/upsample", lambda: lib.bpd_upsample_nearest2d_cpu(x.ctypes.data,o.ctypes.data,c_int(N),c_int(C),c_int(H),c_int(W)))
        x=o
    if i in save_indices: saved[i] = x.copy()

# Report
total_ms = sum(v[0] for v in timings.values())

# Aggregate by op type
conv_ms = sum(v[0] for k,v in timings.items() if '/conv' in k)
bn_ms = sum(v[0] for k,v in timings.items() if '/bn' in k)
silu_ms = sum(v[0] for k,v in timings.items() if '/silu' in k)
mp_ms = sum(v[0] for k,v in timings.items() if '/mp' in k)
add_ms = sum(v[0] for k,v in timings.items() if '/add' in k)
cat_ms = sum(v[0] for k,v in timings.items() if '/cat' in k)
up_ms = sum(v[0] for k,v in timings.items() if '/upsample' in k)
det_ms = sum(v[0] for k,v in timings.items() if '/det' in k)

print("=" * 70)
print("YOLOv5n PROFILER — Where does the time go?")
print("=" * 70)
print()
print(f"{'Op Type':<15} {'Time (ms)':>10} {'% Total':>10} {'Optimize?'}")
print("-" * 50)
for name, ms, pct in sorted([
    ("Conv2d",      conv_ms, conv_ms/total_ms*100),
    ("BatchNorm",   bn_ms,   bn_ms/total_ms*100),
    ("SiLU",        silu_ms, silu_ms/total_ms*100),
    ("MaxPool",     mp_ms,   mp_ms/total_ms*100),
    ("Add (resid)", add_ms,  add_ms/total_ms*100),
    ("Concat",      cat_ms,  cat_ms/total_ms*100),
    ("Upsample",    up_ms,   up_ms/total_ms*100),
    ("Detect conv", det_ms,  det_ms/total_ms*100),
], key=lambda x: -x[1]):
    bar = "#" * int(pct / 2)
    print(f"  {name:<13} {ms:8.1f} ms {pct:7.1f}%  {bar}")

print(f"\n  TOTAL:       {total_ms:8.1f} ms")

# Top 10 individual ops
print()
print("=" * 70)
print("TOP 15 INDIVIDUAL OPS (sorted by time)")
print("=" * 70)
top = sorted(timings.items(), key=lambda x: -x[1][0])[:15]
for name, vals in top:
    ms = vals[0]
    pct = ms / total_ms * 100
    print(f"  {name:<40s} {ms:7.1f} ms ({pct:4.1f}%)")

# Fusion opportunities
print()
print("=" * 70)
print("FUSION OPPORTUNITIES (Conv+BN+SiLU → single kernel)")
print("=" * 70)
fusion_savings = 0
for k in timings:
    if '/conv' in k:
        base = k.replace('/conv', '')
        bn_k = base + '/bn'
        silu_k = base + '/silu'
        if bn_k in timings and silu_k in timings:
            conv_t = timings[k][0]
            bn_t = timings[bn_k][0]
            silu_t = timings[silu_k][0]
            total_t = conv_t + bn_t + silu_t
            # Fused would eliminate BN and SiLU memory passes
            est_fused = conv_t * 1.05  # conv dominates, ~5% overhead for epilogue
            saving = total_t - est_fused
            if saving > 0.1:
                fusion_savings += saving
                print(f"  {base:<35s} {total_t:6.1f}ms → ~{est_fused:5.1f}ms (save {saving:4.1f}ms)")

print(f"\n  Estimated total fusion savings: {fusion_savings:.1f} ms")
print(f"  Current total: {total_ms:.0f} ms → Fused: ~{total_ms - fusion_savings:.0f} ms")
