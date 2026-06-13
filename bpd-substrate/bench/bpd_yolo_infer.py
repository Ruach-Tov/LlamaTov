#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""5c: Verify BPD YOLO on a stock test image — detection boxes must match PyTorch."""
import sys, os; sys.path.insert(0, "/tmp/yolov5")
import torch, numpy as np, ctypes, cv2
torch.set_num_threads(1); torch.backends.mkldnn.enabled = False

# Import the full forward from our module
sys.path.insert(0, os.path.dirname(__file__))

lib = ctypes.CDLL(os.environ.get("BPD_CPU_SO", "build/bpd_cpu.so"))
c_void=ctypes.c_void_p; c_int=ctypes.c_int; c_float=ctypes.c_float
lib.bpd_conv2d_full_cpu.argtypes=[c_void]*4+[c_int]*14; lib.bpd_conv2d_full_cpu.restype=None
lib.bpd_silu_cpu.argtypes=[c_void]*2+[c_int]; lib.bpd_silu_cpu.restype=None
lib.bpd_sigmoid_cpu.argtypes=[c_void]*2+[c_int]; lib.bpd_sigmoid_cpu.restype=None
lib.bpd_batchnorm_cpu_affine_fused.argtypes=[c_void]*8+[c_int]*3+[c_float]; lib.bpd_batchnorm_cpu_affine_fused.restype=None
lib.bpd_maxpool2d_cpu.argtypes=[c_void]*2+[c_int]*8; lib.bpd_maxpool2d_cpu.restype=None
lib.bpd_add_f32_cpu.argtypes=[c_void]*3+[c_int]; lib.bpd_add_f32_cpu.restype=None
lib.bpd_concat_cpu.argtypes=[c_void]*3+[c_int]*5; lib.bpd_concat_cpu.restype=None
lib.bpd_concat4_cpu.argtypes=[c_void]*5+[c_int]*7; lib.bpd_concat4_cpu.restype=None
lib.bpd_upsample_nearest2d_cpu.argtypes=[c_void]*2+[c_int]*4; lib.bpd_upsample_nearest2d_cpu.restype=None

def bpd_cbs(x, cm, bm):
    w=np.ascontiguousarray(cm.weight.data.numpy(),dtype=np.float32)
    b=np.zeros(cm.out_channels,dtype=np.float32)
    N,Ci,H,W=x.shape; kH,kW=cm.kernel_size; sh,sw=cm.stride; ph,pw=cm.padding
    Ho=(H+2*ph-kH)//sh+1; Wo=(W+2*pw-kW)//sw+1
    co=np.zeros((N,cm.out_channels,Ho,Wo),dtype=np.float32)
    lib.bpd_conv2d_full_cpu(np.ascontiguousarray(x).ctypes.data,w.ctypes.data,b.ctypes.data,co.ctypes.data,
        c_int(N),c_int(Ci),c_int(H),c_int(W),c_int(cm.out_channels),c_int(kH),c_int(kW),
        c_int(sh),c_int(sw),c_int(ph),c_int(pw),c_int(1),c_int(1),c_int(cm.groups))
    g=bm.weight.data.numpy().astype(np.float32); bt=bm.bias.data.numpy().astype(np.float32)
    mn=bm.running_mean.data.numpy().astype(np.float32); vr=bm.running_var.data.numpy().astype(np.float32)
    C=bm.num_features; HW=Ho*Wo
    bo=np.zeros_like(co); sb=np.zeros(C,dtype=np.float32); ob=np.zeros(C,dtype=np.float32)
    lib.bpd_batchnorm_cpu_affine_fused(co.ctypes.data,g.ctypes.data,bt.ctypes.data,
        mn.ctypes.data,vr.ctypes.data,bo.ctypes.data,sb.ctypes.data,ob.ctypes.data,
        c_int(N),c_int(C),c_int(HW),c_float(bm.eps))
    so=np.zeros_like(bo); lib.bpd_silu_cpu(bo.ctypes.data,so.ctypes.data,c_int(bo.size))
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

def bpd_bottleneck(x, bmod):
    y=bpd_cbs(x,bmod.cv1.conv,bmod.cv1.bn); y=bpd_cbs(y,bmod.cv2.conv,bmod.cv2.bn)
    if bmod.add:
        o=np.zeros_like(y); lib.bpd_add_f32_cpu(y.ctypes.data,x.ctypes.data,o.ctypes.data,c_int(y.size)); return o
    return y

def bpd_c3(x, c3m):
    b1=bpd_cbs(x,c3m.cv1.conv,c3m.cv1.bn)
    for bn in c3m.m: b1=bpd_bottleneck(b1,bn)
    b2=bpd_cbs(x,c3m.cv2.conv,c3m.cv2.bn)
    N,C1,H,W=b1.shape; _,C2,_,_=b2.shape
    cat=np.zeros((N,C1+C2,H,W),dtype=np.float32)
    lib.bpd_concat_cpu(b1.ctypes.data,b2.ctypes.data,cat.ctypes.data,c_int(N),c_int(C1),c_int(C2),c_int(H),c_int(W))
    return bpd_cbs(cat,c3m.cv3.conv,c3m.cv3.bn)

def bpd_sppf(x, sm):
    y=bpd_cbs(x,sm.cv1.conv,sm.cv1.bn); N,C,H,W=y.shape
    y1=np.zeros_like(y);y2=np.zeros_like(y);y3=np.zeros_like(y)
    lib.bpd_maxpool2d_cpu(y.ctypes.data,y1.ctypes.data,c_int(N),c_int(C),c_int(H),c_int(W),c_int(5),c_int(5),c_int(1),c_int(2))
    lib.bpd_maxpool2d_cpu(y1.ctypes.data,y2.ctypes.data,c_int(N),c_int(C),c_int(H),c_int(W),c_int(5),c_int(5),c_int(1),c_int(2))
    lib.bpd_maxpool2d_cpu(y2.ctypes.data,y3.ctypes.data,c_int(N),c_int(C),c_int(H),c_int(W),c_int(5),c_int(5),c_int(1),c_int(2))
    cat=np.zeros((N,4*C,H,W),dtype=np.float32)
    lib.bpd_concat4_cpu(y.ctypes.data,y1.ctypes.data,y2.ctypes.data,y3.ctypes.data,
        cat.ctypes.data,c_int(N),c_int(C),c_int(C),c_int(C),c_int(C),c_int(H),c_int(W))
    return bpd_cbs(cat,sm.cv2.conv,sm.cv2.bn)

def bpd_up(x):
    N,C,H,W=x.shape; o=np.zeros((N,C,2*H,2*W),dtype=np.float32)
    lib.bpd_upsample_nearest2d_cpu(x.ctypes.data,o.ctypes.data,c_int(N),c_int(C),c_int(H),c_int(W)); return o

def bpd_yolov5n_forward(model, inp_np):
    saved = {}; save_indices = set(model.save); x = inp_np.copy()
    for i, layer in enumerate(model.model):
        lt = type(layer).__name__
        f = layer.f if hasattr(layer, "f") else -1
        if lt == "Detect":
            dets = []
            for di, src in enumerate([17, 20, 23]):
                dets.append(bpd_conv_bare(saved[src], layer.m[di]))
            return dets
        if isinstance(f, list):
            a=x; b=saved[f[1]]; N,Ca,H,W=a.shape; _,Cb,_,_=b.shape
            x=np.zeros((N,Ca+Cb,H,W),dtype=np.float32)
            lib.bpd_concat_cpu(a.ctypes.data,b.ctypes.data,x.ctypes.data,c_int(N),c_int(Ca),c_int(Cb),c_int(H),c_int(W))
        elif lt=="Conv": x=bpd_cbs(x,layer.conv,layer.bn)
        elif lt=="C3": x=bpd_c3(x,layer)
        elif lt=="SPPF": x=bpd_sppf(x,layer)
        elif lt=="Upsample": x=bpd_up(x)
        if i in save_indices: saved[i] = x.copy()

def preprocess_image(img_path, img_size=640):
    """YOLOv5 letterbox preprocessing."""
    img = cv2.imread(img_path)
    h0, w0 = img.shape[:2]
    r = img_size / max(h0, w0)
    new_h, new_w = int(h0 * r), int(w0 * r)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # Pad to 640x640
    dh = (img_size - new_h) / 2
    dw = (img_size - new_w) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    # BGR to RGB, HWC to CHW, normalize
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()
    img = img.astype(np.float32) / 255.0
    return img[np.newaxis]  # add batch dim

# Load model
ckpt = torch.load("/tmp/yolo_canonical/yolov5n.pt", map_location="cpu", weights_only=False)
model = ckpt["model"].float().eval()

print("=" * 60)
print("YOLOv5n STOCK IMAGE VERIFICATION")
print("=" * 60)

for img_name in ["bus.jpg", "zidane.jpg"]:
    img_path = f"/tmp/{img_name}"
    print(f"\nImage: {img_name}")

    # Preprocess
    inp_np = preprocess_image(img_path)
    inp_t = torch.from_numpy(inp_np.copy())
    print(f"  Input shape: {inp_np.shape}")

    # BPD forward
    bpd_dets = bpd_yolov5n_forward(model, inp_np)

    # PyTorch forward (raw detect conv outputs)
    with torch.no_grad():
        saved_pt = {}; x_pt = inp_t; save_indices = set(model.save)
        for i, layer in enumerate(model.model):
            if type(layer).__name__ == "Detect": break
            f = layer.f if hasattr(layer, "f") else -1
            if isinstance(f, list):
                x_pt = layer([x_pt, saved_pt[f[1]]])
            else:
                x_pt = layer(x_pt)
            if i in save_indices: saved_pt[i] = x_pt
        detect = model.model[-1]
        pt_dets = [detect.m[di](saved_pt[src]).numpy().astype(np.float32)
                   for di, src in enumerate([17, 20, 23])]

    # Compare all 3 heads
    all_match = True
    for di in range(3):
        d = np.abs(bpd_dets[di].view(np.int32).flatten().astype(np.int64) -
                   pt_dets[di].view(np.int32).flatten().astype(np.int64))
        max_ulp = int(d.max())
        if max_ulp > 0: all_match = False
        print(f"  Head {di}: {list(bpd_dets[di].shape)} max_ULP={max_ulp} {'0 ULP ✅' if max_ulp==0 else '❌'}")

    if all_match:
        print(f"  🎉 {img_name}: ALL HEADS BIT_IDENTICAL")
    else:
        print(f"  ❌ {img_name}: DIVERGENCE DETECTED")

    # Also run full PyTorch inference to show what detections WOULD be
    with torch.no_grad():
        pt_full = model(inp_t)
    if isinstance(pt_full, tuple):
        pred = pt_full[0]  # [1, num_dets, 85]
        # Filter by confidence > 0.25
        conf = pred[0, :, 4]
        mask = conf > 0.25
        n_det = mask.sum().item()
        if n_det > 0:
            boxes = pred[0, mask]
            print(f"  PyTorch detections (conf>0.25): {n_det}")
            for j in range(min(5, n_det)):
                cls = boxes[j, 5:].argmax().item()
                cf = boxes[j, 4].item()
                x1, y1, x2, y2 = boxes[j, :4].tolist()
                print(f"    [{j}] class={cls} conf={cf:.3f} box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
        else:
            print(f"  No detections above 0.25 confidence")

print()
print("=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
