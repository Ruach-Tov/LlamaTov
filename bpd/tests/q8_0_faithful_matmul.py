# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
# FAITHFUL ggml Q8_0 matmul (decoded from reference) -> verify 0 ULP vs spec (Iyun, Heath)
import sys, struct, numpy as np
sys.path.insert(0,'.')
def read_bin(p):
    with open(p,'rb') as f:
        struct.unpack('<I',f.read(4)); struct.unpack('<I',f.read(4)); ne=struct.unpack('<4q',f.read(32)); struct.unpack('<4Q',f.read(32)); nb=struct.unpack('<Q',f.read(8))[0]; raw=f.read(nb)
    a=np.frombuffer(raw,dtype='<f4').astype(np.float32); dims=[d for d in ne if d>1] or [ne[0]]
    return a.reshape(dims[::-1]) if len(dims)>1 else a
from llamatov_run import parse_gguf
import struct as st
def fp16(x):  # fp32 -> fp16 -> fp32 round-trip (ggml stores scales as fp16)
    return np.frombuffer(np.float16(x).tobytes(),dtype=np.float16).astype(np.float32)[0]
md,ts,do=parse_gguf('/tmp/llamatov-data/ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45')
# read RAW Q8_0 weight blocks for attn_output: block_q8_0 = {fp16 d; int8 qs[32]} = 34 bytes, QK=32
dims,typ,off=ts['blk.0.attn_output.weight']  # type 8 = Q8_0
ncols,nrows=dims[0],dims[1]  # [2048,2048]
QK=32; bpr=ncols//QK  # blocks per row
with open('/tmp/llamatov-data/ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45','rb') as f:
    f.seek(do+off); raw=np.frombuffer(f.read(nrows*bpr*34),dtype=np.uint8).reshape(nrows,bpr,34)
wd=raw[:,:,0:2].copy().view(np.float16).astype(np.float32).reshape(nrows,bpr)  # block scales
wq=raw[:,:,2:34].view(np.int8).reshape(nrows,bpr,QK)                            # int8 weights
def quant_q8_0(x):  # quantize_row_q8_0: per block of 32
    n=len(x)//QK; d=np.zeros(n,dtype=np.float32); q=np.zeros((n,QK),dtype=np.int8)
    for i in range(n):
        blk=x[i*QK:(i+1)*QK]; amax=float(np.max(np.abs(blk))); dd=amax/127.0; idd=1.0/dd if dd else 0.0
        d[i]=fp16(dd); q[i]=np.round(blk*idd).astype(np.int8)
    return d,q
kqv=read_bin('<home>/tmp/spec_dump/0048_kqv_out-0.bin'); spec=read_bin('<home>/tmp/spec_dump/0050_attn_out-0.bin')
ntok=kqv.shape[0] if kqv.ndim>1 else 1; kqv2=kqv.reshape(ntok,ncols)
out=np.zeros((ntok,nrows),dtype=np.float32)
for t in range(ntok):
    ad,aq=quant_q8_0(kqv2[t])  # quantize activation row to Q8_0
    for r in range(nrows):
        sumf=np.float32(0)
        for b in range(bpr):
            sumi=int(np.dot(wq[r,b].astype(np.int32),aq[b].astype(np.int32)))  # int8xint8->int32
            sumf=np.float32(sumf + fp16(wd[r,b])*fp16(ad[b])*np.float32(sumi))
        out[t,r]=sumf
def maxabs(a,b): a=a.ravel();b=b.ravel();n=min(a.size,b.size);return float(np.max(np.abs(a[:n]-b[:n])))
def ulp(a,b):
    a=a.ravel().astype(np.float32);b=b.ravel().astype(np.float32);n=min(a.size,b.size);a,b=a[:n],b[:n]
    ai=a.view(np.int32).astype(np.int64);bi=b.view(np.int32).astype(np.int64);ai=np.where(ai<0,2**31-ai,ai);bi=np.where(bi<0,2**31-bi,bi)
    return int(np.max(np.abs(ai-bi)))
print(f'o_proj FAITHFUL-Q8_0 (decoded reference) vs spec: max_abs={maxabs(out,spec):.3e} max_ULP={ulp(out,spec)}')
print(f'  -> {"0 ULP / BIT-IDENTICAL by construction!" if ulp(out,spec)==0 else ("tiny ("+f"{maxabs(out,spec):.1e}"+") - much closer than fp32 0.001" if maxabs(out,spec)<0.001 else "still diverges")}')
