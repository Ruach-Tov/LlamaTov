#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Sweep MLIR GEMM tile sizes: for each (BM,BN,BK,TM,TN), generate via the emitter,
lower to PTX, VERIFY bit-exact vs numpy, then TIME. Report GFLOPS for correct configs.
Compare to the tuned cuda-c GEMM (41% cuBLAS) and cuBLAS itself."""
import os, re, subprocess, time
import numpy as np
import sys as _sys; _sys.path.insert(0, _os.path.join(_BPD, "lib"))
import toolchain as tc
import os as _os, sys as _sys
def _bpd_root(_p=_os.path.dirname(_os.path.abspath(__file__))):
    while _p != '/' and _os.path.basename(_p) != 'bpd':
        _p = _os.path.dirname(_p)
    return _p if _os.path.basename(_p) == 'bpd' else _os.path.dirname(_os.path.abspath(__file__))
_BPD = _bpd_root()

REPO="<repo>/Ruach-Tov"; MB="/tmp/gpu-work/mlir_backend"
CUDA=tc.cuda_root()  # shared toolchain (ENV-SHIFT defense)
SWIPL="/run/current-system/sw/bin/swipl"
ENV=dict(os.environ, PATH=f"/run/current-system/sw/bin:{CUDA}/bin:"+os.environ.get("PATH",""),
         LD_LIBRARY_PATH=f"/run/opengl-driver/lib:{CUDA}/lib")
def run(c,sh=False,t=120): return subprocess.run(c,capture_output=True,text=True,env=ENV,cwd=MB,timeout=t,shell=sh)

PEAK_FP32=5.70e3  # P4 GFLOPS

# build a generic launcher (NTH passed as arg)
open(f"{MB}/gs_run.cu","w").write(r'''
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>
#define CK(x) do{CUresult r=(x);if(r!=CUDA_SUCCESS){const char*s;cuGetErrorString(r,&s);printf("ERR %s:%s\n",#x,s);return 1;}}while(0)
int main(int c,char**v){
  int M=atoi(v[2]),N=atoi(v[3]),K=atoi(v[4]),BM=atoi(v[5]),BN=atoi(v[6]),NTH=atoi(v[7]);
  int verify=atoi(v[8]);
  CK(cuInit(0)); CUdevice d; CK(cuDeviceGet(&d,0)); CUcontext ctx; CK(cuCtxCreate(&ctx,0,d));
  CUmodule m; CK(cuModuleLoad(&m,v[1])); CUfunction fn; CK(cuModuleGetFunction(&fn,m,"k_gemm"));
  size_t ba=(size_t)M*K*4, bb=(size_t)K*N*4, bo=(size_t)M*N*4;
  float*ah=(float*)malloc(ba),*bh=(float*)malloc(bb),*ch=(float*)malloc(bo);
  FILE*fa=fopen("ga.bin","rb");fread(ah,4,(size_t)M*K,fa);fclose(fa);
  FILE*fb=fopen("gb.bin","rb");fread(bh,4,(size_t)K*N,fb);fclose(fb);
  CUdeviceptr A,B,C; CK(cuMemAlloc(&A,ba));CK(cuMemAlloc(&B,bb));CK(cuMemAlloc(&C,bo));
  CK(cuMemcpyHtoD(A,ah,ba));CK(cuMemcpyHtoD(B,bh,bb));
  void* args[]={&A,&B,&C,&M,&N,&K};
  unsigned gx=(N+BN-1)/BN, gy=(M+BM-1)/BM;
  CK(cuLaunchKernel(fn,gx,gy,1,NTH,1,1,0,0,args,0)); CK(cuCtxSynchronize());
  if(verify){ CK(cuMemcpyDtoH(ch,C,bo)); FILE*fo=fopen("gc.bin","wb");fwrite(ch,4,(size_t)M*N,fo);fclose(fo); }
  CUevent t0,t1;CK(cuEventCreate(&t0,0));CK(cuEventCreate(&t1,0)); int IT=30;
  CK(cuEventRecord(t0,0)); for(int i=0;i<IT;i++) CK(cuLaunchKernel(fn,gx,gy,1,NTH,1,1,0,0,args,0));
  CK(cuEventRecord(t1,0));CK(cuEventSynchronize(t1)); float ms=0;CK(cuEventElapsedTime(&ms,t0,t1)); ms/=IT;
  double gf=2.0*M*N*K/(ms*1e-3)/1e9;
  printf("ms=%.4f gflops=%.1f\n",ms,gf);
  return 0;
}
''')
run(tc.nvcc_link_cmd(f"{MB}/gs_run.cu", f"{MB}/gs_run"))  # shared toolchain link (devrt+driver -L)

def gen_and_lower(BM,BN,BK,TM,TN):
    g=(f"use_module('{REPO}/bpd/lib/robust_op_match.pl',[op_expr/2]), "
       f"use_module('{REPO}/bpd/kernelgen/schedule/schedule_ir.pl'), "
       f"use_module('{REPO}/bpd/kernelgen/schedule/lower_schedule_mlir.pl'), "
       f"emit_schedule_mlir(bpd_matmul, tiled_gemm({BM},{BN},{BK},{TM},{TN}), '{MB}/sw.mlir'), halt")
    rg=run([SWIPL,"-q","-g",g,"-t","halt"],t=60)
    if not os.path.exists(f"{MB}/sw.mlir"):
        print(f"  [gen fail: {rg.stderr[:80]}]"); return None
    low=('mlir-opt sw.mlir --convert-arith-to-llvm --nvvm-attach-target="chip=sm_61 O=3" '
         '--convert-gpu-to-nvvm --reconcile-unrealized-casts --gpu-module-to-binary="format=isa"')
    r=run(low,sh=True)
    m=re.search(r'assembly = "(.*?)">\]', r.stdout, re.DOTALL)
    if not m:
        print(f"  [lower: stdout={len(r.stdout)}b stderr={r.stderr[:100]}]"); return None
    ptx=(m.group(1).replace("\\0A","\n").replace("\\09","\t").replace("\\00","").replace('\\22','"').replace("\\5C","\\"))
    open(f"{MB}/sw.ptx","w").write(ptx)
    return f"{MB}/sw.ptx"

# test matrix: square-ish projection size
M,N,K=512,512,512
A=np.random.default_rng(0).standard_normal((M,K)).astype(np.float32)
B=np.random.default_rng(1).standard_normal((K,N)).astype(np.float32)
A.tofile(f"{MB}/ga.bin"); B.tofile(f"{MB}/gb.bin"); ref=A@B

# tile configs to sweep (BM,BN,BK,TM,TN) — must have BM%TM==0, BN%TN==0
configs=[
    (32,32,8,2,2),(32,32,16,4,4),(64,64,16,4,4),(64,64,32,8,8),
    (64,128,16,4,4),(128,64,16,4,4),(128,128,16,8,8),(128,128,32,8,4),
    (128,128,8,8,8),(64,64,8,8,8),
]
print(f"=== MLIR GEMM tile sweep (M=N=K={M}, P4) ===")
print(f"{'tile':<28}{'NTH':<6}{'correct':<9}{'ms':<10}{'GFLOPS':<9}{'%FP32':<7}")
results=[]
for (BM,BN,BK,TM,TN) in configs:
    NTH=(BM//TM)*(BN//TN)
    if NTH>1024: 
        print(f"BM{BM}_BN{BN}_TM{TM}_TN{TN:<14} NTH={NTH} SKIP(>1024)"); continue
    ptx=gen_and_lower(BM,BN,BK,TM,TN)
    tag=f"BM{BM}_BN{BN}_BK{BK}_TM{TM}_TN{TN}"
    if not ptx: print(f"{tag:<28}{NTH:<6}LOWER-FAIL"); continue
    try: os.remove(f"{MB}/gc.bin")
    except: pass
    r=run([f"{MB}/gs_run",ptx,str(M),str(N),str(K),str(BM),str(BN),str(NTH),"1"])
    mm=re.search(r"gflops=([\d.]+)",r.stdout); ms_m=re.search(r"ms=([\d.]+)",r.stdout)
    if not mm: print(f"{tag:<28}{NTH:<6}RUN-FAIL {r.stdout[:40]}{r.stderr[:40]}"); continue
    gf=float(mm.group(1)); ms=float(ms_m.group(1))
    correct=False
    if os.path.exists(f"{MB}/gc.bin"):
        got=np.fromfile(f"{MB}/gc.bin",np.float32).reshape(M,N)
        d=np.abs(got-ref); correct=bool(((d/(np.abs(ref)+1e-30)>1e-2)&(d>1e-2)).sum()==0)
    pct=gf/PEAK_FP32*100
    print(f"{tag:<28}{NTH:<6}{str(correct):<9}{ms:<10.4f}{gf:<9.1f}{pct:<7.1f}")
    if correct: results.append((tag,gf))
results.sort(key=lambda x:-x[1])
if results:
    print(f"\nBEST MLIR GEMM: {results[0][0]} = {results[0][1]:.1f} GFLOPS ({results[0][1]/PEAK_FP32*100:.1f}% FP32-peak)")
print("(ref: tuned cuda-c rect GEMM ~1479 GFLOPS at BM128 = 26% FP32 / 41% cuBLAS; cuBLAS ~3607)")
