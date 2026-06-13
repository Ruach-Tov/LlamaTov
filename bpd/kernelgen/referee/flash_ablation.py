#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""flash_ablation.py — the RELAXING FA-2 harness. Given a dict of named kernel
variants (each a .cu source string + launch config), it builds every one, verifies
bit-exact vs the attention oracle, reports ptxas spill, ranks by time, and lets you
CUPTI the winner. Each variant flips ONE knob from the known-good warp kernel, so the
ranked table answers "did it help?" instantly — no entangled mystery.

The ablation ladder (single-knob steps toward FA-2):
  rung0  warp-shuffle vectorized (known-good, 0.341ms, 43.4% exec-dep)
  rung2  +QPW queries/warp (amortize shuffle, keep occupancy)
  rung4  register-GEMM score (no shuffle) — the big one
  rung5  +distributed acc (prevent reg-gemm spill)
  rung6  +shared softmax row-reduce

Usage on enclave:  python3 flash_ablation.py
Author: Iyun, 2026-06-08
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kernel_harness import (Env, build_cubin, run_kernel, verify, time_kernel,
                            attention_reference, deterministic_qkv)

S, D = 512, 128

# ── a variant = (source, kernel_name, (grid, block, shmem)) ──────────────────
# Each is a self-contained .cu. Launch geometry derived from #define knobs.
def variants():
    """Return [(name, src, kname, launch_fn)]. launch_fn(S,D)->(grid,block,shmem).
    Add a new rung by appending one entry — that's the whole loop."""
    V = []

    # rung0: the known-good warp-shuffle vectorized flash (reference baseline)
    V.append(("rung0_warp_shuffle", _SRC_WARP, "k_flash", _cfg(16, 32)))

    # rung2: QPW=2 queries per warp (amortize the shuffle over 2 independent dots)
    V.append(("rung2_qpw2", _SRC_QPW.replace("__QPW__", "2"), "k_flash", _cfg(16, 32)))
    V.append(("rung2_qpw4", _SRC_QPW.replace("__QPW__", "4"), "k_flash", _cfg(16, 32)))

    # rung4a: register-score, thread-per-query, NO shuffle (block=256 threads=256 queries)
    def cfg_tpq(S, D):
        BLK=256; BC=32
        return ((S+BLK-1)//BLK,), (BLK,), 2*BC*D*4
    V.append(("rung4a_regscore", _SRC_REGSCORE, "k_flash", cfg_tpq))

    # rung4b: FA-2 register micro-tile (BR=16, 64 threads/block, score reg-gemm + distributed acc)
    def cfg_regtile(S, D):
        BR=16; BC=32; blk=(BR//2)*(BC//4)  # 64
        shmem=(BR*D + BC*D + BC*D + BR*(BC//4) + BR*BC)*4
        return ((S+BR-1)//BR,), (blk,), shmem
    V.append(("rung4b_regtile", _SRC_REGTILE, "k_flash", cfg_regtile))

    # rung4c/d: occupancy knob — smaller TM/TN -> MORE threads/block (CUPTI said occupancy)
    def cfg_rt(TM, TN):
        def fn(S, D, TM=TM, TN=TN):
            BR=16; BC=32; blk=(BR//TM)*(BC//TN)
            shmem=(BR*D + BC*D + BC*D + BR*(BC//TN) + BR*BC)*4
            return ((S+BR-1)//BR,), (blk,), shmem
        return fn
    V.append(("rung4c_tm1tn2", _SRC_REGTILE_T.replace("__TM__","1").replace("__TN__","2"), "k_flash", cfg_rt(1,2)))
    V.append(("rung4d_tm2tn2", _SRC_REGTILE_T.replace("__TM__","2").replace("__TN__","2"), "k_flash", cfg_rt(2,2)))
    V.append(("rung4e_tm1tn4", _SRC_REGTILE_T.replace("__TM__","1").replace("__TN__","4"), "k_flash", cfg_rt(1,4)))

    return V

def _cfg(WPB, BC):
    def fn(S, D, WPB=WPB, BC=BC):
        return ((S + WPB - 1)//WPB,), (32*WPB,), 2*BC*D*4
    return fn

# ── kernel sources (one knob different each) ─────────────────────────────────
_SRC_WARP = r'''
#define WPB 16
#define BC 32
extern "C" __global__ void k_flash(const float* Q,const float* K,const float* V,
                                   float* O,int S,int D,float scale){
  int lane=threadIdx.x&31, wid=threadIdx.x>>5; int q=blockIdx.x*WPB+wid;
  extern __shared__ float sh[]; float* Ks=sh; float* Vs=sh+BC*D;
  float4 qreg=(q<S)?*reinterpret_cast<const float4*>(Q+(long)q*D+lane*4):make_float4(0,0,0,0);
  float4 acc=make_float4(0,0,0,0); float m=-3.4e38f,l=0.f; int tid=threadIdx.x,nth=WPB*32;
  for(int k0=0;k0<S;k0+=BC){
    for(int idx=tid; idx<BC*D/4; idx+=nth){ int e4=idx*4; int kr=e4/D,kc=e4%D; int gk=k0+kr;
      float4 kv=(gk<S)?*reinterpret_cast<const float4*>(K+(long)gk*D+kc):make_float4(0,0,0,0);
      float4 vv=(gk<S)?*reinterpret_cast<const float4*>(V+(long)gk*D+kc):make_float4(0,0,0,0);
      *reinterpret_cast<float4*>(Ks+idx*4)=kv; *reinterpret_cast<float4*>(Vs+idx*4)=vv; }
    __syncthreads();
    if(q<S){ for(int kk=0;kk<BC;kk++){ int gk=k0+kk; if(gk>=S)break;
      float4 ks=*reinterpret_cast<float4*>(Ks+kk*D+lane*4);
      float pa=qreg.x*ks.x+qreg.y*ks.y+qreg.z*ks.z+qreg.w*ks.w;
      for(int off=16;off>0;off>>=1) pa+=__shfl_down_sync(0xffffffff,pa,off);
      float s=__shfl_sync(0xffffffff,pa,0)*scale;
      float mn=fmaxf(m,s),corr=expf(m-mn),p=expf(s-mn); l=l*corr+p;
      float4 vs=*reinterpret_cast<float4*>(Vs+kk*D+lane*4);
      acc.x=acc.x*corr+p*vs.x; acc.y=acc.y*corr+p*vs.y; acc.z=acc.z*corr+p*vs.z; acc.w=acc.w*corr+p*vs.w;
      m=mn; } }
    __syncthreads();
  }
  if(q<S){ float inv=1.f/l; float4 o=make_float4(acc.x*inv,acc.y*inv,acc.z*inv,acc.w*inv);
    *reinterpret_cast<float4*>(O+(long)q*D+lane*4)=o; }
}
'''

# rung2: QPW queries per warp. WPB warps each own QPW query rows. The QPW dots are
# independent (different queries) -> ILP that may hide shuffle latency. Block still
# WPB*32 threads (occupancy preserved); each warp now covers QPW rows.
_SRC_QPW = r'''
#define WPB 16
#define BC 32
#define QPW __QPW__
extern "C" __global__ void k_flash(const float* Q,const float* K,const float* V,
                                   float* O,int S,int D,float scale){
  int lane=threadIdx.x&31, wid=threadIdx.x>>5; int q0=(blockIdx.x*WPB+wid)*QPW;
  extern __shared__ float sh[]; float* Ks=sh; float* Vs=sh+BC*D;
  float4 qreg[QPW]; float4 acc[QPW]; float m[QPW],l[QPW];
  #pragma unroll
  for(int u=0;u<QPW;u++){ int q=q0+u;
    qreg[u]=(q<S)?*reinterpret_cast<const float4*>(Q+(long)q*D+lane*4):make_float4(0,0,0,0);
    acc[u]=make_float4(0,0,0,0); m[u]=-3.4e38f; l[u]=0.f; }
  int tid=threadIdx.x,nth=WPB*32;
  for(int k0=0;k0<S;k0+=BC){
    for(int idx=tid; idx<BC*D/4; idx+=nth){ int e4=idx*4; int kr=e4/D,kc=e4%D; int gk=k0+kr;
      float4 kv=(gk<S)?*reinterpret_cast<const float4*>(K+(long)gk*D+kc):make_float4(0,0,0,0);
      float4 vv=(gk<S)?*reinterpret_cast<const float4*>(V+(long)gk*D+kc):make_float4(0,0,0,0);
      *reinterpret_cast<float4*>(Ks+idx*4)=kv; *reinterpret_cast<float4*>(Vs+idx*4)=vv; }
    __syncthreads();
    for(int kk=0;kk<BC;kk++){ int gk=k0+kk; if(gk>=S)break;
      float4 ks=*reinterpret_cast<float4*>(Ks+kk*D+lane*4);
      float4 vs=*reinterpret_cast<float4*>(Vs+kk*D+lane*4);
      // QPW INDEPENDENT partial dots in flight (the ILP)
      float pa[QPW];
      #pragma unroll
      for(int u=0;u<QPW;u++) pa[u]=qreg[u].x*ks.x+qreg[u].y*ks.y+qreg[u].z*ks.z+qreg[u].w*ks.w;
      #pragma unroll
      for(int off=16;off>0;off>>=1)
        #pragma unroll
        for(int u=0;u<QPW;u++) pa[u]+=__shfl_down_sync(0xffffffff,pa[u],off);
      #pragma unroll
      for(int u=0;u<QPW;u++){ int q=q0+u; if(q>=S) continue;
        float s=__shfl_sync(0xffffffff,pa[u],0)*scale;
        float mn=fmaxf(m[u],s),corr=expf(m[u]-mn),p=expf(s-mn); l[u]=l[u]*corr+p;
        acc[u].x=acc[u].x*corr+p*vs.x; acc[u].y=acc[u].y*corr+p*vs.y;
        acc[u].z=acc[u].z*corr+p*vs.z; acc[u].w=acc[u].w*corr+p*vs.w; m[u]=mn; }
    }
    __syncthreads();
  }
  #pragma unroll
  for(int u=0;u<QPW;u++){ int q=q0+u; if(q<S){ float inv=1.f/l[u];
    float4 o=make_float4(acc[u].x*inv,acc[u].y*inv,acc[u].z*inv,acc[u].w*inv);
    *reinterpret_cast<float4*>(O+(long)q*D+lane*4)=o; } }
}
'''


# rung4a: register-score, ONE THREAD PER QUERY (full D per thread -> NO shuffle).
# Q's full D in registers (qr[D]). Score vs each key = thread-local dot over D.
# acc[D] in registers (this is the spill test — measured in isolation). One knob:
# the shuffle is GONE, replaced by a thread-local D-reduction.
_SRC_REGSCORE = r"""
#define WPB 8
#define BC 32
extern "C" __global__ void k_flash(const float* Q,const float* K,const float* V,
                                   float* O,int S,int D,float scale){
  int q = blockIdx.x*WPB + (threadIdx.x>>5)*0 + threadIdx.x;  // one thread per query
  // NOTE: block = WPB*32 threads, each a distinct query (thread-per-query, NOT warp-per-query)
  q = blockIdx.x*blockDim.x + threadIdx.x;
  if(q>=S) return;
  extern __shared__ float sh[]; float* Ks=sh; float* Vs=sh+BC*D;
  float qr[128]; for(int d=0;d<D;d++) qr[d]=Q[(long)q*D+d];   // Q cached in registers
  float acc[128]; for(int d=0;d<D;d++) acc[d]=0.f;
  float m=-3.4e38f,l=0.f; int tid=threadIdx.x,nth=blockDim.x;
  for(int k0=0;k0<S;k0+=BC){
    for(int idx=tid; idx<BC*D; idx+=nth){ int kr=idx/D,kc=idx%D; int gk=k0+kr;
      Ks[idx]=(gk<S)?K[(long)gk*D+kc]:0.f; Vs[idx]=(gk<S)?V[(long)gk*D+kc]:0.f; }
    __syncthreads();
    for(int kk=0;kk<BC;kk++){ int gk=k0+kk; if(gk>=S)break;
      float s=0.f; const float* krow=Ks+kk*D;
      for(int d=0;d<D;d++) s+=qr[d]*krow[d];     // thread-local dot, NO shuffle
      s*=scale;
      float mn=fmaxf(m,s),corr=expf(m-mn),p=expf(s-mn); l=l*corr+p;
      const float* vrow=Vs+kk*D;
      for(int d=0;d<D;d++) acc[d]=acc[d]*corr+p*vrow[d];
      m=mn;
    }
    __syncthreads();
  }
  float inv=1.f/l; for(int d=0;d<D;d++) O[(long)q*D+d]=acc[d]*inv;
}
"""


# rung4b: FA-2 register micro-tile score (tiled_gemm structure) — NO shuffle, NO full-D-per-thread.
_SRC_REGTILE = r"""
#define BR 16
#define TM 2
#define BC 32
#define TN 4
extern "C" __global__ void k_flash(const float* Q,const float* K,const float* V,
                                   float* O,int S,int D,float scale){
  // thread grid: (BR/TM) row-threads x (BC/TN) key-threads
  int nrt = BR/TM;            // row-thread count = 8
  int nkt = BC/TN;            // key-thread count = 8
  int ti = threadIdx.x % nrt; // row-thread index
  int tj = threadIdx.x / nrt; // key-thread index
  int q0 = blockIdx.x*BR + ti*TM;     // this thread's first query row
  const int DACC = 128/ (BC/TN);      // output cols per key-thread = 16
  extern __shared__ float sh[];
  float* Qs = sh;                     // BR*D  (the Q tile, cached in shared — fixes mem-dep)
  float* Ks = Qs + BR*D;              // BC*D
  float* Vs = Ks + BC*D;              // BC*D
  float* red = Vs + BC*D;             // BR*nkt  reduction scratch
  float* pbuf = red + BR*nkt;         // BR*BC   probs buffer
  int tid=threadIdx.x, nth=nrt*nkt;
  // load Q tile (BR rows) into shared once
  for(int idx=tid; idx<BR*D; idx+=nth){ int r=idx/D,c=idx%D; int gq=blockIdx.x*BR+r;
    Qs[idx]=(gq<S)?Q[(long)gq*D+c]:0.f; }
  float m[TM], l[TM], acc[TM][16];
  #pragma unroll
  for(int i=0;i<TM;i++){ m[i]=-3.4e38f; l[i]=0.f; for(int d=0;d<DACC;d++) acc[i][d]=0.f; }
  for(int k0=0;k0<S;k0+=BC){
    for(int idx=tid; idx<BC*D; idx+=nth){ int r=idx/D,c=idx%D; int gk=k0+r;
      Ks[idx]=(gk<S)?K[(long)gk*D+c]:0.f; Vs[idx]=(gk<S)?V[(long)gk*D+c]:0.f; }
    __syncthreads();
    // PHASE 1: score micro-tile s[TM][TN] via k-loop over D — TM*TN independent FMAs, NO shuffle
    float s[TM][TN];
    #pragma unroll
    for(int i=0;i<TM;i++) for(int j=0;j<TN;j++) s[i][j]=0.f;
    for(int d=0;d<D;d++){
      float qf[TM]; float kf[TN];
      #pragma unroll
      for(int i=0;i<TM;i++) qf[i]=Qs[(ti*TM+i)*D+d];
      #pragma unroll
      for(int j=0;j<TN;j++) kf[j]=Ks[(tj*TN+j)*D+d];
      #pragma unroll
      for(int i=0;i<TM;i++) for(int j=0;j<TN;j++) s[i][j]+=qf[i]*kf[j];
    }
    #pragma unroll
    for(int i=0;i<TM;i++) for(int j=0;j<TN;j++) s[i][j]*=scale;
    // PHASE 2: per query row (TM), reduce max over the TN keys ACROSS key-threads via shared
    #pragma unroll
    for(int i=0;i<TM;i++){
      float lm=-3.4e38f;
      #pragma unroll
      for(int j=0;j<TN;j++){ int gk=k0+tj*TN+j; if(gk<S && s[i][j]>lm) lm=s[i][j]; }
      red[(ti*TM+i)*nkt + tj]=lm; }
    __syncthreads();
    #pragma unroll
    for(int i=0;i<TM;i++){
      float rmax=-3.4e38f; for(int u=0;u<nkt;u++){ float v=red[(ti*TM+i)*nkt+u]; if(v>rmax)rmax=v; }
      float mn=fmaxf(m[i],rmax), corr=expf(m[i]-mn);
      // probs for this thread's TN keys
      float ls=0.f; float pj[TN];
      #pragma unroll
      for(int j=0;j<TN;j++){ int gk=k0+tj*TN+j; pj[j]=(gk<S)?expf(s[i][j]-mn):0.f; ls+=pj[j];
        pbuf[(ti*TM+i)*BC + tj*TN+j]=pj[j]; }
      // rescale acc AND l (online softmax: both rescale when the running max updates)
      #pragma unroll
      for(int d=0;d<DACC;d++) acc[i][d]*=corr;
      l[i]*=corr;                   // THE FIX: l rescaled by corr too
      red[(ti*TM+i)*nkt + tj]=ls;   // reuse red for sum
      m[i]=mn;
    }
    __syncthreads();
    // PHASE 3: l += rowsum; acc += P@V (each key-thread owns DACC cols, sums over ALL BC keys)
    #pragma unroll
    for(int i=0;i<TM;i++){
      float rs=0.f; for(int u=0;u<nkt;u++) rs+=red[(ti*TM+i)*nkt+u];
      l[i]+=rs;
      #pragma unroll
      for(int d=0;d<DACC;d++){ int col=tj*DACC+d; float a=acc[i][d];
        for(int kk=0;kk<BC;kk++){ int gk=k0+kk; if(gk>=S)break; a+=pbuf[(ti*TM+i)*BC+kk]*Vs[kk*D+col]; }
        acc[i][d]=a; }
    }
    __syncthreads();
  }
  #pragma unroll
  for(int i=0;i<TM;i++){ int q=q0+i; if(q<S){ float inv=1.f/l[i];
    for(int d=0;d<DACC;d++) O[(long)q*D + tj*DACC + d]=acc[i][d]*inv; } }
}
"""
_SRC_REGTILE_T = r"""
#define BR 16
#define TM __TM__
#define BC 32
#define TN __TN__
extern "C" __global__ void k_flash(const float* Q,const float* K,const float* V,
                                   float* O,int S,int D,float scale){
  // thread grid: (BR/TM) row-threads x (BC/TN) key-threads
  int nrt = BR/TM;            // row-thread count = 8
  int nkt = BC/TN;            // key-thread count = 8
  int ti = threadIdx.x % nrt; // row-thread index
  int tj = threadIdx.x / nrt; // key-thread index
  int q0 = blockIdx.x*BR + ti*TM;     // this thread's first query row
  const int DACC = 128/ (BC/TN);      // output cols per key-thread = 16
  extern __shared__ float sh[];
  float* Qs = sh;                     // BR*D  (the Q tile, cached in shared — fixes mem-dep)
  float* Ks = Qs + BR*D;              // BC*D
  float* Vs = Ks + BC*D;              // BC*D
  float* red = Vs + BC*D;             // BR*nkt  reduction scratch
  float* pbuf = red + BR*nkt;         // BR*BC   probs buffer
  int tid=threadIdx.x, nth=nrt*nkt;
  // load Q tile (BR rows) into shared once
  for(int idx=tid; idx<BR*D; idx+=nth){ int r=idx/D,c=idx%D; int gq=blockIdx.x*BR+r;
    Qs[idx]=(gq<S)?Q[(long)gq*D+c]:0.f; }
  float m[TM], l[TM], acc[TM][16];
  #pragma unroll
  for(int i=0;i<TM;i++){ m[i]=-3.4e38f; l[i]=0.f; for(int d=0;d<DACC;d++) acc[i][d]=0.f; }
  for(int k0=0;k0<S;k0+=BC){
    for(int idx=tid; idx<BC*D; idx+=nth){ int r=idx/D,c=idx%D; int gk=k0+r;
      Ks[idx]=(gk<S)?K[(long)gk*D+c]:0.f; Vs[idx]=(gk<S)?V[(long)gk*D+c]:0.f; }
    __syncthreads();
    // PHASE 1: score micro-tile s[TM][TN] via k-loop over D — TM*TN independent FMAs, NO shuffle
    float s[TM][TN];
    #pragma unroll
    for(int i=0;i<TM;i++) for(int j=0;j<TN;j++) s[i][j]=0.f;
    for(int d=0;d<D;d++){
      float qf[TM]; float kf[TN];
      #pragma unroll
      for(int i=0;i<TM;i++) qf[i]=Qs[(ti*TM+i)*D+d];
      #pragma unroll
      for(int j=0;j<TN;j++) kf[j]=Ks[(tj*TN+j)*D+d];
      #pragma unroll
      for(int i=0;i<TM;i++) for(int j=0;j<TN;j++) s[i][j]+=qf[i]*kf[j];
    }
    #pragma unroll
    for(int i=0;i<TM;i++) for(int j=0;j<TN;j++) s[i][j]*=scale;
    // PHASE 2: per query row (TM), reduce max over the TN keys ACROSS key-threads via shared
    #pragma unroll
    for(int i=0;i<TM;i++){
      float lm=-3.4e38f;
      #pragma unroll
      for(int j=0;j<TN;j++){ int gk=k0+tj*TN+j; if(gk<S && s[i][j]>lm) lm=s[i][j]; }
      red[(ti*TM+i)*nkt + tj]=lm; }
    __syncthreads();
    #pragma unroll
    for(int i=0;i<TM;i++){
      float rmax=-3.4e38f; for(int u=0;u<nkt;u++){ float v=red[(ti*TM+i)*nkt+u]; if(v>rmax)rmax=v; }
      float mn=fmaxf(m[i],rmax), corr=expf(m[i]-mn);
      // probs for this thread's TN keys
      float ls=0.f; float pj[TN];
      #pragma unroll
      for(int j=0;j<TN;j++){ int gk=k0+tj*TN+j; pj[j]=(gk<S)?expf(s[i][j]-mn):0.f; ls+=pj[j];
        pbuf[(ti*TM+i)*BC + tj*TN+j]=pj[j]; }
      // rescale acc AND l (online softmax: both rescale when the running max updates)
      #pragma unroll
      for(int d=0;d<DACC;d++) acc[i][d]*=corr;
      l[i]*=corr;                   // THE FIX: l rescaled by corr too
      red[(ti*TM+i)*nkt + tj]=ls;   // reuse red for sum
      m[i]=mn;
    }
    __syncthreads();
    // PHASE 3: l += rowsum; acc += P@V (each key-thread owns DACC cols, sums over ALL BC keys)
    #pragma unroll
    for(int i=0;i<TM;i++){
      float rs=0.f; for(int u=0;u<nkt;u++) rs+=red[(ti*TM+i)*nkt+u];
      l[i]+=rs;
      #pragma unroll
      for(int d=0;d<DACC;d++){ int col=tj*DACC+d; float a=acc[i][d];
        for(int kk=0;kk<BC;kk++){ int gk=k0+kk; if(gk>=S)break; a+=pbuf[(ti*TM+i)*BC+kk]*Vs[kk*D+col]; }
        acc[i][d]=a; }
    }
    __syncthreads();
  }
  #pragma unroll
  for(int i=0;i<TM;i++){ int q=q0+i; if(q<S){ float inv=1.f/l[i];
    for(int d=0;d<DACC;d++) O[(long)q*D + tj*DACC + d]=acc[i][d]*inv; } }
}
"""

def main():
    env = Env(work="/tmp/gpu-work/ablation")
    Q, K, V = deterministic_qkv(S, D); scale = 1.0/np.sqrt(D)
    ref = attention_reference(Q, K, V, scale)
    print(f"=== FLASH ABLATION LADDER (S={S} D={D}, P4) — each rung flips ONE knob ===")
    print(f"{'variant':<22}{'spill':<8}{'max_abs':<10}{'ok':<5}{'ms':<9}{'vs base'}")
    rows = []
    base_ms = None
    for name, src, kname, cfgfn in variants():
        local = os.path.join(env.work, name + ".cu")
        open(local, "w").write(src)
        cubin = build_cubin(env, local, ptxas_v=False)  # quiet; spill read below
        # read spill via a quiet recompile with -v
        r = env.run([f"{env.cuda}/bin/nvcc", "-arch=sm_61", "-cubin", "-O3",
                     f"-I{env.cuda}/include", "--ptxas-options=-v", local, "-o", local+".cubin"])
        spill = "0B"
        for ln in r.stderr.splitlines():
            if "stack frame" in ln:
                import re
                mm = re.search(r"(\d+) bytes stack frame", ln); spill = (mm.group(1)+"B") if mm else "?"
        if not cubin:
            print(f"{name:<22}BUILD-FAIL"); continue
        grid, block, shmem = cfgfn(S, D)
        args = [Q, K, V, "OUT", S, D, scale]
        try:
            out = run_kernel(env, cubin, kname, grid, block, args, out_idx=3, out_shape=(S, D), shmem=shmem)
        except Exception as e:
            print(f"{name:<22}RUN-FAIL {e}"); continue
        rep = verify(out, ref)
        ms = time_kernel(env, cubin, kname, grid, block, args, out_idx=3, out_shape=(S, D), shmem=shmem, iters=100)
        if base_ms is None: base_ms = ms
        ratio = base_ms/ms
        print(f"{name:<22}{spill:<8}{rep['max_abs']:<10.6f}{str(rep['within_tol']):<5}{ms:<9.4f}{ratio:.2f}x")
        rows.append((name, ms, rep['within_tol']))
    ok = [r for r in rows if r[2]]
    if ok:
        best = min(ok, key=lambda r: r[1])
        print(f"\nBEST (correct): {best[0]} = {best[1]:.4f} ms")
    return 0

if __name__ == "__main__":
    sys.exit(main())
