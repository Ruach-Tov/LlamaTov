// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.

#define WPB 8
#define BC 32
extern "C" __global__ void k_flash_warp_sh(const float* Q,const float* K,const float* V,
                                            float* O,int S,int D,float scale){
  const int DPL = D >> 5;                       // D/32 per lane
  int lane = threadIdx.x & 31;
  int wid  = threadIdx.x >> 5;                  // warp index in block [0,WPB)
  int q    = blockIdx.x*WPB + wid;              // this warp's query row
  extern __shared__ float sh[];                 // Ks[BC*D] then Vs[BC*D]
  float* Ks = sh; float* Vs = sh + BC*D;
  // load this warp's query slice (lane holds D/32)
  float qreg[8];
  #pragma unroll
  for(int r=0;r<DPL;r++) qreg[r] = (q<S) ? Q[(long)q*D + lane*DPL + r] : 0.f;
  float acc[8];
  #pragma unroll
  for(int r=0;r<DPL;r++) acc[r]=0.f;
  float m=-3.4e38f, l=0.f;
  int tid = threadIdx.x, nthreads = WPB*32;
  for(int k0=0;k0<S;k0+=BC){                     // K/V TILES
    // block cooperatively loads K-tile + V-tile into shared (once per block)
    for(int idx=tid; idx<BC*D; idx+=nthreads){
      int kr=idx/D, kc=idx%D; int gk=k0+kr;
      Ks[idx] = (gk<S) ? K[(long)gk*D+kc] : 0.f;
      Vs[idx] = (gk<S) ? V[(long)gk*D+kc] : 0.f;
    }
    __syncthreads();
    // each warp processes its query vs the BC keys in SHARED (warp-cooperative)
    if(q<S){
      for(int kk=0;kk<BC;kk++){
        int gk=k0+kk; if(gk>=S) break;
        float partial=0.f;
        #pragma unroll
        for(int r=0;r<DPL;r++) partial += qreg[r]*Ks[kk*D + lane*DPL + r];
        #pragma unroll
        for(int off=16;off>0;off>>=1) partial += __shfl_down_sync(0xffffffff,partial,off);
        float s = __shfl_sync(0xffffffff,partial,0) * scale;
        float mn=fmaxf(m,s); float corr=expf(m-mn); float p=expf(s-mn);
        l=l*corr+p;
        #pragma unroll
        for(int r=0;r<DPL;r++) acc[r]=acc[r]*corr + p*Vs[kk*D + lane*DPL + r];
        m=mn;
      }
    }
    __syncthreads();                             // before next tile overwrites shared
  }
  if(q<S){ float inv=1.f/l; for(int r=0;r<DPL;r++) O[(long)q*D + lane*DPL + r]=acc[r]*inv; }
}
