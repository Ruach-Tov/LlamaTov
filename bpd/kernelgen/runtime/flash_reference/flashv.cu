// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.

#define WPB 16
#define BC 32
extern "C" __global__ void k_flash_vec(const float* Q,const float* K,const float* V,
                                        float* O,int S,int D,float scale){
  // assumes D=128, DPL=4 -> lane slice = 1 float4
  int lane = threadIdx.x & 31; int wid = threadIdx.x >> 5;
  int q = blockIdx.x*WPB + wid;
  extern __shared__ float sh[]; float* Ks=sh; float* Vs=sh+BC*D;
  float4 qreg = (q<S) ? *reinterpret_cast<const float4*>(Q + (long)q*D + lane*4)
                      : make_float4(0,0,0,0);
  float4 acc = make_float4(0,0,0,0);
  float m=-3.4e38f,l=0.f; int tid=threadIdx.x,nth=WPB*32;
  for(int k0=0;k0<S;k0+=BC){
    // cooperative load: BC*D floats = BC*D/4 float4s
    for(int idx=tid; idx<BC*D/4; idx+=nth){
      int e4=idx*4; int kr=e4/D, kc=e4%D; int gk=k0+kr;
      float4 kv = (gk<S) ? *reinterpret_cast<const float4*>(K+(long)gk*D+kc) : make_float4(0,0,0,0);
      float4 vv = (gk<S) ? *reinterpret_cast<const float4*>(V+(long)gk*D+kc) : make_float4(0,0,0,0);
      *reinterpret_cast<float4*>(Ks+idx*4)=kv; *reinterpret_cast<float4*>(Vs+idx*4)=vv;
    }
    __syncthreads();
    if(q<S){ for(int kk=0;kk<BC;kk++){int gk=k0+kk; if(gk>=S)break;
      float4 ks=*reinterpret_cast<float4*>(Ks+kk*D+lane*4);
      float pa = qreg.x*ks.x + qreg.y*ks.y + qreg.z*ks.z + qreg.w*ks.w;
      #pragma unroll
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
