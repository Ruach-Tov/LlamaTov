// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.

// WARP-COOPERATIVE flash. One warp (32 lanes) per query row.
// acc split across lanes: lane holds acc[lane*DPL .. lane*DPL+DPL).
// MANY warps per block for occupancy: blockDim.x = 32*WARPS_PER_BLOCK.
extern "C" __global__ void k_flash_warp(const float* Q,const float* K,const float* V,
                                         float* O,int S,int D,float scale){
  const int DPL = D >> 5;                       // D/32 floats per lane
  int lane = threadIdx.x & 31;
  int warp = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;   // global warp id = query row
  int q = warp;
  if(q>=S) return;
  const float* qrow = Q + (long)q*D;
  // each lane loads its D/32 slice of the query into registers
  float qreg[8];                                 // DPL <= 8 (D<=256)
  #pragma unroll
  for(int r=0;r<DPL;r++) qreg[r] = qrow[lane*DPL + r];
  float acc[8];
  #pragma unroll
  for(int r=0;r<DPL;r++) acc[r] = 0.f;
  float m=-3.4e38f, l=0.f;
  for(int j=0;j<S;j++){                           // stream over keys
    const float* krow = K + (long)j*D;
    // partial dot over this lane's D/32 slice
    float partial=0.f;
    #pragma unroll
    for(int r=0;r<DPL;r++) partial += qreg[r]*krow[lane*DPL + r];
    // warp-reduce the dot across 32 lanes -> full score s (all lanes get it)
    #pragma unroll
    for(int off=16; off>0; off>>=1) partial += __shfl_down_sync(0xffffffff, partial, off);
    float s = __shfl_sync(0xffffffff, partial, 0) * scale;   // broadcast full score
    // online softmax (every lane computes the same m,l update — scalars, in regs)
    float mn = fmaxf(m, s);
    float corr = expf(m - mn);
    float p = expf(s - mn);
    l = l*corr + p;
    // each lane updates its acc slice with its V slice
    const float* vrow = V + (long)j*D;
    #pragma unroll
    for(int r=0;r<DPL;r++) acc[r] = acc[r]*corr + p*vrow[lane*DPL + r];
    m = mn;
  }
  float inv = 1.f/l;
  float* orow = O + (long)q*D;
  #pragma unroll
  for(int r=0;r<DPL;r++) orow[lane*DPL + r] = acc[r]*inv;
}
