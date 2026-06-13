// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* conv2d via IMPLICIT GEMM (im2col fused into the GEMM B-load) for the P4.
 *
 * The im2col + gemm_rect pipeline materialized a 215MB col[K, Nn] matrix
 * (5.57ms of pure memory traffic). This kernel fuses them: the GEMM's
 * shared-memory B-tile is filled by COMPUTING each col entry on the fly from
 * x (the im2col index map), never writing col to global memory.
 *
 * out_mat[Cout, Nn] = w_mat[Cout, K] @ col[K, Nn]
 *   M = Cout,  K = Cin*KH*KW,  Nn = N*Hout*Wout
 *   col[k, p] = x[n, ic, oh+kh, ow+kw]
 *     k = (ic*KH + kh)*KW + kw  ;  p = (n*Hout + oh)*Wout + ow
 * The relayout is also fused into the C-store: we write directly to NCHW.
 */
/* Tile config — overridable via -DBM=.. etc. Default = the rect-autotuned best
 * for the conv-GEMM shape (M=128,N=93312,K=576): BM128 BN128 BK32 TM8 TN4 =
 * 1479 GFLOPS = 2.3x the old BM64/BN64/BK16 fixed tile. (gemm_sweep_rect found it.)
 * shared mem at this tile: (BM*BK + BK*BN)*4 = (128*32 + 32*128)*4 = 32KB < 48KB. */
#ifndef BM
#define BM 128
#endif
#ifndef BN
#define BN 128
#endif
#ifndef BK
#define BK 32
#endif
#ifndef TM
#define TM 8
#endif
#ifndef TN
#define TN 4
#endif
/* __launch_bounds__(maxThreads, minBlocksPerSM): the 2nd arg forces nvcc to CAP
 * registers so minBlocksPerSM fit on an SM -> higher occupancy. Overridable via
 * -DLB_THREADS -DLB_BLOCKS. Default off (0,0) = no bound. (Mavchin's occupancy
 * experiment: cap regs, re-profile, watch exec-dependency fall as occupancy rises.) */
/* Default __launch_bounds__(512, 2): cap regs to 64 so 2 blocks fit per SM (sm_61
 * 65536 regs/SM). MEASURED: this is the occupancy sweet spot — exec-dependency
 * 36.9%->28.1%, 15.69ms->12.59ms = 1.25x faster than uncapped (128 regs, 1 block/SM).
 * (512,3) overshoots: caps to 40 regs -> register SPILLING -> mem-dep 52%, 10x slower.
 * The floor is 2 blocks/SM. (Mavchin's occupancy experiment, CUPTI-confirmed.) */
#ifndef LB_THREADS
#define LB_THREADS 512
#endif
#ifndef LB_BLOCKS
#define LB_BLOCKS 2
#endif
/* Epilogue-fusion hook: default identity (un-fused). Object-like macro (NOT
 * function-like — nvcc's -D mangles the comma in (v,oc)). Operates on the
 * in-scope 'v' (conv output) and 'oc' (output channel). The fusion recognizer
 * injects -DCONV_EPILOGUE="<lowered elementwise tail over v,oc>" to fuse the
 * activation into the C-store. E.g. relu -> ((v!=v)?v:((v>=0)?v:0)). Defined
 * BEFORE the launch_bounds #if so BOTH branches see it. */
#ifndef CONV_EPILOGUE
#define CONV_EPILOGUE (v)
#endif
#if LB_BLOCKS > 0
extern "C" __global__ void __launch_bounds__(LB_THREADS, LB_BLOCKS) k_conv_implicit(
#else
extern "C" __global__ void k_conv_implicit(
#endif
    const float* __restrict__ x, const float* __restrict__ wmat, float* __restrict__ out,
    int N, int Cin, int H, int W, int Cout, int KH, int KW, int Hout, int Wout) {
  const long K  = (long)Cin*KH*KW;
  const long Nn = (long)N*Hout*Wout;
  const int M = Cout;
  __shared__ float As[BM][BK];   // w_mat tile  [Cout rows x K]
  __shared__ float Bs[BK][BN];   // col   tile  [K    x Nn] — COMPUTED, not loaded
  int brow = blockIdx.y * BM, bcol = blockIdx.x * BN;
  int nthreads = (BM/TM)*(BN/TN);
  int tx = threadIdx.x;
  int trow = (tx / (BN/TN)) * TM;
  int tcol = (tx % (BN/TN)) * TN;
  float acc[TM][TN];
  #pragma unroll
  for(int i=0;i<TM;i++) for(int j=0;j<TN;j++) acc[i][j]=0.0f;

  for (int k0=0;k0<K;k0+=BK){
    // load w_mat tile (a normal global read)
    for(int i=tx;i<BM*BK;i+=nthreads){int r=i/BK,cc=i%BK; long gr=brow+r,gc=k0+cc;
      As[r][cc]=(gr<M&&gc<K)?wmat[gr*K+gc]:0.0f;}
    // FUSED im2col: compute Bs[r][cc] = col[k0+r, bcol+cc] directly from x
    for(int i=tx;i<BK*BN;i+=nthreads){int r=i/BN,cc=i%BN;
      long k = k0+r, p = bcol+cc;
      float v = 0.0f;
      if (k<K && p<Nn){
        int kw = k % KW; long kk = k / KW; int kh = kk % KH; int ic = kk / KH;
        int ow = p % Wout; long t = p / Wout; int oh = t % Hout; int n = t / Hout;
        int ih = oh + kh, iw = ow + kw;        // stride1 pad0 dil1
        v = x[(((long)n*Cin + ic)*H + ih)*W + iw];
      }
      Bs[r][cc]=v;
    }
    __syncthreads();
    #pragma unroll
    for(int kk=0;kk<BK;++kk){
      float ar[TM],br[TN];
      #pragma unroll
      for(int i=0;i<TM;i++) ar[i]=As[trow+i][kk];
      #pragma unroll
      for(int j=0;j<TN;j++) br[j]=Bs[kk][tcol+j];
      #pragma unroll
      for(int i=0;i<TM;i++) for(int j=0;j<TN;j++) acc[i][j]+=ar[i]*br[j];
    }
    __syncthreads();
  }
  // FUSED relayout: C[oc, p] with p=(n*Hout+oh)*Wout+ow -> out[n,oc,oh,ow]
  #pragma unroll
  for(int i=0;i<TM;i++) for(int j=0;j<TN;j++){
    int oc = brow+trow+i; long p = bcol+tcol+j;
    if(oc<M && p<Nn){
      long HW=(long)Hout*Wout; int n = p / HW; long hw = p % HW;
      // EPILOGUE FUSION hook: CONV_EPILOGUE(v, oc) is the lowered elementwise
      // tail applied to the conv output BEFORE the global store — eliminates the
      // separate activation kernel + its full-tensor round-trip. Default = identity
      // (un-fused). The fusion recognizer injects -DCONV_EPILOGUE=... with the
      // composed tail term lowered to a C-expr (v = acc, oc = output channel for
      // per-channel bias). One store, fused.
      float v = acc[i][j];
      out[((long)n*Cout + oc)*HW + hw] = CONV_EPILOGUE;
    }
  }
}
