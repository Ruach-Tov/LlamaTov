%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════════
%% flash_attention.pl — emit the FUSED FlashAttention kernel from the recognized
%% attention chain (QK^T -> scale -> softmax -> xV), so the validated kernel comes
%% from the codegen path, NOT a hand-written file. Wired immediately to avoid
%% technical debt (Heath, 2026-06-08).
%%
%% THE FUSION: the attention chain collapses to ONE kernel via ONLINE SOFTMAX —
%% the O(seq^2) [seq x seq] score matrix is NEVER materialized. Each score is
%% computed, used (running max + running sum + accumulator rescale), and discarded.
%% Verified bit-exact vs torch + the unfused oracle (memory 2d514c64).
%%
%% recognize_attention(+Chain, -Spec): match [matmul, scale, softmax, matmul] (the
%%   QK^T->scale->softmax->xV diamond) -> flash_attn(Scale, Dim) spec.
%% emit_flash_attention(+Spec, +OutFile): generate the fused kernel.
%% Author: Iyun, 2026-06-08
%% ═══════════════════════════════════════════════════════════════════════════
:- module(flash_attention, [
    recognize_attention/2,        % recognize_attention(+Chain, -Spec)
    emit_flash_attention/2,       % emit_flash_attention(+Spec, +OutFile)  [naive default]
    emit_flash_attention/3,       % emit_flash_attention(+Spec, +Opts, +OutFile)
    flash_attn_schedule/2,        % flash_attn_schedule(+Name, -Schedule)
    emit_flash_schedule/4         % emit_flash_schedule(+Spec, +Schedule, +D, +OutFile)
]).
:- use_module(library(lists)).

%% ── recognize the attention diamond in a chain ──────────────────────────────
%% The QK^T->scale->softmax->xV pattern: two matmuls with a scale+softmax between.
%% Chain ops are kind atoms (matmul/gemm, scale/scaling, softmax, matmul/gemm).
recognize_attention(Chain, flash_attn(scaled, softmax)) :-
    match_kinds(Chain, [K1, Sc, Sm, K2]),
    matmul_kind(K1), scale_kind(Sc), softmax_kind(Sm), matmul_kind(K2).
%% also accept the scale folded (QK^T -> softmax_with_scale -> xV)
recognize_attention(Chain, flash_attn(scaled, softmax)) :-
    match_kinds(Chain, [K1, Sm, K2]),
    matmul_kind(K1), softmax_kind(Sm), matmul_kind(K2).

match_kinds(Chain, Kinds) :- maplist(op_kind_of, Chain, Kinds).
op_kind_of(Op, K) :- ( atom_concat(bpd_, K, Op) -> true ; K = Op ).

matmul_kind(K)  :- member(K, [matmul, gemm, mul_mat]).
scale_kind(K)   :- member(K, [scale, scaling]).
softmax_kind(K) :- member(K, [softmax, soft_max, soft_max_ext]).

%% ── emit the fused FlashAttention kernel ─────────────────────────────────────
emit_flash_attention(Spec, OutFile) :- emit_flash_attention(Spec, [], OutFile).

%% emit_flash_attention(+flash_attn(_,_), +Opts, +OutFile)
%% Opts: max_dim(D) — the head dim bound for the register accumulator (default 128).
%% Online-softmax FlashAttention, one thread per query row. The [SxS] scores are
%% never stored. Bit-exact-validated structure (memory 2d514c64).
emit_flash_attention(flash_attn(_ScaleMode, _SmKind), Opts, OutFile) :-
    ( member(max_dim(MaxD), Opts) -> true ; MaxD = 128 ),
    open(OutFile, write, S),
    format(S, "/* GENERATED FUSED FlashAttention (QK^T->scale->softmax->xV) — online softmax,~n", []),
    format(S, " * the [seq x seq] score matrix is NEVER materialized. One thread per query row.~n", []),
    format(S, " * Bit-exact vs torch + unfused oracle (within softmax fp contract). Iyun 2026-06-08. */~n", []),
    %% NO #include — expf/fmaxf are CUDA device intrinsics (available without <cmath>),
    %% keeping the kernel self-contained (compiles with bare nvcc, no -I needed).
    format(S, "extern \"C\" __global__ void k_flash_attn(const float* Q, const float* K, const float* V,~n", []),
    format(S, "                                        float* O, int S, int D, float scale) {~n", []),
    format(S, "  int q = blockIdx.x*blockDim.x + threadIdx.x;   // one query row per thread~n", []),
    format(S, "  if (q >= S) return;~n", []),
    format(S, "  const float* qrow = Q + (long)q*D;~n", []),
    format(S, "  float acc[~w];                                 // register accumulator (D <= ~w)~n", [MaxD, MaxD]),
    format(S, "  for (int d=0; d<D; d++) acc[d] = 0.0f;~n", []),
    format(S, "  float m = -3.4e38f, l = 0.0f;                  // running max, running sum~n", []),
    format(S, "  for (int j=0; j<S; j++) {                      // STREAM over keys — no [SxS] stored~n", []),
    format(S, "    const float* krow = K + (long)j*D;~n", []),
    format(S, "    float s = 0.0f; for (int d=0; d<D; d++) s += qrow[d]*krow[d];~n", []),
    format(S, "    s *= scale;~n", []),
    format(S, "    float m_new = fmaxf(m, s);~n", []),
    format(S, "    float corr = expf(m - m_new);                // rescale prior stats~n", []),
    format(S, "    float p = expf(s - m_new);~n", []),
    format(S, "    l = l*corr + p;~n", []),
    format(S, "    const float* vrow = V + (long)j*D;~n", []),
    format(S, "    for (int d=0; d<D; d++) acc[d] = acc[d]*corr + p*vrow[d];  // online accumulate~n", []),
    format(S, "    m = m_new;~n", []),
    format(S, "  }~n", []),
    format(S, "  float* orow = O + (long)q*D; float inv = 1.0f/l;~n", []),
    format(S, "  for (int d=0; d<D; d++) orow[d] = acc[d]*inv;~n", []),
    format(S, "}~n", []),
    %% machine-readable launch contract (the prevention applied)
    format(S, "// LAUNCH: thread_per_query total=S threads=S block<=1024~n", []),
    close(S),
    format("Generated FUSED FlashAttention kernel (max_dim=~w) -> ~w~n", [MaxD, OutFile]).

%% ═══════════════════════════════════════════════════════════════════════════
%% MOVE 1 (thesis fidelity): flash_attn_schedule — the perf-winning decisions as
%% a PROLOG SCHEDULE TERM (parallel to tiled_gemm). The 10.25x warp-cooperative +
%% shared-K/V + float4 kernel is now LOWERED FROM A SCHEDULE, not hand-written.
%% The autotuner can search flash schedules the same way it searches gemm tiles.
%% Reference kernel: bpd/kernelgen/runtime/flash_reference/flashv.cu (memory 01377050).
%% ═══════════════════════════════════════════════════════════════════════════
%% flash_attn_schedule(Name, schedule(flash, Levers)):
%%   warps_per_block(WPB)  — occupancy lever (16 = swept best, memory a73907d2)
%%   kv_tile(BC)           — block-shared K/V tile width (32 = swept best)
%%   d_split(warp)         — the D-dim accumulator split across the warp's 32 lanes
%%                           (DPL=D/32 floats/lane -> register-resident, NO spill)
%%   vectorize(float4)     — ld.shared.v4 (valid when DPL==4, i.e. D=128)
%%   shared_kv(true)       — K/V tiles loaded once/block, reused by all warps
flash_attn_schedule(tuned_d128,
    schedule(flash, [warps_per_block(16), kv_tile(32), d_split(warp),
                     vectorize(float4), shared_kv(true)])).
%% a scalar (non-vectorized) variant for D not divisible into clean float4 lanes
flash_attn_schedule(warp_shared,
    schedule(flash, [warps_per_block(16), kv_tile(32), d_split(warp),
                     vectorize(scalar), shared_kv(true)])).

%% emit_flash_schedule(+Spec, +Schedule, +D, +OutFile): lower the schedule to CUDA.
%% Generates the warp-cooperative + shared-K/V + (optionally float4) kernel that
%% the hand-written flashv.cu reproduces. The winning kernel, now schedule-derived.
emit_flash_schedule(flash_attn(_,_), schedule(flash, Levers), D, OutFile) :-
    memberchk(warps_per_block(WPB), Levers),
    memberchk(kv_tile(BC), Levers),
    ( memberchk(vectorize(float4), Levers), 0 is D mod 128 -> Vec = float4 ; Vec = scalar ),
    DPL is D // 32,
    open(OutFile, write, S),
    format(S, "/* GENERATED FUSED FlashAttention from flash_attn_schedule(WPB=~w BC=~w d_split(warp) vectorize(~w) shared_kv) — online softmax,~n", [WPB,BC,Vec]),
    format(S, " * warp-cooperative D-tiling (acc split across 32 lanes, DPL=~w, NO spill), block-shared K/V tiles.~n", [DPL]),
    format(S, " * The [seq x seq] scores NEVER materialized. Bit-exact vs torch+SDPA. Schedule-derived (Iyun 2026-06-08). */~n", []),
    format(S, "#define WPB ~w~n#define BC ~w~n", [WPB,BC]),
    format(S, "extern \"C\" __global__ void k_flash_attn(const float* Q,const float* K,const float* V,~n", []),
    format(S, "                                        float* O,int S,int D,float scale){~n", []),
    ( Vec == float4 -> emit_flash_vec_body(S) ; emit_flash_scalar_body(S, DPL) ),
    format(S, "}~n", []),
    format(S, "// LAUNCH: warp_per_query warps_per_block=~w block=~w shmem=2*~w*D*4~n", [WPB, WPB*32, BC]),
    close(S),
    format("Generated SCHEDULE-DERIVED FlashAttention (WPB=~w BC=~w ~w) -> ~w~n", [WPB,BC,Vec,OutFile]).

%% the float4 vectorized body (D=128, DPL=4 -> lane slice = 1 float4) — flashv.cu
emit_flash_vec_body(S) :-
    format(S, "  int lane=threadIdx.x&31, wid=threadIdx.x>>5; int q=blockIdx.x*WPB+wid;~n", []),
    format(S, "  extern __shared__ float sh[]; float* Ks=sh; float* Vs=sh+BC*D;~n", []),
    format(S, "  float4 qreg=(q<S)?*reinterpret_cast<const float4*>(Q+(long)q*D+lane*4):make_float4(0,0,0,0);~n", []),
    format(S, "  float4 acc=make_float4(0,0,0,0); float m=-3.4e38f,l=0.f; int tid=threadIdx.x,nth=WPB*32;~n", []),
    format(S, "  for(int k0=0;k0<S;k0+=BC){~n", []),
    format(S, "    for(int idx=tid; idx<BC*D/4; idx+=nth){ int e4=idx*4; int kr=e4/D,kc=e4%D; int gk=k0+kr;~n", []),
    format(S, "      float4 kv=(gk<S)?*reinterpret_cast<const float4*>(K+(long)gk*D+kc):make_float4(0,0,0,0);~n", []),
    format(S, "      float4 vv=(gk<S)?*reinterpret_cast<const float4*>(V+(long)gk*D+kc):make_float4(0,0,0,0);~n", []),
    format(S, "      *reinterpret_cast<float4*>(Ks+idx*4)=kv; *reinterpret_cast<float4*>(Vs+idx*4)=vv; }~n", []),
    format(S, "    __syncthreads();~n", []),
    format(S, "    if(q<S){ for(int kk=0;kk<BC;kk++){ int gk=k0+kk; if(gk>=S)break;~n", []),
    format(S, "      float4 ks=*reinterpret_cast<float4*>(Ks+kk*D+lane*4);~n", []),
    format(S, "      float pa=qreg.x*ks.x+qreg.y*ks.y+qreg.z*ks.z+qreg.w*ks.w;~n", []),
    format(S, "      for(int off=16;off>0;off>>=1) pa+=__shfl_down_sync(0xffffffff,pa,off);~n", []),
    format(S, "      float s=__shfl_sync(0xffffffff,pa,0)*scale;~n", []),
    format(S, "      float mn=fmaxf(m,s),corr=expf(m-mn),p=expf(s-mn); l=l*corr+p;~n", []),
    format(S, "      float4 vs=*reinterpret_cast<float4*>(Vs+kk*D+lane*4);~n", []),
    format(S, "      acc.x=acc.x*corr+p*vs.x; acc.y=acc.y*corr+p*vs.y; acc.z=acc.z*corr+p*vs.z; acc.w=acc.w*corr+p*vs.w;~n", []),
    format(S, "      m=mn; } }~n", []),
    format(S, "    __syncthreads();~n  }~n", []),
    format(S, "  if(q<S){ float inv=1.f/l; float4 o=make_float4(acc.x*inv,acc.y*inv,acc.z*inv,acc.w*inv);~n", []),
    format(S, "    *reinterpret_cast<float4*>(O+(long)q*D+lane*4)=o; }~n", []).

%% the scalar warp-cooperative body (general DPL) — flashws.cu structure
emit_flash_scalar_body(S, DPL) :-
    format(S, "  const int DPL=~w; int lane=threadIdx.x&31, wid=threadIdx.x>>5; int q=blockIdx.x*WPB+wid;~n", [DPL]),
    format(S, "  extern __shared__ float sh[]; float* Ks=sh; float* Vs=sh+BC*D;~n", []),
    format(S, "  float qreg[8]; for(int r=0;r<DPL;r++) qreg[r]=(q<S)?Q[(long)q*D+lane*DPL+r]:0.f;~n", []),
    format(S, "  float acc[8]; for(int r=0;r<DPL;r++) acc[r]=0.f; float m=-3.4e38f,l=0.f; int tid=threadIdx.x,nth=WPB*32;~n", []),
    format(S, "  for(int k0=0;k0<S;k0+=BC){~n", []),
    format(S, "    for(int idx=tid; idx<BC*D; idx+=nth){ int kr=idx/D,kc=idx%D; int gk=k0+kr;~n", []),
    format(S, "      Ks[idx]=(gk<S)?K[(long)gk*D+kc]:0.f; Vs[idx]=(gk<S)?V[(long)gk*D+kc]:0.f; }~n", []),
    format(S, "    __syncthreads();~n", []),
    format(S, "    if(q<S){ for(int kk=0;kk<BC;kk++){ int gk=k0+kk; if(gk>=S)break;~n", []),
    format(S, "      float pa=0.f; for(int r=0;r<DPL;r++) pa+=qreg[r]*Ks[kk*D+lane*DPL+r];~n", []),
    format(S, "      for(int off=16;off>0;off>>=1) pa+=__shfl_down_sync(0xffffffff,pa,off);~n", []),
    format(S, "      float s=__shfl_sync(0xffffffff,pa,0)*scale;~n", []),
    format(S, "      float mn=fmaxf(m,s),corr=expf(m-mn),p=expf(s-mn); l=l*corr+p;~n", []),
    format(S, "      for(int r=0;r<DPL;r++) acc[r]=acc[r]*corr+p*Vs[kk*D+lane*DPL+r];~n", []),
    format(S, "      m=mn; } }~n", []),
    format(S, "    __syncthreads();~n  }~n", []),
    format(S, "  if(q<S){ float inv=1.f/l; for(int r=0;r<DPL;r++) O[(long)q*D+lane*DPL+r]=acc[r]*inv; }~n", []).
