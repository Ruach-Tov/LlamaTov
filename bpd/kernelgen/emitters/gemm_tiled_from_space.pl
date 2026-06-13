%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════════
%% gemm_tiled_from_space.pl — project a gpu_gemm_point(BM,BN,BK,TM,TN) to a
%% register-blocked shared-memory CUDA SGEMM kernel (the high-performance shape).
%%
%% Structure (classic register-blocked SGEMM, the one that closes on cuBLAS):
%%   - grid: (ceil(N/BN), ceil(N/BM)); block: (BN/TN)*(BM/TM) threads (1-D)
%%   - each block computes a BM x BN tile of C
%%   - each thread computes a TM x TN register micro-tile
%%   - K loop in BK strips: cooperatively + coalesced load A[BM x BK] and
%%     B[BK x BN] into shared memory; __syncthreads; then for each of BK steps,
%%     each thread loads TM A-values + TN B-values into registers and does
%%     TM*TN FMAs into its accumulator tile. Each smem value is reused TM (or
%%     TN) times from registers — that reuse is what lifts arithmetic intensity.
%%
%% fma_mode controls the inner accumulate (strict mul+add vs fused fmaf).
%% ABI: extern "C" __global__ void k_gemm(const float* A,const float* B,
%%      float* C, long n)  [square n x n], matching perf_fixture.
%%
%% Author: Iyun, 2026-06-07 (v2 register-blocked)
%% ═══════════════════════════════════════════════════════════════════════════

:- module(gemm_tiled_from_space, [emit_gemm_tiled/6, emit_gemm_tiled/7, emit_gemm_tiled/8, emit_gemm_tiled/9,
                                  emit_gemm_tiled_rect/8, emit_gemm_tiled_rect_splitk/8]).
:- use_module(library(lists)).

%% emit_gemm_tiled(+BM,+BN,+BK,+TM,+TN,+OutFile)  [fma_mode strict by default]
emit_gemm_tiled(BM, BN, BK, TM, TN, OutFile) :-
    emit_gemm_tiled(BM, BN, BK, TM, TN, strict, OutFile).

emit_gemm_tiled(BM, BN, BK, TM, TN, FmaMode, OutFile) :-
    emit_gemm_tiled(BM, BN, BK, TM, TN, FmaMode, 1, OutFile).

%% emit_gemm_tiled(+BM,+BN,+BK,+TM,+TN,+FmaMode,+VEC,+OutFile)
%% VEC in {1,4}: width of vectorized global->shared loads (float vs float4).
%% VEC=4 emits 128-bit float4 loads (4x fewer load instructions, better DRAM
%% throughput); requires BK%%4==0 (A inner dim) and BN%%4==0 (B inner dim) and
%% the runtime n%%4==0 (alignment) — guaranteed for our tile-multiple sweep sizes.
emit_gemm_tiled(BM, BN, BK, TM, TN, FmaMode, VEC, OutFile) :-
    emit_gemm_tiled(BM, BN, BK, TM, TN, FmaMode, VEC, 0, OutFile).

%% emit_gemm_tiled(+BM,+BN,+BK,+TM,+TN,+FmaMode,+VEC,+PIPE,+OutFile)
%% PIPE=1 -> DOUBLE-BUFFERED (software-pipelined): two shared buffers; while
%% computing strip k, prefetch strip k+1 from global into the other buffer, so
%% memory latency overlaps compute (one __syncthreads per strip instead of two).
emit_gemm_tiled(BM, BN, BK, TM, TN, FmaMode, VEC, 1, OutFile) :- !,
    emit_gemm_pipelined(BM, BN, BK, TM, TN, FmaMode, VEC, OutFile).
emit_gemm_tiled(BM, BN, BK, TM, TN, FmaMode, VEC, _, OutFile) :-
    NThreads is (BM // TM) * (BN // TN),
    mac(FmaMode, Mac),                  %% the inner FMA C-expression for this mode
    open(OutFile, write, S),
    format(S, "/* GENERATED register-blocked GEMM (single-buffer): BM=~w BN=~w BK=~w TM=~w TN=~w threads=~w fma=~w vec=~w */~n",
           [BM, BN, BK, TM, TN, NThreads, FmaMode, VEC]),
    format(S, "extern \"C\" __global__ void k_gemm(const float* A, const float* B, float* C, long n) {~n", []),
    format(S, "  const int BM=~w, BN=~w, BK=~w, TM=~w, TN=~w;~n", [BM, BN, BK, TM, TN]),
    format(S, "  __shared__ float As[BM*BK];  // BM x BK strip of A~n", []),
    format(S, "  __shared__ float Bs[BK*BN];  // BK x BN strip of B~n", []),
    format(S, "  int tid = threadIdx.x;~n", []),
    %% this thread's micro-tile position within the block tile
    format(S, "  int tRow = tid / (BN/TN);   // which TM-row group~n", []),
    format(S, "  int tCol = tid % (BN/TN);   // which TN-col group~n", []),
    %% block's top-left output coordinate
    format(S, "  int blockRow = blockIdx.y * BM;~n", []),
    format(S, "  int blockCol = blockIdx.x * BN;~n", []),
    %% accumulator micro-tile in registers
    format(S, "  float acc[TM][TN];~n", []),
    format(S, "  for(int i=0;i<TM;i++) for(int j=0;j<TN;j++) acc[i][j]=0.0f;~n", []),
    %% cooperative-load index helpers: linear tid covers As (BM*BK) and Bs (BK*BN)
    format(S, "  int nthreads = blockDim.x;~n", []),
    %% K loop in BK strips
    format(S, "  for(int k0=0; k0<n; k0+=BK) {~n", []),
    %% ── coalesced cooperative loads (scalar VEC=1 or float4 VEC=4) ──
    emit_load_A(S, VEC),
    emit_load_B(S, VEC),
    format(S, "    __syncthreads();~n", []),
    %% ── compute: for each of BK steps, load TM A-vals + TN B-vals to regs, FMA ──
    format(S, "    for(int kk=0; kk<BK; kk++) {~n", []),
    format(S, "      float a_reg[TM], b_reg[TN];~n", []),
    format(S, "      for(int i=0;i<TM;i++) a_reg[i] = As[(tRow*TM+i)*BK + kk];~n", []),
    format(S, "      for(int j=0;j<TN;j++) b_reg[j] = Bs[kk*BN + (tCol*TN+j)];~n", []),
    format(S, "      for(int i=0;i<TM;i++) for(int j=0;j<TN;j++) ~w;~n", [Mac]),
    format(S, "    }~n", []),
    format(S, "    __syncthreads();~n", []),
    format(S, "  }~n", []),
    %% ── write back the micro-tile (scalar, or float4 when VEC=4 & TN%4==0) ──
    emit_writeback(S, VEC, TN),
    format(S, "}~n", []),
    close(S),
    format("Generated register-blocked GEMM (BM=~w BN=~w BK=~w TM=~w TN=~w, ~w threads) -> ~w~n",
           [BM, BN, BK, TM, TN, NThreads, OutFile]).

%% ═══ DOUBLE-BUFFERED (software-pipelined) variant ═══════════════════════════
%% Two shared buffers As[2][..], Bs[2][..]. Prologue loads strip 0 into buf 0.
%% Each iteration: compute on the CURRENT buffer while the next strip's global
%% loads (issued before __syncthreads) stream into the OTHER buffer — the memory
%% latency overlaps the FMAs. One __syncthreads per strip (vs two single-buffered).
emit_gemm_pipelined(BM, BN, BK, TM, TN, FmaMode, VEC, OutFile) :-
    NThreads is (BM // TM) * (BN // TN),
    mac(FmaMode, Mac),
    open(OutFile, write, S),
    format(S, "/* GENERATED register-blocked GEMM (DOUBLE-BUFFERED): BM=~w BN=~w BK=~w TM=~w TN=~w threads=~w fma=~w vec=~w */~n",
           [BM, BN, BK, TM, TN, NThreads, FmaMode, VEC]),
    format(S, "extern \"C\" __global__ void k_gemm(const float* A, const float* B, float* C, long n) {~n", []),
    format(S, "  const int BM=~w, BN=~w, BK=~w, TM=~w, TN=~w;~n", [BM, BN, BK, TM, TN]),
    format(S, "  __shared__ float As[2][BM*BK];~n", []),
    format(S, "  __shared__ float Bs[2][BK*BN];~n", []),
    format(S, "  int tid=threadIdx.x, nthreads=blockDim.x;~n", []),
    format(S, "  int tRow=tid/(BN/TN), tCol=tid%(BN/TN);~n", []),
    format(S, "  int blockRow=blockIdx.y*BM, blockCol=blockIdx.x*BN;~n", []),
    format(S, "  float acc[TM][TN];~n", []),
    format(S, "  for(int i=0;i<TM;i++) for(int j=0;j<TN;j++) acc[i][j]=0.0f;~n", []),
    format(S, "  int buf=0;~n", []),
    emit_load_buf(S, VEC, "0", "0"),
    format(S, "  __syncthreads();~n", []),
    format(S, "  for(int k0=0; k0<n; k0+=BK) {~n", []),
    format(S, "    int nb = 1-buf; int k1 = k0+BK;~n", []),
    format(S, "    if(k1<n) {~n", []),
    emit_load_buf(S, VEC, "nb", "k1"),
    format(S, "    }~n", []),
    format(S, "    for(int kk=0; kk<BK; kk++) {~n", []),
    format(S, "      float a_reg[TM], b_reg[TN];~n", []),
    format(S, "      for(int i=0;i<TM;i++) a_reg[i]=As[buf][(tRow*TM+i)*BK+kk];~n", []),
    format(S, "      for(int j=0;j<TN;j++) b_reg[j]=Bs[buf][kk*BN+(tCol*TN+j)];~n", []),
    format(S, "      for(int i=0;i<TM;i++) for(int j=0;j<TN;j++) ~w;~n", [Mac]),
    format(S, "    }~n", []),
    format(S, "    __syncthreads();~n", []),
    format(S, "    buf = nb;~n", []),
    format(S, "  }~n", []),
    emit_writeback(S, VEC, TN),
    format(S, "}~n", []),
    close(S),
    format("Generated DOUBLE-BUFFERED GEMM (BM=~w BN=~w BK=~w TM=~w TN=~w vec=~w, ~w threads) -> ~w~n",
           [BM, BN, BK, TM, TN, VEC, NThreads, OutFile]).

%% ── C write-back: scalar (VEC=1, or TN not %4) vs float4 (VEC=4 & TN%4==0) ──
%% Each thread''s acc[i][0..TN-1] map to TN consecutive columns of C (row-major,
%% contiguous), so a row of the micro-tile is a clean float4 store when TN%4==0.
emit_writeback(S, VEC, TN) :-
    ( VEC =:= 4, 0 =:= TN mod 4 )
    -> %% float4 store: one 128-bit write per group of 4 columns
       format(S, "  for(int i=0;i<TM;i++) for(int j=0;j<TN;j+=4) {~n", []),
       format(S, "    int gr=blockRow+tRow*TM+i, gc=blockCol+tCol*TN+j;~n", []),
       format(S, "    float4 r=make_float4(acc[i][j],acc[i][j+1],acc[i][j+2],acc[i][j+3]);~n", []),
       format(S, "    if(gr<n && gc+3<n) *reinterpret_cast<float4*>(&C[(long)gr*n+gc])=r;~n", []),
       format(S, "    else for(int jj=0;jj<4;jj++){ if(gr<n&&gc+jj<n) C[(long)gr*n+gc+jj]=acc[i][j+jj]; }~n", []),
       format(S, "  }~n", [])
    ;  %% scalar store
       format(S, "  for(int i=0;i<TM;i++) for(int j=0;j<TN;j++) {~n", []),
       format(S, "    int gr=blockRow+tRow*TM+i, gc=blockCol+tCol*TN+j;~n", []),
       format(S, "    if(gr<n && gc<n) C[(long)gr*n+gc]=acc[i][j];~n", []),
       format(S, "  }~n", []).

%% load a K-strip at global k-offset Koff into shared buffer Buf (string exprs).
emit_load_buf(S, 1, Buf, Koff) :-
    format(S, "    for(int idx=tid; idx<BM*BK; idx+=nthreads){ int r=idx/BK,c=idx%BK; int gr=blockRow+r,gc=(~w)+c; As[~w][idx]=(gr<n&&gc<n)?A[(long)gr*n+gc]:0.0f; }~n", [Koff, Buf]),
    format(S, "    for(int idx=tid; idx<BK*BN; idx+=nthreads){ int r=idx/BN,c=idx%BN; int gr=(~w)+r,gc=blockCol+c; Bs[~w][idx]=(gr<n&&gc<n)?B[(long)gr*n+gc]:0.0f; }~n", [Koff, Buf]).
emit_load_buf(S, 4, Buf, Koff) :-
    format(S, "    for(int v=tid; v<(BM*BK)/4; v+=nthreads){ int idx=v*4; int r=idx/BK,c=idx%BK; int gr=blockRow+r,gc=(~w)+c; float4 t=*reinterpret_cast<const float4*>(&A[(long)gr*n+gc]); As[~w][idx]=t.x;As[~w][idx+1]=t.y;As[~w][idx+2]=t.z;As[~w][idx+3]=t.w; }~n", [Koff, Buf, Buf, Buf, Buf]),
    format(S, "    for(int v=tid; v<(BK*BN)/4; v+=nthreads){ int idx=v*4; int r=idx/BN,c=idx%BN; int gr=(~w)+r,gc=blockCol+c; float4 t=*reinterpret_cast<const float4*>(&B[(long)gr*n+gc]); Bs[~w][idx]=t.x;Bs[~w][idx+1]=t.y;Bs[~w][idx+2]=t.z;Bs[~w][idx+3]=t.w; }~n", [Koff, Buf, Buf, Buf, Buf]).

%% ── cooperative global->shared loads, scalar (VEC=1) or float4 (VEC=4) ──
%% A tile is BM x BK (inner dim BK contiguous in A''s row); B tile is BK x BN
%% (inner dim BN contiguous in B''s row). VEC=4 reads 4 contiguous floats as one
%% 128-bit float4 (4x fewer load instructions). No boundary mask: the sweep uses
%% tile-multiple sizes, so every tile is fully in-bounds (n %% BM/BN/BK == 0).
emit_load_A(S, 1) :-
    format(S, "    for(int idx=tid; idx<BM*BK; idx+=nthreads) {~n", []),
    format(S, "      int r=idx/BK, c=idx%BK; int gr=blockRow+r, gc=k0+c;~n", []),
    format(S, "      As[idx] = (gr<n && gc<n) ? A[(long)gr*n+gc] : 0.0f;~n", []),
    format(S, "    }~n", []).
emit_load_A(S, 4) :-
    format(S, "    for(int v=tid; v<(BM*BK)/4; v+=nthreads) {~n", []),
    format(S, "      int idx=v*4; int r=idx/BK, c=idx%BK; int gr=blockRow+r, gc=k0+c;~n", []),
    format(S, "      float4 t = *reinterpret_cast<const float4*>(&A[(long)gr*n+gc]);~n", []),
    format(S, "      As[idx]=t.x; As[idx+1]=t.y; As[idx+2]=t.z; As[idx+3]=t.w;~n", []),
    format(S, "    }~n", []).

emit_load_B(S, 1) :-
    format(S, "    for(int idx=tid; idx<BK*BN; idx+=nthreads) {~n", []),
    format(S, "      int r=idx/BN, c=idx%BN; int gr=k0+r, gc=blockCol+c;~n", []),
    format(S, "      Bs[idx] = (gr<n && gc<n) ? B[(long)gr*n+gc] : 0.0f;~n", []),
    format(S, "    }~n", []).
emit_load_B(S, 4) :-
    format(S, "    for(int v=tid; v<(BK*BN)/4; v+=nthreads) {~n", []),
    format(S, "      int idx=v*4; int r=idx/BN, c=idx%BN; int gr=k0+r, gc=blockCol+c;~n", []),
    format(S, "      float4 t = *reinterpret_cast<const float4*>(&B[(long)gr*n+gc]);~n", []),
    format(S, "      Bs[idx]=t.x; Bs[idx+1]=t.y; Bs[idx+2]=t.z; Bs[idx+3]=t.w;~n", []),
    format(S, "    }~n", []).

%% ═══ The inner multiply-accumulate expression: emit_fma is a KERNEL knob ═══════
%% emit_fma(true)  -> explicit fmaf(a,b,acc): ONE rounding, guaranteed FMA in SASS
%%                    independent of compiler flags. Faster + more accurate products.
%% emit_fma(false) -> acc + a*b: source-level separate mul+add. NOTE: nvcc default
%%                    -fmad=true STILL FUSES this to FFMA in SASS! To actually get
%%                    an unfused (two-rounding) kernel you ALSO need -fmad=false at
%%                    compile time. (Verified by cuobjdump: 'acc+a*b' default -> 128
%%                    FFMA; only -fmad=false -> 32 FADD+32 FMUL.)
%%
%% IMPORTANT — "strict" is NOT a kernel mode; it is a property of the REFERENCE we
%% verify against. A reference is defined by a tuple of facts:
%%   reference_fp(Name, fuses_fma=Bool, accumulate_precision=fp32|fp64, ...)
%% To bit-MATCH a given reference, choose the kernel's emit_fma (and -fmad, and
%% accumulator width) to mirror THAT reference's facts. e.g.:
%%   - torch-CPU oracle (no FMA fusion): emit_fma=false + compile -fmad=false
%%   - cuBLAS / nvcc-fused GPU reference: emit_fma=true
%% So "strict-vs-this-reference" is computed per-reference from its fp-facts, not
%% baked into the emitter as a single global mode. (See FP-contract memory 0036fe96.)
mac_fma(true,  "acc[i][j] = fmaf(a_reg[i], b_reg[j], acc[i][j])").
mac_fma(false, "acc[i][j] = acc[i][j] + a_reg[i] * b_reg[j]").

%% mac/2 — back-compat shim. Accepts the new boolean (true/false) OR the legacy
%% atoms (contract->fma, strict->no-fma) so existing callers keep working.
mac(true,     Expr) :- !, mac_fma(true, Expr).
mac(false,    Expr) :- !, mac_fma(false, Expr).
mac(contract, Expr) :- !, mac_fma(true, Expr).     %% legacy alias
mac(strict,   Expr) :- !, mac_fma(false, Expr).    %% legacy alias
mac(_,        Expr) :- mac_fma(false, Expr).        %% default: source mul+add

%% ═══ RECTANGULAR register-blocked GEMM: C[MxN] = A[MxK] * B[KxN] ═══════════════
%% emit_gemm_tiled_rect(+BM,+BN,+BK,+TM,+TN,+EmitFma,+VEC,+OutFile)
%% EmitFma: true (explicit fmaf) | false (source mul+add) — the KERNEL knob.
%%   (legacy atoms contract/strict still accepted via the mac/2 shim.) "Strict vs
%%   a reference" is decided per-reference from its fp-facts, NOT here.
%% The general case (square is M=N=K). Separate strides: A row-stride K, B row-
%% stride N, C row-stride N. Separate bounds: M rows, N cols, K reduction depth.
%% Full boundary guards (M,N,K need NOT be multiples of the tile — conv im2col
%% gives M=128,N=93312,K=576). Launch: grid.x=ceil(N/BN), grid.y=ceil(M/BM),
%% block=(BM/TM)*(BN/TN) threads. ABI: k_gemm(A,B,C, int M, int N, int K).
%% (VEC handled scalar here for boundary-safety; float4 needs aligned tiles —
%%  the rect case prioritizes shape-generality over the last vectorization step.)
emit_gemm_tiled_rect(BM, BN, BK, TM, TN, FmaMode, _VEC, OutFile) :-
    NThreads is (BM // TM) * (BN // TN),
    mac(FmaMode, Mac),
    open(OutFile, write, S),
    format(S, "/* GENERATED RECTANGULAR register-blocked GEMM C[MxN]=A[MxK]*B[KxN]: BM=~w BN=~w BK=~w TM=~w TN=~w threads=~w fma=~w */~n",
           [BM, BN, BK, TM, TN, NThreads, FmaMode]),
    %% EPILOGUE-FUSION hook (object-like macro — nvcc -D mangles function-macro
    %% commas). Operates on the in-scope 'v' (the C output) and 'gr' (output row =
    %% the M index, for per-row bias). Default identity. The head_fusion recognizer
    %% injects -DGEMM_EPILOGUE="<lowered elementwise tail>" so matmul->bias/silu/
    %% residual fuses into the C-store (the L3 transformer projections). MEASURED for
    %% conv/pool: long tails win 2.1x; matmul epilogue is the L3-critical case.
    format(S, "#ifndef GEMM_EPILOGUE~n#define GEMM_EPILOGUE (v)~n#endif~n", []),
    format(S, "extern \"C\" __global__ void k_gemm(const float* A, const float* B, float* C, int M, int N, int K) {~n", []),
    format(S, "  const int BM=~w, BN=~w, BK=~w, TM=~w, TN=~w;~n", [BM, BN, BK, TM, TN]),
    format(S, "  __shared__ float As[BM*BK];  // BM x BK strip of A (row-stride K)~n", []),
    format(S, "  __shared__ float Bs[BK*BN];  // BK x BN strip of B (row-stride N)~n", []),
    format(S, "  int tid = threadIdx.x, nthreads = blockDim.x;~n", []),
    format(S, "  int tRow = tid / (BN/TN);~n", []),
    format(S, "  int tCol = tid % (BN/TN);~n", []),
    format(S, "  int blockRow = blockIdx.y * BM;   // along M~n", []),
    format(S, "  int blockCol = blockIdx.x * BN;   // along N~n", []),
    format(S, "  float acc[TM][TN];~n", []),
    format(S, "  for(int i=0;i<TM;i++) for(int j=0;j<TN;j++) acc[i][j]=0.0f;~n", []),
    %% K-loop in BK strips, bound by K (the reduction depth)
    format(S, "  for(int k0=0; k0<K; k0+=BK) {~n", []),
    %% load A tile: BM rows x BK cols, A row-stride = K, guard gr<M && gc<K
    format(S, "    for(int idx=tid; idx<BM*BK; idx+=nthreads) {~n", []),
    format(S, "      int r=idx/BK, c=idx%BK; int gr=blockRow+r, gc=k0+c;~n", []),
    format(S, "      As[idx] = (gr<M && gc<K) ? A[(long)gr*K + gc] : 0.0f;~n", []),
    format(S, "    }~n", []),
    %% load B tile: BK rows x BN cols, B row-stride = N, guard gr<K && gc<N
    format(S, "    for(int idx=tid; idx<BK*BN; idx+=nthreads) {~n", []),
    format(S, "      int r=idx/BN, c=idx%BN; int gr=k0+r, gc=blockCol+c;~n", []),
    format(S, "      Bs[idx] = (gr<K && gc<N) ? B[(long)gr*N + gc] : 0.0f;~n", []),
    format(S, "    }~n", []),
    format(S, "    __syncthreads();~n", []),
    %% compute: BK steps, TM A-vals x TN B-vals per thread, FMA into acc
    format(S, "    for(int kk=0; kk<BK; kk++) {~n", []),
    format(S, "      float a_reg[TM], b_reg[TN];~n", []),
    format(S, "      for(int i=0;i<TM;i++) a_reg[i] = As[(tRow*TM+i)*BK + kk];~n", []),
    format(S, "      for(int j=0;j<TN;j++) b_reg[j] = Bs[kk*BN + (tCol*TN+j)];~n", []),
    format(S, "      for(int i=0;i<TM;i++) for(int j=0;j<TN;j++) ~w;~n", [Mac]),
    format(S, "    }~n", []),
    format(S, "    __syncthreads();~n", []),
    format(S, "  }~n", []),
    %% writeback: C row-stride = N, guard row<M && col<N
    format(S, "  for(int i=0;i<TM;i++) for(int j=0;j<TN;j++) {~n", []),
    format(S, "    int gr=blockRow+tRow*TM+i, gc=blockCol+tCol*TN+j;~n", []),
    format(S, "    if(gr<M && gc<N) { float v = acc[i][j]; C[(long)gr*N + gc] = GEMM_EPILOGUE; }~n", []),
    format(S, "  }~n", []),
    format(S, "}~n", []),
    close(S),
    format("Generated RECTANGULAR GEMM (BM=~w BN=~w BK=~w TM=~w TN=~w, ~w threads) -> ~w~n",
           [BM, BN, BK, TM, TN, NThreads, OutFile]).

%% ═══ SPLIT-K rectangular GEMM: deterministic workspace combine (NOT atomicAdd) ══
%% emit_gemm_tiled_rect_splitk(+BM,+BN,+BK,+TM,+TN,+SPLITS,+EmitFma,+OutFile)
%% cuBLAS's sgemm_largek strategy: divide K across SPLITS thread-block planes
%% (gridDim.z=SPLITS), each computes a PARTIAL sum over its K-stripe into
%% workspace[split*M*N + ...]. A separate k_splitk_reduce kernel then sums the
%% SPLITS partials into C in FIXED order — DETERMINISTIC, bit-exact, NO atomicAdd
%% (the one true non-determinism source we never emit; cf arxiv 2606.00279).
%% More parallelism for small-M shapes (conv-GEMM M=128) + cuBLAS's accuracy
%% structure (each accumulator sums K/SPLITS terms -> less magnitude spread).
%% ABI: k_gemm_splitk(A,B,Wspace, int M,int N,int K, int SPLITS)
%%      k_splitk_reduce(Wspace,C, int M,int N,int SPLITS)
%% Launch: gemm grid(ceil(N/BN),ceil(M/BM),SPLITS), block=NTH; reduce over M*N.
emit_gemm_tiled_rect_splitk(BM, BN, BK, TM, TN, _SPLITS, EmitFma, OutFile) :-
    NThreads is (BM // TM) * (BN // TN),
    mac(EmitFma, Mac),
    open(OutFile, write, S),
    format(S, "/* GENERATED SPLIT-K rect GEMM (deterministic workspace combine, no atomicAdd): BM=~w BN=~w BK=~w TM=~w TN=~w thr=~w */~n",
           [BM, BN, BK, TM, TN, NThreads]),
    %% ── the partial-GEMM kernel: each gridDim.z plane does a K-stripe ──
    format(S, "extern \"C\" __global__ void k_gemm_splitk(const float* A, const float* B, float* W, int M, int N, int K, int SPLITS) {~n", []),
    format(S, "  const int BM=~w, BN=~w, BK=~w, TM=~w, TN=~w;~n", [BM, BN, BK, TM, TN]),
    format(S, "  __shared__ float As[BM*BK];~n  __shared__ float Bs[BK*BN];~n", []),
    format(S, "  int tid=threadIdx.x, nthreads=blockDim.x;~n", []),
    format(S, "  int tRow=tid/(BN/TN), tCol=tid%(BN/TN);~n", []),
    format(S, "  int blockRow=blockIdx.y*BM, blockCol=blockIdx.x*BN;~n", []),
    format(S, "  int split=blockIdx.z;~n", []),
    %% this split owns K-range [k_lo, k_hi)
    format(S, "  int kchunk=(K + SPLITS - 1)/SPLITS;~n", []),
    format(S, "  int k_lo=split*kchunk, k_hi=k_lo+kchunk; if(k_hi>K) k_hi=K;~n", []),
    format(S, "  float acc[TM][TN];~n  for(int i=0;i<TM;i++)for(int j=0;j<TN;j++)acc[i][j]=0.0f;~n", []),
    format(S, "  for(int k0=k_lo; k0<k_hi; k0+=BK) {~n", []),
    format(S, "    for(int idx=tid; idx<BM*BK; idx+=nthreads){int r=idx/BK,c=idx%BK; int gr=blockRow+r,gc=k0+c;~n", []),
    format(S, "      As[idx]=(gr<M&&gc<k_hi)?A[(long)gr*K+gc]:0.0f;}~n", []),
    format(S, "    for(int idx=tid; idx<BK*BN; idx+=nthreads){int r=idx/BN,c=idx%BN; int gr=k0+r,gc=blockCol+c;~n", []),
    format(S, "      Bs[idx]=(gr<k_hi&&gc<N)?B[(long)gr*N+gc]:0.0f;}~n", []),
    format(S, "    __syncthreads();~n", []),
    format(S, "    for(int kk=0; kk<BK && (k0+kk)<k_hi; kk++){~n", []),
    format(S, "      float a_reg[TM], b_reg[TN];~n", []),
    format(S, "      for(int i=0;i<TM;i++) a_reg[i]=As[(tRow*TM+i)*BK+kk];~n", []),
    format(S, "      for(int j=0;j<TN;j++) b_reg[j]=Bs[kk*BN+(tCol*TN+j)];~n", []),
    format(S, "      for(int i=0;i<TM;i++)for(int j=0;j<TN;j++) ~w;~n", [Mac]),
    format(S, "    }~n    __syncthreads();~n  }~n", []),
    %% write this split's partial to W[split][M][N]
    format(S, "  long plane=(long)split*M*N;~n", []),
    format(S, "  for(int i=0;i<TM;i++)for(int j=0;j<TN;j++){~n", []),
    format(S, "    int gr=blockRow+tRow*TM+i, gc=blockCol+tCol*TN+j;~n", []),
    format(S, "    if(gr<M&&gc<N) W[plane + (long)gr*N + gc]=acc[i][j];~n", []),
    format(S, "  }~n}~n", []),
    %% ── the deterministic reduction kernel: sum SPLITS partials in fixed order ──
    format(S, "extern \"C\" __global__ void k_splitk_reduce(const float* W, float* C, int M, int N, int SPLITS) {~n", []),
    format(S, "  long idx=(long)blockIdx.x*blockDim.x+threadIdx.x;~n", []),
    format(S, "  long MN=(long)M*N; if(idx>=MN) return;~n", []),
    format(S, "  float s=0.0f;~n", []),
    format(S, "  for(int sp=0; sp<SPLITS; sp++) s += W[(long)sp*MN + idx];~n", []),
    format(S, "  C[idx]=s;~n}~n", []),
    close(S),
    format("Generated SPLIT-K rect GEMM (BM=~w BN=~w BK=~w TM=~w TN=~w) -> ~w~n",
           [BM, BN, BK, TM, TN, OutFile]).
