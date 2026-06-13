%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════════
%% lower_schedule_cuda.pl — lower a SCHEDULE (schedule_ir.pl) to a CUDA-C kernel.
%% Reproduces the proven tiled-reduce kernel from the schedule primitives, so the
%% schedule IR is validated against the existing 143 GB/s cuda-c reduce.
%% Author: Iyun, 2026-06-08
%% ═══════════════════════════════════════════════════════════════════════════
:- module(lower_schedule_cuda, [emit_schedule_cuda/3]).
:- use_module(library(lists)).
:- use_module(schedule_ir).

%% emit_schedule_cuda(+Op, +ScheduleName, +OutFile)
emit_schedule_cuda(Op, SchedName, OutFile) :-
    tile_schedule(Op, SchedName, schedule(reduce, Kind, _Prims)),
    cuda_comb(Kind, CombExpr),     % "acc + v" or "(v>acc)?v:acc"
    open(OutFile, write, S),
    format(S, "/* GENERATED from SCHEDULE-IR (tiled_row_reduce, ~w) -> CUDA-C. One schedule, N backends. */~n", [Kind]),
    format(S, "extern \"C\" __global__ void k_reduce(const float* x, float* out, int R, int C) {~n", []),
    %% block_map(r, R)
    format(S, "  int r = blockIdx.x;  if (r >= R) return;~n", []),
    format(S, "  int t = threadIdx.x;~n  const float* row = x + (long)r * C;~n", []),
    %% accumulate_strided
    sched_init(Kind, Init),
    format(S, "  float acc = ~w;~n", [Init]),
    format(S, "  for (int c = t; c < C; c += blockDim.x) { float v = row[c]; acc = ~w; }~n", [CombExpr]),
    %% warp_shuffle
    format(S, "  for (int o = 16; o > 0; o >>= 1) { float v = __shfl_down_sync(0xffffffff, acc, o); acc = ~w; }~n", [CombExpr]),
    %% stage_shared + barrier + warp0_reduce
    format(S, "  __shared__ float sh[32];~n  int lane = t & 31, wid = t >> 5;~n", []),
    format(S, "  if (lane == 0) sh[wid] = acc;~n  __syncthreads();~n", []),
    format(S, "  if (wid == 0) {~n", []),
    format(S, "    acc = (t < (blockDim.x + 31) / 32) ? sh[lane] : (~w);~n", [Init]),
    format(S, "    for (int o = 16; o > 0; o >>= 1) { float v = __shfl_down_sync(0xffffffff, acc, o); acc = ~w; }~n", [CombExpr]),
    %% finalize (mean divides) + guarded_store
    sched_final_cuda(Kind, FinalExpr),
    format(S, "    if (lane == 0) out[r] = ~w;~n", [FinalExpr]),
    format(S, "  }~n}~n", []),
    close(S),
    format("Generated SCHEDULE-IR CUDA reduce (~w) -> ~w~n", [Kind, OutFile]).

%% the combine as a CUDA C-expression on (acc, v)
cuda_comb(sum,  "acc + v").
cuda_comb(mean, "acc + v").
cuda_comb(max,  "(v > acc) ? v : acc").
cuda_comb(min,  "(v < acc) ? v : acc").

sched_init(Kind, I) :- schedule_combine(Kind, I, _).

sched_final_cuda(mean, "acc / C") :- !.
sched_final_cuda(_,    "acc").

%% ── tiled_gemm schedule -> CUDA-C (reproduces the tuned rect GEMM) ───────────
%% emit_schedule_cuda(Op, tiled_gemm(BM,BN,BK,TM,TN), OutFile): lower the GEMM
%% schedule's primitives to the register-blocked shared-memory kernel. Validation:
%% should reproduce emit_gemm_tiled_rect FUNCTIONALLY (same tiling, same epilogue
%% hook). One schedule -> this cuda-c kernel AND (next) the hand-written MLIR.
emit_schedule_cuda(Op, tiled_gemm(BM,BN,BK,TM,TN), OutFile) :-
    tile_schedule(Op, tiled_gemm(BM,BN,BK,TM,TN), schedule(gemm, _Contract, _Prims)),
    NThreads is (BM // TM) * (BN // TN),
    open(OutFile, write, S),
    format(S, "/* GENERATED from SCHEDULE-IR tiled_gemm(BM=~w BN=~w BK=~w TM=~w TN=~w, ~w threads) -> CUDA-C. */~n",
           [BM, BN, BK, TM, TN, NThreads]),
    format(S, "#ifndef GEMM_EPILOGUE~n#define GEMM_EPILOGUE (v)~n#endif~n", []),
    format(S, "extern \"C\" __global__ void k_gemm(const float* A, const float* B, float* C, int M, int N, int K) {~n", []),
    format(S, "  const int BM=~w, BN=~w, BK=~w, TM=~w, TN=~w;~n", [BM, BN, BK, TM, TN]),
    %% stage_shared(as_tile) + stage_shared(bs_tile)
    format(S, "  __shared__ float As[BM*BK];~n  __shared__ float Bs[BK*BN];~n", []),
    format(S, "  int tid = threadIdx.x, nthreads = blockDim.x;~n", []),
    %% thread_tile(TM,TN): thread owns a TMxTN micro-tile
    format(S, "  int tRow = tid / (BN/TN);~n  int tCol = tid % (BN/TN);~n", []),
    %% block_map_2d(bi,bj): one block per BMxBN output tile
    format(S, "  int blockRow = blockIdx.y * BM;~n  int blockCol = blockIdx.x * BN;~n", []),
    %% register_init(acc,TM,TN,0)
    format(S, "  float acc[TM][TN];~n  for(int i=0;i<TM;i++) for(int j=0;j<TN;j++) acc[i][j]=0.0f;~n", []),
    %% k_loop(BK): stage tiles, barrier, accumulate, barrier
    format(S, "  for(int k0=0; k0<K; k0+=BK) {~n", []),
    format(S, "    for(int idx=tid; idx<BM*BK; idx+=nthreads) {~n", []),
    format(S, "      int r=idx/BK, c=idx%BK; int gr=blockRow+r, gc=k0+c;~n", []),
    format(S, "      As[idx] = (gr<M && gc<K) ? A[(long)gr*K + gc] : 0.0f;~n    }~n", []),
    format(S, "    for(int idx=tid; idx<BK*BN; idx+=nthreads) {~n", []),
    format(S, "      int r=idx/BN, c=idx%BN; int gr=k0+r, gc=blockCol+c;~n", []),
    format(S, "      Bs[idx] = (gr<K && gc<N) ? B[(long)gr*N + gc] : 0.0f;~n    }~n", []),
    format(S, "    __syncthreads();~n", []),
    %% register_accumulate(acc, as_tile, bs_tile, TM, TN, BK)
    format(S, "    for(int kk=0; kk<BK; kk++) {~n", []),
    format(S, "      float a_reg[TM], b_reg[TN];~n", []),
    format(S, "      for(int i=0;i<TM;i++) a_reg[i] = As[(tRow*TM+i)*BK + kk];~n", []),
    format(S, "      for(int j=0;j<TN;j++) b_reg[j] = Bs[kk*BN + (tCol*TN+j)];~n", []),
    format(S, "      for(int i=0;i<TM;i++) for(int j=0;j<TN;j++) acc[i][j] = fmaf(a_reg[i], b_reg[j], acc[i][j]);~n", []),
    format(S, "    }~n    __syncthreads();~n  }~n", []),
    %% guarded_store_2d(c_out, acc, ..., epilogue)
    format(S, "  for(int i=0;i<TM;i++) for(int j=0;j<TN;j++) {~n", []),
    format(S, "    int gr=blockRow+tRow*TM+i, gc=blockCol+tCol*TN+j;~n", []),
    format(S, "    if(gr<M && gc<N) { float v = acc[i][j]; C[(long)gr*N + gc] = GEMM_EPILOGUE; }~n", []),
    format(S, "  }~n}~n", []),
    close(S),
    format("Generated SCHEDULE-IR tiled GEMM CUDA (~wx~w tile) -> ~w~n", [BM, BN, OutFile]).
