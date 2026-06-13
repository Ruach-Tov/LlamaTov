%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════════
%% q8_0_from_facts.pl — emit the Q8_0 quantized dot product FROM its op_expr fact.
%%
%% The Q8_0 dot is the novel piece of fact-driven llama-Q8_0: a quantized GEMV
%% where each 32-element block is (1 fp16 scale d) + (32 int8 quants q). The dot
%% of two Q8_0-quantized vectors is, per block:
%%     block_dot = (xd * yd) * SUM_i( xq[i] * yq[i] )        [int32 accumulate, fp scale]
%% and the full dot sums block_dots. This is ALGEBRAICALLY the dequant-then-dot
%% (validated bit-exact, memory 4727ad11) — so the emitter is verified vs dequant-dot.
%%
%% Two backend lowerings (matching the prior gen_q8dot_ll.py variants):
%%   scalar — a plain int32 accumulation loop (portable, every arch)
%%   dp4a   — uses __dp4a (4-way int8 dot) — Pascal sm_61+ has it (the P4 native int8 path)
%%
%% op_expr(bpd_q8_0_dot, q8_0_dot(block(32), scale(d), quant(int8))).
%% emit_q8_0_dot(+Mode, +OutFile)   Mode = scalar | dp4a
%% emit_from_fact(+OpExpr, +Opts, +OutFile)
%% Author: Iyun, 2026-06-08
%% ═══════════════════════════════════════════════════════════════════════════
:- module(q8_0_from_facts, [
    q8_0_op_expr/1,         % q8_0_op_expr(-Expr)  [the fact shape]
    emit_q8_0_dot/2,        % emit_q8_0_dot(+Mode, +OutFile)
    emit_from_fact/3        % emit_from_fact(+OpExpr, +Opts, +OutFile)
]).
:- use_module(library(lists)).

%% the Q8_0 dot fact: per-block 32 elems, fp16 scale, int8 quants.
%% Pure-math AST: scale_dot(mul(xd,yd), int_dot(xq,yq)) over blocks.
q8_0_op_expr(q8_0_dot(block(32), scale(fp16), quant(int8))).

%% emit_from_fact: dispatch on the q8_0_dot fact (default dp4a — the P4 native path).
%% With epilogue(Chain) in Opts, FUSE the elementwise tail into the GEMV store — one
%% kernel for the recognized chain (q8_0_dot -> silu / bias / ...), no intermediate
%% activation leaving the kernel. The epilogue C-expr is derived via epilogue_fusion.
%% With prologue(quant) in Opts, FUSE the upstream activation quantize INTO the GEMV load —
%% the f32 activation is quantized into shared memory (same arithmetic as standalone
%% k_quant_q8), no global round-trip of the quantized activation. Symmetric to epilogue
%% fusion: epilogue fuses the downstream store, prologue fuses the upstream load.
emit_from_fact(q8_0_dot(block(_), _, _), Opts, OutFile) :-
    member(prologue(quant), Opts), !,
    ( member(epilogue(EpiC), Opts) -> true ; EpiC = "acc" ),
    emit_q8_0_dot_prologue_quant(EpiC, OutFile).
emit_from_fact(q8_0_dot(block(_), _, _), Opts, OutFile) :-
    member(epilogue(add_residual), Opts), !,
    emit_q8_0_dot_add_residual(OutFile).
emit_from_fact(q8_0_dot(block(_), _, _), Opts, OutFile) :-
    member(mode(tiled(BM, BK, VEC)), Opts), !,
    emit_q8_0_gemv_tiled(BM, BK, VEC, OutFile).
emit_from_fact(q8_0_dot(block(_), _, _), Opts, OutFile) :-
    member(mode(tiled_v4(BM, BK)), Opts), !,
    emit_q8_0_gemv_tiled_v4(BM, BK, OutFile).
emit_from_fact(q8_0_dot(block(_), _, _), Opts, OutFile) :-
    member(mode(tiled_v4_reghoist(BM, BK)), Opts), !,
    emit_q8_0_gemv_tiled_v4_reghoist(BM, BK, OutFile).
emit_from_fact(q8_0_dot(block(_), _, _), Opts, OutFile) :-
    member(mode(tiled_v4_qfused(BM, BK)), Opts), !,
    emit_q8_0_gemv_tiled_v4_qfused(BM, BK, OutFile).
emit_from_fact(q8_0_dot(block(_), _, _), Opts, OutFile) :-
    member(mode(tiled_v4_addres(BM, BK)), Opts), !,
    emit_q8_0_gemv_tiled_v4_addres(BM, BK, OutFile).
emit_from_fact(q8_0_dot(block(_), _, _), Opts, OutFile) :-
    member(mode(tiled_v4_silu(BM, BK)), Opts), !,
    emit_q8_0_gemv_tiled_v4_silu(BM, BK, OutFile).
emit_from_fact(q8_0_dot(block(_), _, _), Opts, OutFile) :-
    member(mode(canonical_serial_gemv), Opts), !,
    emit_q8_0_gemv_canonical_serial(OutFile).
emit_from_fact(q8_0_dot(block(_), _, _), Opts, OutFile) :- !,
    ( member(mode(M), Opts) -> true ; M = dp4a ),
    ( member(epilogue(EpiC), Opts) -> true ; EpiC = "acc" ),
    emit_q8_0_dot_epi(M, EpiC, OutFile).

%% non-fused variants delegate to the epilogue form with identity (acc).
emit_q8_0_dot(M, OutFile) :- emit_q8_0_dot_epi(M, "acc", OutFile).

%% emit_q8_0_dot(+Mode, +OutFile): generate the Q8_0 GEMV kernel.
%% Computes y[row] = sum over K-blocks of block_dot(W_block[row], X_block).
%% W is the Q8_0 weight (per-row blocks), X the Q8_0-quantized activation.
emit_q8_0_dot_epi(scalar, EpiC, OutFile) :-
    open(OutFile, write, S),
    q8_header(S, "scalar (int32 accumulation loop, portable)"),
    BlockSz = 64, format(S, "extern \"C\" __global__ void __launch_bounds__(~w, 4) k_q8_0_gemv(~n", [BlockSz]),
    format(S, "    const signed char* Wq, const __half* Wd,   // weight: [M*K/32 blocks] int8 + fp16 scales~n", []),
    format(S, "    const signed char* Xq, const __half* Xd,   // activation: [K/32 blocks] int8 + scales~n", []),
    format(S, "    float* Y, int M, int K) {                  // Y[M], K = cols (mult of 32)~n", []),
    format(S, "  int row = blockIdx.x*blockDim.x + threadIdx.x; if (row >= M) return;~n", []),
    format(S, "  int nblk = K / 32; float acc = 0.0f;~n", []),
    format(S, "  for (int b = 0; b < nblk; b++) {~n", []),
    format(S, "    const signed char* wq = Wq + (long)row*K + b*32;~n", []),
    format(S, "    const signed char* xq = Xq + b*32;~n", []),
    format(S, "    int isum = 0;~n", []),
    format(S, "    for (int i = 0; i < 32; i++) isum += (int)wq[i] * (int)xq[i];  // int32 accumulate~n", []),
    format(S, "    float wd = __half2float(Wd[(long)row*nblk + b]);~n", []),
    format(S, "    float xd = __half2float(Xd[b]);~n", []),
    format(S, "    acc += (wd * xd) * (float)isum;            // fp scale per block~n", []),
    format(S, "  }~n", []),
    format(S, "  Y[row] = ~w;   // fused epilogue over acc~n", [EpiC]),
    format(S, "}~n", []),
    format(S, "// LAUNCH: thread_per_row total=M~n", []),
    close(S),
    format("Generated FACT-DERIVED Q8_0 GEMV (scalar, epi=~w) -> ~w~n", [EpiC, OutFile]).

emit_q8_0_dot_epi(dp4a, EpiC, OutFile) :-
    open(OutFile, write, S),
    q8_header(S, "dp4a (Pascal __dp4a 4-way int8 dot, sm_61+ native int8 path)"),
    BlockSz = 64, format(S, "extern \"C\" __global__ void __launch_bounds__(~w, 4) k_q8_0_gemv(~n", [BlockSz]),
    format(S, "    const signed char* Wq, const __half* Wd,~n", []),
    format(S, "    const signed char* Xq, const __half* Xd,~n", []),
    format(S, "    float* Y, int M, int K) {~n", []),
    format(S, "  int row = blockIdx.x*blockDim.x + threadIdx.x; if (row >= M) return;~n", []),
    format(S, "  int nblk = K / 32;~n", []),
    emit_dp4a_accum_body(S, "Xq", "Xd"),   % <-- shared accumulation body (load from global Xq/Xd)
    format(S, "  Y[row] = ~w;   // fused epilogue over acc~n", [EpiC]),
    format(S, "}~n", []),
    format(S, "// LAUNCH: thread_per_row total=M~n", []),
    close(S),
    format("Generated FACT-DERIVED Q8_0 GEMV (dp4a, epi=~w) -> ~w~n", [EpiC, OutFile]).

%% emit_from_fact with epilogue(add_residual): FUSE the downstream residual-add INTO the GEMV
%% store. The standalone k_add (y[i] = a[i] + b[i]) after o-proj/down-proj is eliminated; the
%% fused kernel adds the residual at STORE time: Y[row] = acc + Resid[row]. Eliminates the
%% k_add launch + its global round-trip (read GEMV output, read residual, write sum), 127x/token.
%% BIT-EXACT by construction: the accumulation `acc` is the SAME emitted body
%% (emit_dp4a_accum_body), and `+ Resid[row]` is a SINGLE f32 add (the same one k_add does) —
%% no reduction-tree, no FMA-fusion ambiguity (a lone binary add can't be re-associated). So the
%% float DAG is identical to unfused gemv-then-add. Symmetric to prologue(quant): prologue fuses
%% the upstream quantize-load, this fuses the downstream residual-store.
emit_q8_0_dot_add_residual(OutFile) :-
    open(OutFile, write, S),
    q8_header(S, "dp4a + residual-add EPILOGUE fused (Y = gemv(x) + residual, no k_add round-trip)"),
    BlockSz = 64, format(S, "extern \"C\" __global__ void __launch_bounds__(~w, 4) k_q8_0_gemv_addres(~n", [BlockSz]),
    format(S, "    const signed char* Wq, const __half* Wd,~n", []),
    format(S, "    const signed char* Xq, const __half* Xd,~n", []),
    format(S, "    const float* Resid,~n", []),          % <-- the residual to add at store
    format(S, "    float* Y, int M, int K) {~n", []),
    format(S, "  int row = blockIdx.x*blockDim.x + threadIdx.x; if (row >= M) return;~n", []),
    format(S, "  int nblk = K / 32;~n", []),
    emit_dp4a_accum_body(S, "Xq", "Xd"),   % <-- SAME accumulation body -> bit-exact float side
    format(S, "  Y[row] = acc + Resid[row];   // fused residual-add epilogue (single f32 add)~n", []),
    format(S, "}~n", []),
    format(S, "// LAUNCH: thread_per_row total=M~n", []),
    close(S),
    format("Generated FACT-DERIVED Q8_0 GEMV (dp4a + residual-add epilogue) -> ~w~n", [OutFile]).

%% ── TILED Q8_0 GEMV (projected from gpu_gemv_point(BM,BK,VEC)) ──────────────
%% The high-reuse GEMV shape: a BLOCK of BM warps co-processes BM output rows,
%% staging the activation (Xq/Xd) in SHARED MEMORY once and reusing it across all
%% BM rows. Each warp (32 lanes) cooperatively computes one row's dp4a: the lanes
%% split the nblk K-blocks, each accumulates partial dp4a into a per-lane int sum,
%% then a warp-shuffle tree reduces to the row result. This lifts L2/coalescing
%% reuse (the measured gap vs ggml). BM is the key reuse axis; BK strips the K
%% dim into shared; VEC sets the per-step int32-word load width.
%% NOTE: the warp-shuffle reduction changes the FLOAT accumulation ORDER vs the
%% serial thread-per-row dp4a -> NOT bit-exact to the plain kernel; the equivalence
%% class is MEASURED per (BM,BK,VEC) point by the sweep (and a canonical-order
%% variant can be declared later, as we did for rmsnorm, if a point wins).
emit_q8_0_gemv_tiled(BM, BK, VEC, OutFile) :-
    open(OutFile, write, S),
    q8_header(S, "TILED Q8_0 GEMV (BM warps/block share shared-mem activation; warp-reduce dp4a)"),
    format(S, "// gpu_gemv_point(BM=~w, BK=~w, VEC=~w)~n", [BM, BK, VEC]),
    BlockSz is BM*32, format(S, "extern \"C\" __global__ void __launch_bounds__(~w, 4) k_q8_0_gemv(~n", [BlockSz]),
    format(S, "    const signed char* Wq, const __half* Wd,~n", []),
    format(S, "    const signed char* Xq, const __half* Xd,~n", []),
    format(S, "    float* Y, int M, int K) {~n", []),
    format(S, "  const int BM = ~w, BK = ~w;~n", [BM, BK]),
    format(S, "  int warp = threadIdx.x >> 5;      // which row-lane (0..BM-1)~n", []),
    format(S, "  int lane = threadIdx.x & 31;      // lane within the warp~n", []),
    format(S, "  int row = blockIdx.x*BM + warp;~n", []),
    format(S, "  int nblk = K / 32;~n", []),
    %% stage the activation (Xq/Xd) into shared memory, all threads cooperate
    format(S, "  extern __shared__ char smem[];~n", []),
    format(S, "  signed char* sXq = (signed char*)smem;          // K int8~n", []),
    format(S, "  __half* sXd = (__half*)(smem + K);               // nblk fp16 scales~n", []),
    format(S, "  for (int i = threadIdx.x; i < K; i += blockDim.x) sXq[i] = Xq[i];~n", []),
    format(S, "  for (int i = threadIdx.x; i < nblk; i += blockDim.x) sXd[i] = Xd[i];~n", []),
    format(S, "  __syncthreads();   // activation staged, reused by all BM rows~n", []),
    format(S, "  if (row >= M) return;~n", []),
    %% each warp computes one row: lanes split the K-blocks, partial f32 acc
    format(S, "  float acc = 0.0f;~n", []),
    format(S, "  for (int b = lane; b < nblk; b += 32) {~n", []),
    format(S, "    const int* wq4 = (const int*)(Wq + (long)row*K + b*32);~n", []),
    format(S, "    const int* xq4 = (const int*)(sXq + b*32);~n", []),
    format(S, "    int isum = 0;~n", []),
    format(S, "    #pragma unroll~n", []),
    format(S, "    for (int j = 0; j < 8; j += ~w) {~n", [VEC]),
    format(S, "      #pragma unroll~n", []),
    format(S, "      for (int v = 0; v < ~w; v++) isum = __dp4a(wq4[j+v], xq4[j+v], isum);~n", [VEC]),
    format(S, "    }~n", []),
    format(S, "    float wd = __half2float(Wd[(long)row*nblk + b]);~n", []),
    format(S, "    float xd = __half2float(sXd[b]);~n", []),
    format(S, "    acc += (wd * xd) * (float)isum;~n", []),
    format(S, "  }~n", []),
    %% warp-shuffle tree reduction of the per-lane partials -> row result
    format(S, "  #pragma unroll~n", []),
    format(S, "  for (int s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffff, acc, s);~n", []),
    format(S, "  if (lane == 0) Y[row] = acc;~n", []),
    format(S, "}~n", []),
    format(S, "// LAUNCH: grid=(M+BM-1)/BM, block=BM*32, shared=(K + nblk*2) bytes~n", []),
    close(S),
    format("Generated FACT-DERIVED Q8_0 GEMV TILED (BM=~w BK=~w VEC=~w) -> ~w~n", [BM, BK, VEC, OutFile]).

%% ── TILED Q8_0 GEMV with 128-bit (int4) weight loads ───────────────────────
%% The stall profile (cupti-from-prolog) prescribed this: the tiled GEMV's memory_dependency
%% (39-59%) + texture stalls come from reading each 32-byte weight block as 8 separate 32-bit
%% loads. int4 (128-bit) loads fetch 4 int32 per transaction -> 2 loads/block instead of 8,
%% 4x fewer load instructions, better DRAM coalescing. The ARITHMETIC is UNCHANGED: same __dp4a
%% on the same int32 values in the same order, same warp-shuffle reduction -> BIT-EXACT to
%% emit_q8_0_gemv_tiled (only the load WIDTH changes, not the float DAG). Measured against the
%% stall report: should move memory_dependency + texture down. (Heath's 0-ULP third pillar: this
%% stays bit-exact-to-tiled, so it canonicalizes with the same one dossier, no new tolerance.)
emit_q8_0_gemv_tiled_v4(BM, BK, OutFile) :-
    open(OutFile, write, S),
    q8_header(S, "TILED Q8_0 GEMV, 128-bit int4 weight loads (coalesced; stall-prescribed)"),
    format(S, "// gpu_gemv_point(BM=~w, BK=~w, VEC=int4/128-bit)~n", [BM, BK]),
    BlockSz is BM*32, format(S, "extern \"C\" __global__ void __launch_bounds__(~w, 4) k_q8_0_gemv(~n", [BlockSz]),
    format(S, "    const signed char* Wq, const __half* Wd,~n", []),
    format(S, "    const signed char* Xq, const __half* Xd,~n", []),
    format(S, "    float* Y, int M, int K) {~n", []),
    format(S, "  const int BM = ~w;~n", [BM]),
    format(S, "  int warp = threadIdx.x >> 5;~n", []),
    format(S, "  int lane = threadIdx.x & 31;~n", []),
    format(S, "  int row = blockIdx.x*BM + warp;~n", []),
    format(S, "  int nblk = K / 32;~n", []),
    format(S, "  extern __shared__ char smem[];~n", []),
    format(S, "  signed char* sXq = (signed char*)smem;~n", []),
    format(S, "  __half* sXd = (__half*)(smem + K);~n", []),
    format(S, "  for (int i = threadIdx.x; i < K; i += blockDim.x) sXq[i] = Xq[i];~n", []),
    format(S, "  for (int i = threadIdx.x; i < nblk; i += blockDim.x) sXd[i] = Xd[i];~n", []),
    format(S, "  __syncthreads();~n", []),
    format(S, "  if (row >= M) return;~n", []),
    format(S, "  float acc = 0.0f;~n", []),
    format(S, "  for (int b = lane; b < nblk; b += 32) {~n", []),
    format(S, "    const int4* wq16 = (const int4*)(Wq + (long)row*K + b*32);~n", []),
    format(S, "    const int4* xq16 = (const int4*)(sXq + b*32);~n", []),
    format(S, "    int4 w0 = wq16[0], w1 = wq16[1];   // 2x 128-bit loads = the 8 int32 of one block~n", []),
    format(S, "    int4 x0 = xq16[0], x1 = xq16[1];~n", []),
    format(S, "    int isum = 0;~n", []),
    format(S, "    isum = __dp4a(w0.x, x0.x, isum); isum = __dp4a(w0.y, x0.y, isum);~n", []),
    format(S, "    isum = __dp4a(w0.z, x0.z, isum); isum = __dp4a(w0.w, x0.w, isum);~n", []),
    format(S, "    isum = __dp4a(w1.x, x1.x, isum); isum = __dp4a(w1.y, x1.y, isum);~n", []),
    format(S, "    isum = __dp4a(w1.z, x1.z, isum); isum = __dp4a(w1.w, x1.w, isum);~n", []),
    format(S, "    float wd = __half2float(Wd[(long)row*nblk + b]);~n", []),
    format(S, "    float xd = __half2float(sXd[b]);~n", []),
    format(S, "    acc += (wd * xd) * (float)isum;~n", []),
    format(S, "  }~n", []),
    format(S, "  #pragma unroll~n", []),
    format(S, "  for (int s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffff, acc, s);~n", []),
    format(S, "  if (lane == 0) Y[row] = acc;~n", []),
    format(S, "}~n", []),
    format(S, "// LAUNCH: grid=(M+BM-1)/BM, block=BM*32, shared=(K + nblk*2) bytes~n", []),
    close(S),
    format("Generated FACT-DERIVED Q8_0 GEMV TILED-V4 (BM=~w BK=~w int4-loads) -> ~w~n", [BM, BK, OutFile]).

%% ── TILED-V4 with REGISTER-HOISTED loop invariants ──────────────────────────
%% Counters prescribed (cupti_suggest use_registers) + SASS confirmed: the base v4 re-reads K
%% (c[0x0][0x16c]) 23x — the compiler folds (long)row*K and (long)row*nblk against the constant
%% (param) bank EVERY iteration instead of hoisting. On the tiny attn shape (M=896) this is the
%% 20.7% constant_memory stall. This variant hoists row_base=(long)row*K and row_nblk=(long)row*nblk
%% into registers BEFORE the loop. The accumulate expression is BYTE-IDENTICAL to v4 (same FMA
%% contraction, same order) -> BIT-EXACT. Only the address math is reorganized (same addresses).
emit_q8_0_gemv_tiled_v4_reghoist(BM, BK, OutFile) :-
    open(OutFile, write, S),
    format(S, "// TILED-V4 reghoist (BM=~w BK=~w, K/row-base hoisted to registers)~n", [BM, BK]),
    format(S, "#include <cuda_fp16.h>~n", []),
    BlockSz is BM*32, format(S, "extern \"C\" __global__ void __launch_bounds__(~w, 4) k_q8_0_gemv(~n", [BlockSz]),
    format(S, "    const signed char* Wq, const __half* Wd,~n", []),
    format(S, "    const signed char* Xq, const __half* Xd,~n", []),
    format(S, "    float* Y, int M, int K) {~n", []),
    format(S, "  const int BM = ~w;~n", [BM]),
    format(S, "  const int Kr = K;                 // hoist K out of the constant bank~n", []),
    format(S, "  int warp = threadIdx.x >> 5;~n", []),
    format(S, "  int lane = threadIdx.x & 31;~n", []),
    format(S, "  int row = blockIdx.x*BM + warp;~n", []),
    format(S, "  int nblk = Kr / 32;~n", []),
    format(S, "  extern __shared__ char smem[];~n", []),
    format(S, "  signed char* sXq = (signed char*)smem;~n", []),
    format(S, "  __half* sXd = (__half*)(smem + Kr);~n", []),
    format(S, "  for (int i = threadIdx.x; i < Kr; i += blockDim.x) sXq[i] = Xq[i];~n", []),
    format(S, "  for (int i = threadIdx.x; i < nblk; i += blockDim.x) sXd[i] = Xd[i];~n", []),
    format(S, "  __syncthreads();~n", []),
    format(S, "  if (row >= M) return;~n", []),
    format(S, "  const signed char* Wq_row = Wq + (long)row*Kr;   // hoist row*K base~n", []),
    format(S, "  const __half* Wd_row = Wd + (long)row*nblk;       // hoist row*nblk base~n", []),
    format(S, "  float acc = 0.0f;~n", []),
    format(S, "  for (int b = lane; b < nblk; b += 32) {~n", []),
    format(S, "    const int4* wq16 = (const int4*)(Wq_row + b*32);~n", []),
    format(S, "    const int4* xq16 = (const int4*)(sXq + b*32);~n", []),
    format(S, "    int4 w0 = wq16[0], w1 = wq16[1];~n", []),
    format(S, "    int4 x0 = xq16[0], x1 = xq16[1];~n", []),
    format(S, "    int isum = 0;~n", []),
    format(S, "    isum = __dp4a(w0.x, x0.x, isum); isum = __dp4a(w0.y, x0.y, isum);~n", []),
    format(S, "    isum = __dp4a(w0.z, x0.z, isum); isum = __dp4a(w0.w, x0.w, isum);~n", []),
    format(S, "    isum = __dp4a(w1.x, x1.x, isum); isum = __dp4a(w1.y, x1.y, isum);~n", []),
    format(S, "    isum = __dp4a(w1.z, x1.z, isum); isum = __dp4a(w1.w, x1.w, isum);~n", []),
    format(S, "    float wd = __half2float(Wd_row[b]);~n", []),
    format(S, "    float xd = __half2float(sXd[b]);~n", []),
    format(S, "    acc += (wd * xd) * (float)isum;~n", []),
    format(S, "  }~n", []),
    format(S, "  #pragma unroll~n", []),
    format(S, "  for (int s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffff, acc, s);~n", []),
    format(S, "  if (lane == 0) Y[row] = acc;~n", []),
    format(S, "}~n", []),
    close(S),
    format("Generated FACT-DERIVED Q8_0 GEMV TILED-V4-REGHOIST (BM=~w BK=~w) -> ~w~n", [BM, BK, OutFile]).

%% ── TILED-V4 + QUANT PROLOGUE FUSED (the scanner's quant_into_gemv, composed with v4) ──────
%% The v4 tiled kernel ALREADY stages the activation into shared memory. qfused just changes WHAT
%% gets staged: instead of COPYING pre-quantized Xq/Xd from global (a separate k_quant_q8 launch +
%% a global write+read of the int8 activation), we QUANTIZE the f32 activation INTO shared mem in
%% the same staging slot. Everything after the __syncthreads (int4-load dp4a, FMA accumulate,
%% shuffle tree) is BYTE-IDENTICAL to v4 -> bit-exact. Removes one launch + the int8 global
%% round-trip. The quant arithmetic is IDENTICAL to the standalone k_quant_q8 (amax->fp16->rintf).
%% NOTE: this folds the SERIAL quant (per-block thread); W1's parallel quant is order-insensitive
%% (max), so a future warp-cooperative prologue stays bit-exact too — but serial-prologue first.
emit_q8_0_gemv_tiled_v4_qfused(BM, BK, OutFile) :-
    open(OutFile, write, S),
    q8_header(S, "TILED-V4 + quant-prologue FUSED (int4 loads; activation quantized into shared mem)"),
    format(S, "// reduction_order(q8_gemv_dp4a, lanes(32), strided, accum(fma), tree(shfl_down,5)) — bit-exact to tiled_v4~n", []),
    BlockSz is BM*32, format(S, "extern \"C\" __global__ void __launch_bounds__(~w, 4) k_q8_0_gemv(~n", [BlockSz]),
    format(S, "    const signed char* Wq, const __half* Wd,~n", []),
    format(S, "    const float* X,                          // f32 activation IN (not pre-quantized)~n", []),
    format(S, "    float* Y, int M, int K) {~n", []),
    format(S, "  const int BM = ~w;~n", [BM]),
    format(S, "  int warp = threadIdx.x >> 5;~n", []),
    format(S, "  int lane = threadIdx.x & 31;~n", []),
    format(S, "  int row = blockIdx.x*BM + warp;~n", []),
    format(S, "  int nblk = K / 32;~n", []),
    format(S, "  extern __shared__ char smem[];~n", []),
    format(S, "  signed char* sXq = (signed char*)smem;~n", []),
    format(S, "  __half* sXd = (__half*)(smem + K);~n", []),
    format(S, "  // PROLOGUE: quantize f32 X into shared mem (one Q8_0 block per thread, strided).~n", []),
    format(S, "  // IDENTICAL arithmetic to standalone k_quant_q8 (amax -> fp16 scale -> rintf).~n", []),
    format(S, "  for (int bq = threadIdx.x; bq < nblk; bq += blockDim.x) {~n", []),
    format(S, "    float amax = 0.0f;~n", []),
    format(S, "    for (int i = 0; i < 32; i++) { float a = fabsf(X[bq*32+i]); if (a > amax) amax = a; }~n", []),
    format(S, "    float d = (amax > 0.0f) ? amax/127.0f : 1.0f; __half dh = __float2half(d); sXd[bq] = dh;~n", []),
    format(S, "    float dq = __half2float(dh);~n", []),
    format(S, "    for (int i = 0; i < 32; i++) { int q = (int)rintf(X[bq*32+i]/dq); q = q<-127?-127:(q>127?127:q); sXq[bq*32+i] = (signed char)q; }~n", []),
    format(S, "  }~n", []),
    format(S, "  __syncthreads();   // shared quantized activation ready for all rows in this block~n", []),
    format(S, "  if (row >= M) return;~n", []),
    format(S, "  // ── from here, BYTE-IDENTICAL to tiled_v4: int4 loads + fma + shuffle tree ──~n", []),
    format(S, "  float acc = 0.0f;~n", []),
    format(S, "  for (int b = lane; b < nblk; b += 32) {~n", []),
    format(S, "    const int4* wq16 = (const int4*)(Wq + (long)row*K + b*32);~n", []),
    format(S, "    const int4* xq16 = (const int4*)(sXq + b*32);~n", []),
    format(S, "    int4 w0 = wq16[0], w1 = wq16[1];~n", []),
    format(S, "    int4 x0 = xq16[0], x1 = xq16[1];~n", []),
    format(S, "    int isum = 0;~n", []),
    format(S, "    isum = __dp4a(w0.x, x0.x, isum); isum = __dp4a(w0.y, x0.y, isum);~n", []),
    format(S, "    isum = __dp4a(w0.z, x0.z, isum); isum = __dp4a(w0.w, x0.w, isum);~n", []),
    format(S, "    isum = __dp4a(w1.x, x1.x, isum); isum = __dp4a(w1.y, x1.y, isum);~n", []),
    format(S, "    isum = __dp4a(w1.z, x1.z, isum); isum = __dp4a(w1.w, x1.w, isum);~n", []),
    format(S, "    float wd = __half2float(Wd[(long)row*nblk + b]);~n", []),
    format(S, "    float xd = __half2float(sXd[b]);~n", []),
    format(S, "    acc = __fmaf_rn(wd*xd, (float)isum, acc);~n", []),
    format(S, "  }~n", []),
    format(S, "  #pragma unroll~n", []),
    format(S, "  for (int s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffff, acc, s);~n", []),
    format(S, "  if (lane == 0) Y[row] = acc;~n", []),
    format(S, "}~n", []),
    format(S, "// LAUNCH: grid=(M+BM-1)/BM, block=BM*32, shared=(K + nblk*2) bytes. ONE kernel (no separate quant).~n", []),
    close(S),
    format("Generated FACT-DERIVED Q8_0 GEMV TILED-V4-QFUSED (BM=~w BK=~w) -> ~w~n", [BM, BK, OutFile]).

%% ── TILED-V4 + RESIDUAL-ADD EPILOGUE (the scanner's epilogue, composed with v4) ─────────────
%% The o-proj and down-proj GEMVs are followed by a residual add (k_add: y=gemv+resid). This folds
%% that add INTO the GEMV's store: Y[row] = acc + Resid[row]. NO redundancy (unlike qfused) — each
%% output row adds its OWN residual element once, in the epilogue. Removes the k_add launch (1800
%% of them in decode = the most-launched tiny kernel -> sharpens the stall profile per Heath). The
%% v4 float DAG to `acc` is BYTE-IDENTICAL to tiled_v4, and `acc + Resid[row]` is the SAME f32 add
%% k_add does -> BIT-EXACT to (v4 then k_add).
emit_q8_0_gemv_tiled_v4_addres(BM, BK, OutFile) :-
    open(OutFile, write, S),
    q8_header(S, "TILED-V4 + residual-add epilogue (int4 loads; Y = gemv + residual, one kernel)"),
    format(S, "// reduction_order(q8_gemv_dp4a, lanes(32), strided, accum(fma), tree(shfl_down,5)) — bit-exact to tiled_v4~n", []),
    BlockSz is BM*32, format(S, "extern \"C\" __global__ void __launch_bounds__(~w, 4) k_q8_0_gemv_addres(~n", [BlockSz]),
    format(S, "    const signed char* Wq, const __half* Wd,~n", []),
    format(S, "    const signed char* Xq, const __half* Xd,~n", []),
    format(S, "    const float* Resid,                      // residual added in the epilogue~n", []),
    format(S, "    float* Y, int M, int K) {~n", []),
    format(S, "  const int BM = ~w;~n", [BM]),
    format(S, "  int warp = threadIdx.x >> 5;~n", []),
    format(S, "  int lane = threadIdx.x & 31;~n", []),
    format(S, "  int row = blockIdx.x*BM + warp;~n", []),
    format(S, "  int nblk = K / 32;~n", []),
    format(S, "  extern __shared__ char smem[];~n", []),
    format(S, "  signed char* sXq = (signed char*)smem;~n", []),
    format(S, "  __half* sXd = (__half*)(smem + K);~n", []),
    format(S, "  for (int i = threadIdx.x; i < K; i += blockDim.x) sXq[i] = Xq[i];~n", []),
    format(S, "  for (int i = threadIdx.x; i < nblk; i += blockDim.x) sXd[i] = Xd[i];~n", []),
    format(S, "  __syncthreads();~n", []),
    format(S, "  if (row >= M) return;~n", []),
    format(S, "  float acc = 0.0f;~n", []),
    format(S, "  for (int b = lane; b < nblk; b += 32) {~n", []),
    format(S, "    const int4* wq16 = (const int4*)(Wq + (long)row*K + b*32);~n", []),
    format(S, "    const int4* xq16 = (const int4*)(sXq + b*32);~n", []),
    format(S, "    int4 w0 = wq16[0], w1 = wq16[1];~n", []),
    format(S, "    int4 x0 = xq16[0], x1 = xq16[1];~n", []),
    format(S, "    int isum = 0;~n", []),
    format(S, "    isum = __dp4a(w0.x, x0.x, isum); isum = __dp4a(w0.y, x0.y, isum);~n", []),
    format(S, "    isum = __dp4a(w0.z, x0.z, isum); isum = __dp4a(w0.w, x0.w, isum);~n", []),
    format(S, "    isum = __dp4a(w1.x, x1.x, isum); isum = __dp4a(w1.y, x1.y, isum);~n", []),
    format(S, "    isum = __dp4a(w1.z, x1.z, isum); isum = __dp4a(w1.w, x1.w, isum);~n", []),
    format(S, "    float wd = __half2float(Wd[(long)row*nblk + b]);~n", []),
    format(S, "    float xd = __half2float(sXd[b]);~n", []),
    format(S, "    acc = __fmaf_rn(wd*xd, (float)isum, acc);~n", []),
    format(S, "  }~n", []),
    format(S, "  #pragma unroll~n", []),
    format(S, "  for (int s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffff, acc, s);~n", []),
    format(S, "  if (lane == 0) Y[row] = acc + Resid[row];   // FUSED residual add (= k_add, bit-exact)~n", []),
    format(S, "}~n", []),
    format(S, "// LAUNCH: grid=(M+BM-1)/BM, block=BM*32, shared=(K + nblk*2) bytes. Fuses the k_add.~n", []),
    close(S),
    format("Generated FACT-DERIVED Q8_0 GEMV TILED-V4-ADDRES (BM=~w BK=~w) -> ~w~n", [BM, BK, OutFile]).

%% epilogue_compute(silu): the gate GEMV stores silu(acc) = acc/(1+exp(-acc)). The SiLU
%% transcendental is HIDDEN under the memory-bound GEMV's DRAM stalls (measured +0.2% on the gate
%% shape). k_silu_mul then reduces to a bare multiply. The v4 body to `acc` is BYTE-IDENTICAL to
%% tiled_v4 -> reduction_order untouched -> bit-exact GEMV; silu is the same x/(1+exp(-x)) the
%% k_silu_mul did. Lowers the backend-neutral epilogue_compute(silu) fact for CUDA/sm_61.
emit_q8_0_gemv_tiled_v4_silu(BM, BK, OutFile) :-
    open(OutFile, write, S),
    q8_header(S, "TILED-V4 + SiLU epilogue (int4 loads; Y = silu(gemv), hides transcendental under memory)"),
    format(S, "// reduction_order(q8_gemv_dp4a, lanes(32), strided, accum(fma), tree(shfl_down,5)) — bit-exact to tiled_v4~n", []),
    format(S, "extern \"C\" __global__ void k_q8_0_gemv_silu(~n", []),
    format(S, "    const signed char* Wq, const __half* Wd,~n", []),
    format(S, "    const signed char* Xq, const __half* Xd,~n", []),
    format(S, "    float* Y, int M, int K) {~n", []),
    format(S, "  const int BM = ~w;~n", [BM]),
    format(S, "  int warp = threadIdx.x >> 5;~n", []),
    format(S, "  int lane = threadIdx.x & 31;~n", []),
    format(S, "  int row = blockIdx.x*BM + warp;~n", []),
    format(S, "  int nblk = K / 32;~n", []),
    format(S, "  extern __shared__ char smem[];~n", []),
    format(S, "  signed char* sXq = (signed char*)smem;~n", []),
    format(S, "  __half* sXd = (__half*)(smem + K);~n", []),
    format(S, "  for (int i = threadIdx.x; i < K; i += blockDim.x) sXq[i] = Xq[i];~n", []),
    format(S, "  for (int i = threadIdx.x; i < nblk; i += blockDim.x) sXd[i] = Xd[i];~n", []),
    format(S, "  __syncthreads();~n", []),
    format(S, "  if (row >= M) return;~n", []),
    format(S, "  float acc = 0.0f;~n", []),
    format(S, "  for (int b = lane; b < nblk; b += 32) {~n", []),
    format(S, "    const int4* wq16 = (const int4*)(Wq + (long)row*K + b*32);~n", []),
    format(S, "    const int4* xq16 = (const int4*)(sXq + b*32);~n", []),
    format(S, "    int4 w0 = wq16[0], w1 = wq16[1];~n", []),
    format(S, "    int4 x0 = xq16[0], x1 = xq16[1];~n", []),
    format(S, "    int isum = 0;~n", []),
    format(S, "    isum = __dp4a(w0.x, x0.x, isum); isum = __dp4a(w0.y, x0.y, isum);~n", []),
    format(S, "    isum = __dp4a(w0.z, x0.z, isum); isum = __dp4a(w0.w, x0.w, isum);~n", []),
    format(S, "    isum = __dp4a(w1.x, x1.x, isum); isum = __dp4a(w1.y, x1.y, isum);~n", []),
    format(S, "    isum = __dp4a(w1.z, x1.z, isum); isum = __dp4a(w1.w, x1.w, isum);~n", []),
    format(S, "    float wd = __half2float(Wd[(long)row*nblk + b]);~n", []),
    format(S, "    float xd = __half2float(sXd[b]);~n", []),
    format(S, "    acc = __fmaf_rn(wd*xd, (float)isum, acc);~n", []),
    format(S, "  }~n", []),
    format(S, "  #pragma unroll~n", []),
    format(S, "  for (int s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffff, acc, s);~n", []),
    format(S, "  if (lane == 0) Y[row] = acc / (1.0f + expf(-acc));   // FUSED SiLU (expf to match k_silu_mul exactly; hidden under memory)~n", []),
    format(S, "}~n", []),
    format(S, "// LAUNCH: grid=(M+BM-1)/BM, block=BM*32, shared=(K + nblk*2) bytes. Fuses SiLU; k_silu_mul -> bare mul.~n", []),
    close(S),
    format("Generated FACT-DERIVED Q8_0 GEMV TILED-V4-SILU (BM=~w BK=~w) -> ~w~n", [BM, BK, OutFile]).

%% ── THE CANONICAL Q8_0 GEMV REDUCTION ORDER (the contract, 2026-06-12) ──────
%% "Torch isn't the contract; our canonical computation is." The tiled/v4 GEMV's float DAG is
%% DECLARED here as the canonical order, so the serial reference renders the SAME bits (0-ULP
%% pair-gate), exactly as we did for rmsnorm. The order is TOTAL and includes the FMA contraction
%% (the per-block accumulate is a FUSED multiply-add — one rounding — which is part of the order,
%% not a compiler accident; making it explicit is the BPD thesis at its sharpest):
%%   lanes(32): 32 lane-partials; lane L folds blocks b = L, L+32, L+64, ...
%%   accum(fma): acc = fma(wd*xd, (float)isum, acc)  — FUSED, one rounding per block
%%   tree(shfl_down, 5): the 32 partials merge via 5-level shuffle-down (s=16,8,4,2,1)
%%   isum: integer dp4a — order-insensitive, exact (not a rounding site)
reduction_order(q8_gemv_dp4a, lanes(32), strided, accum(fma), tree(shfl_down, 5)).

%% canonical-serial Q8_0 GEMV: ONE thread per row, reproducing reduction_order(q8_gemv_dp4a,...)
%% EXACTLY — 32 lane-partials (each an fma-contracted strided fold), then the 5-level shuffle-tree
%% merge — so it is 0-ULP IDENTICAL to the tiled/v4 kernel (proven: host reproduction of this order
%% = 0 ULP). The canonical REFERENCE for any-M / oracle / decode_referee recompute. Both render
%% from one order spec; same DAG => same bits.
emit_q8_0_gemv_canonical_serial(OutFile) :-
    open(OutFile, write, S),
    q8_header(S, "CANONICAL-SERIAL Q8_0 GEMV (reproduces the tiled/v4 reduction order, 0-ULP)"),
    format(S, "// reduction_order(q8_gemv_dp4a, lanes(32), strided, accum(fma), tree(shfl_down,5))~n", []),
    BlockSz = 64, format(S, "extern \"C\" __global__ void __launch_bounds__(~w, 4) k_q8_0_gemv(~n", [BlockSz]),
    format(S, "    const signed char* Wq, const __half* Wd,~n", []),
    format(S, "    const signed char* Xq, const __half* Xd,~n", []),
    format(S, "    float* Y, int M, int K) {~n", []),
    format(S, "  int row = blockIdx.x*blockDim.x + threadIdx.x; if (row >= M) return;~n", []),
    format(S, "  int nblk = K / 32;~n", []),
    format(S, "  float lane_acc[32];~n", []),
    format(S, "  // 1) 32 lane-partials: lane L folds blocks b=L,L+32,... with FUSED multiply-add~n", []),
    format(S, "  for (int lane = 0; lane < 32; lane++) {~n", []),
    format(S, "    float acc = 0.0f;~n", []),
    format(S, "    for (int b = lane; b < nblk; b += 32) {~n", []),
    format(S, "      const int* wq4 = (const int*)(Wq + (long)row*K + b*32);~n", []),
    format(S, "      const int* xq4 = (const int*)(Xq + b*32);~n", []),
    format(S, "      int isum = 0;~n", []),
    format(S, "      #pragma unroll~n", []),
    format(S, "      for (int j = 0; j < 8; j++) isum = __dp4a(wq4[j], xq4[j], isum);~n", []),
    format(S, "      float wd = __half2float(Wd[(long)row*nblk + b]);~n", []),
    format(S, "      float xd = __half2float(Xd[b]);~n", []),
    format(S, "      acc = __fmaf_rn(wd*xd, (float)isum, acc);   // FUSED: matches the tiled kernel's contraction~n", []),
    format(S, "    }~n", []),
    format(S, "    lane_acc[lane] = acc;~n", []),
    format(S, "  }~n", []),
    format(S, "  // 2) the 5-level shuffle-down tree merge (s=16,8,4,2,1), same as the warp reduction~n", []),
    format(S, "  for (int s = 16; s > 0; s >>= 1)~n", []),
    format(S, "    for (int t = 0; t < s; t++) lane_acc[t] += lane_acc[t+s];~n", []),
    format(S, "  Y[row] = lane_acc[0];~n", []),
    format(S, "}~n", []),
    format(S, "// LAUNCH: thread_per_row total=M (canonical reference; 0-ULP to tiled/v4)~n", []),
    close(S),
    format("Generated FACT-DERIVED Q8_0 GEMV CANONICAL-SERIAL -> ~w~n", [OutFile]).

%% emit_dp4a_accum_body(+Stream, +XqName, +XdName) — the dp4a accumulation loop, parameterized
%% ONLY by where the activation quants/scales are read from (XqName/XdName). Used by BOTH the
%% plain gemv (reads global Xq/Xd) AND the quant-prologue-fused gemv (reads shared mem). Emitting
%% the SAME body from one helper makes the float reduction DAG identical by construction -> nvcc
%% makes the same FMA decisions -> the fused kernel is BIT-EXACT to the unfused (Mavchin's DAG
%% theorem: same source structure = same dependency DAG = same fusion = same bits). The ONLY
%% difference between fused and unfused is the load source, exactly as required.
emit_dp4a_accum_body(S, Xq, Xd) :-
    format(S, "  float acc = 0.0f;~n", []),
    format(S, "  for (int b = 0; b < nblk; b++) {~n", []),
    format(S, "    const int* wq4 = (const int*)(Wq + (long)row*K + b*32);  // 8 packed int8x4~n", []),
    format(S, "    const int* xq4 = (const int*)(~w + b*32);~n", [Xq]),
    format(S, "    int isum = 0;~n", []),
    format(S, "    #pragma unroll~n", []),
    format(S, "    for (int j = 0; j < 8; j++) isum = __dp4a(wq4[j], xq4[j], isum);  // 4-way int8 dot~n", []),
    format(S, "    float wd = __half2float(Wd[(long)row*nblk + b]);~n", []),
    format(S, "    float xd = __half2float(~w[b]);~n", [Xd]),
    format(S, "    acc += (wd * xd) * (float)isum;~n", []),
    format(S, "  }~n", []).

%% emit_from_fact with prologue(quant): FUSE the upstream activation quantize INTO the GEMV.
%% The f32 activation is quantized into SHARED memory once per thread-block (the same rintf +
%% fp16-scale arithmetic as the standalone k_quant_q8, byte-for-byte), then each thread runs the
%% VERBATIM dp4a accumulation reading from shared instead of global. Eliminates the per-GEMV
%% global round-trip of the quantized activation (measured 178x/token forced write+read). The
%% float side is the SAME emitted body (emit_dp4a_accum_body) -> bit-exact by construction.
emit_q8_0_dot_prologue_quant(EpiC, OutFile) :-
    open(OutFile, write, S),
    q8_header(S, "dp4a + quant PROLOGUE fused (activation quantized into shared mem; no global round-trip)"),
    format(S, "extern \"C\" __global__ void k_q8_0_gemv_qfused(~n", []),
    format(S, "    const signed char* Wq, const __half* Wd,~n", []),
    format(S, "    const float* X,~n", []),                 % <-- f32 activation in (NOT pre-quantized)
    format(S, "    float* Y, int M, int K) {~n", []),
    format(S, "  extern __shared__ char smem[];~n", []),
    format(S, "  int nblk = K / 32;~n", []),
    format(S, "  signed char* Xq = (signed char*)smem;            // K int8~n", []),
    format(S, "  __half* Xd = (__half*)(smem + K);                 // nblk fp16 scales~n", []),
    format(S, "  // PROLOGUE: quantize the f32 activation into shared mem, one block per thread~n", []),
    format(S, "  // (IDENTICAL arithmetic to the standalone k_quant_q8: amax -> fp16 scale -> rintf).~n", []),
    format(S, "  for (int bq = threadIdx.x; bq < nblk; bq += blockDim.x) {~n", []),
    format(S, "    float amax = 0.0f;~n", []),
    format(S, "    for (int i = 0; i < 32; i++) { float a = fabsf(X[bq*32+i]); if (a > amax) amax = a; }~n", []),
    format(S, "    float d = (amax > 0.0f) ? amax/127.0f : 1.0f; __half dh = __float2half(d); Xd[bq] = dh;~n", []),
    format(S, "    float dq = __half2float(dh);~n", []),
    format(S, "    for (int i = 0; i < 32; i++) { int q = (int)rintf(X[bq*32+i]/dq); q = q<-127?-127:(q>127?127:q); Xq[bq*32+i] = (signed char)q; }~n", []),
    format(S, "  }~n", []),
    format(S, "  __syncthreads();   // shared activation ready for all rows in this block~n", []),
    format(S, "  int row = blockIdx.x*blockDim.x + threadIdx.x; if (row >= M) return;~n", []),
    emit_dp4a_accum_body(S, "Xq", "Xd"),   % <-- SAME body, reading shared Xq/Xd -> bit-exact float side
    format(S, "  Y[row] = ~w;   // fused epilogue over acc~n", [EpiC]),
    format(S, "}~n", []),
    format(S, "// LAUNCH: thread_per_row total=M, block=BS, shared = K + nblk*2 bytes~n", []),
    close(S),
    format("Generated FACT-DERIVED Q8_0 GEMV (dp4a + quant-prologue fused, epi=~w) -> ~w~n", [EpiC, OutFile]).

q8_header(S, Desc) :-
    format(S, "/* GENERATED from op_expr q8_0_dot(block(32), scale(fp16), quant(int8)) — ~w.~n", [Desc]),
    format(S, " * per block: (xd*yd) * sum_i(xq[i]*yq[i]). Algebraically == dequant-then-dot~n", []),
    format(S, " * (verified bit-exact, memory 4727ad11). Fact-derived (Iyun, 2026-06-08). */~n", []),
    format(S, "#include <cuda_fp16.h>~n", []).
