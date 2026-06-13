%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════════
%% norm_softmax_from_facts.pl — MOVE 5 (thesis fidelity): emit rms_norm and softmax
%% kernels FROM their op_expr facts, not hand-written .cu in the layer harness.
%%
%% op_expr(bpd_rmsnorm) = rmsnorm(Axis, const(Eps), var)  — row-wise:
%%     y[i,:] = x[i,:] * rsqrt(mean(x[i,:]^2) + eps) * w[:]
%% op_expr(bpd_softmax) = softmax(Axis, var)              — row-wise:
%%     y[i,:] = exp(x[i,:] - max) / sum(exp(x[i,:] - max))
%% Both are ROW REDUCTIONS — emitted from the fact's reduction structure.
%% epilogue-capable (Move 3): the folded tail rides on the row store.
%% Author: Iyun, 2026-06-08
%% ═══════════════════════════════════════════════════════════════════════════
:- module(norm_softmax_from_facts, [
    emit_rmsnorm_cuda/2,    % emit_rmsnorm_cuda(+Eps, +OutFile)
    emit_rmsnorm_cuda_blockrow/2,  % block-per-row parallel reduction variant
    emit_rmsnorm_cuda_canonical_serial/2,  % canonical-order serial reference (0-ULP to blockrow)
    emit_softmax_cuda/1,    % emit_softmax_cuda(+OutFile)
    emit_from_fact/3        % emit_from_fact(+OpExpr, +Opts, +OutFile)
]).
:- discontiguous emit_from_fact/3.
:- use_module(library(lists)).

%% emit_from_fact(+OpExpr, +Opts, +OutFile): dispatch on the op_expr fact shape.
%% This is the thesis move: the KERNEL is derived from the FACT, not hand-written.
emit_from_fact(rmsnorm(_Axis, const(FactEps), _Var), Opts, OutFile) :- !,
    % eps(E) in Opts OVERRIDES the fact's baked eps — lets the model's true norm_eps
    % flow through (the fact default is generic; real models specify their own).
    ( member(eps(E), Opts) -> Eps = E ; Eps = FactEps ),
    % mode(block_row): block-per-row parallel reduction (decode M=1 -> the thread_per_row
    % form serializes one thread over N; block_row uses the whole block to reduce).
    ( member(mode(block_row), Opts)
    -> emit_rmsnorm_cuda_blockrow(Eps, OutFile)
    ;  member(mode(canonical_serial), Opts)
    -> emit_rmsnorm_cuda_canonical_serial(Eps, OutFile)
    ;  emit_rmsnorm_cuda(Eps, OutFile) ).

%% ── THE CANONICAL REDUCTION ORDER (the contract, Bocher's ruling 2026-06-11) ──────────────
%% "Torch isn't the contract; our canonical tree is." Torch's left-fold was never a truth — it
%% was an accident of torch's implementation; IEEE rounds every order differently and no order
%% is "the real sum." What made a reference a reference was REPRODUCIBILITY; a declared canonical
%% order gives us reproducibility WE own, rendered into every kernel from one spec (the BPD thesis
%% applied to arithmetic: both sides of the reduction boundary declared, same DAG => same bits).
%%
%% THE ORDER IS TOTAL and pinned as contract (condition 2): lanes(256) is part of the contract,
%% NOT blockDim-incidental. A future config wanting a different lane count is a NEW canonical
%% order = a new reference migration with its own dossier.
reduction_order(rms_ss, lanes(256), strided, tree(pairwise, 8)).
emit_from_fact(softmax(_Axis, _Var), _Opts, OutFile) :- !,
    emit_softmax_cuda(OutFile).

%% rms_norm: one thread per row, reduce sum-of-squares, rsqrt, scale by weight.
%% Derived from rmsnorm(Axis, const(Eps), var): the reduction is sum(v^2), the
%% pointwise map is v * rsqrt(mean + eps) * w.
emit_rmsnorm_cuda(Eps, OutFile) :-
    open(OutFile, write, S),
    format(S, "/* GENERATED from op_expr rmsnorm(1, const(~w), var) — row reduction + scale.~n", [Eps]),
    format(S, " * y[i,:] = x[i,:] * rsqrt(mean(x[i,:]^2) + eps) * w[:]. Fact-derived (Iyun). */~n", []),
    format(S, "extern \"C\" __global__ void k_rmsnorm(const float* x, const float* w, float* y, int M, int N) {~n", []),
    format(S, "  int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= M) return;~n", []),
    format(S, "  const float eps = ~wf;~n", [Eps]),
    format(S, "  const float* row = x + (long)i*N; float ss = 0.0f;~n", []),
    format(S, "  for (int j=0;j<N;j++) ss += row[j]*row[j];        // reduction: sum of squares~n", []),
    format(S, "  float inv = rsqrtf(ss/N + eps);                   // rsqrt(mean + eps)~n", []),
    format(S, "  float* o = y + (long)i*N;~n", []),
    format(S, "  for (int j=0;j<N;j++) o[j] = row[j]*inv*w[j];     // scale by inv * weight~n", []),
    format(S, "}~n", []),
    format(S, "// LAUNCH: thread_per_row total=M~n", []),
    close(S),
    format("Generated FACT-DERIVED rms_norm (eps=~w) -> ~w~n", [Eps, OutFile]).

%% rms_norm BLOCK-PER-ROW: one BLOCK per row, the block's threads cooperatively reduce the
%% sum-of-squares via a shared-memory tree reduction, then normalize in parallel. Fixes the
%% decode pathology of the thread_per_row form: with M=1 (one token) that form runs ONE thread
%% serially over N elements (the rest of the block idle); block_row uses all blockDim threads.
%% NOTE: the tree reduction changes the float accumulation ORDER vs the serial sum, so this is
%% NOT bit-exact to emit_rmsnorm_cuda — it's tolerance(eps), to be MEASURED by the gate and
%% GR-certified (rmsnorm output feeds the layer; a reduction-order change can shift logits at
%% the ULP scale, must verify no token flips). Same math, different reduction tree, different
%% bits (Mavchin's DAG theorem — here we ACCEPT the difference and bound it, rather than avoid
%% it, because the speedup is the point and the drift is well-conditioned for rmsnorm).
emit_rmsnorm_cuda_blockrow(Eps, OutFile) :-
    open(OutFile, write, S),
    format(S, "/* GENERATED from op_expr rmsnorm(1, const(~w), var) — BLOCK-per-row parallel reduction.~n", [Eps]),
    format(S, " * y[i,:] = x[i,:] * rsqrt(mean(x[i,:]^2) + eps) * w[:]. One block per row; block~n", []),
    format(S, " * cooperatively reduces sum-of-squares (shared-mem tree). Fact-derived (Iyun). */~n", []),
    format(S, "extern \"C\" __global__ void k_rmsnorm(const float* x, const float* w, float* y, int M, int N) {~n", []),
    format(S, "  int row = blockIdx.x; if (row >= M) return;~n", []),
    format(S, "  int t = threadIdx.x; int nt = blockDim.x;~n", []),
    format(S, "  const float eps = ~wf;~n", [Eps]),
    format(S, "  const float* xr = x + (long)row*N;~n", []),
    format(S, "  float* yr = y + (long)row*N;~n", []),
    format(S, "  // each thread accumulates a strided slice of the sum-of-squares~n", []),
    format(S, "  float local = 0.0f;~n", []),
    format(S, "  for (int j = t; j < N; j += nt) local += xr[j]*xr[j];~n", []),
    format(S, "  extern __shared__ float sred[];~n", []),
    format(S, "  sred[t] = local; __syncthreads();~n", []),
    format(S, "  // shared-memory tree reduction (parallel sum of the per-thread partials)~n", []),
    format(S, "  for (int s = nt/2; s > 0; s >>= 1) { if (t < s) sred[t] += sred[t+s]; __syncthreads(); }~n", []),
    format(S, "  float inv = rsqrtf(sred[0]/N + eps);~n", []),
    format(S, "  __syncthreads();~n", []),
    format(S, "  // parallel scale: each thread writes a strided slice~n", []),
    format(S, "  for (int j = t; j < N; j += nt) yr[j] = xr[j]*inv*w[j];~n", []),
    format(S, "}~n", []),
    format(S, "// LAUNCH: block_per_row grid=M block=BS shared=BS*4 bytes~n", []),
    close(S),
    format("Generated FACT-DERIVED rms_norm BLOCK-ROW (eps=~w) -> ~w~n", [Eps, OutFile]).

%% rms_norm CANONICAL-SERIAL: one thread, but it reproduces the EXACT reduction_order(rms_ss,...)
%% the block_row kernel uses — 256 strided left-fold lane-partials, then the 8-level pairwise
%% tree — so it is 0-ULP IDENTICAL to the block_row kernel (proven: host reproduction = 0 ULP).
%% This is the canonical REFERENCE for any-M (prefill, oracles, decode_referee recompute). Both
%% kernels render from the SAME order spec; same DAG => same bits (Bocher's ruling: move the
%% reference to canonical-tree; the contract is "our declared order", not torch's left-fold).
%% LANES=256 is the contract (reduction_order fact), not blockDim-incidental.
emit_rmsnorm_cuda_canonical_serial(Eps, OutFile) :-
    open(OutFile, write, S),
    format(S, "/* GENERATED from op_expr rmsnorm(1, const(~w), var) — CANONICAL-SERIAL reference.~n", [Eps]),
    format(S, " * Reproduces reduction_order(rms_ss, lanes(256), strided, tree(pairwise,8)) — the~n", []),
    format(S, " * SAME order as the block_row kernel, serially. 0-ULP to block_row by construction.~n", []),
    format(S, " * The contract reference (Bocher: torch isn't the contract; our canonical tree is). */~n", []),
    format(S, "extern \"C\" __global__ void k_rmsnorm(const float* x, const float* w, float* y, int M, int N) {~n", []),
    format(S, "  int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= M) return;~n", []),
    format(S, "  const float eps = ~wf;~n", [Eps]),
    format(S, "  const int NL = 256;   // canonical lane count (contract)~n", []),
    format(S, "  const float* row = x + (long)i*N;~n", []),
    format(S, "  float sred[256];~n", []),
    format(S, "  // 1) each of NL lanes left-folds its strided slice (same as block_row lane t)~n", []),
    format(S, "  for (int t = 0; t < NL; t++) {~n", []),
    format(S, "    float acc = 0.0f;~n", []),
    format(S, "    for (int j = t; j < N; j += NL) acc += row[j]*row[j];~n", []),
    format(S, "    sred[t] = acc;~n", []),
    format(S, "  }~n", []),
    format(S, "  // 2) the 8-level pairwise tree (same merge order as block_row's __syncthreads loop)~n", []),
    format(S, "  for (int s = NL/2; s > 0; s >>= 1)~n", []),
    format(S, "    for (int t = 0; t < s; t++) sred[t] += sred[t+s];~n", []),
    format(S, "  float inv = rsqrtf(sred[0]/N + eps);~n", []),
    format(S, "  float* o = y + (long)i*N;~n", []),
    format(S, "  for (int j = 0; j < N; j++) o[j] = row[j]*inv*w[j];~n", []),
    format(S, "}~n", []),
    format(S, "// LAUNCH: thread_per_row total=M (canonical reference; 0-ULP to block_row)~n", []),
    close(S),
    format("Generated FACT-DERIVED rms_norm CANONICAL-SERIAL (eps=~w) -> ~w~n", [Eps, OutFile]).

%% softmax: one thread per row — max (reduction), exp-shift, sum (reduction), divide.
%% Derived from softmax(Axis, var): two reductions (max, sum) framing exp.
emit_softmax_cuda(OutFile) :-
    open(OutFile, write, S),
    format(S, "/* GENERATED from op_expr softmax(1, var) — row max/sum reductions.~n", []),
    format(S, " * y[i,:] = exp(x[i,:]-max) / sum(exp(x[i,:]-max)). Fact-derived (Iyun). */~n", []),
    format(S, "extern \"C\" __global__ void k_softmax(const float* x, float* y, int M, int N, float scale) {~n", []),
    format(S, "  int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= M) return;~n", []),
    format(S, "  const float* row = x + (long)i*N; float* o = y + (long)i*N;~n", []),
    format(S, "  float mx = -3.4e38f;~n", []),
    format(S, "  for (int j=0;j<N;j++){ float v=row[j]*scale; if(v>mx) mx=v; }   // reduction: row max~n", []),
    format(S, "  float sum = 0.0f;~n", []),
    format(S, "  for (int j=0;j<N;j++){ float e=expf(row[j]*scale-mx); o[j]=e; sum+=e; } // exp-shift + sum~n", []),
    format(S, "  float inv = 1.0f/sum;~n", []),
    format(S, "  for (int j=0;j<N;j++) o[j] *= inv;                              // normalize~n", []),
    format(S, "}~n", []),
    format(S, "// LAUNCH: thread_per_row total=M~n", []),
    close(S),
    format("Generated FACT-DERIVED softmax -> ~w~n", [OutFile]).
