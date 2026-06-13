%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════════
%% soa_chain_emitter.pl — RUNG 3: two-matmul Q8_0 chain (gemv → quantize → gemv).
%%
%% THE COMPOSITION TEST (Heath's plan): does bit-identity SURVIVE composition
%% across matmuls? A single gemv is rung 2 (proven 0-ULP). A chain feeds gemv1's
%% FLOAT output through the ggml-exact Q8_0 re-quantization and into gemv2. The
%% re-quantization is the FP-sensitive crux — if THAT is bit-identical, the whole
%% chain is, and (by induction) so is the full model. This is the rung that
%% proves "bit-identity composes" — the property the full-model claim rests on.
%%
%% THE INTERMEDIATE QUANTIZATION (ggml-EXACT, the crux):
%%   per block of 32: amax = max|x_j|;  d = amax/127 (fp32);  id = (d!=0)?1/d:0
%%                     stored scale ad = fp16(d) value;  q_j = (i8)roundf(x_j * id)
%%   This MUST match ggml byte-for-byte or the chain diverges. We emit it as a
%%   FACT-driven kernel @bpd_q8_0_quantize and verify it 0-ULP as its own atom.
%%
%% CHAIN: dst = W2 · quantize(W1 · x)
%%   gemv1 (@bpd_soa_gemv_q8_0): x(q8_0) → y1(float, nrows1)
%%   quantize (@bpd_q8_0_quantize): y1(float) → (q8_0: aq2, ad2)
%%   gemv2 (@bpd_soa_gemv_q8_0): → dst(float, nrows2)
%%
%% Verified per-rung 0-ULP vs CPU reference doing the IDENTICAL chain.
%% Composes rung-2 gemv (soa_gemv_emitter) + the new quantize atom.
%%
%% Author: Iyun, 2026-06-06 (rung-3 of the bottom-up SoA chain)
%% ═══════════════════════════════════════════════════════════════════════════

:- module(soa_chain_emitter, [
    emit_q8_0_quantize/1,   % emit_q8_0_quantize(+OutFile)  — the re-quant atom
    emit_soa_chain/1        % emit_soa_chain(+OutFile)      — the gemv→quant→gemv chain glue
]).

:- use_module(library(lists)).

qk(32).

%% ───────────────────────────────────────────────────────────────────────────
%% emit_q8_0_quantize — the ggml-exact activation quantization, as a kernel.
%%   void @bpd_q8_0_quantize(float* %x, i8* %q, float* %d, i32 %nb)
%%   per block b in 0..nb: amax=max|x|; dd=amax/127; id=1/dd (or 0);
%%                         d[b]=fp16round(dd); q[b*32+j]=(i8)round(x*id)
%% Emitted with scalar libm semantics matching ggml (roundf = round-half-away;
%% fp16 store/load via llvm fptrunc/fpext to half). Verified 0-ULP as an atom.
%% ───────────────────────────────────────────────────────────────────────────

emit_q8_0_quantize(OutFile) :-
    qk(QK),
    open(OutFile, write, S),
    format(S, '; ══════════════════════════════════════════════════════════~n', []),
    format(S, '; bpd_q8_0_quantize — ggml-exact activation quantization (rung-3 atom)~n', []),
    format(S, ';   per block: d=amax/127 (fp32), stored d=fp16(d), q=(i8)round(x/d)~n', []),
    format(S, '; ══════════════════════════════════════════════════════════~n~n', []),
    %% llvm intrinsics for the FP-knob-exact ops
    format(S, 'declare float @llvm.fabs.f32(float)~n', []),
    format(S, 'declare float @llvm.maxnum.f32(float, float)~n', []),
    format(S, 'declare float @llvm.round.f32(float)   ; round-half-away-from-zero (matches roundf)~n~n', []),
    format(S, 'define void @bpd_q8_0_quantize(float* noalias %x, i8* noalias %q, float* noalias %d, i32 %nb) {~n', []),
    format(S, 'entry:~n', []),
    format(S, '  br label %blk.header~n~n', []),
    %% per-block loop
    format(S, 'blk.header:~n', []),
    format(S, '  %b = phi i32 [ 0, %entry ], [ %b.next, %blk.body ]~n', []),
    format(S, '  %bcmp = icmp slt i32 %b, %nb~n', []),
    format(S, '  br i1 %bcmp, label %blk.body, label %done~n~n', []),
    format(S, 'blk.body:~n', []),
    format(S, '  %xb.off = mul i32 %b, ~w~n', [QK]),
    format(S, '  %xb.off.i = sext i32 %xb.off to i64~n', []),
    format(S, '  %xb = getelementptr float, float* %x, i64 %xb.off.i~n', []),
    %% pass 1: amax = max_j |x_j|  (unrolled over QK lanes)
    format(S, '  ; pass 1: amax over ~w lanes~n', [QK]),
    emit_amax(S, QK),
    %% d = amax/127 ; store fp16(d) ; id = (d!=0)? 1/d : 0
    format(S, '  %dd = fdiv float %amax, 1.270000e+02~n', []),
    format(S, '  %dh = fptrunc float %dd to half      ; store as fp16~n', []),
    format(S, '  %dh.f = fpext half %dh to float       ; the fp16-rounded value used downstream~n', []),
    format(S, '  %dp = getelementptr float, float* %d, i32 %b~n', []),
    format(S, '  store float %dh.f, float* %dp, align 4~n', []),
    format(S, '  %is0 = fcmp oeq float %dd, 0.000000e+00~n', []),
    format(S, '  %inv = fdiv float 1.000000e+00, %dd~n', []),
    format(S, '  %id = select i1 %is0, float 0.000000e+00, float %inv~n', []),
    %% pass 2: q_j = (i8) round(x_j * id)   (NOTE: ggml uses id from fp32 d, NOT fp16 d)
    format(S, '  ; pass 2: quantize ~w lanes with id = 1/d (fp32 d, ggml-exact)~n', [QK]),
    format(S, '  %qb.off = mul i32 %b, ~w~n', [QK]),
    format(S, '  %qb.off.i = sext i32 %qb.off to i64~n', []),
    format(S, '  %qb = getelementptr i8, i8* %q, i64 %qb.off.i~n', []),
    emit_quant_lanes(S, QK),
    format(S, '  %b.next = add i32 %b, 1~n', []),
    format(S, '  br label %blk.header~n~n', []),
    format(S, 'done:~n', []),
    format(S, '  ret void~n', []),
    format(S, '}~n', []),
    close(S),
    format("Emitted q8_0 quantize atom to ~w~n", [OutFile]).

%% amax over QK lanes: load each, fabs, maxnum-reduce into %amax
emit_amax(S, QK) :-
    format(S, '  %x0 = load float, float* %xb, align 4~n', []),
    format(S, '  %a0 = call float @llvm.fabs.f32(float %x0)~n', []),
    format(S, '  %amax0 = fadd float %a0, 0.000000e+00~n', []),   % seed (a0)
    Last is QK - 1,
    foldl_amax(S, 1, Last),
    LastIdx is QK - 1,
    format(S, '  %amax = fadd float %amax~w, 0.000000e+00~n', [LastIdx]).

foldl_amax(_, I, Last) :- I > Last, !.
foldl_amax(S, I, Last) :-
    Prev is I - 1,
    format(S, '  %xp~w = getelementptr float, float* %xb, i32 ~w~n', [I, I]),
    format(S, '  %x~w = load float, float* %xp~w, align 4~n', [I, I]),
    format(S, '  %a~w = call float @llvm.fabs.f32(float %x~w)~n', [I, I]),
    format(S, '  %amax~w = call float @llvm.maxnum.f32(float %amax~w, float %a~w)~n', [I, Prev, I]),
    I1 is I + 1,
    foldl_amax(S, I1, Last).

%% quantize QK lanes: q_j = (i8) round(x_j * id)
emit_quant_lanes(S, QK) :-
    Last is QK - 1,
    forall(between(0, Last, J), emit_quant_lane(S, J)).

emit_quant_lane(S, J) :-
    format(S, '  %qx~w.p = getelementptr float, float* %xb, i32 ~w~n', [J, J]),
    format(S, '  %qx~w = load float, float* %qx~w.p, align 4~n', [J, J]),
    format(S, '  %qm~w = fmul float %qx~w, %id~n', [J, J]),
    format(S, '  %qr~w = call float @llvm.round.f32(float %qm~w)~n', [J, J]),
    format(S, '  %qi~w = fptosi float %qr~w to i8~n', [J, J]),
    format(S, '  %qp~w = getelementptr i8, i8* %qb, i32 ~w~n', [J, J]),
    format(S, '  store i8 %qi~w, i8* %qp~w, align 1~n', [J, J]).

%% ───────────────────────────────────────────────────────────────────────────
%% emit_soa_chain — glue: declares the three composed functions + a driver that
%% chains them. The driver is intentionally thin: it sequences gemv1 → quantize
%% → gemv2. (The heavy lifting is the three verified atoms.) The CHAIN's
%% bit-identity is verified by the gate doing the identical CPU sequence.
%% ───────────────────────────────────────────────────────────────────────────

emit_soa_chain(OutFile) :-
    open(OutFile, write, S),
    format(S, '; ══════════════════════════════════════════════════════════~n', []),
    format(S, '; bpd_soa_chain_q8_0 — rung-3: dst = W2 · quantize(W1 · x)~n', []),
    format(S, ';   composes gemv1 → q8_0_quantize → gemv2 (all verified atoms)~n', []),
    format(S, '; ══════════════════════════════════════════════════════════~n~n', []),
    format(S, 'declare void @bpd_soa_gemv_q8_0(i8*, float*, i8*, float*, float*, i32, i32)~n', []),
    format(S, 'declare void @bpd_q8_0_quantize(float*, i8*, float*, i32)~n~n', []),
    %% chain driver: y1 = gemv1(x);  (aq2,ad2) = quantize(y1);  dst = gemv2(aq2,ad2)
    %%   buffers (y1 float[nrows1], aq2 i8[nrows1], ad2 float[nrows1/32]) caller-provided.
    format(S, 'define void @bpd_soa_chain_q8_0(~n', []),
    format(S, '    i8* %w1q, float* %w1d, i8* %xq, float* %xd,~n', []),
    format(S, '    float* %y1, i8* %aq2, float* %ad2,~n', []),
    format(S, '    i8* %w2q, float* %w2d, float* %dst,~n', []),
    format(S, '    i32 %nrows1, i32 %nb1, i32 %nrows2, i32 %nb2) {~n', []),
    format(S, 'entry:~n', []),
    format(S, '  ; gemv1: y1 = W1 · x   (ncols_dst=1 chain, single column)~n', []),
    format(S, '  call void @bpd_soa_gemv_q8_0(i8* %w1q, float* %w1d, i8* %xq, float* %xd, float* %y1, i32 %nrows1, i32 %nb1)~n', []),
    format(S, '  ; quantize y1 → (aq2, ad2)   (nb1 = nrows1/32 blocks of the y1 vector)~n', []),
    format(S, '  call void @bpd_q8_0_quantize(float* %y1, i8* %aq2, float* %ad2, i32 %nb2)~n', []),
    format(S, '  ; gemv2: dst = W2 · quantize(y1)~n', []),
    format(S, '  call void @bpd_soa_gemv_q8_0(i8* %w2q, float* %w2d, i8* %aq2, float* %ad2, float* %dst, i32 %nrows2, i32 %nb2)~n', []),
    format(S, '  ret void~n', []),
    format(S, '}~n', []),
    close(S),
    format("Emitted SoA chain glue to ~w~n", [OutFile]).
