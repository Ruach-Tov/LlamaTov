%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ════════════════════════════════════════════════════════════════════════
%% rope_from_facts.pl — emit the RoPE (rotary position embedding) kernel FROM
%% its op_expr fact, not hand-written .cu. Part of the host-island removal:
%% device RoPE writes roped q/k so attention can stay on-device.
%%
%% op_expr(bpd_rope) = rope(theta(Theta), pairing(half_split), freq(inv_theta_pow))
%%   half = hd/2;  freqs[j] = 1 / theta^(2j/hd)         for j in 0..half-1
%%   angle[j] = pos * freqs[j]
%%   HALF-SPLIT (NeoX) pairing — x[j] pairs with x[j+half]:
%%     out[j]      = x[j]*cos(angle[j]) - x[j+half]*sin(angle[j])
%%     out[j+half] = x[j]*sin(angle[j]) + x[j+half]*cos(angle[j])
%%
%% This MATCHES llamatov_run.apply_rope exactly (the reference): same freq
%% formula, same half-split pairing, same position semantics. Device sinf/cosf
%% vs host torch.cos/sin is the soft device-vs-host variance class (A1-exact
%% self-consistent, A2-soft vs torch) — like rms_norm's expf, by design.
%% Author: Iyun, 2026-06-10
%% ════════════════════════════════════════════════════════════════════════
:- module(rope_from_facts, [
    emit_rope_cuda/2,       % emit_rope_cuda(+Theta, +OutFile)
    emit_from_fact/3        % emit_from_fact(+OpExpr, +Opts, +OutFile)
]).
:- use_module(library(lists)).

%% emit_from_fact(+OpExpr, +Opts, +OutFile): dispatch on the rope fact shape.
%% theta(T) in Opts OVERRIDES the fact's baked theta — lets the model's true
%% rope_theta flow through (qwen=500000.0; the fact default is generic).
emit_from_fact(rope(theta(FactTheta), pairing(half_split), freq(inv_theta_pow)), Opts, OutFile) :- !,
    ( member(theta(T), Opts) -> Theta = T ; Theta = FactTheta ),
    emit_rope_cuda(Theta, OutFile).

%% Derive the kernel from the fact. ONE THREAD PER (row, dim-pair j): each
%% thread reads x[row,j] and x[row,j+half], computes the rotation, writes both.
%% Grid covers nrows*half threads. Works for q and k (separate launches; nh may
%% differ between them, so n_head is a runtime arg = the tensor's head count).
%%   X layout: [nrows, n_head*hd] row-major; row r, head h, dim d at X[r*(n_head*hd) + h*hd + d]
%%   positions: int per row (absolute sequence position).
emit_rope_cuda(Theta, OutFile) :-
    open(OutFile, write, S),
    format(S, "/* GENERATED from op_expr rope(theta(~w), half_split, inv_theta_pow) — NeoX rotary.~n", [Theta]),
    format(S, "   out[j]=x[j]*cos-x[j+half]*sin; out[j+half]=x[j]*sin+x[j+half]*cos. */~n", []),
    format(S, "extern \"C\" __global__ void k_rope(~n", []),
    format(S, "    float* X,              // [nrows, n_head*hd] in/out, roped in place~n", []),
    format(S, "    const int* pos,        // [nrows] absolute sequence position per row~n", []),
    format(S, "    int nrows, int n_head, int hd) {~n", []),
    format(S, "  int half = hd / 2;~n", []),
    format(S, "  long total = (long)nrows * n_head * half;~n", []),
    format(S, "  long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;~n", []),
    format(S, "  if (gid >= total) return;~n", []),
    format(S, "  int j     = gid % half;            // dim-pair index 0..half-1~n", []),
    format(S, "  long hr   = gid / half;            // (row*n_head + head)~n", []),
    format(S, "  int head  = hr % n_head;~n", []),
    format(S, "  int row   = hr / n_head;~n", []),
    format(S, "  float p   = (float) pos[row];~n", []),
    % freq[j] = 1 / theta^(2j/hd) = theta^(-2j/hd). Theta baked from the fact/opt.
    format(S, "  float freq = powf(~wf, -(2.0f * (float)j) / (float)hd);~n", [Theta]),
    format(S, "  float ang  = p * freq;~n", []),
    format(S, "  float c = cosf(ang), s = sinf(ang);~n", []),
    format(S, "  long base = (long)row * n_head * hd + (long)head * hd;~n", []),
    format(S, "  float x1 = X[base + j];~n", []),
    format(S, "  float x2 = X[base + j + half];~n", []),
    format(S, "  X[base + j]        = x1 * c - x2 * s;~n", []),
    format(S, "  X[base + j + half] = x1 * s + x2 * c;~n", []),
    format(S, "}~n", []),
    format(S, "// LAUNCH: total threads = nrows*n_head*(hd/2)~n", []),
    close(S),
    format("Generated FACT-DERIVED RoPE (theta=~w, half-split NeoX) -> ~w~n", [Theta, OutFile]).
