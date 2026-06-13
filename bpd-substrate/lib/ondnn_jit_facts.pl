%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ondnn_jit_facts.pl — The oneDNN gelu_erf JIT kernel as Prolog facts.
%%
%% Each fact represents one x86 instruction from the disassembly.
%% The LLVM IR emitter walks these facts and generates equivalent IR.
%%
%% This is the BPD thesis: the disassembly IS the specification,
%% Prolog facts ARE the representation, LLVM IR IS the projection.
%%
%% Register tracking: which ymm register holds which value at each point.
%%
%% Constant table (r9 + offset):
%%   0x080 = 0.5,  0x0a0 = 1.0,  0x0c0 = 2.0
%%   0x120 = ln2,  0x140 = abs_mask(0x7FFFFFFF),  0x160 = sign_mask(0x80000000)
%%   0x180 = i32 bias (127),  0x1a0 = log2e,  0x1c0 = exp_hi,  0x1e0 = exp_lo
%%   0x200 = exp_c1,  0x220 = exp_c2,  0x240 = exp_c3,  0x260 = exp_c4,  0x280 = exp_c5
%%   0x2a0 = erf_p,  0x2c0 = sqrt2_inv
%%   0x300 = erf_a1,  0x320 = erf_a2,  0x340 = erf_a3,  0x360 = erf_a4,  0x380 = erf_a5

:- module(ondnn_jit_facts, [jit_insn/4, jit_const/3]).

%% jit_insn(Offset, Op, Dst, Src) — one instruction
%% We track the data flow, not the x86 encoding.

%% Constants
jit_const(0x080, half, '0x3FE0000000000000').
jit_const(0x0a0, one, '0x3FF0000000000000').
jit_const(0x0c0, two, '0x4000000000000000').
jit_const(0x120, ln2, '0x3FE62E4300000000').
jit_const(0x140, abs_mask, 'i32 2147483647').
jit_const(0x160, sign_mask, 'i32 -2147483648').
jit_const(0x180, bias_127, 'i32 127').
jit_const(0x1a0, log2e, '0x3FF7154760000000').
jit_const(0x1c0, exp_hi, '0x40562E4300000000').
jit_const(0x1e0, exp_lo, '0xC055D58A00000000').
jit_const(0x200, exp_c1, '0x3FEFFFFF60000000').
jit_const(0x220, exp_c2, '0x3FDFFFDC60000000').
jit_const(0x240, exp_c3, '0x3FC555A800000000').
jit_const(0x260, exp_c4, '0x3FA573A1A0000000').
jit_const(0x280, exp_c5, '0x3F80F9F9C0000000').
jit_const(0x2a0, erf_p, '0x3FD4F740A0000000').
jit_const(0x2c0, sqrt2_inv, '0x3FE6A09E60000000').
jit_const(0x300, erf_a1, '0x3FD04F20C0000000').
jit_const(0x320, erf_a2, '0xBFD23531C0000000').
jit_const(0x340, erf_a3, '0x3FF6BE1C60000000').
jit_const(0x360, erf_a4, '0xBFF7401C60000000').
jit_const(0x380, erf_a5, '0x3FF0FB8440000000').

%% === PHASE 1: Setup (0x38-0x75) ===
%% x_saved = input[i]
jit_insn(0x38, load,    x,      '[input]').
jit_insn(0x3c, mov,     x_saved, x).
%% v = x * sqrt2_inv
jit_insn(0x40, fmul,    v,       x * sqrt2_inv).
%% |v| = v AND abs_mask
jit_insn(0x4d, and_abs, av,      v * abs_mask).   % vandps (combines mov+and)
%% t = 1.0 / (1.0 + p * |v|)
jit_insn(0x5f, fmul,    pav,     av * erf_p).       % p * |v|
jit_insn(0x63, fadd,    denom,   pav + one).         % 1 + p*|v|
jit_insn(0x75, fdiv,    t,       one / denom).       % 1.0 / (1+p*|v|)

%% === PHASE 2: -v² and exp setup (0x79-0x9d) ===
jit_insn(0x79, fmul,    v_sq,    v * v).             % v²
jit_insn(0x7d, xor_neg, neg_v2,  v_sq * sign_mask). % -v²
jit_insn(0x94, fmin,    xc_hi,   neg_v2 * exp_hi).  % clamp high
jit_insn(0x9d, fmax,    xc,      xc_hi * exp_lo).   % clamp low

%% === PHASE 3: Exp polynomial (0xa1-0x170) ===
jit_insn(0xa1, mov,     xc_saved, xc).
jit_insn(0xa5, fmul,    z_raw,   xc * log2e).        % xc * log2e
jit_insn(0xae, fadd,    z,       z_raw + half).       % + 0.5
jit_insn(0xb7, floor,   n,       z).                  % floor(z+0.5)
jit_insn(0xc1, fmul,    nln2,    n * ln2).            % n * ln2
jit_insn(0xca, fsub,    f,       xc_saved - nln2).    % f = xc - n*ln2
jit_insn(0xce, fsub,    n_adj,   n - one).            % n_adj = n - 1
%% Convert and scale — simplified for <8 x i32> (LLVM handles AVX1 split)
jit_insn(0xd7, cvt,     ni,      n_adj).              % fptosi <8 x float> to <8 x i32>
jit_insn(0xe1, iadd,    ni_biased, ni + bias_127).    % add 127
jit_insn(0xff, ishl,    scale_i, ni_biased * 23).     % shift to exponent field
jit_insn(0x109, bitcast_i2f, scale, scale_i).         % reinterpret as float
%% Exp Horner polynomial in f: ((((c5*f+c4)*f+c3)*f+c2)*f+c1)*f+1
jit_insn(0x122, fmul,   ep0,     f * exp_c5).
jit_insn(0x126, fadd,   ep1,     ep0 + exp_c4).
jit_insn(0x12f, fmul,   ep2,     ep1 * f).
jit_insn(0x133, fadd,   ep3,     ep2 + exp_c3).
jit_insn(0x13c, fmul,   ep4,     ep3 * f).
jit_insn(0x140, fadd,   ep5,     ep4 + exp_c2).
jit_insn(0x149, fmul,   ep6,     ep5 * f).
jit_insn(0x14d, fadd,   ep7,     ep6 + exp_c1).
jit_insn(0x156, fmul,   ep8,     ep7 * f).
jit_insn(0x15a, fadd,   exp_poly, ep8 + one).
%% Final exp: poly * scale * 2.0
jit_insn(0x163, fmul,   exp_raw, exp_poly * scale).
jit_insn(0x167, fmul,   exp_val, exp_raw * two).
%% Negate exp
jit_insn(0x170, xor_neg, neg_exp, exp_val * sign_mask).

%% === PHASE 4: Erf Abramowitz (0x179-0x1d4) ===
%% Extract sign bit of original x
jit_insn(0x17d, and_sign, x_sign, x_saved * sign_mask). % sign bit of x
%% (-exp) * t
jit_insn(0x186, fmul,   et,      neg_exp * t).
%% Horner4: ((((a5*t + a4)*t + a3)*t + a2)*t + a1)
jit_insn(0x193, fmul,   hp0,     t * erf_a5).
jit_insn(0x197, fadd,   hp1,     hp0 + erf_a4).
jit_insn(0x1a0, fmul,   hp2,     hp1 * t).
jit_insn(0x1a4, fadd,   hp3,     hp2 + erf_a3).
jit_insn(0x1ad, fmul,   hp4,     hp3 * t).
jit_insn(0x1b1, fadd,   hp5,     hp4 + erf_a2).
jit_insn(0x1ba, fmul,   hp6,     hp5 * t).
jit_insn(0x1be, fadd,   hp7,     hp6 + erf_a1).
%% (-exp*t) * horner4
jit_insn(0x1c7, fmul,   eth,     et * hp7).
%% + 1.0 (since et is negative: 1 + (-exp*t*h) = 1 - exp*t*h = erf_abs)
jit_insn(0x1cb, fadd,   erf_abs, eth + one).
%% XOR with sign of x
jit_insn(0x1d4, xor_sign, erf_signed, erf_abs * x_sign).

%% === PHASE 5: Gelu assembly (0x1d8-0x1e5) ===
jit_insn(0x1d8, fmul,   x_half,  x_saved * half).     % x * 0.5
jit_insn(0x1e1, fmul,   prod,    erf_signed * x_half). % erf * (x*0.5)
jit_insn(0x1e5, fadd,   result,  prod + x_half).       % + x*0.5
