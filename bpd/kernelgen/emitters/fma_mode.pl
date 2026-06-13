%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════════
%% fma_mode.pl — FMA contraction as an explicit, configurable generator
%% parameter, consistent across all backends. Parallel to nan_mode.
%%
%% For reductions / multiply-accumulate (matmul, dot, FFN), the bit-identity
%% contract is determined by FMA contraction:
%%
%%   strict   — a*b + c stays as separate mul + add (TWO roundings).
%%              Matches torch-CPU bit-exact. cuda-oxide: default (no contract);
%%              CUDA: -fmad=false; CPU: a*b+c.
%%   contract — fuse to fma(a,b,c) (ONE rounding). Faster; matches nvcc -O3 /
%%              cuBLAS-style. cuda-oxide: f32::mul_add -> __nv_fmaf;
%%              CUDA: -fmad=true; CPU: fmaf(a,b,c).
%%
%% EMPIRICALLY VERIFIED (256x256 naive GEMM, P4):
%%   oxide-strict == nvcc(-fmad=false) == torch.matmul   (0-ULP)
%%   oxide-contract(mul_add) == nvcc(-fmad=true)         (0-ULP)
%%   strict vs contract: 52238/65536 differ (the knob is real)
%%
%% mac_form(+Backend, +FmaMode, +A, +B, +Acc, -Form) gives the multiply-
%% accumulate expression per backend per mode.
%%
%% Author: Iyun, 2026-06-07 (Heath: add the fma_mode parameter)
%% ═══════════════════════════════════════════════════════════════════════════

:- module(fma_mode, [mac_form/6, nvcc_fma_flag/2, default_fma_mode/2]).

%% default_fma_mode(+Op, -Mode): a fact may declare fma(strict|contract).
%%   default strict (the torch-matching, reproducible contract).
default_fma_mode(Op, Mode) :-
    catch(robust_op_match(_, Op, _, _, Ev), _, fail),
    member(coordinates_pinned(Coords), Ev),
    member(fma(M), Coords), !, Mode = M.
default_fma_mode(_, strict).

%% mac_form(+Backend, +FmaMode, +AExpr, +BExpr, +AccExpr, -Form)
%% Acc = Acc + A*B, expressed per the mode.
mac_form(rust, strict,   A, B, Acc, Form) :-
    format(atom(Form), "~w + ~w * ~w", [Acc, A, B]).
mac_form(rust, contract, A, B, Acc, Form) :-
    format(atom(Form), "~w.mul_add(~w, ~w)", [A, B, Acc]).   %% a.mul_add(b, acc) = a*b+acc, one rounding
mac_form(cuda, strict,   A, B, Acc, Form) :-
    format(atom(Form), "~w + ~w * ~w", [Acc, A, B]).         %% -fmad=false prevents fusion
mac_form(cuda, contract, A, B, Acc, Form) :-
    format(atom(Form), "fmaf(~w, ~w, ~w)", [A, B, Acc]).
mac_form(cpu, strict,    A, B, Acc, Form) :-
    format(atom(Form), "~w + ~w * ~w", [Acc, A, B]).
mac_form(cpu, contract,  A, B, Acc, Form) :-
    format(atom(Form), "fmaf(~w, ~w, ~w)", [A, B, Acc]).

%% nvcc compile flag for the C++/CUDA backend.
nvcc_fma_flag(strict,   "-fmad=false").
nvcc_fma_flag(contract, "-fmad=true").
