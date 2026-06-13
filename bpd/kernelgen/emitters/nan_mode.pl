%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════════
%% nan_mode.pl — NaN-handling as an EXPLICIT, configurable generator parameter,
%% consistent across ALL backends (cuda-oxide Rust, C++/CUDA, CPU reference).
%%
%% NaN behavior must not be a hidden codegen assumption. The generator takes a
%% nan_mode that every backend honors identically:
%%
%%   propagate  — IEEE: NaN flows through (matches torch). Uses the
%%                NaN-preserving formulation; codegen must NOT assume no-NaNs
%%                (cuda-oxide: fcmp UNE for !=, no nnan; CUDA: no -ffast-math;
%%                 CPU: the x!=x?x:... form).
%%   fast       — assume-no-NaN: NaN behavior unspecified, faster. Uses the
%%                simpler formulation; allows nnan fast-math
%%                (cuda-oxide: nnan flag; CUDA: __expf/-use_fast_math; CPU: the
%%                 plain form).
%%
%% A fact may DECLARE its required nan_propagation (e.g. relu: ieee). The
%% generator can honor the fact (default) or override via an explicit nan_mode
%% for performance experiments — but the choice is always RECORDED in the
%% generated artifact so the bit-identity contract is intentional.
%%
%% nan_variant(+Op, +NanMode, -Form) gives the per-element form per backend via
%% the {rust,cuda,cpu}_form/3 selectors.
%%
%% Author: Iyun, 2026-06-06 (Heath: make no-NaNs fast-math a generator param)
%% ═══════════════════════════════════════════════════════════════════════════

:- module(nan_mode, [nan_form/4, default_nan_mode/2]).

%% default_nan_mode(+Op, -Mode): honor the fact's nan_propagation declaration.
%%   ieee -> propagate (the safe, torch-matching default).
default_nan_mode(Op, propagate) :-
    catch(robust_op_match(unary_elementwise, Op, _, _, Ev), _, fail),
    member(nan_propagation(ieee), Ev), !.
default_nan_mode(_, propagate).   %% safe default if unspecified

%% nan_form(+Backend, +NanMode, +BaseExpr, -Form)
%% For ops whose formulation has a conditional flush-to-zero (relu/elu),
%% propagate mode wraps with a NaN-passthrough guard; fast mode uses BaseExpr.
%% BaseExpr is the per-backend plain form over `v`.
%%
%% propagate: prepend the NaN passthrough. The selectors below produce the
%% guarded form. Transcendental ops (tanh/silu/gelu) propagate NaN inherently
%% (the math fn returns NaN for NaN), so they need no guard in either mode.

%% rust:
nan_form(rust, propagate, BaseExpr, Form) :-
    ( needs_nan_guard(BaseExpr)
    -> format(atom(Form), "if v != v { v } else { ~w }", [BaseExpr])  %% UNE-based passthrough
    ;  Form = BaseExpr ).
nan_form(rust, fast, BaseExpr, BaseExpr).

%% cuda:
nan_form(cuda, propagate, BaseExpr, Form) :-
    ( needs_nan_guard(BaseExpr)
    -> format(atom(Form), "v != v ? v : (~w)", [BaseExpr])
    ;  Form = BaseExpr ).
nan_form(cuda, fast, BaseExpr, BaseExpr).

%% cpu (the reference): same logic, C-ish for the host fn.
nan_form(cpu, propagate, BaseExpr, Form) :-
    ( needs_nan_guard(BaseExpr)
    -> format(atom(Form), "v != v ? v : (~w)", [BaseExpr])
    ;  Form = BaseExpr ).
nan_form(cpu, fast, BaseExpr, BaseExpr).

%% A form needs a NaN guard only if it contains a comparison that would flush
%% NaN (a conditional select on v). Transcendentals (exp/tanh/erf) don't.
needs_nan_guard(E) :-
    ( sub_string(E, _, _, _, ">=") ; sub_string(E, _, _, _, "> 0") ; sub_string(E, _, _, _, "<= 0") ),
    \+ sub_string(E, _, _, _, "exp"),
    \+ sub_string(E, _, _, _, "tanh"),
    \+ sub_string(E, _, _, _, "erf").
