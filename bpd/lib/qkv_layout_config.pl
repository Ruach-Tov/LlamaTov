%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% qkv_layout_config.pl — PARAMETERIZED qkv layout fix at the source (Iyun, 2026-05-29, Heath).
%% The qkv error source is fixed NOT by a hardcoded patch but by PARAMETERS + their SETTINGS:
%% the qkv kernel carries layout parameters; the adapter derives the weave per setting; the same
%% kernel is correct for any (model, target) by choosing the setting. Our target (Ollama=ggml) is
%% the DEFAULT setting -> weave [] -> 0 ULP (already correct). Other settings reconcile to other refs.

:- module(qkv_layout_config,
    [ qkv_param/3, qkv_setting/2, qkv_weave_for/2, qkv_default_setting/1, qkv_sweep_grid/2 ]).
:- use_module(library(lists)).
:- use_module(layout_adapter, []).

%% ── THE PARAMETERS (the degrees of freedom of the qkv layout fix) ──
%% qkv_param(Name, Values, Default). These are the knobs; a SETTING binds each to a value.
qkv_param(rope_layout,  [ggml_interleave, hf_split_half, neox],  ggml_interleave).
qkv_param(target,       [sm_61, sm_80, sm_89, sse3, riscv_v, systolic_8x8],  sm_61).
qkv_param(dataflow,     [row_major, col_major],  row_major).
qkv_param(head_dim,     [64, 128, 256],  64).   % per-head rotary block size (the permute granularity)

%% ── SETTINGS (named bindings of the parameters) ──
%% qkv_setting(Name, [param=value,...]). A setting is a point in the parameter space.
%% DEFAULT = our actual target: Ollama=ggml on GPU. Native, no weave, 0 ULP.
qkv_setting(ollama_ggml_gpu, [rope_layout=ggml_interleave, target=sm_61, dataflow=row_major, head_dim=64]).
%% reference settings (reconcile to a given reference's convention):
qkv_setting(hf_reference,    [rope_layout=hf_split_half, target=sm_61, dataflow=row_major, head_dim=64]).
%% example non-uniform targets (the parameterized drop-in cases):
qkv_setting(sse3_pooled,     [rope_layout=hf_split_half, target=sse3, dataflow=row_major, head_dim=64]).
qkv_setting(systolic_grid,   [rope_layout=ggml_interleave, target=systolic_8x8, dataflow=col_major, head_dim=64]).

qkv_default_setting(ollama_ggml_gpu).

%% ── THE FIX: derive the weave for a setting (parameter-driven, via the adapter) ──
%% qkv_weave_for(SettingName, ModelConv, Weave). Looks up the setting's rope_layout + target,
%% asks the adapter for the reconciling weave. For the DEFAULT (ggml/sm_61) -> [] (0 ULP, correct).
qkv_weave_for(SettingName, Weave) :-
    qkv_setting(SettingName, Params),
    member(rope_layout=Conv, Params), member(target=Target, Params),
    catch(layout_adapter:layout_adapter(qkv, Conv, Target, Weave), _, Weave=[]).

%% ── THE SWEEP GRID: enumerate all settings for optimization (per target, scored stall-vs-flow) ──
%% qkv_sweep_grid(Target, Settings) — the points to evaluate. Each is (setting, derived-weave),
%% to be scored by a stall-vs-flow model (CUPTI-style) and the winner chosen per (model, target).
qkv_sweep_grid(Target, Grid) :-
    findall(setting(N, Conv, Weave),
        ( qkv_setting(N, Ps), member(target=Target, Ps), member(rope_layout=Conv, Ps),
          ( catch(layout_adapter:layout_adapter(qkv, Conv, Target, Weave),_,fail) -> true ; Weave=[] ) ),
        Grid).
