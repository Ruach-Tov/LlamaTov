%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% arch_params.pl — Architecture parameter table as Prolog facts.
%%
%% Maps architecture names (from GGUF metadata) to the parameters
%% that control compute graph instantiation. Replaces the 446-case
%% switch in llama.cpp with a declarative lookup table.
%%
%% Each architecture is a point in a ~22-dimensional parameter space.
%% The graph template is ONE parameterized pattern; the parameters
%% select which variant to instantiate.

:- module(arch_params, [
    arch_family/2,           % arch_family(+ArchName, -Family)
    arch_param/3             % arch_param(+ArchName, +ParamName, -Value)
]).

%% ═══════════════════════════════════════════════════════════════
%% FAMILY CLASSIFICATION
%% ═══════════════════════════════════════════════════════════════

arch_family(llama, transformer).
arch_family(qwen2, transformer).
arch_family(falcon, transformer).
arch_family(gemma, transformer).
arch_family(gemma2, transformer).
arch_family(phi3, transformer).
arch_family(starcoder2, transformer).
arch_family(gpt2, transformer).
arch_family(bloom, transformer).
arch_family(deepseek, transformer).
arch_family(mistral, transformer).
arch_family(granite, transformer).

arch_family(mamba, ssm).
arch_family(falcon_h1, hybrid).
arch_family(jamba, hybrid).
arch_family(rwkv6, rwkv).
arch_family(rwkv7, rwkv).

%% ═══════════════════════════════════════════════════════════════
%% TRANSFORMER PARAMETERS (from gen_builder.py empirical analysis)
%% ═══════════════════════════════════════════════════════════════

%% Qwen2 — simplest transformer variant
arch_param(qwen2, norm_type, rms).
arch_param(qwen2, has_rope_factors, false).
arch_param(qwen2, kq_scale, fixed).
arch_param(qwen2, has_kq_norm, false).
arch_param(qwen2, has_ffn_bias, false).
arch_param(qwen2, has_moe, false).
arch_param(qwen2, has_output_bias, true).
arch_param(qwen2, qkv_style, separate).
arch_param(qwen2, ffn_activation, silu).
arch_param(qwen2, ffn_mode, parallel).
arch_param(qwen2, has_ffn_gate, true).
arch_param(qwen2, residual_style, single).
arch_param(qwen2, position_type, rope).

%% Llama — canonical transformer with more features
arch_param(llama, norm_type, rms).
arch_param(llama, has_rope_factors, true).
arch_param(llama, kq_scale, configurable).
arch_param(llama, has_kq_norm, true).
arch_param(llama, has_ffn_bias, true).
arch_param(llama, has_moe, true).
arch_param(llama, has_output_bias, false).
arch_param(llama, qkv_style, separate).
arch_param(llama, ffn_activation, silu).
arch_param(llama, ffn_mode, parallel).
arch_param(llama, has_ffn_gate, true).
arch_param(llama, residual_style, single).
arch_param(llama, position_type, rope).

%% Falcon — different norm, fused QKV, no gate
arch_param(falcon, norm_type, layer).
arch_param(falcon, has_rope_factors, false).
arch_param(falcon, kq_scale, fixed).
arch_param(falcon, has_kq_norm, false).
arch_param(falcon, has_ffn_bias, false).
arch_param(falcon, has_moe, false).
arch_param(falcon, has_output_bias, false).
arch_param(falcon, qkv_style, fused).
arch_param(falcon, ffn_activation, gelu).
arch_param(falcon, ffn_mode, sequential).
arch_param(falcon, has_ffn_gate, false).
arch_param(falcon, residual_style, double).
arch_param(falcon, position_type, rope).

%% ═══════════════════════════════════════════════════════════════
%% PARAMETER LOOKUP WITH DEFAULTS
%% ═══════════════════════════════════════════════════════════════

%% Default values for parameters not explicitly set
default_param(norm_type, rms).
default_param(has_rope_factors, false).
default_param(kq_scale, fixed).
default_param(has_kq_norm, false).
default_param(has_ffn_bias, false).
default_param(has_moe, false).
default_param(has_output_bias, false).
default_param(qkv_style, separate).
default_param(ffn_activation, silu).
default_param(ffn_mode, parallel).
default_param(has_ffn_gate, true).
default_param(residual_style, single).
default_param(position_type, rope).

%% Look up a parameter with fallback to default
param(Arch, Name, Value) :-
    ( arch_param(Arch, Name, V) -> Value = V
    ; default_param(Name, Value)
    ).
