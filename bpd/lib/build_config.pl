%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% build_config.pl — the CPU substrate build configuration, as declarative facts.
%%
%% Per Heath's direction: the build config (ISA flags) is a PARAMETER, represented
%% in Prolog, not a magic string buried in a Makefile. The asm_facts+gdb pipeline
%% (2026-05-31) proved that omitting the ISA flag makes gcc EMULATE SIMD intrinsics
%% with slow code (punpcklbw+pmaddwd) instead of emitting the real instruction
%% (pmaddubsw). So the build flags are LOAD-BEARING for performance AND must match
%% the target ISA. This module makes that contract queryable + checkable.

:- module(build_config, [
    cpu_target/1,
    cpu_isa_feature/2,        % cpu_isa_feature(Target, Feature)
    build_flag/3,             % build_flag(Target, Profile, Flag)
    build_profile/2,          % build_profile(Profile, Description)
    required_isa_for/2,       % required_isa_for(Intrinsic, Feature)
    build_command/3           % build_command(Target, Profile, FlagString)
]).

%% ─── The target CPU (this box) ───
cpu_target(ivy_bridge_e5_2697_v2).

%% ISA features the target ACTUALLY has (lifted from /proc/cpuinfo flags).
%% Ivy Bridge: AVX1, SSSE3, SSE4.1/4.2, F16C — but NO AVX2, NO FMA.
cpu_isa_feature(ivy_bridge_e5_2697_v2, avx).
cpu_isa_feature(ivy_bridge_e5_2697_v2, ssse3).
cpu_isa_feature(ivy_bridge_e5_2697_v2, sse4_1).
cpu_isa_feature(ivy_bridge_e5_2697_v2, sse4_2).
cpu_isa_feature(ivy_bridge_e5_2697_v2, f16c).
%% Explicitly NOT present (these would SIGILL):
%%   NOT avx2, NOT fma, NOT avx512.

%% ─── Build profiles (the compile flags as facts) ───
build_profile(strict_slow, 'gcc -O2 only. PERF TRAP: SIMD intrinsics EMULATED, not emitted.').
build_profile(isa_matched, 'gcc -O2 with target ISA flags. Intrinsics emit real instructions.').

%% strict_slow: the old default (CPU_FP_strict=-O2). Compiles but slow.
build_flag(ivy_bridge_e5_2697_v2, strict_slow, '-O2').

%% isa_matched: the CORRECT profile. -mavx -mf16c -mssse3 enable the intrinsics our
%% kernels use (_mm_maddubs_epi16 = SSSE3, _mm256_* = AVX, f16<->f32 = F16C).
%% Explicitly -mno-avx2 -mno-fma to match the target (and prevent gcc auto-vec from
%% emitting AVX2/FMA that would SIGILL).
build_flag(ivy_bridge_e5_2697_v2, isa_matched, '-O2').
build_flag(ivy_bridge_e5_2697_v2, isa_matched, '-mavx').
build_flag(ivy_bridge_e5_2697_v2, isa_matched, '-mf16c').
build_flag(ivy_bridge_e5_2697_v2, isa_matched, '-mssse3').
build_flag(ivy_bridge_e5_2697_v2, isa_matched, '-mno-avx2').
build_flag(ivy_bridge_e5_2697_v2, isa_matched, '-mno-fma').
build_flag(ivy_bridge_e5_2697_v2, isa_matched, '-funroll-loops').  %% unroll q8_0 dot loops (decode 259->229 ms/tok, bit-identical)

%% ─── Which ISA feature each SIMD intrinsic REQUIRES to emit the real instruction ───
%% (Without the feature, gcc emulates -> slow. This is the maddubs lesson, encoded.)
required_isa_for('_mm_maddubs_epi16', ssse3).       % -> pmaddubsw (else punpcklbw+pmaddwd emulation)
required_isa_for('_mm_sign_epi8', ssse3).           % -> psignb
required_isa_for('_mm256_loadu_ps', avx).           % -> vmovups (256-bit)
required_isa_for('_mm256_add_ps', avx).
required_isa_for('_mm256_mul_ps', avx).
required_isa_for('_cvtsh_ss', f16c).                % f16->f32 hardware
required_isa_for('_mm_madd_epi16', sse2).           % -> pmaddwd (baseline)

%% ─── build_command/3: assemble the flag string for a target+profile ───
build_command(Target, Profile, FlagString) :-
    findall(F, build_flag(Target, Profile, F), Flags),
    atomic_list_concat(Flags, ' ', FlagString).
