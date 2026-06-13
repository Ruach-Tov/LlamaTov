%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ir_match_recipes.pl — Canonical 0-ULP bit-exact formulations for BPD ops vs PyTorch.
%%
%% Substrate-of-source: Iyun's empirical 0-ULP bisection against torch 2.7.0 CPU references
%% (2026-05-31), flags -mavx -mf16c -mno-avx2 -mno-fma (Ivy Bridge / Sandybridge tier).
%% Each recipe was found by capturing torch's EXACT output bits and bisecting candidate
%% formulations until 0 ULP across robust vectors (edges/denormals/extremes).
%%
%% Substrate-of-purpose: declarative facts that drive (a) ir_compare diagram rendering
%% (mavhir), (b) the BPD LLVM emitter's choice of opcode sequence, (c) the divergence-map
%% verification. The CENTRAL finding: the same math with different OPCODES gives different
%% bits, so the exact formulation (divide vs reciprocal-mul, f32 vs f64 accumulate) IS the
%% specification. You cannot guess it; it is measured against the reference.
%%
%% This is tracked source under git. Outputs that consume it (diagrams) are .o-infixed in
%% /tmp/output-only/.

:- module(ir_match_recipes, [
     ir_match_recipe/4,        % ir_match_recipe(Op, Reference, Formulation, Props)
     reduction_acc_type/2,     % reduction_acc_type(Reference, AccType)
     opcode_seq/2              % opcode_seq(Op, ListOfLLVMOpcodes)
   ]).


%% clauses grouped by family, not contiguous — declared for warning-free consult.
:- discontiguous ir_match_recipe/4.
%% ── reduction accumulation rule (governs the whole reduce/norm class) ──
%% PyTorch CPU ATen reductions use acc_type<float> = DOUBLE. ggml uses f32 throughout.
%% The REFERENCE dictates the accumulator precision.
reduction_acc_type(pytorch_cpu, double).   %% Σ in f64, cast to f32 (rms/layernorm/softmax-sum/mean/var)
reduction_acc_type(ggml,        float).     %% Σ in f32 throughout (also 0-ULP for ITS own dumps)

%% ── proven 0-ULP recipes (Op, Reference, Formulation-string, Props) ──
%% Props: list of formulation(Kind), acc(Type), ulp(N), robust(Bool), note(Text)

ir_match_recipe(silu, pytorch_cpu,
    'x / (1.0f + expf(-x))',
    [ formulation(divide), acc(f32), ulp(0), robust(true),
      note('DIVIDE not reciprocal-mul: x*(1/(1+expf)) is 1 ULP off. clang expf == torch expf.') ]).

ir_match_recipe(rms_norm, pytorch_cpu,
    'ss = Sigma (double)x_i^2 ; ms = (float)(ss/n) ; scale = 1.0f/sqrtf(ms+eps) ; out = x*scale',
    [ formulation(divide_form), acc(f64), ulp(0), robust(true), eps(1.0e-5),
      note('DOUBLE accumulate the sum-of-squares. f32-acc is 2 ULP off. Source bpd_rmsnorm_cpu uses f32-acc -> nonzero ULP vs torch.') ]).

ir_match_recipe(softmax, pytorch_cpu,
    'mx=max(x) ; e_i=expf(x_i-mx) ; s=Sigma e_i ; out_i = e_i * (1.0f/s)',
    [ formulation(reciprocal_mul), acc(f32), ulp(0), robust(true),
      note('RECIPROCAL-MULTIPLY (opposite of silu): tmp=1/s computed once, then per-element multiply. e_i/s (divide) is 1 ULP off. max-subtract for stability.') ]).

ir_match_recipe(gelu, pytorch_cpu,
    'erf-form: 0.5*x*(1+erff(x/sqrt(2)))',
    [ formulation(erf_form), acc(f32), ulp(unverified), robust(false),
      note('F.gelu DEFAULT is the ERF form. The tanh-approx form (approximate=tanh) is a DIFFERENT op = mingpt_newgelu, which IS 0-ULP separately. Form-selection is the fix.') ]).

%% ── LLVM opcode sequences (the IR-match target the emitter must produce) ──
opcode_seq(silu, [fneg, 'call @expf', fadd, fdiv]).
opcode_seq(softmax_tail, ['fdiv(1.0,s) hoisted', 'fmul per-element']).
opcode_seq(rms_norm, ['fpext->f64 per term', 'fmul/fadd in f64', 'fptrunc->f32', 'fdiv 1/sqrtf']).

%% ── matmul reduction-order invariant (the gemm story, for completeness) ──
%% 0-ULP preserved by reschedules keeping per-output k-accumulation ORDER.
%% BROKEN by changing reduction GROUPING (multi-accumulator, k-blocking with partials).
%% Reference matmuls are SIZE-DEPENDENT block-structured: OpenBLAS Sandybridge kc, ggml tinyBLAS BK.
ir_match_recipe(gemm, openblas_sandybridge,
    'C[i,j] += Sigma over kc-blocks of (sequential f32 sum within block); kc per param.h',
    [ formulation(kc_blocked), acc(f32), ulp(0), robust(true),
      locked_axis(k_reduction), free_axis(m_n_schedule),
      note('kc=256 matched at SZ=512 empirically; param.h SANDYBRIDGE Q=384,P(mc)=768,UNROLL 16x4. 1024 needs multi-level. Sweep mc/nc/MR/NR + packing on FREE axes -> 0-ULP perf.') ]).

ir_match_recipe(gemm, ggml_b5311,
    'tinyBLAS_Q0_AVX gemm<4,2,BM=4> tiled accumulation',
    [ formulation(tiled_tinyblas), acc(f32), ulp(0), robust(true),
      locked_axis(k_reduction), free_axis(m_n_schedule),
      note('Pinned llama.cpp 51fb96b tag b5311. The dump is the TILED path, not per-row vec_dot.') ]).
