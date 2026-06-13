%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════════
%% cuda_c_from_facts.pl — generate C++/CUDA kernels from robust_op_match.pl.
%%
%% The SECOND backend of the multi-backend generator. Reads the SAME canonical
%% facts as oxide_from_facts.pl. Because the canonical formulation is already
%% C-style (e.g. 'x >= 0 ? x : 0.0f'), the C++/CUDA backend emits it nearly
%% verbatim into a __global__ kernel — the formulation IS the kernel body.
%%
%% Multi-backend by construction: one fact -> {Rust/cuda-oxide, C++/CUDA}, both
%% bit-identical-to-the-fact, hence cross-checkable (the differential detector).
%%
%% Usage:  emit_cuda_c_from_fact(+Op, +OutFile)   -> a compile-with-nvcc .cu file
%%
%% Author: Iyun, 2026-06-06 (C++/CUDA backend of the multi-backend generator)
%% ═══════════════════════════════════════════════════════════════════════════

:- module(cuda_c_from_facts, [emit_cuda_c_from_fact/2, cuda_c_supported_op/1, emit_cuda_c_dump/4, emit_cuda_c_from_fact_mode/3, emit_cuda_c_kernel_only/2]).
:- use_module(library(lists)).

%% Load the canonical facts (robust_op_match/5), resolved relative to this
%% emitter's location (bpd/kernelgen/emitters/ -> ../../lib/robust_op_match.pl),
%% with absolute fallbacks for the enclave/laptop checkouts.
:- ( prolog_load_context(directory, Dir),
     atomic_list_concat([Dir, '/../../lib/robust_op_match.pl'], RelPath),
     exists_file(RelPath)
   -> consult(RelPath)
   ;  exists_file('lib/robust_op_match.pl')
   -> consult('lib/robust_op_match.pl')
   ;  exists_file('lib/robust_op_match.pl')
   -> consult('lib/robust_op_match.pl')
   ;  exists_file('robust_op_match.pl')
   -> consult('robust_op_match.pl')
   ;  true ).

%% ── C formulation -> CUDA-C per-element expression over `v` ─────────────────
%% The canonical formulation is already C; we map the free var x -> v and keep
%% the FP form (expf/tanhf/literal-subtract) EXACTLY (libdevice provides these
%% on the device, so expf/tanhf resolve to __nv_* — same path nvcc uses).

%% Import the shared expression IR (lower_cuda + op_expr) so cuda_c and mlir_gpu
%% generate from the SAME neutral AST. Resolve relative to this emitter.
:- ( prolog_load_context(directory, ED2),
     atomic_list_concat([ED2, '/expr_ir.pl'], EP2), exists_file(EP2)
   -> use_module(EP2, [lower_cuda/2, op_expr/2])
   ;  exists_file('kernelgen/emitters/expr_ir.pl')
   -> use_module('kernelgen/emitters/expr_ir.pl', [lower_cuda/2, op_expr/2])
   ;  exists_file('kernelgen/emitters/expr_ir.pl')
   -> use_module('kernelgen/emitters/expr_ir.pl', [lower_cuda/2, op_expr/2])
   ;  true ).

%% op_cuda_expr(+Op,+NanMode,-CudaExpr): the UNIFIED body source. Prefer the
%% shared neutral AST (op_expr -> lower_cuda, same term mlir_gpu uses); fall back
%% to the legacy formulation-string translate only for ops without an expr term.
op_cuda_expr(Op, NanMode, CudaExpr) :-
    ( op_expr(Op, Expr)
    -> lower_cuda(Expr, Raw)
    ;  robust_op_match(unary_elementwise, Op, _, _, Ev),
       member(formulation(F), Ev), formulation_to_cuda(F, Raw) ),
    apply_nan_mode_c(NanMode, Raw, CudaExpr).

%% -- nan_mode for the C++/CUDA backend (parallel to oxide) --
apply_nan_mode_c(propagate, Expr, Expr).
apply_nan_mode_c(fast, Expr, Out) :-
    ( strip_nan_guard_c(Expr, Out) -> true ; Out = Expr ).

%% strip a leading C NaN guard "v != v ? v : (<REST>)" -> <REST>
strip_nan_guard_c(Expr, Rest) :-
    atom_string(Expr, S),
    ( string_concat("v != v ? v : (", Mid, S),
      string_concat(Rest, ")", Mid)
    -> true ; fail ).

default_nan_mode_c(Op, propagate) :-
    robust_op_match(unary_elementwise, Op, _, _, Ev),
    member(nan_propagation(ieee), Ev), !.
default_nan_mode_c(_, propagate).

formulation_to_cuda(CForm, CudaExpr) :-
    atom_string(CForm, S),
    cuda_translate(S, CudaExpr).

cuda_translate("x != x ? x : (x >= 0 ? x : 0.0f)", "v != v ? v : (v >= 0.0f ? v : 0.0f)").
cuda_translate("tanhf(x)",                    "tanhf(v)").
cuda_translate("x * 0.5f * (1.0f + erff(x * 0.7071067811865476f))", "v * 0.5f * (1.0f + erff(v * 0.7071067811865476f))").
cuda_translate("x <= 0 ? expf(x) - 1.0f : x", "v <= 0.0f ? expf(v) - 1.0f : v").
cuda_translate("(scale*alpha) * (expf(x) - 1.0f) for x<=0, else scale*x",
    "v <= 0.0f ? (1.0507009873f*1.6732632423f)*(expf(v)-1.0f) : 1.0507009873f*v").
cuda_translate("x / (1 + expf(-x))",          "v / (1.0f + expf(-v))").
cuda_translate("1 / (1 + expf(-x))",          "1.0f / (1.0f + expf(-v))").

cuda_c_supported_op(Op) :-
    robust_op_match(unary_elementwise, Op, _, _, Ev),
    member(formulation(F), Ev),
    formulation_to_cuda(F, _).

%% ── emit a CUDA-C kernel + host harness for an op from its fact ─────────────
emit_cuda_c_from_fact(Op, OutFile) :-
    default_nan_mode_c(Op, Mode),
    emit_cuda_c_from_fact_mode(Op, Mode, OutFile).

emit_cuda_c_from_fact_mode(Op, NanMode, OutFile) :-
    robust_op_match(unary_elementwise, Op, Ref, Tier, Ev),
    member(formulation(F), Ev),
    op_cuda_expr(Op, NanMode, CudaExpr),       %% unified: shared AST -> lower_cuda
    (atom_concat('bpd_', Name, Op) -> true ; Name = Op),
    open(OutFile, write, S),
    format(S, '/* GENERATED from robust_op_match(~w) — canonical fact (op_expr AST).~n', [Op]),
    format(S, ' * formulation: ~w~n', [F]),
    format(S, ' * reference: ~w  tier: ~w~n', [Ref, Tier]),
    format(S, ' * C++/CUDA backend. Compile: nvcc -arch=sm_61. Run on the P4.~n */~n', []),
    format(S, '#include <cstdio>~n#include <cstdint>~n#include <cuda_runtime.h>~n~n', []),
    %% the device kernel — the formulation IS the body
    format(S, '__global__ void ~w_kernel(const float* x, float* c, int n) {~n', [Name]),
    format(S, '    int i = blockIdx.x * blockDim.x + threadIdx.x;~n', []),
    format(S, '    if (i < n) { float v = x[i]; c[i] = ~w; }~n', [CudaExpr]),
    format(S, '}~n~n', []),
    %% CPU reference (host) — same expression
    format(S, 'static float ~w_ref(float v) { return ~w; }~n~n', [Name, CudaExpr]),
    emit_cuda_host(S, Name),
    close(S),
    format("Generated C++/CUDA ~w from canonical fact -> ~w~n", [Name, OutFile]).

emit_cuda_host(S, Name) :-
    format(S, 'int main() {~n', []),
    format(S, '    printf("=== GENERATED-FROM-FACT C++/CUDA ~w on Tesla P4 ===\\n\\n");~n', [Name]),
    %% same edge-case + range test set as the cuda-oxide harness (for cross-check)
    format(S, '    const int EDGE=9; const int n=1024;~n', []),
    format(S, '    float xh[n];~n', []),
    format(S, '    xh[0]=0.0f; xh[1]=-0.0f; xh[2]=NAN; xh[3]=-1.0f; xh[4]=1.0f;~n', []),
    format(S, '    xh[5]=1.17549435e-38f; xh[6]=-1.17549435e-38f; xh[7]=3.4e38f; xh[8]=-3.4e38f;~n', []),
    format(S, '    for (int i=0;i<n-EDGE;i++){ float t=(float)i*0.013f-6.6f; xh[i+EDGE]=t*((i%3==0)?-1.0f:1.0f); }~n', []),
    format(S, '    float *xd,*cd; cudaMalloc(&xd,n*4); cudaMalloc(&cd,n*4);~n', []),
    format(S, '    cudaMemcpy(xd,xh,n*4,cudaMemcpyHostToDevice);~n', []),
    format(S, '    int blk=256, grd=(n+blk-1)/blk;~n', []),
    format(S, '    ~w_kernel<<<grd,blk>>>(xd,cd,n); cudaDeviceSynchronize();~n', [Name]),
    format(S, '    float ch[n]; cudaMemcpy(ch,cd,n*4,cudaMemcpyDeviceToHost);~n', []),
    format(S, '    int diffs=0, first=-1;~n', []),
    format(S, '    for (int i=0;i<n;i++){ float want=~w_ref(xh[i]);~n', [Name]),
    format(S, '        uint32_t wb=*(uint32_t*)&want, gb=*(uint32_t*)&ch[i];~n', []),
    format(S, '        int eq = (wb==gb) || (isnan(want)&&isnan(ch[i]));~n', []),
    format(S, '        if(!eq){ if(first<0) first=i; diffs++; } }~n', []),
    format(S, '    if(diffs==0) printf("  *** 0 differences — BIT-IDENTICAL (%d elems) ***\\n", n);~n', []),
    format(S, '    else { printf("  %d differ. first idx %d x=%g\\n", diffs, first, xh[first]); return 1; }~n', []),
    format(S, '    return 0;~n}~n', []).

%% ── dump-mode: a CUDA-C runner that reads input.bin, writes <op>_cudac.bin ──
%% Used by the multi-backend referee for cross-backend comparison on a fixed input.
emit_cuda_c_dump(Op, InBin, OutBin, OutFile) :-
    op_cuda_expr(Op, propagate, CudaExpr),  %% unified AST (re-derives from op_expr)
    (atom_concat(bpd_, Name, Op) -> true ; Name = Op),
    open(OutFile, write, S),
    format(S, "#include <cstdio>~n#include <cstdlib>~n#include <cuda_runtime.h>~n~n", []),
    format(S, "__global__ void k(const float* x, float* c, int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){float v=x[i]; c[i]=~w;}}~n~n", [CudaExpr]),
    format(S, "int main(){FILE*f=fopen(\"~w\",\"rb\"); fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);~n", [InBin]),
    format(S, "  int n=sz/4; float*xh=(float*)malloc(sz); fread(xh,4,n,f); fclose(f);~n", []),
    format(S, "  float*xd,*cd; cudaMalloc(&xd,sz); cudaMalloc(&cd,sz); cudaMemcpy(xd,xh,sz,cudaMemcpyHostToDevice);~n", []),
    format(S, "  k<<<(n+255)/256,256>>>(xd,cd,n); cudaDeviceSynchronize();~n", []),
    format(S, "  float*ch=(float*)malloc(sz); cudaMemcpy(ch,cd,sz,cudaMemcpyDeviceToHost);~n", []),
    format(S, "  FILE*o=fopen(\"~w\",\"wb\"); fwrite(ch,4,n,o); fclose(o); return 0;}~n", [OutBin]),
    close(S),
    format("dump-runner ~w -> ~w~n", [Name, OutFile]).

%% ── emit a KERNEL-ONLY .cu (no host) matching the perf_fixture ABI ──────────
%% Signature: extern "C" __global__ void <op>(const float* src, float* dst, long n)
%% Compiled to a cubin (nvcc -cubin) + loaded by perf_fixture for timing.
emit_cuda_c_kernel_only(Op, OutFile) :-
    op_cuda_expr(Op, propagate, CudaExpr),  %% unified AST (re-derives from op_expr)
    (atom_concat('bpd_', Name, Op) -> true ; Name = Op),
    open(OutFile, write, S),
    format(S, '/* GENERATED kernel-only from robust_op_match(~w) for perf_fixture. */~n', [Op]),
    format(S, 'extern "C" __global__ void k_~w(const float* src, float* dst, long n) {~n', [Name]),
    format(S, '    long i = (long)(blockIdx.x) * blockDim.x + threadIdx.x;~n', []),
    format(S, '    if (i < n) { float v = src[i]; dst[i] = ~w; }~n', [CudaExpr]),
    format(S, '}~n', []),
    close(S),
    format("Generated cuda-c kernel-only ~w -> ~w~n", [Name, OutFile]).
