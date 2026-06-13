%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════════
%% gemm_from_facts.pl — generate naive GEMM kernels (cuda-oxide Rust + C++/CUDA)
%% from the bpd_matmul reduction fact, honoring fma_mode.
%%
%% The accumulation order is pinned (k-ascending serial) by the fact; fma_mode
%% selects mul+add (strict, matches torch) vs fused fma (contract, matches
%% nvcc -O3). The same fact drives both backends.
%%
%% Author: Iyun, 2026-06-07 (matmul as a fact-driven, contract-pinned op)
%% ═══════════════════════════════════════════════════════════════════════════

:- module(gemm_from_facts, [emit_gemm_oxide/3, emit_gemm_cuda/3, emit_gemm_kernel_only/2, emit_gemm_verify/5]).
:- use_module(library(lists)).
:- use_module(fma_mode).

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

%% emit_gemm_oxide(+FmaMode, +OutBin, +OutFile): cuda-oxide GEMM dump runner.
emit_gemm_oxide(FmaMode, OutBin, OutFile) :-
    mac_form(rust, FmaMode, "a[row * ns + k]", "b[k * ns + col]", "sum", Mac),
    open(OutFile, write, S),
    format(S, "// GENERATED from robust_op_match(reduction, bpd_matmul). fma_mode=~w~n", [FmaMode]),
    format(S, "// accumulation: k-ascending serial (pinned). Reads A.bin,B.bin -> ~w~n", [OutBin]),
    format(S, "use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};~n", []),
    format(S, "use cuda_device::{DisjointSlice, cuda_module, kernel, thread};~n", []),
    format(S, "use std::io::{Read, Write};~n~n", []),
    format(S, "#[cuda_module]~nmod kernels {~n    use super::*;~n    #[kernel]~n", []),
    format(S, "    pub fn gemm(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>, n: u32) {~n", []),
    format(S, "        let idx = thread::index_1d(); let lin = idx.get(); let ns = n as usize;~n", []),
    format(S, "        if lin < ns*ns {~n            let row = lin/ns; let col = lin%ns;~n", []),
    format(S, "            let mut sum = 0.0f32; let mut k = 0usize;~n", []),
    format(S, "            while k < ns { sum = ~w; k += 1; }~n", [Mac]),
    format(S, "            if let Some(ce) = c.get_mut(idx) { *ce = sum; }~n        }~n    }~n}~n~n", []),
    gemm_oxide_host(S, OutBin),
    close(S),
    format("Generated cuda-oxide GEMM (fma=~w) -> ~w~n", [FmaMode, OutFile]).

gemm_oxide_host(S, OutBin) :-
    format(S, "fn main() {~n    let n: usize = 256;~n", []),
    format(S, "    let mut ba=Vec::new(); std::fs::File::open(\"/tmp/gpu-work/referee/A.bin\").unwrap().read_to_end(&mut ba).unwrap();~n", []),
    format(S, "    let mut bb=Vec::new(); std::fs::File::open(\"/tmp/gpu-work/referee/B.bin\").unwrap().read_to_end(&mut bb).unwrap();~n", []),
    format(S, "    let a: Vec<f32> = ba.chunks_exact(4).map(|x| f32::from_le_bytes([x[0],x[1],x[2],x[3]])).collect();~n", []),
    format(S, "    let b: Vec<f32> = bb.chunks_exact(4).map(|x| f32::from_le_bytes([x[0],x[1],x[2],x[3]])).collect();~n", []),
    format(S, "    let ctx=CudaContext::new(0).unwrap(); let stream=ctx.default_stream();~n", []),
    format(S, "    let ad=DeviceBuffer::from_host(&stream,&a).unwrap(); let bd=DeviceBuffer::from_host(&stream,&b).unwrap();~n", []),
    format(S, "    let mut cd=DeviceBuffer::<f32>::zeroed(&stream,n*n).unwrap();~n", []),
    format(S, "    let m=kernels::load(&ctx).unwrap();~n", []),
    format(S, "    m.gemm(&stream, LaunchConfig::for_num_elems((n*n) as u32), &ad,&bd,&mut cd, n as u32).unwrap();~n", []),
    format(S, "    let ch=cd.to_host_vec(&stream).unwrap();~n", []),
    format(S, "    let mut o=std::fs::File::create(\"~w\").unwrap();~n", [OutBin]),
    format(S, "    for v in &ch { o.write_all(&v.to_le_bytes()).unwrap(); }~n", []),
    format(S, "    println!(\"gemm dumped {}x{}\", n, n);~n}~n", []).

%% emit_gemm_cuda(+FmaMode, +OutBin, +OutFile): C++/CUDA GEMM dump runner.
%% NB: the nvcc compile flag (nvcc_fma_flag) must match FmaMode.
emit_gemm_cuda(FmaMode, OutBin, OutFile) :-
    mac_form(cuda, FmaMode, "A[row*N+k]", "B[k*N+col]", "sum", Mac),
    open(OutFile, write, S),
    format(S, "/* GENERATED from robust_op_match(reduction, bpd_matmul). fma_mode=~w */~n", [FmaMode]),
    format(S, "#include <cstdio>~n#include <cstdlib>~n#include <cuda_runtime.h>~n~n", []),
    format(S, "__global__ void gemm(const float* A, const float* B, float* C, int N) {~n", []),
    format(S, "  int row=blockIdx.y*blockDim.y+threadIdx.y; int col=blockIdx.x*blockDim.x+threadIdx.x;~n", []),
    format(S, "  if(row<N&&col<N){ float sum=0.0f; for(int k=0;k<N;k++) sum=~w; C[row*N+col]=sum; }~n}~n~n", [Mac]),
    format(S, "int main(int argc,char**argv){ int N=argc>1?atoi(argv[1]):256; size_t b=(size_t)N*N*4;~n", []),
    format(S, "  float*Ah=(float*)malloc(b),*Bh=(float*)malloc(b),*Ch=(float*)malloc(b);~n", []),
    format(S, "  FILE*fa=fopen(\"/tmp/gpu-work/referee/A.bin\",\"rb\");fread(Ah,4,(size_t)N*N,fa);fclose(fa);~n", []),
    format(S, "  FILE*fb=fopen(\"/tmp/gpu-work/referee/B.bin\",\"rb\");fread(Bh,4,(size_t)N*N,fb);fclose(fb);~n", []),
    format(S, "  float*Ad,*Bd,*Cd; cudaMalloc(&Ad,b);cudaMalloc(&Bd,b);cudaMalloc(&Cd,b);~n", []),
    format(S, "  cudaMemcpy(Ad,Ah,b,cudaMemcpyHostToDevice);cudaMemcpy(Bd,Bh,b,cudaMemcpyHostToDevice);~n", []),
    format(S, "  dim3 blk(16,16),grd((N+15)/16,(N+15)/16); gemm<<<grd,blk>>>(Ad,Bd,Cd,N); cudaDeviceSynchronize();~n", []),
    format(S, "  cudaMemcpy(Ch,Cd,b,cudaMemcpyDeviceToHost);~n", []),
    format(S, "  FILE*fo=fopen(\"~w\",\"wb\");fwrite(Ch,4,(size_t)N*N,fo);fclose(fo); printf(\"gemm dumped %dx%d\\n\",N,N); return 0;}~n", [OutBin]),
    close(S),
    format("Generated CUDA GEMM (fma=~w, compile with ~w) -> ~w~n", [FmaMode, FmaMode, OutFile]).

%% ── emit a KERNEL-ONLY gemm .cu matching perf_fixture's gemm ABI ────────────
%% Signature: extern "C" __global__ void k_gemm(const float* A, const float* B,
%%            float* C, long n)  [n x n square]. fma_mode selects mul+add vs fmaf.
%% Compiled to cubin + loaded by perf_fixture (gemm class -> GFLOPS).
emit_gemm_kernel_only(FmaMode, OutFile) :-
    mac_form(cuda, FmaMode, "A[row*n+k]", "B[k*n+col]", "sum", Mac),
    open(OutFile, write, S),
    format(S, "/* GENERATED kernel-only GEMM (fma_mode=~w) for perf_fixture. */~n", [FmaMode]),
    format(S, "extern \"C\" __global__ void k_gemm(const float* A, const float* B, float* C, long n) {~n", []),
    format(S, "  long row=(long)blockIdx.y*blockDim.y+threadIdx.y;~n", []),
    format(S, "  long col=(long)blockIdx.x*blockDim.x+threadIdx.x;~n", []),
    format(S, "  if(row<n&&col<n){ float sum=0.0f; for(long k=0;k<n;k++) sum=~w; C[row*n+col]=sum; }~n", [Mac]),
    format(S, "}~n", []),
    close(S),
    format("Generated kernel-only GEMM (fma=~w) -> ~w~n", [FmaMode, OutFile]).

%% ── emit a self-contained GEMM dump runner for VERIFICATION ─────────────────
%% Reads A.bin,B.bin (n*n f32 each), computes C, writes OutBin. n from arg.
emit_gemm_verify(FmaMode, InA, InB, OutBin, OutFile) :-
    mac_form(cuda, FmaMode, "A[row*N+k]", "B[k*N+col]", "sum", Mac),
    open(OutFile, write, S),
    format(S, "/* GENERATED GEMM verify runner (fma_mode=~w). */~n", [FmaMode]),
    format(S, "#include <cstdio>~n#include <cstdlib>~n#include <cuda_runtime.h>~n", []),
    format(S, "__global__ void gemm(const float* A,const float* B,float* C,int N){~n", []),
    format(S, "  int row=blockIdx.y*blockDim.y+threadIdx.y,col=blockIdx.x*blockDim.x+threadIdx.x;~n", []),
    format(S, "  if(row<N&&col<N){ float sum=0.0f; for(int k=0;k<N;k++) sum=~w; C[row*N+col]=sum; }~n}~n", [Mac]),
    format(S, "int main(int c,char**v){ int N=atoi(v[1]); size_t b=(size_t)N*N*4;~n", []),
    format(S, "  float*Ah=(float*)malloc(b),*Bh=(float*)malloc(b),*Ch=(float*)malloc(b);~n", []),
    format(S, "  FILE*fa=fopen(\"~w\",\"rb\");fread(Ah,4,(size_t)N*N,fa);fclose(fa);~n", [InA]),
    format(S, "  FILE*fb=fopen(\"~w\",\"rb\");fread(Bh,4,(size_t)N*N,fb);fclose(fb);~n", [InB]),
    format(S, "  float*Ad,*Bd,*Cd;cudaMalloc(&Ad,b);cudaMalloc(&Bd,b);cudaMalloc(&Cd,b);~n", []),
    format(S, "  cudaMemcpy(Ad,Ah,b,cudaMemcpyHostToDevice);cudaMemcpy(Bd,Bh,b,cudaMemcpyHostToDevice);~n", []),
    format(S, "  dim3 bl(16,16),gr((N+15)/16,(N+15)/16);gemm<<<gr,bl>>>(Ad,Bd,Cd,N);cudaDeviceSynchronize();~n", []),
    format(S, "  cudaMemcpy(Ch,Cd,b,cudaMemcpyDeviceToHost);~n", []),
    format(S, "  FILE*fo=fopen(\"~w\",\"wb\");fwrite(Ch,4,(size_t)N*N,fo);fclose(fo);return 0;}~n", [OutBin]),
    close(S),
    format("Generated GEMM verify runner (fma=~w) -> ~w~n", [FmaMode, OutFile]).
