%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% bpd_profile.pl — profile-guided tuning from Prolog. Makes mavchin's cpu_profile.pl
%% (perf_event_open hardware counters) a CALLABLE ORACLE over our own generated kernels.
%%
%% Per Heath: the PMU-from-Prolog query is a tool we just gained — adapt it to measure and
%% tune OUR code the "old-fashioned" way, but driven from the logic layer. This closes the
%% codegen loop: lift -> emit -> MEASURE (real PMU) -> verify (referee) -> optimize.
%%
%% profile_qdot_kernel/3 generates a tiny C harness that calls a row-sequential q8_0 dot
%% kernel from our .so across NROWS weight rows (the decode access pattern), links it with
%% the profiler helper, runs it, and parses cycles/IPC/cyc-per-block/L1-miss/branch-miss
%% into a Prolog dict. The result is a queryable FACT about the kernel's PMU behavior.

:- module(bpd_profile, [
    profile_qdot_kernel/3,     % profile_qdot_kernel(+SoPath, +KernelName, -Metrics)
    profiler_helper/1          % profiler_helper(-Path) — ensure built
]).

:- use_module(library(process)).
:- use_module(library(readutil)).

profiler_helper('/tmp/bpd_cpu_profiler.so').

%% NROWS/K for the decode access pattern (134MB cold, K=2048).
harness_src(Kernel, Src) :-
    format(atom(Src),
'#include <stdio.h>\n#include <stdlib.h>\n#include <stdint.h>\n\c
extern int bpd_cpu_profile(void(*)(void),int,long long*,long long*,long long*,long long*,long long*,long long*);\n\c
extern float ~w(const uint8_t* w, const uint8_t* a, int n_blocks);\n\c
#define K 2048\n#define NB (K/32)\n#define BYTES (NB*34)\n#define NROWS 65536\n\c
static uint8_t* W; static uint8_t* A; static volatile float sink;\n\c
static void run(void){ float acc=0; for(int r=0;r<NROWS;r++) acc+=~w(W+(size_t)r*BYTES,A,NB); sink=acc; }\n\c
int main(){ srand(1); W=malloc((size_t)NROWS*BYTES); A=malloc(BYTES);\n\c
  for(size_t i=0;i<(size_t)NROWS*BYTES;i++)W[i]=rand()&0xFF; for(size_t i=0;i<BYTES;i++)A[i]=rand()&0xFF;\n\c
  for(int r=0;r<NROWS;r++){W[(size_t)r*BYTES]=0;W[(size_t)r*BYTES+1]=0x3C;} for(int b=0;b<NB;b++){A[b*34]=0;A[b*34+1]=0x3C;}\n\c
  long long c=0,i=0,l1=0,llc=0,br=0,clk=0; bpd_cpu_profile(run,3,&c,&i,&l1,&llc,&br,&clk);\n\c
  double dots=(double)NROWS*3, by=(double)NROWS*BYTES, sec=(double)clk/3/1e9;\n\c
  printf("RESULT gbps=%.3f ipc=%.3f cyc_per_block=%.3f l1m_per_dot=%.3f brm_per_dot=%.3f\\n",\n\c
    by/1e9/sec, (double)i/c, (double)c/(dots*NB), (double)l1/dots, (double)br/dots);\n\c
  return 0; }\n', [Kernel, Kernel]).

profile_qdot_kernel(SoPath, Kernel, Metrics) :-
    harness_src(Kernel, Src),
    SrcFile = '/tmp/bpd_prof_gen.c',
    OutBin = '/tmp/bpd_prof_gen',
    open(SrcFile, write, S), write(S, Src), close(S),
    %% compile: harness + profiler helper source + the .so under test
    process_create(path(clang),
        ['-mavx','-mf16c','-mssse3','-mno-avx2','-mno-fma','-O2',
         SrcFile, '/tmp/bpd_cpu_profiler.c', SoPath, '-lm', '-o', OutBin],
        [stdout(null), stderr(null), process(P0)]), process_wait(P0, _),
    %% run with LD_LIBRARY_PATH for the .so deps
    getenv_or('LD_LIBRARY_PATH', '', _),
    process_create(OutBin, [],
        [stdout(pipe(Out)), stderr(null), environment(['LD_LIBRARY_PATH'='/tmp/iyun_build'])]),
    read_string(Out, _, Result), close(Out),
    parse_result(Result, Metrics).

getenv_or(Var, Default, Val) :- ( getenv(Var, Val) -> true ; Val = Default ).

parse_result(Str, metrics{gbps:G, ipc:I, cyc_per_block:C, l1m_per_dot:L, brm_per_dot:B}) :-
    ( re_matchsub("gbps=([0-9.]+) ipc=([0-9.]+) cyc_per_block=([0-9.]+) l1m_per_dot=([0-9.]+) brm_per_dot=([0-9.]+)",
        Str, Sub, []) ->
        number_string(G, Sub.1), number_string(I, Sub.2), number_string(C, Sub.3),
        number_string(L, Sub.4), number_string(B, Sub.5)
    ; G = -1, I = -1, C = -1, L = -1, B = -1 ).
