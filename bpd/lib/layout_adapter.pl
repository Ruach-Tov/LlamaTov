%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% layout_adapter.pl — PARAMETERIZED memory-signature weave for drop-in kernels.
%% (Iyun, 2026-05-29, Heath's vision.) The q/k weight-layout permutation is ONE PARAMETER in a
%% sweepable space. Activation records (q/k/v buffers) are the OPERANDS of systolic arrays;
%% the arithmetic is fixed by the array, the FEEDING of operands (layout/weave) is 90% of the battle,
%% and the optimum is TARGET-DEPENDENT (uniform vs SSE3 local pools vs RISC-V vs systolic-NUMA).
%%
%% GOAL: the SAME kernel drops into ANY model on ANY target; the ADAPTER absorbs layout-matching.
%%   kernel(params) + layout_adapter(Kernel, ModelConv, Target) -> a composable WEAVE over tensors.
%%
%% Builds on: layout_algebra (transpose/permute/reshape/view/gather + inverse + solve_backward),
%%            hardware_facts (cache_line_size/2, hw_memory_bus_width/2, hw_shared_memory_per_sm/2).

:- module(layout_adapter,
    [ layout_adapter/4, feed/3, solve_required_layout/4,
      mem_topology/2, rope_convention_transform/2, sweep_axes/2 ]).
:- use_module(library(lists)).
:- use_module('layout_algebra').

%% ── TARGET MEMORY TOPOLOGY KIND (the axis hardware_facts lacks; grounded per target) ──
%% mem_topology(Target, Kind) — Kind in uniform | pooled(NPools) | numa(NNodes) | systolic(Rows,Cols).
mem_topology(sm_61, uniform).            % GPU global memory ~ uniform (coalesced)
mem_topology(sm_80, uniform).
mem_topology(sm_89, uniform).
mem_topology(sse3,  pooled(4)).          % SSE3-style: small local pools around each ALU lane
mem_topology(riscv_v, pooled(8)).        % RISC-V vector: vector-register-local pools
mem_topology(systolic_8x8, systolic(8,8)). % a systolic array: operands must match the PE grid

%% ── MODEL WEIGHT CONVENTION -> rope transform (the PROVEN q/k axis = instance #1) ──
%% rope_convention_transform(ModelConv, Transform). ggml stores q/k interleaved [re0,im0,re1,im1..];
%% hf split-half [re0,re1,..|im0,im1,..]. The transform aligning ggml->hf per head_dim d is
%% permute(evens ++ odds). Identity if conventions already match.
rope_convention_transform(ggml_interleave, identity).             % our runner's native (target=Ollama=ggml)
rope_convention_transform(hf_split_half,  rope_permute).          % needs the evens-then-odds permute
rope_convention_transform(neox,           identity).              % neox = ggml-like (no extra weave at weight level)
rope_convention_transform(none,           identity).

%% the actual permute index map for a given head_dim (evens then odds)
rope_permute_indices(HeadDim, Idx) :-
    findall(E, (between(0, HeadDim, E0), E0 < HeadDim, 0 =:= E0 mod 2, E = E0), Evens),
    findall(O, (between(0, HeadDim, O0), O0 < HeadDim, 1 =:= O0 mod 2, O = O0), Odds),
    append(Evens, Odds, Idx).

%% ── TILE / CACHE-LINE ALIGNMENT (from real hardware_facts) ──
%% tile_alignment_transform(Target, HeadDim, Transform). Pad/reshape the operand so a row aligns to
%% the cache line (coalesced reads). If head_dim*dtype already aligns, identity.
tile_alignment_transform(Target, RowBytes, Transform) :-
    ( catch(hardware_facts:cache_line_size(Target, CL), _, fail)
      -> ( 0 =:= RowBytes mod CL -> Transform = identity
         ; Pad is CL - (RowBytes mod CL), Transform = pad_to(CL, Pad) )
      ;  Transform = identity ).   % unknown target -> no alignment constraint asserted

%% ── POOL PARTITION (non-uniform memory: scatter operand into local pools) ──
%% pool_partition_transform(Topology, Transform). For pooled/numa/systolic, scatter the activation
%% record so each ALU/PE's operands live in ITS local pool. Uniform -> identity (no partition).
pool_partition_transform(uniform,        identity).
pool_partition_transform(pooled(N),      scatter_pools(N)).
pool_partition_transform(numa(N),        scatter_pools(N)).
pool_partition_transform(systolic(R,C),  tile_to_grid(R,C)).

%% ── THE ADAPTER: compose the weave for (Kernel, ModelConv, Target) ──
%% layout_adapter(Kernel, ModelConv, Target, Weave). Weave = ordered list of layout transforms.
layout_adapter(_Kernel, ModelConv, Target, Weave) :-
    rope_convention_transform(ModelConv, T1),
    ( mem_topology(Target, Topo) -> true ; Topo = uniform ),
    pool_partition_transform(Topo, T3),
    exclude(==(identity), [T1, T3], Weave).   % drop no-ops; T2 (alignment) added at feed-time w/ shape

%% feed(Weave, RawTensor, ReadyTensor) — apply the weave (forward) to lay weights out for the kernel.
%% Uses layout_algebra primitives where the transform maps to one (rope_permute -> permute, etc).
feed(Weave, Raw, Ready) :- foldl(apply_transform, Weave, Raw, Ready).
apply_transform(identity, T, T).
apply_transform(rope_permute, t(Ne,Nb,O), Out) :-
    % per-head permute on the head_dim dimension (here modeled as a permute relation; concrete index
    % map = rope_permute_indices(HeadDim). The algebra carries the structure; the kernel-gen applies it.
    Out = t(Ne, Nb, O).                       % structure-preserving placeholder; concrete perm at codegen
apply_transform(scatter_pools(_N), T, T).     % scatter = gather-class (layout_algebra gather); structure carried
apply_transform(tile_to_grid(_R,_C), T, T).

%% solve_required_layout(Weave, DesiredKernelLayout, RequiredWeightLayout, _) — Heath's weave-backward:
%% given the layout the kernel WANTS, derive the layout the weights must be in. Uses solve_backward
%% over the layout_algebra transforms the weave maps to.
solve_required_layout(Weave, Desired, Required, AlgChain) :-
    maplist(weave_to_alg, Weave, AlgChain0), exclude(==(identity), AlgChain0, AlgChain),
    ( AlgChain == [] -> Required = Desired
    ; layout_algebra:solve_backward(AlgChain, Desired, Required) ).
weave_to_alg(rope_permute, permute([0,2,4,6,1,3,5,7])).   % example head_dim=8; generalized by rope_permute_indices
weave_to_alg(scatter_pools(N), gather(N)).
weave_to_alg(tile_to_grid(R,_C), reshape([R])).
weave_to_alg(identity, identity).

%% ── THE SWEEP: enumerate the parameter axes to optimize over (per model x target) ──
%% sweep_axes(Target, Axes). Each axis is a name + candidate values; the sweep is the cross product,
%% scored stall-vs-flow (CUPTI-style). This is the "90% of the battle" search space, DECLARED.
sweep_axes(Target, Axes) :-
    ( mem_topology(Target, Topo) -> true ; Topo = uniform ),
    Axes = [ axis(rope_layout, [ggml_interleave, hf_split_half]),
             axis(tile_align,   [none, cache_line]),
             axis(dataflow,     [row_major, col_major]),
             axis(pool_partition, Topo) ].
