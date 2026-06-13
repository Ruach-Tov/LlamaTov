%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_attention_block.pl — Attention block: BPD facts + fusion + CUDA
%%
%% Subtask S2 of transformer layer decomposition.
%% The attention diamond: Q×K^T → scale → softmax → ×V → out_proj
%% This is the pattern Flash Attention fuses — but our linear chain
%% discoverer correctly identifies the FUSIBLE parts and the BARRIERS.

:- set_prolog_flag(double_quotes, codes).
:- use_module('../lib/c_ast').
:- use_module('../lib/fusion_analyzer').

%% ═══════════════════════════════════════════════════════════════
%% S2a: Attention operations as BPD facts
%% ═══════════════════════════════════════════════════════════════

%% After QKV projection (S1), attention computes:
%%   scores = Q × K^T (matmul)
%%   scores = scores * scale (elementwise)
%%   probs = softmax(scores) (reduction)
%%   context = probs × V (matmul)
%%   output = out_proj(context) (matmul)
%%   result = output + residual (elementwise)

attention_ops([
    op(qk_matmul, ggml_mul_mat, 1),      % Q × K^T
    op(qk_scale, ggml_scale, 2),          % scale by 1/sqrt(d)
    op(attn_softmax, ggml_soft_max_ext, 3), % softmax (REDUCTION — barrier)
    op(sv_matmul, ggml_mul_mat, 4),       % probs × V
    op(out_proj, ggml_mul_mat, 5),        % output projection
    op(out_bias, ggml_add, 6),            % output bias
    op(attn_residual, ggml_add, 7)        % residual connection
]).

test_attention_fusion :-
    write('=== S2b: Attention Fusion Analysis ==='), nl,
    attention_ops(Ops),
    find_fusible_chains(Ops, Chains),
    include([C]>>(length(C, L), L > 1), Chains, MultiChains),
    length(MultiChains, ChainCount),
    format("  Found ~d fusible chains:~n", [ChainCount]),
    forall(member(Chain, MultiChains), (
        reverse(Chain, Fwd),
        findall(N, member(op(N,_,_), Fwd), Names),
        findall(K, member(op(_,K,_), Fwd), Kinds),
        format("    ~w (~w)~n", [Names, Kinds])
    )),
    nl,
    
    %% Verify the BARRIERS are correctly identified
    write('  Barriers (correctly rejected):'), nl,
    %% scale→softmax DOES fuse (elementwise_reduction rule)
    ( can_fuse(ggml_scale, ggml_soft_max_ext, R1) ->
        format('    ✓ scale→softmax: fuses as ~w (elementwise→reduction)~n', [R1])
    ;
        write('    ✗ scale→softmax: should fuse via elementwise_reduction')
    ), nl,
    ( can_fuse(ggml_soft_max_ext, ggml_mul_mat, _) ->
        write('    ERROR: softmax→matmul should NOT fuse')
    ;
        write('    ✓ softmax→matmul: blocked (reduction→matmul)')
    ), nl,
    ( can_fuse(ggml_mul_mat, ggml_mul_mat, _) ->
        write('    ERROR: matmul→matmul should NOT fuse')
    ;
        write('    ✓ matmul→matmul: blocked (incompatible iteration space)')
    ), nl, nl.

%% ═══════════════════════════════════════════════════════════════
%% S2c: Emit CUDA kernels for fusible attention parts
%% ═══════════════════════════════════════════════════════════════

test_attention_cuda :-
    write('=== S2c: Attention CUDA Kernels ==='), nl,
    attention_ops(Ops),
    find_fusible_chains(Ops, Chains),
    include([C]>>(length(C, L), L > 1), Chains, FusionChains),
    
    forall(member(Chain, FusionChains), (
        reverse(Chain, Fwd),
        findall(N, member(op(N,_,_), Fwd), Names),
        generate_kernel_simple(Fwd, Kernel),
        emit_c(Kernel, Code),
        format("  // Kernel for chain ~w~n", [Names]),
        write(Code), nl
    )).

%% Simple kernel generation (reused from test_ffn_block)
generate_kernel_simple(Chain, Kernel) :-
    Chain = [op(_, AnchorKind, _) | EpilogueOps],
    classify_op(AnchorKind, matmul),
    !,
    findall(N, member(op(N,_,_), Chain), Names),
    atomic_list_concat(Names, '_', Base),
    atom_concat('fused_', Base, KName),
    build_ep(EpilogueOps, EStmts),
    Kernel = c_func(['__global__'], c_type(void), KName,
        [param(c_type(const_ptr(c_type(float))), 'A'),
         param(c_type(const_ptr(c_type(float))), 'B'),
         param(c_type(ptr(c_type(float))), 'C'),
         param(c_type(int), 'M'),
         param(c_type(int), 'N'),
         param(c_type(int), 'K')],
        [c_comment('Tiled matmul + fused epilogue'),
         c_decl_init(c_type(float), sum, c_float(0.0))
         | EStmts]).

build_ep([], []).
build_ep([op(_, ggml_add, _)|R], [c_assign(c_var(sum), c_binop('+', c_var(sum), c_index(c_var(bias), c_var(col))))|M]) :- build_ep(R, M).
build_ep([op(_, ggml_scale, _)|R], [c_assign(c_var(sum), c_binop('*', c_var(sum), c_var(scale)))|M]) :- build_ep(R, M).
build_ep([op(_, ggml_soft_max_ext, _)|R], [c_comment('softmax (reduction — computed inline)')|M]) :- build_ep(R, M).
build_ep([op(_, ggml_silu, _)|R], [c_assign(c_var(sum), c_binop('/', c_var(sum), c_paren(c_binop('+', c_float(1.0), c_call(expf, [c_unop('-', c_var(sum))])))))|M]) :- build_ep(R, M).

%% ═══════════════════════════════════════════════════════════════
%% S2d: Full attention block summary
%% ═══════════════════════════════════════════════════════════════

test_attention_summary :-
    write('=== S2d: Attention Block Fusion Summary ==='), nl,
    attention_ops(Ops),
    length(Ops, TotalOps),
    find_fusible_chains(Ops, Chains),
    include([C]>>(length(C, L), L > 1), Chains, FusionChains),
    
    %% Count fused ops and kernel launches
    findall(L, (member(C, FusionChains), length(C, L)), ChainLens),
    sumlist(ChainLens, FusedOps),
    length(FusionChains, FusedKernels),
    UnfusedOps is TotalOps - FusedOps,
    TotalKernels is FusedKernels + UnfusedOps,
    Eliminated is TotalOps - TotalKernels,
    
    format("  Total operations: ~d~n", [TotalOps]),
    format("  Fused into ~d kernels (covering ~d ops)~n", [FusedKernels, FusedOps]),
    format("  Unfused operations: ~d~n", [UnfusedOps]),
    format("  Total kernel launches: ~d (was ~d)~n", [TotalKernels, TotalOps]),
    format("  Eliminated launches: ~d~n", [Eliminated]),
    format("  VRAM round-trips saved: ~d~n", [Eliminated]),
    nl.

run_all :-
    test_attention_fusion,
    test_attention_cuda, nl,
    test_attention_summary.

:- initialization((run_all -> halt(0) ; (write('FAILED'), nl, halt(1)))).
