%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_ffn_block.pl — FFN SwiGLU block: BPD facts + round-trip + fusion analysis
%%
%% Subtask S3 of the transformer layer decomposition.
%% Tests: (1) parse real qwen2 FFN C, (2) extract BPD facts,
%%        (3) run fusion analysis, (4) emit CUDA kernels.

:- set_prolog_flag(double_quotes, codes).
:- use_module('../lib/c_ast').
:- use_module('../lib/fusion_analyzer').

%% ═══════════════════════════════════════════════════════════════
%% S3a: Parse the FFN section from qwen2.cpp
%% ═══════════════════════════════════════════════════════════════

test_parse_ffn :-
    write('=== S3a: Parse FFN from qwen2.cpp ==='), nl,
    atom_codes('cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il); cb(cur, "ffn_norm", il);', InputCodes),
    atom_codes(InputAtom, InputCodes),
    c_ast:c_parse_stmts_v3(InputAtom, NormAST),
    length(NormAST, NormCount),
    format("  Parsed FFN norm: ~d statements~n", [NormCount]),
    
    %% Round-trip the norm
    phrase(c_ast:emit_stmts(NormAST, 2), NormCodes),
    atom_codes(NormS, NormCodes),
    write(NormS), nl.

%% ═══════════════════════════════════════════════════════════════
%% S3b: BPD facts for FFN SwiGLU block
%% ═══════════════════════════════════════════════════════════════

%% The expanded FFN SwiGLU as primitive operations:
%%   1. ffn_norm: RMSNorm(ffn_inp)
%%   2. gate_proj: matmul(ffn_gate, cur)  
%%   3. silu_act: SiLU(gate_output)
%%   4. up_proj: matmul(ffn_up, cur)
%%   5. gate_mul: gate_output * up_output (elementwise)
%%   6. down_proj: matmul(ffn_down, gate_mul_output)
%%   7. residual: ffn_inp + down_output

ffn_ops([
    op(ffn_norm, ggml_rms_norm, 1),
    op(gate_proj, ggml_mul_mat, 2),
    op(silu_act, ggml_silu, 3),
    op(up_proj, ggml_mul_mat, 4),
    op(gate_mul, ggml_mul, 5),
    op(down_proj, ggml_mul_mat, 6),
    op(residual_add, ggml_add, 7)
]).

test_fusion_ffn :-
    write('=== S3b: FFN Fusion Analysis ==='), nl,
    ffn_ops(Ops),
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
    nl.

%% ═══════════════════════════════════════════════════════════════
%% S3c: Emit CUDA kernels for FFN fusions
%% ═══════════════════════════════════════════════════════════════

test_emit_ffn_cuda :-
    write('=== S3c: FFN CUDA Kernel Emission ==='), nl,
    ffn_ops(Ops),
    compile_graph(Ops, p4, Program),
    emit_program(Program, Code),
    write(Code), nl.

%% Need compile_graph from fusion_to_cuda — inline simplified version
compile_graph(Ops, HW, Program) :-
    find_fusible_chains(Ops, Chains),
    include([C]>>(length(C, L), L > 1), Chains, FusionChains),
    generate_all_kernels(FusionChains, HW, Kernels),
    Program = [
        c_include_sys('stdio.h'),
        c_include_sys('math.h'),
        c_include_sys('cuda_runtime.h'),
        c_blank
        | Kernels
    ].

generate_all_kernels([], _, []).
generate_all_kernels([Chain|Rest], HW, [Kernel, c_blank | More]) :-
    reverse(Chain, Fwd),
    generate_kernel(Fwd, HW, Kernel),
    generate_all_kernels(Rest, HW, More).

%% Matmul + elementwise epilogue
generate_kernel(Chain, _HW, Kernel) :-
    Chain = [op(AnchorName, AnchorKind, _) | EpilogueOps],
    classify_op(AnchorKind, matmul),
    !,
    findall(N, member(op(N,_,_), Chain), Names),
    atomic_list_concat(Names, '_', Base),
    atom_concat('fused_', Base, KName),
    build_epilogue(EpilogueOps, EStmts),
    Kernel = c_func(['__global__'], c_type(void), KName,
        [param(c_type(const_ptr(c_type(float))), 'A'),
         param(c_type(const_ptr(c_type(float))), 'B'),
         param(c_type(ptr(c_type(float))), 'C'),
         param(c_type(int), 'M'),
         param(c_type(int), 'N'),
         param(c_type(int), 'K')],
        [c_comment('Tiled matmul + fused epilogue'),
         c_comment('(tile loading elided — see gen_fused_kernel.pl)'),
         c_decl_init(c_type(float), sum, c_float(0.0)),
         c_blank,
         c_comment('--- FUSED EPILOGUE ---')
         | EStmts]).

%% Elementwise-only chain
generate_kernel(Chain, _HW, Kernel) :-
    findall(N, member(op(N,_,_), Chain), Names),
    atomic_list_concat(Names, '_', Base),
    atom_concat('fused_ew_', Base, KName),
    build_epilogue(Chain, EStmts),
    Kernel = c_func(['__global__'], c_type(void), KName,
        [param(c_type(ptr(c_type(float))), data),
         param(c_type(int), n)],
        [c_decl_init(c_type(float), sum, c_float(0.0)),
         c_blank,
         c_comment('--- FUSED ELEMENTWISE ---')
         | EStmts]).

build_epilogue([], []).
build_epilogue([op(_, ggml_add, _)|R], [S|M]) :-
    S = c_assign(c_var(sum), c_binop('+', c_var(sum), c_index(c_var(bias), c_var(col)))),
    build_epilogue(R, M).
build_epilogue([op(_, ggml_silu, _)|R], [S|M]) :-
    S = c_assign(c_var(sum), c_binop('/', c_var(sum),
        c_paren(c_binop('+', c_float(1.0), c_call(expf, [c_unop('-', c_var(sum))]))))),
    build_epilogue(R, M).
build_epilogue([op(_, ggml_mul, _)|R], [S|M]) :-
    S = c_assign(c_var(sum), c_binop('*', c_var(sum), c_index(c_var(gate), c_var(col)))),
    build_epilogue(R, M).
build_epilogue([op(_, Kind, _)|R], M) :-
    classify_op(Kind, layout),
    build_epilogue(R, M).

%% ═══════════════════════════════════════════════════════════════
%% RUN ALL
%% ═══════════════════════════════════════════════════════════════

run_all :-
    test_parse_ffn, nl,
    test_fusion_ffn,
    test_emit_ffn_cuda.

:- initialization((run_all -> halt(0) ; (write('FAILED'), nl, halt(1)))).
