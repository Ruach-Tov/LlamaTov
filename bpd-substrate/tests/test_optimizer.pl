%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════
%% Licensed under RTAAL-1.0 (Ruach Tov AI Agent License)
%% See: LICENSE-RTAAL-1-0.md in the project root
%% ═══════════════════════════════════════════════════════════════════════

%% test_optimizer.pl — Tests for the Prolog kernel fusion optimizer

:- use_module(library(plunit)).
:- use_module('../lib/graph_optimizer').
:- use_module('../lib/auto_fuser').
:- use_module('../lib/region_inference').

:- begin_tests(graph_optimizer).

test(yolo_cbs_fusion) :-
    %% YOLO L0 CBS: conv + BN + SiLU
    %% We mock the facts.
    Graph = [
        op(conv), op_kind(conv, conv2d), op_inputs(conv, [x, w]), op_output(conv, out_conv),
        sequence(b1, conv, 1),
        op(bn), op_kind(bn, bias_add), op_inputs(bn, [out_conv, bias]), op_output(bn, out_bn),
        sequence(b1, bn, 2),
        op(silu), op_kind(silu, ggml_silu), op_inputs(silu, [out_bn]), op_output(silu, out_silu),
        sequence(b1, silu, 3)
    ],
    %% Note: bias_add is elementwise. conv2d is spatial.
    fuse_graph(Graph, cuBLAS, Fused), !,
    %% We expect conv to be fused with bn, then with silu.
    %% The final graph should have one op.
    findall(O, member(op(O), Fused), Ops),
    length(Ops, 1).

test(llama_transformer_block) :-
    %% Llama block: matmul + rmsnorm epilogue + swiglu
    %% Mocking facts
    Graph2 = [
        op(mm), op_kind(mm, matmul), op_inputs(mm, [w, x]), op_output(mm, mm_out),
        sequence(b1, mm, 1),
        op(add), op_kind(add, ggml_add), op_inputs(add, [mm_out, bias]), op_output(add, add_out),
        sequence(b1, add, 2),
        op(silu), op_kind(silu, ggml_silu), op_inputs(silu, [add_out]), op_output(silu, silu_out),
        sequence(b1, silu, 3)
    ],
    fuse_graph(Graph2, cuBLAS, Fused), !,
    findall(O, member(op(O), Fused), Ops),
    length(Ops, 1).

:- end_tests(graph_optimizer).

:- initialization(run_tests, main).




