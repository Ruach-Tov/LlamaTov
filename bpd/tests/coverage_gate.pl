%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% coverage_gate.pl — Stage 2a: kernel coverage gate for the mistral/llama graph.
%% (Iyun 2026-05-29.) mistral uses llm_build_llama in this llama.cpp (no distinct builder),
%% so mistral's graph == the llama graph we round-trip 100%.
%% Gate: does ggml_dispatch_table have a kernel for every ggml op in the graph?
%% ggml_has_op/1 accepts the SHORT op name (add, mul, ...), not the ggml_ prefix.
%% Invoke: swipl -q -g "use_module(lib/llama_cpp_lifter),use_module(lib/ggml_dispatch_table),consult(tests/coverage_gate),run,halt"

mistral_graph_ops(Ks) :-
    Src = "/tmp/llama_cpp_test/src/llama-model.cpp",
    read_file_to_string(Src, Text, []), string_length(Text, L),
    llama_cpp_lifter:builder_body(Text, L, llama, Body),
    ( catch(llama_cpp_lifter:lift_op_sequence(Body, Ops), _, fail) -> true ; Ops = [] ),
    findall(K, member(op(ggml_op, K), Ops), Ks0), sort(Ks0, Ks).

covered(K)  :- catch(once(ggml_dispatch_table:ggml_has_op(K)), _, fail).

run :-
    mistral_graph_ops(Ks), length(Ks, NG),
    format("=== Stage 2a: kernel coverage gate (mistral == llama graph) ===~n",[]),
    format("distinct ggml ops: ~w  ~w~n",[NG, Ks]),
    partition(covered, Ks, Cov, Miss),
    length(Cov, NC), length(Miss, NM),
    format("COVERED (~w): ~w~n",[NC, Cov]),
    format("MISSING (~w): ~w~n",[NM, Miss]),
    ( NM =:= 0
      -> format("** GATE PASS: Stage 3 dispatch REACHABLE for mistral **~n",[])
      ;  format("** GATE: ~w ops need kernels (note: some MISSING may be view/graph ops not~n",[NM]),
         format("   needing arithmetic kernels: build_forward_expand=graph-build, get_rows/reshape=view.~n",[]),
         format("   Genuine compute kernels likely needed: rms_norm, rope_ext, scale.) **~n",[]) ).
