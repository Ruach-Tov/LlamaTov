%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% FAILED-TEST INJECTION for the fusion scanner: feed it graphs where fusions
%% MUST NOT fire, confirm it stays silent; and graphs where they must, confirm it fires.
:- use_module('lib/fusion_rules').
:- dynamic op_kind/2, op_input/2, op_output/2, op_reads/3, op_writes/3.
fusion_rules:op_kind(O,K) :- op_kind(O,K).
fusion_rules:op_input(O,T) :- op_input(O,T).
fusion_rules:op_output(O,T) :- op_output(O,T).
fusion_rules:op_reads(O,T,R) :- op_reads(O,T,R).
fusion_rules:op_writes(O,T,R) :- op_writes(O,T,R).
:- initialization(main).

scan(N) :-
    Rules = [epilogue_matmul_elementwise, elementwise_chain, layout_transparent, quant_into_gemv],
    enumerate_valid_fusions(Rules, Fs), sort(Fs, U), length(U, N).

main :-
    %% INJECTION 1: multi-consumer quant (like attn's xqa->q,k,v) — quant_into_gemv MUST NOT fire
    assertz(op_kind(mq, k_quant_q8)), assertz(op_output(mq, xq)),
    assertz(op_writes(mq, xq, region(quantized_activation, e896))),
    assertz(op_kind(g1, k_q8_0_gemv)), assertz(op_input(g1, xq)),
    assertz(op_reads(g1, xq, region(quantized_activation,e896))), assertz(op_output(g1, o1)),
    assertz(op_kind(g2, k_q8_0_gemv)), assertz(op_input(g2, xq)),
    assertz(op_reads(g2, xq, region(quantized_activation,e896))), assertz(op_output(g2, o2)),
    scan(N1),
    ( N1 =:= 0 -> format("multi-consumer quant: no fusion fired -> SCANNER-OK~n")
    ; format("multi-consumer quant: ~w fusions fired -> SCANNER-BROKEN (must respect no_other_consumers)~n",[N1]) ),
    %% INJECTION 2: now retract g2 (single consumer) — quant_into_gemv MUST fire exactly once
    retract(op_kind(g2, k_q8_0_gemv)), retract(op_input(g2, xq)),
    retract(op_reads(g2, xq, region(quantized_activation,e896))), retract(op_output(g2, o2)),
    scan(N2),
    ( N2 =:= 1 -> format("single-consumer quant: exactly 1 fusion -> SCANNER-OK~n")
    ; format("single-consumer quant: ~w fusions -> SCANNER-BROKEN~n",[N2]) ),
    %% INJECTION 3: region mismatch — gemv reading an UNQUANTIZED region must NOT fuse
    retractall(op_reads(g1, xq, _)),
    assertz(op_reads(g1, xq, region(elementwise, e896))),
    scan(N3),
    ( N3 =:= 0 -> format("region-mismatch: no fusion -> SCANNER-OK~n")
    ; format("region-mismatch: ~w fired -> SCANNER-BROKEN (region terms not enforced)~n",[N3]) ),
    halt.
