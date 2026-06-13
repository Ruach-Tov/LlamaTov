%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
:- use_foreign_library('/tmp/cupti_q8.so', install_cupti_q8).
:- use_module('lib/cupti_profile.pl').
:- initialization(main).

% pull the dominant stall reasons from a stall report into the StallList shape
% optimization_needed/2 expects: [stall(Type, Pct), ...]
report_to_stalllist(SL, Out) :-
    findall(stall(T, P),
        ( member(M, SL),
          ( M = stall(T, P) -> true
          ; M = suggest(_, T, P) -> true
          ; M = T-P -> true
          ; fail )
        ), Out0),
    ( Out0 = [] -> Out = SL ; Out = Out0 ).

prescribe(SL) :-
    ( report_to_stalllist(SL, Stalls),
      findall(Opt, (member(stall(Ty,Pc), Stalls), Pc >= 1,
                    ( optimization_needed([stall(Ty,Pc)], Opt) -> true ; Opt = none )), Opts0),
      sort(Opts0, Opts),
      format("  PRESCRIBED OPTIMIZATIONS: ~w~n", [Opts])
    -> true ; format("  (could not map stalls to optimization_needed/2; raw report above)~n", []) ).

profile(Tag, Cubin, M, K, BM) :-
    format("~n=== ~w  (M=~w K=~w BM=~w) ===~n", [Tag, M, K, BM]),
    cupti_init,
    ( BM =:= 0
    -> ( catch(run_q8_gemv(Cubin, M, K, 300), E, (format("  launch err: ~w~n",[E]), fail)) -> true ; true )
    ;  ( catch(run_q8_gemv_tiled(Cubin, M, K, BM, 300), E, (format("  launch err: ~w~n",[E]), fail)) -> true ; true ) ),
    cupti_flush,
    ( cupti_total_samples(N) -> format("  PC samples: ~w~n", [N]) ; true ),
    ( cupti_stall_report(SL) -> format("  STALL REPORT: ~w~n", [SL]), prescribe(SL) ; format("  no stall report~n",[]) ).

main :-
    SerCubin = '/tmp/fact_cubins/q8_serial_prof_5c3d5c5c.cubin',
    TilCubin = '/tmp/fact_cubins/q8_tiled16_prof_bfee4566.cubin',
    profile('SERIAL ffn_down', SerCubin, 896, 4864, 0),
    profile('TILED  ffn_down', TilCubin, 896, 4864, 16),
    profile('TILED  vocab',    TilCubin, 151936, 896, 16),
    halt.
