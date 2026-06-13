%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
:- use_foreign_library('/tmp/cupti_q8.so', install_cupti_q8).
:- use_module(library(process)).
:- initialization(main).

%% stall_v4_compare.pl — re-measure the tiled (int32) vs v4 (int4) GEMV stall profiles, to confirm
%% the int4 weight loads collapse the texture stall (the cupti-from-prolog prescription, verified).
%% SELF-CONTAINED: generates its own cubins via the emitter (no hardcoded /tmp cubin hashes — the
%% earlier version baked session-specific hashes that don't exist on a fresh checkout; this fixes it).
%% Requires cupti_q8.so built (bpd/tools/cupti/build instructions in q8_gemv_launcher.c header).

gen_cubin(Mode, Cubin) :-
    %% emit + build via the Python fact_dispatch path (standalone nvcc is broken on the enclave).
    format(atom(Py),
      "import sys; sys.path.insert(0,\"bpd\"); sys.path.insert(0,\"bpd/lib\"); import fact_dispatch as fd; \c
       print(fd._emit_and_build([\"FACTS\", f\"{fd._EMIT}/q8_0_from_facts.pl\"], \c
       f'q8_0_op_expr(E), emit_from_fact(E, [mode(~w)], \"{fd._CACHE}/stallcmp.cu\")', \"stallcmp_~w\"))",
      [Mode, Mode]),
    process_create(path(python3), ['-c', Py], [stdout(pipe(Out)), process(P)]),
    read_string(Out, _, S0), process_wait(P, _),
    split_string(S0, "\n", "\n ", Lines),
    ( member(L, Lines), string_concat(_, ".cubin", L) -> atom_string(Cubin, L) ; Cubin = '' ).

profile(Tag, Cubin, M, K, BM) :-
    format("~n=== ~w (M=~w K=~w BM=~w) ===~n", [Tag, M, K, BM]),
    cupti_init,
    ( catch(run_q8_gemv_tiled(Cubin, M, K, BM, 300), E, (format("  err ~w~n",[E]), fail)) -> true ; true ),
    cupti_flush,
    ( cupti_total_samples(N) -> format("  samples: ~w~n", [N]) ; true ),
    ( cupti_stall_report(SL) -> format("  STALLS: ~w~n", [SL]) ; format("  no report~n",[]) ).

main :-
    gen_cubin('tiled(16,256,1)', Til),
    gen_cubin('tiled_v4(16,256)', V4),
    ( Til \== '', V4 \== ''
    -> profile('TILED  ffn_down (int32 loads)', Til, 896, 4864, 16),
       profile('V4     ffn_down (int4 loads)',  V4,  896, 4864, 16),
       profile('TILED  vocab    (int32 loads)', Til, 151936, 896, 16),
       profile('V4     vocab    (int4 loads)',  V4,  151936, 896, 16)
    ;  format("cubin generation failed (Til=~w V4=~w)~n", [Til, V4]) ),
    halt.
