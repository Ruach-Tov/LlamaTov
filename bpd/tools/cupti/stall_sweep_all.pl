%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
:- use_foreign_library('/tmp/cupti_q8.so', install_cupti_q8).

% MULTI-ANGLE post-fusion stall sweep. Compare signal quality vs the pre-fusion sweep (e6b75fab).
shapes([
    shape(attn,     896,    896),
    shape(ffn_down, 896,    4864),
    shape(ffn_up,   4864,   896),
    shape(vocab,    151936, 896)
]).

measure(shape(Name, M, K)) :-
    format("~n=== ~w (M=~w, K=~w) ===~n", [Name, M, K]),
    cupti_init,
    run_q8_gemv_tiled('/tmp/v4kernel.cubin', M, K, 16, 3000),
    cupti_flush,
    ( cupti_total_samples(N) -> format("  samples: ~w~n", [N]) ; format("  (no sample count)~n", []) ),
    ( cupti_stall_report(Stalls)
      -> format("  STALLS:~n", []), forall(member(S, Stalls), format("    ~w~n", [S]))
      ;  format("  (no stall report)~n", []) ),
    ( cupti_suggest(Sugg)
      -> format("  SUGGEST: ~w~n", [Sugg])
      ;  true ),
    cupti_reset.

main :-
    shapes(Shapes),
    forall(member(Sh, Shapes), measure(Sh)),
    halt.

:- initialization(main).
