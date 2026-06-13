%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
:- use_module('lib/numerical_stability.pl').
:- initialization(main).
main :-
    style_check(-discontiguous),
    %% reduction-sensitivity: a reduce over >64 elements should warn (now that the clause parses).
    assertz(numerical_stability:op(r1, reduce_sum, [in], [out])),
    assertz(numerical_stability:reduce_dim_size(r1, 896)),
    ( numerical_stability:check_reduction_sensitivity(r1, Res),
      ( Res = warning(reduction_sensitive, _, _) -> format("DETECTOR-LIVE: reduction_sensitive FIRES on 896-elem reduce (was a dead clause before the fix)~n", []) ; format("DETECTOR: reduction returned ~w~n", [Res]) )
    ; format("DETECTOR-FAIL: check_reduction_sensitivity did not run~n", []) ),
    halt.
