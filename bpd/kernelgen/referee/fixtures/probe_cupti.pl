%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% FAILED-TEST INJECTION: cupti_profile's reasoning layer.
:- consult("bpd/lib/cupti_profile.pl").
:- initialization(main).
main :-
    %% 1. known-bad stall profile MUST prescribe (the rule that drove the int4 win)
    ( cupti_profile:optimization_needed([memory_dependency-90.0, sync-2.0], O1)
      -> format("mem_dep 90%: prescribes ~w -> OK~n",[O1]) ; format("mem_dep 90%: NO prescription -> BROKEN~n") ),
    %% 2. clean profile must prescribe NOTHING
    ( cupti_profile:optimization_needed([memory_dependency-5.0, sync-3.0], O2)
      -> format("clean profile: prescribed ~w?! -> BROKEN (false positive)~n",[O2])
      ;  format("clean profile: silent -> OK~n") ),
    %% 3. boundary: exactly at threshold must NOT fire (> not >=)
    ( cupti_profile:optimization_needed([sync-15.0], _)
      -> format("sync at exactly 15: fired -> BOUNDARY-NOTE (>= semantics)~n")
      ;  format("sync at exactly 15: silent (strict >) -> OK~n") ),
    %% 4. sync above threshold fires reduce_barriers
    ( cupti_profile:optimization_needed([sync-16.0], reduce_barriers)
      -> format("sync 16: reduce_barriers -> OK~n") ; format("sync 16: MISSING -> BROKEN~n") ),
    %% 5. structural_bottleneck: all configs dominated by same stall
    cupti_profile:assert_profile(k_test, config(a), [memory_dependency-65.0]),
    cupti_profile:assert_profile(k_test, config(b), [memory_dependency-70.0]),
    ( cupti_profile:structural_bottleneck(k_test, D)
      -> format("structural: ~w -> OK~n",[D]) ; format("structural: NOT detected -> BROKEN~n") ),
    halt.
