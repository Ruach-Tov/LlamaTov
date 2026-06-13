%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
:- consult("bpd/lib/auto_fuser.pl").
:- initialization(main).
chk(Label, Goal, Expect) :-
    ( catch(Goal, E, (format("~w: EXCEPTION ~w -> GATE-BROKEN~n", [Label, E]), fail)) ->
        ( Expect == pass -> format("~w: OK~n", [Label]) ; format("~w: ACCEPTED-BUT-SHOULD-FAIL -> GATE-BROKEN~n", [Label]) )
    ;   ( Expect == fail -> format("~w: correctly rejected -> OK~n", [Label]) ; format("~w: FAILED-BUT-SHOULD-PASS -> BROKEN~n", [Label]) ) ).
main :-
    % 1. known chain plans correctly?
    chk("plan gemm+bias+relu", (auto_fuser:fusion_plan([gemm,bias_add,relu], P1), P1 = [kernel(spatial_with_epilogue,[gemm,bias_add,relu],3)]), pass),
    % 2. reduction breaks the epilogue chain (rmsnorm cannot be absorbed into gemm epilogue)
    chk("rmsnorm starts new kernel", (auto_fuser:fusion_plan([gemm,rmsnorm,silu], P2), P2 = [kernel(spatial_with_epilogue,[gemm],3), kernel(reduction_with_epilogue,[rmsnorm,silu],1)]), pass),
    % 3. UNKNOWN op: what happens? (the dead-detector probe — does it fail loud or silently misplan?)
    chk("unknown op LOUD failure", auto_fuser:fusion_plan([gemm, frobnicate, relu], _), fail),
    % 4. pipeline_depth on empty plan — edge
    chk("empty chain", auto_fuser:fusion_plan([], []), pass),
    % 5. classify_op disjointness: no op in two classes (silent ambiguity check)
    ( setof(Op, C1^C2^(auto_fuser:classify_op(Op,C1), auto_fuser:classify_op(Op,C2), C1 \= C2), Ambig)
      -> format("AMBIGUOUS classifications: ~w -> GATE-BROKEN~n", [Ambig])
      ;  format("classification disjoint: OK~n") ),
    % 6. does the scanner know the PRODUCTION op names? (drift check: engine uses rms_norm, scanner says rmsnorm?)
    ( auto_fuser:classify_op(rms_norm, _) -> format("knows rms_norm (engine name): OK~n") ; format("does NOT know rms_norm (engine spells it rms_norm; scanner has rmsnorm) -> NAME-DRIFT~n") ),
    ( auto_fuser:classify_op(silu_mul, _) -> format("knows silu_mul (engine name): OK~n") ; format("does NOT know silu_mul -> NAME-DRIFT~n") ),
    ( auto_fuser:classify_op(q8_gemv, _) -> format("knows q8_gemv (engine name): OK~n") ; format("does NOT know q8_gemv -> NAME-DRIFT~n") ),
    halt.
