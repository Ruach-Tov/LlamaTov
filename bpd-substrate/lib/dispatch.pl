%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════
%% dispatch.pl — Safe kernel dispatch with invariant checking
%%
%% Every kernel launch goes through dispatch/3, which runs
%% diagnose_graph/1 BEFORE dispatch. If any invariant fails,
%% dispatch refuses to launch and reports the violation.
%%
%% Licensed under GPLv2
%% ═══════════════════════════════════════════════════════════════════════

/** <module> Safe Kernel Dispatch

Every kernel launch must go through dispatch/3. Before launching,
the compute graph is validated against all structural invariants:
view containment, dtype coherence, write-read coherence, temporal
ordering, and shape compatibility.

If any invariant fails, dispatch REFUSES to launch and returns
an error with a human-readable diagnostic. This catches bugs at
dispatch time, not at result-comparison time.

Example:
==
?- dispatch(k_sw_antidiag, config(block_size(128)), Result).
Result = ok(score=208, time_ms=2.3)

?- dispatch(k_attention_buggy, config(...), Result).
Result = error(pre_dispatch_check_failed, [
    diag(dtype_coherence, mm-attn, error(dtype_mismatch, ...)),
    diag(write_read, cpy-attn, error(write_read_disjoint, ...))
])
==

@author Ruach Tov Collective
@see lib/compute_graph_invariants.pl for the invariant definitions
@see lib/cupti_profile.pl for post-dispatch performance profiling
*/

:- module(dispatch, [
    dispatch/3,
    dispatch_unchecked/3,
    register_graph/2,
    set_dispatch_policy/1,
    dispatch_policy/1
]).

:- use_module('compute_graph_invariants').

:- dynamic dispatch_policy/1.
:- dynamic registered_graph/2.

%% Default policy: always check invariants before dispatch
dispatch_policy(check_always).

%! set_dispatch_policy(+Policy) is det.
%  Policy is one of:
%    check_always    — run all invariants before every dispatch (safest)
%    check_once      — run invariants on first dispatch, cache result
%    check_never     — skip invariant checks (fastest, UNSAFE)
%    check_errors    — only check error-level invariants, skip warnings
set_dispatch_policy(Policy) :-
    retractall(dispatch_policy(_)),
    assertz(dispatch_policy(Policy)).

%! register_graph(+KernelName, +GraphSetupGoal) is det.
%  Register a compute graph for a kernel. GraphSetupGoal is a Prolog
%  goal that asserts the tensor and op facts for this kernel's graph.
register_graph(KernelName, GraphSetupGoal) :-
    retractall(registered_graph(KernelName, _)),
    assertz(registered_graph(KernelName, GraphSetupGoal)).

%! dispatch(+KernelName, +Config, -Result) is det.
%  Safe dispatch: check invariants, then launch.
%  Refuses to launch if any error-level invariant fails.
dispatch(KernelName, Config, Result) :-
    dispatch_policy(Policy),
    (Policy = check_never ->
        dispatch_unchecked(KernelName, Config, Result)
    ;
        %% Set up the graph if registered
        (registered_graph(KernelName, SetupGoal) ->
            clear_graph,
            call(SetupGoal)
        ; true),
        
        %% Run invariant checks
        check_all_invariants(Diagnostics),
        
        %% Separate errors from warnings
        include(is_error_diag, Diagnostics, Errors),
        include(is_warning_diag, Diagnostics, Warnings),
        
        %% Report warnings
        (Warnings \= [] ->
            length(Warnings, NW),
            format('[dispatch] ~w warnings for ~w:~n', [NW, KernelName]),
            forall(member(W, Warnings),
                (W = diag(Class, Subject, warning(Type, Msg)) ->
                    format('  [WARN] ~w/~w: ~w~n', [Class, Subject, Msg])
                ;
                    format('  [WARN] ~w~n', [W])
                ))
        ; true),
        
        %% Check errors
        (Errors \= [] ->
            %% REFUSE TO LAUNCH
            length(Errors, NE),
            format('[dispatch] BLOCKED: ~w error(s) for ~w:~n', [NE, KernelName]),
            forall(member(E, Errors),
                (E = diag(Class, Subject, error(Type, Msg)) ->
                    format('  [ERROR] ~w/~w: ~w~n', [Class, Subject, Msg])
                ;
                    format('  [ERROR] ~w~n', [E])
                )),
            Result = error(pre_dispatch_check_failed, Errors)
        ;
            (Policy = check_errors ->
                dispatch_unchecked(KernelName, Config, Result)
            ;
                %% All clear — dispatch
                format('[dispatch] All invariants passed for ~w. Launching.~n', [KernelName]),
                dispatch_unchecked(KernelName, Config, Result)
            )
        )
    ).

%! dispatch_unchecked(+KernelName, +Config, -Result) is det.
%  Launch without invariant checks. Used internally after checks pass,
%  or when policy is check_never.
dispatch_unchecked(KernelName, Config, Result) :-
    format('[dispatch] Launching ~w with ~w~n', [KernelName, Config]),
    %% Actual kernel launch would go here — via ctypes, FFI, or system call
    %% For now, return a placeholder
    Result = ok(launched(KernelName, Config)).

%% Helper: classify diagnostics
is_error_diag(diag(_, _, error(_, _))).
is_warning_diag(diag(_, _, warning(_, _))).
