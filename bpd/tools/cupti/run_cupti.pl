%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
:- module(runner, []).
:- use_module('tools/cupti/cupti_facts.pl').
:- initialization(main).
main :-
    style_check(-discontiguous),
    load_cupti('/tmp/cupti_occ.facts'),
    nl, copy_inefficiency(V), format("COPY VERDICT: ~q~n",[V]),
    nl, kernel_time_summary,
    nl, occupancy_summary,
    halt.
