%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% FAILED-TEST INJECTION: cupti_facts ingestion + window + aggregation math.
:- consult("bpd/tools/cupti/cupti_facts.pl").
:- initialization(main).
main :-
    %% synthesize a fact file: 2 big htod (weights) + 3 small htod (tokens) + kernels
    open("/tmp/synth_cupti.facts", write, S),
    format(S, "memcpy(htod, 99000000, 100, 200, 100).~n", []),   % big weight upload
    format(S, "memcpy(htod, 88000000, 300, 400, 100).~n", []),
    format(S, "memcpy(htod, 4096, 1000, 1010, 10).~n", []),      % token uploads (small)
    format(S, "memcpy(htod, 4096, 2000, 2010, 10).~n", []),
    format(S, "memcpy(htod, 4096, 3000, 3010, 10).~n", []),
    format(S, "memcpy(dtoh, 4, 3500, 3510, 10).~n", []),
    format(S, "kernel(k_gemv, 1100, 1200, 100, 56, 256).~n", []),
    format(S, "kernel(k_gemv, 2100, 2200, 100, 56, 256).~n", []),
    format(S, "kernel(k_rms, 2300, 2350, 50, 1, 256).~n", []),
    close(S),
    cupti_facts:load_cupti("/tmp/synth_cupti.facts"),
    %% 1. window detection: last small htod=3000, prior=2000 -> window [2000, maxEnd=3510]
    ( cupti_facts:steady_state_window(W0, W1), W0 =:= 2000, W1 =:= 3510
      -> format("window [~w,~w] -> OK~n",[W0,W1])
      ;  cupti_facts:steady_state_window(X0, X1), format("window [~w,~w] EXPECTED [2000,3510] -> BROKEN~n",[X0,X1]) ),
    %% 2. copies in window: 2 small htod (2000,3000) + 1 dtoh
    cupti_facts:copies_in_window(2000, 3510, copies(HN-HB, DN-DB)),
    ( HN =:= 2, HB =:= 8192, DN =:= 1, DB =:= 4
      -> format("copies_in_window htod ~w/~wB dtoh ~w/~wB -> OK~n",[HN,HB,DN,DB])
      ;  format("copies_in_window got htod ~w/~w dtoh ~w/~w -> BROKEN~n",[HN,HB,DN,DB]) ),
    halt.
