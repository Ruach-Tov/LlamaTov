%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
% cupti_facts.pl — the ACQUISITION→REASONING bridge for cupti-from-prolog.
%
% Loads CUPTI activity facts (memcpy/4, kernel/6 emitted by libcupti_trace.so) and provides
% scientific queries over host<->device traffic and kernel time. This is the BODY the
% reasoning head (cupti_profile.pl) was missing: it actually quantifies inefficiencies.
%
% Facts ingested (from the tracer):
%   memcpy(Kind, Bytes, StartNs, EndNs, DurationNs).   Kind = htod|dtoh|dtod|htoh|other
%   kernel(Name, StartNs, EndNs, DurationNs, GridX, BlockX).
%
% Usage:
%   ?- load_cupti('/tmp/cupti.facts').
%   ?- copy_summary.                         % all host<->device traffic by kind
%   ?- steady_state_window(Start, End).      % auto-detect the per-token replay window
%   ?- copy_inefficiency(Verdict).           % the headline: which copies are avoidable
%   ?- kernel_time_summary.                  % per-kernel-type total GPU time

:- module(cupti_facts, [
    load_cupti/1,
    copy_summary/0,
    kernel_time_summary/0,
    steady_state_window/2,
    copy_inefficiency/1,
    copies_in_window/3,
    host_device_class/2,
    occupancy_summary/0,
    launch_config/2
]).

:- dynamic memcpy/5.
:- dynamic kernel/6.
:- dynamic kernel_launch/8.

% ── ingestion ──────────────────────────────────────────────────────────────
load_cupti(Path) :-
    retractall(memcpy(_,_,_,_,_)),
    retractall(kernel(_,_,_,_,_,_)),
    retractall(kernel_launch(_,_,_,_,_,_,_,_)),
    consult(Path),
    findall(_, memcpy(_,_,_,_,_), Ms), length(Ms, NM),
    findall(_, kernel(_,_,_,_,_,_), Ks), length(Ks, NK),
    findall(_, kernel_launch(_,_,_,_,_,_,_,_), Ls), length(Ls, NL),
    format("loaded ~w memcpy + ~w kernel + ~w launch-config events~n", [NM, NK, NL]).

% A copy crosses the host<->device boundary iff it's htod or dtoh (dtod is on-device).
host_device_class(htod, host_to_device).
host_device_class(dtoh, device_to_host).

is_host_crossing(Kind) :- (Kind == htod ; Kind == dtoh).

% ── copy summary (all events) ──────────────────────────────────────────────
copy_summary :-
    format("=== HOST<->DEVICE COPY SUMMARY (all events) ===~n", []),
    forall(member(K, [htod, dtoh, dtod, htoh, other]),
        ( aggregate_all(count, memcpy(K,_,_,_,_), N),
          aggregate_all(sum(B), memcpy(K,B,_,_,_), Bytes),
          aggregate_all(sum(D), memcpy(K,_,_,_,D), DurNs),
          ( N > 0
          -> ( is_host_crossing(K) -> Tag = "  [HOST CROSSING]" ; Tag = "" ),
             format("  ~w~t~8|: ~d copies, ~D bytes, ~D ns~w~n", [K, N, Bytes, DurNs, Tag])
          ; true )
        )).

% ── steady-state window detection ──────────────────────────────────────────
% The big one-time weight uploads happen at seed/capture; steady-state per-token replay is
% the TAIL. Heuristic: find the last small (<= 1MB) htod (an embedding upload = a token's
% input) and take the window from the PRIOR such upload to the last event.
steady_state_window(Start, End) :-
    findall(S, ( memcpy(htod, B, S, _, _), B =< 1048576 ), Smalls),
    sort(0, @>=, Smalls, [End0|_]),            % most recent small htod start
    msort(Smalls, Sorted),
    ( nth0(I, Sorted, End0), I0 is I-1, I0 >= 0, nth0(I0, Sorted, Start)
    -> true ; Start = End0 ),
    findall(E, memcpy(_,_,_,E,_), Es), max_list(Es, End).

copies_in_window(Start, End, copies(HtodN-HtodB, DtohN-DtohB)) :-
    aggregate_all(count, ( memcpy(htod,_,S,_,_), S >= Start, S =< End ), HtodN),
    aggregate_all(sum(B), ( memcpy(htod,B,S,_,_), S >= Start, S =< End ), HtodB),
    aggregate_all(count, ( memcpy(dtoh,_,S,_,_), S >= Start, S =< End ), DtohN),
    aggregate_all(sum(B), ( memcpy(dtoh,B,S,_,_), S >= Start, S =< End ), DtohB).

% ── the headline: copy inefficiency ────────────────────────────────────────
% Classify each per-token host crossing as necessary or avoidable, with the reason.
%   htod small (embedding) : NECESSARY — the token's input must enter the device.
%   dtoh large (full logits): AVOIDABLE — pulled back only for argmax; a device-side
%                             reduction returns 1 token (4 B) instead of vocab*4 bytes.
copy_inefficiency(verdict(Necessary, Avoidable, Detail)) :-
    steady_state_window(Start, End),
    copies_in_window(Start, End, copies(_-HtodB, _-DtohB)),
    Necessary = necessary(embedding_upload, HtodB),
    ( DtohB > 4096
    -> Saved is DtohB - 4,
       Avoidable = avoidable(logits_readback_for_argmax, DtohB,
                             device_argmax_returns_one_token, saves(Saved)),
       Detail = "DtoH pulls the full logits vector to host for argmax; a device-side argmax/sample returns one token. Eliminating it removes nearly all per-token host->device readback traffic."
    ; Avoidable = none,
      Detail = "no large readback in window" ).

% ── kernel time summary ────────────────────────────────────────────────────
kernel_time_summary :-
    format("=== KERNEL GPU TIME by type (ns) ===~n", []),
    findall(Name, kernel(Name,_,_,_,_,_), Names0), sort(Names0, Names),
    forall(member(Name, Names),
        ( aggregate_all(count, kernel(Name,_,_,_,_,_), N),
          %% duration = End - Start (field4 carries this too, but compute it to be format-robust).
          aggregate_all(sum(End - Start), kernel(Name,Start,End,_,_,_), Dur),
          format("  ~w~t~28|: ~d launches, ~D ns total~n", [Name, N, Dur]))),
    aggregate_all(sum(E2 - S2), kernel(_,S2,E2,_,_,_), Total),
    format("  ~w~t~28|: ~D ns total~n", ['TOTAL', Total]).

% ── occupancy / launch-config analysis ─────────────────────────────────────
% kernel_launch(Name, Regs, StaticShmem, DynShmem, LocalMem, BlockSize, GridCount, DevId).
% sm_61 (Tesla P4) limits: 65536 regs/SM, 98304 B shared/SM, 2048 threads/SM, 32 blocks/SM.
sm61_limit(regs_per_sm, 65536).
sm61_limit(shmem_per_sm, 98304).
sm61_limit(threads_per_sm, 2048).
sm61_limit(blocks_per_sm, 32).

launch_config(Name, config(Regs, Shmem, Block, Grid)) :-
    kernel_launch(Name, Regs, SS, DS, _, Block, Grid, _),
    Shmem is SS + DS.

% Theoretical max resident BLOCKS per SM given the launch config (the binding limit).
theoretical_blocks_per_sm(Name, Blocks, LimitedBy) :-
    kernel_launch(Name, Regs, SS, DS, _, Block, _, _),
    Shmem is SS + DS,
    sm61_limit(regs_per_sm, RL), sm61_limit(shmem_per_sm, ShL),
    sm61_limit(threads_per_sm, TL), sm61_limit(blocks_per_sm, BL),
    ( Block > 0 -> ByThreads is TL // Block ; ByThreads = BL ),
    ( Regs > 0, Block > 0 -> ByRegs is RL // (Regs * Block) ; ByRegs = BL ),
    ( Shmem > 0 -> ByShmem is ShL // Shmem ; ByShmem = BL ),
    Blocks0 is min(BL, min(ByThreads, min(ByRegs, ByShmem))),
    Blocks is max(1, Blocks0),
    ( Blocks =:= ByThreads -> LimitedBy = block_size
    ; Blocks =:= ByRegs   -> LimitedBy = registers
    ; Blocks =:= ByShmem  -> LimitedBy = shared_memory
    ; LimitedBy = block_count ).

% Achieved-occupancy ESTIMATE: resident warps / max warps. (Theoretical, from launch config
% — not measured achieved occupancy, which needs the metric API. Still diagnostic: a low
% theoretical occupancy is a hard ceiling no scheduling can beat.)
theoretical_occupancy(Name, Pct, LimitedBy) :-
    theoretical_blocks_per_sm(Name, Blocks, LimitedBy),
    kernel_launch(Name, _, _, _, _, Block, _, _),
    sm61_limit(threads_per_sm, TL),
    ResidentThreads is Blocks * Block,
    Pct is (ResidentThreads * 100) // TL.

occupancy_summary :-
    format("=== THEORETICAL OCCUPANCY (sm_61, from launch config) ===~n", []),
    findall(Name, kernel_launch(Name,_,_,_,_,_,_,_), Ns0), sort(Ns0, Names),
    forall(member(Name, Names),
        ( launch_config(Name, config(R,Sh,B,_)),
          theoretical_occupancy(Name, Pct, Lim),
          ( Pct < 50 -> Flag = "  [LOW — ceiling]" ; Flag = "" ),
          format("  ~w~t~24|: ~d% occ, ~d regs, ~d B shmem, block ~d, limited by ~w~w~n",
                 [Name, Pct, R, Sh, B, Lim, Flag]) )).
