%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% dashboard_common.pl — shared dashboard utilities (Iyun + mavchin, 2026-05-29).
%% DRY: the freshness stamp is used by all four dashboard generators (Tables 10000/10001/10010/10011).
%% Makes every dashboard PROVABLY LIVE on its own face — and combined with the diviner's
%% safe-silent-fallback (serves last-good on pull failure), the stamp makes staleness VISIBLE
%% (shows the actual generated commit), so the dashboard cannot lie about its own freshness.

:- module(dashboard_common,
    [ freshness_stamp/2, freshness_stamp_svg/4, git_short_head/1, now_utc/1 ]).

%% freshness_stamp(Commit, When) — the current git commit hash + now-UTC, read at render time so
%% the stamp always reflects the ACTUAL generated state (not a passed-in / possibly-stale value).
freshness_stamp(Commit, When) :-
    ( catch(git_short_head(C), _, fail) -> Commit = C ; Commit = unknown ),
    ( catch(now_utc(W), _, fail) -> When = W ; When = unknown ).

%% freshness_stamp_svg(Stream, X, Y, FontSize) — emit the footer <text> element. One call per
%% generator; each just picks its X/Y/size. The text is uniform across all four dashboards.
freshness_stamp_svg(S, X, Y, FontSize) :-
    freshness_stamp(Commit, When),
    format(S, '  <text x="~w" y="~w" font-size="~w" fill="#7a6a5a" font-style="italic">facts as of commit ~w \u00b7 generated ~w UTC \u00b7 live via diviner git-pull</text>~n',
           [X, Y, FontSize, Commit, When]).

%% git_short_head(Hash) — short commit hash via `git rev-parse --short HEAD`.
git_short_head(Hash) :-
    process_create(path(git), ['rev-parse','--short','HEAD'],
                   [stdout(pipe(Out)), stderr(null), process(P)]),
    read_string(Out, _, S0), close(Out), process_wait(P, _),
    split_string(S0, "\n", "\n ", [HS|_]), atom_string(Hash, HS).

%% now_utc(When) — 'YYYY-MM-DD HH:MM' in UTC, zero-padded.
now_utc(When) :-
    get_time(T), stamp_date_time(T, DT, 'UTC'),
    date_time_value(date, DT, date(Y,Mo,D)), date_time_value(time, DT, time(H,Mi,_)),
    format(atom(When), '~w-~|~`0t~w~2+-~|~`0t~w~2+ ~|~`0t~w~2+:~|~`0t~w~2+', [Y,Mo,D,H,Mi]).
