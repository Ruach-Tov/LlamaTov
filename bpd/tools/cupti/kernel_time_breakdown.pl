%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
:- initialization(main).
main :-
    consult('/tmp/CUPTI_FACTS'),
    findall(Name, kernel(Name,_,_,_,_,_), Names0), sort(Names0, Names),
    format("=== ~w ===~n", ['/tmp/CUPTI_FACTS']),
    forall(member(Name, Names),
        ( aggregate_all(count, kernel(Name,_,_,_,_,_), N),
          aggregate_all(sum(E-S), kernel(Name,S,E,_,_,_), Dur),
          format("  ~w~t~26|: ~d launches  ~D ns total~n", [Name, N, Dur]))),
    aggregate_all(sum(E2-S2), kernel(_,S2,E2,_,_,_), Total),
    format("  TOTAL kernel ns: ~D~n", [Total]),
    halt.
