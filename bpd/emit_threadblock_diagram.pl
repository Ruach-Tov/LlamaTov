%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% emit_threadblock_diagram.pl — driver: generate one SVG per kernel name on cmdline.
%%
%% Usage:
%%   swipl -g main bpd/emit_threadblock_diagram.pl -- OUTDIR KERNEL1 [KERNEL2 ...]
%%
%% Writes one SVG per kernel into OUTDIR named "<kernel>.threadblock.svg".

:- use_module('lib/threadblock_diagram.pl').

:- initialization(main, main).

main(Argv) :-
    (   Argv = [OutDir|Kernels],
        Kernels \= []
    ->  forall(member(K, Kernels),
               ( atom_string(KAtom, K),
                 atomic_list_concat([OutDir, '/', KAtom, '.threadblock.svg'], Path),
                 write_diagram_for_kernel(KAtom, Path)
               ))
    ;   format(user_error,
               "Usage: swipl -g main emit_threadblock_diagram.pl -- OUTDIR KERNEL1 [KERNEL2 ...]~n",
               []),
        halt(1)
    ),
    halt(0).
