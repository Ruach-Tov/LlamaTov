%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% render_ascii.pl — ASCII/box-drawing renderer for the tensor-op type system.
%%
%% TOP LAYER (first backend). Every table is a PURE RENDER of the fact base declared
%% in tensor_schema.pl. The HTML/SVG backend (future) will share this exact
%% render(Coord, View) query logic and differ only in the cell-formatting backend.
%%
%% Views (column projections over the same op-row relation):
%%   dashboard      (DEFAULT) : # | op | elems | maxULP | ndiff | verdict
%%   program                  : node | op | tensor | description (describe/2)
%%   correspondence           : ggml-node | our-buffer | our-op
%%   taps                     : ggml-node | tap-after | our-buffer | dump-size
%%   bandwidth                : op | bytes_r | bytes_w | arith_intensity | area-art
%%
%% Off-page connectors (filmstrip): the last row links to the next coordinate
%% segment, the header to the previous — derived from layer_coords adjacency.

:- module(render_ascii, [ render/2, render/3 ]).
:- use_module(library(tensor_schema)).
:- use_module(library(lists)).

%% ── layer_coords adjacency (the drill-down filmstrip order) ────────────────────
%% Mirror of divergence_heatmap.pl's layer_coords. Drives prev/next connectors.
layer_segment_order([attn_norm, qkv, rope, score, o_proj,
                     residual1, ffn_norm, swiglu, residual2]).

seg_next(Seg, Next) :-
    layer_segment_order(L), nth0(I, L, Seg),
    I1 is I + 1, nth0(I1, L, Next).
seg_prev(Seg, Prev) :-
    layer_segment_order(L), nth0(I, L, Seg),
    I > 0, I0 is I - 1, nth0(I0, L, Prev).

%% extract the segment name from a coordinate path [mistral,layer(l),attn,SEG]
coord_segment(Coord, Seg) :- last(Coord, Seg).

%% ── render/2: render a coordinate's default (dashboard) view ───────────────────
render(Coord) :- render(Coord, dashboard).

render(Coord, View) :- render(Coord, View, user_output).

%% ── render/3: render to a stream ───────────────────────────────────────────────
render(Coord, View, S) :-
    coord_segment(Coord, Seg),
    title_for(View, Seg, Title),
    format(S, '~n  ~w~n', [Title]),
    columns(View, Cols),
    rows(Coord, View, Rows),
    print_table(S, Cols, Rows),
    print_connectors(S, Seg).

title_for(dashboard, Seg, T)      :- format(atom(T), 'Table 10011.~w  \u2014 conformance dashboard', [Seg]).
title_for(program, Seg, T)        :- format(atom(T), 'Table 10011.~w  \u2014 op program', [Seg]).
title_for(correspondence, Seg, T) :- format(atom(T), 'Table 10011.~w  \u2014 ggml\u2194ours correspondence', [Seg]).
title_for(taps, Seg, T)           :- format(atom(T), 'Table 10011.~w  \u2014 measurement taps', [Seg]).
title_for(bandwidth, Seg, T)      :- format(atom(T), 'Table 10011.~w  \u2014 memory bandwidth', [Seg]).

%% ── column specs: list of col(Header, Width, Align) ────────────────────────────
columns(dashboard, [col('#',3,r), col('op',30,l), col('elems',9,r),
                    col('maxULP',8,r), col('ndiff',13,r), col('verdict',22,l)]).
columns(program,   [col('num',6,l), col('node',6,l), col('ggml-op',12,l),
                    col('tensors',30,l), col('what it computes',34,l)]).
columns(correspondence, [col('ggml node',16,l), col('our op',16,l), col('verdict',20,l)]).
columns(taps,      [col('ggml node',16,l), col('our buffer',18,l), col('dump size',14,l)]).
columns(bandwidth, [col('op',26,l), col('bytes_r',10,r), col('bytes_w',10,r),
                    col('AI',6,r), col('size',28,l)]).

%% ── rows: per-view projection over the fact base ───────────────────────────────
rows(Coord, dashboard, Rows) :-
    findall(R, dashboard_row(Coord, R), Rows0),
    enumerate(Rows0, 1, Rows).
rows(Coord, program, Rows) :-
    findall([Node, OpU, Tensor, Desc],
            ( op(Node, Coord, OpType, _, out(Tensor)),
              upcase_atom(OpType, OpU),
              ( op_role(Node, Role), describe(Role, Desc) -> true ; Desc = '' )
            ), Rows0),
    add_seq_num(Rows0, 1, Rows).
rows(Coord, bandwidth, Rows) :-
    findall(Row, bandwidth_row(Coord, Row), Rows).

%% dashboard row: op description + verdict (from op_verdict or layout_relation)
dashboard_row(Coord, [OpDesc, Elems, MaxUlp, NdiffStr, Verdict]) :-
    op(Node, Coord, _, _, _),
    ( op_role(Node, Role), describe(Role, OpDesc) -> true ; OpDesc = Node ),
    verdict_of(Node, Elems, MaxUlp, Ndiff, Verdict),
    format(atom(NdiffStr), '~w', [Ndiff]).

%% verdict resolution: measured op_verdict, else layout_relation, else covered
verdict_of(Node, Elems, MaxUlp, Ndiff, Verdict) :-
    ( op_verdict(Node, measured(max_ulp(MaxUlp), ndiff(Ndiff), max_abs(_), elems(Elems), ref(_), test(_)))
    -> ( MaxUlp =:= 0 -> Verdict = '\u2705 0-ULP bit_identical'
       ; format(atom(Verdict), '\u274c DIVERGENT', []) )
    ; layout_relation(Node, _, _, equiv_proof(Basis, _))
    -> Elems = '-', MaxUlp = '-', Ndiff = '-',
       format(atom(Verdict), '\u2705 layout_equiv (~w)', [Basis])
    ; Elems = '-', MaxUlp = '-', Ndiff = '-', Verdict = '\u00b7 covered (consumer 0-ULP)'
    ).

%% bandwidth row: placeholder until mavchin's op_bandwidth/5 lands
bandwidth_row(Coord, [OpDesc, R, W, AI, Art]) :-
    op(Node, Coord, _, _, out(Tensor)),
    ( op_role(Node, Role), describe(Role, OpDesc) -> true ; OpDesc = Node ),
    ( tensor(Tensor, dtype(D), shape(Shape), _),
      tensor_bytes(D, Shape, Bytes)
    -> R = '?', W = Bytes, AI = '?', area_art(Bytes, Art)
    ;  R = '?', W = '?', AI = '?', Art = '' ).

%% tensor_bytes: element count * bytes_per_element (mavchin's bytes_per_element)
bytes_per_element(f32, 4). bytes_per_element(f16, 2).
bytes_per_element(q8_0, 1). bytes_per_element(i32, 4).
tensor_bytes(D, Shape, Bytes) :-
    bytes_per_element(D, BPE), proji(Shape, 1, Prod), Bytes is Prod * BPE.
proji([], A, A).
proji([H|T], A, R) :- A1 is A*H, proji(T, A1, R).

%% proportional-area art (placeholder bar; mavchin's spec refines this)
area_art(Bytes, Art) :-
    Blocks is max(1, integer(truncate(Bytes / 4096))),
    Blocks1 is min(Blocks, 24),
    length(L, Blocks1), maplist(=('\u2588'), L), atomic_list_concat(L, Bar),
    format(atom(Art), '~w ~w B', [Bar, Bytes]).

%% ── connectors: off-page filmstrip links from layer_coords adjacency ───────────
print_connectors(S, Seg) :-
    ( seg_prev(Seg, Prev), describe(previous(Prev), PrevDesc)
      -> format(S, '  ~w   (Table 10011.~w)~n', [PrevDesc, Prev]) ; true ),
    ( seg_next(Seg, Next), describe(continues(Next), NextDesc)
      -> format(S, '  ~w   (Table 10011.~w)~n', [NextDesc, Next]) ; true ).


%% add_seq_num: prepend a %04d sequential line number (our 'num'); keep ggml node id.
add_seq_num([], _, []).
add_seq_num([[Node|Rest]|T], N, [[NumA, Node|Rest]|Out]) :-
    fmt04(N, NumA), N1 is N+1, add_seq_num(T, N1, Out).

%% fmt04: integer -> '%04d' atom (0001, 0029, ...)
fmt04(N, A) :- format(atom(A), '~|~`0t~d~4+', [N]).

%% ── generic box-drawing table printer ──────────────────────────────────────────
print_table(S, Cols, Rows) :-
    header_line(S, Cols),
    rule_line(S, Cols),
    forall(member(Row, Rows), data_line(S, Cols, Row)).

header_line(S, Cols) :-
    format(S, '  ', []),
    forall(member(col(H,W,_), Cols), ( pad(H, W, P), format(S, '~w  ', [P]) )),
    nl(S).
rule_line(S, Cols) :-
    total_width(Cols, TW),
    length(L, TW), maplist(=('\u2501'), L), atomic_list_concat(L, Rule),
    format(S, '  ~w~n', [Rule]).
data_line(S, Cols, Row) :-
    format(S, '  ', []),
    pair_cols(Cols, Row, Pairs),
    forall(member(col(_,W,A)-V, Pairs),
           ( fmt_cell(V, Vs), align(Vs, W, A, P), format(S, '~w  ', [P]) )),
    nl(S).

pair_cols([], _, []).
pair_cols([C|Cs], [V|Vs], [C-V|Ps]) :- pair_cols(Cs, Vs, Ps).
pair_cols([C|Cs], [], [C-''|Ps]) :- pair_cols(Cs, [], Ps).

fmt_cell(V, S) :- ( atom(V) -> S = V ; number(V) -> atom_number(S, V) ; term_to_atom(V, S) ).
pad(A, W, P) :- align(A, W, l, P).
align(A, W, l, P) :- atom_length(A, L), ( L >= W -> P = A ; Sp is W-L, spaces(Sp, S), atom_concat(A, S, P) ).
align(A, W, r, P) :- atom_length(A, L), ( L >= W -> P = A ; Sp is W-L, spaces(Sp, S), atom_concat(S, A, P) ).
spaces(0, '') :- !.
spaces(N, S) :- N > 0, N1 is N-1, spaces(N1, S0), atom_concat(' ', S0, S).
total_width(Cols, TW) :- foldl([col(_,W,_),A,B]>>(B is A+W+2), Cols, 0, TW).

enumerate([], _, []).
enumerate([R|Rs], N, [[N|R]|Out]) :- N1 is N+1, enumerate(Rs, N1, Out).
