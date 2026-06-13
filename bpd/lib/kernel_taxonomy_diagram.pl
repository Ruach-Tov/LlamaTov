%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% kernel_taxonomy_diagram.pl — taxonomy-class view of kernel signatures.
%%
%% Co-linear projection alongside norm_shape_diagram.pl's per-op detailed view:
%%   norm_shape_diagram : emit_norm_shape_svg(Op, Path)   — hand-decomposed per-op shape
%%   kernel_taxonomy    : emit_taxonomy_svg(Op, Path)      — arg-token -> 5-shape-class idiom
%% Both read the SAME substrate (op_signatures.pl) and SHARE the LCARS palette
%% (region_color/2 from norm_shape_diagram). Same op, two co-linear views — like
%% ls-by-origin-story's --lang projections.
%%
%% Substrate-of-source: Iyun's 5-shape-class taxonomy (2026-05-31), derived from the
%% op_signatures.pl arg-token grammar (in, out, n, in2, out_scalar, n_rows, scalar).
%% Each arg-token has a geometric role; the token-SET classifies the op into a canonical
%% tensor-diagram idiom from the literature.
%%
%% Tracked source under git (bpd/lib/). Outputs are .o-infixed in /tmp/output-only/.

:- module(kernel_taxonomy_diagram, [
     shape_class/2,            % shape_class(Op, Class)
     classify_args/2,          % classify_args(ArgList, Class)
     resolve_op/2,             % resolve_op(NameOrShort, Canonical)
     emit_taxonomy_svg/2,      % emit_taxonomy_svg(Op, OutPath)
     emit_op_diagram/2,        % emit_op_diagram(Op, Dir) — co-linear: taxonomy + (when applicable) shape
     emit_all_taxonomy/1,      % emit_all_taxonomy(OutDir) — sweep every op_signature
     main/0
   ]).

:- use_module(library(lists)).
:- use_module(op_signatures).       % op_signature(Fn, ArgList)
:- use_module(norm_shape_diagram).  % shared region_color/2 LCARS palette

%% ── the taxonomy: arg-token SET -> shape-class -> literature idiom ──
%% map         : in,out (same rank)        elementwise unary       (i -> i)
%% scaled_map  : + scalar                  alpha-scaled map        (alpha.x)
%% reduce      : out_scalar                 collapse to scalar      (i -> .)   einsum
%% binary      : in,in2,out                two legs merge          (i,i -> i) contraction
%% rowwise     : n_rows                    per-row reduce+broadcast (i,j -> i -> i,j)
classify_args(Args, reduce)     :- memberchk(out_scalar, Args), !.
classify_args(Args, binary)     :- memberchk(in2, Args), !.
classify_args(Args, rowwise)    :- memberchk(n_rows, Args), !.
classify_args(Args, scaled_map) :- memberchk(scalar, Args), memberchk(in, Args), !.
classify_args(Args, map)        :- memberchk(in, Args), memberchk(out, Args), !.
classify_args(_,    other).

shape_class(Op, Class) :-
    resolve_op(Op, Canon),
    op_signatures:op_signature(Canon, Args),
    classify_args(Args, Class).

%% resolve_op/2 — accept short names (silu) or canonical (bpd_silu_cpu / bpd_silu).
%% Tries the name as-is, then bpd_<op>_cpu, then bpd_<op>.
resolve_op(Op, Op) :- op_signatures:op_signature(Op, _), !.
resolve_op(Op, Canon) :-
    atom_concat('bpd_', Op, B0),
    ( atom_concat(B0, '_cpu', Canon), op_signatures:op_signature(Canon, _) -> true
    ; Canon = B0, op_signatures:op_signature(Canon, _) ), !.

%% ── idiom geometry: per class, the abstract diagram recipe (region kind list) ──
%% Each idiom is a list of placed-regions + flow-arrows, rendered with the shared palette.
idiom(map, [
    region(in,  input,  strip),
    region(op,  node,   'f'),
    region(out, output, strip),
    flow(in, op, identity), flow(op, out, identity) ]).
idiom(scaled_map, [
    region(alpha, parameter, node_small),
    region(in,    input,     strip),
    region(op,    node,      '\u00d7'),
    region(out,   output,    strip),
    flow(alpha, op, feed), flow(in, op, identity), flow(op, out, identity) ]).
idiom(reduce, [
    region(in,  input,  strip),
    region(op,  reduction, '\u03a3'),
    region(out, output, scalar_cell),
    flow(in, op, collapse), flow(op, out, emit) ]).
idiom(binary, [
    region(in,  input,  strip),
    region(in2, input,  strip),
    region(op,  node,   '\u2299'),
    region(out, output, strip),
    flow(in, op, merge), flow(in2, op, merge), flow(op, out, identity) ]).
idiom(rowwise, [
    region(in,  input,  grid2d),
    region(op,  reduction, '\u03a3\u2192'),
    region(out, output, grid2d),
    flow(in, op, rowcollapse), flow(op, out, broadcast) ]).

%% ── SVG emission ──
emit_taxonomy_svg(Op, Path) :-
    resolve_op(Op, Canon),
    shape_class(Canon, Class),
    op_signatures:op_signature(Canon, Args),
    ( idiom(Class, _) -> true ; throw(no_idiom(Class)) ),
    W = 440, H = 200,
    format(atom(Hdr),
      '<svg xmlns="http://www.w3.org/2000/svg" width="~w" height="~w" viewBox="0 0 ~w ~w">', [W,H,W,H]),
    svg_defs(Defs),
    bg(W,H,Bg),
    title_block(Op, Args, Class, W, Title),
    render_idiom(Class, W, H, Body),
    atomic_list_concat([Hdr, Defs, Bg, Title, Body, '</svg>'], SVG),
    setup_call_cleanup(open(Path, write, S), write(S, SVG), close(S)).

svg_defs('<defs><marker id="ah" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#666"/></marker></defs>').
bg(W,H,A) :- format(atom(A), '<rect x="0" y="0" width="~w" height="~w" fill="#101418" stroke="#2a3138"/>', [W,H]).

title_block(Op, Args, Class, W, A) :-
    X is W // 2,
    term_to_atom(Args, ArgsA),
    format(atom(A),
     '<text x="~w" y="24" font-family="monospace" font-size="14" text-anchor="middle" fill="#e8eef2" font-weight="bold">~w</text><text x="~w" y="42" font-family="monospace" font-size="10.5" text-anchor="middle" fill="#8aa">~w  \u2014  ~w</text>',
     [X, Op, X, ArgsA, Class]).

%% render_idiom: thin first pass — labelled colored regions + an idiom caption.
%% (Full per-region geometry mirrors the Python prototype; this is the Prolog-native port.)
render_idiom(Class, W, _H, A) :-
    idiom_caption(Class, Cap),
    region_color(input, CIn), region_color(output, COut),
    region_color(reduction, CRed), region_color(parameter, CPar),
    Xc is W // 2,
    ( Class == reduce ->
        Regions = [rect(60,90,18,90,CIn,'in'), circle(330,135,12,CRed,'\u03a3'), rect(372,128,16,16,COut,'scalar')]
    ; Class == binary ->
        Regions = [rect(50,80,18,40,CIn,'in'), rect(50,128,18,40,CIn,'in2'), circle(210,124,11,'#3a5566','\u2299'), rect(340,104,18,40,COut,'out')]
    ; Class == rowwise ->
        Regions = [grid(60,86,6,6,CIn,'in (r\u00d7c)'), circle(270,135,11,CRed,'\u03a3\u2192'), grid(330,86,6,6,COut,'out')]
    ; Class == scaled_map ->
        Regions = [circle(150,86,7,CPar,'\u03b1'), rect(50,100,18,80,CIn,'in'), circle(210,135,11,'#3a5566','\u00d7'), rect(340,100,18,80,COut,'out')]
    ; %% map / other
        Regions = [rect(50,100,18,80,CIn,'in'), circle(210,135,11,'#3a5566','f'), rect(340,100,18,80,COut,'out')]
    ),
    maplist(render_region, Regions, Parts),
    atomic_list_concat(Parts, RegionSVG),
    format(atom(Caption),
      '<text x="~w" y="70" font-family="monospace" font-size="10.5" text-anchor="middle" fill="#9ab">~w</text>', [Xc, Cap]),
    atomic_list_concat([Caption, RegionSVG], A).

idiom_caption(map,        'elementwise map  (i \u2192 i)').
idiom_caption(scaled_map, 'scaled map  (\u03b1\u00b7x)').
idiom_caption(reduce,     'reduction  (i \u2192 \u00b7)   collapse to scalar').
idiom_caption(binary,     'binary contraction  (i,i \u2192 i)').
idiom_caption(rowwise,    'row-wise reduce + broadcast  (i,j \u2192 i \u2192 i,j)').
idiom_caption(other,      '(shape-class not yet mapped)').

render_region(rect(X,Y,Wd,Ht,Col,Lab), A) :-
    LX is X + Wd//2, LY is Y + Ht + 14,
    format(atom(A),
     '<rect x="~w" y="~w" width="~w" height="~w" fill="~w" stroke="#cdd" stroke-width="1"/><text x="~w" y="~w" font-family="monospace" font-size="9" text-anchor="middle" fill="#9ab">~w</text>',
     [X,Y,Wd,Ht,Col,LX,LY,Lab]).
render_region(circle(X,Y,R,Col,Glyph), A) :-
    format(atom(A),
     '<circle cx="~w" cy="~w" r="~w" fill="~w" stroke="#cdd" stroke-width="1.4"/><text x="~w" y="~w" font-family="monospace" font-size="11" text-anchor="middle" fill="#fff" font-weight="bold">~w</text>',
     [X,Y,R,Col,X,Y+4,Glyph]).
render_region(grid(X,Y,Cols,Rows,Col,Lab), A) :-
    Cell = 13,
    findall(Cell_, ( between(0,5,R), between(0,5,C),
                     PX is X+C*Cell, PY is Y+R*Cell,
                     format(atom(Cell_), '<rect x="~w" y="~w" width="~w" height="~w" fill="~w" stroke="#345" stroke-width="0.6"/>', [PX,PY,Cell,Cell,Col]) ),
            Cells),
    ignore((Cols=Cols,Rows=Rows)),
    atomic_list_concat(Cells, GridSVG),
    LX is X+3*Cell, LY is Y+6*Cell+14,
    format(atom(A), '~w<text x="~w" y="~w" font-family="monospace" font-size="9" text-anchor="middle" fill="#9ab">~w</text>', [GridSVG,LX,LY,Lab]).

%% ── combiner: co-linear projections (taxonomy-view + per-op detail-view) ──
%% Makes the co-linear discipline load-bearing at the API layer: one call, two views.
%% Writes Dir/{Op}.taxonomy.o.svg always; Dir/{Op}.shape.o.svg when norm_shape_diagram
%% has a hand-decomposed operand model for the op.
%% Bridge: op_signatures.pl names (bpd_rmsnorm_cpu) <-> norm_shape_diagram detail names
%% (rms_norm). The two generators evolved separate name-spaces (concatenated vs ggml-style
%% underscored); this explicit table is the substrate-of-record for the join.
op_detail_alias(bpd_rmsnorm_cpu,   rms_norm).
op_detail_alias(bpd_layernorm_cpu, layer_norm).
op_detail_alias(bpd_l2norm_cpu,    l2_norm).
op_detail_alias(bpd_groupnorm_cpu, group_norm).

emit_op_diagram(Op, Dir) :-
    resolve_op(Op, Canon),
    format(atom(TaxP), '~w/~w.taxonomy.o.svg', [Dir, Canon]),
    emit_taxonomy_svg(Canon, TaxP),
    %% detail-view (when norm_shape_diagram has a hand-decomposed operand model)
    ( op_detail_alias(Canon, DOp), norm_shape_diagram:operand(DOp, _, _, _) ->
        format(atom(ShP), '~w/~w.shape.o.svg', [Dir, DOp]),
        norm_shape_diagram:emit_norm_shape_svg(DOp, ShP)
    ; true ).

%% ── sweep: emit a taxonomy SVG for every op_signature ──
emit_all_taxonomy(OutDir) :-
    findall(Op, op_signature(Op, _), Ops0),
    sort(Ops0, Ops),
    forall(member(Op, Ops),
      ( shape_class(Op, Cls),
        ( Cls == other -> true   % skip unmapped (e.g. unsupported sigs)
        ; format(atom(P), '~w/~w.taxonomy.o.svg', [OutDir, Op]),
          catch(emit_taxonomy_svg(Op, P), E, (print_message(warning, E), true))
        ) )).

main :-
    ( current_prolog_flag(argv, [Dir|_]) -> OutDir = Dir ; OutDir = '/tmp/output-only-iyun' ),
    ( exists_directory(OutDir) -> true ; make_directory(OutDir) ),
    emit_all_taxonomy(OutDir),
    findall(Op-C, shape_class(Op,C), Pairs),
    aggregate_all(count, member(_-_, Pairs), N),
    format("emitted taxonomy diagrams for ~w ops -> ~w~n", [N, OutDir]).
