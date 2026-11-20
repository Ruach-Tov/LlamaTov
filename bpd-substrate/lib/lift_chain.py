#!/usr/bin/env python3
"""lift_chain.py — auto-LIFT a KernelBench problem's forward method
into chain facts for the bridge (bridge_emit.pl).

The lifter parses the forward() body's op sequence (the x = ... lines)
into chain(Problem, [op1(args), op2(args), ...]) Prolog facts.
Covers the elementwise vocabulary; reduction/matrix ops are recognized
and marked (they select the buffered-row/whole-matrix patterns).
ZERO per-problem hand-work: the same parser lifts any problem file.
"""
import ast
import re
import sys


def lift(path):
    """Parse a KernelBench problem file → (name, [ops]) or raise."""
    src = open(path).read()
    tree = ast.parse(src)
    # find Model.forward:
    fwd = None
    init = None
    _cls_methods = {}   # method-name -> FunctionDef (the Swin
    #                      method-entry rung: self.METHOD(x) resolves)
    _UC_ACT_OPS = {'ReLU': ('relu',), 'ReLU6': ('clamp', 0.0, 6.0),
                   'Sigmoid': ('sigmoid',), 'Tanh': ('tanh',),
                   'GELU': ('gelu',)}

    def _walk_user_forward(cname, bind, outer_i, attr_map):
        # RUNG-3c: merge the class-walk's computed env (init-attrs
        # like self.padding) into the forward bind:
        _ce4 = _class_envs.get(str(outer_i))
        if _ce4:
            bind = {**_ce4, **bind}
        _in_user_class[0] += 1
        try:
            return _walk_user_forward_inner(cname, bind, outer_i,
                                            attr_map)
        finally:
            _in_user_class[0] -= 1

    def _walk_user_forward_inner(cname, bind, outer_i, attr_map):
        """THE FORWARD-WALK HOP (the #18 relay's 3rd leg): walk the
        user-class forward in APPLICATION ORDER. attr_map: attr ->
        (cls, path) from the init-walk. Applications of activation-
        class attrs emit OPS (fusable islands); others emit stages.
        The return-cat emits the #17 branch form (brackets + m_cat2).
        Returns the ordered spec-list or None (honest refuse)."""
        _ucd = _user_classes.get(cname)
        if _ucd is None:
            return None
        _fwd = _ucd.get('forward')
        if _fwd is None:
            return None

        def _app_chain(e):
            # self.A(self.B(x)) -> [B-spec, A-spec]; self.A(x) -> [A-spec]
            out = []
            cur = e
            chain = []
            while isinstance(cur, ast.Call) and \
                    isinstance(cur.func, ast.Attribute) and \
                    isinstance(cur.func.value, ast.Name) and \
                    cur.func.value.id == 'self' and len(cur.args) == 1:
                chain.append(cur.func.attr)
                cur = cur.args[0]
            if not (isinstance(cur, ast.Name)):
                return None
            for attr in reversed(chain):
                if attr not in attr_map and \
                        f'{outer_i}.{attr}' in \
                        _walked_subinstances:
                    # RUNG-3d (the forward-SPLICE — the GELU
                    # islands): the attr is a WALKED SUB-INSTANCE
                    # (Iyun's 3d stash) → splice its FORWARD-walk
                    # inline (fc1 → act[the island] → drop → fc2 →
                    # drop in FORWARD order) instead of an opaque
                    # stage. The sub-attr_map builds from the
                    # walked init-stages' paths:
                    _swc, _swb = _walked_subinstances[
                        f'{outer_i}.{attr}']
                    _swpath = f'{outer_i}.{attr}'
                    _swint = _walk_user_class(_swc, _swb,
                                              _swpath) or []
                    _swmap = {}
                    for _swu in _swint:
                        if len(_swu) > 3 and \
                                isinstance(_swu[3], str):
                            _swat = _swu[3][len(_swpath) + 1:]
                            _swmap[_swat] = (_swu[1], _swu[3])
                    _swf = _walk_user_forward(_swc, _swb,
                                              _swpath, _swmap)
                    if _swf is not None:
                        out.extend(_swf)
                        continue
                    # sub-forward refused → the opaque stage:
                    out.append(('stage', _swc, None, _swpath))
                    continue
                if attr not in attr_map:
                    # RUNG-2: the bind-aware fallback — a
                    # conditionally-instantiated module (BasicLayer's
                    # self.downsample = downsample(...) inside an
                    # init-If — the simple init-walk missed it) whose
                    # BIND carries a class-name (≠ None → the module
                    # EXISTS at {outer_i}.{attr}, get_submodule-true):
                    # emit as an opaque module-stage (torch-executed,
                    # hook-captured; the internals = rung-3):
                    if attr in bind and isinstance(bind[attr], str):
                        out.append(('stage', bind[attr], None,
                                    f'{outer_i}.{attr}'))
                        continue
                    return None
                cls_, path_ = attr_map[attr]
                if cls_ in _UC_ACT_OPS:
                    out.append(_UC_ACT_OPS[cls_])
                else:
                    out.append(('stage', cls_, None, path_))
            return out

        specs = []
        _entry_ctr = [0]
        _thr = [None]   # RUNG-3c: the mini SHAPE-TRACKER (concrete
        #   dims where known; the shape-unpack binds concrete —
        #   the symbolic fallback resolves against the replay-time
        #   tensor = WRONG for mid-flow captures [the deferred-
        #   check trap, dodged]).

        def _thr_update(mspecs):
            for _ms in mspecs:
                if _ms[0] != 'stage':
                    _thr[0] = None
                    continue
                _mn = _ms[1]
                _ma = _ms[2] if len(_ms) > 2 else []
                if _mn in ('m_view', 'm_reshape'):
                    _thr[0] = [(a if isinstance(a, int) and
                                a > 0 else None)
                               for a in (_ma or [])]
                elif _mn == 'm_transpose' and \
                        _thr[0] is not None and \
                        len(_ma) == 2 and \
                        all(isinstance(a, int) for a in _ma):
                    _ns2 = list(_thr[0])
                    try:
                        _ns2[_ma[0]], _ns2[_ma[1]] = \
                            _ns2[_ma[1]], _ns2[_ma[0]]
                        _thr[0] = _ns2
                    except IndexError:
                        _thr[0] = None
                elif _mn == 'm_permute' and \
                        _thr[0] is not None and \
                        all(isinstance(a, int) for a in _ma):
                    try:
                        _thr[0] = [_thr[0][a] for a in _ma]
                    except IndexError:
                        _thr[0] = None
                elif _mn == 'm_contiguous':
                    pass
                else:
                    _thr[0] = None

        def _marg_ev(a):
            # RUNG-3a': resolve a method-chain arg from the bind
            # (concrete int · '__SIZEi__' symbolic · Constants ·
            # simple BinOps of bound values [H*W] · -1):
            if isinstance(a, ast.Constant):
                return a.value
            if isinstance(a, ast.UnaryOp) and \
                    isinstance(a.op, ast.USub) and \
                    isinstance(a.operand, ast.Constant):
                return -a.operand.value
            if isinstance(a, ast.Name) and a.id in bind:
                return bind[a.id]
            if isinstance(a, ast.Attribute) and \
                    isinstance(a.value, ast.Name) and \
                    a.value.id == 'self' and a.attr in bind:
                return bind[a.attr]
            if isinstance(a, ast.BinOp) and \
                    isinstance(a.op, (ast.Mult, ast.FloorDiv)):
                _l2 = _marg_ev(a.left)
                _r2 = _marg_ev(a.right)
                if isinstance(_l2, (int, float)) and \
                        isinstance(_r2, (int, float)):
                    return int(_l2 * _r2) \
                        if isinstance(a.op, ast.Mult) \
                        else int(_l2 // _r2)
                # RUNG-3c: the SYMBOLIC-EXPR form (C //
                # self.num_heads where C='__SIZE2__' → the
                # '__SIZE2_DIV3__' spelling the wrapper resolves):
                if isinstance(_l2, str) and \
                        (_l2.startswith('__SIZE') or
                         _l2.startswith('__ENTRY')) and \
                        _l2.endswith('__') and \
                        isinstance(_r2, (int, float)):
                    _op4 = 'DIV' if isinstance(a.op,
                                               ast.FloorDiv) \
                        else 'MUL'
                    return f'{_l2[:-2]}_{_op4}{int(_r2)}__'
            return None

        def _method_chain(e):
            # x.view(a,b).transpose(i,j)... → [('stage','m_view',
            # args), ('stage','m_transpose',args), ...] — the
            # thread-var method-chain (3a's lite gymnastics). Only
            # SAFE methods; unresolvable arg → None (honest).
            _safe = ('view', 'reshape', 'transpose', 'permute',
                     'flatten', 'contiguous', 'squeeze',
                     'unsqueeze')
            calls = []
            cur = e
            while isinstance(cur, ast.Call) and \
                    isinstance(cur.func, ast.Attribute) and \
                    cur.func.attr in _safe:
                calls.append(cur)
                cur = cur.func.value
            if not isinstance(cur, ast.Name):
                return None
            if not calls:
                return None
            out2 = []
            for cc in reversed(calls):
                _cargs = []
                for a in cc.args:
                    _av = _marg_ev(a)
                    if _av is None:
                        return None
                    _cargs.append(_av)
                out2.append(('stage', f'm_{cc.func.attr}', _cargs))
            return out2

        def _walk_free_fn(ffd, ffbind):
            # RUNG-3c: the free-function body walker (the bare-
            # function splice). The body is shape-math: tuple-
            # unpacks from x.shape (symbolic) + view/permute/
            # reshape chains on the local thread. Returns m_-stage
            # specs or None. The local bind: params (values or
            # '__thread__') + the shape-symbols.
            _lb = dict(ffbind)

            def _ff_arg(a):
                # resolve a view/permute arg in the free-fn scope:
                if isinstance(a, ast.Constant):
                    return a.value
                if isinstance(a, ast.UnaryOp) and \
                        isinstance(a.op, ast.USub) and \
                        isinstance(a.operand, ast.Constant):
                    return -a.operand.value
                if isinstance(a, ast.Name) and a.id in _lb:
                    return _lb[a.id]
                if isinstance(a, ast.BinOp) and \
                        isinstance(a.op, (ast.FloorDiv, ast.Mult)):
                    _l3 = _ff_arg(a.left)
                    _r3 = _ff_arg(a.right)
                    if isinstance(_l3, (int, float)) and \
                            isinstance(_r3, (int, float)):
                        return int(_l3 // _r3) if \
                            isinstance(a.op, ast.FloorDiv) \
                            else int(_l3 * _r3)
                    # the SYMBOLIC-EXPR form (the co-change with
                    # the wrapper: '__SIZE{i}_DIV{n}__' etc):
                    if isinstance(_l3, str) and \
                            (_l3.startswith('__SIZE') or
                             _l3.startswith('__ENTRY')) and \
                            _l3.endswith('__') and \
                            isinstance(_r3, (int, float)):
                        _base = _l3[:-2]
                        _op3 = 'DIV' if isinstance(a.op,
                                                   ast.FloorDiv) \
                            else 'MUL'
                        return f'{_base}_{_op3}{int(_r3)}__'
                    return None
                return None

            def _ff_scalar(e):
                # B = int(windows.shape[0] / (H*W/ws/ws)) — the
                # window_reverse head: int()-cast + shape-Subscript
                # + Div-arithmetic (symbolic-safe):
                if isinstance(e, ast.Call) and \
                        isinstance(e.func, ast.Name) and \
                        e.func.id == 'int' and len(e.args) == 1:
                    return _ff_scalar(e.args[0])
                if isinstance(e, ast.Subscript) and \
                        isinstance(e.value, ast.Attribute) and \
                        e.value.attr == 'shape' and \
                        isinstance(e.slice, ast.Constant):
                    return f'__SIZE{e.slice.value}__'
                if isinstance(e, ast.BinOp) and \
                        isinstance(e.op, (ast.Div, ast.FloorDiv,
                                          ast.Mult)):
                    _l4 = _ff_scalar(e.left)
                    _r4 = _ff_scalar(e.right)
                    if isinstance(_l4, (int, float)) and \
                            isinstance(_r4, (int, float)):
                        if isinstance(e.op, ast.Mult):
                            return _l4 * _r4
                        return _l4 / _r4 \
                            if isinstance(e.op, ast.Div) \
                            else _l4 // _r4
                    if isinstance(_l4, str) and \
                            (_l4.startswith('__SIZE') or
                             _l4.startswith('__ENTRY')) and \
                            isinstance(_r4, (int, float)):
                        _op5 = 'MUL' if isinstance(e.op, ast.Mult) \
                            else 'DIV'
                        return f'{_l4[:-2]}_{_op5}{int(_r4)}__'
                    return None
                if isinstance(e, ast.Name) and e.id in _lb:
                    return _lb[e.id]
                if isinstance(e, ast.Constant):
                    return e.value
                return None

            ffspecs = []
            for fst in ffd.body:
                if isinstance(fst, ast.Expr):
                    continue    # docstring
                # the scalar-assign head (B = int(...)):
                if isinstance(fst, ast.Assign) and \
                        len(fst.targets) == 1 and \
                        isinstance(fst.targets[0], ast.Name) and \
                        not (isinstance(fst.value, ast.Call) and
                             isinstance(fst.value.func,
                                        ast.Attribute)):
                    _sc5 = _ff_scalar(fst.value)
                    if _sc5 is not None:
                        _lb[fst.targets[0].id] = _sc5
                        continue
                if isinstance(fst, ast.Assign) and \
                        len(fst.targets) == 1 and \
                        isinstance(fst.targets[0], ast.Tuple) and \
                        isinstance(fst.value, ast.Attribute) and \
                        fst.value.attr == 'shape':
                    for _fti, _fte in enumerate(
                            fst.targets[0].elts):
                        if isinstance(_fte, ast.Name):
                            _lb[_fte.id] = f'__SIZE{_fti}__'
                    continue
                if isinstance(fst, ast.Assign) and \
                        len(fst.targets) == 1 and \
                        isinstance(fst.targets[0], ast.Name):
                    # the method-chain on the local thread:
                    _fcalls = []
                    _fcur = fst.value
                    _safe2 = ('view', 'reshape', 'transpose',
                              'permute', 'flatten', 'contiguous',
                              'squeeze', 'unsqueeze')
                    while isinstance(_fcur, ast.Call) and \
                            isinstance(_fcur.func,
                                       ast.Attribute) and \
                            _fcur.func.attr in _safe2:
                        _fcalls.append(_fcur)
                        _fcur = _fcur.func.value
                    if not isinstance(_fcur, ast.Name) or \
                            not _fcalls:
                        return None
                    for _fc in reversed(_fcalls):
                        _fargs = []
                        for a in _fc.args:
                            _fav2 = _ff_arg(a)
                            if _fav2 is None:
                                return None
                            _fargs.append(_fav2)
                        ffspecs.append(('stage',
                                        f'm_{_fc.func.attr}',
                                        _fargs))
                    continue
                if isinstance(fst, ast.Return):
                    if isinstance(fst.value, ast.Name):
                        return ffspecs
                    return None
                return None
            return ffspecs

        def _bind_test_ev(test):
            # RUNG-2: evaluate a forward-If test from the BIND env
            # (self.X → bind[X]). Handles: self.X (truthiness) ·
            # self.X is None / is not None · bare Compare of
            # bind-attrs. None = unevaluable.
            if isinstance(test, ast.Attribute) and \
                    isinstance(test.value, ast.Name) and \
                    test.value.id == 'self' and test.attr in bind:
                return bool(bind[test.attr])
            if isinstance(test, ast.Compare) and \
                    len(test.ops) == 1 and \
                    isinstance(test.comparators[0], ast.Constant) \
                    and test.comparators[0].value is None and \
                    isinstance(test.left, ast.Attribute) and \
                    isinstance(test.left.value, ast.Name) and \
                    test.left.value.id == 'self' and \
                    test.left.attr in bind:
                _bv = bind[test.left.attr]
                if isinstance(test.ops[0], ast.Is):
                    return _bv is None
                if isinstance(test.ops[0], ast.IsNot):
                    return _bv is not None
            # RUNG-3a': Compare with a NUMERIC Constant from the
            # bind (self.shift_size > 0 — the blocks alternate 0/3):
            if isinstance(test, ast.Compare) and \
                    len(test.ops) == 1 and \
                    isinstance(test.comparators[0], ast.Constant) \
                    and isinstance(test.comparators[0].value,
                                   (int, float)) and \
                    isinstance(test.left, ast.Attribute) and \
                    isinstance(test.left.value, ast.Name) and \
                    test.left.value.id == 'self' and \
                    test.left.attr in bind and \
                    isinstance(bind[test.left.attr], (int, float)):
                _bl = bind[test.left.attr]
                _bc = test.comparators[0].value
                _op = test.ops[0]
                if isinstance(_op, ast.Gt):
                    return _bl > _bc
                if isinstance(_op, ast.GtE):
                    return _bl >= _bc
                if isinstance(_op, ast.Lt):
                    return _bl < _bc
                if isinstance(_op, ast.LtE):
                    return _bl <= _bc
                if isinstance(_op, ast.Eq):
                    return _bl == _bc
                if isinstance(_op, ast.NotEq):
                    return _bl != _bc
            return None

        def _walk_stmts(body):
            # the rung-2 statement walker (returns False on refuse);
            # RUNG-3a' extends: tuple-unpacks (bind-attr + symbolic
            # shape) · assert-skip · residual save/add · thread-var
            # method-chains (m_-stages) · F.pad. The multi-var
            # window-flow (x_windows etc) = RUNG-3c (refuses at the
            # free-function call — the named movement).
            for st in body:
                if isinstance(st, ast.Expr):
                    continue    # docstring
                if isinstance(st, ast.Assert):
                    # asserts are runtime shape-checks, not dataflow —
                    # the honest skip (the wrap executes the real
                    # module; the assert fires there if ever):
                    continue
                if isinstance(st, ast.Assign) and \
                        len(st.targets) == 1 and \
                        isinstance(st.targets[0], ast.Tuple):
                    # H, W = self.input_resolution (the bind-tuple —
                    # concrete if the bind carries it) / B, L, C =
                    # x.shape (SYMBOLIC — carried as named opaques):
                    _tgt = st.targets[0]
                    if isinstance(st.value, ast.Attribute) and \
                            isinstance(st.value.value, ast.Name) and \
                            st.value.value.id == 'self' and \
                            st.value.attr in bind and \
                            isinstance(bind[st.value.attr],
                                       (tuple, list)):
                        _bt = bind[st.value.attr]
                        if len(_bt) == len(_tgt.elts):
                            for _te, _tv in zip(_tgt.elts, _bt):
                                if isinstance(_te, ast.Name):
                                    bind[_te.id] = _tv
                            continue
                    if isinstance(st.value, ast.Attribute) and \
                            st.value.attr == 'shape' and \
                            _thr[0] is not None and \
                            len(_thr[0]) == len(_tgt.elts):
                        for _ti, _te in enumerate(_tgt.elts):
                            if isinstance(_te, ast.Name):
                                _tv5 = _thr[0][_ti]
                                bind[_te.id] = _tv5 \
                                    if _tv5 is not None \
                                    else f'__SIZE{_ti}__'
                        continue
                    if isinstance(st.value, ast.Attribute) and \
                            st.value.attr == 'shape':
                        # symbolic dims — THE EXISTING SPELLING
                        # ('__SIZE{i}__' — the #34 m_view form; the
                        # wrapper already resolves it via
                        # src.size(int(a[6:-2]))): B,L,C = x.shape →
                        # B='__SIZE0__' etc (valid while the
                        # intervening ops are shape-preserving —
                        # true for 3a's scope [norm1]):
                        _ek = _entry_ctr[0]
                        _entry_ctr[0] += 1
                        specs.append(('stage', 'm_entry_mark',
                                      [_ek]))
                        for _ti, _te in enumerate(_tgt.elts):
                            if isinstance(_te, ast.Name):
                                bind[_te.id] = \
                                    f'__ENTRY{_ek}_SIZE{_ti}__'
                        continue
                    specs.append(('__refuse__',
                                  'tuple_assign_unrecognized'))
                    return False
                if isinstance(st, ast.If):
                    # THE EVALUABLE-FORWARD-IF (use_checkpoint →
                    # False → the else; the IsNot-downsample →
                    # per-bind True/False):
                    _tv2 = _bind_test_ev(st.test)
                    if _tv2 is None:
                        specs.append(('__refuse__',
                                      'forward_if_unevaluable'))
                        return False
                    _branch = st.body if _tv2 else st.orelse
                    if not _walk_stmts(_branch):
                        return False
                    continue
                if isinstance(st, ast.For) and \
                        isinstance(st.target, ast.Name) and \
                        isinstance(st.iter, ast.Attribute) and \
                        isinstance(st.iter.value, ast.Name) and \
                        st.iter.value.id == 'self':
                    # THE DEPTH-2 BLOCKS-LOOP: for blk in
                    # self.blocks — consult the class-level
                    # _modulelists (Iyun's HOP-2 populates it
                    # inside the class-walk; the seam key is
                    # '{outer_i}.{attr}'):
                    _blattr = st.iter.attr
                    _blelems = _modulelists.get(
                        f'{outer_i}.{_blattr}')
                    if _blelems is None:
                        specs.append((
                            '__refuse__',
                            f'modulelist_loop_depth2:'
                            f'{outer_i}.{_blattr}'))
                        return False
                    for _bli, (_blcls, _blbind) in \
                            enumerate(_blelems):
                        _blpath = f'{outer_i}.{_blattr}.{_bli}'
                        _blint = _walk_user_class(_blcls, _blbind,
                                                  _blpath)
                        if _blint is None:
                            specs.append((
                                '__refuse__',
                                f'modulelist_element:{_blpath}:'
                                f'{_blcls}'))
                            return False
                        _blmap = {}
                        for _blu in _blint:
                            _blp5 = _blu[3]
                            _blat = _blp5[len(_blpath) + 1:] \
                                if _blp5.startswith(_blpath) \
                                else _blp5
                            _blmap[_blat] = (_blu[1], _blp5)
                        _blf = _walk_user_forward(
                            _blcls, _blbind, _blpath, _blmap)
                        if _blf is None:
                            # the block's forward refused (the
                            # SwinMLPBlock gymnastics — rung-3):
                            # DEGRADE to the block's init-order
                            # stages (partial progress — the
                            # blocks still walk as modules; the
                            # internals stay torch-side). The
                            # honest form: BasicLayer's walk
                            # SURVIVES; the refusal is named at
                            # the block level via the composite
                            # report (not a walk-kill):
                            for _blu in _blint:
                                specs.append(('stage', _blu[3]))
                            continue
                        specs.extend(_blf)
                    continue
                if isinstance(st, ast.Assign) and \
                        len(st.targets) == 1 and \
                        isinstance(st.targets[0], ast.Name):
                    _tgt2 = st.targets[0].id
                    # RUNG-3a': NAME = x (the residual-save —
                    # shortcut = x):
                    if isinstance(st.value, ast.Name):
                        specs.append(('save',
                                      f'{outer_i}::{_tgt2}'
                                      if outer_i else _tgt2))
                        continue
                    # RUNG-3a': x = NAME + <chain> or x = <chain> +
                    # NAME (the residual-add — shortcut +
                    # drop_path(x)):
                    if isinstance(st.value, ast.BinOp) and \
                            isinstance(st.value.op, ast.Add):
                        _lh, _rh = st.value.left, st.value.right
                        _sv = None
                        _ch = None
                        if isinstance(_lh, ast.Name):
                            _sv, _ch = _lh.id, _rh
                        elif isinstance(_rh, ast.Name):
                            _sv, _ch = _rh.id, _lh
                        if _sv is not None:
                            ch3 = _app_chain(_ch)
                            if ch3 is not None:
                                specs.extend(ch3)
                                # the ANCHOR for the saved name:
                                # the chain's first module-stage
                                # (norm2) — VALID ONLY when the
                                # SAVED VAR ITSELF feeds the
                                # chain (x + f(g(x)): the inner
                                # arg == the Name-side → norm2's
                                # INPUT is exactly the saved
                                # value at add-time). For
                                # shortcut + f(x) the inner arg
                                # (x) ≠ shortcut → NO add-anchor
                                # (the save-op's anchor stands):
                                _anch = None
                                _inner8 = _ch
                                while isinstance(_inner8,
                                                 ast.Call) and \
                                        _inner8.args:
                                    _inner8 = _inner8.args[0]
                                if isinstance(_inner8,
                                              ast.Name) and \
                                        _inner8.id == _sv:
                                    for _c9 in ch3:
                                        if _c9[0] == 'stage' and \
                                                len(_c9) > 3 and \
                                                isinstance(
                                                    _c9[3], str):
                                            _anch = _c9[3]
                                            break
                                specs.append((
                                    'add_saved',
                                    f'{outer_i}::{_sv}'
                                    if outer_i else _sv,
                                    _anch))
                                continue
                        specs.append(('__refuse__',
                                      'binop_unrecognized'))
                        return False
                    # STEP-4a (the #30 pairing prereq): THE
                    # ROLL-ARM — shifted_x = torch.roll(x,
                    # shifts=(±s, ±s), dims=(1, 2)): a
                    # torch-executed m_-stage (the wrapper
                    # replays torch.roll on the thread; the
                    # shift-values resolve via _marg_ev
                    # [self.shift_size / negated / literal]):
                    if isinstance(st.value, ast.Call) and \
                            isinstance(st.value.func,
                                       ast.Attribute) and \
                            isinstance(st.value.func.value,
                                       ast.Name) and \
                            st.value.func.value.id == 'torch' \
                            and st.value.func.attr == 'roll' \
                            and st.value.args and \
                            isinstance(st.value.args[0],
                                       ast.Name):
                        _rl_ok = True
                        _rl_shifts = None
                        _rl_dims = None
                        for _rkw in st.value.keywords:
                            _rvals = []
                            _rsrc = _rkw.value
                            _rlist = (_rsrc.elts if isinstance(
                                _rsrc, ast.Tuple) else [_rsrc])
                            for _re in _rlist:
                                # the negated-attr form
                                # (-self.shift_size):
                                if isinstance(_re,
                                              ast.UnaryOp) and \
                                        isinstance(_re.op,
                                                   ast.USub):
                                    _rin = _marg_ev(_re.operand)
                                    _rv9 = (-_rin if isinstance(
                                        _rin, (int, float))
                                        else None)
                                else:
                                    _rv9 = _marg_ev(_re)
                                if _rv9 is None or not \
                                        isinstance(_rv9,
                                                   (int, float)):
                                    _rl_ok = False
                                    break
                                _rvals.append(int(_rv9))
                            if not _rl_ok:
                                break
                            if _rkw.arg == 'shifts':
                                _rl_shifts = _rvals
                            elif _rkw.arg == 'dims':
                                _rl_dims = _rvals
                        if _rl_ok and _rl_shifts is not None \
                                and _rl_dims is not None:
                            specs.append((
                                'stage', 'm_roll',
                                [_rl_shifts, _rl_dims]))
                            continue
                        specs.append(('__refuse__',
                                      'roll_args'))
                        return False
                    # RUNG-3c: THE FREE-FUNCTION SPLICE —
                    # x_windows = window_partition(shifted_x,
                    # self.window_size): a Call to a module-level
                    # def → bind call-args to def-params → splice
                    # the body through _walk_stmts RECURSIVELY
                    # (the local bind carries the params; the
                    # symbolic dims + method-chains carry the
                    # body's shape-math); the return threads out:
                    if isinstance(st.value, ast.Call) and \
                            isinstance(st.value.func, ast.Name) \
                            and st.value.func.id in \
                            _free_functions:
                        _ffd = _free_functions[st.value.func.id]
                        _ffparams = [a.arg for a in
                                     _ffd.args.args]
                        _ffbind = {}
                        _ff_ok = True
                        for _fi, _fa in enumerate(st.value.args):
                            if _fi >= len(_ffparams):
                                _ff_ok = False
                                break
                            _fpn = _ffparams[_fi]
                            if isinstance(_fa, ast.Name) and \
                                    _fa.id in bind:
                                # a bind-valued Name (H, W — the
                                # window_reverse args):
                                _ffbind[_fpn] = bind[_fa.id]
                            elif isinstance(_fa, ast.Name):
                                # the thread (the tensor arg):
                                _ffbind[_fpn] = '__thread__'
                            else:
                                _fav = _marg_ev(_fa)
                                if _fav is None:
                                    _ff_ok = False
                                    break
                                _ffbind[_fpn] = _fav
                        if _ff_ok:
                            _ffspecs = _walk_free_fn(_ffd, _ffbind)
                            if _ffspecs is not None:
                                # the ENTRY-MARK: the wrapper
                                # snapshots the thread here;
                                # __ENTRY_SIZEi__ resolves
                                # against the SNAPSHOT (the
                                # fn-entry tensor), not the
                                # rolling thread — the
                                # deferred-check trap's
                                # free-fn twin:
                                _ek2 = _entry_ctr[0]
                                _entry_ctr[0] += 1
                                specs.append(('stage',
                                              'm_entry_mark',
                                              [_ek2]))
                                # anchor the fn's shape-symbols
                                # to the ENTRY snapshot (post-
                                # hoc rewrite — only this
                                # splice's specs carry ENTRY):
                                for _fs9 in _ffspecs:
                                    if _fs9[0] == 'stage' and \
                                            len(_fs9) > 2 and \
                                            isinstance(_fs9[2],
                                                       list):
                                        _fs9[2][:] = [
                                            (a.replace(
                                                '__SIZE',
                                                f'__ENTRY{_ek2}'
                                                '_SIZE')
                                             if isinstance(
                                                 a, str) and
                                             a.startswith(
                                                 '__SIZE')
                                             else a)
                                            for a in _fs9[2]]
                                specs.extend(_ffspecs)
                                _thr_update(_ffspecs)
                                continue
                        specs.append(('__refuse__',
                                      f'free_fn:'
                                      f'{st.value.func.id}'))
                        return False
                    # RUNG-3c: the SLICE-CROP (the un-pad:
                    # shifted_x[:, P_t:-P_b, P_l:-P_r, :]
                    # [.contiguous()]) → m_narrow chains (narrow IS
                    # a replayable tensor-method; the lengths
                    # compute CONCRETE via the shape-tracker):
                    _slval = st.value
                    _had_contig = False
                    if isinstance(_slval, ast.Call) and \
                            isinstance(_slval.func,
                                       ast.Attribute) and \
                            _slval.func.attr == 'contiguous' and \
                            not _slval.args:
                        _had_contig = True
                        _slval = _slval.func.value
                    if isinstance(_slval, ast.Subscript) and \
                            isinstance(_slval.value, ast.Name) \
                            and isinstance(_slval.slice,
                                           ast.Tuple):
                        _sl_ok = True
                        _narrows = []
                        for _sdi, _sel in enumerate(
                                _slval.slice.elts):
                            if not isinstance(_sel, ast.Slice):
                                _sl_ok = False
                                break
                            if _sel.lower is None and \
                                    _sel.upper is None:
                                continue    # ':' — full dim
                            _lo = 0 if _sel.lower is None \
                                else _marg_ev(_sel.lower)
                            _hi = _marg_ev(_sel.upper) \
                                if _sel.upper is not None \
                                else None
                            _dimlen = (_thr[0][_sdi]
                                       if _thr[0] is not None and
                                       _sdi < len(_thr[0])
                                       else None)
                            if not isinstance(_lo, int) or \
                                    _dimlen is None:
                                _sl_ok = False
                                break
                            if _hi is None:
                                _ln = _dimlen - _lo
                            elif isinstance(_hi, int):
                                _ln = (_dimlen + _hi - _lo) \
                                    if _hi < 0 else (_hi - _lo)
                            else:
                                _sl_ok = False
                                break
                            _narrows.append((_sdi, _lo, _ln))
                        if _sl_ok:
                            for _nd, _nl, _nn in _narrows:
                                specs.append(('stage', 'm_narrow',
                                              [_nd, _nl, _nn]))
                            if _had_contig:
                                specs.append(('stage',
                                              'm_contiguous', []))
                            if _thr[0] is not None:
                                _ns6 = list(_thr[0])
                                for _nd, _nl, _nn in _narrows:
                                    if _nd < len(_ns6):
                                        _ns6[_nd] = _nn
                                _thr[0] = _ns6
                            continue
                    # RUNG-3c: F.pad(x, [pads], 'constant', 0) →
                    # the F_pad stage (the shift-branch; the pad
                    # values resolve from the bind [P_l etc — the
                    # padding tuple-unpack]):
                    if isinstance(st.value, ast.Call) and \
                            isinstance(st.value.func,
                                       ast.Attribute) and \
                            isinstance(st.value.func.value,
                                       ast.Name) and \
                            st.value.func.value.id == 'F' and \
                            st.value.func.attr == 'pad' and \
                            len(st.value.args) >= 2:
                        _pdargs = []
                        _pd_ok = True
                        _pdl = st.value.args[1]
                        if isinstance(_pdl, (ast.List, ast.Tuple)):
                            for _pde in _pdl.elts:
                                _pdv = _marg_ev(_pde)
                                if _pdv is None or \
                                        isinstance(_pdv, str):
                                    _pd_ok = False
                                    break
                                _pdargs.append(_pdv)
                        else:
                            _pd_ok = False
                        if _pd_ok:
                            specs.append(('stage', 'F_pad',
                                          [_pdargs]))
                            if _thr[0] is not None:
                                _ns = list(_thr[0])
                                _pi5 = 0
                                for _di in range(len(_ns) - 1,
                                                 -1, -1):
                                    if _pi5 + 1 < len(_pdargs):
                                        if _ns[_di] is not None:
                                            _ns[_di] += \
                                                _pdargs[_pi5] + \
                                                _pdargs[_pi5 + 1]
                                        _pi5 += 2
                                _thr[0] = _ns
                            continue
                        specs.append(('__refuse__', 'f_pad_args'))
                        return False
                    # RUNG-3a': thread method-chains (x.view(...).
                    # transpose(...) → m_-stages with (symbolic)
                    # args):
                    _mch = _method_chain(st.value)
                    if _mch is not None:
                        specs.extend(_mch)
                        _thr_update(_mch)
                        continue
                    ch2 = _app_chain(st.value)
                    if ch2 is None:
                        # a self.CUSTOM(x) application (the
                        # downsample=PatchMerging branch) — the
                        # rung-3 door, named:
                        specs.append(('__refuse__',
                                      'user_class_in_forward'))
                        return False
                    specs.extend(ch2)
                    _thr[0] = None    # module app: shape unknown
                    continue
                if isinstance(st, ast.Return):
                    if not _walk_return(st):
                        return False
                    continue
                specs.append(('__refuse__', 'forward_stmt_kind'))
                return False
            return True

        def _walk_return(st):
            # the Return-arm (rung-1's cat-form, extracted):
            if st.value is None:
                return True
            rv = st.value
            if isinstance(rv, ast.Name):
                return True    # return x — done
            if isinstance(rv, ast.Call) and \
                    isinstance(rv.func, ast.Attribute) and \
                    isinstance(rv.func.value, ast.Name) and \
                    rv.func.value.id == 'torch' and \
                    rv.func.attr == 'cat' and rv.args and \
                    isinstance(rv.args[0], ast.List):
                rdim = None
                if len(rv.args) == 2 and \
                        isinstance(rv.args[1], ast.Constant):
                    rdim = rv.args[1].value
                for kw in rv.keywords:
                    if kw.arg == 'dim' and \
                            isinstance(kw.value, ast.Constant):
                        rdim = kw.value.value
                if rdim is None:
                    return False
                n_br = len(rv.args[0].elts)
                specs.append(('branch_fanout_begin', n_br, rdim))
                for bi, bel in enumerate(rv.args[0].elts):
                    bch = _app_chain(bel)
                    if bch is None:
                        return False
                    specs.append(('branch_begin', bi))
                    specs.extend(bch)
                    specs.append(('branch_end', bi))
                specs.append(('stage', 'm_cat2', [rdim]))
                specs.append(('branch_fanout_end',))
                return True
            return False    # other return-shapes: refuse

        if not _walk_stmts(_fwd.body):
            _rf = [s for s in specs if s[0] == '__refuse__']
            return None
        # RUNG-3d GATE: attach the ANCHOR to each save. THE
        # ANCHOR-VALIDITY RULE (the #30 x2-pairing root — the
        # residue-11's mechanism): cap_in[next-module] equals the
        # saved value ONLY if the thread is UNCHANGED between the
        # save and that module — any intervening m_/F_ stage
        # (view/permute/roll reorder the tensor) breaks the
        # equality (the partitioned-tensor-as-x2 defect; clean
        # only where the permute was degenerate, H/ws=1). THE
        # FIX: prefer the PRECEDING module-stage's OUTPUT
        # (cap_out via the 'x_' anchor-form the gate already
        # supports) when any m_/F_ stage intervenes forward;
        # forward cap_in only when the path is stage-clean:
        for _si7, _sp7 in enumerate(specs):
            if _sp7[0] == 'save' and len(_sp7) == 2:
                _fwd_dirty = False
                _fwd_anchor = None
                for _sp8 in specs[_si7 + 1:]:
                    if _sp8[0] == 'stage' and \
                            str(_sp8[1]) in (
                                'm_permute', 'm_roll',
                                'm_transpose', 'F_pad'):
                        # REORDERING stages break cap_in≡saved
                        # (views/contiguous preserve element-
                        # order — elementwise-add-safe):
                        _fwd_dirty = True
                        continue
                    if _sp8[0] == 'stage' and len(_sp8) > 3 and \
                            isinstance(_sp8[3], str):
                        _fwd_anchor = _sp8[3]
                        break
                if _fwd_anchor is not None and not _fwd_dirty:
                    specs[_si7] = ('save', _sp7[1], _fwd_anchor)
                    continue
                # backward: the previous module-stage's OUTPUT
                # (the saved var was assigned from the clean
                # thread right after it):
                _bwd_anchor = None
                for _sp9 in reversed(specs[:_si7]):
                    if _sp9[0] == 'stage' and \
                            str(_sp9[1]) in (
                                'm_permute', 'm_roll',
                                'm_transpose', 'F_pad'):
                        break
                    if _sp9[0] == 'stage' and len(_sp9) > 3 and \
                            isinstance(_sp9[3], str):
                        _bwd_anchor = _sp9[3]
                        break
                if _bwd_anchor is not None:
                    specs[_si7] = ('save', _sp7[1],
                                   'x_' + _bwd_anchor)
                    continue
                # block-entry save (backward exhausted at the
                # walk-start): the BLOCK module's own input IS
                # the saved value (cap_in[block-path] — V2
                # post-norm: shortcut = x at block entry):
                _blkpath = _sp7[1].split('::')[0]
                if _blkpath and '.' in _blkpath:
                    specs[_si7] = ('save', _sp7[1], _blkpath)
                elif _fwd_anchor is not None:
                    # neither side clean: keep the old forward
                    # anchor (honest degradation — the gate
                    # fires and MEASURES the mismatch):
                    specs[_si7] = ('save', _sp7[1], _fwd_anchor)
        return specs

    _in_user_class = [0]   # RUNG-2: the recursion-DEPTH counter
    #   (0 = top; 1 = inside a user class; 2 = inside a nested one).
    #   Depth-2 ALLOWED (BasicLayer→SwinMLPBlock); depth-3 refuses
    #   BY NAME ('nested_custom_module_depth3' — SwinMLPBlock→Mlp =
    #   rung-3's boundary, hit and named).
    _pev_shared = [None]   # RUNG-3c: the populate's evaluator,
    #   exported for the class-walk's init-attr eval (the nested-
    #   def scoping fix — the silent-except had eaten the
    #   NameError; the file-trace method found it).
    _walked_subinstances = {}  # RUNG-3d: path -> (cname, sub_bind)
    #   for a WALKED sub-instance (Mlp); the forward-arm looks it
    #   up by '{outer_i}.mlp' to splice _walk_user_forward.
    _class_envs = {}    # RUNG-3c: outer_i -> the class-walk's
    #   computed _cenv (init-computed attrs like SwinMLPBlock's
    #   self.padding — the forward-walk's bind merges these).
    _modulelists = {}   # attr-name -> [(class_name, bind_dict), ...]
    #                     (RUNG-1: the ModuleList-loop seam — the
    #                      init side FILLS this [Iyun's half: the
    #                      append-loop bind via _ev_scalar with the
    #                      loop env]; the forward side CONSUMES it
    #                      [the loop-arm below]. The stub recognizer
    #                      in this hop handles the DIRECT form only:
    #                      nn.ModuleList([UserClass(...), ...]).)
    _free_functions = {}   # RUNG-3c: module-level FunctionDefs
    #   (window_partition/window_reverse — the bare-function
    #    splice targets). name -> FunctionDef.
    _user_classes = {}  # class-name -> {'init': FunctionDef,
    #                     'forward': FunctionDef, 'params': [names]}
    #                     (the #18 custom-class-recursion rung: every
    #                     file-defined nn.Module class, not just Model)
    for node in tree.body:
        # RUNG-3c: top-level (module-level) function defs — the
        # free-function registry (only DIRECT children of the
        # module; nested defs are locals):
        if isinstance(node, ast.FunctionDef):
            _free_functions[node.name] = node
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name != 'Model':
            _uc_init = None
            _uc_fwd = None
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    if item.name == '__init__':
                        _uc_init = item
                    elif item.name == 'forward':
                        _uc_fwd = item
            if _uc_fwd is not None:
                _uc_params = ([a.arg for a in _uc_init.args.args
                               if a.arg != 'self']
                              if _uc_init else [])
                _user_classes[node.name] = {
                    'init': _uc_init, 'forward': _uc_fwd,
                    'params': _uc_params}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'Model':
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    if item.name == 'forward':
                        fwd = item
                    elif item.name == '__init__':
                        init = item
                    else:
                        _cls_methods[item.name] = item
    if fwd is None:
        raise ValueError('no Model.forward')
    # module-level constants (divisor = 2.0 etc):
    def _const_eval(e):
        """Fold constant expressions: Constant, -Constant, BinOp of
        constants (16*5*5), Name in consts. None if unresolvable."""
        if isinstance(e, ast.Constant):
            return e.value
        if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub):
            v2 = _const_eval(e.operand)
            return -v2 if v2 is not None else None
        if isinstance(e, ast.BinOp):
            l2, r2 = _const_eval(e.left), _const_eval(e.right)
            if l2 is None or r2 is None:
                return None
            try:
                return {ast.Mult: lambda a, b: a * b,
                        ast.Add: lambda a, b: a + b,
                        ast.Sub: lambda a, b: a - b,
                        ast.FloorDiv: lambda a, b: a // b}[
                    type(e.op)](l2, r2)
            except Exception:
                return None
        if isinstance(e, ast.Name):
            return consts.get(e.id)
        return None

    consts = {}

    def resolve_const(e):
        """literal_eval extended: resolves Names via consts (in order),
        chained assigns (a = b = 1), and tuples of both."""
        if isinstance(e, ast.Constant):
            return e.value
        if isinstance(e, ast.Name) and e.id in consts:
            return consts[e.id]
        if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub):
            v = resolve_const(e.operand)
            return -v if v is not None else None
        if isinstance(e, ast.BinOp):
            # constant arithmetic incl [1024]*16 (the #3 form):
            l4, r4 = resolve_const(e.left), resolve_const(e.right)
            if l4 is None or r4 is None:
                return None
            try:
                return {ast.Mult: lambda a, b: a * b,
                        ast.Add: lambda a, b: a + b,
                        ast.Sub: lambda a, b: a - b,
                        ast.FloorDiv: lambda a, b: a // b}[
                    type(e.op)](l4, r4)
            except Exception:
                return None
        if isinstance(e, (ast.Tuple, ast.List)):
            elems = []
            for el in e.elts:
                r = resolve_const(el)
                if r is None:
                    return None
                elems.append(r)
            return tuple(elems)
        return None

    for node in tree.body:
        if isinstance(node, ast.Assign):
            val = resolve_const(node.value)
            if val is None:
                continue
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    consts[tgt.id] = val
                elif isinstance(tgt, ast.Tuple) and \
                     isinstance(val, tuple) and \
                     len(tgt.elts) == len(val):
                    # height, width = 384, 384 (#65's form):
                    for el, vv in zip(tgt.elts, val):
                        if isinstance(el, ast.Name):
                            consts[el.id] = vv
    # resolve __init__ parameter values: signature defaults overridden by
    # get_init_inputs() positional mapping (the KernelBench harness
    # convention — proven from source, not assumed):
    init_args = {}
    if init:
        params = [a.arg for a in init.args.args[1:]]   # skip self
        defaults = init.args.defaults
        # defaults align to the TAIL of params:
        for p, dflt in zip(params[len(params)-len(defaults):], defaults):
            try:
                init_args[p] = ast.literal_eval(dflt)
            except Exception:
                pass
        # get_init_inputs(): positional values (Names resolved via consts):
        gii = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and \
               node.name == 'get_init_inputs':
                gii = node
        if gii:
            for st in gii.body:
                if isinstance(st, ast.Return) and \
                   isinstance(st.value, ast.List):
                    for p, el in zip(params, st.value.elts):
                        if isinstance(el, ast.Constant):
                            init_args[p] = el.value
                        elif isinstance(el, ast.Name) and el.id in consts:
                            init_args[p] = consts[el.id]
    # make init-args visible to scalar_of via consts (params shadow module
    # consts inside __init__ scope):
    for k, v2 in init_args.items():
        if k not in consts:
            consts[k] = v2

    # __init__ self.X = value bindings (scalars only) + nn-module ops:
    attrs = {}
    _flag_exprs = {}
    scalar_params = set()   # self.X params that are provably scalar-shaped
    channel_params = set()  # (C,1,1)-shaped params: channel-broadcast
    stage_classes = {}      # self.X -> nn module class name (Linear/Conv2d/..)
    stage_widths = {}       # self.X -> output width (Linear out_features) if provable
    stage_chans = {}        # self.X -> out_channels (Conv*) if provable
    stage_spatial = {}      # self.X -> (kind, k, s, p) spatial transform
    module_red_dims = {}    # self.X -> dim arg for nn.Softmax(dim=..)
    STAGE_RANKS = {'Linear': 2, 'Gemm': 2, 'Bilinear': 2, 'Identity': None,
                   'Conv1d': 3, 'ConvTranspose1d': 3, 'BMM': 3,
                   'Conv2d': 4, 'ConvTranspose2d': 4,
                   'Conv3d': 5, 'ConvTranspose3d': 5,
                   # rank-PRESERVING stages (the #55/#98 root — absent
                   # classes nulled the rank at the stage; mavhir's
                   # trace + Mavdil's probes):
                   'MaxPool1d': 3, 'AvgPool1d': 3,
                   'MaxPool2d': 4, 'AvgPool2d': 4,
                   'MaxPool3d': 5, 'AvgPool3d': 5}
    module_ops = {}   # self.X -> op tuple, for nn activation modules
    param_shapes = {}  # self.X -> resolved dim list (the #14 width)
    NN_OP_MODULES = {
        'ReLU': ('relu',), 'Tanh': ('tanh',), 'Sigmoid': ('sigmoid',),
        'GELU': ('gelu',), 'Mish': ('mish',), 'Hardswish': ('hardswish',),
        'Hardtanh': ('hardtanh',), 'Softmax': ('reduction', 'softmax'),
        'LogSoftmax': ('reduction', 'log_softmax'),
        'ReLU6': ('clamp', 0.0, 6.0),
    }
    if init:
        for node in ast.walk(init):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                t = node.targets[0]
                if isinstance(t, ast.Attribute) and \
                   isinstance(t.value, ast.Name) and t.value.id == 'self':
                    v = node.value
                    if isinstance(v, ast.Call):
                        fn = v.func
                        cls = fn.attr if isinstance(fn, ast.Attribute) else \
                              (fn.id if isinstance(fn, ast.Name) else None)
                        # nn.Parameter(torch.randn(1,1,..)) / (torch.tensor(x))
                        # = SCALAR param (all dims literal 1, or 0-d tensor):
                        if cls == 'Parameter' and v.args:
                            inner = v.args[0]
                            # track the param's SHAPE (dims resolved
                            # via module-level consts — the #14
                            # matmul width):
                            if isinstance(inner, ast.Call):
                                _ifn = inner.func
                                _in2 = (_ifn.attr if isinstance(
                                    _ifn, ast.Attribute) else None)
                                if _in2 in ('randn', 'zeros', 'ones'):
                                    _dims2 = []
                                    for a in inner.args:
                                        if isinstance(a, ast.Constant):
                                            _dims2.append(a.value)
                                        elif isinstance(a, ast.Name) \
                                                and a.id in consts:
                                            _dims2.append(
                                                consts[a.id])
                                        else:
                                            _dims2 = None
                                            break
                                    if _dims2:
                                        param_shapes[t.attr] = _dims2
                            if isinstance(inner, ast.Call):
                                ifn = inner.func
                                iname = ifn.attr if isinstance(ifn, ast.Attribute) else None
                                if iname == 'tensor':
                                    scalar_params.add(t.attr)
                                elif iname in ('randn', 'zeros', 'ones'):
                                    dims = inner.args
                                    if dims and all(
                                        isinstance(a, ast.Constant) and a.value == 1
                                        for a in dims):
                                        scalar_params.add(t.attr)
                                    elif len(dims) == 1:
                                        shp = None
                                        if isinstance(dims[0], ast.Name) and \
                                           dims[0].id in consts:
                                            shp = consts[dims[0].id]
                                        elif isinstance(dims[0], ast.Tuple):
                                            elems = []
                                            for el in dims[0].elts:
                                                if isinstance(el, ast.Constant):
                                                    elems.append(el.value)
                                                elif isinstance(el, ast.Name) and \
                                                     el.id in consts:
                                                    elems.append(consts[el.id])
                                                else:
                                                    elems = None
                                                    break
                                            shp = tuple(elems) if elems else None
                                        if isinstance(shp, (tuple, list)):
                                            if all(d == 1 for d in shp):
                                                scalar_params.add(t.attr)
                                            elif shp[0] > 1 and \
                                                 all(d == 1 for d in shp[1:]):
                                                # (C,1,1)-style: CHANNEL param
                                                channel_params.add(t.attr)
                        if cls == 'LeakyReLU':
                            # Resolve the slope from a Constant OR a Name (an
                            # __init__ param reference against module consts).
                            # Doresh caught (#12): the old handler only read
                            # ast.Constant, so nn.LeakyReLU(negative_slope) with
                            # a Name arg silently fell through to the DEFAULT
                            # 0.01 instead of the configured 0.1 -> wrong
                            # hyperparameter, confidently wrong. Resolve Names
                            # via consts; if UNresolved, FLAG (never silently
                            # default).
                            def _slope_of(node):
                                if isinstance(node, ast.Constant):
                                    return float(node.value)
                                if isinstance(node, ast.Name) and node.id in consts:
                                    return float(consts[node.id])
                                return None  # unresolved -> flag, don't default
                            slope = None
                            src = None
                            if v.args:
                                slope = _slope_of(v.args[0]); src = v.args[0]
                            for kw in v.keywords:
                                if kw.arg == 'negative_slope':
                                    slope = _slope_of(kw.value); src = kw.value
                            if slope is None and src is not None:
                                # a slope was GIVEN but couldn't be resolved:
                                # flag it, don't silently use the default.
                                module_ops[t.attr] = (
                                    'unknown_call', 'leaky_relu_unresolved_slope')
                                continue
                            if slope is None:
                                slope = 0.01  # no arg given: PyTorch default IS 0.01
                            module_ops[t.attr] = ('leaky_relu', slope)
                            continue
                        if cls == 'Hardtanh' and (v.keywords or v.args):
                            # CUSTOM bounds (the #30 catch: nn.Hardtanh
                            # (min_val=m, max_val=M) ≠ the default
                            # (-1,1) clamp — resolve like leaky_relu's
                            # slope; flag-don't-default on unresolved):
                            lo, hi = -1.0, 1.0
                            ok = True
                            vals = {}
                            for kw in v.keywords:
                                if kw.arg in ('min_val', 'max_val'):
                                    try:
                                        vals[kw.arg] = ast.literal_eval(
                                            kw.value)
                                    except Exception:
                                        if isinstance(kw.value, ast.Name) \
                                           and kw.value.id in init_args:
                                            vals[kw.arg] = \
                                                init_args[kw.value.id]
                                        else:
                                            ok = False
                            for j, a in enumerate(v.args[:2]):
                                key = ('min_val', 'max_val')[j]
                                try:
                                    vals[key] = ast.literal_eval(a)
                                except Exception:
                                    if isinstance(a, ast.Name) and \
                                       a.id in init_args:
                                        vals[key] = init_args[a.id]
                                    else:
                                        ok = False
                            if not ok:
                                module_ops[t.attr] = (
                                    'unknown_call',
                                    'hardtanh_unresolved_bounds')
                                continue
                            lo = float(vals.get('min_val', lo))
                            hi = float(vals.get('max_val', hi))
                            module_ops[t.attr] = ('clamp', lo, hi)
                            continue
                        # AvgPool3d(kernel_size=2) → a LIFTABLE op
                        # (#72; the #49/#84 REGRESSION LESSON: this
                        # elif once severed the Softmax dim-tracking
                        # from its NN_OP_MODULES branch — order
                        # matters in this cascade):
                        if cls == 'AvgPool3d':
                            _k = None
                            if v.args and isinstance(v.args[0],
                                                     ast.Constant):
                                _k = v.args[0].value
                            for kw in v.keywords:
                                if kw.arg == 'kernel_size' and \
                                        isinstance(kw.value,
                                                   ast.Constant):
                                    _k = kw.value.value
                            if _k == 2 and not any(
                                    kw.arg in ('stride', 'padding')
                                    for kw in v.keywords):
                                module_ops[t.attr] = ('avgpool3d_k2',)
                        if cls in NN_OP_MODULES:
                            module_ops[t.attr] = NN_OP_MODULES[cls]
                            if cls in ('Softmax', 'LogSoftmax'):
                                d = None
                                for kw in v.keywords:
                                    if kw.arg == 'dim' and \
                                       isinstance(kw.value, ast.Constant):
                                        d = kw.value.value
                                if d is None and v.args and \
                                   isinstance(v.args[0], ast.Constant):
                                    d = v.args[0].value
                                module_red_dims[t.attr] = d
                            continue
                        if cls is not None:
                            stage_classes[t.attr] = cls
                            if cls in ('AdaptiveAvgPool1d',
                                       'AdaptiveAvgPool2d',
                                       'AdaptiveAvgPool3d',
                                       'AdaptiveMaxPool1d',
                                       'AdaptiveMaxPool2d',
                                       'AdaptiveMaxPool3d') and v.args:
                                try:
                                    osz = ast.literal_eval(v.args[0])
                                    if isinstance(osz, int):
                                        osz = (osz,)
                                    if isinstance(osz, (tuple, list)) and \
                                       all(isinstance(z, int) for z in osz):
                                        stage_spatial[t.attr] = \
                                            ('adaptive', tuple(osz), 0, 0,
                                             0)
                                except Exception:
                                    pass
                            if cls in ('Conv1d', 'Conv2d', 'Conv3d',
                                       'ConvTranspose1d', 'ConvTranspose2d',
                                       'ConvTranspose3d',
                                       'MaxPool1d', 'MaxPool2d', 'MaxPool3d',
                                       'AvgPool1d', 'AvgPool2d', 'AvgPool3d'):
                                # spatial transform params (k, stride, pad):
                                def _cv(node, dflt=None):
                                    if node is None:
                                        return dflt
                                    try:
                                        val = ast.literal_eval(node)
                                    except Exception:
                                        if isinstance(node, ast.Name) and \
                                           node.id in consts:
                                            val = consts[node.id]
                                        else:
                                            return None
                                    return val
                                kw = {k.arg: k.value for k in v.keywords}
                                if cls.startswith(('Conv', 'ConvT')):
                                    kk = _cv(v.args[2] if len(v.args) > 2
                                             else kw.get('kernel_size'))
                                    ss = _cv(v.args[3] if len(v.args) > 3
                                             else kw.get('stride'), 1)
                                    pp = _cv(v.args[4] if len(v.args) > 4
                                             else kw.get('padding'), 0)
                                    # output_padding (the #44 catch:
                                    # convT out = (d-1)s-2p+k+OP —
                                    # missing OP made 255² of a 256²):
                                    op = _cv(v.args[5] if len(v.args) > 5
                                             else kw.get('output_padding'),
                                             0)
                                    kind = ('convT' if 'Transpose' in cls
                                            else 'conv')
                                else:
                                    kk = _cv(v.args[0] if len(v.args) > 0
                                             else kw.get('kernel_size'))
                                    ss = _cv(v.args[1] if len(v.args) > 1
                                             else kw.get('stride'), None)
                                    if ss is None:
                                        ss = kk  # pool default stride = k
                                    pp = _cv(kw.get('padding'), 0)
                                    op = 0
                                    kind = 'pool'
                                def _ok(z):
                                    return isinstance(z, int) or (
                                        isinstance(z, (tuple, list)) and
                                        all(isinstance(w, int) for w in z))
                                if kind != 'convT':
                                    op = 0
                                if all(_ok(z) for z in (kk, ss, pp)) and \
                                        _ok(op):
                                    stage_spatial[t.attr] = (kind, kk, ss,
                                                             pp, op)
                            if cls in ('Conv1d', 'Conv2d', 'Conv3d',
                                       'ConvTranspose1d', 'ConvTranspose2d',
                                       'ConvTranspose3d') and len(v.args) >= 2:
                                w = None
                                a1 = v.args[1]
                                if isinstance(a1, ast.Constant):
                                    w = a1.value
                                elif isinstance(a1, ast.Name) and a1.id in consts:
                                    w = consts[a1.id]
                                if isinstance(w, int):
                                    stage_chans[t.attr] = w
                            if cls in ('Linear', 'Gemm') and len(v.args) >= 2:
                                w = None
                                a1 = v.args[1]
                                if isinstance(a1, ast.Constant):
                                    w = a1.value
                                elif isinstance(a1, ast.Name) and a1.id in consts:
                                    w = consts[a1.id]
                                if isinstance(w, int):
                                    stage_widths[t.attr] = w
                    if isinstance(v, ast.Name) and v.id in consts:
                        attrs[t.attr] = consts[v.id]
                    elif isinstance(v, ast.Call) and \
                            isinstance(v.func, ast.Name) and \
                            v.func.id == 'len' and \
                            len(v.args) == 1 and \
                            isinstance(v.args[0], ast.Name) and \
                            v.args[0].id in consts and \
                            hasattr(consts[v.args[0].id], '__len__'):
                        # self.num_layers = len(depths) (depths an init-arg)
                        attrs[t.attr] = len(consts[v.args[0].id])
                    else:
                        try:
                            attrs[t.attr] = ast.literal_eval(v)
                        except Exception:
                            # a boolean/comparison flag over init-args
                            # (self.use_residual = (stride==1 and ...)):
                            # store the raw expr; evaluated lazily at the
                            # forward's If-test (_eval_flag in scope there).
                            _flag_exprs[t.attr] = v
    def scalar_of(e):
        """Resolve an expression to a scalar if possible."""
        if isinstance(e, ast.Constant):
            return e.value
        if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub):
            v = scalar_of(e.operand)
            return -v if v is not None else None
        if isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name) \
           and e.value.id == 'self' and e.attr in attrs:
            return attrs[e.attr]
        if isinstance(e, ast.Name) and e.id in consts:
            return consts[e.id]
        return None
    def _eval_flag(e):
        """Evaluate a boolean/comparison expr over init consts+attrs
        (self.use_residual = (stride == 1 and in == out)). Returns the
        bool, or None if not provably evaluable (refuse, never guess)."""
        if isinstance(e, ast.Constant):
            return e.value
        if isinstance(e, ast.Compare) and len(e.ops) == 1:
            _l = scalar_of(e.left)
            _r = scalar_of(e.comparators[0])
            if _l is None or _r is None:
                return None
            _op = e.ops[0]
            if isinstance(_op, ast.Eq):
                return _l == _r
            if isinstance(_op, ast.NotEq):
                return _l != _r
            if isinstance(_op, ast.Lt):
                return _l < _r
            if isinstance(_op, ast.Gt):
                return _l > _r
            if isinstance(_op, ast.LtE):
                return _l <= _r
            if isinstance(_op, ast.GtE):
                return _l >= _r
            return None
        if isinstance(e, ast.BoolOp):
            _vals = [_eval_flag(v) for v in e.values]
            if any(_v is None for _v in _vals):
                return None
            if isinstance(e.op, ast.And):
                return all(_vals)
            if isinstance(e.op, ast.Or):
                return any(_vals)
            return None
        return scalar_of(e)
    def _axis_is_last(d, rank):
        """True iff reduction axis d is provably the LAST contiguous dim.
        dim=-1 always last; dim=k needs known rank with k == rank-1.
        Unknown dim or unknown rank -> NOT provable -> refuse."""
        if d == -1:
            return True
        if d is None or rank is None:
            return False
        return d == rank - 1

    current_rank = [None]   # rank of x after the latest stage (boxed)
    current_width = [None]  # last-dim width after the latest stage (if provable)
    current_chan = [None]   # channel count (dim 1) after the latest stage
    # input SPATIAL dims from get_inputs (torch.rand/randn args past
    # batch+channel, resolved via consts) — for spatial-size proofs:
    input_spatial = None
    input_batch = None
    input_chan_in = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'get_inputs':
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    f = sub.func
                    fn = f.attr if isinstance(f, ast.Attribute) else None
                    if fn in ('rand', 'randn'):
                        dims = []
                        for a in sub.args:
                            try:
                                dims.append(ast.literal_eval(a))
                            except Exception:
                                if isinstance(a, ast.Name) and a.id in consts:
                                    dims.append(consts[a.id])
                                else:
                                    dims.append(None)
                        if len(dims) >= 3 and all(
                                isinstance(d, int) for d in dims):
                            input_batch = dims[0]
                            input_chan_in = dims[1]
                            input_spatial = dims[2:]
                        break
    current_spatial = [list(input_spatial) if input_spatial else None]
    # second forward argument (two-input models: forward(self, x, y)):
    fwd_args = [a.arg for a in fwd.args.args[1:]]   # skip self
    arg2 = fwd_args[1] if len(fwd_args) > 1 else None
    ops = []
    _pending = {'t': None}
    _last_target = ['x']  # the chain-thread tracker (the #92 fix:
    # SSA-style forwards name each step — x_conv, x_norm... — the
    # binop handlers key on 'x'; the tracker knows which NAME holds
    # the current chain value)
    global LAST_STAGE_CLASSES
    LAST_STAGE_CLASSES = dict(stage_classes)

    # ═══ THE CONTAINER UNLOCK (Heath's loop-lift insight, the
    # static half): when the forward is a bare container call
    # (return self.X(x)) and X = nn.Sequential built in __init__
    # (possibly via a loop over const-evaluable args), SYMBOLICALLY
    # EXECUTE the init to unroll the module list, then SYNTHESIZE a
    # flat forward walk from it. The Logtalk lift (lib/logtalk/)
    # holds the declarative mirror; this is the walking half. ═══
    def _walk_user_class(cname, bind, outer_i):
        # RUNG-2: the depth-counted entry (depth-2 allowed;
        # depth-3 refuses upstream via the detection-arm guard):
        _in_user_class[0] += 1
        try:
            return _walk_user_class_inner(cname, bind, outer_i)
        finally:
            _in_user_class[0] -= 1

    def _walk_user_class_inner(cname, bind, outer_i):
        # THE B-INIT+C FILL: walk a user-class instance's __init__ (the
        # direct self.X = nn.Module assigns) with the ctor-args bound as env
        # (the #20 env-form), producing the internal module-stages with
        # composed paths ({outer_i}.<attr>). The forward-walk (next hop)
        # walks the class's forward where the #17 cat-arm fires. Returns a
        # list of ('stage', cls, args, path) specs, or None (honest refuse).
        _ucd = _user_classes.get(cname)
        if _ucd is None:
            return None
        _init = _ucd.get('init')
        if _init is None:
            return None
        # a child env seeded with the ctor-arg bind (in_channels=96, etc)
        _cenv = dict(bind)
        _internal = []
        for _ist in _init.body:
            # skip docstring / super().__init__() / non-self-assigns
            if isinstance(_ist, ast.Expr):
                continue
            # STEP-4b (the guard-miss walker-defect — the seg32
            # interv-roll root): an INIT-GUARD that REASSIGNS a
            # ctor-bound attr ('if min(self.input_resolution) <=
            # self.window_size: self.shift_size = 0; ...') —
            # evaluate the test against the env; if TRUE, apply
            # the body's self-assigns INTO the env (the walker's
            # forward-walk then sees the guarded values — the
            # real model's geometry, not the ctor's):
            if isinstance(_ist, ast.If):
                def _ig_ev(e):
                    if isinstance(e, ast.Constant):
                        return e.value
                    if isinstance(e, ast.Attribute) and \
                            isinstance(e.value, ast.Name) and \
                            e.value.id == 'self':
                        return _cenv.get(e.attr)
                    if isinstance(e, ast.Name):
                        return _cenv.get(e.id)
                    if isinstance(e, ast.Call) and \
                            isinstance(e.func, ast.Name) and \
                            e.func.id in ('min', 'max') and \
                            len(e.args) == 1:
                        _seq = _ig_ev(e.args[0])
                        if isinstance(_seq, (tuple, list)) and \
                                all(isinstance(q, (int, float))
                                    for q in _seq):
                            return (min(_seq) if e.func.id ==
                                    'min' else max(_seq))
                        return None
                    if isinstance(e, ast.Compare) and \
                            len(e.ops) == 1:
                        _l9 = _ig_ev(e.left)
                        _r9 = _ig_ev(e.comparators[0])
                        if isinstance(_l9, (int, float)) and \
                                isinstance(_r9, (int, float)):
                            _op9 = e.ops[0]
                            if isinstance(_op9, ast.LtE):
                                return _l9 <= _r9
                            if isinstance(_op9, ast.Lt):
                                return _l9 < _r9
                            if isinstance(_op9, ast.GtE):
                                return _l9 >= _r9
                            if isinstance(_op9, ast.Gt):
                                return _l9 > _r9
                            if isinstance(_op9, ast.Eq):
                                return _l9 == _r9
                        return None
                    return None
                _igt = _ig_ev(_ist.test)
                if _igt is True:
                    for _igs in _ist.body:
                        if isinstance(_igs, ast.Assign) and \
                                len(_igs.targets) == 1 and \
                                isinstance(_igs.targets[0],
                                           ast.Attribute) and \
                                isinstance(
                                    _igs.targets[0].value,
                                    ast.Name) and \
                                _igs.targets[0].value.id == \
                                'self':
                            _igv = _ig_ev(_igs.value)
                            if _igv is not None:
                                _cenv[_igs.targets[0].attr] = \
                                    _igv
                                # the forward-walk reads BIND
                                # (_cenv is a COPY — dict(bind));
                                # the guard must reach both:
                                bind[_igs.targets[0].attr] = \
                                    _igv
                # False or unevaluable: fall through (the
                # else-less guards of this family need no else-
                # handling; unevaluable = leave ctor-binds)
                continue
            if not (isinstance(_ist, ast.Assign) and len(_ist.targets) == 1
                    and isinstance(_ist.targets[0], ast.Attribute)
                    and isinstance(_ist.targets[0].value, ast.Name)
                    and _ist.targets[0].value.id == 'self'):
                continue
            _attr = _ist.targets[0].attr
            _v = _ist.value
            # self.X = nn.Module(...) -> a module-stage; resolve its class
            if isinstance(_v, ast.Call) and isinstance(_v.func, ast.Attribute):
                _cls = _v.func.attr  # Conv2d / ReLU / BatchNorm2d
                # THE PARAMETER-NAMING ARM (#30's root): nn.Parameter/
                # register_buffer are NOT modules — emitting them as
                # stages creates unhookable paths (get_submodule fails
                # = the logit_scale refusal) AND save-anchors could
                # land on them. NAME them into the env (the bind may
                # eval the expr later); never a stage:
                if _cls in ('Parameter', 'Buffer'):
                    continue
                # EVERY self.X = nn.Module is a real submodule at
                # features.N.<attr> (get_submodule-resolvable). Emit it as a
                # stage; the FORWARD-walk (next hop) applies them in order
                # (the activation-modules are stages too, not init-ops —
                # they're APPLIED in forward, defined here).
                _internal.append(('stage', _cls, None,
                                  f'{outer_i}.{_attr}'))
            elif isinstance(_v, ast.Call) and \
                    isinstance(_v.func, ast.Name):
                # RUNG-3c: self.X = PARAM(args) — the func is a
                # NAME (a ctor-param carrying a class-ref:
                # norm_layer=nn.LayerNorm → self.norm1 =
                # norm_layer(dim)) or a user-class ctor.
                # RUNG-3d (Iyun): if the func is a REAL user-class
                # (Mlp) and we're within the depth-bound, RECURSE
                # (walk it — the #18 direct-form one-depth-deeper:
                # bind its ctor-args, walk init+forward → the
                # fc1/GELU-island/fc2 stages) instead of opaque.
                if _v.func.id in _user_classes and \
                        _in_user_class[0] < 3 and \
                        _pev_shared[0] is not None:
                    # bind the Mlp ctor-args from THIS class's bind-env
                    _pevf = _pev_shared[0]
                    _sub_bind = {}
                    _sub_init = _user_classes[_v.func.id].get('init')
                    _sub_ps = ([a.arg for a in _sub_init.args.args[1:]]
                               if _sub_init else [])
                    for _spi, _spa in enumerate(_v.args):
                        if _spi < len(_sub_ps):
                            _spv = _pevf(_spa, dict(bind))
                            if _spv is not None:
                                _sub_bind[_sub_ps[_spi]] = _spv
                    for _skw in _v.keywords:
                        if _skw.arg is not None:
                            _skv = _pevf(_skw.value, dict(bind))
                            if _skv is not None:
                                _sub_bind[_skw.arg] = _skv
                    # recurse: walk the sub-class at {outer_i}.{attr}
                    _sub_stages = _walk_user_class(
                        _v.func.id, _sub_bind, f'{outer_i}.{_attr}')
                    if _sub_stages is not None:
                        # RUNG-3d: stash the sub-instance's (cname, bind) so
                        # the forward-arm can splice its forward (the Mlp's
                        # fc1->GELU-island->fc2) at the self.X(...) call.
                        _walked_subinstances[f'{outer_i}.{_attr}'] = (
                            _v.func.id, dict(_sub_bind))
                        _internal.extend(_sub_stages)
                    else:
                        # sub-walk refused -> degrade to opaque
                        _internal.append(
                            ('stage', _v.func.id, None,
                             f'{outer_i}.{_attr}'))
                else:
                    _cls2 = bind.get(_v.func.id) \
                        if isinstance(bind.get(_v.func.id), str) \
                        else (_v.func.id
                              if _v.func.id in _user_classes
                              else None)
                    if _cls2 is None:
                        # RUNG-3d: the PARAM-DEFAULT read — the
                        # func-Name is a ctor-param with an nn.X
                        # default (Mlp's act_layer=nn.GELU → the
                        # class-name 'GELU' → the activation-
                        # detection sees it → THE ISLAND):
                        _cls2 = 'Module'
                        _cin6 = _user_classes.get(cname, {}) \
                            .get('init')
                        if _cin6 is not None:
                            _ps6 = _cin6.args
                            _names6 = [a.arg for a in
                                       _ps6.args[1:]]
                            _defs6 = _ps6.defaults
                            _off6 = len(_names6) - len(_defs6)
                            for _di6, _dn6 in enumerate(_names6):
                                if _dn6 == _v.func.id and \
                                        _di6 >= _off6:
                                    _de6 = _defs6[_di6 - _off6]
                                    if isinstance(_de6,
                                                  ast.Attribute):
                                        _cls2 = _de6.attr
                                    elif isinstance(_de6,
                                                    ast.Name):
                                        _cls2 = _de6.id
                                    break
                    _internal.append(('stage', _cls2, None,
                                      f'{outer_i}.{_attr}'))
            else:
                # a non-module assign in init (scalar etc) -> record
                # in env — RUNG-3c: upgraded from literal_eval to
                # Iyun's _pev (the evaluator: BinOps/IfExps/self.X
                # from the bind — SwinMLPBlock's self.padding =
                # (ws - shift, ...) computes):
                try:
                    if _pev_shared[0] is not None:
                        _pv3 = _pev_shared[0](_v, dict(_cenv))
                        if _pv3 is not None:
                            _cenv[_attr] = _pv3
                    else:
                        _cenv[_attr] = ast.literal_eval(_v)
                except Exception:
                    pass
        # RUNG-2 WIRING (the consumer-call): populate any nested
        # ModuleLists THIS class builds (BasicLayer's self.blocks —
        # the comprehension form) under the path-prefixed key
        # ('{outer_i}.blocks'), so the forward-walk's blocks-loop
        # arm finds them. Iyun's parameterized populate = the tool;
        # this call = the wiring (best-effort: a populate-miss just
        # leaves the consult empty → the named refusal stands):
        _class_envs[str(outer_i)] = dict(_cenv)
        try:
            _populate_modulelists(_pinit_body=_init.body,
                                  _penv_base=_cenv,
                                  _ppath_prefix=outer_i)
        except Exception:
            pass
        return _internal
    def _unroll_sequential(attr_name):
        """Symbolically execute __init__ to recover the module list
        assigned to self.<attr_name> as nn.Sequential(*list).
        Returns a list of ('module', ClassName, [args]) or None."""
        if init is None:
            return None
        env = dict(consts)  # local symbolic env (init args included)
        lists = {}          # list-var -> [module specs]
        _list_pos = {}      # list-var -> next outer-position P
        seq_attr = {}
        # container-reach: local helper defs in __init__ that RETURN
        # nn.Sequential(...). {name: FunctionDef}. _module_spec resolves
        # calls to these by inlining the returned Sequential.
        _helpers = {}
        if init is not None:
            for _hn in init.body:
                if isinstance(_hn, ast.FunctionDef):
                    _helpers[_hn.name] = _hn
        def _ev_scalar(e, senv):
            # scalar expression eval in a child env (for helper bodies)
            if isinstance(e, ast.Constant):
                return e.value
            if isinstance(e, ast.Name):
                return senv.get(e.id, env.get(e.id))
            if isinstance(e, ast.BinOp):
                l = _ev_scalar(e.left, senv)
                r = _ev_scalar(e.right, senv)
                if l is None or r is None:
                    return None
                try:
                    return {ast.Mult: lambda a, b: a * b,
                            ast.Add: lambda a, b: a + b,
                            ast.Sub: lambda a, b: a - b,
                            ast.FloorDiv: lambda a, b: a // b,
                            ast.Div: lambda a, b: a / b,
                            ast.Mod: lambda a, b: a % b}[type(e.op)](l, r)
                except Exception:
                    return None
            if isinstance(e, ast.Call) and isinstance(e.func, ast.Name):
                if e.func.id in ('int', 'float', 'round') \
                        and len(e.args) == 1:
                    v = _ev_scalar(e.args[0], senv)
                    if v is None:
                        return None
                    try:
                        return {'int': int, 'float': float,
                                'round': round}[e.func.id](v)
                    except Exception:
                        return None
                if e.func.id in ('max', 'min') and e.args:
                    vs = [_ev_scalar(a, senv) for a in e.args]
                    if any(x is None for x in vs):
                        return None
                    try:
                        return (max if e.func.id == 'max'
                                else min)(*vs)
                    except Exception:
                        return None
            if isinstance(e, ast.IfExp):
                t = _ev_scalar(e.test, senv)
                if t is None:
                    return None
                return _ev_scalar(e.body, senv) if t \
                    else _ev_scalar(e.orelse, senv)
            if isinstance(e, ast.BoolOp):
                # stride == 1 and inp == oup
                vals = [_ev_scalar(v, senv) for v in e.values]
                if any(x is None for x in vals):
                    return None
                return (all(vals) if isinstance(e.op, ast.And)
                        else any(vals))
            if isinstance(e, ast.Compare) and len(e.ops) == 1:
                op = e.ops[0]
                # Is/IsNot vs None: handle None-as-a-VALUE (min_value is
                # None is a legitimate test; don't treat None-operand as
                # unevaluable for identity comparisons).
                if isinstance(op, (ast.Is, ast.IsNot)) \
                        and isinstance(e.comparators[0], ast.Constant) \
                        and e.comparators[0].value is None:
                    l = _ev_scalar(e.left, senv)
                    # l may legitimately be None here
                    if isinstance(op, ast.Is):
                        return l is None
                    return l is not None
                l = _ev_scalar(e.left, senv)
                r = _ev_scalar(e.comparators[0], senv)
                if l is None or r is None:
                    return None
                try:
                    return {ast.Lt: lambda a, b: a < b,
                            ast.Gt: lambda a, b: a > b,
                            ast.LtE: lambda a, b: a <= b,
                            ast.GtE: lambda a, b: a >= b,
                            ast.Eq: lambda a, b: a == b,
                            ast.NotEq: lambda a, b: a != b,
                            ast.Is: lambda a, b: a is b,
                            ast.IsNot: lambda a, b: a is not b}[
                        type(op)](l, r)
                except Exception:
                    return None
            return None
        def _exec_scalar(body, senv):
            # execute a scalar-helper body (assigns, ifs, augassign,
            # return) -> the returned scalar, or None if not evaluable.
            for st in body:
                if isinstance(st, ast.Expr):
                    continue  # docstring / bare expr
                if isinstance(st, ast.Return):
                    return _ev_scalar(st.value, senv) \
                        if st.value is not None else None
                if isinstance(st, ast.Assign) and len(st.targets) == 1 \
                        and isinstance(st.targets[0], ast.Name):
                    v = _ev_scalar(st.value, senv)
                    senv[st.targets[0].id] = v
                    continue
                if isinstance(st, ast.AugAssign) \
                        and isinstance(st.target, ast.Name):
                    cur = senv.get(st.target.id)
                    rhs = _ev_scalar(st.value, senv)
                    if cur is None or rhs is None:
                        return None
                    try:
                        senv[st.target.id] = {
                            ast.Add: lambda a, b: a + b,
                            ast.Sub: lambda a, b: a - b,
                            ast.Mult: lambda a, b: a * b}[
                            type(st.op)](cur, rhs)
                    except Exception:
                        return None
                    continue
                if isinstance(st, ast.If):
                    tv = _ev_scalar(st.test, senv)
                    if tv is None:
                        return None
                    branch = st.body if tv else st.orelse
                    r = _exec_scalar(branch, senv)
                    if r is not None:
                        return r
                    continue
                return None  # unhandled stmt-kind -> not evaluable
            return None
        def _ev(e):
            if isinstance(e, ast.Constant):
                return e.value
            if isinstance(e, ast.Name):
                return env.get(e.id)
            if isinstance(e, ast.Attribute) and \
                    isinstance(e.value, ast.Name) and \
                    e.value.id == 'self':
                # self.X: from attrs (captured init-side), else env
                # (self.embed_dim mirrors the embed_dim init-arg name).
                if e.attr in attrs:
                    return attrs[e.attr]
                return env.get(e.attr)
            if isinstance(e, ast.Call) and isinstance(e.func, ast.Name) \
                    and e.func.id == 'len' and len(e.args) == 1:
                _lv = _ev(e.args[0])
                return len(_lv) if _lv is not None and \
                    hasattr(_lv, '__len__') else None
            if isinstance(e, ast.Subscript):
                # depths[i_layer] / num_heads[i_layer] (list-index)
                _base = _ev(e.value)
                if _base is None:
                    return None
                _sl = e.slice
                _idx = _ev(_sl.value if isinstance(_sl, ast.Index) else _sl)
                if _idx is None:
                    return None
                try:
                    return _base[_idx]
                except Exception:
                    return None
            if isinstance(e, ast.List):
                out = []
                for el in e.elts:
                    v3 = _ev(el)
                    if v3 is None:
                        return None
                    out.append(v3)
                return out
            if isinstance(e, ast.Tuple):
                # e.g. nn.AdaptiveAvgPool2d((1, 1)) — a tuple-literal arg.
                out = []
                for el in e.elts:
                    v3 = _ev(el)
                    if v3 is None:
                        return None
                    out.append(v3)
                return tuple(out)
            if isinstance(e, ast.BinOp):
                l3, r3 = _ev(e.left), _ev(e.right)
                if l3 is None or r3 is None:
                    return None
                try:
                    return {ast.Mult: lambda a, b: a * b,
                            ast.Add: lambda a, b: a + b,
                            ast.Sub: lambda a, b: a - b,
                            ast.FloorDiv: lambda a, b: a // b,
                            ast.Div: lambda a, b: a / b,
                            ast.Mod: lambda a, b: a % b}[
                        type(e.op)](l3, r3)
                except Exception:
                    return None
            # int()/float()/round() wrappers (container-reach:
            # int(32*alpha) in helper-call args). Honest-None otherwise.
            if isinstance(e, ast.Call) and isinstance(e.func, ast.Name) \
                    and e.func.id in ('int', 'float', 'round') \
                    and len(e.args) == 1:
                _v = _ev(e.args[0])
                if _v is None:
                    return None
                try:
                    return {'int': int, 'float': float,
                            'round': round}[e.func.id](_v)
                except Exception:
                    return None
            # Compare: i == 0 (needed by the IfExp test).
            if isinstance(e, ast.Compare) and len(e.ops) == 1:
                _op = e.ops[0]
                if isinstance(_op, (ast.Is, ast.IsNot)) \
                        and isinstance(e.comparators[0], ast.Constant) \
                        and e.comparators[0].value is None:
                    _lv = _ev(e.left)
                    return (_lv is None) if isinstance(_op, ast.Is) \
                        else (_lv is not None)
                _lc = _ev(e.left)
                _rc = _ev(e.comparators[0])
                if _lc is None or _rc is None:
                    return None
                try:
                    return {ast.Eq: lambda a, b: a == b,
                            ast.NotEq: lambda a, b: a != b,
                            ast.Lt: lambda a, b: a < b,
                            ast.Gt: lambda a, b: a > b,
                            ast.LtE: lambda a, b: a <= b,
                            ast.GtE: lambda a, b: a >= b}[
                        type(_op)](_lc, _rc)
                except Exception:
                    return None
            # BoolOp: and/or (defensive).
            if isinstance(e, ast.BoolOp):
                _bvs = [_ev(v) for v in e.values]
                if any(_x is None for _x in _bvs):
                    return None
                return (all(_bvs) if isinstance(e.op, ast.And)
                        else any(_bvs))
            # IfExp ternary: stride = s if i == 0 else 1
            if isinstance(e, ast.IfExp):
                _t = _ev(e.test)
                if _t is None:
                    return None
                return _ev(e.body) if _t else _ev(e.orelse)
            # max()/min() variadic builtins (_make_divisible uses max()).
            if isinstance(e, ast.Call) and isinstance(e.func, ast.Name) \
                    and e.func.id in ('max', 'min') and e.args:
                _vs = [_ev(a) for a in e.args]
                if any(_x is None for _x in _vs):
                    return None
                try:
                    return (max if e.func.id == 'max' else min)(*_vs)
                except Exception:
                    return None
            # a Call to a LOCAL SCALAR-HELPER (e.g. _make_divisible):
            # mini-evaluate its body (assigns + ifs + return of arithmetic)
            # with the call-args bound in a child scalar-env. Best-effort
            # -> None on anything not scalar-evaluable (honest degrade).
            if isinstance(e, ast.Call) and isinstance(e.func, ast.Name) \
                    and e.func.id in _helpers:
                _hd = _helpers[e.func.id]
                _senv = {}
                _params = _hd.args.args
                _defaults = _hd.args.defaults
                # bind positional args
                for _pi, _pa in enumerate(_params):
                    if _pi < len(e.args):
                        _av = _ev(e.args[_pi])
                        if _av is None:
                            return None
                        _senv[_pa.arg] = _av
                    else:
                        # default (defaults align to the tail of params)
                        _di = _pi - (len(_params) - len(_defaults))
                        if 0 <= _di < len(_defaults):
                            _dv = _ev_scalar(_defaults[_di], _senv)
                            _senv[_pa.arg] = _dv
                        else:
                            _senv[_pa.arg] = None
                return _exec_scalar(_hd.body, _senv)
            return None
        def _helper_build_list(body, henv):
            # Walk a rich helper body (scalar-locals + one list built by
            # append/extend, conditionally) and resolve its final
            # return nn.Sequential(*LIST), <...>  -> [module specs].
            # Uses _ev_scalar for scalar-locals; _module_spec for modules.
            # Best-effort: None on anything not expressible.
            # ENV-SWAP (Bocher's root fix): _module_spec's _ev reads the
            # CLOSURE env, but the helper's scalar-locals (inp/hidden_dim/
            # expand_ratio) live in _senv. Swap _senv into env for the walk
            # so module-args (nn.Conv2d(inp, hidden_dim, ...)) resolve;
            # restore at the end (the #19 try/finally pattern).
            _lists = {}   # list-var name -> [module specs]
            _senv = dict(henv)
            _hbl_saved = dict(env)
            env.update(_senv)  # helper-locals visible to _module_spec's _ev
            def _walk(stmts):
                for st in stmts:
                    if isinstance(st, ast.Expr):
                        # bare call: LIST.append(module) / LIST.extend([...])
                        v = st.value
                        if isinstance(v, ast.Call) \
                                and isinstance(v.func, ast.Attribute) \
                                and isinstance(v.func.value, ast.Name) \
                                and v.func.value.id in _lists:
                            _lname = v.func.value.id
                            if v.func.attr == 'append' and len(v.args) == 1:
                                sp = _module_spec(v.args[0])
                                if sp is None:
                                    return False
                                _lists[_lname].append(sp)
                                continue
                            if v.func.attr == 'extend' and len(v.args) == 1 \
                                    and isinstance(v.args[0], ast.List):
                                for _el in v.args[0].elts:
                                    sp = _module_spec(_el)
                                    if sp is None:
                                        return False
                                    _lists[_lname].append(sp)
                                continue
                        continue  # other bare exprs: ignore
                    if isinstance(st, ast.Assign) and len(st.targets) == 1 \
                            and isinstance(st.targets[0], ast.Name):
                        tgt = st.targets[0].id
                        if isinstance(st.value, ast.List):
                            # LIST = [ ... modules ... ]
                            _specs0 = []
                            for _el in st.value.elts:
                                sp = _module_spec(_el)
                                if sp is None:
                                    return False
                                _specs0.append(sp)
                            _lists[tgt] = _specs0
                        else:
                            # scalar local (hidden_dim, use_res_connect):
                            # write to BOTH _senv (for _ev_scalar) and env
                            # (for _module_spec's _ev module-args).
                            _sv = _ev_scalar(st.value, _senv)
                            _senv[tgt] = _sv
                            env[tgt] = _sv
                        continue
                    if isinstance(st, ast.If):
                        tv = _ev_scalar(st.test, _senv)
                        if tv is None:
                            return False
                        _br = _walk(st.body if tv else st.orelse)
                        if _br is False:
                            return False
                        if isinstance(_br, tuple) and _br[0] == 'RET':
                            return _br  # propagate the branch's Return up
                        continue
                    if isinstance(st, ast.Return):
                        return 'RET', st.value
                    return False  # unhandled
                return None
            # run the walk, capturing the return
            _ret_val = None
            def _run(stmts):
                nonlocal _ret_val
                for st in stmts:
                    r = _walk([st])
                    if r is False:
                        return False
                    if isinstance(r, tuple) and r[0] == 'RET':
                        _ret_val = r[1]
                        return True
                return None
            if _run(body) is False:
                return None
            if _ret_val is None:
                return None
            # the return is  nn.Sequential(*LIST), <bool>  OR nn.Sequential(*LIST)
            _rv = _ret_val
            if isinstance(_rv, ast.Tuple) and _rv.elts:
                _rv = _rv.elts[0]  # take element 0, ignore the bool
            if not (isinstance(_rv, ast.Call)
                    and isinstance(_rv.func, ast.Attribute)
                    and _rv.func.attr == 'Sequential'):
                return None
            # resolve Sequential(*LIST) -> the built list
            if len(_rv.args) == 1 and isinstance(_rv.args[0], ast.Starred) \
                    and isinstance(_rv.args[0].value, ast.Name) \
                    and _rv.args[0].value.id in _lists:
                return _lists[_rv.args[0].value.id]
            # or Sequential(direct-args)
            _out = []
            for _a in _rv.args:
                sp = _module_spec(_a)
                if sp is None:
                    return None
                _out.append(sp)
            return _out
        def _module_spec(call):
            # nn.Linear(a, b) / nn.ReLU() → ('module', name, args):
            # [0]-subscript of a helper-call: helper(...)[0] takes the
            # Sequential from a (Sequential, bool) tuple-return.
            if isinstance(call, ast.Subscript) \
                    and isinstance(call.value, ast.Call) \
                    and isinstance(call.value.func, ast.Name) \
                    and call.value.func.id in _helpers:
                _idx = call.slice
                if isinstance(_idx, ast.Constant) and _idx.value == 0:
                    # resolve the helper-call (returns a ('seq',...) for
                    # element 0 already — the rich-resolve ignores [1]).
                    return _module_spec(call.value)
                return None
            if not isinstance(call, ast.Call):
                return None
            fn3 = call.func
            cls3 = (fn3.attr if isinstance(fn3, ast.Attribute)
                    else (fn3.id if isinstance(fn3, ast.Name)
                          else None))
            if cls3 is None:
                return None
            # THE CUSTOM-CLASS RECURSION ENTRY (#18's FireModule —
            # the skeleton; the walk-fills follow [the joint build:
            # B-init/C = Iyun's fill; B-forward = mine]): a Call to
            # a USER-DEFINED nn.Module class → the class-walk.
            # THIS SKELETON: detect + bind ctor-args + emit the
            # ('user_class', name, {param: value}) spec — the fills
            # turn it into the walked stage-specs. Depth-guard: the
            # class-walk sets _in_user_class; a user-class ctor met
            # INSIDE one refuses BY NAME (nested_custom_module —
            # Swin's rung, not this one).
            if isinstance(fn3, ast.Name) and fn3.id in _user_classes:
                if _in_user_class[0] >= 2:
                    ops.append(('unknown_call',
                                'nested_custom_module_depth3'))
                    return None
                _ucd = _user_classes[fn3.id]
                _ucbind = {}
                for _upi, _upn in enumerate(_ucd['params']):
                    if _upi < len(call.args):
                        _ubv = _ev(call.args[_upi])
                        if _ubv is None:
                            return None
                        _ucbind[_upn] = _ubv
                for _ukw in call.keywords:
                    if _ukw.arg is not None:
                        _ukv = _ev(_ukw.value)
                        if _ukv is None:
                            return None
                        _ucbind[_ukw.arg] = _ukv
                return ('user_class', fn3.id, _ucbind)
            # container-reach: a Call to a LOCAL HELPER that returns
            # nn.Sequential(...). Inline it: bind params to _ev(call-args)
            # in a child env, walk the Return(nn.Sequential(...)).
            # GUARD: helper body must be [optional-docstring,
            # Return(Call-to-nn.Sequential)] -> else honest None.
            if isinstance(fn3, ast.Name) and fn3.id in _helpers:
                _hdef = _helpers[fn3.id]
                _body = [s for s in _hdef.body
                         if not (isinstance(s, ast.Expr)
                                 and isinstance(s.value, ast.Constant)
                                 and isinstance(s.value.value, str))]
                # bind params -> _ev(call-args) [+ kwargs] in a child env
                _saved = dict(env)
                try:
                    _params = [a.arg for a in _hdef.args.args]
                    for _pi, _pn in enumerate(_params):
                        if _pi < len(call.args):
                            _bv = _ev(call.args[_pi])
                            if _bv is None:
                                return None
                            env[_pn] = _bv
                    for _kw in call.keywords:
                        if _kw.arg is not None:
                            _kv = _ev(_kw.value)
                            if _kv is None:
                                return None
                            env[_kw.arg] = _kv
                    # SIMPLE form: single Return(nn.Sequential(direct-args))
                    if len(_body) == 1 and isinstance(_body[0], ast.Return) \
                            and isinstance(_body[0].value, ast.Call) \
                            and isinstance(_body[0].value.func, ast.Attribute) \
                            and _body[0].value.func.attr == 'Sequential' \
                            and not any(isinstance(a, ast.Starred)
                                        for a in _body[0].value.args):
                        _ret = _body[0].value
                        _specs = []
                        for _inner_i, _el in enumerate(_ret.args):
                            _sp = _module_spec(_el)
                            if _sp is None:
                                return None
                            if _sp[0] == 'seq':
                                for _nk, _ncls, _na, _npath in _sp[1]:
                                    _specs.append((_nk, _ncls, _na,
                                                   f'{_inner_i}.{_npath}'))
                            else:
                                _specs.append((_sp[0], _sp[1], _sp[2],
                                               str(_inner_i)))
                        return ('seq', _specs)
                    # RICH form: scalar-locals + layers=[] built by
                    # append/extend (conditional) + return
                    # nn.Sequential(*layers), <bool>. Build the module-list.
                    _mods = _helper_build_list(_body, env)
                    if not _mods:  # None, False, or empty -> refuse
                        return None
                    _specs = []
                    for _inner_i, _sp in enumerate(_mods):
                        if _sp[0] == 'seq':
                            for _nk, _ncls, _na, _npath in _sp[1]:
                                _specs.append((_nk, _ncls, _na,
                                               f'{_inner_i}.{_npath}'))
                        else:
                            _specs.append((_sp[0], _sp[1], _sp[2],
                                           str(_inner_i)))
                    return ('seq', _specs)
                finally:
                    env.clear()
                    env.update(_saved)
            args3 = []
            for a in call.args:
                v3 = _ev(a)
                if v3 is None:
                    return None
                args3.append(v3)
            return ('module', cls3, args3)
        def _if_test_ev(e):
            # the evaluable-init-If test (the #21 If-arm, rebuilt —
            # Iyun's design): Compare/BoolOp/Name/Constant over the
            # env (init args + consts). None = unevaluable.
            if isinstance(e, ast.Constant):
                return e.value
            if isinstance(e, ast.Name):
                return env.get(e.id)
            if isinstance(e, ast.Compare) and len(e.ops) == 1:
                l4 = _if_test_ev(e.left)
                r4 = _if_test_ev(e.comparators[0])
                if l4 is None or r4 is None:
                    return None
                import operator as _op4
                f4 = {ast.Eq: _op4.eq, ast.NotEq: _op4.ne,
                      ast.Lt: _op4.lt, ast.LtE: _op4.le,
                      ast.Gt: _op4.gt, ast.GtE: _op4.ge}.get(
                    type(e.ops[0]))
                return f4(l4, r4) if f4 else None
            if isinstance(e, ast.BoolOp):
                vals = [_if_test_ev(v4) for v4 in e.values]
                if any(v4 is None for v4 in vals):
                    return None
                return (all(vals) if isinstance(e.op, ast.And)
                        else any(vals))
            return None
        def _exec_body(body):
            for st in body:
                # the evaluable init-If (the #21 arm): expand the
                # taken branch inline; unevaluable → honest False:
                if isinstance(st, ast.If):
                    # Iyun's semantics (the correct init-side form):
                    # unevaluable test -> SKIP the If (honest
                    # DEGRADATION — the stages inside stay unknown;
                    # init-walking is best-effort). Forward-side
                    # Ifs refuse; init-side Ifs degrade.
                    _tv = _if_test_ev(st.test)
                    if _tv is None:
                        continue
                    if _exec_body(st.body if _tv else st.orelse) \
                            is False:
                        return False
                    continue
                if isinstance(st, ast.Assign) and \
                        len(st.targets) == 1:
                    t3 = st.targets[0]
                    if isinstance(t3, ast.Name):
                        if isinstance(st.value, ast.List) and \
                                not st.value.elts:
                            lists[t3.id] = []
                            continue
                        if isinstance(st.value, ast.List) and \
                                st.value.elts:
                            # LIST = [ module, module, ... ] (features =
                            # [nn.Conv2d(...), nn.BatchNorm2d(...), ...]).
                            # If every elt is a module-spec, seed the list;
                            # else fall through to scalar _ev.
                            _seed = []
                            _ok_seed = True
                            for _el in st.value.elts:
                                sp = _module_spec(_el)
                                if sp is None:
                                    _ok_seed = False
                                    break
                                if sp[0] == 'seq':
                                    _seed.extend(sp[1])
                                elif sp[0] == 'user_class':
                                    # the #18 skeleton: the class-walk
                                    # fill (B-init/C) replaces this —
                                    # for now the placeholder rides
                                    # (the emitter names the refusal):
                                    _seed.append(sp)
                                else:
                                    _seed.append(sp)
                            if _ok_seed:
                                # seed with runtime-true positions 0,1,2...
                                _pseed = []
                                for _si, _ss in enumerate(_seed):
                                    if len(_ss) > 3:
                                        _pseed.append(_ss)  # already pathed
                                    else:
                                        _pseed.append(
                                            (_ss[0], _ss[1], _ss[2],
                                             str(_si)))
                                lists[t3.id] = _pseed
                                _list_pos[t3.id] = len(_pseed)
                                continue
                        v3 = _ev(st.value)
                        if v3 is not None:
                            env[t3.id] = v3
                            continue
                        continue
                    if isinstance(t3, ast.Attribute) and \
                            isinstance(t3.value, ast.Name) and \
                            t3.value.id == 'self':
                        # self.X = nn.Sequential(*layers) OR the
                        # DIRECT form Sequential(item, item, ...)
                        # (VGG's shape — day 2 extension 1):
                        v4 = st.value
                        # RUNG-1 STUB (the DIRECT ModuleList form):
                        # self.X = nn.ModuleList([UserClass(...),..])
                        # → _modulelists[X]. The LOOP-APPEND form
                        # (self.X = nn.ModuleList(); for i in
                        # range(N): self.X.append(UserClass(args)))
                        # is the init-side FILL (Iyun's half — the
                        # per-iteration bind via _ev_scalar with the
                        # loop env).
                        if isinstance(v4, ast.Call) and \
                                isinstance(v4.func, ast.Attribute) \
                                and v4.func.attr == 'ModuleList' \
                                and len(v4.args) == 1 and \
                                isinstance(v4.args[0], ast.List):
                            _mlspecs = []
                            for _mle in v4.args[0].elts:
                                _mlsp = _module_spec(_mle)
                                if _mlsp is None or \
                                        _mlsp[0] != 'user_class':
                                    _mlspecs = None
                                    break
                                _mlspecs.append((_mlsp[1], _mlsp[2]))
                            if _mlspecs is not None:
                                _modulelists[t3.attr] = _mlspecs
                                continue
                        if isinstance(v4, ast.Call) and \
                                isinstance(v4.func, ast.Attribute) \
                                and v4.func.attr == 'Sequential':
                            if len(v4.args) == 1 and \
                                    isinstance(v4.args[0],
                                               ast.Starred) and \
                                    isinstance(v4.args[0].value,
                                               ast.Name):
                                seq_attr[t3.attr] = \
                                    v4.args[0].value.id
                            elif v4.args and not any(
                                    isinstance(a, ast.Starred)
                                    for a in v4.args):
                                _direct = []
                                for _outer_i, a in enumerate(v4.args):
                                    spec = _module_spec(a)
                                    if spec is None:
                                        _direct = None
                                        break
                                    if spec[0] == 'seq':
                                        # prepend the OUTER index so the
                                        # path is runtime-true nested:
                                        # outer(helper-call).inner(within).
                                        for _sk, _scl, _sa, _sp in spec[1]:
                                            _direct.append(
                                                (_sk, _scl, _sa,
                                                 f'{_outer_i}.{_sp}'))
                                    elif spec[0] == 'user_class':
                                        # THE #18 FILL: expand the user-class
                                        # instance (FireModule) — walk its
                                        # init with the bound ctor-args, emit
                                        # the internal stages with composed
                                        # paths ({outer_i}.<attr>). The
                                        # forward-walk (next hop) walks the
                                        # class forward where #17's cat fires.
                                        _uc_internal = _walk_user_class(
                                            spec[1], spec[2], _outer_i)
                                        if _uc_internal is None:
                                            _direct = None
                                            break
                                        # THE FORWARD-WALK HOP: the
                                        # init-walk gives the attr_map
                                        # (attr -> (cls, path)); the
                                        # forward-walk orders the
                                        # APPLICATION (activations as
                                        # OPS -> fusable islands; the
                                        # return-cat as the #17 branch
                                        # form). Fall back to the
                                        # init-order stages if the
                                        # forward refuses (honest
                                        # degradation):
                                        _amap = {}
                                        for _uci in _uc_internal:
                                            _p5 = _uci[3]
                                            _at5 = _p5.split('.', 1)[1] \
                                                if '.' in _p5 else _p5
                                            _amap[_at5] = (_uci[1], _p5)
                                        _ucf = _walk_user_forward(
                                            spec[1], spec[2], _outer_i,
                                            _amap)
                                        if _ucf is not None:
                                            _direct.extend(_ucf)
                                        else:
                                            for _uci in _uc_internal:
                                                _direct.append(_uci)
                                    else:
                                        # plain module at outer position:
                                        # its path IS the outer index.
                                        _direct.append(
                                            (spec[0], spec[1], spec[2],
                                             str(_outer_i)))
                                if _direct:
                                    _dl = f'__direct_{t3.attr}'
                                    lists[_dl] = _direct
                                    seq_attr[t3.attr] = _dl
                        continue
                elif isinstance(st, ast.Expr) and \
                        isinstance(st.value, ast.Call) and \
                        isinstance(st.value.func, ast.Attribute) \
                        and st.value.func.attr == 'append' and \
                        isinstance(st.value.func.value, ast.Name) \
                        and st.value.func.value.id in lists:
                    spec = _module_spec(st.value.args[0])
                    if spec is None:
                        return False
                    _lnm = st.value.func.value.id
                    # the outer position P = the count of TOP-LEVEL
                    # features-children so far. Each append (a block or a
                    # plain module) occupies ONE position. Track it via a
                    # per-list counter keyed on the current path-prefixes.
                    _pos = _list_pos.get(_lnm, 0)
                    if spec[0] == 'seq':
                        # a block-Sequential at position P: its inner-specs
                        # get the runtime-true nested path P.<inner>.
                        for _sk, _scl, _sa, _sp in spec[1]:
                            lists[_lnm].append(
                                (_sk, _scl, _sa, f'{_pos}.{_sp}'))
                    else:
                        # a plain module at position P: path = P
                        lists[_lnm].append(
                            (spec[0], spec[1], spec[2], str(_pos)))
                    _list_pos[_lnm] = _pos + 1
                elif isinstance(st, ast.Expr) and \
                        isinstance(st.value, ast.Call) and \
                        isinstance(st.value.func, ast.Attribute) and \
                        st.value.func.attr == 'append' and \
                        isinstance(st.value.func.value, ast.Attribute) and \
                        isinstance(st.value.func.value.value, ast.Name) and \
                        st.value.func.value.value.id == 'self' and \
                        len(st.value.args) == 1:
                    # RUNG-1 INIT-FILL (Iyun's half): self.ATTR.append(X)
                    # inside the range-For (i bound in env). X is either
                    # a UserClass(kwargs) call directly, or a local var
                    # bound to one (the #29 two-step: layer=BasicLayer(..);
                    # self.layers.append(layer)). Resolve X -> a Call to a
                    # user-class; _ev_scalar the kwargs in the current env
                    # (has the loop-var i_layer + the init-env); build the
                    # per-iteration (class_name, bind_dict) into _modulelists.
                    _mlattr = st.value.func.value.attr
                    _appx = st.value.args[0]
                    # resolve local-var indirection: layer=BasicLayer(...)
                    if isinstance(_appx, ast.Name):
                        _appx = env.get('__localcall__' + _appx.id, _appx)
                    if isinstance(_appx, ast.Call) and \
                            isinstance(_appx.func, ast.Name) and \
                            _appx.func.id in _user_classes:
                        _mlcls = _appx.func.id
                        _mlbind = {}
                        # positional args (mapped by __init__ param order)
                        _ucinit = _user_classes[_mlcls].get('init')
                        _uparams = ([a.arg for a in _ucinit.args.args[1:]]
                                    if _ucinit else [])
                        for _pi, _pa in enumerate(_appx.args):
                            if _pi < len(_uparams):
                                _pv = _ev_scalar(_pa, env)
                                if _pv is not None:
                                    _mlbind[_uparams[_pi]] = _pv
                        # keyword args (dim=.., depth=.., downsample=..)
                        for _kw in _appx.keywords:
                            if _kw.arg is None:
                                continue
                            # class-ref arg (downsample=PatchMerging|None):
                            # store the class-NAME/None (the conditional-
                            # instantiation refuses-by-name inside the walk)
                            _kv = _kw.value
                            if isinstance(_kv, ast.Name) and \
                                    _kv.id in _user_classes:
                                _mlbind[_kw.arg] = _kv.id
                            elif isinstance(_kv, ast.IfExp):
                                # PatchMerging if i<N-1 else None (IfExp):
                                # eval the test, pick the branch (class-name
                                # or None) via _ev_scalar / class-name
                                _tv = _ev_scalar(_kv.test, env)
                                _br = _kv.body if _tv else _kv.orelse
                                if isinstance(_br, ast.Name) and \
                                        _br.id in _user_classes:
                                    _mlbind[_kw.arg] = _br.id
                                elif isinstance(_br, ast.Constant):
                                    _mlbind[_kw.arg] = _br.value
                                else:
                                    _bv = _ev_scalar(_br, env)
                                    if _bv is not None:
                                        _mlbind[_kw.arg] = _bv
                            else:
                                _kvv = _ev_scalar(_kv, env)
                                if _kvv is not None:
                                    _mlbind[_kw.arg] = _kvv
                                # else: inert arg (dpr-slice) refuse-by-name
                                # (no shape-consequence) -- just omit it
                        _modulelists.setdefault(_mlattr, []).append(
                            (_mlcls, _mlbind))
                    continue
                elif isinstance(st, ast.Assign) and \
                        len(st.targets) == 1 and \
                        isinstance(st.targets[0], ast.Name) and \
                        isinstance(st.value, ast.Call) and \
                        isinstance(st.value.func, ast.Name) and \
                        st.value.func.id in _user_classes:
                    # RUNG-1: capture layer=BasicLayer(..) so the following
                    # self.ATTR.append(layer) can resolve the local-var.
                    env['__localcall__' + st.targets[0].id] = st.value
                    continue
                elif isinstance(st, ast.For) and \
                        isinstance(st.target, ast.Name) and \
                        isinstance(st.iter, ast.Call) and \
                        isinstance(st.iter.func, ast.Name) and \
                        st.iter.func.id == 'range':
                    # for i in range(n): evaluate range's args via _ev,
                    # iterate, bind i, recurse (init-side range-loop; the
                    # inner MobileNetV2 block-repeat loop).
                    _rargs = [_ev(a) for a in st.iter.args]
                    if any(_r is None for _r in _rargs):
                        continue  # un-evaluable range -> DEGRADE (skip)
                    try:
                        _rng = range(*[int(_r) for _r in _rargs])
                    except Exception:
                        continue
                    for _iv in _rng:
                        env[st.target.id] = _iv
                        if _exec_body(st.body) is False:
                            return False
                elif isinstance(st, ast.For) and \
                        isinstance(st.target, ast.Name):
                    it3 = _ev(st.iter)
                    if it3 is None or \
                            not isinstance(it3, (list, tuple)):
                        # un-evaluable iter (e.g. for m in self.modules():
                        # the weight-init loop) -> DEGRADE (skip the loop;
                        # it's not structure). Init-side best-effort
                        # doctrine: skip, never abort the whole walk.
                        continue
                    for item in it3:
                        env[st.target.id] = item
                        if _exec_body(st.body) is False:
                            return False
                elif isinstance(st, ast.For) and \
                        isinstance(st.target, ast.Tuple) and \
                        all(isinstance(e, ast.Name)
                            for e in st.target.elts):
                    # tuple-target loop: for t,c,n,s in <table>
                    # (the setting-table is a literal list-of-lists).
                    # zip-bind each name to item[j] per iteration.
                    it3 = _ev(st.iter)
                    if it3 is None or \
                            not isinstance(it3, (list, tuple)):
                        continue  # un-evaluable -> DEGRADE (skip)
                    _names = [e.id for e in st.target.elts]
                    for item in it3:
                        if not isinstance(item, (list, tuple)) or \
                                len(item) != len(_names):
                            return False
                        for _j, _nm in enumerate(_names):
                            env[_nm] = item[_j]
                        if _exec_body(st.body) is False:
                            return False
                # (other statements: super().__init__ etc — skip)
            return True
        if _exec_body(init.body) is False:
            return None
        lv = seq_attr.get(attr_name)
        if lv is None or lv not in lists:
            return None
        return lists[lv]

    # detect the bare-container forward:
    _CONT_ACT = {'ReLU': ('relu',), 'ReLU6': ('clamp', 0.0, 6.0),
                 'Tanh': ('tanh',),
                 'Sigmoid': ('sigmoid',), 'GELU': ('gelu',),
                 'Hardswish': ('hardswish',), 'Mish': ('mish',)}
    _fwd_body = [st for st in fwd.body
                 if not (isinstance(st, ast.Expr) and
                         isinstance(st.value, ast.Constant))]

    # ═══ EXTENSION 2 (day 2): the MULTI-CONTAINER walk — forwards
    # of the shape x = self.C1(x); x = torch.flatten(x, 1);
    # x = self.C2(x); return x (VGG). Each container-attr expands
    # via _unroll; interleaved functionals become F_ stages. Falls
    # through untouched if any statement doesn't fit. ═══
    def _multi_container_synth():
        _out = []
        _n_containers = 0
        # PRE-PASS: expand evaluable If-statements into their taken branch
        # (hasattr(self,'X') / self.FLAG resolved from init state). An
        # unevaluable test -> refuse the whole synth (never guess a branch).
        def _if_test(t):
            # hasattr(self, 'NAME') -> is NAME a known module/attr
            if isinstance(t, ast.Call) and isinstance(t.func, ast.Name) \
                    and t.func.id == 'hasattr' and len(t.args) == 2 \
                    and isinstance(t.args[0], ast.Name) \
                    and t.args[0].id == 'self' \
                    and isinstance(t.args[1], ast.Constant):
                _nm = t.args[1].value
                return (_nm in stage_classes) or (_nm in seq_attr) \
                    or (_nm in attrs)
            # self.FLAG -> the init-recorded bool
            if isinstance(t, ast.Attribute) \
                    and isinstance(t.value, ast.Name) \
                    and t.value.id == 'self':
                if t.attr in attrs and isinstance(attrs[t.attr], bool):
                    return attrs[t.attr]
                if t.attr in _flag_exprs:
                    return _eval_flag(_flag_exprs[t.attr])
            return None  # unevaluable
        _expanded = []
        for st in _fwd_body:
            if isinstance(st, ast.If):
                _tv = _if_test(st.test)
                if _tv is None:
                    return None  # unevaluable test -> refuse
                _expanded.extend(st.body if _tv else st.orelse)
            else:
                _expanded.append(st)
        for st in _expanded:
            # identity = x  -> a residual save (the #70-form)
            if isinstance(st, ast.Assign) and len(st.targets) == 1 \
                    and isinstance(st.targets[0], ast.Name) \
                    and st.targets[0].id != 'x' \
                    and isinstance(st.value, ast.Name) \
                    and st.value.id == 'x':
                _out.append(('save', st.targets[0].id))
                continue
            # x = x + SAVED  or  x += SAVED  (AugAssign normalized below)
            if isinstance(st, ast.Assign) and len(st.targets) == 1 \
                    and isinstance(st.targets[0], ast.Name) \
                    and st.targets[0].id == 'x' \
                    and isinstance(st.value, ast.BinOp) \
                    and isinstance(st.value.op, ast.Add) \
                    and isinstance(st.value.left, ast.Name) \
                    and st.value.left.id == 'x' \
                    and isinstance(st.value.right, ast.Name):
                _out.append(('add_saved', st.value.right.id))
                continue
            if isinstance(st, ast.Return):
                if isinstance(st.value, ast.Name):
                    continue  # return x
                # return torch.flatten(x, D) — the shape-fn Return
                # arm (#18's Model.forward tail; mirrors the main
                # Return handler's arm):
                if isinstance(st.value, ast.Call) and \
                        isinstance(st.value.func, ast.Attribute) and \
                        isinstance(st.value.func.value, ast.Name) and \
                        st.value.func.value.id == 'torch' and \
                        st.value.func.attr in ('flatten', 'squeeze',
                                               'unsqueeze') and \
                        st.value.args and \
                        isinstance(st.value.args[0], ast.Name):
                    _rfa6 = []
                    _rf6 = True
                    for _ra6 in st.value.args[1:]:
                        if isinstance(_ra6, ast.Constant):
                            _rfa6.append(_ra6.value)
                        else:
                            _rf6 = False
                            break
                    if _rf6:
                        _out.append(('stage',
                                     f'F_{st.value.func.attr}',
                                     _rfa6))
                        continue
                return None
            if not (isinstance(st, ast.Assign) and
                    len(st.targets) == 1 and
                    isinstance(st.targets[0], ast.Name) and
                    isinstance(st.value, ast.Call)):
                return None
            vc = st.value
            fn5 = vc.func
            if isinstance(fn5, ast.Attribute) and \
                    isinstance(fn5.value, ast.Name) and \
                    fn5.value.id == 'self' and len(vc.args) == 1:
                _mods5 = _unroll_sequential(fn5.attr)
                if _mods5 is None:
                    # plain module (nn.Linear/etc), not a Sequential:
                    # a single opaque stage, addressed as self.<attr>.
                    _out.append(('stage', fn5.attr))
                    _n_containers += 1
                    continue
                for _i5, _m5 in enumerate(_mods5):
                    # the forward-walk hop's PRE-FINALIZED tuples ride
                    # through: ops (('relu',)/('clamp',lo,hi)) and the
                    # branch brackets and pathed m_cat2 stages are
                    # emitted AS-IS (not module-specs to classify):
                    if _m5[0] in ('relu', 'clamp', 'sigmoid', 'tanh',
                                  'gelu', 'branch_fanout_begin',
                                  'branch_begin', 'branch_end',
                                  'branch_fanout_end'):
                        _out.append(_m5)
                        continue
                    if _m5[0] == 'stage' and len(_m5) > 2 and \
                            _m5[1] == 'm_cat2':
                        _out.append(_m5)
                        continue
                    if _m5[0] == 'stage' and len(_m5) > 3 and \
                            isinstance(_m5[3], str):
                        # a pathed stage from the class-walk: emit
                        # with the composed features-path directly:
                        _out.append(('stage',
                                     f'{fn5.attr}.{_m5[3]}'))
                        _n_containers += 1
                        continue
                    _cls5 = _m5[1]
                    # nested path if the spec carries one (helper-inlined:
                    # 4-tuple with runtime-true inner path), else flat _i5.
                    _path5 = (_m5[3] if len(_m5) > 3 else str(_i5))
                    if _cls5 in _CONT_ACT:
                        _out.append(_CONT_ACT[_cls5])
                    elif _cls5 in ('Linear', 'Conv1d', 'Conv2d',
                                   'Conv3d', 'BatchNorm1d',
                                   'BatchNorm2d', 'BatchNorm3d',
                                   'MaxPool2d', 'AvgPool2d',
                                   'AdaptiveAvgPool2d', 'Dropout',
                                   'Flatten'):
                        _out.append(('stage',
                                     f'{fn5.attr}.{_path5}'))
                    else:
                        return None
                _n_containers += 1
                continue
            _fname5 = (fn5.attr if isinstance(fn5, ast.Attribute)
                       else (fn5.id if isinstance(fn5, ast.Name)
                             else None))
            if _fname5 in ('flatten', 'squeeze', 'unsqueeze'):
                _fa5 = []
                for a in vc.args[1:]:
                    _cv5 = _const_eval(a)
                    if _cv5 is None:
                        return None
                    _fa5.append(_cv5)
                _out.append(('stage', f'F_{_fname5}', _fa5))
                continue
            if _fname5 in ('view', 'reshape'):
                # x.view(x.size(0), -1) -> m_view [__SIZE0__, -1]
                # (the single-stmt path emits exactly this; match the
                # x.size(0) arg-shape; anything else -> refuse honestly).
                _vargs = []
                for a in vc.args:
                    if isinstance(a, ast.Call) and \
                            isinstance(a.func, ast.Attribute) and \
                            a.func.attr == 'size' and \
                            len(a.args) == 1 and \
                            isinstance(a.args[0], ast.Constant):
                        _vargs.append(f'__SIZE{a.args[0].value}__')
                    elif isinstance(a, ast.Constant):
                        _vargs.append(a.value)
                    elif isinstance(a, ast.UnaryOp) and \
                            isinstance(a.op, ast.USub) and \
                            isinstance(a.operand, ast.Constant):
                        _vargs.append(-a.operand.value)
                    else:
                        return None
                _out.append(('stage', 'm_view', _vargs))
                continue
            return None
        if _n_containers >= 1 and \
                any(o[0] not in ('stage',) for o in _out):
            return _out
        return None

    if len(_fwd_body) > 1:
        _msynth = _multi_container_synth()
        if _msynth:
            return _msynth

    if len(_fwd_body) == 1 and \
            isinstance(_fwd_body[0], ast.Return) and \
            isinstance(_fwd_body[0].value, ast.Call) and \
            isinstance(_fwd_body[0].value.func, ast.Attribute) and \
            isinstance(_fwd_body[0].value.func.value, ast.Name) and \
            _fwd_body[0].value.func.value.id == 'self':
        _cattr = _fwd_body[0].value.func.attr
        _mods = _unroll_sequential(_cattr)
        if _mods:
            _synth = []
            _ok = True
            for _i2, _m2 in enumerate(_mods):
                _cls2 = _m2[1]
                _path2 = (_m2[3] if len(_m2) > 3 else str(_i2))
                if _cls2 in _CONT_ACT:
                    _synth.append(_CONT_ACT[_cls2])
                elif _cls2 in ('Linear', 'Conv1d', 'Conv2d',
                               'Conv3d', 'BatchNorm1d',
                               'BatchNorm2d', 'BatchNorm3d',
                               'MaxPool2d', 'AvgPool2d',
                               'AdaptiveAvgPool2d',
                               'Dropout', 'Flatten'):
                    # container-indexed stage: runtime-true nested
                    # path (helper-inlined) or flat index.
                    _synth.append(('stage',
                                   f'{_cattr}.{_path2}'))
                else:
                    _ok = False
                    break
            if _ok and any(o[0] != 'stage' for o in _synth):
                ops = _synth
                # the width from the last Linear before each
                # activation is derivable; leave widths None —
                # the runtime-width lesson (#38) covers launch:
                return ops

    # ═══ #34's RUNG (the fby lift): FLATTEN range-loops into the
    # stmt stream with markers — the body stmts then flow through
    # the SAME dispatch as straight-line code (no second walker,
    # no silent divergence). Guards: `for T in range(X)` (one arg,
    # no orelse, Name target); append-calls become loop_collect.
    # PRE-EXPAND fwd.body: (1) resolve evaluable If-statements (hasattr /
    # self.FLAG from init state) -> splice the taken branch; (2) normalize
    # AugAssign (x += y -> x = x + y) so the existing rails carry it. An
    # unevaluable If test -> leave the If in place (the downstream walk's
    # honest handling stands; never guess a branch).
    def _fwd_if_test(t):
        if isinstance(t, ast.Call) and isinstance(t.func, ast.Name) \
                and t.func.id == 'hasattr' and len(t.args) == 2 \
                and isinstance(t.args[0], ast.Name) \
                and t.args[0].id == 'self' \
                and isinstance(t.args[1], ast.Constant):
            _nm = t.args[1].value
            return (_nm in stage_classes) or (_nm in seq_attr) \
                or (_nm in attrs)
        if isinstance(t, ast.Attribute) \
                and isinstance(t.value, ast.Name) \
                and t.value.id == 'self':
            if t.attr in attrs and isinstance(attrs[t.attr], bool):
                return attrs[t.attr]
            if t.attr in _flag_exprs:
                return _eval_flag(_flag_exprs[t.attr])
        return None
    def _norm_aug(s):
        if isinstance(s, ast.AugAssign) and isinstance(s.target, ast.Name):
            return ast.Assign(
                targets=[ast.Name(id=s.target.id, ctx=ast.Store())],
                value=ast.BinOp(
                    left=ast.Name(id=s.target.id, ctx=ast.Load()),
                    op=s.op,
                    right=s.value))
        return s
    # THE METHOD-ENTRY PRE-PASS (the Swin rung, #29/#30): resolve
    # x = self.METHOD(x) where METHOD is a Model FunctionDef —
    # splice the method's body inline (docstring stripped; the
    # method's 'return EXPR' becomes 'x = EXPR'; the method's arg
    # name is rewritten to the caller's thread var). Depth-1 only
    # (methods calling methods splice on the next pass iteration —
    # bounded at 4 rounds; honest-stop beyond).
    def _splice_methods(body):
        out2 = []
        changed = False
        for st in body:
            if isinstance(st, ast.Assign) and \
                    len(st.targets) == 1 and \
                    isinstance(st.targets[0], ast.Name) and \
                    isinstance(st.value, ast.Call) and \
                    isinstance(st.value.func, ast.Attribute) and \
                    isinstance(st.value.func.value, ast.Name) and \
                    st.value.func.value.id == 'self' and \
                    st.value.func.attr in _cls_methods and \
                    len(st.value.args) == 1 and \
                    isinstance(st.value.args[0], ast.Name):
                _mdef = _cls_methods[st.value.func.attr]
                _caller_var = st.targets[0].id
                _arg_var = st.value.args[0].id
                # the method's own first arg after self:
                _params = [a.arg for a in _mdef.args.args
                           if a.arg != 'self']
                if len(_params) != 1:
                    out2.append(st)
                    continue
                _mparam = _params[0]
                # rename the method's param to the caller's arg,
                # then its returns to assigns of the caller target:
                class _Ren(ast.NodeTransformer):
                    def visit_Name(self, n):
                        if n.id == _mparam:
                            n.id = _arg_var
                        return n
                _new_sts = []
                for _ms in _mdef.body:
                    if isinstance(_ms, ast.Expr) and \
                            isinstance(_ms.value, ast.Constant):
                        continue    # docstring
                    _ms2 = _Ren().visit(
                        ast.parse(ast.unparse(_ms)).body[0])
                    if isinstance(_ms2, ast.Return):
                        if _ms2.value is None:
                            continue
                        _ms2 = ast.Assign(
                            targets=[ast.Name(id=_caller_var,
                                              ctx=ast.Store())],
                            value=_ms2.value)
                        _ms2 = ast.fix_missing_locations(
                            ast.copy_location(_ms2, st))
                    _new_sts.append(_ms2)
                out2.extend(_new_sts)
                changed = True
            else:
                out2.append(st)
        return out2, changed
    _mb = list(fwd.body)
    for _round in range(4):
        _mb, _ch = _splice_methods(_mb)
        if not _ch:
            break
    fwd = ast.FunctionDef(
        name=fwd.name, args=fwd.args, body=_mb,
        decorator_list=[], returns=None, type_comment=None)
    fwd = ast.fix_missing_locations(ast.copy_location(
        fwd, tree.body[0]))
    _fwd_expanded = []
    for _st in fwd.body:
        if isinstance(_st, ast.If):
            _tv = _fwd_if_test(_st.test)
            if _tv is True:
                _fwd_expanded.extend(_norm_aug(x) for x in _st.body)
                continue
            if _tv is False:
                _fwd_expanded.extend(_norm_aug(x) for x in _st.orelse)
                continue
            # unevaluable -> keep the If (downstream honest handling)
            _fwd_expanded.append(_st)
            continue
        _fwd_expanded.append(_norm_aug(_st))

    _stream = []
    for _st in _fwd_expanded:
        if isinstance(_st, ast.For) and \
                isinstance(_st.target, ast.Name) and \
                isinstance(_st.iter, ast.Attribute) and \
                isinstance(_st.iter.value, ast.Name) and \
                _st.iter.value.id == 'self':
            # for layer in self.layers (a ModuleList iteration —
            # the Swin shape): each element is a CUSTOM module
            # (BasicLayer) — walking them = the custom-module-
            # recursion rung, NOT this one. REFUSE BY NAME (the
            # doctrine: the census prints the true boundary):
            _stream.append(('MODULELIST_LOOP', _st.iter.attr,
                            _st.target.id, list(_st.body)))
            continue
        if isinstance(_st, ast.For) and \
                isinstance(_st.target, ast.Name) and \
                isinstance(_st.iter, ast.Call) and \
                isinstance(_st.iter.func, ast.Name) and \
                _st.iter.func.id == 'range' and \
                len(_st.iter.args) == 1 and not _st.orelse:
            _stream.append(('LOOP_BEGIN', _st.target.id,
                            ast.unparse(_st.iter.args[0])))
            _stream.extend(_st.body)
            _stream.append(('LOOP_END',))
        else:
            _stream.append(_st)

    _mflow_names = set()
    # RUNG-1 (Iyun's init-half): a DEDICATED pass populating _modulelists
    # from the ModuleList-append-loop in Model.__init__, BEFORE the forward
    # consult. Self-contained (top-level _ev_scalar not in scope here).
    def _populate_modulelists(_pinit_body=None, _penv_base=None,
                              _ppath_prefix=None):
        # RUNG-2: parameterized over (init-body, env, path-prefix) so it
        # runs inside ANY class-walk (BasicLayer's self.blocks), not just
        # Model. Defaults to Model's init.body + consts + no-prefix.
        if _pinit_body is None:
            if init is None:
                return
            _pinit_body = init.body
        # base env: init-arg/const values (embed_dim, depths, etc.)
        _penv = dict(consts) if _penv_base is None else dict(_penv_base)
        # self-attr values assigned in init (self.num_layers = len(depths))
        _pattrs = {}
        _pinstances = {}  # self.X -> (UserClass-name, {ctor-arg: value})
        def _resolve_instance_attr(icls, iargs, attr):
            # walk UserClass's init with iargs bound, find `attr = <expr>`
            # or `self.attr = <expr>`, return its _pev'd value (cross-module).
            _ucd2 = _user_classes.get(icls)
            if _ucd2 is None:
                return None
            _init2 = _ucd2.get('init')
            if _init2 is None:
                return None
            # child env: the ctor-args (img_size=224, patch_size=4, ...) +
            # the init's own local assigns as we walk.
            _ienv = dict(iargs)
            # to_2tuple(img_size) -> (img_size, img_size): a common helper;
            # emulate for the img_size/patch_size 2-tuple pattern.
            for _ist2 in _init2.body:
                if isinstance(_ist2, ast.Assign) and len(_ist2.targets) == 1:
                    _tg = _ist2.targets[0]
                    # local: name = <expr>  (patches_resolution = [..])
                    if isinstance(_tg, ast.Name):
                        _vv = _pev_env(_ist2.value, _ienv)
                        if _vv is not None:
                            _ienv[_tg.id] = _vv
                    # self.attr = <expr | name>  (self.patches_resolution = ..)
                    elif isinstance(_tg, ast.Attribute) and \
                            isinstance(_tg.value, ast.Name) and \
                            _tg.value.id == 'self':
                        if _tg.attr == attr:
                            # found the target attr
                            if isinstance(_ist2.value, ast.Name):
                                return _ienv.get(_ist2.value.id)
                            return _pev_env(_ist2.value, _ienv)
            # also: the attr may be a bare local (patches_resolution) if the
            # class exposes it via self.attr = patches_resolution earlier.
            return _ienv.get(attr)
        def _pev_env(e, env):
            # a thin wrapper: _pev with to_2tuple emulation for the
            # img_size/patch_size 2-tuple idiom (to_2tuple(x) -> (x, x)).
            if isinstance(e, ast.Call) and isinstance(e.func, ast.Name) \
                    and e.func.id == 'to_2tuple' and len(e.args) == 1:
                _tv = _pev(e.args[0], env)
                return (_tv, _tv) if _tv is not None else None
            if isinstance(e, ast.List):
                _lv = []
                for _el in e.elts:
                    _v = _pev(_el, env)
                    if _v is None:
                        return None
                    _lv.append(_v)
                return _lv
            if isinstance(e, ast.Subscript):
                _b = _pev(e.value, env)
                if _b is None:
                    return None
                _s = e.slice
                _sv = _s.value if isinstance(_s, ast.Index) else _s
                _iv = _pev(_sv, env)
                if _iv is None:
                    return None
                try:
                    return _b[_iv]
                except Exception:
                    return None
            return _pev(e, env)
        def _pev(e, env):
            # RUNG-3a' extension (on Iyun's _pev): ast.Tuple —
            # input_resolution=(pr0//2**i, pr1//2**i) — evaluates
            # element-wise (the H,W unpack needs the tuple in the
            # bind):
            if isinstance(e, ast.Tuple):
                _tv2 = []
                for _el2 in e.elts:
                    _ev2 = _pev(_el2, env)
                    if _ev2 is None:
                        return None
                    _tv2.append(_ev2)
                return tuple(_tv2)
            if isinstance(e, ast.List):
                # RUNG-3c: ast.List (self.padding = [ws-ss, ss, ...]):
                _lv3 = []
                for _el3 in e.elts:
                    _ev3 = _pev(_el3, env)
                    if _ev3 is None:
                        return None
                    _lv3.append(_ev3)
                return _lv3
            if isinstance(e, ast.Constant):
                return e.value
            if isinstance(e, ast.Name):
                return env.get(e.id, consts.get(e.id))
            if isinstance(e, ast.Attribute) and \
                    isinstance(e.value, ast.Name) and e.value.id == 'self':
                if e.attr in _pattrs:
                    return _pattrs[e.attr]
                return env.get(e.attr, consts.get(e.attr))
            if isinstance(e, ast.Attribute) and \
                    isinstance(e.value, ast.Attribute) and \
                    isinstance(e.value.value, ast.Name) and \
                    e.value.value.id == 'self':
                # CROSS-MODULE: self.X.ATTR where self.X = UserClass(args).
                # Walk UserClass's init with the ctor-arg env, read ATTR.
                _xname = e.value.attr
                _inst = _pinstances.get(_xname)
                if _inst is not None:
                    _icls, _iargs = _inst
                    return _resolve_instance_attr(_icls, _iargs, e.attr)
                return None
            if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub):
                _v = _pev(e.operand, env)
                return -_v if _v is not None else None
            if isinstance(e, ast.BinOp):
                _l, _r = _pev(e.left, env), _pev(e.right, env)
                if _l is None or _r is None:
                    return None
                try:
                    return {ast.Mult: lambda a, b: a * b,
                            ast.Add: lambda a, b: a + b,
                            ast.Sub: lambda a, b: a - b,
                            ast.FloorDiv: lambda a, b: a // b,
                            ast.Div: lambda a, b: a / b,
                            ast.Mod: lambda a, b: a % b,
                            ast.Pow: lambda a, b: a ** b}[type(e.op)](_l, _r)
                except Exception:
                    return None
            if isinstance(e, ast.Call) and isinstance(e.func, ast.Name):
                if e.func.id in ('int', 'float', 'round') and \
                        len(e.args) == 1:
                    _v = _pev(e.args[0], env)
                    if _v is None:
                        return None
                    try:
                        return {'int': int, 'float': float,
                                'round': round}[e.func.id](_v)
                    except Exception:
                        return None
                if e.func.id == 'len' and len(e.args) == 1:
                    _v = _pev(e.args[0], env)
                    return len(_v) if _v is not None and \
                        hasattr(_v, '__len__') else None
            if isinstance(e, ast.Subscript):
                _b = _pev(e.value, env)
                if _b is None:
                    return None
                _s = e.slice
                _sv = _s.value if isinstance(_s, ast.Index) else _s
                _iv = _pev(_sv, env)
                if _iv is None:
                    return None
                try:
                    return _b[_iv]
                except Exception:
                    return None
            if isinstance(e, ast.IfExp):
                _t = _pev(e.test, env)
                if _t is None:
                    return None
                return _pev(e.body, env) if _t else _pev(e.orelse, env)
            if isinstance(e, ast.Compare) and len(e.ops) == 1:
                _l = _pev(e.left, env)
                _r = _pev(e.comparators[0], env)
                if _l is None or _r is None:
                    return None
                try:
                    return {ast.Lt: lambda a, b: a < b,
                            ast.Gt: lambda a, b: a > b,
                            ast.LtE: lambda a, b: a <= b,
                            ast.GtE: lambda a, b: a >= b,
                            ast.Eq: lambda a, b: a == b,
                            ast.NotEq: lambda a, b: a != b}[
                        type(e.ops[0])](_l, _r)
                except Exception:
                    return None
            return None
        def _resolve_method_seq7(mname, call_args, env):
            # self.<mname>(call_args) returning nn.Sequential(...) -> a
            # ('__seq__', mname, [module-attrs], bind) spec for the forward-arm.
            _mfd = _cls_methods.get(mname)
            if _mfd is None:
                return None
            _mparams = [a.arg for a in _mfd.args.args[1:]]
            _menv = dict(env)
            for _pi, _pa in enumerate(call_args):
                if _pi < len(_mparams):
                    _pv = _pev(_pa, env)
                    if _pv is not None:
                        _menv[_mparams[_pi]] = _pv
            for _mst in _mfd.body:
                if isinstance(_mst, ast.Return) and \
                        isinstance(_mst.value, ast.Call) and \
                        isinstance(_mst.value.func, ast.Attribute) and \
                        _mst.value.func.attr == 'Sequential':
                    _mods = []
                    for _marg in _mst.value.args:
                        if isinstance(_marg, ast.Call) and \
                                isinstance(_marg.func, ast.Attribute):
                            _mods.append(_marg.func.attr)
                    _sbind = dict(_menv)
                    _sbind['__seq_mods__'] = _mods
                    _sbind['__seq_method__'] = mname
                    return ('__seq__:' + mname, _sbind)
            return None
        def _bind_ctor(call, env):
            # a Call to a user-class -> (class_name, bind_dict)
            if not (isinstance(call, ast.Call) and
                    isinstance(call.func, ast.Name) and
                    call.func.id in _user_classes):
                return None
            _cn = call.func.id
            _bd = {}
            _uci = _user_classes[_cn].get('init')
            _ups = ([a.arg for a in _uci.args.args[1:]] if _uci else [])
            for _pi, _pa in enumerate(call.args):
                if _pi < len(_ups):
                    _v = _pev(_pa, env)
                    if _v is not None:
                        _bd[_ups[_pi]] = _v
            for _kw in call.keywords:
                if _kw.arg is None:
                    continue
                _kv = _kw.value
                if isinstance(_kv, ast.Name) and _kv.id in _user_classes:
                    _bd[_kw.arg] = _kv.id
                elif isinstance(_kv, ast.IfExp):
                    _tv = _pev(_kv.test, env)
                    if _tv is not None:
                        _br = _kv.body if _tv else _kv.orelse
                        if isinstance(_br, ast.Name) and \
                                _br.id in _user_classes:
                            _bd[_kw.arg] = _br.id
                        elif isinstance(_br, ast.Constant):
                            _bd[_kw.arg] = _br.value
                        else:
                            _bv = _pev(_br, env)
                            if _bv is not None:
                                _bd[_kw.arg] = _bv
                else:
                    _v = _pev(_kv, env)
                    if _v is not None:
                        _bd[_kw.arg] = _v
            return (_cn, _bd)
        _pev_shared[0] = _pev
        # first pass: capture self-attr assigns (num_layers=len(depths) etc)
        for _st in _pinit_body:
            # RUNG-3c COMPLETION: capture LOCAL Name-assigns too
            # (patches_resolution = self.patch_embed.
            # patches_resolution — the cross-module read Iyun's
            # _pinstances-walk evaluates; the LOCAL var then feeds
            # the layers-populate's ctor-args):
            if isinstance(_st, ast.Assign) and \
                    len(_st.targets) == 1 and \
                    isinstance(_st.targets[0], ast.Name):
                _lv2 = _pev(_st.value, _penv)
                if _lv2 is not None:
                    _penv[_st.targets[0].id] = _lv2
                continue
            if isinstance(_st, ast.Assign) and len(_st.targets) == 1 and \
                    isinstance(_st.targets[0], ast.Attribute) and \
                    isinstance(_st.targets[0].value, ast.Name) and \
                    _st.targets[0].value.id == 'self':
                # track self.X = UserClass(ctor-args) for cross-module reads
                _sv2 = _st.value
                if isinstance(_sv2, ast.Call) and \
                        isinstance(_sv2.func, ast.Name) and \
                        _sv2.func.id in _user_classes:
                    _cargs = {}
                    _cinit = _user_classes[_sv2.func.id].get('init')
                    _cps = ([a.arg for a in _cinit.args.args[1:]]
                            if _cinit else [])
                    for _pi2, _pa2 in enumerate(_sv2.args):
                        if _pi2 < len(_cps):
                            _pv2 = _pev(_pa2, _penv)
                            if _pv2 is not None:
                                _cargs[_cps[_pi2]] = _pv2
                    for _kw2 in _sv2.keywords:
                        if _kw2.arg is not None:
                            _kv2 = _pev(_kw2.value, _penv)
                            if _kv2 is not None:
                                _cargs[_kw2.arg] = _kv2
                    _pinstances[_st.targets[0].attr] = (_sv2.func.id, _cargs)
                _v = _pev(_st.value, _penv)
                if _v is not None:
                    _pattrs[_st.targets[0].attr] = _v
        # second pass: find the ModuleList-append-loop / comprehension
        for _st in _pinit_body:
            # COMPREHENSION form: self.ATTR = nn.ModuleList(
            #   [UserClass(...) for i in range(N)])  (SwinMLPBlocks in
            #   BasicLayer's self.blocks). Evaluate range(N) from the env,
            #   per-i bind the ctor-args.
            if isinstance(_st, ast.Assign) and len(_st.targets) == 1 and \
                    isinstance(_st.targets[0], ast.Attribute) and \
                    isinstance(_st.targets[0].value, ast.Name) and \
                    _st.targets[0].value.id == 'self' and \
                    isinstance(_st.value, ast.Call) and \
                    isinstance(_st.value.func, ast.Attribute) and \
                    _st.value.func.attr == 'ModuleList' and \
                    len(_st.value.args) == 1 and \
                    isinstance(_st.value.args[0], ast.ListComp):
                _lc = _st.value.args[0]
                _mlattr = _st.targets[0].attr
                # the comprehension elt must be a UserClass call
                if not (isinstance(_lc.elt, ast.Call) and
                        isinstance(_lc.elt.func, ast.Name) and
                        _lc.elt.func.id in _user_classes):
                    continue
                # single generator: for VAR in range(N)
                if len(_lc.generators) != 1:
                    continue
                _gen = _lc.generators[0]
                if not (isinstance(_gen.target, ast.Name) and
                        isinstance(_gen.iter, ast.Call) and
                        isinstance(_gen.iter.func, ast.Name) and
                        _gen.iter.func.id == 'range'):
                    continue
                _cra = [_pev(a, _penv) for a in _gen.iter.args]
                if any(_r is None for _r in _cra):
                    continue
                try:
                    _crng = range(*[int(_r) for _r in _cra])
                except Exception:
                    continue
                _mlkey = (f'{_ppath_prefix}.{_mlattr}'
                          if _ppath_prefix else _mlattr)
                for _ci in _crng:
                    _cenv = dict(_penv)
                    _cenv[_gen.target.id] = _ci
                    _cpair = _bind_ctor(_lc.elt, _cenv)
                    if _cpair is not None:
                        _modulelists.setdefault(_mlkey, []).append(_cpair)
                continue
            # LOCAL-LIST form (#14 DenseNet): local=[]; for i in range(N):
            #   local.append(self._make_layer(args)); self.ATTR = ModuleList(local)
            if isinstance(_st, ast.Assign) and len(_st.targets) == 1 and \
                    isinstance(_st.targets[0], ast.Attribute) and \
                    isinstance(_st.targets[0].value, ast.Name) and \
                    _st.targets[0].value.id == 'self' and \
                    isinstance(_st.value, ast.Call) and \
                    isinstance(_st.value.func, ast.Attribute) and \
                    _st.value.func.attr == 'ModuleList' and \
                    len(_st.value.args) == 1 and \
                    isinstance(_st.value.args[0], ast.Name):
                _llname = _st.value.args[0].id
                _mlattr7 = _st.targets[0].attr
                _llfor = None
                for _s2 in _pinit_body:
                    if isinstance(_s2, ast.For) and \
                            isinstance(_s2.target, ast.Name) and \
                            isinstance(_s2.iter, ast.Call) and \
                            isinstance(_s2.iter.func, ast.Name) and \
                            _s2.iter.func.id == 'range':
                        for _b2 in _s2.body:
                            if isinstance(_b2, ast.Expr) and \
                                    isinstance(_b2.value, ast.Call) and \
                                    isinstance(_b2.value.func, ast.Attribute) \
                                    and _b2.value.func.attr == 'append' and \
                                    isinstance(_b2.value.func.value, ast.Name) \
                                    and _b2.value.func.value.id == _llname:
                                _llfor = _s2
                                break
                    if _llfor is not None:
                        break
                if _llfor is None:
                    continue
                _ra7 = [_pev(a, _penv) for a in _llfor.iter.args]
                if any(_r is None for _r in _ra7):
                    continue
                try:
                    _rng7 = range(*[int(_r) for _r in _ra7])
                except Exception:
                    continue
                _elt7 = None
                for _b2 in _llfor.body:
                    if isinstance(_b2, ast.Expr) and \
                            isinstance(_b2.value, ast.Call) and \
                            isinstance(_b2.value.func, ast.Attribute) and \
                            _b2.value.func.attr == 'append' and \
                            isinstance(_b2.value.func.value, ast.Name) and \
                            _b2.value.func.value.id == _llname and \
                            len(_b2.value.args) == 1:
                        _elt7 = _b2.value.args[0]
                        break
                if _elt7 is None:
                    continue
                _mlkey7 = (f'{_ppath_prefix}.{_mlattr7}'
                           if _ppath_prefix else _mlattr7)
                for _iv7 in _rng7:
                    _lenv7 = dict(_penv)
                    _lenv7[_llfor.target.id] = _iv7
                    # method-call element: self._make_layer(a,b) -> the returned
                    # Sequential-spec, params bound.
                    if isinstance(_elt7, ast.Call) and \
                            isinstance(_elt7.func, ast.Attribute) and \
                            isinstance(_elt7.func.value, ast.Name) and \
                            _elt7.func.value.id == 'self' and \
                            _elt7.func.attr in _cls_methods:
                        _mspec = _resolve_method_seq7(
                            _elt7.func.attr, _elt7.args, _lenv7)
                        if _mspec is not None:
                            _modulelists.setdefault(
                                _mlkey7, []).append(_mspec)
                        continue
                    _cp7 = _bind_ctor(_elt7, _lenv7)
                    if _cp7 is not None:
                        _modulelists.setdefault(_mlkey7, []).append(_cp7)
                continue
            # self.ATTR = nn.ModuleList()  (empty) -> track the attr
            # the following range-For appends to it.
            if isinstance(_st, ast.For) and \
                    isinstance(_st.target, ast.Name) and \
                    isinstance(_st.iter, ast.Call) and \
                    isinstance(_st.iter.func, ast.Name) and \
                    _st.iter.func.id == 'range':
                _ra = [_pev(a, _penv) for a in _st.iter.args]
                if any(_r is None for _r in _ra):
                    continue
                try:
                    _rng = range(*[int(_r) for _r in _ra])
                except Exception:
                    continue
                # scan the loop body for self.ATTR.append(X) + local binds
                for _iv in _rng:
                    _lenv = dict(_penv)
                    _lenv[_st.target.id] = _iv
                    _locals = {}
                    for _bst in _st.body:
                        # layer = BasicLayer(...)  (local-var capture)
                        if isinstance(_bst, ast.Assign) and \
                                len(_bst.targets) == 1 and \
                                isinstance(_bst.targets[0], ast.Name) and \
                                isinstance(_bst.value, ast.Call):
                            _locals[_bst.targets[0].id] = _bst.value
                        # self.ATTR.append(X)
                        elif isinstance(_bst, ast.Expr) and \
                                isinstance(_bst.value, ast.Call) and \
                                isinstance(_bst.value.func, ast.Attribute) \
                                and _bst.value.func.attr == 'append' and \
                                isinstance(_bst.value.func.value,
                                           ast.Attribute) and \
                                isinstance(_bst.value.func.value.value,
                                           ast.Name) and \
                                _bst.value.func.value.value.id == 'self' \
                                and len(_bst.value.args) == 1:
                            _mlattr = _bst.value.func.value.attr
                            _appx = _bst.value.args[0]
                            if isinstance(_appx, ast.Name):
                                _appx = _locals.get(_appx.id, _appx)
                            _pair = _bind_ctor(_appx, _lenv)
                            if _pair is not None:
                                _mlkey = (f'{_ppath_prefix}.{_mlattr}'
                                          if _ppath_prefix else _mlattr)
                                _modulelists.setdefault(
                                    _mlkey, []).append(_pair)
    _populate_modulelists()
    _fwd_listvars = {}  # forward list-vars: name -> [ref names]
    for stmt in _stream:
        if isinstance(stmt, tuple) and stmt[0] == 'MODULELIST_LOOP':
            # RUNG-1: the ModuleList-loop arm (for layer in
            # self.ATTR: x = layer(x)) — consult _modulelists; per
            # element: the #18 class-walk at path ATTR.N (each a
            # depth-1 entry — the top-loop insight: no guard raise).
            # _modulelists miss → the named refusal stands (honest
            # degradation; the loop-append fill is Iyun's half).
            _mlattr2 = stmt[1]
            _mltarget = stmt[2] if len(stmt) > 2 else None
            _mlbody = stmt[3] if len(stmt) > 3 else None
            _mlelems2 = _modulelists.get(_mlattr2)
            if _mlelems2 is None:
                ops.append(('unknown_call',
                            f'modulelist_loop:{_mlattr2}'))
                continue
            # RUNG #14: the STATEFUL-CAT-LOOP body-capture (the
            # _fwd_listvars socket): features=[x]; for layer in
            # self.layers: nf = layer(x); features.append(nf);
            # x = torch.cat(features, 1). Recognize the 3-stmt
            # body-form; per-iteration emit the element-stages +
            # the GROWING m_cat (operand-count = i+2):
            _scl_form = None
            if _mlbody is not None and len(_mlbody) == 3 and \
                    _mltarget is not None:
                _b1, _b2, _b3 = _mlbody
                if (isinstance(_b1, ast.Assign) and
                        len(_b1.targets) == 1 and
                        isinstance(_b1.targets[0], ast.Name) and
                        isinstance(_b1.value, ast.Call) and
                        isinstance(_b1.value.func, ast.Name) and
                        _b1.value.func.id == _mltarget and
                        len(_b1.value.args) == 1 and
                        isinstance(_b1.value.args[0],
                                   ast.Name) and
                        isinstance(_b2, ast.Expr) and
                        isinstance(_b2.value, ast.Call) and
                        isinstance(_b2.value.func,
                                   ast.Attribute) and
                        _b2.value.func.attr == 'append' and
                        isinstance(_b2.value.func.value,
                                   ast.Name) and
                        isinstance(_b3, ast.Assign) and
                        isinstance(_b3.value, ast.Call) and
                        isinstance(_b3.value.func,
                                   ast.Attribute) and
                        _b3.value.func.attr == 'cat'):
                    _lv9 = _b2.value.func.value.id
                    if _lv9 in _fwd_listvars:
                        _cd10 = 1
                        if len(_b3.value.args) == 2 and \
                                isinstance(_b3.value.args[1],
                                           ast.Constant):
                            _cd10 = _b3.value.args[1].value
                        _scl_form = (_lv9, _cd10)
            _ml_ok = True
            for _mli2, (_mlcls2, _mlbind2) in enumerate(_mlelems2):
                _mlpath2 = f'{_mlattr2}.{_mli2}'
                # the __seq__ element (Iyun's method-resolution):
                # the walked Sequential — emit its module-stages
                # directly (the proven fold: BN→ReLU→Conv→Drop):
                if isinstance(_mlcls2, str) and \
                        _mlcls2.startswith('__seq__:'):
                    _sqm = _mlbind2.get('__seq_mods__', [])
                    for _sqi, _sqcls in enumerate(_sqm):
                        # THE MISSION-FRAME FIX (Heath: never
                        # rest at torch-does-it-all): activation
                        # modules emit as OPS (the fusable
                        # islands — #18's exact form: relu
                        # kernels between conv/bn stages), NOT
                        # as stages:
                        if _sqcls in _UC_ACT_OPS:
                            ops.append(_UC_ACT_OPS[_sqcls])
                            continue
                        ops.append(('stage',
                                    f'{_mlattr2}.{_mli2}.{_sqi}'))
                    if _scl_form is not None:
                        # the growing cat: cat(features, 1) where
                        # features = [x-original] + all previous
                        # new_features + this one. Emit the
                        # m_cat_growing marker (the wrapper
                        # composes cap-list operands):
                        ops.append(('stage', 'm_cat_growing',
                                    [_scl_form[1], _mli2 + 2,
                                     _mlattr2]))
                    continue
                _mlint2 = _walk_user_class(_mlcls2, _mlbind2,
                                           _mlpath2)
                if _mlint2 is None:
                    ops.append(('unknown_call',
                                f'modulelist_element:{_mlpath2}:'
                                f'{_mlcls2}'))
                    _ml_ok = False
                    continue
                _mlamap2 = {}
                for _mlu in _mlint2:
                    _mlp5 = _mlu[3]
                    _mlat5 = _mlp5.split('.', 1)[1] \
                        if '.' in _mlp5 else _mlp5
                    _mlamap2[_mlat5] = (_mlu[1], _mlp5)
                _mlf2 = _walk_user_forward(_mlcls2, _mlbind2,
                                           _mlpath2, _mlamap2)
                if _mlf2 is not None:
                    for _mlo in _mlf2:
                        if _mlo[0] == 'stage' and len(_mlo) > 3 \
                                and isinstance(_mlo[3], str):
                            # the class-walk already composed the
                            # path ({outer_i}.<attr> with outer_i =
                            # 'layers.N') — use as-is:
                            ops.append(('stage', _mlo[3]))
                        else:
                            ops.append(_mlo)
                else:
                    for _mlu in _mlint2:
                        ops.append(('stage', _mlu[3]))
            continue
        if isinstance(stmt, tuple):
            if stmt[0] == 'LOOP_BEGIN':
                ops.append(('loop_begin', stmt[1], stmt[2]))
            else:
                ops.append(('loop_end',))
            continue
        # outputs.append(NAME) inside a flattened loop:
        if isinstance(stmt, ast.Expr) and \
                isinstance(stmt.value, ast.Call) and \
                isinstance(stmt.value.func, ast.Attribute) and \
                stmt.value.func.attr == 'append' and \
                isinstance(stmt.value.func.value, ast.Name) and \
                stmt.value.args and \
                isinstance(stmt.value.args[0], ast.Name):
            ops.append(('loop_collect', stmt.value.func.value.id,
                        stmt.value.args[0].id))
            continue
        # outputs = [] — the collector init (bookkeeping, no op):
        if isinstance(stmt, ast.Assign) and \
                isinstance(stmt.value, ast.List) and \
                not stmt.value.elts:
            continue
        # outputs = [b1, b2, b3, b4] — a list-of-Names (refs to prior
        # branch-stages, e.g. GoogleNet's Inception). Bind the name->refs
        # so a later torch.cat(outputs, dim) can resolve it. Bookkeeping;
        # the branch-stages already emitted (each branchN = self.branchN(x)).
        if isinstance(stmt, ast.Assign) and \
                len(stmt.targets) == 1 and \
                isinstance(stmt.targets[0], ast.Name) and \
                isinstance(stmt.value, ast.List) and \
                stmt.value.elts and \
                all(isinstance(e, ast.Name) for e in stmt.value.elts):
            _fwd_listvars[stmt.targets[0].id] = \
                [e.id for e in stmt.value.elts]
            continue
        # q, k, v = self.MOD(x).split(self.ATTR, dim=lit) — THE QKV
        # SPLIT (#43/#50): the inner module runs as a stage; the
        # split is a TORCH-STAGE forking the flow into named
        # outputs (the wrapper's hooks gate by stage-adjacency —
        # name-agnostic; the downstream per-name stmts are stages):
        if isinstance(stmt, ast.Assign) and \
                isinstance(stmt.targets[0], ast.Tuple) and \
                all(isinstance(e, ast.Name)
                    for e in stmt.targets[0].elts) and \
                isinstance(stmt.value, ast.Call) and \
                isinstance(stmt.value.func, ast.Attribute) and \
                stmt.value.func.attr == 'split' and \
                isinstance(stmt.value.func.value, ast.Call):
            _inner = stmt.value.func.value
            _ifn = _inner.func
            if isinstance(_ifn, ast.Attribute) and \
                    isinstance(_ifn.value, ast.Name) and \
                    _ifn.value.id == 'self' and \
                    len(_inner.args) == 1 and \
                    isinstance(_inner.args[0], ast.Name):
                # THE QKV SPLIT: record the REAL args so the wrapper-env
                # can replay q,k,v = src.split(split_size, dim=dim).
                # GUARD: first arg must be self.ATTR (split_size, a model
                # attr) and dim= a literal int; else refuse honestly.
                _sv = stmt.value
                _sz_attr = None
                if (_sv.args and isinstance(_sv.args[0], ast.Attribute)
                        and isinstance(_sv.args[0].value, ast.Name)
                        and _sv.args[0].value.id == 'self'):
                    _sz_attr = _sv.args[0].attr
                _dim = None
                for _kw in _sv.keywords:
                    if _kw.arg == 'dim' and isinstance(
                            _kw.value, ast.Constant) and isinstance(
                            _kw.value.value, int):
                        _dim = _kw.value.value
                if _sz_attr is None or _dim is None:
                    ops.append(('stage', 'm_split_unresolved',
                                [ast.dump(_sv)]))
                    continue
                ops.append(('stage', _ifn.attr))
                _names = [e.id for e in stmt.targets[0].elts]
                # record shape: [ATTR_NAME, DIM, *NAMES]
                ops.append(('stage', 'm_split',
                            [_sz_attr, _dim] + _names))
                _mflow_names.update(_names)
                current_rank[0] = None
                current_width[0] = None
                continue
        # M_FLOW: post-split named-flow self-stmts (#43/#50 QKV):
        #   k = k.view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        # The view/transpose args are RUNTIME names (B,T,C,self.n_head)
        # the literal-recorder can't capture, so we record the whole
        # stmt as (owner, unparsed_source) and the wrapper-env replays
        # it via eval against the bound flows. GUARD: only fires when a
        # named flow (a split output or a prior m_flow target) is
        # referenced, so the legacy single-thread x-path is untouched.
        if isinstance(stmt, ast.Assign) and \
                len(stmt.targets) == 1 and \
                isinstance(stmt.targets[0], ast.Name) and \
                _mflow_names:
            _owner = stmt.targets[0].id
            _refd = {n.id for n in ast.walk(stmt.value)
                     if isinstance(n, ast.Name)}
            # EXCLUDE the fused-activation kernel-island (F.relu(att),
            # F.gelu, torch.sigmoid, .relu() etc): it must fall through
            # to the real kernel detection, not be replayed as m_flow.
            _is_kernel_act = False
            _vv = stmt.value
            _act = {'relu', 'tanh', 'sigmoid', 'gelu', 'mish',
                    'silu', 'softmax', 'elu', 'selu'}
            if isinstance(_vv, ast.Call) and \
                    isinstance(_vv.func, ast.Attribute) and \
                    _vv.func.attr in _act:
                _is_kernel_act = True
            if (_refd & _mflow_names) and not _is_kernel_act:
                # references a live named flow -> replay-by-source
                ops.append(('stage', 'm_flow',
                            [_owner, ast.unparse(stmt.value)]))
                _mflow_names.add(_owner)
                current_rank[0] = None
                current_width[0] = None
                continue
        # A, B, _ = x.size() — the size tuple-unpack (#34): pure
        # SHAPE bookkeeping (the wrapper reads real shapes at run
        # time); record which name binds which dim, no op:
        if isinstance(stmt, ast.Assign) and \
                isinstance(stmt.targets[0], ast.Tuple) and \
                isinstance(stmt.value, ast.Call) and \
                isinstance(stmt.value.func, ast.Attribute) and \
                stmt.value.func.attr == 'size' and \
                isinstance(stmt.value.func.value, ast.Name) and \
                not stmt.value.args and \
                all(isinstance(e, ast.Name)
                    for e in stmt.targets[0].elts):
            _su_names = [e.id for e in stmt.targets[0].elts]
            ops.append(('size_unpack', stmt.value.func.value.id,
                        _su_names))
            # THE WIDTH-FROM-CONSTANTS BRIDGE (#43's rung): bind the
            # size-names to the BENCHMARK's input shape (get_inputs'
            # rand-args resolved through module-level constants) —
            # the benchmark's contract, statically readable:
            try:
                _mod_tree = ast.parse(open(path).read())
                _mconsts = {}
                for _mn in _mod_tree.body:
                    if isinstance(_mn, ast.Assign) and \
                            isinstance(_mn.targets[0], ast.Name):
                        try:
                            _mconsts[_mn.targets[0].id] = \
                                ast.literal_eval(_mn.value)
                        except Exception:
                            pass
                _gi = next((n for n in _mod_tree.body
                            if isinstance(n, ast.FunctionDef) and
                            n.name == 'get_inputs'), None)
                _shape_vals = None
                if _gi is not None:
                    for _gs in ast.walk(_gi):
                        if isinstance(_gs, ast.Call) and \
                                isinstance(_gs.func, ast.Attribute) \
                                and _gs.func.attr in ('rand', 'randn'):
                            _sv = []
                            for _ga in _gs.args:
                                if isinstance(_ga, ast.Name) and \
                                        _ga.id in _mconsts:
                                    _sv.append(_mconsts[_ga.id])
                                elif isinstance(_ga, ast.Constant):
                                    _sv.append(_ga.value)
                                else:
                                    _sv = None
                                    break
                            if _sv:
                                _shape_vals = _sv
                                break
                if _shape_vals and \
                        len(_shape_vals) == len(_su_names):
                    for _nm2, _vv in zip(_su_names, _shape_vals):
                        if _nm2 != '_':
                            ops.append(('size_binding', _nm2, _vv))
            except Exception:
                pass
            continue
        if isinstance(stmt, ast.Return) and stmt.value is not None \
                and isinstance(stmt.value, ast.Call):
            _rv = stmt.value
            # return torch.stack(NAME, dim=lit) — the loop-collector
            # materialization (#34's tail): a torch-stage:
            _rfn = _rv.func
            _rdim = None
            for kw in _rv.keywords:
                if kw.arg == 'dim' and isinstance(kw.value,
                                                  ast.Constant):
                    _rdim = kw.value.value
            if len(_rv.args) == 2 and isinstance(_rv.args[1],
                                                 ast.Constant):
                _rdim = _rv.args[1].value
            if isinstance(_rfn, ast.Attribute) and \
                    isinstance(_rfn.value, ast.Name) and \
                    _rfn.value.id == 'torch' and \
                    _rfn.attr == 'stack' and _rv.args and \
                    isinstance(_rv.args[0], ast.Name) and \
                    _rdim is not None:
                ops.append(('stage', 'm_stack', [_rv.args[0].id,
                                                 _rdim]))
                continue
            # BRANCH-CAT: return torch.cat([branchA, branchB, ...], dim)
            # where each branch is a module-application chain from the
            # shared thread (self.OUTER(self.INNER(x)) or self.MOD(x)).
            # Emit each branch's stages, then m_cat2 (the branch fan-out).
            # A NEW arm, sibling to m_stack, before the return_call flag.
            # branch-cat of a LIST-VAR: return torch.cat(outputs, dim)
            # where outputs = [b1,b2,b3,b4] (pre-bound branch-refs, e.g.
            # GoogleNet). The branch-stages already emitted; wrap them in
            # fan-out brackets (arity generalizes from #17's 2 to N) + m_cat2.
            if isinstance(_rfn, ast.Attribute) and \
                    isinstance(_rfn.value, ast.Name) and \
                    _rfn.value.id == 'torch' and \
                    _rfn.attr == 'cat' and _rv.args and \
                    isinstance(_rv.args[0], ast.Name) and \
                    _rv.args[0].id in _fwd_listvars and \
                    _rdim is not None:
                _refs = _fwd_listvars[_rv.args[0].id]
                # REFS-ONLY (Bocher's ruling): the branches are PRE-BOUND
                # (branchN = self.branchN(x) already emitted as stages that
                # unroll via the container rails; each roots at x by runtime
                # capture). The topology rides the NAMES — no brackets
                # needed. Carry the tail-refs IN the m_cat2 op so the gate
                # reads them directly. The refs name the branch TAILS
                # (branch1x1 / branch3x3.1 / ... — get_submodule-resolvable).
                # resolve each branch-ref to its TAIL stage-name (the
                # branch may have unrolled to ref.0/.1 — the tail is the
                # LAST emitted stage matching ref or ref.*). The gate hooks
                # the get_submodule-resolvable terminal.
                _tails = []
                for _rname in _refs:
                    _tail = _rname
                    for _o in ops:
                        if _o[0] == 'stage' and len(_o) > 1 and \
                                (_o[1] == _rname or
                                 _o[1].startswith(_rname + '.')):
                            _tail = _o[1]
                    _tails.append(_tail)
                ops.append(('stage', 'm_cat2', [_rdim, _tails]))
                continue
            if isinstance(_rfn, ast.Attribute) and \
                    isinstance(_rfn.value, ast.Name) and \
                    _rfn.value.id == 'torch' and \
                    _rfn.attr == 'cat' and _rv.args and \
                    isinstance(_rv.args[0], (ast.List, ast.Tuple)) and \
                    _rdim is not None:
                _branches = _rv.args[0].elts
                # a branch-expr lifter: self.OUTER(self.INNER(x)) or
                # self.MOD(x) -> [stage names]. Returns None if the
                # branch isn't a clean module-chain (honest refusal).
                def _lift_branch(e):
                    _st = []
                    _cur = e
                    # unwrap nested self-module calls, outermost-first;
                    # collect innermost-first so stages emit in order.
                    _chain = []
                    while isinstance(_cur, ast.Call) and \
                            isinstance(_cur.func, ast.Attribute) and \
                            isinstance(_cur.func.value, ast.Name) and \
                            _cur.func.value.id == 'self' and \
                            len(_cur.args) == 1:
                        _chain.append(_cur.func.attr)
                        _cur = _cur.args[0]
                    # innermost arg must be a bare Name/self-Attr
                    # (the shared thread x)
                    if not (isinstance(_cur, ast.Name) or
                            (isinstance(_cur, ast.Attribute) and
                             isinstance(_cur.value, ast.Name) and
                             _cur.value.id == 'self')):
                        return None
                    if not _chain:
                        return None
                    # emit innermost-first (reverse the outermost-first
                    # collection): each module-attr becomes a stage
                    # (unless it's an activation op-kind -> the op).
                    for _attr in reversed(_chain):
                        if _attr in module_ops:
                            _st.append(('op', module_ops[_attr]))
                        else:
                            _st.append(('stage', _attr))
                    return _st
                _all = []
                _ok = True
                for _b in _branches:
                    _lb = _lift_branch(_b)
                    if _lb is None:
                        _ok = False
                        break
                    _all.append(_lb)
                if _ok and _all:
                    # EXPLICIT fan-out brackets (Bocher: keep them,
                    # strictly better for #6/deeper/expr-tails; ride
                    # as META so segments render clean).
                    ops.append(('branch_fanout_begin', len(_all), _rdim))
                    for _bi, _bstages in enumerate(_all):
                        ops.append(('branch_begin', _bi))
                        for _kind, _val in _bstages:
                            if _kind == 'stage':
                                ops.append(('stage', _val))
                            else:
                                ops.append(_val)
                        ops.append(('branch_end', _bi))
                    ops.append(('stage', 'm_cat2', [_rdim]))
                    ops.append(('branch_fanout_end',))
                    continue
            # return torch.flatten(x, D) / squeeze / unsqueeze — the
            # shape-fn RETURN arm (#18's Model.forward tail; the
            # mid-flow F_ arm's Return-position sibling):
            if isinstance(_rfn, ast.Attribute) and \
                    isinstance(_rfn.value, ast.Name) and \
                    _rfn.value.id == 'torch' and \
                    _rfn.attr in ('flatten', 'squeeze',
                                  'unsqueeze') and \
                    _rv.args and isinstance(_rv.args[0], ast.Name):
                _rfa = []
                _rf_ok = True
                for _ra in _rv.args[1:]:
                    _rcv = _const_eval(_ra)
                    if _rcv is None:
                        _rf_ok = False
                        break
                    _rfa.append(_rcv)
                if _rf_ok:
                    ops.append(('stage', f'F_{_rfn.attr}', _rfa))
                    continue
            # #28 (the vocabulary rung): return self.MODULE(NAME) —
            # a module-application in the return (mlp_head): emit
            # the stage and end the walk (the thread returns):
            if isinstance(_rfn, ast.Attribute) and \
                    isinstance(_rfn.value, ast.Name) and \
                    _rfn.value.id == 'self' and \
                    len(_rv.args) == 1 and \
                    isinstance(_rv.args[0], ast.Name):
                ops.append(('stage', _rfn.attr))
                continue
            # any OTHER call in the return = real work the walk
            # didn't claim — flag it (the silent-drop lesson):
            ops.append(('unknown_call', 'return_call'))
            continue
        if not isinstance(stmt, ast.Assign):
            continue
        v = stmt.value
        # the chain-thread tracker update (runs for EVERY assign,
        # before the handlers — the PREVIOUS stmt's target became
        # the thread if the value consumed the thread; heuristic:
        # any assignment whose VALUE references the current thread
        # moves the thread to its target):
        # (apply the PREVIOUS stmt's thread move first)
        if _pending.get('t'):
            _last_target[0] = _pending['t']
        try:
            _refs = {n.id for n in ast.walk(v)
                     if isinstance(n, ast.Name)}
            # a bare-Name alias (original_x = x — the SAVE form)
            # does NOT move the thread: the save COPIES the value,
            # the chain continues in the source var (the #70/#40/
            # #20 regression: the save stole the thread and every
            # later chain-op mismatched):
            _core = v
            while isinstance(_core, ast.Call) and \
                    isinstance(_core.func, ast.Attribute) and \
                    _core.func.attr in ('clone', 'detach'):
                _core = _core.func.value
            if isinstance(_core, ast.Name):
                # a bare Name or x.clone().detach() chain = a SAVE
                # (aliases/copies; does NOT consume the thread):
                _pending['t'] = None
            elif _last_target[0] in _refs and \
                    isinstance(stmt.targets[0], ast.Name):
                _pending['t'] = stmt.targets[0].id
            else:
                _pending['t'] = None
        except Exception:
            _pending['t'] = None
        # THE PEEL (the #55/#98 double/missed-walk fix): a statement
        # like self.pool(x.unsqueeze(1)).squeeze(1) must process
        # innermost-out — inner shape-op, then the module, then the
        # trailing shape-op — each EXACTLY ONCE. Unwrap trailing
        # .shapeop(...) wrappers; process nested arg shape-ops first:
        _SHAPE_OPS = ('squeeze', 'unsqueeze', 'view', 'reshape',
                      'flatten', 'permute', 'transpose',
                      'contiguous')
        _post_shape = []
        while (isinstance(v, ast.Call)
               and isinstance(v.func, ast.Attribute)
               and isinstance(v.func.value, ast.Call)
               and v.func.attr in _SHAPE_OPS):
            try:
                _pargs = [ast.literal_eval(a) for a in v.args]
            except Exception:
                _pargs = None
            _post_shape.append((v.func.attr, _pargs))
            v = v.func.value
        if (isinstance(v, ast.Call) and len(v.args) == 1
                and isinstance(v.args[0], ast.Call)
                and isinstance(v.args[0].func, ast.Attribute)
                and isinstance(v.args[0].func.value, ast.Name)
                and v.args[0].func.value.id == 'x'
                and v.args[0].func.attr in _SHAPE_OPS):
            _in = v.args[0].func.attr
            _in_args = []
            try:
                _in_args = [ast.literal_eval(a)
                            for a in v.args[0].args]
            except Exception:
                _in_args = None
            ops.append(('stage', f'm_{_in}', _in_args))
            if _in == 'unsqueeze' and current_rank[0] is not None:
                current_rank[0] += 1
            elif _in == 'squeeze' and current_rank[0] is not None:
                current_rank[0] -= 1
            else:
                current_rank[0] = None
            v = ast.copy_location(
                ast.Call(func=v.func, args=[v.args[0].func.value],
                         keywords=v.keywords), v)

        def _apply_post_shape():
            for _op, _pargs in reversed(_post_shape):
                ops.append(('stage', f'm_{_op}', _pargs))
                if _op == 'unsqueeze' and current_rank[0] is not None:
                    current_rank[0] += 1
                elif _op == 'squeeze' and current_rank[0] is not None:
                    current_rank[0] -= 1
                else:
                    current_rank[0] = None
        # x = self.MODULE(x): activation-module -> op; else stage marker:
        if isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) \
           and isinstance(v.func.value, ast.Name) and v.func.value.id == 'self':
            # THE NESTED-CALL UNWRAP (keystone item 2b — replaces
            # the honesty guard): self.OUTER(self.INNER(name)) where
            # INNER is a self-module and the innermost arg is a bare
            # Name/self-Attr → emit the INNER stage FIRST, then let
            # the OUTER process normally (the arg rewritten to the
            # inner's output). ONE level only; anything deeper still
            # flags (honest refusal):
            if v.args and isinstance(v.args[0], ast.Call):
                _inner = v.args[0]
                _ifn = _inner.func
                _inner_is_self_mod = (
                    isinstance(_ifn, ast.Attribute) and
                    isinstance(_ifn.value, ast.Name) and
                    _ifn.value.id == 'self' and
                    len(_inner.args) == 1 and
                    (isinstance(_inner.args[0], ast.Name) or
                     (isinstance(_inner.args[0], ast.Attribute) and
                      isinstance(_inner.args[0].value, ast.Name) and
                      _inner.args[0].value.id == 'self')) and
                    _ifn.attr not in module_ops)
                if _inner_is_self_mod:
                    ops.append(('stage', _ifn.attr))
                    _icls = stage_classes.get(_ifn.attr)
                    if _icls == 'Linear':
                        _ishape = param_shapes.get(_ifn.attr)
                        current_width[0] = None
                        current_rank[0] = 2
                    v = ast.copy_location(
                        ast.Call(func=v.func, args=[_inner.args[0]],
                                 keywords=v.keywords), v)
                else:
                    ops.append(('unknown_call',
                                f'nested_call_arg_{v.func.attr}'))
                    continue
            if v.func.attr in module_ops:
                op = module_ops[v.func.attr]
                if op[0] == 'reduction':
                    d = module_red_dims.get(v.func.attr)
                    if not _axis_is_last(d, current_rank[0]):
                        if op[1] == 'softmax' and d == 1 and \
                           current_rank[0] is not None and \
                           current_rank[0] >= 3:
                            ops.append(('reduction_chan', 'softmax', current_chan[0],
                                (__import__('math').prod(current_spatial[0])
                                 if current_spatial[0] else None)))
                            continue
                        ops.append(('unknown_call',
                                    'non_lastdim_reduction'))
                        continue
                    # lastdim module-reduction: carry the PROVEN width
                    # (#84's nn.Softmax(dim=1) on 2D — the width comes
                    # from the Linear stage; without it the emitter
                    # can't select a regime and refuses):
                    ops.append(('reduction', op[1], current_width[0]))
                    continue
                if op[0] == 'avgpool3d_k2':
                    # carry the module attr so the guard can revert
                    # to the true stage form in mixed chains (#50):
                    ops.append(('avgpool3d_k2', v.func.attr))
                else:
                    ops.append(op)
            else:
                # a Sequential-attr (x = self.branchN(x) where branchN is
                # nn.Sequential): UNROLL to nested .0/.1 stages (the fold
                # islands), the #19 rails. Else emit the flat stage.
                _seqattr = v.func.attr
                if stage_classes.get(_seqattr) == 'Sequential':
                    _seqmods = _unroll_sequential(_seqattr)
                    if _seqmods is not None:
                        for _sm in _seqmods:
                            if _sm[0] == 'op':
                                ops.append(_sm[1])
                            elif _sm[0] == 'clamp':
                                ops.append(_sm)
                            else:
                                # ('stage', cls, args, path) -> nested path
                                _spath = _sm[3] if len(_sm) > 3 else None
                                if _spath is not None:
                                    ops.append(('stage',
                                                f'{_seqattr}.{_spath}'))
                                else:
                                    ops.append(('stage', _seqattr))
                        # thread continues; skip the flat-stage emit below
                        current_width[0] = None
                        current_rank[0] = None
                        continue
                # #28: the SUBSCRIPT-ARG (x = self.to_cls_token(
                # x[:, 0]) — the [:,0] SELECT must emit BEFORE the
                # stage; silently dropping it was the silent-skip
                # class):
                if v.args and isinstance(v.args[0],
                                         ast.Subscript):
                    _sub9 = v.args[0].slice
                    _sel_ok = False
                    if isinstance(_sub9, ast.Tuple) and \
                            len(_sub9.elts) == 2 and \
                            isinstance(_sub9.elts[0],
                                       ast.Slice) and \
                            _sub9.elts[0].lower is None and \
                            _sub9.elts[0].upper is None and \
                            isinstance(_sub9.elts[1],
                                       ast.Constant):
                        ops.append(('stage', 'm_select',
                                    [1, _sub9.elts[1].value]))
                        _sel_ok = True
                    if not _sel_ok:
                        ops.append(('unknown_call',
                                    'subscript_arg'))
                        continue
                ops.append(('stage', v.func.attr))
                cls = stage_classes.get(v.func.attr)
                if v.func.attr in stage_spatial and \
                   current_spatial[0] is not None:
                    kind, kk, ssz, pp, opad = stage_spatial[v.func.attr]
                    try:
                        n_sp = (len(current_spatial[0])
                                if kind != 'adaptive' else 0)
                        def _bc(z):
                            return (list(z) if isinstance(z, (tuple, list))
                                    else [z] * n_sp)
                        if kind == 'adaptive':
                            current_spatial[0] = list(kk)
                        elif kind == 'conv':
                            kkl, ssl, ppl = _bc(kk), _bc(ssz), _bc(pp)
                            current_spatial[0] = [
                                (d + 2 * ppl[j] - kkl[j]) // ssl[j] + 1
                                for j, d in enumerate(current_spatial[0])]
                        elif kind == 'convT':
                            kkl, ssl, ppl = _bc(kk), _bc(ssz), _bc(pp)
                            opl = _bc(opad)
                            current_spatial[0] = [
                                (d - 1) * ssl[j] - 2 * ppl[j] + kkl[j]
                                + opl[j]
                                for j, d in enumerate(current_spatial[0])]
                        elif kind == 'pool':
                            kkl, ssl, ppl = _bc(kk), _bc(ssz), _bc(pp)
                            current_spatial[0] = [
                                (d + 2 * ppl[j] - kkl[j]) // ssl[j] + 1
                                for j, d in enumerate(current_spatial[0])]
                    except Exception:
                        current_spatial[0] = None
                elif v.func.attr not in stage_spatial and \
                     stage_classes.get(v.func.attr) not in (
                         'GroupNorm', 'BatchNorm1d', 'BatchNorm2d',
                         'BatchNorm3d', 'InstanceNorm2d', 'InstanceNorm3d',
                         'LayerNorm', 'Dropout', 'Linear', 'Gemm'):
                    pass  # unknown spatial effect for non-tracked stages:
                if v.func.attr in stage_classes and \
                   v.func.attr not in stage_spatial and \
                   stage_classes.get(v.func.attr) in (
                       'Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d',
                       'ConvTranspose2d', 'ConvTranspose3d', 'MaxPool1d',
                       'MaxPool2d', 'MaxPool3d', 'AvgPool1d', 'AvgPool2d',
                       'AvgPool3d'):
                    current_spatial[0] = None  # spatial stage, params unproven
                if v.func.attr in stage_chans:
                    current_chan[0] = stage_chans[v.func.attr]
                elif cls in ('MaxPool1d', 'MaxPool2d', 'MaxPool3d',
                             'AvgPool1d', 'AvgPool2d', 'AvgPool3d',
                             'AdaptiveAvgPool1d', 'AdaptiveAvgPool2d',
                             'AdaptiveAvgPool3d', 'AdaptiveMaxPool1d',
                             'AdaptiveMaxPool2d', 'AdaptiveMaxPool3d',
                             'Dropout', 'GroupNorm', 'BatchNorm1d',
                             'BatchNorm2d', 'BatchNorm3d', 'InstanceNorm2d',
                             'InstanceNorm3d', 'LayerNorm'):
                    pass  # channel-preserving stages keep current_chan
                else:
                    current_chan[0] = None
                if v.func.attr in stage_widths:
                    current_width[0] = stage_widths[v.func.attr]
                elif cls not in ('GroupNorm', 'BatchNorm1d', 'LayerNorm',
                                 'Dropout', 'Identity'):
                    current_width[0] = None
                RANK_PRESERVING = {'GroupNorm', 'BatchNorm1d', 'BatchNorm2d',
                                   'BatchNorm3d', 'LayerNorm', 'InstanceNorm1d',
                                   'InstanceNorm2d', 'InstanceNorm3d',
                                   'Dropout', 'AvgPool1d', 'AvgPool2d',
                                   'AvgPool3d', 'MaxPool1d', 'MaxPool2d',
                                   'MaxPool3d'}
                if cls in RANK_PRESERVING:
                    pass   # keep current_rank
                else:
                    current_rank[0] = STAGE_RANKS.get(cls) if cls else None
            _apply_post_shape()
            continue
        # ORIG = x — plain-name residual save (#70's form):
        if isinstance(stmt.targets[0], ast.Name) and \
           stmt.targets[0].id != 'x' and isinstance(v, ast.Name) and \
           v.id == 'x':
            ops.append(('save', stmt.targets[0].id))
            continue
        # x = ORIG.clone()... — residual save (target name captured):
        if isinstance(stmt.targets[0], ast.Name) and \
           stmt.targets[0].id != 'x' and isinstance(v, ast.Call):
            fn = v.func
            # x.clone() or x.clone().detach() chains:
            base = fn
            while isinstance(base, ast.Attribute) and \
                  base.attr in ('clone', 'detach'):
                base = base.value
                if isinstance(base, ast.Call):
                    base = base.func
            if isinstance(base, ast.Name) and base.id == 'x':
                ops.append(('save', stmt.targets[0].id))
                continue
        # x - x.mean(dim=K, keepdim=True) when width==1: mean of a
        # single element IS the element → x − x ≡ 0 EXACTLY (#80's
        # degenerate identity — DERIVED, not assumed; the census entry
        # carries PASS-DEGENERATE framing):
        if isinstance(v, ast.BinOp) and isinstance(v.op, ast.Sub) and \
           isinstance(v.left, ast.Name) and v.left.id == 'x' and \
           isinstance(v.right, ast.Call) and \
           isinstance(v.right.func, ast.Attribute) and \
           v.right.func.attr == 'mean' and \
           isinstance(v.right.func.value, ast.Name) and \
           v.right.func.value.id == 'x' and current_width[0] == 1:
            kd = any(kw.arg == 'keepdim' and
                     getattr(kw.value, 'value', False) is True
                     for kw in v.right.keywords)
            if kd:
                ops.append(('zero_fold',))
                continue
        # x = x * sigmoid(x + 3) / 6 — the sigmoid-hardswish variant
        # (#58): Div(Mult(x, sigmoid(x+3)), 6) — lift as ONE fused op
        # (caught by the silent-drop audit: the old scalar branch
        # matched /6 and DROPPED the x*sigmoid(x+3) — bug-class 9):
        if isinstance(v, ast.BinOp) and isinstance(v.op, ast.Div) and \
           scalar_of(v.right) == 6.0 and \
           isinstance(v.left, ast.BinOp) and \
           isinstance(v.left.op, ast.Mult) and \
           isinstance(v.left.left, ast.Name) and v.left.left.id == 'x' and \
           isinstance(v.left.right, ast.Call):
            sf = v.left.right.func
            sname = sf.attr if isinstance(sf, ast.Attribute) else \
                    (sf.id if isinstance(sf, ast.Name) else None)
            if sname == 'sigmoid' and v.left.right.args and \
               isinstance(v.left.right.args[0], ast.BinOp) and \
               isinstance(v.left.right.args[0].op, ast.Add) and \
               scalar_of(v.left.right.args[0].right) == 3.0:
                ops.append(('xsig3_div6',))
                continue
        # x = x * S / x = x / S / x = x + S / x = x - S
        # (v.left MUST be plain x — a nested left expression through
        # the scalar branch would SILENT-DROP it; bug-class 9):
        if isinstance(v, ast.BinOp):
            s = scalar_of(v.right)
            if s is not None and isinstance(v.left, ast.Name) and \
               v.left.id == 'x':
                kind = {ast.Mult: 'multiply', ast.Div: 'divide',
                        ast.Add: 'add_scalar', ast.Sub: 'subtract'}.get(type(v.op))
                if kind:
                    ops.append((kind, float(s)))
                    continue
            elif s is not None:
                ops.append(('unknown_call', 'binop_nested_left'))
                continue
            # x OP self.ATTR.view(...) — channel-broadcast via explicit
            # view (#89's subtract.view(1,-1,1,1,1)) — unwrap to the
            # param Attribute (the view IS the chan-broadcast):
            vr = v.right
            if isinstance(vr, ast.Call) and \
               isinstance(vr.func, ast.Attribute) and \
               vr.func.attr == 'view' and \
               isinstance(vr.func.value, ast.Attribute) and \
               isinstance(vr.func.value.value, ast.Name) and \
               vr.func.value.value.id == 'self':
                view_args = []
                for a in vr.args:
                    try:
                        view_args.append(ast.literal_eval(a))
                    except Exception:
                        view_args.append(None)
                # (1, -1, 1...) = channel-broadcast:
                if view_args and view_args[0] == 1 and view_args[1] == -1 \
                   and all(z == 1 for z in view_args[2:]):
                    kind = {ast.Mult: 'mul_param_chan',
                            ast.Add: 'add_param_chan',
                            ast.Sub: 'sub_param_chan'}.get(type(v.op))
                    if kind:
                        ops.append((kind, vr.func.value.attr))
                        continue
            # x = x + self.ATTR / x = x * self.ATTR — PARAM-tensor ops:
            # COMMUTATIVE FORMS: #84 writes `self.scale * x` -- the param on the
            # LEFT. mul and add commute, so the same lift is valid. sub and div
            # DO NOT: `self.b - x` is not `x - self.b`, so those stay refused
            # in reversed form rather than lifted wrongly.
            if (not (isinstance(v.right, ast.Attribute)
                     and isinstance(v.right.value, ast.Name)
                     and v.right.value.id == 'self')) \
               and isinstance(v.left, ast.Attribute) \
               and isinstance(v.left.value, ast.Name) \
               and v.left.value.id == 'self' \
               and isinstance(v.op, (ast.Mult, ast.Add)):
                v = ast.BinOp(left=v.right, op=v.op, right=v.left)
            if isinstance(v.right, ast.Attribute) and \
               isinstance(v.right.value, ast.Name) and \
               v.right.value.id == 'self' and \
               v.right.attr not in attrs:
                kind = {ast.Mult: 'mul_param', ast.Add: 'add_param',
                        ast.Sub: 'sub_param'}.get(type(v.op))
                if kind:
                    if v.right.attr in scalar_params:
                        ops.append((kind + '_scalar', v.right.attr))
                    elif v.right.attr in channel_params:
                        ops.append((kind + '_chan', v.right.attr))
                    else:
                        ops.append((kind, v.right.attr))
                    continue
            # x = x OP ARG2 — second-input elementwise (two-input model):
            if arg2 is not None and isinstance(v.right, ast.Name) \
               and v.right.id == arg2 and isinstance(v.left, ast.Name) \
               and v.left.id == 'x':
                ikind = {ast.Add: 'add_input2', ast.Mult: 'mul_input2',
                         ast.Sub: 'sub_input2'}.get(type(v.op))
                if ikind:
                    ops.append((ikind,))
                    continue
            # SAVED OP chain — the REVERSED residual (#92's
            # x_res = x_conv + x_hard_swish: left = an EARLIER
            # intermediate, right = the current chain thread):
            if isinstance(v.left, ast.Name) and \
               isinstance(v.right, ast.Name) and \
               v.right.id == _last_target[0] and \
               v.left.id != _last_target[0] and \
               isinstance(v.op, ast.Add):
                ops.append(('add_saved', v.left.id))
                if isinstance(stmt.targets[0], ast.Name):
                    _last_target[0] = stmt.targets[0].id
                continue
            # THREAD = THREAD + self.MODULE(SAVED) — the module-
            # shortcut residual (#25's ShuffleNet form: 'out +=
            # self.shortcut(x)'). REAL vocabulary (a stage applied
            # to a SAVED value + add — the branch-cat cousin); NOT
            # yet expressible: refuse by NAME (not the generic
            # binop marker) so the census prints the true reason:
            if isinstance(v.right, ast.Call) and \
               isinstance(v.right.func, ast.Attribute) and \
               isinstance(v.right.func.value, ast.Name) and \
               v.right.func.value.id == 'self' and \
               isinstance(v.left, ast.Name) and \
               v.left.id == _last_target[0]:
                ops.append(('unknown_call',
                            'module_shortcut_residual'))
                continue
            # THREAD = THREAD OP SAVED_NAME — the residual (the
            # normalized AugAssign form: 'out += identity' →
            # 'out = out + identity'; #8/#25's shape — the arm the
            # AugAssign rail EXPOSED; left = the current thread by
            # NAME, not just literal 'x'):
            if isinstance(v.right, ast.Name) and \
               isinstance(v.left, ast.Name) and \
               v.left.id == _last_target[0] and \
               v.right.id != v.left.id and \
               isinstance(stmt.targets[0], ast.Name) and \
               stmt.targets[0].id == v.left.id:
                skind2 = {ast.Add: 'add_saved', ast.Mult: 'mul_saved',
                          ast.Sub: 'sub_saved'}.get(type(v.op))
                if skind2:
                    ops.append((skind2, v.right.id))
                    continue
            # x = x OP SAVED_NAME — residual ops (NOT x+x — add_self):
            if isinstance(v.right, ast.Name) \
               and isinstance(v.left, ast.Name) and v.left.id == 'x' \
               and v.right.id != 'x':
                skind = {ast.Add: 'add_saved', ast.Mult: 'mul_saved',
                         ast.Sub: 'sub_saved'}.get(type(v.op))
                if skind:
                    ops.append((skind, v.right.id))
                    continue
            # x = torch.sigmoid(x) * x — REVERSED swish (sigmoid on left);
            # commutative product, same value BUT preserve source order:
            if isinstance(v.op, ast.Mult) and isinstance(v.left, ast.Call) \
               and isinstance(v.right, ast.Name) and v.right.id == 'x':
                fn = v.left.func
                if isinstance(fn, ast.Attribute) and fn.attr == 'sigmoid':
                    ops.append(('swish_rev',))
                    continue
            # x = x * ACT(x) — self-gated: swish (sigmoid) / x·mish(x):
            if isinstance(v.op, ast.Mult) and isinstance(v.right, ast.Call):
                fn = v.right.func
                if isinstance(fn, ast.Attribute) and fn.attr == 'sigmoid':
                    ops.append(('swish',))
                    continue
                if isinstance(fn, ast.Attribute) and fn.attr == 'mish':
                    ops.append(('xmish',))
                    continue
                if isinstance(fn, ast.Attribute) and fn.attr == 'hardswish':
                    ops.append(('xhardswish',))
                    continue
                if isinstance(fn, ast.Attribute) and fn.attr == 'clamp':
                    # x * clamp((x+3)/6, 0, 1) = written-out hardswish-CLIP
                    # form; keep the SOURCE arithmetic (clip form) — this
                    # is what the problem computes, NOT F.hardswish:
                    ca = v.right.args
                    if len(ca) >= 3:
                        ops.append(('xclip_hswish',))
                        continue
            # x = x + x:
            if isinstance(v.op, ast.Add) and isinstance(v.left, ast.Name) \
               and isinstance(v.right, ast.Name) and v.left.id == v.right.id:
                ops.append(('add_self',))
                continue
            # x = x - torch.mean(x, dim=(...), keepdim=True) — the
            # SELF-REFERENTIAL spatial-mean subtract (the #15 class):
            if isinstance(v.op, ast.Sub) and isinstance(v.left, ast.Name) \
               and isinstance(v.right, ast.Call) \
               and isinstance(v.right.func, ast.Attribute) \
               and v.right.func.attr == 'mean' \
               and v.right.args \
               and isinstance(v.right.args[0], ast.Name) \
               and v.right.args[0].id == v.left.id:
                _dims = None
                _keep = False
                for kw in v.right.keywords:
                    if kw.arg == 'dim':
                        try:
                            _dims = tuple(ast.literal_eval(kw.value)) \
                                if not isinstance(kw.value, ast.Constant) \
                                else (kw.value.value,)
                        except Exception:
                            _dims = None
                    if kw.arg == 'keepdim':
                        try:
                            _keep = bool(ast.literal_eval(kw.value))
                        except Exception:
                            _keep = False
                if _dims is not None and _keep:
                    ops.append(('sub_own_mean', _dims))
                    continue
                ops.append(('unknown_call', 'sub_own_mean_unresolved'))
                continue
            # THE ATTENTION BINOPS (#43/#50): a BinOp containing @
            # (MatMult) anywhere = a TORCH-STAGE (the two-tensor
            # matmul + optional scaling — the wrapper replicates
            # the expression; no kernel content). Guard: every leaf
            # is a Name/self-Attr/Constant/math-call/size-call:
            def _stageable(e):
                if isinstance(e, ast.Name) or isinstance(e, ast.Constant):
                    return True
                if isinstance(e, ast.Attribute):
                    return True
                if isinstance(e, ast.BinOp):
                    return _stageable(e.left) and _stageable(e.right)
                if isinstance(e, ast.UnaryOp):
                    return _stageable(e.operand)
                if isinstance(e, ast.Call):
                    return all(_stageable(a) for a in e.args) and \
                        _stageable(e.func) if not isinstance(
                            e.func, ast.Name) else all(
                            _stageable(a) for a in e.args)
                if isinstance(e, ast.Subscript):
                    return True
                if isinstance(e, ast.Compare):
                    return all(_stageable(x2) for x2 in
                               [e.left] + e.comparators)
                return False
            def _has_matmult(e):
                if isinstance(e, ast.BinOp):
                    if isinstance(e.op, ast.MatMult):
                        return True
                    return _has_matmult(e.left) or _has_matmult(e.right)
                return False
            if _has_matmult(v) and _stageable(v):
                # M_FLOW: record the TARGET name so the wrapper-env
                # can bind the result (att = eval('q @ k...')).
                _tgt = (stmt.targets[0].id
                        if isinstance(stmt.targets[0], ast.Name)
                        else None)
                ops.append(('stage', 'm_expr', [_tgt, ast.unparse(v)]))
                current_rank[0] = None
                current_width[0] = None
                continue
            # BinOp that matched NO case above (e.g. x = x + self.bias where
            # self.bias is a non-scalar nn.Parameter): scalar_of(rhs) was None,
            # so it fell through. FLAG it — never silently drop (a dropped
            # bias-add produces a confidently-wrong kernel: caught by Doresh
            # on #91, max abs diff 0.69). Honest-failure: name the gap.
            opname = {ast.Mult: 'multiply', ast.Div: 'divide',
                      ast.Add: 'add', ast.Sub: 'subtract'}.get(type(v.op), 'op')
            ops.append(('unknown_call', f'binop_{opname}_nonscalar_rhs'))
            continue
        # x = x.METHOD(...) — method-calls on the tensor. ALSO
        # NAME = NAME.METHOD(...) self-flow (att=att.masked_fill —
        # (B) the METHOD-CHAIN VOCABULARY (unfold/expand/reshape
        # on ANY Name-base → m_-stages; the target may be a
        # non-thread var — cls_tokens — the cat's _tref consumes
        # Names, so the emission order carries it):
        if isinstance(stmt, ast.Assign) and \
                len(stmt.targets) == 1 and \
                isinstance(stmt.targets[0], ast.Name) and \
                isinstance(stmt.value, ast.Call):
            _vcalls = []
            _vcur = stmt.value
            _vsafe = ('unfold', 'expand', 'reshape', 'view',
                      'permute', 'transpose', 'contiguous',
                      'squeeze', 'unsqueeze', 'flatten')
            while isinstance(_vcur, ast.Call) and \
                    isinstance(_vcur.func, ast.Attribute) and \
                    _vcur.func.attr in _vsafe:
                _vcalls.append(_vcur)
                _vcur = _vcur.func.value
            _vbase_ok = isinstance(_vcur, ast.Name) or (
                isinstance(_vcur, ast.Attribute) and
                isinstance(_vcur.value, ast.Name) and
                _vcur.value.id == 'self')
            if _vcalls and _vbase_ok:
                def _v28_arg(a):
                    # Constant · -lit · consts-Name · self.ATTR ·
                    # NAME.shape[i] → __SIZE{i}__ · p*p*NAME.shape[i]
                    if isinstance(a, ast.Constant):
                        return a.value
                    if isinstance(a, ast.UnaryOp) and \
                            isinstance(a.op, ast.USub) and \
                            isinstance(a.operand, ast.Constant):
                        return -a.operand.value
                    if isinstance(a, ast.Name) and \
                            a.id in consts and \
                            isinstance(consts[a.id],
                                       (int, float)):
                        return consts[a.id]
                    if isinstance(a, ast.Attribute) and \
                            isinstance(a.value, ast.Name) and \
                            a.value.id == 'self':
                        _av8 = attrs.get(a.attr)
                        return _av8 if isinstance(
                            _av8, (int, float)) else None
                    if isinstance(a, ast.Subscript) and \
                            isinstance(a.value, ast.Attribute) \
                            and a.value.attr == 'shape' and \
                            isinstance(a.slice, ast.Constant):
                        return f'__SIZE{a.slice.value}__'
                    if isinstance(a, ast.BinOp) and \
                            isinstance(a.op, ast.Mult):
                        _l8 = _v28_arg(a.left)
                        _r8 = _v28_arg(a.right)
                        if isinstance(_l8, (int, float)) and \
                                isinstance(_r8, (int, float)):
                            return int(_l8 * _r8)
                        # symbolic × concrete → the _MUL form:
                        if isinstance(_l8, str) and \
                                _l8.startswith('__SIZE') and \
                                isinstance(_r8, (int, float)):
                            return f'{_l8[:-2]}_MUL{int(_r8)}__'
                        if isinstance(_r8, str) and \
                                _r8.startswith('__SIZE') and \
                                isinstance(_l8, (int, float)):
                            return f'{_r8[:-2]}_MUL{int(_l8)}__'
                    return None
                _v28_ok = True
                _v28specs = []
                for _vc8 in reversed(_vcalls):
                    _va8 = []
                    for a in _vc8.args:
                        _r9 = _v28_arg(a)
                        if _r9 is None:
                            _v28_ok = False
                            break
                        _va8.append(_r9)
                    if not _v28_ok:
                        break
                    _v28specs.append(
                        ('stage', f'm_{_vc8.func.attr}', _va8))
                if _v28_ok and _v28specs:
                    # the base: a Parameter-attr (cls_token) rides
                    # as a param-source stage-note; a Name-base
                    # threads:
                    if isinstance(_vcur, ast.Attribute):
                        _v28specs.insert(0, (
                            'stage', 'm_param_src',
                            [_vcur.attr]))
                    ops.extend(_v28specs)
                    continue
        # the same value threading through its own name, #43):
        if isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) \
           and isinstance(v.func.value, ast.Name) \
           and (v.func.value.id == 'x' or
                (isinstance(stmt.targets[0], ast.Name) and
                 v.func.value.id == stmt.targets[0].id)):
            mname = v.func.attr
            # masked_fill(self.buf[...] == 0, const) — a TORCH-STAGE
            # (the wrapper replicates the whole method expression;
            # the mask is a buffer-slice compare — no kernel
            # content). Guard: all args stageable leaves:
            if mname == 'masked_fill':
                def _mf_ok(e):
                    if isinstance(e, (ast.Name, ast.Constant,
                                      ast.Attribute)):
                        return True
                    if isinstance(e, ast.Subscript):
                        return True
                    if isinstance(e, ast.Compare):
                        return all(_mf_ok(x3) for x3 in
                                   [e.left] + e.comparators)
                    if isinstance(e, ast.Call):
                        return all(_mf_ok(a3) for a3 in e.args)
                    if isinstance(e, ast.UnaryOp):
                        return _mf_ok(e.operand)
                    return False
                if all(_mf_ok(a) for a in v.args):
                    # M_FLOW: record the OWNER so the wrapper-env
                    # binds the result (att = att.masked_fill(...)).
                    _own = (stmt.targets[0].id
                            if isinstance(stmt.targets[0], ast.Name)
                            else None)
                    ops.append(('stage', 'm_method',
                                [_own, ast.unparse(v)]))
                    current_rank[0] = None
                    current_width[0] = None
                    continue
            if mname in ('squeeze', 'unsqueeze', 'view', 'reshape',
                         'flatten', 'permute', 'transpose', 'contiguous'):
                # THE FREE-VIEW UNLOCK (#38): x.view(b, c, -1) on a
                # contiguous tensor is ZERO-COPY — no stage break;
                # the rank drops to 3 (making a dim=2 softmax the
                # LAST dim — the row-softmax machinery). The
                # view-back (5 Name/attr args) restores rank 5,
                # also free:
                if mname == 'view' and len(v.args) == 3 and \
                        isinstance(v.args[2], ast.UnaryOp) and \
                        isinstance(v.args[2].op, ast.USub):
                    current_rank[0] = 3
                    current_width[0] = (
                        __import__('math').prod(current_spatial[0])
                        if current_spatial[0] else None)
                    continue
                if mname == 'view' and len(v.args) == 5:
                    current_rank[0] = 5
                    continue
                _margs = []
                for a in v.args:
                    _cv = _const_eval(a)
                    if _cv is None:
                        # x.size(0)-style runtime args: record the
                        # SPECIAL marker for view(x.size(0), -1) —
                        # the wrapper spells x.size(0) directly:
                        if (isinstance(a, ast.Call)
                                and isinstance(a.func, ast.Attribute)
                                and a.func.attr == 'size'
                                and isinstance(a.func.value, ast.Name)
                                and len(a.args) == 1
                                and isinstance(a.args[0], ast.Constant)):
                            _margs.append(
                                f'__SIZE{a.args[0].value}__')
                            continue
                        _margs = None
                        break
                    _margs.append(_cv)
                ops.append(('stage', f'm_{mname}', _margs))
                if mname == 'unsqueeze' and current_rank[0] is not None:
                    current_rank[0] += 1
                elif mname == 'squeeze' and current_rank[0] is not None:
                    current_rank[0] -= 1
                elif mname == 'contiguous':
                    pass  # rank-preserving
                else:
                    current_rank[0] = None
                _apply_post_shape()
                continue
            if mname in ('clamp',):
                lo = hi = None
                for kw in v.keywords:
                    if kw.arg == 'min':
                        lo = scalar_of(kw.value)
                    if kw.arg == 'max':
                        hi = scalar_of(kw.value)
                if lo is not None and hi is not None:
                    ops.append(('clamp', float(lo), float(hi)))
                    continue
        # x = torch.max(x, dim=D, keepdim=..).values — the .values
        # Attribute form ≡ [0]-subscript (#80):
        if isinstance(v, ast.Attribute) and v.attr == 'values' and \
           isinstance(v.value, ast.Call):
            inner = v.value
            fn = inner.func
            name = (fn.attr if isinstance(fn, ast.Attribute) else
                    (fn.id if isinstance(fn, ast.Name) else None))
            if name in ('min', 'max'):
                d = None
                for kw in inner.keywords:
                    if kw.arg == 'dim':
                        if isinstance(kw.value, ast.Constant):
                            d = kw.value.value
                        elif isinstance(kw.value, ast.Attribute) and \
                             isinstance(kw.value.value, ast.Name) and \
                             kw.value.value.id == 'self' and \
                             kw.value.attr in init_args:
                            d = init_args[kw.value.attr]
                kd = any(kw.arg == 'keepdim' and
                         getattr(kw.value, 'value', False) is True
                         for kw in inner.keywords)
                if _axis_is_last(d, current_rank[0]):
                    ops.append(('reduction', name, current_width[0]))
                    if kd:
                        current_width[0] = 1
                    continue
                ops.append(('unknown_call', 'values_nonlast_reduce'))
                continue
        # x = torch.min(x, dim=D, keepdim=..)[0] — subscripted reduce:
        if isinstance(v, ast.Subscript) and isinstance(v.value, ast.Call):
            inner = v.value
            fn = inner.func
            name = fn.attr if isinstance(fn, ast.Attribute) else \
                   (fn.id if isinstance(fn, ast.Name) else None)
            if name in ('min', 'max'):
                d = None
                for kw in inner.keywords:
                    if kw.arg == 'dim' and isinstance(kw.value, ast.Constant):
                        d = kw.value.value
                    # dim=self.ATTR where ATTR is an init-arg (#24):
                    elif kw.arg == 'dim' and \
                            isinstance(kw.value, ast.Attribute) and \
                            isinstance(kw.value.value, ast.Name) and \
                            kw.value.value.id == 'self' and \
                            kw.value.attr in init_args:
                        d = init_args[kw.value.attr]
                if d is None and len(inner.args) >= 2 and \
                   isinstance(inner.args[1], ast.Constant):
                    d = inner.args[1].value
                if not _axis_is_last(d, current_rank[0]):
                    if d == 1 and current_rank[0] is not None and \
                       current_rank[0] >= 3:
                        ops.append(('reduction_chan', name, current_chan[0],
                                    (__import__('math').prod(current_spatial[0])
                                     if current_spatial[0] else None)))
                        continue
                    # THE ARBITRARY-AXIS FOLD (the #24 site — the
                    # subscripted torch.min(x, dim=k)[0] form; no
                    # keepdim → the rank SHRINKS, the folded dim
                    # DROPS from spatial):
                    if name in ('min', 'max') and d is not None and \
                       d >= 2 and current_rank[0] is not None and \
                       d < current_rank[0] - 1 and \
                       current_spatial[0] and \
                       len(current_spatial[0]) > (d - 2):
                        _sp = current_spatial[0]
                        _chan_f = _sp[d - 2]
                        _after = _sp[d - 1:]
                        _spatial_f = (__import__('math').prod(_after)
                                      if _after else 1)
                        ops.append(('reduction_chan_fold', name,
                                    _chan_f, _spatial_f))
                        _kd = any(kw.arg == 'keepdim' and
                                  getattr(kw.value, 'value', False)
                                  is True for kw in inner.keywords)
                        _sp2 = list(_sp)
                        if _kd:
                            _sp2[d - 2] = 1
                        else:
                            _sp2.pop(d - 2)
                            current_rank[0] -= 1
                        current_spatial[0] = _sp2
                        continue
                    ops.append(('unknown_call', 'non_lastdim_reduction'))
                    continue
                if current_width[0] == 1:
                    # size-1 reduce = IDENTITY (bit-exact: max/min of one
                    # element is itself) — drop the op (#18 class):
                    kd1 = any(kw.arg == 'keepdim' and
                              getattr(kw.value, 'value', False) is True
                              for kw in inner.keywords)
                    if kd1:
                        continue
                    current_width[0] = None  # rank shrinks, width unknown
                    continue
                kd1 = any(kw.arg == 'keepdim' and
                          getattr(kw.value, 'value', False) is True
                          for kw in inner.keywords)
                ops.append(('reduction', name, current_width[0]))
                if kd1:
                    current_width[0] = 1
                continue
        # torch.multiply(tanh(softplus(x)), x) = MISH (#52's spelled-out
        # form — the composition IS the activation):
        if isinstance(v, ast.Call):
            fnm = v.func
            nm = fnm.attr if isinstance(fnm, ast.Attribute) else \
                 (fnm.id if isinstance(fnm, ast.Name) else None)
            if nm == 'multiply' and len(v.args) == 2 and \
               isinstance(v.args[1], ast.Name) and v.args[1].id == 'x' and \
               isinstance(v.args[0], ast.Call):
                inner1 = v.args[0]
                n1 = inner1.func.attr if isinstance(inner1.func, ast.Attribute) else None
                if n1 == 'tanh' and inner1.args and \
                   isinstance(inner1.args[0], ast.Call):
                    inner2 = inner1.args[0]
                    n2 = inner2.func.attr if isinstance(inner2.func, ast.Attribute) else None
                    if n2 == 'softplus' and inner2.args and \
                       isinstance(inner2.args[0], ast.Name) and \
                       inner2.args[0].id == 'x':
                        ops.append(('mish',))
                        continue
        # x = torch.F(x, ...) / F.f(x):
        if isinstance(v, ast.Call):
            fn = v.func
            name = fn.attr if isinstance(fn, ast.Attribute) else \
                   (fn.id if isinstance(fn, ast.Name) else None)
            if name in ('relu', 'tanh', 'sigmoid', 'gelu', 'mish',
                        'hardswish', 'hardtanh'):
                # THE L3 NESTED-CALL UNLOCK: F.relu(self.conv1(x))
                # — the inner MODULE call must emit as a stage
                # FIRST (LeNet/AlexNet/VGG's dominant form; the L2
                # walk never met it — L2 forwards are flat):
                if v.args and isinstance(v.args[0], ast.Call):
                    _in = v.args[0]
                    if isinstance(_in.func, ast.Attribute) and \
                            isinstance(_in.func.value, ast.Name) and \
                            _in.func.value.id == 'self' and \
                            len(_in.args) == 1:
                        ops.append(('stage', _in.func.attr))
                    else:
                        ops.append(('unknown_call',
                                    f'nested_call_in_{name}'))
                        continue
                ops.append((name,))
                continue
            if name == 'leaky_relu':
                slope = 0.01
                if len(v.args) >= 2:
                    s = scalar_of(v.args[1])
                    if s is not None:
                        slope = float(s)
                for kw in v.keywords:
                    if kw.arg == 'negative_slope':
                        s = scalar_of(kw.value)
                        if s is not None:
                            slope = float(s)
                ops.append(('leaky_relu', slope))
                continue
            if name in ('min', 'max') and len(v.args) == 2 and \
                    not v.keywords:
                # torch.min(x, torch.tensor(V)) — ELEMENTWISE min/max
                # vs a scalar tensor (the #83 unlock, mavhir's sweep):
                # min = clamp-from-above, max = clamp-from-below.
                # Resolve V from a torch.tensor(literal-or-init-arg)
                # wrapper or a bare literal/name:
                a1 = v.args[1]
                vv = None
                if isinstance(a1, ast.Call) and \
                        isinstance(a1.func, ast.Attribute) and \
                        a1.func.attr == 'tensor' and a1.args:
                    a1 = a1.args[0]
                try:
                    vv = float(ast.literal_eval(a1))
                except Exception:
                    if isinstance(a1, ast.Name) and a1.id in init_args:
                        vv = float(init_args[a1.id])
                    # self.ATTR where ATTR is an init-arg (the #31/#68
                    # regression from the #83 branch intercepting —
                    # the older resolution restored):
                    elif isinstance(a1, ast.Attribute) and \
                            isinstance(a1.value, ast.Name) and \
                            a1.value.id == 'self' and \
                            a1.attr in init_args:
                        vv = float(init_args[a1.attr])
                if vv is not None:
                    ops.append(('clamp_max' if name == 'min'
                                else 'clamp_min', vv))
                    continue
                ops.append(('unknown_call',
                            f'{name}_with_unresolved_tensor'))
                continue
            if name == 'clamp':
                lo = hi = None
                for kw in v.keywords:
                    if kw.arg == 'min':
                        lo = scalar_of(kw.value)
                    if kw.arg == 'max':
                        hi = scalar_of(kw.value)
                if lo is None and len(v.args) >= 3:
                    lo = scalar_of(v.args[1])
                    hi = scalar_of(v.args[2])
                if lo is not None and hi is not None:
                    ops.append(('clamp', float(lo), float(hi)))
                    continue
                if lo is not None:
                    ops.append(('clamp_min', float(lo)))
                    continue
                if hi is not None:
                    ops.append(('clamp_max', float(hi)))
                    continue
                ops.append(('unknown_call', 'clamp_unresolved'))
                continue
            if name in ('group_norm', 'batch_norm', 'layer_norm',
                        'instance_norm', 'normalize'):
                ops.append(('stage', f'F_{name}'))
                continue
            # pooling/dropout: rank-preserving stage-ops (torch runs them):
            if name in ('avg_pool1d', 'avg_pool2d', 'avg_pool3d',
                        'max_pool1d', 'max_pool2d', 'max_pool3d',
                        'adaptive_avg_pool1d', 'adaptive_avg_pool2d',
                        'adaptive_avg_pool3d', 'dropout'):
                try:
                    _fargs = [ast.literal_eval(a) for a in v.args[1:]]
                    # kwargs → positional-style dict (the L3 pools:
                    # max_pool2d(x, kernel_size=2, stride=2)):
                    _fkw = {}
                    for kw in v.keywords:
                        _kv = ast.literal_eval(kw.value)
                        _fkw[kw.arg] = _kv
                    if _fkw:
                        _fargs = _fargs + [('__KW__', _fkw)]
                except Exception:
                    _fargs = None
                ops.append(('stage', f'F_{name}', _fargs))
                if name != 'dropout':
                    current_width[0] = None   # pools change spatial dims
                continue
            # shape-ops: stage-markers with UNKNOWN rank after (conservative —
            # post-op reductions will refuse via the axis-check):
            if name in ('squeeze', 'unsqueeze', 'view', 'reshape',
                        'flatten', 'permute', 'transpose'):
                # record literal args (the L3 unlock: torch.flatten
                # (x, 1) etc — the wrapper replays them; constant
                # EXPRESSIONS (16*5*5) fold via _const_eval):
                _fargs2 = []
                for a in v.args[1:]:
                    _cv = _const_eval(a)
                    if _cv is None:
                        _fargs2 = None
                        break
                    _fargs2.append(_cv)
                ops.append(('stage', f'F_{name}', _fargs2))
                if name == 'unsqueeze' and current_rank[0] is not None:
                    current_rank[0] += 1
                elif name == 'squeeze' and current_rank[0] is not None:
                    current_rank[0] -= 1
                else:
                    current_rank[0] = None
                _apply_post_shape()
                continue
            if name in ('group_norm', 'batch_norm', 'layer_norm',
                        'instance_norm', 'normalize'):
                ops.append(('stage', f'F_{name}'))
                continue
            # torch.min(x, torch.tensor(CONST)): elementwise vs a literal —
            # min(x, c) = clamp_max(c); max(x, c) = clamp_min(c):
            if name in ('min', 'max') and len(v.args) >= 2 and \
               isinstance(v.args[1], ast.Call) and \
               not any(kw.arg == 'dim' for kw in v.keywords):
                ifn = v.args[1].func
                iname = ifn.attr if isinstance(ifn, ast.Attribute) else None
                if iname == 'tensor' and v.args[1].args and \
                   isinstance(v.args[1].args[0], ast.Constant):
                    cval = float(v.args[1].args[0].value)
                    ops.append(('clamp_max' if name == 'min'
                                else 'clamp_min', cval))
                    continue
                # torch.min(x, torch.tensor(self.ATTR)) — param-valued
                # elementwise min/max (#31's form): scalar param p[0]:
                if iname == 'tensor' and v.args[1].args and \
                   isinstance(v.args[1].args[0], ast.Attribute) and \
                   isinstance(v.args[1].args[0].value, ast.Name) and \
                   v.args[1].args[0].value.id == 'self':
                    ops.append(('min_param_scalar' if name == 'min'
                                else 'max_param_scalar',
                                v.args[1].args[0].attr))
                    continue
            # torch.min(x, self.ATTR)/torch.max(x, self.ATTR): ELEMENTWISE
            # binary min/max with a param (NOT a reduction — no dim arg):
            if name in ('min', 'max') and len(v.args) >= 2 and \
               isinstance(v.args[1], ast.Attribute) and \
               isinstance(v.args[1].value, ast.Name) and \
               v.args[1].value.id == 'self' and \
               not any(kw.arg == 'dim' for kw in v.keywords):
                attr = v.args[1].attr
                if attr in scalar_params:
                    ops.append((f'{name}_param_scalar', attr))
                    continue
                ops.append(('unknown_call', f'{name}_with_tensor_param'))
                continue
            if name in ('softmax', 'logsumexp', 'mean', 'max', 'min', 'sum'):
                d = None
                for kw in v.keywords:
                    if kw.arg == 'dim':
                        try:
                            d = ast.literal_eval(kw.value)
                        except Exception:
                            pass
                if d is None and len(v.args) >= 2:
                    try:
                        d = ast.literal_eval(v.args[1])
                    except Exception:
                        pass
                # dim=self.ATTR (resolved via init args — #8's sum_dim):
                if d is None:
                    for kw in v.keywords:
                        if kw.arg == 'dim' and \
                           isinstance(kw.value, ast.Attribute) and \
                           isinstance(kw.value.value, ast.Name) and \
                           kw.value.value.id == 'self' and \
                           kw.value.attr in init_args:
                            d = init_args[kw.value.attr]
                    if d is None and len(v.args) >= 2 and \
                       isinstance(v.args[1], ast.Attribute) and \
                       isinstance(v.args[1].value, ast.Name) and \
                       v.args[1].value.id == 'self' and \
                       v.args[1].attr in init_args:
                        d = init_args[v.args[1].attr]
                # dim-TUPLE: trailing-contiguous tuples lift to the
                # Case-1 tuple template (mean/sum only):
                if isinstance(d, (tuple, list)) and \
                   name in ('mean', 'sum') and \
                   current_rank[0] is not None:
                    rk = current_rank[0]
                    dd = sorted(x if x >= 0 else rk + x for x in d)
                    if dd == list(range(rk - len(dd), rk)):
                        kdt = any(kw.arg == 'keepdim' and
                                  getattr(kw.value, 'value', False) is True
                                  for kw in v.keywords)
                        # SIZE-1 tuple reduce = IDENTITY (the #42/#18
                        # class: after a keepdim tuple-mean the spatial
                        # dims are all 1; sum/mean over them is the
                        # element itself — bit-exact drop):
                        if current_spatial[0] is not None and \
                           len(dd) == len(current_spatial[0]) and \
                           all(z == 1 for z in current_spatial[0]):
                            if kdt:
                                continue
                            current_rank[0] = (current_rank[0] - len(dd)
                                               if current_rank[0] else None)
                            current_spatial[0] = None
                            continue
                        nred = None
                        rows = None
                        if current_spatial[0] is not None and \
                           len(dd) <= len(current_spatial[0]) + 1:
                            import math
                            sp = current_spatial[0]
                            if len(dd) == len(sp):        # spatial only
                                nred = math.prod(sp)
                                if input_batch and current_chan[0]:
                                    rows = input_batch * current_chan[0]
                            elif len(dd) == len(sp) + 1:  # chan+spatial
                                if current_chan[0]:
                                    nred = math.prod(sp) * current_chan[0]
                                    rows = input_batch
                        ops.append(('reduction_tuple', name, nred, rows))
                        if kdt and current_spatial[0] is not None and \
                           len(dd) == len(current_spatial[0]):
                            # keepdim: spatial dims collapse to 1s —
                            # subsequent dim-1 reduces see inner=1
                            # (the shape-collapse that turns chan-lse
                            # into ROW-lse, #42):
                            current_spatial[0] = [1] * len(dd)
                        continue
                    ops.append(('unknown_call', 'nontrailing_tuple_reduction'))
                    continue
                if not _axis_is_last(d, current_rank[0]):
                    if d == 1 and current_spatial[0] is not None and \
                       all(z == 1 for z in current_spatial[0]) and \
                       name in ('min', 'max', 'softmax', 'mean', 'sum',
                                'logsumexp'):
                        # spatial all-1s: dim-1 IS lastdim-equivalent
                        # (the trailing dims are size-1) → ROW-reduce
                        # with width = chan_count (#42's lse-over-C
                        # after global-avg-pool):
                        kdr = any(kw.arg == 'keepdim' and
                                  getattr(kw.value, 'value', False) is True
                                  for kw in v.keywords)
                        ops.append(('reduction', name, current_chan[0]))
                        if kdr:
                            current_chan[0] = 1
                            current_width[0] = 1
                        continue
                    if name in ('min', 'max', 'softmax', 'mean', 'sum', 'logsumexp') and d == 1 and \
                       current_rank[0] is not None and current_rank[0] >= 3:
                        ops.append(('reduction_chan', name, current_chan[0],
                                    (__import__('math').prod(current_spatial[0])
                                     if current_spatial[0] else None)))
                        continue
                    # THE ARBITRARY-AXIS FOLD (#13/#24/#36 — measured
                    # bit-exact for mean AND min at the real shapes):
                    # a reduce over spatial axis d (2 <= d < rank-1)
                    # folds to a chan-reduce: outer = prod(dims before
                    # d), chan = spatial[d-2], spatial = prod(after).
                    # The chan template's semantics don't care what
                    # the outer is.
                    if name in ('min', 'max', 'mean', 'sum') and \
                       isinstance(d, int) and d >= 2 and \
                       current_rank[0] is not None and \
                       d < current_rank[0] - 1 and \
                       current_spatial[0] and \
                       len(current_spatial[0]) > (d - 2):
                        _sp = current_spatial[0]
                        _chan_f = _sp[d - 2]
                        _after = _sp[d - 1:]
                        _spatial_f = (__import__('math').prod(_after)
                                      if _after else 1)
                        ops.append(('reduction_chan_fold', name,
                                    _chan_f, _spatial_f))
                        # shape after (keepdim assumed — the folded
                        # axis becomes 1):
                        _sp2 = list(_sp)
                        _sp2[d - 2] = 1
                        current_spatial[0] = _sp2
                        continue
                    ops.append(('unknown_call', 'non_lastdim_reduction'))
                    continue
                kd2 = any(kw.arg == 'keepdim' and
                          getattr(kw.value, 'value', False) is True
                          for kw in v.keywords)
                if current_width[0] == 1:
                    # size-1 reduce = IDENTITY, bit-exact for sum/mean/
                    # max/min (x itself) AND lse (torch: log(exp(x-x))+x
                    # = x exactly) — drop (#18's serial-identity chain):
                    if kd2:
                        continue
                    current_width[0] = None
                    continue
                ops.append(('reduction', name, current_width[0]))
                if kd2:
                    current_width[0] = 1
                continue
            # torch.matmul(x, self.W.T) — a GEMM STAGE (torch runs
            # it — the #14 unlock): rank stays 2; the width becomes
            # the weight's OUT dim (W stored (out,in), used .T):
            if name == 'matmul' and len(v.args) == 2 and \
                    isinstance(v.args[0], ast.Name):
                _w = v.args[1]
                # unwrap .T:
                if isinstance(_w, ast.Attribute) and _w.attr == 'T':
                    _w = _w.value
                if isinstance(_w, ast.Attribute) and \
                        isinstance(_w.value, ast.Name) and \
                        _w.value.id == 'self':
                    ops.append(('stage', 'm_matmul', [_w.attr]))
                    current_rank[0] = 2
                    _wshape = param_shapes.get(_w.attr)
                    current_width[0] = (_wshape[0] if _wshape
                                        else None)
                    current_chan[0] = None
                    current_spatial[0] = None
                    continue
            # ═══ RUNG ONE (the keystone ladder): .to(x.device) is
            # IDENTITY in the single-GPU emit world — a device move
            # only. THE GUARD: identity ONLY when the arg is a
            # .device attribute; a DTYPE move (torch.float16 etc)
            # would change bits and stays UNKNOWN (named, refused):
            # ═══ KEYSTONE ITEM 2: torch.cat((a, b), dim=D) with
            # Name/Attribute args = a TORCH-EXECUTED STAGE (m_cat2).
            # The wrapper's forward replicates it; the chain thread
            # continues from its output. Guard: exactly-two tensor
            # refs (Name or self.Attribute), dim literal:
            if name == 'cat' and len(v.args) >= 1:
                _tup = v.args[0]
                _dimv = None
                for kw in v.keywords:
                    if kw.arg == 'dim' and isinstance(kw.value,
                                                      ast.Constant):
                        _dimv = kw.value.value
                if len(v.args) == 2 and isinstance(v.args[1],
                                                   ast.Constant):
                    _dimv = v.args[1].value
                def _tref(e):
                    return isinstance(e, ast.Name) or (
                        isinstance(e, ast.Attribute) and
                        isinstance(e.value, ast.Name) and
                        e.value.id == 'self') or (
                        # x[t] — Subscript of a Name by a Name
                        # (the loop index; #34's per-step slice):
                        isinstance(e, ast.Subscript) and
                        isinstance(e.value, ast.Name) and
                        isinstance(e.slice, ast.Name))
                if isinstance(_tup, (ast.Tuple, ast.List)) and \
                        len(_tup.elts) == 2 and \
                        all(_tref(e) for e in _tup.elts) and \
                        _dimv is not None:
                    ops.append(('stage', 'm_cat2', [_dimv]))
                    current_rank[0] = None
                    current_width[0] = None
                    continue
            # .contiguous() = memory-layout only — VALUE-identity:
            if name == 'contiguous' and not v.args and not v.keywords:
                continue
            if name == 'to' and len(v.args) == 1 and \
                    isinstance(v.args[0], ast.Attribute) and \
                    v.args[0].attr == 'device' and not v.keywords:
                continue  # identity — no op emitted
            ops.append(('unknown_call', name))
            continue
        # #28 VOCABULARY (the cost-read rung — Model-level):
        # (A) the scalar-attr assign: p = self.patch_size → consts
        # (the ctor-arg value via attrs; the 'unrecognized_assign'
        # of the unfold-line's PRECEDING stmt):
        if isinstance(stmt, ast.Assign) and \
                len(stmt.targets) == 1 and \
                isinstance(stmt.targets[0], ast.Name) and \
                isinstance(stmt.value, ast.Attribute) and \
                isinstance(stmt.value.value, ast.Name) and \
                stmt.value.value.id == 'self':
            _sav = attrs.get(stmt.value.attr)
            if _sav is not None and \
                    isinstance(_sav, (int, float)):
                consts[stmt.targets[0].id] = _sav
                continue
        # CATCH-ALL: any assignment statement that matched NO handler above
        # (not a stage, not a recognized BinOp, not a recognized Call) is
        # FLAGGED, never silently dropped. A silently-dropped statement
        # produces a confidently-wrong kernel — the exact hole Doresh found
        # on #91. The honest-failure property ('name the gap, never fake')
        # requires that NOTHING falls through unflagged.
        if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
            # the identity self-assign (x = x — the method-splice's
            # return-artifact: 'return x' spliced as 'x = x') is a
            # NO-OP, not a gap — skip it (rung-2: #29's
            # forward_features splice-tail):
            if isinstance(stmt.value, ast.Name) and \
                    stmt.value.id == stmt.targets[0].id:
                continue
            ops.append(('unknown_call', 'unrecognized_assign'))
        elif isinstance(stmt, ast.Assign) and \
                isinstance(stmt.targets[0], ast.Attribute):
            # THE SILENT-DROP FIX (the #33 i2h finding): an
            # ATTRIBUTE-target assign (self.hidden = ...) that no
            # handler above claimed must be FLAGGED like a
            # Name-target — a silently-dropped stage produces a
            # confidently-wrong kernel (witnessed live on #33
            # before this line existed):
            ops.append(('unknown_call', 'attr_target_assign'))
    # ═══ THE REPORTING BRIDGE (Heath's nomenclature ruling): when
    # the walk produced STAGE-ONLY ops (no fusable content), say
    # WHICH stages are unopened composites — the walker already
    # knows (stage_classes); this makes it SAY (the orphan-guard
    # shape: name what you did not look inside). The declined-
    # composites list IS the work queue.
    _COMPOSITE_CLASSES = {
        'Sequential': 'container_unroll_failed',
        'ModuleList': 'container_unroll_failed',
        'ModuleDict': 'container_unroll_failed',
        'LSTM': 'monolithic_cudnn',
        'GRU': 'monolithic_cudnn',
        'RNN': 'monolithic_cudnn',
        'MultiheadAttention': 'monolithic_fused_attention',
        'TransformerEncoder': 'composite_transformer',
        'TransformerEncoderLayer': 'composite_transformer',
    }
    _fusable = [o for o in ops if o[0] not in (
        'stage', 'size_unpack', 'loop_begin', 'loop_end',
        'loop_collect', 'save')]
    if not _fusable:
        _opaque = []
        for o in ops:
            if o[0] != 'stage':
                continue
            _sname = o[1]
            if _sname.startswith(('m_', 'F_')):
                continue
            # a '.N' suffix means an ALREADY-UNROLLED nested path (the
            # child of a container that was unrolled) — it's a resolved
            # leaf, not an un-unrolled container; don't re-flag it by the
            # parent's class (the spurious-composite fix).
            if '.' in _sname:
                continue
            _scls = stage_classes.get(_sname)
            if _scls in _COMPOSITE_CLASSES:
                _opaque.append((_sname, _scls,
                                _COMPOSITE_CLASSES[_scls]))
            elif _scls is None:
                _opaque.append((_sname, 'unknown',
                                'unresolved_callable'))
        if _opaque:
            ops.append(('opaque_composites', _opaque))
    # THE SOFTMAX-WIDTH RESOLUTION (#43's rung): a width-None softmax
    # reduction whose PRECEDING m_flow/m_method source's last
    # subscript-slice names a BOUND size-name gets the binding's
    # value (the mask's ':T]' names the row width — the QKV shape;
    # anything else stays None → the honest emit-time refusal):
    _bindings = {o[1]: o[2] for o in ops if o[0] == 'size_binding'}
    if _bindings:
        import re as _re2
        _last_src = None
        for _oi, o in enumerate(ops):
            if o[0] == 'stage' and o[1] in ('m_flow', 'm_method',
                                            'm_expr') and \
                    len(o) > 2 and o[2] and len(o[2]) > 1:
                _last_src = o[2][1]
            elif o[0] == 'reduction' and o[1] in ('softmax',
                                                  'logsumexp') and \
                    len(o) > 2 and o[2] is None and _last_src:
                _m3 = _re2.findall(r':\s*(\w+)\s*\]', _last_src)
                if _m3 and _m3[-1] in _bindings:
                    ops[_oi] = (o[0], o[1], _bindings[_m3[-1]])
    return ops


# ═══ THE TRANSFORM ERA (front B, rung one) ═══
# The first rewrite: ELIMINATE-IDENTITY-STAGES (eval-mode Dropout).
# Arithmetic-preserving BY CONSTRUCTION — the whole-model gate
# (lib/wholegate.py) must read zero; any non-zero indicts the
# rewrite machinery, not the arithmetic. Provenance: the rewritten
# ops carry ('rewrite_provenance', [...]) as their first entry so
# a rewritten unit is never indistinguishable from an unrewritten
# one (Mavdil's dual).
LAST_STAGE_CLASSES = {}


def rewrite_eliminate_identity(ops, stage_classes=None):
    """Drop stages that are identity in eval mode (Dropout).
    Returns (new_ops, eliminated) — eliminated names the dropped
    stages for the provenance record."""
    sc = stage_classes if stage_classes is not None \
        else LAST_STAGE_CLASSES
    out, eliminated = [], []
    for o in ops:
        if o[0] == 'stage' and len(o) >= 2 and \
                isinstance(o[1], str) and \
                sc.get(o[1]) == 'Dropout':
            eliminated.append(o[1])
            continue
        out.append(o)
    if eliminated:
        out = [('rewrite_provenance',
                ['eliminate_identity:' + ','.join(eliminated)])] \
            + out
    return out, eliminated


def _avgpool_guard(ops):
    """avgpool3d_k2 emits ONLY in pure-avgpool chains (the #72
    route); in MIXED chains it stays a STAGE (the #50 regression:
    the op fired mid-chain where the elementwise harness has no
    spelling — vocabulary gap at emit)."""
    _lifted = [o for o in ops if o[0] != 'stage']
    if any(o[0] == 'avgpool3d_k2' for o in _lifted) and \
            any(o[0] != 'avgpool3d_k2' for o in _lifted):
        return [(('stage', o[1]) if (o[0] == 'avgpool3d_k2'
                                     and len(o) > 1) else o)
                for o in ops]
    return ops


def to_prolog(pid, ops):
    ops = _drop_dead_saves(ops)
    ops = _avgpool_guard(ops)
    """Chain fact for the bridge; epilogue ops only (post-stage)."""
    terms_meta = []
    # find the last 'stage' marker; the epilogue is what follows:
    # the opaque-composites REPORT is META (never epilogue content —
    # the no-op-kernel regression caught by test-after-edit):
    for _o in ops:
        if _o[0] == 'opaque_composites':
            for _attr, _cls, _why in _o[1]:
                terms_meta.append(
                    f'%% opaque_composite {_attr} {_cls} {_why}')
    ops = [o for o in ops if o[0] != 'opaque_composites']
    idx = max((i for i, o in enumerate(ops) if o[0] == 'stage'), default=-1)
    epi = ops[idx + 1:]
    # COMPLETENESS CHECK (Doresh caught on #94): 'everything after the LAST
    # stage' silently DISCARDS real fusable ops SANDWICHED between non-terminal
    # stage markers (e.g. gemm->+bias->hardtanh->mish->groupnorm drops the
    # bias/hardtanh/mish -> emits out[i]=x, a pure no-op, reported PASS).
    # If there are any NON-stage ops BEFORE the last stage marker, this is a
    # multi-stage fusion the epilogue-extraction can't express — FLAG it,
    # never silently drop. (idx-1 because ops[idx] is the last stage itself.)
    pre = ops[:idx] if idx >= 0 else []
    dropped = [o for o in pre if o[0] not in ('stage', 'save')]
    # ('save', name) = the residual's bookkeeping (x.clone().detach()),
    # not a dropped computation (the #51 unlock — mavhir's trace,
    # Mavdil's tested confirmation)
    if dropped:
        # real ops lost between stages: refuse honestly, name the gap.
        return f"chain({pid}, [UNSUPPORTED_multi_stage_gap])."
    terms = []
    for o in epi:
        if o[0] == 'multiply':
            terms.append(f'multiply({o[1]})')
        elif o[0] == 'divide':
            terms.append(f'divide({o[1]})')
        elif o[0] == 'add_scalar':
            terms.append(f'add_scalar({o[1]})')
        elif o[0] == 'subtract':
            terms.append(f'subtract({o[1]})')
        elif o[0] == 'clamp':
            terms.append(f'clamp({o[1]}, {o[2]})')
        elif o[0] == 'clamp_min':
            terms.append(f'clamp_min({o[1]})')
        elif o[0] == 'clamp_max':
            terms.append(f'clamp_max({o[1]})')
        elif o[0] == 'swish':
            terms.append('swish')
        elif o[0] == 'add_self':
            terms.append('add_self')
        elif o[0] == 'xmish':
            terms.append('xmish')
        elif o[0] == 'mish':
            terms.append('mish')
        elif o[0] == 'xsig3_div6':
            terms.append('xsig3_div6')
        elif o[0] == 'zero_fold':
            terms.append('zero_fold')
        elif o[0] == 'swish_rev':
            terms.append('swish_rev')
        elif o[0] == 'xclip_hswish':
            terms.append('xclip_hswish')
        elif o[0] == 'xhardswish':
            terms.append('xhardswish')
        elif o[0] in ('add_input2', 'mul_input2', 'sub_input2'):
            terms.append(o[0])
        elif o[0] == 'mul_saved':
            terms.append(f"mul_saved('{o[1]}')")
        elif o[0] == 'sub_saved':
            terms.append(f"sub_saved('{o[1]}')")
        elif o[0] == 'leaky_relu':
            terms.append(f'leaky_relu({o[1]})')
        elif o[0] == 'reduction':
            terms.append(f'reduction({o[1]})')
            if len(o) > 2 and o[2] is not None:
                terms_meta.append(f'%% reduce_width {o[2]}')
        elif o[0] == 'reduction_chan':
            terms.append(f'reduction_chan({o[1]})')
            if len(o) > 2 and o[2] is not None:
                terms_meta.append(f'%% chan_count {o[2]}')
            if len(o) > 3 and o[3] is not None:
                terms_meta.append(f'%% spatial_inner {o[3]}')
        elif o[0] == 'reduction_tuple':
            terms.append(f'reduction_tuple({o[1]})')
            if len(o) > 2 and o[2] is not None:
                terms_meta.append(f'%% tuple_n {o[2]}')
            if len(o) > 3 and o[3] is not None:
                terms_meta.append(f'%% tuple_rows {o[3]}')
        elif o[0] in ('add_param', 'mul_param', 'sub_param',
                      'add_param_scalar', 'mul_param_scalar',
                      'sub_param_scalar', 'add_param_chan',
                      'mul_param_chan', 'sub_param_chan',
                      'min_param_scalar', 'max_param_scalar'):
            terms.append(f"{o[0]}('{o[1]}')")
        elif o[0] == 'save':
            terms.append(f"save('{o[1]}')")
        elif o[0] == 'loop_begin':
            terms.append(f"loop_begin('{o[1]}', '{o[2]}')")
        elif o[0] == 'loop_end':
            terms.append('loop_end')
        elif o[0] == 'loop_collect':
            terms.append(f"loop_collect('{o[1]}', '{o[2]}')")

        elif o[0] == 'size_unpack':
            terms.append(
                f"size_unpack('{o[1]}', {o[2]!r})".replace('\'', "'"))
        elif o[0] == 'add_saved':
            terms.append(f"add_saved('{o[1]}')")
        elif o[0] == 'reduction_chan_fold':
            terms.append(f'reduction_chan_fold({o[1]})')
            terms_meta.append(f'%% fold_chan {o[2]}')
            terms_meta.append(f'%% fold_spatial {o[3]}')
        elif o[0] == 'avgpool3d_k2':
            terms.append('avgpool3d_k2')
        elif o[0] == 'sub_own_mean':
            # the self-referential spatial-mean subtract (#15):
            # only the all-spatial form (dims = every dim after
            # batch+chan) emits — the row-form (rows=B*C,
            # width=prod(spatial)):
            terms.append(f"sub_own_mean({','.join(str(d) for d in o[1])})")
        elif o[0] in ('relu', 'tanh', 'sigmoid', 'gelu', 'mish',
                      'hardswish', 'hardtanh'):
            terms.append(o[0])
        else:
            # ★ the REASON is already carried in o[1] and was being discarded.
            # An UNSUPPORTED term that does not say WHY sends the reader back to
            # the source to re-derive what the lifter already knew.
            _why = o[1] if len(o) > 1 and isinstance(o[1], str) else None
            terms.append('UNSUPPORTED_%s%s' % (o[0], (':' + _why) if _why else ''))
    fact = f"chain({pid}, [{', '.join(terms)}])."
    if terms_meta:
        fact += '\n' + '\n'.join(terms_meta)
    return fact


def _drop_dead_saves(ops):
    """THE DEAD-SAVE DROP (#21): a ('save', NAME) with no downstream
    consumer (add/mul/sub_saved) is dead code — the residual branch
    that would consume it was dropped by the If-resolve
    (use_residual=False). A dead save emits an identity kernel with
    misaligned comparison boundaries (the seg0 5.6-differ). Applied
    at the render choke-points (all lift paths flow through)."""
    _consumed = {o[1] for o in ops if o[0] in
                 ('add_saved', 'mul_saved', 'sub_saved')}
    return [o for o in ops if not (
        o[0] == 'save' and o[1] not in _consumed)]


def to_prolog_segments(pid, ops):
    ops = _drop_dead_saves(ops)
    ops = _avgpool_guard(ops)
    """Multi-stage form: one chain fact PER inter-stage segment that has
    ops. Returns (facts, n_segments, ok). Each segment must be fully
    expressible (no UNSUPPORTED terms) or ok=False (honest GAP —
    never emit a partial segment)."""
    # split ops by stage markers into segments:
    segments = []   # (after_stage_idx, [ops], stage_name)
    cur = []
    stage_no = 0
    empties = []    # (seg_idx, between-stages note) — the LEGIBLE ZEROS
    prev_stage = 'input'
    loop_meta = []  # bookkeeping terms (size_unpack, loop markers):
                    # SEGMENT-TRANSPARENT — the wrapper's forward
                    # replication carries them; the fuse-step never
                    # sees them (they have no kernel content).
    for o in ops:
        if o[0] == 'opaque_composites':
            for _attr, _cls, _why in o[1]:
                loop_meta.append(('opaque_composite',
                                  _attr, _cls, _why))
            continue
        if o[0] in ('size_unpack', 'size_binding', 'loop_begin',
                    'loop_end', 'loop_collect',
                    'branch_fanout_begin', 'branch_begin',
                    'branch_end', 'branch_fanout_end', 'branch_ref'):
            loop_meta.append(o)
            continue
        if o[0] == 'stage':
            if cur:
                segments.append((stage_no, cur))
            elif stage_no > 0:
                # consecutive stages with NOTHING between: emit a
                # legible zero, not a silent absence (Mavdil's #84
                # seg1 question — 'absence is ambiguous by
                # construction'; the reader must see the segmenter
                # CONSIDERED it):
                empties.append((stage_no, f'{prev_stage}→{o[1]}'))
            cur = []
            stage_no += 1
            prev_stage = o[1]
        else:
            cur.append(o)
    if cur:
        segments.append((stage_no, cur))
    facts = []
    ok = True
    for seg_idx, seg_ops in segments:
        fake = [('stage', 'seg')] + seg_ops   # reuse to_prolog's epilogue path
        fact = to_prolog(f'{pid}_seg{seg_idx}', fake)
        if 'UNSUPPORTED' in fact:
            ok = False
        facts.append(fact)
    for seg_idx, note in empties:
        facts.append(f'%% seg{seg_idx}: no fusable content ({note}, '
                     f'consecutive torch stages) — considered, not dropped')
    for o in loop_meta:
        if o[0] == 'opaque_composite':
            facts.append(f'%% opaque_composite {o[1]} {o[2]} {o[3]}')
        else:
            facts.append(f'%% loop_meta: {o!r}')
    return facts, len(segments), ok


if __name__ == '__main__':
    path = sys.argv[1]
    pid = sys.argv[2] if len(sys.argv) > 2 else 'problem'
    ops = lift(path)
    if len(sys.argv) > 3 and sys.argv[3] == '--wrapper-info':
        # THE WRAPPER-EMITTER's export (the whole-model gate, Heath's
        # unlock): the full ops walk as JSON — stage markers (torch's
        # prefix, in order) + the lifted tail ops. The split point IS
        # where the leading stage-run ends.
        import json
        # the SAME guard the emit path applies (the #50 skip: the
        # wrapper counted the RAW walk — 1 lifted run — while the
        # emit's guard had reverted mid-chain avgpools to stages —
        # 2 segments; the walks MUST match):
        print(json.dumps({'pid': pid,
                          'ops': [list(o)
                                  for o in _avgpool_guard(ops)]}))
        sys.exit(0)
    if len(sys.argv) > 3 and sys.argv[3] == '--segments':
        facts, n, ok = to_prolog_segments(pid, ops)
        for f in facts:
            print(f)
        if not ok:
            sys.exit(4)
    else:
        print(to_prolog(pid, ops))
