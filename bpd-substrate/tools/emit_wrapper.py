#!/usr/bin/env python3
"""THE WRAPPER-EMITTER (the whole-model gate — Heath's unlock).

Generates a Model-equivalent Python wrapper per problem:
  torch runs the stage-prefix (MODULE-REUSE: the benchmark Model's own
  modules, same code, same mode — the mode-check lesson), our compiled
  CUDA kernel runs each lifted segment, torch stages replay BETWEEN
  segments where the walk interleaves them (the #8 shape).

The gate then becomes: get_inputs() -> Model AND wrapper -> ulp-diff.

Usage: emit_wrapper.py <problem.py> <pid> <kernel.cu> [out.py]
The kernel.cu must carry CALL-MANIFEST headers (launch geometry,
params) — the manifests ARE the wrapper's inputs.
"""
import json
import torch.nn.functional as _Fmod
import torch
import re
import subprocess
import sys
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def wrapper_source(problem_path, pid, ops, manifests):
    """Build the wrapper .py source. ops = the lift walk (stages +
    lifted tail ops). manifests = parsed CALL-MANIFEST dicts (one per
    emitted kernel, in launch order)."""
    # split the walk into runs: torch-stage runs vs lifted runs:
    runs = []
    cur_kind = None
    _META = ('size_unpack', 'size_binding', 'loop_begin',
             'loop_end', 'loop_collect',
             # the branch-cat topology brackets (#17's fan-out —
             # notation, not lifted runs):
             'branch_fanout_begin', 'branch_begin', 'branch_end',
             'branch_fanout_end')
    # THE FAN-OUT SOURCING TABLE (the #18 fix: run-splitting strips
    # the brackets [right], but run-SOURCING needs them — inside a
    # fanout, a branch's EXPECTED is the channel-SLICE of the cat-
    # consumer's input, and the post-cat stage's SRC is the true
    # cat of the branch tails):
    _fanouts = []   # {'stem': stage-name, 'dim': D, 'branches':
    #                  [[stage-names...], ...], 'cat_after': bool}
    _cur_fan = None
    _cur_branch = None
    _last_stage_name = None
    for o in ops:
        if o[0] == 'branch_fanout_begin':
            # NESTED-FANOUT NAMED-REFUSAL: today's lift refuses nested
            # customs at depth-1, so a nested branch_fanout_begin can
            # only reach here from an inline nested cat that a future
            # rung enables. A silent overwrite of _cur_fan would DROP
            # the outer fanout from _fanouts (branch_end and
            # branch_fanout_end silently no-op when _cur_fan is None).
            # Refuse loudly rather than gate wrong-numbers; the
            # fanout-stack shape is the general fix when nested is
            # actually enabled.
            if _cur_fan is not None:
                raise SystemExit(
                    f'{pid}: nested branch_fanout not supported '
                    f'— inner begin encountered while outer still '
                    f'open. Named refusal (the fanout-stack shape '
                    f'is the general fix; today\'s lift refuses '
                    f'nested customs at depth-1).')
            _cur_fan = {'stem': _last_stage_name, 'dim': o[2],
                        'branches': []}
            continue
        if o[0] == 'branch_begin':
            _cur_branch = []
            continue
        if o[0] == 'branch_end':
            if _cur_fan is not None and _cur_branch is not None:
                _cur_fan['branches'].append(_cur_branch)
            _cur_branch = None
            continue
        if o[0] == 'branch_fanout_end':
            if _cur_fan is not None:
                if any(not _b for _b in _cur_fan['branches']):
                    raise SystemExit(
                        f'{pid}: a fanout branch has no real '
                        f'module stages — cannot cap_out-source '
                        f'the cat. Named refusal (Mavhir '
                        f'guard-2: the empty-branch case).')
                _fanouts.append(_cur_fan)
            _cur_fan = None
            continue
        if o[0] == 'stage':
            _snm0 = o[1]
            if _cur_branch is not None and \
                    not _snm0.startswith(('m_', 'F_')):
                _cur_branch.append(_snm0)
            if not _snm0.startswith(('m_', 'F_')):
                _last_stage_name = _snm0
    for o in ops:
        if o[0] in _META:
            # loop/shape bookkeeping: NOT a lifted run (the gate's
            # multi-fire capture carries the loop; the wrapper's
            # forward delegates to inner). Skip from run-splitting:
            continue
        kind = 'stage' if o[0] == 'stage' else 'lifted'
        if kind != cur_kind:
            runs.append((kind, []))
            cur_kind = kind
        runs[-1][1].append(o)
    # THE SAVES TABLE (rung-3d): scoped-name -> anchor-path (the
    # save's value = cap_in[anchor] — the path-scoped residuals):
    _saves = {}
    for _o9 in ops:
        if _o9[0] == 'save' and len(_o9) > 2:
            _saves[_o9[1]] = _o9[2]
        elif _o9[0] == 'add_saved' and len(_o9) > 2 and \
                _o9[2] is not None:
            # the add's own anchor (the chain's first module —
            # its INPUT is the LHS value AT ADD-TIME) WINS over
            # any save-op anchor: the save may be STALE (the
            # thread reassigned since — the unshifted blocks'
            # 'x = shifted_x' save anchored to drop_path had
            # shadowed the norm2-anchor → v+v instead of v+x2):
            _saves[_o9[1]] = _o9[2]
    # pure-save runs emit NO kernel (the save marks a capture point;
    # the saved tensor reaches the kernel as x2 — #51):
    runs = [(k, ops_) for k, ops_ in runs
            if not (k == 'lifted'
                    and all(o[0] == 'save' for o in ops_))]
    # THE FIRE-SELECTOR (rung-3d): a lifted-run whose src-module
    # fires N-per-forward (drop_path in BOTH residuals) needs the
    # fire-index = the run's ordinal among runs sharing that src
    # (residual-1 → fire-0, residual-2 → fire-1). Computed
    # statically from the runs+stages sequence:
    _fire_sel = []
    _stage_count = {}
    _last_stage2 = None
    for _k10, _r10 in runs:
        if _k10 == 'stage':
            for _o10 in _r10:
                if not _o10[1].startswith(('m_', 'F_')):
                    _p10 = (_o10[3]
                            if len(_o10) > 3 and
                            isinstance(_o10[3], str)
                            else _o10[1])
                    _last_stage2 = _p10
                    _stage_count[_p10] = \
                        _stage_count.get(_p10, 0) + 1
        else:
            # the src's fire-index = OCCURRENCES of the src-path
            # in the stage-stream so far, minus 1 (mlp.drop
            # appears twice before the 2nd residual → fire-1):
            _fire_sel.append(
                max(0, _stage_count.get(_last_stage2, 1) - 1))

    n_lifted = sum(1 for k, _ in runs if k == 'lifted')
    # SPLIT-UNIT GROUPING (the #42 unlock): a split route emits
    # multiple kernels for ONE lifted run, with suffixed pids
    # (imp42_t, imp42_r) — group consecutive suffixed manifests into
    # one chain; the wrapper launches them sequentially (kernel i's
    # out feeds kernel i+1):
    mgroups = []
    for mn in manifests:
        mpid = mn.get('pid', '')
        _sfx = mpid[len(pid):].lstrip('_') if mpid.startswith(pid) \
            else ''
        _is_chain_part = (_sfx and not _sfx.startswith('seg'))
        if (_is_chain_part and mgroups):
            # non-seg suffix (_t/_r — the split route): chain into
            # the previous group (ONE lifted run, kernels sequential).
            # _segN manifests are SEPARATE lifted runs — never chained.
            mgroups[-1].append(mn)
        else:
            mgroups.append([mn])
    if n_lifted != len(mgroups):
        raise SystemExit(
            f'{pid}: {n_lifted} lifted runs but {len(mgroups)} '
            f'manifest groups ({len(manifests)} kernels) — the wrapper '
            f'cannot map them; refuse.')

    _has_input2 = any('input2' in str(mn.get('fact', ''))
                      for mn in manifests)
    lines = [
        f'"""Auto-emitted whole-model wrapper for {pid}.',
        'torch runs the stage-prefix (module-reuse); the emitted CUDA',
        'kernel(s) run the lifted tail. The gate: get_inputs() -> both',
        '-> ulp-diff."""',
        'import importlib.util as _ilu',
        'import torch.nn.functional as _F',
        'import torch',
        '',
        '# ABSOLUTE path resolved at emit time (Mavdil: a wrapper that',
        '# only runs from one directory is a wrapper the next person',
        '# reports as broken):',
        f'_spec = _ilu.spec_from_file_location('
        f'"_prob", {os.path.abspath(problem_path)!r})',
        '_prob = _ilu.module_from_spec(_spec)',
        '_spec.loader.exec_module(_prob)',
        '',
        '',
        'class WrappedModel(torch.nn.Module):',
        '    """Drop-in Model replacement: same init args, same forward',
        '    contract; the lifted tail runs on the emitted kernel."""',
        '',
        '    def __init__(self, *a, **kw):',
        '        super().__init__()',
        '        self.inner = _prob.Model(*a, **kw)',
        '        from tools.kernel_runner import KernelRunner',
        f'        self.runner = KernelRunner({pid!r})',
        '',
        ('    def forward(self, x, *extra):' if _has_input2
         else '    def forward(self, x):'),
    ]
    mi = 0
    for kind, run in runs:
        if kind == 'stage':
            for o in run:
                name = o[1]
                if name == 'm_matmul':
                    # the #14 GEMM stage: torch.matmul(x, W.T)
                    # with W = the recorded model attr:
                    _wa = (o[2] or ['weight'])[0]
                    lines.append(
                        f'        x = torch.matmul('
                        f'x, model.{_wa}.T)')
                    continue
                if name in ('m_split', 'm_expr', 'm_method',
                            'm_cat2', 'm_stack', 'm_index'):
                    # the QKV/fby stage kinds: hook-carried (the
                    # gate = hooks on the REAL forward; this
                    # decorative forward just marks them):
                    lines.append(
                        f'        # {name}: hook-carried stage')
                    continue
                if name.startswith('m_'):
                    # tensor-method shape-op replay (the #55/#98
                    # unlock): args recorded literal at lift:
                    margs = o[2] if len(o) > 2 else None
                    if margs is None:
                        # unresolved runtime args (B,T,C views): the
                        # forward here is DECORATIVE for hook-gated
                        # models (the gate = hooks on the REAL
                        # model's forward — it never runs this).
                        # Emit a placeholder, don't refuse the gate:
                        lines.append(
                            f'        # {name}: runtime-arg shape-op '
                            f'(hook-gate carries the real forward)')
                        continue
                    def _szs(a):
                        # __SIZE{i}__ / __SIZE{i}_DIV{n}__ /
                        # __SIZE{i}_MUL{n}__ (the rung-3c forms):
                        s = a[6:-2]
                        if '_DIV' in s:
                            i, n = s.split('_DIV')
                            return f'(x.size({i}) // {n})'
                        if '_MUL' in s:
                            i, n = s.split('_MUL')
                            return f'(x.size({i}) * {n})'
                        return f'x.size({s})'
                    # the ENTRY-MARK (source register): a
                    # snapshot assignment, not a method (the
                    # param_src lesson one layer over — the
                    # marker must be known to EVERY consumer;
                    # this renderer is the second):
                    if name == 'm_entry_mark':
                        _ek8 = (margs or [0])[0]
                        lines.append(
                            f'        x_entry{_ek8} = x')
                        continue

                    def _szs_entry(a):
                        # __ENTRY{k}_SIZE{i}__ →
                        # x_entry{k}.size(i)
                        import re as _re8
                        _m8 = _re8.match(
                            r'__ENTRY(\d+)_SIZE(.*)__$', a)
                        k, s = _m8.group(1), _m8.group(2)
                        if '_DIV' in s:
                            i, n = s.split('_DIV')
                            return (f'(x_entry{k}.size({i})'
                                    f' // {n})')
                        if '_MUL' in s:
                            i, n = s.split('_MUL')
                            return (f'(x_entry{k}.size({i})'
                                    f' * {n})')
                        return f'x_entry{k}.size({s})'
                    astr = ', '.join(
                        (_szs_entry(a)
                         if isinstance(a, str)
                         and a.startswith('__ENTRY')
                         else _szs(a)
                         if isinstance(a, str)
                         and a.startswith('__SIZE') else repr(a))
                        for a in margs)
                    lines.append(
                        f'        x = x.{name[2:]}({astr})')
                    continue
                if name.startswith('F_'):
                    fargs = o[2] if len(o) > 2 else None
                    if fargs is None:
                        raise SystemExit(
                            f'{pid}: functional stage {name} with '
                            f'unresolved args; refuse.')
                    _parts = []
                    for a in fargs:
                        if (isinstance(a, (tuple, list)) and len(a) == 2
                                and a[0] == '__KW__'):
                            for k2, v2 in a[1].items():
                                _parts.append(f'{k2}={v2!r}')
                        else:
                            _parts.append(repr(a))
                    astr = ', '.join(_parts)
                    sep = ', ' if astr else ''
                    _fn4 = name[2:]
                    _ns = ('_F' if hasattr(_Fmod, _fn4)
                           else 'torch')
                    lines.append(
                        f'        x = {_ns}.{_fn4}(x{sep}{astr})')
                    continue
                if '.' in name:
                    # container-indexed stage (network.0 — the L3
                    # container unlock): get_submodule resolves:
                    lines.append(
                        f'        x = self.inner.get_submodule('
                        f'{name!r})(x)')
                else:
                    lines.append(
                        f'        x = self.inner.{name}(x)')
        else:
            # launch every kernel in this run's manifest group
            # sequentially (split units chain: out_i -> in_{i+1}):
            _flat = sum(len(g) for g in mgroups[:mi])
            for _j in range(len(mgroups[mi])):
                _m2 = mgroups[mi][_j]
                if 'input2' in str(_m2.get('fact', '')):
                    lines.append(
                        f'        x = self.runner.launch('
                        f'{_flat + _j}, x, x2=extra[0])')
                else:
                    lines.append(
                        f'        x = self.runner.launch('
                        f'{_flat + _j}, x)')
            mi += 1
    lines += [
        '        return x',
        '',
        '',
        '# what the KERNEL computes (the gate asserts this against the',
        '# manifest chain — a wrapper must declare what it did NOT do;',
        "# Mavdil's contract item 3):",
        f'COVERS = {[o[0] for o in ops if o[0] != "stage"]!r}',
        f'STAGES = {[o[1] for o in ops if o[0] == "stage" and not o[1].startswith(("m_", "F_"))]!r}',
        '# the full stage sequence incl shape-ops (for gate replay):',
        f'ALL_STAGES = {[list(o[1:]) for o in ops if o[0] == "stage"]!r}',
        '# the size_unpack dim-names (B,T,C) for the M_FLOW env:',
        f'SIZE_NAMES = {next((list(o[2]) for o in ops if o[0] == "size_unpack" and len(o) > 2), [])!r}',
        '',
        '',
        f'RUNS = {[(k, [((o[1], (o[2] if len(o) > 2 else None)) if k == "stage" else o[0]) for o in r]) for k, r in runs]!r}',
        f'FANOUTS = {_fanouts!r}',
        f'SAVES = {_saves!r}',
        f'FIRE_SEL = {_fire_sel!r}',
        '',
        '',
        'def gate(seed=0):',
        '    def _szv(t, a):',
        '        # __SIZE{i}__ or __SIZE{i}_DIV{n}__/__SIZE{i}_MUL{n}__',
        '        # (the rung-3c symbolic-expr form):',
        '        s = a[6:-2]',
        '        if "_DIV" in s:',
        '            i, n = s.split("_DIV")',
        '            return t.size(int(i)) // int(n)',
        '        if "_MUL" in s:',
        '            i, n = s.split("_MUL")',
        '            return t.size(int(i)) * int(n)',
        '        return t.size(int(s))',
        '    """The whole-model gate, HOOK-CAPTURE form (Mavdil contract',
        '    item 1: cuDNN does NOT guarantee bit-identical output across',
        '    two invocations — the prefix must be captured from the',
        '    Model\'s OWN forward, never re-run; his #16 measurement:',
        '    3.5M false diffs re-run vs 0 hooked). ONE Model forward with',
        '    a hook on the last stage; the captured tensor feeds our',
        '    kernel; the Model\'s own output is the reference. No .eval()',
        '    (contract item 2: mode stated, stateful modules invoked',
        '    exactly once)."""',
        '    import torch as _t',
        '    _t.set_grad_enabled(False)   # the replay evals consume',
        '    # PARAMETERS (self.bias in m_flow sources) — without this',
        '    # their results track grad and .numpy() throws (#43):',
        '    import numpy as np',
        '    _t.manual_seed(seed)',
        '    init = _prob.get_init_inputs()',
        '    m = _prob.Model(*init).cuda()   # NO .eval() — the benchmark',
        '                                    # mode is the constructed mode',
        '    ins = [t.cuda() if hasattr(t, "cuda") else t',
        '           for t in _prob.get_inputs()]',
        '    # HOOK EVERY STAGE (Mavdil design (A) + the composition',
        '    # assertion): ONE Model forward captures every boundary;',
        '    # each lifted segment gates against its captured pair; the',
        '    # composition link is ASSERTED, not assumed.',
        '    _flat_stages = ALL_STAGES',
        '    cap_out, cap_in = {}, {}   # name -> LIST of fires',
        '    hooks = []',
        '    # RUNG-3d: dedupe (a module appearing at MULTIPLE',
        '    # stage-positions — drop_path in both residuals —',
        '    # must hook ONCE; double-hooks doubled every fire:',
        '    _seen_st = set()',
        '    _sv_anchors = [a[2:] if a.startswith("x_") else a',
        '                   for a in SAVES.values()]',
        '    _uniq_stages = [s for s in',
        '                    list(STAGES) + _sv_anchors',
        '                    if not (s in _seen_st or',
        '                            _seen_st.add(s))]',
        '    for s in _uniq_stages:',
        '        mod = m.get_submodule(s)',
        '        # CPU capture (Mavdil: the hook holding a GPU',
        '        # intermediate through the forward IS the OOM peak;',
        '        # off-device at capture, back at use):',
        '        hooks.append(mod.register_forward_hook(',
        '            lambda mo, i, o, _s=s:',
        '            cap_out.setdefault(_s, []).append('
        'o.detach().cpu())))',
        '        hooks.append(mod.register_forward_pre_hook(',
        '            lambda mo, i, _s=s:',
        '            cap_in.setdefault(_s, []).append('
        'i[0].detach().cpu())))',
        '    with _t.no_grad():',
        '        ref = m(*ins)               # the ONE forward',
        '    ref = ref.detach().cpu()',
        '    torch.cuda.empty_cache()  # the launch phase needs the',
        '    # headroom (ref+src+out together OOM the big units —',
        '    # the peak moves, it does not vanish; m stays for the',
        '    # runner param resolution)',
        '    for h in hooks:',
        '        h.remove()',
        '    from tools.kernel_runner import KernelRunner',
        f'    runner = KernelRunner({pid!r}, model=m)',
        '    results = {}',
        f'    GROUP_SIZES = {[len(g) for g in mgroups]!r}',
        '    ki = 0',
        '    stage_idx = -1',
        '    for ri, (kind, names) in enumerate(RUNS):',
        '        if kind == "stage":',
        '            stage_idx += len(names)',
        '            continue',
        '        # this lifted run: input = the preceding stage\'s',
        '        # captured OUTPUT; expected = the next stage\'s captured',
        '        # INPUT (or the model output if this run is last):',
        '        # the preceding boundary: the last MODULE stage\'s',
        '        # captured output, with any TRAILING shape-ops',
        '        # replayed (the #55 shape — pool(x.unsqueeze).squeeze):',
        '        _mods_before = [s for s in _flat_stages[:stage_idx + 1]',
        '                        if not s[0].startswith(("m_", "F_"))]',
        '        # MULTI-FIRE (the #34 loop): stages inside a host',
        '        # loop fire N times; gate EVERY fire (the per-step',
        '        # composition — the sequence gated step-by-step):',
        '        _n_fires = (len(cap_out[_mods_before[-1][0]])',
        '                    if _mods_before else 1)',
        '        # RUNG-3d: the fire-selector — this run pairs ONE',
        '        # fire (its ordinal among same-src runs); iterate',
        '        # only that index:',
        '        _fs = (FIRE_SEL[ki] if ki < len(FIRE_SEL)',
        '               else 0)',
        '        _fire_list = ([_fs] if _fs < _n_fires',
        '                      else list(range(_n_fires)))',
        '        _agg_nd, _agg_ulp, _agg_N = 0, 0, 0; _agg_abs = 0.0; _agg_ref = 0.0',
        '        for _fire in _fire_list:',
        '            import torch as _t_env, math as _m_env',
        '            import torch.nn.functional as _F_env',
        '            # M_FLOW per-fire env (the linker symbol table):',
        '            #   var-name -> tensor, threaded through the',
        '            #   replay so split/flow/expr/method resolve by',
        '            #   name. Seed size-names from ins[0] + self=m.',
        '            _env = {}',
        '            _sz = list(ins[0].size())',
        '            for _si, _sn in enumerate(SIZE_NAMES):',
        '                if _sn and _si < len(_sz):',
        '                    _env[_sn] = _sz[_si]',
        '            _side_thread = [None]',
        '            _entry_snap = {}',
        '            import re as _re9',
        '            if _mods_before:',
        '                src = cap_out[_mods_before[-1][0]][_fire].cuda()',
        '                # THE FAN-OUT SRC FIX (#18): if the preceding',
        '                # module is a branch-tail whose fanout CAT',
        '                # feeds this run, replay the TRUE cat of ALL',
        '                # the branch tails (sequence gave only the',
        '                # LAST branch):',
        '                for _fo2 in FANOUTS:',
        '                    if _fo2["branches"] and \\',
        '                            _mods_before[-1][0] == \\',
        '                            _fo2["branches"][-1][-1] and \\',
        '                            _flat_stages[stage_idx][0] != \\',
        '                            _fo2["branches"][-1][-1]:',
        '                        # (the tail\'s OWN relu run keeps its',
        '                        # plain cap_out src; only the POST-cat',
        '                        # run [anchored past the tail] gets',
        '                        # the true-cat replay)',
        '                        for _b2 in _fo2["branches"]:',
        '                            if _b2[-1] not in cap_out:',
        '                                raise SystemExit(',
        '                                    f"fanout branch tail '
        '{_b2[-1]!r} not hook-captured '
        '(not get_submodule-resolvable?) — '
        'named refusal (Mavhir guard-3)")',
        '                        src = torch.cat(',
        '                            [cap_out[_b2[-1]][_fire].cuda()',
        '                             for _b2 in _fo2["branches"]],',
        '                            dim=_fo2["dim"])',
        '                        break',
        '                _last_mod_pos = max(_j for _j, s in',
        '                                    enumerate(',
        '                                        _flat_stages[:stage_idx + 1])',
        '                                    if not s[0].startswith(',
        '                                        ("m_", "F_")))',
        '            else:',
        '                # NO module stage before this run (the #14 form:',
        '                # a functional/tensor-op prefix only) — replay',
        '                # from the raw input:',
        '                src = ins[0]',
        '                _last_mod_pos = -1',
        '            for _s in _flat_stages[_last_mod_pos + 1:stage_idx + 1]:',
        '                _nm = _s[0]',
        '                _args = _s[1] if len(_s) > 1 else None',
        '                if _nm == "m_split":',
        '                    # q,k,v = src.split(getattr(m,ATTR),dim)',
        '                    _sz_attr = _args[0]; _dim = _args[1]',
        '                    _names = _args[2:]',
        '                    _parts = src.split(',
        '                        getattr(m, _sz_attr), dim=_dim)',
        '                    for _nn, _pp in zip(_names, _parts):',
        '                        _env[_nn] = _pp',
        '                    src = _parts[-1]',
        '                elif _nm in ("m_flow", "m_expr",',
        '                             "m_method"):',
        '                    _own = _args[0]; _srcstr = _args[1]',
        '                    _res = eval(_srcstr,',
        '                        {"torch": _t_env, "math": _m_env,',
        '                         "F": _F_env, "self": m},',
        '                        _env)',
        '                    if _own:',
        '                        if not _t_env.is_tensor(_res):',
        '                            raise TypeError(',
        '                                "m_flow bind %s not a Tensor: %r"',
        '                                % (_own, type(_res)))',
        '                        _env[_own] = _res',
        '                    src = _res',
        '                elif _nm == "m_matmul":',
        '                    src = torch.matmul(',
        '                        src, getattr(m, (_s[1] or ["weight"])[0]).T)',
        '                elif _nm == "m_cat2":',
        '                    # THE SIDE-BRANCH CAT (#28: the',
        '                    # param-src branch [cls_token.expand]',
        '                    # feeds the cat BESIDE the thread —',
        '                    # cat(side, thread) restores the full',
        '                    # tensor; the 512-out bug was the side',
        '                    # REPLACING the thread):',
        '                    if _side_thread[0] is not None:',
        '                        _cd9 = (_s[1] or [1])[0]',
        '                        src = torch.cat(',
        '                            (src, _side_thread[0]),',
        '                            dim=(_cd9 if isinstance(',
        '                                _cd9, int) else 1))',
        '                        _side_thread[0] = None',
        '                    # else: the FAN-OUT SRC FIX above',
        '                    # already replayed it — skip.',
        '                elif _nm == "m_cat_growing":',
        '                    # #14: the stateful-cat-loop — the',
        '                    # thread = cat([x0] + the first K',
        '                    # element-outputs, dim). x0 = the',
        '                    # models INPUT; element-outputs from',
        '                    # cap_out at each elements LAST',
        '                    # stage ({attr}.{k}.{last}):',
        '                    _gd = _args[0]',
        '                    _gc = _args[1]',
        '                    _gattr = _args[2]',
        '                    _gops = [ins[0]]',
        '                    for _gk in range(_gc - 1):',
        '                        _gpre = f"{_gattr}.{_gk}."',
        '                        _glast = [s2[0] for s2 in',
        '                                  ALL_STAGES',
        '                                  if s2[0].startswith(',
        '                                      _gpre)][-1]',
        '                        _gl9 = cap_out[_glast]',
        '                        _gops.append(_gl9[min(_fire,',
        '                            len(_gl9) - 1)].cuda())',
        '                    src = torch.cat(_gops, dim=_gd)',
        '                elif _nm == "m_param_src":',
        '                    # #28: the chain sources from a named',
        '                    # Parameter (cls_token.expand...) — a',
        '                    # SIDE-branch: STASH the incoming',
        '                    # thread; the branch tail cats back:',
        '                    _side_thread[0] = src',
        '                    src = getattr(m, (_s[1] or [""])[0])'
        '.detach().float().cuda()',
        '                elif _nm == "m_entry_mark":',
        '                    _entry_snap[(_s[1] or [0])[0]',
        '                                if len(_s) > 1 else 0]'
        ' = src',
        '                elif _nm.startswith("m_"):',
        '                    # size-symbols resolve against the',
        '                    # THREAD (the forward x) — when a',
        '                    # side-branch is active, src is the',
        '                    # side (cls_token) and __SIZE0__',
        '                    # must read the STASHED thread (the',
        '                    # deferred-check trap, dodged again):',
        '                    _szsrc = (_side_thread[0]',
        '                              if _side_thread[0]',
        '                              is not None else src)',
        '                    def _eres(a):',
        '                        _em9 = _re9.match(',
        '                            r"__ENTRY(\\d+)_(SIZE.*)$", a)',
        '                        if _em9:',
        '                            return _szv(_entry_snap[',
        '                                int(_em9.group(1))],',
        '                                "__" + _em9.group(2))',
        '                        return _szv(_szsrc, a)',
        '                    _ra = [(_eres(a)',
        '                            if isinstance(a, str)',
        '                            and (a.startswith("__SIZE")',
        '                                 or a.startswith(',
        '                                     "__ENTRY"))',
        '                            else a)',
        '                           for a in (_s[1] or [])]',
        '                    src = getattr(src, _nm[2:])(*_ra)',
        '                elif _nm.startswith("F_"):',
        '                    import torch.nn.functional as _F2',
        '                    _fa, _fk = [], {}',
        '                    for a in (_s[1] or []):',
        '                        if (isinstance(a, (tuple, list))',
        '                                and len(a) == 2',
        '                                and a[0] == "__KW__"):',
        '                            _fk.update(a[1])',
        '                        else:',
        '                            _fa.append(a)',
        '                    _fn2 = _nm[2:]',
        '                    _mod2 = (_F2 if hasattr(_F2, _fn2)',
        '                             else torch)',
        '                    src = getattr(_mod2, _fn2)(',
        '                        src, *_fa, **_fk)',
        '            nxt = None',
        '            _suffix = []',
        '            _interv = []  # F_/m_ stages between our run and nxt',
        '            for kind2, names2 in RUNS[ri + 1:]:',
        '                if kind2 == "stage":',
        '                    if all((n if isinstance(n, str) else n[0])',
        '                           .startswith(("m_", "F_"))',
        '                           for n in names2):',
        '                        _suffix.extend(names2)',
        '                        _interv.extend(names2)',
        '                        continue',
        '                    for n in names2:',
        '                        _n0 = n if isinstance(n, str) else n[0]',
        '                        if _n0.startswith(("m_", "F_")):',
        '                            _interv.append(n)',
        '                        else:',
        '                            nxt = _n0',
        '                            break',
        '                    break',
        '            # expected STAYS CPU — the diff site reads via',
        '            # .cpu(); uploading it cost 2GB against the launch',
        '            # headroom (the #19 OOM):',
        '            # RUNG-3d GATE-FIX: the once-per-forward nxt-',
        '            # module (deep Swin stages fire once while the',
        '            # kernel-fire counter spans BOTH wrapper runs',
        '            # or multiple pairings) — clamp to the module',
        '            # own fire-count:',
        '            _nf = (len(cap_in[nxt]) if nxt is not None',
        '                   else 0)',
        '            expected = (cap_in[nxt][min(_fire, _nf - 1)]',
        '                        if nxt is not None and _nf > 0',
        '                        else ref)',
        '            # THE FAN-OUT SOURCING FIX (#18): a branch-tail',
        '            # run pairs with the cat-consumer, so expected =',
        '            # the channel-SLICE of cap_in[nxt] along the cat',
        '            # dim (branch i owns its contiguous span); the',
        '            # sequence-nxt for branch-A is branch-B\'s head',
        '            # (WRONG capture) — override via FANOUTS:',
        '            _tail_of = None',
        '            for _fo in FANOUTS:',
        '                for _bi2, _br in enumerate(_fo["branches"]):',
        '                    if _br and _mods_before and \\',
        '                            _mods_before[-1][0] == _br[-1]:',
        '                        _tail_of = (_fo, _bi2)',
        '            if _tail_of is not None:',
        '                _fo, _bi2 = _tail_of',
        '                # the cat consumer = the first real stage',
        '                # AFTER the fanout in RUNS-order; its cap_in',
        '                # is the CAT — slice our branch\'s span:',
        '                _dimF = _fo["dim"]',
        '                _catin = None',
        '                _seen_tail = False',
        '                for kind3, names3 in RUNS:',
        '                    if kind3 != "stage":',
        '                        continue',
        '                    for n3 in names3:',
        '                        _n30 = n3 if isinstance(n3, str) else n3[0]',
        '                        if _seen_tail and not \\',
        '                                _n30.startswith(("m_", "F_")):',
        '                            _catin = _n30',
        '                            break',
        '                        if _n30 == _fo["branches"][-1][-1]:',
        '                            _seen_tail = True',
        '                    if _catin:',
        '                        break',
        '                if _catin is not None:',
        '                    _full = cap_in[_catin][_fire]',
        '                    _off = 0',
        '                    for _bj in range(_bi2):',
        '                        _off += cap_out[_fo["branches"][_bj]',
        '                                        [-1]][_fire].size(_dimF)',
        '                    _span = cap_out[_fo["branches"][_bi2]',
        '                                    [-1]][_fire].size(_dimF)',
        '                    expected = _full.narrow(',
        '                        _dimF, _off, _span)',
        '                else:',
        '                    # UNREACHABLE-CAT-CONSUMER NAMED-REFUSAL: the',
        '                    # RUNS-order scan skips m_/F_ stages after',
        '                    # _seen_tail; if nothing real follows the',
        '                    # cat in RUNS, _catin stays None and expected',
        '                    # would silently fall to cap_in[nxt] — which',
        '                    # for a branch-tail run is the ORIGINAL',
        '                    # sequence-pairing bug (the wrong branch\\\'s',
        '                    # head). Refuse loudly rather than trust the',
        '                    # silent-failure path.',
        f'                    raise SystemExit(',
        f'                        \"{pid}: branch-tail run has no reachable \"',
        f'                        \"cat-consumer in RUNS-order; expected would \"',
        f'                        \"fall to sequence-nxt (the ORIGINAL bug). \"',
        f'                        \"Named refusal rather than trusting the \"',
        f'                        \"silent-failure path — the sub-rung is: \"',
        f'                        \"either extend the scan to reach past m_/F_ \"',
        f'                        \"chains, or handle the return-after-cat case \"',
        f'                        \"explicitly (expected = ref if this run\\\'s \"',
        f'                        \"cat is the model output).\")',
        '            _flat_ki = sum(GROUP_SIZES[:ki])',
        '            got = src',
        '            for _gj in range(GROUP_SIZES[ki]):',
        '                _mf = runner.manifests[_flat_ki + _gj]',
        '                if "input2" in str(_mf.get("fact", "")):',
        '                    got = runner.launch(_flat_ki + _gj, got,',
        '                                        x2=ins[1])',
        '                elif "add_saved" in str(_mf.get("fact", "")):',
        '                    # the saved tensor: an INTERMEDIATE capture',
        '                    # (#92: add_saved(x_conv) → cap_out[conv])',
        '                    # or the model INPUT (#51: original_x =',
        '                    # a save-seg0 of ins[0]):',
        '                    import re as _re2',
        '                    _sm = _re2.search(',
        r'                    "add_saved\\(.([\\w.:]+).\\)",',
        '                        str(_mf.get("fact", "")))',
        '                    _sv = _sm.group(1) if _sm else ""',
        '                    _x2 = None',
        '                    if _sv in SAVES and \\',
        '                            SAVES[_sv] in cap_in:',
        '                        _sl9 = cap_in[SAVES[_sv]]',
        '                        # the PROPORTIONAL fire-map (the',
        '                        # cross-run fix: fire-1-of-4 with a',
        '                        # 2-entry anchor -> idx (1*2)//4=0 -',
        '                        # the SAME run; the min-clamp had',
        '                        # read run-2 against run-1 v/exp):',
        '                        _x2i = ((_fire * len(_sl9)) //',
        '                                max(1, _n_fires))',
        '                        _x2 = _sl9[min(_x2i,',
        '                                       len(_sl9) - 1)]'
        '.cuda()',
        '                    if _x2 is None and \\',
        '                            _sv.startswith("x_") and \\',
        '                            _sv[2:] in cap_out:',
        '                        _x2 = cap_out[_sv[2:]][_fire].cuda()',
        '                    if _x2 is None and (',
        '                            "original_x" in _sv or',
        '                            _sv.endswith("::input")):',
        '                        # the #51 class ONLY (a save of',
        '                        # the model input itself):',
        '                        _x2 = ins[0]',
        '                    if _x2 is None:',
        '                        # NO SILENT FALLBACK (Mavdil:',
        '                        # a perfectly-shaped tensor of',
        '                        # unrelated values gates as a',
        '                        # plain DIFFERS — a trap with',
        '                        # no reachable negative):',
        '                        raise RuntimeError(',
        '                            f"x2-lookup MISSED for '
        '{_sv!r} (fire {_fire}) — '
        'refusing the input-image fallback")',
        '                    import os as _os9',
        '                    if _os9.environ.get(',
        '                            "W3V_DUMP_SEG") == str(ki):',
        '                        torch.save(',
        '                            {"v": got.cpu(),',
        '                             "x2": (_x2.cpu()',
        '                                    if _x2 is not None',
        '                                    else None),',
        '                             "v_contig":',
        '                             got.is_contiguous(),',
        '                             "x2_contig":',
        '                             (_x2.is_contiguous()',
        '                              if _x2 is not None',
        '                              else None),',
        '                             "v_dev": str(got.device),',
        '                             "x2_dev":',
        '                             (str(_x2.device)',
        '                              if _x2 is not None',
        '                              else None)},',
        '                            f"/tmp/w3v_dump_seg{ki}'
        '_launch.pt")',
        '                    got = runner.launch(_flat_ki + _gj, got,',
        '                                        x2=_x2)',
        '                else:',
        '                    got = runner.launch(_flat_ki + _gj, got)',
        '            # ISLAND REBIND (M_FLOW): got is the kernel',
        '            # output (post-relu). The tail flows (y=att@v)',
        '            # read the POST-kernel owner, so rebind _env',
        '            # [island_owner]=got. island_owner = the last',
        '            # m_flow/expr/method owner before this run.',
        '            _isl_owner = None',
        '            for _bs in _flat_stages[:stage_idx + 1]:',
        '                if _bs[0] in ("m_flow", "m_expr",',
        '                              "m_method") and len(_bs) > 1 \\',
        '                        and _bs[1] and _bs[1][0]:',
        '                    _isl_owner = _bs[1][0]',
        '            if _isl_owner is not None:',
        '                _env[_isl_owner] = got',
        '            if nxt is not None and _interv:',
        '                # THE INTERVENING-FUNCTIONAL REPLAY (the L3',
        '                # container fix: cap_in[nxt] is measured AFTER',
        '                # the F_/m_ stages between our run and nxt —',
        '                # apply them to got before comparing):',
        '                import torch.nn.functional as _F3',
        '                _got_entries = {}',
        '                for _iv in _interv:',
        '                    _ivn = _iv if isinstance(_iv, str) else _iv[0]',
        '                    _iva = ([] if isinstance(_iv, str)',
        '                            else (_iv[1] or []))',
        '                    if _ivn == "m_entry_mark":',
        '                        _gme = ((_iv[1] or [0])[0]',
        '                                if (not isinstance(',
        '                                    _iv, str)',
        '                                    and len(_iv) > 1)',
        '                                else 0)',
        '                        _got_entries[_gme] = got',
        '                        continue',
        '                    _fa2, _fk2 = [], {}',
        '                    for a in _iva:',
        '                        if (isinstance(a, (tuple, list))',
        '                                and len(a) == 2',
        '                                and a[0] == "__KW__"):',
        '                            _fk2.update(a[1])',
        '                        elif (isinstance(a, str)',
        '                              and a.startswith(',
        '                                  "__ENTRY")):',
        '                            _em8 = _re9.match(',
        '                                r"__ENTRY(\\d+)_'
        '(SIZE.*)$", a)',
        '                            _fa2.append(_szv(',
        '                                _got_entries[int(',
        '                                    _em8.group(1))],',
        '                                "__" + _em8.group(2)))',
        '                        elif (isinstance(a, str)',
        '                                and a.startswith("__SIZE")):',
        '                            _fa2.append(_szv(got, a))',
        '                        else:',
        '                            _fa2.append(a)',
        '                    if _ivn.startswith("F_"):',
        '                        _fn3 = _ivn[2:]',
        '                        # torch-namespace ops (flatten etc) vs',
        '                        # true F.* ops:',
        '                        _mod3 = (_F3 if hasattr(_F3, _fn3)',
        '                                 else torch)',
        '                        got = getattr(_mod3, _fn3)(',
        '                            got, *_fa2, **_fk2)',
        '                    elif _ivn in ("m_flow", "m_expr",',
        '                                  "m_method"):',
        '                        _res = eval(_iv[1][1],',
        '                            {"torch": torch, "math": _m_env,',
        '                             "F": _F3, "self": m},',
        '                            _env)',
        '                        if _iv[1][0]:',
        '                            _env[_iv[1][0]] = _res',
        '                        got = _res',
        '                    elif _ivn.startswith("m_"):',
        '                        got = (got if _ivn == "m_cat2"',
        '                               else getattr(',
        '                                   got, _ivn[2:])(*_fa2))',
        '            if nxt is None and _suffix:',
        '                # non-module suffix (F_/m_ stages): replay on OUR',
        '                # output; the comparison stays terminal vs ref:',
        '                _got_entries = {}',
        '                for _sfx in _suffix:',
        '                    _snm = _sfx if isinstance(_sfx, str) else _sfx[0]',
        '                    if (not isinstance(_sfx, str)',
        '                            and _sfx[0] ==',
        '                            "m_entry_mark"):',
        '                        _got_entries[((_sfx[1]',
        '                                       or [0])[0]',
        '                                      if len(_sfx) > 1',
        '                                      else 0)] = got',
        '                        continue',
        '                    _sar = ([] if isinstance(_sfx, str)',
        '                            else (_sfx[1] or []))',
        '                    if _snm in ("m_flow", "m_expr",',
        '                                "m_method"):',
        '                        _res = eval(_sfx[1][1],',
        '                            {"torch": torch, "math": _m_env,',
        '                             "F": __import__(',
        '                                "torch.nn.functional",',
        '                                fromlist=["x"]),',
        '                             "self": m},',
        '                            _env)',
        '                        if _sfx[1][0]:',
        '                            _env[_sfx[1][0]] = _res',
        '                        got = _res',
        '                    elif _snm.startswith("m_"):',
        '                        got = (got if _snm == "m_cat2"',
        '                               else getattr(',
        '                                   got, _snm[2:])(*_sar))',
        '                    else:',
        '                        import torch.nn.functional as _Fx',
        '                        got = getattr(_Fx, _snm[2:])(got, *_sar)',
        '            import os as _os8',
        '            if _os8.environ.get(',
        '                    "W3V_DUMP_SEG") == str(ki):',
        '                torch.save(',
        '                    {"got": got.cpu(),',
        '                     "expected": expected.cpu()},',
        '                    f"/tmp/w3v_dump_seg{ki}_cmp.pt")',
        '            r = expected.cpu().numpy().ravel()',
        '            g = got.cpu().numpy().ravel().reshape(r.shape)',
        '            ulp = np.abs(r.view(np.int32).astype(np.int64) -',
        '                         g.view(np.int32).astype(np.int64))',
        '            _agg_nd += int((r != g).sum())',
        '            _agg_ulp = max(_agg_ulp, int(ulp.max()))',
        '            _d64 = np.abs(r.astype(np.float64) - g.astype(np.float64))',
        '            _agg_abs = max(_agg_abs, float(_d64.max()) if _d64.size else 0.0)',
        '            _agg_ref = max(_agg_ref, float(np.abs(r.astype(np.float64)).max()) if r.size else 0.0)',
        '            _agg_N += int(r.size)',
        '        results[f"seg{ki}"] = {',
        '            "N": _agg_N, "n_diff": _agg_nd,',
        '            "max_ulp": _agg_ulp, "fires": _n_fires,',
        '            "max_abs": _agg_abs, "ref_absmax": _agg_ref,',
        '            "composition": "ASSERTED" if nxt is not None',
        '                           else "terminal"}',
        '        ki += 1',
        '    return results',
        '',
        '',
        'get_inputs = _prob.get_inputs',
        'get_init_inputs = _prob.get_init_inputs',
        '',
        '',
        'if __name__ == "__main__":',
        '    # self-demonstrating (Mavdil: a library with no __main__',
        '    # looks exactly like a hang):',
        '    print(gate())',
        '',
    ]
    src = '\n'.join(lines)
    # the runner import must work from ANY cwd (his PYTHONPATH catch):
    src = src.replace(
        'import importlib.util as _ilu',
        'import sys as _sys\n'
        f'_sys.path.insert(0, {REPO!r})  # tools.kernel_runner '
        '(any-cwd bootstrap)\n'
        'import importlib.util as _ilu')
    return src


def main():
    problem_path, pid, kernel_cu = sys.argv[1], sys.argv[2], sys.argv[3]
    out = sys.argv[4] if len(sys.argv) > 4 else f'wrapper_{pid}.py'
    r = subprocess.run(
        ['python3', os.path.join(REPO, 'lib', 'lift_chain.py'),
         problem_path, pid, '--wrapper-info'],
        capture_output=True, text=True)
    info = json.loads(r.stdout)
    cu = open(kernel_cu).read()
    manifests = [json.loads(m) for m in
                 re.findall(r'/\* CALL-MANIFEST (.*?) \*/', cu)]
    # split-units carry the unit manifest + sub-manifests; keep the
    # per-kernel ones (sub-pids) if present, else the unit one:
    subs = [m for m in manifests if m['pid'] != pid]
    manifests = subs if subs else manifests
    src = wrapper_source(problem_path, pid, info['ops'], manifests)
    with open(out, 'w') as f:
        f.write(src)
    print(f'[wrapper] {out} ({len(src.splitlines())} lines)')


if __name__ == '__main__':
    main()
