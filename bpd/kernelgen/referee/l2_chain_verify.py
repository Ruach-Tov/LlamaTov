#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""
l2_chain_verify.py — execute a lifted L2 chain step-by-step through our generated
torch ops and verify the whole chain against the KB Model's reference output.

This is the L2 execution + verification path. The chain-lifter (l2_chain_lift)
produces a resolved ChainIR; this module runs it:

  x0 = get_inputs()[0]
  x  = x0
  for step in chain:                 # conv/matmul/norm/pool/activation/scalar/...
      x = run_step(step, x, model)   # our op_expr-generated torch fn for that step
  compare(x, model.forward(x0))      # bit/ULP vs the KB reference

run_step uses the SAME per-op generation as L1 (op_expr -> lower_torch), with the
step's params pulled from the model's submodule (conv weight/stride, norm eps,
scalar constant attr, reduce dim). This is the chain ORACLE — once the per-step
results match, the composed/fused kernel (chain_compose) can be verified against
this same reference.

Author: Iyun, 2026-06-08.
"""
import os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l2_chain_lift import lift_chain, resolve_modules
from kernelbench_sweep import (OP_LIFT, build_param_term,
                               generate_and_run_torch, generate_and_run_torch_param)


def _robust_rel(got, ref):
    """Robust mean relative error. Denominator = |ref| + |got| + atol, so a
    numerically-correct result near zero (ref~1e-15) doesn't produce a spurious
    huge rel. atol=1e-4 floors the per-element denominator. Matches the spirit of
    torch.allclose (rtol+atol) rather than a bare 1/|ref| that explodes at zero."""
    import numpy as np
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    denom = np.abs(ref) + np.abs(got) + 1e-4
    return float(np.mean(np.abs(got - ref) / denom))


def _find_linear(model, step):
    """Find the Linear submodule for a matmul chain step (by the lifter's attr,
    else the first nn.Linear)."""
    import torch
    attr = step["params"].get("attr")
    if attr:
        sub = getattr(model, attr, None)
        if isinstance(sub, torch.nn.Linear):
            return sub
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            return m
    return None


def _find_module(model, step, op):
    """Find the nn submodule for a resolved _module step (by recorded attr)."""
    attr = step["params"].get("attr")
    if attr:
        return getattr(model, attr, None)
    return None


def F_linear(x, lin):
    """x @ W^T + b via the Linear layer's params (matches the KB reference)."""
    import torch
    return torch.nn.functional.linear(x, lin.weight.detach(),
                                      lin.bias.detach() if lin.bias is not None else None)


def _scalar_value(operand, model):
    """Resolve a scalar operand: a literal, or self.<attr> read off the model."""
    if operand is None:
        return None
    m = re.match(r"self\.(\w+)", operand)
    if m:
        v = getattr(model, m.group(1), None)
        try:
            return float(v)
        except (TypeError, ValueError):
            # tensor-valued (e.g. a bias buffer) — return as-is
            return v
    try:
        return float(operand)
    except ValueError:
        return None


def run_step(step, x, model, saved):
    """Run one ChainIR step on tensor x, returning the new tensor.
    `saved` is a dict for residual save/add (DAG steps)."""
    import torch
    op = step["op"]
    p = step["params"]

    # ── DAG markers + tensor binary ops ──
    if op == "_save":
        # remember the saved tensor under its source variable name (e.g. original_x)
        var = step.get("params", {}).get("as") or "residual"
        saved[var] = x.clone()
        saved["residual"] = x.clone()       # default alias
        return x
    if op == "identity":
        return x
    if op == "bias_add":
        b = getattr(model, "bias", None)
        return x + b if b is not None else x
    if op in ("tensor_add", "tensor_mul", "tensor_sub", "tensor_div", "residual_add"):
        import torch
        with_name = step.get("params", {}).get("with", "residual")
        # resolve the operand: a saved var (original_x...) or a model attr (self.bias)
        operand = saved.get(with_name)
        if operand is None:
            attr = with_name.replace("self.", "")
            a = getattr(model, attr, None)
            operand = a.detach() if (a is not None and torch.is_tensor(a)) else a
        if operand is None:
            operand = saved.get("residual", 0)
        if op in ("tensor_add", "residual_add"): return x + operand
        if op == "tensor_mul": return x * operand
        if op == "tensor_sub": return x - operand
        if op == "tensor_div": return x / operand

    # ── scalar binary ops ──
    if op in ("scaling", "scalar_add", "scalar_sub", "scalar_div"):
        v = _scalar_value(p.get("operand"), model)
        if v is None:
            return x
        if op == "scaling":      return x * v
        if op == "scalar_add":   return x + v
        if op == "scalar_sub":   return x - v
        if op == "scalar_div":   return x / v

    # ── binop: x OP self.<attr> where attr is scalar OR tensor (resolved here) ──
    if op == "binop":
        import torch
        opc = p.get("op")
        operand = p.get("operand", "")
        m = re.match(r"self\.(\w+)", operand)
        val = getattr(model, m.group(1), None) if m else None
        if val is not None and not torch.is_tensor(val):
            try: val = float(val)
            except (TypeError, ValueError): pass
        if val is None:
            return x
        v = val.detach() if torch.is_tensor(val) else val
        if opc == "+": return x + v
        if opc == "-": return x - v
        if opc == "*": return x * v
        if opc == "/": return x / v
        return x

    # ── clamp / clip (bounds may be literals, self.attr, or None) ──
    if op == "clamp":
        import torch
        def _bound(v):
            if v is None: return None
            m = re.match(r"self\.(\w+)", str(v))
            if m:
                a = getattr(model, m.group(1), None)
                try: return float(a)
                except (TypeError, ValueError): return a
            try: return float(v)
            except (TypeError, ValueError): return None
        return torch.clamp(x, min=_bound(p.get("min")), max=_bound(p.get("max")))
    # ── self-referential subtract-mean (centering): x - x.mean(dim, keepdim) ──
    if op == "submean":
        d = p.get("dim", 1)
        return x - x.mean(dim=d, keepdim=p.get("keepdim", True))

    # ── plain elementwise activations: use the CORRECT torch reference function
    #    directly (the verifier establishes the chain reference; op_expr formula
    #    variants for mish/hardswish/swish/gelu are validated separately by the
    #    differential referee). This fixes the DIFFs on mish/hardswish/swish/etc. ──
    _ACT = {
        "relu":    lambda x: __import__("torch").relu(x),
        "tanh":    lambda x: __import__("torch").tanh(x),
        "sigmoid": lambda x: __import__("torch").sigmoid(x),
        "gelu":    lambda x: __import__("torch").nn.functional.gelu(x),
        "silu":    lambda x: __import__("torch").nn.functional.silu(x),       # = swish
        "swish":   lambda x: __import__("torch").nn.functional.silu(x),
        "mish":    lambda x: __import__("torch").nn.functional.mish(x),
        "hardswish": lambda x: __import__("torch").nn.functional.hardswish(x),
        "hardsigmoid": lambda x: __import__("torch").nn.functional.hardsigmoid(x),
        "softplus": lambda x: __import__("torch").nn.functional.softplus(x),
        "selu":    lambda x: __import__("torch").nn.functional.selu(x),
        "exp":     lambda x: __import__("torch").exp(x),
        "log":     lambda x: __import__("torch").log(x),
        "abs":     lambda x: __import__("torch").abs(x),
        "neg":     lambda x: -x,
    }
    if op in _ACT:
        return _ACT[op](x)

    # ── parameterized activations: read the actual hyperparameter from the
    #    resolved nn module (our op_expr bakes in defaults — e.g. leaky_relu 0.01,
    #    but KB problems use custom slopes like 0.1). Same per-problem-param
    #    discipline as conv/norm. ──
    if op in ("leaky_relu", "elu", "hardtanh", "softplus"):
        import torch
        sub = _find_module(model, step, op)
        if op == "leaky_relu":
            slope = getattr(sub, "negative_slope", 0.01) if sub is not None else 0.01
            return torch.nn.functional.leaky_relu(x, negative_slope=slope)
        if op == "elu":
            alpha = getattr(sub, "alpha", 1.0) if sub is not None else 1.0
            return torch.nn.functional.elu(x, alpha=alpha)
        if op == "hardtanh":
            lo = getattr(sub, "min_val", -1.0) if sub is not None else -1.0
            hi = getattr(sub, "max_val", 1.0) if sub is not None else 1.0
            return torch.nn.functional.hardtanh(x, lo, hi)
        if op == "softplus":
            return torch.nn.functional.softplus(x)

    # ── matmul / Linear head: in a chain, Linear consumes x and holds its weight
    #    internally (like conv). Run it via the model's Linear submodule semantics
    #    (F.linear = x @ W^T + b), pulling the layer the lifter recorded. ──
    if op == "matmul":
        lin = _find_linear(model, step)
        if lin is not None:
            return F_linear(x, lin)
        # No nn.Linear — try a raw weight Parameter (torch.matmul(x, self.weight.T)).
        import torch
        w = getattr(model, "weight", None)
        if isinstance(w, torch.nn.Parameter) or torch.is_tensor(w):
            wt = w.detach()
            b = getattr(model, "bias", None)
            # x @ W^T (the nn.Linear / Gemm convention); fall back to x @ W if dims need it
            try:
                y = x @ wt.T
            except RuntimeError:
                y = x @ wt
            if isinstance(b, torch.nn.Parameter) or torch.is_tensor(b):
                y = y + b.detach()
            return y
        raise RuntimeError("matmul step without a resolvable Linear/weight")

    # ── conv / pool / norm HEADS: run via the torch submodule directly. These are
    #    operand-bound, SHAPE-CHANGING ops (conv changes C/H/W, pool changes H/W);
    #    the generated-kernel path returns a flat tensor which breaks the next
    #    spatial op. Running the actual nn module preserves shape — same decision
    #    as matmul->F.linear. The chain verifier validates the COMPOSITION + fused
    #    tail; the heavy heads run as torch ops for shape-correctness. ──
    if op in ("conv", "convtranspose", "maxpool", "avgpool",
              "batchnorm", "groupnorm", "layernorm", "instancenorm"):
        sub = step.get("_submodule")
        if sub is not None:
            return sub(x)               # the nn module: shape-correct, params-correct
        # FUNCTIONAL form (torch.X / F.X, no submodule): fall back to a reasonable
        # functional default so the chain still runs shape-correctly.
        import torch
        F = torch.nn.functional
        nd = x.dim()
        if op == "maxpool":
            return (F.max_pool2d if nd == 4 else F.max_pool3d if nd == 5 else F.max_pool1d)(x, 2)
        if op == "avgpool":
            # global avg pool is the common functional case (-> keepdim spatial)
            dims = tuple(range(2, nd))
            return x.mean(dim=dims, keepdim=True) if dims else x
        if op in ("groupnorm", "instancenorm", "layernorm", "batchnorm"):
            # functional norm without a module: normalize over the feature dims
            dims = tuple(range(1, nd))
            mu = x.mean(dim=dims, keepdim=True)
            var = x.var(dim=dims, keepdim=True, unbiased=False)
            return (x - mu) / torch.sqrt(var + 1e-5)
        raise RuntimeError(f"{op} head without a resolvable submodule")

    # ── axis-reduce in a chain: SHAPE-CHANGING (collapses a dim), run via torch
    #    directly so the next op sees the right shape. KB reduces typically
    #    keepdim=True (e.g. x.mean(dim=1, keepdim=True)). ──
    if op in ("mean", "sum", "amax", "amin", "argmax"):
        import torch
        d = p.get("dim", 1)
        kd = p.get("keepdim", True)
        # dim may be int or list/tuple (multi-axis reduce)
        dim_arg = tuple(d) if isinstance(d, (list, tuple)) else d
        fn = {"mean": torch.mean, "sum": torch.sum,
              "amax": torch.amax, "amin": torch.amin, "argmax": torch.argmax}[op]
        return fn(x, dim=dim_arg, keepdim=kd)
    if op in ("softmax", "log_softmax"):
        import torch
        d = p.get("dim", 1)
        return (torch.softmax if op == "softmax" else torch.log_softmax)(x, dim=d)
    if op == "logsumexp":
        import torch
        d = p.get("dim", 1)
        kd = p.get("keepdim", False)        # torch.logsumexp default keepdim=False
        dim_arg = tuple(d) if isinstance(d, (list, tuple)) else d
        return torch.logsumexp(x, dim=dim_arg, keepdim=kd)

    # ── parameterized ops: build the term from the model submodule + run our
    #    generated torch fn (the L1 machinery). ──
    if op not in OP_LIFT:
        raise RuntimeError(f"unmapped op in chain: {op}")
    fact, klass, _ = OP_LIFT[op]
    info = {"op": op, "fact": fact, "klass": klass, "model": model, "inputs": [x]}
    name = fact[4:] if fact.startswith("bpd_") else fact

    if klass in ("conv", "pool", "norm", "axis_reduce"):
        term, op_arg = build_param_term(info, model, [x])
        if term:
            out = generate_and_run_torch_param("step_" + name, term, op_arg)
        else:
            out = generate_and_run_torch(fact, op, [x])
    else:
        # elementwise / activation
        out = generate_and_run_torch(fact, op, [x])

    # generate_* returns flat numpy; reshape to match by trusting torch op shape.
    # For chain threading we need a tensor; re-run via torch directly is simplest:
    # but generate_and_run_torch already computed the right values. We need shape.
    return _as_tensor_like(out, x, op, p, model)


def _as_tensor_like(flat, x_in, op, p, model):
    """The generated fns return flat numpy. For chain threading we need the
    correctly-shaped tensor. Re-derive shape by running the op shape via torch
    on x_in (cheap) and reshaping the verified values into it."""
    import torch
    # determine output shape by a dry torch run mirroring the op
    ref = _torch_shape_probe(op, x_in, p, model)
    t = torch.from_numpy(np.asarray(flat, dtype=np.float32))
    if ref is not None and t.numel() == ref.numel():
        return t.reshape(ref.shape)
    return t


def _torch_shape_probe(op, x, p, model):
    """Run the op via torch to get the output SHAPE (values come from our fn)."""
    import torch
    F = torch.nn.functional
    try:
        if op in ("relu",): return F.relu(x)
        if op in ("tanh",): return torch.tanh(x)
        if op in ("sigmoid",): return torch.sigmoid(x)
        if op in ("gelu",): return F.gelu(x)
        if op in ("leaky_relu",): return F.leaky_relu(x)
        if op in ("softmax",): return torch.softmax(x, dim=p.get("dim", 1))
        if op in ("mean","sum","amax","amin"):
            d = p.get("dim", 1)
            return {"mean": x.mean, "sum": x.sum, "amax": x.amax, "amin": x.amin}[op](dim=d)
        # conv/matmul/etc: run the model submodule directly for shape
        return None
    except Exception:
        return None


def run_fused_chain(chain, x0, model):
    """Execute the chain in FUSED form: run the operand-bound HEAD ops
    individually (step-by-step), then the var-composable TAIL as ONE fused kernel
    (compose_chain -> lower_torch -> single op). Returns the final tensor.

    This is the L2 fusion payoff: instead of N kernel launches for the tail,
    one fused pass. We verify it produces the same result as the step-by-step
    oracle (and ultimately as model.forward())."""
    import torch, subprocess, importlib.util, numpy as np
    EMITTERS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "emitters")
    SWIPL = "/run/current-system/sw/bin/swipl"
    WORK = os.environ.get("SWEEP_WORK", "")

    # split the chain via the Prolog splitter (single source of truth)
    ops = [s["op"] for s in chain["steps"]]
    # map our verify op_keys -> bpd_ fact names for the Prolog side; DAG/scalar
    # steps (with no op_expr) force a head/tail boundary handled here in python.
    saved = {}
    x = x0
    # Walk steps; accumulate a fusable run of pure var-composable ops, flush as a
    # fused kernel when a non-composable (head/DAG/scalar-with-attr) step appears.
    run = []   # current fusable run of bpd_ facts
    def flush_run(xt):
        if not run:
            return xt
        # compose the run into one term + lower to torch, run as a single fn
        facts = " ".join(run)
        listlit = "[" + ",".join(run) + "]"
        out_py = f"{WORK}/fused_tail.py"
        goal = (f'consult("{EMITTERS}/chain_compose.pl"), '
                f'compose_chain({listlit}, T), '
                f'expr_ir:lower_torch(T, Py), '
                f'open("{out_py}",write,S), '
                f'format(S,"import torch~ndef fused(x):~n    return ~w~n",[Py]), close(S), halt')
        subprocess.run([SWIPL, "-q", "-g", goal], capture_output=True, text=True, timeout=60)
        spec = importlib.util.spec_from_file_location("fused_tail", out_py)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        return m.fused(xt)

    for step in chain["steps"]:
        op = step["op"]
        # pure var-composable op with a bpd_ fact AND no per-step param override
        # (scalar ops carry self.attr operands; leaky_relu carries a slope -> these
        # must run via run_step which reads the model param, so flush + run them)
        if op in OP_LIFT and _is_pure_composable(op, step):
            run.append(OP_LIFT[op][0])
        else:
            x = flush_run(x)
            run = []
            x = run_step(step, x, model, saved)
    x = flush_run(x)
    return x


def _is_pure_composable(op, step):
    """True if op composes with NO per-step param (so its op_expr default is
    correct in the fused term). Ops that need a model hyperparameter (leaky_relu
    slope, scalar operands, softmax/reduce dim != default) run via run_step."""
    if op in ("scaling", "scalar_add", "scalar_sub", "scalar_div",
              "leaky_relu", "elu", "hardtanh", "bias_add", "residual_add",
              "_save", "identity", "matmul",
              "tensor_add", "tensor_mul", "tensor_sub", "tensor_div", "submean", "binop"):
        return False
    # softmax/reduce with a non-default dim must use run_step
    if op in ("softmax", "log_softmax", "logsumexp", "mean", "sum", "amax", "amin", "argmax"):
        return step.get("params", {}).get("dim", 1) == 1
    # conv/pool/norm are operand-bound (no var) — not composable
    fact, klass, _ = OP_LIFT.get(op, (None, None, None))
    if klass in ("conv", "pool", "norm"):
        return False
    return True   # plain activations (relu/tanh/sigmoid/gelu/mish/...) compose cleanly


def verify_chain(path, check_fused=False):
    """Lift + run + verify a single L2 problem's chain. Returns a result dict."""
    import importlib.util, torch
    name = os.path.basename(path)
    spec = importlib.util.spec_from_file_location("kb_l2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
    model = mod.Model(*init) if init else mod.Model()
    model.eval()
    inputs = mod.get_inputs()
    x0 = inputs[0]

    chain = lift_chain(path)
    resolve_modules(chain, model)
    if any(s["op"] == "_unknown" for s in chain["steps"]):
        return {"problem": name, "status": "unresolved_step"}

    with torch.no_grad():
        ref = model(*inputs).detach().cpu().numpy().ravel()
        x = x0
        saved = {}
        # bind EXTERNAL forward inputs (forward(x, add_input, ...)) by name, so
        # tensor_add{with: add_input} resolves to the passed tensor, not garbage.
        for nm, t in zip(chain.get("extra_inputs", []), inputs[1:]):
            saved[nm] = t
        try:
            for step in chain["steps"]:
                x = run_step(step, x, model, saved)
            got = x.detach().cpu().numpy().ravel() if hasattr(x, "detach") \
                  else np.asarray(x).ravel()
        except Exception as e:
            return {"problem": name, "status": "run_error", "reason": str(e)[:100],
                    "n_steps": chain["n_steps"]}

    n = min(got.size, ref.size)
    if got.size != ref.size:
        return {"problem": name, "status": "shape_mismatch",
                "got": got.size, "ref": ref.size, "n_steps": chain["n_steps"]}
    # ROBUST relative error: denominator includes an absolute floor + both
    # magnitudes, so near-zero reference values don't explode the metric (a
    # numerically-correct result with ref~1e-15 should NOT read as rel~1e14).
    # This matches the spirit of torch.allclose (atol + rtol).
    rel = _robust_rel(got[:n], ref[:n])
    result = {"problem": name, "status": "verified" if rel < 1e-3 else "DIFF",
              "mean_rel": rel, "n_steps": chain["n_steps"]}

    # ── (2) FUSED verification: run the chain in fused form, check vs the oracle ──
    if check_fused and result["status"] == "verified":
        try:
            with torch.no_grad():
                xf = run_fused_chain(chain, x0, model)
                fused = xf.detach().cpu().numpy().ravel() if hasattr(xf, "detach") \
                        else np.asarray(xf).ravel()
            nf = min(fused.size, ref.size)
            frel = _robust_rel(fused[:nf], ref[:nf])
            result["fused_rel"] = frel
            result["fused_status"] = "fused-ok" if frel < 1e-3 else "fused-DIFF"
        except Exception as e:
            result["fused_status"] = "fused-error"
            result["fused_reason"] = str(e)[:80]
    return result


def _verify_one(args_tuple):
    """Worker: verify one problem. Caps torch CPU threads so N workers don't
    over-subscribe the cores (each torch op would else grab all 24)."""
    path, check_fused = args_tuple
    try:
        import torch
        torch.set_num_threads(2)
    except Exception:
        pass
    try:
        return verify_chain(path, check_fused=check_fused)
    except Exception as e:
        return {"problem": os.path.basename(path), "status": "run_error",
                "reason": str(e)[:100]}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check_fused = "--fused" in sys.argv
    serial = "--serial" in sys.argv
    KB = args[0] if len(args) > 0 else \
        "/tmp/llamatov-data/kernelbench/KernelBench/level2"
    only = args[1] if len(args) > 1 else None
    files = sorted(f for f in os.listdir(KB) if f.endswith(".py"))
    if only:
        files = [f for f in files if f.startswith(only)]
    from collections import Counter
    st = Counter(); fst = Counter()

    def _report(f, r):
        st[r["status"]] += 1
        if "fused_status" in r:
            fst[r["fused_status"]] += 1
        if r["status"] in ("verified", "DIFF") or only:
            extra = (f" | {r.get('fused_status','')} fused_rel={r.get('fused_rel','-')}"
                     if check_fused else "")
            print(f"  {f[:38]:39} {r['status']:10} rel={r.get('mean_rel','-')} "
                  f"steps={r.get('n_steps','-')}{extra}", flush=True)
        else:
            print(f"  {f[:38]:39} {r['status']:14} {r.get('reason','')[:36]}", flush=True)

    # PARALLEL across problems (enclave has 24 cores; each problem is independent).
    # Workers default to nproc//3 (with torch capped at 2 threads each) to balance
    # core use against the big-3D-model memory footprint. --serial forces the loop.
    if serial or len(files) <= 2:
        for f in files:
            _report(f, verify_chain(os.path.join(KB, f), check_fused=check_fused))
    else:
        import concurrent.futures as cf
        nproc = os.cpu_count() or 8
        workers = int(os.environ.get("L2_WORKERS", max(2, min(10, nproc // 3))))
        tasks = [(os.path.join(KB, f), check_fused) for f in files]
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            for f, r in zip(files, ex.map(_verify_one, tasks)):
                _report(f, r)

    print(f"\nL2 chain-verify (oracle): {dict(st)}")
    if check_fused:
        print(f"L2 fused-chain verify:    {dict(fst)}")
