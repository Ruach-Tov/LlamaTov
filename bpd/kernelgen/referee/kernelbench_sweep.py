# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
import sys as _sys; _sys.path.insert(0, _os.path.join(_BPD, "lib"))
import toolchain as tc
#!/usr/bin/env python3
import os as _os, sys as _sys
def _bpd_root(_p=_os.path.dirname(_os.path.abspath(__file__))):
    while _p != '/' and _os.path.basename(_p) != 'bpd':
        _p = _os.path.dirname(_p)
    return _p if _os.path.basename(_p) == 'bpd' else _os.path.dirname(_os.path.abspath(__file__))
_BPD = _bpd_root()

"""
kernelbench_sweep.py — automated KernelBench lift + verify + benchmark driver.

═══════════════════════════════════════════════════════════════════════════════
WHAT THIS DOES (the big picture)
═══════════════════════════════════════════════════════════════════════════════

Stanford KernelBench (https://github.com/ScalingIntelligence/KernelBench) is a
benchmark of ~250 PyTorch reference computations across 4 levels:
  L1 = single ops (relu, matmul, ...), L2 = fused sequences, L3 = whole models.

Each problem is a Python file defining:
  - class Model(nn.Module) with a forward() = the REFERENCE computation
  - get_inputs()      = the test tensors
  - get_init_inputs() = constructor params

This driver AUTOMATES the full multi-backend pipeline for each problem:

  1. LIFT     — import the problem module; identify the op in forward();
                map it to our canonical fact (bpd/lib/robust_op_match.pl).
  2. GENERATE — emit a kernel for the op across our backends
                (cuda-c, MLIR-GPU) via the bpd/kernelgen emitters.
  3. VERIFY   — run the generated kernel on the GPU (P4) over the problem's
                own get_inputs(); compare bit-for-bit (ULP) against the
                reference Model.forward() — the CORRECTNESS gate (differential
                referee). Separates real ULP from NaN / signed-zero semantics.
  4. BENCHMARK— time the kernel (perf_fixture) and report throughput vs the P4
                roofline — the PERFORMANCE gate.
  5. RECORD   — one row per (problem, backend): lifted? generated? correct?
                maxULP, throughput, %peak. Written to a sweep table (JSON+TSV).

The result is a standing, reproducible sweep: point it at KernelBench and it
tells you, automatically, which problems our generator covers, whether the
generated kernels are bit-identical to PyTorch, and how fast they run — the
correctness gate and the perf gate, over the whole benchmark.

═══════════════════════════════════════════════════════════════════════════════
WHY THIS SHAPE
═══════════════════════════════════════════════════════════════════════════════

- Python driver: KernelBench problems ARE Python; importing them is the honest
  way to get the exact reference output + inputs (no re-implementing forward()).
- The reference is torch.Model.forward() run on CPU (PyTorch 2.7 cannot run on
  the P4 — Pascal sm_61 was dropped; so the GPU side is OUR kernel, the oracle
  is torch-CPU). This is the KernelBench-honest correctness definition.
- Lift is conservative: we only claim a problem if we can map its op to a fact
  we actually generate. Unmapped problems are recorded as 'unsupported' (honest
  coverage), not silently skipped.
- Bit-identity caveats from our prior findings are baked in: transcendentals
  (tanh/gelu/sigmoid/silu) differ from torch-CPU by ~1-2 ULP (GPU libdevice vs
  CPU SLEEF) — that is EXPECTED, not a failure; the driver flags it as
  'transcendental-ulp' rather than 'WRONG'. Matmul differs unless fma_mode and
  accumulation order match (the FMA-not-reduction-order finding).

═══════════════════════════════════════════════════════════════════════════════
HOW THE LIFT WORKS (op identification)
═══════════════════════════════════════════════════════════════════════════════

We identify the op by inspecting the problem's forward() source + a small set of
torch-call signatures, then map to our fact name. The mapping table (OP_LIFT)
mirrors bpd/tests/kernelbench_l1_problems.pl's op-kind decisions:

  torch.relu / F.relu / nn.ReLU        -> bpd_relu     (elementwise)
  torch.tanh / nn.Tanh                 -> bpd_tanh     (elementwise, transcendental)
  torch.sigmoid                        -> bpd_sigmoid  (elementwise, transcendental)
  F.elu / nn.ELU                       -> bpd_elu      (elementwise)
  F.gelu / gelu                        -> bpd_gelu     (elementwise, transcendental)
  x * sigmoid(x) / F.silu / Swish      -> bpd_silu     (elementwise, transcendental)
  torch.matmul / torch.mm / A @ B      -> bpd_matmul   (reduction)

Each lifted op carries its CLASS (elementwise|reduction) which selects the
backend emitter + the perf metric (GB/s vs GFLOPS).

═══════════════════════════════════════════════════════════════════════════════
USAGE
═══════════════════════════════════════════════════════════════════════════════
  python3 kernelbench_sweep.py [--level 1] [--problems 19,22,31] [--backends cuda_c,mlir_gpu]
                               [--limit N] [--out sweep_results]

Run on the GPU host (enclave) where the P4 + toolchain live.
Author: Iyun, 2026-06-07
"""

import os, sys, re, json, subprocess, importlib.util, argparse, struct, traceback
import numpy as np

# ── paths (the GPU-host layout) ────────────────────────────────────────────
REPO       = os.environ.get("RUACHTOV_REPO", "")
KB_ROOT    = os.environ.get("KERNELBENCH",   "")
KERNELGEN  = f"{REPO}/bpd/kernelgen"
EMITTERS   = f"{KERNELGEN}/emitters"
WORK       = os.environ.get("SWEEP_WORK",    "")
CUDA       = tc.cuda_root()  # shared toolchain (ENV-SHIFT defense)
REDIST     = "/nix/store/560i0agldlr2h4h3bx6mq2lifw6w1iaa-cuda-native-redist-12.8/lib"
STUBS      = "/nix/store/3n5kqxw44phkj9bcwdzdpj1z31q4ajg9-cuda_cudart-12.8.90-stubs/lib/stubs"
SWIPL      = "/run/current-system/sw/bin/swipl"
os.makedirs(WORK, exist_ok=True)

# ── the lift table: torch op -> (our fact, op class, transcendental?) ──────
# class drives emitter selection + perf metric; transcendental drives the
# expected-ULP interpretation (don't flag GPU-vs-CPU 1-ULP as WRONG).
OP_LIFT = {
    "relu":    ("bpd_relu",    "elementwise", False),
    "tanh":    ("bpd_tanh",    "elementwise", True),
    "sigmoid": ("bpd_sigmoid", "elementwise", True),
    "elu":     ("bpd_elu",     "elementwise", False),
    "gelu":    ("bpd_gelu",    "elementwise", True),
    "silu":    ("bpd_silu",    "elementwise", True),
    "matmul":  ("bpd_matmul",  "reduction",   False),
    "identity":("bpd_identity",  "elementwise", False),
    # ── extended op coverage (op_expr terms exist; torch backend verifies all) ──
    "leaky_relu": ("bpd_leaky_relu", "elementwise", False),
    "softplus":   ("bpd_softplus",   "elementwise", True),
    "softsign":   ("bpd_softsign",   "elementwise", False),
    "hardsigmoid":("bpd_hardsigmoid","elementwise", False),
    "hardtanh":   ("bpd_hardtanh",   "elementwise", False),
    "selu":       ("bpd_selu",       "elementwise", True),
    "gelu_tanh":  ("bpd_gelu_tanh", "elementwise", True),
    "softmax":    ("bpd_softmax",    "axis_reduce", True),
    "log_softmax":("bpd_log_softmax","axis_reduce", True),
    "sum":        ("bpd_sum",        "axis_reduce", False),
    "mean":       ("bpd_mean",       "axis_reduce", False),
    "amax":       ("bpd_max",        "axis_reduce", False),
    "amin":       ("bpd_min",        "axis_reduce", False),
    "argmax":     ("bpd_argmax",     "axis_reduce", False),
    "l1norm":     ("bpd_l1norm",     "norm",        False),
    "l2norm":     ("bpd_l2norm",     "norm",        True),
    "rmsnorm":    ("bpd_rmsnorm",    "norm",        True),
    "frobnorm":   ("bpd_frobnorm",   "norm",        True),
    "layernorm":  ("bpd_layernorm",  "norm",        True),
    "instancenorm":("bpd_instancenorm","norm",      True),
    "batchnorm":  ("bpd_batchnorm",  "norm",        True),
    "groupnorm":  ("bpd_groupnorm",  "norm",        True),
    "maxpool":    ("bpd_maxpool2d",  "pool",        False),
    "avgpool":    ("bpd_avgpool2d",  "pool",        False),
    "conv":       ("bpd_conv2d",     "conv",        False),
    "convtranspose":("bpd_conv_transpose2d","conv", False),
    "cross_entropy":("bpd_cross_entropy","loss",    True),
    "huber":      ("bpd_huber",      "loss",        False),
    "mse":        ("bpd_mse",        "loss",        False),
    "hinge":      ("bpd_hinge",      "loss",        False),
    "kl_div":     ("bpd_kl_div",     "loss",        True),
    "triplet":    ("bpd_triplet",    "loss",        False),
}

# regexes that recognize an op inside a forward() body (conservative: first match wins)
# ORDER MATTERS: more specific patterns first. silu (x*sigmoid(x)) must beat the
# bare sigmoid pattern; Swish problems literally contain "x * torch.sigmoid(x)".
LIFT_PATTERNS = [
    (re.compile(r"F\.silu|nn\.SiLU|Swish|swish|\*\s*torch\.sigmoid|sigmoid\(x\)\s*\*\s*x"), "silu"),
    (re.compile(r"F\.gelu|gelu|Gelu|GELU"),                         "gelu"),
    (re.compile(r"torch\.relu|F\.relu|nn\.ReLU|\.relu\("),          "relu"),
    (re.compile(r"torch\.tanh|nn\.Tanh|\.tanh\("),                  "tanh"),
    (re.compile(r"F\.elu|nn\.ELU"),                                 "elu"),
    (re.compile(r"torch\.sigmoid|nn\.Sigmoid|\.sigmoid\("),         "sigmoid"),
    (re.compile(r"torch\.matmul|torch\.mm|torch\.bmm|@\s"),         "matmul"),
    # ── extended patterns (specific before general) ──
    (re.compile(r"ConvTranspose|conv_transpose"),                   "convtranspose"),
    (re.compile(r"nn\.Conv|F\.conv|conv[123]d"),                    "conv"),
    (re.compile(r"MaxPool|max_pool|AdaptiveMax"),                   "maxpool"),
    (re.compile(r"AvgPool|avg_pool|AdaptiveAvg"),                   "avgpool"),
    (re.compile(r"leaky_relu|LeakyReLU"),                           "leaky_relu"),
    (re.compile(r"softplus|Softplus"),                              "softplus"),
    (re.compile(r"softsign|Softsign|/ \(1 \+ torch\.abs"),          "softsign"),
    (re.compile(r"hardsigmoid|HardSigmoid|Hardsigmoid"),            "hardsigmoid"),
    (re.compile(r"hardtanh|Hardtanh|HardTanh"),                     "hardtanh"),
    (re.compile(r"torch\.selu|F\.selu|nn\.SELU"),                   "selu"),
    (re.compile(r"0\.044715"),                                     "gelu_tanh"),
    # softmax + axis-reductions BEFORE norms (specific; norms can false-positive)
    (re.compile(r"log_softmax|LogSoftmax"),                         "softmax"),
    (re.compile(r"softmax|Softmax"),                                "softmax"),
    (re.compile(r"torch\.argmax|\.argmax\("),                       "argmax"),
    (re.compile(r"torch\.sum\(.*dim|\.sum\(.*dim"),                 "sum"),
    (re.compile(r"torch\.mean\(.*dim|\.mean\(.*dim"),               "mean"),
    (re.compile(r"torch\.max\(.*dim|\.max\(.*dim"),                 "amax"),
    (re.compile(r"torch\.min\(.*dim|\.min\(.*dim"),                 "amin"),
    # norms
    (re.compile(r"BatchNorm|batch_norm"),                           "batchnorm"),
    (re.compile(r"GroupNorm|group_norm"),                           "groupnorm"),
    (re.compile(r"InstanceNorm|instance_norm"),                     "instancenorm"),
    (re.compile(r"LayerNorm|layer_norm"),                           "layernorm"),
    (re.compile(r"RMSNorm|rms_norm|/ rms|rms ="),                  "rmsnorm"),
    (re.compile(r"p='fro'|Frobenius|frobenius"),                    "frobnorm"),
    (re.compile(r"p=2|L2Norm|l2_norm"),                            "l2norm"),
    (re.compile(r"mean\(torch\.abs|L1.*[Nn]orm"),                   "l1norm"),
    # losses
    (re.compile(r"cross_entropy|CrossEntropy"),                     "cross_entropy"),
    (re.compile(r"smooth_l1|Huber|huber"),                          "huber"),
    (re.compile(r"kl_div|KLDiv"),                                   "kl_div"),
    (re.compile(r"triplet|Triplet"),                               "triplet"),
    (re.compile(r"clamp\(1 ?- |[Hh]inge"),                          "hinge"),
    (re.compile(r"mse_loss|MSELoss"),                               "mse"),
]

# ── ULP comparison (shared with the differential referee) ──────────────────
def ulp_compare(a, b):
    """Return (max_ulp, n_differ, n_nan_mismatch, n_signed_zero) over finite elems."""
    a = np.ascontiguousarray(a, np.float32).ravel()
    b = np.ascontiguousarray(b, np.float32).ravel()
    n = min(a.size, b.size); a = a[:n]; b = b[:n]
    nan_mismatch = int((np.isnan(a) != np.isnan(b)).sum())
    both_zero = (a == 0) & (b == 0)
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    signed_zero = int((both_zero & (ai != bi)).sum())
    mask = ~(np.isnan(a) | np.isnan(b)) & ~both_zero
    u = np.abs(ai - bi); u[~mask] = 0
    return int(u.max()), int((u > 0).sum()), nan_mismatch, signed_zero


def lift_problem(path):
    """Parse a KernelBench problem file: identify the op from forward(); import
    the module to obtain Model + get_inputs (the reference + test data).
    Returns dict(op, fact, klass, transcendental, model, inputs) or dict(unsupported)."""
    src = open(path).read()
    name = os.path.basename(path)

    # 1-pre. definitive source signals that must beat the filename pass
    op_key = None
    if "0.044715" in src:
        op_key = "gelu_tanh"
    # 1a. FILENAME-priority pass — KB filenames are descriptive + unambiguous
    #     (36_RMSNorm_, 24_LogSoftmax, ...). Strongest signal; checked first to
    #     avoid source-regex false positives (e.g. RMSNorm's internal mean()).
    fn_lower = name.lower()
    FILENAME_OPS = [
        ("logsoftmax", "log_softmax"), ("log_softmax", "log_softmax"),
        ("rmsnorm", "rmsnorm"), ("frobenius", "frobnorm"), ("l1norm", "l1norm"),
        ("l2norm", "l2norm"), ("layernorm", "layernorm"), ("groupnorm", "groupnorm"),
        ("batchnorm", "batchnorm"), ("instancenorm", "instancenorm"),
        ("softmax", "softmax"), ("convtranspose", "convtranspose"),
        ("leakyrelu", "leaky_relu"), ("hardsigmoid", "hardsigmoid"),
        ("hardtanh", "hardtanh"), ("softplus", "softplus"), ("softsign", "softsign"),
    ]
    for needle, key in (FILENAME_OPS if op_key is None else []):
        if needle in fn_lower.replace("_", ""):
            op_key = key; break
    # 1b. fall back to source-pattern scan
    if op_key is None:
        for rx, key in LIFT_PATTERNS:
            if rx.search(src):
                op_key = key; break
    if op_key is None or op_key not in OP_LIFT:
        return {"problem": name, "status": "unsupported", "reason": "no recognized op in forward()"}

    fact, klass, transc = OP_LIFT[op_key]

    # 2. PRE-IMPORT shape guard — some KB Models allocate giant default tensors
    #    inside get_inputs() (e.g. 32768 x 65535 = 8.6GB) at import, before any
    #    cap can apply. Estimate the largest tensor from the module-level int
    #    constants and the rand/randn shapes; if it exceeds the cap, monkeypatch
    #    torch.rand* to clamp the dims before importing.
    cap = int(os.environ.get("SWEEP_MAX_ELEMS", "1048576"))
    import torch
    _orig = (torch.rand, torch.randn, torch.randint)
    _batch_map = {}   # original leading dim -> clamped leading dim (consistent
                      # across all tensors in this import: pred/target, x/labels)
    def _clamp_shape(args):
        # clamp dims so total elements <= cap, but PRESERVE the channel dim (idx 1)
        # for rank>=3 tensors (N,C,...) — conv/norm weights depend on C, and the
        # weight comes from the model (un-clamped), so clamping C breaks x-vs-w.
        dims = [a for a in args if isinstance(a, int)]
        if not dims: return args
        import math
        prod = math.prod(dims)
        if prod <= cap: return args
        # which positions are clampable: all except channel (index 1) when rank>=3
        clampable = set(range(len(dims)))
        if len(dims) >= 3:
            clampable.discard(1)
        fixed_prod = math.prod(dims[i] for i in range(len(dims)) if i not in clampable) or 1
        budget = max(1, cap // fixed_prod)
        nfree = len(clampable)
        free_prod = math.prod(dims[i] for i in clampable) or 1
        factor = (free_prod / budget) ** (1.0 / nfree) if free_prod > budget else 1.0
        newdims = list(dims)
        for i in clampable:
            newdims[i] = max(1, int(dims[i] / factor))
        # CONSISTENT leading (batch) dim across all tensors in this import, so
        # multi-tensor ops (loss pred/target, triplet a/p/n) keep aligned batches.
        if 0 in clampable and dims:
            newdims[0] = _batch_map.setdefault(dims[0], newdims[0])
        out, j = [], 0
        for a in args:
            if isinstance(a, int): out.append(newdims[j]); j += 1
            else: out.append(a)
        return tuple(out)
    def _guarded(fn):
        def g(*args, **kw): return fn(*_clamp_shape(args), **kw)
        return g
    torch.rand, torch.randn = _guarded(_orig[0]), _guarded(_orig[1])
    def _guarded_randint(*a, **k):
        # randint(low, high, size) — clamp the size's batch dim to match preds.
        if len(a) >= 3 and isinstance(a[2], (tuple, list)):
            sz = list(a[2])
            if sz and sz[0] in _batch_map:
                sz[0] = _batch_map[sz[0]]
            a = (a[0], a[1], tuple(sz)) + a[3:]
        return _orig[2](*a, **k)
    torch.randint = _guarded_randint
    try:
        spec = importlib.util.spec_from_file_location("kb_problem", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
        model = mod.Model(*init) if init else mod.Model()
        inputs = mod.get_inputs()
    except Exception as e:
        torch.rand, torch.randn, torch.randint = _orig
        return {"problem": name, "status": "lift_error", "op": op_key, "reason": str(e)[:120]}
    finally:
        torch.rand, torch.randn, torch.randint = _orig

    return {"problem": name, "status": "lifted", "op": op_key, "fact": fact,
            "klass": klass, "transcendental": transc, "model": model, "inputs": inputs}


def _t(x):
    """Format a scalar/tuple param as a single int for the op_expr term (KB uses
    square kernels/strides, so the first element suffices)."""
    if isinstance(x, (tuple, list)): return int(x[0])
    return int(x)

def _tup(x):
    """Format a param as a verbatim Python tuple/int literal for the op_expr term,
    preserving asymmetry: (3,5) stays (3,5); a scalar 3 stays 3. Tuples are
    rendered without spaces so they parse as a single Prolog atom argument."""
    if isinstance(x, (tuple, list)):
        return "(" + ",".join(str(int(v)) for v in x) + ")"
    return str(int(x))

def build_param_term(info, model, kb_inputs):
    """Extract the ACTUAL op params from the KB Model's nn submodule and build a
    parameterized op_expr term string + the operand arg. Returns (term, op_arg);
    term=None falls back to the fixed-default op_expr. This converts 'recognized'
    -> 'verified' by matching each problem's real stride/pad/kernel/dims/groups."""
    import torch
    klass = info["klass"]
    x = kb_inputs[0]
    if klass == "conv":
        conv = next((m for m in model.modules()
                     if isinstance(m, (torch.nn.modules.conv._ConvNd,))), None)
        if conv is None: return None, (x,)
        nd = {3:1, 4:2, 5:3}.get(x.dim(), 2)
        transp = 1 if isinstance(conv, torch.nn.modules.conv._ConvTransposeNd) else 0
        # full tuple params (verbatim Python tuples) so ASYMMETRIC kernels work.
        st, pad, dil = _tup(conv.stride), _tup(conv.padding), _tup(conv.dilation)
        b = conv.bias.detach() if conv.bias is not None else None
        if transp == 1:
            opad = _tup(getattr(conv, "output_padding", 0))
            term = f"conv({nd},1,{st},{pad},{dil},{conv.groups},{opad})"
        else:
            term = f"conv({nd},0,{st},{pad},{dil},{conv.groups})"
        return term, (x, conv.weight.detach(), b)
    if klass == "pool":
        pool = next((m for m in model.modules()
                     if isinstance(m, (torch.nn.modules.pooling._MaxPoolNd,
                                       torch.nn.modules.pooling._AvgPoolNd))), None)
        if pool is None: return None, x
        nd = {3:1, 4:2, 5:3}.get(x.dim(), 2)
        kind = "max" if isinstance(pool, torch.nn.modules.pooling._MaxPoolNd) else "avg"
        st = _t(pool.stride) if pool.stride else _t(pool.kernel_size)
        pad = _t(pool.padding); dil = _t(getattr(pool, "dilation", 1) or 1)
        term = f"pool({kind},{nd},{_t(pool.kernel_size)},{st},{pad},{dil},var)"
        return term, x
    if klass == "norm":
        # batch/group/instance/layer-norm: read eps; build over the right axis.
        nrm = next((m for m in model.modules()
                    if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                                      torch.nn.BatchNorm3d, torch.nn.GroupNorm,
                                      torch.nn.LayerNorm, torch.nn.InstanceNorm1d,
                                      torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d))), None)
        if nrm is None: return None, x      # vector norms (l1/l2/rms/frob) use defaults
        eps = getattr(nrm, "eps", 1e-5)
        if isinstance(nrm, torch.nn.GroupNorm):
            return f"groupnorm({nrm.num_groups},const({eps}),var)", x
        if isinstance(nrm, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            return f"batchnorm(const({eps}),var)", x
        # layernorm/instancenorm: standardize over the normalized axes. layernorm
        # normalizes the LAST len(normalized_shape) dims; express via stat_norm
        # over those dims (use a tuple axis). For the common case normalize over
        # the trailing dims -> we standardize over the last-dim axis as a proxy.
        return None, x   # layer/instance handled by their default stat_norm term
    if klass == "axis_reduce":
        dim = getattr(model, "dim", None)
        if dim is None: return None, x
        op = info["op"]
        kindmap = {"sum":"sum","mean":"mean","amax":"max","amin":"min","argmax":"argmax",
                   "max":"max","min":"min"}
        k = kindmap.get(op)
        if k: return f"axis_reduce({k},{int(dim)},var)", x
        # softmax/log_softmax carry a dim too
        if op in ("softmax",): return f"softmax({int(dim)},var)", x
        if op in ("log_softmax",): return f"log_softmax({int(dim)},var)", x
        return None, x
    return None, (tuple(kb_inputs) if klass == "loss" else x)

def reference_output(model, inputs):
    """Run the KernelBench reference: Model.forward() on CPU -> numpy f32."""
    import torch
    with torch.no_grad():
        out = model(*inputs)
    return out.detach().cpu().numpy().astype(np.float32)


# Cap the verification input size. KernelBench's default sizes can be enormous
# (ReLU: 4096*393216 = 1.6B elems = 6.4GB, exceeds the P4's 8GB). For an
# ELEMENTWISE op every element is processed identically, so a representative
# slice fully validates correctness. We keep edge cases (front of the buffer)
# plus a sampled tail. Perf measurement uses its own (larger) sizes separately.
MAX_VERIFY_ELEMS = int(os.environ.get("SWEEP_MAX_ELEMS", str(1 << 24)))  # 16M default

def cap_inputs(inputs):
    """Return inputs with the first tensor flattened + capped to MAX_VERIFY_ELEMS
    (1-D). For elementwise ops a representative slice fully validates; this keeps
    both the torch reference and the GPU run fast + within the P4's 8GB."""
    import torch
    t = inputs[0].reshape(-1)
    if t.numel() > MAX_VERIFY_ELEMS:
        t = t[:MAX_VERIFY_ELEMS].contiguous()
    return [t] + list(inputs[1:])

def generate_and_run_elementwise(fact, op_key, inputs):
    """Generate the C++/CUDA kernel for an elementwise op, run on the P4 over the
    (already-capped) flattened input, return GPU output (numpy f32).
    (cuda-c backend: most robust path; MLIR-GPU added once parity lands.)"""
    import numpy as np
    x = inputs[0].detach().cpu().numpy().astype(np.float32).ravel()
    n = x.size
    x.tofile(f"{WORK}/in.bin")

    # emit the cuda-c kernel via the Prolog emitter, then a thin runner that
    # reads in.bin -> writes out.bin. We reuse emit_cuda_c_dump.
    name = fact[4:] if fact.startswith("bpd_") else fact
    swi = (f'consult("{EMITTERS}/cuda_c_from_facts.pl"), '
           f'emit_cuda_c_dump({fact}, "{WORK}/in.bin", "{WORK}/out.bin", "{WORK}/{name}.cu"), halt')
    r = subprocess.run([SWIPL, "-q", "-g", swi], capture_output=True, text=True, timeout=60)
    if not os.path.exists(f"{WORK}/{name}.cu"):
        raise RuntimeError(f"emit failed: {r.stderr[:200]}")

    # compile + run
    nvcc = f"{CUDA}/bin/nvcc"
    env = tc.nvcc_env()
    subprocess.run([nvcc, "-O3", "-arch=sm_61", "-Wno-deprecated-gpu-targets",
                    "-cudart", "shared", f"-I{CUDA}/include",
                    f"-L{CUDA}/lib", f"-L{REDIST}", "-L/run/opengl-driver/lib", f"-L{STUBS}",
                    f"{WORK}/{name}.cu", "-o", f"{WORK}/{name}"],
                   check=True, capture_output=True, text=True, timeout=120, env=env)
    subprocess.run([f"{WORK}/{name}"], check=True, capture_output=True, timeout=60, env=env)
    return np.fromfile(f"{WORK}/out.bin", np.float32)


# ── MLIR-GPU backend: same fact -> gpu-dialect MLIR -> NVVM -> PTX/cubin -> P4 ──
# The second GPU backend in the cross-backend matrix. mlir_gpu_from_facts.pl emits
# the gpu.module kernel; mlir_gpu_pipeline.sh lowers it (elementwise->PTX,
# transcendental->libdevice-linked cubin); mlirgpu_run launches it over in.bin.
# MLIR-GPU now covers all the elementwise ops the expr IR defines (relu/tanh/
# sigmoid/silu/elu/gelu) — sigmoid+silu came free once the body is generated from
# the neutral expr term (expr_ir.pl) instead of a hand-written MLIR table.
# The MLIR-GPU emitter is OP-GENERIC: gpu_body derives the kernel from the op_expr
# AST via lower_mlir/3. So ANY elementwise op whose AST lower_mlir covers works
# without emitter changes — this set was just a conservative gate. Expanded to the
# elementwise family the emitter already lowers (verified: leaky_relu/hardsigmoid/
# mish/softplus/selu/gelu_tanh all generate valid MLIR from their facts).
MLIR_GPU_OPS = {"bpd_relu", "bpd_tanh", "bpd_sigmoid", "bpd_silu", "bpd_elu",
                "bpd_gelu", "bpd_gelu_tanh", "bpd_leaky_relu", "bpd_hardsigmoid",
                "bpd_softplus", "bpd_selu", "bpd_mish"}
MLIRGPU_RUN  = f"{WORK}/mlirgpu_run"
MB           = "/tmp/gpu-work/mlir_backend"  # pipeline working dir

def mlir_gpu_supported(fact):
    return fact in MLIR_GPU_OPS

def _mlir_gpu_build_module(fact):
    """Generate + lower the MLIR-GPU kernel for a fact. Returns (module_path, op_name)
    where module is a .ptx (elementwise) or .cubin (transcendental, libdevice-linked)."""
    name = fact[4:] if fact.startswith("bpd_") else fact
    env = dict(os.environ, LD_LIBRARY_PATH=f"/run/opengl-driver/lib:{CUDA}/lib",
               PATH=f"/run/current-system/sw/bin:{os.environ.get('PATH','')}")
    # 1. emit the gpu-dialect MLIR into the pipeline working dir
    swi = (f'consult("{EMITTERS}/mlir_gpu_from_facts.pl"), '
           f'emit_mlir_gpu({fact}, "{MB}/{name}_gpu_gen.mlir"), halt')
    subprocess.run([SWIPL, "-q", "-g", swi], capture_output=True, text=True, timeout=60, env=env)
    # 2. run the lowering pipeline (-> {name}_mlirgpu.ptx or .cubin)
    subprocess.run(["bash", f"{KERNELGEN}/runtime/mlir_gpu_pipeline.sh", name],
                   capture_output=True, text=True, timeout=120, env=env, cwd=MB)
    ptx, cubin = f"{MB}/{name}_mlirgpu.ptx", f"{MB}/{name}_mlirgpu.cubin"
    if os.path.exists(cubin):
        return cubin, name
    if os.path.exists(ptx):
        return ptx, name
    return None, name

def generate_and_run_mlir_gpu(fact, op_key, inputs):
    """Generate the MLIR-GPU kernel, run it on the P4 over the (capped) input,
    return GPU output. The SECOND GPU backend in the cross-backend matrix."""
    x = inputs[0].detach().cpu().numpy().astype(np.float32).ravel()
    n = x.size
    x.tofile(f"{WORK}/in.bin")
    mod, name = _mlir_gpu_build_module(fact)
    if not mod:
        raise RuntimeError("mlir-gpu lowering produced no module")
    env = tc.nvcc_env()
    subprocess.run([MLIRGPU_RUN, mod, name, str(n)],
                   check=True, capture_output=True, timeout=60, env=env, cwd=WORK)
    return np.fromfile(f"{WORK}/out.bin", np.float32)

def benchmark_mlir_gpu(fact, op_key, klass):
    """Time the MLIR-GPU kernel via perf_fixture (loads the same ptx/cubin)."""
    _ensure_perf_fixture()
    mod, name = _mlir_gpu_build_module(fact)
    if not mod:
        return None
    env = tc.nvcc_env()
    out = subprocess.run([PERF_FIXTURE, "elementwise", name, mod, str(PERF_ELEMS), "200"],
                         capture_output=True, text=True, timeout=120, env=env).stdout
    m_bw = re.search(r"GB/s:\s*([\d.]+)\s*\(([\d.]+)%", out)
    m_t  = re.search(r"median=([\d.]+)\s*ms", out)
    if not m_bw:
        return None
    return {"gbs": float(m_bw.group(1)), "pct_peak": float(m_bw.group(2)),
            "ms": float(m_t.group(1)) if m_t else None}


# ── LLVM backend (backend 4): fact -> op_expr -> lower_llvm -> .ll -> clang -> CPU
# A CPU backend; verifies against the torch-CPU reference directly (no P4).
CLANG = "/run/current-system/sw/bin/clang"
_LLVM_HARNESS = f"{WORK}/llvm_runner"   # built per-op (links the generated .o)

def generate_and_run_llvm(fact, op_key, inputs):
    """Emit the op as LLVM IR (llvm_from_facts.pl -> lower_llvm), clang-compile,
    run over in.bin -> out.bin. Backend 4, from the SAME AST as cuda_c/mlir."""
    x = inputs[0].detach().cpu().numpy().astype(np.float32).ravel()
    n = x.size
    x.tofile(f"{WORK}/in.bin")
    name = fact[4:] if fact.startswith("bpd_") else fact
    swi = (f'consult("{EMITTERS}/llvm_from_facts.pl"), '
           f'emit_llvm_from_fact({fact}, "{WORK}/{name}.ll"), halt')
    subprocess.run([SWIPL, "-q", "-g", swi], capture_output=True, text=True, timeout=60)
    if not os.path.exists(f"{WORK}/{name}.ll"):
        raise RuntimeError("llvm emit produced no .ll")
    # small C harness that reads in.bin, calls <name>(src,dst,n), writes out.bin
    harness = f"""#include <stdio.h>
#include <stdlib.h>
extern void {name}(float*,float*,long);
int main(){{ FILE*fi=fopen("in.bin","rb"); fseek(fi,0,2); long b=ftell(fi); fseek(fi,0,0);
 long n=b/4; float*in=malloc(b),*out=malloc(b); fread(in,4,n,fi); fclose(fi);
 {name}(in,out,n); FILE*fo=fopen("out.bin","wb"); fwrite(out,4,n,fo); fclose(fo); return 0; }}"""
    open(f"{WORK}/{name}_h.c", "w").write(harness)
    subprocess.run([CLANG, "-O2", "-c", f"{WORK}/{name}.ll", "-o", f"{WORK}/{name}.o"],
                   check=True, capture_output=True, text=True, timeout=60)
    subprocess.run([CLANG, "-O2", f"{WORK}/{name}_h.c", f"{WORK}/{name}.o", "-lm",
                    "-o", f"{WORK}/{name}_llvm"], check=True, capture_output=True, text=True, timeout=60)
    subprocess.run([f"{WORK}/{name}_llvm"], check=True, capture_output=True, timeout=60, cwd=WORK)
    return np.fromfile(f"{WORK}/out.bin", np.float32)


# ── PyTorch backend (backend 5): fact -> op_expr -> lower_torch -> Python op(x)
def _run_torch_module(name, op_arg):
    """Import the generated {name}_torch.py and call name(op_arg) -> flat numpy."""
    import importlib.util, torch
    spec = importlib.util.spec_from_file_location(name, f"{WORK}/{name}_torch.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    fn = getattr(mod, name)
    out = fn(op_arg)
    return out.detach().cpu().numpy().astype(np.float32).ravel()

def generate_and_run_torch(fact, op_key, inputs):
    """Emit the op as a torch Python fn (torch_from_facts.pl -> lower_torch),
    import it, run. inputs may be a single tensor (elementwise), or a tuple
    (matmul (A,B); conv (x,w); loss (pred,target)/(anchor,pos,neg))."""
    name = fact[4:] if fact.startswith("bpd_") else fact
    swi = (f'consult("{EMITTERS}/torch_from_facts.pl"), '
           f'emit_torch_from_fact({fact}, "{WORK}/{name}_torch.py"), halt')
    subprocess.run([SWIPL, "-q", "-g", swi], capture_output=True, text=True, timeout=60)
    if not os.path.exists(f"{WORK}/{name}_torch.py"):
        raise RuntimeError("torch emit produced no .py")
    # multi-operand ops expect the tuple; single-tensor ops expect the tensor.
    op_arg = inputs if isinstance(inputs, tuple) else inputs[0]
    return _run_torch_module(name, op_arg)

def generate_and_run_torch_param(name, term, op_arg):
    """Per-problem PARAMETERIZED: emit a torch fn from an EXPLICIT op_expr term
    string (params already substituted from the KB Model), run it on op_arg.
    term is a Prolog term string e.g. 'conv(2,0,4,2,1,1)' or
    'pool(max,2,3,2,1,1,var)' or 'stat_norm(1,const(1.0e-5),var)'."""
    swi = (f'consult("{EMITTERS}/torch_from_facts.pl"), '
           f'emit_torch_term({name}, {term}, "{WORK}/{name}_torch.py"), halt')
    subprocess.run([SWIPL, "-q", "-g", swi], capture_output=True, text=True, timeout=60)
    if not os.path.exists(f"{WORK}/{name}_torch.py"):
        raise RuntimeError(f"torch param emit produced no .py for {term}")
    return _run_torch_module(name, op_arg)


# Perf-fixture binary (built once, lazily). Lives next to this driver.
PERF_FIXTURE = f"{WORK}/perf_fixture"
_PERF_SRC = f"{KERNELGEN}/referee/perf_fixture.cu"

def _ensure_perf_fixture():
    """Compile perf_fixture once (it's backend-agnostic — loads any PTX/cubin)."""
    if os.path.exists(PERF_FIXTURE):
        return
    env = tc.nvcc_env()
    subprocess.run(tc.nvcc_link_cmd(_PERF_SRC, PERF_FIXTURE, extra_L=[REDIST, STUBS], extra=["-Wno-deprecated-gpu-targets"]),
                   check=True, capture_output=True, text=True, timeout=120, env=env)

# Perf measurement uses a steady-state size (not the verify cap): big enough that
# kernel time dominates launch overhead, small enough to fit the P4. 16M elems.
PERF_ELEMS = int(os.environ.get("SWEEP_PERF_ELEMS", str(1 << 24)))

def benchmark_elementwise(fact, op_key, klass):
    """Generate a kernel-only cubin (perf_fixture ABI), time it on the P4 at the
    steady-state size, return dict(gbs, pct_peak, ms) — the PERF gate.
    Times the SAME generated kernel that was just verified (same fact/formulation)."""
    _ensure_perf_fixture()
    name = fact[4:] if fact.startswith("bpd_") else fact
    # 1. emit kernel-only .cu (perf_fixture ABI: extern "C" <op>(src,dst,long n))
    swi = (f'consult("{EMITTERS}/cuda_c_from_facts.pl"), '
           f'emit_cuda_c_kernel_only({fact}, "{WORK}/{name}_k.cu"), halt')
    subprocess.run([SWIPL, "-q", "-g", swi], capture_output=True, text=True, timeout=60)
    if not os.path.exists(f"{WORK}/{name}_k.cu"):
        return None
    # 2. compile to cubin
    env = tc.nvcc_env()
    subprocess.run(tc.nvcc_compile_cmd(f"{WORK}/{name}_k.cu", f"{WORK}/{name}.cubin"),
                   check=True, capture_output=True, text=True, timeout=120, env=env)
    # 3. run perf_fixture; kernel is named k_<name> (prefixed to avoid libm
    #    symbol collisions: a bare extern "C" tanh clashes with CUDA's math tanh)
    out = subprocess.run([PERF_FIXTURE, "elementwise", f"k_{name}", f"{WORK}/{name}.cubin",
                          str(PERF_ELEMS), "200"],
                         capture_output=True, text=True, timeout=120, env=env).stdout
    m_bw = re.search(r"GB/s:\s*([\d.]+)\s*\(([\d.]+)%", out)
    m_t  = re.search(r"median=([\d.]+)\s*ms", out)
    if not m_bw:
        return None
    return {"gbs": float(m_bw.group(1)), "pct_peak": float(m_bw.group(2)),
            "ms": float(m_t.group(1)) if m_t else None}


# ── GPU-shape perf: time the generated cuda kernel for the new shape classes
# (axis-reduce, pool, conv2d) on the P4 via perf_kernel, compute % of roofline,
# and FLAG anomalies (correct-but-slow kernels). This is the in-sweep perf gate
# your instinct asked for: a kernel can be bit-correct yet 30x off its roofline,
# and the sweep should surface that automatically per backend.
PEAK_BW = 192.2   # P4 GB/s
PEAK_FP = 5700.0  # P4 GFLOPS (fp32)
PERF_KERNEL = f"{WORK}/perf_kernel"
# below this fraction of the kernel's OWN roofline -> flag as a perf anomaly.
ANOMALY_FRAC = 0.30

def _emit_compile_shape_cubin(fact, emit_pred, kfn):
    """Emit + compile a GPU-shape cuda kernel; return cubin path or None."""
    name = fact[4:] if fact.startswith("bpd_") else fact
    swi = (f'consult("{EMITTERS}/expr_ir.pl"), '
           f'{emit_pred}({fact}, "{WORK}/{name}_p.cu"), halt')
    subprocess.run([SWIPL, "-q", "-g", swi], capture_output=True, text=True, timeout=60)
    if not os.path.exists(f"{WORK}/{name}_p.cu"):
        return None
    env = tc.nvcc_env()
    r = subprocess.run([f"{CUDA}/bin/nvcc", "-arch=sm_61", "-cubin", "-O3",
                        f"-I{CUDA}/include", f"{WORK}/{name}_p.cu", "-o", f"{WORK}/{name}_p.cubin"],
                       capture_output=True, text=True, timeout=120, env=env)
    return f"{WORK}/{name}_p.cubin" if r.returncode == 0 else None

def _compile_static_cubin(src_name):
    """Compile a STATIC runtime .cu (e.g. conv_implicit.cu) to a cubin. Cached."""
    src = os.path.join(KERNELGEN, "runtime", src_name)
    if not os.path.exists(src):
        return None
    out = f"{WORK}/{src_name.replace('.cu','')}.cubin"
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(src):
        return out                                  # cached
    env = tc.nvcc_env()
    r = subprocess.run([f"{CUDA}/bin/nvcc", "-arch=sm_61", "-cubin", "-O3",
                        f"-I{CUDA}/include", src, "-o", out],
                       capture_output=True, text=True, timeout=120, env=env)
    return out if r.returncode == 0 else None


def benchmark_torch_gpu(klass, term, op_key=None):
    """Time the EQUIVALENT torch op on the P4 at the SAME canonical shapes as
    benchmark_gpu_shape / benchmark_elementwise, so OUR generated kernel's ms is
    directly comparable to torch's. Returns dict(torch_ms, torch_gbs) or None.

    Requires torch-on-P4 (the sm_61 build). A second GPU reference alongside the
    roofline: the ratio our_ms / torch_ms says how our codegen compares to torch's
    own kernels at identical work.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None
    except Exception:
        return None
    dev = torch.device("cuda")
    F = torch.nn.functional

    def _time(make, run, reps=50):
        x = make()
        for _ in range(5): run(x)
        torch.cuda.synchronize()
        import time as _t
        t0 = _t.perf_counter()
        for _ in range(reps): run(x)
        torch.cuda.synchronize()
        return (_t.perf_counter() - t0) / reps * 1e3

    try:
        if klass == "elementwise":
            # same steady-state size the elementwise perf fixture uses (~64M f32)
            n = 1 << 26
            ms = _time(lambda: torch.randn(n, device=dev), lambda x: torch.relu(x))
            gb = (2 * n * 4) / (ms * 1e-3) / 1e9
            return {"torch_ms": ms, "torch_gbs": gb}
        if klass == "axis_reduce":
            R, C = 4096, 4096
            ms = _time(lambda: torch.randn(R, C, device=dev),
                       lambda x: torch.sum(x, dim=1))
            gb = (R*C*4 + R*4) / (ms * 1e-3) / 1e9
            return {"torch_ms": ms, "torch_gbs": gb}
        if klass == "pool":
            mm = re.match(r"pool\((\w+),(\d+),(\d+),(\d+),(\d+),(\d+)", term or "")
            if not mm or mm.group(2) != "2":
                return None
            NC, H, W = 2048, 56, 56
            K, S, P = int(mm.group(3)), int(mm.group(4)), int(mm.group(5))
            kind = mm.group(1)
            pool = (lambda x: F.max_pool2d(x, K, S, P)) if "max" in kind \
                   else (lambda x: F.avg_pool2d(x, K, S, P))
            ms = _time(lambda: torch.randn(1, NC, H, W, device=dev), pool)
            return {"torch_ms": ms}
        if klass == "conv":
            mm = re.match(r"conv\((\d+),(\d+)", term or "")
            if not mm or mm.group(1) != "2" or mm.group(2) != "0":
                return None
            N, Cin, H, W, Cout, KH, KW = 32, 64, 56, 56, 128, 3, 3
            w = torch.randn(Cout, Cin, KH, KW, device=dev)
            ms = _time(lambda: torch.randn(N, Cin, H, W, device=dev),
                       lambda x: F.conv2d(x, w))
            flops = 2*N*Cout*(H-KH+1)*(W-KW+1)*Cin*KH*KW
            gf = flops / (ms * 1e-3) / 1e9
            return {"torch_ms": ms, "torch_gflops": gf}
    except Exception:
        return None
    return None


def benchmark_gpu_shape(fact, klass, term):
    """Build the cuda kernel for a new-shape op, time it on the P4, return
    dict(ms, throughput, unit, pct_peak, roofline, anomaly). klass in
    {axis_reduce, pool, conv}. Returns None if not GPU-benchmarkable."""
    if not os.path.exists(PERF_KERNEL):
        return None
    env = tc.nvcc_env()
    def _time(cubin, kfn, shape_args):
        out = subprocess.run([PERF_KERNEL, cubin, kfn] + [str(a) for a in shape_args],
                             capture_output=True, text=True, timeout=120, env=env).stdout
        m = re.search(r"median_ms=([\d.]+)", out)
        return float(m.group(1)) if m else None

    if klass == "axis_reduce":
        cubin = _emit_compile_shape_cubin(fact, "emit_cuda_axis_reduce", "k_reduce")
        if not cubin: return None
        R, C = 4096, 4096
        ms = _time(cubin, "k_reduce", ["reduce", R, C])
        if ms is None: return None
        gb = (R*C*4 + R*4) / (ms*1e-3) / 1e9          # memory-bound
        pct = gb / PEAK_BW * 100
        return {"ms": ms, "throughput": gb, "unit": "GB/s", "pct_peak": pct,
                "roofline": "BW", "anomaly": pct < ANOMALY_FRAC*100}
    if klass == "pool":
        # parse pool(kind,nd,k,s,p,d,var) -> only 2D benched here
        mm = re.match(r"pool\((\w+),(\d+),(\d+),(\d+),(\d+),(\d+)", term or "")
        if not mm or mm.group(2) != "2": return None
        cubin = _emit_compile_shape_cubin(fact, "emit_cuda_pool", "k_pool")
        if not cubin: return None
        NC, H, W = 2048, 56, 56
        K, S, P = int(mm.group(3)), int(mm.group(4)), int(mm.group(5))
        Ho = (H + 2*P - K)//S + 1; Wo = (W + 2*P - K)//S + 1
        # match the emitter's dispatch: K*K<49 -> simple (thread/output), else warp-tiled
        pshape = "pool2d" if K*K < 49 else "pool2d_warp"
        ms = _time(cubin, "k_pool", [pshape, NC, H, W, Ho, Wo])
        if ms is None: return None
        gb = (NC*H*W*4 + NC*Ho*Wo*4) / (ms*1e-3) / 1e9
        pct = gb / PEAK_BW * 100
        return {"ms": ms, "throughput": gb, "unit": "GB/s", "pct_peak": pct,
                "roofline": "BW", "anomaly": pct < ANOMALY_FRAC*100}
    if klass == "conv":
        # only standard (transposed=0) 2D benched here
        mm = re.match(r"conv\((\d+),(\d+)", term or "")
        if not mm or mm.group(1) != "2" or mm.group(2) != "0": return None
        N,Cin,H,W,Cout,KH,KW = 32,64,56,56,128,3,3
        Ho,Wo = H-KH+1, W-KW+1; ws = Cout*Cin*KH*KW
        flops = 2*N*Cout*Ho*Wo*Cin*KH*KW                # compute-bound
        # PREFER the im2col implicit-GEMM kernel (6.25x over naive, 13.6% peak).
        # Fall back to the naive emit_cuda_conv if the static kernel isn't available.
        cubin = _compile_static_cubin("conv_implicit.cu")
        kernel = "implicit"
        if cubin:
            # PASS the compiled tile geometry (BM,BN,NTH) so the launch matches the
            # kernel's #defines (BM128 BN128 TM8 TN4 -> NTH=(128/8)*(128/4)=512) AND
            # picks up the __launch_bounds__(512,2) occupancy build. Without these
            # the launch used stale perf_kernel defaults -> the occupancy win didn't
            # show (the sweep-instrumentation bug Heath flagged).
            CBM, CBN, CTM, CTN = 128, 128, 8, 4
            CNTH = (CBM // CTM) * (CBN // CTN)
            ms = _time(cubin, "k_conv_implicit",
                       ["conv2d_implicit", N,Cin,H,W,Cout,KH,KW,Ho,Wo,ws, CBM,CBN,CNTH])
        else:
            cubin = _emit_compile_shape_cubin(fact, "emit_cuda_conv", "k_conv")
            kernel = "naive"
            ms = _time(cubin, "k_conv", ["conv2d", N,Cin,H,W,Cout,KH,KW,Ho,Wo,ws]) if cubin else None
        if ms is None: return None
        gf = flops / (ms*1e-3) / 1e9
        pct = gf / PEAK_FP * 100
        return {"ms": ms, "throughput": gf, "unit": "GFLOPS", "pct_peak": pct,
                "roofline": "FP", "anomaly": pct < ANOMALY_FRAC*100, "kernel": kernel}
    return None


# Matmul dims: KB problem 1 is N=4096 (square). Verification needs the full
# N*N*N compute on CPU (torch) which is slow for big N, so we cap the VERIFY
# size to a square slice; perf uses a larger (memory-fitting) size.
MM_VERIFY_N = int(os.environ.get("SWEEP_MM_VERIFY_N", "512"))   # 512^3 ~ 0.27 GFLOP, fast CPU ref
MM_PERF_N   = int(os.environ.get("SWEEP_MM_PERF_N",   "1024"))  # 1024^3 ~ 2.1 GFLOP

def verify_and_benchmark_matmul(info):
    """Reduction path: generate naive GEMM (fma=strict, to match torch), verify
    vs torch.matmul over a capped square, then benchmark GFLOPS vs the roofline.
    Returns a row dict with correctness + gflops/pct_peak/ms."""
    import torch, numpy as np
    row = {}
    # 1. VERIFY: square A,B of size MM_VERIFY_N (deterministic), torch.matmul ref.
    n = MM_VERIFY_N
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n)).astype(np.float32)
    B = rng.standard_normal((n, n)).astype(np.float32)
    A.tofile(f"{WORK}/A.bin"); B.tofile(f"{WORK}/B.bin")
    ref = torch.matmul(torch.from_numpy(A), torch.from_numpy(B)).numpy()
    # generate the verify runner (fma=strict matches torch-CPU: mul+add, 2 roundings)
    swi = (f'consult("{EMITTERS}/gemm_from_facts.pl"), '
           f'emit_gemm_verify(strict, "{WORK}/A.bin", "{WORK}/B.bin", "{WORK}/C.bin", "{WORK}/gemm_v.cu"), halt')
    subprocess.run([SWIPL, "-q", "-g", swi], capture_output=True, text=True, timeout=60)
    env = tc.nvcc_env()
    subprocess.run([f"{CUDA}/bin/nvcc", "-O3", "-arch=sm_61", "-Wno-deprecated-gpu-targets",
                    "-cudart", "shared", f"-I{CUDA}/include", f"-L{CUDA}/lib", f"-L{REDIST}",
                    "-L/run/opengl-driver/lib", f"-L{STUBS}", f"{WORK}/gemm_v.cu", "-o", f"{WORK}/gemm_v"],
                   check=True, capture_output=True, text=True, timeout=120, env=env)
    subprocess.run([f"{WORK}/gemm_v", str(n)], check=True, capture_output=True, timeout=120, env=env)
    got = np.fromfile(f"{WORK}/C.bin", np.float32).ravel()
    refr = ref.ravel()
    mx, nd, nan_mm, sz = ulp_compare(got, refr)
    # CORRECTNESS BAR FOR REDUCTIONS: bit-identity vs torch is the WRONG bar —
    # torch.matmul uses a BLOCKED/tiled summation order (MKL), our naive kernel
    # sums k-sequentially. Float addition is non-associative, so different
    # reduction ORDERS give different last bits even for the identical math.
    # (Confirmed: torch itself differs from an fp64 reference by ~1e6 ULP.)
    # So we use KernelBench's own criterion: relative error tolerance (rtol/atol),
    # AND separately report the ULP (which reflects the reduction-order gap, not
    # an error). Bit-identity would only hold vs a SAME-ORDER reference.
    denom = np.abs(refr) + 1e-30
    rel = np.abs(got - refr) / denom
    max_rel = float(np.nanmax(rel))
    mean_rel = float(np.nanmean(rel))
    # KernelBench uses ~1e-2 rtol for fp32 matmul; we use a tight 1e-3 mean / 1e-1 max.
    if mean_rel < 1e-4 and max_rel < 1e-1:
        verdict = f"rel-ok(mean={mean_rel:.1e},ulp={mx}/reduction-order)"
    else:
        verdict = f"DIFF rel(mean={mean_rel:.1e},max={max_rel:.1e})"
    row.update(correctness=verdict, max_ulp=mx, n_differ=nd,
               max_rel=max_rel, mean_rel=mean_rel)

    # 2. BENCHMARK: kernel-only GEMM cubin, perf_fixture gemm class -> GFLOPS.
    if not verdict.startswith("DIFF"):
        _ensure_perf_fixture()
        swi2 = (f'consult("{EMITTERS}/gemm_from_facts.pl"), '
                f'emit_gemm_kernel_only(strict, "{WORK}/gemm_k.cu"), halt')
        subprocess.run([SWIPL, "-q", "-g", swi2], capture_output=True, text=True, timeout=60)
        subprocess.run([f"{CUDA}/bin/nvcc", "-arch=sm_61", "-cubin", "-O3",
                        f"-I{CUDA}/include", f"{WORK}/gemm_k.cu", "-o", f"{WORK}/gemm.cubin"],
                       check=True, capture_output=True, text=True, timeout=120, env=env)
        out = subprocess.run([PERF_FIXTURE, "gemm", "k_gemm", f"{WORK}/gemm.cubin",
                              str(MM_PERF_N), "50"], capture_output=True, text=True, timeout=180, env=env).stdout
        m = re.search(r"GFLOPS:\s*([\d.]+)\s*\(([\d.]+)%", out)
        mt = re.search(r"median=([\d.]+)\s*ms", out)
        if m:
            row.update(gflops=float(m.group(1)), pct_peak=float(m.group(2)),
                       ms=float(mt.group(1)) if mt else None)
    return row


def sweep(level, problems, backends, limit, outprefix):
    level_dir = f"{KB_ROOT}/level{level}"
    files = sorted(f for f in os.listdir(level_dir) if f.endswith(".py"))
    if problems:  # filter to specific problem numbers
        want = set(problems)
        files = [f for f in files if f.split("_")[0] in want]
    if limit:
        files = files[:limit]

    rows = []
    print(f"=== KernelBench sweep: level{level}, {len(files)} problems, backends={backends} ===\n")
    print(f"{'problem':40} {'op':8} {'status':12} {'correctness':24} {'perf / note'}")
    print("-" * 110)

    for f in files:
        path = f"{level_dir}/{f}"
        info = lift_problem(path)
        row = {"problem": f, "level": level, **{k: info.get(k) for k in ("op", "status", "fact", "klass")}}

        if info["status"] != "lifted":
            row["correctness"] = "-"
            print(f"{f[:39]:40} {str(info.get('op','-')):8} {info['status']:12} {'-':24} {info.get('reason','')[:30]}")
            rows.append(row); continue

        # ── REDUCTION class (matmul): different ABI (A,B,C,n), reference
        #    torch.matmul, metric GFLOPS. Bit-identity vs torch holds for
        #    fma=strict (mul+add) at matching accumulation order (our finding).
        if info["klass"] == "reduction":
            try:
                r = verify_and_benchmark_matmul(info)
                row.update(r)
                pstr = (f"{r.get('gflops',0):.0f} GFLOPS ({r.get('pct_peak',0):.1f}% peak)"
                        if r.get("gflops") else "")
                print(f"{f[:39]:40} {info['op']:8} {'verified':12} {r['correctness']:24} {pstr}")
            except Exception as e:
                row.update(status="gen_run_error", correctness="-", reason=str(e)[:120])
                print(f"{f[:39]:40} {info['op']:8} {'gen_error':12} {'-':24} {str(e)[:40]}")
                traceback.print_exc(file=sys.stderr)
            rows.append(row); continue

        # New op classes (axis_reduce, norm, pool, conv, loss): verified via the
        # TORCH backend — it generates op(inputs) from the op_expr AST and compares
        # to the KB Model reference. GPU kernels for these exist (axis-reduce/pool/
        # conv2d on the P4) but aren't in the per-backend verify path yet; torch is
        # the universal correctness arm.
        if info["klass"] in ("axis_reduce", "norm", "pool", "conv", "loss"):
            brow = dict(row, backend="torch")
            try:
                import torch
                model, kb_inputs = info["model"], info["inputs"]
                name = info["fact"][4:] if info["fact"].startswith("bpd_") else info["fact"]
                ref = reference_output(model, kb_inputs).ravel()
                # PER-PROBLEM PARAM EXTRACTION: read the actual params off the KB
                # Model's nn submodule + build a parameterized op_expr term, so the
                # generated fn matches THIS problem (not representative defaults).
                term, op_arg = build_param_term(info, model, kb_inputs)
                if term:
                    # use a problem-unique fn name so files/modules don't collide
                    # across problems sharing a fact (e.g. all pools -> bpd_maxpool2d)
                    uname = re.sub(r"[^a-zA-Z0-9]", "_", f.replace(".py","")).lower()
                    uname = "op_" + uname
                    got = generate_and_run_torch_param(uname, term, op_arg)
                else:
                    # fallback to the fixed-default op_expr; pass the raw kb_inputs
                    # (single-tensor ops use [0]; loss/triplet use the tuple).
                    fb = tuple(kb_inputs) if info["klass"] == "loss" else kb_inputs
                    got = generate_and_run_torch(info["fact"], info["op"], fb)
                got = np.asarray(got).ravel()
                n = min(got.size, ref.size)
                mx, nd, nan_mm, sz = ulp_compare(got[:n], ref[:n])
                rel = float(np.mean(np.abs(got[:n]-ref[:n])/(np.abs(ref[:n])+1e-30)))
                if mx == 0:
                    verdict = "BIT-IDENTICAL"
                elif rel < 1e-4:
                    verdict = f"rel-ok({rel:.1e})"
                else:
                    verdict = f"DIFF {mx}ULP rel={rel:.1e}"
                brow.update(correctness=verdict, max_ulp=mx, mean_rel=rel)
                # ── IN-SWEEP GPU PERF: time the generated CUDA kernel for the new
                #    shape classes on the P4, record throughput + flag anomalies
                #    (correct-but-slow). Only when the op is correct (no point
                #    timing a wrong kernel). The torch row carries correctness;
                #    a separate 'cuda_c' row carries the GPU perf + anomaly flag.
                perf_note = "(cpu)"
                if not verdict.startswith("DIFF") and info["klass"] in ("axis_reduce", "pool", "conv"):
                    try:
                        perf = benchmark_gpu_shape(info["fact"], info["klass"], term)
                        # torch-GPU reference at the SAME shape (needs torch-on-P4).
                        tgpu = benchmark_torch_gpu(info["klass"], term, info["op"])
                        if perf:
                            # our-vs-torch ratio: <1 means we beat torch, >1 we're slower
                            tm = tgpu.get("torch_ms") if tgpu else None
                            ratio = (perf["ms"] / tm) if (tm and tm > 0) else None
                            grow = dict(row, backend="cuda_c", correctness="(perf)",
                                        gbs=(perf["throughput"] if perf["unit"]=="GB/s" else None),
                                        gflops=(perf["throughput"] if perf["unit"]=="GFLOPS" else None),
                                        pct_peak=perf["pct_peak"], ms=perf["ms"],
                                        torch_ms=tm, vs_torch=ratio,
                                        status=("PERF-ANOMALY" if perf["anomaly"] else "perf-ok"))
                            rows.append(grow)
                            flag = "  ⚠ANOMALY" if perf["anomaly"] else ""
                            vs = f"  vs-torch {ratio:.2f}x" if ratio else (f"  torch {tm:.3f}ms" if tm else "")
                            perf_note = (f"GPU {perf['throughput']:.1f} {perf['unit']} "
                                         f"({perf['pct_peak']:.0f}% {perf['roofline']}-peak){flag}{vs}")
                    except Exception as pe:
                        perf_note = f"(cpu; gpu-perf-err {str(pe)[:20]})"
                print(f"{f[:36]:37} {info['op']:8} {'torch':8} {'verified':10} {verdict:20} {perf_note}")
            except Exception as e:
                brow.update(status="gen_run_error", correctness="-", reason=str(e)[:120])
                print(f"{f[:36]:37} {info['op']:8} {'torch':8} {'gen_error':10} {str(e)[:34]}")
            rows.append(brow); continue

        # anything else still unwired
        if info["klass"] != "elementwise":
            row["status"] = "deferred"; row["correctness"] = "-"
            print(f"{f[:39]:40} {info['op']:8} {'deferred':12} {'-':24} class={info['klass']} — not wired")
            rows.append(row); continue

        # Cap the input ONCE, up front, so BOTH the torch reference and the GPU
        # kernels operate on the same manageable slice (KB defaults can be 6.4GB).
        import torch
        capped = cap_inputs(info["inputs"])
        ref = reference_output(info["model"], capped).ravel()

        # CROSS-BACKEND MATRIX: run each requested backend over the SAME input +
        # reference, emit one row per (problem, backend). cuda_c is always
        # available; mlir_gpu where mlir_gpu_supported(fact).
        for backend in backends:
            brow = dict(row, backend=backend)
            if backend == "mlir_gpu" and not mlir_gpu_supported(info["fact"]):
                brow.update(status="unsupported_backend", correctness="-")
                print(f"{f[:36]:37} {info['op']:7} {backend:8} {'n/a':12} {'(no mlir-gpu fact)':24}")
                rows.append(brow); continue
            try:
                if backend == "mlir_gpu":
                    got = generate_and_run_mlir_gpu(info["fact"], info["op"], capped)
                elif backend == "llvm":
                    got = generate_and_run_llvm(info["fact"], info["op"], capped)
                elif backend == "torch":
                    got = generate_and_run_torch(info["fact"], info["op"], capped)
                else:
                    got = generate_and_run_elementwise(info["fact"], info["op"], capped)
                refb = ref[:got.size]
                mx, nd, nan_mm, sz = ulp_compare(got, refb)
                if mx == 0 and nan_mm == 0 and sz == 0:
                    verdict = "BIT-IDENTICAL"
                elif info["transcendental"] and mx <= 4:
                    verdict = f"transcendental-ulp({mx})"
                elif sz > 0 and mx == 0 and nd == 0:
                    verdict = f"signed-zero({sz})"
                else:
                    verdict = f"DIFF {mx}ULP/{nd}d" + (f"+{nan_mm}nan" if nan_mm else "") + (f"+{sz}sz" if sz else "")
                brow.update(correctness=verdict, max_ulp=mx, n_differ=nd,
                            nan_mismatch=nan_mm, signed_zero=sz)
                perf_str = ""
                # perf gate is GPU roofline-relative (GB/s vs 192 GB/s); only the
                # GPU backends report it. llvm/torch are CPU -> correctness only.
                if not verdict.startswith("DIFF") and backend in ("cuda_c", "mlir_gpu"):
                    try:
                        perf = (benchmark_mlir_gpu if backend == "mlir_gpu" else benchmark_elementwise)(
                            info["fact"], info["op"], info["klass"])
                        if perf:
                            tgpu = benchmark_torch_gpu(info["klass"], None, info["op"])
                            tm = tgpu.get("torch_ms") if tgpu else None
                            ratio = (perf["ms"] / tm) if (tm and tm > 0) else None
                            brow.update(gbs=perf["gbs"], pct_peak=perf["pct_peak"], ms=perf["ms"],
                                        torch_ms=tm, vs_torch=ratio)
                            vs = f"  vs-torch {ratio:.2f}x" if ratio else ""
                            perf_str = f"{perf['gbs']:.0f} GB/s ({perf['pct_peak']:.0f}% peak){vs}"
                    except Exception as pe:
                        perf_str = f"perf-err: {str(pe)[:24]}"
                elif backend in ("llvm", "torch"):
                    perf_str = "(cpu, correctness-only)"
                print(f"{f[:36]:37} {info['op']:7} {backend:8} {'verified':12} {verdict:22} {perf_str}")
            except Exception as e:
                brow.update(status="gen_run_error", correctness="-", reason=str(e)[:120])
                print(f"{f[:36]:37} {info['op']:7} {backend:8} {'gen_error':12} {str(e)[:30]}")
            rows.append(brow)

    # write the sweep table (JSON + TSV)
    with open(f"{outprefix}.json", "w") as fh:
        json.dump(rows, fh, indent=2, default=str)
    with open(f"{outprefix}.tsv", "w") as fh:
        cols = ["problem", "level", "op", "klass", "backend", "status", "correctness",
                "max_ulp", "n_differ", "nan_mismatch", "signed_zero",
                "mean_rel", "max_rel", "gbs", "gflops", "pct_peak", "ms",
                "torch_ms", "vs_torch"]
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

    # summary
    lifted = [r for r in rows if r["status"] in ("lifted", "verified")]
    verified = [r for r in rows if r.get("correctness", "").startswith(
        ("BIT-IDENTICAL", "transcendental", "signed-zero", "rel-ok"))]
    print(f"\n=== summary: {len(files)} problems | "
          f"{sum(1 for r in rows if r.get('op'))} recognized | "
          f"{len(verified)} verified-correct | results -> {outprefix}.{{json,tsv}} ===")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="KernelBench lift + verify + benchmark sweep")
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--problems", type=lambda s: s.split(","), default=None,
                    help="comma-separated problem numbers, e.g. 19,22,31")
    ap.add_argument("--backends", type=lambda s: s.split(","), default=["cuda_c", "mlir_gpu", "llvm", "torch"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=f"{WORK}/sweep_results")
    args = ap.parse_args()
    sweep(args.level, args.problems, args.backends, args.limit, args.out)
