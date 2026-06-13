#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""
l2_chain_lift.py — lift a KernelBench Level-2 problem (a CHAIN of ops) into an
ordered sequence of op_expr facts.

L1 problems are a single op; L2 problems are 3-6 op CHAINS, e.g.
  x = self.conv_transpose(x)   # conv
  x = x.mean(dim=2, keepdim=True)  # axis-reduce
  x = x + self.bias            # bias-add
  x = torch.softmax(x, dim=1)  # softmax
  x = torch.tanh(x)            # tanh
  x = x * self.scaling_factor  # scale

The chain-lifter parses forward() line by line, classifies each step (reusing the
L1 LIFT_PATTERNS + per-step recognizers), and returns a ChainIR: an ordered list
of steps with their op, fact, class, and any scalar params. This ChainIR is the
input the fusion engine (iterative_fusion + fusion_cost gate) consumes to decide
which adjacent steps to fuse, and the differential referee verifies the whole
chain against the KB Model's reference output.

Author: Iyun, 2026-06-08 — the L2 chain pass.

RELATED (pre-existing, complementary — different backend lens):
  bpd/lib/fusion_analyzer.pl + bpd/tests/kernelbench_l2_problems.pl encode the
  same 100 L2 problems HAND-written as GGML op-lists (ggml_mul_mat/ggml_scale/...)
  for the llama.cpp/ggml backend, with a train/test-split rule-coverage analysis.
  This module instead AUTO-LIFTS from the KB forward() source into op_expr facts
  (bpd_*) for the 5-backend codegen + torch verification. The two are consistent
  (same ops, different vocab); this path keeps per-op params (e.g. leaky_relu's
  actual slope) that the ggml mapping collapses. Worth converging the "100 L2
  problems" source eventually — a question for Heath.
"""
import os, re, sys

# reuse the L1 op-recognition tables.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernelbench_sweep import LIFT_PATTERNS, OP_LIFT


# ── per-step classifiers for the forward-body line forms L2 uses ──────────────
# Each returns (op_key, params_dict) or None. Ordered specific->general.
def _classify_step(line):
    s = line.strip()
    # strip trailing comments
    s = re.sub(r"#.*$", "", s).strip()
    if not s or s.startswith(("return", "\"\"\"", "'''", "Args", "x (", "torch.Tensor")):
        return None

    # the RHS of "x = <expr>" (or "var = <expr>")
    m = re.match(r"[A-Za-z_][\w]*\s*=\s*(.+)$", s)
    rhs = m.group(1).strip() if m else s

    # 1. residual save: original_x = x.clone()  -> 'save' marker capturing the
    #    target variable NAME so downstream tensor ops can reference it (DAG).
    if re.search(r"\.clone\(\)", rhs):
        lm = re.match(r"([A-Za-z_]\w*)\s*=", s)   # the LHS variable being saved
        return ("_save", {"as": lm.group(1)} if lm else {})

    # 2. binary op: x OP <operand>.  The OPERAND determines scalar vs tensor:
    #    - literal number or a known scalar attr -> SCALAR op (scaling/scalar_add/..)
    #    - self.bias / self.weight / a saved var (original_x) -> TENSOR op
    #      (broadcast-add / residual add/mul/sub/div), the DAG case.
    # 1b. self-gating swish: sigmoid(x) * x  OR  x * sigmoid(x)  (Swish = x*σ(x))
    if re.match(r"(?:torch\.|F\.)?sigmoid\(x\)\s*\*\s*x\s*$", rhs) or \
       re.match(r"x\s*\*\s*(?:torch\.|F\.)?sigmoid\(x\)\s*$", rhs):
        return ("silu", {})

    mb = re.match(r"x\s*([*/+\-])\s*([A-Za-z_][\w.]*|[\d.]+)\s*$", rhs)
    if mb:
        opc, operand = mb.group(1), mb.group(2)
        if re.match(r"[\d.]+$", operand):
            # literal -> definitely scalar
            smap = {"*": "scaling", "+": "scalar_add", "-": "scalar_sub", "/": "scalar_div"}
            return (smap[opc], {"operand": operand})
        # self.bias + -> bias_add (the common broadcast bias)
        if operand == "self.bias" and opc == "+":
            return ("bias_add", {})
        # a bound variable (original_x...) is always a tensor (residual)
        if re.match(r"[A-Za-z_]\w*$", operand) and operand != "x":
            tmap = {"+": "tensor_add", "*": "tensor_mul", "-": "tensor_sub", "/": "tensor_div"}
            return (tmap[opc], {"with": operand})
        # self.<attr> — AMBIGUOUS (scalar like multiplier, or tensor like sum_tensor).
        # Emit a 'binop' resolved at run time by the operand's type.
        return ("binop", {"op": opc, "operand": operand})

    # 5. axis-reduce — BOTH forms: x.mean(dim=..) method AND torch.mean(x, dim=..)
    #    function. Captures full dim spec (int / tuple / list) + keepdim flag.
    ma = re.match(r"(?:x\.|torch\.)(mean|sum|amax|amin|max|min|logsumexp)\(", rhs)
    if ma:
        kindmap = {"mean": "mean", "sum": "sum", "amax": "amax",
                   "amin": "amin", "max": "amax", "min": "amin",
                   "logsumexp": "logsumexp"}
        # dim= can be an int, (a,b), or [a,b,...]
        md = re.search(r"dim\s*=\s*(\(([^)]*)\)|\[([^\]]*)\]|\d+)", rhs)
        params = {}
        if md:
            spec = md.group(1)
            if spec.startswith(("(", "[")):
                dims = [int(d) for d in re.findall(r"\d+", spec)]
                params["dim"] = dims
            else:
                params["dim"] = int(spec)
        params["keepdim"] = ("keepdim=True" in rhs.replace(" ", ""))
        return (kindmap[ma.group(1)], params)

    # 5b. clamp / clip: torch.clamp(x, min=A, max=B), x.clamp(min=,max=), OR
    #     positional torch.clamp(x, lo, hi) with literals or self.attr bounds.
    mc = re.match(r"(?:torch\.|x\.)cl(?:amp|ip)\((.*)\)\s*$", rhs)
    if mc:
        inner = mc.group(1)
        # keyword form
        lo = re.search(r"min\s*=\s*([-\w.]+)", inner)
        hi = re.search(r"max\s*=\s*([-\w.]+)", inner)
        if lo or hi:
            return ("clamp", {"min": lo.group(1) if lo else None,
                              "max": hi.group(1) if hi else None})
        # positional form: clamp(x, lo, hi) -> split off the leading 'x'
        parts = [a.strip() for a in inner.split(",")]
        if len(parts) >= 3 and parts[0] == "x":
            return ("clamp", {"min": parts[1], "max": parts[2]})
        if len(parts) == 2 and parts[0] == "x":
            return ("clamp", {"min": parts[1], "max": None})
        return ("clamp", {"min": None, "max": None})

    # 5c. self-referential subtract-mean: x = x - x.mean(dim=, keepdim=)
    #     (centering — a single fused op, NOT a plain reduce that drops the subtract)
    ms = re.match(r"x\s*-\s*x\.mean\(", rhs)
    if ms:
        md = re.search(r"dim\s*=\s*(\d+)", rhs)
        return ("submean", {"dim": int(md.group(1)) if md else 1,
                            "keepdim": "keepdim=True" in rhs.replace(" ", "")})

    # 5d. FUNCTIONAL activation calls: F.mish(x), torch.nn.functional.gelu(x),
    #     torch.sigmoid(x), torch.relu(x) etc. (these were being DROPPED — e.g.
    #     conv->F.mish->F.mish lifted to just [conv]). Map the fn name to our op.
    mf = re.match(r"(?:torch\.nn\.functional\.|F\.|torch\.|nn\.functional\.)"
                  r"(relu|leaky_relu|gelu|silu|swish|mish|sigmoid|tanh|hardswish|"
                  r"hardsigmoid|hardtanh|softplus|selu|elu|softsign|relu6|elu_)\(x", rhs)
    if mf:
        fn = mf.group(1).rstrip("_")
        amap = {"swish": "silu", "relu6": "hardtanh"}
        params = {}
        if fn == "leaky_relu":
            sl = re.search(r"negative_slope\s*=\s*([\d.]+)", rhs) or \
                 re.search(r"leaky_relu\(x\s*,\s*([\d.]+)", rhs)
            if sl: params["slope"] = float(sl.group(1))
        return (amap.get(fn, fn), params)

    # 6. module call: self.<name>(x)  -> classify by the layer kind (resolved later
    #    from the model's submodule); record the attribute name for binding.
    mm = re.match(r"self\.([\w]+)\(x\)", rhs)
    if mm:
        return ("_module", {"attr": mm.group(1)})

    # 7. fall back to the L1 source-pattern recognizers on this single line.
    for rx, key in LIFT_PATTERNS:
        if rx.search(rhs):
            # extract a dim= if present (softmax/reduce)
            md = re.search(r"dim\s*=\s*(\d+)", rhs)
            params = {"dim": int(md.group(1))} if md else {}
            return (key, params)
    return ("_unknown", {"src": rhs[:40]})


def lift_chain(path):
    """Parse a KB L2 problem's forward() into an ordered ChainIR.
    Returns dict(problem, steps=[...], n_steps, recognized, unknown)."""
    src = open(path).read()
    name = os.path.basename(path)
    fwd = re.search(r"def forward\(self(.*?)\):(.*?)(?=\n    def |\nclass |\ndef get_)",
                    src, re.DOTALL)
    if not fwd:
        return {"problem": name, "status": "no_forward"}
    # forward parameter names beyond self (e.g. 'x, add_input') — extra ones are
    # EXTERNAL inputs passed at call time (get_inputs() returns them in order).
    sig = fwd.group(1).strip().lstrip(",").strip()
    fwd_params = [p.strip().split(":")[0].split("=")[0].strip()
                  for p in sig.split(",") if p.strip()]
    extra_inputs = fwd_params[1:] if len(fwd_params) > 1 else []   # all but x
    body = fwd.group(2)

    steps, unknown = [], 0
    for raw in body.splitlines():
        cls = _classify_step(raw)
        if cls is None:
            continue
        op_key, params = cls
        step = {"op": op_key, "params": params, "line": raw.strip()[:60]}
        if op_key == "_unknown":
            unknown += 1
        steps.append(step)

    # drop trailing test-config lines that slipped past (heuristic: stop after the
    # first step that is a bare assignment to batch_size etc — but _classify_step
    # already filters most). Keep only real op steps + markers.
    op_steps = [s for s in steps if not s["op"].startswith("_") or s["op"] in ("_module", "_save")]
    recognized = sum(1 for s in op_steps
                     if s["op"] in OP_LIFT or s["op"] in ("_module", "_save", "residual_add",
                        "bias_add", "tensor_add", "tensor_mul", "tensor_sub", "tensor_div", "clamp", "submean", "binop"))
    return {
        "problem": name,
        "steps": op_steps,
        "n_steps": len(op_steps),
        "recognized": recognized,
        "unknown": sum(1 for s in op_steps if s["op"] == "_unknown"),
        "extra_inputs": extra_inputs,   # forward params beyond x (external tensors)
    }


# ── resolve _module steps to concrete ops via the instantiated KB Model ───────
# Maps an nn submodule type -> our op_key. The chain-lifter records the attr name;
# this resolves it once the model is built (in the sweep), turning _module steps
# into concrete (op, params).
MODULE_OP = {
    "Conv1d": "conv", "Conv2d": "conv", "Conv3d": "conv",
    "ConvTranspose1d": "convtranspose", "ConvTranspose2d": "convtranspose",
    "ConvTranspose3d": "convtranspose",
    "Linear": "matmul", "Bilinear": "matmul",
    "BatchNorm1d": "batchnorm", "BatchNorm2d": "batchnorm", "BatchNorm3d": "batchnorm",
    "GroupNorm": "groupnorm", "LayerNorm": "layernorm",
    "InstanceNorm1d": "instancenorm", "InstanceNorm2d": "instancenorm", "InstanceNorm3d": "instancenorm",
    "MaxPool1d": "maxpool", "MaxPool2d": "maxpool", "MaxPool3d": "maxpool",
    "AvgPool1d": "avgpool", "AvgPool2d": "avgpool", "AvgPool3d": "avgpool",
    "ReLU": "relu", "LeakyReLU": "leaky_relu", "GELU": "gelu", "Sigmoid": "sigmoid",
    "Tanh": "tanh", "SELU": "selu", "ELU": "elu", "Softplus": "softplus",
    "Softmax": "softmax", "LogSoftmax": "log_softmax", "Hardtanh": "hardtanh",
    "Hardswish": "hardswish", "Mish": "mish", "SiLU": "silu", "Swish": "silu",
    # inference-identity + adaptive pooling
    "Dropout": "identity", "Dropout2d": "identity", "Dropout3d": "identity",
    "AdaptiveAvgPool1d": "avgpool", "AdaptiveAvgPool2d": "avgpool", "AdaptiveAvgPool3d": "avgpool",
    "AdaptiveMaxPool1d": "maxpool", "AdaptiveMaxPool2d": "maxpool", "AdaptiveMaxPool3d": "maxpool",
    "Identity": "identity",
}

def resolve_modules(chain_ir, model):
    """Turn _module steps into concrete ops using the model's submodules.
    Returns the chain_ir with each _module step's 'op' replaced (in place)."""
    for step in chain_ir.get("steps", []):
        if step["op"] != "_module":
            continue
        attr = step["params"].get("attr")
        sub = getattr(model, attr, None)
        if sub is None:
            step["op"] = "_unknown"; continue
        kind = type(sub).__name__
        step["op"] = MODULE_OP.get(kind, "_unknown")
        step["module_type"] = kind
        step["_submodule"] = sub        # keep for ggml param extraction
    return chain_ir


# ── ggml param resolution: GGML kind + concrete param per step ───────────────
# Maps a resolved ChainIR step to the ggml op kind AND its concrete scalar param
# (resolved from the model: leaky slope, scale factor, group count, etc.) so the
# emitted ggml C carries REAL values, not defaults.
_GGML_KIND = {
    "relu": "ggml_relu", "leaky_relu": "ggml_leaky_relu", "gelu": "ggml_gelu",
    "silu": "ggml_silu", "sigmoid": "ggml_sigmoid", "tanh": "ggml_tanh",
    "elu": "ggml_elu", "softplus": "ggml_softplus", "hardsigmoid": "ggml_hardsigmoid",
    "hardtanh": "ggml_clamp", "selu": "ggml_elu", "mish": "ggml_mish",
    "identity": "ggml_cont",
    "scaling": "ggml_scale", "scalar_add": "ggml_add", "scalar_sub": "ggml_sub",
    "scalar_div": "ggml_div", "bias_add": "ggml_add", "residual_add": "ggml_add",
    "softmax": "ggml_soft_max_ext", "log_softmax": "ggml_log_soft_max",
    "logsumexp": "ggml_sum_rows", "sum": "ggml_sum_rows", "mean": "ggml_mean",
    "amax": "ggml_argmax", "amin": "ggml_argmin", "argmax": "ggml_argmax",
    "groupnorm": "ggml_group_norm", "batchnorm": "ggml_norm", "layernorm": "ggml_norm",
    "instancenorm": "ggml_norm", "rmsnorm": "ggml_rms_norm",
    "maxpool": "ggml_pool_2d", "avgpool": "ggml_pool_2d",
    "conv": "ggml_conv_2d", "convtranspose": "ggml_conv_transpose_2d",
    "matmul": "ggml_mul_mat",
    "_save": None, "identity_skip": None,
}


def _ggml_param(step, model):
    """Resolve a step's concrete ggml scalar param from the model, or None."""
    op = step["op"]
    p = step["params"]
    sub = step.get("_submodule")
    if op == "leaky_relu":
        return getattr(sub, "negative_slope", 0.01) if sub is not None else 0.01
    if op == "elu":
        return getattr(sub, "alpha", 1.0) if sub is not None else 1.0
    if op == "groupnorm":
        return getattr(sub, "num_groups", 8) if sub is not None else 8
    if op in ("scaling", "scalar_add", "scalar_sub", "scalar_div"):
        operand = p.get("operand")
        m = re.match(r"self\.(\w+)", operand or "")
        if m:
            v = getattr(model, m.group(1), None)
            try:
                return float(v)
            except (TypeError, ValueError):
                return None       # tensor operand (bias) -> binary, no scalar
        try:
            return float(operand)
        except (TypeError, ValueError):
            return None
    if op in ("softmax", "log_softmax", "mean", "sum", "amax", "amin"):
        return p.get("dim", 1)
    return None


def _ggml_kind_for(step):
    """ggml op kind for a step, recovering conv/pool DIMENSIONALITY from the
    module_type (Conv1d/2d/3d, ConvTranspose1d/2d/3d, MaxPool1d/2d/3d) — the
    generic _GGML_KIND collapses these, so we refine here."""
    op = step["op"]
    mt = step.get("module_type", "")
    if op == "conv":
        if "1d" in mt: return "ggml_conv_1d"
        if "3d" in mt: return "ggml_conv_3d"
        return "ggml_conv_2d"
    if op == "convtranspose":
        if "1d" in mt: return "ggml_conv_transpose_1d"
        if "3d" in mt: return "ggml_conv_transpose_3d"
        return "ggml_conv_transpose_2d"
    if op in ("maxpool", "avgpool"):
        return "ggml_pool_1d" if "1d" in mt else "ggml_pool_2d"
    return _GGML_KIND.get(op, "ggml_unknown")


def chain_to_ggml_ir(chain_ir, model):
    """Produce the param-carrying ggml op-list for emit_ggml_graph:
       [op(name, ggml_kind, idx, param), ...]  (param omitted -> op/3).
    Resolves each step's concrete scalar from the model."""
    ops = []
    idx = 0
    for step in chain_ir.get("steps", []):
        op = step["op"]
        if op in ("_save",):           # DAG markers don't emit a ggml node
            continue
        if op not in _GGML_KIND and op not in ("conv", "convtranspose", "maxpool", "avgpool"):
            continue
        kind = _ggml_kind_for(step)
        if kind is None:
            continue
        idx += 1
        short = op.replace("bpd_", "")
        name = f"{short}_{idx}"
        param = _ggml_param(step, model)
        if param is None:
            ops.append(f"op({name}, {kind}, {idx})")
        else:
            ops.append(f"op({name}, {kind}, {idx}, {param})")
    return ops


if __name__ == "__main__":
    KB = sys.argv[1] if len(sys.argv) > 1 else \
        "/tmp/llamatov-data/kernelbench/KernelBench/level2"
    files = sorted(f for f in os.listdir(KB) if f.endswith(".py"))
    total_steps = total_recog = total_unknown = fully = 0
    for f in files:
        r = lift_chain(os.path.join(KB, f))
        if r.get("status") == "no_forward":
            continue
        total_steps += r["n_steps"]; total_recog += r["recognized"]; total_unknown += r["unknown"]
        if r["unknown"] == 0 and r["n_steps"] > 0:
            fully += 1
    print(f"L2 chain-lift: {len(files)} problems")
    print(f"  total op-steps: {total_steps}")
    print(f"  recognized: {total_recog}  unknown: {total_unknown}")
    print(f"  fully-recognized chains: {fully}/{len(files)}")
