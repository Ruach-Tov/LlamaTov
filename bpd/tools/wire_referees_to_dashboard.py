#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""wire_referees_to_dashboard.py — closed-form bridge from referee outputs
to bpd/lib/robust_op_match.o.pl (Table(10100) facts).

Consumes BOTH referee output formats:

  (1) stanford_referee.py — human-readable text rows:
        problem-name              STATUS         detail-string
      Statuses: BIT_IDENTICAL / ROBUST_GAP / DIVERGENT / SHAPE_DIVERGENT /
                MISSING_KERNEL / NO_FILE / NOT_ROUTED

  (2) verify_robust_ops_auto.py — Prolog facts:
        robust_op_result(Op, Ref, Tier, [evidence], [coordinates_pinned]).

Unifies both into robust_op_match/5 facts the dashboard reads.

No human in the loop. The judge (stanford_referee) is authoritative; the
dashboard renders the verdict. Size-sweep and edge-case stability are part
of the referee's gate (Stanford gate + robust gate). Per Heath's framing:
the dashboard IS the accept/reject visualization, the referee IS the judge.

Status -> tier mapping (closed-form):

    BIT_IDENTICAL + robust gate pass     -> gold     (0-ULP across sweep + edges)
    BIT_IDENTICAL, no robust info        -> silver   (0-ULP canonical only)
    ROBUST_GAP                           -> silver   (canonical 0-ULP, robust fails)
    SHAPE_DIVERGENT                      -> bronze   (some shapes pass)
    DIVERGENT (max ULP <= 4)             -> bronze
    DIVERGENT (max ULP > 4) / MISSING    -> untested (no closure achieved)

Usage:
    # Read both sources, write unified facts file
    python3 bpd/bench/stanford_referee.py > /tmp/referee.txt
    python3 bpd/tests/verify_robust_ops_auto.py > /tmp/robust_harness.txt
    python3 bpd/tools/wire_referees_to_dashboard.py \\
        --stanford /tmp/referee.txt \\
        --harness  /tmp/robust_harness.txt \\
        > bpd/lib/robust_op_match.o.pl
"""
import argparse
import re
import sys
from collections import OrderedDict


# ---- pattern membership (must match bpd/robust_match_status.pl emission_pattern/3) ----

PATTERN_OF = {
    # unary_elementwise (covers the L1 referee's current scope).
    # Aliases included: stanford_referee normalizes "LeakyReLU" -> "leakyrelu"
    # via lower(), so both "leaky_relu" (snake) and "leakyrelu" (camel-collapsed)
    # need to resolve to the same pattern.
    "relu":          "unary_elementwise",
    "leaky_relu":    "unary_elementwise",
    "leakyrelu":     "unary_elementwise",
    "sigmoid":       "unary_elementwise",
    "tanh":          "unary_elementwise",
    "swish":         "unary_elementwise",
    "gelu":          "unary_elementwise",
    "selu":          "unary_elementwise",
    "elu":           "unary_elementwise",
    "hardsigmoid":   "unary_elementwise",
    "hard_sigmoid":  "unary_elementwise",
    "softplus":      "unary_elementwise",
    "softsign":      "unary_elementwise",
    "exp":           "unary_elementwise",
    "log":           "unary_elementwise",
    "sqrt":          "unary_elementwise",
    "abs":           "unary_elementwise",
    "ceil":          "unary_elementwise",
    "neg":           "unary_elementwise",
    # binary_elementwise
    "add":           "binary_elementwise",
    # reduction
    "sum":           "reduction",
    "mean":          "reduction",
    "max":           "reduction",
    "min":           "reduction",
    "norm":          "reduction",
    "argmax":        "reduction",
    "argmin":        "reduction",
    # reduction_then_elementwise
    "softmax":       "reduction_then_elementwise",
    "log_softmax":   "reduction_then_elementwise",
    "layer_norm":    "reduction_then_elementwise",
    "rms_norm":      "reduction_then_elementwise",
    # matmul -> conv_im2col by Stanford taxonomy convention (matmul is row 6 in Table)
    "matmul":        "conv_im2col",
    "mm":            "conv_im2col",
}


def normalize_op(raw):
    """Normalize an op name from any referee source to the PATTERN_OF key.

    stanford_referee names: '19_ReLU' / '21_Sigmoid' / 'matmul' (already normalized).
    harness names: 'relu' / 'sigmoid' / 'elu' / etc.
    """
    # Strip Stanford problem prefix: "19_ReLU" -> "relu"
    m = re.match(r"^\d+_(.+?)(_)?$", raw)
    if m:
        raw = m.group(1)
    # Strip "Square_matrix_multiplication" -> "matmul"
    if "matrix_multiplication" in raw.lower():
        return "matmul"
    return raw.lower()


def bpd_op_name(op):
    """Render the op as the bpd_<op> form the dashboard renders."""
    return f"bpd_{op}"


# ---- parsers ----

STATUS_ALTS = "BIT_IDENTICAL|ROBUST_GAP|DIVERGENT|SHAPE_DIVERGENT|MISSING_KERNEL|NO_FILE|NOT_ROUTED"
# Match status with leading whitespace OR underscore (stanford_referee's
# <26-char-wide format runs problem name and status together when the
# problem name exceeds 26 chars: "1_Square_matrix_multiplication_BIT_IDENTICAL").
STANFORD_LINE = re.compile(
    rf"^(\S+?)[\s_]+({STATUS_ALTS})\s*(.*)$"
)


def parse_stanford(text):
    """Parse stanford_referee.py output text.

    Returns list of (op, status, detail) tuples.
    """
    rows = []
    for line in text.splitlines():
        m = STANFORD_LINE.match(line.strip())
        if m:
            problem_raw, status, detail = m.group(1), m.group(2), m.group(3).strip()
            op = normalize_op(problem_raw)
            rows.append((op, status, detail))
    return rows


HARNESS_LINE = re.compile(
    r"robust_op_result\((\w+),\s*(\w+),\s*(\w+),\s*(\[.*?\]),\s*(\[.*?\])\)\.\s*$"
)


def parse_harness(text):
    """Parse verify_robust_ops_auto.py output Prolog facts.

    Returns list of (op, ref, tier, evidence, coords) tuples.
    """
    rows = []
    for line in text.splitlines():
        m = HARNESS_LINE.match(line.strip())
        if m:
            rows.append(m.group(1, 2, 3, 4, 5))
    return rows


# ---- status -> tier mapping (closed-form, no human gating) ----

def stanford_status_to_tier(status, detail):
    """Map stanford_referee status + detail to (tier, evidence_terms).

    Status atoms are quoted with single-quotes so Prolog reads them as atoms
    rather than variables (BIT_IDENTICAL etc. start uppercase = variable).
    """
    status_atom = f"'{status}'"
    if status == "BIT_IDENTICAL":
        # +robust marker in detail means robust gate also passed
        if "+robust" in detail.lower():
            return "gold", [f"stanford_status({status_atom})",
                            f"detail(\"{detail}\")",
                            "robust_gate(passed)"]
        return "silver", [f"stanford_status({status_atom})",
                          f"detail(\"{detail}\")",
                          "robust_gate(not_run_or_unmarked)"]
    if status == "ROBUST_GAP":
        return "silver", [f"stanford_status({status_atom})",
                          f"detail(\"{detail}\")",
                          "canonical_pass(true)",
                          "robust_gate(failed)"]
    if status == "SHAPE_DIVERGENT":
        return "bronze", [f"stanford_status({status_atom})",
                          f"detail(\"{detail}\")"]
    if status == "DIVERGENT":
        # Extract max ULP from detail if possible (e.g. "64x1024:2 256x1024:3")
        ulps = re.findall(r":(\d+)", detail)
        max_ulp = max(int(u) for u in ulps) if ulps else 999
        if max_ulp <= 4:
            return "bronze", [f"stanford_status({status_atom})",
                              f"detail(\"{detail}\")",
                              f"max_ulp({max_ulp})"]
        return "untested", [f"stanford_status({status_atom})",
                            f"detail(\"{detail}\")",
                            f"max_ulp({max_ulp})",
                            "above_bronze_threshold(true)"]
    # MISSING_KERNEL, NO_FILE, NOT_ROUTED
    return "untested", [f"stanford_status({status_atom})"]


def harness_to_facts(rows):
    """Translate harness rows to robust_op_match facts.

    Harness already emits tier directly; we just translate the shape.
    """
    out = []
    for op, ref, tier, evidence_inner, coords in rows:
        pattern = PATTERN_OF.get(op)
        if not pattern:
            continue
        # Strip outer [] from evidence and append coordinates_pinned/1
        ev_inner = evidence_inner[1:-1].strip()
        merged_ev = f"[{ev_inner},coordinates_pinned({coords})]" if ev_inner \
                    else f"[coordinates_pinned({coords})]"
        out.append((pattern, bpd_op_name(op), ref, tier, merged_ev))
    return out


def stanford_to_facts(rows, reference="pytorch_mkl"):
    """Translate stanford_referee rows to robust_op_match facts.

    Stanford referee always compares against pytorch (Model.forward()),
    so the reference column is pytorch_mkl by default.
    """
    out = []
    for op, status, detail in rows:
        pattern = PATTERN_OF.get(op)
        if not pattern:
            continue
        tier, evidence = stanford_status_to_tier(status, detail)
        ev_str = "[" + ",".join(evidence) + "]"
        out.append((pattern, bpd_op_name(op), reference, tier, ev_str))
    return out


# ---- precedence: lowest tier wins (Heath: gold = 0 ULP, any divergence -> not gold) ----

TIER_ORDER = {"gold": 4, "silver": 3, "bronze": 2, "untested": 1}


def _tier_rank(fact):
    return TIER_ORDER.get(fact[3], 0)


def _merge_evidence(low_fact, high_fact):
    """When the conservative (low) tier wins, retain its evidence but APPEND
    the higher-tier fact's evidence as a disagreement_with/1 term so the
    drill-down can show what the other referee saw."""
    low_ev = low_fact[4]
    high_ev = high_fact[4]
    # low_ev and high_ev are both "[...]" strings; embed high_ev as a Prolog
    # disagreement_with/1 term tagged with the higher tier
    disagreement_term = f"disagreement_with({high_fact[3]},{high_ev})"
    low_ev_inner = low_ev[1:-1].strip()
    if low_ev_inner:
        merged_ev = f"[{low_ev_inner},{disagreement_term}]"
    else:
        merged_ev = f"[{disagreement_term}]"
    return (low_fact[0], low_fact[1], low_fact[2], low_fact[3], merged_ev)


def merge_facts(stanford_facts, harness_facts):
    """When both sources cover the same cell, the LOWEST tier wins.

    Per Heath (2026-05-31): "the standard for Gold is a 0 ULP match. Any
    divergence above 0 → not gold." So if stanford reports ROBUST_GAP (silver)
    and harness reports gold for the same cell, the cell renders SILVER and
    the harness's gold-evidence is captured as disagreement_with/1 in the
    silver fact's evidence (queryable in drill-down).

    Tier order: gold(4) > silver(3) > bronze(2) > untested(1).
    """
    by_key = {}
    for f in stanford_facts + harness_facts:
        key = (f[0], f[1], f[2])
        if key in by_key:
            existing = by_key[key]
            # Compare ranks: keep the LOWER one, attach the higher as disagreement
            if _tier_rank(f) < _tier_rank(existing):
                by_key[key] = _merge_evidence(f, existing)
            elif _tier_rank(f) > _tier_rank(existing):
                by_key[key] = _merge_evidence(existing, f)
            # equal: keep first (stanford was inserted first via list concat order)
        else:
            by_key[key] = f
    return list(by_key.values())


# ---- emit ----

HEADER = """%% robust_op_match.o.pl — Table(10100) facts, GENERATED FILE.
%%
%% Auto-generated by bpd/tools/wire_referees_to_dashboard.py from:
%%   - stanford_referee.py (authoritative L1 judge: Stanford gate + robust gate)
%%   - verify_robust_ops_auto.py (Iyun's harness, emits coordinates_pinned/1)
%%
%% Per Heath's direction (2026-05-31): Table(10100) IS the accept/reject
%% visualization. The referee is the closed-form judge. No human in the loop.
%%
%% Tier mapping (status -> tier):
%%   BIT_IDENTICAL + robust pass     -> gold
%%   BIT_IDENTICAL canonical only    -> silver
%%   ROBUST_GAP                      -> silver
%%   SHAPE_DIVERGENT                 -> bronze
%%   DIVERGENT (ULP <= 4)            -> bronze
%%   DIVERGENT (ULP > 4) / MISSING   -> untested
%%
%% DO NOT EDIT BY HAND — re-run the pipeline.

:- module(robust_op_match, [robust_op_match/5]).

"""


def emit(facts):
    print(HEADER, end="")
    for (pattern, op, ref, tier, ev) in facts:
        print(f"robust_op_match({pattern}, {op}, {ref}, {tier}, {ev}).")
    print(f"\n%% {len(facts)} facts emitted")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stanford", help="Path to stanford_referee.py output (text)")
    ap.add_argument("--harness",  help="Path to verify_robust_ops_auto.py output (Prolog)")
    args = ap.parse_args()

    stanford_facts = []
    if args.stanford:
        with open(args.stanford) as f:
            stanford_facts = stanford_to_facts(parse_stanford(f.read()))

    harness_facts = []
    if args.harness:
        with open(args.harness) as f:
            harness_facts = harness_to_facts(parse_harness(f.read()))

    facts = merge_facts(stanford_facts, harness_facts)
    emit(facts)


if __name__ == "__main__":
    main()
