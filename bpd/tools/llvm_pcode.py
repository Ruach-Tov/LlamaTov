#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""llvm_pcode.py — lift LLVM-IR into the same threaded-pcode fact form as asm_pcode.py.

LLVM-IR is a TYPED SSA instruction LIST (linear, like asm) — so it lifts to a parse list too:
    llvm(Index, Result, Opcode, Type, [Operands]).

This is the third fact-language (with c_ast tree + asm pcode list). LLVM-IR is the ideal
0-ULP SPEC carrier because it's TYPED: fdiv-vs-fmul (the form axis), float-vs-double +
fpext/fptrunc (the acc_type axis) are all explicit. Operands modeled: ssa(N), lit(V,Type),
arg(Name), const(V). Bidirectional (lift + emit). Transforms to/from asm-pcode + c_ast =
the compiler-in-Prolog: same recipe expressible in all three, 0-ULP-checkable in each.
"""
import subprocess, re, sys, json

# opcodes that carry fp-precision semantics (the 0-ULP-relevant ops)
FP_OPS = {"fadd","fsub","fmul","fdiv","fneg","fpext","fptrunc","fcmp","fma","frem"}
MEM_OPS = {"load","store","getelementptr"}
CALL = "call"

def _ret_type(rest):
    m = re.search(r"\b(float|double|half|i\d+|ptr|void)\b", rest)
    return m.group(1) if m else None

def parse_operand(tok):
    tok = tok.strip().rstrip(",")
    if not tok: return None
    if tok.startswith("%"):
        return ("ssa", tok[1:])
    if tok.startswith("@"):
        return ("glob", tok[1:])
    # float literal like 1.000000e+00 or 0x... or int
    if re.match(r"^-?\d+\.\d+e[+-]\d+$", tok) or re.match(r"^0x[0-9A-Fa-f]+$", tok):
        return ("fp_lit", tok)
    if re.match(r"^-?\d+$", tok):
        return ("int_lit", int(tok))
    return ("tok", tok)

def lift(ll_path):
    chain = []
    with open(ll_path) as f:
        for line in f:
            s = line.strip()
            # form: %res = opcode type op1, op2 ...   OR   store/call without result
            # normalize: drop a leading 'tail ' so 'tail call' -> 'call'
            s_norm = s[5:] if s.startswith("tail ") else s
            m = re.match(r"(%\S+)\s*=\s*(?:tail\s+)?([a-z]+)\s+(.*)", s_norm if s.startswith("tail ") else s)
            if m:
                res, op, rest = m.group(1)[1:], m.group(2), m.group(3)
            else:
                m2 = re.match(r"(store|call|ret|br)\s+(.*)", s_norm)
                if not m2: continue
                res, op, rest = None, m2.group(1), m2.group(2)
            # for a call, model the callee as a glob operand and keep the opcode 'call'
            if op == "call":
                cm = re.search(r"@(\w+)", rest)
                callee = cm.group(1) if cm else "?"
                chain.append((res, "call", _ret_type(rest), [("glob", callee)]))
                continue
            if op not in FP_OPS and op not in MEM_OPS and op != CALL and op not in {"ret","br","tail"}:
                # also catch 'tail call'
                if not s.startswith("tail call"): continue
            # extract a leading type token if present (float/double/i32/ptr)
            tym = re.match(r"((?:float|double|half|i\d+|ptr)\b)\s*(.*)", rest)
            typ = tym.group(1) if tym else None
            body = tym.group(2) if tym else rest
            # operands: split on commas, strip type prefixes inside
            ops = []
            for piece in body.split(","):
                piece = piece.strip()
                # drop leading type on each operand (e.g. "float %11" -> "%11")
                pm = re.match(r"(?:float|double|half|i\d+|ptr|noundef|nonnull)\s+(.*)", piece)
                tok = pm.group(1) if pm else piece
                tok = tok.split()[0] if tok.split() else tok
                po = parse_operand(tok)
                if po: ops.append(po)
            chain.append((res, op, typ, ops))
    return chain

def to_prolog(chain):
    out = []
    for n, (res, op, typ, ops) in enumerate(chain):
        opl = "[" + ",".join(_term(o) for o in ops) + "]"
        r = f"ssa('{res}')" if res else "none"
        out.append(f"llvm({n}, {r}, {op}, {typ or 'none'}, {opl}).")
    return "\n".join(out)

def _term(o):
    t = o[0]
    if t == "ssa": return f"ssa('{o[1]}')"
    if t == "glob": return f"glob('{o[1]}')"
    if t == "fp_lit": return f"fp_lit('{o[1]}')"
    if t == "int_lit": return f"int_lit({o[1]})"
    return f"tok('{o[1]}')"

# --- 0-ULP-relevant structural queries on the IR (typed = precise) ---
def query_form(chain):
    """Detect divide vs reciprocal-mul form, and acc precision (fpext/fptrunc presence)."""
    has_fdiv = any(op=="fdiv" for _,op,_,_ in chain)
    has_fmul = any(op=="fmul" for _,op,_,_ in chain)
    has_fpext = any(op=="fpext" for _,op,_,_ in chain)
    has_fptrunc = any(op=="fptrunc" for _,op,_,_ in chain)
    types = set(t for _,_,t,_ in chain if t in {"float","double"})
    fp_seq = [op for _,op,_,_ in chain if op in FP_OPS]
    return {
        "form": "divide" if has_fdiv and not (has_fmul and not has_fdiv) else ("reciprocal_mul" if has_fmul else "none"),
        "has_fdiv": has_fdiv, "has_fmul": has_fmul,
        "acc_widening": has_fpext, "narrowing": has_fptrunc,
        "fp_types_used": sorted(types),
        "fp_opcode_sequence": fp_seq,
    }

def query_fusion(chain):
    stages=[]; prev=None
    for (res,op,typ,ops) in chain:
        if op=='fcmp': prev='fcmp'
        elif op=='select' and prev=='fcmp': stages.append('relu_or_clamp'); prev=None
        elif op=='fadd': stages.append('add_bias_or_residual')
        elif op=='fsub': stages.append('subtract')
        elif op=='fmul': stages.append('scale_or_mul')
        elif op=='fdiv': stages.append('divide')
        elif op=='call':
            callee = ops[0][1] if ops and ops[0][0]=='glob' else '?'
            stages.append('call:'+str(callee))
        elif op in ('fpext','fptrunc'): stages.append('cast:'+op)
    nonr=[x for x in stages if not x.startswith('cast')]
    return {'num_stages':len(stages),'fused_chain':stages,'is_fusion':len(nonr)>1}

if __name__ == "__main__":
    path, mode = sys.argv[1], (sys.argv[2] if len(sys.argv)>2 else "prolog")
    chain = lift(path)
    if mode == "prolog": print(to_prolog(chain))
    elif mode == "form": print(json.dumps(query_form(chain), indent=2))
    elif mode == "fusion": print(json.dumps(query_fusion(chain), indent=2))
    else: print(json.dumps({"instrs": len(chain)}))
