#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""asm_pcode.py — BIDIRECTIONAL assembly <-> threaded p-code lift.

Assembly is a LINEAR sequence, so it lifts to a parse LIST (threaded p-code / tokenized
form), not a tree (unlike C, which needs c_ast's tree because it nests). The primary output
is a chain of Prolog facts modeling each instruction:

    pcode(Index, Addr, Mnemonic, [Operand,...]).

with operands themselves modeled (not raw text) so the lift is SEMANTIC, round-trippable:
    reg(Name)            %xmm0 -> reg(xmm0)
    mem(Disp, Base, Index, Scale)   0x10(%r8,%rax,4)
    imm(Value)           $0x20 -> imm(32)
    sym(Name)            a label/symbol operand

Bidirectional: lift(asm_text) -> [pcode...]; emit([pcode...]) -> asm_text.
Round-trip is semantic modulo formatting (same discipline as c_ast c_float_lit).
This is the asm analog of c_ast.pl. With transforms pcode <-> LLVM-IR, the Prolog
substrate becomes a compiler that can model the platform's own output.
"""
import subprocess, re, sys, json

# ---------- operand model (semantic, not raw text) ----------
def parse_operand(op):
    op = op.strip()
    if not op: return None
    if op.startswith("%"):
        return ("reg", op[1:])
    if op.startswith("$"):
        v = op[1:]
        try: return ("imm", int(v, 0))
        except ValueError: return ("imm_sym", v)
    # memory: [disp](base,index,scale)  e.g. 0x10(%r8,%rax,4) or (%rax) or 0x60(%rsp)
    m = re.match(r"(-?0x[0-9a-f]+|-?\d+)?\((%\w+)(?:,(%\w+))?(?:,(\d+))?\)", op)
    if m:
        disp = int(m.group(1), 0) if m.group(1) else 0
        base = m.group(2)[1:] if m.group(2) else None
        idx  = m.group(3)[1:] if m.group(3) else None
        scale = int(m.group(4)) if m.group(4) else 1
        return ("mem", disp, base, idx, scale)
    if re.match(r"^-?0x[0-9a-f]+$", op) or re.match(r"^-?\d+$", op):
        return ("addr", int(op, 0))
    return ("sym", op)

def lift(binary, start, stop):
    out = subprocess.run(["objdump","-d","--no-show-raw-insn",
                          f"--start-address={start}", f"--stop-address={stop}", binary],
                         capture_output=True, text=True).stdout
    chain = []
    for line in out.splitlines():
        m = re.match(r"\s*([0-9a-f]+):\s+([a-z][a-z0-9.]*)\s*(.*)", line)
        if not m: continue
        addr = int(m.group(1), 16); mnem = m.group(2)
        ops_s = m.group(3).split("#")[0].strip()
        # split operands on commas NOT inside parens
        ops = []
        if ops_s:
            depth = 0; cur = ""
            for ch in ops_s:
                if ch == "(": depth += 1
                if ch == ")": depth -= 1
                if ch == "," and depth == 0:
                    ops.append(cur); cur = ""
                else: cur += ch
            if cur: ops.append(cur)
        ops = [parse_operand(o) for o in ops]
        chain.append((addr, mnem, ops))
    return chain

# ---------- lift from pre-rendered objdump text (no objdump call) ----------
def lift_from_text(text):
    """Parse objdump-style asm text (one instruction per line, leading hex addr)
    into the same (addr, mnem, ops) chain that lift() produces from a binary.

    Used when the caller already has objdump output as a .asm file (e.g., the
    ir_compare_diagram dashboard which reads .asm files from /tmp/output-only/).
    """
    chain = []
    for line in text.splitlines():
        m = re.match(r"\s*([0-9a-f]+):\s+([a-z][a-z0-9.]*)\s*(.*)", line)
        if not m:
            continue
        addr = int(m.group(1), 16)
        mnem = m.group(2)
        ops_s = m.group(3).split("#")[0].strip()
        # split operands on commas NOT inside parens (same logic as lift())
        ops = []
        if ops_s:
            depth = 0
            cur = ""
            for ch in ops_s:
                if ch == "(":
                    depth += 1
                if ch == ")":
                    depth -= 1
                if ch == "," and depth == 0:
                    ops.append(cur)
                    cur = ""
                else:
                    cur += ch
            if cur:
                ops.append(cur)
        ops = [parse_operand(o) for o in ops]
        chain.append((addr, mnem, ops))
    return chain


# ---------- emit: pcode chain -> AT&T asm text (reverse direction) ----------
def emit_operand(o):
    t = o[0]
    if t == "reg": return f"%{o[1]}"
    if t == "imm": return f"${hex(o[1])}"
    if t == "imm_sym": return f"${o[1]}"
    if t == "addr": return hex(o[1])
    if t == "sym": return o[1]
    if t == "mem":
        _, disp, base, idx, scale = o
        d = hex(disp) if disp else ""
        inner = f"%{base}" if base else ""
        if idx: inner += f",%{idx},{scale}"
        return f"{d}({inner})"
    return "?"

def emit(chain):
    lines = []
    for (addr, mnem, ops) in chain:
        ops_s = ", ".join(emit_operand(o) for o in ops)
        lines.append(f"{mnem} {ops_s}".rstrip())
    return "\n".join(lines)

# ---------- Prolog fact serialization ----------
def to_prolog_term(o):
    t = o[0]
    if t == "reg": return f"reg({o[1]})"
    if t == "imm": return f"imm({o[1]})"
    if t == "imm_sym": return f"imm_sym('{o[1]}')"
    if t == "addr": return f"addr({o[1]})"
    if t == "sym": return f"sym('{o[1]}')"
    if t == "mem": return f"mem({o[1]},{o[2] or 'none'},{o[3] or 'none'},{o[4]})"
    return "unknown"

def to_prolog(chain):
    out = []
    for n, (addr, mnem, ops) in enumerate(chain):
        opl = "[" + ",".join(to_prolog_term(o) for o in ops) + "]"
        out.append(f"pcode({n}, {hex(addr)}, {mnem.replace('.','_')}, {opl}).")
    return "\n".join(out)

if __name__ == "__main__":
    binary, start, stop, mode = sys.argv[1], sys.argv[2], sys.argv[3], (sys.argv[4] if len(sys.argv)>4 else "prolog")
    chain = lift(binary, start, stop)
    if mode == "prolog":
        print(to_prolog(chain))
    elif mode == "emit":            # reverse: pcode -> asm text
        print(emit(chain))
    elif mode == "roundtrip":       # lift -> emit, compare to a re-lift of the emitted text
        asm1 = emit(chain)
        # semantic round-trip check: re-parse the emitted operands, compare structure
        relifted = [(a, m, [parse_operand(emit_operand(o)) for o in ops]) for (a,m,ops) in chain]
        same = all(ops == r[2] for (a,m,ops), r in zip(chain, relifted))
        print(json.dumps({"instrs": len(chain), "semantic_roundtrip_ok": same}))
    else:
        print(json.dumps({"instrs": len(chain)}))
