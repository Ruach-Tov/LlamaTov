#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""asm_unroll.py — detect and measure loop-unrolling in assembly.

Per Heath's direction (2026-05-31): ggml's assembly has 2x/4x/8x loop unrolling.
For each loop body, detect the repeated instruction-shape sequence and report the
unroll factor. This enables:
  (1) visual collapse in ir_compare_diagram — "show 1 copy + ×N annotation"
  (2) loop-rolling as the inverse-of-O2 transformation — lift unrolled asm back
      toward pre-O2 LLVM-IR ("what kappa is" = read the unroll factor structurally).

Substrate-of-approach:
  1. asm_pcode.lift() gives us the linear instruction chain
  2. asm_loops.find_loops() gives us the loop bodies (between back-edge and header)
  3. For each loop body, NORMALIZE each instruction to a (mnemonic, operand-shape)
     fingerprint where:
       - reg(_)         -> R       (any general/vector register)
       - mem(d,b,i,s)   -> M(d,b)  (any memory with that displacement-class + base)
       - imm(v)         -> I       (any immediate)
       - sym(_)         -> S
  4. Find the longest tandem-repeat: a sequence T such that T*N (concatenated N times)
     covers the body (or a contiguous segment of it).
  5. Report: unroll_factor=N, rolled_body=T, original_body_length=|T|*N

Output: extends asm_loops's per-loop dict with:
   "unroll_factor": int          (1 = not unrolled; N = N-way unrolled)
   "rolled_body": [...]          (one copy of the repeated shape, in fingerprint form)
   "rolled_body_insns": int      (length of the rolled body)

Usage:
   asm_unroll.py BINARY START STOP

Same args as asm_loops.py; the JSON output supersets asm_loops's.
"""
import subprocess, re, sys, json, os

# Allow importing from same dir as a module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asm_pcode
import asm_loops


# ---------- fingerprinting: instructions -> shape tokens ----------

def operand_shape(op):
    """Normalize an operand to its shape-class (ignore specific register names etc.)."""
    if op is None:
        return None
    t = op[0]
    if t == "reg":
        # Group by register class so xmm0/xmm1/xmm2 all collapse to "X",
        # but %rax/%rsp don't collapse with %xmm0.
        name = op[1]
        if re.match(r"x?mm\d+|ymm\d+|zmm\d+", name):
            return ("R", "vec")
        if re.match(r"r[abcd]x|r[sb]p|r[sd]i|r\d+|e[abcd]x|e[sb]p|e[sd]i", name):
            return ("R", "gpr")
        return ("R", "other")
    if t == "imm":
        return ("I",)            # collapse all immediate values
    if t == "imm_sym":
        return ("I",)
    if t == "addr":
        return ("A",)
    if t == "sym":
        return ("S",)
    if t == "mem":
        # Memory shape: keep base-register-class + has_index + scale
        _, disp, base, idx, scale = op
        base_class = None
        if base:
            if re.match(r"r[abcd]x|r[sb]p|r[sd]i|r\d+", base):
                base_class = "gpr"
            else:
                base_class = "other"
        idx_class = None
        if idx:
            if re.match(r"r[abcd]x|r[sb]p|r[sd]i|r\d+", idx):
                idx_class = "gpr"
            else:
                idx_class = "other"
        # Note: we keep disp as a shape-token "has_disp"/"no_disp" rather than the
        # value, because unrolled loads have STRIDING displacements (0x10, 0x20, 0x30...)
        # that are exactly the unroll signature.
        has_disp = disp != 0
        return ("M", base_class, idx_class, scale, has_disp)
    return ("?",)


def insn_fingerprint(insn):
    """Tuple representing the SHAPE of one instruction (ignoring specific reg/imm/disp values)."""
    addr, mnem, ops = insn
    return (mnem, tuple(operand_shape(o) for o in ops))


# ---------- tandem-repeat detection ----------

def detect_unroll(body):
    """Given a body (list of pcode tuples), find the maximum N such that
    the body is a tandem repeat of a shorter sequence T (length |T| = |body|/N).

    Returns (unroll_factor, rolled_body) where rolled_body is one copy of T.
    Returns (1, body) if no repeat detected (i.e., not unrolled).

    We test divisors of len(body) from largest to smallest. For each candidate
    period P = len(body) / N, we check if body[0:P] == body[P:2P] == ... fingerprint-wise.
    """
    fps = [insn_fingerprint(i) for i in body]
    n = len(fps)
    if n < 2:
        return 1, body
    # Try unroll factors from highest to lowest (we want max compression)
    # Only consider unroll factors >= 2 and <= n
    best_factor = 1
    best_period = n
    for factor in range(n, 1, -1):
        if n % factor != 0:
            continue
        period = n // factor
        # Check if body is factor copies of body[0:period]
        ok = True
        first = fps[0:period]
        for k in range(1, factor):
            if fps[k*period:(k+1)*period] != first:
                ok = False
                break
        if ok:
            best_factor = factor
            best_period = period
            break  # first hit is highest factor
    rolled = body[0:best_period]
    return best_factor, rolled


def detect_unroll_segment(body):
    """A weaker variant: find the longest contiguous SEGMENT of body that is unrolled.

    Some loops have a preamble + unrolled body + epilogue. We slide a window over body
    looking for the largest [start, end] segment that has a >= 2 tandem-repeat structure.

    Returns dict with: factor, segment_start, segment_end, rolled_body, preamble, epilogue.
    Returns {factor:1, ...} if no segment unrolled.
    """
    fps = [insn_fingerprint(i) for i in body]
    n = len(fps)
    if n < 4:
        return {"factor": 1, "segment_start": 0, "segment_end": n,
                "rolled_body": body, "preamble": [], "epilogue": []}

    best = {"factor": 1, "segment_start": 0, "segment_end": n,
            "rolled_body": body, "preamble": [], "epilogue": [],
            "score": 0}

    # Try each starting offset; for each, find the best repeating period
    for start in range(n):
        for period in range(1, (n - start) // 2 + 1):
            # how many full periods fit starting here?
            remaining = n - start
            max_factor = remaining // period
            if max_factor < 2:
                continue
            # find largest factor such that body[start:start+factor*period] is factor copies of body[start:start+period]
            first = fps[start:start+period]
            factor = 1
            for k in range(1, max_factor + 1):
                if fps[start + (k-1)*period : start + k*period] == first:
                    factor = k
                else:
                    break
            if factor >= 2:
                # score by total instructions covered (the bigger the segment, the better)
                covered = factor * period
                # prefer larger covered area; tiebreak by larger factor (more compression)
                score = covered * 100 + factor
                if score > best["score"]:
                    seg_end = start + covered
                    best = {
                        "factor": factor,
                        "segment_start": start,
                        "segment_end": seg_end,
                        "rolled_body": body[start:start+period],
                        "preamble": body[0:start],
                        "epilogue": body[seg_end:n],
                        "score": score,
                    }
    return best


# ---------- output formatting ----------

def render_insn(insn):
    """Render a pcode tuple for human reading: 'mnem op1, op2'."""
    addr, mnem, ops = insn
    ops_s = ", ".join(asm_pcode.emit_operand(o) for o in ops if o is not None)
    return f"{mnem} {ops_s}".rstrip()


def render_fp(fp):
    """Render a fingerprint tuple as a short string."""
    mnem, ops = fp
    op_strs = []
    for o in ops:
        if o is None:
            continue
        if o[0] == "R":
            op_strs.append("R" if len(o) == 1 else f"R:{o[1]}")
        elif o[0] == "I":
            op_strs.append("I")
        elif o[0] == "M":
            op_strs.append(f"M[{o[1] or '_'}]")
        else:
            op_strs.append(o[0])
    return f"{mnem} {', '.join(op_strs)}".rstrip()


def main():
    if len(sys.argv) != 4:
        print("Usage: asm_unroll.py BINARY START STOP", file=sys.stderr)
        sys.exit(2)
    binary, start, stop = sys.argv[1], sys.argv[2], sys.argv[3]

    chain = asm_loops.lift(binary, start, stop)  # uses asm_loops's lift form (with tgt)
    loops = asm_loops.find_loops(chain)

    # Also lift via asm_pcode for the semantic-operand form (so we can fingerprint)
    pcode_chain = asm_pcode.lift(binary, start, stop)
    # Index pcode_chain by address for body extraction
    pcode_by_addr = {addr: (addr, mnem, ops) for (addr, mnem, ops) in pcode_chain}

    enriched_loops = []
    for loop in loops:
        header_addr = int(loop["header"], 16)
        back_addr = int(loop["back_edge"], 16)
        # Pull semantic body from pcode_chain
        body = [pcode_by_addr[(addr)] for (addr, _, _) in pcode_chain
                if header_addr <= addr <= back_addr]

        # Detect unrolling, both as full-body and as segment
        full_factor, full_rolled = detect_unroll(body)
        segment = detect_unroll_segment(body)

        # Address ranges for the rolled body (so consumers like asm_dataflow.trace
        # can re-lift JUST one rolled copy via asm_dataflow.lift(binary, start, end)).
        # Per Iyun's coordination: asm_dataflow.trace on the rolled-body gives the
        # clean per-accumulator dataflow signature; unroll factor reconstructs the
        # full summation tree from there.
        full_start_addr = body[0][0] if body else None
        full_end_addr = body[-1][0] if body else None
        seg_body = segment["rolled_body"]
        seg_start_addr = seg_body[0][0] if seg_body else None
        seg_end_addr = seg_body[-1][0] if seg_body else None

        loop_out = dict(loop)
        loop_out["unroll_full"] = {
            "factor": full_factor,
            "rolled_body_insns": len(full_rolled),
            "rolled_body": [render_fp(insn_fingerprint(i)) for i in full_rolled],
            "rolled_body_start_addr": hex(full_rolled[0][0]) if full_rolled else None,
            "rolled_body_end_addr": hex(full_rolled[-1][0]) if full_rolled else None,
        }
        loop_out["unroll_segment"] = {
            "factor": segment["factor"],
            "segment_start_idx": segment["segment_start"],
            "segment_end_idx": segment["segment_end"],
            "preamble_insns": len(segment["preamble"]),
            "epilogue_insns": len(segment["epilogue"]),
            "rolled_body_insns": len(segment["rolled_body"]),
            "rolled_body": [render_fp(insn_fingerprint(i)) for i in segment["rolled_body"]],
            "rolled_body_start_addr": hex(seg_start_addr) if seg_start_addr is not None else None,
            "rolled_body_end_addr": hex(seg_end_addr) if seg_end_addr is not None else None,
        }
        enriched_loops.append(loop_out)

    out = {
        "total_insns": len(chain),
        "loops_found": len(loops),
        "loops": enriched_loops,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
