#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""asm_facts.py — lift a binary function's disassembly into structured facts (Prolog-ready).
Takes (binary, symbol-or-address-range), runs objdump, parses each instruction into a fact:
  instr(Index, Addr, Mnemonic, [Operands], RawText).
Then answers structural queries used by the 0-ULP recipe work:
  - final_rounding_placement: the last fp-narrowing conversion (cvtsd2ss/cvtpd2ps/fptrunc) before a store
  - accumulator_structure: count of distinct fp-accumulator registers + lane width (addpd vs addss)
  - reduction_combine_order: the sequence of partial-sum combines in the epilogue
This is the automated 'asm-read' step: read the spec from the binary instead of guessing.
Emits Prolog facts so the query logic lives in Prolog (queryable from the OO substrate).
"""
import subprocess, re, sys, json

def disasm(binary, start, stop):
    out = subprocess.run(["objdump","-d","--no-show-raw-insn",
                          f"--start-address={start}", f"--stop-address={stop}", binary],
                         capture_output=True, text=True).stdout
    instrs = []
    for line in out.splitlines():
        m = re.match(r"\s*([0-9a-f]+):\s+([a-z][a-z0-9.]*)\s*(.*)", line)
        if not m: continue
        addr = int(m.group(1), 16); mnem = m.group(2); ops_s = m.group(3).split("#")[0].strip()
        ops = [o.strip() for o in ops_s.split(",")] if ops_s else []
        instrs.append({"addr": addr, "mnem": mnem, "ops": ops, "raw": f"{mnem} {ops_s}".strip()})
    return instrs

# --- instruction classification (the semantic lift) ---
FP_NARROW = {"cvtsd2ss","cvtpd2ps","vcvtsd2ss","vcvtpd2ps"}      # f64->f32 (the final rounding)
FP_WIDEN  = {"cvtss2sd","cvtps2pd","vcvtss2sd","vcvtps2pd"}      # f32->f64 (cast on load)
FP_ADD_PACKED = {"addpd","vaddpd","addps","vaddps"}             # packed (SIMD lanes)
FP_ADD_SCALAR = {"addsd","vaddsd","addss","vaddss"}             # scalar (tail / combine)
FP_MUL = {"mulsd","mulpd","mulss","mulps","vmulsd","vmulpd"}
STORE = {"movsd","movss","movups","movaps","vmovups","vmovsd"}  # to memory when dest is mem

def is_mem(op): return "(" in op or op.startswith("0x")
def regs(ops): return [o for o in ops if o.startswith("%")]

def query_final_rounding(instrs):
    """Find the last f64->f32 narrowing conversion + whether a divide/mul precedes it (mean form)."""
    narrows = [i for i in instrs if i["mnem"] in FP_NARROW]
    if not narrows: return {"found": False, "note": "no f64->f32 narrowing in range"}
    last = narrows[-1]
    # look at the ~6 instructions before it for a div/mul (the mean scaling)
    idx = instrs.index(last)
    window = instrs[max(0,idx-6):idx]
    divs = [w["raw"] for w in window if w["mnem"] in {"divsd","divpd","divss"}]
    muls = [w["raw"] for w in window if w["mnem"] in FP_MUL]
    return {"found": True, "addr": hex(last["addr"]), "insn": last["raw"],
            "narrowing": last["mnem"], "preceding_div": divs, "preceding_mul": muls,
            "scaling_form": "divide" if divs else ("reciprocal_mul" if muls else "none_in_window")}

def query_accumulator_structure(instrs):
    packed = [i for i in instrs if i["mnem"] in FP_ADD_PACKED]
    scalar = [i for i in instrs if i["mnem"] in FP_ADD_SCALAR]
    acc_regs = set()
    for i in packed+scalar: acc_regs.update(regs(i["ops"]))
    widen = [i for i in instrs if i["mnem"] in FP_WIDEN]
    return {"packed_adds": len(packed), "scalar_adds": len(scalar),
            "distinct_acc_regs": len(acc_regs), "acc_regs": sorted(acc_regs),
            "lane_width": "2x f64 (packed xmm)" if any(i["mnem"]=="addpd" for i in packed) else "scalar/other",
            "widen_loads": len(widen)}


# --- F16C f32<->f16 conversion + rounding-mode facts (Iyun: lift rounding mode, compare to ggml) ---
FP_NARROW_F16 = {"vcvtps2ph", "cvtps2ph"}   # f32 -> f16 (the cache-store rounding)
FP_WIDEN_F16  = {"vcvtph2ps", "cvtph2ps"}   # f16 -> f32 (the cache-load)

def _f16_round_name(imm):
    # vcvtps2ph imm8: bit 2 (0x04) -> use MXCSR; else bits 1-0 select static mode.
    if imm & 0x04:
        return "round_mxcsr"
    return {0x0: "round_nearest_even", 0x1: "round_down",
            0x2: "round_up", 0x3: "round_toward_zero"}.get(imm & 0x03, "round_unknown")

def query_f16_conversion(instrs):
    """Lift every f32->f16 narrowing conversion + its ROUNDING MODE (imm8 of vcvtps2ph).
    ggml uses _cvtss_sh(x,0) = round-to-nearest-even (imm 0). Flag any non-RNE conversion."""
    convs = []
    for i in instrs:
        if i["mnem"] in FP_NARROW_F16:
            imm = None
            for op in i.get("ops", []):
                if op.startswith("$0x"):
                    try: imm = int(op[1:], 16)
                    except ValueError: imm = None
                    break
            convs.append({"addr": hex(i["addr"]), "insn": i["raw"], "imm": imm,
                          "rounding_mode": _f16_round_name(imm) if imm is not None else "unknown",
                          "matches_ggml": (_f16_round_name(imm) == "round_nearest_even")})  # semantic: RNE (NO_EXC 0x8 flag is RNE-equivalent)
    n_bad = sum(1 for c in convs if not c["matches_ggml"])
    return {"f32_to_f16_conversions": len(convs), "non_rne": n_bad, "conversions": convs,
            "verdict": ("matches_ggml (all RNE imm=0)" if convs and n_bad == 0
                        else ("no f16 conversion in range" if not convs else "DIVERGENT rounding mode"))}



# --- KERNEL SIGNATURE: lift the 5 parameter classes for static parameter-conformance diffing ---
# Each class corresponds to a bit-identity bug class we hit empirically:
#   1 vector_width (softmax 4-vs-8), 2 rounding_modes (f16/quantize), 3 reduction_tree (tiling/hsum),
#   4 numeric_constants (127/eps/scale), 5 reciprocal_form (127/max vs 1/d, divps vs rcpps).

# instructions whose operand width tells us the SIMD block width
_WIDTH_OPS = ("add","sub","mul","max","min","mov","load","cvt","round","shuf","blend","fmadd","and","or","xor")
def _op_width(i):
    # %ymm = 256-bit (8x f32), %xmm = 128-bit (4x f32), %zmm = 512-bit (16x f32)
    for o in i["ops"]:
        if "%zmm" in o: return 512
        if "%ymm" in o: return 256
        if "%xmm" in o: return 128
    return None

_ROUND_INSNS = {"vcvtps2ph","cvtps2ph","vroundps","vroundss","vroundpd","vroundsd","roundps","roundss"}
_RECIP_DIV   = {"divps","divss","divpd","divsd","vdivps","vdivss","vdivpd","vdivsd"}
_RECIP_APPROX= {"rcpps","rcpss","vrcpps","vrcpss"}
_REDUCE_INSNS= {"vextractf128","extractf128","vextracti128","movhlps","vmovhlps","vshufps","shufps",
                "vunpckhps","unpckhps","vperm2f128","vmovshdup","movshdup","vhaddps","haddps"}

def _imm(i):
    for o in i["ops"]:
        if o.startswith("$0x"):
            try: return int(o[1:], 16)
            except ValueError: return None
    return None

def lift_signature(instrs):
    # 1. VECTOR WIDTH histogram (across width-bearing ops)
    widths = {}
    for i in instrs:
        if any(i["mnem"].startswith(p) or p in i["mnem"] for p in _WIDTH_OPS):
            w = _op_width(i)
            if w: widths[w] = widths.get(w, 0) + 1
    max_width = max(widths) if widths else None

    # 2. ROUNDING MODES (the imm of conversion/round instructions)
    rounds = []
    for i in instrs:
        if i["mnem"] in _ROUND_INSNS:
            imm = _imm(i)
            rounds.append({"insn": i["mnem"], "imm": imm,
                           "mode": _round_mode_name(i["mnem"], imm)})

    # 3. REDUCTION TREE SHAPE (the ordered sequence of hsum/lane-combine ops)
    tree = [i["mnem"] for i in instrs if i["mnem"] in _REDUCE_INSNS]

    # 4. NUMERIC CONSTANTS (immediates referenced; shift/tile/scale params)
    consts = {}
    for i in instrs:
        imm = _imm(i)
        if imm is not None and i["mnem"] not in _ROUND_INSNS:
            consts[imm] = consts.get(imm, 0) + 1

    # 5. RECIPROCAL FORM (divide vs approx-reciprocal vs none)
    n_div = sum(1 for i in instrs if i["mnem"] in _RECIP_DIV)
    n_rcp = sum(1 for i in instrs if i["mnem"] in _RECIP_APPROX)
    recip = "true_divide" if n_div else ("approx_rcp" if n_rcp else "none")

    return {
        "vector_width": {"max": max_width, "histogram": dict(sorted(widths.items()))},
        "rounding_modes": rounds,
        "reduction_tree": tree,
        "numeric_constants": dict(sorted(consts.items())),
        "reciprocal_form": {"form": recip, "n_div": n_div, "n_rcp": n_rcp},
    }

def _round_mode_name(mnem, imm):
    if imm is None: return "unknown"
    if mnem in ("vcvtps2ph","cvtps2ph"):
        if imm & 0x04: return "mxcsr"
        return {0:"nearest_even",1:"down",2:"up",3:"toward_zero"}.get(imm & 0x03, "unknown")
    # vroundps/ss imm: bit2 = MXCSR, bits1-0 = mode
    if imm & 0x04: return "mxcsr"
    return {0:"nearest_even",1:"down",2:"up",3:"toward_zero"}.get(imm & 0x03, "unknown")

def diff_signatures(sig_a, sig_b, name_a="ggml", name_b="ours"):
    """Structurally diff two kernel signatures, CLASSIFYING each divergence as
    CORRECTNESS-CRITICAL (implies bit-divergence) or PERF-ONLY (bit-identical but slower/different).

    Validated insight: vector-width and op-count can differ while the result is 0-ULP (e.g. ggml
    8-wide SIMD max vs our scalar max -- max is ORDER-INDEPENDENT). Only the rounding MODE VALUE,
    the reciprocal FORM, and the reduction-tree SET-OF-COMBINE-OPS (which encodes accumulation order)
    are correctness-critical. Vector width / op counts are perf parameters (relevant to the 1.000x
    perf goal, not bit-identity).
    """
    correctness, perf = [], []

    caveats = []
    # mxcsr resolves to nearest_even under the default MXCSR state -> normalize for comparison
    def _norm_modes(rm):
        return sorted(set(("nearest_even" if r["mode"] == "mxcsr" else r["mode"]) for r in rm))
    # --- CORRECTNESS axes ---
    # rounding MODE VALUES (mxcsr treated as nearest_even-equivalent under default rounding)
    ma = _norm_modes(sig_a["rounding_modes"]); mb = _norm_modes(sig_b["rounding_modes"])
    if ma != mb:
        correctness.append({"axis": "rounding_mode", name_a: ma, name_b: mb,
                            "note": "different rounding MODE -> bit-divergence"})
    elif any(r["mode"] == "mxcsr" for r in sig_a["rounding_modes"] + sig_b["rounding_modes"]):
        caveats.append("one side uses MXCSR-mode rounding (assumed default nearest_even); "
                       "verify MXCSR is not altered at runtime")
    # reciprocal form (true_divide vs approx_rcp vs none)
    if sig_a["reciprocal_form"]["form"] != sig_b["reciprocal_form"]["form"]:
        correctness.append({"axis": "reciprocal_form",
                            name_a: sig_a["reciprocal_form"]["form"], name_b: sig_b["reciprocal_form"]["form"],
                            "note": "true_divide vs approx_rcp vs different operand changes the result"})
    # reduction tree: a DIFFERENT set of combine-ops is a CAVEAT, not an automatic fail -- it only
    # matters for ORDER-DEPENDENT reductions (fp sum). For order-independent ops (max/min) a SIMD
    # tree vs scalar loop is bit-identical. The tool cannot know the op semantics statically, so it
    # flags the structural difference as a caveat for the human/dynamic-check to resolve.
    ta = sorted(set(sig_a["reduction_tree"])); tb = sorted(set(sig_b["reduction_tree"]))
    if ta != tb:
        caveats.append(f"reduction-tree shape differs ({name_a}={ta or 'scalar'}, {name_b}={tb or 'scalar'}): "
                       "correctness-critical ONLY for order-dependent reductions (fp sum); "
                       "benign for order-independent (max/min). Resolve with a dynamic component-check.")

    # --- PERF-ONLY axes (differ but typically bit-identical) ---
    if sig_a["vector_width"]["max"] != sig_b["vector_width"]["max"]:
        perf.append({"axis": "vector_width", name_a: sig_a["vector_width"]["max"],
                     name_b: sig_b["vector_width"]["max"], "note": "SIMD width (perf, order-independent ops still 0-ULP)"})
    na = len(sig_a["rounding_modes"]); nb = len(sig_b["rounding_modes"])
    if na != nb:
        perf.append({"axis": "rounding_op_count", name_a: na, name_b: nb,
                     "note": "count differs (SIMD vs scalar) -- benign if modes match"})

    verdict = ("BIT-IDENTICAL-COMPATIBLE (no correctness divergence)" if not correctness
               else f"CORRECTNESS DIVERGENT on {len(correctness)} axis/axes")
    return {"correctness_divergences": correctness, "perf_divergences": perf, "caveats": caveats,
            "n_correctness": len(correctness), "n_perf": len(perf), "verdict": verdict}


if __name__ == "__main__":
    if sys.argv[1] == "diff":
        # asm_facts.py diff <binA> <startA> <stopA> <binB> <startB> <stopB>
        ia = disasm(sys.argv[2], sys.argv[3], sys.argv[4])
        ib = disasm(sys.argv[5], sys.argv[6], sys.argv[7])
        print(json.dumps(diff_signatures(lift_signature(ia), lift_signature(ib)), indent=2))
        sys.exit(0)
    binary, start, stop, q = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    instrs = disasm(binary, start, stop)
    if q == "facts":
        for n,i in enumerate(instrs):
            print(f'instr({n}, {hex(i["addr"])}, {i["mnem"]}, {i["ops"]}).')
    elif q == "final_rounding":
        print(json.dumps(query_final_rounding(instrs), indent=2))
    elif q == "f16_conversion":
        print(json.dumps(query_f16_conversion(instrs), indent=2))
    elif q == "signature":
        print(json.dumps(lift_signature(instrs), indent=2))
    elif q == "accumulators":
        print(json.dumps(query_accumulator_structure(instrs), indent=2))
    else:
        print(json.dumps({"instrs": len(instrs)}, indent=2))
