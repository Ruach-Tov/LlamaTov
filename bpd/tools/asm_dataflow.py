#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""asm_dataflow.py v3 — LITERAL summation-tree extraction (mode=trace).
Beyond summary (v2), v3 emits the EXACT ordered sequence of accumulation events for a chosen
accumulator register: every (mulps src -> addps acc) in program order, with the resolved memory
operands. This is the literal summation TREE the kernel executes — a declarative fact lifted from
the binary, not imported from docs. Answers: in what exact order does THIS output sum its products?

Usage: asm_dataflow.py <bin> <start> <stop> [mode] [acc_reg]
  mode=summary (default) | trace (literal ordered events) | tree (acc-focused chain)
"""
import subprocess, re, sys, json

def lift(binary, start, stop):
    out=subprocess.run(["objdump","-d","--no-show-raw-insn",
        f"--start-address={start}",f"--stop-address={stop}",binary],capture_output=True,text=True).stdout
    chain=[]
    for line in out.splitlines():
        m=re.match(r"\s*([0-9a-f]+):\s+([a-z][a-z0-9.]*)\s*(.*)",line)
        if not m: continue
        ops=m.group(3).split("#")[0].strip()
        parts=[];depth=0;cur=""
        for ch in ops:
            if ch=="(":depth+=1
            if ch==")":depth-=1
            if ch==","and depth==0: parts.append(cur.strip());cur=""
            else: cur+=ch
        if cur.strip(): parts.append(cur.strip())
        chain.append({"addr":int(m.group(1),16),"mnem":m.group(2),"ops":parts})
    return chain

MUL={"mulps","mulss","vmulps","mulpd","mulsd","vfmadd231ps","vfmadd213ps"}
ADD={"addps","addss","vaddps","addpd","addsd"}
FMA={"vfmadd231ps","vfmadd213ps","vfmadd231ss"}
def is_reg(o): return o.startswith("%")
def memref(o):
    m=re.match(r"(-?0x[0-9a-f]+)?\((%\w+)(?:,(%\w+),(\d+))?\)",o)
    if not m: return o if not is_reg(o) else None
    d=int(m.group(1),16) if m.group(1) else 0
    return f"[{d:+#x}({m.group(2)}{','+m.group(3)+'*'+m.group(4) if m.group(3) else ''})]"

def provenance_str(reg, prov):
    """Render a register's current provenance as a readable expr."""
    p=prov.get(reg)
    if not p: return reg
    if p[0]=="load": return p[1]
    if p[0]=="mul": return f"({provenance_str('',{})or p[1]}*{p[2]})" if False else f"({p[1]}*{p[2]})"
    return reg

def trace_literal(chain, target_acc=None):
    """Emit the literal ordered accumulation events. If target_acc given, only that accumulator."""
    reg={}  # reg -> readable provenance string
    events=[]
    for ins in chain:
        m=ins["mnem"]; ops=ins["ops"]; addr=ins["addr"]
        if m in ("movaps","movups","movsd","movss","vmovaps","vmovups") and len(ops)==2:
            src,dst=ops
            if is_reg(dst): reg[dst]=memref(src) if not is_reg(src) else reg.get(src,src)
        elif m=="shufps":
            pass
        elif m in MUL and len(ops)>=2 and is_reg(ops[-1]):
            a=reg.get(ops[0],ops[0]); b=reg.get(ops[1],ops[1]) if len(ops)>2 else reg.get(ops[0],ops[0])
            if m in FMA:  # fma: dst = src1*src2 + dst (records as an accumulation directly)
                acc=ops[-1]
                if target_acc is None or acc==target_acc:
                    events.append({"addr":hex(addr),"op":"fma","acc":acc,"mul":f"{reg.get(ops[0],ops[0])}*{reg.get(ops[1],ops[1])}"})
                reg[acc]=f"({acc}+{reg.get(ops[0],ops[0])}*{reg.get(ops[1],ops[1])})"
            else:
                reg[ops[-1]]=f"({a}*{b})"
        elif m in ADD and len(ops)==2 and is_reg(ops[1]):
            acc=ops[1]; src=reg.get(ops[0],ops[0])
            if target_acc is None or acc==target_acc:
                events.append({"addr":hex(addr),"op":"add","acc":acc,"adds":src})
            reg[acc]=f"({acc}+{src})"
    return events


def query_pairing(chain):
    """ACCESS-GEOMETRY query: for each fp mul/sub/add that combines TWO memory-sourced operands,
    report the byte-offset (stride) between the two loads feeding it. The operand-pair stride IS
    a structural PARAMETER: it reveals pairing/layout/mode.
      adjacent (stride 4 = 1 f32)         -> interleaved / STANDARD pairing (2i,2i+1)
      half-width apart (e.g. 0x80=32 f32) -> split-half / NEOX pairing (i, i+n/2)
      other constant                      -> a fixed gather/interleave stride
    This is the fact that distinguishes rope NEOX vs standard from frame one (the mode = the stride).
    Tracks, per register, the (base, index, disp) of its last memory load; when two register-held
    loads feed one arithmetic op, emits the disp delta.
    """
    MUL={"mulps","mulss","vmulps","mulpd","mulsd","vmulss"}
    ADDSUB={"addps","addss","vaddps","subss","vsubss","subps","addpd","subsd","addsd"}
    reg={}  # reg -> ("mem", disp, base, index) of its current load
    pairs=[]
    def memref(o):
        m=re.match(r"(-?0x[0-9a-f]+)?\((%\w+)(?:,(%\w+),(\d+))?\)", o)
        if not m: return None
        return (int(m.group(1),16) if m.group(1) else 0, m.group(2), m.group(3))
    for ins in chain:
        m=ins["mnem"]; ops=ins["ops"]
        if m in ("movss","movaps","movups","vmovss","vmovups","vmovaps","movsd") and len(ops)==2 and ops[1].startswith("%"):
            mr = memref(ops[0]) if not ops[0].startswith("%") else None
            # track array accesses; prefer scaled-index (real arrays) but keep all for histogram
            if mr:
                reg[ops[1]] = ("mem",)+mr
            else:
                reg[ops[1]] = reg.get(ops[0]) if ops[0].startswith("%") else None
        elif (m in MUL or m in ADDSUB) and len(ops)>=2:
            srcs=[o for o in ops if o.startswith("%")]
            mems=[reg.get(o) for o in srcs if reg.get(o) and reg.get(o)[0]=="mem"]
            if len(mems)>=2:
                (d0,b0,i0),(d1,b1,i1) = mems[0][1:4], mems[1][1:4]
                if b0==b1 and i0==i1:  # same base+index → the delta is a pure operand-pair stride
                    stride=abs(d1-d0)
                    kind=("adjacent_standard" if stride==4 else
                          "split_half_neox" if stride>=64 else
                          f"stride_{stride}")
                    pairs.append({"addr":hex(ins["addr"]),"op":m,"byte_stride":stride,
                                  "floats_apart":stride//4,"pairing":kind})
    # dominant pairing: weight the ROTATION combine (add/sub of two same-base loads = the x0*c +/- x1*s
    # structure). Those carry the layout signal; multiply-only pairs are often coefficient setup.
    from collections import Counter
    rot=[p for p in pairs if p["op"] in ("subss","vsubss","addss","vaddss","subps","addps")]
    pool=rot if rot else pairs
    # ignore stride_0 (self/coefficient) for the layout verdict
    layout=[p for p in pool if p["byte_stride"]>0]
    c=Counter(p["pairing"] for p in layout)
    dominant=c.most_common(1)[0][0] if c else "none"
    verdict=("STANDARD (interleaved 2i,2i+1)" if dominant=="adjacent_standard" else
             "NEOX (split-half i,i+n/2)" if dominant=="split_half_neox" else
             f"custom ({dominant})" if dominant!="none" else "indeterminate")
    return {"pairs_found":len(pairs),"rotation_pairs":len(rot),
            "dominant_layout_pairing":dominant,"layout_verdict":verdict,
            "pairing_histogram":dict(Counter(p["pairing"] for p in pairs)),
            "rotation_samples":[p for p in pool if p["byte_stride"]>0][:6]}


def query_operand_frame(chain):
    """THE GATING QUESTION (phylogenetic root): for the fp compute ops, what is the OPERAND FRAME?
    This determines which downstream queries are even MEANINGFUL:
      memory_x_memory  -> query_pairing is DEFINED (relative memory stride is a real frame).
      memory_x_register-> broadcast/streaming pattern (one operand reused from a register).
      register_x_register -> REGISTER-BLOCKED (no memory frame; pairing is category-undefined).
    'Pairing' means how memory ACCESSES are paired by stride -- it only exists in the mem x mem frame.
    Returns the dominant frame + the unlocked/locked downstream queries.
    """
    MUL={"mulps","mulss","vmulps","mulpd","mulsd","vmulss"}
    ADDSUB={"addps","addss","vaddps","subss","vsubss","subps","addpd","subsd","addsd","vfmadd231ps","vfmadd213ps"}
    reg_is_mem={}  # reg -> True if its current value came from a memory load
    counts={"mem_x_mem":0,"mem_x_reg":0,"reg_x_reg":0}
    def is_memload(o): return ("(" in o) and not o.startswith("%")
    for ins in chain:
        m=ins["mnem"]; ops=ins["ops"]
        if m in ("movss","movaps","movups","vmovss","vmovups","vmovaps","movsd") and len(ops)==2 and ops[1].startswith("%"):
            reg_is_mem[ops[1]] = is_memload(ops[0]) or (ops[0].startswith("%") and reg_is_mem.get(ops[0],False))
        elif m=="shufps" or m=="vbroadcastss":
            if len(ops)>=2 and ops[-1].startswith("%"): reg_is_mem[ops[-1]]=False  # broadcast = register frame
        elif m in MUL or m in ADDSUB:
            srcs=[o for o in ops[:-1] if not o.startswith("@")]
            mem_direct=sum(1 for o in srcs if is_memload(o))
            reg_from_mem=sum(1 for o in srcs if o.startswith("%") and reg_is_mem.get(o,False))
            reg_pure=sum(1 for o in srcs if o.startswith("%") and not reg_is_mem.get(o,False))
            total_mem = mem_direct + reg_from_mem
            if total_mem>=2: counts["mem_x_mem"]+=1
            elif total_mem==1: counts["mem_x_reg"]+=1
            elif reg_pure>=2: counts["reg_x_reg"]+=1
    total=sum(counts.values()) or 1
    dominant=max(counts,key=counts.get)
    unlocks={"mem_x_mem":["query_pairing (memory stride frame is DEFINED)"],
             "mem_x_reg":["broadcast/streaming analysis"],
             "reg_x_reg":["register-tile/accumulator analysis (query_pairing UNDEFINED here)"]}
    return {"operand_frame_counts":counts,"dominant_frame":dominant,
            "fraction":round(counts[dominant]/total,3),
            "pairing_query_meaningful": dominant=="mem_x_mem",
            "unlocks":unlocks[dominant]}

if __name__=="__main__":
    b,s,e=sys.argv[1],sys.argv[2],sys.argv[3]
    mode=sys.argv[4] if len(sys.argv)>4 else "trace"
    acc=sys.argv[5] if len(sys.argv)>5 else None
    chain=lift(b,s,e)
    if mode=="frame":
        import json as _j; print(_j.dumps(query_operand_frame(chain),indent=2)); raise SystemExit
    if mode=="pairing":
        import json as _j; print(_j.dumps(query_pairing(chain),indent=2)); raise SystemExit
    ev=trace_literal(chain, acc)
    if mode=="trace":
        for e in ev: print(f'{e["addr"]} {e["op"]:4s} {e["acc"]:8s} <- {e.get("adds") or e.get("mul")}')
        print(f"# {len(ev)} accumulation events" + (f" into {acc}" if acc else " (all accumulators)"))
    else:
        print(json.dumps(ev,indent=2))
