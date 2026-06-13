#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""asm_loops.py — loop-structure query over the p-code chain. Extends asm_facts toward
detecting REDUCTION-ORDER parameters (like sgemm's kc-panel block size kappa) structurally.

A loop = a backward jump (target address < jump address, within the function). For each loop
we report: header addr, the back-edge, body length, and any constant compared against the
induction variable (cmp reg, imm  /  add reg, imm) — which reveals trip counts / block sizes.

The kc-panel block (kappa) shows up as: the increment of the K-loop counter, OR the immediate
loaded as the inner-accumulation trip count before a partial-sum store. So 'what is kappa' becomes
a structural query: find the reduction loop nest, read its block-size constant.
"""
import subprocess, re, sys, json

def lift(binary, start, stop):
    out = subprocess.run(["objdump","-d","--no-show-raw-insn",
                          f"--start-address={start}", f"--stop-address={stop}", binary],
                         capture_output=True, text=True).stdout
    chain = []
    for line in out.splitlines():
        m = re.match(r"\s*([0-9a-f]+):\s+([a-z][a-z0-9.]*)\s*(.*)", line)
        if not m: continue
        addr = int(m.group(1),16); mnem = m.group(2); ops = m.group(3).split("#")[0].strip()
        # jump target (if any)
        tgt = None
        jm = re.search(r"\b([0-9a-f]+)\s*<", ops) or re.match(r"^([0-9a-f]+)$", ops)
        if mnem.startswith("j") and jm:
            try: tgt = int(jm.group(1),16)
            except: pass
        chain.append({"addr":addr,"mnem":mnem,"ops":ops,"tgt":tgt})
    return chain

def find_loops(chain):
    """Back-edges: a jump whose target is an EARLIER address in range = a loop back-edge."""
    loops = []
    for ins in chain:
        if ins["mnem"].startswith("j") and ins["tgt"] is not None and ins["tgt"] < ins["addr"]:
            header = ins["tgt"]; back = ins["addr"]
            body = [c for c in chain if header <= c["addr"] <= back]
            # constants compared/added on the induction var within the body
            consts = []
            for b in body:
                cm = re.search(r"\$0x([0-9a-f]+)", b["ops"])
                if b["mnem"] in ("cmp","add","sub","lea") and cm:
                    consts.append({"insn":f'{b["mnem"]} {b["ops"]}', "imm":int(cm.group(1),16)})
            # SIMD accumulation in the body? (the reduction signature)
            packed = sum(1 for b in body if b["mnem"] in ("addps","vaddps","mulps","vmulps","vfmadd231ps","vfmadd213ps","fmaddps"))
            loops.append({"header":hex(header),"back_edge":hex(back),"body_insns":len(body),
                          "simd_fma_or_add":packed,"constants":consts[:8]})
    return loops

if __name__ == "__main__":
    binary, start, stop = sys.argv[1], sys.argv[2], sys.argv[3]
    chain = lift(binary, start, stop)
    loops = find_loops(chain)
    print(json.dumps({"total_insns":len(chain),"loops_found":len(loops),"loops":loops}, indent=2))
