#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""asm_transcendental.py v2 — extract transcendental/dispatch facts as PROLOG FACTS.

Generalizes from "classify one range" to "lift the transcendental dispatch structure of a kernel
into declarative facts" that compose with the rest of the substrate:

  transcendental_call(KernelSym, CalleeSym, Kind).   % Kind = libm | sleef | other
  poly_approx(KernelSym, ConstLoads, MulAddOps, DegreeEst).
  dispatch(KernelSym, IndirectVia).                  % indirect call (vtable / fn-ptr dispatch)
  vector_variant(BaseFn, Width, ISA).                % Sleef_erff8_u10avx -> base erff, width 8, avx
  signature(KernelSym, CALL|POLYNOMIAL|ARITHMETIC_ONLY|DISPATCH|NONE).

It scans the FULL function (resolving the symbol's extent), recognizes SLEEF naming
(Sleef_<fn><width>_<acc><isa>), and emits facts a Prolog layer can query + join with
ir_param_axes (e.g. reference_pins(pytorch_gelu, erf_impl, sleef_erff_u10)).
"""
import subprocess, re, sys, json

LIBM = {"tanhf","tanh","expf","exp","erff","erf","logf","log","sinf","cosf","sinhf","coshf",
        "expm1f","log1pf","powf","sqrtf","cbrtf","atanf","asinf","acosf","exp2f","log2f"}

def syms(binary):
    out=subprocess.run(["nm",binary],capture_output=True,text=True).stdout
    d={}
    for line in out.splitlines():
        m=re.match(r"([0-9a-f]+)\s+[a-zA-Z]\s+(\S+)",line)
        if m: d[int(m.group(1),16)]=m.group(2)
    return d

def func_extent(symtab, start):
    addrs=sorted(symtab)
    nxt=[a for a in addrs if a>start]
    return nxt[0] if nxt else start+0x2000

def disasm(binary,start,stop):
    return subprocess.run(["objdump","-d",f"--start-address={start}",f"--stop-address={stop}",binary],
                          capture_output=True,text=True).stdout

def demangle_sleef(sym):
    # Sleef_<finz_|cinz_>?<fn><width>_<acc><isa>  e.g. Sleef_erff8_u10avx2
    m=re.match(r"Sleef_(?:finz_|cinz_)?([a-z]+?)(\d+)_(u\d+)([a-z0-9]*)",sym)
    if not m: return None
    return {"base":m.group(1),"width":int(m.group(2)),"acc":m.group(3),"isa":m.group(4) or "scalar"}

def extract_facts(binary, kernel_sym, start, stop):
    asm=disasm(binary,start,stop)
    facts=[]; calls=[]; indirect=0; const_loads=0; fma=0; sleef=[]
    for line in asm.splitlines():
        m=re.search(r"call\s+[0-9a-f]+\s+<([^@>]+)(@plt)?>",line)
        if m:
            callee=m.group(1); base=re.sub(r"^__","",callee)
            sl=demangle_sleef(callee)
            if sl:
                kind="sleef"; sleef.append((callee,sl))
            elif base in LIBM: kind="libm"
            else: kind="other"
            if kind in ("sleef","libm"): calls.append((callee,kind))
        if re.search(r"call\s+\*",line): indirect+=1
        if re.search(r"(movss|movaps|movups|vmovss|vbroadcastss|vmovaps)\s+.*\(%rip\)",line): const_loads+=1
        if re.search(r"\b(mulss|addss|mulps|addps|vmulps|vaddps|vfmadd\w+|vmulss)\b",line): fma+=1
    k=re.sub(r"[^A-Za-z0-9_]","_",kernel_sym)[:40]
    for callee,kind in sorted(set(calls)):
        c=re.sub(r"[^A-Za-z0-9_]","_",callee)
        facts.append(f"transcendental_call('{k}', '{c}', {kind}).")
        sl=demangle_sleef(callee)
        if sl:
            facts.append(f"vector_variant('{c}', {sl['base']}, {sl['width']}, {sl['acc']}, {sl['isa']}).")
    if indirect: facts.append(f"dispatch('{k}', indirect, {indirect}).")
    if const_loads>=2 and fma>=4 and not calls:
        facts.append(f"poly_approx('{k}', {const_loads}, {fma}, {fma//2}).")
    # signature
    if calls: sig="CALL"
    elif indirect: sig="DISPATCH"
    elif const_loads>=2 and fma>=4: sig="POLYNOMIAL"
    elif fma>0: sig="ARITHMETIC_ONLY"
    else: sig="NONE"
    facts.append(f"signature('{k}', {sig}).")
    return facts

if __name__=="__main__":
    binary=sys.argv[1]; start=int(sys.argv[2],16)
    st=syms(binary); kernel_sym=st.get(start, f"sub_{start:x}")
    stop=int(sys.argv[3],16) if len(sys.argv)>3 else func_extent(st,start)
    mode=sys.argv[4] if len(sys.argv)>4 else "prolog"
    facts=extract_facts(binary,kernel_sym,start,stop)
    if mode=="prolog":
        for f in facts: print(f)
    else:
        print(json.dumps({"kernel":kernel_sym,"facts":facts},indent=2))
