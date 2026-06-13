#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""gemm_diagram.py — render a GEMM kernel as the canonical GotoBLAS/BLIS panel picture,
annotated with OUR coordinate system: the reduction-locked K axis vs free M/N axes,
and the tile params (mc, nc, kc, MR, NR) that we've been hunting for 0-ULP performance.

This visualizes the central finding: C[mc x nc] += A_panel[mc x kc] * B_panel[kc x nc],
where the kc K-blocking is the REDUCTION-ORDER INVARIANT (must match the reference)
and mc/nc/MR/NR are the FREE schedule axes (sweep for speed, stays 0-ULP).
"""
import sys, os

def rect(x,y,w,h,fill,stroke="#333",sw=1.2,dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'
def txt(x,y,t,size=12,anchor="middle",color="#222",weight="normal",style=""):
    return f'<text x="{x}" y="{y}" font-family="monospace" font-size="{size}" text-anchor="{anchor}" fill="{color}" font-weight="{weight}" {style}>{t}</text>'
def line(x1,y1,x2,y2,c="#555",w=1.5,dash="",marker=""):
    d=f' stroke-dasharray="{dash}"' if dash else ""; m=f' marker-end="url(#ah)"' if marker else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="{w}"{d}{m}/>'

def gemm_panel(name="bpd_gemm  (C = A·B)", mc=4, nc=6, kc=5, MR=2, NR=2, W=720, H=560):
    s=['<defs><marker id="ah" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#c44"/></marker></defs>']
    s.append(rect(0,0,W,H,"#fbfcfe","#ccd",1))
    s.append(txt(W/2,26,name,15,weight="bold"))
    s.append(txt(W/2,46,"GotoBLAS/BLIS panel decomposition  \u2014  reduction-order = K-axis (LOCKED), M/N = free schedule",11,color="#777"))
    cell=30
    # layout: C (top-right), A (left), B (top), arranged as the classic block-matmul picture
    Ax,Ay = 90, 300          # A panel: mc rows x kc cols
    Bx,By = 90+kc*cell+50, 90  # B panel: kc rows x nc cols (above-right)
    Cx,Cy = Bx, 300           # C tile: mc rows x nc cols (right of A, below B)
    # --- A panel (mc x kc) : the streaming A-panel ---
    for r in range(mc):
        for c in range(kc):
            s.append(rect(Ax+c*cell, Ay+r*cell, cell, cell, "#dde7f5","#2c4a7c"))
    s.append(txt(Ax+kc*cell/2, Ay+mc*cell+22, f"A panel  (mc={mc} \u00d7 kc={kc})", 11, color="#2c4a7c", weight="bold"))
    # --- B panel (kc x nc) : the packed B-panel ---
    for r in range(kc):
        for c in range(nc):
            s.append(rect(Bx+c*cell, By+r*cell, cell, cell, "#e6f0dd","#3a7a3a"))
    s.append(txt(Bx+nc*cell/2, By-12, f"B panel  (kc={kc} \u00d7 nc={nc})", 11, color="#3a7a3a", weight="bold"))
    # --- C tile (mc x nc) : the output, with an MRxNR microtile highlighted ---
    for r in range(mc):
        for c in range(nc):
            s.append(rect(Cx+c*cell, Cy+r*cell, cell, cell, "#fbeede","#9c5a14"))
    # highlight one MRxNR register microtile
    s.append(rect(Cx, Cy, NR*cell, MR*cell, "none","#e8923c",3))
    s.append(txt(Cx+nc*cell/2, Cy+mc*cell+22, f"C tile  (mc={mc} \u00d7 nc={nc})", 11, color="#9c5a14", weight="bold"))
    s.append(txt(Cx+NR*cell/2, Cy+MR*cell/2+4, f"{MR}\u00d7{NR}", 10, color="#e8923c", weight="bold"))
    # --- the K reduction arrow (LOCKED axis) : A cols & B rows contract over kc ---
    midy = Ay+mc*cell+44
    s.append(line(Ax, midy, Ax+kc*cell, midy, "#c44", 2.5, marker="1"))
    s.append(txt(Ax+kc*cell/2, midy+18, "K reduction (kc-blocked)  \u2014  ORDER LOCKED to reference \u26a0", 10.5, color="#c44", weight="bold"))
    # --- free-axis annotations: M (down A), N (across B) ---
    s.append(line(Ax-22, Ay, Ax-22, Ay+mc*cell, "#2a8", 2, marker="1"))
    s.append(txt(Ax-30, Ay+mc*cell/2, "M", 12, anchor="end", color="#2a8", weight="bold"))
    s.append(line(Bx, By-26, Bx+nc*cell, By-26, "#2a8", 2, marker="1"))
    s.append(txt(Bx+nc*cell/2, By-34, "N  (M,N = FREE: tile / vectorize / pack \u2713 0-ULP)", 10.5, color="#2a8", weight="bold"))
    # legend
    ly=H-26
    s.append(txt(20, ly-14, "Each C[i,j] = \u03a3_k A[i,k]\u00b7B[k,j]  accumulated in kc-blocks (block order = the 0-ULP invariant).", 10.5, anchor="start", color="#555"))
    s.append(txt(20, ly, "Sweep mc/nc/MR/NR + packing for speed WITHOUT changing kc grouping \u2192 match-then-beat at 0 ULP.", 10.5, anchor="start", color="#555"))
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'+"".join(s)+"</svg>"

if __name__=="__main__":
    outdir = sys.argv[1] if len(sys.argv)>1 else "/tmp/output-only"
    os.makedirs(outdir,exist_ok=True)
    # use our real OpenBLAS Sandybridge-ish params (scaled down for legibility): kc grouping locked
    svg = gemm_panel("bpd_gemm  (C = A\u00b7B)  \u2014  OpenBLAS-match kc-block", mc=4,nc=6,kc=5,MR=2,NR=2)
    open(f"{outdir}/gemm_panel.o.svg","w").write(svg)
    print(f"wrote {outdir}/gemm_panel.o.svg")
