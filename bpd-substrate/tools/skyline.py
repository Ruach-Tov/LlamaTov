#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""skyline.py — roofline/"skyline" + e2e profile: actual kernel performance vs ceiling.

Layer 2 of the LlamaTov profiling capability. Consumes per-kernel measured metrics
(from mavchin's CUPTI GPU PMU bridge) and reports:
  - each kernel's ACHIEVED perf vs the hardware ROOF (the "skyline")
  - the bound classification (occupancy / bandwidth / coalescing / load-width)
  - the load-instruction breakdown (gld_8/16/32/64/128bit) — exposes narrow-load pattern
  - the e2e PROFILE: per-op time x layer-count, % of decode time (where time goes)

Stack: e2e_bench.py (tok/sec) + skyline.py (per-kernel roofline + profile) +
       mavchin's CUPTI bridge (emits the JSON) + referee (bit-identity gate).

Input JSON: list of per-kernel dicts. Recognized fields:
  name, M, K, ms, gbps (measured), bytes, occupancy_pct, sectors_per_load,
  fb_read_mb, gld_8bit/16bit/32bit/64bit/128bit, count (launches/forward, default per-layer)
Usage:
  skyline.py --metrics forward_profile.json [--device p4] [--layers 16] [--baseline b.json]
"""
import argparse, json, sys

DEVICES = {
    "p4": {
        "name": "Tesla P4 (GP104, sm_61)",
        "dram_bw_spec_gbs": 192.0, "dram_bw_measured_gbs": 147.0,
        "fp32_tflops": 5.6, "int8_dp4a_tops": 22.0,
        "sms": 20, "l2_mb": 2.0, "warp": 32,
    },
}

def gbps_of(k):
    if k.get("gbps"): return k["gbps"]
    if k.get("bytes") and k.get("ms"): return k["bytes"]/(k["ms"]/1e3)/1e9
    return 0.0

def total_loads(k):
    return sum(k.get(f"gld_{w}bit", 0) for w in (8,16,32,64,128))

def classify(k, dev):
    occ = k.get("occupancy_pct", 0); spl = k.get("sectors_per_load", 0)
    pct_meas = 100*gbps_of(k)/dev["dram_bw_measured_gbs"] if gbps_of(k) else 0
    g128 = k.get("gld_128bit", 0); gnarrow = k.get("gld_16bit",0)+k.get("gld_32bit",0)
    if occ and occ < 50: return "OCCUPANCY"
    if pct_meas > 90:    return "BANDWIDTH(roof)"
    if spl and spl > 1.5: return "COALESCING"
    if gnarrow > 0 and g128 == 0: return "LOAD-WIDTH(narrow)"   # the SoA-vec128 opportunity
    return "latency/other"

def analyze(kernels, dev, layers, label="profile"):
    print(f"=== SKYLINE [{label}] on {dev['name']} ===")
    print(f"    DRAM roof {dev['dram_bw_spec_gbs']:.0f} spec / {dev['dram_bw_measured_gbs']:.0f} measured GB/s"
          f" | {dev['fp32_tflops']:.1f} TFLOP fp32 / {dev['int8_dp4a_tops']:.0f} TOP int8-dp4a | {layers} layers")
    print()
    hdr = (f"  {'kernel':<10}{'MxK':>11}{'ms':>7}{'GB/s':>7}{'%meas':>6}{'occ%':>6}"
           f"{'sec/ld':>7}{'128b?':>6}  bound")
    print(hdr); print("  " + "-"*(len(hdr)-2))
    total_time = 0.0
    for k in kernels:
        gbps = gbps_of(k); pm = 100*gbps/dev["dram_bw_measured_gbs"] if gbps else 0
        mxk = f"{k.get('M','?')}x{k.get('K','?')}"
        has128 = "yes" if k.get("gld_128bit",0)>0 else "NO"
        cnt = k.get("count", layers)
        total_time += k["ms"]*cnt
        print(f"  {k['name']:<10}{mxk:>11}{k['ms']:>7.3f}{gbps:>7.0f}{pm:>5.0f}%{k.get('occupancy_pct',0):>5.0f}%"
              f"{k.get('sectors_per_load',0):>7.2f}{has128:>6}  {classify(k,dev)}")
    print()
    # e2e PROFILE: where the decode time goes (per-op x layer count)
    print(f"  --- e2e PROFILE (per-op time x {layers} layers) ---")
    rows = sorted(kernels, key=lambda k: -k["ms"]*k.get("count",layers))
    for k in rows:
        t = k["ms"]*k.get("count",layers)
        print(f"    {k['name']:<10} {t:>7.3f} ms  ({100*t/total_time:>4.1f}% of matmul time)")
    print(f"    {'TOTAL matmul':<10} {total_time:>7.3f} ms/forward")
    print()
    # the load-width opportunity summary
    narrow = [k for k in kernels if k.get('gld_128bit',0)==0 and (k.get('gld_16bit',0)+k.get('gld_32bit',0))>0]
    if narrow:
        print(f"  LOAD-WIDTH OPPORTUNITY: {len(narrow)}/{len(kernels)} kernels use ZERO 128-bit loads")
        print(f"    (narrow 16/32-bit only -> SoA-vec128 unlocks wider loads, bit-identical)")
    return total_time

def main():
    ap = argparse.ArgumentParser(description="roofline/skyline + e2e profile per kernel")
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--baseline", help="optional baseline profile JSON to compare against")
    ap.add_argument("--device", default="p4")
    ap.add_argument("--layers", type=int, default=16)
    a = ap.parse_args()
    dev = DEVICES[a.device]
    def load(p):
        d = json.load(open(p)); return d.get("kernels", d) if isinstance(d, dict) else d
    var = load(a.metrics)
    if a.baseline:
        bt = analyze(load(a.baseline), dev, a.layers, "BASELINE")
        print("="*70)
        vt = analyze(var, dev, a.layers, "VARIANT")
        print(f"\n  MEASURED matmul-time ratio (baseline/variant): {bt/vt:.4f}x"
              f"  ({bt:.2f} -> {vt:.2f} ms/forward)")
    else:
        analyze(var, dev, a.layers, "profile")

if __name__ == "__main__":
    main()
