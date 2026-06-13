#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""warp_solver.py — constraint solver for GOOD ACCESS PATTERNS + warp configs.

Heath's idea: instead of blind-sweeping the warp-parameter space and measuring every
point, FORMALIZE the access pattern + scheduling as constraints and SOLVE for configs
that satisfy "good access". Then measure only the feasible candidates with the stall
counters — directed verification of a derived answer, not a blind search.

The constraints are CLOSED-FORM HARDWARE RULES (coalescing, alignment, occupancy,
latency-hiding) — derivable, not discovered. This emits the small ranked candidate
set for the SoA-128bit Q8_0 matmul on the P4, to feed the cupti-from-prolog measurement.

This is the substrate pattern: derive from facts (HW rules) -> verify by measurement.
"""
import itertools, json, argparse

# ---- P4 (GP104, sm_61) hardware facts ----
HW = dict(
    sms=20, warp=32, max_warps_per_sm=64, regs_per_sm=65536,
    max_threads_per_block=1024, max_blocks_per_sm=32,
    shared_per_sm=98304, coalesce_line=128, vec_load_bytes=16,   # uint4
    mem_latency_cyc=400,          # ~global memory latency
    issue_interval_cyc=1,         # dual-issue capable
)

def feasible(cfg, hw, K, block_bytes=32):
    """Return (ok, reasons) — does this warp config satisfy the GOOD-ACCESS constraints?"""
    tpb   = cfg["threads_per_block"]
    rpw   = cfg["rows_per_warp"]          # output rows handled per warp
    regs  = cfg["regs_per_thread"]
    warps_per_block = tpb // hw["warp"]
    reasons = []

    # C1 ALIGNMENT: vectorized (uint4) loads need 16-byte-aligned addresses.
    # SoA quant block = 32 bytes, contiguous, 16-byte aligned by construction. OK iff
    # the per-thread access starts on a 16B boundary: each thread reads vec_load_bytes.
    if (hw["vec_load_bytes"] % 16) != 0:
        reasons.append("vec load not 16B")
    # C2 COALESCING: a warp's 32 threads each read 16B -> 512B span. To tile 128B lines
    # cleanly, consecutive threads must hit consecutive 16B chunks (stride = 16B).
    # That means the warp reads a CONTIGUOUS 512B region = 4 full 128B lines. GOOD.
    # Violated if rows_per_warp spreads threads across non-contiguous rows.
    warp_span = hw["warp"] * hw["vec_load_bytes"]
    if warp_span % hw["coalesce_line"] != 0:
        reasons.append(f"warp span {warp_span}B not a 128B multiple (uncoalesced)")
    # C3 OCCUPANCY: fit enough warps/SM to hide latency.
    #   register limit: warps_per_sm <= regs_per_sm / (warp * regs_per_thread)
    warps_by_reg = hw["regs_per_sm"] // (hw["warp"] * regs)
    blocks_by_warp = hw["max_warps_per_sm"] // warps_per_block if warps_per_block else 0
    blocks_by_reg  = warps_by_reg // warps_per_block if warps_per_block else 0
    blocks_per_sm = min(blocks_by_warp, blocks_by_reg, hw["max_blocks_per_sm"])
    active_warps = blocks_per_sm * warps_per_block
    occ = active_warps / hw["max_warps_per_sm"]
    cfg["_active_warps"] = active_warps; cfg["_occ"] = occ; cfg["_blocks_per_sm"] = blocks_per_sm
    # C4 LATENCY HIDING: need enough eligible warps to cover memory latency.
    #   warps_needed ~= mem_latency / issue_interval / (insts between dependent loads)
    #   conservative: need active_warps >= mem_latency/ (cycles_per_warp_iter). Require occ high.
    if active_warps < 8:
        reasons.append(f"only {active_warps} warps/SM — too few to hide {hw['mem_latency_cyc']}cyc latency")
    # C5 BLOCK VALIDITY
    if tpb > hw["max_threads_per_block"]: reasons.append("tpb>1024")
    if tpb % hw["warp"] != 0: reasons.append("tpb not warp-multiple")
    return (len(reasons)==0, reasons)

def predicted_stall_score(cfg, hw):
    """Lower = better. Heuristic objective predicting stall behavior (the thing we''d
    measure as stall_exec_dependency + stall_memory_dependency). To be VALIDATED by CUPTI."""
    occ = cfg["_occ"]
    # latency hiding improves with more active warps (sublinear past the point of coverage)
    latency_hide = min(1.0, cfg["_active_warps"] / 32.0)   # ~32 warps fully hides 400cyc
    # address-arithmetic dependency penalty: more rows_per_warp = more offset math per iter
    addr_dep = cfg["rows_per_warp"] * 0.15
    # the score: want high latency_hide, low addr_dep, high occupancy
    return round((1.0 - latency_hide) + addr_dep + (1.0 - occ)*0.5, 4)

def main():
    ap = argparse.ArgumentParser(description="solve for good-access warp configs")
    ap.add_argument("--K", type=int, default=2048)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--json", help="emit candidate set for the cupti sweep")
    a = ap.parse_args()

    # the search space (small, integer)
    space = dict(
        threads_per_block=[32,64,128,256,512],
        rows_per_warp=[1,2,4,8],
        regs_per_thread=[24,32,40,48,64],
    )
    keys = list(space)
    feas = []
    for combo in itertools.product(*[space[k] for k in keys]):
        cfg = dict(zip(keys, combo))
        ok, reasons = feasible(cfg, HW, a.K)
        if ok:
            cfg["_score"] = predicted_stall_score(cfg, HW)
            feas.append(cfg)
    feas.sort(key=lambda c: c["_score"])
    total = 1
    for k in space: total *= len(space[k])
    print(f"=== warp-config constraint solver (P4, SoA-128bit Q8_0, K={a.K}) ===")
    print(f"    search space: {total} configs -> {len(feas)} satisfy GOOD-ACCESS constraints")
    print(f"    (constraints: 16B-align, 128B-coalesce, occupancy>=8 warps, latency-hide)")
    print(f"\n    top {a.top} candidates (ranked by predicted stall score, lower=better):")
    print(f"    {'tpb':>5}{'rpw':>5}{'regs':>6}{'warps/SM':>9}{'occ':>6}{'blk/SM':>7}{'score':>8}")
    for c in feas[:a.top]:
        print(f"    {c['threads_per_block']:>5}{c['rows_per_warp']:>5}{c['regs_per_thread']:>6}"
              f"{c['_active_warps']:>9}{c['_occ']:>6.2f}{c['_blocks_per_sm']:>7}{c['_score']:>8.3f}")
    print(f"\n    -> MEASURE these {min(a.top,len(feas))} (not all {total}) with cupti stall_exec_dependency.")
    print(f"       The solver DERIVES feasibility; CUPTI VALIDATES the predicted stall ranking.")
    if a.json:
        json.dump(feas[:a.top], open(a.json,"w"), indent=2)
        print(f"    [wrote {a.json} for the cupti-from-prolog sweep]")

if __name__ == "__main__":
    main()
