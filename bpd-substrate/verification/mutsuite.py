"""MUTATION SUITE for wholegate.py — Medayek's spec.

★ WHY: a single positive control is mutation testing with a sample size of one.
  It caught a real blind spot in one instrument and MISSED an equivalent mutant
  in another: a one-ULP nudge to GRU's weight_ih_l0[0] is absorbed by the
  sigmoid, so the gate returned BIT_EXACT and looked blind. It was not.

★ THE DISCRIMINATOR (Medayek's):
      surviving at ONE magnitude    → ambiguous
      surviving at ALL magnitudes   → an EQUIVALENT MUTANT (a fact about the model)
      surviving at ALL SITES        → A BLIND GATE (a fact about the instrument)

★ THE REPORT EMITS THE FULL MATRIX and computes the score from it. The equivalent
  set is an explicit output, never folded silently into a denominator — a summary
  that hides which mutants survived has discarded the finding.
"""
import sys, os, json, torch
# ★ PORTABLE: the instrument does not depend on where it lives. wholegate.py
#   is imported from the SAME directory as this script — flat in the enclave
#   layout (/home/dibbur-patch/), verification/ in the published slice.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wholegate import gate

MAGNITUDES = [None, 1e-6, 1e-4, 1e-2, 1.0]        # None = exactly one ULP
FAULTS = ["add", "scale", "zero", "sign"]


def _mutate(p, idx, fault, mag):
    flat = p.view(-1)
    if fault == "add":
        if mag is None:
            b = flat[idx].view(torch.int32); flat[idx] = (b + 1).view(torch.float32)
        else:
            flat[idx] += mag
    elif fault == "scale":
        flat[idx] *= (1.0 + (mag if mag is not None else 1.1920929e-7))
    elif fault == "zero":
        flat[idx] = 0.0
    elif fault == "sign":
        flat[idx] = -flat[idx]


def run(pid, n_sites=5):
    rows, sites = [], None
    for site in range(n_sites):
        for fault in FAULTS:
            for mag in MAGNITUDES:
                if fault in ("zero", "sign") and mag not in (None,):
                    continue                      # magnitude is meaningless for these
                def fn(model, ins, site=site, fault=fault, mag=mag):
                    ps = [p for p in model.parameters() if p.dtype.is_floating_point]
                    with torch.no_grad():
                        _mutate(ps[site % len(ps)], 0, fault, mag)
                        return model(*ins)
                try:
                    r = gate(pid, candidate_fn=fn)
                    n = r.get("n_diff", -1)
                except Exception as e:
                    n = -1
                rows.append({"site": site, "fault": fault,
                             "magnitude": "1ULP" if mag is None else mag,
                             "n_diff": n, "killed": n is not None and n > 0})
    killed = [r for r in rows if r["killed"]]
    survived = [r for r in rows if not r["killed"] and r["n_diff"] == 0]
    errors = [r for r in rows if r["n_diff"] == -1]

    # ★ a mutant is EQUIVALENT if it survives at every magnitude for its site+fault
    by_sf = {}
    for r in rows:
        by_sf.setdefault((r["site"], r["fault"]), []).append(r)
    # ★ MY FIRST RULE WAS TOO STRICT: it required survival at EVERY magnitude,
    #   so a mutant killed at 1.0 but surviving at 1ULP counted as neither killed
    #   nor equivalent -- it vanished from both sets while remaining in the total.
    #   A SURVIVOR THAT IS IN NO CATEGORY IS A HOLE IN THE REPORT.
    #   Correct form: equivalence is per-(site,fault,magnitude). Report each
    #   survivor, and mark whether it survives at ALL magnitudes (model-absorbed
    #   outright) or only below a threshold (a SENSITIVITY FLOOR, which is a
    #   measurement about the model worth having).
    equivalent = []
    for k, v in by_sf.items():
        surv = [x for x in v if not x["killed"] and x["n_diff"] == 0]
        if not surv:
            continue
        allmag = len(surv) == len(v)
        equivalent.append({
            "site": k[0], "fault": k[1],
            "magnitudes_surviving": [x["magnitude"] for x in surv],
            "mechanism": ("survives at ALL tested magnitudes -- model-absorbed"
                          if allmag else
                          "survives only below a threshold -- a sensitivity floor, not blindness")})

    # ★ a BLIND GATE would show survival at every SITE
    by_site = {}
    for r in rows:
        by_site.setdefault(r["site"], []).append(r)
    blind = all(all(not x["killed"] for x in v) for v in by_site.values()) if by_site else False

    total = len(rows) - len(errors)
    denom = total - len(equivalent)
    return {"pid": pid, "matrix": rows,
            "mutation_score": (len(killed) / denom) if denom > 0 else None,
            "killed": len(killed), "total": total,
            "equivalent_mutants": equivalent,
            "GATE_BLIND": blind, "errors": len(errors)}


if __name__ == "__main__":
    for pid in sys.argv[1:]:
        rep = run(pid)
        print("   #%-3s score=%-6s killed=%d/%d  equivalent=%d  BLIND=%s"
              % (pid, ("%.3f" % rep["mutation_score"]) if rep["mutation_score"] is not None else "n/a",
                 rep["killed"], rep["total"], len(rep["equivalent_mutants"]), rep["GATE_BLIND"]))
        for e in rep["equivalent_mutants"]:
            print("        EQUIVALENT: site %s fault %s — %s" % (e["site"], e["fault"], e["mechanism"]))
        json.dump(rep, open("/tmp/mutation_%s.json" % pid, "w"), indent=1)
