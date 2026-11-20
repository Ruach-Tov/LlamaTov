"""STATIC SEMANTIC AUDIT: diff the emitted constants against the problem's own values.

★ INPUT-INDEPENDENT. A wrongly-clamped kernel whose inputs never exceed the bound
  passes a bit-exact gate honestly and is still wrong -- measured: 0.0000% of one
  problem's values exceed +/-1, so both the wrong and the right clamp agree.

  This reads the CONSTANTS out of the emitted CUDA and the VALUES out of the
  problem source, and compares them. No inputs involved, so no input coverage
  question arises.
"""
import glob, os, re
KB = "/home/dibbur-patch/kb_level2"

# constants a problem declares at module scope, e.g. hardtanh_min = -2
DECL = re.compile(r"^([a-z_][a-z_0-9]*)\s*=\s*(-?\d+\.?\d*)\s*$", re.M)
# float literals in the emitted kernel body
LIT  = re.compile(r"(-?\d+\.\d+)f")

INTERESTING = ("min", "max", "slope", "divisor", "scale", "eps", "value", "factor", "constant")

rows = []
for f in sorted(glob.glob("/home/dibbur-patch/emitted/*.cu")):
    pid = os.path.basename(f)[3:-3]
    kbf = glob.glob(os.path.join(KB, "%s_*.py" % pid))
    if not kbf: continue
    src = open(kbf[0]).read()
    decls = {k: float(v) for k, v in DECL.findall(src)
             if any(t in k for t in INTERESTING)}
    if not decls: continue
    body = open(f).read()
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)      # drop the manifest header
    lits = set(float(x) for x in LIT.findall(body))
    # ★ A DIVISOR IS CORRECTLY EMITTED AS ITS RECIPROCAL -- divisor 2.0 -> 0.5f is
    #   RIGHT, not missing. Accept the value, its negation, and its reciprocal;
    #   otherwise the audit drowns in false positives and gets ignored.
    def present(v):
        for cand in (v, -v, (1.0/v if v else None)):
            if cand is None: continue
            if any(abs(cand - l) < 1e-9 for l in lits): return True
        return False
    missing = {k: v for k, v in decls.items() if not present(v)}
    if missing:
        rows.append((pid, missing, sorted(lits)[:6]))

print("   PROBLEMS WHOSE DECLARED CONSTANTS DO NOT APPEAR IN THE EMITTED KERNEL:\n")
for pid, missing, lits in rows:
    print("   #%-5s declared %s" % (pid, missing))
    print("          emitted literals: %s" % lits)
print("\n   %d flagged" % len(rows))
