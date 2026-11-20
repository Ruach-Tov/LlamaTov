"""Write the producibility manifest: which problems the PIPELINE can still emit.

★ WHY THIS IS A SEPARATE PASS: the check runs auto_pipeline, which WRITES to the
  store. Running it inside the gate tripped the gate's own quiescence guard --
  an instrument that disturbs what it measures is not an instrument.
★ WHY THE PIPELINE AND NOT THE LIFTER: segments-fallback units refuse at
  single-chain lift and emit correctly. Asking the lifter flagged 26 units
  including three sealed the same morning.
"""
import glob, os, subprocess, sys

# ★ PORTABLE: default is the campaign's enclave layout; a stranger sets BPD_ROOT
#   to their own checkout. Nothing else needs changing.
BPD_ROOT = os.environ.get("BPD_ROOT", "/home/dibbur-patch")
KB = os.environ.get("BPD_KB", os.path.join(BPD_ROOT, "kb_level2"))
AP = os.path.join(BPD_ROOT, "lib", "auto_pipeline.py")
OUT = os.path.join(BPD_ROOT, "producible.txt")
ok, gap = [], []
for src in sorted(glob.glob(os.path.join(KB, "*.py")),
                  key=lambda p: int(os.path.basename(p).split("_")[0])):
    pid = os.path.basename(src).split("_")[0]
    try:
        r = subprocess.run([sys.executable, AP, src, "imp%s" % pid],
                           capture_output=True, text=True, timeout=120)
        (gap if "[GAP]" in (r.stdout or "") + (r.stderr or "") else ok).append(pid)
    except Exception:
        gap.append(pid)
open(OUT, "w").write("\n".join(ok) + "\n")
print("producible: %d   gaps: %d" % (len(ok), len(gap)))
print("gaps: %s" % " ".join(gap))
