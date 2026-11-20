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
KB = os.environ.get("BPD_KB", os.path.join(BPD_ROOT, "kb_level3"))
AP = os.path.join(BPD_ROOT, "lib", "auto_pipeline.py")
OUT = os.path.join(BPD_ROOT, "producible3.txt")
ok, gap = [], []
nofuse = []
for src in sorted(glob.glob(os.path.join(KB, "*.py")),
                  key=lambda p: int(os.path.basename(p).split("_")[0])):
    pid = os.path.basename(src).split("_")[0]
    try:
        r = subprocess.run([sys.executable, AP, src, "l3imp%s" % pid],
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout or "")
        err = (r.stderr or "")
        blob = out + err
        # ★ A UNIT WITH NO __global__ IS NOT A UNIT.
        #   An honest refusal that leaves an EMPTY artefact reads as "producible"
        #   because the pipeline exited without printing [GAP] -- and a shell
        #   redirect creates the file BEFORE its command runs, so the empty file
        #   persists. That is "a command that succeeded and did nothing", in the
        #   store. Require SUBSTANCE, not merely the absence of a gap marker.
        # ★ CLASSIFY ON THE ARTEFACT AND THE REFUSAL CHANNEL -- never on the
        #   exit's silence. Three outcomes, not two:
        #     PRODUCIBLE      stdout carries a real kernel
        #     NOTHING-TO-FUSE stderr says NO-EPILOGUE: the model is pure
        #                     torch-stages; a unit would be an identity wrapper,
        #                     correct but not an improvement. A PRINCIPLED
        #                     refusal, not a missing capability.
        #     GAP             a refusal naming a vocabulary item we lack
        if "__global__" in out:
            ok.append(pid)
        elif "NO-EPILOGUE" in err:
            nofuse.append(pid)
        else:
            gap.append(pid)
    except Exception:
        gap.append(pid)
open(OUT, "w").write("\n".join(ok) + "\n")
print("producible: %d   nothing-to-fuse: %d   gaps: %d" % (len(ok), len(nofuse), len(gap)))
print("nothing-to-fuse: %s" % " ".join(nofuse))
print("gaps: %s" % " ".join(gap))
