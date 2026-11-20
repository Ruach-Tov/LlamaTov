"""A CONTROL THAT SHOULD READ POSITIVE.

★ A gate that only ever returns BIT_EXACT on a self-check has proved nothing.
  Perturb ONE weight by one ULP and require the gate to SEE it. If it cannot,
  the gate is blind and its zeros mean nothing.
"""
import sys, os, torch
# ★ PORTABLE: the instrument does not depend on where it lives. wholegate.py
#   is imported from the SAME directory as this script — flat in the enclave
#   layout (/home/dibbur-patch/), verification/ in the published slice.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wholegate import gate

def perturb_one_weight(model, ins):
    with torch.no_grad():
        for p in model.parameters():
            if p.numel() > 0 and p.dtype.is_floating_point:
                flat = p.view(-1)
                bits = flat[0].view(torch.int32)
                flat[0] = (bits + 1).view(torch.float32)   # ★ exactly one ULP
                break
        return model(*ins)

for pid in sys.argv[1:]:
    r = gate(pid, candidate_fn=perturb_one_weight)
    print("   #%-3s %-10s n_diff=%-10s max_ulp=%s of %s"
          % (pid, r["verdict"], r.get("n_diff"), r.get("max_ulp"), r.get("N")))
