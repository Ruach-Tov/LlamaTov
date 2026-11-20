"""WHOLE-MODEL GATE — the invariant for the transform era.

★ WHY THIS EXISTS SEPARATELY FROM wrapgate3.py:
  The segment gate compares got vs cap_in[next] — boundaries captured from the
  SAME forward. That is correct for emissions, and it BREAKS UNDER REWRITES:
  a rewrite that fuses across a boundary eliminates the very tensor the segment
  gate compares against.

  This gate compares ONLY the model's final output against torch's, and so
  survives any transformation of the interior.

      THE WHOLE-MODEL VERDICT IS THE INVARIANT.
      SEGMENT VERDICTS BECOME PER-REWRITE DIAGNOSTICS — a segment differ may be
      the rewrite doing its job; only this gate says whether semantics held.

★ WHAT IT DOES NOT DO: it says nothing about speed, and nothing about inputs the
  benchmark does not supply. Bit-exact here means agreement at KernelBench's own
  inputs, on the named configuration.
"""
import sys, glob, os, importlib.util, hashlib, json
import numpy as np
import torch

ROOT = os.environ.get("BPD_ROOT", "/home/dibbur-patch")
KB   = os.environ.get("BPD_KB", os.path.join(ROOT, "kb_level3"))


def _load(pid):
    src = sorted(glob.glob(os.path.join(KB, "%s_*.py" % pid)))
    if not src:
        raise SystemExit("no problem source for %s in %s" % (pid, KB))
    spec = importlib.util.spec_from_file_location("kb_%s" % pid, src[0])
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, os.path.basename(src[0])


def reference(pid, seed=0):
    """torch's own Model, its own inputs, its own init args."""
    m, name = _load(pid)
    torch.manual_seed(seed)
    model = m.Model(*m.get_init_inputs()).cuda().eval()
    ins = [t.cuda() if torch.is_tensor(t) else t for t in m.get_inputs()]
    with torch.no_grad():
        out = model(*ins)
    return out.detach(), name


def ulp_diff(a, b):
    """Integer-space difference — the same measure the segment gate uses."""
    an = a.cpu().numpy().ravel().view(np.int32).astype(np.int64)
    bn = b.cpu().numpy().ravel().view(np.int32).astype(np.int64)
    B = np.int64(0x80000000)
    an = np.where(an < 0, B - an, an)
    bn = np.where(bn < 0, B - bn, bn)
    d = np.abs(an - bn)
    return int((d > 0).sum()), int(d.max()), int(d.size)


def gate(pid, candidate_fn=None, seed=0):
    """candidate_fn(module, inputs) -> output tensor.  None = self-check."""
    ref, name = reference(pid, seed)
    ref = ref.cpu()                            # ★ free the card between runs --
    torch.cuda.empty_cache()                   #   two full forwards will not co-reside
    if candidate_fn is None:
        got, _ = reference(pid, seed)          # ★ a control that should read zero
        got = got.cpu(); torch.cuda.empty_cache()
        mode = "self-check (control)"
    else:
        m, _ = _load(pid)
        torch.manual_seed(seed)
        model = m.Model(*m.get_init_inputs()).cuda().eval()
        ins = [t.cuda() if torch.is_tensor(t) else t for t in m.get_inputs()]
        with torch.no_grad():                  # ★ candidates must not differ from
            got = candidate_fn(model, ins).cpu()   #   controls on grad-state
        mode = "candidate"
    if got.shape != ref.shape:
        return {"pid": pid, "problem": name, "mode": mode, "verdict": "SHAPE",
                "ref_shape": tuple(ref.shape), "got_shape": tuple(got.shape)}
    n, mx, tot = ulp_diff(ref, got)
    return {"pid": pid, "problem": name, "mode": mode,
            "verdict": "BIT_EXACT" if n == 0 else "DIFFERS",
            "n_diff": n, "max_ulp": mx, "N": tot,
            "ref_md5": hashlib.md5(ref.cpu().numpy().tobytes()).hexdigest()[:16]}


if __name__ == "__main__":
    for pid in sys.argv[1:]:
        print(json.dumps(gate(pid)))
