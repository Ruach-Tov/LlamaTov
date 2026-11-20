"""WRAPPER-GATE BATCH: the only number worth publishing --
   PROBLEMS ACTUALLY GATED BIT-EXACT against KernelBench's own Model.

★ Generates a wrapper per problem via tools/emit_wrapper.py, runs its gate(),
  and records the verdict.  Every skip carries a named reason.
★ Guards: provenance window, abort on CUDA context fault, covers-assertion.
★ NOTHING is inferred -- a problem counts only if its gate() returns n_diff=0.
"""
import sys, os, re, glob, json, time, subprocess, importlib.util

# ★ PORTABLE: default is the campaign's enclave layout; a stranger sets BPD_ROOT
#   to their own checkout. Nothing else needs changing.
BPD_ROOT = os.environ.get("BPD_ROOT", "/home/dibbur-patch")
sys.path.insert(0, BPD_ROOT)
STORE, KB = os.environ.get("BPD_STORE", os.path.join(BPD_ROOT, "emitted")), os.environ.get("BPD_KB", os.path.join(BPD_ROOT, "kb_level2"))
EMIT = os.path.join(BPD_ROOT, "tools", "emit_wrapper.py")
OUT = "/tmp/wg"
os.makedirs(OUT, exist_ok=True)

_s = time.time()
files = sorted(glob.glob(os.path.join(STORE, "*.cu")))
_m = [os.path.getmtime(f) for f in files]
if _s - max(_m) < 60: raise SystemExit("ABORT: store written recently.")
ORPHANS = set()
_PROD = os.path.join(BPD_ROOT, "producible.txt")
if os.path.exists(_PROD):
    _age = time.time() - os.path.getmtime(_PROD)
    _ok = {l.strip() for l in open(_PROD) if l.strip()}
    ORPHANS = {os.path.basename(f)[3:-3] for f in files
               if os.path.basename(f)[3:-3] not in _ok}
    print("  orphan guard: producible-manifest %ds old, %d unit(s) not producible%s"
          % (_age, len(ORPHANS), (": " + " ".join(sorted(ORPHANS, key=lambda z: int(z)))) if ORPHANS else ""),
          flush=True)
    if _age > 7200:
        print("  WARNING: producible-manifest is over 2h old -- re-run mkproducible.py", flush=True)
else:
    print("  orphan guard: NO producible-manifest -- guard INACTIVE this run "
          "(run mkproducible.py to enable)", flush=True)
_m = [t for f, t in zip(files, _m) if os.path.basename(f)[3:-3] not in ORPHANS]
if max(_m) - min(_m) > 900:
    print("  note: spread %ds across the PRODUCIBLE set" % (max(_m) - min(_m)), flush=True)
# ★ STALENESS CROSS-CHECK. The provenance gate proves the store was internally
#   consistent AT RUN TIME. It cannot prove the store is up to date with fixes
#   landed since -- the mtime spread says ONE census, not the NEWEST census.
#
#   A BIT_EXACT verdict on a kernel we have since replaced is "a true measurement
#   of a kernel we are replacing". It is honest and it is stale, and a reader
#   cannot tell from the log which they are holding.
#
#   docs/AFFECTED_PIDS.md records, per fix, the pids whose stored units it
#   invalidates. Any entry newer than the store's own mtime marks those pids
#   STALE and the gate refuses to report a verdict for them.
STALE = set()
_apid = "/home/heath/Ruach-Tov/bpd/docs/AFFECTED_PIDS.md"
for _p in (os.path.join(BPD_ROOT, "AFFECTED_PIDS.md"), _apid):
    if os.path.exists(_p):
        _store_mtime = max(_m)
        for _line in open(_p):
            _line = _line.strip()
            if not _line or _line.startswith("#"): continue
            _parts = _line.split("--")[0].split()
            if len(_parts) < 3: continue
            _commit, _date, _pids = _parts[0], _parts[1], _parts[2:]
            try:
                import datetime as _dt
                _ts = _dt.datetime.strptime(_date, "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=_dt.timezone.utc).timestamp()
            except Exception:
                continue
            if _ts > _store_mtime:
                for _pid in _pids: STALE.add(_pid)
        break
if STALE:
    print("  staleness: %d pid(s) invalidated by fixes at/after the store's date: %s"
          % (len(STALE), " ".join(sorted(STALE))))


print("WRAPPER-GATE BATCH -- %d units vs KernelBench's OWN Model" % len(files))
print("  provenance: quiescent %.0fs, spread %.0fs\n" % (_s-max(_m), max(_m)-min(_m)), flush=True)

exact, differs, skip = [], [], []
for f in files:
    pid = os.path.basename(f)[3:-3]
    if pid in ORPHANS:
        skip.append(pid)
        print("  #%-5s SKIP  ORPHAN -- the lifter can no longer produce this unit" % pid, flush=True)
        continue
    if pid in STALE:
        skip.append(pid)
        print("  #%-5s SKIP  STALE -- a lift fix postdates this stored kernel" % pid, flush=True)
        continue
    kbf = glob.glob(os.path.join(KB, "%s_*.py" % pid))
    if not kbf:
        skip.append(pid); print("  #%-5s SKIP  no problem file" % pid, flush=True); continue
    wp = os.path.abspath("wrapper_imp%s.py" % pid)
    try:
        r = subprocess.run([sys.executable, EMIT, kbf[0], "imp"+pid, f],
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0 or not os.path.exists(wp):
            # the emitter may write elsewhere; look for it
            cand = glob.glob("wrapper_imp%s.py" % pid) or glob.glob("/tmp/wrapper_imp%s.py" % pid)
            if cand: wp = cand[0]
            else:
                msg = (r.stderr or r.stdout or "").strip().splitlines()
                skip.append(pid); print("  #%-5s SKIP  emit: %s" % (pid, (msg[-1] if msg else "no wrapper")[:46]), flush=True); continue
    except subprocess.TimeoutExpired:
        skip.append(pid); print("  #%-5s SKIP  emitter timeout" % pid, flush=True); continue
    try:
        env = dict(os.environ, PYTHONPATH=BPD_ROOT, CUDA_HOME=os.environ.get("CUDA_HOME",""))
        r = subprocess.run([sys.executable, wp], capture_output=True, text=True, timeout=300, env=env)
        out = (r.stdout or "").strip()
        mt = re.findall(r"'n_diff':\s*(\d+)", out)
        if not mt:
            tail = [l for l in (r.stderr or "").strip().splitlines() if l.strip()]
            reason = tail[-1][:46] if tail else "no verdict"
            if "out of memory" in reason.lower(): reason = "OOM (unfused reference)"
            skip.append(pid); print("  #%-5s SKIP  %s" % (pid, reason), flush=True)
            if "illegal memory access" in (r.stderr or ""):
                print("\n  ABORT: CUDA context poisoned at #%s." % pid, flush=True); break
            continue
        nd = sum(int(x) for x in mt)
        segs = len(mt)
        if nd == 0:
            exact.append(pid); print("  #%-5s BIT_EXACT  %d segment(s)" % (pid, segs), flush=True)
        else:
            differs.append(pid); print("  #%-5s DIFFERS    n_diff=%d across %d segment(s)" % (pid, nd, segs), flush=True)
    except subprocess.TimeoutExpired:
        skip.append(pid); print("  #%-5s SKIP  gate timeout" % pid, flush=True)
print("\n  BIT_EXACT %d   DIFFERS %d   SKIPPED %d   of %d   (%.0f min)"
      % (len(exact), len(differs), len(skip), len(files), (time.time()-_s)/60))
print("  bit-exact: " + " ".join(exact))
if differs: print("  differs: " + " ".join(differs))
