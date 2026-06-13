# Scheduled Defenses — operating the three guards

Three guards, one per time-constant of silent breakage (see `docs/UNIFIED_PROJECT_MAP.md`).
Each is built and self-verified; this file documents how they RUN.

## The three defenses

| time-constant | guard | cadence | invocation |
|---|---|---|---|
| **ENV SHIFT** | `lib/toolchain.py` | always (imported) | the single source of CUDA-env truth — import it, don't reinvent |
| **FAST REGRESSION** | `kernelgen/referee/smoke_emitters.py` | per-push | git pre-push hook (below) |
| **SLOW DRIFT** | `kernelgen/referee/drift_sentinel.py` | weekly | cron / ScheduledEvent (below) |

## Defense #1 — toolchain.py (ENV SHIFT)
Not scheduled — it's a *module*. Every place that needs nvcc/CUDA imports it:
```python
import toolchain as tc
cmd = tc.nvcc_compile_cmd(cu, out)          # the canonical compile argv (with -I)
subprocess.run(cmd, env=tc.nvcc_env())      # the canonical environment
lib = tc.libcuda()                          # the driver library
```
Prolog tests: `tc.write_prolog_facts(path)` emits `toolchain_fact/2` to consult — no hardcoded /nix.
It rejects the nix split-output trap (`which nvcc` → a headers-less package) by requiring a COMPLETE
root (bin/nvcc AND include/cuda_runtime.h).

## Defense #2 — smoke_emitters.py (FAST REGRESSION) — PER-PUSH
Install the pre-push hook (enclave-local; `.git/hooks/` isn't tracked):
```sh
cp hooks/pre-push .git/hooks/pre-push && chmod +x .git/hooks/pre-push
```
Now every `git push` runs the emitter smoke first (7 GEMV modes: warning-free + complete kernel).
Manual run / with compile:
```sh
python3 bpd/kernelgen/referee/smoke_emitters.py            # fast (no nvcc)
python3 bpd/kernelgen/referee/smoke_emitters.py --compile  # also nvcc-compiles each
```
Override a known-good push: `git push --no-verify` (you own the consequence).
Gate-the-gates verified: injecting `BlockSz is BM*32` into a serial clause (BM unbound) REDs it.

## Defense #3 — drift_sentinel.py (SLOW DRIFT) — WEEKLY
Runs against the SHIPPING artifact (`<data>/models/qwen_q8.gguf`):
```sh
python3 bpd/kernelgen/referee/drift_sentinel.py   # exit 0 = no drift; nonzero = drift surfaced
```
Scheduled weekly (Mon 09:00) via the agent scheduler (task `drift_sentinel_weekly`), result to the
Iyun inbox. To re-schedule via cron directly:
```cron
0 9 * * 1  cd <repo> && python3 bpd/kernelgen/referee/drift_sentinel.py
```
Checks: gguf_validate(production model) all-pass; the scanner classifies the real engine vocabulary
(rms_norm/silu_mul/q8_gemv/…); the decode-graph fusion scan runs clean.
NOTE: must run with cwd at the repo root or `bpd/` — `gguf_validate.pl` uses relative `lib/...` consult
paths (a documented cwd-fragility).

## The principle
"The diff is only as trustworthy as the gate that runs it, and a gate rots silently." These guards
test the INSTRUMENTS on the schedule each kind of rot moves — the engine half is mature; the gate half
must be kept honest. Bocher's `/tmp` probes are the adversarial fixtures these grew from.
