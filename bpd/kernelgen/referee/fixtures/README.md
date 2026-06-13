# Adversarial Fixtures — the standing enemies that keep the gates honest

> *"Code with enemies stays honest. The only two subsystems my review found clean were the
> two with adversarial fixtures or constant production use."* — Bocher, 2026-06-13

These probes are **Bocher's gate-the-gates fixtures** from the second-perspective end-to-end review
(`docs/REVIEW_SECOND_PERSPECTIVE_BOCHER.md`, `docs/UNIFIED_PROJECT_MAP.md`). Each feeds a known-bad
artifact to a gate and confirms the gate REDs — proving the gate can fail and therefore can protect.
They were authored in `/tmp` during the review; vendored here so they survive and become the
foundation of the scheduled smokes (the principle: every gate gets a standing failed-test fixture).

## The probes

| probe | gate it tests | the poison |
|---|---|---|
| `failed_test_injection.py` | the fusion/comparison gates | a deliberately-wrong fused result |
| `inject_gr.py` | the GR (decode-correctness) verdict | a perturbed decode |
| `inject_longrun.py` | long_run drift/margin referee | injected drift |
| `probe_autofuser.pl` | the auto-fuser scanner | engine-vocabulary op lists |
| `probe_scanner.pl` | the scanner | the real decode graph |
| `probe_numstab.pl` / `probe_numstab2.pl` | numerical_stability detectors | order-sensitive / unstable graphs |
| `probe_cupti.pl` / `probe_cuptifacts.pl` | the CUPTI bridge | trace facts |
| `probe_validate.sh` | gguf_validate | crossword-attack GGUFs + the production control |
| `probe_reader.sh` | the native GGUF reader chain | malformed inputs |
| `probe_emitters.sh` | the emitters | a sweep of emit modes |
| `probe5.pl`, `probe_params.py`, `parse_probe.pl` | misc (parse / param-axis) | — |

## How they relate to the scheduled defenses

These fixtures are the *prototypes* the three scheduled defenses grew from:
- `probe_emitters.sh` → `smoke_emitters.py` (defense #2, per-push)
- `probe_validate.sh` (cwd=bpd/, crossword attacks + production control) → `drift_sentinel.py` (defense #3)
  and `test_drift_sentinel_poison.py` (the drift gate-the-gates)
- `probe_*.pl` numerical_stability / cupti / scanner → the standing checks the drift sentinel runs

## Running them

Most consult from the repo root or `bpd/` (the documented cwd-fragility — `gguf_validate.pl` uses
relative `lib/...` paths). Example:
```sh
cd bpd && sh kernelgen/referee/fixtures/probe_validate.sh
```

## Credit

Authored by **Bocher** (בוחר), decode-correctness / referee lens, during the 2026-06-13
second-perspective review that found 6 real bugs (4 silent) and proved every core comparison gate can
fail. The discipline they embody — *guard the instruments, not just the engine* — is his.
