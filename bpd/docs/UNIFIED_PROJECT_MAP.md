# THE UNIFIED PROJECT MAP — Synthesis of Two Reviews
## Iyun (generator/backend lens) + Bocher (referee/gate lens), 2026-06-13
## For Heath. Sources: Iyun memory 4d87cd9e; Bocher REVIEW_SECOND_PERSPECTIVE_BOCHER.md (7ef1f59d).

> Heath asked each of us to review the whole project end-to-end. We looked through
> different lenses and the two passes interlock into one picture. This is that picture.
> The thesis has TWO halves — *the engine generates correct fast kernels from facts*
> (Iyun's lens) and *the instruments prove it* (Bocher's lens) — and a project is only
> as honest as the weaker half. Both halves are now mapped.

---

## I. WHERE THE PROJECT STANDS (the engine half — Iyun)

**Mission intact and delivered:** a transformer engine GENERATED FROM PROLOG FACTS, 0-ULP
to reference, ~166.5 tok/s = ~108-114% of Ollama on PyTorch-abandoned Pascal (Tesla P4
sm_61). Architecture LIFT→ADDRESS→TRANSFORM→SEARCH→VERIFY is mature; the 2026-06-08
thesis-audit gaps are largely closed (flash_attn_schedule, transformer_layer
plan_from_chains, schedule-IR tiled_*). 58 op_expr facts, 12+ real multi-backend emitters
(cuda_c, cuda_oxide, oxide_from_facts, gemm, ggml, llvm, mlir, mlir_gpu, torch, q8_0, rope,
norm_softmax). RCo/oxide proven 0-ULP cross-backend, 86% of production throughput. GGUF
loader is a clean pure-Python parser, not a stub.

**The performance frontier is mapped and mostly closed.** GEMVs are 76% of kernel time
(177M of 234M ns), at ~83% of the DRAM wall. This session: __launch_bounds__(512,4) lifted
the occupancy-starved lm_head from 83%→98% of wall (+0.63% e2e, 0-ULP, commit 9fdc171f).
Rejected honestly: argmax-into-lm_head (~0.15%), q4 lm_head (breaks 0-ULP), AoSoA weight
slab (-13% lm_head — the engine's split-SoA is ALREADY the optimal streaming layout; the
slab re-introduces per-row interleaving that breaks the quant prefetch burst). Remaining
honest leads: attn_decode_masked at 31% occupancy (8% of time, unexplored), or a regime
change (batching, long-context split-K). Single-token decode is weight-streaming-bound;
that floor is near.

**The fusion frontier is doctrine-bounded.** Three chain-audit fusions shipped (residual-
carry, rope-QK, silu-into-quant). The rms→quant seam was born polyglot, 0-ULP four ways,
but e2e-NEUTRAL (kept off) — which TAUGHT the launch-overhead doctrine: launches are free
under graph capture, so future fusions must promise COMPUTE or DRAM-byte reduction, not
launch elimination. The seam's token-11 divergence (serial-vs-block_row) taught the static-
veto doctrine and BORE the fusion_reduction_gate.

---

## II. WHERE THE INSTRUMENTS STAND (the gate half — Bocher)

Bocher fed every core comparison gate known-bad artifacts (gate-the-gates) and toured all
components. **The gates CAN fail, therefore can protect** — fusion_gate (8 probes), GR
verdict, long_run drift/margin, composition gate, and the day-old fusion_reduction_gate all
RED correctly on poison. But the TOUR found 6 real breakages, 4 of them SILENT:

| # | subsystem | breakage | commit | time-constant |
|---|-----------|----------|--------|---------------|
| 1 | GGUF-BPD | bpd_format/1 rename drift, 7 sites, broken since May 29 | 8cdb3a42 | SLOW DRIFT |
| 2 | auto-fuser | scanner didn't speak engine vocab (rms_norm/silu_mul/q8_gemv) | 949d0e09 | SLOW DRIFT |
| 4 | gguf_validate | FALSE-POSITIVE gate — rejected EVERY real model | 4dd8e66d | SLOW DRIFT |
| 7 | gemv_sweep | Iyun's launch_bounds wave: unbound BM in 4 serial clauses | b5070aad | FAST REGRESSION |
| 8 | KernelBench L1 | nix split-output: PATH nvcc lacks nvvm/cicc + -I heuristic | b886f0db | ENV SHIFT |
| 10| emitters | oxide_from_facts printed UNBOUND VAR into every provenance comment | 3b4665f0 | SLOW DRIFT |

Two subsystems CLEAN — and tellingly, they are **the two with adversarial fixtures or
constant production use**: the native reader chain (crossword attacks) and CUPTI. Plus
numerical_stability now 14/14 (Iyun's 3 dead detectors confirmed fixed).

---

## III. THE UNIFIED FINDING — the two lenses agree on the diagnosis

Both reviews, independently, found the SAME disease in different organs:
**silent breakage that fails toward false confidence.** Iyun found it by accident (3 dead
detectors in a file he touched; a test reporting 0/30 that was really 30/30). Bocher found
it BY DESIGN (gate-the-gates + a component tour). The accidental-vs-systematic contrast IS
the lesson: **a finding you stumble on is a class you aren't hunting.**

The unification: Bocher's THREE TIME-CONSTANTS map cleanly onto WHERE in the
LIFT→...→VERIFY pipeline each rot lives, and each wants a different guard:

- **SLOW DRIFT (months)** — names/schemas/vocabularies drift from production reality.
  Lives in the LIFT/ADDRESS layer (the engine's vocabulary) and the VERIFY layer (a
  validator's assumptions). 4 of 6 bugs. **Defense: SCHEDULED RUNS against the SHIPPING
  artifact** (weekly gguf_validate of the production model; scanner over the real decode
  graph; per-format regeneration smoke). Drift is invisible until something current is
  re-checked against it.

- **FAST REGRESSION (hours)** — optimization collateral. Lives in the TRANSFORM/SEARCH
  layer where we move fast. Iyun's launch_bounds wave (stop 7) is the exemplar — a 12-hour-
  old win splattered into clauses he didn't know existed. **Defense: PER-PUSH SMOKE PROBES**
  in the change path (emit one kernel of each mode; one sweep point asserting verdict==OK).

- **ENVIRONMENT SHIFT (substrate moves)** — nix split-outputs vs assumed toolkit layout.
  Lives BETWEEN the engine and the metal. KernelBench L1 broke THREE distinct ways on this
  one axis, because **production and tests learn env lessons SEPARATELY**. Iyun's -I fix and
  Bocher's nvvm/cicc fix were the same lesson learned twice. **Defense: ONE SHARED
  toolchain-discovery module** that both production and every test import.

---

## IV. THE CROSS-CUTTING SPINE (what makes the difference, both lenses)

1. **ZERO-WARNINGS = CANARY AUDIBILITY.** The gemv_sweep bug printed a singleton warning at
   EVERY consult naming the exact lines — the dead canary nobody read, because warning noise
   buried it. The product of a zero-warnings policy isn't tidiness; it's that the ONE warning
   that matters is audible. (Iyun's adjacent-string fixes + Bocher's singleton sweep started
   this; finish it on emitters/ and lib/.)

2. **CODE WITH ENEMIES STAYS HONEST.** The only two clean subsystems are the two with
   adversarial fixtures (crossword attacks) or constant production use. The defense
   generalizes: give EVERY gate a standing failed-test fixture. Bocher's /tmp probes
   (failed_test_injection.py, inject_gr.py, probe_*.pl/.sh) are these fixtures already built
   — they become the scheduled smokes. A gate without an enemy is a gate that rots toward
   green-or-red-but-always-the-same.

3. **A GATE THAT ALWAYS REDS IS AS DEAD AS ONE THAT ALWAYS GREENS** — and LOOKS vigilant.
   gguf_validate rejected every real model for weeks while appearing to guard. False
   confidence has two faces: the silent pass AND the reflexive reject. Both need the
   gate-the-gates probe (does it pass the GOOD as well as fail the BAD?).

4. **VALIDITY vs PROFITABILITY must be co-located.** The scanner finds valid fusions but
   3 of its 8 are known 0.55x anti-fusions in the veto ledger — a reader without ledger
   knowledge would wire them. The validity layer should CITE the profitability ledger.
   (This is the same lesson as the launch-overhead doctrine: a thing can be valid and a loss.)

5. **SKIPS ARE SOFT ROT.** ggufq conformance is 14/20 SKIPPED (missing v3 fixtures) while
   the production GGUF sits right there. Point the conformance suite at the shipping artifact.

6. **THE PUBLIC SUBSTRATE CAN'T REGENERATE.** bpd-substrate's GGUF infra is only in
   must_close/boundary_dsl — the public copy claims a regeneration capability it lacks.
   Vendor the infra or mark the artifacts frozen.

---

## V. THE PRIORITIZED ASK (one map → one next move per time-constant)

The disease is named; the cure is three scheduled defenses, one per time-constant. In
priority order (each is small, each kills a CLASS):

1. **ONE shared toolchain-discovery module** (kills ENV SHIFT). Highest leverage: the same
   env bug bit production-vs-test THREE times on one harness. Extract CUDA_HOME/include/
   binary discovery (the fact_dispatch.py logic) into a module both import. ~1 file.

2. **Per-push smoke probes** (kills FAST REGRESSION). Bocher's /tmp probes are 80% of this.
   Wire: emit one kernel per mode + one sweep point asserting verdict==OK, on every emitter
   change. Catches the next launch_bounds-class splatter before it ships.

3. **Scheduled runs against the shipping artifact** (kills SLOW DRIFT). Weekly: gguf_validate
   the production model, run the scanner over the real decode graph, regenerate each BPD
   format. Drift only shows when current reality re-checks the assumption.

Cross-cutting, cheap, do-alongside: finish zero-warnings on emitters/+lib/; give every gate
a standing failed-test fixture (Bocher's probes); make the scanner cite the veto ledger;
point ggufq conformance at the production GGUF.

---

## VI. THE META-POINT FOR THE THESIS

The project's deepest claim is "structure not degree — run diff yourself." This review pass
proved the COROLLARY: *the diff is only as trustworthy as the gate that runs it, and a gate
rots silently.* The engine half (Iyun) is mature and near its performance floor. The gate
half (Bocher) is sound in DESIGN — every gate can fail — but was rotting in MAINTENANCE,
because the project builds fast and the instruments aren't themselves continuously tested.
The fix isn't more gates; it's THREE SCHEDULED DEFENSES that test the instruments on the
schedule each kind of rot moves. "The instruments guard the engine; someone must guard the
instruments" — and now we know the someone is a cron job per time-constant.

— Iyun & Bocher, two lenses, one map. 🕯️
