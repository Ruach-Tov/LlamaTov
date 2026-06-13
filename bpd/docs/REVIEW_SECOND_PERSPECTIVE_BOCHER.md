# END-TO-END PROJECT REVIEW — Second Perspective (Bocher, 2026-06-13)
## Companion to Iyun's review (memory 4d87cd9e); Iyun synthesizes both into one map.

## METHOD
Failed-test injection first (per Heath): every core comparison gate fed known-bad
artifacts, verified to RED. Then a component tour: GGUF (both pipelines), auto-fuser,
numerical_stability, gguf_validate + crossword attacks, native reader chain,
CUPTI-from-Prolog, sweep drivers, KernelBench L1, e2e, all 25 emitters.

## PHASE 1 — FAILED-TEST INJECTION: ALL GATES CAN FAIL, THEREFORE CAN PROTECT
- fusion_gate.compare_outputs: 8 probes (bit-flips at edges/middle, NaN, tolerance
  boundary both sides, degenerate zeros) — all correct. Semantics documented:
  same-bits NaN passes (bit comparison, correct); zeros-vs-zeros passes (the
  stale-cubin shape — why content-addressing + token-exact layers exist).
- GR verdict layer: injected 1-bit flip located to exact step; attestation
  mismatch detected.
- long_run drift/margin/near-flip math: engineered drift + thin margin censused
  exactly; token flip located.
- Composition gate: production profile ACCEPTED; QFUSED+fusions REFUSED.
- fusion_reduction_gate (1 day old): unattested fold of reduction-bearing op
  VETOED citing its birth-bug; attested/tolerance/no-reduction all correct.

## PHASE 2 — TOUR FINDINGS (6 fixed, 2 clean, 10 stops)
1. GGUF BPD (FIXED 8cdb3a42): bpd_format/1 rename drift — 7 query sites still
   asked format(F), which hits SWI's I/O builtin. Regeneration broken since
   May 29; output/ artifacts outlived their generator. Also: bpd-substrate copy
   cannot regenerate AT ALL (infra only in must_close/boundary_dsl); env needs
   undocumented nix-shell (lark+pytest).
2. AUTO-FUSER (FIXED 949d0e09): name-drift — scanner knew rmsnorm/silu/gemm,
   not engine names rms_norm/silu_mul/q8_gemv. Could never analyze the real
   decode graph. Planner logic itself sound (probes passed); fusion_scan_decode
   passed all 3 injections (multi-consumer suppressed, single fires once,
   region-mismatch refused).
3. NUMERICAL_STABILITY (CLEAN, 14/14): all six detector families fire on bad,
   silent on benign. Iyun's 3 dead detectors confirmed fixed. LESSON: my round-1
   probes mis-shaped (wrong op types) — failed-test injection has its own
   wrong-oracle hazard; probes must honor the detector's declared contract.
4. GGUF_VALIDATE (FIXED 4dd8e66d): the security validator REJECTED every real
   model — has_rope_params hardcoded llama.* keys (qwen2.rope.* failed);
   type_consistency flagged universal F32-norm+Q8-weight layout. A gate that
   always reds is as dead as one that always greens — and LOOKS vigilant.
   Fixed; production 5/5 PASS; all 8 crossword attacks still REFUSED
   (mixed_quant_layer weight-mixing still caught).
5. NATIVE READER CHAIN (CLEAN): production file parsed (290/34 matching ggufq);
   truncated header, bad magic, string-length-overflow attack all refused with
   structured errors naming position/need/limit. safe_read working.
6. CUPTI-FROM-PROLOG (CLEAN): optimization_needed fires/silences correctly,
   strict-> boundary; structural_bottleneck detects; cupti_facts window+
   aggregation exact on synthetic trace; kernel_time_summary confirmed fixed.
7. GEMV_SWEEP (FIXED b5070aad — biggest catch): launch_bounds wave (9fdc171f,
   12h old) put BlockSz is BM*32 into FOUR serial-mode clauses where BM is
   unbound -> is/2 throws mid-emit -> partial .cu -> sweep reference garbage ->
   every verdict BAD max_ulp=3.3e9. Singleton warnings flagged the exact lines
   at every consult — THE DEAD CANARY NOBODY READ. Fixed BlockSz=64; sweep
   restored (OK, max_ulp=803 known tolerance class).
8. KERNELBENCH L1 (FIXED b886f0db): 30/30 failing TWO ways on nix split-output
   packaging: PATH nvcc lacks nvvm/cicc (exit 127 mid-compile); -I derive
   heuristic also fails. THIRD distinct breakage of this harness. Fix mirrors
   fact_dispatch.py (CUDA_HOME || merged toolkit for binary AND includes).
   30/30 PASS.
9. E2E (CLEAN): 2 passed.
10. EMITTERS (FIXED 3b4665f0): consult sweep over 25 — 11 dirty, 2 with
    singletons: head_fusion dead rows; oxide_from_facts printed an UNBOUND
    VARIABLE into every generated kernel's formulation provenance comment
    (author intended member(formulation(F),Ev), never wrote it).

## TAXONOMY — three time-constants of silent breakage, each needing a different defense
- SLOW DRIFT (months): names/schemas/vocabularies diverge from production
  reality (stops 1,2,4). Defense: SCHEDULED RUNS against production artifacts
  (weekly gguf_validate of the shipping model; scanner pass over the real
  decode graph; regeneration smoke per BPD format).
- FAST REGRESSION (hours): collateral from optimization velocity (stop 7).
  Defense: SMOKE PROBES in the change path (per-push: emit one kernel of each
  mode; one sweep point asserting verdict==OK).
- ENVIRONMENT SHIFT (whenever substrate moves): nix split-outputs vs assumed
  toolkit layout (stop 8, three breakages of one harness). Defense: ONE SHARED
  toolchain-discovery module (production + tests use the same env knowledge).

## CROSS-CUTTING RECOMMENDATIONS
A. Zero-warnings policy on emitters/ and lib/ — warning noise buried the one
   warning that named the gemv_sweep bug. Discontiguous directives + singleton
   fixes are cheap; canary audibility is the product. (Partially done on tour.)
B. The two healthy subsystems (reader chain, CUPTI) are the two with adversarial
   fixtures or constant production use. CODE WITH ENEMIES STAYS HONEST — give
   every gate a standing failed-test fixture (the crossword_attacks pattern,
   generalized; several built on this tour, reusable).
C. Skips-as-soft-rot: ggufq conformance suite 14/20 skipped (missing v3
   fixtures) while the production GGUF sits right there. Point conformance at
   the shipping artifact.
D. bpd-substrate (public repo): either vendor the boundary_dsl infra or mark
   generated artifacts as frozen — currently the public copy claims a
   regeneration capability it does not have.
E. The scanner (validity) could cite the veto ledger (profitability) — 3 of its
   8 found opportunities are known 0.55x anti-fusions; a reader without ledger
   knowledge would wire them.

## STANDING ASSETS CREATED
/tmp probes (reusable as scheduled smokes): failed_test_injection.py (gate
core), inject_gr.py, inject_longrun.py, probe_autofuser.pl, probe_numstab2.pl,
probe_scanner.pl, probe_cupti.pl, probe_cuptifacts.pl, probe_validate.sh,
probe_reader.sh, probe_emitters.sh.
