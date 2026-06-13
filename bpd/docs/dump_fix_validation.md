# Dump-aliasing fix — consumer-side validation (loop closed)

metayen shipped the eval-callback dump fix (bpd-substrate 1523c93) + rebuilt the binary. Consumer-side
re-validation (Iyun, 2026-05-29) confirms the loop closes: trustworthy-by-construction AND verified-by-execution.

## STEP 1 — re-dump (new binary -> new-schema dump)
LLAMA_DUMP_DIR=<home>/tmp/spec_dump_v2 ./build/bin/llama-eval-callback -m BLOB -p "Hello" -n 1
-> 1073 files; manifest gains kind(out|src) + src_index columns. Old spec_dump preserved for A/B.

## STEP 2 — guard (new-schema fast path)
run_guard(spec_dump_v2):
  is_new_schema=True  schema=new(metayen-1523c93)
  name_reuse_violations=0  output_alias_detected=0  op_outputs=550  trustworthy=True
The schema-aware guard fast path fired -> trustworthy-by-construction (no heuristic).

## STEP 3+4 — A/B re-measure (the validation of the fix)
Green program vs the CLEAN dump (kind==out filter, zero aliasing risk), A/B vs old:
  attn_norm: ULP=0 (old: 0 ULP) -> MATCH
  residual1: ULP=0 (old: 0 ULP) -> MATCH
The greens give IDENTICAL 0 ULP against the trustworthy dump -> they were NOT aliasing artifacts.

## Outcome
Producer fix (metayen 1523c93) + binary rebuild + consumer guard (dump_invariants.py schema-aware) +
A/B re-measure = the producer/consumer substrate-bug pattern executed end-to-end, both lanes, clean.
Round-Mistral fixturing now runs on a clean, guarded, trustworthy dump.
