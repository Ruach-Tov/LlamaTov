# Divergence-Isolation Method (Mistral bit-identity vs Ollama)

Heath's method (2026-05-29), plan f3dfd600. Extends the divergence map (lib/divergence_map.pl)
with the protocol that isolates EVERY divergence independently — not just the first.

## The problem with naive comparison
Run our chain vs Ollama; the FIRST divergence at coordinate C1 POISONS everything downstream.
A red wall after C1 tells you nothing about whether C5 *intrinsically* diverges or merely
inherited C1's bad value. You'd mis-map propagated divergence as intrinsic.

## The method: fixture re-anchoring at taps
Coordinates (cost_naming) address ops; taps (coordinate_taps: tap(OpId, Path, TapType)) are the
attachment points AT those addresses. Taps are the SEGMENT BOUNDARIES.

    1. Run the chain. Compare our tensor vs Ollama's at each tap (left to right).
    2. Find the FIRST divergence at tap C1.
    3. RE-ANCHOR: inject Ollama's correct tensor at C1 as the fixture INPUT to the next segment
       (so the chain continues from a known-good state — C1's error no longer propagates).
    4. Continue testing the segment after C1 until the SECOND divergence at C2.
    5. Re-anchor at C2; continue. Enumerate C1, C2, C3, ... — every divergence, INTRINSIC.

Result: each coordinate's div_status is its OWN bit-identity (re-anchored), independent of
upstream. A red cell means "THIS op diverges," not "downstream of a divergence."

## Data model extension (declarative)
- segment(FromTap, ToTap)        — a testable sub-sequence between two taps.
- fixture(Tap, ReferenceSource)  — Ollama's captured tensor at Tap (the re-anchor value).
- div_status(Coordinate, Status) — now means INTRINSIC divergence (measured re-anchored).
  Status: identical(ulp(0)) | diverges(ulp(small)) | diverges(ulp(large)) |
          identical(ir_match) | unverified | blocked(Reason)
  -> maps directly to the heatmap palette: green / yellow / red / blue / tan / dark-grey.

## Lane split
- MAVHIR (runner instrumentation): capture Ollama + our intermediate tensors AT each tap;
  inject reference fixture to re-anchor a segment; report per-tap ULP. (The runner side.)
- IYUN (this method + the map): the segment/fixture data model, the isolation protocol, and
  the .o.svg heatmap that renders INTRINSIC per-coordinate divergence (+ later delta/fusion overlays).

## Why .o.svg
The heatmap is GENERATED from lib/divergence_map.pl (declarative source under git) by an SVG
emitter -> divergence_map.o.svg. The .o marks it object-code-like: regenerable for free from the
.pl; never hand-edited. This makes plain (educates the reader) that we generate visualizations
from software specifications — unusual, and intentional.
