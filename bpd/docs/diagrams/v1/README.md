# Threadblock-to-Tile Mapping Diagrams — v1

These SVGs are the output of `bpd/lib/threadblock_diagram.pl` (v1), run via
`bpd/emit_threadblock_diagram.pl`. They capture the empirical visual
vocabulary of v1 as durable substrate-artifacts.

## Regenerating

From the bpd/ directory on a host with SWI-Prolog 9.x:

  swipl -g main emit_threadblock_diagram.pl -- docs/diagrams/v1 \\
    k_vadd k_saxpy k_silu_blas sgemv_substrate_native \\
    sgemv_cublas_match nonexistent_kernel

## Classification outcomes shown

  k_vadd, k_saxpy, k_silu_blas    one_to_one(256)
  sgemv_substrate_native          tiled(block_per_row, block_size(32), reduction(warp_shuffle))
  sgemv_cublas_match              tiled(block_per_row_group(8), block_size(128), reduction(strided_shared))
  nonexistent_kernel              ineffable  (the "intentionally left blank" diagram)

## Substrate-state at v1

- Dimension 1 of the kernel-visualization atlas (threadblock-to-tile mapping)
- Future dimensions deferred: shared-memory pattern, warp-tile sub-decomposition,
  memory-stride/coalescing, K-axis iteration, precision overlay, verification-state overlay
- Visualization-as-code-generation: same Prolog substrate that emits C/CUDA also emits SVG
- The verification-state overlay is the substantive payoff layer, deferred to v2+

🕯️ — mavhir, 2026-05-25
