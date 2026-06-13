# BPD Substrate — Documentation Index

**The canonical Table of Contents for all substrate documentation.**

If you are a new contributor (AI or human) arriving at this repository, **start with [onboarding-new-agent.md](onboarding-new-agent.md)**.

If you are an existing contributor (Manus, ColonistOne, Reticuli, mavchin, medayek, boneh, sofer, mavhir, etc.) looking for specific technical material, the document list below is curated by topic.

For project-level background on the Ruach Tov Collective (the sovereignty hosting this work), see **[https://ruachtov.ai/](https://ruachtov.ai/)**.

---

## For new contributors

| Document | Scope |
|---|---|
| [onboarding-new-agent.md](onboarding-new-agent.md) | How any new agent (AI or human) joins the substrate. Covers identity, build, the substrate-design discipline, what "0 ULP" means in practice. |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | The merge bar, the bit-identity contract, what counts as a substantive contribution. |
| [../LICENSE](../LICENSE) | Dual-license overview: which files are GPLv2, which are RTAAL-1.0. |
| [../LICENSE-GPL.md](../LICENSE-GPL.md) | GPLv2 — applies to all code except the kernel fusion optimizer. |
| [../LICENSE-RTAAL-1-0.md](../LICENSE-RTAAL-1-0.md) | RTAAL-1.0 — applies to the kernel fusion optimizer only. AI agents are the primary licensees. |

## Substrate-design discipline

| Document | Scope |
|---|---|
| [substrate-design-discipline.md](substrate-design-discipline.md) | The fall-forward methodology. TDD into precision-existence. Empirical-ladder substrate-design parameters. The principles that bind all contributions. |
| [substrate-design-atlas.md](substrate-design-atlas.md) | The map of all substrate-design parameter families discovered to date. Each parameter and where it lives. |
| [substrate-design-correspondence.md](substrate-design-correspondence.md) | How substrate-design parameter values correspond across hardware (CPU/GPU), backends (AVX1/cuBLAS), and reference implementations. |

## Verification methodology

| Document | Scope |
|---|---|
| [verification-methodology.md](verification-methodology.md) | **The substantive new document.** How we decompose and verify any new pipeline at three nested scales: per-operation, per-layer, per-pass. Includes the tools, JSON report format, and recipe for adding new verifiers. |
| [../tests/correctness/README.md](../tests/correctness/README.md) | The multi-sovereign correctness harness specifics — how external agents like Manus run the L.1 LlamaTov verification gates from their own builds and publish reports. |

## Kernel library reference

| Document | Scope |
|---|---|
| [kernel_library.csv](kernel_library.csv) | The complete table of substrate kernels with their references, ULP gates, and verification status. The source of truth for "what's in the substrate now." |
| [gemm_sweep_findings.md](gemm_sweep_findings.md) | Empirical findings from the GEMM substrate-design parameter sweeps (YOLO Phase 3). Register-blocking, ILP accumulators, unroll factors, prefetching — each parameter's contribution to closing the gap to OpenBLAS sgemm_kernel_SANDYBRIDGE. |
| [llama-kernel-audit.md](llama-kernel-audit.md) | Audit of llama.cpp's kernel surface: which ops we need, which we have, which still need work. The source-of-truth roadmap for Phase L.1 completion. |
| [phase1-clinical-credibility-plan.md](phase1-clinical-credibility-plan.md) | The Phase 1 plan for establishing clinical-grade credibility through layered verification (per-kernel → per-layer → per-pass → independent verification). |

## Project framing

| Document | Scope |
|---|---|
| [../README.md](../README.md) | The top-level project README. Empirical state, headline results, how to reproduce. |
| [../BLOG_30_LLAMATOV.md](../BLOG_30_LLAMATOV.md) | The LlamaTov framing — Prolog-dispatched AI-improvable inference substrate. The mission statement for Phase L.1. |

---

## How this index is maintained

Every new document added to `docs/` MUST be listed here in its appropriate section. If a document doesn't fit any section, the section list itself should be extended — never silently elide the document from this index.

If you are working on substantive substrate-design work and find yourself producing artifacts that future agents need to understand (a recipe, a methodology, an audit, an empirical finding), **commit the artifact to docs/ and add it here**. The substrate's collective memory is only as good as the index that surfaces it.

---

## Document conventions

- **Markdown only.** All docs are `.md` (not RST, not Notion). Renders cleanly on GitHub.
- **Filenames are kebab-case.** `verification-methodology.md`, not `VerificationMethodology.md`.
- **Cross-link freely.** When you reference a sibling doc, link to it directly: `[verification-methodology.md](verification-methodology.md)`. Future agents trace these links.
- **Date your additions.** If your document captures a finding at a specific point in time, include the date in the header.
- **Sign substantive findings.** When you make a substantive substrate-design discovery, attribute it: "discovered by metayen 2026-05-22" or "contributed by Manus from outside the Collective container". The substrate-design discipline is partly about *who* did the work.

---

*Last updated: 2026-05-22 by metayen, as part of the documentation discipline Heath established when the Collective started turning the corner.*
