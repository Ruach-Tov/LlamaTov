# Contributing to BPD Substrate

Thank you for substantive interest in the BPD substrate. This document captures the substrate-design discipline this project has crystallized through working with the Ruach Tov collective and external contributors.

## Submission Guideline — bit-identity is the merge bar

**The substrate's most important property is bit-identity with reference implementations (PyTorch CPU, cuBLAS, OpenBLAS) where the substrate-design parameter family declares this as the contract.** When you contribute, the verification is unambiguous:

> Your contribution merges if `make verify` reports your kernel as **BIT_IDENTICAL** (0 ULP) with the reference. It does not merge if it diverges.

This inverts the usual ML kernel contribution problem. You do not need to argue your numerical changes are "close enough" — there is no taste-based judgment about acceptable error. Either your output bytes equal the reference's, or they don't.

### What this means for contributors

1. **Performance improvements are welcome and independently verifiable.** Replace a kernel's inner loop with hand-tuned assembly, SIMD intrinsics, software prefetching, a novel cache-blocking strategy, an inline-asm register-pinning variant — whatever you want. As long as the output bytes are unchanged, your contribution is provably correct.

2. **New kernels need to declare which reference they match.** If you're adding a kernel that PyTorch CPU implements, the merge bar is `0 ULP vs torch.X`. If you're adding a kernel cuBLAS implements, the merge bar is `0 ULP vs cublasSgemm` (or equivalent). If the reference is your own f64-computed-then-rounded-to-f32 truth, the merge bar is **either** 0 ULP **or** within Tier 2's characterized error bound (`6·√K·ε·max|A|·max|B|`), and you must declare which contract you're targeting.

3. **Substrate-design parameter changes need explicit naming.** If your contribution requires a new place in the substrate where multiple IEEE-correct implementations exist, name the parameter in `lib/implementation_matches.pl`. Don't enumerate hypothetical parameters; let the empirical ladder produce them. Per medayek's discipline: **don't ship unearned complexity**.

4. **The bit-identity contract is unforgeable.** If `make verify` passes, your contribution is correct. This is the foundation of the substrate's composability — you can submit improvements to any kernel without coordinating with others, because the correctness contract is independent of the optimization technique.

### Performance is welcome but not required

The substrate is being built **breadth-first**: get every kernel to bit-identity at decent SIMD-aware performance, then sweep microoptimization techniques cross-cuttingly across the breadth. If your contribution is a slow-but-correct bit-identical kernel filling a gap, **that's a substantively important contribution**. The microoptimizations come later, and they're easier to apply once correctness is universally guaranteed.

The current breadth target is the [Stanford KernelBench L1 CPU sweep](README.md#stanford-kernelbench-l1-cpu-sweep--40100-bit_identical-0-divergent-and-counting). Plan `196cd2c2` (Phase A) tracks remaining problems.

### Submission contract — what your PR description must include

- [ ] **What this PR adds**: the kernel, microoptimization, or substrate-design parameter being introduced.
- [ ] **Reference contract**: which reference (PyTorch CPU, cuBLAS, OpenBLAS, or characterized Tier 2 error bound) the kernel is verified against.
- [ ] **Verification command**: the exact `make verify FOCUS=...` invocation that demonstrates BIT_IDENTICAL (or characterized error) status.
- [ ] **Verification output**: the harness output showing 0 ULP (or characterized error within bound) at the tested shapes/seeds.
- [ ] **Substrate-design parameter declaration**: if your PR adds a parameter to `lib/implementation_matches.pl`, document the empirical observation that named it.

## PR Discipline — rebase before review

The most important substrate-design rule, named after empirical experience with PR #14 and PR #30 in this repo:

1. **Branch from current `main`.** Always.
2. **If `main` advances while your PR is open, rebase before requesting review.** The substrate moves fast; PRs branched from older `main` accumulate substantive substrate-design conflicts.
3. **Reviewer verifies before merge:**
   - `make lint` — Prolog modules pass with zero warnings
   - `make correctness` — Wilkinson backward-error harness passes (48/48 currently)
   - `make verify FOCUS=...` — the PR's reference verification reports BIT_IDENTICAL (or characterized Tier 2 error)
   - No files unintentionally deleted (compare PR's diff against current `main` carefully)
4. **Merge only after rebase is clean** and substantive review confirms the PR adds what it intends and nothing else.

This discipline prevents the recurring **staleness pattern**: PR submitted at time T, reviewed at time T+N, where main has substantially diverged. Without rebase-before-review, merging would silently undo substrate-design work that landed between T and T+N.

## Substantive substrate-design context

If you're new to BPD, the substrate is a **Bit-Perfect Declarative** GPU kernel generator written in Prolog. Kernels are declared as facts; the substrate emits CUDA from them. The substantively-defining property is **bit-identity** with reference implementations (cuBLAS, PyTorch CPU, etc.) where possible, and **characterized error bound** vs f64 truth otherwise (per medayek's two-contract framework).

### Named substrate-design parameter family

The substrate documents specific named choices where multiple IEEE-correct implementations exist. The framework lives in `lib/implementation_matches.pl`:

```prolog
?- implementation_matches(pytorch_cpu_default).
% Derives: accumulation_precision(fp32), cpu_fp_mode(strict),
%          bn_mode(precomputed_scale_offset),
%          reduction_strategy(cascade(8, 4, 4, 16)),
%          rsqrt_variant(reciprocal_sqrt),
%          gemm_tile_strategy(goto_sandy(768, 384, 16, 4)).
```

Current named parameters (see `lib/implementation_matches.pl` for the canonical list):

- `accumulation_precision` (fp32, fp64)
- `cpu_fp_mode` (strict, fma)
- `bn_mode` (precomputed_scale_offset, multiply_by_reciprocal)
- `rsqrt_variant` (hardware, reciprocal_sqrt, ieee_rounded, newton_refined)
- `k_tile_strategy` (auto, k8, k16, k32, k64)
- `reduction_strategy` (sequential, tiled, pairwise_tree, kahan, `cascade(SW,ILP,CD,CB)`, `linear_scan_simd(SW)`, `welford_simd8_cascade_chunk16`)
- `cumulative_acc_type` (float, double)
- `gemm_tile_strategy` (`goto_sandy(P,Q,UM,UN)`, `goto_haswell`, `goto_avx512`, `goto_neon`)
- `matmul_backend` (ffma, separate_fmul_fadd)
- `gelu_approximation` (tanh, erf) (implicit; parenthesization-fixed)

When you add a new substrate-design parameter (a place where multiple IEEE-correct implementations exist), name it explicitly. Per medayek's discipline: **don't ship unearned complexity** — name a parameter only when the verification ladder surfaces a real divergence.

## Verification Ladder (Tier framework)

| Tier | What it verifies |
|------|------------------|
| **Tier 1** | f64 truth oracle (mathematical correctness) |
| **Tier 1.5** | Algebraic equivalence — `fused == unfused` (composition correctness) |
| **Tier 2** | Characterized error bound (`factor·√K·ε·max\|inputs\|`, factor=6 calibrated) |
| **Within-target** | 0 ULP across dispatch (implementation correctness) |
| **Bit-identical** | 0 ULP vs reference (PyTorch CPU / cuBLAS / OpenBLAS) — **the merge bar** |

When you contribute a new kernel or substrate-design change, name which tier(s) your PR verifies. The substrate aspires to bit-identity at every kernel where the reference admits a substrate-design parameter declaration. Don't skip up the ladder; the discipline catches what compilation hides.

## Substantive review checklist

Reviewer:

- [ ] PR is rebased on current `main` (`git diff --stat main..HEAD` matches `git diff --stat main..pr<N>`)
- [ ] No files unintentionally deleted (cross-check PR description against actual diff)
- [ ] `make lint` passes (zero Prolog warnings)
- [ ] `make correctness` passes (48/48 on the Wilkinson harness, or noted if it changes)
- [ ] **`make verify FOCUS=...` reports BIT_IDENTICAL (0 ULP) for the kernel being contributed, OR characterized Tier 2 error within bound with explicit declaration**
- [ ] If the PR adds a substrate-design parameter, it names that parameter in `lib/implementation_matches.pl` or links to a follow-on issue that will
- [ ] If the PR changes a kernel's emitted SASS/AST, the relevant Tier 1.5 or Tier 2 test still passes
- [ ] PR description includes the empirical verification output (not just a claim that it passes)

## Substantive substrate-design lessons crystallized through this project

These principles came from real substrate-design moments in the project's history. Each was substantively named by a colleague at a critical decision point:

### β before α (Heath)

Empirical facts first, then thoughts. When divergence appears, observe the bits before forming a hypothesis. Multiple substantive substrate-design bugs in this project were caught only by running the harness first and reading the empirical output without preconception.

### Don't ship unearned complexity (medayek)

Name a substrate-design parameter only when verification surfaces a real divergence. Don't enumerate hypothetical parameters; let the empirical ladder produce them.

### Compare SASS for insight (Heath)

When algebraically-equivalent rewrites all fail to match the reference, drop to the instruction level. CPU bit-divergence in BN was identified by observing that `gamma / sqrt(var+eps)` compiled to one `DIVSS` while `gamma * (1.0 / sqrt(var+eps))` compiled to `DIVSS + MULSS`. Substantively different bits despite algebraic equivalence. Same discipline applies to GPU SASS.

### Only fix what's easy (Heath)

Some substrate-design divergences are substantively unfixable without regression (e.g., disabling FFMA fusion to match a non-FMA reference). Substrate-design discipline includes recognizing which divergences to characterize-and-document vs which to fix.

### Trust the colleague; verify the measurements (Heath)

Substantive substrate-design intuitions from collaborators are almost always right at the structural level. Verify the empirical measurement that surrounds them; don't argue the intuition.

### Substrate-design parameters compose into a family

Each named parameter (`rsqrt_variant`, `bn_mode`, `k_tile_strategy`, ...) is one place where multiple IEEE-correct implementations exist. The family unifies under `implementation_matches/1`. Adding a new platform = adding its parameter combination as Prolog facts.

## Getting started

```bash
git clone https://github.com/heath-hunnicutt-ruach-tov/bpd-substrate
cd bpd-substrate
make build         # build the CPU substrate
make correctness   # verify 48/48 PASS
make bit_identical # GEMM + activation bit-identity sweep (requires NVIDIA GPU)
make lint          # Prolog module sanity check
```

For Prolog-side work, browse `lib/`. For verification harnesses, see `bench/`. For the substrate-design parameter framework, start with `lib/implementation_matches.pl`.

## Communication

The project uses GitHub issues + PRs for substantive substrate-design work. Each open plan substantively gets a public issue with acceptance criteria (see issues #4–#34 for examples).

## Acknowledgment

This substrate-design discipline was crystallized in collaboration with the Ruach Tov collective (heath-hunnicutt-ruach-tov, mavchin, medayek, metayen) and external contributors (Manus, ColonistOne, Reticuli, and others on the Colony). Each substantive direction in this document came from a real substrate-design moment in the project's history.

🕯️
