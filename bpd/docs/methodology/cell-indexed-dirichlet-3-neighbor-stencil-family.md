# The Cell-Indexed Dirichlet 3-Neighbor Stencil Family

**Date crystallized**: 2026-05-18 ~15:35 UTC
**Discovered through**: F4 jacobi1d reconnaissance under Heath's "(D) then re-examine (A)" direction
**Status**: substrate-design taxonomy — informs prioritization of future kernel work

A *family* in this methodology means a set of algorithms across
different application domains that share the same substrate-emit
shape. When the substrate can express one family member well, it can
express every other family member with parameter changes only.

This document names one family explicitly and lists its members. The
purpose is to make visible the leverage that a single well-designed
substrate primitive can provide across multiple domains.

## The family pattern

**Cell-indexed**: one thread processes one cell. The loop iterates
over cells, not interfaces.

**Dirichlet BC**: boundary cells (i=0, i=N-1) are held at fixed
values throughout the computation. Interior cells (1 ≤ i ≤ N-2) read
from their neighbors but the boundary cells never read or write.

**3-neighbor stencil**: each interior cell reads from cells [i-1],
[i], [i+1]. Stencil width is 3 (immediate left, center, immediate
right).

**Single-component**: one float per cell. Distinct from CFD's
3-component conservative state.

The substrate-emit shape:

```c
__global__ void k_<family_member>(const float * in,
                                    float * out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0 || i >= N - 1) return;  // Dirichlet skip
    float L = in[i - 1];
    float C = in[i];
    float R = in[i + 1];
    out[i] = <family-specific combination of L, C, R>;
}
```

Different family members differ ONLY in `<family-specific combination
of L, C, R>`. The loop, the bound, the boundary handling, the memory
layout, and the per-thread structure are all shared.

## Family members

### PDE solvers

**Jacobi1D averaging** (1D smoothing iteration):
```c
out[i] = (L + C + R) / 3.0f;
```
Used in iterative Poisson solvers, image smoothing as a baseline.

**1D heat equation, explicit Euler**:
```c
out[i] = C + alpha * dt_dx2 * (L - 2.0f * C + R);
```
α is diffusivity, dt_dx2 is dt/dx². Used in thermal simulation,
diffusion modeling, pollutant dispersion, option pricing
(Black-Scholes diffusion term).

**1D wave equation, leapfrog**:
```c
out[i] = 2.0f * C - prev[i] + c2_dt2_dx2 * (L - 2.0f * C + R);
```
Needs a second array `prev` for the previous timestep. c² is wave
speed squared. Used in seismic wave propagation, acoustic
simulation, vibrating string physics.

**1D Burgers equation, central differencing**:
```c
out[i] = C - dt * C * (R - L) / (2.0f * dx)
             + nu * dt_dx2 * (L - 2.0f * C + R);
```
Combines advection (first derivative, central difference) and
diffusion (Laplacian). Used in traffic flow modeling, shock
formation studies, gas dynamics simplifications.

**1D Schrödinger equation, Crank-Nicolson** (real-valued component):
```c
out_re[i] = <combination of in_re[L,C,R] and in_im[L,C,R]>;
```
Complex-valued; substrate would need a separate Re/Im pass OR
complex float support (a substrate-design extension axis).

**Iterative Poisson solver (Jacobi method)**:
```c
out[i] = 0.5f * (L + R - dx2 * f[i]);
```
Solves ∇²u = f. f[i] is the source term. Used in electrostatics,
gravitational potential, image processing (Laplacian solves),
reconstruction problems.

### Linear algebra

**Tridiagonal matrix-vector product** `y = T*x`:
```c
out[i] = a[i] * L + b[i] * C + c[i] * R;
```
With variable coefficients per row. Used in any banded linear
system, finite-difference matrices, Markov chains with
nearest-neighbor transitions.

**Back-substitution of Thomas algorithm** (parallel phase):
```c
out[i] = (d[i] - c_prime[i] * R) / b_prime[i];
```
Used in cyclic reduction for parallel tridiagonal solvers.

**Cyclic reduction step** (recursive 3-neighbor with stride doubling):
```c
out[i] = a_new[i] * in[i - stride] + b_new[i] * in[i] + c_new[i] * in[i + stride];
```
Stride doubles each level: 1, 2, 4, 8, ..., N/2. Substrate would
need a `stencil_stride` parameter — a family extension axis.

### Signal and image processing

**1D convolution with kernel size 3**:
```c
out[i] = w0 * L + w1 * C + w2 * R;
```
The weight triple (w0, w1, w2) parameterizes:
- **Box filter**: (1/3, 1/3, 1/3) — this is Jacobi1D
- **Gaussian smoothing**: (1/4, 1/2, 1/4)
- **Edge detection**: (-1, 0, +1)
- **High-pass filter**: (-1/4, 1/2, -1/4)
- **Sharpen**: (-1/2, 2, -1/2)

**1D wavelet lifting step**:
```c
out[i] = C - (L + R) / 2.0f;  // predict step
out[i] = C + (L + R) / 4.0f;  // update step
```
Used in multi-scale signal decomposition.

**Bilateral filter (1D, simplified)**:
```c
float w_L = exp(-|L - C| / sigma);
float w_R = exp(-|R - C| / sigma);
out[i] = (w_L * L + C + w_R * R) / (w_L + 1 + w_R);
```
Non-linear weights but same neighborhood structure.

### Graph algorithms

**Graph Laplacian on a path graph**:
```c
out[i] = 2.0f * C - L - R;  // (i.e., -L + 2*C - R)
```
The graph Laplacian operator restricted to path graphs. Used in
spectral graph theory, mesh smoothing, manifold learning,
spectral clustering.

**Random walk transition on 1D lattice**:
```c
out[i] = 0.5f * (L + R);  // simple random walk, no self-loop
out[i] = (1.0f - p) * C + 0.5f * p * (L + R);  // lazy walk with prob p
```
Used in random-walk-based graph algorithms, PageRank on path graphs.

### Numerical methods

**Simpson's rule, local integration**:
```c
out[i] = dx / 3.0f * (L + 4.0f * C + R);  // ∫ from i-1 to i+1
```
Used for numerical integration of tabulated functions.

**Richardson extrapolation, local 2nd-order**:
```c
out[i] = (4.0f * mid_step[i] - coarse_step[i]) / 3.0f;
```
Where mid_step and coarse_step come from different resolutions.

### Machine learning

**1D depthwise convolution, kernel size 3**:
```c
out[i] = w0 * L + w1 * C + w2 * R + bias;
```
Used in some audio models, time-series transformers, signal-
processing-flavored architectures. Note: typically batched across
channels (an extension axis — substrate would parameterize on
channel count).

**Position-mixing in MLP-Mixer style, local restriction**:
Same 3-neighbor structure when mixing is restricted to immediate
neighbors. Adjacent variant.

**1D average pooling, kernel size 3**:
```c
out[i] = (L + C + R) / 3.0f;
```
Identical to Jacobi1D averaging.

## Family-aware substrate design parameters

Looking across the family, five extension axes emerge:

| Axis | Examples | Substrate impact |
|---|---|---|
| **Weight vector** (w_L, w_C, w_R) | Box filter, Gaussian, edge detection | Per-fact metadata |
| **Variable coefficients per cell** | Tridiagonal matvec, heterogeneous diffusion | Additional array param |
| **Stencil stride** | Cyclic reduction (stride doubles) | Loop offset parameter |
| **Multi-array reads** | Wave equation (needs prev), Poisson (needs f) | Extra input array params |
| **Stencil width** | 5-point higher-order, 2D extensions | Generalize beyond 3-neighbor |

The first three axes can be added without breaking the family's core
shape. The fourth (multi-array reads) is a per-member detail. The
fifth (wider/2D stencils) is a separate family.

## What this taxonomy enables

**Prioritization of substrate work**: building one well-designed
family-aware Jacobi1D consumer predicate produces immediate capability
for ALL family members listed above. Each is a parameter change away.

**Recognition of "this looks familiar"**: when future curriculum work
or domain extension surfaces a new algorithm, checking whether it's a
family member tells us whether substrate work is needed or just
parameter binding.

**Cross-domain methodology**: principles that emerged from CFD work
(physics-for-physics correctness, bit-identical verification, anchor-
point tests, invariant preservation tests) all transfer to family
members in other domains. PDE solvers have analytical references.
Linear algebra has invariants like spectral radius. Signal processing
has Parseval-type invariants. ML has rotation-invariance and
locality properties.

## Relationship to the CFD beachhead

The CFD beachhead established a *different* stencil family:

**Interface-indexed transmissive 2-neighbor 3-component family** vs.
**Cell-indexed Dirichlet 3-neighbor single-component family**.

Five substantive differences:
- Cell-indexed (this family) vs. interface-indexed (CFD)
- Dirichlet BC (this family) vs. transmissive BC (CFD)
- 3-neighbor (this family) vs. 2-neighbor (CFD)
- Single-component (this family) vs. 3-component (CFD)
- Bound is `1 ≤ i ≤ N-2` (this family) vs. `0 ≤ i ≤ N` (CFD)

The two families are genuinely distinct substrate-design targets. The
CFD beachhead proved the substrate can express interface-indexed 2-
neighbor transmissive 3-component stencils. It does NOT automatically
provide capability for cell-indexed Dirichlet 3-neighbor single-
component stencils. The latter is a separate substrate-design
investment.

The two families could eventually be unified under a more general
`stencil_kernel/3` consumer with full parameterization, but that's
substantively significant substrate-design work and not appropriate
for the F4 reconsideration scope.

## What this informs about F4 reconsideration

F4 jacobi1d was originally framed as "one more kernel" — a PolyBench
GPU benchmark to add. The family analysis reveals F4 is actually the
*representative case* of a 15+-member family spanning PDE solvers,
linear algebra, signal processing, graph algorithms, and ML.

This shifts the substrate-design intuition substantively. Building
Jacobi1D well — with the family's extension axes visible in the
substrate-design choices — produces a foundation for the entire
family. Building it as a one-off doesn't.

Whether the (A) reconsideration adopts a family-aware design or a
one-off design is a separate question, and one to be reconsidered
with a clear mind after this document's content has settled.

## Connection to other methodology

This taxonomy operates under several established methodology
principles:

- **Comprehension over verbatim**: naming the family explicitly is an
  act of substrate-comprehension. The substrate should know that these
  algorithms ARE the same shape, not just that each happens to work.

- **Decomposition into trivial factors**: each family member factors
  into "the family pattern + a specific weight/coefficient/extension."
  The substrate captures the family pattern; each member is then
  trivial.

- **Future substrate-design directions** (the meta-compiler picker,
  fix-flag taxonomy expansions, etc.) all become more interesting
  when applied to a 15-member family rather than a single kernel.

## Future maintenance of this doc

When a new family member is identified, add it to the appropriate
section above. When a new substrate-design parameter is needed (e.g.,
to support tridiagonal variable coefficients), add it to the family-
aware-substrate-design-parameters table.

When a substantively different family is identified (e.g.,
interface-indexed periodic 4-neighbor multi-component for some new
domain), it gets its own doc following this template.

The methodology gains *taxonomic* substrate knowledge alongside the
operational and design knowledge already accumulated.

---

*Authored 2026-05-18 ~15:35 UTC by metayen, per Heath's
"ask ourselves what other applications of matrices also use this
pattern" interjection during the F4 reconnaissance.*
