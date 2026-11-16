# One-Way-Trained ReLU — reproduction artifacts

Companion artifacts for *"One-Way-Trained ReLU: error correction that
only revives"* (Hunnicutt, 2026 — paper/one_way_relu.pdf).

## The three headline numbers, each to its own instrument

Per the collective's attribution discipline, every number names the
instrument that produced it and how to re-run that instrument:

| claim | instrument | re-run |
|---|---|---|
| +0.41 pts vs ReLU, t=9.70, 46/50 (Adam, n=50 paired) | `experiment/factorial_emnist.py` paired-seed campaign | `python3 factorial_emnist.py --arms relu leaky_10 oneway_10 --seeds 50 --epochs 10 --device cuda` |
| 4.5% vs 39.5% squelched (SGD vs Adam, plain ReLU) | same harness, `--optimizer sgd` arm + dead-unit tracking | `python3 factorial_emnist.py --arms relu --seeds 10 --optimizer sgd ...` |
| 5.65× op-level fused speedup; 0-ULP bit-exact | `kernels/` emitted set + region-separated referees | see `kernels/` section below |

## Layout

- `experiment/` — the training testbed.
  - `activations.py` — one-way-leaky as `torch.autograd.Function`
    (the reference implementation the kernels are verified against).
  - `factorial.py` — the original MNIST variant of the harness;
    exports shared helpers (`Net`, `dead_stats`, idx reader) that
    `factorial_emnist.py` imports.
  - `factorial_emnist.py` — paired-seed factorial harness
    (EMNIST-Balanced; `--arms`, `--seeds`, `--optimizer {adam,sgd}`,
    `--amp-log` Adam-amplification instrument, `--save-ckpt`).
  - `classifier_stats.py` — confusion families, macro-F1, ECE.
  - `latent_analysis.py` — participation ratio, Fisher ratio, probes.
  - `campaign2.sh`, `night_campaign.sh`, `sgd_recollect.sh` — the
    exact campaign scripts used for the paper's n=50 corpora.
- `kernels/` — the emitted kernel track.
  - `oneway_backward_cpu.ll` — hand-written LLVM IR prototype
    (spec-by-example; region-verified 0-ULP vs activations.py).
  - `oneway_backward_cuda.cu` — CUDA twin (sm_61).
  - `var2_emit_draft.pl` — the two-input (var2) emission draft that
    generates the canonical kernels from op_expr facts.
  - `backward_referee_cpu.py` — four-region bit-exact referee
    (x>0 / x==+0 / x==−0 / x<0, signed-zero aware).
  - `harness_checks.py` — α-dataflow structural check (check 1) +
    emitted-vs-prototype equivalence (check 2).
  - `signed_zero_oracle_report.json` — Medayek's clean-room ±0
    characterization, including the torch CPU/GPU relu(−0.0)
    divergence measurement.
- `paper/` — LaTeX source, house style, compiled PDF.

## Data

EMNIST-Balanced (Cohen et al. 2017), idx format. Download the gzip
archive from the NIST page
(`https://www.nist.gov/itl/products-and-services/emnist-dataset` —
"binary format as the original MNIST dataset" link) and place these
four files, still gzipped, in `experiment/data_emnist/`:

    emnist-balanced-train-images-idx3-ubyte.gz
    emnist-balanced-train-labels-idx1-ubyte.gz
    emnist-balanced-test-images-idx3-ubyte.gz
    emnist-balanced-test-labels-idx1-ubyte.gz

The loader reads them gzipped; do not extract. No preprocessed data ships here — determinism comes
from seeds: every comparison is paired at bit-identical initialization
(seed k produces identical weights in every arm; verified by direct
tensor comparison).

## Honest notes

- Training for the paper's §3 ran on torch 2.7 eager CUDA; the emitted
  kernels are verified + benchmarked but not in the training loop
  (named as future work in the paper).
- The 5.65× is op-level; ~15% whole-step on this GEMM-dominated net
  (Amdahl); the big-performance claims are extrapolations and marked
  as such in the paper.
- fp32 throughout. Paired seeds 0–49. Pre-registration lives in the
  paper's research-log lineage (bpd/docs/relu_gradient_at_zero.md in
  the development history).
