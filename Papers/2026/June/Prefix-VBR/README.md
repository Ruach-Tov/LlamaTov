# Prefix-VBR: Reproducing Table 1

## Quick Start

```bash
# Run on the included full dataset (25.2M FP16 values, 48 MB):
swipl --stack_limit=16G reproduce_table1_granite.pl embeddings_granite_65580x384.f16

# Or on the 100-row CSV sample for quick testing:
swipl --stack_limit=2G reproduce_table1_granite.pl embeddings_sample_100.csv
```

## Files

- `reproduce_table1_granite.pl` — SWI-Prolog program that reads embedding data, applies per-block normalization (block_size=32), and prints the Table 1 distribution histogram. Accepts both FP16 binary (`.f16`) and CSV (`.csv`) input.
- `embeddings_granite_65580x384.f16` — Full dataset: 65,580 granite-embedding vectors × 384 dimensions = 25,182,720 values stored as IEEE 754 half-precision (2 bytes each, little-endian). 48 MB.
- `embeddings_sample_100.csv` — 100-row CSV sample (462 KB) for quick verification.
- `extract_embeddings.sh` — Helper script to re-extract from PostgreSQL (if available).

## Expected Output (full dataset)

```
TABLE 1: Per-block normalized |w_norm|
  Source: granite-embedding, 384 dims
  Block size: 32

  Range          Fraction    Cumulative
  [0, 32)        55.9%       55.9%
  [32, 64)       27.4%       83.3%
  [64, 96)       10.0%       93.3%
  [96, 128)       6.7%      100.0%
```

## Data Format

The `.f16` file contains raw IEEE 754 half-precision floats in little-endian byte order. Each embedding is 384 consecutive FP16 values (768 bytes). The file contains 65,580 embeddings laid out sequentially with no headers or separators.

To read in Python: `numpy.fromfile('embeddings_granite_65580x384.f16', dtype=numpy.float16).reshape(65580, 384)`

## Requirements

- SWI-Prolog 9.x with `--stack_limit=16G` for the full dataset
