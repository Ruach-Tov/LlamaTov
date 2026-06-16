# Prefix-VBR: Reproducing Table 1

## Quick Start

```bash
# Run on the included 100-row sample:
swipl --stack_limit=2G reproduce_table1_granite.pl embeddings_sample_100.csv

# Run on the full 65,580-row dataset (requires PostgreSQL with wordnet_embeddings_granite):
psql -d claude_conversations -t -A -c \
  "SELECT replace(replace(embedding::text,'[',''),']','') \
   FROM wordnet_embeddings_granite" > embeddings_full.csv

swipl --stack_limit=16G reproduce_table1_granite.pl embeddings_full.csv
```

## Files

- `reproduce_table1_granite.pl` — SWI-Prolog program that reads embedding CSV, applies per-block normalization (block_size=32), and prints the Table 1 distribution histogram.
- `embeddings_sample_100.csv` — 100 granite-embedding vectors (384 floats each, 38,400 values) for quick verification. The full dataset (65,580 rows, 25,182,720 values) is extracted from PostgreSQL as shown above.
- `extract_embeddings.sh` — Helper script to extract the full dataset.

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

## Requirements

- SWI-Prolog 9.x
- PostgreSQL with `wordnet_embeddings_granite` table (for full dataset extraction)
- The `granite-embedding` model embeddings were generated using IBM's granite-embedding (62 MB, 384 dimensions) via Ollama
