# GGUF-as-BPD: Staged implementation plan

Each stage is a triple of (parse to object) + (print summary) + (handle a
query). Each stage produces a complete tool over a smaller subset of the
GGUF format than the next. This is per Heath's directive (2026-05-03)
that staging by capability triple, rather than by structural traversal,
keeps `ggufq` working at every checkpoint.

## Stage 0 — Header (✅ complete, 2026-05-03)

**Object format**: `GGUFHeader` (4 fields)
**Summary**: human-readable header summary with validity markers
**Query support**:
  - `.` (identity)
  - `.FIELD` (field access for each of magic/version/tensor_count/metadata_kv_count)
  - `keys`
  - `--summary`, `--json` flags

**Files**: `gguf.bpd`, `generate.py`, `ggufq.py`, `test_stage0.py`, `output/`
**Conformance**: agrees with `papers/kan-acceleration/gguf_parser.py` on header fields.

## Stage 1 — Metadata KV section

**Object format**: + `metadata: dict[str, MetadataValue]`
**Summary**: + key listing with array-elision (e.g., `tokenizer.ggml.tokens` shown as `<array string; len=32000, sample=[...]>`)
**Query support**:
  - `.metadata.<key>` access
  - `.metadata | keys`
  - `.metadata | length`
  - String value formatting
  - Array value elision

**BPD additions**: length-prefixed strings, discriminated values via `value_type`, the 13 GGUF metadata value types
**Predicate vocabulary needed**:
  - `field(... length_prefixed_utf8(le_u64) ...)`
  - `discriminated(... by(:value_type), cases(...))`
  - Variable-length section declaration (no `fixed_width` flag)

**End state**: `ggufq` covers ~80% of mavchin's `gguf_parser.py` summary output.

## Stage 2 — Tensor info section

**Object format**: + `tensors: list[TensorInfo]`
**Summary**: + tensor count + type distribution + (optionally truncated) tensor list
**Query support**:
  - `.tensors[]` enumeration
  - `.tensors[].name`, `.tensors[].shape`, `.tensors[].type`
  - `.tensors[].size_bytes` (computed)
  - `.tensors[] | select(.type == "Q4_K")` filter
  - Format-string output: `'{"{i}:\t", .tensors.[].{i}}'`

**BPD additions**: variable-arity records, `repeated(N, T)` where N is a previously-read field
**Predicate vocabulary needed**:
  - `repeated(count_from=:tensor_count, item=:tensor_info_record)`
  - `record(:tensor_info_record, fields=[...])`

**End state**: `ggufq` covers full GGUF metadata-and-shape querying.

## Stage 3 — Tensor data access

**Object format**: + lazy `read_tensor(name) → bytes` capability
**Summary**: + total tensor data size, alignment status
**Query support**:
  - `.tensors[name].data` returning raw bytes
  - `--dequantize` flag (optional) for converting Q4_K/Q6_K/F16 to float arrays

**BPD additions**: alignment vocabulary, computed offsets, type-dependent layout per quantization scheme

**End state**: full file traversal; PyTorch state_dict construction becomes downstream.

## Stage 4 — Built-in canonicality queries

**Object format**: (no change)
**Summary**: (no change)
**Query support**:
  - `--file-name-as-canonical-from-metadata` flag
  - `--check-architecture-consistency`
  - `--check-tensor-name-pattern-consistency`
  - Other named queries that compose primitives

**End state**: workflow-friendly canonical checks for batch validation.

## Stage 5 — Multiple generator backends

**Object format**: (per-backend; same shape)
**Summary**: (per-backend)
**Query support**: (per-backend)

Generate parsers in:
- Haskell (cleanest idiomatic shape per design doc)
- Rust (deployment performance)
- (other backends as adaptation requirements stabilize)

Run mavchin's hand-written parser as conformance reference; assert output
equivalence across backends.

**End state**: polyarchitecture pattern fully established at level 0.

---

## Things deliberately NOT in any stage

- **General jq compatibility**: ggufq imitates jq where syntax overlaps,
  differs explicitly where it must. Not a goal to be jq-compatible for
  arbitrary jq programs.
- **Dequantization beyond what's needed for PyTorch loading**: GGUF's
  many quantization formats (Q4_K_M, Q5_K_S, Q6_K, Q8_0, etc.) all
  require their own dequantization. We use existing libraries (`gguf-py`,
  `llama-cpp-python`) rather than implementing all of them.
- **GGUF write support**: ggufq is read-only. Writing GGUF files is a
  separate concern (and arguably should be done via a different tool).
- **GGUF v1/v2 support**: only v3 is currently supported. The BPD
  declaration explicitly enforces this. Adding v1/v2 support would be
  version-conditional field declarations, deferred until needed.
