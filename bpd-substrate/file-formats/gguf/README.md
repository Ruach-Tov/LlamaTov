# GGUF format declared in BPD — Stage 0

This directory contains the first level-0 BPD declaration: the GGUF
binary file format expressed in Boundary Provenance DSL.

**Stage 0 scope**: the 24-byte fixed header only.
**Status**: Complete and tested. See `STAGES.md` for what comes next.

## Files

```
gguf.bpd                        ← THE source of truth (BPD declaration)
generate.py                     ← Generator: gguf.bpd → output/gguf_reader.py
ggufq.py                        ← jq-style query tool
test_stage0.py                  ← pytest tests including conformance
build.sh                        ← regenerate + test smoke script
README.md                       ← this file
STAGES.md                       ← roadmap and what's deferred

output/
├── gguf_reader.py              ← generated parser (DO NOT EDIT)
└── build/
    └── ggufq                   ← executable wrapper
```

## Quick usage

Build (regenerate + test):

```sh
./build.sh
```

Query a GGUF file:

```sh
output/build/ggufq path/to/model.gguf --summary
output/build/ggufq path/to/model.gguf '.tensor_count'
output/build/ggufq path/to/model.gguf '.version'
output/build/ggufq path/to/model.gguf '.'           # full header as JSON
output/build/ggufq path/to/model.gguf 'keys'        # field names
```

## What the BPD declaration looks like

The full source of truth is `gguf.bpd`. Excerpt:

```prolog
format(:gguf).
format_endianness(:gguf, little).
format_section(:gguf, :header, order=0).

section(:header, fixed_width).
section_size(:header, byte_count(24)).

field(:header, :magic,
      byte_offset(0),
      magic_constant(bytes(4)),
      [must_equal(byte_sequence(0x47, 0x47, 0x55, 0x46))]).

field(:header, :version,
      byte_offset(4),
      format_version(cardinal(u32)),
      [must_equal(3)]).

field(:header, :tensor_count,
      byte_offset(8),
      tensor_count(count(cardinal(u64))),
      [must_be_lt(100000)]).

field(:header, :metadata_kv_count,
      byte_offset(16),
      metadata_kv_count(count(cardinal(u64))),
      [must_be_lt(100000)]).
```

The dimensional types (`tensor_count(count(cardinal(u64)))`) are stacks
of refinements over a substrate type. The substrate (`u64`) describes
the byte layout; the wrappers (`cardinal`, `count`, `tensor_count`)
add semantic content. See `../../design/file-formats-as-bpd.md` for
the full vocabulary.

## How the generator works

`generate.py` is a minimal level-0 BPD generator. It:

1. Tokenizes and parses `gguf.bpd` using a small Prolog-style fact loader
2. Extracts `section_size`, `field`, and constraint facts
3. Emits Python code at `output/gguf_reader.py`

The generated code is a `Gguf` outer dataclass + `GGUFHeader`
per-section dataclass + `parse_gguf()` orchestrator + `parse_gguf_header()`
function using `struct.unpack`. Constraints from the BPD declaration
become runtime validity checks (magic mismatch, wrong version, count
exceeding safety bounds → `GGUFParseError`).

Future stages will:
- Replace this minimal fact-loader with the full BPD parser at
  `must_close/boundary_dsl/parser.py`
- Add additional generator backends (Haskell, Rust, etc.) following the
  polyarchitecture pattern from `mcp_bridge.bnd`

## Conformance

`test_stage0.py::TestConformance` verifies that the generated parser
produces the same field values as `papers/kan-acceleration/gguf_parser.py`
(mavchin's hand-written reference parser) on real GGUF files from
Ollama's blob store. **Different implementations of the same BPD
declaration must agree.** This is the polyarchitecture conformance
pattern (memory `22a7ccc2`) applied at level 0.

## Regenerating after BPD changes

If you edit `gguf.bpd`:

```sh
./build.sh        # regenerate, test, report
```

Or manually:

```sh
python3 generate.py
python3 -m pytest test_stage0.py -v
```

## See also

- `../../design/file-formats-as-bpd.md` — design document for level-0 BPD
- `../wordnet/` — Stage 0 declaration coming next
- `papers/kan-acceleration/gguf_parser.py` — operational reference parser
- `https://github.com/ggml-org/ggml/blob/master/docs/gguf.md` — GGUF format spec
