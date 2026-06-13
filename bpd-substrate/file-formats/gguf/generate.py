# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""generate.py — BPD-to-Python generator for GGUF.

Reads gguf.bpd through the canonical infrastructure (bpd_parser.py +
bpd_to_prolog.py) and emits Python code that parses a GGUF header
into a typed object.

This is the second iteration of this generator. The first iteration
(2026-05-03) had a hand-rolled Prolog-style fact loader that
duplicated capabilities present in the canonical infrastructure.
That duplication was technical debt; this iteration retires it.

Architecture:
    gguf.bpd                          (BPD source — source of truth)
        │
        │ bpd_parser.py (Lark)
        ▼
    AST (Program with annotated clauses)
        │
        │ bpd_to_prolog.py
        ▼
    output/gguf.pl                    (Prolog IR — queryable substrate)
        │
        │ bpd_query.py (subprocess to swipl)
        ▼
    Header field facts as JSON dicts
        │
        │ this generator
        ▼
    output/gguf_reader.py             (generated Python parser)

Stage 0 scope only — emits header parsing code (24 bytes, four
fields). Stage 1.5 (next session) extends to metadata KV processing.

The conformance test in test_stage0.py validates that the generated
parser agrees with mavchin's hand-written reference parser on real
GGUF files from Ollama's blob store.
"""
from __future__ import annotations

import sys
from pathlib import Path

# The canonical infrastructure lives in must_close/boundary_dsl
HERE = Path(__file__).resolve().parent
BPD_DIR = HERE.parent.parent  # file-formats/gguf → file-formats → boundary_dsl
sys.path.insert(0, str(BPD_DIR))

from bpd_parser import parse_bpd_file  # noqa: E402
from bpd_query import BPDQuery  # noqa: E402
from bpd_to_prolog import compile_bpd_to_prolog  # noqa: E402
from verbs import (  # noqa: E402
    # Only SectionDataclassVerb is constructed directly in this module
    # — the orchestrator re-emits the primary section's dataclass with
    # GGUF-specific extra_methods. All other verbs are constructed by
    # factories in verb_factory.py.
    SectionDataclassVerb,
)
from generators.source_code.python.bpd_to_python_file_reader import (  # noqa: E402
    render_python_struct_unpack,
)
from verb_factory import (  # noqa: E402
    make_section_layout_verbs,
    make_field_byte_equality_verbs,
    make_field_numeric_constraint_verbs,
    make_section_dataclass_verbs,
    make_section_length_check_verbs,
)


# ─── Code template ──────────────────────────────────────────────────


PYTHON_TEMPLATE = '''"""{format_name}_reader.py — generated from {format_name}.bpd by generate.py.

DO NOT EDIT BY HAND. Regenerate with `python generate.py`.
Source declaration: {source_path}
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path


{section_size_constants}
{field_byte_eq_constants}
{numeric_constraint_constants}


class {error_class}(Exception):
    """Raised when a {format_upper} file fails to parse or violates a constraint."""


# ─── Generated helper functions (from BPD declarations) ─────────────
#
# Each helper validates one structural property declared in {format_name}.bpd.
# These are at module scope (not nested) so they can be unit-tested
# independently and so call sites do not need to know the helper's
# indentation context.
#
# Future: helpers will be refactored into methods on per-(format,
# section) classes (e.g., formats.{format_name}.{primary_section}.check_length()).
# The current underscore-flat names migrate to the dotted form by replacing
# underscores with dots and moving bodies into class methods.

{section_helper_functions}


{field_byte_eq_helper_functions}


{numeric_constraint_helpers}


# ─── Per-section dataclasses ────────────────────────────────────────


{primary_section_dataclass_def}


# ─── Outer parsed-format container ──────────────────────────────────
#
# Holds the parsed result of every format_section in declared order.
# For Stage 0 (header only), this is just the header. Future stages
# add metadata_kv_section, tensor_info_section, tensor_data_section
# as additional fields as those sections are implemented.


@dataclass(frozen=True)
class {format_dataclass}:
    """Parsed {format_upper} file — Stage 0 (primary section only).

    Future stages will add fields for additional format_sections as
    they are implemented (metadata_kv_section, tensor_info_section,
    tensor_data_section, etc.).
    """
    {primary_section}: {primary_section_dataclass}


# ─── Per-section parsing functions ──────────────────────────────────
#
# Each format_section in {format_name}.bpd produces one parse_*
# function. The orchestrator (parse_{format_name}) calls them in
# declared order and assembles the {format_dataclass} result.
#
# Per-section function interface (β convention from this commit):
#   - Fixed-width sections take `raw: bytes` and `path` (path used for
#     error messages only). The orchestrator reads the section's bytes
#     and passes them in.
#   - Variable-width sections take `f: BinaryIO` and `path`, reading
#     from the current stream position. (None in GGUF Stage 0; this
#     comment documents the future shape for PNG and others.)


def parse_{format_name}_{primary_section}(raw: bytes, path) -> {primary_section_dataclass}:
    """Parse the {header_size}-byte {format_upper} {primary_section} from raw bytes.

    Raises {error_class} on:
      - File too short
      - Magic bytes mismatch
      - Unsupported version
      - Counts exceeding safety bounds
    """
    p = Path(path)

{section_length_checks}

{field_byte_eq_calls}

{struct_unpack_calls}

{numeric_constraint_calls}

{primary_section_construction}


# ─── Top-level orchestrator ─────────────────────────────────────────


def parse_{format_name}(path: str | Path) -> {format_dataclass}:
    """Read and parse a {format_upper} file.

    Calls per-section parse functions in declared order and assembles
    the result. For Stage 0 this is just the {primary_section}.
    """
    p = Path(path)
    with p.open("rb") as f:
        raw_{primary_section} = f.read(HEADER_SIZE)

    {primary_section} = parse_{format_name}_{primary_section}(raw_{primary_section}, p)

    return {format_dataclass}({primary_section}={primary_section})
'''


# ─── Query helpers ─────────────────────────────────────────────────


def query_header_facts(pl_path: Path) -> dict:
    """Query the Prolog IR for header section facts.

    Returns a dict with keys:
        section_size:        int (declared byte_count)
        magic_bytes:         list[int] (the byte sequence)
        supported_version:   int
        tensor_count_max:    int (must_be_lt safety bound)
        metadata_kv_count_max: int
    """
    q = BPDQuery([str(pl_path)])

    # Section size (header is fixed_width with byte_count(N))
    size_result = q.query(
        "findall(N, section_size(header, byte_count(N)), Result)",
        result_key="size",
    )
    if not size_result or len(size_result) != 1:
        raise ValueError(
            f"expected exactly one section_size(header, byte_count(N)) "
            f"fact; got {size_result!r}"
        )
    section_size = size_result[0]

    # Each field's must_equal / must_be_lt constraints
    # Field structure: field(header, FName, byte_offset(O), Type, Constraints)
    fields_result = q.query(
        "findall("
        "[FName, Cs], "
        "field(header, FName, _, _, Cs), "
        "Result"
        ")",
        result_key="fields",
    )

    magic_bytes = None
    supported_version = None
    tensor_count_max = None
    metadata_kv_count_max = None

    for fname, constraints in fields_result:
        # constraints is a list of constraint terms (as JSON dicts)
        for c in constraints:
            if not isinstance(c, dict):
                continue
            cfunctor = c.get("functor")
            cargs = c.get("args", [])
            if cfunctor == "must_equal" and fname == "magic":
                # magic is byte_sequence(B1, B2, ...) inside must_equal
                inner = cargs[0] if cargs else None
                if isinstance(inner, dict) and inner.get("functor") == "byte_sequence":
                    magic_bytes = list(inner.get("args", []))
            elif cfunctor == "must_equal" and fname == "version":
                if cargs and isinstance(cargs[0], int):
                    supported_version = cargs[0]
            elif cfunctor == "must_be_lt":
                if cargs and isinstance(cargs[0], int):
                    if fname == "tensor_count":
                        tensor_count_max = cargs[0]
                    elif fname == "metadata_kv_count":
                        metadata_kv_count_max = cargs[0]

    if magic_bytes is None:
        raise ValueError("could not find magic byte sequence in BPD constraints")
    if supported_version is None:
        raise ValueError("could not find version constraint in BPD")
    if tensor_count_max is None:
        raise ValueError("could not find tensor_count safety bound in BPD")
    if metadata_kv_count_max is None:
        raise ValueError(
            "could not find metadata_kv_count safety bound in BPD"
        )

    return {
        "section_size": section_size,
        "magic_bytes": magic_bytes,
        "supported_version": supported_version,
        "tensor_count_max": tensor_count_max,
        "metadata_kv_count_max": metadata_kv_count_max,
    }


# ─── Generation ─────────────────────────────────────────────────────


def generate_python(
    facts: dict,
    format_metadata: dict,
    section_emissions: list[dict],
    field_byte_eq_emissions: list[dict],
    struct_unpack_emissions: list[dict],
    numeric_constraint_emissions: list[dict],
    dataclass_emissions: list[dict],
    source_path: Path,
    out_path: Path,
) -> None:
    """Render the Python code template using extracted facts and emitter output.

    Args:
        facts: scalar facts extracted from the IR (constants, etc.).
        format_metadata: format-level metadata (format_name, format_upper,
            primary_section, error_class, primary_section_dataclass,
            format_dataclass).
            Produced by query_format_metadata.
        section_emissions: list of per-section dicts each containing
            'constant', 'helper_function', 'call_site'.
            Produced by emit_section_length_check.
        field_byte_eq_emissions: list of per-field dicts each containing
            'constant', 'helper_function', 'call_site'.
            Produced by emit_field_byte_equality_check.
        source_path: the .bpd source file path (for the header comment).
        out_path: the .py output file path.
    """
    # Module-level: section size constant declarations
    section_size_constants = "\n".join(
        e["constant"] for e in section_emissions
    )
    # Module-level: section helper function definitions (no indent)
    section_helper_functions = "\n\n\n".join(
        e["helper_function"] for e in section_emissions
    )
    # Inside read function body (4-space indent for function-body context)
    section_length_checks = "\n".join(
        f"    {e['call_site']}" for e in section_emissions
    )

    # Module-level: field byte-equality constants (e.g., GGUF_MAGIC = ...)
    field_byte_eq_constants = "\n".join(
        e["constant"] for e in field_byte_eq_emissions
    )
    # Module-level: field byte-equality helper functions (no indent)
    field_byte_eq_helper_functions = "\n\n\n".join(
        e["helper_function"] for e in field_byte_eq_emissions
    )
    # Inside read function body (4-space indent)
    field_byte_eq_calls = "\n".join(
        f"    {e['call_site']}" for e in field_byte_eq_emissions
    )

    # Inside primary-section parser body (4-space indent). The
    # struct.unpack call is multi-line, so each line gets the indent
    # prefix.
    struct_unpack_calls_parts = []
    for entry in struct_unpack_emissions:
        if entry["section_name"] != format_metadata["primary_section"]:
            # For Stage 0, only the primary section's struct.unpack is
            # placed in the primary-section parser body. Other sections'
            # unpacks belong in their own parser bodies (future work).
            continue
        for line in entry["emission"]["call_site"].split("\n"):
            struct_unpack_calls_parts.append(f"    {line}")
    struct_unpack_calls = "\n".join(struct_unpack_calls_parts)

    # Module-level: numeric constraint constants (e.g.,
    # FORMAT_VERSION_EXPECTED, TENSOR_COUNT_MAX). Deduplicate by
    # constant name — multiple fields with the same dimension share
    # one constant.
    seen_constants = set()
    numeric_constraint_constants_lines = []
    for emission in numeric_constraint_emissions:
        line = emission["emission"]["constant"]
        if line and line not in seen_constants:
            seen_constants.add(line)
            numeric_constraint_constants_lines.append(line)
    numeric_constraint_constants = "\n".join(numeric_constraint_constants_lines)

    # Module-level: numeric constraint helper functions (no indent)
    numeric_constraint_helpers = "\n\n\n".join(
        e["emission"]["helper_function"]
        for e in numeric_constraint_emissions
    )

    # Inside primary-section parser body (4-space indent). For Stage 0,
    # only constraints on the primary section's fields are placed here;
    # other sections' constraints belong in their own parser bodies
    # (future work, parallel to struct_unpack_calls handling).
    numeric_constraint_calls_parts = []
    for emission in numeric_constraint_emissions:
        if emission["section_name"] != format_metadata["primary_section"]:
            continue
        numeric_constraint_calls_parts.append(
            f"    {emission['emission']['call_site']}"
        )
    numeric_constraint_calls = "\n".join(numeric_constraint_calls_parts)

    # Verb-C: per-section dataclass declaration and construction call.
    # The dataclass emissions populate two placeholders:
    #   {primary_section_dataclass_def} — the @dataclass declaration
    #   {primary_section_construction}  — the return Foo(...) call
    # Convenience methods (is_valid_magic, is_supported_version) are
    # injected via extra_methods on the dataclass — they're currently
    # GGUF-specific and remain template-literal until T-16 generalizes
    # them across formats.
    primary_section_dataclass_def = ""
    primary_section_construction = ""
    if dataclass_emissions:
        # For Stage 0, expect exactly one emission (the primary section).
        # If multiple sections eventually emit dataclasses, the
        # orchestrator joins them; here we take the first/primary.
        primary = dataclass_emissions[0]

        # Build extra_methods for the convenience accessors. Currently
        # GGUF-shaped; T-16 will generalize via per-constraint emission.
        format_upper = format_metadata["format_upper"]
        extra_methods_lines = [
            "def is_valid_magic(self) -> bool:",
            f"    return self.magic == {format_upper}_MAGIC",
            "",
            "def is_supported_version(self) -> bool:",
            "    return self.version == FORMAT_VERSION_EXPECTED",
        ]
        extra_methods = "\n".join(extra_methods_lines)

        # Re-emit the dataclass with extra_methods (the query emits
        # without methods; we inject here at the orchestrator level
        # because the methods are currently GGUF-specific).
        primary["dataclass_emission"] = SectionDataclassVerb(
            format_name=format_metadata["format_name"],
            section_name=primary["section_name"],
            section_size=primary["section_size"],
            fields=tuple(primary["fields"]),
            primary_section_dataclass=format_metadata["primary_section_dataclass"],
            format_upper=format_upper,
            extra_methods=extra_methods,
        ).emit()

        primary_section_dataclass_def = primary["dataclass_emission"]["helper_function"]
        # Construction call: indent each line by 4 spaces (function body)
        construction = primary["construction_emission"]["call_site"]
        primary_section_construction = "\n".join(
            f"    {line}" for line in construction.split("\n")
        )

    code = PYTHON_TEMPLATE.format(
        source_path=source_path,
        supported_version=facts["supported_version"],
        header_size=facts["section_size"],
        tensor_count_max=facts["tensor_count_max"],
        metadata_kv_count_max=facts["metadata_kv_count_max"],
        section_size_constants=section_size_constants,
        section_helper_functions=section_helper_functions,
        section_length_checks=section_length_checks,
        field_byte_eq_constants=field_byte_eq_constants,
        field_byte_eq_helper_functions=field_byte_eq_helper_functions,
        field_byte_eq_calls=field_byte_eq_calls,
        struct_unpack_calls=struct_unpack_calls,
        numeric_constraint_constants=numeric_constraint_constants,
        numeric_constraint_helpers=numeric_constraint_helpers,
        numeric_constraint_calls=numeric_constraint_calls,
        primary_section_dataclass_def=primary_section_dataclass_def,
        primary_section_construction=primary_section_construction,
        # Format-derived names (parameterized so the same template applies
        # across formats; for GGUF these resolve to the existing API
        # surface preserving downstream compatibility):
        format_name=format_metadata["format_name"],
        format_upper=format_metadata["format_upper"],
        primary_section=format_metadata["primary_section"],
        error_class=format_metadata["error_class"],
        primary_section_dataclass=format_metadata["primary_section_dataclass"],
        format_dataclass=format_metadata["format_dataclass"],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(code, encoding="utf-8")
    print(f"Generated {out_path}")


# ─── Driver ─────────────────────────────────────────────────────────


def query_section_emissions(pl_path: Path) -> list[dict]:
    """Query the IR for section_size facts and emit code blocks.

    Phase 3.e migration: delegates to make_section_length_check_verbs
    factory. Returns the per-section emission triples that the orchestrator
    splices into template placeholders.
    """
    return [
        verb.emit() for verb in make_section_length_check_verbs(pl_path)
    ]


def query_format_metadata(pl_path: Path) -> dict:
    """Query the Prolog IR for format-level metadata used to parameterize
    the code template (format name, primary section, derived class names).

    The primary section is the format_section/3 with the lowest order — for
    GGUF it's :header at order(0); for PNG it's :signature at order(0).

    Returns a dict with:
        format_name:               lowercase atom (e.g., 'gguf', 'png')
        format_upper:              uppercase form (e.g., 'GGUF')
        format_camel:              CamelCase outer container (e.g., 'Gguf')
        primary_section:           atom name of the lowest-order format_section
        primary_section_camel:     CamelCase form (e.g., 'Header')
        error_class:               derived (e.g., 'GGUFParseError')
        primary_section_dataclass: derived (e.g., 'GGUFHeader')
        format_dataclass:          derived (e.g., 'Gguf')

    Naming conventions used here:
        format_upper              = format_name.upper() (works for short acronym
                                    formats like GGUF, PNG, GIF, ELF)
        format_camel              = CamelCase(format_name)
        error_class               = "{format_upper}ParseError"
        primary_section_dataclass = "{format_upper}{primary_section_camel}"
        format_dataclass          = format_camel

    The generated parser API:
        parse_{format_name}(path) -> {format_dataclass}
        parse_{format_name}_{primary_section}(raw, path) -> {primary_section_dataclass}

    The orchestrator (parse_{format_name}) reads the file and calls each
    per-section parser in declared order, assembling the result into a
    {format_dataclass} instance. Per-section parsers take the bytes for
    their section (fixed-width) or a file handle (variable-width).
    """
    q = BPDQuery([str(pl_path)])

    format_result = q.query(
        "findall(F, format(F), Result)",
        result_key="formats",
    )
    if not format_result or len(format_result) != 1:
        raise ValueError(
            f"expected exactly one format/1 fact; got {format_result!r}"
        )
    format_name = format_result[0]

    # Find the format_section/3 with the lowest order — that's the primary
    # entry section. For GGUF: :header at order(0). For PNG: :signature at
    # order(0). The format may have multiple format_sections; we pick the
    # one to feature in the entry function and dataclass name.
    sections_result = q.query(
        "findall("
        "[Section, Order], "
        "format_section(_, Section, order(Order)), "
        "Result"
        ")",
        result_key="format_sections",
    )
    if not sections_result:
        raise ValueError(
            f"expected at least one format_section/3 fact for format "
            f"'{format_name}'; got none"
        )
    sections_sorted = sorted(sections_result, key=lambda s: s[1])
    primary_section = sections_sorted[0][0]

    format_upper = format_name.upper()
    # CamelCase forms used in class names. format_camel handles formats
    # whose lowercase atom contains underscores (e.g., 'word_net' →
    # 'WordNet'); for short acronyms like 'gguf' it produces 'Gguf'.
    primary_section_camel = primary_section.replace("_", " ").title().replace(" ", "")
    format_camel = format_name.replace("_", " ").title().replace(" ", "")
    error_class = f"{format_upper}ParseError"
    # Per-section dataclass: e.g., GGUFHeader, PNGSignature.
    # Convention: {format_upper}{primary_section_camel}
    primary_section_dataclass = f"{format_upper}{primary_section_camel}"
    # Outer format container: e.g., Gguf, Png, WordNet.
    # Convention: {format_camel} (just the format atom in CamelCase).
    # This holds per-section parsed results assembled by parse_{format}.
    format_dataclass = format_camel

    return {
        "format_name": format_name,
        "format_upper": format_upper,
        "format_camel": format_camel,
        "primary_section": primary_section,
        "primary_section_camel": primary_section_camel,
        "error_class": error_class,
        "primary_section_dataclass": primary_section_dataclass,
        "format_dataclass": format_dataclass,
    }


def query_field_byte_equality_emissions(pl_path: Path) -> list[dict]:
    """Query the IR for fields with must_equal(byte_sequence(...))
    constraints and emit code blocks.

    Phase 3.b migration: this function now delegates to the
    make_field_byte_equality_verbs factory rather than walking
    field/5 facts inline. The factory contains the previously-duplicated
    constraint-walking logic; this function only does the verb→emission
    conversion.

    Returns:
        list of emission triples (constant, helper_function, call_site).
        Each emission corresponds to one byte-equality constraint.
    """
    return [verb.emit() for verb in make_field_byte_equality_verbs(pl_path)]


def query_struct_unpack_emissions(pl_path: Path) -> list[dict]:
    """Query the IR for sections whose fields can be unpacked via
    struct.unpack and produce a per-section emission for each.

    Phase 3.a migration: this function now delegates to the verb-program
    pipeline (make_section_layout_verbs + render_python_struct_unpack)
    rather than walking field/5 facts inline. The previous ~180-line
    implementation has been replaced with the ~25-line verb-based path.

    A field is eligible for struct.unpack iff:
      - Its type's struct_type_code_for resolution succeeds (primitive
        types like u32, u64 — possibly wrapped in dimensional functors).
      - It does NOT have a must_equal(byte_sequence(...)) constraint
        (those are handled by emit_field_byte_equality_check).

    Both filters are applied automatically by render_python_struct_unpack:
    byte_sequence types return None from struct_type_code_for, and
    non-primitive types do too — they get skipped during rendering.

    Returns a list of dicts, each with:
        'section_name': the section atom
        'emission': the triple-return dict from emit_struct_unpack

    The orchestrator collects emissions per section and inserts the
    call_site into the section's parser body at the correct position.

    Architecture: this function preserves byte-equivalent output (verified
    by test_section_layout_render_equivalence.py) while moving the
    field-walking algorithm into the Prolog rule section_layout/2 and
    its Python wrapper query_section_layouts(). Future targets (CUDA,
    assembly) will share the same data-pipeline by providing their own
    render_* functions on the SectionLayoutVerb.
    """
    # Build a SectionLayoutVerb per section via the Phase 2.3c.viii factory.
    # This walks the IR through the Prolog rule section_layout/2 and
    # returns one verb per fixed-width section that passes structural
    # validation (gaps, overlaps, undershoot, etc. surface in
    # provenance_checks before we get here).
    verbs_by_section = make_section_layout_verbs(pl_path)

    # Render each verb via the Phase 2.4.c renderer. Sections whose
    # rendered output has an empty call_site (e.g., a section with no
    # struct-eligible fields) are skipped — same shape as the legacy
    # path's "no emission" case.
    emissions = []
    for section_name in sorted(verbs_by_section.keys()):
        verb = verbs_by_section[section_name]
        emission = render_python_struct_unpack(verb)
        if emission["call_site"]:
            emissions.append({
                "section_name": section_name,
                "emission": emission,
            })
    return emissions


def query_field_numeric_constraint_emissions(pl_path: Path) -> list[dict]:
    """Query the IR for field constraints and emit per-constraint
    helper triples.

    Phase 3.c migration: delegates to make_field_numeric_constraint_verbs
    factory rather than walking field/5 inline.

    Returns:
        list of dicts with keys section_name, field_name, constraint_kind,
        emission (the triple-return dict).
    """
    return [
        {
            "section_name": entry["section_name"],
            "field_name": entry["field_name"],
            "constraint_kind": entry["constraint_kind"],
            "emission": entry["verb"].emit(),
        }
        for entry in make_field_numeric_constraint_verbs(pl_path)
    ]


def query_section_dataclass_emissions(pl_path: Path) -> list[dict]:
    """Query the IR for sections and emit dataclass + construction triples.

    Phase 3.d migration: delegates to make_section_dataclass_verbs factory.

    Returns:
        list of dicts with section_name, dataclass_emission,
        construction_emission, section_size, fields.
    """
    return [
        {
            "section_name": entry["section_name"],
            "dataclass_emission": entry["dataclass_verb"].emit(),
            "construction_emission": entry["construction_verb"].emit(),
            "section_size": entry["section_size"],
            "fields": entry["fields"],
        }
        for entry in make_section_dataclass_verbs(pl_path)
    ]



def main() -> int:
    source_path = HERE / "gguf.bpd"
    pl_path = HERE / "output" / "gguf.pl"
    py_path = HERE / "output" / "gguf_reader.py"

    if not source_path.exists():
        print(f"ERROR: {source_path} not found", file=sys.stderr)
        return 1

    # 1. Parse to AST (validates BPD syntax; raises if malformed)
    program = parse_bpd_file(str(source_path))
    print(f"Parsed gguf.bpd: {len(program.clauses)} clauses")

    # 2. Compile to Prolog IR
    pl_path.parent.mkdir(parents=True, exist_ok=True)
    compile_bpd_to_prolog(str(source_path), str(pl_path))
    print(f"Compiled to {pl_path}")

    # 3. Query the IR for header facts
    facts = query_header_facts(pl_path)
    print(
        f"Extracted facts: section_size={facts['section_size']}, "
        f"magic={facts['magic_bytes']}, "
        f"version={facts['supported_version']}"
    )

    # 4. Query section_size facts and produce per-section emissions
    section_emissions = query_section_emissions(pl_path)
    print(
        f"Section emissions: {len(section_emissions)} fixed-width sections"
    )

    # 4b. Query fields with must_equal(byte_sequence(...)) constraints
    field_byte_eq_emissions = query_field_byte_equality_emissions(pl_path)
    print(
        f"Field byte-equality emissions: {len(field_byte_eq_emissions)} fields"
    )

    # 4c. Query struct.unpack-eligible field runs per section.
    struct_unpack_emissions = query_struct_unpack_emissions(pl_path)
    print(
        f"struct.unpack emissions: {len(struct_unpack_emissions)} sections"
    )

    # 4ca. Query numeric constraint emissions (must_equal, must_be_lt, etc.)
    numeric_constraint_emissions = query_field_numeric_constraint_emissions(pl_path)
    print(
        f"Numeric constraint emissions: {len(numeric_constraint_emissions)} constraints"
    )

    # 4cb. Query per-section dataclass emissions (verb-C: dataclass
    # declaration + construction call).
    dataclass_emissions = query_section_dataclass_emissions(pl_path)
    print(
        f"Dataclass emissions: {len(dataclass_emissions)} sections"
    )

    # 4d. Query format-level metadata for template parameterization.
    # Produces format_name/format_upper/error_class/etc. so the same
    # PYTHON_TEMPLATE applies across formats.
    format_metadata = query_format_metadata(pl_path)
    print(
        f"Format metadata: {format_metadata['format_name']} "
        f"(orchestrator=parse_{format_metadata['format_name']}, "
        f"section_parser=parse_{format_metadata['format_name']}_{format_metadata['primary_section']}, "
        f"outer_dataclass={format_metadata['format_dataclass']}, "
        f"primary_section_dataclass={format_metadata['primary_section_dataclass']})"
    )

    # 5. Generate Python parser code
    generate_python(
        facts, format_metadata, section_emissions, field_byte_eq_emissions,
        struct_unpack_emissions, numeric_constraint_emissions,
        dataclass_emissions, source_path, py_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
