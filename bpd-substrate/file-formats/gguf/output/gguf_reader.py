# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""gguf_reader.py — generated from gguf.bpd by generate.py.

DO NOT EDIT BY HAND. Regenerate with `python generate.py`.
Source declaration: <repo>/must_close/boundary_dsl/file-formats/gguf/gguf.bpd
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path


HEADER_SIZE: int = 24
GGUF_MAGIC: bytes = bytes([0x47, 0x47, 0x55, 0x46])
FORMAT_VERSION_EXPECTED: int = 3
TENSOR_COUNT_MAX: int = 100000
METADATA_KV_COUNT_MAX: int = 100000
ARRAY_ELEMENT_COUNT_MAX: int = 10000000
TENSOR_NDIM_MAX: int = 8


class GGUFParseError(Exception):
    """Raised when a GGUF file fails to parse or violates a constraint."""


# ─── Generated helper functions (from BPD declarations) ─────────────
#
# Each helper validates one structural property declared in gguf.bpd.
# These are at module scope (not nested) so they can be unit-tested
# independently and so call sites do not need to know the helper's
# indentation context.
#
# Future: helpers will be refactored into methods on per-(format,
# section) classes (e.g., formats.gguf.header.check_length()).
# The current underscore-flat names migrate to the dotted form by replacing
# underscores with dots and moving bodies into class methods.

def check_section_length_gguf_header(raw: bytes, path) -> None:
    """Raise GGUFParseError if raw is shorter than HEADER_SIZE.

    Generated from BPD: section_size(header, byte_count(24)).
    """
    if len(raw) < HEADER_SIZE:
        raise GGUFParseError(
            f"file {path} too short for GGUF header: "
            f"got {len(raw)} bytes, need {HEADER_SIZE}"
        )


def check_field_magic_gguf_header(raw: bytes, path) -> bytes:
    """Extract magic bytes from raw and verify against GGUF_MAGIC.

    Generated from BPD: field(header, magic, byte_offset(0), ...,
                              [must_equal(byte_sequence(...))]).

    Returns:
        The extracted bytes (always equal to GGUF_MAGIC on success).

    Raises:
        GGUFParseError: if the extracted bytes do not equal GGUF_MAGIC.
    """
    actual = raw[0:4]
    if actual != GGUF_MAGIC:
        raise GGUFParseError(
            f"file {path} bad GGUF magic: "
            f"got {actual!r}, expected {GGUF_MAGIC!r}"
        )
    return actual


def check_field_version_must_equal_gguf_header(value: int, path) -> None:
    """Verify version satisfies must_equal(3).

    Generated from BPD: field(header, version, ..., [must_equal(3)]).

    Raises:
        GGUFParseError: if value != FORMAT_VERSION_EXPECTED.
    """
    if value != FORMAT_VERSION_EXPECTED:
        raise GGUFParseError(
            f"unsupported GGUF version: "
            f"got {value}, expected {FORMAT_VERSION_EXPECTED}"
        )


def check_field_tensor_count_must_be_lt_gguf_header(value: int, path) -> None:
    """Verify tensor_count satisfies must_be_lt(100000).

    Generated from BPD: field(header, tensor_count, ..., [must_be_lt(100000)]).

    Raises:
        GGUFParseError: if value >= TENSOR_COUNT_MAX.
    """
    if value >= TENSOR_COUNT_MAX:
        raise GGUFParseError(
            f"GGUF tensor_count bound violation: "
            f"got {value}, max {TENSOR_COUNT_MAX}"
        )


def check_field_metadata_kv_count_must_be_lt_gguf_header(value: int, path) -> None:
    """Verify metadata_kv_count satisfies must_be_lt(100000).

    Generated from BPD: field(header, metadata_kv_count, ..., [must_be_lt(100000)]).

    Raises:
        GGUFParseError: if value >= METADATA_KV_COUNT_MAX.
    """
    if value >= METADATA_KV_COUNT_MAX:
        raise GGUFParseError(
            f"GGUF metadata_kv_count bound violation: "
            f"got {value}, max {METADATA_KV_COUNT_MAX}"
        )


def check_field_element_count_must_be_lt_gguf_metadata_array(value: int, path) -> None:
    """Verify element_count satisfies must_be_lt(10000000).

    Generated from BPD: field(metadata_array, element_count, ..., [must_be_lt(10000000)]).

    Raises:
        GGUFParseError: if value >= ARRAY_ELEMENT_COUNT_MAX.
    """
    if value >= ARRAY_ELEMENT_COUNT_MAX:
        raise GGUFParseError(
            f"GGUF element_count bound violation: "
            f"got {value}, max {ARRAY_ELEMENT_COUNT_MAX}"
        )


def check_field_n_dimensions_must_be_lt_gguf_tensor_info_record(value: int, path) -> None:
    """Verify n_dimensions satisfies must_be_lt(8).

    Generated from BPD: field(tensor_info_record, n_dimensions, ..., [must_be_lt(8)]).

    Raises:
        GGUFParseError: if value >= TENSOR_NDIM_MAX.
    """
    if value >= TENSOR_NDIM_MAX:
        raise GGUFParseError(
            f"GGUF n_dimensions bound violation: "
            f"got {value}, max {TENSOR_NDIM_MAX}"
        )


# ─── Per-section dataclasses ────────────────────────────────────────


@dataclass(frozen=True)
class GGUFHeader:
    """Parsed GGUF header (first 24 bytes of a GGUF file)."""
    magic: bytes
    version: int
    tensor_count: int
    metadata_kv_count: int

    def is_valid_magic(self) -> bool:
        return self.magic == GGUF_MAGIC

    def is_supported_version(self) -> bool:
        return self.version == FORMAT_VERSION_EXPECTED


# ─── Outer parsed-format container ──────────────────────────────────
#
# Holds the parsed result of every format_section in declared order.
# For Stage 0 (header only), this is just the header. Future stages
# add metadata_kv_section, tensor_info_section, tensor_data_section
# as additional fields as those sections are implemented.


@dataclass(frozen=True)
class Gguf:
    """Parsed GGUF file — Stage 0 (primary section only).

    Future stages will add fields for additional format_sections as
    they are implemented (metadata_kv_section, tensor_info_section,
    tensor_data_section, etc.).
    """
    header: GGUFHeader


# ─── Per-section parsing functions ──────────────────────────────────
#
# Each format_section in gguf.bpd produces one parse_*
# function. The orchestrator (parse_gguf) calls them in
# declared order and assembles the Gguf result.
#
# Per-section function interface (β convention from this commit):
#   - Fixed-width sections take `raw: bytes` and `path` (path used for
#     error messages only). The orchestrator reads the section's bytes
#     and passes them in.
#   - Variable-width sections take `f: BinaryIO` and `path`, reading
#     from the current stream position. (None in GGUF Stage 0; this
#     comment documents the future shape for PNG and others.)


def parse_gguf_header(raw: bytes, path) -> GGUFHeader:
    """Parse the 24-byte GGUF header from raw bytes.

    Raises GGUFParseError on:
      - File too short
      - Magic bytes mismatch
      - Unsupported version
      - Counts exceeding safety bounds
    """
    p = Path(path)

    check_section_length_gguf_header(raw, p)

    magic = check_field_magic_gguf_header(raw, p)

    version, tensor_count, metadata_kv_count = struct.unpack(
        "<IQQ", raw[4:HEADER_SIZE]
    )

    check_field_version_must_equal_gguf_header(version, p)
    check_field_tensor_count_must_be_lt_gguf_header(tensor_count, p)
    check_field_metadata_kv_count_must_be_lt_gguf_header(metadata_kv_count, p)

    return GGUFHeader(
        magic=magic,
        version=version,
        tensor_count=tensor_count,
        metadata_kv_count=metadata_kv_count,
    )


# ─── Top-level orchestrator ─────────────────────────────────────────


def parse_gguf(path: str | Path) -> Gguf:
    """Read and parse a GGUF file.

    Calls per-section parse functions in declared order and assembles
    the result. For Stage 0 this is just the header.
    """
    p = Path(path)
    with p.open("rb") as f:
        raw_header = f.read(HEADER_SIZE)

    header = parse_gguf_header(raw_header, p)

    return Gguf(header=header)
