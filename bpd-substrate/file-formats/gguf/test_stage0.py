# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
import os as _os2
"""test_stage0.py — tests for Stage 0 GGUF header parser and ggufq tool.

Verifies:
  - The generated parser correctly reads valid v3 GGUF files
  - The version=3 constraint correctly rejects v2 files
  - Non-GGUF files are rejected with bad-magic error
  - File-too-short is rejected
  - The ggufq query tool returns expected outputs
  - Conformance with mavchin's hand-written reference parser

Run with: cd .. && python -m pytest test_stage0.py -v
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

# Make the generated parser importable
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "output"))

from gguf_reader import (  # noqa: E402
    GGUF_MAGIC,
    GGUFHeader,
    GGUFParseError,
    Gguf,
    HEADER_SIZE,
    FORMAT_VERSION_EXPECTED,
    parse_gguf,
    parse_gguf_header,
)


# ─── Test fixtures: locating real GGUF files ─────────────────────


@pytest.fixture(scope="module")
def ollama_blobs_dir() -> Path:
    """The Ollama blobs directory; skip tests if not present."""
    p = Path.home() / ".ollama" / "models" / "blobs"
    if not p.exists():
        pytest.skip(f"Ollama blobs not found at {p}")
    return p


def _find_gguf_by_version(blobs_dir: Path, version: int) -> Path | None:
    """Find the first GGUF file with the given version in blobs_dir."""
    for blob in sorted(blobs_dir.iterdir()):
        if not blob.is_file():
            continue
        try:
            with blob.open("rb") as f:
                header = f.read(8)
            if len(header) < 8:
                continue
            if header[:4] != GGUF_MAGIC:
                continue
            ver = struct.unpack("<I", header[4:8])[0]
            if ver == version:
                return blob
        except (OSError, struct.error):
            continue
    return None


# The SHIPPING artifact: a real v3 GGUF that always sits on disk. Point conformance at it so the v3
# tests RUN against what we actually ship, instead of skipping when Ollama blobs are absent.
PRODUCTION_GGUF = Path(_os2.environ.get("LLAMATOV_MODEL", "models/qwen_q8.gguf"))


def _ollama_blobs_dir_optional() -> Path | None:
    p = Path.home() / ".ollama" / "models" / "blobs"
    return p if p.exists() else None


@pytest.fixture(scope="module")
def gguf_v3_file() -> Path:
    """A real v3 GGUF file. Prefer Ollama's blob store; else fall back to the PRODUCTION model (the
    shipping artifact). Only skips if neither is available."""
    blobs = _ollama_blobs_dir_optional()
    if blobs is not None:
        p = _find_gguf_by_version(blobs, version=3)
        if p is not None:
            return p
    if PRODUCTION_GGUF.exists():
        return PRODUCTION_GGUF
    pytest.skip("no v3 GGUF: neither Ollama blobs nor the production model present")


@pytest.fixture(scope="module")
def gguf_v2_file(ollama_blobs_dir: Path) -> Path:
    """A real v2 GGUF file from Ollama's blob store."""
    p = _find_gguf_by_version(ollama_blobs_dir, version=2)
    if p is None:
        pytest.skip("no v2 GGUF file in Ollama blobs")
    return p


@pytest.fixture
def synthetic_truncated_file(tmp_path: Path) -> Path:
    """A file that has the right magic but is too short for a header."""
    p = tmp_path / "truncated.gguf"
    p.write_bytes(GGUF_MAGIC + b"\x03\x00\x00\x00")  # 8 bytes total
    return p


@pytest.fixture
def synthetic_bad_magic_file(tmp_path: Path) -> Path:
    """A file that has wrong magic bytes."""
    p = tmp_path / "wrong.gguf"
    p.write_bytes(b"NOPE" + b"\x03\x00\x00\x00" + b"\x00" * 16)
    return p


@pytest.fixture
def synthetic_v3_file(tmp_path: Path) -> Path:
    """A synthetic v3 GGUF file with known values for value-equality tests."""
    p = tmp_path / "synthetic.gguf"
    p.write_bytes(
        GGUF_MAGIC
        + struct.pack("<I", 3)         # version
        + struct.pack("<Q", 42)         # tensor_count
        + struct.pack("<Q", 7)          # metadata_kv_count
    )
    return p


# ─── Parser tests ────────────────────────────────────────────────


class TestParser:
    """Direct tests of the generated parse_gguf function."""

    def test_constants_match_bpd_declaration(self):
        """The constants in the generated module match the BPD declaration."""
        assert GGUF_MAGIC == b"GGUF"
        assert FORMAT_VERSION_EXPECTED == 3
        assert HEADER_SIZE == 24

    def test_synthetic_v3_parses_correctly(self, synthetic_v3_file: Path):
        """A well-formed synthetic v3 file parses to expected values."""
        header = parse_gguf(synthetic_v3_file).header
        assert isinstance(header, GGUFHeader)
        assert header.magic == b"GGUF"
        assert header.version == 3
        assert header.tensor_count == 42
        assert header.metadata_kv_count == 7
        assert header.is_valid_magic()
        assert header.is_supported_version()

    def test_real_v3_parses_without_error(self, gguf_v3_file: Path):
        """A real v3 GGUF file from Ollama parses without raising."""
        header = parse_gguf(gguf_v3_file).header
        assert header.magic == b"GGUF"
        assert header.version == 3
        assert header.tensor_count > 0
        assert header.metadata_kv_count > 0

    def test_v2_file_rejected(self, gguf_v2_file: Path):
        """A real v2 GGUF file is rejected with version error."""
        with pytest.raises(GGUFParseError) as exc_info:
            parse_gguf(gguf_v2_file)
        assert "version" in str(exc_info.value).lower()
        assert "2" in str(exc_info.value)

    def test_bad_magic_rejected(self, synthetic_bad_magic_file: Path):
        """A file with wrong magic is rejected with magic error."""
        with pytest.raises(GGUFParseError) as exc_info:
            parse_gguf(synthetic_bad_magic_file)
        assert "magic" in str(exc_info.value).lower()

    def test_truncated_file_rejected(self, synthetic_truncated_file: Path):
        """A file too short for a header is rejected."""
        with pytest.raises(GGUFParseError) as exc_info:
            parse_gguf(synthetic_truncated_file)
        assert "short" in str(exc_info.value).lower() or "header" in str(exc_info.value).lower()

    def test_nonexistent_file_raises(self, tmp_path: Path):
        """A nonexistent file raises FileNotFoundError, not GGUFParseError."""
        with pytest.raises(FileNotFoundError):
            parse_gguf(tmp_path / "does_not_exist.gguf")


# ─── ggufq CLI tests ─────────────────────────────────────────────


def _run_ggufq(args: list[str]) -> tuple[int, str, str]:
    """Run ggufq.py with given args; return (returncode, stdout, stderr)."""
    cmd = [sys.executable, str(HERE / "ggufq.py"), *args]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    return result.returncode, result.stdout, result.stderr


class TestGgufqCLI:
    """Tests of the ggufq command-line tool."""

    def test_summary_on_v3_file(self, gguf_v3_file: Path):
        """--summary produces expected output structure."""
        rc, stdout, stderr = _run_ggufq([str(gguf_v3_file), "--summary"])
        assert rc == 0
        assert "GGUF header summary" in stdout
        assert "GGUF" in stdout
        assert "version" in stdout
        assert "tensor_count" in stdout
        assert "metadata_kv_count" in stdout

    def test_default_action_is_summary(self, gguf_v3_file: Path):
        """ggufq FILE with no args/query prints summary."""
        rc, stdout, stderr = _run_ggufq([str(gguf_v3_file)])
        assert rc == 0
        assert "GGUF header summary" in stdout

    def test_query_tensor_count_returns_integer(self, gguf_v3_file: Path):
        """`.tensor_count` returns a parseable integer."""
        rc, stdout, stderr = _run_ggufq([str(gguf_v3_file), ".tensor_count"])
        assert rc == 0
        value = int(stdout.strip())
        assert value > 0

    def test_query_version_returns_3(self, gguf_v3_file: Path):
        """`.version` on a v3 file returns 3."""
        rc, stdout, stderr = _run_ggufq([str(gguf_v3_file), ".version"])
        assert rc == 0
        assert stdout.strip() == "3"

    def test_query_magic_returns_gguf(self, gguf_v3_file: Path):
        """`.magic` returns 'GGUF' (raw, no quotes)."""
        rc, stdout, stderr = _run_ggufq([str(gguf_v3_file), ".magic"])
        assert rc == 0
        assert stdout.strip() == "GGUF"

    def test_identity_query_returns_json(self, gguf_v3_file: Path):
        """`.` returns parseable JSON with all four fields."""
        rc, stdout, stderr = _run_ggufq([str(gguf_v3_file), "."])
        assert rc == 0
        data = json.loads(stdout)
        assert set(data.keys()) == {"magic", "version", "tensor_count", "metadata_kv_count"}
        assert data["magic"] == "GGUF"
        assert data["version"] == 3

    def test_keys_returns_field_names(self, gguf_v3_file: Path):
        """`keys` returns a JSON list of field names."""
        rc, stdout, stderr = _run_ggufq([str(gguf_v3_file), "keys"])
        assert rc == 0
        data = json.loads(stdout)
        assert isinstance(data, list)
        assert "magic" in data
        assert "version" in data
        assert "tensor_count" in data
        assert "metadata_kv_count" in data

    def test_unknown_field_error(self, gguf_v3_file: Path):
        """`.bogus_field` errors with helpful message."""
        rc, stdout, stderr = _run_ggufq([str(gguf_v3_file), ".bogus_field"])
        assert rc == 3
        assert "unknown field" in stderr.lower()
        assert "available fields" in stderr.lower()

    def test_v2_file_error_exit_code(self, gguf_v2_file: Path):
        """v2 GGUF file produces parse-error exit code 2."""
        rc, stdout, stderr = _run_ggufq([str(gguf_v2_file), "--summary"])
        assert rc == 2
        assert "version" in stderr.lower()

    def test_nonexistent_file_error(self, tmp_path: Path):
        """Nonexistent file produces exit code 2 with clear message."""
        rc, stdout, stderr = _run_ggufq([str(tmp_path / "no.gguf"), "--summary"])
        assert rc == 2
        assert "not found" in stderr.lower()


# ─── Conformance with mavchin's hand-written parser ─────────────


class TestConformance:
    """The generated parser must produce values consistent with mavchin's
    hand-written reference parser at papers/kan-acceleration/gguf_parser.py.

    This is the polyarchitecture conformance pattern from mcp_bridge.bnd
    (memory 22a7ccc2) applied at level 0: different implementations of the
    same BPD declaration must agree on field values.
    """

    @pytest.fixture(scope="class")
    def reference_parser(self):
        """Import mavchin's hand-written parser as the conformance reference. Resolved relative to the
        repo root (not a hardcoded <home> path), and skips gracefully if absent OR unreadable
        (PermissionError on a multi-user enclave) — a missing reference is a skip, not an error."""
        repo_root = Path(__file__).resolve().parents[3]
        ref_path = repo_root / "papers" / "kan-acceleration"
        try:
            present = (ref_path / "gguf_parser.py").exists()
        except (PermissionError, OSError):
            present = False
        if not present:
            pytest.skip("reference parser not found/readable at " + str(ref_path))
        sys.path.insert(0, str(ref_path))
        try:
            import gguf_parser  # noqa
            return gguf_parser
        finally:
            sys.path.pop(0)

    def test_version_agrees(self, gguf_v3_file: Path, reference_parser):
        """Generated parser's version field matches reference parser's."""
        ours = parse_gguf(gguf_v3_file).header
        theirs = reference_parser.GGUFFile.from_path(str(gguf_v3_file))
        assert ours.version == theirs.version, (
            f"version mismatch: ours={ours.version}, theirs={theirs.version}"
        )

    def test_tensor_count_agrees(self, gguf_v3_file: Path, reference_parser):
        """Generated parser's tensor_count matches reference parser's."""
        ours = parse_gguf(gguf_v3_file).header
        theirs = reference_parser.GGUFFile.from_path(str(gguf_v3_file))
        assert ours.tensor_count == theirs.tensor_count, (
            f"tensor_count mismatch: ours={ours.tensor_count}, "
            f"theirs={theirs.tensor_count}"
        )

    def test_metadata_count_agrees(self, gguf_v3_file: Path, reference_parser):
        """Generated parser's metadata_kv_count matches the reference parser's
        actual metadata count after full parsing.

        Note: mavchin's parser doesn't store metadata_kv_count as a field
        (it parses the metadata section entirely), so we compare against
        len(metadata).
        """
        ours = parse_gguf(gguf_v3_file).header
        theirs = reference_parser.GGUFFile.from_path(str(gguf_v3_file))
        assert ours.metadata_kv_count == len(theirs.metadata), (
            f"metadata count mismatch: header says {ours.metadata_kv_count}, "
            f"reference parser found {len(theirs.metadata)} metadata entries"
        )
