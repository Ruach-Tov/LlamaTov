# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""ggufq.py — jq-style query tool for GGUF files.

Stage 0: header-only queries.

Usage:
    ggufq FILE.gguf '.magic'
    ggufq FILE.gguf '.version'
    ggufq FILE.gguf '.tensor_count'
    ggufq FILE.gguf '.metadata_kv_count'
    ggufq FILE.gguf '.'              # whole header
    ggufq FILE.gguf 'keys'           # list available fields
    ggufq FILE.gguf --summary        # human-readable header summary
    ggufq FILE.gguf --json           # full header as JSON

Where the syntax overlaps with jq, we imitate jq exactly. Where it
must differ (because we operate on a typed GGUF object rather than
arbitrary JSON), the differences are explicit.

Stage 0 supported syntax:
    .                — identity (return whole object)
    .FIELD           — field access
    keys             — return list of field names

Stage 0 NOT supported (deferred to later stages):
    | (pipes)        — Stage 1+
    [] (iteration)   — Stage 2 (needs tensor list)
    select(...)      — Stage 2+
    string interpolation — Stage 4
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

# Add the output dir to path for the generated module
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "output"))

from gguf_reader import (  # noqa: E402
    GGUFHeader,
    GGUFParseError,
    parse_gguf,
)


def header_to_dict(h: GGUFHeader) -> dict:
    """Convert a GGUFHeader to a JSON-serializable dict."""
    return {
        "magic": h.magic.decode("ascii", errors="replace"),
        "version": h.version,
        "tensor_count": h.tensor_count,
        "metadata_kv_count": h.metadata_kv_count,
    }


def header_keys(h: GGUFHeader) -> list[str]:
    """List the field names of a GGUFHeader."""
    return [f.name for f in dataclasses.fields(h)]


# ─── jq-style query evaluator ────────────────────────────────────


class QueryError(Exception):
    """Raised on query syntax error or evaluation error."""


def evaluate_query(query: str, h: GGUFHeader) -> object:
    """Evaluate a jq-style query against a GGUF header.

    Stage 0 grammar:
        query := '.' | '.' IDENT | 'keys'
    """
    q = query.strip()

    if q == "":
        raise QueryError("empty query")

    if q == ".":
        return header_to_dict(h)

    if q == "keys":
        return header_keys(h)

    if q.startswith("."):
        field_name = q[1:]
        if not field_name:
            return header_to_dict(h)
        if not field_name.isidentifier():
            raise QueryError(
                f"invalid field name: {field_name!r}\n"
                "Stage 0 supports only simple field access; pipes, "
                "iteration, and filters arrive in later stages."
            )
        valid_fields = header_keys(h)
        if field_name not in valid_fields:
            raise QueryError(
                f"unknown field: {field_name!r}\n"
                f"available fields: {', '.join(valid_fields)}"
            )
        # Get the field value and JSON-ify it
        d = header_to_dict(h)
        return d[field_name]

    raise QueryError(
        f"unsupported query: {q!r}\n"
        "Stage 0 supports: '.', '.FIELD', and 'keys'."
    )


def render_value(value: object) -> str:
    """Render a query result for output."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2)
    if isinstance(value, str):
        # Bare strings: print without quotes for shell pipeline-friendliness,
        # matching jq's `-r` raw-output mode. For Stage 0 we always raw-output
        # strings since there's no ambiguity.
        return value
    return str(value)


# ─── Built-in summary printer ────────────────────────────────────


def render_summary(h: GGUFHeader) -> str:
    """Render a human-readable summary of the header."""
    magic_str = h.magic.decode("ascii", errors="replace")
    valid_marker = "✓" if h.is_valid_magic() else "✗"
    version_marker = "✓" if h.is_supported_version() else "✗"
    return (
        f"GGUF header summary\n"
        f"───────────────────\n"
        f"  magic            {magic_str!r:<10} {valid_marker}\n"
        f"  version          {h.version:<10} {version_marker}\n"
        f"  tensor_count     {h.tensor_count:>10,}\n"
        f"  metadata_kv_count{h.metadata_kv_count:>10,}\n"
    )


# ─── CLI ─────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="ggufq",
        description="jq-style query tool for GGUF files (Stage 0: header only)",
        epilog="Where syntax overlaps with jq, we imitate jq. Where it differs, we differ explicitly.",
    )
    parser.add_argument("file", help="GGUF file to query")
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help='jq-style query (e.g., ".tensor_count")',
    )
    parser.add_argument(
        "--summary",
        "-s",
        action="store_true",
        help="print human-readable header summary",
    )
    parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="print whole header as JSON (equivalent to query '.')",
    )

    args = parser.parse_args()

    try:
        header = parse_gguf(args.file).header
    except GGUFParseError as e:
        print(f"ggufq: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError:
        print(f"ggufq: file not found: {args.file}", file=sys.stderr)
        return 2

    if args.summary:
        print(render_summary(header), end="")
        return 0

    if args.json or args.query == ".":
        print(json.dumps(header_to_dict(header), indent=2))
        return 0

    if args.query is None:
        # Default behavior: print summary
        print(render_summary(header), end="")
        return 0

    # Otherwise, evaluate the query
    try:
        result = evaluate_query(args.query, header)
    except QueryError as e:
        print(f"ggufq: {e}", file=sys.stderr)
        return 3

    print(render_value(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
